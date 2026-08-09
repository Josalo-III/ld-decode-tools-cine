/******************************************************************************
 * comblocked.cpp
 * ld-chroma-decoder — Colourisation filter for ld-decode
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * SPDX-FileCopyrightText: 2018 Chad Page
 * SPDX-FileCopyrightText: 2018-2019 Simon Inns
 * SPDX-FileCopyrightText: 2020-2021 Adam Sampson
 * SPDX-FileCopyrightText: 2021 Phillip Blucas
 * SPDX-FileCopyrightText: 2025-2026 Joseph Burns
 *
 * This file is part of ld-decode-tools.
 * A subset of comb functions used when --ntsc-phase-comp is active
 ******************************************************************************/

#include "comb.h"
#include "combmath.h"
#include "feasibleband.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <QtGlobal>

namespace {

inline double smoothGate01(double t)
{
    t = std::clamp(t, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Mean over an even effective sample weight, but with an integer centroid.
// Support is effectiveWidth+1 samples: half weight at center +/- half and
// full weight between.  The endpoints are the same carrier phase, so their
// two halves preserve the complete-cycle population of the old even window.
// Edge replication keeps the coordinate fixed instead of sliding the aperture
// away from the requested sample.
inline double centeredEvenWeightMean(const double *values,
                                     const double *prefix,
                                     int width,
                                     int center,
                                     int effectiveWidth)
{
    if (!values || !prefix || width <= 0 || effectiveWidth <= 0 ||
        (effectiveWidth & 1))
        return 0.0;

    const int half = effectiveWidth / 2;
    const int lo = center - half;
    const int hi = center + half;
    double sum = 0.0;

    if (lo >= 0 && hi < width) {
        sum = 0.5 * (values[lo] + values[hi]) +
              (prefix[hi] - prefix[lo + 1]);
    } else {
        for (int k = -half; k <= half; ++k) {
            const int x = std::clamp(center + k, 0, width - 1);
            const double w = (k == -half || k == half) ? 0.5 : 1.0;
            sum += w * values[x];
        }
    }

    return sum / static_cast<double>(effectiveWidth);
}

// produceY coarse-platform selector (witness isolation knob).
//
// Default decomposes raw on the cheap, raster-aligned four-sample coarse. The
// witness unlocks a centered, lurch-sharpened sliding-boxcar coarse as its LF
// platform. There is exactly one platform in either mode: comb supplies the
// middle band and provisional top, while the HF election may replace only that
// top band.
//
// LD_COARSE_SHARP = L (float, witness-only isolation knob):
//   unset       : witness-native lurch level (1.0).
//   <=0         : disable the witness lurch for an A/B.
//   > 0         : scale the lurch snap gate (<1 gentler, >1 snaps weaker
//                 steps too).
// Structural carrier-amplitude ceiling, measured (2026-07-28) instead of
// guessed. The measured range ledger:
//   - GGV-1069 colour bars: 36 IRE max (SMPTE f6000 and full-height
//     f22000 agree) -- the BROADCAST-LEGAL maximum.
//   - Most saturated camera-scene material (Gilgol bikini zone): 32 IRE
//     (raw envelope 31.7, conservation-exact channel 32.2).
//   - Optical-composite EFFECTS run far hotter: the tractor beam measures
//     72 IRE on the conservation-exact (D-S)/2 channel -- real carrier,
//     no estimation. A bars-derived 38 IRE ceiling visibly aliased the
//     beam (and the pre-existing 50 IRE bound was already clipping its
//     top 0.8% of samples).
// A hull may encode impossibles ONLY, so the GLOBAL bound must clear the
// hottest real carrier this material contains: 75 IRE nominal, burst-
// scaled. One global limit cannot discriminate a range this wide (legal
// carrier spans ~1 IRE on gray struts to 72 IRE on the beam); regional
// discrimination belongs to a pooled exact-channel ceiling, not to this
// constant. carrierScale is the burst correlation magnitude (~10 IRE at
// spec burst).
inline double maxCarrierAmpIREFromScale(double carrierScale)
{
    constexpr double kCarrierMaxPerBurstScale = 7.5; // 75 IRE at spec burst
    constexpr double kCarrierMaxFloorIRE      = 75.0;
    return std::max(kCarrierMaxFloorIRE,
                    carrierScale * kCarrierMaxPerBurstScale);
}

// 4fSC sampling, in MHz. The lurch solve's phase chains step four samples, so
// a chain's angular frequency is w = 8*pi*f/kSampleRateMHz.
inline constexpr double kSubcarrierMHz = 3.579545;
inline constexpr double kSampleRateMHz = 4.0 * kSubcarrierMHz;

// The coarse platform's LF AUTHORITY: the frequency below which the platform,
// rather than the same-phase difference facts, decides the platform.
//
// This replaces LD_COARSE_SHARP, which scaled the plateau substitution that
// solveLurchYCurve() retired. A level made sense for a snap (apply more or
// less of a substitution); it is meaningless against a solve, and blending a
// solved curve back toward the blurry registration to honour one would be a
// smoother in a new costume. The honest knob for the same duty is where the
// platform's authority ends.
//
// Default is the encoder's own chroma bandlimit, and that is a derived value
// rather than a starting guess, because of how the phase chains alias.
//
// A chain samples Y every four samples, so a component at frequency f advances
// 8*pi*f/fs per chain step: f and fSC +- f land on the SAME chain frequency
// (at f = fSC exactly, w = 2*pi = 0, which is why fSC content aliases to chain
// DC and neither term here can see it -- the platform is silent about Y at the
// subcarrier by construction, and must stay silent rather than invent).
//
// So "the platform owns chain frequencies below w_c" means it owns
//
//     [0, f_c]   AND   [fSC - f_c, fSC + f_c]
//
// Setting f_c to the encoder's chroma bandlimit therefore hands the platform
// exactly the LEGAL CARRIER BAND -- where the platform, being a four-sample mean
// and so carrier-free by construction, is precisely the right authority -- and
// hands the difference facts the bands where legal carrier cannot live and a
// same-phase raw difference is therefore pure luma. The two authorities land on
// the two content classes with no overlap. Raising f_c hands the boxcar bands
// it cannot see (blurrier); lowering it hands the facts the carrier band, where
// their residual is legal envelope motion misread as luma.
inline double lurchPlatformCutoffMHz()
{
    static const double mhz = []{
        const char *s = std::getenv("LDCD_LURCH_PLATFORM_MHZ");
        if (!s || !*s) return 1.3;
        const double v = std::atof(s);
        return (std::isfinite(v) && v > 0.0) ? v : 1.3;
    }();
    return mhz;
}

// A/B escape: 0 restores the plateau snap at both platform call sites.
inline bool lurchSolveEnabled()
{
    static const bool on = []{
        const char *s = std::getenv("LDCD_LURCH_SOLVE");
        return !(s && std::atoi(s) == 0);
    }();
    return on;
}

// Certified Dirichlet rows in the solve. DEFAULT OFF -- MEASURED UNLAWFUL.
//
// The idea is right and the shape is not. A covered frame certifies ONE PARITY,
// so pinning per sample gives even lines exact Y and leaves odd lines solved,
// inside one frame: per-line substitution of exact values into a shared
// estimate path, which is the consumption shape the parity law already
// falsified twice.
//
// Measured, s1x11 @2795, 9 output frames, vertical-Nyquist projection (a
// second-difference roughness metric UNDERSTATED this at +2.7% -- it cannot
// separate "sharper" from "alternating", so it was the wrong instrument):
//
//   covered frames   pin ON   2-line alternation  +8.90%   HF luma -1.45%
//                    pin OFF                      -0.58%           +0.44%
//   uncovered        (no facts; identical either way)  -2.6%       +1.05%
//
// The whole covered-frame regression is the pinning. Note this is NOT a
// pin-versus-skip question: on a fully certified line the two are the same
// operation, so skipping the solve there fails identically.
//
// The lawful shape is the one the doctrine already names -- certified luma is a
// phase-free comb reference, so the UNCOVERED parity's lines should receive the
// facts vertically from their two certified neighbours, which reaches both
// parities together. Kept behind this escape for that work; it is not a tuning
// option in the meantime.
//
// Also the truth-path switch for any referee grading this platform. A hold-out
// here is NOT sufficient on its own: buildApertureMeans() already subtracts the
// exact carrier when forming the pool, so the platform carries truth on covered
// lines whatever this is set to.
inline bool lurchPinEnabled()
{
    static const bool on = []{
        const char *s = std::getenv("LDCD_LURCH_PIN");
        return (s && std::atoi(s) != 0);
    }();
    return on;
}

// Retained record from the removed LDCD_PROBE_KNEE census
//
// Transfer-knee stats. Measurement only. Comb inherits
// 1D's bandwidth limits: fine-detail AMPLITUDE is stripped from 1D and its
// descendants -- a transfer-curve divergence at the top of the scale, HF
// only. This measured that curve directly: at pixels where the carrier-band
// energy is PROVEN not-carrier (carrierIllegalProof high), the retracted
// top is trustworthy luma amplitude; bin |comb top| against |retracted top|
// and the bin where the ratio departs from unity IS the knee -- the
// measured point at which comb's HF stops being trusted, which the coming
// roll-off keys to. Sign-flip fraction per bin rode along for the
// grammar-side sign fix (the taps own the sign).

// Retained record from the removed LDCD_PROBE_DSREF census
//
// (D-S)/2 referee (LDCD_PROBE_DSREF=1): grades every carrier estimator
// against the assembler's exact-carrier side channel on covered lines.
// The confiscation ledger: error vs exact truth, split flat / detail
// (hLumaDelta >= 6 IRE), per frame. Thread-safe use: run -t 1.
//
// [flat=0 / detail=1][estimator: 0=1D 1=comb 2=fit 3=retracted]
// NOTE: in the default (anchor) retracted mode the retr column is
// SELF-REFERENTIAL on covered samples (retracted == exact there);
// grade that mode by saturation-restricted fSC-in-Y instead.
//
// Referee repair (2026-08-02): the certified head and
// factFit stamp locked1DSource / carrierFit with the
// exact fact at precisely the samples this referee
// grades, so the raw planes read |e|=0 by construction.
// Grade the PRE-fact estimator stashes instead.

// Retained record from the removed LDCD_PROBE_CCREF census
//
// Cross-colour return referee (LDCD_PROBE_CCREF=1): grades the CC return
// against the exact-carrier side channel on covered samples.
//
// On a covered sample the ideal return fraction is not an opinion, it is
// arithmetic. With ex the certified carrier and C = raw - combY the comb's
// carrier estimate,
//
//     leak = C - ex          the luma the comb confiscated (signed IRE)
//     m*   = leak / C        the return fraction that lands on truth
//
// leak IS cross-colour, measured rather than inferred. The return publishes
// combY + m*C, and the truth is combY + leak, so its Y error is
// |m*C - leak| against the comb's own |leak|. Every covered sample therefore
// carries a VERDICT -- helped, harmed, or inert -- not a correlation.
//
// Ratios of per-sample quantities are never averaged: a bin's ideal mask is
// the properly weighted S(C*leak)/S(C*C), and its leak fraction is the
// pooled S|leak|/S|C|. Both are well conditioned where a mean of leak/C
// would be all tail.
//
// Run with -t 1 (unsynchronised accumulators). Run with LDCD_ANCHOR_1D=0 for
// the calibration read: in the default anchored mode the comb already
// consumes the retraction ladder on covered frames, so leak ~ 0 there by
// construction and the ledger measures nothing that transfers to uncovered
// lines. Default-on is still worth a run of its own -- there it answers
// whether the return correctly ABSTAINS on covered lines, or is damaging
// them today. The detector is honest in both modes: its edge read is fed by
// the true locked1DSource, which the anchoring never touches.
//
// Detector components, each graded separately against the measured leak.
// Which of these predicts the leak, and which is the noise, is the whole
// question -- a mask driven by an uninformative read is the noisiest
// election participant by construction.
// RDev (2026-08-02) is the LIVE committed-verdict read: the anchored
// delta devIRE mapped to [0,1] as devIRE / (2*cutoff), so bin center
// 0.5 sits at the shipping cutoff and the per-bin ideal-m curve IS the
// cutoff sweep. The gA/edge/lurch columns grade retired reads, kept for
// contrast.
//
// Fixed-scalar sweep: what a constant return fraction would have scored.
// If the adaptive mask cannot beat the best constant, the adaptation is
// not earning its variance.
//
// AMPLITUDE CONDITIONING -- without this the whole instrument lies.
//
// C and leak share the capture's noise. If the comb's luma were PERFECT,
// C = c_true + n_d and leak = (n_d + n_s)/2, so
//     S(C*leak)/S(C*C) -> sigma^2/2 / sigma^2 = 0.5
// in the noise floor with ZERO cross-colour present. An ideal mask near
// 0.5 at low carrier is therefore the signature of noise, not of leak,
// and "returning" it would only push the capture's own noise into Y --
// Lhat has half the noise of raw, so following raw there is a downgrade
// the L1 ledger cannot see.
//
// Only bins whose ideal mask stands well clear of 0.5 at real carrier
// amplitude are evidence of confiscated luma. Hence: an amplitude census,
// and a second copy of every read table restricted to real carrier.
//
// STAR LAW test (user, 2026-07-30): a black-white-black transient inside
// 4 px cannot be legal carrier -- the encoder bandlimits chroma to
// 1.3 MHz, so a legal envelope cannot rise and fall in under ~5.5
// samples at 4fSC. At such a site the lawful subtrahend is ZERO and
// composite IS the luma. Prediction, testable on covered lines: at
// clean star sites ex ~ 0 and m* ~ 1; at the same shape WITH ringing
// (the bail branch) truth shows carrier and m* < 1. This grades the
// composite-domain test itself, so once certified it can run on EVERY
// frame, covered or not.
// Skirt (±3..±5) true carrier: grades the refined bail — "peak takes
// raw, skirt goes to 1D-as-chroma". If skirt |ex| is at the noise
// floor, 1D's chroma reading of the skirt is wrong but bail-safe; if
// skirt |ex| is real carrier, the bail branch is doing its job.
//
// Vertical-context census: how many of the four frame-line neighbours
// (l±1, l±2, same column ±1) also carry the signature. A point star is
// vertically compact (vertRun 0-1); a strut passes the horizontal test
// on every line (vertRun 3-4). If |ex| rises with vertRun, struts carry
// real carrier that the substitution would wrongly republish into Y.
//
// ---- MODEL-vs-TRUTH EXPLANATION LEDGER ----
// Not "how wrong" (the DSREF ledger already says) but WHERE and WHY:
// which site classes carry each estimator's error mass, and how much of
// the model's error at real carrier is amplitude (a per-window gain
// explains it) versus shape/phase (no scalar can). Site classes are
// assigned by priority star > legal > detail > flat, so each covered
// sample lands in exactly one.
//
// Gain-explainability: non-overlapping 16-sample windows on covered
// lines, evaluated only where the window holds real carrier
// (mean ex^2 >= 4 IRE^2). Per window the best scalar gain g minimizes
// sum(est - g*ex)^2; the residual after g is the SHAPE/PHASE part of the
// error — the part no calibration could ever fix (the 857f457 law,
// now with a number attached).
//
// RESIDUAL STRUCTURE. Magnitude is not the artifact; structure is. A
// return that is wrong on 16% of samples is noise if those samples are
// scattered and an artifact class if they are organised, and the L1
// ledger above cannot tell the two apart.
//
// What truth can and cannot see: certified coverage is ONE PARITY of
// every other frame, so within a covered frame the covered lines are the
// frame lines of a single parity, pitch 2. Truth can therefore measure
// vertical structure at FIELD pitch (l-2, l, l+2, all covered) but can
// never see frame-pitch (l-1, l, l+1) structure in the residual -- the
// other parity has no truth to be wrong against. The classic lineAlt
// artifact lives at frame pitch, so it is measured here on the MASK
// instead, which exists on every line: if the mask's frame-pitch D2
// exceeds its field-pitch D2, the return's ACTION alternates by parity
// even though its error cannot be graded there. Nearer neighbours should
// disagree less; when they disagree more, the disagreement is parity.
//
// 4. The mask's parity geometry, on ALL lines. Frame pitch (+/-1)
// against field pitch (+/-2): nearer neighbours should disagree LESS.
// A CONTROL rides along: |carrier| through the identical stencils.
// For any vertically smooth quantity D2 ~ k^2, so the +/-1 stencil
// should read about a QUARTER of the +/-2 one -- ratio ~0.25. The
// control establishes what this material's own vertical detail does
// to that expectation, so the mask's ratio is graded against a
// measurement rather than against theory.
//
// 5. Vertical coherence at ±2 (adjacent covered lines), real carrier
// only. Truth's own coherence is the ceiling (doc: 0.907); the fit's
// coherence against it says whether the model's error is per-line
// noise (low r) or a coherent, structured misread (r near truth's,
// but offset).
//
// Amplitude census. mIdeal ~ 0.5 in the lowest bin is the noise
// pedestal, not evidence of leak; read the high bins.
//
// The same reads with the noise pedestal excluded. These are the
// only rows that can convict a detector.
//
// Retired notch edge read, rebuilt for CONTRAST from the
// PRE-head estimator stash (referee repair 2026-08-02:
// locked1DSource IS the exact fact on covered samples, so
// the old rebuild here was grading certified luma under a
// detector label).
//
// Star-signature classification, from the COMPOSITE'S OWN
// SHAPE (no estimator input): white peak over agreeing
// dark flanks, down by ±2 (b-w-b inside 4 px), classified
// clean vs ringing by the ±3..±5 skirt. Thresholds are
// what the actuator would use; truth grades them here.

// Retained record from the removed LDCD_PROBE_SYNC census
//
// SYNC-TONE stability probe: the premise test for the user's
// segment-tracked certified phase reference. On each covered frame, pool
// the def lines' working-space certified IQ (the 4fsc demod caches ARE
// certified there under stage 1) per region; report the amplitude-weighted
// phase delta against the PREVIOUS covered frame -- the step a running
// tracker would have to ride. If the deltas are small and slowly varying,
// the sync tone is real and a past-only tracker is a lawful
// construction-time reference for every frame; if they are content-noise,
// the premise fails here and we say so.
//
// The census carried an anticipating tracker in validation form: a
// per-region alpha-beta loop on (phase, rate) that PREDICTED the region
// phase from state before consuming the measurement, so the reported miss
// was genuine held-out prediction error -- the number a consumption design
// would live on. A frame-wide median miss above the cut threshold reset
// state (scene cut: content hue changed wholesale; a reference must
// re-seed, not average across).

// Retained record from the removed LDCD_PROBE_OFFGRID census
//
// Off-grid leakage stats. Measurement only. Stage 1
// of the fit reset: a carrier waveform belongs to the span of the grammar
// basis over every legal 4-sample window (the two quadrature waveforms span
// the whole fSC subspace; off-span = DC + 2fSC content, which no lawful
// carrier may carry). The per-sample scalar surgeries between the fit's
// basis exit and publication push the emitted carrier out of that span, so
// raw - fit carries OFF-GRID alternations the election cannot compare with
// comb's on-grid residue. This measured the off-span energy fraction per
// published carrier source, binned by window amplitude; the differential
// against comb under the identical operator is the honest read (lawful
// envelope motion leaks a little in any 4-sample window for every source).
//
// Phase half: position, not span. A wrong-phase carrier is perfectly
// in-span; what damages the election is the fit's carrier sitting off
// POSITION -- rotated against the physical scalar and jittering window
// to window. Measured at strong windows only.
//
// Dropout half: strong-window stats exclude exactly the failure the beam
// sheet showed -- windows where the fit's amplitude COLLAPSES while comb
// still carries the chroma. Split by the legality proof: at proven-
// illegal energy the "dropout" is the fit correctly refusing what comb
// wrongly models (virtuous); at legal carrier it is lost lock (the
// defect the reset must cure).
//
// Referee repair (2026-08-02): on covered frames the certified head and
// factFit make the "fit" and "1D" sources the exact carrier, so "off-grid
// leakage" and "fit-vs-comb dropout" would be truth-vs-comb measurements
// wearing estimator labels (the pre-repair "95.7% lock" calibration note is
// void for the same reason). The census asked about estimator behaviour, so
// it skipped covered frames outright.

} // namespace

// Locked-path pre-processing: burst detection, carrier grammar, and luma cache.
void Comb::FrameBuffer::phaseLocked()
{
    if (!configuration.phaseCompensation)
        return;

    const int left       = videoParameters.activeVideoStart;
    const int right      = videoParameters.activeVideoEnd;
    const int firstLine  = videoParameters.firstActiveFrameLine;
    const int lastLine   = videoParameters.lastActiveFrameLine;
    const int fullWidth  = videoParameters.fieldWidth;
    const int width      = right - left;

    if (left >= right || firstLine >= lastLine)
        return;

    const int requiredLines = lastLine + 1;
    if ((int)carrierGrammar.size() < requiredLines)
        carrierGrammar.resize(requiredLines);
    // Basis coefficients — computed once, used by all passes
    double Ce = 1.0, Se = 0.0;
    basisCoeffs(Ce, Se);
    for (int i = 0; i < 4; ++i) {
        double sp, cp;
        shiftedBasis(i, Ce, Se, sp, cp);
        spLUT_locked[i] = sp;
        cpLUT_locked[i] = cp;
    }
    basisLockedInit = true;

    const double K   = 0.5 * M_PI;
    const double Rb0 = -K * CAL_EPS_SAMPLES + (CAL_LO_ROT_DEG * M_PI / 180.0);
    const double cRb = std::cos(Rb0);
    const double sRb = std::sin(Rb0);

    const bool   floorEnable = configuration.burstFloorEnable;
    const double floorFactor = configuration.burstFloorFactor;
    const auto  &T           = configuration.tunables;
    constexpr double MIN_PHASE_CONFIDENCE = 1e-6;

    // --- Pass 1: burst detection -> carrier grammar ---
    for (int line = firstLine; line < lastLine; ++line) {
        const quint16 *rawLine = rawbuffer.data() + line * fullWidth;
        auto burst = detectBurst(rawLine, videoParameters, floorEnable, floorFactor);
        double bcos = burst.bcos, bsin = burst.bsin;
        const double bc2 = bcos * cRb - bsin * sRb;
        const double bs2 = bcos * sRb + bsin * cRb;
        CombCarrierGrammar &grammar = carrierGrammar[line];
        grammar.burstCos = bc2;
        grammar.burstSin = bs2;
        grammar.carrierScale = burst.carrierScale * invIreScale;
        grammar.phaseConfidence =
            std::clamp((grammar.carrierScale - 3.0) / 7.0, 0.0, 1.0);
        grammar.grammarLocked = grammar.phaseConfidence > MIN_PHASE_CONFIDENCE;
        grammar.phaseError = 0.0;
        grammar.phaseScheduleConflict = 0.0;
        grammar.affine.valid = false;

        double lutTi[4], lutTq[4];
        fusedDemodLUT(bc2, bs2, spLUT_locked, cpLUT_locked, lutTi, lutTq);
        for (int i = 0; i < 4; ++i) {
            grammar.demodLUTTi[i] = (float)lutTi[i];
            grammar.demodLUTTq[i] = (float)lutTq[i];
        }
    }

    if (!lockedLumaBaseY4_flat.empty() &&
        !lockedLumaSmooth_flat.empty() &&
        demodWidth == width)
    {
        // Sharpened boxcar coarse platform for produceY. This is the
        // lurch-corrected LF authority and it lives ONLY in the --luma-witness
        // fork: default reconstructs on lockedLumaBaseY4, so the sharp buffer
        // is left unallocated (comb.cpp) and the build below is skipped. Under
        // witness it is built when the sweep knob asks for it. Same
        // construction the constrained witness used: a
        // sliding 4-sample boxcar (carrier-cancelled per aperture, evaluated
        // every sample) lurch-sharpened so a confirmed luma step lands at one
        // column instead of smearing across four. The gate is scaled by the
        // sweep level.
        const bool buildSharp =
            configuration.lumaWitness && !lockedLumaSharp_flat.empty();

        for (int line = firstLine; line < lastLine; ++line) {
            const quint16 *rawLine = rawbuffer.data() + line * fullWidth;
            buildCompositeLumaDecompositionLine(rawLine, left, width,
                                                lockedLumaBaseY4_line(line),
                                                nullptr,
                                                lockedLumaSmooth_line(line));

            // Vertical-contrast service: the 1D is the first stage to cross a
            // vertical contrast, so the lateral coarse delta is registered here
            // once for every later client (Frame B reach exemption,
            // hLumaDeltaIRE, cross-color, FVF vertical regime).
            if (float *hDelta = lockedLumaHDeltaIRE_line(line)) {
                const double *smooth = lockedLumaSmooth_line(line);
                for (int rel = 0; rel < width; ++rel) {
                    const int rm = std::max(0, rel - 2);
                    const int rp = std::min(width - 1, rel + 2);
                    hDelta[rel] = static_cast<float>(
                        std::fabs(smooth[rp] - smooth[rm]) * invIreScale);
                }
            }

            // The coarse-residual aperture-mean pool is built once for every
            // path by buildApertureMeans() (from split1D, before this runs);
            // read it here for the sharpened boxcar below.
            double *apMean = lockedApertureMean_line(line);

            if (!buildSharp)
                continue;

            double *sharp = lockedLumaSharp_line(line);
            if (width < 4) {
                std::copy(lockedLumaSmooth_line(line),
                          lockedLumaSmooth_line(line) + width, sharp);
                continue;
            }
            // Derived FROM the pool above, not a private rebuild.
            const double *boxcar = apMean;

            if (lurchSolveEnabled()) {
                // The solve owns the registration too -- same construction,
                // then the difference facts move it.
                solveLurchYCurve(line, boxcar, width - 3, width, sharp);
            } else {
                const int lastStart = width - 4; // last legal aperture start

                // Register the even four-sample means at integer xi by
                // averaging the two half-sample apertures on either side.
                // Their combination is the phase-balanced five-sample support
                // (0.5,1,1,1,0.5)/4 centred exactly at xi.
                for (int xi = 0; xi < width; ++xi) {
                    const int s0 = std::clamp(xi - 2, 0, lastStart);
                    const int s1 = std::clamp(xi - 1, 0, lastStart);
                    sharp[xi] = 0.5 * (boxcar[s0] + boxcar[s1]);
                }
                // Canonical runs (built once in split1D), edges vertically
                // corroborated (median-of-three) so the snap stops sawing
                // bright vertical contours; apply-only here.
                const std::vector<LurchStepRun> corrRuns =
                    corroborateLurchEdges(line);
                applyLurchSteps(corrRuns, boxcar, width - 3,
                                width, 1.0, sharp, nullptr);
            }

            // Retained record from the removed LDCD_LURCH_FEASIBLE census
            //
            // FEASIBLE-LUMA RESTRAINT (was LDCD_LURCH_FEASIBLE=1, default
            // OFF; never promoted, so the shipping sharpener has never been
            // restrained by it).
            //
            // Lurch earns its place because the four-sample coarse smears
            // boundaries and lurch un-smears them. So the restriction cannot
            // be "sharpen less"; it has to be "never sharpen to a value that
            // cannot be true". feasibleband.h's pair-sum law supplies exactly
            // that, and for LUMA rather than carrier: the carrier is
            // antisymmetric over +-2 samples, so
            //     composite[i] + composite[i+-2] = Y[i] + Y[i+-2]
            // is carrier-free and EXACT. With luma bounded, each available
            // neighbour pins Y[i] to an interval. A sharpened platform
            // outside that interval asserts a luma the composite forbids.
            //
            // This is the primitive's first consumer. It forbids impossibles
            // and says nothing about which surviving value is preferred --
            // no threshold, nothing fitted, no per-scene constant.
            //
            // The census separated two cases, because only the first is about
            // lurch: BASE-IN means the unsharpened platform was feasible and
            // the sharpening pushed it out (lurch overshoot); BASE-OUT means
            // the platform was already infeasible before lurch touched it,
            // which is a different defect and not this restraint's business.
            // Only BASE-IN samples were ever clamped back to the interval.
        }
        lockedLumaCacheValid = true;
    }

    return;
}

// Demodulate the blind 1D bandpass through the locked carrier grammar, publish
// the common-4fsc IQ used by the 2D candidate builders, and remodulate the
// phase-corrected scalar source used by locked 1D/2D/3D selection.
//
// This version adds a cheap local anti cross color pass:
//
//   - first pass:  demodulate the phase-locked 1D scalar into local I/Q
//   - second pass: preserve local I/Q DC, damp local AC/winding when it looks
//                  like edge-generated or dubbed cross-color
//   - publish:     write demod buffers, locked 4fsc buffers, magnitude, and
//                  lockedSource from the cleaned I/Q

void Comb::FrameBuffer::buildCarrierAnalysis(FrameBuffer *prevFrame)
{
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;
    const int fullWidth = videoParameters.fieldWidth;

    static const double parallaxRepairTolIRE = []{
        const char *s = std::getenv("LD_1D_PARALLAX_TOL_IRE");
        return s ? std::atof(s) : 0.5;
    }();

    if (!configuration.phaseCompensation || width <= 0 || first >= last ||
        demodWidth != width || demodLines < last)
        return;

    carrierRetractionModelValid = false;

    const size_t count = static_cast<size_t>(demodLines) * demodWidth;
    if (carrierAnalysis_flat.size() < count) {
        carrierAnalysis_flat.assign(count, lddecode::CarrierAnalysisRecord{});
    }

    constexpr int kNarrowWin = 16;
    constexpr int kWideWin = 32;
    static const double cosRef[4] = { 1.0, 0.0, -1.0, 0.0 };
    static const double sinRef[4] = { 0.0, 1.0,  0.0, -1.0 };

    std::vector<double> preI(width + 1, 0.0);
    std::vector<double> preQ(width + 1, 0.0);

    for (int line = first; line < last; ++line) {
        const quint16 *rawLine =
            rawbuffer.data() + static_cast<size_t>(line) * fullWidth;
        double *baseline = locked1DRawBandpass_line(line);
        if (!baseline)
            continue;

        auto rawAtRel = [&](int rel) -> double {
            const int r = std::clamp(rel, 0, width - 1);
            return static_cast<double>(rawLine[left + r]);
        };

        // Canonical full-resolution source authority used by every later
        // carrier client. Keep this formula and its edge convention in one
        // place so analysis and rendering cannot drift apart.
        //
        // The ±2 bandpass taps use half-sample edge REFLECTION, matching the
        // bucket split1D convention ([-1]->[0], [-2]->[1] at the left; mirror
        // at the right). Edge-clamp (repeating the boundary sample) injects a
        // DC pedestal at the wrong carrier phase, so the aperture goes
        // asymmetric at the frame boundary and the bandpass fires on that
        // false step — a spurious carrier fringe along the picture edge that
        // also pollutes the interline schedule fingerprint out there.
        // Reflection continues the waveform instead. (Only the bandpass taps
        // reflect; the coarse-residual/luma-floor reads below keep rawAtRel's
        // clamp — those four-sample means never wanted reflection.)
        auto rawMirror = [&](int rel) -> double {
            if (rel < 0)              rel = -rel - 1;
            else if (rel >= width)    rel = 2 * width - 1 - rel;
            rel = std::clamp(rel, 0, width - 1);   // safety for tiny widths
            return static_cast<double>(rawLine[left + rel]);
        };
        for (int rel = 0; rel < width; ++rel) {
            const double c  = static_cast<double>(rawLine[left + rel]);
            const double m2 = rawMirror(rel - 2);
            const double p2 = rawMirror(rel + 2);
            baseline[rel] = 0.50 * c - 0.25 * (m2 + p2);
        }

        seedCombAttributionPerLine(line);
        lddecode::CarrierAnalysisRecord *analysis = carrierAnalysis_line(line);
        if (analysis) {
            std::fill(
                analysis,
                analysis + width,
                lddecode::CarrierAnalysisRecord{});
        }

        if (!analysis)
            continue;

        preI[0] = 0.0;
        preQ[0] = 0.0;
        for (int rel = 0; rel < width; ++rel) {
            const int p = carrierSampleClass(line, left + rel) & 3;
            preI[rel + 1] = preI[rel] + baseline[rel] * cosRef[p];
            preQ[rel + 1] = preQ[rel] + baseline[rel] * sinRef[p];
        }

        const double tolSamples =
            std::max(0.0, parallaxRepairTolIRE) * irescale;
        for (int rel = 0; rel < width; ++rel) {
            const int p = carrierSampleClass(line, left + rel) & 3;
            const int na = std::clamp(rel - kNarrowWin / 2, 0, width);
            const int nb = std::clamp(na + kNarrowWin, 0, width);
            const double nn = static_cast<double>(std::max(1, nb - na));
            const double ZnI = (preI[nb] - preI[na]) / nn;
            const double ZnQ = (preQ[nb] - preQ[na]) / nn;
            const double shortSample =
                2.0 * (ZnI * cosRef[p] + ZnQ * sinRef[p]);

            const int wa = std::clamp(rel - kWideWin / 2, 0, width);
            const int wb = std::clamp(wa + kWideWin, 0, width);
            const double wn = static_cast<double>(std::max(1, wb - wa));
            const double ZwI = (preI[wb] - preI[wa]) / wn;
            const double ZwQ = (preQ[wb] - preQ[wa]) / wn;
            const double wideSample =
                2.0 * (ZwI * cosRef[p] + ZwQ * sinRef[p]);

            lddecode::CarrierResidualOption options[4];
            int optionCount = 0;
            const int sFirst = std::max(0, rel - 3);
            const int sLast = width >= 4 ? std::min(rel, width - 4) : -1;
            for (int s = sFirst; s <= sLast && optionCount < 4; ++s) {
                const double y4 = 0.25 * (
                    rawAtRel(s + 0) + rawAtRel(s + 1) +
                    rawAtRel(s + 2) + rawAtRel(s + 3));
                options[optionCount].sample = rawAtRel(rel) - y4;
                options[optionCount].membershipDeltaIRE =
                    s + 4 < width
                        ? 0.25 * (rawAtRel(s + 4) - rawAtRel(s)) * invIreScale
                        : 0.0;
                ++optionCount;
            }

            const double movingMean = 0.25 * (
                rawAtRel(rel - 1) + rawAtRel(rel) +
                rawAtRel(rel + 1) + rawAtRel(rel + 2));
            const double movingResidual = rawAtRel(rel) - movingMean;
            const double sourceSample = baseline[rel];

            auto &record = analysis[rel];
            record.fit.sourceSample = static_cast<float>(sourceSample);
            record.fit.shortSample = static_cast<float>(shortSample);
            record.fit.wideSample = static_cast<float>(wideSample);
            record.fit.sourceMinusShortIRE = static_cast<float>(
                (sourceSample - shortSample) * invIreScale);
            record.fit.shortMinusWideIRE = static_cast<float>(
                (shortSample - wideSample) * invIreScale);
            record.fit.sourceMinusWideIRE = static_cast<float>(
                (sourceSample - wideSample) * invIreScale);
            record.fit.valid = true;
            record.residual = lddecode::analyzeCarrierResidualOptions(
                options,
                optionCount,
                shortSample,
                tolSamples,
                std::max(0.0, parallaxRepairTolIRE),
                movingResidual,
                invIreScale);
        }
    }

    // ------------------------------------------------------------------
    // Schedule-conformance registration (grammar-as-table).
    //
    // With every line's canonical bandpass harvested, register each
    // pixel's carrier-band energy against the schedule: legal carrier
    // MUST invert across Opposite-relation partners — the same-field ±2
    // lines within this frame, and the same line on the neighbouring
    // frame (the comb's own temporal structure). Energy that MATCHES where the
    // schedule demands inversion admits no legal carrier interpretation
    // and is registered ScheduleIllegal — luma by law — at entry, rather
    // than entering the carrier column with a bad grade for every
    // consumer to re-adjudicate.  Same-relation partners are
    // non-discriminative (legal and alien both match) and abstain.
    // 4-sample windows keep the correlation phase-flat; an energy floor
    // keeps noise Unresolved.  Conservative tie-break: any legal vote
    // wins — real chroma is never claimed as luma.
    if (width >= 4) {
        const bool prevUsable =
            prevFrame &&
            prevFrame->demodWidth == demodWidth &&
            prevFrame->demodLines == demodLines &&
            !prevFrame->locked1DRawBandpass_flat.empty();

        const double rmsFloor = 3.0 * irescale;           // 3 IRE RMS
        const double energyFloor = 4.0 * rmsFloor * rmsFloor;
        constexpr double kCorrVote = 0.5;

        for (int line = first; line < last; ++line) {
            lddecode::CarrierAnalysisRecord *analysis =
                carrierAnalysis_line(line);
            const double *bp0 = locked1DRawBandpass_line(line);
            const CombCarrierGrammar *g0 = carrierGrammarLine(line);
            if (!analysis || !bp0 || !g0 || !g0->grammarLocked)
                continue;

            // Opposite-relation partner rows, certified by the grammar.
            const double *axes[3];
            int nAxes = 0;
            auto addAxis = [&](const double *bpP,
                               const CombCarrierGrammar *gP) {
                if (!bpP || !gP || !gP->grammarLocked || nAxes >= 3)
                    return;
                const auto rel = lddecode::carrierGrammarSignedPhaseRelation(
                    g0, left, gP, left);
                if (rel == lddecode::CarrierPhaseRelation::Opposite)
                    axes[nAxes++] = bpP;
            };
            if (line - 2 >= first)
                addAxis(locked1DRawBandpass_line(line - 2),
                        carrierGrammarLine(line - 2));
            if (line + 2 < last)
                addAxis(locked1DRawBandpass_line(line + 2),
                        carrierGrammarLine(line + 2));
            if (prevUsable)
                addAxis(prevFrame->locked1DRawBandpass_line(line),
                        prevFrame->carrierGrammarLine(line));
            if (nAxes == 0) {
                continue;
            }

            for (int rel = 0; rel < width; ++rel) {
                // Centered 5-tap, half-weight ends (integer centroid): the
                // old [rel, rel+3] window registered every conformance
                // verdict 1.5 samples right of the sample it judged.
                double e0 = 0.0;
                for (int k = -2; k <= 2; ++k) {
                    const int j = std::clamp(rel + k, 0, width - 1);
                    const double w = (k == -2 || k == 2) ? 0.5 : 1.0;
                    e0 += w * bp0[j] * bp0[j];
                }
                if (e0 < energyFloor) {
                    continue;
                }

                bool legalVote = false;
                bool illegalVote = false;
                int legalAxisVotes = 0, illegalAxisVotes = 0, usableAxes = 0;
                double minCorr =  1e300;   // most-legal (negative) axis
                double maxCorr = -1e300;   // most-illegal (positive) axis
                for (int a = 0; a < nAxes; ++a) {
                    const double *bpP = axes[a];
                    double dot = 0.0;
                    double eP = 0.0;
                    for (int k = -2; k <= 2; ++k) {
                        const int j = std::clamp(rel + k, 0, width - 1);
                        const double w = (k == -2 || k == 2) ? 0.5 : 1.0;
                        dot += w * bp0[j] * bpP[j];
                        eP  += w * bpP[j] * bpP[j];
                    }
                    if (eP < energyFloor)
                        continue;
                    ++usableAxes;
                    const double corr = dot / std::sqrt(e0 * eP);
                    minCorr = std::min(minCorr, corr);
                    maxCorr = std::max(maxCorr, corr);
                    if (corr <= -kCorrVote) {
                        legalVote = true;
                        ++legalAxisVotes;
                    } else if (corr >= kCorrVote) {
                        illegalVote = true;
                        ++illegalAxisVotes;
                    }
                }

                // Scanner layer (grammar-as-table): publish the graded
                // MEASUREMENT, not a decision.  carrierConformance is the
                // relation-signed correlation, biased to the most-legal axis
                // when one inverts (real chroma is never disowned on the
                // strength of a matching neighbour) and otherwise reporting
                // the most-illegal evidence. conformanceSupportFraction records
                // the fraction of axes SUPPORTING that selected sign, not the
                // number merely available.  The old usableAxes/3 assigned a
                // lone coincidental legal vote full authority when all three
                // axes were present. Thresholding conformance at
                // -/+kCorrVote reproduces the legacy enum; action remains a
                // downstream policy.
                if (usableAxes > 0) {
                    const double conformance = legalVote ? minCorr : maxCorr;
                    const int supportingVotes = legalVote
                        ? legalAxisVotes : illegalAxisVotes;
                    // Axes decisively voting AGAINST the selected sign.
                    // Abstaining axes appear in neither fraction.
                    const int contradictingVotes = legalVote
                        ? illegalAxisVotes : legalAxisVotes;
                    analysis[rel].carrierConformance =
                        static_cast<float>(std::clamp(conformance, -1.0, 1.0));
                    analysis[rel].conformanceUsableAxisFraction =
                        static_cast<float>(std::min(1.0,
                            static_cast<double>(usableAxes) / 3.0));
                    analysis[rel].conformanceSupportFraction =
                        static_cast<float>(std::min(1.0,
                            static_cast<double>(supportingVotes) / 3.0));
                    analysis[rel].conformanceContradictionFraction =
                        static_cast<float>(std::min(1.0,
                            static_cast<double>(contradictingVotes) / 3.0));
                }

                analysis[rel].scheduleConformance = legalVote
                    ? lddecode::CarrierScheduleConformance::LegalCarrier
                    : illegalVote
                        ? lddecode::CarrierScheduleConformance::ScheduleIllegal
                        : lddecode::CarrierScheduleConformance::Unresolved;
            }
        }
    }

    // Retained record from the removed LDCD_DUMP_FP dump
    //
    // Its subject was the three orthogonal views of "luma entered the
    // bandpass", per pixel along a scanline:
    //   incoh = sourceMinusWideIRE  (horizontal: source minus wide coherent fit)
    //   lurch = maxAbsMembershipIRE (carrier-free luma movement through aperture)
    //   conf  = carrierConformance  (interline: -1 inverts like carrier,
    //                                +1 matches where schedule demands inversion
    //                                = image-locked luma leak)
}

// ---------------------------------------------------------------------------
// Corner-leak corrector.
//
// The locked bandpass bp[x] = 0.50*c - 0.25*(m2 + p2) has response sin^2(w) and
// cancels a CONSTANT luma foundation exactly (-0.25 + 0.50 - 0.25 = 0).  The
// carrier is stacked on luma, so where the foundation bends the cancellation
// fails by exactly the curvature:
//
//     leak[x] = -0.25 * (Y[x-2] - 2*Y[x] + Y[x+2])        (an identity)
//
// Consequences this stage relies on:
//   * a constant-slope ramp has ZERO curvature and therefore leaks NOTHING --
//     gradients and shading are invisible, and this stage is inert there by
//     construction rather than by a gate;
//   * only CORNERS leak, so a ramped edge is two curvature events;
//   * the alternation seen at edges arises at demod (the LUT flips sign per the
//     schedule), not in the raw leak, so one correction serves every line.
//
// Recovery.  The complementary notch is exactly
//     notch[x] = raw[x] - bp[x] = 0.25*Y[x-2] + 0.5*Y[x] + 0.25*Y[x+2],
// i.e. Y through a KNOWN 3-tap stride-2 smoother S (response cos^2(w)).  With
// m = D2{notch} we have m = S{kappa} for kappa = D2{Y}, so the curvature is
// recovered by deconvolving S.  Van Cittert (kappa += m - S{kappa}) propagates
// error as (I - S) = sin^2(w): it converges everywhere EXCEPT at fSC, where it
// is frozen at the initial guess.  Starting from kappa = 0 therefore makes NO
// claim about the one mode that is genuinely unknowable on a single line -- the
// regulariser is the physics, and there is no tuning constant in it.
//
// Gating.  Three carrier-free tests, each used for what it can actually do:
//   * lurch  -- PRESENCE of luma motion across a cycle (it cannot localise a
//               corner: full-cycle smear, one corner per cycle);
//   * schedule -- energy matching where the schedule demands inversion is luma
//               BY LAW; legal carrier is protected;
//   * parallax -- from the collected aperture-mean pool.  A legal carrier nulls
//               in EVERY legal four-sample window, so it contributes no spread
//               between the apertures covering a sample; only luma moves them.
//               The ratio spread/|carrier| is literally "what fraction of this
//               carrier-band energy fails to null in the aperture".
// All gates multiply KAPPA (a luma-domain quantity), never the carrier, so they
// cannot manufacture sidebands.
//
// DIAGNOSTIC ONLY: the published leak has no consumer yet, so the render is
// unchanged.  When it is adopted, chroma = bp - leak and Y = raw - chroma, so
// the leak returns to luma and Y + chroma == raw exactly.
// ---------------------------------------------------------------------------

// Fill the sliding four-sample aperture-mean pool for every active line.
//
// apMean[v] = mean(raw[left+v .. left+v+3]), indexed by aperture START, so the
// legal apertures covering sample x are v in {x-3, x-2, x-1, x}. Published
// unfiltered: no sharpening, no gate, no absolute value -- consumers own the
// decisions. A legal carrier sums to zero over ANY legal four-sample window, so
// each mean is that window's LUMA mean exactly; the divergence between the
// apertures covering one sample is therefore pure luma with the carrier removed
// exactly (the coarse-residual parallax), and each mean also bounds the carrier
// at every sample it covers (the feasibility hull). Running sum, O(1)/sample.
//
// Built from split1D so it exists on EVERY path (bucket and locked) without
// burst-lock rotation; phaseLocked and buildCornerLeak read the same pool.
void Comb::FrameBuffer::buildApertureMeans()
{
    if (lockedApertureMean_flat.empty()) return;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int fullWidth = videoParameters.fieldWidth;
    const int width     = right - left;
    if (left >= right || firstLine >= lastLine) return;

    const int lastStart = width - 4;
    for (int line = firstLine; line < lastLine; ++line) {
        double *apMean = lockedApertureMean_line(line);
        if (!apMean) continue;
        const quint16 *rawLine = rawbuffer.data() + size_t(line) * fullWidth;
        // dG-certified luma (user direction, 2026-07-28: "when [lurch] is
        // finding a luma edge, we have some fields where that's not in
        // doubt"). Each mean's contract is "that window's LUMA mean
        // exactly" -- true for stationary carrier, broken by the partial-
        // window leak at edges. On twin-covered lines raw - exact IS the
        // luma, so subtracting the exact carrier honours the contract
        // exactly where it used to fail: lurch edges, the coarse-residual
        // hull, and the witness coarse all read certified luma there.
        const float *exRow = exactCarrierRow(line);
        auto src = [&](int xi) -> double {
            const double r = (double)rawLine[left + xi];
            if (exRow) {
                const float e = exRow[left + xi];
                if (std::isfinite(e)) return r - (double)e;
            }
            return r;
        };
        if (width >= 4) {
            double sum4 = src(0) + src(1) + src(2) + src(3);
            for (int xi = 0; xi <= lastStart; ++xi) {
                apMean[xi] = 0.25 * sum4;
                if (xi < lastStart)
                    sum4 += src(xi + 4) - src(xi);
            }
            // Tail: no legal aperture starts here. Hold the last real mean so
            // the buffer stays readable; consumers needing "a real aperture
            // started here" must respect xi <= width-4.
            for (int xi = lastStart + 1; xi < width; ++xi)
                apMean[xi] = apMean[lastStart];
        } else {
            double avg = 0.0;
            for (int xi = 0; xi < width; ++xi)
                avg += (double)rawLine[left + xi];
            avg /= std::max(1, width);
            for (int xi = 0; xi < width; ++xi) apMean[xi] = avg;
        }
    }
}

// Clamp a carrier row into the coarse-residual feasible range, in place.
// carrierAtLeft[x] is the carrier estimate at sample left+x, x in [0,width).
// The excess leaves the carrier; the caller decides where it lands (the bucket
// path lets luma = raw - chroma absorb it downstream, the locked path folds it
// back into the leak). RESTRICTS only -- a real carrier over real luma already
// lies inside [floor, ceiling], so only impossible carrier is moved. Rotation-
// free and O(width): the feasibility hull is a default-path client.
void Comb::FrameBuffer::applyCarrierFeasibilityHull(int line,
                                                    double *carrierAtLeft)
{
    if (!carrierAtLeft) return;
    const double *apMean = lockedApertureMean_line(line);
    if (!apMean) return;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int fullWidth = videoParameters.fieldWidth;
    const int width     = right - left;
    if (width <= 0) return;
    const quint16 *rawLine = rawbuffer.data() + size_t(line) * fullWidth;
    for (int x = 0; x < width; ++x) {
        const auto rng = lddecode::carrierFeasibleRange(
            (double)rawLine[left + x], apMean, x, width);
        carrierAtLeft[x] = std::clamp(carrierAtLeft[x], rng.floor, rng.ceiling);
    }
}

// Coarse-residual parallax for one line. For each sample, demodulate the four
// covering aperture residuals (raw - apMean) on the grammar phase, envelope-
// smooth each, then take the spread across the four over their mean magnitude.
// A legal carrier is aperture-invariant (nulls in every legal four-window), so
// the four views agree and the ratio is low; moving luma shifts them apart and
// the ratio is high. Because it reads four-sample windows, it resolves compact
// colour that a wide coherent window would average away.
void Comb::FrameBuffer::apertureParallaxLine(
        int line,
        std::vector<double> &vI, std::vector<double> &vQ,
        std::vector<double> &sI, std::vector<double> &sQ,
        double *ratioOut) const
{
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int fullWidth = videoParameters.fieldWidth;
    const int width     = right - left;
    if (width <= 0 || !ratioOut) return;
    for (int x = 0; x < width; ++x) ratioOut[x] = 1.0;   // unknown => act
    const double *apMean = lockedApertureMean_line(line);
    if (!apMean) return;
    const quint16 *rawLine = rawbuffer.data() + size_t(line) * fullWidth;
    const int lastStart = width - 4;
    for (int k = 0; k < 4; ++k) {
        for (int x = 0; x < width; ++x) {
            const int v = std::clamp(x - (3 - k), 0, std::max(0, lastStart));
            const double r = (double)rawLine[left + x] - apMean[v];
            const int ph = carrierSampleClass(line, left + x);
            vI[size_t(k) * width + x] = 2.0 * r * sin4fsc(ph);
            vQ[size_t(k) * width + x] = 2.0 * r * cos4fsc(ph);
        }
    }
    // 3-sample smooth per view (envelope extraction after demod).
    for (int k = 0; k < 4; ++k) {
        const double *pi = &vI[size_t(k) * width];
        const double *pq = &vQ[size_t(k) * width];
        double *oi = &sI[size_t(k) * width];
        double *oq = &sQ[size_t(k) * width];
        for (int x = 0; x < width; ++x) {
            const int a = std::max(0, x - 1), b = std::min(width - 1, x + 1);
            oi[x] = (pi[a] + pi[x] + pi[b]) / 3.0;
            oq[x] = (pq[a] + pq[x] + pq[b]) / 3.0;
        }
    }
    for (int x = 0; x < width; ++x) {
        double mi = 0.0, mq = 0.0;
        for (int k = 0; k < 4; ++k) {
            mi += sI[size_t(k) * width + x];
            mq += sQ[size_t(k) * width + x];
        }
        mi *= 0.25; mq *= 0.25;
        double div = 0.0;
        for (int k = 0; k < 4; ++k) {
            const double di = sI[size_t(k) * width + x] - mi;
            const double dq = sQ[size_t(k) * width + x] - mq;
            div += std::hypot(di, dq);
        }
        div *= 0.25;
        const double mag = std::hypot(mi, mq);
        ratioOut[x] = div / std::max(mag, 1e-6);
    }
}

// RECOVERY PROFILE depth n, not a convergence tolerance. The Van Cittert
// iteration kappa += (m - S{kappa}) propagates error as (I - S)^n = sin^2n(w),
// so after n rounds the recovered fraction at frequency w is 1 - sin^2n(w):
// zero AT fSC always (the mode is never claimed -- the fSC null is a
// single-line law), approaching 1 elsewhere. n therefore chooses HOW MUCH of
// the near-fSC neighbourhood is claimed, and the worst-case noise gain of the
// implied inverse is bounded by n itself (G_n <= n) -- there is no tuning
// constant hiding here, only a claimed-bandwidth/noise trade. Measured: the
// strong-edge saturation DIPS then RECOVERS as n rises (a PARTIAL doublet
// subtraction leaves a residue that cancels chroma; a converged one restores
// it).
static constexpr int    kCornerRecoveryDepth  = 60;
// Outer rounds of the two-way contamination fix (see buildCornerLeak).
static constexpr int    kCornerOuterRounds    = 2;
// Parallax ratio: below soft the energy nulls in every aperture like legal
// carrier (protect); above hard it fails to null (luma, act). Measured
// populations: colour p50 0.05-0.12, pure luma p50 0.89.
static constexpr double kCornerParallaxSoft   = 0.15;
static constexpr double kCornerParallaxHard   = 0.45;

// TEMPORARY INSTRUMENT (LDCD_PROBE_VC=1): grades the Van Cittert D2Y estimate
// against certified luma on covered def lines. Strip once the question closes.
namespace {
// AMPLITUDE-CONDITIONED (iceberg plan Section 6, "Referee noise floor"):
// any agreement measure between two noisy quantities converges to a fixed
// positive value with no signal present, and three instruments in this tree
// have already been caught that way. Binned by the stranded peak's own
// bandpass amplitude |C| in IRE, so a usable regime can be SEEN rather than
// diluted into a single pooled number.
constexpr int kVcBins = 5;
const double kVcEdge[kVcBins] = { 2.0, 5.0, 10.0, 20.0, 1e9 };
const char  *kVcName[kVcBins] = { "|C|<2", "2-5", "5-10", "10-20", ">20" };

struct LdcdVcGrade {
    long   n[kVcBins]      = {0};
    double sTrue[kVcBins]  = {0}, sKap[kVcBins] = {0}, sRes[kVcBins] = {0};
    double sTrueF[kVcBins] = {0}, sKapF[kVcBins] = {0}, sResF[kVcBins] = {0};
    double sXc[kVcBins]    = {0};
};
thread_local LdcdVcGrade gVc;

void ldcdVcGrade(double t, double k, double r,
                 double tF, double kF, double rF, double inv, double ampIRE)
{
    LdcdVcGrade &g = gVc;
    int b = 0;
    while (b < kVcBins - 1 && ampIRE >= kVcEdge[b]) ++b;
    t *= inv; k *= inv; r *= inv; tF *= inv; kF *= inv; rF *= inv;
    g.n[b]++;
    g.sTrue[b] += t * t; g.sKap[b] += k * k; g.sRes[b] += r * r;
    g.sXc[b] += t * k;
    g.sTrueF[b] += tF; g.sKapF[b] += kF; g.sResF[b] += rF;
}

void ldcdVcGradeReport(int tag)
{
    LdcdVcGrade &g = gVc;
    long tot = 0;
    for (int b = 0; b < kVcBins; ++b) tot += g.n[b];
    if (tot < 1000) return;
    std::fprintf(stderr, "[VC] frame %d  (kappa vs certified D2{Lhat}, "
                         "binned by stranded-peak |C|)\n", tag);
    std::fprintf(stderr, "      %-8s %9s %8s %8s %9s %7s %9s %9s\n",
                 "bin", "n", "truth", "kappa", "resid/tru", "corr",
                 "kapF/truF", "resF/truF");
    for (int b = 0; b < kVcBins; ++b) {
        if (g.n[b] < 200) continue;
        const double n = (double)g.n[b];
        const double rt = std::sqrt(g.sTrue[b] / n),
                     rk = std::sqrt(g.sKap[b] / n),
                     rr = std::sqrt(g.sRes[b] / n);
        std::fprintf(stderr,
            "      %-8s %9ld %8.3f %8.3f %8.0f%% %7.3f %8.0f%% %8.0f%%\n",
            kVcName[b], g.n[b], rt, rk, 100.0 * rr / std::max(1e-9, rt),
            g.sXc[b] / n / std::max(1e-12, rt * rk),
            100.0 * g.sKapF[b] / std::max(1e-9, g.sTrueF[b]),
            100.0 * g.sResF[b] / std::max(1e-9, g.sTrueF[b]));
    }
    g = LdcdVcGrade();
}
} // namespace

void Comb::FrameBuffer::buildCornerLeak()
{
    static const bool probeVanCittert = []{
        const char *s = std::getenv("LDCD_PROBE_VC");
        return s && std::atoi(s) != 0;
    }();
    // No consumer yet, so the default path must not pay for it: ~11% of a
    // locked decode (the 60 Van Cittert sweeps dominate). Enable explicitly
    // with LDCD_CORNER_LEAK=1 while developing. Remove this gate when the
    // corrected source is adopted -- a stage with a real client is not
    // optional, and "on by default when it earns its place" is the rule.
    static const bool enabled = []{
        const char *s = std::getenv("LDCD_CORNER_LEAK");
        return s && std::atoi(s) != 0;
    }();
    if (!enabled)                         return;
    if (!configuration.phaseCompensation) return;
    if (lockedCornerLeak_flat.empty())    return;

    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;
    const int fullWidth = videoParameters.fieldWidth;
    if (width < 8 || first >= last) return;

    auto ramp = [](double v, double a, double b) {
        return std::clamp((v - a) / (b - a), 0.0, 1.0);
    };

    std::vector<double> notch(width), notchAdj(width),
                        mObs(width), kappa(width), sKappa(width);
    std::vector<double> ratio(width), gate(width), gateSmooth(width);
    std::vector<double> envI(width), envQ(width), sEnvI(width), sEnvQ(width),
                        envExcess(width);
    // Four aperture views of the residual, demodulated.
    std::vector<double> vI(size_t(4) * width), vQ(size_t(4) * width);
    std::vector<double> sI(size_t(4) * width), sQ(size_t(4) * width);

    // S: the notch kernel, [0.25, 0.5, 0.25] at stride 2.
    auto applyS = [&](const std::vector<double> &in, std::vector<double> &out) {
        for (int i = 0; i < width; ++i) {
            if (i >= 2 && i < width - 2)
                out[i] = 0.25 * in[i - 2] + 0.5 * in[i] + 0.25 * in[i + 2];
            else
                out[i] = in[i];
        }
    };

    for (int line = first; line < last; ++line) {
        double *leakRow = lockedCornerLeak_line(line);
        if (!leakRow) continue;
        // Clear unconditionally: a stale leak from an earlier frame must never
        // survive into this one (FrameBuffers persist across batches).
        std::fill(leakRow, leakRow + width, 0.0);
        const double *bpLine = locked1DRawBandpass_line(line);
        if (!bpLine) continue;

        const quint16 *rawLine =
            rawbuffer.data() + static_cast<size_t>(line) * fullWidth;
        const lddecode::CarrierAnalysisRecord *analysisRow =
            carrierAnalysis_line(line);
        const double *apMean = lockedApertureMean_line(line);

        for (int x = 0; x < width; ++x)
            notch[x] = (double)rawLine[left + x] - bpLine[x];

        // ---- Parallax: does this carrier-band energy null in every aperture?
        // Shared with the cross-color detector's carrier-legality veto.
        apertureParallaxLine(line, vI, vQ, sI, sQ, ratio.data());

        // ---- Gate: parallax alone. Multiplies KAPPA (a luma-domain quantity),
        // never the carrier, so it cannot manufacture sidebands.
        //
        // A lurch-presence gate and a schedule gate were both measured here and
        // both REMOVED, on evidence:
        //   * lurch changed nothing once parallax was present (identical to
        //     0.2% on every metric) -- it asks a cruder version of the same
        //     question;
        //   * the schedule THROTTLED the correction 2-4x (chevron beading
        //     -18.3% -> -7.5%, luma return +59.9% -> +15.5%) while helping
        //     neither the monochrome case nor compact colour. It is the wrong
        //     instrument for this job: the schedule reports that legal carrier
        //     is PRESENT, but the corner leak rides ON TOP of legal carrier at
        //     an edge, so it protected whole pixels. Parallax reports what
        //     FRACTION of the energy fails to null in the aperture, which is
        //     the proportional question removing a component actually poses.
        // Measured with parallax alone: compact colour (earring) +0.2%,
        // monochrome false colour -20.7% frame-wide, luma return +26.6%.
        //
        // THE GATE MUST VARY AT ENVELOPE SCALE. Multiplication in space is
        // convolution in frequency: a per-pixel gain smears kappa's spectrum
        // into fSC and MANUFACTURES carrier-rate content, which then lands in
        // luma via Y = notch + leak and reads as a garish alternating edge.
        // Measured with a raw per-pixel gate: ~28% of the injected luma change
        // sat at fSC. Van Cittert cannot produce that itself (it stalls at fSC
        // by construction), so the gate was the only possible source. Smooth
        // the gate with the encoder's own chroma-envelope kernel before
        // applying it -- a weight gathered AND applied at envelope scale cannot
        // manufacture out-of-band sidebands.
        // The schedule gate is RESTORED. It was removed earlier on chroma-side
        // metrics (beading/saturation), which the Y-alternation finding then
        // invalidated: with parallax alone the corrector injects +28.5% luma
        // alternation at thin straps versus +10.5% with the schedule in place,
        // and the straps read garishly worse. Chroma metrics could not see it
        // because strap SATURATION is flat in every configuration -- the damage
        // is entirely on the luma side.
        // Schedule side of the gate: the sanctioned table-owned luma proof,
        // not a local ramp. carrierIllegalProof() is ZERO through the entire
        // ambiguous middle by design ("real chroma is never claimed as luma"),
        // which is exactly what a leak that RETURNS energy to Y needs -- the
        // previous local form set unsupported evidence to a half-gate
        // (+ (1-supp)*0.5), desaturating genuine chroma wherever the axes could
        // not decide. Contradiction is consumed distinctly from absence, the
        // same fail-closed rule the schedule licenses use: one axis that
        // decisively votes legal carrier revokes the luma claim outright,
        // whereas an absent/abstaining axis merely fails to support it.
        // LDCD_CORNER_ABSTAIN=1 frees the abstaining case below. MEASURED AND
        // DEFAULTED OFF 2026-08-05 -- the reasoning is sound and the result
        // does not earn its place; see the census and the trade recorded at
        // the gate.
        static const bool abstainIsNeutral = []{
            const char *s = std::getenv("LDCD_CORNER_ABSTAIN");
            return s && std::atoi(s) != 0;
        }();
        for (int x = 0; x < width; ++x) {
            const double par = ramp(ratio[x], kCornerParallaxSoft,
                                    kCornerParallaxHard);
            double g = par;
            double contra = 0.0, proof = 1.0;
            if (analysisRow) {
                contra = std::clamp(
                    (double)analysisRow[x].conformanceContradictionFraction,
                    0.0, 1.0);
                if (contra > 0.0)
                    g = 0.0;                 // observed legal-carrier vote: protect
                else {
                    proof = lddecode::carrierIllegalProof(
                        (double)analysisRow[x].carrierConformance,
                        (double)analysisRow[x].conformanceSupportFraction);
                    // AN ABSTENTION IS NOT A VOTE AGAINST (user, 2026-08-05).
                    // The schedule has three states and only two are opinions:
                    //   contra > 0  an axis decisively votes LEGAL CARRIER
                    //               -> outright veto, unchanged;
                    //   proof  > 0  the schedule positively proves illegality
                    //               -> it scales the correction, unchanged;
                    //   proof == 0  with no contradiction, the schedule has NO
                    //               OPINION -- and multiplying by it turned
                    //               silence into a veto.
                    // carrierIllegalProof is zero through the whole ambiguous
                    // middle BY DESIGN ("real chroma is never claimed as
                    // luma"), a precision guarantee written for a channel with
                    // no independent corroboration. Parallax IS independent
                    // corroboration: measured at diagonal luma edges on the
                    // cube it reads 0.70 against 0.47 elsewhere, versus the
                    // established populations (colour p50 0.05-0.12, pure luma
                    // p50 0.89). Census over the shot: the licence refused ~85%
                    // of the correction there, and 57-65% of the refusals were
                    // this abstention -- against only 4-13% genuine contra.
                    //
                    // This is NOT the reverted "parallax alone" form (+28.5%
                    // luma alternation at thin straps against +10.5%). That
                    // experiment removed the whole schedule block, contra veto
                    // included. Here a decisive legal-carrier vote still
                    // revokes outright and positive proof still scales; only
                    // silence stops vetoing, and there parallax decides alone.
                    // The per-sample step this leaves at proof == 0 is carried
                    // into gateSmooth by the encoder's chroma-envelope kernel
                    // below, which is why the gate is smoothed before it acts.
                    //
                    // MEASURED, AND IT DOES NOT EARN ITS PLACE (2026-08-05).
                    // Freeing the abstention raises the diagonal gate 0.111 ->
                    // 0.639 as intended, but it also raises the gate EVERYWHERE
                    // (non-diagonal 0.033 -> 0.533, 16x), and the cost lands in
                    // luma:
                    //               abstain OFF            abstain ON
                    //  cube strut   alt -1.2%              fc -2.3%, alt +4.9%
                    //  faces/unifm  alt -3.0%              colour 0, alt +2.4%
                    //  beach strap  alt -1.1%, sat +0.1%   sat -0.4%, alt +2.7%
                    // As licensed today the corrector is a mild WIN on luma
                    // alternation everywhere; freeing the abstention flips it to
                    // a loss in the same class and sign as the +28.5% at thin
                    // straps that restored this gate originally (smaller only
                    // because contra still vetoes). An error class outranks the
                    // colour gain.
                    //
                    // THE STATED RATIONALE IS WRONG EVEN THOUGH THE BEHAVIOUR IS
                    // RIGHT. The abstention veto is justified as chroma
                    // precision, but freeing it costs 0.3-0.4% saturation on the
                    // most saturated program material on the disc -- it was
                    // never protecting real chroma. What it actually does is act
                    // as an ACCURACY FILTER on the Van Cittert leak estimate,
                    // confining the correction to samples where that estimate is
                    // well-founded; without it the estimate's own carrier-rate
                    // error reaches Y. The open question is the estimate's
                    // accuracy at ~0.64 authority, not its permission. Re-test
                    // this licence AFTER the estimate improves; do not re-run it
                    // as a tuning exercise before then.
                    if (proof > 0.0 || !abstainIsNeutral)
                        g *= proof;
                }
            }
            gate[x] = g;

        }
        for (int x = 0; x < width; ++x) {
            double acc = 0.0, wsum = 0.0;
            for (int t = 0; t < lddecode::kChromaEnvelopeTaps; ++t) {
                const int o = x + t - lddecode::kChromaEnvelopeTaps / 2;
                if (o < 0 || o >= width) continue;
                const double w = lddecode::kChromaEnvelopeFilter[t];
                acc += w * gate[o]; wsum += w;
            }
            gateSmooth[x] = (wsum > 0.0) ? acc / wsum : gate[x];
        }

        // ---- One lawful estimate of the carrier-envelope curvature D2A. -----
        //
        // The bandpass assumes a constant carrier ENVELOPE as well as a constant
        // luma foundation. With c[x+-2] = -A[x+-2]*basis,
        //   B{C}[x] = basis*(0.5*A[x] + 0.25*A[x-2] + 0.25*A[x+2])
        //           = C[x] + 0.25*basis*D2{A}[x]
        // so the full error the bandpass carries is TWO curvature terms:
        //   bp = C + 0.25*basis*D2A - 0.25*D2Y
        //
        // D2A appears in BOTH corrections this loop performs, and they MUST be
        // one estimate or their difference lands in Y:
        //   * the envelope over-estimate withdrawn from the leak (+envExcess);
        //   * the chroma's own image in the notch band. N = I - B has response
        //     cos^2(w) (zero only AT fSC), and N{C} = -0.25*basis*D2A, so a
        //     compact colour feature -- fast envelope, content away from band
        //     centre -- injects curvature into D2{notch} that the luma solve
        //     would otherwise read as false leak. Removing it is notch +=
        //     0.25*basis*D2A = notch + envExcess.
        // Previously these were TWO different D2A estimates: the notch used a
        // BROADBAND N{chromaEst} off the raw (luma- and noise-contaminated)
        // chroma estimate, the leak used a lawfully projected one off bp. They
        // disagreed, and the disagreement was carrier-rate alternation in Y.
        //
        // The single estimate: demodulate the current carrier estimate, project
        // onto the encoder's own chroma-envelope band (the sanctioned P, applied
        // EXACTLY ONCE to a quantity that is never a prior P output, so it
        // cannot compound), stride-2 second difference, remodulate.
        //   envExcess[x] = 0.25 * basis * D2A[x]   (composite domain)
        //
        // Recurrence: the carrier estimate is bp cleaned of the LUMA leak
        // (bp + 0.25*gate*kappa) -- NOT of the envelope term itself, since
        // feeding the projected envelope back into its own input is exactly
        // what would compound the FIR. Round 0 has kappa = 0, so the first
        // Ahat = P{demod bp}; later rounds sharpen it with the luma solve.
        // Because envExcess no longer needs kappa, the notch correction now
        // runs from round 0 -- the envelope contamination was always present,
        // the old chicken-and-egg (chromaEst needs kappa) was the only reason
        // it waited. A wrong leak shows up as carrier-rate ALTERNATION in Y
        // (Y = notch + leak; a correct leak reconstructs the sharp luma with
        // none), so that alternation is the metric to watch.
        std::fill(kappa.begin(), kappa.end(), 0.0);
        std::fill(envExcess.begin(), envExcess.end(), 0.0);
        for (int outer = 0; outer < kCornerOuterRounds; ++outer) {
            // Carrier estimate: bp with the current (gated) luma leak removed.
            for (int x = 0; x < width; ++x) {
                const int ph = carrierSampleClass(line, left + x);
                const double cEst = bpLine[x] + 0.25 * gateSmooth[x] * kappa[x];
                envI[x] = 2.0 * cEst * sin4fsc(ph);
                envQ[x] = 2.0 * cEst * cos4fsc(ph);
            }
            // Lawful envelope: the sanctioned projection, applied once.
            lddecode::projectExpressibleChromaEnvelope(envI.data(), nullptr,
                                                       width, sEnvI.data());
            lddecode::projectExpressibleChromaEnvelope(envQ.data(), nullptr,
                                                       width, sEnvQ.data());
            // The single D2A term, in composite: envExcess = 0.25*basis*D2A.
            for (int x = 0; x < width; ++x) {
                const int m2 = std::clamp(x - 2, 0, width - 1);
                const int p2 = std::clamp(x + 2, 0, width - 1);
                const int ph = carrierSampleClass(line, left + x);
                const double d2I = sEnvI[m2] - 2.0 * sEnvI[x] + sEnvI[p2];
                const double d2Q = sEnvQ[m2] - 2.0 * sEnvQ[x] + sEnvQ[p2];
                // 0.5*(...) undoes the 2x demod gain.
                envExcess[x] = 0.25 * 0.5 * (d2I * sin4fsc(ph) + d2Q * cos4fsc(ph));
            }
            // Notch corrected for the chroma that leaks INTO it, using the SAME
            // envExcess: N{chroma} = -envExcess in composite, so notch += it.
            for (int x = 0; x < width; ++x)
                notchAdj[x] = notch[x] + envExcess[x];
            std::fill(mObs.begin(), mObs.end(), 0.0);
            for (int x = 2; x < width - 2; ++x)
                mObs[x] = notchAdj[x - 2] - 2.0 * notchAdj[x] + notchAdj[x + 2];

            // Van Cittert deconvolution of S. Error propagates as
            // (I - S) = sin^2, so this stalls at fSC by construction and never
            // claims that mode.
            std::fill(kappa.begin(), kappa.end(), 0.0);
            for (int it = 0; it < kCornerRecoveryDepth; ++it) {
                applyS(kappa, sKappa);
                for (int x = 0; x < width; ++x)
                    kappa[x] += (mObs[x] - sKappa[x]);
            }
        }
        // TEMPORARY INSTRUMENT (LDCD_PROBE_VC=1): grade the Van Cittert
        // estimator against CERTIFIED TRUTH, before any gate.
        //
        //   dG conservation:  merged = Lhat + BP(chat),  exactCarrier = BP(chat)
        //   so on a covered def line   Lhat[x] = raw[x] - exactCarrier[x]
        //
        // is carrier-free luma at FULL bandwidth, fSC included. kappa claims to
        // estimate D2_stride2{Y}; here D2_stride2{Lhat} is that quantity as a
        // conservation fact. The referee is legitimate because
        // locked1DRawBandpass is the pure 0.50c - 0.25(m2+p2) formula off raw,
        // written once in buildCarrierAnalysis and never fact-injected -- kappa
        // has not seen the exact channel.
        //
        // The question: is the residual (truth - kappa) concentrated at fSC?
        // Van Cittert's error profile is sin^2, frozen at the subcarrier, so it
        // should recover the non-fSC part and miss the rest -- and that missed
        // part, returned as its complement in Y, is the alternation.
        if (probeVanCittert) {
            const float *ex = exactCarrierRow(line);
            if (ex) {
                static const int c4[4] = { 1, 0, -1, 0 };
                static const int s4[4] = { 0, 1, 0, -1 };
                auto fscMag = [&](const double *v, int x) {
                    double i = 0.0, q = 0.0;
                    for (int k = -4; k < 4; ++k) {
                        const int ph = (x + k) & 3;
                        i += v[x + k] * c4[ph];
                        q += v[x + k] * s4[ph];
                    }
                    return std::hypot(i, q) * 0.25;
                };
                std::vector<double> yTrue(width, 0.0), d2True(width, 0.0),
                                    resid(width, 0.0);
                std::vector<char> ok(width, 0);
                for (int x = 0; x < width; ++x) {
                    const double e = (double)ex[left + x];
                    if (!std::isfinite(e)) continue;
                    yTrue[x] = (double)rawLine[left + x] - e;
                    ok[x] = 1;
                }
                for (int x = 2; x < width - 2; ++x) {
                    if (!ok[x - 2] || !ok[x] || !ok[x + 2]) continue;
                    d2True[x] = yTrue[x - 2] - 2.0 * yTrue[x] + yTrue[x + 2];
                    resid[x]  = d2True[x] - kappa[x];
                }
                for (int x = 6; x < width - 6; ++x) {
                    if (!ok[x - 2] || !ok[x] || !ok[x + 2]) continue;
                    // Condition on the stranded peak's OWN amplitude: the
                    // bandpass envelope |C| over +-2 samples, in IRE.
                    double amp = 0.0;
                    for (int k = -2; k <= 2; ++k)
                        amp = std::max(amp, std::fabs(bpLine[x + k]));
                    ldcdVcGrade(d2True[x], kappa[x], resid[x],
                                fscMag(d2True.data(), x),
                                fscMag(kappa.data(), x),
                                fscMag(resid.data(), x), 1.0 / irescale,
                                amp / irescale);
                }
            }
        }

        for (int x = 0; x < width; ++x)
            kappa[x] *= gateSmooth[x];

        // Total withdrawal: envelope over-estimate MINUS the luma leak.
        //   chroma = bp - leakRow,  Y = raw - chroma.
        // kappa already estimates the stride-2 second difference D2Y, and the
        // luma leak the bandpass added is exactly -0.25*D2Y, so the luma term
        // is -0.25*kappa[x] with no further smoothing. An earlier stride-1
        // [0.25,0.5,0.25] smoother on kappa here was a SECOND regulariser on
        // the fSC neighbourhood (its response is exactly 0.5 at fSC), redundant
        // with the Van Cittert recovery profile that already owns that band;
        // it silently halved every fSC-rate leak the solve had recovered.
        // TWO SEPARATIONS TRIED AND BOTH FALSIFIED (2026-08-05), while hunting
        // why this stage's luma alternation grows with its own correction.
        //
        // (a) "envExcess is the fSC injector." It is a remodulation onto the
        //     carrier basis, so it is at fSC by construction and carries no
        //     licence, while kappa is licensed and cannot produce fSC -- so it
        //     looked like the whole alternation channel. Withholding it from
        //     the leak made alternation WORSE, not better (cube -1.2% -> -0.3%,
        //     faces -3.0% -> -0.1%): it is mildly helping.
        // (b) "the gate's spatial variation AM-modulates kappa into fSC."
        //     Removing the gate entirely -- the largest correction and the only
        //     one with zero AM -- made alternation worse again (+6.6% cube,
        //     +13.8% faces, +13.6% straps).
        //
        // Both refuted the same way: alternation is strictly MONOTONIC in
        // correction magnitude, through gate 0.11 -> 0.64 -> 1.0. The cause is
        // kappa itself, and the referee below (LDCD_PROBE_VC) named it: graded
        // against certified D2{Lhat}, kappa's residual is 67% (cube) / 91%
        // (beach) fSC, and kappa does NOT abstain at the subcarrier -- it emits
        // 55%/95% of truth's fSC magnitude in the WRONG PHASE, so the error
        // adds instead of cancelling. Every unit applied pushes wrong-phase fSC
        // into Y. That is not fixable by licensing or by term separation: the
        // fSC null is a single-line law, and no single-line test can separate
        // fSC luma from fSC chroma. See the iceberg pre-implementation plan.
        for (int x = 0; x < width; ++x)
            leakRow[x] = -0.25 * kappa[x] + envExcess[x];

        // ---- Coarse-residual carrier hull (P5). -----------------------------
        //
        // The corner leak estimates carrier from a single line and, at a
        // compact feature, the encoder bandwidth law forbids the lawful
        // envelope from modelling the fast on/off -- so P2 under-corrects and
        // leaves carrier-rate content that is really luma. The coarse residuals
        // bound the carrier INDEPENDENTLY, from the luma side where compact
        // content is lawful (luma is never bandlimited): carrier must lie in
        // [raw - max_v apMean, raw - min_v apMean] over the covering apertures.
        // Clamp the emitted carrier (bp - leak) into that range and fold the
        // excess back into the leak (so it lands in luma via Y = raw - chroma);
        // both bounds are applied so an oscillating carrier is not rectified.
        // The clamp itself is the shared applyCarrierFeasibilityHull(), which
        // the bucket path also calls -- here it acts on the emitted carrier and
        // the difference is backed out into leakRow. v1 is unguarded: the dark-
        // side ceiling is clean but the pixel luma floor can be violated at a
        // lone dark sample, which would clip legal carrier -- the decomposed
        // metric (satRet/hueRot) is what reveals it.
        static const bool hullEnabled = []{
            const char *s = std::getenv("LDCD_CORNER_HULL");
            return !s || std::atoi(s) != 0;         // default ON when leak runs
        }();
        if (hullEnabled && apMean) {
            for (int x = 0; x < width; ++x)
                notchAdj[x] = bpLine[x] - leakRow[x];   // emitted carrier
            applyCarrierFeasibilityHull(line, notchAdj.data());
            for (int x = 0; x < width; ++x)
                leakRow[x] = bpLine[x] - notchAdj[x];   // excess -> leak -> luma
        }
    }

    if (probeVanCittert) ldcdVcGradeReport((int)heldSeq1);

    // Retained record from the removed LDCD_DUMP_CORNER dump
    //
    // The old scalar was a stride-1 second difference of bp. For a pure fSC
    // carrier |D2_1{C}| = 2|C| and |D2_2{C}| = 4|C|: BOTH are proportional to
    // carrier amplitude pointwise, so a second-difference scalar cannot tell a
    // beading reduction from a desaturation. Retired. The correction's effect
    // decomposes EXACTLY into two disjoint channels:
    //     Y  += leak                       (luma gains the withdrawn energy)
    //     rendered chroma envelope += -P{demod(leak)}
    // so the honest instruments are (a) the fSC-rate content the leak injects
    // into luma -- zero for any pure legal carrier, hence non-tautological and
    // PRIMARY -- and (b) what the withdrawal does to the rendered envelope,
    // split into magnitude (legal-carrier retention, a cost) versus hue
    // rotation (a cross-colour signature, a defect). Conservation is demoted to
    // an assertion: Y + chroma == raw holds for ANY leak by construction, so it
    // measures arithmetic, not the model.
}

// Shared with the retraction ladder (defined near the retracted-view code):
// working-space phase snap -- est keeps amplitude, ref supplies phase.
// Optional per-sample gate [0,1] scales the snap's authority (twin
// agreement for temporal references); nullptr = full authority.
static void ldcdApplyPhaseSnap(const std::vector<double> &est,
                               const std::vector<double> &ref,
                               std::vector<double> &out,
                               int width, double irescale,
                               double ampMinIRE, double ampTauIRE,
                               bool clampRatio,
                               const double *gate = nullptr);

// Windowed vector-coherence ALPHA between two sign-aligned scalar
// carriers in the snap basis: 0 where either window is incomplete, ramped
// 0.55 -> 0.85 on the resultant coherence. A RAMP, never a cut -- binary
// per-sample gates interleave two differently-phased renders at pixel
// pitch along edges, which renders as checkerboard (the OOB-cut lesson,
// re-learned on the beach 2026-07-30).
static void ldcdSideCoherenceAlpha(const std::vector<double> &a,
                                   const std::vector<double> &b,
                                   int width, std::vector<double> &alpha)
{
    constexpr int kSnapHalf = 4;
    static const int cB[4] = { 1, 0, -1, 0 };
    static const int sB[4] = { 0, 1, 0, -1 };
    std::fill(alpha.begin(), alpha.end(), 0.0);
    // Centered 9-tap, half-weight ends (the 6f502ec integer-centroid
    // convention): the old [-3,+4] window carried a +0.5 centroid that
    // biased WHERE the alpha opens by half a sample at envelope edges.
    for (int xi = kSnapHalf; xi < width - kSnapHalf; ++xi) {
        double aI = 0, aQ = 0, bI = 0, bQ = 0;
        bool ok = true;
        for (int k = xi - kSnapHalf; k <= xi + kSnapHalf && ok; ++k) {
            if (!std::isfinite(a[k]) || !std::isfinite(b[k])) {
                ok = false; break;
            }
            const double w =
                (k == xi - kSnapHalf || k == xi + kSnapHalf) ? 0.5 : 1.0;
            const int ph = k & 3;
            aI += w * a[k] * cB[ph]; aQ += w * a[k] * sB[ph];
            bI += w * b[k] * cB[ph]; bQ += w * b[k] * sB[ph];
        }
        if (!ok) continue;
        const double ma = std::hypot(aI, aQ);
        const double mb = std::hypot(bI, bQ);
        const double mr = std::hypot(aI + bI, aQ + bQ);
        const double coh = mr / (ma + mb + 1e-12);
        alpha[xi] = std::clamp((coh - 0.55) / 0.30, 0.0, 1.0);
    }
}

// Grammar-sign-aligned certified bracket mean: the +/-1 certified rows
// brought into the center line's carrier-phase space. NaN where either
// bracket sample is absent or the grammar cannot certify the relation.
// ONE implementation -- the head snap and the retraction ladder both call
// here.
static void ldcdBuildCertBracketAligned(
    const lddecode::CarrierGrammarState *gC,
    const lddecode::CarrierGrammarState *gU,
    const lddecode::CarrierGrammarState *gD,
    const float *exU, const float *exD,
    int left, int width, std::vector<double> &bAlign,
    std::vector<double> *legUOut = nullptr,
    std::vector<double> *legDOut = nullptr)
{
    std::fill(bAlign.begin(), bAlign.end(),
              std::numeric_limits<double>::quiet_NaN());
    if (legUOut) std::fill(legUOut->begin(), legUOut->end(),
                           std::numeric_limits<double>::quiet_NaN());
    if (legDOut) std::fill(legDOut->begin(), legDOut->end(),
                           std::numeric_limits<double>::quiet_NaN());
    if (!gC || !gU || !gD || !exU || !exD) return;
    for (int xi = 0; xi < width; ++xi) {
        const int h = left + xi;
        const float eu = exU[h];
        const float ed = exD[h];
        if (!std::isfinite(eu) || !std::isfinite(ed)) continue;
        const auto rU = lddecode::carrierGrammarSignedPhaseRelation(gC, h, gU, h);
        const auto rD = lddecode::carrierGrammarSignedPhaseRelation(gC, h, gD, h);
        double sU, sD;
        if (rU == lddecode::CarrierPhaseRelation::Opposite) sU = -1.0;
        else if (rU == lddecode::CarrierPhaseRelation::Same) sU = 1.0;
        else continue;
        if (rD == lddecode::CarrierPhaseRelation::Opposite) sD = -1.0;
        else if (rD == lddecode::CarrierPhaseRelation::Same) sD = 1.0;
        else continue;
        bAlign[xi] = 0.5 * (sU * (double)eu + sD * (double)ed);
        if (legUOut) (*legUOut)[xi] = sU * (double)eu;
        if (legDOut) (*legDOut)[xi] = sD * (double)ed;
    }
}

void Comb::FrameBuffer::buildPhaseCorrected1D()
{
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;
    const int fullWidth = videoParameters.fieldWidth;

    static const double parallaxRepairMaxDeltaIRE = []{
        const char *s = std::getenv("LD_1D_PARALLAX_MAX_DELTA_IRE");
        return s ? std::atof(s) : 0.35;
    }();
    // Pass-1.5 mode. The repair is an EXPERIMENT and is opt-in, as the
    // buffer-flow doc has always described it:
    //   unset   -- analysis only, the source is untouched (default)
    //   report  -- analyse and log, still do not touch the source
    //   apply   -- commit the bounded move
    // It had lost its gate and was running unconditionally on every locked
    // frame, worth ~0.9 IRE rms of luma difference from bucket frame-wide.
    // Restoring the gate is a doc/code reconciliation, not a new policy.
    static const bool parallaxRepairApply = []{
        const char *s = std::getenv("LD_1D_PARALLAX_REPAIR");
        return s && std::strcmp(s, "apply") == 0;
    }();

    if (!configuration.phaseCompensation || width <= 0 || first >= last)
        return;

    const size_t magnitudeCount =
        static_cast<size_t>(demodLines) * demodWidth;
    if (demodIQMag4fsc_flat.size() < magnitudeCount)
        demodIQMag4fsc_flat.resize(magnitudeCount, 0.0f);

    // scratch_preI carries the ordinary raw 1D bandpass line.
    // scratch_preQ carries the restrained source actually emitted downstream.
    if (static_cast<int>(scratch_preI.size()) < width)
        scratch_preI.resize(width, 0.0);
    if (static_cast<int>(scratch_preQ.size()) < width)
        scratch_preQ.resize(width, 0.0);

    auto clamp01 = [](double v) {
        return std::clamp(v, 0.0, 1.0);
    };

    // Wide-window cross-color detector (narrow-vs-wide carrier coherence).
    //
    // Real chroma is a coherent carrier sinusoid that holds phase over many
    // cycles; luma-near-fsc looks carrier-like over one cycle but is
    // incoherent over a wide aperture.
    //
    // NARROW (tighter) estimate: a rolling, current-centered mean of the
    // 2-sample carrier-fit envelope hypot(bp[x], bp[x+1]).  The point envelope
    // equals the carrier amplitude A only at perfect fsc and ripples at 2fsc
    // under real phase error; a 4-cycle centered mean nulls that 2fsc/4fsc
    // ripple, so the narrow term reads a clean A on coherent carrier.  This
    // replaces the smooth carrierFit envelope the witness rollback removed.
    //
    // WIDE estimate: an 8-cycle coherent demod sum.  On coherent carrier both
    // read A, so risk is a true ripple-free 0; on luma-near-fsc the coherent
    // wide fit cancels while the narrow mean-of-envelope does not, so risk > 0.
    //
    // The metric is published as carrierImpurity (a provisional oracle).
    // It is NEVER applied to the carrier source; the source is emitted clean.
    // Suppression happens downstream as alpha at color demod and Y subtraction.
    // measurePostCombImpurity() later replaces this provisional 1D read with
    // the elected-comb measurement that splitIQlocked() actually consumes.
    // The doc's exact form is used with no shaping: any shape made it
    // unresponsive somewhere.
    constexpr int    kNarrowWin = 16;          // 4 carrier cycles (nulls 2fsc/4fsc)
    constexpr int    kWideWin = 32;            // 8 carrier cycles
    constexpr double kImpurityFloorIRE = 2.0;  // ignore low-amplitude noise

    // Period-4 quadrature reference.  Any fixed phase-locked basis works; the
    // coherent-sum magnitude is invariant to its rotation.
    static const double cosRef[4] = { 1.0, 0.0, -1.0, 0.0 };
    static const double sinRef[4] = { 0.0, 1.0,  0.0, -1.0 };

    // Per-line scratch, allocated once and reused across lines.
    std::vector<double> demI(width), demQ(width);
    std::vector<double> preI(width + 1), preQ(width + 1);
    std::vector<double> env(width), preEnv(width + 1);
    std::vector<double> nativeI4(width), nativeQ4(width);
    std::vector<double> snapEst(width), snapOut(width), snapRef(width);

    for (int line = first; line < last; ++line) {
        const quint16 *rawLine =
            rawbuffer.data() + static_cast<size_t>(line) * fullWidth;

        double *lockedSource = locked1DSource_line(line);
        if (!lockedSource)
            continue;

        AttributionEvidence *attribution = attributionEvidence_line(line);
        lddecode::CarrierAnalysisRecord *carrierAnalysis =
            carrierAnalysis_line(line);

        CombCarrierGrammar *grammar = carrierGrammarLine(line);
        const bool grammarLocked = grammar && grammar->grammarLocked;
        const double burstCos = grammarLocked ? grammar->burstCos : 1.0;
        const double burstSin = grammarLocked ? grammar->burstSin : 0.0;

        double lutI[4];
        double lutQ[4];
        if (grammarLocked) {
            for (int i = 0; i < 4; ++i) {
                lutI[i] = static_cast<double>(grammar->demodLUTTi[i]);
                lutQ[i] = static_cast<double>(grammar->demodLUTTq[i]);
            }
        } else {
            fusedDemodLUT(
                burstCos, burstSin, spLUT_locked, cpLUT_locked, lutI, lutQ);
        }

        double i4Scale[4];
        double q4Scale[4];
        double magnitudeScale[4];

        for (int phase = 0; phase < 4; ++phase) {
            lockedTo4fsc(
                lutI[phase], lutQ[phase], burstCos, burstSin,
                i4Scale[phase], q4Scale[phase]);

            magnitudeScale[phase] =
                boundedMag(i4Scale[phase], q4Scale[phase]);
        }

        // Retained record from the removed LDCD_DUMP_GRAMMAR_L0/L1 dump
        //
        // Its subject was the 180-degree alternation the sync probe found
        // between alternate def lines in the working space, read directly off
        // the per-line burst phasor, samplePhase0 and lineFlip.

        float *demodI = demodTI_line(line);
        float *demodQ = demodTQ_line(line);
        float *demodI4 = demodTI4fsc_line(line);
        float *demodQ4 = demodTQ4fsc_line(line);
        float *lockedI4 = locked1DTI4fsc_line(line);
        float *lockedQ4 = locked1DTQ4fsc_line(line);
        float *magnitude =
            demodIQMag4fsc_flat.data() +
            static_cast<size_t>(line) * demodWidth;

        double *rawBandpass = locked1DRawBandpass_line(line);
        double *bpLine = scratch_preI.data();
        double *restrainedLine = scratch_preQ.data();

        auto rawAtRel = [&](int rel) -> double {
            const int r = std::clamp(rel, 0, width - 1);
            return static_cast<double>(rawLine[left + r]);
        };

        // Pass 1: consume the canonical full-resolution baseline harvested by
        // buildCarrierAnalysis(). The fallback preserves standalone safety but
        // normal locked orchestration has exactly one producer.
        //
        // The corner leak is withdrawn HERE, at the point the source authority
        // enters the stage, so every later consumer (repair, restraint, demod,
        // publish, and the candidates built from the published source) sees one
        // corrected carrier rather than a patched output. Because
        //     source = bp - leak   and   Y = raw - source,
        // the withdrawn leak lands in luma automatically: Y + chroma == raw
        // stays exact, which is the conservation a desaturating suppressor
        // cannot satisfy. No output-side correction pass, so the produceY
        // boundary is untouched.
        //
        // lockedCornerLeak is zero-filled unless buildCornerLeak() ran, so this
        // subtraction is inert by construction when the stage is disabled.
        const double *cornerLeak = lockedCornerLeak_line(line);
        if (rawBandpass) {
            if (cornerLeak) {
                for (int rel = 0; rel < width; ++rel)
                    bpLine[rel] = rawBandpass[rel] - cornerLeak[rel];
            } else {
                for (int rel = 0; rel < width; ++rel)
                    bpLine[rel] = rawBandpass[rel];
            }
        } else {
            for (int rel = 0; rel < width; ++rel) {
                const double c  = rawAtRel(rel);
                const double m2 = rawAtRel(rel - 2);
                const double p2 = rawAtRel(rel + 2);
                bpLine[rel] = 0.50 * c - 0.25 * (m2 + p2);
            }
        }

        // The 1D output is never touched from other lines -- 1D is
        // downstream's safe retreat.

        // Pass 1.5: coarse-residual feasibility repair for locked 1D.
        //
        // The ordinary 1D bandpass remains the source authority; every scanner
        // below only gets to justify a small bounded move toward a short-fit-
        // compatible subset of legal residual options. Coarse residuals are
        // options, not carrier estimates. The short fit is a selector, not
        // carrier authority. The moving-centered residual is support/conflict
        // evidence only. The wide fit is comparison-only and never participates
        // in survivor selection.
        if (carrierAnalysis)
        {
            const double maxDeltaSamples =
                std::max(0.0, parallaxRepairMaxDeltaIRE) * irescale;
            float *repairStrengthRow =
                locked1DParallaxRepairStrength_line(line);
            if (repairStrengthRow)
                std::fill(repairStrengthRow, repairStrengthRow + width, 0.0f);
            float *repairDeltaRow =
                locked1DParallaxRepairDelta_line(line);
            if (repairDeltaRow)
                std::fill(repairDeltaRow, repairDeltaRow + width, 0.0f);

            for (int rel = 0; rel < width; ++rel) {
                const auto &record = carrierAnalysis[rel];
                const auto &residualDiag = record.residual;
                const int optionCount = residualDiag.optionCount;
                const double sourceSample = bpLine[rel];
                const int survivorCount = residualDiag.survivorCount();
                const double survivorLo = residualDiag.survivorLo;
                const double survivorHi = residualDiag.survivorHi;

                double proposedDelta = 0.0;
                double appliedDelta = 0.0;

                // The branch taxonomy below is unchanged; the labels are the
                // "reason" strings the removed COARSERESREPAIR log emitted.
                if (carrierAnalysis[rel].scheduleConformance ==
                    lddecode::CarrierScheduleConformance::ScheduleIllegal) {
                    // schedule-illegal-luma. Registered as luma at analysis
                    // time: there is no carrier here to repair, and the
                    // residual options are luma interpretations that would
                    // only masquerade as survivor conflict.
                } else if (optionCount <= 0) {
                    // no-options.
                } else if (survivorCount <= 0) {
                    // conflict-no-survivors.
                } else if (survivorCount == optionCount) {
                    // no-discrimination-all-survive.
                } else if (sourceSample >= survivorLo &&
                           sourceSample <= survivorHi)
                {
                    // source-inside.
                } else {
                    const bool movingCompatible = residualDiag.movingCompatible;

                    if (!movingCompatible) {
                        // moving-conflict.
                    } else {
                        const double target =
                            std::clamp(sourceSample, survivorLo, survivorHi);
                        proposedDelta = target - sourceSample;
                        appliedDelta = std::clamp(
                            proposedDelta,
                            -maxDeltaSamples,
                            maxDeltaSamples);

                        if (!parallaxRepairApply) {
                            // report-only. Analysis complete, but the repair
                            // is opt-in: the ordinary bandpass remains source
                            // authority and no repair hold is published
                            // downstream.
                            appliedDelta = 0.0;
                        } else {
                            // apply.
                            bpLine[rel] = sourceSample + appliedDelta;
                            if (repairStrengthRow && maxDeltaSamples > 1e-9) {
                                repairStrengthRow[rel] = static_cast<float>(
                                    std::clamp(
                                        std::fabs(appliedDelta) / maxDeltaSamples,
                                        0.0,
                                        1.0));
                            }
                            // Publish the signed move so the retraction stage
                            // can align carrierFit with the repaired carrier.
                            if (repairDeltaRow)
                                repairDeltaRow[rel] =
                                    static_cast<float>(appliedDelta);
                        }
                    }
                }
            }
        }

        float *impurityRow = carrierImpurity_line(line);

        // Pass 2: wide-window cross-color detector.  Publishes carrierImpurity;
        // the emitted source is the bandpass under the envelope-legality
        // restraint below (an envelope-scale weight, never a carrier-rate gain).
        //
        // Demodulate the bandpass into quadrature against the period-4
        // reference (for the wide coherent fit) and form the 2-sample fit
        // envelope (for the rolling narrow fit).
        for (int rel = 0; rel < width; ++rel) {
            const int p = carrierSampleClass(line, left + rel) & 3;
            demI[rel] = bpLine[rel] * cosRef[p];
            demQ[rel] = bpLine[rel] * sinRef[p];
            const int relN = std::min(rel + 1, width - 1);
            const double b0 = bpLine[rel], b1 = bpLine[relN];
            env[rel] = std::sqrt(b0 * b0 + b1 * b1);
        }
        preI[0] = 0.0;
        preQ[0] = 0.0;
        preEnv[0] = 0.0;
        for (int rel = 0; rel < width; ++rel) {
            preI[rel + 1] = preI[rel] + demI[rel];
            preQ[rel + 1] = preQ[rel] + demQ[rel];
            preEnv[rel + 1] = preEnv[rel] + env[rel];
        }

        // Narrow fit: rolling, current-centered mean of the 2-sample envelope.
        // The point envelope = A on coherent carrier but ripples at 2fsc under
        // phase error; the mean over 4 cycles nulls that ripple.  env[k]
        // describes the pair (k,k+1), so its physical coordinate is k+0.5:
        // indices center-8 through center+7 are already centred on `center`.
        auto narrowEnvIRE = [&](int center) -> double {
            const int a = std::clamp(center - kNarrowWin / 2, 0, width);
            const int b = std::clamp(a + kNarrowWin, 0, width);
            const double n = static_cast<double>(std::max(1, b - a));
            return ((preEnv[b] - preEnv[a]) / n) * invIreScale;
        };

        // Envelope-legality restraint on the emitted source (encoder
        // bandwidth law, imposed at envelope scale).
        //
        // A legal chroma envelope is bandlimited to 1.3 MHz -- ~11 samples
        // at 4fSC -- so corroboration evidence about the envelope is
        // meaningful only at that scale, and the ceiling is the encoder's
        // own law: demodulated I/Q passed through the encoder's 1.3 MHz
        // chroma kernel is everything the encoder could have modulated
        // here; envelope the source holds above that (plus noise slack) is
        // inexpressible as chroma.  It is luma the blind bandpass
        // swallowed, and restraining it returns the energy to Y through
        // raw - lockedSource, where it belongs.  Both sides of the ratio
        // are smoothed by the SAME kernel, so the comparison never mixes
        // scales.  (Two falsified ceilings, kept as negative results: the
        // 2-sample POINT envelope under an envelope-scale ceiling rectifies
        // noise/sideband ripple into ~25% desaturation of legal saturated
        // bars; a coherent VECTOR mean over the law window punishes legal
        // I/Q modulation -- the law bounds envelope bandwidth, not phasor
        // constancy -- and still cost the bars ~21%.)
        //
        // The historical prohibition on any source gain ("checkerboard by
        // construction") was a prohibition on CARRIER-RATE gain: a
        // bandlimited carrier times a fast gain is amplitude modulation that
        // manufactures out-of-band sidebands.  This weight is gathered at
        // envelope scale and applied through the encoder's own envelope
        // kernel, so it varies no faster than a legal envelope may -- it
        // cannot manufacture sidebands.  Genuine chroma, including legal
        // 1.3 MHz edges, is expressible by construction and passes at
        // w = 1; the weight never exceeds 1, so it can only return energy
        // to Y, never manufacture carrier.

        // Aperture cross-color detector.  Publishes gA = aperture contamination
        // as carrierImpurity; gA is never a source gain.
        //
        // gA is NOT applied to the source.  Coherent contamination (dubbed
        // cross-color) is corroborated at envelope scale, so the legality
        // restraint above passes it untouched; discriminating it from
        // authentic chroma is gA's job, and that correction lives on the
        // COLOR side only, as the gA alpha applied in splitIQlocked().
        // A per-pixel gA gain on the source would be carrier-rate AM, and
        // removing coherent carrier from the source would strand its
        // complement in Y as checkerboard. The lockedProduct chroma path
        // therefore remains separate from luma policy.
        //
        // Winding is deliberately NOT used: control measurements on saturated
        // clothing showed coh/turn overlap authentic chroma and contamination,
        // so winding cannot discriminate without a luma-coupling guard.  gA reads
        // 0 on the uniforms and 0.2-0.3 on title cross-color, so aperture alone
        // is the correct discriminator.
        // Hoist line-invariant bounds out of the per-pixel loop. fvfMetrics is
        // a 2D vector keyed on (line, rel); both bounds are constants within
        // this line. Attribution likewise is a per-line pointer.
        const int fvfRelLimit =
            (line >= 0 && line < static_cast<int>(fvfMetrics.size()))
                ? static_cast<int>(fvfMetrics[line].size())
                : 0;
        auto *fvfLineRow =
            fvfRelLimit > 0 ? fvfMetrics[line].data() : nullptr;
        for (int rel = 0; rel < width; ++rel) {
            // Stable centre Zwide on the cycle grid (8-cycle complex mean).
            const double ZwI = centeredEvenWeightMean(
                demI.data(), preI.data(), width, rel, kWideWin);
            const double ZwQ = centeredEvenWeightMean(
                demQ.data(), preQ.data(), width, rel, kWideWin);

            const double narrowMag = narrowEnvIRE(rel);
            const double wideMag = 2.0 * boundedMag(ZwI, ZwQ) * invIreScale;

            // Aperture contamination: clamp((narrow - wide)/max(floor, narrow)).
            double gA = 0.0;
            if (narrowMag > kImpurityFloorIRE && wideMag < narrowMag) {
                gA = clamp01(
                    (narrowMag - wideMag) /
                    std::max(kImpurityFloorIRE, narrowMag));
            }

            // The emitted source is the full-resolution ordinary carrier plus
            // any explicitly bounded 1D repair. Whitestar and the fits are
            // evidence/policy inputs only; no diagnostic projection becomes
            // picture here.
            //
            // The envelope-legality hull was REMOVED from the 1D source
            // (2026-07-25). A hull presumes a safer value to retreat to when
            // the estimate looks illegal; in produceY an outlier candidate can
            // simply lose an election. In 1D there is no such harbour -- the
            // bandpass IS the only estimate -- so bounding it is not a choice
            // between candidates but an unconditional subtraction from the sole
            // source, and whatever it removes lands in luma via Y = raw - src.
            // Worse, the ceiling is measured with the encoder's 9-tap 1.3 MHz
            // kernel, which smooths ACROSS a thin feature and therefore reads
            // its envelope as smaller than it is: the hull then cuts LEGAL
            // carrier at exactly the compact features it should protect, and
            // that carrier reappears in luma as carrier-rate alternation.
            // Measured on the beach strap (pure luma, --chroma-gain 0, vs
            // bucket): hull on 148 rms / peak 1383; hull off 83 / 230 -- the
            // strap-local dominant term while barely moving the frame-wide
            // figure, which is the signature of something that only bites on
            // thin detail. With the hull and the Pass-1.5 repair both out, the
            // locked 1D luma is BIT-IDENTICAL to bucket.
            restrainedLine[rel] = bpLine[rel];

            if (impurityRow)
                impurityRow[rel] = static_cast<float>(gA);

            if (fvfLineRow && rel < fvfRelLimit) {
                fvfLineRow[rel].intakeNyquistRiskIRE = gA * narrowMag;
            }

            if (attribution) {
                AttributionFacts &facts = attribution[rel].facts;
                facts.bandpassFineIRE = narrowMag;
                facts.bandpassCoarseIRE = wideMag;
                facts.lumaExcursionIRE = gA * narrowMag;
            }
        }

        // Optional publication is deliberately outside the render-facing loop
        // above.  When no analysis client is active, ordinary locked output has
        // no extra per-pixel branch or storage traffic.
        if (carrierAnalysis) {
            for (int rel = 0; rel < width; ++rel) {
                const int p = carrierSampleClass(line, left + rel) & 3;
                const double ZwI = centeredEvenWeightMean(
                    demI.data(), preI.data(), width, rel, kWideWin);
                const double ZwQ = centeredEvenWeightMean(
                    demQ.data(), preQ.data(), width, rel, kWideWin);
                const double wideSample =
                    2.0 * (ZwI * cosRef[p] + ZwQ * sinRef[p]);
                const double wideMag =
                    2.0 * boundedMag(ZwI, ZwQ) * invIreScale;
                const double narrowMag = narrowEnvIRE(rel);

                auto &record = carrierAnalysis[rel];
                if (!record.fit.valid) {
                    const double ZnI = centeredEvenWeightMean(
                        demI.data(), preI.data(), width, rel, kNarrowWin);
                    const double ZnQ = centeredEvenWeightMean(
                        demQ.data(), preQ.data(), width, rel, kNarrowWin);
                    const double shortSample =
                        2.0 * (ZnI * cosRef[p] + ZnQ * sinRef[p]);
                    const double sourceSample = bpLine[rel];

                    record.fit.sourceSample = static_cast<float>(sourceSample);
                    record.fit.shortSample = static_cast<float>(shortSample);
                    record.fit.wideSample = static_cast<float>(wideSample);
                    record.fit.sourceMinusShortIRE = static_cast<float>(
                        (sourceSample - shortSample) * invIreScale);
                    record.fit.shortMinusWideIRE = static_cast<float>(
                        (shortSample - wideSample) * invIreScale);
                    record.fit.sourceMinusWideIRE = static_cast<float>(
                        (sourceSample - wideSample) * invIreScale);
                    record.fit.valid = true;
                }

                record.fit.shortMagnitudeIRE = static_cast<float>(narrowMag);
                record.fit.wideMagnitudeIRE = static_cast<float>(wideMag);
                const float apertureImpurity = static_cast<float>(
                    (narrowMag > kImpurityFloorIRE && wideMag < narrowMag)
                        ? clamp01((narrowMag - wideMag) /
                                  std::max(kImpurityFloorIRE, narrowMag))
                        : 0.0);
                record.carrierImpurity = std::max(
                    record.carrierImpurity,
                    apertureImpurity);
            }
        }

        // Pass 3a: publish two distinct NATIVE products without confusing
        // their phase contracts:
        //
        //   * demodI/Q and demodI/Q4 are sample-local carrier products;
        //   * restrainedLine is already a scalar carrier in physical composite
        //     sample geometry.
        //
        // The scalar must therefore be copied, not demodulated and remodulated.
        // The locked basis includes CAL_EPS while the Grid4fsc remodulator does
        // not; round-tripping the scalar through those unlike bases multiplies
        // it by cos(CAL_EPS*pi/2), leaving a small carrier residue in every
        // downstream raw-minus-carrier reconstruction.
        // Certified-field head replacement (stage 1): on def lines the 1D
        // scalar IS the conservation fact -- chat is already in def phase and
        // raw-aligned, so no basis work is needed. Per-sample repair holes
        // (NaN) keep the model's scalar; the hole boundary is a sample the
        // corrector already owns, not a policy seam. Every product derived
        // below -- demod caches, magnitude, attribution, clpbuffer[0] --
        // inherits truth from this one site.
        const float *certExRow =
            (certifiedOneDLevel() >= 1 && frameHasExactCoverage())
                ? exactCarrierRow(line)
                : nullptr;

        // Stage 1b -- comp-line HEAD phase snap (user, 2026-07-30: "move the
        // snap earlier... Frame B is equally important"). On comp lines of
        // covered frames the 1D scalar takes the certified bracket's phase
        // and keeps its own amplitude, HERE, so Field A/B, Frame A/B, the
        // CCR reads and the chroma demod all inherit it through the derived
        // products -- no candidate-local rotation anywhere. Same shared snap
        // the retraction ladder uses (amplitude floor 1 IRE, tau 3; the
        // reference's phase was validated >= 2 IRE, coherence 0.907). The
        // scalar-publish law is honoured: the snap composes waveforms in
        // the working space, no demod/remod round-trip through unlike bases.
        // Escape LDCD_CERT_PHASE=0.
        static const bool certHeadPhaseOn = []{
            const char *e = std::getenv("LDCD_CERT_PHASE");
            return !(e && std::atoi(e) == 0);
        }();
        if (certHeadPhaseOn && certifiedOneDLevel() >= 1 &&
            frameHasExactCoverage() && !certifiedDefLine(line) &&
            line - 1 >= first && line + 1 < last) {
            std::vector<double> legU(width), legD(width), hueAlpha(width);
            ldcdBuildCertBracketAligned(
                carrierGrammarLine(line),
                carrierGrammarLine(line - 1),
                carrierGrammarLine(line + 1),
                exactCarrierRow(line - 1), exactCarrierRow(line + 1),
                left, width, snapRef, &legU, &legD);
            std::copy(restrainedLine, restrainedLine + width, snapEst.begin());
            std::fill(snapOut.begin(), snapOut.end(),
                      std::numeric_limits<double>::quiet_NaN());
            ldcdApplyPhaseSnap(snapEst, snapRef, snapOut, width, irescale,
                               1.0, 3.0, false);
            // Hue-boundary verdict as a smoothed ALPHA, never a cut: leg
            // coherence ramp, lateral boxcar ~1 carrier cycle, then blend.
            // A binary keep here interleaved snapped/unsnapped carrier at
            // pixel pitch along chroma edges -- rendered checkerboard.
            ldcdSideCoherenceAlpha(legU, legD, width, hueAlpha);
            constexpr int kAr = 4;
            for (int xi = 0; xi < width; ++xi) {
                double acc = 0.0; int n = 0;
                for (int j = std::max(0, xi - kAr);
                     j <= std::min(width - 1, xi + kAr); ++j) {
                    acc += hueAlpha[j]; ++n;
                }
                const double al = acc / std::max(1, n);
                restrainedLine[xi] = al * snapOut[xi] +
                                     (1.0 - al) * snapEst[xi];
            }
        }

        for (int rel = 0; rel < width; ++rel) {
            const int h = left + rel;
            const int phase = carrierSampleClass(line, h);
            double source = restrainedLine[rel];
            if (certExRow && std::isfinite(certExRow[h]))
                source = (double)certExRow[h];

            const double i = source * lutI[phase];
            const double q = source * lutQ[phase];
            const double i4 = source * i4Scale[phase];
            const double q4 = source * q4Scale[phase];
            nativeI4[rel] = i4;
            nativeQ4[rel] = q4;

            if (demodI) demodI[rel] = static_cast<float>(i);
            if (demodQ) demodQ[rel] = static_cast<float>(q);
            if (demodI4) demodI4[rel] = static_cast<float>(i4);
            if (demodQ4) demodQ4[rel] = static_cast<float>(q4);
            lockedSource[rel] = source;
        }

        // Pass 3b: the pre-comb IQ authority is a separate pair of
        // integer-centred baseband products.  The symmetric three-sample
        // aperture cancels the alternating product image while keeping both
        // axes registered at the native sample h.  It is deliberately never
        // remodulated into lockedSource: raw-minus-carrier must continue to
        // consume the physical scalar above.
        for (int rel = 0; rel < width; ++rel) {
            const int rm = std::max(0, rel - 1);
            const int rp = std::min(width - 1, rel + 1);
            const double i4 = centeredCarrierProduct3(
                nativeI4[rm], nativeI4[rel], nativeI4[rp]);
            const double q4 = centeredCarrierProduct3(
                nativeQ4[rm], nativeQ4[rel], nativeQ4[rp]);

            if (lockedI4) lockedI4[rel] = static_cast<float>(i4);
            if (lockedQ4) lockedQ4[rel] = static_cast<float>(q4);

            const double chromaMagnitude = boundedMag(i4, q4);
            magnitude[rel] = static_cast<float>(chromaMagnitude);
            if (attribution) {
                AttributionFacts &facts = attribution[rel].facts;
                facts.locked1DChromaIRE =
                    chromaMagnitude * invIreScale;
            }
        }

        if (grammar)
            grammar->projectionValid = false;

        double *published = clpbuffer[0].pixel[line];
        for (int rel = 0; rel < width; ++rel)
            published[left + rel] = lockedSource[rel];
    }

    // Trailing pass: ±2 vertical partner evidence, computed once the IQ
    // caches above are complete for every line.  Two verdicts per pixel:
    //
    //   SAME  — two schedule-admitted carrier operands positively share this
    //           pixel's chroma region (relation-signed hue agreement above
    //           the chroma floor).  A real chroma-region boundary and
    //           cross-color both fail interline carrier verification, so
    //           the gA detector alone cannot tell them apart; this is the
    //           discriminator.  Consumed by the suppression alpha.
    //
    //   ALIEN — a relation-admitted operand is ANTI-aligned at comparable
    //           magnitude after relation signing: raw-identical content where
    //           the carrier grammar says the operands are phase-comparable.
    //           That is the comb's cancellation partner, not a hue boundary
    //           to cede to 1D (near-carrier periodic luma, the Borg-cube grid).
    //           Consumed by the produceY retracted admission.
    //
    // Both are evidence only; consumers convert.  Diagnostic gate:
    // LD_REGION_KEEP=0 disables the SAME verdict (suppression veto behaves
    // exactly as before 2026-07-02) without touching the ALIEN fact — the
    // single-variable isolation for witness-render regressions.
    static const bool regionKeepEnabled = []{
        const char *s = std::getenv("LD_REGION_KEEP");
        return !(s && s[0] == '0');
    }();

    if (!regionSamePartner_flat.empty() &&
        !regionAlienPartner_flat.empty() && demodWidth >= width) {
        const lddecode::CombReachSourceFrame iqSource = iqReachSource();

        struct RegionLegReach {
            bool allow = false;
            lddecode::CarrierPhaseRelation relation =
                lddecode::CarrierPhaseRelation::Unknown;
        };

        for (int line = first; line < last; ++line) {
            float *sameRow = regionSamePartner_line(line);
            float *alienRow = regionAlienPartner_line(line);
            if (!sameRow || !alienRow)
                continue;
            std::fill(sameRow, sameRow + width, 0.0f);
            std::fill(alienRow, alienRow + width, 0.0f);

            const float *i0 = locked1DTI4fsc_line(line);
            const float *q0 = locked1DTQ4fsc_line(line);
            if (!i0 || !q0)
                continue;

            auto legReach = [&](int target) -> RegionLegReach {
                RegionLegReach leg;
                if (target < first || target >= last)
                    return leg;
                const lddecode::CombReachReply reach = combReachIndex.query(
                    {line, target, left, left,
                     lddecode::CombReachUse::IQCompare, iqSource});
                if (!reach.allowIQCompare)
                    return leg;
                leg.allow =
                    reach.carrierRelation == lddecode::CarrierPhaseRelation::Same ||
                    reach.carrierRelation == lddecode::CarrierPhaseRelation::Opposite;
                leg.relation = reach.carrierRelation;
                return leg;
            };
            const RegionLegReach upReach = legReach(line - 2);
            const RegionLegReach dnReach = legReach(line + 2);
            const float *iUp = upReach.allow
                ? locked1DTI4fsc_line(line - 2) : nullptr;
            const float *qUp = upReach.allow
                ? locked1DTQ4fsc_line(line - 2) : nullptr;
            const float *iDn = dnReach.allow
                ? locked1DTI4fsc_line(line + 2) : nullptr;
            const float *qDn = dnReach.allow
                ? locked1DTQ4fsc_line(line + 2) : nullptr;
            if (!iUp && !iDn)
                continue;

            for (int rel = 0; rel < width; ++rel) {
                // Symmetric 7-tap horizontal aggregate, matching the region
                // evaluator in buildCombTapLine.  Its input is already the
                // full, integer-centred IQ vector; divide by the complete
                // weight (6), not by the old per-axis weight (3).
                auto fullIQ = [&](const float *iR, const float *qR) {
                    static constexpr double w[7] =
                        {0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5};
                    double si = 0.0;
                    double sq = 0.0;
                    for (int k = -3; k <= 3; ++k) {
                        const int rk = std::clamp(rel + k, 0, width - 1);
                        si += w[k + 3] * static_cast<double>(iR[rk]);
                        sq += w[k + 3] * static_cast<double>(qR[rk]);
                    }
                    return std::complex<double>(si / 6.0, sq / 6.0);
                };
                const std::complex<double> z0 = fullIQ(i0, q0);
                const std::complex<double> zUp =
                    iUp ? fullIQ(iUp, qUp) : std::complex<double>(0.0, 0.0);
                const std::complex<double> zDn =
                    iDn ? fullIQ(iDn, qDn) : std::complex<double>(0.0, 0.0);
                const auto region =
                    CombContentReach::evaluateIntrafieldRegionReach(
                        z0, zUp, zDn,
                        upReach.relation,
                        dnReach.relation,
                        upReach.allow && iUp,
                        dnReach.allow && iDn,
                        0.5, 0.5, 0.5,
                        invIreScale,
                        5.0);

                if (regionKeepEnabled &&
                    (region.up == CombContentReach::RegionRelation::SameRegion ||
                     region.down == CombContentReach::RegionRelation::SameRegion))
                {
                    sameRow[rel] = 1.0f;
                }
                if (region.up == CombContentReach::RegionRelation::AlienCancel ||
                    region.down == CombContentReach::RegionRelation::AlienCancel)
                {
                    alienRow[rel] = 1.0f;
                }
            }
        }
    }
}

void Comb::FrameBuffer::measurePostCombImpurity()
{
    if (!configuration.phaseCompensation) return;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;

    if (width <= 0 || firstLine >= lastLine) return;

    constexpr int    kNarrowWin = 16;
    constexpr int    kWideWin   = 32;
    constexpr double kImpurityFloorIRE = 2.0;

    static const double cosRef[4] = { 1.0, 0.0, -1.0, 0.0 };
    static const double sinRef[4] = { 0.0, 1.0,  0.0, -1.0 };

    auto clamp01 = [](double v) {
        return std::clamp(v, 0.0, 1.0);
    };

    const int srcBuf = std::clamp(static_cast<int>(configuration.dimensions) - 1, 0, 2);

    // Bandwidth-law cross-color contributor (opt-in, default off so the
    // authoritative impurity is byte-identical when disabled). Scores demod
    // envelope activity too fast to be legal carrier and MAXes it into gA; it
    // is a discriminator input for the color-side handler, never a source gain.
    static const bool bwCrossColor = []{
        const char *s = std::getenv("LDCD_BW_CROSSCOLOR");
        return s && std::atoi(s) != 0;
    }();

    // Coarse-residual carrier-legality veto (opt-in): where the aperture
    // parallax confirms legal carrier, pull gA down so the detector stops
    // desaturating compact colour it mistook for contamination. Default off so
    // the authoritative impurity is byte-identical when disabled.
    static const bool ccParallax = []{
        const char *s = std::getenv("LDCD_CC_PARALLAX");
        return s && std::atoi(s) != 0;
    }();

    std::vector<double> demI(width), demQ(width);
    std::vector<double> preI(width + 1), preQ(width + 1);
    std::vector<double> env(width), preEnv(width + 1);
    std::vector<double> lawI(width), lawQ(width);   // encoder-band envelope
    std::vector<double> pvI(size_t(4) * width), pvQ(size_t(4) * width);
    std::vector<double> psI(size_t(4) * width), psQ(size_t(4) * width);
    std::vector<double> parRatio(width);            // aperture parallax ratio

    for (int line = firstLine; line < lastLine; ++line) {
        const double *combLine = clpbuffer[srcBuf].pixel[line];
        float *impurityRow = carrierImpurity_line(line);
        if (!impurityRow) continue;

        for (int rel = 0; rel < width; ++rel) {
            const double s = combLine[left + rel];
            const int p = carrierSampleClass(line, left + rel) & 3;
            demI[rel] = s * cosRef[p];
            demQ[rel] = s * sinRef[p];
            const int relN = std::min(rel + 1, width - 1);
            const double sN = combLine[left + relN];
            env[rel] = std::sqrt(s * s + sN * sN);
        }

        preI[0] = 0.0;
        preQ[0] = 0.0;
        preEnv[0] = 0.0;
        for (int rel = 0; rel < width; ++rel) {
            preI[rel + 1] = preI[rel] + demI[rel];
            preQ[rel + 1] = preQ[rel] + demQ[rel];
            preEnv[rel + 1] = preEnv[rel] + env[rel];
        }

        // Lawful (<=1.3 MHz) envelope of the elected chroma: the most the
        // encoder could have modulated here. The demod's 2fSC image sits at
        // Nyquist, which this kernel nulls, so lawI/lawQ is the legal envelope
        // with the image already gone.
        if (bwCrossColor) {
            lddecode::projectExpressibleChromaEnvelope(
                demI.data(), nullptr, width, lawI.data());
            lddecode::projectExpressibleChromaEnvelope(
                demQ.data(), nullptr, width, lawQ.data());
        }

        if (ccParallax)
            apertureParallaxLine(line, pvI, pvQ, psI, psQ, parRatio.data());

        for (int rel = 0; rel < width; ++rel) {
            const double ZwI = centeredEvenWeightMean(
                demI.data(), preI.data(), width, rel, kWideWin);
            const double ZwQ = centeredEvenWeightMean(
                demQ.data(), preQ.data(), width, rel, kWideWin);

            // env[k] is centred at k+0.5, so this asymmetric index range is
            // the integer-centred physical aperture.
            const int na = std::clamp(rel - kNarrowWin / 2, 0, width);
            const int nb = std::clamp(na + kNarrowWin, 0, width);
            const double nn = std::max(1, nb - na);
            const double narrowMag =
                ((preEnv[nb] - preEnv[na]) / nn) * invIreScale;

            const double wideMag = 2.0 * boundedMag(ZwI, ZwQ) * invIreScale;

            double gA = 0.0;
            if (narrowMag > kImpurityFloorIRE && wideMag < narrowMag) {
                gA = clamp01(
                    (narrowMag - wideMag) /
                    std::max(kImpurityFloorIRE, narrowMag));
            }

            // Bandwidth-law contributor: the fraction of the elected chroma
            // envelope that a legal <=1.3 MHz envelope cannot reconstruct, i.e.
            // activity too fast to be carrier. imageFree = [1,2,1]/4 (stride 1)
            // removes the 2fSC image at Nyquist but keeps the 1.3 MHz-to-Nyquist
            // band; law = P{demod} keeps only the legal envelope; their
            // difference is the too-fast part. Floor-guarded so the soft 1.3 MHz
            // skirt and noise do not flag legal chroma edges. Scored, never
            // filtered -- MAXed into gA for the same color-side handler.
            if (bwCrossColor) {
                const int rm = std::max(0, rel - 1);
                const int rp = std::min(width - 1, rel + 1);
                const double ifI = 0.25 * (demI[rm] + 2.0 * demI[rel] + demI[rp]);
                const double ifQ = 0.25 * (demQ[rm] + 2.0 * demQ[rel] + demQ[rp]);
                const double crossMag = 2.0 *
                    std::hypot(ifI - lawI[rel], ifQ - lawQ[rel]) * invIreScale;
                const double actualMag =
                    2.0 * std::hypot(ifI, ifQ) * invIreScale;
                if (crossMag > kImpurityFloorIRE && actualMag > kImpurityFloorIRE)
                    gA = std::max(gA, clamp01(crossMag / actualMag));
            }

            // Carrier-legality veto: where the aperture parallax says this
            // carrier-band energy NULLS in every legal four-window (low ratio),
            // it is legal carrier -- including compact colour a wide coherent
            // window would smear -- so pull gA down. Where it fails to null
            // (high ratio), it is moving luma and gA passes. The ramp reuses the
            // corner-leak parallax thresholds (the same physical question).
            if (ccParallax) {
                const double keep = clamp01(
                    (parRatio[rel] - kCornerParallaxSoft) /
                    (kCornerParallaxHard - kCornerParallaxSoft));
                gA *= keep;
            }

            // The post-comb read is authoritative for CCR targeting. The
            // locked-1D seed published earlier is only a provisional read used
            // before the elected comb exists; once we have the elected result,
            // stale pre-comb suspicion must not linger and suppress solved
            // pixels.
            impurityRow[rel] = static_cast<float>(gA);
        }
    }
}

// RETAINED SCALES from two removed cross-colour ramps (2026-08-05). The
// constants themselves went with the CCREF read table, their last consumer;
// the MEASUREMENTS that set them are the part worth keeping, because anything
// rebuilt on these axes has to land in the same brackets.
//
// Vertical-image-detail corroboration ramp, shared by the coherent
// (splitIQlocked) and residual (filterIQLocked) transfer policies. Input was
// the 1D vertical-contrast service (|smooth[rel+2]-smooth[rel-2]|, IRE):
// below soft the coarse field is laterally flat and the edge read is silent;
// hard matched the established FIELD_LUMA_EDGE scale.
//     soft 6.0 IRE   hard 18.0 IRE   (18 = a solid vertical stroke)
//
// Concert gate: carrier-free confirmation that the notch edge is a real luma
// transition and not saturated-carrier leak.  The notch (edgeRamp) localizes
// the corner sharply but doubles residual carrier in saturated colour;
// maxAbsMembershipIRE (the same-carrier-phase membership change = lurch) reads
// luma movement through the aperture with the carrier cancelled, so it is dark
// in a saturated-chroma interior and bright at a genuine luma edge.  MEASURED
// on the beach strap (frame 52100, line 150): smooth/interior lurch 0.3-1.4,
// true skin<->strap edges 5-10.
//     soft 1.0 IRE   hard 4.0 IRE    -- chosen to bracket that gap

// Carrier-presence floor for the return. Numerically the aperture read's own
// kImpurityFloorIRE (buildPhaseCorrected1D / measurePostCombImpurity) -- the
// amplitude below which carrier-band energy is not a measurement. Named here
// so the composite verdict can obey the same law the aperture read already
// obeys internally.
static constexpr double kCcCarrierFloorIRE = 2.0;

// Committed-verdict cutoff (2026-08-02): the measured disagreement between
// the elected carrier and the anchored (anticipation-corrected) carrier, in
// IRE, below which the return bails and above which it commits fully.
// Confidence never sets a level; this is the single bail threshold. Sits at
// the noise-pedestal floor -- below kCcCarrierFloorIRE the "disagreement" is
// the capture's own noise, not confiscated luma (CCREF, 2026-07-30).
// LDCD_CC_CUTOFF overrides for A/B.
static constexpr double kCcCommitCutoffIRE = kCcCarrierFloorIRE;

// Fallback witness thresholds where no anchored plane exists (--dg-discard,
// non-cadence sources). gA's referee-endorsed regime starts at 0.40 (CCREF:
// anti-informative 0.2-0.4, superb above); regionKeep past 0.5 is a
// positively same-region vertical partner = a colour boundary, stand down.
// Binary predicates, never scales.
static constexpr double kCcFallbackGALo = 0.40;
static constexpr double kCcFallbackRegionKeepMax = 0.5;

void Comb::FrameBuffer::splitIQlocked(const FrameBuffer *prevF,
                                      const FrameBuffer *nextF)
{
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;
    const auto &T       = configuration.tunables;
    const int srcBuf    = std::clamp(static_cast<int>(configuration.dimensions) - 1, 0, 2);

    if (width <= 0 || firstLine >= lastLine) return;

    // Star classification is a split decision: establish the licensed
    // raw-domain footprint before publishing any elected carrier.  The same
    // immutable footprint is consumed by produceY and the final IQ renderer.
    buildStarFootprint(prevF, nextF);

    if (!basisLockedInit) {
        double Ce = 1.0, Se = 0.0;
        basisCoeffs(Ce, Se);
        for (int i = 0; i < 4; ++i) {
            double sp = 0.0, cp = 0.0;
            shiftedBasis(i, Ce, Se, sp, cp);
            spLUT_locked[i] = sp;
            cpLUT_locked[i] = cp;
        }
        basisLockedInit = true;
    }

    auto ensureScratch = [&](std::vector<double> &v) {
        if ((int)v.size() < width)
            v.resize(width, 0.0);
    };

    // These are consumed later by the locked IQ filter path.
    ensureScratch(scratch_preI);
    ensureScratch(scratch_preQ);

    auto finiteOrZero = [](double v) -> double {
        return std::isfinite(v) ? v : 0.0;
    };

    const double ccWeight = std::max(0.0, T.CC_SUPPRESSION_WEIGHT);
    const double giProduct = configuration.gi_product;
    const double gqProduct = configuration.gq_product;

    // Chroma-facts switch (escape LDCD_CHROMA_FACTS=0): on covered frames
    // the rendered products demodulate the anchored fact-family carrier
    // and the suppression envelope retires -- frame-uniform, so both
    // parities and every line move together.
    static const bool chromaFactsOn = []{
        const char *e = std::getenv("LDCD_CHROMA_FACTS");
        return !(e && std::atoi(e) == 0);
    }();

    // Retained record from the removed LDCD_PROBE_XFER census
    //
    // Cross-frame transfer referee: held-out grading of the uncovered-frame
    // successor BEFORE it is built. Snapshot each covered frame's exact
    // plane; when the next covered frame arrives (pitch 2 -- twice the real
    // transfer pitch, so every number is a pessimistic bound), grade three
    // transfer grades per 32x128 region, exact against exact:
    //   raw  -- copy the neighbour's carrier verbatim
    //   rot  -- best per-region phase rotation (what the sync tone buys)
    //   full -- best gain + rotation (2-parameter LS)
    // Ledger: rms residual in IRE per grade, region percentiles, and the
    // transferable fraction (regions whose best-grade residual < 1 IRE).
    // A pitch-4 same-parity comparison ran first (same line, no vertical
    // contamination) to separate the chat noise floor from real content
    // drift in the static regions. Off-parity rows were graded by the
    // rotation-invariant grades only -- rot/full absorb the 180-degree row
    // flip -- so `raw` was same-line only.
    //
    // LUMA TWEEN REFEREE ([LTWEEN]): grade the tween witness for the
    // uncovered-frame return. Certified luma (the retracted plane -- full
    // frame on covered frames: exact on covered lines, certified-comb on
    // comp lines) is snapshotted for the last three covered frames A,B,C;
    // tween B' = (A+C)/2 and grade against B's own certified luma. Pitch 4
    // end-to-end vs the real mechanism's pitch 2, so every number is a
    // pessimistic bound. Ledger split by band: the witness's duty is the
    // CARRIER BAND (bp = [-.25,.5,-.25] at lag 2), where trapped luma
    // lives. hit = sites |bpB| >= 2 IRE (true luma-in-band, the
    // cross-colour candidates). Precision side: when the tween witness
    // FIRES (claims trapped luma), is it right? A false fire would return
    // carrier into Y -- the one failure mode that manufactures checker.
    //
    // FLOW-TWEEN REFEREE ([LTWEEN2]): grade TRUE tweening against the
    // averaged form above (user, 2026-07-31: "The luma can't be averaged;
    // that isn't tweening. Tweening... is creating in-between animations of
    // the boundary vectors -- think of cheap slow motion that averages the
    // frame vs optical flow"). Per 32x64 block, find the symmetric
    // half-vector d minimizing SAD between A(-d) and C(+d); the midpoint is
    // B'(x) = 0.5*(A(x-d) + C(x+d)) -- boundaries land at their interpolated
    // POSITIONS instead of superimposing as a double exposure. Same grades
    // as LTWEEN for direct comparison.
    //
    // Zero-motion preference, the law that form needed: noise happily buys
    // a random 1-px "improvement", and 1 px of spurious flow decorrelates
    // the 3-px carrier band that the witness lives in. Motion must EARN its
    // claim -- keep d=0 unless the best displacement beats stillness by 20%
    // and 0.5 IRE.

    // Retained record from the removed LDCD_PROBE_CCDEV census
    //
    // Disposable dev-envelope census (LDCD_PROBE_CCDEV=1, engaged runs
    // only): one atomic line per frame -- the committed detector's
    // measured-disagreement distribution, to adjudicate the self-blinding
    // hypothesis (quiet frames: anchored tracks the same alias the
    // elected carrier carries, so dev sits under the cutoff with the
    // cross-colour still standing).

    for (int line = firstLine; line < lastLine; ++line) {
        const double *src = clpbuffer[srcBuf].pixel[line];

        const CombCarrierGrammar *grammar = carrierGrammarLine(line);
        const bool grammarLocked = grammar && grammar->grammarLocked;

        const double bcos = grammarLocked ? grammar->burstCos : 1.0;
        const double bsin = grammarLocked ? grammar->burstSin : 0.0;

        double lutTi[4];
        double lutTq[4];

        if (grammarLocked) {
            for (int i = 0; i < 4; ++i) {
                lutTi[i] = finiteOrZero((double)grammar->demodLUTTi[i]);
                lutTq[i] = finiteOrZero((double)grammar->demodLUTTq[i]);
            }
        } else {
            fusedDemodLUT(bcos, bsin, spLUT_locked, cpLUT_locked, lutTi, lutTq);
            for (int i = 0; i < 4; ++i) {
                lutTi[i] = finiteOrZero(lutTi[i]);
                lutTq[i] = finiteOrZero(lutTq[i]);
            }
        }

        float  *tiRow       = demodTI_line(line);
        float  *tqRow       = demodTQ_line(line);
        float  *ti4Row      = demodTI4fsc_line(line);
        float  *tq4Row      = demodTQ4fsc_line(line);
        float  *prodIRow    = lockedProductI_line(line);
        float  *prodQRow    = lockedProductQ_line(line);
        float  *maskRawRow  = lockedCcMaskRaw_line(line);
        double *carrierComp = lockedCarrierComposite_line(line);

        const float *impRow = carrierImpurity_line(line);
        const float *sameRegionRow = regionSamePartner_line(line);
        // NOTE (renderer fact, 2026-07-31): a fact-family demod of the
        // RENDERED product planes was built here and measured inert --
        // filterIQLocked renders colour as demod(raw - Y), the exact
        // complement of emitted Y, and never reads those planes. The
        // covered-frame chroma win is carried by the Y facts plus the
        // suppression retirement (ccRetired, pass 2). Removed; do not
        // rebuild a product-side chroma path.
        //
        // A carrier-valued CCR reference exists only on covered frames.  An
        // uncovered carrier fit remains evidence, never a replacement value;
        // iceberg reaches CCR separately as recovered luma.
        const double *ccFactRow = factBackedCarrier_line(line);
        const double *ccAnchRow = ccFactRow;
        // Detector reference pair (2026-08-02, round 2): the OBSERVATION
        // side is the untouched locked 1D -- the safe retreat, which the
        // construction law guarantees no comb/echo signal ever enters --
        // never the elected carrier. Measured failure that forced this:
        // the tap fact-injection pulled the elected carrier toward the
        // anchored story, collapsing |elected - anchored| and blinding the
        // committed verdict while the alias-conforming cross-colour (which
        // combing cannot cancel) still stood: the strong frames' -53%
        // chroma fell to -4%. 1D-vs-anchored is immune to every future
        // comb change: alias and stolen luma live in the 1D and not in the
        // anchored model (fires); legal chroma lives identically in both
        // (bails).
        const double *ccObs1D = locked1DSource_line(line);

        if (prodIRow)    std::fill(prodIRow, prodIRow + width, 0.0f);
        if (prodQRow)    std::fill(prodQRow, prodQRow + width, 0.0f);
        if (carrierComp) std::fill(carrierComp, carrierComp + width, 0.0);

        for (int xi = 0; xi < width; ++xi) {
            const int h  = left + xi;
            const int ph = carrierSampleClass(line, h);

            // Plain locked demod at the native composite coordinate h.  Keep I
            // and Q as independent product streams: their common output
            // centroid is established later by the centered axis-specific
            // FIRs, not by averaging adjacent demod products here.  Such an
            // average is an extra half-sample filter on the demodulator and,
            // once remodulated, gives produceY a different carrier geometry.
            //
            // No line affine, local affine, or sliding-window carrier fit.
            // This function may suppress transfer, but it must not reshape the
            // carrier that produceY subtracts.
            const double ti = finiteOrZero(src[h] * lutTi[ph]);
            const double tq = finiteOrZero(src[h] * lutTq[ph]);

            double ti4 = 0.0;
            double tq4 = 0.0;
            lockedTo4fsc(ti, tq, bcos, bsin, ti4, tq4);

            ti4 = finiteOrZero(ti4);
            tq4 = finiteOrZero(tq4);

            if (tiRow)  tiRow[xi]  = (float)ti;
            if (tqRow)  tqRow[xi]  = (float)tq;
            if (ti4Row) ti4Row[xi] = (float)ti4;
            if (tq4Row) tq4Row[xi] = (float)tq4;

            // Critical geometry rule:
            // src is already the elected scalar carrier in physical composite
            // geometry.  It is the subtraction authority; the demodulated
            // products above are colour/evidence products, not a reason to
            // synthesize a second, numerically different scalar.  This also
            // makes the invariant exact when the locked basis or calibration
            // trim changes.
            if (carrierComp)
                carrierComp[xi] = finiteOrZero(src[h]);

            // Suppression policy below only runs when --cross-color-return
            // is engaged (the mask buffers exist); the default path pays
            // nothing.
            if (!maskRawRow) {
                const float prodI = (float)finiteOrZero(ti * giProduct);
                const float prodQ = (float)finiteOrZero(tq * gqProduct);
                if (prodIRow) prodIRow[xi] = prodI;
                if (prodQRow) prodQRow[xi] = prodQ;
                scratch_preI[xi] = prodI;
                scratch_preQ[xi] = prodQ;
                continue;
            }

            const double gA =
                impRow ? std::clamp((double)impRow[xi], 0.0, 1.0) : 0.0;

            // Named detector-to-policy conversion: a positively same-region
            // ±2 partner means this chroma continues vertically — a color
            // boundary, not cross-color — so the suppression stands down.
            const double regionKeep = sameRegionRow
                ? std::clamp((double)sameRegionRow[xi], 0.0, 1.0)
                : 0.0;

            // COMMITTED VERDICT (user law, 2026-08-02). Evidence confidence
            // cannot be expected to correlate with truth, so confidence
            // never sets a level: one cutoff below which the operation
            // bails, full commit above it. A committed operation returns a
            // clear result pro or con; a hedged one shelters the defect it
            // should expose. The evidence-ceiling cap and both analog
            // reads' level-setting retire together (research phase: data
            // over shelter). RETGRADE conviction that forced this: 77.5%
            // of the un-recovered theft budget sat where the old mask
            // idled below its own scaling (quiet/partial flat sites), and
            // gA is anti-informative in exactly that regime (CCREF bins
            // 0.2-0.4).
            //
            // Where the anchored plane exists the verdict is a MEASUREMENT:
            // the elected carrier's disagreement with the anticipation-
            // corrected carrier, in IRE. The gA aperture read remains the
            // witness only where no anchored plane exists (--dg-discard,
            // non-cadence sources: the chain-link fence regime it was
            // built for), applied as a binary predicate, never a scale.
            static const double ccCommitCutoffIRE = []{
                const char *s = std::getenv("LDCD_CC_CUTOFF");
                return s ? std::atof(s) : kCcCommitCutoffIRE;
            }();
            double lumaWeight = 0.0;
            double devIRE = -1.0;
            if (ccAnchRow && ccObs1D) {
                // |delta| oscillates at the carrier; its envelope is the
                // max over ±2 samples (1.25 cycles at 4fSC).
                const int am = std::max(0, xi - 2);
                const int ap = std::min(width - 1, xi + 2);
                double devPeak = 0.0;
                for (int k = am; k <= ap; ++k)
                    devPeak = std::max(devPeak,
                        std::fabs(finiteOrZero(ccObs1D[k]) -
                                  ccAnchRow[k]));
                devIRE = devPeak * invIreScale;
                if (devIRE >= ccCommitCutoffIRE)
                    lumaWeight = std::min(ccWeight, 1.0);
            } else if (gA >= kCcFallbackGALo &&
                       regionKeep < kCcFallbackRegionKeepMax) {
                lumaWeight = std::min(ccWeight, 1.0);
            }

            // The detector's own verdict, stashed for the off-lattice
            // audit BEFORE any fact overrides it.
            if (float *dv = ccDetectorVerdict_line(line))
                dv[xi] = (float)lumaWeight;

            // VERDICT IS THE FACT on covered samples: the false fraction of
            // the elected carrier is |C - ex| / |C| exactly. The detector is
            // not consulted where conservation already answered -- this is
            // what makes the turquoise/checker class impossible on the
            // lattice by construction. Denominator floored at 1 IRE (below
            // it the ratio is noise-on-noise; the fact verdict there is
            // simply "nothing worth returning").
            // NOTE (facts v4): per-sample fact SUPPRESSION was tried in two
            // forms (vector error, then amplitude excess) and both injected
            // luma-detail-rate structure into an envelope designed for
            // band-limited verdicts (+7.5% / +18.5% covered chroma
            // alternation). There is no separate chroma side to fix:
            // colour renders as demod(raw - Y), so chroma follows Y
            // (renderer fact, 2026-07-31). Facts reach the mask only
            // through the regional audit below; the Y side consumes the
            // fact as a VALUE in produceY.

            // Noise-pedestal restraint is inherited, not re-gated: the
            // commit cutoff sits at the pedestal floor (kCcCarrierFloorIRE
            // -- below it C and ex share the capture's noise, CCREF
            // 2026-07-30), so the committed verdict cannot fire on the
            // dark-noise class the old LDCD_CC_AMPGATE experiment guarded.

            // The verdict is NOT applied here.  Applied per-sample it carries
            // regionKeep's hard flips and gA's ring chatter at pixel pitch --
            // amplitude modulation that beats sidebands back into the chroma
            // passband and shreds both sides of a hue boundary (the residual
            // path documented this failure mode and band-limits; the coherent
            // path must too).  Pass 2 below smooths it into an envelope that
            // varies no faster than the chroma it gates, then applies.
            if (maskRawRow)
                maskRawRow[xi] = (float)lumaWeight;

            const float prodI = (float)finiteOrZero(ti * giProduct);
            const float prodQ = (float)finiteOrZero(tq * gqProduct);

            if (prodIRow) prodIRow[xi] = prodI;
            if (prodQRow) prodQRow[xi] = prodQ;

            scratch_preI[xi] = prodI;
            scratch_preQ[xi] = prodQ;
        }
    }

    // Pass 2 (engaged when cross-color return is nonzero): band-limit the
    // suppression verdict into an envelope, then scale the published chroma
    // products.  In-field vertical mix (±2 lines, the same-parity partners
    // the verdict itself was judged against) then a lateral boxcar with a
    // radius of about one carrier cycle -- the same construction the
    // residual path uses, so suppression cannot alias in either renderer.
    if (ccWeight > 0.0 && !lockedCcMaskRaw_flat.empty() &&
        !lockedCcMask_flat.empty()) {
        constexpr int kCcMaskRadius = 4; // 9-tap, first null ~1.6 MHz

        // OFF-LATTICE FACT AUDIT (detectors-trade-for-truth, piece 3): per
        // region, the nearest covered facts grade the detector's claims --
        // auditW = (return truth honoured) / (return claimed) over fired
        // covered samples with real carrier, pooled from whichever of
        // {self, prevF, nextF} carries coverage. The lattice is one parity
        // of every other frame, so no estimate is ever more than a frame
        // from its audit. A weight from fact-agreement, never a coefficient
        // on a guess; regions without evidence stay neutral (the referee's
        // global verdict is that the return helps).
        static const bool ccFactsOn2 = []{
            const char *e = std::getenv("LDCD_CC_FACTS");
            return !(e && std::atoi(e) == 0);
        }();
        const int anx = (width + 127) / 128;
        const int any_ = (lastLine + 31) / 32;
        std::vector<double> auditW((size_t)anx * any_, 1.0);
        // SUPPRESSION-DUTY verdict (the regional court): same fact
        // grading, FAIL-CLOSED default 0. It answers one question --
        // where the detector's luma claim CONFLICTS with asserted legal
        // chroma, do the nearest facts endorse the detector? -- and a
        // region without evidence answers "no": the conflicted portion
        // of the mask abstains rather than act on an unadjudicated
        // claim. (The Y-return's auditW keeps its neutral default; that
        // duty has no opposing claim to adjudicate.)
        std::vector<double> auditSupp((size_t)anx * any_, 0.0);
        if (ccFactsOn2) {
            std::vector<double> claimed((size_t)anx * any_, 0.0),
                honoured((size_t)anx * any_, 0.0);
            // Sources: this frame if covered, else the previous covered
            // neighbour (its detector plane is complete; the NEXT frame's
            // splitIQlocked has not run yet, so it cannot serve). The
            // graded quantity is the DETECTOR plane, never the published
            // mask — on covered lines the mask is the fact and a fact
            // grading itself reads 1.0 by construction.
            const FrameBuffer *auditSrc =
                frameHasExactCoverage() ? this
                : (prevF && prevF->frameHasExactCoverage()) ? prevF
                : nullptr;
            if (auditSrc) {
                const FrameBuffer *fb = auditSrc;
                for (int line = firstLine; line < lastLine; ++line) {
                    const float *ex = fb->exactCarrierRow(line);
                    if (!ex) continue;
                    const double *cc =
                        fb->lockedCarrierComposite_line(line);
                    const float *dv = fb->ccDetectorVerdict_line(line);
                    if (!cc || !dv) continue;
                    const size_t rb = (size_t)(line / 32) * anx;
                    for (int xi = 0; xi < width; ++xi) {
                        const int h = left + xi;
                        if (!std::isfinite(ex[h])) continue;
                        const double C = cc[xi];
                        if (!std::isfinite(C) ||
                            std::fabs(C) < 2.0 * irescale) continue;
                        const double m = std::clamp((double)dv[xi], 0.0, 1.0);
                        if (m <= 0.02) continue;
                        const double claim = m * std::fabs(C);
                        const double leakAbs =
                            std::fabs(C - (double)ex[h]);
                        const size_t r = rb + xi / 128;
                        claimed[r] += claim;
                        honoured[r] += std::min(claim, leakAbs);
                    }
                }
            }
            for (size_t r = 0; r < auditW.size(); ++r)
                if (claimed[r] > 8.0 * irescale) {
                    auditW[r] = std::clamp(honoured[r] / claimed[r],
                                           0.0, 1.0);
                    auditSupp[r] = auditW[r];
                }
        }
        // Publish for produceY's Y-return consumption.
        ccAuditW_flat.assign(auditW.begin(), auditW.end());
        ccAuditNX = anx; ccAuditNY = any_;
        auto auditAt = [&](int line, int xi) {
            // Bilinear over region centers: the audit scales authority as a
            // smooth field, never a per-sample cut.
            const double ry = std::clamp((line - 16.0) / 32.0, 0.0,
                                         (double)(any_ - 1) - 1e-6);
            const double rx = std::clamp((xi - 64.0) / 128.0, 0.0,
                                         (double)(anx - 1) - 1e-6);
            const int r0 = (int)ry, c0 = (int)rx;
            const double wy = ry - r0, wx = rx - c0;
            const int r1 = std::min(r0 + 1, any_ - 1);
            const int c1 = std::min(c0 + 1, anx - 1);
            return (1 - wy) * ((1 - wx) * auditW[(size_t)r0 * anx + c0] +
                               wx * auditW[(size_t)r0 * anx + c1]) +
                   wy * ((1 - wx) * auditW[(size_t)r1 * anx + c0] +
                         wx * auditW[(size_t)r1 * anx + c1]);
        };
        auto auditSuppAt = [&](int line, int xi) {
            const double ry = std::clamp((line - 16.0) / 32.0, 0.0,
                                         (double)(any_ - 1) - 1e-6);
            const double rx = std::clamp((xi - 64.0) / 128.0, 0.0,
                                         (double)(anx - 1) - 1e-6);
            const int r0 = (int)ry, c0 = (int)rx;
            const double wy = ry - r0, wx = rx - c0;
            const int r1 = std::min(r0 + 1, any_ - 1);
            const int c1 = std::min(c0 + 1, anx - 1);
            return (1 - wy) * ((1 - wx) * auditSupp[(size_t)r0 * anx + c0] +
                               wx * auditSupp[(size_t)r0 * anx + c1]) +
                   wy * ((1 - wx) * auditSupp[(size_t)r1 * anx + c0] +
                         wx * auditSupp[(size_t)r1 * anx + c1]);
        };
        std::vector<double> vmix(width, 0.0);

        // ---- Vertical envelope stage (EXPERIMENT: LDCD_CC_VMIX) ----
        // The published mask is APPLIED on the frame raster, so its envelope
        // has to be smooth on the frame raster. The shipped mix reaches only
        // ±2 -- the same-parity partners the verdict was judged against --
        // which leaves the mask smoothed along the field axis and raw along
        // the axis it acts on. CCREF measured the consequence: mask
        // D2frame/D2field 1.18-1.27 against a |C| control at 0.73-0.82, i.e.
        // the nearer neighbour disagreeing MORE than the farther one.
        //
        //   0 field  ±2 only, [.25 .5 .25]                 (shipped baseline)
        //   1 frame  ±1 only, [.25 .5 .25]
        //   2 both   5-tap over l-2..l+2
        //   3 stage  ±2 in-field consensus, THEN ±1 across parity
        //
        // Mode 3 is the "move both together" shape: the verdict goes on being
        // formed from its own in-field evidence, and only the ACTION is made
        // parity-symmetric afterwards. This selector is experiment
        // scaffolding -- whatever wins becomes the unconditional kernel.
        static const int ccVMix = []{
            const char *s = std::getenv("LDCD_CC_VMIX");
            return s ? std::atoi(s) : 0;
        }();
        const size_t vpN = static_cast<size_t>(lastLine) * width;
        std::vector<double> ccVPlane(vpN, 0.0);
        {
            auto stage = [&](auto rowAt, int step, std::vector<double> &dst) {
                for (int l = firstLine; l < lastLine; ++l) {
                    const auto *c = rowAt(l);
                    if (!c) continue;
                    const auto *u = rowAt(l - step);
                    const auto *d = rowAt(l + step);
                    const double norm =
                        0.5 + (u ? 0.25 : 0.0) + (d ? 0.25 : 0.0);
                    double *o = dst.data() + static_cast<size_t>(l) * width;
                    for (int xi = 0; xi < width; ++xi)
                        o[xi] = (0.5 * c[xi] + (u ? 0.25 * u[xi] : 0.0) +
                                 (d ? 0.25 * d[xi] : 0.0)) / norm;
                }
            };
            auto rawRow = [&](int l) -> const float * {
                return (l >= firstLine && l < lastLine)
                    ? lockedCcMaskRaw_line(l) : nullptr;
            };
            if (ccVMix == 1) {
                stage(rawRow, 1, ccVPlane);
            } else if (ccVMix == 2) {
                for (int l = firstLine; l < lastLine; ++l) {
                    const float *c = rawRow(l);
                    if (!c) continue;
                    const float *u1 = rawRow(l - 1), *d1 = rawRow(l + 1);
                    const float *u2 = rawRow(l - 2), *d2 = rawRow(l + 2);
                    const double norm = 0.4 + (u1 ? 0.2 : 0.0)
                                            + (d1 ? 0.2 : 0.0)
                                            + (u2 ? 0.1 : 0.0)
                                            + (d2 ? 0.1 : 0.0);
                    double *o = ccVPlane.data() + static_cast<size_t>(l) * width;
                    for (int xi = 0; xi < width; ++xi)
                        o[xi] = ((0.4 * c[xi] +
                             (u1 ? 0.2 * u1[xi] : 0.0) +
                             (d1 ? 0.2 * d1[xi] : 0.0) +
                             (u2 ? 0.1 * u2[xi] : 0.0) +
                             (d2 ? 0.1 * d2[xi] : 0.0)) / norm);
                }
            } else if (ccVMix == 3) {
                std::vector<double> tmp(vpN, 0.0);
                stage(rawRow, 2, tmp);
                auto tmpRow = [&](int l) -> const double * {
                    return (l >= firstLine && l < lastLine)
                        ? tmp.data() + static_cast<size_t>(l) * width
                        : nullptr;
                };
                stage(tmpRow, 1, ccVPlane);
            } else {
                stage(rawRow, 2, ccVPlane);
            }
        }

        // SUPPRESSION RETIRES ON COVERED FRAMES (chroma-facts): the
        // rendered products are demodulated from the fact-family carrier,
        // so the false colour this envelope exists to suppress is excluded
        // at construction -- scaling those products by a fit-vague
        // envelope could only desaturate real chroma (the bikini ring).
        // Frame-uniform: the published mask reads zero on every line, so
        // produceY's estimator return stands down with it and neither
        // parity is treated alone. The detector verdict planes and the
        // regional audit above keep running: covered frames remain the
        // audit source that grades the uncovered frames' firings.
        // True coverage test, not the anchored-plane proxy: the plane now
        // publishes on uncovered frames too after two-sided refinement, and
        // the retirement premise -- false colour excluded at construction --
        // holds only where construction consumed same-frame FACTS.
        const bool ccRetired =
            chromaFactsOn && frameHasExactCoverage();

        for (int line = firstLine; line < lastLine; ++line) {
            const float *r0 = lockedCcMaskRaw_line(line);
            float *out = lockedCcMask_line(line);
            const float *exRow2 = exactCarrierRow(line);
            float *prodIRow = lockedProductI_line(line);
            float *prodQRow = lockedProductQ_line(line);
            if (!r0 || !out)
                continue;
            if (ccRetired) {
                std::fill(out, out + width, 0.0f);
                continue;
            }

            const AttributionEvidence *attributionRow =
                attributionEvidence_line(line);
            const float *impurityRow = carrierImpurity_line(line);
            const float *sameRegionRow = regionSamePartner_line(line);
            const lddecode::CarrierAnalysisRecord *analysisRow =
                carrierAnalysis_line(line);
            const lddecode::CarrierAnalysisRecord *analysisUpRow =
                (line - 2 >= firstLine) ? carrierAnalysis_line(line - 2)
                                        : nullptr;
            const lddecode::CarrierAnalysisRecord *analysisDnRow =
                (line + 2 < lastLine) ? carrierAnalysis_line(line + 2)
                                      : nullptr;

            // Delta-value return may rebuild products from certified carrier
            // only.  Uncovered returns act in Y and retain raw-Y complement
            // chroma; no estimated carrier is demodulated here.
            const double *factProdRow = factBackedCarrier_line(line);
            const double *anchProdRow = factProdRow;
            double lutTi2[4] = {0, 0, 0, 0}, lutTq2[4] = {0, 0, 0, 0};
            const double giProduct2 = configuration.gi_product;
            const double gqProduct2 = configuration.gq_product;
            if (anchProdRow) {
                const CombCarrierGrammar *g2 = carrierGrammarLine(line);
                if (g2 && g2->grammarLocked) {
                    for (int i = 0; i < 4; ++i) {
                        lutTi2[i] = finiteOrZero((double)g2->demodLUTTi[i]);
                        lutTq2[i] = finiteOrZero((double)g2->demodLUTTq[i]);
                    }
                } else {
                    const double bc2 = g2 && g2->grammarLocked
                        ? g2->burstCos : 1.0;
                    const double bs2 = g2 && g2->grammarLocked
                        ? g2->burstSin : 0.0;
                    fusedDemodLUT(bc2, bs2, spLUT_locked, cpLUT_locked,
                                  lutTi2, lutTq2);
                    for (int i = 0; i < 4; ++i) {
                        lutTi2[i] = finiteOrZero(lutTi2[i]);
                        lutTq2[i] = finiteOrZero(lutTq2[i]);
                    }
                }
            }

            auto grammarPassAt = [&](int xi) {
                double legal = analysisRow
                    ? lddecode::carrierLegalProof(
                          (double)analysisRow[xi].carrierConformance,
                          (double)analysisRow[xi].conformanceSupportFraction)
                    : 0.0;
                legal = std::max(legal, std::max(
                    analysisUpRow
                        ? lddecode::carrierLegalProof(
                              (double)analysisUpRow[xi].carrierConformance,
                              (double)analysisUpRow[xi]
                                  .conformanceSupportFraction)
                        : 0.0,
                    analysisDnRow
                        ? lddecode::carrierLegalProof(
                              (double)analysisDnRow[xi].carrierConformance,
                              (double)analysisDnRow[xi]
                                  .conformanceSupportFraction)
                        : 0.0));
                return std::clamp(legal, 0.0, 1.0);
            };

            const double *vRow =
                ccVPlane.data() + static_cast<size_t>(line) * width;
            for (int xi = 0; xi < width; ++xi) {
                const double verticalMean = vRow[xi];
                const double impulse = attributionRow
                    ? std::clamp(
                          attributionRow[xi].facts.lumaImpulseRisk, 0.0, 1.0)
                    : 0.0;

                // A moving compact luma impulse is not expected at the same
                // column on its +/-2 partners.  Preserve its own measured raw
                // cross-colour evidence in proportion to the shared impulse
                // geometry instead of allowing the vertical envelope to halve
                // it.  This cannot invent suppression: it only raises the mix
                // toward r0 when r0 is already the stronger measurement.
                //
                // The one-sided max() encodes the FIELD kernel's geometry --
                // it assumes the neighbours never carry the impulse, so the
                // envelope can only ever be too LOW at an impulse site. Under
                // a frame-pitch kernel that premise inverts: a compact impulse
                // does land on the ±1 partners, verticalMean is raised at the
                // star, r0 - verticalMean goes negative, the max clamps the
                // correction away, and the mask sits high on the star itself
                // -- returning carrier-band energy into the impulse and
                // spreading it. Measured: star falloff +6.4% (attack shot),
                // +8.2% (cube), i.e. visibly less crisp.
                //
                // LDCD_CC_IMPULSE=1 holds the site at its OWN line's verdict
                // from either direction, so the term stops depending on which
                // neighbours happen to share the impulse. At impulse == 1 the
                // mask is exactly r0 under any kernel.
                // LDCD_CC_IMPULSE=2 removes the impulse term outright. This is
                // a MECHANISM PROBE, not a candidate: the term exists to keep
                // chroma suppression alive at stars, so dropping it trades
                // false colour for impulse crispness. It answers one question
                // -- is the impulse boost what the luma return is riding into
                // the star? -- and if so the fix is to split the mask's two
                // duties, not to delete the term.
                static const int ccImpulseMode = []{
                    const char *s = std::getenv("LDCD_CC_IMPULSE");
                    return s ? std::atoi(s) : 0;
                }();
                vmix[xi] =
                    ccImpulseMode == 2
                        ? verticalMean
                        : (ccImpulseMode == 1
                               ? verticalMean +
                                     impulse * ((double)r0[xi] - verticalMean)
                               : verticalMean +
                                     impulse * std::max(
                                         0.0,
                                         (double)r0[xi] - verticalMean));
            }

            // L4 aggregation preserves the authority of a narrow convicted
            // line without adding a second star/impulse detector. A one-pixel
            // verdict in this nine-tap aperture collapsed to 1/9 under a mean
            // and to 1/3 under RMS; L4 retains 9^(-1/4) ~= 0.58. The result is
            // still a smooth neighbourhood envelope, never exceeds the largest
            // measured verdict in its aperture, and leaves broad verdicts
            // unchanged.
            double sumFourthPowers = 0.0;
            int lo = 0, hi = -1;
            for (int xi = 0; xi < width; ++xi) {
                const int nlo = std::max(0, xi - kCcMaskRadius);
                const int nhi = std::min(width - 1, xi + kCcMaskRadius);
                while (hi < nhi) {
                    ++hi;
                    const double v2 = vmix[hi] * vmix[hi];
                    sumFourthPowers += v2 * v2;
                }
                while (lo < nlo) {
                    const double v2 = vmix[lo] * vmix[lo];
                    sumFourthPowers -= v2 * v2;
                    ++lo;
                }
                double m = std::clamp(
                    std::pow(std::max(0.0, sumFourthPowers) /
                                 (double)(nhi - nlo + 1),
                             0.25),
                    0.0, 1.0);

                // Keep the policy phase-invariant.  A complete carrier cycle
                // either retains grammar protection or admits the strong
                // aperture override as a unit; per-sample gating here would
                // amplitude-modulate chroma at fSC and recreate the very
                // transition checkerboard this envelope is meant to avoid.
                // Symmetric 5-sample cycle unit (integer centroid; the
                // old [xi-1, xi+2] unit gated half a sample right).
                double cycleLegal = 0.0;
                for (int k = -2; k <= 2; ++k)
                    cycleLegal = std::max(
                        cycleLegal,
                        grammarPassAt(std::clamp(xi + k, 0, width - 1)));

                // ADJUDICATION (replaces the strongApertureOverride,
                // 2026-07-31). Where the detector's luma claim conflicts
                // with grammar-certified legal chroma, neither side
                // overwrites: the conflicted portion's authority is the
                // REGIONAL FACT VERDICT (auditSupp -- the nearest covered
                // frames grading this detector's claims against exact
                // truth, fail-closed where unevidenced). The override let
                // an amplitude threshold cancel positive carrier-law
                // evidence -- gA reads 0.3-0.6 at genuine chroma edges,
                // so certified garment boundaries were suppressed as if
                // convicted (the bikini's loose uncovered boundaries).
                // Its legitimate client, alias-conforming cross-colour
                // that PASSES grammar (the fence/cube), is served by the
                // same court: there the facts endorse the detector and
                // full authority survives. Unconflicted claims (no
                // legality assertion) act as before.
                m *= (1.0 - cycleLegal) +
                     cycleLegal * auditSuppAt(line, xi);

                out[xi] = (float)m;
                // Delta-value return (2026-08-02), coherent renderer's half
                // of the one policy: a fired site's products come from the
                // ANCHORED (lawful) carrier, not from amplitude-killing the
                // elected one. Fallback (no anchored plane) keeps the kill.
                if (prodIRow && prodQRow) {
                    if (anchProdRow) {
                        const int ph = carrierSampleClass(line, left + xi);
                        const double a = anchProdRow[xi];
                        prodIRow[xi] = (float)((1.0 - m) * prodIRow[xi] +
                            m * a * lutTi2[ph] * giProduct2);
                        prodQRow[xi] = (float)((1.0 - m) * prodQRow[xi] +
                            m * a * lutTq2[ph] * gqProduct2);
                    } else {
                        prodIRow[xi] = (float)(prodIRow[xi] * (1.0 - m));
                        prodQRow[xi] = (float)(prodQRow[xi] * (1.0 - m));
                    }
                }
            }
        }
    }

    // ---- LEGAL-BAND CARRIER BETWEEN ANCHORS (user law, 2026-07-31) ----
    // Encoder physics: chroma was bandlimited to 1.3 MHz BEFORE modulation,
    // so demod-envelope energy beyond the legal band CANNOT be chroma -- it
    // is luma trapped in bandpass, by law, no detector consulted. On
    // uncovered frames ("between anchors") the subtraction authority is
    // redefined at construction: the composite carrier becomes the remod of
    // the legal-band I/Q envelopes, so produceY's comb Y = raw - carrier
    // keeps the illegal residue as luma automatically, and the rendered
    // products are limited to the same band so Y + remod(chroma) stays
    // conservative. Covered frames are untouched (facts own them; their
    // referees also read this plane as the comb's claim). The star law is
    // the pointwise extreme of this rule; this is the whole-spectrum form.
    // Escape LDCD_LEGAL_BAND=0.
    static const bool legalBandOn = []{
        const char *e = std::getenv("LDCD_LEGAL_BAND");
        return !(e && std::atoi(e) == 0);
    }();
    // Cleared every frame regardless of path taken: this buffer is reused
    // across frames, and a covered frame (which skips the legal-band pass)
    // must not serve a stale residue plane to produceY on its comp lines.
    bandResidueY_flat.assign((size_t)demodLines * demodWidth, 0.0f);
    // True coverage test, not the anchored-plane proxy (see ccRetired).
    const bool legalFrameCovered =
        chromaFactsOn && frameHasExactCoverage();
    // Retained record from the removed LDCD_PROBE_LEGAL census
    // The census also entered on COVERED frames (measurement only, no
    // writes): with exact truth in hand it graded whether band-limiting the
    // elected carrier moves it toward or away from ex ([LEGALREF]), split
    // flat/detail at hLumaDelta >= 6 IRE, and reported truth's own
    // out-of-band content (demod ex with the same line LUT, band-limit,
    // remod, compare against itself).
    if (legalBandOn && !legalFrameCovered) {
        // 21-tap legal-band LPF (Hann-windowed sinc, 1.3 MHz at 4fSC
        // sampling) -- same design family as filterIQLocked's FIRs.
        static std::once_flag lbInit;
        static std::vector<double> hLB;
        std::call_once(lbInit, [] {
            const double fsMHz = 14.31818, fny = fsMHz * 0.5;
            // Cutoff sweepable for the truth-referee (LDCD_LEGAL_MHZ);
            // production default 1.3 = the encoder's nominal limit.
            const double cutoff = [] {
                const char *e = std::getenv("LDCD_LEGAL_MHZ");
                return e ? std::atof(e) : 1.3;
            }();
            const double wn = cutoff / fny;
            const int N = 21, M = (N - 1) / 2;
            hLB.assign(N, 0.0);
            double sum = 0.0;
            for (int n = -M; n <= M; ++n) {
                const double x = (n == 0) ? (2.0 * wn)
                    : (std::sin(2.0 * M_PI * wn * n) / (M_PI * n));
                const double w = 0.5 * (1.0 + std::cos(M_PI * n / M));
                hLB[n + M] = x * w;
                sum += x * w;
            }
            for (double &v : hLB) v /= sum;
        });
        const int M = ((int)hLB.size() - 1) / 2;
        std::vector<double> i4f(width), q4f(width);
        // Legal carrier per sample, stored for the application stage below
        // (claims are assembled regionally there; this loop only observes).
        const int lbLines = lastLine - firstLine;
        std::vector<float> legalPlane(
            (size_t)lbLines * width,
            std::numeric_limits<float>::quiet_NaN());
        for (int line = firstLine; line < lastLine; ++line) {
            const CombCarrierGrammar *grammar = carrierGrammarLine(line);
            if (!grammar || !grammar->grammarLocked) continue;
            const float *ti4Row = demodTI4fsc_line(line);
            const float *tq4Row = demodTQ4fsc_line(line);
            double *carrierComp = lockedCarrierComposite_line(line);
            if (!ti4Row || !tq4Row || !carrierComp) continue;
            auto lpf = [&](const float *srcRow, std::vector<double> &dst) {
                for (int xi = 0; xi < width; ++xi) {
                    double acc = 0.0;
                    for (int k = -M; k <= M; ++k) {
                        const int xk =
                            std::clamp(xi + k, 0, width - 1);
                        acc += hLB[k + M] * (double)srcRow[xk];
                    }
                    dst[xi] = acc;
                }
            };
            lpf(ti4Row, i4f);
            lpf(tq4Row, q4f);
            for (int xi = 0; xi < width; ++xi) {
                const int h = left + xi;
                // x2: the remod's 0.5 assumes the per-sample product's
                // image terms fold back into the reconstruction. A
                // FILTERED envelope has no image terms, so its remod
                // carries half weight -- verified by the since-removed
                // identity census (unfiltered round-trip exact at 0.03 IRE;
                // filtered reported |src|/2 before this factor).
                const double legalCar = 2.0 *
                    remodGrid4fscToComposite(line, h, i4f[xi], q4f[xi]);
                if (!std::isfinite(legalCar)) continue;
                legalPlane[(size_t)(line - firstLine) * width + xi] =
                    (float)legalCar;
            }
            // NOTE (renderer fact, 2026-07-31): a product-plane LPF stood
            // here and was verified bit-identical OFF vs ON -- the
            // renderer never reads the product planes (colour is
            // demod(raw - Y)). The band law reaches the render only
            // through carrierComp in the application stage below.
        }

        // ---- APPLICATION STAGE: the residue's luma claim (attribution) ----
        // The band law REVOKED the residue's chroma claim; it asserted
        // nothing. The residue moves to Y only by the luma claim the
        // temporal certified-luma witness grants (bandResidueLumaClaim,
        // attributiondefs.h): tween of the two covered neighbours'
        // certified retraction -- luma has no phase, so a tween error is a
        // soft blur, admissible as a WEIGHT where carrier transfer was
        // falsified outright. Claims are pooled per 32x128 region and
        // applied bilinearly (a smooth field, never a per-sample cut);
        // scene cuts and motion self-protect (mismatch -> claim -> 0).
        //
        // DELIVERY: published as bandResidueY (w·(src − legal) per
        // sample), consumed by produceY as a VALUE inside the returned-Y
        // candidate. NEVER written into carrierComp: the elected scalar
        // is one object consumed as both value and election evidence
        // (n.cc), and a second numerically different copy rendered as
        // dot fringing at garment edges (reach-error class, 2026-07-31).
        // Modes (LDCD_LEGAL_CARRIER): 0 off, 1 blanket (A/B; measured
        // +7.7% fsc-band Y = noise return), 2 witness-weighted (default).
        static const int legalCarrierMode = []{
            const char *e = std::getenv("LDCD_LEGAL_CARRIER");
            return e ? std::atoi(e) : 2;
        }();
        if (!legalFrameCovered && legalCarrierMode > 0) {
            const bool witnessOk = legalCarrierMode != 1 &&
                prevF && nextF &&
                prevF->frameHasExactCoverage() &&
                nextF->frameHasExactCoverage() &&
                prevF->carrierRetractedValid &&
                nextF->carrierRetractedValid;
            if (legalCarrierMode == 1) {
                for (int line = firstLine; line < lastLine; ++line) {
                    const double *carrierComp =
                        lockedCarrierComposite_line(line);
                    if (!carrierComp) continue;
                    const float *lp = legalPlane.data() +
                        (size_t)(line - firstLine) * width;
                    float *br = bandResidueY_flat.data() +
                        (size_t)line * demodWidth;
                    for (int xi = 0; xi < width; ++xi)
                        if (std::isfinite((double)lp[xi]))
                            br[xi] = (float)(carrierComp[xi] -
                                             (double)lp[xi]);
                }
            } else if (witnessOk) {
                const int rnx = (width + 127) / 128;
                const int rny = (lbLines + 31) / 32;
                std::vector<double> claimSum((size_t)rnx * rny, 0.0);
                std::vector<double> matchSum((size_t)rnx * rny, 0.0);
                // Asserted-chroma conflict (the residual garment-edge
                // fringing, 2026-07-31): at a co-located chroma+luma edge
                // the witness rightly sees luma structure, but the residue
                // there is the LEGAL chroma edge's own transient. A luma
                // claim firing against an asserted chroma claim is the
                // header's conflict case -- suppressed by carrierLegalProof
                // (same ±2-row construction as pass 2's grammarPassAt),
                // per-cycle at application to stay phase-invariant.
                std::vector<float> proofPlane(
                    (size_t)lbLines * width, 0.0f);
                for (int line = firstLine; line < lastLine; ++line) {
                    const double *carrierComp =
                        lockedCarrierComposite_line(line);
                    const float *rP = prevF->carrierRetracted_line(line);
                    const float *rN = nextF->carrierRetracted_line(line);
                    if (!carrierComp || !rP || !rN) continue;
                    const float *lp = legalPlane.data() +
                        (size_t)(line - firstLine) * width;
                    float *pf = proofPlane.data() +
                        (size_t)(line - firstLine) * width;
                    const lddecode::CarrierAnalysisRecord *anC =
                        carrierAnalysis_line(line);
                    const lddecode::CarrierAnalysisRecord *anU =
                        (line - 2 >= firstLine)
                            ? carrierAnalysis_line(line - 2) : nullptr;
                    const lddecode::CarrierAnalysisRecord *anD =
                        (line + 2 < lastLine)
                            ? carrierAnalysis_line(line + 2) : nullptr;
                    const size_t rb =
                        (size_t)((line - firstLine) / 32) * rnx;
                    for (int xi = 2; xi < width - 2; ++xi) {
                        if (!std::isfinite((double)lp[xi])) continue;
                        const double res = carrierComp[xi] - (double)lp[xi];
                        const double ares = std::fabs(res);
                        if (ares <= 0.0) continue;
                        double proof = anC
                            ? lddecode::carrierLegalProof(
                                  (double)anC[xi].carrierConformance,
                                  (double)anC[xi]
                                      .conformanceSupportFraction)
                            : 0.0;
                        if (anU) proof = std::max(proof,
                            lddecode::carrierLegalProof(
                                (double)anU[xi].carrierConformance,
                                (double)anU[xi]
                                    .conformanceSupportFraction));
                        if (anD) proof = std::max(proof,
                            lddecode::carrierLegalProof(
                                (double)anD[xi].carrierConformance,
                                (double)anD[xi]
                                    .conformanceSupportFraction));
                        pf[xi] = (float)proof;
                        // Witness fine structure: carrier-band bandpass
                        // (lag-2 [-.25,.5,-.25], the LTWEEN stencil) of
                        // EACH covered neighbour separately. TWIN
                        // CONFIRMATION, not a tween: an averaged witness
                        // carries both neighbours' edges at their own
                        // displaced positions, and under motion the claim
                        // fired at ghost geometry -- checker stamped in
                        // the NEXT frame's garment pattern (user-observed,
                        // 2026-07-31: "the checkered stripes... precisely
                        // align with those of the subsequent frame").
                        // Structure is evidence only where BOTH neighbours
                        // attest it independently -- same sign, each above
                        // the firing floor -- credited at the weaker
                        // magnitude. A moving edge has one attester per
                        // ghost position and dies; static detail has two.
                        auto bpOf = [&](const float *r, int x) {
                            return (double)r[x] -
                                   0.5 * ((double)r[x - 2] +
                                          (double)r[x + 2]);
                        };
                        const double bpP = bpOf(rP, xi);
                        const double bpN = bpOf(rN, xi);
                        // Amplitude condition (the CCREF law, third
                        // appearance): bp and residue are both noisy;
                        // min(noise, noise) reads as a match. Fire only
                        // above the 2 IRE floor -- the operating point at
                        // which the LTWEEN referee measured precision 0.70.
                        const double floorS = 2.0 * irescale;
                        const bool attested =
                            bpP * bpN > 0.0 &&
                            std::fabs(bpP) >= floorS &&
                            std::fabs(bpN) >= floorS;
                        lddecode::BandRevokedResidueEvidence ev;
                        ev.residueIRE = ares;
                        ev.witnessMatchIRE = attested
                            ? std::min(std::fabs(bpP), std::fabs(bpN))
                            : 0.0;
                        ev.witnessSupport = 1.0;
                        const size_t r = rb + xi / 128;
                        claimSum[r] += ares;
                        matchSum[r] +=
                            lddecode::bandResidueLumaClaim(ev) * ares *
                            (1.0 - proof);
                    }
                }
                std::vector<double> wReg((size_t)rnx * rny, 0.0);
                for (size_t r = 0; r < wReg.size(); ++r)
                    if (claimSum[r] > 4.0 * irescale)
                        wReg[r] = std::clamp(
                            matchSum[r] / claimSum[r], 0.0, 1.0);
                for (int line = firstLine; line < lastLine; ++line) {
                    const double *carrierComp =
                        lockedCarrierComposite_line(line);
                    if (!carrierComp) continue;
                    const float *lp = legalPlane.data() +
                        (size_t)(line - firstLine) * width;
                    float *br = bandResidueY_flat.data() +
                        (size_t)line * demodWidth;
                    for (int xi = 0; xi < width; ++xi) {
                        if (!std::isfinite((double)lp[xi])) continue;
                        const double ry = std::clamp(
                            ((line - firstLine) - 16.0) / 32.0, 0.0,
                            (double)(rny - 1) - 1e-6);
                        const double rx = std::clamp(
                            (xi - 64.0) / 128.0, 0.0,
                            (double)(rnx - 1) - 1e-6);
                        const int r0 = (int)ry, c0 = (int)rx;
                        const double wy = ry - r0, wx = rx - c0;
                        const int r1 = std::min(r0 + 1, rny - 1);
                        const int c1 = std::min(c0 + 1, rnx - 1);
                        auto W = [&](int rr, int cc) {
                            return wReg[(size_t)rr * rnx + cc];
                        };
                        const double w =
                            (1 - wy) * ((1 - wx) * W(r0, c0) +
                                        wx * W(r0, c1)) +
                            wy * ((1 - wx) * W(r1, c0) +
                                  wx * W(r1, c1));
                        if (w <= 0.0) continue;
                        // Phase-invariant conflict suppression: a complete
                        // carrier cycle acts as a unit (pass 2's rule --
                        // per-sample gating amplitude-modulates chroma at
                        // fSC and manufactures the very checker it fights).
                        const float *pfRow = proofPlane.data() +
                            (size_t)(line - firstLine) * width;
                        // Symmetric 5-sample cycle unit (see cycleLegal).
                        double cycleProof = 0.0;
                        for (int k = -2; k <= 2; ++k)
                            cycleProof = std::max(
                                cycleProof, (double)pfRow[
                                    std::clamp(xi + k, 0, width - 1)]);
                        const double wc = w * (1.0 - cycleProof);
                        if (wc <= 0.0) continue;
                        br[xi] = (float)(wc *
                            (carrierComp[xi] - (double)lp[xi]));
                    }
                }
            }
        }
    }

    // Final split authority for licensed stars.  Earlier passes may have
    // measured, returned, or legal-band-shaped this carrier; none of those
    // estimates may survive the exact-evidence license.  Keep the footprint
    // itself for complementary Y; filterIQLocked then applies its normal
    // aperture to the resulting raw - Y zero, with no special post-filter
    // shape.
    if (starFootprintBuilt &&
        starFootprint_flat.size() >=
            static_cast<size_t>(frameHeight) * demodWidth) {
        for (int line = firstLine; line < lastLine; ++line) {
            const std::uint8_t *star = starFootprint_flat.data() +
                static_cast<size_t>(line) * demodWidth;
            double *carrier = lockedCarrierComposite_line(line);
            float *prodI = lockedProductI_line(line);
            float *prodQ = lockedProductQ_line(line);
            float *maskRaw = lockedCcMaskRaw_line(line);
            float *mask = lockedCcMask_line(line);
            for (int xi = 0; xi < width; ++xi) {
                if (!star[xi]) continue;
                starSplitZeroSamples++;
                if (carrier) carrier[xi] = 0.0;
                if (prodI) prodI[xi] = 0.0f;
                if (prodQ) prodQ[xi] = 0.0f;
                // A zero split cannot subsequently request anchored-carrier
                // return through the cross-colour policy.
                if (mask && mask[xi] > 0.0f)
                    starAnchoredReturnBlocked++;
                if (maskRaw) maskRaw[xi] = 0.0f;
                if (mask) mask[xi] = 0.0f;
            }
        }
    }
}

void Comb::FrameBuffer::filterIQLocked()
{
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;

    constexpr bool   EXP_IQ_FIR_ENABLE = true;
    constexpr int    EXP_FIR_TAPS      = 21;
    constexpr double EXP_I_CUTOFF_MHZ  = 1.5;
    constexpr double EXP_Q_CUTOFF_MHZ  = 0.67;
    static_assert((EXP_FIR_TAPS & 1) == 1,
                  "locked I/Q FIRs must not add a fractional output delay");

    if (!EXP_IQ_FIR_ENABLE) return;

    static std::once_flag firInitFlag;
    static std::vector<double> hI, hQ;
    std::call_once(firInitFlag, [&](){
        auto designLPF = [&](double cutoffMHz)->std::vector<double> {
            const double fsMHz = 14.31818;
            const double fny   = fsMHz * 0.5;
            const double fc    = std::max(0.001, std::min(cutoffMHz, fny - 0.001));
            const double wn    = fc / fny;
            const int    N     = (EXP_FIR_TAPS | 1);
            const int    M     = (N - 1) / 2;
            std::vector<double> h(N, 0.0);
            double sum = 0.0;
            for (int n = -M; n <= M; ++n) {
                const double x = (n == 0) ? (2.0 * wn)
                    : (std::sin(2.0 * M_PI * wn * n) / (M_PI * n));
                const double w = 0.5 * (1.0 + std::cos(M_PI * n / (M + 1e-9)));
                const double v = x * w;
                h[n + M] = v;
                sum += v;
            }
            if (sum != 0.0) for (double &v : h) v /= sum;
            return h;
        };
        hI = designLPF(EXP_I_CUTOFF_MHZ);
        hQ = designLPF(EXP_Q_CUTOFF_MHZ);
    });

    const int NI = (int)hI.size(), NQ = (int)hQ.size();
    if (NI <= 0 || NQ <= 0) return;

    const int MI = (NI - 1) / 2;
    const int MQ = (NQ - 1) / 2;
    const double* tapsI = hI.data();
    const double* tapsQ = hQ.data();
    const int pad = std::max(MI, MQ);

    if ((int)scratch_preI.size() < width) scratch_preI.resize(width, 0.0);
    if ((int)scratch_preQ.size() < width) scratch_preQ.resize(width, 0.0);
    const int extWidth = width + 2 * pad;
    if ((int)scratch_preI_ext.size() < extWidth) scratch_preI_ext.resize(extWidth, 0.0);
    if ((int)scratch_preQ_ext.size() < extWidth) scratch_preQ_ext.resize(extWidth, 0.0);

    for (int line = firstLine; line < lastLine; ++line) {
        double* Irow = componentFrame->u(line);
        double* Qrow = componentFrame->v(line);

        // Colour is the carrier residual the elected luma left behind:
        //
        //     chroma = raw - Y
        //
        // One scalar, two outputs -- the same shape the bucket path has always
        // had (Y[h] = line[h] - val; I/Q = demod(val)).  Because Y is whatever
        // the election emitted, the published colour is its EXACT complement
        // and Y + chroma == raw holds at every pixel by construction.
        //
        // Consuming splitIQlocked's cached products instead lets the two
        // outputs drift: those products are demodulated from the elected comb
        // scalar, while produceY emits a band reassembly that only telescopes
        // back to raw - carrierComp when the comb plane wins the top band.
        // Measured on the beach, ~20% of pixels take a non-comb top band and
        // the emitted Y departs from the complement by 3.5 IRE RMS (max ~11),
        // so the colour on those pixels does not belong to the luma beside it.
        // That is exactly the population the election deviates on -- edges and
        // compact detail.
        //
        // No additional local DC follower here: that would give the colour a
        // different low-frequency convention from the luma it is derived from.
        const quint16* rawLine = rawbuffer.data() + line * videoParameters.fieldWidth;
        const double*  Yrow    = componentFrame->y(line);
        const CombCarrierGrammar *grammar = carrierGrammarLine(line);
        const bool grammarLocked = grammar && grammar->grammarLocked;
        const double bcos = grammarLocked ? grammar->burstCos : 1.0;
        const double bsin = grammarLocked ? grammar->burstSin : 0.0;
        double lutTi[4], lutTq[4];
        if (grammarLocked) {
            for (int i = 0; i < 4; ++i) {
                lutTi[i] = (double)grammar->demodLUTTi[i];
                lutTq[i] = (double)grammar->demodLUTTq[i];
            }
        } else {
            fusedDemodLUT(bcos, bsin, spLUT_locked, cpLUT_locked, lutTi, lutTq);
        }

        // Cross-colour suppression consumes the band-limited envelope
        // splitIQlocked already published (in-field +/-2 vertical mix + lateral
        // boxcar). One policy, computed once; this renderer only applies it, so
        // the suppression cannot alias here.
        //
        // Only certified carrier may replace the raw-Y complement.  On
        // uncovered frames cross-colour return has already acted in Y, so
        // chroma remains exactly raw - emitted Y.
        const float *maskRow = lockedCcMask_line(line);
        const double *factRenderRow = factBackedCarrier_line(line);
        const double *anchRow = factRenderRow;
        const double giProduct = configuration.gi_product;
        const double gqProduct = configuration.gq_product;

        for (int i = 0; i < width; ++i) {
            const int h = left + i;
            const int ph = carrierSampleClass(line, h);
            const double chroma = (double)rawLine[h] - Yrow[h];
            const double m = maskRow
                ? std::clamp((double)maskRow[i], 0.0, 1.0)
                : 0.0;
            const double chromaEff = anchRow
                ? (1.0 - m) * chroma + m * anchRow[i]
                : chroma;
            scratch_preI[i] = (chromaEff * lutTi[ph]) * giProduct;
            scratch_preQ[i] = (chromaEff * lutTq[ph]) * gqProduct;
        }

        // Integer-centred image cancellation (2026-08-01): the RENDER demod
        // was the last consumer left product-raw when 6f502ec registered
        // the measurement paths at the integer carrier sample.  A 4fsc
        // product carries a 2fSC image; the LPFs below reject it on flat
        // fields, but at sharp edges (and the certified family sharpens Y,
        // whose complement this is) the image's sidebands leak under the
        // stopband and render as edge-locked chroma asymmetry -- observed
        // as the smear left of the luma on covered frames.  Same symmetric
        // 3-tap the fix installed everywhere else: no fractional delay, no
        // new registration convention.  Escape LDCD_RENDER_IC=0.
        {
            static const bool renderIC = []{
                const char *e = std::getenv("LDCD_RENDER_IC");
                return !(e && std::atoi(e) == 0);
            }();
            if (renderIC && width >= 2) {
                if ((int)scratch_preI_ext.size() < width)
                    scratch_preI_ext.resize(width);
                if ((int)scratch_preQ_ext.size() < width)
                    scratch_preQ_ext.resize(width);
                std::copy(scratch_preI.begin(), scratch_preI.begin() + width,
                          scratch_preI_ext.begin());
                std::copy(scratch_preQ.begin(), scratch_preQ.begin() + width,
                          scratch_preQ_ext.begin());
                // centeredCarrierProduct3 carries the measurement paths'
                // x2 product-recovery gain (0.5a + b + 0.5c: baseband gain
                // 2, null at 2fSC).  The render chain has its own amplitude
                // convention, so the image cancellation must be UNITY at
                // baseband -- without the 0.5 the whole picture rendered
                // at double saturation (measured x1.93 flat on the GGV
                // colour bars; user-reported as excess saturation).
                for (int i = 0; i < width; ++i) {
                    const int im = std::max(0, i - 1);
                    const int ip = std::min(width - 1, i + 1);
                    scratch_preI[i] = 0.5 * centeredCarrierProduct3(
                        scratch_preI_ext[im], scratch_preI_ext[i],
                        scratch_preI_ext[ip]);
                    scratch_preQ[i] = 0.5 * centeredCarrierProduct3(
                        scratch_preQ_ext[im], scratch_preQ_ext[i],
                        scratch_preQ_ext[ip]);
                }
            }
        }

        constexpr double lockedPreRot =
            LOCKED_CHROMA_PREFILTER_ROT_DEG * M_PI / 180.0;
        const double preC = std::cos(lockedPreRot);
        const double preS = std::sin(lockedPreRot);
        for (int i = 0; i < width; ++i) {
            const double ti = scratch_preI[i];
            const double tq = scratch_preQ[i];
            scratch_preI[i] = ti * preC - tq * preS;
            scratch_preQ[i] = ti * preS + tq * preC;
        }

        double *preIext = scratch_preI_ext.data();
        double *preQext = scratch_preQ_ext.data();
        const double leftI = (width > 0) ? scratch_preI[0] : 0.0;
        const double leftQ = (width > 0) ? scratch_preQ[0] : 0.0;
        const double rightI = (width > 0) ? scratch_preI[width - 1] : 0.0;
        const double rightQ = (width > 0) ? scratch_preQ[width - 1] : 0.0;
        for (int i = 0; i < pad; ++i) { preIext[i] = leftI; preQext[i] = leftQ; }
        std::copy(scratch_preI.data(), scratch_preI.data() + width, preIext + pad);
        std::copy(scratch_preQ.data(), scratch_preQ.data() + width, preQext + pad);
        for (int i = 0; i < pad; ++i) {
            preIext[pad + width + i] = rightI;
            preQext[pad + width + i] = rightQ;
        }

        // ENCODER-MATCHED MODE (A/B: LDCD_RENDER_FEASIBLE=1). The tailored
        // FIRs below remove noise outside each component's legal trace, which
        // is their PURPOSE -- but a 21-tap kernel also spreads every chroma
        // transition (~18 px on Q), and that width is a knock-on, not design
        // (user, 2026-08-01). This mode instead applies the ENCODER'S OWN
        // 9-tap uvFilter (feasibleband.h, 1.3 MHz) to both axes: the same
        // primitive the carrier model already uses, so what it removes is
        // exactly what the encoder could not have modulated -- cleaning the
        // components mindfully of the NTSC spec rather than by a chosen
        // cutoff. Must be judged on BOTH axes at once: cross-colour admitted
        // (GGV monochrome bars) AND transition width.
        static const bool feasibleMode = []{
            const char *e = std::getenv("LDCD_RENDER_FEASIBLE");
            return e && std::atoi(e) != 0;
        }();
        if (feasibleMode) {
            std::vector<double> outI(width), outQ(width);
            lddecode::projectExpressibleChromaEnvelope(
                scratch_preI.data(), nullptr, width, outI.data());
            lddecode::projectExpressibleChromaEnvelope(
                scratch_preQ.data(), nullptr, width, outQ.data());
            for (int i = 0; i < width; ++i) {
                Irow[left + i] = outI[i];
                Qrow[left + i] = outQ[i];
            }
            continue;
        }

        // Both axes are evaluated by symmetric look-around at the same output
        // coordinate h.  The I and Q kernels intentionally have different
        // cutoffs (the oval correction), but the same odd support and centre,
        // so this final bandwidth filter adds no further relative displacement.
        // Registration belongs to the pre-comb products, not to this renderer.
        for (int i = 0; i < width; ++i) {
            double accI = tapsI[MI] * preIext[pad + i];
            double accQ = tapsQ[MQ] * preQext[pad + i];
            const double *cI = preIext + pad + i;
            const double *cQ = preQext + pad + i;
            for (int k = 1; k <= MI; ++k) {
                accI += tapsI[MI + k] * (cI[k] + cI[-k]);
            }
            for (int k = 1; k <= MQ; ++k) {
                accQ += tapsQ[MQ + k] * (cQ[k] + cQ[-k]);
            }
            const int h = left + i;
            Irow[h] = accI;
            Qrow[h] = accQ;
        }
    }
}

// Retained record from the removed LDCD_PROBE_ANCHOR census
//
// ==== Exact-carrier anchor extraction + transfer-error probe ====
//
// The (D-S)/2 exact channel certifies, on covered lines, where compact
// colour starts and stops and at what amplitude -- conservation facts,
// not estimates. Before any estimator may be TETHERED to those anchors,
// the tether needs measured margins: how far does a run's edge and
// amplitude drift per covered-line step (intra-field, the +-2-frame-line
// vertical coherence of one anchored field) and per anchored-frame step
// (cross-letter, A->C = two film frames of motion)? The census extracted
// carrier runs from the exact envelope (hysteresis threshold, sub-pixel
// edges) and printed per-frame aggregates of both transfer errors. The
// p90s ARE the dilation margins the anchor consumers must grant; the
// numbers it produced are carried in buildAnchorCeiling() below.
//
// Anchor = an EDGE (where compact colour starts or stops), so transfer
// was graded edge-to-edge: each edge matched the nearest same-polarity
// edge on the partner line within a window, or counted as unmatched (a
// topology event -- run merge/split, object end -- not a drift). Grading
// runs against runs conflated the two and blew up the tails.
//
// Strong-run view: genuine compact colour, not threshold-skimming
// fragments. Weak wide runs at the ignition threshold have noise-set
// edges and mismatch freely; the tether is calibrated on runs whose
// amplitude proves an object.

// ==== Anchor ceiling: regional carrier-amplitude bound from the exact
// channel (user direction, 2026-07-28: "I didn't want to interpolate the
// non-anchors, just constrain them with the bandwidth"). ====
//
// Per covered line the exact envelope certifies the carrier amplitude;
// pooling it over a lateral window (max-dilate, so the bound can only be
// generous) and padding by the measured transfer margins (amplitude p90
// ~1.5 IRE intra-frame) yields a REGIONAL ceiling: an amplitude no legal
// carrier in this neighbourhood exceeds. Uncovered lines inside an
// anchored frame take the max of the covered lines bracketing them
// (coverage is every other frame line there). Frames with no coverage
// publish no plane and every consumer stays inert -- cross-frame
// authority waits for the confirm-only machinery (45% of edges lapse
// across film frames; a bound must not pretend otherwise). Samples the
// error-corrector DENIED contribute nothing. The plane carries BOUNDS,
// never values -- nothing from it is ever rendered.
void Comb::FrameBuffer::buildAnchorCeiling()
{
    anchorCeiling_flat.clear();
    anchorCoveredLine.clear();
    // Built-for-this-frame marker, set before the early returns: on an
    // uncovered frame "no plane" IS the built result, and every consumer
    // asking later must not trigger a rebuild.
    anchorCeilingValid = true;
    if (exactCarrier_flat.empty()) return;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int fullWidth = videoParameters.fieldWidth;
    if (right - left <= 8 || firstLine >= lastLine || fullWidth <= 0) return;

    // Pooling radius sits just above the measured intra-frame edge-transfer
    // p90 (4 px): wide enough that anchor drift cannot fake a violation,
    // tight enough that a gray strut keeps its own low ceiling instead of
    // inheriting a neighbour's colour. Margins from the measured amplitude
    // drift (p90 1.5 IRE); floor covers the exact channel's noise envelope.
    constexpr int    kLatRadius     = 8;    // lateral pooling half-window
    constexpr double kMarginRel     = 0.10; // relative pad
    constexpr double kMarginAbsIRE  = 1.5;  // absolute pad
    constexpr double kFloorIRE      = 2.5;  // authority floor (noise env)

    const double inf = std::numeric_limits<double>::infinity();
    // Pooled per-line envelope rows, +inf where no authority.
    std::vector<std::vector<double>> pooled(lastLine);
    std::vector<double> env(fullWidth, std::numeric_limits<double>::quiet_NaN());
    bool anyCovered = false;

    for (int line = firstLine; line < lastLine; ++line) {
        const float *ex = exactCarrierRow(line);
        if (!ex) continue;
        int nFinite = 0;
        for (int h = left; h < right; ++h) {
            const float a = ex[h];
            const float b = (h + 1 < right) ? ex[h + 1] : a;
            if (std::isfinite(a) && std::isfinite(b)) {
                env[h] = std::hypot((double)a, (double)b) * invIreScale;
                ++nFinite;
            } else {
                env[h] = std::numeric_limits<double>::quiet_NaN();
            }
        }
        if (nFinite < (right - left) / 2) continue;
        anyCovered = true;

        auto &row = pooled[line];
        row.assign(fullWidth, inf);
        // Sliding max over +-kLatRadius, NaN-aware (denied samples give no
        // authority; a window with too little coverage stays +inf).
        for (int h = left; h < right; ++h) {
            double m = 0.0; int nv = 0;
            const int a = std::max(left, h - kLatRadius);
            const int b = std::min(right - 1, h + kLatRadius);
            for (int j = a; j <= b; ++j) {
                if (std::isfinite(env[j])) { m = std::max(m, env[j]); ++nv; }
            }
            if (nv >= (b - a + 1) / 2)
                row[h] = std::max(kFloorIRE,
                                  m * (1.0 + kMarginRel) + kMarginAbsIRE);
        }
    }

    if (!anyCovered) return;

    anchorCoveredLine.assign(lastLine, 0);
    for (int line = firstLine; line < lastLine; ++line)
        if (!pooled[line].empty()) anchorCoveredLine[line] = 1;

    anchorCeiling_flat.assign((size_t)lastLine * fullWidth,
                              std::numeric_limits<float>::infinity());
    for (int line = firstLine; line < lastLine; ++line) {
        float *out = anchorCeiling_flat.data() + (size_t)line * fullWidth;
        // This line's authority: its own pooled row, else the max of the
        // covered rows within +-2 (the bracketing covered lines).
        const std::vector<double> *src[3] = {nullptr, nullptr, nullptr};
        if (!pooled[line].empty()) src[0] = &pooled[line];
        else {
            if (line - 1 >= firstLine && !pooled[line - 1].empty())
                src[1] = &pooled[line - 1];
            if (line + 1 < lastLine && !pooled[line + 1].empty())
                src[2] = &pooled[line + 1];
            if (!src[1] && line - 2 >= firstLine && !pooled[line - 2].empty())
                src[1] = &pooled[line - 2];
            if (!src[2] && line + 2 < lastLine && !pooled[line + 2].empty())
                src[2] = &pooled[line + 2];
        }
        if (!src[0] && !src[1] && !src[2]) continue;
        for (int h = left; h < right; ++h) {
            double c = src[0] ? (*src[0])[h] : 0.0;
            if (!src[0]) {
                // An uncovered line is bounded only where EVERY bracketing
                // covered line grants authority; the bound is their max.
                double c1 = src[1] ? (*src[1])[h] : inf;
                double c2 = src[2] ? (*src[2])[h] : inf;
                if (!std::isfinite(c1) || !std::isfinite(c2)) { c = inf; }
                else c = std::max(c1, c2);
            }
            out[h] = (float)c;
        }
    }
}

// ---- Star law constants ----
// Signature thresholds are exactly the ones the certified referee graded
// (five-scene battery, 2026-07-30): FP ~0 on starfield and face closeups.
static constexpr int    kStarRegLines        = 32;
static constexpr int    kStarRegCols         = 128;
static constexpr double kStarPeakMinIRE      = 10.0;
static constexpr double kStarFlankAgreeIRE   = 6.0;
// The decoder's IRE convention already places black at 0.  7.5 IRE is the
// historical NTSC setup pedestal and is itself a visibly faded black here;
// retain it only as a generous noise/fade ceiling.  The former 20 IRE limit
// admitted low-luminance coloured regions (the Borg tractor beam) as space.
static constexpr double kStarBlackCeilIRE    = 7.5;
// A dark coloured background can satisfy the luma-black test.  Two raw
// composite cycles on each side of the compact event provide a phase-free
// continuity veto: a real carrier run persists through the star, while black
// space has no phasor to agree.  This is abstention only, never carrier source.
static constexpr double kStarCarrierRunMinIRE = 2.0;
static constexpr double kStarCarrierRunCosMin = 0.80;
// License thresholds: certified |ex| at signature sites. Measured noise
// floor 0.4-0.9 IRE where the law holds absolutely (battle 0.64, Kira 0.53);
// the compact-colour failure scenes pool 2.4-3.0 (temple, dark interior).
static constexpr double kStarLicenseRegionIRE = 1.5;
static constexpr double kStarLicenseFrameIRE  = 1.2;
static constexpr int    kStarLicenseRegionMinN = 3;
static constexpr int    kStarLicenseFrameMinN  = 20;

int Comb::FrameBuffer::certifiedOneDLevel()
{
    // DEFAULT ON, full family (promoted 2026-08-01, user directive on the
    // Emissary title bevel). The contract: certified defs CEDE to center —
    // their separation is already perfect and any comb can only mix truth
    // with a model — and comp pixels comb vigorously with the perfect legs
    // the defs give them. Measured on the title bevel: covered-frame
    // chroma alternation 31.4/29.4 -> 8.3/8.0 (-73%), visually near-clean.
    // The former promotion gate (beach 2026-07-30: anchors-vs-between-
    // frames contrast reads as discrediting alternation) was re-adjudicated
    // by the user: the uncovered frames are the same if not better — the
    // certified frames just show them up more — and that contrast does not
    // justify hiding the feature. Between-frame closure (cross-frame
    // transfer of the certified split on static content) remains the open
    // successor arc, pursued WITH the family live, not ahead of it.
    // LDCD_CERT_1D=0..3 remains as the A/B escape only.
    static const int level = []{
        const char *s = std::getenv("LDCD_CERT_1D");
        if (!s) return 3;
        const int v = std::atoi(s);
        return std::clamp(v, 0, 3);
    }();
    return level;
}

bool Comb::FrameBuffer::certifiedDefLine(int line) const
{
    if (certifiedOneDLevel() == 0) return false;
    if (line < 0 || line >= frameHeight) return false;
    if ((int)certifiedLineCache.size() != frameHeight)
        certifiedLineCache.assign(frameHeight, -1);
    qint8 &c = certifiedLineCache[line];
    if (c >= 0) return c != 0;
    c = 0;
    const float *ex = exactCarrierRow(line);
    if (ex) {
        const int left  = videoParameters.activeVideoStart;
        const int right = videoParameters.activeVideoEnd;
        for (int h = left; h < right; ++h)
            if (std::isfinite(ex[h])) { c = 1; break; }
    }
    return c != 0;
}

// Composite-shape star signature at h. Returns peak IRE over the flank mean
// (0 = no signature); flank outputs are raw-scale. The single source of the
// shape test: the actuator, the evidence build and the CCREF probe (since
// removed) all call here, so the certified grading and the shipped behaviour
// cannot drift.
double Comb::FrameBuffer::starSignatureAt(const quint16 *rawLine, int h,
                                          double *flankOut,
                                          double *flankLOut,
                                          double *flankROut,
                                          bool *carrierRunVetoOut) const
{
    if (carrierRunVetoOut) *carrierRunVetoOut = false;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    if (h - 5 < left || h + 5 >= right) return 0.0;
    const double inv = 1.0 / irescale;
    auto rw = [&](int k) { return (double)rawLine[h + k]; };
    const double flankL = (rw(-5) + rw(-4) + rw(-3)) / 3.0;
    const double flankR = (rw(3) + rw(4) + rw(5)) / 3.0;
    const double flank  = 0.5 * (flankL + flankR);
    const double peakIRE = (rw(0) - flank) * inv;
    if (peakIRE < kStarPeakMinIRE) return 0.0;
    if (std::fabs(flankL - flankR) * inv > kStarFlankAgreeIRE) return 0.0;
    const double dev2IRE = std::max(std::fabs(rw(-2) - flankL),
                                    std::fabs(rw(2) - flankR)) * inv;
    if (dev2IRE > 0.4 * peakIRE) return 0.0;
    if ((flank - (double)videoParameters.black16bIre) * inv >
        kStarBlackCeilIRE) return 0.0;   // stars in black space, narrowly

    // Raw-domain carrier continuity.  Each side is eight samples (two whole
    // 4fSC cycles), outside both the five-sample star event and its black
    // return triplets.  Full cycles cancel DC exactly.  Absolute sample class
    // is used only to compare the two sides on this same line; no burst hue,
    // 1D estimate, demodulated colour, or output phase enters the decision.
    if (h - 13 >= left && h + 13 < right) {
        auto sidePhasor = [&](int k0, double &i, double &q) {
            static constexpr double c4[4] = { 1.0, 0.0, -1.0, 0.0 };
            static constexpr double s4[4] = { 0.0, 1.0, 0.0, -1.0 };
            i = 0.0;
            q = 0.0;
            for (int k = k0; k < k0 + 8; ++k) {
                const int ph = (h + k) & 3;
                const double v = rw(k);
                i += v * c4[ph];
                q += v * s4[ph];
            }
            // 2/N recovers the sinusoid's peak amplitude for N=8.
            i *= 0.25;
            q *= 0.25;
        };
        double li = 0.0, lq = 0.0, ri = 0.0, rq = 0.0;
        sidePhasor(-13, li, lq); // h-13 .. h-6
        sidePhasor(6, ri, rq);   // h+6  .. h+13
        const double la = std::hypot(li, lq);
        const double ra = std::hypot(ri, rq);
        if (la * inv >= kStarCarrierRunMinIRE &&
            ra * inv >= kStarCarrierRunMinIRE) {
            const double coherence =
                (li * ri + lq * rq) / std::max(1e-12, la * ra);
            if (coherence >= kStarCarrierRunCosMin) {
                if (carrierRunVetoOut) *carrierRunVetoOut = true;
                return 0.0;
            }
        }
    }
    if (flankOut)  *flankOut  = flank;
    if (flankLOut) *flankLOut = flankL;
    if (flankROut) *flankROut = flankR;
    return peakIRE;
}

// Certified star evidence, pooled regionally. Covered frames only produce
// evidence; every frame builds the (possibly empty) grid so neighbours can
// pool without shape checks. Idempotent per load.
void Comb::FrameBuffer::buildStarEvidence() const
{
    if (starEvidenceBuilt) return;
    starEvidenceBuilt = true;
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    if (right - left <= 0 || lastLine <= firstLine) return;
    starRegionsX = (right - left + kStarRegCols - 1) / kStarRegCols;
    starRegionsY = (lastLine + kStarRegLines - 1) / kStarRegLines;
    starEvidenceSum.assign((size_t)starRegionsX * starRegionsY, 0.0f);
    starEvidenceCnt.assign((size_t)starRegionsX * starRegionsY, 0);
    if (!frameHasExactCoverage()) return;
    const double inv = 1.0 / irescale;
    for (int line = firstLine; line < lastLine; ++line) {
        const float *ex = exactCarrierRow(line);
        if (!ex) continue;
        const quint16 *rawLine = rawbuffer.data() +
            (size_t)line * videoParameters.fieldWidth;
        for (int h = left + 5; h < right - 5; ++h) {
            if (!std::isfinite(ex[h])) continue;
            if (starSignatureAt(rawLine, h, nullptr) <= 0.0) continue;
            const size_t r = (size_t)(line / kStarRegLines) * starRegionsX +
                             (h - left) / kStarRegCols;
            starEvidenceSum[r] += (float)(std::fabs((double)ex[h]) * inv);
            starEvidenceCnt[r]++;
        }
    }
}

// Build the one authoritative star footprint before the elected carrier is
// published.  Exact evidence licenses the raw-domain signature; neither the
// comb result nor a demodulated hue participates in the decision.  The mask is
// subsequently consumed by both halves of the split: zero carrier here and
// raw luma in produceY().
void Comb::FrameBuffer::buildStarFootprint(const FrameBuffer *prevF,
                                           const FrameBuffer *nextF)
{
    if (starFootprintBuilt) return;
    starFootprintBuilt = true;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;

    starFootprint_flat.assign(
        static_cast<size_t>(frameHeight) * demodWidth, 0);
    starLicensedRegions = 0;
    starSignatureCenters = 0;
    starLicensedCenters = 0;
    starSubstitutedSamples = 0;
    starExtendedCenters = 0;
    starAddedSamples = 0;
    starAddedExactN = 0;
    starCarrierRunVetoes = 0;
    starSplitZeroSamples = 0;
    starAnchoredReturnBlocked = 0;
    starUsedPrevEvidence = false;
    starUsedNextEvidence = false;
    starAddedExactSum = 0.0;
    starAddedExactMax = 0.0;

    static const bool starFixOn = []{
        const char *s = std::getenv("LDCD_STAR_FIX");
        return !(s && std::atoi(s) == 0);
    }();
    if (width <= 0 || firstLine >= lastLine || demodWidth < width)
        return;

    // Evidence is a per-load cache, so it is const-safe to prepare on the
    // temporal witnesses while deciding the current frame's license.
    buildStarEvidence();
    if (prevF) prevF->buildStarEvidence();
    if (nextF) nextF->buildStarEvidence();

    std::vector<std::uint8_t> license;
    bool anyLicense = false;
    if (starFixOn && starRegionsX > 0) {
        const int nReg = starRegionsX * starRegionsY;
        std::vector<double> evSum(nReg, 0.0);
        std::vector<long> evCnt(nReg, 0);
        auto addFrom = [&](const FrameBuffer *f) -> bool {
            if (!f || !f->starEvidenceBuilt ||
                f->starRegionsX != starRegionsX ||
                f->starRegionsY != starRegionsY) return false;
            for (int r = 0; r < nReg; ++r) {
                evSum[r] += f->starEvidenceSum[r];
                evCnt[r] += f->starEvidenceCnt[r];
            }
            return true;
        };
        if (frameHasExactCoverage()) addFrom(this);
        else if (!hasSceneSplit) {
            // Edit flags mark the first field of the new scene.  Never let a
            // regional class license borrow from the opposite side of that
            // boundary: frame 165 of the Borg excerpt inherited tractor-beam
            // carrier from the preceding shot and lost every black-space star
            // for exactly one frame.  Within a shot the original two-sided
            // rule remains; at a boundary the one same-shot witness suffices.
            if (!isSceneStart)
                starUsedPrevEvidence = addFrom(prevF);
            if (!(nextF && nextF->isSceneStart))
                starUsedNextEvidence = addFrom(nextF);
        }

        double total = 0.0;
        long totalN = 0;
        for (int r = 0; r < nReg; ++r) {
            total += evSum[r];
            totalN += evCnt[r];
        }
        const bool frameLicense =
            totalN >= kStarLicenseFrameMinN &&
            total / totalN <= kStarLicenseFrameIRE;
        license.assign(nReg, 0);
        for (int r = 0; r < nReg; ++r) {
            const bool licensed =
                evCnt[r] >= kStarLicenseRegionMinN
                    ? evSum[r] / evCnt[r] <= kStarLicenseRegionIRE
                    : frameLicense;
            license[r] = licensed ? 1 : 0;
            if (licensed) {
                anyLicense = true;
                starLicensedRegions++;
            }
        }
    }

    if (!anyLicense) return;

    for (int line = firstLine; line < lastLine; ++line) {
        const quint16 *rawLine = rawbuffer.data() +
            static_cast<size_t>(line) * videoParameters.fieldWidth;
        const float *exact = exactCarrierRow(line);
        std::uint8_t *footprint = starFootprint_flat.data() +
            static_cast<size_t>(line) * demodWidth;
        for (int h = left + 5; h < right - 5; ++h) {
            double flankL = 0.0, flankR = 0.0;
            bool carrierRunVeto = false;
            const double peakIRE = starSignatureAt(
                rawLine, h, nullptr, &flankL, &flankR, &carrierRunVeto);
            if (carrierRunVeto) {
                starCarrierRunVetoes++;
                continue;
            }
            if (peakIRE <= 0.0) continue;
            starSignatureCenters++;

            const size_t region =
                static_cast<size_t>(line / kStarRegLines) * starRegionsX +
                (h - left) / kStarRegCols;
            if (!anyLicense || region >= license.size() || !license[region])
                continue;
            starLicensedCenters++;

            int leftReturn = -3, rightReturn = 3;
            double leftErr = std::fabs(
                static_cast<double>(rawLine[h + leftReturn]) - flankL);
            double rightErr = std::fabs(
                static_cast<double>(rawLine[h + rightReturn]) - flankR);
            for (int k = -4; k >= -5; --k) {
                const double e = std::fabs(
                    static_cast<double>(rawLine[h + k]) - flankL);
                if (e < leftErr) { leftErr = e; leftReturn = k; }
            }
            for (int k = 4; k <= 5; ++k) {
                const double e = std::fabs(
                    static_cast<double>(rawLine[h + k]) - flankR);
                if (e < rightErr) { rightErr = e; rightReturn = k; }
            }
            const int runLo = leftReturn + 1;
            const int runHi = rightReturn - 1;
            if (runLo < -2 || runHi > 2) starExtendedCenters++;

            for (int k = runLo; k <= runHi; ++k) {
                const int xi = h + k - left;
                footprint[xi] = 1;
                starSubstitutedSamples++;
                if (k < -2 || k > 2) {
                    starAddedSamples++;
                    if (exact && std::isfinite(exact[h + k])) {
                        const double exIRE =
                            std::fabs(static_cast<double>(exact[h + k])) /
                            irescale;
                        starAddedExactSum += exIRE;
                        starAddedExactMax = std::max(starAddedExactMax,
                                                     exIRE);
                        starAddedExactN++;
                    }
                }
            }
        }
    }
}

void Comb::FrameBuffer::produceY(const FrameBuffer *prevF,
                                 const FrameBuffer *nextF)
{
    if (!configuration.phaseCompensation) return;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int fullWidth = videoParameters.fieldWidth;
    const int width     = right - left;
    if (width <= 0) return;

    const int srcBuf = std::clamp((int)configuration.dimensions - 1, 0, 2);
    const bool showMap = configuration.showMap;

    // splitIQlocked owns the star decision.  produceY only consumes its
    // published zero-carrier footprint as the complementary raw-luma value.
    buildStarFootprint(prevF, nextF); // idempotent safety for direct callers

    // Retained record from the removed LDCD_PROBE_SYNC census (produceY half)
    //
    // The per-frame pooling read the def lines' working-space certified IQ
    // per region. The working space is hue-common across lines UP TO
    // lineFlip (BurstLockedSigned: polarity baked in); pooling without it
    // measured a fictitious 180-degree alternation, so the convention must
    // be applied. The PREVIOUS covered frame also had to be snapshotted
    // before the accumulator replaced it -- the first build compared the
    // frame against itself (act = 0 by construction).
    //
    // END-TO-END SHIPMENT CHECK (covered frames, -t 1): compare the
    // ASSEMBLER-SHIPPED increment against the increment the decoder itself
    // measures between consecutive covered frames in ITS OWN working space.
    // This is the user's demanded certainty that rotating the certifieds
    // into 4fsc needs no adjustment of the tone: corr +1 = conventions
    // aligned, corr -1 = handedness flip (fix at one defined boundary),
    // drifting offset = contract broken. Reported, never assumed.
    //
    // RAW REPLICA (user: "DecoderPool and CadenceAssembler do their work
    // with composite before split1D... be sure we are comparing apples to
    // apples"). Reproduce the assembler's measurement EXACTLY,
    // decoder-side: certified carrier (exactCarrierRow IS the shipped chat)
    // + raw burst samples of the same def lines, same k&3 basis, same
    // field-row signing (frame line L of the def parity is field line L/2),
    // same burst derotation. Three trajectories then separate the question:
    // ship vs rawAct proves the CONTRACT; rawAct vs lutAct measures what the
    // decoder's own burst-locking adds on top.

    // Anchor ceiling: pooled regional amplitude bound from the exact
    // channel. Cheap no-op on frames without coverage. The retraction
    // stage's certified-anchor fit hull usually built it already this
    // frame; build here only when that stage did not run.
    if (!anchorCeilingValid) buildAnchorCeiling();

    // Retained record from the removed LDCD_PROBE_YCAND census
    // Per-CANDIDATE grading against certified luma. User question
    // (2026-08-01): "we've established that there's still some HF Y left on
    // the table -- do you see low-hanging fruit in pulling more HF Y from
    // comb, witness or return?"  At covered samples Ltrue = raw - exact is
    // fact, so every candidate can be graded individually, and the ORACLE
    // bound (per-pixel best candidate) sizes exactly how much fidelity the
    // election is leaving behind.  Run HELD OUT (LDCD_CERT_1D=0).
    //
    // HF-AS-QUALITY-PROXY TEST (user proposal, 2026-08-02: "Rewarding HF
    // content is a nice, candidate neutral score that would give retracted a
    // boost without calling it by name"). Before wiring any such term into
    // the cost, ask truth whether it is a valid proxy: does the candidate
    // carrying the MOST HF actually stand closest to certified luma?
    // Answered as (a) how often argmax|HF| == the truth-best candidate,
    // against the 1/n chance baseline, and (b) what a pure "always pick max
    // HF" selector would have scored versus the shipping election and the
    // per-pixel oracle. The verdict is recorded at the ELECTION v2 block in
    // the pixel loop below.
    //
    // RETURN-DELIVERY SPLIT (user, 2026-08-02: return must "win on the
    // merits where it's delivering", since comb and retracted do not address
    // bandpass luma at all). Delivering = where the returned candidate
    // actually departs from comb, i.e. where it is claiming stolen bandpass
    // luma. Everywhere else it is comb by construction and grading it says
    // nothing. The per-plane proximity01 / return-evidence capture that fed
    // the [YRETPROX] line is gone with it; the argument it tested survives
    // at the cost term itself.
    // Retained record from the removed LDCD_PROBE_RETHULL census
    // Is the return hull clamping GOOD values? User challenge (2026-08-02):
    // "Retracted has such better stats than comb that I would need a
    // tiebreaker to award it to comb before I'd take it from retracted. Do
    // you have a confirmation that retracted is in the wrong on those
    // differences?" -- correct: the hull's segment [comb, raw] has BOTH
    // endpoints defined by comb, so it encodes "comb's carrier claim is
    // right", which is an assumption, not an impossible (the carrier is
    // signed; if comb's sign or magnitude is wrong the true luma lies
    // outside). Graded against certified truth on def lines: pre-hull vs
    // post-hull distance to raw - exact, over all fired samples and over the
    // binding subset. Run in SHIPPING config with LDCD_CHROMA_FACTS=0 (so
    // return is not retired on covered frames).

    // Retained record from the removed LDCD_PROBE_RETIMPACT census
    // Quantify return's impact on UNCOVERED frames, where no certified fact
    // exists to grade against (YCAND/YCERT are structurally blind there).
    // User (2026-08-01): "you have tested uncovered frames with other tools
    // and held back certified at other times. Please quantify return's
    // impact." Reference: the antRefLuma chain (Pass 1.7's tween, validated
    // TRUTH-GRADE on static content -- ANTGRADE box: 0.78 IRE mean|err|,
    // corr +0.997 vs certified truth at the title bevel), read DIRECTLY
    // since produceY and buildCertifiedCarrierStage share the same
    // FrameBuffer. Direct-tier samples only (antRefLuma_flat[line] itself
    // finite, not the vertical-bracket fallback) -- the higher-confidence
    // tier. This is an APPROXIMATE reference, not fact: it must be reported
    // as such, and motion is expected to degrade it (the ghost lesson)
    // rather than 0 error meaning certification. Independent of
    // LDCD_CERT_1D/_TONE (reads the chain plane directly); meaningful only
    // where antRefAge>=1 (chain reached this frame) and NOT
    // frameHasExactCoverage() (this frame is itself uncovered -- the
    // population the fact-based censuses cannot reach).

    // Retained record from the removed LDCD_PROBE_RETGRADE census
    // Grade return against CERTIFIED TRUTH, on the OLD (uncertified)
    // carrier -- user's design (2026-08-01): "if return is run on the old
    // dg-discard carrier of the def, we have our measure against certified."
    // True --dg-discard is an assembler-time (ld-cinemap) flag, not
    // available at decode time; the equivalent here is LDCD_CERT_1D=0 (comb
    // computes its own estimate instead of ceding to exact) PLUS
    // LDCD_CHROMA_FACTS=0 (the suppression mask does not retire to zero on
    // covered frames -- ccRetired depends on frame coverage, not on CERT_1D,
    // so without this second gate return is a silent no-op regardless of
    // CERT_1D, as measured on the cube). Both held out together put return
    // in EXACTLY the regime it operates in everywhere else (ordinary
    // carrier, no certification) -- but on def lines, where truth exists to
    // grade it. The census was run TWICE (--cross-color-return 0 and 1.0)
    // and paired by frame index externally: eOff - eOn > 0 means return
    // recovered luma 1D's bandpass stole; < 0 is a false positive (return
    // moved AWAY from truth). dsExactRow had to be read directly
    // (certifiedDefLine() short-circuits to false whenever
    // certifiedOneDLevel()==0).
    //
    // Residual-budget decomposition (user, 2026-08-01: "figure out what the
    // best thread to pull on might be to get at that 88%"): cells of
    // [mask state x site class]. maskBin: 0 = quiet (<0.05), 1 = partial
    // (0.05-0.5), 2 = strong (>0.5) -- with return OFF the mask plane is
    // all-zero (not built), so bins 1-2 populate only in the ON run; the
    // OFF/ON comparison per SITE CLASS gives recovery, the ON run's mask
    // bins split the residual into fired-but-undershot vs never-fired.
    // siteClass: 2 = impulse (lumaImpulseRisk>0.3, wins over edge),
    // 1 = edge (hLumaDelta >= 6 IRE, the cc edge-read soft knee),
    // 0 = flat/texture. Each error was further split into its CARRIER-BAND
    // and out-of-band parts: the in-band share is the theft signature,
    // since luma stolen by the 1D bandpass lives near fSC, so residual
    // error that is IN-band at a quiet-mask site is recoverable theft the
    // detector missed, while broadband error there was never stolen luma at
    // all and return cannot fix it.

    // Election bypass: see the per-pixel site. Diagnostic A/B only.
    static const bool electBypass = []{
        const char *s = std::getenv("LDCD_ELECT_BYPASS");
        return s && std::atoi(s) != 0;
    }();

    // Fixed-kernel notch candidate (plane 5). OFF by default.
    //
    // The exact relation to comb, which governs everything below:
    //
    //   notchY - combY = carrierComp - sin^2(D)*raw
    //
    // where sin^2(D) = [-1,0,2,0,-1]/4 is the notch's complement. Split raw
    // into pedestal + carrier + luma: sin^2(D) takes the whole carrier plus
    // the near-fSC part of the luma, so wherever comb's carrier estimate is
    // sound the two candidates differ by EXACTLY -sin^2(D)*L -- the near-fSC
    // luma, and nothing else. Where the picture has none, both publish the
    // same pedestal and the roster's identity dedup correctly refuses to
    // count one hypothesis twice.
    //
    // PARTICIPATION, program material (2026-08-09, Emissarymovie-s1x11, an
    // 11-disc stack, cube at -s 2600, ntsc3d + fvf + witness + ccr 1.0):
    //   offered on every election pixel, 40-50% ADMITTED, 50-60% duplicate,
    //   0% infeasible; at seated pixels mean blindness 1.50 IRE with only
    //   1.4% above the 10 IRE chromaT calls high chroma, so the gate leaves
    //   the candidate 74% of its weight (mean factor 0.7385).
    // Effect on emitted Y vs notch-off: 96% of pixels on alternating output
    // frames, mean 0.14 IRE, p95 0.44, max 17.9 -- and 0.3% on the frames
    // between. That odd/even split is UNEXPLAINED; do not assume the
    // pulldown letters without measuring it. (Census predates the covered-
    // frame cede below, 2026-08-09 -- covered frames no longer seat the
    // notch at all, so re-measure before reasoning from these numbers.)
    //
    // RETRACTED, and worth the space because it nearly buried the plane: the
    // first census ran on ve-snw-cut and reported 99.18% duplicate, 0.82%
    // admitted, mean blindness 10.93 IRE, 48% of seated pixels above 10 IRE
    // -- from which this comment concluded the notch was structurally
    // incapable of contributing, since distinctness and blindness appeared
    // to COINCIDE. They do not. The pattern is not a stack and carries no
    // near-fSC luma for the notch to disagree with comb about, so every
    // disagreement it did have was carrier by construction. The identity
    // above is material-independent; the inference drawn from it was not.
    //
    // The gate is nonetheless load-bearing, measured on the pattern where it
    // could be isolated: sweeping tau left participation pinned while
    // max|dY| ran 0.30 IRE (tau=1) -> 1.39 (tau=4) -> 44.9 (tau=1000).
    // Ungated, this plane publishes tens of IRE of "no summit here" at the
    // summits. Tau is the first thing to sweep on real material too.
    //
    // STILL UNMEASURED: whether any of this is BETTER. Participation and
    // departure are not fidelity. Grade against certified luma before
    // promoting or removing.
    static const bool notchCandidate = []{
        const char *s = std::getenv("LDCD_Y_NOTCH");
        return s && std::atoi(s) != 0;
    }();
    // CERTIFIED CEDE for the notch (user, 2026-08-09: "pulling notch out
    // of covered frames altogether, given its ungovernability"): the same
    // rule every comb candidate carries, in the only shape a raw-only
    // kernel can carry it. On a covered frame the def lines are fact
    // (the early-out below) and the comp lines' candidates all descend
    // from the certified ladder, so they dedup into ONE roster entry —
    // which made the notch the lone dissenter with its largest blend
    // share exactly where the material is already separated, and its
    // touch, confined to every other line, rendered as interline
    // alternation on the certified fields. A plane that cannot inherit
    // certification has no franchise on a covered frame. Per-frame, never
    // per-line: uniform provenance.
    const bool notchLive = notchCandidate && !frameHasExactCoverage();
    // Blindness scale for the notch's alpha withdrawal, IRE. Tight by
    // design: chromaT treats 10 IRE as high chroma, and the notch's own
    // failure mode (publishing a near-zero top just above the carrier, where
    // cos^2 is only 0.07 at 4.2 MHz) has to be caught before it softens the
    // blend. The first thing to sweep.
    static const double notchBlindTauIRE = []{
        const char *s = std::getenv("LDCD_Y_NOTCH_TAU");
        return s ? std::atof(s) : 4.0;
    }();

    // Line-scope copy of the LDCD_CC_FACTS switch for the certified
    // early-out below (the pixel loop resolves its own; same env, same
    // once-only value).
    static const bool certChromaFactsY = []{
        const char *e = std::getenv("LDCD_CC_FACTS");
        return !(e && std::atoi(e) == 0);
    }();

    // PERF (2026-08-06): per-line plane-luma caches for the election's
    // samplers. planeY is pure per (plane, sample) given the line's row
    // pointers, but the per-pixel scoring reads it through overlapping
    // windowed apertures — of order 40–90 evaluations per pixel. Each
    // active plane is evaluated once per sample instead; the samplers keep
    // their loop order and weights, so every downstream sum is performed in
    // the same order on the same values (byte-identical by construction;
    // verified against the pre-change binary, md5 dc131e1c on 24 frames of
    // covered program material).
    std::vector<double> pyRow0(width), pyRow1(width), pyRow3(width),
                        pyRow4(width), pyRow5(width);
    // Companion caches, same contract: candidate residuals (planeY - coarse,
    // the samplers' own subtraction, done once) and the carrier-basis LUT
    // values per sample class (the per-tap grammar lookup, done once).
    std::vector<double> resRow0(width), resRow1(width), resRow3(width),
                        resRow4(width), resRow5(width),
                        spRowV(width), cpRowV(width);


    // produceY is a pure consumer. splitIQlocked() exports the selected comb
    // scalar at the same physical integer sample as raw, and that exact scalar
    // is the coherent subtraction authority. In 3D, clpbuffer[2] has already
    // been selected by split3D/getBestCandidate before splitIQlocked() runs, so
    // coherent 3D high-frequency luma follows the same raw-minus-carrier path;
    // no separate post-comb Y channel or temporal Y election is required.

    for (int line = firstLine; line < lastLine; ++line) {
        if (line >= demodLines) continue;

        const quint16 *rawLine = rawbuffer.data() + line * fullWidth;
        double *Y = componentFrame->y(line);
        const double *clpLine = clpbuffer[srcBuf].pixel[line];
        const double *carrierComp = lockedCarrierComposite_line(line);

        // --luma-witness IS the access to this candidate, and the only gate
        // it needs (author, 2026-08-08). carrierRetractedValid now tracks the
        // option exactly, and the accessor tests it, so asking for the row is
        // asking for the feature.
        //
        // A COVERAGE gate stood here and was wrong: it demanded the witness be
        // fact-backed before it could be heard, which is not what the witness
        // ever claimed to be -- it is an estimate-grade luma hypothesis by
        // design, and the Y election is what adjudicates it. Worse, it
        // withdrew the candidate from the material that has no facts at all:
        // interlaced footage carries no 3:2 cadence, so frameHasExactCoverage()
        // is false on every frame and the feature could never appear on the
        // very footage a user enables it for.
        const bool coveredFrame = frameHasExactCoverage();
        const float *retractedRow = carrierRetractedValid
            ? carrierRetracted_line(line) : nullptr;
        const float *ccMaskRow = lockedCcMask_line(line);
        const float *icebergYRow = icebergRecoveredY_line(line);
        const float *icebergWRow = icebergReturnWeight_line(line);
        const double icebergPolicyWeight = std::clamp(
            configuration.tunables.CC_SUPPRESSION_WEIGHT, 0.0, 1.0);
        // Plane 4 may use covered carrier fact.  On uncovered frames its new
        // value source is explicitly recovered luma, never a carrier estimate.
        const double *returnedFactCarrierRow =
            factBackedCarrier_line(line);
        const bool returnedHasDerivedCarrier =
            returnedFactCarrierRow;

        // Diagnostic view export (A/B only, same family as LD_RETRACTED_ADMIT).
        // LDCD_YVIEW publishes ONE election contestant AS Y so each candidate
        // can be rendered and compared in the identical pipeline geometry.
        // Pair with --chroma-gain 0 for a pure grayscale read.  This is an
        // inspection port, not a policy path: it bypasses the election
        // entirely and must never be a default.
        //
        //   mono      raw composite (carrier included) -- ground reference
        //   comb      plane 0, raw - lockedCarrierComposite
        //   retracted plane 1, the published retracted product (default:
        //             raw - w·carrierFit; LDCD_RETRACTED_SOURCE overrides)
        //   oned      plane 3, raw - locked1DSource
        //   returned  plane 4, combY + ccMask*(raw - combY)
        //   notch     plane 5, [1,0,2,0,1]/4 on raw (no LDCD_Y_NOTCH needed:
        //             the view samples the plane directly)
        //   (unset)   the elected output
        //
        // The plane views route through planeY() below so they are EXACTLY
        // the values the election sees -- no second implementation to drift.
        static const int yViewMode = []{
            const char *s = std::getenv("LDCD_YVIEW");
            if (!s) return -1;
            // "retracted" and "returned" share the "ret" prefix; discriminate
            // on the fourth character rather than the first three.
            if (std::strcmp(s, "mono") == 0)      return 100;
            // The NATIVE inverse-encoder view: raw minus the per-line fit,
            // with no vertical promotion.  "retracted" below is raw minus
            // combedCarrier, i.e. the promoted, comb-conditioned product.
            if (std::strcmp(s, "native") == 0)    return 101;
            if (std::strcmp(s, "comb") == 0)      return 0;
            if (std::strcmp(s, "retracted") == 0) return 1;
            if (std::strcmp(s, "oned") == 0)      return 3;
            if (std::strcmp(s, "returned") == 0)  return 4;
            if (std::strcmp(s, "notch") == 0)     return 5;
            return -1;
        }();
        if (yViewMode == 100) {
            for (int h = left; h < right; ++h)
                Y[h] = (double)rawLine[h];
            continue;
        }
        if (yViewMode == 101) {
            const float *fitRow = carrierFit_line(line);
            for (int h = left; h < right; ++h)
                Y[h] = fitRow ? ((double)rawLine[h] - (double)fitRow[h - left])
                              : (double)rawLine[h];
            continue;
        }

        // ---- Certified early-out (user direction, 2026-08-09) ----
        //
        // On a certified sample the conservation fact Ltrue = raw - exact IS
        // the luma, at full band. There is nothing for an election to decide:
        // every candidate is obliged to carry that value unadulterated (the
        // ratchet law), so adjudicating between them can only dilute a fact
        // with estimates. Emit it and run no scoring at all -- no candidate
        // planes, no cleanliness projections, no neighbour probes. A quantity
        // that cannot change the answer must not be computed.
        //
        // This is also the structural cure for a defect the notch candidate
        // exposed: a plane derived purely from raw cannot inherit
        // certification, so it entered the blend at certified positions as an
        // estimate and pulled Y off truth -- and worse at certified samples
        // than uncertified ones, because the fact-carrying candidates dedup
        // into a single entry there and the lone dissenter's share of the
        // blend rises. Ceding the plane to fact would have fixed the notch;
        // this fixes the CLASS, for every present and future candidate,
        // by removing the decision rather than policing the candidates.
        const float *certExactRow = exactCarrierRow(line);
        const bool certLineActive =
            certChromaFactsY && certExactRow && certifiedDefLine(line);
        bool certLineComplete = false;
        if (certLineActive) {
            certLineComplete = true;
            for (int h = left; h < right; ++h)
                if (!std::isfinite(certExactRow[h])) {
                    certLineComplete = false;
                    break;
                }
        }
        if (certLineComplete) {
            // Whole line is fact: skip the election block entirely, including
            // its per-line plane caches. Falls through to the star footprint
            // pass below, which is a separate declaration and still applies.
            for (int h = left; h < right; ++h)
                Y[h] = (double)rawLine[h] - (double)certExactRow[h];
        } else if (retractedRow || ccMaskRow) {
            // ================= HF luma election =================
            //
            // One selected coarse owns LF. Comb owns the middle band and is
            // the provisional top; a per-pixel election adjudicates only that
            // top among complete raw-carrier luma candidates. This replaces the prior
            // gate = max(gA, deltaGate, wGate); yOut = combY + ccReturn*gate*delta
            // blend, which produced a 2fSC checkerboard at chroma-amplitude
            // transitions: wGate (a DISTANCE between candidates, |combY - wY|)
            // and delta (= retractedY - combY) are both carrier-band, and the
            // witness lurch leaks Δchroma at carrier rate where chroma amplitude
            // is changing, so wGate*delta = fsc*fsc = DC + 2fSC. The cure is to
            // make every contributor a complete luma VALUE and let direct,
            // named measurements about each candidate (never a distance between
            // candidates) caution -- never override -- a selection anchored by
            // robust consensus and geometry.
            // Pattern: ld-disc-stacker neighbor modes (medoid center + inlier DQ
            // + capped quality penalty + neighbor selection) and the FVF
            // neighbor anchor (comb.cpp:1398).
            //
            // At most three contestants are active:
            //   0 coherentY  = raw - carrierComp     (phase-solved selected comb)
            //   1 retractedY = carrierRetracted      (raw - combedCarrier)
            //   3 1D         = raw - locked1DSource  (replaces comb only if DQ'd)
            //   4 returnedY  = combY + ccMask*(raw - combY), derived and
            //                  admitted only after the base population is fixed.
            // Coarse-platform selector. Default owns one cheap raster-aligned
            // four-sample coarse. --luma-witness unlocks the heavier centered,
            // lurch-sharpened platform. The selected platform is the sole LF
            // authority and defines the top-band coordinate. No second coarse
            // is mixed into reconstruction.
            const bool useSharpCoarse =
                configuration.lumaWitness && !lockedLumaSharp_flat.empty();
            auto coarseFloor_line = [&](int l) -> const double * {
                return useSharpCoarse ? lockedLumaSharp_line(l)
                                      : lockedLumaBaseY4_line(l);
            };
            const double *coarseRow =
                (lockedLumaCacheValid && demodWidth == width)
                    ? coarseFloor_line(line)
                    : nullptr;
            const double *oneDRow = locked1DSource_line(line); // may be null
            const std::uint8_t *bandRow = chromaBoundaryBand_line(line);
            const float *dsExactRow = exactCarrierRow(line);
            const float *anchorRow = anchorCeilingRow(line);
            const lddecode::CarrierAnalysisRecord *analysisRow =
                carrierAnalysis_line(line);
            const float *alienRow = regionAlienPartner_line(line);
            // Star/impulse facts (single producer: compactLumaExcursionEvidence)
            // for the impulse-seniority bias in the election scoring.
            const AttributionEvidence *attribRow = attributionEvidence_line(line);
            // Retracted-admission mode (isolation switch for the cube/beach
            // A/B).  Default: conflicted fits admit retracted only with
            // neighbour spatial support.  LD_RETRACTED_ADMIT=trust restores
            // the residualTrust-only hard gate; =all seats unconditionally
            // (pre-2026-07-02 behaviour).
            static const int retractedAdmitMode = []{
                const char *s = std::getenv("LD_RETRACTED_ADMIT");
                if (!s) return 0;              // 0 = spatial (default)
                if (s[0] == 't') return 1;     // trust-only hard gate
                if (s[0] == 'a') return 2;     // admit all
                return 0;
            }();
            const bool retractedAdmitSpatial = (retractedAdmitMode == 0);
            const bool retractedAdmitAll = (retractedAdmitMode == 2);

            // Structural carrier-amplitude ceiling (samples): I/Q are bounded
            // sinusoids, so apparent carrier beyond this must be luma. Used as
            // the feasibility DQ. The optional witness hull uses the same bound.
            // Measured limits (bars/beach) -- see maxCarrierAmpIREFromScale.
            const CombCarrierGrammar *grammarLine =
                carrierGrammarLine(line);
            const double maxCarrierAmpSamples =
                maxCarrierAmpIREFromScale(
                    grammarLine ? grammarLine->carrierScale : 0.0) * irescale;

            // Election tolerances (IRE -> samples).
            const double inlierTol  = 4.0 * irescale; // medoid inlier gate
            const double phasePenSamp =
                std::max(0.0, configuration.tunables.PRODUCE_Y_PHASE_PENALTY_IRE)
                * irescale; // capped phase hygiene penalty

            // Carrier-basis window norms (constant per line: the 4-sample window
            // always spans the full set of phases regardless of start).
            double basisSN = 0.0, basisCN = 0.0;
            for (int i = 0; i < 4; ++i) {
                basisSN += spLUT_locked[i] * spLUT_locked[i];
                basisCN += cpLUT_locked[i] * cpLUT_locked[i];
            }

            // Candidate plane sampler: complete raw - carrier luma at sample hh.
            // Any non-finite source falls back to comb so a plane never poisons
            // a neighbor probe with NaN.
            auto planeY = [&](int plane, int hh) -> double {
                const int xx = hh - left;
                if (plane == 1 && retractedRow) {
                    const double r = (double)retractedRow[xx];
                    if (std::isfinite(r)) return r;
                } else if (plane == 3 && oneDRow) {
                    const double o = oneDRow[xx];
                    if (std::isfinite(o)) return (double)rawLine[hh] - o;
                } else if (plane == 4 && ccMaskRow) {
                    const double c = carrierComp ? carrierComp[xx] : clpLine[hh];
                    const double comb = (double)rawLine[hh] -
                        (std::isfinite(c) ? c : 0.0);
                    const double m = std::clamp((double)ccMaskRow[xx], 0.0, 1.0);
                    if (!coveredFrame && icebergYRow && icebergWRow &&
                        std::isfinite((double)icebergYRow[xx])) {
                        const double iw = icebergPolicyWeight * std::clamp(
                            (double)icebergWRow[xx], 0.0, 1.0);
                        if (iw > 0.0)
                            return comb + iw *
                                ((double)icebergYRow[xx] - comb);
                    }
                    // Delta-value return (2026-08-02): the returned VALUE is
                    // the measured contamination -- the elected carrier's
                    // excess over the anchored carrier -- so lawful chroma
                    // stays in chroma. raw - anchored == retracted luma by
                    // construction, so a full commit hands the site the
                    // anticipation-anchored luma. Lab conviction: the
                    // committed verdict with the old full-residual value
                    // recovered 20.8% of the stolen-luma gap but paid
                    // 1.2-2.3 IRE at fired sites on colored content (legal
                    // carrier dumped into Y). Where no anchored plane
                    // exists the full-residual return stands -- there the
                    // verdict itself came from the fallback witness.
                    //
                    // WITHDRAWN 2026-08-02 -- a segment hull [comb, raw] was
                    // installed here and removed the same day. It was NOT a
                    // hull on an impossible: both endpoints are defined by
                    // comb, so its width IS the comb's carrier claim, and
                    // passing raw only means the residual carrier changes
                    // SIGN -- ordinary for a signed oscillation whenever
                    // comb's sign or magnitude is wrong. It therefore
                    // encoded "comb's carrier claim is correct" as if it
                    // were a law, and handed the tie to comb (3.22 IRE at
                    // return-delivering sites) over retracted (2.32 IRE).
                    // Measured: it bound on 5.2% of pixels but on 0.00% of
                    // COVERED def samples -- i.e. exclusively where no exact
                    // channel exists to confirm it -- and moved the 2fSC
                    // speckle class it was meant to fix by +0.0..+0.3%.
                    // A restriction that acts only where it cannot be
                    // verified, against the better estimator, needs a
                    // tiebreaker; none exists. The genuine impossible (a
                    // carrier larger than any legal carrier) is already
                    // enforced by feasible() on the roster.
                    if (returnedHasDerivedCarrier && retractedRow) {
                        const double r = (double)retractedRow[xx];
                        if (std::isfinite(r)) return comb + m * (r - comb);
                    }
                    return comb + m * ((double)rawLine[hh] - comb);
                } else if (plane == 5) {
                    // Fixed-kernel notch: cos^2(w) = [1,0,2,0,1]/4 on raw.
                    //
                    // The one contestant with NO lineage. No fit, no mask, no
                    // phase solve, nothing estimated -- it reads raw and a
                    // constant, so it carries no per-pixel claim that can be
                    // wrong, and its behaviour is known in closed form at
                    // every frequency. That independence is the whole case
                    // for it: comb, retracted and returned all descend from
                    // one IQ lineage and can therefore agree for a common
                    // reason rather than a corroborating one.
                    //
                    // WHY THE SQUARE. The bare [1,0,1]/2 notch is |cos w| in
                    // MAGNITUDE but cos w in SIGN, and cos w is NEGATIVE from
                    // fSC to Nyquist -- exactly the span the top-band
                    // extractor (1 - fourMean) holds at unity. It would
                    // publish the raster's finest luma polarity-reversed: a
                    // one-pixel highlight rendered as a one-pixel hole. In
                    // luma there is no phase convention to absorb that the
                    // way the carrier grammar absorbs a chroma flip -- the
                    // number IS the brightness. Squaring keeps the null at
                    // fSC and unity at DC and 2fSC while never going
                    // negative; the price is a wider skirt (0.50 at half-fSC
                    // where |cos| held 0.707).
                    //
                    // At +-2 reach this kernel is FORCED, not preferred:
                    // symmetric [a,0,b,0,a] with unity DC (b + 2a = 1) and a
                    // null at fSC (b - 2a = 0) has the single solution
                    // a = 1/4, b = 1/2. Odd taps cannot buy anything -- they
                    // contribute exactly nothing at fSC (cos(pi/2) = 0) while
                    // breaking the DC/Nyquist balance a luma candidate needs.
                    // A narrower notch requires +-4 and negative outer taps,
                    // which ring; these taps are non-negative and sum to 1,
                    // so this kernel is a weighted average and cannot
                    // overshoot its own support. Its reach also stays inside
                    // the +-2 apertures that judge it (fourMean,
                    // carrierCleanlinessOf, completeTopAt), so no referee
                    // here is scoring content it cannot see.
                    const int hm = std::clamp(hh - 2, left, right - 1);
                    const int hp = std::clamp(hh + 2, left, right - 1);
                    return 0.25 * ((double)rawLine[hm] +
                                   2.0 * (double)rawLine[hh] +
                                   (double)rawLine[hp]);
                }
                const double c = carrierComp ? carrierComp[xx] : clpLine[hh];
                return (double)rawLine[hh] - (std::isfinite(c) ? c : 0.0);
            };

            // Candidate export port (see LDCD_YVIEW above): publish one
            // contestant's COMPLETE luma, exactly as the election samples it.
            if (yViewMode >= 0) {
                for (int h = left; h < right; ++h)
                    Y[h] = planeY(yViewMode, h);
                continue;
            }

            // Fill the per-line plane caches (see the declaration above the
            // line loop). Planes whose source row is absent fall through to
            // comb inside planeY, so their cache aliases plane 0.
            for (int hh = left; hh < right; ++hh) {
                const int xx = hh - left;
                pyRow0[xx] = planeY(0, hh);
                if (retractedRow) pyRow1[xx] = planeY(1, hh);
                if (oneDRow)      pyRow3[xx] = planeY(3, hh);
                if (ccMaskRow)    pyRow4[xx] = planeY(4, hh);
                if (notchLive) pyRow5[xx] = planeY(5, hh);
            }
            const double *pyR0 = pyRow0.data();
            const double *pyR1 = retractedRow ? pyRow1.data() : pyRow0.data();
            const double *pyR3 = oneDRow      ? pyRow3.data() : pyRow0.data();
            const double *pyR4 = ccMaskRow    ? pyRow4.data() : pyRow0.data();
            const double *pyR5 = notchLive ? pyRow5.data() : pyRow0.data();
            auto planeYc = [&](int plane, int hh) -> double {
                const int xx = hh - left;
                switch (plane) {
                    case 1:  return pyR1[xx];
                    case 3:  return pyR3[xx];
                    case 4:  return pyR4[xx];
                    case 5:  return pyR5[xx];
                    default: return pyR0[xx];
                }
            };
            for (int hh = left; hh < right; ++hh) {
                const int xx = hh - left;
                const int cls = carrierSampleClass(line, hh);
                spRowV[xx] = spLUT_locked[cls];
                cpRowV[xx] = cpLUT_locked[cls];
            }
            const double *spRowP = spRowV.data();
            const double *cpRowP = cpRowV.data();
            const double *resR0 = resRow0.data();
            const double *resR1 = retractedRow ? resRow1.data() : resRow0.data();
            const double *resR3 = oneDRow      ? resRow3.data() : resRow0.data();
            const double *resR4 = ccMaskRow    ? resRow4.data() : resRow0.data();
            const double *resR5 =
                notchLive ? resRow5.data() : resRow0.data();
            if (coarseRow) {
                for (int xx = 0; xx < width; ++xx) {
                    resRow0[xx] = pyR0[xx] - coarseRow[xx];
                    if (retractedRow) resRow1[xx] = pyR1[xx] - coarseRow[xx];
                    if (oneDRow)      resRow3[xx] = pyR3[xx] - coarseRow[xx];
                    if (ccMaskRow)    resRow4[xx] = pyR4[xx] - coarseRow[xx];
                    if (notchLive) resRow5[xx] = pyR5[xx] - coarseRow[xx];
                }
            }
            auto resAt = [&](int plane, int xx) -> double {
                switch (plane) {
                    case 1:  return resR1[xx];
                    case 3:  return resR3[xx];
                    case 4:  return resR4[xx];
                    case 5:  return resR5[xx];
                    default: return resR0[xx];
                }
            };


            // Three-band composition. The selected coarse is the LF platform.
            // The cheap block coarse publishes one value per carrier cycle, so
            // its information-rate Nyquist is fSC/2.  A centered two-cycle
            // (8-sample) mean of comb-minus-platform is therefore deliberately
            // omitted: that band belongs to the platform. Comb supplies the
            // middle between the 8- and 4-sample apertures. Election contestants
            // contribute only the remainder above the legal four-sample
            // aperture, whose response is zero at fSC; the election consequently
            // retains full authority over carrier-rate cross-colour.
            auto candidateResidualAt = [&](int plane, int hh) -> double {
                return resAt(plane, hh - left);
            };
            auto candidateEvenMeanAt = [&](int plane, int h0,
                                           int effectiveWidth) -> double {
                double sum = 0.0;
                const int half = effectiveWidth / 2;
                for (int k = -half; k <= half; ++k) {
                    const int hh = std::clamp(h0 + k, left, right - 1);
                    const double w =
                        (k == -half || k == half) ? 0.5 : 1.0;
                    sum += w * candidateResidualAt(plane, hh);
                }
                return sum / (double)effectiveWidth;
            };
            auto candidateFourMeanAt = [&](int plane, int h0) -> double {
                return candidateEvenMeanAt(plane, h0, 4);
            };
            auto candidatePlatformResidualAt = [&](int plane,
                                                    int h0) -> double {
                return candidateEvenMeanAt(plane, h0, 8);
            };
            auto candidateMiddleAt = [&](int plane, int h0) -> double {
                return candidateFourMeanAt(plane, h0) -
                       candidatePlatformResidualAt(plane, h0);
            };
            auto candidateTopAt = [&](int plane, int h0) -> double {
                return candidateResidualAt(plane, h0) -
                       candidateFourMeanAt(plane, h0);
            };
            auto completeTopAt = [&](auto completeAt,
                                     const double *platform,
                                     int h0) -> double {
                if (!platform)
                    return std::numeric_limits<double>::quiet_NaN();
                const int x0 = h0 - left;
                const double center = completeAt(h0) - platform[x0];
                if (!std::isfinite(center))
                    return std::numeric_limits<double>::quiet_NaN();
                double middle = 0.0;
                for (int k = -2; k <= 2; ++k) {
                    const int hh = std::clamp(h0 + k, left, right - 1);
                    const int xx = hh - left;
                    const double complete = completeAt(hh);
                    if (!std::isfinite(complete))
                        return std::numeric_limits<double>::quiet_NaN();
                    const double w = (k == -2 || k == 2) ? 0.5 : 1.0;
                    middle += w * (complete - platform[xx]);
                }
                return center - middle * 0.25;
            };

            // Carrier-basis cleanliness: 1 - (AC energy explained by the carrier
            // basis). This is a direct waveform measurement, not an aggregate
            // "confidence" whose provenance is hidden.
            // Cycle-integrated over four effective samples on the symmetric
            // half-endpoint support x-2..x+2. The aperture is registered at
            // the pixel it judges and does not flicker at carrier rate. Remove
            // the weighted mean before both projection and norm: DC cannot
            // project onto a complete carrier cycle, and it must not inflate
            // the denominator and make a DC-displaced candidate look clean.
            auto carrierCleanlinessOf = [&](int plane, int h0) -> double {
                double hf5[5], s5[5], c5[5], w5[5];
                double meanHF = 0.0;
                for (int j = 0; j < 5; ++j) {
                    const int k = j - 2;
                    const int hh = std::clamp(h0 + k, left, right - 1);
                    const double w = (j == 0 || j == 4) ? 0.5 : 1.0;
                    hf5[j] = resAt(plane, hh - left);
                    // Index the carrier basis by the grammar sample class, NOT
                    // hh & 3. The locked demod (the basis these LUTs were built
                    // for) uses carrierSampleClass(line, h); a raw-position
                    // index applies a per-line rotation, making cleanliness
                    // line-dependent -> a line-alternating election penalty
                    // (checkerboard) on luma transitions.
                    s5[j] = spRowP[hh - left];
                    c5[j] = cpRowP[hh - left];
                    w5[j] = w;
                    meanHF += w * hf5[j];
                }
                meanHF *= 0.25;
                double dotS = 0.0, dotC = 0.0, nrm = 0.0;
                for (int j = 0; j < 5; ++j) {
                    const double ac = hf5[j] - meanHF;
                    dotS += w5[j] * ac * s5[j];
                    dotC += w5[j] * ac * c5[j];
                    nrm  += w5[j] * ac * ac;
                }
                const double carrierE =
                    (basisSN > 1e-9 ? dotS * dotS / basisSN : 0.0) +
                    (basisCN > 1e-9 ? dotC * dotC / basisCN : 0.0);
                return std::clamp(1.0 - carrierE / (nrm + 1e-9), 0.0, 1.0);
            };

            // Cycle-integrated carrier remaining in raw - candidate Y. This is
            // the amount that candidate would still publish as chroma, measured
            // on the same integer-centred four-effective-sample cycle.
            // It is an amplitude measurement with explicit provenance, not a
            // candidate label or an aggregate quality judgment.
            auto remainingCarrierMagnitudeOf = [&](int plane, int h0) -> double {
                double dotS = 0.0, dotC = 0.0;
                for (int k = -2; k <= 2; ++k) {
                    const int hh = std::clamp(h0 + k, left, right - 1);
                    const double w = (k == -2 || k == 2) ? 0.5 : 1.0;
                    const double residualCarrier =
                        (double)rawLine[hh] - planeYc(plane, hh);
                    dotS += w * residualCarrier * spRowP[hh - left];
                    dotC += w * residualCarrier * cpRowP[hh - left];
                }
                const double carrierE =
                    (basisSN > 1e-9 ? dotS * dotS / basisSN : 0.0) +
                    (basisCN > 1e-9 ? dotC * dotC / basisCN : 0.0);
                return std::sqrt(std::max(0.0, carrierE));
            };

            const bool coarseLines = lockedLumaCacheValid && demodWidth == width;
            struct ProduceYNeighborRows {
                int line = -1;
                bool have = false;
                const quint16 *raw = nullptr;
                const double *cc = nullptr;
                const double *clp = nullptr;
                const float *ret = nullptr;
                const double *coarse = nullptr;
                const lddecode::CarrierAnalysisRecord *analysis = nullptr;
                const CombCarrierGrammar *grammar = nullptr;
            };
            auto makeNeighborRows = [&](int l) {
                ProduceYNeighborRows n;
                n.line = l;
                n.have = coarseLines && l >= firstLine &&
                         l < lastLine && l < demodLines;
                if (!n.have) return n;
                n.raw = rawbuffer.data() + l * fullWidth;
                n.cc = lockedCarrierComposite_line(l);
                n.clp = clpbuffer[srcBuf].pixel[l];
                n.ret = carrierRetracted_line(l);
                n.coarse = coarseFloor_line(l);
                n.analysis = carrierAnalysis_line(l);
                n.grammar = carrierGrammarLine(l);
                return n;
            };
            ProduceYNeighborRows north1 = makeNeighborRows(line - 1);
            ProduceYNeighborRows south1 = makeNeighborRows(line + 1);
            ProduceYNeighborRows north2 = makeNeighborRows(line - 2);
            ProduceYNeighborRows south2 = makeNeighborRows(line + 2);

            // Regime-sensitive vertical anchors. Frame/progressive evidence
            // uses adjacent picture lines (±1); field/interlace evidence uses
            // same-field neighbours (±2). FVF can vary this per pixel.
            const bool variantFrameRegime =
                configuration.twoDVariant ==
                    Comb::Configuration::TwoDVariant::FrameAAdaptiveIQ ||
                configuration.twoDVariant ==
                    Comb::Configuration::TwoDVariant::FrameBDirectIQ;
            const bool variantFvf =
                configuration.twoDVariant ==
                    Comb::Configuration::TwoDVariant::FieldVsFrame;
            const bool haveFvfLine =
                line >= 0 && line < (int)fvfMetrics.size() &&
                (int)fvfMetrics[line].size() >= width;
            auto verticalStepAt = [&](int xi) {
                // In Y the carrier phase no longer matters, so vertical
                // evidence uses the IMMEDIATELY adjacent picture line whenever
                // the cadence says the frame is progressive (telecine) --
                // spacing was a carrier-relation concern, not a luma one.
                if (carrierFrameVerticalAllowed(line))
                    return 1;
                if (variantFvf && haveFvfLine)
                    return fvfMetrics[line][xi].frameModel ? 1 : 2;
                return variantFrameRegime ? 1 : 2;
            };
            auto northRowsForStep = [&](int step) -> const ProduceYNeighborRows& {
                return (step == 1) ? north1 : north2;
            };
            auto southRowsForStep = [&](int step) -> const ProduceYNeighborRows& {
                return (step == 1) ? south1 : south2;
            };
            auto legalRel = [&](const CombCarrierGrammar *gNbr) {
                if (!grammarLine || !gNbr)
                    return false;
                const auto rel = lddecode::carrierGrammarSignedPhaseRelation(
                    grammarLine, left, gNbr, left);
                return rel == lddecode::CarrierPhaseRelation::Same ||
                       rel == lddecode::CarrierPhaseRelation::Opposite;
            };

            // Robust top band at a neighbour pixel: median of that pixel's
            // complete luma planes after removing its one selected coarse and
            // the centered four-sample middle band.
            // Returns false where the neighbour lacks a usable coarse.
            auto neighborHFAt = [&](const quint16 *rawP, const double *ccP,
                                    const double *clpP,
                                    const float *retP,
                                    const double *coaP, int hh, double &out) -> bool {
                if (!rawP || !coaP) return false;
                double v[2]; int n = 0;
                const double identityTol = 1e-6 * irescale;
                auto appendUnique = [&](double value) {
                    if (!std::isfinite(value)) return;
                    for (int i = 0; i < n; ++i)
                        if (std::fabs(v[i] - value) <= identityTol)
                            return;
                    v[n++] = value;
                };
                auto combComplete = [&](int hk) {
                    const int xk = hk - left;
                    const double c = ccP ? ccP[xk]
                                         : (clpP ? clpP[hk] : 0.0);
                    return std::isfinite(c)
                        ? (double)rawP[hk] - c
                        : std::numeric_limits<double>::quiet_NaN();
                };
                if (ccP || clpP)
                    appendUnique(completeTopAt(combComplete, coaP, hh));
                if (retP) {
                    auto retractedComplete = [&](int hk) {
                        return (double)retP[hk - left];
                    };
                    appendUnique(completeTopAt(retractedComplete, coaP, hh));
                }
                if (n == 0) return false;
                if (n == 1) out = v[0];
                else out = 0.5 * (v[0] + v[1]);
                return true;
            };

            // ld-disc-stacker primitives (mode 3/6), specialised for the small
            // candidate set. medoid = robust self-center (agreement fast path
            // and the no-neighbour fallback); closest reconciles a nomination
            // to it. The full election scoring lives inline below.
            auto medoidD = [](const double *a, int n) -> double {
                if (n == 1) return a[0];
                if (n == 2) return 0.5 * (a[0] + a[1]);
                int best = 0; double bestTot = 1e300;
                for (int i = 0; i < n; ++i) {
                    double t = 0.0;
                    for (int j = 0; j < n; ++j) t += std::fabs(a[i] - a[j]);
                    if (t < bestTot) { bestTot = t; best = i; }
                }
                return a[best];
            };
            auto closestD = [](const double *a, int n, double target) -> double {
                double best = a[0];
                for (int i = 1; i < n; ++i)
                    if (std::fabs(target - a[i]) < std::fabs(target - best))
                        best = a[i];
                return best;
            };

            for (int h = left; h < right; ++h) {
                const int xi = h - left;
                const double rawH = (double)rawLine[h];

                // Per-sample arm of the certified early-out. Reached only on
                // PARTIALLY certified lines -- fully certified ones never
                // enter this block. Placed before any candidate work so a
                // certified sample costs one compare and one subtraction.
                if (certLineActive && std::isfinite(certExactRow[h])) {
                    Y[h] = rawH - (double)certExactRow[h];
                    continue;
                }

                // combY -- senior comb candidate.
                double combY;
                if (carrierComp) {
                    const double c = carrierComp[xi];
                    combY = std::isfinite(c)
                        ? rawH - c
                        : (std::isfinite(clpLine[h])
                               ? rawH - clpLine[h]
                               : rawH);
                } else {
                    const double c = clpLine[h];
                    combY = std::isfinite(c) ? rawH - c : rawH;
                }

                double ccReturn = ccMaskRow
                    ? std::clamp((double)ccMaskRow[xi], 0.0, 1.0)
                    : 0.0;
                // Fact-audit on the Y-RETURN duty only (regional, smooth;
                // built in splitIQlocked pass 2 from the nearest covered
                // facts grading the detector's claims).
                if (ccReturn > 0.0 && !ccAuditW_flat.empty() &&
                    ccAuditNX > 0) {
                    const double ry = std::clamp(
                        (line - 16.0) / 32.0, 0.0,
                        (double)(ccAuditNY - 1) - 1e-6);
                    const double rx = std::clamp(
                        (xi - 64.0) / 128.0, 0.0,
                        (double)(ccAuditNX - 1) - 1e-6);
                    const int r0 = (int)ry, c0 = (int)rx;
                    const double wy = ry - r0, wx = rx - c0;
                    const int r1 = std::min(r0 + 1, ccAuditNY - 1);
                    const int c1 = std::min(c0 + 1, ccAuditNX - 1);
                    auto A = [&](int rr, int cc2) {
                        return (double)ccAuditW_flat[
                            (size_t)rr * ccAuditNX + cc2];
                    };
                    ccReturn *= (1 - wy) * ((1 - wx) * A(r0, c0) +
                                            wx * A(r0, c1)) +
                                wy * ((1 - wx) * A(r1, c0) +
                                      wx * A(r1, c1));
                }
                // Iceberg carries its own current-frame membership licence.
                // It joins the return policy after the legacy detector audit;
                // the audit grades that detector, not the independent matched-
                // luma construction.
                if (!coveredFrame && icebergYRow && icebergWRow &&
                    std::isfinite((double)icebergYRow[xi])) {
                    const double iceReturn = icebergPolicyWeight * std::clamp(
                        (double)icebergWRow[xi], 0.0, 1.0);
                    ccReturn = std::max(ccReturn, iceReturn);
                }
                // A mask buffer exists for the whole frame whenever the feature
                // is enabled. Do not let that allocation alone activate the
                // larger election at an unmarked pixel when no retracted luma
                // plane otherwise requires it.
                if (!retractedRow && ccReturn <= 0.0) {
                    Y[h] = combY;
                    continue;
                }

                // Coarse floor unavailable -> emit combY.
                if (!coarseRow) {
                    Y[h] = combY;
                    continue;
                }
                const double coarse = coarseRow[xi];
                const double combFour = candidateFourMeanAt(0, h);
                // The comb's platform-residual band (the centred 8-sample mean
                // of comb-minus-platform) must be carried, not dropped.  It was
                // omitted on the theory that the selected coarse already owns
                // everything below the 8-sample aperture, but the coarse is a
                // raster-aligned BLOCK mean -- piecewise constant across each
                // 4fSC cycle -- not a centred 8-sample mean.  Its block-pitch
                // stairstep is exactly what this band corrects, so discarding
                // it published that stairstep as luma: +20% line-alternation at
                // Gilgol Beach chroma transitions (the bead crawl) and inflated
                // GGV false saturation, because the discarded remainder is
                // carrier-band.  Carrying it makes the identity explicit --
                //   Y = combY + (electedTop - combTop)
                // -- which IS the stated design: coarse owns LF, comb owns the
                // middle and the provisional top, and the election swaps only
                // the top band.
                const double combPlatform = candidatePlatformResidualAt(0, h);
                const double combMiddle = combFour - combPlatform;
                const double combTop0 = candidateResidualAt(0, h) - combFour;
                auto reconstructTop = [&](int plane, double top) {
                    // Non-comb winners keep their own four-mean so the top
                    // band is rebuilt in the same candidate geometry that won.
                    const double fourMean =
                        (plane == 0)
                            ? (combMiddle + combPlatform)
                            : candidateFourMeanAt(plane, h);
                    return coarse + fourMean + top;
                };
                const int verticalStep = verticalStepAt(xi);
                const ProduceYNeighborRows &northRows =
                    northRowsForStep(verticalStep);
                const ProduceYNeighborRows &southRows =
                    southRowsForStep(verticalStep);
                const bool requireVerticalCarrierRelation =
                    (verticalStep == 2);
                const bool relLegalN =
                    !requireVerticalCarrierRelation ||
                    legalRel(northRows.grammar);
                const bool relLegalS =
                    !requireVerticalCarrierRelation ||
                    legalRel(southRows.grammar);

                auto feasible = [&](double y) {
                    const double c = rawH - y;
                    return c <= maxCarrierAmpSamples && c >= -maxCarrierAmpSamples;
                };

                // Roster with structural feasibility DQ. Coherent comb is the
                // senior hypothesis; 1D replaces it only if comb is infeasible.
                // Retracted Y may join as the one independent base challenger.
                // Top-band values; name retained locally. Three base planes
                // can be live at once (comb-or-1D, retracted, notch) and
                // addBaseCandidate is unbounded by design, so keep the margin
                // the roster had before the notch joined.
                double candY[4];
                int    candPlane[4];
                int    nCand = 0;
                const double identityTol = 1e-6 * irescale;
                auto addBaseCandidate = [&](double completeY, int plane) {
                    const double y = (plane == 0)
                        ? combTop0 : candidateTopAt(plane, h);
                    if (!std::isfinite(completeY) || !feasible(completeY) ||
                        !std::isfinite(y))
                        return;
                    // Population statistics describe distinct numerical
                    // hypotheses, not the number of pipelines which happened
                    // to publish one. The plane provenance remains on the
                    // retained value; duplicate hypotheses must not count
                    // coherent Y twice.
                    for (int k = 0; k < nCand; ++k)
                        if (std::fabs(candY[k] - y) <= identityTol)
                            return;
                    candY[nCand] = y;
                    candPlane[nCand] = plane;
                    ++nCand;
                };
                const bool combOK = std::isfinite(combY) && feasible(combY);

                if (combOK) {
                    addBaseCandidate(combY, 0);
                }

                // ABSTENTION IS AN ANSWER, and it is checked at the door.
                // A non-finite retracted sample is the witness declaring it
                // has no product here (see the publication's abstention
                // rung). The old conduct substituted combY and ran the whole
                // admission on it, seating a SECOND copy of comb under the
                // witness's plane number -- identity dedup usually collapsed
                // it, so the damage was quiet, but a candidate must never
                // answer with another candidate's value. Now the seat simply
                // goes unfilled; the election is downstream of the comb, so
                // the roster is the fallback and no candidate needs an
                // internal retreat.
                if (retractedRow && std::isfinite((double)retractedRow[xi])) {
                    // Evidence admission for retractedY (raw - combedCarrier),
                    // the leg that keeps near-carrier HF luma the comb strips.
                    //
                    // A low four-view residualTrust marks a conflicted fit —
                    // but that conflict is present for BOTH broadband texture
                    // noise (beach: retracted is junk, comb should stand) AND
                    // real near-carrier periodic structure (Borg-cube grid:
                    // retracted carries the detail comb destroyed).  Trust
                    // alone cannot separate them, so it must not hard-DQ:
                    // that is a fit-quality measure vetoing geometry, which
                    // the election forbids.
                    //
                    // The separator is SPATIAL COHERENCE. Real structure
                    // agrees with its regime-appropriate vertical neighbours:
                    // frame/progressive uses adjacent picture lines (±1),
                    // field/interlace uses same-field partners (±2). Per-pixel
                    // texture noise agrees with neither. So: a clean fit is
                    // admitted outright; a conflicted fit is admitted only when
                    // its HF is corroborated by a neighbour. Feasibility
                    // remains the only true DQ.
                    const double r = (double)retractedRow[xi];
                    const double ry = r;
                    bool retractedAdmitted =
                        retractedAdmitAll ||
                        !analysisRow ||
                        (analysisRow[xi].parallax.residualValid &&
                         analysisRow[xi].parallax.residualTrust >= 0.5f);
                    // Schedule-illegality admits by LAW, not corroboration:
                    // an alien ±2 partner means this pixel's carrier-band
                    // energy is raw-identical where the schedule demands
                    // inversion — structurally not carrier, hence luma.
                    // retractedY is the plane that keeps it as luma, and it
                    // is exactly the line-pitch detail (Borg-cube grid) that
                    // a same-field agreement check can never certify, because
                    // that detail IS the same-field disagreement.
                    if (!retractedAdmitted && alienRow && alienRow[xi] > 0.5f)
                        retractedAdmitted = true;
                    // Same law from the analysis-time registration, which
                    // adds the FRAME axis (static line-decorrelated detail
                    // the ±2 tests cannot reach).
                    if (!retractedAdmitted && analysisRow &&
                        analysisRow[xi].scheduleConformance ==
                            lddecode::CarrierScheduleConformance::ScheduleIllegal)
                        retractedAdmitted = true;
                    if (!retractedAdmitted && retractedAdmitSpatial &&
                        retractedRow && std::isfinite(r)) {
                        auto currentRetracted = [&](int hh) {
                            return (double)retractedRow[hh - left];
                        };
                        const double rTop = completeTopAt(
                            currentRetracted, coarseRow, h);
                        // Frame/progressive vertical checks are luma-image
                        // checks and do not require carrier grammar. Field/
                        // interlace checks use ±2 same-field partners; there
                        // the schedule must certify phase-comparable lines
                        // before matched HF is accepted as structure.
                        if (relLegalN && northRows.ret && northRows.coarse) {
                            auto northRetracted = [&](int hh) {
                                return (double)northRows.ret[hh - left];
                            };
                            const double nTop = completeTopAt(
                                northRetracted, northRows.coarse, h);
                            if (std::isfinite(nTop) &&
                                std::fabs(rTop - nTop) <= inlierTol)
                                retractedAdmitted = true;
                        }
                        if (!retractedAdmitted && relLegalS &&
                            southRows.ret && southRows.coarse) {
                            auto southRetracted = [&](int hh) {
                                return (double)southRows.ret[hh - left];
                            };
                            const double sTop = completeTopAt(
                                southRetracted, southRows.coarse, h);
                            if (std::isfinite(sTop) &&
                                std::fabs(rTop - sTop) <= inlierTol)
                                retractedAdmitted = true;
                        }
                    }
                    if (retractedAdmitted)
                        addBaseCandidate(ry, 1);
                }
                if (!combOK && oneDRow) {
                    const double o = oneDRow[xi];
                    if (std::isfinite(o)) {
                        const double y1 = rawH - o;
                        addBaseCandidate(y1, 3);
                    }
                }

                // Fixed-kernel notch. No admission test exists here, and none
                // is needed: there is no estimate to vet, so feasibility --
                // the only true DQ -- is the entire gate. That law already
                // does real work on this plane. The notch's implied carrier
                // is raw - notchY = sin^2(w)*raw, i.e. the WHOLE local fSC
                // band, so wherever that exceeds the structural amplitude
                // ceiling the candidate is claiming a carrier no legal
                // carrier could be, and feasible() removes it with no special
                // case. It joins as a BASE candidate: unlike returned it is
                // not derived from comb, so it is entitled to move the
                // population statistics the base set defines. notchLive,
                // not notchCandidate: covered frames are ceded whole (see
                // the declaration).
                if (notchLive)
                    addBaseCandidate(planeYc(5, h), 5);

                // Returned Y is derived from combY, so it is a selectable
                // challenger but not another independent observation when the
                // base candidates establish their center, subset, or scoring
                // scales. It joins only after those quantities are fixed.
                // VERDICT IS THE FACT, as a VALUE (detectors-trade-for-
                // truth): on covered samples the returned candidate IS
                // certified luma, raw - ex. The mask formalism (m·C) can
                // only correct ALONG C; the true correction leak = C - ex
                // opposes C on roughly half the fired samples (referee:
                // helped 45% under the magnitude-only override). A fact
                // enters as the value, or not at all.
                static const bool ccFactsY = []{
                    const char *e = std::getenv("LDCD_CC_FACTS");
                    return !(e && std::atoi(e) == 0);
                }();
                const bool haveFactY = ccFactsY && dsExactRow &&
                                       std::isfinite(dsExactRow[h]);
                // Band-revoked residue with an affirmed luma claim
                // (splitIQlocked application stage): joins the returned
                // candidate as a VALUE, so the election's hull and
                // feasibility machinery contest it like any other claim.
                // Facts outrank it (the fact branch is already exact).
                const float *bandRow = haveFactY
                    ? nullptr : bandResidueY_line(line);
                const double bandVal =
                    (bandRow && std::isfinite((double)bandRow[xi]))
                        ? (double)bandRow[xi] : 0.0;
                // NOTE (open, 2026-08-02): this admission formula and the
                // value the election actually samples (planeY plane 4) are
                // NOT the same expression -- admission tests
                // comb + ccReturn*(raw - comb) + bandVal while the election
                // samples the delta form comb + m*(retracted - comb). The
                // roster therefore judges feasibility on a claim it will not
                // evaluate. Left as found pending a measured decision; do
                // not "fix" it by clamping, which was tried and withdrawn
                // (see planeY plane 4).
                const double returnedY = haveFactY
                    ? rawH - (double)dsExactRow[h]
                    : planeYc(4, h) + bandVal;
                const bool returnedFeasible =
                    (haveFactY || ccReturn > 0.0 || bandVal != 0.0) &&
                    std::isfinite(returnedY) && feasible(returnedY);

                if (nCand == 0) {
                    // No legal top band exists here. Publishing the illegal
                    // one truncated to the hull edge (Y = raw -/+ maxAmp) is
                    // not a rejection: it moves Y TOWARD raw, so it keeps a
                    // signed slice of the carrier waveform as luma. raw
                    // alternates about the luma at carrier rate, so that
                    // slice renders as a 2fSC speckle of up to ~20 IRE at
                    // exactly the near-peak samples where the estimate broke.
                    // Reject instead: keep the bands below the top aperture,
                    // which are 4- and 8-sample centred means and therefore
                    // cancel carrier by construction, and publish no top.
                    Y[h] = reconstructTop(0, 0.0);
                    continue;
                }
                // Feasibility is the only DQ (stated doctrine, now enforced).
                // The former inlier gate (|cand - center| <= inlierTol around
                // a medoid that ties to comb on a 2-candidate roster) was a
                // second centrist filter: it ejected the detailed candidate at
                // exactly the pixels where it differed enough to matter, so
                // the election's scoring never saw the contest. Outliers are
                // bounded by the scoring itself -- a candidate far from every
                // adjacent neighbour pays its full distance as cost.
                //
                // A prior "base agreement" early-out used to commit outright
                // to the medoid-nearest candidate's own plane whenever the
                // base tops sat within agreeTol of each other -- bypassing
                // the blend on a coarser, differently-scaled metric than the
                // one the blend actually scores on (user, 2026-07-28: "the
                // medoid isn't that great, we shouldn't allow it to govern").
                // Removed: every base candidate now always reaches the cost
                // loop and the confidence-alpha blend below, agreeing or not.
                int inIdx[4];
                int nIn = 0;
                for (int k = 0; k < nCand; ++k)
                    inIdx[nIn++] = k;
                // The named cross-colour mask, rather than distance from the
                // base center, admits returned Y. A distance gate here removes
                // exactly the strong HF that the return exists to recover.
                const bool returnedAdmitted = returnedFeasible;
                // ELECTION BYPASS (diagnostic A/B only, LDCD_ELECT_BYPASS=1;
                // measurement instrument for the chroma-lean hunt, 2026-08-01).
                // Forces the top band to the PURE COMB candidate, so emitted Y
                // is the comb complement everywhere comb is feasible and the
                // election's edge-population departure (measured 3.5 IRE RMS on
                // ~20% of edge pixels) cannot express. If the locked-vs-bucket
                // chroma lean (I -0.39 / Q -0.20 px) collapses toward the
                // bucket referee under this, the election's complement
                // departure is the source; if it survives, the election is
                // exonerated and the bias is upstream of the roster.
                // NOT a shipping mode: it discards every other candidate.
                if (electBypass && combOK) {
                    Y[h] = reconstructTop(0, candidateTopAt(0, h));
                    continue;
                }

                if (nIn == 1 && !returnedAdmitted) {
                    Y[h] = reconstructTop(candPlane[inIdx[0]], candY[inIdx[0]]);
                    continue;
                }

                // Inlier HF set + per-inlier carrier-basis cleanliness. This is
                // a cautionary term, not the positive reason to select HF.
                // Sized for the full roster: three BASE candidates
                // (comb-or-1D, retracted, notch) plus the derived return.
                double inHF[4], inCarrierCleanliness[4];
                double inCrossColorReturnEvidence[4] = {0.0, 0.0, 0.0, 0.0};
                for (int k = 0; k < nIn; ++k) {
                    inHF[k] = candY[inIdx[k]];
                    inCarrierCleanliness[k] =
                        carrierCleanlinessOf(candPlane[inIdx[k]], h);
                }
                const int baseNIn = nIn;
                auto planeForTop = [&](double top) {
                    for (int k = 0; k < baseNIn; ++k)
                        if (inHF[k] == top)
                            return candPlane[inIdx[k]];
                    for (int k = baseNIn; k < nIn; ++k)
                        if (inHF[k] == top)
                            return 4;
                    return 0;
                };

                // Single self-anchor: medoid of the BASE inlier HFs (mode 6).
                // The derived return may be selected, but does not move this
                // population statistic.
                const double selfAnchor = medoidD(inHF, baseNIn);
                if (returnedAdmitted) {
                    inHF[nIn] = candidateTopAt(4, h);
                    inCarrierCleanliness[nIn] =
                        carrierCleanlinessOf(4, h);
                    ++nIn;
                }

                // Let the named cross-colour evidence affect scoring according
                // to what each candidate actually does. The comb plane defines
                // zero return. A candidate earns only the cycle-integrated
                // carrier reduction it delivers relative to comb, and never
                // more than either the measured false-colour amount or the
                // explicit policy cap reported by the tunable.
                // Retracted Y can therefore receive this evidence when it
                // already outperforms nominal returned
                // Y; a label cannot win an advantage its samples did not earn.
                const double combRemainMag0 =
                    remainingCarrierMagnitudeOf(0, h);
                if (ccReturn > 0.0) {
                    const double combCarrierMagnitude = combRemainMag0;
                    const double measuredFalseCarrier =
                        ccReturn * combCarrierMagnitude;
                    const double crossColorReturnCap =
                        std::max(0.0,
                            configuration.tunables
                                .PRODUCE_Y_CC_RETURN_EVIDENCE_CAP_IRE) *
                        irescale;
                    for (int k = 0; k < nIn; ++k) {
                        const int plane = (k < baseNIn)
                            ? candPlane[inIdx[k]] : 4;
                        // The notch is excluded by construction, not by
                        // policy. This term pays for carrier reduction
                        // measured RELATIVE TO COMB, and the notch's null
                        // removes the entire fSC band at every pixel in the
                        // frame -- so it would collect the maximum award
                        // everywhere for an arithmetic identity rather than
                        // for anything it discovered. A candidate must not
                        // earn evidence its samples did not earn, and a
                        // constant is not evidence.
                        if (plane == 5) {
                            inCrossColorReturnEvidence[k] = 0.0;
                            continue;
                        }
                        const double deliveredReduction = std::max(
                            0.0,
                            combCarrierMagnitude -
                                ((plane == 0) ? combRemainMag0
                                 : remainingCarrierMagnitudeOf(plane, h)));
                        inCrossColorReturnEvidence[k] = std::min(
                            crossColorReturnCap,
                            std::min(measuredFalseCarrier,
                                     deliveredReduction));
                    }
                }

                // Four independent image neighbours (N/S at regime-sensitive
                // vertical step, E/W at ±1 sample). Keep them separate: a line
                // or edge may continue in one direction while legitimately
                // crossing a transition in another.
                //
                // NOTE on twin pinning (user direction, 2026-07-28): a
                // distance-scaling weight on the pinned N/S testimony was
                // built and measured INERT here -- it only re-ranks which
                // neighbour anchors the boost, and cannot move the blend
                // toward the pinned value when no candidate already sits
                // near it (referee unchanged; comp-vs-bracket residual
                // 1.93->1.96 IRE). Pinning needs a value-level lever;
                // anchorLinePinned() is the infrastructure awaiting it.
                double dirHF[4], dirImageHF[4]; int nDir = 0;
                auto appendDirection = [&](const quint16 *rawP,
                                           const double *ccP,
                                           const double *clpP,
                                           const float *retP,
                                           const double *geometryFloor,
                                           const double *dcFloor,
                                           int hh) {
                    if (nDir >= 4 || !dcFloor) return;
                    double geometryHF, imageHF;
                    if (neighborHFAt(rawP, ccP, clpP, retP,
                                     geometryFloor, hh, geometryHF) &&
                        neighborHFAt(rawP, ccP, clpP, retP,
                                     dcFloor, hh, imageHF)) {
                        dirHF[nDir] = geometryHF;
                        dirImageHF[nDir] = imageHF;
                        ++nDir;
                    }
                };
                appendDirection(northRows.raw, northRows.cc, northRows.clp,
                                northRows.ret,
                                northRows.coarse, northRows.coarse, h);
                appendDirection(southRows.raw, southRows.cc, southRows.clp,
                                southRows.ret,
                                southRows.coarse, southRows.coarse, h);
                if (h - 1 >= left)
                    appendDirection(rawLine, carrierComp, clpLine,
                                    retractedRow,
                                    coarseRow, coarseRow, h - 1);
                if (h + 1 < right)
                    appendDirection(rawLine, carrierComp, clpLine,
                                    retractedRow,
                                    coarseRow, coarseRow, h + 1);

                // ---- Election scoring: neighbour boost + legality. ----
                // (2026-07-26 redesign, user-directed.)
                //
                // The anchor machinery (medoid self-anchor averaged with an
                // inlier-MEAN neighbour anchor) was a CENTRIST instrument:
                // after boldness was removed (correctly -- it selected
                // carrier) no pro-detail term remained, and softness became
                // the election's default. The elected output left the sharp
                // candidates' quality off to the side.
                //
                //   * NEIGHBOUR BOOST, not anchor. Each candidate's base cost
                //     is its distance to the NEAREST immediately adjacent
                //     neighbour top (E/W +-1; N/S +-1 wherever the cadence
                //     allows frame-vertical -- in Y, carrier phase no longer
                //     matters, so there is no spacing). Proximity to a real
                //     neighbour value lowers cost; no mean is formed anywhere.
                //   * LEGALITY is the pro-detail term, replacing boldness.
                //     Retained HF above comb's top earns preference only in
                //     proportion to the per-pixel PROOF that the energy
                //     cannot be carrier (carrierIllegalProof on the
                //     registered conformance). Detail wins only where it is
                //     provably luma; dot crawl is provably carrier and earns
                //     nothing. Legal-schedule errors remain -- some errors
                //     are legal -- but the rest are cut down.
                //   * Carrier-basis cleanliness stays as the capped caution.
                //   * IMPULSE SENIORITY: at star/impulse pixels (the single
                //     lumaImpulseRisk producer, kept as-is -- its job is
                //     rescuing stars mangled by comb) retracted outranks comb
                //     by a small bias.
                //   * HIGH-CHROMA DEMOTION: retracted is demoted where the
                //     measured chroma is strong (the tractor-beam checkers),
                //     until the retracted product is clean there.
                const double imagePrefCap =
                    std::max(0.0, configuration.tunables.PRODUCE_Y_HF_IMAGE_PREFERENCE_IRE)
                    * irescale;
                const double proxTol =
                    std::max(0.5, configuration.tunables.PRODUCE_Y_HF_CONTINUATION_IRE)
                    * irescale;
                constexpr double kImpulseRetractedBiasIRE = 1.0;
                constexpr double kHighChromaSoftIRE      = 10.0;
                constexpr double kHighChromaHardIRE      = 20.0;
                constexpr double kHighChromaDemoteIRE    = 1.5;

                // Capped-caution reference: median cleanliness of the base set.
                //
                // The notch is excluded from the REFERENCE POPULATION, not
                // from the penalty. Its residual has zero carrier-basis
                // energy by construction, so its cleanliness is pinned at 1.0
                // as a structural constant rather than a measurement of this
                // pixel. Leaving it in drags the median up and charges every
                // OTHER candidate for the notch's arithmetic. Its own penalty
                // then falling to zero is honest -- it genuinely carries no
                // carrier residue -- and that honesty is precisely why it
                // cannot serve as the bar. With no measured reference left,
                // medianW = 0 makes the caution inert rather than inventing a
                // threshold out of a constant.
                double sw[4];
                int refN = 0;
                for (int i = 0; i < baseNIn && i < nIn; ++i) {
                    if (candPlane[inIdx[i]] == 5) continue;
                    sw[refN++] = inCarrierCleanliness[i];
                }
                double medianW = 0.0;
                if (refN > 0) {
                    std::sort(sw, sw + refN);
                    medianW = sw[refN / 2];
                }

                const double combTopHere = combTop0;
                const double illegalProof = analysisRow
                    ? lddecode::carrierIllegalProof(
                          (double)analysisRow[xi].carrierConformance,
                          (double)analysisRow[xi].conformanceSupportFraction)
                    : 0.0;
                const double impulseT = attribRow
                    ? std::clamp(attribRow[xi].facts.lumaImpulseRisk, 0.0, 1.0)
                    : 0.0;
                // High chroma means LEGAL carrier energy. The raw remaining-
                // carrier magnitude also counts confiscated illegal luma (the
                // grid), which would demote retracted exactly where it should
                // win; the illegal-proof share is excluded.
                const double chromaT = std::clamp(
                    (combRemainMag0 * invIreScale -
                     kHighChromaSoftIRE) /
                        (kHighChromaHardIRE - kHighChromaSoftIRE),
                    0.0, 1.0) * (1.0 - illegalProof);

                // Darkest-choice penalty: comb's weak impulse highlights.
                // At a genuine luma peak comb rounds AND darkens; retracted
                // keeps the peak, so the roster's DARKEST candidate is the
                // wrong one there, in proportion to the evidence that the
                // peak is real luma (illegal-proof or the star/impulse
                // channel) and bright (raw's own top band) -- and it pays
                // NOTHING where the energy is legal carrier, where the
                // most-subtracted candidate is usually the correct one
                // (otherwise this term would be reverse-boldness and
                // re-select carrier).
                //
                // Weight-space reform (user, 2026-07-28, at the beam
                // highlight join): the argmin-era form charged the gate
                // TIMES the roster spread -- big enough to always flip the
                // winner, and safe when only ordering mattered. Feeding
                // alpha, spread is an inter-candidate DISTANCE modulating
                // the mixing proportion, and at a highlight over dense
                // chroma that distance is the comb/retracted divergence
                // itself -- distance x difference in the render, the exact
                // product the blend-weight doctrine forbids, visible as the
                // join band. The evidence gate is already normalised, so it
                // enters as a multiplicative weight factor (1 - gate) on
                // the darkest candidate instead: smooth demotion along the
                // evidence ramp, a true veto at full proof, and the
                // candidates' distances nowhere in the weight.
                constexpr double kPeakSoftIRE = 8.0;
                constexpr double kPeakHardIRE = 20.0;
                int darkestIdx = 0;
                double darkestVetoGate = 0.0;
                if (baseNIn > 1) {
                    for (int k = 1; k < baseNIn; ++k) {
                        if (inHF[k] < inHF[darkestIdx]) darkestIdx = k;
                    }
                    auto rawComplete = [&](int hk) {
                        return (double)rawLine[hk];
                    };
                    const double rawTopIRE = std::fabs(
                        completeTopAt(rawComplete, coarseRow, h)) * invIreScale;
                    const double peakT = std::clamp(
                        (rawTopIRE - kPeakSoftIRE) /
                            (kPeakHardIRE - kPeakSoftIRE),
                        0.0, 1.0);
                    darkestVetoGate =
                        std::max(illegalProof, impulseT) * peakT;
                }

                double resultHF = inHF[0];
                // The Y-election band cede is REMOVED (user, 2026-07-28).
                // It hard-picked the comb candidate inside chromaBoundaryBand
                // to kill the bikini-diagonal per-column winner interleave
                // (2026-07-27), an artifact of the winner-take-all era that
                // the confidence-alpha blend obviates: alpha varies smoothly
                // where candidates tie, so there is no per-column flip left
                // to suppress. And the cede was measured blanketing the
                // entire cube face (band seeds fire on dense alternating
                // detail everywhere -- thin struts against shadow meet the
                // DifferentRegion definition), silencing the blend and the
                // retracted candidate exactly where retracted is the solid
                // image (probe 2026-07-28: bandCede 99-100% of face px,
                // comb top D2-RMS letter-alternating 7.8/4.9, retracted
                // stable ~6.2). The band plane stays published (Field B's
                // own settled cede is upstream and unaffected).
                // Confidence-alpha blend (user direction): the per-pixel
                // WINNER flip between candidates -- comb/retracted measured
                // 46/54 at boundary teeth, and frame-to-frame flips render
                // as strobing on detail -- is the same per-pixel-decision
                // artifact family as the Field B beading. The election's
                // cost terms stay exactly as they are (they carry the image
                // shaping: neighbour boost, legality, impulse seniority,
                // darkest-peak penalty), but they now shape ALPHA instead of
                // electing one winner: Y = sum w_k * reconstruct_k with
                // w = exp(-(cost-min)/tau). A decisive cost gap still yields
                // alpha ~= 1 (tau matched to the cost scale), so commitment
                // survives where evidence is clear and blending concentrates
                // where ambiguity -- and hence the strobing -- lives.
                // Vetoes remain binary and upstream: admission and the
                // feasibility DQ.
                double blendNum = 0.0, blendDen = 0.0;
                constexpr double kBlendTauIRE = 0.75;
                const double blendTau = kBlendTauIRE * irescale;
                double costs[4];
                // ELECTION v2 (user-approved shape, 2026-08-02):
                //  1. HF-surplus reward made CANDIDATE-NEUTRAL -- it no
                //     longer needs illegalProof to vouch for the site.
                //     Truth-validated as a general score, never a selector:
                //     argmax|HF| is the truth-best candidate 46-48% vs 33%
                //     chance across cube/title/bridge, yet a pure max-HF
                //     pick LOSES to the election (0.50 vs 0.46). A score,
                //     not a rule.
                //  2. The named plane==1 (retracted) adjustment retires --
                //     the neutral HF term is what it was approximating.
                //  3. The return-evidence term stops being multiplied by
                //     proximity01 (see above).
                //  4. Blend moves from an absolute-difference exponential
                //     to a ratio/log curve, so a candidate that is much
                //     better pedestals regardless of the local cost scale.
                // BITMASK so the four changes can be bisected:
                //   1 = candidate-neutral HF-surplus reward
                //   2 = retire the named plane==1 (retracted) term
                //   4 = drop proximity01 scaling on return evidence
                //   8 = ratio/log blend curve
                //  15 = all four (the bundle first measured)
                //
                // DEFAULT 7 (= 1|2|4, everything except the ratio blend),
                // promoted 2026-08-02 on a truth-graded bisection, held out
                // (CERT_1D=0 + CHROMA_FACTS=0 + native retraction), cube and
                // title, frame-wide elected error vs certified luma:
                //     HF neutral   -0.74% / -0.22%
                //     retire named +0.06% / -1.29%
                //     un-gate ev   -1.88% / -0.64%   <- largest single win
                //     ratio blend  +3.33% / +2.77%   <- the only loser
                //     all four     +0.41% / +0.14%   (blend cancels the rest)
                //     1|2|4        -2.56% / -2.14%   <- promoted
                // At return-delivering sites 1|2|4 is -11.4% / -4.4%.
                // The ratio/log blend is REJECTED BY MEASUREMENT: pedestaling
                // harder only pays where the pedestaled candidate is right,
                // and return is the best candidate at its delivering sites on
                // the cube but NOT on the title (1.82 vs a 1.70 blend). A
                // sharper curve cannot fix a score that cannot tell those two
                // regimes apart. LDCD_ELECT_V2=8 keeps it available for A/B.
                static const int electV2 = []{
                    const char *e = std::getenv("LDCD_ELECT_V2");
                    return e ? std::atoi(e) : 7;
                }();
                const bool ev2Hf    = (electV2 & 1) != 0;
                const bool ev2Named = (electV2 & 2) != 0;
                const bool ev2Prox  = (electV2 & 4) != 0;
                const bool ev2Blend = (electV2 & 8) != 0;
                static const double kHfBaseShare = []{
                    const char *e = std::getenv("LDCD_HF_BASE");
                    return e ? std::atof(e) : 0.40;
                }();
                {
                    double bestCost = 1e300;
                    for (int k = 0; k < nIn; ++k) {
                        const int plane =
                            (k < baseNIn) ? candPlane[inIdx[k]] : 4;
                        // Neighbour boost: nearest adjacent top. With no
                        // usable neighbour, the medoid stands in as the sole
                        // fallback reference.
                        double nd = 1e300;
                        for (int d = 0; d < nDir; ++d) {
                            const double dd = std::fabs(inHF[k] - dirHF[d]);
                            if (dd < nd) { nd = dd; }
                        }
                        if (nDir == 0)
                            nd = std::fabs(inHF[k] - selfAnchor);
                        const double proximity01 =
                            1.0 - std::clamp(nd / proxTol, 0.0, 1.0);

                        double cost = nd;
                        if (medianW > 0.0)
                            cost += (std::max(0.0,
                                         medianW - inCarrierCleanliness[k]) /
                                     medianW) * phasePenSamp;
                        const double extra = std::isfinite(combTopHere)
                            ? std::max(0.0, std::fabs(inHF[k]) -
                                            std::fabs(combTopHere))
                            : 0.0;
                        // Early-transition slope (user, 2026-07-27): the
                        // comb-to-challenger handoff must occur while the
                        // candidates still nearly agree, hiding the seam. At
                        // unit slope the reward only overcame comb's
                        // neighbour advantage after the surplus was already
                        // large, so the flip landed mid-divergence and drew a
                        // visible step along rising edges. Doubling the slope
                        // halves the surplus at which the flip happens; the
                        // cap is unchanged, so large-surplus behaviour is
                        // identical.
                        constexpr double kEarlyHandoffSlope = 2.0;
                        // v2: a baseline share of the HF reward is paid
                        // without the estimator's proof; the proof still
                        // tops it up. Candidate-neutral by construction --
                        // it names only |HF| against comb's |HF|.
                        const double hfVouch = ev2Hf
                            ? (kHfBaseShare + (1.0 - kHfBaseShare) *
                                                  illegalProof)
                            : illegalProof;
                        const double legality =
                            hfVouch * std::min(kEarlyHandoffSlope * extra,
                                               imagePrefCap);
                        cost -= legality;
                        // Cross-colour return evidence. v1 scaled it by
                        // proximity01, which vanishes exactly where the
                        // candidate departs -- i.e. where return delivers.
                        cost -= (ev2Prox ? 1.0 : proximity01) *
                            std::max(0.0, inCrossColorReturnEvidence[k]);
                        if (plane == 1 && !ev2Named)
                            cost += chromaT * kHighChromaDemoteIRE * irescale -
                                    impulseT * kImpulseRetractedBiasIRE *
                                        irescale;
                        if (k < 4) costs[k] = cost;
                        // Strict < keeps roster order as the neutral
                        // tie-break: coherent comb stays senior on ties.
                        if (cost < bestCost) {
                            bestCost = cost;
                            resultHF = inHF[k];
                        }
                    }
                    // Anchor authority (user direction, 2026-07-28): where
                    // the exact channel granted a regional amplitude
                    // ceiling, a candidate whose implied carrier
                    // (raw - complete Y) exceeds it is claiming carrier the
                    // anchors certify cannot exist there. The excess enters
                    // as a normalised weight factor -- authority in the
                    // scoring, per the weight doctrine, never a hard cut --
                    // and the bound was max-pooled + margin-padded, so
                    // legal carrier never pays. Inert wherever no plane or
                    // no authority (+inf).
                    constexpr double kAnchorTauIRE = 1.5;
                    const double anchorCeilIRE =
                        anchorRow ? (double)anchorRow[h]
                                  : std::numeric_limits<double>::infinity();
                    for (int k = 0; k < nIn && k < 4; ++k) {
                        const int plane =
                            (k < baseNIn) ? candPlane[inIdx[k]] : 4;
                        const double yk = reconstructTop(plane, inHF[k]);
                        if (!std::isfinite(yk)) continue;
                        // v1: exp(-dCost/tau), an ABSOLUTE difference
                        // against a fixed 0.75 IRE scale -- where local
                        // costs are compressed it cannot separate anyone.
                        // v2: ratio/log curve w = (1 + dCost/floor)^-p, so
                        // the pedestal follows the RATIO of the scores.
                        static const double kElectFloorIRE = []{
                            const char *e = std::getenv("LDCD_ELECT_FLOOR");
                            return e ? std::atof(e) : 0.15;
                        }();
                        static const double kElectPow = []{
                            const char *e = std::getenv("LDCD_ELECT_POW");
                            return e ? std::atof(e) : 2.5;
                        }();
                        double w;
                        if (ev2Blend) {
                            const double d = std::max(0.0,
                                costs[k] - bestCost) /
                                (kElectFloorIRE * irescale);
                            w = std::exp(-kElectPow * std::log1p(d));
                        } else {
                            w = std::exp(-(costs[k] - bestCost) / blendTau);
                        }
                        if (k == darkestIdx && k < baseNIn && baseNIn > 1)
                            w *= 1.0 - darkestVetoGate;
                        // Notch blindness -- the term that makes a fixed
                        // kernel safe to seat.
                        //
                        // For every ESTIMATING candidate, "carrier I removed"
                        // and "content I could not see" are different
                        // quantities: they subtract a claim. For a fixed
                        // kernel they are THE SAME QUANTITY by construction,
                        // so the instrument already in this election measures
                        // the notch's local blindness with nothing new built
                        // -- raw - notchY IS exactly what the null swallowed.
                        //
                        // Where the pixel's energy sits in that null the
                        // notch's near-zero top band is an ABSTENTION, not a
                        // measurement of "no detail here". Winner-take-all
                        // had no way to say that: a candidate was seated or
                        // discarded, and a seated abstention is a confident
                        // wrong answer. A blend can simply withdraw the
                        // weight, which is what this does.
                        //
                        // RAW magnitude, deliberately NOT the illegalProof-
                        // excluded form chromaT uses above. chromaT asks
                        // "is this legal chroma?"; this asks "how much of
                        // this pixel is in my null?", and the kernel is
                        // equally blind to legal carrier and to confiscated
                        // luma. Opposite treatment, and the difference is the
                        // point: it is what keeps the notch silent at the
                        // iceberg summits, where its answer would be zero and
                        // zero would be wrong.
                        if (plane == 5) {
                            const double blindIRE =
                                remainingCarrierMagnitudeOf(5, h) *
                                invIreScale;
                            if (blindIRE > 0.0 && notchBlindTauIRE > 0.0)
                                w *= std::exp(-blindIRE / notchBlindTauIRE);
                        }
                        if (std::isfinite(anchorCeilIRE)) {
                            const double complete = (k < baseNIn)
                                ? planeY(candPlane[inIdx[k]], h)
                                : returnedY;
                            const double cIRE =
                                std::fabs(rawH - complete) * invIreScale;
                            const double excess = cIRE - anchorCeilIRE;
                            if (excess > 0.0)
                                w *= std::exp(-excess / kAnchorTauIRE);
                        }
                        blendNum += w * yk;
                        blendDen += w;
                    }
                }

                const int winnerPlane = planeForTop(resultHF);
                Y[h] = (blendDen > 1e-12)
                    ? blendNum / blendDen
                    : reconstructTop(winnerPlane, resultHF);
            }
        } else if (carrierComp) {
            for (int h = left; h < right; ++h) {
                const double c = carrierComp[h - left];
                Y[h] = std::isfinite(c)
                    ? (double)rawLine[h] - c
                    : (double)rawLine[h] - clpLine[h];
            }
        } else {
            for (int h = left; h < right; ++h) {
                const double c = clpLine[h];
                Y[h] = std::isfinite(c) ? (double)rawLine[h] - c : (double)rawLine[h];
            }
        }

        // The split has already declared zero carrier on this footprint.
        // Honour that complement after every Y branch: Y = raw - 0.  This is
        // not a second decision and does not reshape the footprint.
        if (starFootprintBuilt &&
            starFootprint_flat.size() >=
                static_cast<size_t>(frameHeight) * demodWidth) {
            const std::uint8_t *star = starFootprint_flat.data() +
                static_cast<size_t>(line) * demodWidth;
            for (int xi = 0; xi < width; ++xi)
                if (star[xi]) Y[left + xi] = (double)rawLine[left + xi];
        }

        if (showMap) {
            std::fill(w2d_frame_weight[line].begin(),
                      w2d_frame_weight[line].end(), 0.0f);
        }
    }




    // Retained record from the removed LDCD_PROBE_YCERT census
    //
    // Grade emitted Y against CERTIFIED LUMA. User direction (2026-08-01):
    // "use the certified/covered truth to compare excess sharpening. We want
    // the sharpening, but not to the point of disruption of color."  On
    // covered def lines the conservation fact Ltrue = raw - exact IS the
    // luma, at full band, per sample.  So the witness's sharpening can be
    // graded against truth instead of against another decoder's convention:
    //   fidelity   mean |Y - Ltrue| over the line (IRE)
    //   slope      mean |dY| / mean |dLtrue| at strong truth edges
    //              (1.0 = matches truth; >1 = sharpened PAST the source)
    //   overshoot  mean of (|Y-Ltrue| at edge-adjacent samples) in IRE, the
    //              part that must leave through chroma (chroma = raw - Y)
    //   band       fraction of |Y - Ltrue| energy above 1.3 MHz -- excess
    //              that is not even expressible as a chroma correction
    //
    // NOTE, kept because it bites any successor: such a census must NOT use
    // certifiedDefLine() -- that helper is gated on the certified family, so
    // with LDCD_CERT_1D=0 (the held-out configuration it needs) it reports
    // no def lines at all. Read the exact plane directly: truth exists
    // independently of whether the decoder is allowed to consume it.
    //
    // ---- CHROMA quality against the CERTIFIED CARRIER ([CCERT]) ----
    // User (2026-08-01): "Chroma side cost should look to the certified
    // carrier. It's sparse, but nothing else is in the same league."  At a
    // covered sample the TRUE carrier is exact, and the render's implied
    // carrier is (raw - Y) -- the exact complement of emitted luma.  Both
    // are composite-domain carriers, so their 4fSC quadrature envelopes are
    // directly comparable:
    //   env(x) = hypot(c[x], c[x+1])     (adjacent samples are 90 deg)
    // Reported (all truth-referenced, no bucket anywhere):
    //   amp     mean envRender / mean envTruth at strong truth chroma
    //           -- SATURATION fidelity against fact (1.0 = correct)
    //   width   10-90% transition width of each envelope at truth chroma
    //           edges -- the width question, re-asked against truth after
    //           the bucket-referenced version was falsified by the user
    //   err     mean |envRender - envTruth| in IRE
}
// Lurch preconditioner for the coarse luma prior.
//
// The legal 4-sample means cancel carrier, but they are boxcars: a luma step
// smears across the four windows that straddle it, and every pre-2D consumer
// of the coarse prior (the carrier fit, the witness, the patch gates) then
// sees step energy stranded in the carrier band.  Membership movement between
// adjacent windows,
//
//     D[s] = M[s+1] - M[s] = (raw[s+4] - raw[s]) / 4,
//
// compares samples of identical carrier phase, so a luma step produces a
// same-sign run of D across the straddling windows — phase-invariant — while
// a chroma envelope edge alternates sign window to window and is rejected by
// the run test.  Where a step is confirmed, the prior takes the nearest
// same-side window mean instead of the boxcar ramp: the transition lands at
// one column instead of four.  One steep transition per aperture, by design.
//
// `prior` is blended in place; `gateOut` (optional) reports per-pixel
// sharpening activity so a consumer can stand down its own edge correction.
// (LurchStepRun itself lives in comb.h -- it is production data now.)

// Detect luma step runs in a coarse mean sequence -- the difference facts
// that own HF (a legal carrier is aperture-invariant, so a same-sign run of
// D across straddling windows is a luma step; a chroma envelope edge
// alternates sign and is rejected). Gates are stored at UNIT gain; consumers
// scale (clamp(gate*gain,0,1) reproduces any detection-time gain exactly).
// Canonical runs on the shared aperture pool come from buildLurchStepRuns();
// this stays callable directly for OTHER mean sequences (e.g. the carrier
// fit's winFloor), which are different quantities, not duplication.
static void detectLurchSteps(const double *means, int meanCount,
                             double irescale, double invIreScale,
                             std::vector<LurchStepRun> &runs)
{
    runs.clear();
    if (!means || meanCount < 6)
        return;

    const auto smoothStep01 = [](double t) {
        t = std::clamp(t, 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    };

    // Per-window movement floor: a confirmed step of >= ~1.2 IRE moves each
    // straddling window by >= ~0.3 IRE.
    const double dThreshSamples = 0.30 * irescale;
    const int dCount = meanCount - 1;

    int s = 0;
    while (s < dCount) {
        const double d0 = means[s + 1] - means[s];
        if (std::fabs(d0) < dThreshSamples) {
            ++s;
            continue;
        }

        const bool positive = d0 > 0.0;
        const int a = s;
        int b = s;
        while (b + 1 < dCount) {
            const double dn = means[b + 2] - means[b + 1];
            if (std::fabs(dn) < dThreshSamples || (dn > 0.0) != positive)
                break;
            ++b;
        }
        s = b + 1;

        // An ideal step moves exactly four windows; allow slack for noise
        // and local gradient, but a long run is a ramp, not a step.
        const int runLength = b - a + 1;
        if (runLength > 6)
            continue;

        const double stepSamples =
            means[std::min(b + 1, meanCount - 1)] - means[a];
        const double stepIRE = std::fabs(stepSamples) * invIreScale;
        const double gate = smoothStep01((stepIRE - 1.25) / 2.75);
        if (gate <= 0.0)
            continue;

        // Amplitude-weighted centroid of |D| locates the edge even when the
        // threshold trimmed the run asymmetrically (an ideal step's run is
        // s in [e-4, e-1], centroid e-2.5).
        double wSum = 0.0;
        double wPos = 0.0;
        for (int k = a; k <= b; ++k) {
            const double w = std::fabs(means[k + 1] - means[k]);
            wSum += w;
            wPos += w * (double)k;
        }
        const double centroid =
            (wSum > 1e-12) ? (wPos / wSum) : 0.5 * (double)(a + b);

        LurchStepRun run;
        run.a = a;
        run.b = b;
        run.edge = centroid + 2.5;
        run.stepSamples = stepSamples;
        run.stepAbsIRE = stepIRE;
        run.gate = gate;
        runs.push_back(run);
    }

    if (runs.empty())
        return;

    // Ringing suppression.  Sharp analog edges overshoot, which reverses the
    // membership movement and shows up as a smaller opposite run right beside
    // the true step.  Snapping to those fragments puts line-varying plateaus
    // into raw - prior at exactly the edges this pass exists to clean — the
    // fringes survive every comb stage because the contamination is upstream
    // of all of them.  A run within a few windows of a materially stronger
    // run is overshoot, not a second edge.
    for (size_t i = 0; i < runs.size(); ++i) {
        for (size_t j = 0; j < runs.size(); ++j) {
            if (i == j)
                continue;
            const int gap = (runs[i].a > runs[j].b)
                ? runs[i].a - runs[j].b
                : runs[j].a - runs[i].b;
            if (gap <= 3 && runs[j].stepAbsIRE >= 2.5 * runs[i].stepAbsIRE) {
                runs[i].suppressed = true;
                break;
            }
        }
    }
}

// Sample-resolution step detector on CERTIFIED luma (Lhat = raw - exact,
// covered lines only). The aperture detector above fuses same-sign corners
// closer than ~4 px and lets ringing suppression eat close opposite pairs
// -- the one-corner-per-4-px limit. Lhat needs no aperture: it is already
// carrier-free by conservation, so steps resolve at sample resolution and
// the limit does not exist here. Detection thresholds mirror the aperture
// detector's IRE scale; the per-sample difference threshold is higher
// (0.60 vs 0.30 IRE) because samples do not enjoy the window's 4x
// averaging. NaN-tolerant: a NaN sample ends any run.
static void detectCertifiedLurchSteps(const double *lhat, int count,
                                      double irescale, double invIreScale,
                                      std::vector<LurchStepRun> &runs)
{
    runs.clear();
    if (!lhat || count < 6)
        return;

    const auto smoothStep01 = [](double t) {
        t = std::clamp(t, 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    };

    const double dThresh = 0.60 * irescale;
    const int dCount = count - 1;

    int s = 0;
    while (s < dCount) {
        if (!std::isfinite(lhat[s]) || !std::isfinite(lhat[s + 1])) {
            ++s;
            continue;
        }
        const double d0 = lhat[s + 1] - lhat[s];
        if (std::fabs(d0) < dThresh) {
            ++s;
            continue;
        }
        const bool positive = d0 > 0.0;
        const int a = s;
        int b = s;
        while (b + 1 < dCount) {
            if (!std::isfinite(lhat[b + 2])) break;
            const double dn = lhat[b + 2] - lhat[b + 1];
            if (std::fabs(dn) < dThresh || (dn > 0.0) != positive)
                break;
            ++b;
        }
        s = b + 1;

        // A sample-resolution step spans 1-2 differences; longer is a ramp.
        const int runLength = b - a + 1;
        if (runLength > 3)
            continue;

        const double stepSamples = lhat[b + 1] - lhat[a];
        const double stepIRE = std::fabs(stepSamples) * invIreScale;
        const double gate = smoothStep01((stepIRE - 1.25) / 2.75);
        if (gate <= 0.0)
            continue;

        double wSum = 0.0, wPos = 0.0;
        for (int k = a; k <= b; ++k) {
            const double w = std::fabs(lhat[k + 1] - lhat[k]);
            wSum += w;
            wPos += w * (double)k;
        }
        const double centroid =
            (wSum > 1e-12) ? (wPos / wSum) : 0.5 * (double)(a + b);

        LurchStepRun run;
        run.a = a;
        run.b = b;
        run.edge = centroid + 0.5;   // d[k] sits between samples k, k+1
        run.stepSamples = stepSamples;
        run.stepAbsIRE = stepIRE;
        run.gate = gate;
        run.certified = true;
        runs.push_back(run);
    }

    // Same ringing rule as the aperture detector, in sample units.
    for (size_t i = 0; i < runs.size(); ++i) {
        for (size_t j = 0; j < runs.size(); ++j) {
            if (i == j) continue;
            const int gap = (runs[i].a > runs[j].b)
                ? runs[i].a - runs[j].b
                : runs[j].a - runs[i].b;
            if (gap <= 3 && runs[j].stepAbsIRE >= 2.5 * runs[i].stepAbsIRE) {
                runs[i].suppressed = true;
                break;
            }
        }
    }
}

// Fill the canonical per-line lurch run lists from the shared aperture pool:
// ONE detection per line per frame, unit gain, meanCount = width-3 (the real
// aperture starts). Every consumer -- the witness coarse-sharpener, the edge
// probes, and the coming 2D threshold work -- reads these instead of privately
// re-running the scan. Runs on every path from split1D, right after the pool
// itself is built; O(width) per line, so the default path pays noise.
void Comb::FrameBuffer::buildLurchStepRuns()
{
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int width     = right - left;

    if ((int)lurchStepRuns.size() < lastLine)
        lurchStepRuns.resize(lastLine);
    for (auto &rowRuns : lurchStepRuns)
        rowRuns.clear();

    if (width < 8 || firstLine >= lastLine || lockedApertureMean_flat.empty())
        return;

    // FALSIFIED -- do not re-propose: SUB-PIXEL EDGE REFINEMENT FROM THE
    // DISCARDED HALF-SAMPLE TERM.
    //
    // The moving coarse registers to integer xi by AVERAGING the two
    // apertures whose centres straddle xi at +-0.5, so only their sum
    // survives; their difference is a carrier-free luma gradient that is
    // thrown away. The idea was to recover it as a sub-pixel correction to
    // the detected edge.
    //
    // Measured against certified luma by the since-removed LURCHVET census,
    // using only
    // run-time-available quantities, over 9 scenes on 4 sources
    // (n = 180,485 steps above 10 IRE):
    //
    //   scene       p50 placement err change     native slope
    //   cube            -14.6%                      -2.14
    //   s2x10            -1.2%                      -0.48
    //   vol@520          +5.4%                      -0.18
    //   ggv@18k          +7.7%                      +0.05
    //   ggv@1.3k         +9.3%                      -0.08
    //   beach           +13.8%                      +0.19
    //   s1x11@45k       +17.5%                      -0.81
    //   s1x11@60k       +17.6%                      -5.06
    //   POOLED           +7.6%                      -0.23
    //
    // The relationship is not a physical constant: the slope CHANGES SIGN
    // between sources, and on GGV -- 150k of those samples -- it is
    // indistinguishable from zero. A coefficient fitted on the cube (-1.8)
    // makes placement WORSE on eight of nine scenes. No fixed coefficient can
    // exist, and a per-scene fit would be calibration, which this project
    // does not do. The apparent effect was one shot's worth of frames
    // mistaken for a mechanism.
    //
    // What DID survive is the census that killed it: the detector's
    // reliability curve (blind below ~2.5 IRE, ~1 px at 5-10, 0.7-0.8 px
    // above 10, 15-18% of runs with no certified counterpart), which is
    // stable across all four sources and is the basis for restricting where
    // these runs are allowed to act.

    for (int line = firstLine; line < lastLine; ++line) {
        const double *apMean = lockedApertureMean_line(line);
        if (!apMean) continue;
        detectLurchSteps(apMean, width - 3, irescale, invIreScale,
                         lurchStepRuns[line]);
    }

    // CERTIFIED LUMA FACTS (user, 2026-07-29: "Lurch: yes, this is a big
    // priority, the 1 corner per 4 pixel limit can be broken"). On covered
    // lines Lhat = raw - exact is per-sample certified luma; its steps are
    // facts, consumed as facts (never calibration). Three parity-lawful
    // moves -- the parity law says exact consumption must reach both
    // parities together, and it has bitten twice this arc:
    //   1. REPOSITION: a covered line's aperture run within tolerance of a
    //      certified step takes the certified sub-sample edge.
    //   2. (in corroborateLurchEdges) comp lines bracketed by two
    //      certified partners snap to their MEAN -- the certified
    //      interpolation -- instead of median-of-three.
    //   3. INSERTION, always as a vertical TRIPLE: a certified step the
    //      aperture missed on line L, matched by a certified step on
    //      L+2, inserts runs on L, L+1 (interpolated) and L+2 together --
    //      corner density rises on both parities at once, never on one.
    // A/B escape LDCD_LURCH_CERT=0; default ON. Uncovered frames untouched.
    static const bool lurchCert = []{
        const char *e = std::getenv("LDCD_LURCH_CERT");
        return !(e && std::atoi(e) == 0);
    }();
    if (lurchCert && frameHasExactCoverage()) {
        constexpr double kCertMatchPx = 1.5;
        std::vector<std::vector<LurchStepRun>> certMissed(lastLine);
        // Pending common-mode corrections per (line, run index): a comp
        // run bracketed by two covered lines receives a delta from each;
        // they must AVERAGE, not stack.
        std::vector<std::vector<std::pair<double, int>>> pend(lastLine);
        std::vector<double> lhat(width);
        std::vector<LurchStepRun> certRuns;
        // Retained record from the removed LDCD_PROBE_LURCHVET census
        //
        // Detector vetting census. READ-ONLY. A covered line carries BOTH
        // the aperture means the detector sees and the certified luma
        // saying what is actually there, so the detector could be scored
        // against truth instead of against its own math -- which was
        // impossible before the exact channel existed. It scored the
        // detection as it stood BEFORE any certified repositioning, and
        // altered nothing: not truth, not the runs, not conduct.
        //
        // Stratified by certified step magnitude, because the two detectors
        // are NOT equally sensitive: detectCertifiedLurchSteps trips at
        // 0.60 IRE per sample on noise-free truth, while the aperture
        // detector works on 4-sample means with a ~1.2 IRE step floor. An
        // unstratified "miss" count mostly measures that design gap, not
        // detector error. Position error was measured to the NEAREST
        // aperture run of the same sign at any distance, so the 1.5 px
        // match window did not truncate it.
        //
        // The aperture side was stratified by the run's OWN step height --
        // the quantity a restriction can key on at run time. Is a "false"
        // run genuinely spurious, or is the CLAIM FLAG the artefact?
        // ar.certified is set only on the first match and only within
        // 1.5 px, so several runs on one wide feature leave all but one
        // flagged false BY CONSTRUCTION; the census therefore measured
        // distance to the nearest same-sign certified step regardless of
        // the flag, plus the run's own width, so the three readings
        // separate. Uncorroborated meant no certified step of the same
        // sign within 3 px -- distance-based, never the claim flag.
        //
        // It also characterised the population truth does NOT corroborate:
        // if those runs sit where the CARRIER is strong they are leakage
        // through imperfect four-sample cancellation and the bandwidth law
        // has a claim on them; if their carrier looks like everyone else's
        // they are not a carrier phenomenon and no bandwidth constraint
        // will find them.
        //
        // The sub-pixel arm used OPERATIONAL quantities only: the discarded
        // half-sample gradient was located from the APERTURE run's own edge
        // and normalised by its own step height, because that is all a live
        // correction can see. Locating from the certified edge or scaling
        // by the certified step height would fit a coefficient on certified
        // data the implementation does not have. Its verdict -- and the
        // detector reliability curve that survived it -- is the FALSIFIED
        // block at the head of this function.
        for (int line = firstLine; line < lastLine; ++line) {
            const float *ex = exactCarrierRow(line);
            if (!ex) continue;
            const quint16 *rawLine = rawbuffer.data()
                + static_cast<size_t>(line) * videoParameters.fieldWidth;
            int nFinite = 0;
            for (int xi = 0; xi < width; ++xi) {
                const float e = ex[left + xi];
                if (std::isfinite(e)) {
                    lhat[xi] = static_cast<double>(rawLine[left + xi]) -
                               static_cast<double>(e);
                    ++nFinite;
                } else {
                    lhat[xi] = std::numeric_limits<double>::quiet_NaN();
                }
            }
            if (nFinite < 16) continue;
            detectCertifiedLurchSteps(lhat.data(), width, irescale,
                                      invIreScale, certRuns);
            for (const LurchStepRun &cr : certRuns) {
                if (cr.suppressed) continue;
                bool matched = false;
                for (LurchStepRun &ar : lurchStepRuns[line]) {
                    if (ar.suppressed) continue;
                    if ((ar.stepSamples > 0.0) != (cr.stepSamples > 0.0))
                        continue;
                    if (std::fabs(ar.edge - cr.edge) > kCertMatchPx)
                        continue;
                    // COHERENT repositioning (the parity law's third
                    // lesson this arc, caught by the straightness dump):
                    // moving only the covered line to truth breaks
                    // line-to-line consistency, because adjacent aperture
                    // detections share systematic bias -- truth on one
                    // parity + bias on the other IS alternation (scatter
                    // 0.384 -> 0.437 px, strut lineAlt +3%). So the
                    // certified CORRECTION delta propagates to the
                    // vertically-matched comp runs above and below at the
                    // same time: every line moves together, common-mode,
                    // and the shared detector bias is what the delta
                    // cancels.
                    const double delta = cr.edge - ar.edge;
                    ar.edge = cr.edge;          // the certified position
                    ar.gate = std::max(ar.gate, cr.gate);
                    ar.certified = true;
                    for (int nl = line - 1; nl <= line + 1; nl += 2) {
                        if (nl < firstLine || nl >= lastLine) continue;
                        auto &rowRuns = lurchStepRuns[nl];
                        if (pend[nl].size() < rowRuns.size())
                            pend[nl].resize(rowRuns.size(), {0.0, 0});
                        for (size_t ri = 0; ri < rowRuns.size(); ++ri) {
                            LurchStepRun &mr = rowRuns[ri];
                            if (mr.suppressed || mr.certified) continue;
                            if ((mr.stepSamples > 0.0) !=
                                (cr.stepSamples > 0.0)) continue;
                            if (std::fabs(mr.edge - (cr.edge - delta)) >
                                kCertMatchPx) continue;
                            pend[nl][ri].first += delta;
                            pend[nl][ri].second += 1;
                            break;
                        }
                    }
                    matched = true;
                    break;
                }
                if (!matched)
                    certMissed[line].push_back(cr);
            }
        }

        // Apply the averaged common-mode corrections.
        for (int line = firstLine; line < lastLine; ++line) {
            auto &rowRuns = lurchStepRuns[line];
            for (size_t ri = 0; ri < pend[line].size() &&
                                ri < rowRuns.size(); ++ri) {
                if (pend[line][ri].second > 0)
                    rowRuns[ri].edge += pend[line][ri].first /
                                        pend[line][ri].second;
            }
        }

        // Insertion as vertical triples (move 3).
        for (int line = firstLine; line + 2 < lastLine; ++line) {
            for (LurchStepRun &r : certMissed[line]) {
                if (r.suppressed) continue;   // reused as consumed marker
                LurchStepRun *o = nullptr;
                for (LurchStepRun &c : certMissed[line + 2]) {
                    if (c.suppressed) continue;
                    if ((c.stepSamples > 0.0) != (r.stepSamples > 0.0))
                        continue;
                    if (std::fabs(c.edge - r.edge) > kCertMatchPx) continue;
                    o = &c;
                    break;
                }
                if (!o) continue;
                // The comp line between them must not already carry an
                // aperture run here (it would double-report the corner).
                bool compHasIt = false;
                for (const LurchStepRun &m : lurchStepRuns[line + 1]) {
                    if (m.suppressed) continue;
                    if (std::fabs(m.edge - 0.5 * (r.edge + o->edge)) <=
                        kCertMatchPx) { compHasIt = true; break; }
                }
                LurchStepRun mid;
                mid.edge = 0.5 * (r.edge + o->edge);
                mid.stepSamples = 0.5 * (r.stepSamples + o->stepSamples);
                mid.stepAbsIRE = 0.5 * (r.stepAbsIRE + o->stepAbsIRE);
                mid.gate = std::min(r.gate, o->gate);
                mid.certified = true;
                mid.a = std::max(0, (int)mid.edge - 2);
                mid.b = std::min(width - 4, (int)mid.edge + 1);
                lurchStepRuns[line].push_back(r);
                if (!compHasIt)
                    lurchStepRuns[line + 1].push_back(mid);
                lurchStepRuns[line + 2].push_back(*o);
                o->suppressed = true;         // consumed
                r.suppressed = true;
            }
        }
    }
}

// See comb.h. The vertical partner step mirrors the election's Y-geometry
// rule: immediately adjacent lines wherever the cadence allows the frame-
// vertical model, same-field +-2 otherwise.
std::vector<LurchStepRun> Comb::FrameBuffer::corroborateLurchEdges(int line) const
{
    std::vector<LurchStepRun> runs = lurchStepRuns_line(line);
    if (runs.empty()) return runs;

    constexpr double kLurchMatchPx = 1.5;
    const int step = carrierFrameVerticalAllowed(line) ? 1 : 2;
    const auto &up = lurchStepRuns_line(line - step);
    const auto &dn = lurchStepRuns_line(line + step);

    auto matchEdge = [&](const std::vector<LurchStepRun> &nbr,
                         const LurchStepRun &run, double &edgeOut,
                         bool &certOut) -> bool {
        double bestD = kLurchMatchPx;
        bool found = false;
        for (const LurchStepRun &o : nbr) {
            if (o.suppressed) continue;
            if ((o.stepSamples > 0.0) != (run.stepSamples > 0.0)) continue;
            const double d = std::fabs(o.edge - run.edge);
            if (d <= bestD) {
                bestD = d; edgeOut = o.edge; certOut = o.certified;
                found = true;
            }
        }
        return found;
    };

    for (LurchStepRun &run : runs) {
        if (run.suppressed) continue;
        double eu, ed;
        bool cu = false, cd = false;
        if (!matchEdge(up, run, eu, cu) || !matchEdge(dn, run, ed, cd))
            continue;                       // no full vertical company
        if (cu && cd && !run.certified) {
            // Both partners carry certified positions: their mean is the
            // certified interpolation, strictly better information than
            // this line's own aperture centroid. The comp line inherits
            // the anchors -- both parities improve together.
            run.edge = 0.5 * (eu + ed);
            continue;
        }
        const double e = run.edge;
        // median of three, by selection
        run.edge = std::max(std::min(eu, e), std::min(std::max(eu, e), ed));
    }
    return runs;
}

// Retained record from the removed LDCD_PROBE_RETRHULL census
//
// Post-law hull-violation stats. Measurement only.
// Pass 1 clamps the carrier fit into the residual-consensus feasible range
// (region-pure four-view complements + the rolling witness), but the encoder
// bandwidth law is imposed AFTER that clamp at publication, and its 9-tap
// FIR mixes neighbours -- the published fit can leave the per-sample range
// the clamp enforced. This measured how often and by how much, deciding
// whether re-imposing the hull after the law is load-bearing or insurance.
// Its verdict is quoted at the re-clamp itself (cube 45.4% of samples
// outside, mean excess 0.56 IRE, max 44; beach 37.9%).

// Retained record from the removed LDCD_PROBE_DISENT census
//
// Pair class-map probe at luma steps. MEASUREMENT ONLY -- writes nothing.
//
// LAW (1D safe retreat, 2026-07-26): the 1D stage may contain no comb,
// blend, or influence from other lines. Downstream 2D is programmed to
// fall back to 1D precisely when its vertical machinery produces errors;
// 1D is the safe retreat, and the fallback hierarchy only works if 1D is
// structurally incapable of vertical error. Adjacent-line data may CONFIRM
// (phase relations, luma contrasts) as verdict inputs; rejection comes
// from the line's own model; no cross-line sample may contribute signal.
//
// P12 briefly subtracted the pair common mode
//     cm = 0.5*(B{raw_n} + B{raw_p}) = Lbar + 0.5*dC
// from the emitted carrier at lurch footprints. That is a 2-tap interfield
// comb inside 1D -- a violation of the law above -- and it injected 0.5*dC
// at vertical colour boundaries. WITHDRAWN. What the measurement itself
// established is kept (and this probe reproduces it on demand): the leak
// is CLASS-INVARIANT in composite -- |cm| and its I/Q split are uniform
// across all four parity x flip cells -- so the field asymmetry seen in
// renders was manufactured by per-line verdict actuators, never present in
// the error. The complement-pair tiling ([+ - - +]; even lines pair down,
// odd pair up, swapping per frame) is confirmed against the grammar here,
// not presumed.
//
// Reverse-engineering stats for the pair disentangle (LDCD_PROBE_DISENT=1).
// Measurement only. The per-class map is the point: the leak's demod hue is
// set by (x mod 4, lineFlip), so the SAME luma edge presents a different
// error per line class -- Y colliding with different conventions along the
// phase sequence. These counters expose that map and the field asymmetry.
//
// Complement partner: the schedule tiles [+ - - +] down the frame, so
// exactly one adjacent frame line carries the opposite signed class.
// The grammar names it; nothing is presumed from the schedule.
//
// Per-run error profile against the ANTICIPATED doublet: a chain-
// sharp (w = 2) ramp with lurch's pinned plateau levels at lurch's
// edge, through the bandpass's own operator -0.25*D2_2. r says
// whether the anticipation carries the SHAPE (evidence can predict
// the waveform) or only a location and magnitude bound; beta says
// whether lurch's amplitude scales it correctly where the shape
// holds. peak|cm| vs step height h feeds the scale law.

// Retained record from the removed LDCD_PROBE_EDGEFATE census
//
// Downstream-fate stats (LDCD_PROBE_EDGEFATE=1). Measurement only. Every
// render judged so far was ntsc1d -- pure 1D, no comb ever touched the edge
// bands. This probe asks what the REAL pipeline does at lurch footprints:
// how far 2D moves the carrier off its 1D source there (vs a control of all
// other pixels), how often it effectively passes 1D through (the fallback),
// and what 3D adds. The answer decides WHERE edge evidence should be
// delivered: into the fallback conditioning, into the comb's own gates, or
// into the election.
//
// clpbuffer[0] is not the untouched 1D baseline on certified frames, so this
// particular diagnostic has no comparable reference there.

// Retained record from the removed LDCD_PROBE_INVENT census
//
// Reconciliation (user, 2026-08-02: "we want all the data we can get").
// The premise behind the lawful-subspace push was that the anchored model
// admits alias on the cube body -- but that reading was rendered |C|
// (chroma magnitude after demod gain and transformIQ), which is NOT
// carrier IRE and cannot be compared with the truth ledger, where the fit
// grades 0.39-0.89 IRE against exact. One of those two readings misleads.
//
// This settles it in one unit: every carrier claimant vs the exact channel
// in IRE, on covered def samples, full frame and inside a box.
// Claimants: 1D (locked1DSource), comb (the elected composite carrier),
// fit (carrierFit), retracted (raw - carrierRetracted).
// Per claimant: mean |claim|, mean |claim - exact|, and two rates --
//   invent  = P(|claim| > |exact| + 1 IRE)   claiming carrier truth lacks
//   confisc = P(|exact| > |claim| + 1 IRE)   missing carrier truth has
// against mean |exact|, the true carrier mass in that population.
// Run with LDCD_CERT_1D=0: exact is read directly (truth survives) while
// every claimant stays an estimator.
//
// The claimant the RENDERER actually consumes: rendered chroma
// is demod(raw - Y) of the ELECTED Y, not of any carrier plane.
// If the planes are clean and this is not, the invention lives
// in the election / Y assembly, not in the carrier model.

// Retained record from the removed LDCD_PROBE_TWEEN census
//
// User design call (2026-08-02): amplitude work in COMPOSITE is out -- an
// envelope needs a window (a smoother on composite blurs, never cancels)
// and the per-cycle alternative scores at carrier rate, which amplitude-
// modulates the very thing it judges. The luma route is an identity
// instead: carrier = raw - L, per sample, no window anywhere. Because
// alias-conforming cross-colour IS luma, a true L excludes it from the
// carrier model BY CONSTRUCTION rather than by bound -- the only form of
// "lawful subspace" that can reject energy no spectral law can touch.
//
// The shipped anticipated rung is ONE-SIDED (causal copy from the previous
// cover) = copying, not tweening, and it renders as the ghost the beach
// exposed. The cadence makes the two-sided form available for free: every
// uncovered frame sits at the midpoint between two covers.
//
// HELD OUT HONESTLY: target = the MIDDLE of three consecutive covers,
// predicted from the outer two, graded per sample against its own exact
// channel. That straddle spans 4 film frames with the target at the
// midpoint -- exactly twice the real job's distance -- so every number is
// a conservative bound on the real tween.
//
// Columns, all on one population:
//   tween = 0.5*(L_prev2 + L_current)   the proposal
//   copy  = L_prev2                     what ships today (one-sided)
//   fit   = raw - fit                   the incumbent, from the PRE-STAMP
//                                       fit stash so it grades an
//                                       estimator, not the fact factFit
//                                       writes over it
// Split by DIRECT (both predictors land on their own def lines -- the
// parity-union question) vs BRACKET, and by straddle disagreement
// |L_prev2 - L_current| (the motion axis: where the two covers agree the
// tween should be truth; where they disagree the disagreement IS the
// out-of-band residual, NOT a witness.
//
// Run: LDCD_PROBE_TWEEN=1 LDCD_CERT_1D=0 -t 1. Exact is read directly
// (certifiedDefLine short-circuits under CERT_1D=0), so truth survives the
// hold-out while every estimator stays an estimator.
//
// CONSECUTIVE-COVER UNION CENSUS. certPrev2 and certPrev are
// adjacent covers, i.e. exactly the pair that straddles ONE
// uncovered frame -- the real application geometry, as opposed to
// the ±2 straddle this probe must grade on (truth only exists at
// covers). The question the tween lives or dies on: does the
// yin-yang def-parity mirror make those two covers
// parity-complementary, so their union delivers DIRECT certified
// luma on every line of the frame between them (no vertical
// bracket -- no smoother)? Measured here over the whole active
// area, needing no truth.

void Comb::FrameBuffer::lurchSharpenCoarsePrior(const double *means,
                                                int meanCount,
                                                int width,
                                                double *prior,
                                                double *gateOut,
                                                double gateGain) const
{
    if (!means || meanCount < 6 || width <= 0) {
        if (gateOut && width > 0)
            std::fill(gateOut, gateOut + width, 0.0);
        return;
    }

    std::vector<LurchStepRun> runs;
    detectLurchSteps(means, meanCount, irescale, invIreScale, runs);
    applyLurchSteps(runs, means, meanCount, width, gateGain, prior, gateOut);
}

// The application half of the lurch sharpener, split out so consumers of the
// CANONICAL run lists (buildLurchStepRuns) apply them without re-detecting.
// gateGain scales the stored unit-gain gates: clamp(gate*gain,0,1) is exactly
// the value detection at that gain would have produced.
void Comb::FrameBuffer::applyLurchSteps(const std::vector<LurchStepRun> &runs,
                                        const double *means, int meanCount,
                                        int width, double gateGain,
                                        double *prior, double *gateOut) const
{
    if (gateOut && width > 0)
        std::fill(gateOut, gateOut + width, 0.0);

    if (!means || !prior || meanCount < 6 || width <= 0 || runs.empty())
        return;

    using StepRun = LurchStepRun;

    // Apply the strongest surviving run per pixel, always blending from the
    // unsharpened base so overlapping runs never compound.
    std::vector<double> base(prior, prior + width);
    std::vector<double> localGate(width, 0.0);

    // A side anchor is only trustworthy if its window is clear of every
    // detected run.  A short bar (e.g. a colored patch abutting black) has
    // two close opposite runs; an anchor that crosses into the other run's
    // span would snap the bar interior to the far plateau.  With no clean
    // anchor on the required side, leave the pixel on the base prior — an
    // honest blur beats a confident wrong plateau.
    const auto anchorContaminated = [&runs](int s) {
        for (const StepRun &other : runs) {
            if (other.suppressed)
                continue;
            if (s >= other.a && s <= other.b + 1)
                return true;
        }
        return false;
    };

    for (const StepRun &run : runs) {
        if (run.suppressed)
            continue;
        const double g = std::clamp(run.gate * gateGain, 0.0, 1.0);

        const int xiFirst =
            std::clamp((int)std::floor(run.edge) - 4, 0, width - 1);
        const int xiLast =
            std::clamp((int)std::ceil(run.edge) + 3, 0, width - 1);

        for (int xi = xiFirst; xi <= xiLast; ++xi) {
            if (g <= localGate[xi])
                continue;

            // One window of margin beyond the detected run: threshold
            // trimming can leave the run's end windows still straddling the
            // step, and snapping to a straddling window is worse than not
            // snapping at all.
            const int side = ((double)xi < run.edge)
                ? std::clamp(std::min(xi, run.a - 1), 0, meanCount - 1)
                : std::clamp(std::max(xi - 3, run.b + 2), 0, meanCount - 1);

            if (anchorContaminated(side))
                continue;

            localGate[xi] = g;
            prior[xi] = base[xi] * (1.0 - g) + means[side] * g;
        }
    }

    if (gateOut) {
        for (int xi = 0; xi < width; ++xi)
            gateOut[xi] = localGate[xi];
    }
}

// ---------------------------------------------------------------------------
// The coarse luma platform, SOLVED rather than snapped.
//
// applyLurchSteps() above pattern-matches one shape out of the membership
// sequence -- a confirmed step -- and substitutes an idealisation of it. What
// it does not match, it discards: movement under 0.30 IRE per window, any run
// whose sign is not monotone, every run longer than six windows ("a long run
// is a ramp, not a step"), and every step under 1.25 IRE. The survivors
// collapse to six scalars and are applied as a piecewise-constant plateau. So
// the platform learns "a step of about this size sits near this column" and
// nothing else, and it learns it as a cliff.
//
// THE FACTS ARE CONTINUOUS AND THEY ARE EVERYWHERE. Two samples four apart
// share carrier phase, so with no detection of any kind:
//
//     raw[x+4] - raw[x] = (Y[x+4] - Y[x]) + (C[x+4] - C[x])
//
// and the residual C[x+4] - C[x] is exactly the carrier ENVELOPE's drift over
// four samples. The encoder low-passes that envelope to ~1.3 MHz before
// modulation (feasibleband.h's bandwidth law), so ABOVE that frequency the
// difference facts are exact by law, and BELOW it the envelope may legitimately
// move and their residual is that legal motion. The band split is therefore not
// a taste parameter: the difference facts own HF, the coarse platform owns LF, and
// the crossover is the encoder's chroma bandwidth. lurchPlatformCutoffMHz() is
// that crossover, and it is the platform's whole LF-authority control.
//
// THE SOLVE. Per line, minimise
//
//     wA * sum_x ( Y[x+4] - Y[x] - d[x] )^2  +  wB * sum_x ( Y[x] - A[x] )^2
//
// A being the integer-centred moving coarse. Y[x] couples only to Y[x+-4], so
// the system separates into four independent tridiagonal chains by carrier
// phase class and Thomas solves each exactly in O(width). The 2026 predecessor
// (ad5dd4a, deleted by the witness rollback in fc3cd4b) ran 14 alternating
// Gauss-Seidel sweeps over this same system; the iteration was never needed.
//
// It also tied Y itself to the already-snapped prior, and had to note that
// "anchoring to the blurry boxcar would make the anchor fight the difference
// facts at every edge" (its word for the platform) -- true, and an artifact
// of tying the wrong
// quantity. The platform's only real duty is the slow part, and that part is
// slow BY LAW. Tied at LF it is never consulted at an edge, so the snap is
// not needed as a precondition and the second pass disappears.
//
// THE FACTS DO NOT ENTER HERE YET. Pinning certified samples as Dirichlet rows
// was built and measured, and it violates the parity law -- see
// lurchPinEnabled() for the numbers and for why skipping certified lines
// instead would fail in exactly the same way. It is off by default.
//
// The solve is nonetheless not fact-blind: buildApertureMeans() forms the pool
// from raw - exact wherever the twin channel certifies a sample, so the
// platform is already certified-derived on covered lines -- at window-mean
// resolution,
// spread over four samples, which is why that consumption does not alternate
// and a per-sample substitution does.
// ---------------------------------------------------------------------------
void Comb::FrameBuffer::solveLurchYCurve(int line, const double *apMean,
                                         int meanCount, int width,
                                         double *yOut)
{
    if (!apMean || !yOut || width <= 0 || meanCount <= 0)
        return;

    const int lastStart = meanCount - 1;

    // Registration: the integer-centred moving coarse. Averaging the two
    // half-sample apertures straddling xi gives the phase-balanced five-sample
    // support (0.5,1,1,1,0.5)/4 centred exactly at xi -- the same platform both
    // consumers built privately before this existed. It is also the fallback:
    // every early return below leaves the platform exactly as it was.
    if ((int)scratch_lurchPlatform.size() < width)
        scratch_lurchPlatform.resize(width);
    double *platform = scratch_lurchPlatform.data();
    for (int xi = 0; xi < width; ++xi) {
        const int s0 = std::clamp(xi - 2, 0, lastStart);
        const int s1 = std::clamp(xi - 1, 0, lastStart);
        platform[xi] = 0.5 * (apMean[s0] + apMean[s1]);
        yOut[xi]   = platform[xi];
    }

    const int left      = videoParameters.activeVideoStart;
    const int fullWidth = videoParameters.fieldWidth;
    if (line < 0 || fullWidth <= 0 || left < 0 ||
        (size_t)(line + 1) * fullWidth > rawbuffer.size())
        return;
    const quint16 *rawLine = rawbuffer.data() + (size_t)line * fullWidth;
    const auto raw = [&](int xi) -> double {
        return (double)rawLine[left + xi];
    };

    if ((int)scratch_lurchPin.size() < width)
        scratch_lurchPin.resize(width);
    double *pin = scratch_lurchPin.data();
    const float *exRow = lurchPinEnabled() ? exactCarrierRow(line) : nullptr;
    int pinCount = 0;
    for (int xi = 0; xi < width; ++xi) {
        pin[xi] = std::numeric_limits<double>::quiet_NaN();
        if (!exRow)
            continue;
        const float e = exRow[left + xi];
        if (!std::isfinite(e))
            continue;
        pin[xi]  = raw(xi) - (double)e;
        yOut[xi] = pin[xi];
        ++pinCount;
    }
    if (pinCount >= width)
        return;                     // fully certified: conservation answered

    // Along a phase chain the solution is
    //     Y(w) = [wA*|1-e^-iw|^2 * facts + wB * A(w)] / [wA*|1-e^-iw|^2 + wB]
    // so the platform dominates below w_c where 2*(1-cos w_c) = wB/wA. The chain
    // steps four samples, which puts w_c = 8*pi*f_c/fs.
    constexpr double wA = 1.0;
    const double wc = std::clamp(
        8.0 * M_PI * lurchPlatformCutoffMHz() / kSampleRateMHz, 1e-4, M_PI);
    const double wB = 2.0 * (1.0 - std::cos(wc));

    const int maxChain = (width + 3) / 4 + 1;
    if ((int)scratch_lurchWork.size() < 2 * maxChain)
        scratch_lurchWork.resize(2 * maxChain);
    double *cp = scratch_lurchWork.data();
    double *dp = cp + maxChain;

    const auto solveChains = [&]() {
        for (int k = 0; k < 4; ++k) {
            const int N = (width - k + 3) / 4;
            if (N < 2)
                continue;
            for (int n = 0; n < N; ++n) {
                const int x = k + 4 * n;
                double diag, lower, upper, rhs;
                if (std::isfinite(pin[x])) {
                    diag = 1.0; lower = 0.0; upper = 0.0; rhs = pin[x];
                } else {
                    const bool hasNext = (n <= N - 2);
                    const bool hasPrev = (n >= 1);
                    diag  = wB + wA * ((hasNext ? 1.0 : 0.0) +
                                       (hasPrev ? 1.0 : 0.0));
                    lower = hasPrev ? -wA : 0.0;
                    upper = hasNext ? -wA : 0.0;
                    rhs   = wB * platform[x];
                    if (hasNext) rhs -= wA * (raw(x + 4) - raw(x));
                    if (hasPrev) rhs += wA * (raw(x) - raw(x - 4));
                }
                // Strictly diagonally dominant (diag = wB + sum|off|, wB > 0),
                // so Thomas is stable with no pivoting.
                const double denom = diag - lower * (n ? cp[n - 1] : 0.0);
                const double inv =
                    (std::fabs(denom) > 1e-12) ? (1.0 / denom) : 0.0;
                cp[n] = upper * inv;
                dp[n] = (rhs - lower * (n ? dp[n - 1] : 0.0)) * inv;
            }
            yOut[k + 4 * (N - 1)] = dp[N - 1];
            for (int n = N - 2; n >= 0; --n)
                yOut[k + 4 * n] = dp[n] - cp[n] * yOut[k + 4 * (n + 1)];
        }
    };

    solveChains();

    // FEASIBILITY, as a hard box. composite[i] + composite[i+-2] = Y[i] +
    // Y[i+-2] is carrier-free and exact, so with luma globally bounded each
    // available neighbour pins Y[i] to an interval. The global bounds are
    // deliberately generous: a FeasibleInterval forbids what cannot be true and
    // says nothing about which surviving value is preferred, and the failure
    // this formulation is actually exposed to is a chain of cumulative
    // differences DRIFTING, which a wide box still catches.
    //
    // Violators are pinned and the chain re-solved, so a clamp propagates along
    // the phase chain rather than truncating one sample in isolation.
    const double yLo = (double)videoParameters.black16bIre -  25.0 * irescale;
    const double yHi = (double)videoParameters.black16bIre + 125.0 * irescale;

    for (int pass = 0; pass < 2; ++pass) {
        int clamped = 0;
        for (int xi = 0; xi < width; ++xi) {
            if (std::isfinite(pin[xi]))
                continue;
            double nb[2];
            int nbCount = 0;
            if (xi - 2 >= 0)    nb[nbCount++] = raw(xi - 2);
            if (xi + 2 < width) nb[nbCount++] = raw(xi + 2);
            const lddecode::FeasibleInterval f =
                lddecode::lumaFeasibleFromPairSums(raw(xi), nb, nbCount,
                                                   yLo, yHi);
            if (!f.valid())
                continue;
            const double c = f.clamp(yOut[xi]);
            if (c != yOut[xi]) {
                pin[xi] = c;
                ++clamped;
            }
        }
        if (clamped == 0)
            break;
        solveChains();
    }
}

// Build the certified carrier ladder from the shared carrier record.
//
// The locked orchestration is intentionally single-pass here:
//   1. buildCarrierAnalysis() harvests canonical bandpass and schedule
//      conformance data.
//   2. buildPhaseCorrected1D() builds the corrected 1D baseline.
//   3. buildCertifiedCarrierLadder() calls buildCertifiedCarrierStage()
//      once, after the corrected 1D baseline exists.
//
// The certified plane is the primary product. If --luma-witness is enabled,
// the same pass also performs its four-view attribution and creates its
// private retracted-luma view. Uncovered iceberg recovery is luma-only and
// does not consume these carrier-model controls.
static int ldcdRetractedSourceMode()
{
    static const int mode = []{
        const char *s = std::getenv("LDCD_RETRACTED_SOURCE");
        if (!s)
            return 3;
        if (s[0] == 'a')
            return 3;
        if (s[0] == 'c')
            return 0;
        if (s[0] == 'n')
            return 1;
        if (s[0] == 'p')
            return 2;
        return 3;
    }();
    return mode;
}

static bool ldcdPhaseSnapOn()
{
    static const bool on = []{
        const char *e = std::getenv("LDCD_PHASE_SNAP");
        return !(e && std::atoi(e) == 0);
    }();
    return on;
}

// Fact anchor switch (LDCD_FACT_FIT, default on): shared between the Pass
// 1.7 stamp and the Pass 2 valve, which must agree on whether the def-line
// fit rows are facts. One resolution, per the single-producer rule.
static bool ldcdFactFitOn()
{
    static const bool on = []{
        const char *e = std::getenv("LDCD_FACT_FIT");
        return !(e && std::atoi(e) == 0);
    }();
    return on;
}

// Working-space snap: per W=8 window the estimator keeps its amplitude and
// takes the reference's phase, fading in above the amplitude floor (the
// phase of a near-zero carrier is meaningless -- a clamp on an
// impossible). est and out must not alias: windows read est after earlier
// out writes. clampRatio bounds |est|/|ref| to [0.5, 2.0] -- used when est
// is the fit alone, whose window amplitude is wild (ratio p10 0.45 / p90
// 4.1 vs truth): the reference is CERTIFIED carrier, so an estimator
// amplitude far outside its neighbourhood is estimator failure, not
// content. A hull on the impossible, not a value substitution.
static void ldcdApplyPhaseSnap(const std::vector<double> &est,
                               const std::vector<double> &ref,
                               std::vector<double> &out,
                               int width, double irescale,
                               double ampMinIRE, double ampTauIRE,
                               bool clampRatio,
                               const double *gate)
{
    // Centered 9-tap, half-weight ends (integer centroid; the old [-3,+4]
    // W=8 window estimated phase/amplitude half a sample right of the
    // sample it snapped). Endpoint classes coincide at 4fSC, so the class
    // population and the 0.25 normalisation are unchanged.
    constexpr int kSnapHalf = 4;
    for (int xi = kSnapHalf; xi < width - kSnapHalf; ++xi) {
        double eI = 0.0, eQ = 0.0, rI = 0.0, rQ = 0.0;
        bool ok = true;
        for (int k = xi - kSnapHalf; k <= xi + kSnapHalf && ok; ++k) {
            const double b = ref[k];
            if (!std::isfinite(b)) { ok = false; break; }
            // est carries NaN wherever its estimator declined to form (the
            // comp rung publishes nothing rather than substituting a fit).
            // Refusing the window is deliberate: the snap needs a complete
            // amplitude read, and the tail below leaves those samples at
            // their unsnapped value -- or NaN, which retreats correctly.
            if (!std::isfinite(est[k])) { ok = false; break; }
            static const int cB[4] = { 1, 0, -1, 0 };
            static const int sB[4] = { 0, 1, 0, -1 };
            const double w =
                (k == xi - kSnapHalf || k == xi + kSnapHalf) ? 0.5 : 1.0;
            const int ph = k & 3;
            eI += w * est[k] * cB[ph]; eQ += w * est[k] * sB[ph];
            rI += w * b * cB[ph];      rQ += w * b * sB[ph];
        }
        if (!ok) continue;
        // I/Q sums over W=8 carry ~4x the waveform amplitude; 0.25
        // normalizes for the IRE thresholds. The ratio needs no
        // normalization.
        const double aRefRaw = std::hypot(rI, rQ);
        const double aRef = aRefRaw * 0.25;
        if (aRef < ampMinIRE * irescale) continue;
        const double aEstRaw = std::hypot(eI, eQ);
        double ratio = aEstRaw / aRefRaw;
        if (clampRatio) ratio = std::clamp(ratio, 0.5, 2.0);
        const double snapped = ratio * ref[xi];
        // Ramp from the floor, continuous at it -- a step in a blend
        // weight is a contour.
        double sf = 1.0 - std::exp(
            -(aRef - ampMinIRE * irescale) /
             (ampTauIRE * irescale));
        if (gate) sf *= std::clamp(gate[xi], 0.0, 1.0);
        out[xi] = sf * snapped + (1.0 - sf) * est[xi];
    }
    for (int xi = 0; xi < width; ++xi)
        if (!std::isfinite(out[xi]))
            out[xi] = est[xi];
}

// Iceberg correspondence helpers.  They operate on certified-neighbour luma
// solely to locate and render a luma return; none produces carrier.
namespace {
// Offset search for the tween chase: demeaned NCC of the carrier-free
// platform rows over a 48-sample window, lags +-12 (sized for motion
// across a telecine cycle), parabolic refinement, gated on window energy
// and peak correlation. Correspondence PRIOR only -- it narrows the anchor
// search; it never supplies a rendered value.
double ldcdP6WindowLag(const double *est, const double *ref,
                       int w0, int W, double minRms)
{
    constexpr int kLag = 12;
    double meanE = 0.0, meanR = 0.0;
    for (int i = 0; i < W; ++i) { meanE += est[w0 + i]; meanR += ref[w0 + i]; }
    meanE /= W; meanR /= W;
    double eE = 0.0;
    for (int i = kLag; i < W - kLag; ++i) {
        const double de = est[w0 + i] - meanE;
        eE += de * de;
    }
    const int inner = W - 2 * kLag;
    if (eE / inner < minRms * minRms)
        return std::numeric_limits<double>::quiet_NaN();
    double corr[2 * kLag + 1];
    int best = -1;
    double bestC = 0.5;
    for (int l = -kLag; l <= kLag; ++l) {
        double dot = 0.0, er = 0.0;
        for (int i = kLag; i < W - kLag; ++i) {
            const double de = est[w0 + i] - meanE;
            const double dr = ref[w0 + i + l] - meanR;
            dot += de * dr;
            er  += dr * dr;
        }
        const double c = dot / std::sqrt(std::max(1e-12, eE * er));
        corr[l + kLag] = c;
        if (c > bestC) { bestC = c; best = l + kLag; }
    }
    if (best <= 0 || best >= 2 * kLag)
        return std::numeric_limits<double>::quiet_NaN();
    const double A = corr[best - 1], B = corr[best], Cc = corr[best + 1];
    const double den = A + Cc - 2.0 * B;
    const double d = (std::fabs(den) < 1e-12)
        ? 0.0 : std::clamp((A - Cc) / (2.0 * den), -0.5, 0.5);
    return (double)(best - kLag) + d;
}

// TEMPORARY TELEMETRY (LDCD_ICEBERG_STATS=1): per-frame engagement of the
// anchor tween, so selectivity is measured rather than inferred. Strip
// with the campaign's instruments.
struct LdcdIceStat {
    long anchors = 0, matched = 0, licensed = 0, renderedSamples = 0;
    // Per-gate death census (temporary, LDCD_ICEBERG_STATS). 7.5% of anchors
    // match and the two ACCEPTANCE gates were measured near-inert, so the
    // refusals are upstream in candidate generation. This counts which gate
    // actually spends them. Strip when the question closes.
    long withPrior = 0, pairs = 0, dRadius = 0, dSlopeSign = 0,
         dSlopeRatio = 0, dRms = 0, dAmbig = 0, noCand = 0,
         dPlatVet = 0, lagFilled = 0, inhMatched = 0;
};
thread_local LdcdIceStat gIceStat;

// Catmull-Rom evaluation of a row at a fractional position; NaN outside
// support or across a NaN sample. Shared by the tween delivery and the P6
// referee (one implementation, per the no-duplicate-math law).
template <typename T>
double ldcdP6CrEval(const T *row, int W, double pos)
{
    const int base = (int)std::floor(pos);
    if (base < 1 || base + 2 >= W)
        return std::numeric_limits<double>::quiet_NaN();
    const double y0 = row[base - 1], y1 = row[base],
                 y2 = row[base + 1], y3 = row[base + 2];
    if (!std::isfinite(y0) || !std::isfinite(y1) ||
        !std::isfinite(y2) || !std::isfinite(y3))
        return std::numeric_limits<double>::quiet_NaN();
    const double t = pos - base;
    const double t2 = t * t, t3 = t2 * t;
    return 0.5 * ((2.0 * y1) + (-y0 + y2) * t +
                  (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t2 +
                  (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t3);
}

} // namespace

bool Comb::FrameBuffer::buildIcebergReturn(const FrameBuffer *prevF,
                                           const FrameBuffer *nextF)
{
    // This current-time stage publishes recovered LUMA for cross-colour
    // return.  Clear it before every attempt so recycled frame buffers can
    // never retain a prior frame's opinion.
    icebergRecoveredY_flat.clear();
    icebergReturnWeight_flat.clear();
    static const bool icebergTween = []{
        const char *e = std::getenv("LDCD_ICEBERG");
        return e && std::atoi(e) != 0;
    }();
    if (!icebergTween) return false;
    if (frameHasExactCoverage()) return false;
    const FrameBuffer *nb[2] = {
        (prevF && prevF->frameHasExactCoverage()) ? prevF : nullptr,
        (nextF && nextF->frameHasExactCoverage()) ? nextF : nullptr,
    };
    // Both certified neighbours are required. At stream or batch edges the
    // feature abstains and ordinary comb Y remains untouched.
    if (!nb[0] || !nb[1]) return false;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;
    if (width <= 0) return false;

    // THE TWEEN'S LOCATOR (author-directed, 2026-08-08). Correspondence --
    // "how far has the picture moved since the cover" -- was read off
    // lockedLumaSmooth, which is carrier-free BECAUSE it is smoothed. On a
    // surface whose detail is entirely fine, smoothing leaves nothing to
    // correlate: the cube face reads as a blank, no window locks, and the
    // matcher never even attempts the population this campaign exists for.
    //
    // The locator is the elementary notch [1,0,1]/2 on raw (author, 2026-08-08).
    // Its magnitude response is |cos w|: unity at DC, unity again at Nyquist,
    // zero only at fSC, 0.707 at half-fSC and at 3/2-fSC. So it passes the
    // finest horizontal detail in the raster untouched in magnitude and pays
    // only in the band immediately around the carrier -- the opposite trade
    // from the smoothed platform, which keeps the carrier out by throwing the
    // detail away. The wider +-2 form (raw minus the 0.50c-0.25(m2+p2)
    // bandpass) was tried first and moved engagement under half a percent;
    // its null goes as cos^2 and takes too much of the neighbourhood with it.
    //
    // The notch's own faults -- summits sitting inside the band are absent
    // from it, and it carries the leak doublets -- are the cheap ones here:
    // the same operator runs on this frame and on both covers, so those
    // artefacts CORRELATE rather than corrupt. It locates; it never supplies
    // shape. LDCD_ICE_NOTCH=0 restores the smoothed-platform locator.
    std::vector<double> notchC(width), notchP(width), notchN(width);
    static const bool iceNotchLocator = []{
        const char *e = std::getenv("LDCD_ICE_NOTCH");
        return !e || std::atoi(e) != 0;
    }();

    // MATCH ACCEPTANCE, as two separately measurable levers (2026-08-08).
    // The anchor matcher refuses a pair on two grounds, and neither has been
    // asked to prove it earns its refusals:
    //
    //   AMBIGUITY  a pair dies when the runner-up snippet fits nearly as
    //              well (within 25%) from a different place. On a dense
    //              repetitive surface -- the cube face -- near-ties are the
    //              NORMAL case, so this gate would refuse hardest exactly
    //              where the content is densest, however good the locator.
    //   FIT        a pair dies when the best snippet difference exceeds
    //              8 IRE rms.
    //
    // LDCD_ICE_AMBIG=0 drops the first; LDCD_ICE_RMS=<IRE> moves the second.
    // The question being asked is whether ambiguity-aversion bought real
    // protection or only cost good locks -- so it has to be measured with
    // the veto OFF and the result LOOKED AT, not argued from the count.
    static const bool iceAmbigVeto = []{
        const char *e = std::getenv("LDCD_ICE_AMBIG");
        return !e || std::atoi(e) != 0;
    }();
    static const double iceMatchRmsIRE = []{
        const char *e = std::getenv("LDCD_ICE_RMS");
        const double v = e ? std::atof(e) : 8.0;
        return (v > 0.0) ? v : 8.0;
    }();
    // BLIND SEARCH WIDTH, 24 px (author-endorsed 2026-08-08). Measured on
    // the cube face: 12 -> 24 is nearly all the coverage the widening has to
    // give, while 36 and 48 add only scraps on crests already served and keep
    // paying -- ambiguity refusals run 10% / 19% / 30% / 31% of matches
    // across 12 / 24 / 36 / 48. 24 buys the new territory before the pool
    // stops getting cleaner. LDCD_ICE_RADIUS scales it (0.5 = the old 12).
    constexpr double kIceBlindRadiusPx = 24.0;
    static const double iceBlindRadiusScale = []{
        const char *e = std::getenv("LDCD_ICE_RADIUS");
        const double v = e ? std::atof(e) : 1.0;
        return (v > 0.0) ? v : 1.0;
    }();
    // Platform vet on the matched pool. LDCD_ICE_PLATVET=0 disables it,
    // so the widening can be measured with and without its referee.
    static const bool icePlatVet = []{
        const char *e = std::getenv("LDCD_ICE_PLATVET");
        return !e || std::atoi(e) != 0;
    }();
    static const double icePlatFloorIRE = []{
        const char *e = std::getenv("LDCD_ICE_PLATFLOOR");
        const double v = e ? std::atof(e) : 2.0;
        return (v > 0.0) ? v : 2.0;
    }();
    static const double icePlatTolIRE = []{
        const char *e = std::getenv("LDCD_ICE_PLATTOL");
        const double v = e ? std::atof(e) : 3.0;
        return (v > 0.0) ? v : 3.0;
    }();

    const size_t iceSize = static_cast<size_t>(demodLines) * demodWidth;
    icebergRecoveredY_flat.assign(
        iceSize, std::numeric_limits<float>::quiet_NaN());
    icebergReturnWeight_flat.assign(iceSize, 0.0f);
    bool anyIcebergReturn = false;

    for (int line = firstLine; line < lastLine; ++line) {
        const quint16 *rawLine = rawbuffer.data() +
            static_cast<size_t>(line) * videoParameters.fieldWidth;
        // Build a symmetric certified-luma roster.  Both covers participate
        // on every line so provenance cannot alternate with field parity.
        {
            // PARITY LAW: exact consumption must reach both parities
            // together. A direct-only reference alternates source film
            // frames line-by-line (even lines from prev, odd from next)
            // and any motion becomes a LINE-ALTERNATING luma error --
            // measured +9.4% pillar lineAlt in the v1 build. So every
            // line's Lhat draws from BOTH neighbours symmetrically:
            // direct same-line from the parity that covers it, bracket
            // mean from the other; motion error goes common-mode.
            //
            // THE BRACKET IS NOT A COMPROMISE -- IT BUYS UNIFORM
            // PROVENANCE, AND THAT IS THE WHOLE MECHANISM. Every line
            // here carries the SAME mixture of prev/next and the SAME
            // mixture of direct/bracket, so no term in Lhat depends on
            // line parity and there is nothing for an error to alternate
            // with.
            //
            // FALSIFIED 2026-08-02, do not attempt a third time: the
            // "sharp direct + smooth delta" build. With D = L_prev -
            // L_next it is algebraically exact to keep the SHARP direct
            // value on every line and reach the true midpoint via
            //     M = L_prev - D/2   (lines the prev cover owns)
            //     M = L_next + D/2   (lines the next cover owns)
            // smoothing only D, which is a motion difference field and
            // therefore lawful to smooth. It measured +9.1%
            // parity-locked luma (LHAT A/B on the cube, the same
            // signature as v1's +9.4%) and chroma +2.5% AWAY from the
            // certified truth level. The reason generalises:
            //
            //   ANY TERM WHOSE SIGN OR PROVENANCE DEPENDS ON LINE PARITY
            //   CONVERTS ITS OWN ERROR INTO LINE ALTERNATION.
            //
            // D is applied as -D/2 on one parity and +D/2 on the other,
            // so wherever the smoothed D departs from the true D -- i.e.
            // wherever D itself carries vertical structure -- the residual
            // enters with alternating sign and IS alternation. Exactness
            // of the algebra does not help; only exactness of D would,
            // and that is motion compensation, not smoothing. Sharpness
            // at this site has to come from a better estimate of the
            // OTHER parity's luma (uniform provenance preserved), never
            // from a parity-signed correction.
            std::vector<int> lhatN(width, 0);
            // Each cover's own value, kept apart: their DISAGREEMENT is the
            // occlusion reading (see the guard below). Both are filled on
            // every line by direct-or-bracket, so provenance stays uniform
            // across parities and the guard cannot alternate.
            std::vector<double> sideVal[2];
            sideVal[0].assign(width, std::numeric_limits<double>::quiet_NaN());
            sideVal[1].assign(width, std::numeric_limits<double>::quiet_NaN());
            for (int side = 0; side < 2; ++side) {
                const FrameBuffer *fb = nb[side];
                if (!fb) continue;
                const float *pex0 = fb->exactCarrierRow(line);
                const float *pexU = (line - 1 >= firstLine)
                    ? fb->exactCarrierRow(line - 1) : nullptr;
                const float *pexD = (line + 1 < lastLine)
                    ? fb->exactCarrierRow(line + 1) : nullptr;
                const quint16 *rawNb = fb->rawbuffer.data()
                    + static_cast<size_t>(line) * videoParameters.fieldWidth;
                const quint16 *rawNbU = fb->rawbuffer.data()
                    + static_cast<size_t>(std::max(line - 1, 0)) *
                      videoParameters.fieldWidth;
                const quint16 *rawNbD = fb->rawbuffer.data()
                    + static_cast<size_t>(std::min(line + 1,
                            videoParameters.lastActiveFrameLine - 1)) *
                      videoParameters.fieldWidth;
                for (int xi = 0; xi < width; ++xi) {
                    const int h = left + xi;
                    if (pex0 && std::isfinite(pex0[h])) {
                        const double v = static_cast<double>(rawNb[h]) -
                                         static_cast<double>(pex0[h]);
                        lhatN[xi] += 1;
                        sideVal[side][xi] = v;
                    } else if (pexU && pexD &&
                               std::isfinite(pexU[h]) &&
                               std::isfinite(pexD[h])) {
                        const double lu = static_cast<double>(rawNbU[h]) -
                                          static_cast<double>(pexU[h]);
                        const double ld = static_cast<double>(rawNbD[h]) -
                                          static_cast<double>(pexD[h]);
                        const double v = 0.5 * (lu + ld);
                        lhatN[xi] += 1;
                        sideVal[side][xi] = v;
                    }
                }
            }
            // Both sides or nothing: a one-sided reference re-creates the
            // frame-source alternation the symmetric build exists to
            // prevent (stream edges lose the comb, keeping snap-only).
            // BUG FIXED 2026-08-02: this read `(nb[0] && nb[1]) ? 2 : 1`,
            // i.e. where only ONE neighbour was covered it happily built the
            // reference from that one side -- exactly the one-sided
            // direct-reference shape the parity law falsified (+9.4%
            // line-alternating luma), and precisely the opposite of what the
            // comment above it promises. Measured consequence: the frames
            // carrying the WORST temporal excursions (peaks 88-94 IRE) were
            // the batch-boundary frames running one-sided, and the occlusion
            // guard could not act on them either, because it needs two covers
            // to form a disagreement. Both sides or nothing, as stated.
            // ICEBERG DELIVERY, REBUILT (2026-08-06; opt-in LDCD_ICEBERG=1
            // pending the visual verdict). LAW: PARAMETERS INTERPOLATE; THE
            // RENDER HAPPENS ONCE (After Effects model, author-directed).
            // The first build violated law 4 by rendering both displaced
            // covers and cross-dissolving them (0.5*(vP+vN)) -- a
            // better-registered double exposure, whose sub-sample
            // misregistration blunted every corner (the author's diagnosis:
            // anchors' broken handles were being relaxed by the mix).
            //
            // Corrected construction, per line:
            //   1. ANCHORS: sharpest contrasts in each cover's certified
            //      luma (sub-sample position by parabola on |slope|).
            //   2. CORRESPONDENCE: prev<->next anchor matching, with the
            //      platform-chase lags as PRIOR only (evidence for the
            //      match; never a render input). No match -> abstain.
            //   3. TWEEN IN PARAMETER SPACE: position along the anchor's
            //      own motion vector at the temporal fraction (adjacent
            //      output frames at 24p: tau = 1/2); the anchor-centered
            //      SHAPE as an aligned lerp of the two covers' local
            //      profiles -- both re-sampled into anchor coordinates
            //      BEFORE the lerp, so the corner interpolates as a corner.
            //   4. THE PIN: this frame's own four-view membership evidence
            //      corrects the tweened position (bounded, ramped) and
            //      licenses the render (presence + position, both
            //      smoothstepped). No corroboration -> abstain.
            //   5. RENDER ONCE: the tweened shape placed at the tweened
            //      position, feathered into the reference. Nothing is ever
            //      superimposed; the bracket appears only as what the
            //      pipeline already was, at ramped abstention boundaries.
            // The outlawed mid-band displaced-average delivery is REMOVED
            // (not grandfathered by its P6 number); mid geometry arrives
            // only through the rendered anchors' side levels.
            static const bool iceStats = []{
                const char *e = std::getenv("LDCD_ICEBERG_STATS");
                return e && std::atoi(e) != 0;
            }();
            if (icebergTween && nb[0] && nb[1] && lockedLumaCacheValid &&
                nb[0]->lockedLumaCacheValid && nb[1]->lockedLumaCacheValid) {
                const double *platC = lockedLumaSmooth_line(line);
                const double *platP = nb[0]->lockedLumaSmooth_line(line);
                const double *platN = nb[1]->lockedLumaSmooth_line(line);

                // Kept as the BELOW-fSC platform even when the locator below
                // switches to the notch: the match is made on the notch, so
                // the platform is a different band and can referee it.
                const double *vetP = platP;
                const double *vetN = platN;

                // Does the platform corroborate this pairing? Compares the
                // two covers' platform shape at the two anchor positions,
                // level removed (a legitimate lighting change moves the
                // level, not the shape). Returns TRUE -- pass -- whenever it
                // cannot see: off the row, or too flat to carry an opinion.
                auto platformAgrees = [&](double posA, double posB) -> bool {
                    if (!vetP || !vetN) return true;
                    constexpr int kPv = 4;
                    constexpr int nPv = 2 * kPv + 1;
                    double a[nPv], b[nPv], ma = 0.0, mb = 0.0;
                    double aLo = 1e300, aHi = -1e300;
                    double bLo = 1e300, bHi = -1e300;
                    for (int k = -kPv; k <= kPv; ++k) {
                        const double va =
                            ldcdP6CrEval(vetP, width, posA + k);
                        const double vb =
                            ldcdP6CrEval(vetN, width, posB + k);
                        if (!std::isfinite(va) || !std::isfinite(vb))
                            return true;
                        a[k + kPv] = va; b[k + kPv] = vb;
                        ma += va; mb += vb;
                        aLo = std::min(aLo, va); aHi = std::max(aHi, va);
                        bLo = std::min(bLo, vb); bHi = std::max(bHi, vb);
                    }
                    ma /= nPv; mb /= nPv;
                    const double floorAmp = icePlatFloorIRE * irescale;
                    if ((aHi - aLo) < floorAmp || (bHi - bLo) < floorAmp)
                        return true;
                    double ss = 0.0;
                    for (int k = 0; k < nPv; ++k) {
                        const double d = (a[k] - ma) - (b[k] - mb);
                        ss += d * d;
                    }
                    return std::sqrt(ss / nPv) <=
                           icePlatTolIRE * irescale;
                };

                // Swap the smoothed platform for the notch as the locator.
                // Falls back silently to the platform if any of the three
                // bandpass rows is absent, so a missing row costs the old
                // behaviour rather than the tween.
                if (iceNotchLocator && left >= 1 &&
                    left + width + 1 <= videoParameters.fieldWidth) {
                    const size_t stride = videoParameters.fieldWidth;
                    const quint16 *rawC = rawLine;
                    const quint16 *rawP = nb[0]->rawbuffer.data()
                        + static_cast<size_t>(line) * stride;
                    const quint16 *rawN = nb[1]->rawbuffer.data()
                        + static_cast<size_t>(line) * stride;
                    auto notchAt = [](const quint16 *r, int i) {
                        return 0.5 * (static_cast<double>(r[i - 1]) +
                                      static_cast<double>(r[i + 1]));
                    };
                    for (int x = 0; x < width; ++x) {
                        const int i = left + x;
                        notchC[x] = notchAt(rawC, i);
                        notchP[x] = notchAt(rawP, i);
                        notchN[x] = notchAt(rawN, i);
                    }
                    platC = notchC.data();
                    platP = notchP.data();
                    platN = notchN.data();
                }
                const auto *evRow = coarseYEvidence_line(line);
                constexpr int kTwWin = 48, kTwStride = 16;
                const double minRms = 1.5 * irescale;

                // Chase lags at window centers: correspondence prior.
                const int nWin =
                    std::max(0, (width - 28 - kTwWin) / kTwStride + 1);
                std::vector<double> lagPA(nWin), lagNA(nWin);
                std::vector<char> wOk(nWin, 0);
                for (int i = 0; i < nWin; ++i) {
                    const int w0 = 14 + i * kTwStride;
                    lagPA[i] = ldcdP6WindowLag(platC, platP, w0, kTwWin,
                                               minRms);
                    lagNA[i] = ldcdP6WindowLag(platC, platN, w0, kTwWin,
                                               minRms);
                    wOk[i] = std::isfinite(lagPA[i]) &&
                             std::isfinite(lagNA[i]);
                }
                // REGIONAL TRACK (author, 2026-08-08): let a peak ride the
                // region's trend instead of being told its region is
                // unknown. A window that failed to lock used to report NO
                // PRIOR, so every anchor inside it dropped to the blind
                // search -- even when the windows two along had measured the
                // motion perfectly well. That is how a profound crest ends
                // up with no partner in range: nothing told the matcher
                // where to look, and a search centred on standstill misses
                // a feature that moved.
                //
                // Nothing new is tracked. The gaps in the lag field are
                // filled from the locked windows either side, which is the
                // smooth-motion assumption correspondence v2 already leans
                // on -- used here to SUPPLY a prior rather than only to
                // reject outliers. Cost is one pass over ~45 windows per
                // line, against the NCC searches already run.
                //
                // Two disciplines. A fill is never made across bracketing
                // locks that DISAGREE -- that is an occlusion or an object
                // boundary, where blind is the honest answer. And a fill is
                // marked as inherited, so a match made on borrowed evidence
                // is held to a tighter shape test than one made on a
                // measured lock. LDCD_ICE_LAGFILL=0 restores the old
                // lock-or-blind behaviour.
                static const bool iceLagFill = []{
                    const char *e = std::getenv("LDCD_ICE_LAGFILL");
                    return !e || std::atoi(e) != 0;
                }();
                std::vector<char> wInh(nWin, 0);
                if (iceLagFill && nWin > 0) {
                    constexpr int kMaxGapWin = 3;
                    const double lagTol = 2.0;
                    for (int i = 0; i < nWin; ++i) {
                        if (wOk[i]) continue;
                        int jL = -1, jR = -1;
                        for (int j = i - 1;
                             j >= 0 && i - j <= kMaxGapWin; --j)
                            if (wOk[j]) { jL = j; break; }
                        for (int j = i + 1;
                             j < nWin && j - i <= kMaxGapWin; ++j)
                            if (wOk[j]) { jR = j; break; }
                        if (jL >= 0 && jR >= 0) {
                            if (std::fabs(lagPA[jL] - lagPA[jR]) > lagTol ||
                                std::fabs(lagNA[jL] - lagNA[jR]) > lagTol)
                                continue;  // occlusion / boundary: stay blind
                            const double t = double(i - jL) / double(jR - jL);
                            lagPA[i] = lagPA[jL] +
                                       t * (lagPA[jR] - lagPA[jL]);
                            lagNA[i] = lagNA[jL] +
                                       t * (lagNA[jR] - lagNA[jL]);
                        } else if (jL >= 0) {
                            lagPA[i] = lagPA[jL]; lagNA[i] = lagNA[jL];
                        } else if (jR >= 0) {
                            lagPA[i] = lagPA[jR]; lagNA[i] = lagNA[jR];
                        } else {
                            continue;
                        }
                        wInh[i] = 1;
                        gIceStat.lagFilled++;
                    }
                }
                auto lagsAt = [&](double x, double &lp, double &ln,
                                  bool *inherited = nullptr) {
                    const int i = std::clamp(
                        (int)std::lround((x - 14.0 - kTwWin / 2.0) /
                                         kTwStride),
                        0, std::max(0, nWin - 1));
                    if (nWin <= 0 || !(wOk[i] || wInh[i])) return false;
                    lp = lagPA[i]; ln = lagNA[i];
                    if (inherited) *inherited = (wInh[i] != 0);
                    return true;
                };

                // Anchor extraction from a certified cover row.
                struct IceAnchor { double pos; double slope; };
                // THE STRONGEST CREST IN A NEIGHBOURHOOD WINS IT, not the
                // leftmost (author, 2026-08-08). The rule this replaces
                // walked left to right and blocked the two samples after
                // every acceptance -- so a mediocre crest one sample to the
                // left of a profound one took the slot, and the profound one
                // was never tested at all. On a face whose detail sits at
                // 2-3 sample pitch that is the normal case, not an edge
                // case: it sampled the texture on a phase-arbitrary grid
                // instead of picking its peaks, and the anchors that
                // survived were not the strongest ones. Collect every
                // candidate first, then suppress within +-2 by SLOPE
                // MAGNITUDE, ties to the earlier sample so the result does
                // not depend on scan order. LDCD_ICE_NMS=0 restores the old
                // first-come rule for A/B.
                static const bool iceNms = []{
                    const char *e = std::getenv("LDCD_ICE_NMS");
                    return !e || std::atoi(e) != 0;
                }();
                struct IceCand { double pos; double slope; double a; int x; };
                auto extract = [&](const std::vector<double> &row,
                                   std::vector<IceAnchor> &out) {
                    out.clear();
                    const double slopeFloor = 2.0 * irescale;
                    std::vector<IceCand> cand;
                    int lastA = -3;
                    for (int x = 6; x < width - 6; ++x) {
                        if (!iceNms && x - lastA < 3) continue;
                        if (!std::isfinite(row[x - 2]) ||
                            !std::isfinite(row[x - 1]) ||
                            !std::isfinite(row[x])     ||
                            !std::isfinite(row[x + 1]) ||
                            !std::isfinite(row[x + 2]))
                            continue;
                        const double sM = 0.5 * (row[x]     - row[x - 2]);
                        const double s0 = 0.5 * (row[x + 1] - row[x - 1]);
                        const double sP = 0.5 * (row[x + 2] - row[x]);
                        const double a = std::fabs(s0);
                        if (a < slopeFloor) continue;
                        if (a < std::fabs(sM) || a <= std::fabs(sP))
                            continue;
                        const double den =
                            std::fabs(sM) + std::fabs(sP) - 2.0 * a;
                        const double d = (std::fabs(den) < 1e-12) ? 0.0 :
                            std::clamp((std::fabs(sM) - std::fabs(sP)) /
                                       (2.0 * den), -0.5, 0.5);
                        lastA = x;
                        cand.push_back({ (double)x + d, s0, a, x });
                    }
                    if (!iceNms) {
                        for (const IceCand &c : cand)
                            out.push_back({ c.pos, c.slope });
                        return;
                    }
                    const int n = (int)cand.size();
                    for (int i = 0; i < n; ++i) {
                        bool wins = true;
                        for (int j = i - 1;
                             j >= 0 && cand[i].x - cand[j].x <= 2; --j) {
                            if (cand[j].a >= cand[i].a) { wins = false; break; }
                        }
                        if (wins) {
                            for (int j = i + 1;
                                 j < n && cand[j].x - cand[i].x <= 2; ++j) {
                                if (cand[j].a > cand[i].a) {
                                    wins = false;
                                    break;
                                }
                            }
                        }
                        if (wins)
                            out.push_back({ cand[i].pos, cand[i].slope });
                    }
                };
                std::vector<IceAnchor> ancP, ancN;
                extract(sideVal[0], ancP);
                extract(sideVal[1], ancN);

                constexpr int kSnip = 5;   // shape support: pos +- 5
                // SEGMENTS (2026-08-06): matched anchors are KNOTS of a
                // per-line piecewise curve; spans render once each,
                // assignment never accumulation (see the span pass below).
                struct IceKnot {
                    double q, pA, pB, w;
                    double shape[2 * kSnip + 1];
                };
                std::vector<IceKnot> knots;

                // Anchor-centered profiles, once per anchor.
                auto snipOf = [&](const std::vector<double> &row, double pos,
                                  double *out) -> bool {
                    for (int k = -kSnip; k <= kSnip; ++k) {
                        const double v = ldcdP6CrEval(
                            row.data(), width, pos + k);
                        if (!std::isfinite(v)) return false;
                        out[k + kSnip] = v;
                    }
                    return true;
                };
                const int nP = (int)ancP.size(), nN = (int)ancN.size();
                std::vector<std::array<double, 2 * kSnip + 1>> snipP(nP),
                    snipN(nN);
                std::vector<char> okP(nP, 0), okN(nN, 0);
                for (int i = 0; i < nP; ++i)
                    okP[i] = snipOf(sideVal[0], ancP[i].pos,
                                    snipP[i].data());
                for (int j = 0; j < nN; ++j)
                    okN[j] = snipOf(sideVal[1], ancN[j].pos,
                                    snipN[j].data());

                // CORRESPONDENCE v2 (2026-08-06, after the segments
                // verdict): the platform-chase prior fails exactly on
                // platform-flat texture (the cube face: all detail is HF,
                // which lockedLumaSmooth excludes), so the population the
                // author pointed at never even attempted matching. The
                // match test is now SHAPE ITSELF — aligned-snippet rms,
                // stronger than any prior and available per candidate pair
                // — with the prior narrowing the search where it exists and
                // a bounded blind search where it does not. Two abstention
                // disciplines guard the blind form: a near-tie at a
                // distinct position (periodic texture's false-match mode)
                // abstains, and every match must agree with its
                // neighbours' displacement (the motion field is smooth;
                // isolated matches without consensus fall back to
                // requiring the prior, the old regime as floor).
                struct IceMatch {
                    int ai, bi;
                    double disp, maxDiff;
                    bool hadPrior;
                };
                std::vector<IceMatch> mats;
                for (int i = 0; i < nP; ++i) {
                    gIceStat.anchors++;
                    if (!okP[i]) continue;
                    const IceAnchor &A = ancP[i];
                    double lp = 0.0, ln = 0.0;
                    bool priorInherited = false;
                    const bool havePrior =
                        lagsAt(A.pos, lp, ln, &priorInherited);
                    const double cPred =
                        havePrior ? (A.pos - lp + ln) : A.pos;
                    // LDCD_ICE_RADIUS scales the blind search only, to
                    // separate "the partner exists but sits further than we
                    // look" from "the partner was never detected in the
                    // other cover at all". Those need opposite cures.
                    // A MEASURED lock pins: its window correlated, so 3
                    // samples is the honest tolerance. An INHERITED lag only
                    // says which way the region went -- the peak rides that
                    // trend, it is not fixed to it -- so it keeps the blind
                    // width and merely re-centres it. Centring a wide search
                    // on the trend is strictly better than centring it on
                    // standstill, which is what the old no-prior case did.
                    const double radius =
                        (havePrior && !priorInherited)
                            ? 3.0
                            : (kIceBlindRadiusPx * iceBlindRadiusScale);
                    int best = -1, second = -1;
                    double bestS = 1e300, secondS = 1e300, bestMax = 0.0;
                    if (havePrior) gIceStat.withPrior++;
                    for (int j = 0; j < nN; ++j) {
                        if (!okN[j]) continue;
                        const IceAnchor &B = ancN[j];
                        gIceStat.pairs++;
                        if (std::fabs(B.pos - cPred) > radius) {
                            gIceStat.dRadius++; continue;
                        }
                        if ((A.slope > 0) != (B.slope > 0)) {
                            gIceStat.dSlopeSign++; continue;
                        }
                        const double lo = std::min(std::fabs(A.slope),
                                                   std::fabs(B.slope));
                        const double hi = std::max(std::fabs(A.slope),
                                                   std::fabs(B.slope));
                        if (lo < 0.5 * hi) {
                            gIceStat.dSlopeRatio++; continue;
                        }
                        double ss = 0.0, mx = 0.0;
                        for (int k = 0; k < 2 * kSnip + 1; ++k) {
                            const double d = snipP[i][k] - snipN[j][k];
                            ss += d * d;
                            mx = std::max(mx, std::fabs(d));
                        }
                        const double rms =
                            std::sqrt(ss / (2 * kSnip + 1));
                        if (rms < bestS) {
                            secondS = bestS; second = best;
                            bestS = rms; best = j; bestMax = mx;
                        } else if (rms < secondS) {
                            secondS = rms; second = j;
                        }
                    }
                    if (best < 0) { gIceStat.noCand++; continue; }
                    // A match made on an INHERITED lag rests on the
                    // region's trend rather than on a lock of its own, so
                    // it must clear a tighter shape test to stand.
                    const double rmsCap = iceMatchRmsIRE * irescale *
                                          (priorInherited ? 0.75 : 1.0);
                    if (bestS > rmsCap) {
                        gIceStat.dRms++; continue;
                    }
                    if (priorInherited) gIceStat.inhMatched++;
                    // PLATFORM VET (author, 2026-08-08). The wide blind
                    // search reaches far enough to pair an anchor with a
                    // stranger, so the pool needs a referee that did not
                    // help choose it. The below-fSC platform is one: the
                    // match was made on the notch, a different band. Where
                    // the platform is flat it has nothing to say and passes
                    // -- which is the platform-flat population the widening
                    // just bought, so the vet costs it nothing. It convicts
                    // only where it carries shape and that shape disagrees.
                    if (icePlatVet && !platformAgrees(A.pos,
                                                      ancN[best].pos)) {
                        gIceStat.dPlatVet++; continue;
                    }
                    if (iceAmbigVeto &&
                        second >= 0 && secondS <= bestS * 1.25 &&
                        std::fabs(ancN[second].pos - ancN[best].pos) > 1.5) {
                        gIceStat.dAmbig++; continue;
                    }
                    IceMatch m;
                    m.ai = i; m.bi = best;
                    m.disp = ancN[best].pos - A.pos;
                    m.maxDiff = bestMax;
                    m.hadPrior = havePrior;
                    mats.push_back(m);
                }

                std::vector<char> keepM(mats.size(), 0);
                for (size_t m = 0; m < mats.size(); ++m) {
                    const double p0 = ancP[mats[m].ai].pos;
                    double ds[33]; int nd = 0;
                    for (size_t o = 0; o < mats.size() && nd < 33; ++o)
                        if (std::fabs(ancP[mats[o].ai].pos - p0) <= 32.0)
                            ds[nd++] = mats[o].disp;
                    if (nd >= 3) {
                        std::sort(ds, ds + nd);
                        keepM[m] =
                            std::fabs(mats[m].disp - ds[nd / 2]) <= 2.0;
                    } else {
                        keepM[m] = mats[m].hadPrior;
                    }
                }

                // Pass 2: consensus-guided rematch. Periodic texture is
                // structurally ambiguous to a blind search (the near-tie
                // abstention above is correct there), but once the
                // confident seeds have voted, the LOCAL CONSENSUS
                // DISPLACEMENT disambiguates: a previously-ambiguous
                // anchor re-matches inside a tight window around the
                // consensus, where a quasi-periodic face offers at most
                // one candidate. Each cover anchor may be claimed once.
                {
                    std::vector<char> matchedA(nP, 0), takenN(nN, 0);
                    for (size_t m = 0; m < mats.size(); ++m)
                        if (keepM[m]) {
                            matchedA[mats[m].ai] = 1;
                            takenN[mats[m].bi] = 1;
                        }
                    for (int i = 0; i < nP; ++i) {
                        if (matchedA[i] || !okP[i]) continue;
                        const IceAnchor &A = ancP[i];
                        double ds[33]; int nd = 0;
                        for (size_t o = 0; o < mats.size() && nd < 33; ++o)
                            if (keepM[o] &&
                                std::fabs(ancP[mats[o].ai].pos - A.pos)
                                    <= 32.0)
                                ds[nd++] = mats[o].disp;
                        if (nd < 2) continue;
                        std::sort(ds, ds + nd);
                        const double med = ds[nd / 2];
                        const double cPred = A.pos + med;
                        int best = -1;
                        double bestS = 1e300, bestMax = 0.0;
                        for (int j = 0; j < nN; ++j) {
                            if (!okN[j] || takenN[j]) continue;
                            const IceAnchor &B = ancN[j];
                            if (std::fabs(B.pos - cPred) > 2.5) continue;
                            if ((A.slope > 0) != (B.slope > 0)) continue;
                            const double lo = std::min(
                                std::fabs(A.slope), std::fabs(B.slope));
                            const double hi = std::max(
                                std::fabs(A.slope), std::fabs(B.slope));
                            if (lo < 0.5 * hi) continue;
                            double ss = 0.0, mx = 0.0;
                            for (int k = 0; k < 2 * kSnip + 1; ++k) {
                                const double d =
                                    snipP[i][k] - snipN[j][k];
                                ss += d * d;
                                mx = std::max(mx, std::fabs(d));
                            }
                            const double rms =
                                std::sqrt(ss / (2 * kSnip + 1));
                            if (rms < bestS) {
                                bestS = rms; best = j; bestMax = mx;
                            }
                        }
                        if (best < 0 || bestS > iceMatchRmsIRE * irescale)
                            continue;
                        IceMatch m2;
                        m2.ai = i; m2.bi = best;
                        m2.disp = ancN[best].pos - A.pos;
                        m2.maxDiff = bestMax;
                        m2.hadPrior = true;
                        mats.push_back(m2);
                        keepM.push_back(1);
                        matchedA[i] = 1;
                        takenN[best] = 1;
                    }
                }

                for (size_t m = 0; m < mats.size(); ++m) {
                    if (!keepM[m]) continue;
                    const IceAnchor &A = ancP[mats[m].ai];
                    const IceAnchor &Mn = ancN[mats[m].bi];
                    gIceStat.matched++;

                    constexpr double tau = 0.5;
                    double q = A.pos + tau * (Mn.pos - A.pos);

                    // The pin: presence + position from this frame's OWN
                    // membership evidence at the tweened site. Bounded
                    // position correction; ramped license; no evidence ->
                    // abstain.
                    if (!evRow) continue;
                    const int qi = (int)std::lround(q);
                    if (qi < 6 || qi >= width - 6) continue;
                    const auto &ev = evRow[qi];
                    double sw = 0.0, swx = 0.0;
                    for (int v = 0; v < ev.viewCount; ++v) {
                        const auto &vw = ev.views[v];
                        const double w =
                            std::fabs((double)vw.membershipDeltaIRE) *
                            std::max(0.0f, vw.membershipSupport);
                        swx += w * ((double)vw.apertureCenter + 0.5 -
                                    (double)qi);
                        sw  += w;
                    }
                    if (sw <= 1e-9) continue;
                    const double pPar = (double)qi + swx / sw;
                    double wAmp = std::clamp((sw - 0.3) / 0.7, 0.0, 1.0);
                    wAmp = wAmp * wAmp * (3.0 - 2.0 * wAmp);
                    double wPos = 1.0 - std::clamp(
                        (std::fabs(pPar - q) - 0.75) / 1.25, 0.0, 1.0);
                    wPos = wPos * wPos * (3.0 - 2.0 * wPos);
                    const double lic = wAmp * wPos;
                    if (lic <= 0.0) continue;
                    gIceStat.licensed++;
                    q += std::clamp(pPar - q, -0.75, 0.75) * lic;

                    // Tweened shape from the cached aligned snippets; the
                    // match's own post-alignment disagreement ramps the
                    // render (content change, not motion).
                    double shape[2 * kSnip + 1];
                    for (int k = 0; k < 2 * kSnip + 1; ++k)
                        shape[k] = (1.0 - tau) * snipP[mats[m].ai][k] +
                                   tau * snipN[mats[m].bi][k];
                    double wShape = 1.0 - std::clamp(
                        (mats[m].maxDiff / irescale - 6.0) / 8.0,
                        0.0, 1.0);
                    wShape = wShape * wShape * (3.0 - 2.0 * wShape);
                    const double wAll = lic * wShape;
                    if (wAll <= 0.0) continue;

                    IceKnot K;
                    K.q = q; K.pA = A.pos; K.pB = Mn.pos; K.w = wAll;
                    std::copy(shape, shape + 2 * kSnip + 1, K.shape);
                    knots.push_back(K);
                }

                std::sort(knots.begin(), knots.end(),
                          [](const IceKnot &a, const IceKnot &b) {
                              return a.q < b.q;
                          });

                std::vector<double> vRow(width,
                    std::numeric_limits<double>::quiet_NaN());
                std::vector<double> wRow(width, 0.0);
                constexpr double kMaxSpan = 24.0;
                constexpr double tau = 0.5;

                // Inter-knot spans: one render per sample, weight linear
                // between the knot licenses (continuous across knots).
                for (size_t i = 0; i + 1 < knots.size(); ++i) {
                    const IceKnot &K1 = knots[i];
                    const IceKnot &K2 = knots[i + 1];
                    const double d = K2.q - K1.q;
                    if (d <= 1.0 || d > kMaxSpan) continue;
                    const double dpA = K2.pA - K1.pA;
                    const double dpB = K2.pB - K1.pB;
                    // Collapsed or crossing cover spans = occlusion or
                    // mismatch: abstain.
                    if (dpA <= 0.5 || dpB <= 0.5) continue;
                    const int xs = (int)std::ceil(K1.q);
                    const int xe = (int)std::ceil(K2.q) - 1;
                    if (xs > xe) continue;
                    double tmp[32];
                    double spanDiff = 0.0;
                    bool ok = true;
                    for (int x = xs; x <= xe; ++x) {
                        const double t = ((double)x - K1.q) / d;
                        const double vP = ldcdP6CrEval(
                            sideVal[0].data(), width, K1.pA + t * dpA);
                        const double vN = ldcdP6CrEval(
                            sideVal[1].data(), width, K1.pB + t * dpB);
                        if (!std::isfinite(vP) || !std::isfinite(vN)) {
                            ok = false; break;
                        }
                        tmp[x - xs] = (1.0 - tau) * vP + tau * vN;
                        spanDiff = std::max(spanDiff, std::fabs(vP - vN));
                    }
                    if (!ok) continue;
                    double rampS = 1.0 - std::clamp(
                        (spanDiff / irescale - 6.0) / 8.0, 0.0, 1.0);
                    rampS = rampS * rampS * (3.0 - 2.0 * rampS);
                    if (rampS <= 0.0) continue;
                    for (int x = xs; x <= xe; ++x) {
                        const double t = ((double)x - K1.q) / d;
                        const double w =
                            (K1.w + t * (K2.w - K1.w)) * rampS;
                        if (w > wRow[x]) {
                            wRow[x] = w;
                            vRow[x] = tmp[x - xs];
                        }
                    }
                }

                // Stamps only where the chain is open: isolated knots and
                // chain ends. Assignment via the same winner rule, so a
                // seam sample takes the stronger single render, never a
                // mix.
                for (size_t i = 0; i < knots.size(); ++i) {
                    const IceKnot &K = knots[i];
                    const double leftGap =
                        (i > 0) ? K.q - knots[i - 1].q : 1e9;
                    const double rightGap =
                        (i + 1 < knots.size()) ? knots[i + 1].q - K.q : 1e9;
                    auto stampSide = [&](int dx0, int dx1) {
                        for (int dx = dx0; dx <= dx1; ++dx) {
                            const int u = (int)std::floor(K.q) + dx;
                            if (u < 0 || u >= width) continue;
                            const double rel = (double)u - K.q;
                            const double v = ldcdP6CrEval(
                                K.shape, 2 * kSnip + 1, rel + kSnip);
                            if (!std::isfinite(v)) continue;
                            double f = 1.0 - std::clamp(
                                (std::fabs(rel) - 2.5) / 2.5, 0.0, 1.0);
                            f = f * f * (3.0 - 2.0 * f);
                            const double w = K.w * f;
                            if (w > wRow[u]) {
                                wRow[u] = w;
                                vRow[u] = v;
                            }
                        }
                    };
                    if (leftGap > kMaxSpan)  stampSide(-kSnip, 0);
                    if (rightGap > kMaxSpan) stampSide(0, kSnip);
                }

                // Taper the WEIGHT field at every rendered-run edge so the
                // correction never appears or vanishes in one sample
                // (weight smoothing is lawful; the rendered VALUE is never
                // smoothed).
                {
                    std::vector<double> dist(width, 0.0);
                    double run = 0.0;
                    for (int x = 0; x < width; ++x) {
                        run = (wRow[x] > 0.0) ? run + 1.0 : 0.0;
                        dist[x] = run;
                    }
                    run = 0.0;
                    for (int x = width - 1; x >= 0; --x) {
                        run = (wRow[x] > 0.0) ? run + 1.0 : 0.0;
                        dist[x] = std::min(dist[x], run);
                    }
                    for (int x = 0; x < width; ++x) {
                        double t = std::clamp(dist[x] / 3.0, 0.0, 1.0);
                        wRow[x] *= t * t * (3.0 - 2.0 * t);
                    }
                }

                // Publish the completed luma value and its licence.  The Y
                // election owns the eventual feather against ordinary comb Y;
                // no complementary carrier is constructed here.
                for (int x = 0; x < width; ++x) {
                    if (wRow[x] <= 0.0 || lhatN[x] < 2) continue;
                    if (!std::isfinite(vRow[x])) continue;
                    const double w = std::min(1.0, wRow[x]);
                    const size_t oi = static_cast<size_t>(line) * demodWidth + x;
                    icebergRecoveredY_flat[oi] = static_cast<float>(vRow[x]);
                    icebergReturnWeight_flat[oi] = static_cast<float>(w);
                    anyIcebergReturn = true;
                    gIceStat.renderedSamples++;
                }
            }

        }
    }
    {
        static const bool iceStatsPrint = []{
            const char *e = std::getenv("LDCD_ICEBERG_STATS");
            return e && std::atoi(e) != 0;
        }();
        if (iceStatsPrint) {
            std::fprintf(stderr,
                "[ICE] frame %d: anchors %ld, matched %ld, licensed %ld, "
                "rendered %ld samples\n",
                (int)heldSeq1, gIceStat.anchors, gIceStat.matched,
                gIceStat.licensed, gIceStat.renderedSamples);
            std::fprintf(stderr,
                "[ICEGATE] frame %d: prior %ld/%ld anchors | pairs %ld -> "
                "radius %ld, slopeSign %ld, slopeRatio %ld | noCand %ld, "
                "rms %ld, platVet %ld, ambig %ld | lagFilled %ld, "
                "inhMatched %ld\n",
                (int)heldSeq1, gIceStat.withPrior, gIceStat.anchors,
                gIceStat.pairs, gIceStat.dRadius, gIceStat.dSlopeSign,
                gIceStat.dSlopeRatio, gIceStat.noCand, gIceStat.dRms,
                gIceStat.dPlatVet, gIceStat.dAmbig, gIceStat.lagFilled,
                gIceStat.inhMatched);
            gIceStat = LdcdIceStat();
        }
    }

    return anyIcebergReturn;
}

void Comb::FrameBuffer::buildCertifiedCarrierLadder(const FrameBuffer *prevF)
{
    buildCertifiedCarrierStage(prevF);
}


// ---------------------------------------------------------------------------
// Certified twin-bracket carrier hull. See feasibleband.h for the law, the
// measurement record and the falsified alternatives.
//
// Placement: CURRENT time, because it needs a covered NEIGHBOUR; the fit and
// the retraction ladder are built at load time from this frame alone, which is
// why the uncovered letters currently ride an unbounded fit ("B soft ... the
// open cross-frame transfer work", buildCertifiedCarrierStage). Measured:
// denying the merge wholesale (--dg-discard) leaves an uncovered frame's
// carrier bit-identical, so no certified fact reaches this object by any other
// route.
//
// ONE bracket per line parity is the ordinary 3:2 geometry -- the two nearest
// twins certify OPPOSITE parities, so each line of an uncovered frame has a
// single certified side, and kMarginOneSided is what applies there. A single
// neighbour is therefore sufficient and the guard below asks only for one.
// An earlier draft demanded both, because the (since falsified) inter-bracket
// motion precondition was measured BETWEEN them; that gate is gone and its
// requirement went with it.
//
// It clamps the PUBLISHED carrier rather than the fit. The certified statement
// is about carrier magnitude and does not care which stage produced the value,
// and clamping the publication keeps one story for every consumer instead of
// bounding a base that later stages re-inflate.
//
// Structurally inert without the inputs: no cadence roles -> no spare -> no
// exact channel -> no bracket -> unbounded interval everywhere. Under
// --dg-discard nothing exists to read and this cannot fire.
// ---------------------------------------------------------------------------
bool Comb::FrameBuffer::applyCertifiedCarrierHull(const FrameBuffer *prevF,
                                                  const FrameBuffer *nextF)
{
    // OPT-IN, not a default: one of four measured sources violates the bound
    // at 9.8% and the regime is not yet characterised (see feasibleband.h).
    // A false forbid confiscates real colour, so this does not ship on.
    static const bool hullOn = []{
        const char *e = std::getenv("LDCD_CARRIER_HULL");
        return e && std::atoi(e) != 0;
    }();
    static const auto hullEnvD = [](const char *n, double d) {
        const char *e = std::getenv(n); return e ? std::atof(e) : d;
    };
    // One bracket per line parity is the ordinary 3:2 geometry for the
    // uncovered letters (the two neighbouring twins certify OPPOSITE
    // parities), and one-sided needs the wider margin: 2.3% violation at 2.0
    // against 0.90% two-sided, 0.95% at 3.0.
    static const double kMarginOneSided = hullEnvD("LDCD_HULL_MARGIN", 3.0);
    static const double kMarginTwoSided = hullEnvD("LDCD_HULL_MARGIN2", 2.0);
    // Block mean-square -> per-sample amplitude the carrier may legitimately
    // reach. 0.58% per-sample violation at 2.0; 4.50% at 1.5.
    static const double kCrest        = hullEnvD("LDCD_HULL_CREST", 2.0);
    constexpr int kBlockLines   = 8;   // same-parity lines per block
    constexpr int kBlockSamples = 16;

    if (!hullOn) return false;
    if (!carrierRetractedValid) return false;
    if (ldcdRetractedSourceMode() != 3) return false;
    // Covered frames carry their own twin; measured, their fit already tracks
    // the certified carrier at r = +0.99 and has nothing to be bounded from.
    if (frameHasExactCoverage()) return false;
    // One bracket per line parity is the ordinary 3:2 geometry, so a single
    // neighbour is enough; whichever certifies a line supplies that line.
    if (!prevF && !nextF) return false;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int width     = videoParameters.activeVideoEnd - left;
    if (width <= 0 || firstLine >= lastLine) return false;

    const int fw = videoParameters.fieldWidth;

    // The carrier-antisymmetric projection of a composite line, active span.
    // Shared definition; the bracket and this frame are measured identically.
    auto projectBand = [&](const quint16 *rawLine, std::vector<double> &out) {
        out.assign(width, 0.0);
        for (int xi = 0; xi < width; ++xi) {
            const int h = left + xi;
            const double hm2 = (h - 2 >= 0) ? static_cast<double>(rawLine[h - 2])
                                            : static_cast<double>(rawLine[h]);
            out[xi] = lddecode::carrierBandProjection(
                static_cast<double>(rawLine[h]), hm2);
        }
    };
    const FrameBuffer *nb[2] = { prevF, nextF };
    bool changed = false;

    // Per-block-row scratch, allocated once. Every projection below is taken
    // exactly once per line and then read by every block column.
    const size_t rowStride = static_cast<size_t>(width);
    std::vector<double> selfBand(rowStride * kBlockLines);
    std::vector<double> nbBand[2] = {
        std::vector<double>(rowStride * kBlockLines),
        std::vector<double>(rowStride * kBlockLines) };
    std::vector<double> scratchA(width);
    const float *nbExact[2][kBlockLines] = {};

    for (int base = firstLine; base < lastLine; ++base) {
        // Blocks run over SAME-PARITY lines: which neighbour certifies a line
        // is a parity fact, so a block that mixed parities would mix brackets.
        if (base + 2 * (kBlockLines - 1) >= lastLine) continue;
        if ((base - firstLine) % (2 * kBlockLines) >= 2) continue;

        for (int li = 0; li < kBlockLines; ++li) {
            const int line = base + 2 * li;
            double *sb = selfBand.data() + rowStride * li;
            projectBand(rawbuffer.data() + static_cast<size_t>(line) * fw,
                        scratchA);
            std::copy(scratchA.begin(), scratchA.end(), sb);

            for (int s = 0; s < 2; ++s) {
                nbExact[s][li] = nullptr;
                const FrameBuffer *n = nb[s];
                if (!n || !n->frameHasExactCoverage()) continue;
                const float *ex = n->exactCarrierRow(line);
                if (!ex) continue;
                nbExact[s][li] = ex;
                projectBand(n->rawbuffer.data() + static_cast<size_t>(line) * fw,
                            scratchA);
                std::copy(scratchA.begin(), scratchA.end(),
                          nbBand[s].data() + rowStride * li);
            }
        }

        for (int x0 = 0; x0 + kBlockSamples <= width; x0 += kBlockSamples) {
            lddecode::CertifiedBracketBlock stat[2];
            double selfEnergy = 0.0;
            long   selfN = 0;

            for (int li = 0; li < kBlockLines; ++li) {
                const double *sb = selfBand.data() + rowStride * li;
                for (int xi = x0; xi < x0 + kBlockSamples; ++xi) {
                    selfEnergy += sb[xi] * sb[xi];
                    ++selfN;
                }
                for (int s = 0; s < 2; ++s) {
                    const float *ex = nbExact[s][li];
                    if (!ex) continue;
                    const double *band = nbBand[s].data() + rowStride * li;
                    for (int xi = x0; xi < x0 + kBlockSamples; ++xi) {
                        const double c = static_cast<double>(ex[left + xi]);
                        if (!std::isfinite(c)) continue;   // not certified here
                        stat[s].bracketCarrierEnergy += c * c;
                        stat[s].bracketBandEnergy += band[xi] * band[xi];
                        ++stat[s].samples;
                    }
                }
            }

            if (selfN <= 0) continue;
            const double localBand = selfEnergy / static_cast<double>(selfN);

            // The loosest interval any bracket admits: a bracket that reports
            // more carrier forbids less, and forbidding is the only thing this
            // is allowed to do.
            lddecode::FeasibleInterval iv;   // unbounded
            int bracketsUsed = 0;
            for (int s = 0; s < 2; ++s) {
                if (stat[s].samples <= 0) continue;
                ++bracketsUsed;
            }
            if (bracketsUsed == 0) continue;

            const double margin = (bracketsUsed >= 2) ? kMarginTwoSided
                                                      : kMarginOneSided;
            bool bounded = false;
            for (int s = 0; s < 2; ++s) {
                if (stat[s].samples <= 0) continue;
                lddecode::CertifiedBracketBlock b = stat[s];
                b.bracketCarrierEnergy /= static_cast<double>(b.samples);
                b.bracketBandEnergy    /= static_cast<double>(b.samples);
                b.localBandEnergy       = localBand;
                const lddecode::FeasibleInterval one =
                    lddecode::carrierFeasibleFromCertifiedBracket(
                        b, margin, kCrest);
                if (one.hi > 1e299) continue;      // this bracket forbids nothing
                if (!bounded) { iv = one; bounded = true; }
                else {
                    // Loosest of the available brackets, never the tightest:
                    // each is an independent claim and only the weakest is
                    // safe to impose.
                    if (one.hi > iv.hi) { iv.hi = one.hi; iv.lo = one.lo; }
                }
            }
            if (!bounded) continue;

            // carrierRetracted_flat holds raw - carrier (the retracted LUMA);
            // the carrier is what raw minus it publishes. Bound the carrier
            // and store the luma the bounded carrier implies, so whatever the
            // hull refuses returns to Y rather than vanishing.
            for (int li = 0; li < kBlockLines; ++li) {
                const int line = base + 2 * li;
                const quint16 *rawLine = rawbuffer.data()
                    + static_cast<size_t>(line) * fw;
                float *retractedRow = carrierRetracted_flat.data()
                    + static_cast<size_t>(line) * demodWidth;
                for (int xi = x0; xi < x0 + kBlockSamples; ++xi) {
                    const double lum = static_cast<double>(retractedRow[xi]);
                    if (!std::isfinite(lum)) continue;
                    const double raw = static_cast<double>(rawLine[left + xi]);
                    const double carrier = raw - lum;
                    const double bounded = iv.clamp(carrier);
                    if (bounded == carrier) continue;
                    retractedRow[xi] = static_cast<float>(raw - bounded);
                    changed = true;
                }
            }
        }
    }

    if (!changed) return false;
    // This optional diagnostic bound remains private.  An uncovered bounded
    // estimate is still an estimate and cannot be published as carrier.
    return true;
}

// Sync-tone actuator. See comb.h. Runs between fit construction and its
// first consumer inside buildCertifiedCarrierStage.
void Comb::FrameBuffer::applyToneToFit(const FrameBuffer *prevF)
{
    // DEFAULT ON (promoted 2026-08-01 with the anchoring family, user
    // directive): the research verdict is that the certified carrier has
    // an exact repeating cycle, and the anticipated phase lands almost
    // exactly on the measured phase at the next cover -- so the curve is
    // usable CAUSALLY, from the previous covered frame alone.  This is
    // the anticipation actuator: prev cover's certified regional phase,
    // advanced by the tracked global rate, corrects the uncovered fit's
    // trend at load time (before Pass 2 and before split2D consumes the
    // ladder).  A tone corrects trend, never performs surgery (+-25 deg
    // hull, confidence-scaled, bilinear-smooth).  Escape LDCD_CERT_TONE=0
    // for A/B only.
    static const bool toneOn = []{
        const char *e = std::getenv("LDCD_CERT_TONE");
        return !(e && std::atoi(e) == 0);
    }();
    if (!toneOn || certifiedOneDLevel() == 0) return;
    if (frameHasExactCoverage()) return;                // anchors stand
    if (!prevF || !prevF->frameHasExactCoverage()) return;
    const QVector<float> &pay =
        !syncIncFirst.isEmpty() ? syncIncFirst : syncIncSecond;
    if (pay.size() < 4) return;
    const double gOmega = pay[0];
    const double gConf  = pay[1];
    const double dtF    = pay[2];
    if (gConf < 0.2 || dtF <= 0.0 || dtF > 8.0) return;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int width     = videoParameters.activeVideoEnd - left;
    if (width <= 0) return;
    const int nx = (width + 127) / 128;
    const int ny = (lastLine + 31) / 32;
    const int nRegPay = (pay.size() - 4) / 2;
    static const int kcB[4] = { 1, 0, -1, 0 };
    static const int ksB[4] = { 0, 1, 0, -1 };

    // Anchor: prevF's certified regional phase, RAW convention (the
    // convention proven identical to the assembler's to 0.01 deg). Def
    // lines only, field-row signing (line/2)&1.
    std::vector<double> aI((size_t)nx * ny, 0.0), aQ((size_t)nx * ny, 0.0);
    std::vector<long>   aN((size_t)nx * ny, 0);
    int defPar = -1;
    for (int line = firstLine; line < lastLine; ++line) {
        if (!prevF->certifiedDefLine(line)) continue;
        if (defPar < 0) defPar = line & 1;
        const float *ex = prevF->exactCarrierRow(line);
        if (!ex) continue;
        const double rs = ((line / 2) & 1) ? -1.0 : 1.0;
        const size_t rb = (size_t)(line / 32) * nx;
        for (int h = left; h < left + width; ++h) {
            if (!std::isfinite(ex[h])) continue;
            const int ph = h & 3;
            const size_t r = rb + (h - left) / 128;
            aI[r] += rs * (double)ex[h] * kcB[ph];
            aQ[r] += rs * (double)ex[h] * ksB[ph];
            aN[r]++;
        }
    }
    if (defPar < 0) return;

    // Local fit phase, pooled over the SAME parity lines with the SAME
    // signing so the delta lives in one convention and any lattice
    // constant cancels.
    std::vector<double> fI((size_t)nx * ny, 0.0), fQ((size_t)nx * ny, 0.0);
    std::vector<long>   fN((size_t)nx * ny, 0);
    for (int line = firstLine; line < lastLine; ++line) {
        if ((line & 1) != defPar) continue;
        const float *fit = carrierFit_flat.data()
                           + (size_t)line * demodWidth;
        const double rs = ((line / 2) & 1) ? -1.0 : 1.0;
        const size_t rb = (size_t)(line / 32) * nx;
        for (int xi = 0; xi < width; ++xi) {
            const float v = fit[xi];
            if (!std::isfinite(v)) continue;
            const int ph = (left + xi) & 3;
            const size_t r = rb + xi / 128;
            fI[r] += rs * (double)v * kcB[ph];
            fQ[r] += rs * (double)v * ksB[ph];
            fN[r]++;
        }
    }

    // Per-region delta: (anchor advanced by the global rate) minus fit,
    // confidence-scaled, hulled at +-25 deg (a tone corrects trend, never
    // performs surgery).
    const double floorRaw = 2.0 * irescale;
    constexpr double kMaxDelta = 25.0 * M_PI / 180.0;
    std::vector<double> delta((size_t)nx * ny, 0.0);
    for (int r = 0; r < nx * ny; ++r) {
        if (aN[r] < 64 || fN[r] < 64) continue;
        if (std::hypot(aI[r], aQ[r]) / aN[r] < floorRaw) continue;
        if (std::hypot(fI[r], fQ[r]) / fN[r] < floorRaw) continue;
        double conf = gConf;
        if (r < nRegPay) conf *= std::clamp((double)pay[4 + r * 2 + 1],
                                            0.0, 1.0);
        if (conf <= 0.05) continue;
        const double aPh = std::atan2(aQ[r], aI[r]) + gOmega * dtF;
        const double d0 = std::atan2(
            std::sin(aPh) * fI[r] - std::cos(aPh) * fQ[r],
            std::cos(aPh) * fI[r] + std::sin(aPh) * fQ[r]);
        delta[r] = std::clamp(conf * d0, -kMaxDelta, kMaxDelta);
    }

    // Bilinear delta field over region centers, then the quadrature
    // rotation per line. Smooth by construction: no seams, no per-sample
    // policy.
    std::vector<double> rot(width), src(width);
    for (int line = firstLine; line < lastLine; ++line) {
        float *fit = carrierFit_flat.data() + (size_t)line * demodWidth;
        const double ry = std::clamp(
            (line - 16.0) / 32.0, 0.0, (double)(ny - 1) - 1e-6);
        const int r0 = (int)ry;
        const double wy = ry - r0;
        for (int xi = 0; xi < width; ++xi) {
            const double rx = std::clamp(
                (xi - 64.0) / 128.0, 0.0, (double)(nx - 1) - 1e-6);
            const int c0 = (int)rx;
            const double wx = rx - c0;
            const double d =
                (1 - wy) * ((1 - wx) * delta[(size_t)r0 * nx + c0] +
                            wx * delta[(size_t)r0 * nx + c0 + 1 < (size_t)nx * ny ? (size_t)r0 * nx + std::min(c0 + 1, nx - 1) : (size_t)r0 * nx + c0]) +
                wy * ((1 - wx) * delta[(size_t)std::min(r0 + 1, ny - 1) * nx + c0] +
                      wx * delta[(size_t)std::min(r0 + 1, ny - 1) * nx + std::min(c0 + 1, nx - 1)]);
            src[xi] = fit[xi];
            rot[xi] = d;
        }
        for (int xi = 0; xi < width; ++xi) {
            const double sm = xi > 0 ? src[xi - 1] : src[xi];
            const double sp = xi + 1 < width ? src[xi + 1] : src[xi];
            const double q = 0.5 * (sm - sp);
            fit[xi] = (float)(std::cos(rot[xi]) * src[xi] +
                              std::sin(rot[xi]) * q);
        }
    }
}

// Retained record from the removed LDCD_PROBE_CERTREG census
//
// Measured on BO-beta, three covered frames (407 / 413 / 417), after the
// identifiability floor and the interior-minimum test were both in place:
//
//   aim         s=0 59.7% | |s|=1 26.4% | |s|=2 13.9%, stable to 0.5%
//               across frames -- monotonically decreasing, which is the
//               shape a real slope population has. Before the two tests it
//               read 30 / 37 / 33 with the largest bin PINNED at the search
//               limit, i.e. a saturating search rather than a picture.
//   explanation bracket disagreement over the columns that adopt an aim:
//               1.61 -> 0.93 IRE, a 42% reduction, stable to 0.01 IRE. The
//               residual sits just under the 1.0 IRE identifiability floor,
//               which is where it belongs: the aim removes the part that
//               was explainable and leaves the noise.
//   ladder      covered frame: 50.1% certified carrier as itself, 49.9% the
//               fact-only comp rung, 0.0% fallback. Uncovered frame: 100%
//               locked 1D observation. No fit anywhere on either.

void Comb::FrameBuffer::buildCertifiedCarrierStage(const FrameBuffer *prevF)
{
    carrierRetractedValid = false;
    anchoredCarrierProvenance = AnchoredCarrierProvenance::None;
    anchored1DSource_flat.clear();

    // GATED ON THE PRESENCE OF CERTIFIED (author, 2026-08-08: "the proper
    // home, the only gating should be on the presence of certified").
    //
    // The --luma-witness gate that stood here was historical, not
    // mechanical: this stage reads raw, the exact channel, the grammar, the
    // unconditionally-built lockedLumaSmooth service and the analysis
    // record. It never read lockedLumaSharp, which is the witness's actual
    // product and is consumed only by produceY and the Frame B coarse
    // selector. The certified carrier is a conservation fact; it had no
    // business being a child of a produceY election escape.
    //
    // Per-frame coverage is deliberately NOT tested here. A frame without
    // coverage still reaches the ladder, but publishes no anchored plane: it
    // serves locked 1D through combSource1D_line and is combed by whatever
    // comb the user selected. No retracted-luma storage participates in that
    // decision.
    if (!configuration.phaseCompensation ||
        certifiedOneDLevel() < 1)
        return;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;
    const auto &T       = configuration.tunables;

    if (width <= 0 || firstLine >= lastLine)
        return;
    if (demodWidth != width || demodLines < lastLine)
        return;
    if (!basisLockedInit)
        return;

    const size_t need = static_cast<size_t>(demodLines) * static_cast<size_t>(demodWidth);
    if (carrierFit_flat.size() < need)
        carrierFit_flat.assign(need, 0.0f);
    // Zeroed EVERY frame, not merely sized: a line that takes a no-model
    // path writes no marker, so a stale 1 from the previous frame would
    // certify a model that was never solved.
    carrierFitLineValid.assign(static_cast<size_t>(lastLine), 0);
    if (configuration.lumaWitness) {
        if (carrierRetracted_flat.size() < need)
            carrierRetracted_flat.assign(need, 0.0f);
    } else {
        carrierRetracted_flat.clear();
    }
    if (flatFloor_flat.size() < need)
        flatFloor_flat.assign(need, 0.0f);
    if (combedCarrier_flat.size() < need)
        combedCarrier_flat.assign(need, 0.0f);
    if (carrierCorroboration_flat.size() < need)
        carrierCorroboration_flat.assign(need, 0.0f);
    if (carrierEligibility_flat.size() < need)
        carrierEligibility_flat.assign(need, 0.0f);
    // Certified registration (see comb.h): reset to "no fact" every frame,
    // so a line that declines to measure can never serve a stale aim.
    if (certRegistration_flat.size() < need)
        certRegistration_flat.assign(need, kCertRegNone);
    else
        std::fill(certRegistration_flat.begin(),
                  certRegistration_flat.end(), kCertRegNone);
    if (coarseYEvidence_flat.size() < need)
        coarseYEvidence_flat.assign(need, lddecode::FourViewPixelEvidence{});
    if (carrierImpurity_flat.size() < need)
        carrierImpurity_flat.assign(need, 0.0f);
    if (configuration.lumaWitness && carrierAnalysis_flat.size() < need)
        return; // shared analysis must already have been produced

    // The carrier-retracted model and its promotion passes are the expensive
    // --luma-witness product.  Certified carrier construction was moved out
    // from under that option, but the witness model accidentally came with it
    // and made every locked decode pay the witness cost.  Keep the fact-only
    // certified ladder below unconditional; build this estimate/view only
    // when the user explicitly asks for the witness.
    if (configuration.lumaWitness) {
    if ((int)scratch_preI.size()        < width) scratch_preI.resize(width, 0.0);
    if ((int)scratch_preQ.size()        < width) scratch_preQ.resize(width, 0.0);
    if ((int)scratch_lineWorkA.size()   < width) scratch_lineWorkA.resize(width, 0.0);
    if ((int)scratch_lineWorkB.size()   < width) scratch_lineWorkB.resize(width, 0.0);
    if ((int)scratch_lineWorkC.size()   < width) scratch_lineWorkC.resize(width, 0.0);
    if ((int)scratch_lineWorkD.size()   < width) scratch_lineWorkD.resize(width, 0.0);
    if ((int)scratch_lumaBaseY4.size()  < width) scratch_lumaBaseY4.resize(width, 0.0);
    if ((int)scratch_lumaSmooth.size()  < width) scratch_lumaSmooth.resize(width, 0.0);
    if ((int)scratch_lateralLine.size() < width) scratch_lateralLine.resize(width, 0.0);

    double *rawWhole   = scratch_preI.data();
    double *coarseY    = scratch_preQ.data();
    double *carrierFit = scratch_lineWorkA.data();
    double *basisI     = scratch_lineWorkB.data();
    double *flattened  = scratch_lineWorkC.data();
    double *basisQ     = scratch_lineWorkD.data();
    double *refinedY   = scratch_lumaSmooth.data();
    double *slideMean4 = scratch_lateralLine.data();

    std::vector<double> winFloor;
    std::vector<double> winI;
    std::vector<double> winQ;
    std::vector<double> winErrorIRE;
    std::vector<double> winLatticeIRE;
    std::vector<double> winYSpanIRE;
    std::vector<double> winScore;
    std::vector<std::uint8_t> winFitValid;
    std::vector<std::uint8_t> boundaryMark;
    std::vector<double> winEnvScratch;
    std::vector<double> envI, envQ, envTmp;
    std::vector<double> partWeight(static_cast<size_t>(width), 1.0);

    auto median3 = [](double a, double b, double c) -> double {
        if (a > b) std::swap(a, b);
        if (b > c) std::swap(b, c);
        if (a > b) std::swap(a, b);
        return b;
    };

    // Flat-region fast path: when all four views agree to within ~0.05 IRE,
    // the cost matrix is dominated by arithmetic noise and the medoid is
    // within the same tolerance of the plain mean. Skip 16 abs + compares
    // (twice per pixel across the two call sites) and return the mean. The
    // anchor only matters as a tiebreak when views disagree, so it does not
    // need to participate in the flat case.
    const double medoidFlatTol = 0.05 * irescale;
    auto medoid4Anchored = [medoidFlatTol](double a, double b, double c, double d,
                                            double anchor) -> double {
        const double lo = std::min(std::min(a, b), std::min(c, d));
        const double hi = std::max(std::max(a, b), std::max(c, d));
        if (hi - lo < medoidFlatTol)
            return 0.25 * (a + b + c + d);

        double v[4] = {a, b, c, d};
        double best = v[0];
        double bestCost = 1e300;
        double bestAnchorDist = 1e300;

        for (int i = 0; i < 4; ++i) {
            double cost = 0.0;
            for (int j = 0; j < 4; ++j)
                cost += std::fabs(v[i] - v[j]);

            const double anchorDist = std::fabs(v[i] - anchor);
            if (cost < bestCost - 1e-9 ||
                (std::fabs(cost - bestCost) <= 1e-9 &&
                 anchorDist < bestAnchorDist)) {
                best = v[i];
                bestCost = cost;
                bestAnchorDist = anchorDist;
            }
        }

        return best;
    };

    auto smoothStep01 = [](double t) {
        t = std::clamp(t, 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    };

    if (!carrierRetractionModelValid) {
        // ---------------------------------------------------------------
        // Analysis/model promotion: per-line carrier withdrawal.
        //
        // This version does not start from the ±2 complement estimate.  It first
        // derives a per-sample Y prior from the legal 4-sample luma-floor views,
        // then projects raw - refinedY into locked IQ.  The four carrier windows
        // touching the current sample are treated as attribution evidence rather
        // than being blended as an immediate heuristic result.
        //
        // In other words:
        //
        //     raw = Y + C
        //
        // is resolved by asking which legal Y floor leaves the most coherent C,
        // rather than by subtracting a complement-estimated C and trusting whatever
        // remains as Y.
        // ---------------------------------------------------------------

    for (int line = firstLine; line < lastLine; ++line) {
        float *fitRow       = carrierFit_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        float *floorRow     = flatFloor_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        auto *evidenceRow   = coarseYEvidence_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        auto *analysisRow   = carrierAnalysis_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        float *eligibilityRow = carrierEligibility_flat.data()
                              + static_cast<size_t>(line) * demodWidth;

        for (int xi = 0; xi < width; ++xi)
            analysisRow[xi].parallax = lddecode::CarrierParallaxDiagnostics{};

        const CombCarrierGrammar *grammar = carrierGrammarLine(line);
        const bool grammarLocked = grammar && grammar->grammarLocked;

        const quint16 *rawLine =
            rawbuffer.data() + static_cast<size_t>(line) * videoParameters.fieldWidth;

        const double *baseY4Src;
        if (lockedLumaCacheValid && demodWidth == width &&
            !lockedLumaBaseY4_flat.empty())
        {
            baseY4Src = lockedLumaBaseY4_line(line);
        } else {
            buildCompositeLumaDecompositionLine(rawLine, left, width,
                                                scratch_lumaBaseY4.data(),
                                                nullptr, nullptr);
            baseY4Src = scratch_lumaBaseY4.data();
        }

        for (int xi = 0; xi < width; ++xi) {
            rawWhole[xi] = static_cast<double>(rawLine[left + xi]);
            coarseY[xi]  = baseY4Src[xi];
            refinedY[xi] = coarseY[xi];
        }

        if (!grammarLocked) {
            for (int xi = 0; xi < width; ++xi) {
                fitRow[xi]       = 0.0f;
                floorRow[xi]     = static_cast<float>(coarseY[xi]);
                evidenceRow[xi].viewCount = 0;
                eligibilityRow[xi] = 0.0f;
            }
            continue;
        }

        // Graded schedule participation, consumed everywhere this stage used
        // to consume the thresholded enum.  carrierTrust() is the table-owned
        // mapping from the conformance MEASUREMENT (+ support fraction) to a
        // carrier-trust weight; participation doubles it and saturates so the
        // legacy keep-set (Legal, Unresolved, quiet — trust >= 0.5) keeps its
        // old FULL weight and only the illegal side grades: one axis vote
        // ~0.67, two ~0.33, decisive proof 0.  The old per-sample enum test
        // was a 1-bit sampler on a smoothly drifting vertical correlation —
        // half of all Illegal verdicts rested on a single axis vote, and the
        // bit flipping at line pitch was the dominant vertical raggedness of
        // the retracted view (Borg-cube study, 2026-07-20).
        // NOTE (2026-08-07): a parallax vote CANNOT be added here. The
        // four-view parallax record is reset at the top of this line's body
        // and is not populated until ~500 lines below, from these very window
        // fits -- so at this point analysisRow[].parallax.valid is false
        // everywhere and any test against it is dead code (verified: a build
        // gating on it was bit-identical). Correcting the solve with parallax
        // requires a SECOND solve after the evidence exists, not a weight
        // change in the first one. See the Bug-Hunt entry for the confound
        // this exposes in the shadow measurement.
        for (int xi = 0; xi < width; ++xi) {
            const auto &a = analysisRow[xi];
            partWeight[xi] = std::min(1.0, 2.0 * lddecode::carrierTrust(
                static_cast<double>(a.carrierConformance),
                static_cast<double>(a.conformanceSupportFraction)));
            eligibilityRow[xi] = static_cast<float>(partWeight[xi]);
        }

        const double bcos = grammar->burstCos;
        const double bsin = grammar->burstSin;
        const double maxCarrierSamples =
            maxCarrierAmpIREFromScale(grammar->carrierScale) * irescale;

        // basisI/Q at position h depend only on the carrier sample class
        // given the line's burst phasor and locked basis.  The class is the
        // GRAMMAR sample class, carrierSampleClass(line, h) = (h+samplePhase0)
        // & 3 — not a raw h & 3.  These are identical while samplePhase0 == 0
        // (its current value everywhere), but every other locked-path site
        // (Pass-3 demod, splitIQlocked, filterIQLocked, the HF-election
        // phaseConf) indexes by the grammar class; hardcoding the raw index
        // here was the lone site that would silently diverge the instant the
        // schedule set a nonzero sample phase.  Consult the schedule, do not
        // assume it.  Precompute the four phase values and fill by lookup.
        double basisI4[4];
        double basisQ4[4];
        for (int p = 0; p < 4; ++p) {
            basisI4[p] = remodLockedToShiftedComposite(
                1.0, 0.0, p, bcos, bsin,
                spLUT_locked, cpLUT_locked);
            basisQ4[p] = remodLockedToShiftedComposite(
                0.0, 1.0, p, bcos, bsin,
                spLUT_locked, cpLUT_locked);
        }
        for (int xi = 0; xi < width; ++xi) {
            const int idx = carrierSampleClass(line, left + xi);
            basisI[xi] = basisI4[idx];
            basisQ[xi] = basisQ4[idx];
        }

        if (width >= 4) {
            const int meanCount = width - 3;
            if ((int)winFloor.size() < meanCount) {
                winFloor.resize(meanCount, 0.0);
                winI.resize(meanCount, 0.0);
                winQ.resize(meanCount, 0.0);
                winErrorIRE.resize(meanCount, 0.0);
                winLatticeIRE.resize(meanCount, 0.0);
                winYSpanIRE.resize(meanCount, 0.0);
                winScore.resize(meanCount, 0.0);
                winFitValid.resize(meanCount, std::uint8_t{0});
            }
            if ((int)boundaryMark.size() < width)
                boundaryMark.resize(width, 0);

            // winFloor IS the shared aperture pool: 0.25*(raw[s..s+3]) is
            // exactly what buildApertureMeans() published (both are exact
            // integer sums, so the copy is byte-identical to the old private
            // rebuild). One scan, many readers.
            {
                const double *apRow = lockedApertureMean_line(line);
                if (apRow) {
                    std::copy(apRow, apRow + meanCount, winFloor.begin());
                } else {
                    for (int s = 0; s < meanCount; ++s)
                        winFloor[s] =
                            0.25 * (rawWhole[s + 0] + rawWhole[s + 1] +
                                    rawWhole[s + 2] + rawWhole[s + 3]);
                }
            }

            // Luma prior: the integer-centred moving coarse, not a medoid of the
            // four covering means.  The medoid was robust but is still a
            // boxcar statistic with half the smear baked in; the prior is
            // the rolling legal mean, and the lurch preconditioner restores
            // the step placement the boxcar blurs.
            // Lurch preconditioner: resolve the prior's step placement before
            // the carrier fit consumes it, so step energy stays out of
            // raw - refinedY and never enters the carrier band. winFloor IS
            // the shared aperture pool (copied above), so this is the same
            // quantity the witness platform solves -- one construction, two
            // readers.
            if (lurchSolveEnabled()) {
                solveLurchYCurve(line, winFloor.data(), meanCount, width,
                                 refinedY);
            } else {
                for (int xi = 0; xi < width; ++xi) {
                    const int s0 = std::clamp(xi - 2, 0, meanCount - 1);
                    const int s1 = std::clamp(xi - 1, 0, meanCount - 1);
                    refinedY[xi] = 0.5 * (winFloor[s0] + winFloor[s1]);
                }
                // Canonical runs (built once in split1D on the same pool),
                // edges vertically corroborated; apply-only here.
                const std::vector<LurchStepRun> corrRuns =
                    corroborateLurchEdges(line);
                applyLurchSteps(corrRuns, winFloor.data(),
                                meanCount, width, 1.0, refinedY, nullptr);
            }

            for (int s = 0; s < meanCount; ++s) {
                double sII = 0.0, sIQ = 0.0, sQQ = 0.0;
                double sIY = 0.0, sQY = 0.0;
                double sampleWeight = 0.0;

                double refinedMean = 0.0;
                double minRefined = 1e300;
                double maxRefined = -1e300;

                for (int k = 0; k < 4; ++k) {
                    const int xi = s + k;
                    refinedMean += refinedY[xi];
                    minRefined = std::min(minRefined, refinedY[xi]);
                    maxRefined = std::max(maxRefined, refinedY[xi]);

                    // Registration doubt DOWNWEIGHTS, it does not remove.
                    // The old hard removal changed the normal matrix's sample
                    // population whenever the per-pixel verdict bit flipped,
                    // so adjacent lines solved structurally different systems
                    // and their fits decorrelated (which Pass 2's license then
                    // read as noise).  A weight leaves both sides of the solve
                    // in the same geometry and lets doubt fade the sample
                    // smoothly; w == 0 (decisive proof) still removes exactly.
                    const double w = partWeight[xi];
                    if (w <= 0.0)
                        continue;

                    const double bI = basisI[xi];
                    const double bQ = basisQ[xi];
                    const double residual = rawWhole[xi] - refinedY[xi];

                    sII += w * bI * bI;
                    sIQ += w * bI * bQ;
                    sQQ += w * bQ * bQ;
                    sIY += w * bI * residual;
                    sQY += w * bQ * residual;
                    sampleWeight += w;
                }

                refinedMean *= 0.25;

                double fitI = 0.0;
                double fitQ = 0.0;
                const double det = sII * sQQ - sIQ * sIQ;
                // Effective-sample floor: a 2-parameter solve needs more than
                // two samples' worth of participating evidence (the old rule
                // was >= 3 of 4 hard samples; the graded analogue crosses the
                // same boundary smoothly instead of at one sample's bit).
                const bool fitValid =
                    sampleWeight >= 2.5 && std::fabs(det) > 1e-9;
                if (fitValid) {
                    const double inv = 1.0 / det;
                    fitI = ( sQQ * sIY - sIQ * sQY) * inv;
                    fitQ = (-sIQ * sIY + sII * sQY) * inv;
                }

                winI[s] = fitI;
                winQ[s] = fitQ;
                winFitValid[s] = fitValid ? std::uint8_t{1} : std::uint8_t{0};

                double errSq = 0.0;
                double basis01 = 0.0; // +-+-
                double basis02 = 0.0; // ++--
                double basis03 = 0.0; // +--+
                double fitAbs = 0.0;
                double gradedWeight = 0.0;

                for (int k = 0; k < 4; ++k) {
                    const int xi = s + k;
                    const double w = partWeight[xi];
                    if (w <= 0.0)
                        continue;

                    const double bI = basisI[xi];
                    const double bQ = basisQ[xi];

                    const double fit = fitI * bI + fitQ * bQ;
                    const double residual = rawWhole[xi] - refinedY[xi];
                    const double e = residual - fit;

                    errSq += w * e * e;
                    fitAbs += w * std::fabs(fit);
                    gradedWeight += w;

                    const double we = w * e;
                    if (k == 0) {
                        basis01 += we;
                        basis02 += we;
                        basis03 += we;
                    } else if (k == 1) {
                        basis01 -= we;
                        basis02 += we;
                        basis03 -= we;
                    } else if (k == 2) {
                        basis01 += we;
                        basis02 -= we;
                        basis03 -= we;
                    } else {
                        basis01 -= we;
                        basis02 -= we;
                        basis03 += we;
                    }
                }

                const double gradeInv = gradedWeight > 0.0
                    ? 1.0 / gradedWeight
                    : 0.0;
                const double errIRE =
                    std::sqrt(gradeInv * errSq) * invIreScale;
                const double latticeIRE =
                    gradeInv * std::max({std::fabs(basis01),
                                         std::fabs(basis02),
                                         std::fabs(basis03)}) * invIreScale;
                const double floorDriftIRE =
                    std::fabs(winFloor[s] - refinedMean) * invIreScale;
                const double ySpanIRE =
                    (maxRefined - minRefined) * invIreScale;
                const double ampIRE =
                    gradeInv * fitAbs * invIreScale;

                winErrorIRE[s] = errIRE;
                winLatticeIRE[s] = latticeIRE;
                winYSpanIRE[s] = ySpanIRE;

                // The score prefers:
                //   - residual that remodulates cleanly through locked IQ,
                //   - little remaining +-+- / ++-- / +--+ lattice residue,
                //   - a legal floor that agrees with the refined-Y prior,
                //   - low refined-Y span inside the 4-sample cell.
                //
                // The small amplitude term prevents the all-Y interpretation
                // from winning merely because it is conservative.
                winScore[s] =
                    errIRE +
                    0.75 * latticeIRE +
                    0.25 * floorDriftIRE +
                    0.15 * ySpanIRE -
                    0.10 * std::min(ampIRE, 24.0);
            }

            // NOTE: the encoder bandwidth law is NOT applied to winI/winQ here.
            // It is imposed ONCE, at the model boundary below, after the
            // four-view attribution and residual clamp have had their say.
            // Filtering here as well would be a second forward application of
            // the same FIR, and a forward FIR is not a projection: P(P(x)) !=
            // P(x) (measured 0.25 relative error, and 4.5 dB of EXTRA loss at
            // 1.3 MHz for a third application).  Legal chroma near the top of
            // the encoder's own passband would be progressively attenuated and
            // dumped into Y — the opposite of the intent.

            // Residual-side chroma-boundary discovery (the lurch dual).
            // The per-window LS fits are the carrier profile along the
            // line, and they were solved against the lurch-sharpened Y
            // prior, so luma steps are already out of this profile: what
            // steps here is chroma.  A boundary is a step between two
            // internally COHERENT runs — d(p) compares the disjoint
            // adjacent windows ending at p and starting at p+1, so a mark
            // at p places the transition between samples p and p+1.  The
            // side-coherence requirement leaves broadband texture
            // (incoherent profile) unmarked: there the four-view spread is
            // noise, not geometry, and filtering would only cost the
            // attribution its robustness.
            std::fill(boundaryMark.begin(),
                      boundaryMark.begin() + width, std::uint8_t{0});
            {
                const double stepFloor = 3.0 * irescale;
                for (int p = 7; p + 5 < meanCount + 3; ++p) {
                    const int sl = p - 3;       // window [p-3 .. p]
                    const int sr = p + 1;       // window [p+1 .. p+4]
                    const int sll = sl - 4;
                    const int srr = sr + 4;
                    if (srr >= meanCount)
                        break;
                    if (!winFitValid[sl] || !winFitValid[sr] ||
                        !winFitValid[sll] || !winFitValid[srr])
                    {
                        continue;
                    }
                    const double dI = winI[sr] - winI[sl];
                    const double dQ = winQ[sr] - winQ[sl];
                    const double stepSq = dI * dI + dQ * dQ;
                    const double magLSq =
                        winI[sl] * winI[sl] + winQ[sl] * winQ[sl];
                    const double magRSq =
                        winI[sr] * winI[sr] + winQ[sr] * winQ[sr];
                    if (stepSq < stepFloor * stepFloor ||
                        stepSq < (0.35 * 0.35) * std::max(magLSq, magRSq))
                        continue;
                    const double cI = winI[sl] - winI[sll];
                    const double cQ = winQ[sl] - winQ[sll];
                    const double dI2 = winI[srr] - winI[sr];
                    const double dQ2 = winQ[srr] - winQ[sr];
                    if (cI * cI + cQ * cQ > 0.25 * stepSq ||
                        dI2 * dI2 + dQ2 * dQ2 > 0.25 * stepSq)
                        continue;
                    boundaryMark[p] = 1;
                }
            }

            for (int xi = 0; xi < width; ++xi) {
                const int sFirst = std::max(0, xi - 3);
                const int sLast  = std::min(xi, meanCount - 1);

                lddecode::FourViewCarrierView views[4];
                int viewCount = 0;
                // The pixel's own schedule doubt no longer withholds the fit:
                // the covering windows were solved with that doubt already
                // downweighted, so the model's opinion AT this position is
                // well-formed evidence.  How much of the resulting carrier may
                // ACT is the eligibility weight published above — authority is
                // graded downstream, not amputated here.  (The old gate also
                // published fitRow = 0 at every proven-illegal pixel, which
                // made those samples unobservable to Pass 2's license and
                // partner correlations.)

                // Region-pure aperture law: a window straddling a
                // discovered chroma boundary is not evidence for any pixel.
                // A boundary at p splits samples p | p+1, so window
                // [s .. s+3] straddles iff a mark lies in [s .. s+2]; the
                // window ENDING at the boundary stays pure for the left
                // side and the one STARTING after it for the right, so
                // every pixel keeps at least one same-side view unless
                // marks are pathologically dense — then fall back to the
                // unfiltered set rather than starve attribution.
                auto windowStraddles = [&](int s) {
                    const int pHi = std::min(s + 2, width - 1);
                    for (int p = s; p <= pHi; ++p)
                        if (boundaryMark[p])
                            return true;
                    return false;
                };

                for (int pass = 0;
                     pass < 2 && viewCount == 0;
                     ++pass) {
                for (int s = sFirst; s <= sLast; ++s) {
                    if (viewCount >= 4)
                        break;
                    if (!winFitValid[s])
                        continue;
                    if (pass == 0 && windowStraddles(s))
                        continue;
                    const double carrierSample = rawWhole[xi] - winFloor[s];
                    const double fittedSample =
                        winI[s] * basisI[xi] + winQ[s] * basisQ[xi];
                    views[viewCount].apertureCenter = static_cast<double>(s) + 1.5;
                    views[viewCount].yFloor = winFloor[s];
                    views[viewCount].carrierSample = carrierSample;
                    views[viewCount].fittedSample = fittedSample;
                    views[viewCount].carrierI = winI[s];
                    views[viewCount].carrierQ = winQ[s];
                    views[viewCount].sampleFitErrorIRE =
                        std::fabs(carrierSample - fittedSample) * invIreScale;
                    views[viewCount].remodErrorIRE = winErrorIRE[s];
                    views[viewCount].latticeRiskIRE = winLatticeIRE[s];
                    views[viewCount].ySpanIRE = winYSpanIRE[s];
                    views[viewCount].score = winScore[s];
                    {
                        double mDeltaIRE = 0.0;
                        double mSupport = 0.0;
                        double mLocalX = 0.0;
                        const int s0 = s;
                        if (s0 + 4 < width) {
                            const double enterLeaveSample =
                                rawWhole[s0 + 4] - rawWhole[s0];
                            mDeltaIRE = 0.25 * enterLeaveSample * invIreScale;
                            const double deltaMagIRE = std::fabs(mDeltaIRE);
                            const double deltaGate =
                                smoothStep01((deltaMagIRE - 0.35) / 4.0);
                            const double fitGate =
                                1.0 - smoothStep01((winErrorIRE[s0] - 1.5) / 5.0);
                            const double latticeGate =
                                1.0 - smoothStep01((winLatticeIRE[s0] - 1.0) / 5.0);
                            mSupport = std::clamp(
                                deltaGate * fitGate * latticeGate, 0.0, 1.0);
                            mLocalX = 0.5 * ((double)s0 + (double)(s0 + 4))
                                    - (double)xi;
                        }
                        views[viewCount].membershipDeltaIRE    = mDeltaIRE;
                        views[viewCount].membershipSupport     = mSupport;
                        views[viewCount].membershipLocalX      = mLocalX;
                    }
                    ++viewCount;
                }
                }  // pass: region-pure first, unfiltered fallback

                evidenceRow[xi].viewCount = viewCount;

                for (int v = 0; v < viewCount; ++v) {
                    auto &dst = evidenceRow[xi].views[v];
                    const auto &src = views[v];
                    dst.apertureCenter = static_cast<float>(src.apertureCenter);
                    dst.yFloor = static_cast<float>(src.yFloor);
                    dst.sampleFitErrorIRE = static_cast<float>(src.sampleFitErrorIRE);
                    dst.remodErrorIRE = static_cast<float>(src.remodErrorIRE);
                    dst.latticeRiskIRE = static_cast<float>(src.latticeRiskIRE);
                    dst.ySpanIRE = static_cast<float>(src.ySpanIRE);
                    dst.membershipDeltaIRE    = static_cast<float>(src.membershipDeltaIRE);
                    dst.membershipSupport     = static_cast<float>(src.membershipSupport);
                }

                auto parallax = lddecode::buildFourViewCarrierAttribution(
                    views,
                    viewCount,
                    invIreScale);

                double modelI = parallax.valid ? parallax.commonI : 0.0;
                double modelQ = parallax.valid ? parallax.commonQ : 0.0;

                // MERIT MODEL (default ON 2026-08-07; LDCD_FIT_MERIT=0
                // restores the medoid for A/B). The medoid's cost is
                // dominated by proximity-to-peers, and at texture all four
                // views are contaminated DIFFERENTLY: the least-wrong view
                // is by construction the outlier, and consensus hands the
                // model to the contaminated majority (author: "consensus
                // can fail to use a good solution that's right there,
                // because it's outnumbered... Any constituent in an
                // election needs to avoid DQ'ing itself"). Here the MODEL
                // is a confidence blend of the views weighted by each
                // view's OWN merit only -- fit error, membership evidence
                // (aperture-movement = texture, evidence-triggered), score
                // -- no peer-proximity term anywhere (weight = candidate
                // confidence, never inter-candidate distance). I/Q blend
                // in the shared basis frame is envelope-scale mixing.
                // The published parallax RECORD keeps the shared-header
                // medoid semantics untouched for its evidence consumers;
                // only the composed model changes. Membership localizer
                // mirrors the header's 1.35-sigma form.
                static const bool meritModel = []{
                    const char *e = std::getenv("LDCD_FIT_MERIT");
                    return !(e && std::atoi(e) == 0);
                }();
                static const double meritTau = []{
                    const char *e = std::getenv("LDCD_MERIT_TAU");
                    const double v = e ? std::atof(e) : 2.0;
                    return v > 0.05 ? v : 2.0;
                }();
                if (meritModel && viewCount > 0) {
                    double sw = 0.0, si = 0.0, sq = 0.0;
                    for (int v = 0; v < viewCount; ++v) {
                        const auto &vw = views[v];
                        const double lx = vw.membershipLocalX;
                        const double localizer =
                            std::exp(-0.5 * lx * lx / (1.35 * 1.35));
                        const double memb =
                            std::fabs(vw.membershipDeltaIRE) *
                            std::clamp(vw.membershipSupport, 0.0, 1.0) *
                            localizer;
                        const double cost =
                            0.35 * std::max(0.0, vw.sampleFitErrorIRE) +
                            0.10 * memb +
                            0.05 * std::max(0.0, vw.score);
                        const double w = std::exp(-cost / meritTau);
                        sw += w;
                        si += w * vw.carrierI;
                        sq += w * vw.carrierQ;
                    }
                    if (sw > 1e-9) {
                        modelI = si / sw;
                        modelQ = sq / sw;
                    }
                }

                double residualCarrierLo = -1e300;
                double residualCarrierHi =  1e300;
                double residualTightenSupport = 0.0;
                double residualTightenSpreadIRE = 0.0;
                double residualTightenFitErrorIRE = 0.0;
                if (viewCount > 0) {
                    residualCarrierLo =  1e300;
                    residualCarrierHi = -1e300;
                    double sumFitError = 0.0;

                    int residualN = 0;

                    for (int v = 0; v < viewCount; ++v) {
                        const auto &view = views[v];
                        residualCarrierLo = std::min(residualCarrierLo, view.carrierSample);
                        residualCarrierHi = std::max(residualCarrierHi, view.carrierSample);
                        sumFitError += view.sampleFitErrorIRE;
                        ++residualN;
                    }

                    // Fifth residual witness: a centered/rolling legal 4fSC
                    // cancellation complement.  The four aperture views tell
                    // which legal windows including xi say what carrier remains;
                    // this rolling witness is aperture-independent in the sense
                    // that it is not one of the four ownership views being
                    // scored.  It is still a carrier-cancelling Y estimate, so
                    // raw - moving floor is mostly carrier plus whatever HF-Y
                    // the rolling window could not preserve.
                    double movingResidualSample = 0.0;
                    // The rolling witness obeys the same region-pure
                    // aperture law as the four ownership views.
                    const int movingS0 =
                        std::clamp(xi - 2, 0, meanCount - 1);
                    const int movingS1 =
                        std::clamp(xi - 1, 0, meanCount - 1);
                    if (meanCount > 0 &&
                        !windowStraddles(movingS0) &&
                        !windowStraddles(movingS1)) {
                        const double movingFloor =
                            0.5 * (winFloor[movingS0] + winFloor[movingS1]);
                        movingResidualSample = rawWhole[xi] - movingFloor;
                        residualCarrierLo = std::min(residualCarrierLo, movingResidualSample);
                        residualCarrierHi = std::max(residualCarrierHi, movingResidualSample);
                        const double movingFitError = parallax.valid
                            ? std::fabs(movingResidualSample - parallax.commonSample) * invIreScale
                            : 0.0;
                        sumFitError += movingFitError;
                        ++residualN;
                    }

                    residualTightenFitErrorIRE =
                        sumFitError / static_cast<double>(std::max(1, residualN));
                    residualTightenSpreadIRE =
                        (residualCarrierHi - residualCarrierLo) * invIreScale;

                    const double spreadGate = 1.0 - smoothStep01(
                        (residualTightenSpreadIRE - 2.0) / 8.0);
                    const double fitGate = 1.0 - smoothStep01(
                        (residualTightenFitErrorIRE - 1.5) / 6.0);
                    residualTightenSupport = std::clamp(
                        0.35 + 0.65 * spreadGate * fitGate,
                        0.0,
                        1.0);

                    lddecode::CarrierResidualConsensus consensus;
                    consensus.lo = residualCarrierLo;
                    consensus.hi = residualCarrierHi;
                    consensus.trust = residualTightenSupport;
                    consensus.valid = true;
                    parallax.residualConsensus = consensus;
                }

                {
                    auto &dst = analysisRow[xi].parallax;
                    dst.commonSample = static_cast<float>(parallax.commonSample);
                    dst.commonMagnitudeIRE = static_cast<float>(parallax.commonMagIRE);
                    dst.ySpreadIRE = static_cast<float>(parallax.ySpreadIRE);
                    dst.yCurvatureIRE = static_cast<float>(parallax.yCurvatureIRE);
                    dst.carrierSpreadIRE = static_cast<float>(parallax.carrierSpreadIRE);
                    dst.carrierCoherence = static_cast<float>(parallax.carrierCoherence);
                    dst.sampleFitErrorIRE = static_cast<float>(parallax.sampleFitErrorIRE);
                    dst.sampleCoherence = static_cast<float>(parallax.sampleCoherence);
                    dst.latticeRiskIRE = static_cast<float>(parallax.latticeRiskIRE);
                    dst.valid = parallax.valid;
                    if (parallax.residualConsensus.valid) {
                        dst.residualLo = static_cast<float>(
                            parallax.residualConsensus.lo);
                        dst.residualHi = static_cast<float>(
                            parallax.residualConsensus.hi);
                        dst.residualTrust = static_cast<float>(
                            parallax.residualConsensus.trust);
                        dst.residualValid = true;
                    }
                }

                // The model sample is the four-view model remodulated on the
                // grammar basis, then RESTRICTED: the residual-consensus
                // range and the amplitude ceiling remove impossible values
                // and nothing else. Two operations that used to sit between
                // remod and publication were deleted as UNLAWFUL under the
                // coarse-residual law (statistics and range-restriction only
                // -- never blend, never average, never output):
                //   * the commonSample anchor blend pulled the waveform
                //     toward a consensus of the coarse-residual VALUES --
                //     averaged coarse residuals becoming output;
                //   * the survivor-extent move treated the interval between
                //     discrete survivors as a legal continuum, which the
                //     analysis contract forbids in as many words ("the
                //     unobserved values between them do not become legal").
                // Measured before removal (by the since-removed off-grid
                // census): the fit's
                // span, rotation, and jitter were already the cleanest in
                // the tree -- nothing real leaned on either operation.
                double cf = modelI * basisI[xi] + modelQ * basisQ[xi];
                // RESIDUAL-CONSENSUS CLAMP RETIRED FROM THE CANDIDATE
                // (default off 2026-08-07; LDCD_FIT_CONSENSUS=1 restores).
                // Author: "Any constituent in an election needs to avoid
                // DQ'ing itself -- let the election do that. Consensus for
                // that view is bizarre, way to water it down." The clamp's
                // own comment claimed it removes impossible values; that is
                // only true where a coarse floor is clean. At texture every
                // floor is contaminated by the disputed detail itself, so
                // the bounds are a consensus of contaminated witnesses --
                // caution wearing a law's clothing -- and with trust floored
                // at 0.35 it never abstained. Caution inside a candidate
                // softens the bypass view toward the comb it exists to
                // bypass and destroys the distinctness the election needs;
                // adjudication (feasibility, cleanliness, anchor authority,
                // neighbours) is the ELECTION's duty. The consensus RECORD
                // still publishes unchanged -- evidence stays, actuation
                // goes. The amplitude clamp below stays: that one is
                // physics.
                static const bool fitConsensusOn = []{
                    const char *e = std::getenv("LDCD_FIT_CONSENSUS");
                    return e && std::atoi(e) == 1;
                }();
                if (fitConsensusOn && residualTightenSupport > 0.0)
                    cf = std::clamp(cf, residualCarrierLo, residualCarrierHi);
                cf = std::clamp(cf, -maxCarrierSamples, maxCarrierSamples);

                carrierFit[xi] = cf;
                flattened[xi] = rawWhole[xi] - cf;

                fitRow[xi] = static_cast<float>(cf);
            }

            // ---------------------------------------------------------------
            // PARALLAX OWNERSHIP POLICING of the published fit (default ON,
            // escape LDCD_FIT_POLICE=0; author-directed 2026-08-07: "we're
            // policing the old [estimate] -- it's about luma vs carrier, not
            // 'what is the carrier?'").
            //
            // The fit above stands unmodified as an ESTIMATE. What is policed
            // is its ENTITLEMENT: a good fit says what the carrier can
            // explain, never that it may remove that energy from Y (the
            // residual-Y/ownership law). The coarse-residual parallax is the
            // disambiguator it was built to be -- energy that holds still
            // across the four legal apertures behaves like carrier; energy
            // that moves with the aperture IS the image. That movement is an
            // affirmative luma witness (not a mere revocation), so where it
            // convicts, the claim's authority falls and the energy stays in
            // the picture. Picture terms: the fit stops billing paper grain
            // as paint.
            //
            // Form: w <= 1 on the claim -- the one sanctioned gain shape on a
            // carrier source (the envelope-legality hull precedent): it can
            // only return energy to Y, never manufacture carrier. The verdict
            // is gathered per sample, then smoothed to envelope scale by the
            // encoder's own 9-tap chroma kernel before application (the AM
            // law: gain gathered AND applied at envelope scale). Schedule
            // verdicts outrank: only Unresolved samples are policed -- Legal
            // keeps its claim, Illegal is already the eligibility channel's
            // duty. Parallax absence is absence: w = 1. Applied BEFORE the
            // bandwidth law below so the law still runs last on the final
            // model. This deliberately does NOT re-solve or re-compose
            // anything: a rebuilt estimate must survive the medoid vote and
            // the raw-bounds consensus clamp, where one clean view is
            // outnumbered and freed texture is re-confiscated (measured
            // 2026-08-07: the re-solve route recovered almost no retention
            // through the deployed composition). The verdict here acts on the
            // OUTPUT, where nothing downstream can re-take what it frees.
            //
            // Known cost channel (the reason for the escape and the eye
            // gate): a wrong conviction of real paint leaves unsubtracted
            // carrier in the retracted view as carrier-rate speckle. The
            // verdict therefore acts only in proportion to its own
            // decisiveness, and the retracted plane remains an election
            // candidate the cleanliness scoring can demote.
            // ---------------------------------------------------------------
            {
                static const bool fitPoliceOn = []{
                    const char *e = std::getenv("LDCD_FIT_POLICE");
                    return !(e && std::atoi(e) == 0);
                }();
                if (fitPoliceOn) {
                    double *wRaw = winEnvScratch.data();
                    if ((int)winEnvScratch.size() < width) {
                        winEnvScratch.resize(width, 1.0);
                        wRaw = winEnvScratch.data();
                    }
                    for (int xi = 0; xi < width; ++xi) {
                        double w = 1.0;
                        const auto &a = analysisRow[xi];
                        if (a.scheduleConformance ==
                                lddecode::CarrierScheduleConformance
                                    ::Unresolved &&
                            a.parallax.valid) {
                            const auto &p = a.parallax;
                            // Same normalisation as the shipped
                            // parallaxProtect consumer (Pass 1.75): one form
                            // of this reading exists in the tree.
                            const double den = std::max(
                                3.0,
                                0.35 * (double)p.commonMagnitudeIRE + 1.0);
                            const double spreadT = std::clamp(
                                (double)p.carrierSpreadIRE / den, 0.0, 1.0);
                            const double latticeT = std::clamp(
                                (double)p.latticeRiskIRE / den, 0.0, 1.0);
                            const double protect = std::max(spreadT,
                                std::max(latticeT,
                                    1.0 - (double)p.carrierCoherence));
                            w = 1.0 - protect;
                        }
                        wRaw[xi] = w;
                    }
                    for (int xi = 0; xi < width; ++xi) {
                        double acc = 0.0, norm = 0.0;
                        for (int t = 0; t < lddecode::kChromaEnvelopeTaps;
                             ++t) {
                            const int j = std::clamp(
                                xi + t - lddecode::kChromaEnvelopeTaps / 2,
                                0, width - 1);
                            const double kw =
                                lddecode::kChromaEnvelopeFilter[t];
                            acc += kw * wRaw[j];
                            norm += kw;
                        }
                        const double wSm = std::clamp(
                            norm > 1e-12 ? acc / norm : 1.0, 0.0, 1.0);
                        const double cf = carrierFit[xi] * wSm;
                        carrierFit[xi] = cf;
                        flattened[xi] = rawWhole[xi] - cf;
                        fitRow[xi] = static_cast<float>(cf);
                    }
                }
            }

            // ---------------------------------------------------------------
            // ENCODER BANDWIDTH LAW, enforced at publication.
            //
            // Projecting winI/winQ above is not sufficient on its own:
            // finalizeCarrierSample() then blends the sample toward
            // parallax.commonSample and clamps it into the residual-consensus
            // bounds, and BOTH of those are built from raw residuals
            // (rawWhole - winFloor), which are full-band by construction.
            // Those steps are right on their own terms -- they pull a bad
            // window fit back toward what every legal Y floor says at this
            // sample -- but they re-admit exactly the out-of-band energy the
            // encoder could never have modulated.  Measured: the early
            // projection alone moved the fit only 23.5% -> 20.7% out of band.
            //
            // So the law is applied last, to the published model, where
            // nothing downstream can violate it.  Demodulate the finished fit
            // against the line's own grammar basis, bandlimit the envelope,
            // remodulate.  The 2x2 normal matrix is CONSTANT along the line:
            // basisI/basisQ depend only on the carrier sample class, and any
            // four consecutive samples span all four classes, so it is
            // inverted once per line rather than per sample.
            // ---------------------------------------------------------------
            {
                if ((int)envI.size() < width) {
                    envI.resize(width, 0.0);
                    envQ.resize(width, 0.0);
                    envTmp.resize(width, 0.0);
                }

                double sII = 0.0, sIQ = 0.0, sQQ = 0.0;
                for (int k = 0; k < 4 && k < width; ++k) {
                    sII += basisI[k] * basisI[k];
                    sIQ += basisI[k] * basisQ[k];
                    sQQ += basisQ[k] * basisQ[k];
                }
                const double det = sII * sQQ - sIQ * sIQ;

                if (std::fabs(det) > 1e-9) {
                    const double inv = 1.0 / det;
                    // Centered 5-tap, half-weight ends (integer centroid;
                    // the old [xi-1, xi+2] window solved for the envelope
                    // half a sample right of the sample it published).  At
                    // 4fSC the endpoint classes coincide, so each carrier
                    // class keeps total weight 1 and the normal matrix is
                    // unchanged.
                    for (int xi = 0; xi < width; ++xi) {
                        double sIY = 0.0, sQY = 0.0;
                        for (int k = -2; k <= 2; ++k) {
                            const int j = std::clamp(xi + k, 0, width - 1);
                            const double w = (k == -2 || k == 2) ? 0.5 : 1.0;
                            const double v = w * carrierFit[j];
                            sIY += basisI[j] * v;
                            sQY += basisQ[j] * v;
                        }
                        envI[xi] = ( sQQ * sIY - sIQ * sQY) * inv;
                        envQ[xi] = (-sIQ * sIY + sII * sQY) * inv;
                    }

                    lddecode::projectExpressibleChromaEnvelope(
                        envI.data(), nullptr, width, envTmp.data());
                    std::copy(envTmp.begin(), envTmp.begin() + width,
                              envI.begin());
                    lddecode::projectExpressibleChromaEnvelope(
                        envQ.data(), nullptr, width, envTmp.data());
                    std::copy(envTmp.begin(), envTmp.begin() + width,
                              envQ.begin());

                    for (int xi = 0; xi < width; ++xi) {
                        const double cf = std::clamp(
                            envI[xi] * basisI[xi] + envQ[xi] * basisQ[xi],
                            -maxCarrierSamples, maxCarrierSamples);
                        carrierFit[xi] = cf;
                        flattened[xi] = rawWhole[xi] - cf;
                        fitRow[xi] = static_cast<float>(cf);
                    }
                }
            }

            // Re-impose the residual-consensus hull AFTER the law. The law's
            // FIR mixes neighbours and re-manufactures carrier the per-sample
            // feasible range forbids (measured: cube 45.4% of samples
            // outside, mean excess 0.56 IRE, max 44; beach 37.9%). The two
            // constraints cannot be imposed by one pass each in either order
            // -- but they are not symmetric: a clamp is a TRUE projection
            // (idempotent, per-sample), so it cannot compound the FIR, and
            // what it removes returns to Y via flattened = raw - cf, which
            // is lawful -- carrier outside the hull IS luma by the
            // conservation facts. The residual envelope kink this leaves is
            // bounded by the clamp delta; publishing impossible carrier is
            // the greater crime (it is the cube face's manufactured Y). Law
            // once, hull last. Kill switch for A/B only: LDCD_RETRHULL=0.
            {
                static const bool retrHullOn = []{
                    const char *s = std::getenv("LDCD_RETRHULL");
                    return !s || std::atoi(s) != 0;
                }();
                if (retrHullOn) {
                    for (int xi = 0; xi < width; ++xi) {
                        const auto &pp = analysisRow[xi].parallax;
                        if (!pp.residualValid) continue;
                        const double cf = std::clamp(
                            carrierFit[xi],
                            static_cast<double>(pp.residualLo),
                            static_cast<double>(pp.residualHi));
                        if (cf != carrierFit[xi]) {
                            carrierFit[xi] = cf;
                            flattened[xi]  = rawWhole[xi] - cf;
                            fitRow[xi]     = static_cast<float>(cf);
                        }
                    }
                }
            }
            // A carrier model was solved for this line, so a candidate may
            // stand on it. Both no-model paths -- the unlocked-grammar
            // `continue` above and the degenerate-width else below -- leave
            // this at 0, and the witness abstains there rather than
            // publishing raw.
            carrierFitLineValid[line] = 1;
        } else {
            for (int xi = 0; xi < width; ++xi) {
                const double cf = 0.0;
                carrierFit[xi] = cf;
                flattened[xi] = rawWhole[xi];
                fitRow[xi] = 0.0f;
                evidenceRow[xi].viewCount = 0;
            }
        }
/*
        // Optional LS refit at luma edges, retained from the previous design.
        // It is now a secondary edge repair, not the primary saturated-fill
        // carrier estimator. 
        {
            constexpr double EDGE_SOFT_IRE  = 3.0;
            constexpr double EDGE_HARD_IRE  = 10.0;
            constexpr double DISC_SOFT_IRE  = 1.0;
            constexpr double DISC_HARD_IRE  = 4.0;
            constexpr int    LS_HALF_WIN    = 2;
            constexpr bool   LS_BRIGHT_COLOR_GUARD = true;
            constexpr double LS_BRIGHT_SIDE_SOFT_IRE = 3.0;
            constexpr double LS_BRIGHT_SIDE_HARD_IRE = 10.0;
            constexpr double LS_BRIGHT_COLOR_START_IRE = 8.0;
            constexpr double LS_BRIGHT_COLOR_FULL_IRE = 24.0;

            double *edgeGate = slideMean4;
            bool anyEdge = false;

            for (int xi = 0; xi < width; ++xi) {
                const int xm = std::max(0, xi - 2);
                const int xp = std::min(width - 1, xi + 2);
                const double gradIRE =
                    std::fabs(refinedY[xp] - refinedY[xm]) * invIreScale;
                const double gate = std::clamp(
                    (gradIRE - EDGE_SOFT_IRE) /
                    std::max(1e-9, EDGE_HARD_IRE - EDGE_SOFT_IRE),
                    0.0, 1.0);
                edgeGate[xi] = gate;
                if (gate > 0.0)
                    anyEdge = true;
            }

            if (anyEdge) {
                for (int xi = 0; xi < width; ++xi) {
                    if (edgeGate[xi] <= 0.0)
                        continue;

                    const int a = std::max(0, xi - LS_HALF_WIN);
                    const int b = std::min(width - 1, xi + LS_HALF_WIN);

                    double sII = 0.0, sIQ = 0.0, sQQ = 0.0;
                    double sIY = 0.0, sQY = 0.0;

                    for (int k = a; k <= b; ++k) {
                        const double obs = rawWhole[k] - refinedY[k];
                        const double bI = basisI[k];
                        const double bQ = basisQ[k];

                        const double dist = std::fabs(static_cast<double>(k - xi));
                        const double w = 1.0 - 0.65 * std::min(
                            1.0, dist / std::max(1.0, static_cast<double>(LS_HALF_WIN)));

                        sII += w * bI * bI;
                        sIQ += w * bI * bQ;
                        sQQ += w * bQ * bQ;
                        sIY += w * bI * obs;
                        sQY += w * bQ * obs;
                    }

                    double fitI = 0.0, fitQ = 0.0;
                    const double det = sII * sQQ - sIQ * sIQ;
                    if (std::fabs(det) > 1e-9) {
                        const double inv = 1.0 / det;
                        fitI = ( sQQ * sIY - sIQ * sQY) * inv;
                        fitQ = (-sIQ * sIY + sII * sQY) * inv;
                    }

                    double lsFit = fitI * basisI[xi] + fitQ * basisQ[xi];
                    lsFit = std::clamp(lsFit,
                        -maxCarrierSamples, maxCarrierSamples);

                    const double discrepancyIRE =
                        std::fabs(lsFit - carrierFit[xi]) * invIreScale;
                    const double discGate = std::clamp(
                        (discrepancyIRE - DISC_SOFT_IRE) /
                        std::max(1e-9, DISC_HARD_IRE - DISC_SOFT_IRE),
                        0.0, 1.0);

                    double brightColorProtect = 0.0;
                    if (LS_BRIGHT_COLOR_GUARD) {
                        const int xm = std::max(0, xi - 2);
                        const int xp = std::min(width - 1, xi + 2);
                        const double lumaM = refinedY[xm];
                        const double lumaP = refinedY[xp];
                        const double dir = (lumaP >= lumaM) ? 1.0 : -1.0;
                        const double lumaMid = 0.5 * (lumaM + lumaP);
                        const double brightOffsetIRE =
                            dir * (refinedY[xi] - lumaMid) * invIreScale;
                        const double brightSideGate = smoothStep01(
                            (brightOffsetIRE - LS_BRIGHT_SIDE_SOFT_IRE) /
                            std::max(1e-9, LS_BRIGHT_SIDE_HARD_IRE -
                                           LS_BRIGHT_SIDE_SOFT_IRE));

                        const int brightIdx = (dir > 0.0) ? xp : xm;
                        const int brightJ = (brightIdx + 1 < width)
                            ? (brightIdx + 1)
                            : (brightIdx > 0 ? brightIdx - 1 : brightIdx);
                        const int xiJ = (xi + 1 < width)
                            ? (xi + 1)
                            : (xi > 0 ? xi - 1 : xi);
                        const double brightAmpIRE =
                            boundedMag(carrierFit[brightIdx],
                                       carrierFit[brightJ]) * invIreScale;
                        const double localAmpIRE =
                            boundedMag(carrierFit[xi],
                                       carrierFit[xiJ]) * invIreScale;
                        const double coloredBrightIRE =
                            std::max(brightAmpIRE, localAmpIRE);
                        const double brightColorGate = smoothStep01(
                            (coloredBrightIRE - LS_BRIGHT_COLOR_START_IRE) /
                            std::max(1e-9, LS_BRIGHT_COLOR_FULL_IRE -
                                           LS_BRIGHT_COLOR_START_IRE));

                        brightColorProtect = brightSideGate * brightColorGate;
                    }

                    double parallaxProtect = 0.0;
                    if (parallaxRow[xi].valid) {
                        const auto &p = parallaxRow[xi];
                        const double spreadT = std::clamp(
                            p.carrierSpreadIRE /
                            std::max(3.0, 0.35 * p.commonMagIRE + 1.0),
                            0.0,
                            1.0);
                        const double latticeT = std::clamp(
                            p.latticeRiskIRE /
                            std::max(3.0, 0.35 * p.commonMagIRE + 1.0),
                            0.0,
                            1.0);
                        parallaxProtect = std::max(
                            spreadT,
                            std::max(latticeT, 1.0 - p.carrierCoherence));
                    }

                    const double g =
                        edgeGate[xi] * discGate *
                        (1.0 - brightColorProtect) *
                        (1.0 - parallaxProtect);
                    if (g > 0.0) {
                        const double blended =
                            carrierFit[xi] * (1.0 - g) + lsFit * g;
                        carrierFit[xi] = blended;
                        flattened[xi] = rawWhole[xi] - blended;
                        fitRow[xi] = static_cast<float>(blended);
                    }
                }
            }
        }
*/
        // ---------------------------------------------------------------
        // PAIRED-EXCURSION RETRACTION HULL — the FINAL operation on the
        // claim (default ON, escape LDCD_FIT_PAIRHULL=0; author-directed
        // 2026-08-07). The retraction's founding job, restated by the
        // author: get the carrier out of the way with as little luma loss
        // as possible — the fit is a nuisance remover, never a chroma
        // product ("we will never use the carrier fit for direct use as
        // such"), and "carrier is essentially a series of largely
        // symmetric alternations along the line; we want those extensions
        // retracted to baseline." The one carrier component vouched for is
        // GRAMMAR: the alternations are symmetric within the schedule.
        //
        // The impossibility enforced, which waveform subtraction never
        // was: carrier alone cannot make schedule-paired excursions
        // SAME-SIGNED, and cannot make the true alternation deeper than
        // the WEAKER partner (the envelope law forbids the difference
        // across two samples). Per sample, the claim is clamped to the
        // SHARED depth of its paired extensions off the solve's own luma
        // prior:
        //
        //   same-signed pair     -> nothing is extended: claim -> 0
        //   opposite-signed pair -> |claim| <= min(own extension, best
        //                           partner's opposite extension) --
        //                           credited at the weaker (twin
        //                           confirmation turned lateral; the star
        //                           law generalized: the UNPAIRED part of
        //                           any excursion cannot be carrier)
        //
        // Clean carrier passes untouched (the partner attests the full
        // depth). One-sided structure — summits, corners, stars, the
        // iceberg band — is protected by law, not by tune.
        //
        // Placement and form are deliberate: this runs AFTER the bandwidth
        // law, per-sample and unsmoothed, so the published claim honors
        // the bound EXACTLY. A first build applied it as a ratio weight,
        // 9-tap-smoothed, before the law — and measured near-inert
        // (+0.006..0.012 retention): the smoothing diluted a 1–3 px
        // summit's protection across nine samples and the law re-expanded
        // the envelope over the clamp. Both dilutions optimized the
        // claim's chroma legality, which this view does not care about. A
        // hard bound on an impossibility has nothing to interleave (the
        // step-law's own exemption), and any out-of-band content the
        // clamp introduces lands in Y as broadband residue, not
        // alternation — the softer crime by this view's own charter.
        // LDCD_PAIRHULL_SMOOTH=1 restores the diluted form for A/B.
        // ---------------------------------------------------------------
        {
            static const bool pairHullOn = []{
                const char *e = std::getenv("LDCD_FIT_PAIRHULL");
                return !(e && std::atoi(e) == 0);
            }();
            // The convicted per-sample clamp, kept for A/B only
            // (LDCD_PAIRHULL_HARD=1): clipping one extension without its
            // partner turns the remainder into a waveform no carrier could
            // be, and the difference lands in Y as carrier-rate
            // checkerboard (author's eye, 2026-08-07: "come back like
            // gangbusters" — the §10 blend-weight-step class; the hard-zero
            // exemption covers genuine absence, not a bound flipping
            // between two renders of the same sample).
            static const bool pairHullHard = []{
                const char *e = std::getenv("LDCD_PAIRHULL_HARD");
                return e && std::atoi(e) == 1;
            }();
            const CombCarrierGrammar *hullGrammar = carrierGrammarLine(line);
            if (pairHullOn && hullGrammar && hullGrammar->grammarLocked) {
                // SYMMETRIC-AMPLITUDE FORM (author, 2026-08-07): "the
                // alternations are the peaks and nadirs of the waveform we
                // decode from... It's not the same thing to blend the
                // opposing extensions as to reduce and increase amplitude
                // in a symmetric way." The extensions are samples of ONE
                // waveform; a lawful retraction reduces that waveform's
                // AMPLITUDE — peak and nadir together, one factor. A cycle
                // holds two opposite-phase pairs, and their shared depths
                // are the attested magnitudes of the two quadratures, so
                //     attestedA = hypot(depth_inphase, depth_quadrature)
                // is the amplitude of the alternation actually present.
                // The bound is a pure amplitude statement — claim amplitude
                // <= attested amplitude — and the scaled claim stays
                // carrier-shaped, so the Y-side remainder is at worst a
                // smooth under-retraction, never checker.
                if ((int)winEnvScratch.size() < width)
                    winEnvScratch.resize(width, 1.0);
                double *dEx = winEnvScratch.data();
                for (int xi = 0; xi < width; ++xi) {
                    double d = 0.0;
                    const double c = carrierFit[xi];
                    if (xi >= 2 && xi + 2 < width &&
                        std::fabs(c) > 1e-9) {
                        const double s = c > 0.0 ? 1.0 : -1.0;
                        const double eO = rawWhole[xi] - refinedY[xi];
                        const double extOwn = std::max(0.0, s * eO);
                        const double eL =
                            rawWhole[xi - 2] - refinedY[xi - 2];
                        const double eR =
                            rawWhole[xi + 2] - refinedY[xi + 2];
                        const double extPart = std::max(
                            std::max(0.0, -s * eL),
                            std::max(0.0, -s * eR));
                        d = std::min(extOwn, extPart);
                    }
                    dEx[xi] = d;
                }
                double *wArr = slideMean4;   // scratch; floor build below
                                             // rewrites it afterwards
                for (int xi = 0; xi < width; ++xi) {
                    double w = 1.0;
                    if (xi >= 3 && xi + 3 < width) {
                        const double c = carrierFit[xi];
                        if (pairHullHard) {
                            const double ac = std::fabs(c);
                            if (ac > 0.25 * irescale)
                                w = std::clamp(dEx[xi] / ac, 0.0, 1.0);
                        } else {
                            const double dQ =
                                std::max(dEx[xi - 1], dEx[xi + 1]);
                            const double attestedA =
                                std::hypot(dEx[xi], dQ);
                            const double quad = 0.5 *
                                (carrierFit[xi - 1] - carrierFit[xi + 1]);
                            const double claimA = std::hypot(c, quad);
                            if (claimA > 0.25 * irescale)
                                w = std::clamp(attestedA / claimA,
                                               0.0, 1.0);
                        }
                    }
                    wArr[xi] = w;
                }
                if (!pairHullHard) {
                    // Light [1,2,1] settle of the weight (weight smoothing
                    // is lawful; the value never is). The amplitude form's
                    // inputs are already 3–5 sample aggregates; this kills
                    // residual single-sample chatter without diluting a
                    // summit's cycle, whose depths collapse over ~5 samples.
                    for (int xi = 0; xi < width; ++xi) {
                        const int xm = std::max(0, xi - 1);
                        const int xp = std::min(width - 1, xi + 1);
                        dEx[xi] = 0.25 * (wArr[xm] + 2.0 * wArr[xi] +
                                          wArr[xp]);
                    }
                    for (int xi = 0; xi < width; ++xi)
                        wArr[xi] = dEx[xi];
                }
                for (int xi = 0; xi < width; ++xi) {
                    const double cf = carrierFit[xi] * wArr[xi];
                    carrierFit[xi] = cf;
                    flattened[xi] = rawWhole[xi] - cf;
                    fitRow[xi] = static_cast<float>(cf);
                }
            }
        }

        // Build the carrier-cancelled floor from every legal 4-sample mean of
        // the final flattened waveform.
        if (width >= 4) {
            const int meanCount = width - 3;
            for (int s = 0; s < meanCount; ++s) {
                slideMean4[s] =
                    0.25 * (flattened[s + 0] +
                            flattened[s + 1] +
                            flattened[s + 2] +
                            flattened[s + 3]);
            }

            for (int xi = 0; xi < width; ++xi) {
                double v[4] = {0.0, 0.0, 0.0, 0.0};
                int n = 0;
                const int sFirst = std::max(0, xi - 3);
                const int sLast  = std::min(xi, meanCount - 1);

                for (int s = sFirst; s <= sLast && n < 4; ++s)
                    v[n++] = slideMean4[s];

                double floor = refinedY[xi];
                if (n >= 4)
                    floor = medoid4Anchored(v[0], v[1], v[2], v[3], refinedY[xi]);
                else if (n == 3)
                    floor = median3(v[0], v[1], v[2]);
                else if (n == 2)
                    floor = 0.5 * (v[0] + v[1]);
                else if (n == 1)
                    floor = v[0];

                floorRow[xi] = static_cast<float>(floor);
            }
        } else {
            double mean = 0.0;
            for (int xi = 0; xi < width; ++xi)
                mean += flattened[xi];
            mean /= static_cast<double>(std::max(1, width));

            for (int xi = 0; xi < width; ++xi)
                floorRow[xi] = static_cast<float>(mean);
        }
    }
        carrierRetractionModelValid = true;
    }

    if (!carrierRetractionModelValid)
        return;

    // ---------------------------------------------------------------
    // ---------------------------------------------------------------
    // Pass 1.7: FACT ANCHOR (user-approved 2026-08-01; position reversed
    // from the original facts-last placement on user direction: "put it
    // early enough to benefit the line cancellation").  At certified
    // samples the fit IS the exact carrier -- facts as facts, no gain,
    // no blend, no hull.  Stamped BEFORE the cross-line passes, so the
    // 1.75 bridge, Pass 1.9's +-2 evidence legs, and Pass 2's interline
    // cancellation all comb against FACT legs: same contract as the comb
    // candidates' cede/comb pair in a different shape -- def lines
    // idempotent/const, comp lines improved by combing against truth.
    // PERSISTENCE LAW: the passes downstream carry const-guards (the
    // bridge never adopts into a certified line; 1.9 never repairs a
    // certified sample; Pass 2's valve passes a def line through un-combed
    // — "writes combedCarrier, not the fit" protected the buffer but let
    // the published product comb the fact with comp estimates, the valve
    // violation closed 2026-08-09; the tone no-ops on covered frames), so
    // a stamped fact survives to publication structurally, not by
    // re-stamping.
    //
    // FACT STAMP: at certified samples the fit is replaced by the exact
    // fact, so no consumer of the fit can read an estimate where truth is
    // available.
    // Escape LDCD_FACT_FIT=0 (also inert at LDCD_CERT_1D=0 and under
    // --dg-discard, where no facts exist).
    //
    // The REGIONAL FACT AUDIT that used to be pooled here on the way past --
    // mean |fit - exact| per 128x32 region, measured pre-stamp -- is REMOVED
    // (2026-08-08). Its sole consumer was factTrustAt, the term that
    // conditioned the fit's share of the comp-line merge. With the fit off
    // the certified ladder there is no share left to condition, and the
    // audit was building a grid every covered frame that nothing read.
    // The idea is sound and is recorded in the return's architecture; it is
    // the wiring that had no remaining end.
    // ---------------------------------------------------------------
    {
        if (ldcdFactFitOn() && certifiedOneDLevel() >= 1 &&
            frameHasExactCoverage()) {
            for (int line = firstLine; line < lastLine; ++line) {
                if (!certifiedDefLine(line)) continue;
                const float *exRow = exactCarrierRow(line);
                if (!exRow) continue;
                float *fitRow = carrierFit_flat.data()
                    + static_cast<size_t>(line) * demodWidth;
                for (int xi = 0; xi < width; ++xi) {
                    const float ex = exRow[left + xi];
                    if (!std::isfinite(ex)) continue;
                    fitRow[xi] = ex;
                }
            }
        }
    }

    // Pass 1.75: vertical amplitude continuity for the fit (the beam fix).
    //
    // Measured (by the since-removed off-grid census): on LEGAL carrier the
    // fit holds lock on
    // 95.7% (cube) / 99.9% (beach) of comb-strong windows -- the dropouts
    // cluster on fast diagonal envelopes crossing texture (the tractor
    // beam), where one 4-sample window's normal equations lose the carrier
    // while the SAME COLUMN one line away holds it. Each dropout leaves the
    // full lobe in retracted Y as a checker patch.
    //
    // The bridge is SELECTION UNDER EVIDENCE, never averaging: at a window
    // whose fit amplitude collapsed while the canonical raw bandpass is
    // strong and the energy is NOT proven-illegal (the grid's virtuous
    // refusals stay refused), each grammar-locked vertical neighbour's fit
    // is converted along the compatible rotation path -- partner locked
    // frame -> common 4fsc -> this line's locked frame, through the burst
    // phasors -- and remodulated on THIS line's basis. The candidate that
    // explains this line's own raw bandpass window better than the
    // collapsed fit does (and better than the other partner) is adopted
    // whole. Adopted samples are then RESTRICTED as always: residual-
    // consensus range and amplitude ceiling. No qualifying partner => the
    // dropout stands -- an honest hole, never a synthetic patch.
    //
    // flatFloor at bridged windows still reflects the pre-bridge fit
    // (sparse; revisit if Pass 2's gates misbehave at bridged columns).
    // Kill switch for A/B only: LDCD_FIT_BRIDGE=0.
    {
        static const bool bridgeOn = []{
            const char *s = std::getenv("LDCD_FIT_BRIDGE");
            return !s || std::atoi(s) != 0;
        }();
        static const double kBridgeRawStrongIRE = 8.0;
        static const double kBridgeCollapse     = 0.4;
        static const double kBridgePartnerHold  = 0.6;

        std::vector<std::uint8_t> bridged(width);
        auto lineBasis = [&](const CombCarrierGrammar *g, double *bI, double *bQ) {
            for (int p = 0; p < 4; ++p) {
                bI[p] = remodLockedToShiftedComposite(
                    1.0, 0.0, p, g->burstCos, g->burstSin,
                    spLUT_locked, cpLUT_locked);
                bQ[p] = remodLockedToShiftedComposite(
                    0.0, 1.0, p, g->burstCos, g->burstSin,
                    spLUT_locked, cpLUT_locked);
            }
        };
        // 4-sample window LS demod on a line basis; every window holds each
        // class once, so the normal matrix is the same for all windows.
        auto windowIQ = [&](const float *row, int x, int line,
                            const double *bI, const double *bQ,
                            double Sii, double Siq, double Sqq,
                            double &io, double &qo) -> bool {
            double SiY = 0.0, SqY = 0.0;
            for (int k = 0; k < 4; ++k) {
                const int cls = carrierSampleClass(line, left + x + k);
                SiY += bI[cls] * (double)row[x + k];
                SqY += bQ[cls] * (double)row[x + k];
            }
            const double det = Sii * Sqq - Siq * Siq;
            if (std::fabs(det) < 1e-12) return false;
            io = ( Sqq * SiY - Siq * SqY) / det;
            qo = (-Siq * SiY + Sii * SqY) / det;
            return true;
        };
        auto winRmsIRE = [&](const auto *row, int x) {
            double e = 0.0;
            for (int k = 0; k < 4; ++k)
                e += (double)row[x + k] * (double)row[x + k];
            return std::sqrt(e * 0.25) * invIreScale;
        };

        auto bridgeLine = [&](int line) {
            const CombCarrierGrammar *gL = carrierGrammarLine(line);
            if (!gL || !gL->grammarLocked) return;
            // Fact persistence: a certified line is const -- the bridge
            // never adopts into it (it serves as a fact PARTNER instead).
            if (certifiedDefLine(line)) return;
            const double *bpL = locked1DRawBandpass_line(line);
            float *fitL = carrierFit_flat.data()
                          + static_cast<size_t>(line) * demodWidth;
            const auto *anL = carrierAnalysis_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
            const double maxCarrierL =
                maxCarrierAmpIREFromScale(gL->carrierScale) * irescale;
            if (!bpL) return;

            double bIL[4], bQL[4];
            lineBasis(gL, bIL, bQL);
            double SiiL = 0, SiqL = 0, SqqL = 0;
            for (int p = 0; p < 4; ++p) {
                SiiL += bIL[p] * bIL[p];
                SiqL += bIL[p] * bQL[p];
                SqqL += bQL[p] * bQL[p];
            }

            const int partnerStep = carrierFrameVerticalAllowed(line) ? 1 : 2;
            std::fill(bridged.begin(), bridged.end(), std::uint8_t{0});

            for (int x = 0; x + 3 < width; ++x) {
                if (bridged[x]) continue;
                const double rawAmp = winRmsIRE(bpL, x);
                if (rawAmp < kBridgeRawStrongIRE) continue;
                if (winRmsIRE(fitL, x) >= kBridgeCollapse * rawAmp) continue;
                if (lddecode::carrierIllegalProof(
                        (double)anL[x + 1].carrierConformance,
                        (double)anL[x + 1].conformanceSupportFraction) >= 0.7)
                    continue;

                double errOwn = 0.0;
                for (int k = 0; k < 4; ++k) {
                    const double d = bpL[x + k] - (double)fitL[x + k];
                    errOwn += d * d;
                }

                double bestErr = errOwn;
                double bestC[4];
                bool haveBest = false;
                for (int dp = -partnerStep; dp <= partnerStep;
                     dp += 2 * partnerStep) {
                    const int lp = line + dp;
                    if (lp < firstLine || lp >= lastLine) continue;
                    const CombCarrierGrammar *gP = carrierGrammarLine(lp);
                    if (!gP || !gP->grammarLocked) continue;
                    const double *bpP = locked1DRawBandpass_line(lp);
                    const float *fitP = carrierFit_flat.data()
                                        + static_cast<size_t>(lp) * demodWidth;
                    const auto *anP = carrierAnalysis_flat.data()
                                      + static_cast<size_t>(lp) * demodWidth;
                    if (!bpP) continue;
                    // Partner must HOLD lock on its own raw band, lawfully.
                    const double rawAmpP = winRmsIRE(bpP, x);
                    if (rawAmpP < kBridgeRawStrongIRE) continue;
                    if (winRmsIRE(fitP, x) < kBridgePartnerHold * rawAmpP)
                        continue;
                    if (lddecode::carrierIllegalProof(
                            (double)anP[x + 1].carrierConformance,
                            (double)anP[x + 1].conformanceSupportFraction)
                            >= 0.7)
                        continue;

                    double bIP[4], bQP[4];
                    lineBasis(gP, bIP, bQP);
                    double SiiP = 0, SiqP = 0, SqqP = 0;
                    for (int p = 0; p < 4; ++p) {
                        SiiP += bIP[p] * bIP[p];
                        SiqP += bIP[p] * bQP[p];
                        SqqP += bQP[p] * bQP[p];
                    }
                    double iP, qP;
                    if (!windowIQ(fitP, x, lp, bIP, bQP,
                                  SiiP, SiqP, SqqP, iP, qP))
                        continue;
                    // Compatible rotation path: partner locked -> 4fsc ->
                    // this line's locked frame.
                    double i4, q4, iL, qL;
                    lockedTo4fsc(iP, qP, gP->burstCos, gP->burstSin, i4, q4);
                    fourfscToLocked(i4, q4, gL->burstCos, gL->burstSin,
                                    iL, qL);
                    double c[4];
                    double err = 0.0;
                    for (int k = 0; k < 4; ++k) {
                        const int cls = carrierSampleClass(line, left + x + k);
                        c[k] = iL * bIL[cls] + qL * bQL[cls];
                        const double d = bpL[x + k] - c[k];
                        err += d * d;
                    }
                    if (err < bestErr) {
                        bestErr = err;
                        for (int k = 0; k < 4; ++k) bestC[k] = c[k];
                        haveBest = true;
                    }
                }
                if (!haveBest) continue;

                for (int k = 0; k < 4; ++k) {
                    double nv = bestC[k];
                    const auto &pp = anL[x + k].parallax;
                    if (pp.residualValid)
                        nv = std::clamp(nv, (double)pp.residualLo,
                                            (double)pp.residualHi);
                    nv = std::clamp(nv, -maxCarrierL, maxCarrierL);
                    fitL[x + k] = static_cast<float>(nv);
                    bridged[x + k] = 1;
                }
            }
        };
        if (bridgeOn) {
            // Two sweeps, top-down then bottom-up: a window bridged in one
            // sweep becomes a QUALIFYING PARTNER for the adjacent line later
            // in sweep order, so cures cascade through multi-line dropout
            // clusters from both intact boundary lines inward. Still
            // selection under evidence at every step -- the cascade extends
            // only where each adoption beats the collapsed fit on this
            // line's own raw bandpass window.
            for (int line = firstLine; line < lastLine; ++line)
                bridgeLine(line);
            for (int line = lastLine - 1; line >= firstLine; --line)
                bridgeLine(line);
        }
    }

    // ---------------------------------------------------------------
    // RETIRED: the locked-1D repair return path.
    //
    // This used to add `locked1DParallaxRepairDelta` into carrierFit, to keep
    // the retraction tracking the repaired locked-1D carrier.  That
    // requirement belonged to the era when the fitted carrier was itself
    // promoted into the 1D intake; for an INDEPENDENT inverse-encoder view it
    // is backwards.  It made the dependency
    //
    //     raw analysis -> 1D decision and repair -> carrier-retracted view
    //
    // where the intended shape is a shared, application-neutral analysis
    // feeding ordinary locked 1D and the native retraction as SIBLINGS.  With
    // the delta folded in, a change made to improve 1D silently altered the
    // supposedly independent view, and the two could no longer serve as
    // checks on one another.
    //
    // Removing it also resolves a second inconsistency: flatFloor is built
    // from `raw - carrierFit` inside the model block above, while these
    // deltas were applied afterwards, so Pass 2's reach gates were evaluating
    // floors derived from a different carrier operand than the one they were
    // promoting.  Both operands are now the same model.
    //
    // Shared inputs remain legitimate and are unaffected: burst calibration,
    // carrier grammar, raw samples, the coarse-residual options in
    // carrierAnalysis, and the encoder bandwidth law are all
    // application-neutral.  What is gone is the correction selected
    // specifically for the locked-1D OUTPUT.  If a promoted product that
    // deliberately tracks the repaired 1D source is ever wanted, it should be
    // derived as its own buffer rather than by mutating the native model.
    // ---------------------------------------------------------------
    const lddecode::CombReachSourceFrame carrierFitSource =
        lddecode::makeCarrierFitScalarReachSource();

    // Shared soft gate for the vertical passes below (Pass 1.9 evidence
    // admission and Pass 2's reach/content-break guards read the same ramp).
    auto softReachGate = [](double diffIRE, double softIRE, double hardIRE) {
        if (diffIRE <= softIRE)
            return 1.0;
        if (diffIRE >= hardIRE)
            return 0.0;
        const double t = (diffIRE - softIRE) /
                         std::max(1e-9, hardIRE - softIRE);
        return 1.0 - (t * t * (3.0 - 2.0 * t));
    };

    // ---------------------------------------------------------------
    // Pass 1.9: vertical schedule law on the fit — proven-standing
    // exclusion (the title-bevel fix, 2026-08-01).
    //
    // The encoder bandwidth law (imposed once, at the Pass-1 model
    // boundary) is a LATERAL law: legal carrier lives in fSC±1.3 MHz.
    // The schedule adds a VERTICAL law the fit never honoured: legal
    // carrier INVERTS against every grammar-certified Opposite partner.
    // A component that MATCHES its Opposite partners is image-locked
    // luma standing in the carrier band — the bandpass leak of a strong
    // vertical image edge (same D²Y ripple on every line).  The fit
    // follows raw and absorbs that leak; every full-subtraction product
    // (native, the anchor ladder's fit rung) then confiscates it from Y,
    // and the renderer's complement demodulates it as line-alternating
    // false colour (the demod axis flips per line, so standing composite
    // paints alternating green/magenta).  Measured on the Emissary title
    // bevel: retracted Y sat a mean 5.6 IRE below comb Y at the leak
    // columns; every election share given the fit-derived candidate
    // rendered as the alternation.
    //
    // Shape of the law, and why it is imposed on the MODEL:
    //   * The pedestal law forbids scaling the withdrawal (raw − w·fit
    //     was the retired floating-pedestal product).  So the fit itself
    //     is repaired: the PROVEN-standing component is removed from the
    //     model, and what the encoder could not have modulated never
    //     enters the carrier — Y keeps it by construction, upstream of
    //     every election and every consumer.
    //   * Proof, not detection: the signed schedule correlation of the
    //     fit against FIELD-PURE legs only (±2 same-field Opposite; the
    //     ±1 interfield leg is 20 ms away and motion scrambles its
    //     relation — the beach garment lesson), integrated VERTICALLY
    //     down the column, pointwise laterally.  The lateral envelope
    //     aperture Pass 2's corroboration uses is WRONG for this law and
    //     was measured failing on the bevel itself: a standing leak is a
    //     1-6 px COLUMN (it is luma; only chroma is bound by the 1.3 MHz
    //     envelope limit), and lateral smoothing diluted its pointwise
    //     +0.9 match with the surrounding legal letter chroma down to
    //     sc ~0.34, under the knee.  Vertically the aperture and the
    //     evidence agree: the leak persists down the image edge (a
    //     letter stem is 50+ lines), so the proof integrates where the
    //     phenomenon lives.  A minimum evidence run (6 contributing leg
    //     lines) keeps transient flukes out.  The license is the mirror
    //     of scheduleAlternationLicense: it opens only where the fit
    //     provably MATCHES where the schedule demands inversion
    //     (sc ≥ +0.5, full at +0.9).  Everything short of proof leaves
    //     the model untouched — no confidence hedging, the unproven fit
    //     stays committed.
    //   * The standing estimator is the waveform mid-sum s = ½(fit +
    //     opp): exact for a chroma+standing mix (c+s with partner −c+s
    //     gives s exactly), zero for pure legal chroma.  The license is
    //     envelope-smooth by construction, so lic·s cannot manufacture
    //     out-of-band sidebands (no fast-gain AM).
    //   * HULL direction: the subtraction may only REMOVE carrier —
    //     the repaired sample is clamped to the same sign and no larger
    //     magnitude than the fit (a true per-sample projection; what it
    //     removes returns to Y via raw − fit, which is lawful).
    //   * flatFloor keeps the pre-law fit (same sparse-staleness caveat
    //     Pass 1.75 carries; revisit if Pass 2's gates misbehave at
    //     repaired columns).
    // Known cost, accepted with eyes open: sustained diagonal chroma at
    // slopes that rotate the carrier 180° per 2 lines reads as matching
    // at ±2 and would be confiscated toward Y where the envelope
    // correlation stays ≥ +0.5 over a full coherence length — the same
    // verdict Pass 2's ownership license already renders on that
    // content ("energy stays in Y").  Kill switch for A/B only:
    // LDCD_STANDING_LAW=0.
    // ---------------------------------------------------------------
    {
        static const bool standingLawOn = []{
            const char *s = std::getenv("LDCD_STANDING_LAW");
            return !s || std::atoi(s) != 0;
        }();
        if (standingLawOn) {
            const double ampFloorS = 3.0 * irescale;    // 3 IRE
            const double powFloorS = ampFloorS * ampFloorS;
            constexpr int kVertHalf = 8;   // vertical window: ±8 lines
            constexpr int kMinLegLines = 6; // minimum evidence run

            // Sweep 1: per-line pointwise leg products against the
            // ORIGINAL fit (nothing is modified until sweep 2, so plane
            // reads are order-safe).  Five planes over the active span.
            const int nLines = lastLine - firstLine;
            const size_t planeN = static_cast<size_t>(nLines) * width;
            std::vector<float> pP(planeN, 0.0f);   // Σ g·fit·leg
            std::vector<float> pS(planeN, 0.0f);   // Σ g·fit²
            std::vector<float> pL(planeN, 0.0f);   // Σ g·leg²
            std::vector<float> pO(planeN, 0.0f);   // Σ g·leg
            std::vector<float> pW(planeN, 0.0f);   // Σ g

            for (int line = firstLine; line < lastLine; ++line) {
                const CombCarrierGrammar *gL = carrierGrammarLine(line);
                if (!gL || !gL->grammarLocked)
                    continue;
                const float *fitRow = carrierFit_flat.data()
                                      + static_cast<size_t>(line) * demodWidth;
                const float *floorRow = flatFloor_flat.data()
                                        + static_cast<size_t>(line) * demodWidth;
                float *rP = pP.data() +
                    static_cast<size_t>(line - firstLine) * width;
                float *rS = pS.data() +
                    static_cast<size_t>(line - firstLine) * width;
                float *rL = pL.data() +
                    static_cast<size_t>(line - firstLine) * width;
                float *rO = pO.data() +
                    static_cast<size_t>(line - firstLine) * width;
                float *rW = pW.data() +
                    static_cast<size_t>(line - firstLine) * width;

                for (int dir = -1; dir <= 1; dir += 2) {
                    const int targetLine = line + 2 * dir;
                    if (targetLine < firstLine || targetLine >= lastLine)
                        continue;
                    const CombCarrierGrammar *gT =
                        carrierGrammarLine(targetLine);
                    if (!gT || !gT->grammarLocked)
                        continue;
                    const lddecode::CombReachReply reach =
                        combReachIndex.query(
                            {line, targetLine, left, left,
                             lddecode::CombReachUse::FieldScalarCancel,
                             carrierFitSource});
                    if (!(reach.allowScalarCancel && reach.mayBecomeVideo &&
                          reach.carrierRelation ==
                              lddecode::CarrierPhaseRelation::Opposite))
                        continue;
                    const float *legFit = carrierFit_flat.data()
                        + static_cast<size_t>(targetLine) * demodWidth;
                    const float *legFloor = flatFloor_flat.data()
                        + static_cast<size_t>(targetLine) * demodWidth;
                    for (int xi = 0; xi < width; ++xi) {
                        const double legBreakIRE =
                            std::fabs(static_cast<double>(floorRow[xi]) -
                                      static_cast<double>(legFloor[xi])) *
                            invIreScale;
                        const double g = softReachGate(legBreakIRE, 3.0, 10.0);
                        if (g <= 0.0)
                            continue;
                        const double c = static_cast<double>(fitRow[xi]);
                        const double n = static_cast<double>(legFit[xi]);
                        rP[xi] += static_cast<float>(g * c * n);
                        rS[xi] += static_cast<float>(g * c * c);
                        rL[xi] += static_cast<float>(g * n * n);
                        rO[xi] += static_cast<float>(g * n);
                        rW[xi] += static_cast<float>(g);
                    }
                }
            }

            // Sweep 2: sliding vertical sums per column, license, repair.
            // In-place emit is safe: line L is only ever modified at step
            // L, and every windowed read comes from the sweep-1 planes.
            std::vector<double> vP(width), vS(width), vL(width), vN(width);
            auto addLine = [&](int l, double sgn) {
                if (l < firstLine || l >= lastLine) return;
                const size_t r0 = static_cast<size_t>(l - firstLine) * width;
                for (int xi = 0; xi < width; ++xi) {
                    vP[xi] += sgn * pP[r0 + xi];
                    vS[xi] += sgn * pS[r0 + xi];
                    vL[xi] += sgn * pL[r0 + xi];
                    if (pW[r0 + xi] > 0.0f)
                        vN[xi] += sgn;
                }
            };
            std::fill(vP.begin(), vP.end(), 0.0);
            std::fill(vS.begin(), vS.end(), 0.0);
            std::fill(vL.begin(), vL.end(), 0.0);
            std::fill(vN.begin(), vN.end(), 0.0);
            for (int l = firstLine; l <= firstLine + kVertHalf &&
                                    l < lastLine; ++l)
                addLine(l, +1.0);

            for (int line = firstLine; line < lastLine; ++line) {
                float *fitRow = carrierFit_flat.data()
                                + static_cast<size_t>(line) * demodWidth;
                const size_t r0 =
                    static_cast<size_t>(line - firstLine) * width;
                const CombCarrierGrammar *gL = carrierGrammarLine(line);
                // Fact persistence: never repair a certified sample (the
                // stamped fact is const; its legs still serve as evidence).
                const float *exGuard = certifiedDefLine(line)
                    ? exactCarrierRow(line) : nullptr;

                if (gL && gL->grammarLocked) {
                    for (int xi = 0; xi < width; ++xi) {
                        if (exGuard &&
                            std::isfinite(exGuard[left + xi]))
                            continue;
                        const double selfE = vS[xi];
                        const double legE = vL[xi];
                        const bool observable =
                            vN[xi] >= kMinLegLines &&
                            selfE > 0.0 && legE > 0.0 &&
                            selfE / std::max(1.0, vN[xi]) >= powFloorS;
                        const double sc = observable
                            ? vP[xi] / std::sqrt(selfE * legE)
                            : std::numeric_limits<double>::quiet_NaN();
                        const double licHere = (observable && sc == sc)
                            ? lddecode::scheduleAlternationLicense(-sc)
                            : 0.0;
                        const double wHere =
                            static_cast<double>(pW[r0 + xi]);
                        const double fit = static_cast<double>(fitRow[xi]);
                        double s = std::numeric_limits<double>::quiet_NaN();
                        double nf = fit;
                        if (licHere > 0.0 && wHere > 0.0) {
                            s = 0.5 * (fit +
                                static_cast<double>(pO[r0 + xi]) / wHere);
                            nf = fit - licHere * s;
                            // Removal-only hull: same sign, no larger
                            // magnitude — a true per-sample projection.
                            if (fit >= 0.0)
                                nf = std::clamp(nf, 0.0, fit);
                            else
                                nf = std::clamp(nf, fit, 0.0);
                        }
                        if (nf != fit) {
                            fitRow[xi] = static_cast<float>(nf);
                        }
                    }
                }

                // Slide the vertical window.
                addLine(line - kVertHalf, -1.0);
                addLine(line + kVertHalf + 1, +1.0);
            }
        }
    }

    // ---------------------------------------------------------------
    // Pass 1.95: CERTIFIED-ANCHOR ENVELOPE HULL on the fit (user,
    // 2026-08-09: the fit "ought to also be bandwidth constrained along
    // feasibleband lines between certified anchors").
    //
    // Between the certified anchor lines the fit is an estimate, but the
    // anchors certify what carrier amplitude this neighbourhood actually
    // holds: the anchor ceiling is pooled from the exact channel's own
    // envelope, laterally max-dilated and margin-padded, so it encodes
    // an impossible, never a preference. The fit's envelope may not
    // assert more than the facts bracketing it license.
    //
    // Form is the sanctioned one (the envelope-legality precedent, the
    // same shape as the parallax policing in Pass 1): a w <= 1 hull
    // gathered AND applied at envelope scale through the encoder's own
    // 9-tap chroma kernel (feasibleband.h), so it can only return energy
    // to Y and cannot manufacture out-of-band sidebands. Both sides of
    // the comparison live at envelope scale — the fit side is the
    // smoothed quadrature envelope, the ceiling is regional and slower
    // still — per the falsified-ceiling record (a point envelope held to
    // an envelope-scale bound rectifies noise into real attenuation of
    // legal colour; measured -25% on saturated bars).
    //
    // Def lines are skipped whole: their fit is the fact (Pass 1.7), and
    // a fact is not scaled. Uncovered frames publish no ceiling and the
    // pass is inert. Where the ceiling carries no authority (+inf), w
    // stays 1 — absence of fact bounds nothing.
    // ---------------------------------------------------------------
    {
        if (!anchorCeilingValid) buildAnchorCeiling();
        if (frameHasExactCoverage() && !anchorCeiling_flat.empty()) {
            std::vector<double> envRaw(width), envSm(width),
                                wRaw(width), wSm(width);
            for (int line = firstLine; line < lastLine; ++line) {
                if (certifiedDefLine(line)) continue;
                const float *ceilRow = anchorCeilingRow(line);
                if (!ceilRow) continue;
                float *fitRow = carrierFit_flat.data()
                                + static_cast<size_t>(line) * demodWidth;

                // Quadrature envelope of the fit (adjacent 4fSC samples
                // are 90 degrees apart), then the encoder kernel brings
                // it to envelope scale before any comparison.
                for (int xi = 0; xi < width; ++xi) {
                    const double a = static_cast<double>(fitRow[xi]);
                    const double b = static_cast<double>(
                        fitRow[std::min(xi + 1, width - 1)]);
                    envRaw[xi] = std::sqrt(a * a + b * b) * invIreScale;
                }
                lddecode::projectExpressibleChromaEnvelope(
                    envRaw.data(), nullptr, width, envSm.data());

                bool any = false;
                for (int xi = 0; xi < width; ++xi) {
                    const double c = static_cast<double>(ceilRow[left + xi]);
                    double w = 1.0;
                    if (std::isfinite(c) && envSm[xi] > 1e-6)
                        w = std::min(1.0, c / envSm[xi]);
                    wRaw[xi] = w;
                    if (w < 1.0) any = true;
                }
                if (!any) continue;

                // The gain itself is smoothed by the same kernel before
                // application (the AM law: gathered and applied at
                // envelope scale), then clamped as a hull.
                lddecode::projectExpressibleChromaEnvelope(
                    wRaw.data(), nullptr, width, wSm.data());
                for (int xi = 0; xi < width; ++xi) {
                    const double w = std::clamp(wSm[xi], 0.0, 1.0);
                    if (w < 1.0)
                        fitRow[xi] = static_cast<float>(fitRow[xi] * w);
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // Pass 2: interline comb on carrierFit → combedCarrier.
    //
    // Leg roster, not a fixed pairing.  The previous form admitted only
    // grammar-Opposite ±1 partners; on the NTSC schedule those alternate
    // sides by line parity (even lines pair down, odd lines pair up, never
    // both), so the stage degenerated into fixed disjoint 2-line couples —
    // no comparison ever crossed a couple boundary, and every decision
    // rendered at a 2-line quantum (measured on the Borg cube: state runs
    // of median 1 line, boundary Y-steps 2x the picture's natural line
    // variation).  A vertical structure was combed through different
    // arithmetic on adjacent lines.
    //
    // The comb-like replacement gathers every grammar-certified Opposite
    // partner among {±1, ±2}: the ±1 partner (whichever side the schedule
    // makes Opposite, interfield) and both ±2 partners (same-field, always
    // Opposite on the NTSC schedule — relations compose Opp∘Same = Opp).
    // Every line now has both-side vertical support, so the couple quantum
    // dissolves; which legs exist is the grammar's answer per pair, never a
    // structural assumption here.  Legality: FrameScalarCancel for the
    // interfield rung (carries the edit-split frame-vertical block),
    // FieldScalarCancel for the same-field rung.  Only Opposite relations
    // cancel — a Same partner sees chroma and image-locked alien with the
    // same sign and cannot discriminate, so it contributes nothing here.
    // (softReachGate is shared with Pass 1.9, defined above.)
    // ---------------------------------------------------------------

    // Sync-tone actuator: fit is complete here; every consumer below
    // (comb pass, legs, retraction output) reads the rotated fit.
    applyToneToFit(prevF);

    // Per-leg raw-gate scratch, hoisted across lines.  Slot order is fixed
    // (up2, up1, dn1, dn2); an absent leg keeps null pointers and a zeroed
    // gate row.
    constexpr int kNLegs = 4;
    std::vector<double> legGateScratch[kNLegs];
    for (int k = 0; k < kNLegs; ++k)
        legGateScratch[k].assign(static_cast<size_t>(width), 0.0);

    // Corroboration evidence scratch (see the envelope-scale corroboration
    // block inside the loop), hoisted across lines like the gate scratch.
    std::vector<double> corrSelf(static_cast<size_t>(width), 0.0);
    std::vector<double> corrNum(static_cast<size_t>(width), 0.0);
    std::vector<double> corrDen(static_cast<size_t>(width), 0.0);
    std::vector<double> corrScr0(static_cast<size_t>(width), 0.0);
    std::vector<double> corrScr1(static_cast<size_t>(width), 0.0);
    std::vector<double> corrScr2(static_cast<size_t>(width), 0.0);

    for (int line = firstLine; line < lastLine; ++line) {
        const float *fitRow = carrierFit_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        const float *floorRow = flatFloor_flat.data()
                                + static_cast<size_t>(line) * demodWidth;
        float *combRow = combedCarrier_flat.data()
                         + static_cast<size_t>(line) * demodWidth;
        float *wRow = carrierCorroboration_flat.data()
                      + static_cast<size_t>(line) * demodWidth;

        const CombCarrierGrammar *grammar = carrierGrammarLine(line);
        if (!grammar || !grammar->grammarLocked) {
            std::fill(combRow, combRow + width, 0.0f);
            // No grammar, no corroboration: the published product withdraws
            // nothing here (fail closed, same term as the license).
            std::fill(wRow, wRow + width, 0.0f);
            continue;
        }

        // THE VALVE (user, 2026-08-09): the covered interfield comb flows
        // one way. Certified spreads its value — the comp lines' ±1 legs
        // are the fact-stamped def fits (Pass 1.7), and those lines comb
        // vigorously against them below — but the reverse is prohibited: a
        // def line is already separated, and blending it with comp-line
        // estimates adulterates a fact with a model. The Pass 1.7
        // persistence law protected the FIT from every downstream pass but
        // let this pass comb the fact into its published product anyway
        // ("Pass 2 writes combedCarrier, not the fit" satisfied the letter
        // and lost the law). So the def line passes through un-combed,
        // whole: combedCarrier IS the fit, and the corroboration is 1.0 —
        // a conservation identity is its own proof, and the withdrawal it
        // licenses is exactly raw - exact = certified luma. Uniform per
        // line: the sparse repair-denied samples on a def line carry the
        // law-bounded estimate fit and inherit the line's authority rather
        // than manufacturing a per-sample provenance seam.
        if (ldcdFactFitOn() && certifiedDefLine(line)) {
            for (int xi = 0; xi < width; ++xi) {
                combRow[xi] = fitRow[xi];
                wRow[xi] = 1.0f;
            }
            continue;
        }

        const float *eligRow =
            carrierEligibility_flat.data() + static_cast<size_t>(line) * demodWidth;

        // Build the leg roster for this line.
        struct FitLeg {
            const float *fit = nullptr;
            const float *lumaFloor = nullptr;
            const float *elig = nullptr;   // graded participation [0,1]
            double *wRaw = nullptr;
            bool present = false;
        };
        FitLeg legs[kNLegs];
        constexpr int legOffset[kNLegs] = {-2, -1, +1, +2};
        constexpr lddecode::CombReachUse legUse[kNLegs] = {
            lddecode::CombReachUse::FieldScalarCancel,
            lddecode::CombReachUse::FrameScalarCancel,
            lddecode::CombReachUse::FrameScalarCancel,
            lddecode::CombReachUse::FieldScalarCancel,
        };
        for (int k = 0; k < kNLegs; ++k) {
            const int targetLine = line + legOffset[k];
            legs[k].wRaw = legGateScratch[k].data();
            if (targetLine < firstLine || targetLine >= lastLine)
                continue;
            const CombCarrierGrammar *g = carrierGrammarLine(targetLine);
            if (!g || !g->grammarLocked)
                continue;
            const lddecode::CombReachReply reach = combReachIndex.query(
                {line, targetLine, left, left, legUse[k], carrierFitSource});
            if (!(reach.allowScalarCancel && reach.mayBecomeVideo &&
                  reach.carrierRelation ==
                      lddecode::CarrierPhaseRelation::Opposite))
                continue;
            legs[k].fit = carrierFit_flat.data() +
                          static_cast<size_t>(targetLine) * demodWidth;
            legs[k].lumaFloor = flatFloor_flat.data() +
                                static_cast<size_t>(targetLine) * demodWidth;
            legs[k].elig = carrierEligibility_flat.data() +
                           static_cast<size_t>(targetLine) * demodWidth;
            legs[k].present = true;
        }

        // -------------------------------------------------------------
        // Envelope-scale schedule corroboration for the PUBLISHED product.
        //
        // Same evidence family as the ownership license below — relation-
        // folded alternation of this line's fit against its certified-
        // Opposite legs, mapped through scheduleAlternationLicense() — but
        // integrated at the aperture the encoder bandwidth law defines.
        // The license's one-cycle window returns chance-level verdicts on
        // dithered content (cube box: 43% of pixels "inverting", a coin
        // flip per pixel); the same correlation integrated at envelope
        // scale is decisive, because a legal envelope cannot vary faster
        // than ~1.3 MHz and so its alternation cannot flicker at pixel
        // pitch.
        //
        // LP(fit·leg) extracts the relation-folded envelope inner product
        // directly in the raw domain — the carrier-rate cross terms land
        // at 2fSC and are rejected by the smoothing — so no basis or sign
        // convention is applied a second time.  The smoother is the
        // encoder's own envelope filter (feasibleband.h), applied twice
        // for one full coherence length of support: the weight therefore
        // cannot vary faster than the envelope it scales, and w·fit stays
        // inside the legal band by construction — the fast-gain AM defect
        // documented at the Pass-2 emit cannot occur on this product.
        //
        // The proof standard is uniform: an unproven fit withdraws NOTHING,
        // quiet included.  This deliberately breaks with the ownership
        // license's quiet-licenses-at-1 convention: that convention is
        // about confiscating a ~0 REMAINDER, but here w scales a published
        // withdrawal, and the Pass-2 mechanism study already measured that
        // subtracting 2-3 IRE of absorbed alien on scattered quiet lines
        // is not harmless to column straightness.  Measured on the cube
        // box: quiet-at-1 put 0.65 IRE of unlawful withdrawal through
        // (48% lawful share); quiet-at-0 withdraws less but lawfully.
        // Sub-floor legal chroma stays in Y at <= 3 IRE — the loud lawful
        // carrier is what the corroboration exists to prove.  Eligibility
        // is NOT consulted here: schedule-conformance already downweighted
        // the solve that produced the fit, and counting the same evidence
        // twice is how verdicts start flipping at one line's bit.
        // -------------------------------------------------------------
        {
            const double ampFloorC = 3.0 * irescale;   // 3 IRE envelope
            const double powFloorC = 0.5 * ampFloorC * ampFloorC;

            // Two passes of the encoder envelope filter; out == in is fine
            // (the intermediate lives in corrScr2), out == corrScr2 is not.
            auto smoothEnv = [&](const double *in, double *out) {
                lddecode::projectExpressibleChromaEnvelope(
                    in, nullptr, width, corrScr2.data());
                lddecode::projectExpressibleChromaEnvelope(
                    corrScr2.data(), nullptr, width, out);
            };

            for (int xi = 0; xi < width; ++xi) {
                const double c = static_cast<double>(fitRow[xi]);
                corrScr0[xi] = c * c;
            }
            smoothEnv(corrScr0.data(), corrSelf.data());

            std::fill(corrNum.begin(), corrNum.begin() + width, 0.0);
            std::fill(corrDen.begin(), corrDen.begin() + width, 0.0);

            for (int k = 0; k < kNLegs; ++k) {
                if (!legs[k].present)
                    continue;
                // FIELD-PURE evidence only: the ±2 legs are the same field,
                // simultaneous with this line.  The ±1 interfield leg is
                // 20 ms away — under vertical motion it presents the other
                // field's displaced content, whose carrier relation is
                // scrambled by the motion phase (measured on the beach
                // garment edges: ±2 corr −1 while ±1 corr +1 at the same
                // pixel).  Folding it into the aggregate made w flicker at
                // line pitch across moving chroma edges, and a line-rate w
                // on a strong carrier renders a checkerboard.  Grammar
                // legality is not content simultaneity; temporal evidence
                // belongs to the temporal machinery, not this weight.
                if (legOffset[k] == -1 || legOffset[k] == +1)
                    continue;
                const float *legFit = legs[k].fit;
                for (int xi = 0; xi < width; ++xi) {
                    const double c = static_cast<double>(fitRow[xi]);
                    const double n = static_cast<double>(legFit[xi]);
                    corrScr0[xi] = c * n;
                    corrScr1[xi] = n * n;
                }
                smoothEnv(corrScr0.data(), corrScr0.data());
                smoothEnv(corrScr1.data(), corrScr1.data());
                for (int xi = 0; xi < width; ++xi) {
                    if (corrScr1[xi] < powFloorC)
                        continue;   // this leg is unobservable here
                    // Content-break guard, same carrier-free criterion the
                    // cancellation trusts (flatFloor delta): a leg across a
                    // luma content boundary is a straddling window, and a
                    // straddling window is not evidence for any pixel — its
                    // own corr is exactly the noisy verdict that wandered at
                    // edge strips.  Graded, so authority fades instead of
                    // flipping; both legs faded -> unobservable -> w=0,
                    // uniformly across the strip.
                    const double legBreakIRE =
                        std::fabs(static_cast<double>(floorRow[xi]) -
                                  static_cast<double>(legs[k].lumaFloor[xi])) *
                        invIreScale;
                    const double g = softReachGate(legBreakIRE, 3.0, 10.0);
                    if (g <= 0.0)
                        continue;
                    corrNum[xi] += g * corrScr0[xi];
                    corrDen[xi] += g * std::sqrt(corrSelf[xi] * corrScr1[xi]);
                }
            }

            for (int xi = 0; xi < width; ++xi) {
                double w;
                if (corrSelf[xi] < powFloorC)
                    w = 0.0;    // quiet: unproven, withdraw nothing
                else if (corrDen[xi] <= 0.0)
                    w = 0.0;    // loud, no observable partner: fail closed
                else
                    w = lddecode::scheduleAlternationLicense(
                        corrNum[xi] / corrDen[xi]);
                wRow[xi] = static_cast<float>(w);
            }
        }

        // Reach gate: per-pixel cancellation weight toward one leg.
        //
        // Mismatch and amplitude are evaluated as 2-sample quadrature
        // envelopes, never instantaneous samples: |fit + neighbor| dips to
        // zero twice per carrier cycle, so a per-sample gate oscillates at
        // carrier rate.  On slanted chroma (~2 px/line drift = 180° carrier
        // rotation per line) the grammar-Opposite neighbor presents
        // SAME-SIGN chroma; the envelope gate sees a constant 2A mismatch
        // and stands the cancellation down consistently, where the
        // per-sample gate cancelled real chroma in carrier-rate bursts —
        // a checkerboard manufactured inside combedCarrier itself.
        auto reachGate = [&](int xi, const FitLeg &leg,
                             double *lumaGateOut = nullptr,
                             double *carrGateOut = nullptr) {
            if (!leg.present)
                return 0.0;
            // No per-pixel eligibility test here: the fit now exists at every
            // grammar-locked sample (schedule doubt downweights the solve, it
            // no longer amputates the estimate), and participation is applied
            // once, at the weight/emit stage — never twice.

            const double lumaDiffIRE =
                std::fabs(static_cast<double>(floorRow[xi]) -
                          static_cast<double>(leg.lumaFloor[xi])) * invIreScale;

            const int xj = (xi + 1 < width) ? xi + 1
                         : (xi > 0 ? xi - 1 : xi);
            const double c0 = static_cast<double>(fitRow[xi]);
            const double c1 = static_cast<double>(fitRow[xj]);
            const double n0 = static_cast<double>(leg.fit[xi]);
            const double n1 = static_cast<double>(leg.fit[xj]);

            // Direct sqrt, not std::hypot: these are bounded carrier-fit
            // samples (IRE-scale, no overflow risk), so hypot's IEEE
            // over/underflow guarding is pure cost in a 6-call-per-pixel
            // inner loop.
            const double sum0 = c0 + n0, sum1 = c1 + n1;
            const double carrierMismatchIRE =
                std::sqrt(sum0 * sum0 + sum1 * sum1) * invIreScale;
            const double carrierAmpIRE = 0.5 *
                (std::sqrt(c0 * c0 + c1 * c1) +
                 std::sqrt(n0 * n0 + n1 * n1)) * invIreScale;

            double lumaGate = softReachGate(lumaDiffIRE, 3.0, 10.0);

            const double carrierSoftIRE = std::max(3.0, 0.25 * carrierAmpIRE);
            const double carrierHardIRE = std::max(10.0, 0.80 * carrierAmpIRE);
            double carrierGate =
                softReachGate(carrierMismatchIRE,
                              carrierSoftIRE,
                              carrierHardIRE);

            // A/B switch (LD_P2_CARRGATE=0) for the carrier-mismatch gate.
            // It tests |fit + neighbour|, which is small only when the partner
            // INVERTS.  Image-locked alien luma matches its partner instead,
            // so the mismatch reads ~2A and the gate closes -- measured at
            // 0.001 on the cube face, i.e. fully shut on 75% of loud pixels,
            // which is exactly the content 0.5*(fit - neighbour) resolves
            // correctly.  lumaGate remains as the independent, carrier-free
            // content-break guard when this is disabled.
            static const bool carrGateEnabled = []{
                const char *s = std::getenv("LD_P2_CARRGATE");
                return !(s && s[0] == '0');
            }();
            if (!carrGateEnabled)
                carrierGate = 1.0;

            if (lumaGateOut) *lumaGateOut = lumaGate;
            if (carrGateOut) *carrGateOut = carrierGate;
            return lumaGate * carrierGate;
        };

        // -----------------------------------------------------------------
        // Fused reach-gate sweep: Pass A materializes each leg's raw gates
        // (the stencil below reads ±2 horizontal neighbors), Pass B applies
        // the 5-tap smooth, the decision blend, the license, and emits.
        // -----------------------------------------------------------------

        // Pass A: raw gates per leg.
        for (int xi = 0; xi < width; ++xi) {
            for (int k = 0; k < kNLegs; ++k)
                legs[k].wRaw[xi] = reachGate(xi, legs[k]);
        }

        // Pass B: inline 5-tap smooth + decision blend + combRow output.
        // The smooth kernel is [1,2,3,2,1] (sum = 9 in the interior).
        // At edges, the kernel is clamped and the divisor adjusts.
        constexpr double kWeights[5] = {1.0, 2.0, 3.0, 2.0, 1.0};

        for (int xi = 0; xi < width; ++xi) {
            // A refused center sample owns no carrier column.  Keep the DQ
            // explicit at the interline publication boundary as well as in
            // the fit, so smoothing of neighboring reach gates cannot give
            // it a route back into video.  Participation is graded now: the
            // emit below scales by it, so a decisively proven-illegal centre
            // (participation 0) still publishes no carrier, while doubt short
            // of proof fades the carrier's authority instead of flipping it
            // at one line's verdict bit.
            const double centerParticipation =
                static_cast<double>(eligRow[xi]);
            if (centerParticipation <= 0.0) {
                combRow[xi] = 0.0f;
                continue;
            }

            // Ownership weight for the un-cancelled remainder (see the emit
            // block below): the OPERAND schedule-compatibility license.  The
            // confiscation operand is this line's carrierFit; the license
            // tests the operand itself against each certified-Opposite
            // leg's fit.  On-schedule chroma inverts (signed corr -> -1:
            // licensed, keep the confiscation, microscopic runs included); a
            // fit that absorbed alien luma MATCHES its partner where the
            // schedule demands inversion (cube grid: corr -> +1, license 0,
            // energy stays in Y).  A quiet operand licenses at 1
            // (confiscating ~nothing is harmless); a loud operand with no
            // observable partner fails closed.
            //
            // The window is one full 4-sample carrier cycle — the same
            // geometry as the residual-carrier operand license in produceY —
            // not a 2-sample pair.  The 2-sample dot product was measured
            // moving 0.145 per line against a license band only 0.4 wide,
            // parking 73% of loud cube-face pixels in the ±0.5 dead zone:
            // an aperture problem, not an evidence problem.
            double ownedFallback;
            {
                // Centered 5-tap, half-weight ends (integer centroid);
                // total weight stays 4 so the cycle-energy floor keeps
                // its meaning.
                double e0 = 0.0;
                for (int k = -2; k <= 2; ++k) {
                    const int j = std::clamp(xi + k, 0, width - 1);
                    const double w = (k == -2 || k == 2) ? 0.5 : 1.0;
                    const double c = static_cast<double>(fitRow[j]);
                    e0 += w * c * c;
                }
                const double ampFloor = 3.0 * irescale;      // 3 IRE envelope
                // Energy of a carrier of envelope A over one full cycle is
                // 2*A^2 (cos^2 + sin^2 pairs), so the 3 IRE envelope floor
                // keeps its meaning on the 4-sample window.
                const double eFloor = 2.0 * ampFloor * ampFloor;
                if (e0 < eFloor) {
                    ownedFallback = 1.0;   // harmless confiscation
                } else {
                    // Integrate the legs into ONE energy-weighted signed
                    // correlation rather than taking the most-legal single
                    // observation.  With a one-partner roster the two forms
                    // coincide; with three noisy legs, min() lets any single
                    // draw below -0.5 license the confiscation (measured:
                    // open rate doubled on the cube face), which inverts the
                    // proof standard — confiscation requires the observations
                    // TOGETHER to prove carrier-law inversion.  Real chroma
                    // inverts against every Opposite leg, so integration
                    // costs it nothing; a decorrelated alien fit cannot
                    // manufacture a negative aggregate from one lucky leg.
                    double dotSum = 0.0, normSum = 0.0;
                    for (int k = 0; k < kNLegs; ++k) {
                        if (!legs[k].present)
                            continue;
                        const float *pf = legs[k].fit;
                        double dot = 0.0, eP = 0.0;
                        for (int j = -2; j <= 2; ++j) {
                            const int jj = std::clamp(xi + j, 0, width - 1);
                            const double w = (j == -2 || j == 2) ? 0.5 : 1.0;
                            const double c = static_cast<double>(fitRow[jj]);
                            const double n = static_cast<double>(pf[jj]);
                            dot += w * c * n;
                            eP += w * n * n;
                        }
                        if (eP < eFloor)
                            continue;
                        dotSum += dot;
                        normSum += std::sqrt(e0 * eP);
                    }
                    if (normSum > 0.0) {
                        const double sc = dotSum / normSum;
                        ownedFallback =
                            lddecode::scheduleAlternationLicense(sc);
                    } else {
                        ownedFallback = 0.0;  // loud, unobservable: fail closed
                    }
                }

                // Ownership FLOOR (LDCD_OWNERSHIP_FLOOR, default 0 = no
                // change).  A single scalar interpolating the standdown
                // remainder between the conservative promoted product (floor
                // 0: unproven carrier is left in Y) and a more aggressive
                // withdrawal (floor 1: the fit's full standdown remainder is
                // always confiscated, matching native's unconditional
                // withdrawal in that band).  Measured: promoted manufactures
                // ~2x less illegal energy than native at every Pass-2
                // strength band, so this is a swept aggression/cleanliness
                // trade, not a correctness fix -- it moves along the same
                // fast-varying-gain amplitude-modulation defect the encoder
                // bandwidth law names, it does not resolve it.
                static const double ownershipFloor = []{
                    const char *s = std::getenv("LDCD_OWNERSHIP_FLOOR");
                    const double v = s ? std::atof(s) : 0.0;
                    return std::clamp(v, 0.0, 1.0);
                }();
                if (ownershipFloor > 0.0)
                    ownedFallback = std::max(ownedFallback, ownershipFloor);
            }

            // Inline 5-tap smooth of each leg's raw gates.
            double legW[kNLegs];
            {
                double sumW = 0.0;
                double sums[kNLegs] = {0.0, 0.0, 0.0, 0.0};
                for (int dx = -2; dx <= 2; ++dx) {
                    const int xx = std::clamp(xi + dx, 0, width - 1);
                    const double w = kWeights[dx + 2];
                    sumW += w;
                    for (int k = 0; k < kNLegs; ++k)
                        sums[k] += w * legs[k].wRaw[xx];
                }

                // Decision blend by local registered amplitude: raw gates at
                // low amplitude (cheap, exact), smoothed gates where carrier
                // is loud (a per-pixel gate step on loud carrier is itself a
                // rendered artifact).
                const int xm = std::max(0, xi - 1);
                const int xp = std::min(width - 1, xi + 1);
                double amp = 0.0;
                auto includeRegisteredAmp = [&](const float *fit,
                                                const float *elig,
                                                int x) {
                    if (fit && elig && elig[x] > 0.0f)
                        amp = std::max(amp,
                                       std::fabs(static_cast<double>(fit[x])));
                };
                includeRegisteredAmp(fitRow, eligRow, xi);
                includeRegisteredAmp(fitRow, eligRow, xm);
                includeRegisteredAmp(fitRow, eligRow, xp);
                for (int k = 0; k < kNLegs; ++k) {
                    if (!legs[k].present)
                        continue;
                    includeRegisteredAmp(legs[k].fit, legs[k].elig, xi);
                    includeRegisteredAmp(legs[k].fit, legs[k].elig, xm);
                    includeRegisteredAmp(legs[k].fit, legs[k].elig, xp);
                }
                const double blend =
                    smoothStep01((amp * invIreScale - 8.0) / 10.0);

                // The leg's graded participation multiplies its weight, so
                // the horizontal smoother cannot resurrect a proven-illegal
                // partner sample (x0 stays 0), and partial doubt fades the
                // leg instead of flipping it at the verdict threshold.
                for (int k = 0; k < kNLegs; ++k) {
                    legW[k] = legs[k].present
                        ? (legs[k].wRaw[xi] * (1.0 - blend) +
                              (sums[k] / sumW) * blend) *
                          static_cast<double>(legs[k].elig[xi])
                        : 0.0;
                }
            }

            double wSum = 0.0;
            for (int k = 0; k < kNLegs; ++k)
                wSum += legW[k];

            if (wSum > 1e-9) {
                double neighborFit = 0.0;
                for (int k = 0; k < kNLegs; ++k) {
                    if (legW[k] > 0.0)
                        neighborFit += legW[k] *
                            static_cast<double>(legs[k].fit[xi]);
                }
                neighborFit /= wSum;

                // Strength-scaled cancellation, not a hard switch.  For
                // genuinely inverting chroma 0.5*(C - (-C)) = C at any
                // strength, so this is amplitude-neutral where the comb is
                // right; where the neighbor does NOT invert (slanted chroma,
                // pedestal) a partial gate now yields partial retention
                // instead of full cancellation of real signal.
                const double cancelled =
                    0.5 * (static_cast<double>(fitRow[xi]) - neighborFit);
                const double strength = std::min(1.0, wSum);

                // Ownership rule: the un-cancelled remainder is carrier this
                // stage would confiscate from Y.  It may only be confiscated
                // where carrier is owned.  The license is nonzero ONLY where
                // an observed on-schedule alternation proves the operand
                // behaves as carrier; it is zero through the entire ambiguous
                // middle (so genuine/slanted chroma is untouched).  Withdraw
                // only the standdown remainder, never the active
                // cancellation.
                //
                // Schedule participation joins the LICENCE on that remainder,
                // for the same reason and on the same term.  It must NOT
                // scale the cancelled term: 0.5*(fit - neighbour) is
                // self-proving by construction -- it yields C where the
                // partner inverts (real chroma, correctly kept as carrier)
                // and 0 where the partner matches (image-locked alien,
                // correctly left in Y).  Scaling that by doubt defeats the
                // separation the comb exists to perform, and on
                // alien-dominated mixed content (the cube face) it withholds
                // the very chroma the cancellation had already isolated.
                combRow[xi] = static_cast<float>(
                    static_cast<double>(fitRow[xi]) * (1.0 - strength) *
                        ownedFallback * centerParticipation +
                    cancelled * strength);
            } else {
                // Reach fully closed: the whole sample is un-cancelled carrier.
                // Confiscate it only where carrier is owned.
                combRow[xi] = static_cast<float>(
                    centerParticipation *
                    static_cast<double>(fitRow[xi]) * ownedFallback);
            }
        }
    }
    } // --luma-witness carrier-retracted model

    // NOTE: the encoder bandwidth law is imposed ONCE, on the native
    // carrierFit at the Pass-1 model boundary.  It is deliberately NOT
    // re-applied to combedCarrier here.  A forward FIR is not a projection, so
    // a second application would attenuate legal chroma near the top of the
    // encoder's own passband (measured 4.5 dB of extra loss at 1.3 MHz for a
    // third pass) and leave it in Y.
    //
    // That does leave a real, separate defect on the promoted product: the
    // emit scales the fit by per-pixel factors (strength x ownedFallback x
    // participation), and a bandlimited carrier times a fast-varying gain is
    // amplitude modulation, which manufactures out-of-band sidebands.  The
    // cure for that is to stop scaling a carrier-band signal by a
    // carrier-rate control, not to filter the damage afterwards.

    // ---------------------------------------------------------------
    // Final publication: retracted Y derives from the promoted carrier model,
    // not from the workprint fit. flatFloor has already served Pass 2 and has
    // no downstream consumer.
    // ---------------------------------------------------------------
    // Which carrier model the published view withdraws:
    //
    // PEDESTAL LAW (user, 2026-07-26): every election candidate rides the
    // SAME BASIS — raw minus a FULL, COMMITTED carrier subtraction.  The
    // candidates differ by WHICH carrier model they subtract, never by HOW
    // MUCH of it.  The previous default (raw - w·carrierFit) scaled the
    // subtraction by the corroboration, so the candidate's baseline floated
    // by (1-w)·fit relative to the roster — at w→0 it stood a full legal-
    // carrier lobe above every other candidate and won bright pixels on
    // that pedestal alone (the >100 IRE speck population, ~40% of it
    // schedule-LEGAL kept energy).  Feasibility cannot catch it: |raw - Y|
    // ~= 0 at passthrough is maximally feasible.  Scaling the withdrawal
    // was also a second confidence hedge on a promoted model: the fit is
    // already schedule-aware through the graded participation weights, and
    // since P14 it is hull-bounded, so full subtraction cannot over-
    // withdraw beyond what the legal Y floors permit.  Corroboration stays
    // EVIDENCE (admission/scoring); it is no longer a subtraction gain.
    //
    //   native (default) — raw - carrierFit: the full hull-bounded lawful
    //       fit.  In-band luma the schedule refused never entered the fit
    //       (graded participation), so the grid detail survives HERE, in
    //       the model, not via a downstream haircut.
    //   corr (LDCD_RETRACTED_SOURCE=corr) — raw - w·carrierFit: the retired
    //       floating-pedestal product, kept as an A/B escape only.
    //   promoted (LDCD_RETRACTED_SOURCE=promoted) — raw - combedCarrier:
    //       Pass 2's per-pixel confiscation policy (measured: the most
    //       lawful share of the old products, 71%, but rendered as a
    //       per-pixel patchwork and nearly inert on the cube face).
    //
    // One default product; the env selections are A/B escapes.  Every
    // downstream consumer — centre and vertical neighbours alike — sees the
    // same selection, so the election is fed a consistent product rather
    // than a mixture.
    //   anchor (DEFAULT, promoted 2026-07-28 on render judgement: A/C
    //       great, D good, B soft) -- the certified-carrier ladder below:
    //       covered sample -> (def-spare)/2 as itself; comp sample ->
    //       certified-luma vertical comb, fact-only; locked 1D
    //       OBSERVATION where no certified product exists. The fit does
    //       not appear on this ladder at all (2026-08-08): the ladder's
    //       product is published as FactBacked and read as a comb source,
    //       so every rung on it must be fact or observation, never an
    //       estimate. On material without dG coverage every sample takes
    //       the observation and the frame is combed by the user's
    //       selected comb, which is what it was always entitled to.
    //   native (LDCD_RETRACTED_SOURCE=native) -- raw - carrierFit
    //       everywhere: the pre-anchor default, kept as the A/B escape.
    // This selector belongs solely to the optional witness view. It has no
    // authority over the certified carrier plane.
    const int retractedSource = ldcdRetractedSourceMode();

    // Certified-luma vertical comb for the COMP lines in anchor mode (user,
    // 2026-07-28: the reform used the certified CARRIER on covered lines and
    // left the comps with the fit -- "I wonder if the reform of retracted
    // didn't make full use of the 'real luma' we have available from the
    // certified fields"). The conservation pair supplies both halves: on
    // covered lines Lhat = raw - exact is certified luma, CARRIER-FREE, so
    // a comp line combed against its two certified brackets has no phase
    // relation to honour at all:
    //     R = raw_comp - (Lhat_up + Lhat_dn)/2 = C_comp + (L_comp - L_avg)
    // The carrier-band filter (unity at fSC and +-1.3 MHz, zero at DC/2fSC)
    // then takes the carrier out of R. Every operand is fact, so the rung is
    // fact-only: no fit, no hedge, and no product at all where the brackets
    // are not both certified.
    //
    // WHAT CANCELS, EXACTLY (corrected 2026-08-08). The bracket mean removes
    // shared luma to FIRST order -- a symmetric two-tap mean is exact for
    // luma that is vertically constant or vertically linear at that column.
    // What it leaves is the luma's vertical SECOND difference, and on a
    // diagonal that is not a small residue: shear an edge by s samples per
    // line and the survivor scales as s^2, a lateral doublet sitting on the
    // edge with real energy in the carrier band. The FIR then claims it as
    // carrier and it is retracted out of Y -- the aliasing along diagonals.
    // The old text here claimed the cancellation was exact and the claim
    // was wrong; it held only for vertical structure.
    //
    // The bracket is therefore AIMED rather than taken straight down the
    // column -- see the registration below. Both bracket rows are certified
    // carrier-free luma, so the aim is measured fact against fact, which is
    // what entitles it to run inside a fact-only rung.
    std::vector<double> certComp(
        width, std::numeric_limits<double>::quiet_NaN());

    // ANTICIPATED RUNG -- REMOVED 2026-08-08 (author: "I was only ever
    // interested in the sync tone"). It stood the previous cover's certified
    // luma up as a carrier reference on uncovered frames, chained forward to
    // a horizon of 8 frames, and merged it with the fit under an out-of-band
    // alpha.
    //
    // The licensed duty was always the PHASE CURVE: carrier phase is
    // raster-anchored and rides the cycle, so it survives motion. That duty
    // is applyToneToFit(), which remains on and is untouched by this removal.
    // The WAVEFORM is content and does not survive motion -- the chained
    // reference holds objects at their old positions, the displaced-edge
    // mismatch's in-band share passes the FIR as carrier, and the rung's
    // out-of-band alpha is measured blind to exactly that class.
    //
    // Measured before it was defaulted off (Wolf 359 ship, 2026-08-07):
    // 23-47 chroma codes of paint at object edges, uncovered frames only,
    // removed by standing the rung down. The RLAD census agreed -- the
    // rung's departure from the fit read as STANDING (image-locked luma),
    // not carrier.
    //
    // Its antRefLuma chain went with it. That chain snapshotted and copied a
    // frame-sized float plane on EVERY certified frame, outside the witness
    // block, to feed only this default-off consumer.

    // Retained record from the removed LDCD_DUMP_RLAD dump
    //
    // The user's direction (2026-07-29): the certified fields' exact shade is
    // the truth reference for diagnosing the comp-line estimators in thorny
    // areas where carrier and HF luma resist separation.

    // Escape for A/B only, default ON:
    //   LDCD_PHASE_SNAP=0 -- disable the working-space snap
    // LDCD_OOB_ALPHA is retired with the fit merge whose weight it set; see
    // the comp rung below.
    const bool phaseSnap = ldcdPhaseSnapOn();
    // The phase reference was validated at >=2 IRE (vertical coherence
    // 0.907); below that its phase is noise and snapping toward it costs
    // Y (measured: fading from 0.25 IRE gave back the alpha reform's D2
    // win).
    constexpr double kSnapAmpTauIRE = 3.0;
    constexpr double kSnapAmpMinIRE = 1.0;

    static const bool anchor1D = []{
        const char *e = std::getenv("LDCD_ANCHOR_1D");
        return !(e && std::atoi(e) == 0);
    }();
    const bool publishCertified = anchor1D && frameHasExactCoverage();
    if (publishCertified) {
        anchored1DSource_flat.assign(
            static_cast<size_t>(demodLines) * demodWidth, 0.0);
    }

    for (int line = firstLine; line < lastLine; ++line) {
        const quint16 *rawLine =
            rawbuffer.data() + static_cast<size_t>(line) * videoParameters.fieldWidth;
        const float *fitRowPub = carrierFit_flat.data()
                                 + static_cast<size_t>(line) * demodWidth;
        const float *combRowPub = combedCarrier_flat.data()
                                  + static_cast<size_t>(line) * demodWidth;
        const float *wRowPub = carrierCorroboration_flat.data()
                               + static_cast<size_t>(line) * demodWidth;
        float *retractedRow = configuration.lumaWitness
            ? carrierRetracted_flat.data()
                  + static_cast<size_t>(line) * demodWidth
            : nullptr;
        double *anchoredRow = publishCertified
            ? anchored1DSource_flat.data()
                  + static_cast<size_t>(line) * demodWidth
            : nullptr;

        const float *exRowPub = exactCarrierRow(line);
        const float *exU = (line - 1 >= firstLine)
            ? exactCarrierRow(line - 1) : nullptr;
        const float *exD = (line + 1 < lastLine)
            ? exactCarrierRow(line + 1) : nullptr;

        {
            std::fill(certComp.begin(), certComp.end(),
                      std::numeric_limits<double>::quiet_NaN());
            // PRECONDITION, ASKED CHEAPLY (2026-08-08). exactCarrierRow
            // returns a valid ALL-NaN row rather than nullptr wherever the
            // twin merge left no facts, so `exU && exD` is true on every line
            // of an uncovered frame AND on every def line of a covered one --
            // the two populations whose brackets carry nothing. The rung then
            // ran at full cost to publish nothing: the registration search
            // sampled seven shifts of a seven-tap SAD per column against NaN,
            // R stayed NaN, the FIR declined every sample, and the snap
            // refused every window, leaving certComp at the NaN filled above.
            //
            // certifiedDefLine is the memoised per-line verdict for "does
            // this row carry any certified sample", so this asks the block's
            // own stated precondition -- it exists only where both brackets
            // are certified, i.e. the comp lines of a covered frame -- for
            // the price of two cached reads. Behaviour-identical: every write
            // the skipped path could make is to a local, or to certComp and
            // certRegistration, both already sitting at their absent values.
            if (exU && exD &&
                certifiedDefLine(line - 1) && certifiedDefLine(line + 1)) {
                const quint16 *rawU = rawbuffer.data()
                    + static_cast<size_t>(line - 1) * videoParameters.fieldWidth;
                const quint16 *rawD = rawbuffer.data()
                    + static_cast<size_t>(line + 1) * videoParameters.fieldWidth;
                // The two certified luma rows. Carrier-free by conservation,
                // so everything measured on them below is fact.
                std::vector<double> lU(width,
                    std::numeric_limits<double>::quiet_NaN());
                std::vector<double> lD(width,
                    std::numeric_limits<double>::quiet_NaN());
                for (int xi = 0; xi < width; ++xi) {
                    const float eu = exU[left + xi];
                    const float ed = exD[left + xi];
                    if (std::isfinite(eu))
                        lU[xi] = (double)rawU[left + xi] - (double)eu;
                    if (std::isfinite(ed))
                        lD[xi] = (double)rawD[left + xi] - (double)ed;
                }

                // THE AIM (author, 2026-08-08: "I could survive the comp
                // getting its own comb if it respected the innovations
                // around diagonals that we had accomplished"). Frame B's
                // registration cannot be CALLED here -- it runs in split2D,
                // downstream, and its own input on these lines is this
                // rung's output -- so the innovation is lifted rather than
                // the caller: same search span, same structure-anchored
                // premise, same 8% margin, and the same sign convention, so
                // the number published is one Frame B can use unchanged.
                //
                // Searched on certified luma rather than on IQ. The metric
                // is a windowed SAD, not the magnitude of a windowed sum:
                // in the complex IQ domain a vector sum is the natural
                // aggregate, but on real luma signed deviations cancel
                // within the window and a symmetric edge would read as a
                // perfect fit at every shift.
                //
                // The margin is what keeps this from being a lever. s = 0
                // is the default and a non-zero aim must beat it by 8% to
                // displace it, so flat and vertical content register at
                // zero -- exactly the old conduct -- and only genuine
                // lateral advance moves the bracket.
                // SAMPLE WIDER THAN YOU ADOPT. The search runs to +-3 but
                // only +-2 may be adopted, so that every adoptable shift has
                // neighbours on both sides and can be REQUIRED to be a
                // genuine interior minimum (below). Without the extra rung
                // the limit shifts cannot be tested at all, and they collect
                // the search's failures: measured at 20% of columns pinned
                // to |s| = 2 when the test could not be applied.
                constexpr int kRegMax = 2;            // adoptable
                constexpr int kRegSearch = 3;         // sampled
                constexpr double kRegMargin = 1.08;   // Frame B's constant
                static const double kRegWin[7] =
                    { 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5 };
                constexpr double kRegWinSum = 6.0;
                auto lumaAt = [&](const std::vector<double> &row,
                                  int j) -> double {
                    return row[std::clamp(j, 0, width - 1)];
                };
                std::vector<int> reg(width, 0);
                qint8 *regRow = certRegistration_line(line);
                for (int xi = 0; xi < width; ++xi) {
                    double dev[2 * kRegSearch + 1];
                    bool have[2 * kRegSearch + 1];
                    for (int si = 0; si <= 2 * kRegSearch; ++si) {
                        const int s = si - kRegSearch;
                        double acc = 0.0;
                        bool ok = true;
                        for (int k = -3; k <= 3 && ok; ++k) {
                            const double a = lumaAt(lU, xi + k - s);
                            const double b = lumaAt(lD, xi + k + s);
                            if (!std::isfinite(a) || !std::isfinite(b))
                                ok = false;
                            else
                                acc += kRegWin[k + 3] * std::fabs(a - b);
                        }
                        have[si] = ok;
                        dev[si] = ok ? acc / kRegWinSum : 0.0;
                    }
                    if (!have[kRegSearch]) continue; // no s=0 baseline: no fact
                    // IDENTIFIABILITY FLOOR. An aim is a claim about where a
                    // feature went, and it is only answerable when the two
                    // certified rows actually DISAGREE where they are being
                    // read. Below the floor the rows already match at s = 0:
                    // there is nothing for a shift to explain, every shift
                    // scores the same to within noise, and letting the 8%
                    // margin pick among them is reading a coin toss as
                    // geometry. Same shape as the phase snap's amplitude
                    // floor -- a clamp on an impossible, not a tuning lever
                    // -- and set at the level that measurement validated as
                    // the point below which the quantity is noise.
                    //
                    // Measured cost of NOT having it (BO-beta frame 407,
                    // covered): 70% of columns registered non-zero and 33%
                    // pinned at the |s| = 2 search limit, which is a
                    // saturating search, not a picture.
                    constexpr double kRegIdentifyIRE = 1.0;
                    if (dev[kRegSearch] * invIreScale < kRegIdentifyIRE) {
                        reg[xi] = 0;
                        if (regRow) regRow[xi] = 0;
                        continue;
                    }
                    // INTERIOR MINIMUM. A shift is adopted only if it is a
                    // demonstrated minimum -- strictly better than the
                    // positions either side of it -- as well as beating s = 0
                    // by the margin. A diagonal produces a real trough; the
                    // near-ties that dense fine texture produces do not, and
                    // this is what separates them. It also retires the
                    // pinned-at-the-limit population by construction, because
                    // a shift at the edge of the ADOPTABLE range still has
                    // both neighbours inside the SAMPLED range.
                    int bestS = 0;
                    double bestDev = dev[kRegSearch] / kRegMargin;
                    for (int si = 0; si <= 2 * kRegSearch; ++si) {
                        const int s = si - kRegSearch;
                        if (s == 0 || std::abs(s) > kRegMax) continue;
                        if (!have[si] || !have[si - 1] || !have[si + 1])
                            continue;
                        if (dev[si] >= dev[si - 1] || dev[si] >= dev[si + 1])
                            continue;
                        if (dev[si] < bestDev) {
                            bestDev = dev[si];
                            bestS = s;
                        }
                    }
                    reg[xi] = bestS;
                    if (regRow) regRow[xi] = static_cast<qint8>(bestS);
                }

                // R = comp composite minus the AIMED certified-luma bracket
                // mean. At s = 0 this is the straight vertical mean it has
                // always been; where the structure runs across the raster
                // the two legs are read at the positions the same feature
                // actually occupies, so the shared luma cancels on diagonals
                // as well as on verticals -- which is the whole claim the
                // old comment made and could not keep.
                std::vector<double> R(width,
                    std::numeric_limits<double>::quiet_NaN());
                for (int xi = 0; xi < width; ++xi) {
                    const int s = reg[xi];
                    const double lu = lumaAt(lU, xi - s);
                    const double ld = lumaAt(lD, xi + s);
                    if (!std::isfinite(lu) || !std::isfinite(ld)) continue;
                    R[xi] = (double)rawLine[left + xi] - 0.5 * (lu + ld);
                }
                // Carrier-band FIR on the certified-bracket difference.
                // Every operand is fact: raw is the observation, and the
                // bracket is raw - exact on two certified lines. This rung
                // is fact-only end to end, and where it cannot form it
                // publishes NOTHING rather than reaching for a substitute.
                //
                // THE FIT IS GONE (author, 2026-08-08: "the comp must comb
                // ONLY with the certified carrier. It boggles my mind that
                // this would involve a fit when we have certified carrier").
                // What stood here was a per-sample confidence-alpha merge
                // between this comb and the carrier FIT, floored at parity
                // and conditioned by a regional audit of the fit. Two things
                // were wrong with it, neither of them a tuning question:
                //
                //   * It laundered an estimate into the FACT CHANNEL. This
                //     product is published directly as FactBacked and read
                //     by Frame B and every 2D candidate through
                //     combSource1D_line -- the plane comb.h describes as
                //     "exact on covered lines and the certified-comb
                //     construction on comp lines". A fit share inside it is
                //     an estimate combing into a fact-bearing position,
                //     which the ratchet forbids outright.
                //   * On the failure it was hired for -- diagonals, where
                //     the un-aimed bracket mean leaves the luma's vertical
                //     second difference behind for the FIR to claim as
                //     carrier -- it delivered half a mis-aimed comb plus
                //     half a fit. That is precisely the "partial correction
                //     is the worst geometry" case Frame B's registration
                //     exists to refuse.
                //
                // The out-of-band residual |R - bp| went with it. It was
                // only ever the mixing weight, and it was structurally blind
                // to the diagonal error anyway: the false claim IS the
                // in-band share, so an out-of-band test cannot see it (the
                // anticipated rung states this same blind spot explicitly).
                // Aiming the bracket is the answer to that failure; a hedge
                // against it is not.
                constexpr double kT0 = 0.676462;
                constexpr double kT2 = -0.250000;
                constexpr double kT4 = -0.088231;
                std::vector<double> est(width,
                    std::numeric_limits<double>::quiet_NaN());
                for (int xi = 0; xi < width; ++xi) {
                    bool ok = true;
                    double taps[5];
                    static const int off[5] = { 0, -2, 2, -4, 4 };
                    for (int k = 0; k < 5 && ok; ++k) {
                        const int j = std::clamp(xi + off[k], 0, width - 1);
                        taps[k] = R[j];
                        if (!std::isfinite(taps[k])) ok = false;
                    }
                    if (!ok) continue;
                    est[xi] = kT0 * taps[0] +
                              kT2 * (taps[1] + taps[2]) +
                              kT4 * (taps[3] + taps[4]);
                }

                // Phase alignment in the 4fsc working space (user: "we
                // generally rotate all candidates in an election into the
                // same working space... the candidates in the Y election
                // need to have their phases aligned before it gets baked
                // in"). MEASURED basis (since-removed LDCD_DUMP_RLAD dump,
                // 2026-07-29): the true carrier holds phase down the strut
                // (vertical coherence 0.907 across covered lines), while the
                // fit's phase error is spatially incoherent (rms 52 deg at
                // real carrier, line-to-line ANTI-correlated) -- so the
                // certified brackets are the phase reference and the local
                // estimator keeps only its amplitude:
                //     carrier = |est| * bAlign / |bAlign|
                // per short window, where bAlign is the grammar-sign-
                // aligned bracket mean. The snap fades in with reference
                // amplitude (phase of a near-zero carrier is meaningless,
                // a clamp on an impossible) and is skipped where the
                // bracket window is incomplete.
                if (phaseSnap) {
                    std::vector<double> bAlign(width);
                    ldcdBuildCertBracketAligned(
                        carrierGrammarLine(line),
                        carrierGrammarLine(line - 1),
                        carrierGrammarLine(line + 1),
                        exU, exD, left, width, bAlign);
                    ldcdApplyPhaseSnap(est, bAlign, certComp,
                                       width, irescale,
                                       kSnapAmpMinIRE, kSnapAmpTauIRE,
                                       false);
                } else {
                    for (int xi = 0; xi < width; ++xi)
                        certComp[xi] = est[xi];
                }
            }

            // Anticipated rung: see the declaration comment above.  Built
            // per line from the chained cover's certified luma; carrier
            // stays native to this frame.
        }

        // THE SAFE RETREAT for the certified ladder (author, 2026-08-08: the
        // only gating "should be on the presence of certified"). Where no
        // certified product exists the ladder falls to the locked 1D
        // OBSERVATION -- never to the fit. This plane is published as
        // FactBacked and read as a comb source through combSource1D_line, so
        // a fit sample here is an estimate wearing a fact's name at exactly
        // the place no downstream consumer can tell the difference. That was
        // the comp merge's laundering at its SECOND site: stripping the
        // merge alone would have moved the fit from blended to whole rather
        // than removing it.
        const double *obs1D = locked1DSource_line(line);

        for (int xi = 0; xi < width; ++xi) {
            // The certified plane has exactly three rungs: measured twin
            // carrier, fact-only comp construction, observed locked 1D. It
            // is computed independently of every witness/fit selector.
            const float ex = exRowPub ? exRowPub[left + xi]
                                      : std::numeric_limits<float>::quiet_NaN();
            const double certifiedCarrier = std::isfinite(ex)
                ? static_cast<double>(ex)
                : (std::isfinite(certComp[xi])
                    ? certComp[xi]
                    : (obs1D ? obs1D[xi] : 0.0));

            if (anchoredRow)
                anchoredRow[xi] = certifiedCarrier;

            if (retractedRow) {
                // ABSTENTION, NOT A FALLBACK (user, 2026-08-09: "the cede
                // should be to some kind of 'I'm null, don't include me'").
                // Where this line has no solved carrier model the fit row
                // holds a written ZERO, and raw - 0 is the whole composite,
                // carrier included -- a checkerboard seated as a base
                // candidate with full rights over the centre and the
                // population statistics. The election is DOWNSTREAM of the
                // comb, so every candidate has the rest of the roster to
                // fall back on and none of them needs an internal retreat.
                // NaN is the honest publication: the admission below reads
                // it as absent and the seat simply goes unfilled.
                //
                // Facts are exempt -- they do not come from the fit. A
                // covered sample keeps its exact channel and a comp sample
                // its certified construction even on a line the solver
                // never reached.
                const bool haveFact =
                    std::isfinite(ex) || std::isfinite(certComp[xi]);
                if (!carrierFitLineSolved(line) && !haveFact) {
                    retractedRow[xi] =
                        std::numeric_limits<float>::quiet_NaN();
                    continue;
                }
                double witnessCarrier;
                switch (retractedSource) {
                case 1:
                    witnessCarrier = static_cast<double>(fitRowPub[xi]);
                    break;
                case 2:
                    witnessCarrier = static_cast<double>(combRowPub[xi]);
                    break;
                case 3:
                    // THE WITNESS AND THE LADDER PART COMPANY AT THE LAST
                    // RUNG (2026-08-09). Both take fact where fact exists:
                    // the exact channel, then the fact-only comp
                    // construction. Where neither exists they must NOT
                    // agree, because they answer to opposite laws.
                    //
                    // The ladder's last rung is the locked 1D OBSERVATION,
                    // and that is correct for it: its plane is published as
                    // FactBacked and read as a comb source, so every rung
                    // must be fact or observation, never an estimate.
                    //
                    // The witness is the opposite creature. It is an
                    // estimate-grade luma hypothesis BY DESIGN, and the Y
                    // election exists to adjudicate it. Its last rung is
                    // therefore the frame's own photograph: the policed,
                    // bandwidth-lawful, tone-corrected fit -- values
                    // borrowed from no other frame.
                    //
                    // Riding the ladder's rung here was a real defect, not
                    // a nicety. On an uncovered frame it made
                    // retractedY == raw - locked1DSource, which is plane 3
                    // exactly -- and plane 3 is seated only when comb is
                    // INFEASIBLE, a retreat of last resort, while plane 1
                    // is seated on admission. So --luma-witness installed
                    // 1D as a permanent base candidate with full rights
                    // over the centre, the subset and the population
                    // statistics, on every uncovered frame. Measured by the
                    // author's eye on the Borg cube: the lattice columns
                    // rendered with "1D style errors instead of good
                    // detail", and the same range without --luma-witness
                    // was sharper. It also meant the entire ~2,450-line fit
                    // solve ran on uncovered frames and was discarded at
                    // this line.
                    //
                    // "1D is the safe retreat" is a law about a RETREAT.
                    // Seating it as a contender inverts the law it is named
                    // for.
                    witnessCarrier = std::isfinite(ex)
                        ? static_cast<double>(ex)
                        : (std::isfinite(certComp[xi])
                            ? certComp[xi]
                            : static_cast<double>(fitRowPub[xi]));
                    break;
                default:
                    witnessCarrier = static_cast<double>(wRowPub[xi]) *
                                     static_cast<double>(fitRowPub[xi]);
                    break;
                }
                retractedRow[xi] = static_cast<float>(
                    static_cast<double>(rawLine[left + xi]) - witnessCarrier);
            }
        }
    }

    // TEMPORARY INSTRUMENT (LDCD_PROBE_FITTHEFT=1): held-out fit-theft
    // grading on covered def lines. Question (2026-08-06, user): "It may be
    // that some of the detail along the line is being 'retracted' as
    // carrier" -- does the deployed fit's carrier claim absorb along-line
    // luma detail (top band above the 4-sample aperture)? Column b grades
    // the parallax common-carrier claim (the proposed corrector's source)
    // against the same truth. REFEREE DISCIPLINE: run with LDCD_FACT_FIT=0
    // and LDCD_LURCH_CERT=0 or the graded fit is fact-stamped and the
    // result is a wiring diagram; the tag prints both states. Strip when
    // the question closes.
    {
        static const bool fitTheftOn = []{
            const char *e = std::getenv("LDCD_PROBE_FITTHEFT");
            return e && std::atoi(e) != 0;
        }();
        // VETTING HARNESS (author directive 2026-08-07: "no carrier
        // estimate in the current system should fail to be vetted against
        // certified. Pit the --dg-discard version of our model against
        // certified to test it and tune against it.")
        //   LDCD_THEFT_DUMP=path   normal run: append covered def lines'
        //                          exact carrier, keyed (heldSeq1, line).
        //   LDCD_THEFT_TRUTH=path  --dg-discard run: load the dump and
        //                          grade the fully fact-deprived published
        //                          fit against it (tag "dep"). The discard
        //                          raw keeps ~0.5 IRE out-of-band twin
        //                          noise the merge would have cleaned;
        //                          in-band carrier truth is unaffected.
        static const char *theftDumpPath = std::getenv("LDCD_THEFT_DUMP");
        static const char *theftTruthPath = std::getenv("LDCD_THEFT_TRUTH");
        if (fitTheftOn && theftDumpPath && frameHasExactCoverage()) {
            static std::mutex dumpMu;
            std::lock_guard<std::mutex> lk(dumpMu);
            if (FILE *f = std::fopen(theftDumpPath, "ab")) {
                for (int line = firstLine; line < lastLine; ++line) {
                    if (!certifiedDefLine(line)) continue;
                    const float *ex = exactCarrierRow(line);
                    if (!ex) continue;
                    std::int32_t hdr[3] = { (std::int32_t)heldSeq1,
                                            (std::int32_t)line,
                                            (std::int32_t)width };
                    std::fwrite(hdr, sizeof(hdr), 1, f);
                    std::vector<float> row(width);
                    for (int xi = 0; xi < width; ++xi)
                        row[xi] = ex[left + xi];
                    std::fwrite(row.data(), sizeof(float), width, f);
                }
                std::fclose(f);
            }
        }
        const bool theftDeprived =
            fitTheftOn && theftTruthPath && !frameHasExactCoverage();
        if (fitTheftOn && (frameHasExactCoverage() || theftDeprived)) {
            using TruthKey = std::pair<int, int>;
            static const std::map<TruthKey, std::vector<float>> truthMap =
                []{
                    std::map<TruthKey, std::vector<float>> m;
                    const char *p = std::getenv("LDCD_THEFT_TRUTH");
                    if (!p) return m;
                    FILE *f = std::fopen(p, "rb");
                    if (!f) return m;
                    std::int32_t hdr[3];
                    while (std::fread(hdr, sizeof(hdr), 1, f) == 1) {
                        std::vector<float> row(hdr[2]);
                        if (std::fread(row.data(), sizeof(float),
                                       row.size(), f) != row.size())
                            break;
                        m[{hdr[0], hdr[1]}] = std::move(row);
                    }
                    std::fclose(f);
                    return m;
                }();
            static const bool tagFactFit = []{
                const char *e = std::getenv("LDCD_FACT_FIT");
                return !(e && std::atoi(e) == 0);
            }();
            static const bool tagLurchCert = []{
                const char *e = std::getenv("LDCD_LURCH_CERT");
                return !(e && std::atoi(e) == 0);
            }();
            constexpr int kNB = 5;
            const double binLo[kNB] = { 0.0, 2.0, 5.0, 10.0, 20.0 };
            struct Acc {
                long n = 0, nP = 0;
                double sT = 0, sF = 0, stt = 0, sff = 0, stf = 0;
                double sfe2 = 0;
                double sTP = 0, sFP = 0, sPP = 0;
                double sttP = 0, sppP = 0, stpP = 0;
                double spe2 = 0;
            } acc[kNB];
            std::vector<double> lt(width), lf(width), lp(width);
            auto top4 = [&](const std::vector<double> &row, int xi) {
                const double m = (0.5 * row[xi - 2] + row[xi - 1] + row[xi] +
                                  row[xi + 1] + 0.5 * row[xi + 2]) / 4.0;
                return row[xi] - m;
            };
            for (int line = firstLine; line < lastLine; ++line) {
                const float *truthRow = nullptr;
                if (theftDeprived) {
                    auto it = truthMap.find({(int)heldSeq1, line});
                    if (it == truthMap.end() ||
                        (int)it->second.size() != width)
                        continue;
                    truthRow = it->second.data();
                } else {
                    if (!certifiedDefLine(line)) continue;
                    const float *ex = exactCarrierRow(line);
                    if (!ex) continue;
                    truthRow = ex + left;
                }
                const quint16 *raw = rawbuffer.data()
                    + static_cast<size_t>(line) * videoParameters.fieldWidth;
                const float *fitRow = carrierFit_flat.data()
                    + static_cast<size_t>(line) * demodWidth;
                const lddecode::CarrierAnalysisRecord *anRow =
                    carrierAnalysis_line(line);
                for (int xi = 0; xi < width; ++xi) {
                    const int h = left + xi;
                    const double e = (double)truthRow[xi];
                    lt[xi] = std::isfinite(e)
                        ? (double)raw[h] - e
                        : std::numeric_limits<double>::quiet_NaN();
                    lf[xi] = (double)raw[h] - (double)fitRow[xi];
                    lp[xi] = (anRow && anRow[xi].parallax.valid)
                        ? (double)raw[h] -
                              (double)anRow[xi].parallax.commonSample
                        : std::numeric_limits<double>::quiet_NaN();
                }
                for (int xi = 2; xi < width - 2; ++xi) {
                    bool ok = true;
                    for (int k = -2; k <= 2 && ok; ++k)
                        if (!std::isfinite(lt[xi + k])) ok = false;
                    if (!ok) continue;
                    const double t = top4(lt, xi);
                    const double f = top4(lf, xi);
                    const double tIRE = std::fabs(t) * invIreScale;
                    int b = 0;
                    for (int i = kNB - 1; i > 0; --i)
                        if (tIRE >= binLo[i]) { b = i; break; }
                    Acc &a = acc[b];
                    a.n++;
                    a.sT += std::fabs(t); a.sF += std::fabs(f);
                    a.stt += t * t; a.sff += f * f; a.stf += t * f;
                    const double fe =
                        ((double)fitRow[xi] - (double)truthRow[xi]) *
                        invIreScale;
                    a.sfe2 += fe * fe;
                    bool okP = true;
                    for (int k = -2; k <= 2 && okP; ++k)
                        if (!std::isfinite(lp[xi + k])) okP = false;
                    if (okP) {
                        const double p = top4(lp, xi);
                        a.nP++;
                        a.sTP += std::fabs(t); a.sFP += std::fabs(f);
                        a.sPP += std::fabs(p);
                        a.sttP += t * t; a.sppP += p * p; a.stpP += t * p;
                        const double pe =
                            ((double)anRow[xi].parallax.commonSample -
                             (double)truthRow[xi]) * invIreScale;
                        a.spe2 += pe * pe;
                    }
                }
            }
            auto safeDiv = [](double a2, double b2) {
                return b2 > 1e-12 ? a2 / b2 : 0.0;
            };
            std::fprintf(stderr,
                "[THEFT] frame %d mode=%s factFit=%d lurchCert=%d "
                "(retention / theftCoh / errIRE by |topTrue| bin)\n",
                (int)heldSeq1, theftDeprived ? "dep" : "cov",
                tagFactFit ? 1 : 0, tagLurchCert ? 1 : 0);
            for (int b = 0; b < kNB; ++b) {
                const Acc &a = acc[b];
                const double thF = a.stt - a.stf;
                const double thFden =
                    a.stt * (a.stt - 2.0 * a.stf + a.sff);
                const double thP = a.sttP - a.stpP;
                const double thPden =
                    a.sttP * (a.sttP - 2.0 * a.stpP + a.sppP);
                std::fprintf(stderr,
                    "[THEFT]   bin%d n=%ld fit ret %.3f coh %.2f err %.2f"
                    " | par n=%ld (%.0f%%) ret %.3f coh %.2f err %.2f"
                    " fitRetHere %.3f\n",
                    b, a.n,
                    safeDiv(a.sF, a.sT),
                    thFden > 1e-12 ? thF / std::sqrt(thFden) : 0.0,
                    a.n ? std::sqrt(a.sfe2 / a.n) : 0.0,
                    a.nP, a.n ? 100.0 * a.nP / a.n : 0.0,
                    safeDiv(a.sPP, a.sTP),
                    thPden > 1e-12 ? thP / std::sqrt(thPden) : 0.0,
                    a.nP ? std::sqrt(a.spe2 / a.nP) : 0.0,
                    safeDiv(a.sFP, a.sTP));
            }
        }
    }

    carrierRetractedValid = configuration.lumaWitness;
    if (publishCertified)
        anchoredCarrierProvenance = AnchoredCarrierProvenance::FactBacked;
}
