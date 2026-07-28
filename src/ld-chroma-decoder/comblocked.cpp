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
inline double coarseSharpLevel()
{
    static const double level = []{
        const char *s = std::getenv("LD_COARSE_SHARP");
        if (!s || !*s) return 1.0;
        const double v = std::atof(s);
        return std::isfinite(v) ? v : 0.0;
    }();
    return level;
}

// Transfer-knee stats (LDCD_PROBE_KNEE=1). Measurement only. Comb inherits
// 1D's bandwidth limits: fine-detail AMPLITUDE is stripped from 1D and its
// descendants -- a transfer-curve divergence at the top of the scale, HF
// only. This measures that curve directly: at pixels where the carrier-band
// energy is PROVEN not-carrier (carrierIllegalProof high), the retracted
// top is trustworthy luma amplitude; bin |comb top| against |retracted top|
// and the bin where the ratio departs from unity IS the knee -- the
// measured point at which comb's HF stops being trusted, which the coming
// roll-off keys to. Sign-flip fraction per bin rides along for the
// grammar-side sign fix (the taps own the sign).
// (D-S)/2 referee (LDCD_PROBE_DSREF=1): grades every carrier estimator
// against the assembler's exact-carrier side channel on covered lines.
// The confiscation ledger: error vs exact truth, split flat / detail
// (hLumaDelta >= 6 IRE), per frame. Thread-safe use: run -t 1.
struct DsRefProbe {
    static bool on()
    {
        static const bool v = std::getenv("LDCD_PROBE_DSREF") != nullptr;
        return v;
    }
    struct Est {
        long n = 0;
        double sumAbs = 0, sum = 0, sumSq = 0, maxAbs = 0;
        void add(double eIRE)
        {
            if (!std::isfinite(eIRE)) return;
            n++; sumAbs += std::fabs(eIRE); sum += eIRE; sumSq += eIRE * eIRE;
            maxAbs = std::max(maxAbs, std::fabs(eIRE));
        }
    };
    // [flat=0 / detail=1][estimator: 0=1D 1=comb 2=fit 3=retracted]
    Est e[2][4];
    long covered = 0;
    long frameIdx = 0;

    void flush()
    {
        static const char *nm[4] = { "1D", "comb", "fit", "retr" };
        if (covered > 0) {
            std::fprintf(stderr, "[DSREF f=%ld] covered px=%ld\n", frameIdx, covered);
            for (int b = 0; b < 2; ++b) {
                std::fprintf(stderr, "  %s:", b ? "detail" : "flat  ");
                for (int k = 0; k < 4; ++k) {
                    const Est &E = e[b][k];
                    if (E.n == 0) { std::fprintf(stderr, "  %s n=0", nm[k]); continue; }
                    std::fprintf(stderr,
                        "  %s |e| %.2f bias %+.2f rms %.2f max %.1f (n=%ld)",
                        nm[k], E.sumAbs / E.n, E.sum / E.n,
                        std::sqrt(E.sumSq / E.n), E.maxAbs, E.n);
                }
                std::fprintf(stderr, "\n");
            }
        }
        frameIdx++;
        covered = 0;
        for (int b = 0; b < 2; ++b) for (int k = 0; k < 4; ++k) e[b][k] = Est();
    }
};

DsRefProbe g_dsRefProbe;

struct KneeProbe {
    std::mutex mu;
    static constexpr int kBins = 10;
    // Bin edges on |retracted top| in IRE.
    static constexpr double kEdge[kBins] =
        {1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 1e9};
    long   n[kBins]       = {};
    double sumRef[kBins]  = {};
    double sumComb[kBins] = {};
    long   nFlip[kBins]   = {};

    static bool on()
    {
        static const bool v = []{
            const char *s = std::getenv("LDCD_PROBE_KNEE");
            return s && std::atoi(s) != 0;
        }();
        return v;
    }

    void sample(double refTopIRE, double combTopIRE)
    {
        const double aRef = std::fabs(refTopIRE);
        int b = 0;
        while (b < kBins - 1 && aRef >= kEdge[b]) ++b;
        std::lock_guard<std::mutex> lk(mu);
        ++n[b];
        sumRef[b]  += aRef;
        sumComb[b] += std::fabs(combTopIRE);
        // Sign flip only counts where both carry real amplitude.
        if (aRef > 1.0 && std::fabs(combTopIRE) > 1.0 &&
            refTopIRE * combTopIRE < 0.0)
            ++nFlip[b];
    }

    ~KneeProbe()
    {
        if (!on()) return;
        long total = 0;
        for (int b = 0; b < kBins; ++b) total += n[b];
        if (total <= 0) return;
        std::fprintf(stderr,
            "\n[KNEE] proven-luma pixels %ld  (|retrTop| bins, IRE)\n"
            "[KNEE]   bin        n     ref    comb   comb/ref  flip%%\n",
            total);
        for (int b = 0; b < kBins; ++b) {
            if (n[b] <= 0) continue;
            const double inv = 1.0 / (double)n[b];
            const double mRef = sumRef[b] * inv, mComb = sumComb[b] * inv;
            std::fprintf(stderr,
                "[KNEE]  %5.1f-%-5.1f %8ld %6.2f %7.2f %9.3f %6.1f\n",
                b ? kEdge[b - 1] : 0.0,
                kEdge[b] > 1e8 ? 99.0 : kEdge[b],
                n[b], mRef, mComb,
                mRef > 1e-9 ? mComb / mRef : 0.0,
                100.0 * (double)nFlip[b] * inv);
        }
    }
};

constexpr double KneeProbe::kEdge[KneeProbe::kBins];

KneeProbe g_kneeProbe;

// Off-grid leakage stats (LDCD_PROBE_OFFGRID=1). Measurement only. Stage 1
// of the fit reset: a carrier waveform belongs to the span of the grammar
// basis over every legal 4-sample window (the two quadrature waveforms span
// the whole fSC subspace; off-span = DC + 2fSC content, which no lawful
// carrier may carry). The per-sample scalar surgeries between the fit's
// basis exit and publication push the emitted carrier out of that span, so
// raw - fit carries OFF-GRID alternations the election cannot compare with
// comb's on-grid residue. This measures the off-span energy fraction per
// published carrier source, binned by window amplitude; the differential
// against comb under the identical operator is the honest read (lawful
// envelope motion leaks a little in any 4-sample window for every source).
struct OffGridProbe {
    std::mutex mu;
    // [source: 0=fit 1=comb 2=locked1D][amplitude bin: <5, 5-15, >=15 IRE]
    long   n[3][3]      = {};
    double sumOff[3][3] = {};
    double sumTot[3][3] = {};

    static bool on()
    {
        static const bool v = []{
            const char *s = std::getenv("LDCD_PROBE_OFFGRID");
            return s && std::atoi(s) != 0;
        }();
        return v;
    }

    void sample(int src, int bin, double off, double tot)
    {
        std::lock_guard<std::mutex> lk(mu);
        ++n[src][bin];
        sumOff[src][bin] += off;
        sumTot[src][bin] += tot;
    }

    // Phase half: position, not span. A wrong-phase carrier is perfectly
    // in-span; what damages the election is the fit's carrier sitting off
    // POSITION -- rotated against the physical scalar and jittering window
    // to window. Measured at strong windows only.
    long   nPh = 0;
    double sumDPhase = 0.0, sumAbsDPhase = 0.0;   // fit vs comb, radians
    long   nJit[3] = {};
    double sumJit[3] = {};                        // per-source |dphase/window|

    void phasePair(double dphase)
    {
        std::lock_guard<std::mutex> lk(mu);
        ++nPh;
        sumDPhase += dphase;
        sumAbsDPhase += std::fabs(dphase);
    }

    void jitter(int src, double dphase)
    {
        std::lock_guard<std::mutex> lk(mu);
        ++nJit[src];
        sumJit[src] += std::fabs(dphase);
    }

    // Dropout half: strong-window stats exclude exactly the failure the beam
    // sheet showed -- windows where the fit's amplitude COLLAPSES while comb
    // still carries the chroma. Split by the legality proof: at proven-
    // illegal energy the "dropout" is the fit correctly refusing what comb
    // wrongly models (virtuous); at legal carrier it is lost lock (the
    // defect the reset must cure).
    long nStrong[2] = {}, nDrop[2] = {};   // [0]=legal-ish, [1]=proven illegal

    void dropout(bool fitDropped, bool provenIllegal)
    {
        std::lock_guard<std::mutex> lk(mu);
        ++nStrong[provenIllegal ? 1 : 0];
        if (fitDropped) ++nDrop[provenIllegal ? 1 : 0];
    }

    ~OffGridProbe()
    {
        if (!on()) return;
        static const char *kSrc[3] = {"fit      ", "comb     ", "locked1D "};
        static const char *kBin[3] = {"<5 IRE ", "5-15   ", ">=15   "};
        bool any = false;
        for (int s = 0; s < 3 && !any; ++s)
            for (int b = 0; b < 3; ++b)
                if (n[s][b]) { any = true; break; }
        if (!any) return;
        std::fprintf(stderr,
            "\n[OFFGRID] off-span energy fraction of the published carrier "
            "(4-sample grammar-basis windows)\n");
        for (int s = 0; s < 3; ++s) {
            std::fprintf(stderr, "[OFFGRID] %s", kSrc[s]);
            for (int b = 0; b < 3; ++b) {
                if (n[s][b] > 0 && sumTot[s][b] > 1e-9)
                    std::fprintf(stderr, "  %s%5.2f%% (n=%ld)", kBin[b],
                                 100.0 * sumOff[s][b] / sumTot[s][b],
                                 n[s][b]);
                else
                    std::fprintf(stderr, "  %s    --", kBin[b]);
            }
            std::fprintf(stderr, "\n");
        }
        if (nPh > 0)
            std::fprintf(stderr,
                "[OFFGRID] phase(fit vs comb) strong windows: mean %+.1f deg  "
                "|mean| %.1f deg  (n=%ld)\n",
                sumDPhase / nPh * 180.0 / M_PI,
                sumAbsDPhase / nPh * 180.0 / M_PI, nPh);
        static const char *kSrc2[3] = {"fit", "comb", "locked1D"};
        for (int s = 0; s < 3; ++s)
            if (nJit[s] > 0)
                std::fprintf(stderr,
                    "[OFFGRID] phase jitter %s: %.1f deg/window (n=%ld)\n",
                    kSrc2[s], sumJit[s] / nJit[s] * 180.0 / M_PI, nJit[s]);
        if (nStrong[0] + nStrong[1] > 0)
            std::fprintf(stderr,
                "[OFFGRID] fit amplitude dropout at comb-strong windows: "
                "legal %.1f%% (n=%ld)  proven-illegal %.1f%% (n=%ld)\n",
                nStrong[0] ? 100.0 * (double)nDrop[0] / (double)nStrong[0] : 0.0,
                nStrong[0],
                nStrong[1] ? 100.0 * (double)nDrop[1] / (double)nStrong[1] : 0.0,
                nStrong[1]);
    }
};

OffGridProbe g_offGrid;

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
        const double sharpLevel = coarseSharpLevel();
        const bool buildSharp =
            configuration.lumaWitness &&
            sharpLevel > 0.0 && !lockedLumaSharp_flat.empty();
        std::vector<double> gateScratch;
        if (buildSharp && width >= 4)
            gateScratch.assign(width, 0.0);

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
            const int lastStart = width - 4;   // last legal aperture start

            // Register the even four-sample means at integer xi by averaging
            // the two half-sample apertures on either side.  Their combination
            // is the phase-balanced five-sample support
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
                            width, sharpLevel, sharp, gateScratch.data());
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

    static const int crDiagLine = []{
        const char *s = std::getenv("COARSE_RESID_DIAG_LINE");
        return s ? std::atoi(s) : -1;
    }();
    static const int crDiagC0 = []{
        const char *s = std::getenv("COARSE_RESID_DIAG_C0");
        return s ? std::atoi(s) : -1;
    }();
    static const int crDiagC1 = []{
        const char *s = std::getenv("COARSE_RESID_DIAG_C1");
        return s ? std::atoi(s) : -1;
    }();
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

        if (crDiagLine >= 0 && line == crDiagLine && crDiagC0 >= 0) {
            const int c0 = std::clamp(crDiagC0, 0, width - 1);
            const int c1 = std::clamp(
                crDiagC1 < 0 ? crDiagC0 : crDiagC1,
                c0,
                width - 1);

            std::fprintf(stderr,
                "COARSERESOPT header line rel h phase raw oldBp viewCount "
                "resLo resHi resSpreadIRE maxAbsMembershipIRE view s y4 residual "
                "residualIRE membershipDeltaIRE membershipLocalX\n");

            for (int rel = c0; rel <= c1; ++rel) {
                const int sFirst = std::max(0, rel - 3);
                const int sLast = width >= 4 ? std::min(rel, width - 4) : -1;
                double residuals[4] = {0.0, 0.0, 0.0, 0.0};
                double memberships[4] = {0.0, 0.0, 0.0, 0.0};
                double localXs[4] = {0.0, 0.0, 0.0, 0.0};
                double y4s[4] = {0.0, 0.0, 0.0, 0.0};
                int starts[4] = {0, 0, 0, 0};
                double lo = 1e300;
                double hi = -1e300;
                double maxMembership = 0.0;
                int n = 0;

                for (int s = sFirst; s <= sLast && n < 4; ++s, ++n) {
                    const double y4 = 0.25 * (
                        rawAtRel(s + 0) + rawAtRel(s + 1) +
                        rawAtRel(s + 2) + rawAtRel(s + 3));
                    const double residual = rawAtRel(rel) - y4;
                    double membership = 0.0;
                    double localX = 0.0;
                    if (s + 4 < width) {
                        membership = 0.25 *
                            (rawAtRel(s + 4) - rawAtRel(s)) * invIreScale;
                        localX = 0.5 * (static_cast<double>(s) +
                                        static_cast<double>(s + 4)) - rel;
                    }
                    starts[n] = s;
                    y4s[n] = y4;
                    residuals[n] = residual;
                    memberships[n] = membership;
                    localXs[n] = localX;
                    lo = std::min(lo, residual);
                    hi = std::max(hi, residual);
                    maxMembership = std::max(maxMembership, std::fabs(membership));
                }

                if (n == 0) {
                    std::fprintf(stderr,
                        "COARSERESOPT line=%d rel=%d h=%d phase=%d raw=%.6f "
                        "oldBp=%.6f viewCount=0 resLo=0.000000 resHi=0.000000 "
                        "resSpreadIRE=0.000000 maxAbsMembershipIRE=0.000000 "
                        "view=-1 s=-1 y4=0.000000 residual=0.000000 "
                        "residualIRE=0.000000 membershipDeltaIRE=0.000000 "
                        "membershipLocalX=0.000000\n",
                        line, rel, left + rel,
                        carrierSampleClass(line, left + rel) & 3,
                        rawAtRel(rel), baseline[rel]);
                    continue;
                }

                const double spreadIRE = (hi - lo) * invIreScale;
                for (int v = 0; v < n; ++v) {
                    std::fprintf(stderr,
                        "COARSERESOPT line=%d rel=%d h=%d phase=%d raw=%.6f "
                        "oldBp=%.6f viewCount=%d resLo=%.6f resHi=%.6f "
                        "resSpreadIRE=%.6f maxAbsMembershipIRE=%.6f "
                        "view=%d s=%d y4=%.6f residual=%.6f residualIRE=%.6f "
                        "membershipDeltaIRE=%.6f membershipLocalX=%.6f\n",
                        line, rel, left + rel,
                        carrierSampleClass(line, left + rel) & 3,
                        rawAtRel(rel), baseline[rel], n, lo, hi, spreadIRE,
                        maxMembership, v, starts[v], y4s[v], residuals[v],
                        residuals[v] * invIreScale, memberships[v], localXs[v]);
                }
            }
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

        // --- Disposable schedule-conformance instrumentation (env-gated) ---
        // Set LDCD_DUMP_CONFORMANCE=1 to print per-frame verdict statistics
        // to stderr. Zero cost when unset. Remove with the rethink.
        const bool dumpConf = std::getenv("LDCD_DUMP_CONFORMANCE") != nullptr;
        long long cBelowFloor = 0, cUnres = 0, cLegal = 0, cIllegal = 0, cNoAxis = 0;
        long long illBy[4]        = {0, 0, 0, 0}; // illegal grouped by illegal-axis-vote count (1..3)
        long long illAmp[4]       = {0, 0, 0, 0}; // illegal grouped by per-sample RMS IRE bucket
        long long usableAxisHist[4] = {0, 0, 0, 0};
        long long strongTot = 0, strongLegal = 0, strongIllegal = 0, strongUnres = 0;
        long long thirdTot[3] = {0, 0, 0}, thirdIll[3] = {0, 0, 0};

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
                if (dumpConf)
                    cNoAxis += width;
                continue;
            }

            for (int rel = 0; rel < width; ++rel) {
                const int w0 = std::clamp(rel, 0, width - 4);
                double e0 = 0.0;
                for (int k = 0; k < 4; ++k)
                    e0 += bp0[w0 + k] * bp0[w0 + k];
                if (e0 < energyFloor) {
                    if (dumpConf)
                        ++cBelowFloor;
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
                    for (int k = 0; k < 4; ++k) {
                        dot += bp0[w0 + k] * bpP[w0 + k];
                        eP  += bpP[w0 + k] * bpP[w0 + k];
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

                if (dumpConf) {
                    const double rmsIRE = std::sqrt(e0 * 0.25) * invIreScale;
                    const int third = std::clamp(
                        (line - first) * 3 / std::max(1, last - first), 0, 2);
                    const bool strong = rmsIRE >= 10.0;
                    ++thirdTot[third];
                    ++usableAxisHist[std::clamp(usableAxes, 0, 3)];
                    if (strong)
                        ++strongTot;
                    const auto verdict = analysis[rel].scheduleConformance;
                    if (verdict == lddecode::CarrierScheduleConformance::LegalCarrier) {
                        ++cLegal;
                        if (strong)
                            ++strongLegal;
                    } else if (verdict ==
                               lddecode::CarrierScheduleConformance::ScheduleIllegal) {
                        ++cIllegal;
                        ++thirdIll[third];
                        if (strong)
                            ++strongIllegal;
                        ++illBy[std::clamp(illegalAxisVotes, 0, 3)];
                        const int bkt = rmsIRE < 5.0 ? 0
                                      : rmsIRE < 10.0 ? 1
                                      : rmsIRE < 20.0 ? 2 : 3;
                        ++illAmp[bkt];
                    } else {
                        ++cUnres;
                        if (strong)
                            ++strongUnres;
                    }
                }
            }
        }

        if (dumpConf) {
            auto pct = [](long long a, long long b) {
                return b > 0 ? 100.0 * static_cast<double>(a)
                                     / static_cast<double>(b)
                             : 0.0;
            };
            const long long tested = cLegal + cIllegal + cUnres;
            std::fprintf(stderr,
                "[CONF] lines=%d..%d tested=%lld legal=%lld(%.1f%%) "
                "illegal=%lld(%.1f%%) unres=%lld(%.1f%%) belowFloor=%lld noAxisPix=%lld\n",
                first, last, tested,
                cLegal, pct(cLegal, tested),
                cIllegal, pct(cIllegal, tested),
                cUnres, pct(cUnres, tested),
                cBelowFloor, cNoAxis);
            std::fprintf(stderr,
                "[CONF] illegal-by-axisvotes 1=%lld 2=%lld 3=%lld | "
                "usableAxisHist a1=%lld a2=%lld a3=%lld\n",
                illBy[1], illBy[2], illBy[3],
                usableAxisHist[1], usableAxisHist[2], usableAxisHist[3]);
            std::fprintf(stderr,
                "[CONF] illegal-by-amp(IRE) <5=%lld 5-10=%lld 10-20=%lld >=20=%lld\n",
                illAmp[0], illAmp[1], illAmp[2], illAmp[3]);
            std::fprintf(stderr,
                "[CONF] strong(>=10IRE) tot=%lld legal=%lld(%.1f%%) "
                "illegal=%lld(%.1f%%) unres=%lld(%.1f%%)\n",
                strongTot,
                strongLegal, pct(strongLegal, strongTot),
                strongIllegal, pct(strongIllegal, strongTot),
                strongUnres, pct(strongUnres, strongTot));
            std::fprintf(stderr,
                "[CONF] illegal-by-third top=%.1f%% mid=%.1f%% bot=%.1f%%\n",
                pct(thirdIll[0], thirdTot[0]),
                pct(thirdIll[1], thirdTot[1]),
                pct(thirdIll[2], thirdTot[2]));
        }
    }

    // --- Disposable 1D-fingerprint dump (env-gated). The three orthogonal
    // views of "luma entered the bandpass", per pixel along a scanline:
    //   incoh = sourceMinusWideIRE  (horizontal: source minus wide coherent fit)
    //   lurch = maxAbsMembershipIRE (carrier-free luma movement through aperture)
    //   conf  = carrierConformance  (interline: -1 inverts like carrier,
    //                                +1 matches where schedule demands inversion
    //                                = image-locked luma leak)
    // Enable with LDCD_DUMP_FP_L (frame line) and LDCD_DUMP_FP_C0/C1 (active
    // column range = h - left). Run -t 1. Zero cost when unset.
    static const int fpLine = []{ const char *s = std::getenv("LDCD_DUMP_FP_L"); return s ? std::atoi(s) : -1; }();
    static const int fpC0   = []{ const char *s = std::getenv("LDCD_DUMP_FP_C0"); return s ? std::atoi(s) : -1; }();
    static const int fpC1   = []{ const char *s = std::getenv("LDCD_DUMP_FP_C1"); return s ? std::atoi(s) : -1; }();
    if (fpLine >= first && fpLine < last && fpC0 >= 0) {
        const lddecode::CarrierAnalysisRecord *rec = carrierAnalysis_line(fpLine);
        // Notch = raw - 1D bandpass (the CCR pixel edge read).  Recomputed
        // here from the canonical bandpass so the dump shows the SAME signal
        // splitIQlocked differences at +/-2.  IRE units.  The +/-2 difference
        // uses CCR's clamp (max/min), not the bandpass reflection, matching
        // the edge read exactly.
        const double  *bpLine  = locked1DRawBandpass_line(fpLine);
        const quint16 *rawLine =
            rawbuffer.data() + static_cast<size_t>(fpLine) * fullWidth;
        auto notchIRE = [&](int rel) -> double {
            const int r = std::clamp(rel, 0, width - 1);
            return ((double)rawLine[left + r] - bpLine[r]) * invIreScale;
        };
        if (rec && bpLine) {
            const int c1 = (fpC1 >= 0 ? fpC1 : fpC0);
            for (int rel = fpC0; rel <= c1 && rel < width; ++rel) {
                const auto &r = rec[rel];
                const char *sc = r.scheduleConformance ==
                        lddecode::CarrierScheduleConformance::LegalCarrier ? "LEG"
                    : r.scheduleConformance ==
                        lddecode::CarrierScheduleConformance::ScheduleIllegal ? "ILL"
                    : "unr";
                const int    xm     = std::max(0, rel - 2);
                const int    xp     = std::min(width - 1, rel + 2);
                const double notch  = notchIRE(rel);
                const double nedge  = std::fabs(notchIRE(xp) - notchIRE(xm));
                const double rawIRE = (double)rawLine[left + rel] * invIreScale;
                const double bpIRE  = bpLine[rel] * invIreScale;
                std::fprintf(stderr,
                    "[FP] line=%d h=%d raw=%.2f bp=%.2f src=%.2f wide=%.2f "
                    "incoh=%.2f lurch=%.2f notch=%.2f nedge=%.2f conf=%+.2f "
                    "supp=%.2f sched=%s\n",
                    fpLine, rel + left,
                    rawIRE, bpIRE,
                    r.fit.sourceSample * invIreScale,
                    r.fit.wideSample * invIreScale,
                    r.fit.sourceMinusWideIRE,
                    r.residual.maxAbsMembershipIRE,
                    notch, nedge,
                    r.carrierConformance,
                    r.conformanceSupportFraction, sc);
            }
        }
    }
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
        if (width >= 4) {
            double sum4 = (double)rawLine[left + 0] + (double)rawLine[left + 1]
                        + (double)rawLine[left + 2] + (double)rawLine[left + 3];
            for (int xi = 0; xi <= lastStart; ++xi) {
                apMean[xi] = 0.25 * sum4;
                if (xi < lastStart)
                    sum4 += (double)rawLine[left + xi + 4]
                          - (double)rawLine[left + xi];
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

void Comb::FrameBuffer::buildCornerLeak()
{
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
        for (int x = 0; x < width; ++x) {
            double g = ramp(ratio[x], kCornerParallaxSoft, kCornerParallaxHard);
            if (analysisRow) {
                const double contra = std::clamp(
                    (double)analysisRow[x].conformanceContradictionFraction,
                    0.0, 1.0);
                if (contra > 0.0)
                    g = 0.0;                 // observed legal-carrier vote: protect
                else
                    g *= lddecode::carrierIllegalProof(
                        (double)analysisRow[x].carrierConformance,
                        (double)analysisRow[x].conformanceSupportFraction);
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

    // ---- Disposable metrics (env-gated), without changing any output. -------
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
    static const int clDump = []{
        const char *s = std::getenv("LDCD_DUMP_CORNER"); return s ? std::atoi(s) : 0;
    }();
    if (clDump) {
        std::vector<double> lI(width), lQ(width), lIs(width), lQs(width);
        std::vector<double> c0I(width), c0Q(width), c0Is(width), c0Qs(width);
        std::vector<double> c1I(width), c1Q(width), c1Is(width), c1Qs(width);
        double yAltAll = 0.0, yAltHot = 0.0, maxLeak = 0.0;
        double satNum = 0.0, satDen = 0.0, hueW = 0.0, hueAcc = 0.0;
        long nAll = 0, nHot = 0;
        for (int line = first; line < last; ++line) {
            const double *bpLine = locked1DRawBandpass_line(line);
            const double *leakRow = lockedCornerLeak_line(line);
            if (!bpLine || !leakRow) continue;
            for (int x = 0; x < width; ++x) {
                const int ph = carrierSampleClass(line, left + x);
                const double s = sin4fsc(ph), c = cos4fsc(ph);
                lI[x]  = 2.0 * leakRow[x] * s;         lQ[x]  = 2.0 * leakRow[x] * c;
                c0I[x] = 2.0 * bpLine[x] * s;          c0Q[x] = 2.0 * bpLine[x] * c;
                const double cc = bpLine[x] - leakRow[x];
                c1I[x] = 2.0 * cc * s;                 c1Q[x] = 2.0 * cc * c;
            }
            // Lowpass each demodulated stream to the lawful envelope band; this
            // strips the 2fSC demod image so the fSC-rate part is what remains.
            lddecode::projectExpressibleChromaEnvelope(lI.data(),  nullptr, width, lIs.data());
            lddecode::projectExpressibleChromaEnvelope(lQ.data(),  nullptr, width, lQs.data());
            lddecode::projectExpressibleChromaEnvelope(c0I.data(), nullptr, width, c0Is.data());
            lddecode::projectExpressibleChromaEnvelope(c0Q.data(), nullptr, width, c0Qs.data());
            lddecode::projectExpressibleChromaEnvelope(c1I.data(), nullptr, width, c1Is.data());
            lddecode::projectExpressibleChromaEnvelope(c1Q.data(), nullptr, width, c1Qs.data());
            for (int x = 6; x < width - 6; ++x) {
                maxLeak = std::max(maxLeak, std::fabs(leakRow[x]) * invIreScale);
                // (a) fSC-rate envelope of the injected luma change = alternation.
                const double yAlt = std::hypot(lIs[x], lQs[x]) * invIreScale;
                yAltAll += yAlt; ++nAll;
                const double a0 = std::hypot(c0Is[x], c0Qs[x]);
                const double a1 = std::hypot(c1Is[x], c1Qs[x]);
                if (a0 * invIreScale > 6.0) {          // real carrier present
                    yAltHot += yAlt; ++nHot;
                    satNum += a1 * invIreScale; satDen += a0 * invIreScale;
                    // (b) hue rotation, amplitude-weighted, corrected vs raw.
                    double dth = std::atan2(c1Qs[x], c1Is[x])
                               - std::atan2(c0Qs[x], c0Is[x]);
                    while (dth >  M_PI) dth -= 2.0 * M_PI;
                    while (dth < -M_PI) dth += 2.0 * M_PI;
                    hueAcc += a0 * std::fabs(dth); hueW += a0;
                }
            }
        }
        if (nAll > 0)
            std::fprintf(stderr,
                "[CORNER] Yalt %.3f IRE (hot %.3f)  satRet %+.1f%%  hueRot %.1f deg"
                "  maxLeak %.1f IRE  nHot=%ld\n",
                yAltAll / nAll, nHot ? yAltHot / nHot : 0.0,
                100.0 * (satNum - satDen) / std::max(satDen, 1e-9),
                nHot ? (hueAcc / std::max(hueW, 1e-9)) * 180.0 / M_PI : 0.0,
                maxLeak, nHot);
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

    // Phase-1 cross-color diagnostic gate (parsed once).  See the measurement
    // block in Pass 2.  A per-framebuffer heartbeat is emitted before the early
    // return so a silent dump is self-explaining: it reports whether the locked
    // path is even active and whether the requested line is in the active range.
    static const int ccDiagLine = []{ const char *s = std::getenv("CC_DIAG_LINE"); return s ? std::atoi(s) : -1; }();
    static const int ccDiagC0   = []{ const char *s = std::getenv("CC_DIAG_C0");   return s ? std::atoi(s) : -1; }();
    static const int ccDiagC1   = []{ const char *s = std::getenv("CC_DIAG_C1");   return s ? std::atoi(s) : -1; }();
    static const int crDiagLine = []{ const char *s = std::getenv("COARSE_RESID_DIAG_LINE"); return s ? std::atoi(s) : -1; }();
    static const int crDiagC0   = []{ const char *s = std::getenv("COARSE_RESID_DIAG_C0");   return s ? std::atoi(s) : -1; }();
    static const int crDiagC1   = []{ const char *s = std::getenv("COARSE_RESID_DIAG_C1");   return s ? std::atoi(s) : -1; }();
    static const double crDiagFitTolIRE = []{
        const char *s = std::getenv("COARSE_RESID_DIAG_FIT_TOL_IRE");
        return s ? std::atof(s) : 1.5;
    }();
    static const double parallaxRepairTolIRE = []{
        const char *s = std::getenv("LD_1D_PARALLAX_TOL_IRE");
        return s ? std::atof(s) : 0.5;
    }();
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
    if (ccDiagLine >= 0) {
        std::fprintf(stderr,
            "CCDIAG-FB phaseComp=%d activeLines=[%d,%d) width=%d target=%d %s\n",
            configuration.phaseCompensation ? 1 : 0, first, last, width, ccDiagLine,
            (ccDiagLine >= first && ccDiagLine < last) ? "in-range" : "OUT-OF-RANGE");
    }
    if (crDiagLine >= 0) {
        std::fprintf(stderr,
            "COARSERES-FB phaseComp=%d activeLines=[%d,%d) width=%d target=%d %s\n",
            configuration.phaseCompensation ? 1 : 0, first, last, width, crDiagLine,
            (crDiagLine >= first && crDiagLine < last) ? "in-range" : "OUT-OF-RANGE");
    }

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

        // Pair class-map probe (measurement only; inert unless
        // LDCD_PROBE_DISENT). The 1D output is never touched from other
        // lines -- 1D is downstream's safe retreat.
        probeEdgePairClassMap(line);

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
            const bool repairLogThisLine =
                crDiagLine >= 0 && line == crDiagLine && crDiagC0 >= 0;
            const int repairLogFirst =
                repairLogThisLine ? std::clamp(crDiagC0, 0, width - 1) : 0;
            const int repairLogLast =
                repairLogThisLine
                    ? std::clamp(crDiagC1 < 0 ? crDiagC0 : crDiagC1,
                                 repairLogFirst, width - 1)
                    : -1;

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

            if (repairLogThisLine) {
                std::fprintf(stderr,
                    "COARSERESREPAIR header line rel h phase mode reason "
                    "sourceBp shortFit wideFit movingResidual optionCount "
                    "survivorCount survivorLo survivorHi tolIRE maxDeltaIRE "
                    "proposedDeltaIRE appliedDeltaIRE movingDistIRE "
                    "maxAbsMembershipIRE sourceMinusShortIRE shortMinusWideIRE "
                    "sourceMinusWideIRE\n");
            }

            for (int rel = 0; rel < width; ++rel) {
                const int p = carrierSampleClass(line, left + rel) & 3;
                const auto &record = carrierAnalysis[rel];
                const auto &residualDiag = record.residual;
                const int optionCount = residualDiag.optionCount;
                const double shortSample = record.fit.shortSample;
                const double wideSample = record.fit.wideSample;
                const double movingResidual = residualDiag.movingResidualSample;
                const double sourceSample = bpLine[rel];
                const int survivorCount = residualDiag.survivorCount();
                const double survivorLo = residualDiag.survivorLo;
                const double survivorHi = residualDiag.survivorHi;
                const double maxAbsMembershipIRE =
                    residualDiag.maxAbsMembershipIRE;

                const char *reason = "no-options";
                double proposedDelta = 0.0;
                double appliedDelta = 0.0;
                double movingDistIRE = 0.0;

                if (carrierAnalysis[rel].scheduleConformance ==
                    lddecode::CarrierScheduleConformance::ScheduleIllegal) {
                    // Registered as luma at analysis time: there is no
                    // carrier here to repair, and the residual options are
                    // luma interpretations that would only masquerade as
                    // survivor conflict.
                    reason = "schedule-illegal-luma";
                } else if (optionCount <= 0) {
                    reason = "no-options";
                } else if (survivorCount <= 0) {
                    reason = "conflict-no-survivors";
                } else if (survivorCount == optionCount) {
                    reason = "no-discrimination-all-survive";
                } else if (sourceSample >= survivorLo &&
                           sourceSample <= survivorHi)
                {
                    reason = "source-inside";
                } else {
                    movingDistIRE = residualDiag.movingDistanceIRE;
                    const bool movingCompatible = residualDiag.movingCompatible;

                    if (!movingCompatible) {
                        reason = "moving-conflict";
                    } else {
                        const double target =
                            std::clamp(sourceSample, survivorLo, survivorHi);
                        proposedDelta = target - sourceSample;
                        appliedDelta = std::clamp(
                            proposedDelta,
                            -maxDeltaSamples,
                            maxDeltaSamples);

                        if (!parallaxRepairApply) {
                            // Analysis complete, but the repair is opt-in:
                            // the ordinary bandpass remains source authority
                            // and no repair hold is published downstream.
                            reason = "report-only";
                            appliedDelta = 0.0;
                        } else {
                            reason = "apply";
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

                if (repairLogThisLine &&
                    rel >= repairLogFirst && rel <= repairLogLast)
                {
                    std::fprintf(stderr,
                        "COARSERESREPAIR line=%d rel=%d h=%d phase=%d "
                        "mode=%s reason=%s sourceBp=%.6f shortFit=%.6f "
                        "wideFit=%.6f movingResidual=%.6f optionCount=%d "
                        "survivorCount=%d survivorLo=%.6f survivorHi=%.6f "
                        "tolIRE=%.6f maxDeltaIRE=%.6f proposedDeltaIRE=%.6f "
                        "appliedDeltaIRE=%.6f movingDistIRE=%.6f "
                        "maxAbsMembershipIRE=%.6f sourceMinusShortIRE=%.6f "
                        "shortMinusWideIRE=%.6f sourceMinusWideIRE=%.6f\n",
                        line, rel, left + rel, p, "default", reason, sourceSample,
                        shortSample, wideSample, movingResidual, optionCount,
                        survivorCount, survivorLo, survivorHi,
                        std::max(0.0, parallaxRepairTolIRE),
                        std::max(0.0, parallaxRepairMaxDeltaIRE),
                        proposedDelta * invIreScale,
                        appliedDelta * invIreScale,
                        movingDistIRE,
                        maxAbsMembershipIRE,
                        (sourceSample - shortSample) * invIreScale,
                        (shortSample - wideSample) * invIreScale,
                        (sourceSample - wideSample) * invIreScale);
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

        // Wide coherent envelope: |sum(I,Q)| over the window, normalized so a
        // coherent carrier of amplitude A returns ~A regardless of width.
        auto wideEnvIRE = [&](int center) -> double {
            const double meanI = centeredEvenWeightMean(
                demI.data(), preI.data(), width, center, kWideWin);
            const double meanQ = centeredEvenWeightMean(
                demQ.data(), preQ.data(), width, center, kWideWin);
            return 2.0 * boundedMag(meanI, meanQ) * invIreScale;
        };

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
        const bool crDiagThisLine =
            crDiagLine >= 0 && line == crDiagLine && crDiagC0 >= 0;
        const int crDiagFirst =
            crDiagThisLine ? std::clamp(crDiagC0, 0, width - 1) : 0;
        const int crDiagLast =
            crDiagThisLine
                ? std::clamp(crDiagC1 < 0 ? crDiagC0 : crDiagC1,
                             crDiagFirst, width - 1)
                : -1;
        if (crDiagThisLine) {
            std::fprintf(stderr,
                "COARSERESFIT header line rel h phase sourceBp shortFit wideFit "
                "shortTolIRE shortLo shortHi optionCount survivorCount survivorLo "
                "survivorHi nearestShortDistIRE sourceClampDeltaIRE "
                "wideClampDeltaIRE sourceMinusShortIRE shortMinusWideIRE "
                "sourceMinusWideIRE narrowMagIRE wideMagIRE impurity\n");
            std::fprintf(stderr,
                "COARSERESFITOPT header line rel view s residual residualIRE "
                "distShortIRE distWideIRE inShortBand membershipDeltaIRE "
                "membershipLocalX\n");
        }

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

            if (crDiagThisLine && rel >= crDiagFirst && rel <= crDiagLast) {
                const int p = carrierSampleClass(line, left + rel) & 3;
                const double wideSample =
                    2.0 * (ZwI * cosRef[p] + ZwQ * sinRef[p]);

                const double ZnI = centeredEvenWeightMean(
                    demI.data(), preI.data(), width, rel, kNarrowWin);
                const double ZnQ = centeredEvenWeightMean(
                    demQ.data(), preQ.data(), width, rel, kNarrowWin);
                const double shortSample =
                    2.0 * (ZnI * cosRef[p] + ZnQ * sinRef[p]);

                const double fitTolSamples =
                    std::max(0.0, crDiagFitTolIRE) * irescale;
                const double shortLo = shortSample - fitTolSamples;
                const double shortHi = shortSample + fitTolSamples;

                const int sFirst = std::max(0, rel - 3);
                const int sLast = (width >= 4)
                    ? std::min(rel, width - 4)
                    : -1;

                double survivorLo = 1e300;
                double survivorHi = -1e300;
                double nearestDistIRE = 1e300;
                int optionCount = 0;
                int survivorCount = 0;

                for (int s = sFirst; s <= sLast && optionCount < 4; ++s) {
                    const double y4 =
                        0.25 * (rawAtRel(s + 0) +
                                rawAtRel(s + 1) +
                                rawAtRel(s + 2) +
                                rawAtRel(s + 3));
                    const double residual = rawAtRel(rel) - y4;
                    const double distShortIRE =
                        std::fabs(residual - shortSample) * invIreScale;
                    const double distWideIRE =
                        std::fabs(residual - wideSample) * invIreScale;
                    const double distBandIRE =
                        (residual < shortLo)
                            ? (shortLo - residual) * invIreScale
                            : ((residual > shortHi)
                                ? (residual - shortHi) * invIreScale
                                : 0.0);
                    nearestDistIRE = std::min(nearestDistIRE, distBandIRE);

                    const bool inBand =
                        residual >= shortLo && residual <= shortHi;
                    if (inBand) {
                        ++survivorCount;
                        survivorLo = std::min(survivorLo, residual);
                        survivorHi = std::max(survivorHi, residual);
                    }

                    double membershipDeltaIRE = 0.0;
                    double membershipLocalX = 0.0;
                    if (s + 4 < width) {
                        membershipDeltaIRE =
                            0.25 * (rawAtRel(s + 4) - rawAtRel(s)) * invIreScale;
                        membershipLocalX =
                            0.5 * (static_cast<double>(s) +
                                   static_cast<double>(s + 4)) -
                            static_cast<double>(rel);
                    }

                    std::fprintf(stderr,
                        "COARSERESFITOPT line=%d rel=%d view=%d s=%d "
                        "residual=%.6f residualIRE=%.6f distShortIRE=%.6f "
                        "distWideIRE=%.6f inShortBand=%d membershipDeltaIRE=%.6f "
                        "membershipLocalX=%.6f\n",
                        line, rel, optionCount, s, residual,
                        residual * invIreScale, distShortIRE, distWideIRE,
                        inBand ? 1 : 0, membershipDeltaIRE, membershipLocalX);
                    ++optionCount;
                }

                if (survivorCount == 0) {
                    survivorLo = 0.0;
                    survivorHi = 0.0;
                }
                if (nearestDistIRE == 1e300)
                    nearestDistIRE = 0.0;

                auto clampDeltaIRE = [&](double sample) {
                    if (survivorCount <= 0)
                        return 0.0;
                    const double clamped =
                        std::clamp(sample, survivorLo, survivorHi);
                    return (clamped - sample) * invIreScale;
                };

                const double sourceSample = bpLine[rel];
                const double sourceMinusShortIRE =
                    (sourceSample - shortSample) * invIreScale;
                const double shortMinusWideIRE =
                    (shortSample - wideSample) * invIreScale;
                const double sourceMinusWideIRE =
                    (sourceSample - wideSample) * invIreScale;
                std::fprintf(stderr,
                    "COARSERESFIT line=%d rel=%d h=%d phase=%d "
                    "sourceBp=%.6f shortFit=%.6f wideFit=%.6f "
                    "shortTolIRE=%.6f shortLo=%.6f shortHi=%.6f "
                    "optionCount=%d survivorCount=%d survivorLo=%.6f "
                    "survivorHi=%.6f nearestShortDistIRE=%.6f "
                    "sourceClampDeltaIRE=%.6f wideClampDeltaIRE=%.6f "
                    "sourceMinusShortIRE=%.6f shortMinusWideIRE=%.6f "
                    "sourceMinusWideIRE=%.6f narrowMagIRE=%.6f wideMagIRE=%.6f "
                    "impurity=%.6f\n",
                    line, rel, left + rel, p, sourceSample, shortSample,
                    wideSample, std::max(0.0, crDiagFitTolIRE), shortLo, shortHi,
                    optionCount, survivorCount, survivorLo, survivorHi,
                    nearestDistIRE, clampDeltaIRE(sourceSample),
                    clampDeltaIRE(wideSample), sourceMinusShortIRE,
                    shortMinusWideIRE, sourceMinusWideIRE, narrowMag, wideMag,
                    gA);
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

        // ---- Phase-1 cross-color measurement diagnostic (gated, no output) ----
        // Touches nothing in the signal path: not the source, not the impurity
        // buffer, not alphaEff.  When CC_DIAG_LINE is set it dumps, at columns
        // [CC_DIAG_C0,CC_DIAG_C1] (active-picture-relative), the purity facts
        // plus residual-winding facts.  Purpose: see whether contamination reads
        // rotating/scattered (high turn / low coherence, or low coherence with
        // amplitude) while authentic chroma reads aligned (residual phase steady,
        // stable centre nonzero) on the actual title frame, before any
        // correction is enabled.  (Gate vars are parsed once at function top.)
        if (ccDiagLine >= 0 && line == ccDiagLine && ccDiagC0 >= 0) {
            constexpr double kPi = 3.14159265358979323846;
            const int c0 = std::clamp(ccDiagC0, 0, width - 1);
            const int c1 = std::clamp(ccDiagC1 < 0 ? ccDiagC0 : ccDiagC1, c0, width - 1);
            for (int rel = c0; rel <= c1; ++rel) {
                // Stable centre Zwide (8-cycle complex mean).
                const double ZwI = centeredEvenWeightMean(
                    demI.data(), preI.data(), width, rel, kWideWin);
                const double ZwQ = centeredEvenWeightMean(
                    demQ.data(), preQ.data(), width, rel, kWideWin);
                const double stableAmpIRE = 2.0 * std::hypot(ZwI, ZwQ) * invIreScale;

                const double narrowMag = narrowEnvIRE(rel);
                const double wideMag   = wideEnvIRE(rel);
                const double gA = (narrowMag > kImpurityFloorIRE && wideMag < narrowMag)
                    ? clamp01((narrowMag - wideMag) / std::max(kImpurityFloorIRE, narrowMag))
                    : 0.0;

                // Residual winding over the narrow support: per-cycle complex
                // envelope minus the stable centre, amplitude-weighted phase walk.
                double sw = 0.0, scos = 0.0, ssin = 0.0, sturn = 0.0;
                double sRmag = 0.0, sZmag = 0.0;
                int nk = 0;
                double prevTheta = 0.0;
                bool havePrev = false;
                const int ka = rel - kNarrowWin / 2;
                const int kb = rel + kNarrowWin / 2;
                for (int k = ka; k <= kb; ++k) {
                    auto demAt = [&](const std::vector<double> &v, int x) {
                        return v[std::clamp(x, 0, width - 1)];
                    };
                    const double ZcI = centeredCarrierCycle4Mean(
                        demAt(demI, k - 2), demAt(demI, k - 1),
                        demAt(demI, k), demAt(demI, k + 1),
                        demAt(demI, k + 2));
                    const double ZcQ = centeredCarrierCycle4Mean(
                        demAt(demQ, k - 2), demAt(demQ, k - 1),
                        demAt(demQ, k), demAt(demQ, k + 1),
                        demAt(demQ, k + 2));
                    const double Ri = ZcI - ZwI;
                    const double Rq = ZcQ - ZwQ;
                    const double Rmag = std::hypot(Ri, Rq);
                    sRmag += Rmag;
                    sZmag += std::hypot(ZcI, ZcQ);
                    ++nk;
                    const double theta = std::atan2(Rq, Ri);
                    if (havePrev) {
                        double d = theta - prevTheta;
                        while (d >  kPi) d -= 2.0 * kPi;
                        while (d < -kPi) d += 2.0 * kPi;
                        const double w = Rmag;
                        sw    += w;
                        scos  += w * std::cos(d);
                        ssin  += w * std::sin(d);
                        sturn += w * d;
                    }
                    prevTheta = theta;
                    havePrev = true;
                }
                const double incrementCoherence = sw > 0.0 ? std::hypot(scos, ssin) / sw : 0.0;
                const double netTurn = sw > 0.0 ? std::fabs(sturn) / sw : 0.0;
                const double meanRIRE = nk > 0 ? (sRmag / nk) * invIreScale : 0.0;
                const double meanZIRE = nk > 0 ? (sZmag / nk) * invIreScale : 0.0;
                const double residualStrength =
                    meanRIRE / std::max(kImpurityFloorIRE, meanZIRE);

                std::fprintf(stderr,
                    "CCDIAG line=%d col=%d narrow=%.2f wide=%.2f stable=%.2f "
                    "gA=%.3f coh=%.3f turn=%.3f rstr=%.3f\n",
                    line, rel, narrowMag, wideMag, stableAmpIRE,
                    gA, incrementCoherence, netTurn, residualStrength);
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
        for (int rel = 0; rel < width; ++rel) {
            const int h = left + rel;
            const int phase = carrierSampleClass(line, h);
            const double source = restrainedLine[rel];

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

    static const int pcDiagLine = []{
        const char *s = std::getenv("CC_DIAG_LINE"); return s ? std::atoi(s) : -1;
    }();
    static const int pcDiagC0 = []{
        const char *s = std::getenv("CC_DIAG_C0"); return s ? std::atoi(s) : -1;
    }();
    static const int pcDiagC1 = []{
        const char *s = std::getenv("CC_DIAG_C1"); return s ? std::atoi(s) : -1;
    }();

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

            if (pcDiagLine >= 0 && line == pcDiagLine &&
                rel >= pcDiagC0 && rel <= pcDiagC1) {
                std::fprintf(stderr,
                    "CCPOST line=%d col=%d srcBuf=%d narrow=%.2f wide=%.2f gA=%.3f\n",
                    line, rel, srcBuf, narrowMag, wideMag, gA);
            }
        }
    }
}

// Cross-color vertical-image-detail corroboration ramp, shared by the
// coherent (splitIQlocked) and residual (filterIQLocked) transfer policies.
// Input is the 1D vertical-contrast service (|smooth[rel+2]-smooth[rel-2]|,
// IRE): below soft the coarse field is laterally flat and the edge read is
// silent; hard matches the established FIELD_LUMA_EDGE scale (18 IRE = a
// solid vertical stroke).
static constexpr double kCcEdgeSoftIRE = 6.0;
static constexpr double kCcEdgeHardIRE = 18.0;

// Concert gate: carrier-free confirmation that the notch edge is a real luma
// transition and not saturated-carrier leak.  The notch (edgeRamp) localizes
// the corner sharply but doubles residual carrier in saturated colour;
// maxAbsMembershipIRE (the same-carrier-phase membership change = lurch) reads
// luma movement through the aperture with the carrier cancelled, so it is dark
// in a saturated-chroma interior and bright at a genuine luma edge.  Measured
// on the beach strap (frame 52100, line 150): smooth/interior lurch 0.3-1.4,
// true skin<->strap edges 5-10.  soft/hard bracket that gap.
static constexpr double kCcLurchSoftIRE = 1.0;
static constexpr double kCcLurchHardIRE = 4.0;

void Comb::FrameBuffer::splitIQlocked()
{
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;
    const auto &T       = configuration.tunables;
    const int srcBuf    = std::clamp(static_cast<int>(configuration.dimensions) - 1, 0, 2);

    if (width <= 0 || firstLine >= lastLine) return;

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
        const float *hDeltaRow = lockedLumaHDeltaIRE_line(line);
        // Pixel-accurate notch luma (raw - 1D bandpass) for the cross-color
        // edge read only: sharper than the coarse ±2-on-lockedLumaSmooth
        // service, which localizes edges to ~2 columns and leaves diagonals
        // aliased. The shared lockedLumaHDeltaIRE service is left untouched for
        // its reach/FVF clients.
        const quint16 *ccRawLine =
            rawbuffer.data() + static_cast<size_t>(line) * videoParameters.fieldWidth;
        const double *ccSrc1D = locked1DSource_line(line);
        const lddecode::CarrierAnalysisRecord *analysisRow =
            carrierAnalysis_line(line);
        // ±2 same-field conformance context for the boundary/desert split
        // (see the transfer policy below).
        const lddecode::CarrierAnalysisRecord *analysisUpRow =
            (line - 2 >= firstLine) ? carrierAnalysis_line(line - 2) : nullptr;
        const lddecode::CarrierAnalysisRecord *analysisDnRow =
            (line + 2 < lastLine) ? carrierAnalysis_line(line + 2) : nullptr;

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

            // Two reads with DIFFERENT guards, because their failure modes
            // are mirror images.  One law over both: suppression may never
            // exceed measured evidence (the cap), so ccWeight redistributes
            // what was found but cannot manufacture.
            //
            // Aperture read: gA * (1 - regionKeep), NOT proof-gated.  gA is
            // the only witness against alias-conforming cross-colour (the
            // chain-link fence): that energy INVERTS across ±2 like legal
            // carrier -- that is exactly why the comb passes it as chroma --
            // so no conformance-based proof can ever convict it intra-field,
            // and proof-gating this read let the fence confetti through
            // untouched.  Its guard against real-chroma ringing (gA reads
            // 0.3-0.6 at genuine chroma texture and edges) is regionKeep's
            // same-hue vertical continuity.
            //
            // Edge read: min(edgeRamp, proof), proof-REQUIRED.  edgeRamp
            // (the 1D vertical-contrast service) is an edge detector by
            // construction, so ungated it desaturates every garment
            // boundary with a luma step (bikini against bright sand).  The
            // proof has two grades over the bandpass conformance:
            //   * strict carrierIllegalProof(): energy decisively MATCHING
            //     where the schedule demands inversion is luma by law --
            //     vertical strokes convict on the ±2 axes alone;
            //   * ambient carrierTrust() complement, confined to legality
            //     DESERTS: the ambiguous middle holds both genuine hue
            //     boundaries (correlation windows straddle two hues, worst
            //     near-complementary: cyan fabric against skin fakes
            //     "fails to invert") and in-field diagonal detail.
            //     Conformance cannot split them, vertical context can: a
            //     hue boundary is a thin ambiguous band BETWEEN
            //     certified-legal regions and inherits their protection
            //     (carrierLegalProof at ±2, per-column); diagonal detail
            //     sits where nothing certifies legal and stays actionable.
            // Notch edge: raw - 1D bandpass, differenced at ±2 (same carrier
            // phase, so the carrier cancels while the un-smoothed luma keeps a
            // per-pixel edge). This is the "go back to notch for pixel
            // accuracy" read; its residual-carrier leak in saturated colour is
            // held by the grammar-pass exemption below. Falls back to the
            // shared smooth service if the notch inputs are unavailable.
            double hDeltaIRE = 0.0;
            if (ccRawLine && ccSrc1D) {
                const int xm = std::max(0, xi - 2);
                const int xp = std::min(width - 1, xi + 2);
                const double notchM =
                    (double)ccRawLine[left + xm] - ccSrc1D[xm];
                const double notchP =
                    (double)ccRawLine[left + xp] - ccSrc1D[xp];
                hDeltaIRE = std::fabs(notchP - notchM) * invIreScale;
            } else if (hDeltaRow) {
                hDeltaIRE = (double)hDeltaRow[xi];
            }
            const double edgeRamp = std::clamp(
                (hDeltaIRE - kCcEdgeSoftIRE) /
                    (kCcEdgeHardIRE - kCcEdgeSoftIRE),
                0.0, 1.0);

            // Burden of proof is on the CHROMA: it proves it is not luma ONLY
            // by passing carrier-grammar (carrierLegalProof). The former
            // "prove the energy is illegal" gate (carrierIllegalProof + the
            // carrierTrust complement) is deleted -- that inverted burden is a
            // rebellious shape that mutes the suppression on exactly the
            // diagonals the Luma Delta must act on. The luma edge (edgeRamp)
            // convicts by default; a pixel is spared ONLY where its own chroma
            // -- or a same-column +/-2 neighbour's, so a thin real hue boundary
            // inherits its legal regions' grammar proof -- is grammar-certified
            // legal. In-field diagonal cross-color certifies nowhere and stays
            // actionable.
            double grammarPass = 0.0;
            if (analysisRow) {
                grammarPass = lddecode::carrierLegalProof(
                    (double)analysisRow[xi].carrierConformance,
                    (double)analysisRow[xi].conformanceSupportFraction);
            }
            grammarPass = std::max(grammarPass, std::max(
                analysisUpRow
                    ? lddecode::carrierLegalProof(
                          (double)analysisUpRow[xi].carrierConformance,
                          (double)analysisUpRow[xi].conformanceSupportFraction)
                    : 0.0,
                analysisDnRow
                    ? lddecode::carrierLegalProof(
                          (double)analysisDnRow[xi].carrierConformance,
                          (double)analysisDnRow[xi].conformanceSupportFraction)
                    : 0.0));

            // Concert edge gate (LDCD_CCR_CONCERT): replace the binary
            // schedule veto (1 - grammarPass) with a carrier-free lurch
            // confirmation.  The veto spared the whole legal-carrier strap and
            // let the 1D bead survive on it (measured: CCR acted 0.00 across
            // h=376-401 while the true skin->strap edge sat under it); the
            // lurch gate instead fires the notch edge exactly where luma is
            // moving through the aperture and stays dark in the saturated
            // interior where the notch edge is pure carrier leak.
            static const bool ccrConcert = []{
                const char *s = std::getenv("LDCD_CCR_CONCERT");
                return s && std::atoi(s) != 0;
            }();
            double edgeGate = 1.0 - grammarPass;
            if (ccrConcert) {
                const double lurchIRE = analysisRow
                    ? (double)analysisRow[xi].residual.maxAbsMembershipIRE
                    : 0.0;
                edgeGate = std::clamp(
                    (lurchIRE - kCcLurchSoftIRE) /
                        (kCcLurchHardIRE - kCcLurchSoftIRE),
                    0.0, 1.0);
            }
            const double apertureRead = gA * (1.0 - regionKeep);
            const double edgeRead = edgeRamp * edgeGate;
            const double ccEvidence = std::max(gA, edgeRead);
            const double lumaWeight = std::clamp(
                std::max(apertureRead, edgeRead) * ccWeight,
                0.0, ccEvidence);

            // The verdict is NOT applied here.  Applied per-sample it carries
            // regionKeep's hard flips and gA's ring chatter at pixel pitch --
            // amplitude modulation that beats sidebands back into the chroma
            // passband and shreds both sides of a hue boundary (the residual
            // path documented this failure mode and band-limits; the coherent
            // path must too).  Pass 2 below smooths it into an envelope that
            // varies no faster than the chroma it gates, then applies.
            if (maskRawRow)
                maskRawRow[xi] = (float)lumaWeight;

            // Disposable CCR-path dump (env-gated): shows which read drives
            // lumaWeight at a chosen strap line, so we can see whether the
            // aperture (gA) path shadows the edge concert.
            static const int ccL  = []{ const char *s=std::getenv("LDCD_DUMP_CC_L");  return s?std::atoi(s):-1; }();
            static const int ccC0 = []{ const char *s=std::getenv("LDCD_DUMP_CC_C0"); return s?std::atoi(s):-1; }();
            static const int ccC1 = []{ const char *s=std::getenv("LDCD_DUMP_CC_C1"); return s?std::atoi(s):-1; }();
            if (line == ccL && (int)(h - left) >= ccC0 && (int)(h - left) <= ccC1) {
                const double lurchIRE = analysisRow
                    ? (double)analysisRow[xi].residual.maxAbsMembershipIRE : 0.0;
                std::fprintf(stderr,
                    "[CC] line=%d h=%d gA=%.2f regKeep=%.2f edgeRamp=%.2f "
                    "gPass=%.2f lurch=%.2f edgeGate=%.2f aperRead=%.2f "
                    "edgeRead=%.2f lumaW=%.2f drive=%s\n",
                    line, h, gA, regionKeep, edgeRamp, grammarPass, lurchIRE,
                    edgeGate, apertureRead, edgeRead, lumaWeight,
                    apertureRead >= edgeRead ? "APER" : "EDGE");
            }

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
        // A grammar-legal four-sample cycle remains chroma unless the
        // aperture detector supplies a strong contrary measurement of its
        // own.  This preserves the aperture path's ability to catch
        // alias-conforming cross-colour (which can pass carrier grammar), but
        // prevents its modest response at genuine hue/luma transitions from
        // overriding positive carrier-law evidence.  The transition probe is
        // well separated from the cube case: legal transition reads stay
        // below 0.30, while legal cube-alias reads cluster around 0.47.
        constexpr double kCcLegalOverrideSoft = 0.25;
        constexpr double kCcLegalOverrideHard = 0.45;
        std::vector<double> vmix(width, 0.0);

        for (int line = firstLine; line < lastLine; ++line) {
            const float *r0 = lockedCcMaskRaw_line(line);
            float *out = lockedCcMask_line(line);
            float *prodIRow = lockedProductI_line(line);
            float *prodQRow = lockedProductQ_line(line);
            if (!r0 || !out)
                continue;

            const float *rU =
                (line - 2 >= firstLine) ? lockedCcMaskRaw_line(line - 2)
                                        : nullptr;
            const float *rD =
                (line + 2 < lastLine) ? lockedCcMaskRaw_line(line + 2)
                                      : nullptr;
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

            const double norm =
                0.5 + (rU ? 0.25 : 0.0) + (rD ? 0.25 : 0.0);
            for (int xi = 0; xi < width; ++xi) {
                const double verticalMean =
                    (0.5 * r0[xi] +
                     (rU ? 0.25 * rU[xi] : 0.0) +
                     (rD ? 0.25 * rD[xi] : 0.0)) / norm;
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
                vmix[xi] = verticalMean + impulse *
                    std::max(0.0, (double)r0[xi] - verticalMean);
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
                const int cycleStart = (width >= 4)
                    ? std::clamp(xi - 1, 0, width - 4)
                    : 0;
                const int cycleCount = std::min(4, width);
                double cycleLegal = 0.0;
                double cycleAperture = 0.0;
                for (int k = 0; k < cycleCount; ++k) {
                    const int xk = cycleStart + k;
                    cycleLegal = std::max(cycleLegal, grammarPassAt(xk));
                    const double aperture = impurityRow
                        ? std::clamp((double)impurityRow[xk], 0.0, 1.0)
                        : 0.0;
                    const double sameRegion = sameRegionRow
                        ? std::clamp((double)sameRegionRow[xk], 0.0, 1.0)
                        : 0.0;
                    cycleAperture = std::max(
                        cycleAperture, aperture * (1.0 - sameRegion));
                }
                const double strongApertureOverride = smoothGate01(
                    (cycleAperture - kCcLegalOverrideSoft) /
                    (kCcLegalOverrideHard - kCcLegalOverrideSoft));
                const double grammarKeep =
                    cycleLegal * (1.0 - strongApertureOverride);
                m *= 1.0 - grammarKeep;

                out[xi] = (float)m;
                if (prodIRow) prodIRow[xi] = (float)(prodIRow[xi] * (1.0 - m));
                if (prodQRow) prodQRow[xi] = (float)(prodQRow[xi] * (1.0 - m));
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
        const float *maskRow = lockedCcMask_line(line);
        const double giProduct = configuration.gi_product;
        const double gqProduct = configuration.gq_product;

        for (int i = 0; i < width; ++i) {
            const int h = left + i;
            const int ph = carrierSampleClass(line, h);
            const double chroma = (double)rawLine[h] - Yrow[h];
            const double m = maskRow
                ? std::clamp((double)maskRow[i], 0.0, 1.0)
                : 0.0;
            const double alphaEff = 1.0 - m;
            scratch_preI[i] = (chroma * lutTi[ph]) * giProduct * alphaEff;
            scratch_preQ[i] = (chroma * lutTq[ph]) * gqProduct * alphaEff;
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

void Comb::FrameBuffer::produceY()
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

    // Off-grid leakage probe (measurement only; inert unless
    // LDCD_PROBE_OFFGRID). All published carriers exist by this point.
    probeOffGrid();

    // ---- HF-election diagnostic probe (no output influence) ----
    // Column/line gated per-pixel dump of the produceY HF luma election:
    // candidate roster + planes, self/neighbor/decision anchors, the winner,
    // the regime vertical step, and whether the clean 1D candidate was
    // roster-excluded.  Enable with LDCD_PY_L0/L1 (frame line range) and
    // LDCD_PY_C0/C1 (active-picture column range, i.e. h - left).  Run -t 1.
    static const int pyDiagL0 = []{ const char *s = std::getenv("LDCD_PY_L0"); return s ? std::atoi(s) : -1; }();
    static const int pyDiagL1 = []{ const char *s = std::getenv("LDCD_PY_L1"); return s ? std::atoi(s) : -1; }();
    static const int pyDiagC0 = []{ const char *s = std::getenv("LDCD_PY_C0"); return s ? std::atoi(s) : -1; }();
    static const int pyDiagC1 = []{ const char *s = std::getenv("LDCD_PY_C1"); return s ? std::atoi(s) : -1; }();
    const bool pyDiag = pyDiagL0 >= 0 && pyDiagC0 >= 0;
    const bool consDump = std::getenv("LDCD_DUMP_CONS") != nullptr;
    long long consN = 0, consP0N = 0, consOtherN = 0;
    double consSum = 0.0, consP0Sum = 0.0, consOtherSum = 0.0, consMax = 0.0;

    // --- Disposable emission-hull instrumentation (env-gated). Set
    // LDCD_DUMP_YHULL=1 to print, per frame, how many emitted pixels fail
    // |raw - Y| <= maxCarrierAmpSamples -- the legality test the election
    // applies to CANDIDATES but never re-applies to the emitted band splice
    // reconstructTop(top) = coarse + combMiddle + combPlatform + top.
    // Broken out by which top band won so a fix can be targeted. Set
    // LDCD_DUMP_YHULL=2 to also print each individual violation. Zero cost
    // when unset. Remove with the rethink.
    static const char *dumpHullEnv = std::getenv("LDCD_DUMP_YHULL");
    const bool dumpHull = dumpHullEnv != nullptr;
    const bool dumpHullVerbose = dumpHullEnv && std::atoi(dumpHullEnv) >= 2;
    long long hullTotal = 0, hullOver = 0;
    long long hullOverByPlane[7] = {0, 0, 0, 0, 0, 0, 0}; // 0,1,3,4 planes; 5=blend; 6=clamped
    double hullOverSumIRE = 0.0, hullMaxOvershootIRE = 0.0;
    double hullMismatchSumIRE = 0.0, hullMaxMismatchIRE = 0.0;

    // --- Disposable retracted-win schedule attribution (env-gated). Set
    // LDCD_DUMP_RETR=1. Answers the decisive question: when the retracted
    // plane wins the election, is its kept near-carrier energy schedule-LEGAL
    // (comb was right, retracted is passthrough failure) or schedule-ILLEGAL
    // (real grid luma comb destroyed, retracted correctly kept it)? Split by
    // the "bright passthrough" signature: retracted sits at raw while comb
    // removed a full carrier lobe. Zero cost when unset. Remove with the
    // rethink.
    const bool dumpRetr = std::getenv("LDCD_DUMP_RETR") != nullptr;
    // [conformance 0=Unresolved 1=Legal 2=Illegal][passthrough 0/1]
    long long retrWinBySchedule[3][2] = {{0,0},{0,0},{0,0}};
    long long retrWinTotal = 0, retrPassthroughTotal = 0;
    double retrLicenseSumOnPass = 0.0;
    // Very-bright retracted wins (the ones the eye spots): output above a high
    // IRE threshold. Split by schedule so we learn whether the visible bright
    // specks are legal carrier (passthrough failure) or illegal grid luma.
    long long retrBrightBySchedule[3] = {0, 0, 0};
    long long retrBrightTotal = 0;

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

        // The witness/retraction path remains an independent luma hypothesis.
        // Its existence is a fact about the completed analysis, not a chroma
        // rendering mode.
        const float *retractedRow =
            carrierRetractedValid ? carrierRetracted_line(line) : nullptr;
        const float *ccMaskRow = lockedCcMask_line(line);
        const float *ccMaskRawRow = lockedCcMaskRaw_line(line);

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

        if (retractedRow || ccMaskRow) {
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
                configuration.lumaWitness && coarseSharpLevel() > 0.0 &&
                !lockedLumaSharp_flat.empty();
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
            const float *dsHDeltaRow = lockedLumaHDeltaIRE_line(line);
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
            // the feasibility DQ. Same bound buildCarrierRetracted clamps with.
            const CombCarrierGrammar *grammarLine =
                carrierGrammarLine(line);
            const double maxCarrierAmpSamples = grammarLine
                ? std::max(24.0, grammarLine->carrierScale * 5.0) * irescale
                : 24.0 * irescale;

            // Checks the emitted band splice against the same hull the
            // election applies to candidates -- candidate feasibility says
            // nothing about the reconstructTop() value actually written to
            // Y[h], since only plane 0 telescopes back to combY exactly.
            auto tallyHull = [&](double rawSample, double emittedY, int plane, int h,
                                 double fourMismatch, double planeComplete) {
                if (!dumpHull) return;
                ++hullTotal;
                const double c = rawSample - emittedY;
                if (c > maxCarrierAmpSamples || c < -maxCarrierAmpSamples) {
                    ++hullOver;
                    const int p = std::clamp(plane, 0, 6);
                    ++hullOverByPlane[p];
                    const double overIRE =
                        (std::fabs(c) - maxCarrierAmpSamples) * invIreScale;
                    hullOverSumIRE += overIRE;
                    hullMaxOvershootIRE = std::max(hullMaxOvershootIRE, overIRE);
                    hullMismatchSumIRE += std::fabs(fourMismatch) * invIreScale;
                    hullMaxMismatchIRE = std::max(hullMaxMismatchIRE,
                                                  std::fabs(fourMismatch) * invIreScale);
                    if (dumpHullVerbose)
                        std::fprintf(stderr,
                            "[YHULL-EV] line=%d h=%d plane=%d raw=%.2f Y=%.2f "
                            "c=%.2f maxAmp=%.2f overIRE=%.2f "
                            "four0-fourP=%.2f planeComplete=%.2f "
                            "cIfOwnBands=%.2f\n",
                            line, h, plane, rawSample * invIreScale,
                            emittedY * invIreScale, c * invIreScale,
                            maxCarrierAmpSamples * invIreScale, overIRE,
                            fourMismatch * invIreScale,
                            planeComplete * invIreScale,
                            (rawSample - planeComplete) * invIreScale);
                }
            };

            // Retracted-win schedule attribution. Called at each emission with
            // the winning plane; only plane 1 (retracted) is tallied. The
            // "bright passthrough" signature is retracted sitting at raw while
            // comb removed a full carrier lobe -- the failure the reframe
            // targets. The schedule enum then says whether that kept energy is
            // legal carrier (comb right, retracted wrong) or illegal grid luma
            // (retracted right).
            const double passRetrTolSamp = 6.0 * irescale;
            const double passLobeSamp    = 12.0 * irescale;
            auto tallyRetr = [&](double rawSample, int winnerPlane,
                                 double combVal, double retrVal, int h,
                                 const lddecode::CarrierAnalysisRecord *rec) {
                if (!dumpRetr || winnerPlane != 1) return;
                ++retrWinTotal;
                const bool passthrough =
                    std::fabs(retrVal - rawSample) < passRetrTolSamp &&
                    std::fabs(combVal - rawSample) > passLobeSamp;
                if (passthrough) ++retrPassthroughTotal;
                int sc = 0; // Unresolved
                double lic = 0.0;
                if (rec) {
                    switch (rec->scheduleConformance) {
                        case lddecode::CarrierScheduleConformance::LegalCarrier:
                            sc = 1; break;
                        case lddecode::CarrierScheduleConformance::ScheduleIllegal:
                            sc = 2; break;
                        default: sc = 0; break;
                    }
                    lic = lddecode::carrierScheduleLicense(
                        (double)rec->carrierConformance,
                        (double)rec->conformanceSupportFraction,
                        (double)rec->conformanceContradictionFraction);
                }
                ++retrWinBySchedule[sc][passthrough ? 1 : 0];
                if (passthrough) retrLicenseSumOnPass += lic;
                // retrVal is a sample-domain luma; convert to IRE for the
                // brightness gate the same way the hull dump does.
                if (retrVal * invIreScale > 100.0) {
                    ++retrBrightTotal;
                    ++retrBrightBySchedule[sc];
                }
            };

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
                    return comb + m * ((double)rawLine[hh] - comb);
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
                const int xx = hh - left;
                return planeY(plane, hh) - coarseRow[xx];
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
                    const double dc = coarseRow[hh - left];
                    hf5[j] = planeY(plane, hh) - dc;
                    // Index the carrier basis by the grammar sample class, NOT
                    // hh & 3. The locked demod (the basis these LUTs were built
                    // for) uses carrierSampleClass(line, h); a raw-position
                    // index applies a per-line rotation, making cleanliness
                    // line-dependent -> a line-alternating election penalty
                    // (checkerboard) on luma transitions.
                    const int idx = carrierSampleClass(line, hh);
                    s5[j] = spLUT_locked[idx];
                    c5[j] = cpLUT_locked[idx];
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
                        (double)rawLine[hh] - planeY(plane, hh);
                    const int idx = carrierSampleClass(line, hh);
                    dotS += w * residualCarrier * spLUT_locked[idx];
                    dotC += w * residualCarrier * cpLUT_locked[idx];
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

                // (D-S)/2 referee: grade every estimator against the exact
                // carrier wherever the side channel covers this sample.
                // Before any early-out so coverage is complete.
                if (DsRefProbe::on() && dsExactRow &&
                    std::isfinite(dsExactRow[h])) {
                    const double ex = dsExactRow[h];
                    const double hd = dsHDeltaRow ? dsHDeltaRow[xi] : 0.0;
                    const int bin = hd >= 6.0 ? 1 : 0;
                    g_dsRefProbe.covered++;
                    if (oneDRow && std::isfinite(oneDRow[xi]))
                        g_dsRefProbe.e[bin][0].add((oneDRow[xi] - ex) * invIreScale);
                    g_dsRefProbe.e[bin][1].add((rawH - combY - ex) * invIreScale);
                    const float *fitRowDs = carrierFit_line(line);
                    if (fitRowDs && std::isfinite(fitRowDs[xi]))
                        g_dsRefProbe.e[bin][2].add((fitRowDs[xi] - ex) * invIreScale);
                    if (retractedRow && std::isfinite(retractedRow[xi]))
                        g_dsRefProbe.e[bin][3].add(
                            (rawH - (double)retractedRow[xi] - ex) * invIreScale);
                }

                const double ccReturn = ccMaskRow
                    ? std::clamp((double)ccMaskRow[xi], 0.0, 1.0)
                    : 0.0;
                const double ccMeasuredHere = ccMaskRawRow
                    ? std::clamp((double)ccMaskRawRow[xi], 0.0, 1.0)
                    : 0.0;
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
                const double combMiddle = candidateMiddleAt(0, h);
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
                auto reconstructTop = [&](int plane, double top) {
                    // Non-comb winners keep their own four-mean so the top
                    // band is rebuilt in the same candidate geometry that won.
                    const double fourMean =
                        (plane == 0)
                            ? (combMiddle + combPlatform)
                            : candidateFourMeanAt(plane, h);
                    return coarse + fourMean + top;
                };
                // Transfer-knee probe (measurement only; inert unless
                // LDCD_PROBE_KNEE): at proven-luma pixels the retracted top
                // is trustworthy amplitude; bin comb's top against it.
                if (KneeProbe::on() && retractedRow && analysisRow) {
                    const double proofK = lddecode::carrierIllegalProof(
                        (double)analysisRow[xi].carrierConformance,
                        (double)analysisRow[xi].conformanceSupportFraction);
                    if (proofK >= 0.7) {
                        const double rT = candidateTopAt(1, h);
                        const double cT = candidateTopAt(0, h);
                        if (std::isfinite(rT) && std::isfinite(cT))
                            g_kneeProbe.sample(rT * invIreScale,
                                               cT * invIreScale);
                    }
                }

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
                double candY[3]; // top-band values; name retained locally
                int    candPlane[3];
                int    nCand = 0;
                const double identityTol = 1e-6 * irescale;
                auto addBaseCandidate = [&](double completeY, int plane) {
                    const double y = candidateTopAt(plane, h);
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

                if (retractedRow) {
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
                    const double r = retractedRow ? (double)retractedRow[xi] : combY;
                    const double ry = std::isfinite(r) ? r : combY;
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

                // Returned Y is derived from combY, so it is a selectable
                // challenger but not another independent observation when the
                // base candidates establish their center, subset, or scoring
                // scales. It joins only after those quantities are fixed.
                const double returnedY =
                    combY + ccReturn * (rawH - combY);
                const bool returnedFeasible =
                    ccReturn > 0.0 && std::isfinite(returnedY) &&
                    feasible(returnedY);

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
                    tallyHull(rawH, Y[h], 6 /* no legal top */, h, 0.0, combY);
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
                int inIdx[3];
                int nIn = 0;
                for (int k = 0; k < nCand; ++k)
                    inIdx[nIn++] = k;
                // The named cross-colour mask, rather than distance from the
                // base center, admits returned Y. A distance gate here removes
                // exactly the strong HF that the return exists to recover.
                const bool returnedAdmitted = returnedFeasible;
                if (nIn == 1 && !returnedAdmitted) {
                    Y[h] = reconstructTop(candPlane[inIdx[0]], candY[inIdx[0]]);
                    if (dumpHull) {
                        const int wp = candPlane[inIdx[0]];
                        tallyHull(rawH, Y[h], wp, h,
                                  candidateFourMeanAt(0, h) - candidateFourMeanAt(wp, h),
                                  planeY(wp, h));
                    }
                    tallyRetr(rawH, candPlane[inIdx[0]], combY, planeY(1, h), h,
                              analysisRow ? &analysisRow[xi] : nullptr);
                    continue;
                }

                // Inlier HF set + per-inlier carrier-basis cleanliness. This is
                // a cautionary term, not the positive reason to select HF.
                double inHF[3], inCarrierCleanliness[3];
                double inCrossColorReturnEvidence[3] = {0.0, 0.0, 0.0};
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
                if (ccReturn > 0.0) {
                    const double combCarrierMagnitude =
                        remainingCarrierMagnitudeOf(0, h);
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
                        const double deliveredReduction = std::max(
                            0.0,
                            combCarrierMagnitude -
                                remainingCarrierMagnitudeOf(plane, h));
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

                // Diagnostics for the pyDiag dump: proximity01 and the earned
                // legality reward (IRE) take the retired continuation/retained
                // slots; the decision anchor is retired (no anchor exists).
                double diagContinuation[3] = {
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN()
                };
                double diagRetained[3] = {
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN()
                };
                double diagNeighborAnchor = std::numeric_limits<double>::quiet_NaN();
                const double diagDecisionAnchor = std::numeric_limits<double>::quiet_NaN();

                // Capped-caution reference: median cleanliness of the base set.
                double sw[3];
                const int refN = std::clamp(baseNIn, 1, nIn);
                for (int i = 0; i < refN; ++i)
                    sw[i] = inCarrierCleanliness[i];
                std::sort(sw, sw + refN);
                const double medianW = sw[refN / 2];

                const double combTopHere = candidateTopAt(0, h);
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
                    (remainingCarrierMagnitudeOf(0, h) * invIreScale -
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
                // Y-election band cede. Measured at the bikini-diagonal teeth
                // (2026-07-27): inside a chroma-boundary band every candidate's
                // top deviates by up to ~11 IRE from mono -- none is
                // trustworthy -- and the per-column winner flips comb/retracted
                // 46/54, a decision interleave at maximal spread. That
                // interleave IS the witness fishboning, the same crime as the
                // Field B per-column verdicts, relocated to the Y election.
                // A boundary band takes ONE decision: the comb candidate (the
                // error-comb chain: Field B ceded to 1D there, so this is the
                // soft, artifact-free reconstruction). Scoring resumes outside
                // bands. Field-B-less variants (line) publish no band plane
                // and are unaffected.
                bool bandCede = false;
                if (bandRow && bandRow[xi]) {
                    for (int k = 0; k < baseNIn; ++k) {
                        if (candPlane[inIdx[k]] == 0) {
                            resultHF = inHF[k];
                            bandCede = true;
                            break;
                        }
                    }
                }
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
                // Vetoes remain binary and upstream: admission, feasibility
                // DQ, and the chroma-boundary band cede.
                double blendNum = 0.0, blendDen = 0.0;
                constexpr double kBlendTauIRE = 0.75;
                const double blendTau = kBlendTauIRE * irescale;
                double costs[4];
                if (!bandCede) {
                    double bestCost = 1e300;
                    for (int k = 0; k < nIn; ++k) {
                        const int plane =
                            (k < baseNIn) ? candPlane[inIdx[k]] : 4;
                        // Neighbour boost: nearest adjacent top. With no
                        // usable neighbour, the medoid stands in as the sole
                        // fallback reference.
                        double nd = 1e300, nv = selfAnchor;
                        for (int d = 0; d < nDir; ++d) {
                            const double dd = std::fabs(inHF[k] - dirHF[d]);
                            if (dd < nd) { nd = dd; nv = dirHF[d]; }
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
                        const double legality =
                            illegalProof * std::min(kEarlyHandoffSlope * extra,
                                                    imagePrefCap);
                        cost -= legality;
                        // Cross-colour return evidence helps only to the
                        // degree the image supports the candidate, as before.
                        cost -= proximity01 *
                            std::max(0.0, inCrossColorReturnEvidence[k]);
                        if (plane == 1)
                            cost += chromaT * kHighChromaDemoteIRE * irescale -
                                    impulseT * kImpulseRetractedBiasIRE *
                                        irescale;
                        if (k < 3) {
                            diagContinuation[k] = proximity01;
                            diagRetained[k] = legality * invIreScale;
                        }
                        if (k < 4) costs[k] = cost;
                        // Strict < keeps roster order as the neutral
                        // tie-break: coherent comb stays senior on ties.
                        if (cost < bestCost) {
                            bestCost = cost;
                            resultHF = inHF[k];
                            diagNeighborAnchor = nv;
                        }
                    }
                    for (int k = 0; k < nIn && k < 4; ++k) {
                        const int plane =
                            (k < baseNIn) ? candPlane[inIdx[k]] : 4;
                        const double yk = reconstructTop(plane, inHF[k]);
                        if (!std::isfinite(yk)) continue;
                        double w =
                            std::exp(-(costs[k] - bestCost) / blendTau);
                        if (k == darkestIdx && k < baseNIn && baseNIn > 1)
                            w *= 1.0 - darkestVetoGate;
                        blendNum += w * yk;
                        blendDen += w;
                    }
                }

                const int winnerPlane = planeForTop(resultHF);
                Y[h] = (!bandCede && blendDen > 1e-12)
                    ? blendNum / blendDen
                    : reconstructTop(winnerPlane, resultHF);
                if (consDump) {
                    // Bucket writes Y and chroma from ONE scalar:
                    //   Y = raw - val, chroma = demod(val)  => Y + chroma == raw.
                    // Locked demodulates chroma from carrierComp but emits Y
                    // from a band reassembly, so the identity only survives
                    // where the winner telescopes back to raw - carrierComp.
                    const double bucketY = rawH - (carrierComp ? carrierComp[xi]
                                                               : clpLine[h]);
                    const double d = Y[h] - bucketY;
                    ++consN; consSum += d * d;
                    consMax = std::max(consMax, std::fabs(d));
                    if (winnerPlane == 0) { ++consP0N; consP0Sum += d * d; }
                    else                  { ++consOtherN; consOtherSum += d * d; }
                }

                tallyRetr(rawH, winnerPlane, combY, planeY(1, h), h,
                          analysisRow ? &analysisRow[xi] : nullptr);

                if (dumpHull) {
                    int winnerPlaneHull = 5; // fallback: unattributed blend
                    for (int k = 0; k < nIn; ++k) {
                        if (inHF[k] == resultHF) {
                            winnerPlaneHull = (k < baseNIn) ? candPlane[inIdx[k]] : 4;
                            break;
                        }
                    }
                    const int wp = (winnerPlaneHull == 5) ? 0 : winnerPlaneHull;
                    tallyHull(rawH, Y[h], winnerPlaneHull, h,
                              candidateFourMeanAt(0, h) - candidateFourMeanAt(wp, h),
                              planeY(wp, h));
                }

                if (pyDiag && line >= pyDiagL0 && line <= pyDiagL1 &&
                    xi >= pyDiagC0 && xi <= pyDiagC1) {
                    // Winner plane: resultHF is one of the inHF entries.
                    int winnerPlane = -1;
                    for (int k = 0; k < nIn; ++k)
                        if (inHF[k] == resultHF) {
                            winnerPlane = (k < baseNIn) ? candPlane[inIdx[k]] : 4;
                            break;
                        }
                    // Roster string: planes of the admitted base candidates.
                    char roster[64]; int rp = 0;
                    for (int k = 0; k < baseNIn && rp < 60; ++k)
                        rp += std::snprintf(roster + rp, sizeof(roster) - rp,
                                            "%d ", candPlane[inIdx[k]]);
                    if (rp == 0) { roster[0] = '-'; roster[1] = 0; }
                    else roster[rp ? rp - 1 : 0] = 0;
                    // Neighbour image-HF continuation values (N,S,E,W order).
                    char nbr[96]; int np = 0;
                    for (int d = 0; d < nDir && np < 90; ++d)
                        np += std::snprintf(nbr + np, sizeof(nbr) - np,
                                            "%.2f ", dirImageHF[d] / irescale);
                    if (np == 0) { nbr[0] = '-'; nbr[1] = 0; }
                    else nbr[np ? np - 1 : 0] = 0;
                    // Per-candidate elected TOP (IRE), including 1D even when
                    // excluded, plus the comb-owned middle. Keeping the bands
                    // separate makes it impossible for this diagnostic to hide
                    // a coarse or middle-band substitution under an HF label.
                    const double combTop = candidateTopAt(0, h) / irescale;
                    const double combPlatformResidual =
                        candidatePlatformResidualAt(0, h) / irescale;
                    const double retrTop = retractedRow
                        ? candidateTopAt(1, h) / irescale
                        : std::numeric_limits<double>::quiet_NaN();
                    const double oneDTop = oneDRow
                        ? candidateTopAt(3, h) / irescale
                        : std::numeric_limits<double>::quiet_NaN();
                    const double retnTop = candidateTopAt(4, h) / irescale;
                    auto rawComplete = [&](int hh) {
                        return (double)rawLine[hh];
                    };
                    // Mono composite top: ground-reference with carrier still
                    // included. Its line alternation exposes raw passthrough.
                    const double monoTop = completeTopAt(
                        rawComplete, coarseRow, h) / irescale;
                    const int returnedIndex = returnedAdmitted
                        ? nIn - 1 : -1;
                    const double returnedImagePref =
                        (returnedIndex >= 0 && returnedIndex < 3)
                        ? diagContinuation[returnedIndex]
                        : std::numeric_limits<double>::quiet_NaN();
                    const double returnedCcEvidence = returnedIndex >= 0
                        ? inCrossColorReturnEvidence[returnedIndex] / irescale
                        : std::numeric_limits<double>::quiet_NaN();
                    // Diagnostic-only: nativeTop is NOT a roster candidate.
                    // Measured here to answer the separability question the
                    // full-field-Y-runoff plan raised -- how far apart are
                    // native (raw-carrierFit, no vertical promotion) and
                    // promoted (retrTop, raw-combedCarrier) in the top band
                    // -- without seating native and paying its election cost.
                    const float *nativeCarrierRow = carrierFit_line(line);
                    double nativeTop = std::numeric_limits<double>::quiet_NaN();
                    if (nativeCarrierRow) {
                        auto nativeComplete = [&](int hh) {
                            return (double)rawLine[hh] -
                                (double)nativeCarrierRow[hh - left];
                        };
                        nativeTop = completeTopAt(
                            nativeComplete, coarseRow, h) / irescale;
                    }
                    // Per-admitted-candidate cleanliness and cc-return
                    // evidence, paired positionally with roster=[...]
                    // (baseNIn entries; the derived return is reported
                    // separately above as it is not a base candidate).
                    char cleanStr[64]; int cp = 0;
                    for (int k = 0; k < baseNIn && cp < 60; ++k)
                        cp += std::snprintf(cleanStr + cp, sizeof(cleanStr) - cp,
                                            "%.2f ", inCarrierCleanliness[k]);
                    if (cp == 0) { cleanStr[0] = '-'; cleanStr[1] = 0; }
                    else cleanStr[cp ? cp - 1 : 0] = 0;
                    char ccEvStr[64]; int cep = 0;
                    for (int k = 0; k < baseNIn && cep < 60; ++k)
                        cep += std::snprintf(ccEvStr + cep, sizeof(ccEvStr) - cep,
                                            "%.2f ", inCrossColorReturnEvidence[k] / irescale);
                    if (cep == 0) { ccEvStr[0] = '-'; ccEvStr[1] = 0; }
                    else ccEvStr[cep ? cep - 1 : 0] = 0;
                    // Image evidence per admitted candidate: proximity01 (the
                    // neighbour-boost factor) now fills the imgPref slot.
                    char imgStr[64]; int ip = 0;
                    for (int k = 0; k < baseNIn && ip < 60 && k < 3; ++k)
                        ip += std::snprintf(imgStr + ip, sizeof(imgStr) - ip,
                                            "%.3f ", diagContinuation[k]);
                    if (ip == 0) { imgStr[0] = '-'; imgStr[1] = 0; }
                    else imgStr[ip ? ip - 1 : 0] = 0;
                    char contStr[64]; int cnp = 0;
                    for (int k = 0; k < baseNIn && cnp < 60; ++k)
                        cnp += std::snprintf(contStr + cnp, sizeof(contStr) - cnp,
                                            "%.3f ", diagContinuation[k]);
                    if (cnp == 0) { contStr[0] = '-'; contStr[1] = 0; }
                    else contStr[cnp ? cnp - 1 : 0] = 0;
                    char retnStr[64]; int rtp = 0;
                    for (int k = 0; k < baseNIn && rtp < 60; ++k)
                        rtp += std::snprintf(retnStr + rtp, sizeof(retnStr) - rtp,
                                            "%.3f ", diagRetained[k]);
                    if (rtp == 0) { retnStr[0] = '-'; retnStr[1] = 0; }
                    else retnStr[rtp ? rtp - 1 : 0] = 0;
                    std::fprintf(stderr,
                        "PYDIAG line=%d h=%d xi=%d vstep=%d combOK=%d "
                        "1Dexcl=%d nCand=%d roster=[%s] clean=[%s] "
                        "ccEv=[%s] imgPref=[%s] cont=[%s] retn=[%s] "
                        "selfA=%.2f nbrA=%.2f decA=%.2f winPlane=%d "
                        "winTop=%.2f coarseIRE=%.2f combMiddle=%.2f "
                        "combPlatformResidual=%.2f ccRaw=%.2f ccRet=%.2f "
                        "monoTop=%.2f combTop=%.2f retrTop=%.2f nativeTop=%.2f "
                        "oneDTop=%.2f retnTop=%.2f "
                        "retnImg=%.3f retnCcEv=%.2f "
                        "nbrImgTop=[%s]\n",
                        line, h, xi, verticalStep, combOK ? 1 : 0,
                        (combOK && oneDRow) ? 1 : 0, nCand, roster, cleanStr,
                        ccEvStr, imgStr, contStr, retnStr,
                        selfAnchor / irescale, diagNeighborAnchor / irescale,
                        diagDecisionAnchor / irescale, winnerPlane,
                        resultHF / irescale, coarse / irescale,
                        combMiddle / irescale, combPlatformResidual,
                        ccMeasuredHere, ccReturn, monoTop, combTop, retrTop,
                        nativeTop, oneDTop,
                        retnTop, returnedImagePref, returnedCcEvidence, nbr);
                }
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

        if (showMap) {
            std::fill(w2d_frame_weight[line].begin(),
                      w2d_frame_weight[line].end(), 0.0f);
        }
    }

    if (consDump && consN > 0) {
        std::fprintf(stderr,
            "[CONS] Y vs (raw-carrierComp): rms=%.3f IRE max=%.2f IRE  "
            "plane0 n=%lld rms=%.3f | other n=%lld rms=%.3f (%.1f%% of px)\n",
            std::sqrt(consSum / consN) * invIreScale,
            consMax * invIreScale,
            consP0N, consP0N ? std::sqrt(consP0Sum / consP0N) * invIreScale : 0.0,
            consOtherN, consOtherN ? std::sqrt(consOtherSum / consOtherN) * invIreScale : 0.0,
            100.0 * (double)consOtherN / (double)consN);
    }

    if (dumpHull) {
        auto pct = [](long long a, long long b) {
            return b > 0 ? 100.0 * static_cast<double>(a)
                                 / static_cast<double>(b)
                         : 0.0;
        };
        std::fprintf(stderr,
            "[YHULL] tested=%lld over=%lld(%.4f%%) sumOverIRE=%.1f maxOverIRE=%.2f\n",
            hullTotal, hullOver, pct(hullOver, hullTotal),
            hullOverSumIRE, hullMaxOvershootIRE);
        std::fprintf(stderr,
            "[YHULL] over-by-plane comb=%lld retracted=%lld oneD=%lld "
            "returned=%lld blend=%lld clamped=%lld | "
            "sumMismatchIRE=%.1f maxMismatchIRE=%.2f\n",
            hullOverByPlane[0], hullOverByPlane[1], hullOverByPlane[3],
            hullOverByPlane[4], hullOverByPlane[5], hullOverByPlane[6],
            hullMismatchSumIRE, hullMaxMismatchIRE);
    }

    if (dumpRetr) {
        auto pct = [](long long a, long long b) {
            return b > 0 ? 100.0 * static_cast<double>(a)
                                 / static_cast<double>(b) : 0.0;
        };
        std::fprintf(stderr,
            "[RETR] wins=%lld passthrough=%lld(%.2f%%) meanLicenseOnPass=%.3f\n",
            retrWinTotal, retrPassthroughTotal,
            pct(retrPassthroughTotal, retrWinTotal),
            retrPassthroughTotal > 0
                ? retrLicenseSumOnPass / (double)retrPassthroughTotal : 0.0);
        std::fprintf(stderr,
            "[RETR] all-wins   bySchedule unres=%lld legal=%lld illegal=%lld\n",
            retrWinBySchedule[0][0] + retrWinBySchedule[0][1],
            retrWinBySchedule[1][0] + retrWinBySchedule[1][1],
            retrWinBySchedule[2][0] + retrWinBySchedule[2][1]);
        std::fprintf(stderr,
            "[RETR] passthrough bySchedule unres=%lld legal=%lld illegal=%lld"
            " <- LEGAL here = passthrough failure; ILLEGAL = real grid luma\n",
            retrWinBySchedule[0][1], retrWinBySchedule[1][1],
            retrWinBySchedule[2][1]);
        std::fprintf(stderr,
            "[RETR] bright(>100IRE) total=%lld bySchedule unres=%lld legal=%lld"
            " illegal=%lld <- the specks the eye spots\n",
            retrBrightTotal, retrBrightBySchedule[0],
            retrBrightBySchedule[1], retrBrightBySchedule[2]);
    }

    if (DsRefProbe::on())
        g_dsRefProbe.flush();
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

    for (int line = firstLine; line < lastLine; ++line) {
        const double *apMean = lockedApertureMean_line(line);
        if (!apMean) continue;
        detectLurchSteps(apMean, width - 3, irescale, invIreScale,
                         lurchStepRuns[line]);
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
                         const LurchStepRun &run, double &edgeOut) -> bool {
        double bestD = kLurchMatchPx;
        bool found = false;
        for (const LurchStepRun &o : nbr) {
            if (o.suppressed) continue;
            if ((o.stepSamples > 0.0) != (run.stepSamples > 0.0)) continue;
            const double d = std::fabs(o.edge - run.edge);
            if (d <= bestD) { bestD = d; edgeOut = o.edge; found = true; }
        }
        return found;
    };

    for (LurchStepRun &run : runs) {
        if (run.suppressed) continue;
        double eu, ed;
        if (!matchEdge(up, run, eu) || !matchEdge(dn, run, ed))
            continue;                       // no full vertical company
        const double e = run.edge;
        // median of three, by selection
        run.edge = std::max(std::min(eu, e), std::min(std::max(eu, e), ed));
    }
    return runs;
}

// Reverse-engineering stats for the pair disentangle (LDCD_PROBE_DISENT=1).
// Measurement only. The per-class map is the point: the leak's demod hue is
// set by (x mod 4, lineFlip), so the SAME luma edge presents a different
// error per line class -- Y colliding with different conventions along the
// phase sequence. These counters expose that map and the field asymmetry.
namespace {

struct DisentProbe {
    std::mutex mu;
    long nRuns = 0, nNoPartner = 0;
    long partnerUp[2] = {0, 0}, partnerDn[2] = {0, 0};
    // [line parity][lineFlip < 0]: samples and IRE sums.
    long   cn[2][2]   = {{0, 0}, {0, 0}};
    double cAbs[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    double cI[2][2]   = {{0.0, 0.0}, {0.0, 0.0}};
    double cQ[2][2]   = {{0.0, 0.0}, {0.0, 0.0}};
    // Scale law: per-run (step height h, footprint peak |cm|) regression.
    long   nScale = 0;
    double sH = 0.0, sP = 0.0, sHP = 0.0, sHH = 0.0;
    long peakHist[5] = {0, 0, 0, 0, 0};   // <1, 1-2, 2-4, 4-8, >=8 IRE
    // Shape conformance: per-run corr r of measured cm against the
    // anticipated doublet, and amplitude ratio beta where the shape holds.
    long rHist[5] = {0, 0, 0, 0, 0};      // <0, 0-.3, .3-.6, .6-.8, >=.8
    long   nBeta = 0;
    double sBeta = 0.0;

    static bool on()
    {
        static const bool v = []{
            const char *s = std::getenv("LDCD_PROBE_DISENT");
            return s && std::atoi(s) != 0;
        }();
        return v;
    }

    void run(int parity, int partnerDelta)
    {
        if (!on()) return;
        std::lock_guard<std::mutex> lk(mu);
        ++nRuns;
        if (partnerDelta < 0) ++partnerUp[parity & 1];
        else                  ++partnerDn[parity & 1];
    }

    void noPartner(long gatedRuns)
    {
        if (!on()) return;
        std::lock_guard<std::mutex> lk(mu);
        nRuns += gatedRuns;
        nNoPartner += gatedRuns;
    }

    void sample(int parity, int flipNeg, double cmIRE, double iIRE, double qIRE)
    {
        if (!on()) return;
        std::lock_guard<std::mutex> lk(mu);
        ++cn[parity & 1][flipNeg & 1];
        cAbs[parity & 1][flipNeg & 1] += cmIRE;
        cI[parity & 1][flipNeg & 1]   += iIRE;
        cQ[parity & 1][flipNeg & 1]   += qIRE;
    }

    void runProfile(double hIRE, double peakIRE, double r, double beta,
                    bool shapeValid)
    {
        if (!on()) return;
        std::lock_guard<std::mutex> lk(mu);
        ++nScale;
        sH += hIRE; sP += peakIRE; sHP += hIRE * peakIRE; sHH += hIRE * hIRE;
        peakHist[peakIRE < 1.0 ? 0 : peakIRE < 2.0 ? 1
               : peakIRE < 4.0 ? 2 : peakIRE < 8.0 ? 3 : 4]++;
        if (shapeValid) {
            rHist[r < 0.0 ? 0 : r < 0.3 ? 1 : r < 0.6 ? 2 : r < 0.8 ? 3 : 4]++;
            if (r >= 0.6) { ++nBeta; sBeta += beta; }
        }
    }

    ~DisentProbe()
    {
        if (!on() || nRuns <= 0) return;
        std::fprintf(stderr,
            "\n[DISENT] runs %ld  noPartner %.1f%%  partner up/dn: "
            "even %ld/%ld  odd %ld/%ld\n",
            nRuns, 100.0 * (double)nNoPartner / (double)nRuns,
            partnerUp[0], partnerDn[0], partnerUp[1], partnerDn[1]);
        for (int p = 0; p < 2; ++p)
            for (int f = 0; f < 2; ++f) {
                if (cn[p][f] <= 0) continue;
                const double inv = 1.0 / (double)cn[p][f];
                std::fprintf(stderr,
                    "[DISENT] class parity=%d flip=%c: n=%ld  |cm| %.2f IRE"
                    "  |I| %.2f  |Q| %.2f\n",
                    p, f ? '-' : '+', cn[p][f],
                    cAbs[p][f] * inv, cI[p][f] * inv, cQ[p][f] * inv);
            }
        if (nScale > 1) {
            const double det = nScale * sHH - sH * sH;
            const double slope = (det > 1e-9)
                ? (nScale * sHP - sH * sP) / det : 0.0;
            const double icept = (sP - slope * sH) / nScale;
            std::fprintf(stderr,
                "[DISENT] scale: n=%ld  peak|cm| = %.3f*h %+.2f IRE  "
                "(h mean %.1f, peak mean %.2f)\n",
                nScale, slope, icept, sH / nScale, sP / nScale);
            std::fprintf(stderr,
                "[DISENT] peak|cm| IRE: <1:%.1f%% 1-2:%.1f%% 2-4:%.1f%% "
                "4-8:%.1f%% >=8:%.1f%%\n",
                100.0 * peakHist[0] / nScale, 100.0 * peakHist[1] / nScale,
                100.0 * peakHist[2] / nScale, 100.0 * peakHist[3] / nScale,
                100.0 * peakHist[4] / nScale);
            long nR = 0;
            for (int i = 0; i < 5; ++i) nR += rHist[i];
            if (nR > 0)
                std::fprintf(stderr,
                    "[DISENT] shape r: <0:%.1f%% 0-.3:%.1f%% .3-.6:%.1f%% "
                    ".6-.8:%.1f%% >=.8:%.1f%%   beta(r>=.6) %.2f (n=%ld)\n",
                    100.0 * rHist[0] / nR, 100.0 * rHist[1] / nR,
                    100.0 * rHist[2] / nR, 100.0 * rHist[3] / nR,
                    100.0 * rHist[4] / nR,
                    nBeta ? sBeta / nBeta : 0.0, nBeta);
        }
    }
};

DisentProbe g_disentProbe;

// Downstream-fate stats (LDCD_PROBE_EDGEFATE=1). Measurement only. Every
// render judged so far was ntsc1d -- pure 1D, no comb ever touched the edge
// bands. This probe asks what the REAL pipeline does at lurch footprints:
// how far 2D moves the carrier off its 1D source there (vs a control of all
// other pixels), how often it effectively passes 1D through (the fallback),
// and what 3D adds. The answer decides WHERE edge evidence should be
// delivered: into the fallback conditioning, into the comb's own gates, or
// into the election.
struct EdgeFateProbe {
    std::mutex mu;
    // [isEdgeFootprint]: pixels, |clp1-clp0| and |clp2-clp1| sums (IRE),
    // and counts of |clp1-clp0| under the pass-through thresholds.
    long   n[2]     = {0, 0};
    double d21[2]   = {0.0, 0.0};
    double d32[2]   = {0.0, 0.0};
    long   nEx[2]   = {0, 0};      // < 0.1 IRE: effectively 1D
    long   nNear[2] = {0, 0};      // < 0.5 IRE
    bool   have3D   = false;

    static bool on()
    {
        static const bool v = []{
            const char *s = std::getenv("LDCD_PROBE_EDGEFATE");
            return s && std::atoi(s) != 0;
        }();
        return v;
    }

    void merge(const long *ln, const double *ld21, const double *ld32,
               const long *lex, const long *lnear, bool dims3)
    {
        std::lock_guard<std::mutex> lk(mu);
        for (int e = 0; e < 2; ++e) {
            n[e] += ln[e]; d21[e] += ld21[e]; d32[e] += ld32[e];
            nEx[e] += lex[e]; nNear[e] += lnear[e];
        }
        have3D = have3D || dims3;
    }

    ~EdgeFateProbe()
    {
        if (!on() || (n[0] + n[1]) <= 0) return;
        static const char *kName[2] = {"control", "edge   "};
        std::fprintf(stderr, "\n");
        for (int e = 1; e >= 0; --e) {
            if (n[e] <= 0) continue;
            const double inv = 1.0 / (double)n[e];
            std::fprintf(stderr,
                "[EDGEFATE] %s n=%ld  |2D-1D| %.3f IRE  1D-passthru "
                "<0.1:%.1f%% <0.5:%.1f%%%s%.3f IRE\n",
                kName[e], n[e], d21[e] * inv,
                100.0 * (double)nEx[e] * inv,
                100.0 * (double)nNear[e] * inv,
                have3D ? "  |3D-2D| " : "  (no 3D) ",
                have3D ? d32[e] * inv : 0.0);
        }
    }
};

EdgeFateProbe g_edgeFate;

// Post-law hull-violation stats (LDCD_PROBE_RETRHULL=1). Measurement only.
// Pass 1 clamps the carrier fit into the residual-consensus feasible range
// (region-pure four-view complements + the rolling witness), but the encoder
// bandwidth law is imposed AFTER that clamp at publication, and its 9-tap
// FIR mixes neighbours -- the published fit can leave the per-sample range
// the clamp enforced. This measures how often and by how much, deciding
// whether re-imposing the hull after the law is load-bearing or insurance.
struct RetrHullProbe {
    std::mutex mu;
    long   n = 0, nOut = 0;
    double sumExcess = 0.0, maxExcess = 0.0, sumWidth = 0.0;

    static bool on()
    {
        static const bool v = []{
            const char *s = std::getenv("LDCD_PROBE_RETRHULL");
            return s && std::atoi(s) != 0;
        }();
        return v;
    }

    void sample(double fit, double lo, double hi, double invIre)
    {
        std::lock_guard<std::mutex> lk(mu);
        ++n;
        sumWidth += (hi - lo) * invIre;
        double ex = 0.0;
        if (fit < lo)      ex = (lo - fit) * invIre;
        else if (fit > hi) ex = (fit - hi) * invIre;
        if (ex > 0.0) {
            ++nOut;
            sumExcess += ex;
            maxExcess = std::max(maxExcess, ex);
        }
    }

    ~RetrHullProbe()
    {
        if (!on() || n <= 0) return;
        std::fprintf(stderr,
            "\n[RETRHULL] n=%ld  outside %.1f%%  excess mean %.3f IRE "
            "(of violators)  max %.2f  hull width mean %.2f IRE\n",
            n, 100.0 * (double)nOut / (double)n,
            nOut ? sumExcess / nOut : 0.0, maxExcess,
            sumWidth / n);
    }
};

RetrHullProbe g_retrHull;

} // namespace

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
void Comb::FrameBuffer::probeEdgePairClassMap(int line)
{
    if (!DisentProbe::on()) return;

    const double *apMean = lockedApertureMean_line(line);
    if (!apMean) return;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int fullWidth = videoParameters.fieldWidth;
    const int width     = right - left;
    if (width < 16) return;

    const std::vector<LurchStepRun> &runs = lurchStepRuns_line(line);
    long gated = 0;
    for (const LurchStepRun &run : runs)
        if (!run.suppressed && run.gate > 0.0) ++gated;
    if (gated == 0) return;

    // Complement partner: the schedule tiles [+ - - +] down the frame, so
    // exactly one adjacent frame line carries the opposite signed class.
    // The grammar names it; nothing is presumed from the schedule.
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int want = (carrierSignedSampleClass(line, left) + 2) & 3;
    int partner = -1;
    if (line - 1 >= firstLine
            && carrierSignedSampleClass(line - 1, left) == want)
        partner = line - 1;
    else if (line + 1 < lastLine
            && carrierSignedSampleClass(line + 1, left) == want)
        partner = line + 1;
    if (partner < 0) {
        g_disentProbe.noPartner(gated);
        return;
    }

    const auto sstep = [](double t) {
        t = std::clamp(t, 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    };

    const quint16 *rawLine = rawbuffer.data() + size_t(line)    * fullWidth;
    const quint16 *rawP    = rawbuffer.data() + size_t(partner) * fullWidth;
    auto bpAt = [&](const quint16 *rl, int x) {
        const int m2 = std::clamp(x - 2, 0, width - 1);
        const int p2 = std::clamp(x + 2, 0, width - 1);
        return 0.50 * (double)rl[left + x]
             - 0.25 * ((double)rl[left + m2] + (double)rl[left + p2]);
    };

    // Footprint weights: max-combine overlapping runs so each sample is
    // counted once in the class-map stats regardless of run overlap.
    std::vector<double> wAcc(width, 0.0);
    bool any = false;
    for (const LurchStepRun &run : runs) {
        if (run.suppressed || run.gate <= 0.0) continue;
        g_disentProbe.run(line & 1, partner - line);
        const int x0 = (int)std::floor(run.edge) - 6;
        const int x1 = (int)std::ceil(run.edge) + 6;
        for (int x = std::max(0, x0); x <= std::min(width - 1, x1); ++x) {
            const double tIn =
                std::min((double)(x - x0), (double)(x1 - x)) / 3.0;
            const double w = run.gate * sstep(tIn);
            if (w > wAcc[x]) { wAcc[x] = w; any = true; }
        }

        // Per-run error profile against the ANTICIPATED doublet: a chain-
        // sharp (w = 2) ramp with lurch's pinned plateau levels at lurch's
        // edge, through the bandpass's own operator -0.25*D2_2. r says
        // whether the anticipation carries the SHAPE (evidence can predict
        // the waveform) or only a location and magnitude bound; beta says
        // whether lurch's amplitude scales it correctly where the shape
        // holds. peak|cm| vs step height h feeds the scale law.
        const int xa = std::max(0, x0), xb = std::min(width - 1, x1);
        if (xb - xa < 6) continue;
        const double lo = apMean[std::clamp(run.a, 0, width - 1)];
        const double hi = apMean[std::clamp(run.b + 1, 0, width - 1)];
        auto rampAt = [&](double j) {
            return lo + (hi - lo)
                 * std::clamp((j - run.edge) / 2.0 + 0.5, 0.0, 1.0);
        };
        double peak = 0.0, sCC = 0.0, sPP = 0.0, sCP = 0.0;
        for (int x = xa; x <= xb; ++x) {
            const double cm = 0.5 * (bpAt(rawLine, x) + bpAt(rawP, x));
            const double d2 = rampAt((double)(x - 2)) - 2.0 * rampAt((double)x)
                            + rampAt((double)(x + 2));
            const double pred = -0.25 * d2;
            peak = std::max(peak, std::fabs(cm));
            sCC += cm * cm; sPP += pred * pred; sCP += cm * pred;
        }
        const bool shapeValid = sPP > 1e-9 && sCC > 1e-9;
        const double r    = shapeValid ? sCP / std::sqrt(sCC * sPP) : 0.0;
        const double beta = (sPP > 1e-9) ? sCP / sPP : 0.0;
        g_disentProbe.runProfile(run.stepAbsIRE, peak * invIreScale,
                                 r, beta, shapeValid);
    }
    if (!any) return;

    const int parity  = line & 1;
    const int flipNeg = carrierLineFlip(line) < 0 ? 1 : 0;
    for (int x = 0; x < width; ++x) {
        if (wAcc[x] <= 0.0) continue;
        const double cm = 0.5 * (bpAt(rawLine, x) + bpAt(rawP, x));
        const int ph = carrierSampleClass(line, left + x);
        g_disentProbe.sample(parity, flipNeg,
            std::fabs(cm) * invIreScale,
            std::fabs(2.0 * cm * sin4fsc(ph)) * invIreScale,
            std::fabs(2.0 * cm * cos4fsc(ph)) * invIreScale);
    }
}

// Off-grid leakage of the published carriers (LDCD_PROBE_OFFGRID).
// Measurement only. See OffGridProbe for what the numbers decide.
void Comb::FrameBuffer::probeOffGrid()
{
    if (!OffGridProbe::on()) return;

    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int width     = right - left;
    if (width < 8 || firstLine >= lastLine) return;

    const double floorTot = (0.5 * irescale) * (0.5 * irescale) * 4.0;

    for (int line = firstLine; line < lastLine && line < demodLines; ++line) {
        const CombCarrierGrammar *grammar = carrierGrammarLine(line);
        if (!grammar || !grammar->grammarLocked) continue;
        double bI4[4], bQ4[4];
        for (int p = 0; p < 4; ++p) {
            bI4[p] = (double)grammar->demodLUTTi[p];
            bQ4[p] = (double)grammar->demodLUTTq[p];
        }

        const float  *fitRow  = carrierRetractedValid
            ? carrierFit_line(line) : nullptr;
        const double *combRow = lockedCarrierComposite_line(line);
        const double *oneDRow = locked1DSource_line(line);
        const lddecode::CarrierAnalysisRecord *anRow =
            carrierAnalysis_line(line);

        const double strongIRE = 8.0;
        double lastPhase[3];
        bool   havePhase[3] = {false, false, false};

        for (int x = 0; x + 3 < width; x += 2) {
            double Sii = 0, Siq = 0, Sqq = 0;
            for (int k = 0; k < 4; ++k) {
                const int cls = carrierSampleClass(line, left + x + k);
                Sii += bI4[cls] * bI4[cls];
                Siq += bI4[cls] * bQ4[cls];
                Sqq += bQ4[cls] * bQ4[cls];
            }
            const double det = Sii * Sqq - Siq * Siq;
            if (std::fabs(det) < 1e-12) continue;

            double phase[3];
            bool   strong[3] = {false, false, false};
            double amp[3] = {0.0, 0.0, 0.0};
            auto solve = [&](int src, auto at) {
                double SiY = 0, SqY = 0, Stt = 0;
                for (int k = 0; k < 4; ++k) {
                    const int cls = carrierSampleClass(line, left + x + k);
                    const double v = at(x + k);
                    SiY += bI4[cls] * v; SqY += bQ4[cls] * v; Stt += v * v;
                }
                amp[src] = std::sqrt(Stt * 0.25) * invIreScale;
                if (Stt < floorTot) return;
                const double p = ( Sqq * SiY - Siq * SqY) / det;
                const double q = (-Siq * SiY + Sii * SqY) / det;
                const double off = std::max(0.0, Stt - (p * SiY + q * SqY));
                g_offGrid.sample(src, amp[src] < 5.0 ? 0 : amp[src] < 15.0 ? 1 : 2,
                                 off, Stt);
                if (amp[src] >= strongIRE) {
                    phase[src] = std::atan2(q, p);
                    strong[src] = true;
                }
            };
            if (fitRow)  solve(0, [&](int xi) { return (double)fitRow[xi]; });
            if (combRow) solve(1, [&](int xi) { return combRow[xi]; });
            if (oneDRow) solve(2, [&](int xi) { return oneDRow[xi]; });

            auto wrapPi = [](double a) {
                while (a >  M_PI) a -= 2.0 * M_PI;
                while (a < -M_PI) a += 2.0 * M_PI;
                return a;
            };
            if (strong[0] && strong[1])
                g_offGrid.phasePair(wrapPi(phase[0] - phase[1]));
            if (fitRow && combRow && amp[1] >= strongIRE) {
                const bool provenIllegal = anRow &&
                    lddecode::carrierIllegalProof(
                        (double)anRow[x + 1].carrierConformance,
                        (double)anRow[x + 1].conformanceSupportFraction)
                        >= 0.7;
                g_offGrid.dropout(amp[0] < 0.5 * amp[1], provenIllegal);
            }
            for (int s = 0; s < 3; ++s) {
                if (!strong[s]) { havePhase[s] = false; continue; }
                if (havePhase[s])
                    g_offGrid.jitter(s, wrapPi(phase[s] - lastPhase[s]));
                lastPhase[s] = phase[s];
                havePhase[s] = true;
            }
        }
    }
}

// Downstream fate of the edge footprints (LDCD_PROBE_EDGEFATE). Measurement
// only -- reads the finished scalar planes after 2D/3D have run, writes
// nothing. See EdgeFateProbe above for what the numbers decide.
void Comb::FrameBuffer::probeEdgeFate(int dimensions)
{
    if (!EdgeFateProbe::on()) return;

    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int width     = right - left;
    if (width < 16 || firstLine >= lastLine) return;

    const bool dims3 = dimensions == 3;
    long   ln[2]    = {0, 0};
    double ld21[2]  = {0.0, 0.0};
    double ld32[2]  = {0.0, 0.0};
    long   lex[2]   = {0, 0};
    long   lnear[2] = {0, 0};

    std::vector<std::uint8_t> mask(width);
    for (int line = firstLine; line < lastLine; ++line) {
        const double *c0 = clpbuffer[0].pixel[line];
        const double *c1 = clpbuffer[1].pixel[line];
        const double *c2 = dims3 ? clpbuffer[2].pixel[line] : nullptr;
        if (!c0 || !c1) continue;

        const std::vector<LurchStepRun> &runs = lurchStepRuns_line(line);
        std::fill(mask.begin(), mask.end(), 0);
        for (const LurchStepRun &run : runs) {
            if (run.suppressed || run.gate <= 0.0) continue;
            const int x0 = std::max(0, (int)std::floor(run.edge) - 6);
            const int x1 = std::min(width - 1, (int)std::ceil(run.edge) + 6);
            for (int x = x0; x <= x1; ++x) mask[x] = 1;
        }

        for (int x = 0; x < width; ++x) {
            const int e = mask[x];
            const double d21 =
                std::fabs(c1[left + x] - c0[left + x]) * invIreScale;
            ++ln[e];
            ld21[e] += d21;
            if (c2)
                ld32[e] += std::fabs(c2[left + x] - c1[left + x]) * invIreScale;
            if (d21 < 0.1) ++lex[e];
            if (d21 < 0.5) ++lnear[e];
        }
    }
    g_edgeFate.merge(ln, ld21, ld32, lex, lnear, dims3);
}

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

// Build the carrier-retracted view from the shared carrier model.
//
// The locked orchestration is intentionally single-pass here:
//   1. buildCarrierAnalysis() harvests canonical bandpass and schedule
//      conformance data.
//   2. buildPhaseCorrected1D() builds the corrected 1D baseline.
//   3. buildCarrierRetracted() calls buildCarrierRetractionStage(false)
//      once, after the corrected 1D baseline exists.
//
// buildCarrierRetractionStage() then performs four-view carrier/Y attribution,
// line-to-line cancellation on carrierFit, and raw - combedCarrier to produce
// the flattened carrier-retracted view.
void Comb::FrameBuffer::buildCarrierRetracted()
{
    buildCarrierRetractionStage(false);
}

void Comb::FrameBuffer::buildCarrierRetractionStage(bool analysisOnly)
{
    carrierRetractedValid = false;

    if (!configuration.phaseCompensation ||
        !configuration.lumaWitness)
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
    if (carrierRetracted_flat.size() < need)
        carrierRetracted_flat.assign(need, 0.0f);
    if (flatFloor_flat.size() < need)
        flatFloor_flat.assign(need, 0.0f);
    if (combedCarrier_flat.size() < need)
        combedCarrier_flat.assign(need, 0.0f);
    if (carrierCorroboration_flat.size() < need)
        carrierCorroboration_flat.assign(need, 0.0f);
    if (carrierEligibility_flat.size() < need)
        carrierEligibility_flat.assign(need, 0.0f);
    if (coarseYEvidence_flat.size() < need)
        coarseYEvidence_flat.assign(need, lddecode::FourViewPixelEvidence{});
    if (carrierImpurity_flat.size() < need)
        carrierImpurity_flat.assign(need, 0.0f);
    if (carrierAnalysis_flat.size() < need)
        return; // shared analysis must already have been produced

    // --- Disposable dead-zone instrumentation (env-gated). Set
    // LDCD_DUMP_DEADZONE=1 to print the amplified footprint of the schedule
    // DQ downstream: how many pixels lose their carrier estimate, split by
    // cause. Zero cost when unset. Remove with the rethink. ---
    const bool dumpDead = std::getenv("LDCD_DUMP_DEADZONE") != nullptr;
    long long dzActive = 0, dzIneligible = 0;
    long long dzDead = 0, dzDeadIllegal = 0, dzDeadFitStarved = 0;
    long long dzWinTotal = 0, dzWinInvalid = 0, dzWinRank = 0, dzWinDet = 0;

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
            std::max(24.0, grammar->carrierScale * 5.0) * irescale;

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
            for (int xi = 0; xi < width; ++xi) {
                const int s0 = std::clamp(xi - 2, 0, meanCount - 1);
                const int s1 = std::clamp(xi - 1, 0, meanCount - 1);
                refinedY[xi] = 0.5 * (winFloor[s0] + winFloor[s1]);
            }

            // Lurch preconditioner: sharpen the prior before the carrier fit
            // consumes it, so step energy stays out of raw - refinedY and
            // never enters the carrier band. Canonical runs (built once in
            // split1D on the same pool), edges vertically corroborated;
            // apply-only here.
            const std::vector<LurchStepRun> corrRuns =
                corroborateLurchEdges(line);
            applyLurchSteps(corrRuns, winFloor.data(),
                            meanCount, width, 1.0, refinedY, nullptr);

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

                if (dumpDead) {
                    ++dzWinTotal;
                    if (!fitValid) {
                        ++dzWinInvalid;
                        if (sampleWeight < 2.5)
                            ++dzWinRank;   // starved of participating weight
                        else
                            ++dzWinDet;    // killed by singular normal matrix
                    }
                }

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

                if (dumpDead) {
                    ++dzActive;
                    if (partWeight[xi] <= 0.0)
                        ++dzIneligible;
                    if (viewCount == 0) {
                        ++dzDead;
                        // Pass 1 harvests any covering winFitValid window, so
                        // viewCount==0 means the fit was starved (no covering
                        // window survived weight/det).
                        if (partWeight[xi] <= 0.0)
                            ++dzDeadIllegal;
                        else
                            ++dzDeadFitStarved;
                    }
                }

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

                // ---- Stage-0 line-solve harness dump (env-gated, disposable).
                // Emits everything the offline event solver needs, at the one
                // point where the four-view feasible band is already built:
                //   raw, bp        -> notch = raw-bp, and the leak lives in bp
                //   lurch          -> carrier-free luma-motion evidence
                //   conf/sched     -> interline grammar (luma by law)
                //   yFloor[0..3]   -> the FOUR LEGAL VIEWS as a feasible band
                //                     (constraints, never collapsed here)
                // LDCD_DUMP_SOLVE_L / _C0 / _C1.  Run -t 1.
                {
                    static const int slL  = []{ const char *s=std::getenv("LDCD_DUMP_SOLVE_L");  return s?std::atoi(s):-1; }();
                    static const int slC0 = []{ const char *s=std::getenv("LDCD_DUMP_SOLVE_C0"); return s?std::atoi(s):-1; }();
                    static const int slC1 = []{ const char *s=std::getenv("LDCD_DUMP_SOLVE_C1"); return s?std::atoi(s):-1; }();
                    if (line == slL && xi >= slC0 && xi <= slC1) {
                        const double *bpL = locked1DRawBandpass_line(line);
                        const double rawI = (double)rawLine[left + xi] * invIreScale;
                        const double bpI  = bpL ? bpL[xi] * invIreScale : 0.0;
                        const double *apM = lockedApertureMean_line(line);
                        const double *clk = lockedCornerLeak_line(line);
                        std::fprintf(stderr, "[LEAK] h=%d leak=%.4f\n",
                                     left + xi, clk ? clk[xi] * invIreScale : 0.0);
                        std::fprintf(stderr,
                            "[SOLVE] h=%d raw=%.3f bp=%.3f am0=%.3f am1=%.3f "
                            "am2=%.3f am3=%.3f lurch=%.3f conf=%+.2f "
                            "supp=%.2f nviews=%d f0=%.2f f1=%.2f f2=%.2f f3=%.2f\n",
                            left + xi, rawI, bpI,
                            apM ? apM[std::max(0,xi-3)] * invIreScale : 0.0,
                            apM ? apM[std::max(0,xi-2)] * invIreScale : 0.0,
                            apM ? apM[std::max(0,xi-1)] * invIreScale : 0.0,
                            apM ? apM[xi] * invIreScale : 0.0,
                            (double)analysisRow[xi].residual.maxAbsMembershipIRE,
                            (double)analysisRow[xi].carrierConformance,
                            (double)analysisRow[xi].conformanceSupportFraction,
                            viewCount,
                            viewCount > 0 ? evidenceRow[xi].views[0].yFloor * invIreScale : 0.0f,
                            viewCount > 1 ? evidenceRow[xi].views[1].yFloor * invIreScale : 0.0f,
                            viewCount > 2 ? evidenceRow[xi].views[2].yFloor * invIreScale : 0.0f,
                            viewCount > 3 ? evidenceRow[xi].views[3].yFloor * invIreScale : 0.0f);
                    }
                }

                auto parallax = lddecode::buildFourViewCarrierAttribution(
                    views,
                    viewCount,
                    invIreScale);

                double modelI = parallax.valid ? parallax.commonI : 0.0;
                double modelQ = parallax.valid ? parallax.commonQ : 0.0;

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
                // Measured before removal (LDCD_PROBE_OFFGRID): the fit's
                // span, rotation, and jitter were already the cleanest in
                // the tree -- nothing real leaned on either operation.
                double cf = modelI * basisI[xi] + modelQ * basisQ[xi];
                if (residualTightenSupport > 0.0)
                    cf = std::clamp(cf, residualCarrierLo, residualCarrierHi);
                cf = std::clamp(cf, -maxCarrierSamples, maxCarrierSamples);

                carrierFit[xi] = cf;
                flattened[xi] = rawWhole[xi] - cf;

                fitRow[xi] = static_cast<float>(cf);
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
                    for (int xi = 0; xi < width; ++xi) {
                        const int s = std::clamp(xi - 1, 0, width - 4);
                        double sIY = 0.0, sQY = 0.0;
                        for (int k = 0; k < 4; ++k) {
                            const double v = carrierFit[s + k];
                            sIY += basisI[s + k] * v;
                            sQY += basisQ[s + k] * v;
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

            // Post-law hull-violation probe (measurement only; inert unless
            // LDCD_PROBE_RETRHULL). Reports what the re-clamp below is about
            // to correct: how far the law's FIR moved the published fit back
            // outside the per-sample feasible range the Pass-1 clamp had
            // enforced.
            if (RetrHullProbe::on()) {
                for (int xi = 0; xi < width; ++xi) {
                    const auto &pp = analysisRow[xi].parallax;
                    if (!pp.residualValid) continue;
                    g_retrHull.sample(carrierFit[xi],
                                      static_cast<double>(pp.residualLo),
                                      static_cast<double>(pp.residualHi),
                                      invIreScale);
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

    if (analysisOnly)
        return;
    if (!carrierRetractionModelValid)
        return;

    // ---------------------------------------------------------------
    // Pass 1.75: vertical amplitude continuity for the fit (the beam fix).
    //
    // Measured (LDCD_PROBE_OFFGRID): on LEGAL carrier the fit holds lock on
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

        long bridgedWindows = 0;
        auto bridgeLine = [&](int line) {
            const CombCarrierGrammar *gL = carrierGrammarLine(line);
            if (!gL || !gL->grammarLocked) return;
            const double *bpL = locked1DRawBandpass_line(line);
            float *fitL = carrierFit_flat.data()
                          + static_cast<size_t>(line) * demodWidth;
            const auto *anL = carrierAnalysis_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
            const double maxCarrierL =
                std::max(24.0, gL->carrierScale * 5.0) * irescale;
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
                ++bridgedWindows;
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
        if (bridgeOn && std::getenv("LDCD_PROBE_OFFGRID"))
            std::fprintf(stderr, "[BRIDGE] windows adopted: %ld\n",
                         bridgedWindows);
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
    // ---------------------------------------------------------------
    auto softReachGate = [](double diffIRE, double softIRE, double hardIRE) {
        if (diffIRE <= softIRE)
            return 1.0;
        if (diffIRE >= hardIRE)
            return 0.0;
        const double t = (diffIRE - softIRE) /
                         std::max(1e-9, hardIRE - softIRE);
        return 1.0 - (t * t * (3.0 - 2.0 * t));
    };

    // Diagnostic gate-chain dump (evidence-only, same family as LDCD_YVIEW).
    // LDCD_PASS2_DUMP=<prefix> writes every per-pixel term of the Pass-2
    // engage/disengage decision to <prefix>_NNN.bin, one file per frame
    // processed, so the mechanism can be studied offline instead of inferred
    // from the rendered output.  Channel plan (float32, [line][ch][x]):
    //   0 eligCenter    1 legCount(elig at xi)  2 legMask(b0 up2,b1 up1,
    //                                             b2 dn1,b3 dn2)
    //   3 gUp2raw       4 gUp1raw       5 gDn1raw       6 gDn2raw
    //   7 wSumRaw       8 neighborFit   9 strength(min(1,wSum))
    //  10 ownedFallback 11 corrCode(+3 quiet, +4 unobservable, else signed corr)
    //  12 fitRow       13 combRow      14 corroboration(envelope-scale w)
    static const char *pass2DumpPrefix = std::getenv("LDCD_PASS2_DUMP");
    constexpr int kP2D = 15;
    std::vector<float> p2dump;
    if (pass2DumpPrefix)
        p2dump.assign(static_cast<size_t>(lastLine - firstLine) * kP2D * width,
                      0.0f);
    auto p2rec = [&](int line, int ch, int xi, double v) {
        p2dump[(static_cast<size_t>(line - firstLine) * kP2D + ch) * width +
               xi] = static_cast<float>(v);
    };

    // Per-leg raw-gate scratch, hoisted across lines.  Slot order is fixed
    // (up2, up1, dn1, dn2) so the dump channels stay identifiable; an
    // absent leg keeps null pointers and a zeroed gate row.
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
                if (pass2DumpPrefix)
                    p2rec(line, 14, xi, w);
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
            if (pass2DumpPrefix) {
                p2rec(line, 0, xi, eligRow[xi]);
                int mask = 0, count = 0;
                for (int k = 0; k < kNLegs; ++k) {
                    if (legs[k].present && legs[k].elig[xi] > 0.5f) {
                        mask |= 1 << k;
                        ++count;
                    }
                    p2rec(line, 3 + k, xi, legs[k].wRaw[xi]);
                }
                p2rec(line, 1, xi, count);
                p2rec(line, 2, xi, mask);
            }
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
            double p2corrCode = 4.0;       // dump-only: +4 = unobservable
            {
                const int w0 = std::clamp(xi - 1, 0, width - 4);
                double e0 = 0.0;
                for (int k = 0; k < 4; ++k) {
                    const double c = static_cast<double>(fitRow[w0 + k]);
                    e0 += c * c;
                }
                const double ampFloor = 3.0 * irescale;      // 3 IRE envelope
                // Energy of a carrier of envelope A over one full cycle is
                // 2*A^2 (cos^2 + sin^2 pairs), so the 3 IRE envelope floor
                // keeps its meaning on the 4-sample window.
                const double eFloor = 2.0 * ampFloor * ampFloor;
                if (e0 < eFloor) {
                    ownedFallback = 1.0;   // harmless confiscation
                    p2corrCode = 3.0;      // dump-only: +3 = quiet operand
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
                        for (int j = 0; j < 4; ++j) {
                            const double c = static_cast<double>(fitRow[w0 + j]);
                            const double n = static_cast<double>(pf[w0 + j]);
                            dot += c * n;
                            eP += n * n;
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
                        p2corrCode = sc;
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

                if (pass2DumpPrefix)
                    p2rec(line, 8, xi, neighborFit);
            } else {
                // Reach fully closed: the whole sample is un-cancelled carrier.
                // Confiscate it only where carrier is owned.
                combRow[xi] = static_cast<float>(
                    centerParticipation *
                    static_cast<double>(fitRow[xi]) * ownedFallback);
            }

            if (pass2DumpPrefix) {
                double wSumRaw = 0.0;
                for (int k = 0; k < kNLegs; ++k)
                    wSumRaw += legs[k].wRaw[xi];
                p2rec(line, 7, xi, wSumRaw);
                p2rec(line, 9, xi, std::min(1.0, wSum));
                p2rec(line, 10, xi, ownedFallback);
                p2rec(line, 11, xi, p2corrCode);
                p2rec(line, 12, xi, static_cast<double>(fitRow[xi]));
                p2rec(line, 13, xi, static_cast<double>(combRow[xi]));
            }
        }
    }

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

    if (pass2DumpPrefix) {
        static std::atomic<int> p2DumpCounter{0};
        const int n = p2DumpCounter.fetch_add(1);
        char path[512];
        std::snprintf(path, sizeof(path), "%s_%03d.bin", pass2DumpPrefix, n);
        if (FILE *fp = std::fopen(path, "wb")) {
            const qint32 hdr[6] = {0x50325644, firstLine, lastLine,
                                   width, kP2D, left};
            std::fwrite(hdr, sizeof(qint32), 6, fp);
            std::fwrite(p2dump.data(), sizeof(float), p2dump.size(), fp);
            std::fclose(fp);
            std::fprintf(stderr, "PASS2DUMP wrote %s heldSeq=%d/%d\n",
                         path, heldSeq1, heldSeq2);
        }
    }

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
    static const int retractedSource = []{
        const char *s = std::getenv("LDCD_RETRACTED_SOURCE");
        if (!s)
            return 1;
        if (s[0] == 'c')
            return 0;
        if (s[0] == 'n')
            return 1;
        if (s[0] == 'p')
            return 2;
        return 1;
    }();

    for (int line = firstLine; line < lastLine; ++line) {
        const quint16 *rawLine =
            rawbuffer.data() + static_cast<size_t>(line) * videoParameters.fieldWidth;
        const float *fitRowPub = carrierFit_flat.data()
                                 + static_cast<size_t>(line) * demodWidth;
        const float *combRowPub = combedCarrier_flat.data()
                                  + static_cast<size_t>(line) * demodWidth;
        const float *wRowPub = carrierCorroboration_flat.data()
                               + static_cast<size_t>(line) * demodWidth;
        float *retractedRow = carrierRetracted_flat.data()
                              + static_cast<size_t>(line) * demodWidth;

        for (int xi = 0; xi < width; ++xi) {
            double carrier;
            switch (retractedSource) {
            case 1:
                carrier = static_cast<double>(fitRowPub[xi]);
                break;
            case 2:
                carrier = static_cast<double>(combRowPub[xi]);
                break;
            default:
                carrier = static_cast<double>(wRowPub[xi]) *
                          static_cast<double>(fitRowPub[xi]);
                break;
            }
            retractedRow[xi] = static_cast<float>(
                static_cast<double>(rawLine[left + xi]) - carrier);
        }
    }

    carrierRetractedValid = true;

    if (dumpDead) {
        auto pct = [](long long a, long long b) {
            return b > 0 ? 100.0 * static_cast<double>(a)
                                 / static_cast<double>(b)
                         : 0.0;
        };
        std::fprintf(stderr,
            "[DEAD] active=%lld ineligible(illegal)=%lld(%.1f%%) "
            "dead(viewCount==0)=%lld(%.1f%%) [illegal=%lld fitStarved=%lld]\n",
            dzActive,
            dzIneligible, pct(dzIneligible, dzActive),
            dzDead, pct(dzDead, dzActive),
            dzDeadIllegal, dzDeadFitStarved);
        std::fprintf(stderr,
            "[DEAD] windows=%lld invalid=%lld(%.1f%%) [rank(illegal)=%lld det=%lld] "
            "amplification dead/illegal=%.2fx\n",
            dzWinTotal, dzWinInvalid, pct(dzWinInvalid, dzWinTotal),
            dzWinRank, dzWinDet,
            dzIneligible > 0 ? static_cast<double>(dzDead)
                                   / static_cast<double>(dzIneligible)
                             : 0.0);
    }
}
