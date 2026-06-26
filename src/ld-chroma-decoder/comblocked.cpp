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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <QtGlobal>

namespace {

// Carrier impurity (cross-color) detector evidence and policy, restored with
// the witness/carrier-retraction path.  detectCarrierImpurity is the
// six-evidence classifier buildCarrierRetracted() uses; carrierImpurityTransfer-
// Strength is the explicit detector->policy conversion that gates how much of
// the measured incoherent excess is moved from carrier to Y.
struct CarrierImpurityEvidence {
    double narrowMagIRE = 0.0;
    double wideMagIRE = 0.0;
    double phaseAgreement = 0.0;
    double carrierCoherence = 0.0;
    double carrierConflict = 0.0;
    double lumaMembership = 0.0;
};

inline double smoothGate01(double t)
{
    t = std::clamp(t, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

inline double detectCarrierImpurity(const CarrierImpurityEvidence &e)
{
    if (e.narrowMagIRE <= 1.5 || e.wideMagIRE >= e.narrowMagIRE)
        return 0.0;

    const double excessFraction = std::clamp(
        (e.narrowMagIRE - e.wideMagIRE) /
            std::max(1.5, e.narrowMagIRE),
        0.0,
        1.0);

    const double lumaSupport = std::max(
        std::clamp(e.lumaMembership, 0.0, 1.0),
        std::clamp(e.carrierConflict, 0.0, 1.0));

    const double coherentChromaProtect =
        std::clamp(e.phaseAgreement, 0.0, 1.0) *
        std::clamp(e.carrierCoherence, 0.0, 1.0) *
        (1.0 - std::clamp(e.carrierConflict, 0.0, 1.0));

    // Magnitude disagreement is only the invitation. Luma-membership or
    // carrier-model conflict must support it, while a coherent phase-matched
    // carrier protects legitimate chroma envelope transitions.
    const double classification = std::clamp(
        0.20 + 0.80 * lumaSupport - 0.65 * coherentChromaProtect,
        0.0,
        1.0);

    return excessFraction * classification;
}

inline double carrierImpurityTransferStrength(double impurity,
                                              double lumaMembership)
{
    // Explicit detector-to-policy conversion. The detector does not state how
    // much waveform to move; membership support controls that promotion.
    return std::clamp(
        impurity * (0.35 + 0.65 * std::clamp(lumaMembership, 0.0, 1.0)),
        0.0,
        0.85);
}

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
        for (int line = firstLine; line < lastLine; ++line) {
            const quint16 *rawLine = rawbuffer.data() + line * fullWidth;
            buildCompositeLumaDecompositionLine(rawLine, left, width,
                                                lockedLumaBaseY4_line(line),
                                                nullptr,
                                                lockedLumaSmooth_line(line));
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
    if (ccDiagLine >= 0) {
        std::fprintf(stderr,
            "CCDIAG-FB phaseComp=%d activeLines=[%d,%d) width=%d target=%d %s\n",
            configuration.phaseCompensation ? 1 : 0, first, last, width, ccDiagLine,
            (ccDiagLine >= first && ccDiagLine < last) ? "in-range" : "OUT-OF-RANGE");
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
    // The metric is published as carrierImpurity (a disqualification oracle).
    // It is NEVER applied to the carrier source; the source is emitted clean.
    // Suppression happens downstream as alpha at color demod and Y subtraction.
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

    for (int line = first; line < last; ++line) {
        const quint16 *rawLine =
            rawbuffer.data() + static_cast<size_t>(line) * fullWidth;

        double *lockedSource = locked1DSource_line(line);
        if (!lockedSource)
            continue;

        seedCombAttributionPerLine(line);
        AttributionEvidence *attribution = attributionEvidence_line(line);

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
        double remodScale[4];

        const auto remodPlan =
            lddecode::carrierGrammarCompositeRemodPlan(
                grammar, 1.0, lddecode::CarrierSignFrame::Grid4fsc);

        for (int phase = 0; phase < 4; ++phase) {
            lockedTo4fsc(
                lutI[phase], lutQ[phase], burstCos, burstSin,
                i4Scale[phase], q4Scale[phase]);

            magnitudeScale[phase] =
                std::hypot(i4Scale[phase], q4Scale[phase]);

            remodScale[phase] =
                lddecode::carrierGrammarRemod4fscToComposite(
                    remodPlan,
                    phase - remodPlan.samplePhase0,
                    i4Scale[phase],
                    q4Scale[phase]);
        }

        // ---- Per-leg remod-scale diagnostic (gated, no output) ----
        // lockedSource = source * remodScale[phase].  If remodScale is not flat
        // across the four legs, that multiply is a period-4 amplitude modulation
        // on the carrier == 2fsc checkerboard, structural to the locked path.
        static const int ccDiagRemodLine = []{
            const char *s = std::getenv("CC_DIAG_LINE"); return s ? std::atoi(s) : -1;
        }();
        if (ccDiagRemodLine >= 0 && line == ccDiagRemodLine) {
            std::fprintf(stderr,
                "CCREMOD line=%d locked=%d remod=[%.4f %.4f %.4f %.4f] "
                "mag=[%.4f %.4f %.4f %.4f] i4=[%.4f %.4f %.4f %.4f] "
                "q4=[%.4f %.4f %.4f %.4f]\n",
                line, grammarLocked ? 1 : 0,
                remodScale[0], remodScale[1], remodScale[2], remodScale[3],
                magnitudeScale[0], magnitudeScale[1], magnitudeScale[2], magnitudeScale[3],
                i4Scale[0], i4Scale[1], i4Scale[2], i4Scale[3],
                q4Scale[0], q4Scale[1], q4Scale[2], q4Scale[3]);
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

        double *bpLine = scratch_preI.data();
        double *restrainedLine = scratch_preQ.data();

        auto rawAtRel = [&](int rel) -> double {
            const int r = std::clamp(rel, 0, width - 1);
            return static_cast<double>(rawLine[left + r]);
        };

        // Pass 1: ordinary carrier-spaced 1D bandpass from raw composite.
        //
        //     bp[x] = -0.25 raw[x-2] + 0.50 raw[x] - 0.25 raw[x+2]
        for (int rel = 0; rel < width; ++rel) {
            const double c  = rawAtRel(rel);
            const double m2 = rawAtRel(rel - 2);
            const double p2 = rawAtRel(rel + 2);

            bpLine[rel] = 0.50 * c - 0.25 * (m2 + p2);
        }

        float *impurityRow = carrierImpurity_line(line);

        // Pass 2: wide-window cross-color detector.  Publishes carrierImpurity;
        // the emitted source is the clean bandpass (no gain on the carrier).
        //
        // Demodulate the bandpass into quadrature against the period-4
        // reference (for the wide coherent fit) and form the 2-sample fit
        // envelope (for the rolling narrow fit).
        for (int rel = 0; rel < width; ++rel) {
            const int p = carrierSampleClass(line, left + rel) & 3;
            demI[rel] = bpLine[rel] * cosRef[p];
            demQ[rel] = bpLine[rel] * sinRef[p];
            const int relN = std::min(rel + 1, width - 1);
            env[rel] = std::hypot(bpLine[rel], bpLine[relN]);
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
            const int a = std::clamp(center - kWideWin / 2, 0, width);
            const int b = std::clamp(a + kWideWin, 0, width);
            const double sumI = preI[b] - preI[a];
            const double sumQ = preQ[b] - preQ[a];
            const double n = static_cast<double>(std::max(1, b - a));
            return (2.0 * std::hypot(sumI, sumQ) / n) * invIreScale;
        };

        // Narrow fit: rolling, current-centered mean of the 2-sample envelope.
        // The point envelope = A on coherent carrier but ripples at 2fsc under
        // phase error; the centered mean over 4 cycles nulls that ripple.
        auto narrowEnvIRE = [&](int center) -> double {
            const int a = std::clamp(center - kNarrowWin / 2, 0, width);
            const int b = std::clamp(a + kNarrowWin, 0, width);
            const double n = static_cast<double>(std::max(1, b - a));
            return ((preEnv[b] - preEnv[a]) / n) * invIreScale;
        };

        // Aperture cross-color detector.  Publishes gA = aperture contamination
        // as carrierImpurity; the emitted source is the CLEAN, FULL bandpass.
        //
        // The correction is NOT applied to the source.  In the locked path Y is
        // raw - clpLine and (with --no-residual-color) chroma is demod(clpLine)
        // from lockedProduct.  Reducing clpLine would leave the removed residual
        // as carrier-band energy in Y == checkerboard (and in residualColor mode
        // luma and chroma are rigidly complementary, so any source correction is
        // a checkerboard by construction).  The carrier source must therefore be
        // emitted at full strength so Y = raw - full carrier has no carrier-band
        // ripple; the cross-color correction lives on the COLOR side only, as the
        // gA alpha applied in splitIQlocked().  This requires the lockedProduct
        // chroma path: run --no-residual-color.
        //
        // Winding is deliberately NOT used: control measurements on saturated
        // uniforms showed coh/turn overlap authentic chroma and contamination,
        // so winding cannot discriminate without a luma-coupling guard.  gA reads
        // 0 on the uniforms and 0.2-0.3 on title cross-color, so aperture alone
        // is the correct discriminator.
        for (int rel = 0; rel < width; ++rel) {
            // Stable centre Zwide on the cycle grid (8-cycle complex mean).
            const int wa = std::clamp(rel - kWideWin / 2, 0, width);
            const int wb = std::clamp(wa + kWideWin, 0, width);
            const double wn = std::max(1, wb - wa);
            const double ZwI = (preI[wb] - preI[wa]) / wn;
            const double ZwQ = (preQ[wb] - preQ[wa]) / wn;

            const double narrowMag = narrowEnvIRE(rel);
            const double wideMag = 2.0 * std::hypot(ZwI, ZwQ) * invIreScale;

            // Aperture contamination: clamp((narrow - wide)/max(floor, narrow)).
            double gA = 0.0;
            if (narrowMag > kImpurityFloorIRE && wideMag < narrowMag) {
                gA = clamp01(
                    (narrowMag - wideMag) /
                    std::max(kImpurityFloorIRE, narrowMag));
            }

            // Source emitted CLEAN and FULL — the detector never touches it.
            restrainedLine[rel] = bpLine[rel];

            if (impurityRow)
                impurityRow[rel] = static_cast<float>(gA);

            if (line >= 0 && line < static_cast<int>(fvfMetrics.size()) &&
                rel < static_cast<int>(fvfMetrics[line].size()))
            {
                fvfMetrics[line][rel].intakeNyquistRiskIRE =
                    gA * narrowMag;
            }

            if (attribution) {
                AttributionFacts &facts = attribution[rel].facts;
                facts.bandpassFineIRE = narrowMag;
                facts.bandpassCoarseIRE = wideMag;
                facts.lumaExcursionIRE = gA * narrowMag;
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
                const int wa = std::clamp(rel - kWideWin / 2, 0, width);
                const int wb = std::clamp(wa + kWideWin, 0, width);
                const double wn = std::max(1, wb - wa);
                const double ZwI = (preI[wb] - preI[wa]) / wn;
                const double ZwQ = (preQ[wb] - preQ[wa]) / wn;
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
                const int ka = std::clamp(rel - kNarrowWin / 2, 0, width);
                const int kb = std::clamp(ka + kNarrowWin, 0, width);
                for (int k = ka; k < kb; ++k) {
                    const int ca = std::clamp(k - 2, 0, width);
                    const int cb = std::clamp(k + 2, 0, width);
                    const double cn = std::max(1, cb - ca);
                    const double ZcI = (preI[cb] - preI[ca]) / cn;
                    const double ZcQ = (preQ[cb] - preQ[ca]) / cn;
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

        // Pass 3: publish the restrained source through the existing locked
        // carrier grammar.
        for (int rel = 0; rel < width; ++rel) {
            const int h = left + rel;
            const int phase = carrierSampleClass(line, h);
            const double source = restrainedLine[rel];

            const double i = source * lutI[phase];
            const double q = source * lutQ[phase];
            const double i4 = source * i4Scale[phase];
            const double q4 = source * q4Scale[phase];

            if (demodI) demodI[rel] = static_cast<float>(i);
            if (demodQ) demodQ[rel] = static_cast<float>(q);
            if (demodI4) demodI4[rel] = static_cast<float>(i4);
            if (demodQ4) demodQ4[rel] = static_cast<float>(q4);
            if (lockedI4) lockedI4[rel] = static_cast<float>(i4);
            if (lockedQ4) lockedQ4[rel] = static_cast<float>(q4);

            const double chromaMagnitude =
                std::fabs(source) * magnitudeScale[phase];

            magnitude[rel] = static_cast<float>(chromaMagnitude);
            lockedSource[rel] = source * remodScale[phase];

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

    std::vector<double> demI(width), demQ(width);
    std::vector<double> preI(width + 1), preQ(width + 1);
    std::vector<double> env(width), preEnv(width + 1);

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
            env[rel] = std::hypot(s, combLine[left + relN]);
        }

        preI[0] = 0.0;
        preQ[0] = 0.0;
        preEnv[0] = 0.0;
        for (int rel = 0; rel < width; ++rel) {
            preI[rel + 1] = preI[rel] + demI[rel];
            preQ[rel + 1] = preQ[rel] + demQ[rel];
            preEnv[rel + 1] = preEnv[rel] + env[rel];
        }

        for (int rel = 0; rel < width; ++rel) {
            const int wa = std::clamp(rel - kWideWin / 2, 0, width);
            const int wb = std::clamp(wa + kWideWin, 0, width);
            const double wn = std::max(1, wb - wa);
            const double ZwI = (preI[wb] - preI[wa]) / wn;
            const double ZwQ = (preQ[wb] - preQ[wa]) / wn;

            const int na = std::clamp(rel - kNarrowWin / 2, 0, width);
            const int nb = std::clamp(na + kNarrowWin, 0, width);
            const double nn = std::max(1, nb - na);
            const double narrowMag = ((preEnv[nb] - preEnv[na]) / nn) * invIreScale;

            const double wideMag = 2.0 * std::hypot(ZwI, ZwQ) * invIreScale;

            double gA = 0.0;
            if (narrowMag > kImpurityFloorIRE && wideMag < narrowMag) {
                gA = clamp01(
                    (narrowMag - wideMag) /
                    std::max(kImpurityFloorIRE, narrowMag));
            }

            // Union with the pre-comb read buildPhaseCorrected1D() already
            // published into this same buffer.  The post-comb pass measures what
            // survived combing, but the interfield comb smears sharp cross-color
            // transients into low-amplitude pseudo-coherence that aperture purity
            // can no longer distinguish from authentic chroma -- gA collapses and
            // the contamination leaks.  The seed saw that transient sharp, pre-comb
            // (narrow >> wide), so take the stronger of the two reads: a pixel
            // flagged before combing is not lost after it.  This reuses the seed's
            // gA from carrierImpurity (no recompute); the suppression still happens
            // only color-side, Y stays raw - full carrier, so no checkerboard.
            impurityRow[rel] =
                std::max(impurityRow[rel], static_cast<float>(gA));

            if (pcDiagLine >= 0 && line == pcDiagLine &&
                rel >= pcDiagC0 && rel <= pcDiagC1) {
                std::fprintf(stderr,
                    "CCPOST line=%d col=%d srcBuf=%d narrow=%.2f wide=%.2f gA=%.3f\n",
                    line, rel, srcBuf, narrowMag, wideMag, gA);
            }
        }
    }
}

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

    // Keep these sized because other recent versions of this function used them
    // directly; this prevents stale/undersized scratch state from reappearing if
    // small edits are made around this function.
    ensureScratch(scratch_lineWorkA);
    ensureScratch(scratch_lineWorkC);
    ensureScratch(scratch_lineWorkD);
    ensureScratch(scratch_yhp);
    ensureScratch(scratch_yI);
    ensureScratch(scratch_yQ);
    ensureScratch(scratch_hpI);
    ensureScratch(scratch_hpQ);
    ensureScratch(scratch_hpY);
    ensureScratch(scratch_outMixed);

    auto finiteOrZero = [](double v) -> double {
        return std::isfinite(v) ? v : 0.0;
    };

    const double ccWeight = std::max(0.0, T.CC_SUPPRESSION_WEIGHT);

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

        // Remod coefficients for burst-locked IQ back into composite sample space.
        //
        // c = ti * 0.5 * (bcos*sp - bsin*cp)
        //   + tq * 0.5 * (bsin*sp + bcos*cp)
        double remodI[4];
        double remodQ[4];

        for (int ph = 0; ph < 4; ++ph) {
            const double sp = spLUT_locked[ph];
            const double cp = cpLUT_locked[ph];

            remodI[ph] = finiteOrZero(0.5 * (bcos * sp - bsin * cp));
            remodQ[ph] = finiteOrZero(0.5 * (bsin * sp + bcos * cp));
        }

        float  *tiRow       = demodTI_line(line);
        float  *tqRow       = demodTQ_line(line);
        float  *ti4Row      = demodTI4fsc_line(line);
        float  *tq4Row      = demodTQ4fsc_line(line);
        float  *prodIRow    = lockedProductI_line(line);
        float  *prodQRow    = lockedProductQ_line(line);
        double *carrierComp = lockedCarrierComposite_line(line);

        const float *impRow = carrierImpurity_line(line);

        if (prodIRow)    std::fill(prodIRow, prodIRow + width, 0.0f);
        if (prodQRow)    std::fill(prodQRow, prodQRow + width, 0.0f);
        if (carrierComp) std::fill(carrierComp, carrierComp + width, 0.0);

        for (int xi = 0; xi < width; ++xi) {
            const int h  = left + xi;
            const int ph = carrierSampleClass(line, h);

            // Plain locked demod only.  No line affine, no local affine, no
            // sliding-window carrier fit.  This function may suppress transfer,
            // but it must not reshape the carrier that produceY subtracts.
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

            const double plainCarrier =
                finiteOrZero(ti * remodI[ph] + tq * remodQ[ph]);

            // Critical geometry rule:
            // produceY subtracts only the carrier in the original comb geometry.
            // Do not feed it any local affine, fitted carrier, or vetter-shaped
            // carrier.
            if (carrierComp)
                carrierComp[xi] = plainCarrier;

            const double gA =
                impRow ? std::clamp((double)impRow[xi], 0.0, 1.0) : 0.0;

            const double ccAlpha =
                std::clamp(1.0 - gA * ccWeight, 0.0, 1.0);

            const double xferTi = finiteOrZero(ti * ccAlpha);
            const double xferTq = finiteOrZero(tq * ccAlpha);

            const float prodI = (float)finiteOrZero(xferTi * GI_PRODUCT);
            const float prodQ = (float)finiteOrZero(xferTq * GQ_PRODUCT);

            if (prodIRow) prodIRow[xi] = prodI;
            if (prodQRow) prodQRow[xi] = prodQ;

            scratch_preI[xi] = prodI;
            scratch_preQ[xi] = prodQ;
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
    const auto &T       = configuration.tunables;

    constexpr bool   EXP_IQ_FIR_ENABLE = true;
    constexpr int    EXP_FIR_TAPS      = 21;
    constexpr double EXP_I_CUTOFF_MHZ  = 1.5;
    constexpr double EXP_Q_CUTOFF_MHZ  = 0.67;

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

        if (configuration.residualColor) {
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

            // Same aperture cross-color suppression as the coherent path
            // (splitIQlocked): scale the demodulated residual chroma by
            // (1 - gA*weight).  Y is already raw - full carrier, so this scales
            // only the rendered colour and never re-introduces carrier-band
            // energy into luma -- coherent and residual modes now agree.
            const float *impRow = carrierImpurity_line(line);
            const double ccWeight =
                std::max(0.0, configuration.tunables.CC_SUPPRESSION_WEIGHT);

            for (int i = 0; i < width; ++i) {
                const int h = left + i;
            
                // Residual-colour mode derives chroma from the same carrier residual
                // that produceY left behind: raw - Y.  Do not apply an additional local
                // DC follower here; that gives residual colour a different low-frequency
                // convention from the luma it is derived from.
                const double chroma = (double)rawLine[h] - Yrow[h];
            
                const int ph = carrierSampleClass(line, h);
                const double alphaEff =
                    std::max(0.0, 1.0 - (impRow ? (double)impRow[i] : 0.0) * ccWeight);
            
                scratch_preI[i] = (chroma * lutTi[ph]) * GI_PRODUCT * alphaEff;
                scratch_preQ[i] = (chroma * lutTq[ph]) * GQ_PRODUCT * alphaEff;
            }
        } else {
            // The normal locked path consumes the cache prepared by splitIQlocked().
            const float *prodIRow = lockedProductI_line(line);
            const float *prodQRow = lockedProductQ_line(line);
            for (int i = 0; i < width; ++i) {
                scratch_preI[i] = prodIRow ? (double)prodIRow[i] : 0.0;
                scratch_preQ[i] = prodQRow ? (double)prodQRow[i] : 0.0;
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
    const bool residualVideo = configuration.residualVideo;
    const bool use3DY = configuration.residualVideo3D
                       && prevFrameForVet != nullptr
                       && nextFrameForVet != nullptr;

    // produceY is a pure consumer: it subtracts the composite carrier that
    // splitIQlocked aligned and emitted. Where that carrier was drift-corrected
    // to the raw carrier, only true carrier is removed and composite HF luma
    // survives into Y. With --no-residual-video there is no aligned carrier, so
    // fall back to full-strength subtraction of the selected comb scalar.
    //
    // When --residual-video-3d is on (and prev/next frames are available), the
    // 3D Y election owns the per-pixel output: getBestY votes between the
    // current 2D residual (raw - clpLine) and its temporal neighbors. This is
    // a separate feature from residual-Y carrier correction -- election runs
    // on the baseline 2D residual so temporal candidates are comparable across
    // frames, independent of any current-frame carrier reshape.
    for (int line = firstLine; line < lastLine; ++line) {
        if (line >= demodLines) continue;

        const quint16 *rawLine = rawbuffer.data() + line * fullWidth;
        double *Y = componentFrame->y(line);
        const double *clpLine = clpbuffer[srcBuf].pixel[line];
        const double *carrierComp =
            residualVideo ? lockedCarrierComposite_line(line) : nullptr;

        if (use3DY) {
            for (int h = left; h < right; ++h) {
                const double c = clpLine[h];
                const double y2D = std::isfinite(c)
                    ? (double)rawLine[h] - c
                    : (double)rawLine[h];
                Y[h] = getBestY(line, h, y2D,
                                *prevFrameForVet, *nextFrameForVet);
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
void Comb::FrameBuffer::lurchSharpenCoarsePrior(const double *means,
                                                int meanCount,
                                                int width,
                                                double *prior,
                                                double *gateOut) const
{
    if (gateOut && width > 0)
        std::fill(gateOut, gateOut + width, 0.0);

    if (!means || !prior || meanCount < 6 || width <= 0)
        return;

    const auto smoothStep01 = [](double t) {
        t = std::clamp(t, 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    };

    // Per-window movement floor: a confirmed step of >= ~1.2 IRE moves each
    // straddling window by >= ~0.3 IRE.
    const double dThreshSamples = 0.30 * irescale;

    const int dCount = meanCount - 1;

    struct StepRun {
        int a = 0;
        int b = 0;
        double edge = 0.0;
        double stepAbsIRE = 0.0;
        double gate = 0.0;
        bool suppressed = false;
    };
    std::vector<StepRun> runs;

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

        StepRun run;
        run.a = a;
        run.b = b;
        run.edge = centroid + 2.5;
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

        const int xiFirst =
            std::clamp((int)std::floor(run.edge) - 4, 0, width - 1);
        const int xiLast =
            std::clamp((int)std::ceil(run.edge) + 3, 0, width - 1);

        for (int xi = xiFirst; xi <= xiLast; ++xi) {
            if (run.gate <= localGate[xi])
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

            localGate[xi] = run.gate;
            prior[xi] = base[xi] * (1.0 - run.gate) + means[side] * run.gate;
        }
    }

    if (gateOut) {
        for (int xi = 0; xi < width; ++xi)
            gateOut[xi] = localGate[xi];
    }
}

// Build the carrier-retracted view and its derived products.
//
// Per-line pass:
//   1. Four-view carrier/Y attribution on legal 4fSC luma floors
//      → carrierFit_flat; gated LS can refit contaminated luma edges.
//   2. Provisional raw - carrierFit is used only as a local residual scratch
//      while the line fit is being built.  The provisional floor stays in
//      scratch until the combed carrier exists.
//
// Cross-line pass (after all per-line fits):
//   3. Line-to-line cancellation on carrierFit → combedCarrier_flat
//      Real chroma inverts between opposite-phase lines, alien-Y doesn't.
//      combedCarrier preserves chroma and rejects alien-Y.
//   4. raw - combedCarrier → carrierRetracted_flat (flattened view)
//   5. Sliding 4-sample mean of carrierRetracted_flat → flatFloor_flat
//      (carrier-free luma floor; the 4-sample mean cancels carrier-shaped
//      residual at color transitions, leaving only genuine DC alien-Y)
//
// Called between phaseLocked() and split2D().  Only needs the burst phasor
// and lockedLumaBaseY4 (both from phaseLocked) — no comb output required.
void Comb::FrameBuffer::buildCarrierRetracted()
{
    carrierRetractedValid = false;

    if (!configuration.phaseCompensation)
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
    if (lsRefitGate_flat.size() < need)
        lsRefitGate_flat.assign(need, 0.0f);
    if (coarseYEvidence_flat.size() < need)
        coarseYEvidence_flat.assign(need, lddecode::FourViewPixelEvidence{});
    if (carrierImpurity_flat.size() < need)
        carrierImpurity_flat.assign(need, 0.0f);
    if (carrierParallax_flat.size() < need)
        carrierParallax_flat.assign(need, lddecode::FourViewCarrierAttribution{});

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

    // ---------------------------------------------------------------
    // Pass 1: per-line carrier withdrawal.
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
        float *retractedRow = carrierRetracted_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        float *floorRow     = flatFloor_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        float *gateRow      = lsRefitGate_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        auto *evidenceRow   = coarseYEvidence_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        auto *parallaxRow   = carrierParallax_flat.empty()
                              ? nullptr
                              : carrierParallax_flat.data()
                                + static_cast<size_t>(line) * demodWidth;
        float *impurityRow  = carrierImpurity_flat.data()
                              + static_cast<size_t>(line) * demodWidth;

        std::fill(gateRow, gateRow + width, 0.0f);
        std::fill(evidenceRow, evidenceRow + width, lddecode::FourViewPixelEvidence{});
        std::fill(impurityRow, impurityRow + width, 0.0f);

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
            evidenceRow[xi].rawSample = static_cast<float>(rawWhole[xi]);
        }

        if (!grammarLocked) {
            for (int xi = 0; xi < width; ++xi) {
                fitRow[xi]       = 0.0f;
                retractedRow[xi] = static_cast<float>(rawWhole[xi]);
                floorRow[xi]     = static_cast<float>(coarseY[xi]);
                if (parallaxRow)
                    parallaxRow[xi] = lddecode::FourViewCarrierAttribution{};
            }
            continue;
        }

        const double bcos = grammar->burstCos;
        const double bsin = grammar->burstSin;
        const double maxCarrierSamples =
            std::max(24.0, grammar->carrierScale * 5.0) * irescale;

        // basisI/Q at position h depend only on (h & 3) given the line's
        // burst phasor and locked basis (remod4fscToShiftedComposite indexes
        // spLUT/cpLUT by (h & 3); lockedTo4fsc takes no h).  Precompute the
        // four phase values and fill by lookup — replaces 2*width function
        // calls per line with 8 function calls + 2*width table reads.
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
            const int idx = (left + xi) & 3;
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
            }

            for (int s = 0; s < meanCount; ++s) {
                winFloor[s] =
                    0.25 * (rawWhole[s + 0] +
                            rawWhole[s + 1] +
                            rawWhole[s + 2] +
                            rawWhole[s + 3]);
            }

            // Luma prior: the centered moving coarse, not a medoid of the
            // four covering means.  The medoid was robust but is still a
            // boxcar statistic with half the smear baked in; the prior is
            // the rolling legal mean, and the lurch preconditioner restores
            // the step placement the boxcar blurs.
            for (int xi = 0; xi < width; ++xi) {
                const int sc = std::clamp(xi - 1, 0, meanCount - 1);
                refinedY[xi] = winFloor[sc];
            }

            // Lurch preconditioner: sharpen the prior before the carrier fit
            // consumes it, so step energy stays out of raw - refinedY and
            // never enters the carrier band.
            lurchSharpenCoarsePrior(winFloor.data(), meanCount, width,
                                    refinedY, nullptr);

            for (int s = 0; s < meanCount; ++s) {
                double sII = 0.0, sIQ = 0.0, sQQ = 0.0;
                double sIY = 0.0, sQY = 0.0;

                double refinedMean = 0.0;
                double minRefined = 1e300;
                double maxRefined = -1e300;

                for (int k = 0; k < 4; ++k) {
                    const int xi = s + k;
                    const double bI = basisI[xi];
                    const double bQ = basisQ[xi];

                    const double residual = rawWhole[xi] - refinedY[xi];

                    sII += bI * bI;
                    sIQ += bI * bQ;
                    sQQ += bQ * bQ;
                    sIY += bI * residual;
                    sQY += bQ * residual;

                    refinedMean += refinedY[xi];
                    minRefined = std::min(minRefined, refinedY[xi]);
                    maxRefined = std::max(maxRefined, refinedY[xi]);
                }

                refinedMean *= 0.25;

                double fitI = 0.0;
                double fitQ = 0.0;
                const double det = sII * sQQ - sIQ * sIQ;
                if (std::fabs(det) > 1e-9) {
                    const double inv = 1.0 / det;
                    fitI = ( sQQ * sIY - sIQ * sQY) * inv;
                    fitQ = (-sIQ * sIY + sII * sQY) * inv;
                }

                winI[s] = fitI;
                winQ[s] = fitQ;

                double errSq = 0.0;
                double basis01 = 0.0; // +-+-
                double basis02 = 0.0; // ++--
                double basis03 = 0.0; // +--+
                double fitAbs = 0.0;

                for (int k = 0; k < 4; ++k) {
                    const int xi = s + k;
                    const double bI = basisI[xi];
                    const double bQ = basisQ[xi];

                    const double fit = fitI * bI + fitQ * bQ;
                    const double residual = rawWhole[xi] - refinedY[xi];
                    const double e = residual - fit;

                    errSq += e * e;
                    fitAbs += std::fabs(fit);

                    if (k == 0) {
                        basis01 += e;
                        basis02 += e;
                        basis03 += e;
                    } else if (k == 1) {
                        basis01 -= e;
                        basis02 += e;
                        basis03 -= e;
                    } else if (k == 2) {
                        basis01 += e;
                        basis02 -= e;
                        basis03 -= e;
                    } else {
                        basis01 -= e;
                        basis02 -= e;
                        basis03 += e;
                    }
                }

                const double errIRE = std::sqrt(0.25 * errSq) * invIreScale;
                const double latticeIRE =
                    0.25 * std::max({std::fabs(basis01),
                                     std::fabs(basis02),
                                     std::fabs(basis03)}) * invIreScale;
                const double floorDriftIRE =
                    std::fabs(winFloor[s] - refinedMean) * invIreScale;
                const double ySpanIRE =
                    (maxRefined - minRefined) * invIreScale;
                const double ampIRE =
                    0.25 * fitAbs * invIreScale;

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

            for (int xi = 0; xi < width; ++xi) {
                const int sFirst = std::max(0, xi - 3);
                const int sLast  = std::min(xi, meanCount - 1);

                lddecode::FourViewCarrierView views[4];
                int viewCount = 0;

                for (int s = sFirst; s <= sLast; ++s) {
                    if (viewCount >= 4)
                        break;
                    const double carrierSample = rawWhole[xi] - winFloor[s];
                    const double fittedSample =
                        winI[s] * basisI[xi] + winQ[s] * basisQ[xi];
                    views[viewCount].apertureStart = s;
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
                        double mDeltaSample = 0.0;
                        double mDeltaIRE = 0.0;
                        double mSupport = 0.0;
                        double mLocalX = 0.0;
                        const int s0 = s;
                        if (s0 + 4 < width) {
                            const double enterLeaveSample =
                                rawWhole[s0 + 4] - rawWhole[s0];
                            mDeltaSample = 0.25 * enterLeaveSample;
                            mDeltaIRE = mDeltaSample * invIreScale;
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
                        views[viewCount].membershipDeltaSample = mDeltaSample;
                        views[viewCount].membershipDeltaIRE    = mDeltaIRE;
                        views[viewCount].membershipSupport     = mSupport;
                        views[viewCount].membershipLocalX      = mLocalX;
                    }
                    ++viewCount;
                }

                evidenceRow[xi].viewCount = viewCount;
                for (int v = 0; v < viewCount; ++v) {
                    auto &dst = evidenceRow[xi].views[v];
                    const auto &src = views[v];
                    dst.apertureStart = src.apertureStart;
                    dst.apertureCenter = static_cast<float>(src.apertureCenter);
                    dst.yFloor = static_cast<float>(src.yFloor);
                    dst.carrierSample = static_cast<float>(src.carrierSample);
                    dst.fittedSample = static_cast<float>(src.fittedSample);
                    dst.carrierI = static_cast<float>(src.carrierI);
                    dst.carrierQ = static_cast<float>(src.carrierQ);
                    dst.sampleFitErrorIRE = static_cast<float>(src.sampleFitErrorIRE);
                    dst.remodErrorIRE = static_cast<float>(src.remodErrorIRE);
                    dst.latticeRiskIRE = static_cast<float>(src.latticeRiskIRE);
                    dst.ySpanIRE = static_cast<float>(src.ySpanIRE);
                    dst.membershipDeltaSample = static_cast<float>(src.membershipDeltaSample);
                    dst.membershipDeltaIRE    = static_cast<float>(src.membershipDeltaIRE);
                    dst.membershipSupport     = static_cast<float>(src.membershipSupport);
                    dst.membershipLocalX      = static_cast<float>(src.membershipLocalX);
                    dst.score = static_cast<float>(src.score);
                }

                auto parallax = lddecode::buildFourViewCarrierAttribution(
                    views,
                    viewCount,
                    invIreScale);

                double modelI = parallax.valid ? parallax.commonI : 0.0;
                double modelQ = parallax.valid ? parallax.commonQ : 0.0;

                parallax.discResponseI = 0.0;
                parallax.discResponseQ = 0.0;
                parallax.discResponseMagIRE = 0.0;
                parallax.discResponseSupport = 0.0;
                parallax.discResponseBlend = 0.0;

                double residualCarrierLo = -1e300;
                double residualCarrierHi =  1e300;
                double residualTightenSample = 0.0;
                double residualTightenSupport = 0.0;
                double residualTightenGain = 1.0;
                double residualTightenSpreadIRE = 0.0;
                double residualTightenFitErrorIRE = 0.0;
                if (viewCount > 0) {
                    residualCarrierLo =  1e300;
                    residualCarrierHi = -1e300;
                    double sumResidual = 0.0;
                    double sumFitError = 0.0;
                    double minFloor = 1e300;
                    double maxFloor = -1e300;
                    double maxLurchIRE = 0.0;
                    double maxLurchSupport = 0.0;

                    // Residual complements are observations, not estimates.
                    // For a carrier-compatible local structure, the smallest
                    // absolute residual amplitude is the conservative survivor:
                    // any larger same-structure carrier claim must be justified
                    // by additional evidence rather than by the workprint alone.
                    double minCompatibleCarrierSample = 0.0;
                    double minCompatibleAbs = 1e300;
                    double minCompatibleSupport = 0.0;
                    int compatibleResidualN = 0;

                    const double referenceSample = (parallax.valid &&
                        std::fabs(parallax.commonSample) > 0.5 / std::max(1e-12, invIreScale))
                            ? parallax.commonSample
                            : (modelI * basisI[xi] + modelQ * basisQ[xi]);
                    const int referenceSign = (referenceSample > 0.0) ? +1 :
                                              ((referenceSample < 0.0) ? -1 : 0);

                    int residualN = 0;

                    for (int v = 0; v < viewCount; ++v) {
                        const auto &view = views[v];
                        residualCarrierLo = std::min(residualCarrierLo, view.carrierSample);
                        residualCarrierHi = std::max(residualCarrierHi, view.carrierSample);
                        sumResidual += view.carrierSample;
                        sumFitError += view.sampleFitErrorIRE;
                        minFloor = std::min(minFloor, view.yFloor);
                        maxFloor = std::max(maxFloor, view.yFloor);
                        // Lurch evidence comes from the membership-change
                        // observations.
                        const double membershipLocalizer = std::exp(
                            -0.5 * (view.membershipLocalX * view.membershipLocalX)
                                 / (1.35 * 1.35));
                        const double lurch =
                            std::fabs(view.membershipDeltaIRE) *
                            std::clamp(view.membershipSupport, 0.0, 1.0) *
                            membershipLocalizer;
                        if (lurch > maxLurchIRE) {
                            maxLurchIRE = lurch;
                            maxLurchSupport = std::clamp(view.membershipSupport, 0.0, 1.0);
                        }

                        const int sampleSign = (view.carrierSample > 0.0) ? +1 :
                                               ((view.carrierSample < 0.0) ? -1 : 0);
                        const bool signCompatible =
                            (referenceSign == 0 || sampleSign == 0 || sampleSign == referenceSign);
                        const double viewFitGate = 1.0 - smoothStep01(
                            (std::max(0.0, view.sampleFitErrorIRE) - 1.5) / 6.0);
                        const double viewLatticeGate = 1.0 - smoothStep01(
                            (std::max(0.0, view.latticeRiskIRE) - 1.5) / 6.0);
                        const double viewSupport = std::clamp(
                            0.35 + 0.65 * viewFitGate * viewLatticeGate,
                            0.0,
                            1.0);
                        if (signCompatible && viewSupport > 0.20) {
                            const double a = std::fabs(view.carrierSample);
                            if (a < minCompatibleAbs) {
                                minCompatibleAbs = a;
                                minCompatibleCarrierSample = view.carrierSample;
                            }
                            minCompatibleSupport = std::max(minCompatibleSupport, viewSupport);
                            ++compatibleResidualN;
                        }
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
                    double movingResidualSample = residualTightenSample;
                    double movingResidualFitErrorIRE = residualTightenFitErrorIRE;
                    double movingResidualCoherence = 0.0;
                    double movingResidualPull = 0.0;
                    if (meanCount > 0) {
                        const int centeredStart = std::clamp(xi - 1, 0, meanCount - 1);
                        movingResidualSample = rawWhole[xi] - winFloor[centeredStart];
                        residualCarrierLo = std::min(residualCarrierLo, movingResidualSample);
                        residualCarrierHi = std::max(residualCarrierHi, movingResidualSample);
                        sumResidual += movingResidualSample;
                        const double movingFitError = parallax.valid
                            ? std::fabs(movingResidualSample - parallax.commonSample) * invIreScale
                            : 0.0;
                        sumFitError += movingFitError;
                        movingResidualFitErrorIRE = movingFitError;

                        const int movingSign = (movingResidualSample > 0.0) ? +1 :
                                               ((movingResidualSample < 0.0) ? -1 : 0);
                        const bool movingSignCompatible =
                            (referenceSign == 0 || movingSign == 0 || movingSign == referenceSign);
                        const double movingSupport = 1.0 - smoothStep01(
                            (movingFitError - 1.5) / 6.0);
                        if (movingSignCompatible && movingSupport > 0.20) {
                            const double a = std::fabs(movingResidualSample);
                            if (a < minCompatibleAbs) {
                                minCompatibleAbs = a;
                                minCompatibleCarrierSample = movingResidualSample;
                            }
                            minCompatibleSupport = std::max(minCompatibleSupport, movingSupport);
                            ++compatibleResidualN;
                        }
                        ++residualN;
                    }

                    residualTightenSample =
                        sumResidual / static_cast<double>(std::max(1, residualN));
                    residualTightenFitErrorIRE =
                        sumFitError / static_cast<double>(std::max(1, residualN));
                    residualTightenSpreadIRE =
                        (residualCarrierHi - residualCarrierLo) * invIreScale;

                    const double variableYIRE = (maxFloor >= minFloor)
                        ? (maxFloor - minFloor) * invIreScale
                        : 0.0;
                    const double spreadGate = 1.0 - smoothStep01(
                        (residualTightenSpreadIRE - 2.0) / 8.0);
                    const double fitGate = 1.0 - smoothStep01(
                        (residualTightenFitErrorIRE - 1.5) / 6.0);
                    const double lurchGate = smoothStep01((maxLurchIRE - 0.75) / 5.0);
                    const double yVariableGate = smoothStep01((variableYIRE - 1.0) / 8.0);
                    residualTightenSupport = std::clamp(
                        0.35 + 0.65 * spreadGate * fitGate,
                        0.0,
                        1.0);
                    residualTightenGain = std::clamp(
                        0.20 + 0.55 * residualTightenSupport +
                        0.25 * std::max(lurchGate, yVariableGate),
                        0.0,
                        1.0);

                    movingResidualCoherence = 1.0 - std::clamp(
                        std::fabs(movingResidualSample - residualTightenSample) * invIreScale /
                        std::max(3.0, 0.35 * std::fabs(residualTightenSample) * invIreScale + 1.0),
                        0.0,
                        1.0);
                    movingResidualPull = movingResidualCoherence * residualTightenSupport;

                    lddecode::CarrierResidualConsensus consensus;
                    consensus.carrierSample = residualTightenSample;
                    consensus.lo = residualCarrierLo;
                    consensus.hi = residualCarrierHi;
                    consensus.trust = residualTightenSupport;
                    if (compatibleResidualN > 0 && minCompatibleAbs < 1e299) {
                        consensus.minCompatibleCarrierSample = minCompatibleCarrierSample;
                        consensus.minCompatibleAbsIRE = minCompatibleAbs * invIreScale;
                        consensus.minCompatibleSupport = std::clamp(
                            minCompatibleSupport *
                            (0.35 + 0.65 * std::max(lurchGate, yVariableGate)),
                            0.0,
                            1.0);
                    }
                    consensus.spreadIRE = residualTightenSpreadIRE;
                    consensus.lumaVarianceIRE = variableYIRE;
                    consensus.variableYIRE = std::max(variableYIRE, maxLurchIRE * maxLurchSupport);
                    consensus.fitErrorIRE = residualTightenFitErrorIRE;
                    consensus.valid = true;
                    parallax.residualConsensus = consensus;
                    parallax.movingResidualSample = movingResidualSample;
                    parallax.movingResidualFitErrorIRE = movingResidualFitErrorIRE;
                    parallax.movingResidualCoherence = movingResidualCoherence;
                    parallax.movingResidualPull = movingResidualPull;
                }

                parallax.residualTightenSample = residualTightenSample;
                parallax.residualTightenSupport = residualTightenSupport;
                parallax.residualTightenGain = residualTightenGain;
                parallax.residualTightenSpreadIRE = residualTightenSpreadIRE;
                parallax.residualTightenFitErrorIRE = residualTightenFitErrorIRE;
                if (parallaxRow)
                    parallaxRow[xi] = parallax;

                auto finalizeCarrierSample = [&](double candidateI) {
                    double sample = candidateI * basisI[xi] + modelQ * basisQ[xi];

                    if (parallax.valid) {
                        constexpr double SAMPLE_DISC_SOFT_IRE = 1.5;
                        constexpr double SAMPLE_DISC_HARD_IRE = 5.0;
                        constexpr double CONTEXT_SOFT_IRE = 3.0;
                        constexpr double CONTEXT_HARD_IRE = 12.0;

                        const double sampleDisagreementIRE =
                            std::fabs(sample - parallax.commonSample) * invIreScale;
                        const double disagreementGate = smoothStep01(
                            (sampleDisagreementIRE - SAMPLE_DISC_SOFT_IRE) /
                            std::max(1e-9, SAMPLE_DISC_HARD_IRE - SAMPLE_DISC_SOFT_IRE));
                        const double contextIRE =
                            std::max(parallax.yCurvatureIRE,
                                     0.5 * parallax.ySpreadIRE);
                        const double contextGate = smoothStep01(
                            (contextIRE - CONTEXT_SOFT_IRE) /
                            std::max(1e-9, CONTEXT_HARD_IRE - CONTEXT_SOFT_IRE));

                        // The per-pixel floor residual is not a replacement model;
                        // it is a witness that can pull a bad window IQ fit back
                        // toward what all legal Y floors say at this sample.
                        const double sampleTrust =
                            std::clamp(parallax.sampleCoherence, 0.0, 1.0);
                        const double sampleAnchor =
                            std::min(0.85,
                                     disagreementGate *
                                     sampleTrust *
                                     (0.25 + 0.75 * contextGate));

                        sample = sample * (1.0 - sampleAnchor) +
                                 parallax.commonSample * sampleAnchor;
                    }

                    if (residualTightenSupport > 0.0)
                        sample = std::clamp(sample, residualCarrierLo, residualCarrierHi);

                    return std::clamp(sample, -maxCarrierSamples, maxCarrierSamples);
                };

                const double baselineModelI = modelI;
                const double baselineCf = finalizeCarrierSample(baselineModelI);
                double cf = baselineCf;

                carrierFit[xi] = cf;
                flattened[xi] = rawWhole[xi] - cf;

                fitRow[xi] = static_cast<float>(cf);
                retractedRow[xi] = static_cast<float>(flattened[xi]);
            }
        } else {
            for (int xi = 0; xi < width; ++xi) {
                const double cf = 0.0;
                carrierFit[xi] = cf;
                flattened[xi] = rawWhole[xi];
                fitRow[xi] = 0.0f;
                retractedRow[xi] = static_cast<float>(flattened[xi]);
                if (parallaxRow)
                    parallaxRow[xi] = lddecode::FourViewCarrierAttribution{};
            }
        }

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
                            std::hypot(carrierFit[brightIdx],
                                       carrierFit[brightJ]) * invIreScale;
                        const double localAmpIRE =
                            std::hypot(carrierFit[xi],
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
                    if (parallaxRow && parallaxRow[xi].valid) {
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
                    gateRow[xi] = static_cast<float>(g);

                    if (g > 0.0) {
                        const double blended =
                            carrierFit[xi] * (1.0 - g) + lsFit * g;
                        carrierFit[xi] = blended;
                        flattened[xi] = rawWhole[xi] - blended;
                        fitRow[xi] = static_cast<float>(blended);
                        retractedRow[xi] = static_cast<float>(flattened[xi]);
                    }
                }
            }
        }

        // ---------------------------------------------------------------
        // Carrier-band Y attribution from the wide coherent fit.
        //
        // The narrow four-view carrier fit is the full-spectrum workprint.
        // The wide fit is not a carrier source, and it is not a confidence
        // flag.  It measures a specific transferable component:
        //
        //     carrierBandY = narrowWorkprintCarrier - wideCoherentCarrier
        //
        // where the narrow one-cycle explanation carries more carrier-band
        // energy than the longer coherent carrier explanation can sustain.
        // That component is attributed to Y before the carrier model is
        // promoted.  In model terms:
        //
        //     CarrierWorkprint = fit
        //     CarrierModel     = CarrierWorkprint - carrierBandY
        //     YModel           = raw - CarrierModel
        //
        // Lurch / membership-change evidence from the legal 4-sample averages
        // determines how much of the measured component may be transferred at
        // hard luma transitions.  This is deliberately upstream of retracted Y,
        // because retracted Y is only trustworthy after this attribution.
        // ---------------------------------------------------------------
        {
            if ((int)scratch_attrWideCarrier.size() < width) {
                scratch_attrWideCarrier.resize(width, 0.0);
                scratch_attrBandYClaim.resize(width, 0.0);
                scratch_attrMembershipY.resize(width, 0.0);
            }
            double *wideCarrierSample = scratch_attrWideCarrier.data();
            double *carrierBandYClaim = scratch_attrBandYClaim.data();
            double *membershipYIRE = scratch_attrMembershipY.data();

            std::fill(wideCarrierSample, wideCarrierSample + width, 0.0);
            std::fill(carrierBandYClaim, carrierBandYClaim + width, 0.0);
            std::fill(membershipYIRE, membershipYIRE + width, 0.0);

            constexpr int CC_WIN = 32;
            constexpr int CC_HALF = CC_WIN / 2;

            if (width >= CC_WIN) {
                // Gram matrix for uniform-weight LS over CC_WIN samples.
                // basisI/Q have period 4, so over 4N samples the sums are
                // just N times the single-period values.
                const int periods = CC_WIN / 4;
                double sII_1 = 0.0, sIQ_1 = 0.0, sQQ_1 = 0.0;
                for (int p = 0; p < 4; ++p) {
                    sII_1 += basisI4[p] * basisI4[p];
                    sIQ_1 += basisI4[p] * basisQ4[p];
                    sQQ_1 += basisQ4[p] * basisQ4[p];
                }
                const double gramII = sII_1 * periods;
                const double gramIQ = sIQ_1 * periods;
                const double gramQQ = sQQ_1 * periods;
                const double gramDet = gramII * gramQQ - gramIQ * gramIQ;

                if (std::fabs(gramDet) > 1e-9) {
                    const double gramInv = 1.0 / gramDet;

                    // Initial window [0, CC_WIN-1].
                    double sIY = 0.0, sQY = 0.0;
                    for (int k = 0; k < CC_WIN; ++k) {
                        const double obs = rawWhole[k] - coarseY[k];
                        sIY += basisI[k] * obs;
                        sQY += basisQ[k] * obs;
                    }

                    for (int xi = 0; xi < width; ++xi) {
                        int wantA = xi - CC_HALF;
                        int wantB = xi + CC_HALF - 1;
                        if (wantA < 0) { wantB += -wantA; wantA = 0; }
                        if (wantB >= width) {
                            const int ov = wantB - (width - 1);
                            wantB -= ov;
                            wantA -= ov;
                            if (wantA < 0) wantA = 0;
                        }

                        if (xi > 0) {
                            int prevA = (xi - 1) - CC_HALF;
                            int prevB = (xi - 1) + CC_HALF - 1;
                            if (prevA < 0) { prevB += -prevA; prevA = 0; }
                            if (prevB >= width) {
                                const int ov = prevB - (width - 1);
                                prevB -= ov;
                                prevA -= ov;
                                if (prevA < 0) prevA = 0;
                            }

                            for (int k = prevA; k < wantA; ++k) {
                                const double obs = rawWhole[k] - coarseY[k];
                                sIY -= basisI[k] * obs;
                                sQY -= basisQ[k] * obs;
                            }
                            for (int k = prevB + 1; k <= wantB; ++k) {
                                const double obs = rawWhole[k] - coarseY[k];
                                sIY += basisI[k] * obs;
                                sQY += basisQ[k] * obs;
                            }
                        }

                        const double wideI = ( gramQQ * sIY - gramIQ * sQY) * gramInv;
                        const double wideQ = (-gramIQ * sIY + gramII * sQY) * gramInv;
                        const double wideSample = wideI * basisI[xi] + wideQ * basisQ[xi];
                        wideCarrierSample[xi] = wideSample;

                        const double narrowSample = carrierFit[xi];
                        const int xi1 = std::min(xi + 1, width - 1);
                        const double fitMagIRE =
                            std::hypot(static_cast<double>(fitRow[xi]),
                                       static_cast<double>(fitRow[xi1])) * invIreScale;
                        const double wideMagIRE = std::hypot(wideI, wideQ) * invIreScale;

                        // Membership/lurch evidence: this is not edge detection.
                        // It measures how much the legal carrier-cancelling
                        // averages change when a pixel enters/leaves the 4-sample
                        // cancellation window.  That component is attributed to
                        // luma movement through the coarse aperture.
                        double lurchIRE = 0.0;
                        for (int v = 0; v < evidenceRow[xi].viewCount; ++v) {
                            const auto &ev = evidenceRow[xi].views[v];
                            const double localizer = std::exp(
                                -0.5 * (static_cast<double>(ev.membershipLocalX) *
                                        static_cast<double>(ev.membershipLocalX))
                                     / (1.35 * 1.35));
                            const double l =
                                std::fabs(static_cast<double>(ev.membershipDeltaIRE)) *
                                std::clamp(static_cast<double>(ev.membershipSupport), 0.0, 1.0) *
                                localizer;
                            lurchIRE = std::max(lurchIRE, l);
                        }
                        membershipYIRE[xi] = lurchIRE;

                        const double membershipFraction =
                            smoothGate01((lurchIRE - 0.50) / 4.0);

                        double phaseAgreement = 0.0;
                        double carrierCoherence = 0.0;
                        double carrierConflict = 1.0;
                        if (parallaxRow && parallaxRow[xi].valid) {
                            const auto &p = parallaxRow[xi];
                            const double narrowMag =
                                std::hypot(p.commonI, p.commonQ);
                            const double wideMag = std::hypot(wideI, wideQ);
                            if (narrowMag > 1e-9 && wideMag > 1e-9) {
                                phaseAgreement = std::clamp(
                                    (p.commonI * wideI + p.commonQ * wideQ) /
                                        (narrowMag * wideMag),
                                    0.0,
                                    1.0);
                            }

                            carrierCoherence =
                                std::clamp(p.carrierCoherence, 0.0, 1.0);
                            const double spreadConflict = std::clamp(
                                p.carrierSpreadIRE /
                                    std::max(3.0, 0.35 * p.commonMagIRE + 1.0),
                                0.0,
                                1.0);
                            const double latticeConflict = std::clamp(
                                p.latticeRiskIRE /
                                    std::max(3.0, 0.35 * p.commonMagIRE + 1.0),
                                0.0,
                                1.0);
                            const double fitConflict = smoothGate01(
                                (p.sampleFitErrorIRE - 1.0) / 5.0);
                            carrierConflict = std::max({
                                spreadConflict,
                                latticeConflict,
                                fitConflict,
                                1.0 - carrierCoherence
                            });
                        }

                        const CarrierImpurityEvidence impurityEvidence {
                            fitMagIRE,
                            wideMagIRE,
                            phaseAgreement,
                            carrierCoherence,
                            carrierConflict,
                            membershipFraction
                        };
                        const double impurity =
                            detectCarrierImpurity(impurityEvidence);
                        const double transferFraction =
                            carrierImpurityTransferStrength(
                                impurity, membershipFraction);

                        double yClaim = (narrowSample - wideSample) * transferFraction;

                        // Preserve sign and prevent the attribution transfer from
                        // crossing the workprint through zero in a single pass.
                        const double maxClaim = 0.90 * std::fabs(narrowSample);
                        yClaim = std::clamp(yClaim, -maxClaim, maxClaim);

                        carrierBandYClaim[xi] = yClaim;

                        // Publish detector evidence, not the transferred
                        // waveform amount. Consumers must choose their own
                        // named policy conversion.
                        impurityRow[xi] = static_cast<float>(impurity);
                    }
                }
            }

            // Apply the attribution transfer to the carrier workprint before
            // Pass 2 promotion.  This is not a trust/distrust action; it moves
            // a measured waveform component out of the carrier model and into
            // the subsequent retracted-Y model.
            for (int xi = 0; xi < width; ++xi) {
                const double yClaim = carrierBandYClaim[xi];
                if (yClaim == 0.0)
                    continue;

                double sample = carrierFit[xi] - yClaim;

                // The five residual complements still define the local surviving
                // carrier interval.  Keep the transfer inside that interval when
                // the attribution record provides one.
                if (parallaxRow && parallaxRow[xi].valid &&
                    parallaxRow[xi].residualConsensus.valid)
                {
                    const auto &rc = parallaxRow[xi].residualConsensus;
                    const double padIRE = 0.75 + 2.0 * (1.0 - std::clamp(rc.trust, 0.0, 1.0));
                    const double padSamples = padIRE / std::max(1e-12, invIreScale);
                    const double lo = std::min(rc.lo, rc.hi) - padSamples;
                    const double hi = std::max(rc.lo, rc.hi) + padSamples;
                    sample = std::clamp(sample, lo, hi);

                    // The min-compatible survivor stays recorded in the
                    // consensus as attribution evidence, but it is not applied
                    // as a clamp or attractor here.  Hard limits on the model
                    // must describe the possible range of values — which the
                    // lo/hi interval above does, spanning all five residual
                    // complements — never a bound particular to one source.
                    // The smallest single view is often the most contaminated
                    // one at compact chroma (a straddling aperture), so
                    // ceiling the model to it gutted real patch carrier.

                    auto p2 = parallaxRow[xi];
                    p2.residualConsensus.workprintSample = carrierFit[xi];
                    p2.residualConsensus.workprintCorrectionIRE =
                        std::fabs(sample - carrierFit[xi]) * invIreScale;
                    parallaxRow[xi] = p2;
                }

                sample = std::clamp(sample, -maxCarrierSamples, maxCarrierSamples);
                carrierFit[xi] = sample;
                flattened[xi] = rawWhole[xi] - sample;
                fitRow[xi] = static_cast<float>(sample);
                retractedRow[xi] = static_cast<float>(flattened[xi]);
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

    // ---------------------------------------------------------------
    // Pass 2: line-to-line cancellation on carrierFit → combedCarrier.
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

    for (int line = firstLine; line < lastLine; ++line) {
        const float *fitRow = carrierFit_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        const float *floorRow = flatFloor_flat.data()
                                + static_cast<size_t>(line) * demodWidth;
        float *combRow = combedCarrier_flat.data()
                         + static_cast<size_t>(line) * demodWidth;

        const CombCarrierGrammar *grammar = carrierGrammarLine(line);
        if (!grammar || !grammar->grammarLocked) {
            std::fill(combRow, combRow + width, 0.0f);
            continue;
        }

        const int lineAbove = line - 1;
        const int lineBelow = line + 1;

        const CombCarrierGrammar *gAbove =
            (lineAbove >= firstLine) ? carrierGrammarLine(lineAbove) : nullptr;
        const CombCarrierGrammar *gBelow =
            (lineBelow < lastLine) ? carrierGrammarLine(lineBelow) : nullptr;

        const bool haveAbove = gAbove && gAbove->grammarLocked &&
            (gAbove->lineFlip != grammar->lineFlip);
        const bool haveBelow = gBelow && gBelow->grammarLocked &&
            (gBelow->lineFlip != grammar->lineFlip);

        const float *fitAbove = haveAbove
            ? (carrierFit_flat.data() + static_cast<size_t>(lineAbove) * demodWidth)
            : nullptr;
        const float *fitBelow = haveBelow
            ? (carrierFit_flat.data() + static_cast<size_t>(lineBelow) * demodWidth)
            : nullptr;
        const float *floorAbove = haveAbove
            ? (flatFloor_flat.data() + static_cast<size_t>(lineAbove) * demodWidth)
            : nullptr;
        const float *floorBelow = haveBelow
            ? (flatFloor_flat.data() + static_cast<size_t>(lineBelow) * demodWidth)
            : nullptr;

        // Reach gate: determines per-pixel cancellation strength toward
        // each neighbor line.  Inlined here (was a lambda) to keep the
        // data flow visible in the fused loop below.
        //
        // Mismatch and amplitude are evaluated as 2-sample quadrature
        // envelopes, never instantaneous samples: |fit + neighbor| dips to
        // zero twice per carrier cycle, so a per-sample gate oscillates at
        // carrier rate.  On slanted chroma (~2 px/line drift = 180° carrier
        // rotation per line) the opposite-lineFlip neighbor presents
        // SAME-SIGN chroma; the envelope gate sees a constant 2A mismatch
        // and stands the cancellation down consistently, where the
        // per-sample gate cancelled real chroma in carrier-rate bursts —
        // a checkerboard manufactured inside combedCarrier itself.
        auto reachGate = [&](int xi, const float *neighborFit,
                             const float *neighborFloor) {
            if (!neighborFit || !neighborFloor)
                return 0.0;

            const double lumaDiffIRE =
                std::fabs(static_cast<double>(floorRow[xi]) -
                          static_cast<double>(neighborFloor[xi])) * invIreScale;

            const int xj = std::min(xi + 1, width - 1);
            const double c0 = static_cast<double>(fitRow[xi]);
            const double c1 = static_cast<double>(fitRow[xj]);
            const double n0 = static_cast<double>(neighborFit[xi]);
            const double n1 = static_cast<double>(neighborFit[xj]);

            const double carrierMismatchIRE =
                std::hypot(c0 + n0, c1 + n1) * invIreScale;
            const double carrierAmpIRE = 0.5 *
                (std::hypot(c0, c1) + std::hypot(n0, n1)) * invIreScale;

            double lumaGate = softReachGate(lumaDiffIRE, 3.0, 10.0);

            const double carrierSoftIRE = std::max(3.0, 0.25 * carrierAmpIRE);
            const double carrierHardIRE = std::max(10.0, 0.80 * carrierAmpIRE);
            double carrierGate =
                softReachGate(carrierMismatchIRE,
                              carrierSoftIRE,
                              carrierHardIRE);

            return lumaGate * carrierGate;
        };

        // -----------------------------------------------------------------
        // Fused reach-gate sweep.
        //
        // The original three-pass design was:
        //   Sweep 1: compute raw gates and decisionBlend into per-pixel arrays
        //   Sweep 2: 5-tap [1,2,3,2,1] smooth of raw gates → smooth arrays
        //   Sweep 3: blend raw/smooth by decisionBlend, produce combRow
        //
        // Fused into two passes: first compute the raw gates (needed as
        // lookahead input for the stencil), then a single output pass that
        // evaluates the 5-tap smooth, the decision blend, and the final
        // combRow per pixel — eliminating 3 scratch arrays and 2 loop
        // traversals.
        // -----------------------------------------------------------------
        double *wAboveRaw = scratch_preI.data();
        double *wBelowRaw = scratch_preQ.data();

        // Pass A: raw gates — must be fully materialized before the stencil
        // can read ±2 neighbors.
        for (int xi = 0; xi < width; ++xi) {
            wAboveRaw[xi] =
                haveAbove ? reachGate(xi, fitAbove, floorAbove) : 0.0;
            wBelowRaw[xi] =
                haveBelow ? reachGate(xi, fitBelow, floorBelow) : 0.0;
        }

        // Pass B: inline 5-tap smooth + decision blend + combRow output.
        // The smooth kernel is [1,2,3,2,1] (sum = 9 in the interior).
        // At edges, the kernel is clamped and the divisor adjusts.
        constexpr double kWeights[5] = {1.0, 2.0, 3.0, 2.0, 1.0};

        for (int xi = 0; xi < width; ++xi) {
            // Inline 5-tap smooth of the raw gate arrays.
            double sumW = 0.0, sumAbove = 0.0, sumBelow = 0.0;
            for (int dx = -2; dx <= 2; ++dx) {
                const int xx = std::clamp(xi + dx, 0, width - 1);
                const double w = kWeights[dx + 2];
                sumW += w;
                sumAbove += w * wAboveRaw[xx];
                sumBelow += w * wBelowRaw[xx];
            }
            const double smoothAbove = sumAbove / sumW;
            const double smoothBelow = sumBelow / sumW;

            // Inline localCarrierAmpIRE + decision blend.
            const int xm = std::max(0, xi - 1);
            const int xp = std::min(width - 1, xi + 1);
            double amp = std::fabs(static_cast<double>(fitRow[xi]));
            amp = std::max(amp, std::fabs(static_cast<double>(fitRow[xm])));
            amp = std::max(amp, std::fabs(static_cast<double>(fitRow[xp])));
            if (fitAbove) {
                amp = std::max(amp, std::fabs(static_cast<double>(fitAbove[xi])));
                amp = std::max(amp, std::fabs(static_cast<double>(fitAbove[xm])));
                amp = std::max(amp, std::fabs(static_cast<double>(fitAbove[xp])));
            }
            if (fitBelow) {
                amp = std::max(amp, std::fabs(static_cast<double>(fitBelow[xi])));
                amp = std::max(amp, std::fabs(static_cast<double>(fitBelow[xm])));
                amp = std::max(amp, std::fabs(static_cast<double>(fitBelow[xp])));
            }
            const double blend = smoothStep01((amp * invIreScale - 8.0) / 10.0);

            // Final gated cancellation.
            const double wAbove =
                wAboveRaw[xi] * (1.0 - blend) + smoothAbove * blend;
            const double wBelow =
                wBelowRaw[xi] * (1.0 - blend) + smoothBelow * blend;
            const double wSum = wAbove + wBelow;

            if (wSum > 1e-9) {
                double neighborFit = 0.0;
                if (wAbove > 0.0 && fitAbove)
                    neighborFit += wAbove * static_cast<double>(fitAbove[xi]);
                if (wBelow > 0.0 && fitBelow)
                    neighborFit += wBelow * static_cast<double>(fitBelow[xi]);
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

                combRow[xi] = static_cast<float>(
                    static_cast<double>(fitRow[xi]) * (1.0 - strength) +
                    cancelled * strength);
            } else {
                combRow[xi] = fitRow[xi];
            }
        }
    }

    // ---------------------------------------------------------------
    // Final publication: retracted Y and final floor derive from the
    // promoted carrier model, not from the workprint fit.
    // ---------------------------------------------------------------
    for (int line = firstLine; line < lastLine; ++line) {
        const quint16 *rawLine =
            rawbuffer.data() + static_cast<size_t>(line) * videoParameters.fieldWidth;
        const float *combRow = combedCarrier_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        float *retractedRow = carrierRetracted_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        float *floorRow = flatFloor_flat.data()
                          + static_cast<size_t>(line) * demodWidth;

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
            refinedY[xi] = baseY4Src[xi];
            flattened[xi] = rawWhole[xi] - static_cast<double>(combRow[xi]);
            retractedRow[xi] = static_cast<float>(flattened[xi]);
        }

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

    carrierRetractedValid = true;
}
