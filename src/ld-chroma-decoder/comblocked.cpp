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

            double dc = (double)rawLine[left] - Yrow[left];
            constexpr double DC_ALPHA = 1.0 / 64.0;
            for (int i = 0; i < width; ++i) {
                const int h = left + i;
                const double chromaRaw = (double)rawLine[h] - Yrow[h];
                dc += DC_ALPHA * (chromaRaw - dc);
                const double chroma = chromaRaw - dc;
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
