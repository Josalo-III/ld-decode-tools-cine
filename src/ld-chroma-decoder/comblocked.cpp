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
        std::vector<double> boxcar;
        std::vector<double> gateScratch;
        if (buildSharp && width >= 4) {
            boxcar.assign(width, 0.0);
            gateScratch.assign(width, 0.0);
        }

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

            if (!buildSharp)
                continue;

            double *sharp = lockedLumaSharp_line(line);
            if (width < 4) {
                std::copy(lockedLumaSmooth_line(line),
                          lockedLumaSmooth_line(line) + width, sharp);
                continue;
            }
            // Sliding 4-sample boxcar (means the lurch pass reads).
            double sum4 = (double)rawLine[left + 0] + (double)rawLine[left + 1]
                        + (double)rawLine[left + 2] + (double)rawLine[left + 3];
            const int lastStart = width - 4;
            for (int xi = 0; xi <= lastStart; ++xi) {
                boxcar[xi] = 0.25 * sum4;
                if (xi < lastStart)
                    sum4 += (double)rawLine[left + xi + 4]
                          - (double)rawLine[left + xi];
            }
            for (int xi = lastStart + 1; xi < width; ++xi)
                boxcar[xi] = boxcar[lastStart];

            // The lurch prior is registered to the current pixel, not to the
            // window's left edge. This must match buildCarrierRetractionStage:
            // pixel xi starts from the legal mean whose aperture is xi-1..xi+2.
            // Copying boxcar[xi] here shifts the whole witness floor one sample
            // and publishes its edge error as a carrier-rate raster when HF is
            // reconstructed on top of it.
            for (int xi = 0; xi < width; ++xi) {
                const int sc = std::clamp(xi - 1, 0, lastStart);
                sharp[xi] = boxcar[sc];
            }
            lurchSharpenCoarsePrior(boxcar.data(), width - 3, width,
                                    sharp, gateScratch.data(), sharpLevel);
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
        // carrier client. Keep this formula and its clamped edge convention in
        // one place so analysis and rendering cannot drift apart.
        for (int rel = 0; rel < width; ++rel) {
            const double c  = rawAtRel(rel);
            const double m2 = rawAtRel(rel - 2);
            const double p2 = rawAtRel(rel + 2);
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
    // frame (the comb's own temporal structure, never the
    // residual-video-3D enhancement).  Energy that MATCHES where the
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
        if (rawBandpass) {
            for (int rel = 0; rel < width; ++rel)
                bpLine[rel] = rawBandpass[rel];
        } else {
            for (int rel = 0; rel < width; ++rel) {
                const double c  = rawAtRel(rel);
                const double m2 = rawAtRel(rel - 2);
                const double p2 = rawAtRel(rel + 2);
                bpLine[rel] = 0.50 * c - 0.25 * (m2 + p2);
            }
        }

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
            const int a = std::clamp(center - kWideWin / 2, 0, width);
            const int b = std::clamp(a + kWideWin, 0, width);
            const double sumI = preI[b] - preI[a];
            const double sumQ = preQ[b] - preQ[a];
            const double n = static_cast<double>(std::max(1, b - a));
            return (2.0 * boundedMag(sumI, sumQ) / n) * invIreScale;
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
            const int wa = std::clamp(rel - kWideWin / 2, 0, width);
            const int wb = std::clamp(wa + kWideWin, 0, width);
            const double wn = std::max(1, wb - wa);
            const double ZwI = (preI[wb] - preI[wa]) / wn;
            const double ZwQ = (preQ[wb] - preQ[wa]) / wn;

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

                const int na = std::clamp(rel - kNarrowWin / 2, 0, width);
                const int nb = std::clamp(na + kNarrowWin, 0, width);
                const double nn = static_cast<double>(std::max(1, nb - na));
                const double ZnI = (preI[nb] - preI[na]) / nn;
                const double ZnQ = (preQ[nb] - preQ[na]) / nn;
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

            // The emitted source remains the full-resolution ordinary carrier
            // plus any explicitly bounded 1D repair. Whitestar and the fits are
            // evidence/policy inputs only; no diagnostic projection becomes
            // picture here.
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
                const int wa = std::clamp(rel - kWideWin / 2, 0, width);
                const int wb = std::clamp(wa + kWideWin, 0, width);
                const double wn = std::max(1, wb - wa);
                const double ZwI = (preI[wb] - preI[wa]) / wn;
                const double ZwQ = (preQ[wb] - preQ[wa]) / wn;
                const double wideSample =
                    2.0 * (ZwI * cosRef[p] + ZwQ * sinRef[p]);
                const double wideMag =
                    2.0 * boundedMag(ZwI, ZwQ) * invIreScale;
                const double narrowMag = narrowEnvIRE(rel);

                auto &record = carrierAnalysis[rel];
                if (!record.fit.valid) {
                    const int na = std::clamp(rel - kNarrowWin / 2, 0, width);
                    const int nb = std::clamp(na + kNarrowWin, 0, width);
                    const double nn = static_cast<double>(std::max(1, nb - na));
                    const double ZnI = (preI[nb] - preI[na]) / nn;
                    const double ZnQ = (preQ[nb] - preQ[na]) / nn;
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
        // carrier grammar. The burst-aware LUTs build common-4fsc IQ; the
        // scalar publish must then defer to the carrier-grammar remod cursor so
        // samplePhase0 / line-flip policy lives in one place instead of in the
        // old per-leg scale math.
        auto remodCursor = lddecode::carrierGrammarCompositeRemodCursor(
            grammar, left, 1.0, lddecode::CarrierSignFrame::Grid4fsc);
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
            lockedSource[rel] =
                lddecode::carrierGrammarRemod4fscToComposite(
                    remodCursor, i4, q4);

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
                // Balanced 7-tap horizontal aggregate, matching the region
                // evaluator in buildCombTapLine: even/odd offsets carry the
                // two carrier axes, and the 0.5 end weights equalize them
                // (3:3) so the vector stays phase-flat while the wider
                // support keeps the hue verdict stable at low saturation.
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
                    return std::complex<double>(si / 3.0, sq / 3.0);
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

            const double wideMag = 2.0 * boundedMag(ZwI, ZwQ) * invIreScale;

            double gA = 0.0;
            if (narrowMag > kImpurityFloorIRE && wideMag < narrowMag) {
                gA = clamp01(
                    (narrowMag - wideMag) /
                    std::max(kImpurityFloorIRE, narrowMag));
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

            const double apertureRead = gA * (1.0 - regionKeep);
            const double edgeRead = edgeRamp * (1.0 - grammarPass);
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
    const auto &T       = configuration.tunables;

    // Gated cross-color term probe (see CCTERM below).
    static const int ccProbeLine = []{
        const char *s = std::getenv("LD_CC_PROBE_LINE"); return s ? std::atoi(s) : -1;
    }();
    static const int ccProbeC0 = []{
        const char *s = std::getenv("LD_CC_PROBE_C0"); return s ? std::atoi(s) : 0;
    }();
    static const int ccProbeC1 = []{
        const char *s = std::getenv("LD_CC_PROBE_C1"); return s ? std::atoi(s) : (1<<30);
    }();

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

            // Cross-color returned Y participates in produceY's election.
            // Residual colour is raw - the elected Y, so it is complementary
            // to whichever complete luma candidate actually won. Applying the
            // mask again here would suppress colour even where the election
            // rejected returned Y, bypassing that decision and discarding
            // energy instead of transferring it.
            const float *maskRow = lockedCcMask_line(line);
            const float *maskRawRow = lockedCcMaskRaw_line(line);
            const lddecode::CarrierAnalysisRecord *analysisRow =
                carrierAnalysis_line(line);
            const lddecode::CarrierAnalysisRecord *analysisUpRow =
                (line - 2 >= firstLine) ? carrierAnalysis_line(line - 2) : nullptr;
            const lddecode::CarrierAnalysisRecord *analysisDnRow =
                (line + 2 < lastLine) ? carrierAnalysis_line(line + 2) : nullptr;
            const AttributionEvidence *attributionRow =
                attributionEvidence_line(line);
            const double *lumaRow = lockedLumaCacheValid
                ? lockedLumaSmooth_line(line) : nullptr;
            const quint16 *rawRow = rawbuffer.data()
                + static_cast<size_t>(line) * videoParameters.fieldWidth;
            const double giProduct = configuration.gi_product;
            const double gqProduct = configuration.gq_product;

            for (int i = 0; i < width; ++i) {
                const int h = left + i;
                const int ph = carrierSampleClass(line, h);

                // Residual-colour mode derives chroma from the same carrier residual
                // that produceY left behind: raw - Y.  Do not apply an additional local
                // DC follower here; that gives residual colour a different low-frequency
                // convention from the luma it is derived from.
                const double chroma = (double)rawLine[h] - Yrow[h];

                if (ccProbeLine == line && i >= ccProbeC0 && i <= ccProbeC1) {
                    const float conf = analysisRow ? analysisRow[i].carrierConformance : 0.0f;
                    const float axisSupport = analysisRow
                        ? analysisRow[i].conformanceSupportFraction : 0.0f;
                    double grammarPass = analysisRow
                        ? lddecode::carrierLegalProof(
                              (double)analysisRow[i].carrierConformance,
                              (double)analysisRow[i].conformanceSupportFraction)
                        : 0.0;
                    grammarPass = std::max(grammarPass, std::max(
                        analysisUpRow
                            ? lddecode::carrierLegalProof(
                                  (double)analysisUpRow[i].carrierConformance,
                                  (double)analysisUpRow[i].conformanceSupportFraction)
                            : 0.0,
                        analysisDnRow
                            ? lddecode::carrierLegalProof(
                                  (double)analysisDnRow[i].carrierConformance,
                                  (double)analysisDnRow[i].conformanceSupportFraction)
                            : 0.0));
                    const double impulse = attributionRow
                        ? std::clamp(
                              attributionRow[i].facts.lumaImpulseRisk, 0.0, 1.0)
                        : 0.0;
                    std::fprintf(stderr,
                        "CCTERM line=%d col=%d maskRaw=%.3f maskSmooth=%.3f "
                        "conf=%.3f axisSupport=%.3f grammarPass=%.3f "
                        "impulse=%.3f lumaIRE=%.3f rawIRE=%.3f\n",
                        line, i, maskRawRow ? maskRawRow[i] : 0.0f,
                        maskRow ? maskRow[i] : 0.0f, conf, axisSupport,
                        grammarPass, impulse,
                        lumaRow ? lumaRow[i] * invIreScale : 0.0,
                        rawRow ? rawRow[left + i] * invIreScale : 0.0);
                }

                scratch_preI[i] = (chroma * lutTi[ph]) * giProduct;
                scratch_preQ[i] = (chroma * lutTq[ph]) * gqProduct;
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

void Comb::FrameBuffer::buildResidualCarrierEstimateRow(
    int line,
    const quint16 *rawLine,
    const double *lumaBasis,
    const float *tiLockedRow,
    const float *tqLockedRow,
    double *carrierOut)
{
    if (!rawLine || !lumaBasis || !tiLockedRow || !tqLockedRow || !carrierOut)
        return;

    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;
    if (width <= 0)
        return;

    auto ensureScratch = [&](std::vector<double> &v) {
        if ((int)v.size() < width)
            v.resize(width, 0.0);
    };

    ensureScratch(scratch_lineWorkA);
    ensureScratch(scratch_lineWorkB);
    ensureScratch(scratch_lineWorkC);
    ensureScratch(scratch_lineWorkD);
    ensureScratch(scratch_yhp);
    ensureScratch(scratch_yI);
    ensureScratch(scratch_yQ);
    ensureScratch(scratch_hpI);
    ensureScratch(scratch_hpQ);
    ensureScratch(scratch_hpY);
    ensureScratch(scratch_outMixed);

    const CombCarrierGrammar *grammar = carrierGrammarLine(line);
    const bool grammarLocked = grammar && grammar->grammarLocked;
    const double bcos = grammarLocked ? grammar->burstCos : 1.0;
    const double bsin = grammarLocked ? grammar->burstSin : 0.0;
    const double invI = invIreScale;

    constexpr int WIN = 16;
    constexpr int HALF = WIN / 2;
    constexpr double MIN_FIT_IRE = 2.0;
    constexpr double MAX_FIT_IRE = 35.0;
    constexpr double SAT_TROUBLE_IRE = 18.0;
    constexpr double VET_ALIGN_PHASE_MAX_DEG = 12.0;
    constexpr double VET_ALIGN_MIN_FIT_CORRELATION = 0.75;
    constexpr double VET_ALIGN_MAX_SHEAR = 0.15;
    constexpr double Y_LOCAL_MAX_PHASE_DEG = 14.0;
    constexpr double Y_LOCAL_GAIN_MIN = 0.90;
    constexpr double Y_LOCAL_GAIN_MAX = 1.10;
    constexpr double Y_LOCAL_MAX_SHEAR = 0.12;
    double *ti4fsc      = scratch_lineWorkA.data();
    double *tEnergy     = scratch_lineWorkB.data();
    double *tq4fsc      = scratch_lineWorkC.data();
    double *rEnergy     = scratch_lineWorkD.data();
    double *cSTT00      = scratch_yhp.data();
    double *cSTT01      = scratch_yI.data();
    double *cSTT11      = scratch_yQ.data();
    double *cSRT00      = scratch_hpI.data();
    double *cSRT01      = scratch_hpQ.data();
    double *cSRT10      = scratch_hpY.data();
    double *cSRT11      = scratch_outMixed.data();

    for (int x = 0; x < width; ++x) {
        const int h = left + x;
        const double ti0 = (double)tiLockedRow[x];
        const double tq0 = (double)tqLockedRow[x];

        double ti = 0.0, tq = 0.0;
        lockedTo4fsc(ti0, tq0, bcos, bsin, ti, tq);
        ti4fsc[x] = ti;
        tq4fsc[x] = tq;

        // Compare target and raw residual in the SAME shifted, grammar-indexed
        // 4fSC basis. demod4fscFromComposite() uses the unshifted raw h&3
        // basis; mixing it with the locked shifted basis manufactured a fixed
        // CAL_EPS phase correction (about -6.3 degrees) in every accepted fit.
        const int phase = carrierSampleClass(line, h);
        const double residual = (double)rawLine[h] - lumaBasis[x];
        const double ri = residual * spLUT_locked[phase] * 2.0;
        const double rq = residual * cpLUT_locked[phase] * 2.0;

        const double magT_ire = boundedMag(ti, tq) * invI;
        double w = 1.0;
        if (magT_ire > MAX_FIT_IRE) {
            const double t = (magT_ire - MAX_FIT_IRE) / (MAX_FIT_IRE + 1e-9);
            w = 1.0 / (1.0 + 4.0 * t * t);
        }

        cSTT00[x] = w * ti * ti;
        cSTT01[x] = w * ti * tq;
        cSTT11[x] = w * tq * tq;
        cSRT00[x] = w * ri * ti;
        cSRT01[x] = w * ri * tq;
        cSRT10[x] = w * rq * ti;
        cSRT11[x] = w * rq * tq;
        tEnergy[x] = w * (ti * ti + tq * tq);
        rEnergy[x] = w * (ri * ri + rq * rq);
    }

    const int winN = (width <= WIN) ? width : WIN;
    double sSTT00 = 0.0, sSTT01 = 0.0, sSTT11 = 0.0;
    double sSRT00 = 0.0, sSRT01 = 0.0, sSRT10 = 0.0, sSRT11 = 0.0;
    double sTEnergy = 0.0, sREnergy = 0.0;

    for (int i = 0; i < winN; ++i) {
        sSTT00 += cSTT00[i];
        sSTT01 += cSTT01[i];
        sSTT11 += cSTT11[i];
        sSRT00 += cSRT00[i];
        sSRT01 += cSRT01[i];
        sSRT10 += cSRT10[i];
        sSRT11 += cSRT11[i];
        sTEnergy += tEnergy[i];
        sREnergy += rEnergy[i];
    }

    for (int x = 0; x < width; ++x) {
        const int h = left + x;

        if (width > WIN && x > HALF && x <= width - HALF) {
            const int rem = x - HALF - 1;
            const int add = x + HALF - 1;
            sSTT00 += cSTT00[add] - cSTT00[rem];
            sSTT01 += cSTT01[add] - cSTT01[rem];
            sSTT11 += cSTT11[add] - cSTT11[rem];
            sSRT00 += cSRT00[add] - cSRT00[rem];
            sSRT01 += cSRT01[add] - cSRT01[rem];
            sSRT10 += cSRT10[add] - cSRT10[rem];
            sSRT11 += cSRT11[add] - cSRT11[rem];
            sTEnergy += tEnergy[add] - tEnergy[rem];
            sREnergy += rEnergy[add] - rEnergy[rem];
        }

        const double plainCarrier = remodLockedToShiftedComposite(
            (double)tiLockedRow[x], (double)tqLockedRow[x], h,
            bcos, bsin, spLUT_locked, cpLUT_locked);

        double STT[2][2] = {{sSTT00, sSTT01}, {sSTT01, sSTT11}};
        double SRT[2][2] = {{sSRT00, sSRT01}, {sSRT10, sSRT11}};
        // All samples in the aperture participate.  An instantaneous
        // amplitude floor rejects the zero crossings of a perfectly valid
        // carrier; eligibility is instead based on the aperture RMS below.
        const int n = winN;

        double Rm[2][2] = {{1, 0}, {0, 1}};
        double U[2][2]  = {{1, 0}, {0, 1}};
        double STTinv[2][2];
        const bool invOk = mat2_inv(STT, STTinv);
        const double tWindowIRE =
            std::sqrt(std::max(0.0, sTEnergy) / std::max(1, n)) * invI;
        const double rWindowIRE =
            std::sqrt(std::max(0.0, sREnergy) / std::max(1, n)) * invI;
        bool vetAccept = invOk && n >= 8 &&
                         tWindowIRE >= MIN_FIT_IRE &&
                         rWindowIRE >= MIN_FIT_IRE;
        if (vetAccept) {
            double Avet[2][2];
            double RmVet[2][2];
            double UVet[2][2];
            mat2_mul(SRT, STTinv, Avet);
            polar_decompose_2x2(Avet, RmVet, UVet);

            const double phase = std::atan2(RmVet[1][0], RmVet[0][0]);
            double l1 = 1.0, l2 = 1.0, V_[2][2];
            eig2_sym(UVet, l1, l2, V_);
            const double g = 0.5 * (std::max(0.0, l1) + std::max(0.0, l2));
            const double shear =
                (g > 1e-12) ? std::fabs(std::max(0.0, l1) - std::max(0.0, l2)) / g
                            : 0.0;
            // Actual normalized least-squares fit.  The former "rho" was
            // ||SRT|| / trace(STT): primarily the fitted amplitude ratio, not
            // a correlation or residual measure.  It therefore rejected the
            // very gain mismatch this affine stage exists to correct.  For
            // A = SRT * inv(STT), trace(A*SRT^T) is the response energy
            // explained by the two-component carrier model.  Normalize it by
            // the measured response energy so this gate answers only whether
            // T explains R, independent of their relative scale.
            const double modelEnergy =
                Avet[0][0] * SRT[0][0] + Avet[0][1] * SRT[0][1] +
                Avet[1][0] * SRT[1][0] + Avet[1][1] * SRT[1][1];
            const double explainedFraction = std::clamp(
                modelEnergy / std::max(1e-9, sREnergy), 0.0, 1.0);
            const double fitCorrelation = std::sqrt(explainedFraction);
            const double pMaxVet = VET_ALIGN_PHASE_MAX_DEG * M_PI / 180.0;
            vetAccept = std::fabs(phase) <= pMaxVet &&
                        fitCorrelation >= VET_ALIGN_MIN_FIT_CORRELATION &&
                        shear <= VET_ALIGN_MAX_SHEAR;

            if (vetAccept && n >= 16) {
                Rm[0][0] = RmVet[0][0]; Rm[0][1] = RmVet[0][1];
                Rm[1][0] = RmVet[1][0]; Rm[1][1] = RmVet[1][1];
                U[0][0]  = UVet[0][0];  U[0][1]  = UVet[0][1];
                U[1][0]  = UVet[1][0];  U[1][1]  = UVet[1][1];
            }
        }

        const double ti0 = ti4fsc[x];
        const double tq0 = tq4fsc[x];
        const bool satTrouble = (boundedMag(ti0, tq0) * invI > SAT_TROUBLE_IRE);
        clamp_rotation_gain_shear(
            Rm, U,
            Y_LOCAL_MAX_PHASE_DEG * M_PI / 180.0,
            !satTrouble,
            Y_LOCAL_GAIN_MIN, Y_LOCAL_GAIN_MAX,
            satTrouble ? 0.0 : Y_LOCAL_MAX_SHEAR);

        const double tiAdj = Rm[0][0] * ti0 + Rm[0][1] * tq0;
        const double tqAdj = Rm[1][0] * ti0 + Rm[1][1] * tq0;
        const double cHat = remod4fscToShiftedComposite(
            tiAdj, tqAdj, h, spLUT_locked, cpLUT_locked);

        // Residual Y is the coherent comb subtraction with a cleaner local
        // affine/polar round trip.  It has the SAME selected-comb phase lineage
        // as coherent Y; the fit may improve that subtraction, but an ownership
        // policy must not turn it into a different raw-passthrough source.
        // Where the fit is unavailable Rm/U remain identity, so this naturally
        // falls back to the coherent carrier.
        const double fittedCarrier = std::isfinite(cHat) ? cHat : plainCarrier;
        carrierOut[x] = fittedCarrier;
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
    const bool residualColor = configuration.residualColor;
    const bool use3DY = configuration.residualVideo3D
                       && prevFrameForVet != nullptr
                       && nextFrameForVet != nullptr;

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

    // produceY is a pure consumer: it subtracts the composite carrier that
    // splitIQlocked aligned and emitted. Where that carrier was drift-corrected
    // to the raw carrier, only true carrier is removed and composite HF luma
    // survives into Y. With --no-residual-video there is no aligned carrier, so
    // fall back to full-strength subtraction of the selected comb scalar.
    //
    // When --residual-video-3d is on (and prev/next frames are available), the
    // 3D Y election owns the per-pixel output: getBestY votes between the best
    // pre-output luma each frame has published so far. With --luma-witness
    // that means the carrier-retracted view; otherwise it falls back to the
    // coherent residual comb Y (raw - carrierComp), then the plain 2D comb
    // baseline. Residual-Y 3D stays a distinct temporal feature; it just no
    // longer ignores the better local luma model when carrier retraction is
    // available.

    // Residual-Y carrier prepass. Historically this affine/polar refinement
    // made the phase-solved coherent-Y subtraction exact enough to admit the
    // remaining HF Y wholesale. It consumes the same selected-comb locked IQ
    // as coherent Y; polar refinement is its advantage, not a second phase
    // lineage. An identity fit therefore collapses onto coherent Y.
    // buildResidualCarrierEstimateRow is a pure
    // per-line function of raw, the selected luma basis, and the locked demod
    // planes; produceY
    // reads each line's estimate for the centre pixel and for up to four
    // vertical neighbours (±1/±2). Reconstructed locally that was five heavy
    // rebuilds of every line's value -- and outside FVF two of the four
    // neighbour rebuilds are the wrong vertical step and discarded. Build every
    // line once here into a frame buffer; the centre and neighbours below are
    // then zero-cost row lookups keyed by line.
    const bool residualBaseReady =
        lockedLumaCacheValid && demodWidth == width &&
        !lockedLumaBaseY4_flat.empty() && !demodTI_flat.empty() &&
        !demodTQ_flat.empty();
    const bool residualUsesWitnessBasis =
        configuration.lumaWitness && coarseSharpLevel() > 0.0 &&
        !lockedLumaSharp_flat.empty();
    if (!lockedResidualCarrierValid.empty())
        std::fill(lockedResidualCarrierValid.begin(),
                  lockedResidualCarrierValid.end(), std::uint8_t{0});
    if (residualBaseReady && !lockedResidualCarrier_flat.empty()) {
        for (int l = firstLine; l < lastLine; ++l) {
            if (l >= demodLines) continue;
            double *out = lockedResidualCarrier_rowForBuild(l);
            if (!out) continue;
            const double *lumaBasis = residualUsesWitnessBasis
                ? lockedLumaSharp_line(l)
                : lockedLumaBaseY4_line(l);
            buildResidualCarrierEstimateRow(
                l, rawbuffer.data() + l * fullWidth,
                lumaBasis, demodTI_line(l), demodTQ_line(l),
                out);
            lockedResidualCarrierValid[l] = 1;
        }

        // Operand schedule-compatibility license.  The subtraction operand
        // cHat is comb IQ affined/polar-refined — its lineage is already
        // schedule-legal — so the license asks only the residual question:
        // did the refinement absorb off-schedule energy?  Test the OPERAND
        // itself against its ±1 partner's operand under the grammar relation,
        // one 4-sample cycle: on-schedule chroma inverts (relation-folded
        // corr -> -1), an absorbed alien waveform (fine static grid) MATCHES
        // where inversion is demanded (corr -> +1).  No raw-bandpass mixture
        // term enters, so hue transitions and compact/microscopic color pass
        // on their own alternation.  A center operand below the noise floor
        // licenses at 1 (subtracting ~nothing is harmless and gating it
        // would only manufacture pepper); a loud center with no observable
        // partner fails closed at 0.
        {
            const double rmsFloorLic = 3.0 * irescale;
            const double energyFloorLic = 4.0 * rmsFloorLic * rmsFloorLic;
            for (int l = firstLine; l < lastLine; ++l) {
                if (l >= demodLines) continue;
                float *lic = lockedResidualCarrierLicense_rowForBuild(l);
                const double *c0 = lockedResidualCarrier_line(l);
                if (!lic) continue;
                if (!c0) { continue; }
                const CombCarrierGrammar *g0lic = carrierGrammarLine(l);
                // Partner rows with relation fold: legal behaviour maps to
                // corr -1 for Opposite (invert) and +corr fold for Same.
                const double *pRow[2]; double pSign[2]; int nP = 0;
                auto addPartner = [&](int lp) {
                    if (nP >= 2 || lp < firstLine || lp >= lastLine ||
                        lp >= demodLines || !g0lic)
                        return;
                    const double *cp = lockedResidualCarrier_line(lp);
                    const CombCarrierGrammar *gp = carrierGrammarLine(lp);
                    if (!cp || !gp || !gp->grammarLocked)
                        return;
                    const auto rel =
                        lddecode::carrierGrammarSignedPhaseRelation(
                            g0lic, left, gp, left);
                    if (rel == lddecode::CarrierPhaseRelation::Opposite) {
                        pRow[nP] = cp; pSign[nP] = 1.0; ++nP;
                    } else if (rel == lddecode::CarrierPhaseRelation::Same) {
                        pRow[nP] = cp; pSign[nP] = -1.0; ++nP;
                    }
                };
                addPartner(l - 1);
                addPartner(l + 1);
                for (int xi = 0; xi < width; ++xi) {
                    const int w0 = std::clamp(xi, 0, width - 4);
                    double e0 = 0.0;
                    for (int k = 0; k < 4; ++k)
                        e0 += c0[w0 + k] * c0[w0 + k];
                    if (e0 < energyFloorLic) {
                        lic[xi] = 1.0f;   // harmless subtraction
                        continue;
                    }
                    double best = 9.0;    // most-legal signed corr observed
                    for (int a = 0; a < nP; ++a) {
                        double dot = 0.0, eP = 0.0;
                        for (int k = 0; k < 4; ++k) {
                            dot += c0[w0 + k] * pRow[a][w0 + k];
                            eP  += pRow[a][w0 + k] * pRow[a][w0 + k];
                        }
                        if (eP < energyFloorLic)
                            continue;
                        const double sc =
                            pSign[a] * dot / std::sqrt(e0 * eP);
                        if (best > 1.0 || sc < best)
                            best = sc;
                    }
                    lic[xi] = (best <= 1.0)
                        ? static_cast<float>(
                              lddecode::scheduleAlternationLicense(best))
                        : 0.0f;           // loud, unobservable: fail closed
                }
            }
        }
    }

    for (int line = firstLine; line < lastLine; ++line) {
        if (line >= demodLines) continue;

        const quint16 *rawLine = rawbuffer.data() + line * fullWidth;
        double *Y = componentFrame->y(line);
        const double *clpLine = clpbuffer[srcBuf].pixel[line];
        const double *carrierComp =
            residualVideo ? lockedCarrierComposite_line(line) : nullptr;

        // Carrier-retracted luma model. produceY owns luma: where the
        // retracted view is valid and residual colour is active, it becomes a
        // candidate in the HF luma election below -- NOT the owner. It is raw
        // minus the promoted interline carrier, so the cross-color the
        // coherent carrier would have removed from Y survives here as smooth
        // luma, where it competes with combY and 1D. The matching chroma
        // reduction is derived downstream in filterIQLocked (chroma = raw - Y),
        // keeping raw = Y + C by construction. Gated on residualColor so
        // --no-residual-color yields pure comb colour on the baseline Y. 3D
        // election stays disjoint.
        const float *retractedRow =
            (residualColor && carrierRetractedValid)
                ? carrierRetracted_line(line) : nullptr;
        const float *ccMaskRow = lockedCcMask_line(line);
        const float *ccMaskRawRow = lockedCcMaskRaw_line(line);
        const double *residualCarrierRow =
            lockedResidualCarrier_line(line);
        if (use3DY) {
            for (int h = left; h < right; ++h) {
                Y[h] = getBestY(line, h, *prevFrameForVet, *nextFrameForVet);
            }
        } else if (residualCarrierRow || retractedRow || ccMaskRow) {
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
            // Contestants (each a complete raw - carrier, converted below to
            // HF relative to the active decomposition basis):
            //   0 coherentY  = raw - carrierComp     (phase-solved selected comb)
            //   1 retractedY = carrierRetracted      (raw - combedCarrier)
            //   2 residualY  = raw - cHat            (affine/polar residual path)
            //   3 1D         = raw - locked1DSource  ADMITTED ONLY IF comb DQ'd
            //   4 returnedY   = combY + ccMask*(raw - combY)
            // Comb is the improvement on 1D; 1D has no voice while comb stands.
            // Coarse-platform selector. Default owns one cheap raster-aligned
            // four-sample coarse. --luma-witness unlocks the heavier centered,
            // lurch-sharpened platform. The selected platform is the sole LF
            // authority and also defines the residual fit and top-band
            // coordinate. No second coarse is mixed into reconstruction.
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
            const lddecode::CarrierAnalysisRecord *analysisRow =
                carrierAnalysis_line(line);
            const float *alienRow = regionAlienPartner_line(line);
            const double *affineResidualCarrierRow = residualCarrierRow;

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

            // Election tolerances (IRE -> samples).
            const double agreeTol   = 2.0 * irescale; // agreement early-out band
            const double inlierTol  = 4.0 * irescale; // medoid inlier gate
            const double phasePenSamp =
                std::max(0.0, configuration.tunables.PRODUCE_Y_PHASE_PENALTY_IRE)
                * irescale; // capped phase hygiene penalty

            // Candidate-owned residual-carrier subtraction under the OPERAND
            // schedule-compatibility license (see the prepass above).  cHat is
            // comb IQ affined/polar-refined -- its lineage is already schedule-
            // legal -- so the license tests the operand itself against its ±1
            // partner rather than collecting raw-bandpass votes: raw-vs-raw
            // correlation carries a shared-luma mixture term at edges and an
            // axis-corroboration tax that zeroed the subtraction along hue
            // transitions and compact color, publishing raw carrier as
            // residualY there.  Unlicensed (absorbed-alien / unobservable)
            // energy remains available as Y.
            auto residualCarrierForCandidate = [=](
                    const float *licRow,
                    int xi, double fittedCarrier) {
                if (!licRow || !std::isfinite(fittedCarrier))
                    return 0.0;
                return fittedCarrier * (double)licRow[xi];
            };
            const float *residualLicenseRow =
                lockedResidualCarrierLicense_line(line);

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
                } else if (plane == 2 && affineResidualCarrierRow) {
                    const double c = affineResidualCarrierRow[xx];
                    if (std::isfinite(c))
                        return (double)rawLine[hh] -
                            residualCarrierForCandidate(residualLicenseRow, xx, c);
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
            auto candidateMeanAt = [&](int plane, int h0,
                                       int firstK, int lastK) -> double {
                double sum = 0.0;
                const int count = lastK - firstK + 1;
                const int available = right - left;
                const int start = (available >= count)
                    ? std::clamp(h0 + firstK, left, right - count)
                    : left;
                const int used = std::min(count, available);
                for (int k = 0; k < used; ++k) {
                    const int hh = start + k;
                    sum += candidateResidualAt(plane, hh);
                }
                return sum / (double)std::max(1, used);
            };
            auto candidateFourMeanAt = [&](int plane, int h0) -> double {
                return candidateMeanAt(plane, h0, -1, 2);
            };
            auto candidatePlatformResidualAt = [&](int plane,
                                                    int h0) -> double {
                return candidateMeanAt(plane, h0, -3, 4);
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
                const int count = std::min(4, right - left);
                const int start = (right - left >= 4)
                    ? std::clamp(h0 - 1, left, right - 4)
                    : left;
                for (int k = 0; k < count; ++k) {
                    const int hh = start + k;
                    const int xx = hh - left;
                    const double complete = completeAt(hh);
                    if (!std::isfinite(complete))
                        return std::numeric_limits<double>::quiet_NaN();
                    middle += complete - platform[xx];
                }
                return center - middle / (double)std::max(1, count);
            };

            // Carrier-basis cleanliness: 1 - (AC energy explained by the carrier
            // basis). This is a direct waveform measurement, not an aggregate
            // "confidence" whose provenance is hidden.
            // Cycle-integrated over a complete 4-sample window, so it does NOT
            // flicker at carrier rate. Remove the window mean before both the
            // projection and norm: DC cannot project onto a complete carrier
            // cycle, and it must not inflate the denominator and make a
            // DC-displaced candidate look artificially clean. The selected
            // coarse is the decomposition reference for this mode.
            auto carrierCleanlinessOf = [&](int plane, int h0) -> double {
                const int hs = (right - left >= 4)
                    ? std::clamp(h0, left, right - 4)
                    : left;
                double hf4[4], s4[4], c4[4];
                double meanHF = 0.0;
                for (int j = 0; j < 4; ++j) {
                    const int hh = std::min(right - 1, hs + j);
                    const double dc = coarseRow[hh - left];
                    hf4[j] = planeY(plane, hh) - dc;
                    // Index the carrier basis by the grammar sample class, NOT
                    // hh & 3. The locked demod (the basis these LUTs were built
                    // for) uses carrierSampleClass(line, h); a raw-position
                    // index applies a per-line rotation, making cleanliness
                    // line-dependent -> a line-alternating election penalty
                    // (checkerboard) on luma transitions.
                    const int idx = carrierSampleClass(line, hh);
                    s4[j] = spLUT_locked[idx];
                    c4[j] = cpLUT_locked[idx];
                    meanHF += hf4[j];
                }
                meanHF *= 0.25;
                double dotS = 0.0, dotC = 0.0, nrm = 0.0;
                for (int j = 0; j < 4; ++j) {
                    const double ac = hf4[j] - meanHF;
                    dotS += ac * s4[j];
                    dotC += ac * c4[j];
                    nrm  += ac * ac;
                }
                const double carrierE =
                    (basisSN > 1e-9 ? dotS * dotS / basisSN : 0.0) +
                    (basisCN > 1e-9 ? dotC * dotC / basisCN : 0.0);
                return std::clamp(1.0 - carrierE / (nrm + 1e-9), 0.0, 1.0);
            };

            // Cycle-integrated carrier remaining in raw - candidate Y. This is
            // the amount that candidate would still publish as chroma, measured
            // on the locked carrier basis over a complete four-sample cycle.
            // It is an amplitude measurement with explicit provenance, not a
            // candidate label or an aggregate quality judgment.
            auto residualCarrierMagnitudeOf = [&](int plane, int h0) -> double {
                const int hs = (right - left >= 4)
                    ? std::clamp(h0, left, right - 4)
                    : left;
                double dotS = 0.0, dotC = 0.0;
                for (int j = 0; j < 4; ++j) {
                    const int hh = std::min(right - 1, hs + j);
                    const double residualCarrier =
                        (double)rawLine[hh] - planeY(plane, hh);
                    const int idx = carrierSampleClass(line, hh);
                    dotS += residualCarrier * spLUT_locked[idx];
                    dotC += residualCarrier * cpLUT_locked[idx];
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
                const double *coh = nullptr;
                const float *cohLic = nullptr;
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
                n.cc = residualVideo ? lockedCarrierComposite_line(l) : nullptr;
                n.clp = clpbuffer[srcBuf].pixel[l];
                n.ret = carrierRetracted_line(l);
                n.coarse = coarseFloor_line(l);
                n.analysis = carrierAnalysis_line(l);
                n.grammar = carrierGrammarLine(l);
                // Prebuilt once in the residual carrier prepass above; a null
                // row means the prepass could not build it (same gating as the
                // former inline reconstruction).
                n.coh = lockedResidualCarrier_line(l);
                n.cohLic = lockedResidualCarrierLicense_line(l);
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
                                    const double *cohP,
                                    const float *cohLicP,
                                    const double *coaP, int hh, double &out) -> bool {
                if (!rawP || !coaP) return false;
                const int xx = hh - left;
                double v[3]; int n = 0;
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
                if (cohP) {
                    auto residualComplete = [&](int hk) {
                        const int xk = hk - left;
                        const double c = cohP[xk];
                        return std::isfinite(c)
                            ? (double)rawP[hk] - residualCarrierForCandidate(
                                  cohLicP, xk, c)
                            : std::numeric_limits<double>::quiet_NaN();
                    };
                    appendUnique(completeTopAt(residualComplete, coaP, hh));
                }
                if (n == 0) return false;
                if (n == 1) out = v[0];
                else if (n == 2) out = 0.5 * (v[0] + v[1]);
                else {
                    const double d01 = std::fabs(v[0] - v[1]);
                    const double d02 = std::fabs(v[0] - v[2]);
                    const double d12 = std::fabs(v[1] - v[2]);
                    out = (d01 <= d02 && d01 <= d12) ? 0.5 * (v[0] + v[1])
                        : (d02 <= d01 && d02 <= d12) ? 0.5 * (v[0] + v[2])
                                                     : 0.5 * (v[1] + v[2]);
                }
                return true;
            };

            // ld-disc-stacker primitives (mode 3/6), specialised for the small
            // candidate set. medoid = robust self-center; closest = reconcile a
            // nomination to it. closestImage nominates the candidate nearest a
            // neighbour, with carrier-basis cleanliness as a capped penalty and
            // image-supported HF as a separately capped preference.  The image
            // preference is deliberately not a carrier classifier: it asks only
            // whether a candidate's luma HF continues into the local picture.
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
            auto closestImageD = [](const double *a, const double *carrierCleanliness,
                                    const double *imagePref,
                                    const double *crossColorReturnEvidence,
                                    int n,
                                    int referenceCount,
                                    double target, double phaseCap,
                                    double imageCap) -> double {
                double sw[5];
                const int refN = std::clamp(referenceCount, 1, n);
                for (int i = 0; i < refN; ++i)
                    sw[i] = carrierCleanliness[i];
                for (int i = 0; i < refN; ++i)
                    for (int j = i + 1; j < refN; ++j)
                        if (sw[j] < sw[i]) std::swap(sw[i], sw[j]);
                const double medianW = sw[refN / 2];
                double best = a[0]; double bestCost = 1e300;
                for (int i = 0; i < n; ++i) {
                    double dist = std::fabs(target - a[i]);
                    if (medianW > 0.0)
                        dist += (std::max(0.0, medianW - carrierCleanliness[i]) /
                                 medianW) * phaseCap;
                    dist -= std::clamp(imagePref[i], 0.0, 1.0) * imageCap;
                    // This term is already an evidence-bounded carrier
                    // reduction in sample units. It cannot DQ a candidate, and
                    // its IRE cap keeps geometry dominant when the candidate is
                    // far from the image tally.
                    dist -= std::max(0.0, crossColorReturnEvidence[i]);
                    if (dist < bestCost) { bestCost = dist; best = a[i]; }
                }
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
                if (!residualCarrierRow && !retractedRow && ccReturn <= 0.0) {
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
                auto reconstructTop = [&](double top) {
                    return coarse + combMiddle + combPlatform + top;
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

                // Roster (with feasibility DQ). The affine/polar residual-Y
                // carrier estimate is retained as a contrast candidate against
                // retractedY. 1D remains emergency-only and is
                // admitted only if comb DQ'd.
                double candY[5]; // top-band values; name retained locally
                int    candPlane[5];
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
                    // retained value; an affine identity fallback must not
                    // count coherent Y twice.
                    for (int k = 0; k < nCand; ++k)
                        if (std::fabs(candY[k] - y) <= identityTol)
                            return;
                    candY[nCand] = y;
                    candPlane[nCand] = plane;
                    ++nCand;
                };
                const bool combOK = std::isfinite(combY) && feasible(combY);
                if (combOK)
                    addBaseCandidate(combY, 0);
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
                // Diagnostic seat exclusion (A/B only, same family as
                // LD_RETRACTED_ADMIT): LD_RESIDUALY_SEAT=0 keeps the
                // affine/polar residual candidate off the roster so a decode
                // can attribute an artifact to this seat in one variable.
                static const bool residualYSeat = []{
                    const char *s = std::getenv("LD_RESIDUALY_SEAT");
                    return !(s && s[0] == '0');
                }();
                if (affineResidualCarrierRow && residualYSeat) {
                    const double c = affineResidualCarrierRow[xi];
                    const double cy = std::isfinite(c)
                        ? rawH - residualCarrierForCandidate(residualLicenseRow, xi, c)
                        : combY;
                    addBaseCandidate(cy, 2);
                }
                if (!combOK && oneDRow) {
                    const double o = oneDRow[xi];
                    const double y1 = std::isfinite(o) ? rawH - o : combY;
                    addBaseCandidate(y1, 3);
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
                    // Nothing feasible: clamp combY into the legal band.
                    const double c = rawH - combY;
                    Y[h] = (c > maxCarrierAmpSamples) ? rawH - maxCarrierAmpSamples
                         : (c < -maxCarrierAmpSamples) ? rawH + maxCarrierAmpSamples
                         : combY;
                    continue;
                }
                // Establish the base population without the derived return.
                double lo = candY[0], hi = candY[0], sum = candY[0];
                for (int k = 1; k < nCand; ++k) {
                    lo = std::min(lo, candY[k]);
                    hi = std::max(hi, candY[k]);
                    sum += candY[k];
                }
                const double baseMean = sum / nCand;
                const bool baseAgrees = hi - lo <= agreeTol;
                if (baseAgrees && !returnedFeasible) {
                    Y[h] = reconstructTop(baseMean);
                    continue;
                }

                // Robust center: medoid (min sum of absolute distances).
                double center = baseMean;
                if (!baseAgrees) {
                    center = candY[0];
                    double bestTot = 1e300;
                    for (int i = 0; i < nCand; ++i) {
                        double t = 0.0;
                        for (int j = 0; j < nCand; ++j)
                            t += std::fabs(candY[i] - candY[j]);
                        if (t < bestTot) { bestTot = t; center = candY[i]; }
                    }
                }

                // Inlier DQ around the center.
                int inIdx[5];
                int nIn = 0;
                for (int k = 0; k < nCand; ++k)
                    if (std::fabs(candY[k] - center) <= inlierTol)
                        inIdx[nIn++] = k;
                // The named cross-colour mask, rather than distance from the
                // base center, admits returned Y. A distance gate here removes
                // exactly the strong HF that the return exists to recover.
                const bool returnedAdmitted = returnedFeasible;
                if (nIn == 1 && !returnedAdmitted) {
                    Y[h] = reconstructTop(candY[inIdx[0]]);
                    continue;
                }

                // Inlier HF set + per-inlier carrier-basis cleanliness. This is
                // a cautionary term, not the positive reason to select HF.
                double inHF[5], inCarrierCleanliness[5];
                double inCrossColorReturnEvidence[5] = {
                    0.0, 0.0, 0.0, 0.0, 0.0
                };
                for (int k = 0; k < nIn; ++k) {
                    inHF[k] = candY[inIdx[k]];
                    inCarrierCleanliness[k] =
                        carrierCleanlinessOf(candPlane[inIdx[k]], h);
                }
                const int baseNIn = nIn;

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
                // Residual/retracted Y can therefore receive
                // this evidence when they already outperform nominal returned
                // Y; a label cannot win an advantage its samples did not earn.
                if (ccReturn > 0.0) {
                    const double combCarrierMagnitude =
                        residualCarrierMagnitudeOf(0, h);
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
                                residualCarrierMagnitudeOf(plane, h));
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
                                           const double *cohP,
                                           const float *cohLicP,
                                           const double *geometryFloor,
                                           const double *dcFloor,
                                           int hh) {
                    if (nDir >= 4 || !dcFloor) return;
                    double geometryHF, imageHF;
                    if (neighborHFAt(rawP, ccP, clpP, retP, cohP, cohLicP,
                                     geometryFloor, hh, geometryHF) &&
                        neighborHFAt(rawP, ccP, clpP, retP, cohP, cohLicP,
                                     dcFloor, hh, imageHF)) {
                        dirHF[nDir] = geometryHF;
                        dirImageHF[nDir] = imageHF;
                        ++nDir;
                    }
                };
                appendDirection(northRows.raw, northRows.cc, northRows.clp,
                                northRows.ret, northRows.coh, northRows.cohLic,
                                northRows.coarse, northRows.coarse, h);
                appendDirection(southRows.raw, southRows.cc, southRows.clp,
                                southRows.ret, southRows.coh, southRows.cohLic,
                                southRows.coarse, southRows.coarse, h);
                if (h - 1 >= left)
                    appendDirection(rawLine, carrierComp, clpLine,
                                    retractedRow, affineResidualCarrierRow, residualLicenseRow,
                                    coarseRow, coarseRow, h - 1);
                if (h + 1 < right)
                    appendDirection(rawLine, carrierComp, clpLine,
                                    retractedRow, affineResidualCarrierRow, residualLicenseRow,
                                    coarseRow, coarseRow, h + 1);

                // Positive image evidence for the elected top in the selected
                // decomposition coordinate. The one selected coarse is already
                // the LF output platform; it is never an election contestant.
                double imagePref[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
                if (nDir > 0) {
                    double imageHF[5];
                    for (int k = 0; k < nIn; ++k)
                        imageHF[k] = inHF[k];

                    double minMag = std::fabs(imageHF[0]);
                    double maxMag = minMag;
                    for (int k = 1; k < baseNIn; ++k) {
                        const double m = std::fabs(imageHF[k]);
                        minMag = std::min(minMag, m);
                        maxMag = std::max(maxMag, m);
                    }
                    const double magSpan = maxMag - minMag;
                    const double continuationTol =
                        std::max(0.5, configuration.tunables.PRODUCE_Y_HF_CONTINUATION_IRE)
                        * irescale;
                    for (int k = 0; k < nIn; ++k) {
                        double continuation = 0.0;
                        for (int d = 0; d < nDir; ++d) {
                            const double match = 1.0 -
                                std::clamp(std::fabs(imageHF[k] - dirImageHF[d]) /
                                           continuationTol, 0.0, 1.0);
                            continuation = std::max(continuation, match);
                        }
                        const double retained = (magSpan > 1e-9)
                            ? std::clamp((std::fabs(imageHF[k]) - minMag) /
                                         magSpan, 0.0, 1.0)
                            : 0.0;
                        imagePref[k] = continuation * retained;
                    }
                }

                // Each neighbour nominates a real candidate. Carrier-basis
                // cleanliness can caution; image continuation can affirm HF.
                // Both terms are capped, so neither can defeat a large
                // geometric disagreement with the neighbour.
                const double imagePrefCap =
                    std::max(0.0, configuration.tunables.PRODUCE_Y_HF_IMAGE_PREFERENCE_IRE)
                    * irescale;
                double noms[4]; int nNom = 0;
                for (int d = 0; d < nDir; ++d)
                    noms[nNom++] = closestImageD(
                        inHF, inCarrierCleanliness, imagePref,
                        inCrossColorReturnEvidence, nIn, baseNIn,
                        dirHF[d],
                        phasePenSamp, imagePrefCap);

                double resultHF;
                double diagNeighborAnchor = std::numeric_limits<double>::quiet_NaN();
                double diagDecisionAnchor = std::numeric_limits<double>::quiet_NaN();
                if (nNom > 0) {
                    const double neighborSelection = closestD(noms, nNom, selfAnchor);
                    // Mode-3 smart mean supplies the neighbour-side anchor.
                    double s = 0.0; int c = 0;
                    for (int k = 0; k < nIn; ++k)
                        if (std::fabs(inHF[k] - neighborSelection) <= inlierTol) {
                            s += inHF[k]; ++c;
                        }
                    const double neighborAnchor = (c > 0) ? s / c : neighborSelection;
                    // Reconcile the mode-6 self/neighbor anchors, then close
                    // the election onto an actual candidate.  Returning the
                    // anchor average here would manufacture a softened value
                    // after the image preference had selected sharper HF.
                    const double decisionAnchor = 0.5 * (selfAnchor + neighborAnchor);
                    diagNeighborAnchor = neighborAnchor;
                    diagDecisionAnchor = decisionAnchor;
                    resultHF = closestImageD(
                        inHF, inCarrierCleanliness, imagePref,
                        inCrossColorReturnEvidence, nIn, baseNIn,
                        decisionAnchor,
                        phasePenSamp, imagePrefCap);
                } else {
                    // With no spatial nomination, still close onto a real
                    // candidate. The medoid/mean is an anchor, not an output.
                    resultHF = closestImageD(
                        inHF, inCarrierCleanliness, imagePref,
                        inCrossColorReturnEvidence, nIn, baseNIn,
                        selfAnchor,
                        phasePenSamp, imagePrefCap);
                }

                Y[h] = reconstructTop(resultHF);

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
                    const double resTop = affineResidualCarrierRow
                        ? candidateTopAt(2, h) / irescale
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
                    const double returnedImagePref = returnedIndex >= 0
                        ? imagePref[returnedIndex]
                        : std::numeric_limits<double>::quiet_NaN();
                    const double returnedCcEvidence = returnedIndex >= 0
                        ? inCrossColorReturnEvidence[returnedIndex] / irescale
                        : std::numeric_limits<double>::quiet_NaN();
                    std::fprintf(stderr,
                        "PYDIAG line=%d h=%d xi=%d vstep=%d combOK=%d "
                        "1Dexcl=%d nCand=%d roster=[%s] "
                        "selfA=%.2f nbrA=%.2f decA=%.2f winPlane=%d "
                        "winTop=%.2f coarseIRE=%.2f combMiddle=%.2f "
                        "combPlatformResidual=%.2f ccRaw=%.2f ccRet=%.2f "
                        "monoTop=%.2f combTop=%.2f retrTop=%.2f "
                        "resTop=%.2f oneDTop=%.2f retnTop=%.2f "
                        "retnImg=%.3f retnCcEv=%.2f "
                        "nbrImgTop=[%s]\n",
                        line, h, xi, verticalStep, combOK ? 1 : 0,
                        (combOK && oneDRow) ? 1 : 0, nCand, roster,
                        selfAnchor / irescale, diagNeighborAnchor / irescale,
                        diagDecisionAnchor / irescale, winnerPlane,
                        resultHF / irescale, coarse / irescale,
                        combMiddle / irescale, combPlatformResidual,
                        ccMeasuredHere, ccReturn, monoTop, combTop, retrTop,
                        resTop, oneDTop,
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
                                                double *gateOut,
                                                double gateGain) const
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
        // gateGain sweeps the snap aggressiveness: <1 softens the contour
        // (weaker steps stay on the boxcar ramp), >1 snaps weaker steps too.
        const double gate =
            std::clamp(smoothStep01((stepIRE - 1.25) / 2.75) * gateGain,
                       0.0, 1.0);
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
    static const double parallaxRepairMaxDeltaIRE = []{
        const char *s = std::getenv("LD_1D_PARALLAX_MAX_DELTA_IRE");
        return s ? std::atof(s) : 0.35;
    }();
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
    if (carrierEligible_flat.size() < need)
        carrierEligible_flat.assign(need, std::uint8_t{0});
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
        auto *eligibleRow   = carrierEligible_flat.data()
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
            }
            continue;
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
                int carrierSampleCount = 0;

                double refinedMean = 0.0;
                double minRefined = 1e300;
                double maxRefined = -1e300;

                for (int k = 0; k < 4; ++k) {
                    const int xi = s + k;
                    refinedMean += refinedY[xi];
                    minRefined = std::min(minRefined, refinedY[xi]);
                    maxRefined = std::max(maxRefined, refinedY[xi]);

                    // Registration rejection is a DQ, not a zero-valued
                    // carrier observation.  A zero target would still enter
                    // the normal matrix and attenuate the legal samples in a
                    // mixed window.  Remove the sample from both sides of the
                    // solve; support/rank below decides whether enough
                    // independent carrier phases remain to solve and grade.
                    if (analysisRow[xi].scheduleConformance ==
                        lddecode::CarrierScheduleConformance::ScheduleIllegal)
                    {
                        continue;
                    }

                    const double bI = basisI[xi];
                    const double bQ = basisQ[xi];
                    const double residual = rawWhole[xi] - refinedY[xi];

                    sII += bI * bI;
                    sIQ += bI * bQ;
                    sQQ += bQ * bQ;
                    sIY += bI * residual;
                    sQY += bQ * residual;
                    ++carrierSampleCount;
                }

                refinedMean *= 0.25;

                double fitI = 0.0;
                double fitQ = 0.0;
                const double det = sII * sQQ - sIQ * sIQ;
                const bool fitValid =
                    carrierSampleCount >= 3 && std::fabs(det) > 1e-9;
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
                        if (carrierSampleCount < 3)
                            ++dzWinRank;   // killed by illegal-sample removal
                        else
                            ++dzWinDet;    // killed by singular normal matrix
                    }
                }

                double errSq = 0.0;
                double basis01 = 0.0; // +-+-
                double basis02 = 0.0; // ++--
                double basis03 = 0.0; // +--+
                double fitAbs = 0.0;
                int gradedSampleCount = 0;

                for (int k = 0; k < 4; ++k) {
                    const int xi = s + k;
                    if (analysisRow[xi].scheduleConformance ==
                        lddecode::CarrierScheduleConformance::ScheduleIllegal)
                    {
                        continue;
                    }

                    const double bI = basisI[xi];
                    const double bQ = basisQ[xi];

                    const double fit = fitI * bI + fitQ * bQ;
                    const double residual = rawWhole[xi] - refinedY[xi];
                    const double e = residual - fit;

                    errSq += e * e;
                    fitAbs += std::fabs(fit);
                    ++gradedSampleCount;

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

                const double gradeInv = gradedSampleCount > 0
                    ? 1.0 / static_cast<double>(gradedSampleCount)
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
                const bool sampleCarrierEligible =
                    analysisRow[xi].scheduleConformance !=
                    lddecode::CarrierScheduleConformance::ScheduleIllegal;
                eligibleRow[xi] = sampleCarrierEligible ? std::uint8_t{1}
                                                        : std::uint8_t{0};

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
                     sampleCarrierEligible && pass < 2 && viewCount == 0;
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
                    if (!sampleCarrierEligible)
                        ++dzIneligible;
                    if (viewCount == 0) {
                        ++dzDead;
                        // For an eligible pixel, pass 1 harvests any covering
                        // winFitValid window, so viewCount==0 there means the
                        // fit was starved (no covering window survived rank/det).
                        if (!sampleCarrierEligible)
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
                    if (meanCount > 0 &&
                        !windowStraddles(std::clamp(xi - 1, 0, meanCount - 1))) {
                        const int centeredStart = std::clamp(xi - 1, 0, meanCount - 1);
                        movingResidualSample = rawWhole[xi] - winFloor[centeredStart];
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

                // Consume the same short-fit-selected residual feasibility that
                // repaired locked 1D. The four-view fit remains the retraction
                // candidate, but where it falls outside a discriminating,
                // moving-supported subset it receives the same small bounded
                // correction. Shared diagnostics constrain this application;
                // no residual or fit sample is substituted wholesale.
                bool sharedConstraintApplied = false;
                double sharedDelta = 0.0;
                const auto &sharedResidual = analysisRow[xi].residual;
                const int sharedSurvivors = sharedResidual.survivorCount();
                const bool sharedUseful =
                    sampleCarrierEligible &&
                    sharedResidual.valid &&
                    sharedSurvivors > 0 &&
                    sharedSurvivors < sharedResidual.optionCount &&
                    sharedResidual.movingCompatible;
                if (sharedUseful &&
                    !(cf >= sharedResidual.survivorLo &&
                      cf <= sharedResidual.survivorHi))
                {
                    const double target = std::clamp(
                        cf,
                        sharedResidual.survivorLo,
                        sharedResidual.survivorHi);
                    const double maxDelta =
                        std::max(0.0, parallaxRepairMaxDeltaIRE) * irescale;
                    sharedDelta = std::clamp(
                        target - cf,
                        -maxDelta,
                        maxDelta);
                    cf += sharedDelta;
                    sharedConstraintApplied = sharedDelta != 0.0;
                }

                if (crDiagLine == line && crDiagC0 >= 0 &&
                    xi >= crDiagC0 &&
                    xi <= (crDiagC1 < 0 ? crDiagC0 : crDiagC1))
                {
                    std::fprintf(stderr,
                        "CARRIERRETRACTREPAIR line=%d rel=%d before=%.6f "
                        "after=%.6f applied=%d deltaIRE=%.6f "
                        "optionCount=%d survivorCount=%d survivorLo=%.6f "
                        "survivorHi=%.6f movingCompatible=%d\n",
                        line, xi, baselineCf, cf,
                        sharedConstraintApplied ? 1 : 0,
                        sharedDelta * invIreScale,
                        static_cast<int>(sharedResidual.optionCount),
                        sharedSurvivors,
                        sharedResidual.survivorLo,
                        sharedResidual.survivorHi,
                        sharedResidual.movingCompatible ? 1 : 0);
                }

                carrierFit[xi] = cf;
                flattened[xi] = rawWhole[xi] - cf;

                fitRow[xi] = static_cast<float>(cf);
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
    // Return path: align carrierFit with the repaired 1D carrier.
    //
    // The fit was solved from raw before Pass 1.5 existed for this frame;
    // the feasibility repair then moved the 1D carrier by certified,
    // bounded deltas.  Without folding those same moves into the model,
    // the retraction (and the witness above it) consumes a pre-repair
    // carrier while every other client consumes the repaired one.  This
    // applies the published deltas — the identical certified moves, not a
    // new estimate — so 1D source authority is unchanged.  Deltas are
    // sparse and zero elsewhere.
    // ---------------------------------------------------------------
    const lddecode::CombReachSourceFrame carrierFitSource =
        lddecode::makeCarrierFitScalarReachSource();

    for (int line = firstLine; line < lastLine; ++line) {
        const float *deltaRow = locked1DParallaxRepairDelta_line(line);
        if (!deltaRow)
            continue;
        float *fitRow = carrierFit_flat.data()
                        + static_cast<size_t>(line) * demodWidth;
        for (int rel = 0; rel < width; ++rel) {
            if (deltaRow[rel] != 0.0f)
                fitRow[rel] += deltaRow[rel];
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

        auto legalOppositeCarrierFitReach = [&](int targetLine,
                                                const CombCarrierGrammar *target) {
            if (!target || !target->grammarLocked)
                return false;
            const lddecode::CombReachReply reach = combReachIndex.query(
                {line, targetLine, left, left,
                 lddecode::CombReachUse::FrameScalarCancel,
                 carrierFitSource});
            return reach.allowScalarCancel && reach.mayBecomeVideo &&
                   reach.carrierRelation ==
                       lddecode::CarrierPhaseRelation::Opposite;
        };

        const bool haveAbove = legalOppositeCarrierFitReach(lineAbove, gAbove);
        const bool haveBelow = legalOppositeCarrierFitReach(lineBelow, gBelow);

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
        const std::uint8_t *eligRow =
            carrierEligible_flat.data() + static_cast<size_t>(line) * demodWidth;
        const std::uint8_t *eligAbove = haveAbove
            ? (carrierEligible_flat.data() +
               static_cast<size_t>(lineAbove) * demodWidth)
            : nullptr;
        const std::uint8_t *eligBelow = haveBelow
            ? (carrierEligible_flat.data() +
               static_cast<size_t>(lineBelow) * demodWidth)
            : nullptr;

        // Reach gate: determines per-pixel cancellation strength toward
        // each neighbor line.  Inlined here (was a lambda) to keep the
        // data flow visible in the fused loop below.
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
        auto reachGate = [&](int xi, const float *neighborFit,
                             const float *neighborFloor,
                             const std::uint8_t *neighborElig) {
            if (!neighborFit || !neighborFloor || !neighborElig)
                return 0.0;
            if (!eligRow[xi] || !neighborElig[xi])
                return 0.0;

            const double lumaDiffIRE =
                std::fabs(static_cast<double>(floorRow[xi]) -
                          static_cast<double>(neighborFloor[xi])) * invIreScale;

            auto pairEligibleAt = [&](int x) {
                return x >= 0 && x < width &&
                    eligRow[x] && neighborElig[x];
            };
            int xj = xi + 1;
            if (!pairEligibleAt(xj))
                xj = xi - 1;
            if (!pairEligibleAt(xj))
                return 0.0;
            const double c0 = static_cast<double>(fitRow[xi]);
            const double c1 = static_cast<double>(fitRow[xj]);
            const double n0 = static_cast<double>(neighborFit[xi]);
            const double n1 = static_cast<double>(neighborFit[xj]);

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
                haveAbove
                    ? reachGate(xi, fitAbove, floorAbove, eligAbove)
                    : 0.0;
            wBelowRaw[xi] =
                haveBelow
                    ? reachGate(xi, fitBelow, floorBelow, eligBelow)
                    : 0.0;
        }

        // Pass B: inline 5-tap smooth + decision blend + combRow output.
        // The smooth kernel is [1,2,3,2,1] (sum = 9 in the interior).
        // At edges, the kernel is clamped and the divisor adjusts.
        constexpr double kWeights[5] = {1.0, 2.0, 3.0, 2.0, 1.0};

        for (int xi = 0; xi < width; ++xi) {
            // A refused center sample owns no carrier column.  Keep the DQ
            // explicit at the interline publication boundary as well as in
            // the fit, so smoothing of neighboring reach gates cannot give
            // it a route back into video.
            if (!eligRow[xi]) {
                combRow[xi] = 0.0f;
                continue;
            }

            // Ownership weight for the un-cancelled remainder (see the emit
            // block below): the OPERAND schedule-compatibility license.  The
            // confiscation operand is this line's carrierFit; the license
            // tests the operand itself against each certified-Opposite
            // partner's fit over one quadrature pair.  On-schedule chroma
            // inverts (signed corr -> -1: licensed, keep the confiscation,
            // microscopic runs included); a fit that absorbed alien luma
            // MATCHES its partner where the schedule demands inversion
            // (cube grid: corr -> +1, license 0, energy stays in Y).  A
            // quiet operand licenses at 1 (confiscating ~nothing is
            // harmless); a loud operand with no observable partner fails
            // closed.  No raw-bandpass votes or axis corroboration.
            double ownedFallback;
            {
                const int xj = (xi + 1 < width) ? xi + 1
                             : (xi > 0 ? xi - 1 : xi);
                const double c0 = static_cast<double>(fitRow[xi]);
                const double c1 = static_cast<double>(fitRow[xj]);
                const double e0 = c0 * c0 + c1 * c1;
                const double ampFloor = 3.0 * irescale;      // 3 IRE envelope
                const double eFloor = ampFloor * ampFloor;
                if (e0 < eFloor) {
                    ownedFallback = 1.0;   // harmless confiscation
                } else {
                    double best = 9.0;     // most-legal signed corr observed
                    auto observePartner = [&](const float *pf) {
                        if (!pf) return;
                        const double n0 = static_cast<double>(pf[xi]);
                        const double n1 = static_cast<double>(pf[xj]);
                        const double eP = n0 * n0 + n1 * n1;
                        if (eP < eFloor) return;
                        const double sc =
                            (c0 * n0 + c1 * n1) / std::sqrt(e0 * eP);
                        if (best > 1.0 || sc < best) best = sc;
                    };
                    observePartner(fitAbove);
                    observePartner(fitBelow);
                    ownedFallback = (best <= 1.0)
                        ? lddecode::scheduleAlternationLicense(best)
                        : 0.0;             // loud, unobservable: fail closed
                }
            }

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
            double amp = 0.0;
            auto includeRegisteredAmp = [&](const float *fit,
                                            const std::uint8_t *elig,
                                            int x) {
                if (fit && elig && elig[x])
                    amp = std::max(amp, std::fabs(static_cast<double>(fit[x])));
            };
            includeRegisteredAmp(fitRow, eligRow, xi);
            includeRegisteredAmp(fitRow, eligRow, xm);
            includeRegisteredAmp(fitRow, eligRow, xp);
            includeRegisteredAmp(fitAbove, eligAbove, xi);
            includeRegisteredAmp(fitAbove, eligAbove, xm);
            includeRegisteredAmp(fitAbove, eligAbove, xp);
            includeRegisteredAmp(fitBelow, eligBelow, xi);
            includeRegisteredAmp(fitBelow, eligBelow, xm);
            includeRegisteredAmp(fitBelow, eligBelow, xp);
            const double blend = smoothStep01((amp * invIreScale - 8.0) / 10.0);

            // Final gated cancellation.
            // Do not let the horizontal gate smoother cross a per-pixel DQ
            // on the partner row.  The smooth value is only usable where the
            // exact partner sample remains registered as carrier-capable.
            const bool aboveEligible = eligAbove && eligAbove[xi];
            const bool belowEligible = eligBelow && eligBelow[xi];
            const double wAbove = aboveEligible
                ? wAboveRaw[xi] * (1.0 - blend) + smoothAbove * blend
                : 0.0;
            const double wBelow = belowEligible
                ? wBelowRaw[xi] * (1.0 - blend) + smoothBelow * blend
                : 0.0;
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

                // Ownership rule: the un-cancelled remainder is carrier this
                // stage would confiscate from Y.  It may only be confiscated
                // where carrier is owned.  carrierIllegalProof is nonzero ONLY
                // where the scanner has positively proven (two-axis) the energy
                // is luma-by-law; it is zero through the entire ambiguous middle
                // (so genuine/slanted chroma is untouched) and zero for proven-
                // legal inverting chroma (so the amplitude-neutral retention
                // invariant above is preserved).  Withdraw only the standdown
                // remainder, never the active cancellation.
                combRow[xi] = static_cast<float>(
                    static_cast<double>(fitRow[xi]) * (1.0 - strength) *
                        ownedFallback +
                    cancelled * strength);
            } else {
                // Reach fully closed: the whole sample is un-cancelled carrier.
                // Confiscate it only where carrier is owned.
                combRow[xi] = static_cast<float>(
                    static_cast<double>(fitRow[xi]) * ownedFallback);
            }
        }
    }

    // ---------------------------------------------------------------
    // Final publication: retracted Y derives from the promoted carrier model,
    // not from the workprint fit. flatFloor has already served Pass 2 and has
    // no downstream consumer.
    // ---------------------------------------------------------------
    for (int line = firstLine; line < lastLine; ++line) {
        const quint16 *rawLine =
            rawbuffer.data() + static_cast<size_t>(line) * videoParameters.fieldWidth;
        const float *combRow = combedCarrier_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        float *retractedRow = carrierRetracted_flat.data()
                              + static_cast<size_t>(line) * demodWidth;

        for (int xi = 0; xi < width; ++xi) {
            retractedRow[xi] = static_cast<float>(
                static_cast<double>(rawLine[left + xi]) -
                static_cast<double>(combRow[xi]));
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
