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

#include "cadencedefs.h"
#include "comb.h"
#include "combmath.h"
#include "feasibleband.h"

#include <algorithm>
#include <array>
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

inline bool lurchPinEnabled()
{
    static const bool on = []{
        const char *s = std::getenv("LDCD_LURCH_PIN");
        return (s && std::atoi(s) != 0);
    }();
    return on;
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
    const double Rb0 = -K * CAL_EPS_SAMPLES;
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
        const bool buildSharp =
            configuration.yElection.lsc && !lockedLumaSharp_flat.empty();

        for (int line = firstLine; line < lastLine; ++line) {
            const quint16 *rawLine = rawbuffer.data() + line * fullWidth;
            buildCompositeLumaDecompositionLine(rawLine, left, width,
                                                lockedLumaBaseY4_line(line),
                                                nullptr,
                                                lockedLumaSmooth_line(line));

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
                solveLurchYCurve(line, boxcar, width - 3, width, sharp);
            } else {
                const int lastStart = width - 4; // last legal aperture start

                for (int xi = 0; xi < width; ++xi) {
                    const int s0 = std::clamp(xi - 2, 0, lastStart);
                    const int s1 = std::clamp(xi - 1, 0, lastStart);
                    sharp[xi] = 0.5 * (boxcar[s0] + boxcar[s1]);
                }
                const std::vector<LurchStepRun> corrRuns =
                    corroborateLurchEdges(line);
                applyLurchSteps(corrRuns, boxcar, width - 3,
                                width, 1.0, sharp, nullptr);
            }

        }
        lockedLumaCacheValid = true;
    }

    return;
}


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

    buildBandFacts();

    {
        std::vector<double> nhLuma(width, 0.0);
        for (int line = first; line < last; ++line) {
            float *hDelta = lockedLumaHDeltaIRE_line(line);
            if (!hDelta)
                continue;

            const double *bp   = locked1DRawBandpass_line(line);
            const double *wLaw = bandWLaw_line(line);
            const double *keep = bandKeep_line(line, 2);

            if (bp && wLaw && keep) {
                const quint16 *rawLine =
                    rawbuffer.data() + static_cast<size_t>(line) * fullWidth;
                for (int rel = 0; rel < width; ++rel) {
                    nhLuma[rel] = static_cast<double>(rawLine[left + rel]) -
                                  bp[rel] * wLaw[rel] * keep[rel];
                }
            } else if (const double *smooth = lockedLumaSmooth_line(line)) {
                std::copy(smooth, smooth + width, nhLuma.begin());
            } else {
                continue;
            }

            for (int rel = 0; rel < width; ++rel) {
                const int rm = std::max(0, rel - 2);
                const int rp = std::min(width - 1, rel + 2);
                hDelta[rel] = static_cast<float>(
                    std::fabs(nhLuma[rp] - nhLuma[rm]) * invIreScale);
            }
        }
    }
}

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
            lddecode::projectExpressibleChromaEnvelope(envI.data(), nullptr,
                                                       width, sEnvI.data(), 0);
            lddecode::projectExpressibleChromaEnvelope(envQ.data(), nullptr,
                                                       width, sEnvQ.data(), 1);
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
        for (int x = 0; x < width; ++x)
            leakRow[x] = -0.25 * kappa[x] + envExcess[x];

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

}

static void ldcdApplyPhaseSnap(const std::vector<double> &est,
                               const std::vector<double> &ref,
                               std::vector<double> &out,
                               int width, double irescale,
                               double ampMinIRE, double ampTauIRE,
                               bool clampRatio,
                               const double *gate = nullptr);

static void ldcdSideCoherenceAlpha(const std::vector<double> &a,
                                   const std::vector<double> &b,
                                   int width, std::vector<double> &alpha)
{
    constexpr int kSnapHalf = 4;
    static const int cB[4] = { 1, 0, -1, 0 };
    static const int sB[4] = { 0, 1, 0, -1 };
    std::fill(alpha.begin(), alpha.end(), 0.0);
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

    if (static_cast<int>(scratch_preI.size()) < width)
        scratch_preI.resize(width, 0.0);
    if (static_cast<int>(scratch_preQ.size()) < width)
        scratch_preQ.resize(width, 0.0);

    auto clamp01 = [](double v) {
        return std::clamp(v, 0.0, 1.0);
    };

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

        const bool certifiedDef =
            certifiedOneDLevel() >= 1 && frameHasExactCoverage() &&
            certifiedDefLine(line);
        const float *certExRow =
            certifiedDef ? exactCarrierRow(line) : nullptr;
        const bool useCertRow = (certExRow != nullptr);

        if (useCertRow) {
            if (float *impurityRow = carrierImpurity_line(line))
                std::fill(impurityRow, impurityRow + width, 0.0f);
            if (float *repairStrengthRow =
                    locked1DParallaxRepairStrength_line(line))
                std::fill(repairStrengthRow, repairStrengthRow + width, 0.0f);
            if (float *repairDeltaRow =
                    locked1DParallaxRepairDelta_line(line))
                std::fill(repairDeltaRow, repairDeltaRow + width, 0.0f);
        } else {
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

                if (carrierAnalysis[rel].scheduleConformance ==
                    lddecode::CarrierScheduleConformance::ScheduleIllegal) {
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

        auto narrowEnvIRE = [&](int center) -> double {
            const int a = std::clamp(center - kNarrowWin / 2, 0, width);
            const int b = std::clamp(a + kNarrowWin, 0, width);
            const double n = static_cast<double>(std::max(1, b - a));
            return ((preEnv[b] - preEnv[a]) / n) * invIreScale;
        };

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
            restrainedLine[rel] = bpLine[rel];

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
        }   // end of the comp-line estimate; def lines skipped all of it

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
            // Decided once per line above, not once per sample here.
            const double sourceRaw = useCertRow
                ? static_cast<double>(certExRow[h])
                : restrainedLine[rel];
            const double source = std::isfinite(sourceRaw)
                ? sourceRaw
                : restrainedLine[rel];

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

void Comb::FrameBuffer::buildCrossColorReturn()
{
    if (!configuration.phaseCompensation) return;
    // Opt-in. Not engaged means not measured.
    if (std::max(0.0, configuration.tunables.CC_SUPPRESSION_WEIGHT) <= 0.0)
        return;


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

    static const bool bwCrossColor = []{
        const char *s = std::getenv("LDCD_BW_CROSSCOLOR");
        return s && std::atoi(s) != 0;
    }();

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

        if (bwCrossColor) {
            lddecode::projectExpressibleChromaEnvelope(
                demI.data(), nullptr, width, lawI.data(), 0);
            lddecode::projectExpressibleChromaEnvelope(
                demQ.data(), nullptr, width, lawQ.data(), 1);
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

            if (ccParallax) {
                const double keep = clamp01(
                    (parRatio[rel] - kCornerParallaxSoft) /
                    (kCornerParallaxHard - kCornerParallaxSoft));
                gA *= keep;
            }

            impurityRow[rel] = static_cast<float>(gA);
        }
    }
}

static constexpr double kCcCarrierFloorIRE = 2.0;

static constexpr double kCcCommitCutoffIRE = kCcCarrierFloorIRE;

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

    ensureScratch(scratch_preI);
    ensureScratch(scratch_preQ);

    auto finiteOrZero = [](double v) -> double {
        return std::isfinite(v) ? v : 0.0;
    };

    const double ccWeight = std::max(0.0, T.CC_SUPPRESSION_WEIGHT);
    const double giProduct = configuration.gi_product;
    const double gqProduct = configuration.gq_product;

    static const bool chromaFactsOn = []{
        const char *e = std::getenv("LDCD_CHROMA_FACTS");
        return !(e && std::atoi(e) == 0);
    }();

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
        const double *ccFactRow = factBackedCarrier_line(line);
        const double *ccAnchRow = ccFactRow;
        const double *ccObs1D = locked1DSource_line(line);

        if (prodIRow)    std::fill(prodIRow, prodIRow + width, 0.0f);
        if (prodQRow)    std::fill(prodQRow, prodQRow + width, 0.0f);
        if (carrierComp) std::fill(carrierComp, carrierComp + width, 0.0);

        for (int xi = 0; xi < width; ++xi) {
            const int h  = left + xi;
            const int ph = carrierSampleClass(line, h);
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

            if (carrierComp)
                carrierComp[xi] = finiteOrZero(src[h]);
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

            const double regionKeep = sameRegionRow
                ? std::clamp((double)sameRegionRow[xi], 0.0, 1.0)
                : 0.0;
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

            if (float *dv = ccDetectorVerdict_line(line))
                dv[xi] = (float)lumaWeight;

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

    if (ccWeight > 0.0 && !lockedCcMaskRaw_flat.empty() &&
        !lockedCcMask_flat.empty()) {
        constexpr int kCcMaskRadius = 4; // 9-tap, first null ~1.6 MHz

        static const bool ccFactsOn2 = []{
            const char *e = std::getenv("LDCD_CC_FACTS");
            return !(e && std::atoi(e) == 0);
        }();
        const int anx = (width + 127) / 128;
        const int any_ = (lastLine + 31) / 32;
        std::vector<double> auditW((size_t)anx * any_, 1.0);
        std::vector<double> auditSupp((size_t)anx * any_, 0.0);
        if (ccFactsOn2) {
            std::vector<double> claimed((size_t)anx * any_, 0.0),
                honoured((size_t)anx * any_, 0.0);
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

                double cycleLegal = 0.0;
                for (int k = -2; k <= 2; ++k)
                    cycleLegal = std::max(
                        cycleLegal,
                        grammarPassAt(std::clamp(xi + k, 0, width - 1)));

                m *= (1.0 - cycleLegal) +
                     cycleLegal * auditSuppAt(line, xi);

                out[xi] = (float)m;
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
    static const bool legalBandOn = []{
        const char *e = std::getenv("LDCD_LEGAL_BAND");
        return !(e && std::atoi(e) == 0);
    }();
    bandResidueY_flat.assign((size_t)demodLines * demodWidth, 0.0f);
    const bool legalFrameCovered =
        chromaFactsOn && frameHasExactCoverage();
    if (legalBandOn && !legalFrameCovered) {
        static std::once_flag lbInit;
        static std::vector<double> hLB;
        std::call_once(lbInit, [] {
            const double fsMHz = 14.31818, fny = fsMHz * 0.5;
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
                const double legalCar = 2.0 *
                    remodGrid4fscToComposite(line, h, i4f[xi], q4f[xi]);
                if (!std::isfinite(legalCar)) continue;
                legalPlane[(size_t)(line - firstLine) * width + xi] =
                    (float)legalCar;
            }
        }
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
                        auto bpOf = [&](const float *r, int x) {
                            return (double)r[x] -
                                   0.5 * ((double)r[x - 2] +
                                          (double)r[x + 2]);
                        };
                        const double bpP = bpOf(rP, xi);
                        const double bpN = bpOf(rN, xi);
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

        static const bool feasibleMode = []{
            const char *e = std::getenv("LDCD_RENDER_FEASIBLE");
            return e && std::atoi(e) != 0;
        }();
        if (feasibleMode) {
            std::vector<double> outI(width), outQ(width);
            lddecode::projectExpressibleChromaEnvelope(
                scratch_preI.data(), nullptr, width, outI.data(), 0);
            lddecode::projectExpressibleChromaEnvelope(
                scratch_preQ.data(), nullptr, width, outQ.data(), 1);
            for (int i = 0; i < width; ++i) {
                Irow[left + i] = outI[i];
                Qrow[left + i] = outQ[i];
            }
            continue;
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

void Comb::FrameBuffer::buildAnchorCeiling()
{
    anchorCeiling_flat.clear();
    anchorFloor_flat.clear();
    anchorCoveredLine.clear();
    anchorCeilingValid = true;
    if (exactCarrier_flat.empty()) return;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int fullWidth = videoParameters.fieldWidth;
    if (right - left <= 8 || firstLine >= lastLine || fullWidth <= 0) return;
    constexpr int    kLatRadius     = 8;    // lateral pooling half-window
    constexpr double kMarginRel     = 0.10; // relative pad
    constexpr double kMarginAbsIRE  = 1.5;  // absolute pad
    constexpr double kFloorIRE      = 2.5;  // authority floor (noise env)
    constexpr int    kFloorErode    = 1;
    constexpr double kFloorAuthIRE  = 2.5;

    const double inf = std::numeric_limits<double>::infinity();
    // Pooled per-line envelope rows, +inf where no authority.
    std::vector<std::vector<double>> pooled(lastLine), pooledLo(lastLine);
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
        auto &lo = pooledLo[line];
        lo.assign(fullWidth, 0.0);
        for (int h = left; h < right; ++h) {
            double m = inf; bool ok = true;
            const int a = std::max(left, h - kFloorErode);
            const int b = std::min(right - 1, h + kFloorErode);
            for (int j = a; j <= b; ++j) {
                if (!std::isfinite(env[j])) { ok = false; break; }
                m = std::min(m, env[j]);
            }
            if (!ok) continue;
            const double f = m * (1.0 - kMarginRel) - kMarginAbsIRE;
            lo[h] = (f >= kFloorAuthIRE) ? f : 0.0;
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
                double c1 = src[1] ? (*src[1])[h] : inf;
                double c2 = src[2] ? (*src[2])[h] : inf;
                if (!std::isfinite(c1) || !std::isfinite(c2)) { c = inf; }
                else c = std::max(c1, c2);
            }
            out[h] = (float)c;
        }
    }

    anchorFloor_flat.assign((size_t)lastLine * fullWidth, 0.0f);
    for (int line = firstLine; line < lastLine; ++line) {
        float *out = anchorFloor_flat.data() + (size_t)line * fullWidth;
        const std::vector<double> *src[3] = {nullptr, nullptr, nullptr};
        if (!pooledLo[line].empty()) src[0] = &pooledLo[line];
        else {
            if (line - 1 >= firstLine && !pooledLo[line - 1].empty())
                src[1] = &pooledLo[line - 1];
            if (line + 1 < lastLine && !pooledLo[line + 1].empty())
                src[2] = &pooledLo[line + 1];
            if (!src[1] && line - 2 >= firstLine &&
                !pooledLo[line - 2].empty())
                src[1] = &pooledLo[line - 2];
            if (!src[2] && line + 2 < lastLine &&
                !pooledLo[line + 2].empty())
                src[2] = &pooledLo[line + 2];
        }
        if (!src[0] && !src[1] && !src[2]) continue;
        for (int h = left; h < right; ++h) {
            double f = 0.0;
            if (src[0]) {
                f = (*src[0])[h];
            } else if (src[1] && src[2]) {
                const double f1 = (*src[1])[h];
                const double f2 = (*src[2])[h];
                f = (f1 > 0.0 && f2 > 0.0) ? std::min(f1, f2) : 0.0;
            }
            out[h] = (float)f;
        }
    }
}

static constexpr int    kStarRegLines        = 32;
static constexpr int    kStarRegCols         = 128;
static constexpr double kStarPeakMinIRE      = 10.0;
static constexpr double kStarFlankAgreeIRE   = 6.0;
static constexpr double kStarBlackCeilIRE    = 7.5;
static constexpr double kStarCarrierRunMinIRE = 2.0;
static constexpr double kStarCarrierRunCosMin = 0.80;
static constexpr double kStarLicenseRegionIRE = 1.5;
static constexpr double kStarLicenseFrameIRE  = 1.2;
static constexpr int    kStarLicenseRegionMinN = 3;
static constexpr int    kStarLicenseFrameMinN  = 20;

int Comb::FrameBuffer::certifiedOneDLevel()
{
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

void Comb::FrameBuffer::probeCoveredTruth() const
{
    static const bool on = []{
        const char *e = std::getenv("LDCD_PROBE_COVTRUTH");
        return e && std::atoi(e) != 0;
    }();
    if (!on || !frameHasExactCoverage()) return;
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;
    if (width <= 16) return;

    long   n[2] = {0, 0};
    double sDiscVar[2] = {0, 0};
    long   nDV[2] = {0, 0};
    double sCombErr[2] = {0, 0}, sCombDot[2] = {0, 0};
    double sCombEE[2] = {0, 0}, sCombLL[2] = {0, 0};
    long   nCE[2] = {0, 0};
    double sLuma[2] = {0, 0}, sCarr[2] = {0, 0};
    double sVCohC[2] = {0, 0}, sVCohL[2] = {0, 0};
    long   nVC[2] = {0, 0};
    double sRatio[2] = {0, 0};

    std::vector<double> Lc(width), Lu(width), Ld(width);
    auto lineLuma = [&](int line, std::vector<double> &out) -> bool {
        if (line < firstLine || line >= lastLine) return false;
        if (!certifiedDefLine(line)) return false;
        const float *ex = exactCarrierRow(line);
        if (!ex) return false;
        const quint16 *raw = rawbuffer.data() +
            static_cast<size_t>(line) * videoParameters.fieldWidth;
        for (int xi = 0; xi < width; ++xi) {
            const double e = (double)ex[left + xi];
            if (!std::isfinite(e)) return false;
            out[xi] = (double)raw[left + xi] - e;
        }
        return true;
    };

    for (int line = firstLine + 2; line < lastLine - 2; ++line) {
        if (!lineLuma(line, Lc)) continue;
        const bool haveUp = lineLuma(line - 2, Lu);
        const bool haveDn = lineLuma(line + 2, Ld);
        const float *ex  = exactCarrierRow(line);
        const float *exU = haveUp ? exactCarrierRow(line - 2) : nullptr;
        const float *exD = haveDn ? exactCarrierRow(line + 2) : nullptr;

        for (int xi = 4; xi < width - 4; ++xi) {
            const int h = left + xi;
            // Geometry from certified luma only.
            const double gx = std::fabs(Lc[xi] - Lc[xi - 2]);
            double gy = 0.0; int ng = 0;
            if (haveUp) { gy += std::fabs(Lc[xi] - Lu[xi]); ++ng; }
            if (haveDn) { gy += std::fabs(Lc[xi] - Ld[xi]); ++ng; }
            if (ng) gy /= ng;
            const bool vert = (gx * invIreScale > 8.0) &&
                              (ng == 2) &&
                              (gy < 0.35 * gx);
            const int b = vert ? 1 : 0;

            // Luma's fSC-band content: the +-2 bandpass on Ltrue.
            const double lumaBand = std::fabs(
                0.5 * Lc[xi] - 0.25 * (Lc[xi - 2] + Lc[xi + 2]))
                * invIreScale;
            // True carrier envelope.
            const double e0 = (double)ex[h], e1 = (double)ex[h + 1];
            const double carrEnv = std::hypot(e0, e1) * invIreScale;

            ++n[b];
            sLuma[b] += lumaBand;
            sCarr[b] += carrEnv;
            sRatio[b] += (lumaBand > 1e-6)
                ? std::min(10.0, carrEnv / lumaBand) : 10.0;

            {
                const double *ccRow = lockedCarrierComposite_line(line);
                if (ccRow) {
                    const double ccErr = (ccRow[h] - e0) * invIreScale;
                    const double lbpS =
                        (0.5 * Lc[xi] - 0.25 * (Lc[xi - 2] + Lc[xi + 2]))
                        * invIreScale;
                    sCombErr[b] += std::fabs(ccErr);
                    sCombDot[b] += ccErr * lbpS;
                    sCombEE[b]  += ccErr * ccErr;
                    sCombLL[b]  += lbpS * lbpS;
                    ++nCE[b];
                }
            }
            if (haveUp && haveDn) {
                const double lu2 = 0.5 * Lu[xi] -
                    0.25 * (Lu[xi - 2] + Lu[xi + 2]);
                const double ld2 = 0.5 * Ld[xi] -
                    0.25 * (Ld[xi - 2] + Ld[xi + 2]);
                const double lc2 = 0.5 * Lc[xi] -
                    0.25 * (Lc[xi - 2] + Lc[xi + 2]);
                sDiscVar[b] += 0.5 *
                    std::fabs(lc2 - 0.5 * (lu2 + ld2)) * invIreScale;
                ++nDV[b];
            }

            // Vertical coherence across the certified +-2 bracket.
            if (exU && exD) {
                const double u0 = (double)exU[h], d0 = (double)exD[h];
                if (std::isfinite(u0) && std::isfinite(d0)) {
                    const double mid = 0.5 * (u0 + d0);
                    const double den = std::fabs(e0) + std::fabs(mid);
                    if (den * invIreScale > 0.5) {
                        sVCohC[b] += (e0 * mid >= 0.0 ? 1.0 : -1.0) *
                            std::min(std::fabs(e0), std::fabs(mid)) /
                            std::max(std::fabs(e0), std::fabs(mid));
                        // Luma fSC-band, same test.
                        const double lu = 0.5 * Lu[xi] -
                            0.25 * (Lu[xi - 2] + Lu[xi + 2]);
                        const double ld = 0.5 * Ld[xi] -
                            0.25 * (Ld[xi - 2] + Ld[xi + 2]);
                        const double lc = 0.5 * Lc[xi] -
                            0.25 * (Lc[xi - 2] + Lc[xi + 2]);
                        const double lmid = 0.5 * (lu + ld);
                        const double lden = std::fabs(lc) + std::fabs(lmid);
                        if (lden * invIreScale > 0.5) {
                            sVCohL[b] += (lc * lmid >= 0.0 ? 1.0 : -1.0) *
                                std::min(std::fabs(lc), std::fabs(lmid)) /
                                std::max(std::fabs(lc), std::fabs(lmid));
                            ++nVC[b];
                        }
                    }
                }
            }
        }
    }
    for (int b = 0; b < 2; ++b) {
        if (!n[b]) continue;
        const double inv = 1.0 / (double)n[b];
        const double invV = nVC[b] ? 1.0 / (double)nVC[b] : 0.0;
        qInfo().noquote() << QString::asprintf(
            "COVTRUTH %s n=%7ld  luma-in-band %6.3f IRE  carrier %6.3f IRE  "
            "carr/luma %5.2f  vCoh carrier %+5.3f  vCoh lumaBand %+5.3f  "
            "(nVC %ld)",
            b ? "VERT " : "other", n[b], sLuma[b] * inv, sCarr[b] * inv,
            sRatio[b] * inv, sVCohC[b] * invV, sVCohL[b] * invV, nVC[b]);
        if (nCE[b]) {
            const double invC = 1.0 / (double)nCE[b];
            const double den = std::sqrt(sCombEE[b] * sCombLL[b]);
            qInfo().noquote() << QString::asprintf(
                "COMBTRIAL %s combErr %6.3f IRE  r(combErr, lumaLeak) %+5.3f"
                "  discOwnVar %6.3f IRE (n %ld)",
                b ? "VERT " : "other", sCombErr[b] * invC,
                den > 1e-12 ? sCombDot[b] / den : 0.0,
                nDV[b] ? sDiscVar[b] / (double)nDV[b] : 0.0, nDV[b]);
        }
    }
}
void Comb::FrameBuffer::probeCompactSpans() const
{
    static const int level = []{
        const char *e = std::getenv("LDCD_PROBE_SPAN");
        return e ? std::atoi(e) : 0;
    }();
    if (level < 1) return;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;
    if (width <= 16) return;

    constexpr double kStrongIRE = 6.0;
    constexpr int    kSpanMax   = 3;    // "below 4 pixels in length"

    long spans = 0, spansCert = 0;

    constexpr int kNB = 4;
    auto bucketOf = [](int L) {
        return L <= 3 ? 0 : L <= 7 ? 1 : L <= 15 ? 2 : 3;
    };
    long   covRuns[kNB] = {0}, covPx[kNB] = {0};
    long   covHit[7][kNB] = {{0}}, covFrag[7][kNB] = {{0}};
    double covRatio[7][kNB] = {{0}};

    std::vector<double> env(width), parSpread, envPar;
    std::vector<std::complex<double>> Zc, Zt;
    for (int line = firstLine; line < lastLine; ++line) {
        const double *bpn   = locked1DRawBandpass_line(line);
        const double *wLaw2 = bandWLaw_line(line);
        const double *keep2 = bandKeep_line(line, 2);
        const double *keep1 = bandKeep_line(line, 1);
        const float  *parI  = parallaxI_line(line);
        const float  *parQ  = parallaxQ_line(line);
        if (!bpn || !wLaw2 || !keep2 || !keep1 || !parI || !parQ)
            continue;
        // Signed-IQ demod of the band content, this line's own grammar.
        auto cur = lddecode::carrierGrammarSignedSampleCursor(
            configuration.phaseCompensation ? carrierGrammarLine(line)
                                            : nullptr, left);
        Zc.assign(width, {0.0, 0.0});
        for (int xi = 0; xi < width; ++xi)
            Zc[xi] = lddecode::carrierGrammarDemodSignedCompositeTo4fsc(
                cur, bpn[xi]);
        for (int xi = 0; xi < width; ++xi)
            env[xi] = std::hypot(bpn[xi],
                                 bpn[std::min(xi + 1, width - 1)]) *
                      invIreScale;

        const double *ap = lockedApertureMean_line(line);
        parSpread.assign(width, -1.0);
        envPar.assign(width, 0.0);
        for (int x = 0; x < width; ++x) {
            envPar[x] = std::hypot((double)parI[x], (double)parQ[x]) *
                        invIreScale;
            if (ap && demodWidth == width) {
                double lo = 1e18, hi = -1e18; int nv = 0;
                for (int v = std::max(0, x - 3);
                     v <= std::min(x, width - 4); ++v) {
                    lo = std::min(lo, ap[v]);
                    hi = std::max(hi, ap[v]); ++nv;
                }
                if (nv >= 2) parSpread[x] = (hi - lo) * invIreScale;
            }
        }

        const bool cert = certifiedDefLine(line);
        const float *ex = cert ? exactCarrierRow(line) : nullptr;

        if (ex) {
            auto bpTAt = [&](int h) -> double {
                const int hm = std::max(left, h - 2);
                const int hp = std::min(right - 1, h + 2);
                const double a = ex[h], m = ex[hm], p = ex[hp];
                if (!std::isfinite(a) || !std::isfinite(m) ||
                    !std::isfinite(p))
                    return (double)NAN;
                return 0.5 * a - 0.25 * (m + p);
            };
            auto curT = lddecode::carrierGrammarSignedSampleCursor(
                configuration.phaseCompensation ? carrierGrammarLine(line)
                                                : nullptr, left);
            Zt.assign(width, {(double)NAN, (double)NAN});
            for (int xi = 0; xi < width; ++xi) {
                const double v = bpTAt(left + xi);
                const auto z =
                    lddecode::carrierGrammarDemodSignedCompositeTo4fsc(
                        curT, std::isfinite(v) ? v : 0.0);
                if (std::isfinite(v)) Zt[xi] = z;
            }
        }

        if (level >= 3 && ex) {
            const float *fit = carrierFit_line(line);
            auto notch1At = [&](int xi, int x1) {
                return std::hypot(bpn[xi] * wLaw2[xi] * keep1[xi],
                                  bpn[x1] * wLaw2[x1] * keep1[x1]) *
                       invIreScale;
            };
            auto claimAt = [&](int s, int xi) -> double {
                const int x1 = std::min(xi + 1, width - 1);
                switch (s) {
                case 0:   // fit
                    if (!fit) return 0.0;
                    return std::hypot((double)fit[xi], (double)fit[x1]) *
                           invIreScale;
                case 1:   // notch +/-2, shipping product
                    return std::hypot(bpn[xi] * wLaw2[xi] * keep2[xi],
                                      bpn[x1] * wLaw2[x1] * keep2[x1]) *
                           invIreScale;
                case 3:   // notch +/-1, shipping product
                    return notch1At(xi, x1);
                case 4:   // parallax-unwound band estimate
                    return envPar[xi];
                case 5:   // agreement of the independent witnesses
                    return std::min(envPar[xi], notch1At(xi, x1));
                case 6:   // shape law alone: band x wLaw, no testimony
                    return std::hypot(bpn[xi] * wLaw2[xi],
                                      bpn[x1] * wLaw2[x1]) * invIreScale;
                default:  // raw band envelope (the 1D/locator space)
                    return env[xi];
                }
            };
            int ts = -1;
            for (int xi = 0; xi <= width; ++xi) {
                double envT = 0.0;
                if (xi < width) {
                    const float ea = ex[left + xi];
                    const float eb = ex[left + std::min(xi + 1, width - 1)];
                    if (std::isfinite(ea) && std::isfinite(eb))
                        envT = std::hypot((double)ea, (double)eb) *
                               invIreScale;
                }
                const bool in = (xi < width) && envT >= kStrongIRE;
                if (in) { if (ts < 0) ts = xi; continue; }
                if (ts < 0) continue;
                const int trs = ts, tre = xi;
                ts = -1;
                const int Lt = tre - trs;
                if (Lt < 2 || Lt > 31) continue;
                const int b = bucketOf(Lt);
                ++covRuns[b]; covPx[b] += Lt;
                for (int s = 0; s < 7; ++s) {
                    bool prevIn = false;
                    for (int k = trs; k < tre; ++k) {
                        const double c = claimAt(s, k);
                        const float ea = ex[left + k];
                        const float eb =
                            ex[left + std::min(k + 1, width - 1)];
                        const double t =
                            std::hypot((double)ea, (double)eb) *
                            invIreScale;
                        if (t > 1e-9) covRatio[s][b] += c / t;
                        const bool on = c >= kStrongIRE;
                        if (on) {
                            ++covHit[s][b];
                            if (!prevIn) ++covFrag[s][b];
                        }
                        prevIn = on;
                    }
                }
            }
        }

        int runStart = -1;
        for (int xi = 0; xi <= width; ++xi) {
            const bool in = (xi < width) && env[xi] >= kStrongIRE;
            if (in) { if (runStart < 0) runStart = xi; continue; }
            if (runStart < 0) continue;
            const int rs = runStart, re = xi;
            runStart = -1;
            const int L = re - rs;
            if (L > kSpanMax) continue;
            ++spans;
            if (!ex) continue;

            if (rs < 3 || re + 3 > width) continue;

            double te = 0.0, teMax = 0.0; int n = 0; bool finite = true;
            for (int k = rs; k < re; ++k) {
                const int h = left + k;
                const float ea = ex[h];
                const float eb = ex[std::min(h + 1, right - 1)];
                if (!std::isfinite(ea) || !std::isfinite(eb)) {
                    finite = false; break;
                }
                const double t =
                    std::hypot((double)ea, (double)eb) * invIreScale;
                te += t; teMax = std::max(teMax, t); ++n;
            }
            if (!finite || n == 0) continue;
            ++spansCert;

            double parRatio = -1.0;
            {
                double s = 0.0; int m = 0;
                for (int k = rs; k < re; ++k)
                    if (parSpread[k] >= 0.0 && env[k] > 1e-9) {
                        s += parSpread[k] / env[k]; ++m;
                    }
                if (m) parRatio = s / m;
            }

            if (level >= 2) {
                char row[512];
                int p = std::snprintf(row, sizeof row,
                    "SPANW %.3f %.3f %d %.3f", te / n, teMax, L, parRatio);
                for (int k = rs - 3;
                     k < re + 3 && p < (int)sizeof row - 48; ++k)
                    p += std::snprintf(row + p, sizeof row - p,
                        " %.2f %.2f %.2f %.2f",
                        Zc[k].real() * invIreScale,
                        Zc[k].imag() * invIreScale,
                        Zt[k].real() * invIreScale,
                        Zt[k].imag() * invIreScale);
                std::fprintf(stderr, "%s\n", row);
            }
        }
    }
    if (spansCert)
        qInfo().noquote() << QString::asprintf(
            "SPANCENSUS seq %d  spans<4 %ld  certified-graded %ld",
            (int)heldSeq1, spans, spansCert);
    if (level >= 3) {
        for (int b = 0; b < kNB; ++b) {
            if (!covRuns[b]) continue;
            std::fprintf(stderr,
                "SPANCOV %d %d %ld %ld  %ld %ld %ld %ld %ld %ld %ld"
                "  %ld %ld %ld %ld %ld %ld %ld"
                "  %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                (int)heldSeq1, b, covRuns[b], covPx[b],
                covHit[0][b], covHit[1][b], covHit[2][b], covHit[3][b],
                covHit[4][b], covHit[5][b], covHit[6][b],
                covFrag[0][b], covFrag[1][b], covFrag[2][b], covFrag[3][b],
                covFrag[4][b], covFrag[5][b], covFrag[6][b],
                covRatio[0][b] / covPx[b], covRatio[1][b] / covPx[b],
                covRatio[2][b] / covPx[b], covRatio[3][b] / covPx[b],
                covRatio[4][b] / covPx[b], covRatio[5][b] / covPx[b],
                covRatio[6][b] / covPx[b]);
        }
    }
}

void Comb::FrameBuffer::probeCompactSites() const
{
    static const bool on = []{
        const char *e = std::getenv("LDCD_PROBE_COMPACT");
        return e && std::atoi(e) != 0;
    }();
    if (!on || !frameHasExactCoverage()) return;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int wlim      = right - left;
    if (wlim <= 16) return;

        constexpr double kStrongIRE = 6.0;
        constexpr int    kCompactMax = 8;
        constexpr int    kCompactMin = 2;
        long   nC = 0;
        double sF = 0.0, sN = 0.0, sB = 0.0;
        double sFF = 0.0, sNN = 0.0, sFN = 0.0, mF = 0.0, mN = 0.0;
        std::vector<double> envc;
        for (int line = firstLine; line < lastLine; ++line) {
            if (!certifiedDefLine(line)) continue;
            const float  *ex  = exactCarrierRow(line);
            const float  *fit = carrierFit_line(line);
            const double *bp  = locked1DRawBandpass_line(line);
            if (!ex || !bp || !fit) continue;
            envc.assign(wlim, 0.0);
            for (int xi = 0; xi < wlim; ++xi) {
                const double a = (double)ex[left + xi];
                const double b = (double)ex[left + std::min(xi + 1, wlim - 1)];
                envc[xi] = (std::isfinite(a) && std::isfinite(b))
                    ? std::hypot(a, b) * invIreScale : 0.0;
            }
            int xi = 0;
            while (xi < wlim) {
                if (envc[xi] < kStrongIRE) { ++xi; continue; }
                int j = xi;
                while (j < wlim && envc[j] >= kStrongIRE) ++j;
                const int len = j - xi;
                if (len >= kCompactMin && len <= kCompactMax) {
                    for (int k = xi; k < j; ++k) {
                        const double e = (double)ex[left + k];
                        if (!std::isfinite(e)) continue;
                        const double eF = ((double)fit[k] - e) * invIreScale;
                        const double eN = (bp[k] - e) * invIreScale;
                        const double eB = 0.5 * (eF + eN);
                        ++nC;
                        sF += std::fabs(eF); sN += std::fabs(eN);
                        sB += std::fabs(eB);
                        mF += eF; mN += eN;
                        sFF += eF * eF; sNN += eN * eN; sFN += eF * eN;
                    }
                }
                xi = j;
            }
        }
        if (nC > 100) {
            const double inv = 1.0 / (double)nC;
            const double cf = mF * inv, cn = mN * inv;
            const double vF = sFF * inv - cf * cf;
            const double vN = sNN * inv - cn * cn;
            const double cv = sFN * inv - cf * cn;
            const double r = (vF > 0 && vN > 0)
                ? cv / std::sqrt(vF * vN) : 0.0;
            qInfo().noquote() << QString::asprintf(
                "NOTCHCOMPACT n=%ld  |fit err| %.3f  |notch err| %.3f  "
                "|50/50| %.3f  r(fit,notch) %+0.3f", nC, sF * inv, sN * inv,
                sB * inv, r);
        }
}

void Comb::FrameBuffer::probeCarrierBandwidth() const
{
    static const bool on = []{
        const char *e = std::getenv("LDCD_PROBE_CARRIERBW");
        return e && std::atoi(e) != 0;
    }();
    if (!on) return;
    static const double kStepMinIRE = []{
        const char *e = std::getenv("LDCD_PROBE_CARRIERBW_STEP");
        return e ? std::atof(e) : 25.0;
    }();
    static const double kFlatIRE = []{
        const char *e = std::getenv("LDCD_PROBE_CARRIERBW_FLAT");
        return e ? std::atof(e) : 3.0;
    }();
    static const double kContrast = []{
        const char *e = std::getenv("LDCD_PROBE_CARRIERBW_CONTRAST");
        return e ? std::atof(e) : 4.0;
    }();

    constexpr int W    = 8;          // window half-width, lane samples
    constexpr int M    = 2 * W;      // first-difference length
    constexpr int NB   = M / 2 + 1;  // reported bins (DC .. lane Nyquist)
    constexpr int SPAN = M + 1;      // lane samples per window
    constexpr int FL   = 6;          // plateau length, lane samples
    constexpr int NANG = 6;          // colour-direction bins over 180 deg

    // Cumulative across the run. Unsynchronised -- run with -t 1.
    static double accP[2][NB] = {};      // per lane, step-normalised power
    static double accBlank[NB] = {};     // blank power, absolute IRE^2
    static double accInv[2] = {};        // sum 1/step^2 over accepted edges
    static long   accN[2] = {};
    static long   accNBlank = 0;
    static double accAngP[NANG][NB] = {};
    static double accAngInv[NANG] = {};
    static long   accAngN[NANG] = {};
    static double accAx[2][NB] = {};
    static double accAxInv[2] = {};
    static long   accAxN[2] = {};
    static long   accTheta[12] = {};     // burst angle histogram, 30 deg bins
    static double accTheta90 = 0.0;      // mean of theta folded to [0,90)
    static long   accTheta90N = 0;
    static long dLines = 0, dWin = 0, dNaN = 0, dFlat = 0, dStep = 0;
    static double dFlatSeen = 0.0, dStepSeen = 0.0;
    static long dFlatN = 0, dStepN = 0;
    static const double kHist[6] = { 5, 10, 15, 20, 25, 35 };
    static long dHist[6] = {};

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    if (right - left <= 4 * SPAN || firstLine >= lastLine) return;

    // 16-point DFT basis, built once.
    static const std::array<std::array<double, M>, NB> cosT = []{
        std::array<std::array<double, M>, NB> t{};
        for (int m = 0; m < NB; ++m)
            for (int k = 0; k < M; ++k)
                t[m][k] = std::cos(2.0 * M_PI * m * k / M);
        return t;
    }();
    static const std::array<std::array<double, M>, NB> sinT = []{
        std::array<std::array<double, M>, NB> t{};
        for (int m = 0; m < NB; ++m)
            for (int k = 0; k < M; ++k)
                t[m][k] = std::sin(2.0 * M_PI * m * k / M);
        return t;
    }();

    std::vector<double> lane[2];
    for (int line = firstLine; line < lastLine; ++line) {
        if (!certifiedDefLine(line)) continue;
        const float *ex = exactCarrierRow(line);
        if (!ex) continue;
        dLines++;

        for (int p = 0; p < 2; ++p) {
            lane[p].clear();
            for (int h = left + (((left & 1) == p) ? 0 : 1); h < right; h += 2)
                lane[p].push_back(std::isfinite(ex[h])
                    ? (double)ex[h] * (((h >> 1) & 1) ? -1.0 : 1.0) * invIreScale
                    : std::numeric_limits<double>::quiet_NaN());
        }
        const int n = (int)std::min(lane[0].size(), lane[1].size());
        if (n < SPAN) continue;

        int bLabel = -1;
        {
            static const double cB[4] = { 1, 0, -1, 0 };
            static const double sB[4] = { 0, 1, 0, -1 };
            const int b0 = std::clamp(videoParameters.colourBurstStart, 0,
                                      videoParameters.fieldWidth);
            const int b1 = std::clamp(videoParameters.colourBurstEnd, 0,
                                      videoParameters.fieldWidth);
            const int cyc = (b1 - b0) / 4;
            if (cyc >= 2) {
                const quint16 *rawLine =
                    rawbuffer.data() + (size_t)line * videoParameters.fieldWidth;
                double bI = 0.0, bQ = 0.0;
                for (int h = b0; h < b0 + 4 * cyc; ++h) {
                    bI += (double)rawLine[h] * cB[h & 3];
                    bQ += (double)rawLine[h] * sB[h & 3];
                }
                if (bI != 0.0 || bQ != 0.0) {
                    double th = std::atan2(bQ, bI) * 180.0 / M_PI;
                    if (th < 0.0) th += 360.0;
                    accTheta[std::clamp((int)(th / 30.0), 0, 11)]++;
                    double f90 = std::fmod(th, 90.0);
                    accTheta90 += f90; accTheta90N++;
                    // Nearer lane A's axis (0 deg) or lane B's (90 deg),
                    // modulo 180 since an axis has no sign.
                    const double f180 = std::fmod(th, 180.0);
                    bLabel = (f180 < 45.0 || f180 >= 135.0) ? 0 : 1;
                }
            }
        }

        for (int k = 0; k + SPAN <= n; ++k) {
            double step[2] = {0.0, 0.0};
            bool ok[2] = {false, false};
            double d[2][M];
            for (int p = 0; p < 2; ++p) {
                const double *v = lane[p].data() + k;
                dWin++;
                bool finite = true;
                for (int j = 0; j < SPAN; ++j)
                    if (!std::isfinite(v[j])) { finite = false; break; }
                if (!finite) { dNaN++; continue; }
                // Plateaus: first FL and last FL lane samples, flat.
                double loL = v[0], hiL = v[0], sL = 0.0;
                double loR = v[SPAN - FL], hiR = v[SPAN - FL], sR = 0.0;
                for (int j = 0; j < FL; ++j) {
                    loL = std::min(loL, v[j]); hiL = std::max(hiL, v[j]);
                    sL += v[j];
                    const double r = v[SPAN - FL + j];
                    loR = std::min(loR, r); hiR = std::max(hiR, r);
                    sR += r;
                }
                const double flatWorst = std::max(hiL - loL, hiR - loR);
                dFlatSeen += flatWorst; dFlatN++;
                const double st = (sR - sL) / FL;
                dStepSeen += std::fabs(st); dStepN++;
                if (flatWorst > kFlatIRE) { dFlat++; continue; }
                for (int b = 0; b < 6; ++b)
                    if (std::fabs(st) >= kHist[b]) dHist[b]++;
                if (std::fabs(st) < kStepMinIRE ||
                    std::fabs(st) < kContrast * flatWorst) { dStep++; continue; }
                for (int j = 0; j < M; ++j) d[p][j] = (v[j + 1] - v[j]) / st;
                step[p] = st; ok[p] = true;
            }
            for (int p = 0; p < 2; ++p) {
                if (!ok[p]) continue;
                // PAR when this lane is the one nearer the burst axis.
                const int ax = (bLabel < 0) ? -1 : ((p == bLabel) ? 0 : 1);
                for (int m = 0; m < NB; ++m) {
                    double re = 0.0, im = 0.0;
                    for (int j = 0; j < M; ++j) {
                        re += d[p][j] * cosT[m][j];
                        im -= d[p][j] * sinT[m][j];
                    }
                    const double pw = re * re + im * im;
                    accP[p][m] += pw;
                    if (ax >= 0) accAx[ax][m] += pw;
                }
                accInv[p] += 1.0 / (step[p] * step[p]);
                accN[p]++;
                if (ax >= 0) {
                    accAxInv[ax] += 1.0 / (step[p] * step[p]);
                    accAxN[ax]++;
                }
            }
            if (ok[0] && ok[1]) {
                double a = std::atan2(step[1], step[0]);
                if (a < 0.0) a += M_PI;                 // direction, not sign
                int ab = (int)(a / M_PI * NANG);
                ab = std::clamp(ab, 0, NANG - 1);
                for (int p = 0; p < 2; ++p) {
                    for (int m = 0; m < NB; ++m) {
                        double re = 0.0, im = 0.0;
                        for (int j = 0; j < M; ++j) {
                            re += d[p][j] * cosT[m][j];
                            im -= d[p][j] * sinT[m][j];
                        }
                        accAngP[ab][m] += re * re + im * im;
                    }
                    accAngInv[ab] += 1.0 / (step[p] * step[p]);
                    accAngN[ab]++;
                }
            }
            if (ok[0] || ok[1]) k += SPAN - 1;   // keep sites independent
        }

        for (int p = 0; p < 2; ++p) {
            for (int k = 0; k + SPAN <= n; k += SPAN) {
                const double *v = lane[p].data() + k;
                bool finite = true;
                double lo = v[0], hi = v[0];
                for (int j = 0; j < SPAN; ++j) {
                    if (!std::isfinite(v[j])) { finite = false; break; }
                    lo = std::min(lo, v[j]); hi = std::max(hi, v[j]);
                }
                if (!finite || hi - lo > kFlatIRE) continue;
                for (int m = 0; m < NB; ++m) {
                    double re = 0.0, im = 0.0;
                    for (int j = 0; j < M; ++j) {
                        const double dv = v[j + 1] - v[j];
                        re += dv * cosT[m][j];
                        im -= dv * sinT[m][j];
                    }
                    accBlank[m] += re * re + im * im;
                }
                accNBlank++;
            }
        }
    }

    std::fprintf(stderr,
        "[CARRIERBW] frame %d  certified lines %ld  windows %ld  ->"
        " NaN %ld  flat %ld  step %ld  accepted %ld/%ld  blanks %ld\n"
        "            mean worst-plateau spread %.1f IRE   mean |step|"
        " %.1f IRE\n",
        (int)heldSeq1, dLines, dWin, dNaN, dFlat, dStep,
        accN[0], accN[1], accNBlank,
        dFlatN ? dFlatSeen / dFlatN : 0.0, dStepN ? dStepSeen / dStepN : 0.0);
    std::fprintf(stderr, "            flat-window |step| >=");
    for (int b = 0; b < 6; ++b)
        std::fprintf(stderr, "  %.0f:%ld", kHist[b], dHist[b]);
    std::fprintf(stderr, " IRE\n");
    if (accN[0] + accN[1] == 0) return;
    const double binMHz = 2.0 * 3.579545 / M;   // lane rate is 2fSC
    auto report = [&](const char *tag, const double *P, double inv, long nn) {
        if (nn <= 0) return;
        char buf[512]; int o = 0;
        o += std::snprintf(buf + o, sizeof(buf) - o, "  %-10s", tag);
        for (int m = 0; m < NB - 2; ++m) {
            const double floorM = accNBlank
                ? accBlank[m] / accNBlank * (inv / nn) : 0.0;
            o += std::snprintf(buf + o, sizeof(buf) - o, "%7.3f",
                               P[m] / nn - floorM);
        }
        std::fprintf(stderr, "%s   n=%ld\n", buf, nn);
    };
    std::fprintf(stderr,
        "[CARRIERBW] frame %d  step>=%.0f IRE  flat<=%.1f IRE"
        "  blanks=%ld\n            bin MHz ", (int)heldSeq1,
        kStepMinIRE, kFlatIRE, accNBlank);
    for (int m = 0; m < NB - 2; ++m)
        std::fprintf(stderr, "%7.2f", m * binMHz);
    std::fprintf(stderr, "\n");
    report("lane A", accP[0], accInv[0], accN[0]);
    report("lane B", accP[1], accInv[1], accN[1]);
    // Burst-relative axes. If these are no more separated than lane A/B,
    // the sequence never swapped the axes between lanes.
    report("burst PAR", accAx[0], accAxInv[0], accAxN[0]);
    report("burst PERP", accAx[1], accAxInv[1], accAxN[1]);
    if (accTheta90N) {
        std::fprintf(stderr,
            "            burst angle in lane basis: mean %.1f deg mod 90"
            "   histogram/30deg:", accTheta90 / accTheta90N);
        for (int b = 0; b < 12; ++b)
            std::fprintf(stderr, " %ld", accTheta[b]);
        std::fprintf(stderr, "\n");
    }
    for (int a = 0; a < NANG; ++a)
        if (accAngN[a] > 0) {
            char t[24];
            std::snprintf(t, sizeof(t), "dir %d-%d", a * 180 / NANG,
                          (a + 1) * 180 / NANG);
            report(t, accAngP[a], accAngInv[a], accAngN[a]);
        }
    std::fprintf(stderr,
        "  ref 2.49MHz 0.982  0.885  0.649  0.337  0.103\n"
        "  ref 1.50MHz 0.944  0.764  0.472  0.196  0.046   <- encoder law\n"
        "  ref 1.16MHz 0.904  0.648  0.325  0.100  0.016\n"
        "  ref 0.93MHz 0.853  0.513  0.193  0.041  0.004\n"
        "  ref 0.72MHz 0.773  0.338  0.074  0.007  0.000\n"
        "  ref 0.55MHz 0.653  0.157  0.013  0.000  0.000\n"
        "  ref 0.41MHz 0.495  0.042  0.001  0.000  0.000\n"
        "            (rows start at the 0.45 MHz bin; DC is the self-check)\n");
}

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

void Comb::FrameBuffer::buildNotchHfCurves(
    int line, std::vector<double> &bp,
    std::vector<double> &wLaw, std::vector<double> &keep,
    std::vector<quint8> *heard, int reach) const
{
    // Confirmation ramp: removal begins at rho = -confirmLo and is full
    // at rho = -(confirmLo + confirmW). Sweepable for the referee.
    static const double confirmLo = []{
        const char *e = std::getenv("LDCD_NOTCHHF_CONFIRM_LO");
        return e ? std::atof(e) : 0.15;
    }();
    static const double confirmW = []{
        const char *e = std::getenv("LDCD_NOTCHHF_CONFIRM_W");
        return e ? std::atof(e) : 0.45;
    }();
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int w     = right - left;
    const int fullW = videoParameters.fieldWidth;
    bp.assign(w, 0.0);
    wLaw.assign(w, 1.0);
    keep.assign(w, 0.0);
    if (heard) heard->assign(w, 0);
    if (w <= 8) return;

    const quint16 *rawC = rawbuffer.data() + size_t(line) * fullW;
    const quint16 *rawU = (line - reach >= 0)
        ? rawbuffer.data() + size_t(line - reach) * fullW : nullptr;
    const quint16 *rawD = (line + reach < frameHeight)
        ? rawbuffer.data() + size_t(line + reach) * fullW : nullptr;
    const CombCarrierGrammar *gC = carrierGrammarLine(line);
    const CombCarrierGrammar *gU = (line - reach >= 0)
        ? carrierGrammarLine(line - reach) : nullptr;
    const CombCarrierGrammar *gD = (line + reach < frameHeight)
        ? carrierGrammarLine(line + reach) : nullptr;
    auto bpOf = [](const quint16 *row, int h) {
        return 0.5 * (double)row[h] -
               0.25 * ((double)row[h - 2] + (double)row[h + 2]);
    };

    const double *bpC  = locked1DRawBandpass_line(line);
    const double *bpCU = (line - reach >= 0)
        ? locked1DRawBandpass_line(line - reach) : nullptr;
    const double *bpCD = (line + reach < frameHeight)
        ? locked1DRawBandpass_line(line + reach) : nullptr;

    std::vector<double> bpU(w, 0.0), bpD(w, 0.0), envObs(w), envLaw(w);
    for (int xi = 0; xi < w; ++xi) {
        const int h = left + xi;
        bp[xi] = bpC ? bpC[xi] : bpOf(rawC, h);
        if (rawU) bpU[xi] = bpCU ? bpCU[xi] : bpOf(rawU, h);
        if (rawD) bpD[xi] = bpCD ? bpCD[xi] : bpOf(rawD, h);
    }
    for (int xi = 0; xi < w; ++xi)
        envObs[xi] = std::hypot(bp[xi], bp[std::min(xi + 1, w - 1)]);
    lddecode::projectExpressibleChromaEnvelope(
        envObs.data(), nullptr, w, envLaw.data());
    static const bool envUnbiased = []{
        const char *s = std::getenv("LDCD_ENV_UNBIASED");
        return s && std::atoi(s) != 0;
    }();
    for (int xi = 0; xi < w; ++xi)
        if (envObs[xi] > 1e-9)
            wLaw[xi] = envUnbiased
                ? std::min(4.0, envLaw[xi] / envObs[xi])
                : std::min(1.0, envLaw[xi] / envObs[xi]);

    const double noiseFloor = 1.0 * irescale;
    if (!rawU && !rawD) return;
    for (int xi = 3; xi < w - 3; ++xi) {
        double sCC = 0, sCU = 0, sUU = 0, sCD = 0, sDD = 0;
        for (int k = -3; k <= 3; ++k) {
            const double c = bp[xi + k];
            const double u = bpU[xi + k];
            const double d = bpD[xi + k];
            sCC += c * c;
            sCU += c * u; sUU += u * u;
            sCD += c * d; sDD += d * d;
        }
        if (sCC < noiseFloor * noiseFloor * 7) {
            if (heard) (*heard)[xi] = 1;
            continue;
        }
        const int hCol = left + xi;
        auto signOf = [&](const CombCarrierGrammar *gP) -> double {
            if (reach == 2) return -1.0;
            if (!gC || !gP) return 0.0;
            switch (lddecode::carrierGrammarSignedPhaseRelation(
                        gC, hCol, gP, hCol)) {
                case lddecode::CarrierPhaseRelation::Opposite: return -1.0;
                case lddecode::CarrierPhaseRelation::Same:     return  1.0;
                default: return 0.0;
            }
        };
        const double relU = signOf(gU), relD = signOf(gD);
        double rhoU = 0.0, rhoD = 0.0;
        bool haveU = false, haveD = false;
        if (rawU && relU != 0.0 && sUU >= noiseFloor * noiseFloor * 7) {
            rhoU = -relU * sCU / std::sqrt(sCC * sUU);
            haveU = true;
        }
        if (rawD && relD != 0.0 && sDD >= noiseFloor * noiseFloor * 7) {
            rhoD = -relD * sCD / std::sqrt(sCC * sDD);
            haveD = true;
        }
        if (!haveU && !haveD) continue;
        const double sPP2 = sUU + sDD;
        const double sCP2 = -(relU * sCU) - (relD * sCD);
        const double rhoP = (sPP2 > 1e-12)
            ? sCP2 / std::sqrt(sCC * sPP2) : 0.0;
        double kp = std::clamp((-rhoP - confirmLo) / confirmW, 0.0, 1.0);
        constexpr double kDecisive = -0.5;
        if ((haveU && rhoU <= kDecisive) || (haveD && rhoD <= kDecisive))
            kp = 1.0;
        keep[xi] = kp;
        if (heard) (*heard)[xi] = 1;
    }
}

void Comb::FrameBuffer::buildBandFacts()
{
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int w         = right - left;
    if (w <= 8 || demodWidth != w || bandWLaw_flat.empty()) return;

    static const int cB4[4] = { 1, 0, -1, 0 };
    static const int sB4[4] = { 0, 1, 0, -1 };
    std::vector<double> bp, wLaw, keep;
    std::vector<quint8> heard;
    for (int line = firstLine; line < lastLine; ++line) {
        double *wRow  = bandWLaw_flat.data()  + size_t(line) * demodWidth;
        double *k1Row = bandKeep1_flat.data() + size_t(line) * demodWidth;
        double *k2Row = bandKeep2_flat.data() + size_t(line) * demodWidth;
        quint8 *h1Row = bandHeard1_flat.data() + size_t(line) * demodWidth;
        quint8 *h2Row = bandHeard2_flat.data() + size_t(line) * demodWidth;
        float  *pIRow = parallaxI_flat.data() + size_t(line) * demodWidth;
        float  *pQRow = parallaxQ_flat.data() + size_t(line) * demodWidth;

        buildNotchHfCurves(line, bp, wLaw, keep, &heard, 2);
        if ((int)bp.size() != w) continue;
        for (int xi = 0; xi < w; ++xi) {
            wRow[xi]  = wLaw[xi];
            k2Row[xi] = keep[xi];
            h2Row[xi] = heard[xi];
        }
        buildNotchHfCurves(line, bp, wLaw, keep, &heard, 1);
        for (int xi = 0; xi < w; ++xi) {
            k1Row[xi] = keep[xi];
            h1Row[xi] = heard[xi];
        }
        const quint16 *raw = rawbuffer.data() +
            size_t(line) * videoParameters.fieldWidth;
        const double *ap = lockedApertureMean_line(line);
        for (int x = 0; x < w; ++x) {
            double sI = 0.0, sQ = 0.0, sWt = 0.0;
            if (ap) {
                for (int v = std::max(0, x - 3);
                     v <= std::min(x, w - 4); ++v) {
                    const double r0 = (double)raw[left + v]     - ap[v];
                    const double r1 = (double)raw[left + v + 1] - ap[v];
                    const double r2 = (double)raw[left + v + 2] - ap[v];
                    const double r3 = (double)raw[left + v + 3] - ap[v];
                    const double a = 0.5 * (r0 - r2);
                    const double b = 0.5 * (r1 - r3);
                    const double s = 0.5 * (r0 + r2);
                    const int c0 = carrierSampleClass(line, left + v);
                    const int c1 = (c0 + 1) & 3;
                    const double vI =  sB4[c1] * a - sB4[c0] * b;
                    const double vQ = -cB4[c1] * a + cB4[c0] * b;
                    const double wv =
                        1.0 / (std::fabs(s) * invIreScale + 1.0);
                    sI += wv * vI; sQ += wv * vQ; sWt += wv;
                }
            }
            pIRow[x] = (sWt > 1e-12) ? (float)(sI / sWt) : 0.0f;
            pQRow[x] = (sWt > 1e-12) ? (float)(sQ / sWt) : 0.0f;
        }
    }
}

static bool ldcdIceVectorMode()
{
    static const bool v = []{
        const char *e = std::getenv("LDCD_ICE_LEGACY");
        return !(e && std::atoi(e) != 0);
    }();
    return v;
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
    static const bool oneDThroughProduceY = []{
        const char *e = std::getenv("LDCD_1D_PRODUCEY");
        return e && std::atoi(e) != 0;
    }();
    if (configuration.dimensions == 1 && !oneDThroughProduceY) {
        for (int line = firstLine; line < lastLine; ++line) {
            const quint16 *raw = rawbuffer.constData()
                                 + static_cast<size_t>(line) * fullWidth;
            const double *carrier = clpbuffer[0].pixel[line];
            double *Y = componentFrame->y(line);
            if (!carrier || !Y) continue;
            for (int h = left; h < right; ++h)
                Y[h] = static_cast<double>(raw[h]) - carrier[h];
        }
        return;
    }

    buildStarFootprint(prevF, nextF); // idempotent safety for direct callers

    if (!anchorCeilingValid) buildAnchorCeiling();

    // TEMPORARY INSTRUMENT (LDCD_PROBE_CARRIERBW=1), off by default.
    probeCarrierBandwidth();
    probeCompactSites();
    probeCompactSpans();
    probeCoveredTruth();

    static const bool electBypass = []{
        const char *s = std::getenv("LDCD_ELECT_BYPASS");
        return s && std::atoi(s) != 0;
    }();

    const bool notchCandidate = [&]{
        const char *s = std::getenv("LDCD_Y_NOTCH");   // A/B override wins
        if (s) return std::atoi(s) != 0;
        return configuration.yElection.ntc;            // roster (--y-election)
    }();
    const bool notchLive = notchCandidate && !frameHasExactCoverage();
    const bool iceLive = !frameHasExactCoverage() &&
        !icebergReturnWeight_flat.empty();
    static const double notchBlindTauIRE = []{
        const char *s = std::getenv("LDCD_Y_NOTCH_TAU");
        return s ? std::atof(s) : 4.0;
    }();

    static const bool certChromaFactsY = []{
        const char *e = std::getenv("LDCD_CC_FACTS");
        return !(e && std::atoi(e) == 0);
    }();

    std::vector<double> pyRow0(width), pyRow1(width), pyRow3(width),
                        pyRow4(width), pyRow5(width), pyRow6(width),
                        pyRow7(width);
    std::vector<double> nhCarrier(width);
    std::vector<quint8> nhHeard;

    const bool compactStageLive = !frameHasExactCoverage();
    std::vector<double> compactRow(width);
    std::vector<std::uint8_t> compactRegime(width);
    std::vector<double> crashRefRow(width);

    std::vector<double> resRow0(width), resRow1(width), resRow3(width),
                        resRow4(width), resRow5(width), resRow6(width),
                        resRow7(width), spRowV(width), cpRowV(width);
    static const bool probeNhGrade = []{
        const char *e = std::getenv("LDCD_PROBE_NHGRADE");
        return e && std::atoi(e) != 0;
    }();
    static const double nhEdges[7] = { 0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0 };
    double gErrC[8] = {0}, gErrN[8] = {0};
    long   gCnt[8]  = {0}, gWin[8]  = {0};

    static const char *nhDumpPath = std::getenv("LDCD_PROBE_NHDUMP");
    static std::FILE *nhDumpFile =
        nhDumpPath ? std::fopen(nhDumpPath, "wb") : nullptr;

    for (int line = firstLine; line < lastLine; ++line) {
        if (line >= demodLines) continue;

        const quint16 *rawLine = rawbuffer.data() + line * fullWidth;
        double *Y = componentFrame->y(line);
        const double *clpLine = clpbuffer[srcBuf].pixel[line];
        const double *carrierComp = lockedCarrierComposite_line(line);
        static const bool attrView = []{
            const char *e = std::getenv("LDCD_ATTR_VIEW");
            return e && std::atoi(e) != 0;
        }();
        std::vector<double> attrC;
        std::vector<std::uint8_t> attrOn;
        if (attrView && !frameHasExactCoverage() &&
            configuration.phaseCompensation) {
            const CombCarrierGrammar *gr = carrierGrammarLine(line);
            lddecode::CarrierGrammarDemodCoefficients dc;
            if (gr && lddecode::carrierGrammarLockedDemodCoefficients(
                          gr, left, dc)) {
                attrC.assign(width, 0.0);
                attrOn.assign(width, 0);
                // Composite residual against a legal 4-mean and its
                // unsmoothed point envelope.
                std::vector<double> r(width, 0.0), E(width, 0.0),
                    Es(width, 0.0), ti(width, 0.0), tq(width, 0.0);
                for (int x = 0; x < width; ++x) {
                    const int h = left + x;
                    if (h - 1 < 0 || h + 2 >= fullWidth) continue;
                    const double m = 0.25 *
                        ((double)rawLine[h - 1] + (double)rawLine[h] +
                         (double)rawLine[h + 1] + (double)rawLine[h + 2]);
                    r[x] = (double)rawLine[h] - m;
                    lddecode::carrierGrammarLockedDemodCoefficients(
                        gr, h, dc);
                    ti[x] = dc.ti; tq[x] = dc.tq;
                }
                for (int x = 0; x + 1 < width; ++x)
                    E[x] = std::hypot(r[x], r[x + 1]);
                for (int x = 1; x + 1 < width; ++x)
                    Es[x] = 0.25 * (E[x - 1] + 2.0 * E[x] + E[x + 1]);
                // Step response of the encoder kernel, fractional lookup.
                static const std::array<double, 10> stepCum = []{
                    std::array<double, 10> c{}; double t = 0.0;
                    for (int j = 0; j < lddecode::kChromaEnvelopeTaps; ++j)
                        t += lddecode::kChromaEnvelopeFilter[j];
                    double a = 0.0; c[0] = 0.0;
                    for (int j = 0; j < lddecode::kChromaEnvelopeTaps; ++j) {
                        a += lddecode::kChromaEnvelopeFilter[j] / t;
                        c[j + 1] = a;
                    }
                    return c;
                }();
                auto stepAt = [&](double d) -> double {
                    const double u = d + 4.5;   // tap centres at -4..+4
                    if (u <= 0.0) return 0.0;
                    if (u >= 9.0) return 1.0;
                    const int j = (int)u;
                    const double f = u - j;
                    return stepCum[j] + f * (stepCum[std::min(j + 1, 9)] -
                                             stepCum[j]);
                };
                const double stepThr = 5.0 * irescale;
                const double hiThr   = 10.0 * irescale;
                int lastB = -100;
                for (int x = 12; x < width - 13; ++x) {
                    const double g  = Es[x + 2] - Es[x - 2];
                    const double gl = Es[x + 1] - Es[x - 3];
                    const double gr2 = Es[x + 3] - Es[x - 1];
                    if (std::fabs(g) < stepThr) continue;
                    if (std::fabs(g) < std::fabs(gl) ||
                        std::fabs(g) < std::fabs(gr2)) continue;
                    double hi = 0.0;
                    for (int k = -3; k <= 3; ++k)
                        hi = std::max(hi, Es[x + k]);
                    if (hi < hiThr) continue;
                    if (x - lastB < 8) continue;
                    lastB = x;
                    double sIa = 0, sQa = 0, sIb = 0, sQb = 0;
                    double nTa = 0, nQa2 = 0, nTb = 0, nQb2 = 0;
                    for (int k = 3; k <= 10; ++k) {
                        sIa += r[x - k] * ti[x - k];
                        sQa += r[x - k] * tq[x - k];
                        nTa += ti[x - k] * ti[x - k];
                        nQa2 += tq[x - k] * tq[x - k];
                        sIb += r[x + k] * ti[x + k];
                        sQb += r[x + k] * tq[x + k];
                        nTb += ti[x + k] * ti[x + k];
                        nQb2 += tq[x + k] * tq[x + k];
                    }
                    if (nTa < 1e-9 || nQa2 < 1e-9 ||
                        nTb < 1e-9 || nQb2 < 1e-9) continue;
                    const double Ia = sIa / nTa, Qa = sQa / nQa2;
                    const double Ib = sIb / nTb, Qb = sQb / nQb2;
                    // Boundary position: fit the modelled carrier against
                    // the composite residual over the transition.
                    double bestJ = 1e300, bestX0 = (double)x;
                    for (double x0 = x - 2.0; x0 <= x + 2.0; x0 += 0.25) {
                        double J = 0.0;
                        for (int u = -4; u <= 4; ++u) {
                            const int xx = x + u;
                            const double Sv = stepAt(xx - x0);
                            const double Cm =
                                (Ia + (Ib - Ia) * Sv) * ti[xx] +
                                (Qa + (Qb - Qa) * Sv) * tq[xx];
                            const double d = r[xx] - Cm;
                            J += d * d;
                        }
                        if (J < bestJ) { bestJ = J; bestX0 = x0; }
                    }
                    for (int u = -4; u <= 4; ++u) {
                        const int xx = x + u;
                        const double Sv = stepAt(xx - bestX0);
                        attrC[xx] =
                            (Ia + (Ib - Ia) * Sv) * ti[xx] +
                            (Qa + (Qb - Qa) * Sv) * tq[xx];
                        attrOn[xx] = 1;
                    }
                }
            }
        }

        const bool coveredFrame = frameHasExactCoverage();
        const float *retractedRow =
            (carrierRetractedValid && configuration.yElection.rcy)
            ? carrierRetracted_line(line) : nullptr;
        const float *ccMaskRow = lockedCcMask_line(line);
        const float *icebergYRow = icebergRecoveredY_line(line);
        const float *icebergWRow = icebergReturnWeight_line(line);
        const double icebergPolicyWeight = std::clamp(
            configuration.tunables.CC_SUPPRESSION_WEIGHT, 0.0, 1.0);
        const double *returnedFactCarrierRow =
            factBackedCarrier_line(line);
        const bool returnedHasDerivedCarrier =
            returnedFactCarrierRow;
        static const int yViewMode = []{
            const char *s = std::getenv("LDCD_YVIEW");
            if (!s) return -1;
            if (std::strcmp(s, "mono") == 0)      return 100;
            if (std::strcmp(s, "native") == 0)    return 101;
            if (std::strcmp(s, "comb") == 0)      return 0;
            if (std::strcmp(s, "retracted") == 0) return 1;
            if (std::strcmp(s, "oned") == 0)      return 3;
            if (std::strcmp(s, "returned") == 0)  return 4;
            if (std::strcmp(s, "notchhf") == 0)   return 6;
            if (std::strcmp(s, "notch") == 0)     return 5;
            if (std::strcmp(s, "ice") == 0)       return 7;
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
        if (nhDumpFile && carrierComp) {
            const double *dBp = locked1DRawBandpass_line(line);
            const double *dW  = bandWLaw_line(line);
            const double *dK  = bandKeep_line(line, 2);
            const quint8 *dH  = bandHeard_line(line, 2);
            if (dBp && dW && dK && dH) {
                const float nanf = std::numeric_limits<float>::quiet_NaN();
                std::vector<float> rec(3 * (size_t)width);
                for (int xi = 0; xi < width; ++xi) {
                    const double C = carrierComp[xi];
                    rec[xi] = std::isfinite(C)
                        ? (float)(C / irescale) : nanf;
                    rec[width + xi] = dH[xi]
                        ? (float)(dBp[xi] * dW[xi] * dK[xi] / irescale) : nanf;
                    rec[2 * (size_t)width + xi] = certExactRow
                        ? (float)((double)certExactRow[left + xi] / irescale)
                        : nanf;
                }
                const qint32 hdr[3] = { heldSeq1, (qint32)line,
                                        (qint32)width };
                std::fwrite(hdr, sizeof(qint32), 3, nhDumpFile);
                std::fwrite(rec.data(), sizeof(float), rec.size(), nhDumpFile);
            }
        }
        if (probeNhGrade && certExactRow && carrierComp) {
            const double *nhBp = locked1DRawBandpass_line(line);
            const double *nhW  = bandWLaw_line(line);
            const double *nhK  = bandKeep_line(line, 2);
            const quint8 *nhH  = bandHeard_line(line, 2);
            if (nhBp && nhW && nhK && nhH) {
                for (int h = left; h < right; ++h) {
                    const int xi = h - left;
                    if (!nhH[xi]) continue;
                    const double E = (double)certExactRow[h];
                    const double C = carrierComp[xi];
                    if (!std::isfinite(E) || !std::isfinite(C)) continue;
                    const double N = nhBp[xi] * nhW[xi] * nhK[xi];
                    if (!std::isfinite(N)) continue;
                    const double s =
                        (std::fabs(C) - std::fabs(N)) / irescale;
                    int b = 0;
                    for (int e = 0; e < 7; ++e) if (s >= nhEdges[e]) b = e + 1;
                    const double eC = std::fabs(C - E) / irescale;
                    const double eN = std::fabs(N - E) / irescale;
                    gErrC[b] += eC; gErrN[b] += eN; ++gCnt[b];
                    if (eN < eC) ++gWin[b];
                }
            }
        }
        if (certLineComplete) {
            for (int h = left; h < right; ++h)
                Y[h] = (double)rawLine[h] - (double)certExactRow[h];
        } else if (carrierComp) {
            const bool useSharpCoarse =
                configuration.yElection.lsc && !lockedLumaSharp_flat.empty();
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
            {
                static const bool bandCensus = []{
                    const char *e = std::getenv("LDCD_BANDCENSUS");
                    return e && e[0] == '1';
                }();
                if (bandCensus && bandRow) {
                    static std::mutex bcMtx;
                    static long bcN[3] = {0,0,0}, bcB[3] = {0,0,0};
                    static const struct { int y0,y1,x0,x1; } bz[3] = {
                        {140,320,560,745},   // shirt seam (right figure)
                        {264,359,265,390},   // bikini/shadow
                        {144,269,265,400},   // strap
                    };
                    std::lock_guard<std::mutex> lk(bcMtx);
                    for (int z = 0; z < 3; ++z) {
                        if (line < bz[z].y0 || line > bz[z].y1) continue;
                        for (int x = bz[z].x0;
                             x <= bz[z].x1 && x < demodWidth; ++x) {
                            ++bcN[z];
                            if (bandRow[x]) ++bcB[z];
                        }
                    }
                    if (line + 1 == videoParameters.lastActiveFrameLine)
                        std::fprintf(stderr,
                            "[BANDC] f%d shirt %.1f%% bikini %.1f%% strap %.1f%%\n",
                            (int)heldSeq1,
                            bcN[0] ? 100.0*bcB[0]/bcN[0] : -1.0,
                            bcN[1] ? 100.0*bcB[1]/bcN[1] : -1.0,
                            bcN[2] ? 100.0*bcB[2]/bcN[2] : -1.0);
                }
            }
            const float *dsExactRow = exactCarrierRow(line);
            const float *anchorRow = anchorCeilingRow(line);
            const lddecode::CarrierAnalysisRecord *analysisRow =
                carrierAnalysis_line(line);
            const float *alienRow = regionAlienPartner_line(line);
            const AttributionEvidence *attribRow = attributionEvidence_line(line);
            static const int retractedAdmitMode = []{
                const char *s = std::getenv("LD_RETRACTED_ADMIT");
                if (!s) return 0;              // 0 = spatial (default)
                if (s[0] == 't') return 1;     // trust-only hard gate
                if (s[0] == 'a') return 2;     // admit all
                return 0;
            }();
            const bool retractedAdmitSpatial = (retractedAdmitMode == 0);
            const bool retractedAdmitAll = (retractedAdmitMode == 2);
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
                } else if (plane == 7 && icebergYRow && icebergWRow) {
                    const double w = std::clamp(
                        (double)icebergWRow[xx], 0.0, 1.0);
                    const double corr = (double)icebergYRow[xx];
                    if (w > 0.0 && std::isfinite(corr)) {
                        const double c = carrierComp ? carrierComp[xx]
                                                     : clpLine[hh];
                        const double comb = (double)rawLine[hh] -
                            (std::isfinite(c) ? c : 0.0);
                        return comb + w * corr;
                    }
                } else if (plane == 4 && ccMaskRow) {
                    const double c = carrierComp ? carrierComp[xx] : clpLine[hh];
                    const double comb = (double)rawLine[hh] -
                        (std::isfinite(c) ? c : 0.0);
                    const double m = std::clamp((double)ccMaskRow[xx], 0.0, 1.0);
                    if (!coveredFrame && !ldcdIceVectorMode() &&
                        icebergYRow && icebergWRow &&
                        std::isfinite((double)icebergYRow[xx])) {
                        const double iw = icebergPolicyWeight * std::clamp(
                            (double)icebergWRow[xx], 0.0, 1.0);
                        if (iw > 0.0)
                            return comb + iw *
                                ((double)icebergYRow[xx] - comb);
                    }
                    if (returnedHasDerivedCarrier && retractedRow) {
                        const double r = (double)retractedRow[xx];
                        if (std::isfinite(r)) return comb + m * (r - comb);
                    }
                    return comb + m * ((double)rawLine[hh] - comb);
                } else if (plane == 5) {
                    const int hm = std::clamp(hh - 2, left, right - 1);
                    const int hp = std::clamp(hh + 2, left, right - 1);
                    return 0.25 * ((double)rawLine[hm] +
                                   2.0 * (double)rawLine[hh] +
                                   (double)rawLine[hp]);
                } else if (plane == 6) {
                    return (double)rawLine[hh] - nhCarrier[hh - left];
                }
                const double c = carrierComp ? carrierComp[xx] : clpLine[hh];
                return (double)rawLine[hh] - (std::isfinite(c) ? c : 0.0);
            };
            if (notchLive) {
                const double *nhBpRow = locked1DRawBandpass_line(line);
                const double *nhWRow  = bandWLaw_line(line);
                const double *nhKRow  = bandKeep_line(line, 2);
                const quint8 *nhHRow  = bandHeard_line(line, 2);
                if (nhBpRow && nhWRow && nhKRow && nhHRow) {
                    nhHeard.assign(nhHRow, nhHRow + width);
                    for (int xx = 0; xx < width; ++xx)
                        nhCarrier[xx] =
                            nhBpRow[xx] * nhWRow[xx] * nhKRow[xx];
                } else {
                    std::fill(nhCarrier.begin(), nhCarrier.end(), 0.0);
                    nhHeard.assign(width, 0);
                }
            }

            std::fill(crashRefRow.begin(), crashRefRow.end(),
                      std::numeric_limits<double>::quiet_NaN());
            std::fill(compactRow.begin(), compactRow.end(),
                      std::numeric_limits<double>::quiet_NaN());
            std::fill(compactRegime.begin(), compactRegime.end(), 0);
            if (compactStageLive) {
                static const int cB4c[4] = { 1, 0, -1, 0 };
                static const int sB4c[4] = { 0, 1, 0, -1 };
                constexpr double kCompactIRE  = 2.0;
                constexpr double kRefFloorIRE = 2.0;
                constexpr int    kBurstMax    = 6;
                constexpr int    kBindGapMax  = 2;
                constexpr int    kSkirtPad    = 4;
                const double *cbp = locked1DRawBandpass_line(line);
                const double *cbU = (line - 2 >= 0)
                    ? locked1DRawBandpass_line(line - 2) : nullptr;
                const double *cbD = (line + 2 < frameHeight)
                    ? locked1DRawBandpass_line(line + 2) : nullptr;
                std::vector<double> gSkirt;
                auto coverPair = [&](const FrameBuffer *cov, int r0,
                                     int r1, double &pI,
                                     double &pQ) -> double {
                    if (!cov || !cov->holdsRealFrame() ||
                        !cov->frameHasExactCoverage()) return -1.0;
                    const float *ex = cov->exactCarrierRow(line);
                    const CombCarrierGrammar *gC = carrierGrammarLine(line);
                    const CombCarrierGrammar *gP =
                        cov->carrierGrammarLine(line);
                    if (!ex || !gC || !gP) return -1.0;
                    double Sii = 0, Siq = 0, Sqq = 0, SiY = 0, SqY = 0;
                    int n = 0;
                    for (int k = r0; k < r1; ++k) {
                        const int h = left + k;
                        const float v = ex[h];
                        if (!std::isfinite(v)) continue;
                        double sgn;
                        switch (lddecode::carrierGrammarSignedPhaseRelation(
                                    gC, h, gP, h)) {
                            case lddecode::CarrierPhaseRelation::Same:
                                sgn = 1.0; break;
                            case lddecode::CarrierPhaseRelation::Opposite:
                                sgn = -1.0; break;
                            default: continue;
                        }
                        const int cls = carrierSampleClass(line, h);
                        const double bi = cB4c[cls], bq = sB4c[cls];
                        const double c = sgn * (double)v;
                        Sii += bi * bi; Siq += bi * bq; Sqq += bq * bq;
                        SiY += c * bi;  SqY += c * bq;
                        n++;
                    }
                    if (n < 2) return -1.0;
                    const double det = Sii * Sqq - Siq * Siq;
                    if (std::fabs(det) < 1e-9) return -1.0;
                    pI = ( Sqq * SiY - Siq * SqY) / det;
                    pQ = (-Siq * SiY + Sii * SqY) / det;
                    return std::hypot(pI, pQ);
                };
                auto renderRegion = [&](int r0, int r1, bool regimeOK) {
                    const int s2 = std::max(0, r0 - kSkirtPad);
                    const int e2 = std::min(width, r1 + kSkirtPad);
                    double aI, aQ, bI, bQ;
                    const double ampP = coverPair(prevF, s2, e2, aI, aQ);
                    const double ampN = coverPair(nextF, s2, e2, bI, bQ);
                    double uI, uQ, amp;
                    if (ampP >= ampN) { uI = aI; uQ = aQ; amp = ampP; }
                    else              { uI = bI; uQ = bQ; amp = ampN; }
                    if (amp * invIreScale < kRefFloorIRE) return;
                    uI /= amp; uQ /= amp;
                    gSkirt.assign(size_t(e2 - s2), 0.0);
                    double gMax = 0.0;
                    for (int k = s2; k < e2; ++k) {
                        double g = 0.0;
                        for (int j = std::max(r0, k - 4);
                             j < std::min(r1, k + 5); ++j)
                            g += lddecode::kChromaEnvelopeFilter[4 + (k - j)];
                        gSkirt[k - s2] = g;
                        if (g > gMax) gMax = g;
                    }
                    if (gMax < 1e-9) return;
                    double num = 0.0, den = 0.0;
                    for (int k = s2; k < e2; ++k) {
                        double row;
                        const double c0 = cbp ? cbp[k] : 0.0;
                        if (cbU && cbD)
                            row = 0.5 * c0 - 0.25 * (cbU[k] + cbD[k]);
                        else if (cbU) row = 0.5 * (c0 - cbU[k]);
                        else if (cbD) row = 0.5 * (c0 - cbD[k]);
                        else          row = c0;
                        const int cls = carrierSampleClass(line, left + k);
                        const double basis = (gSkirt[k - s2] / gMax) *
                            (uI * cB4c[cls] + uQ * sB4c[cls]);
                        num += row * basis;
                        den += basis * basis;
                    }
                    if (den < 1e-9) return;
                    const double A = num / den;
                    if (A <= 0.0) return;      // nothing along the
                                               // certified direction:
                                               // abstain, never a zero claim
                    for (int k = s2; k < e2; ++k) {
                        const int cls = carrierSampleClass(line, left + k);
                        compactRow[k] = A * (gSkirt[k - s2] / gMax) *
                            (uI * cB4c[cls] + uQ * sB4c[cls]);
                        compactRegime[k] = regimeOK ? 1 : 0;
                    }
                };
                if (cbp) {
                    int rs = -1, regS = -1, regE = -1;
                    bool regOK = false;
                    auto flushRegion = [&]() {
                        if (regS < 0) return;
                        renderRegion(regS, regE, regOK);
                        regS = regE = -1;
                    };
                    for (int xi = 0; xi <= width; ++xi) {
                        const bool in = (xi < width) &&
                            std::hypot(cbp[xi],
                                cbp[std::min(xi + 1, width - 1)]) *
                                invIreScale >= kCompactIRE;
                        if (in) { if (rs < 0) rs = xi; continue; }
                        if (rs < 0) continue;
                        const int b0 = rs, b1 = xi; rs = -1;
                        const bool elemOK =
                            (b1 - b0 >= 2) && (b1 - b0 <= 3);
                        if (b1 - b0 > kBurstMax) { flushRegion(); continue; }
                        if (regS >= 0 && b0 - regE <= kBindGapMax) {
                            regE = b1;
                            regOK = regOK && elemOK;
                        } else {
                            flushRegion();
                            regS = b0; regE = b1;
                            regOK = elemOK;
                        }
                    }
                    flushRegion();
                }
            }

            if (yViewMode >= 0) {
                for (int h = left; h < right; ++h)
                    Y[h] = planeY(yViewMode, h);
                continue;
            }
            for (int hh = left; hh < right; ++hh) {
                const int xx = hh - left;
                pyRow0[xx] = planeY(0, hh);
                if (retractedRow) pyRow1[xx] = planeY(1, hh);
                if (oneDRow)      pyRow3[xx] = planeY(3, hh);
                if (ccMaskRow)    pyRow4[xx] = planeY(4, hh);
                if (notchLive) {
                    pyRow5[xx] = planeY(5, hh);
                    pyRow6[xx] = planeY(6, hh);
                }
                if (iceLive) pyRow7[xx] = planeY(7, hh);
            }
            const double *pyR0 = pyRow0.data();
            const double *pyR1 = retractedRow ? pyRow1.data() : pyRow0.data();
            const double *pyR3 = oneDRow      ? pyRow3.data() : pyRow0.data();
            const double *pyR4 = ccMaskRow    ? pyRow4.data() : pyRow0.data();
            const double *pyR5 = notchLive ? pyRow5.data() : pyRow0.data();
            const double *pyR6 = notchLive ? pyRow6.data() : pyRow0.data();
            const double *pyR7 = iceLive ? pyRow7.data() : pyRow0.data();
            auto planeYc = [&](int plane, int hh) -> double {
                const int xx = hh - left;
                switch (plane) {
                    case 1:  return pyR1[xx];
                    case 3:  return pyR3[xx];
                    case 4:  return pyR4[xx];
                    case 5:  return pyR5[xx];
                    case 6:  return pyR6[xx];
                    case 7:  return pyR7[xx];
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
            const double *resR6 =
                notchLive ? resRow6.data() : resRow0.data();
            const double *resR7 = iceLive ? resRow7.data() : resRow0.data();
            if (coarseRow) {
                for (int xx = 0; xx < width; ++xx) {
                    resRow0[xx] = pyR0[xx] - coarseRow[xx];
                    if (retractedRow) resRow1[xx] = pyR1[xx] - coarseRow[xx];
                    if (oneDRow)      resRow3[xx] = pyR3[xx] - coarseRow[xx];
                    if (ccMaskRow)    resRow4[xx] = pyR4[xx] - coarseRow[xx];
                    if (notchLive) {
                        resRow5[xx] = pyR5[xx] - coarseRow[xx];
                        resRow6[xx] = pyR6[xx] - coarseRow[xx];
                    }
                    if (iceLive) resRow7[xx] = pyR7[xx] - coarseRow[xx];
                }
            }
            auto resAt = [&](int plane, int xx) -> double {
                switch (plane) {
                    case 1:  return resR1[xx];
                    case 3:  return resR3[xx];
                    case 4:  return resR4[xx];
                    case 5:  return resR5[xx];
                    case 6:  return resR6[xx];
                    case 7:  return resR7[xx];
                    default: return resR0[xx];
                }
            };

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

            auto carrierCleanlinessOf = [&](int plane, int h0) -> double {
                double hf5[5], s5[5], c5[5], w5[5];
                double meanHF = 0.0;
                for (int j = 0; j < 5; ++j) {
                    const int k = j - 2;
                    const int hh = std::clamp(h0 + k, left, right - 1);
                    const double w = (j == 0 || j == 4) ? 0.5 : 1.0;
                    hf5[j] = resAt(plane, hh - left);
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

                if (certLineActive && std::isfinite(certExactRow[h])) {
                    Y[h] = rawH - (double)certExactRow[h];
                    continue;
                }

                if (compactStageLive && std::isfinite(compactRow[xi])) {
                    static const bool cregimeOn = []{
                        const char *e = std::getenv("LDCD_CREGIME");
                        return !e || e[0] != '0';
                    }();
                    if (cregimeOn && notchLive && compactRegime[xi]) {
                        Y[h] = planeYc(6, h);
                        continue;
                    }
                    Y[h] = rawH - compactRow[xi];
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
                if (!coveredFrame && !ldcdIceVectorMode() &&
                    icebergYRow && icebergWRow &&
                    std::isfinite((double)icebergYRow[xi])) {
                    const double iceReturn = icebergPolicyWeight * std::clamp(
                        (double)icebergWRow[xi], 0.0, 1.0);
                    ccReturn = std::max(ccReturn, iceReturn);
                }
                const bool notchSeatable =
                    notchLive && !nhHeard.empty() && nhHeard[xi];
                const bool haveChallenger =
                    !coveredFrame &&
                    (retractedRow || ccReturn > 0.0 || notchSeatable);
                if (!haveChallenger) {
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
                const double combPlatform = candidatePlatformResidualAt(0, h);
                const double combMiddle = combFour - combPlatform;
                const double combTop0 = candidateResidualAt(0, h) - combFour;
                auto reconstructTop = [&](int plane, double top) {
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

                double candY[6];
                int    candPlane[6];
                int    nCand = 0;
                const double identityTol = 1e-6 * irescale;
                auto addBaseCandidate = [&](double completeY, int plane) {
                    const double y = (plane == 0)
                        ? combTop0 : candidateTopAt(plane, h);
                    if (!std::isfinite(completeY) || !feasible(completeY) ||
                        !std::isfinite(y))
                        return;
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
                const bool gilgolBandDQ = bandRow && bandRow[xi] != 0;
                if (!gilgolBandDQ &&
                    retractedRow && std::isfinite((double)retractedRow[xi])) {
                    const double r = (double)retractedRow[xi];
                    const double ry = r;
                    bool retractedAdmitted =
                        retractedAdmitAll ||
                        !analysisRow ||
                        (analysisRow[xi].parallax.residualValid &&
                         analysisRow[xi].parallax.residualTrust >= 0.5f);
                    if (!retractedAdmitted && alienRow && alienRow[xi] > 0.5f)
                        retractedAdmitted = true;
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
                if (notchLive) {
                    if (nhHeard[xi]) addBaseCandidate(planeYc(6, h), 6);
                }

                if (iceLive && icebergWRow && icebergYRow &&
                    icebergWRow[xi] > 0.0f &&
                    std::isfinite((double)icebergYRow[xi]))
                    addBaseCandidate(planeYc(7, h), 7);
                static const bool ccFactsY = []{
                    const char *e = std::getenv("LDCD_CC_FACTS");
                    return !(e && std::atoi(e) == 0);
                }();
                const bool haveFactY = ccFactsY && dsExactRow &&
                                       std::isfinite(dsExactRow[h]);
                const float *bandRow = haveFactY
                    ? nullptr : bandResidueY_line(line);
                const double bandVal =
                    (bandRow && std::isfinite((double)bandRow[xi]))
                        ? (double)bandRow[xi] : 0.0;
                const double returnedY = haveFactY
                    ? rawH - (double)dsExactRow[h]
                    : planeYc(4, h) + bandVal;
                const bool returnedFeasible =
                    (haveFactY || ccReturn > 0.0 || bandVal != 0.0) &&
                    std::isfinite(returnedY) && feasible(returnedY);

                if (nCand == 0) {
                    Y[h] = reconstructTop(0, 0.0);
                    continue;
                }
                int inIdx[5];
                int nIn = 0;
                for (int k = 0; k < nCand; ++k)
                    inIdx[nIn++] = k;
                const bool returnedAdmitted = returnedFeasible;
                if (electBypass && combOK) {
                    Y[h] = reconstructTop(0, candidateTopAt(0, h));
                    continue;
                }

                if (nIn == 1 && !returnedAdmitted) {
                    Y[h] = reconstructTop(candPlane[inIdx[0]], candY[inIdx[0]]);
                    crashRefRow[xi] = Y[h];
                    continue;
                }

                double inHF[5], inCarrierCleanliness[5];
                double inCrossColorReturnEvidence[5] =
                    {0.0, 0.0, 0.0, 0.0, 0.0};
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

                const double selfAnchor = medoidD(inHF, baseNIn);
                if (returnedAdmitted) {
                    inHF[nIn] = candidateTopAt(4, h);
                    inCarrierCleanliness[nIn] =
                        carrierCleanlinessOf(4, h);
                    ++nIn;
                }
                double brightestLive =
                    -std::numeric_limits<double>::infinity();
                for (int k = 0; k < nIn; ++k) {
                    const int plane = (k < baseNIn)
                        ? candPlane[inIdx[k]] : 4;
                    const double complete = reconstructTop(plane, inHF[k]);
                    if (std::isfinite(complete) && complete > brightestLive)
                        brightestLive = complete;
                }
                if (std::isfinite(brightestLive))
                    crashRefRow[xi] = brightestLive;

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
                static const double kNotchHfKneeIRE = []{
                    const char *e = std::getenv("LDCD_NOTCHHF_KNEE");
                    return e ? std::atof(e) : 2.5;
                }();
                static const double kNotchHfTaperTauIRE = []{
                    const char *e = std::getenv("LDCD_NOTCHHF_TAU");
                    return e ? std::atof(e) : 8.0;
                }();
                static const double kNotchHfTaperGainIRE = []{
                    const char *e = std::getenv("LDCD_NOTCHHF_GAIN");
                    return e ? std::atof(e) : 0.75;
                }();
                const double kNotchHfTaperGain = kNotchHfTaperGainIRE * irescale;
                const double kNotchHfKnee      = kNotchHfKneeIRE * irescale;
                const double kNotchHfTaperTau  = kNotchHfTaperTauIRE * irescale;
                double sw[5];
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
                const double chromaT = std::clamp(
                    (combRemainMag0 * invIreScale -
                     kHighChromaSoftIRE) /
                        (kHighChromaHardIRE - kHighChromaSoftIRE),
                    0.0, 1.0) * (1.0 - illegalProof);
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
                double blendNum = 0.0, blendDen = 0.0;
                constexpr double kBlendTauIRE = 0.75;
                const double blendTau = kBlendTauIRE * irescale;
                double costs[4];
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
                        constexpr double kEarlyHandoffSlope = 2.0;
                        const double hfVouch = ev2Hf
                            ? (kHfBaseShare + (1.0 - kHfBaseShare) *
                                                  illegalProof)
                            : illegalProof;
                        const double legality =
                            hfVouch * std::min(kEarlyHandoffSlope * extra,
                                               imagePrefCap);
                        cost -= legality;
                        if (plane == 6 && kNotchHfTaperTau > kNotchHfKnee) {
                            const double taper = std::clamp(
                                (extra - kNotchHfKnee) /
                                    (kNotchHfTaperTau - kNotchHfKnee),
                                0.0, 1.0);
                            cost -= taper * std::min(kNotchHfTaperGain,
                                                     imagePrefCap);
                        }
                        cost -= (ev2Prox ? 1.0 : proximity01) *
                            std::max(0.0, inCrossColorReturnEvidence[k]);
                        if (plane == 1 && !ev2Named)
                            cost += chromaT * kHighChromaDemoteIRE * irescale -
                                    impulseT * kImpulseRetractedBiasIRE *
                                        irescale;
                        if (k < 4) costs[k] = cost;
                        if (cost < bestCost) {
                            bestCost = cost;
                            resultHF = inHF[k];
                        }
                    }
                    constexpr double kAnchorTauIRE = 1.5;
                    const double anchorCeilIRE =
                        anchorRow ? (double)anchorRow[h]
                                  : std::numeric_limits<double>::infinity();
                    for (int k = 0; k < nIn && k < 4; ++k) {
                        const int plane =
                            (k < baseNIn) ? candPlane[inIdx[k]] : 4;
                        const double yk = reconstructTop(plane, inHF[k]);
                        if (!std::isfinite(yk)) continue;
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
                        if (plane == 5 || plane == 6) {
                            const double blindIRE =
                                remainingCarrierMagnitudeOf(plane, h) *
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

            if (!coveredFrame) {
                const double *gbp = locked1DRawBandpass_line(line);
                const double kCrashFloor = 3.0 * irescale;
                for (int h = left; h < right; ++h) {
                    const int xx = h - left;
                    const double ref = crashRefRow[xx];
                    if (!std::isfinite(ref)) continue;
                    double tau = kCrashFloor;
                    if (gbp) {
                        const int x1 = std::min(xx + 1, width - 1);
                        const double e =
                            1.2 * std::hypot(gbp[xx], gbp[x1]);
                        if (e > tau) tau = e;
                    }
                    if (ref - Y[h] > tau) Y[h] = ref;
                }
            }
        } else {
            for (int h = left; h < right; ++h) {
                const double c = clpLine[h];
                Y[h] = std::isfinite(c) ? (double)rawLine[h] - c : (double)rawLine[h];
            }
        }
        if (!attrOn.empty()) {
            for (int x = 0; x < width; ++x)
                if (attrOn[x])
                    Y[left + x] = (double)rawLine[left + x] - attrC[x];
        }
        static const bool iceView = []{
            const char *e = std::getenv("LDCD_ICE_VIEW");
            return e && std::atoi(e) != 0;
        }();
        if (iceView && icebergYRow && icebergWRow) {
            const bool vecCorr = ldcdIceVectorMode();
            for (int h = left; h < right; ++h) {
                const int xx = h - left;
                if (!(icebergWRow[xx] > 0.0f) ||
                    !std::isfinite((double)icebergYRow[xx])) continue;
                if (vecCorr) {
                    const double w = std::min(1.0,
                        (double)icebergWRow[xx]);
                    Y[h] += w * (double)icebergYRow[xx];
                } else {
                    Y[h] = (double)icebergYRow[xx];
                }
            }
        }

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

    if (probeNhGrade) {
        for (int b = 0; b < 8; ++b)
            if (gCnt[b])
                std::fprintf(stderr, "[NHGRADE] bin=%d n=%ld errC=%.6f "
                                     "errN=%.6f win=%ld\n",
                             b, gCnt[b], gErrC[b], gErrN[b], gWin[b]);
    }

    {
        static const char *yrefBank = std::getenv("LDCD_YREF_BANK");
        static const char *yrefPath = std::getenv("LDCD_YREF");
        const int rFirst = videoParameters.firstActiveFrameLine;
        const int rLast  = videoParameters.lastActiveFrameLine;
        const int rLeft  = videoParameters.activeVideoStart;
        const int rWidth = videoParameters.activeVideoEnd - rLeft;
        const int rNl    = rLast - rFirst;
        const float rNaN = std::numeric_limits<float>::quiet_NaN();
        if (yrefBank && holdsRealFrame() && frameHasExactCoverage() &&
            rWidth > 0 && rNl > 0) {
            std::vector<unsigned char> df(rNl, 0);
            std::vector<float> ty((size_t)rNl * rWidth, rNaN);
            std::vector<float> ed((size_t)rNl * rWidth, rNaN);
            std::vector<double> env(rWidth), envS(rWidth);
            for (int line = rFirst; line < rLast; ++line) {
                const int li = line - rFirst;
                df[li] = certifiedDefLine(line) ? 1 : 0;
                const double *Yr = componentFrame->y(line);
                float *tr = ty.data() + (size_t)li * rWidth;
                for (int x = 0; x < rWidth; ++x)
                    tr[x] = (float)Yr[rLeft + x];
                const float *ex = exactCarrierRow(line);
                if (!ex) continue;
                for (int x = 0; x < rWidth; ++x) {
                    const int h = rLeft + x;
                    const float a = ex[h];
                    const float b = (h + 1 < videoParameters.fieldWidth)
                        ? ex[h + 1] : rNaN;
                    env[x] = (std::isfinite(a) && std::isfinite(b))
                        ? std::hypot((double)a, (double)b) : -1.0;
                }
                for (int x = 2; x < rWidth - 2; ++x) {
                    double s5 = 0; int n5 = 0;
                    for (int k = -2; k <= 2; ++k)
                        if (env[x + k] >= 0) { s5 += env[x + k]; ++n5; }
                    envS[x] = (n5 == 5) ? s5 / 5.0 : -1.0;
                }
                float *er = ed.data() + (size_t)li * rWidth;
                for (int x = 4; x < rWidth - 4; ++x)
                    if (envS[x - 2] >= 0 && envS[x + 2] >= 0)
                        er[x] = (float)(std::fabs(envS[x + 2] -
                                                  envS[x - 2]) / 4.0);
            }
            static std::mutex yrefMtx;
            std::lock_guard<std::mutex> lk(yrefMtx);
            static std::FILE *bf = std::fopen(yrefBank, "wb");
            if (bf) {
                const qint32 hdr[5] = { heldSeq1, rFirst, rLast,
                                        rLeft, rWidth };
                std::fwrite(hdr, sizeof(hdr), 1, bf);
                std::fwrite(df.data(), 1, df.size(), bf);
                std::fwrite(ty.data(), sizeof(float), ty.size(), bf);
                std::fwrite(ed.data(), sizeof(float), ed.size(), bf);
                std::fflush(bf);
            }
        }
        if (yrefPath && holdsRealFrame() && !frameHasExactCoverage() &&
            rWidth > 0 && rNl > 0) {
            struct YRefRec {
                int f0, f1, left, width;
                std::vector<unsigned char> df;
                std::vector<float> ty, ed;
            };
            static const std::map<qint32, YRefRec> *bank = []()
                -> const std::map<qint32, YRefRec> * {
                const char *pp = std::getenv("LDCD_YREF");
                std::FILE *f = pp ? std::fopen(pp, "rb") : nullptr;
                if (!f) return nullptr;
                auto *m = new std::map<qint32, YRefRec>();
                for (;;) {
                    qint32 hdr[5];
                    if (std::fread(hdr, sizeof(hdr), 1, f) != 1) break;
                    YRefRec r;
                    r.f0 = hdr[1]; r.f1 = hdr[2];
                    r.left = hdr[3]; r.width = hdr[4];
                    const size_t nl = (size_t)std::max(r.f1 - r.f0, 0);
                    r.df.resize(nl);
                    r.ty.resize(nl * r.width);
                    r.ed.resize(nl * r.width);
                    if (std::fread(r.df.data(), 1, nl, f) != nl ||
                        std::fread(r.ty.data(), sizeof(float),
                                   r.ty.size(), f) != r.ty.size() ||
                        std::fread(r.ed.data(), sizeof(float),
                                   r.ed.size(), f) != r.ed.size()) break;
                    (*m)[hdr[0]] = std::move(r);
                }
                std::fclose(f);
                if (m->empty()) { delete m; return nullptr; }
                return m;
            }();
            auto it = bank ? bank->find(heldSeq1)
                           : std::map<qint32, YRefRec>::const_iterator{};
            if (bank && it != bank->end() && it->second.f0 == rFirst &&
                it->second.f1 == rLast && it->second.left == rLeft &&
                it->second.width == rWidth) {
                struct Cell { long n = 0; double s2 = 0, mx = 0; long big = 0; };
                static std::mutex cMtx;
                static Cell all[4], lic[4];
                static long framesGraded = 0;
                const YRefRec &R = it->second;
                const bool haveLic = icebergReturnWeight_flat.size() >=
                    (size_t)rLast * demodWidth;
                std::lock_guard<std::mutex> lk(cMtx);
                for (int line = rFirst; line < rLast; ++line) {
                    const int li = line - rFirst;
                    if (!R.df[li]) continue;
                    const double *Yr = componentFrame->y(line);
                    const float *tr = R.ty.data() + (size_t)li * rWidth;
                    const float *er = R.ed.data() + (size_t)li * rWidth;
                    for (int x = 8; x < rWidth - 8; ++x) {
                        if (!std::isfinite(tr[x])) continue;
                        const double e =
                            (Yr[rLeft + x] - (double)tr[x]) / irescale;
                        const double dIRE = std::isfinite(er[x])
                            ? er[x] / irescale : 0.0;
                        const int b = (dIRE < 0.5) ? 0 : (dIRE < 1.0) ? 1
                                    : (dIRE < 2.0) ? 2 : 3;
                        auto upd = [&](Cell &c) {
                            c.n++; c.s2 += e * e;
                            if (std::fabs(e) > c.mx) c.mx = std::fabs(e);
                            if (std::fabs(e) > 3.0) c.big++;
                        };
                        upd(all[b]);
                        if (haveLic && icebergReturnWeight_flat[
                                (size_t)line * demodWidth + x] > 0.0f)
                            upd(lic[b]);
                    }
                }
                ++framesGraded;
                double s2 = 0; long n = 0, big = 0; double mx = 0;
                double ls2 = 0; long ln = 0;
                for (int b = 0; b < 4; ++b) {
                    s2 += all[b].s2; n += all[b].n; big += all[b].big;
                    mx = std::max(mx, all[b].mx);
                    ls2 += lic[b].s2; ln += lic[b].n;
                }
                std::fprintf(stderr,
                    "[YREF] f%d cum: frames %ld n %ld rms %.4f max %.2f "
                    ">3IRE %.3f%% | bins rms %.3f/%.3f/%.3f/%.3f "
                    "(n %ld/%ld/%ld/%ld) | LIC n %ld rms %.4f\n",
                    (int)heldSeq1, framesGraded, n,
                    n ? std::sqrt(s2 / n) : 0.0, mx,
                    n ? 100.0 * big / n : 0.0,
                    all[0].n ? std::sqrt(all[0].s2 / all[0].n) : 0.0,
                    all[1].n ? std::sqrt(all[1].s2 / all[1].n) : 0.0,
                    all[2].n ? std::sqrt(all[2].s2 / all[2].n) : 0.0,
                    all[3].n ? std::sqrt(all[3].s2 / all[3].n) : 0.0,
                    all[0].n, all[1].n, all[2].n, all[3].n,
                    ln, ln ? std::sqrt(ls2 / ln) : 0.0);
            }
        }
    }
}
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
    static const bool lurchCert = []{
        const char *e = std::getenv("LDCD_LURCH_CERT");
        return !(e && std::atoi(e) == 0);
    }();
    if (lurchCert && frameHasExactCoverage()) {
        constexpr double kCertMatchPx = 1.5;
        std::vector<std::vector<LurchStepRun>> certMissed(lastLine);
        std::vector<std::vector<std::pair<double, int>>> pend(lastLine);
        std::vector<double> lhat(width);
        std::vector<LurchStepRun> certRuns;
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
            run.edge = 0.5 * (eu + ed);
            continue;
        }
        const double e = run.edge;
        // median of three, by selection
        run.edge = std::max(std::min(eu, e), std::min(std::max(eu, e), ed));
    }
    return runs;
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

void Comb::FrameBuffer::solveLurchYCurve(int line, const double *apMean,
                                         int meanCount, int width,
                                         double *yOut)
{
    if (!apMean || !yOut || width <= 0 || meanCount <= 0)
        return;

    const int lastStart = meanCount - 1;

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
        if (std::strcmp(s, "notchhf") == 0)
            return 4;
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

static bool ldcdFactFitOn()
{
    static const bool on = []{
        const char *e = std::getenv("LDCD_FACT_FIT");
        return !(e && std::atoi(e) == 0);
    }();
    return on;
}
static void ldcdApplyPhaseSnap(const std::vector<double> &est,
                               const std::vector<double> &ref,
                               std::vector<double> &out,
                               int width, double irescale,
                               double ampMinIRE, double ampTauIRE,
                               bool clampRatio,
                               const double *gate)
{
    constexpr int kSnapHalf = 4;
    for (int xi = kSnapHalf; xi < width - kSnapHalf; ++xi) {
        double eI = 0.0, eQ = 0.0, rI = 0.0, rQ = 0.0;
        bool ok = true;
        for (int k = xi - kSnapHalf; k <= xi + kSnapHalf && ok; ++k) {
            const double b = ref[k];
            if (!std::isfinite(b)) { ok = false; break; }
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

namespace {
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

struct LdcdIceStat {
    long anchors = 0, matched = 0, licensed = 0, renderedSamples = 0;
    long withPrior = 0, pairs = 0, dRadius = 0, dSlopeSign = 0,
         dSlopeRatio = 0, dRms = 0, dAmbig = 0, noCand = 0,
         dPlatVet = 0, lagFilled = 0, inhMatched = 0;
};
thread_local LdcdIceStat gIceStat;

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

namespace {
struct IceBankRec {
    int lines0 = 0, lines1 = 0, left = 0, width = 0;
    bool platValid = false, rawValid = false;
    std::vector<float> lhat, plat, rawm;
};
const std::map<qint32, IceBankRec> *ldcdIceBankLoad()
{
    static const std::map<qint32, IceBankRec> *bank = []()
        -> const std::map<qint32, IceBankRec> * {
        const char *p = std::getenv("LDCD_ICE_BORROW");
        if (!p || !p[0]) return nullptr;
        std::FILE *f = std::fopen(p, "rb");
        if (!f) return nullptr;
        auto *m = new std::map<qint32, IceBankRec>();
        for (;;) {
            qint32 hdr[6];
            if (std::fread(hdr, sizeof(hdr), 1, f) != 1) break;
            IceBankRec r;
            r.lines0 = hdr[1]; r.lines1 = hdr[2];
            r.left = hdr[3]; r.width = hdr[4];
            r.platValid = (hdr[5] & 1) != 0;
            r.rawValid  = (hdr[5] & 2) != 0;
            const size_t nl = (size_t)std::max(r.lines1 - r.lines0, 0);
            r.lhat.resize(nl * r.width);
            r.plat.resize(nl * r.width);
            r.rawm.resize(nl * (r.width + 2));
            if (std::fread(r.lhat.data(), sizeof(float), r.lhat.size(), f)
                    != r.lhat.size() ||
                std::fread(r.plat.data(), sizeof(float), r.plat.size(), f)
                    != r.plat.size() ||
                std::fread(r.rawm.data(), sizeof(float), r.rawm.size(), f)
                    != r.rawm.size())
                break;
            (*m)[hdr[0]] = std::move(r);
        }
        std::fclose(f);
        if (m->empty()) { delete m; return nullptr; }
        return m;
    }();
    return bank;
}
const float *ldcdIceBankRow(const IceBankRec *r,
                            const std::vector<float> &plane,
                            int line, int rowWidth)
{
    if (!r || line < r->lines0 || line >= r->lines1) return nullptr;
    return plane.data() + (size_t)(line - r->lines0) * rowWidth;
}
} // namespace

bool Comb::FrameBuffer::buildIcebergReturn(const FrameBuffer *prevF,
                                           const FrameBuffer *nextF)
{
    icebergRecoveredY_flat.clear();
    icebergReturnWeight_flat.clear();
    const bool icebergTween = [&]{
        const char *e = std::getenv("LDCD_ICEBERG");
        if (e) return std::atoi(e) != 0;
        return configuration.yElection.ice;
    }();
    if (!icebergTween) return false;
    if (frameHasExactCoverage()) {
        static const char *bankPath = std::getenv("LDCD_ICE_BANK");
        if (bankPath && holdsRealFrame()) {
            const int firstLine = videoParameters.firstActiveFrameLine;
            const int lastLine  = videoParameters.lastActiveFrameLine;
            const int left      = videoParameters.activeVideoStart;
            const int right     = videoParameters.activeVideoEnd;
            const int width     = right - left;
            const int nl        = lastLine - firstLine;
            if (width > 0 && nl > 0) {
                const bool rawOK = left >= 1 &&
                    left + width + 1 <= videoParameters.fieldWidth;
                std::vector<float> lh((size_t)nl * width,
                    std::numeric_limits<float>::quiet_NaN());
                std::vector<float> pl((size_t)nl * width,
                    std::numeric_limits<float>::quiet_NaN());
                std::vector<float> rm((size_t)nl * (width + 2), 0.0f);
                for (int line = firstLine; line < lastLine; ++line) {
                    const int li = line - firstLine;
                    const float *pex0 = exactCarrierRow(line);
                    const float *pexU = (line - 1 >= firstLine)
                        ? exactCarrierRow(line - 1) : nullptr;
                    const float *pexD = (line + 1 < lastLine)
                        ? exactCarrierRow(line + 1) : nullptr;
                    const quint16 *rw = rawbuffer.data()
                        + (size_t)line * videoParameters.fieldWidth;
                    const quint16 *rwU = rawbuffer.data()
                        + (size_t)std::max(line - 1, 0) *
                          videoParameters.fieldWidth;
                    const quint16 *rwD = rawbuffer.data()
                        + (size_t)std::min(line + 1,
                              videoParameters.lastActiveFrameLine - 1) *
                          videoParameters.fieldWidth;
                    float *lrow = lh.data() + (size_t)li * width;
                    for (int xi = 0; xi < width; ++xi) {
                        const int h = left + xi;
                        if (pex0 && std::isfinite(pex0[h])) {
                            lrow[xi] = (float)((double)rw[h] -
                                               (double)pex0[h]);
                        } else if (pexU && pexD &&
                                   std::isfinite(pexU[h]) &&
                                   std::isfinite(pexD[h])) {
                            lrow[xi] = (float)(0.5 *
                                (((double)rwU[h] - (double)pexU[h]) +
                                 ((double)rwD[h] - (double)pexD[h])));
                        }
                    }
                    if (lockedLumaCacheValid) {
                        const double *ps = lockedLumaSmooth_line(line);
                        if (ps) {
                            float *prow = pl.data() + (size_t)li * width;
                            for (int xi = 0; xi < width; ++xi)
                                prow[xi] = (float)ps[xi];
                        }
                    }
                    if (rawOK) {
                        float *rrow = rm.data() + (size_t)li * (width + 2);
                        for (int xi = 0; xi < width + 2; ++xi)
                            rrow[xi] = (float)rw[left - 1 + xi];
                    }
                }
                static std::mutex bankMtx;
                std::lock_guard<std::mutex> lk(bankMtx);
                static std::FILE *bf = std::fopen(bankPath, "wb");
                if (bf) {
                    const qint32 hdr[6] = { heldSeq1, firstLine, lastLine,
                        left, width,
                        (qint32)((lockedLumaCacheValid ? 1 : 0) |
                                 (rawOK ? 2 : 0)) };
                    std::fwrite(hdr, sizeof(hdr), 1, bf);
                    std::fwrite(lh.data(), sizeof(float), lh.size(), bf);
                    std::fwrite(pl.data(), sizeof(float), pl.size(), bf);
                    std::fwrite(rm.data(), sizeof(float), rm.size(), bf);
                    std::fflush(bf);
                }
            }
        }
        return false;
    }
    const FrameBuffer *nb[2] = {
        (prevF && prevF->frameHasExactCoverage()) ? prevF : nullptr,
        (nextF && nextF->frameHasExactCoverage()) ? nextF : nullptr,
    };
    const IceBankRec *borrow[2] = { nullptr, nullptr };
    if (!nb[0] && !nb[1] && holdsRealFrame()) {
        if (const auto *bank = ldcdIceBankLoad()) {
            const int gFirst = videoParameters.firstActiveFrameLine;
            const int gLast  = videoParameters.lastActiveFrameLine;
            const int gLeft  = videoParameters.activeVideoStart;
            const int gWidth = videoParameters.activeVideoEnd - gLeft;
            auto geomOK = [&](const IceBankRec &r) {
                return r.lines0 == gFirst && r.lines1 == gLast &&
                       r.left == gLeft && r.width == gWidth;
            };
            auto it = bank->lower_bound(heldSeq1);
            if (it != bank->end() && geomOK(it->second))
                borrow[1] = &it->second;
            if (it != bank->begin() && geomOK(std::prev(it)->second))
                borrow[0] = &std::prev(it)->second;
            static const bool borrowStats = []{
                const char *e = std::getenv("LDCD_ICEBERG_STATS");
                return e && std::atoi(e) != 0;
            }();
            if (borrowStats && (borrow[0] || borrow[1]))
                std::fprintf(stderr,
                    "[ICEBORROW] frame %d: prev %d next %d\n",
                    (int)heldSeq1,
                    borrow[0] ? (int)std::prev(it)->first : -1,
                    borrow[1] ? (int)it->first : -1);
        }
    }
    if ((!nb[0] && !borrow[0]) || (!nb[1] && !borrow[1])) return false;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;
    if (width <= 0) return false;

    std::vector<double> notchC(width), notchP(width), notchN(width);
    static const bool iceNotchLocator = []{
        const char *e = std::getenv("LDCD_ICE_NOTCH");
        return !e || std::atoi(e) != 0;
    }();

    static const bool iceAmbigVeto = []{
        const char *e = std::getenv("LDCD_ICE_AMBIG");
        return !e || std::atoi(e) != 0;
    }();
    static const double iceMatchRmsIRE = []{
        const char *e = std::getenv("LDCD_ICE_RMS");
        const double v = e ? std::atof(e) : 8.0;
        return (v > 0.0) ? v : 8.0;
    }();
    constexpr double kIceBlindRadiusPx = 24.0;
    static const double iceBlindRadiusScale = []{
        const char *e = std::getenv("LDCD_ICE_RADIUS");
        const double v = e ? std::atof(e) : 1.0;
        return (v > 0.0) ? v : 1.0;
    }();
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
        {
            std::vector<int> lhatN(width, 0);
            std::vector<double> sideVal[2];
            sideVal[0].assign(width, std::numeric_limits<double>::quiet_NaN());
            sideVal[1].assign(width, std::numeric_limits<double>::quiet_NaN());
            for (int side = 0; side < 2; ++side) {
                const FrameBuffer *fb = nb[side];
                if (!fb) {
                    if (borrow[side]) {
                        const float *bl = ldcdIceBankRow(borrow[side],
                            borrow[side]->lhat, line, width);
                        if (bl)
                            for (int xi = 0; xi < width; ++xi)
                                if (std::isfinite(bl[xi])) {
                                    lhatN[xi] += 1;
                                    sideVal[side][xi] = (double)bl[xi];
                                }
                    }
                    continue;
                }
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
            static const bool iceStats = []{
                const char *e = std::getenv("LDCD_ICEBERG_STATS");
                return e && std::atoi(e) != 0;
            }();
            std::vector<double> platBorrowBuf[2];
            auto sidePlatOK = [&](int k) -> bool {
                if (nb[k]) return nb[k]->lockedLumaCacheValid;
                return borrow[k] && borrow[k]->platValid &&
                       ldcdIceBankRow(borrow[k], borrow[k]->plat,
                                      line, width) != nullptr;
            };
            auto sidePlatRow = [&](int k) -> const double * {
                if (nb[k]) return nb[k]->lockedLumaSmooth_line(line);
                const float *r = ldcdIceBankRow(borrow[k], borrow[k]->plat,
                                                line, width);
                platBorrowBuf[k].resize(width);
                for (int x = 0; x < width; ++x)
                    platBorrowBuf[k][x] = (double)r[x];
                return platBorrowBuf[k].data();
            };
            if (icebergTween && sidePlatOK(0) && sidePlatOK(1) &&
                lockedLumaCacheValid) {
                const double *platC = lockedLumaSmooth_line(line);
                const double *platP = sidePlatRow(0);
                const double *platN = sidePlatRow(1);

                const double *vetP = platP;
                const double *vetN = platN;

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

                if (iceNotchLocator && left >= 1 &&
                    left + width + 1 <= videoParameters.fieldWidth) {
                    const size_t stride = videoParameters.fieldWidth;
                    const quint16 *rawC = rawLine;
                    auto notchAt = [](const quint16 *r, int i) {
                        return 0.5 * (static_cast<double>(r[i - 1]) +
                                      static_cast<double>(r[i + 1]));
                    };
                    auto fillSide = [&](int k, std::vector<double> &dst) {
                        if (nb[k]) {
                            const quint16 *r = nb[k]->rawbuffer.data()
                                + static_cast<size_t>(line) * stride;
                            for (int x = 0; x < width; ++x)
                                dst[x] = notchAt(r, left + x);
                            return true;
                        }
                        if (!borrow[k] || !borrow[k]->rawValid) return false;
                        const float *r = ldcdIceBankRow(borrow[k],
                            borrow[k]->rawm, line, width + 2);
                        if (!r) return false;
                        for (int x = 0; x < width; ++x)
                            dst[x] = 0.5 * ((double)r[x] + (double)r[x + 2]);
                        return true;
                    };
                    if (fillSide(0, notchP) && fillSide(1, notchN)) {
                        for (int x = 0; x < width; ++x)
                            notchC[x] = notchAt(rawC, left + x);
                        platC = notchC.data();
                        platP = notchP.data();
                        platN = notchN.data();
                    }
                }
                static const bool iceLegacy = []{
                    const char *e = std::getenv("LDCD_ICE_LEGACY");
                    return e && std::atoi(e) != 0;
                }();
                if (!iceLegacy) {
                    std::vector<double> nl(width, 0.0), env(width, 0.0);
                    for (int x = 0; x < width; ++x) {
                        const int h = left + x;
                        if (h - 1 < 0 ||
                            h + 1 >= videoParameters.fieldWidth) continue;
                        nl[x] = 0.5 * ((double)rawLine[h - 1] +
                                       (double)rawLine[h + 1]);
                    }
                    const double *bpR = locked1DRawBandpass_line(line);
                    if (bpR)
                        for (int x = 0; x + 1 < width; ++x)
                            env[x] = std::hypot(bpR[x], bpR[x + 1]);
                    std::vector<double> vRow(width,
                        std::numeric_limits<double>::quiet_NaN());
                    std::vector<double> wRow(width, 0.0);
                    const double riseThr = 3.0 * irescale;   // mesa height
                    const double kMesaCurv = 0.30;           // flat-top gate
                    const double envFloor = 2.0 * irescale;  // remnant floor
                    for (int x = 10; x < width - 11; ++x) {
                        if (nl[x] < nl[x - 1] || nl[x] < nl[x + 1]) continue;
                        const double base0 =
                            0.5 * (nl[x - 4] + nl[x + 4]);
                        const double rise = nl[x] - base0;
                        if (rise < riseThr) continue;
                        const double curv = std::fabs(
                            nl[x - 1] - 2.0 * nl[x] + nl[x + 1]);
                        if (curv > kMesaCurv * rise) continue;   // intact top
                        if (env[x] < envFloor) continue;         // no remnant
                        gIceStat.anchors++;
                        // Pin refinement from the remnant envelope (parabolic
                        // on the local env crest; the vector never places).
                        double p = (double)x;
                        {
                            int xe = x;
                            for (int k = -2; k <= 2; ++k)
                                if (env[x + k] > env[xe]) xe = x + k;
                            const double e0 = env[xe - 1], e1 = env[xe],
                                         e2 = env[xe + 1];
                            const double den = e0 - 2.0 * e1 + e2;
                            if (std::fabs(den) > 1e-9) {
                                double d = 0.5 * (e0 - e2) / den;
                                if (std::fabs(d) <= 1.0) p = xe + d;
                            }
                        }
                        auto shapeAt = [](double u, double wf) {
                            double t2 = 1.0 - std::fabs(u) / wf;
                            if (t2 <= 0.0) return 0.0;
                            return t2 * t2 * (3.0 - 2.0 * t2);
                        };
                        struct FitR { double h, w, del, J; bool ok; };
                        auto fitCover = [&](const std::vector<double> &cov)
                            -> FitR {
                            FitR best{0, 2.0, 0, 1e9, false};
                            for (double del = -2.0; del <= 2.01;
                                 del += 0.5) {
                                for (double wf = 1.5; wf <= 4.01;
                                     wf += 0.5) {
                                    const int span = (int)std::ceil(wf) + 1;
                                    const double c0 = p + del;
                                    const int lo = (int)std::floor(
                                        c0 - span - 2);
                                    const int hi = (int)std::ceil(
                                        c0 + span + 2);
                                    if (lo < 0 || hi >= width) continue;
                                    double bacc = 0; int bn = 0;
                                    bool bad = false;
                                    for (int xx = lo; xx <= hi; ++xx) {
                                        if (!std::isfinite(cov[xx]))
                                            { bad = true; break; }
                                        if (std::fabs(xx - c0) >
                                            (double)span) {
                                            bacc += cov[xx]; ++bn;
                                        }
                                    }
                                    if (bad || bn < 3) continue;
                                    const double bs = bacc / bn;
                                    double num = 0, den = 0;
                                    for (int xx = lo; xx <= hi; ++xx) {
                                        const double sh = shapeAt(
                                            xx - c0, wf);
                                        num += (cov[xx] - bs) * sh;
                                        den += sh * sh;
                                    }
                                    if (den < 1e-9) continue;
                                    const double hh = num / den;
                                    if (hh <= 0.0) continue;
                                    double ss = 0; int nn = 0;
                                    for (int xx = lo; xx <= hi; ++xx) {
                                        if (std::fabs(xx - c0) >
                                            (double)span) continue;
                                        const double d2 = cov[xx] - bs -
                                            hh * shapeAt(xx - c0, wf);
                                        ss += d2 * d2; ++nn;
                                    }
                                    const double J = std::sqrt(ss /
                                        std::max(nn, 1)) /
                                        std::max(hh, 1.0 * irescale);
                                    if (J < best.J)
                                        best = {hh, wf, del, J, true};
                                }
                            }
                            return best;
                        };
                        const FitR A = fitCover(sideVal[0]);
                        const FitR B = fitCover(sideVal[1]);
                        if (!A.ok || !B.ok) continue;
                        if (A.h < riseThr || B.h < riseThr) continue;
                        if (A.J > 0.35 || B.J > 0.35) continue;
                        if (std::fabs(A.h - B.h) >
                            std::max(3.0 * irescale,
                                     0.5 * std::max(A.h, B.h))) continue;
                        if (std::fabs(A.del + B.del) > 2.0) continue;
                        gIceStat.matched++;
                        const double hT = 0.5 * (A.h + B.h);
                        const double wT = std::max(1.5,
                            0.5 * (A.w + B.w));
                        const double deficit = hT - rise;
                        if (deficit <= 1.0 * irescale) continue;
                        const double aPred = hT *
                            (1.0 - shapeAt(1.0, wT));
                        if (env[x] < 0.4 * aPred) continue;
                        double J2;
                        {
                            double ss = 0; int nn = 0; bool bad = false;
                            const int span = (int)std::ceil(wT) + 1;
                            const int qi = (int)std::lround(p);
                            for (int u = -span; u <= span; ++u) {
                                const int xx = qi + u;
                                if (xx < 1 || xx >= width - 1)
                                    { bad = true; break; }
                                const double mObs = nl[xx] - base0;
                                const double mPred = 0.5 *
                                    (hT * shapeAt(xx - p - 1.0, wT) +
                                     hT * shapeAt(xx - p + 1.0, wT));
                                const double d2 = mObs - mPred;
                                ss += d2 * d2; ++nn;
                            }
                            if (bad || !nn) continue;
                            J2 = std::sqrt(ss / nn) /
                                 std::max(hT, 1.0 * irescale);
                        }
                        if (J2 > 0.40) continue;
                        gIceStat.licensed++;
                        auto ramp01 = [](double v) {
                            v = std::clamp(v, 0.0, 1.0);
                            return v * v * (3.0 - 2.0 * v);
                        };
                        double lic = std::clamp(
                            deficit / (3.0 * irescale), 0.0, 1.0);
                        lic *= ramp01((0.35 - std::max(A.J, B.J)) / 0.15);
                        lic *= ramp01((0.40 - J2) / 0.15);
                        if (lic <= 0.0) continue;
                        const double corrCap = 6.0 * irescale;
                        const int xl = std::max(1,
                            (int)std::floor(p - wT) - 1);
                        const int xr = std::min(width - 2,
                            (int)std::ceil(p + wT) + 1);
                        for (int u = xl; u <= xr; ++u) {
                            const double sh = shapeAt(u - p, wT);
                            const double shN = 0.5 *
                                (shapeAt(u - p - 1.0, wT) +
                                 shapeAt(u - p + 1.0, wT));
                            double corr = hT * (sh - shN);
                            corr = std::clamp(corr, -corrCap, corrCap);
                            const double w = lic * (0.5 + 0.5 * sh);
                            if (w > wRow[u] && std::fabs(corr) > 1e-9) {
                                wRow[u] = w;
                                vRow[u] = corr;
                            }
                        }
                    }
                    // Taper the weight at run edges (value never smoothed).
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
                            double t2 = std::clamp(dist[x] / 3.0, 0.0, 1.0);
                            wRow[x] *= t2 * t2 * (3.0 - 2.0 * t2);
                        }
                    }
                    for (int x = 0; x < width; ++x) {
                        if (wRow[x] <= 0.0 || lhatN[x] < 2) continue;
                        if (!std::isfinite(vRow[x])) continue;
                        const double w = std::min(1.0, wRow[x]);
                        const size_t oi =
                            static_cast<size_t>(line) * demodWidth + x;
                        icebergRecoveredY_flat[oi] =
                            static_cast<float>(vRow[x]);
                        icebergReturnWeight_flat[oi] =
                            static_cast<float>(w);
                        anyIcebergReturn = true;
                        gIceStat.renderedSamples++;
                    }
                } else {
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
                    const double rmsCap = iceMatchRmsIRE * irescale *
                                          (priorInherited ? 0.75 : 1.0);
                    if (bestS > rmsCap) {
                        gIceStat.dRms++; continue;
                    }
                    if (priorInherited) gIceStat.inhMatched++;
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
                } // iceLegacy
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

void Comb::FrameBuffer::applyToneToFit(const FrameBuffer *prevF)
{
    static const bool toneOn = []{
        const char *e = std::getenv("LDCD_CERT_TONE");
        return !(e && std::atoi(e) == 0);
    }();
    if (!toneOn || certifiedOneDLevel() == 0) return;
    if (frameHasExactCoverage()) return;                // anchors stand

    static const bool actOn = []{
        const char *e = std::getenv("LDCD_PROBE_TONEACT");
        return e && std::atoi(e) != 0;
    }();
    auto standDown = [&](const char *why) {
        if (actOn)
            std::fprintf(stderr,
                "[TONEACT] frame %d  uncovered  STAND DOWN: %s\n",
                (int)heldSeq1, why);
    };

    if (!prevF || !prevF->frameHasExactCoverage()) {
        standDown("previous frame is not covered");
        return;
    }
    const QVector<float> &pay =
        !syncIncFirst.isEmpty() ? syncIncFirst : syncIncSecond;
    if (pay.size() < 4) { standDown("no dgSyncIncrement payload"); return; }
    const double gOmega = pay[0];
    const double gConf  = pay[1];
    const double dtF    = pay[2];
    if (gConf < 0.2 || dtF <= 0.0 || dtF > 8.0) {
        standDown("payload out of range (gConf/dtF)");
        return;
    }

    const qint32 targetCid = !syncIncFirst.isEmpty() ? cadenceIdFirst
                                                     : cadenceIdSecond;
    const int targetIdx = cadenceIndex(targetCid);
    if (targetIdx < 0) {
        standDown("sentinel cadenceId: no cadence position");
        return;
    }
    const int anchorIdx =
        ((targetIdx - (int)dtF) % CADENCE_NTSC_CYCLE +
         CADENCE_NTSC_CYCLE) % CADENCE_NTSC_CYCLE;

    double polarity = 0.0;
    {
        struct PolarityRow { int anchor, target; double rot; };
        static const PolarityRow kPolarity[] = {
            { 0, 4, 0.0   },   // Adef -> B2   (normal dominance)
            { 0, 3, M_PI  },   // Adef -> B1   (inverted dominance)
            { 7, 8, M_PI  },   // Cdef -> D1   (normal dominance)
            { 7, 9, 0.0   },   // Cdef -> D2   (inverted dominance)
        };
        bool listed = false;
        for (const PolarityRow &row : kPolarity) {
            if (row.anchor == anchorIdx && row.target == targetIdx) {
                polarity = row.rot;
                listed = true;
                break;
            }
        }
        if (!listed) {
            char buf[96];
            std::snprintf(buf, sizeof(buf),
                          "unlisted polarity pair (anchor %d -> target %d)",
                          anchorIdx, targetIdx);
            standDown(buf);
            return;
        }
    }

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int width     = videoParameters.activeVideoEnd - left;
    if (width <= 0) { standDown("zero active width"); return; }
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
    if (defPar < 0) { standDown("no def lines in the anchor frame"); return; }

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

    const double floorRaw = 2.0 * irescale;
    constexpr double kMaxDelta = 25.0 * M_PI / 180.0;

    std::vector<double> backstopAbs;

    std::vector<double> actD0;
    // Signed per-region readings; only their median leaves this stage.
    std::vector<double> regionD0;
    double confSum = 0.0;

    for (int r = 0; r < nx * ny; ++r) {
        if (aN[r] < 64 || fN[r] < 64) continue;
        if (std::hypot(aI[r], aQ[r]) / aN[r] < floorRaw) continue;
        if (std::hypot(fI[r], fQ[r]) / fN[r] < floorRaw) continue;
        double conf = gConf;
        if (r < nRegPay) conf *= std::clamp((double)pay[4 + r * 2 + 1],
                                            0.0, 1.0);
        if (conf <= 0.05) continue;
        // The anchor is brought into THIS field's space before it is
        // differenced -- the conversion, applied once, from the table.
        const double aPh =
            std::atan2(aQ[r], aI[r]) + gOmega * dtF + polarity;
        const double d0 = std::atan2(
            std::sin(aPh) * fI[r] - std::cos(aPh) * fQ[r],
            std::cos(aPh) * fI[r] + std::sin(aPh) * fQ[r]);
        backstopAbs.push_back(std::fabs(d0));
        regionD0.push_back(d0);
        confSum += conf;
        if (actOn) actD0.push_back(std::fabs(d0) * (180.0 / M_PI));
    }

    double deltaGlobal = 0.0;
    double madDeg = -1.0;
    if (!regionD0.empty()) {
        std::vector<double> bs = backstopAbs;
        std::sort(bs.begin(), bs.end());
        const double med = bs[bs.size() / 2];
        if (med > kMaxDelta) {
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true))
                std::fprintf(stderr,
                    "[TONE] anchor %d -> target %d (dt %.0f fields):"
                    " median |d0| %.1f deg after conversion -- polarity"
                    " table is wrong for this pair; stage stood down.\n",
                    anchorIdx, targetIdx, dtF, med * 180.0 / M_PI);
        } else {
            std::vector<double> sv = regionD0;
            std::sort(sv.begin(), sv.end());
            const double medSigned = sv[sv.size() / 2];
            std::vector<double> dev(sv.size());
            for (size_t i = 0; i < sv.size(); ++i)
                dev[i] = std::fabs(sv[i] - medSigned);
            std::sort(dev.begin(), dev.end());
            const double mad = dev[dev.size() / 2];
            madDeg = mad * 180.0 / M_PI;
            if (std::fabs(medSigned) <= mad) {
                deltaGlobal = 0.0;
            } else {
                const double confMean = confSum / (double)regionD0.size();
                deltaGlobal = std::clamp(confMean * medSigned,
                                         -kMaxDelta, kMaxDelta);
            }
        }
    }

    if (actOn) {
        std::fprintf(stderr,
            "[TONEACT] frame %d  anchor %d -> target %d  polarity %.0f"
            "  dtF %.0f  voting regions %zu\n",
            (int)heldSeq1, anchorIdx, targetIdx, polarity * 180.0 / M_PI,
            dtF, actD0.size());
        if (!actD0.empty()) {
            std::sort(actD0.begin(), actD0.end());
            const size_t n = actD0.size();
            std::fprintf(stderr,
                "      |d0| post-conv p50 %7.2f  p90 %7.2f  max %7.2f"
                "   MAD %7.2f   applied %7.2f\n",
                actD0[n / 2],
                actD0[(n * 9) / 10 < n ? (n * 9) / 10 : n - 1],
                actD0[n - 1], madDeg, deltaGlobal * 180.0 / M_PI);
        }
    }

    if (deltaGlobal == 0.0) return;          // nothing to apply
    const double cosRot = std::cos(deltaGlobal);
    const double sinRot = std::sin(deltaGlobal);
    std::vector<double> src(width);
    for (int line = firstLine; line < lastLine; ++line) {
        float *fit = carrierFit_flat.data() + (size_t)line * demodWidth;
        for (int xi = 0; xi < width; ++xi) src[xi] = fit[xi];
        for (int xi = 0; xi < width; ++xi) {
            const double sm = xi > 0 ? src[xi - 1] : src[xi];
            const double sp = xi + 1 < width ? src[xi + 1] : src[xi];
            const double q = 0.5 * (sm - sp);
            fit[xi] = (float)(cosRot * src[xi] + sinRot * q);
        }
    }
}

void Comb::FrameBuffer::buildLumaWitnessModel()
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
    // The fit's STATE, carried in components from attribution to publication.
    // See "the carrier never leaves the basis span" below.
    std::vector<double> fitCompI(static_cast<size_t>(std::max(0, width)), 0.0);
    std::vector<double> fitCompQ(static_cast<size_t>(std::max(0, width)), 0.0);
    std::vector<double> partWeight(static_cast<size_t>(width), 1.0);

    auto median3 = [](double a, double b, double c) -> double {
        if (a > b) std::swap(a, b);
        if (b > c) std::swap(b, c);
        if (a > b) std::swap(a, b);
        return b;
    };

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

            for (int xi = 0; xi < width; ++xi) {
                const int s0 = std::clamp(xi - 2, 0, meanCount - 1);
                const int s1 = std::clamp(xi - 1, 0, meanCount - 1);
                refinedY[xi] = 0.5 * (winFloor[s0] + winFloor[s1]);
            }
            lurchSharpenCoarsePrior(winFloor.data(), meanCount, width,
                                    refinedY, nullptr);

            for (int s = 0; s < meanCount; ++s) {
                double sII = 0.0, sIQ = 0.0, sQQ = 0.0;
                double sIY = 0.0, sQY = 0.0;
                double sampleWeight = 0.0;

                double refinedMean = 0.0;
                double minRefined = 1e300;
                double maxRefined = -1e300;
                double wLaneI = 0.0, wLaneQ = 0.0;

                for (int k = 0; k < 4; ++k) {
                    const int xi = s + k;
                    refinedMean += refinedY[xi];
                    minRefined = std::min(minRefined, refinedY[xi]);
                    maxRefined = std::max(maxRefined, refinedY[xi]);
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
                    if (std::fabs(bI) >= std::fabs(bQ)) wLaneI += w;
                    else                                wLaneQ += w;
                }

                refinedMean *= 0.25;

                double fitI = 0.0;
                double fitQ = 0.0;
                static const bool oneParamFit = []{
                    const char *e = std::getenv("LDCD_FIT_1P");
                    return e && std::atoi(e) != 0;
                }();
                if (oneParamFit) sIQ = 0.0;
                const double det = sII * sQQ - sIQ * sIQ;
                const bool fitValid =
                    oneParamFit
                        ? (wLaneI >= 1.25 && wLaneQ >= 1.25 &&
                           sII > 1e-9 && sQQ > 1e-9)
                        : (sampleWeight >= 2.5 && std::fabs(det) > 1e-9);
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

                winScore[s] =
                    errIRE +
                    0.75 * latticeIRE +
                    0.25 * floorDriftIRE +
                    0.15 * ySpanIRE -
                    0.10 * std::min(ampIRE, 24.0);
            }

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

                    double movingResidualSample = 0.0;
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

                const double bIxi = basisI[xi];
                const double bQxi = basisQ[xi];
                const double bNorm2 = bIxi * bIxi + bQxi * bQxi;
                const double bInv = bNorm2 > 1e-12 ? (1.0 / bNorm2) : 0.0;
                auto admitCompositeDelta = [&](double &cI, double &cQ,
                                               double d) {
                    cI += d * bIxi * bInv;
                    cQ += d * bQxi * bInv;
                };

                auto finalizeCarrierPair = [&](double candidateI,
                                               double candidateQ,
                                               double &outI, double &outQ) {
                    outI = candidateI;
                    outQ = candidateQ;
                    double sample = outI * bIxi + outQ * bQxi;

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

                        admitCompositeDelta(
                            outI, outQ,
                            (parallax.commonSample - sample) * sampleAnchor);
                        sample = outI * bIxi + outQ * bQxi;
                    }

                    if (residualTightenSupport > 0.0) {
                        const double bounded = std::clamp(
                            sample, residualCarrierLo, residualCarrierHi);
                        admitCompositeDelta(outI, outQ, bounded - sample);
                        sample = bounded;
                    }

                    const double capped = std::clamp(
                        sample, -maxCarrierSamples, maxCarrierSamples);
                    admitCompositeDelta(outI, outQ, capped - sample);
                };

                double cI = 0.0, cQ = 0.0;
                finalizeCarrierPair(modelI, modelQ, cI, cQ);
                const double baselineCf = cI * bIxi + cQ * bQxi;
                double cf = baselineCf;

                bool sharedConstraintApplied = false;
                double sharedDelta = 0.0;
                const auto &sharedResidual = analysisRow[xi].residual;
                const int sharedSurvivors = sharedResidual.survivorCount();
                const bool sharedUseful =
                    partWeight[xi] > 0.0 &&
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
                    admitCompositeDelta(cI, cQ, sharedDelta);
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
                fitCompI[xi] = cI;
                fitCompQ[xi] = cQ;
                flattened[xi] = rawWhole[xi] - cf;

                fitRow[xi] = static_cast<float>(cf);
            }

            {
                if ((int)envI.size() < width) {
                    envI.resize(width, 0.0);
                    envQ.resize(width, 0.0);
                    envTmp.resize(width, 0.0);
                }

                std::copy(fitCompI.begin(), fitCompI.begin() + width,
                          envI.begin());
                std::copy(fitCompQ.begin(), fitCompQ.begin() + width,
                          envQ.begin());

                lddecode::projectExpressibleChromaEnvelope(
                    envI.data(), nullptr, width, envTmp.data(), 0);
                std::copy(envTmp.begin(), envTmp.begin() + width,
                          envI.begin());
                lddecode::projectExpressibleChromaEnvelope(
                    envQ.data(), nullptr, width, envTmp.data(), 1);
                std::copy(envTmp.begin(), envTmp.begin() + width,
                          envQ.begin());

                for (int xi = 0; xi < width; ++xi) {
                    const double modelled =
                        envI[xi] * basisI[xi] + envQ[xi] * basisQ[xi];
                    const double capped = std::clamp(
                        modelled, -maxCarrierSamples, maxCarrierSamples);
                    const double g = (modelled != 0.0)
                        ? (capped / modelled)
                        : 1.0;
                    envI[xi] *= g;
                    envQ[xi] *= g;
                    fitCompI[xi] = envI[xi];
                    fitCompQ[xi] = envQ[xi];
                    carrierFit[xi] = capped;
                    flattened[xi] = rawWhole[xi] - capped;
                    fitRow[xi] = static_cast<float>(capped);
                }
            }
        } else {
            for (int xi = 0; xi < width; ++xi) {
                const double cf = 0.0;
                carrierFit[xi] = cf;
                fitCompI[xi] = 0.0;
                fitCompQ[xi] = 0.0;
                flattened[xi] = rawWhole[xi];
                fitRow[xi] = 0.0f;
                evidenceRow[xi].viewCount = 0;
            }
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
        carrierRetractionModelValid = true;
    }

    if (!carrierRetractionModelValid)
        return;

    const lddecode::CombReachSourceFrame carrierFitSource =
        lddecode::makeCarrierFitScalarReachSource();

    auto softReachGate = [](double diffIRE, double softIRE, double hardIRE) {
        if (diffIRE <= softIRE)
            return 1.0;
        if (diffIRE >= hardIRE)
            return 0.0;
        const double t = (diffIRE - softIRE) /
                         std::max(1e-9, hardIRE - softIRE);
        return 1.0 - (t * t * (3.0 - 2.0 * t));
    };

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
    constexpr int kNLegs = 4;
    std::vector<double> legGateScratch[kNLegs];
    for (int k = 0; k < kNLegs; ++k)
        legGateScratch[k].assign(static_cast<size_t>(width), 0.0);

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
        auto reachGate = [&](int xi, const FitLeg &leg,
                             double *lumaGateOut = nullptr,
                             double *carrGateOut = nullptr) {
            if (!leg.present)
                return 0.0;

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
            const double centerParticipation =
                static_cast<double>(eligRow[xi]);
            if (centerParticipation <= 0.0) {
                combRow[xi] = 0.0f;
                continue;
            }

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
                const double eFloor = 2.0 * ampFloor * ampFloor;
                if (e0 < eFloor) {
                    ownedFallback = 1.0;   // harmless confiscation
                    p2corrCode = 3.0;      // dump-only: +3 = quiet operand
                } else {
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

                const double cancelled =
                    0.5 * (static_cast<double>(fitRow[xi]) - neighborFit);
                const double strength = std::min(1.0, wSum);

                combRow[xi] = static_cast<float>(
                    static_cast<double>(fitRow[xi]) * (1.0 - strength) *
                        ownedFallback * centerParticipation +
                    cancelled * strength);

                if (pass2DumpPrefix)
                    p2rec(line, 8, xi, neighborFit);
            } else {
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

    static const int retractedSource = []{
        const char *s = std::getenv("LDCD_RETRACTED_SOURCE");
        if (!s)
            return 0;
        if (s[0] == 'n')
            return 1;
        if (s[0] == 'p')
            return 2;
        return 0;
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
    {
        static const char *depPath = std::getenv("LDCD_THEFT_DEP");
        if (depPath && !frameHasExactCoverage()) {
            using DepKey = std::pair<int, int>;
            static const std::map<DepKey, char> truthKeys = []{
                std::map<DepKey, char> s;
                const char *p = std::getenv("LDCD_THEFT_TRUTH");
                if (!p) return s;
                FILE *f = std::fopen(p, "rb");
                if (!f) return s;
                std::int32_t hdr[3];
                while (std::fread(hdr, sizeof(hdr), 1, f) == 1) {
                    if (std::fseek(f, (long)hdr[2] * sizeof(float),
                                   SEEK_CUR))
                        break;
                    s[{hdr[0], hdr[1]}] = 1;
                }
                std::fclose(f);
                return s;
            }();
            if (!truthKeys.empty()) {
                static std::mutex depMu;
                std::lock_guard<std::mutex> lk(depMu);
                if (FILE *f = std::fopen(depPath, "ab")) {
                    std::vector<float> row(width);
                    auto writeD = [&](const double *src) {
                        for (int xi = 0; xi < width; ++xi)
                            row[xi] = src ? (float)src[xi]
                                : std::numeric_limits<float>::quiet_NaN();
                        std::fwrite(row.data(), sizeof(float), width, f);
                    };
                    auto writeF = [&](const float *src) {
                        for (int xi = 0; xi < width; ++xi)
                            row[xi] = src ? src[xi]
                                : std::numeric_limits<float>::quiet_NaN();
                        std::fwrite(row.data(), sizeof(float), width, f);
                    };
                    for (int line = firstLine; line < lastLine; ++line) {
                        if (!truthKeys.count({(int)heldSeq1, line}))
                            continue;
                        const std::int32_t hdr[4] = {
                            (std::int32_t)heldSeq1, (std::int32_t)line,
                            (std::int32_t)width, 9 };
                        std::fwrite(hdr, sizeof(hdr), 1, f);
                        const float sc = (float)irescale;
                        std::fwrite(&sc, sizeof(float), 1, f);
                        const quint16 *raw = rawbuffer.data() +
                            static_cast<size_t>(line) *
                            videoParameters.fieldWidth;
                        for (int xi = 0; xi < width; ++xi)
                            row[xi] = (float)raw[left + xi];
                        std::fwrite(row.data(), sizeof(float), width, f);
                        writeD(locked1DRawBandpass_line(line));
                        writeD(bandWLaw_line(line));
                        writeD(bandKeep_line(line, 1));
                        writeF(carrierFit_flat.empty() ? nullptr
                            : carrierFit_flat.data() +
                              static_cast<size_t>(line) * demodWidth);
                        writeF(combedCarrier_flat.empty() ? nullptr
                            : combedCarrier_flat.data() +
                              static_cast<size_t>(line) * demodWidth);
                        for (int xi = 0; xi < width; ++xi)
                            row[xi] = (float)carrierSampleClass(
                                line, left + xi);
                        std::fwrite(row.data(), sizeof(float), width, f);
                        // Rows 8-9: the per-line burst phasor in the frame
                        // the demod LUTs actually use, so an offline grader
                        // can rotate lattice-frame phasors into the locked
                        // filter frame and name axes. Constant along the
                        // line; kept as rows so nPlanes stays the format.
                        const CombCarrierGrammar *bg = carrierGrammarLine(line);
                        const float bcv = bg && bg->grammarLocked
                            ? (float)bg->burstCos
                            : std::numeric_limits<float>::quiet_NaN();
                        const float bsv = bg && bg->grammarLocked
                            ? (float)bg->burstSin
                            : std::numeric_limits<float>::quiet_NaN();
                        std::fill(row.begin(), row.end(), bcv);
                        std::fwrite(row.data(), sizeof(float), width, f);
                        std::fill(row.begin(), row.end(), bsv);
                        std::fwrite(row.data(), sizeof(float), width, f);
                    }
                    std::fclose(f);
                }
            }
        }
    }

}

void Comb::FrameBuffer::buildCertifiedCarrierStage(const FrameBuffer *prevF)
{
    carrierRetractedValid = false;
    anchoredCarrierProvenance = AnchoredCarrierProvenance::None;
    anchored1DSource_flat.clear();
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

    auto resetStagePlane = [need](auto &plane, const auto &value) {
        if (plane.size() < need) plane.assign(need, value);
        else std::fill(plane.begin(), plane.end(), value);
    };

    resetStagePlane(carrierFit_flat, 0.0f);
    carrierFitLineValid.assign(static_cast<size_t>(lastLine), 0);
    if (configuration.lumaWitness) {
        resetStagePlane(carrierRetracted_flat, 0.0f);
    } else {
        carrierRetracted_flat.clear();
    }
    resetStagePlane(flatFloor_flat, 0.0f);
    resetStagePlane(combedCarrier_flat, 0.0f);
    resetStagePlane(carrierCorroboration_flat, 0.0f);
    resetStagePlane(carrierEligibility_flat, 0.0f);
    resetStagePlane(certRegistration_flat, kCertRegNone);
    resetStagePlane(coarseYEvidence_flat, lddecode::FourViewPixelEvidence{});
    if (carrierImpurity_flat.size() < need)
        carrierImpurity_flat.assign(need, 0.0f);

    if (configuration.lumaWitness && carrierAnalysis_flat.size() < need)
        return; // shared analysis must already have been produced
    if (!frameHasExactCoverage())
        return;

    const int retractedSource = ldcdRetractedSourceMode();

    std::vector<double> certComp(
        width, std::numeric_limits<double>::quiet_NaN());

    const bool phaseSnap = ldcdPhaseSnapOn();
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
        // The witness publishes its own view now. This loop keeps only
        // the certified half: anchoredRow, written from certifiedCarrier.
        float *retractedRow = nullptr;
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
                    constexpr double kRegIdentifyIRE = 1.0;
                    if (dev[kRegSearch] * invIreScale < kRegIdentifyIRE) {
                        reg[xi] = 0;
                        if (regRow) regRow[xi] = 0;
                        continue;
                    }
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

                std::vector<double> R(width,
                    std::numeric_limits<double>::quiet_NaN());
                for (int xi = 0; xi < width; ++xi) {
                    const int s = reg[xi];
                    const double lu = lumaAt(lU, xi - s);
                    const double ld = lumaAt(lD, xi + s);
                    if (!std::isfinite(lu) || !std::isfinite(ld)) continue;
                    R[xi] = (double)rawLine[left + xi] - 0.5 * (lu + ld);
                }
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

        }

        const double *obs1D = locked1DSource_line(line);

        const double *lawBpRow = nullptr, *lawWRow = nullptr,
                     *lawKRow = nullptr;
        if (retractedSource == 4) {
            lawBpRow = locked1DRawBandpass_line(line);
            lawWRow  = bandWLaw_line(line);
            lawKRow  = bandKeep_line(line, 2);
        }

        for (int xi = 0; xi < width; ++xi) {
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
                const bool haveFact =
                    std::isfinite(ex) || std::isfinite(certComp[xi]);
                if (retractedSource != 4 &&
                    !carrierFitLineSolved(line) && !haveFact) {
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
                case 4:
                    witnessCarrier = std::isfinite(ex)
                        ? static_cast<double>(ex)
                        : (std::isfinite(certComp[xi])
                            ? certComp[xi]
                            : ((lawBpRow && lawWRow && lawKRow)
                                ? lawBpRow[xi] * lawWRow[xi] * lawKRow[xi]
                                : 0.0));
                    break;
                case 3:
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

    {
        static const bool fitTheftOn = []{
            const char *e = std::getenv("LDCD_PROBE_FITTHEFT");
            return e && std::atoi(e) != 0;
        }();
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


    if (publishCertified)
        anchoredCarrierProvenance = AnchoredCarrierProvenance::FactBacked;
}
