/******************************************************************************
 * combcandidate.cpp
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
 *
 * Implements Comb::FrameBuffer::getCandidate(),
 * separated from comb.cpp so that candidate selection and 2D helpers
 * live in a single translation unit.
 ******************************************************************************/

#include "comb.h"
#include "combmath.h"

#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <array>
#include <limits>

// -------------------------------------------------------------------------

// 2D section

static inline int reflectCombRel(int rel, int width)
{
    if (width <= 0) return 0;
    if (rel < 0) return -rel;
    if (rel >= width) return (width - 1) - (rel - (width - 1));
    return rel;
}

static inline double combKMetric(double cc, double symC, double cn, double symN)
{
    // |comp| carries a 2fsc (2-sample period) Nyquist ripple from the chroma
    // subcarrier envelope: |comp[rel]| peaks anti-phase to symMag (the average
    // of the rel±1 magnitudes). Comparing the raw rectified samples therefore
    // injects that 2px ripple into the weight, modulating the vertical comb
    // gain and producing the bevel zipper. Summing |comp| with its anti-phase
    // neighbour-average yields a 2px-flat envelope magnitude (peak+trough ≈ A
    // at every sample), so the disagreement/credit metric below is computed on
    // a ripple-free magnitude.
    const double magC = std::fabs(cc) + symC;
    const double magN = std::fabs(cn) + symN;
    double k = std::fabs(magC - magN);
    k -= (magC + magN) * 0.10;
    return std::max(0.0, k);
}

static inline double combSmoothGate(double xIRE, double softIRE, double hardIRE)
{
    if (xIRE <= softIRE) return 1.0;
    if (xIRE >= hardIRE) return 0.0;
    double t = (xIRE - softIRE) / std::max(1e-9, hardIRE - softIRE);
    return 1.0 - std::clamp(t, 0.0, 1.0);
}

static inline double combSimilarityFactor(double sim, double start, double full)
{
    if (sim <= start) return 0.0;
    if (sim >= full) return 1.0;
    double t = (sim - start) / std::max(1e-9, full - start);
    return std::clamp(t, 0.0, 1.0);
}

void Comb::FrameBuffer::invalidateCombTapCache()
{
    tapLineCacheLine = { -1, -1, -1 };
    for (auto &tapLine : tapLineCache) {
        tapLine.cacheLine = -1;
        tapLine.builtFlags = 0;
    }
    // The memoised smoothed signed-IQ rows read the locked demod, which is
    // rebuilt per frame; clear their validity in lockstep with the tap cache.
    std::fill(smoothedLockedRowValid.begin(),
              smoothedLockedRowValid.end(),
              std::uint8_t{0});
}

// Fill the 7-tap smoothed signed-IQ row for `line` once per frame.  The
// balanced end-weighted aperture (0.5,1,1,1,1,1,0.5) equalises the two carrier
// axis sums (3:3) and keeps the vector phase-flat; the /3.0 normalisation is
// carried verbatim from the previous inline evaluator so the region verdicts
// are unchanged.
void Comb::FrameBuffer::ensureSmoothedLockedRow(int line)
{
    if (line < 0 || line >= demodLines)
        return;
    if ((int)smoothedLockedRowValid.size() != demodLines)
        return; // non-locked path: buffers not sized
    if (smoothedLockedRowValid[line])
        return;

    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;
    if (width <= 0)
        return;

    const float *iRow = locked1DTI4fsc_line(line);
    const float *qRow = locked1DTQ4fsc_line(line);
    float *sI = smoothedLockedTI_flat.data() + static_cast<size_t>(line) * demodWidth;
    float *sQ = smoothedLockedTQ_flat.data() + static_cast<size_t>(line) * demodWidth;

    static constexpr double w[7] = {0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5};
    for (int rel = 0; rel < width; ++rel) {
        double si = 0.0;
        double sq = 0.0;
        for (int k = -3; k <= 3; ++k) {
            const int rk = reflectCombRel(rel + k, width);
            si += w[k + 3] * static_cast<double>(iRow[rk]);
            sq += w[k + 3] * static_cast<double>(qRow[rk]);
        }
        sI[rel] = static_cast<float>(si / 3.0);
        sQ[rel] = static_cast<float>(sq / 3.0);
    }
    smoothedLockedRowValid[line] = std::uint8_t{1};
}

const Comb::FrameBuffer::CombTapLine &Comb::FrameBuffer::ensureCombTapLine(int lineNumber)
{
    const int slot = precleanRingSlot(lineNumber);
    CombTapLine &tapLine = tapLineCache[slot];
    if (tapLineCacheLine[slot] != lineNumber || tapLine.cacheLine != lineNumber) {
        tapLineCacheLine[slot] = lineNumber;
        tapLine.cacheLine = -1;
        tapLine.builtFlags = 0;
        buildCombTapLine(lineNumber, tapLine);
    }
    return tapLine;
}

void Comb::FrameBuffer::buildCombTapLine(int lineNumber, CombTapLine &tapLine)
{
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;

    tapLine.width = std::max(0, width);
    if (width <= 0 || lineNumber < first || lineNumber >= last) return;

    tapLine.cacheLine = lineNumber;
    tapLine.width = width;

    const unsigned flags = combTapBuildFlags_;
    const bool wantFieldA = (flags & TapBuildFieldA) != 0;
    const bool wantFieldB = (flags & TapBuildFieldB) != 0;
    const bool wantFrame  = (flags & TapBuildFrame)  != 0;

    // Contour is not only an intrafield-comb helper.  Frame B uses the same
    // same-column evidence as support for +/-1 interfield cancellation at
    // luma-contrast sites.
    const bool wantContour = wantFieldA || wantFieldB || wantFrame;

    auto clampSameFieldLine = [&](int ln)->int {
        const int parity = carrierLineParity(lineNumber);
        ln = std::clamp(ln, first, last - 1);
        if (carrierLineParity(ln) != parity) {
            if (ln + 1 < last && carrierLineParity(ln + 1) == parity) ln = ln + 1;
            else if (ln - 1 >= first && carrierLineParity(ln - 1) == parity) ln = ln - 1;
        }
        return ln;
    };

    tapLine.ln0  = lineNumber;
    tapLine.lnU1 = lineNumber - 1;
    tapLine.lnD1 = lineNumber + 1;
    tapLine.lnU2 = clampSameFieldLine(lineNumber - 2);
    tapLine.lnD2 = clampSameFieldLine(lineNumber + 2);
    tapLine.lnU4 = clampSameFieldLine(lineNumber - 4);
    tapLine.lnD4 = clampSameFieldLine(lineNumber + 4);

    tapLine.haveU1 = wantFrame  && (tapLine.lnU1 >= first && tapLine.lnU1 < last);
    tapLine.haveD1 = wantFrame  && (tapLine.lnD1 >= first && tapLine.lnD1 < last);
    tapLine.haveU2 = wantContour && (tapLine.lnU2 >= first && tapLine.lnU2 < last);
    tapLine.haveD2 = wantContour && (tapLine.lnD2 >= first && tapLine.lnD2 < last);
    tapLine.haveU4 = wantContour && (tapLine.lnU4 >= first && tapLine.lnU4 < last);
    tapLine.haveD4 = wantContour && (tapLine.lnD4 >= first && tapLine.lnD4 < last);

    auto ensureWidth = [&](auto &v) {
        if ((int)v.size() != width) v.resize(width);
    };

    ensureWidth(tapLine.tap0);

    if (wantFrame) {
        ensureWidth(tapLine.tapU1);
        ensureWidth(tapLine.tapD1);
        ensureWidth(tapLine.pairU1);
        ensureWidth(tapLine.pairD1);
    }

    if (wantContour) {
        ensureWidth(tapLine.tapU2);
        ensureWidth(tapLine.tapD2);
        ensureWidth(tapLine.pairU2);
        ensureWidth(tapLine.pairD2);
        ensureWidth(tapLine.intrafieldRegionReach);
        ensureWidth(tapLine.intrafieldRegionCede);
        ensureWidth(tapLine.regionUp4);
        ensureWidth(tapLine.regionDown4);
    }

    if (wantContour) {
        ensureWidth(tapLine.tapU4);
        ensureWidth(tapLine.tapD4);
        ensureWidth(tapLine.contour);
        ensureWidth(tapLine.movingCoarseContour);
        ensureWidth(tapLine.coarse0IRE);
        ensureWidth(tapLine.coarseU2IRE);
        ensureWidth(tapLine.coarseD2IRE);
    }

    if (wantFieldB || wantFrame) {
        ensureWidth(tapLine.hLumaDeltaIRE);
    }

    auto getCompRow = [&](int ln)->const double* {
        if (ln < first || ln >= last) return nullptr;
        if (configuration.phaseCompensation)
            return locked1DSource_line(ln);
        return bucketScalar1D_line(ln) + left;
    };

    struct RowRefs {
        int ln = -1;
        const double *comp = nullptr;
        bool haveLine = false;
    };

    auto rowRefs = [&](int ln, bool haveLine)->RowRefs {
        RowRefs r;
        r.ln = ln;
        r.haveLine = haveLine;
        r.comp = haveLine ? getCompRow(ln) : nullptr;
        return r;
    };

    RowRefs r0  = rowRefs(tapLine.ln0,  true);
    RowRefs rU1 = rowRefs(tapLine.lnU1, tapLine.haveU1);
    RowRefs rD1 = rowRefs(tapLine.lnD1, tapLine.haveD1);
    RowRefs rU2 = rowRefs(tapLine.lnU2, tapLine.haveU2);
    RowRefs rD2 = rowRefs(tapLine.lnD2, tapLine.haveD2);
    RowRefs rU4 = rowRefs(tapLine.lnU4, tapLine.haveU4);
    RowRefs rD4 = rowRefs(tapLine.lnD4, tapLine.haveD4);

    auto getLumaRow = [&](int ln)->const double* {
        if (!configuration.phaseCompensation ||
            !lockedLumaCacheValid ||
            demodWidth < width ||
            lockedLumaSmooth_flat.empty() ||
            ln < 0 || ln >= demodLines)
        {
            return nullptr;
        }
        return lockedLumaSmooth_line(ln);
    };

    const double *luma0  = getLumaRow(tapLine.ln0);
    const double *lumaU2 = getLumaRow(tapLine.lnU2);
    const double *lumaD2 = getLumaRow(tapLine.lnD2);
    const double *lumaU4 = getLumaRow(tapLine.lnU4);
    const double *lumaD4 = getLumaRow(tapLine.lnD4);

    auto fillTap = [&](const RowRefs &r,
                       std::vector<CombTapScalar> &dst) {
        if (r.comp == nullptr) {
            for (int rel = 0; rel < width; ++rel)
                dst[rel] = CombTapScalar();
            return;
        }

        for (int rel = 0; rel < width; ++rel) {
            const int rm1 = reflectCombRel(rel - 1, width);
            const int rp1 = reflectCombRel(rel + 1, width);

            CombTapScalar &s = dst[rel];
            s.comp = r.comp[rel];
            s.symMag = 0.5 * (std::fabs(r.comp[rm1]) + std::fabs(r.comp[rp1]));
        }
    };

    {
        fillTap(r0, tapLine.tap0);

        if (wantFrame) {
            fillTap(rU1, tapLine.tapU1);
            fillTap(rD1, tapLine.tapD1);
        }

        if (wantContour) {
            fillTap(rU2, tapLine.tapU2);
            fillTap(rD2, tapLine.tapD2);
            fillTap(rU4, tapLine.tapU4);
            fillTap(rD4, tapLine.tapD4);
        }
    }

    const auto &T = configuration.tunables;
    const double kRange = T.FIELD_K_RANGE_IRE * irescale;
    const double invK = (kRange > 1e-9) ? (1.0 / kRange) : 0.0;
    const double invI = invIreScale;
    const bool wantCenterEnvelope = wantFieldA || wantFieldB || wantFrame;
    if (wantCenterEnvelope) {
        ensureWidth(tapLine.centerEnvelope);
        ensureWidth(tapLine.centerAdmittedChromaT);
        const auto *centerAnalysis = carrierAnalysis_line(tapLine.ln0);
        for (int rel = 0; rel < width; ++rel) {
            const CombTapScalar &s = tapLine.tap0[rel];
            const double envC = std::hypot(s.comp, s.symMag);
            tapLine.centerEnvelope[rel] = envC;
            const double carrierTrust = centerAnalysis
                ? lddecode::carrierTrust(
                    centerAnalysis[rel].carrierConformance,
                    centerAnalysis[rel].conformanceConfidence)
                : 0.5;
            const double admission =
                std::clamp(2.0 * (carrierTrust - 0.5), 0.0, 1.0);
            tapLine.centerAdmittedChromaT[rel] = admission *
                std::clamp((envC * invI - 2.0) / 8.0, 0.0, 1.0);
        }
    }

    const lddecode::CombReachSourceFrame scalarSource = scalarReachSource();
    const lddecode::CombReachSourceFrame iqSource = iqReachSource();

    auto legalGateForReachUse = [](const lddecode::CombReachReply &reach,
                                   lddecode::CombReachUse reachUse)->double {
        switch (reachUse) {
        case lddecode::CombReachUse::FieldScalarAverage:
        case lddecode::CombReachUse::FrameScalarAverage:
            return reach.allowScalarAverage ? 1.0 : 0.0;

        case lddecode::CombReachUse::FieldScalarCancel:
        case lddecode::CombReachUse::FrameScalarCancel:
            return reach.allowScalarCancel ? 1.0 : 0.0;

        case lddecode::CombReachUse::ScalarSignCompare:
            return reach.allowScalarSignCompare ? 1.0 : 0.0;

        case lddecode::CombReachUse::ScalarMagnitudeCompare:
            return reach.allowScalarMagnitudeCompare ? 1.0 : 0.0;

        case lddecode::CombReachUse::IQCompare:
            return reach.allowIQCompare ? 1.0 : 0.0;

        case lddecode::CombReachUse::IQAverage:
            return reach.allowIQAverage ? 1.0 : 0.0;

        case lddecode::CombReachUse::IQCancel:
            return reach.allowIQCancel ? 1.0 : 0.0;
        }

        return 0.0;
    };

    // carrierGrammarSignedPhaseRelation uses (h + phase0 + flip) & 3 for both
    // center and target with the same h, so h cancels — the reply is identical
    // for every pixel on this line pair. Hoisted once per call.
    auto fillPair = [&](const std::vector<CombTapScalar> &nbr,
                        int targetLine,
                        bool haveNbr,
                        std::vector<CombTapPair> &dst,
                        const lddecode::CombReachSourceFrame &reachSource,
                        lddecode::CombReachUse reachUse) {
        if (!haveNbr) {
            for (int rel = 0; rel < width; ++rel)
                dst[rel] = CombTapPair();
            return;
        }

        const double lineReachLegalGate = legalGateForReachUse(
            combReachIndex.query({lineNumber, targetLine, left, left, reachUse, reachSource}),
            reachUse);

        const bool needScalarWeight =
            (reachUse != lddecode::CombReachUse::IQCancel);

        if (!needScalarWeight) {
            for (int rel = 0; rel < width; ++rel) {
                CombTapPair &p = dst[rel];
                p = CombTapPair();
                p.reachLegalGate = lineReachLegalGate;
            }
            return;
        }

        for (int rel = 0; rel < width; ++rel) {
            CombTapPair &p = dst[rel];
            p = CombTapPair();
            p.reachLegalGate = lineReachLegalGate;

            const CombTapScalar &c = tapLine.tap0[rel];
            const CombTapScalar &n = nbr[rel];
            p.diffIRE = std::fabs(c.comp - n.comp) * invI;
            p.kScore = combKMetric(c.comp, c.symMag, n.comp, n.symMag);
            p.weight = (kRange > 1e-9) ? (1.0 - p.kScore * invK) : 1.0;
            p.weight = std::clamp(p.weight, 0.0, 1.0);
        }
    };

    {
        if (wantFrame) {
            fillPair(tapLine.tapU1,
                     tapLine.lnU1,
                     tapLine.haveU1,
                     tapLine.pairU1,
                     iqSource,
                     lddecode::CombReachUse::IQCancel);

            fillPair(tapLine.tapD1,
                     tapLine.lnD1,
                     tapLine.haveD1,
                     tapLine.pairD1,
                     iqSource,
                     lddecode::CombReachUse::IQCancel);
        }
    }

    {
        if (wantContour) {
            fillPair(tapLine.tapU2,
                     tapLine.lnU2,
                     tapLine.haveU2,
                     tapLine.pairU2,
                     scalarSource,
                     lddecode::CombReachUse::FieldScalarCancel);

            fillPair(tapLine.tapD2,
                     tapLine.lnD2,
                     tapLine.haveD2,
                     tapLine.pairD2,
                     scalarSource,
                     lddecode::CombReachUse::FieldScalarCancel);
        }
    }

    if (wantContour) {
        std::fill(tapLine.intrafieldRegionReach.begin(),
                  tapLine.intrafieldRegionReach.end(),
                  CombContentReach::IntrafieldRegionReach{});
        std::fill(tapLine.intrafieldRegionCede.begin(),
                  tapLine.intrafieldRegionCede.end(),
                  std::uint8_t{0});
        std::fill(tapLine.regionUp4.begin(),
                  tapLine.regionUp4.end(),
                  CombContentReach::RegionRelation::Unknown);
        std::fill(tapLine.regionDown4.begin(),
                  tapLine.regionDown4.end(),
                  CombContentReach::RegionRelation::Unknown);

        const size_t iqCount =
            static_cast<size_t>(demodLines) * static_cast<size_t>(demodWidth);
        const bool haveSignedIQ =
            configuration.phaseCompensation &&
            demodWidth >= width &&
            tapLine.ln0 >= 0 && tapLine.ln0 < demodLines &&
            tapLine.lnU2 >= 0 && tapLine.lnU2 < demodLines &&
            tapLine.lnD2 >= 0 && tapLine.lnD2 < demodLines &&
            locked1DTI4fsc_flat.size() >= iqCount &&
            locked1DTQ4fsc_flat.size() >= iqCount &&
            carrierAnalysis_flat.size() >= iqCount;

        if (haveSignedIQ) {
            // ±2 grammar reach (hoisted per line).
            const lddecode::CombReachReply upReach = combReachIndex.query(
                {lineNumber, tapLine.lnU2, left, left,
                 lddecode::CombReachUse::IQCompare, iqSource});
            const lddecode::CombReachReply downReach = combReachIndex.query(
                {lineNumber, tapLine.lnD2, left, left,
                 lddecode::CombReachUse::IQCompare, iqSource});

            // Balanced 7-tap horizontal aggregate, memoised per line.  Even
            // offsets carry one carrier axis and odd offsets the other, so the
            // 0.5 end weights equalize the axis sums (3:3) and keep the vector
            // phase-flat.  The narrow 3-tap vector was noise-limited in
            // low-saturation skin (hue sigma ~25 deg), so the verdicts
            // flickered at pixel pitch and one-sided combing toggled column to
            // column — the beaded fringe on garment edges.  Each line's
            // smoothed row is computed once (ensureSmoothedLockedRow) and
            // reused by every center that references it as a ±2/±4 partner.
            ensureSmoothedLockedRow(tapLine.ln0);
            ensureSmoothedLockedRow(tapLine.lnU2);
            ensureSmoothedLockedRow(tapLine.lnD2);
            const float *sI0  = smoothedLockedTI_line(tapLine.ln0);
            const float *sQ0  = smoothedLockedTQ_line(tapLine.ln0);
            const float *sIUp = smoothedLockedTI_line(tapLine.lnU2);
            const float *sQUp = smoothedLockedTQ_line(tapLine.lnU2);
            const float *sIDn = smoothedLockedTI_line(tapLine.lnD2);
            const float *sQDn = smoothedLockedTQ_line(tapLine.lnD2);
            const auto *analysis0 = carrierAnalysis_line(tapLine.ln0);
            const auto *analysisUp = carrierAnalysis_line(tapLine.lnU2);
            const auto *analysisDn = carrierAnalysis_line(tapLine.lnD2);
            auto trustAt = [](const lddecode::CarrierAnalysisRecord *row,
                              int rel) {
                return row
                    ? lddecode::carrierTrust(row[rel].carrierConformance,
                                             row[rel].conformanceConfidence)
                    : 0.5;
            };

            // ±4 grammar reach and smoothed rows for contour-influence gating.
            const bool have4IQ =
                tapLine.lnU4 >= 0 && tapLine.lnU4 < demodLines &&
                tapLine.lnD4 >= 0 && tapLine.lnD4 < demodLines;
            lddecode::CombReachReply up4Reach, dn4Reach;
            bool haveUp4 = false, haveDn4 = false;
            const float *sIUp4 = nullptr, *sQUp4 = nullptr;
            const float *sIDn4 = nullptr, *sQDn4 = nullptr;
            const lddecode::CarrierAnalysisRecord *analysisUp4 = nullptr;
            const lddecode::CarrierAnalysisRecord *analysisDn4 = nullptr;
            if (have4IQ && tapLine.haveU4) {
                up4Reach = combReachIndex.query(
                    {lineNumber, tapLine.lnU4, left, left,
                     lddecode::CombReachUse::IQCompare, iqSource});
                if (up4Reach.allowIQCompare) {
                    ensureSmoothedLockedRow(tapLine.lnU4);
                    sIUp4 = smoothedLockedTI_line(tapLine.lnU4);
                    sQUp4 = smoothedLockedTQ_line(tapLine.lnU4);
                    analysisUp4 = carrierAnalysis_line(tapLine.lnU4);
                    haveUp4 = true;
                }
            }
            if (have4IQ && tapLine.haveD4) {
                dn4Reach = combReachIndex.query(
                    {lineNumber, tapLine.lnD4, left, left,
                     lddecode::CombReachUse::IQCompare, iqSource});
                if (dn4Reach.allowIQCompare) {
                    ensureSmoothedLockedRow(tapLine.lnD4);
                    sIDn4 = smoothedLockedTI_line(tapLine.lnD4);
                    sQDn4 = smoothedLockedTQ_line(tapLine.lnD4);
                    analysisDn4 = carrierAnalysis_line(tapLine.lnD4);
                    haveDn4 = true;
                }
            }
            const bool want4Region = (haveUp4 || haveDn4);

            for (int rel = 0; rel < width; ++rel) {
                const std::complex<double> z0(sI0[rel], sQ0[rel]);

                tapLine.intrafieldRegionReach[rel] =
                    CombContentReach::evaluateIntrafieldRegionReach(
                        z0,
                        std::complex<double>(sIUp[rel], sQUp[rel]),
                        std::complex<double>(sIDn[rel], sQDn[rel]),
                        upReach.carrierRelation,
                        downReach.carrierRelation,
                        upReach.allowIQCompare,
                        downReach.allowIQCompare,
                        trustAt(analysis0, rel),
                        trustAt(analysisUp, rel),
                        trustAt(analysisDn, rel),
                        invI,
                        5.0);

                if (want4Region) {
                    const auto region4 =
                        CombContentReach::evaluateIntrafieldRegionReach(
                            z0,
                            haveUp4
                                ? std::complex<double>(sIUp4[rel], sQUp4[rel])
                                : std::complex<double>(0.0, 0.0),
                            haveDn4
                                ? std::complex<double>(sIDn4[rel], sQDn4[rel])
                                : std::complex<double>(0.0, 0.0),
                            up4Reach.carrierRelation,
                            dn4Reach.carrierRelation,
                            haveUp4,
                            haveDn4,
                            trustAt(analysis0, rel),
                            trustAt(analysisUp4, rel),
                            trustAt(analysisDn4, rel),
                            invI,
                            5.0);
                    tapLine.regionUp4[rel] = region4.up;
                    tapLine.regionDown4[rel] = region4.down;
                }
            }

            // Raw no-valid-partner cede flag: a Different leg with no
            // positively same-region partner.  Consumers OR a small
            // horizontal window over this so three-region cedes (drop
            // shadows) hold a uniform height across columns.
            for (int rel = 0; rel < width; ++rel) {
                const auto &region = tapLine.intrafieldRegionReach[rel];
                if (!region.valid)
                    continue;
                const bool upDifferent = region.up ==
                    CombContentReach::RegionRelation::DifferentRegion;
                const bool downDifferent = region.down ==
                    CombContentReach::RegionRelation::DifferentRegion;
                const bool upSame = region.up ==
                    CombContentReach::RegionRelation::SameRegion;
                const bool downSame = region.down ==
                    CombContentReach::RegionRelation::SameRegion;
                std::uint8_t flag = 0;
                if ((upDifferent && !downSame) || (downDifferent && !upSame))
                    flag |= 1;                       // bit0: cede
                if (region.strongAsym)
                    flag |= 3;                       // bit1: shadow band —
                                                     // membership outranks a
                                                     // Same partner, so the
                                                     // whole band takes ONE
                                                     // render (the 1D), never
                                                     // a comb/cede interleave
                tapLine.intrafieldRegionCede[rel] = flag;
            }
        }
    }

    auto sampleTapComp = [&](const std::vector<CombTapScalar> &tap, int rel)->double {
        rel = std::clamp(rel, 0, width - 1);
        return tap[rel].comp;
    };

    auto notchTap = [&](const std::vector<CombTapScalar> &tap, int rel)->double {
        if (rel < 2) rel = 2;
        if (rel > width - 3) rel = width - 3;
        return 0.5 * (sampleTapComp(tap, rel - 2) + sampleTapComp(tap, rel + 2));
    };

    {
    if (wantFieldB || wantFrame) {
        for (int rel = 0; rel < width; ++rel) {
            if (configuration.phaseCompensation) {
                if (luma0 && width >= 5) {
                    // Carrier-free smoothed luma, already hoisted for the
                    // coarse rows.  The notch pair this replaces averaged
                    // rel-2/rel+2 composite samples — four apart, SAME
                    // carrier phase — so in saturated color it passed
                    // full-amplitude quadrature carrier into this "luma"
                    // delta and every downstream hEdge gate alternated at
                    // carrier pitch.  Composite level is never a luma
                    // witness in high color.
                    const int rm = std::clamp(rel - 2, 0, width - 1);
                    const int rp = std::clamp(rel + 2, 0, width - 1);
                    tapLine.hLumaDeltaIRE[rel] =
                        std::fabs(luma0[rp] - luma0[rm]) * invI;
                } else if (width >= 5) {
                    const double lumL = notchTap(tapLine.tap0, rel - 1);
                    const double lumR = notchTap(tapLine.tap0, rel + 1);
                    tapLine.hLumaDeltaIRE[rel] = std::fabs(lumR - lumL) * invI;
                } else {
                    tapLine.hLumaDeltaIRE[rel] = 0.0;
                }
            } else {
                const int rm1 = std::clamp(rel - 1, 0, width - 1);
                const int rp1 = std::clamp(rel + 1, 0, width - 1);
                tapLine.hLumaDeltaIRE[rel] =
                    std::fabs(tapLine.tap0[rp1].comp - tapLine.tap0[rm1].comp) * invI;
            }

        }
    }
    }

    // There is deliberately no amplitude/compactness-based "chroma shape"
    // exception here.  High-frequency energy defaults to luma.  Physical
    // appearance (large envelope, narrow span, isolation, proximity to a luma
    // edge) may limit a reach that already has carrier authority, but it may
    // never register that energy as carrier or grant an escape from the comb.
    // Carrier privileges come from the grammar/conformance tables and their
    // named consumers.  The retired irrationalChroma path violated this rule
    // by promoting exactly the fine luma structures the decoder must protect.

    {
    if (wantContour) {
        // Hoist tunables and data() pointers out of the hot loop.  Reading 11
        // vectors per pixel via std::vector<T>::operator[] generates redundant
        // address math; raw pointers + cached tunables let the loop become a
        // tight straight-line block.
        const double soft = T.FIELD_CONTOUR_SOFT_IRE;
        const double hard = T.FIELD_CONTOUR_HARD_IRE;
        const double simStart = T.FIELD_CONTOUR_SIM_START;
        const double simFull  = T.FIELD_CONTOUR_SIM_FULL;
        const double farInf   = T.FIELD_CONTOUR_FAR_INFLUENCE;
        const bool kValid = (kRange > 1e-9);
        const bool hasLuma = (luma0 && lumaU2 && lumaD2 && lumaU4 && lumaD4);
        const bool haveU2 = tapLine.haveU2;
        const bool haveD2 = tapLine.haveD2;
        const bool haveU4 = tapLine.haveU4;
        const bool haveD4 = tapLine.haveD4;

        const CombTapScalar *t0  = tapLine.tap0.data();
        const CombTapScalar *tU2 = tapLine.tapU2.data();
        const CombTapScalar *tD2 = tapLine.tapD2.data();
        const CombTapScalar *tU4 = tapLine.tapU4.data();
        const CombTapScalar *tD4 = tapLine.tapD4.data();
        const CombTapPair   *pU2 = tapLine.pairU2.data();
        const CombTapPair   *pD2 = tapLine.pairD2.data();
        CombTapContour      *outContour = tapLine.contour.data();
        CombContentReach::MovingCoarseContour *outMCC = tapLine.movingCoarseContour.data();
        double *outCoarse0 = tapLine.coarse0IRE.data();
        double *outCoarseU2 = tapLine.coarseU2IRE.data();
        double *outCoarseD2 = tapLine.coarseD2IRE.data();

        for (int rel = 0; rel < width; ++rel) {
            const CombTapScalar &sC  = t0[rel];
            const CombTapScalar &sU2 = tU2[rel];
            const CombTapScalar &sD2 = tD2[rel];
            const CombTapScalar &sU4 = tU4[rel];
            const CombTapScalar &sD4 = tD4[rel];

            const double aC  = std::fabs(sC.comp);
            const double aU2 = std::fabs(sU2.comp);
            const double aD2 = std::fabs(sD2.comp);
            const double aU4 = std::fabs(sU4.comp);
            const double aD4 = std::fabs(sD4.comp);

            CombTapContour c;
            c.curvMidIRE = std::fabs(aU2 - 2.0 * aC + aD2) * invI;
            c.midOk = combSmoothGate(c.curvMidIRE, soft, hard);

            const double u4Pred = 2.0 * aU2 - aC;
            const double d4Pred = 2.0 * aD2 - aC;

            c.upResIRE = std::fabs(aU4 - u4Pred) * invI;
            c.dnResIRE = std::fabs(aD4 - d4Pred) * invI;
            c.upSideOk = combSmoothGate(c.upResIRE, soft, hard);
            c.dnSideOk = combSmoothGate(c.dnResIRE, soft, hard);

            const double upK = combKMetric(sU2.comp, sU2.symMag, sU4.comp, sU4.symMag);
            const double dnK = combKMetric(sD2.comp, sD2.symMag, sD4.comp, sD4.symMag);

            c.upSim = (pU2[rel].weight > 0.0)
                ? std::clamp(kValid ? (1.0 - upK * invK) : 1.0, 0.0, 1.0)
                : 0.0;
            c.dnSim = (pD2[rel].weight > 0.0)
                ? std::clamp(kValid ? (1.0 - dnK * invK) : 1.0, 0.0, 1.0)
                : 0.0;

            c.upTrust = c.midOk * c.upSideOk;
            c.dnTrust = c.midOk * c.dnSideOk;

            c.upInfluence = farInf * c.upTrust *
                            combSimilarityFactor(c.upSim, simStart, simFull);
            c.dnInfluence = farInf * c.dnTrust *
                            combSimilarityFactor(c.dnSim, simStart, simFull);

            if (tapLine.regionUp4[rel] == CombContentReach::RegionRelation::DifferentRegion)
                c.upInfluence = 0.0;
            if (tapLine.regionDown4[rel] == CombContentReach::RegionRelation::DifferentRegion)
                c.dnInfluence = 0.0;

            outContour[rel] = c;

            outCoarse0[rel] = luma0 ? (luma0[rel] * invI) : 0.0;
            outCoarseU2[rel] = lumaU2 ? (lumaU2[rel] * invI) : outCoarse0[rel];
            outCoarseD2[rel] = lumaD2 ? (lumaD2[rel] * invI) : outCoarse0[rel];

            outMCC[rel] = hasLuma
                ? CombContentReach::evaluateMovingCoarseContour(
                    luma0[rel] * invI,
                    lumaU2[rel] * invI,
                    lumaD2[rel] * invI,
                    lumaU4[rel] * invI,
                    lumaD4[rel] * invI,
                    haveU2, haveD2, haveU4, haveD4,
                    soft, hard)
                : CombContentReach::MovingCoarseContour();
        }
    }
    }

    // reachGate composition policy:
    //
    //   Frame ±1 (Frame B):  reachGate = bevelGate * max(reachLegalGate, interfieldIQReachFloor).
    //     Frame B is deliberately the over-combing path.  Legality establishes
    //     the baseline; the IQ floor only RAISES reach at counterpart sites
    //     where the IQ evidence shows the center is alien chroma — on real
    //     vertical chroma columns oppositeIQFit collapses to 0 and the floor
    //     adds nothing.  bevelGate then throttles reach where the close-
    //     focused (±1/±2 luma) coarse contour shows a curvature break AND
    //     chroma is present — the ±1 zipper guard for bevels.  Low-chroma
    //     regions are essentially inert (chromaWeight → 0).
    //
    //   Field ±2 (Field A / Field B / preclean):  reachGate = mc * reachLegalGate.
    //     The intrafield combs need actual reach limiting around vertical
    //     content boundaries.  Source: movingCoarseContour exclusively.  When
    //     mc.valid -> 0.25 + 0.75 * trust; otherwise gate = 1.0 (deterministic
    //     fall-through, not a fallback to a second confidence stream).
    auto applyFrameReachWithIQFloor = [&](std::vector<CombTapPair> &upPair,
                                        std::vector<CombTapPair> &dnPair,
                                        bool haveUp,
                                        bool haveDn) {
        if (!haveUp || !haveDn || (int)upPair.size() < width || (int)dnPair.size() < width)
            return;

        // Demod the ±1 comp taps to Grid4fscIQ on the fly so we can layer
        // interfieldIQReachFloor on top of legality.  The floor only RAISES
        // reachGate at sites where the IQ evidence shows center is alien
        // chroma (anti-phase counterpart) or displaced from neighbor-common;
        // on real vertical chroma columns oppositeIQFit collapses to 0 and
        // the floor adds nothing.  columnSupport = 1.0 by design: saturated-
        // area checkers do not sit on horizontal luma edges and the floor's
        // internal gates already discriminate.  See
        // project_frameb_comb_must_run.
        const double minChromaIRE = std::max(0.0, T.FRAME_B_CHROMA_MIN_IRE);
        auto phaseCursor = [&](int ln) {
            return carrierGrammarSignedSampleCursor(
                configuration.phaseCompensation ? carrierGrammarLine(ln) : nullptr,
                left);
        };
        auto phase0Cursor = phaseCursor(tapLine.ln0);
        auto phaseUCursor = phaseCursor(tapLine.lnU1);
        auto phaseDCursor = phaseCursor(tapLine.lnD1);

        // Close-focused coarse contour for the ±1 bevel zipper guard.
        // movingCoarseContour evaluates straightness over ±2/±4; reuse the
        // same evaluator with ±1/±2 luma so the test fires on the actual
        // partners Frame B reaches into.  Chroma magnitude weights the
        // penalty: low-chroma areas don't zipper regardless, so the throttle
        // is essentially inert there.
        const double *frameLuma0  = (lockedLumaCacheValid && !lockedLumaSmooth_flat.empty() && demodWidth >= width)
                                    ? lockedLumaSmooth_line(tapLine.ln0)  : nullptr;
        const double *frameLumaU1 = (frameLuma0 && tapLine.lnU1 >= 0 && tapLine.lnU1 < demodLines)
                                    ? lockedLumaSmooth_line(tapLine.lnU1) : nullptr;
        const double *frameLumaD1 = (frameLuma0 && tapLine.lnD1 >= 0 && tapLine.lnD1 < demodLines)
                                    ? lockedLumaSmooth_line(tapLine.lnD1) : nullptr;
        const double *frameLumaU2 = (frameLuma0 && tapLine.lnU2 >= 0 && tapLine.lnU2 < demodLines)
                                    ? lockedLumaSmooth_line(tapLine.lnU2) : nullptr;
        const double *frameLumaD2 = (frameLuma0 && tapLine.lnD2 >= 0 && tapLine.lnD2 < demodLines)
                                    ? lockedLumaSmooth_line(tapLine.lnD2) : nullptr;
        const bool haveCloseLuma = frameLuma0 && frameLumaU1 && frameLumaD1
                                   && frameLumaU2 && frameLumaD2;
        const double soft = T.FIELD_CONTOUR_SOFT_IRE;
        const double hard = T.FIELD_CONTOUR_HARD_IRE;
        const double bevelPenalty = std::clamp(T.FRAME_B_BEVEL_REACH_PENALTY, 0.0, 1.0);
        const double satPenalty  = std::clamp(T.FRAME_BEVEL_SAT_PENALTY, 0.0, 1.0);
        const double xColPenalty = std::clamp(T.FRAME_BEVEL_XCOL_PENALTY, 0.0, 1.0);
        const double xColThreshIRE = std::max(1.0, T.FRAME_LUMA_EDGE_THRESH_IRE);

        if ((int)scratch_impulseExempt.size() < width)
            scratch_impulseExempt.resize(width);
        std::fill(scratch_impulseExempt.begin(),
                  scratch_impulseExempt.begin() + width, 0.0);

        for (int rel = 0; rel < width; ++rel) {
            double cI, cQ, uI, uQ, dI, dQ;
            carrierGrammarDemodSignedCompositeTo4fsc(
                phase0Cursor, tapLine.tap0[rel].comp, cI, cQ);
            carrierGrammarDemodSignedCompositeTo4fsc(
                phaseUCursor, tapLine.tapU1[rel].comp, uI, uQ);
            carrierGrammarDemodSignedCompositeTo4fsc(
                phaseDCursor, tapLine.tapD1[rel].comp, dI, dQ);

            // Frame B's reach-floor inputs are IRE-domain signed IQ to match
            // the reach-floor contract (IRE thresholds in, IRE thresholds out).
            const double cIf = cI * invI;
            const double cQf = cQ * invI;
            const double uIf = uI * invI;
            const double uQf = uQ * invI;
            const double dIf = dI * invI;
            const double dQf = dQ * invI;
            // Fast overload: supplying pre-computed magnitudes saves ~12
            // hypots/pixel that the slow overload recomputes internally.
            // Direct sqrt, not std::hypot: bounded IRE-domain IQ, no overflow
            // risk, so hypot's IEEE guarding is pure cost in this per-pixel loop.
            const double cMag = std::sqrt(cIf * cIf + cQf * cQf);
            const double uMag = std::sqrt(uIf * uIf + uQf * uQf);
            const double dMag = std::sqrt(dIf * dIf + dQf * dQf);
            const CombContentReach::InterfieldIQReachFloor floor =
                CombContentReach::interfieldIQReachFloor(
                    cIf, cQf, uIf, uQf, dIf, dQf,
                    true, true, minChromaIRE, 1.0,
                    cMag, uMag, dMag);

            double upGate = std::max(upPair[rel].reachLegalGate, floor.up);
            double dnGate = std::max(dnPair[rel].reachLegalGate, floor.down);

            double impulseExempt = 0.0;
            if (frameLuma0) {
                // Baseline FVF impulse exemption from the long-standing path.
                const int l2 = std::max(0, rel - 2);
                const int r2 = std::min(width - 1, rel + 2);
                const double cIRE_luma = frameLuma0[rel] * invI;
                const double lIRE_luma = frameLuma0[l2] * invI;
                const double rIRE_luma = frameLuma0[r2] * invI;
                const double surround = 0.5 * (lIRE_luma + rIRE_luma);
                const double peakIRE = std::fabs(cIRE_luma - surround);
                const double flanksAgree = std::fabs(lIRE_luma - rIRE_luma);
                constexpr double IMPULSE_LO_IRE = 5.0;
                constexpr double IMPULSE_HI_IRE = 15.0;
                if (flanksAgree < 4.0) {
                    impulseExempt = std::clamp(
                        (peakIRE - IMPULSE_LO_IRE) /
                        (IMPULSE_HI_IRE - IMPULSE_LO_IRE),
                        0.0, 1.0);
                }
            }
            scratch_impulseExempt[rel] = impulseExempt;

            if (haveCloseLuma && bevelPenalty > 0.0) {
                const auto mcNear = CombContentReach::evaluateMovingCoarseContour(
                    frameLuma0[rel]  * invI,
                    frameLumaU1[rel] * invI, frameLumaD1[rel] * invI,
                    frameLumaU2[rel] * invI, frameLumaD2[rel] * invI,
                    true, true, true, true,
                    soft, hard);

                if (mcNear.valid) {
	                    const double chromaWeight =
	                        (rel < (int)tapLine.centerAdmittedChromaT.size())
	                            ? tapLine.centerAdmittedChromaT[rel]
	                            : 0.0;
                    const double curvature =
                        1.0 - std::clamp(mcNear.straightness, 0.0, 1.0);
                    double bevelGate =
                        std::clamp(1.0 - bevelPenalty * chromaWeight * curvature,
                                   0.0, 1.0);
                    // Saturation penalty: the linear bevel gate doesn't pull
                    // hard enough at intermediate chroma because chromaWeight
                    // ramps slowly (10 IRE to saturate).  Squaring chroma and
                    // curvature makes the penalty bite harder where it matters
                    // — high saturation on a non-straight edge — without
                    // touching low-chroma or straight regions.
                    if (satPenalty > 0.0) {
                        const double satBite = satPenalty *
                            chromaWeight * chromaWeight *
                            curvature * curvature;
                        bevelGate *= std::clamp(1.0 - satBite, 0.0, 1.0);
                    }
                    if (xColPenalty > 0.0 &&
                        rel < (int)tapLine.hLumaDeltaIRE.size()) {
                        const double hEdge = std::clamp(
                            (tapLine.hLumaDeltaIRE[rel] - 0.30 * xColThreshIRE) /
                            (0.70 * xColThreshIRE),
                            0.0, 1.0);
                        const double xColorRisk =
                            chromaWeight * hEdge * curvature;
                        bevelGate *= std::clamp(
                            1.0 - xColPenalty * xColorRisk, 0.0, 1.0);
                    }
                    const double frameExempt = impulseExempt;
                    const double effectiveGate =
                        bevelGate + (1.0 - bevelGate) * frameExempt;
                    upGate *= effectiveGate;
                    dnGate *= effectiveGate;
                }
            }

            upPair[rel].reachGate = std::clamp(upGate, 0.0, 1.0);
            dnPair[rel].reachGate = std::clamp(dnGate, 0.0, 1.0);
        }
    };

    auto applyFieldReachWithMovingCoarse = [&](std::vector<CombTapPair> &upPair,
                                               std::vector<CombTapPair> &dnPair,
                                               bool haveUp,
                                               bool haveDn) {
        if (!haveUp || !haveDn || (int)upPair.size() < width || (int)dnPair.size() < width)
            return;
        const double edgeThreshIRE = std::max(1.0, T.FIELD_LUMA_EDGE_THRESH_IRE);
        for (int rel = 0; rel < width; ++rel) {
            double upContourGate = 1.0;
            double dnContourGate = 1.0;
            if (rel < (int)tapLine.movingCoarseContour.size() &&
                tapLine.movingCoarseContour[rel].valid)
            {
                const auto &mc = tapLine.movingCoarseContour[rel];
                upContourGate = 0.25 + 0.75 * std::clamp(mc.upTrust, 0.0, 1.0);
                dnContourGate = 0.25 + 0.75 * std::clamp(mc.downTrust, 0.0, 1.0);

                if (rel < (int)tapLine.hLumaDeltaIRE.size()) {
	                    const double chromaT =
	                        (rel < (int)tapLine.centerAdmittedChromaT.size())
	                            ? tapLine.centerAdmittedChromaT[rel]
	                            : 0.0;
                    const double hEdge =
                        std::clamp(
                            (tapLine.hLumaDeltaIRE[rel] - 0.30 * edgeThreshIRE) /
                            (0.70 * edgeThreshIRE),
                            0.0, 1.0);
                    const double sideBalance =
                        1.0 - std::fabs(
                            std::clamp(mc.upTrust, 0.0, 1.0) -
                            std::clamp(mc.downTrust, 0.0, 1.0));
                    const double bevelRisk =
                        chromaT *
                        hEdge *
                        (1.0 - std::clamp(mc.straightness, 0.0, 1.0)) *
                        (0.35 + 0.65 * std::clamp(sideBalance, 0.0, 1.0));
                    const double bevelGate =
                        std::clamp(
                            1.0 - T.FIELD_B_BEVEL_REACH_PENALTY * bevelRisk,
                            0.0, 1.0);
                    upContourGate *= bevelGate;
                    dnContourGate *= bevelGate;
                }
            }
            upPair[rel].reachGate = std::clamp(
                upContourGate * upPair[rel].reachLegalGate, 0.0, 1.0);
            dnPair[rel].reachGate = std::clamp(
                dnContourGate * dnPair[rel].reachLegalGate, 0.0, 1.0);
        }
    };

    {
        if (wantFrame) {
            applyFrameReachWithIQFloor(tapLine.pairU1, tapLine.pairD1,
                                     tapLine.haveU1, tapLine.haveD1);
        }
    }

    {
        if (wantFieldA || wantFieldB) {
            applyFieldReachWithMovingCoarse(tapLine.pairU2, tapLine.pairD2,
                                            tapLine.haveU2, tapLine.haveD2);
        }
    }

    tapLine.builtFlags = flags;
}

// Field A - we sample 2 and 4 lines above and below, with the 4s asymmetrically
// influencing the 2s,and 2s then influencing the evaluated pixel. Strictly intra-field.

void Comb::FrameBuffer::computeFieldALine(const CombTapLine &tapLine,
                                           double *outFieldLine,
                                           double *outGate)
{
    const int width = tapLine.width;
    if (width <= 0 || !outFieldLine || (int)tapLine.tap0.size() < width) return;

    if (outGate) std::fill(outGate, outGate + width, 1.0f);

    const auto  &T   = configuration.tunables;
    const double invI = invIreScale;
    const double hEdgeThreshIRE = std::max(1.0, T.FIELD_LUMA_EDGE_THRESH_IRE);

    for (int rel = 0; rel < width; ++rel) {
        const double C    = tapLine.tap0[rel].comp;
        const double Cup2 = tapLine.tapU2[rel].comp;
        const double Cdn2 = tapLine.tapD2[rel].comp;
        const double Cup4 = tapLine.tapU4[rel].comp;
        const double Cdn4 = tapLine.tapD4[rel].comp;

        const double reachUp2 = tapLine.pairU2[rel].reachGate;
        const double reachDn2 = tapLine.pairD2[rel].reachGate;

        double wUp2 = tapLine.pairU2[rel].weight * reachUp2;
        double wDn2 = tapLine.pairD2[rel].weight * reachDn2;

        const CombTapContour &curve = tapLine.contour[rel];

        // Per-side coarse-luma-edge facts.  Mild mismatch can shape Field A,
        // but a real vertical context break invalidates the same-context
        // vertical comb premise.  Do not let one surviving side normalize into
        // a full-strength ordered comb artifact.
        const double chromaT =
            (rel < (int)tapLine.centerAdmittedChromaT.size())
                ? tapLine.centerAdmittedChromaT[rel]
                : 0.0;

        double eUp = 0.0;
        double eDn = 0.0;
        double verticalContextBreak = 0.0;
        bool hardVerticalBreak = false;

        if (rel < (int)tapLine.coarse0IRE.size() &&
            rel < (int)tapLine.coarseU2IRE.size() &&
            rel < (int)tapLine.coarseD2IRE.size())
        {
            const double LUMA_EDGE_LO_IRE = 6.0;
            const double LUMA_EDGE_HI_IRE = 20.0;

            const double dUpIRE = std::fabs(
                tapLine.coarse0IRE[rel] - tapLine.coarseU2IRE[rel]);
            const double dDnIRE = std::fabs(
                tapLine.coarse0IRE[rel] - tapLine.coarseD2IRE[rel]);

            eUp = std::clamp(
                (dUpIRE - LUMA_EDGE_LO_IRE) /
                (LUMA_EDGE_HI_IRE - LUMA_EDGE_LO_IRE),
                0.0,
                1.0);

            eDn = std::clamp(
                (dDnIRE - LUMA_EDGE_LO_IRE) /
                (LUMA_EDGE_HI_IRE - LUMA_EDGE_LO_IRE),
                0.0,
                1.0);

            verticalContextBreak = std::max(eUp, eDn);

            // Same starting point as Field B.  If this cedes too much, try
            // 16.0; avoid 20.0 as the first stop because the ordered-column
            // failure is already eligible before the old HI threshold.
            const double HARD_VERTICAL_BREAK_IRE = 14.0;
            hardVerticalBreak = (std::max(dUpIRE, dDnIRE) >= HARD_VERTICAL_BREAK_IRE);

            if (!hardVerticalBreak) {
                // Mild contrast remains flexible, but do not fully kill a side
                // here.  Full side loss plus sc2 normalization is the failure.
                wUp2 *= (1.0 - 0.80 * eUp);
                wDn2 *= (1.0 - 0.80 * eDn);
            }
        }

        double boundaryCede = 0.0;

        if (!hardVerticalBreak &&
            rel < (int)tapLine.pairU2.size() &&
            rel < (int)tapLine.pairD2.size() &&
            rel < (int)tapLine.hLumaDeltaIRE.size())
        {
            const double hEdge = std::clamp(
                (tapLine.hLumaDeltaIRE[rel] - 0.45 * hEdgeThreshIRE) /
                (0.55 * hEdgeThreshIRE),
                0.0,
                1.0);

            if (hEdge > 0.0) {
                const double dUp0IRE = tapLine.pairU2[rel].diffIRE;
                const double dDn0IRE = tapLine.pairD2[rel].diffIRE;
                const double scalarUpDn = std::fabs(Cup2 - Cdn2) * invI;

                const double lumaUpDn =
                    (rel < (int)tapLine.coarseU2IRE.size() &&
                     rel < (int)tapLine.coarseD2IRE.size())
                        ? std::fabs(tapLine.coarseU2IRE[rel] - tapLine.coarseD2IRE[rel])
                        : scalarUpDn;

                const double dUpDnIRE =
                    scalarUpDn * (1.0 - chromaT) + lumaUpDn * chromaT;

                const double diffGapIRE = std::fabs(dUp0IRE - dDn0IRE);
                const double bestDiffIRE = std::min(dUp0IRE, dDn0IRE);
                const double worstDiffIRE = std::max(dUp0IRE, dDn0IRE);
                const double diffRatio =
                    (worstDiffIRE > 1e-9) ? (bestDiffIRE / worstDiffIRE) : 1.0;

                const double wBest = std::min(tapLine.pairU2[rel].weight,
                                              tapLine.pairD2[rel].weight);
                const double wWorst = std::max(tapLine.pairU2[rel].weight,
                                               tapLine.pairD2[rel].weight);
                const double wRatio =
                    (wWorst > 1e-9) ? (wBest / wWorst) : 1.0;

                const double matchIRE = 3.5;
                const double betweenIRE = 6.0;
                const double edgeUdIRE = 8.0;

                if (dUpDnIRE > edgeUdIRE) {
                    const bool preferUp = (dUp0IRE <= dDn0IRE);
                    const bool hardPreferUp =
                        preferUp &&
                        ((dUp0IRE < matchIRE && dDn0IRE > betweenIRE) ||
                         (bestDiffIRE < 4.5 && diffGapIRE > 2.5 && diffRatio < 0.55) ||
                         (bestDiffIRE < 5.0 && wRatio < 0.45));
                    const bool hardPreferDn =
                        !preferUp &&
                        ((dDn0IRE < matchIRE && dUp0IRE > betweenIRE) ||
                         (bestDiffIRE < 4.5 && diffGapIRE > 2.5 && diffRatio < 0.55) ||
                         (bestDiffIRE < 5.0 && wRatio < 0.45));

                    bool allowOneSidedBoundary = false;
                    if (rel < (int)tapLine.intrafieldRegionReach.size()) {
                        const auto &region = tapLine.intrafieldRegionReach[rel];
                        allowOneSidedBoundary = region.valid &&
                            ((preferUp &&
                              region.up == CombContentReach::RegionRelation::SameRegion &&
                              region.down == CombContentReach::RegionRelation::DifferentRegion) ||
                             (!preferUp &&
                              region.down == CombContentReach::RegionRelation::SameRegion &&
                              region.up == CombContentReach::RegionRelation::DifferentRegion));
                    }

                    if ((hardPreferUp || hardPreferDn) && !allowOneSidedBoundary) {
                        wUp2 = 0.0;
                        wDn2 = 0.0;
                        boundaryCede = 1.0;
                    } else if (hardPreferUp) {
                        wDn2 = 0.0;
                        wUp2 = std::max(wUp2, 0.40 + 0.60 * hEdge);
                    } else if (hardPreferDn) {
                        wUp2 = 0.0;
                        wDn2 = std::max(wDn2, 0.40 + 0.60 * hEdge);
                    } else {
                        const double sideGapT =
                            std::clamp((diffGapIRE - 1.0) / 4.0, 0.0, 1.0);
                        const double cedeT =
                            std::clamp((bestDiffIRE - 3.0) / 6.0, 0.0, 1.0);

                        if (preferUp)
                            wDn2 *= std::max(0.0, 1.0 - 0.85 * hEdge * (0.35 + 0.65 * sideGapT));
                        else
                            wUp2 *= std::max(0.0, 1.0 - 0.85 * hEdge * (0.35 + 0.65 * sideGapT));

                        boundaryCede = std::max(boundaryCede, 0.70 * hEdge * cedeT);
                    }
                }
            }
        }

        // Windowed OR of the no-valid-partner flag: three-region cedes
        // (drop shadows) must hold a uniform height, not flicker column to
        // column, so cede outranks one-sided anywhere in the window.
        // Bit1 marks the drop-shadow (strong magnitude-asymmetry) islands
        // that qualify for the zero-chroma render.
        // Per-column cede island read: no rel±4 OR window.  The prior
        // 9-column contagion spread a single cede across eight neighbours,
        // giving Field B's render a horizontal spatial signature — lateral
        // influence.  Vertical-comb geometry is per-column: read this
        // column's cede flags only.
        bool centerIsland = false;
        bool shadowIsland = false;
        if ((int)tapLine.intrafieldRegionCede.size() > rel) {
            const std::uint8_t v = tapLine.intrafieldRegionCede[rel];
            if (v & 1) centerIsland = true;
            if (v & 2) shadowIsland = true;
        }
        if (!centerIsland && rel < (int)tapLine.intrafieldRegionReach.size()) {
            const auto &region = tapLine.intrafieldRegionReach[rel];
            if (region.valid) {
                const bool upDifferent = region.up ==
                    CombContentReach::RegionRelation::DifferentRegion;
                const bool downDifferent = region.down ==
                    CombContentReach::RegionRelation::DifferentRegion;
                const bool upSame = region.up ==
                    CombContentReach::RegionRelation::SameRegion;
                const bool downSame = region.down ==
                    CombContentReach::RegionRelation::SameRegion;

                if (downDifferent && upSame) {
                    wDn2 = 0.0;
                } else if (upDifferent && downSame) {
                    wUp2 = 0.0;
                }
            }
        }

        double sc2 = 1.0;

        if (hardVerticalBreak || centerIsland) {
            // A hard vertical context break means Field A has no valid
            // same-context answer here.  Signed-IQ center islands carry the
            // same verdict: neither +/-2 leg belongs to the center region.
            // Both cases also block the revive path.
            wUp2 = 0.0;
            wDn2 = 0.0;
            boundaryCede = 1.0;
        } else if ((wUp2 > 0.0) || (wDn2 > 0.0)) {
            // Keep the old strong-asymmetry handling only when the hard
            // vertical context veto has not fired.  Otherwise this becomes
            // the ordered one-sided comb failure.
            if (wDn2 > 3.0 * wUp2)      wUp2 = 0.0;
            else if (wUp2 > 3.0 * wDn2) wDn2 = 0.0;

            const double denom = wUp2 + wDn2;
            if (denom > 1e-9) {
                sc2 = 2.0 / denom;
                if (sc2 < 1.0) sc2 = 1.0;
            } else {
                wUp2 = 0.0;
                wDn2 = 0.0;
            }
        } else {
            double dMag  = std::fabs(std::fabs(Cup2) - std::fabs(Cdn2));
            double sumUD = std::fabs(Cup2 + Cdn2);

            if (dMag - std::fabs(sumUD * 0.2) <= 0.0) {
                wUp2 = reachUp2;
                wDn2 = reachDn2;
                sc2 = 1.0;
            } else {
                wUp2 = 0.0;
                wDn2 = 0.0;
            }
        }

        auto refineNearWithFar = [&](double nearS, double farS, double influence)->double {
            if (influence <= 0.0) return nearS;
            if (nearS == 0.0) return nearS;
            if ((nearS > 0.0) != (farS > 0.0)) return nearS;

            const double nearMag = std::fabs(nearS);
            const double farMag  = std::fabs(farS);
            const double mag = (nearMag + influence * farMag) / (1.0 + influence);
            return std::copysign(mag, nearS);
        };

        const double Cup2Adj = refineNearWithFar(Cup2, Cup4, curve.upInfluence);
        const double Cdn2Adj = refineNearWithFar(Cdn2, Cdn4, curve.dnInfluence);

        double tc = 0.0;
        const bool combed = (wUp2 > 0.0 || wDn2 > 0.0);

        if (combed) {
            double t2  = ((C - Cup2Adj) * wUp2 * sc2);
            t2        += ((C - Cdn2Adj) * wDn2 * sc2);
            tc        += 0.25 * t2;
        } else if (shadowIsland && chromaT < 0.375) {
            // Same law as Field B: at a drop-shadow island the 1D holds
            // genuine partial chroma unless horizontal luma structure can
            // generate cross-color, so the render follows the carrier-free
            // hEdge ramp — 1D fade where flat, zero where a luma edge
            // would bead through the 1D.
            const double hEdgeT =
                (rel < (int)tapLine.hLumaDeltaIRE.size())
                    ? std::clamp(
                        (tapLine.hLumaDeltaIRE[rel] - 0.35 * hEdgeThreshIRE) /
                        (0.65 * hEdgeThreshIRE),
                        0.0,
                        1.0)
                    : 0.0;
            tc = C * (1.0 - hEdgeT);
        } else {
            tc = C;
        }

        // Zipper defense: cede on either participating vertical luma break,
        // not only the both-sides case.  min(eUp,eDn) asked "are both sides
        // bad?"  Field A's model validity needs "did either side cross a real
        // context boundary?"
        if (combed) {
            const double lumaEdgeCede = std::max(eUp, eDn);

            double bevelCede = 0.0;
            if (rel < (int)tapLine.movingCoarseContour.size() &&
                tapLine.movingCoarseContour[rel].valid &&
                rel < (int)tapLine.hLumaDeltaIRE.size())
            {
                const auto &mc = tapLine.movingCoarseContour[rel];
                const double hEdge = std::clamp(
                    (tapLine.hLumaDeltaIRE[rel] - 0.30 * hEdgeThreshIRE) /
                    (0.70 * hEdgeThreshIRE),
                    0.0,
                    1.0);

                const double sideBalance =
                    1.0 - std::fabs(
                        std::clamp(mc.upTrust, 0.0, 1.0) -
                        std::clamp(mc.downTrust, 0.0, 1.0));

                const double bevelRisk =
                    chromaT *
                    hEdge *
                    (1.0 - std::clamp(mc.straightness, 0.0, 1.0)) *
                    (0.35 + 0.65 * std::clamp(sideBalance, 0.0, 1.0));

                bevelCede =
                    std::clamp(T.FIELD_B_BEVEL_CEDE_STRENGTH * bevelRisk, 0.0, 1.0);
            }

            const double totalCede = std::max({lumaEdgeCede, bevelCede, boundaryCede});
            if (totalCede > 0.0)
                tc = tc * (1.0 - totalCede) + C * totalCede;
        }

        if (!std::isfinite(tc))
            tc = C;

        outFieldLine[rel] = tc;

        double gateA = std::max(wUp2, wDn2);
        gateA = std::clamp(gateA, 0.0, 1.0);
        if (outGate) outGate[rel] = gateA;
    }

    // (FieldAStats per-line logging removed: Field A is no longer in the
    // election, and the per-line spam buried the active diagnostics.)
}

// Field B
// Simplified Field comb as a FrameBuffer member:
// - uses only 2 vertical neighbours
void Comb::FrameBuffer::computeFieldBLine(int lineNumber,
                                               double *outFieldLine,
                                               std::uint8_t *outReasonLine)
{
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int width = videoParameters.activeVideoEnd - videoParameters.activeVideoStart;

    if (width <= 0 || lineNumber < first || lineNumber >= last || !outFieldLine) {
        if (outFieldLine) std::fill(outFieldLine, outFieldLine + std::max(width, 0), 0.0);
        if (outReasonLine) std::fill(outReasonLine, outReasonLine + std::max(width, 0), FieldBReasonNone);
        return;
    }

    const CombTapLine &tapLine = ensureCombTapLine(lineNumber);
    computeFieldBLine(tapLine, outFieldLine, outReasonLine);
}

void Comb::FrameBuffer::computeFieldBLine(const CombTapLine &tapLine,
                                           double *outFieldLine,
                                           std::uint8_t *outReasonLine)
{
    const int width =
        videoParameters.activeVideoEnd - videoParameters.activeVideoStart;
    const int first = videoParameters.firstActiveFrameLine;
    const int last = videoParameters.lastActiveFrameLine;
    const int lineNumber = tapLine.cacheLine;

    if (width <= 0 || !outFieldLine)
        return;

    if (lineNumber < first || lineNumber >= last) {
        std::fill(outFieldLine, outFieldLine + width, 0.0);
        if (outReasonLine)
            std::fill(outReasonLine, outReasonLine + width, FieldBReasonNone);
        return;
    }

    const bool haveCenter = static_cast<int>(tapLine.tap0.size()) >= width;
    const bool haveVertical =
        tapLine.haveU2 && tapLine.haveD2 &&
        static_cast<int>(tapLine.tapU2.size()) >= width &&
        static_cast<int>(tapLine.tapD2.size()) >= width &&
        static_cast<int>(tapLine.pairU2.size()) >= width &&
        static_cast<int>(tapLine.pairD2.size()) >= width;

    if (!haveCenter) {
        std::fill(outFieldLine, outFieldLine + width, 0.0);
        if (outReasonLine)
            std::fill(outReasonLine, outReasonLine + width, FieldBReasonNone);
        return;
    }

    if (!haveVertical) {
        for (int rel = 0; rel < width; ++rel)
            outFieldLine[rel] = tapLine.tap0[rel].comp;
        if (outReasonLine)
            std::fill(outReasonLine, outReasonLine + width, FieldBReasonCenter);
        return;
    }

    // =========================== RED LABEL ================================
    // FIELD B MAINTENANCE RULE
    //
    // DO NOT FIX PICTURE-CONTENT PROBLEMS BY ADDING CODE HERE.
    //
    // Field B only consumes the two +/-2 tap weights prepared upstream. It
    // does not discover chroma, parse boundaries, repair contours, revive
    // rejected reaches, or reinterpret carrier evidence. If a tap has the
    // wrong authority, fix CombTapLine construction or its reach contract.
    // Keep this function a transparent, bounded scalar combine.
    // =====================================================================

    for (int rel = 0; rel < width; ++rel) {
        const double center = tapLine.tap0[rel].comp;
        const double up = tapLine.tapU2[rel].comp;
        const double down = tapLine.tapD2[rel].comp;

        double wUp = std::clamp(
            tapLine.pairU2[rel].weight * tapLine.pairU2[rel].reachGate,
            0.0,
            1.0);
        double wDown = std::clamp(
            tapLine.pairD2[rel].weight * tapLine.pairD2[rel].reachGate,
            0.0,
            1.0);

        const bool useUp = wUp > 1e-9;
        const bool useDown = wDown > 1e-9;
        double output = center;
        std::uint8_t reason = FieldBReasonCenter;

        if (useUp && useDown) {
            const double scale = 2.0 / (wUp + wDown);
            output = 0.25 * (
                (center - up) * wUp * scale +
                (center - down) * wDown * scale);
            reason = FieldBReasonBlend;
        } else if (useUp) {
            output = 0.25 * (center - up) * wUp;
            reason = FieldBReasonBlend;
        } else if (useDown) {
            output = 0.25 * (center - down) * wDown;
            reason = FieldBReasonBlend;
        }

        if (!std::isfinite(output)) {
            output = center;
            reason = FieldBReasonCenter;
        }

        outFieldLine[rel] = output;
        if (outReasonLine)
            outReasonLine[rel] = reason;
    }
}


static inline double cmag(const std::complex<double> &z) { return std::hypot(z.real(), z.imag()); }
static inline double cmag2(const std::complex<double> &z) { return z.real() * z.real() + z.imag() * z.imag(); }
static inline double dotIQ(const std::complex<double> &a, const std::complex<double> &b) { return a.real()*b.real() + a.imag()*b.imag(); }

struct ColumnPhaseAlignmentLimits {
    double minFitIRE = 0.0;
    double pMax = 0.0;
    double tanPMax = 0.0;
};

static inline std::complex<double> applyColumnPhaseAlignment(
    const std::complex<double> &center,
    const std::complex<double> &neighbor,
    double invI,
    const ColumnPhaseAlignmentLimits &limits)
{
    const double a0 = cmag(center);
    const double an = cmag(neighbor);
    if (a0 * invI < limits.minFitIRE || an * invI < limits.minFitIRE)
        return neighbor;

    const double dot = dotIQ(neighbor, center);
    const double cross = neighbor.real() * center.imag() - neighbor.imag() * center.real();

    // The rotation we want is by phase = atan2(cross, dot).  When the clamp
    // doesn't bind, cos(phase) = dot/h and sin(phase) = cross/h where
    // h = hypot(dot, cross) — so one hypot replaces atan2 + cos + sin.
    // The clamp binds iff |phase| > pMax, equivalent to
    //   dot < 0  (phase in second/third quadrant, always > pMax for pMax<90°)
    //   or  |cross| > dot * tan(pMax).
    // Only then do we fall back to atan2+cos+sin to honor the clamp.
    double c, s;
    if (dot > 0.0 && std::fabs(cross) <= dot * limits.tanPMax) {
        const double h = std::hypot(dot, cross);
        if (h <= 1e-18) return neighbor;
        c = dot / h;
        s = cross / h;
    } else {
        double phase = std::atan2(cross, dot);
        phase = std::clamp(phase, -limits.pMax, limits.pMax);
        c = std::cos(phase);
        s = std::sin(phase);
    }

    return std::complex<double>(
        c * neighbor.real() - s * neighbor.imag(),
        s * neighbor.real() + c * neighbor.imag());
}

void Comb::FrameBuffer::computeIQFrameAFromPreparedVectors(
    int line,
    const std::vector<std::complex<double>> &centerIQ,
    std::vector<std::complex<double>> &upIQ,
    std::vector<std::complex<double>> &dnIQ,
    std::vector<std::complex<double>> &outFrameIQ,
    const CombTapLine *reachTapLine)
{
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;

    if (width <= 0) {
        outFrameIQ.clear();
        return;
    }
    outFrameIQ.resize(width);

    auto clearFrameIQ = [&]() {
        std::fill(outFrameIQ.begin(), outFrameIQ.end(), std::complex<double>(0.0, 0.0));
    };
    if (line < first || line >= last) {
        clearFrameIQ();
        return;
    }
    if (line >= demodLines || demodWidth <= 0) {
        clearFrameIQ();
        return;
    }

    const auto  &T    = configuration.tunables;
    const double invI = invIreScale;

    const bool verticalAllowed = carrierFrameVerticalAllowed(line);
    const bool haveUpLine = verticalAllowed && (line - 1 >= first);
    const bool haveDnLine = verticalAllowed && (line + 1 <  last);

    // Signed correlation in [-1..1].  Caller supplies pre-computed magnitudes
    // to avoid recomputing hypot per call site.
    auto corrSignedMags = [&](const std::complex<double> &a,
                              const std::complex<double> &b,
                              double ma, double mb)->double {
        if (ma <= 1e-12 || mb <= 1e-12) return 0.0;
        return dotIQ(a, b) / (ma*mb + 1e-12);
    };

    // Soft signed contribution and its weight, computed together so the shared
    // (c, ac, w) calculation is not run twice per neighbor.  Magnitudes are
    // supplied by the caller (precomputed once per pixel).
    auto softAlignBoth = [&](const std::complex<double> &Z0,
                             const std::complex<double> &Zn,
                             double a0, double an,
                             std::complex<double> &contribOut,
                             double &weightOut)
    {
        if (a0 <= 1e-12 || an <= 1e-12) {
            contribOut = {0.0, 0.0};
            weightOut  = 0.0;
            return;
        }
        const double c  = dotIQ(Z0, Zn) / (a0*an + 1e-12); // signed corr [-1..1]
        const double ac = std::fabs(c);

        const double t0 = 0.55;
        const double t1 = 0.85;

        double w = (ac - t0) / (t1 - t0);
        w = std::clamp(w, 0.0, 1.0);

        // No floor: if correlation is below the ramp start, the neighbor
        // does not contribute.  Edge safety is reach's job; a wFloor here
        // leaks badly-correlated material through throttled-but-nonzero reach.

        const double s = (c >= 0.0) ? 1.0 : -1.0;
        contribOut = Zn * (w * s);
        weightOut  = w;
    };

    const double COMB_STRENGTH  = std::max(0.0, T.FRAME_COMB_STRENGTH);
    const double MAX_DELTA_IRE  = T.FRAME_IQ_RAW_MAX_DELTA_IRE;
    const double MIN_CHROMA_IRE = T.FRAME_CHROMA_MIN_IRE;
    const double phaseAlignMax = T.FRAME_IQ_COLUMN_PHASE_ALIGN_MAX_DEG * M_PI / 180.0;
    const ColumnPhaseAlignmentLimits phaseAlignLimits{
        std::max(2.0, 0.5 * MIN_CHROMA_IRE),
        phaseAlignMax,
        std::tan(phaseAlignMax)
    };

    // ------------------------------------------------------------
    // Build IQ vectors for center and 1 neighbors
    // ------------------------------------------------------------
    if ((int)centerIQ.size() < width || (int)upIQ.size() < width || (int)dnIQ.size() < width) {
        clearFrameIQ();
        return;
    }

    std::copy(centerIQ.begin(), centerIQ.begin() + width, outFrameIQ.begin());

    // Frame A's alignment is intentionally column-local. A line/global affine
    // can learn from adjacent material and push that error into the current
    // column; vertical combing should not borrow lateral context here.
    for (int x = 0; x < width; ++x) {
        upIQ[x] = applyColumnPhaseAlignment(centerIQ[x], upIQ[x], invI, phaseAlignLimits);
        dnIQ[x] = applyColumnPhaseAlignment(centerIQ[x], dnIQ[x], invI, phaseAlignLimits);
    }

    // ------------------------------------------------------------
    // Combine (soft signed contributions + boundary-aware asymmetry)
    // ------------------------------------------------------------
    for (int x = 0; x < width; ++x) {
        const std::complex<double> Z0 = centerIQ[x];
        const double a0 = cmag(Z0);
        const double a0_ire = a0 * invI;

        if (a0_ire <= MIN_CHROMA_IRE) {
            outFrameIQ[x] = Z0;
            continue;
        }

        std::complex<double> ZUpRaw = upIQ[x];
        std::complex<double> ZDnRaw = dnIQ[x];

        const double aUp = cmag(ZUpRaw);
        const double aDn = cmag(ZDnRaw);

        // Reach contract, shared with Frame B (consumed exactly as in
        // computeFrameBLine).  pairU1/pairD1.reachGate
        // already carries legality, the interfield IQ floor, and the bevel/
        // zipper throttle (composed in applyFrameReachWithIQFloor), so the
        // zipper defense Frame A used to hand-roll here — transition side-
        // picking, TRANS_SUPPRESS, up/down disagreement backoff — is gone:
        // duplicating reach is what made it mushy.  Hard zero-reach refusal:
        // a side with reachGate 0 does not contribute; both 0 emits center.
        double upReach = 0.0;
        double dnReach = 0.0;
        if (reachTapLine &&
            x < (int)reachTapLine->pairU1.size() &&
            x < (int)reachTapLine->pairD1.size())
        {
            upReach = haveUpLine
                ? std::clamp(reachTapLine->pairU1[x].reachGate, 0.0, 1.0)
                : 0.0;
            dnReach = haveDnLine
                ? std::clamp(reachTapLine->pairD1[x].reachGate, 0.0, 1.0)
                : 0.0;
        }

        const bool useUp = (upReach > 0.0) && (aUp > 1e-9);
        const bool useDn = (dnReach > 0.0) && (aDn > 1e-9);

        if (!useUp && !useDn) {
            outFrameIQ[x] = Z0;
            continue;
        }

        // Per-side signed correlation, hoisted for reuse by both the phase-
        // protection block and cohGate below.
        const double corrUp = useUp
            ? std::fabs(corrSignedMags(Z0, ZUpRaw, a0, aUp)) : 0.0;
        const double corrDn = useDn
            ? std::fabs(corrSignedMags(Z0, ZDnRaw, a0, aDn)) : 0.0;

        // Phase protection: when center and one neighbor agree on phase,
        // the other neighbor must not rotate the hue.  Project the
        // disagreeing neighbor onto center's phase axis so it can still
        // contribute amplitude (luma-slope influence) but not alter hue.
        // The threshold for "agrees" is the cohGate pass point — the same
        // correlation that fully passes the adaptive-strength gate.
        {
            const double agreeThresh = std::clamp(T.FRAME_IQ_COH_PASS_CORR, 0.0, 1.0);
            const bool upAgrees = useUp && (corrUp >= agreeThresh);
            const bool dnAgrees = useDn && (corrDn >= agreeThresh);
            const double a0sq = a0 * a0;

            if (upAgrees && !dnAgrees && useDn && a0sq > 1e-18) {
                const double proj = dotIQ(ZDnRaw, Z0) / a0sq;
                ZDnRaw = Z0 * proj;
            } else if (dnAgrees && !upAgrees && useUp && a0sq > 1e-18) {
                const double proj = dotIQ(ZUpRaw, Z0) / a0sq;
                ZUpRaw = Z0 * proj;
            }
        }

        // Combine neighbors with soft signed contributions, weighted by reach
        // (no integer dilution).  softAlignBoth returns contribution and weight
        // together so the shared correlation calc isn't repeated per side; the
        // reach gate scales both, so a throttled side pulls less without any
        // separate boundary heuristic.
        std::complex<double> Zsum = Z0;
        double wsum = 1.0;

        if (useUp) {
            std::complex<double> contrib;
            double w;
            softAlignBoth(Z0, ZUpRaw, a0, aUp, contrib, w);
            w *= upReach;
            if (w > 0.0) {
                Zsum += contrib * upReach;
                wsum += w;
            }
        }
        if (useDn) {
            std::complex<double> contrib;
            double w;
            softAlignBoth(Z0, ZDnRaw, a0, aDn, contrib, w);
            w *= dnReach;
            if (w > 0.0) {
                Zsum += contrib * dnReach;
                wsum += w;
            }
        }

        std::complex<double> Zframe = Zsum / wsum;

        std::complex<double> delta = Zframe - Z0;
        const double deltaMagIRE = cmag(delta) * invI;

        if (MAX_DELTA_IRE > 0.0 && deltaMagIRE > MAX_DELTA_IRE && deltaMagIRE > 1e-9)
            delta *= (MAX_DELTA_IRE / deltaMagIRE);

        // --------------------------------------------------------
        // Adaptive comb strength: COMB_STRENGTH_LO .. COMB_STRENGTH.
        // This is Frame A's signature — the strength tracks signed center/
        // neighbor correlation.  The old vertical-disagreement backoff (disGate)
        // and gamma=2 selectivity are removed: cross-boundary safety is reach's
        // job now, and squaring softened the mid-correlation comb.  cohGate
        // ramps to a full-firm pass at FRAME_IQ_COH_PASS_CORR (the knob), so the
        // floor stays firm (LO = 0.8) and correlation only modulates the top.
        // --------------------------------------------------------
        const double COMB_STRENGTH_HI = COMB_STRENGTH;
        const double COMB_STRENGTH_LO = std::min(0.8, COMB_STRENGTH_HI);

        // Coherence: reuse the hoisted per-side correlations.
        const double coh = std::max(corrUp, corrDn);

        // Map coherence -> [0..1].  cohPass is the correlation required to pass
        // cohGate fully; the ramp starts a fixed 0.30 below it.
        const double cohPass  = std::clamp(T.FRAME_IQ_COH_PASS_CORR, 0.0, 1.0);
        const double cohStart = std::max(0.0, cohPass - 0.30);
        double cohGate = (cohPass > cohStart)
            ? (coh - cohStart) / (cohPass - cohStart)
            : (coh >= cohPass ? 1.0 : 0.0);
        cohGate = std::clamp(cohGate, 0.0, 1.0);

        double localStrength =
            COMB_STRENGTH_LO + (COMB_STRENGTH_HI - COMB_STRENGTH_LO) * cohGate;

        outFrameIQ[x] = Z0 + (delta * localStrength);
    }
}
// Frame A: adaptive interframe IQ comb fed by the Field B preclean ring.
void Comb::FrameBuffer::computeFrameALine(
    int line,
    std::vector<std::complex<double>> &outFrameIQ)
{
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;
    const bool verticalAllowed = carrierFrameVerticalAllowed(line);

	    if (width <= 0) {
	        outFrameIQ.clear();
	        return;
	    }
	    outFrameIQ.resize(width);
	    auto clearFrameIQ = [&]() {
	        std::fill(outFrameIQ.begin(), outFrameIQ.end(), std::complex<double>(0.0, 0.0));
	    };
	    if (line < first || line >= last) {
	        clearFrameIQ();
	        return;
	    }
	    if (line >= demodLines || demodWidth <= 0) {
	        clearFrameIQ();
	        return;
	    }

    // Frame A is a client of the same reach contract as Frame B: the per-pixel
    // pairU1/pairD1.reachGate (legality + interfield IQ floor + bevel/zipper
    // throttle, all composed in applyFrameReachWithIQFloor) is what makes
    // zippers structurally impossible, so Frame A no longer needs — and must not
    // duplicate — its own boundary suppression.  The tap line is already built
    // with TapBuildFrame in this flow; ensure is a cache hit here.
    const CombTapLine &reachTapLine = ensureCombTapLine(line);

    auto tiLine = [&](int ln)->const float* { return demodTI4fsc_line(ln); };
    auto tqLine = [&](int ln)->const float* { return demodTQ4fsc_line(ln); };

    const float *ti0_raw  = tiLine(line);
    const float *tq0_raw  = tqLine(line);
    const float *tiUp_raw = (verticalAllowed && line - 1 >= first) ? tiLine(line - 1) : nullptr;
    const float *tqUp_raw = (verticalAllowed && line - 1 >= first) ? tqLine(line - 1) : nullptr;
    const float *tiDn_raw = (verticalAllowed && line + 1 <  last)  ? tiLine(line + 1) : nullptr;
    const float *tqDn_raw = (verticalAllowed && line + 1 <  last)  ? tqLine(line + 1) : nullptr;

    auto scalarLine = [&](int ln)->const double* {
        if (ln < first || ln >= last) return nullptr;
        return configuration.phaseCompensation
            ? locked1DSource_line(ln)
            : bucketScalar1D_line(ln);
    };

    const double *preclean0  = precleanLinePtr(line, width);
    const double *precleanUp = verticalAllowed ? precleanLinePtr(line - 1, width) : nullptr;
    const double *precleanDn = verticalAllowed ? precleanLinePtr(line + 1, width) : nullptr;

    const double *scalar0  = scalarLine(line);
    const double *scalarUp = verticalAllowed ? scalarLine(line - 1) : nullptr;
    const double *scalarDn = verticalAllowed ? scalarLine(line + 1) : nullptr;

    auto phaseCursor = [&](int ln) {
        return carrierGrammarSignedSampleCursor(
            configuration.phaseCompensation ? carrierGrammarLine(ln) : nullptr,
            left);
    };
    auto preclean0Cursor  = phaseCursor(line);
    auto precleanUpCursor = phaseCursor(line - 1);
    auto precleanDnCursor = phaseCursor(line + 1);
    auto scalar0Cursor    = phaseCursor(line);
    auto scalarUpCursor   = phaseCursor(line - 1);
    auto scalarDnCursor   = phaseCursor(line + 1);

    if ((int)scratch_centerIQ.size() != width) scratch_centerIQ.resize(width);
    if ((int)scratch_upIQ.size() != width) scratch_upIQ.resize(width);
    if ((int)scratch_dnIQ.size() != width) scratch_dnIQ.resize(width);
    for (int x = 0; x < width; ++x) {
        if (preclean0)
            scratch_centerIQ[x] =
                carrierGrammarDemodSignedCompositeTo4fsc(preclean0Cursor, preclean0[x]);
        else if (ti0_raw && tq0_raw)
            scratch_centerIQ[x] = std::complex<double>((double)ti0_raw[x], (double)tq0_raw[x]);
        else if (scalar0)
            scratch_centerIQ[x] =
                carrierGrammarDemodSignedCompositeTo4fsc(scalar0Cursor, scalar0[x]);
        else scratch_centerIQ[x] = std::complex<double>(0.0, 0.0);

        if (precleanUp) {
            scratch_upIQ[x] =
                carrierGrammarDemodSignedCompositeTo4fsc(precleanUpCursor, precleanUp[x]);
        } else {
            if (tiUp_raw && tqUp_raw)
                scratch_upIQ[x] = std::complex<double>((double)tiUp_raw[x], (double)tqUp_raw[x]);
            else if (scalarUp)
                scratch_upIQ[x] =
                    carrierGrammarDemodSignedCompositeTo4fsc(scalarUpCursor, scalarUp[x]);
            else
                scratch_upIQ[x] = std::complex<double>(0.0, 0.0);
        }

        if (precleanDn) {
            scratch_dnIQ[x] =
                carrierGrammarDemodSignedCompositeTo4fsc(precleanDnCursor, precleanDn[x]);
        } else {
            if (tiDn_raw && tqDn_raw)
                scratch_dnIQ[x] = std::complex<double>((double)tiDn_raw[x], (double)tqDn_raw[x]);
            else if (scalarDn)
                scratch_dnIQ[x] =
                    carrierGrammarDemodSignedCompositeTo4fsc(scalarDnCursor, scalarDn[x]);
            else
                scratch_dnIQ[x] = std::complex<double>(0.0, 0.0);
        }
    }

    computeIQFrameAFromPreparedVectors(line, scratch_centerIQ, scratch_upIQ, scratch_dnIQ,
                                      outFrameIQ, &reachTapLine);
}

// Frame B: direct interframe IQ comb.
// Sources from the Field B preclean ring, with locked-1D IQ as fallback.
// Unlike Frame A, this intentionally does not phase-align the neighbors before
// averaging; that plainness is part of what keeps Frame B from inheriting
// Frame A's saturated-edge alternation failure mode.
// Reach is consumed from pairU1/pairD1.reachGate (legality + interfield IQ
// floor); the combine itself is unconditional when reach is non-zero.
void Comb::FrameBuffer::computeFrameBLine(
    int line,
    std::vector<std::complex<double>> &outFrameIQ,
    std::vector<double> &outFrameScalar)
{
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;

    if (width <= 0) {
        outFrameIQ.clear();
        outFrameScalar.clear();
        return;
    }
    outFrameIQ.resize(width);
    outFrameScalar.resize(width);
    auto clearFrameOutputs = [&]() {
        std::fill(outFrameIQ.begin(), outFrameIQ.end(), std::complex<double>(0.0, 0.0));
        std::fill(outFrameScalar.begin(), outFrameScalar.end(), 0.0);
    };
    if (line < first || line >= last) {
        clearFrameOutputs();
        return;
    }
    if (line >= demodLines || demodWidth <= 0) {
        clearFrameOutputs();
        return;
    }

    const bool verticalAllowed = carrierFrameVerticalAllowed(line);
    auto tiLine = [&](int ln)->const float* { return locked1DTI4fsc_line(ln); };
    auto tqLine = [&](int ln)->const float* { return locked1DTQ4fsc_line(ln); };

    const float *ti0_raw  = tiLine(line);
    const float *tq0_raw  = tqLine(line);
    const bool haveUpLine = verticalAllowed && (line - 1 >= first);
    const bool haveDnLine = verticalAllowed && (line + 1 < last);
    const float *tiUp_raw = haveUpLine ? tiLine(line - 1) : nullptr;
    const float *tqUp_raw = haveUpLine ? tqLine(line - 1) : nullptr;
    const float *tiDn_raw = haveDnLine ? tiLine(line + 1) : nullptr;
    const float *tqDn_raw = haveDnLine ? tqLine(line + 1) : nullptr;
    if (!ti0_raw || !tq0_raw) {
        clearFrameOutputs();
        return;
    }

    const CombTapLine &reachTapLine = ensureCombTapLine(line);
    const double *preclean0  = precleanLinePtr(line, width);
    const double *precleanUp = haveUpLine ? precleanLinePtr(line - 1, width) : nullptr;
    const double *precleanDn = haveDnLine ? precleanLinePtr(line + 1, width) : nullptr;

    auto phaseCursor = [&](int ln) {
        return carrierGrammarSignedSampleCursor(
            configuration.phaseCompensation ? carrierGrammarLine(ln) : nullptr,
            left);
    };

    auto phase0Cursor  = phaseCursor(line);
    auto phaseUpCursor = phaseCursor(haveUpLine ? line - 1 : line);
    auto phaseDnCursor = phaseCursor(haveDnLine ? line + 1 : line);

    // Preclean and cached locked 1D have different scalar round-trip
    // contracts. Keep both cursors live so an explicit forced-center override
    // cannot desync the carrier phase of every pixel that follows it.
    auto signedRemodCursor = carrierGrammarSignedSampleCursor(
        configuration.phaseCompensation ? carrierGrammarLine(line) : nullptr,
        left);
    auto gridRemodCursor = lddecode::carrierGrammarCompositeRemodCursor(
        configuration.phaseCompensation ? carrierGrammarLine(line) : nullptr,
        left,
        1.0,
        lddecode::CarrierSignFrame::Grid4fsc);

    const auto &T = configuration.tunables;
    const double combStrength =
        std::clamp(std::max(0.0, T.FRAME_B_COMB_STRENGTH), 0.0, 1.0);
    const double maxDeltaIRE = std::max(0.0, T.FRAME_B_RAW_MAX_DELTA_IRE);
    static const bool forceFrameBLocked1D = [] {
        const char *s = std::getenv("LD_FRAME_B_FORCE_LOCKED_1D");
        return s && std::atoi(s) != 0;
    }();

    for (int x = 0; x < width; ++x) {
        const bool useLockedCenter = forceFrameBLocked1D;

        // Always consume the signed preclean cursors. Conditional consumption
        // shifts all later samples onto the wrong carrier leg after the first
        // forced-center override.
        const std::complex<double> Z0Preclean = preclean0
            ? carrierGrammarDemodSignedCompositeTo4fsc(phase0Cursor, preclean0[x])
            : std::complex<double>(0.0, 0.0);
        const std::complex<double> ZUp = precleanUp
            ? carrierGrammarDemodSignedCompositeTo4fsc(phaseUpCursor, precleanUp[x])
            : std::complex<double>(0.0, 0.0);
        const std::complex<double> ZDn = precleanDn
            ? carrierGrammarDemodSignedCompositeTo4fsc(phaseDnCursor, precleanDn[x])
            : std::complex<double>(0.0, 0.0);
        const std::complex<double> Z0 = useLockedCenter
            ? std::complex<double>((double)ti0_raw[x], (double)tq0_raw[x])
            : Z0Preclean;

        double upReach = 0.0;
        double dnReach = 0.0;
        if (x < (int)reachTapLine.pairU1.size() &&
            x < (int)reachTapLine.pairD1.size())
        {
            upReach = haveUpLine
                ? std::clamp(reachTapLine.pairU1[x].reachGate, 0.0, 1.0)
                : 0.0;
            dnReach = haveDnLine
                ? std::clamp(reachTapLine.pairD1[x].reachGate, 0.0, 1.0)
                : 0.0;
        }

        std::complex<double> Zout = Z0;
        // Compact patches cede atomically to center 1D. Mixing a cached center
        // with signed-preclean neighbors would combine different sign frames;
        // feeding cached 1D to all three legs would still run the mechanism we
        // are explicitly declining here.
        if (!useLockedCenter && (upReach > 0.0 || dnReach > 0.0)) {
            const bool haveUp = upReach > 0.0 && cmag2(ZUp) > 1e-18;
            const bool haveDn = dnReach > 0.0 && cmag2(ZDn) > 1e-18;

            double wsum = 0.0;
            std::complex<double> target(0.0, 0.0);
            if (haveUp) { target += ZUp * upReach; wsum += upReach; }
            if (haveDn) { target += ZDn * dnReach; wsum += dnReach; }

            if (wsum > 1e-12) {
                target /= wsum;
                std::complex<double> delta = target - Z0;
                const double deltaIRE = cmag(delta) * invIreScale;
                if (maxDeltaIRE > 0.0 && deltaIRE > maxDeltaIRE && deltaIRE > 1e-9)
                    delta *= (maxDeltaIRE / deltaIRE);

                const double reachAuthority = std::clamp(wsum / 2.0, 0.0, 1.0);
                // The ±1 comb is a PROJECTION, not a gain.  In the signed
                // frame, vertically common alien appears anti-phase on the
                // partners, and Zout = Z0 + p·(target−Z0) solves the two-line
                // alternation model exactly at p = 0.5: Zout = chroma, and Y
                // keeps its own carrier-band luma at unit gain.  p > 0.5 does
                // not cancel harder — it re-injects the alien INVERTED at
                // (2p−1) into chroma and overdrives near-carrier luma in Y by
                // the same factor (the 0.80 regression: +60% luma-band gain
                // read as "crisper" static detail and serrated the cube's
                // diagonals while manufacturing diagonal cross-color inside
                // the elected comb).  The 0.5 here is the projection itself;
                // combStrength expresses a fraction of it, so overdrive past
                // the projection is structurally inexpressible.
                const double pull =
                    0.5 * std::clamp(combStrength * reachAuthority, 0.0, 1.0);
                Zout = Z0 + delta * pull;
                if (!std::isfinite(Zout.real()) || !std::isfinite(Zout.imag()))
                    Zout = Z0;
            }
        }

        // Graceful failure at highlights. The forced-center path emits Z0
        // (raw locked IQ) directly, so the comb-branch finite check above
        // never sees it; a non-finite vector on EITHER path falls back to
        // the finite Field B preclean rather than propagating NaN/Inf.  No
        // magnitude clamp here: cmag(Zout) is fullSignedIQ scale (a sum of
        // three samples), not the composite-scalar scale of maxCarrierAmp, so
        // bounding against that rail clipped legitimate high-energy chroma into
        // a carrier-rate checkerboard.  A finite outlier, if any survives, is a
        // LOCAL spike (isolated vs neighbours), not a global-amplitude problem.
        if (!std::isfinite(Zout.real()) || !std::isfinite(Zout.imag())) {
            Zout = (std::isfinite(Z0Preclean.real()) &&
                    std::isfinite(Z0Preclean.imag()))
                       ? Z0Preclean
                       : std::complex<double>(0.0, 0.0);
        }

        outFrameIQ[x] = Zout;
        if (useLockedCenter) {
            outFrameScalar[x] =
                lddecode::carrierGrammarRemod4fscToComposite(
                    gridRemodCursor, Zout.real(), Zout.imag());
            lddecode::carrierGrammarAdvanceSignedSampleCursor(signedRemodCursor);
        } else {
            outFrameScalar[x] = carrierGrammarRemodSigned4fscToComposite(
                signedRemodCursor, Zout.real(), Zout.imag());
            lddecode::carrierGrammarAdvanceRemodCursor(gridRemodCursor);
        }
    }

}

// 3D Section
// getCandidate - prescreen for 3D election
// 3D Section
// getCandidate - prescreen for 3D election
Comb::FrameBuffer::Candidate Comb::FrameBuffer::getCandidate(
    qint32 refLineNumber, qint32 refH,
    const FrameBuffer &frameBuffer, qint32 lineNumber, qint32 h,
    double adjustPenalty) const
{
    Candidate result;
    result.penalty = configuration.candidatePenaltyHardMax;
    result.sample  = 0.0;
    result.yPen    = configuration.candidatePenaltyHardMax;
    result.iqPen   = 0.0;

    const int firstLine  = videoParameters.firstActiveFrameLine;
    const int lastLine   = videoParameters.lastActiveFrameLine;
    const int left       = videoParameters.activeVideoStart;
    const int right      = videoParameters.activeVideoEnd;
    const int fieldWidth = videoParameters.fieldWidth;

    auto clampH = [&](int x) -> int {
        return std::clamp(x, left, right - 1);
    };

    // Bounds check.
    if ((unsigned)(lineNumber - firstLine) >= (unsigned)(lastLine - firstLine) ||
        (unsigned)(refLineNumber - firstLine) >= (unsigned)(lastLine - firstLine)) {
        result.penalty = 1000.0;
        result.yPen    = 1000.0;
        result.iqPen   = 0.0;
        return result;
    }

    // Cross-frame ScalarSignCompare on the mode's actual 1D scalar.  Both the
    // bucket scalar and the locked 1D scalar are PhasePreservedCarrier, so
    // grammar legality answers identically for either; the historical Bucket
    // mislabel (a workaround for the retired common-phase classification) is
    // no longer needed.  The sample read below follows the same mode switch.
    const lddecode::CombReachReply phaseReach = combReachIndex.queryAgainst(
        frameBuffer.combReachIndex,
        {refLineNumber,
         lineNumber,
         refH,
         h,
         lddecode::CombReachUse::ScalarSignCompare,
         scalarReachSource()});

    if (!phaseReach.allowScalarSignCompare ||
        phaseReach.carrierRelation != lddecode::CarrierPhaseRelation::Opposite) {
        result.penalty = 1000.0;
        result.yPen    = 1000.0;
        result.iqPen   = 0.0;
        return result;
    }

    const int hh = clampH(h);

    // 1D sample: locked path reads the phase-corrected blind bandpass;
    // bucket path reads clpbuffer[0] directly.
    const double *lockedRow = frameBuffer.configuration.phaseCompensation
        ? frameBuffer.locked1DSource_line(lineNumber) : nullptr;

    if (lockedRow && (hh - left) >= 0 && (hh - left) < (right - left)) {
        result.sample = lockedRow[hh - left];
    } else {
        result.sample = frameBuffer.bucketScalar1D_line(lineNumber)[hh];
    }

    // --- Luma Penalty with Neighbor Shaping ---
    //
    // This is the already-paid luma-domain evidence:
    //
    //     reconstructed Y = raw - 2D chroma/composite estimate
    //
    // It compares current/reference against candidate over a small cross:
    //
    //     center line: x-1, x, x+1
    //     vertical:    y-1, y+1 at x
    //
    // getBestCandidate can now reuse result.yPen directly instead of
    // re-deriving a same-pixel scalar chroma distance from result.sample.
    const quint16 *refRawC  = rawbuffer.data() + refLineNumber * fieldWidth;
    const double  *refClpC  = clpbuffer[1].pixel[refLineNumber];

    const quint16 *candRawC = frameBuffer.rawbuffer.data() + lineNumber * fieldWidth;
    const double  *candClpC = frameBuffer.clpbuffer[1].pixel[lineNumber];

    const bool verticalAllowed =
        carrierFrameVerticalAllowed(refLineNumber) &&
        frameBuffer.carrierFrameVerticalAllowed(lineNumber);

    const bool haveUp = verticalAllowed &&
                        (refLineNumber - 1 >= firstLine) &&
                        (lineNumber - 1 >= firstLine);

    const bool haveDn = verticalAllowed &&
                        (refLineNumber + 1 < lastLine) &&
                        (lineNumber + 1 < lastLine);

    const quint16 *refRawU = haveUp
        ? (rawbuffer.data() + (refLineNumber - 1) * fieldWidth)
        : refRawC;
    const double *refClpU = haveUp
        ? clpbuffer[1].pixel[refLineNumber - 1]
        : refClpC;

    const quint16 *refRawD = haveDn
        ? (rawbuffer.data() + (refLineNumber + 1) * fieldWidth)
        : refRawC;
    const double *refClpD = haveDn
        ? clpbuffer[1].pixel[refLineNumber + 1]
        : refClpC;

    const quint16 *candRawU = haveUp
        ? (frameBuffer.rawbuffer.data() + (lineNumber - 1) * fieldWidth)
        : candRawC;
    const double *candClpU = haveUp
        ? frameBuffer.clpbuffer[1].pixel[lineNumber - 1]
        : candClpC;

    const quint16 *candRawD = haveDn
        ? (frameBuffer.rawbuffer.data() + (lineNumber + 1) * fieldWidth)
        : candRawC;
    const double *candClpD = haveDn
        ? frameBuffer.clpbuffer[1].pixel[lineNumber + 1]
        : candClpC;

    auto getLuma = [&](const quint16 *raw, const double *chroma, int idx) -> double {
        return static_cast<double>(raw[idx]) - chroma[idx];
    };

    const int r0 = clampH(refH - 1);
    const int r1 = clampH(refH);
    const int r2 = clampH(refH + 1);

    const int c0 = clampH(h - 1);
    const int c1 = hh;
    const int c2 = clampH(h + 1);

    const double dC0 = std::fabs(getLuma(refRawC, refClpC, r0) -
                                 getLuma(candRawC, candClpC, c0));
    const double dC1 = std::fabs(getLuma(refRawC, refClpC, r1) -
                                 getLuma(candRawC, candClpC, c1));
    const double dC2 = std::fabs(getLuma(refRawC, refClpC, r2) -
                                 getLuma(candRawC, candClpC, c2));

    const double dU = std::fabs(getLuma(refRawU, refClpU, r1) -
                                getLuma(candRawU, candClpU, c1));
    const double dD = std::fabs(getLuma(refRawD, refClpD, r1) -
                                getLuma(candRawD, candClpD, c1));

    const double yPen = ((dC0 + dC1 + dC2 + dU + dD) / 5.0) * invIreScale;

    // --- Chroma/2D Penalty ---
    //
    // Preserve the existing chroma disagreement evidence separately instead of
    // collapsing it into result.penalty only. This lets getBestCandidate treat
    // low-yPen/high-iqPen as "picture-compatible but chroma-grid divergent",
    // which is the compact-checkerboard repair case.
    const int fRef  = carrierLineFlip(refLineNumber);
    const int fCand = carrierLineFlip(lineNumber);

    double iqPen =
        (std::fabs((fRef * refClpC[r0]) - (fCand * candClpC[c0])) * 0.5 +
         std::fabs((fRef * refClpC[r1]) - (fCand * candClpC[c1])) * 1.0 +
         std::fabs((fRef * refClpC[r2]) - (fCand * candClpC[c2])) * 0.5) / 2.0;

    iqPen = (iqPen * invIreScale) * 0.28 * configuration.chromaWeight;

    double penalty = yPen + iqPen + adjustPenalty;
    if (penalty > configuration.candidatePenaltyHardMax)
        penalty = configuration.candidatePenaltyHardMax;

    result.yPen    = yPen;
    result.iqPen   = iqPen;
    result.penalty = penalty;

    return result;
}
// getBestY - Dedicated 3D residual-Y election/blend.
//
// The temporal vote should see the best local luma model each frame has
// already produced, not the stale raw - clpbuffer[1] baseline. Priority:
//   1. carrier-retracted Y, when residualColor + carrierRetractedValid
//   2. coherent residual comb Y = raw - lockedCarrierComposite
//   3. plain 2D comb Y = raw - clpbuffer[1]
double Comb::FrameBuffer::getBestY(qint32 line, qint32 h,
                                   const FrameBuffer &prev, const FrameBuffer &next) const
{
    const auto &T = configuration.tunables;
    const int fw  = videoParameters.fieldWidth;
    const int left = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;

    // Helper to extract the best pre-output luma available from a framebuffer.
    auto getY = [&](const FrameBuffer& fb, int ln, int x) -> double {
        if (ln < videoParameters.firstActiveFrameLine ||
            ln >= videoParameters.lastActiveFrameLine)
            return 0.0;

        const int xx = std::clamp(x, left, right - 1);
        const int rel = xx - left;
        const double raw = (double)fb.rawbuffer.data()[ln * fw + xx];

        if (fb.configuration.residualColor && fb.carrierRetractedValid) {
            const float *retRow = fb.carrierRetracted_line(ln);
            if (retRow) {
                const double r = (double)retRow[rel];
                if (std::isfinite(r))
                    return r;
            }
        }

        if (fb.configuration.residualVideo) {
            const double *carrierRow = fb.lockedCarrierComposite_line(ln);
            if (carrierRow) {
                const double c = carrierRow[rel];
                if (std::isfinite(c))
                    return raw - c;
            }
        }

        const double clp = fb.clpbuffer[1].pixel[ln][xx];
        return std::isfinite(clp) ? (raw - clp) : raw;
    };

    double yCurr = getY(*this, line, h);
    double yPrev = getY(prev, line, h);
    double yNext = getY(next, line, h);

    // --- Neighbor Shaping ---
    // Calculate a "Spatial Consensus" for the current pixel using N/S/E/W
    // from the current frame's best local luma model, then see which temporal
    // candidates fit that spatial context.

    // Simple Consensus: Median of Current, Up, Down, Left, Right
    double valC = yCurr;
    double valL = (h > videoParameters.activeVideoStart) ? getY(*this, line, h-1) : valC;
    double valR = (h < videoParameters.activeVideoEnd-1) ? getY(*this, line, h+1) : valC;
    double valU = getY(*this, line - 1, h); // safe, returns 0.0 if OOB
    double valD = getY(*this, line + 1, h);

    // Using median3 primitive from comb_math.h
    double medH = median3(valL, valC, valR);
    double medV = median3(valU, valC, valD);
    double spatialTarget = 0.5 * (medH + medV);

    double shapeStr = T.NEIGHBOR_SHAPE_STRENGTH;
    double penPrev  = std::fabs(yPrev - spatialTarget) * shapeStr;
    double penNext  = std::fabs(yNext - spatialTarget) * shapeStr;

    // --- Agreement Scores ---
    double baseRadius = T.AGREEMENT_REWARD_RADIUS_IRE * irescale;
    double vetoLimit  = T.deviationThreshold * irescale;

    double diffPrev = std::fabs(yPrev - yCurr) + penPrev;
    double diffNext = std::fabs(yNext - yCurr) + penNext;

    // Weights (0.0 to 1.0)
    auto calcWeight = [&](double diff) -> double {
        if (diff < baseRadius) return 1.0;
        if (diff > vetoLimit)  return 0.0;
        return 1.0 - (diff - baseRadius) / (vetoLimit - baseRadius);
    };

    double wP = calcWeight(diffPrev);
    double wN = calcWeight(diffNext);

    // --- Mode Selection ---
    if (T.RESIDUAL_Y_ELECTION) {
        // Winner-Take-All
        // If both are good, pick the one closest to spatial target (least noisy)
        if (wP > 0.5 && wN > 0.5) {
            return (penPrev < penNext) ? yPrev : yNext;
        }
        if (wP > 0.5) return yPrev;
        if (wN > 0.5) return yNext;
        return yCurr;
    } else {
        // Blend (Median-ish)
        // Note: Weighted average of Curr, Prev, Next.
        // If weights are 1, it's (P+C+N)/3.
        double totalW = 1.0 + wP + wN;
        return (yCurr * 1.0 + yPrev * wP + yNext * wN) / totalW;
    }
}
