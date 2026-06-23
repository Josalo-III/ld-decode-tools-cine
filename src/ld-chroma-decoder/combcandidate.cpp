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

    if (wantFieldB || wantFrame)
        ensureWidth(tapLine.hLumaDeltaIRE);

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
        ensureWidth(tapLine.centerChromaT);
        for (int rel = 0; rel < width; ++rel) {
            const CombTapScalar &s = tapLine.tap0[rel];
            const double envC = std::hypot(s.comp, s.symMag);
            tapLine.centerEnvelope[rel] = envC;
            tapLine.centerChromaT[rel] =
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
                     lddecode::CombReachUse::FieldScalarAverage);

            fillPair(tapLine.tapD2,
                     tapLine.lnD2,
                     tapLine.haveD2,
                     tapLine.pairD2,
                     scalarSource,
                     lddecode::CombReachUse::FieldScalarAverage);
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
                if (width >= 5) {
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
            const double cMag = std::hypot(cIf, cQf);
            const double uMag = std::hypot(uIf, uQf);
            const double dMag = std::hypot(dIf, dQf);
            const CombContentReach::InterfieldIQReachFloor floor =
                CombContentReach::interfieldIQReachFloor(
                    cIf, cQf, uIf, uQf, dIf, dQf,
                    true, true, minChromaIRE, 1.0,
                    cMag, uMag, dMag);

            double upGate = std::max(upPair[rel].reachLegalGate, floor.up);
            double dnGate = std::max(dnPair[rel].reachLegalGate, floor.down);

            double impulseExempt = 0.0;
            if (frameLuma0) {
                const int l2 = std::max(0, rel - 2);
                const int r2 = std::min(width - 1, rel + 2);
                const double cIRE_luma = frameLuma0[rel] * invI;
                const double lIRE_luma = frameLuma0[l2] * invI;
                const double rIRE_luma = frameLuma0[r2] * invI;
                const double surround = 0.5 * (lIRE_luma + rIRE_luma);
                const double peakIRE = std::fabs(cIRE_luma - surround);
                const double flanksAgree = std::fabs(lIRE_luma - rIRE_luma);
                const double IMPULSE_LO_IRE = 5.0;
                const double IMPULSE_HI_IRE = 15.0;
                if (flanksAgree < 4.0) {
                    impulseExempt = std::clamp(
                        (peakIRE - IMPULSE_LO_IRE) /
                        (IMPULSE_HI_IRE - IMPULSE_LO_IRE),
                        0.0, 1.0);
                }
                scratch_impulseExempt[rel] = impulseExempt;
            }

            if (haveCloseLuma && bevelPenalty > 0.0) {
                const auto mcNear = CombContentReach::evaluateMovingCoarseContour(
                    frameLuma0[rel]  * invI,
                    frameLumaU1[rel] * invI, frameLumaD1[rel] * invI,
                    frameLumaU2[rel] * invI, frameLumaD2[rel] * invI,
                    true, true, true, true,
                    soft, hard);

                if (mcNear.valid) {
	                    const double chromaWeight =
	                        (rel < (int)tapLine.centerChromaT.size())
	                            ? tapLine.centerChromaT[rel]
	                            : std::clamp(
	                                (std::hypot(tapLine.tap0[rel].comp,
	                                            tapLine.tap0[rel].symMag) * invI - 2.0) / 8.0,
	                                0.0, 1.0);
                    const double bevelGate =
                        std::clamp(1.0 - bevelPenalty * chromaWeight *
                                   (1.0 - std::clamp(mcNear.straightness, 0.0, 1.0)),
                                   0.0, 1.0);
                    const double effectiveGate =
                        bevelGate + (1.0 - bevelGate) * impulseExempt;
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
	                        (rel < (int)tapLine.centerChromaT.size())
	                            ? tapLine.centerChromaT[rel]
	                            : std::clamp(
	                                (std::hypot(tapLine.tap0[rel].comp,
	                                            tapLine.tap0[rel].symMag) * invI - 2.0) / 8.0,
	                                0.0, 1.0);
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
void Comb::FrameBuffer::computeContourFieldLine(int lineNumber,
                                          double *outFieldLine,
                                          double  *outGate)
{
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;

    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;

    if (width <= 0 || !outFieldLine) {
        if (outGate && width > 0)
            std::fill(outGate, outGate + width, 1.0f);
        return;
    }

    if (lineNumber < first || lineNumber >= last) {
        std::fill(outFieldLine, outFieldLine + width, 0.0);
        if (outGate) std::fill(outGate, outGate + width, 1.0f);
        return;
    }

    if (outGate)
        std::fill(outGate, outGate + width, 1.0f);

    auto clampSameFieldLine = [&](int ln) -> int {
        // For intrafield sampling we must stay on the same field parity as lineNumber.
        // Plain clamping can jump to the opposite field at the top/bottom edges.
        const int parity = lineNumber & 1;

        ln = std::clamp(ln, first, last - 1);

        if ((ln & 1) != parity) {
            // Prefer stepping inward rather than outward.
            if (ln + 1 < last && ((ln + 1) & 1) == parity)
                ln = ln + 1;
            else if (ln - 1 >= first && ((ln - 1) & 1) == parity)
                ln = ln - 1;
        }

        return ln;
    };

    const int ln0   = clampSameFieldLine(lineNumber);
    const int lnUp2 = clampSameFieldLine(lineNumber - 2);
    const int lnDn2 = clampSameFieldLine(lineNumber + 2);
    const int lnUp4 = clampSameFieldLine(lineNumber - 4);
    const int lnDn4 = clampSameFieldLine(lineNumber + 4);

    const double *row0   = nullptr;
    const double *rowUp2 = nullptr;
    const double *rowDn2 = nullptr;
    const double *rowUp4 = nullptr;
    const double *rowDn4 = nullptr;

    if (configuration.phaseCompensation) {
        auto getRow = [&](int ln) -> const double* {
            if (ln < first || ln >= last)
                return nullptr;

            return locked1DSource_line(ln);
        };

        row0   = getRow(ln0);
        rowUp2 = getRow(lnUp2);
        rowDn2 = getRow(lnDn2);
        rowUp4 = getRow(lnUp4);
        rowDn4 = getRow(lnDn4);
    } else {
        row0   = clpbuffer[0].pixel[ln0]   + left;
        rowUp2 = clpbuffer[0].pixel[lnUp2] + left;
        rowDn2 = clpbuffer[0].pixel[lnDn2] + left;
        rowUp4 = clpbuffer[0].pixel[lnUp4] + left;
        rowDn4 = clpbuffer[0].pixel[lnDn4] + left;
    }

    if (!row0) {
        std::fill(outFieldLine, outFieldLine + width, 0.0);
        if (outGate) std::fill(outGate, outGate + width, 1.0f);
        return;
    }

    // If any vertical support row is unavailable, preserve the local source.
    // This only removes synthetic zero collapse; it does not create a new
    // partial-comb behavior.
    if (!rowUp2 || !rowDn2 || !rowUp4 || !rowDn4) {
        std::copy(row0, row0 + width, outFieldLine);
        if (outGate) std::fill(outGate, outGate + width, 0.0f);
        return;
    }

    const auto  &T    = configuration.tunables;
    const double invI = this->invIreScale;

    // Phase relationship range, in source units.
    const double kRange = T.FIELD_K_RANGE_IRE * irescale;
    const double invK   = (kRange > 1e-9) ? (1.0 / kRange) : 0.0;

    // Luma-edge exclusion for far reach. This prevents reaching across
    // disparate vertical regions.
    const double EDGE_SOFT_IRE = 6.0;
    const double EDGE_HARD_IRE = 14.0;

    auto edgeGateAt = [&](int rel) -> double {
        const int rm1 = (rel > 0) ? (rel - 1) : 0;
        const int rp1 = (rel + 1 < width) ? (rel + 1) : (width - 1);

        const double eIRE = std::fabs(row0[rp1] - row0[rm1]) * invI;

        if (eIRE <= EDGE_SOFT_IRE) return 1.0;
        if (eIRE >= EDGE_HARD_IRE) return 0.0;

        double t = (eIRE - EDGE_SOFT_IRE) / (EDGE_HARD_IRE - EDGE_SOFT_IRE);
        t = std::clamp(t, 0.0, 1.0);
        return 1.0 - t;
    };

    auto phaseDiffMetric = [&](double C0,
                               double sym0,
                               double Cn,
                               double symn) -> double
    {
        double k = 0.0;

        k  = std::fabs(std::fabs(C0) - std::fabs(Cn));
        k += std::fabs(sym0 - symn);

        // Small bonus for strong signal; helps avoid weak/noisy toggles.
        k -= (std::fabs(C0) + std::fabs(Cn)) * 0.10;

        if (k < 0.0)
            k = 0.0;

        return k;
    };

    for (int h = left; h < right; ++h) {
        const int rel = h - left;

        const int rm1 = (rel > 0) ? (rel - 1) : 0;
        const int rp1 = (rel + 1 < width) ? (rel + 1) : (width - 1);

        // Center and symmetric lateral context.
        const double C    = row0[rel];
        const double C_m1 = row0[rm1];
        const double C_p1 = row0[rp1];

        const double symCur =
            0.5 * (std::fabs(C_m1) + std::fabs(C_p1));

        // Near samples, ±2 same-field.
        const double U2    = rowUp2[rel];
        const double D2    = rowDn2[rel];
        const double U2_m1 = rowUp2[rm1];
        const double U2_p1 = rowUp2[rp1];
        const double D2_m1 = rowDn2[rm1];
        const double D2_p1 = rowDn2[rp1];

        const double symU2 =
            0.5 * (std::fabs(U2_m1) + std::fabs(U2_p1));

        const double symD2 =
            0.5 * (std::fabs(D2_m1) + std::fabs(D2_p1));

        // Far samples, ±4 same-field.
        const double U4    = rowUp4[rel];
        const double D4    = rowDn4[rel];
        const double U4_m1 = rowUp4[rm1];
        const double U4_p1 = rowUp4[rp1];
        const double D4_m1 = rowDn4[rm1];
        const double D4_p1 = rowDn4[rp1];

        const double symU4 =
            0.5 * (std::fabs(U4_m1) + std::fabs(U4_p1));

        const double symD4 =
            0.5 * (std::fabs(D4_m1) + std::fabs(D4_p1));

        // ------------------------------------------------------------
        // Near weights, same original logic.
        // ------------------------------------------------------------
        const double kp2 = phaseDiffMetric(C, symCur, U2, symU2);
        const double kn2 = phaseDiffMetric(C, symCur, D2, symD2);

        double wUp2 = (kRange > 1e-9) ? (1.0 - kp2 * invK) : 1.0;
        double wDn2 = (kRange > 1e-9) ? (1.0 - kn2 * invK) : 1.0;

        wUp2 = std::clamp(wUp2, 0.0, 1.0);
        wDn2 = std::clamp(wDn2, 0.0, 1.0);

        double sc2 = 1.0;
        bool haveNear = false;

        if (wUp2 > 0.0 || wDn2 > 0.0) {
            if (wDn2 > 3.0 * wUp2)
                wUp2 = 0.0;
            else if (wUp2 > 3.0 * wDn2)
                wDn2 = 0.0;

            const double denom = wUp2 + wDn2;

            if (denom > 1e-9) {
                sc2 = 2.0 / denom;
                if (sc2 < 1.0)
                    sc2 = 1.0;

                haveNear = true;
            } else {
                wUp2 = 0.0;
                wDn2 = 0.0;
            }
        } else {
            // If up/down are similar to each other, allow both.
            const double dMag  = std::fabs(std::fabs(U2) - std::fabs(D2));
            const double sumUD = std::fabs(U2 + D2);

            if (dMag - std::fabs(sumUD * 0.2) <= 0.0) {
                wUp2 = 1.0;
                wDn2 = 1.0;
                sc2 = 1.0;
                haveNear = true;
            } else {
                wUp2 = 0.0;
                wDn2 = 0.0;
            }
        }

        // ------------------------------------------------------------
        // Far weights, same original logic: ramped by near confidence and
        // horizontal edge gate.
        // ------------------------------------------------------------
        const double kp4 = phaseDiffMetric(C, symCur, U4, symU4);
        const double kn4 = phaseDiffMetric(C, symCur, D4, symD4);

        double wUp4 = (kRange > 1e-9) ? (1.0 - kp4 * invK) : 1.0;
        double wDn4 = (kRange > 1e-9) ? (1.0 - kn4 * invK) : 1.0;

        wUp4 = std::clamp(wUp4, 0.0, 1.0);
        wDn4 = std::clamp(wDn4, 0.0, 1.0);

        const double nearConfUp = wUp2;
        const double nearConfDn = wDn2;

        const double eGate = edgeGateAt(rel);

        wUp4 *= nearConfUp * eGate;
        wDn4 *= nearConfDn * eGate;

        const double FAR_SCALE = 0.65;
        wUp4 *= FAR_SCALE;
        wDn4 *= FAR_SCALE;

        // ------------------------------------------------------------
        // Combine near and far components.
        //
        // Original behavior is preserved when any component exists.
        // Only the true no-answer collapse is changed from zero to C.
        // ------------------------------------------------------------
        double tc = 0.0;
        bool haveAnswer = false;

        if (haveNear && (wUp2 > 0.0 || wDn2 > 0.0)) {
            double t2  = (C - U2) * wUp2 * sc2;
            t2        += (C - D2) * wDn2 * sc2;
            t2        *= 0.25;

            tc += t2;
            haveAnswer = true;
        }

        if (wUp4 > 0.0 || wDn4 > 0.0) {
            const double denom = wUp4 + wDn4;

            if (denom > 1e-9) {
                double sc4 = 2.0 / denom;
                if (sc4 < 1.0)
                    sc4 = 1.0;

                double t4  = (C - U4) * wUp4 * sc4;
                t4        += (C - D4) * wDn4 * sc4;
                t4        *= 0.25;

                tc += t4;
                haveAnswer = true;
            }
        }

        if (!haveAnswer)
            tc = C;

        if (!std::isfinite(tc))
            tc = C;

        outFieldLine[rel] = tc;

        // Gate for scorer: how confident is A here?
        // Use near confidence primarily, with far only if present.
        double gateA = std::max(wUp2, wDn2);
        gateA = std::max(gateA, 0.5 * std::max(wUp4, wDn4));
        gateA = std::clamp(gateA, 0.0, 1.0);

        if (!haveAnswer)
            gateA = 0.0;

        if (outGate)
            outGate[rel] = (float)gateA;
    }
}

void Comb::FrameBuffer::computeContourFieldLine(const CombTapLine &tapLine,
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

        // Per-side coarse-luma-edge facts, cribbed from Field B. eUp/eDn rise
        // where the center crosses a horizontal luma step on that side; chromaT
        // weights the bevel cede by chroma presence (envelope, ripple-free).
	        const double chromaT =
	            (rel < (int)tapLine.centerChromaT.size())
	                ? tapLine.centerChromaT[rel]
	                : std::clamp((std::hypot(C, tapLine.tap0[rel].symMag) * invI - 2.0) / 8.0,
	                             0.0, 1.0);
        double eUp = 0.0, eDn = 0.0;
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
                (LUMA_EDGE_HI_IRE - LUMA_EDGE_LO_IRE), 0.0, 1.0);
            eDn = std::clamp(
                (dDnIRE - LUMA_EDGE_LO_IRE) /
                (LUMA_EDGE_HI_IRE - LUMA_EDGE_LO_IRE), 0.0, 1.0);
            // Back the vertical comb off the side that crosses a luma edge.
            // This is the one-sided-bevel handler; the min()-based cede below
            // only catches the both-sides case.
            wUp2 *= (1.0 - eUp);
            wDn2 *= (1.0 - eDn);
        }

        double boundaryCede = 0.0;
        if (rel < (int)tapLine.pairU2.size() &&
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

                    if (hardPreferUp) {
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

        double sc2 = 1.0;
        if ((wUp2 > 0.0) || (wDn2 > 0.0)) {
            if (wDn2 > 3.0 * wUp2)      wUp2 = 0.0;
            else if (wUp2 > 3.0 * wDn2) wDn2 = 0.0;

            const double denom = wUp2 + wDn2;
            if (denom > 1e-9) {
                sc2 = 2.0 / denom;
                if (sc2 < 1.0) sc2 = 1.0;
            } else {
                wUp2 = wDn2 = 0.0;
            }
        } else {
            double dMag  = std::fabs(std::fabs(Cup2) - std::fabs(Cdn2));
            double sumUD = std::fabs(Cup2 + Cdn2);
            if (dMag - std::fabs(sumUD * 0.2) <= 0.0) {
                wUp2 = reachUp2;
                wDn2 = reachDn2;
                sc2 = 1.0;
            } else {
                wUp2 = wDn2 = 0.0;
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
        } else {
            tc = C;
        }

        // Zipper defense cribbed from Field B (computeSimpleFieldLine): the
        // bevel zipper is vertical combing across a horizontal luma edge. After
        // the per-side weight suppression above, cede the comb back toward the
        // local center C across both-sided coarse-luma edges and non-straight
        // bevels, weighted by chroma presence — "chroma may only sharpen a
        // break, never originate one." Field A already inherits the bevel REACH
        // throttle through pairU2/pairD2.reachGate; these are the additional
        // luma-edge defenses Field B needed on top of it.
        if (combed) {
            const double lumaEdgeCede = std::min(eUp, eDn);

            double bevelCede = 0.0;
            if (rel < (int)tapLine.movingCoarseContour.size() &&
                tapLine.movingCoarseContour[rel].valid &&
                rel < (int)tapLine.hLumaDeltaIRE.size())
            {
                const auto &mc = tapLine.movingCoarseContour[rel];
                const double hEdge = std::clamp(
                    (tapLine.hLumaDeltaIRE[rel] - 0.30 * hEdgeThreshIRE) /
                    (0.70 * hEdgeThreshIRE), 0.0, 1.0);
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
void Comb::FrameBuffer::computeSimpleFieldLine(int lineNumber,
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
    computeSimpleFieldLine(tapLine, outFieldLine, outReasonLine);
}

void Comb::FrameBuffer::computeSimpleFieldLine(const CombTapLine &tapLine,
                                               double *outFieldLine,
                                               std::uint8_t *outReasonLine)
{
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;

    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;

    const int lineNumber = tapLine.cacheLine;

    if (width <= 0 || !outFieldLine)
        return;

    if (lineNumber < first || lineNumber >= last) {
        std::fill(outFieldLine, outFieldLine + width, 0.0);
        if (outReasonLine) std::fill(outReasonLine, outReasonLine + width, FieldBReasonNone);
        return;
    }

    if ((int)tapLine.tap0.size() < width) {
        std::fill(outFieldLine, outFieldLine + width, 0.0);
        if (outReasonLine) std::fill(outReasonLine, outReasonLine + width, FieldBReasonNone);
        return;
    }

    // Missing vertical support rows: preserve the local 1D chroma estimate.
    // This avoids synthetic zero chroma without changing normal Field B geometry.
    if (!tapLine.haveU2 || !tapLine.haveD2 ||
        (int)tapLine.tapU2.size() < width ||
        (int)tapLine.tapD2.size() < width)
    {
        for (int rel = 0; rel < width; ++rel)
            outFieldLine[rel] = tapLine.tap0[rel].comp;
        if (outReasonLine) std::fill(outReasonLine, outReasonLine + width, FieldBReasonCenter);
        return;
    }

    const auto &T = configuration.tunables;

    const double kRange = T.FIELD_K_RANGE_IRE * irescale;
    const double invK   = (kRange > 1e-9) ? (1.0 / kRange) : 0.0;
    const double hEdgeThreshIRE = std::max(1.0, T.FIELD_LUMA_EDGE_THRESH_IRE);
    auto softenDominantWeights = [](double &wA, double &wB) {
        if (wA <= 0.0 || wB <= 0.0)
            return;

        const double strong = std::max(wA, wB);
        const double weak = std::min(wA, wB);
        if (strong <= 1e-9)
            return;

        const double balance = weak / strong;
        const double minBalance = 1.0 / 3.0;
        if (balance >= minBalance)
            return;

        // Field B zippering is driven more by abrupt side dropouts than by
        // genuinely one-sided evidence. Lift the weaker side toward the 3:1
        // floor instead of zeroing it so saturation fallback can cede smoothly.
        const double liftedWeak = strong * (minBalance + (1.0 - minBalance) * balance);

        if (wA >= wB)
            wB = liftedWeak;
        else
            wA = liftedWeak;
    };

    for (int rel = 0; rel < width; ++rel) {
        std::uint8_t reason = FieldBReasonNone;
        const CombTapScalar &tapC = tapLine.tap0[rel];
        const CombTapScalar &tapUp = tapLine.tapU2[rel];
        const CombTapScalar &tapDn = tapLine.tapD2[rel];

        const double C   = tapC.comp;
        const double Cup = tapUp.comp;
        const double Cdn = tapDn.comp;

        // Phase-flat chroma envelopes.  At 4fsc the center sample sits on one
        // carrier axis and its two horizontal neighbours on the orthogonal axis
        // (C_p1 = -C_m1), so hypot(on-axis, orthogonal-axis) = sqrt(I^2+Q^2),
        // independent of carrier phase.  Building the election from envelopes
        // instead of |C| removes the 2-pixel carrier-rate oscillation that made
        // the comb engage/disengage in alternating blocks (the zipper).  Hue/
        // phase legality is NOT measured here — that stays with reachGate below.
        const double envC =
            (rel < (int)tapLine.centerEnvelope.size())
                ? tapLine.centerEnvelope[rel]
                : std::hypot(C, tapC.symMag);
        const double envUp = std::hypot(Cup, tapUp.symMag);
        const double envDn = std::hypot(Cdn, tapDn.symMag);

        double kp = std::fabs(envC - envUp) - 0.10 * (envC + envUp);
        double kn = std::fabs(envC - envDn) - 0.10 * (envC + envDn);

        if (kp < 0.0) kp = 0.0;
        if (kn < 0.0) kn = 0.0;

        const double hEdgeForSat =
            (rel < (int)tapLine.hLumaDeltaIRE.size())
                ? std::clamp(
                    (tapLine.hLumaDeltaIRE[rel] - 0.35 * hEdgeThreshIRE) /
                    (0.65 * hEdgeThreshIRE),
                    0.0,
                    1.0)
                : 0.0;
        const double chromaT =
            (rel < (int)tapLine.centerChromaT.size())
                ? tapLine.centerChromaT[rel]
                : std::clamp((envC * invIreScale - 2.0) / 8.0, 0.0, 1.0);

        if (chromaT > 0.0 &&
            rel < (int)tapLine.coarse0IRE.size() &&
            rel < (int)tapLine.coarseU2IRE.size() &&
            rel < (int)tapLine.coarseD2IRE.size())
        {
            const double lc = tapLine.coarse0IRE[rel] * irescale;
            const double lu = tapLine.coarseU2IRE[rel] * irescale;
            const double ld = tapLine.coarseD2IRE[rel] * irescale;
            double lumaKp = std::fabs(lc - lu);
            double lumaKn = std::fabs(lc - ld);
            lumaKp = std::max(lumaKp - (lc + lu) * 0.10, 0.0);
            lumaKn = std::max(lumaKn - (lc + ld) * 0.10, 0.0);
            kp = kp * (1.0 - chromaT) + lumaKp * chromaT;
            kn = kn * (1.0 - chromaT) + lumaKn * chromaT;
        }

        const double satRelax = chromaT * (1.0 - 0.55 * hEdgeForSat);
        const double effectiveKRange = kRange * (1.0 + 0.75 * satRelax);
        const double localInvK =
            (effectiveKRange > 1e-9) ? (1.0 / effectiveKRange) : 0.0;

        double wUp = (effectiveKRange > 1e-9) ? (1.0 - kp * localInvK) : 1.0;
        double wDn = (effectiveKRange > 1e-9) ? (1.0 - kn * localInvK) : 1.0;

        wUp = std::clamp(wUp, 0.0, 1.0);
        wDn = std::clamp(wDn, 0.0, 1.0);

        if (rel < (int)tapLine.pairU2.size())
            wUp *= tapLine.pairU2[rel].reachGate;
        if (rel < (int)tapLine.pairD2.size())
            wDn *= tapLine.pairD2[rel].reachGate;

        double lumaEdgeUp = 0.0;
        double lumaEdgeDn = 0.0;
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
            lumaEdgeUp = std::clamp(
                (dUpIRE - LUMA_EDGE_LO_IRE) /
                (LUMA_EDGE_HI_IRE - LUMA_EDGE_LO_IRE),
                0.0, 1.0);
            lumaEdgeDn = std::clamp(
                (dDnIRE - LUMA_EDGE_LO_IRE) /
                (LUMA_EDGE_HI_IRE - LUMA_EDGE_LO_IRE),
                0.0, 1.0);
            wUp *= (1.0 - lumaEdgeUp);
            wDn *= (1.0 - lumaEdgeDn);
        }

        double boundaryCede = 0.0;
        if (rel < (int)tapLine.pairU2.size() &&
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
                const double scalarUpDn = std::fabs(Cup - Cdn) * invIreScale;
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
                const double kBest = std::min(kp, kn);
                const double kWorst = std::max(kp, kn);
                const double kRatio =
                    (kWorst > 1e-9) ? (kBest / kWorst) : 1.0;

                const double matchIRE = 3.5;
                const double betweenIRE = 6.0;
                const double edgeUdIRE = 8.0;

                if (dUpDnIRE > edgeUdIRE) {
                    const bool preferUp = (dUp0IRE <= dDn0IRE);
                    const bool hardPreferUp =
                        preferUp &&
                        ((dUp0IRE < matchIRE && dDn0IRE > betweenIRE) ||
                         (bestDiffIRE < 4.5 && diffGapIRE > 2.5 && diffRatio < 0.55) ||
                         (bestDiffIRE < 5.0 && kRatio < 0.45));
                    const bool hardPreferDn =
                        !preferUp &&
                        ((dDn0IRE < matchIRE && dUp0IRE > betweenIRE) ||
                         (bestDiffIRE < 4.5 && diffGapIRE > 2.5 && diffRatio < 0.55) ||
                         (bestDiffIRE < 5.0 && kRatio < 0.45));

                    if (hardPreferUp) {
                        wDn = 0.0;
                        wUp = std::max(wUp, 0.40 + 0.60 * hEdge);
                        reason = FieldBReasonBoundaryUp;
                    } else if (hardPreferDn) {
                        wUp = 0.0;
                        wDn = std::max(wDn, 0.40 + 0.60 * hEdge);
                        reason = FieldBReasonBoundaryDown;
                    } else {
                        const double sideGap = diffGapIRE;
                        const double sideGapT =
                            std::clamp((sideGap - 1.0) / 4.0, 0.0, 1.0);
                        const double cedeT =
                            std::clamp((std::min(dUp0IRE, dDn0IRE) - 3.0) / 6.0,
                                       0.0,
                                       1.0);

                        if (preferUp)
                            wDn *= std::max(0.0, 1.0 - 0.85 * hEdge * (0.35 + 0.65 * sideGapT));
                        else
                            wUp *= std::max(0.0, 1.0 - 0.85 * hEdge * (0.35 + 0.65 * sideGapT));

                        boundaryCede = std::max(boundaryCede, 0.70 * hEdge * cedeT);
                        if (boundaryCede > 0.0 && reason == FieldBReasonNone)
                            reason = FieldBReasonBoundaryCede;
                    }
                }
            }
        }

        if (rel < (int)tapLine.movingCoarseContour.size() &&
            tapLine.movingCoarseContour[rel].valid &&
            rel < (int)tapLine.hLumaDeltaIRE.size())
        {
            const auto &mc = tapLine.movingCoarseContour[rel];
            const double hEdge =
                std::clamp(
                    (tapLine.hLumaDeltaIRE[rel] - 0.30 * hEdgeThreshIRE) /
                    (0.70 * hEdgeThreshIRE),
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
            const double bevelCede =
                std::clamp(T.FIELD_B_BEVEL_CEDE_STRENGTH * bevelRisk, 0.0, 1.0);
            boundaryCede = std::max(boundaryCede, bevelCede);
            if (bevelCede > 0.0 && reason == FieldBReasonNone)
                reason = FieldBReasonBoundaryCede;
        }

        double sc = 1.0;
        bool haveAnswer = false;

        if (wUp > 0.0 || wDn > 0.0) {
            softenDominantWeights(wUp, wDn);

            const double denom = wUp + wDn;

            if (denom > 1e-9) {
                sc = 2.0 / denom;
                if (sc < 1.0)
                    sc = 1.0;

                haveAnswer = true;
            } else {
                wUp = 0.0;
                wDn = 0.0;
            }
        } else {
            double reviveStrength = 0.0;
            double reviveUp = 0.0;
            double reviveDn = 0.0;

            if (rel < (int)tapLine.movingCoarseContour.size() &&
                tapLine.movingCoarseContour[rel].valid &&
                rel < (int)tapLine.coarse0IRE.size() &&
                rel < (int)tapLine.coarseU2IRE.size() &&
                rel < (int)tapLine.coarseD2IRE.size())
            {
                const auto &mc = tapLine.movingCoarseContour[rel];
                const double coarseCenter = tapLine.coarse0IRE[rel];
                const double coarseUp = tapLine.coarseU2IRE[rel];
                const double coarseDn = tapLine.coarseD2IRE[rel];
                const double dUpCoarse = std::fabs(coarseCenter - coarseUp);
                const double dDnCoarse = std::fabs(coarseCenter - coarseDn);
                const double denom = dUpCoarse + dDnCoarse;
                const double coarseBalance =
                    (denom > 1e-9)
                        ? (1.0 - std::fabs(dUpCoarse - dDnCoarse) / denom)
                        : 1.0;
                const double coarseStraight =
                    std::clamp(mc.straightness, 0.0, 1.0);
                const double coarseCurvFit =
                    std::clamp(1.0 - mc.curvMidIRE / 8.0, 0.0, 1.0);

                reviveStrength =
                    coarseBalance *
                    (0.60 * coarseStraight + 0.40 * coarseCurvFit);

                reviveUp =
                    ((rel < (int)tapLine.pairU2.size())
                         ? tapLine.pairU2[rel].reachGate
                         : 1.0) *
                    (0.35 + 0.65 * std::clamp(mc.upTrust, 0.0, 1.0));
                reviveDn =
                    ((rel < (int)tapLine.pairD2.size())
                         ? tapLine.pairD2[rel].reachGate
                         : 1.0) *
                    (0.35 + 0.65 * std::clamp(mc.downTrust, 0.0, 1.0));
            } else {
                const double dMag  = std::fabs(std::fabs(Cup) - std::fabs(Cdn));
                const double sumUD = std::fabs(Cup + Cdn);
                reviveStrength = (dMag - std::fabs(sumUD * 0.2) <= 0.0) ? 1.0 : 0.0;
                reviveUp = (rel < (int)tapLine.pairU2.size())
                    ? tapLine.pairU2[rel].reachGate
                    : 1.0;
                reviveDn = (rel < (int)tapLine.pairD2.size())
                    ? tapLine.pairD2[rel].reachGate
                    : 1.0;
            }

            if (reviveStrength > 0.0) {
                wUp = reviveUp * reviveStrength;
                wDn = reviveDn * reviveStrength;
                sc = 1.0;
                haveAnswer = (wUp > 0.0 || wDn > 0.0);
                if (haveAnswer)
                    reason = (rel < (int)tapLine.movingCoarseContour.size() &&
                              tapLine.movingCoarseContour[rel].valid)
                        ? FieldBReasonReviveCoarse
                        : FieldBReasonReviveScalar;
            } else {
                wUp = 0.0;
                wDn = 0.0;
            }
        }

        double tc = 0.0;

        if (haveAnswer) {
            const double lumaEdgeCede = std::min(lumaEdgeUp, lumaEdgeDn);
            const double totalCede = std::max(boundaryCede, lumaEdgeCede);

            // Field B is a same-field ±2 scalar comb on locked1DSource. Cede
            // toward C only on real contour breaks (boundary or coarse luma
            // edge), per the doctrine "chroma may only sharpen a break, never
            // originate one."  A saturation-magnitude cede was removed: it
            // fired inside uniform saturated chroma columns and mixed the
            // modulated locked1DSource into the comb output per pixel, which
            // produced the 2fsc luma checkerboard after produceY's carrier
            // subtraction.
            tc  = (C - Cup) * wUp * sc;
            tc += (C - Cdn) * wDn * sc;
            tc *= 0.25;
            if (totalCede > 0.0) {
                tc = tc * (1.0 - totalCede) + C * totalCede;
                if (reason == FieldBReasonNone)
                    reason = FieldBReasonBoundaryCede;
            } else if (reason == FieldBReasonNone) {
                reason = FieldBReasonBlend;
            }
        } else {
            tc = C;
            reason = FieldBReasonCenter;
        }

        if (!std::isfinite(tc))
            tc = C;

        outFieldLine[rel] = tc;
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

void Comb::FrameBuffer::computeFrameIQFromPreparedVectors(
    int line,
    const std::vector<std::complex<double>> &centerIQ,
    std::vector<std::complex<double>> &upIQ,
    std::vector<std::complex<double>> &dnIQ,
    std::vector<std::complex<double>> &outFrameIQ,
    const std::vector<float> *tiOverride,
    const std::vector<float> *tqOverride,
    const CombTapLine *reachTapLine,
    bool allowSymmetricLeakCancel)
{
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;

    (void)allowSymmetricLeakCancel;

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

    auto tiLine = [&](int ln)->const float* {
        if (tiOverride && (int)tiOverride->size() >= (ln + 1) * demodWidth)
            return tiOverride->data() + static_cast<size_t>(ln) * demodWidth;
        const float *cached = locked1DTI4fsc_line(ln);
        if (!locked1DTI4fsc_flat.empty()) return cached;
        return demodTI4fsc_line(ln);
    };
    auto tqLine = [&](int ln)->const float* {
        if (tqOverride && (int)tqOverride->size() >= (ln + 1) * demodWidth)
            return tqOverride->data() + static_cast<size_t>(ln) * demodWidth;
        const float *cached = locked1DTQ4fsc_line(ln);
        if (!locked1DTQ4fsc_flat.empty()) return cached;
        return demodTQ4fsc_line(ln);
    };

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

        const double wFloor = 0.15;
        w = wFloor + (1.0 - wFloor) * w;

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

    // Frame B's own (interfield, +/-1) vertical column guard. It must be
    // measured across the adjacent opposite-field lines, NOT borrowed from the
    // same-field +/-2/+/-4 contour the Field B path uses. Chroma-cancelled luma
    // (lockedLumaSmooth) at line-1/line/line+1 gives a +/-1-native column read
    // below; null pointers (cache invalid / edge lines) leave columnStraight 0,
    // making the alien-cancel inert there.
    auto lumaSmoothRow = [&](int ln)->const double* {
        if (!configuration.phaseCompensation || !lockedLumaCacheValid ||
            lockedLumaSmooth_flat.empty() || demodWidth < width ||
            ln < first || ln >= last || ln >= demodLines)
            return nullptr;
        return lockedLumaSmooth_line(ln);
    };
    const double *luma1Up = lumaSmoothRow(line - 1);
    const double *luma1C  = lumaSmoothRow(line);
    const double *luma1Dn = lumaSmoothRow(line + 1);

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

        const bool haveUp = (aUp > 1e-9);
        const bool haveDn = (aDn > 1e-9);

        if (!haveUp && !haveDn) {
            outFrameIQ[x] = Z0;
            continue;
        }

        // --- Boundary limits for horizontal edges between disparate vertical regions ---
        // Detect: Up and Down are different, and center is not safely "same material" as both.
        // Behavior:
        //  - If center matches one side clearly -> pick that side.
        //  - If center is "between" (transition) -> avoid reaching (use only better side, but suppress its weight).
        bool useUp = haveUp;
        bool useDn = haveDn;
        double boundaryWeightScale = 1.0;
        // +/-1-native column support: low chroma-cancelled-luma curvature across
        // line-1/line/line+1 == a smooth interfield column. A horizontal edge
        // (top of a glyph) curves sharply here, so support falls and the
        // alien-cancel backs off, instead of trusting the same-field contour
        // which describes a different (+/-2) geometry.
        double columnStraight = 0.0;
        if (luma1Up && luma1C && luma1Dn) {
            const double curv1 =
                std::fabs(luma1Up[x] - 2.0 * luma1C[x] + luma1Dn[x]) * invI;
            columnStraight = combSmoothGate(curv1,
                                            T.FIELD_CONTOUR_SOFT_IRE,
                                            T.FIELD_CONTOUR_HARD_IRE);
        }
        // Bevel reach throttle, cribbed from Frame B. The ±1 reachGate
        // (pairU1/pairD1, built by applyFrameReachWithIQFloor) now carries the
        // moving-coarse bevel throttle in IRE units — the same gate that keeps
        // Frame B zipper-free. Frame A previously ignored it because the OLD
        // composite-domain gate over-fired and stood the comb down at the
        // saturated edges it exists to clean; in the IRE domain it discriminates
        // properly, so it is applied per-side to the interfield contributions
        // below (exactly as Field B applies pairU2/pairD2.reachGate).
        const double reachUp1 =
            (reachTapLine && x < (int)reachTapLine->pairU1.size())
                ? std::clamp(reachTapLine->pairU1[x].reachGate, 0.0, 1.0)
                : 1.0;
        const double reachDn1 =
            (reachTapLine && x < (int)reachTapLine->pairD1.size())
                ? std::clamp(reachTapLine->pairD1[x].reachGate, 0.0, 1.0)
                : 1.0;

        // Cache dUpDown_ire — reused below for the disagree gate (saves one
        // hypot per pixel on the haveUp && haveDn path).
        double dUpDown_ire = -1.0;

        if (haveUp && haveDn) {
            const double dUp0_ire = cmag(ZUpRaw - Z0) * invI;
            const double dDn0_ire = cmag(ZDnRaw - Z0) * invI;
            dUpDown_ire           = cmag(ZUpRaw - ZDnRaw) * invI;

            const double MATCH_IRE      = 3.5;   // center matches this side
            const double BETWEEN_IRE    = 6.0;   // center is far from this side
            const double EDGE_UD_IRE    = 8.0;   // up/down are materially different
            const double TRANS_SUPPRESS = 0.35;  // suppress transition-zone reach

            if (dUpDown_ire > EDGE_UD_IRE) {
                if (dUp0_ire < MATCH_IRE && dDn0_ire > BETWEEN_IRE) {
                    useUp = true;  useDn = false;
                } else if (dDn0_ire < MATCH_IRE && dUp0_ire > BETWEEN_IRE) {
                    useDn = true;  useUp = false;
                } else {
                    // Transition/boundary pixel: don't reach across.
                    // Pick the nearer side, but suppress contribution so we don't smear or inject edge-locked crawl.
                    if (dUp0_ire <= dDn0_ire) { useUp = true; useDn = false; }
                    else                      { useDn = true; useUp = false; }
                    boundaryWeightScale = TRANS_SUPPRESS;
                }
            }
        }

        // Combine neighbors with soft signed contributions, using *weighted* averaging (no integer dilution).
        // softAlignBoth returns contribution and weight together so the
        // shared correlation calc isn't repeated per side.
        std::complex<double> Zsum = Z0;
        double wsum = 1.0;

        if (useUp) {
            std::complex<double> contrib;
            double w;
            softAlignBoth(Z0, ZUpRaw, a0, aUp, contrib, w);
            const double sideScale = boundaryWeightScale * reachUp1;
            w *= sideScale;
            if (w > 0.0) {
                Zsum += contrib * sideScale;
                wsum += w;
            }
        }
        if (useDn) {
            std::complex<double> contrib;
            double w;
            softAlignBoth(Z0, ZDnRaw, a0, aDn, contrib, w);
            const double sideScale = boundaryWeightScale * reachDn1;
            w *= sideScale;
            if (w > 0.0) {
                Zsum += contrib * sideScale;
                wsum += w;
            }
        }

        std::complex<double> Zframe = Zsum / wsum;

        std::complex<double> delta = Zframe - Z0;
        double deltaMagIRE = cmag(delta) * invI;

        double motionGate = 1.0; // placeholder for motion gating

        const double effMaxDeltaIRE = MAX_DELTA_IRE * motionGate;

        // --- Existing clamp (pre-strength) ---
        if (deltaMagIRE > effMaxDeltaIRE && deltaMagIRE > 1e-9) {
            delta *= (effMaxDeltaIRE / deltaMagIRE);
            deltaMagIRE = effMaxDeltaIRE;
        }

        // --------------------------------------------------------
        // Adaptive comb strength: 0.5 .. COMB_STRENGTH
        // Use strong comb only when coherence is high AND vertical neighbors agree.
        // --------------------------------------------------------
        const double COMB_STRENGTH_HI = COMB_STRENGTH;
        const double COMB_STRENGTH_LO = std::min(0.8, COMB_STRENGTH_HI);

        // Coherence vs center (signed corr magnitude) for allowed neighbors.
        // Use cached magnitudes — corrSignedMags takes them directly.
        double coh = 0.0;
        if (useUp) coh = std::max(coh, std::fabs(corrSignedMags(Z0, ZUpRaw, a0, aUp)));
        if (useDn) coh = std::max(coh, std::fabs(corrSignedMags(Z0, ZDnRaw, a0, aDn)));

        // Map coherence -> [0..1]
        const double COH_T0 = 0.55;
        const double COH_T1 = 0.85;
        double cohGate = (coh - COH_T0) / (COH_T1 - COH_T0);
        cohGate = std::clamp(cohGate, 0.0, 1.0);

        // Vertical agreement gate (1 when Up/Dn agree; 0 when they disagree strongly).
        // Reuse dUpDown_ire if it was computed in the boundary block above;
        // otherwise compute it once here.
        double disGate = 1.0;
        if (haveUp && haveDn) {
            const double dUD_ire = (dUpDown_ire >= 0.0)
                                   ? dUpDown_ire
                                   : (cmag(ZUpRaw - ZDnRaw) * invI);
            const double DISAGREE_LO_IRE = 3.5;
            const double DISAGREE_HI_IRE = 14.0;
            double t = (dUD_ire - DISAGREE_LO_IRE) /
                       (DISAGREE_HI_IRE - DISAGREE_LO_IRE);
            t = std::clamp(t, 0.0, 1.0);
            disGate = 1.0 - t;
        }

        double strengthMix = cohGate * disGate;

        // Make it a bit more selective without hard switching
        strengthMix = strengthMix * strengthMix; // gamma=2

        double localStrength =
            COMB_STRENGTH_LO + (COMB_STRENGTH_HI - COMB_STRENGTH_LO) * strengthMix;

        // Provisional output (before optional under-comb correction)
        std::complex<double> Zout = Z0 + (delta * localStrength * motionGate);

        // --------------------------------------------------------
        // Optional one-sided "under-comb" booster
        // With adaptive strength, you may want this OFF initially.
        // If you keep it ON, it should boost relative to localStrength, not COMB_STRENGTH.
        // --------------------------------------------------------
        if (false)  // flip to true only if needed
        {
            double targetIRE = 0.0;

            if (useUp && !useDn) {
                targetIRE = cmag(ZUpRaw - Z0) * invI;
            }
            else if (useDn && !useUp) {
                targetIRE = cmag(ZDnRaw - Z0) * invI;
            }
            else if (useUp && useDn) {
                const double tU = cmag(ZUpRaw - Z0) * invI;
                const double tD = cmag(ZDnRaw - Z0) * invI;
                targetIRE = std::min(tU, tD);
            }

            const double TARGET_FRAC = 0.60;
            targetIRE *= TARGET_FRAC;

            const double TARGET_MIN_IRE = 1.0;
            if (targetIRE > TARGET_MIN_IRE) {
                std::complex<double> dOut = Zout - Z0;
                double actualIRE = cmag(dOut) * invI;

                if (actualIRE + 1e-9 < targetIRE) {
                    double boost = targetIRE / (actualIRE + 1e-9);

                    const double BOOST_MAX = 1.35;
                    boost = std::clamp(boost, 1.0, BOOST_MAX);

                    std::complex<double> dBoost = dOut * boost;

                    // Reapply max-delta safety AFTER boost (output space)
                    double dBoostIRE = cmag(dBoost) * invI;
                    double effMaxOutIRE = effMaxDeltaIRE * localStrength;

                    if (dBoostIRE > effMaxOutIRE && dBoostIRE > 1e-9) {
                        dBoost *= (effMaxOutIRE / dBoostIRE);
                    }

                    Zout = Z0 + dBoost;
                }
            }
        }

        // Interfield alien-chroma cancel, vector form.
        //
        // Where the center IQ is the lone phase-displaced intruder on a
        // coherent vertical column, the sign-aligned average above reinforces
        // it instead of cancelling it (softAlignContrib flips an anti-phase
        // neighbor before adding). Pull the center toward the agreeing
        // neighbors' common carrier by the displacement confidence instead, so
        // only the alien displacement is removed and the shared real chroma is
        // preserved. Strength is 0 unless the neighbors agree, the center is
        // displaced from them, and the column is supported, so this is inert on
        // ordinary chroma and genuine transitions.
        if (haveUp && haveDn) {
            double columnSupport = columnStraight;
            if (reachTapLine && x < (int)reachTapLine->hLumaDeltaIRE.size()) {
                const double lumaEdgeFit = std::clamp(
                    (reachTapLine->hLumaDeltaIRE[x] - 6.0) / 12.0, 0.0, 1.0);
                columnSupport = std::clamp(
                    0.45 * lumaEdgeFit + 0.55 * columnStraight, 0.0, 1.0);
            }

            const double cancelStrength =
                CombContentReach::interfieldAlienCancelStrength(
                    Z0.real() * invI, Z0.imag() * invI,
                    ZUpRaw.real() * invI, ZUpRaw.imag() * invI,
                    ZDnRaw.real() * invI, ZDnRaw.imag() * invI,
                    true, true,
                    2.0,
                    columnSupport);

            if (cancelStrength > 0.0) {
                const std::complex<double> neighborCommon =
                    0.5 * (ZUpRaw + ZDnRaw);

                const double aNc = cmag(neighborCommon);
                const double sdCenterCommon =
                    corrSignedMags(Z0, neighborCommon, a0, aNc);

                // Co-directional tinted case:
                // real chroma biases center and neighbors into the same hue direction.
                // The error is an amplitude/offset displacement from the stable neighbor
                // carrier, so the correct target is the neighbor common itself.
                //
                // Anti-phase case:
                // center and neighbor common are opposite phases.  Pull only toward the
                // midpoint/null target to avoid swapping the alternation phase.
                const double coDirectional =
                    std::clamp((sdCenterCommon - 0.35) / 0.50, 0.0, 1.0);

                const std::complex<double> nullTarget =
                    0.5 * (Z0 + neighborCommon);

                const std::complex<double> target =
                    nullTarget * (1.0 - coDirectional) +
                    neighborCommon * coDirectional;

                // Bevel reach throttle (cribbed from Frame B): the alien-cancel
                // pull is an interfield reach across ±1, so it must obey the
                // same bevel gate as the comb above. columnSupport can stay high
                // across a horizontal luma edge that also grazes vertical detail
                // (its lumaEdgeFit term keys on horizontal luma delta); the ±1
                // reachGate carries the moving-coarse bevel throttle that backs
                // the pull off the actual edge.
                const double bevelReach = std::min(reachUp1, reachDn1);
                const double targetStrength =
                    cancelStrength * (0.70 + 0.30 * coDirectional) * bevelReach;

                Zout = Zout * (1.0 - targetStrength) +
                       target * targetStrength;
            }
		}

        outFrameIQ[x] = Zout;
    }
}
// Frame A: adaptive interframe IQ comb fed by the Field B preclean ring.
void Comb::FrameBuffer::computeFrameAAdaptiveIQLine(
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

    const std::vector<float> *tiOverride = nullptr;
    const std::vector<float> *tqOverride = nullptr;
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

    const CombTapLine &reachTapLine = ensureCombTapLine(line);
    computeFrameIQFromPreparedVectors(line, scratch_centerIQ, scratch_upIQ, scratch_dnIQ,
                                      outFrameIQ, tiOverride, tqOverride,
                                      &reachTapLine);
}
// Frame B: direct interframe IQ comb.
// Sources from the Field B preclean ring, with locked-1D IQ as fallback.
// Unlike Frame A, this intentionally does not phase-align the neighbors before
// averaging; that plainness is part of what keeps Frame B from inheriting
// Frame A's saturated-edge alternation failure mode.
// Reach is consumed from pairU1/pairD1.reachGate (legality + interfield IQ
// floor); the combine itself is unconditional when reach is non-zero.
void Comb::FrameBuffer::computeFrameBDirectIQLine(
    int line,
    std::vector<std::complex<double>> &outFrameIQ,
    const std::vector<float> *tiOverride,
    const std::vector<float> *tqOverride)
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

    // Preclean stays scalar (Field B is crisper that way).  The demod here
    // must put the result in Grid4fscIQ — a polarity-preserving frame — so
    // that Frame B's interfield combine cancels alien Y and preserves real
    // chroma.  Using the unsigned sample class would produce "canonical 4fsc
    // demod of a LockedCommonPhaseScalar," which inherits the polarity
    // erasure: real chroma comes out anti-phased between fields and the
    // combine flips it, while alien Y comes out co-phased and gets preserved
    // — exactly backwards.  carrierGrammarSignedSampleClass folds lineFlip
    // into the phase (phase + 2 mod 4 == negate both IQ components), which is
    // the lineFlip step that locked1DSource publication left out by
    // construction.
    auto tiLine = [&](int ln)->const float* {
        if (tiOverride && (int)tiOverride->size() >= (ln + 1) * demodWidth)
            return tiOverride->data() + static_cast<size_t>(ln) * demodWidth;
        return locked1DTI4fsc_line(ln);
    };
    auto tqLine = [&](int ln)->const float* {
        if (tqOverride && (int)tqOverride->size() >= (ln + 1) * demodWidth)
            return tqOverride->data() + static_cast<size_t>(ln) * demodWidth;
        return locked1DTQ4fsc_line(ln);
    };

    const float *ti0_raw  = tiLine(line);
    const float *tq0_raw  = tqLine(line);
    const float *tiUp_raw = (verticalAllowed && line - 1 >= first) ? tiLine(line - 1) : nullptr;
    const float *tqUp_raw = (verticalAllowed && line - 1 >= first) ? tqLine(line - 1) : nullptr;
    const float *tiDn_raw = (verticalAllowed && line + 1 <  last)  ? tiLine(line + 1) : nullptr;
    const float *tqDn_raw = (verticalAllowed && line + 1 <  last)  ? tqLine(line + 1) : nullptr;
	    if (!ti0_raw || !tq0_raw) {
	        clearFrameIQ();
	        return;
	    }

    const bool haveUpLine = (verticalAllowed && line - 1 >= first);
    const bool haveDnLine = (verticalAllowed && line + 1 <  last);
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

    if ((int)scratch_centerIQ.size() != width) scratch_centerIQ.resize(width);
    if ((int)scratch_upIQ.size() != width) scratch_upIQ.resize(width);
    if ((int)scratch_dnIQ.size() != width) scratch_dnIQ.resize(width);
    for (int x = 0; x < width; ++x) {
        if (preclean0)
            scratch_centerIQ[x] =
                carrierGrammarDemodSignedCompositeTo4fsc(phase0Cursor, preclean0[x]);
        else
            scratch_centerIQ[x] = { (double)ti0_raw[x], (double)tq0_raw[x] };

        if (precleanUp)
            scratch_upIQ[x] =
                carrierGrammarDemodSignedCompositeTo4fsc(phaseUpCursor, precleanUp[x]);
        else if (tiUp_raw && tqUp_raw)
            scratch_upIQ[x] = { (double)tiUp_raw[x], (double)tqUp_raw[x] };
        else
            scratch_upIQ[x] = { 0.0, 0.0 };

        if (precleanDn)
            scratch_dnIQ[x] =
                carrierGrammarDemodSignedCompositeTo4fsc(phaseDnCursor, precleanDn[x]);
        else if (tiDn_raw && tqDn_raw)
            scratch_dnIQ[x] = { (double)tiDn_raw[x], (double)tqDn_raw[x] };
        else
            scratch_dnIQ[x] = { 0.0, 0.0 };
    }

    computeFrameBDirectIQFromPreparedVectors(line, scratch_centerIQ, scratch_upIQ, scratch_dnIQ,
                                             outFrameIQ, &reachTapLine);
}

void Comb::FrameBuffer::computeFrameBDirectIQFromPreparedVectors(
    int line,
    const std::vector<std::complex<double>> &centerIQ,
    const std::vector<std::complex<double>> &upIQ,
    const std::vector<std::complex<double>> &dnIQ,
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

	    if ((int)centerIQ.size() < width ||
	        (int)upIQ.size() < width ||
	        (int)dnIQ.size() < width)
	    {
	        clearFrameIQ();
	        return;
	    }

    const auto &T = configuration.tunables;

    const double combStrength =
        std::clamp(std::max(0.0, T.FRAME_B_COMB_STRENGTH), 0.0, 1.0);

    const double maxDeltaIRE = std::max(0.0, T.FRAME_B_RAW_MAX_DELTA_IRE);

    const bool verticalAllowed = carrierFrameVerticalAllowed(line);
    const bool haveUpLine = verticalAllowed && (line - 1 >= first);
    const bool haveDnLine = verticalAllowed && (line + 1 < last);

    // Plain ±1 interfield comb.  Per project_frameb_comb_must_run: when a
    // legal partner exists this MUST run — never gate to bare center via
    // confidence correlations.  Checker suppression belongs in the reach
    // floor (interfieldIQReachFloor in applyFrameReachWithIQFloor), which raises
    // pairU1/pairD1.reachGate at counterpart sites; here we simply consume
    // those gates.  Geometry: pull = 0.5 * combStrength * reachAuthority,
    // so when both reaches are 1 the output is the 3-tap interfield average
    // (Z0 + target)/2 — anti-phase alien cancels to zero, same-phase real
    // chroma (delta≈0) is unchanged.
    for (int x = 0; x < width; ++x) {
        const std::complex<double> Z0 = centerIQ[x];

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

        if (upReach <= 0.0 && dnReach <= 0.0) {
            outFrameIQ[x] = Z0;
            continue;
        }

        const std::complex<double> ZUp = upIQ[x];
        const std::complex<double> ZDn = dnIQ[x];
        const bool haveUp = upReach > 0.0 && cmag2(ZUp) > 1e-18;
        const bool haveDn = dnReach > 0.0 && cmag2(ZDn) > 1e-18;

        double wsum = 0.0;
        std::complex<double> target(0.0, 0.0);
        if (haveUp) { target += ZUp * upReach; wsum += upReach; }
        if (haveDn) { target += ZDn * dnReach; wsum += dnReach; }

        if (wsum <= 1e-12) {
            outFrameIQ[x] = Z0;
            continue;
        }

        target /= wsum;
        std::complex<double> delta = target - Z0;

        // Bound the per-pixel swing.  Flat cap (no two-sided-reach relax):
        // bevels are exactly the sites where both reaches go high AND the
        // cross-bevel delta is large, and at those sites the unclamped pull
        // produces the 2-pixel zipper.  The reach throttle in
        // applyFrameReachWithIQFloor is what cancels the bevel pull at its
        // source; this cap is the second line of defence against any
        // pathological intermediate target the throttle leaves through.
        const double deltaIRE = cmag(delta) * invIreScale;
        if (maxDeltaIRE > 0.0 && deltaIRE > maxDeltaIRE && deltaIRE > 1e-9)
            delta *= (maxDeltaIRE / deltaIRE);

        const double reachAuthority = std::clamp(wsum / 2.0, 0.0, 1.0);
        const double pull = std::clamp(0.5 * combStrength * reachAuthority, 0.0, 1.0);

        outFrameIQ[x] = Z0 + delta * pull;

        if (!std::isfinite(outFrameIQ[x].real()) ||
            !std::isfinite(outFrameIQ[x].imag()))
        {
            outFrameIQ[x] = Z0;
        }
    }
}

void Comb::FrameBuffer::computeFrameBDirectIQCompositeLine(
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

    // Symmetric round-trip with the demod: Frame B's input was demodded with
    // carrierGrammarSignedSampleClass (Grid4fscIQ in), so the remod here must
    // also be signed to land back in the physical composite frame that
    // produceY's `raw - clpLine` subtraction expects.  Using the unsigned
    // sample class would leave clpLine in a lineFlip-stripped convention and
    // mis-cancel the chroma in produceY → checkerboard with flipped polarity
    // rather than no checkerboard.
    auto remodCursor = carrierGrammarSignedSampleCursor(
        configuration.phaseCompensation ? carrierGrammarLine(line) : nullptr,
        left);

    const auto &T = configuration.tunables;
    const double combStrength =
        std::clamp(std::max(0.0, T.FRAME_B_COMB_STRENGTH), 0.0, 1.0);
    const double maxDeltaIRE = std::max(0.0, T.FRAME_B_RAW_MAX_DELTA_IRE);

    for (int x = 0; x < width; ++x) {
        const std::complex<double> Z0 = preclean0
            ? carrierGrammarDemodSignedCompositeTo4fsc(phase0Cursor, preclean0[x])
            : std::complex<double>((double)ti0_raw[x], (double)tq0_raw[x]);
        const std::complex<double> ZUp = precleanUp
            ? carrierGrammarDemodSignedCompositeTo4fsc(phaseUpCursor, precleanUp[x])
            : ((tiUp_raw && tqUp_raw)
                ? std::complex<double>((double)tiUp_raw[x], (double)tqUp_raw[x])
                : std::complex<double>(0.0, 0.0));
        const std::complex<double> ZDn = precleanDn
            ? carrierGrammarDemodSignedCompositeTo4fsc(phaseDnCursor, precleanDn[x])
            : ((tiDn_raw && tqDn_raw)
                ? std::complex<double>((double)tiDn_raw[x], (double)tqDn_raw[x])
                : std::complex<double>(0.0, 0.0));

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
        if (upReach > 0.0 || dnReach > 0.0) {
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
                const double pull =
                    std::clamp(0.5 * combStrength * reachAuthority, 0.0, 1.0);
                Zout = Z0 + delta * pull;
                if (!std::isfinite(Zout.real()) || !std::isfinite(Zout.imag()))
                    Zout = Z0;
            }
        }

        outFrameIQ[x] = Zout;
        outFrameScalar[x] =
            carrierGrammarRemodSigned4fscToComposite(remodCursor, Zout.real(), Zout.imag());
    }
}

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

    const int firstLine  = videoParameters.firstActiveFrameLine;
    const int lastLine   = videoParameters.lastActiveFrameLine;
    const int left       = videoParameters.activeVideoStart;
    const int right      = videoParameters.activeVideoEnd;
    const int fieldWidth = videoParameters.fieldWidth;

    auto clampH = [&](int x) -> int {
        return std::clamp(x, left, right - 1);
    };

    // Bounds check
    if ((unsigned)(lineNumber - firstLine) >= (unsigned)(lastLine - firstLine) ||
        (unsigned)(refLineNumber - firstLine) >= (unsigned)(lastLine - firstLine)) {
        result.penalty = 1000.0;
        return result;
    }

    // Source is Bucket (polarity Preserved) on purpose, NOT a mislabel: this is
    // a cross-frame ScalarSignCompare, and only a polarity-preserved source
    // returns allowScalarSignCompare. Re-tagging this LockedCommonPhaseScalar —
    // as an earlier note suggested — would yield the CommonPhaseOnly verdict,
    // drop allowScalarSignCompare, and reject every temporal candidate (penalty
    // 1000), disabling 3D entirely. The actual polarity guard here is the
    // carrierLineFlip gate at the getBestCandidate call site. Honest expression
    // of "grammar phase-relation sign-compare on a common-phase scalar is legal"
    // is deferred to the future consolidated reach system.
    const lddecode::CombReachReply phaseReach = combReachIndex.queryAgainst(
        frameBuffer.combReachIndex,
        {refLineNumber,
         lineNumber,
         refH,
         h,
         lddecode::CombReachUse::ScalarSignCompare,
         lddecode::makeBucketScalarReachSource()});
    if (!phaseReach.allowScalarSignCompare ||
        phaseReach.carrierRelation != lddecode::CarrierPhaseRelation::Opposite) {
        result.penalty = 1000.0;
        return result;
    }

    const int hh = clampH(h);

    // 1D sample: locked path reads the phase-corrected blind bandpass;
    // bucket path reads clpbuffer[0] directly.
    const double *lockedRow = frameBuffer.configuration.phaseCompensation
        ? frameBuffer.locked1DSource_line(lineNumber) : nullptr;
    if (lockedRow && (hh - left) >= 0 && (hh - left) < (right - left))
    {
        result.sample = lockedRow[hh - left];
    } else {
        result.sample = frameBuffer.bucketScalar1D_line(lineNumber)[hh];
    }

    // --- Luma Penalty with Neighbor Shaping (Cross Pattern) ---
    // Horizontal (Current Line)
    const quint16 *refRawC  = rawbuffer.data() + refLineNumber * fieldWidth;
    const double  *refClpC  = clpbuffer[1].pixel[refLineNumber];
    const quint16 *candRawC = frameBuffer.rawbuffer.data() + lineNumber * fieldWidth;
    const double  *candClpC = frameBuffer.clpbuffer[1].pixel[lineNumber];

    // Vertical (Up/Down Lines) - Neighbor Shaping
    const bool verticalAllowed =
        carrierFrameVerticalAllowed(refLineNumber) &&
        frameBuffer.carrierFrameVerticalAllowed(lineNumber);
    const bool haveUp = verticalAllowed &&
                        (refLineNumber - 1 >= firstLine) &&
                        (lineNumber - 1 >= firstLine);
    const bool haveDn = verticalAllowed &&
                        (refLineNumber + 1 < lastLine) &&
                        (lineNumber + 1 < lastLine);

    // Pointers for Ref Neighbors
    const quint16 *refRawU = haveUp ? (rawbuffer.data() + (refLineNumber - 1) * fieldWidth) : refRawC;
    const double  *refClpU = haveUp ? (clpbuffer[1].pixel[refLineNumber - 1]) : refClpC;
    const quint16 *refRawD = haveDn ? (rawbuffer.data() + (refLineNumber + 1) * fieldWidth) : refRawC;
    const double  *refClpD = haveDn ? (clpbuffer[1].pixel[refLineNumber + 1]) : refClpC;

    // Pointers for Cand Neighbors
    const quint16 *candRawU = haveUp ? (frameBuffer.rawbuffer.data() + (lineNumber - 1) * fieldWidth) : candRawC;
    const double  *candClpU = haveUp ? (frameBuffer.clpbuffer[1].pixel[lineNumber - 1]) : candClpC;
    const quint16 *candRawD = haveDn ? (frameBuffer.rawbuffer.data() + (lineNumber + 1) * fieldWidth) : candRawC;
    const double  *candClpD = haveDn ? (frameBuffer.clpbuffer[1].pixel[lineNumber + 1]) : candClpC;

    auto getLuma = [&](const quint16* r, const double* c, int idx) -> double {
        return (double)r[idx] - c[idx];
    };

    // Horizontal Diffs
    int r0 = clampH(refH - 1), r1 = clampH(refH), r2 = clampH(refH + 1);
    int c0 = clampH(h - 1),    c1 = hh,           c2 = clampH(h + 1);

    double dC0 = std::fabs(getLuma(refRawC, refClpC, r0) - getLuma(candRawC, candClpC, c0));
    double dC1 = std::fabs(getLuma(refRawC, refClpC, r1) - getLuma(candRawC, candClpC, c1));
    double dC2 = std::fabs(getLuma(refRawC, refClpC, r2) - getLuma(candRawC, candClpC, c2));

    // Vertical Diffs (Center Horizontal only)
    double dU = std::fabs(getLuma(refRawU, refClpU, r1) - getLuma(candRawU, candClpU, c1));
    double dD = std::fabs(getLuma(refRawD, refClpD, r1) - getLuma(candRawD, candClpD, c1));

    // Weighted Luma Penalty
    // Neighbors get weighted by NEIGHBOR_SHAPE_STRENGTH (implicit here via averaging, could be tuned)
    // 3 Horizontal + 2 Vertical = 5 samples
    double yPen = (dC0 + dC1 + dC2 + dU + dD) / 5.0;
    yPen *= invIreScale;

    // --- Chrominance Penalty (2D) ---
    // (Kept similar to previous, but could also add vertical if desired. Sticking to H for speed/legacy match)
    const int fRef = carrierLineFlip(refLineNumber);
    const int fCand = carrierLineFlip(lineNumber);

    double iqPen = (std::fabs(fRef * refClpC[r0] - fCand * candClpC[c0]) * 0.5 +
                    std::fabs(fRef * refClpC[r1] - fCand * candClpC[c1]) * 1.0 +
                    std::fabs(fRef * refClpC[r2] - fCand * candClpC[c2]) * 0.5) / 2.0;
    iqPen = (iqPen * invIreScale) * 0.28 * configuration.chromaWeight;

    double penalty = yPen + iqPen + adjustPenalty;

    if (penalty > configuration.candidatePenaltyHardMax) penalty = configuration.candidatePenaltyHardMax;
    result.penalty = penalty;
    return result;
}
// getBestY - Dedicated 3D Residual Y Election/Blend
double Comb::FrameBuffer::getBestY(qint32 line, qint32 h, double currentY2D,
                                   const FrameBuffer &prev, const FrameBuffer &next) const
{
    const auto &T = configuration.tunables;
    const int fw  = videoParameters.fieldWidth;

    // Helper to extract Y from a framebuffer
    auto getY = [&](const FrameBuffer& fb, int ln, int x) -> double {
        if (ln < videoParameters.firstActiveFrameLine || ln >= videoParameters.lastActiveFrameLine) return 0.0;
        double raw = (double)fb.rawbuffer.data()[ln * fw + x];
        double clp = fb.clpbuffer[1].pixel[ln][x];
        return raw - clp;
    };

    double yCurr = currentY2D;
    double yPrev = getY(prev, line, h);
    double yNext = getY(next, line, h);

    // --- Neighbor Shaping ---
    // Calculate a "Spatial Consensus" for the current pixel using N/S/E/W
    // This assumes the 2D Comb (currentY2D) captures the spatial truth reasonably well,
    // but we check the *input* spatial context to see if candidates fit.

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
