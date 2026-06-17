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
    tapLine.haveU2 = (wantFieldA || wantFieldB) && (tapLine.lnU2 >= first && tapLine.lnU2 < last);
    tapLine.haveD2 = (wantFieldA || wantFieldB) && (tapLine.lnD2 >= first && tapLine.lnD2 < last);
    tapLine.haveU4 = wantFieldA && (tapLine.lnU4 >= first && tapLine.lnU4 < last);
    tapLine.haveD4 = wantFieldA && (tapLine.lnD4 >= first && tapLine.lnD4 < last);

    tapLine.haveIQ0 = false;
    tapLine.haveIQU1 = false;
    tapLine.haveIQD1 = false;
    tapLine.haveIQU2 = false;
    tapLine.haveIQD2 = false;
    tapLine.haveIQU4 = false;
    tapLine.haveIQD4 = false;

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
    if (wantFieldA || wantFieldB) {
        ensureWidth(tapLine.tapU2);
        ensureWidth(tapLine.tapD2);
        ensureWidth(tapLine.pairU2);
        ensureWidth(tapLine.pairD2);
    }
    if (wantFieldA) {
        ensureWidth(tapLine.tapU4);
        ensureWidth(tapLine.tapD4);
        ensureWidth(tapLine.contour);
    }
    if (wantFieldB || wantFrame)
        ensureWidth(tapLine.hLumaDeltaIRE);

    const bool wantIQ = configuration.phaseCompensation;
    if (wantIQ) {
        ensureWidth(tapLine.tap0IQ);
        if (wantFrame) {
            ensureWidth(tapLine.tapU1IQ);
            ensureWidth(tapLine.tapD1IQ);
        }
        if (wantFieldA || wantFieldB) {
            ensureWidth(tapLine.tapU2IQ);
            ensureWidth(tapLine.tapD2IQ);
        }
        if (wantFieldA) {
            ensureWidth(tapLine.tapU4IQ);
            ensureWidth(tapLine.tapD4IQ);
        }
    }

    auto getCompRow = [&](int ln)->const double* {
        if (ln < first || ln >= last) return nullptr;
        if (configuration.phaseCompensation) {
            return locked1DSource_line(ln);
        }
        return bucketScalar1D_line(ln) + left;
    };
    auto getTiRow = [&](int ln)->const float* {
        if (!configuration.phaseCompensation || ln < 0 || ln >= demodLines || demodWidth <= 0) return nullptr;
        return demodTI4fsc_line(ln);
    };
    auto getTqRow = [&](int ln)->const float* {
        if (!configuration.phaseCompensation || ln < 0 || ln >= demodLines || demodWidth <= 0) return nullptr;
        return demodTQ4fsc_line(ln);
    };

    auto getMagRow = [&](int ln)->const float* {
        if (!configuration.phaseCompensation || ln < 0 || ln >= demodLines || demodWidth <= 0) return nullptr;
        return demodIQMag4fsc_line(ln);
    };
    struct RowRefs {
        int ln = -1;
        const double *comp = nullptr;
        const float *ti = nullptr;
        const float *tq = nullptr;
        const float *mag = nullptr;
        bool haveLine = false;
    };
    auto rowRefs = [&](int ln, bool haveLine, bool wantIQ)->RowRefs {
        RowRefs r;
        r.ln = ln;
        r.haveLine = haveLine;
        r.comp = haveLine ? getCompRow(ln) : nullptr;
        r.ti = (haveLine && wantIQ) ? getTiRow(ln) : nullptr;
        r.tq = (haveLine && wantIQ) ? getTqRow(ln) : nullptr;
        r.mag = (haveLine && wantIQ) ? getMagRow(ln) : nullptr;
        return r;
    };

    RowRefs r0  = rowRefs(tapLine.ln0,  true,            wantIQ);
    RowRefs rU1 = rowRefs(tapLine.lnU1, tapLine.haveU1,  wantIQ && wantFrame);
    RowRefs rD1 = rowRefs(tapLine.lnD1, tapLine.haveD1,  wantIQ && wantFrame);
    RowRefs rU2 = rowRefs(tapLine.lnU2, tapLine.haveU2,  wantIQ && (wantFieldA || wantFieldB));
    RowRefs rD2 = rowRefs(tapLine.lnD2, tapLine.haveD2,  wantIQ && (wantFieldA || wantFieldB));
    RowRefs rU4 = rowRefs(tapLine.lnU4, tapLine.haveU4,  wantIQ && wantFieldA);
    RowRefs rD4 = rowRefs(tapLine.lnD4, tapLine.haveD4,  wantIQ && wantFieldA);

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
    const double *lumaU1 = getLumaRow(tapLine.lnU1);
    const double *lumaD1 = getLumaRow(tapLine.lnD1);
    const double *lumaU2 = getLumaRow(tapLine.lnU2);
    const double *lumaD2 = getLumaRow(tapLine.lnD2);

    auto fillTap = [&](const RowRefs &r,
                       std::vector<CombTapScalar> &dst,
                       std::vector<CombTapIQ> *dstIQ,
                       bool &haveIQDst) {
        const bool haveComp = (r.comp != nullptr);
        const bool haveIQRow = (dstIQ != nullptr &&
                                r.ti != nullptr &&
                                r.tq != nullptr &&
                                demodWidth >= width);
        haveIQDst = haveIQRow;

        if (!haveComp) {
            for (int rel = 0; rel < width; ++rel) {
                dst[rel] = CombTapScalar();
                if (dstIQ)
                    (*dstIQ)[rel] = CombTapIQ();
            }
            return;
        }

        const bool haveMagRow = (haveIQRow && r.mag != nullptr);
        for (int rel = 0; rel < width; ++rel) {
            const int rm1 = reflectCombRel(rel - 1, width);
            const int rp1 = reflectCombRel(rel + 1, width);
            CombTapScalar &s = dst[rel];
            s.comp = r.comp[rel];
            s.symMag = 0.5 * (std::fabs(r.comp[rm1]) + std::fabs(r.comp[rp1]));

            if (haveIQRow) {
                const double ti = r.ti[rel];
                const double tq = r.tq[rel];
                CombTapIQ &iq = (*dstIQ)[rel];
                iq.ti = (float)ti;
                iq.tq = (float)tq;
                iq.iqMag = haveMagRow
                    ? static_cast<double>(r.mag[rel])
                    : std::hypot(ti, tq);
            }
        }
    };

    fillTap(r0, tapLine.tap0, wantIQ ? &tapLine.tap0IQ : nullptr, tapLine.haveIQ0);
    if (wantFrame) {
        fillTap(rU1, tapLine.tapU1, wantIQ ? &tapLine.tapU1IQ : nullptr, tapLine.haveIQU1);
        fillTap(rD1, tapLine.tapD1, wantIQ ? &tapLine.tapD1IQ : nullptr, tapLine.haveIQD1);
    }
    if (wantFieldA || wantFieldB) {
        fillTap(rU2, tapLine.tapU2, wantIQ ? &tapLine.tapU2IQ : nullptr, tapLine.haveIQU2);
        fillTap(rD2, tapLine.tapD2, wantIQ ? &tapLine.tapD2IQ : nullptr, tapLine.haveIQD2);
    }
    if (wantFieldA) {
        fillTap(rU4, tapLine.tapU4, wantIQ ? &tapLine.tapU4IQ : nullptr, tapLine.haveIQU4);
        fillTap(rD4, tapLine.tapD4, wantIQ ? &tapLine.tapD4IQ : nullptr, tapLine.haveIQD4);
    }

    const auto &T = configuration.tunables;
    const double kRange = T.FIELD_K_RANGE_IRE * irescale;
    const double invK = (kRange > 1e-9) ? (1.0 / kRange) : 0.0;
    const double invI = invIreScale;
    const lddecode::CombReachSourceFrame scalarSource = scalarReachSource();
    const lddecode::CombReachSourceFrame iqSource = iqReachSource();

    auto coherenceAt = [&](const std::vector<CombTapIQ> &nbrIQ, bool haveNbrIQ, int rel)->double {
        rel = std::clamp(rel, 0, width - 1);
        if (!tapLine.haveIQ0 || !haveNbrIQ) return 1.0;
        const CombTapIQ &c = tapLine.tap0IQ[rel];
        const CombTapIQ &n = nbrIQ[rel];
        if (c.iqMag * invI < 2.5 || n.iqMag * invI < 2.5) return 1.0;
        const double corr = ((double)c.ti * (double)n.ti + (double)c.tq * (double)n.tq) /
                            (c.iqMag * n.iqMag + 1e-12);
        return std::clamp((corr - 0.55) / (0.85 - 0.55), 0.0, 1.0);
    };

    // Pair and contour fields are evidence, not decisions: candidates consume
    // them differently so Field A/B and Frame A/B keep their distinct behavior.
    auto fillPair = [&](const std::vector<CombTapScalar> &nbr,
                        const std::vector<CombTapIQ> &nbrIQ,
                        const double *nbrLuma,
                        int targetLine,
                        bool haveNbr,
                        bool haveNbrIQ,
                        std::vector<CombTapPair> &dst) {
        if (!haveNbr) {
            for (int rel = 0; rel < width; ++rel)
                dst[rel] = CombTapPair();
            return;
        }

        for (int rel = 0; rel < width; ++rel) {
            CombTapPair &p = dst[rel];
            p = CombTapPair();
            const int h = left + rel;
            p.scalarReach = combReachIndex.query({
                lineNumber,
                targetLine,
                h,
                h,
                lddecode::CombReachUse::FieldScalarAverage,
                scalarSource});
            p.iqReach = combReachIndex.query({
                lineNumber,
                targetLine,
                h,
                h,
                lddecode::CombReachUse::IQCompare,
                iqSource});
            const CombTapScalar &c = tapLine.tap0[rel];
            const CombTapScalar &n = nbr[rel];
            if (haveNbr) {
                p.diffIRE = std::fabs(c.comp - n.comp) * invI;
                p.kScore = combKMetric(c.comp, c.symMag, n.comp, n.symMag);
                p.weight = (kRange > 1e-9) ? (1.0 - p.kScore * invK) : 1.0;
                p.weight = std::clamp(p.weight, 0.0, 1.0);
            }
            if (tapLine.haveIQ0 && haveNbrIQ) {
                const CombTapIQ &ciq = tapLine.tap0IQ[rel];
                const CombTapIQ &niq = nbrIQ[rel];
                p.iqDiffIRE = std::hypot((double)ciq.ti - (double)niq.ti,
                                         (double)ciq.tq - (double)niq.tq) * invI;
            }
            if (luma0 && nbrLuma) {
                p.lumaDiffIRE = std::fabs(luma0[rel] - nbrLuma[rel]) * invI;
            }
            if (configuration.phaseCompensation && haveNbrIQ) {
                const int relL = std::clamp(rel - 4, 0, width - 1);
                const int relR = std::clamp(rel + 4, 0, width - 1);
                p.coherence = (width >= 9)
                    ? std::min({coherenceAt(nbrIQ, haveNbrIQ, relL),
                                coherenceAt(nbrIQ, haveNbrIQ, rel),
                                coherenceAt(nbrIQ, haveNbrIQ, relR)})
                    : coherenceAt(nbrIQ, haveNbrIQ, rel);
            }
        }
    };

    if (wantFrame) {
        fillPair(tapLine.tapU1, tapLine.tapU1IQ, lumaU1, tapLine.lnU1, tapLine.haveU1, tapLine.haveIQU1, tapLine.pairU1);
        fillPair(tapLine.tapD1, tapLine.tapD1IQ, lumaD1, tapLine.lnD1, tapLine.haveD1, tapLine.haveIQD1, tapLine.pairD1);
    }
    if (wantFieldA || wantFieldB) {
        fillPair(tapLine.tapU2, tapLine.tapU2IQ, lumaU2, tapLine.lnU2, tapLine.haveU2, tapLine.haveIQU2, tapLine.pairU2);
        fillPair(tapLine.tapD2, tapLine.tapD2IQ, lumaD2, tapLine.lnD2, tapLine.haveD2, tapLine.haveIQD2, tapLine.pairD2);
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
            tapLine.hLumaDeltaIRE[rel] = std::fabs(tapLine.tap0[rp1].comp - tapLine.tap0[rm1].comp) * invI;
        }
        }
    }

    if (wantFieldA) {
        for (int rel = 0; rel < width; ++rel) {
        CombTapContour c;
        auto curveMag = [&](const std::vector<CombTapScalar> &tap)->double {
            return std::fabs(tap[rel].comp);
        };
        const double aC  = curveMag(tapLine.tap0);
        const double aU2 = curveMag(tapLine.tapU2);
        const double aD2 = curveMag(tapLine.tapD2);
        const double aU4 = curveMag(tapLine.tapU4);
        const double aD4 = curveMag(tapLine.tapD4);

        c.curvMidIRE = std::fabs(aU2 - 2.0 * aC + aD2) * invI;
        c.midOk = combSmoothGate(c.curvMidIRE, T.FIELD_CONTOUR_SOFT_IRE, T.FIELD_CONTOUR_HARD_IRE);

        const double u4Pred = 2.0 * aU2 - aC;
        const double d4Pred = 2.0 * aD2 - aC;
        c.upResIRE = std::fabs(aU4 - u4Pred) * invI;
        c.dnResIRE = std::fabs(aD4 - d4Pred) * invI;
        c.upSideOk = combSmoothGate(c.upResIRE, T.FIELD_CONTOUR_SOFT_IRE, T.FIELD_CONTOUR_HARD_IRE);
        c.dnSideOk = combSmoothGate(c.dnResIRE, T.FIELD_CONTOUR_SOFT_IRE, T.FIELD_CONTOUR_HARD_IRE);

        const double upK = combKMetric(tapLine.tapU2[rel].comp, tapLine.tapU2[rel].symMag,
                                       tapLine.tapU4[rel].comp, tapLine.tapU4[rel].symMag);
        const double dnK = combKMetric(tapLine.tapD2[rel].comp, tapLine.tapD2[rel].symMag,
                                       tapLine.tapD4[rel].comp, tapLine.tapD4[rel].symMag);
        c.upSim = (tapLine.pairU2[rel].weight > 0.0)
            ? std::clamp((kRange > 1e-9) ? (1.0 - upK * invK) : 1.0, 0.0, 1.0)
            : 0.0;
        c.dnSim = (tapLine.pairD2[rel].weight > 0.0)
            ? std::clamp((kRange > 1e-9) ? (1.0 - dnK * invK) : 1.0, 0.0, 1.0)
            : 0.0;
        c.upTrust = c.midOk * c.upSideOk;
        c.dnTrust = c.midOk * c.dnSideOk;
        c.upInfluence = T.FIELD_CONTOUR_FAR_INFLUENCE * c.upTrust *
                         combSimilarityFactor(c.upSim, T.FIELD_CONTOUR_SIM_START, T.FIELD_CONTOUR_SIM_FULL);
        c.dnInfluence = T.FIELD_CONTOUR_FAR_INFLUENCE * c.dnTrust *
                         combSimilarityFactor(c.dnSim, T.FIELD_CONTOUR_SIM_START, T.FIELD_CONTOUR_SIM_FULL);
        tapLine.contour[rel] = c;
        }
    }

    auto applyReachLimiters = [&](std::vector<CombTapPair> &upPair,
                                  std::vector<CombTapPair> &dnPair,
                                  bool haveUp,
                                  bool haveDn,
                                  bool haveUpIQ,
                                  bool haveDnIQ,
                                  const std::vector<CombTapIQ> &upIQ,
                                  const std::vector<CombTapIQ> &dnIQ,
                                  bool useContour) {
        if (!haveUp || !haveDn || (int)upPair.size() < width || (int)dnPair.size() < width)
            return;

        for (int rel = 0; rel < width; ++rel) {
            double upGate = 1.0;
            double dnGate = 1.0;
            double contourStraight = 0.0;
            if (useContour && rel < (int)tapLine.contour.size()) {
                const CombTapContour &c = tapLine.contour[rel];
                const double contourTrust = std::clamp(
                    c.midOk * 0.5 * (c.upSideOk + c.dnSideOk), 0.0, 1.0);
                const double contourCurvNorm = std::clamp(
                    c.curvMidIRE / std::max(1.0, T.FIELD_CONTOUR_HARD_IRE), 0.0, 1.0);
                contourStraight = std::clamp(
                    0.70 * contourTrust + 0.30 * (1.0 - contourCurvNorm), 0.0, 1.0);
            }
            const double contourBend = useContour ? (1.0 - contourStraight) : 1.0;

            if (configuration.phaseCompensation && tapLine.haveIQ0 && haveUpIQ && haveDnIQ) {
                const double m0  = tapLine.tap0IQ[rel].iqMag * invI;
                const double mUp = upIQ[rel].iqMag * invI;
                const double mDn = dnIQ[rel].iqMag * invI;
                const double satHi = std::max({m0, mUp, mDn});
                const double satLo = std::min(mUp, mDn);
                const double satGate = std::clamp((satHi - 6.0) / 12.0, 0.0, 1.0);
                const double oneSideGate = std::clamp((3.5 - satLo) / 3.5, 0.0, 1.0);
                const double asym = std::fabs(mUp - mDn) / std::max(std::max(mUp, mDn), 1e-6);
                const double dUp0 = std::fabs(mUp - m0);
                const double dDn0 = std::fabs(mDn - m0);
                const double matchGap = std::fabs(dUp0 - dDn0) /
                    std::max(std::max({m0, mUp, mDn}), 1.0);
                const double inhibit = std::clamp(
                    satGate * (0.60 * oneSideGate + 0.40 * asym) *
                    (0.55 + 0.45 * matchGap) *
                    (0.15 + 0.85 * contourBend),
                    0.0, 1.0);

                if (dUp0 <= dDn0) dnGate *= (1.0 - inhibit);
                else              upGate *= (1.0 - inhibit);

                auto materialGate = [&](const CombTapPair &p)->double {
                    if (!std::isfinite(p.lumaDiffIRE) || !std::isfinite(p.iqDiffIRE))
                        return 1.0;
                    const double lumaT = std::clamp((p.lumaDiffIRE - 3.0) / 9.0, 0.0, 1.0);
                    const double iqT = std::clamp((p.iqDiffIRE - 3.0) / 9.0, 0.0, 1.0);
                    const double incoherentT = std::clamp((0.90 - p.coherence) / 0.55, 0.0, 1.0);
                    const double bothDifferent = lumaT * std::max(iqT, incoherentT);
                    const double gatedByBend = bothDifferent * contourBend;
                    return 1.0 - std::clamp(gatedByBend, 0.0, 0.95);
                };
                upGate *= materialGate(upPair[rel]);
                dnGate *= materialGate(dnPair[rel]);
            }

            double upContourGate = 1.0;
            double dnContourGate = 1.0;
            if (useContour && rel < (int)tapLine.contour.size()) {
                const CombTapContour &c = tapLine.contour[rel];
                upContourGate = 0.25 + 0.75 * c.upTrust;
                dnContourGate = 0.25 + 0.75 * c.dnTrust;
            }

            const double upScalarGate =
                upPair[rel].scalarReach.allowScalarAverage ? 1.0 : 0.0;
            const double dnScalarGate =
                dnPair[rel].scalarReach.allowScalarAverage ? 1.0 : 0.0;
            const double upIQGate =
                upPair[rel].iqReach.allowIQCompare ? 1.0 : 0.0;
            const double dnIQGate =
                dnPair[rel].iqReach.allowIQCompare ? 1.0 : 0.0;

            upPair[rel].scalarReachGate = upScalarGate;
            dnPair[rel].scalarReachGate = dnScalarGate;
            upPair[rel].iqReachGate = upGate * upIQGate;
            dnPair[rel].iqReachGate = dnGate * dnIQGate;
            upPair[rel].contourReachGate = upContourGate;
            dnPair[rel].contourReachGate = dnContourGate;
            upPair[rel].reachGate = std::clamp(upGate * upContourGate, 0.0, 1.0);
            dnPair[rel].reachGate = std::clamp(dnGate * dnContourGate, 0.0, 1.0);
        }
    };

    if (wantFrame) {
        applyReachLimiters(tapLine.pairU1, tapLine.pairD1,
                           tapLine.haveU1, tapLine.haveD1,
                           tapLine.haveIQU1, tapLine.haveIQD1,
                           tapLine.tapU1IQ, tapLine.tapD1IQ,
                           wantFieldA);
    }
    if (wantFieldA || wantFieldB) {
        applyReachLimiters(tapLine.pairU2, tapLine.pairD2,
                           tapLine.haveU2, tapLine.haveD2,
                           tapLine.haveIQU2, tapLine.haveIQD2,
                           tapLine.tapU2IQ, tapLine.tapD2IQ,
                           wantFieldA);
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

    for (int rel = 0; rel < width; ++rel) {
        const double C    = tapLine.tap0[rel].comp;
        const double Cup2 = tapLine.tapU2[rel].comp;
        const double Cdn2 = tapLine.tapD2[rel].comp;
        const double Cup4 = tapLine.tapU4[rel].comp;
        const double Cdn4 = tapLine.tapD4[rel].comp;

        const double reachUp2 = tapLine.pairU2[rel].reachGate *
                                tapLine.pairU2[rel].scalarReachGate;
        const double reachDn2 = tapLine.pairD2[rel].reachGate *
                                tapLine.pairD2[rel].scalarReachGate;
        double wUp2 = tapLine.pairU2[rel].weight * reachUp2;
        double wDn2 = tapLine.pairD2[rel].weight * reachDn2;
        const CombTapContour &curve = tapLine.contour[rel];

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
        if (wUp2 > 0.0 || wDn2 > 0.0) {
            double t2  = ((C - Cup2Adj) * wUp2 * sc2);
            t2        += ((C - Cdn2Adj) * wDn2 * sc2);
            tc        += 0.25 * t2;
        }

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
void Comb::FrameBuffer::computeSimpleFieldLine(int lineNumber, double *outFieldLine)
{
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int width = videoParameters.activeVideoEnd - videoParameters.activeVideoStart;

    if (width <= 0 || lineNumber < first || lineNumber >= last || !outFieldLine) {
        if (outFieldLine) std::fill(outFieldLine, outFieldLine + std::max(width, 0), 0.0);
        return;
    }

    const CombTapLine &tapLine = ensureCombTapLine(lineNumber);
    computeSimpleFieldLine(tapLine, outFieldLine);
}

void Comb::FrameBuffer::computeSimpleFieldLine(const CombTapLine &tapLine,
                                               double *outFieldLine)
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
        return;
    }

    auto clampSameFieldLine = [&](int ln) -> int {
        const int parity = lineNumber & 1;

        ln = std::clamp(ln, first, last - 1);

        if ((ln & 1) != parity) {
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

    auto getRow = [&](int ln) -> const double* {
        if (ln < first || ln >= last)
            return nullptr;

        if (configuration.phaseCompensation)
            return locked1DSource_line(ln);

        const double *row = bucketScalar1D_line(ln);
        return row ? (row + left) : nullptr;
    };

    const double *row0   = getRow(ln0);
    const double *rowUp2 = getRow(lnUp2);
    const double *rowDn2 = getRow(lnDn2);

    if (!row0) {
        std::fill(outFieldLine, outFieldLine + width, 0.0);
        return;
    }

    // Missing vertical support rows: preserve the local 1D chroma estimate.
    // This avoids synthetic zero chroma without changing normal Field B geometry.
    if (!rowUp2 || !rowDn2) {
        std::copy(row0, row0 + width, outFieldLine);
        return;
    }

    const auto &T = configuration.tunables;

    const double kRange = T.FIELD_K_RANGE_IRE * irescale;
    const double invK   = (kRange > 1e-9) ? (1.0 / kRange) : 0.0;
    const double invI   = this->invIreScale;

    // Carrier-free luma-edge amount. This is used only as an authority/blend
    // signal after the original vertical comb has been computed. It does not
    // kill neighbor support.
    const double LUMA_EDGE_LO_IRE = 4.5;
    const double LUMA_EDGE_HI_IRE = 16.0;

    auto edgeAmount = [&](double dIRE) -> double {
        if (!std::isfinite(dIRE))
            return 0.0;

        double t = (dIRE - LUMA_EDGE_LO_IRE) /
                   (LUMA_EDGE_HI_IRE - LUMA_EDGE_LO_IRE);

        return std::clamp(t, 0.0, 1.0);
    };

    // Saturated-gradient cede is deliberately partial. Full hard partner rejection
    // caused the diagonal-line staircase by changing reach topology. This only
    // reduces the finished Field B candidate's authority in marginal saturated
    // edge zones.
    const double SAT_CHROMA_LO_IRE = 5.0;
    const double SAT_CHROMA_HI_IRE = 18.0;
    const double MAX_EDGE_CEDE     = 0.70;

    auto ramp = [](double v, double lo, double hi) -> double {
        if (hi <= lo)
            return (v >= hi) ? 1.0 : 0.0;

        return std::clamp((v - lo) / (hi - lo), 0.0, 1.0);
    };

    for (int rel = 0; rel < width; ++rel) {
        const int rm1 = (rel > 0) ? (rel - 1) : 0;
        const int rp1 = (rel + 1 < width) ? (rel + 1) : (width - 1);

        const double C   = row0[rel];
        const double Cup = rowUp2[rel];
        const double Cdn = rowDn2[rel];

        const double C_m1 = row0[rm1];
        const double C_p1 = row0[rp1];

        const double Cup_m1 = rowUp2[rm1];
        const double Cup_p1 = rowUp2[rp1];

        const double Cdn_m1 = rowDn2[rm1];
        const double Cdn_p1 = rowDn2[rp1];

        // Symmetric lateral magnitude context. This remains the original
        // support/weight evidence for the vertical comb.
        const double symCur =
            0.5 * (std::fabs(C_m1) + std::fabs(C_p1));

        const double symUp =
            0.5 * (std::fabs(Cup_m1) + std::fabs(Cup_p1));

        const double symDn =
            0.5 * (std::fabs(Cdn_m1) + std::fabs(Cdn_p1));

        double kp = 0.0;
        double kn = 0.0;

        kp  = std::fabs(std::fabs(C) - std::fabs(Cup));
        kp += std::fabs(symCur - symUp);
        kp -= (std::fabs(C) + std::fabs(Cup)) * 0.10;

        kn  = std::fabs(std::fabs(C) - std::fabs(Cdn));
        kn += std::fabs(symCur - symDn);
        kn -= (std::fabs(C) + std::fabs(Cdn)) * 0.10;

        if (kp < 0.0) kp = 0.0;
        if (kn < 0.0) kn = 0.0;

        double wUp = (kRange > 1e-9) ? (1.0 - kp * invK) : 1.0;
        double wDn = (kRange > 1e-9) ? (1.0 - kn * invK) : 1.0;

        wUp = std::clamp(wUp, 0.0, 1.0);
        wDn = std::clamp(wDn, 0.0, 1.0);
        if (rel < (int)tapLine.pairU2.size())
            wUp *= tapLine.pairU2[rel].scalarReachGate;
        if (rel < (int)tapLine.pairD2.size())
            wDn *= tapLine.pairD2[rel].scalarReachGate;

        // Edge amounts are measured before any support cull, but they do not
        // alter support. They only modulate the final vertical authority.
        const double eUp =
            (rel >= 0 && rel < (int)tapLine.pairU2.size())
                ? edgeAmount(tapLine.pairU2[rel].lumaDiffIRE)
                : 0.0;

        const double eDn =
            (rel >= 0 && rel < (int)tapLine.pairD2.size())
                ? edgeAmount(tapLine.pairD2[rel].lumaDiffIRE)
                : 0.0;

        double sc = 1.0;
        bool haveAnswer = false;
        bool revivedUD = false;

        if (wUp > 0.0 || wDn > 0.0) {
            if (wDn > 3.0 * wUp)
                wUp = 0.0;
            else if (wUp > 3.0 * wDn)
                wDn = 0.0;

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
            // Original same-magnitude up/down revival. Do not suppress this
            // with the luma-edge detector; suppressing it recreated the moving
            // diagonal staircase.
            const double dMag  = std::fabs(std::fabs(Cup) - std::fabs(Cdn));
            const double sumUD = std::fabs(Cup + Cdn);

            if (dMag - std::fabs(sumUD * 0.2) <= 0.0) {
                wUp = (rel < (int)tapLine.pairU2.size())
                    ? tapLine.pairU2[rel].scalarReachGate
                    : 1.0;
                wDn = (rel < (int)tapLine.pairD2.size())
                    ? tapLine.pairD2[rel].scalarReachGate
                    : 1.0;
                sc = 1.0;
                haveAnswer = (wUp > 0.0 || wDn > 0.0);
                revivedUD = haveAnswer;
            } else {
                wUp = 0.0;
                wDn = 0.0;
            }
        }

        double tc = 0.0;

        if (haveAnswer) {
            tc  = (C - Cup) * wUp * sc;
            tc += (C - Cdn) * wDn * sc;
            tc *= 0.25;

            // Variable vertical-authority cede.
            //
            // Important: this does not remove either neighbor from the comb.
            // It only blends the finished vertical candidate toward C when the
            // answer is saturated, edge-adjacent, and marginal.
            const double chromaIRE = std::fabs(C) * invI;
            const double satT = ramp(chromaIRE,
                                     SAT_CHROMA_LO_IRE,
                                     SAT_CHROMA_HI_IRE);

            const double bothEdgeT  = std::min(eUp, eDn);
            const double oneEdgeT   = std::max(eUp, eDn) - bothEdgeT;

            // Both-edge cases are stronger; one-edge cases get a weaker cede
            // so a clean partner can still contribute without hard topology
            // changes.
            const double edgeT =
                std::clamp(bothEdgeT + 0.35 * oneEdgeT, 0.0, 1.0);

            const double supportSum = wUp + wDn;
            const double maxW = std::max(wUp, wDn);
            const double minW = std::min(wUp, wDn);

            const double balance =
                (maxW > 1e-9) ? (minW / maxW) : 0.0;
                // Chroma-envelope discontinuity across the same-field vertical partners.
                //
                // This catches saturated color-boundary cases such as brown -> bright gold.
                // It does not reject the partners; it only increases the final cede toward C
                // when Field B is already in the saturated/edge/marginal regime.
                const double dChromaUpIRE = std::fabs(std::fabs(C) - std::fabs(Cup)) * invI;
                const double dChromaDnIRE = std::fabs(std::fabs(C) - std::fabs(Cdn)) * invI;
                
                const double chromaStepIRE =
                    std::min(dChromaUpIRE, dChromaDnIRE);
                
                const double chromaWorstIRE =
                    std::max(dChromaUpIRE, dChromaDnIRE);
                
                const double CHROMA_STEP_LO_IRE = 3.0;
                const double CHROMA_STEP_HI_IRE = 12.0;
                
                const double CHROMA_WORST_LO_IRE = 6.0;
                const double CHROMA_WORST_HI_IRE = 22.0;
                
                const double chromaStepT =
                    ramp(chromaStepIRE, CHROMA_STEP_LO_IRE, CHROMA_STEP_HI_IRE);
                
                const double chromaWorstT =
                    ramp(chromaWorstIRE, CHROMA_WORST_LO_IRE, CHROMA_WORST_HI_IRE);
                
                const double chromaBoundaryT =
                    std::clamp(0.65 * chromaStepT + 0.35 * chromaWorstT, 0.0, 1.0);
			// Marginality: low total support, strong one-sidedness, or revived
            // up/down admission. This keeps strong clean Field B answers intact.
            const double lowSupportT =
                std::clamp((1.35 - supportSum) / 1.35, 0.0, 1.0);

            const double imbalanceT =
                std::clamp((0.55 - balance) / 0.55, 0.0, 1.0);

            double marginalT = std::max(lowSupportT, imbalanceT);
            
            if (revivedUD)
                marginalT = std::max(marginalT, 0.50);
            
            // Saturated color-boundary boost.
            //
            // Brown/gold title boundaries can have enough vertical support to avoid the
            // normal low-support path while still alternating visibly. The chroma envelope
            // disagreement makes that case marginal for Field B authority without changing
            // vertical reach topology.
            marginalT = std::max(marginalT, 0.75 * chromaBoundaryT);
            
            double cede =
                MAX_EDGE_CEDE * satT * edgeT * marginalT;
                
            cede = std::clamp(cede, 0.0, MAX_EDGE_CEDE);

            if (cede > 0.0)
                tc = tc * (1.0 - cede) + C * cede;
        } else {
            // True no-answer collapse: cede to the local 1D chroma estimate.
            tc = C;
        }

        if (!std::isfinite(tc))
            tc = C;

        outFieldLine[rel] = tc;
    }
}

static inline double cmag(const std::complex<double> &z) { return std::hypot(z.real(), z.imag()); }
static inline double dotIQ(const std::complex<double> &a, const std::complex<double> &b) { return a.real()*b.real() + a.imag()*b.imag(); }

static inline std::complex<double> applyColumnPhaseAlignment(
    const std::complex<double> &center,
    const std::complex<double> &neighbor,
    double invI,
    double minChromaIRE,
    const Comb::Configuration::Tunables &T)
{
    const double minFitIRE = std::max(2.0, 0.5 * minChromaIRE);
    const double a0 = cmag(center);
    const double an = cmag(neighbor);
    if (a0 * invI < minFitIRE || an * invI < minFitIRE)
        return neighbor;

    const double dot = dotIQ(neighbor, center);
    const double cross = neighbor.real() * center.imag() - neighbor.imag() * center.real();
    double phase = std::atan2(cross, dot);
    const double pMax = T.FRAME_IQ_COLUMN_PHASE_ALIGN_MAX_DEG * M_PI / 180.0;
    phase = std::clamp(phase, -pMax, pMax);

    const double c = std::cos(phase);
    const double s = std::sin(phase);
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

    outFrameIQ.assign(width, std::complex<double>(0.0, 0.0));

    if (width <= 0 || line < first || line >= last) return;
    if (line >= demodLines || demodWidth <= 0) return;

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

    // Signed correlation in [-1..1]
    auto corrSigned = [&](const std::complex<double> &a, const std::complex<double> &b)->double {
        const double ma = cmag(a);
        const double mb = cmag(b);
        if (ma <= 1e-12 || mb <= 1e-12) return 0.0;
        return dotIQ(a, b) / (ma*mb + 1e-12);
    };

    // ------------------------------------------------------------
    // Helper: soft signed contribution
    // ------------------------------------------------------------
    auto softAlignContrib = [&](const std::complex<double> &Z0,
                                const std::complex<double> &Zn)->std::complex<double>
    {
        const double a0 = cmag(Z0);
        const double an = cmag(Zn);
        if (a0 <= 1e-12 || an <= 1e-12) return {0.0, 0.0};

        const double c  = dotIQ(Z0, Zn) / (a0*an + 1e-12); // signed corr [-1..1]
        const double ac = std::fabs(c);

        const double t0 = 0.55;
        const double t1 = 0.85;

        double w = (ac - t0) / (t1 - t0);
        w = std::clamp(w, 0.0, 1.0);

        const double wFloor = 0.15;
        w = wFloor + (1.0 - wFloor) * w;

        const double s = (c >= 0.0) ? 1.0 : -1.0;
        return (Zn * (w * s));
    };

    // Companion: compute the same weight used by softAlignContrib (so we can do weighted averaging)
    auto softAlignWeight = [&](const std::complex<double> &Z0,
                               const std::complex<double> &Zn)->double
    {
        const double a0 = cmag(Z0);
        const double an = cmag(Zn);
        if (a0 <= 1e-12 || an <= 1e-12) return 0.0;

        const double c  = dotIQ(Z0, Zn) / (a0*an + 1e-12);
        const double ac = std::fabs(c);

        const double t0 = 0.55;
        const double t1 = 0.85;

        double w = (ac - t0) / (t1 - t0);
        w = std::clamp(w, 0.0, 1.0);

        const double wFloor = 0.15;
        w = wFloor + (1.0 - wFloor) * w;

        return w;
    };

    const double COMB_STRENGTH  = std::max(0.0, T.FRAME_COMB_STRENGTH);
    const double MAX_DELTA_IRE  = T.FRAME_IQ_RAW_MAX_DELTA_IRE;
    const double MIN_CHROMA_IRE = T.FRAME_CHROMA_MIN_IRE;

    // ------------------------------------------------------------
    // Build IQ vectors for center and 1 neighbors
    // ------------------------------------------------------------
    if ((int)centerIQ.size() < width || (int)upIQ.size() < width || (int)dnIQ.size() < width) return;

    // Frame A's alignment is intentionally column-local. A line/global affine
    // can learn from adjacent material and push that error into the current
    // column; vertical combing should not borrow lateral context here.
    for (int x = 0; x < width; ++x) {
        upIQ[x] = applyColumnPhaseAlignment(centerIQ[x], upIQ[x], invI, MIN_CHROMA_IRE, T);
        dnIQ[x] = applyColumnPhaseAlignment(centerIQ[x], dnIQ[x], invI, MIN_CHROMA_IRE, T);
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
        double sharedUpGate = 1.0;
        double sharedDnGate = 1.0;
        double columnStraight = 0.0;
        if (reachTapLine && x < (int)reachTapLine->contour.size()) {
            const CombTapContour &c = reachTapLine->contour[x];
            const double contourTrust = std::clamp(
                c.midOk * 0.5 * (c.upSideOk + c.dnSideOk), 0.0, 1.0);
            const double contourCurvNorm = std::clamp(
                c.curvMidIRE / std::max(1.0, T.FIELD_CONTOUR_HARD_IRE), 0.0, 1.0);
            columnStraight = std::clamp(
                0.70 * contourTrust + 0.30 * (1.0 - contourCurvNorm), 0.0, 1.0);
        }
        if (reachTapLine && x < reachTapLine->width &&
            x < (int)reachTapLine->pairU1.size() &&
            x < (int)reachTapLine->pairD1.size())
        {
            sharedUpGate = std::clamp(reachTapLine->pairU1[x].iqReachGate *
                                      reachTapLine->pairU1[x].contourReachGate, 0.0, 1.0);
            sharedDnGate = std::clamp(reachTapLine->pairD1[x].iqReachGate *
                                      reachTapLine->pairD1[x].contourReachGate, 0.0, 1.0);
            // Frame A should not dither between comb/no-comb on a straight
            // vertical column. Keep the limiter available for bends and
            // unrelated material, but force a minimum reach when contour
            // evidence says the column is coherent.
            const double straightFloor = 0.72 * columnStraight;
            sharedUpGate = std::max(sharedUpGate, straightFloor);
            sharedDnGate = std::max(sharedDnGate, straightFloor);
            if (sharedUpGate <= 0.02) useUp = false;
            if (sharedDnGate <= 0.02) useDn = false;
        }

        if (haveUp && haveDn) {
            const double dUp0_ire    = cmag(ZUpRaw - Z0) * invI;
            const double dDn0_ire    = cmag(ZDnRaw - Z0) * invI;
            const double dUpDown_ire = cmag(ZUpRaw - ZDnRaw) * invI;

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
        std::complex<double> Zsum = Z0;
        double wsum = 1.0;

        if (useUp) {
            const double sideScale = boundaryWeightScale * sharedUpGate;
            const double w = softAlignWeight(Z0, ZUpRaw) * sideScale;
            if (w > 0.0) {
                Zsum += softAlignContrib(Z0, ZUpRaw) * sideScale;
                wsum += w;
            }
        }
        if (useDn) {
            const double sideScale = boundaryWeightScale * sharedDnGate;
            const double w = softAlignWeight(Z0, ZDnRaw) * sideScale;
            if (w > 0.0) {
                Zsum += softAlignContrib(Z0, ZDnRaw) * sideScale;
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
        const double COMB_STRENGTH_LO = std::min(0.5, COMB_STRENGTH_HI);

        // Coherence vs center (signed corr magnitude) for allowed neighbors
        double coh = 0.0;
        if (useUp) coh = std::max(coh, std::fabs(corrSigned(Z0, ZUpRaw)));
        if (useDn) coh = std::max(coh, std::fabs(corrSigned(Z0, ZDnRaw)));

        // Map coherence -> [0..1]
        const double COH_T0 = 0.55;
        const double COH_T1 = 0.85;
        double cohGate = (coh - COH_T0) / (COH_T1 - COH_T0);
        cohGate = std::clamp(cohGate, 0.0, 1.0);

        // Vertical agreement gate (1 when Up/Dn agree; 0 when they disagree strongly)
        double disGate = 1.0;
        if (haveUp && haveDn) {
            const double dUD_ire = cmag(ZUpRaw - ZDnRaw) * invI;
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

    outFrameIQ.assign(width, std::complex<double>(0.0, 0.0));
    if (width <= 0 || line < first || line >= last) return;
    if (line >= demodLines || demodWidth <= 0) return;

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
    if (!ti0_raw || !tq0_raw) return;

    // Preclean demod helper: demod precleanRing[ln][x] -> canonical 4fsc bucket IQ.
    // This is the *only* demod used for the preclean path (no shared demod buffers).
    auto demodPrecleanAt = [&](int ln, int x, std::complex<double> &Z)->bool {
        if (ln < first || ln >= last) return false;
        const double *row = precleanLinePtr(ln, width);
        if (!row) return false;
        const int h = left + x;
        const double c = row[x];
        double i4fsc = 0.0, q4fsc = 0.0;
        demod4fscFromComposite(c, h, i4fsc, q4fsc);
        Z = std::complex<double>(i4fsc, q4fsc);
        return true;
    };

    if ((int)scratch_centerIQ.size() != width) scratch_centerIQ.resize(width);
    if ((int)scratch_upIQ.size() != width) scratch_upIQ.resize(width);
    if ((int)scratch_dnIQ.size() != width) scratch_dnIQ.resize(width);
    for (int x = 0; x < width; ++x) {
        std::complex<double> z;
        if (demodPrecleanAt(line, x, z)) scratch_centerIQ[x] = z;
        else scratch_centerIQ[x] = std::complex<double>((double)ti0_raw[x], (double)tq0_raw[x]);

        if (verticalAllowed && demodPrecleanAt(line - 1, x, z)) {
            scratch_upIQ[x] = z;
        } else {
            if (tiUp_raw && tqUp_raw) z = std::complex<double>((double)tiUp_raw[x], (double)tqUp_raw[x]);
            else z = std::complex<double>(0.0, 0.0);
            scratch_upIQ[x] = z;
        }

        if (verticalAllowed && demodPrecleanAt(line + 1, x, z)) {
            scratch_dnIQ[x] = z;
        } else {
            if (tiDn_raw && tqDn_raw) z = std::complex<double>((double)tiDn_raw[x], (double)tqDn_raw[x]);
            else z = std::complex<double>(0.0, 0.0);
            scratch_dnIQ[x] = z;
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
// Gate: suppress when the two neighbors disagree with each other (motion).
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

    outFrameIQ.assign(width, std::complex<double>(0.0, 0.0));
    if (width <= 0 || line < first || line >= last) return;
    if (line >= demodLines || demodWidth <= 0) return;

    // Preclean demod: Field B precleaned scalar -> 4fsc grid IQ.
    // The preclean is computed via computeSimpleFieldLine (the same function
    // that produces the main Field B candidate), so it carries the anti-phase
    // reach guard, authority blend, and combKMetric upgrades.
    auto demodPrecleanAt = [&](int ln, int x, std::complex<double> &Z)->bool {
        if (ln < first || ln >= last) return false;
        const double *row = precleanLinePtr(ln, width);
        if (!row) return false;
        const int h = left + x;
        double i4fsc = 0.0, q4fsc = 0.0;
        demod4fscFromComposite(row[x], h, i4fsc, q4fsc);
        Z = std::complex<double>(i4fsc, q4fsc);
        return true;
    };

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
    if (!ti0_raw || !tq0_raw) return;

    const bool haveUpLine = (verticalAllowed && line - 1 >= first);
    const bool haveDnLine = (verticalAllowed && line + 1 <  last);
    const CombTapLine &reachTapLine = ensureCombTapLine(line);

    if ((int)scratch_centerIQ.size() != width) scratch_centerIQ.resize(width);
    if ((int)scratch_upIQ.size() != width) scratch_upIQ.resize(width);
    if ((int)scratch_dnIQ.size() != width) scratch_dnIQ.resize(width);
    for (int x = 0; x < width; ++x) {
        std::complex<double> z;
        if (demodPrecleanAt(line, x, z))
            scratch_centerIQ[x] = z;
        else
            scratch_centerIQ[x] = { (double)ti0_raw[x], (double)tq0_raw[x] };

        if (haveUpLine && demodPrecleanAt(line - 1, x, z))
            scratch_upIQ[x] = z;
        else if (tiUp_raw && tqUp_raw)
            scratch_upIQ[x] = { (double)tiUp_raw[x], (double)tqUp_raw[x] };
        else
            scratch_upIQ[x] = { 0.0, 0.0 };

        if (haveDnLine && demodPrecleanAt(line + 1, x, z))
            scratch_dnIQ[x] = z;
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

    outFrameIQ.assign(width, std::complex<double>(0.0, 0.0));

    if (width <= 0 || line < first || line >= last)
        return;

    if ((int)centerIQ.size() < width ||
        (int)upIQ.size() < width ||
        (int)dnIQ.size() < width)
    {
        return;
    }

    const auto  &T    = configuration.tunables;
    const double invI = invIreScale;

    const double COMB_STRENGTH  = std::max(0.0, T.FRAME_B_COMB_STRENGTH);
    const double MIN_CHROMA_IRE = T.FRAME_CHROMA_MIN_IRE;

    const bool verticalAllowed = carrierFrameVerticalAllowed(line);
    const bool haveUpLine = verticalAllowed && (line - 1 >= first);
    const bool haveDnLine = verticalAllowed && (line + 1 <  last);

    auto corrAbs = [](const std::complex<double> &a,
                      const std::complex<double> &b) -> double
    {
        const double ma = cmag(a);
        const double mb = cmag(b);

        if (ma <= 1e-12 || mb <= 1e-12)
            return 0.0;

        const double dot = a.real() * b.real() + a.imag() * b.imag();
        return std::clamp(std::fabs(dot) / (ma * mb + 1e-12), 0.0, 1.0);
    };

    auto rampDown = [](double v, double lo, double hi) -> double {
        if (hi <= lo)
            return (v <= lo) ? 1.0 : 0.0;

        double t = (v - lo) / (hi - lo);
        t = std::clamp(t, 0.0, 1.0);

        return 1.0 - t;
    };

    auto rampUp = [](double v, double lo, double hi) -> double {
        if (hi <= lo)
            return (v >= hi) ? 1.0 : 0.0;

        double t = (v - lo) / (hi - lo);
        return std::clamp(t, 0.0, 1.0);
    };

    // Temporary post-VDIS Frame B safety.
    //
    // These are deliberately local and conservative. They do not recreate VDIS;
    // they only prevent Frame B from applying a full interfield correction when
    // its two one-line neighbors disagree or when center/neighbor IQ is weakly
    // related.
    const double NBR_DISAGREE_LO_IRE = 4.0;
    const double NBR_DISAGREE_HI_IRE = 14.0;

    const double CENTER_DISAGREE_LO_IRE = 5.0;
    const double CENTER_DISAGREE_HI_IRE = 18.0;

    const double COH_LO = 0.45;
    const double COH_HI = 0.82;

    const double MAX_DELTA_IRE = 18.0;

    for (int x = 0; x < width; ++x) {
        const std::complex<double> Z0 = centerIQ[x];
        const std::complex<double> ZUp = upIQ[x];
        const std::complex<double> ZDn = dnIQ[x];

        const double a0IRE = cmag(Z0) * invI;

        bool haveUp = haveUpLine && (cmag(ZUp) > 1e-9);
        bool haveDn = haveDnLine && (cmag(ZDn) > 1e-9);

        if (a0IRE < MIN_CHROMA_IRE || (!haveUp && !haveDn)) {
            outFrameIQ[x] = Z0;
            continue;
        }

        double upReach = 1.0;
        double dnReach = 1.0;

        if (reachTapLine &&
            x < reachTapLine->width &&
            x < (int)reachTapLine->pairU1.size() &&
            x < (int)reachTapLine->pairD1.size())
        {
            upReach = std::clamp(reachTapLine->pairU1[x].iqReachGate *
                                 reachTapLine->pairU1[x].contourReachGate, 0.0, 1.0);
            dnReach = std::clamp(reachTapLine->pairD1[x].iqReachGate *
                                 reachTapLine->pairD1[x].contourReachGate, 0.0, 1.0);
        }

        if (upReach <= 0.02)
            haveUp = false;

        if (dnReach <= 0.02)
            haveDn = false;

        if (!haveUp && !haveDn) {
            outFrameIQ[x] = Z0;
            continue;
        }

        const double dUp0IRE = haveUp ? (cmag(ZUp - Z0) * invI) : 1e9;
        const double dDn0IRE = haveDn ? (cmag(ZDn - Z0) * invI) : 1e9;

        double upWeight = 0.0;
        double dnWeight = 0.0;

        if (haveUp) {
            const double coh = corrAbs(Z0, ZUp);
            const double cohGate = rampUp(coh, COH_LO, COH_HI);
            const double centerGate = rampDown(dUp0IRE,
                                               CENTER_DISAGREE_LO_IRE,
                                               CENTER_DISAGREE_HI_IRE);

            upWeight = upReach * cohGate * centerGate;
        }

        if (haveDn) {
            const double coh = corrAbs(Z0, ZDn);
            const double cohGate = rampUp(coh, COH_LO, COH_HI);
            const double centerGate = rampDown(dDn0IRE,
                                               CENTER_DISAGREE_LO_IRE,
                                               CENTER_DISAGREE_HI_IRE);

            dnWeight = dnReach * cohGate * centerGate;
        }

        // If both neighbors exist but disagree with each other, reduce the
        // total authority. Do not pick alternating winners line-by-line unless
        // one side is clearly better.
        double neighborAgreementGate = 1.0;

        if (haveUp && haveDn) {
            const double dUDIRE = cmag(ZUp - ZDn) * invI;

            neighborAgreementGate = rampDown(dUDIRE,
                                             NBR_DISAGREE_LO_IRE,
                                             NBR_DISAGREE_HI_IRE);

            const double better = std::max(upWeight, dnWeight);
            const double worse  = std::min(upWeight, dnWeight);

            if (better > 1e-9 && worse < 0.35 * better) {
                if (upWeight < dnWeight)
                    upWeight = 0.0;
                else
                    dnWeight = 0.0;
            }
        }

        const double wsum = upWeight + dnWeight;

        if (wsum <= 1e-9) {
            outFrameIQ[x] = Z0;
            continue;
        }

        const std::complex<double> nbrAvg =
            (ZUp * upWeight + ZDn * dnWeight) / wsum;

        // Interfield correction proposal.
        const std::complex<double> delta = Z0 - nbrAvg;

        double deltaIRE = cmag(delta) * invI;
        std::complex<double> deltaClamped = delta;

        if (deltaIRE > MAX_DELTA_IRE && deltaIRE > 1e-9) {
            deltaClamped *= (MAX_DELTA_IRE / deltaIRE);
            deltaIRE = MAX_DELTA_IRE;
        }

        // Render-safe authority:
        // - local candidate evidence controls the correction
        // - neighbor disagreement suppresses it
        // - weak one-sided reach does not get full strength
        const double evidenceGate =
            std::clamp(0.5 * wsum, 0.0, 1.0);

        const double localStrength =
            COMB_STRENGTH * evidenceGate * neighborAgreementGate;

        outFrameIQ[x] = Z0 - deltaClamped * (0.5 * localStrength);

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

    outFrameIQ.assign(width, std::complex<double>(0.0, 0.0));
    outFrameScalar.assign(width, 0.0);
    if (width <= 0 || line < first || line >= last) return;
    if (line >= demodLines || demodWidth <= 0) return;

    computeFrameBDirectIQLine(line, outFrameIQ);

    for (int x = 0; x < width; ++x) {
        const int h = left + x;
        const auto &z = outFrameIQ[x];
        outFrameScalar[x] = remod4fscToCompositePhase(z.real(), z.imag(), carrierSampleClass(line, h));
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
