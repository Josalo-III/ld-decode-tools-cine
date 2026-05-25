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
#include <limits>

// VDIS - Vertical Differential Isolation System.
// Reduces artifacts at horizontal boundaries between different regions.
// We detect for strong vertical differentials in both chroma phase (IQ space)
// and scalar magnitude between upper and lower samples in the field.
// If checks fail, 1D only in FVF and 3D. Fields excluded from FVF if 2 fails.
// This is the heavy-handed lathe when nothing else works and 1D is tempting
void Comb::FrameBuffer::computeVDISLine(int lineNumber)
{
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;
    if (width <= 0) return;

    const auto &T = configuration.tunables;

    // Ensure flag buffer is sized and cleared
    if ((int)scratch_vdis_flag.size() < width) scratch_vdis_flag.resize(width, 0);
    else std::fill(scratch_vdis_flag.begin(), scratch_vdis_flag.end(), 0);

    // ----------------------------------------------------------------
    // Scalar (2) leg: amplitude-based disagreement
    // ----------------------------------------------------------------
    const int up2 = lineNumber - 2;
    const int dn2 = lineNumber + 2;
    const bool haveUp2 = (up2 >= first && up2 < last);
    const bool haveDn2 = (dn2 >= first && dn2 < last);

    if (haveUp2 && haveDn2) {
        const double th1d_ire = T.VDIS_1D_DIFF_THRESH_IRE;
        const double th1d_s   = (th1d_ire > 0.0) ? th1d_ire * irescale : 0.0;

        if (th1d_s > 0.0) {
            for (int rel = 0; rel < width; ++rel) {
                int h = left + rel;
                double c = clpbuffer[0].pixel[lineNumber][h];
                double u = clpbuffer[0].pixel[up2][h];
                double d = clpbuffer[0].pixel[dn2][h];
                double maxDiff = std::max(std::fabs(c - u), std::fabs(c - d));
                if (maxDiff > th1d_s) {
                    scratch_vdis_flag[rel] = 1;
                }
            }
        }
    }

    // ----------------------------------------------------------------
    // IQ (1) leg: chroma phase disagreement (VDIS_USE_PLUS1)
    // ----------------------------------------------------------------
    if (T.VDIS_USE_PLUS1 &&
        lineNumber >= first && lineNumber < last &&
        lineNumber < demodLines && demodWidth > 0)
    {
        const int up1 = lineNumber - 1;
        const int dn1 = lineNumber + 1;
        const bool haveUp1 = (up1 >= 0 && up1 < demodLines);
        const bool haveDn1 = (dn1 >= 0 && dn1 < demodLines);

        if (haveUp1 || haveDn1) {
            const float *ti0 = demodTI4fsc_line(lineNumber);
            const float *tq0 = demodTQ4fsc_line(lineNumber);
            const float *tiU = haveUp1 ? demodTI4fsc_line(up1) : nullptr;
            const float *tqU = haveUp1 ? demodTQ4fsc_line(up1) : nullptr;
            const float *tiD = haveDn1 ? demodTI4fsc_line(dn1) : nullptr;
            const float *tqD = haveDn1 ? demodTQ4fsc_line(dn1) : nullptr;

            const double minChroma = T.VDIS_MIN_CHROMA_IRE * irescale;
            const double cosThresh = std::cos(T.VDIS_PHASE_THRESH_DEG * M_PI / 180.0);

            const int W = std::min(width, demodWidth);
            for (int rel = 0; rel < W; ++rel) {
                    const std::complex<double> Z0((double)ti0[rel], (double)tq0[rel]);
                    double m0 = std::hypot(Z0.real(), Z0.imag());
                    if (m0 < minChroma) continue;

                    bool fire = false;

                    if (haveUp1) {
                        const std::complex<double> ZU((double)tiU[rel], (double)tqU[rel]);
                        double mU = std::hypot(ZU.real(), ZU.imag());
                        if (mU >= minChroma) {
                            double cosv = compareGridIQ(lineNumber, Z0, up1, ZU);
                            if (cosv < cosThresh) fire = true;
                        }
                    }
                    if (!fire && haveDn1) {
                        const std::complex<double> ZD((double)tiD[rel], (double)tqD[rel]);
                        double mD = std::hypot(ZD.real(), ZD.imag());
                        if (mD >= minChroma) {
                            double cosv = compareGridIQ(lineNumber, Z0, dn1, ZD);
                            if (cosv < cosThresh) fire = true;
                        }
                    }

                if (fire) scratch_vdis_flag[rel] = 1;
            }
        }
    }
}
// -------------------------------------------------------------------------
// Returns true if the consolidated VDIS mask has flagged position (lineNumber, h)
// as a vertical differential isolation region. The mask is populated by
// computeVDISLine and consolidated by consolidateVDISRegions during split2D.
bool Comb::FrameBuffer::hasVDIS(int lineNumber, int h) const
{
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;

    if (lineNumber < first || lineNumber >= last) return false;
    if (h < left || h >= right) return false;

    int rel = h - left;
    if (lineNumber < 0 || lineNumber >= (int)vdisMask.size()) return false;
    const auto &row = vdisMask[lineNumber];
    if (rel < 0 || rel >= (int)row.size()) return false;
    return row[rel] != 0;
}

// VDIS region consolidation
//
// Input:  vdisMask[line][rel] == 0/1 from computeVDISLine per-line flags.
// Output: vdisMask is rewritten to:
//   0 = no VDIS
//   1 = soft VDIS region
//   2 = hard VDIS region
//
// We use a small 3x3 neighbourhood count:
//   vcount >= HARD_MIN  => strong cluster => hard VDIS (2)
//   SOFT_MIN <=vcount< HARD_MIN => soft VDIS belt (1)
//   vcount <= NOISE_MAX => isolated speck => cleared to 0
//   else => keep original value.
void Comb::FrameBuffer::consolidateVDISRegions(
    std::vector<std::vector<char>> &vdisMask,
    const LdDecodeMetaData::VideoParameters &vp)
{
    const int firstLine = vp.firstActiveFrameLine;
    const int lastLine  = vp.lastActiveFrameLine;
    const int left      = vp.activeVideoStart;
    const int right     = vp.activeVideoEnd;
    const int width     = right - left;

    if (width <= 0) return;
    if ((int)vdisMask.size() < lastLine) return;

    // Thresholds; adjust as needed.
    const int HARD_MIN  = 5;  // strong 3x3 cluster -> hard VDIS
    const int SOFT_MIN  = 2;  // 2-4 neighbors   -> soft VDIS
    const int NOISE_MAX = 1;  // <= 1 neighbor   -> noise

    std::vector<std::vector<char>> outMask = vdisMask;

    for (int line = firstLine; line < lastLine; ++line) {
        if ((int)vdisMask[line].size() < width) continue;

        for (int rel = 0; rel < width; ++rel) {
            int vcount = 0;

            // Count VDIS flags in 3x3 neighbourhood
            for (int dy = -1; dy <= +1; ++dy) {
                int ln = line + dy;
                if (ln < firstLine || ln >= lastLine) continue;
                if ((int)vdisMask[ln].size() < width) continue;
                const auto &row = vdisMask[ln];

                for (int dx = -1; dx <= +1; ++dx) {
                    int rr = rel + dx;
                    if (rr < 0 || rr >= width) continue;
                    if (row[rr]) ++vcount;
                }
            }

            char newVal = 0;

            if (vcount >= HARD_MIN) {
                newVal = 2; // hard region
            } else if (vcount >= SOFT_MIN) {
                newVal = 1; // soft belt
            } else if (vcount <= NOISE_MAX) {
                newVal = 0; // speck -> clear
            } else {
                // mid case: keep original (usually 1)
                newVal = vdisMask[line][rel];
            }

            outMask[line][rel] = newVal;
        }
    }

    vdisMask.swap(outMask);
}

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
    double k = 0.0;
    k  = std::fabs(std::fabs(cc) - std::fabs(cn));
    k += std::fabs(symC - symN);
    k -= (std::fabs(cc) + std::fabs(cn)) * 0.10;
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
            if (ln < 0 || ln >= (int)locked1DSource.size()) return nullptr;
            const auto &row = locked1DSource[ln];
            if ((int)row.size() < width) return nullptr;
            return row.data();
        }
        return clpbuffer[0].pixel[ln] + left;
    };
    auto getTiRow = [&](int ln)->const float* {
        if (!configuration.phaseCompensation || ln < 0 || ln >= demodLines || demodWidth <= 0) return nullptr;
        return demodTI4fsc_line(ln);
    };
    auto getTqRow = [&](int ln)->const float* {
        if (!configuration.phaseCompensation || ln < 0 || ln >= demodLines || demodWidth <= 0) return nullptr;
        return demodTQ4fsc_line(ln);
    };

    struct RowRefs {
        int ln = -1;
        const double *comp = nullptr;
        const float *ti = nullptr;
        const float *tq = nullptr;
        bool haveLine = false;
    };
    auto rowRefs = [&](int ln, bool haveLine, bool wantIQ)->RowRefs {
        RowRefs r;
        r.ln = ln;
        r.haveLine = haveLine;
        r.comp = haveLine ? getCompRow(ln) : nullptr;
        r.ti = (haveLine && wantIQ) ? getTiRow(ln) : nullptr;
        r.tq = (haveLine && wantIQ) ? getTqRow(ln) : nullptr;
        return r;
    };

    RowRefs r0  = rowRefs(tapLine.ln0,  true,            wantIQ);
    RowRefs rU1 = rowRefs(tapLine.lnU1, tapLine.haveU1,  wantIQ && wantFrame);
    RowRefs rD1 = rowRefs(tapLine.lnD1, tapLine.haveD1,  wantIQ && wantFrame);
    RowRefs rU2 = rowRefs(tapLine.lnU2, tapLine.haveU2,  wantIQ && (wantFieldA || wantFieldB));
    RowRefs rD2 = rowRefs(tapLine.lnD2, tapLine.haveD2,  wantIQ && (wantFieldA || wantFieldB));
    RowRefs rU4 = rowRefs(tapLine.lnU4, tapLine.haveU4,  wantIQ && wantFieldA);
    RowRefs rD4 = rowRefs(tapLine.lnD4, tapLine.haveD4,  wantIQ && wantFieldA);

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
                iq.iqMag = std::hypot(ti, tq);
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

    auto coherenceAt = [&](const std::vector<CombTapIQ> &nbrIQ, bool haveNbrIQ, int neighborLine, int rel)->double {
        rel = std::clamp(rel, 0, width - 1);
        if (!tapLine.haveIQ0 || !haveNbrIQ) return 1.0;
        const CombTapIQ &c = tapLine.tap0IQ[rel];
        const CombTapIQ &n = nbrIQ[rel];
        if (c.iqMag * invI < 2.5 || n.iqMag * invI < 2.5) return 1.0;
        const std::complex<double> center((double)c.ti, (double)c.tq);
        const std::complex<double> neighbor((double)n.ti, (double)n.tq);
        const double corr = compareGridIQ(tapLine.ln0, center, neighborLine, neighbor);
        return std::clamp((corr - 0.55) / (0.85 - 0.55), 0.0, 1.0);
    };

    // Pair and contour fields are evidence, not decisions: candidates consume
    // them differently so Field A/B and Frame A/B keep their distinct behavior.
    auto fillPair = [&](const std::vector<CombTapScalar> &nbr,
                        const std::vector<CombTapIQ> &nbrIQ,
                        bool haveNbr,
                        bool haveNbrIQ,
                        int neighborLine,
                        std::vector<CombTapPair> &dst) {
        if (!haveNbr) {
            for (int rel = 0; rel < width; ++rel)
                dst[rel] = CombTapPair();
            return;
        }

        for (int rel = 0; rel < width; ++rel) {
            CombTapPair &p = dst[rel];
            p = CombTapPair();
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
                const std::complex<double> center((double)ciq.ti, (double)ciq.tq);
                const std::complex<double> neighbor((double)niq.ti, (double)niq.tq);
                p.iqDiffIRE = std::hypot(center.real() - neighbor.real(),
                                         center.imag() - neighbor.imag()) * invI;
            }
            if (configuration.phaseCompensation && haveNbrIQ) {
                const int relL = std::clamp(rel - 4, 0, width - 1);
                const int relR = std::clamp(rel + 4, 0, width - 1);
                p.coherence = (width >= 9)
                    ? std::min({coherenceAt(nbrIQ, haveNbrIQ, neighborLine, relL),
                                coherenceAt(nbrIQ, haveNbrIQ, neighborLine, rel),
                                coherenceAt(nbrIQ, haveNbrIQ, neighborLine, relR)})
                    : coherenceAt(nbrIQ, haveNbrIQ, neighborLine, rel);
            }
        }
    };

    if (wantFrame) {
        fillPair(tapLine.tapU1, tapLine.tapU1IQ, tapLine.haveU1, tapLine.haveIQU1, tapLine.lnU1, tapLine.pairU1);
        fillPair(tapLine.tapD1, tapLine.tapD1IQ, tapLine.haveD1, tapLine.haveIQD1, tapLine.lnD1, tapLine.pairD1);
    }
    if (wantFieldA || wantFieldB) {
        fillPair(tapLine.tapU2, tapLine.tapU2IQ, tapLine.haveU2, tapLine.haveIQU2, tapLine.lnU2, tapLine.pairU2);
        fillPair(tapLine.tapD2, tapLine.tapD2IQ, tapLine.haveD2, tapLine.haveIQD2, tapLine.lnD2, tapLine.pairD2);
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

    tapLine.builtFlags = flags;
}

// Field A - we sample 2 and 4 lines above and below, with the 4s asymmetrically 
// influencing the 2s,and 2s then influencing the evaluated pixel. Strictly intra-field.
void Comb::FrameBuffer::computeFieldALine(int lineNumber,
                                          double *outFieldLine,
                                          double  *outGate)
{
    const int width = videoParameters.activeVideoEnd - videoParameters.activeVideoStart;
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;

    if (width <= 0 || lineNumber < first || lineNumber >= last) {
        if (outFieldLine) std::fill(outFieldLine, outFieldLine + std::max(width, 0), 0.0);
        if (outGate)      std::fill(outGate,      outGate      + std::max(width, 0), 1.0f);
        return;
    }
    if (!outFieldLine) return;

    if (outGate) std::fill(outGate, outGate + width, 1.0f);

    const CombTapLine &tapLine = ensureCombTapLine(lineNumber);
    computeFieldALine(tapLine, outFieldLine, outGate);
}

void Comb::FrameBuffer::computeFieldALine(const CombTapLine &tapLine,
                                           double *outFieldLine,
                                           double *outGate)
{
    const int width = tapLine.width;
    if (width <= 0 || !outFieldLine || (int)tapLine.tap0.size() < width) return;

    if (outGate) std::fill(outGate, outGate + width, 1.0f);

    qint64 dbgN = 0;
    qint64 dbgCollapsedN = 0;
    double dbgSumGate = 0.0;
    double dbgSumAbsTc = 0.0;
    double dbgSumAbsC = 0.0;

    for (int rel = 0; rel < width; ++rel) {
        const double C    = tapLine.tap0[rel].comp;
        const double Cup2 = tapLine.tapU2[rel].comp;
        const double Cdn2 = tapLine.tapD2[rel].comp;
        const double Cup4 = tapLine.tapU4[rel].comp;
        const double Cdn4 = tapLine.tapD4[rel].comp;

        double wUp2 = tapLine.pairU2[rel].weight;
        double wDn2 = tapLine.pairD2[rel].weight;
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
                wUp2 = wDn2 = 1.0;
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

        if (configuration.debugPhaseLegs) {
            ++dbgN;
            dbgSumGate += gateA;
            dbgSumAbsTc += std::fabs(tc) * invIreScale;
            dbgSumAbsC += std::fabs(C) * invIreScale;
            if ((wUp2 + wDn2) < 0.10) ++dbgCollapsedN;
        }
    }

    if (configuration.debugPhaseLegs && dbgN > 0) {
        const double invN = 1.0 / (double)dbgN;
        qInfo().noquote() << QString("FieldAStats line=%1 n=%2 collapsed=%3 gate=%4 absTcIRE=%5 absCIRE=%6")
            .arg(tapLine.ln0)
            .arg(dbgN)
            .arg(dbgCollapsedN)
            .arg(dbgSumGate * invN, 0, 'f', 3)
            .arg(dbgSumAbsTc * invN, 0, 'f', 3)
            .arg(dbgSumAbsC * invN, 0, 'f', 3);
    }
}

// Field B
// Simplified Field comb as a FrameBuffer member:
// - uses only 2 vertical neighbours
void Comb::FrameBuffer::computeFieldBLine(int lineNumber, double *outFieldLine)
{
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int width = videoParameters.activeVideoEnd - videoParameters.activeVideoStart;

    if (width <= 0 || lineNumber < first || lineNumber >= last || !outFieldLine) {
        if (outFieldLine) std::fill(outFieldLine, outFieldLine + std::max(width, 0), 0.0);
        return;
    }

    const CombTapLine &tapLine = ensureCombTapLine(lineNumber);
    computeFieldBLine(tapLine, outFieldLine);
}

void Comb::FrameBuffer::computeFieldBLine(const CombTapLine &tapLine, double *outFieldLine)
{
    const int width = tapLine.width;
    if (width <= 0 || !outFieldLine || (int)tapLine.tap0.size() < width) {
        if (outFieldLine) std::fill(outFieldLine, outFieldLine + std::max(width, 0), 0.0);
        return;
    }

    for (int rel = 0; rel < width; ++rel) {
        const double C   = tapLine.tap0[rel].comp;
        const double Cup = tapLine.tapU2[rel].comp;
        const double Cdn = tapLine.tapD2[rel].comp;

        // Soft weights from pre-computed kRange magnitude agreement (same formula
        // as Luma Wow). In locked mode the tap .comp values already come from
        // locked1DSource (phase-aligned), so the weight already encodes phase
        // quality. A separate coherence multiplier is redundant and creates sharp
        // drops at color-region edges that feed splitIQ checkerboards.
        double wUp = tapLine.pairU2[rel].weight;
        double wDn = tapLine.pairD2[rel].weight;

        double tc = 0.0;
        if (wUp > 0.0 || wDn > 0.0) {
            if (wDn > 3.0 * wUp)      wUp = 0.0;
            else if (wUp > 3.0 * wDn) wDn = 0.0;

            const double denom = wUp + wDn;
            if (denom > 1e-9) {
                double sc = 2.0 / denom;
                if (sc < 1.0) sc = 1.0;
                tc  = (C - Cup) * wUp * sc;
                tc += (C - Cdn) * wDn * sc;
                tc *= 0.25;
            }
        } else {
            // Both neighbors dissimilar to center. If they agree with each other,
            // their differences point the same direction — use their average.
            const double dMag  = std::fabs(std::fabs(Cup) - std::fabs(Cdn));
            const double sumUD = std::fabs(Cup + Cdn);
            if (dMag - std::fabs(sumUD * 0.2) <= 0.0)
                tc = (2.0 * C - Cup - Cdn) * 0.25;
            // else tc = 0.0 — luma pass-through fallback
        }

        double maxPhysDelta = 0.0;
        if (configuration.phaseCompensation &&
            tapLine.haveIQ0 && tapLine.haveIQU2 && tapLine.haveIQD2)
        {
            double envClamp = 0.5 * tapLine.tap0IQ[rel].iqMag;
            envClamp = std::max(envClamp, 0.5 * tapLine.tapU2IQ[rel].iqMag);
            envClamp = std::max(envClamp, 0.5 * tapLine.tapD2IQ[rel].iqMag);

            const double diffUpIRE = tapLine.pairU2[rel].iqDiffIRE;
            const double diffDnIRE = tapLine.pairD2[rel].iqDiffIRE;
            double iqDeltaClamp = 0.0;
            if (std::isfinite(diffUpIRE))
                iqDeltaClamp = std::max(iqDeltaClamp, 0.325 * diffUpIRE * irescale);
            if (std::isfinite(diffDnIRE))
                iqDeltaClamp = std::max(iqDeltaClamp, 0.325 * diffDnIRE * irescale);

            const double envFloor = 0.35 * envClamp;
            maxPhysDelta = std::min(envClamp, std::max(envFloor, iqDeltaClamp));
        }
        if (maxPhysDelta <= 0.0)
            maxPhysDelta = std::max(std::fabs(C - Cup), std::fabs(C - Cdn)) * 0.65;
        outFieldLine[rel] = std::clamp(tc, -maxPhysDelta, maxPhysDelta);
    }
}

static inline double cmag(const std::complex<double> &z) { return std::hypot(z.real(), z.imag()); }
static inline double dotIQ(const std::complex<double> &a, const std::complex<double> &b) { return a.real()*b.real() + a.imag()*b.imag(); }

// FrameScalar / Frame A composite cancellation candidate.
// Interfield version of Field B:
// - uses adjacent frame lines (±1), not same-field lines (±2)
// - operates in composite space
// - when available, uses Field B-precleaned rows as the composite source so the
//   frame comb becomes a second-stage cancellation over Field B's cleanup
// - writes a composite residual candidate suitable for scratch_fieldBCenter
void Comb::FrameBuffer::computeFrameScalarLine(int lineNumber, double *outFrameLine)
{
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int width = videoParameters.activeVideoEnd - videoParameters.activeVideoStart;

    if (width <= 0 || lineNumber < first || lineNumber >= last || !outFrameLine) {
        if (outFrameLine) std::fill(outFrameLine, outFrameLine + std::max(width, 0), 0.0);
        return;
    }

    const CombTapLine &tapLine = ensureCombTapLine(lineNumber);
    computeFrameScalarLine(tapLine, outFrameLine);
}

void Comb::FrameBuffer::computeFrameScalarLine(const CombTapLine &tapLine, double *outFrameLine)
{
    const int left  = videoParameters.activeVideoStart;
    const int width = tapLine.width;

    if (width <= 0 || !outFrameLine || (int)tapLine.tap0.size() < width) {
        if (outFrameLine) std::fill(outFrameLine, outFrameLine + std::max(width, 0), 0.0);
        return;
    }

    const auto  &T    = configuration.tunables;
    const double invI = invIreScale;
    const bool haveUp = tapLine.haveU1;
    const bool haveDn = tapLine.haveD1;

    auto applyMat = [](double I, double Q, const double M[2][2], double &outI, double &outQ) {
        outI = M[0][0] * I + M[0][1] * Q;
        outQ = M[1][0] * I + M[1][1] * Q;
    };

    auto solveFamilyRotation = [&](const std::vector<CombTapIQ> &nbrIQ,
                                   bool haveNbr,
                                   bool haveNbrIQ,
                                   int bucketFamily,
                                   double Rm[2][2]) -> bool {
        Rm[0][0] = 1.0; Rm[0][1] = 0.0;
        Rm[1][0] = 0.0; Rm[1][1] = 1.0;

        if (!configuration.phaseCompensation || !haveNbr || !tapLine.haveIQ0 || !haveNbrIQ)
            return false;

        constexpr double MIN_FIT_IRE = 2.5;
        const double pMax = T.Y_LINE_MAX_PHASE_DEG * M_PI / 180.0;

        double STT[2][2] = {{0,0},{0,0}};
        double SRT[2][2] = {{0,0},{0,0}};
        int n = 0;

        for (int x = 0; x < width; ++x) {
            const int h = left + x;
            if ((carrierSampleClass(tapLine.ln0, h) & 1) != bucketFamily)
                continue;
            const CombTapIQ &c = tapLine.tap0IQ[x];
            const CombTapIQ &nb = nbrIQ[x];
            const double I0 = c.ti;
            const double Q0 = c.tq;
            const double In = nb.ti;
            const double Qn = nb.tq;

            const double a0 = c.iqMag * invI;
            const double an = nb.iqMag * invI;

            if (a0 < MIN_FIT_IRE || an < MIN_FIT_IRE)
                continue;

            STT[0][0] += In * In;
            STT[0][1] += In * Qn;
            STT[1][0] += Qn * In;
            STT[1][1] += Qn * Qn;

            SRT[0][0] += I0 * In;
            SRT[0][1] += I0 * Qn;
            SRT[1][0] += Q0 * In;
            SRT[1][1] += Q0 * Qn;

            ++n;
        }

        if (n < 32)
            return false;

        double STTinv[2][2];
        if (!mat2_inv(STT, STTinv))
            return false;

        double A[2][2];
        double U[2][2];
        mat2_mul(SRT, STTinv, A);
        polar_decompose_2x2(A, Rm, U);
        clamp_rotation_gain_shear(Rm, U, pMax,
                                  T.Y_LINE_ALLOW_GAIN_ON_IQ,
                                  T.Y_LINE_GAIN_MIN,
                                  T.Y_LINE_GAIN_MAX,
                                  T.Y_LINE_MAX_SHEAR);
        return true;
    };

    auto bucketErrorForNeighbor = [&](const std::vector<CombTapIQ> &nbrIQ,
                                      bool haveNbr,
                                      bool haveNbrIQ,
                                      int neighborLine,
                                      int bucketClass,
                                      double bucketPol,
                                      const double Rm[2][2]) -> std::pair<double, int> {
        constexpr double MIN_FIT_IRE = 2.5;
        double err = 0.0;
        int errN = 0;

        if (!configuration.phaseCompensation || !haveNbr || !tapLine.haveIQ0 || !haveNbrIQ)
            return {err, errN};

        for (int x = 0; x < width; ++x) {
            const int h = left + x;
            if (carrierSampleClass(tapLine.ln0, h) != bucketClass)
                continue;
            const CombTapIQ &c = tapLine.tap0IQ[x];
            const CombTapIQ &nb = nbrIQ[x];
            const double a0 = c.iqMag * invI;
            if (a0 < MIN_FIT_IRE)
                continue;

            double In = 0.0;
            double Qn = 0.0;
            applyMat(nb.ti, nb.tq, Rm, In, Qn);
            const std::complex<double> neighborIQ =
                orientGridIQToReference(tapLine.ln0,
                                        neighborLine,
                                        std::complex<double>(In, Qn),
                                        bucketPol);
            const double an = std::hypot(neighborIQ.real(), neighborIQ.imag()) * invI;
            if (an < MIN_FIT_IRE)
                continue;

            const double C0 = remodGridComposite(tapLine.ln0, h, c.ti, c.tq);
            const double Cn = remodGridComposite(tapLine.ln0,
                                                 h,
                                                 neighborIQ.real(),
                                                 neighborIQ.imag());
            err += std::fabs(C0 + Cn);
            ++errN;
        }

        return {err, errN};
    };

    double RmUp[2][2][2];
    double RmDn[2][2][2];
    double bucketPol[4] = {1.0, 1.0, 1.0, 1.0};

    solveFamilyRotation(tapLine.tapU1IQ, haveUp, tapLine.haveIQU1, 0, RmUp[0]);
    solveFamilyRotation(tapLine.tapU1IQ, haveUp, tapLine.haveIQU1, 1, RmUp[1]);
    solveFamilyRotation(tapLine.tapD1IQ, haveDn, tapLine.haveIQD1, 0, RmDn[0]);
    solveFamilyRotation(tapLine.tapD1IQ, haveDn, tapLine.haveIQD1, 1, RmDn[1]);

    for (int bucketClass = 0; bucketClass < 4; ++bucketClass) {
        double bestErr = std::numeric_limits<double>::infinity();
        double bestPol = 1.0;
        const int bucketFamily = bucketClass & 1;

        for (double candPol : {1.0, -1.0}) {
            double err = 0.0;
            int errN = 0;
            if (haveUp) {
                auto [e, n] = bucketErrorForNeighbor(tapLine.tapU1IQ, haveUp, tapLine.haveIQU1,
                                                     tapLine.lnU1, bucketClass, candPol,
                                                     RmUp[bucketFamily]);
                err += e;
                errN += n;
            }
            if (haveDn) {
                auto [e, n] = bucketErrorForNeighbor(tapLine.tapD1IQ, haveDn, tapLine.haveIQD1,
                                                     tapLine.lnD1, bucketClass, candPol,
                                                     RmDn[bucketFamily]);
                err += e;
                errN += n;
            }

            if (errN >= 16 && err < bestErr) {
                bestErr = err;
                bestPol = candPol;
            }
        }

        bucketPol[bucketClass] = bestPol;
    }

    // Ownership evidence for this line (populated by buildPhaseCorrected1D).
    // lumaClaim identifies alien-Y pixels where the sign convention is wrong
    // and the raw composite comb should be used instead.
    const int ownerLine = tapLine.ln0;
    const bool haveOwnership = configuration.phaseCompensation
        && ownerLine >= 0 && ownerLine < (int)ownershipEvidence.size()
        && (int)ownershipEvidence[ownerLine].size() >= width;
    const OwnershipEvidence *ownRow = haveOwnership
        ? ownershipEvidence[ownerLine].data() : nullptr;
    const double *pre0 = precleanLinePtr(tapLine.ln0, width);
    const double *preU = precleanLinePtr(tapLine.lnU1, width);
    const double *preD = precleanLinePtr(tapLine.lnD1, width);

    for (int rel = 0; rel < width; ++rel) {
        const int bucketClass = carrierSampleClass(tapLine.ln0, left + rel);
        const int bucketFamily = bucketClass & 1;
        const double (*RmUpUse)[2] = RmUp[bucketFamily];
        const double (*RmDnUse)[2] = RmDn[bucketFamily];

        // Save raw composite values before the sign convention overwrites them.
        // In the common 4fsc domain, alien Y maps identically on both fields
        // (it's a luma transient, not carrier-modulated), so the raw comb
        // rawC - rawCup cancels alien Y while the signed comb amplifies it.
        const double rawC   = pre0 ? pre0[rel] : tapLine.tap0[rel].comp;
        const double rawCup = haveUp ? (preU ? preU[rel] : tapLine.tapU1[rel].comp) : rawC;
        const double rawCdn = haveDn ? (preD ? preD[rel] : tapLine.tapD1[rel].comp) : rawC;

        double C = rawC;
        double Cup = rawCup;
        double Cdn = rawCdn;
        double I0c = 0.0, Q0c = 0.0;
        double IUc = 0.0, QUc = 0.0;
        double IDc = 0.0, QDc = 0.0;
        const bool haveCenterIQ = configuration.phaseCompensation && tapLine.haveIQ0;
        const bool haveUpIQ = configuration.phaseCompensation && haveUp && tapLine.haveIQU1;
        const bool haveDnIQ = configuration.phaseCompensation && haveDn && tapLine.haveIQD1;

        if (configuration.phaseCompensation) {
            const int h = left + rel;

            if (haveCenterIQ) {
                I0c = tapLine.tap0IQ[rel].ti;
                Q0c = tapLine.tap0IQ[rel].tq;
                C = remodGridComposite(tapLine.ln0, h, I0c, Q0c);
            }

            if (haveUpIQ) {
                double Iu = 0.0;
                double Qu = 0.0;
                applyMat(tapLine.tapU1IQ[rel].ti, tapLine.tapU1IQ[rel].tq, RmUpUse, Iu, Qu);
                const std::complex<double> upIQ =
                    orientGridIQToReference(tapLine.ln0,
                                            tapLine.lnU1,
                                            std::complex<double>(Iu, Qu),
                                            bucketPol[bucketClass]);
                IUc = upIQ.real();
                QUc = upIQ.imag();
                Cup = remodGridComposite(tapLine.ln0, h, IUc, QUc);
            }

            if (haveDnIQ) {
                double Id = 0.0;
                double Qd = 0.0;
                applyMat(tapLine.tapD1IQ[rel].ti, tapLine.tapD1IQ[rel].tq, RmDnUse, Id, Qd);
                const std::complex<double> dnIQ =
                    orientGridIQToReference(tapLine.ln0,
                                            tapLine.lnD1,
                                            std::complex<double>(Id, Qd),
                                            bucketPol[bucketClass]);
                IDc = dnIQ.real();
                QDc = dnIQ.imag();
                Cdn = remodGridComposite(tapLine.ln0, h, IDc, QDc);
            }
        }

        // Ownership blend factor for this pixel. As lumaClaim → 1, all gate
        // signals migrate from the signed domain to the raw composite domain,
        // so the transition to the unsigned regime is integrated and continuous.
        const double lcEff = ownRow ? ownRow[rel].assessment.lumaClaim : 0.0;

        // dynamicVThreshold: at sharp luma edges it tightens to 6 IRE to reject
        // motion/boundary noise, but alien-Y edges are exactly those pixels.
        // As lcEff rises, relax it back toward 10 so alien-Y pixels can enter
        // the softWeight path rather than forcing tc = C passthrough.
        const double edgeSqueeze = (tapLine.hLumaDeltaIRE[rel] > 12.0) ? 1.0 : 0.0;
        const double dynamicVThreshold = 10.0 - 4.0 * edgeSqueeze * (1.0 - lcEff);

        // Keep Frame A's local decision-making in scalar composite space.
        // Carrier alignment has already happened at the line level via the
        // bucket/sign setup and ownership blend, so repeating per-pixel IQ
        // coherence tests here tends to reintroduce checkerboard texture.
        const double cohUp = haveUp ? 1.0 : 0.0;
        const double cohDn = haveDn ? 1.0 : 0.0;

        const double scalarDiffUpIRE = haveUp
            ? std::fabs(C - Cup) * invI
            : std::numeric_limits<double>::infinity();
        const double scalarDiffDnIRE = haveDn
            ? std::fabs(C - Cdn) * invI
            : std::numeric_limits<double>::infinity();

        double diffUpIRE = scalarDiffUpIRE;
        double diffDnIRE = scalarDiffDnIRE;

        // Blend diff toward the raw composite diff as lcEff rises. In the raw
        // domain, alien-Y has the same value on both fields (common 4fsc frame),
        // so rawDiff ≈ 0 and easily passes the threshold.
        if (lcEff > 0.0) {
            const double rawDiffUpIRE = haveUp
                ? std::fabs(rawC - rawCup) * invI
                : std::numeric_limits<double>::infinity();
            const double rawDiffDnIRE = haveDn
                ? std::fabs(rawC - rawCdn) * invI
                : std::numeric_limits<double>::infinity();
            if (haveUp) diffUpIRE = diffUpIRE * (1.0 - lcEff) + rawDiffUpIRE * lcEff;
            if (haveDn) diffDnIRE = diffDnIRE * (1.0 - lcEff) + rawDiffDnIRE * lcEff;
        }

        auto softWeight = [&](bool haveNbr, double coh, double diffIRE) -> double {
            if (!haveNbr || !std::isfinite(diffIRE))
                return 0.0;
            // Frame A should degrade gracefully under 1D-style alternation rather
            // than dropping straight to preservation. Let coherence and diff shape
            // the weight continuously, with a soft shoulder beyond the nominal
            // threshold instead of a hard veto.
            const double diffNorm = diffIRE / std::max(dynamicVThreshold, 1e-9);
            const double diffT = std::clamp(1.15 - diffNorm, 0.0, 1.0);
            const double cohT  = std::clamp((coh - 0.20) / 0.80, 0.0, 1.0);
            return cohT * diffT;
        };

        const double wUp = softWeight(haveUp, cohUp, diffUpIRE);
        const double wDn = softWeight(haveDn, cohDn, diffDnIRE);

        double tc = C;
        const double wSum = wUp + wDn;
        const double tcAlt = (2.0 * C - Cup - Cdn) * 0.25;
        if (wSum > 1e-9) {
            const double nbrMix = (wUp * Cup + wDn * Cdn) / wSum;
            tc = 0.5 * (C - nbrMix);
        }

        if (haveUp && haveDn) {
            // Frame A exception: detect the same geometry the comb itself uses.
            // If Up and Down agree with each other in scalar composite space, and
            // their average strongly opposes the current line, that's the
            // alternating frame pattern we want to cancel rather than preserve.
            const double nbrMean = 0.5 * (Cup + Cdn);
            const double nbrAgreeIRE = std::fabs(Cup - Cdn) * invI;
            const double centerOppIRE = std::fabs(C - nbrMean) * invI;
            const double agreeT = std::clamp(
                1.0 - (nbrAgreeIRE / std::max(0.85 * dynamicVThreshold, 1e-9)),
                0.0, 1.0);
            const double opposeT = std::clamp(
                (centerOppIRE - 0.45 * dynamicVThreshold) /
                std::max(0.70 * dynamicVThreshold, 1e-9),
                0.0, 1.0);
            const double weakAdmissionT = std::clamp(1.0 - (wSum / 0.35), 0.0, 1.0);
            const double altT = agreeT * opposeT * weakAdmissionT;
            if (altT > 0.0)
                tc = tc * (1.0 - altT) + tcAlt * altT;
        }

        if (std::fabs(tc - C) < 1e-12 && haveUp && haveDn) {
            // Apply the fallback in scalar composite space; the candidate setup
            // above has already produced aligned composite witnesses.
            const double dMag  = std::fabs(std::fabs(Cup) - std::fabs(Cdn));
            const double sumUD = std::fabs(Cup + Cdn);
            if (dMag - std::fabs(sumUD * 0.2) <= 0.0)
                tc = tcAlt;
            // else tc = C — magnitude fallback also failed; preserve signal
        }

        // Ownership-guided alien-Y cancellation.
        // When lumaClaim is high, the sign convention is wrong for these pixels
        // (alien Y is a luma transient, not carrier-modulated, so the interfield
        // sign flip that aligns chroma instead anti-aligns alien Y).
        // Fall back to the raw composite comb which naturally cancels alien Y:
        // in the common 4fsc domain, alien Y is identical on both fields, so
        // rawC - rawCup ≈ 0 for alien Y while preserving partial chroma.
        if (ownRow) {
            const double lc = ownRow[rel].assessment.lumaClaim;
            if (lc > 0.0 && (haveUp || haveDn)) {
                double rawNbrMix;
                if (haveUp && haveDn)
                    rawNbrMix = 0.5 * (rawCup + rawCdn);
                else if (haveUp)
                    rawNbrMix = rawCup;
                else
                    rawNbrMix = rawCdn;
                const double tcRaw = 0.5 * (rawC - rawNbrMix);
                tc = tc * (1.0 - lc) + tcRaw * lc;
            }
        }

        double maxPhysDelta = 0.0;
        if (haveUp) maxPhysDelta = std::max(maxPhysDelta, std::fabs(C - Cup));
        if (haveDn) maxPhysDelta = std::max(maxPhysDelta, std::fabs(C - Cdn));
        maxPhysDelta *= 0.65;

        outFrameLine[rel] = std::clamp(tc, -maxPhysDelta, maxPhysDelta);
    }
}

// use buffers preprocessed by an intrafield comb to feed demod for IQ interfield comb
void Comb::FrameBuffer::computeFrameIQFromPreparedVectors(
    int line,
    const std::vector<std::complex<double>> &centerIQ,
    std::vector<std::complex<double>> &upIQ,
    std::vector<std::complex<double>> &dnIQ,
    std::vector<std::complex<double>> &outFrameIQ,
    const std::vector<float> *tiOverride,
    const std::vector<float> *tqOverride,
    bool enableLateralRefine,
    bool allowSymmetricLeakCancel)
{
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;

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

    auto reflectIndex = [&](int x)->int {
        if (x < 0) return -x;
        if (x >= width) return (width - 1) - (x - (width - 1));
        return x;
    };

    auto corrSigned = [&](int referenceLine,
                          const std::complex<double> &referenceIQ,
                          int sourceLine,
                          const std::complex<double> &sourceIQ)->double {
        return compareGridIQ(referenceLine, referenceIQ, sourceLine, sourceIQ);
    };

    // ------------------------------------------------------------
    // Helper: soft signed contribution
    // ------------------------------------------------------------
    auto softAlignContrib = [&](int referenceLine,
                                const std::complex<double> &Z0,
                                int sourceLine,
                                const std::complex<double> &Zn)->std::complex<double>
    {
        const double a0 = cmag(Z0);
        const std::complex<double> alignedZn =
            alignGridIQForComparison(referenceLine, sourceLine, Zn);
        const double an = cmag(alignedZn);
        if (a0 <= 1e-12 || an <= 1e-12) return {0.0, 0.0};

        const double c  = corrSigned(referenceLine, Z0, sourceLine, Zn);
        const double ac = std::fabs(c);

        const double t0 = 0.55;
        const double t1 = 0.85;

        double w = (ac - t0) / (t1 - t0);
        w = std::clamp(w, 0.0, 1.0);

        const double wFloor = 0.15;
        w = wFloor + (1.0 - wFloor) * w;

        const double s = (c >= 0.0) ? 1.0 : -1.0;
        return (alignedZn * (w * s));
    };

    // Companion: compute the same weight used by softAlignContrib (so we can do weighted averaging)
    auto softAlignWeight = [&](int referenceLine,
                               const std::complex<double> &Z0,
                               int sourceLine,
                               const std::complex<double> &Zn)->double
    {
        const double a0 = cmag(Z0);
        const std::complex<double> alignedZn =
            alignGridIQForComparison(referenceLine, sourceLine, Zn);
        const double an = cmag(alignedZn);
        if (a0 <= 1e-12 || an <= 1e-12) return 0.0;

        const double c  = corrSigned(referenceLine, Z0, sourceLine, Zn);
        const double ac = std::fabs(c);

        const double t0 = 0.55;
        const double t1 = 0.85;

        double w = (ac - t0) / (t1 - t0);
        w = std::clamp(w, 0.0, 1.0);

        const double wFloor = 0.15;
        w = wFloor + (1.0 - wFloor) * w;

        return w;
    };

    auto applyMat = [](const std::complex<double> &z, const double M[2][2])->std::complex<double> {
        const double I = z.real(), Q = z.imag();
        return std::complex<double>(M[0][0]*I + M[0][1]*Q,
                                    M[1][0]*I + M[1][1]*Q);
    };

    const double COMB_STRENGTH  = std::max(0.0, enableLateralRefine
                                                  ? T.FRAME_COMB_STRENGTH
                                                  : T.FRAME_B_COMB_STRENGTH);
    const double MAX_DELTA_IRE  = T.FRAME_IQ_RAW_MAX_DELTA_IRE;
    const double MIN_CHROMA_IRE = T.FRAME_CHROMA_MIN_IRE;

    const double VDIS_IQ_THRESH_IRE  = std::max(4.0, T.VDIS_MIN_CHROMA_IRE);
    const double VDIS_RAMP_RANGE_IRE = 4.0;

    // ------------------------------------------------------------
    // Build IQ vectors for center and 1 neighbors
    // ------------------------------------------------------------
    if ((int)centerIQ.size() < width || (int)upIQ.size() < width || (int)dnIQ.size() < width) return;

    // We treat the center line as mutable below (trim rotation), so take a copy.
    std::vector<std::complex<double>> center = centerIQ;

    // ------------------------------------------------------------
    // Per-line trim (rotation only)
    //
    // IMPORTANT: FrameIQ operates in canonical 4fsc bucket IQ (rsin/rcos) space.
    // The original trim solve compared against a locked/burst demod of the raw
    // composite, which is a different coordinate frame. That mismatch can produce
    // structured grid artifacts when inputs are canonical.
    //
    // For now, keep this trim disabled in canonical mode; if we need an
    // equivalent correction later, it must be derived entirely within the
    // canonical frame (e.g. by comparing candidates, not by re-demodding raw).
    // ------------------------------------------------------------
    auto applyLineTrimRm = [&](int ln, std::vector<std::complex<double>> &v)
    {
        (void)ln;
        (void)v;
        return;

    };

    applyLineTrimRm(line,   center);
    applyLineTrimRm(line-1, upIQ);
    applyLineTrimRm(line+1, dnIQ);

    // ------------------------------------------------------------
    // Per-neighbor affine-like solve => constrained rotation Rm (produceY-style)
    // ------------------------------------------------------------
    auto solveNeighborRotationFromAffine = [&](const std::vector<std::complex<double>> &nbr,
                                               double Rm[2][2])
    {
        Rm[0][0] = 1.0; Rm[0][1] = 0.0;
        Rm[1][0] = 0.0; Rm[1][1] = 1.0;

        double STT[2][2] = {{0,0},{0,0}};
        double SRT[2][2] = {{0,0},{0,0}};
        int n = 0;

        const double MIN_FIT_IRE = std::max(2.0, 0.5 * MIN_CHROMA_IRE);

        for (int x = 0; x < width; ++x) {
            const std::complex<double> Z0 = center[x];
            const std::complex<double> Zn = nbr[x];
            const double a0 = cmag(Z0);
            const double an = cmag(Zn);
            if (a0 * invI < MIN_FIT_IRE) continue;
            if (an * invI < MIN_FIT_IRE) continue;

            const double I0 = Z0.real(), Q0 = Z0.imag();
            const double In = Zn.real(), Qn = Zn.imag();

            STT[0][0] += In*In;
            STT[0][1] += In*Qn;
            STT[1][0] += Qn*In;
            STT[1][1] += Qn*Qn;

            SRT[0][0] += I0*In;
            SRT[0][1] += I0*Qn;
            SRT[1][0] += Q0*In;
            SRT[1][1] += Q0*Qn;

            ++n;
        }

        if (n < 64) return;

        double STTinv[2][2];
        if (!mat2_inv(STT, STTinv)) return;

        double A[2][2];
        mat2_mul(SRT, STTinv, A);

        double U[2][2];
        polar_decompose_2x2(A, Rm, U);

        const double pMax = T.Y_LINE_MAX_PHASE_DEG * M_PI / 180.0;
        clamp_rotation_gain_shear(Rm, U, pMax,
                                  T.Y_LINE_ALLOW_GAIN_ON_IQ,
                                  T.Y_LINE_GAIN_MIN, T.Y_LINE_GAIN_MAX,
                                  T.Y_LINE_MAX_SHEAR);
    };

    double RmUp[2][2], RmDn[2][2];
    solveNeighborRotationFromAffine(upIQ, RmUp);
    solveNeighborRotationFromAffine(dnIQ, RmDn);

    for (int x = 0; x < width; ++x) {
        upIQ[x] = applyMat(upIQ[x], RmUp);
        dnIQ[x] = applyMat(dnIQ[x], RmDn);
    }

    // ------------------------------------------------------------
    // Combine (soft signed contributions + boundary-aware asymmetry)
    // ------------------------------------------------------------
    for (int x = 0; x < width; ++x) {
        const std::complex<double> Z0 = center[x];
        const double a0 = cmag(Z0);
        const double a0_ire = a0 * invI;
        const bool weakCenterChroma = (a0_ire <= MIN_CHROMA_IRE);

        std::complex<double> ZUpRaw = upIQ[x];
        std::complex<double> ZDnRaw = dnIQ[x];

        if (enableLateralRefine) {
            // Frame A only: lateral pre-clean for the samples we comb with (Up/Dn only).
            auto refineNeighbor = [&](const std::vector<std::complex<double>> &nbr,
                                      int neighborLine,
                                      const std::complex<double> &Z,
                                      int xi)->std::complex<double>
            {
                const int xm1 = reflectIndex(xi - 1);
                const int xp1 = reflectIndex(xi + 1);
                const std::complex<double> Zm = nbr[xm1];
                const std::complex<double> Zp = nbr[xp1];

                const double c0 = std::fabs(corrSigned(neighborLine, Z, neighborLine, Zm));
                const double c1 = std::fabs(corrSigned(neighborLine, Z, neighborLine, Zp));
                const double cmin = std::min(c0, c1);

                const double T0 = 0.55;
                const double T1 = 0.85;
                double t = (cmin - T0) / (T1 - T0);
                t = std::clamp(t, 0.0, 1.0);

                const double alpha = 0.40 * t;
                if (alpha <= 0.0) return Z;

                const std::complex<double> Zlr = 0.5 * (Zm + Zp);
                return (Z + alpha * Zlr) / (1.0 + alpha);
            };

            ZUpRaw = refineNeighbor(upIQ, line - 1, ZUpRaw, x);
            ZDnRaw = refineNeighbor(dnIQ, line + 1, ZDnRaw, x);
        }

        const double aUp = cmag(ZUpRaw);
        const double aDn = cmag(ZDnRaw);

        const bool haveUp = (aUp > 1e-9);
        const bool haveDn = (aDn > 1e-9);

        if (!haveUp && !haveDn) {
            outFrameIQ[x] = Z0;
            continue;
        }

        // --- IQ-based VDIS gating (aligned space) ---
        double vdisGate = 1.0;
        double dUpDown_ire = 0.0;
        if (haveUp && haveDn) {
            dUpDown_ire = cmag(ZUpRaw - ZDnRaw) * invI;

            if (!allowSymmetricLeakCancel &&
                T.VDIS_HARD_FALLBACK &&
                dUpDown_ire > VDIS_IQ_THRESH_IRE) {
                outFrameIQ[x] = Z0;
                continue;
            }

            if (dUpDown_ire > VDIS_IQ_THRESH_IRE) {
                double t = (dUpDown_ire - VDIS_IQ_THRESH_IRE) / VDIS_RAMP_RANGE_IRE;
                t = std::clamp(t, 0.0, 1.0);
                double suppress = T.VDIS_SUPPRESS_FACTOR;
                vdisGate = 1.0 - (suppress * t);
                if (vdisGate < 0.0) vdisGate = 0.0;
            }
        }

        // --- Boundary limits for horizontal edges between disparate vertical regions ---
        // Default path can still asymmetrically favor one side, but the leakage
        // cleanup path below is explicitly symmetric and local at this x only.
        bool useUp = haveUp;
        bool useDn = haveDn;
        double boundaryWeightScale = 1.0;

        // Frame B local vertical cancellation:
        // for the locked-1D path, stay strictly at this x. Use only the current
        // sample and the immediate vertical neighbors; no horizontal borrowing
        // or side-picking. Signed-IQ evidence still modulates confidence.
        double leakageBoost = 0.0;
        double leakNeighborAgreement = 0.0;
        double leakCenterMismatch = 0.0;
        double leakSignedOpposition = 0.0;
        std::complex<double> Zsym(0.0, 0.0);
        const auto ramp01 = [](double value, double start, double full)->double {
            if (full <= start) return (value >= full) ? 1.0 : 0.0;
            return std::clamp((value - start) / (full - start), 0.0, 1.0);
        };
        if (allowSymmetricLeakCancel) {
            if (haveUp && haveDn) {
                Zsym = 0.5 * (ZUpRaw + ZDnRaw);
            } else if (haveUp) {
                Zsym = ZUpRaw;
            } else if (haveDn) {
                Zsym = ZDnRaw;
            }

            const double corrUp0 = haveUp ? corrSigned(line, Z0, line - 1, ZUpRaw) : 0.0;
            const double corrDn0 = haveDn ? corrSigned(line, Z0, line + 1, ZDnRaw) : 0.0;
            const double corrUD  = (haveUp && haveDn) ? corrSigned(line - 1, ZUpRaw, line + 1, ZDnRaw) : 0.0;

            const double negUp = haveUp
                ? ramp01(-corrUp0,
                         T.FRAME_B_LEAK_NEG_CORR_START,
                         T.FRAME_B_LEAK_NEG_CORR_FULL)
                : 0.0;
            const double negDn = haveDn
                ? ramp01(-corrDn0,
                         T.FRAME_B_LEAK_NEG_CORR_START,
                         T.FRAME_B_LEAK_NEG_CORR_FULL)
                : 0.0;
            const double neighAgree = (haveUp && haveDn)
                ? ramp01(corrUD,
                         T.FRAME_B_LEAK_NEIGHBOR_AGREE_START,
                         T.FRAME_B_LEAK_NEIGHBOR_AGREE_FULL)
                : 0.35;
            const double centerDelta = cmag(Zsym - Z0) * invI;
            const double centerMismatch = ramp01(centerDelta,
                                                 T.FRAME_B_LEAK_CENTER_DELTA_START_IRE,
                                                 T.FRAME_B_LEAK_CENTER_DELTA_FULL_IRE);
            const double signedOpposition = (haveUp && haveDn)
                ? std::min(negUp, negDn)
                : std::max(negUp, negDn);
            const double leakEvidence = std::clamp(
                0.80 * centerMismatch + 0.20 * signedOpposition,
                0.0, 1.0);

            leakNeighborAgreement = neighAgree;
            leakCenterMismatch = centerMismatch;
            leakSignedOpposition = signedOpposition;
            leakageBoost = std::max(leakEvidence, 0.35 * neighAgree);
        }
        const bool useSymmetricLeakCancel = allowSymmetricLeakCancel && (haveUp || haveDn);

        if (weakCenterChroma && !useSymmetricLeakCancel) {
            outFrameIQ[x] = Z0;
            continue;
        }

        if (!allowSymmetricLeakCancel && haveUp && haveDn) {
            const double dUp0_ire = cmag(ZUpRaw - Z0) * invI;
            const double dDn0_ire = cmag(ZDnRaw - Z0) * invI;

            const double EDGE_UD_IRE   = VDIS_IQ_THRESH_IRE; // reuse VDIS threshold as "disparate regions"
            const double MATCH_IRE     = 3.5;                // "center matches this side"
            const double BETWEEN_IRE   = 6.0;                // "center is far from this side"
            const double TRANS_SUPPRESS = 0.35;              // how much to suppress in transition zone

            if (dUpDown_ire > EDGE_UD_IRE) {
                if (dUp0_ire < MATCH_IRE && dDn0_ire > BETWEEN_IRE) {
                    useUp = true;  useDn = false;
                } else if (dDn0_ire < MATCH_IRE && dUp0_ire > BETWEEN_IRE) {
                    useDn = true;  useUp = false;
                } else {
                    if (dUp0_ire <= dDn0_ire) { useUp = true; useDn = false; }
                    else                      { useDn = true; useUp = false; }
                    boundaryWeightScale = TRANS_SUPPRESS;
                }
            }
        }

        std::complex<double> Zframe = Z0;
        if (useSymmetricLeakCancel) {
            Zframe = Zsym;
            useUp = true;
            useDn = true;
        } else {
            std::complex<double> Zsum = Z0;
            double wsum = 1.0;

            if (useUp) {
                const double w = softAlignWeight(line, Z0, line - 1, ZUpRaw) * boundaryWeightScale;
                if (w > 0.0) {
                    Zsum += softAlignContrib(line, Z0, line - 1, ZUpRaw) * boundaryWeightScale;
                    wsum += w;
                }
            }
            if (useDn) {
                const double w = softAlignWeight(line, Z0, line + 1, ZDnRaw) * boundaryWeightScale;
                if (w > 0.0) {
                    Zsum += softAlignContrib(line, Z0, line + 1, ZDnRaw) * boundaryWeightScale;
                    wsum += w;
                }
            }
            Zframe = Zsum / wsum;
        }

        std::complex<double> delta = Zframe - Z0;
        double deltaMagIRE = cmag(delta) * invI;

        double motionGate = 1.0; // placeholder for motion gating
        double gate = motionGate * vdisGate;
        if (useSymmetricLeakCancel) {
            const double gateFloor = 0.60 + (0.30 * leakageBoost);
            gate = std::max(gate, gateFloor);
        }

        double effMaxDeltaIRE = MAX_DELTA_IRE * gate;
        if (useSymmetricLeakCancel) {
            effMaxDeltaIRE *= 1.20 + (0.55 * leakageBoost);
        }

        if (deltaMagIRE > effMaxDeltaIRE && deltaMagIRE > 1e-9) {
            delta *= (effMaxDeltaIRE / deltaMagIRE);
            deltaMagIRE = effMaxDeltaIRE;
        }

        const double COMB_STRENGTH_HI = COMB_STRENGTH;
        const double COMB_STRENGTH_LO = std::min(0.5, COMB_STRENGTH_HI);

        double coh = 0.0;
        if (useSymmetricLeakCancel) {
            coh = std::fabs(corrSigned(line, Z0, line, Zsym));
        } else {
            if (useUp) coh = std::max(coh, std::fabs(corrSigned(line, Z0, line - 1, ZUpRaw)));
            if (useDn) coh = std::max(coh, std::fabs(corrSigned(line, Z0, line + 1, ZDnRaw)));
        }

        // Map coherence -> [0..1]
        const double COH_T0 = 0.55;
        const double COH_T1 = 0.85;
        double cohGate = (coh - COH_T0) / (COH_T1 - COH_T0);
        cohGate = std::clamp(cohGate, 0.0, 1.0);
        if (useSymmetricLeakCancel) {
            const double leakCoherenceFloor =
                0.35 + (0.35 * leakNeighborAgreement) + (0.20 * leakCenterMismatch);
            cohGate = std::max(cohGate, std::clamp(leakCoherenceFloor, 0.0, 1.0));
        }

        // Vertical agreement gate (1 when Up/Dn agree; 0 when they disagree strongly)
        double disGate = 1.0;
        if (haveUp && haveDn) {
            const double dUD_ire = cmag(ZUpRaw - ZDnRaw) * invI;
            double t = (dUD_ire - VDIS_IQ_THRESH_IRE) / VDIS_RAMP_RANGE_IRE;
            t = std::clamp(t, 0.0, 1.0);
            disGate = 1.0 - t;
        }
        if (useSymmetricLeakCancel) {
            const double leakDisagreementFloor =
                0.45 + (0.35 * leakNeighborAgreement) + (0.10 * leakSignedOpposition);
            disGate = std::max(disGate, std::clamp(leakDisagreementFloor, 0.0, 1.0));
        }

        double strengthMix = cohGate * disGate;

        if (useSymmetricLeakCancel) {
            const double leakStrengthFloor =
                0.60 + (0.25 * leakageBoost) + (0.10 * leakCenterMismatch);
            strengthMix = std::max(strengthMix, std::clamp(leakStrengthFloor, 0.0, 1.0));
            strengthMix = std::clamp(
                strengthMix + (T.FRAME_B_LEAK_STRENGTH_BOOST * leakageBoost),
                0.0, 1.0);
        } else {
            // Make it a bit more selective without hard switching
            strengthMix = strengthMix * strengthMix; // gamma=2
        }

        const double combStrengthLo = useSymmetricLeakCancel
            ? std::min(COMB_STRENGTH_HI, 0.90)
            : COMB_STRENGTH_LO;
        double localStrength =
            combStrengthLo + (COMB_STRENGTH_HI - combStrengthLo) * strengthMix;

        // Provisional output (before optional under-comb correction)
        std::complex<double> Zout = Z0 + (delta * localStrength * gate);

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
// process intrafield scalar comb as prep for interfield comb
void Comb::FrameBuffer::computeFrameALine(
    int line,
    std::vector<std::complex<double>> &outFrameIQ,
    bool enableLateralRefine)
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
        demodGridComposite(ln, h, c, i4fsc, q4fsc);
        Z = std::complex<double>(i4fsc, q4fsc);
        return true;
    };

    scratch_centerIQ.resize(width);
    scratch_upIQ.resize(width);
    scratch_dnIQ.resize(width);
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

    computeFrameIQFromPreparedVectors(line, scratch_centerIQ, scratch_upIQ, scratch_dnIQ, outFrameIQ, tiOverride, tqOverride, enableLateralRefine, false);
}

const Comb::FrameBuffer::FrameBLineCache *
Comb::FrameBuffer::ensureFrameBLine(int lineNumber)
{
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;

    if (!configuration.phaseCompensation) return nullptr;
    if (width <= 0 || lineNumber < first || lineNumber >= last) return nullptr;
    if (lineNumber < 0 || lineNumber >= demodLines || demodWidth <= 0) return nullptr;

    const int slot = precleanRingSlot(lineNumber);
    FrameBLineCache &cache = frameBRing[slot];

    if (cache.valid &&
        cache.line == lineNumber &&
        (int)cache.ti.size() == width &&
        (int)cache.tq.size() == width) {
        return &cache;
    }

    cache.line = lineNumber;
    cache.valid = false;

    cache.ti.assign(width, 0.0f);
    cache.tq.assign(width, 0.0f);

    if (!clpbuffer[0].pixel[lineNumber])
        return nullptr;

    const double *src = clpbuffer[0].pixel[lineNumber];
    for (int x = 0; x < width; ++x) {
        const int h = left + x;
        double i4fsc = 0.0;
        double q4fsc = 0.0;
        demodGridComposite(lineNumber, h, src[h], i4fsc, q4fsc);
        cache.ti[x] = static_cast<float>(i4fsc);
        cache.tq[x] = static_cast<float>(q4fsc);
    }

    cache.valid = true;
    return &cache;
}

void Comb::FrameBuffer::accumulateFrameBDebugStats(
    int line,
    const std::vector<std::complex<double>> &outFrameIQ,
    const std::vector<double> &outFrameScalar)
{
    if (!configuration.debugPhaseLegs || !configuration.phaseCompensation)
        return;

    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;
    if (width <= 8 || line <= first || line + 1 >= last)
        return;
    if ((int)outFrameIQ.size() < width || (int)outFrameScalar.size() < width)
        return;

    const auto *center = ensureFrameBLine(line);
    const auto *up = ensureFrameBLine(line - 1);
    const auto *dn = ensureFrameBLine(line + 1);
    if (!center || !up || !dn || !center->valid || !up->valid || !dn->valid)
        return;

    if (line < 0 || line >= (int)locked1DSource.size() ||
        line - 1 < 0 || line + 1 >= (int)locked1DSource.size())
        return;
    if ((int)locked1DSource[line].size() < width ||
        (int)locked1DSource[line - 1].size() < width ||
        (int)locked1DSource[line + 1].size() < width)
        return;

    const double invI = invIreScale;
    constexpr double EDGE_GATE_IRE = 4.0;

    for (int rel = 4; rel < width - 4; ++rel) {
        const double vEdgeIRE = std::max(
            std::fabs(locked1DSource[line][rel] - locked1DSource[line - 1][rel]),
            std::fabs(locked1DSource[line][rel] - locked1DSource[line + 1][rel])) * invI;
        if (vEdgeIRE < EDGE_GATE_IRE)
            continue;

        const int h = left + rel;
        const int phase = carrierSampleClass(line, h);

        const std::complex<double> z0(
            static_cast<double>(center->ti[rel]),
            static_cast<double>(center->tq[rel]));
        const std::complex<double> zu(
            static_cast<double>(up->ti[rel]),
            static_cast<double>(up->tq[rel]));
        const std::complex<double> zd(
            static_cast<double>(dn->ti[rel]),
            static_cast<double>(dn->tq[rel]));

        const std::complex<double> edgeDir = zu - zd;
        const double edgeMagSq = edgeDir.real() * edgeDir.real() +
                                 edgeDir.imag() * edgeDir.imag();
        if (edgeMagSq <= 1e-12)
            continue;

        auto accumulateOne = [&](const std::complex<double> &z,
                                 FrameBDebugLegStats &s)
        {
            const double dUp = std::hypot((z - zu).real(), (z - zu).imag()) * invI;
            const double dDn = std::hypot((z - zd).real(), (z - zd).imag()) * invI;
            const double signedBias = dDn - dUp;
            const std::complex<double> fromBase = z - z0;
            const double signedBaseShift =
                ((fromBase.real() * edgeDir.real()) +
                 (fromBase.imag() * edgeDir.imag())) /
                std::sqrt(edgeMagSq) * invI;
            ++s.edgeN;
            s.sumEdgeBias += signedBias;
            s.sumAbsEdgeBias += std::fabs(signedBias);
            s.sumBaseShift += signedBaseShift;
            s.sumAbsBaseShift += std::fabs(signedBaseShift);
        };

        accumulateOne(outFrameIQ[rel], frameBDebugStats.iqLegs[phase]);

        double ti = 0.0, tq = 0.0;
        demodGridComposite(line, h, outFrameScalar[rel], ti, tq);
        accumulateOne(std::complex<double>(ti, tq), frameBDebugStats.scalarLegs[phase]);
    }
}
// Frame B — direct interfield comb in BurstLockedSigned IQ.
//
// This path is intentionally aggressive. It forms the ±1 vertical-neighbour
// average on the current line's 4fsc grid, then uses that witness to comb the
// current sample directly. No preclean, no lateral help, no confidence gating:
// the purpose of Frame B is to be the plain vertical frame comb in the locked
// IQ domain.
//
// Edges and lines where the carrier schedule forbids vertical comparison
// (carrierFrameVerticalAllowed == false) fall through to zero IQ; the caller
// gates on that anyway. Otherwise we use both neighbours when present and a
// single neighbour at frame edges.
//
// The scalar output is the IQ result remodulated back to composite at this
// line's own 4fsc carrier grid, matching how Field A/B scalars are exported.
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

    outFrameIQ.assign(width, std::complex<double>(0.0, 0.0));
    outFrameScalar.assign(width, 0.0);

    if (width <= 0 || line < first || line >= last) return;
    if (line >= demodLines || demodWidth <= 0) return;
    if (!carrierFrameVerticalAllowed(line)) return;

    const FrameBLineCache *center = ensureFrameBLine(line);
    const FrameBLineCache *up     = ensureFrameBLine(line - 1);
    const FrameBLineCache *dn     = ensureFrameBLine(line + 1);

    const bool haveCenter =
        center && center->valid && center->line == line &&
        (int)center->ti.size() == width &&
        (int)center->tq.size() == width;

    const bool haveUp =
        up && up->valid && up->line == line - 1 &&
        (int)up->ti.size() == width &&
        (int)up->tq.size() == width;

    const bool haveDn =
        dn && dn->valid && dn->line == line + 1 &&
        (int)dn->ti.size() == width &&
        (int)dn->tq.size() == width;

    if (!haveCenter || (!haveUp && !haveDn)) return;

    const auto &T = configuration.tunables;
    const double combStrength = std::clamp(T.FRAME_B_COMB_STRENGTH, 0.0, 1.0);
    const double invI = invIreScale;

    // ---------------------------------------------------------------------
    // Frame B phase-family solve
    //
    // ensureFrameBLine() demodulates each source line in that source line's
    // grammar. Frame B, however, does a subtractive interfield comb on the
    // current line. Therefore the neighbours must be transported into the
    // current line's cancellation family before they are averaged.
    //
    // Solve one polarity per carrierSampleClass bucket. This keeps the fix
    // line/bucket scoped and avoids per-pixel sign flipping.
    //
    // The error metric is cancellation-domain composite disagreement:
    //
    //     minimize abs(C0 + Cn)
    //
    // not same-color similarity. A same-color orientation is exactly what can
    // drive saturated chroma toward black under 0.5 * (Z0 - Zavg).
    // ---------------------------------------------------------------------
    double bucketPol[4] = { 1.0, 1.0, 1.0, 1.0 };

    constexpr double MIN_SOLVE_CHROMA_IRE = 2.5;
    constexpr int    MIN_BUCKET_SOLVE_N   = 16;

    auto cachedIQAt = [](const FrameBLineCache *c, int x) -> std::complex<double> {
        return std::complex<double>(
            static_cast<double>(c->ti[x]),
            static_cast<double>(c->tq[x]));
    };

    auto tryAccumulatePolarityError =
        [&](const std::complex<double> &Z0,
            int h,
            const FrameBLineCache *nbr,
            int nbrLine,
            double pol,
            double &err,
            int &n)
    {
        if (!nbr) return;

        const std::complex<double> ZnRaw = cachedIQAt(nbr, h - left);
        const double magNIRE = std::hypot(ZnRaw.real(), ZnRaw.imag()) * invI;
        if (magNIRE < MIN_SOLVE_CHROMA_IRE)
            return;

        const std::complex<double> Zn =
            orientGridIQToReference(line, nbrLine, ZnRaw, pol);

        const double C0 = remodGridComposite(line, h, Z0.real(), Z0.imag());
        const double Cn = remodGridComposite(line, h, Zn.real(), Zn.imag());

        err += std::fabs(C0 + Cn);
        ++n;
    };

    for (int bucketClass = 0; bucketClass < 4; ++bucketClass) {
        double bestErr = std::numeric_limits<double>::infinity();
        double bestPol = 1.0;

        for (double pol : { 1.0, -1.0 }) {
            double err = 0.0;
            int n = 0;

            for (int x = 0; x < width; ++x) {
                const int h = left + x;
                if (carrierSampleClass(line, h) != bucketClass)
                    continue;

                const std::complex<double> Z0 = cachedIQAt(center, x);
                const double mag0IRE = std::hypot(Z0.real(), Z0.imag()) * invI;
                if (mag0IRE < MIN_SOLVE_CHROMA_IRE)
                    continue;

                if (haveUp)
                    tryAccumulatePolarityError(Z0, h, up, line - 1, pol, err, n);
                if (haveDn)
                    tryAccumulatePolarityError(Z0, h, dn, line + 1, pol, err, n);
            }

            if (n >= MIN_BUCKET_SOLVE_N && err < bestErr) {
                bestErr = err;
                bestPol = pol;
            }
        }

        bucketPol[bucketClass] = bestPol;
    }

    // ---------------------------------------------------------------------
    // Direct Frame B comb
    //
    // Keep the candidate's personality:
    //   - no lateral refinement
    //   - no election-style safety net
    //   - active leg only: I for bucket family 0, Q for bucket family 1
    //
    // The only changed contract is that neighbour witnesses are now transported
    // into the current line's cancellation orientation before averaging.
    // ---------------------------------------------------------------------
    for (int x = 0; x < width; ++x) {
        const int h = left + x;
        const int bucketClass = carrierSampleClass(line, h);

        const std::complex<double> Z0 = cachedIQAt(center, x);

        std::complex<double> ZuC(0.0, 0.0);
        std::complex<double> ZdC(0.0, 0.0);

        if (haveUp) {
            const std::complex<double> ZuRaw = cachedIQAt(up, x);
            ZuC = orientGridIQToReference(
                line,
                line - 1,
                ZuRaw,
                bucketPol[bucketClass]);
        }

        if (haveDn) {
            const std::complex<double> ZdRaw = cachedIQAt(dn, x);
            ZdC = orientGridIQToReference(
                line,
                line + 1,
                ZdRaw,
                bucketPol[bucketClass]);
        }

        std::complex<double> Zavg(0.0, 0.0);
        if (haveUp && haveDn)      Zavg = 0.5 * (ZuC + ZdC);
        else if (haveUp)           Zavg = ZuC;
        else if (haveDn)           Zavg = ZdC;
        else {
            outFrameIQ[x] = Z0;
            continue;
        }

        double outI = Z0.real();
        double outQ = Z0.imag();

        if ((bucketClass & 1) == 0) {
            const double Itarget = 0.5 * (Z0.real() - Zavg.real());
            outI = Z0.real() + ((Itarget - Z0.real()) * combStrength);
        } else {
            const double Qtarget = 0.5 * (Z0.imag() - Zavg.imag());
            outQ = Z0.imag() + ((Qtarget - Z0.imag()) * combStrength);
        }

        outFrameIQ[x] = std::complex<double>(outI, outQ);
    }

    // Remodulate to composite at this line's 4fsc carrier grid for the
    // scalar export consumed by the FVF scorer's Frame B candidate.
    for (int x = 0; x < width; ++x) {
        const int h = left + x;
        const auto &z = outFrameIQ[x];
        outFrameScalar[x] = remodGridComposite(line, h, z.real(), z.imag());
    }

    accumulateFrameBDebugStats(line, outFrameIQ, outFrameScalar);
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

    // Bounds check
    if ((unsigned)(lineNumber - firstLine) >= (unsigned)(lastLine - firstLine) ||
        (unsigned)(refLineNumber - firstLine) >= (unsigned)(lastLine - firstLine)) {
        result.penalty = 1000.0;
        return result;
    }

    // Phase check
    const qint32 wantPhase = carrierOppositeSampleClass(refLineNumber, refH);
    const qint32 havePhase = frameBuffer.carrierSignedSampleClass(lineNumber, h);
    if (wantPhase != havePhase) {
        result.penalty = 1000.0;
        return result;
    }

    // VDIS veto
    auto clampH = [&](int idx)->int {
        if (idx < left) return left;
        if (idx >= right) return right - 1;
        return idx;
    };
    const int hh = clampH(h);
    if (frameBuffer.hasVDIS(lineNumber, hh)) {
        return result;
    }

    // 1D sample
    result.sample = frameBuffer.clpbuffer[0].pixel[lineNumber][hh];

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
    const double cand0Aligned = orientGridScalarToReference(refLineNumber, lineNumber, candClpC[c0]);
    const double cand1Aligned = orientGridScalarToReference(refLineNumber, lineNumber, candClpC[c1]);
    const double cand2Aligned = orientGridScalarToReference(refLineNumber, lineNumber, candClpC[c2]);

    double iqPen = (std::fabs(refClpC[r0] - cand0Aligned) * 0.5 +
                    std::fabs(refClpC[r1] - cand1Aligned) * 1.0 +
                    std::fabs(refClpC[r2] - cand2Aligned) * 0.5) / 2.0;
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
