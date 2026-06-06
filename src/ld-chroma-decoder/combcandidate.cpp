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
            auto vdisSample = [&](int ln, int rel)->double {
                if (configuration.phaseCompensation) {
                    const double *row = locked1DSource_line(ln);
                    if (row && rel >= 0 && rel < width)
                        return row[rel];
                    return 0.0;
                }
                return clpbuffer[0].pixel[ln][left + rel];
            };
            for (int rel = 0; rel < width; ++rel) {
                double c = vdisSample(lineNumber, rel);
                double u = vdisSample(up2, rel);
                double d = vdisSample(dn2, rel);
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
                double I0 = ti0[rel];
                double Q0 = tq0[rel];
                double m0 = std::hypot(I0, Q0);
                if (m0 < minChroma) continue;

                bool fire = false;

                if (haveUp1) {
                    double IU = tiU[rel], QU = tqU[rel];
                    double mU = std::hypot(IU, QU);
                    if (mU >= minChroma) {
                        double dot = I0 * IU + Q0 * QU;
                        double cosv = dot / (m0 * mU + 1e-12);
                        if (cosv < cosThresh) fire = true;
                    }
                }
                if (!fire && haveDn1) {
                    double ID = tiD[rel], QD = tqD[rel];
                    double mD = std::hypot(ID, QD);
                    if (mD >= minChroma) {
                        double dot = I0 * ID + Q0 * QD;
                        double cosv = dot / (m0 * mD + 1e-12);
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
            return locked1DSource_line(ln);
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
        fillPair(tapLine.tapU1, tapLine.tapU1IQ, lumaU1, tapLine.haveU1, tapLine.haveIQU1, tapLine.pairU1);
        fillPair(tapLine.tapD1, tapLine.tapD1IQ, lumaD1, tapLine.haveD1, tapLine.haveIQD1, tapLine.pairD1);
    }
    if (wantFieldA || wantFieldB) {
        fillPair(tapLine.tapU2, tapLine.tapU2IQ, lumaU2, tapLine.haveU2, tapLine.haveIQU2, tapLine.pairU2);
        fillPair(tapLine.tapD2, tapLine.tapD2IQ, lumaD2, tapLine.haveD2, tapLine.haveIQD2, tapLine.pairD2);
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

            upPair[rel].iqReachGate = upGate;
            dnPair[rel].iqReachGate = dnGate;
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
    computeContourFieldLine(tapLine, outFieldLine, outGate);
}

void Comb::FrameBuffer::computeContourFieldLine(const CombTapLine &tapLine,
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

        const double reachUp2 = tapLine.pairU2[rel].reachGate;
        const double reachDn2 = tapLine.pairD2[rel].reachGate;
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

void Comb::FrameBuffer::computeSimpleFieldLine(const CombTapLine &tapLine, double *outFieldLine)
{
    const int width = tapLine.width;
    if (width <= 0 || !outFieldLine || (int)tapLine.tap0.size() < width) {
        if (outFieldLine) std::fill(outFieldLine, outFieldLine + std::max(width, 0), 0.0);
        return;
    }

    const bool have4 = tapLine.haveU4 && tapLine.haveD4 &&
                        (int)tapLine.tapU4.size() >= width &&
                        (int)tapLine.tapD4.size() >= width &&
                        (int)tapLine.contour.size() >= width;

    for (int rel = 0; rel < width; ++rel) {
        const double C   = tapLine.tap0[rel].comp;
        double Cup = tapLine.tapU2[rel].comp;
        double Cdn = tapLine.tapD2[rel].comp;

        if (have4) {
            const CombTapContour &curve = tapLine.contour[rel];
            if (curve.upSideOk > 0.5 && curve.dnSideOk > 0.5) {
                const double Cup4 = tapLine.tapU4[rel].comp;
                const double Cdn4 = tapLine.tapD4[rel].comp;
                auto refineNear = [](double nearS, double farS, double influence) {
                    if (influence <= 0.0 || nearS == 0.0) return nearS;
                    if ((nearS > 0.0) != (farS > 0.0)) return nearS;
                    const double nearMag = std::fabs(nearS);
                    const double farMag  = std::fabs(farS);
                    return std::copysign(
                        (nearMag + influence * farMag) / (1.0 + influence), nearS);
                };
                Cup = refineNear(Cup, Cup4, curve.upInfluence);
                Cdn = refineNear(Cdn, Cdn4, curve.dnInfluence);
            }
        }

        // Soft weights from pre-computed kRange magnitude agreement (same formula
        // as Luma Wow). In locked mode the tap .comp values already come from
        // locked1DSource_flat (phase-aligned), so the weight already encodes phase
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
    const double pMax = T.Y_LINE_MAX_PHASE_DEG * M_PI / 180.0;
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

    const double VDIS_IQ_THRESH_IRE  = std::max(4.0, T.VDIS_MIN_CHROMA_IRE);
    const double VDIS_RAMP_RANGE_IRE = 4.0;

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

        // --- IQ-based VDIS gating (aligned space) ---
        double vdisGate = 1.0;
        double dUpDown_ire = 0.0;
        if (haveUp && haveDn) {
            dUpDown_ire = cmag(ZUpRaw - ZDnRaw) * invI;

            if (T.VDIS_HARD_FALLBACK && dUpDown_ire > VDIS_IQ_THRESH_IRE) {
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
            sharedUpGate = std::clamp(reachTapLine->pairU1[x].reachGate, 0.0, 1.0);
            sharedDnGate = std::clamp(reachTapLine->pairD1[x].reachGate, 0.0, 1.0);
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
        const double gate = motionGate * vdisGate;

        const double effMaxDeltaIRE = MAX_DELTA_IRE * gate;

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
            double t = (dUD_ire - VDIS_IQ_THRESH_IRE) / VDIS_RAMP_RANGE_IRE;
            t = std::clamp(t, 0.0, 1.0);
            disGate = 1.0 - t;
        }

        double strengthMix = cohGate * disGate;

        // Make it a bit more selective without hard switching
        strengthMix = strengthMix * strengthMix; // gamma=2

        double localStrength =
            COMB_STRENGTH_LO + (COMB_STRENGTH_HI - COMB_STRENGTH_LO) * strengthMix;

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

    // Preclean demod: Field B precleaned scalar → 4fsc grid IQ.
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

    // Locked1D fallback when preclean is unavailable for a line.
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

    scratch_centerIQ.resize(width);
    scratch_upIQ.resize(width);
    scratch_dnIQ.resize(width);
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
    const bool verticalAllowed = carrierFrameVerticalAllowed(line);

    outFrameIQ.assign(width, std::complex<double>(0.0, 0.0));
    if (width <= 0 || line < first || line >= last) return;
    if ((int)centerIQ.size() < width || (int)upIQ.size() < width || (int)dnIQ.size() < width) return;

    const auto  &T    = configuration.tunables;
    const double invI = invIreScale;
    const double COMB_STRENGTH   = std::max(0.0, T.FRAME_B_COMB_STRENGTH);
    const double MIN_CHROMA_IRE  = T.FRAME_CHROMA_MIN_IRE;
    const double VDIS_THRESH_IRE = std::max(4.0, T.VDIS_MIN_CHROMA_IRE);
    const double VDIS_RAMP_IRE   = 4.0;

    const bool haveUpLine = (verticalAllowed && line - 1 >= first);
    const bool haveDnLine = (verticalAllowed && line + 1 <  last);
    (void)reachTapLine;

    for (int x = 0; x < width; ++x) {
        const std::complex<double> Z0 = centerIQ[x];
        const std::complex<double> ZUp = upIQ[x];
        const std::complex<double> ZDn = dnIQ[x];
        bool haveUp = haveUpLine && (cmag(ZUp) > 1e-9);
        bool haveDn = haveDnLine && (cmag(ZDn) > 1e-9);

        if (cmag(Z0) * invI < MIN_CHROMA_IRE || (!haveUp && !haveDn)) {
            outFrameIQ[x] = Z0;
            continue;
        }

        const std::complex<double> nbrAvg = (haveUp && haveDn) ? 0.5 * (ZUp + ZDn)
                                          : haveUp             ? ZUp
                                                               : ZDn;

        // Motion gate: suppress when neighbors disagree with each other.
        double motionGate = 1.0;
        if (haveUp && haveDn) {
            const double nbrDiffIRE = cmag(ZUp - ZDn) * invI;
            if (T.VDIS_HARD_FALLBACK && nbrDiffIRE > VDIS_THRESH_IRE) {
                outFrameIQ[x] = Z0;
                continue;
            }
            if (nbrDiffIRE > VDIS_THRESH_IRE) {
                const double t = std::clamp(
                    (nbrDiffIRE - VDIS_THRESH_IRE) / VDIS_RAMP_IRE, 0.0, 1.0);
                motionGate = std::max(0.0, 1.0 - T.VDIS_SUPPRESS_FACTOR * t);
            }
        }

        // Interfield difference extraction: delta is the component of center
        // that differs from its interfield neighbors (the per-field error).
        // Remove it from center, scaled by comb strength and motion gate.
        const std::complex<double> delta = Z0 - nbrAvg;
        outFrameIQ[x] = Z0 - delta * (0.5 * COMB_STRENGTH * motionGate);
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

    // 1D sample: locked path reads from locked1DSource_flat (combed carrier);
    // bucket path reads from clpbuffer[0] (blind bandpass).
    const double *lockedRow = frameBuffer.configuration.phaseCompensation
        ? frameBuffer.locked1DSource_line(lineNumber) : nullptr;
    if (lockedRow && (hh - left) >= 0 && (hh - left) < (right - left))
    {
        result.sample = lockedRow[hh - left];
    } else {
        result.sample = frameBuffer.clpbuffer[0].pixel[lineNumber][hh];
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
