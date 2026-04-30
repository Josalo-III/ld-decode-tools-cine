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
 * Implements Comb::FrameBuffer::getCandidate() and vetComposite1D(),
 * separated from comb.cpp so that candidate selection and vet logic
 * live in a single translation unit.
 ******************************************************************************/

#include "comb.h"
#include "combmath.h"

#include <cmath>
#include <algorithm>
#include <limits>

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


// Field A - we sample 2 and 4 lines above and below, with the 4s asymmetrically
// influencing the 2s,and 2s then influencing the evaluated pixel. Strictly intra-field.
void Comb::FrameBuffer::computeField2DLine(int lineNumber,
                                          double *outFieldLine,
                                          double  *outGate)
{
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;

    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;

    if (width <= 0 || lineNumber < first || lineNumber >= last) {
        if (outFieldLine) std::fill(outFieldLine, outFieldLine + std::max(width, 0), 0.0);
        if (outGate)      std::fill(outGate,      outGate      + std::max(width, 0), 1.0f);
        return;
    }
    if (!outFieldLine) return;

    if (outGate) std::fill(outGate, outGate + width, 1.0f);

    auto clampSameFieldLine = [&](int ln)->int {
        // For intrafield sampling we must stay on the same field parity as lineNumber.
        // Plain clamping can jump to the opposite field at the top/bottom edges.
        const int parity = (lineNumber & 1);
        ln = std::clamp(ln, first, last - 1);
        if ((ln & 1) != parity) {
            // Prefer stepping inward rather than outward.
            if (ln + 1 < last && ((ln + 1) & 1) == parity) ln = ln + 1;
            else if (ln - 1 >= first && ((ln - 1) & 1) == parity) ln = ln - 1;
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
        auto getRow = [&](int ln)->const double* {
            if (ln < 0 || ln >= (int)locked1DSource.size()) return nullptr;
            const auto &row = locked1DSource[ln];
            if ((int)row.size() < width) return nullptr;
            return row.data();
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

    if (!row0 || !rowUp2 || !rowDn2 || !rowUp4 || !rowDn4) {
        std::fill(outFieldLine, outFieldLine + width, 0.0);
        if (outGate) std::fill(outGate, outGate + width, 1.0f);
        return;
    }

    const auto  &T    = configuration.tunables;
    const double invI = this->invIreScale;

    // Phase relationship range (like FieldB; in IRE)
    const double kRange = T.FIELD_K_RANGE_IRE * irescale;
    const double invK   = (kRange > 1e-9) ? (1.0 / kRange) : 0.0;

    // Luma-edge exclusion for far reach (prevents reaching across disparate vertical regions)
    const double EDGE_SOFT_IRE = 6.0;
    const double EDGE_HARD_IRE = 14.0;

    auto edgeGateAt = [&](int rel)->double {
        const int rm1 = (rel > 0) ? (rel - 1) : 0;
        const int rp1 = (rel + 1 < width) ? (rel + 1) : (width - 1);
        const double eIRE = std::fabs(row0[rp1] - row0[rm1]) * invI;

        if (eIRE <= EDGE_SOFT_IRE) return 1.0;
        if (eIRE >= EDGE_HARD_IRE) return 0.0;
        double t = (eIRE - EDGE_SOFT_IRE) / (EDGE_HARD_IRE - EDGE_SOFT_IRE);
        t = std::clamp(t, 0.0, 1.0);
        return 1.0 - t;
    };

    for (int h = left; h < right; ++h) {
        const int rel = h - left;

        const int rm1 = (rel > 0) ? (rel - 1) : 0;
        const int rp1 = (rel + 1 < width) ? (rel + 1) : (width - 1);

        // Center and symmetric lateral context (reduces column bias)
        const double C    = row0[rel];
        const double C_m1 = row0[rm1];
        const double C_p1 = row0[rp1];
        const double symCur = 0.5 * (std::fabs(C_m1) + std::fabs(C_p1));

        // --- Near samples (2) ---
        const double U2    = rowUp2[rel];
        const double D2    = rowDn2[rel];
        const double U2_m1 = rowUp2[rm1];
        const double U2_p1 = rowUp2[rp1];
        const double D2_m1 = rowDn2[rm1];
        const double D2_p1 = rowDn2[rp1];
        const double symU2 = 0.5 * (std::fabs(U2_m1) + std::fabs(U2_p1));
        const double symD2 = 0.5 * (std::fabs(D2_m1) + std::fabs(D2_p1));

        // --- Far samples (4) ---
        const double U4    = rowUp4[rel];
        const double D4    = rowDn4[rel];
        const double U4_m1 = rowUp4[rm1];
        const double U4_p1 = rowUp4[rp1];
        const double D4_m1 = rowDn4[rm1];
        const double D4_p1 = rowDn4[rp1];
        const double symU4 = 0.5 * (std::fabs(U4_m1) + std::fabs(U4_p1));
        const double symD4 = 0.5 * (std::fabs(D4_m1) + std::fabs(D4_p1));

        // ------------------------------------------------------------
        // PASS: compute near weights wUp2/wDn2 based on magnitude-phase agreement
        // Similar to FieldB logic, but symmetric and tuned for A.
        // ------------------------------------------------------------
        auto phaseDiffMetric = [&](double C0, double sym0, double Cn, double symn)->double {
            double k = 0.0;
            k  = std::fabs(std::fabs(C0) - std::fabs(Cn));
            k += std::fabs(sym0 - symn);
            // small bonus for strong signal (helps avoid weak = noisy toggles)
            k -= (std::fabs(C0) + std::fabs(Cn)) * 0.10;
            if (k < 0.0) k = 0.0;
            return k;
        };

        double kp2 = phaseDiffMetric(C, symCur, U2, symU2);
        double kn2 = phaseDiffMetric(C, symCur, D2, symD2);

        double wUp2 = (kRange > 1e-9) ? (1.0 - kp2 * invK) : 1.0;
        double wDn2 = (kRange > 1e-9) ? (1.0 - kn2 * invK) : 1.0;
        wUp2 = std::clamp(wUp2, 0.0, 1.0);
        wDn2 = std::clamp(wDn2, 0.0, 1.0);

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
            // If up/down are similar to each other, allow both (classic fallback)
            double dMag  = std::fabs(std::fabs(U2) - std::fabs(D2));
            double sumUD = std::fabs(U2 + D2);
            if (dMag - std::fabs(sumUD * 0.2) <= 0.0) {
                wUp2 = wDn2 = 1.0;
                sc2 = 1.0;
            } else {
                wUp2 = wDn2 = 0.0;
            }
        }

        // ------------------------------------------------------------
        // Far weights (4), but *ramped* by near confidence and edge gate
        // This avoids far popping that creates patterned alternation.
        // ------------------------------------------------------------
        double kp4 = phaseDiffMetric(C, symCur, U4, symU4);
        double kn4 = phaseDiffMetric(C, symCur, D4, symD4);

        double wUp4 = (kRange > 1e-9) ? (1.0 - kp4 * invK) : 1.0;
        double wDn4 = (kRange > 1e-9) ? (1.0 - kn4 * invK) : 1.0;
        wUp4 = std::clamp(wUp4, 0.0, 1.0);
        wDn4 = std::clamp(wDn4, 0.0, 1.0);

        // Ramp FAR by NEAR (no hard on/off)
        const double nearConfUp = wUp2;
        const double nearConfDn = wDn2;

        // Additional suppression on strong horizontal edges
        const double eGate = edgeGateAt(rel);

        wUp4 *= nearConfUp * eGate;
        wDn4 *= nearConfDn * eGate;

        // Prefer near unless far is clearly better; keep far subtle
        const double FAR_SCALE = 0.65; // far contributes less authority by default
        wUp4 *= FAR_SCALE;
        wDn4 *= FAR_SCALE;

        // ------------------------------------------------------------
        // Combine near and far contributions (still a  comb)
        // ------------------------------------------------------------
        double tc = 0.0;

        // Near comb component
        if (wUp2 > 0.0 || wDn2 > 0.0) {
            double t2  = ((C - U2) * wUp2 * sc2);
            t2        += ((C - D2) * wDn2 * sc2);
            t2        *= 0.25;
            tc        += t2;
        }

        // Far comb component (no separate sc; weights already ramped)
        if (wUp4 > 0.0 || wDn4 > 0.0) {
            const double denom = wUp4 + wDn4;
            double sc4 = 1.0;
            if (denom > 1e-9) {
                sc4 = 2.0 / denom;
                if (sc4 < 1.0) sc4 = 1.0;
            }
            double t4  = ((C - U4) * wUp4 * sc4);
            t4        += ((C - D4) * wDn4 * sc4);
            t4        *= 0.25;
            tc        += t4;
        }

        outFieldLine[rel] = tc;

        // Gate for scorer: how confident is A here?
        // Use near confidence primarily, with far only if its present.
        double gateA = std::max(wUp2, wDn2);
        gateA = std::max(gateA, 0.5 * std::max(wUp4, wDn4)); // far contributes but less
        gateA = std::clamp(gateA, 0.0, 1.0);

        if (outGate) outGate[rel] = (float)gateA;
    }
}


// Field B
// Simplified Field comb as a FrameBuffer member:
// - uses only 2 vertical neighbours
void Comb::FrameBuffer::computeSimpleField2DLine(int lineNumber, double *outFieldLine)
{
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;

    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;

    if (width <= 0 || lineNumber < first || lineNumber >= last) {
        if (outFieldLine) std::fill(outFieldLine, outFieldLine + std::max(width, 0), 0.0);
        return;
    }
    if (!outFieldLine) return;

    auto clampSameFieldLine = [&](int ln)->int {
        const int parity = (lineNumber & 1);
        ln = std::clamp(ln, first, last - 1);
        if ((ln & 1) != parity) {
            if (ln + 1 < last && ((ln + 1) & 1) == parity) ln = ln + 1;
            else if (ln - 1 >= first && ((ln - 1) & 1) == parity) ln = ln - 1;
        }
        return ln;
    };

    const int ln0   = clampSameFieldLine(lineNumber);
    const int lnUp2 = clampSameFieldLine(lineNumber - 2);
    const int lnDn2 = clampSameFieldLine(lineNumber + 2);

    const double *row0   = nullptr;
    const double *rowUp2 = nullptr;
    const double *rowDn2 = nullptr;

    if (configuration.phaseCompensation) {
        auto getRow = [&](int ln)->const double* {
            if (ln < 0 || ln >= (int)locked1DSource.size()) return nullptr;
            const auto &row = locked1DSource[ln];
            if ((int)row.size() < width) return nullptr;
            return row.data();
        };
        row0   = getRow(ln0);
        rowUp2 = getRow(lnUp2);
        rowDn2 = getRow(lnDn2);
    } else {
        row0   = clpbuffer[0].pixel[ln0]   + left;
        rowUp2 = clpbuffer[0].pixel[lnUp2] + left;
        rowDn2 = clpbuffer[0].pixel[lnDn2] + left;
    }

    if (!row0 || !rowUp2 || !rowDn2) {
        std::fill(outFieldLine, outFieldLine + width, 0.0);
        return;
    }

    const auto &T = configuration.tunables;
    const double kRange = T.FIELD_K_RANGE_IRE * irescale;
    const double invK   = (kRange > 1e-9) ? (1.0 / kRange) : 0.0;

    for (int h = left; h < right; ++h) {
        const int rel = h - left;
        const int rm1 = (rel > 0) ? (rel - 1) : 0;
        const int rp1 = (rel + 1 < width) ? (rel + 1) : (width - 1);

        const double C    = row0[rel];
        const double Cup  = rowUp2[rel];
        const double Cdn  = rowDn2[rel];

        const double C_m1   = row0[rm1];
        const double C_p1   = row0[rp1];
        const double Cup_m1 = rowUp2[rm1];
        const double Cup_p1 = rowUp2[rp1];
        const double Cdn_m1 = rowDn2[rm1];
        const double Cdn_p1 = rowDn2[rp1];

        // Symmetric lateral magnitude context (removes column bias)
        const double symCur = 0.5 * (std::fabs(C_m1)   + std::fabs(C_p1));
        const double symUp  = 0.5 * (std::fabs(Cup_m1) + std::fabs(Cup_p1));
        const double symDn  = 0.5 * (std::fabs(Cdn_m1) + std::fabs(Cdn_p1));

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

        double sc = 1.0;

        if ((wUp > 0.0) || (wDn > 0.0)) {
            if (wDn > 3.0 * wUp)      wUp = 0.0;
            else if (wUp > 3.0 * wDn) wDn = 0.0;

            const double denom = wUp + wDn;
            if (denom > 1e-9) {
                sc = 2.0 / denom;
                if (sc < 1.0) sc = 1.0;
            } else {
                wUp = wDn = 0.0;
            }
        } else {
            double dMag  = std::fabs(std::fabs(Cup) - std::fabs(Cdn));
            double sumUD = std::fabs(Cup + Cdn);
            if (dMag - std::fabs(sumUD * 0.2) <= 0.0) {
                wUp = wDn = 1.0;
                sc = 1.0;
            } else {
                wUp = wDn = 0.0;
            }
        }

        double tc = 0.0;
        if (wUp > 0.0 || wDn > 0.0) {
            tc  = ((C - Cup) * wUp * sc);
            tc += ((C - Cdn) * wDn * sc);
            tc *= 0.25;
        } else {
            tc = 0.0;
        }

        outFieldLine[rel] = tc;
    }

    // In locked (phase-compensated) mode, the -damper is not applied;
    // it is intended only for the phase-blind (bucket) path.
    return;
}

// Demodulates the Field B scalar raster (simpleField2D[line]) into the locked
// demodTI/TQ buffers for use by computeFrameIQLine. Field B provides a 2 intra-field
// comb estimate that serves as a cleaner input to the frame-comb demodulation than
// the raw composite, reducing subcarrier leakage into the frame IQ estimate.
void Comb::FrameBuffer::demodSimpleField2DLine(int line)
{
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;

    if (line < first || line >= last) return;
    if (width <= 0) return;
    if (line >= demodLines || demodWidth <= 0) return;

    if ((int)demodBurstCos.size() <= line ||
        (int)demodBurstSin.size() <= line) {
        // No LO for this line
        return;
    }

    // FieldB scalar raster must already be present in the ring for this line.
    const double *fieldLine = simpleField2DLinePtr(line, width);
    if (!fieldLine) return;

    float *ti = demodTI_line(line);
    float *tq = demodTQ_line(line);
    if (!ti || !tq) return;

    const double bcos = (double)demodBurstCos[line];
    const double bsin = (double)demodBurstSin[line];
    double lutTi[4], lutTq[4];
    fusedDemodLUT(bcos, bsin, spLUT_locked, cpLUT_locked, lutTi, lutTq);

    // Ensure locked basis LUT is ready (same lazy-init as splitIQlocked)
    if (!basisLockedInit) {
        double Ce = 1.0, Se = 0.0;
        basisCoeffs(Ce, Se);
        for (int i = 0; i < 4; ++i) {
            double sp, cp;
            shiftedBasis(i, Ce, Se, sp, cp);
            spLUT_locked[i] = sp;
            cpLUT_locked[i] = cp;
        }
        basisLockedInit = true;
    }

    int fieldCountTotal = 0, frameCountTotal = 0;
    for (int rel = 0; rel < width; ++rel) {
        const int h = left + rel;
        const double c = fieldLine[rel];
        const int ph = (h & 3);
        ti[rel] = (float)(c * lutTi[ph]);
        tq[rel] = (float)(c * lutTq[ph]);
    }
}

// VDIS - Vertical Differential Isolation System.
// Reduces artifacts at horizontal boundaries between different regions.
// We detect for strong vertical differentials in both chroma phase (IQ space)
// and scalar magnitude between upper and lower samples in the field.
// If 1 checks fail, 1D only in FVF and 3D. Fields excluded from FVF if 2 fails.
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
            const float *ti0 = demodTI_line(lineNumber);
            const float *tq0 = demodTQ_line(lineNumber);
            const float *tiU = haveUp1 ? demodTI_line(up1) : nullptr;
            const float *tqU = haveUp1 ? demodTQ_line(up1) : nullptr;
            const float *tiD = haveDn1 ? demodTI_line(dn1) : nullptr;
            const float *tqD = haveDn1 ? demodTQ_line(dn1) : nullptr;

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


// Frame comb in IQ space: averages the 1 neighbouring lines (adjacent in the
// interlaced frame, therefore from the opposite field) to produce a frame-comb
// estimate. Operates in demodulated IQ rather than raw composite to allow
// phase-aware alignment and Nyquist/zipper repair. VDIS gating suppresses the
// frame estimate where vertical chroma phase disagreement is detected.
void Comb::FrameBuffer::computeFrameIQLine(
    int line,
    std::vector<std::complex<double>> &outFrameIQ)
{
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;

    outFrameIQ.assign(width, std::complex<double>(0.0, 0.0));

    if (width <= 0 || line < first || line >= last) return;
    if (line >= demodLines || demodWidth <= 0)      return;

    const auto  &T    = configuration.tunables;
    const double invI = this->invIreScale;

    const float *ti0_raw  = demodTI_line(line);
    const float *tq0_raw  = demodTQ_line(line);
    const float *tiUp_raw = (line - 1 >= first) ? demodTI_line(line - 1) : nullptr;
    const float *tqUp_raw = (line - 1 >= first) ? demodTQ_line(line - 1) : nullptr;
    const float *tiDn_raw = (line + 1 <  last)  ? demodTI_line(line + 1) : nullptr;
    const float *tqDn_raw = (line + 1 <  last)  ? demodTQ_line(line + 1) : nullptr;

    if (!ti0_raw || !tq0_raw) return;

    auto cmag = [](const std::complex<double> &z)->double {
        return std::hypot(z.real(), z.imag());
    };
    auto dotIQ = [](const std::complex<double> &a, const std::complex<double> &b)->double {
        return a.real()*b.real() + a.imag()*b.imag();
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

    // ------------------------------------------------------------
    // Preclean demod helper: demod simpleField2D[ln][x] -> IQ using burst + locked basis.
    // Falls back to canonical demodTI/TQ if preclean isn't available.
    // ------------------------------------------------------------
    auto havePrecleanLine = [&](int ln)->bool {
        if (ln < first || ln >= last) return false;
        if (!simpleField2DLinePtr(ln, width)) return false;
        if (ln < 0 || ln >= (int)demodBurstCos.size() || ln >= (int)demodBurstSin.size()) return false;
        return true;
    };

    auto ensureLockedBasis = [&](){
        if (basisLockedInit) return;
        double Ce = 1.0, Se = 0.0;
        basisCoeffs(Ce, Se);
        for (int i = 0; i < 4; ++i) {
            double sp, cp;
            shiftedBasis(i, Ce, Se, sp, cp);
            spLUT_locked[i] = sp;
            cpLUT_locked[i] = cp;
        }
        basisLockedInit = true;
    };

    auto demodPrecleanAt = [&](int ln, int x, std::complex<double> &Z)->bool {
        if (!havePrecleanLine(ln)) return false;

        ensureLockedBasis();

        const double bcos = (double)demodBurstCos[ln];
        const double bsin = (double)demodBurstSin[ln];

        const int h   = left + x;
        const int idx = (h & 3);
        const double sp = spLUT_locked[idx];
        const double cp = cpLUT_locked[idx];

        const double *row = simpleField2DLinePtr(ln, width);
        if (!row) return false;
        const double c = row[x];

        const double lsin = c * sp * 2.0;
        const double lcos = c * cp * 2.0;
        const double Ii   = (lsin * bcos - lcos * bsin);
        const double Qi   = (lsin * bsin + lcos * bcos);

        Z = std::complex<double>(Ii, Qi);
        return true;
    };

    auto applyMat = [](const std::complex<double> &z, const double M[2][2])->std::complex<double> {
        const double I = z.real(), Q = z.imag();
        return std::complex<double>(M[0][0]*I + M[0][1]*Q,
                                    M[1][0]*I + M[1][1]*Q);
    };

    const double COMB_STRENGTH  = std::max(1.0, T.FRAME_COMB_STRENGTH);
    const double MAX_DELTA_IRE  = T.FRAME_IQ_RAW_MAX_DELTA_IRE;
    const double MIN_CHROMA_IRE = T.FRAME_CHROMA_MIN_IRE;

    const double VDIS_IQ_THRESH_IRE  = std::max(4.0, T.VDIS_MIN_CHROMA_IRE);
    const double VDIS_RAMP_RANGE_IRE = 4.0;

    // ------------------------------------------------------------
    // Build IQ vectors for center and 1 neighbors
    // ------------------------------------------------------------
    const bool usePreclean = havePrecleanLine(line);

    std::vector<std::complex<double>> centerIQ(width);
    std::vector<std::complex<double>> upIQ(width);
    std::vector<std::complex<double>> dnIQ(width);

    for (int x = 0; x < width; ++x) {
        if (usePreclean) {
            std::complex<double> z;
            if (demodPrecleanAt(line, x, z)) centerIQ[x] = z;
            else centerIQ[x] = std::complex<double>((double)ti0_raw[x], (double)tq0_raw[x]);

            if (!demodPrecleanAt(line - 1, x, z)) {
                if (tiUp_raw && tqUp_raw) z = std::complex<double>((double)tiUp_raw[x], (double)tqUp_raw[x]);
                else z = std::complex<double>(0.0, 0.0);
            }
            upIQ[x] = z;

            if (!demodPrecleanAt(line + 1, x, z)) {
                if (tiDn_raw && tqDn_raw) z = std::complex<double>((double)tiDn_raw[x], (double)tqDn_raw[x]);
                else z = std::complex<double>(0.0, 0.0);
            }
            dnIQ[x] = z;
        } else {
            centerIQ[x] = std::complex<double>((double)ti0_raw[x], (double)tq0_raw[x]);
            upIQ[x]     = (tiUp_raw && tqUp_raw) ? std::complex<double>((double)tiUp_raw[x], (double)tqUp_raw[x])
                                                 : std::complex<double>(0.0, 0.0);
            dnIQ[x]     = (tiDn_raw && tqDn_raw) ? std::complex<double>((double)tiDn_raw[x], (double)tqDn_raw[x])
                                                 : std::complex<double>(0.0, 0.0);
        }
    }

    // ------------------------------------------------------------
    // 4fsc-referenced per-line trim (rotation only)
    // ------------------------------------------------------------
    auto applyLineTrimRm = [&](int ln, std::vector<std::complex<double>> &v)
    {
        if (!T.Y_LINE_AFFINE_TRIM_ENABLE) return;

        const int actualHeight = (int)(rawbuffer.size() / (size_t)videoParameters.fieldWidth);
        if (ln < first || ln >= last || ln < 0 || ln >= actualHeight) return;

        const float *tiRow = demodTI_line(ln);
        const float *tqRow = demodTQ_line(ln);
        if (!tiRow || !tqRow) return;

        if (ln >= (int)demodBurstCos.size() || ln >= (int)demodBurstSin.size()) return;

        if (!basisLockedInit) {
            double Ce = 1.0, Se = 0.0;
            basisCoeffs(Ce, Se);
            for (int i = 0; i < 4; ++i) {
                double sp, cp;
                shiftedBasis(i, Ce, Se, sp, cp);
                spLUT_locked[i] = sp;
                cpLUT_locked[i] = cp;
            }
            basisLockedInit = true;
        }

        const quint16 *rawLine = rawbuffer.data() + (size_t)ln * (size_t)videoParameters.fieldWidth;

        const double bcos = (double)demodBurstCos[ln];
        const double bsin = (double)demodBurstSin[ln];

        double STT[2][2] = {{0,0},{0,0}};
        double SRT[2][2] = {{0,0},{0,0}};

        double dc = (double)rawLine[left];
        constexpr double DC_ALPHA = 1.0 / 64.0;

        const double MIN_FIT_IRE = std::max(2.0, 0.5 * T.FRAME_CHROMA_MIN_IRE);

        int n = 0;
        for (int x = 0, h = left; h < right; ++h, ++x) {
            const double vraw_s = (double)rawLine[h];
            dc += DC_ALPHA * (vraw_s - dc);
            const double vraw = vraw_s - dc;

            const int idx = (h & 3);
            const double sp = spLUT_locked[idx];
            const double cp = cpLUT_locked[idx];

            const double lsin_r = vraw * sp * 2.0;
            const double lcos_r = vraw * cp * 2.0;
            const double ri     = (lsin_r * bcos - lcos_r * bsin);
            const double rq     = (lsin_r * bsin + lcos_r * bcos);

            const double ti = (double)tiRow[x];
            const double tq = (double)tqRow[x];

            if (std::hypot(ti, tq) * invI < MIN_FIT_IRE) continue;

            STT[0][0] += ti*ti; STT[0][1] += ti*tq;
            STT[1][0] += ti*tq; STT[1][1] += tq*tq;

            SRT[0][0] += ri*ti; SRT[0][1] += ri*tq;
            SRT[1][0] += rq*ti; SRT[1][1] += rq*tq;

            ++n;
        }

        if (n < 64) return;

        double STTinv[2][2];
        if (!mat2_inv(STT, STTinv)) return;

        double A[2][2];
        mat2_mul(SRT, STTinv, A);

        double Rm[2][2] = {{1,0},{0,1}}, U[2][2];
        polar_decompose_2x2(A, Rm, U);

        const double pMax = T.Y_LINE_MAX_PHASE_DEG * M_PI / 180.0;
        clamp_rotation_gain_shear(Rm, U, pMax,
                                  T.Y_LINE_ALLOW_GAIN_ON_IQ,
                                  T.Y_LINE_GAIN_MIN, T.Y_LINE_GAIN_MAX,
                                  T.Y_LINE_MAX_SHEAR);

        for (int x = 0; x < width; ++x) {
            const double I = v[x].real();
            const double Q = v[x].imag();
            v[x] = std::complex<double>(Rm[0][0]*I + Rm[0][1]*Q,
                                        Rm[1][0]*I + Rm[1][1]*Q);
        }
    };

    applyLineTrimRm(line,   centerIQ);
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
            const std::complex<double> Z0 = centerIQ[x];
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
    // Nyquist/zipper repair (local 1 search only when alternation is detected)
    // ------------------------------------------------------------
    auto sgnCorr = [&](const std::complex<double> &a, const std::complex<double> &b)->int {
        const double ma = cmag(a);
        const double mb = cmag(b);
        if (ma <= 1e-12 || mb <= 1e-12) return 0;
        const double d = dotIQ(a,b);
        return (d >= 0.0) ? 1 : -1;
    };

    std::vector<int> sgnUp(width, 0), sgnDn(width, 0);
    for (int x = 0; x < width; ++x) {
        if (cmag(centerIQ[x]) * invI > MIN_CHROMA_IRE) {
            sgnUp[x] = sgnCorr(centerIQ[x], upIQ[x]);
            sgnDn[x] = sgnCorr(centerIQ[x], dnIQ[x]);
        }
    }

    auto isNyq5 = [&](const std::vector<int> &sgn, int x)->bool {
        if (x < 2 || x >= width - 2) return false;
        const int s0 = sgn[x];
        if (s0 == 0) return false;
        return (sgn[x-1] == -s0) && (sgn[x-2] == s0) && (sgn[x+1] == -s0) && (sgn[x+2] == s0);
    };

    const int NYQ_RUN_MIN = 3;

    auto isNyqRun = [&](const std::vector<int> &sgn, int x)->bool {
        const int half = NYQ_RUN_MIN / 2;
        if (x < half || x >= width - half) return false;
        if (sgn[x] == 0) return false;

        int leftRun = 0;
        for (int i = x - 1; i >= 0; --i) {
            if (sgn[i] == 0) break;
            if (sgn[i] != -sgn[i + 1]) break;
            ++leftRun;
        }

        int rightRun = 0;
        for (int i = x + 1; i < width; ++i) {
            if (sgn[i] == 0) break;
            if (sgn[i] != -sgn[i - 1]) break;
            ++rightRun;
        }

        const int runLen = 1 + leftRun + rightRun;
        return (runLen >= NYQ_RUN_MIN) && (leftRun >= half) && (rightRun >= half);
    };

    const double NYQ_CONF_GOOD     = 0.67;
    const double NYQ_SHIFT_PENALTY = 0.25;
    const double NYQ_MAX_DELTA_IRE = 6.0;

    auto pickBestFrom3 = [&](const std::vector<std::complex<double>> &nbr, int x,
                             std::complex<double> &bestZ, double &bestCorrAbs)->bool
    {
        bestCorrAbs = 0.0;
        double bestScore = -1e9;
        bool any = false;
        const std::complex<double> &Z0 = centerIQ[x];
        const double a0 = cmag(Z0);
        if (a0 <= 1e-12) return false;

        for (int dx = -1; dx <= 1; ++dx) {
            const int xx = x + dx;
            if (xx < 0 || xx >= width) continue;
            const std::complex<double> Zn = nbr[xx];
            const double an = cmag(Zn);
            if (an <= 1e-12) continue;

            const double cabs = std::fabs(dotIQ(Z0, Zn)) / (a0*an + 1e-12);
            const double score = cabs - NYQ_SHIFT_PENALTY * (double)std::abs(dx);
            if (!any || score > bestScore) {
                any = true;
                bestScore = score;
                bestCorrAbs = cabs;
                bestZ = Zn;
            }
        }
        return any;
    };

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

        // Local alternation repair: tiny 1 search as a replacement sample (local only).
        if (isNyq5(sgnUp, x) || isNyqRun(sgnUp, x)) {
            std::complex<double> bestZ;
            double bestC = 0.0;
            if (pickBestFrom3(upIQ, x, bestZ, bestC) && bestC >= NYQ_CONF_GOOD) {
                const double dCenterIRE = cmag(bestZ - Z0) * invI;
                const double dOrigIRE   = cmag(bestZ - upIQ[x]) * invI;
                if (dCenterIRE <= NYQ_MAX_DELTA_IRE && dOrigIRE <= NYQ_MAX_DELTA_IRE) {
                    ZUpRaw = bestZ;
                }
            }
        }
        if (isNyq5(sgnDn, x) || isNyqRun(sgnDn, x)) {
            std::complex<double> bestZ;
            double bestC = 0.0;
            if (pickBestFrom3(dnIQ, x, bestZ, bestC) && bestC >= NYQ_CONF_GOOD) {
                const double dCenterIRE = cmag(bestZ - Z0) * invI;
                const double dOrigIRE   = cmag(bestZ - dnIQ[x]) * invI;
                if (dCenterIRE <= NYQ_MAX_DELTA_IRE && dOrigIRE <= NYQ_MAX_DELTA_IRE) {
                    ZDnRaw = bestZ;
                }
            }
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
            const double w = softAlignWeight(Z0, ZUpRaw) * boundaryWeightScale;
            if (w > 0.0) {
                Zsum += softAlignContrib(Z0, ZUpRaw) * boundaryWeightScale;
                wsum += w;
            }
        }
        if (useDn) {
            const double w = softAlignWeight(Z0, ZDnRaw) * boundaryWeightScale;
            if (w > 0.0) {
                Zsum += softAlignContrib(Z0, ZDnRaw) * boundaryWeightScale;
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
        const double COMB_STRENGTH_HI = COMB_STRENGTH;  // your existing max (e.g. 2.0)
        const double COMB_STRENGTH_LO = 0.75;            // new floor per your tests

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



// Note: fvf_is_tri_safe and getNotchLumaEven2* live in comb.h as inline helpers.

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
    const qint32 wantPhase = (2 + (getLinePhase(refLineNumber) ? 2 : 0) + refH) & 3;
    const qint32 havePhase = ((frameBuffer.getLinePhase(lineNumber) ? 2 : 0) + h) & 3;
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
    const bool haveUp = (refLineNumber - 1 >= firstLine) && (lineNumber - 1 >= firstLine);
    const bool haveDn = (refLineNumber + 1 < lastLine)   && (lineNumber + 1 < lastLine);

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
    int fRef = 1, fCand = 1;
    if (!lineFlip.empty()) {
        if (refLineNumber < (int)lineFlip.size()) fRef = lineFlip[refLineNumber];
        if (lineNumber < (int)lineFlip.size())    fCand = lineFlip[lineNumber];
    }

    double iqPen = (std::fabs(fRef * refClpC[r0] - fCand * candClpC[c0]) * 0.5 +
                    std::fabs(fRef * refClpC[r1] - fCand * candClpC[c1]) * 1.0 +
                    std::fabs(fRef * refClpC[r2] - fCand * candClpC[c2]) * 0.5) / 2.0;
    iqPen = (iqPen * invIreScale) * 0.28 * configuration.chromaWeight;

    double penalty = yPen + iqPen + adjustPenalty;

    if (penalty > configuration.candidatePenaltyHardMax) penalty = configuration.candidatePenaltyHardMax;
    result.penalty = penalty;
    return result;
}
// -------------------------------------------------------------------------
// Accept by default; veto only when local 2x2 LS alignment is poor.
// Criteria: large phase, low normalized correlation (rho), or excessive shear.
// No anti-column heuristics.
// -------------------------------------------------------------------------
Comb::FrameBuffer::Vet1DResult Comb::FrameBuffer::vetComposite1D(qint32 line, qint32 h, bool requireVerticalConfirm) const
{
    Vet1DResult R;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;

    if (line < firstLine || line >= lastLine || h < left || h >= right) return R;

    const int xi = h - left;
    const int srcBuf = std::clamp(static_cast<int>(configuration.dimensions) - 1, 0, 2);
    const double *clpLine = clpbuffer[srcBuf].pixel[line];
    const quint16 *rawRow = rawbuffer.data() + line * videoParameters.fieldWidth;

    if (xi >= 0 && xi < demodWidth && !scratch_comp_res.empty())
        R.composite_bandpass = scratch_comp_res[xi];
    else
        R.composite_bandpass = (double)rawRow[h] - clpLine[h];

    // Default: accept unless alignment test fails
    R.accept = true;
    R.confidence = 1.0;

    if (line >= demodLines || demodWidth <= 0) return R;

    const float* tiRow  = demodTI_line(line);
    const float* tqRow  = demodTQ_line(line);
    const float* triRow = demodTRI_line(line);
    const float* trqRow = demodTRQ_line(line);
    if (!tiRow || !tqRow || !triRow || !trqRow) return R;

    const auto &T = configuration.tunables;
    const int WIN  = std::max(4, (T.VET_ALIGN_WIN_SAMPLES / 4) * 4);
    const int HALF = WIN / 2;

    int a = xi - HALF, b = xi + HALF - 1;
    if (a < 0) { b += -a; a = 0; }
    if (b >= demodWidth) { int over = b - (demodWidth - 1); b -= over; a -= over; if (a < 0) a = 0; }

    // Accumulate Σ T T^T and Σ R T^T
    double STT[2][2] = {{0,0},{0,0}};
    double SRT[2][2] = {{0,0},{0,0}};
    for (int x = a; x <= b; ++x) {
        const double ti = (double)tiRow[x];
        const double tq = (double)tqRow[x];
        const double ri = (double)triRow[x];
        const double rq = (double)trqRow[x];
        STT[0][0] += ti*ti; STT[0][1] += ti*tq;
        STT[1][0] += ti*tq; STT[1][1] += tq*tq;
        SRT[0][0] += ri*ti; SRT[0][1] += ri*tq;
        SRT[1][0] += rq*ti; SRT[1][1] += rq*tq;
    }

    // Invertibility guard
    double STTinv[2][2];
    if (!mat2_inv(STT, STTinv)) return R;

    // A = SRT * inv(STT); polar decomposition
    double tmp[2][2], A[2][2], Rm[2][2], U[2][2];
    mat2_mul(SRT, STTinv, tmp);
    A[0][0] = tmp[0][0]; A[0][1] = tmp[0][1];
    A[1][0] = tmp[1][0]; A[1][1] = tmp[1][1];
    polar_decompose_2x2(A, Rm, U);

    // Extract metrics
    const double phase = std::atan2(Rm[1][0], Rm[0][0]); // radians
    double l1,l2,V_[2][2]; eig2_sym(U,l1,l2,V_);
    const double s1 = std::max(0.0, l1), s2 = std::max(0.0, l2);
    const double g  = 0.5*(s1+s2);
    const double shear = (g>1e-12)? std::fabs(s1-s2)/g : 0.0;

    // Normalized correlation rho (scalar proxy): ||Σ conj(T)·R|| / (Σ|T|^2)
    // Equivalent to Frobenius alignment proxy: trace(SRT SRT^T)^{1/2} / trace(STT) (rough proxy)
    const double srt00 = SRT[0][0], srt01 = SRT[0][1], srt10 = SRT[1][0], srt11 = SRT[1][1];
    const double num = std::sqrt(srt00*srt00 + srt01*srt01 + srt10*srt10 + srt11*srt11);
    const double den = std::max(1e-9, STT[0][0] + STT[1][1]);
    const double rho = num / den;

    // Gate
    const double pMax = T.VET_ALIGN_PHASE_MAX_DEG * M_PI / 180.0;
    if (std::fabs(phase) > pMax || rho < T.VET_ALIGN_MIN_RHO || shear > T.VET_ALIGN_MAX_SHEAR) {
        R.accept = false;
    }

    // Confidence combines rho, phase tightness, and shear tightness
    double c_phase = 1.0 - std::min(1.0, std::fabs(phase)/ (pMax + 1e-12));
    double c_shear = 1.0 - std::min(1.0, shear / (T.VET_ALIGN_MAX_SHEAR + 1e-12));
    double c = 0.5*std::max(0.0, std::min(1.0, rho)) + 0.25*c_phase + 0.25*c_shear;
    if (c < 0.0) c = 0.0; else if (c > 1.0) c = 1.0;
    R.confidence = c;

    // Optional vertical confirmation
    if (requireVerticalConfirm && R.accept) {
        int agrees = 0;
        auto checkLine = [&](int ln){
            if (ln < firstLine || ln >= lastLine) return;
            const float* ti2 = demodTI_line(ln);
            const float* tq2 = demodTQ_line(ln);
            const float* ri2 = demodTRI_line(ln);
            const float* rq2 = demodTRQ_line(ln);
            if (!ti2 || !tq2 || !ri2 || !rq2) return;
            double STT2[2][2]={{0,0},{0,0}}, SRT2[2][2]={{0,0},{0,0}};
            for (int x = a; x <= b; ++x) {
                const double ti = (double)ti2[x], tq = (double)tq2[x];
                const double rI = (double)ri2[x], rQ = (double)rq2[x];
                STT2[0][0]+=ti*ti; STT2[0][1]+=ti*tq; STT2[1][0]+=ti*tq; STT2[1][1]+=tq*tq;
                SRT2[0][0]+=rI*ti; SRT2[0][1]+=rI*tq; SRT2[1][0]+=rQ*ti; SRT2[1][1]+=rQ*tq;
            }
            double inv2[2][2];
            if (!mat2_inv(STT2, inv2)) return;
            double tmp2[2][2], A2[2][2], R2[2][2], U2[2][2];
            mat2_mul(SRT2, inv2, tmp2);
            A2[0][0]=tmp2[0][0]; A2[0][1]=tmp2[0][1];
            A2[1][0]=tmp2[1][0]; A2[1][1]=tmp2[1][1];
            polar_decompose_2x2(A2, R2, U2);
            double ph = std::atan2(R2[1][0], R2[0][0]);
            double l1_,l2_,Vt[2][2]; eig2_sym(U2,l1_,l2_,Vt);
            double s1_ = std::max(0.0, l1_), s2_ = std::max(0.0, l2_);
            double g_ = 0.5*(s1_+s2_);
            double sh = (g_>1e-12)? std::fabs(s1_-s2_)/g_ : 0.0;
            const double num2 = std::sqrt(SRT2[0][0]*SRT2[0][0] + SRT2[0][1]*SRT2[0][1] +
                                          SRT2[1][0]*SRT2[1][0] + SRT2[1][1]*SRT2[1][1]);
            const double den2 = std::max(1e-9, STT2[0][0] + STT2[1][1]);
            double rho2 = num2 / den2;
            if (std::fabs(ph) <= pMax && rho2 >= T.VET_ALIGN_MIN_RHO && sh <= T.VET_ALIGN_MAX_SHEAR) ++agrees;
        };
        checkLine(line - 2);
        checkLine(line + 2);
        R.verticalAgree = agrees;
        if (agrees == 0) { R.accept = false; R.confidence *= 0.5; }
    }

    // Legacy diagnostic fields (no L/R election now)
    R.leftScore  = std::numeric_limits<double>::infinity();
    R.rightScore = std::numeric_limits<double>::infinity();
    R.bestIndex  = -1;
    R.bestScore  = R.accept ? 0.0 : 1.0;
    R.adjNeighborCount   = 0;
    R.adjNeighborSupport = 0.0;

    return R;
}

// ---------------------------------------------------------------------
// getBestY - Dedicated 3D Residual Y Election/Blend
// ---------------------------------------------------------------------
double Comb::FrameBuffer::getBestY(qint32 line, qint32 h, double currentY2D,
                                   const FrameBuffer &prev, const FrameBuffer &next) const
{
    const auto &T = configuration.tunables;
    const int fw  = videoParameters.fieldWidth;
    const int left = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const double invI = this->invIreScale;

    // Helper to extract Y from a framebuffer
    auto getY = [&](const FrameBuffer& fb, int ln, int x) -> double {
        if (ln < videoParameters.firstActiveFrameLine || ln >= videoParameters.lastActiveFrameLine) return 0.0;
        double raw = (double)fb.rawbuffer.data()[ln * fw + x];
        double clp = fb.clpbuffer[1].pixel[ln][x];
        return raw - clp;
    };

    // Cheap "chroma-likeness" metric on Y: local 4fSC energy proxy.
    // Align to a 4-sample group and compute I/Q-like magnitude:
    //   I ~= y1 - y3, Q ~= y2 - y0 (bucket-demod equivalent; magnitude is phase-agnostic).
    auto chromaLikeMagIRE = [&](const FrameBuffer& fb, int ln, int x) -> double {
        if (ln < videoParameters.firstActiveFrameLine || ln >= videoParameters.lastActiveFrameLine) return 0.0;
        if (x < left || x >= right) return 0.0;
        int p = x - ((x - left) & 3);
        if (p < left) p = left;
        if (p > right - 4) p = right - 4;
        if (p < left) return 0.0;
        const double y0 = getY(fb, ln, p + 0);
        const double y1 = getY(fb, ln, p + 1);
        const double y2 = getY(fb, ln, p + 2);
        const double y3 = getY(fb, ln, p + 3);
        const double i = (y1 - y3);
        const double q = (y2 - y0);
        return std::hypot(i, q) * invI;
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

    // Pivot: protect luma against chroma contamination by demoting candidates
    // whose Y contains strong 4fSC-like structure.
    if (T.VET_Y_CHROMA_LIKE_WEIGHT > 0.0) {
        const double cPrev = chromaLikeMagIRE(prev, line, h);
        const double cNext = chromaLikeMagIRE(next, line, h);
        // Scale to sample space so it composes with the existing penalties.
        penPrev += (cPrev * irescale) * T.VET_Y_CHROMA_LIKE_WEIGHT;
        penNext += (cNext * irescale) * T.VET_Y_CHROMA_LIKE_WEIGHT;
    }

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
