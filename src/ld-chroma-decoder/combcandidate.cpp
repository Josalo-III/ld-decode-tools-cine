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

    auto reflectRel = [&](int r)->int {
        if (r < 0) return -r;
        if (r >= width) return (width - 1) - (r - (width - 1));
        return r;
    };

    // Debug-only: measure whether Field A is actually producing a vertical estimate
    // or collapsing into near-zero output (which reads as "1D remains").
    qint64 dbgN = 0;
    qint64 dbgCollapsedN = 0;
    double dbgSumGate = 0.0;
    double dbgSumAbsTc = 0.0;
    double dbgSumAbsC = 0.0;

    // Phase relationship range (like FieldB; in IRE)
    const double kRange = T.FIELD_K_RANGE_IRE * irescale;
    const double invK   = (kRange > 1e-9) ? (1.0 / kRange) : 0.0;

    // Luma-edge exclusion for far reach (prevents reaching across disparate vertical regions)
    for (int h = left; h < right; ++h) {
        const int rel = h - left;

        const int rm1 = reflectRel(rel - 1);
        const int rp1 = reflectRel(rel + 1);

        // Conservative reform: Field A should look like Field B, with an
        // additional cautious ±4 reach. Avoid extra heuristics that create
        // phase-leg / column artifacts.

        const double C    = row0[rel];
        const double Cup2 = rowUp2[rel];
        const double Cdn2 = rowDn2[rel];
        const double Cup4 = rowUp4[rel];
        const double Cdn4 = rowDn4[rel];

        const double C_m1    = row0[rm1];
        const double C_p1    = row0[rp1];
        const double Cup2_m1 = rowUp2[rm1];
        const double Cup2_p1 = rowUp2[rp1];
        const double Cdn2_m1 = rowDn2[rm1];
        const double Cdn2_p1 = rowDn2[rp1];
        const double Cup4_m1 = rowUp4[rm1];
        const double Cup4_p1 = rowUp4[rp1];
        const double Cdn4_m1 = rowDn4[rm1];
        const double Cdn4_p1 = rowDn4[rp1];

        const double symCur  = 0.5 * (std::fabs(C_m1)    + std::fabs(C_p1));
        const double symUp2  = 0.5 * (std::fabs(Cup2_m1) + std::fabs(Cup2_p1));
        const double symDn2  = 0.5 * (std::fabs(Cdn2_m1) + std::fabs(Cdn2_p1));
        const double symUp4  = 0.5 * (std::fabs(Cup4_m1) + std::fabs(Cup4_p1));
        const double symDn4  = 0.5 * (std::fabs(Cdn4_m1) + std::fabs(Cdn4_p1));

        auto kMetric = [&](double cc, double symC, double cn, double symN)->double {
            double k = 0.0;
            k  = std::fabs(std::fabs(cc) - std::fabs(cn));
            k += std::fabs(symC - symN);
            k -= (std::fabs(cc) + std::fabs(cn)) * 0.10;
            if (k < 0.0) k = 0.0;
            return k;
        };

        // Near weights (same as Field B)
        double kp2 = kMetric(C, symCur, Cup2, symUp2);
        double kn2 = kMetric(C, symCur, Cdn2, symDn2);
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
            double dMag  = std::fabs(std::fabs(Cup2) - std::fabs(Cdn2));
            double sumUD = std::fabs(Cup2 + Cdn2);
            if (dMag - std::fabs(sumUD * 0.2) <= 0.0) {
                wUp2 = wDn2 = 1.0;
                sc2 = 1.0;
            } else {
                wUp2 = wDn2 = 0.0;
            }
        }

        // 5-tap contour check:
        // Use the full set { -4, -2, 0, +2, +4 } to decide whether far reach is
        // on-contour. Far does not contribute directly; it only refines the ±2 tap.
        const double FAR_INFLUENCE_BASE = 0.55;

        auto smoothGate = [&](double xIRE, double softIRE, double hardIRE)->double {
            if (xIRE <= softIRE) return 1.0;
            if (xIRE >= hardIRE) return 0.0;
            double t = (xIRE - softIRE) / (hardIRE - softIRE);
            t = std::clamp(t, 0.0, 1.0);
            return 1.0 - t;
        };

        const double aC   = std::fabs(C);
        const double aU2  = std::fabs(Cup2);
        const double aD2  = std::fabs(Cdn2);
        const double aU4  = std::fabs(Cup4);
        const double aD4  = std::fabs(Cdn4);

        // Detect "hard corner" / boundary crossing at the center using a discrete curvature.
        const double curvMidIRE = std::fabs(aU2 - 2.0 * aC + aD2) * invI;
        const double midOk = smoothGate(curvMidIRE, 4.0, 10.0);

        // Side-specific far consistency: far should lie near the line extrapolated from {0,±2}.
        const double u4Pred = 2.0 * aU2 - aC;
        const double d4Pred = 2.0 * aD2 - aC;
        const double uResIRE = std::fabs(aU4 - u4Pred) * invI;
        const double dResIRE = std::fabs(aD4 - d4Pred) * invI;
        const double upSideOk = smoothGate(uResIRE, 4.0, 10.0);
        const double dnSideOk = smoothGate(dResIRE, 4.0, 10.0);

        // Similarity check in the existing k-metric space (near vs far).
        auto farSim = [&](double nearS, double nearSym, double farS, double farSym)->double {
            const double k = kMetric(nearS, nearSym, farS, farSym);
            const double w = (kRange > 1e-9) ? (1.0 - k * invK) : 1.0;
            return std::clamp(w, 0.0, 1.0);
        };
        const double upSim = (wUp2 > 0.0) ? farSim(Cup2, symUp2, Cup4, symUp4) : 0.0;
        const double dnSim = (wDn2 > 0.0) ? farSim(Cdn2, symDn2, Cdn4, symDn4) : 0.0;

        // Convert similarity into a 0..1 factor (soft start at 0.55).
        auto simFactor = [&](double sim)->double {
            const double start = 0.55;
            const double full  = 0.85;
            if (sim <= start) return 0.0;
            if (sim >= full) return 1.0;
            double t = (sim - start) / (full - start);
            return std::clamp(t, 0.0, 1.0);
        };

        const double upInfluence = FAR_INFLUENCE_BASE * midOk * upSideOk * simFactor(upSim);
        const double dnInfluence = FAR_INFLUENCE_BASE * midOk * dnSideOk * simFactor(dnSim);

        // Far reach is evaluated in magnitude space, so apply it in a sign-safe way:
        // refine the magnitude of the ±2 tap, but keep the near tap's sign. If the
        // far tap disagrees in sign (phase), ignore far refinement to avoid
        // injecting checkerboard/alternation into the intrafield estimate.
        auto refineNearWithFar = [&](double nearS, double farS, double influence)->double {
            if (influence <= 0.0) return nearS;
            if (nearS == 0.0) return nearS;
            if ((nearS > 0.0) != (farS > 0.0)) return nearS;
            const double nearMag = std::fabs(nearS);
            const double farMag  = std::fabs(farS);
            const double mag = (nearMag + influence * farMag) / (1.0 + influence);
            return std::copysign(mag, nearS);
        };

        const double Cup2Adj = refineNearWithFar(Cup2, Cup4, upInfluence);
        const double Cdn2Adj = refineNearWithFar(Cdn2, Cdn4, dnInfluence);

        double tc = 0.0;
        if (wUp2 > 0.0 || wDn2 > 0.0) {
            double t2  = ((C - Cup2Adj) * wUp2 * sc2);
            t2        += ((C - Cdn2Adj) * wDn2 * sc2);
            tc        += 0.25 * t2;
        }

        outFieldLine[rel] = tc;

        double gateA = std::max(wUp2, wDn2);
        // Gate reflects near evidence; far only refines the near tap.
        gateA = std::clamp(gateA, 0.0, 1.0);
        if (outGate) outGate[rel] = (float)gateA;

        // Collect debug stats over active region only.
        if (configuration.debugPhaseLegs) {
            ++dbgN;
            dbgSumGate += gateA;
            dbgSumAbsTc += std::fabs(tc) * invI;
            dbgSumAbsC += std::fabs(C) * invI;
            if ((wUp2 + wDn2) < 0.10) ++dbgCollapsedN;
        }
    }

    if (configuration.debugPhaseLegs && dbgN > 0) {
        const double invN = 1.0 / (double)dbgN;
        qInfo().noquote() << QString("FieldAStats line=%1 n=%2 collapsed=%3 gate=%4 absTcIRE=%5 absCIRE=%6")
            .arg(lineNumber)
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
    const double invI   = this->invIreScale;

    auto reflectRel = [&](int r)->int {
        if (r < 0) return -r;
        if (r >= width) return (width - 1) - (r - (width - 1));
        return r;
    };

    // Phase-coherence gate for vertical reaches (locked mode only).
    // Uses the existing phase-corrected 1D IQ produced upstream by buildPhaseCorrected1D
    // (demodTI/demodTQ). This avoids any additional demod and keys directly off the
    // same locked phase model that the rest of the pipeline uses.
    const float *ti0 = configuration.phaseCompensation ? demodTI_line(ln0)   : nullptr;
    const float *tq0 = configuration.phaseCompensation ? demodTQ_line(ln0)   : nullptr;
    const float *tiU = configuration.phaseCompensation ? demodTI_line(lnUp2) : nullptr;
    const float *tqU = configuration.phaseCompensation ? demodTQ_line(lnUp2) : nullptr;
    const float *tiD = configuration.phaseCompensation ? demodTI_line(lnDn2) : nullptr;
    const float *tqD = configuration.phaseCompensation ? demodTQ_line(lnDn2) : nullptr;

    auto phaseCoherenceIQ = [&](double I0, double Q0, double In, double Qn)->double {
        const double m0 = std::hypot(I0, Q0);
        const double mn = std::hypot(In, Qn);
        if (m0 * invI < 2.5 || mn * invI < 2.5) return 1.0; // too weak to trust phase
        const double corr = (I0 * In + Q0 * Qn) / (m0 * mn + 1e-12); // [-1..1]
        if (corr <= 0.0) return 0.0;
        const double t0 = 0.55;
        const double t1 = 0.85;
        double w = (corr - t0) / (t1 - t0);
        return std::clamp(w, 0.0, 1.0);
    };

    for (int h = left; h < right; ++h) {
        const int rel = h - left;
        const int rm1 = reflectRel(rel - 1);
        const int rp1 = reflectRel(rel + 1);

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

        if (configuration.phaseCompensation && ti0 && tq0 && tiU && tqU && tiD && tqD) {
            const double I0 = (double)ti0[rel];
            const double Q0 = (double)tq0[rel];
            const double IU = (double)tiU[rel];
            const double QU = (double)tqU[rel];
            const double ID = (double)tiD[rel];
            const double QD = (double)tqD[rel];
            wUp *= phaseCoherenceIQ(I0, Q0, IU, QU);
            wDn *= phaseCoherenceIQ(I0, Q0, ID, QD);
        }

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

// NOTE: demodSimpleField2DLine was removed. FrameIQ preclean demods directly from
// the simpleField2D ring using the canonical 4fsc basis to avoid shared-buffer
// side effects and to guarantee a single demod contract for the preclean path.

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

static inline double cmag(const std::complex<double> &z) { return std::hypot(z.real(), z.imag()); }
static inline double dotIQ(const std::complex<double> &a, const std::complex<double> &b) { return a.real()*b.real() + a.imag()*b.imag(); }

void Comb::FrameBuffer::computeFrameIQFromPreparedVectors(
    int line,
    const std::vector<std::complex<double>> &centerIQ,
    std::vector<std::complex<double>> upIQ,
    std::vector<std::complex<double>> dnIQ,
    std::vector<std::complex<double>> &outFrameIQ,
    const std::vector<float> *tiOverride,
    const std::vector<float> *tqOverride,
    bool enableLateralRefine)
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
        return demodTI_line(ln);
    };
    auto tqLine = [&](int ln)->const float* {
        if (tqOverride && (int)tqOverride->size() >= (ln + 1) * demodWidth)
            return tqOverride->data() + static_cast<size_t>(ln) * demodWidth;
        return demodTQ_line(ln);
    };

    auto reflectIndex = [&](int x)->int {
        if (x < 0) return -x;
        if (x >= width) return (width - 1) - (x - (width - 1));
        return x;
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

        if (a0_ire <= MIN_CHROMA_IRE) {
            outFrameIQ[x] = Z0;
            continue;
        }

        std::complex<double> ZUpRaw = upIQ[x];
        std::complex<double> ZDnRaw = dnIQ[x];

        if (enableLateralRefine) {
            // Frame A only: lateral pre-clean for the samples we comb with (Up/Dn only).
            auto refineNeighbor = [&](const std::vector<std::complex<double>> &nbr,
                                      const std::complex<double> &Z,
                                      int xi)->std::complex<double>
            {
                const int xm1 = reflectIndex(xi - 1);
                const int xp1 = reflectIndex(xi + 1);
                const std::complex<double> Zm = nbr[xm1];
                const std::complex<double> Zp = nbr[xp1];

                const double c0 = std::fabs(corrSigned(Z, Zm));
                const double c1 = std::fabs(corrSigned(Z, Zp));
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

            ZUpRaw = refineNeighbor(upIQ, ZUpRaw, x);
            ZDnRaw = refineNeighbor(dnIQ, ZDnRaw, x);
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
        const double COMB_STRENGTH_HI = COMB_STRENGTH;
        const double COMB_STRENGTH_LO = 0.8;

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

void Comb::FrameBuffer::computeFrameIQPrecleanLine(
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
    if (line >= demodLines || demodWidth <= 0) return;

    const std::vector<float> *tiOverride = nullptr;
    const std::vector<float> *tqOverride = nullptr;
    auto tiLine = [&](int ln)->const float* { return demodTI_line(ln); };
    auto tqLine = [&](int ln)->const float* { return demodTQ_line(ln); };

    const float *ti0_raw  = tiLine(line);
    const float *tq0_raw  = tqLine(line);
    const float *tiUp_raw = (line - 1 >= first) ? tiLine(line - 1) : nullptr;
    const float *tqUp_raw = (line - 1 >= first) ? tqLine(line - 1) : nullptr;
    const float *tiDn_raw = (line + 1 <  last)  ? tiLine(line + 1) : nullptr;
    const float *tqDn_raw = (line + 1 <  last)  ? tqLine(line + 1) : nullptr;
    if (!ti0_raw || !tq0_raw) return;

    // Preclean demod helper: demod simpleField2D[ln][x] -> canonical 4fsc bucket IQ.
    // This is the *only* demod used for the preclean path (no shared demod buffers).
    auto demodPrecleanAt = [&](int ln, int x, std::complex<double> &Z)->bool {
        if (ln < first || ln >= last) return false;
        const double *row = simpleField2DLinePtr(ln, width);
        if (!row) return false;
        const int h = left + x;
        const int ph = (h & 3);
        const double c = row[x];
        Z = std::complex<double>(c * sin4fsc(ph) * 2.0,
                                 c * cos4fsc(ph) * 2.0);
        return true;
    };

    std::vector<std::complex<double>> centerIQ(width), upIQ(width), dnIQ(width);
    for (int x = 0; x < width; ++x) {
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
    }

    computeFrameIQFromPreparedVectors(line, centerIQ, upIQ, dnIQ, outFrameIQ, tiOverride, tqOverride, true);
}

void Comb::FrameBuffer::computeFrameIQLocked1DLine(
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

    outFrameIQ.assign(width, std::complex<double>(0.0, 0.0));
    if (width <= 0 || line < first || line >= last) return;
    if (line >= demodLines || demodWidth <= 0) return;

    auto tiLine = [&](int ln)->const float* {
        if (tiOverride && (int)tiOverride->size() >= (ln + 1) * demodWidth)
            return tiOverride->data() + static_cast<size_t>(ln) * demodWidth;
        return demodTI_line(ln);
    };
    auto tqLine = [&](int ln)->const float* {
        if (tqOverride && (int)tqOverride->size() >= (ln + 1) * demodWidth)
            return tqOverride->data() + static_cast<size_t>(ln) * demodWidth;
        return demodTQ_line(ln);
    };

    const float *ti0_raw  = tiLine(line);
    const float *tq0_raw  = tqLine(line);
    const float *tiUp_raw = (line - 1 >= first) ? tiLine(line - 1) : nullptr;
    const float *tqUp_raw = (line - 1 >= first) ? tqLine(line - 1) : nullptr;
    const float *tiDn_raw = (line + 1 <  last)  ? tiLine(line + 1) : nullptr;
    const float *tqDn_raw = (line + 1 <  last)  ? tqLine(line + 1) : nullptr;
    if (!ti0_raw || !tq0_raw) return;

    std::vector<std::complex<double>> centerIQ(width), upIQ(width), dnIQ(width);
    for (int x = 0; x < width; ++x) {
        centerIQ[x] = std::complex<double>((double)ti0_raw[x], (double)tq0_raw[x]);
        upIQ[x]     = (tiUp_raw && tqUp_raw) ? std::complex<double>((double)tiUp_raw[x], (double)tqUp_raw[x])
                                             : std::complex<double>(0.0, 0.0);
        dnIQ[x]     = (tiDn_raw && tqDn_raw) ? std::complex<double>((double)tiDn_raw[x], (double)tqDn_raw[x])
                                             : std::complex<double>(0.0, 0.0);
    }

    computeFrameIQFromPreparedVectors(line, centerIQ, upIQ, dnIQ, outFrameIQ, tiOverride, tqOverride, false);
}

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
// ---------------------------------------------------------------------
// getBestY - Dedicated 3D Residual Y Election/Blend
// ---------------------------------------------------------------------
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
