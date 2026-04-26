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
// vetComposite1D implementation moved here from comb.cpp
//  - Early accept with a local complex LS alignment gate.
//  - Default is to accept (pure residual everywhere).
//  - Veto (accept=false) only when local phase is misaligned or correlation is weak.
//  - No anti-columnar or edge heuristics here.
//  - Uses pre-FIR demod arrays (TI/TQ) and raw demod (TRI/TRQ).
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