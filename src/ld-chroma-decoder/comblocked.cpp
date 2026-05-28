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
#include <mutex>

namespace {

enum class YSourceNativeSpace {
    RawComposite,
    RawCompositeHigh,
    CompositeLuma,
    LockedIQ,
    BandpassComb
};

enum class YSourceHomeOrientation {
    None,
    GrammarComposite,
    BurstLockedIQ
};

struct YSourceView {
    const char *name = "";
    YSourceNativeSpace nativeSpace = YSourceNativeSpace::RawComposite;
    YSourceHomeOrientation homeOrientation = YSourceHomeOrientation::None;
    double *samples = nullptr;
};

inline double sourceToWorkingSample(const YSourceView &source, int x)
{
    if (!source.samples) return 0.0;

    // produceY composes in a common 4fsc working space. The intake contract
    // tracks each source's home orientation into that table explicitly so we
    // can add real transforms later without rediscovering source identity.
    switch (source.homeOrientation) {
    case YSourceHomeOrientation::None:
    case YSourceHomeOrientation::GrammarComposite:
    case YSourceHomeOrientation::BurstLockedIQ:
        return source.samples[x];
    }

    return source.samples[x];
}

} // namespace

// Locked-path pre-processing: burst detection, raw composite demodulation into
// TRI/TRQ, and a per-line affine solve stored in carrierGrammar.
//
// Locked-path pre-processing.  buildPhaseCorrected1D now sources from
// combedCarrier (the LS carrier model after line-to-line cancellation),
// not clpbuffer[0].  split1D is skipped entirely for the locked path.
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

        double lutTi[4], lutTq[4];
        fusedDemodLUT(bc2, bs2, spLUT_locked, cpLUT_locked, lutTi, lutTq);
        for (int i = 0; i < 4; ++i) {
            grammar.demodLUTTi[i] = (float)lutTi[i];
            grammar.demodLUTTq[i] = (float)lutTq[i];
        }
    }
    // cache fsc-cancelled luma and varietals
    if (!lockedLumaBaseY4_flat.empty() &&
        !lockedLumaSmooth_flat.empty() &&
        demodWidth == width)
    {
        for (int line = firstLine; line < lastLine; ++line) {
            const quint16 *rawLine = rawbuffer.data() + line * fullWidth;
            buildCompositeLumaDecompositionLine(rawLine, left, width,
                                                lockedLumaBaseY4_line(line),
                                                nullptr,
                                                lockedLumaSmooth_line(line));
        }
        lockedLumaCacheValid = true;
    }
    // --- Pass 2: raw composite demod -> TRI/TRQ ---
    // Pre-demod the full line so the windowed fit in Pass 3 can read
    // neighbour samples without re-tracking dc or re-demodding from raw.
    {
        const size_t triNeed = static_cast<size_t>(requiredLines) * static_cast<size_t>(width);
        if (demodTRI_flat.size() < triNeed) {
            demodTRI_flat.assign(triNeed, 0.0f);
            demodTRQ_flat.assign(triNeed, 0.0f);
        }

        for (int line = firstLine; line < lastLine; ++line) {
            const quint16 *rawLine = rawbuffer.data() + line * fullWidth;
            float *triRow          = demodTRI_line(line);
            float *trqRow          = demodTRQ_line(line);
            const CombCarrierGrammar *grammar = carrierGrammarLine(line);
            double lutTi[4], lutTq[4];
            if (grammar && grammar->grammarLocked) {
                for (int i = 0; i < 4; ++i) {
                    lutTi[i] = (double)grammar->demodLUTTi[i];
                    lutTq[i] = (double)grammar->demodLUTTq[i];
                }
            } else {
                fusedDemodLUT(1.0, 0.0, spLUT_locked, cpLUT_locked, lutTi, lutTq);
            }

            double dc = (double)rawLine[left];
            constexpr double DC_ALPHA = 1.0 / 64.0;

            for (int xi = 0; xi < width; ++xi) {
                const int h      = left + xi;
                dc += DC_ALPHA * ((double)rawLine[h] - dc);
                const double vraw = (double)rawLine[h] - dc;
                const int ph = carrierSampleClass(line, h);
                triRow[xi] = (float)(vraw * lutTi[ph]);
                trqRow[xi] = (float)(vraw * lutTq[ph]);
            }
        }
    }

    // --- Pass 3: sinusoidal fit + affine solve -> carrierGrammar.affine ---
    // Reads TRI/TRQ from Pass 2. For each sample, estimates local chroma amplitude
    // from a windowed mean of TRI/TRQ magnitudes, computes a fitted IQ that prefers
    // the window-coherent phase direction when available, and uses a soft quality
    // weight from the residual ratio for the affine solve. This keeps steep saturated
    // regions contributing reduced support instead of dropping out entirely.
    {
        const int WIN  = std::max(4, (T.SINFIT_WIN_SAMPLES / 4) * 4);
        const int HALF = WIN / 2;
        const bool writeAffine = configuration.residualVideo && T.Y_LINE_AFFINE_TRIM_ENABLE;
        constexpr double PHASE_ERROR_CAP = M_PI / 8.0;

        for (int line = firstLine; line < lastLine; ++line) {
            const quint16 *rawLine = rawbuffer.data() + line * fullWidth;
            CombCarrierGrammar *grammar = carrierGrammarLine(line);
            if (!grammar || !grammar->grammarLocked) {
                if (grammar) {
                    grammar->phaseError = 0.0;
                    grammar->affine.valid = false;
                }
                continue;
            }
            const double bcos      = grammar->burstCos;
            const double bsin      = grammar->burstSin;
            const float *triRow    = demodTRI_line(line);
            const float *trqRow    = demodTRQ_line(line);
            double STT[2][2] = {{0,0},{0,0}};
            double SRT[2][2] = {{0,0},{0,0}};

            if ((int)scratch_sinfit_mag.size() < width) scratch_sinfit_mag.resize(width, 0.0);
            if ((int)scratch_sinfit_resmag.size() < width) scratch_sinfit_resmag.resize(width, 0.0);
            double *magRow = scratch_sinfit_mag.data();
            double *resRow = scratch_sinfit_resmag.data();

            // Precompute per-sample magnitudes and residual magnitudes.
            for (int k = 0; k < width; ++k) {
                const int hk = left + k;
                const double rik = (double)triRow[k];
                const double rqk = (double)trqRow[k];
                const double mag_k = std::hypot(rik, rqk);
                magRow[k] = mag_k;
                if (mag_k > 1e-9) {
                    // TRI/TRQ is BurstLockedSigned: the burst-locked demod
                    // already captured the carrier polarity.  Remod without
                    // lineScale to avoid double-applying the sign.
                    const double fitted_k = remodLockedToShiftedComposite(
                        rik, rqk, hk, bcos, bsin, spLUT_locked, cpLUT_locked);
                    const double corr_k   = (double)rawLine[hk] - fitted_k;
                    double rsk = 0.0, rck = 0.0;
                    demod4fscFromComposite(corr_k, hk, rsk, rck);
                    // Residual magnitude is frame-invariant under the burst rotation,
                    // so keep it in common 4fsc rather than rotating and then taking hypot.
                    resRow[k] = std::hypot(rsk, rck);
                } else {
                    resRow[k] = 0.0;
                }
            }

            // Sliding window sums for amp/res. The window shifts to stay inside bounds.
            const int winN = (width <= WIN) ? width : WIN;
            int a = 0;
            int b = winN - 1;
            double sumAmp = 0.0, sumRes = 0.0, sumI = 0.0, sumQ = 0.0;
            for (int k = a; k <= b; ++k) {
                sumAmp += magRow[k];
                sumRes += resRow[k];
                sumI += (double)triRow[k];
                sumQ += (double)trqRow[k];
            }

            for (int xi = 0; xi < width; ++xi) {
                const int h  = left + xi;
                const double ri = (double)triRow[xi];
                const double rq = (double)trqRow[xi];

                // Windowed amplitude and residual from TRI/TRQ neighbours.
                // Window shifts (not shrinks) near edges to keep a stable support.
                if (width > WIN) {
                    int aWant = xi - HALF;
                    int bWant = xi + HALF - 1;
                    if (aWant < 0) {
                        bWant += -aWant;
                        aWant = 0;
                    }
                    if (bWant >= width) {
                        int ov = bWant - (width - 1);
                        bWant -= ov;
                        aWant -= ov;
                        if (aWant < 0) aWant = 0;
                    }
                    // Update sliding sums to new [aWant, bWant].
                    while (a < aWant) {
                        sumAmp -= magRow[a];
                        sumRes -= resRow[a];
                        sumI -= (double)triRow[a];
                        sumQ -= (double)trqRow[a];
                        ++a;
                    }
                    while (a > aWant) {
                        --a;
                        sumAmp += magRow[a];
                        sumRes += resRow[a];
                        sumI += (double)triRow[a];
                        sumQ += (double)trqRow[a];
                    }
                    while (b < bWant) {
                        ++b;
                        sumAmp += magRow[b];
                        sumRes += resRow[b];
                        sumI += (double)triRow[b];
                        sumQ += (double)trqRow[b];
                    }
                    while (b > bWant) {
                        sumAmp -= magRow[b];
                        sumRes -= resRow[b];
                        sumI -= (double)triRow[b];
                        sumQ -= (double)trqRow[b];
                        --b;
                    }
                }
                const double ampEst = sumAmp / (double)winN;
                const double resAmp = sumRes / (double)winN;
                const double meanI = sumI / (double)winN;
                const double meanQ = sumQ / (double)winN;
                const double meanMag = std::hypot(meanI, meanQ);
                const double coherence = (ampEst > 1e-9)
                    ? std::clamp(meanMag / ampEst, 0.0, 1.0)
                    : 0.0;

                // Fitted IQ at xi: keep the windowed amplitude estimate, but prefer
                // a phase direction that is coherent across the local support.
                const double mag0 = std::hypot(ri, rq);
                double localFitI = ri, localFitQ = rq;
                if (mag0 > 1e-9) {
                    localFitI = ri * (ampEst / mag0);
                    localFitQ = rq * (ampEst / mag0);
                }
                double fI = localFitI;
                double fQ = localFitQ;
                if (meanMag > 1e-9) {
                    const double phaseFitI = meanI * (ampEst / meanMag);
                    const double phaseFitQ = meanQ * (ampEst / meanMag);
                    const double phaseBlend = coherence * coherence;
                    fI = localFitI + (phaseFitI - localFitI) * phaseBlend;
                    fQ = localFitQ + (phaseFitQ - localFitQ) * phaseBlend;
                }

                const double ratio = (ampEst > 1e-9) ? (resAmp / ampEst) : 1.0;
                const double vetScale = std::max(0.25, T.SINFIT_VET_THRESHOLD_IRE);
                const double vetNorm = ratio / vetScale;
                const double qualityWeight = (0.25 + 0.75 * coherence)
                    / (1.0 + vetNorm * vetNorm);

                // Accumulate the line-level rotation fit with soft support so
                // steep saturated regions inform the solve without dominating it.
                if (qualityWeight > 1e-6) {
                    STT[0][0] += qualityWeight * fI*fI; STT[0][1] += qualityWeight * fI*fQ;
                    STT[1][0] += qualityWeight * fI*fQ; STT[1][1] += qualityWeight * fQ*fQ;
                    SRT[0][0] += qualityWeight * ri*fI; SRT[0][1] += qualityWeight * ri*fQ;
                    SRT[1][0] += qualityWeight * rq*fI; SRT[1][1] += qualityWeight * rq*fQ;
                }
            }
            // Affine solve — stored for buildPhaseCorrected1D to apply after split1D
            LineAffine &la = grammar->affine;
            la.valid = false;
            grammar->phaseError = 0.0;
            double STTinv[2][2];
            if (mat2_inv(STT, STTinv)) {
                double tmp[2][2], A[2][2];
                mat2_mul(SRT, STTinv, tmp);
                A[0][0]=tmp[0][0]; A[0][1]=tmp[0][1];
                A[1][0]=tmp[1][0]; A[1][1]=tmp[1][1];
                double Rm[2][2], U[2][2];
                polar_decompose_2x2(A, Rm, U);
                const double measuredPhase = std::atan2(Rm[1][0], Rm[0][0]);
                grammar->phaseError = std::clamp(
                    measuredPhase,
                    -PHASE_ERROR_CAP,
                    PHASE_ERROR_CAP);
                // Unclamped phase error magnitude against quarter-turn: how
                // far the burst-derived carrier model diverges from the
                // metadata-derived schedule.  ≥45° would indicate a full
                // polarity disagreement.
                grammar->phaseScheduleConflict = std::clamp(
                    std::fabs(measuredPhase) / (M_PI / 4.0), 0.0, 1.0);
                if (grammar->phaseScheduleConflict < 0.1)
                    grammar->lineFlipAuthority =
                        lddecode::CarrierPhaseAuthority::BurstMeasured;
                if (!writeAffine)
                    continue;
                const double pMax = T.Y_LINE_MAX_PHASE_DEG * M_PI / 180.0;
                clamp_rotation_gain_shear(Rm, U, pMax,
                                          T.Y_LINE_ALLOW_GAIN_ON_IQ,
                                          T.Y_LINE_GAIN_MIN, T.Y_LINE_GAIN_MAX,
                                          T.Y_LINE_MAX_SHEAR);
                la.R[0][0]=Rm[0][0]; la.R[0][1]=Rm[0][1];
                la.R[1][0]=Rm[1][0]; la.R[1][1]=Rm[1][1];
                la.valid = true;
            }
        }

        if (T.Y_LINE_PHASE_ERROR_LUT_ENABLE && !writeAffine) {
            for (int line = firstLine; line < lastLine; ++line) {
                CombCarrierGrammar *grammar = carrierGrammarLine(line);
                if (!grammar || !grammar->grammarLocked)
                    continue;
                if (grammar->phaseConfidence < T.Y_LINE_PHASE_ERROR_MIN_CONF)
                    continue;

                const double phase = grammar->phaseError;
                if (!std::isfinite(phase) || std::fabs(phase) < 1e-12)
                    continue;

                const double c = std::cos(phase);
                const double s = std::sin(phase);
                for (int i = 0; i < 4; ++i) {
                    const double ti = (double)grammar->demodLUTTi[i];
                    const double tq = (double)grammar->demodLUTTq[i];
                    grammar->demodLUTTi[i] = (float)(c * ti - s * tq);
                    grammar->demodLUTTq[i] = (float)(s * ti + c * tq);
                }
            }
        }
    }
}

// Demodulates the combed carrier (from buildCarrierRetracted) into two
// explicit products:
//   1) demodTI/TQ: line-local locked IQ after burst alignment and affine trim.
//   2) demodTI4fsc/TQ4fsc + locked1DSource: the common 4fsc export derived
//      from that locked IQ, used as the cross-line scalar reference for 2D work.
//
// The combed carrier is the LS carrier fit after line-to-line cancellation —
// alien-Y at fsc has been rejected and only carrier-shaped energy that
// inverts between opposite-phase lines survives.
void Comb::FrameBuffer::buildPhaseCorrected1D()
{
    const int first  = videoParameters.firstActiveFrameLine;
    const int last   = videoParameters.lastActiveFrameLine;
    const int left   = videoParameters.activeVideoStart;
    const int right  = videoParameters.activeVideoEnd;
    const int width  = right - left;
    const auto &T    = configuration.tunables;

    if (width <= 0 || first >= last) return;

    if ((int)scratch_lumaBaseY4.size() < width) scratch_lumaBaseY4.resize(width, 0.0);
    if ((int)scratch_lumaSmooth.size()  < width) scratch_lumaSmooth.resize(width, 0.0);
    if ((int)scratch_preI.size() < width) scratch_preI.resize(width, 0.0);
    if ((int)scratch_preQ.size() < width) scratch_preQ.resize(width, 0.0);

    if ((int)locked1DSource.size() < last) locked1DSource.resize(last);

    // Source: combed carrier from buildCarrierRetracted().
    // This replaces the blind bandpass (clpbuffer[0]) with the LS carrier
    // model that has been cleaned by line-to-line cancellation.  Alien-Y
    // at fsc has been rejected; only carrier-shaped energy that inverts
    // between opposite-phase lines survives.
    const bool haveCombedCarrier = carrierRetractedValid && !combedCarrier_flat.empty();

    for (int line = first; line < last; ++line) {
        const quint16 *rawLine = rawbuffer.data() + line * videoParameters.fieldWidth;

        // xi-indexed combed carrier for this line (null if retraction unavailable)
        const float *combSrc = haveCombedCarrier ? combedCarrier_line(line) : nullptr;

        auto &ldsRow = locked1DSource[line];
        if ((int)ldsRow.size() < width) ldsRow.assign(width, 0.0);
        seedCombOwnershipPerLine(line);

        CombCarrierGrammar *grammar = carrierGrammarLine(line);
        const bool grammarLocked = grammar && grammar->grammarLocked;
        const double bcos = grammarLocked ? grammar->burstCos : 1.0;
        const double bsin = grammarLocked ? grammar->burstSin : 0.0;
        const double *baseY4 = (lockedLumaCacheValid &&
            !lockedLumaBaseY4_flat.empty() && demodWidth == width)
            ? lockedLumaBaseY4_line(line) : nullptr;
        const float *floorRow = (carrierRetractedValid &&
            !flatFloor_flat.empty() && demodWidth == width)
            ? flatFloor_line(line) : nullptr;
        double sumFwdError = 0.0, sumChromaMag = 0.0;
        double sumResidualSq = 0.0, sumErrorSq = 0.0;
        int projCount = 0;

        double lutTi[4], lutTq[4];
        if (grammarLocked) {
            for (int i = 0; i < 4; ++i) {
                lutTi[i] = (double)grammar->demodLUTTi[i];
                lutTq[i] = (double)grammar->demodLUTTq[i];
            }
        } else {
            fusedDemodLUT(bcos, bsin, spLUT_locked, cpLUT_locked, lutTi, lutTq);
        }

        float *tiRow  = demodTI_line(line);
        float *tqRow  = demodTQ_line(line);
        float *ti4Row = demodTI4fsc_line(line);
        float *tq4Row = demodTQ4fsc_line(line);
        float *locked1DTi4Row = locked1DTI4fsc_line(line);
        float *locked1DTq4Row = locked1DTQ4fsc_line(line);

        const bool haveAffine =
            configuration.residualVideo &&
            T.Y_LINE_AFFINE_TRIM_ENABLE &&
            grammarLocked &&
            grammar->affine.valid;

        const LineAffine *lineAffine =
            haveAffine ? &grammar->affine : nullptr;

        auto applyLineAffine = [&](double &ti, double &tq) {
            if (!lineAffine) return;
            const double ai = lineAffine->R[0][0] * ti + lineAffine->R[0][1] * tq;
            const double aq = lineAffine->R[1][0] * ti + lineAffine->R[1][1] * tq;
            ti = ai;
            tq = aq;
        };

        // Luma smooth for directional edge analysis (still from the luma
        // decomposition, not the carrier source).
        const double *lumaSmooth = nullptr;
        if (lockedLumaCacheValid &&
            !lockedLumaSmooth_flat.empty() &&
            demodWidth == width)
        {
            lumaSmooth = lockedLumaSmooth_line(line);
        } else {
            buildCompositeLumaDecompositionLine(rawLine, left, width,
                                                scratch_lumaBaseY4.data(),
                                                nullptr,
                                                scratch_lumaSmooth.data());
            lumaSmooth = scratch_lumaSmooth.data();
        }

        double *tiBase = scratch_preI.data();
        double *tqBase = scratch_preQ.data();

        // Demod the combed carrier through the locked basis.
        // combSrc is xi-indexed (0..width-1); the demod LUT uses the
        // carrier sample class at absolute position h.
        for (int xi = 0; xi < width; ++xi) {
            const int h = left + xi;
            const int ph = carrierSampleClass(line, h);
            const double cv = combSrc ? static_cast<double>(combSrc[xi]) : 0.0;
            double ti = cv * lutTi[ph];
            double tq = cv * lutTq[ph];
            applyLineAffine(ti, tq);
            tiBase[xi] = ti;
            tqBase[xi] = tq;
        }

        // Sample accessor for bandpass-scale metrics on the combed carrier.
        auto sampleComb = [&](int rel)->double {
            rel = std::clamp(rel, 0, width - 1);
            return combSrc ? static_cast<double>(combSrc[rel]) : 0.0;
        };

        for (int xi = 0; xi < width; ++xi) {
            const int h = left + xi;
            const int ph = carrierSampleClass(line, h);
            double ti = tiBase[xi];
            double tq = tqBase[xi];

            double intakeNyquistRiskIRE = 0.0;
            double lumaIncursionRiskIRE = 0.0;
            double residualFitErrorIRE = 0.0;
            double fine = 0.0, mid = 0.0, coarse = 0.0;
            double directionalEdgeSupport = 0.0;
            double bpLumaModeled = 0.0;
            double icebergAlienYFraction = 0.0;

            // Bandpass-scale metrics on the combed carrier.
            if (xi >= 4 && xi < width - 4) {
                const double c0 = sampleComb(xi);
                fine   = std::fabs(c0 - 0.5 * (sampleComb(xi - 1) + sampleComb(xi + 1))) * invIreScale;
                mid    = std::fabs(c0 - 0.5 * (sampleComb(xi - 2) + sampleComb(xi + 2))) * invIreScale;
                coarse = std::fabs(c0 - 0.5 * (sampleComb(xi - 4) + sampleComb(xi + 4))) * invIreScale;
            } else {
                fine = std::fabs(sampleComb(xi) -
                                 0.5 * (sampleComb(xi - 1) + sampleComb(xi + 1))) * invIreScale;
                mid = std::fabs(sampleComb(xi) -
                                0.5 * (sampleComb(xi - 2) + sampleComb(xi + 2))) * invIreScale;
                coarse = std::fabs(sampleComb(xi) -
                                   0.5 * (sampleComb(xi - 4) + sampleComb(xi + 4))) * invIreScale;
            }

            const double denom = fine + mid + coarse + 1e-9;
            const double fineFrac = fine / denom;
            const double nonFineFrac = std::max(mid, coarse) / denom;
            const double dominance =
                std::clamp((fineFrac - nonFineFrac - 0.15) / 0.35, 0.0, 1.0);

            intakeNyquistRiskIRE = fine * dominance;

            const int xm1 = std::clamp(xi - 1, 0, width - 1);
            const int xp1 = std::clamp(xi + 1, 0, width - 1);
            const int xm2 = std::clamp(xi - 2, 0, width - 1);
            const int xp2 = std::clamp(xi + 2, 0, width - 1);

            const double tiLm1 = tiBase[xm1];
            const double tqLm1 = tqBase[xm1];
            const double tiLp1 = tiBase[xp1];
            const double tqLp1 = tqBase[xp1];
            const double tiLm2 = tiBase[xm2];
            const double tqLm2 = tqBase[xm2];
            const double tiLp2 = tiBase[xp2];
            const double tqLp2 = tqBase[xp2];

            const double avg1I = 0.5 * (tiLm1 + tiLp1);
            const double avg1Q = 0.5 * (tqLm1 + tqLp1);
            const double avg2I = 0.5 * (tiLm2 + tiLp2);
            const double avg2Q = 0.5 * (tqLm2 + tqLp2);

            const double err1IRE = std::hypot(ti - avg1I, tq - avg1Q) * invIreScale;
            const double err2IRE = std::hypot(ti - avg2I, tq - avg2Q) * invIreScale;
            const double iqMagIRE = std::hypot(ti, tq) * invIreScale;

            residualFitErrorIRE = 0.65 * err1IRE + 0.35 * err2IRE;

            const double incoherence = std::clamp(
                (residualFitErrorIRE - std::max(1.0, 0.25 * iqMagIRE)) / 4.0,
                0.0, 1.0);

            lumaIncursionRiskIRE = intakeNyquistRiskIRE * incoherence;

            {
                const double cLm2 = lumaSmooth[std::clamp(xi - 2, 0, width - 1)];
                const double cLm1 = lumaSmooth[std::clamp(xi - 1, 0, width - 1)];
                const double c0   = lumaSmooth[xi];
                const double cLp1 = lumaSmooth[std::clamp(xi + 1, 0, width - 1)];
                const double cLp2 = lumaSmooth[std::clamp(xi + 2, 0, width - 1)];

                const double gLm = cLm1 - cLm2;
                const double gL0 = c0 - cLm1;
                const double g0R = cLp1 - c0;
                const double gRp = cLp2 - cLp1;

                auto slopeAgreement = [](double a, double b)->double {
                    const double aa = std::fabs(a);
                    const double bb = std::fabs(b);
                    if (aa < 1e-9 || bb < 1e-9 || (a * b) <= 0.0)
                        return 0.0;
                    return std::sqrt(std::min(aa, bb) / std::max(aa, bb));
                };

                const double monotonicity =
                    0.4 * slopeAgreement(gL0, g0R) +
                    0.3 * slopeAgreement(gLm, gL0) +
                    0.3 * slopeAgreement(g0R, gRp);

                const double edgeSpanIRE = std::fabs(cLp1 - cLm1) * invIreScale;
                const double longSpanIRE = std::fabs(cLp2 - cLm2) * invIreScale;
                const double edgeStrengthIRE = std::max(edgeSpanIRE, 0.75 * longSpanIRE);

                const double leftCrestIRE =
                    std::fabs(c0 - 0.5 * (cLm1 + cLm2)) * invIreScale;
                const double rightCrestIRE =
                    std::fabs(c0 - 0.5 * (cLp1 + cLp2)) * invIreScale;
                const double crestAsym =
                    std::fabs(leftCrestIRE - rightCrestIRE) /
                    (leftCrestIRE + rightCrestIRE + 1e-9);

                const double centerAsym =
                    std::fabs(std::fabs(gL0) - std::fabs(g0R)) /
                    (std::fabs(gL0) + std::fabs(g0R) + 1e-9);

                directionalEdgeSupport = monotonicity * std::clamp(
                    (edgeStrengthIRE - (0.20 * fine) - 0.5) /
                    std::max(1.5, (0.50 * fine) + 1.0),
                    0.0, 1.0);

                const double directionalEdgeAsymmetry = std::clamp(
                    (0.65 * crestAsym) + (0.35 * centerAsym),
                    0.0, 1.0);

                directionalEdgeSupport *= (1.0 - 0.35 * directionalEdgeAsymmetry);
            }

            if (T.LUMA_ICEBERG_RECOVERY > 0.0 &&
                directionalEdgeSupport > 0.0)
            {
                if (floorRow && baseY4) {
                    bpLumaModeled =
                        (static_cast<double>(floorRow[xi]) - baseY4[xi]) *
                        T.LUMA_ICEBERG_RECOVERY *
                        directionalEdgeSupport;
                } else if (xi >= 2 && xi < width - 2) {
                    const double bpLumaPredicted =
                        (lumaSmooth[xi] -
                         0.5 * (lumaSmooth[xi - 2] + lumaSmooth[xi + 2])) * 0.5;
                    bpLumaModeled =
                        bpLumaPredicted *
                        T.LUMA_ICEBERG_RECOVERY *
                        directionalEdgeSupport;
                }

                const double modeledAlienYIRE = std::fabs(bpLumaModeled) * invIreScale;
                icebergAlienYFraction = std::clamp(
                    modeledAlienYIRE / std::max(1.0, iqMagIRE),
                    0.0, 1.0);
            }

            if (line >= 0 && line < (int)fvfMetrics.size() &&
                xi < (int)fvfMetrics[line].size())
            {
                fvfMetrics[line][xi].intakeNyquistRiskIRE = intakeNyquistRiskIRE;
                fvfMetrics[line][xi].lumaIncursionRiskIRE = lumaIncursionRiskIRE;
                fvfMetrics[line][xi].residualFitErrorIRE  = residualFitErrorIRE;
            }

            if (line >= 0 && line < (int)ownershipEvidence.size() &&
                xi < (int)ownershipEvidence[line].size())
            {
                OwnershipEvidence &e = ownershipEvidence[line][xi];
                e.facts.bandpassFineIRE = fine;
                e.facts.bandpassMidIRE = mid;
                e.facts.bandpassCoarseIRE = coarse;
                e.facts.lumaExcursionIRE = intakeNyquistRiskIRE;
                e.facts.residualFitErrorIRE = residualFitErrorIRE;
                e.facts.lumaIncursionRiskIRE = lumaIncursionRiskIRE;
                e.facts.icebergAlienYFraction = icebergAlienYFraction;
                e.facts.locked1DChromaIRE = std::hypot(ti, tq) * invIreScale;
            }

#if 0   // disabled: produceY now handles alien-Y via flatFloor decomposition;
        // enabling here would double-count in the 1D→2D→3D candidate path
            if (bpLumaModeled != 0.0) {
                double corrTi = bpLumaModeled * lutTi[ph];
                double corrTq = bpLumaModeled * lutTq[ph];
                applyLineAffine(corrTi, corrTq);
                ti -= corrTi;
                tq -= corrTq;
            }
#endif

            tiRow[xi] = (float)ti;
            tqRow[xi] = (float)tq;

            double ti4 = 0.0;
            double tq4 = 0.0;
            lockedTo4fsc(ti, tq, bcos, bsin, ti4, tq4);

            ti4Row[xi] = (float)ti4;
            tq4Row[xi] = (float)tq4;
            locked1DTi4Row[xi] = (float)ti4;
            locked1DTq4Row[xi] = (float)tq4;

            ldsRow[xi] = remod4fscToShiftedComposite(ti4, tq4, h, spLUT_locked, cpLUT_locked);

            if (baseY4 && grammarLocked) {
                const double residual = (double)rawLine[h] - baseY4[xi];
                const double rI = residual * lutTi[ph];
                const double rQ = residual * lutTq[ph];
                double rI4, rQ4;
                lockedTo4fsc(rI, rQ, bcos, bsin, rI4, rQ4);
                const double cModel = remod4fscToShiftedComposite(rI4, rQ4, h, spLUT_locked, cpLUT_locked);
                const double fwdErr = residual - cModel;
                sumFwdError   += std::fabs(fwdErr);
                sumChromaMag  += std::hypot(rI4, rQ4);
                sumResidualSq += residual * residual;
                sumErrorSq    += fwdErr * fwdErr;
                ++projCount;
            }
        }

        if (grammarLocked && grammar && projCount > 0) {
            const double invCount = 1.0 / (double)projCount;
            grammar->meanForwardErrorIRE = sumFwdError * invCount * invIreScale;
            grammar->meanChromaMagIRE    = sumChromaMag * invCount * invIreScale;
            grammar->carrierFitRatio     = (sumResidualSq > 1e-12)
                ? std::clamp(1.0 - sumErrorSq / sumResidualSq, 0.0, 1.0)
                : 0.0;
            grammar->projectionValid = true;
        } else if (grammar) {
            grammar->projectionValid = false;
        }

        if (FIELD_BUCKET_SMOOTH_STRENGTH > 0.0) {
            if ((int)scratch_filter_temp.size() < width)
                scratch_filter_temp.assign(width, 0.0);

            auto reflectXi = [&](int x)->int {
                if (x < 0) return -x;
                if (x >= width) return (width - 1) - (x - (width - 1));
                return x;
            };

            for (int xi = 0; xi < width; ++xi) {
                const int xm4 = reflectXi(xi - 4);
                const int xp4 = reflectXi(xi + 4);
                const double raw = ldsRow[xi];
                const double est = 0.5 * (ldsRow[xm4] + ldsRow[xp4]);
                scratch_filter_temp[xi] =
                    raw + (est - raw) * FIELD_BUCKET_SMOOTH_STRENGTH;
            }

            for (int xi = 0; xi < width; ++xi)
                ldsRow[xi] = scratch_filter_temp[xi];
        }
    }

    // Populate clpbuffer[0] from locked1DSource so all downstream consumers
    // (produceY no-residual path, diagnostics, any future code) get the
    // upgraded carrier model output without needing to branch on locked path.
    if (configuration.phaseCompensation) {
        for (int line = first; line < last; ++line) {
            if (line < (int)locked1DSource.size() &&
                (int)locked1DSource[line].size() >= width)
            {
                double *dst = clpbuffer[0].pixel[line];
                const auto &src = locked1DSource[line];
                for (int xi = 0; xi < width; ++xi)
                    dst[left + xi] = src[xi];
            }
        }
    }

}

void Comb::FrameBuffer::splitIQlocked()
{
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;
    const int srcBuf    = std::clamp(static_cast<int>(configuration.dimensions) - 1, 0, 2);
    const auto &T       = configuration.tunables;

    if (width <= 0 || firstLine >= lastLine) return;

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

    if ((int)scratch_preI.size() < width) scratch_preI.resize(width, 0.0);
    if ((int)scratch_preQ.size() < width) scratch_preQ.resize(width, 0.0);

    // splitIQlocked owns the post-comb locked demod. It refreshes the stable
    // selected-comb demod reference and seeds the downstream locked-product cache.
    for (int line = firstLine; line < lastLine; ++line) {
        const double *src = clpbuffer[srcBuf].pixel[line];

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

        float *tiRow = demodTI_line(line);
        float *tqRow = demodTQ_line(line);
        float *ti4Row = demodTI4fsc_line(line);
        float *tq4Row = demodTQ4fsc_line(line);
        float *prodIRow = lockedProductI_line(line);
        float *prodQRow = lockedProductQ_line(line);

        const LineAffine *lineAffine = nullptr;
        if (configuration.residualVideo && T.Y_LINE_AFFINE_TRIM_ENABLE
                && grammarLocked && grammar->affine.valid) {
            lineAffine = &grammar->affine;
        }
        auto applyLineAffine = [&](double &ti, double &tq) {
            if (!lineAffine) return;
            const double ai = lineAffine->R[0][0] * ti + lineAffine->R[0][1] * tq;
            const double aq = lineAffine->R[1][0] * ti + lineAffine->R[1][1] * tq;
            ti = ai;
            tq = aq;
        };

        for (int xi = 0; xi < width; ++xi) {
            const int h = left + xi;
            const int ph = carrierSampleClass(line, h);
            double ti = src[h] * lutTi[ph];
            double tq = src[h] * lutTq[ph];
            applyLineAffine(ti, tq);

            double ti4 = 0.0;
            double tq4 = 0.0;
            lockedTo4fsc(ti, tq, bcos, bsin, ti4, tq4);

            const float prodI = (float)(ti * GI_PRODUCT);
            const float prodQ = (float)(tq * GQ_PRODUCT);

            if (tiRow) tiRow[xi] = (float)ti;
            if (tqRow) tqRow[xi] = (float)tq;
            if (ti4Row) ti4Row[xi] = (float)ti4;
            if (tq4Row) tq4Row[xi] = (float)tq4;
            if (prodIRow) prodIRow[xi] = prodI;
            if (prodQRow) prodQRow[xi] = prodQ;
            scratch_preI[xi] = prodI;
            scratch_preQ[xi] = prodQ;
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

            double dc = (double)rawLine[left] - Yrow[left];
            constexpr double DC_ALPHA = 1.0 / 64.0;
            for (int i = 0; i < width; ++i) {
                const int h = left + i;
                const double chromaRaw = (double)rawLine[h] - Yrow[h];
                dc += DC_ALPHA * (chromaRaw - dc);
                const double chroma = chromaRaw - dc;
                const int ph = carrierSampleClass(line, h);
                scratch_preI[i] = (chroma * lutTi[ph]) * GI_PRODUCT;
                scratch_preQ[i] = (chroma * lutTq[ph]) * GQ_PRODUCT;
            }
        } else {
            // The normal locked path consumes the cache prepared by splitIQlocked()
            // and optionally refined by produceY().
            const float *prodIRow = lockedProductI_line(line);
            const float *prodQRow = lockedProductQ_line(line);
            for (int i = 0; i < width; ++i) {
                scratch_preI[i] = prodIRow ? (double)prodIRow[i] : 0.0;
                scratch_preQ[i] = prodQRow ? (double)prodQRow[i] : 0.0;
            }
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

// ---------------------------------------------------------------------------
// produceY helpers
// ---------------------------------------------------------------------------

namespace {

// Ownership alpha blend given pre-computed per-pixel scalars.
// Returns the effective alpha (0 = full chroma subtraction, 1 = no subtraction
// modified by ownership and saturation guards).
double computeOwnershipAlpha(
    const lddecode::CombOwnershipAssessment &a,
    double alphaVet,
    double lumaIRE,
    double chromaIRE,
    double ownershipWeight,
    double ownershipChromaWeight,
    double brightStartIRE, double brightFullIRE,
    double satStartIRE,    double satFullIRE)
{
    const double lc = std::clamp(a.lumaClaim,    0.0, 1.0);
    const double cc = std::clamp(a.chromaClaim,  0.0, 1.0);
    const double uc = std::clamp(a.uncertainClaim, 0.0, 1.0);

    const double chromaFrac = std::clamp(
        1.0 - lc + ownershipChromaWeight * cc, 0.0, 1.0);

    const double brightT = std::clamp(
        (lumaIRE - brightStartIRE) /
        std::max(1e-9, brightFullIRE - brightStartIRE),
        0.0, 1.0);

    const double satT = std::clamp(
        (chromaIRE - satStartIRE) /
        std::max(1e-9, satFullIRE - satStartIRE),
        0.0, 1.0);

    // Strong chroma or explicit chroma ownership should block luma-ownership
    // from reducing residual chroma subtraction. This is especially important
    // for saturated blue, where lumaIRE may not trip the brightness ramp.
    const double chromaBlock = std::clamp(cc + 0.65 * satT, 0.0, 1.0);
    const double support = lc * (1.0 - chromaBlock) * (1.0 - 0.5 * uc);

    // Saturated color is already a warning that carrier-band energy is probably
    // chroma-owned; do not require high luma for ownership reassignment to bow out.
    const double ownershipProtect = std::clamp(
        std::max(satT, brightT * satT), 0.0, 1.0);

    const double blend = std::clamp(
        ownershipWeight * support * (1.0 - ownershipProtect), 0.0, 1.0);

    return alphaVet * (1.0 - blend) + chromaFrac * blend;
}

} // namespace

void Comb::FrameBuffer::ensureProduceYScratch(int width)
{
    if ((int)scratch_frameBCenter.size() < width) scratch_frameBCenter.resize(width, 0.0);
    if ((int)scratch_fieldBCenter.size() < width) scratch_fieldBCenter.resize(width, 0.0);
    if ((int)scratch_comp_res.size()     < width) scratch_comp_res.resize(width, 0.0);
    if ((int)scratch_fieldGate.size()    < width) scratch_fieldGate.resize(width, 1.0);
    if ((int)scratch_fieldLine.size()    < width) scratch_fieldLine.resize(width, 0.0);
    if ((int)scratch_fieldBLine.size()   < width) scratch_fieldBLine.resize(width, 0.0);
    if ((int)scratch_lateralLine.size()  < width) scratch_lateralLine.resize(width, 0.0);
    if ((int)scratch_yhp.size()          < width) scratch_yhp.resize(width, 0.0);
    if ((int)scratch_yI.size()           < width) scratch_yI.resize(width, 0.0);
    if ((int)scratch_yQ.size()           < width) scratch_yQ.resize(width, 0.0);
    if ((int)scratch_hpI.size()          < width) scratch_hpI.resize(width, 0.0);
    if ((int)scratch_hpQ.size()          < width) scratch_hpQ.resize(width, 0.0);
    if ((int)scratch_hpY.size()          < width) scratch_hpY.resize(width, 0.0);
    if ((int)scratch_outMixed.size()     < width) scratch_outMixed.resize(width, 0.0);
    if ((int)scratch_filter_temp.size()  < width) scratch_filter_temp.resize(width, 0.0);
    if ((int)scratch_lumaBaseY4.size()   < width) scratch_lumaBaseY4.resize(width, 0.0);
}

// Windowed affine solve for one scan line.
//
// Reads tiRow/tqRow (BurstLockedSigned IQ from phaseLocked) and hiRaw
// (raw − baseY4).  Accumulates per-pixel moment matrices (STT, SRT),
// slides a WIN-wide window across them, inverts STT, polar-decomposes the
// affine, vets phase/rho/shear, clamps rotation/gain/shear, remods cHat.
//
// Outputs written to member scratch buffers:
//   scratch_fieldGate   → cHat  (remodulated coherent carrier estimate)
//   scratch_fieldLine   → tiAdjLocked (IQ after affine, in locked space)
//   scratch_fieldBLine  → tqAdjLocked
//   scratch_lateralLine → vetConf (VET confidence [0,1])
void Comb::FrameBuffer::buildCoherentCarrierRow(
    const float *tiRow, const float *tqRow,
    const double *hiRaw,
    int left, int width,
    double bcos, double bsin,
    int WIN, int HALF,
    double invI)
{
    const auto &T = configuration.tunables;

    constexpr double MIN_FIT_IRE = 2.0;
    constexpr double MAX_FIT_IRE = 35.0;
    constexpr double SAT_TROUBLE_IRE = 18.0;

    double *cHat        = scratch_fieldGate.data();
    double *tiAdjLocked = scratch_fieldLine.data();
    double *tqAdjLocked = scratch_fieldBLine.data();
    double *vetConf     = scratch_lateralLine.data();

    double *cSTT00 = scratch_yhp.data();
    double *cSTT01 = scratch_yI.data();
    double *cSTT11 = scratch_yQ.data();
    double *cSRT00 = scratch_hpI.data();
    double *cSRT01 = scratch_hpQ.data();
    double *cSRT10 = scratch_hpY.data();
    double *cSRT11 = scratch_outMixed.data();
    double *cN     = scratch_filter_temp.data();

    // Pass 1: accumulate per-pixel moment matrices
    for (int x = 0; x < width; ++x) {
        const int h = left + x;

        double ti = 0.0, tq = 0.0;
        lockedTo4fsc((double)tiRow[x], (double)tqRow[x], bcos, bsin, ti, tq);

        double ri = 0.0, rq = 0.0;
        demod4fscFromComposite(hiRaw[x], h, ri, rq);

        tiAdjLocked[x] = ti;
        tqAdjLocked[x] = tq;

        const double magT_ire = std::hypot(ti, tq) * invI;
        const double magR_ire = std::hypot(ri, rq) * invI;

        if (magT_ire < MIN_FIT_IRE || magR_ire < MIN_FIT_IRE) {
            cSTT00[x] = cSTT01[x] = cSTT11[x] = 0.0;
            cSRT00[x] = cSRT01[x] = cSRT10[x] = cSRT11[x] = 0.0;
            cN[x] = 0.0;
            continue;
        }

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
        cN[x] = 1.0;
    }

    // Pass 2: sliding-window affine solve + VET + remod
    const int winN = (width <= WIN) ? width : WIN;

    double sSTT00 = 0.0, sSTT01 = 0.0, sSTT11 = 0.0;
    double sSRT00 = 0.0, sSRT01 = 0.0, sSRT10 = 0.0, sSRT11 = 0.0;
    double sN = 0.0;

    for (int i = 0; i < winN; ++i) {
        sSTT00 += cSTT00[i]; sSTT01 += cSTT01[i]; sSTT11 += cSTT11[i];
        sSRT00 += cSRT00[i]; sSRT01 += cSRT01[i];
        sSRT10 += cSRT10[i]; sSRT11 += cSRT11[i];
        sN += cN[i];
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
            sN += cN[add] - cN[rem];
        }

        double STT[2][2] = {{sSTT00, sSTT01}, {sSTT01, sSTT11}};
        double SRT[2][2] = {{sSRT00, sSRT01}, {sSRT10, sSRT11}};
        const int n = (int)(sN + 0.5);

        double STTinv[2][2];
        const bool invOk = mat2_inv(STT, STTinv);

        double RmVet[2][2] = {{1, 0}, {0, 1}};
        double UVet[2][2]  = {{1, 0}, {0, 1}};
        bool vetAccept = invOk && n >= 8;
        double vetConfidence = 0.0;

        if (vetAccept) {
            double Avet[2][2];
            mat2_mul(SRT, STTinv, Avet);
            polar_decompose_2x2(Avet, RmVet, UVet);

            const double phase = std::atan2(RmVet[1][0], RmVet[0][0]);

            double l1 = 1.0, l2 = 1.0;
            double V_[2][2];
            eig2_sym(UVet, l1, l2, V_);

            const double g = 0.5 * (std::max(0.0, l1) + std::max(0.0, l2));
            const double shear = (g > 1e-12)
                ? std::fabs(std::max(0.0, l1) - std::max(0.0, l2)) / g : 0.0;

            const double numRho = std::sqrt(
                SRT[0][0]*SRT[0][0] + SRT[0][1]*SRT[0][1] +
                SRT[1][0]*SRT[1][0] + SRT[1][1]*SRT[1][1]);
            const double rho = numRho / std::max(1e-9, STT[0][0] + STT[1][1]);

            const double pMaxVet = T.VET_ALIGN_PHASE_MAX_DEG * M_PI / 180.0;
            vetAccept = std::fabs(phase) <= pMaxVet &&
                        rho >= T.VET_ALIGN_MIN_RHO &&
                        shear <= T.VET_ALIGN_MAX_SHEAR;

            const double c_phase =
                1.0 - std::min(1.0, std::fabs(phase) / (pMaxVet + 1e-12));
            const double c_shear =
                1.0 - std::min(1.0, shear / (T.VET_ALIGN_MAX_SHEAR + 1e-12));
            double c = 0.5  * std::max(0.0, std::min(1.0, rho)) +
                       0.25 * c_phase + 0.25 * c_shear;
            vetConfidence = vetAccept ? std::clamp(c, 0.0, 1.0) : 0.0;
        }

        double Rm[2][2] = {{1, 0}, {0, 1}};
        double U[2][2]  = {{1, 0}, {0, 1}};
        if (T.Y_LOCAL_AFFINE_ENABLE && n >= 16 && invOk && vetAccept) {
            Rm[0][0] = RmVet[0][0]; Rm[0][1] = RmVet[0][1];
            Rm[1][0] = RmVet[1][0]; Rm[1][1] = RmVet[1][1];
            U[0][0]  = UVet[0][0];  U[0][1]  = UVet[0][1];
            U[1][0]  = UVet[1][0];  U[1][1]  = UVet[1][1];
        }

        const double ti0 = tiAdjLocked[x];
        const double tq0 = tqAdjLocked[x];
        const bool satTrouble = (std::hypot(ti0, tq0) * invI > SAT_TROUBLE_IRE);

        clamp_rotation_gain_shear(Rm, U,
            T.Y_LOCAL_MAX_PHASE_DEG * M_PI / 180.0,
            /*allowGain=*/!satTrouble,
            T.Y_LOCAL_GAIN_MIN, T.Y_LOCAL_GAIN_MAX,
            satTrouble ? 0.0 : T.Y_LOCAL_MAX_SHEAR);

        const double ti_adj = Rm[0][0] * ti0 + Rm[0][1] * tq0;
        const double tq_adj = Rm[1][0] * ti0 + Rm[1][1] * tq0;

        // Use the shifted basis to round-trip the same ε offset used by the
        // locked demod (spLUT_locked built with CAL_EPS_SAMPLES). Plain
        // remod4fscToCompositePhase leaves a carrier-shaped residual in Y
        // because the demod/remod bases don't cancel.
        cHat[x]    = remod4fscToShiftedComposite(ti_adj, tq_adj, h, spLUT_locked, cpLUT_locked);
        vetConf[x] = vetConfidence;

        double ti_locked_adj = 0.0, tq_locked_adj = 0.0;
        fourfscToLocked(ti_adj, tq_adj, bcos, bsin, ti_locked_adj, tq_locked_adj);
        tiAdjLocked[x] = ti_locked_adj;
        tqAdjLocked[x] = tq_locked_adj;
    }
}

void Comb::FrameBuffer::produceY()
{
    if (!configuration.phaseCompensation) return;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;
    if (width <= 0) return;

    const auto &T = configuration.tunables;
    const bool enableResidualY = T.VET_ENABLE_RESIDUAL_Y;
    const double invI = this->invIreScale;

    const int srcBuf = std::clamp((int)configuration.dimensions - 1, 0, 2);

    const int WIN  = std::max(4, (T.VET_ALIGN_WIN_SAMPLES / 4) * 4);
    const int HALF = WIN / 2;

    const double MIN_ALPHA = 0.75;
    const double MAX_ALPHA = 1.25;
    const double MIN_SUB_CHROMA_IRE = 2.0;

    const bool showMap = configuration.showMap;
    const bool chromaLikeEnabled = (T.VET_Y_CHROMA_LIKE_WEIGHT > 0.0);
    const double chromaLikeWeight = T.VET_Y_CHROMA_LIKE_WEIGHT;
    const bool do3D =
        (configuration.residualVideo3D && prevFrameForVet && nextFrameForVet);

    ensureProduceYScratch(width);

    for (int line = firstLine; line < lastLine; ++line) {
        if (line >= demodLines) continue;

        const quint16 *rawLine = rawbuffer.data() + line * videoParameters.fieldWidth;
        const CombCarrierGrammar *grammar = carrierGrammarLine(line);
        const bool grammarLocked = grammar && grammar->grammarLocked;
        const double bcos = grammarLocked ? grammar->burstCos : 1.0;
        const double bsin = grammarLocked ? grammar->burstSin : 0.0;

        double *Y    = componentFrame->y(line);
        const double *clpLine = clpbuffer[srcBuf].pixel[line];

        if (!enableResidualY) {
            for (int h = left; h < right; ++h) {
                Y[h] = (double)rawLine[h] - clpLine[h];
            }

            if (showMap) {
                std::fill(w2d_frame_weight[line].begin(),
                          w2d_frame_weight[line].end(), 0.0f);
            }
            continue;
        }

        float *prodIRow = lockedProductI_line(line);
        float *prodQRow = lockedProductQ_line(line);

        if (!prodIRow || !prodQRow) {
            for (int h = left; h < right; ++h) {
                Y[h] = (double)rawLine[h] - clpLine[h];
            }

            if (showMap) {
                std::fill(w2d_frame_weight[line].begin(),
                          w2d_frame_weight[line].end(), 0.0f);
            }
            continue;
        }

        double lutTi[4], lutTq[4];
        if (grammarLocked) {
            for (int i = 0; i < 4; ++i) {
                lutTi[i] = (double)grammar->demodLUTTi[i];
                lutTq[i] = (double)grammar->demodLUTTq[i];
            }
        } else {
            fusedDemodLUT(bcos, bsin, spLUT_locked, cpLUT_locked, lutTi, lutTq);
        }

        double *baseY4 = scratch_frameBCenter.data();
        double *hiRaw  = scratch_fieldBCenter.data();

        const double *baseY4Src = nullptr;
        if (lockedLumaCacheValid &&
            !lockedLumaBaseY4_flat.empty() &&
            demodWidth == width)
        {
            baseY4Src = lockedLumaBaseY4_line(line);
        } else {
            buildCompositeLumaDecompositionLine(rawLine, left, width,
                                                scratch_lumaBaseY4.data(),
                                                nullptr,
                                                nullptr);
            baseY4Src = scratch_lumaBaseY4.data();
        }

        for (int x = 0; x < width; ++x) {
            baseY4[x] = baseY4Src[x];
            hiRaw[x]  = (double)rawLine[left + x] - baseY4Src[x];
        }

        YSourceView coarseY {
            "coarseY",
            YSourceNativeSpace::CompositeLuma,
            YSourceHomeOrientation::None,
            baseY4
        };
        YSourceView highRawY {
            "highRawY",
            YSourceNativeSpace::RawCompositeHigh,
            YSourceHomeOrientation::GrammarComposite,
            hiRaw
        };

        double *cHat        = scratch_fieldGate.data();
        double *tiAdjLocked = scratch_fieldLine.data();
        double *tqAdjLocked = scratch_fieldBLine.data();
        double *vetConf     = scratch_lateralLine.data();

        // Carrier estimate comes from the dimensionally-appropriate source.
        // For 1D the elected buffer is locked1DSource, which has no inter-line
        // cancellation: luma-near-fsc rides on it and subtracting it from raw
        // pulls a phase-alternating error into Y (the alien-chroma checkerboard).
        // combedCarrier from buildCarrierRetracted is the LS carrier fit after
        // line-to-line cancellation — real chroma is preserved, alien-Y is
        // zeroed. 2D/3D paths already carry cancellation in their own comb
        // election, so clpLine stays correct there.
        const float *combedCarrierRow = (srcBuf == 0 && carrierRetractedValid)
            ? combedCarrier_line(line) : nullptr;

        for (int x = 0; x < width; ++x) {
            const int h = left + x;
            const int ph = carrierSampleClass(line, h);
            const double carrier = combedCarrierRow
                ? (double)combedCarrierRow[x]
                : clpLine[h];
            cHat[x]        = carrier;
            tiAdjLocked[x] = carrier * lutTi[ph];
            tqAdjLocked[x] = carrier * lutTq[ph];
            vetConf[x]     = 1.0;
        }

        YSourceView coherentCombY {
            "coherentCombY",
            YSourceNativeSpace::BandpassComb,
            YSourceHomeOrientation::GrammarComposite,
            cHat
        };

        const bool ownershipEnabled =
            T.VET_OWNERSHIP_ENABLE &&
            line >= 0 &&
            line < (int)ownershipEvidence.size() &&
            (int)ownershipEvidence[line].size() >= width;

        const OwnershipEvidence *ownRow =
            ownershipEnabled ? ownershipEvidence[line].data() : nullptr;

        const double ownershipWeight = T.VET_OWNERSHIP_LUMA_WEIGHT;
        const double ownershipChromaWeight = T.VET_OWNERSHIP_CHROMA_WEIGHT;

        auto alphaWithOwnership = [&](int x, double alphaVet) -> double {
            return computeOwnershipAlpha(
                ownRow[x].assessment, alphaVet,
                (baseY4[x] - videoParameters.black16bIre) * invI,
                std::hypot(tiAdjLocked[x], tqAdjLocked[x]) * invI,
                ownershipWeight, ownershipChromaWeight,
                T.VET_OWNERSHIP_BRIGHT_START_IRE, T.VET_OWNERSHIP_BRIGHT_FULL_IRE,
                T.VET_OWNERSHIP_SAT_START_IRE,    T.VET_OWNERSHIP_SAT_FULL_IRE);
        };

        auto writePixelNoOwnership = [&](int x, double alphaEff) {
            const int h = left + x;
            const double coarseWorking = sourceToWorkingSample(coarseY, x);
            const double residualWorking = sourceToWorkingSample(highRawY, x);
            const double coherentWorking = sourceToWorkingSample(coherentCombY, x);

            const double yOut = coarseWorking + (residualWorking - alphaEff * coherentWorking);
            Y[h] = do3D ? getBestY(line, h, yOut, *prevFrameForVet, *nextFrameForVet)
                         : yOut;

            if (showMap) {
                w2d_frame_weight[line][x] = (float)alphaEff;
            }

            prodIRow[x] = (float)((alphaEff * tiAdjLocked[x]) * GI_PRODUCT);
            prodQRow[x] = (float)((alphaEff * tqAdjLocked[x]) * GQ_PRODUCT);
        };

        auto writePixelWithOwnership = [&](int x, double alphaVet) {
            const double alphaEff = alphaWithOwnership(x, alphaVet);
            const int h = left + x;
            const double coarseWorking = sourceToWorkingSample(coarseY, x);
            const double residualWorking = sourceToWorkingSample(highRawY, x);
            const double coherentWorking = sourceToWorkingSample(coherentCombY, x);

            const double yOut = coarseWorking + (residualWorking - alphaEff * coherentWorking);
            Y[h] = do3D ? getBestY(line, h, yOut, *prevFrameForVet, *nextFrameForVet)
                         : yOut;

            if (showMap) {
                w2d_frame_weight[line][x] = (float)alphaEff;
            }

            prodIRow[x] = (float)((alphaEff * tiAdjLocked[x]) * GI_PRODUCT);
            prodQRow[x] = (float)((alphaEff * tqAdjLocked[x]) * GQ_PRODUCT);
        };

        auto computeAlphaVet = [&](int p) -> double {
            const double r0 = sourceToWorkingSample(highRawY, p + 0);
            const double r1 = sourceToWorkingSample(highRawY, p + 1);
            const double r2 = sourceToWorkingSample(highRawY, p + 2);
            const double r3 = sourceToWorkingSample(highRawY, p + 3);

            const double c0 = sourceToWorkingSample(coherentCombY, p + 0);
            const double c1 = sourceToWorkingSample(coherentCombY, p + 1);
            const double c2 = sourceToWorkingSample(coherentCombY, p + 2);
            const double c3 = sourceToWorkingSample(coherentCombY, p + 3);

            const double rawI = r1 - r3;
            const double rawQ = r2 - r0;
            const double subI = c1 - c3;
            const double subQ = c2 - c0;

            const double subEnergy = subI * subI + subQ * subQ;
            const double subMagIRE = std::sqrt(subEnergy) * invI;

            if (!chromaLikeEnabled || subMagIRE < MIN_SUB_CHROMA_IRE) {
                return 1.0;
            }

            const double alphaFit = std::clamp(
                (rawI * subI + rawQ * subQ) / (subEnergy + 1e-12),
                MIN_ALPHA, MAX_ALPHA);

            const double conf = std::clamp(
                0.25 * (vetConf[p + 0] +
                        vetConf[p + 1] +
                        vetConf[p + 2] +
                        vetConf[p + 3]),
                0.0, 1.0);

            const double profileWeight = chromaLikeWeight * (1.0 - conf);
            return 1.0 + profileWeight * (alphaFit - 1.0);
        };

        if (width < 4) {
            if (ownershipEnabled) {
                for (int x = 0; x < width; ++x) {
                    writePixelWithOwnership(x, 1.0);
                }
            } else {
                for (int x = 0; x < width; ++x) {
                    writePixelNoOwnership(x, 1.0);
                }
            }
            continue;
        }

        const int tailStart = (width / 4) * 4;

        if (ownershipEnabled) {
            for (int p = 0; p + 3 < width; p += 4) {
                const double alphaVet = computeAlphaVet(p);
                writePixelWithOwnership(p + 0, alphaVet);
                writePixelWithOwnership(p + 1, alphaVet);
                writePixelWithOwnership(p + 2, alphaVet);
                writePixelWithOwnership(p + 3, alphaVet);
            }

            if (tailStart < width) {
                const int p = std::max(0, width - 4);
                const double alphaVet = computeAlphaVet(p);
                for (int x = tailStart; x < width; ++x) {
                    writePixelWithOwnership(x, alphaVet);
                }
            }
        } else {
            for (int p = 0; p + 3 < width; p += 4) {
                const double alphaVet = computeAlphaVet(p);
                writePixelNoOwnership(p + 0, alphaVet);
                writePixelNoOwnership(p + 1, alphaVet);
                writePixelNoOwnership(p + 2, alphaVet);
                writePixelNoOwnership(p + 3, alphaVet);
            }

            if (tailStart < width) {
                const int p = std::max(0, width - 4);
                const double alphaVet = computeAlphaVet(p);
                for (int x = tailStart; x < width; ++x) {
                    writePixelNoOwnership(x, alphaVet);
                }
            }
        }
    }

}

// Build the carrier-retracted view and its derived products.
//
// Per-line pass:
//   1. Windowed LS carrier fit on (raw - lockedLumaBaseY4) → carrierFit_flat
//   2. raw - carrierFit → carrierRetracted_flat (flattened view)
//   3. Sliding 4-sample mean of flattened → flatFloor_flat (carrier-free
//      luma floor; the 4-sample mean cancels carrier-shaped residual at
//      color transitions, leaving only genuine DC alien-Y)
//
// Cross-line pass (after all per-line fits):
//   4. Line-to-line cancellation on carrierFit → combedCarrier_flat
//      Real chroma inverts between opposite-phase lines, alien-Y doesn't.
//      combedCarrier preserves chroma and rejects alien-Y.
//
// Called between phaseLocked() and split2D().  Only needs the burst phasor
// and lockedLumaBaseY4 (both from phaseLocked) — no comb output required.
void Comb::FrameBuffer::buildCarrierRetracted()
{
    carrierRetractedValid = false;

    if (!configuration.phaseCompensation)
        return;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;

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

    const auto &T    = configuration.tunables;
    const int fitWin = std::max(32, (T.SINFIT_WIN_SAMPLES / 4) * 4);
    const int halfWin = fitWin / 2;

    if ((int)scratch_preI.size()        < width) scratch_preI.resize(width, 0.0);
    if ((int)scratch_preQ.size()        < width) scratch_preQ.resize(width, 0.0);
    if ((int)scratch_fieldLine.size()   < width) scratch_fieldLine.resize(width, 0.0);
    if ((int)scratch_fieldBLine.size()  < width) scratch_fieldBLine.resize(width, 0.0);
    if ((int)scratch_lumaBaseY4.size()  < width) scratch_lumaBaseY4.resize(width, 0.0);
    if ((int)scratch_lateralLine.size() < width) scratch_lateralLine.resize(width, 0.0);

    double *rawWhole    = scratch_preI.data();
    double *coarseY     = scratch_preQ.data();
    double *carrierFit  = scratch_fieldLine.data();
    double *flattened   = scratch_fieldBLine.data();
    double *slideMean4  = scratch_lateralLine.data();

    // ---------------------------------------------------------------
    // Pass 1: per-line LS carrier fit, flattened, and flatFloor
    // ---------------------------------------------------------------
    for (int line = firstLine; line < lastLine; ++line) {
        float *fitRow       = carrierFit_flat.data()       + static_cast<size_t>(line) * demodWidth;
        float *retractedRow = carrierRetracted_flat.data()  + static_cast<size_t>(line) * demodWidth;
        float *floorRow     = flatFloor_flat.data()         + static_cast<size_t>(line) * demodWidth;

        const CombCarrierGrammar *grammar = carrierGrammarLine(line);
        const bool grammarLocked = grammar && grammar->grammarLocked;

        const quint16 *rawLine = rawbuffer.data() + line * videoParameters.fieldWidth;

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
        }

        if (!grammarLocked) {
            for (int xi = 0; xi < width; ++xi) {
                fitRow[xi]       = 0.0f;
                retractedRow[xi] = static_cast<float>(rawWhole[xi]);
                floorRow[xi]     = static_cast<float>(coarseY[xi]);
            }
            continue;
        }

        const double bcos = grammar->burstCos;
        const double bsin = grammar->burstSin;
        const double maxCarrierSamples =
            std::max(24.0, grammar->carrierScale * 3.0) * irescale;

        // Windowed LS carrier withdrawal.
        for (int xi = 0; xi < width; ++xi) {
            int a = xi - halfWin;
            int b = xi + halfWin - 1;
            if (a < 0) {
                b += -a;
                a = 0;
            }
            if (b >= width) {
                const int over = b - (width - 1);
                b    -= over;
                a    -= over;
                if (a < 0) a = 0;
            }

            double sII = 0.0, sIQ = 0.0, sQQ = 0.0;
            double sIY = 0.0, sQY = 0.0, wSum = 0.0;

            for (int k = a; k <= b; ++k) {
                const int    hk   = left + k;
                const double obs  = rawWhole[k] - coarseY[k];

                const double bI = remodLockedToShiftedComposite(
                    1.0, 0.0, hk, bcos, bsin, spLUT_locked, cpLUT_locked);
                const double bQ = remodLockedToShiftedComposite(
                    0.0, 1.0, hk, bcos, bsin, spLUT_locked, cpLUT_locked);

                const double dist = std::fabs(static_cast<double>(k - xi));
                const double w = 1.0 - 0.65 * std::min(
                    1.0, dist / std::max(1.0, static_cast<double>(halfWin)));

                sII  += w * bI * bI;
                sIQ  += w * bI * bQ;
                sQQ  += w * bQ * bQ;
                sIY  += w * bI * obs;
                sQY  += w * bQ * obs;
                wSum += w;
            }

            double fitI = 0.0, fitQ = 0.0;
            const double det = sII * sQQ - sIQ * sIQ;
            if (wSum > 1e-9 && std::fabs(det) > 1e-9) {
                const double inv = 1.0 / det;
                fitI = ( sQQ * sIY - sIQ * sQY) * inv;
                fitQ = (-sIQ * sIY + sII * sQY) * inv;
            }

            const int    h   = left + xi;
            const double bI0 = remodLockedToShiftedComposite(
                1.0, 0.0, h, bcos, bsin, spLUT_locked, cpLUT_locked);
            const double bQ0 = remodLockedToShiftedComposite(
                0.0, 1.0, h, bcos, bsin, spLUT_locked, cpLUT_locked);

            double c = fitI * bI0 + fitQ * bQ0;
            c = std::clamp(c, -maxCarrierSamples, maxCarrierSamples);

            carrierFit[xi]  = c;
            flattened[xi]   = rawWhole[xi] - c;
            fitRow[xi]      = static_cast<float>(c);
            retractedRow[xi] = static_cast<float>(flattened[xi]);
        }

        // flatFloor: sliding 4-sample mean of flattened.
        // A 4-sample mean at fsc cancels any carrier-shaped oscillation,
        // so color-transition residual is stripped and only genuine DC-offset
        // alien luma survives.
        if (width >= 4) {
            const int meanCount = width - 3;
            for (int s = 0; s < meanCount; ++s) {
                slideMean4[s] = 0.25 * (flattened[s]     + flattened[s + 1]
                                       + flattened[s + 2] + flattened[s + 3]);
            }

            for (int xi = 0; xi < width; ++xi) {
                const int sFirst = std::max(0, xi - 3);
                const int sLast  = std::min(xi, meanCount - 1);
                double sum = 0.0;
                int n = 0;
                for (int s = sFirst; s <= sLast; ++s) {
                    sum += slideMean4[s];
                    ++n;
                }
                floorRow[xi] = static_cast<float>(n > 0 ? sum / n : coarseY[xi]);
            }
        } else {
            for (int xi = 0; xi < width; ++xi)
                floorRow[xi] = static_cast<float>(coarseY[xi]);
        }
    }

    // ---------------------------------------------------------------
    // Pass 2: line-to-line cancellation on carrierFit → combedCarrier
    //
    // Between adjacent lines with opposite chroma phase:
    //   real chroma inverts  →  carrierFit[N] ≈ -carrierFit[N±1]
    //   alien luma persists  →  carrierFit[N] ≈ +carrierFit[N±1]
    //
    // Subtraction preserves chroma (doubled) and cancels alien-Y (zeroed).
    // We average the upward and downward neighbors when both are available.
    // ---------------------------------------------------------------
    for (int line = firstLine; line < lastLine; ++line) {
        const float *fitRow = carrierFit_flat.data()
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
            (lineBelow < lastLine)  ? carrierGrammarLine(lineBelow) : nullptr;

        const bool haveAbove = gAbove && gAbove->grammarLocked;
        const bool haveBelow = gBelow && gBelow->grammarLocked;

        const float *fitAbove = haveAbove
            ? (carrierFit_flat.data() + static_cast<size_t>(lineAbove) * demodWidth)
            : nullptr;
        const float *fitBelow = haveBelow
            ? (carrierFit_flat.data() + static_cast<size_t>(lineBelow) * demodWidth)
            : nullptr;

        // Scale by 0.5 so pure chroma emerges at carrier amplitude A
        // (the raw subtraction doubles it to 2A; this restores parity
        // with split1D's output scale).
        if (haveAbove && haveBelow) {
            for (int xi = 0; xi < width; ++xi)
                combRow[xi] = static_cast<float>(
                    0.5 * (fitRow[xi] - 0.5 * (fitAbove[xi] + fitBelow[xi])));
        } else if (haveAbove) {
            for (int xi = 0; xi < width; ++xi)
                combRow[xi] = static_cast<float>(
                    0.5 * (fitRow[xi] - fitAbove[xi]));
        } else if (haveBelow) {
            for (int xi = 0; xi < width; ++xi)
                combRow[xi] = static_cast<float>(
                    0.5 * (fitRow[xi] - fitBelow[xi]));
        } else {
            // No neighbors — can't cancel alien-Y.  Pass through the
            // per-line fit at half amplitude for scale consistency.
            for (int xi = 0; xi < width; ++xi)
                combRow[xi] = static_cast<float>(0.5 * fitRow[xi]);
        }
    }

    carrierRetractedValid = true;
}
