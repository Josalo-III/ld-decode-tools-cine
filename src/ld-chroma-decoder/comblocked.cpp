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
#include "cadencedefs.h"
#include "combmath.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <QtGlobal>

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
//   2) demodTI4fsc/TQ4fsc + locked1DSource_flat: the common 4fsc export derived
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

        double *ldsRow = locked1DSource_line(line);
        if (!ldsRow)
            continue;
        seedCombAttributionPerLine(line);

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
        const float *carrierFitRow = (carrierRetractedValid &&
            !carrierFit_flat.empty() && demodWidth == width)
            ? carrierFit_line(line) : nullptr;
        const float *retractedRow = (carrierRetractedValid &&
            !carrierRetracted_flat.empty() && demodWidth == width)
            ? carrierRetracted_line(line) : nullptr;
        const float *retractedAbove = (carrierRetractedValid &&
            !carrierRetracted_flat.empty() && demodWidth == width)
            ? carrierRetracted_line(line - 1) : nullptr;
        const float *retractedBelow = (carrierRetractedValid &&
            !carrierRetracted_flat.empty() && demodWidth == width)
            ? carrierRetracted_line(line + 1) : nullptr;
        AttributionEvidence *attrRow = attributionEvidence_line(line);
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

            if (attrRow)
            {
                AttributionEvidence &e = attrRow[xi];
                e.facts.bandpassFineIRE = fine;
                e.facts.bandpassMidIRE = mid;
                e.facts.bandpassCoarseIRE = coarse;
                e.facts.lumaExcursionIRE = intakeNyquistRiskIRE;
                e.facts.residualFitErrorIRE = residualFitErrorIRE;
                e.facts.lumaIncursionRiskIRE = lumaIncursionRiskIRE;
                e.facts.icebergAlienYFraction = icebergAlienYFraction;
                e.facts.locked1DChromaIRE = std::hypot(ti, tq) * invIreScale;

                // Luma-side checkerboard witness from the retracted view
                // (raw - asserted carrier).  This is not a chroma source; it
                // marks carrier-phase structure that survived into the luma
                // witness so attribution can treat it as suspect during final
                // subtraction/election.
                if (retractedRow) {
                    const double center = static_cast<double>(retractedRow[xi]);
                    const double left2 = static_cast<double>(
                        retractedRow[std::clamp(xi - 2, 0, width - 1)]);
                    const double right2 = static_cast<double>(
                        retractedRow[std::clamp(xi + 2, 0, width - 1)]);
                    const double horizontalAltIRE =
                        std::fabs(center - 0.5 * (left2 + right2)) * invIreScale;
                    double verticalAltIRE = 0.0;
                    if (retractedAbove && retractedBelow) {
                        verticalAltIRE = std::fabs(
                            center - 0.5 * (static_cast<double>(retractedAbove[xi])
                                          + static_cast<double>(retractedBelow[xi])))
                            * invIreScale;
                    }
                    const double checkerIRE = std::max(horizontalAltIRE, verticalAltIRE);
                    const double chromaIRE = std::max(1.0, std::hypot(ti, tq) * invIreScale);
                    e.facts.quarterCheckerboardRisk = std::clamp(
                        (checkerIRE - std::max(0.75, 0.20 * chromaIRE)) /
                        std::max(3.0, 0.35 * chromaIRE),
                        0.0, 1.0);
                }

                // Sine/cosine (I/Q lattice) carrier-residual sideband evidence.
                // The carrier withdrawal leftover (excursion - carrierFit) is
                // split by carrier sample axis over a small local window.  Real
                // sideband (envelope curvature the ±2 model can't follow) rides
                // the dominant axis and correlates with envelope curvature; a
                // luma/cross-color impostor does not.
                if (carrierFitRow && floorRow) {
                    const int W = 4;
                    auto clampIdx = [&](int k) { return std::clamp(k, 0, width - 1); };
                    auto resAt = [&](int k) -> double {
                        const int kk = clampIdx(k);
                        const double exc = static_cast<double>(rawLine[left + kk])
                                         - static_cast<double>(floorRow[kk]);
                        return exc - static_cast<double>(carrierFitRow[kk]);
                    };
                    auto envAt = [&](int k) -> double {
                        const int kk = clampIdx(k);
                        const int kn = clampIdx(kk + 1);
                        return std::hypot(static_cast<double>(carrierFitRow[kk]),
                                          static_cast<double>(carrierFitRow[kn]))
                               * invIreScale;
                    };

                    int nSin = 0, nCos = 0, nWin = 0;
                    double sumSin = 0.0, sumCos = 0.0;
                    double sr = 0.0, sc = 0.0, src = 0.0, srr = 0.0, scc = 0.0;
                    for (int k = xi - W; k <= xi + W; ++k) {
                        const int kk = clampIdx(k);
                        const double rabs = std::fabs(resAt(kk)) * invIreScale;
                        const int cls = carrierSampleClass(line, left + kk) & 3;
                        if (cls == 0 || cls == 2) { sumSin += rabs; ++nSin; }
                        else                       { sumCos += rabs; ++nCos; }
                        const double curv = std::fabs(envAt(kk - 1)
                                                      - 2.0 * envAt(kk)
                                                      + envAt(kk + 1));
                        sr += rabs;  sc += curv;  src += rabs * curv;
                        srr += rabs * rabs;  scc += curv * curv;  ++nWin;
                    }

                    const double sbSin = nSin ? sumSin / nSin : 0.0;
                    const double sbCos = nCos ? sumCos / nCos : 0.0;
                    double sbCoh = 0.0;
                    if (nWin > 1) {
                        const double cov = nWin * src - sr * sc;
                        const double vr  = nWin * srr - sr * sr;
                        const double vc  = nWin * scc - sc * sc;
                        const double den = std::sqrt(std::max(0.0, vr)
                                                     * std::max(0.0, vc));
                        sbCoh = (den > 1e-9) ? std::clamp(cov / den, 0.0, 1.0) : 0.0;
                    }

                    e.facts.sidebandSinResidualIRE = sbSin;
                    e.facts.sidebandCosResidualIRE = sbCos;
                    e.facts.sidebandAxisAsymmetry =
                        (sbSin + sbCos > 1e-9) ? (sbSin - sbCos) / (sbSin + sbCos) : 0.0;
                    e.facts.sidebandCurvatureCoherence = sbCoh;
                }
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

    }

    // Populate clpbuffer[0] from locked1DSource_flat so all downstream consumers
    // (produceY no-residual path, diagnostics, any future code) get the
    // upgraded carrier model output without needing to branch on locked path.
    if (configuration.phaseCompensation) {
        for (int line = first; line < last; ++line) {
            const double *src = locked1DSource_line(line);
            if (src)
            {
                double *dst = clpbuffer[0].pixel[line];
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
    const auto &T       = configuration.tunables;
    const int srcBuf    = std::clamp(static_cast<int>(configuration.dimensions) - 1, 0, 2);

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
    const auto &T       = configuration.tunables;

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

// Attribution alpha blend given pre-computed per-pixel scalars.
// Returns the effective alpha (0 = full chroma subtraction, 1 = no subtraction
// modified by attribution and saturation guards).
double computeAttributionAlpha(
    const lddecode::CombAttributionAssessment &a,
    double alphaVet,
    double lumaIRE,
    double chromaIRE,
    double attributionWeight,
    double attributionChromaWeight,
    double brightStartIRE, double brightFullIRE,
    double satStartIRE,    double satFullIRE)
{
    const double lc = std::clamp(a.lumaClaim,    0.0, 1.0);
    const double cc = std::clamp(a.chromaClaim,  0.0, 1.0);
    const double uc = std::clamp(a.uncertainClaim, 0.0, 1.0);

    const double chromaFrac = std::clamp(
        1.0 - lc + attributionChromaWeight * cc, 0.0, 1.0);

    const double brightT = std::clamp(
        (lumaIRE - brightStartIRE) /
        std::max(1e-9, brightFullIRE - brightStartIRE),
        0.0, 1.0);

    const double satT = std::clamp(
        (chromaIRE - satStartIRE) /
        std::max(1e-9, satFullIRE - satStartIRE),
        0.0, 1.0);

    // Strong chroma or explicit chroma attribution should block luma attribution
    // from reducing residual chroma subtraction. This is especially important
    // for saturated blue, where lumaIRE may not trip the brightness ramp.
    const double chromaBlock = std::clamp(cc + 0.65 * satT, 0.0, 1.0);
    const double support = lc * (1.0 - chromaBlock) * (1.0 - 0.5 * uc);

    // Saturated color is already a warning that carrier-band energy is probably
    // chroma-owned; do not require high luma for attribution reassignment to bow out.
    const double attributionProtect = std::clamp(
        std::max(satT, brightT * satT), 0.0, 1.0);

    const double blend = std::clamp(
        attributionWeight * support * (1.0 - attributionProtect), 0.0, 1.0);

    return alphaVet * (1.0 - blend) + chromaFrac * blend;
}

} // namespace

void Comb::FrameBuffer::ensureProduceYScratch(int width)
{
    if ((int)scratch_frameBCenter.size() < width) scratch_frameBCenter.resize(width, 0.0);
    if ((int)scratch_fieldBCenter.size() < width) scratch_fieldBCenter.resize(width, 0.0);
    if ((int)scratch_residualContested.size() < width) scratch_residualContested.resize(width, 0.0);
    if ((int)scratch_lineWorkB.size()    < width) scratch_lineWorkB.resize(width, 1.0);
    if ((int)scratch_lineWorkA.size()    < width) scratch_lineWorkA.resize(width, 0.0);
    if ((int)scratch_lineWorkC.size()    < width) scratch_lineWorkC.resize(width, 0.0);
    if ((int)scratch_lateralLine.size()  < width) scratch_lateralLine.resize(width, 0.0);
    if ((int)scratch_yhp.size()          < width) scratch_yhp.resize(width, 0.0);
    if ((int)scratch_yI.size()           < width) scratch_yI.resize(width, 0.0);
    if ((int)scratch_yQ.size()           < width) scratch_yQ.resize(width, 0.0);
    if ((int)scratch_hpI.size()          < width) scratch_hpI.resize(width, 0.0);
    if ((int)scratch_hpQ.size()          < width) scratch_hpQ.resize(width, 0.0);
    if ((int)scratch_hpY.size()          < width) scratch_hpY.resize(width, 0.0);
    if ((int)scratch_outMixed.size()     < width) scratch_outMixed.resize(width, 0.0);
    if ((int)scratch_lineWorkD.size()    < width) scratch_lineWorkD.resize(width, 0.0);
    if ((int)scratch_lumaBaseY4.size()   < width) scratch_lumaBaseY4.resize(width, 0.0);
}

void Comb::FrameBuffer::produceY()
{
    if (!configuration.phaseCompensation) return;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int fullWidth = videoParameters.fieldWidth;
    const int width     = right - left;
    if (width <= 0) return;

    const auto &T = configuration.tunables;
    const bool enableResidualY = T.VET_ENABLE_RESIDUAL_Y;
    const double invI = this->invIreScale;

    const int srcBuf = std::clamp((int)configuration.dimensions - 1, 0, 2);

    const double MIN_ALPHA = 0.75;
    const double MAX_ALPHA = 1.25;
    const double MIN_SUB_CHROMA_IRE = 2.0;

    const bool showMap = configuration.showMap;
    const bool chromaLikeEnabled = (T.VET_Y_CHROMA_LIKE_WEIGHT > 0.0);
    const double chromaLikeWeight = T.VET_Y_CHROMA_LIKE_WEIGHT;
    const bool do3D =
        (configuration.residualVideo3D && prevFrameForVet && nextFrameForVet);

    ensureProduceYScratch(width);

    qint64 retractedVetN = 0;
    qint64 retractedVetActiveN = 0;
    double retractedVetBlendSum = 0.0;
    double retractedVetBlendMax = 0.0;
    qint64 retractedVetStableProtectN = 0;
    qint64 retractedVetStableProtectActiveN = 0;
    double retractedVetStableProtectSum = 0.0;
    double retractedVetStableProtectMax = 0.0;

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

        double *cHat        = scratch_lineWorkB.data();
        double *tiAdjLocked = scratch_lineWorkA.data();
        double *tqAdjLocked = scratch_lineWorkC.data();
        double *vetConf     = scratch_lateralLine.data();
        double *retractedY  = scratch_lineWorkD.data();

        // Carrier estimate comes from the dimensionally-appropriate source.
        // For 1D the elected buffer is locked1DSource_flat, which has no inter-line
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

        const float *retractedRow =
            (T.VET_RETRACTED_Y_ENABLE &&
             carrierRetractedValid &&
             !carrierRetracted_flat.empty() &&
             demodWidth == width)
            ? carrierRetracted_line(line) : nullptr;

        if (retractedRow) {
            for (int x = 0; x < width; ++x)
                retractedY[x] = static_cast<double>(retractedRow[x]);
        }

        YSourceView retractedYView {
            "retractedY",
            YSourceNativeSpace::CompositeLuma,
            YSourceHomeOrientation::None,
            retractedRow ? retractedY : nullptr
        };

        const AttributionEvidence *attrRow =
            T.VET_ATTRIBUTION_ENABLE ? attributionEvidence_line(line) : nullptr;
        const bool attributionEnabled = (attrRow != nullptr);

        const double attributionWeight = T.VET_ATTRIBUTION_LUMA_WEIGHT;
        const double attributionChromaWeight = T.VET_ATTRIBUTION_CHROMA_WEIGHT;

        auto alphaWithAttribution = [&](int x, double alphaVet) -> double {
            return computeAttributionAlpha(
                attrRow[x].assessment, alphaVet,
                (baseY4[x] - videoParameters.black16bIre) * invI,
                std::hypot(tiAdjLocked[x], tqAdjLocked[x]) * invI,
                attributionWeight, attributionChromaWeight,
                T.VET_ATTRIBUTION_BRIGHT_START_IRE, T.VET_ATTRIBUTION_BRIGHT_FULL_IRE,
                T.VET_ATTRIBUTION_SAT_START_IRE,    T.VET_ATTRIBUTION_SAT_FULL_IRE);
        };

        auto smoothStep01 = [](double t) {
            t = std::clamp(t, 0.0, 1.0);
            return t * t * (3.0 - 2.0 * t);
        };

        auto currentYAt = [&](int x, double alphaEff) -> double {
            x = std::clamp(x, 0, width - 1);
            return sourceToWorkingSample(coarseY, x) +
                (sourceToWorkingSample(highRawY, x) -
                 alphaEff * sourceToWorkingSample(coherentCombY, x));
        };

        auto residualCandidateAt = [&](int x, double alphaEff) -> double {
            return currentYAt(x, alphaEff);
        };

        const bool haveCachedBaseY4 =
            lockedLumaCacheValid &&
            !lockedLumaBaseY4_flat.empty() &&
            demodWidth == width;
        const bool haveCombedCarrier =
            (srcBuf == 0 &&
             carrierRetractedValid &&
             !combedCarrier_flat.empty() &&
             demodWidth == width);
        const bool haveRetractedLines =
            (T.VET_RETRACTED_Y_ENABLE &&
             carrierRetractedValid &&
             !carrierRetracted_flat.empty() &&
             demodWidth == width);

        const float *lsGateRow = lsRefitGate_line(line);

        auto lineInActive = [&](int y) -> bool {
            return y >= firstLine && y < lastLine && y < demodLines;
        };

        auto canSampleLumaLine = [&](int y) -> bool {
            return lineInActive(y) && (y == line || haveCachedBaseY4);
        };

        auto carrierAtLine = [&](int y, int x) -> double {
            x = std::clamp(x, 0, width - 1);
            if (haveCombedCarrier)
                return static_cast<double>(combedCarrier_line(y)[x]);
            return clpbuffer[srcBuf].pixel[y][left + x];
        };

        auto residualCandidateAtLine = [&](int y, int x, double alphaEff) -> double {
            if (y == line)
                return residualCandidateAt(x, alphaEff);

            x = std::clamp(x, 0, width - 1);
            const double *baseRow = lockedLumaBaseY4_line(y);
            const double base = baseRow[x];
            const double high = static_cast<double>(rawbuffer[y * fullWidth + left + x]) - base;
            return base + (high - alphaEff * carrierAtLine(y, x));
        };

        auto currentSameLatticeAlt = [&](int x, double alphaEff) -> double {
            const int xm = std::clamp(x - 2, 0, width - 1);
            const int xp = std::clamp(x + 2, 0, width - 1);
            return std::fabs(currentYAt(x, alphaEff) -
                             0.5 * (currentYAt(xm, alphaEff) +
                                    currentYAt(xp, alphaEff)));
        };

        auto mean4CurrentY = [&](int x, double alphaEff) -> double {
            double sum = 0.0;
            for (int k = -1; k <= 2; ++k)
                sum += currentYAt(std::clamp(x + k, 0, width - 1), alphaEff);
            return 0.25 * sum;
        };

        auto mean4Source = [&](const YSourceView &source, int x) -> double {
            double sum = 0.0;
            for (int k = -1; k <= 2; ++k)
                sum += sourceToWorkingSample(source, std::clamp(x + k, 0, width - 1));
            return 0.25 * sum;
        };

        auto retractedSequencedCandidateAt = [&](int x, double alphaEff) -> double {
            const double residualY = residualCandidateAt(x, alphaEff);
            if (!retractedYView.samples)
                return residualY;

            const double residualFine = residualY - mean4CurrentY(x, alphaEff);
            const double retractedBase = mean4Source(retractedYView, x);
            return retractedBase + residualFine;
        };

        auto mean4ResidualCandidateAtLine = [&](int y, int x, double alphaEff) -> double {
            if (y == line)
                return mean4CurrentY(x, alphaEff);

            double sum = 0.0;
            for (int k = -1; k <= 2; ++k)
                sum += residualCandidateAtLine(y, std::clamp(x + k, 0, width - 1), alphaEff);
            return 0.25 * sum;
        };

        auto mean4RetractedAtLine = [&](int y, int x) -> double {
            if (y == line)
                return mean4Source(retractedYView, x);

            const float *row = carrierRetracted_line(y);
            double sum = 0.0;
            for (int k = -1; k <= 2; ++k)
                sum += static_cast<double>(row[std::clamp(x + k, 0, width - 1)]);
            return 0.25 * sum;
        };

        auto retractedSequencedCandidateAtLine = [&](int y, int x, double alphaEff) -> double {
            if (y == line)
                return retractedSequencedCandidateAt(x, alphaEff);

            const double residualY = residualCandidateAtLine(y, x, alphaEff);
            if (!haveRetractedLines)
                return residualY;

            const double residualFine =
                residualY - mean4ResidualCandidateAtLine(y, x, alphaEff);
            const double retractedBase = mean4RetractedAtLine(y, x);
            return retractedBase + residualFine;
        };

        auto sequencedSameLatticeAlt = [&](int x, double alphaEff) -> double {
            const int xm = std::clamp(x - 2, 0, width - 1);
            const int xp = std::clamp(x + 2, 0, width - 1);
            return std::fabs(retractedSequencedCandidateAt(x, alphaEff) -
                             0.5 * (retractedSequencedCandidateAt(xm, alphaEff) +
                                    retractedSequencedCandidateAt(xp, alphaEff)));
        };

        auto smartNeighborCandidateAt = [&](int x, double alphaEff) -> double {
            const double residualY = residualCandidateAt(x, alphaEff);
            if (!retractedYView.samples)
                return residualY;

            const double residualAltIRE = currentSameLatticeAlt(x, alphaEff) * invI;
            const double sequencedAltIRE = sequencedSameLatticeAlt(x, alphaEff) * invI;
            const double altAdvantageIRE = residualAltIRE - sequencedAltIRE;
            if (altAdvantageIRE > T.VET_RETRACTED_ALT_START_IRE)
                return retractedSequencedCandidateAt(x, alphaEff);
            return residualY;
        };

        auto sameLatticeAltAtLine = [&](int y, int x, double alphaEff, bool retracted) -> double {
            const int xm = std::clamp(x - 2, 0, width - 1);
            const int xp = std::clamp(x + 2, 0, width - 1);
            if (retracted) {
                return std::fabs(retractedSequencedCandidateAtLine(y, x, alphaEff) -
                                 0.5 * (retractedSequencedCandidateAtLine(y, xm, alphaEff) +
                                        retractedSequencedCandidateAtLine(y, xp, alphaEff)));
            }
            return std::fabs(residualCandidateAtLine(y, x, alphaEff) -
                             0.5 * (residualCandidateAtLine(y, xm, alphaEff) +
                                    residualCandidateAtLine(y, xp, alphaEff)));
        };

        auto smartNeighborCandidateAtLine = [&](int y, int x, double alphaEff) -> double {
            if (y == line)
                return smartNeighborCandidateAt(x, alphaEff);

            const double residualY = residualCandidateAtLine(y, x, alphaEff);
            if (!haveRetractedLines)
                return residualY;

            const double residualAltIRE =
                sameLatticeAltAtLine(y, x, alphaEff, false) * invI;
            const double sequencedAltIRE =
                sameLatticeAltAtLine(y, x, alphaEff, true) * invI;
            const double altAdvantageIRE = residualAltIRE - sequencedAltIRE;
            if (altAdvantageIRE > T.VET_RETRACTED_ALT_START_IRE)
                return retractedSequencedCandidateAtLine(y, x, alphaEff);
            return residualY;
        };

        auto neighborAnchorAt = [&](int x, double alphaEff) -> double {
            double sum = 0.0;
            int n = 0;

            auto add = [&](int y, int xx) {
                if (!canSampleLumaLine(y))
                    return;
                sum += smartNeighborCandidateAtLine(y, xx, alphaEff);
                ++n;
            };

            // Post-demod this is luma-only, so immediate lateral neighbors are
            // no longer composite-color suspects.
            if (x > 0)
                add(line, x - 1);
            if (x + 1 < width)
                add(line, x + 1);

            const bool frameVertical =
                carrierFrameVerticalAllowed(line) &&
                (cadenceId >= 0 || cadenceId == CADENCE_PROGRESSIVE);
            const int verticalStep = frameVertical ? 1 : 2;
            add(line - verticalStep, x);
            add(line + verticalStep, x);

            return n ? (sum / static_cast<double>(n))
                     : residualCandidateAt(x, alphaEff);
        };

        auto stableChromaProtectAt = [&](int x) -> double {
            if (!T.VET_RETRACTED_STABLE_CHROMA_PROTECT)
                return 0.0;

            const int run = std::max(4, T.VET_RETRACTED_STABLE_CHROMA_RUN);
            if (width < run)
                return 0.0;

            int first = x - (run / 2);
            first = std::clamp(first, 0, width - run);

            double meanI = 0.0;
            double meanQ = 0.0;
            for (int k = 0; k < run; ++k) {
                meanI += tiAdjLocked[first + k];
                meanQ += tqAdjLocked[first + k];
            }
            meanI /= static_cast<double>(run);
            meanQ /= static_cast<double>(run);

            const double meanMagIRE = std::hypot(meanI, meanQ) * invI;
            const double highGate = smoothStep01(
                (meanMagIRE - T.VET_RETRACTED_STABLE_CHROMA_START_IRE) /
                std::max(1e-9, T.VET_RETRACTED_STABLE_CHROMA_FULL_IRE -
                               T.VET_RETRACTED_STABLE_CHROMA_START_IRE));
            if (highGate <= 0.0)
                return 0.0;

            double maxDevIRE = 0.0;
            for (int k = 0; k < run; ++k) {
                const double dI = tiAdjLocked[first + k] - meanI;
                const double dQ = tqAdjLocked[first + k] - meanQ;
                maxDevIRE = std::max(maxDevIRE, std::hypot(dI, dQ) * invI);
            }

            const double consistentGate = 1.0 - std::clamp(
                maxDevIRE / std::max(1e-9, T.VET_RETRACTED_STABLE_CHROMA_DEV_IRE),
                0.0, 1.0);
            return highGate * consistentGate;
        };

        auto retractedBlendFor = [&](int x, double alphaEff, double chromaIRE) -> double {
            if (!retractedYView.samples || T.VET_RETRACTED_Y_WEIGHT <= 0.0)
                return 0.0;

            auto finishRetractedBlend = [&](double blend) {
                if (configuration.debugPhaseLegs) {
                    ++retractedVetN;
                    retractedVetBlendSum += blend;
                    retractedVetBlendMax = std::max(retractedVetBlendMax, blend);
                    if (blend > 1e-4)
                        ++retractedVetActiveN;
                }
                return blend;
            };

            const double currentAltIRE = currentSameLatticeAlt(x, alphaEff) * invI;
            const double retractedAltIRE = sequencedSameLatticeAlt(x, alphaEff) * invI;
            const double altAdvantageIRE = currentAltIRE - retractedAltIRE;
            const double altGate = smoothStep01(
                (altAdvantageIRE - T.VET_RETRACTED_ALT_START_IRE) /
                std::max(1e-9, T.VET_RETRACTED_ALT_FULL_IRE -
                               T.VET_RETRACTED_ALT_START_IRE));

            double anchorGate = 0.0;
            if (T.VET_NEIGHBOR_ANCHOR_ENABLE) {
                const double anchor = neighborAnchorAt(x, alphaEff);
                const double residualDistIRE =
                    std::fabs(residualCandidateAt(x, alphaEff) - anchor) * invI;
                const double retractedDistIRE =
                    std::fabs(retractedSequencedCandidateAt(x, alphaEff) - anchor) * invI;
                const double anchorAdvantageIRE = residualDistIRE - retractedDistIRE;
                anchorGate = smoothStep01(
                    (anchorAdvantageIRE - T.VET_NEIGHBOR_ANCHOR_START_IRE) /
                    std::max(1e-9, T.VET_NEIGHBOR_ANCHOR_FULL_IRE -
                                   T.VET_NEIGHBOR_ANCHOR_START_IRE));
                anchorGate *= std::clamp(T.VET_NEIGHBOR_ANCHOR_WEIGHT, 0.0, 1.0);
            }

            // LS refit gate: when buildCarrierRetracted detected a luma
            // edge where the LS sinusoidal fit disagrees with the
            // complement, the retracted view at this sample is built
            // from a structurally superior carrier model.  Inject that
            // as candidate support so the vet lets the retracted view
            // dominate.  Attribution and chroma protection still apply
            // downstream — this only opens the candidateGate, it
            // doesn't bypass the safety checks.
            const double lsGate = (lsGateRow && x >= 0 && x < width)
                ? static_cast<double>(lsGateRow[x]) : 0.0;

            const double candidateGate = std::max({altGate, anchorGate, lsGate});
            if (candidateGate <= 0.0)
                return finishRetractedBlend(0.0);

            double attributionInvite = 0.0;
            double chromaProtect = 0.0;
            if (attrRow) {
                const auto &a = attrRow[x].assessment;
                attributionInvite = std::clamp(
                    0.60 * a.lumaClaim + 0.40 * a.checkerboardRisk -
                    0.50 * a.chromaClaim,
                    0.0, 1.0);
                chromaProtect = std::clamp(a.chromaClaim, 0.0, 1.0);
            }

            const double satT = std::clamp(
                (chromaIRE - T.VET_ATTRIBUTION_SAT_START_IRE) /
                std::max(1e-9, T.VET_ATTRIBUTION_SAT_FULL_IRE -
                               T.VET_ATTRIBUTION_SAT_START_IRE),
                0.0, 1.0);
            chromaProtect = std::max(chromaProtect, 0.35 * satT);
            const double stableProtect = stableChromaProtectAt(x);
            if (configuration.debugPhaseLegs) {
                ++retractedVetStableProtectN;
                retractedVetStableProtectSum += stableProtect;
                retractedVetStableProtectMax = std::max(
                    retractedVetStableProtectMax, stableProtect);
                if (stableProtect > 1e-4)
                    ++retractedVetStableProtectActiveN;
            }
            chromaProtect = std::max(chromaProtect, stableProtect);

            if (attributionInvite <= 0.0)
                return finishRetractedBlend(0.0);

            const double invite = attributionInvite;
            const double protect = 1.0 - 0.75 * chromaProtect;
            const double blend = std::clamp(
                T.VET_RETRACTED_Y_WEIGHT * candidateGate * invite * protect,
                0.0, 1.0);
            return finishRetractedBlend(blend);
        };

        auto writePixelNoAttribution = [&](int x, double alphaEff) {
            const int h = left + x;
            const double coarseWorking = sourceToWorkingSample(coarseY, x);
            const double residualWorking = sourceToWorkingSample(highRawY, x);
            const double coherentWorking = sourceToWorkingSample(coherentCombY, x);
            const double chromaIRE = std::hypot(tiAdjLocked[x], tqAdjLocked[x]) * invI;
            const double retBlend = retractedBlendFor(x, alphaEff, chromaIRE);

            const double residualY = coarseWorking + (residualWorking - alphaEff * coherentWorking);
            double retractedThenResidualY = residualY;
            if (retBlend > 0.0) {
                const double residualFine = residualY - mean4CurrentY(x, alphaEff);
                const double retractedBase = mean4Source(retractedYView, x);
                retractedThenResidualY = retractedBase + residualFine;
            }
            const double yOut = residualY * (1.0 - retBlend) +
                retractedThenResidualY * retBlend;
            Y[h] = do3D ? getBestY(line, h, yOut, *prevFrameForVet, *nextFrameForVet)
                         : yOut;

            if (showMap) {
                w2d_frame_weight[line][x] = (float)alphaEff;
            }

            const int ph = carrierSampleClass(line, h);
            const double finalCarrier = static_cast<double>(rawLine[h]) - yOut;
            const double prodI = finalCarrier * lutTi[ph];
            const double prodQ = finalCarrier * lutTq[ph];
            prodIRow[x] = (float)(prodI * GI_PRODUCT);
            prodQRow[x] = (float)(prodQ * GQ_PRODUCT);
        };

        auto writePixelWithAttribution = [&](int x, double alphaVet) {
            const double alphaEff = alphaWithAttribution(x, alphaVet);
            const int h = left + x;
            const double coarseWorking = sourceToWorkingSample(coarseY, x);
            const double residualWorking = sourceToWorkingSample(highRawY, x);
            const double coherentWorking = sourceToWorkingSample(coherentCombY, x);
            const double chromaIRE = std::hypot(tiAdjLocked[x], tqAdjLocked[x]) * invI;
            const double retBlend = retractedBlendFor(x, alphaEff, chromaIRE);

            const double residualY = coarseWorking + (residualWorking - alphaEff * coherentWorking);
            double retractedThenResidualY = residualY;
            if (retBlend > 0.0) {
                const double residualFine = residualY - mean4CurrentY(x, alphaEff);
                const double retractedBase = mean4Source(retractedYView, x);
                retractedThenResidualY = retractedBase + residualFine;
            }
            const double yOut = residualY * (1.0 - retBlend) +
                retractedThenResidualY * retBlend;
            Y[h] = do3D ? getBestY(line, h, yOut, *prevFrameForVet, *nextFrameForVet)
                         : yOut;

            if (showMap) {
                w2d_frame_weight[line][x] = (float)alphaEff;
            }

            const int ph = carrierSampleClass(line, h);
            const double finalCarrier = static_cast<double>(rawLine[h]) - yOut;
            const double prodI = finalCarrier * lutTi[ph];
            const double prodQ = finalCarrier * lutTq[ph];
            prodIRow[x] = (float)(prodI * GI_PRODUCT);
            prodQRow[x] = (float)(prodQ * GQ_PRODUCT);
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
            if (attributionEnabled) {
                for (int x = 0; x < width; ++x) {
                    writePixelWithAttribution(x, 1.0);
                }
            } else {
                for (int x = 0; x < width; ++x) {
                    writePixelNoAttribution(x, 1.0);
                }
            }
            continue;
        }

        const int tailStart = (width / 4) * 4;

        if (attributionEnabled) {
            for (int p = 0; p + 3 < width; p += 4) {
                const double alphaVet = computeAlphaVet(p);
                writePixelWithAttribution(p + 0, alphaVet);
                writePixelWithAttribution(p + 1, alphaVet);
                writePixelWithAttribution(p + 2, alphaVet);
                writePixelWithAttribution(p + 3, alphaVet);
            }

            if (tailStart < width) {
                const int p = std::max(0, width - 4);
                const double alphaVet = computeAlphaVet(p);
                for (int x = tailStart; x < width; ++x) {
                    writePixelWithAttribution(x, alphaVet);
                }
            }
        } else {
            for (int p = 0; p + 3 < width; p += 4) {
                const double alphaVet = computeAlphaVet(p);
                writePixelNoAttribution(p + 0, alphaVet);
                writePixelNoAttribution(p + 1, alphaVet);
                writePixelNoAttribution(p + 2, alphaVet);
                writePixelNoAttribution(p + 3, alphaVet);
            }

            if (tailStart < width) {
                const int p = std::max(0, width - 4);
                const double alphaVet = computeAlphaVet(p);
                for (int x = tailStart; x < width; ++x) {
                    writePixelNoAttribution(x, alphaVet);
                }
            }
        }
    }

    if (configuration.debugPhaseLegs && retractedVetN > 0) {
        qInfo("RetractedYVet n=%lld active=%lld activeFrac=%.4f avgBlend=%.4f maxBlend=%.4f "
              "stableProtectActive=%lld stableProtectFrac=%.4f avgStableProtect=%.4f maxStableProtect=%.4f",
              (long long)retractedVetN,
              (long long)retractedVetActiveN,
              static_cast<double>(retractedVetActiveN) /
                  std::max(1.0, static_cast<double>(retractedVetN)),
              retractedVetBlendSum / static_cast<double>(retractedVetN),
              retractedVetBlendMax,
              (long long)retractedVetStableProtectActiveN,
              static_cast<double>(retractedVetStableProtectActiveN) /
                  std::max(1.0, static_cast<double>(retractedVetStableProtectN)),
              retractedVetStableProtectSum /
                  std::max(1.0, static_cast<double>(retractedVetStableProtectN)),
              retractedVetStableProtectMax);
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
    const auto &T       = configuration.tunables;

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
    if (lsRefitGate_flat.size() < need)
        lsRefitGate_flat.assign(need, 0.0f);

    if ((int)scratch_preI.size()        < width) scratch_preI.resize(width, 0.0);
    if ((int)scratch_preQ.size()        < width) scratch_preQ.resize(width, 0.0);
    if ((int)scratch_lineWorkA.size()   < width) scratch_lineWorkA.resize(width, 0.0);
    if ((int)scratch_lineWorkC.size()   < width) scratch_lineWorkC.resize(width, 0.0);
    if ((int)scratch_lumaBaseY4.size()  < width) scratch_lumaBaseY4.resize(width, 0.0);
    if ((int)scratch_lateralLine.size() < width) scratch_lateralLine.resize(width, 0.0);

    double *rawWhole    = scratch_preI.data();
    double *coarseY     = scratch_preQ.data();
    double *carrierFit  = scratch_lineWorkA.data();
    double *flattened   = scratch_lineWorkC.data();
    double *slideMean4  = scratch_lateralLine.data();

    // Carrier-symmetry witness (debug).  The complement cancellation rests on
    // the carrier's +/- excursions being symmetric.  A polarity-gain asymmetry
    // (nonlinear clipping) or a per-bucket gain difference leaks a saturation-
    // proportional 2fsc residual.  We stratify obs = raw - coarseY (the carrier
    // excursion, IRE) by local chroma amplitude so we can see whether the
    // asymmetry scales with saturation.
    const bool measureSym = configuration.debugPhaseLegs;
    struct SymBin {
        double sumAbsBucket[4] = {0.0, 0.0, 0.0, 0.0};
        qint64 nBucket[4]      = {0, 0, 0, 0};
        double sumPos = 0.0, sumNeg = 0.0;
        qint64 nPos = 0, nNeg = 0;
    };
    static constexpr double kSymEdges[3] = { 8.0, 16.0, 28.0 }; // IRE bin edges
    SymBin symBins[4];

    // ---------------------------------------------------------------
    // Pass 1: per-line carrier withdrawal (complement cancellation),
    // flattened view, and flatFloor.
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

        const double maxCarrierSamples =
            std::max(24.0, grammar->carrierScale * 5.0) * irescale;

        // Carrier neutralization via complement cancellation with envelope-
        // curvature correction.
        //
        // The ±2-sample complement extracts the carrier assuming a locally
        // flat chroma envelope.  Where the envelope varies (color onsets,
        // saturation contours), the flat-envelope assumption leaves a
        // residual proportional to the envelope's second difference:
        //
        //   error[h] = 0.25 * (A[h-2] - 2*A[h] + A[h+2]) * sign[h]
        //
        // where A[h] is the unsigned carrier amplitude on the same lattice
        // (samples 2 apart share a carrier phase axis).  We estimate A from
        // the complement output, compute the curvature, and add the
        // correction back.  This lets the model track the envelope without
        // imposing a bandwidth cutoff — the sidebands are modeled
        // structurally rather than filtered.
        double *eCorr = slideMean4;  // scratch reuse
        for (int xi = 0; xi < width; ++xi) {
            const int b = carrierSampleClass(line, left + xi) & 3;
            eCorr[xi] = CARRIER_BUCKET_GAIN[b] * (rawWhole[xi] - coarseY[xi]);
        }

        // Step 1: standard complement → carrierFit (uncorrected).
        for (int xi = 0; xi < width; ++xi) {
            const double e  = eCorr[xi];
            const double eM = eCorr[std::max(0, xi - 2)];
            const double eP = eCorr[std::min(width - 1, xi + 2)];
            carrierFit[xi] = 0.5 * (e - 0.5 * (eM + eP));
        }

        // Step 2: envelope-curvature correction.
        // The same-lattice envelope is |carrierFit| at ±2 positions (same
        // carrier phase axis).  The curvature is its second difference.
        // The correction is 0.25 * curvature * sign(carrierFit[xi]).
        for (int xi = 0; xi < width; ++xi) {
            const double c  = carrierFit[xi];
            const int m2 = std::max(0, xi - 2);
            const int p2 = std::min(width - 1, xi + 2);
            const double envC  = std::fabs(carrierFit[xi]);
            const double envM2 = std::fabs(carrierFit[m2]);
            const double envP2 = std::fabs(carrierFit[p2]);

            const double curvature = envM2 - 2.0 * envC + envP2;
            const double sign = (c >= 0.0) ? 1.0 : -1.0;

            eCorr[xi] = c + 0.25 * curvature * sign;
        }

        // Step 3: clamp, write the corrected carrier and the retracted view.
        for (int xi = 0; xi < width; ++xi) {
            double cf = std::clamp(eCorr[xi], -maxCarrierSamples, maxCarrierSamples);

            carrierFit[xi]   = cf;
            flattened[xi]    = rawWhole[xi] - cf;
            fitRow[xi]       = static_cast<float>(cf);
            retractedRow[xi] = static_cast<float>(flattened[xi]);
        }

        // LS carrier refit at luma edges.
        //
        // The complement cancellation cannot structurally separate
        // alien-Y from carrier.  Where a luma edge is present AND a
        // windowed LS sinusoidal fit disagrees with the complement, the
        // complement has absorbed alien-Y as if it were carrier.  The LS
        // fit projects onto explicit carrier basis functions, naturally
        // rejecting non-sinusoidal alien-Y.
        //
        // Two conditions must both be met:
        //   1. coarseY gradient (luma edge present)
        //   2. LS fit disagrees with complement (the edge actually
        //      contaminated the complement — if they agree, no gain)
        //
        // The resulting gate is stored in lsRefitGate_flat so
        // retractedBlendFor() can ensure the retracted view dominates
        // at these samples.  Without that coupling, the bandpass-
        // clipped Y wins the vet and the refit is wasted.
        {
            constexpr double EDGE_SOFT_IRE  = 3.0;
            constexpr double EDGE_HARD_IRE  = 10.0;
            constexpr double DISC_SOFT_IRE  = 1.0;
            constexpr double DISC_HARD_IRE  = 4.0;
            constexpr int    LS_HALF_WIN    = 16;
            const double bcos = grammar->burstCos;
            const double bsin = grammar->burstSin;
            auto smoothStep01 = [](double t) {
                t = std::clamp(t, 0.0, 1.0);
                return t * t * (3.0 - 2.0 * t);
            };

            float *gateRow = lsRefitGate_flat.data()
                             + static_cast<size_t>(line) * demodWidth;

            // eCorr is free after Step 3; reuse as the edge pre-gate.
            double *edgeGate = eCorr;
            bool anyEdge = false;
            for (int xi = 0; xi < width; ++xi) {
                const int xm = std::max(0, xi - 2);
                const int xp = std::min(width - 1, xi + 2);
                const double gradIRE =
                    std::fabs(coarseY[xp] - coarseY[xm]) * invIreScale;
                const double gate = std::clamp(
                    (gradIRE - EDGE_SOFT_IRE) /
                    std::max(1e-9, EDGE_HARD_IRE - EDGE_SOFT_IRE),
                    0.0, 1.0);
                edgeGate[xi] = gate;
                if (gate > 0.0) anyEdge = true;
                gateRow[xi] = 0.0f;
            }

            if (anyEdge) {
                for (int xi = 0; xi < width; ++xi) {
                    if (edgeGate[xi] <= 0.0) continue;

                    int a = std::max(0, xi - LS_HALF_WIN);
                    int b = std::min(width - 1, xi + LS_HALF_WIN - 1);

                    double sII = 0.0, sIQ = 0.0, sQQ = 0.0;
                    double sIY = 0.0, sQY = 0.0;

                    for (int k = a; k <= b; ++k) {
                        const int hk = left + k;
                        const double obs = rawWhole[k] - coarseY[k];
                        const double bI = remodLockedToShiftedComposite(
                            1.0, 0.0, hk, bcos, bsin,
                            spLUT_locked, cpLUT_locked);
                        const double bQ = remodLockedToShiftedComposite(
                            0.0, 1.0, hk, bcos, bsin,
                            spLUT_locked, cpLUT_locked);
                        const double dist = std::fabs(
                            static_cast<double>(k - xi));
                        const double w = 1.0 - 0.65 * std::min(
                            1.0, dist / std::max(1.0,
                                static_cast<double>(LS_HALF_WIN)));
                        sII += w * bI * bI;
                        sIQ += w * bI * bQ;
                        sQQ += w * bQ * bQ;
                        sIY += w * bI * obs;
                        sQY += w * bQ * obs;
                    }

                    double fitI = 0.0, fitQ = 0.0;
                    const double det = sII * sQQ - sIQ * sIQ;
                    if (std::fabs(det) > 1e-9) {
                        const double inv = 1.0 / det;
                        fitI = ( sQQ * sIY - sIQ * sQY) * inv;
                        fitQ = (-sIQ * sIY + sII * sQY) * inv;
                    }

                    const int h = left + xi;
                    const double bI0 = remodLockedToShiftedComposite(
                        1.0, 0.0, h, bcos, bsin,
                        spLUT_locked, cpLUT_locked);
                    const double bQ0 = remodLockedToShiftedComposite(
                        0.0, 1.0, h, bcos, bsin,
                        spLUT_locked, cpLUT_locked);

                    double lsFit = fitI * bI0 + fitQ * bQ0;
                    lsFit = std::clamp(lsFit,
                        -maxCarrierSamples, maxCarrierSamples);

                    const double discrepancyIRE =
                        std::fabs(lsFit - carrierFit[xi]) * invIreScale;
                    const double discGate = std::clamp(
                        (discrepancyIRE - DISC_SOFT_IRE) /
                        std::max(1e-9, DISC_HARD_IRE - DISC_SOFT_IRE),
                        0.0, 1.0);

                    double brightColorProtect = 0.0;
                    if (T.LS_REFIT_BRIGHT_COLOR_GUARD) {
                        const int xm = std::max(0, xi - 2);
                        const int xp = std::min(width - 1, xi + 2);
                        const double lumaM = coarseY[xm];
                        const double lumaP = coarseY[xp];
                        const double dir = (lumaP >= lumaM) ? 1.0 : -1.0;
                        const double lumaMid = 0.5 * (lumaM + lumaP);
                        const double brightOffsetIRE =
                            dir * (coarseY[xi] - lumaMid) * invIreScale;
                        const double brightSideGate = smoothStep01(
                            (brightOffsetIRE - T.LS_REFIT_BRIGHT_SIDE_SOFT_IRE) /
                            std::max(1e-9, T.LS_REFIT_BRIGHT_SIDE_HARD_IRE -
                                           T.LS_REFIT_BRIGHT_SIDE_SOFT_IRE));

                        const int brightIdx = (dir > 0.0) ? xp : xm;
                        const int brightJ = (brightIdx + 1 < width)
                            ? (brightIdx + 1)
                            : (brightIdx > 0 ? brightIdx - 1 : brightIdx);
                        const int xiJ = (xi + 1 < width) ? (xi + 1) : (xi > 0 ? xi - 1 : xi);
                        const double brightAmpIRE =
                            std::hypot(carrierFit[brightIdx], carrierFit[brightJ]) * invIreScale;
                        const double localAmpIRE =
                            std::hypot(carrierFit[xi], carrierFit[xiJ]) * invIreScale;
                        const double coloredBrightIRE = std::max(brightAmpIRE, localAmpIRE);
                        const double brightColorGate = smoothStep01(
                            (coloredBrightIRE - T.LS_REFIT_BRIGHT_COLOR_START_IRE) /
                            std::max(1e-9, T.LS_REFIT_BRIGHT_COLOR_FULL_IRE -
                                           T.LS_REFIT_BRIGHT_COLOR_START_IRE));

                        brightColorProtect = brightSideGate * brightColorGate;
                    }

                    const double g =
                        edgeGate[xi] * discGate * (1.0 - brightColorProtect);
                    gateRow[xi] = static_cast<float>(g);

                    if (g > 0.0) {
                        const double blended =
                            carrierFit[xi] * (1.0 - g) + lsFit * g;
                        carrierFit[xi]   = blended;
                        fitRow[xi]       = static_cast<float>(blended);
                        retractedRow[xi] = static_cast<float>(
                            rawWhole[xi] - blended);
                    }
                }
            }
        }

        // flatFloor: coarseY is the DC luma floor for this line.
        // (Previously derived from a 4-sample mean of flattened; now that
        // flattened has no coarseY baseline, coarseY directly is cleaner.)
        for (int xi = 0; xi < width; ++xi)
            floorRow[xi] = static_cast<float>(coarseY[xi]);

        if (measureSym) {
            for (int xi = 0; xi < width; ++xi) {
                // Local chroma amplitude from orthogonal carrier samples:
                // carrierFit[xi], carrierFit[xi+1] are ~90 deg apart at 4fsc,
                // so hypot ~= the chroma envelope magnitude.
                const int j = (xi + 1 < width) ? xi + 1 : (xi > 0 ? xi - 1 : xi);
                const double ampIRE = std::hypot(carrierFit[xi], carrierFit[j]) * invIreScale;
                // Measure the raw carrier excursion (raw - coarseY) so the
                // per-bucket asymmetry reflects the input, not the reconstructed
                // carrier (eCorr has been repurposed by the complement passes).
                const double obsIRE = (rawWhole[xi] - coarseY[xi]) * invIreScale;

                int bin = 0;
                while (bin < 3 && ampIRE >= kSymEdges[bin]) ++bin;
                SymBin &b = symBins[bin];

                const int bucket = carrierSampleClass(line, left + xi) & 3;
                b.sumAbsBucket[bucket] += std::fabs(obsIRE);
                b.nBucket[bucket]      += 1;

                if (obsIRE >= 0.0) { b.sumPos += obsIRE;  ++b.nPos; }
                else               { b.sumNeg += -obsIRE; ++b.nNeg; }
            }
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
    auto softReachGate = [](double diffIRE, double softIRE, double hardIRE) {
        if (diffIRE <= softIRE)
            return 1.0;
        if (diffIRE >= hardIRE)
            return 0.0;
        const double t = (diffIRE - softIRE) /
                         std::max(1e-9, hardIRE - softIRE);
        return 1.0 - (t * t * (3.0 - 2.0 * t));
    };

    for (int line = firstLine; line < lastLine; ++line) {
        const float *fitRow = carrierFit_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        const float *floorRow = flatFloor_flat.data()
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

        // Only opposite-polarity neighbors can cancel alien-Y while preserving
        // chroma.  Same-polarity lines make chroma and same-signed alien-Y
        // indistinguishable in this scalar domain; subtracting them cancels the
        // chroma witness itself.
        const bool haveAbove = gAbove && gAbove->grammarLocked &&
            (gAbove->lineFlip != grammar->lineFlip);
        const bool haveBelow = gBelow && gBelow->grammarLocked &&
            (gBelow->lineFlip != grammar->lineFlip);

        const float *fitAbove = haveAbove
            ? (carrierFit_flat.data() + static_cast<size_t>(lineAbove) * demodWidth)
            : nullptr;
        const float *fitBelow = haveBelow
            ? (carrierFit_flat.data() + static_cast<size_t>(lineBelow) * demodWidth)
            : nullptr;
        const float *floorAbove = haveAbove
            ? (flatFloor_flat.data() + static_cast<size_t>(lineAbove) * demodWidth)
            : nullptr;
        const float *floorBelow = haveBelow
            ? (flatFloor_flat.data() + static_cast<size_t>(lineBelow) * demodWidth)
            : nullptr;

        auto reachGate = [&](int xi, const float *neighborFit, const float *neighborFloor) {
            if (!neighborFit || !neighborFloor)
                return 0.0;

            const double lumaDiffIRE =
                std::fabs(static_cast<double>(floorRow[xi])
                        - static_cast<double>(neighborFloor[xi])) * invIreScale;
            const double lumaGate = softReachGate(lumaDiffIRE, 3.0, 10.0);

            // Opposite-polarity chroma should satisfy fitRow ~= -neighborFit.
            // A large same-sample mismatch means the reach crosses a color
            // boundary, which is where the 1D complement leaves alternation.
            const double centerFit = static_cast<double>(fitRow[xi]);
            const double neighbor = static_cast<double>(neighborFit[xi]);
            const double carrierMismatchIRE =
                std::fabs(centerFit + neighbor) * invIreScale;
            const double carrierAmpIRE =
                0.5 * (std::fabs(centerFit) + std::fabs(neighbor)) * invIreScale;
            const double carrierSoftIRE = std::max(3.0, 0.25 * carrierAmpIRE);
            const double carrierHardIRE = std::max(10.0, 0.80 * carrierAmpIRE);
            const double carrierGate = softReachGate(
                carrierMismatchIRE, carrierSoftIRE, carrierHardIRE);

            return lumaGate * carrierGate;
        };

        // Scale by 0.5 so pure chroma emerges at carrier amplitude A
        // (the raw subtraction doubles it to 2A; this restores parity
        // with split1D's output scale).
        for (int xi = 0; xi < width; ++xi) {
            const double wAbove = haveAbove ? reachGate(xi, fitAbove, floorAbove) : 0.0;
            const double wBelow = haveBelow ? reachGate(xi, fitBelow, floorBelow) : 0.0;
            const double wSum = wAbove + wBelow;

            if (wSum > 1e-9) {
                const double neighborFit =
                    ((wAbove * static_cast<double>(fitAbove[xi])) +
                     (wBelow * static_cast<double>(fitBelow[xi]))) / wSum;
                combRow[xi] = static_cast<float>(
                    0.5 * (static_cast<double>(fitRow[xi]) - neighborFit));
            } else {
                // No safe opposite-polarity reach here. Keep the per-line fit
                // at full scale so chroma amplitude stays consistent with
                // split1D and produceY subtraction.
                combRow[xi] = fitRow[xi];
            }
        }
    }

    if (measureSym) {
        qInfo("CarrierSymmetry (obs=raw-coarseY IRE, binned by local chroma amp IRE):");
        qInfo("  ampBin          n   meanPos  meanNeg  polAsym     m0    m2  asym02     m1    m3  asym13");
        const char *names[4] = { "[0,8)", "[8,16)", "[16,28)", "[28,+inf)" };
        for (int i = 0; i < 4; ++i) {
            const SymBin &b = symBins[i];
            const qint64 n = b.nPos + b.nNeg;
            if (n == 0) continue;
            const double mPos = b.nPos ? b.sumPos / (double)b.nPos : 0.0;
            const double mNeg = b.nNeg ? b.sumNeg / (double)b.nNeg : 0.0;
            const double polAsym = (mPos + mNeg > 1e-9) ? (mPos - mNeg) / (mPos + mNeg) : 0.0;
            double m[4];
            for (int k = 0; k < 4; ++k)
                m[k] = b.nBucket[k] ? b.sumAbsBucket[k] / (double)b.nBucket[k] : 0.0;
            const double a02 = (m[0] + m[2] > 1e-9) ? (m[0] - m[2]) / (m[0] + m[2]) : 0.0;
            const double a13 = (m[1] + m[3] > 1e-9) ? (m[1] - m[3]) / (m[1] + m[3]) : 0.0;
            qInfo("  %-9s %9lld  %7.3f  %7.3f  %+6.3f   %5.2f %5.2f %+6.3f   %5.2f %5.2f %+6.3f",
                  names[i], (long long)n, mPos, mNeg, polAsym,
                  m[0], m[2], a02, m[1], m[3], a13);
        }

        // Patch dump: find the 16×8 region with the strongest periodic
        // (checkerboard) alternation in the retracted view, RESTRICTED to
        // luma-flat regions so the search finds the flat-field checker rather
        // than edge ringing.  Score uses a sign-alternating sum: sum of
        // (-1)^dx * ret[dx]; a checkerboard reinforces, a monotonic edge
        // cancels.  Flatness is gated on flatFloor (the stored coarseY): a
        // patch is rejected if its horizontal or vertical luma gradient over
        // the smooth floor exceeds GRAD_MAX_IRE.
        const int patchW = 16, patchH = 8;
        const double GRAD_MAX_IRE = 4.0;
        int bestPx = -1, bestPy = -1;
        double bestAlt = 0.0;
        for (int py = firstLine; py <= lastLine - patchH; py += 4) {
            for (int px = 0; px <= width - patchW; px += 4) {
                // Flatness gate on the smooth luma floor.
                double maxGrad = 0.0;
                for (int dy = 0; dy < patchH; ++dy) {
                    const float *f = flatFloor_flat.data()
                                     + static_cast<size_t>(py + dy) * demodWidth;
                    const float *fUp = flatFloor_flat.data()
                                     + static_cast<size_t>(py + dy + 1) * demodWidth;
                    for (int dx = 0; dx < patchW - 1; ++dx) {
                        const double gx = std::fabs(f[px + dx + 1] - f[px + dx]) * invIreScale;
                        const double gy = (dy + 1 < patchH)
                            ? std::fabs(fUp[px + dx] - f[px + dx]) * invIreScale : 0.0;
                        maxGrad = std::max(maxGrad, std::max(gx, gy));
                    }
                }
                if (maxGrad > GRAD_MAX_IRE) continue;

                double alt = 0.0;
                for (int dy = 0; dy < patchH; ++dy) {
                    const float *r = carrierRetracted_flat.data()
                                     + static_cast<size_t>(py + dy) * demodWidth;
                    double rowAlt = 0.0;
                    for (int dx = 0; dx < patchW; ++dx)
                        rowAlt += ((dx & 1) ? -1.0 : 1.0) * r[px + dx];
                    alt += std::fabs(rowAlt);
                }
                if (alt > bestAlt) {
                    bestAlt = alt;
                    bestPx = px;
                    bestPy = py;
                }
            }
        }
        if (bestPx >= 0) {
            qInfo("PatchDump (max 2fsc alternation) at (%d,%d), %dx%d, IRE:",
                  bestPx + left, bestPy, patchW, patchH);
            for (int dy = 0; dy < patchH; ++dy) {
                const float *retRow = carrierRetracted_flat.data()
                                      + static_cast<size_t>(bestPy + dy) * demodWidth;
                QString row;
                for (int dx = 0; dx < patchW; ++dx) {
                    double ire = retRow[bestPx + dx] * invIreScale;
                    row += QString::asprintf(" %+7.2f", ire);
                }
                qInfo("  line %3d: %s", bestPy + dy, qPrintable(row));
            }
            qInfo("PatchDump: carrierFit at same location:");
            for (int dy = 0; dy < patchH; ++dy) {
                const float *fitR = carrierFit_flat.data()
                                    + static_cast<size_t>(bestPy + dy) * demodWidth;
                QString row;
                for (int dx = 0; dx < patchW; ++dx) {
                    double ire = fitR[bestPx + dx] * invIreScale;
                    row += QString::asprintf(" %+7.2f", ire);
                }
                qInfo("  line %3d: %s", bestPy + dy, qPrintable(row));
            }
            qInfo("PatchDump: raw-coarseY (excursion) at same location:");
            for (int dy = 0; dy < patchH; ++dy) {
                const quint16 *rawLine = rawbuffer.data()
                    + static_cast<size_t>(bestPy + dy) * videoParameters.fieldWidth;
                const float *floorR = flatFloor_flat.data()
                                      + static_cast<size_t>(bestPy + dy) * demodWidth;
                QString row;
                for (int dx = 0; dx < patchW; ++dx) {
                    double raw = static_cast<double>(rawLine[left + bestPx + dx]);
                    double floor = static_cast<double>(floorR[bestPx + dx]);
                    double ire = (raw - floor) * invIreScale;
                    row += QString::asprintf(" %+7.2f", ire);
                }
                qInfo("  line %3d: %s", bestPy + dy, qPrintable(row));
            }
            // Per-line summary: where does the line-to-line alternation enter?
            // Also compare: split1D (raw bandpass) vs carrier retraction.
            // split1D_Y = raw - split1D = 0.5*raw + 0.25*(raw[-2]+raw[+2])
            // retracted = raw - complement(raw - coarseY)
            // difference = retracted - split1D_Y = complement(coarseY)
            qInfo("PatchDump: per-line means (cols 4-15 of patch, IRE):");
            qInfo("  line  phase  rawMean  coarseY  excMean  fitMean  retMean  split1dY  delta");
            for (int dy = 0; dy < patchH; ++dy) {
                const int ln = bestPy + dy;
                const quint16 *rawLn = rawbuffer.data()
                    + static_cast<size_t>(ln) * videoParameters.fieldWidth;
                const float *flrLn = flatFloor_flat.data()
                    + static_cast<size_t>(ln) * demodWidth;
                const float *fitLn = carrierFit_flat.data()
                    + static_cast<size_t>(ln) * demodWidth;
                const float *retLn = carrierRetracted_flat.data()
                    + static_cast<size_t>(ln) * demodWidth;

                double sRaw = 0, sCY = 0, sExc = 0, sFit = 0, sRet = 0, sS1d = 0;
                const int c0 = 4, c1 = 16;
                for (int dx = c0; dx < c1; ++dx) {
                    const int ax = left + bestPx + dx;
                    const double r  = static_cast<double>(rawLn[ax]);
                    const double rm = static_cast<double>(rawLn[std::max(left, ax - 2)]);
                    const double rp = static_cast<double>(rawLn[std::min(right - 1, ax + 2)]);
                    const double cy = static_cast<double>(flrLn[bestPx + dx]);
                    const double ft = static_cast<double>(fitLn[bestPx + dx]);
                    const double rt = static_cast<double>(retLn[bestPx + dx]);
                    // split1D Y = raw - bandpass = raw - 0.5*(raw - 0.5*(raw[-2]+raw[+2]))
                    //           = 0.5*raw + 0.25*(raw[-2] + raw[+2])
                    const double s1y = 0.5 * r + 0.25 * (rm + rp);
                    sRaw += r; sCY += cy; sExc += (r - cy); sFit += ft; sRet += rt; sS1d += s1y;
                }
                const int n = c1 - c0;
                const int phase = carrierLineFlip(ln);
                const double retM = sRet / n * invIreScale;
                const double s1dM = sS1d / n * invIreScale;
                qInfo("  %4d  %+2d   %7.2f  %7.2f  %+7.2f  %+7.2f  %7.2f  %7.2f  %+6.2f",
                      ln, phase,
                      sRaw / n * invIreScale,
                      sCY / n * invIreScale,
                      sExc / n * invIreScale,
                      sFit / n * invIreScale,
                      retM, s1dM, retM - s1dM);
            }
        }
    }

    carrierRetractedValid = true;
}
