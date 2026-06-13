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

// Locked-path pre-processing: burst detection, luma-base caching, and raw
// composite demodulation into TRI/TRQ. The per-line affine is solved later in
// buildPhaseCorrected1D(), after the locked carrier source has been prepared.
//
// Locked-path pre-processing.  buildPhaseCorrected1D now sources from
// combedCarrier (the carrier model after line-to-line cancellation),
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
        grammar.phaseError = 0.0;
        grammar.phaseScheduleConflict = 0.0;
        grammar.affine.valid = false;

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

}

// Demodulates the combed carrier (from buildCarrierRetracted) into two
// explicit products:
//   1) demodTI/TQ: line-local locked IQ after burst alignment and affine trim.
//   2) demodTI4fsc/TQ4fsc + locked1DSource_flat: the common 4fsc export derived
//      from that locked IQ, used as the cross-line scalar reference for 2D work.
//
// The combed carrier is the carrier model after line-to-line cancellation —
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

    // Source: the complementary carrier from buildConstrainedYWitness() when
    // available, otherwise the combed carrier from buildCarrierRetracted().
    //
    // Compact-patch transfer is the exception. It belongs to the Y witness,
    // but must not author the evidence used to elect Field/Frame cleanup.
    // At those marked pixels, locked1DSource uses the independent combed
    // carrier so the election can judge and remove any 1D colour alternation.
    const bool haveCombedCarrier =
        carrierRetractedValid &&
        !combedCarrier_flat.empty() &&
        demodWidth == width;

    const bool haveCarrierWitnessCorrection =
        witnessValid &&
        !carrierWitness_flat.empty() &&
        demodWidth == width;

    if ((int)scratch_sinfit_mag.size() < width) scratch_sinfit_mag.resize(width, 0.0);
    if ((int)scratch_sinfit_resmag.size() < width) scratch_sinfit_resmag.resize(width, 0.0);
    // Solve the line affine here, after buildCarrierRetracted(), so downstream
    // clients still see one published affine after the locked carrier source is
    // prepared. Do not use carrierRetracted_flat as a luma referee here: it is
    // derived from the carrier fit and can contain the same color checker error.
    {
        const int WIN  = std::max(4, (T.SINFIT_WIN_SAMPLES / 4) * 4);
        const int HALF = WIN / 2;
        const bool writeAffine = configuration.residualVideo && T.Y_LINE_AFFINE_TRIM_ENABLE;
        constexpr double PHASE_ERROR_CAP = M_PI / 8.0;

        for (int line = first; line < last; ++line) {
            CombCarrierGrammar *grammar = carrierGrammarLine(line);
            if (grammar) {
                grammar->affine.valid = false;
                grammar->phaseError = 0.0;
                grammar->phaseScheduleConflict = 0.0;
            }
            if (!grammar || !grammar->grammarLocked)
                continue;

            const quint16 *rawLine = rawbuffer.data() + line * videoParameters.fieldWidth;
            const double bcos = grammar->burstCos;
            const double bsin = grammar->burstSin;
            const float *triRow = demodTRI_line(line);
            const float *trqRow = demodTRQ_line(line);

            double *magRow = scratch_sinfit_mag.data();
            double *resRow = scratch_sinfit_resmag.data();

            for (int k = 0; k < width; ++k) {
                const int hk = left + k;
                const double rik = static_cast<double>(triRow[k]);
                const double rqk = static_cast<double>(trqRow[k]);
                const double mag_k = std::hypot(rik, rqk);
                magRow[k] = mag_k;

                if (mag_k > 1e-9) {
                    const double fitted_k = remodLockedToShiftedComposite(
                        rik, rqk, hk, bcos, bsin, spLUT_locked, cpLUT_locked);
                    const double corr_k = static_cast<double>(rawLine[hk]) - fitted_k;
                    double rsk = 0.0, rck = 0.0;
                    demod4fscFromComposite(corr_k, hk, rsk, rck);
                    resRow[k] = std::hypot(rsk, rck);
                } else {
                    resRow[k] = 0.0;
                }
            }

            const int winN = (width <= WIN) ? width : WIN;
            int a = 0;
            int b = winN - 1;
            double sumAmp = 0.0, sumRes = 0.0, sumI = 0.0, sumQ = 0.0;
            auto addWin = [&](int k) {
                sumAmp += magRow[k];
                sumRes += resRow[k];
                sumI += static_cast<double>(triRow[k]);
                sumQ += static_cast<double>(trqRow[k]);
            };
            auto subWin = [&](int k) {
                sumAmp -= magRow[k];
                sumRes -= resRow[k];
                sumI -= static_cast<double>(triRow[k]);
                sumQ -= static_cast<double>(trqRow[k]);
            };
            for (int k = a; k <= b; ++k)
                addWin(k);

            double STT[2][2] = {{0,0},{0,0}};
            double SRT[2][2] = {{0,0},{0,0}};

            for (int xi = 0; xi < width; ++xi) {
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
                    while (a < aWant) subWin(a++);
                    while (a > aWant) addWin(--a);
                    while (b < bWant) addWin(++b);
                    while (b > bWant) subWin(b--);
                }

                const double ri = static_cast<double>(triRow[xi]);
                const double rq = static_cast<double>(trqRow[xi]);
                const double ampEst = sumAmp / static_cast<double>(winN);
                const double resAmp = sumRes / static_cast<double>(winN);
                const double meanI = sumI / static_cast<double>(winN);
                const double meanQ = sumQ / static_cast<double>(winN);
                const double meanMag = std::hypot(meanI, meanQ);
                const double coherence = (ampEst > 1e-9)
                    ? std::clamp(meanMag / ampEst, 0.0, 1.0)
                    : 0.0;

                const double mag0 = std::hypot(ri, rq);
                double localFitI = ri;
                double localFitQ = rq;
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
                const double qualityWeight = (0.25 + 0.75 * coherence) /
                    (1.0 + vetNorm * vetNorm);

                if (qualityWeight > 1e-6) {
                    STT[0][0] += qualityWeight * fI*fI; STT[0][1] += qualityWeight * fI*fQ;
                    STT[1][0] += qualityWeight * fI*fQ; STT[1][1] += qualityWeight * fQ*fQ;
                    SRT[0][0] += qualityWeight * ri*fI; SRT[0][1] += qualityWeight * ri*fQ;
                    SRT[1][0] += qualityWeight * rq*fI; SRT[1][1] += qualityWeight * rq*fQ;
                }
            }

            LineAffine &la = grammar->affine;
            double STTinv[2][2];
            if (!mat2_inv(STT, STTinv))
                continue;

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
            grammar->phaseScheduleConflict = std::clamp(
                std::fabs(measuredPhase) / (M_PI / 4.0), 0.0, 1.0);
            if (grammar->phaseScheduleConflict < 0.1)
                grammar->lineFlipAuthority =
                    lddecode::CarrierPhaseAuthority::BurstMeasured;

            if (writeAffine) {
                const double pMax = T.Y_LINE_MAX_PHASE_DEG * M_PI / 180.0;
                clamp_rotation_gain_shear(Rm, U, pMax,
                                          T.Y_LINE_ALLOW_GAIN_ON_IQ,
                                          T.Y_LINE_GAIN_MIN, T.Y_LINE_GAIN_MAX,
                                          T.Y_LINE_MAX_SHEAR);
                la.R[0][0]=Rm[0][0]; la.R[0][1]=Rm[0][1];
                la.R[1][0]=Rm[1][0]; la.R[1][1]=Rm[1][1];
                la.valid = true;
            } else if (T.Y_LINE_PHASE_ERROR_LUT_ENABLE &&
                       grammar->phaseConfidence >= T.Y_LINE_PHASE_ERROR_MIN_CONF)
            {
                const double phase = grammar->phaseError;
                if (std::isfinite(phase) && std::fabs(phase) >= 1e-12) {
                    const double c = std::cos(phase);
                    const double s = std::sin(phase);
                    for (int i = 0; i < 4; ++i) {
                        const double ti = static_cast<double>(grammar->demodLUTTi[i]);
                        const double tq = static_cast<double>(grammar->demodLUTTq[i]);
                        grammar->demodLUTTi[i] = static_cast<float>(c * ti - s * tq);
                        grammar->demodLUTTq[i] = static_cast<float>(s * ti + c * tq);
                    }
                }
            }
        }
    }

    std::vector<double> envLine;
    for (int line = first; line < last; ++line) {
        const quint16 *rawLine = rawbuffer.data() + line * videoParameters.fieldWidth;

        /*
         * Carrier source contract:
         *
         *   carrierWitnessRow:
         *       The witness carrier, preferred in its line-cancelled form
         *       (carrierWitnessCombed: opposite-lineFlip neighbors rejected the
         *       luma pedestal the per-pixel 1D witness leaves at compact-patch
         *       geometry).  Either form is a composite-grid residual in the
         *       same horizontal sample coordinates as rawbuffer.
         *
         *   combSrc:
         *       line-to-line carrier model from buildCarrierRetracted().  This is
         *       already a carrier-shaped model after opposite-line cancellation.
         *
         * These are not the same source type.  The selected source must carry an
         * explicit sync contract so demod/remod does not accidentally apply the
         * wrong phase/sign convention.
         */
        enum class LockedCarrierSourceKind {
            None,
            WitnessCompositeResidual,
            CombedCarrierModel
        };

        const float *combSrc = haveCombedCarrier
            ? combedCarrier_line(line)
            : nullptr;

        const float *witnessCarrierRow = haveCarrierWitnessCorrection
            ? (carrierWitnessCombed_flat.empty()
                ? carrierWitness_line(line)
                : carrierWitnessCombed_line(line))
            : nullptr;
        const float *correctionMaskRow = haveCarrierWitnessCorrection
            ? carrierCorrectionMask_line(line)
            : nullptr;

        LockedCarrierSourceKind carrierSourceKind = LockedCarrierSourceKind::None;

        if (witnessCarrierRow) {
            carrierSourceKind = LockedCarrierSourceKind::WitnessCompositeResidual;
        } else if (combSrc) {
            carrierSourceKind = LockedCarrierSourceKind::CombedCarrierModel;
        }

        auto carrierSourceAt = [&](int rel)->double {
            rel = std::clamp(rel, 0, width - 1);

            // A compact transfer repairs yWitness only. Feeding its carrier
            // complement into the candidate family lets the 1D decision
            // certify itself through Field/Frame agreement.
            if (correctionMaskRow &&
                correctionMaskRow[rel] > 0.5f &&
                combSrc)
            {
                return static_cast<double>(combSrc[rel]);
            }

            switch (carrierSourceKind) {
            case LockedCarrierSourceKind::WitnessCompositeResidual:
                return static_cast<double>(witnessCarrierRow[rel]);

            case LockedCarrierSourceKind::CombedCarrierModel:
                return static_cast<double>(combSrc[rel]);

            case LockedCarrierSourceKind::None:
            default:
                return 0.0;
            }
        };

        const float *carrierModelRow =
            (carrierSourceKind == LockedCarrierSourceKind::CombedCarrierModel)
                ? combSrc
                : ((carrierRetractedValid && !carrierFit_flat.empty() && demodWidth == width)
                    ? carrierFit_line(line)
                    : nullptr);

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
        const float *retractedRow = (carrierRetractedValid &&
            !carrierRetracted_flat.empty() && demodWidth == width)
            ? carrierRetracted_line(line) : nullptr;
        const float *retractedAbove = (carrierRetractedValid &&
            !carrierRetracted_flat.empty() && demodWidth == width)
            ? carrierRetracted_line(line - 1) : nullptr;
        const float *retractedBelow = (carrierRetractedValid &&
            !carrierRetracted_flat.empty() && demodWidth == width)
            ? carrierRetracted_line(line + 1) : nullptr;
        const lddecode::FourViewCarrierAttribution *parallaxRow =
            (carrierRetractedValid &&
             !carrierParallax_flat.empty() && demodWidth == width)
            ? (carrierParallax_flat.data() + static_cast<size_t>(line) * demodWidth)
            : nullptr;
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
                                                nullptr,
                                                nullptr,
                                                scratch_lumaSmooth.data());
            lumaSmooth = scratch_lumaSmooth.data();
        }

        double *tiBase = scratch_preI.data();
        double *tqBase = scratch_preQ.data();

        /*
         * Demod the selected carrier source through the locked line grammar.
         *
         * A witness residual is raw-composite residual energy at absolute sample
         * h.  It must enter through the same carrier grammar that parsed the raw
         * composite line.  Do not treat it as a drop-in combedCarrier sample with
         * an implicit local phase convention.
         */
        for (int xi = 0; xi < width; ++xi) {
            const int h = left + xi;
            const double cv = carrierSourceAt(xi);

            double ti = 0.0;
            double tq = 0.0;

            lddecode::CarrierGrammarDemodCoefficients dc;
            if (lddecode::carrierGrammarLockedDemodCoefficients(grammar, h, dc)) {
                ti = cv * dc.ti;
                tq = cv * dc.tq;
            } else {
                const int ph = carrierSampleClass(line, h);
                ti = cv * lutTi[ph];
                tq = cv * lutTq[ph];
            }

            lddecode::carrierGrammarApplyAffine(grammar, ti, tq);

            tiBase[xi] = ti;
            tqBase[xi] = tq;
        }

        // Sample accessor for bandpass-scale metrics on the combed carrier.
        auto sampleComb = [&](int rel)->double {
            return carrierSourceAt(rel);
        };

        // Precompute per-sample envelope for the sideband block below.
        // envLine[xi] = hypot(carrierModel[xi], carrierModel[xi+1]) * invIreScale.
        // Replaces per-pixel envAt() hypot calls: ~27 hypots/pixel → 0 in the window loop.
        const double *envLinePtr = nullptr;
        if (carrierModelRow && floorRow) {
            if ((int)envLine.size() < width)
                envLine.resize(width);
            for (int xi = 0; xi < width; ++xi)
                envLine[xi] = std::hypot(
                    carrierSourceAt(xi),
                    carrierSourceAt(std::min(xi + 1, width - 1)))
                    * invIreScale;
            envLinePtr = envLine.data();
        }

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

                if (parallaxRow && parallaxRow[xi].valid) {
                    const auto &p = parallaxRow[xi];
                    e.facts.carrierResidualConsensus = p.residualConsensus;
                    e.facts.carrierParallaxYSpreadIRE = p.ySpreadIRE;
                    e.facts.carrierParallaxYCurvatureIRE = p.yCurvatureIRE;
                    e.facts.carrierParallaxSpreadIRE = p.carrierSpreadIRE;
                    e.facts.carrierParallaxCoherence = p.carrierCoherence;
                    e.facts.carrierParallaxLatticeRiskIRE = p.latticeRiskIRE;
                    e.facts.carrierResidualIRE = p.movingResidualFitErrorIRE;
                    e.facts.carrierChromaFitIRE =
                        std::fabs(p.movingResidualSample) * invIreScale;
                    e.facts.carrierChromaErrorIRE = p.movingResidualFitErrorIRE;
                    e.facts.carrierChromaCoherence = p.movingResidualCoherence;
                    e.facts.movingResidualSample = p.movingResidualSample;
                    e.facts.movingResidualFitErrorIRE = p.movingResidualFitErrorIRE;
                    e.facts.movingResidualCoherence = p.movingResidualCoherence;
                    e.facts.movingResidualPull = p.movingResidualPull;
                }

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
                if (carrierModelRow && floorRow) {
                    const int W = 4;
                    auto clampIdx = [&](int k) { return std::clamp(k, 0, width - 1); };
                    auto resAt = [&](int k) -> double {
                        const int kk = clampIdx(k);
                        const double exc = static_cast<double>(rawLine[left + kk])
                                         - static_cast<double>(floorRow[kk]);
                        return exc - carrierSourceAt(kk);
                    };
                    auto envAt = [&](int k) -> double {
                        return envLinePtr[clampIdx(k)];
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

            /*
             * Publish the locked 1D scalar back through the line grammar.
             *
             * At this point the selected source has been converted to the common
             * 4fSC IQ frame.  The remod side must use the grammar samplePhase0
             * and declared sign frame instead of the generic shifted-composite
             * helper that assumes the older intake path.
             */
            ldsRow[xi] = lddecode::carrierGrammarRemod4fscToComposite(
                grammar,
                h,
                ti4,
                tq4,
                1.0,
                lddecode::CarrierSignFrame::Grid4fsc);

            if (baseY4 && grammarLocked) {
                const double residual = (double)rawLine[h] - baseY4[xi];
                const double rI = residual * lutTi[ph];
                const double rQ = residual * lutTq[ph];
                double rI4, rQ4;
                lockedTo4fsc(rI, rQ, bcos, bsin, rI4, rQ4);
                const double cModel = lddecode::carrierGrammarRemod4fscToComposite(
                    grammar,
                    h,
                    rI4,
                    rQ4,
                    1.0,
                    lddecode::CarrierSignFrame::Grid4fsc);
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

    // Precompute per-pixel IQ magnitude from the 4fsc demod output.
    // buildCombTapLine fills iqMag via std::hypot for every tap layer it
    // processes — up to 7 layers per line, each recomputing the same
    // magnitude from the same (ti, tq) source data.  This table turns
    // all of those into a single indexed read per pixel per layer.
    {
        const size_t need = static_cast<size_t>(last) * static_cast<size_t>(demodWidth);
        if (demodIQMag4fsc_flat.size() < need)
            demodIQMag4fsc_flat.resize(need);
        for (int line = first; line < last; ++line) {
            const float *ti = demodTI4fsc_line(line);
            const float *tq = demodTQ4fsc_line(line);
            float *mag = demodIQMag4fsc_flat.data()
                         + static_cast<size_t>(line) * demodWidth;
            for (int xi = 0; xi < width; ++xi)
                mag[xi] = static_cast<float>(
                    std::hypot(static_cast<double>(ti[xi]),
                               static_cast<double>(tq[xi])));
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
    if ((int)scratch_lumaBaseY4.size() < width) scratch_lumaBaseY4.resize(width, 0.0);
    if ((int)scratch_lumaHiRaw.size() < width) scratch_lumaHiRaw.resize(width, 0.0);
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
    const bool instrumentProduceY = configuration.stageTimers;
    if (instrumentProduceY) {
        produceYInstrumentation.reset();
    }

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
    const double impuritySuppressionWeight =
        std::clamp(T.CC_SUPPRESSION_WEIGHT, 0.0, 1.0);
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
            if (instrumentProduceY) {
                produceYInstrumentation.pixels += width;
                produceYInstrumentation.residualBypassPixels += width;
            }
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
            if (instrumentProduceY) {
                produceYInstrumentation.pixels += width;
                produceYInstrumentation.noProductFallbackPixels += width;
            }
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

        double *baseY4 = scratch_lumaBaseY4.data();
        double *hiRaw  = scratch_lumaHiRaw.data();

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
        double *witnessY  = scratch_lineWorkD.data();

        // Carrier estimate comes from the dimensionally-appropriate source.
        // For 1D the elected buffer is locked1DSource_flat, which has no inter-line
        // cancellation: luma-near-fsc rides on it and subtracting it from raw
        // pulls a phase-alternating error into Y (the alien-chroma checkerboard).
        // combedCarrier from buildCarrierRetracted is the carrier model after
        // line-to-line cancellation — real chroma is preserved, alien-Y is
        // zeroed. 2D/3D paths already carry cancellation in their own comb
        // election, so clpLine stays correct there.
        // In locked 1D the witness pair IS the model: yWitness is the
        // attribution result, and carrierWitnessCombed = raw - yWitness after
        // interline pedestal rejection is its carrier complement.  produceY
        // publishes that model; it does not hedge between models per pixel —
        // per-pixel authority switching between the witness and the fit-based
        // combedCarrier put a moving seam through every region where the two
        // disagreed.  combedCarrier remains only as the no-witness fallback.
        // HF-luma protection is structural, not hedged: the witness comb
        // returns non-inverting (luma) error to Y, so Y = yWitness + the
        // pedestal the cancellation rejected from the carrier.
        //
        // 2D/3D keep their elected candidate: the election refines the same
        // witness-derived source, it is not a second model family.
        const float *witnessCombedRow = (srcBuf == 0)
            ? carrierWitnessCombed_line(line) : nullptr;
        const float *combedCarrierRow =
            (srcBuf == 0 && !witnessCombedRow && carrierRetractedValid)
            ? combedCarrier_line(line) : nullptr;

        for (int x = 0; x < width; ++x) {
            const int h = left + x;
            const int ph = carrierSampleClass(line, h);
            const double carrier = witnessCombedRow
                ? static_cast<double>(witnessCombedRow[x])
                : (combedCarrierRow
                    ? static_cast<double>(combedCarrierRow[x])
                    : clpLine[h]);
            cHat[x]        = carrier;
            tiAdjLocked[x] = carrier * lutTi[ph];
            tqAdjLocked[x] = carrier * lutTq[ph];
        }

        YSourceView coherentCombY {
            "coherentCombY",
            YSourceNativeSpace::BandpassComb,
            YSourceHomeOrientation::GrammarComposite,
            cHat
        };

        const float *witnessRow =
            (T.VET_WITNESS_Y_ENABLE && witnessValid)
            ? yWitness_line(line) : nullptr;
        const float *witnessConfRow =
            (T.VET_WITNESS_Y_ENABLE && witnessValid)
            ? yWitnessConfidence_line(line) : nullptr;
        const float *witnessObsRow =
            (T.VET_WITNESS_Y_ENABLE && witnessValid)
            ? obstructionRisk_line(line) : nullptr;

        if (witnessRow) {
            for (int x = 0; x < width; ++x)
                witnessY[x] = static_cast<double>(witnessRow[x]);
        }

        YSourceView witnessYView {
            "witnessY",
            YSourceNativeSpace::CompositeLuma,
            YSourceHomeOrientation::None,
            witnessRow ? witnessY : nullptr
        };

        const AttributionEvidence *attrRow =
            T.VET_ATTRIBUTION_ENABLE ? attributionEvidence_line(line) : nullptr;
        const bool attributionEnabled = (attrRow != nullptr);

        // Impurity suppression must stay tied to the wide-window detector.
        // Witness obstruction also includes ambiguity/conflict diagnostics that
        // are not safe to use as direct per-pixel subtraction strength.
        const float *impurityRow = carrierImpurity_line(line);

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
            T.VET_WITNESS_Y_ENABLE && witnessValid;

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
            if (!witnessYView.samples)
                return residualY;

            (void)alphaEff;
            return sourceToWorkingSample(witnessYView,
                                         std::clamp(x, 0, width - 1));
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
                return mean4Source(witnessYView, x);

            const float *row = yWitness_line(y);
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

            (void)alphaEff;
            const float *row = yWitness_line(y);
            if (!row)
                return residualY;
            return static_cast<double>(row[std::clamp(x, 0, width - 1)]);
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
            if (!witnessYView.samples)
                return residualY;

            const double residualAltIRE = currentSameLatticeAlt(x, alphaEff) * invI;
            const double sequencedAltIRE = sequencedSameLatticeAlt(x, alphaEff) * invI;
            const double altAdvantageIRE = residualAltIRE - sequencedAltIRE;
            if (altAdvantageIRE > T.VET_WITNESS_ALT_START_IRE)
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
            if (altAdvantageIRE > T.VET_WITNESS_ALT_START_IRE)
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
            if (!T.VET_WITNESS_STABLE_CHROMA_PROTECT)
                return 0.0;

            const int run = std::max(4, T.VET_WITNESS_STABLE_CHROMA_RUN);
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
                (meanMagIRE - T.VET_WITNESS_STABLE_CHROMA_START_IRE) /
                std::max(1e-9, T.VET_WITNESS_STABLE_CHROMA_FULL_IRE -
                               T.VET_WITNESS_STABLE_CHROMA_START_IRE));
            if (highGate <= 0.0)
                return 0.0;

            double maxDevIRE = 0.0;
            for (int k = 0; k < run; ++k) {
                const double dI = tiAdjLocked[first + k] - meanI;
                const double dQ = tqAdjLocked[first + k] - meanQ;
                maxDevIRE = std::max(maxDevIRE, std::hypot(dI, dQ) * invI);
            }

            const double consistentGate = 1.0 - std::clamp(
                maxDevIRE / std::max(1e-9, T.VET_WITNESS_STABLE_CHROMA_DEV_IRE),
                0.0, 1.0);
            return highGate * consistentGate;
        };

        auto retractedBlendFor = [&](int x, double alphaEff, double chromaIRE) -> double {
            if (!witnessYView.samples || T.VET_WITNESS_Y_WEIGHT <= 0.0)
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

            const double witnessConf =
                witnessConfRow ? std::clamp((double)witnessConfRow[x], 0.0, 1.0)
                               : 0.0;
            const double witnessObstruction =
                witnessObsRow ? std::clamp((double)witnessObsRow[x], 0.0, 1.0)
                              : 0.0;
            const double witnessResidualGapIRE =
                std::fabs(retractedSequencedCandidateAt(x, alphaEff) -
                          residualCandidateAt(x, alphaEff)) * invI;
            const double witnessGapGate = smoothStep01(
                (witnessResidualGapIRE - 0.75) / 3.25);
            const double witnessAuthority =
                witnessConf * (1.0 - 0.65 * witnessObstruction);
            const double witnessGate =
                witnessAuthority * witnessGapGate;

            const double currentAltIRE = currentSameLatticeAlt(x, alphaEff) * invI;
            if (currentAltIRE <= T.VET_WITNESS_ALT_START_IRE &&
                lsGate <= 0.0 &&
                witnessGate <= 0.0)
            {
                return finishRetractedBlend(0.0);
            }

            const double retractedAltIRE = sequencedSameLatticeAlt(x, alphaEff) * invI;
            const double altAdvantageIRE = currentAltIRE - retractedAltIRE;
            const double altGate = smoothStep01(
                (altAdvantageIRE - T.VET_WITNESS_ALT_START_IRE) /
                std::max(1e-9, T.VET_WITNESS_ALT_FULL_IRE -
                               T.VET_WITNESS_ALT_START_IRE));

            double anchorGate = 0.0;
            if (T.VET_NEIGHBOR_ANCHOR_ENABLE && (altGate > 0.0 || lsGate > 0.0)) {
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

            const double witnessBlend =
                T.VET_WITNESS_Y_WEIGHT * witnessGate;

            const double legacyGate = std::max({altGate, anchorGate, lsGate});
            if (legacyGate <= 0.0)
                return finishRetractedBlend(std::clamp(witnessBlend, 0.0, 1.0));

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

            const double protect = 1.0 - 0.75 * chromaProtect;
            const double legacyBlend =
                T.VET_WITNESS_Y_WEIGHT * legacyGate * attributionInvite * protect;
            const double blend = std::clamp(
                std::max(witnessBlend, legacyBlend),
                0.0,
                1.0);
            return finishRetractedBlend(blend);
        };

        auto writePixelNoAttribution = [&](int x, double alphaEff) {
            // Impurity protection: where the wide-window detector flags
            // luma-near-fsc contamination in the carrier estimate, reduce the
            // subtraction strength so the false-chroma component stays in Y.
            // CC_SUPPRESSION_WEIGHT scales the detector sensitivity (0=off, 1=full).
            //
            // Commit to the selected model. The witness owns the reconstructed
            // Y, while compact-transfer pixels enter election through the
            // independent combed carrier. In either case the elected carrier
            // is a complete model, so subtraction is full strength:
            // any alpha below 1 reinjects modeled carrier into Y as ripple —
            // the --ntsc1d (alpha 1, smooth) vs --two-d-variant line (hedged
            // alpha, checkerboard) breakpoint demonstrated this directly on
            // identical carrier data.  Luma protection happened upstream,
            // inside witness construction; it is not re-litigated here.
            // Impurity-driven suppression survives only for the witness-absent
            // fallback sources.
            if (witnessValid) {
                alphaEff = 1.0;
            } else if (impurityRow && impuritySuppressionWeight > 0.0) {
                const double impurity =
                    static_cast<double>(impurityRow[x]) * impuritySuppressionWeight;
                if (impurity > 0.0)
                    alphaEff *= (1.0 - impurity);
            }

            const int h = left + x;
            const double coarseWorking = sourceToWorkingSample(coarseY, x);
            const double residualWorking = sourceToWorkingSample(highRawY, x);
            const double coherentWorking = sourceToWorkingSample(coherentCombY, x);
            const double chromaIRE = std::hypot(tiAdjLocked[x], tqAdjLocked[x]) * invI;
            const double retBlend = retractedBlendFor(x, alphaEff, chromaIRE);
            if (instrumentProduceY) {
                ++produceYInstrumentation.pixels;
                if (witnessYView.samples) ++produceYInstrumentation.retractedAvailablePixels;
                if (retBlend > 0.0) ++produceYInstrumentation.retractedAppliedPixels;
                produceYInstrumentation.retractedBlendSum += retBlend;
                if (do3D) ++produceYInstrumentation.residual3DPixels;
            }

            const double residualY = coarseWorking + (residualWorking - alphaEff * coherentWorking);
            double witnessOrResidualY = residualY;
            if (retBlend > 0.0) {
                witnessOrResidualY = sourceToWorkingSample(witnessYView, x);
            }
            const double yOut = residualY * (1.0 - retBlend) +
                witnessOrResidualY * retBlend;
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
            double alphaEff = alphaWithAttribution(x, alphaVet);

            // Impurity protection: where the wide-window detector flags
            // luma-near-fsc contamination in the carrier estimate, reduce the
            // subtraction strength so the false-chroma component stays in Y.
            // CC_SUPPRESSION_WEIGHT scales the detector sensitivity (0=off, 1=full).
            // Model rule as in writePixelNoAttribution above: with a valid
            // witness, alpha is 1. Compact transfer still repairs yWitness,
            // but its carrier complement does not author the Field/Frame
            // election. Partial subtraction would still reinject the elected
            // carrier into Y as ripple. (The attribution alpha computed above
            // only governs the witness-absent fallback.)
            if (witnessValid) {
                alphaEff = 1.0;
            } else if (impurityRow && impuritySuppressionWeight > 0.0) {
                const double impurity =
                    static_cast<double>(impurityRow[x]) * impuritySuppressionWeight;
                if (impurity > 0.0)
                    alphaEff *= (1.0 - impurity);
            }

            const int h = left + x;
            const double coarseWorking = sourceToWorkingSample(coarseY, x);
            const double residualWorking = sourceToWorkingSample(highRawY, x);
            const double coherentWorking = sourceToWorkingSample(coherentCombY, x);
            const double chromaIRE = std::hypot(tiAdjLocked[x], tqAdjLocked[x]) * invI;
            const double retBlend = retractedBlendFor(x, alphaEff, chromaIRE);
            if (instrumentProduceY) {
                ++produceYInstrumentation.pixels;
                ++produceYInstrumentation.attributionPixels;
                if (witnessYView.samples) ++produceYInstrumentation.retractedAvailablePixels;
                if (retBlend > 0.0) ++produceYInstrumentation.retractedAppliedPixels;
                produceYInstrumentation.retractedBlendSum += retBlend;
                if (do3D) ++produceYInstrumentation.residual3DPixels;
            }

            const double residualY = coarseWorking + (residualWorking - alphaEff * coherentWorking);
            double witnessOrResidualY = residualY;
            if (retBlend > 0.0) {
                witnessOrResidualY = sourceToWorkingSample(witnessYView, x);
            }
            const double yOut = residualY * (1.0 - retBlend) +
                witnessOrResidualY * retBlend;
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
            // LUT-demodulated chroma magnitude — phase-correct, the same
            // measurement the rest of produceY trusts.  Used for both the
            // engagement gate and the projection confidence ramp.
            const double lutMag = 0.25 * (
                std::hypot(tiAdjLocked[p + 0], tqAdjLocked[p + 0]) +
                std::hypot(tiAdjLocked[p + 1], tqAdjLocked[p + 1]) +
                std::hypot(tiAdjLocked[p + 2], tqAdjLocked[p + 2]) +
                std::hypot(tiAdjLocked[p + 3], tqAdjLocked[p + 3])) * invI;

            const double r0 = sourceToWorkingSample(highRawY, p + 0);
            const double r1 = sourceToWorkingSample(highRawY, p + 1);
            const double r2 = sourceToWorkingSample(highRawY, p + 2);
            const double r3 = sourceToWorkingSample(highRawY, p + 3);

            const double c0 = sourceToWorkingSample(coherentCombY, p + 0);
            const double c1 = sourceToWorkingSample(coherentCombY, p + 1);
            const double c2 = sourceToWorkingSample(coherentCombY, p + 2);
            const double c3 = sourceToWorkingSample(coherentCombY, p + 3);

            // Fixed-index carrier projection.  The ratio rawI/subI is
            // coordinate-independent so the fixed-index frame is fine
            // for the scalar projection even though the magnitude is
            // slightly inflated relative to the LUT demod.
            const double rawI = r1 - r3;
            const double rawQ = r2 - r0;
            const double subI = c1 - c3;
            const double subQ = c2 - c0;

            const double subEnergy = subI * subI + subQ * subQ;
            const double subMagIRE = std::sqrt(subEnergy) * invI;

            if (instrumentProduceY) {
                produceYInstrumentation.alphaVetSubMagSum += subMagIRE;
                produceYInstrumentation.alphaVetSubMagMax =
                    std::max(produceYInstrumentation.alphaVetSubMagMax, subMagIRE);
                produceYInstrumentation.alphaVetLutMagSum += lutMag;
                produceYInstrumentation.alphaVetLutMagMax =
                    std::max(produceYInstrumentation.alphaVetLutMagMax, lutMag);
                if (lutMag < MIN_SUB_CHROMA_IRE)
                    ++produceYInstrumentation.alphaVetGateFailCalls;
            }

            if (!chromaLikeEnabled || lutMag < MIN_SUB_CHROMA_IRE) {
                return 1.0;
            }

            // Guard: if the fixed-index sub-energy is negligible the
            // projection denominator is unreliable regardless of lutMag.
            if (subEnergy < 1e-6)
                return 1.0;

            const double alphaFit = std::clamp(
                (rawI * subI + rawQ * subQ) / (subEnergy + 1e-12),
                MIN_ALPHA, MAX_ALPHA);

            // Projection confidence scales with chroma magnitude:
            // just above MIN_SUB_CHROMA_IRE the SNR is marginal;
            // at VET_ALPHA_FIT_FULL_IRE it is fully reliable.
            const double fitReliability = smoothStep01(
                (lutMag - MIN_SUB_CHROMA_IRE) /
                std::max(1e-9, T.VET_ALPHA_FIT_FULL_IRE - MIN_SUB_CHROMA_IRE));

            const double profileWeight = chromaLikeWeight * fitReliability;
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
                if (instrumentProduceY) {
                    ++produceYInstrumentation.alphaVetCalls;
                    produceYInstrumentation.alphaVetPixels += 4;
                    produceYInstrumentation.alphaVetDeltaSum += std::fabs(alphaVet - 1.0);
                    if (std::fabs(alphaVet - 1.0) > 1e-6) {
                        ++produceYInstrumentation.alphaVetAdjustedCalls;
                    }
                }
                writePixelWithAttribution(p + 0, alphaVet);
                writePixelWithAttribution(p + 1, alphaVet);
                writePixelWithAttribution(p + 2, alphaVet);
                writePixelWithAttribution(p + 3, alphaVet);
            }

            if (tailStart < width) {
                const int p = std::max(0, width - 4);
                const double alphaVet = computeAlphaVet(p);
                if (instrumentProduceY) {
                    const int tailPixels = width - tailStart;
                    ++produceYInstrumentation.alphaVetCalls;
                    produceYInstrumentation.alphaVetPixels += tailPixels;
                    produceYInstrumentation.alphaVetDeltaSum += std::fabs(alphaVet - 1.0);
                    if (std::fabs(alphaVet - 1.0) > 1e-6) {
                        ++produceYInstrumentation.alphaVetAdjustedCalls;
                    }
                }
                for (int x = tailStart; x < width; ++x) {
                    writePixelWithAttribution(x, alphaVet);
                }
            }
        } else {
            for (int p = 0; p + 3 < width; p += 4) {
                const double alphaVet = computeAlphaVet(p);
                if (instrumentProduceY) {
                    ++produceYInstrumentation.alphaVetCalls;
                    produceYInstrumentation.alphaVetPixels += 4;
                    produceYInstrumentation.alphaVetDeltaSum += std::fabs(alphaVet - 1.0);
                    if (std::fabs(alphaVet - 1.0) > 1e-6) {
                        ++produceYInstrumentation.alphaVetAdjustedCalls;
                    }
                }
                writePixelNoAttribution(p + 0, alphaVet);
                writePixelNoAttribution(p + 1, alphaVet);
                writePixelNoAttribution(p + 2, alphaVet);
                writePixelNoAttribution(p + 3, alphaVet);
            }

            if (tailStart < width) {
                const int p = std::max(0, width - 4);
                const double alphaVet = computeAlphaVet(p);
                if (instrumentProduceY) {
                    const int tailPixels = width - tailStart;
                    ++produceYInstrumentation.alphaVetCalls;
                    produceYInstrumentation.alphaVetPixels += tailPixels;
                    produceYInstrumentation.alphaVetDeltaSum += std::fabs(alphaVet - 1.0);
                    if (std::fabs(alphaVet - 1.0) > 1e-6) {
                        ++produceYInstrumentation.alphaVetAdjustedCalls;
                    }
                }
                for (int x = tailStart; x < width; ++x) {
                    writePixelNoAttribution(x, alphaVet);
                }
            }
        }
    }

    if (configuration.outputDiagnosticY &&
        carrierRetractedValid &&
        !carrierRetracted_flat.empty() &&
        demodWidth == width)
    {
        for (int line = firstLine; line < lastLine; ++line) {
            const CombCarrierGrammar *grammar = carrierGrammarLine(line);
            if (!grammar || !grammar->grammarLocked)
                continue;

            double *Y = componentFrame->y(line);
            const float *diagnosticY = witnessValid
                ? yWitness_line(line)
                : carrierRetracted_line(line);
            if (!Y || !diagnosticY)
                continue;

            for (int x = 0; x < width; ++x)
                Y[left + x] = static_cast<double>(diagnosticY[x]);
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

// Lurch preconditioner for the coarse luma prior.
//
// The legal 4-sample means cancel carrier, but they are boxcars: a luma step
// smears across the four windows that straddle it, and every pre-2D consumer
// of the coarse prior (the carrier fit, the witness, the patch gates) then
// sees step energy stranded in the carrier band.  Membership movement between
// adjacent windows,
//
//     D[s] = M[s+1] - M[s] = (raw[s+4] - raw[s]) / 4,
//
// compares samples of identical carrier phase, so a luma step produces a
// same-sign run of D across the straddling windows — phase-invariant — while
// a chroma envelope edge alternates sign window to window and is rejected by
// the run test.  Where a step is confirmed, the prior takes the nearest
// same-side window mean instead of the boxcar ramp: the transition lands at
// one column instead of four.  One steep transition per aperture, by design.
//
// `prior` is blended in place; `gateOut` (optional) reports per-pixel
// sharpening activity so a consumer can stand down its own edge correction.
void Comb::FrameBuffer::lurchSharpenCoarsePrior(const double *means,
                                                int meanCount,
                                                int width,
                                                double *prior,
                                                double *gateOut) const
{
    if (gateOut && width > 0)
        std::fill(gateOut, gateOut + width, 0.0);

    if (!means || !prior || meanCount < 6 || width <= 0)
        return;

    const auto smoothStep01 = [](double t) {
        t = std::clamp(t, 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    };

    // Per-window movement floor: a confirmed step of >= ~1.2 IRE moves each
    // straddling window by >= ~0.3 IRE.
    const double dThreshSamples = 0.30 * irescale;

    const int dCount = meanCount - 1;

    struct StepRun {
        int a = 0;
        int b = 0;
        double edge = 0.0;
        double stepAbsIRE = 0.0;
        double gate = 0.0;
        bool suppressed = false;
    };
    std::vector<StepRun> runs;

    int s = 0;
    while (s < dCount) {
        const double d0 = means[s + 1] - means[s];
        if (std::fabs(d0) < dThreshSamples) {
            ++s;
            continue;
        }

        const bool positive = d0 > 0.0;
        const int a = s;
        int b = s;
        while (b + 1 < dCount) {
            const double dn = means[b + 2] - means[b + 1];
            if (std::fabs(dn) < dThreshSamples || (dn > 0.0) != positive)
                break;
            ++b;
        }
        s = b + 1;

        // An ideal step moves exactly four windows; allow slack for noise
        // and local gradient, but a long run is a ramp, not a step.
        const int runLength = b - a + 1;
        if (runLength > 6)
            continue;

        const double stepSamples =
            means[std::min(b + 1, meanCount - 1)] - means[a];
        const double stepIRE = std::fabs(stepSamples) * invIreScale;
        const double gate = smoothStep01((stepIRE - 1.25) / 2.75);
        if (gate <= 0.0)
            continue;

        // Amplitude-weighted centroid of |D| locates the edge even when the
        // threshold trimmed the run asymmetrically (an ideal step's run is
        // s in [e-4, e-1], centroid e-2.5).
        double wSum = 0.0;
        double wPos = 0.0;
        for (int k = a; k <= b; ++k) {
            const double w = std::fabs(means[k + 1] - means[k]);
            wSum += w;
            wPos += w * (double)k;
        }
        const double centroid =
            (wSum > 1e-12) ? (wPos / wSum) : 0.5 * (double)(a + b);

        StepRun run;
        run.a = a;
        run.b = b;
        run.edge = centroid + 2.5;
        run.stepAbsIRE = stepIRE;
        run.gate = gate;
        runs.push_back(run);
    }

    if (runs.empty())
        return;

    // Ringing suppression.  Sharp analog edges overshoot, which reverses the
    // membership movement and shows up as a smaller opposite run right beside
    // the true step.  Snapping to those fragments puts line-varying plateaus
    // into raw - prior at exactly the edges this pass exists to clean — the
    // fringes survive every comb stage because the contamination is upstream
    // of all of them.  A run within a few windows of a materially stronger
    // run is overshoot, not a second edge.
    for (size_t i = 0; i < runs.size(); ++i) {
        for (size_t j = 0; j < runs.size(); ++j) {
            if (i == j)
                continue;
            const int gap = (runs[i].a > runs[j].b)
                ? runs[i].a - runs[j].b
                : runs[j].a - runs[i].b;
            if (gap <= 3 && runs[j].stepAbsIRE >= 2.5 * runs[i].stepAbsIRE) {
                runs[i].suppressed = true;
                break;
            }
        }
    }

    // Apply the strongest surviving run per pixel, always blending from the
    // unsharpened base so overlapping runs never compound.
    std::vector<double> base(prior, prior + width);
    std::vector<double> localGate(width, 0.0);

    // A side anchor is only trustworthy if its window is clear of every
    // detected run.  A short bar (e.g. a colored patch abutting black) has
    // two close opposite runs; an anchor that crosses into the other run's
    // span would snap the bar interior to the far plateau.  With no clean
    // anchor on the required side, leave the pixel on the base prior — an
    // honest blur beats a confident wrong plateau.
    const auto anchorContaminated = [&runs](int s) {
        for (const StepRun &other : runs) {
            if (other.suppressed)
                continue;
            if (s >= other.a && s <= other.b + 1)
                return true;
        }
        return false;
    };

    for (const StepRun &run : runs) {
        if (run.suppressed)
            continue;

        const int xiFirst =
            std::clamp((int)std::floor(run.edge) - 4, 0, width - 1);
        const int xiLast =
            std::clamp((int)std::ceil(run.edge) + 3, 0, width - 1);

        for (int xi = xiFirst; xi <= xiLast; ++xi) {
            if (run.gate <= localGate[xi])
                continue;

            // One window of margin beyond the detected run: threshold
            // trimming can leave the run's end windows still straddling the
            // step, and snapping to a straddling window is worse than not
            // snapping at all.
            const int side = ((double)xi < run.edge)
                ? std::clamp(std::min(xi, run.a - 1), 0, meanCount - 1)
                : std::clamp(std::max(xi - 3, run.b + 2), 0, meanCount - 1);

            if (anchorContaminated(side))
                continue;

            localGate[xi] = run.gate;
            prior[xi] = base[xi] * (1.0 - run.gate) + means[side] * run.gate;
        }
    }

    if (gateOut) {
        for (int xi = 0; xi < width; ++xi)
            gateOut[xi] = localGate[xi];
    }
}

// Build the carrier-retracted view and its derived products.
//
// Per-line pass:
//   1. Four-view carrier/Y attribution on legal 4fSC luma floors
//      → carrierFit_flat; gated LS can refit contaminated luma edges.
//   2. Provisional raw - carrierFit is used only as a local residual scratch
//      while the line fit is being built.  The provisional floor stays in
//      scratch until the combed carrier exists.
//
// Cross-line pass (after all per-line fits):
//   3. Line-to-line cancellation on carrierFit → combedCarrier_flat
//      Real chroma inverts between opposite-phase lines, alien-Y doesn't.
//      combedCarrier preserves chroma and rejects alien-Y.
//   4. raw - combedCarrier → carrierRetracted_flat (flattened view)
//   5. Sliding 4-sample mean of carrierRetracted_flat → flatFloor_flat
//      (carrier-free luma floor; the 4-sample mean cancels carrier-shaped
//      residual at color transitions, leaving only genuine DC alien-Y)
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
    if (coarseYEvidence_flat.size() < need)
        coarseYEvidence_flat.assign(need, lddecode::FourViewPixelEvidence{});
    if (carrierImpurity_flat.size() < need)
        carrierImpurity_flat.assign(need, 0.0f);
    if (carrierParallax_flat.size() < need)
        carrierParallax_flat.assign(need, lddecode::FourViewCarrierAttribution{});

    if ((int)scratch_preI.size()        < width) scratch_preI.resize(width, 0.0);
    if ((int)scratch_preQ.size()        < width) scratch_preQ.resize(width, 0.0);
    if ((int)scratch_lineWorkA.size()   < width) scratch_lineWorkA.resize(width, 0.0);
    if ((int)scratch_lineWorkB.size()   < width) scratch_lineWorkB.resize(width, 0.0);
    if ((int)scratch_lineWorkC.size()   < width) scratch_lineWorkC.resize(width, 0.0);
    if ((int)scratch_lineWorkD.size()   < width) scratch_lineWorkD.resize(width, 0.0);
    if ((int)scratch_lumaBaseY4.size()  < width) scratch_lumaBaseY4.resize(width, 0.0);
    if ((int)scratch_lumaSmooth.size()  < width) scratch_lumaSmooth.resize(width, 0.0);
    if ((int)scratch_lateralLine.size() < width) scratch_lateralLine.resize(width, 0.0);
    if ((int)scratch_yhp.size()         < width) scratch_yhp.resize(width, 0.0);
    if ((int)scratch_yI.size()          < width) scratch_yI.resize(width, 0.0);
    if ((int)scratch_yQ.size()          < width) scratch_yQ.resize(width, 0.0);
    if ((int)scratch_hpI.size()         < width) scratch_hpI.resize(width, 0.0);
    if ((int)scratch_hpQ.size()         < width) scratch_hpQ.resize(width, 0.0);
    if ((int)scratch_hpY.size()         < width) scratch_hpY.resize(width, 0.0);

    double *rawWhole   = scratch_preI.data();
    double *coarseY    = scratch_preQ.data();
    double *carrierFit = scratch_lineWorkA.data();
    double *basisI     = scratch_lineWorkB.data();
    double *flattened  = scratch_lineWorkC.data();
    double *basisQ     = scratch_lineWorkD.data();
    double *refinedY   = scratch_lumaSmooth.data();
    double *slideMean4 = scratch_lateralLine.data();
    double *discResponseSupport = scratch_yhp.data();
    double *discResponseI = scratch_yI.data();
    double *discResponseQ = scratch_yQ.data();
    double *discResponseSupportSmooth = scratch_hpY.data();
    double *discResponseISmooth = scratch_hpI.data();
    double *discResponseQSmooth = scratch_hpQ.data();

    std::vector<double> winFloor;
    std::vector<double> winI;
    std::vector<double> winQ;
    std::vector<double> winErrorIRE;
    std::vector<double> winLatticeIRE;
    std::vector<double> winYSpanIRE;
    std::vector<double> winScore;

    auto median3 = [](double a, double b, double c) -> double {
        if (a > b) std::swap(a, b);
        if (b > c) std::swap(b, c);
        if (a > b) std::swap(a, b);
        return b;
    };

    // Flat-region fast path: when all four views agree to within ~0.05 IRE,
    // the cost matrix is dominated by arithmetic noise and the medoid is
    // within the same tolerance of the plain mean. Skip 16 abs + compares
    // (twice per pixel across the two call sites) and return the mean. The
    // anchor only matters as a tiebreak when views disagree, so it does not
    // need to participate in the flat case.
    const double medoidFlatTol = 0.05 * irescale;
    auto medoid4Anchored = [medoidFlatTol](double a, double b, double c, double d,
                                            double anchor) -> double {
        const double lo = std::min(std::min(a, b), std::min(c, d));
        const double hi = std::max(std::max(a, b), std::max(c, d));
        if (hi - lo < medoidFlatTol)
            return 0.25 * (a + b + c + d);

        double v[4] = {a, b, c, d};
        double best = v[0];
        double bestCost = 1e300;
        double bestAnchorDist = 1e300;

        for (int i = 0; i < 4; ++i) {
            double cost = 0.0;
            for (int j = 0; j < 4; ++j)
                cost += std::fabs(v[i] - v[j]);

            const double anchorDist = std::fabs(v[i] - anchor);
            if (cost < bestCost - 1e-9 ||
                (std::fabs(cost - bestCost) <= 1e-9 &&
                 anchorDist < bestAnchorDist)) {
                best = v[i];
                bestCost = cost;
                bestAnchorDist = anchorDist;
            }
        }

        return best;
    };

    auto smoothStep01 = [](double t) {
        t = std::clamp(t, 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    };

    // ---------------------------------------------------------------
    // Pass 1: per-line carrier withdrawal.
    //
    // This version does not start from the ±2 complement estimate.  It first
    // derives a per-sample Y prior from the legal 4-sample luma-floor views,
    // then projects raw - refinedY into locked IQ.  The four carrier windows
    // touching the current sample are treated as attribution evidence rather
    // than being blended as an immediate heuristic result.
    //
    // In other words:
    //
    //     raw = Y + C
    //
    // is resolved by asking which legal Y floor leaves the most coherent C,
    // rather than by subtracting a complement-estimated C and trusting whatever
    // remains as Y.
    // ---------------------------------------------------------------
    for (int line = firstLine; line < lastLine; ++line) {
        float *fitRow       = carrierFit_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        float *retractedRow = carrierRetracted_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        float *floorRow     = flatFloor_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        float *gateRow      = lsRefitGate_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        auto *evidenceRow   = coarseYEvidence_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        auto *parallaxRow   = carrierParallax_flat.empty()
                              ? nullptr
                              : carrierParallax_flat.data()
                                + static_cast<size_t>(line) * demodWidth;
        float *impurityRow  = carrierImpurity_flat.data()
                              + static_cast<size_t>(line) * demodWidth;

        std::fill(gateRow, gateRow + width, 0.0f);
        std::fill(evidenceRow, evidenceRow + width, lddecode::FourViewPixelEvidence{});
        std::fill(impurityRow, impurityRow + width, 0.0f);

        const CombCarrierGrammar *grammar = carrierGrammarLine(line);
        const bool grammarLocked = grammar && grammar->grammarLocked;

        const quint16 *rawLine =
            rawbuffer.data() + static_cast<size_t>(line) * videoParameters.fieldWidth;

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
            refinedY[xi] = coarseY[xi];
            evidenceRow[xi].rawSample = static_cast<float>(rawWhole[xi]);
        }

        if (!grammarLocked) {
            for (int xi = 0; xi < width; ++xi) {
                fitRow[xi]       = 0.0f;
                retractedRow[xi] = static_cast<float>(rawWhole[xi]);
                floorRow[xi]     = static_cast<float>(coarseY[xi]);
                if (parallaxRow)
                    parallaxRow[xi] = lddecode::FourViewCarrierAttribution{};
            }
            continue;
        }

        const double bcos = grammar->burstCos;
        const double bsin = grammar->burstSin;
        const double maxCarrierSamples =
            std::max(24.0, grammar->carrierScale * 5.0) * irescale;

        // basisI/Q at position h depend only on (h & 3) given the line's
        // burst phasor and locked basis (remod4fscToShiftedComposite indexes
        // spLUT/cpLUT by (h & 3); lockedTo4fsc takes no h).  Precompute the
        // four phase values and fill by lookup — replaces 2*width function
        // calls per line with 8 function calls + 2*width table reads.
        double basisI4[4];
        double basisQ4[4];
        for (int p = 0; p < 4; ++p) {
            basisI4[p] = remodLockedToShiftedComposite(
                1.0, 0.0, p, bcos, bsin,
                spLUT_locked, cpLUT_locked);
            basisQ4[p] = remodLockedToShiftedComposite(
                0.0, 1.0, p, bcos, bsin,
                spLUT_locked, cpLUT_locked);
        }
        for (int xi = 0; xi < width; ++xi) {
            const int idx = (left + xi) & 3;
            basisI[xi] = basisI4[idx];
            basisQ[xi] = basisQ4[idx];
        }

        if (width >= 4) {
            const int meanCount = width - 3;
            if ((int)winFloor.size() < meanCount) {
                winFloor.resize(meanCount, 0.0);
                winI.resize(meanCount, 0.0);
                winQ.resize(meanCount, 0.0);
                winErrorIRE.resize(meanCount, 0.0);
                winLatticeIRE.resize(meanCount, 0.0);
                winYSpanIRE.resize(meanCount, 0.0);
                winScore.resize(meanCount, 0.0);
            }

            for (int s = 0; s < meanCount; ++s) {
                winFloor[s] =
                    0.25 * (rawWhole[s + 0] +
                            rawWhole[s + 1] +
                            rawWhole[s + 2] +
                            rawWhole[s + 3]);
            }

            // Luma prior: the centered moving coarse, not a medoid of the
            // four covering means.  The medoid was robust but is still a
            // boxcar statistic with half the smear baked in; the prior is
            // the rolling legal mean, and the lurch preconditioner restores
            // the step placement the boxcar blurs.
            for (int xi = 0; xi < width; ++xi) {
                const int sc = std::clamp(xi - 1, 0, meanCount - 1);
                refinedY[xi] = winFloor[sc];
            }

            // Lurch preconditioner: sharpen the prior before the carrier fit
            // consumes it, so step energy stays out of raw - refinedY and
            // never enters the carrier band.
            lurchSharpenCoarsePrior(winFloor.data(), meanCount, width,
                                    refinedY, nullptr);

            for (int s = 0; s < meanCount; ++s) {
                double sII = 0.0, sIQ = 0.0, sQQ = 0.0;
                double sIY = 0.0, sQY = 0.0;

                double refinedMean = 0.0;
                double minRefined = 1e300;
                double maxRefined = -1e300;

                for (int k = 0; k < 4; ++k) {
                    const int xi = s + k;
                    const double bI = basisI[xi];
                    const double bQ = basisQ[xi];

                    const double residual = rawWhole[xi] - refinedY[xi];

                    sII += bI * bI;
                    sIQ += bI * bQ;
                    sQQ += bQ * bQ;
                    sIY += bI * residual;
                    sQY += bQ * residual;

                    refinedMean += refinedY[xi];
                    minRefined = std::min(minRefined, refinedY[xi]);
                    maxRefined = std::max(maxRefined, refinedY[xi]);
                }

                refinedMean *= 0.25;

                double fitI = 0.0;
                double fitQ = 0.0;
                const double det = sII * sQQ - sIQ * sIQ;
                if (std::fabs(det) > 1e-9) {
                    const double inv = 1.0 / det;
                    fitI = ( sQQ * sIY - sIQ * sQY) * inv;
                    fitQ = (-sIQ * sIY + sII * sQY) * inv;
                }

                winI[s] = fitI;
                winQ[s] = fitQ;

                double errSq = 0.0;
                double basis01 = 0.0; // +-+-
                double basis02 = 0.0; // ++--
                double basis03 = 0.0; // +--+
                double fitAbs = 0.0;

                for (int k = 0; k < 4; ++k) {
                    const int xi = s + k;
                    const double bI = basisI[xi];
                    const double bQ = basisQ[xi];

                    const double fit = fitI * bI + fitQ * bQ;
                    const double residual = rawWhole[xi] - refinedY[xi];
                    const double e = residual - fit;

                    errSq += e * e;
                    fitAbs += std::fabs(fit);

                    if (k == 0) {
                        basis01 += e;
                        basis02 += e;
                        basis03 += e;
                    } else if (k == 1) {
                        basis01 -= e;
                        basis02 += e;
                        basis03 -= e;
                    } else if (k == 2) {
                        basis01 += e;
                        basis02 -= e;
                        basis03 -= e;
                    } else {
                        basis01 -= e;
                        basis02 -= e;
                        basis03 += e;
                    }
                }

                const double errIRE = std::sqrt(0.25 * errSq) * invIreScale;
                const double latticeIRE =
                    0.25 * std::max({std::fabs(basis01),
                                     std::fabs(basis02),
                                     std::fabs(basis03)}) * invIreScale;
                const double floorDriftIRE =
                    std::fabs(winFloor[s] - refinedMean) * invIreScale;
                const double ySpanIRE =
                    (maxRefined - minRefined) * invIreScale;
                const double ampIRE =
                    0.25 * fitAbs * invIreScale;

                winErrorIRE[s] = errIRE;
                winLatticeIRE[s] = latticeIRE;
                winYSpanIRE[s] = ySpanIRE;

                // The score prefers:
                //   - residual that remodulates cleanly through locked IQ,
                //   - little remaining +-+- / ++-- / +--+ lattice residue,
                //   - a legal floor that agrees with the refined-Y prior,
                //   - low refined-Y span inside the 4-sample cell.
                //
                // The small amplitude term prevents the all-Y interpretation
                // from winning merely because it is conservative.
                winScore[s] =
                    errIRE +
                    0.75 * latticeIRE +
                    0.25 * floorDriftIRE +
                    0.15 * ySpanIRE -
                    0.10 * std::min(ampIRE, 24.0);
            }

            std::fill(discResponseSupport, discResponseSupport + width, 0.0);
            std::fill(discResponseI, discResponseI + width, 0.0);
            std::fill(discResponseQ, discResponseQ + width, 0.0);
            std::fill(discResponseSupportSmooth, discResponseSupportSmooth + width, 0.0);
            std::fill(discResponseISmooth, discResponseISmooth + width, 0.0);
            std::fill(discResponseQSmooth, discResponseQSmooth + width, 0.0);

            if (T.RETRACTED_DISC_RESPONSE_ENABLE) {
                bool anyDiscSupport = false;
                for (int s = 0; s < meanCount; ++s) {
                    const double ampIRE =
                        std::hypot(winI[s], winQ[s]) * invIreScale;
                    const double ampGate = smoothStep01(
                        (ampIRE - T.RETRACTED_DISC_RESPONSE_STABLE_START_IRE) /
                        std::max(1e-9,
                            T.RETRACTED_DISC_RESPONSE_STABLE_FULL_IRE -
                            T.RETRACTED_DISC_RESPONSE_STABLE_START_IRE));
                    const double fitGate = 1.0 - smoothStep01((winErrorIRE[s] - 1.5) / 5.0);
                    const double latticeGate = 1.0 - smoothStep01((winLatticeIRE[s] - 1.0) / 5.0);
                    const double spanGate = 1.0 - smoothStep01((winYSpanIRE[s] - 3.0) / 9.0);
                    const double support = ampGate * fitGate * latticeGate * spanGate;
                    if (support <= 1e-6)
                        continue;

                    anyDiscSupport = true;
                    for (int k = 0; k < 4; ++k) {
                        const int xi = s + k;
                        discResponseSupport[xi] += support;
                        discResponseI[xi] += support * winI[s];
                        discResponseQ[xi] += support * winQ[s];
                    }
                }

                if (anyDiscSupport) {
                    for (int xi = 0; xi < width; ++xi) {
                        const double support = discResponseSupport[xi];
                        if (support > 1e-9) {
                            discResponseI[xi] /= support;
                            discResponseQ[xi] /= support;
                            discResponseSupport[xi] = std::clamp(0.25 * support, 0.0, 1.0);
                        }
                    }

                    const int radius = std::clamp(
                        T.RETRACTED_DISC_RESPONSE_RADIUS,
                        1,
                        std::max(1, width - 1));
                    for (int xi = 0; xi < width; ++xi) {
                        double sumBase = 0.0;
                        double sumSupport = 0.0;
                        double sumWeight = 0.0;
                        double sumI = 0.0;
                        double sumQ = 0.0;

                        for (int dx = -radius; dx <= radius; ++dx) {
                            const int xx = std::clamp(xi + dx, 0, width - 1);
                            const double baseW =
                                static_cast<double>(radius + 1 - std::abs(dx));
                            const double support = discResponseSupport[xx];
                            sumBase += baseW;
                            sumSupport += baseW * support;
                            const double w = baseW * support;
                            sumWeight += w;
                            sumI += w * discResponseI[xx];
                            sumQ += w * discResponseQ[xx];
                        }

                        discResponseSupportSmooth[xi] =
                            (sumBase > 1e-9)
                                ? std::clamp(sumSupport / sumBase, 0.0, 1.0)
                                : 0.0;
                        if (sumWeight > 1e-9) {
                            discResponseISmooth[xi] = sumI / sumWeight;
                            discResponseQSmooth[xi] = sumQ / sumWeight;
                        }
                    }
                }
            }

            for (int xi = 0; xi < width; ++xi) {
                const int sFirst = std::max(0, xi - 3);
                const int sLast  = std::min(xi, meanCount - 1);

                lddecode::FourViewCarrierView views[4];
                int viewCount = 0;

                for (int s = sFirst; s <= sLast; ++s) {
                    if (viewCount >= 4)
                        break;
                    const double carrierSample = rawWhole[xi] - winFloor[s];
                    const double fittedSample =
                        winI[s] * basisI[xi] + winQ[s] * basisQ[xi];
                    views[viewCount].apertureStart = s;
                    views[viewCount].apertureCenter = static_cast<double>(s) + 1.5;
                    views[viewCount].yFloor = winFloor[s];
                    views[viewCount].carrierSample = carrierSample;
                    views[viewCount].fittedSample = fittedSample;
                    views[viewCount].carrierI = winI[s];
                    views[viewCount].carrierQ = winQ[s];
                    views[viewCount].sampleFitErrorIRE =
                        std::fabs(carrierSample - fittedSample) * invIreScale;
                    views[viewCount].remodErrorIRE = winErrorIRE[s];
                    views[viewCount].latticeRiskIRE = winLatticeIRE[s];
                    views[viewCount].ySpanIRE = winYSpanIRE[s];
                    views[viewCount].score = winScore[s];
                    {
                        double mDeltaSample = 0.0;
                        double mDeltaIRE = 0.0;
                        double mSupport = 0.0;
                        double mLocalX = 0.0;
                        const int s0 = s;
                        if (s0 + 4 < width) {
                            const double enterLeaveSample =
                                rawWhole[s0 + 4] - rawWhole[s0];
                            mDeltaSample = 0.25 * enterLeaveSample;
                            mDeltaIRE = mDeltaSample * invIreScale;
                            const double deltaMagIRE = std::fabs(mDeltaIRE);
                            const double deltaGate =
                                smoothStep01((deltaMagIRE - 0.35) / 4.0);
                            const double fitGate =
                                1.0 - smoothStep01((winErrorIRE[s0] - 1.5) / 5.0);
                            const double latticeGate =
                                1.0 - smoothStep01((winLatticeIRE[s0] - 1.0) / 5.0);
                            mSupport = std::clamp(
                                deltaGate * fitGate * latticeGate, 0.0, 1.0);
                            mLocalX = 0.5 * ((double)s0 + (double)(s0 + 4))
                                    - (double)xi;
                        }
                        views[viewCount].membershipDeltaSample = mDeltaSample;
                        views[viewCount].membershipDeltaIRE    = mDeltaIRE;
                        views[viewCount].membershipSupport     = mSupport;
                        views[viewCount].membershipLocalX      = mLocalX;
                    }
                    ++viewCount;
                }

                evidenceRow[xi].viewCount = viewCount;
                for (int v = 0; v < viewCount; ++v) {
                    auto &dst = evidenceRow[xi].views[v];
                    const auto &src = views[v];
                    dst.apertureStart = src.apertureStart;
                    dst.apertureCenter = static_cast<float>(src.apertureCenter);
                    dst.yFloor = static_cast<float>(src.yFloor);
                    dst.carrierSample = static_cast<float>(src.carrierSample);
                    dst.fittedSample = static_cast<float>(src.fittedSample);
                    dst.carrierI = static_cast<float>(src.carrierI);
                    dst.carrierQ = static_cast<float>(src.carrierQ);
                    dst.sampleFitErrorIRE = static_cast<float>(src.sampleFitErrorIRE);
                    dst.remodErrorIRE = static_cast<float>(src.remodErrorIRE);
                    dst.latticeRiskIRE = static_cast<float>(src.latticeRiskIRE);
                    dst.ySpanIRE = static_cast<float>(src.ySpanIRE);
                    dst.membershipDeltaSample = static_cast<float>(src.membershipDeltaSample);
                    dst.membershipDeltaIRE    = static_cast<float>(src.membershipDeltaIRE);
                    dst.membershipSupport     = static_cast<float>(src.membershipSupport);
                    dst.membershipLocalX      = static_cast<float>(src.membershipLocalX);
                    dst.score = static_cast<float>(src.score);
                }

                auto parallax = lddecode::buildFourViewCarrierAttribution(
                    views,
                    viewCount,
                    invIreScale);

                double modelI = parallax.valid ? parallax.commonI : 0.0;
                double modelQ = parallax.valid ? parallax.commonQ : 0.0;
                double discResponseBlend = 0.0;
                double stableI = 0.0;
                double stableQ = 0.0;
                double stableMagIRE = 0.0;

                if (parallax.valid &&
                    T.RETRACTED_DISC_RESPONSE_ENABLE &&
                    discResponseSupportSmooth[xi] > 0.0)
                {
                    stableI = discResponseISmooth[xi];
                    stableQ = discResponseQSmooth[xi];
                    stableMagIRE = std::hypot(stableI, stableQ) * invIreScale;
                    const double chromaGate = smoothStep01(
                        (std::max(stableMagIRE, parallax.commonMagIRE) -
                         T.RETRACTED_DISC_RESPONSE_STABLE_START_IRE) /
                        std::max(1e-9,
                            T.RETRACTED_DISC_RESPONSE_STABLE_FULL_IRE -
                            T.RETRACTED_DISC_RESPONSE_STABLE_START_IRE));
                    const double yContextGate = smoothStep01(
                        (std::max(parallax.yCurvatureIRE,
                                  0.5 * parallax.ySpreadIRE) - 2.0) / 10.0);
                    const double fitRisk = smoothStep01(
                        (parallax.sampleFitErrorIRE - 1.0) / 5.0);
                    const double spreadRisk = std::clamp(
                        parallax.carrierSpreadIRE /
                        std::max(4.0, 0.35 * parallax.commonMagIRE + 1.0),
                        0.0,
                        1.0);
                    const double hostileGate = std::max(
                        yContextGate,
                        std::max(fitRisk,
                                 std::max(spreadRisk,
                                          1.0 - parallax.carrierCoherence)));
                    discResponseBlend = std::min(
                        std::clamp(T.RETRACTED_DISC_RESPONSE_WEIGHT, 0.0, 1.0),
                        std::clamp(T.RETRACTED_DISC_RESPONSE_WEIGHT, 0.0, 1.0) *
                        discResponseSupportSmooth[xi] *
                        chromaGate *
                        (0.20 + 0.80 * hostileGate));

                    modelI = modelI * (1.0 - discResponseBlend) +
                             stableI * discResponseBlend;
                    modelQ = modelQ * (1.0 - discResponseBlend) +
                             stableQ * discResponseBlend;
                }

                parallax.discResponseI = stableI;
                parallax.discResponseQ = stableQ;
                parallax.discResponseMagIRE = stableMagIRE;
                parallax.discResponseSupport = discResponseSupportSmooth[xi];
                parallax.discResponseBlend = discResponseBlend;

                double residualCarrierLo = -1e300;
                double residualCarrierHi =  1e300;
                double residualTightenSample = 0.0;
                double residualTightenSupport = 0.0;
                double residualTightenGain = 1.0;
                double residualTightenSpreadIRE = 0.0;
                double residualTightenFitErrorIRE = 0.0;
                if (viewCount > 0) {
                    residualCarrierLo =  1e300;
                    residualCarrierHi = -1e300;
                    double sumResidual = 0.0;
                    double sumFitError = 0.0;
                    double minFloor = 1e300;
                    double maxFloor = -1e300;
                    double maxLurchIRE = 0.0;
                    double maxLurchSupport = 0.0;

                    // Residual complements are observations, not estimates.
                    // For a carrier-compatible local structure, the smallest
                    // absolute residual amplitude is the conservative survivor:
                    // any larger same-structure carrier claim must be justified
                    // by additional evidence rather than by the workprint alone.
                    double minCompatibleCarrierSample = 0.0;
                    double minCompatibleAbs = 1e300;
                    double minCompatibleSupport = 0.0;
                    int compatibleResidualN = 0;

                    const double referenceSample = (parallax.valid &&
                        std::fabs(parallax.commonSample) > 0.5 / std::max(1e-12, invIreScale))
                            ? parallax.commonSample
                            : (modelI * basisI[xi] + modelQ * basisQ[xi]);
                    const int referenceSign = (referenceSample > 0.0) ? +1 :
                                              ((referenceSample < 0.0) ? -1 : 0);

                    int residualN = 0;

                    for (int v = 0; v < viewCount; ++v) {
                        const auto &view = views[v];
                        residualCarrierLo = std::min(residualCarrierLo, view.carrierSample);
                        residualCarrierHi = std::max(residualCarrierHi, view.carrierSample);
                        sumResidual += view.carrierSample;
                        sumFitError += view.sampleFitErrorIRE;
                        minFloor = std::min(minFloor, view.yFloor);
                        maxFloor = std::max(maxFloor, view.yFloor);
                        // Lurch evidence comes from the membership-change
                        // observations.
                        const double membershipLocalizer = std::exp(
                            -0.5 * (view.membershipLocalX * view.membershipLocalX)
                                 / (1.35 * 1.35));
                        const double lurch =
                            std::fabs(view.membershipDeltaIRE) *
                            std::clamp(view.membershipSupport, 0.0, 1.0) *
                            membershipLocalizer;
                        if (lurch > maxLurchIRE) {
                            maxLurchIRE = lurch;
                            maxLurchSupport = std::clamp(view.membershipSupport, 0.0, 1.0);
                        }

                        const int sampleSign = (view.carrierSample > 0.0) ? +1 :
                                               ((view.carrierSample < 0.0) ? -1 : 0);
                        const bool signCompatible =
                            (referenceSign == 0 || sampleSign == 0 || sampleSign == referenceSign);
                        const double viewFitGate = 1.0 - smoothStep01(
                            (std::max(0.0, view.sampleFitErrorIRE) - 1.5) / 6.0);
                        const double viewLatticeGate = 1.0 - smoothStep01(
                            (std::max(0.0, view.latticeRiskIRE) - 1.5) / 6.0);
                        const double viewSupport = std::clamp(
                            0.35 + 0.65 * viewFitGate * viewLatticeGate,
                            0.0,
                            1.0);
                        if (signCompatible && viewSupport > 0.20) {
                            const double a = std::fabs(view.carrierSample);
                            if (a < minCompatibleAbs) {
                                minCompatibleAbs = a;
                                minCompatibleCarrierSample = view.carrierSample;
                            }
                            minCompatibleSupport = std::max(minCompatibleSupport, viewSupport);
                            ++compatibleResidualN;
                        }
                        ++residualN;
                    }

                    // Fifth residual witness: a centered/rolling legal 4fSC
                    // cancellation complement.  The four aperture views tell
                    // which legal windows including xi say what carrier remains;
                    // this rolling witness is aperture-independent in the sense
                    // that it is not one of the four ownership views being
                    // scored.  It is still a carrier-cancelling Y estimate, so
                    // raw - moving floor is mostly carrier plus whatever HF-Y
                    // the rolling window could not preserve.
                    double movingResidualSample = residualTightenSample;
                    double movingResidualFitErrorIRE = residualTightenFitErrorIRE;
                    double movingResidualCoherence = 0.0;
                    double movingResidualPull = 0.0;
                    if (meanCount > 0) {
                        const int centeredStart = std::clamp(xi - 1, 0, meanCount - 1);
                        movingResidualSample = rawWhole[xi] - winFloor[centeredStart];
                        residualCarrierLo = std::min(residualCarrierLo, movingResidualSample);
                        residualCarrierHi = std::max(residualCarrierHi, movingResidualSample);
                        sumResidual += movingResidualSample;
                        const double movingFitError = parallax.valid
                            ? std::fabs(movingResidualSample - parallax.commonSample) * invIreScale
                            : 0.0;
                        sumFitError += movingFitError;
                        movingResidualFitErrorIRE = movingFitError;

                        const int movingSign = (movingResidualSample > 0.0) ? +1 :
                                               ((movingResidualSample < 0.0) ? -1 : 0);
                        const bool movingSignCompatible =
                            (referenceSign == 0 || movingSign == 0 || movingSign == referenceSign);
                        const double movingSupport = 1.0 - smoothStep01(
                            (movingFitError - 1.5) / 6.0);
                        if (movingSignCompatible && movingSupport > 0.20) {
                            const double a = std::fabs(movingResidualSample);
                            if (a < minCompatibleAbs) {
                                minCompatibleAbs = a;
                                minCompatibleCarrierSample = movingResidualSample;
                            }
                            minCompatibleSupport = std::max(minCompatibleSupport, movingSupport);
                            ++compatibleResidualN;
                        }
                        ++residualN;
                    }

                    residualTightenSample =
                        sumResidual / static_cast<double>(std::max(1, residualN));
                    residualTightenFitErrorIRE =
                        sumFitError / static_cast<double>(std::max(1, residualN));
                    residualTightenSpreadIRE =
                        (residualCarrierHi - residualCarrierLo) * invIreScale;

                    const double variableYIRE = (maxFloor >= minFloor)
                        ? (maxFloor - minFloor) * invIreScale
                        : 0.0;
                    const double spreadGate = 1.0 - smoothStep01(
                        (residualTightenSpreadIRE - 2.0) / 8.0);
                    const double fitGate = 1.0 - smoothStep01(
                        (residualTightenFitErrorIRE - 1.5) / 6.0);
                    const double lurchGate = smoothStep01((maxLurchIRE - 0.75) / 5.0);
                    const double yVariableGate = smoothStep01((variableYIRE - 1.0) / 8.0);
                    residualTightenSupport = std::clamp(
                        0.35 + 0.65 * spreadGate * fitGate,
                        0.0,
                        1.0);
                    residualTightenGain = std::clamp(
                        0.20 + 0.55 * residualTightenSupport +
                        0.25 * std::max(lurchGate, yVariableGate),
                        0.0,
                        1.0);

                    movingResidualCoherence = 1.0 - std::clamp(
                        std::fabs(movingResidualSample - residualTightenSample) * invIreScale /
                        std::max(3.0, 0.35 * std::fabs(residualTightenSample) * invIreScale + 1.0),
                        0.0,
                        1.0);
                    movingResidualPull = movingResidualCoherence * residualTightenSupport;

                    lddecode::CarrierResidualConsensus consensus;
                    consensus.carrierSample = residualTightenSample;
                    consensus.lo = residualCarrierLo;
                    consensus.hi = residualCarrierHi;
                    consensus.trust = residualTightenSupport;
                    if (compatibleResidualN > 0 && minCompatibleAbs < 1e299) {
                        consensus.minCompatibleCarrierSample = minCompatibleCarrierSample;
                        consensus.minCompatibleAbsIRE = minCompatibleAbs * invIreScale;
                        consensus.minCompatibleSupport = std::clamp(
                            minCompatibleSupport *
                            (0.35 + 0.65 * std::max(lurchGate, yVariableGate)),
                            0.0,
                            1.0);
                    }
                    consensus.spreadIRE = residualTightenSpreadIRE;
                    consensus.lumaVarianceIRE = variableYIRE;
                    consensus.variableYIRE = std::max(variableYIRE, maxLurchIRE * maxLurchSupport);
                    consensus.fitErrorIRE = residualTightenFitErrorIRE;
                    consensus.valid = true;
                    parallax.residualConsensus = consensus;
                    parallax.movingResidualSample = movingResidualSample;
                    parallax.movingResidualFitErrorIRE = movingResidualFitErrorIRE;
                    parallax.movingResidualCoherence = movingResidualCoherence;
                    parallax.movingResidualPull = movingResidualPull;
                }

                parallax.residualTightenSample = residualTightenSample;
                parallax.residualTightenSupport = residualTightenSupport;
                parallax.residualTightenGain = residualTightenGain;
                parallax.residualTightenSpreadIRE = residualTightenSpreadIRE;
                parallax.residualTightenFitErrorIRE = residualTightenFitErrorIRE;
                if (parallaxRow)
                    parallaxRow[xi] = parallax;

                auto finalizeCarrierSample = [&](double candidateI) {
                    double sample = candidateI * basisI[xi] + modelQ * basisQ[xi];

                    if (parallax.valid) {
                        constexpr double SAMPLE_DISC_SOFT_IRE = 1.5;
                        constexpr double SAMPLE_DISC_HARD_IRE = 5.0;
                        constexpr double CONTEXT_SOFT_IRE = 3.0;
                        constexpr double CONTEXT_HARD_IRE = 12.0;

                        const double sampleDisagreementIRE =
                            std::fabs(sample - parallax.commonSample) * invIreScale;
                        const double disagreementGate = smoothStep01(
                            (sampleDisagreementIRE - SAMPLE_DISC_SOFT_IRE) /
                            std::max(1e-9, SAMPLE_DISC_HARD_IRE - SAMPLE_DISC_SOFT_IRE));
                        const double contextIRE =
                            std::max(parallax.yCurvatureIRE,
                                     0.5 * parallax.ySpreadIRE);
                        const double contextGate = smoothStep01(
                            (contextIRE - CONTEXT_SOFT_IRE) /
                            std::max(1e-9, CONTEXT_HARD_IRE - CONTEXT_SOFT_IRE));

                        // The per-pixel floor residual is not a replacement model;
                        // it is a witness that can pull a bad window IQ fit back
                        // toward what all legal Y floors say at this sample.
                        const double sampleTrust =
                            std::clamp(parallax.sampleCoherence, 0.0, 1.0);
                        const double sampleAnchor =
                            std::min(0.85,
                                     disagreementGate *
                                     sampleTrust *
                                     (0.25 + 0.75 * contextGate) *
                                     (1.0 - discResponseBlend));

                        sample = sample * (1.0 - sampleAnchor) +
                                 parallax.commonSample * sampleAnchor;
                    }

                    if (residualTightenSupport > 0.0)
                        sample = std::clamp(sample, residualCarrierLo, residualCarrierHi);

                    return std::clamp(sample, -maxCarrierSamples, maxCarrierSamples);
                };

                const double baselineModelI = modelI;
                const double baselineCf = finalizeCarrierSample(baselineModelI);
                double cf = baselineCf;

                carrierFit[xi] = cf;
                flattened[xi] = rawWhole[xi] - cf;

                fitRow[xi] = static_cast<float>(cf);
                retractedRow[xi] = static_cast<float>(flattened[xi]);
            }
        } else {
            for (int xi = 0; xi < width; ++xi) {
                const double cf = 0.0;
                carrierFit[xi] = cf;
                flattened[xi] = rawWhole[xi];
                fitRow[xi] = 0.0f;
                retractedRow[xi] = static_cast<float>(flattened[xi]);
                if (parallaxRow)
                    parallaxRow[xi] = lddecode::FourViewCarrierAttribution{};
            }
        }

        // Optional LS refit at luma edges, retained from the previous design.
        // It is now a secondary edge repair, not the primary saturated-fill
        // carrier estimator.
        {
            constexpr double EDGE_SOFT_IRE  = 3.0;
            constexpr double EDGE_HARD_IRE  = 10.0;
            constexpr double DISC_SOFT_IRE  = 1.0;
            constexpr double DISC_HARD_IRE  = 4.0;
            constexpr int    LS_HALF_WIN    = 2;

            double *edgeGate = slideMean4;
            bool anyEdge = false;

            for (int xi = 0; xi < width; ++xi) {
                const int xm = std::max(0, xi - 2);
                const int xp = std::min(width - 1, xi + 2);
                const double gradIRE =
                    std::fabs(refinedY[xp] - refinedY[xm]) * invIreScale;
                const double gate = std::clamp(
                    (gradIRE - EDGE_SOFT_IRE) /
                    std::max(1e-9, EDGE_HARD_IRE - EDGE_SOFT_IRE),
                    0.0, 1.0);
                edgeGate[xi] = gate;
                if (gate > 0.0)
                    anyEdge = true;
            }

            if (anyEdge) {
                for (int xi = 0; xi < width; ++xi) {
                    if (edgeGate[xi] <= 0.0)
                        continue;

                    const int a = std::max(0, xi - LS_HALF_WIN);
                    const int b = std::min(width - 1, xi + LS_HALF_WIN);

                    double sII = 0.0, sIQ = 0.0, sQQ = 0.0;
                    double sIY = 0.0, sQY = 0.0;

                    for (int k = a; k <= b; ++k) {
                        const double obs = rawWhole[k] - refinedY[k];
                        const double bI = basisI[k];
                        const double bQ = basisQ[k];

                        const double dist = std::fabs(static_cast<double>(k - xi));
                        const double w = 1.0 - 0.65 * std::min(
                            1.0, dist / std::max(1.0, static_cast<double>(LS_HALF_WIN)));

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

                    double lsFit = fitI * basisI[xi] + fitQ * basisQ[xi];
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
                        const double lumaM = refinedY[xm];
                        const double lumaP = refinedY[xp];
                        const double dir = (lumaP >= lumaM) ? 1.0 : -1.0;
                        const double lumaMid = 0.5 * (lumaM + lumaP);
                        const double brightOffsetIRE =
                            dir * (refinedY[xi] - lumaMid) * invIreScale;
                        const double brightSideGate = smoothStep01(
                            (brightOffsetIRE - T.LS_REFIT_BRIGHT_SIDE_SOFT_IRE) /
                            std::max(1e-9, T.LS_REFIT_BRIGHT_SIDE_HARD_IRE -
                                           T.LS_REFIT_BRIGHT_SIDE_SOFT_IRE));

                        const int brightIdx = (dir > 0.0) ? xp : xm;
                        const int brightJ = (brightIdx + 1 < width)
                            ? (brightIdx + 1)
                            : (brightIdx > 0 ? brightIdx - 1 : brightIdx);
                        const int xiJ = (xi + 1 < width)
                            ? (xi + 1)
                            : (xi > 0 ? xi - 1 : xi);
                        const double brightAmpIRE =
                            std::hypot(carrierFit[brightIdx],
                                       carrierFit[brightJ]) * invIreScale;
                        const double localAmpIRE =
                            std::hypot(carrierFit[xi],
                                       carrierFit[xiJ]) * invIreScale;
                        const double coloredBrightIRE =
                            std::max(brightAmpIRE, localAmpIRE);
                        const double brightColorGate = smoothStep01(
                            (coloredBrightIRE - T.LS_REFIT_BRIGHT_COLOR_START_IRE) /
                            std::max(1e-9, T.LS_REFIT_BRIGHT_COLOR_FULL_IRE -
                                           T.LS_REFIT_BRIGHT_COLOR_START_IRE));

                        brightColorProtect = brightSideGate * brightColorGate;
                    }

                    double parallaxProtect = 0.0;
                    if (parallaxRow && parallaxRow[xi].valid) {
                        const auto &p = parallaxRow[xi];
                        const double spreadT = std::clamp(
                            p.carrierSpreadIRE /
                            std::max(3.0, 0.35 * p.commonMagIRE + 1.0),
                            0.0,
                            1.0);
                        const double latticeT = std::clamp(
                            p.latticeRiskIRE /
                            std::max(3.0, 0.35 * p.commonMagIRE + 1.0),
                            0.0,
                            1.0);
                        parallaxProtect = std::max(
                            spreadT,
                            std::max(latticeT, 1.0 - p.carrierCoherence));
                    }

                    const double g =
                        edgeGate[xi] * discGate *
                        (1.0 - brightColorProtect) *
                        (1.0 - parallaxProtect);
                    gateRow[xi] = static_cast<float>(g);

                    if (g > 0.0) {
                        const double blended =
                            carrierFit[xi] * (1.0 - g) + lsFit * g;
                        carrierFit[xi] = blended;
                        flattened[xi] = rawWhole[xi] - blended;
                        fitRow[xi] = static_cast<float>(blended);
                        retractedRow[xi] = static_cast<float>(flattened[xi]);
                    }
                }
            }
        }

        // ---------------------------------------------------------------
        // Carrier-band Y attribution from the wide coherent fit.
        //
        // The narrow four-view carrier fit is the full-spectrum workprint.
        // The wide fit is not a carrier source, and it is not a confidence
        // flag.  It measures a specific transferable component:
        //
        //     carrierBandY = narrowWorkprintCarrier - wideCoherentCarrier
        //
        // where the narrow one-cycle explanation carries more carrier-band
        // energy than the longer coherent carrier explanation can sustain.
        // That component is attributed to Y before the carrier model is
        // promoted.  In model terms:
        //
        //     CarrierWorkprint = fit
        //     CarrierModel     = CarrierWorkprint - carrierBandY
        //     YModel           = raw - CarrierModel
        //
        // Lurch / membership-change evidence from the legal 4-sample averages
        // determines how much of the measured component may be transferred at
        // hard luma transitions.  This is deliberately upstream of retracted Y,
        // because retracted Y is only trustworthy after this attribution.
        // ---------------------------------------------------------------
        {
            if ((int)scratch_attrWideCarrier.size() < width) {
                scratch_attrWideCarrier.resize(width, 0.0);
                scratch_attrBandYClaim.resize(width, 0.0);
                scratch_attrMembershipY.resize(width, 0.0);
            }
            double *wideCarrierSample = scratch_attrWideCarrier.data();
            double *carrierBandYClaim = scratch_attrBandYClaim.data();
            double *membershipYIRE = scratch_attrMembershipY.data();

            std::fill(wideCarrierSample, wideCarrierSample + width, 0.0);
            std::fill(carrierBandYClaim, carrierBandYClaim + width, 0.0);
            std::fill(membershipYIRE, membershipYIRE + width, 0.0);

            constexpr int CC_WIN = 32;
            constexpr int CC_HALF = CC_WIN / 2;

            if (width >= CC_WIN) {
                // Gram matrix for uniform-weight LS over CC_WIN samples.
                // basisI/Q have period 4, so over 4N samples the sums are
                // just N times the single-period values.
                const int periods = CC_WIN / 4;
                double sII_1 = 0.0, sIQ_1 = 0.0, sQQ_1 = 0.0;
                for (int p = 0; p < 4; ++p) {
                    sII_1 += basisI4[p] * basisI4[p];
                    sIQ_1 += basisI4[p] * basisQ4[p];
                    sQQ_1 += basisQ4[p] * basisQ4[p];
                }
                const double gramII = sII_1 * periods;
                const double gramIQ = sIQ_1 * periods;
                const double gramQQ = sQQ_1 * periods;
                const double gramDet = gramII * gramQQ - gramIQ * gramIQ;

                if (std::fabs(gramDet) > 1e-9) {
                    const double gramInv = 1.0 / gramDet;

                    // Initial window [0, CC_WIN-1].
                    double sIY = 0.0, sQY = 0.0;
                    for (int k = 0; k < CC_WIN; ++k) {
                        const double obs = rawWhole[k] - coarseY[k];
                        sIY += basisI[k] * obs;
                        sQY += basisQ[k] * obs;
                    }

                    for (int xi = 0; xi < width; ++xi) {
                        int wantA = xi - CC_HALF;
                        int wantB = xi + CC_HALF - 1;
                        if (wantA < 0) { wantB += -wantA; wantA = 0; }
                        if (wantB >= width) {
                            const int ov = wantB - (width - 1);
                            wantB -= ov;
                            wantA -= ov;
                            if (wantA < 0) wantA = 0;
                        }

                        if (xi > 0) {
                            int prevA = (xi - 1) - CC_HALF;
                            int prevB = (xi - 1) + CC_HALF - 1;
                            if (prevA < 0) { prevB += -prevA; prevA = 0; }
                            if (prevB >= width) {
                                const int ov = prevB - (width - 1);
                                prevB -= ov;
                                prevA -= ov;
                                if (prevA < 0) prevA = 0;
                            }

                            for (int k = prevA; k < wantA; ++k) {
                                const double obs = rawWhole[k] - coarseY[k];
                                sIY -= basisI[k] * obs;
                                sQY -= basisQ[k] * obs;
                            }
                            for (int k = prevB + 1; k <= wantB; ++k) {
                                const double obs = rawWhole[k] - coarseY[k];
                                sIY += basisI[k] * obs;
                                sQY += basisQ[k] * obs;
                            }
                        }

                        const double wideI = ( gramQQ * sIY - gramIQ * sQY) * gramInv;
                        const double wideQ = (-gramIQ * sIY + gramII * sQY) * gramInv;
                        const double wideSample = wideI * basisI[xi] + wideQ * basisQ[xi];
                        wideCarrierSample[xi] = wideSample;

                        const double narrowSample = carrierFit[xi];
                        const int xi1 = std::min(xi + 1, width - 1);
                        const double fitMagIRE =
                            std::hypot(static_cast<double>(fitRow[xi]),
                                       static_cast<double>(fitRow[xi1])) * invIreScale;
                        const double wideMagIRE = std::hypot(wideI, wideQ) * invIreScale;

                        // Membership/lurch evidence: this is not edge detection.
                        // It measures how much the legal carrier-cancelling
                        // averages change when a pixel enters/leaves the 4-sample
                        // cancellation window.  That component is attributed to
                        // luma movement through the coarse aperture.
                        double lurchIRE = 0.0;
                        for (int v = 0; v < evidenceRow[xi].viewCount; ++v) {
                            const auto &ev = evidenceRow[xi].views[v];
                            const double localizer = std::exp(
                                -0.5 * (static_cast<double>(ev.membershipLocalX) *
                                        static_cast<double>(ev.membershipLocalX))
                                     / (1.35 * 1.35));
                            const double l =
                                std::fabs(static_cast<double>(ev.membershipDeltaIRE)) *
                                std::clamp(static_cast<double>(ev.membershipSupport), 0.0, 1.0) *
                                localizer;
                            lurchIRE = std::max(lurchIRE, l);
                        }
                        membershipYIRE[xi] = lurchIRE;

                        // The old cross-color detector used the one-sided
                        // magnitude relation wide < narrow.  Keep that as the
                        // measurement of carrier-band Y excess, but convert it
                        // into a waveform component rather than a confidence flag.
                        double excessFraction = 0.0;
                        // Cross-color witness: widen detection so that the
                        // narrow regime is more visible.  Lower the fit
                        // magnitude threshold from 2.0 to 1.5 and reduce the
                        // normalisation floor accordingly.  When the narrow
                        // fit sees a materially larger carrier than the wide
                        // coherent fit, interpret part of that difference as
                        // luma trespass in the carrier band.
                        if (fitMagIRE > 1.5 && wideMagIRE < fitMagIRE) {
                            const double excessIRE = fitMagIRE - wideMagIRE;
                            excessFraction = std::clamp(
                                excessIRE / std::max(1.5, fitMagIRE),
                                0.0, 1.0);
                        }

                        // Lurch does not invent a carrier-band Y component by
                        // itself; it permits more of the measured wide-vs-narrow
                        // component to be transferred in hard luma transitions.
                        // Soften the gating by decreasing the lurch threshold
                        // and scaling factor: begin transferring earlier when
                        // lurch evidence is modest and allow a higher base
                        // transfer rate.
                        const double membershipFraction = smoothStep01((lurchIRE - 0.50) / 4.0);
                        const double transferFraction = std::clamp(
                            excessFraction * (0.50 + 0.50 * membershipFraction),
                            0.0,
                            0.95);

                        double yClaim = (narrowSample - wideSample) * transferFraction;

                        // Preserve sign and prevent the attribution transfer from
                        // crossing the workprint through zero in a single pass.
                        const double maxClaim = 0.90 * std::fabs(narrowSample);
                        yClaim = std::clamp(yClaim, -maxClaim, maxClaim);

                        carrierBandYClaim[xi] = yClaim;

                        // carrierImpurity is a detection channel, not a residual
                        // accounting: the witness patch gates consume it to
                        // decide 1D authority, so it must publish the full
                        // measured excess.  The double-count with the transfer
                        // above (produceY also suppresses alpha by this impurity)
                        // is resolved at the consumption site — pixels the
                        // witness claims via carrierCorrectionMask bypass the
                        // alpha suppression entirely, and elsewhere the
                        // over-suppression errs toward preserving luma, which
                        // is the documented intent (cross_color_suppression.md).
                        impurityRow[xi] = static_cast<float>(excessFraction);
                    }
                }
            }

            // Apply the attribution transfer to the carrier workprint before
            // Pass 2 promotion.  This is not a trust/distrust action; it moves
            // a measured waveform component out of the carrier model and into
            // the subsequent retracted-Y model.
            for (int xi = 0; xi < width; ++xi) {
                const double yClaim = carrierBandYClaim[xi];
                if (yClaim == 0.0)
                    continue;

                double sample = carrierFit[xi] - yClaim;

                // The five residual complements still define the local surviving
                // carrier interval.  Keep the transfer inside that interval when
                // the attribution record provides one.
                if (parallaxRow && parallaxRow[xi].valid &&
                    parallaxRow[xi].residualConsensus.valid)
                {
                    const auto &rc = parallaxRow[xi].residualConsensus;
                    const double padIRE = 0.75 + 2.0 * (1.0 - std::clamp(rc.trust, 0.0, 1.0));
                    const double padSamples = padIRE / std::max(1e-12, invIreScale);
                    const double lo = std::min(rc.lo, rc.hi) - padSamples;
                    const double hi = std::max(rc.lo, rc.hi) + padSamples;
                    sample = std::clamp(sample, lo, hi);

                    // The min-compatible survivor stays recorded in the
                    // consensus as attribution evidence, but it is not applied
                    // as a clamp or attractor here.  Hard limits on the model
                    // must describe the possible range of values — which the
                    // lo/hi interval above does, spanning all five residual
                    // complements — never a bound particular to one source.
                    // The smallest single view is often the most contaminated
                    // one at compact chroma (a straddling aperture), so
                    // ceiling the model to it gutted real patch carrier.

                    auto p2 = parallaxRow[xi];
                    p2.residualConsensus.workprintSample = carrierFit[xi];
                    p2.residualConsensus.workprintCorrectionIRE =
                        std::fabs(sample - carrierFit[xi]) * invIreScale;
                    parallaxRow[xi] = p2;
                }

                // At large luma transitions the fit tends to ring and smear into
                // the luma domain.  Suppress these echoes by attenuating the
                // carrier model when a coarse-Y jump is observed.  Compute a
                // local luma jump from the refined Y scaffold (two- and four-
                // sample differences) and apply a gate to reduce the sample
                // magnitude.  The gate ramps from 0 at small jumps (<3 IRE)
                // to 1 at very large jumps (>12 IRE).  A 60% attenuation at
                // full gate reduces ringing but preserves narrow colour
                // features.
                {
                    const int xm1 = (xi > 0) ? xi - 1 : 0;
                    const int xp1 = (xi + 1 < width) ? xi + 1 : width - 1;
                    const int xm2 = (xi > 1) ? xi - 2 : 0;
                    const int xp2 = (xi + 2 < width) ? xi + 2 : width - 1;
                    const double d1 = std::fabs(refinedY[xp1] - refinedY[xm1]);
                    const double d2 = 0.75 * std::fabs(refinedY[xp2] - refinedY[xm2]);
                    const double coarseJumpIRE = std::max(d1, d2) * invIreScale;
                    const double jumpGate = smoothStep01((coarseJumpIRE - 3.0) / 9.0);
                    const double atten = 1.0 - 0.60 * jumpGate;
                    sample *= atten;
                }

                sample = std::clamp(sample, -maxCarrierSamples, maxCarrierSamples);
                carrierFit[xi] = sample;
                flattened[xi] = rawWhole[xi] - sample;
                fitRow[xi] = static_cast<float>(sample);
                retractedRow[xi] = static_cast<float>(flattened[xi]);
            }
        }

        // Build the carrier-cancelled floor from every legal 4-sample mean of
        // the final flattened waveform.
        if (width >= 4) {
            const int meanCount = width - 3;
            for (int s = 0; s < meanCount; ++s) {
                slideMean4[s] =
                    0.25 * (flattened[s + 0] +
                            flattened[s + 1] +
                            flattened[s + 2] +
                            flattened[s + 3]);
            }

            for (int xi = 0; xi < width; ++xi) {
                double v[4] = {0.0, 0.0, 0.0, 0.0};
                int n = 0;
                const int sFirst = std::max(0, xi - 3);
                const int sLast  = std::min(xi, meanCount - 1);

                for (int s = sFirst; s <= sLast && n < 4; ++s)
                    v[n++] = slideMean4[s];

                double floor = refinedY[xi];
                if (n >= 4)
                    floor = medoid4Anchored(v[0], v[1], v[2], v[3], refinedY[xi]);
                else if (n == 3)
                    floor = median3(v[0], v[1], v[2]);
                else if (n == 2)
                    floor = 0.5 * (v[0] + v[1]);
                else if (n == 1)
                    floor = v[0];

                floorRow[xi] = static_cast<float>(floor);
            }
        } else {
            double mean = 0.0;
            for (int xi = 0; xi < width; ++xi)
                mean += flattened[xi];
            mean /= static_cast<double>(std::max(1, width));

            for (int xi = 0; xi < width; ++xi)
                floorRow[xi] = static_cast<float>(mean);
        }
    }

    // ---------------------------------------------------------------
    // Pass 2: line-to-line cancellation on carrierFit → combedCarrier.
    // ---------------------------------------------------------------
    const bool progressiveCarrierModel =
        (cadenceId >= 0 || cadenceId == CADENCE_PROGRESSIVE);

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
            (lineBelow < lastLine) ? carrierGrammarLine(lineBelow) : nullptr;

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

        // Reach gate: determines per-pixel cancellation strength toward
        // each neighbor line.  Inlined here (was a lambda) to keep the
        // data flow visible in the fused loop below.
        //
        // Mismatch and amplitude are evaluated as 2-sample quadrature
        // envelopes, never instantaneous samples: |fit + neighbor| dips to
        // zero twice per carrier cycle, so a per-sample gate oscillates at
        // carrier rate.  On slanted chroma (~2 px/line drift = 180° carrier
        // rotation per line) the opposite-lineFlip neighbor presents
        // SAME-SIGN chroma; the envelope gate sees a constant 2A mismatch
        // and stands the cancellation down consistently, where the
        // per-sample gate cancelled real chroma in carrier-rate bursts —
        // a checkerboard manufactured inside combedCarrier itself.
        auto reachGate = [&](int xi, const float *neighborFit,
                             const float *neighborFloor) {
            if (!neighborFit || !neighborFloor)
                return 0.0;

            const double lumaDiffIRE =
                std::fabs(static_cast<double>(floorRow[xi]) -
                          static_cast<double>(neighborFloor[xi])) * invIreScale;

            const int xj = std::min(xi + 1, width - 1);
            const double c0 = static_cast<double>(fitRow[xi]);
            const double c1 = static_cast<double>(fitRow[xj]);
            const double n0 = static_cast<double>(neighborFit[xi]);
            const double n1 = static_cast<double>(neighborFit[xj]);

            const double carrierMismatchIRE =
                std::hypot(c0 + n0, c1 + n1) * invIreScale;
            const double carrierAmpIRE = 0.5 *
                (std::hypot(c0, c1) + std::hypot(n0, n1)) * invIreScale;

            double lumaGate = softReachGate(lumaDiffIRE, 3.0, 10.0);

            const double carrierSoftIRE = std::max(3.0, 0.25 * carrierAmpIRE);
            const double carrierHardIRE = std::max(10.0, 0.80 * carrierAmpIRE);
            double carrierGate =
                softReachGate(carrierMismatchIRE,
                              carrierSoftIRE,
                              carrierHardIRE);

            if (progressiveCarrierModel) {
                const double chromaT = std::clamp(
                    (carrierAmpIRE -
                     T.RETRACTED_PROGRESSIVE_CHROMA_RELAX_START_IRE) /
                    std::max(1e-9,
                        T.RETRACTED_PROGRESSIVE_CHROMA_RELAX_FULL_IRE -
                        T.RETRACTED_PROGRESSIVE_CHROMA_RELAX_START_IRE),
                    0.0, 1.0);
                const double satRelax = chromaT * chromaT * (3.0 - 2.0 * chromaT);
                const double bevelT = std::clamp(
                    (lumaDiffIRE - 3.0) / 7.0, 0.0, 1.0);
                const double bevelGuard =
                    1.0 - (bevelT * bevelT * (3.0 - 2.0 * bevelT));
                const double relax = satRelax * bevelGuard;
                lumaGate = std::max(
                    lumaGate,
                    T.RETRACTED_PROGRESSIVE_LUMA_GATE_FLOOR * relax);
                carrierGate = std::max(
                    carrierGate,
                    T.RETRACTED_PROGRESSIVE_CARRIER_GATE_FLOOR * relax);
            }

            return lumaGate * carrierGate;
        };

        // -----------------------------------------------------------------
        // Fused reach-gate sweep.
        //
        // The original three-pass design was:
        //   Sweep 1: compute raw gates and decisionBlend into per-pixel arrays
        //   Sweep 2: 5-tap [1,2,3,2,1] smooth of raw gates → smooth arrays
        //   Sweep 3: blend raw/smooth by decisionBlend, produce combRow
        //
        // Fused into two passes: first compute the raw gates (needed as
        // lookahead input for the stencil), then a single output pass that
        // evaluates the 5-tap smooth, the decision blend, and the final
        // combRow per pixel — eliminating 3 scratch arrays and 2 loop
        // traversals.
        // -----------------------------------------------------------------
        double *wAboveRaw = scratch_preI.data();
        double *wBelowRaw = scratch_preQ.data();

        // Pass A: raw gates — must be fully materialized before the stencil
        // can read ±2 neighbors.
        for (int xi = 0; xi < width; ++xi) {
            wAboveRaw[xi] =
                haveAbove ? reachGate(xi, fitAbove, floorAbove) : 0.0;
            wBelowRaw[xi] =
                haveBelow ? reachGate(xi, fitBelow, floorBelow) : 0.0;
        }

        // Pass B: inline 5-tap smooth + decision blend + combRow output.
        // The smooth kernel is [1,2,3,2,1] (sum = 9 in the interior).
        // At edges, the kernel is clamped and the divisor adjusts.
        constexpr double kWeights[5] = {1.0, 2.0, 3.0, 2.0, 1.0};

        for (int xi = 0; xi < width; ++xi) {
            // Inline 5-tap smooth of the raw gate arrays.
            double sumW = 0.0, sumAbove = 0.0, sumBelow = 0.0;
            for (int dx = -2; dx <= 2; ++dx) {
                const int xx = std::clamp(xi + dx, 0, width - 1);
                const double w = kWeights[dx + 2];
                sumW += w;
                sumAbove += w * wAboveRaw[xx];
                sumBelow += w * wBelowRaw[xx];
            }
            const double smoothAbove = sumAbove / sumW;
            const double smoothBelow = sumBelow / sumW;

            // Inline localCarrierAmpIRE + decision blend.
            const int xm = std::max(0, xi - 1);
            const int xp = std::min(width - 1, xi + 1);
            double amp = std::fabs(static_cast<double>(fitRow[xi]));
            amp = std::max(amp, std::fabs(static_cast<double>(fitRow[xm])));
            amp = std::max(amp, std::fabs(static_cast<double>(fitRow[xp])));
            if (fitAbove) {
                amp = std::max(amp, std::fabs(static_cast<double>(fitAbove[xi])));
                amp = std::max(amp, std::fabs(static_cast<double>(fitAbove[xm])));
                amp = std::max(amp, std::fabs(static_cast<double>(fitAbove[xp])));
            }
            if (fitBelow) {
                amp = std::max(amp, std::fabs(static_cast<double>(fitBelow[xi])));
                amp = std::max(amp, std::fabs(static_cast<double>(fitBelow[xm])));
                amp = std::max(amp, std::fabs(static_cast<double>(fitBelow[xp])));
            }
            const double blend = smoothStep01((amp * invIreScale - 8.0) / 10.0);

            // Final gated cancellation.
            const double wAbove =
                wAboveRaw[xi] * (1.0 - blend) + smoothAbove * blend;
            const double wBelow =
                wBelowRaw[xi] * (1.0 - blend) + smoothBelow * blend;
            const double wSum = wAbove + wBelow;

            if (wSum > 1e-9) {
                double neighborFit = 0.0;
                if (wAbove > 0.0 && fitAbove)
                    neighborFit += wAbove * static_cast<double>(fitAbove[xi]);
                if (wBelow > 0.0 && fitBelow)
                    neighborFit += wBelow * static_cast<double>(fitBelow[xi]);
                neighborFit /= wSum;

                // Strength-scaled cancellation, not a hard switch.  For
                // genuinely inverting chroma 0.5*(C - (-C)) = C at any
                // strength, so this is amplitude-neutral where the comb is
                // right; where the neighbor does NOT invert (slanted chroma,
                // pedestal) a partial gate now yields partial retention
                // instead of full cancellation of real signal.
                const double cancelled =
                    0.5 * (static_cast<double>(fitRow[xi]) - neighborFit);
                const double strength = std::min(1.0, wSum);

                combRow[xi] = static_cast<float>(
                    static_cast<double>(fitRow[xi]) * (1.0 - strength) +
                    cancelled * strength);
            } else {
                combRow[xi] = fitRow[xi];
            }
        }
    }

    // ---------------------------------------------------------------
    // Final publication: retracted Y and final floor derive from the
    // promoted carrier model, not from the workprint fit.
    // ---------------------------------------------------------------
    for (int line = firstLine; line < lastLine; ++line) {
        const quint16 *rawLine =
            rawbuffer.data() + static_cast<size_t>(line) * videoParameters.fieldWidth;
        const float *combRow = combedCarrier_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        float *retractedRow = carrierRetracted_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        float *floorRow = flatFloor_flat.data()
                          + static_cast<size_t>(line) * demodWidth;

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
            refinedY[xi] = baseY4Src[xi];
            flattened[xi] = rawWhole[xi] - static_cast<double>(combRow[xi]);
            retractedRow[xi] = static_cast<float>(flattened[xi]);
        }

        if (width >= 4) {
            const int meanCount = width - 3;
            for (int s = 0; s < meanCount; ++s) {
                slideMean4[s] =
                    0.25 * (flattened[s + 0] +
                            flattened[s + 1] +
                            flattened[s + 2] +
                            flattened[s + 3]);
            }

            for (int xi = 0; xi < width; ++xi) {
                double v[4] = {0.0, 0.0, 0.0, 0.0};
                int n = 0;
                const int sFirst = std::max(0, xi - 3);
                const int sLast  = std::min(xi, meanCount - 1);

                for (int s = sFirst; s <= sLast && n < 4; ++s)
                    v[n++] = slideMean4[s];

                double floor = refinedY[xi];
                if (n >= 4)
                    floor = medoid4Anchored(v[0], v[1], v[2], v[3], refinedY[xi]);
                else if (n == 3)
                    floor = median3(v[0], v[1], v[2]);
                else if (n == 2)
                    floor = 0.5 * (v[0] + v[1]);
                else if (n == 1)
                    floor = v[0];

                floorRow[xi] = static_cast<float>(floor);
            }
        } else {
            double mean = 0.0;
            for (int xi = 0; xi < width; ++xi)
                mean += flattened[xi];
            mean /= static_cast<double>(std::max(1, width));
            for (int xi = 0; xi < width; ++xi)
                floorRow[xi] = static_cast<float>(mean);
        }
    }

    carrierRetractedValid = true;
}
