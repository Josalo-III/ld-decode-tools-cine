/******************************************************************************
 * comb.cpp
 * ld-chroma-decoder — Colourisation filter for ld-decode
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2018 Chad Page
 * SPDX-FileCopyrightText: 2018-2019 Simon Inns
 * SPDX-FileCopyrightText: 2020-2021 Adam Sampson
 * SPDX-FileCopyrightText: 2021 Phillip Blucas
 * SPDX-FileCopyrightText: 2025-2026 Joseph Burns
 *
 * This file is part of ld-decode-tools.
 ******************************************************************************/

#include "cadencedefs.h"
#include "comb.h"
#include "combmath.h"
#include "framecanvas.h"
#include "deemp.h"
#include "firfilter.h"

#include <QElapsedTimer>
#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <mutex>
#include <numeric>
#include <utility>
#include <vector>
#include <cstring>

namespace {
    // chroma amplitude normalization
    constexpr double BUCKET_CHROMA_SCALE = 1.4;
    constexpr double PRODUCT_CHROMA_SCALE = 1.33;

    inline double median4_average_middle(double a, double b, double c, double d)
    {
        if (a > b) std::swap(a, b);
        if (c > d) std::swap(c, d);
        if (a > c) std::swap(a, c);
        if (b > d) std::swap(b, d);
        if (b > c) std::swap(b, c);
        return 0.5 * (b + c);
    }
}

// 3D candidate palette
enum CandidateIndex : qint32 {
    CAND_LEFT,
    CAND_RIGHT,
    CAND_UP,
    CAND_DOWN,
    CAND_PREV_FIELD,
    CAND_NEXT_FIELD,
    CAND_PREV_FRAME,
    CAND_NEXT_FRAME,
    NUM_CANDIDATES
};

// Map colours for the candidates
static constexpr quint32 CANDIDATE_SHADES[] = {
    0xFF8080, // CAND_LEFT - red
    0xFF8080, // CAND_RIGHT - red
    0xFFFF80, // CAND_UP - yellow
    0xFFFF80, // CAND_DOWN - yellow
    0x80FF80, // CAND_PREV_FIELD - green
    0x80FF80, // CAND_NEXT_FIELD - green
    0x8080FF, // CAND_PREV_FRAME - blue
    0xFF80FF, // CAND_NEXT_FRAME - purple
};
// Render a single character from a minimal 57 bitmap font into a FrameCanvas.
// Supports pulldown film letters, '?' (unknown), and '/' (boundary marker).
// scale controls pixel block size for visibility at different output resolutions.
static void drawChar(FrameCanvas &canvas, int x, int y, char ch, FrameCanvas::Colour col, int scale) {
    // Simple 5x7 font map for A-D, ?, /, i, p
    static const unsigned char font[][7] = {
        {0x04,0x0A,0x11,0x11,0x1F,0x11,0x11}, // A (0)
        {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}, // B (1)
        {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}, // C (2)
        {0x1E,0x11,0x11,0x11,0x11,0x11,0x1E}, // D (3)
        {0x0E,0x11,0x01,0x02,0x04,0x00,0x04}, // ? (4)
        {0x01,0x02,0x02,0x04,0x04,0x08,0x10}, // / (5)
        {0x04,0x00,0x04,0x04,0x04,0x04,0x0E}, // i (6)
        {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10}  // p (7) - rendered as P
    };
    
    int idx = 4; // default to '?'
    if (ch >= 'A' && ch <= 'D') idx = ch - 'A';
    else if (ch == '?' ) idx = 4;
    else if (ch == '/') idx = 5;
    else if (ch == 'i' || ch == 'I') idx = 6;
    else if (ch == 'p' || ch == 'P') idx = 7;
    
    for (int r = 0; r < 7; ++r) {
        unsigned char row = font[idx][r];
        for (int c = 0; c < 5; ++c) {
            if ((row >> (4-c)) & 1) {
                // Draw scaled blocks for visibility
                canvas.drawRectangle(x + c*scale, y + r*scale, scale, scale, col);
            }
        }
    }
}

// Demodulate a single composite sample v at horizontal position h into I and Q
// using the locked basis LUTs (spLUT, cpLUT) and the per-line burst phasor
// (bcos, bsin). Writes the result into outI[xi] and outQ[xi].
inline void demodSample(double v, int phaseIdx, int xi,
                        double bcos, double bsin,
                        const double* spLUT, const double* cpLUT,
                        float* outI, float* outQ)
{
    const int idx = phaseIdx & 3;
    const double sp = spLUT[idx];
    const double cp = cpLUT[idx];

    const double lsin_r = v * sp * 2.0;
    const double lcos_r = v * cp * 2.0;
    const double ri     = (lsin_r * bcos - lcos_r * bsin);
    const double rq     = (lsin_r * bsin + lcos_r * bcos);

    outI[xi] = (float)ri;
    outQ[xi] = (float)rq;
}

// Comb public
Comb::Comb() : configurationSet(false) {}

qint32 Comb::Configuration::getLookBehind() const { return (dimensions == 3) ? 1 : 0; }
qint32 Comb::Configuration::getLookAhead()  const { return (dimensions == 3) ? 1 : 0; }

const Comb::Configuration &Comb::getConfiguration() const { return configuration; }

void Comb::updateConfiguration(const LdDecodeMetaData::VideoParameters &_videoParameters,
                               const Configuration &_configuration)
{
    videoParameters = _videoParameters;
    configuration   = _configuration;

    if (videoParameters.fieldWidth > MAX_WIDTH)
        qCritical() << "Comb: width exceeds maximum";
    if (((videoParameters.fieldHeight * 2) - 1) > MAX_HEIGHT)
        qCritical() << "Comb: height exceeds maximum";
    if (videoParameters.activeVideoStart < 16)
        qCritical() << "Comb: activeVideoStart must be >= 16";
    if (std::fabs((videoParameters.sampleRate / videoParameters.fSC) - 4.0) > 1e-6)
        qCritical() << "Comb: sample rate not ~4*fSC (colour decode may fail)";

    configurationSet = true;
}

// Orchestrates per-frame decoding across all requested frames. Maintains a
// rolling triple-buffer (previous / current / next) so that 3D temporal candidates
// always have access to both neighbours.
void Comb::decodeFrames(const QVector<SourceField> &inputFields,
                        qint32 startIndex, qint32 endIndex,
                        QVector<ComponentFrame> &componentFrames)
{
    assert(configurationSet);
    assert(componentFrames.size() * 2 == (endIndex - startIndex));

    enum StageIndex {
        StagePhaseLocked,
        StagePhaseCorrected1D,
        StageSplit2D,
        StageCopy2DTo3D,
        StageSplit3D,
        StageSplitIQLocked,
        StageDoCNR,
        StageProduceY,
        StageFilterIQLocked,
        StageDoYNR,
        StageTransformIQ,
        StageCount
    };
    struct StageStat {
        const char *name;
        qint64 totalNs = 0;
        qint64 calls = 0;
    };
    std::array<StageStat, StageCount> stageStats = {{
        {"phaseLocked"},
        {"buildPhaseCorrected1D"},
        {"split2D"},
        {"copy2DTo3D"},
        {"split3D"},
        {"splitIQlocked"},
        {"doCNR"},
        {"produceY"},
        {"filterIQLocked"},
        {"doYNR"},
        {"transformIQ"},
    }};
    const bool stageTimers = configuration.stageTimers && configuration.phaseCompensation;
    FrameBuffer::FvfInstrumentation fvfStatsTotal;
    auto accumulateFvfStats = [&](const FrameBuffer::FvfInstrumentation &stats) {
        for (int i = 0; i < 4; ++i) {
            fvfStatsTotal.rawWinnerCounts[i] += stats.rawWinnerCounts[i];
            fvfStatsTotal.finalWinnerCounts[i] += stats.finalWinnerCounts[i];
        }
        fvfStatsTotal.frameAHeadToHeadWins += stats.frameAHeadToHeadWins;
        fvfStatsTotal.frameBHeadToHeadWins += stats.frameBHeadToHeadWins;
        fvfStatsTotal.frameModelPixels += stats.frameModelPixels;
        fvfStatsTotal.fieldModelPixels += stats.fieldModelPixels;
        fvfStatsTotal.islandChangedPixels += stats.islandChangedPixels;
        fvfStatsTotal.blockFieldCommitPixels += stats.blockFieldCommitPixels;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                fvfStatsTotal.islandFlipPairs[r][c] += stats.islandFlipPairs[r][c];
    };
    auto measureStage = [&](StageIndex idx, auto &&fn) {
        if (!stageTimers) {
            fn();
            return;
        }
        QElapsedTimer timer;
        timer.start();
        fn();
        stageStats[idx].totalNs += timer.nsecsElapsed();
        stageStats[idx].calls += 1;
    };

    auto next     = std::make_unique<FrameBuffer>(videoParameters, configuration);
    auto current  = std::make_unique<FrameBuffer>(videoParameters, configuration);
    auto previous = std::make_unique<FrameBuffer>(videoParameters, configuration);

    const qint32 preStart = (configuration.dimensions == 3)
        ? (startIndex - 4)
        : (startIndex - 2);

    for (qint32 fieldIndex = preStart; fieldIndex < endIndex; fieldIndex += 2) {
        // Rotate buffers.
        {
            auto recycle = std::move(previous);
            previous = std::move(current);
            current  = std::move(next);
            next     = std::move(recycle);
        }

        const bool canLoadNext =
            (fieldIndex + 2 >= 0) &&
            (fieldIndex + 3 < inputFields.size());

        if (canLoadNext) {
            next->loadFields(inputFields[fieldIndex + 2],
                             inputFields[fieldIndex + 3]);

            next->split1D();

            if (configuration.phaseCompensation) {
                measureStage(StagePhaseLocked, [&]() { next->phaseLocked(); });
                measureStage(StagePhaseCorrected1D, [&]() { next->buildPhaseCorrected1D(); });
            }

            measureStage(StageSplit2D, [&]() { next->split2D(); });
            if (stageTimers &&
                configuration.twoDVariant ==
                    Comb::Configuration::TwoDVariant::FieldVsFrame) {
                accumulateFvfStats(next->getFvfInstrumentation());
            }
        }

        if (fieldIndex < startIndex)
            continue;

        const bool isStartUp = (fieldIndex < startIndex + 4);

        if (configuration.dimensions == 3) {
            measureStage(StageCopy2DTo3D, [&]() { current->copy2DTo3D(); });

            if (!isStartUp)
                measureStage(StageSplit3D, [&]() { current->split3D(*previous, *next); });
        }

        if (configuration.residualVideo3D) {
            if (!isStartUp) {
                current->prevFrameForVet = previous.get();
                current->nextFrameForVet = next.get();
            } else {
                current->prevFrameForVet = nullptr;
                current->nextFrameForVet = nullptr;
            }
        }

        const qint32 frameIndex = (fieldIndex - startIndex) / 2;
        componentFrames[frameIndex].init(videoParameters);
        current->setComponentFrame(componentFrames[frameIndex]);

        /*
         * Output path.
         *
         * splitIQlocked() is the post-election demod of the selected comb.
         */
        if (configuration.phaseCompensation) {
            measureStage(StageSplitIQLocked, [&]() { current->splitIQlocked(); });
            measureStage(StageDoCNR, [&]() { current->doCNR(); });
            measureStage(StageProduceY, [&]() { current->produceY(); });
            measureStage(StageFilterIQLocked, [&]() { current->filterIQLocked(); });
            measureStage(StageDoYNR, [&]() { current->doYNR(); });
            measureStage(StageTransformIQ, [&]() {
                current->transformIQ(configuration.chromaGain,
                                     configuration.chromaPhase);
            });
        } else {
            current->splitIQ();
            current->adjustY();
            current->filterIQ();
            current->doCNR();
            current->doYNR();
            current->transformIQ(configuration.chromaGain,
                                 configuration.chromaPhase);
        }

        if (configuration.dimensions == 3 && configuration.showMap)
            current->overlayMap(*previous, *next);

        // --- Visual Debug Overlays: Cadence / Film vs Video ------------------------
        if (configuration.debugCadence) {
            FrameCanvas canvas(componentFrames[frameIndex], videoParameters);

            int cidTop    = -1;
            int cidBottom = -1;
            bool editTop    = false;
            bool editBottom = false;

            if (fieldIndex < inputFields.size()) {
                cidTop  = inputFields[fieldIndex].field.cinemap.cadenceId;
                editTop = inputFields[fieldIndex].field.cinemap.isEditBoundary;
            }

            if (fieldIndex + 1 < inputFields.size()) {
                cidBottom = inputFields[fieldIndex + 1].field.cinemap.cadenceId;
                editBottom = inputFields[fieldIndex + 1].field.cinemap.isEditBoundary;
            }

            const int scale = 4;
            const int charW = 5 * scale;
            const int charH = 7 * scale;
            const int pad   = 4;
            const int boxH  = charH + 2 * pad;

            const int xBase = videoParameters.activeVideoStart + 32;
            const int yBase = videoParameters.firstActiveFrameLine + 32;

            FrameCanvas::Colour fg = canvas.rgb(65535, 65535, 65535);
            FrameCanvas::Colour bg = canvas.rgb(0, 0, 0);

            auto fieldLabel = [&](int cid, int fallbackCyclePos) -> char {
                if (cid == lddecode::kCadenceVideo)
                    return 'i';

                if (cid == lddecode::kCadenceProgressive)
                    return 'p';

                if (cadenceKnown(cid))
                    return cadenceFilmLetter(cid);

                return static_cast<char>('0' + ((fallbackCyclePos % 5) + 1));
            };

            const int cyclePos = frameIndex;
            const char labelTop    = fieldLabel(cidTop, cyclePos);
            const char labelBottom = fieldLabel(cidBottom, cyclePos);

            const bool pureFrame =
                (labelTop == labelBottom) &&
                (cidTop >= lddecode::kCadenceProgressive) &&
                (cidBottom >= lddecode::kCadenceProgressive) &&
                !editTop &&
                !editBottom;

            int numChars = 0;

            if (editTop)
                ++numChars; // leading '/'

            ++numChars;     // top label

            if (!pureFrame) {
                if (editBottom && !editTop)
                    ++numChars; // middle '/'

                ++numChars;     // bottom label
            }

            const int totalW = pad + numChars * (charW + pad);
            canvas.fillRectangle(xBase, yBase, totalW, boxH, bg);

            int xOff = xBase + pad;

            auto drawNext = [&](char c) {
                drawChar(canvas, xOff, yBase + pad, c, fg, scale);
                xOff += charW + pad;
            };

            if (editTop)
                drawNext('/');

            drawNext(labelTop);

            if (!pureFrame) {
                if (editBottom && !editTop)
                    drawNext('/');

                drawNext(labelBottom);
            }
        }
    }

    if (stageTimers) {
        QStringList parts;
        for (const StageStat &stat : stageStats) {
            if (stat.calls <= 0) continue;
            const double avgMs = (static_cast<double>(stat.totalNs) / 1.0e6) /
                                 static_cast<double>(stat.calls);
            parts << QString("%1=%2ms (%3 calls)")
                         .arg(stat.name)
                         .arg(avgMs, 0, 'f', 3)
                         .arg(stat.calls);
        }
        if (!parts.isEmpty()) {
            qInfo().noquote() << QString("Locked stage timers: %1").arg(parts.join(", "));
        }

        const qint64 fvfPixels =
            std::accumulate(fvfStatsTotal.finalWinnerCounts.begin(),
                            fvfStatsTotal.finalWinnerCounts.end(), qint64{0});
        if (fvfPixels > 0) {
            qInfo().noquote() << QString(
                "Locked FVF counters: raw(FA=%1,FB=%2,FRA=%3,FRB=%4) "
                "final(FA=%5,FB=%6,FRA=%7,FRB=%8) "
                "model(frame=%9,field=%10) frameHeadToHead(A=%11,B=%12) "
                "cleanup(island=%13,blockField=%14)")
                .arg(fvfStatsTotal.rawWinnerCounts[0])
                .arg(fvfStatsTotal.rawWinnerCounts[1])
                .arg(fvfStatsTotal.rawWinnerCounts[2])
                .arg(fvfStatsTotal.rawWinnerCounts[3])
                .arg(fvfStatsTotal.finalWinnerCounts[0])
                .arg(fvfStatsTotal.finalWinnerCounts[1])
                .arg(fvfStatsTotal.finalWinnerCounts[2])
                .arg(fvfStatsTotal.finalWinnerCounts[3])
                .arg(fvfStatsTotal.frameModelPixels)
                .arg(fvfStatsTotal.fieldModelPixels)
                .arg(fvfStatsTotal.frameAHeadToHeadWins)
                .arg(fvfStatsTotal.frameBHeadToHeadWins)
                .arg(fvfStatsTotal.islandChangedPixels)
                .arg(fvfStatsTotal.blockFieldCommitPixels);

            const auto &fp = fvfStatsTotal.islandFlipPairs;
            qInfo().noquote() << QString(
                "Locked FVF island flips [C->L] "
                "FA(->FB=%1,->FRA=%2,->FRB=%3) FB(->FA=%4,->FRA=%5,->FRB=%6) "
                "FRA(->FA=%7,->FB=%8,->FRB=%9) FRB(->FA=%10,->FB=%11,->FRA=%12)")
                .arg(fp[0][1]).arg(fp[0][2]).arg(fp[0][3])
                .arg(fp[1][0]).arg(fp[1][2]).arg(fp[1][3])
                .arg(fp[2][0]).arg(fp[2][1]).arg(fp[2][3])
                .arg(fp[3][0]).arg(fp[3][1]).arg(fp[3][2]);
        }

    }
}

// Seed clpbuffer[2] (the 3D working plane) from the completed 2D result in
// clpbuffer[1]. Called before split3D so that pixels where no temporal candidate
// improves on 2D are left with the 2D value rather than uninitialised data.
void Comb::FrameBuffer::copy2DTo3D()
{
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;

    for (int line = firstLine; line < lastLine; ++line) {
        if (line >= demodLines) continue;
        const double* src = clpbuffer[1].pixel[line];
        double* dst       = clpbuffer[2].pixel[line];
        // Use std::copy for speed
        std::copy(src + left, src + right, dst + left);
    }
}
// FrameBuffer - Constructor
Comb::FrameBuffer::FrameBuffer(const LdDecodeMetaData::VideoParameters &videoParameters_,
                               const Configuration &configuration_)
    : videoParameters(videoParameters_), configuration(configuration_)
{

    frameHeight = (videoParameters.fieldHeight * 2) - 1;
    irescale    = (videoParameters.white16bIre - videoParameters.black16bIre) / 100.0;
    invIreScale = (irescale != 0.0) ? (1.0 / irescale) : 0.0;

    const int lines = videoParameters.lastActiveFrameLine;
    const int width = videoParameters.activeVideoEnd - videoParameters.activeVideoStart;
    if (lines > 0 && width > 0) {
        const bool wantMap  = configuration.showMap;
        const bool wantLocked = configuration.phaseCompensation;
        // Note: we intentionally allow Frame/FVF selection even without locked mode
        // (a "half-locked" backdoor some users rely on). Storage is gated on the
        // variant selection, while the locked-only computations remain gated on
        // phaseCompensation inside split2D/scoreFieldVsFrame.
        
        const bool wantFvf  =
            (configuration.twoDVariant == Comb::Configuration::TwoDVariant::FieldVsFrame);
        const bool wantVdis = configuration.tunables.VDIS_ENABLE;
        const bool needFrameIQ =
            (configuration.twoDVariant == Comb::Configuration::TwoDVariant::FrameAAdaptiveIQ ||
             configuration.twoDVariant == Comb::Configuration::TwoDVariant::FrameBDirectIQ ||
             wantFvf);

        // 2D score blending visualization (only written when showMap is true)
        if (wantMap) {
            w2d_frame_weight.assign(lines, std::vector<float>(width, 0.0f));
        }
        // FVF-only data and scratch.
        if (wantFvf) {
            w2d_fieldA_gate.assign(lines, std::vector<double>(width, 1.0f));
            fvfMetrics.assign(lines, std::vector<FvfModelMetrics>(width));
            scratch_fvf_winner.assign(width, 1);
            scratch_fvf_winner2.assign(width, 1);
            scratch_fvf_outVal.assign(width, 0.0);
            scratch_fvf_outShade.assign(width, 0.35f);
            scratch_fvf_diffFVF.assign(width, 0.0);
            scratch_fvf_satMap.assign(width, 0.0);
        }
        // VDIS is opt-in.
        if (wantVdis) {
            vdisMask.assign(lines, std::vector<char>(width, 0));
            scratch_vdis_flag.assign(width, 0);
        }
        // Locked-path-only stable 1D source.
        if (wantLocked) {
            lockedLumaBaseY4_flat.assign(size_t(lines + 1) * size_t(width), 0.0);
            lockedLumaSmooth_flat.assign(size_t(lines + 1) * size_t(width), 0.0);
            lockedLumaCacheValid = false;
        }
        // Preclean ring is only needed for Frame/FVF in locked mode.
        if (needFrameIQ) {
            for (int s = 0; s < 3; ++s) {
                precleanRing[s].assign(width, 0.0);
                precleanGateRing[s].assign(width, 1.0);
                precleanRingLine[s] = -1;
            }
        }

        // Accumulators for raster synthesis
        scratch_lineWorkA.assign(width, 0.0);
        scratch_lineWorkB.assign(width, 1.0);
        scratch_lineWorkC.assign(width, 0.0);
        scratch_outMixed.assign(width, 0.0);
        scratch_lateralLine.assign(width, 0.0);
        // low-res luma (chroma cancelled fsc)        
        scratch_lumaBaseY4.assign(width, 0.0);
        scratch_lumaHiRaw.assign(width, 0.0);
        scratch_lumaSmooth.assign(width, 0.0);

        // Filtering/NR temporaries
        scratch_lineWorkD.assign(width, 0.0);
        scratch_hpI.assign(width + 64, 0.0);
        scratch_hpQ.assign(width + 64, 0.0);
        scratch_hpY.assign(width + 64, 0.0);
        
        // Reusable per-line chroma pre-FIR buffers
        scratch_preI.resize(width, 0.0);
        scratch_preQ.resize(width, 0.0);
        // New leakage/coherence scratch
        scratch_yhp.resize(width, 0.0);
        scratch_yI.resize(width, 0.0);
        scratch_yQ.resize(width, 0.0);
        // Initialize demod contiguous buffers geometry
        // demodLines indexed by absolute line number (safe upper bound)
        demodWidth = width;
        demodLines = lines + 1;
        carrierGrammar.assign(demodLines, CombCarrierGrammar{});
        if (wantLocked) {
            demodTI_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            demodTQ_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            demodTI4fsc_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            demodTQ4fsc_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            locked1DTI4fsc_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            locked1DTQ4fsc_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            lockedProductI_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            lockedProductQ_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            carrierImpurity_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            demodTRI_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            demodTRQ_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            locked1DSource_flat.assign(size_t(demodLines) * demodWidth, 0.0);
            attributionEvidence_flat.assign(
                size_t(demodLines) * demodWidth, AttributionEvidence{});
        }
        scratch_frameBDirectIQComposite.assign(width, 0.0);
        scratch_frameAAdaptiveIQComposite.assign(width, 0.0);
    }
}

// Interleave the two source fields into rawbuffer in frame-line order (even lines
// from firstField, odd lines from secondField), record their phase IDs, and derive
// a single cadenceId representative for this frame from the two fields' cinemap
// metadata. Also initialises per-line carrier grammar polarity and clears
// the VDIS mask. capturePartnerSeqNo records the original TBC frame pairing for
// each field, carried forward for reconstruction.
void Comb::FrameBuffer::loadFields(const SourceField &firstField,
                                   const SourceField &secondField)
{
    rawbuffer.clear();
    qint32 fieldLine = 0;
    for (qint32 frameLine = 0; frameLine < frameHeight; frameLine += 2) {
        rawbuffer.append(firstField.data.mid(fieldLine * videoParameters.fieldWidth,
                                             videoParameters.fieldWidth));
        rawbuffer.append(secondField.data.mid(fieldLine * videoParameters.fieldWidth,
                                              videoParameters.fieldWidth));
        fieldLine++;
    }

    firstFieldPhaseID  = firstField.field.fieldPhaseID;
    secondFieldPhaseID = secondField.field.fieldPhaseID;
    
    const bool editSplit = secondField.field.cinemap.isEditBoundary;
    
    const qint32 cidA = firstField.field.cinemap.cadenceId;
    const qint32 cidB = secondField.field.cinemap.cadenceId;
    cadenceId = lddecode::mergeCadenceIdForInterleavedFrame(
        cidA, cidB, editSplit);
    // Clear working planes only in active region for safety
    for (int buf = 0; buf < 3; ++buf) {
        for (int y = videoParameters.firstActiveFrameLine;
             y < videoParameters.lastActiveFrameLine; ++y)
        {
            double *row = clpbuffer[buf].pixel[y];
            std::fill(row + videoParameters.activeVideoStart,
                      row + videoParameters.activeVideoEnd, 0.0);
        }
    }

    componentFrame = nullptr;

    // Initialize per-line grammar with schedule identity and metadata authority.
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    lddecode::initializeCarrierGrammarSchedule(
        carrierGrammar,
        first,
        last,
        firstFieldPhaseID,
        secondFieldPhaseID,
        !editSplit);

    // Clear VDIS mask for this frame
    if (!vdisMask.empty()) {
        if ((int)vdisMask.size() < last) vdisMask.resize(last);
        const int width = videoParameters.activeVideoEnd - videoParameters.activeVideoStart;
        for (int line = first; line < last; ++line) {
            auto &row = vdisMask[line];
            if ((int)row.size() < width) row.assign(width, 0);
            else std::fill(row.begin(), row.end(), 0);
        }
    }
    lockedLumaCacheValid = false;
}



// 1D horizontal bandpass: isolates subcarrier energy by subtracting the average
// of the samples two positions either side (a 2-tap comb at 2fsc), scaled by 0.5.
// Result written to clpbuffer[0].
void Comb::FrameBuffer::split1D()
{
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int fullWidth = videoParameters.fieldWidth;

    if (left >= right || firstLine >= lastLine)
        return;

    for (int line = firstLine; line < lastLine; ++line) {
        const quint16 *src = rawbuffer.data() + line * fullWidth;
        double *dst        = clpbuffer[0].pixel[line];

        for (int h = left; h < right; ++h) {
            int hm2 = h - 2; if (hm2 < left)   hm2 = left  + (left  - hm2 - 1);
            int hp2 = h + 2; if (hp2 >= right)  hp2 = right - 1 - (hp2 - right);
            dst[h] = ((double)src[h] - 0.5 * ((double)src[hm2] + (double)src[hp2])) * 0.5;
        }
    }
}


void Comb::FrameBuffer::seedCombAttributionPerLine(int line)
{
    const int right = videoParameters.activeVideoEnd;
    const int width = right - videoParameters.activeVideoStart;

    if (width <= 0 || line < 0)
        return;

    AttributionEvidence *row = attributionEvidence_line(line);
    if (!row)
        return;

    // Carrier metadata lives in carrierGrammar; consumers read it there directly.
    // Seed only the line-level plausibility prior so later attribution stages start
    // from one canonical carrier verdict instead of privately reconstructing one.
    const CombCarrierGrammar *grammar = carrierGrammarLine(line);
    const double carrierPrior = carrierPlausibility(grammar);

    for (int rel = 0; rel < width; ++rel) {
        row[rel].facts = AttributionFacts{};
        row[rel].assessment = AttributionAssessment{};
        row[rel].assessment.carrierPrior = carrierPrior;
    }
}

void Comb::FrameBuffer::finalizeAttributionClaims(AttributionEvidence &e,
                                                double neighborLumaMeanIRE,
                                                double neighborBaseMeanIRE,
                                                double lineForwardErrorIRE) const
{
    const auto &T = configuration.tunables;
    AttributionRules rules = lddecode::kDefaultAttributionRules;
    rules.conflictSuppress = T.VET_ATTRIBUTION_CONFLICT_SUPPRESS;
    const AttributionFacts &f = e.facts;
    AttributionAssessment &a = e.assessment;

    const double crestIRE = f.bandpassFineIRE;
    const double baseIRE = std::max(f.bandpassMidIRE, f.bandpassCoarseIRE);
    const double maxChromaIRE = lddecode::strongestCombChromaIRE(f);

    a.lumaRisk = std::max(
        std::clamp(f.lumaIncursionRiskIRE / 8.0, 0.0, 1.0),
        std::clamp(f.icebergAlienYFraction, 0.0, 1.0));
    const double parallaxLatticeRisk = std::clamp(
        f.carrierParallaxLatticeRiskIRE /
        std::max(3.0, 0.35 * maxChromaIRE + 1.0),
        0.0,
        1.0);
    a.checkerboardRisk = std::max(
        std::clamp(f.quarterCheckerboardRisk, 0.0, 1.0),
        parallaxLatticeRisk);
    // Scale lumaResidual against the line-level forward-model error rather than
    // a fixed 8.0.  When the grammar has a valid projection, samples that merely
    // match the line's measured noise floor should not accumulate luma risk; the
    // denominator grows proportionally so only samples that significantly exceed
    // the line mean are flagged.  Falls back to 8.0 when lineForwardErrorIRE is
    // unavailable (grammar not locked or projection not valid).
    const double lumaResScale = (lineForwardErrorIRE > 0.0)
        ? std::max(8.0, 2.5 * lineForwardErrorIRE)
        : 8.0;
    a.lumaResidual = std::clamp(
        (f.residualFitErrorIRE - std::max(1.0, 0.2 * maxChromaIRE)) / lumaResScale,
        0.0, 1.0);

    a.baseSupport = std::clamp(
        (baseIRE - (0.25 * crestIRE) - 0.5) / std::max(2.0, (0.55 * crestIRE) + 1.0),
        0.0, 1.0);

    a.neighborSupport = 0.0;
    if (neighborLumaMeanIRE >= 0.0 && neighborBaseMeanIRE >= 0.0) {
        const double lumaDen = std::max(2.0, 0.5 * (f.lumaExcursionIRE + neighborLumaMeanIRE));
        const double baseDen = std::max(2.0, 0.5 * (baseIRE + neighborBaseMeanIRE));
        const double lumaMatch = 1.0 - std::min(1.0, std::fabs(f.lumaExcursionIRE - neighborLumaMeanIRE) / lumaDen);
        const double baseMatch = 1.0 - std::min(1.0, std::fabs(baseIRE - neighborBaseMeanIRE) / baseDen);
        a.neighborSupport = 0.5 * std::max(0.0, lumaMatch) + 0.5 * std::max(0.0, baseMatch);
    }
    a.lumaShapeContinuation = std::clamp((0.65 * a.baseSupport) + (0.35 * a.neighborSupport), 0.0, 1.0);
    
    a.chromaStrength = std::clamp((maxChromaIRE - 2.0) / 10.0, 0.0, 1.0);

    const double sidebandResidualIRE =
        std::max(f.sidebandSinResidualIRE, f.sidebandCosResidualIRE);
    const double sidebandMagSupport = std::clamp(
        (sidebandResidualIRE - 0.35) / 4.0,
        0.0, 1.0);
    const double sidebandAxisSupport = std::clamp(
        (std::fabs(f.sidebandAxisAsymmetry) - 0.10) / 0.65,
        0.0, 1.0);
    const double sidebandCoherence =
        std::clamp(f.sidebandCurvatureCoherence, 0.0, 1.0);
    a.sidebandChromaSupport = sidebandMagSupport *
        (0.35 + 0.45 * sidebandAxisSupport + 0.20 * sidebandCoherence);
    // Coherence fallback (used when frameIQCoherence is unavailable):
    // normalize per-sample fit error against the line's mean forward error
    // rather than the fixed 12.0.  A sample at the line mean gets coherence
    // ≈ 0.5; samples far below get ≈ 1.0; samples far above get ≈ 0.0.
    // Falls back to the hard-coded 12.0 when lineForwardErrorIRE is zero.
    const double cohScale = (lineForwardErrorIRE > 0.0)
        ? std::max(12.0, 2.0 * lineForwardErrorIRE)
        : 12.0;
    a.coherence = (f.frameIQCoherence > 0.0)
        ? f.frameIQCoherence
        : std::clamp(1.0 - (f.residualFitErrorIRE / cohScale), 0.0, 1.0);
    if (f.carrierParallaxCoherence > 0.0) {
        a.coherence = std::max(
            a.coherence,
            std::clamp(f.carrierParallaxCoherence, 0.0, 1.0));
    }
    if (f.carrierResidualIRE > 0.0) {
        const double carrierResidualScale = (lineForwardErrorIRE > 0.0)
            ? std::max(8.0, 2.0 * lineForwardErrorIRE)
            : 8.0;
        const double carrierResidualPenalty = std::clamp(
            (f.carrierResidualIRE - std::max(1.0, 0.20 * maxChromaIRE)) /
            carrierResidualScale,
            0.0,
            1.0);
        a.coherence = std::clamp(
            a.coherence * (1.0 - 0.35 * carrierResidualPenalty),
            0.0,
            1.0);
    }
    if (f.movingResidualCoherence > 0.0) {
        a.coherence = std::min(
            a.coherence,
            std::clamp(f.movingResidualCoherence, 0.0, 1.0));
    }
    a.agreement = 1.0 - std::clamp(f.frameFieldAgreementIRE / 6.0, 0.0, 1.0);
    a.spreadPenalty = std::max({
        std::clamp(f.candidateSpreadIRE / 10.0, 0.0, 1.0),
        std::clamp(f.carrierParallaxSpreadIRE /
                   std::max(4.0, 0.35 * maxChromaIRE + 1.0),
                   0.0,
                   1.0),
        std::clamp(f.movingResidualPull, 0.0, 1.0)
    });
    const double carrierPrior = configuration.phaseCompensation
        ? std::clamp(a.carrierPrior, 0.0, 1.0)
        : 1.0;

    // Carrier plausibility is a line-level grammar result, not a per-pixel
    // reinterpretation. Local evidence may affect chromaClaim, but it should
    // not invent a second carrier-confidence signal downstream.
    a.carrierPlausibility = carrierPrior;

    a.lumaClaim = std::clamp(
        (0.50 * a.lumaRisk) + (0.22 * a.lumaResidual) +
        (0.18 * a.lumaShapeContinuation) + (0.10 * a.checkerboardRisk),
        0.0, 1.0);

    a.chromaClaim = std::clamp(
        std::max(a.chromaStrength, a.sidebandChromaSupport) *
        ((0.65 * a.carrierPlausibility) + (0.35 * a.coherence)),
        0.0, 1.0);

    lddecode::applyAttributionConflictSuppression(
        a,
        rules);

    a.chromaClaim *= std::max(
        0.0,
        1.0 - (T.VET_ATTRIBUTION_CHROMA_WEIGHT *
               std::max(0.0, a.lumaShapeContinuation - 0.25)));
    a.chromaClaim *= std::max(0.0, 1.0 - (0.5 * a.lumaClaim));
    lddecode::normalizeCombAttributionAssessment(a, rules);
}

// ----------------------------------------------------------------------------
// FVF election v2. See docs/FVF_election_v2_design.md.
//   veto -> divergence boundary -> saturation regime -> candidate election.
//   Candidates: Field B, Frame A, Frame B (Field A retired). No notch; luma
//   from the cached base plus raw-candidate implied Y. Saturation from IQ mag.
// ----------------------------------------------------------------------------
void Comb::FrameBuffer::scoreFieldVsFrame(
    int line,
    const CombTapLine &tapLine,
    const double *fieldB,                          // Field B candidate (only field now)
    const std::vector<double> &frameA,             // Frame A candidate
    const std::vector<double> *frameB,             // Frame B candidate (interfield model)
    double *outMixed,
    bool writeWeights,
    const double *lateral1D,
    const std::vector<std::complex<double>> *frameIQ)
{
    // ------------------------------------------------------------------
    // 0. Geometry, guards, tunables, source lines  (unchanged from current)
    // ------------------------------------------------------------------
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;
    if (width <= 0) return;
    if (!fieldB || (int)frameA.size() < width || !outMixed) return;

    const auto  &T    = configuration.tunables;
    const double invI = this->invIreScale;
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int srcBufIndex = configuration.phaseCompensation ? 1 : 0;

    // Tunable thresholds (subset that survives; FA-only ones removed).
    // VERT_THRESH (vertical contrast) lives with the deferred horizontal-edge
    // / trust-reach regime (#8/#9/#10), not here.
    const double HEDGE_THRESH_IRE   = T.FIELD_LUMA_EDGE_THRESH_IRE;
    const double FIELD_DIVERGE_IRE  = 6.0;   // comb (IQ) divergence boundary
    const double LUMA_DIVERGE_IRE   = 6.0;   // luma divergence boundary (new knob)
    const double SAT_REGIME_IRE     = 16.0;  // #20 saturation regime: IQ-mag
                                             // threshold above which -> Frame B.
                                             // Analogous to the prior #24 ramp
                                             // point (sat_t≈0.72); dial in later.
    const double FVF_SMALL_DIFF_IRE =
        (T.FVF_SMALL_DIFF_IRE > 0.0) ? T.FVF_SMALL_DIFF_IRE : 3.0;
    const int    FIELD_BLOCK_SIZE   = 4;     // #27 block-field commit (native)
    const double FIELD_DISAGREE_IRE = 6.0;   // #27 per-block divergence threshold

    auto clampLine = [&](int ln)->int {
        return ln < firstLine ? firstLine : (ln >= lastLine ? lastLine - 1 : ln);
    };
    const double *srcLineM2 = clpbuffer[srcBufIndex].pixel[clampLine(line - 2)] + left;
    const double *srcLineM1 = clpbuffer[srcBufIndex].pixel[clampLine(line - 1)] + left;
    const double *srcLine0  = clpbuffer[srcBufIndex].pixel[clampLine(line)]     + left;
    const double *srcLineP1 = clpbuffer[srcBufIndex].pixel[clampLine(line + 1)] + left;
    const double *srcLineP2 = clpbuffer[srcBufIndex].pixel[clampLine(line + 2)] + left;

    auto sample1D = [&](int rel)->double {
        int r = std::clamp(rel, 0, width - 1);
        return lateral1D ? lateral1D[r] : srcLine0[r];
    };
    auto horizEdgeIRE = [&](int rel)->double {
        return (rel >= 0 && rel < (int)tapLine.hLumaDeltaIRE.size())
            ? tapLine.hLumaDeltaIRE[rel] : 0.0;
    };
    // (Vertical-contrast measure removed: it belongs to the horizontal-edge /
    //  trust-reach regime — #8/#9/#10 — which is DEFERRED until we see output.
    //  The vertical edge here is line-perpendicular = hIRE.)
    // (Notch luma helpers removed; luma comes from the cached base rows and
    //  raw-candidate implied Y.)

    // Scratch output planes (reuse existing members). Winner uses 1/2/3 for
    // FieldB/FrameA/FrameB; 0 retired. Sized once per line. (scratch_fvf_satMap
    // is no longer used — saturation is sourced from IQ magnitude.)
    if ((int)scratch_fvf_winner.size() != width) {
        scratch_fvf_winner.assign(width, 1);
        scratch_fvf_winner2.assign(width, 1);
        scratch_fvf_outVal.assign(width, 0.0);
        scratch_fvf_outShade.assign(width, 0.35f);
        scratch_fvf_diffFVF.assign(width, 0.0);
    }
    std::vector<int>    &winner   = scratch_fvf_winner;
    std::vector<double> &outVal   = scratch_fvf_outVal;
    std::vector<float>  &outShade = scratch_fvf_outShade;
    std::vector<double> &diffFVF  = scratch_fvf_diffFVF;

    // ==================================================================
    // 1. PER-LINE: hard veto + trust regime  (computed ONCE, not per pixel)
    // ==================================================================
    //
    // #4 — Management owns the veto. Creation is external: read grammar +
    // cadence. carrierFrameVerticalAllowed(line) already folds in editSplit
    // (grammar init takes !editSplit frame-wide). This is the ONLY veto.
    const bool interfieldLegal = carrierFrameVerticalAllowed(line);

    // Regime axis (design doc §2):
    //   native  = cadenceId in {-2 video, -1 unknown}  -> trust = measured coherence
    //   model   = cadenceId >= 0 (film) or -3 (29.97p)  -> trust = cadence metadata
    const bool nativeRegime =
        (cadenceId == lddecode::kCadenceVideo) || (cadenceId == CADENCE_UNKNOWN);
    const bool modelRegime  = !nativeRegime;

    // ------------------------------------------------------------------
    // 2. PER-LINE evidence harvest (consolidated)
    // ------------------------------------------------------------------
    // TODO(harvest): single pass producing, in one place:
    //   * reach limiters + vertical contrast      (#10 + #6, FA-free)
    //   * luma-domain contour trust                (#8, rebuilt on luma)
    //   * grammar lattice comb-need per region     (#9, from carrierGrammar
    //                                                spans: alternationCoherence
    //                                                / samePhaseRecurrence —
    //                                                NOT hardcoded ±2)
    // These currently live scattered across buildCombTapLine / inline lambdas.

    // Attribution row (#12). Advisory claims; never a hard veto.
    const AttributionEvidence *fvfAttrRow =
        (demodWidth == width) ? attributionEvidence_line(line) : nullptr;

    // Cached luma rows + raw row. ±1 lines are the adjacent field, so their
    // difference supplies the luma-divergence axis; raw-candidate is implied Y.
    const bool lumaCacheUsable =
        lockedLumaCacheValid && !lockedLumaBaseY4_flat.empty() && demodWidth == width;
    const double *lumaBaseRow = lumaCacheUsable ? lockedLumaBaseY4_line(line) : nullptr;
    const double *lumaUpRow   = (lumaCacheUsable && line - 1 >= firstLine)
                                    ? lockedLumaBaseY4_line(line - 1) : nullptr;
    const double *lumaDnRow   = (lumaCacheUsable && line + 1 < lastLine)
                                    ? lockedLumaBaseY4_line(line + 1) : nullptr;
    const quint16 *rawRow = rawbuffer.data() +
        static_cast<size_t>(line) * static_cast<size_t>(videoParameters.fieldWidth);

    // IQ magnitude pre-pass: one hypot per pixel, consumed by the saturation
    // test and #19's texture ladder.
    if ((int)scratch_fvf_iqMag.size() != width)
        scratch_fvf_iqMag.assign(width, 0.0);
    if (frameIQ && (int)frameIQ->size() >= width) {
        for (int r = 0; r < width; ++r) {
            const auto &z = (*frameIQ)[r];
            scratch_fvf_iqMag[r] = std::hypot(z.real(), z.imag());
        }
    }

    // ==================================================================
    // 2b. DIVERGENCE CLUSTER — GATHER (pool data only; no decisions).
    // One per-line pass fills the condition evidence; the election below is
    // the only thing that acts on it. combDivergence is laterally smoothed
    // (anti-zipper: breaks 4fsc bucket-line alternation).
    // ==================================================================
    if ((int)scratch_fvf_evidence.size() != width)
        scratch_fvf_evidence.assign(width, CombConditionEvidence());
    {
        double prevComb = 0.0;
        for (int rel = 0; rel < width; ++rel) {
            CombConditionEvidence &e = scratch_fvf_evidence[rel];

            const double FBv  = fieldB[rel];
            const double FRBv = (frameB && rel < (int)frameB->size())
                                    ? (*frameB)[rel] : frameA[rel];

            // IQ/comb divergence (smoothed) — the same samples the combs use.
            const double comb = std::fabs(FBv - FRBv) * invI;
            e.combDivergence = (rel > 0) ? 0.5 * (comb + prevComb) : comb;
            prevComb = comb;
            diffFVF[rel] = e.combDivergence;   // consumed by #2/#18, edge, #26/#27

            // Luma divergence between adjacent interleaved fields.
            if (lumaBaseRow) {
                const double yC = lumaBaseRow[rel];
                const double yU = lumaUpRow ? lumaUpRow[rel] : yC;
                const double yD = lumaDnRow ? lumaDnRow[rel] : yC;
                e.lumaDivergence = std::fabs(yC - 0.5 * (yU + yD)) * invI;
            } else {
                e.lumaDivergence = 0.0;
            }

            e.satIRE          = scratch_fvf_iqMag[rel] * invI;
            e.contourNonLocal = (rel < (int)tapLine.contour.size())
                ? std::clamp(tapLine.contour[rel].midOk * 0.5 *
                    (tapLine.contour[rel].upSideOk + tapLine.contour[rel].dnSideOk),
                    0.0, 1.0)
                : 0.0;   // pooled for the (deferred) contour consumer
        }
    }

    // ==================================================================
    // 3. PER-PIXEL election
    // ==================================================================
    for (int rel = 0; rel < width; ++rel) {
        const int rm2 = std::max(0, rel - 2);
        const int rp2 = std::min(width - 1, rel + 2);

        // ---- candidate values (3 candidates) --------------------------
        const double FB     = fieldB[rel];                       // Field B
        const double FR     = frameA[rel];                       // Frame A
        const double FRB    = (frameB && rel < (int)frameB->size())
                                  ? (*frameB)[rel] : FR;         // Frame B
        const double L1     = sample1D(rel);

        // ---- consume pooled condition data (gathered above) -----------
        const CombConditionEvidence &e = scratch_fvf_evidence[rel];
        const double diff_candB_ire = e.combDivergence;   // smoothed comb (IQ) divergence
        const double chromaMagIRE   = e.satIRE;           // saturation test

        // ============================================================
        // TOP-LEVEL REGIME BRANCHES — central management acting on the pool.
        //   1. veto (interfield illegal)                  -> Field B
        //   2. divergence boundary (BOTH regimes)         -> Field B
        //   3. #20 saturation regime (IQ mag), if viable  -> Frame B
        //   else -> candidate score election
        // ============================================================
        int    idx   = 1;        // default Field B
        double val   = FB;
        float  shade = 0.35f;

        if (!interfieldLegal) {
            winner[rel] = 1; outVal[rel] = FB; outShade[rel] = 0.35f;     // veto
            continue;
        }

        // DIVERGENCE BOUNDARY (both regimes) — consumes the pool. Interfield is
        // viable only when both cached luma and comb-domain evidence agree.
        const bool lumaDiverges  = e.lumaDivergence >= LUMA_DIVERGE_IRE;
        const bool combDiverges  = e.combDivergence >= FIELD_DIVERGE_IRE;
        const bool fieldsCoherent = !lumaDiverges && !combDiverges;
        if (!fieldsCoherent) {
            winner[rel] = 1; outVal[rel] = FB; outShade[rel] = 0.35f;
            continue;
        }
        if (chromaMagIRE >= SAT_REGIME_IRE) {
            // #20 SATURATION REGIME — among interfield-VIABLE positions only
            // (the divergence boundary above already excluded the rest).
            // Saturated -> Frame B outright; candidate-independent (no scoring).
            // Replaces the old #20 score biases + #24 force + notch seed.
            winner[rel] = 3; outVal[rel] = FRB; outShade[rel] = 0.85f;  // -> Frame B
            continue;
        }

        // ============================================================
        // 3a. CANDIDATE SCORING  (interfield in play, NOT saturated)
        // Lower score = better. No scoreA (Field A retired).
        // ============================================================
        // ---- ADDITIVE FOUNDATION ------------------------------------
        // The old saturated notch seed is GONE — the saturation REGIME above
        // handles saturation outright, so we never reach here when saturated.
        // The additive base is purely #2/#18 + #12; the multiplicative
        // compartments (#19, #6, #21) scale it.
        double scoreB   = 0.0;   // Field B
        double scoreR_A = 0.0;   // Frame A
        double scoreR_B = 0.0;   // Frame B

        // sat_t: a GRADUAL ramp of IQ magnitude BELOW the regime threshold,
        // consumed ONLY by #19's intrinsic ±4 color enhancement. (Above the
        // threshold we are in the saturation regime and never score here.)
        const double sat_t = std::clamp(
            (chromaMagIRE - 6.0) / std::max(1.0, SAT_REGIME_IRE - 6.0), 0.0, 1.0);
        const double hIRE = horizEdgeIRE(rel);    // used by #6 + #21 + election

        // ---- #2 / #18  cross-domain model distance (directional) ------
        // ONE unit: measure (diff_candB_ire) + apply. This is now the
        // load-bearing frame-sanity term (frameInsane retired). Direction
        // flips by regime — the cross-domain candidate pays distance from
        // the regime's trusted home model.
        {
            const double deadband = T.FVF_MODEL_PRIMARY_DEADBAND_IRE;
            const double pen = T.FVF_MODEL_PRIMARY_WEIGHT *
                               std::max(0.0, diff_candB_ire - deadband);
            if (modelRegime) {
                scoreB   += pen;   // field pays distance from trusted Frame B
            } else {
                scoreR_A += pen;   // frames pay distance from trusted Field B
                scoreR_B += pen;
            }
        }

        // ---- #12  attribution ----------------------------------------
        // Frames pay luma claim (residual reads as luma → interfield chroma
        // suspect); field pays chroma claim (residual reads as chroma →
        // intrafield suspect). #13 checkerFieldRisk block is NOT ported.
        // checkerboardRisk / lumaIncursion are dropped — they only fed the
        // retired #20 cross-color guard.
        double attrLumaClaim = 0.0, attrChromaClaim = 0.0;
        if (fvfAttrRow) {
            attrLumaClaim   = std::clamp(fvfAttrRow[rel].assessment.lumaClaim, 0.0, 1.0);
            attrChromaClaim = std::clamp(fvfAttrRow[rel].assessment.chromaClaim, 0.0, 1.0);
            scoreR_A += T.FVF_ATTRIBUTION_LUMA_WEIGHT   * attrLumaClaim;
            scoreR_B += T.FVF_ATTRIBUTION_LUMA_WEIGHT   * attrLumaClaim;
            scoreB   += T.FVF_ATTRIBUTION_CHROMA_WEIGHT * attrChromaClaim;
        }

        // ---- #19  IQ texture scale ladder (self-contained) -----------
        // fine → Frame B (interframe detail); coarse → Field B (±2/±4
        // stability); dual-±4 → extra Field B. The ±4/coarse bonus is a
        // PROVEN color-quality boost; its saturation scaling (satCoarseAmp)
        // is intrinsic to the mechanism and lives HERE, at its texture origin
        // — NOT in #20. Texture scale and the saturation regime are separate
        // compartments that do not reach into each other. mid: denom-only.
        // (iqCoherence is NOT computed here — coherence "trust" belongs to the
        //  deferred #8 trust regime; its only consumers were the retired #20
        //  guard and #24, and the pre-pass that fed it was removed.)
        if (frameIQ && rel < (int)frameIQ->size()) {
            const int rm1 = std::max(0, rel - 1), rp1 = std::min(width - 1, rel + 1);
            const int rm4 = std::max(0, rel - 4), rp4 = std::min(width - 1, rel + 4);
            const double m0  = scratch_fvf_iqMag[rel];
            const double m_1 = scratch_fvf_iqMag[rm1], mp1 = scratch_fvf_iqMag[rp1];
            const double m_2 = scratch_fvf_iqMag[rm2], mp2 = scratch_fvf_iqMag[rp2];
            const double m_4 = scratch_fvf_iqMag[rm4], mp4 = scratch_fvf_iqMag[rp4];
            const double fine   = std::fabs(m0 - 0.5 * (m_1 + mp1));
            const double mid    = std::fabs(m0 - 0.5 * (m_2 + mp2));   // denom only
            const double coarse = std::fabs(m0 - 0.5 * (m_4 + mp4));
            const double denom  = fine + mid + coarse + 1e-9;
            const double fineFrac   = fine   / denom;
            const double coarseFrac = coarse / denom;

            // ±4 reach color enhancement scales with saturation. sat_t is the
            // #5 product; this consumption is intrinsic to the texture boost.
            const double satCoarseAmp = 1.0 + 0.50 * sat_t;

            scoreR_B *= (1.0 - T.FVF_SCALE_FINE_FRAME_B_BONUS   * fineFrac);
            scoreB   *= (1.0 + 0.06 * fineFrac);

            scoreB   *= (1.0 - T.FVF_SCALE_COARSE_FIELD_A_BONUS * coarseFrac * satCoarseAmp);
            scoreR_A *= (1.0 + 0.08 * coarseFrac * satCoarseAmp);
            scoreR_B *= (1.0 + 0.04 * coarseFrac * satCoarseAmp);

            const bool dual4Accepted =
                tapLine.haveU4 && tapLine.haveD4 &&
                rel < (int)tapLine.contour.size() &&
                tapLine.contour[rel].upSideOk > 0.5 &&
                tapLine.contour[rel].dnSideOk > 0.5;
            if (dual4Accepted)
                scoreB *= (1.0 - T.FVF_SCALE_COARSE_DUAL4_FIELD_A_BONUS *
                                 coarseFrac * satCoarseAmp);
        }

        // ---- #6  vertical-edge Field-B penalty -----------------------
        // A "vertical edge" is one the scanline crosses PERPENDICULARLY — a
        // horizontal luma transition along the line (hIRE), not vertical
        // contrast. A Field B penalty is enough here; a Frame A penalty goes
        // back only if output later shows we need it. No "trust" — trust is the
        // horizontal-edge (line-parallel) regime, deferred with #8/#9/#10.
        {
            const double hedge_t = std::clamp(
                hIRE / std::max(HEDGE_THRESH_IRE, 1.0), 0.0, 1.0);
            scoreB *= (1.0 + T.FVF_HEDGE_FIELD_B_PENALTY * hedge_t);
        }

        // #20 saturation regime is now a TOP-LEVEL branch (above), not a
        // scoring block. The old field/frame sat biases, Frame B cross-color
        // guard, notch seed, and #24 force are ALL gone — replaced by the
        // global "saturated -> Frame B" branch. Nothing saturation-policy is
        // scored here.

        // ---- #21  transition sharpness reward (ported) ---------------
        // Stable-step detection at horizontal edges; reward candidates that
        // make a fast step and settle on both plateaus. A "free sharpen" with
        // none of a filter's downsides. Ported verbatim minus the FA arm.
        {
            constexpr int EDGE_GAP = 2;
            constexpr int EDGE_PROBE_NEAR = 2;
            constexpr int EDGE_PROBE_FAR  = 8;
            const bool canEval =
                (hIRE >= 0.75 * HEDGE_THRESH_IRE) &&
                (rel >= (EDGE_GAP + EDGE_PROBE_FAR)) &&
                (rel + (EDGE_GAP + EDGE_PROBE_FAR) < width) &&
                (line >= firstLine && line < lastLine);
            if (canEval) {
                // Source plateaus from cached base luma (no notch);
                // fallback to the source line only if the cache is unavailable.
                auto srcLuma = [&](int r)->double {
                    r = std::clamp(r, 0, width - 1);
                    return lumaBaseRow ? lumaBaseRow[r] : srcLine0[r];
                };
                const double lNear = srcLuma(rel - (EDGE_GAP + EDGE_PROBE_NEAR));
                const double lFar  = srcLuma(rel - (EDGE_GAP + EDGE_PROBE_FAR));
                const double rNear = srcLuma(rel + (EDGE_GAP + EDGE_PROBE_NEAR));
                const double rFar  = srcLuma(rel + (EDGE_GAP + EDGE_PROBE_FAR));
                const double stepIRE    = std::fabs(rNear - lNear) * invI;
                const double lJitterIRE = std::fabs(lNear - lFar) * invI;
                const double rJitterIRE = std::fabs(rNear - rFar) * invI;
                const double EDGE_STEP_THRESH_IRE = std::max(2.0, 0.9 * HEDGE_THRESH_IRE);
                const double EDGE_PLATEAU_JITTER_MAX_IRE = 1.2;
                const bool stableStep =
                    (stepIRE >= EDGE_STEP_THRESH_IRE) &&
                    (lJitterIRE <= EDGE_PLATEAU_JITTER_MAX_IRE) &&
                    (rJitterIRE <= EDGE_PLATEAU_JITTER_MAX_IRE);
                if (stableStep) {
                    const double lmeanIRE = lNear * invI;
                    const double rmeanIRE = rNear * invI;
                    // Candidate implied Y = raw − candidate (no notch — the
                    // candidate already accounts for chroma, so the point
                    // difference is luma).
                    auto applySharpReward = [&](double &score,
                                                const double *arr,
                                                const std::vector<double> *vec) {
                        const double cm2 = arr ? arr[rm2] : (*vec)[rm2];
                        const double cp2 = arr ? arr[rp2] : (*vec)[rp2];
                        const double m2 = static_cast<double>(rawRow[left + rm2]) - cm2;
                        const double p2 = static_cast<double>(rawRow[left + rp2]) - cp2;
                        const double candStepIRE = std::fabs(p2 - m2) * invI;
                        const double settleL = std::fabs(m2 * invI - lmeanIRE);
                        const double settleR = std::fabs(p2 * invI - rmeanIRE);
                        const double SETTLE_MAX_IRE = 0.35 * stepIRE + 1.0;
                        if (settleL > SETTLE_MAX_IRE || settleR > SETTLE_MAX_IRE) return;
                        const double ratio = candStepIRE / std::max(1e-9, stepIRE);
                        const double sharp = std::clamp((ratio - 0.70) / 0.30, 0.0, 1.0);
                        const double stepStrength = std::clamp(
                            (stepIRE - EDGE_STEP_THRESH_IRE) / 6.0, 0.0, 1.0);
                        score *= (1.3 - T.FVF_TRANSITION_SHARPNESS_WEIGHT * sharp * stepStrength);
                    };
                    applySharpReward(scoreB,   fieldB, nullptr);
                    applySharpReward(scoreR_A, nullptr, &frameA);
                    applySharpReward(scoreR_B, nullptr, frameB ? frameB : &frameA);
                }
            }
        }

        // ---- DEFERRED to "the other side" (after we see output) ------
        // #8/#9/#10 (contour/vertical-trust, lattice comb-need, reach) are the
        // horizontal-edge / trust-reach regime — line-PARALLEL edges. NOT ported
        // now; revisit once output is reviewed. (VDIS is gone; the divergence
        // cluster will own the horizontal-line issue when that regime returns.)

        // ---- NATIVE-REGIME-ONLY scoring (cadenceId in {-2,-1}) ---------
        if (nativeRegime) {
            // #23 small-diff frame reward — gated OUT under progressive. When
            // field and frame agree closely, prefer the frame (more vertical
            // resolution). Uses the comb-divergence comparator.
            if (diff_candB_ire < FVF_SMALL_DIFF_IRE) {
                const double close = 1.0 - std::clamp(
                    diff_candB_ire / std::max(1e-9, FVF_SMALL_DIFF_IRE), 0.0, 1.0);
                scoreR_A *= (1.0 - 0.12 * close);
                scoreR_B *= (1.0 - 0.12 * close);
            }
            // Frames clear a higher bar in the native regime, where interfield
            // is trusted on measured coherence rather than metadata.
            scoreR_A *= T.FRAME_IN_INTERLACE_PENALTY;
            scoreR_B *= T.FRAME_IN_INTERLACE_PENALTY;
            // Native bias toward the field model (now Field B).
            scoreB   *= T.FIELD_MODEL_BIAS;
            // (#27 block-field commit is finalized after the loop, native only.)
        }

        // ---- MODEL-REGIME-ONLY scoring (cadenceId >= 0 or -3) ----------
        if (modelRegime) {
            // Metadata trust: frames compete on evidence. The cross-domain
            // model-distance penalty (#2, shared block) is the SOLE policing
            // of a divergent frame here — so its weight/deadband inherits
            // frameInsane's retired job. No interlace penalty, no field bias.
        }

        // ============================================================
        // 3b. ELECTION (#25) — 3-candidate cascade
        // ============================================================
        auto pick = [&](int candIdx, double candVal, float candShade) {
            idx = candIdx; val = candVal; shade = candShade;
        };
        auto pickBestFrame = [&](float shA, float shB) {
            if (scoreR_A <= scoreR_B) pick(2, FR,  shA);
            else                      pick(3, FRB, shB);
        };

        // hIRE hoisted to the additive-foundation block (shared with #21).

        // Cascade (saturation is no longer here — it is a top-level regime
        // branch that already returned Frame B for saturated pixels):
        //   1. strong horizontal edge — frames more reliable than interfield
        //   2. general score comparison
        if (hIRE > HEDGE_THRESH_IRE && diff_candB_ire > 5.0) {
            // Strong luma edge.
            if (scoreR_B <= scoreR_A && scoreR_B <= scoreB) pick(3, FRB, 0.85f);
            else if (scoreR_A <= scoreB)                    pick(2, FR,  0.7f);
            else                                            pick(1, FB,  0.35f);
        } else {
            const double bestFrame = std::min(scoreR_A, scoreR_B);
            if (modelRegime && bestFrame <= scoreB) {
                pickBestFrame(0.7f, 0.85f);          // metadata model wins ties
            } else if (bestFrame + 1e-12 < scoreB * 0.85) {
                pickBestFrame(0.7f, 0.85f);          // frame convincingly beats field
            } else {
                // Tiebreaker (no-notch): which candidate's implied Y
                // (raw − candidate) sits closest to cached base luma?
                // LAZY — only pixels reaching here pay for it.
                const double yRef   = lumaBaseRow ? lumaBaseRow[rel] : L1;
                const double rawHere = static_cast<double>(rawRow[left + rel]);
                const double dFL = std::fabs((rawHere - FB)  - yRef) * invI;  // Field B
                const double dRL = std::fabs((rawHere - FRB) - yRef) * invI;  // Frame B
                if (dRL + 1.0 < dFL) pickBestFrame(0.7f, 0.85f);
                else                 pick(1, FB, 0.35f);
            }
        }

        (void)FIELD_BLOCK_SIZE;  // pending #27 block-field commit
        winner[rel] = idx; outVal[rel] = val; outShade[rel] = shade;
    }

    // ==================================================================
    // 4. CLEANUP
    // ==================================================================
    // #26 Island cleanup — flip a single-pixel winner to match its neighbours
    // when both sides agree, except in saturated / edge / divergent spots.
    {
        std::vector<int> &w2 = scratch_fvf_winner2;
        std::copy(winner.begin(), winner.end(), w2.begin());
        const double EDGE_STOP_IRE = HEDGE_THRESH_IRE;
        const double DIFF_STOP_IRE = 6.0;
        const double SAT_STOP_IRE  = 6.0;     // don't flip inside coloured areas
        bool changed = false;
        for (int rel = 1; rel < width - 1; ++rel) {
            if (scratch_fvf_iqMag[rel] * invI > SAT_STOP_IRE) continue;
            if (horizEdgeIRE(rel) > EDGE_STOP_IRE)            continue;
            if (diffFVF[rel] > DIFF_STOP_IRE)                 continue;
            const int L = winner[rel - 1], C = winner[rel], R = winner[rel + 1];
            if (L == R && C != L) { w2[rel] = L; changed = true; }
        }
        if (changed) {
            winner.swap(w2);
            for (int rel = 0; rel < width; ++rel) {
                const int idx = winner[rel];
                if      (idx == 1) { outVal[rel] = fieldB[rel]; outShade[rel] = 0.35f; }
                else if (idx == 2) { outVal[rel] = frameA[rel]; outShade[rel] = 0.7f;  }
                else               { outVal[rel] = frameB ? (*frameB)[rel] : frameA[rel];
                                     outShade[rel] = 0.85f; }
            }
        }
    }

    // #27 Block-field commit — NATIVE regime only. Where fields dominate the
    // line and a block's divergence is high, commit the whole block to Field B
    // to de-risk interfield teeth. (Field A retired -> the commit is Field B.)
    {
        int fieldCount = 0, frameCount = 0;
        for (int rel = 0; rel < width; ++rel) {
            const int idx = winner[rel];
            if      (idx == 2 || idx == 3) frameCount++;
            else if (idx == 1)             fieldCount++;
        }
        if (nativeRegime && fieldCount > frameCount * 2 && fieldCount > 0) {
            for (int b = 0; b < width; b += FIELD_BLOCK_SIZE) {
                const int e = std::min(width, b + FIELD_BLOCK_SIZE);
                double blockDivergence = 0.0;
                for (int r = b; r < e; ++r) blockDivergence += diffFVF[r];
                blockDivergence /= (e - b);
                if (blockDivergence > FIELD_DISAGREE_IRE) {
                    int cntField = 0, cntFrame = 0;
                    for (int r = b; r < e; ++r) {
                        if (winner[r] == 1) cntField++; else cntFrame++;
                    }
                    if (cntFrame > 0 && cntField > 0) {       // mixed block
                        for (int r = b; r < e; ++r) {
                            winner[r] = 1; outVal[r] = fieldB[r]; outShade[r] = 0.35f;
                        }
                    }
                }
            }
        }
    }

    // ==================================================================
    // 5. Emit — elected scalar to outMixed (+ diagnostics).
    // ==================================================================
    for (int rel = 0; rel < width; ++rel) {
        outMixed[rel] = outVal[rel];
        if (line >= 0 && line < (int)fvfMetrics.size() &&
            rel < (int)fvfMetrics[line].size())
            fvfMetrics[line][rel].winner = winner[rel];
        if (writeWeights && line >= 0 && line < (int)w2d_frame_weight.size()) {
            float w = outShade[rel];
            if (!std::isfinite(w)) w = 0.0f;
            w2d_frame_weight[line][rel] = w;
        }
    }
    // (fvfInstrumentation counters omitted in the scaffold — re-add at
    //  integration if --debug-phase-legs diagnostics are wanted.)

    // ==================================================================
    // RETIRED / RELOCATED / LEGACY (must NOT reappear here)
    //   removed:   #1 stack-avg, #11 shape, #13 checker, #14 no-comb-no-win,
    //              #15 rolling, #16 FA fast-commit, #17 FA gate, #22 neighbor,
    //              Field A candidate + computeContourFieldLine + fieldAGate
    //   relocated: #5 chroma-mag/sat ramp -> instrumentation outside FVF
    //   legacy:    #7 horizontal-edge Frame-B reward -> bucket / 1D backdoor
    //              only (--two-d-variant FVF, no --ntsc-phase-comp)
    // ==================================================================
}

void Comb::FrameBuffer::collectCombAttributionEvidence(
    int line,
    const double *fieldA,
    const double *fieldB,
    const std::vector<double> &frameScalar,
    const std::vector<std::complex<double>> *frameIQ)
{
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;

    Q_UNUSED(left);

    if (width <= 0 || !fieldA || !fieldB || line < 0)
        return;

    AttributionEvidence *row = attributionEvidence_line(line);
    if (!row)
        return;

    const bool haveFrameScalar = !frameScalar.empty();

    auto sampleFrameScalar = [&](int r) -> double {
        return frameScalar[std::clamp(r, 0, (int)frameScalar.size() - 1)];
    };

    auto frameCoherence = [&](int r) -> double {
        if (!frameIQ || frameIQ->empty())
            return 0.0;

        const int n = (int)frameIQ->size();
        r = std::clamp(r, 0, n - 1);

        std::complex<double> sum = {0.0, 0.0};
        double magSum = 0.0;

        for (int off : {-2, 0, 2}) {
            const auto &z = (*frameIQ)[std::clamp(r + off, 0, n - 1)];
            sum += z;
            magSum += std::hypot(z.real(), z.imag());
        }

        return (magSum > 1e-9)
            ? std::clamp(std::hypot(sum.real(), sum.imag()) / magSum, 0.0, 1.0)
            : 0.0;
    };

    // IQ coherence pre-pass: evaluate frameCoherence() once per pixel into a
    // flat array rather than 758× in the hot loop.  The line-level mean gates
    // per-sample values — a globally incoherent line cannot inflate isolated
    // samples (hot-loop disconnection principle): coherence is a line property,
    // not a pixel property, and should be established before the hot loop runs.
    if ((int)scratch_coe_coherence.size() != width)
        scratch_coe_coherence.assign(width, 0.0);
    double lineMeanFrameCoherence = 0.0;
    for (int r = 0; r < width; ++r) {
        scratch_coe_coherence[r]  = frameCoherence(r);  // 0.0 when !frameIQ
        lineMeanFrameCoherence   += scratch_coe_coherence[r];
    }
    if (width > 0) lineMeanFrameCoherence /= static_cast<double>(width);

    for (int rel = 0; rel < width; ++rel) {
        AttributionEvidence &e = row[rel];
        AttributionFacts &f = e.facts;

        const double fa = fieldA[rel];
        const double fb = fieldB[rel];
        const double fr = haveFrameScalar ? sampleFrameScalar(rel) : 0.0;

        f.fieldAChromaIRE = std::fabs(fa) * invIreScale;
        f.fieldBChromaIRE = std::fabs(fb) * invIreScale;

        f.frameChromaIRE = (frameIQ && rel < (int)frameIQ->size())
            ? std::hypot((*frameIQ)[rel].real(), (*frameIQ)[rel].imag()) * invIreScale
            : (haveFrameScalar ? std::fabs(fr) * invIreScale : 0.0);

        const double lo = haveFrameScalar ? std::min({fa, fb, fr}) : std::min(fa, fb);
        const double hi = haveFrameScalar ? std::max({fa, fb, fr}) : std::max(fa, fb);

        f.candidateSpreadIRE = (hi - lo) * invIreScale;

        f.frameFieldAgreementIRE = haveFrameScalar
            ? std::min(std::fabs(fr - fa), std::fabs(fr - fb)) * invIreScale
            : 0.0;

        // Gate per-sample coherence against the line-level mean.
        // If the line as a whole is incoherent, an isolated coherent pixel
        // is not evidence of a clean carrier — it is measurement noise.
        f.frameIQCoherence = scratch_coe_coherence[rel]
                             * (0.3 + 0.7 * lineMeanFrameCoherence);

    }

    // Refresh carrier prior from the finalized line grammar verdict before finalize.
    // Once the forward projection is available, it becomes the canonical carrier
    // plausibility signal for every pixel on the line.
    const CombCarrierGrammar *lineGrammar = carrierGrammarLine(line);
    const double lineCarrierPrior = carrierPlausibility(lineGrammar);
    for (int rel = 0; rel < width; ++rel)
        row[rel].assessment.carrierPrior = lineCarrierPrior;

    // Extract the line-level forward model error from the grammar (only when
    // the carrier projection was successfully computed on a locked line).
    // 0.0 signals "not available" and causes finalizeAttributionClaims() to fall
    // back to its hard-coded denominators — behaviour is identical to before.
    const double lineForwardErrorIRE = (lineGrammar && lineGrammar->projectionValid)
        ? lineGrammar->meanForwardErrorIRE
        : 0.0;

    // Final attribution needs cross-path evidence plus a same-phase continuity
    // check, not just the local 1D residual snapshot.
    for (int rel = 0; rel < width; ++rel) {
        const int rm4 = std::max(0, rel - 4);
        const int rp4 = std::min(width - 1, rel + 4);
        AttributionEvidence &e = row[rel];
        const AttributionFacts &leftFacts = row[rm4].facts;
        const AttributionFacts &rightFacts = row[rp4].facts;
        const double leftBaseIRE = std::max(leftFacts.bandpassMidIRE,
                                            leftFacts.bandpassCoarseIRE);
        const double rightBaseIRE = std::max(rightFacts.bandpassMidIRE,
                                             rightFacts.bandpassCoarseIRE);

        finalizeAttributionClaims(
            e,
            0.5 * (leftFacts.lumaExcursionIRE + rightFacts.lumaExcursionIRE),
            0.5 * (leftBaseIRE + rightBaseIRE),
            lineForwardErrorIRE);
    }
}

void Comb::FrameBuffer::buildCompositeLumaDecompositionLine(const quint16 *rawLine,
                                                            int left,
                                                            int width,
                                                            double *baseY4,
                                                            double *hiRaw,
                                                            double *lumaSmooth) const
{
    if (!rawLine || width <= 0)
        return;

    if (!baseY4 && !hiRaw && !lumaSmooth)
        return;

    // Degenerate active widths should not happen in normal NTSC 4fSC use.
    if (width < 4) {
        double avg = 0.0;
        for (int x = 0; x < width; ++x)
            avg += (double)rawLine[left + x];
        avg /= (double)width;

        if (baseY4) {
            for (int x = 0; x < width; ++x)
                baseY4[x] = avg;
        }
        if (hiRaw) {
            for (int x = 0; x < width; ++x)
                hiRaw[x] = (double)rawLine[left + x] - avg;
        }
        if (lumaSmooth) {
            for (int x = 0; x < width; ++x)
                lumaSmooth[x] = avg;
        }
        return;
    }

    // First pass: hard 4fSC-cycle luma base and optional high-frequency residual.
    // No temporary block vector; each block average is written directly.
    int p = 0;
    for (; p + 3 < width; p += 4) {
        const double y =
            0.25 * ((double)rawLine[left + p + 0] +
                    (double)rawLine[left + p + 1] +
                    (double)rawLine[left + p + 2] +
                    (double)rawLine[left + p + 3]);

        if (baseY4) {
            baseY4[p + 0] = y;
            baseY4[p + 1] = y;
            baseY4[p + 2] = y;
            baseY4[p + 3] = y;
        }

        if (hiRaw) {
            hiRaw[p + 0] = (double)rawLine[left + p + 0] - y;
            hiRaw[p + 1] = (double)rawLine[left + p + 1] - y;
            hiRaw[p + 2] = (double)rawLine[left + p + 2] - y;
            hiRaw[p + 3] = (double)rawLine[left + p + 3] - y;
        }
    }

    // Tail: reuse final complete 4-sample window.
    if (p < width) {
        const int tb = std::max(0, width - 4);
        const double y =
            0.25 * ((double)rawLine[left + tb + 0] +
                    (double)rawLine[left + tb + 1] +
                    (double)rawLine[left + tb + 2] +
                    (double)rawLine[left + tb + 3]);

        for (int x = p; x < width; ++x) {
            if (baseY4)
                baseY4[x] = y;
            if (hiRaw)
                hiRaw[x] = (double)rawLine[left + x] - y;
        }
    }

    if (!lumaSmooth)
        return;

    // lumaSmooth is the interpolated curve through 4-sample block centers.
    // Reuse the baseY4 block means when available so we don't re-average the
    // same 4-sample windows a second time just to build the smooth scaffold.
    auto blockAvg = [&](int block)->double {
        const int x0 = std::clamp(block * 4, 0, std::max(0, width - 4));
        if (baseY4)
            return baseY4[x0];
        return 0.25 * ((double)rawLine[left + x0 + 0] +
                       (double)rawLine[left + x0 + 1] +
                       (double)rawLine[left + x0 + 2] +
                       (double)rawLine[left + x0 + 3]);
    };

    const int blockCount = (width + 3) / 4;
    if (blockCount <= 1) {
        const double y = blockAvg(0);
        for (int x = 0; x < width; ++x)
            lumaSmooth[x] = y;
        return;
    }

    // Head clamp before first anchor center.
    const double yFirst = blockAvg(0);
    if (width > 0) lumaSmooth[0] = yFirst;
    if (width > 1) lumaSmooth[1] = yFirst;

    for (int b = 0; b < blockCount - 1; ++b) {
        const double y0 = blockAvg(b);
        const double y1 = blockAvg(b + 1);
        const double d  = (y1 - y0) * 0.25;

        const int xStart = std::max(0, b * 4 + 2);
        const int xEnd   = std::min(width, b * 4 + 6);

        for (int x = xStart; x < xEnd; ++x) {
            // t = (x - (b*4 + 1.5)) / 4.0
            const double t = ((double)x - ((double)b * 4.0 + 1.5)) * 0.25;
            lumaSmooth[x] = y0 + (y1 - y0) * t;
        }
    }

    // Tail clamp after last anchor center.
    const double yLast = blockAvg(blockCount - 1);
    const int tailStart = std::max(0, (blockCount - 1) * 4 + 2);
    for (int x = tailStart; x < width; ++x)
        lumaSmooth[x] = yLast;
}
//diagnostic tool for comb development 
void Comb::FrameBuffer::reportPhaseLegStats(const char *label, int srcBufIndex, bool useLockedSource) const
{
    if (!configuration.debugPhaseLegs || !configuration.phaseCompensation)
        return;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;

    if (width <= 8 || firstLine >= lastLine)
        return;

    struct LegStats {
        qint64 n = 0;
        double sumI = 0.0;
        double sumQ = 0.0;
        double sumAbsRes = 0.0;
        double sumSqRes = 0.0;
        qint64 edgeN = 0;
        double sumEdgeBias = 0.0;
        double sumAbsEdgeBias = 0.0;
        double sumBaseShift = 0.0;
        double sumAbsBaseShift = 0.0;
    };

    std::array<LegStats, 4> legs;
    std::complex<double> adjFieldCross = {0.0, 0.0};
    std::complex<double> sameFieldCross = {0.0, 0.0};
    qint64 adjFieldN = 0;
    qint64 sameFieldN = 0;

    auto sampleRow = [&](int line, int rel)->double {
        rel = std::clamp(rel, 0, width - 1);
        if (useLockedSource) {
            const double *row = locked1DSource_line(line);
            if (!row)
                return 0.0;
            return row[rel];
        }

        const int h = left + rel;
        return clpbuffer[srcBufIndex].pixel[line][h];
    };

    auto sampleIQ = [&](int line, int rel)->std::complex<double> {
        const int h = left + std::clamp(rel, 0, width - 1);
        const int ph = carrierSampleClass(line, h);
        double ti = 0.0;
        double tq = 0.0;

        const CombCarrierGrammar *grammar = carrierGrammarLine(line);
        const bool grammarLocked = grammar && grammar->grammarLocked;
        if (grammarLocked)
        {
            ti = (double)grammar->demodLUTTi[ph];
            tq = (double)grammar->demodLUTTq[ph];
        } else {
            double lutTi[4], lutTq[4];
            fusedDemodLUT(1.0, 0.0, spLUT_locked, cpLUT_locked, lutTi, lutTq);
            ti = lutTi[ph];
            tq = lutTq[ph];
        }

        const double c = sampleRow(line, rel);
        return { c * ti, c * tq };
    };

    auto sampleLockedIQ = [&](int line, int rel)->std::complex<double> {
        const int h = left + std::clamp(rel, 0, width - 1);
        const int ph = carrierSampleClass(line, h);
        double ti = 0.0;
        double tq = 0.0;

        const CombCarrierGrammar *grammar = carrierGrammarLine(line);
        if (grammar && grammar->grammarLocked)
        {
            ti = (double)grammar->demodLUTTi[ph];
            tq = (double)grammar->demodLUTTq[ph];
        } else {
            return {0.0, 0.0};
        }

        const double *row = locked1DSource_line(line);
        if (!row)
            return {0.0, 0.0};

        const double c = row[std::clamp(rel, 0, width - 1)];
        return { c * ti, c * tq };
    };

    auto lockedScalar = [&](int line, int rel)->double {
        const double *row = locked1DSource_line(line);
        if (!row)
            return 0.0;
        return row[std::clamp(rel, 0, width - 1)];
    };

    const double invI = invIreScale;

    for (int line = firstLine; line < lastLine; ++line) {
        if (line >= demodLines)
            continue;

        if (useLockedSource) {
            if (!locked1DSource_line(line))
                continue;
        }

        for (int rel = 4; rel < width - 4; ++rel) {
            const int phase = carrierSampleClass(line, left + rel);
            const std::complex<double> z  = sampleIQ(line, rel);
            const std::complex<double> zm = sampleIQ(line, rel - 4);
            const std::complex<double> zp = sampleIQ(line, rel + 4);
            const std::complex<double> r  = z - 0.5 * (zm + zp);
            const double resIRE = std::hypot(r.real(), r.imag()) * invI;

            LegStats &s = legs[phase];
            ++s.n;
            s.sumI += z.real() * invI;
            s.sumQ += z.imag() * invI;
            s.sumAbsRes += resIRE;
            s.sumSqRes += resIRE * resIRE;

            auto addPhaseCross = [&](int otherLine,
                                     std::complex<double> &sumCross,
                                     qint64 &count)
            {
                if (otherLine < firstLine || otherLine >= lastLine)
                    return;
                const std::complex<double> zo = sampleIQ(otherLine, rel);
                const double m0 = std::hypot(z.real(), z.imag()) * invI;
                const double mo = std::hypot(zo.real(), zo.imag()) * invI;
                constexpr double PHASE_COMPARE_MIN_IRE = 4.0;
                if (m0 < PHASE_COMPARE_MIN_IRE || mo < PHASE_COMPARE_MIN_IRE)
                    return;
                const double norm = 1.0 / (std::hypot(z.real(), z.imag()) *
                                           std::hypot(zo.real(), zo.imag()));
                sumCross += zo * std::conj(z) * norm;
                ++count;
            };
            addPhaseCross(line + 1, adjFieldCross, adjFieldN);
            addPhaseCross(line + 2, sameFieldCross, sameFieldN);

            if (!useLockedSource && line > firstLine && line + 1 < lastLine &&
                locked1DSource_line(line - 1) && locked1DSource_line(line + 1))
            {
                const double vEdgeIRE = std::max(
                    std::fabs(lockedScalar(line, rel) - lockedScalar(line - 1, rel)),
                    std::fabs(lockedScalar(line, rel) - lockedScalar(line + 1, rel))) * invI;
                constexpr double EDGE_GATE_IRE = 4.0;
                if (vEdgeIRE >= EDGE_GATE_IRE) {
                    const std::complex<double> z0 = sampleLockedIQ(line, rel);
                    const std::complex<double> zu = sampleLockedIQ(line - 1, rel);
                    const std::complex<double> zd = sampleLockedIQ(line + 1, rel);
                    const std::complex<double> edgeDir = zu - zd;
                    const double edgeMagSq = edgeDir.real() * edgeDir.real() +
                                             edgeDir.imag() * edgeDir.imag();
                    if (edgeMagSq <= 1e-12)
                        continue;

                    const double dUp = std::hypot((z - zu).real(), (z - zu).imag()) * invI;
                    const double dDn = std::hypot((z - zd).real(), (z - zd).imag()) * invI;
                    const double signedBias = dDn - dUp; // >0 means closer to upper line.
                    const std::complex<double> fromBase = z - z0;
                    const double signedBaseShift =
                        ((fromBase.real() * edgeDir.real()) +
                         (fromBase.imag() * edgeDir.imag())) /
                        std::sqrt(edgeMagSq) * invI; // >0 means 2D moved upward from 1D.
                    ++s.edgeN;
                    s.sumEdgeBias += signedBias;
                    s.sumAbsEdgeBias += std::fabs(signedBias);
                    s.sumBaseShift += signedBaseShift;
                    s.sumAbsBaseShift += std::fabs(signedBaseShift);
                }
            }
        }
    }

    QString msg = QString("PhaseLegStats %1 cadence=%2 fieldPhase=%3/%4")
        .arg(label)
        .arg(cadenceId)
        .arg(firstFieldPhaseID)
        .arg(secondFieldPhaseID);

    for (int phase = 0; phase < 4; ++phase) {
        const LegStats &s = legs[phase];
        if (s.n <= 0) {
            msg += QString(" p%1(n=0)").arg(phase);
            continue;
        }

        const double invN = 1.0 / (double)s.n;
        const double meanI = s.sumI * invN;
        const double meanQ = s.sumQ * invN;
        const double meanAbs = s.sumAbsRes * invN;
        const double rms = std::sqrt(s.sumSqRes * invN);

        msg += QString(" p%1(n=%2,meanIQ=%3/%4,resAbs=%5,resRms=%6)")
            .arg(phase)
            .arg(s.n)
            .arg(meanI, 0, 'f', 3)
            .arg(meanQ, 0, 'f', 3)
            .arg(meanAbs, 0, 'f', 3)
            .arg(rms, 0, 'f', 3);
    }

    auto meanComplex = [](const LegStats &s)->std::complex<double> {
        if (s.n <= 0) return {0.0, 0.0};
        const double invN = 1.0 / (double)s.n;
        return {s.sumI * invN, s.sumQ * invN};
    };
    auto meanAbsRes = [](const LegStats &s)->double {
        return (s.n > 0) ? (s.sumAbsRes / (double)s.n) : 0.0;
    };
    auto meanEdgeBias = [](const LegStats &s)->double {
        return (s.edgeN > 0) ? (s.sumEdgeBias / (double)s.edgeN) : 0.0;
    };
    auto meanAbsEdgeBias = [](const LegStats &s)->double {
        return (s.edgeN > 0) ? (s.sumAbsEdgeBias / (double)s.edgeN) : 0.0;
    };
    auto meanBaseShift = [](const LegStats &s)->double {
        return (s.edgeN > 0) ? (s.sumBaseShift / (double)s.edgeN) : 0.0;
    };
    auto meanAbsBaseShift = [](const LegStats &s)->double {
        return (s.edgeN > 0) ? (s.sumAbsBaseShift / (double)s.edgeN) : 0.0;
    };

    const std::complex<double> oddMean  = 0.5 * (meanComplex(legs[1]) + meanComplex(legs[3]));
    const std::complex<double> evenMean = 0.5 * (meanComplex(legs[0]) + meanComplex(legs[2]));
    const double oddRes  = 0.5 * (meanAbsRes(legs[1]) + meanAbsRes(legs[3]));
    const double evenRes = 0.5 * (meanAbsRes(legs[0]) + meanAbsRes(legs[2]));
    const std::complex<double> oddEvenDelta = oddMean - evenMean;

    const std::complex<double> lowPairMean  = 0.5 * (meanComplex(legs[0]) + meanComplex(legs[1]));
    const std::complex<double> highPairMean = 0.5 * (meanComplex(legs[2]) + meanComplex(legs[3]));
    const double lowPairRes  = 0.5 * (meanAbsRes(legs[0]) + meanAbsRes(legs[1]));
    const double highPairRes = 0.5 * (meanAbsRes(legs[2]) + meanAbsRes(legs[3]));
    const std::complex<double> lowHighDelta = lowPairMean - highPairMean;

    msg += QString(" oddEven(dIQ=%1/%2,dMag=%3,dRes=%4)")
        .arg(oddEvenDelta.real(), 0, 'f', 3)
        .arg(oddEvenDelta.imag(), 0, 'f', 3)
        .arg(std::hypot(oddEvenDelta.real(), oddEvenDelta.imag()), 0, 'f', 3)
        .arg(oddRes - evenRes, 0, 'f', 3);

    msg += QString(" lowHigh(dIQ=%1/%2,dMag=%3,dRes=%4)")
        .arg(lowHighDelta.real(), 0, 'f', 3)
        .arg(lowHighDelta.imag(), 0, 'f', 3)
        .arg(std::hypot(lowHighDelta.real(), lowHighDelta.imag()), 0, 'f', 3)
        .arg(lowPairRes - highPairRes, 0, 'f', 3);

    auto phaseDeg = [](std::complex<double> z)->double {
        return std::atan2(z.imag(), z.real()) * 180.0 / M_PI;
    };
    auto coherence = [](std::complex<double> z)->double {
        return std::min(1.0, std::hypot(z.real(), z.imag()));
    };
    const std::complex<double> adjMean =
        (adjFieldN > 0) ? (adjFieldCross / (double)adjFieldN) : std::complex<double>{0.0, 0.0};
    const std::complex<double> sameMean =
        (sameFieldN > 0) ? (sameFieldCross / (double)sameFieldN) : std::complex<double>{0.0, 0.0};
    msg += QString(" fieldPhaseIQ(adjN=%1,adjDeg=%2,adjCoh=%3,sameN=%4,sameDeg=%5,sameCoh=%6)")
        .arg(adjFieldN)
        .arg(phaseDeg(adjMean), 0, 'f', 2)
        .arg(coherence(adjMean), 0, 'f', 3)
        .arg(sameFieldN)
        .arg(phaseDeg(sameMean), 0, 'f', 2)
        .arg(coherence(sameMean), 0, 'f', 3);

    qint64 attrN = 0;
    double sumLumaClaim = 0.0;
    double sumChromaClaim = 0.0;
    double sumUncertainClaim = 0.0;
    double sumLumaIncursion = 0.0;
    double sumCandidateSpread = 0.0;
    double sumFrameCoherence = 0.0;
    double sumCarrierScale = 0.0;
    double sumCarrierConfidence = 0.0;
    double sumCarrierPlausibility = 0.0;
    double sumCarrierPhaseErrorAbs = 0.0;
    double sumPhaseScheduleConflict = 0.0;
    int scheduleConflictLines = 0;
    for (int line = firstLine; line < lastLine; ++line) {
        const AttributionEvidence *row = attributionEvidence_line(line);
        if (!row)
            continue;
        const CombCarrierGrammar *grammar = carrierGrammarLine(line);
        const double lineCarrierScale = grammar ? grammar->carrierScale : 0.0;
        const double lineCarrierConf  = grammar ? std::clamp(grammar->phaseConfidence, 0.0, 1.0) : 0.0;
        const double lineCarrierPlausibility = carrierPlausibility(grammar);
        const double lineCarrierPhase = grammar ? grammar->phaseError : 0.0;
        const double lineConflict = grammar ? grammar->phaseScheduleConflict : 0.0;
        if (lineConflict > 0.0) ++scheduleConflictLines;
        for (int rel = 0; rel < width; ++rel) {
            const AttributionEvidence &e = row[rel];
            ++attrN;
            sumLumaClaim += e.assessment.lumaClaim;
            sumChromaClaim += e.assessment.chromaClaim;
            sumUncertainClaim += e.assessment.uncertainClaim;
            sumLumaIncursion += e.facts.lumaIncursionRiskIRE;
            sumCandidateSpread += e.facts.candidateSpreadIRE;
            sumFrameCoherence += e.facts.frameIQCoherence;
            sumCarrierScale += lineCarrierScale;
            sumCarrierConfidence += lineCarrierConf;
            sumCarrierPlausibility += lineCarrierPlausibility;
            sumCarrierPhaseErrorAbs += std::fabs(lineCarrierPhase);
            sumPhaseScheduleConflict += lineConflict;
        }
    }
    if (attrN > 0) {
        const double invAttrN = 1.0 / (double)attrN;
        msg += QString(" attribution(n=%1,luma=%2,chroma=%3,uncertain=%4,incur=%5,spread=%6,frameCoh=%7,carScale=%8,carConf=%9,carPlaus=%10,carPhaseAbsDeg=%11,schedConf=%12,schedConfLines=%13)")
            .arg(attrN)
            .arg(sumLumaClaim * invAttrN, 0, 'f', 3)
            .arg(sumChromaClaim * invAttrN, 0, 'f', 3)
            .arg(sumUncertainClaim * invAttrN, 0, 'f', 3)
            .arg(sumLumaIncursion * invAttrN, 0, 'f', 3)
            .arg(sumCandidateSpread * invAttrN, 0, 'f', 3)
            .arg(sumFrameCoherence * invAttrN, 0, 'f', 3)
            .arg(sumCarrierScale * invAttrN, 0, 'f', 3)
            .arg(sumCarrierConfidence * invAttrN, 0, 'f', 3)
            .arg(sumCarrierPlausibility * invAttrN, 0, 'f', 3)
            .arg(sumCarrierPhaseErrorAbs * invAttrN * 180.0 / M_PI, 0, 'f', 3)
            .arg(sumPhaseScheduleConflict * invAttrN, 0, 'f', 3)
            .arg(scheduleConflictLines);
    }

    const qint64 oddEdgeN = legs[1].edgeN + legs[3].edgeN;
    const qint64 evenEdgeN = legs[0].edgeN + legs[2].edgeN;
    const double oddEdgeBias = (oddEdgeN > 0)
        ? ((legs[1].sumEdgeBias + legs[3].sumEdgeBias) / (double)oddEdgeN)
        : 0.0;
    const double evenEdgeBias = (evenEdgeN > 0)
        ? ((legs[0].sumEdgeBias + legs[2].sumEdgeBias) / (double)evenEdgeN)
        : 0.0;
    const double oddAbsEdgeBias = (oddEdgeN > 0)
        ? ((legs[1].sumAbsEdgeBias + legs[3].sumAbsEdgeBias) / (double)oddEdgeN)
        : 0.0;
    const double evenAbsEdgeBias = (evenEdgeN > 0)
        ? ((legs[0].sumAbsEdgeBias + legs[2].sumAbsEdgeBias) / (double)evenEdgeN)
        : 0.0;
    const double oddBaseShift = (oddEdgeN > 0)
        ? ((legs[1].sumBaseShift + legs[3].sumBaseShift) / (double)oddEdgeN)
        : 0.0;
    const double evenBaseShift = (evenEdgeN > 0)
        ? ((legs[0].sumBaseShift + legs[2].sumBaseShift) / (double)evenEdgeN)
        : 0.0;
    const double oddAbsBaseShift = (oddEdgeN > 0)
        ? ((legs[1].sumAbsBaseShift + legs[3].sumAbsBaseShift) / (double)oddEdgeN)
        : 0.0;
    const double evenAbsBaseShift = (evenEdgeN > 0)
        ? ((legs[0].sumAbsBaseShift + legs[2].sumAbsBaseShift) / (double)evenEdgeN)
        : 0.0;

    msg += QString(" edgeZip(oddN=%1,evenN=%2,oddPull=%3,evenPull=%4,pullDelta=%5,absPull=%6/%7,oddShift=%8,evenShift=%9,shiftDelta=%10,absShift=%11/%12)")
        .arg(oddEdgeN)
        .arg(evenEdgeN)
        .arg(oddEdgeBias, 0, 'f', 3)
        .arg(evenEdgeBias, 0, 'f', 3)
        .arg(oddEdgeBias - evenEdgeBias, 0, 'f', 3)
        .arg(oddAbsEdgeBias, 0, 'f', 3)
        .arg(evenAbsEdgeBias, 0, 'f', 3)
        .arg(oddBaseShift, 0, 'f', 3)
        .arg(evenBaseShift, 0, 'f', 3)
        .arg(oddBaseShift - evenBaseShift, 0, 'f', 3)
        .arg(oddAbsBaseShift, 0, 'f', 3)
        .arg(evenAbsBaseShift, 0, 'f', 3);

    for (int phase = 0; phase < 4; ++phase) {
        if (legs[phase].edgeN <= 0)
            continue;

        msg += QString(" ep%1(n=%2,pull=%3,absPull=%4,shift=%5,absShift=%6)")
            .arg(phase)
            .arg(legs[phase].edgeN)
            .arg(meanEdgeBias(legs[phase]), 0, 'f', 3)
            .arg(meanAbsEdgeBias(legs[phase]), 0, 'f', 3)
            .arg(meanBaseShift(legs[phase]), 0, 'f', 3)
            .arg(meanAbsBaseShift(legs[phase]), 0, 'f', 3);
    }

    qInfo().noquote() << msg;
}


// split2D dispatcher
void Comb::FrameBuffer::split2D()
{
    const bool writeWeights = configuration.showMap;
    const bool wantFvf = (configuration.twoDVariant == Comb::Configuration::TwoDVariant::FieldVsFrame);
    if (configuration.stageTimers && wantFvf) {
        fvfInstrumentation.reset();
    }
    const bool needFrameACompute = configuration.phaseCompensation &&
        (configuration.twoDVariant == Comb::Configuration::TwoDVariant::FrameAAdaptiveIQ ||
         wantFvf);
    const bool needFrameBCompute = configuration.phaseCompensation &&
        (configuration.twoDVariant == Comb::Configuration::TwoDVariant::FrameBDirectIQ ||
         wantFvf);
    const bool needFrameIQCompute = needFrameACompute || needFrameBCompute;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;

    if (width <= 0 || firstLine >= lastLine) return;

    if (configuration.twoDVariant == Comb::Configuration::TwoDVariant::Line) {
        for (int line = firstLine; line < lastLine; ++line) {
            double *dst = clpbuffer[1].pixel[line];
            const double *lockedRow = configuration.phaseCompensation
                ? locked1DSource_line(line) : nullptr;
            if (lockedRow) {
                for (int rel = 0; rel < width; ++rel) dst[left + rel] = lockedRow[rel];
            } else {
                const double *src1d = bucketScalar1D_line(line);
                for (int rel = 0; rel < width; ++rel) dst[left + rel] = src1d[left + rel];
            }
            if (writeWeights && line < (int)w2d_frame_weight.size())
                std::fill(w2d_frame_weight[line].begin(), w2d_frame_weight[line].end(), 0.0f);
        }
        reportPhaseLegStats("2d-final", 1, false);
        return;
    }

    // The preclean ring is populated on demand below with Field B output, then
    // reused by Frame A as a second-stage scalar cancellation source.
    precleanRingLine = { -1, -1, -1 };
    invalidateCombTapCache();

    // Determine which tap layers the selected variant needs, once for the frame.
    {
        using V = Comb::Configuration::TwoDVariant;
        switch (configuration.twoDVariant) {
        case V::FieldAContour:
            combTapBuildFlags_ = TapBuildFieldA | TapBuildFieldB;
            break;
        case V::FieldBSimple:
            combTapBuildFlags_ = TapBuildFieldB;
            break;
        case V::FrameAAdaptiveIQ:
            combTapBuildFlags_ = TapBuildFrame | TapBuildFieldA | TapBuildFieldB;
            break;
        case V::FrameBDirectIQ:
            combTapBuildFlags_ = TapBuildFrame | TapBuildFieldB;
            break;
        default:
            combTapBuildFlags_ = TapBuildAll;
            break;
        }
    }

    const bool vdisEnabled = configuration.tunables.VDIS_ENABLE;
    if (vdisEnabled) {
        if ((int)vdisMask.size() < lastLine) vdisMask.resize(lastLine);
        for (int line = firstLine; line < lastLine; ++line) {
            if (line >= demodLines) continue;
            computeVDISLine(line);
            if (!scratch_vdis_flag.empty() && line < (int)vdisMask.size()) {
                auto &row = vdisMask[line];
                if ((int)row.size() < width) row.assign(width, 0);
                for (int rel = 0; rel < width; ++rel) row[rel] = scratch_vdis_flag[rel] ? 1 : 0;
            }
        }
        consolidateVDISRegions(vdisMask, videoParameters);
    } else {
        vdisMask.clear();
    }

    std::vector<std::complex<double>> frameIQ;
    std::vector<std::complex<double>> frameAIQ;

    for (int line = firstLine; line < lastLine; ++line) {
        if (line >= demodLines) continue;

        ensureCombTapLine(line);
        const CombTapLine &tapLine = tapLineCache[precleanRingSlot(line)];

        if (needFrameIQCompute) {
            auto ensureFieldBPrecleanLine = [&](int ln) {
                if (ln < firstLine || ln >= lastLine) return;
                if (havePrecleanLine(ln, width)) return;
                double *preclean = precleanLinePtrMutable(ln, width);
                computeSimpleFieldLine(ensureCombTapLine(ln), preclean);
                double *gate = precleanGateLinePtrMutable(ln, width);
                std::fill(gate, gate + width, 1.0);
            };
            ensureFieldBPrecleanLine(line - 1);
            ensureFieldBPrecleanLine(line);
            ensureFieldBPrecleanLine(line + 1);
        }

        if (combTapBuildFlags_ & TapBuildFieldB) {
            const double *fieldBPreclean = precleanLinePtr(line, width);
            if (fieldBPreclean) {
                std::copy(fieldBPreclean, fieldBPreclean + width, scratch_lineWorkC.begin());
            } else {
                computeSimpleFieldLine(tapLine, scratch_lineWorkC.data());
            }
        } else {
            std::fill(scratch_lineWorkC.begin(), scratch_lineWorkC.begin() + width, 0.0);
        }
        if (combTapBuildFlags_ & TapBuildFieldA) {
            computeContourFieldLine(tapLine, scratch_lineWorkA.data(), scratch_lineWorkB.data());
        } else {
            std::fill(scratch_lineWorkA.begin(), scratch_lineWorkA.begin() + width, 0.0);
            std::fill(scratch_lineWorkB.begin(), scratch_lineWorkB.begin() + width, 1.0);
        }

        {
            const double *src1d = configuration.phaseCompensation
                                  ? nullptr
                                  : bucketScalar1D_line(line);
            if ((int)scratch_lateralLine.size() < width)
                scratch_lateralLine.resize(width);
            if (configuration.phaseCompensation) {
                const double *lockedRow = locked1DSource_line(line);
                if (lockedRow) {
                    std::copy(lockedRow, lockedRow + width, scratch_lateralLine.begin());
                }
                else {
                    std::fill(scratch_lateralLine.begin(), scratch_lateralLine.begin() + width, 0.0);
                }
            } else {
                for (int rel = 0; rel < width; ++rel)
                    scratch_lateralLine[rel] = src1d[left + rel];
            }
        }

        if (needFrameACompute) {
            computeFrameAAdaptiveIQLine(line, frameAIQ);
            if ((int)scratch_frameAAdaptiveIQComposite.size() < width)
                scratch_frameAAdaptiveIQComposite.resize(width);
            for (int rel = 0; rel < width; ++rel) {
                const int h = left + rel;
                if (rel < (int)frameAIQ.size()) {
                    const auto &Z = frameAIQ[rel];
                    scratch_frameAAdaptiveIQComposite[rel] = remod4fscToCompositePhase(Z.real(), Z.imag(), carrierSampleClass(line, h));
                } else {
                    scratch_frameAAdaptiveIQComposite[rel] = 0.0;
                }
            }
        }

        if (needFrameBCompute) {
            computeFrameBDirectIQCompositeLine(line, frameIQ, scratch_frameBDirectIQComposite);
            if ((int)scratch_frameBDirectIQComposite.size() < width)
                scratch_frameBDirectIQComposite.resize(width);
        }

        const std::vector<double> &frameAttrScalar =
            needFrameBCompute ? scratch_frameBDirectIQComposite : scratch_frameAAdaptiveIQComposite;
        const std::vector<std::complex<double>> *frameAttrIQ =
            needFrameBCompute ? &frameIQ : (needFrameACompute ? &frameAIQ : nullptr);
        collectCombAttributionEvidence(
            line,
            scratch_lineWorkA.data(),
            scratch_lineWorkC.data(),
            needFrameIQCompute ? frameAttrScalar : scratch_frameBDirectIQComposite,
            frameAttrIQ);

        double *dst = clpbuffer[1].pixel[line];
        auto emitSelected = [&](int rel, double v) {
            dst[left + rel] = v;
        };

        if (configuration.twoDVariant == Comb::Configuration::TwoDVariant::FieldAContour) {
            for (int rel = 0; rel < width; ++rel) emitSelected(rel, scratch_lineWorkA[rel]);
            if (writeWeights) std::fill(w2d_frame_weight[line].begin(), w2d_frame_weight[line].end(), 0.0f);
        }
        else if (configuration.twoDVariant == Comb::Configuration::TwoDVariant::FieldBSimple) {
            for (int rel = 0; rel < width; ++rel) emitSelected(rel, scratch_lineWorkC[rel]);
            if (writeWeights) std::fill(w2d_frame_weight[line].begin(), w2d_frame_weight[line].end(), 0.35f);
        }
        else if (configuration.twoDVariant == Comb::Configuration::TwoDVariant::FrameAAdaptiveIQ && configuration.phaseCompensation) {
            for (int rel = 0; rel < width; ++rel) emitSelected(rel, scratch_frameAAdaptiveIQComposite[rel]);
            if (writeWeights) std::fill(w2d_frame_weight[line].begin(), w2d_frame_weight[line].end(), 0.8f);
        }
        else if (configuration.twoDVariant == Comb::Configuration::TwoDVariant::FrameBDirectIQ && configuration.phaseCompensation) {
            for (int rel = 0; rel < width; ++rel) emitSelected(rel, scratch_frameBDirectIQComposite[rel]);
            if (writeWeights) std::fill(w2d_frame_weight[line].begin(), w2d_frame_weight[line].end(), 0.85f);
        }
        else {
            if (!configuration.phaseCompensation) {
                for (int rel = 0; rel < width; ++rel) {
                    dst[left + rel] = scratch_lineWorkC[rel];
                    if (writeWeights && line < (int)w2d_frame_weight.size()) {
                        w2d_frame_weight[line][rel] = 0.35f; 
                    }
                }
            } else {
                scoreFieldVsFrame(
                    line,
                    tapLine,
                    scratch_lineWorkC.data(),
                    scratch_frameAAdaptiveIQComposite,
                    &scratch_frameBDirectIQComposite,
                    scratch_outMixed.data(),
                    writeWeights,
                    scratch_lateralLine.data(),
                    &frameIQ);

                for (int rel = 0; rel < width; ++rel) {
                    double vMixed = scratch_outMixed[rel];

                    // Keep only the numeric-sanity fallback:
                    if (!std::isfinite(vMixed)) vMixed = scratch_lineWorkC[rel];

                    emitSelected(rel, vMixed);
                }
                }
        }
    }

    reportPhaseLegStats("2d-final", 1, false);
}

// 3D temporal adaptive
void Comb::FrameBuffer::split3D(const FrameBuffer &previousFrame,
                                const FrameBuffer &nextFrame)
{
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;

    auto clampH = [&](int idx)->int { return std::clamp(idx, left, right - 1); };

    for (int line = firstLine; line < lastLine; ++line) {
        bool lineHasVDIS = (line < (int)vdisMask.size());
        
        for (int h = left; h < right; ++h) {
            const int rel = h - left;
        
            if (lineHasVDIS) {
                int maskVal = (rel < (int)vdisMask[line].size()) ? vdisMask[line][rel] : 0;
                if (maskVal == 2) continue; // Keep pre-filled 2D value
            }
        
            qint32 bestIndex;
            double bestSample;
            
            // Pass *this as well so getBestCandidate knows context
            getBestCandidate(line, h, previousFrame, nextFrame, bestIndex, bestSample);
        
            const int h0 = clampH(h);
            const int rel0 = h0 - left;
            double base1d;
            const double *lockedRow = configuration.phaseCompensation
                ? locked1DSource_line(line) : nullptr;
            if (lockedRow && rel0 >= 0)
            {
                base1d = lockedRow[rel0];
            } else {
                base1d = bucketScalar1D_line(line)[h0];
            }
        
            if (bestIndex < CAND_PREV_FIELD) {
                 // Best is 1D/2D; keep pre-filled 2D value
                 // clpbuffer[2] already contains clpbuffer[1]
            } else {
                // Temporal: classic (Y+C) - (Y-C) / 2
                // We overwrite the pre-filled 2D value with the temporal result
                clpbuffer[2].pixel[line][h] = (base1d - bestSample) * 0.5;
            }
        }
    }
}

void Comb::FrameBuffer::getBestCandidate(qint32 lineNumber, qint32 h,
                                         const FrameBuffer &previousFrame,
                                         const FrameBuffer &nextFrame,
                                         qint32 &bestIndex, double &bestSample) const
{
    Candidate c[NUM_CANDIDATES];
    const FrameBuffer* src[NUM_CANDIDATES] = { nullptr };

    static constexpr double LINE_BONUS  = -2.0;
    static constexpr double FIELD_BONUS = -4.0;
    static constexpr double FRAME_BONUS = -5.0;

    // 1D/2D Candidates (Always available via 'this')
    c[CAND_LEFT]  = getCandidate(lineNumber, h, *this, lineNumber, h - 2, 0.0);
    src[CAND_LEFT] = this;
    c[CAND_RIGHT] = getCandidate(lineNumber, h, *this, lineNumber, h + 2, 0.0);
    src[CAND_RIGHT] = this;
    c[CAND_UP]   = getCandidate(lineNumber, h, *this, lineNumber - 2, h, LINE_BONUS);
    src[CAND_UP] = this;
    c[CAND_DOWN] = getCandidate(lineNumber, h, *this, lineNumber + 2, h, LINE_BONUS);
    src[CAND_DOWN] = this;

    // Previous and next field candidates are evaluated independently so a valid
    // prev does not force a symmetric next evaluation.
    
    const bool frameVerticalAllowed = carrierFrameVerticalAllowed(lineNumber);

    // --- Previous Field ---
    bool prevValid = false;
    if (frameVerticalAllowed && lineNumber - 1 >= videoParameters.firstActiveFrameLine) {
        // Inter-field (line above in previous field vs line above in this field)
        // If phases match, the info is in previousFrame. If phases flip, it's in this frame.
        if (carrierLineFlip(lineNumber) == carrierLineFlip(lineNumber - 1)) {
             c[CAND_PREV_FIELD] = getCandidate(lineNumber, h, previousFrame, lineNumber - 1, h, FIELD_BONUS);
             src[CAND_PREV_FIELD] = &previousFrame;
        } else {
             c[CAND_PREV_FIELD] = getCandidate(lineNumber, h, *this, lineNumber - 1, h, FIELD_BONUS);
             src[CAND_PREV_FIELD] = this;
        }
        prevValid = true;
    }
    
    // --- Next Field ---
    // Note: We don't force symmetry. If Prev is valid and Next isn't, we still evaluate Prev.
    if (frameVerticalAllowed && lineNumber + 1 < videoParameters.lastActiveFrameLine) {
        if (carrierLineFlip(lineNumber) == carrierLineFlip(lineNumber + 1)) {
             c[CAND_NEXT_FIELD] = getCandidate(lineNumber, h, nextFrame, lineNumber + 1, h, FIELD_BONUS);
             src[CAND_NEXT_FIELD] = &nextFrame;
        } else {
             c[CAND_NEXT_FIELD] = getCandidate(lineNumber, h, *this, lineNumber + 1, h, FIELD_BONUS);
             src[CAND_NEXT_FIELD] = this;
        }
    }

    // --- Temporal Frame Center (Prev/Next Frame) ---
    // This is where the 8-field cycle breaks 3D. 
    // We explicitly check if the target frame exists AND matches phase.
    
    // Check Previous Frame Phase Match
    // We compare this->linePhase vs prev->linePhase at same line.
    // If they match, it's a valid candidate. If they differ (decimation/cut), it's garbage.
    if (carrierLineFlip(lineNumber) == previousFrame.carrierLineFlip(lineNumber)) {
        c[CAND_PREV_FRAME] = getCandidate(lineNumber, h, previousFrame, lineNumber, h, FRAME_BONUS);
        src[CAND_PREV_FRAME] = &previousFrame;
    } else {
        c[CAND_PREV_FRAME].penalty = 1000.0;
        src[CAND_PREV_FRAME] = nullptr;
    }

    // Check Next Frame Phase Match independently
    if (carrierLineFlip(lineNumber) == nextFrame.carrierLineFlip(lineNumber)) {
        c[CAND_NEXT_FRAME] = getCandidate(lineNumber, h, nextFrame, lineNumber, h, FRAME_BONUS);
        src[CAND_NEXT_FRAME] = &nextFrame;
    } else {
        c[CAND_NEXT_FRAME].penalty = 1000.0;
        src[CAND_NEXT_FRAME] = nullptr;
    }

    // Agreement shaping logic
    if (configuration.dimensions == 3 && configuration.adaptive) {
        const double ref2d = clpbuffer[1].pixel[lineNumber][h];
        const auto &T = configuration.tunables;

        for (int i = 0; i < NUM_CANDIDATES; ++i) {
            const FrameBuffer* s = src[i];
            if (!s || s == this || c[i].penalty >= 1000.0) continue;

            double dIRE = std::fabs(c[i].sample - ref2d) / irescale;
            double delta = 0.0;

            if (dIRE <= T.AGREEMENT_REWARD_RADIUS_IRE) {
                double x = dIRE / T.AGREEMENT_REWARD_RADIUS_IRE;
                delta = - (T.AGREEMENT_REWARD_MAX * configuration.adaptThreshold) * (1.0 - x * x);
            } else if (dIRE <= T.deviationThreshold) {
                delta = 0.0;
            } else {
                delta = T.AGREEMENT_VETO_BASE
                      + T.deviationPenalty * (dIRE - T.deviationThreshold);
            }
            c[i].penalty += delta;
        }
    }

    // Select best
    if (configuration.adaptive) {
        int best = 0;
        for (int i = 1; i < NUM_CANDIDATES; ++i) {
            if (c[i].penalty < c[best].penalty) best = i;
        }
        bestIndex = best;
    } else {
        // Non-adaptive fallback: prefer Previous Frame if valid, else Next, else 2D
        if (src[CAND_PREV_FRAME]) bestIndex = CAND_PREV_FRAME;
        else if (src[CAND_NEXT_FRAME]) bestIndex = CAND_NEXT_FRAME;
        else bestIndex = CAND_UP; // Fallback to 2D
    }

    bestSample = c[bestIndex].sample;
}

// Bucket-path demodulation: separates I and Q from the comb-filtered composite
// using the 4fsc sampling structure directly. At 4 subcarrier, samples fall on
// fixed phase positions (0, 90, 180, 270), so I and Q can be extracted by
// routing each sample into the appropriate accumulator via carrierSampleClass.
// Y is initialised to the raw composite here; adjustY() subtracts the chroma
// estimate afterwards. This path does not perform burst detection or phase
// correction  it relies on the 4fsc sampling assumption holding exactly.
void Comb::FrameBuffer::splitIQ()
{
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;

    for (qint32 lineNumber = firstLine; lineNumber < lastLine; lineNumber++) {
        const quint16 *line = rawbuffer.data() + (lineNumber * videoParameters.fieldWidth);

        double *Y = componentFrame->y(lineNumber);
        double *I = componentFrame->u(lineNumber);
        double *Q = componentFrame->v(lineNumber);

        // Apply per-line subcarrier polarity flip from carrierGrammar (populated in loadFields).
        const int f = carrierLineFlip(lineNumber);

        double si = 0, sq = 0;
        for (qint32 h = videoParameters.activeVideoStart;
             h < videoParameters.activeVideoEnd; h++)
        {
            qint32 phase = carrierSampleClass(lineNumber, h);

            double cavg = clpbuffer[configuration.dimensions - 1].pixel[lineNumber][h];
            cavg *= (double)f; // apply flip

            switch (phase) {
                case 0: sq =  cavg; break;
                case 1: si = -cavg; break;
                case 2: sq = -cavg; break;
                case 3: si =  cavg; break;
                default: break;
            }

            Y[h] = line[h];
            I[h] = si;
            Q[h] = sq;
        }
    }
}



// Bucket-path Y reconstruction: subtracts the chroma estimate (reconstructed
// from the I and Q buckets) from the raw composite to yield luma. The chroma
// remodulation reverses the bucket demod  routing each sample through the
// same switch on carrierSampleClass with sign inversion  and applies the per-line
// subcarrier polarity from carrierLineFlip. Only called in bucket mode
// (phaseCompensation == false); the locked path uses produceY() instead.
void Comb::FrameBuffer::adjustY()
{
    if (configuration.phaseCompensation) return;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;

    for (int line = firstLine; line < lastLine; ++line) {
        double *Y = componentFrame->y(line);
        double *I = componentFrame->u(line);
        double *Q = componentFrame->v(line);

        const int f = carrierLineFlip(line);
        for (int h = left; h < right; ++h) {
            double comp = 0.0;
            switch (carrierSampleClass(line, h)) {
                case 0: comp = -Q[h]; break;
                case 1: comp =  I[h]; break;
                case 2: comp =  Q[h]; break;
                case 3: comp = -I[h]; break;
            }
            comp *= -f;
            Y[h] -= comp;
        }
    }
}

// Bucket-path chroma low-pass filter: applies a symmetric FIR to the I and Q
// planes to remove high-frequency luma leakage left after the bucket demod.
// Not used in the locked path, which applies bandwidth-tailored FIRs in
// filterIQLocked instead.
void Comb::FrameBuffer::filterIQ()
{
    auto iqFilter = makeFIRFilter(c_colorlp_b);
    const int width = videoParameters.activeVideoEnd - videoParameters.activeVideoStart;

    if ((int)scratch_lineWorkD.size() < width) scratch_lineWorkD.assign(width, 0.0);
    double *temp = scratch_lineWorkD.data();

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int startH    = videoParameters.activeVideoStart;

    for (int line = firstLine; line < lastLine; ++line) {
        double *I = componentFrame->u(line) + startH;
        double *Q = componentFrame->v(line) + startH;

        iqFilter.apply(I, temp, width);
        std::copy(temp, temp + width, I);
        iqFilter.apply(Q, temp, width);
        std::copy(temp, temp + width, Q);
    }
}

// Chroma noise reduction (coring): high-pass filters I and Q with a narrow FIR,
// then hard-clamps the result to cNRLevel IRE, and subtracts the clamped
// high-frequency component. Suppresses chroma noise without affecting the
// broad chroma spectrum. Operates on the I/Q planes in place.
void Comb::FrameBuffer::doCNR()
{
    if (configuration.cNRLevel == 0.0) return;

    double nr_c = configuration.cNRLevel * irescale;
    auto iFilter(f_nrc);
    auto qFilter(f_nrc);
    const int delay = c_nrc_b.size() / 2;

    const int extSize = videoParameters.activeVideoEnd + delay;
    if ((int)scratch_hpI.size() < extSize) scratch_hpI.assign(extSize, 0.0);
    if ((int)scratch_hpQ.size() < extSize) scratch_hpQ.assign(extSize, 0.0);
    double *hpI = scratch_hpI.data();
    double *hpQ = scratch_hpQ.data();

    for (int line = videoParameters.firstActiveFrameLine;
         line < videoParameters.lastActiveFrameLine; ++line)
    {
        double *I = componentFrame->u(line);
        double *Q = componentFrame->v(line);

        for (int h = videoParameters.activeVideoStart - delay;
             h < videoParameters.activeVideoStart; ++h) { iFilter.feed(0.0); qFilter.feed(0.0); }

        for (int h = videoParameters.activeVideoStart;
             h < videoParameters.activeVideoEnd; ++h) {
            hpI[h] = iFilter.feed(I[h]);
            hpQ[h] = qFilter.feed(Q[h]);
        }

        for (int h = videoParameters.activeVideoEnd;
             h < videoParameters.activeVideoEnd + delay; ++h) {
            hpI[h] = iFilter.feed(0.0);
            hpQ[h] = qFilter.feed(0.0);
        }

        for (int h = videoParameters.activeVideoStart;
             h < videoParameters.activeVideoEnd; ++h) {
            double ai = hpI[h + delay];
            double aq = hpQ[h + delay];
            if (std::fabs(ai) > nr_c) ai = (ai > 0) ? nr_c : -nr_c;
            if (std::fabs(aq) > nr_c) aq = (aq > 0) ? nr_c : -nr_c;
            I[h] -= ai;
            Q[h] -= aq;
        }
    }
}

// Luma noise reduction (coring): same coring approach as doCNR but applied to
// the Y plane. High-pass filters Y and subtracts any component within yNRLevel
// IRE, attenuating fine-grain luma noise while preserving picture detail.
void Comb::FrameBuffer::doYNR()
{
    if (configuration.yNRLevel == 0.0) return;

    double nr_y = configuration.yNRLevel * irescale;
    auto yFilter(f_nr);
    const int delay = c_nr_b.size() / 2;

    const int extSize = videoParameters.activeVideoEnd + delay;
    if ((int)scratch_hpY.size() < extSize) scratch_hpY.assign(extSize, 0.0);
    double *hpY = scratch_hpY.data();

    for (int line = videoParameters.firstActiveFrameLine;
         line < videoParameters.lastActiveFrameLine; ++line)
    {
        double *Y = componentFrame->y(line);

        for (int h = videoParameters.activeVideoStart - delay;
             h < videoParameters.activeVideoStart; ++h) yFilter.feed(0.0);

        for (int h = videoParameters.activeVideoStart;
             h < videoParameters.activeVideoEnd; ++h) hpY[h] = yFilter.feed(Y[h]);

        for (int h = videoParameters.activeVideoEnd;
             h < videoParameters.activeVideoEnd + delay; ++h) hpY[h] = yFilter.feed(0.0);

        for (int h = videoParameters.activeVideoStart;
             h < videoParameters.activeVideoEnd; ++h) {
            double a = hpY[h + delay];
            if (std::fabs(a) > nr_y) a = (a > 0) ? nr_y : -nr_y;
            Y[h] -= a;
        }
    }
}

// Final chroma rotation and gain: rotates the I/Q plane by chromaPhase degrees
// and scales by chromaGain, converting from the internal demod basis to the
// standard Y'UV colour axes. The two paths use different base rotation angles
// because they produce I/Q in different reference frames: the locked path
// (splitIQlocked / filterIQLocked) produces chroma aligned to the burst-locked
// LO (base 70), while the bucket path (splitIQ / filterIQ) produces chroma
// aligned to the 4fsc sampling grid (base 33).
void Comb::FrameBuffer::transformIQ(double chromaGain, double chromaPhase)
{
    if (configuration.phaseCompensation) {
        constexpr double BASE_LOCKED = 70.0;
        const double theta = (BASE_LOCKED + chromaPhase) * M_PI / 180.0;
        const double c = std::cos(theta);
        const double s = std::sin(theta);

        // see namespace top of file for user control
        const double g = chromaGain * PRODUCT_CHROMA_SCALE;

        const int firstLine = videoParameters.firstActiveFrameLine;
        const int lastLine  = videoParameters.lastActiveFrameLine;
        const int left      = videoParameters.activeVideoStart;
        const int right     = videoParameters.activeVideoEnd;

        for (int line = firstLine; line < lastLine; ++line) {
            double *I = componentFrame->u(line);
            double *Q = componentFrame->v(line);
            for (int h = left; h < right; ++h) {
                const double ti = I[h];
                const double tq = Q[h];
                I[h] = (ti * c - tq * s) * g;
                Q[h] = (ti * s + tq * c) * g;
            }
        }
        return;
    }

    constexpr double BASE_BUCKET = 33.0;
    const double theta = (BASE_BUCKET + chromaPhase) * M_PI / 180.0;

    // see namespace top of file for user control
    const double gBucket = chromaGain * BUCKET_CHROMA_SCALE;

    const double bp = std::sin(theta) * gBucket;
    const double bq = std::cos(theta) * gBucket;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;

    for (int line = firstLine; line < lastLine; ++line) {
        double *I = componentFrame->u(line);
        double *Q = componentFrame->v(line);
        for (int h = left; h < right; ++h) {
            const double Ii = I[h];
            const double Qi = Q[h];
            I[h] = (-bp * Ii) + (bq * Qi);
            Q[h] = ( bq * Ii) + (bp * Qi);
        }
    }
}

// Debug overlay: paints each pixel of the U/V planes with a colour indicating
// which candidate won the 3D election at that position (red = 1D/lateral,
// yellow = 2D vertical, green = field, blue/purple = previous/next frame).
// Useful for diagnosing candidate selection behaviour on problem content.
void Comb::FrameBuffer::overlayMap(const FrameBuffer &previousFrame,
                                   const FrameBuffer &nextFrame)
{
    if (!componentFrame) return;

    FrameCanvas canvas(*componentFrame, videoParameters);

    FrameCanvas::Colour shades[NUM_CANDIDATES];
    for (int i = 0; i < NUM_CANDIDATES; ++i) {
        quint32 s = CANDIDATE_SHADES[i];
        shades[i] = canvas.rgb(((s >> 16) & 0xff) << 8,
                               ((s >> 8)  & 0xff) << 8,
                               ((s      ) & 0xff) << 8);
    }

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;

    for (int line = firstLine; line < lastLine; ++line) {
        double *U = componentFrame->u(line);
        double *V = componentFrame->v(line);
        for (int h = left; h < right; ++h) {
            qint32 bestIndex;
            double bestSample;
            getBestCandidate(line, h, previousFrame, nextFrame, bestIndex, bestSample);
            U[h] = shades[bestIndex].u;
            V[h] = shades[bestIndex].v;
        }
    }
}
