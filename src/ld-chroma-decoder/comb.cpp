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

    auto next     = std::make_unique<FrameBuffer>(videoParameters, configuration);
    auto current  = std::make_unique<FrameBuffer>(videoParameters, configuration);
    auto previous = std::make_unique<FrameBuffer>(videoParameters, configuration);

    const qint32 preStart = (configuration.dimensions == 3) ? (startIndex - 4)
                                                            : (startIndex - 2);

    for (qint32 fieldIndex = preStart; fieldIndex < endIndex; fieldIndex += 2) {
        // Rotate buffers
        {
            auto recycle = std::move(previous);
            previous = std::move(current);
            current  = std::move(next);
            next     = std::move(recycle);
        }

        // Preload "next" framebuffer. Guard against out-of-range indices; the caller
        // is responsible for padding inputFields with blank frames at the boundaries.
        bool canLoadNext = (fieldIndex + 3 < inputFields.size());

        if (canLoadNext && (fieldIndex + 2 >= 0)) {
            next->loadFields(inputFields[fieldIndex + 2], inputFields[fieldIndex + 3]);
            if (configuration.phaseCompensation) {
                // Locked path: the LS carrier model + line-to-line cancellation
                // replaces split1D's blind bandpass entirely.
                //   phaseLocked        → burst grammar, baseY4, raw TRI/TRQ, affine
                //   buildCarrierRetracted → per-line LS carrier fit, flatFloor,
                //                           combed carrier (alien-Y rejected)
                //   split2D → buildPhaseCorrected1D demods the combed carrier
                next->phaseLocked();
                next->buildCarrierRetracted();
            } else {
                // Bucket path: still needs the blind bandpass.
                next->split1D();
            }

            next->split2D();
        }

        if (fieldIndex < startIndex)
            continue;

        // 3D temporal stage (if requested)
        bool isStartUp = (fieldIndex < startIndex + 4); 
        if (configuration.dimensions == 3) {
            current->copy2DTo3D(); 
        }
        
        if (configuration.dimensions == 3 && !isStartUp) {
            // Now refine 3D buffer with temporal candidates where possible
            current->split3D(*previous, *next);
        }

        // Wire up temporal context for Residual Y if enabled
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

        // Coherent Demod + Y pipeline
        if (configuration.phaseCompensation) {
            current->demodMode = FrameBuffer::DemodMode::Locked;

            // 1) Carrier grammar + LS carrier fit + line cancellation (pre-pass)
            // 2) Combed-carrier demod for 1D, 2D scoring (via buildPhaseCorrected1D)
            // 3) Full 2D/3D selection -> clpbuffer[dimensions-1]
            // 4) Re-demod final selected comb and produce locked-product cache
            current->splitIQlocked();
            current->doCNR();
            current->produceY();
            current->filterIQLocked();
            current->doYNR();
            current->transformIQ(configuration.chromaGain, configuration.chromaPhase);
        } else {
            current->demodMode = FrameBuffer::DemodMode::Bucket;
            current->splitIQ();
            current->adjustY();
            current->filterIQ();
            current->doCNR();
            current->doYNR();
            current->transformIQ(configuration.chromaGain, configuration.chromaPhase);
        }

        // 3D map overlay (IQ candidate map)
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

            // Glyph metrics
            const int scale = 4;
            const int charW = 5 * scale;
            const int charH = 7 * scale;
            const int pad   = 4;
            const int boxH  = charH + 2 * pad;

            const int xBase = videoParameters.activeVideoStart + 32;
            const int yBase = videoParameters.firstActiveFrameLine + 32;

            FrameCanvas::Colour fg = canvas.rgb(65535, 65535, 65535);
            FrameCanvas::Colour bg = canvas.rgb(0, 0, 0);

            // Build the sequence of characters to display.
            // Convention:
            //   editTop    -> '/' leads  the frame:  /A  or /AB
            //   editBottom -> '/' splits the frame:  A/B
            //   -2 cadenceId -> "i" (confirmed interlaced)
            //   -3 cadenceId -> "p" (confirmed progressive)
            //   unknown    -> cycle-position digit

            // Determine display label for each field position.
            // Returns a 1-char label: film letter, digit, 'i', or 'p'.
            auto fieldLabel = [&](int cid, int fallbackCyclePos) -> char {
                if (cid == -2) return 'i';
                if (cid == -3) return 'p';
                if (cadenceKnown(cid)) return cadenceFilmLetter(cid);
                return static_cast<char>('0' + ((fallbackCyclePos % 5) + 1));
            };

            const int cyclePos = frameIndex;
            const char labelTop    = fieldLabel(cidTop,    cyclePos);
            const char labelBottom = fieldLabel(cidBottom, cyclePos);

            // Determine whether the two fields show the same label (pure frame)
            const bool pureFrame = (labelTop == labelBottom)
                                && (cidTop >= -3) && (cidBottom >= -3)
                                && !editTop && !editBottom; 

            // Count characters needed: labels + optional slashes
            // Sequence:
            //   editTop:              '/' labelTop [labelBottom if mixed]
            //   editBottom&&!editTop: labelTop '/' labelBottom  (mixed only)
            //   neither:              labelTop [labelBottom if mixed]
            // For pure frames the bottom label is suppressed.

            int numChars = 0;
            if (editTop)                    numChars++; // leading '/'
            numChars++;                                 // top label
            if (!pureFrame) {
                if (editBottom && !editTop) numChars++; // mid '/'
                numChars++;                             // bottom label
            }

            const int totalW = pad + numChars * (charW + pad);
            canvas.fillRectangle(xBase, yBase, totalW, boxH, bg);

            int xOff = xBase + pad;

            auto drawNext = [&](char c) {
                drawChar(canvas, xOff, yBase + pad, c, fg, scale);
                xOff += charW + pad;
            };

            if (editTop)    drawNext('/');
            drawNext(labelTop);
            if (!pureFrame) {
                if (editBottom && !editTop) drawNext('/');
                drawNext(labelBottom);
            }
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
            (configuration.twoDVariant == Comb::Configuration::TwoDVariant::FramePreclean ||
             configuration.twoDVariant == Comb::Configuration::TwoDVariant::FrameRaw ||
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
            locked1DSource.assign(lines, std::vector<double>(width, 0.0));
            ownershipEvidence.assign(lines, std::vector<OwnershipEvidence>(width));
        
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
        scratch_fieldLine.assign(width, 0.0);
        scratch_fieldGate.assign(width, 1.0);
        scratch_fieldBLine.assign(width, 0.0);
        scratch_outMixed.assign(width, 0.0);
        scratch_lateralLine.assign(width, 0.0);
        // low-res luma (chroma cancelled fsc)        
        scratch_lumaBaseY4.assign(width, 0.0);
        scratch_lumaHiRaw.assign(width, 0.0);
        scratch_lumaSmooth.assign(width, 0.0);

        // Filtering/NR temporaries
        scratch_filter_temp.assign(width, 0.0);
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
        demodTI_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
        demodTQ_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
        demodTI4fsc_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
        demodTQ4fsc_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
        locked1DTI4fsc_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
        locked1DTQ4fsc_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
        lockedProductI_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
        lockedProductQ_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
        demodTRI_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
        demodTRQ_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
        scratch_comp_res.assign(width, 0.0);
        scratch_frameBCenter.assign(width, 0.0);
        scratch_fieldBCenter.assign(width, 0.0);
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
    
    auto mergeCadenceForComb = [&](qint32 a, qint32 b) -> qint32 {
        // If the edit split happens between these two fields, force Video mode.
        if (editSplit) return -2;
    
        const bool aFilm = (a >= 0);
        const bool bFilm = (b >= 0);
        if (aFilm && bFilm) return (a < b) ? a : b;
        if (aFilm) return a;
        if (bFilm) return b;
        if (a == -3 || b == -3) return -3;
        return -2;
    };
    
    cadenceId = mergeCadenceForComb(cidA, cidB);
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

    // --- Initialize per-line grammar with full schedule identity ---
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    if ((int)carrierGrammar.size() < last) carrierGrammar.resize(last);
    for (int line = first; line < last; ++line) {
        CombCarrierGrammar &grammar = carrierGrammar[line];
        grammar = CombCarrierGrammar{};

        grammar.fieldPhaseId = getFieldID(line);
        grammar.lineParity = line & 1;
        grammar.fieldLine = line / 2;

        const bool positiveOnEven =
            (grammar.fieldPhaseId == 1) || (grammar.fieldPhaseId == 4);
        const bool evenFieldLine = ((grammar.fieldLine & 1) == 0);
        const bool linePhase = evenFieldLine ? positiveOnEven : !positiveOnEven;

        grammar.lineFlip = linePhase ? -1 : +1;
        grammar.samplePhase0 = 0;
        // lineFlip is derived from fieldPhaseId which comes from capture metadata,
        // so it holds Metadata authority.  rigidScheduleLineFlip mirrors it for
        // now because no independent rigid derivation is available at this stage;
        // phaseLocked() should update phaseScheduleConflict if burst measurement
        // diverges.
        grammar.lineFlipAuthority    = lddecode::CarrierPhaseAuthority::Metadata;
        grammar.rigidScheduleLineFlip = grammar.lineFlip;
        grammar.phaseScheduleConflict = 0.0;
        // If the paired fields already straddle an edit boundary, cross-field
        // vertical reasoning for this frame should be treated as schedule-invalid.
        grammar.frameVerticalAllowed = !editSplit;
    }

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
// Bucket-path only; the locked path skips this and uses the LS combed carrier
// from buildCarrierRetracted() instead.  Result written to clpbuffer[0].
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


void Comb::FrameBuffer::seedCombOwnershipPerLine(int line)
{
    const int right = videoParameters.activeVideoEnd;
    const int width = right - videoParameters.activeVideoStart;

    if (width <= 0 || line < 0)
        return;

    if ((int)ownershipEvidence.size() <= line)
        ownershipEvidence.resize(line + 1);

    auto &row = ownershipEvidence[line];
    if ((int)row.size() < width)
        row.assign(width, OwnershipEvidence{});

    // Carrier metadata lives in carrierGrammar; consumers read it there directly.
    // Seed only the line-level plausibility prior so later ownership stages start
    // from one canonical carrier verdict instead of privately reconstructing one.
    const CombCarrierGrammar *grammar = carrierGrammarLine(line);
    const double carrierPrior = carrierPlausibility(grammar);

    for (int rel = 0; rel < width; ++rel) {
        row[rel].facts = OwnershipFacts{};
        row[rel].assessment = OwnershipAssessment{};
        row[rel].assessment.carrierPrior = carrierPrior;
    }
}

void Comb::FrameBuffer::finalizeOwnershipClaims(OwnershipEvidence &e,
                                                double neighborLumaMeanIRE,
                                                double neighborBaseMeanIRE,
                                                double lineForwardErrorIRE) const
{
    const auto &T = configuration.tunables;
    OwnershipRules rules = lddecode::kDefaultOwnershipRules;
    rules.conflictSuppress = T.VET_OWNERSHIP_CONFLICT_SUPPRESS;
    const OwnershipFacts &f = e.facts;
    OwnershipAssessment &a = e.assessment;

    const double crestIRE = f.bandpassFineIRE;
    const double baseIRE = std::max(f.bandpassMidIRE, f.bandpassCoarseIRE);
    const double maxChromaIRE = lddecode::strongestCombChromaIRE(f);

    a.lumaRisk = std::max(
        std::clamp(f.lumaIncursionRiskIRE / 8.0, 0.0, 1.0),
        std::clamp(f.icebergAlienYFraction, 0.0, 1.0));
    a.checkerboardRisk = std::clamp(f.quarterCheckerboardRisk, 0.0, 1.0);
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
    a.agreement = 1.0 - std::clamp(f.frameFieldAgreementIRE / 6.0, 0.0, 1.0);
    a.spreadPenalty = std::clamp(f.candidateSpreadIRE / 10.0, 0.0, 1.0);
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

    lddecode::applyOwnershipConflictSuppression(
        a,
        rules);

    a.chromaClaim *= std::max(
        0.0,
        1.0 - (T.VET_OWNERSHIP_CHROMA_WEIGHT *
               std::max(0.0, a.lumaShapeContinuation - 0.25)));
    a.chromaClaim *= std::max(0.0, 1.0 - (0.5 * a.lumaClaim));
    lddecode::normalizeCombOwnershipAssessment(a, rules);
}

// 2D comb scorer (4-member Field-vs-Frame election)
// Paradigm: sample @ ±1 (interfield) = Frame. Sample @ ±2 (intra) = Field.
//
// Candidates:
//   idx 0 : Field A — same-line ±2 influenced by ±4 (intra-field, primary phase)
//   idx 1 : Field B — same-line ±2 (intra-field, alt phase)
//   idx 2 : Frame A — scalar interfield Frame (1D-conditioned same-phase
//                     blend of framePreclean; cancels 1D chroma variations)
//   idx 3 : Frame B — Field B-precleaned IQ interfield Frame
//
// Diverging Frame preferences are scored independently:
//   Frame A is favored under a vertical contrast regime and where same-phase
//   1D conditioning has materially corrected a chroma deviation.
//   Frame B is favored under a strong horizontal luma edge.
// Otherwise the existing election rules settle the contest (model-aware
// scoring, FrameIQ coherence, neighbor estimate, sharpness reward, etc.).
//
// Model: Field — when cadenceId is not (>= 0 || -3), i.e. true interlace.
// Model: Frame — progressive (>= 0) or 29.97p (-3); Frame is favored and
//                fields are scored by deviation from the frame model.
void Comb::FrameBuffer::scoreFieldVsFrame(
    int line,
    const CombTapLine &tapLine,
    const double *fieldA,
    const double *fieldB,
    const double *fieldAGate,
    const std::vector<double> &framePreclean,
    const std::vector<double> *frameRaw,
    double *outMixed,
    bool writeWeights,
    const double *lateral1D,
    const std::vector<std::complex<double>> *frameIQ)
{
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;
    if (width <= 0) return;
    if (!fieldA || !fieldB || (int)framePreclean.size() < width || !outMixed) return;
    if (line >= 0 && line < (int)fvfMetrics.size() &&
        (int)fvfMetrics[line].size() < width)
    {
        fvfMetrics[line].assign(width, FvfModelMetrics());
    }

    const auto &T   = configuration.tunables;
    const double invI = this->invIreScale;
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;

    // Minimum run length (in pixels) for field commitment when suppressing interfield teeth
    const int  FIELD_BLOCK_SIZE = 4;

    // Vertical luma contrast threshold above which we consider the field environment "active" (IRE)
    const double VERT_THRESH_IRE    = T.FIELD_VERT_DISAGREE_THRESH_IRE;

    // Horizontal luma edge threshold above which we treat the pixel as a luma transition (IRE)
    const double HEDGE_THRESH_IRE   = T.FIELD_LUMA_EDGE_THRESH_IRE;

    // Maximum distance in luma IRE frame may deviate from the active model before being considered unreliable
    const double FRAME_MAX_DIST_IRE = 4.0;

    // Maximum interfield luma divergence (IRE) below which Frame combing is permitted.
    // Above this threshold the two NTSC fields differ in time; Frame is suppressed.
    const double FIELD_DIVERGE_IRE  = 6.0;

    // Minimum luma difference between Field A and Field B combs to apply A/B divergence penalty (IRE)
    const double FIELD_DISAGREE_IRE = 6.0;

    // Below this FVF candidate difference, candidates are close enough that frame is preferred (IRE)
    const double FVF_SMALL_DIFF_IRE = (T.FVF_SMALL_DIFF_IRE > 0.0) ? T.FVF_SMALL_DIFF_IRE : 3.0;
    const int srcBufIndex = configuration.phaseCompensation ? 1 : 0;

    auto clampLine = [&](int ln)->int {
        if (ln < firstLine) return firstLine;
        if (ln >= lastLine) return lastLine - 1;
        return ln;
    };
    const double *srcLineM2 = clpbuffer[srcBufIndex].pixel[clampLine(line - 2)] + left;
    const double *srcLineM1 = clpbuffer[srcBufIndex].pixel[clampLine(line - 1)] + left;
    const double *srcLine0  = clpbuffer[srcBufIndex].pixel[clampLine(line)] + left;
    const double *srcLineP1 = clpbuffer[srcBufIndex].pixel[clampLine(line + 1)] + left;
    const double *srcLineP2 = clpbuffer[srcBufIndex].pixel[clampLine(line + 2)] + left;

    auto sample1D = [&](int rel)->double {
        if (lateral1D) {
            int r = std::clamp(rel, 0, width - 1);
            return lateral1D[r];
        } else {
            return srcLine0[std::clamp(rel, 0, width - 1)];
        }
    };
    auto getNotchLuma = [&](const double* arr, int rel) -> double {
        if (rel < 2) return arr[rel];
        if (rel >= width - 2) return arr[rel];
        double c = arr[rel], l = arr[rel - 2], r = arr[rel + 2];
        return 0.25 * (l + 2.0 * c + r);
    };
    auto getNotchLumaVec = [&](const std::vector<double>& vec, int rel) -> double {
        if (rel < 2) return vec[rel];
        if (rel >= width - 2) return vec[rel];
        double c = vec[rel], l = vec[rel - 2], r = vec[rel + 2];
        return 0.25 * (l + 2.0 * c + r);
    };
    auto notchScalar = [&](const double *srcLine, int r) -> double {
        r = std::clamp(r, 0, width - 1);
        const int rm2 = std::max(0, r - 2);
        const int rp2 = std::min(width - 1, r + 2);
        double c  = srcLine[r];
        double l  = srcLine[rm2];
        double rv = srcLine[rp2];
        return 0.25 * (l + 2.0 * c + rv);
    };
    auto vertContrastIRE = [&](int rel)->double {
        int upLine = line - 2, dnLine = line + 2;
        if (upLine < firstLine || dnLine >= lastLine) return 0.0;
        rel = std::clamp(rel, 0, width - 1);
        double up = srcLineM2[rel];
        double dn = srcLineP2[rel];
        return std::fabs(up - dn) * invI;
    };
    auto horizEdgeIRE = [&](int rel)->double {
        if (rel >= 0 && rel < (int)tapLine.hLumaDeltaIRE.size())
            return tapLine.hLumaDeltaIRE[rel];
        return 0.0;
    };
    if ((int)scratch_fvf_winner.size() != width) {
        scratch_fvf_winner.assign(width, 1);
        scratch_fvf_winner2.assign(width, 1);
        scratch_fvf_outVal.assign(width, 0.0);
        scratch_fvf_outShade.assign(width, 0.35f);
        scratch_fvf_diffFVF.assign(width, 0.0);
        scratch_fvf_satMap.assign(width, 0.0);
    }

    std::vector<int>    &winner   = scratch_fvf_winner;
    std::vector<double> &outVal   = scratch_fvf_outVal;
    std::vector<float>  &outShade = scratch_fvf_outShade;
    std::vector<double> &diffFVF  = scratch_fvf_diffFVF;
    std::vector<double> &satMap   = scratch_fvf_satMap;
    int fieldCountTotal = 0, frameCountTotal = 0;

    const double SAT_FALLBACK_START = 6.0;
    const double SAT_FALLBACK_FULL  = 20.0;
    double prev_interfield_luma_ire = 0.0;
    double prev_sat_t = 0.0;
    const double *frameRawData =
        (frameRaw && (int)frameRaw->size() >= width) ? frameRaw->data() : nullptr;

    // Core Logic of Field Vs Frame
    // when the footage is progressive we prefer interfield comb
    bool useFrameModel = (cadenceId >= 0 || cadenceId == -3);
    bool localUseFrameModel = useFrameModel;

    FvfModelMetrics *metricRow =
        (line >= 0 && line < (int)fvfMetrics.size() &&
         (int)fvfMetrics[line].size() >= width)
        ? fvfMetrics[line].data()
        : nullptr;

    const OwnershipEvidence *fvfOwnRow =
        (line >= 0 && line < (int)ownershipEvidence.size() &&
         (int)ownershipEvidence[line].size() >= width)
        ? ownershipEvidence[line].data()
        : nullptr;

    // IQ magnitude pre-pass: compute std::hypot() once per pixel (width calls)
    // rather than 7× per sample inside the hot loop.  A second sweep over the
    // flat magnitude array derives the line-level mean IQ coherence, which gates
    // the per-sample iqCoherence diagnostic so that a single noisy pixel cannot
    // claim coarseness on an otherwise smooth line.
    if ((int)scratch_fvf_iqMag.size() != width)
        scratch_fvf_iqMag.assign(width, 0.0);
    double lineMeanIqCoherence = 1.0;   // assume coherent when frameIQ absent
    if (frameIQ && (int)frameIQ->size() >= width) {
        // Pass 1: magnitudes (one hypot per pixel)
        for (int r = 0; r < width; ++r) {
            const auto &z = (*frameIQ)[r];
            scratch_fvf_iqMag[r] = std::hypot(z.real(), z.imag());
        }
        // Pass 2: line-level mean IQ coherence (arithmetic only, no hypot)
        double sumCoh = 0.0;
        for (int r = 0; r < width; ++r) {
            const double m0  = scratch_fvf_iqMag[r];
            const double m_1 = scratch_fvf_iqMag[std::max(0, r - 1)];
            const double mp1 = scratch_fvf_iqMag[std::min(width - 1, r + 1)];
            const double m_2 = scratch_fvf_iqMag[std::max(0, r - 2)];
            const double mp2 = scratch_fvf_iqMag[std::min(width - 1, r + 2)];
            const double m_4 = scratch_fvf_iqMag[std::max(0, r - 4)];
            const double mp4 = scratch_fvf_iqMag[std::min(width - 1, r + 4)];
            const double fine   = std::fabs(m0 - 0.5 * (m_1 + mp1));
            const double mid    = std::fabs(m0 - 0.5 * (m_2 + mp2));
            const double coarse = std::fabs(m0 - 0.5 * (m_4 + mp4));
            const double denom  = fine + mid + coarse + 1e-9;
            sumCoh += 1.0 - coarse / denom;
        }
        lineMeanIqCoherence = (width > 0)
            ? std::clamp(sumCoh / static_cast<double>(width), 0.0, 1.0)
            : 1.0;
    }

    for (int rel = 0; rel < width; ++rel) {
        const int rm1 = std::max(0, rel - 1);
        const int rp1 = std::min(width - 1, rel + 1);
        const int rm2 = std::max(0, rel - 2);
        const int rp2 = std::min(width - 1, rel + 2);
        double FA = fieldA[rel];
        double FB = fieldB[rel];
        double FR = framePreclean[rel];
        double FR_raw = (frameRaw && rel < (int)frameRaw->size()) ? (*frameRaw)[rel] : FR;
        double L1 = sample1D(rel);

        double satFR_demod = 0.0;
        if (frameIQ && rel < (int)frameIQ->size()) {
            std::complex<double> z = (*frameIQ)[rel];
            satFR_demod = std::abs(z);
        } else {
            satFR_demod = std::fabs(FR);
        }

        const double FA_s = FA;
        const double FB_s = FB;
        const double FR_s = FR;
        // Luma proxies: prefer the pure even-offset notch (±2 average), which is
        // less sensitive to single-pixel spikes than a [1,2,1] that includes center.
        double lumFA = getNotchLumaEven2(fieldA, rel, width);
        double lumFB = getNotchLumaEven2(fieldB, rel, width);
        double lumFR = getNotchLumaEven2Vec(framePreclean, rel);
        double lumFRRaw = frameRaw ? getNotchLumaEven2Vec(*frameRaw, rel) : lumFR;

        int maskVal = 0;
        if (line >= firstLine && line < lastLine &&
            line < (int)vdisMask.size() &&
            rel  < (int)vdisMask[line].size())
        {
            maskVal = vdisMask[line][rel];
        }
        bool vdisHard = (maskVal == 2);
        bool vdisSoft = (maskVal == 1);

        double C0   = srcLine0[rel];
        double Cpm1 = srcLineM1[rel];
        double Cpp1 = srcLineP1[rel];
        double Cpm2 = srcLineM2[rel];
        double Cpp2 = srcLineP2[rel];

        double frameLikeStack = 0.5 * (Cpm1 + Cpp1);
        double fieldLikeStack = 0.5 * (Cpm2 + Cpp2);
        double diff_stack_ire = std::fabs(frameLikeStack - fieldLikeStack) * invI;

        double diff_candA_ire = std::fabs(lumFR - lumFA) * invI;
        double diff_candB_ire = std::fabs(lumFR - lumFB) * invI;
        double diff_cand_ire  = std::min(diff_candA_ire, diff_candB_ire);
        double frameModelDistIRE = localUseFrameModel ? diff_cand_ire : diff_candA_ire;
        bool frameInsane = (frameModelDistIRE > FRAME_MAX_DIST_IRE);
    
        double interfield_luma_ire = std::fabs(
            0.5 * (notchScalar(srcLineM1, rel) + notchScalar(srcLineP1, rel))
            - notchScalar(srcLine0, rel)) * invI;
        
        double smoothed_interfield = (rel > 0)
            ? 0.5 * (interfield_luma_ire + prev_interfield_luma_ire)
            : interfield_luma_ire;
        prev_interfield_luma_ire = interfield_luma_ire;
    
            // --- Veto Logic ---
            const bool frameVerticalAllowed = carrierFrameVerticalAllowed(line);
            bool managementVeto = (cadenceId == -2) || !frameVerticalAllowed;
            bool b2VertCoherent = (smoothed_interfield < FIELD_DIVERGE_IRE) && !frameInsane;
            double targetModel = localUseFrameModel ? FR_s : FA_s;

            // diffFVF uses the geometric interfield stack divergence only
            double diff_fvf_ire = diff_stack_ire;
            diffFVF[rel] = diff_fvf_ire;
            
            if (managementVeto) {
                b2VertCoherent = false; 
            }

        double chromaMagIRE = (frameIQ)
            ? (satFR_demod * invI)
            : std::max({ std::fabs(FA), std::fabs(FB), std::fabs(FR) }) * invI;
        satMap[rel] = chromaMagIRE;
        // Saturation ramp used for soft biasing (avoid hard switches).
        double sat_t = 0.0;
        if (SAT_FALLBACK_FULL > SAT_FALLBACK_START) {
            sat_t = std::clamp((chromaMagIRE - SAT_FALLBACK_START) /
                                   (SAT_FALLBACK_FULL - SAT_FALLBACK_START),
                               0.0, 1.0);
        } else {
            sat_t = (chromaMagIRE > SAT_FALLBACK_START) ? 1.0 : 0.0;
        }
        // Light smoothing to avoid per-pixel toggling in highly saturated regions.
        sat_t = (rel > 0) ? (0.5 * (sat_t + prev_sat_t)) : sat_t;
        prev_sat_t = sat_t;

        double vIRE = vertContrastIRE(rel);
        double hIRE = horizEdgeIRE(rel);

        const double TRI_SAFE_IRE = 3.0;
        bool safeA = fvf_is_tri_safe(FA_s, L1, invI, TRI_SAFE_IRE);
        bool safeB = fvf_is_tri_safe(FB_s, L1, invI, TRI_SAFE_IRE);
        bool safeR = fvf_is_tri_safe(FR_s, L1, invI, TRI_SAFE_IRE);

        FvfModelMetrics *metrics = metricRow ? &metricRow[rel] : nullptr;
        double iqCoherence = 0.0;
        if (metrics) {
            metrics->chromaMagIRE = chromaMagIRE;
            metrics->chromaBandEnergyIRE = chromaMagIRE;
            metrics->verticalBoundaryIRE = hIRE;
            metrics->horizontalBoundaryIRE = vIRE;
            metrics->fieldFrameDivergenceIRE = diff_fvf_ire;
            metrics->interfieldDistinctIRE = smoothed_interfield;
            metrics->frameToFieldModelIRE = diff_candA_ire;
            metrics->frameToBestFieldIRE = diff_cand_ire;
            metrics->frameModel = localUseFrameModel;
            metrics->managementVeto = managementVeto;
            metrics->frameVertCoherent = b2VertCoherent;
            metrics->vdisSoft = vdisSoft;
            metrics->vdisHard = vdisHard;
        }

        int    idx   = 1;
        double val   = FB;
        float  shade = 0.35f;
        double scoreA = std::numeric_limits<double>::quiet_NaN();
        double scoreB = std::numeric_limits<double>::quiet_NaN();
        double scoreR_A = std::numeric_limits<double>::quiet_NaN();
        double scoreR_B = std::numeric_limits<double>::quiet_NaN();

        if (vdisHard) {
            // Hard regime: keep original "closest to L1" winner logic (no hysteresis here).
            double bestVal = L1;
            int    bestIdx = 1;
            float  bestSh  = 0.0f;

            auto consider = [&](int candIdx, double candVal,
                                bool safeCand, float shadeCand)
            {
                if (!safeCand) return;
                double curDiff = std::fabs(bestVal - L1) * invI;
                double newDiff = std::fabs(candVal - L1) * invI;
                if (newDiff < curDiff) {
                    bestVal = candVal;
                    bestIdx = candIdx;
                    bestSh  = shadeCand;
                }
            };

            consider(0, FA,         safeA, 0.25f);
            consider(1, FB,         safeB, 0.35f);
            consider(2, FR_s,       safeR, 0.7f);   // Frame A (precleaned)
            consider(3, FR_raw,     safeR, 0.85f);  // Frame B (Field B-precleaned IQ)

            idx   = bestIdx;
            val   = bestVal;
            shade = bestSh;
        }
        else {
            double devA = 0.0, devB = 0.0, devR_A = 0.0, devR_B = 0.0;

            if (T.FVF_SHAPE_STRENGTH > 0.0) {
                double m_c = targetModel;
                auto getM = [&](int r) {
                    r = std::clamp(r, 0, width - 1);
                    if (localUseFrameModel)
                        return framePreclean[r];
                    else
                        return fieldA[r];
                };
                double m_l = getM(rel - 1);
                double m_r = getM(rel + 1);
                double shapeModel = m_c - 0.5 * (m_l + m_r);

                auto getShapeScore = [&](double v, double v_l, double v_r) {
                    double shapeVal = v - 0.5 * (v_l + v_r);
                    return std::fabs(shapeVal - shapeModel);
                };

                double FA_l = fieldA[rm1];
                double FA_r = fieldA[rp1];
                double FB_l = fieldB[rm1];
                double FB_r = fieldB[rp1];
                double FR_l = framePreclean[rm1];
                double FR_r = framePreclean[rp1];
                const std::vector<double> &rawFrameVec = frameRaw ? *frameRaw : framePreclean;
                double FRB_l = rawFrameVec[rm1];
                double FRB_r = rawFrameVec[rp1];

                devA += getShapeScore(FA_s, FA_l, FA_r) * T.FVF_SHAPE_STRENGTH;
                devB += getShapeScore(FB_s, FB_l, FB_r) * T.FVF_SHAPE_STRENGTH;
                devR_A += getShapeScore(FR_s, FR_l, FR_r) * T.FVF_SHAPE_STRENGTH;
                devR_B += getShapeScore(FR_raw, FRB_l, FRB_r) * T.FVF_SHAPE_STRENGTH;
            }

            double satScale = std::clamp((chromaMagIRE - 2.0) / 8.0, 0.0, 1.0);

            double errA_notch = std::fabs(lumFA);
            double errB_notch = std::fabs(lumFB);
            double errR_notch = std::fabs(lumFR);
            double errRRaw_notch = std::fabs(lumFRRaw);

            scoreA = (1.0 - satScale) * devA + satScale * errA_notch;
            scoreB = (1.0 - satScale) * devB + satScale * errB_notch;
            scoreR_A = (1.0 - satScale) * devR_A + satScale * errR_notch;
            scoreR_B = (1.0 - satScale) * devR_B + satScale * errRRaw_notch;

            if (fvfOwnRow) {
                const double lc = std::clamp(fvfOwnRow[rel].assessment.lumaClaim,   0.0, 1.0);
                const double cc = std::clamp(fvfOwnRow[rel].assessment.chromaClaim, 0.0, 1.0);
                scoreR_A += T.FVF_OWNERSHIP_LUMA_WEIGHT   * lc;
                scoreR_B += T.FVF_OWNERSHIP_LUMA_WEIGHT   * lc;
                scoreA   += T.FVF_OWNERSHIP_CHROMA_WEIGHT * cc;
                scoreB   += T.FVF_OWNERSHIP_CHROMA_WEIGHT * cc;
            }

            // ------------------------------------------------------------
            // Field A confidence: if A reports low confidence, make A pay
            // its own cost and let the election decide among the remaining
            // ------------------------------------------------------------
            double gA = 1.0;
            if (fieldAGate) {
                gA = fieldAGate[std::clamp(rel, 0, width - 1)];
                gA = std::clamp(gA, 0.0, 1.0);
            }

            double gAm = gA, gAp = gA;
            if (fieldAGate) {
                gAm = std::clamp(fieldAGate[std::clamp(rel - 1, 0, width - 1)], 0.0, 1.0);
                gAp = std::clamp(fieldAGate[std::clamp(rel + 1, 0, width - 1)], 0.0, 1.0);
            }

            const double gateAltA = std::fabs(gA - 0.5 * (gAm + gAp));

            const double W_A_GATE     = 0.20;
            const double W_A_GATE_ALT = 0.30;

            scoreA += W_A_GATE * (1.0 - gA);
            scoreA += W_A_GATE_ALT * gateAltA;

            // ------------------------------------------------------------
            // Model-aware regime scoring.
            // The active model has a same-domain buddy: Field B for Field A,
            // Frame B for Frame A. Cross-domain candidates pay the model
            // distance; the buddy is left to compete on its own evidence.
            // ------------------------------------------------------------
            if (localUseFrameModel) {
                scoreA += T.FVF_MODEL_PRIMARY_WEIGHT * diff_candA_ire;
                scoreB += T.FVF_MODEL_PRIMARY_WEIGHT * diff_candB_ire;
                if (!managementVeto && !frameInsane) {
                    scoreR_A *= T.FRAME_MODEL_BIAS;
                    scoreR_B *= T.FRAME_MODEL_BIAS;
                }
            } else {
                const double closeFrameBonus = std::clamp(
                    1.0 - (diff_candA_ire / std::max(1e-9, T.FVF_SMALL_DIFF_IRE)),
                    0.0, 1.0);
                scoreA *= T.FIELD_MODEL_BIAS;
                scoreR_A += T.FVF_MODEL_PRIMARY_WEIGHT * diff_candA_ire;
                scoreR_B += T.FVF_MODEL_PRIMARY_WEIGHT * diff_candA_ire;
                scoreR_A *= T.FRAME_IN_INTERLACE_PENALTY;
                scoreR_B *= T.FRAME_IN_INTERLACE_PENALTY;
                scoreR_A *= (1.0 - 0.08 * closeFrameBonus);
                scoreR_B *= (1.0 - 0.08 * closeFrameBonus);
            }

            if (frameIQ && rel < (int)frameIQ->size()) {
                // Pre-pass filled scratch_fvf_iqMag; 7 array reads replace
                // 7 hypot() calls.  rm1/rp1/rm2/rp2 are already clamped above.
                const int rm4 = std::max(0, rel - 4);
                const int rp4 = std::min(width - 1, rel + 4);
                const double m0  = scratch_fvf_iqMag[rel];
                const double m_1 = scratch_fvf_iqMag[rm1];
                const double mp1 = scratch_fvf_iqMag[rp1];
                const double m_2 = scratch_fvf_iqMag[rm2];
                const double mp2 = scratch_fvf_iqMag[rp2];
                const double m_4 = scratch_fvf_iqMag[rm4];
                const double mp4 = scratch_fvf_iqMag[rp4];

                const double fine   = std::fabs(m0 - 0.5 * (m_1 + mp1));
                const double mid    = std::fabs(m0 - 0.5 * (m_2 + mp2));
                const double coarse = std::fabs(m0 - 0.5 * (m_4 + mp4));

                const double denom = fine + mid + coarse + 1e-9;
                const double fineFrac   = fine   / denom;
                const double midFrac    = mid    / denom;
                const double coarseFrac = coarse / denom;
                // Gate iqCoherence against the line-level mean: an isolated
                // noisy pixel cannot claim coarseness on a smooth line.
                iqCoherence = std::clamp(
                    (1.0 - coarseFrac) * (0.3 + 0.7 * lineMeanIqCoherence),
                    0.0, 1.0);
                if (metrics) {
                    metrics->iqFineFrac = fineFrac;
                    metrics->iqMidFrac = midFrac;
                    metrics->iqCoarseFrac = coarseFrac;
                    metrics->iqCoherence = iqCoherence;
                }

                scoreR_A *= (1.0 - T.FVF_SCALE_FINE_FRAME_A_BONUS * fineFrac);
                scoreR_B *= (1.0 - T.FVF_SCALE_FINE_FRAME_B_BONUS * fineFrac);

                scoreB   *= (1.0 - T.FVF_SCALE_MID_FIELD_B_BONUS * midFrac);
                scoreA   *= (1.0 - T.FVF_SCALE_MID_FIELD_A_BONUS * midFrac);
                scoreR_A *= (1.0 - T.FVF_SCALE_MID_FRAME_A_BONUS * midFrac);

                scoreA   *= (1.0 - T.FVF_SCALE_COARSE_FIELD_A_BONUS * coarseFrac);

                const bool dual4Accepted =
                    tapLine.haveU4 && tapLine.haveD4 &&
                    rel < (int)tapLine.contour.size() &&
                    tapLine.contour[rel].upSideOk > 0.5 &&
                    tapLine.contour[rel].dnSideOk > 0.5;
                if (dual4Accepted) {
                    scoreA *= (1.0 - T.FVF_SCALE_COARSE_DUAL4_FIELD_A_BONUS * coarseFrac);
                }
            }


            // ------------------------------------------------------------
            // Saturation regime: in highly saturated regions, Frame is often
            // the least visually toxic when coherent, but Field B tends to
            // introduce zipper/alternation more readily than Field A.
            // Apply a soft bias rather than a hard override.
            // ------------------------------------------------------------
            if (sat_t > 0.15) {
                scoreB *= (1.0 + T.FVF_SAT_FIELD_B_PEN * sat_t);

                const bool satFrameOk = !managementVeto &&
                    (localUseFrameModel ? !frameInsane : b2VertCoherent);
                if (satFrameOk) {
                    scoreR_B *= (1.0 - T.FVF_SAT_FRAME_B_BONUS * sat_t);
                }
            }

            // ------------------------------------------------------------
            // Transition sharpness reward:
            // Detect stable region transitions along the scanline and reward
            // candidates that make a fast (sharp) step and settle on both sides.
            // This acts as a proxy for "sharpness" without applying a filter.
            // ------------------------------------------------------------
            {
                constexpr int EDGE_GAP = 2;   // pixels excluded around the transition
                // Use same-phase notch probes on the source line to detect a stable step.
                // This reuses our existing notch architecture and avoids per-pixel window scans.
                constexpr int EDGE_PROBE_NEAR = 2;
                constexpr int EDGE_PROBE_FAR  = 8;
                const bool canEval =
                    (hIRE >= 0.75 * HEDGE_THRESH_IRE) &&
                    (rel >= (EDGE_GAP + EDGE_PROBE_FAR)) &&
                    (rel + (EDGE_GAP + EDGE_PROBE_FAR) < width) &&
                    (line >= firstLine && line < lastLine);

                if (canEval) {
                    auto srcNotch = [&](int r)->double {
                        r = std::clamp(r, 0, width - 1);
                        return getNotchLumaEven2(srcLine0, r, width);
                    };

                    const double lNear = srcNotch(rel - (EDGE_GAP + EDGE_PROBE_NEAR));
                    const double lFar  = srcNotch(rel - (EDGE_GAP + EDGE_PROBE_FAR));
                    const double rNear = srcNotch(rel + (EDGE_GAP + EDGE_PROBE_NEAR));
                    const double rFar  = srcNotch(rel + (EDGE_GAP + EDGE_PROBE_FAR));

                    const double stepIRE = std::fabs(rNear - lNear) * invI;
                    const double lJitterIRE = std::fabs(lNear - lFar) * invI;
                    const double rJitterIRE = std::fabs(rNear - rFar) * invI;

                    // Require a meaningful step with stable plateaus (discount small fluctuations).
                    const double EDGE_STEP_THRESH_IRE = std::max(2.0, 0.9 * HEDGE_THRESH_IRE);
                    const double EDGE_PLATEAU_JITTER_MAX_IRE = 1.2;
                    const bool stableStep =
                        (stepIRE >= EDGE_STEP_THRESH_IRE) &&
                        (lJitterIRE <= EDGE_PLATEAU_JITTER_MAX_IRE) &&
                        (rJitterIRE <= EDGE_PLATEAU_JITTER_MAX_IRE);

                    if (stableStep) {
                        // Candidate step measured using notch luma (reduces composite phase chatter).
                        const double lmeanIRE = lNear * invI;
                        const double rmeanIRE = rNear * invI;

                        auto applySharpReward = [&](double &score,
                                                    const double *arr,
                                                    const std::vector<double> *vec)
                        {
                            const double m2 = arr ? getNotchLuma(arr, rm2) : getNotchLumaVec(*vec, rm2);
                            const double p2 = arr ? getNotchLuma(arr, rp2) : getNotchLumaVec(*vec, rp2);
                            const double candStepIRE = std::fabs(p2 - m2) * invI;

                            // Reward only if candidate has plausibly settled to the two plateaus.
                            const double settleL = std::fabs(m2 * invI - lmeanIRE);
                            const double settleR = std::fabs(p2 * invI - rmeanIRE);
                            const double SETTLE_MAX_IRE = 0.35 * stepIRE + 1.0;
                            if (settleL > SETTLE_MAX_IRE || settleR > SETTLE_MAX_IRE) return;

                            // Normalize: prefer candidates that reach most of the step quickly.
                            const double ratio = candStepIRE / std::max(1e-9, stepIRE);
                            const double sharp = std::clamp((ratio - 0.70) / 0.30, 0.0, 1.0);
                            const double stepStrength = std::clamp((stepIRE - EDGE_STEP_THRESH_IRE) / 6.0, 0.0, 1.0);
                            score *= (1.3 - T.FVF_TRANSITION_SHARPNESS_WEIGHT * sharp * stepStrength);
                        };

                        applySharpReward(scoreA, fieldA, nullptr);
                        applySharpReward(scoreB, fieldB, nullptr);
                        applySharpReward(scoreR_A, nullptr, &framePreclean);
                        applySharpReward(scoreR_B, nullptr, frameRaw ? frameRaw : &framePreclean);
                    }
                }
            }

            // VDIS
            if (!vdisHard &&
                hIRE < T.NEIGHBOR_EST_EDGE_MAX_IRE &&
                diff_stack_ire < T.NEIGHBOR_EST_FVF_MAX_IRE &&
                chromaMagIRE < T.NEIGHBOR_EST_SAT_MAX_IRE)
            {
                double estA2 = 0.5 * (fieldA[rm2]  + fieldA[rp2]);
                double estB2 = 0.5 * (fieldB[rm2]  + fieldB[rp2]);
                double estF2 = 0.5 * (framePreclean[rm2] + framePreclean[rp2]);
                double estRB2 = frameRawData
                    ? 0.5 * (frameRawData[rm2] + frameRawData[rp2])
                    : estF2;

                double E2 = median4_average_middle(estA2, estB2, estF2, estRB2);

                bool allowPm1 = true;
                {
                    double altF = std::fabs(framePreclean[rm1] - framePreclean[rp1]) * invI;
                    if (altF > 6.0) allowPm1 = false;
                }

                double E = E2;
                if (allowPm1) {
                    double estA1 = 0.5 * (fieldA[rm1]  + fieldA[rp1]);
                    double estB1 = 0.5 * (fieldB[rm1]  + fieldB[rp1]);
                    double estF1 = 0.5 * (framePreclean[rm1] + framePreclean[rp1]);
                    double estRB1 = frameRawData
                        ? 0.5 * (frameRawData[rm1] + frameRawData[rp1])
                        : estF1;

                    double E1 = median4_average_middle(estA1, estB1, estF1, estRB1);

                    const double K_PM1 = 0.5;
                    E = (E2 + K_PM1 * E1) / (1.0 + K_PM1);
                }

                double dA = std::fabs(FA_s - E) * invI;
                double dB = std::fabs(FB_s - E) * invI;
                double dR_A = std::fabs(FR_s - E) * invI;
                double dR_B = std::fabs(FR_raw - E) * invI;

                const double W_NEIGH = T.NEIGHBOR_EST_WEIGHT;
                scoreA += W_NEIGH * dA;
                scoreB += W_NEIGH * dB;
                scoreR_A += W_NEIGH * dR_A;
                scoreR_B += W_NEIGH * dR_B;
            }

            // When Field/Frame are already close, Frame receives a score reward
            // instead of overriding the election after a winner has been chosen.
            if (diff_fvf_ire < FVF_SMALL_DIFF_IRE && (!vdisSoft || safeR)) {
                const double close = 1.0 - std::clamp(diff_fvf_ire / std::max(1e-9, FVF_SMALL_DIFF_IRE), 0.0, 1.0);
                scoreR_A *= (1.0 - 0.12 * close);
                scoreR_B *= (1.0 - 0.12 * close);
            }

            // ------------------------------------------------------------
            // 4-member election: derive Frame A / Frame B preferences.
            // Each Frame has its own score; native-regime nudges only refine
            // the split between the two same-domain buddies.
            // ------------------------------------------------------------
            // Vertical contrast: when ±2 lines differ in luma, Field A's ±2
            // comb is unreliable (luma-chroma crosstalk).  Penalize it and
            // reward Frame A which doesn't use vertical neighbors.
            {
                const double VERT_NORM       = std::max(VERT_THRESH_IRE, 1.0);
                const double frameA_vert_t   = std::clamp(vIRE / VERT_NORM, 0.0, 1.0);
                scoreA   *= (1.0 + T.FVF_VERT_FIELD_A_PENALTY * frameA_vert_t);
                scoreR_A *= (1.0 - T.FVF_VERT_FRAME_A_BONUS * frameA_vert_t);
            }

            // Strong horizontal edges favor Frame B and de-emphasize Field B.
            {
                const double HEDGE_NORM      = std::max(HEDGE_THRESH_IRE, 1.0);
                const double frameB_hedge_t  = std::clamp(hIRE / HEDGE_NORM, 0.0, 1.0);
                scoreB   *= (1.0 + T.FVF_HEDGE_FIELD_B_PENALTY * frameB_hedge_t);
                scoreR_B *= (1.0 - T.FVF_HEDGE_FRAME_B_BONUS * frameB_hedge_t);
            }

            auto pickCandidate = [&](int candIdx, double candVal, float candShade) {
                if (vdisSoft) {
                    if (candIdx == 0 && !safeA) return;
                    if (candIdx == 1 && !safeB) return;
                    if ((candIdx == 2 || candIdx == 3) && !safeR) return;
                }
                idx   = candIdx;
                val   = candVal;
                shade = candShade;
            };

            // Pick whichever Frame variant has the lower (better) score.
            auto pickBestFrame = [&](float shadeA, float shadeB) {
                if (scoreR_A <= scoreR_B) pickCandidate(2, FR_s,   shadeA);
                else                       pickCandidate(3, FR_raw,     shadeB);
            };

            if (hIRE > HEDGE_THRESH_IRE && diff_stack_ire > 5.0) {
                // Strong horizontal luma edge: score-based 4-way election.
                // Interframe combs are more reliable here than interfield.
                if (!frameInsane && scoreR_B <= scoreR_A && scoreR_B <= scoreA &&
                           scoreR_B <= scoreB) {
                    pickCandidate(3, FR_raw, 0.85f);
                } else if (!frameInsane && scoreR_A <= scoreA && scoreR_A <= scoreB) {
                    pickCandidate(2, FR_s, 0.7f);
                } else if (scoreA < scoreB) {
                    pickCandidate(0, FA, 0.25f);
                } else {
                    pickCandidate(1, FB, 0.35f);
                }
            } else {
                const double bestFrame = std::min(scoreR_A, scoreR_B);
                const double bestField = std::min(scoreA, scoreB);

                if (localUseFrameModel && !managementVeto && !frameInsane &&
                    bestFrame <= bestField) {
                    // Progressive: Frame is the model. It wins when its
                    // score matches or beats the best field candidate;
                    // the FRAME_MODEL_BIAS already baked in an advantage.
                    pickBestFrame(0.7f, 0.85f);
                } else if (bestFrame + 1e-12 < scoreA * 0.85 &&
                           bestFrame + 1e-12 < scoreB * 0.85) {
                    // Either regime: Frame convincingly beats both fields.
                    pickBestFrame(0.7f, 0.85f);
                } else if (scoreA < scoreB * 0.8) {
                    pickCandidate(0, FA, 0.25f);
                } else {
                    double dFL = std::fabs(lumFB - L1) * invI;
                    double dRL = std::fabs(lumFRRaw - L1) * invI;
                    if (!frameInsane && dRL + 1.0 < dFL)
                        pickBestFrame(0.7f, 0.85f);
                    else
                        pickCandidate(1, FB, 0.35f);
                }
            }

        }

        winner[rel]   = idx;
        outVal[rel]   = val;
        outShade[rel] = shade;
        if      (idx == 2 || idx == 3) frameCountTotal++;
        else if (idx == 0 || idx == 1) fieldCountTotal++;
        if (metrics)
            metrics->winner = idx;
    }

    // Island cleanup
    auto applyIslandFilter = [&]() {
        std::vector<int> &w2 = scratch_fvf_winner2;
        std::copy(winner.begin(), winner.end(), w2.begin());
        const double EDGE_STOP_IRE = HEDGE_THRESH_IRE;
        const double DIFF_STOP_IRE = 6.0;
        bool changed = false;

        for (int rel = 1; rel < width - 1; ++rel) {
            if (satMap[rel] > SAT_FALLBACK_START) continue;
            if (horizEdgeIRE(rel) > EDGE_STOP_IRE) continue;
            if (diffFVF[rel] > DIFF_STOP_IRE) continue;

            int L = winner[rel - 1];
            int C = winner[rel];
            int R = winner[rel + 1];

            if (L == R && C != L) {
                w2[rel] = L;
                changed = true;
            }
        }

        if (changed) {
            winner.swap(w2);
            for (int rel = 0; rel < width; ++rel) {
                int idx = winner[rel];
                if      (idx == 0) { outVal[rel] = fieldA[rel];                outShade[rel] = 0.25f; }
                else if (idx == 1) { outVal[rel] = fieldB[rel];                outShade[rel] = 0.35f; }
                else if (idx == 2) { outVal[rel] = framePreclean[rel];         outShade[rel] = 0.7f;  }
                else               { outVal[rel] = frameRawData ? frameRawData[rel] : framePreclean[rel]; outShade[rel] = 0.85f; }
            }
        }
    };

    applyIslandFilter();

    // Island cleanup can flip local winners, so refresh regime counts before
    // using them to decide block-level field commitment.
    fieldCountTotal = 0;
    frameCountTotal = 0;
    for (int rel = 0; rel < width; ++rel) {
        const int idx = winner[rel];
        if (idx == 2 || idx == 3) frameCountTotal++;
        else if (idx == 0 || idx == 1) fieldCountTotal++;
    }

    if (!localUseFrameModel && fieldCountTotal > frameCountTotal * 2 && fieldCountTotal > 0) {
        for (int b = 0; b < width; b += FIELD_BLOCK_SIZE) {
            int e = std::min(width, b + FIELD_BLOCK_SIZE);

            double blockDivergence = 0.0;
            for (int r = b; r < e; ++r)
                blockDivergence += diffFVF[r];
            blockDivergence /= (e - b);

            // diffFVF is already in IRE units; compare directly against IRE threshold.
            if (blockDivergence > FIELD_DISAGREE_IRE) {
                int cntA = 0, cntB = 0, cntF = 0;
                for (int r = b; r < e; ++r) {
                    if      (winner[r] == 0) cntA++;
                    else if (winner[r] == 1) cntB++;
                    else                     cntF++;   // Frame A (2) or Frame B (3)
                }
                if (cntF > 0 && (cntA + cntB) > 0) {
                    int blockIdx = (cntA >= cntB) ? 0 : 1;
                    for (int r = b; r < e; ++r) {
                        winner[r] = blockIdx;
                        if (blockIdx == 0) { outVal[r] = fieldA[r]; outShade[r] = 0.25f; }
                        else               { outVal[r] = fieldB[r]; outShade[r] = 0.35f; }
                    }
                }
            }
        }
    }
    for (int rel = 0; rel < width; ++rel) {
        outMixed[rel] = outVal[rel];
        if (line >= 0 && line < (int)fvfMetrics.size() &&
            rel < (int)fvfMetrics[line].size())
        {
            fvfMetrics[line][rel].winner = winner[rel];
        }
        if (writeWeights && line < (int)w2d_frame_weight.size()) {
            float w = outShade[rel];
            if (!std::isfinite(w)) w = 0.0f;
            w2d_frame_weight[line][rel] = w;
        }
    }
}

void Comb::FrameBuffer::collectCombOwnershipEvidence(
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

    if ((int)ownershipEvidence.size() <= line)
        ownershipEvidence.resize(line + 1);

    auto &row = ownershipEvidence[line];
    if ((int)row.size() < width) {
        row.assign(width, OwnershipEvidence());
        seedCombOwnershipPerLine(line);
    }

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
        OwnershipEvidence &e = row[rel];
        OwnershipFacts &f = e.facts;

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
    // 0.0 signals "not available" and causes finalizeOwnershipClaims() to fall
    // back to its hard-coded denominators — behaviour is identical to before.
    const double lineForwardErrorIRE = (lineGrammar && lineGrammar->projectionValid)
        ? lineGrammar->meanForwardErrorIRE
        : 0.0;

    // Final ownership needs cross-path evidence plus a same-phase continuity
    // check, not just the local 1D residual snapshot.
    for (int rel = 0; rel < width; ++rel) {
        const int rm4 = std::max(0, rel - 4);
        const int rp4 = std::min(width - 1, rel + 4);
        OwnershipEvidence &e = row[rel];
        const OwnershipFacts &leftFacts = row[rm4].facts;
        const OwnershipFacts &rightFacts = row[rp4].facts;
        const double leftBaseIRE = std::max(leftFacts.bandpassMidIRE,
                                            leftFacts.bandpassCoarseIRE);
        const double rightBaseIRE = std::max(rightFacts.bandpassMidIRE,
                                             rightFacts.bandpassCoarseIRE);

        finalizeOwnershipClaims(
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
    // Avoid floor() per pixel by filling spans between block anchors directly.
    auto blockAvg = [&](int block)->double {
        const int x0 = std::clamp(block * 4, 0, std::max(0, width - 4));
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
            if (line < 0 || line >= (int)locked1DSource.size())
                return 0.0;
            const auto &row = locked1DSource[line];
            if ((int)row.size() < width)
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

        if (line < 0 || line >= (int)locked1DSource.size())
            return {0.0, 0.0};
        const auto &row = locked1DSource[line];
        if ((int)row.size() < width)
            return {0.0, 0.0};

        const double c = row[std::clamp(rel, 0, width - 1)];
        return { c * ti, c * tq };
    };

    auto lockedScalar = [&](int line, int rel)->double {
        if (line < 0 || line >= (int)locked1DSource.size())
            return 0.0;
        const auto &row = locked1DSource[line];
        if ((int)row.size() < width)
            return 0.0;
        return row[std::clamp(rel, 0, width - 1)];
    };

    const double invI = invIreScale;

    for (int line = firstLine; line < lastLine; ++line) {
        if (line >= demodLines)
            continue;

        if (useLockedSource) {
            if (line < 0 || line >= (int)locked1DSource.size())
                continue;
            if ((int)locked1DSource[line].size() < width)
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
                line - 1 < (int)locked1DSource.size() &&
                line + 1 < (int)locked1DSource.size() &&
                (int)locked1DSource[line - 1].size() >= width &&
                (int)locked1DSource[line + 1].size() >= width)
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

    qint64 ownN = 0;
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
        if (line < 0 || line >= (int)ownershipEvidence.size())
            continue;
        const auto &row = ownershipEvidence[line];
        if ((int)row.size() < width)
            continue;
        const CombCarrierGrammar *grammar = carrierGrammarLine(line);
        const double lineCarrierScale = grammar ? grammar->carrierScale : 0.0;
        const double lineCarrierConf  = grammar ? std::clamp(grammar->phaseConfidence, 0.0, 1.0) : 0.0;
        const double lineCarrierPlausibility = carrierPlausibility(grammar);
        const double lineCarrierPhase = grammar ? grammar->phaseError : 0.0;
        const double lineConflict = grammar ? grammar->phaseScheduleConflict : 0.0;
        if (lineConflict > 0.0) ++scheduleConflictLines;
        for (int rel = 0; rel < width; ++rel) {
            const OwnershipEvidence &e = row[rel];
            ++ownN;
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
    if (ownN > 0) {
        const double invOwnN = 1.0 / (double)ownN;
        msg += QString(" ownership(n=%1,luma=%2,chroma=%3,uncertain=%4,incur=%5,spread=%6,frameCoh=%7,carScale=%8,carConf=%9,carPlaus=%10,carPhaseAbsDeg=%11,schedConf=%12,schedConfLines=%13)")
            .arg(ownN)
            .arg(sumLumaClaim * invOwnN, 0, 'f', 3)
            .arg(sumChromaClaim * invOwnN, 0, 'f', 3)
            .arg(sumUncertainClaim * invOwnN, 0, 'f', 3)
            .arg(sumLumaIncursion * invOwnN, 0, 'f', 3)
            .arg(sumCandidateSpread * invOwnN, 0, 'f', 3)
            .arg(sumFrameCoherence * invOwnN, 0, 'f', 3)
            .arg(sumCarrierScale * invOwnN, 0, 'f', 3)
            .arg(sumCarrierConfidence * invOwnN, 0, 'f', 3)
            .arg(sumCarrierPlausibility * invOwnN, 0, 'f', 3)
            .arg(sumCarrierPhaseErrorAbs * invOwnN * 180.0 / M_PI, 0, 'f', 3)
            .arg(sumPhaseScheduleConflict * invOwnN, 0, 'f', 3)
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
    const bool needFrameIQStorage =
        (configuration.twoDVariant == Comb::Configuration::TwoDVariant::FramePreclean ||
         configuration.twoDVariant == Comb::Configuration::TwoDVariant::FrameRaw ||
         wantFvf);
    const bool needFrameIQCompute = configuration.phaseCompensation && needFrameIQStorage;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;

    if (width <= 0 || firstLine >= lastLine) return;

    if (configuration.phaseCompensation) {
        buildPhaseCorrected1D(); // writes locked-1D directly into locked1DSource
        reportPhaseLegStats("locked1d", 1, true);
    }

    if (configuration.twoDVariant == Comb::Configuration::TwoDVariant::Line) {
        for (int line = firstLine; line < lastLine; ++line) {
            double *dst = clpbuffer[1].pixel[line];
            if (configuration.phaseCompensation &&
                line >= 0 && line < (int)locked1DSource.size() &&
                (int)locked1DSource[line].size() >= width)
            {
                for (int rel = 0; rel < width; ++rel) dst[left + rel] = locked1DSource[line][rel];
            } else {
                const double *src1d = clpbuffer[0].pixel[line];
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
        case V::Field:
            combTapBuildFlags_ = TapBuildFieldA | TapBuildFieldB;
            break;
        case V::FieldB:
            combTapBuildFlags_ = TapBuildFieldB;
            break;
        case V::FramePreclean:
            combTapBuildFlags_ = TapBuildFrame | TapBuildFieldB;
            break;
        case V::FrameRaw:
            combTapBuildFlags_ = TapBuildFieldB;
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
    std::vector<std::complex<double>> frameIQPreclean;

    for (int line = firstLine; line < lastLine; ++line) {
        if (line >= demodLines) continue;

        ensureCombTapLine(line);
        const CombTapLine &tapLine = tapLineCache[precleanRingSlot(line)];

        if (needFrameIQCompute) {
            auto ensureFieldBPrecleanLine = [&](int ln) {
                if (ln < firstLine || ln >= lastLine) return;
                if (havePrecleanLine(ln, width)) return;
                double *preclean = precleanLinePtrMutable(ln, width);
                computeSimpleField2DLine(ensureCombTapLine(ln), preclean);
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
                std::copy(fieldBPreclean, fieldBPreclean + width, scratch_fieldBLine.begin());
            } else {
                computeSimpleField2DLine(tapLine, scratch_fieldBLine.data());
            }
        } else {
            std::fill(scratch_fieldBLine.begin(), scratch_fieldBLine.begin() + width, 0.0);
        }
        if (combTapBuildFlags_ & TapBuildFieldA) {
            computeField2DLine(tapLine, scratch_fieldLine.data(), scratch_fieldGate.data());
        } else {
            std::fill(scratch_fieldLine.begin(), scratch_fieldLine.begin() + width, 0.0);
            std::fill(scratch_fieldGate.begin(), scratch_fieldGate.begin() + width, 1.0);
        }

        {
            const double *src1d = configuration.phaseCompensation
                                  ? nullptr
                                  : clpbuffer[0].pixel[line];
            if ((int)scratch_lateralLine.size() < width)
                scratch_lateralLine.resize(width);
            if (configuration.phaseCompensation) {
                if (line >= 0 && line < (int)locked1DSource.size() &&
                    (int)locked1DSource[line].size() >= width)
                {
                    std::copy(locked1DSource[line].begin(),
                              locked1DSource[line].begin() + width,
                              scratch_lateralLine.begin());
                } else {
                    std::fill(scratch_lateralLine.begin(), scratch_lateralLine.begin() + width, 0.0);
                }
            } else {
                for (int rel = 0; rel < width; ++rel)
                    scratch_lateralLine[rel] = src1d[left + rel];
            }
        }

        if (needFrameIQCompute) {
            computeFrameIQPrecleanLine(line, frameIQPreclean, false);
            if ((int)scratch_fieldBCenter.size() < width)
                scratch_fieldBCenter.resize(width);
            for (int rel = 0; rel < width; ++rel) {
                const int h = left + rel;
                if (rel < (int)frameIQPreclean.size()) {
                    const auto &Z = frameIQPreclean[rel];
                    scratch_fieldBCenter[rel] = remod4fscToCompositePhase(Z.real(), Z.imag(), carrierSampleClass(line, h));
                } else {
                    scratch_fieldBCenter[rel] = 0.0;
                }
            }

            computeFrameBLocked1DLine(line, frameIQ, scratch_frameBCenter);
            if ((int)scratch_frameBCenter.size() < width)
                scratch_frameBCenter.resize(width);
        }

        collectCombOwnershipEvidence(
            line,
            scratch_fieldLine.data(),
            scratch_fieldBLine.data(),
            needFrameIQCompute ? scratch_fieldBCenter : scratch_frameBCenter,
            needFrameIQCompute ? &frameIQ : nullptr);

        double *dst = clpbuffer[1].pixel[line];
        auto emitSelected = [&](int rel, double v) {
            dst[left + rel] = v;
        };

        if (configuration.twoDVariant == Comb::Configuration::TwoDVariant::Field) {
            for (int rel = 0; rel < width; ++rel) emitSelected(rel, scratch_fieldLine[rel]);
            if (writeWeights) std::fill(w2d_frame_weight[line].begin(), w2d_frame_weight[line].end(), 0.0f);
        }
        else if (configuration.twoDVariant == Comb::Configuration::TwoDVariant::FieldB) {
            for (int rel = 0; rel < width; ++rel) emitSelected(rel, scratch_fieldBLine[rel]);
            if (writeWeights) std::fill(w2d_frame_weight[line].begin(), w2d_frame_weight[line].end(), 0.35f);
        }
        else if (configuration.twoDVariant == Comb::Configuration::TwoDVariant::FramePreclean && configuration.phaseCompensation) {
            for (int rel = 0; rel < width; ++rel) emitSelected(rel, scratch_fieldBCenter[rel]);
            if (writeWeights) std::fill(w2d_frame_weight[line].begin(), w2d_frame_weight[line].end(), 0.8f);
        }
        else if (configuration.twoDVariant == Comb::Configuration::TwoDVariant::FrameRaw && configuration.phaseCompensation) {
            for (int rel = 0; rel < width; ++rel) emitSelected(rel, scratch_frameBCenter[rel]);
            if (writeWeights) std::fill(w2d_frame_weight[line].begin(), w2d_frame_weight[line].end(), 0.85f);
        }
        else {
            if (!configuration.phaseCompensation) {
                for (int rel = 0; rel < width; ++rel) {
                    dst[left + rel] = scratch_fieldBLine[rel];
                    if (writeWeights && line < (int)w2d_frame_weight.size()) {
                        w2d_frame_weight[line][rel] = 0.35f; 
                    }
                }
            } else {
                scoreFieldVsFrame(
                    line,
                    tapLine,
                    scratch_fieldLine.data(),
                    scratch_fieldBLine.data(),
                    scratch_fieldGate.data(),
                    scratch_fieldBCenter,
                    &scratch_frameBCenter,
                    scratch_outMixed.data(),
                    writeWeights,
                    scratch_lateralLine.data(),
                    &frameIQ);

                const double OUT_CLAMP_MAG = std::max(32.0 * irescale, 1.0);
                    for (int rel = 0; rel < width; ++rel) {
                        double vMixed = scratch_outMixed[rel];
                    
                        // Keep only the numeric-sanity fallback:
                        if (!std::isfinite(vMixed)) vMixed = scratch_fieldBLine[rel];
                    
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
            double base1d;
            if (configuration.phaseCompensation &&
                line >= 0 && line < (int)locked1DSource.size() &&
                (int)locked1DSource[line].size() > (h0 - left))
            {
                base1d = locked1DSource[line][h0 - left];
            } else {
                base1d = clpbuffer[0].pixel[line][h0];
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
            // Agreement shaping: reward temporal candidates that agree with the 2D
            // estimate (parabolic bonus within AGREEMENT_REWARD_RADIUS_IRE), apply no
            // adjustment in the neutral zone, and heavily penalise deviations beyond
            // deviationThreshold to veto temporally incoherent candidates.
             if (dIRE <= T.AGREEMENT_REWARD_RADIUS_IRE) {
                double x = dIRE / T.AGREEMENT_REWARD_RADIUS_IRE;
                delta = - (T.AGREEMENT_REWARD_MAX * configuration.adaptThreshold) * (1.0 - x * x);
            } else if (dIRE <= T.deviationThreshold) {
                delta = 0.0;
            } else {
                delta = T.AGREEMENT_VETO_BASE + T.deviationPenalty * (dIRE - T.deviationThreshold);
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

    if ((int)scratch_filter_temp.size() < width) scratch_filter_temp.assign(width, 0.0);
    double *temp = scratch_filter_temp.data();

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
    if (demodMode == DemodMode::Locked) {
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
