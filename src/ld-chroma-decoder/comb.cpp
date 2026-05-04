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
    // Locked path: independent per-axis gain (for chroma ellipse)
    constexpr double GI_PRODUCT = 1.0;
    constexpr double GQ_PRODUCT = 0.9;

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

    // Hard-coded, bucket-preserving (h&3) smooth applied to the locked,
    // remodulated scalar 4fsc line in buildPhaseCorrected1D(). This keeps
    // same-phase identity while reducing bucket-local texture injection
    // that can otherwise show up as multi-line variation in Field combs.
    constexpr double FIELD_BUCKET_SMOOTH_STRENGTH = 0.50;
}

// Tiny global LO trim (degrees). Negative usually counteracts a slight green bias.
static constexpr double CAL_LO_ROT_DEG  = 0.0;

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
inline void demodSample(double v, int h, int xi,
                        double bcos, double bsin,
                        const double* spLUT, const double* cpLUT,
                        float* outI, float* outQ)
{
    const int idx = (h & 3);
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
            next->split1D();
            if (configuration.phaseCompensation) {
                // Heavy locked pre-processing between 1D and 2D
                next->phaseLocked();
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

        if (configuration.phaseCompensation) {
            // After 2D/3D selection has settled, downstream locked-path IQ should
            // follow the final selected comb buffer rather than the earlier 1D demod.
            current->rebuildLockedDemodFromSelectedComb();
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

            // 1) Sinusoidal-fit pre-clean and affine solve (fills clpbuffer[0], lineAffineLocked)
            // 2) Phase-corrected 1D demod for 2D work (via buildPhaseCorrected1D inside split2D)
            // 3) Full 2D/3D selection -> clpbuffer[dimensions-1]
            // 4) Re-demod final selected comb into demodTI_flat/TQ_flat
            // 5) Demod raw composite -> TRI/TRQ; build preI/preQ and yI/yQ for residual Y
            current->splitIQlocked();
            // 6) Chroma NR on I/Q
            current->doCNR();
            // 7) Coherent Y rebuild from affine-corrected demod
            current->produceY();
            // 8) Chroma FIR bandwidth limiting in locked space
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
        demodBurstCos.assign(demodLines, 1.0f);
        demodBurstSin.assign(demodLines, 0.0f);
        demodTI_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
        demodTQ_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
        lineAffineLocked.assign(lines, LineAffine{{{1,0},{0,1}}, false});
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
// metadata. Also initialises lineFlip (per-line subcarrier polarity) and clears
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

    // --- Initialize per-line flip from existing getLinePhase() ---
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    lineFlip.assign(last, +1);
    for (int line = first; line < last; ++line) {
        lineFlip[line] = getLinePhase(line) ? -1 : +1;
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
}



// Burst detection (internal linkage).
// Measures the colour burst in the horizontal blanking interval to derive
// a normalised phasor (bcos, bsin) representing the subcarrier reference
// phase for this line. The optional floor clamp prevents burst collapse on
// very noisy lines from producing a near-zero (and hence useless) phasor.
namespace {
    struct BurstInfo { double bsin; double bcos; };
    BurstInfo detectBurst(const quint16 *lineData,
                          const LdDecodeMetaData::VideoParameters &vp,
                          bool floorEnable,
                          double floorFactor)
    {
        double bsin = 0.0, bcos = 0.0;
        for (int i = vp.colourBurstStart; i < vp.colourBurstEnd; ++i) {
            const double s = lineData[i];
            bsin += s * sin4fsc(i);
            bcos += s * cos4fsc(i);
        }
        const int len = vp.colourBurstEnd - vp.colourBurstStart;
        if (len > 0) { const double invLen = 1.0 / len; bsin *= invLen; bcos *= invLen; }
        double mag = std::sqrt(bsin * bsin + bcos * bcos);

        if (floorEnable && mag < floorFactor && mag > 1e-9) {
            const double s = floorFactor / mag;
            bsin *= s; bcos *= s; mag = floorFactor;
        }
        if (mag > 1e-9) { const double invMag = 1.0 / mag; bsin *= invMag; bcos *= invMag; }
        else { bsin = 0.0; bcos = 1.0; }
        return {bsin, bcos};
    }
}

// 1D horizontal bandpass: isolates subcarrier energy by subtracting the average
// of the samples two positions either side (a 2-tap comb at 2fsc), scaled by 0.5.
// This is the simplest possible chroma separator and serves as the baseline
// reference for 2D and 3D candidates. Result written to clpbuffer[0].
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

// Locked-path pre-processing: burst detection, raw composite demodulation into
// TRI/TRQ, and a per-line affine solve stored in lineAffineLocked.
//
// Note: We intentionally do not overwrite clpbuffer[0] here; split1D populates
// clpbuffer[0] (1D bandpass), and buildPhaseCorrected1D demodulates that using
// the locked basis and applies the stored affine afterwards.
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
    if ((int)demodBurstCos.size() < requiredLines) {
        demodBurstCos.assign(requiredLines, 1.0f);
        demodBurstSin.assign(requiredLines, 0.0f);
    }
    if ((int)demodLUTTi_locked.size() < requiredLines) {
        demodLUTTi_locked.assign(requiredLines, std::array<float,4>{});
        demodLUTTq_locked.assign(requiredLines, std::array<float,4>{});
    }
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

    // --- Pass 1: burst detection -> demodBurstCos/Sin ---
    for (int line = firstLine; line < lastLine; ++line) {
        const quint16 *rawLine = rawbuffer.data() + line * fullWidth;
        auto burst = detectBurst(rawLine, videoParameters, floorEnable, floorFactor);
        double bcos = burst.bcos, bsin = burst.bsin;
        const double bc2 = bcos * cRb - bsin * sRb;
        const double bs2 = bcos * sRb + bsin * cRb;
        demodBurstCos[line] = static_cast<float>(bc2);
        demodBurstSin[line] = static_cast<float>(bs2);

        double lutTi[4], lutTq[4];
        fusedDemodLUT(bc2, bs2, spLUT_locked, cpLUT_locked, lutTi, lutTq);
        for (int i = 0; i < 4; ++i) {
            demodLUTTi_locked[line][i] = (float)lutTi[i];
            demodLUTTq_locked[line][i] = (float)lutTq[i];
        }
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
            const float *lutTi = demodLUTTi_locked[line].data();
            const float *lutTq = demodLUTTq_locked[line].data();

            double dc = (double)rawLine[left];
            constexpr double DC_ALPHA = 1.0 / 64.0;

            for (int xi = 0; xi < width; ++xi) {
                const int h      = left + xi;
                dc += DC_ALPHA * ((double)rawLine[h] - dc);
                const double vraw = (double)rawLine[h] - dc;
                const int ph = (h & 3);
                triRow[xi] = (float)(vraw * (double)lutTi[ph]);
                trqRow[xi] = (float)(vraw * (double)lutTq[ph]);
            }
        }
    }

    // --- Pass 3: sinusoidal fit + affine solve -> lineAffineLocked ---
    // Reads TRI/TRQ from Pass 2. For each sample, estimates local chroma amplitude
    // from a windowed mean of TRI/TRQ magnitudes, computes fitted IQ, vets by windowed
    // residual ratio. The fitted IQ serves as the reference for the per-line affine
    // solve, which is stored in lineAffineLocked for buildPhaseCorrected1D to apply.
    {
        const int WIN  = std::max(4, (T.SINFIT_WIN_SAMPLES / 4) * 4);
        const int HALF = WIN / 2;
        const bool doAffine = configuration.residualVideo && T.Y_LINE_AFFINE_TRIM_ENABLE;

        for (int line = firstLine; line < lastLine; ++line) {
            const quint16 *rawLine = rawbuffer.data() + line * fullWidth;
            const double bcos      = (double)demodBurstCos[line];
            const double bsin      = (double)demodBurstSin[line];
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
                const double spk = spLUT_locked[hk & 3];
                const double cpk = cpLUT_locked[hk & 3];
                const double rik = (double)triRow[k];
                const double rqk = (double)trqRow[k];
                const double mag_k = std::hypot(rik, rqk);
                magRow[k] = mag_k;
                if (mag_k > 1e-9) {
                    const double fitted_k = 0.5 * ((rik * bcos + rqk * bsin) * spk
                                                  + (-rik * bsin + rqk * bcos) * cpk);
                    const double corr_k   = (double)rawLine[hk] - fitted_k;
                    const double rsk      = corr_k * spk * 2.0;
                    const double rck      = corr_k * cpk * 2.0;
                    resRow[k] = std::hypot(rsk * bcos - rck * bsin,
                                           rsk * bsin + rck * bcos);
                } else {
                    resRow[k] = 0.0;
                }
            }

            // Sliding window sums for amp/res. The window shifts to stay inside bounds.
            const int winN = (width <= WIN) ? width : WIN;
            int a = 0;
            int b = winN - 1;
            double sumAmp = 0.0, sumRes = 0.0;
            for (int k = a; k <= b; ++k) { sumAmp += magRow[k]; sumRes += resRow[k]; }

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
                    while (a < aWant) { sumAmp -= magRow[a]; sumRes -= resRow[a]; ++a; }
                    while (a > aWant) { --a; sumAmp += magRow[a]; sumRes += resRow[a]; }
                    while (b < bWant) { ++b; sumAmp += magRow[b]; sumRes += resRow[b]; }
                    while (b > bWant) { sumAmp -= magRow[b]; sumRes -= resRow[b]; --b; }
                }
                const double ampEst = sumAmp / (double)winN;
                const double resAmp = sumRes / (double)winN;

                // Fitted IQ at xi: raw demod direction scaled to windowed amplitude
                const double mag0 = std::hypot(ri, rq);
                double fI = ri, fQ = rq;
                if (mag0 > 1e-9) {
                    fI = ri * (ampEst / mag0);
                    fQ = rq * (ampEst / mag0);
                }

                const double ratio = (ampEst > 1e-9) ? (resAmp / ampEst) : 1.0;

                // Accumulate affine matrices using vetted fitted IQ as reference
                if (doAffine && ratio <= T.SINFIT_VET_THRESHOLD_IRE) {
                    STT[0][0] += fI*fI; STT[0][1] += fI*fQ;
                    STT[1][0] += fI*fQ; STT[1][1] += fQ*fQ;
                    SRT[0][0] += ri*fI; SRT[0][1] += ri*fQ;
                    SRT[1][0] += rq*fI; SRT[1][1] += rq*fQ;
                }
            }
            // Affine solve — stored for buildPhaseCorrected1D to apply after split1D
            if (doAffine) {
                LineAffine &la = lineAffineLocked[line];
                la.valid = false;
                double STTinv[2][2];
                if (mat2_inv(STT, STTinv)) {
                    double tmp[2][2], A[2][2];
                    mat2_mul(SRT, STTinv, tmp);
                    A[0][0]=tmp[0][0]; A[0][1]=tmp[0][1];
                    A[1][0]=tmp[1][0]; A[1][1]=tmp[1][1];
                    double Rm[2][2], U[2][2];
                    polar_decompose_2x2(A, Rm, U);
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
        }
    }
}

// Demodulates clpbuffer[0] into demodTI_flat/demodTQ_flat using the locked basis
// and the per-line affine solved in phaseLocked, then remods back to
// phase-normalised composite in clpbuffer[1]. Populates demodTI/TQ for downstream
// consumers (computeFrameIQLine, filterIQLocked, scoreFieldVsFrame) and writes
// the phase-normalised chroma to clpbuffer[1] as the reference used by split2D's
// Field/Frame scoring.
void Comb::FrameBuffer::buildPhaseCorrected1D()
{
    const int first  = videoParameters.firstActiveFrameLine;
    const int last   = videoParameters.lastActiveFrameLine;
    const int left   = videoParameters.activeVideoStart;
    const int right  = videoParameters.activeVideoEnd;
    const int width  = right - left;
    const auto &T    = configuration.tunables;

    if (width <= 0 || first >= last) return;

    for (int line = first; line < last; ++line) {
        const double *src = clpbuffer[0].pixel[line];
        double       *dst = clpbuffer[1].pixel[line];

        const double bcos = (line < (int)demodBurstCos.size()) ? (double)demodBurstCos[line] : 1.0;
        const double bsin = (line < (int)demodBurstSin.size()) ? (double)demodBurstSin[line] : 0.0;
        double lutTi[4], lutTq[4];
        if (line < (int)demodLUTTi_locked.size()) {
            for (int i = 0; i < 4; ++i) {
                lutTi[i] = (double)demodLUTTi_locked[line][i];
                lutTq[i] = (double)demodLUTTq_locked[line][i];
            }
        } else {
            fusedDemodLUT(bcos, bsin, spLUT_locked, cpLUT_locked, lutTi, lutTq);
        }

        float *tiRow = demodTI_line(line);
        float *tqRow = demodTQ_line(line);

        const LineAffine *lineAffine = nullptr;
        if (configuration.residualVideo && T.Y_LINE_AFFINE_TRIM_ENABLE
                && line >= 0 && line < (int)lineAffineLocked.size()
                && lineAffineLocked[line].valid) {
            lineAffine = &lineAffineLocked[line];
        }
        auto applyLineAffine = [&](double &ti, double &tq) {
            if (!lineAffine) return;
            const double ai = lineAffine->R[0][0] * ti + lineAffine->R[0][1] * tq;
            const double aq = lineAffine->R[1][0] * ti + lineAffine->R[1][1] * tq;
            ti = ai;
            tq = aq;
        };

        auto sampleSrc = [&](int rel)->double {
            rel = std::clamp(rel, 0, width - 1);
            return src[left + rel];
        };
        auto sampleIQ = [&](int rel, double &outI, double &outQ) {
            rel = std::clamp(rel, 0, width - 1);
            const int hh = left + rel;
            const int hhPh = (hh & 3);
            const double cc = src[hh];
            outI = cc * lutTi[hhPh];
            outQ = cc * lutTq[hhPh];
            applyLineAffine(outI, outQ);
        };

        for (int xi = 0; xi < width; ++xi) {
            const int h    = left + xi;
            const double c  = src[h];

            const int ph = (h & 3);
            double ti = c * lutTi[ph];
            double tq = c * lutTq[ph];
            applyLineAffine(ti, tq);

            double intakeNyquistRiskIRE = 0.0;
            double lumaIncursionRiskIRE = 0.0;
            double residualFitErrorIRE = 0.0;
            double fine = 0.0, mid = 0.0, coarse = 0.0;

            if (xi >= 4 && xi < width - 4) {
                const double c0 = src[left + xi];
                fine = std::fabs(c0 - 0.5 * (src[left + xi - 1] + src[left + xi + 1])) * invIreScale;
                mid = std::fabs(c0 - 0.5 * (src[left + xi - 2] + src[left + xi + 2])) * invIreScale;
                coarse = std::fabs(c0 - 0.5 * (src[left + xi - 4] + src[left + xi + 4])) * invIreScale;
            } else {
                fine = std::fabs(sampleSrc(xi) -
                                 0.5 * (sampleSrc(xi - 1) + sampleSrc(xi + 1))) * invIreScale;
                mid = std::fabs(sampleSrc(xi) -
                                0.5 * (sampleSrc(xi - 2) + sampleSrc(xi + 2))) * invIreScale;
                coarse = std::fabs(sampleSrc(xi) -
                                   0.5 * (sampleSrc(xi - 4) + sampleSrc(xi + 4))) * invIreScale;
            }
            const double denom = fine + mid + coarse + 1e-9;
            const double fineFrac = fine / denom;
            const double nonFineFrac = std::max(mid, coarse) / denom;
            const double dominance = std::clamp((fineFrac - nonFineFrac - 0.15) / 0.35, 0.0, 1.0);
            intakeNyquistRiskIRE = fine * dominance;

            double tiLm1 = 0.0, tqLm1 = 0.0, tiLp1 = 0.0, tqLp1 = 0.0;
            double tiLm2 = 0.0, tqLm2 = 0.0, tiLp2 = 0.0, tqLp2 = 0.0;
            if (xi >= 2 && xi < width - 2) {
                {
                    const int hh = left + xi - 1;
                    const double cc = src[hh];
                    const int hhPh = (hh & 3);
                    tiLm1 = cc * lutTi[hhPh];
                    tqLm1 = cc * lutTq[hhPh];
                    applyLineAffine(tiLm1, tqLm1);
                }
                {
                    const int hh = left + xi + 1;
                    const double cc = src[hh];
                    const int hhPh = (hh & 3);
                    tiLp1 = cc * lutTi[hhPh];
                    tqLp1 = cc * lutTq[hhPh];
                    applyLineAffine(tiLp1, tqLp1);
                }
                {
                    const int hh = left + xi - 2;
                    const double cc = src[hh];
                    const int hhPh = (hh & 3);
                    tiLm2 = cc * lutTi[hhPh];
                    tqLm2 = cc * lutTq[hhPh];
                    applyLineAffine(tiLm2, tqLm2);
                }
                {
                    const int hh = left + xi + 2;
                    const double cc = src[hh];
                    const int hhPh = (hh & 3);
                    tiLp2 = cc * lutTi[hhPh];
                    tqLp2 = cc * lutTq[hhPh];
                    applyLineAffine(tiLp2, tqLp2);
                }
            } else {
                sampleIQ(xi - 1, tiLm1, tqLm1);
                sampleIQ(xi + 1, tiLp1, tqLp1);
                sampleIQ(xi - 2, tiLm2, tqLm2);
                sampleIQ(xi + 2, tiLp2, tqLp2);
            }

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

            if (line >= 0 && line < (int)fvfMetrics.size() &&
                xi < (int)fvfMetrics[line].size())
            {
                fvfMetrics[line][xi].intakeNyquistRiskIRE = intakeNyquistRiskIRE;
                fvfMetrics[line][xi].lumaIncursionRiskIRE = lumaIncursionRiskIRE;
                fvfMetrics[line][xi].residualFitErrorIRE = residualFitErrorIRE;
            }
            if (line >= 0 && line < (int)ownershipEvidence.size() &&
                xi < (int)ownershipEvidence[line].size())
            {
                OwnershipEvidence &e = ownershipEvidence[line][xi];
                e.bandpassFineIRE = fine;
                e.bandpassMidIRE = mid;
                e.bandpassCoarseIRE = coarse;
                e.lumaExcursionIRE = intakeNyquistRiskIRE;
                e.residualFitErrorIRE = residualFitErrorIRE;
                e.lumaIncursionRiskIRE = lumaIncursionRiskIRE;
                e.locked1DChromaIRE = std::hypot(ti, tq) * invIreScale;
                const double lumaT = std::clamp(lumaIncursionRiskIRE / 8.0, 0.0, 1.0);
                const double chromaStrength = std::clamp((e.locked1DChromaIRE - 2.0) / 10.0, 0.0, 1.0);
                const double chromaCoherence = std::clamp(1.0 - (residualFitErrorIRE / 12.0), 0.0, 1.0);
                e.lumaClaim = lumaT;
                e.chromaClaim = chromaStrength * chromaCoherence * (1.0 - 0.5 * lumaT);
                e.uncertainClaim = std::clamp(1.0 - std::max(e.lumaClaim, e.chromaClaim), 0.0, 1.0);
            }

            tiRow[xi] = (float)ti;
            tqRow[xi] = (float)tq;

            // Remodulate onto the exact 4fsc sample grid (h&3). This deliberately
            // drops the fractional-basis shift (CAL_EPS_SAMPLES) for the remod only,
            // to preserve perfect 4-sample periodicity for downstream comb stages.
            const double sp4 = sin4fsc(ph);
            const double cp4 = cos4fsc(ph);
            const double rsin =  ti * bcos + tq * bsin;
            const double rcos = -ti * bsin + tq * bcos;
            dst[h] = 0.5 * (rsin * sp4 + rcos * cp4);
        }

        // After remodulating onto the integer 4fsc grid, apply a bucket-preserving
        // same-phase smooth (±4) once per line. This avoids recomputing the same
        // operation in multiple Field comb paths during FVF, and does not touch
        // the demodTI/TQ arrays used by Frame B IQ.
        if (FIELD_BUCKET_SMOOTH_STRENGTH > 0.0) {
            if ((int)scratch_filter_temp.size() < width) scratch_filter_temp.assign(width, 0.0);

            auto reflectXi = [&](int x)->int {
                if (x < 0) return -x;
                if (x >= width) return (width - 1) - (x - (width - 1));
                return x;
            };

            for (int xi = 0; xi < width; ++xi) {
                const int xm4 = reflectXi(xi - 4);
                const int xp4 = reflectXi(xi + 4);
                const double raw = dst[left + xi];
                const double est = 0.5 * (dst[left + xm4] + dst[left + xp4]);
                scratch_filter_temp[xi] = raw + (est - raw) * FIELD_BUCKET_SMOOTH_STRENGTH;
            }
            for (int xi = 0; xi < width; ++xi) dst[left + xi] = scratch_filter_temp[xi];
        }
    }
}

void Comb::FrameBuffer::rebuildLockedDemodFromSelectedComb()
{
    const int first  = videoParameters.firstActiveFrameLine;
    const int last   = videoParameters.lastActiveFrameLine;
    const int left   = videoParameters.activeVideoStart;
    const int right  = videoParameters.activeVideoEnd;
    const int width  = right - left;
    const int srcBuf = std::clamp(static_cast<int>(configuration.dimensions) - 1, 0, 2);
    const auto &T    = configuration.tunables;

    if (width <= 0 || first >= last) return;

    for (int line = first; line < last; ++line) {
        const double *src = clpbuffer[srcBuf].pixel[line];

        const double bcos = (line < (int)demodBurstCos.size()) ? (double)demodBurstCos[line] : 1.0;
        const double bsin = (line < (int)demodBurstSin.size()) ? (double)demodBurstSin[line] : 0.0;
        double lutTi[4], lutTq[4];
        if (line < (int)demodLUTTi_locked.size()) {
            for (int i = 0; i < 4; ++i) {
                lutTi[i] = (double)demodLUTTi_locked[line][i];
                lutTq[i] = (double)demodLUTTq_locked[line][i];
            }
        } else {
            fusedDemodLUT(bcos, bsin, spLUT_locked, cpLUT_locked, lutTi, lutTq);
        }

        float *tiRow = demodTI_line(line);
        float *tqRow = demodTQ_line(line);

        const LineAffine *lineAffine = nullptr;
        if (configuration.residualVideo && T.Y_LINE_AFFINE_TRIM_ENABLE
                && line >= 0 && line < (int)lineAffineLocked.size()
                && lineAffineLocked[line].valid) {
            lineAffine = &lineAffineLocked[line];
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
            const int ph = (h & 3);
            double ti = src[h] * lutTi[ph];
            double tq = src[h] * lutTq[ph];
            applyLineAffine(ti, tq);
            tiRow[xi] = (float)ti;
            tqRow[xi] = (float)tq;
        }
    }
}

// 2D comb scorer (4-member Field-vs-Frame election)
// Paradigm: sample @ ±1 (interfield) = Frame. Sample @ ±2 (intra) = Field.
//
// Candidates:
//   idx 0 : Field A — same-line ±2 influenced by ±4 (intra-field, primary phase)
//   idx 1 : Field B — same-line ±2 (intra-field, alt phase)
//   idx 2 : Frame A — precleaned interfield Frame (1D-conditioned same-phase
//                     blend of framePreclean; cancels 1D chroma variations)
//   idx 3 : Frame B — raw interfield Frame (unconditioned; preserves luma
//                     horizontal edge detail)
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
    const double *fieldA,
    const double *fieldB,
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

    // Radius of the horizontal neighbor window used in cross-domain estimation
    const int  NEIGH_RAD        = std::max(1, std::min(3, T.NEIGH_WIN_RADIUS));

    // Minimum run length (in pixels) for field commitment when suppressing interfield teeth
    const int  FIELD_BLOCK_SIZE = 4;

    // Vertical luma contrast threshold above which we consider the field environment "active" (IRE)
    const double VERT_THRESH_IRE    = T.FIELD_VERT_DISAGREE_THRESH_IRE;

    // Horizontal luma edge threshold above which we treat the pixel as a luma transition (IRE)
    const double HEDGE_THRESH_IRE   = T.FIELD_LUMA_EDGE_THRESH_IRE;

    // Chroma magnitude above which the pixel is considered saturated enough to influence election (IRE)
    const double CHROMA_STRONG_IRE  = 6.0;

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

    auto sample1D = [&](int rel)->double {
        if (lateral1D) {
            int r = std::clamp(rel, 0, width - 1);
            return lateral1D[r];
        } else {
            int h = left + std::clamp(rel, 0, width - 1);
            return clpbuffer[srcBufIndex].pixel[line][h];
        }
    };
    auto sampleRawVert = [&](int ln, int rel)->double {
        if (ln < firstLine) ln = firstLine;
        if (ln >= lastLine) ln = lastLine - 1;
        rel = std::clamp(rel, 0, width - 1);
        int h = left + rel;
        return clpbuffer[srcBufIndex].pixel[ln][h];
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
    auto notchScalar = [&](int ln, int r) -> double {
        double c  = sampleRawVert(ln, r);
        double l  = sampleRawVert(ln, r - 2);
        double rv = sampleRawVert(ln, r + 2);
        return 0.25 * (l + 2.0 * c + rv);
    };
    auto vertContrastIRE = [&](int rel)->double {
        int upLine = line - 2, dnLine = line + 2;
        if (upLine < firstLine || dnLine >= lastLine) return 0.0;
        int h = left + rel;
        double up = clpbuffer[srcBufIndex].pixel[upLine][h];
        double dn = clpbuffer[srcBufIndex].pixel[dnLine][h];
        return std::fabs(up - dn) * invI;
    };
    auto horizEdgeIRE = [&](int rel)->double {
        int h = left + rel;
        int hm1 = std::max(left, h - 1);
        int hp1 = std::min(right - 1, h + 1);
        double cL = clpbuffer[srcBufIndex].pixel[line][hm1];
        double cR = clpbuffer[srcBufIndex].pixel[line][hp1];
        return std::fabs(cR - cL) * invI;
    };
    auto vCoherenceErrFrameIRE = [&](int rel, double FR)->double {
        const int h = left + rel;
        auto sampleLine = [&](int ln)->double {
            int l = std::clamp(ln, firstLine, lastLine - 1);
            return clpbuffer[srcBufIndex].pixel[l][h];
        };
        double Cup = sampleLine(line - 1);
        double C0  = sampleLine(line);
        double Cdn = sampleLine(line + 1);
        return 0.5 * (std::fabs(Cup - FR) + std::fabs(Cdn - FR)) * invI;
    };

    if ((int)scratch_fvf_winner.size() != width) {
        scratch_fvf_winner.assign(width, 1);
        scratch_fvf_winner2.assign(width, 1);
        scratch_fvf_outVal.assign(width, 0.0);
        scratch_fvf_outShade.assign(width, 0.35f);
        scratch_fvf_diffFVF.assign(width, 0.0);
        scratch_fvf_satMap.assign(width, 0.0);
        scratch_fvf_frameAVal.assign(width, 0.0);
    } else {
        std::fill(scratch_fvf_winner.begin(), scratch_fvf_winner.end(), 1);
        std::fill(scratch_fvf_outVal.begin(), scratch_fvf_outVal.end(), 0.0);
        std::fill(scratch_fvf_outShade.begin(), scratch_fvf_outShade.end(), 0.35f);
        std::fill(scratch_fvf_diffFVF.begin(), scratch_fvf_diffFVF.end(), 0.0);
        std::fill(scratch_fvf_satMap.begin(), scratch_fvf_satMap.end(), 0.0);
        if ((int)scratch_fvf_frameAVal.size() != width)
            scratch_fvf_frameAVal.assign(width, 0.0);
        else
            std::fill(scratch_fvf_frameAVal.begin(), scratch_fvf_frameAVal.end(), 0.0);
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

    struct Cond1D {
        double raw = 0.0;
        double score = 0.0;
        double outlierIRE = 0.0;
    };
    auto condSamePhase = [&](const double *arr, int rel) -> Cond1D {
        Cond1D c;
        c.raw = arr[rel];
        // Bucketed (h&3) conditioning: compare against same-phase neighbors (±4),
        // not adjacent/±2 samples, to avoid disrupting composite alternation.
        const int rm4 = std::clamp(rel - 4, 0, width - 1);
        const int rp4 = std::clamp(rel + 4, 0, width - 1);
        const double est = 0.5 * (arr[rm4] + arr[rp4]); // adjust this multiplier (+/-) to trim conditioning
        c.outlierIRE = std::fabs(c.raw - est) * invI;

        // Horizontal-only outlier conditioning for scoring: if the pixel strongly
        // deviates from same-phase bucket neighbors (±4), blend toward the estimate.
        const double OUTLIER_WARN_IRE = 3.0;
        const double OUTLIER_FULL_IRE = 10.0;
        double t = (c.outlierIRE - OUTLIER_WARN_IRE) / (OUTLIER_FULL_IRE - OUTLIER_WARN_IRE);
        t = std::clamp(t, 0.0, 1.0);
        // Dial back: even a full outlier only corrects halfway toward est.
        t *= 0.5;
        c.score = c.raw + (est - c.raw) * t;
        return c;
    };
    auto condSamePhaseVec = [&](const std::vector<double> &vec, int rel) -> Cond1D {
        Cond1D c;
        c.raw = vec[rel];
        const int rm4 = std::clamp(rel - 4, 0, width - 1);
        const int rp4 = std::clamp(rel + 4, 0, width - 1);
        const double est = 0.5 * (vec[rm4] + vec[rp4]);
        c.outlierIRE = std::fabs(c.raw - est) * invI;

        const double OUTLIER_WARN_IRE = 3.0;
        const double OUTLIER_FULL_IRE = 10.0;
        double t = (c.outlierIRE - OUTLIER_WARN_IRE) / (OUTLIER_FULL_IRE - OUTLIER_WARN_IRE);
        t = std::clamp(t, 0.0, 1.0);
        t *= 0.5;
        c.score = c.raw + (est - c.raw) * t;
        return c;
    };

    for (int rel = 0; rel < width; ++rel) {
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

        const Cond1D FA_c = condSamePhase(fieldA, rel);
        const Cond1D FB_c = condSamePhase(fieldB, rel);
        const Cond1D FR_c = condSamePhaseVec(framePreclean, rel);

        // Cache the precleaned Frame A value (the 1D-conditioned blend);
        // the island filter and post-processing read from this scratch.
        scratch_fvf_frameAVal[rel] = FR_c.score;

        // Use conditioned candidates for scoring; outputs depend on which
        // Frame variant wins (Frame A → conditioned, Frame B → raw).
        const double FA_s = FA_c.score;
        const double FB_s = FB_c.score;
        const double FR_s = FR_c.score;

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

        double C0   = sampleRawVert(line,     rel);
        double Cpm1 = sampleRawVert(line - 1, rel);
        double Cpp1 = sampleRawVert(line + 1, rel);
        double Cpm2 = sampleRawVert(line - 2, rel);
        double Cpp2 = sampleRawVert(line + 2, rel);

        double frameLikeStack = 0.5 * (Cpm1 + Cpp1);
        double fieldLikeStack = 0.5 * (Cpm2 + Cpp2);
        double diff_stack_ire = std::fabs(frameLikeStack - fieldLikeStack) * invI;

        double diff_candA_ire = std::fabs(lumFR - lumFA) * invI;
        double diff_candB_ire = std::fabs(lumFR - lumFB) * invI;
        double diff_cand_ire  = std::min(diff_candA_ire, diff_candB_ire);
        double frameModelDistIRE = localUseFrameModel ? diff_cand_ire : diff_candA_ire;
        bool frameInsane = (frameModelDistIRE > FRAME_MAX_DIST_IRE);
    
        double interfield_luma_ire = std::fabs(
            0.5 * (notchScalar(line - 1, rel) + notchScalar(line + 1, rel))
            - notchScalar(line, rel)) * invI;
        
        double smoothed_interfield = (rel > 0)
            ? 0.5 * (interfield_luma_ire + prev_interfield_luma_ire)
            : interfield_luma_ire;
        prev_interfield_luma_ire = interfield_luma_ire;
    
	        // --- Veto Logic ---
	        bool managementVeto = (cadenceId == -2);
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

        FvfModelMetrics metrics;
        if (line >= 0 && line < (int)fvfMetrics.size() &&
            rel < (int)fvfMetrics[line].size())
        {
            metrics = fvfMetrics[line][rel];
        }
        metrics.chromaMagIRE = chromaMagIRE;
        metrics.chromaBandEnergyIRE = chromaMagIRE;
        metrics.verticalBoundaryIRE = hIRE;
        metrics.horizontalBoundaryIRE = vIRE;
        metrics.fieldFrameDivergenceIRE = diff_fvf_ire;
        metrics.interfieldDistinctIRE = smoothed_interfield;
        metrics.frameToFieldModelIRE = diff_candA_ire;
        metrics.frameToBestFieldIRE = diff_cand_ire;
        metrics.frameModel = localUseFrameModel;
        metrics.managementVeto = managementVeto;
        metrics.frameVertCoherent = b2VertCoherent;
        metrics.vdisSoft = vdisSoft;
        metrics.vdisHard = vdisHard;

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
            consider(2, FR_c.score, safeR, 0.7f);   // Frame A (precleaned)
            consider(3, FR_raw,     safeR, 0.85f);  // Frame B (raw)

            idx   = bestIdx;
            val   = bestVal;
            shade = bestSh;
        }
        else {
            double devA = 0.0, devB = 0.0, devR_A = 0.0, devR_B = 0.0;

            if (T.FVF_SHAPE_STRENGTH > 0.0) {
                double m_c = targetModel;
                auto getM = [&](int r) {
                    if (localUseFrameModel)
                        return framePreclean[std::clamp(r, 0, width - 1)];
                    else
                        return fieldA[std::clamp(r, 0, width - 1)];
                };
                double m_l = getM(rel - 1);
                double m_r = getM(rel + 1);
                double shapeModel = m_c - 0.5 * (m_l + m_r);

                auto getShapeScore = [&](double v, double v_l, double v_r) {
                    double shapeVal = v - 0.5 * (v_l + v_r);
                    return std::fabs(shapeVal - shapeModel);
                };

                double FA_l = fieldA[std::clamp(rel - 1, 0, width - 1)];
                double FA_r = fieldA[std::clamp(rel + 1, 0, width - 1)];
                double FB_l = fieldB[std::clamp(rel - 1, 0, width - 1)];
                double FB_r = fieldB[std::clamp(rel + 1, 0, width - 1)];
                double FR_l = framePreclean[std::clamp(rel - 1, 0, width - 1)];
                double FR_r = framePreclean[std::clamp(rel + 1, 0, width - 1)];
                const std::vector<double> &rawFrameVec = frameRaw ? *frameRaw : framePreclean;
                double FRB_l = rawFrameVec[std::clamp(rel - 1, 0, width - 1)];
                double FRB_r = rawFrameVec[std::clamp(rel + 1, 0, width - 1)];

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

            // ------------------------------------------------------------
            // Field A confidence: if A reports low confidence, make A pay
            // its own cost and let the election decide among the remaining
            // witnesses. Do not promote Field B directly from this signal.
            // ------------------------------------------------------------
            double gA = 1.0;
            if (line >= 0 && line < (int)w2d_fieldA_gate.size())
                gA = w2d_fieldA_gate[line][std::clamp(rel, 0, width - 1)];
            gA = std::clamp(gA, 0.0, 1.0);

            double gAm = gA, gAp = gA;
            if (line >= 0 && line < (int)w2d_fieldA_gate.size()) {
                gAm = w2d_fieldA_gate[line][std::clamp(rel - 1, 0, width - 1)];
                gAp = w2d_fieldA_gate[line][std::clamp(rel + 1, 0, width - 1)];
                gAm = std::clamp(gAm, 0.0, 1.0);
                gAp = std::clamp(gAp, 0.0, 1.0);
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
                if (!managementVeto && b2VertCoherent) {
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
                auto iqMag = [&](int r)->double {
                    r = std::clamp(r, 0, width - 1);
                    const auto &z = (*frameIQ)[r];
                    return std::hypot(z.real(), z.imag());
                };

                const double fine = std::fabs(iqMag(rel) -
                                              0.5 * (iqMag(rel - 1) + iqMag(rel + 1)));
                const double mid  = std::fabs(iqMag(rel) -
                                              0.5 * (iqMag(rel - 2) + iqMag(rel + 2)));
                const double coarse = std::fabs(iqMag(rel) -
                                                0.5 * (iqMag(rel - 4) + iqMag(rel + 4)));

                const double denom = fine + mid + coarse + 1e-9;
                const double fineFrac   = fine   / denom;
                const double midFrac    = mid    / denom;
                const double coarseFrac = coarse / denom;
                metrics.iqFineFrac = fineFrac;
                metrics.iqMidFrac = midFrac;
                metrics.iqCoarseFrac = coarseFrac;
                metrics.iqCoherence = 1.0 - std::clamp(coarseFrac, 0.0, 1.0);

                const double frameScaleBiasStrength = localUseFrameModel
                    ? T.FRAME_SCALE_BIAS_STRENGTH_PROGRESSIVE
                    : T.FRAME_SCALE_BIAS_STRENGTH_INTERLACE;
                const double FRAME_COARSE_CLAMP     = 0.60;
                const double FIELD_SWITCH_STRENGTH  = 0.10;

                const bool frameScaleDominant = ((fineFrac + midFrac) > coarseFrac + 0.10);

                double frameABonus = frameScaleBiasStrength * midFrac;
                frameABonus *= (1.0 - FRAME_COARSE_CLAMP * coarseFrac);
                scoreR_A *= (1.0 - frameABonus);

                double frameBBonus = frameScaleBiasStrength * fineFrac;
                frameBBonus *= (1.0 - FRAME_COARSE_CLAMP * coarseFrac);
                scoreR_B *= (1.0 - frameBBonus);

                if (!frameScaleDominant) {
                    const double bias = std::clamp(coarseFrac - midFrac, -1.0, 1.0);
                    scoreA *= (1.0 - FIELD_SWITCH_STRENGTH * bias);
                    scoreB *= (1.0 + FIELD_SWITCH_STRENGTH * bias);
                }
            }

            // Progressive: if Frame is the only witness carrying extra chroma energy (often
            // vertical "noise" that crosses lines) and/or its IQ is locally incoherent,
            // bias away from Frame so the interfield combs can clean it up.
            if (localUseFrameModel && !frameInsane) {
                const double fAAbsIRE = std::fabs(FR) * invI;
                const double fBAbsIRE = std::fabs(FR_raw) * invI;
                const double faAbsIRE = std::fabs(FA) * invI;
                const double fbAbsIRE = std::fabs(FB) * invI;

                const double fieldSupportIRE = std::max(faAbsIRE, fbAbsIRE);
                const double frameAOnlyIRE = std::max(0.0, fAAbsIRE - fieldSupportIRE);
                const double frameBOnlyIRE = std::max(0.0, fBAbsIRE - fieldSupportIRE);

                // Penalize frame-only energy (but keep it gentle; fields can be wrong too).
                if (frameAOnlyIRE > 0.75) scoreR_A += 0.10 * frameAOnlyIRE;
                if (frameBOnlyIRE > 0.75) scoreR_B += 0.10 * frameBOnlyIRE;

                // FrameIQ describes the precleaned Frame A path; don't let that
                // artifact signature drag Frame B down as if it were the same signal.
                if (frameIQ && metrics.iqCoherence > 0.0) {
                    const double incoh = std::clamp(1.0 - metrics.iqCoherence, 0.0, 1.0);
                    if (incoh > 0.35) {
                        scoreR_A *= (1.0 + 0.18 * (incoh - 0.35) / 0.65);
                    }
                }
            }

            // ------------------------------------------------------------
            // Saturation regime: in highly saturated regions, Frame is often
            // the least visually toxic when coherent, but Field B tends to
            // introduce zipper/alternation more readily than Field A.
            // Apply a soft bias rather than a hard override.
            // ------------------------------------------------------------
            if (sat_t > 0.0) {
                // Penalize Field B more than Field A as saturation rises.
                const double SAT_FIELD_A_PEN = 0.06;
                const double SAT_FIELD_B_PEN = 0.24;
                scoreA *= (1.0 + SAT_FIELD_A_PEN * sat_t);
                scoreB *= (1.0 + SAT_FIELD_B_PEN * sat_t);

                // Reward Frame when it is allowed/coherent (both regimes),
                // but never punch through management veto or insane frame.
                if (!managementVeto && b2VertCoherent && !frameInsane) {
                    const double SAT_FRAME_BONUS = 0.18;
                  //  scoreR_A *= (1.0 - SAT_FRAME_BONUS * sat_t); --removed due to boundary issues
                    scoreR_B *= (1.0 - SAT_FRAME_BONUS * sat_t);
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
                    const double *srcLine = clpbuffer[srcBufIndex].pixel[line] + left;
                    auto srcNotch = [&](int r)->double {
                        r = std::clamp(r, 0, width - 1);
                        return getNotchLumaEven2(srcLine, r, width);
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

                    if (!stableStep) goto no_sharp_reward;

                    // Candidate step measured using notch luma (reduces composite phase chatter).
                    const int rm2 = std::max(0, rel - 2);
                    const int rp2 = std::min(width - 1, rel + 2);
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
                        const double W_EDGE_SHARP = 0.10;
                        score *= (1.0 - W_EDGE_SHARP * sharp * stepStrength);
                    };

                    applySharpReward(scoreA, fieldA, nullptr);
                    applySharpReward(scoreB, fieldB, nullptr);
                    applySharpReward(scoreR_A, nullptr, &framePreclean);
                    applySharpReward(scoreR_B, nullptr, frameRaw ? frameRaw : &framePreclean);
                }
                no_sharp_reward: ;
            }

            // VDIS
            if (!vdisHard &&
                hIRE < T.NEIGHBOR_EST_EDGE_MAX_IRE &&
                diff_stack_ire < T.NEIGHBOR_EST_FVF_MAX_IRE &&
                chromaMagIRE < T.NEIGHBOR_EST_SAT_MAX_IRE)
            {
                int r_m2 = std::max(0,         rel - 2);
                int r_p2 = std::min(width - 1, rel + 2);

                double estA2 = 0.5 * (fieldA[r_m2]  + fieldA[r_p2]);
                double estB2 = 0.5 * (fieldB[r_m2]  + fieldB[r_p2]);
                double estF2 = 0.5 * (framePreclean[r_m2] + framePreclean[r_p2]);
                double estRB2 = frameRawData
                    ? 0.5 * (frameRawData[r_m2] + frameRawData[r_p2])
                    : estF2;

                double E2 = median4_average_middle(estA2, estB2, estF2, estRB2);

                bool allowPm1 = true;
                {
                    int r_m1 = std::max(0,         rel - 1);
                    int r_p1 = std::min(width - 1, rel + 1);
                    double altF = std::fabs(framePreclean[r_m1] - framePreclean[r_p1]) * invI;
                    if (altF > 6.0) allowPm1 = false;
                }

                double E = E2;
                if (allowPm1) {
                    int r_m1 = std::max(0,         rel - 1);
                    int r_p1 = std::min(width - 1, rel + 1);

                    double estA1 = 0.5 * (fieldA[r_m1]  + fieldA[r_p1]);
                    double estB1 = 0.5 * (fieldB[r_m1]  + fieldB[r_p1]);
                    double estF1 = 0.5 * (framePreclean[r_m1] + framePreclean[r_p1]);
                    double estRB1 = frameRawData
                        ? 0.5 * (frameRawData[r_m1] + frameRawData[r_p1])
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
            // Frame A (precleaned) — favored under vertical contrast and when
            // same-phase 1D conditioning corrected a meaningful chroma deviation.
            {
                const double VERT_NORM       = std::max(VERT_THRESH_IRE, 1.0);
                const double frameA_vert_t   = std::clamp(vIRE / VERT_NORM, 0.0, 1.0);
                const double FRAME_A_VERT_BONUS = 0.18;
                scoreR_A *= (1.0 - FRAME_A_VERT_BONUS * frameA_vert_t);

                const double ONED_NORM_IRE   = 8.0;
                const double frameA_1D_t     = std::clamp(FR_c.outlierIRE / ONED_NORM_IRE, 0.0, 1.0);
                const double FRAME_A_1D_BONUS = 0.12;
                scoreR_A *= (1.0 - FRAME_A_1D_BONUS * frameA_1D_t);
            }
/*
            // Frame B (raw) — favored around horizontal boundaries.
            {
                const double HEDGE_NORM      = std::max(HEDGE_THRESH_IRE, 1.0);
                const double frameB_hedge_t  = std::clamp(hIRE / HEDGE_NORM, 0.0, 1.0);
                const double FRAME_B_HEDGE_BONUS = 0.18;
                scoreR_B *= (1.0 - FRAME_B_HEDGE_BONUS * frameB_hedge_t);
            }

            {
                const double HEDGE_NORM2      = std::max(HEDGE_THRESH_IRE, 1.0);
                const double fieldA_hedge_t  = std::clamp(hIRE / HEDGE_NORM2, 0.0, 1.0);
                const double FIELD_A_HEDGE_BONUS = 0.18;
                scoreR_B *= (1.0 - FIELD_A_HEDGE_BONUS * fieldA_hedge_t);
            }
*/
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
                if (scoreR_A <= scoreR_B) pickCandidate(2, FR_c.score, shadeA);
                else                       pickCandidate(3, FR_raw,     shadeB);
            };

            if (hIRE > HEDGE_THRESH_IRE && diff_stack_ire > 5.0) {
                // Strong horizontal luma edge regime 
                double dF1 = std::fabs(lumFRRaw - L1) * invI;
                if (dF1 <= 3.5 && diff_cand_ire <= 5.0 && !frameInsane &&
                    (scoreR_B <= scoreR_A))
                    pickCandidate(3, FR_raw, 0.85f);   // Frame B
                else {
                    if (scoreA < scoreB) pickCandidate(0, FA, 0.25f);
                    else                 pickCandidate(1, FB, 0.35f);
                }
            } else {
                if (b2VertCoherent)
                    pickCandidate(2, FR_c.score, 0.7f);   // Frame A (vertical-coherent regime)
                else if (std::min(scoreR_A, scoreR_B) + 1e-12 < scoreA * 0.85 &&
                         std::min(scoreR_A, scoreR_B) + 1e-12 < scoreB * 0.85)
                    pickBestFrame(0.7f, 0.85f);
                else if (scoreA < scoreB * 0.8)
                    pickCandidate(0, FA, 0.25f);
                else {
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
        metrics.winner = idx;
        if (line >= 0 && line < (int)fvfMetrics.size() &&
            rel < (int)fvfMetrics[line].size())
        {
            fvfMetrics[line][rel] = metrics;
        }
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
                else if (idx == 2) { outVal[rel] = scratch_fvf_frameAVal[rel]; outShade[rel] = 0.7f;  }
                else               { outVal[rel] = frameRaw ? (*frameRaw)[rel] : framePreclean[rel]; outShade[rel] = 0.85f; }
            }
        }
    };

    applyIslandFilter();

    if (!localUseFrameModel && fieldCountTotal > frameCountTotal * 2 && fieldCountTotal > 0) {
        for (int b = 0; b < width; b += FIELD_BLOCK_SIZE) {
            int e = std::min(width, b + FIELD_BLOCK_SIZE);

            double blockDivergence = 0.0;
            for (int r = b; r < e; ++r)
                blockDivergence += diffFVF[r];
            blockDivergence /= (e - b);

            if (blockDivergence * invI > FIELD_DISAGREE_IRE) {
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

    if (width <= 0 || !fieldA || !fieldB || line < 0)
        return;
    if ((int)ownershipEvidence.size() <= line)
        ownershipEvidence.resize(line + 1);
    auto &row = ownershipEvidence[line];
    if ((int)row.size() < width)
        row.assign(width, OwnershipEvidence());

    const bool haveFrameScalar = frameIQ && !frameScalar.empty();
    auto sampleFrameScalar = [&](int r)->double {
        return frameScalar[std::clamp(r, 0, (int)frameScalar.size() - 1)];
    };

    auto frameCoherence = [&](int r)->double {
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

    for (int rel = 0; rel < width; ++rel) {
        OwnershipEvidence &e = row[rel];
        const double fa = fieldA[rel];
        const double fb = fieldB[rel];
        const double fr = haveFrameScalar ? sampleFrameScalar(rel) : 0.0;

        e.fieldAChromaIRE = std::fabs(fa) * invIreScale;
        e.fieldBChromaIRE = std::fabs(fb) * invIreScale;
        e.frameChromaIRE = (frameIQ && rel < (int)frameIQ->size())
            ? std::hypot((*frameIQ)[rel].real(), (*frameIQ)[rel].imag()) * invIreScale
            : (haveFrameScalar ? std::fabs(fr) * invIreScale : 0.0);

        const double lo = haveFrameScalar ? std::min({fa, fb, fr}) : std::min(fa, fb);
        const double hi = haveFrameScalar ? std::max({fa, fb, fr}) : std::max(fa, fb);
        e.candidateSpreadIRE = (hi - lo) * invIreScale;
        e.frameFieldAgreementIRE = haveFrameScalar
            ? std::min(std::fabs(fr - fa), std::fabs(fr - fb)) * invIreScale
            : 0.0;
        e.frameIQCoherence = frameCoherence(rel);

        const double maxChromaIRE = std::max({e.locked1DChromaIRE,
                                              e.fieldAChromaIRE,
                                              e.fieldBChromaIRE,
                                              e.frameChromaIRE});
        const double lumaT = std::clamp(e.lumaIncursionRiskIRE / 8.0, 0.0, 1.0);
        const double chromaStrength = std::clamp((maxChromaIRE - 2.0) / 10.0, 0.0, 1.0);
        const double chromaCoherence = frameIQ ? e.frameIQCoherence
                                               : std::clamp(1.0 - (e.residualFitErrorIRE / 12.0), 0.0, 1.0);

        e.lumaClaim = lumaT;
        e.chromaClaim = chromaStrength * chromaCoherence * (1.0 - 0.5 * lumaT);
        e.uncertainClaim = std::clamp(1.0 - std::max(e.lumaClaim, e.chromaClaim), 0.0, 1.0);
    }
}

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
        const int ph = h & 3;
        double ti = 0.0;
        double tq = 0.0;

        if (line >= 0 && line < (int)demodLUTTi_locked.size() &&
            line < (int)demodLUTTq_locked.size())
        {
            ti = (double)demodLUTTi_locked[line][ph];
            tq = (double)demodLUTTq_locked[line][ph];
        } else {
            const double bcos = (line >= 0 && line < (int)demodBurstCos.size())
                ? (double)demodBurstCos[line]
                : 1.0;
            const double bsin = (line >= 0 && line < (int)demodBurstSin.size())
                ? (double)demodBurstSin[line]
                : 0.0;
            double lutTi[4], lutTq[4];
            fusedDemodLUT(bcos, bsin, spLUT_locked, cpLUT_locked, lutTi, lutTq);
            ti = lutTi[ph];
            tq = lutTq[ph];
        }

        const double c = sampleRow(line, rel);
        return { c * ti, c * tq };
    };

    auto sampleLockedIQ = [&](int line, int rel)->std::complex<double> {
        const int h = left + std::clamp(rel, 0, width - 1);
        const int ph = h & 3;
        double ti = 0.0;
        double tq = 0.0;

        if (line >= 0 && line < (int)demodLUTTi_locked.size() &&
            line < (int)demodLUTTq_locked.size())
        {
            ti = (double)demodLUTTi_locked[line][ph];
            tq = (double)demodLUTTq_locked[line][ph];
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
            const int phase = (left + rel) & 3;
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
    for (int line = firstLine; line < lastLine; ++line) {
        if (line < 0 || line >= (int)ownershipEvidence.size())
            continue;
        const auto &row = ownershipEvidence[line];
        if ((int)row.size() < width)
            continue;
        for (int rel = 0; rel < width; ++rel) {
            const OwnershipEvidence &e = row[rel];
            ++ownN;
            sumLumaClaim += e.lumaClaim;
            sumChromaClaim += e.chromaClaim;
            sumUncertainClaim += e.uncertainClaim;
            sumLumaIncursion += e.lumaIncursionRiskIRE;
            sumCandidateSpread += e.candidateSpreadIRE;
            sumFrameCoherence += e.frameIQCoherence;
        }
    }
    if (ownN > 0) {
        const double invOwnN = 1.0 / (double)ownN;
        msg += QString(" ownership(n=%1,luma=%2,chroma=%3,uncertain=%4,incur=%5,spread=%6,frameCoh=%7)")
            .arg(ownN)
            .arg(sumLumaClaim * invOwnN, 0, 'f', 3)
            .arg(sumChromaClaim * invOwnN, 0, 'f', 3)
            .arg(sumUncertainClaim * invOwnN, 0, 'f', 3)
            .arg(sumLumaIncursion * invOwnN, 0, 'f', 3)
            .arg(sumCandidateSpread * invOwnN, 0, 'f', 3)
            .arg(sumFrameCoherence * invOwnN, 0, 'f', 3);
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

    std::vector<float> rawDemodTI;
    std::vector<float> rawDemodTQ;

    if (configuration.phaseCompensation) {
        buildPhaseCorrected1D();
        if (needFrameIQCompute) {
            rawDemodTI = demodTI_flat;
            rawDemodTQ = demodTQ_flat;
        }
        // Preserve the per-line locked 1D source before clpbuffer[1] is overwritten
        // in-place with the 2D output as we iterate down the frame.
        if ((int)locked1DSource.size() < lastLine) locked1DSource.resize(lastLine);
        for (int line = firstLine; line < lastLine; ++line) {
            auto &row = locked1DSource[line];
            if ((int)row.size() < width) row.assign(width, 0.0);
            const double *src = clpbuffer[1].pixel[line];
            for (int rel = 0; rel < width; ++rel) row[rel] = src[left + rel];
        }
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
                const double *src1d = clpbuffer[1].pixel[line];
                for (int rel = 0; rel < width; ++rel) dst[left + rel] = src1d[left + rel];
            }
            if (writeWeights && line < (int)w2d_frame_weight.size())
                std::fill(w2d_frame_weight[line].begin(), w2d_frame_weight[line].end(), 0.0f);
        }
        reportPhaseLegStats("2d-final", 1, false);
        return;
    }

    if (!needFrameIQStorage) {
        // Not needed for this run; invalidate ring tags to avoid accidental use.
        precleanRingLine = { -1, -1, -1 };
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
    std::vector<std::complex<double>> frameIQRaw;
    std::vector<double> frameRawCenter;

    for (int line = firstLine; line < lastLine; ++line) {
        // PREP: Precompute next neighbor (line+1) into the preclean ring so the
        // current line can read its result from the ring on the next iteration.
        if (needFrameIQCompute && line + 1 < lastLine) {
            computeField2DLine(line + 1,
                               precleanLinePtrMutable(line + 1, width),
                               precleanGateLinePtrMutable(line + 1, width));
        }
        if (line >= demodLines) continue;

        computeSimpleField2DLine(line, scratch_fieldBLine.data());
        // Read Field A from the preclean ring when available (populated by the previous
        // iteration's prefetch), otherwise fall back to computing it fresh.
        if (needFrameIQCompute && havePrecleanLine(line, width)) {
            const double *cSrc = precleanLinePtr(line, width);
            std::copy(cSrc, cSrc + width, scratch_fieldLine.data());
            const double *gSrc = precleanGateLinePtr(line, width);
            if (gSrc) std::copy(gSrc, gSrc + width, scratch_fieldGate.data());
            else       std::fill(scratch_fieldGate.data(), scratch_fieldGate.data() + width, 1.0);
        } else {
            computeField2DLine(line, scratch_fieldLine.data(), scratch_fieldGate.data());
        }

        {
            const double *src1d = configuration.phaseCompensation
                                  ? nullptr
                                  : clpbuffer[0].pixel[line];
            scratch_lateralLine.assign(width, 0.0);
            if (configuration.phaseCompensation) {
                if (line >= 0 && line < (int)locked1DSource.size() &&
                    (int)locked1DSource[line].size() >= width)
                {
                    for (int rel = 0; rel < width; ++rel)
                        scratch_lateralLine[rel] = locked1DSource[line][rel];
                }
            } else {
                for (int rel = 0; rel < width; ++rel)
                    scratch_lateralLine[rel] = src1d[left + rel];
            }
        }

        if (needFrameIQCompute) {
            computeFrameIQLocked1DLine(line, frameIQRaw, &rawDemodTI, &rawDemodTQ);
            frameRawCenter.assign(width, 0.0);

            double bcosRaw = (line < (int)demodBurstCos.size()) ? (double)demodBurstCos[line] : 1.0;
            double bsinRaw = (line < (int)demodBurstSin.size()) ? (double)demodBurstSin[line] : 0.0;
            for (int rel = 0; rel < width; ++rel) {
                int h = left + rel;
                if (rel < (int)frameIQRaw.size()) {
                    const auto &Z = frameIQRaw[rel];
                    const double Ii = Z.real();
                    const double Qi = Z.imag();
                    const double s4 = sin4fsc(h);
                    const double c4 = cos4fsc(h);
                    const double lsin =  Ii * bcosRaw + Qi * bsinRaw;
                    const double lcos = -Ii * bsinRaw + Qi * bcosRaw;
                    frameRawCenter[rel] = 0.5 * (lsin * s4 + lcos * c4);
                } else {
                    frameRawCenter[rel] = 0.0;
                }
            }

            computeFrameIQPrecleanLine(line, frameIQ);
            
            scratch_fieldBCenter.assign(width, 0.0);

            for (int rel = 0; rel < width; ++rel) {
                int h = left + rel;
                if (rel < (int)frameIQ.size()) {
                    const auto &Z = frameIQ[rel];
                    const double Ii = Z.real();
                    const double Qi = Z.imag();
                    const double s4 = sin4fsc(h);
                    const double c4 = cos4fsc(h);
                    // Frame A IQ is in canonical 4fsc bucket axes; remod directly.
                    const double cOut = 0.5 * (Ii * s4 + Qi * c4);
                    scratch_fieldBCenter[rel] = cOut;
                } else {
                    scratch_fieldBCenter[rel] = 0.0;
                }
            }
            
        }

        collectCombOwnershipEvidence(
            line,
            scratch_fieldLine.data(),
            scratch_fieldBLine.data(),
            needFrameIQCompute ? scratch_fieldBCenter : scratch_frameBCenter,
            needFrameIQCompute ? &frameIQ : nullptr);

        double *dst = clpbuffer[1].pixel[line];

        if (configuration.twoDVariant == Comb::Configuration::TwoDVariant::Field) {
            for (int rel = 0; rel < width; ++rel) dst[left + rel] = scratch_fieldLine[rel];
            if (writeWeights) std::fill(w2d_frame_weight[line].begin(), w2d_frame_weight[line].end(), 0.0f);
        }
        else if (configuration.twoDVariant == Comb::Configuration::TwoDVariant::FieldB) {
            for (int rel = 0; rel < width; ++rel) dst[left + rel] = scratch_fieldBLine[rel];
            if (writeWeights) std::fill(w2d_frame_weight[line].begin(), w2d_frame_weight[line].end(), 0.35f);
        }
        else if (configuration.twoDVariant == Comb::Configuration::TwoDVariant::FramePreclean && configuration.phaseCompensation) {
            for (int rel = 0; rel < width; ++rel) dst[left + rel] = scratch_fieldBCenter[rel];
            if (writeWeights) std::fill(w2d_frame_weight[line].begin(), w2d_frame_weight[line].end(), 0.8f);
        }
        else if (configuration.twoDVariant == Comb::Configuration::TwoDVariant::FrameRaw && configuration.phaseCompensation) {
            for (int rel = 0; rel < width; ++rel) dst[left + rel] = (rel < (int)frameRawCenter.size()) ? frameRawCenter[rel] : 0.0;
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
                    scratch_fieldLine.data(),
                    scratch_fieldBLine.data(),
                    scratch_fieldBCenter,
                    &frameRawCenter,
                    scratch_outMixed.data(),
                    writeWeights,
                    scratch_lateralLine.data(),
                    &frameIQ);

                const double OUT_CLAMP_MAG = std::max(32.0 * irescale, 1.0);
                for (int rel = 0; rel < width; ++rel) {
                    double vMixed = scratch_outMixed[rel];
                
                    // Keep only the numeric-sanity fallback:
                    if (!std::isfinite(vMixed)) vMixed = scratch_fieldBLine[rel];
                
                    dst[left + rel] = vMixed;
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
            const double base1d = clpbuffer[0].pixel[line][h0];
        
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
    
    // --- Previous Field ---
    bool prevValid = false;
    if (lineNumber - 1 >= videoParameters.firstActiveFrameLine) {
        // Inter-field (line above in previous field vs line above in this field)
        // If phases match, the info is in previousFrame. If phases flip, it's in this frame.
        if (getLinePhase(lineNumber) == getLinePhase(lineNumber - 1)) {
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
    if (lineNumber + 1 < videoParameters.lastActiveFrameLine) {
        if (getLinePhase(lineNumber) == getLinePhase(lineNumber + 1)) {
             // In phase -> Next Frame contains the field
             c[CAND_NEXT_FIELD] = getCandidate(lineNumber, h, nextFrame, lineNumber + 1, h, FIELD_BONUS);
             src[CAND_NEXT_FIELD] = &nextFrame;
        } else {
             // Out of phase -> This Frame contains the field
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
    if (getLinePhase(lineNumber) == previousFrame.getLinePhase(lineNumber)) {
        c[CAND_PREV_FRAME] = getCandidate(lineNumber, h, previousFrame, lineNumber, h, FRAME_BONUS);
        src[CAND_PREV_FRAME] = &previousFrame;
    } else {
        // Invalid phase relationship (break in cadence)
        c[CAND_PREV_FRAME].penalty = 1000.0;
        src[CAND_PREV_FRAME] = nullptr;
    }

    // Check Next Frame Phase Match independently
    if (getLinePhase(lineNumber) == nextFrame.getLinePhase(lineNumber)) {
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
// routing each sample into the appropriate accumulator via a switch on (h & 3).
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

        // Apply per-line subcarrier polarity flip from lineFlip (populated in loadFields).
        int f = 1;
        if (!lineFlip.empty() &&
            lineNumber >= firstLine && lineNumber < (int)lineFlip.size()) {
            f = lineFlip[lineNumber];   // +1 or -1
        }

        double si = 0, sq = 0;
        for (qint32 h = videoParameters.activeVideoStart;
             h < videoParameters.activeVideoEnd; h++)
        {
            qint32 phase = h & 3;

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


// Locked-path demodulation: separates I and Q using the per-line burst phasor
// (demodBurstCos/Sin) computed by phaseLocked, rather than relying on the 4fsc
// sampling assumption. Also demodulates the raw composite into TRI/TRQ for
// the residual Y path, and builds the HP-Y leakage buffers (yI, yQ) needed
// by produceY. The affine stored in lineAffineLocked is applied to yI/yQ here
// to align the residual Y demod with the locked chroma reference.
void Comb::FrameBuffer::splitIQlocked()
{
    // Safety check: ensure rawbuffer has enough lines to cover active video
    const int actualHeight = rawbuffer.size() / videoParameters.fieldWidth;
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = std::min(videoParameters.lastActiveFrameLine, actualHeight);

    const int left   = videoParameters.activeVideoStart;
    const int right  = videoParameters.activeVideoEnd;
    const int srcBuf = std::clamp(static_cast<int>(configuration.dimensions) - 1, 0, 2);
    const int width  = right - left;

    // Ensure TRI/TRQ buffers sized  TI/TQ already guaranteed from phaseLocked
    const int requiredLines = lastLine + 1;
    const size_t need = static_cast<size_t>(requiredLines) * static_cast<size_t>(width);
    if (demodTRI_flat.size() < need) {
        demodTRI_flat.assign(need, 0.0f);
        demodTRQ_flat.assign(need, 0.0f);
    }

    // basisLockedInit guaranteed set by phaseLocked; guard retained for safety
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

    double* preI = scratch_preI.data();
    double* preQ = scratch_preQ.data();
    double* yhp  = scratch_yhp.data();
    double* yI   = scratch_yI.data();
    double* yQ   = scratch_yQ.data();

    const double effGI = GI_PRODUCT * configuration.gi_product;
    const double effGQ = GQ_PRODUCT * configuration.gq_product;
    const auto &T = configuration.tunables;

    for (int line = firstLine; line < lastLine; ++line) {
        const quint16 *rawLine = rawbuffer.data() + line * videoParameters.fieldWidth;

        double bcos = 1.0, bsin = 0.0;
        if (line < (int)demodBurstCos.size()) {
            bcos = (double)demodBurstCos[line];
            bsin = (double)demodBurstSin[line];
        }
        double lutTi[4], lutTq[4];
        if (line < (int)demodLUTTi_locked.size()) {
            for (int i = 0; i < 4; ++i) {
                lutTi[i] = (double)demodLUTTi_locked[line][i];
                lutTq[i] = (double)demodLUTTq_locked[line][i];
            }
        } else {
            fusedDemodLUT(bcos, bsin, spLUT_locked, cpLUT_locked, lutTi, lutTq);
        }

        double       *Y     = componentFrame->y(line);
        float        *tiRow = demodTI_line(line);
        float        *tqRow = demodTQ_line(line);
        float        *triRow = demodTRI_line(line);
        float        *trqRow = demodTRQ_line(line);
        const double *srcClp = clpbuffer[srcBuf].pixel[line];

        // PASS 1: initialise Y from comb estimate (needed for HP-Y in Pass 2)
        for (int h = left; h < right; ++h)
            Y[h] = (double)rawLine[h] - srcClp[h];

        // PASS 2: demod raw composite -> TRI/TRQ; build preI/preQ and yI/yQ
        double dc = (double)rawLine[left];
        constexpr double DC_ALPHA = 1.0 / 64.0;

        for (int xi = 0; xi < width; ++xi) {
            const int h = left + xi;
            dc += DC_ALPHA * ((double)rawLine[h] - dc);
            const double vraw = (double)rawLine[h] - dc;
            const int ph = (h & 3);

            const double ti = tiRow ? (double)tiRow[xi] : 0.0;
            const double tq = tqRow ? (double)tqRow[xi] : 0.0;
            const double ri = vraw * lutTi[ph];
            const double rq = vraw * lutTq[ph];

            triRow[xi] = (float)ri;
            trqRow[xi] = (float)rq;

            preI[xi] = ti * effGI;
            preQ[xi] = tq * effGQ;

            const int hm = (h > left)      ? h - 1 : h;
            const int hp = (h < right - 1) ? h + 1 : h;
            const double yh = Y[h] - 0.5 * (Y[hm] + Y[hp]);
            yhp[xi] = yh;
            yI[xi] = yh * lutTi[ph];
            yQ[xi] = yh * lutTq[ph];
        }

        // Apply pre-computed affine from phaseLocked to yI/yQ only.
        // tiRow/tqRow and preI/preQ are already affine-corrected by
        // buildPhaseCorrected1D.
        if (configuration.residualVideo && T.Y_LINE_AFFINE_TRIM_ENABLE
                && line < (int)lineAffineLocked.size()) {
            const LineAffine &la = lineAffineLocked[line];
            if (la.valid) {
                for (int xi = 0; xi < width; ++xi) {
                    const double yi  = yI[xi];
                    const double yqv = yQ[xi];
                    yI[xi] = la.R[0][0]*yi + la.R[0][1]*yqv;
                    yQ[xi] = la.R[1][0]*yi + la.R[1][1]*yqv;
                }
            }
        }
    }
}

// Apply per-line FIRs to the locked demod arrays (demodTI_flat/demodTQ_flat).
// The locked path provides I and Q with bandwidth-tailored, tap matched FIRs.
void Comb::FrameBuffer::filterIQLocked()
{
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;

    // FIR setup (same parameters as in splitIQlocked)
    constexpr bool   EXP_IQ_FIR_ENABLE = true;
    constexpr int    EXP_FIR_TAPS      = 21;
    constexpr double EXP_I_CUTOFF_MHZ  = 1.5;
    constexpr double EXP_Q_CUTOFF_MHZ  = 0.67; // We backed off from spec to avoid shaving the trace

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

    const double effGI = GI_PRODUCT * configuration.gi_product;
    const double effGQ = GQ_PRODUCT * configuration.gq_product;

    if ((int)scratch_preI.size() < width) scratch_preI.resize(width, 0.0);
    if ((int)scratch_preQ.size() < width) scratch_preQ.resize(width, 0.0);
    const int extWidth = width + 2 * pad;
    if ((int)scratch_preI_ext.size() < extWidth) scratch_preI_ext.resize(extWidth, 0.0);
    if ((int)scratch_preQ_ext.size() < extWidth) scratch_preQ_ext.resize(extWidth, 0.0);

    for (int line = firstLine; line < lastLine; ++line) {
        double* Irow = componentFrame->u(line);
        double* Qrow = componentFrame->v(line);
        float*  tiRow = demodTI_line(line);
        float*  tqRow = demodTQ_line(line);

        const quint16* rawLine = rawbuffer.data() + line * videoParameters.fieldWidth;
        const double*  Yrow    = componentFrame->y(line);
        const double   bcos    = (line < (int)demodBurstCos.size()) ? (double)demodBurstCos[line] : 1.0;
        const double   bsin    = (line < (int)demodBurstSin.size()) ? (double)demodBurstSin[line] : 0.0;
        double lutTi[4], lutTq[4];
        if (line < (int)demodLUTTi_locked.size()) {
            for (int i = 0; i < 4; ++i) {
                lutTi[i] = (double)demodLUTTi_locked[line][i];
                lutTq[i] = (double)demodLUTTq_locked[line][i];
            }
        } else {
            fusedDemodLUT(bcos, bsin, spLUT_locked, cpLUT_locked, lutTi, lutTq);
        }

        // If residualColor is active, derive chroma by subtracting the final Y from composite,
        // then demodulate that residual into the locked basis  this gives chroma that is
        // exactly consistent with the Y we produced, regardless of what filtering follows.
        if (configuration.residualColor) {
            double dc = (double)rawLine[left] - Yrow[left];
            constexpr double DC_ALPHA = 1.0 / 64.0;
            for (int i = 0; i < width; ++i) {
                const int h = left + i;
                const double chromaRaw = (double)rawLine[h] - Yrow[h];
                dc += DC_ALPHA * (chromaRaw - dc);
                const double chroma = chromaRaw - dc;
                const int ph = (h & 3);
                scratch_preI[i] = (chroma * lutTi[ph]) * effGI;
                scratch_preQ[i] = (chroma * lutTq[ph]) * effGQ;
            }
        } else {
            for (int i = 0; i < width; ++i) {
                const double ti = tiRow ? (double)tiRow[i] : 0.0;
                const double tq = tqRow ? (double)tqRow[i] : 0.0;
                scratch_preI[i] = ti * effGI;
                scratch_preQ[i] = tq * effGQ;
            }
        }

        // Edge-extend once, then FIR with straight indexing (no per-tap clamps).
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

        // Apply FIRs to preI/preQ and write to I/Q
        for (int i = 0; i < width; ++i) {
            double accI = 0.0, accQ = 0.0;
            const double *cI = preIext + pad + i;
            const double *cQ = preQext + pad + i;
            for (int k = -MI; k <= MI; ++k) {
                accI += cI[k] * tapsI[k + MI];
            }
            for (int k = -MQ; k <= MQ; ++k) {
                accQ += cQ[k] * tapsQ[k + MQ];
            }
            const int h = left + i;
            Irow[h] = accI;
            Qrow[h] = accQ;
        }
    }
}

// Coherent Y rebuild (produceY)
void Comb::FrameBuffer::produceY()
{
    if (!configuration.phaseCompensation) return;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;
    if (width <= 0) return;

    // basisLockedInit guaranteed by phaseLocked in locked mode; guard retained for safety.
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

    const auto &T = configuration.tunables;
    const bool enableResidualY = T.VET_ENABLE_RESIDUAL_Y;
    const double invI = this->invIreScale;

    // Window size (must be multiple of 4)
    const int WIN  = std::max(4, (T.VET_ALIGN_WIN_SAMPLES / 4) * 4);
    const int HALF = WIN / 2;

    // Fit gating (these are deliberately conservative; tune if needed)
    const double MIN_FIT_IRE = 2.0;   // ignore near-zero chroma (ill-conditioned STT)
    const double MAX_FIT_IRE = 35.0;  // roll off very large vectors (often non-linear/clipped)
    const double SAT_TROUBLE_IRE = 18.0; // where we clamp transform to rotation-only more aggressively

    for (int line = firstLine; line < lastLine; ++line) {
        if (line >= demodLines) continue;

        const quint16* rawLine = rawbuffer.data() + line * videoParameters.fieldWidth;
        const double bcos = demodBurstCos[line];
        const double bsin = demodBurstSin[line];
        // Remod coefficients for cval_hat:
        //   c = 0.5 * ((ti*bcos + tq*bsin)*sp + (-ti*bsin + tq*bcos)*cp)
        //     = ti * 0.5*(bcos*sp - bsin*cp) + tq * 0.5*(bsin*sp + bcos*cp)
        double remodI[4], remodQ[4];
        for (int ph = 0; ph < 4; ++ph) {
            const double sp = spLUT_locked[ph];
            const double cp = cpLUT_locked[ph];
            remodI[ph] = 0.5 * (bcos * sp - bsin * cp);
            remodQ[ph] = 0.5 * (bsin * sp + bcos * cp);
        }

        double* Y = componentFrame->y(line);
        const int srcBuf = std::clamp((int)configuration.dimensions - 1, 0, 2);
        const double *clpLine = clpbuffer[srcBuf].pixel[line];

        // Baseline when residual disabled
        if (!enableResidualY) {
            for (int h = left; h < right; ++h) Y[h] = (double)rawLine[h] - clpLine[h];
            if (configuration.showMap) std::fill(w2d_frame_weight[line].begin(), w2d_frame_weight[line].end(), 0.0f);
            continue;
        }

        // Pre-FIR demod arrays
        const float* tiRow  = demodTI_line(line);
        const float* tqRow  = demodTQ_line(line);
        const float* triRow = demodTRI_line(line);
        const float* trqRow = demodTRQ_line(line);
        float* tiRowW = demodTI_line(line);
        float* tqRowW = demodTQ_line(line);

        // If any required demod rows are missing, fall back to non-residual
        if (!tiRow || !tqRow || !triRow || !trqRow) {
            for (int h = left; h < right; ++h) Y[h] = (double)rawLine[h] - clpLine[h];
            if (configuration.showMap) std::fill(w2d_frame_weight[line].begin(), w2d_frame_weight[line].end(), 0.0f);
            continue;
        }

        // Residual for diagnostics/vet
        for (int xi = 0; xi < width; ++xi) {
            const int h = left + xi;
            scratch_comp_res[xi] = ((double)rawLine[h] - clpLine[h]);
        }

        // Per-pixel outputs for the second-pass election:
        // - cHat: remodulated chroma estimate to subtract
        // - tiAdj/tqAdj: adjusted demod vectors (only written back if chosen)
        // - vetConf: vet confidence (0..1)
        if ((int)scratch_fieldGate.size() < width) scratch_fieldGate.resize(width, 1.0);
        if ((int)scratch_lateralLine.size() < width) scratch_lateralLine.resize(width, 0.0);
        double *cHat    = scratch_fieldGate.data();
        double *tiAdjLn = scratch_fieldLine.data();
        double *tqAdjLn = scratch_fieldBLine.data();
        double *vetConf = scratch_lateralLine.data();

        // ------------------------------------------------------------
        // Precompute per-sample window contributions once per line.
        // Reuse existing scratch buffers (width-sized) to avoid new allocations.
        // ------------------------------------------------------------
        // Ensure scratch buffers are large enough for width-sized repurposing.
        if ((int)scratch_preI_ext.size() < width) scratch_preI_ext.resize(width, 0.0);
        if ((int)scratch_preQ_ext.size() < width) scratch_preQ_ext.resize(width, 0.0);
        double *cSTT00 = scratch_yhp.data();
        double *cSTT01 = scratch_yI.data();
        double *cSTT11 = scratch_yQ.data();
        double *cSRT00 = scratch_preI.data();
        double *cSRT01 = scratch_preQ.data();
        double *cSRT10 = scratch_preI_ext.data();
        double *cSRT11 = scratch_preQ_ext.data();
        double *cN     = scratch_filter_temp.data();
        if ((int)scratch_filter_temp.size() < width) scratch_filter_temp.resize(width, 0.0);

        for (int xi = 0; xi < width; ++xi) {
            const double ti = (double)tiRow[xi];
            const double tq = (double)tqRow[xi];
            const double ri = (double)triRow[xi];
            const double rq = (double)trqRow[xi];

            const double magT_ire = std::hypot(ti, tq) * invI;
            const double magR_ire = std::hypot(ri, rq) * invI;

            if (magT_ire < MIN_FIT_IRE || magR_ire < MIN_FIT_IRE) {
                cSTT00[xi] = cSTT01[xi] = cSTT11[xi] = 0.0;
                cSRT00[xi] = cSRT01[xi] = cSRT10[xi] = cSRT11[xi] = 0.0;
                cN[xi] = 0.0;
                continue;
            }

            double w = 1.0;
            if (magT_ire > MAX_FIT_IRE) {
                const double t = (magT_ire - MAX_FIT_IRE) / (MAX_FIT_IRE + 1e-9);
                w = 1.0 / (1.0 + 4.0 * t * t);
            }

            cSTT00[xi] = w * ti * ti;
            cSTT01[xi] = w * ti * tq;
            cSTT11[xi] = w * tq * tq;
            cSRT00[xi] = w * ri * ti;
            cSRT01[xi] = w * ri * tq;
            cSRT10[xi] = w * rq * ti;
            cSRT11[xi] = w * rq * tq;
            cN[xi] = 1.0;
        }

        // Sliding window sums (window shifts to stay in-bounds, not shrink).
        const int winN = (width <= WIN) ? width : WIN;
        int a = 0;
        int b = winN - 1;
        double sSTT00 = 0.0, sSTT01 = 0.0, sSTT11 = 0.0;
        double sSRT00 = 0.0, sSRT01 = 0.0, sSRT10 = 0.0, sSRT11 = 0.0;
        double sN = 0.0;
        for (int xi = a; xi <= b; ++xi) {
            sSTT00 += cSTT00[xi]; sSTT01 += cSTT01[xi]; sSTT11 += cSTT11[xi];
            sSRT00 += cSRT00[xi]; sSRT01 += cSRT01[xi]; sSRT10 += cSRT10[xi]; sSRT11 += cSRT11[xi];
            sN += cN[xi];
        }

        // Process pixels
        for (int x = 0; x < width; ++x) {
            const int h = left + x;

            if (width > WIN) {
                int aWant = x - HALF;
                int bWant = x + HALF - 1;
                if (aWant < 0) { bWant += -aWant; aWant = 0; }
                if (bWant >= width) {
                    int over = bWant - (width - 1);
                    bWant -= over;
                    aWant -= over;
                    if (aWant < 0) aWant = 0;
                }
                while (a < aWant) {
                    sSTT00 -= cSTT00[a]; sSTT01 -= cSTT01[a]; sSTT11 -= cSTT11[a];
                    sSRT00 -= cSRT00[a]; sSRT01 -= cSRT01[a]; sSRT10 -= cSRT10[a]; sSRT11 -= cSRT11[a];
                    sN -= cN[a];
                    ++a;
                }
                while (a > aWant) {
                    --a;
                    sSTT00 += cSTT00[a]; sSTT01 += cSTT01[a]; sSTT11 += cSTT11[a];
                    sSRT00 += cSRT00[a]; sSRT01 += cSRT01[a]; sSRT10 += cSRT10[a]; sSRT11 += cSRT11[a];
                    sN += cN[a];
                }
                while (b < bWant) {
                    ++b;
                    sSTT00 += cSTT00[b]; sSTT01 += cSTT01[b]; sSTT11 += cSTT11[b];
                    sSRT00 += cSRT00[b]; sSRT01 += cSRT01[b]; sSRT10 += cSRT10[b]; sSRT11 += cSRT11[b];
                    sN += cN[b];
                }
                while (b > bWant) {
                    sSTT00 -= cSTT00[b]; sSTT01 -= cSTT01[b]; sSTT11 -= cSTT11[b];
                    sSRT00 -= cSRT00[b]; sSRT01 -= cSRT01[b]; sSRT10 -= cSRT10[b]; sSRT11 -= cSRT11[b];
                    sN -= cN[b];
                    --b;
                }
            }

            double STT[2][2] = {{sSTT00, sSTT01}, {sSTT01, sSTT11}};
            double SRT[2][2] = {{sSRT00, sSRT01}, {sSRT10, sSRT11}};
            const int n = (int)(sN + 0.5);

            // ------------------------------------------------------------
            // Vet: local LS alignment (phase + correlation + shear)
            //
            // IMPORTANT: Don't re-scan the WIN window here per pixel to rebuild
            // STT/SRT; we already have STT/SRT from the sliding sums above.
            // ------------------------------------------------------------
            Vet1DResult vet;
            vet.accept = true;
            vet.confidence = 1.0;
            vet.composite_bandpass = scratch_comp_res[x];

            double STTinv[2][2];
            const bool invOk = mat2_inv(STT, STTinv);

            // Vet metrics are computed from A_vet = SRT * inv(STT) when possible.
            double RmVet[2][2] = {{1,0},{0,1}};
            double UVet[2][2]  = {{1,0},{0,1}};
            if (!invOk || n < 8) {
                vet.accept = false;
                vet.confidence = 0.0;
            } else {
                double Avet[2][2];
                mat2_mul(SRT, STTinv, Avet);
                polar_decompose_2x2(Avet, RmVet, UVet);

                const double phase = std::atan2(RmVet[1][0], RmVet[0][0]); // radians
                double l1, l2, V_[2][2];
                eig2_sym(UVet, l1, l2, V_);
                const double s1 = std::max(0.0, l1), s2 = std::max(0.0, l2);
                const double g  = 0.5 * (s1 + s2);
                const double shear = (g > 1e-12) ? std::fabs(s1 - s2) / g : 0.0;

                const double srt00 = SRT[0][0], srt01 = SRT[0][1], srt10 = SRT[1][0], srt11 = SRT[1][1];
                const double numRho = std::sqrt(srt00*srt00 + srt01*srt01 + srt10*srt10 + srt11*srt11);
                const double denRho = std::max(1e-9, STT[0][0] + STT[1][1]);
                const double rho = numRho / denRho;

                const double pMaxVet = T.VET_ALIGN_PHASE_MAX_DEG * M_PI / 180.0;
                if (std::fabs(phase) > pMaxVet || rho < T.VET_ALIGN_MIN_RHO || shear > T.VET_ALIGN_MAX_SHEAR) {
                    vet.accept = false;
                }

                double c_phase = 1.0 - std::min(1.0, std::fabs(phase) / (pMaxVet + 1e-12));
                double c_shear = 1.0 - std::min(1.0, shear / (T.VET_ALIGN_MAX_SHEAR + 1e-12));
                double c = 0.5 * std::max(0.0, std::min(1.0, rho)) + 0.25 * c_phase + 0.25 * c_shear;
                if (c < 0.0) c = 0.0; else if (c > 1.0) c = 1.0;
                vet.confidence = c;
            }

            // ------------------------------------------------------------
            // Solve local affine (optional): reuse the same polar decomposition
            // computed for the vet when we actually apply the affine.
            // ------------------------------------------------------------
            double Rm[2][2] = {{1,0},{0,1}};
            double U[2][2]  = {{1,0},{0,1}};
            if (T.Y_LOCAL_AFFINE_ENABLE && n >= 16 && invOk) {
                Rm[0][0] = RmVet[0][0]; Rm[0][1] = RmVet[0][1];
                Rm[1][0] = RmVet[1][0]; Rm[1][1] = RmVet[1][1];
                U[0][0]  = UVet[0][0];  U[0][1]  = UVet[0][1];
                U[1][0]  = UVet[1][0];  U[1][1]  = UVet[1][1];
            }

            const double pMax = T.Y_LOCAL_MAX_PHASE_DEG * M_PI / 180.0;

            // Saturated trouble regions: keep transform conservative (rotation-only / no shear).
            const double ti0 = (double)tiRow[x];
            const double tq0 = (double)tqRow[x];
            const double magX_ire = std::hypot(ti0, tq0) * invI;
            const bool satTrouble = (magX_ire > SAT_TROUBLE_IRE);

            clamp_rotation_gain_shear(Rm, U, pMax,
                                      /*allowGain=*/!satTrouble,
                                      T.Y_LOCAL_GAIN_MIN, T.Y_LOCAL_GAIN_MAX,
                                      satTrouble ? 0.0 : T.Y_LOCAL_MAX_SHEAR);

            const int idx = (h & 3);
            const double sp = spLUT_locked[idx], cp = cpLUT_locked[idx];

            // Apply clamped transform to (ti0,tq0) for estimate only
            // NOTE: we apply only Rm here 
            const double ti_adj = Rm[0][0]*ti0 + Rm[0][1]*tq0;
            const double tq_adj = Rm[1][0]*ti0 + Rm[1][1]*tq0;

            const double cval_hat = ti_adj * remodI[idx] + tq_adj * remodQ[idx];

            // Store for the second-pass decision (protect luma against chroma).
            cHat[x] = cval_hat;
            tiAdjLn[x] = ti_adj;
            tqAdjLn[x] = tq_adj;
            vetConf[x] = vet.confidence;
        }

        // ------------------------------------------------------------
        // Second pass: produce residual Y by controlling the subtraction itself.
        //
        // The residual path is the incumbent: alpha=1 subtracts the remodulated
        // chroma estimate as-is. In uncertain cases, use the chroma profile of
        // cHat and the luma profile of the resulting Y to make a bounded alpha
        // correction that minimizes 4fSC-like residue in Y. This keeps the
        // decision at the waveform separation point instead of treating residual
        // Y as an optional replacement candidate.
        // ------------------------------------------------------------
        const bool do3D = (configuration.residualVideo3D && prevFrameForVet && nextFrameForVet);

        const double MIN_ALPHA = 0.75;
        const double MAX_ALPHA = 1.25;
        const double MIN_SUB_CHROMA_IRE = 2.0;

        if (width < 4) {
            // Degenerate: no 4-sample group; use full residual subtraction.
            for (int x = 0; x < width; ++x) {
                const int h = left + x;
                const double yOut = (double)rawLine[h] - cHat[x];
                Y[h] = do3D ? getBestY(line, h, yOut, *prevFrameForVet, *nextFrameForVet) : yOut;
                if (configuration.showMap) w2d_frame_weight[line][x] = (float)vetConf[x];
                tiRowW[x] = (float)tiAdjLn[x];
                tqRowW[x] = (float)tqAdjLn[x];
            }
            continue;
        }

        for (int gp = 0; gp < width; gp += 4) {
            int p = gp;
            if (p > width - 4) p = width - 4;

            const double r0 = (double)rawLine[left + p + 0];
            const double r1 = (double)rawLine[left + p + 1];
            const double r2 = (double)rawLine[left + p + 2];
            const double r3 = (double)rawLine[left + p + 3];

            const double c0 = cHat[p + 0];
            const double c1 = cHat[p + 1];
            const double c2 = cHat[p + 2];
            const double c3 = cHat[p + 3];

            // Phase-agnostic 4fSC vectors: I ~= s1-s3, Q ~= s2-s0.
            const double rawI = r1 - r3;
            const double rawQ = r2 - r0;
            const double subI = c1 - c3;
            const double subQ = c2 - c0;
            const double subEnergy = subI * subI + subQ * subQ;
            const double subMagIRE = std::sqrt(subEnergy) * invI;

            double alpha = 1.0;
            if (T.VET_Y_CHROMA_LIKE_WEIGHT > 0.0 && subMagIRE >= MIN_SUB_CHROMA_IRE) {
                // Least-squares alpha that minimizes 4fSC energy in raw - alpha*cHat.
                const double alphaFit = std::clamp((rawI * subI + rawQ * subQ) / (subEnergy + 1e-12),
                                                   MIN_ALPHA, MAX_ALPHA);
                const double conf = std::clamp(0.25 * (vetConf[p + 0] + vetConf[p + 1] +
                                                       vetConf[p + 2] + vetConf[p + 3]),
                                               0.0, 1.0);
                const double profileWeight = T.VET_Y_CHROMA_LIKE_WEIGHT * (1.0 - conf);
                alpha = 1.0 + profileWeight * (alphaFit - 1.0);
            }

            for (int i = 0; i < 4; ++i) {
                const int x = p + i;
                if (x < 0 || x >= width) continue;
                const int h = left + x;

                const double yOut = (double)rawLine[h] - alpha * cHat[x];
                Y[h] = do3D ? getBestY(line, h, yOut, *prevFrameForVet, *nextFrameForVet) : yOut;

                if (configuration.showMap) w2d_frame_weight[line][x] = (float)alpha;
                tiRowW[x] = (float)(alpha * tiAdjLn[x]);
                tqRowW[x] = (float)(alpha * tqAdjLn[x]);
            }
        }
    }
}

// Bucket-path Y reconstruction: subtracts the chroma estimate (reconstructed
// from the I and Q buckets) from the raw composite to yield luma. The chroma
// remodulation reverses the bucket demod  routing each sample through the
// same switch on (h & 3) with sign inversion  and applies the per-line
// subcarrier polarity from getLinePhase. Only called in bucket mode
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

        bool linePhase = getLinePhase(line);
        for (int h = left; h < right; ++h) {
            double comp = 0.0;
            switch (h & 3) {
                case 0: comp = -Q[h]; break;
                case 1: comp =  I[h]; break;
                case 2: comp =  Q[h]; break;
                case 3: comp = -I[h]; break;
            }
            if (!linePhase) comp = -comp;
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
