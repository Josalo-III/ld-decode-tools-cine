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
}

namespace {
    // Fractional 4fsc sample offset (in samples). Keep small; start at 0.0.
    // Adjust in range 0.05..0.15 if vectorscope lines bow.
    constexpr double CAL_EPS_SAMPLES = -0.07;

    // Tiny global LO trim (degrees). Negative usually counteracts a slight green bias.
    constexpr double CAL_LO_ROT_DEG  = 0.0;

    // Compute basis mix once per function call (shared by splitIQlocked and produceY)
    inline void basisCoeffs(double& Ce, double& Se) {
        const double K = 0.5 * M_PI; // 4fsc per-sample step = π/2
        Ce = std::cos(K * CAL_EPS_SAMPLES);
        Se = std::sin(K * CAL_EPS_SAMPLES);
    }

    // Basis projection for sample h shifted by ε (CAL_EPS_SAMPLES).
    // sp = sin((h + ε) · π/2),  cp = cos((h + ε) · π/2)
    static inline void shiftedBasis(int h, double Ce, double Se, double& sp, double& cp) {
        const int idx = (h & 3);
        const double s4 = sin4fsc(idx);
        const double c4 = cos4fsc(idx);
        sp = Ce * s4 + Se * c4;
        cp = Ce * c4 - Se * s4;
    }

    // Fuse burst rotation into the 4-phase locked basis:
    //   ti = c * 2 * (sp*bcos - cp*bsin)
    //   tq = c * 2 * (sp*bsin + cp*bcos)
    // (where (bcos,bsin) is the per-line burst phasor).
    static inline void fusedDemodLUT(double bcos, double bsin,
                                     const double spLUT[4], const double cpLUT[4],
                                     double outTi[4], double outTq[4])
    {
        for (int i = 0; i < 4; ++i) {
            const double sp = spLUT[i];
            const double cp = cpLUT[i];
            outTi[i] = 2.0 * (sp * bcos - cp * bsin);
            outTq[i] = 2.0 * (sp * bsin + cp * bcos);
        }
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
// Polar-decompose a 2x2 affine matrix A = RU into a rotation R and symmetric U,
// then clamp R to a maximum phase rotation, clamp U's shear metric, and optionally
// clamp the gain (mean singular value). The clamped gain is folded into R so callers
// can apply a single matrix. Used throughout the locked path to keep per-line and
// per-window affine corrections from diverging on noisy or saturated content.
static inline void clamp_rotation_gain_shear(double R[2][2], double U[2][2],
                                             double phaseMaxRad, bool allowGain,
                                             double gMin, double gMax, double shearMax) {
    // Extract phase from R
    double phase = std::atan2(R[1][0], R[0][0]);
    if (std::fabs(phase) > phaseMaxRad) {
        double p = (phase < 0.0 ? -phaseMaxRad : phaseMaxRad);
        double c = std::cos(p), s = std::sin(p);
        R[0][0]=c; R[0][1]=-s; R[1][0]=s; R[1][1]=c;
    }
    // From U (symmetric), extract approximate gain (mean singular value) and shear metric
    // Use eigenvalues of U: s1,s2 (singular values of As symmetric part)
    double l1,l2,V[2][2]; eig2_sym(U,l1,l2,V);
    double s1 = std::max(0.0, l1), s2 = std::max(0.0, l2);
    double g  = 0.5 * (s1 + s2);
    double shear = (g > 1e-12) ? std::fabs(s1 - s2) / g : 0.0;
    // Clamp shear by pulling U toward isotropy
    if (shear > shearMax && (s1 > 0.0 || s2 > 0.0)) {
        double target = g * shearMax;
        double avg = 0.5 * (s1 + s2);
        s1 = avg + 0.5 * target;
        s2 = avg - 0.5 * target;
        // Rebuild U = V diag(s1,s2) V^T
        double VD[2][2] = { {V[0][0]*s1, V[0][1]*s2}, {V[1][0]*s1, V[1][1]*s2} };
        U[0][0] = VD[0][0]*V[0][0] + VD[0][1]*V[0][1];
        U[0][1] = VD[0][0]*V[1][0] + VD[0][1]*V[1][1];
        U[1][0] = VD[1][0]*V[0][0] + VD[1][1]*V[0][1];
        U[1][1] = VD[1][0]*V[1][0] + VD[1][1]*V[1][1];
        g = 0.5 * (s1 + s2);
    }
    // Clamp gain
    if (!allowGain) g = 1.0;
    else            g = std::min(std::max(g, gMin), gMax);
    // Fold gain into R (so we can apply a single 2x2 matrix to signals)
    R[0][0] *= g; R[0][1] *= g; R[1][0] *= g; R[1][1] *= g;
}

// Render a single character from a minimal 57 bitmap font into a FrameCanvas.
// Supports pulldown film letters, '?' (unknown), and '/' (boundary marker).
// scale controls pixel block size for visibility at different output resolutions.
static void drawChar(FrameCanvas &canvas, int x, int y, char ch, FrameCanvas::Colour col, int scale) {
    // Simple 5x7 font map for A-D, ?, and /
    static const unsigned char font[][7] = {
        {0x04,0x0A,0x11,0x11,0x1F,0x11,0x11}, // A (0)
        {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}, // B (1)
        {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}, // C (2)
        {0x1E,0x11,0x11,0x11,0x11,0x11,0x1E}, // D (3)
        {0x0E,0x11,0x01,0x02,0x04,0x00,0x04}, // ? (5)
        {0x01,0x02,0x02,0x04,0x04,0x08,0x10}  // / (6)
    };
    
    int idx = 5; // default to ?
    if (ch >= 'A' && ch <= 'E') idx = ch - 'A';
    else if (ch == '/') idx = 6;
    else if (ch >= '0' && ch <= '9') {
    }
    
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
            // 2) Phase-corrected 1D demod -> demodTI_flat/TQ_flat (via buildPhaseCorrected1D inside split2D)
            // 3) Full 2D scoring (Field A, Field B / Frame via scoreFieldVsFrame) -> clpbuffer[1]
            // 4) Demod raw composite -> TRI/TRQ; build preI/preQ and yI/yQ for residual Y
            current->splitIQlocked();
            // 5) Chroma NR on I/Q
            current->doCNR();
            // 6) Coherent Y rebuild from affine-corrected demod
            current->produceY();
            // 7) Chroma FIR bandwidth limiting in locked space
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
            //   -2 cadenceId -> "i2" (confirmed interlaced)
            //   -3 cadenceId -> "p3" (confirmed progressive)
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
        // 2D score blending visualization (only written when showMap is true)
        w2d_frame_weight.assign(lines, std::vector<float>(width, 0.0f));
        w2d_fieldA_gate.assign(lines, std::vector<double>(width, 1.0f));
        fvfMetrics.assign(lines, std::vector<FvfModelMetrics>(width));
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
        scratch_vdis_flag.assign(width, 0);    
    }
    vdisMask.assign(lines, std::vector<char>(width, 0));
    locked1DSource.assign(lines, std::vector<double>(width, 0.0));
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
    if ((int)vdisMask.size() < last) vdisMask.resize(last);
    const int width = videoParameters.activeVideoEnd - videoParameters.activeVideoStart;
    for (int line = first; line < last; ++line) {
        auto &row = vdisMask[line];
        if ((int)row.size() < width) row.assign(width, 0);
        else std::fill(row.begin(), row.end(), 0);
    }
}

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
            const double bcos      = (double)demodBurstCos[line];
            const double bsin      = (double)demodBurstSin[line];
            float *triRow          = demodTRI_line(line);
            float *trqRow          = demodTRQ_line(line);
            double lutTi[4], lutTq[4];
            fusedDemodLUT(bcos, bsin, spLUT_locked, cpLUT_locked, lutTi, lutTq);

            double dc = (double)rawLine[left];
            constexpr double DC_ALPHA = 1.0 / 64.0;

            for (int xi = 0; xi < width; ++xi) {
                const int h      = left + xi;
                dc += DC_ALPHA * ((double)rawLine[h] - dc);
                const double vraw = (double)rawLine[h] - dc;
                const int ph = (h & 3);
                triRow[xi] = (float)(vraw * lutTi[ph]);
                trqRow[xi] = (float)(vraw * lutTq[ph]);
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

            for (int xi = 0; xi < width; ++xi) {
                const int h  = left + xi;
                const double ri = (double)triRow[xi];
                const double rq = (double)trqRow[xi];

                // Windowed amplitude and residual from TRI/TRQ neighbours
                int a = xi - HALF, b = xi + HALF - 1;
                if (a < 0)      { b += -a;              a = 0; }
                if (b >= width) { int ov = b-(width-1); b -= ov; a -= ov; if (a < 0) a = 0; }
                const int n = b - a + 1;

                double ampEst = 0.0, resAmp = 0.0;
                for (int k = a; k <= b; ++k) {
                    const int hk     = left + k;
                    const double spk = spLUT_locked[hk & 3];
                    const double cpk = cpLUT_locked[hk & 3];
                    const double rik = (double)triRow[k];
                    const double rqk = (double)trqRow[k];
                    const double mag_k = std::hypot(rik, rqk);
                    ampEst += mag_k;

                    if (mag_k > 1e-9) {
                        const double fitted_k = 0.5 * ((rik * bcos + rqk * bsin) * spk
                                                      + (-rik * bsin + rqk * bcos) * cpk);
                        const double corr_k   = (double)rawLine[hk] - fitted_k;
                        const double rsk      = corr_k * spk * 2.0;
                        const double rck      = corr_k * cpk * 2.0;
                        resAmp += std::hypot(rsk * bcos - rck * bsin,
                                             rsk * bsin + rck * bcos);
                    }
                }
                ampEst /= n;
                resAmp /= n;

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

// Demodulates clpbuffer[0] into demodTI_flat/demodTQ_flat using the locked basis,
// then remods back to phase-normalised composite in clpbuffer[1]. Populates
// demodTI/TQ for downstream consumers (computeFrameIQLine, filterIQLocked,
// scoreFieldVsFrame) and writes the phase-normalised chroma to clpbuffer[1]
// as the reference used by split2D's Field/Frame scoring.
void Comb::FrameBuffer::buildPhaseCorrected1D()
{
    const int first  = videoParameters.firstActiveFrameLine;
    const int last   = videoParameters.lastActiveFrameLine;
    const int left   = videoParameters.activeVideoStart;
    const int right  = videoParameters.activeVideoEnd;
    const int width  = right - left;

    if (width <= 0 || first >= last) return;

    for (int line = first; line < last; ++line) {
        const double *src = clpbuffer[0].pixel[line];
        double       *dst = clpbuffer[1].pixel[line];

        const double bcos = (line < (int)demodBurstCos.size()) ? (double)demodBurstCos[line] : 1.0;
        const double bsin = (line < (int)demodBurstSin.size()) ? (double)demodBurstSin[line] : 0.0;
        double lutTi[4], lutTq[4];
        fusedDemodLUT(bcos, bsin, spLUT_locked, cpLUT_locked, lutTi, lutTq);

        float *tiRow = demodTI_line(line);
        float *tqRow = demodTQ_line(line);

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
        };

        for (int xi = 0; xi < width; ++xi) {
            const int h    = left + xi;
            const double c  = src[h];

            const int ph = (h & 3);
            double ti = c * lutTi[ph];
            double tq = c * lutTq[ph];

            double intakeNyquistRiskIRE = 0.0;
            double lumaIncursionRiskIRE = 0.0;
            double residualFitErrorIRE = 0.0;

            if (line >= 0 && line < (int)fvfMetrics.size() &&
                xi < (int)fvfMetrics[line].size())
            {
                const double fine = std::fabs(sampleSrc(xi) -
                                              0.5 * (sampleSrc(xi - 1) + sampleSrc(xi + 1))) * invIreScale;
                const double mid = std::fabs(sampleSrc(xi) -
                                             0.5 * (sampleSrc(xi - 2) + sampleSrc(xi + 2))) * invIreScale;
                const double coarse = std::fabs(sampleSrc(xi) -
                                                0.5 * (sampleSrc(xi - 4) + sampleSrc(xi + 4))) * invIreScale;
                const double denom = fine + mid + coarse + 1e-9;
                const double fineFrac = fine / denom;
                const double nonFineFrac = std::max(mid, coarse) / denom;
                const double dominance = std::clamp((fineFrac - nonFineFrac - 0.15) / 0.35, 0.0, 1.0);
                intakeNyquistRiskIRE = fine * dominance;

                double tiLm1 = 0.0, tqLm1 = 0.0, tiLp1 = 0.0, tqLp1 = 0.0;
                double tiLm2 = 0.0, tqLm2 = 0.0, tiLp2 = 0.0, tqLp2 = 0.0;
                sampleIQ(xi - 1, tiLm1, tqLm1);
                sampleIQ(xi + 1, tiLp1, tqLp1);
                sampleIQ(xi - 2, tiLm2, tqLm2);
                sampleIQ(xi + 2, tiLp2, tqLp2);

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

                fvfMetrics[line][xi].intakeNyquistRiskIRE = intakeNyquistRiskIRE;
                fvfMetrics[line][xi].lumaIncursionRiskIRE = lumaIncursionRiskIRE;
                fvfMetrics[line][xi].residualFitErrorIRE = residualFitErrorIRE;
            }

            tiRow[xi] = (float)ti;
            tqRow[xi] = (float)tq;

            // Remod to phase-normalised composite for clpbuffer[1]
            if (configuration.lockedRemodTo4fsc) {
                // Remodulate onto the exact 4fsc sample grid (h&3). This deliberately
                // drops the fractional-basis shift (CAL_EPS_SAMPLES) for the remod only,
                // to preserve perfect 4-sample periodicity for downstream comb stages.
                const double sp4 = sin4fsc(ph);
                const double cp4 = cos4fsc(ph);
                const double rsin =  ti * bcos + tq * bsin;
                const double rcos = -ti * bsin + tq * bcos;
                dst[h] = 0.5 * (rsin * sp4 + rcos * cp4);
            } else {
                const double sp = spLUT_locked[ph];
                const double cp = cpLUT_locked[ph];
                const double rsin =  ti * bcos + tq * bsin;
                const double rcos = -ti * bsin + tq * bcos;
                dst[h] = 0.5 * (rsin * sp + rcos * cp);
            }
        }
    }
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

    // FieldB scalar raster must already be in simpleField2D[line]
    if ((int)simpleField2D.size() <= line) return;
    const auto &fieldLine = simpleField2D[line];
    if ((int)fieldLine.size() < width) return;

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
static void consolidateVDISRegions(
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
        if (ln < 0 || ln >= (int)simpleField2D.size()) return false;
        if ((int)simpleField2D[ln].size() < width) return false;
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

        const double c = simpleField2D[ln][x];

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



// Triangular safety vs 1D
// Given a candidate value C, the 1D baseline L1, and an IRE threshold,
// returns true if C is considered "near" 1D in a triangular sense.
// Intended to be used in VDIS regions as a gate for Field A & B/Frame.
static inline bool fvf_is_tri_safe(double candVal,
                                   double L1,
                                   double invIreScale,
                                   double triSafeIre)
{
    const double dCand1D_ire = std::fabs(candVal - L1) * invIreScale;
    return (dCand1D_ire <= triSafeIre);
}

// Stronger 4fsc cancellation than the previous [1,2,1] using only even offsets.
// This reduces per-pixel (1fsc) winner flicker on saturated chroma.
static inline double getNotchLumaEven2(const double* arr, int rel, int width) {
    if (!arr || width <= 0) return 0.0;

    // Need rel2
    if (rel < 2) rel = 2;
    if (rel > width - 3) rel = width - 3;

    // True 2-tap comb notch at 4fsc: average of 2
    return 0.5 * (arr[rel - 2] + arr[rel + 2]);
}

static inline double getNotchLumaEven2Vec(const std::vector<double>& vec, int rel) {
    const int width = (int)vec.size();
    return (width > 0) ? getNotchLumaEven2(vec.data(), rel, width) : 0.0;
}
// 2D comb scorer
// Paradigm: sample @  +/-1 (interfield) = Frame. Sample @ +/- 2 (intra) = Field
// Pit the Frame against 2 Fields in either blend or election w/image shaping
// Model: Field - when cadenceId ! >/=0 || -3 (film or 29.97p, respectively)
// We favor the frame estimates slightly when they agree with Field, but they
// are excluded for strong deviation, so fields that diverge from their
// in-frame compliments can remain distinct.
// Model: Frame - we favor Frame and reward/punish fields for fidelity to the model.
void Comb::FrameBuffer::scoreFieldVsFrame(
    int line,
    const double *fieldA,
    const double *fieldB,
    const std::vector<double> &frameB2,
    double *outMixed,
    bool writeWeights,
    const double *lateral1D,
    const std::vector<std::complex<double>> *frameIQ)
{
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;
    if (width <= 0) return;
    if (!fieldA || !fieldB || (int)frameB2.size() < width || !outMixed) return;
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

    std::vector<int>    winner(width, 1);
    std::vector<double> outVal(width, 0.0);
    std::vector<float>  outShade(width, 0.35f);
    std::vector<double> diffFVF(width, 0.0);
    std::vector<double> satMap(width, 0.0);

    const double SAT_FALLBACK_START = 6.0;
    const double SAT_FALLBACK_FULL  = 20.0;
    double prev_interfield_luma_ire = 0.0;

    // Core Logic of Field Vs Frame
    // when the footage is progressive we prefer interfield comb
    bool useFrameModel = (cadenceId >= 0 || cadenceId == -3);
    bool localUseFrameModel = useFrameModel;

    for (int rel = 0; rel < width; ++rel) {
        double FA = fieldA[rel];
        double FB = fieldB[rel];
        double FR = frameB2[rel];
        double L1 = sample1D(rel);

        double satFR_demod = 0.0;
        if (frameIQ && rel < (int)frameIQ->size()) {
            std::complex<double> z = (*frameIQ)[rel];
            satFR_demod = std::abs(z);
        } else {
            satFR_demod = std::fabs(FR);
        }

        double lumFA = getNotchLuma(fieldA, rel);
        double lumFB = getNotchLuma(fieldB, rel);
        double lumFR = getNotchLumaVec(frameB2, rel);

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
	        double targetModel = localUseFrameModel ? FR : FA;

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

        double vIRE = vertContrastIRE(rel);
        double hIRE = horizEdgeIRE(rel);

        const double TRI_SAFE_IRE = 3.0;
        bool safeA = fvf_is_tri_safe(FA, L1, invI, TRI_SAFE_IRE);
        bool safeB = fvf_is_tri_safe(FB, L1, invI, TRI_SAFE_IRE);
        bool safeR = fvf_is_tri_safe(FR, L1, invI, TRI_SAFE_IRE);

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
        double scoreR = std::numeric_limits<double>::quiet_NaN();

        // --- NEW: Force fallback BEFORE the complex scoring happens ---
        if (chromaMagIRE > SAT_FALLBACK_START) {
            if (localUseFrameModel && !managementVeto && b2VertCoherent) {
                idx   = 2;
                val   = FR;
                shade = 0.8f;
            } else {
                idx   = 0; 
                val   = FA;
                shade = 0.25f;
            }
            // By doing this here, we skip the potential for downstream overrides.
        } else if (vdisHard) {
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

            consider(0, FA, safeA, 0.25f);
            consider(1, FB, safeB, 0.35f);
            consider(2, FR, safeR, 0.8f);

            idx   = bestIdx;
            val   = bestVal;
            shade = bestSh;
        }
        else {
            double devA = 0.0, devB = 0.0, devR = 0.0;

            if (T.FVF_SHAPE_STRENGTH > 0.0) {
                double m_c = targetModel;
                auto getM = [&](int r) {
                    if (localUseFrameModel)
                        return frameB2[std::clamp(r, 0, width - 1)];
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
                double FR_l = frameB2[std::clamp(rel - 1, 0, width - 1)];
                double FR_r = frameB2[std::clamp(rel + 1, 0, width - 1)];

                devA += getShapeScore(FA, FA_l, FA_r) * T.FVF_SHAPE_STRENGTH;
                devB += getShapeScore(FB, FB_l, FB_r) * T.FVF_SHAPE_STRENGTH;
                devR += getShapeScore(FR, FR_l, FR_r) * T.FVF_SHAPE_STRENGTH;
            }

            double satScale = std::clamp((chromaMagIRE - 2.0) / 8.0, 0.0, 1.0);

            double errA_notch = std::fabs(lumFA);
            double errB_notch = std::fabs(lumFB);
            double errR_notch = std::fabs(lumFR);

            scoreA = (1.0 - satScale) * devA + satScale * errA_notch;
            scoreB = (1.0 - satScale) * devB + satScale * errB_notch;
            scoreR = (1.0 - satScale) * devR + satScale * errR_notch;

            // ------------------------------------------------------------
            // A cleanup + conditional "B keeps them honest"
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

            double wantB = (1.0 - gA) + 1.5 * gateAltA;
            wantB = std::clamp(wantB, 0.0, 1.0);

            const double W_B_HELP = 0.18;
            scoreB *= (1.0 - W_B_HELP * wantB);

            // ------------------------------------------------------------
            // Model-aware regime scoring.
            // Progressive protects Frame and scores fields by their deviation
            // from the frame model. Interlace treats Field A as the model,
            // lets A/B compete, and only gives Frame a small bonus when it is
            // very close to the field model.
            // ------------------------------------------------------------
            if (localUseFrameModel) {
                scoreA += T.FVF_MODEL_PRIMARY_WEIGHT * diff_candA_ire;
                scoreB += T.FVF_MODEL_PRIMARY_WEIGHT * diff_candB_ire;
                if (!managementVeto && b2VertCoherent) {
                    scoreR *= T.FRAME_MODEL_BIAS;
                }
            } else {
                const double closeFrameBonus = std::clamp(
                    1.0 - (diff_candA_ire / std::max(1e-9, T.FVF_SMALL_DIFF_IRE)),
                    0.0, 1.0);
                scoreA *= T.FIELD_MODEL_BIAS;
                scoreR += T.FVF_MODEL_PRIMARY_WEIGHT * diff_candA_ire;
                scoreR *= T.FRAME_IN_INTERLACE_PENALTY;
                scoreR *= (1.0 - 0.08 * closeFrameBonus);
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
                const double FIELD_A_FINE_PENALTY   = 0.10;
                const double FIELD_B_FINE_PENALTY   = 0.05;
                const double FIELD_SWITCH_STRENGTH  = 0.10;

                const bool fineDominant = (fineFrac > (midFrac + coarseFrac) + 0.10);

                double frameBonus = frameScaleBiasStrength * fineFrac;
                frameBonus *= (1.0 - FRAME_COARSE_CLAMP * coarseFrac);
                scoreR *= (1.0 - frameBonus);

                if (fineDominant) {
                    scoreA *= (1.0 + FIELD_A_FINE_PENALTY * fineFrac);
                    scoreB *= (1.0 + FIELD_B_FINE_PENALTY * fineFrac);
                } else {
                    const double bias = std::clamp(coarseFrac - midFrac, -1.0, 1.0);
                    scoreA *= (1.0 - FIELD_SWITCH_STRENGTH * bias);
                    scoreB *= (1.0 + FIELD_SWITCH_STRENGTH * bias);
                }
            }

            // --- cross-domain neighbor estimate using 2 plus a small 1 term ---
            if (!vdisHard &&
                hIRE < T.NEIGHBOR_EST_EDGE_MAX_IRE &&
                diff_stack_ire < T.NEIGHBOR_EST_FVF_MAX_IRE &&
                chromaMagIRE < T.NEIGHBOR_EST_SAT_MAX_IRE)
            {
                auto median3 = [&](double a, double b, double c)->double {
                    double e0 = a, e1 = b, e2 = c;
                    if (e0 > e1) std::swap(e0, e1);
                    if (e1 > e2) std::swap(e1, e2);
                    if (e0 > e1) std::swap(e0, e1);
                    return e1;
                };

                int r_m2 = std::max(0,         rel - 2);
                int r_p2 = std::min(width - 1, rel + 2);

                double estA2 = 0.5 * (fieldA[r_m2]  + fieldA[r_p2]);
                double estB2 = 0.5 * (fieldB[r_m2]  + fieldB[r_p2]);
                double estF2 = 0.5 * (frameB2[r_m2] + frameB2[r_p2]);

                double E2 = median3(estA2, estB2, estF2);

                bool allowPm1 = true;
                {
                    int r_m1 = std::max(0,         rel - 1);
                    int r_p1 = std::min(width - 1, rel + 1);
                    double altF = std::fabs(frameB2[r_m1] - frameB2[r_p1]) * invI;
                    if (altF > 6.0) allowPm1 = false;
                }

                double E = E2;
                if (allowPm1) {
                    int r_m1 = std::max(0,         rel - 1);
                    int r_p1 = std::min(width - 1, rel + 1);

                    double estA1 = 0.5 * (fieldA[r_m1]  + fieldA[r_p1]);
                    double estB1 = 0.5 * (fieldB[r_m1]  + fieldB[r_p1]);
                    double estF1 = 0.5 * (frameB2[r_m1] + frameB2[r_p1]);

                    double E1 = median3(estA1, estB1, estF1);

                    const double K_PM1 = 0.5;
                    E = (E2 + K_PM1 * E1) / (1.0 + K_PM1);
                }

                double dA = std::fabs(FA - E) * invI;
                double dB = std::fabs(FB - E) * invI;
                double dR = std::fabs(FR - E) * invI;

                const double W_NEIGH = T.NEIGHBOR_EST_WEIGHT;
                scoreA += W_NEIGH * dA;
                scoreB += W_NEIGH * dB;
                scoreR += W_NEIGH * dR;
            }

            // When the fields disagree in interlace, use 1D as a soft reality
            // check to favor the field that is less likely to be alternating.
            if (!localUseFrameModel) {
                double ab_div_ire = std::fabs(lumFA - lumFB) * invI;
                if (ab_div_ire > FIELD_DISAGREE_IRE) {
                    double dA1 = std::fabs(lumFA - L1) * invI;
                    double dB1 = std::fabs(lumFB - L1) * invI;
                    double realityBias = std::min(std::fabs(dA1 - dB1), 4.0);
                    double biasScale = 0.08 * realityBias;
                    if (dA1 + T.ONE_D_NEAR_THRESH_IRE < dB1) {
                        scoreA *= (1.0 - biasScale);
                        scoreB *= (1.0 + biasScale);
                    } else if (dB1 + T.ONE_D_NEAR_THRESH_IRE < dA1) {
                        scoreB *= (1.0 - biasScale);
                        scoreA *= (1.0 + biasScale);
                    }
                }
            }

            auto pickCandidate = [&](int candIdx, double candVal, float candShade) {
                if (vdisSoft) {
                    if (candIdx == 0 && !safeA) return;
                    if (candIdx == 1 && !safeB) return;
                    if (candIdx == 2 && !safeR) return;
                }
                idx   = candIdx;
                val   = candVal;
                shade = candShade;
            };

            if (hIRE > HEDGE_THRESH_IRE && diff_stack_ire > 5.0) {
                double dF1 = std::fabs(lumFR - L1) * invI;
                if (dF1 <= 3.5 && diff_cand_ire <= 5.0 && !frameInsane)
                    pickCandidate(2, FR, 0.75f);
                else {
                    if (scoreA < scoreB) pickCandidate(0, FA, 0.25f);
                    else                 pickCandidate(1, FB, 0.35f);
                }
            } else if (chromaMagIRE > CHROMA_STRONG_IRE && vIRE > VERT_THRESH_IRE) {
                // Strong chroma with vertical contrast indicates per-line alternation
                // that Frame is well-suited to suppress. Bypass interfield gate here.
                if (!frameInsane)
                    pickCandidate(2, FR, 0.8f);
                else {
                    if (scoreA <= scoreB) pickCandidate(0, FA, 0.25f);
                    else                  pickCandidate(1, FB, 0.35f);
                }
            } else {
                if (b2VertCoherent)
                    pickCandidate(2, FR, 0.8f);
                else if (scoreR + 1e-12 < scoreA * 0.85 &&
                         scoreR + 1e-12 < scoreB * 0.85)
                    pickCandidate(2, FR, 0.8f);
                else if (scoreA < scoreB * 0.8)
                    pickCandidate(0, FA, 0.25f);
                else {
                    double dFL = std::fabs(lumFB - L1) * invI;
                    double dRL = std::fabs(lumFR - L1) * invI;
                    if (!frameInsane && dRL + 1.0 < dFL)
                        pickCandidate(2, FR, 0.75f);
                    else
                        pickCandidate(1, FB, 0.35f);
                }
            }


            if (diff_fvf_ire < FVF_SMALL_DIFF_IRE) {
                if ((!vdisSoft || safeR) && (idx == 0 || idx == 1))
                    pickCandidate(2, FR, 0.8f);
            }

            // Subtle hysteresis (switch veto) in soft regions
            if (rel > 0) {
                const int prevIdx = winner[rel - 1];

                if (prevIdx >= 0 && prevIdx <= 2 && idx != prevIdx) {

                    const bool hystOk =
                        !vdisHard &&
                        (chromaMagIRE <= SAT_FALLBACK_START) &&
                        !(hIRE > HEDGE_THRESH_IRE && diff_stack_ire > 5.0) &&
                        !((chromaMagIRE > CHROMA_STRONG_IRE) && (vIRE > VERT_THRESH_IRE));

                    if (hystOk) {
                        auto candScore = [&](int c)->double {
                            switch (c) {
                                case 0: return scoreA;
                                case 1: return scoreB;
                                case 2: return scoreR;
                                default: return 1e30;
                            }
                        };

                        const bool prevSafe =
                            !vdisSoft ||
                            (prevIdx == 0 ? safeA : (prevIdx == 1 ? safeB : safeR));

                        if (prevSafe) {
                            const double newS  = candScore(idx);
                            const double prevS = candScore(prevIdx);

                            const double HYST_ABS_GATE = 0.03;
                            const double HYST_REL_GATE = 0.04;

                            const bool convincinglyBetter =
                                (newS + HYST_ABS_GATE < prevS) &&
                                (newS < prevS * (1.0 - HYST_REL_GATE));

                            if (!convincinglyBetter) {
                                idx = prevIdx;
                                if      (idx == 0) { val = FA; shade = 0.25f; }
                                else if (idx == 1) { val = FB; shade = 0.35f; }
                                else               { val = FR; shade = 0.8f;  }
                            }
                        }
                    }
                }
            }
        }

        winner[rel]   = idx;
        outVal[rel]   = val;
        outShade[rel] = shade;
        metrics.winner = idx;
        if (line >= 0 && line < (int)fvfMetrics.size() &&
            rel < (int)fvfMetrics[line].size())
        {
            fvfMetrics[line][rel] = metrics;
        }
    }
    
    int fieldCountTotal = 0, frameCountTotal = 0;
    for (int rel = 0; rel < width; ++rel) {
        if      (winner[rel] == 2) frameCountTotal++;
        else if (winner[rel] == 0 || winner[rel] == 1) fieldCountTotal++;
    }

    // Island cleanup
    auto applyIslandFilter = [&]() {
        std::vector<int> w2 = winner;
        const double EDGE_STOP_IRE = HEDGE_THRESH_IRE;
        const double DIFF_STOP_IRE = 6.0;

        for (int rel = 1; rel < width - 1; ++rel) {
            if (satMap[rel] > SAT_FALLBACK_START) continue;
            if (horizEdgeIRE(rel) > EDGE_STOP_IRE) continue;
            if (diffFVF[rel] > DIFF_STOP_IRE) continue;

            int L = winner[rel - 1];
            int C = winner[rel];
            int R = winner[rel + 1];

            if (L == R && C != L) {
                w2[rel] = L;
            }
        }

        if (w2 != winner) {
            winner.swap(w2);
            for (int rel = 0; rel < width; ++rel) {
                int idx = winner[rel];
                if      (idx == 0) { outVal[rel] = fieldA[rel];   outShade[rel] = 0.25f; }
                else if (idx == 1) { outVal[rel] = fieldB[rel];   outShade[rel] = 0.35f; }
                else               { outVal[rel] = frameB2[rel];  outShade[rel] = 0.8f;  }
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
                    else if (winner[r] == 2) cntF++;
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


// split2D dispatcher
void Comb::FrameBuffer::split2D()
{
    const bool writeWeights = configuration.showMap;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;

    if ((int)simpleField2D.size() < lastLine) simpleField2D.resize(lastLine);

    if (width <= 0 || firstLine >= lastLine) return;

    if (configuration.phaseCompensation) {
        buildPhaseCorrected1D();
        // Preserve the per-line locked 1D source before clpbuffer[1] is overwritten
        // in-place with the 2D output as we iterate down the frame.
        if ((int)locked1DSource.size() < lastLine) locked1DSource.resize(lastLine);
        for (int line = firstLine; line < lastLine; ++line) {
            auto &row = locked1DSource[line];
            if ((int)row.size() < width) row.assign(width, 0.0);
            const double *src = clpbuffer[1].pixel[line];
            for (int rel = 0; rel < width; ++rel) row[rel] = src[left + rel];
        }
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
        return;
    }

    if ((int)simpleField2D.size() < lastLine) simpleField2D.resize(lastLine);

    const bool vdisEnabled = configuration.tunables.VDIS_ENABLE;
    if (vdisEnabled) {
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
        if ((int)vdisMask.size() < lastLine) vdisMask.resize(lastLine);
        for (int line = firstLine; line < lastLine; ++line) {
            auto &row = vdisMask[line];
            if ((int)row.size() < width) row.assign(width, 0);
            else std::fill(row.begin(), row.end(), 0);
        }
    }

    std::vector<std::complex<double>> frameIQ; 

    for (int line = firstLine; line < lastLine; ++line) {
        // PREP: Precompute next neighbor (line+1) if in range and not already done
        if (line + 1 < lastLine) {
            computeSimpleField2DLine(line + 1, simpleField2D[line + 1].data());
            if (configuration.phaseCompensation)
                demodSimpleField2DLine(line + 1);
        }
        if (line >= demodLines) continue;

        computeSimpleField2DLine(line, scratch_fieldBLine.data());
        computeField2DLine(line, scratch_fieldLine.data(), scratch_fieldGate.data());

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

        simpleField2D[line].assign(width, 0.0);
        for (int rel = 0; rel < width; ++rel)
            simpleField2D[line][rel] = scratch_fieldBLine[rel];

        if (configuration.phaseCompensation) {
            demodSimpleField2DLine(line);
            computeFrameIQLine(line, frameIQ); 
            
            scratch_fieldBCenter.assign(width, 0.0);

            double bcos = (line < (int)demodBurstCos.size()) ? (double)demodBurstCos[line] : 1.0;
            double bsin = (line < (int)demodBurstSin.size()) ? (double)demodBurstSin[line] : 0.0;

            for (int rel = 0; rel < width; ++rel) {
                int h = left + rel;
                if (rel < (int)frameIQ.size()) {
                    const auto &Z = frameIQ[rel];
                    const double Ii = Z.real();
                    const double Qi = Z.imag();
                    const double s4 = sin4fsc(h);
                    const double c4 = cos4fsc(h);
                    const double lsin =  Ii * bcos + Qi * bsin;
                    const double lcos = -Ii * bsin + Qi * bcos;
                    const double cOut = 0.5 * (lsin * s4 + lcos * c4);
                    scratch_fieldBCenter[rel] = cOut;
                } else {
                    scratch_fieldBCenter[rel] = 0.0;
                }
            }
            
        }

        double *dst = clpbuffer[1].pixel[line];

        if (configuration.twoDVariant == Comb::Configuration::TwoDVariant::Field) {
            for (int rel = 0; rel < width; ++rel) dst[left + rel] = scratch_fieldLine[rel];
            if (writeWeights) std::fill(w2d_frame_weight[line].begin(), w2d_frame_weight[line].end(), 0.0f);
        }
        else if (configuration.twoDVariant == Comb::Configuration::TwoDVariant::FieldB) {
            for (int rel = 0; rel < width; ++rel) dst[left + rel] = scratch_fieldBLine[rel];
            if (writeWeights) std::fill(w2d_frame_weight[line].begin(), w2d_frame_weight[line].end(), 0.35f);
        }
        else if (configuration.twoDVariant == Comb::Configuration::TwoDVariant::Frame && configuration.phaseCompensation) {
            for (int rel = 0; rel < width; ++rel) dst[left + rel] = scratch_fieldBCenter[rel];
            if (writeWeights) std::fill(w2d_frame_weight[line].begin(), w2d_frame_weight[line].end(), 0.8f);
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
        fusedDemodLUT(bcos, bsin, spLUT_locked, cpLUT_locked, lutTi, lutTq);

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
        // tiRow/tqRow and preI/preQ are already affine-corrected from phaseLocked.
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
    constexpr double EXP_I_CUTOFF_MHZ  = 1.40;
    constexpr double EXP_Q_CUTOFF_MHZ  = 0.75; // We backed off from spec to avoid shaving the trace

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

    const double effGI = GI_PRODUCT * configuration.gi_product;
    const double effGQ = GQ_PRODUCT * configuration.gq_product;

    if ((int)scratch_preI.size() < width) scratch_preI.resize(width, 0.0);
    if ((int)scratch_preQ.size() < width) scratch_preQ.resize(width, 0.0);

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
        fusedDemodLUT(bcos, bsin, spLUT_locked, cpLUT_locked, lutTi, lutTq);

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
        // Apply FIRs to preI/preQ and write to I/Q
        for (int i = 0; i < width; ++i) {
            double accI = 0.0, accQ = 0.0;
            for (int k = -MI; k <= MI; ++k) {
                int idx = std::clamp(i + k, 0, width - 1);
                accI += scratch_preI[idx] * tapsI[k + MI];
            }
            for (int k = -MQ; k <= MQ; ++k) {
                int idx = std::clamp(i + k, 0, width - 1);
                accQ += scratch_preQ[idx] * tapsQ[k + MQ];
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

    double Ce = 1.0, Se = 0.0;
    basisCoeffs(Ce, Se);
    double spLUT[4], cpLUT[4];
    for (int i = 0; i < 4; ++i) shiftedBasis(i, Ce, Se, spLUT[i], cpLUT[i]);

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

        // Process pixels
        for (int x = 0; x < width; ++x) {
            const int h = left + x;

            // --- If residualVideo3D is enabled and valid, it owns the output ---
            if (configuration.residualVideo3D && prevFrameForVet && nextFrameForVet) {
                Y[h] = getBestY(line, h,
                               (double)rawLine[h] - clpLine[h],
                               *prevFrameForVet, *nextFrameForVet);
                if (configuration.showMap) w2d_frame_weight[line][x] = 0.0f;
                continue;
            }

            // Window [a,b] in demod domain
            int a = x - HALF, b = x + HALF - 1;
            if (a < 0) { b += -a; a = 0; }
            if (b >= width) { int over = b - (width - 1); b -= over; a -= over; if (a < 0) a = 0; }

            // Accumulate  T T^T and  R T^T in window, with gating/rolloff
            double STT[2][2] = {{0,0},{0,0}};
            double SRT[2][2] = {{0,0},{0,0}};
            int n = 0;

            for (int xi = a; xi <= b; ++xi) {
                const double ti = (double)tiRow[xi];
                const double tq = (double)tqRow[xi];
                const double ri = (double)triRow[xi];
                const double rq = (double)trqRow[xi];

                const double magT_ire = std::hypot(ti, tq) * invI;
                const double magR_ire = std::hypot(ri, rq) * invI;

                // Ignore very small vectors (ill-conditioned)
                if (magT_ire < MIN_FIT_IRE) continue;
                if (magR_ire < MIN_FIT_IRE) continue;

                // Soft rolloff on very large vectors (often where saturation/nonlinearity lives)
                double w = 1.0;
                if (magT_ire > MAX_FIT_IRE) {
                    const double t = (magT_ire - MAX_FIT_IRE) / (MAX_FIT_IRE + 1e-9);
                    w = 1.0 / (1.0 + 4.0 * t * t);
                }

                STT[0][0] += w * ti*ti; STT[0][1] += w * ti*tq;
                STT[1][0] += w * ti*tq; STT[1][1] += w * tq*tq;
                SRT[0][0] += w * ri*ti; SRT[0][1] += w * ri*tq;
                SRT[1][0] += w * rq*ti; SRT[1][1] += w * rq*tq;
                ++n;
            }

            // Solve A = SRT * inv(STT)
            double A[2][2] = {{1,0},{0,1}};
            double STTinv[2][2];
            if (T.Y_LOCAL_AFFINE_ENABLE && n >= 16 && mat2_inv(STT, STTinv)) {
                double tmp[2][2];
                mat2_mul(SRT, STTinv, tmp);
                A[0][0] = tmp[0][0]; A[0][1] = tmp[0][1];
                A[1][0] = tmp[1][0]; A[1][1] = tmp[1][1];
            }

            // Polar decompose and clamp
            double Rm[2][2], U[2][2];
            polar_decompose_2x2(A, Rm, U);

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
            const double sp = spLUT[idx], cp = cpLUT[idx];

            // Apply clamped transform to (ti0,tq0) for estimate only
            // NOTE: we apply only Rm here 
            const double ti_adj = Rm[0][0]*ti0 + Rm[0][1]*tq0;
            const double tq_adj = Rm[1][0]*ti0 + Rm[1][1]*tq0;

            const double lsin =  ti_adj * bcos + tq_adj * bsin;
            const double lcos = -ti_adj * bsin + tq_adj * bcos;
            const double cval_hat = 0.5 * (lsin * sp + lcos * cp);

            // Vet: local LS alignment (phase + correlation + shear) veto only if poor
            Vet1DResult vet = vetComposite1D(line, h, /*requireVerticalConfirm=*/false);
            if (configuration.showMap) w2d_frame_weight[line][x] = (float)vet.confidence;

            if (vet.accept) {
                Y[h] = (double)rawLine[h] - cval_hat;
                tiRowW[x] = (float)ti_adj;
                tqRowW[x] = (float)tq_adj;
            } else {
                Y[h] = (double)rawLine[h] - clpLine[h];
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
