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

#include <atomic>
#include <cstdlib>
#include <cstdio>

// TEMPORARY INSTRUMENT (LDCD_PROBE_DIST=1) -- distribution of the temporal
// similarity distance the agreement curve consumes, so its reward radius and
// veto threshold are set from THIS distance rather than inherited from the
// contaminated one they were tuned against. Counts only; cross-thread safe.
// Strip when the threshold is settled.
namespace {
constexpr double kDistEdges[] = {0.5, 1, 2, 3, 4, 5, 6, 7.5, 10, 15, 20, 30, 50};
constexpr int kDistBins = int(sizeof(kDistEdges) / sizeof(kDistEdges[0])) + 1;
struct DistCensus {
    bool on = std::getenv("LDCD_PROBE_DIST") != nullptr;
    std::atomic<long> bin[kDistBins];
    std::atomic<long> total{0};
    DistCensus() { for (auto &b : bin) b.store(0); }
    void add(double d) {
        total.fetch_add(1, std::memory_order_relaxed);
        int k = kDistBins - 1;
        for (int i = 0; i < kDistBins - 1; ++i)
            if (d <= kDistEdges[i]) { k = i; break; }
        bin[k].fetch_add(1, std::memory_order_relaxed);
    }
    ~DistCensus() {
        if (!on) return;
        const double n = std::max(1L, total.load());
        std::fprintf(stderr, "[DIST] temporal candidates %ld\n", total.load());
        long cum = 0;
        for (int i = 0; i < kDistBins; ++i) {
            cum += bin[i].load();
            if (i < kDistBins - 1)
                std::fprintf(stderr, "   d <= %5.1f IRE : %5.1f%%  (cum %5.1f%%)\n",
                             kDistEdges[i], 100.0 * bin[i].load() / n,
                             100.0 * cum / n);
            else
                std::fprintf(stderr, "   d >  %5.1f IRE : %5.1f%%\n",
                             kDistEdges[kDistBins - 2],
                             100.0 * bin[i].load() / n);
        }
    }
};
DistCensus g_distCensus;
} // namespace

#include "combmath.h"
#include "feasibleband.h"
#include "framecanvas.h"
#include "deemp.h"
#include "firfilter.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
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

qint32 Comb::Configuration::getLookBehind() const {
    return (dimensions == 3 && !diagnosticOnly()) ? 1 : 0;
}
qint32 Comb::Configuration::getLookAhead() const {
    return (dimensions == 3 && !diagnosticOnly()) ? 1 : 0;
}

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

    // Geometry or mode may have changed; drop persistent FrameBuffers so the
    // next decodeFrames() call rebuilds them with the new dimensions.
    persistentNext.reset();
    persistentCurrent.reset();
    persistentPrevious.reset();
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

    // Take ownership from the persistent triple-buffer.  First call allocates;
    // every subsequent call reuses the same FrameBuffer storage (the
    // ~180 MB attribution vector is the bulk).  The pre-roll at the head of
    // the per-field loop overwrites all per-field state, so reusing across
    // batches behaves identically to within a batch.
    auto next     = std::move(persistentNext);
    auto current  = std::move(persistentCurrent);
    auto previous = std::move(persistentPrevious);
    if (!next) {
        next     = std::make_unique<FrameBuffer>(videoParameters, configuration);
        current  = std::make_unique<FrameBuffer>(videoParameters, configuration);
        previous = std::make_unique<FrameBuffer>(videoParameters, configuration);
    }

    // Chain pre-roll depth (2026-08-02). The anticipated chain, the tone
    // anchor, and the anchored plane all walk prevF in display order, but
    // the 2D witness path pre-rolled only ONE frame -- so every cross-frame
    // chain silently restarted at every batch head (measured: one dead
    // anticipated chain per 8-frame batch, rendered as a periodic CCR
    // detector dropout). The pool has ALWAYS delivered two real
    // look-behind frames (paddingHistory, sequential in frame order under
    // the queue lock); reach back and load them both so the chain enters
    // the batch alive. Cost: two extra analysed frames per batch, offset
    // by the decoderBatchFrames raise (8 -> 12 keeps the (N+k)/N analysis
    // overhead at its previous level). Determinism holds: batch
    // composition depends only on queue order, never on thread count.
    qint32 preRollFields = 2;
    if (configuration.dimensions == 3 && !configuration.diagnosticOnly())
        preRollFields = 4;
    if (configuration.phaseCompensation && configuration.lumaWitness)
        preRollFields = 6;
    // A/B escape (bisect instrument): LDCD_PREROLL overrides in FIELDS.
    static const int preRollEnv = []{
        const char *s = std::getenv("LDCD_PREROLL");
        return s ? std::atoi(s) : -1;
    }();
    if (preRollEnv >= 2) preRollFields = preRollEnv;
    const qint32 preStart = startIndex - preRollFields;

    // True when the frame analyzed on the PREVIOUS loop iteration (now
    // sitting in `current` after rotation) was genuinely loaded this batch —
    // the guard that keeps the frame-axis conformance test from comparing
    // against a stale recycled buffer at pre-roll or batch boundaries.
    bool prevIterAnalyzed = false;

    // NOTE: the buffers persisting across calls invites an obvious
    // optimisation -- a call that just served frame F leaves them holding
    // {F-1, F, F+1}, and the next call, serving F+1, needs {F, F+1, F+2}: one
    // rotation and one new frame rather than a fresh pre-roll.  It was tried
    // and reverted.  Two things defeat it.  Threads take frames round-robin
    // with a Comb (and triple-buffer) each, so at -t 8 a thread never holds
    // its own predecessor and the reuse never fires; measured 0% at -t 8
    // against 95% at -t 1.  Worse, where it did fire it CHANGED THE PICTURE:
    // the pre-roll analyses its first frame with a null temporal context
    // (prevIterAnalyzed starts false), while a reused frame keeps the analysis
    // it was given with its real predecessor, so -t 1 and -t 8 stopped
    // agreeing.  Output must not depend on --threads.
    //
    // The reuse only becomes available once a thread is handed a RUN of
    // consecutive frames (upstream's batchFrames, which the cadence rework
    // reduced to 1).  Batch composition does not depend on thread count, so
    // that form keeps the decode deterministic.

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

        // PERF (2026-08-06): under the witness path at dimensions == 2, the
        // load-time split2D on an uncovered frame consumes a causal
        // placeholder that refineRetractedTemporal() replaces at decode time,
        // after which split2D RERUNS -- the load-time run is computed and
        // discarded. Deferring split2D to decode time runs it exactly once
        // per frame with identical inputs at that moment:
        //   * rerun frames: the deferred run IS today's rerun (same point in
        //     the sequence, same completed planes);
        //   * non-rerun frames: refineRetractedTemporal mutates nothing on
        //     every early-out (verified: pure reads before each return), the
        //     hull is opt-in, and the only cross-frame writer between load
        //     and decode (buildCarrierAnalysis's prevFrame argument) only
        //     reads -- so the deferred run sees bit-identical inputs.
        // Byte-identity verified against the pre-change binary (md5 on 24
        // frames of covered program material). The deferral is OFF for
        // dimensions == 3 (split3D reads neighbours' load-time 2D), for
        // showMap (overlayMap reads neighbour maps), and under
        // LDCD_PROBE_2D_RERUN=0 (whose meaning is "keep the causal 2D").
        static const bool rerun2D = []{
            const char *e = std::getenv("LDCD_PROBE_2D_RERUN");
            return !(e && std::atoi(e) == 0);
        }();
        // PROMOTED 2026-08-06 (visual verdict: "fine, compared to
        // baseline"). The historical order ran split2D twice on rerun
        // frames, and the divergence hunt proved split2D non-idempotent:
        // collect/finalize ACCRETE onto the persistent per-pixel
        // attribution records (max-then-scale on assessment.coherence, max
        // on attributionConflict), so the old path shipped attribution that
        // mixed the causal placeholder run's evidence with the refined
        // run's on rerun frames (~0.5-1.4% of samples, local amplitudes to
        // ~26%). The single decode-time run is the clean construction —
        // evidence from the completed plane only — and ~13% faster. The
        // three-hash discriminator record lives in the perf ledger.
        const bool defer2D =
            configuration.phaseCompensation && configuration.lumaWitness &&
            configuration.dimensions == 2 && !configuration.showMap &&
            rerun2D;

        if (canLoadNext) {
            next->loadFields(inputFields[fieldIndex + 2],
                             inputFields[fieldIndex + 3]);

            next->split1D();

            // Star-law evidence (produceY license): built for every loaded
            // buffer so temporal neighbours can pool it. Needs only
            // rawbuffer + the exact channel, both ready at load.
            next->buildStarEvidence();

            if (configuration.phaseCompensation) {
                next->phaseLocked();
                next->buildCarrierAnalysis(
                    prevIterAnalyzed ? current.get() : nullptr);
                // Corner-leak corrector: consumes the analysis record, the
                // canonical bandpass and the aperture-mean pool. Diagnostic
                // only -- publishes lockedCornerLeak_flat, changes no output.
                next->buildCornerLeak();
                next->buildPhaseCorrected1D();
                if (configuration.lumaWitness)
                    next->buildCarrierRetracted(
                        current->holdsRealFrame() ? current.get() : nullptr);
            }

            if (!configuration.diagnosticOnly() && !defer2D)
                next->split2D();
        } else {
            // Nothing to load into the recycled buffer, so whatever it still
            // holds belongs to an earlier call.  Say so, rather than leaving a
            // stale frame wearing the `next` label.
            next->forgetHeldFrame();
        }
        prevIterAnalyzed = canLoadNext;

        if (fieldIndex < startIndex)
            continue;

        // Temporal context exists only when all three buffers hold a real
        // neighbouring picture.  The buffers answer for themselves: a frame
        // that was never loaded, was invalidated above, or came from the
        // pool's synthetic black padding reports no identity and is refused.
        // At the end of the stream there is no look-ahead tail, so the last
        // frame keeps its 2D result.
        const bool temporalContextReady =
            previous->holdsRealFrame() &&
            current->holdsRealFrame() &&
            next->holdsRealFrame();

        // Finish the uncovered estimate before any current-time consumer is
        // allowed to see it.  load-time construction is deliberately only a
        // provisional retracted candidate: the fact-corrected carrier plane
        // is published by refineRetractedTemporal() after BOTH covered
        // neighbours are available.  Rebuild 2D here so its comb candidates,
        // and the 3D seed made from them below, consume that completed plane
        // rather than the earlier causal placeholder.
        bool publishedTwoSidedEstimate = false;
        if (configuration.phaseCompensation && configuration.lumaWitness) {
            const FrameBuffer *pf =
                previous->holdsRealFrame() ? previous.get() : nullptr;
            const FrameBuffer *nf =
                next->holdsRealFrame() ? next.get() : nullptr;
            publishedTwoSidedEstimate =
                current->refineRetractedTemporal(pf, nf);
            // Magnitude law after the phase refinement, so the bound applies
            // to the carrier the consumers will actually read.
            if (current->applyCertifiedCarrierHull(pf, nf))
                publishedTwoSidedEstimate = true;
            // INSTRUMENT (temporary): LDCD_PROBE_2D_RERUN=0 keeps the
            // two-sided refinement but withholds it from the comb, so 2D
            // stays on the causal tone-anticipated ladder built at load
            // time.  Separates "what the forward field buys inside the
            // comb" from "what it buys in the Y election".  Strip once the
            // question closes.  (rerun2D hoisted to the loop head, where the
            // split2D deferral consumes it too.)
            if (!configuration.diagnosticOnly() &&
                (defer2D || (publishedTwoSidedEstimate && rerun2D)))
                current->split2D();
        }

        if (configuration.dimensions == 3 &&
            !configuration.diagnosticOnly()) {
            current->copy2DTo3D();

            if (temporalContextReady)
                current->split3D(*previous, *next);
        }

        const qint32 frameIndex = (fieldIndex - startIndex) / 2;
        componentFrames[frameIndex].init(videoParameters,
                                         configuration.diagnosticOnly());
        current->setComponentFrame(componentFrames[frameIndex]);


        // Diagnostic fast path: witness/carrier analysis is complete before
        // the comb/election render stages below. Publish the selected signal
        // directly as monochrome output and skip the normal render path.
        if (configuration.diagnosticOnly()) {
            current->outputDiagnosticFrame();
            continue;
        }

        /*
         * Output path.
         *
         * splitIQlocked() is the post-election demod of the selected comb.
         */
        if (configuration.phaseCompensation) {
            current->measurePostCombImpurity();
            current->splitIQlocked(
                previous->holdsRealFrame() ? previous.get() : nullptr,
                next->holdsRealFrame() ? next.get() : nullptr);
            current->doCNR();
            current->produceY(
                previous->holdsRealFrame() ? previous.get() : nullptr,
                next->holdsRealFrame() ? next.get() : nullptr);
            current->filterIQLocked();
            current->doYNR();
            current->transformIQ(configuration.chromaGain,
                                 configuration.chromaPhase);
        } else {
            current->splitIQ();
            current->adjustY();
            current->filterIQ();
            current->doCNR();
            current->doYNR();
            current->transformIQ(configuration.chromaGain,
                                 configuration.chromaPhase);
        }

        if (configuration.showMap &&
            (configuration.dimensions == 3 ||
             configuration.twoDVariant == Comb::Configuration::TwoDVariant::FieldBSimple))
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

            auto fieldLabel = [&](int cid) -> char {
                if (cid == lddecode::kCadenceVideo)
                    return 'i';                    // -2: 59.94i interlaced video

                if (cid == lddecode::kCadenceProgressive)
                    return 'P';                    // -3: 29.97p progressive

                if (cadenceKnown(cid))
                    return cadenceFilmLetter(cid); // 0..19: film frame letter

                return '?';                        // -1: unknown / unassigned
            };

            const char labelTop    = fieldLabel(cidTop);
            const char labelBottom = fieldLabel(cidBottom);

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

    // Return the triple-buffer to the persistent slots for the next call.
    persistentNext     = std::move(next);
    persistentCurrent  = std::move(current);
    persistentPrevious = std::move(previous);
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
        const bool needFrameIQ =
            (configuration.twoDVariant == Comb::Configuration::TwoDVariant::FrameAAdaptiveIQ ||
             configuration.twoDVariant == Comb::Configuration::TwoDVariant::FrameBDirectIQ ||
             wantFvf);

        // 2D score blending visualization (only written when showMap is true)
        if (wantMap) {
            w2d_frame_weight.assign(lines, std::vector<float>(width, 0.0f));
            fieldBDecisionReason_flat.assign(size_t(lines + 1) * size_t(width), FieldBReasonNone);
        }
        // Band membership feeds the Y election's band cede on every locked
        // decode that builds Field B; it must not depend on showMap.
        if (wantLocked)
            chromaBoundaryBand_flat.assign(size_t(lines + 1) * size_t(width), 0);
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
        // Locked-path-only stable 1D source.
        if (wantLocked) {
            lockedLumaBaseY4_flat.assign(size_t(lines + 1) * size_t(width), 0.0);
            lockedLumaSmooth_flat.assign(size_t(lines + 1) * size_t(width), 0.0);
            // The lurch-sharpened coarse floor is consumed only by the
            // --luma-witness produceY election. Default reconstructs on the
            // cheap baseY4 floor. Allocate sharp only under witness so the
            // baseline path pays neither its buffer nor its build. baseY4 and
            // the geometry-only smooth/hDelta services stay unconditional.
            if (configuration.lumaWitness)
                lockedLumaSharp_flat.assign(size_t(lines + 1) * size_t(width), 0.0);
            else
                lockedLumaSharp_flat.clear();
            lockedLumaHDeltaIRE_flat.assign(size_t(lines + 1) * size_t(width), 0.0f);
            lockedCornerLeak_flat.assign(size_t(lines + 1) * size_t(width), 0.0);
            lockedLumaCacheValid = false;
        }
        // Collected pool: sliding four-sample aperture means. UNCONDITIONAL --
        // it is a running sum (O(1)/sample), and both the locked coarse-residual
        // parallax AND the bucket-path carrier feasibility hull consume it, so
        // it must exist without phaseCompensation. buildApertureMeans() fills it
        // from split1D (which runs on every path, before phaseLocked).
        lockedApertureMean_flat.assign(size_t(lines + 1) * size_t(width), 0.0);
        // Preclean ring is only needed for Frame/FVF in locked mode.
        if (needFrameIQ) {
            for (int s = 0; s < 3; ++s) {
                precleanRing[s].assign(width, 0.0);
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
        // Initialize demod contiguous buffers geometry
        // demodLines indexed by absolute line number (safe upper bound)
        demodWidth = width;
        demodLines = lines + 1;
        carrierGrammar.assign(demodLines, CombCarrierGrammar{});
        combReachIndex.bind(&carrierGrammar,
                            videoParameters.firstActiveFrameLine,
                            videoParameters.lastActiveFrameLine);
        if (wantLocked) {
            demodTI_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            demodTQ_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            demodTI4fsc_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            demodTQ4fsc_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            locked1DTI4fsc_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            locked1DTQ4fsc_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            smoothedLockedTI_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            smoothedLockedTQ_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            smoothedLockedRowValid.assign(demodLines, std::uint8_t{0});
            lockedProductI_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            lockedProductQ_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            lockedCarrierComposite_flat.assign(size_t(demodLines) * demodWidth, 0.0);
            carrierImpurity_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            // Cross-color mask pair only exists when the return feature is
            // explicitly engaged, so the ordinary locked path pays
            // neither the memory nor the second pass.
            if (configuration.tunables.CC_SUPPRESSION_WEIGHT > 0.0) {
                lockedCcMaskRaw_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
                ccDetectorVerdict_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
                lockedCcMask_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            } else {
                lockedCcMaskRaw_flat.clear();
                ccDetectorVerdict_flat.clear();
                lockedCcMask_flat.clear();
            }
            regionSamePartner_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            regionAlienPartner_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            locked1DRawBandpass_flat.assign(size_t(demodLines) * demodWidth, 0.0);
            locked1DSource_flat.assign(size_t(demodLines) * demodWidth, 0.0);
            locked1DParallaxRepairStrength_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
            locked1DParallaxRepairDelta_flat.assign(size_t(demodLines) * demodWidth, 0.0f);
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
// metadata. Also initialises per-line carrier grammar polarity.
// capturePartnerSeqNo records the original TBC frame pairing for each field,
// carried forward for reconstruction.
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

    // Exact-carrier side channel: interleave the fields' planes on the same
    // mapping as rawbuffer (frame line 2k = first field line k, 2k+1 =
    // second). NaN everywhere a field carried no plane.
    {
        const int fw = videoParameters.fieldWidth;
        exactCarrier_flat.assign((size_t)frameHeight * fw,
                                 std::numeric_limits<float>::quiet_NaN());
        exactCoverageCache = -1;   // recompute for the newly held frame
        anchoredCarrierProvenance =
            AnchoredCarrierProvenance::None; // stale plane must not survive reuse
        antRefAge = -1;            // chained anticipated reference likewise
        starEvidenceBuilt = false; // star license evidence follows the frame
        starFootprintBuilt = false;
        starFootprint_flat.clear();
        syncIncFirst  = firstField.dgSyncIncrement;
        syncIncSecond = secondField.dgSyncIncrement;
        certifiedLineCache.clear(); // per-line certified verdicts likewise
        auto copyPlane = [&](const QVector<float> &src, int frameParity) {
            if (src.isEmpty()) return;
            qint32 fl = 0;
            for (qint32 frameLine = frameParity; frameLine < frameHeight;
                 frameLine += 2, ++fl) {
                if ((fl + 1) * fw > src.size()) break;
                std::copy(src.constData() + (size_t)fl * fw,
                          src.constData() + (size_t)(fl + 1) * fw,
                          exactCarrier_flat.data() + (size_t)frameLine * fw);
            }
        };
        copyPlane(firstField.dgExactCarrier, 0);
        copyPlane(secondField.dgExactCarrier, 1);
    }

    heldSeq1 = firstField.field.seqNo;
    heldSeq2 = secondField.field.seqNo;

    firstFieldPhaseID  = firstField.field.fieldPhaseID;
    secondFieldPhaseID = secondField.field.fieldPhaseID;
    
    const bool editSplit = secondField.field.cinemap.isEditBoundary;
    isSceneStart = firstField.field.cinemap.isEditBoundary;
    hasSceneSplit = editSplit;
    const bool progressiveFrameRegimeAllowed =
        firstField.allowProgressiveFrameRegime &&
        secondField.allowProgressiveFrameRegime;
    
    const qint32 cidA = firstField.field.cinemap.cadenceId;
    const qint32 cidB = secondField.field.cinemap.cadenceId;
    // Under autosolve editSplit is narrow and correct: the flag marks the first
    // field of a new scene, so it is set here only when the cut falls BETWEEN
    // this frame's two fields — the "A/A" case, where the leading A belongs to
    // the outgoing shot and combing the pair would reach across the transition.
    // A cut at the leading edge ("/AA") leaves it false and the frame is film,
    // period.
    //
    // Under --set-cadence it does not apply at all. The user reaches for a jam
    // precisely when the solve has failed, so its boundaries are not evidence to
    // defer to; the asserted count is the whole authority. isSceneStart and
    // hasSceneSplit above are untouched — they cannot contradict the count, and
    // only ever stop evidence being borrowed across a marked transition.
    const bool cadenceEditSplit = editSplit && !configuration.imposedCadence;
    cadenceId = lddecode::mergeCadenceIdForInterleavedFrame(
        cidA, cidB, cadenceEditSplit);
    if (!progressiveFrameRegimeAllowed)
        cadenceId = lddecode::kCadenceVideo;
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
    combReachIndex.bind(&carrierGrammar, first, last);

    lockedLumaCacheValid = false;
}



// 1D horizontal bandpass: isolates subcarrier energy by subtracting the average
// of the samples two positions either side (a 2-tap comb at 2fsc), scaled by 0.5.
// Result written to clpbuffer[0].
// ---------------------------------------------------------------------------
// APERTURE-CHROMA PHYSICS (the instrument that measured this, LDCD_PROBE_APERTURE,
// has since been removed; it was measurement only -- it read state, wrote
// nothing, and changed no output). The reasoning below is the record.
//
// The question (writeup S7): do the four demodulated aperture views bound the
// chroma usefully at a hard luma step?
//
// Algebra answers half of it outright, so the probe does not need to ask it.
// The four covering views of sample x are view_k[x] = raw[x] - apMean[x-3+k],
// so their MEAN is
//     m[x] = raw[x] - A[x],      A[x] = (1/4) * sum_k apMean[x-3+k]
// and A is exactly the 7-tap triangle T = {1,2,3,4,3,2,1}/16 applied to LUMA:
// four width-4 boxcars at unit offsets convolve to that triangle, and every
// boxcar is chroma-free because a legal carrier nulls over any four samples.
// Hence
//     m[x] = C[x] + (I - T){Y}[x]
// The four-view mean is the true carrier PLUS A HIGHPASS OF LUMA. T is
// symmetric with unit gain and zero first moment, so it reproduces affine luma
// exactly and m is unbiased on constant and on RAMP luma -- but not at a STEP,
// where a unit step leaves the doublet {-1,-3,-6,+6,+3,+1}/16. That is the same
// family of error the 1D bandpass itself makes (-0.25*D2_2 Y). So the naive
// aperture reference is biased exactly where the edge colour band lives, and
// averaging the views cannot be the instrument.
//
// What survives is the WIDTH of the feasible interval rather than its centre.
// On flat luma the four covering means agree, the hull collapses to a POINT,
// and the carrier is known EXACTLY -- raw minus luma, no filter and no
// assumption. Ambiguity is confined to a narrow zone around each luma edge.
// The envelope is bandlimited BY LAW, so the exact anchors flanking an edge
// constrain the envelope through the zone -- provided the zone is narrower
// than the envelope kernel's reach. That proviso is a measurement, not a
// derivation, and it is the load-bearing one.
//
// Measured here, binned by hull width (which IS edge proximity -- it is the
// luma spread over T's support -- so no edge detector is needed):
//   (1) what fraction of the line is exact-anchor, and how the ambiguous run
//       lengths compare with the 9-tap envelope kernel's +-4 reach;
//   (2) E_mean -- the naive four-view-mean envelope (the biased reference);
//   (3) E_ext  -- the SAME quantity restricted to anchors and extended
//       lawfully by the envelope kernel (the proposed reference);
//   against E_1d -- what the 1D bandpass actually emits.
// E_mean and E_ext are built from one signal m and differ only in whether m is
// trusted where it is provably exact, which isolates the step bias directly.
// ---------------------------------------------------------------------------

void Comb::FrameBuffer::split1D()
{
    // Sample-column windows come from the metadata header, never from
    // hardcoded columns. MEASURED, on a colourburst study that hardcoded the
    // burst as h 40..130: those columns caught sync-edge energy, so the study
    // measured deterministic line structure where it meant to measure jitter.
    // videoParameters carries colourBurstStart/End (and the active-video
    // bounds read below) precisely so no consumer has to guess them.
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int fullWidth = videoParameters.fieldWidth;

    if (left >= right || firstLine >= lastLine)
        return;

    // Fill the coarse-residual aperture-mean pool for every path here, before
    // phaseLocked runs, so the locked path reads the same values and the bucket
    // path can consume it for the carrier feasibility hull below.
    buildApertureMeans();

    // Canonical lurch step runs, one detection per line per frame from the
    // pool above. Consumers (witness sharpener, edge probes, the coming 2D
    // threshold work) read these instead of privately re-running the scan.
    buildLurchStepRuns();

    // Bucket-path carrier feasibility hull. Rotation-free and O(width), so
    // there is no reason to deny it to the cheap path -- but it changes output,
    // so it is opt-in (LDCD_BUCKET_HULL=1) until validated, and it is confined
    // to the bucket path (the locked path clamps inside buildCornerLeak on the
    // corrected carrier instead). The clamp is the SAME shared helper.
    static const bool bucketHull = []{
        const char *s = std::getenv("LDCD_BUCKET_HULL");
        return s && std::atoi(s) != 0;              // default OFF: bucket unchanged
    }();
    const bool applyBucketHull = bucketHull && !configuration.phaseCompensation;

    for (int line = firstLine; line < lastLine; ++line) {
        const quint16 *src = rawbuffer.data() + line * fullWidth;
        double *dst        = clpbuffer[0].pixel[line];

        for (int h = left; h < right; ++h) {
            int hm2 = h - 2; if (hm2 < left)   hm2 = left  + (left  - hm2 - 1);
            int hp2 = h + 2; if (hp2 >= right)  hp2 = right - 1 - (hp2 - right);
            dst[h] = ((double)src[h] - 0.5 * ((double)src[hm2] + (double)src[hp2])) * 0.5;
        }

        if (applyBucketHull)
            applyCarrierFeasibilityHull(line, dst + left);
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

    // Per-frame reset is the textbook _platform_memmove tax (see perf survey
    // 2026-06-25): with the triple-buffer fix in place, the per-batch ctor
    // zero-fill is gone, but this per-line seed still pays roughly the same
    // memmove cost spread frame by frame. Use a single static default-
    // initialized template (correctly carries non-zero defaults such as
    // CombAttributionAssessment::uncertainClaim = 1.0) and one full-record
    // assignment per pixel, instead of two separate copy-assignments from
    // zero-initialized temporaries.
    static const AttributionEvidence kFreshRecord{};
    for (int rel = 0; rel < width; ++rel) {
        row[rel] = kFreshRecord;
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
// FVF election 
void Comb::FrameBuffer::scoreFieldVsFrame(
    int line,
    const CombTapLine &tapLine,
    const std::vector<double> &candidateA,
    const double *fieldB,
    const std::vector<double> *frameB,
    double *outMixed,
    bool writeWeights,
    const double *lateral1D,
    const std::vector<std::complex<double>> *frameIQ)
{
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;

    if (width <= 0) return;
    if (!fieldB || !frameB || (int)candidateA.size() < width ||
        (int)frameB->size() < width || !outMixed)
        return;

    const std::vector<double> &frameB2 = *frameB;
    const double *candidateAData = candidateA.data();
    if (line >= 0 && line < (int)fvfMetrics.size() &&
        (int)fvfMetrics[line].size() < width)
    {
        fvfMetrics[line].assign(width, FvfModelMetrics());
    }

    const auto &T   = configuration.tunables;
    const double invI = this->invIreScale;
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;

    // Radius of the horizontal neighbor window used in cross-domain estimation.
    // Kept local: this is not a tunable in the current header.
    const int  NEIGH_RAD        = 2;

    // Local constants replacing older/nonexistent tunable names.
    const double FRAME_MODEL_BIAS_LOCAL = 0.90;
    const double FRAME_SCALE_BIAS_STRENGTH_PROGRESSIVE_LOCAL =
        T.FVF_SCALE_FINE_FRAME_B_BONUS;
    const double FRAME_SCALE_BIAS_STRENGTH_INTERLACE_LOCAL =
        0.5 * T.FVF_SCALE_FINE_FRAME_B_BONUS;
    const double ONE_D_NEAR_THRESH_IRE_LOCAL = 1.0;

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


    // Minimum luma difference between Field A and Field B combs to apply A/B divergence penalty (IRE)
    const double FIELD_DISAGREE_IRE = 6.0;

    // Below this FVF candidate difference, candidates are close enough that frame is preferred (IRE)
    const double FVF_SMALL_DIFF_IRE = (T.FVF_SMALL_DIFF_IRE > 0.0) ? T.FVF_SMALL_DIFF_IRE : 3.0;
    const int srcBufIndex = configuration.phaseCompensation ? 1 : 0;
    const AttributionEvidence *attrRow = attributionEvidence_line(line);
    const double *attrAuthorityRow =
        ((int)scratch_attrMembershipY.size() >= width)
            ? scratch_attrMembershipY.data()
            : nullptr;
    const double *sample1DRow = lateral1D
        ? lateral1D
        : (clpbuffer[srcBufIndex].pixel[line] + left);
    const int up1Line = std::clamp(line - 1, firstLine, lastLine - 1);
    const int dn1Line = std::clamp(line + 1, firstLine, lastLine - 1);
    const int up2Line = std::clamp(line - 2, firstLine, lastLine - 1);
    const int dn2Line = std::clamp(line + 2, firstLine, lastLine - 1);
    const bool haveVert2 = (line - 2 >= firstLine) && (line + 2 < lastLine);
    const double *srcLine = clpbuffer[srcBufIndex].pixel[line] + left;
    const double *srcUp1 = clpbuffer[srcBufIndex].pixel[up1Line] + left;
    const double *srcDn1 = clpbuffer[srcBufIndex].pixel[dn1Line] + left;
    const double *srcUp2 = clpbuffer[srcBufIndex].pixel[up2Line] + left;
    const double *srcDn2 = clpbuffer[srcBufIndex].pixel[dn2Line] + left;

    auto sample1D = [&](int rel)->double {
        return sample1DRow[std::clamp(rel, 0, width - 1)];
    };
    auto vertContrastIRE = [&](int rel)->double {
        if (!haveVert2) return 0.0;
        rel = std::clamp(rel, 0, width - 1);
        return std::fabs(srcUp2[rel] - srcDn2[rel]) * invI;
    };

    if ((int)scratch_fvf_winner.size() != width) {
        scratch_fvf_winner.assign(width, 1);
        scratch_fvf_winner2.assign(width, 1);
        scratch_fvf_outVal.assign(width, 0.0);
        scratch_fvf_outShade.assign(width, 0.35f);
        scratch_fvf_diffFVF.assign(width, 0.0);
        scratch_fvf_satMap.assign(width, 0.0);
    } else {
        std::fill(scratch_fvf_winner.begin(), scratch_fvf_winner.end(), 1);
        std::fill(scratch_fvf_outVal.begin(), scratch_fvf_outVal.end(), 0.0);
        std::fill(scratch_fvf_outShade.begin(), scratch_fvf_outShade.end(), 0.35f);
        std::fill(scratch_fvf_diffFVF.begin(), scratch_fvf_diffFVF.end(), 0.0);
        std::fill(scratch_fvf_satMap.begin(), scratch_fvf_satMap.end(), 0.0);
    }
    if ((int)scratch_fvf_notchFieldA.size() != width)
        scratch_fvf_notchFieldA.resize(width);
    if ((int)scratch_fvf_notchFieldB.size() != width)
        scratch_fvf_notchFieldB.resize(width);
    if ((int)scratch_fvf_notchFrame.size() != width)
        scratch_fvf_notchFrame.resize(width);
    if ((int)scratch_fvf_notchSource.size() != width)
        scratch_fvf_notchSource.resize(width);

    std::vector<int>    &winner   = scratch_fvf_winner;
    std::vector<double> &outVal   = scratch_fvf_outVal;
    std::vector<float>  &outShade = scratch_fvf_outShade;
    std::vector<double> &diffFVF  = scratch_fvf_diffFVF;
    std::vector<double> &satMap   = scratch_fvf_satMap;
    std::vector<double> &notchCandidateA = scratch_fvf_notchFieldA;
    std::vector<double> &notchFieldB = scratch_fvf_notchFieldB;
    std::vector<double> &notchFrame  = scratch_fvf_notchFrame;
    std::vector<double> &notchSource = scratch_fvf_notchSource;
    int fieldCountTotal = 0, frameCountTotal = 0;

    const double SAT_FALLBACK_START = 6.0;
    const double SAT_FALLBACK_FULL  = 20.0;
    double prev_sat_t = 0.0;

    // Core Logic of Field Vs Frame
    // Progressive uses the frame regime; interlace uses the field regime.
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

    // Pre-compute |frameIQ[r]| once per pixel.  The hot pixel loop reads this
    // magnitude up to 9 times per pixel (rel, rel±1, rel±2, rel±4 in the
    // fine/mid/coarse band split), so caching it cuts ~6.6M hypot calls per
    // frame down to ~width per line.
    const bool haveFrameIQForMag = frameIQ && !frameIQ->empty();
    const int frameIQN = haveFrameIQForMag ? (int)frameIQ->size() : 0;
    if ((int)scratch_fvf_iqMag.size() != width)
        scratch_fvf_iqMag.resize(width);
    if (haveFrameIQForMag) {
        const int n = std::min(width, frameIQN);
        for (int r = 0; r < n; ++r) {
            const auto &z = (*frameIQ)[r];
            scratch_fvf_iqMag[r] = boundedMag(z);
        }
        // If frameIQ is shorter than width, replicate the last valid mag for
        // tail entries — matches the original lambda's std::clamp(r,0,width-1)
        // behaviour against the inner index.
        for (int r = n; r < width; ++r)
            scratch_fvf_iqMag[r] = (n > 0) ? scratch_fvf_iqMag[n - 1] : 0.0;
    } else {
        std::fill(scratch_fvf_iqMag.begin(), scratch_fvf_iqMag.begin() + width, 0.0);
    }

    for (int r = 0; r < width; ++r) {
        notchCandidateA[r] = getNotchLumaEven2(candidateAData, r, width);
        notchFieldB[r] = getNotchLumaEven2(fieldB, r, width);
        notchFrame[r] = getNotchLumaEven2Vec(frameB2, r);
        notchSource[r] = getNotchLumaEven2(srcLine, r, width);
    }

    for (int rel = 0; rel < width; ++rel) {
        double FA = candidateA[rel];
        double FB = fieldB[rel];
        double FR = frameB2[rel];
        double L1 = sample1D(rel);

        const double satFR_demod = (haveFrameIQForMag && rel < frameIQN)
            ? scratch_fvf_iqMag[rel]
            : std::fabs(FR);

        const Cond1D FA_c = condSamePhase(candidateAData, rel);
        const Cond1D FB_c = condSamePhase(fieldB, rel);
        const Cond1D FR_c = condSamePhaseVec(frameB2, rel);

        // Use conditioned candidates for scoring only; output still uses raw winners.
        const double FA_s = FA_c.score;
        const double FB_s = FB_c.score;
        const double FR_s = FR_c.score;

        // Luma proxies: prefer the pure even-offset notch (±2 average), which is
        // less sensitive to single-pixel spikes than a [1,2,1] that includes center.
        double lumFA = notchCandidateA[rel];
        double lumFB = notchFieldB[rel];
        double lumFR = notchFrame[rel];


        double Cpm1 = srcUp1[rel];
        double Cpp1 = srcDn1[rel];
        double Cpm2 = srcUp2[rel];
        double Cpp2 = srcDn2[rel];

        double frameLikeStack = 0.5 * (Cpm1 + Cpp1);
        double fieldLikeStack = 0.5 * (Cpm2 + Cpp2);
        double diff_stack_ire = std::fabs(frameLikeStack - fieldLikeStack) * invI;

        double diff_candA_ire = std::fabs(lumFR - lumFA) * invI;
        double diff_candB_ire = std::fabs(lumFR - lumFB) * invI;
        double frameFieldCandidateDistIRE = diff_candB_ire;
        double frameModelDistIRE = frameFieldCandidateDistIRE;
        // Field-vetoes-frame divergence guard, kept ONLY for the interlace
        // regime.  Before ld-cinemap supplied cadence metadata, interfield
        // divergence was the only protection the fields had, so a frame
        // candidate that strayed far from the field comb was distrusted.  In the
        // metadata-driven Frame model regime the frame model is authoritative
        // (cadence is known), and a frame candidate is SUPPOSED to diverge from
        // a field comb that cannot resolve fine vertical detail -- vetoing it on
        // that divergence just re-imposes the field's limits on the frame.  So
        // this old metric is disabled here, matching the interlace-only field
        // majority guard further down (already gated on !localUseFrameModel).
        bool frameInsane = !localUseFrameModel &&
                           (frameModelDistIRE > FRAME_MAX_DIST_IRE);

        // Management veto is consumed here, but its construction stays outside FVF.
        bool managementVeto = (cadenceId == -2);

        bool b2VertCoherent = !managementVeto && !frameInsane;
        double targetModel = localUseFrameModel ? FR_s : FB_s;

        double diff_fvf_ire = diff_stack_ire;
        diffFVF[rel] = diff_fvf_ire;

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
        double hIRE = (rel < (int)tapLine.hLumaDeltaIRE.size())
            ? tapLine.hLumaDeltaIRE[rel]
            : 0.0;


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
        metrics.interfieldDistinctIRE = 0.0;
        metrics.frameToFieldModelIRE = diff_candB_ire;
        metrics.frameToBestFieldIRE = frameFieldCandidateDistIRE;
        metrics.frameModel = localUseFrameModel;
        metrics.managementVeto = managementVeto;
        metrics.frameVertCoherent = b2VertCoherent;

        int    idx   = 1;
        double val   = FB;
        float  shade = 0.35f;
        double scoreA = std::numeric_limits<double>::quiet_NaN();
        double scoreB = std::numeric_limits<double>::quiet_NaN();
        double scoreR = std::numeric_limits<double>::quiet_NaN();

        {
            double devA = 0.0, devB = 0.0, devR = 0.0;

            if (T.FVF_SHAPE_STRENGTH > 0.0) {
                double m_c = targetModel;
                auto getM = [&](int r) {
                    if (localUseFrameModel)
                        return frameB2[std::clamp(r, 0, width - 1)];
                    else
                        return fieldB[std::clamp(r, 0, width - 1)];
                };
                double m_l = getM(rel - 1);
                double m_r = getM(rel + 1);
                double shapeModel = m_c - 0.5 * (m_l + m_r);

                auto getShapeScore = [&](double v, double v_l, double v_r) {
                    double shapeVal = v - 0.5 * (v_l + v_r);
                    return std::fabs(shapeVal - shapeModel);
                };

                double FA_l = candidateA[std::clamp(rel - 1, 0, width - 1)];
                double FA_r = candidateA[std::clamp(rel + 1, 0, width - 1)];
                double FB_l = fieldB[std::clamp(rel - 1, 0, width - 1)];
                double FB_r = fieldB[std::clamp(rel + 1, 0, width - 1)];
                double FR_l = frameB2[std::clamp(rel - 1, 0, width - 1)];
                double FR_r = frameB2[std::clamp(rel + 1, 0, width - 1)];

                devA += getShapeScore(FA_s, FA_l, FA_r) * T.FVF_SHAPE_STRENGTH;
                devB += getShapeScore(FB_s, FB_l, FB_r) * T.FVF_SHAPE_STRENGTH;
                devR += getShapeScore(FR_s, FR_l, FR_r) * T.FVF_SHAPE_STRENGTH;
            }

            double satScale = std::clamp((chromaMagIRE - 2.0) / 8.0, 0.0, 1.0);

            double errA_notch = std::fabs(lumFA);
            double errB_notch = std::fabs(lumFB);
            double errR_notch = std::fabs(lumFR);

            scoreA = (1.0 - satScale) * devA + satScale * errA_notch;
            scoreB = (1.0 - satScale) * devB + satScale * errB_notch;
            scoreR = (1.0 - satScale) * devR + satScale * errR_notch;

            // Candidate-A special gating is removed. The same-regime buddy
            // competes through the general candidate machinery only.

            // ------------------------------------------------------------
            // Model-aware regime scoring.
            // Progressive protects the frame regime: Frame A participates,
            // and only Field B is measured against the Frame B model.
            // Interlace treats Field B as the model, lets Field A participate
            // on its own image-shaping merits, and gives Frame B only a small
            // score bump when it is very close to Field B.
            // ------------------------------------------------------------
            if (localUseFrameModel) {
                const double modelDistanceIRE = std::max(
                    0.0,
                    diff_candB_ire -
                        std::max(0.0, T.FVF_MODEL_PRIMARY_DEADBAND_IRE));
                scoreB += T.FVF_MODEL_PRIMARY_WEIGHT * modelDistanceIRE;
                if (!managementVeto && b2VertCoherent) {
                    scoreR *= FRAME_MODEL_BIAS_LOCAL;
                }
            } else {
                const double closeFrameBonus = std::clamp(
                    1.0 - (diff_candB_ire / std::max(1e-9, T.FVF_SMALL_DIFF_IRE)),
                    0.0, 1.0);
                scoreB *= T.FIELD_MODEL_BIAS;
                scoreR += T.FVF_MODEL_PRIMARY_WEIGHT * diff_candB_ire;
                scoreR *= T.FRAME_IN_INTERLACE_PENALTY;
                scoreR *= (1.0 - 0.08 * closeFrameBonus);
            }

            if (haveFrameIQForMag && rel < frameIQN) {
                // Pre-computed magnitudes from scratch_fvf_iqMag; index clamped
                // to [0, width-1] as the original lambda did.
                const double *iqMagArr = scratch_fvf_iqMag.data();
                auto mag = [&](int r) -> double {
                    return iqMagArr[std::clamp(r, 0, width - 1)];
                };

                const double mRel = mag(rel);
                const double fine   = std::fabs(mRel - 0.5 * (mag(rel - 1) + mag(rel + 1)));
                const double mid    = std::fabs(mRel - 0.5 * (mag(rel - 2) + mag(rel + 2)));
                const double coarse = std::fabs(mRel - 0.5 * (mag(rel - 4) + mag(rel + 4)));

                const double denom = fine + mid + coarse + 1e-9;
                const double fineFrac   = fine   / denom;
                const double midFrac    = mid    / denom;
                const double coarseFrac = coarse / denom;
                metrics.iqFineFrac = fineFrac;
                metrics.iqMidFrac = midFrac;
                metrics.iqCoarseFrac = coarseFrac;
                metrics.iqCoherence = 1.0 - std::clamp(coarseFrac, 0.0, 1.0);

                const double frameScaleBiasStrength = localUseFrameModel
                    ? FRAME_SCALE_BIAS_STRENGTH_PROGRESSIVE_LOCAL
                    : FRAME_SCALE_BIAS_STRENGTH_INTERLACE_LOCAL;
                const double FRAME_COARSE_CLAMP     = 0.60;
                const double FIELD_SWITCH_STRENGTH  = 0.10;

                const bool fineDominant = (fineFrac > (midFrac + coarseFrac) + 0.10);

                double frameBonus = frameScaleBiasStrength * fineFrac;
                frameBonus *= (1.0 - FRAME_COARSE_CLAMP * coarseFrac);
                scoreR *= (1.0 - frameBonus);

                if (!fineDominant) {
                    const double bias = std::clamp(coarseFrac - midFrac, -1.0, 1.0);
                    scoreB *= (1.0 + FIELD_SWITCH_STRENGTH * bias);
                }
            }

            // ------------------------------------------------------------
            // Saturation regime: in highly saturated regions, Frame is often
            // the least visually toxic when coherent, but Field B tends to
            // introduce zipper/alternation more readily than Field A.
            // Apply a soft bias rather than a hard override.
            // ------------------------------------------------------------
            if (sat_t > 0.0) {
                // Frame A gets a mild saturation penalty (underperforms there)
                // but no reward — regime-neutral on the upside.  Field B gets
                // the heavier penalty; Frame B gets the coherent-frame reward.
                const double SAT_FIELD_A_PEN = 0.06;
                const double SAT_FIELD_B_PEN = 0.14;
                scoreA *= (1.0 + SAT_FIELD_A_PEN * sat_t);
                scoreB *= (1.0 + SAT_FIELD_B_PEN * sat_t);

                if (!managementVeto && b2VertCoherent && !frameInsane) {
                    const double SAT_FRAME_BONUS = 0.18;
                    scoreR *= (1.0 - SAT_FRAME_BONUS * sat_t);
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
                constexpr int EDGE_PROBE_FAR  = 6;
                const bool canEval =
                    (hIRE >= 0.75 * HEDGE_THRESH_IRE) &&
                    (rel >= (EDGE_GAP + EDGE_PROBE_FAR)) &&
                    (rel + (EDGE_GAP + EDGE_PROBE_FAR) < width) &&
                    (line >= firstLine && line < lastLine);

                if (canEval) {
                    auto srcNotch = [&](int r)->double {
                        return notchSource[std::clamp(r, 0, width - 1)];
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
                        const double *notch = arr
                            ? (arr == candidateAData ? notchCandidateA.data() : notchFieldB.data())
                            : notchFrame.data();
                        const double m2 = notch[rm2];
                        const double p2 = notch[rp2];
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
                        // User's transition-crossing award: reward a comb
                        // that crosses a regional transition along the line
                        // MORE QUICKLY. It buys edge clarity without the
                        // usual sharpening costs (halos, instability)
                        // precisely because it is a SCORING term -- it can
                        // only change which existing candidate wins, never
                        // manufacture an edge that no candidate produced.
                        //
                        // Boosted 2026-08-02 (user): uncovered frames carry a
                        // lateral-motion-blur-like softness, and the
                        // covered/uncovered quality gap is what strobes. This
                        // is the safe lever on it.
                        //
                        // Also wires the DECLARED tunable, which was dead: a
                        // hardcoded local shadowed
                        // FVF_TRANSITION_SHARPNESS_WEIGHT so the knob in
                        // comb.h did nothing. LDCD_FVF_SHARP_W overrides.
                        static const double kSharpWEnv = []{
                            const char *e = std::getenv("LDCD_FVF_SHARP_W");
                            return e ? std::atof(e) : -1.0;
                        }();
                        const double W_EDGE_SHARP = (kSharpWEnv >= 0.0)
                            ? kSharpWEnv
                            : T.FVF_TRANSITION_SHARPNESS_WEIGHT;
                        score *= (1.0 - W_EDGE_SHARP * sharp * stepStrength);
                    };

                    applySharpReward(scoreA, candidateAData, nullptr);
                    applySharpReward(scoreB, fieldB, nullptr);
                    applySharpReward(scoreR, nullptr, &frameB2);
                }
                no_sharp_reward: ;
            }
            
            // Attribution alignment scoring.
            // Attribution alignment scoring.
            //
            // This is not checkerboard suppression. Checkerboards are pathologies to fix
            // at their source. Here attribution only biases the candidate contest:
            //   - chroma-claim evidence rewards candidates that align with chroma strength;
            //   - luma-incursion / cross-color evidence penalizes candidates that depart
            //     from the 1D rail in luma-claimed regions.
            {
                if (attrRow) {
                    const auto &facts = attrRow[rel].facts;
                    const auto &ass   = attrRow[rel].assessment;

                    const double attrAuthority = attrAuthorityRow
                        ? attrAuthorityRow[rel]
                        : std::clamp(
                            1.0 -
                                0.65 * std::clamp(ass.attributionConflict, 0.0, 1.0) -
                                0.35 * std::clamp(ass.uncertainClaim, 0.0, 1.0),
                            0.0,
                            1.0);

                    if (attrAuthority > 0.0) {
                        const double chromaClaim = std::clamp(ass.chromaClaim, 0.0, 1.0);
                        const double lumaClaim   = std::clamp(ass.lumaClaim,   0.0, 1.0);
                        const double aMag = std::fabs(FA) * invI;
                        const double bMag = std::fabs(FB) * invI;
                        const double rMag = std::fabs(FR) * invI;
                        const double cMag = std::fabs(L1) * invI;
            
                        const double factA = std::max(0.0, facts.fieldAChromaIRE);
                        const double factB = std::max(0.0, facts.fieldBChromaIRE);
                        const double factR = std::max(0.0, facts.frameChromaIRE);
                        const double factC = std::max(0.0, facts.locked1DChromaIRE);
            
                        const double maxFact =
                            std::max(std::max(factA, factB), std::max(factR, factC));
            
                        // Chroma-claim alignment:
                        // Reward candidates whose chroma magnitude agrees with the
                        // attribution facts for that candidate family.
                        if (chromaClaim > 0.0 && maxFact > 1e-9) {
                            const double denomA = std::max(3.0, std::max(aMag, factA));
                            const double denomB = std::max(3.0, std::max(bMag, factB));
                            const double denomR = std::max(3.0, std::max(rMag, factR));
            
                            const double aAlign =
                                1.0 - std::clamp(std::fabs(aMag - factA) / denomA, 0.0, 1.0);
                            const double bAlign =
                                1.0 - std::clamp(std::fabs(bMag - factB) / denomB, 0.0, 1.0);
                            const double rAlign =
                                1.0 - std::clamp(std::fabs(rMag - factR) / denomR, 0.0, 1.0);
            
                            const double chromaReward =
                                0.12 * attrAuthority * chromaClaim;
            
                            scoreA *= (1.0 - chromaReward * aAlign);
                            scoreB *= (1.0 - chromaReward * bAlign);
                            scoreR *= (1.0 - chromaReward * rAlign);
                        }
            
                        // Cross-color / luma-incursion pressure:
                        // This is the detection side you asked about. It enters as a
                        // luma-pressure term, not as checkerboard suppression.
                        const double incursionFromFacts = std::clamp(
                            std::max(facts.lumaIncursionRiskIRE,
                                     facts.lumaExcursionIRE) / 12.0,
                            0.0, 1.0);
            
                        const double parallaxPressure = std::clamp(
                            std::max(facts.carrierParallaxLatticeRiskIRE,
                                     facts.carrierParallaxYSpreadIRE) / 12.0,
                            0.0, 1.0);
            
                        const double residualPressure = std::clamp(
                            std::max(facts.residualFitErrorIRE,
                                     ass.lumaResidual) / 12.0,
                            0.0, 1.0);
            
                        const double crossColorPressure = std::clamp(
                            std::max(incursionFromFacts,
                                     std::max(parallaxPressure, residualPressure)),
                            0.0, 1.0);
            
                        const double lumaPressure = std::clamp(
                            std::max(lumaClaim,
                                     std::max(ass.lumaRisk,
                                              std::max(ass.lumaShapeContinuation,
                                                       crossColorPressure))),
                            0.0, 1.0);
            
                        if (lumaPressure > 0.0) {
                            // Penalize candidates that move away from the 1D rail in a
                            // luma-claimed / cross-color-claimed region.
                            const double denom = std::max(3.0, cMag);
            
                            const double aTrespass =
                                std::clamp(std::fabs(aMag - cMag) / denom, 0.0, 1.0);
                            const double bTrespass =
                                std::clamp(std::fabs(bMag - cMag) / denom, 0.0, 1.0);
                            const double rTrespass =
                                std::clamp(std::fabs(rMag - cMag) / denom, 0.0, 1.0);
            
                            const double lumaPenalty =
                                0.16 * attrAuthority * lumaPressure;
            
                            scoreA *= (1.0 + lumaPenalty * aTrespass);
                            scoreB *= (1.0 + lumaPenalty * bTrespass);
                            scoreR *= (1.0 + lumaPenalty * rTrespass);
                        }
                    }
                }
            }

			// ------------------------------------------------------------
            // Immediate-neighbor anchor scoring.
            //
            // This is image-local neighbor shaping, not same-phase carrier
            // smoothing.  It biases the election toward candidates that agree
            // with the immediate local neighborhood before output is formed.
            // The anchor is only a scoring reference; it is not an output
            // plane and it does not replace the elected candidate.
            // ------------------------------------------------------------
            {
                const int xm1 = std::max(0, rel - 1);
                const int xp1 = std::min(width - 1, rel + 1);

                const double aAnchor = 0.5 * (candidateAData[xm1] + candidateAData[xp1]);
                const double bAnchor = 0.5 * (fieldB[xm1]     + fieldB[xp1]);
                const double rAnchor = 0.5 * (frameB2[xm1]    + frameB2[xp1]);
                const double cAnchor = 0.5 * (sample1D(xm1)   + sample1D(xp1));

                const double sumAnchor = aAnchor + bAnchor + rAnchor + cAnchor;
                const double minAnchor = std::min(std::min(aAnchor, bAnchor),
                                                  std::min(rAnchor, cAnchor));
                const double maxAnchor = std::max(std::max(aAnchor, bAnchor),
                                                  std::max(rAnchor, cAnchor));

                // Trimmed mean of the four local anchors.  This avoids making
                // any single candidate plane self-certifying while remaining
                // cheaper than a sort/median pass.
                const double neighborAnchor = 0.5 * (sumAnchor - minAnchor - maxAnchor);

                auto anchorPenalty = [](double dIRE) -> double {
                    const double LO_IRE = 0.75;
                    const double HI_IRE = 5.50;
                    return std::clamp((dIRE - LO_IRE) /
                                      (HI_IRE - LO_IRE), 0.0, 1.0);
                };

                // Do not let the anchor act like a blur across a strong local
                // transition.  The transition-sharpness subsystem handles those
                // cases; this anchor is for local selection stability/clarity.
                const double localStepIRE = std::fabs(sample1D(xp1) - sample1D(xm1)) * invI;
                double anchorAuthority = 1.0 -
                    std::clamp((localStepIRE - 4.0) / (12.0 - 4.0), 0.0, 1.0);

                // In strong saturation, keep some influence but do not let the
                // anchor fight saturation policy.
                anchorAuthority *= (1.0 - 0.35 * sat_t);

                if (anchorAuthority > 0.0) {
                    const double W_NEIGHBOR_ANCHOR = 0.10;
                    const double wAnchor = W_NEIGHBOR_ANCHOR * anchorAuthority;

                    const double dA = std::fabs(FA - neighborAnchor) * invI;
                    const double dB = std::fabs(FB - neighborAnchor) * invI;
                    const double dR = std::fabs(FR - neighborAnchor) * invI;

                    scoreA *= (1.0 + wAnchor * anchorPenalty(dA));
                    scoreB *= (1.0 + wAnchor * anchorPenalty(dB));
                    scoreR *= (1.0 + wAnchor * anchorPenalty(dR));
                }
            }
            // ------------------------------------------------------------
            // Impulse scoring.
            //
            // Isolated narrow luma peaks (stars in black sky) produce false
            // chroma in the field combs because 1D reads the sub-carrier-
            // period spike as carrier. The frame comb can cancel the error via
            // interfield differencing. Reward low chroma magnitude at
            // impulse sites (any candidate that stays near zero is right)
            // and give Frame a slight bonus since its mechanism is correct.
            // ------------------------------------------------------------
            {
                const double impulseT =
                    (attrRow && rel < width)
                        ? std::clamp(attrRow[rel].facts.lumaImpulseRisk, 0.0, 1.0)
                        : 0.0;
            
                if (impulseT > 0.0) {
                    const double aM = std::fabs(FA) * invI;
                    const double bM = std::fabs(FB) * invI;
                    const double rM = std::fabs(FR) * invI;
            
                    const double minM = std::min({aM, bM, rM});
            
                    constexpr double IMPULSE_RELATIVE_IQ_PEN = 0.85;
                    constexpr double IMPULSE_RESIDUE_PEN     = 0.20;
                    constexpr double IMPULSE_FRAME_BONUS     = 0.06;
            
                    auto impulsePenalty = [&](double m) {
                        const double aboveBest = std::clamp((m - minM) / 3.0, 0.0, 1.0);
                        const double residue   = std::clamp(m / 5.0, 0.0, 1.0);
            
                        return 1.0 + impulseT * (
                            IMPULSE_RELATIVE_IQ_PEN * aboveBest +
                            IMPULSE_RESIDUE_PEN     * residue
                        );
                    };
            
                    scoreA *= impulsePenalty(aM);
                    scoreB *= impulsePenalty(bM);
                    scoreR *= impulsePenalty(rM);
            
                    if (!frameInsane && !managementVeto && rM <= minM + 0.25) {
                        scoreR *= (1.0 - IMPULSE_FRAME_BONUS * impulseT);
                    }
                }
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
                    if (dA1 + ONE_D_NEAR_THRESH_IRE_LOCAL < dB1) {
                        scoreA *= (1.0 - biasScale);
                        scoreB *= (1.0 + biasScale);
                    } else if (dB1 + ONE_D_NEAR_THRESH_IRE_LOCAL < dA1) {
                        scoreB *= (1.0 - biasScale);
                        scoreA *= (1.0 + biasScale);
                    }
                }
            }

            auto pickCandidate = [&](int candIdx, double candVal, float candShade) {
                idx   = candIdx;
                val   = candVal;
                shade = candShade;
            };

            // Context may establish that Frame B is safe and relevant, but it
            // must not settle the election by itself.  The candidate still has
            // to beat both interfield combs on the accumulated merit score.
            const bool frameHasBestScore =
                !frameInsane && !managementVeto &&
                scoreR + 1e-12 < scoreA &&
                scoreR + 1e-12 < scoreB;

            if (hIRE > HEDGE_THRESH_IRE && diff_stack_ire > 5.0) {
                double dF1 = std::fabs(lumFR - L1) * invI;
                if (dF1 <= 3.5 && frameFieldCandidateDistIRE <= 5.0 &&
                    frameHasBestScore)
                    pickCandidate(2, FR, 0.75f);
                else {
                    if (scoreA < scoreB) pickCandidate(0, FA, 0.25f);
                    else                 pickCandidate(1, FB, 0.35f);
                }
            } else if (chromaMagIRE > CHROMA_STRONG_IRE && vIRE > VERT_THRESH_IRE) {
                // Strong chroma with vertical contrast indicates per-line alternation
                // that Frame is well-suited to suppress.  It may enter without
                // the normal decisive margin, but the interfield candidates
                // remain live choices when their merit scores are better.
                if (frameHasBestScore)
                    pickCandidate(2, FR, 0.8f);
                else {
                    if (scoreA <= scoreB) pickCandidate(0, FA, 0.25f);
                    else                  pickCandidate(1, FB, 0.35f);
                }
            } else {
                if (scoreR + 1e-12 < scoreA * 0.85 &&
                         scoreR + 1e-12 < scoreB * 0.85)
                    pickCandidate(2, FR, 0.8f);
                else if (scoreA < scoreB * 0.8)
                    pickCandidate(0, FA, 0.25f);
                else {
                    double dFL = std::fabs(lumFB - L1) * invI;
                    double dRL = std::fabs(lumFR - L1) * invI;
                    if (frameHasBestScore && dRL + 3.0 < dFL)
                        pickCandidate(2, FR, 0.75f);
                    else
                        pickCandidate(1, FB, 0.35f);
                }
            }

            // Subtle hysteresis (switch veto) in soft regions
            if (rel > 0) {
                const int prevIdx = winner[rel - 1];

                if (prevIdx >= 0 && prevIdx <= 2 && idx != prevIdx) {

                    const bool hystOk =
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

                        {
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
        if      (idx == 2) frameCountTotal++;
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
            const double hEdgeIRE = (rel < (int)tapLine.hLumaDeltaIRE.size())
                ? tapLine.hLumaDeltaIRE[rel]
                : 0.0;
            if (hEdgeIRE > EDGE_STOP_IRE) continue;
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
                if      (idx == 0) { outVal[rel] = candidateA[rel]; outShade[rel] = 0.25f; }
                else if (idx == 1) { outVal[rel] = fieldB[rel];  outShade[rel] = 0.35f; }
                else               { outVal[rel] = frameB2[rel]; outShade[rel] = 0.8f;  }
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
                        if (blockIdx == 0) { outVal[r] = candidateA[r]; outShade[r] = 0.25f; }
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
    const double *frameScalarData = haveFrameScalar ? frameScalar.data() : nullptr;
    const bool frameScalarWideEnough =
        haveFrameScalar && (int)frameScalar.size() >= width;

    // Pre-compute |frameIQ[r]| magnitudes once, then derive coherence from
    // additions instead of redundant hypot calls.  Old path: 4 hypot/pixel in
    // coherence + 1 in the main loop = 5×width.  New path: 1 for the mag
    // pre-pass + 1 for the vector-sum magnitude = 2×width; main loop reuses
    // the pre-computed mag for frameChromaIRE (0 additional).
    const bool haveFrameIQ = frameIQ && !frameIQ->empty();
    const int iqN = haveFrameIQ ? (int)frameIQ->size() : 0;
    const std::complex<double> *frameIQData = haveFrameIQ ? frameIQ->data() : nullptr;

    if ((int)scratch_coe_frameIQMag.size() != width)
        scratch_coe_frameIQMag.resize(width);
    if ((int)scratch_attrWideCarrier.size() != width)
        scratch_attrWideCarrier.resize(width);
    if ((int)scratch_attrBandYClaim.size() != width)
        scratch_attrBandYClaim.resize(width);
    if ((int)scratch_attrMembershipY.size() != width)
        scratch_attrMembershipY.resize(width);
    if (haveFrameIQ) {
        const int n = std::min(width, iqN);
        for (int r = 0; r < n; ++r) {
            const auto &z = frameIQData[r];
            scratch_coe_frameIQMag[r] = boundedMag(z);
        }
        const double tailMag = (n > 0) ? scratch_coe_frameIQMag[n - 1] : 0.0;
        std::fill(scratch_coe_frameIQMag.begin() + n,
                  scratch_coe_frameIQMag.begin() + width,
                  tailMag);
    } else {
        std::fill(scratch_coe_frameIQMag.begin(),
                  scratch_coe_frameIQMag.begin() + width, 0.0);
    }

    // IQ coherence pre-pass: line-level mean gates per-sample values — a
    // globally incoherent line cannot inflate isolated samples.
    if ((int)scratch_coe_coherence.size() != width)
        scratch_coe_coherence.resize(width);
    std::fill(scratch_coe_coherence.begin(), scratch_coe_coherence.begin() + width, 0.0);
    double lineMeanFrameCoherence = 0.0;
    if (haveFrameIQ) {
        for (int r = 0; r < width; ++r) {
            const int rm2 = std::clamp(r - 2, 0, iqN - 1);
            const int rr  = std::clamp(r,     0, iqN - 1);
            const int rp2 = std::clamp(r + 2, 0, iqN - 1);
            const double magSum = scratch_coe_frameIQMag[std::clamp(r - 2, 0, width - 1)]
                                + scratch_coe_frameIQMag[r]
                                + scratch_coe_frameIQMag[std::clamp(r + 2, 0, width - 1)];
            const std::complex<double> sum =
                frameIQData[rm2] + frameIQData[rr] + frameIQData[rp2];
            scratch_coe_coherence[r] = (magSum > 1e-9)
                ? std::clamp(boundedMag(sum) / magSum, 0.0, 1.0)
                : 0.0;
            lineMeanFrameCoherence += scratch_coe_coherence[r];
        }
    }
    if (width > 0) lineMeanFrameCoherence /= static_cast<double>(width);

    for (int rel = 0; rel < width; ++rel) {
        AttributionEvidence &e = row[rel];
        AttributionFacts &f = e.facts;

        const double fa = fieldA[rel];
        const double fb = fieldB[rel];
        const double fr = frameScalarWideEnough
            ? frameScalarData[rel]
            : (haveFrameScalar
                ? frameScalarData[std::clamp(rel, 0, (int)frameScalar.size() - 1)]
                : 0.0);

        f.fieldAChromaIRE = std::fabs(fa) * invIreScale;
        f.fieldBChromaIRE = std::fabs(fb) * invIreScale;

        f.frameChromaIRE = haveFrameIQ
            ? scratch_coe_frameIQMag[rel] * invIreScale
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

        scratch_attrWideCarrier[rel] = std::max(f.bandpassMidIRE, f.bandpassCoarseIRE);
        scratch_attrBandYClaim[rel] = f.lumaExcursionIRE;

        f.lumaImpulseRisk =
            (rel < (int)scratch_impulseExempt.size())
                ? scratch_impulseExempt[rel]
                : 0.0;

    }

    // Carrier prior + finalize in a single pass over the row.
    const CombCarrierGrammar *lineGrammar = carrierGrammarLine(line);
    const double lineCarrierPrior = carrierPlausibility(lineGrammar);
    const double lineForwardErrorIRE = (lineGrammar && lineGrammar->projectionValid)
        ? lineGrammar->meanForwardErrorIRE
        : 0.0;

    for (int rel = 0; rel < width; ++rel) {
        const int rm4 = std::max(0, rel - 4);
        const int rp4 = std::min(width - 1, rel + 4);
        AttributionEvidence &e = row[rel];
        e.assessment.carrierPrior = lineCarrierPrior;

        finalizeAttributionClaims(
            e,
            0.5 * (scratch_attrBandYClaim[rm4] + scratch_attrBandYClaim[rp4]),
            0.5 * (scratch_attrWideCarrier[rm4] + scratch_attrWideCarrier[rp4]),
            lineForwardErrorIRE);
        scratch_attrMembershipY[rel] = std::clamp(
            1.0 -
                0.65 * e.assessment.attributionConflict -
                0.35 * e.assessment.uncertainClaim,
            0.0,
            1.0);
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

    // Default coarse: one raster-aligned 4fSC-cycle mean, repeated across the
    // four samples it owns.  This is the inexpensive coherent-Y basis.  A
    // sliding mean is not a harmless refinement here: it is the unsharpened
    // half of the witness lurch model, costs another whole-line construction,
    // and changes the HF residual's zero from sample to sample.  The witness
    // builds its sliding/lurch-sharpened coarse separately and explicitly.
    if (width < 4) {
        double avg = 0.0;
        for (int x = 0; x < width; ++x)
            avg += static_cast<double>(rawLine[left + x]);
        avg /= static_cast<double>(width);
        for (int x = 0; x < width; ++x) {
            if (baseY4) baseY4[x] = avg;
            if (hiRaw) hiRaw[x] = static_cast<double>(rawLine[left + x]) - avg;
            if (lumaSmooth) lumaSmooth[x] = avg;
        }
        return;
    }

    int p = 0;
    for (; p + 3 < width; p += 4) {
        const double y = 0.25 *
            (static_cast<double>(rawLine[left + p + 0]) +
             static_cast<double>(rawLine[left + p + 1]) +
             static_cast<double>(rawLine[left + p + 2]) +
             static_cast<double>(rawLine[left + p + 3]));
        for (int k = 0; k < 4; ++k) {
            if (baseY4) baseY4[p + k] = y;
            if (hiRaw)
                hiRaw[p + k] = static_cast<double>(rawLine[left + p + k]) - y;
        }
    }

    // Active width normally contains complete 4fSC cycles.  Keep the tail on
    // the final complete cycle if metadata presents an odd width.
    if (p < width) {
        const int tb = width - 4;
        const double y = 0.25 *
            (static_cast<double>(rawLine[left + tb + 0]) +
             static_cast<double>(rawLine[left + tb + 1]) +
             static_cast<double>(rawLine[left + tb + 2]) +
             static_cast<double>(rawLine[left + tb + 3]));
        for (int x = p; x < width; ++x) {
            if (baseY4) baseY4[x] = y;
            if (hiRaw)
                hiRaw[x] = static_cast<double>(rawLine[left + x]) - y;
        }
    }

    if (!lumaSmooth)
        return;

    // Legacy block-centre scaffold for geometry-only consumers.  Reuse the
    // already-built coarse instead of averaging raw a second time.
    auto blockAvg = [&](int block)->double {
        const int x0 = std::clamp(block * 4, 0, std::max(0, width - 4));
        if (baseY4)
            return baseY4[x0];
        return 0.25 *
            (static_cast<double>(rawLine[left + x0 + 0]) +
             static_cast<double>(rawLine[left + x0 + 1]) +
             static_cast<double>(rawLine[left + x0 + 2]) +
             static_cast<double>(rawLine[left + x0 + 3]));
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

// split2D dispatcher
void Comb::FrameBuffer::split2D()
{
    const bool writeWeights = configuration.showMap;
    const bool wantFvf = (configuration.twoDVariant == Comb::Configuration::TwoDVariant::FieldVsFrame);
    const bool fvfUseFrameModel = wantFvf && configuration.phaseCompensation &&
        (cadenceId >= 0 || cadenceId == -3);
    const bool needFrameACompute = configuration.phaseCompensation &&
        (configuration.twoDVariant == Comb::Configuration::TwoDVariant::FrameAAdaptiveIQ ||
         fvfUseFrameModel);
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
        static const bool useLockedRawBandpassLine = [] {
            const char *s = std::getenv("LD_LINE_USE_LOCKED_RAW_BANDPASS");
            return s && std::atoi(s) != 0;
        }();
        for (int line = firstLine; line < lastLine; ++line) {
            double *dst = clpbuffer[1].pixel[line];
            const double *lockedRow = nullptr;
            if (configuration.phaseCompensation) {
                lockedRow = useLockedRawBandpassLine
                    ? locked1DRawBandpass_line(line)
                    : combSource1D_line(line);
            }
            if (lockedRow) {
                for (int rel = 0; rel < width; ++rel) dst[left + rel] = lockedRow[rel];
            } else {
                const double *src1d = bucketScalar1D_line(line);
                for (int rel = 0; rel < width; ++rel) dst[left + rel] = src1d[left + rel];
            }
            if (writeWeights && line < (int)w2d_frame_weight.size())
                std::fill(w2d_frame_weight[line].begin(), w2d_frame_weight[line].end(), 0.0f);
        }
        return;
    }

    // LDCD_OLD_SPLIT2D=1: diagnostic A/B only. Runs the classic ld-decode
    // adaptive 2D kernel (pre-locked, pre-region-grammar) verbatim on the
    // blind 1D bandpass, bypassing every modern protection. Purpose: render
    // the UNPROTECTED error catalogue (e.g. the bikini-bottom upper shadow)
    // that the chroma-boundary band was installed to quash, so the 2D
    // threshold revisit can compare against what the guards actually buy.
    static const bool oldSplit2D = []{
        const char *s = std::getenv("LDCD_OLD_SPLIT2D");
        return s && std::atoi(s) != 0;
    }();
    if (oldSplit2D) {
        static constexpr double blackLine[MAX_WIDTH] = {0};
        for (int lineNumber = firstLine; lineNumber < lastLine; lineNumber++) {
            const double *previousLine = blackLine;
            if (lineNumber - 2 >= firstLine)
                previousLine = clpbuffer[0].pixel[lineNumber - 2];
            const double *currentLine = clpbuffer[0].pixel[lineNumber];
            const double *nextLine = blackLine;
            if (lineNumber + 2 < lastLine)
                nextLine = clpbuffer[0].pixel[lineNumber + 2];

            for (int h = left; h < right; h++) {
                double kp, kn;

                kp  = fabs(fabs(currentLine[h]) - fabs(previousLine[h]));
                kp += fabs(fabs(currentLine[h - 1]) - fabs(previousLine[h - 1]));
                kp -= (fabs(currentLine[h]) + fabs(previousLine[h - 1])) * .10;
                kn  = fabs(fabs(currentLine[h]) - fabs(nextLine[h]));
                kn += fabs(fabs(currentLine[h - 1]) - fabs(nextLine[h - 1]));
                kn -= (fabs(currentLine[h]) + fabs(nextLine[h - 1])) * .10;

                const double kRange = 45 * irescale;
                kp = qBound(0.0, 1 - (kp / kRange), 1.0);
                kn = qBound(0.0, 1 - (kn / kRange), 1.0);

                double sc = 1.0;

                if ((kn > 0) || (kp > 0)) {
                    if (kn > (3 * kp)) kp = 0;
                    else if (kp > (3 * kn)) kn = 0;

                    sc = (2.0 / (kn + kp));
                    if (sc < 1.0) sc = 1.0;
                } else {
                    if ((fabs(fabs(previousLine[h]) - fabs(nextLine[h])) -
                         fabs((nextLine[h] + previousLine[h]) * .2)) <= 0) {
                        kn = kp = 1;
                    }
                }

                double tc1;
                tc1  = ((currentLine[h] - previousLine[h]) * kp * sc);
                tc1 += ((currentLine[h] - nextLine[h]) * kn * sc);
                tc1 /= 4;

                clpbuffer[1].pixel[lineNumber][h] = tc1;
            }
            if (writeWeights && lineNumber < (int)w2d_frame_weight.size())
                std::fill(w2d_frame_weight[lineNumber].begin(),
                          w2d_frame_weight[lineNumber].end(), 0.0f);
        }
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
            combTapBuildFlags_ = TapBuildFrame | TapBuildFieldB;
            break;
        case V::FrameBDirectIQ:
            combTapBuildFlags_ = TapBuildFrame | TapBuildFieldB;
            break;
        case V::FieldVsFrame:
            combTapBuildFlags_ = fvfUseFrameModel
                ? (TapBuildFrame | TapBuildFieldB)
                : TapBuildAll;
            break;
        default:
            combTapBuildFlags_ = TapBuildAll;
            break;
        }
    }


    std::vector<std::complex<double>> frameIQ;
    std::vector<std::complex<double>> frameAIQ;
    for (int line = firstLine; line < lastLine; ++line) {
        if (line >= demodLines) continue;

        ensureCombTapLine(line);
        const CombTapLine &tapLine = tapLineCache[precleanRingSlot(line)];

        auto ensureFieldBPrecleanLine = [&](int ln) {
            if (ln < firstLine || ln >= lastLine) return;
            if (havePrecleanLine(ln, width)) return;
            double *preclean = precleanLinePtrMutable(ln, width);
            const CombTapLine *precleanTapLine = &ensureCombTapLine(ln);
            computeFieldBLine(*precleanTapLine,
                                   preclean,
                                   writeWeights ? fieldBDecisionReason_line(ln) : nullptr);
        };

        if (combTapBuildFlags_ & TapBuildFieldB) {
            const double *fieldBPreclean = precleanLinePtr(line, width);
            if (fieldBPreclean) {
                std::copy(fieldBPreclean, fieldBPreclean + width, scratch_lineWorkC.begin());
            } else {
                computeFieldBLine(tapLine,
                                       scratch_lineWorkC.data(),
                                       writeWeights ? fieldBDecisionReason_line(line)
                                                    : nullptr);
            }
        } else {
            std::fill(scratch_lineWorkC.begin(), scratch_lineWorkC.begin() + width, 0.0);
        }

        if (needFrameIQCompute) {
            // Frame B should always see the same C line that split2D produced,
            // even when Field B's +/-2 reach cedes to the local center value.
            double *preclean = precleanLinePtrMutable(line, width);
            std::copy(scratch_lineWorkC.begin(), scratch_lineWorkC.begin() + width, preclean);
        }
        if (needFrameIQCompute) {
            ensureFieldBPrecleanLine(line - 1);
            ensureFieldBPrecleanLine(line + 1);
        }

        if (combTapBuildFlags_ & TapBuildFieldA) {
            computeFieldALine(tapLine, scratch_lineWorkA.data(), scratch_lineWorkB.data());
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
                const double *lockedRow = combSource1D_line(line);
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

        if (needFrameACompute &&
            certifiedOneDLevel() >= 2 && certifiedDefLine(line)) {
            // Certified cede (CONSTRUCTION, upstream of every election):
            // on a def line the Frame A candidate IS the center. The 4fsc
            // IQ pair comes from the stage-1 locked products of the same
            // certified scalar, so attribution consumers see one story.
            const double *center = locked1DSource_line(line);
            const float *cI4 = locked1DTI4fsc_line(line);
            const float *cQ4 = locked1DTQ4fsc_line(line);
            frameAIQ.assign(width, std::complex<double>(0.0, 0.0));
            if ((int)scratch_frameAAdaptiveIQComposite.size() < width)
                scratch_frameAAdaptiveIQComposite.resize(width);
            for (int rel = 0; rel < width; ++rel) {
                scratch_frameAAdaptiveIQComposite[rel] =
                    center ? center[rel] : 0.0;
                if (cI4 && cQ4)
                    frameAIQ[rel] = std::complex<double>(
                        (double)cI4[rel], (double)cQ4[rel]);
            }
        } else if (needFrameACompute) {
            computeFrameALine(line, frameAIQ);
            if ((int)scratch_frameAAdaptiveIQComposite.size() < width)
                scratch_frameAAdaptiveIQComposite.resize(width);
            // Symmetric round-trip with Frame A's signed demod: remod back
            // through the signed phase so the composite scalar lands in
            // the physical frame produceY's `raw - clpLine` consumes.
            auto phaseCursor = carrierGrammarSignedSampleCursor(
                configuration.phaseCompensation ? carrierGrammarLine(line) : nullptr,
                left);

            // Reconstructed-luma feasibility, applied here because this is
            // where Frame A's IQ becomes the composite carrier that produceY
            // subtracts from raw.  The legs are the preclean carriers Frame A
            // actually combed, so raw-minus-preclean is each leg's luma.
            const bool frameAVertical = carrierFrameVerticalAllowed(line);
            const double *precleanC = precleanLinePtr(line, width);
            const double *precleanU = frameAVertical
                ? precleanLinePtr(line - 1, width) : nullptr;
            const double *precleanD = frameAVertical
                ? precleanLinePtr(line + 1, width) : nullptr;
            const quint16 *rawC = rawbuffer.constData() + line * videoParameters.fieldWidth;
            const quint16 *rawU = (precleanU && line - 1 >= 0)
                ? rawbuffer.constData() + (line - 1) * videoParameters.fieldWidth : nullptr;
            const quint16 *rawD = precleanD
                ? rawbuffer.constData() + (line + 1) * videoParameters.fieldWidth : nullptr;

            for (int rel = 0; rel < width; ++rel) {
                if (rel < (int)frameAIQ.size()) {
                    const auto &Z = frameAIQ[rel];
                    double carrier =
                        carrierGrammarRemodSigned4fscToComposite(phaseCursor, Z.real(), Z.imag());

                    if (precleanC) {
                        const int h = left + rel;
                        const double yC = (double)rawC[h] - precleanC[rel];
                        const double yU = (rawU && precleanU)
                            ? (double)rawU[h] - precleanU[rel]
                            : std::numeric_limits<double>::quiet_NaN();
                        const double yD = (rawD && precleanD)
                            ? (double)rawD[h] - precleanD[rel]
                            : std::numeric_limits<double>::quiet_NaN();
                        carrier = clampCarrierToInputLumaRangeShared(
                            carrier, (double)rawC[h], { yC, yU, yD }, precleanC[rel]);
                    }

                    scratch_frameAAdaptiveIQComposite[rel] = carrier;
                } else {
                    scratch_frameAAdaptiveIQComposite[rel] = 0.0;
                }
            }
        }

        if (needFrameBCompute) {
            computeFrameBLine(line, frameIQ, scratch_frameBDirectIQComposite);
        }

        const std::vector<double> &frameAttrScalar =
            needFrameBCompute ? scratch_frameBDirectIQComposite : scratch_frameAAdaptiveIQComposite;
        const std::vector<std::complex<double>> *frameAttrIQ =
            needFrameBCompute ? &frameIQ : (needFrameACompute ? &frameAIQ : nullptr);
        const double *candidateAForAttr =
            (wantFvf && fvfUseFrameModel && (int)scratch_frameAAdaptiveIQComposite.size() >= width)
                ? scratch_frameAAdaptiveIQComposite.data()
                : scratch_lineWorkA.data();
        collectCombAttributionEvidence(
            line,
            candidateAForAttr,
            scratch_lineWorkC.data(),
            needFrameIQCompute ? frameAttrScalar : scratch_frameBDirectIQComposite,
            frameAttrIQ);

        double *dst = clpbuffer[1].pixel[line];
        auto emitSelected = [&](int rel, double v) {
            dst[left + rel] = v;
        };

        {
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
                    // MEASURED, on the produceY-level oracle: a
                    // mean-of-contestants comparison can look close while
                    // real per-pixel headroom hides behind anti-correlated
                    // errors. So the quality of what this election collapses
                    // into one scalar cannot be read off the mean error of
                    // its ingredients -- it has to be asked per pixel.
                    scoreFieldVsFrame(
                        line,
                        tapLine,
                        (wantFvf && fvfUseFrameModel)
                            ? scratch_frameAAdaptiveIQComposite   // Frame A in the frame regime
                            : scratch_lineWorkA,                  // Field A in the interlace regime
                        scratch_lineWorkC.data(),                 // Field B / simple field
                        &scratch_frameBDirectIQComposite,         // Frame B / direct IQ composite
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

    }

    // Vertical companion to the per-line horizontal dilation
    // (combcandidate.cpp's markIntrafieldChromaBoundaryBand call): every
    // line's chromaBoundaryBand was decided independently above, so a
    // boundary near the seed threshold could flip line to line -- a jagged
    // edge on every consumer that reads the published flat plane and treats
    // it as one uniform region (produceY's band cede; Field B's own cede
    // above already committed per-line before this pass runs and is
    // unaffected). Run once per field, after every line in this field has
    // written its bit, before produceY reads it.
    if (!chromaBoundaryBand_flat.empty()) {
        CombContentReach::markVerticalChromaBoundaryBand(
            chromaBoundaryBand_flat, demodWidth, width, firstLine, lastLine, 1);
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

    static const bool blend3D = []{
        const char *e = std::getenv("LDCD_3D_TEMPORAL_GRAMMAR");
        return !(e && std::atoi(e) == 0);
    }();


    // ---- Acceptance uniformity (user-directed, 2026-07-28) ----
    //
    // The parked failure of the temporal grammar was line-to-line striping,
    // and its mechanism is now named: getCandidate biases a Same-relation
    // partner by +3.0 against an Opposite one, and WHICH relation a line
    // gets alternates with line parity. So the temporal SHARE of the blend
    // alternated by line -- adjacent lines received different AMOUNTS of
    // temporal correction. That is the uniform-render law again, in the
    // acceptance rather than the values.
    //
    // The cure follows this codebase's own precedent (the cross-colour
    // suppression verdict is vertically mixed and laterally boxcar'd into
    // an envelope before it scales any chroma, so suppression cannot
    // alias): decide per pixel, then make the DECISION STRENGTH spatially
    // coherent before it acts. Pass 1 computes each pixel's temporal mean
    // and its raw share; pass 2 smooths the SHARE FIELD -- a weight, never
    // the composite -- with a vertical [1,2,1] (which weights the two
    // parities equally, so a parity-alternating share cannot survive it)
    // and a lateral boxcar of about one carrier cycle; then it blends.
    // Members still face their binary vetoes in pass 1 (getCandidate
    // legality, the 2D-anchored hull), and a convex blend of hull-passing
    // members stays inside the hull.
    if (blend3D) {
        const int width = right - left;
        if (width <= 0 || firstLine >= lastLine) return;
        const size_t n = static_cast<size_t>(lastLine) * static_cast<size_t>(width);
        std::vector<float> shareRaw(n, 0.0f), tMean(n, 0.0f), ref(n, 0.0f);

        const double slack = std::max(
            0.0, configuration.tunables.TEMPORAL_HULL_SLACK_IRE) * irescale;
        const double memberBound = slack + 6.0 * irescale;
        constexpr double kTau = 1.5;

        for (int line = firstLine; line < lastLine; ++line) {
            // Certified cede (construction): no temporal members are even
            // evaluated on def lines; their share stays zero.
            if (certifiedOneDLevel() >= 3 && certifiedDefLine(line))
                continue;
            const double *lockedRow = configuration.phaseCompensation
                ? combSource1D_line(line) : nullptr;
            for (int h = left; h < right; ++h) {
                const size_t idx =
                    static_cast<size_t>(line) * width + (h - left);
                const double ref2d = clpbuffer[1].pixel[line][h];
                ref[idx]   = static_cast<float>(ref2d);
                tMean[idx] = static_cast<float>(ref2d);
                if (!std::isfinite(ref2d)) continue;

                qint32 bestIndex; double bestSample;
                TemporalCandidateSamples ts;
                getBestCandidate(line, h, previousFrame, nextFrame,
                                 bestIndex, bestSample, &ts);

                const int h0 = clampH(h);
                const int rel0 = h0 - left;
                const double base1d = (lockedRow && rel0 >= 0)
                    ? lockedRow[rel0]
                    : bucketScalar1D_line(line)[h0];

                double outs[4], pens[4];
                int nT = 0;
                auto admit = [&](const TemporalCandidateSamples::Sample &sm) {
                    if (nT >= 4 || !sm.valid) return;
                    const double out = (base1d - sm.value) * 0.5;
                    if (!std::isfinite(out)) return;
                    if (std::fabs(out - ref2d) > memberBound) return;
                    outs[nT] = out; pens[nT] = sm.penalty; ++nT;
                };
                admit(ts.previousField);
                admit(ts.nextField);
                admit(ts.previousFrame);
                admit(ts.nextFrame);
                if (nT == 0) continue;

                double pMin = ts.best2DPenalty;
                for (int k = 0; k < nT; ++k) pMin = std::min(pMin, pens[k]);

                const double w2d =
                    std::exp(-(ts.best2DPenalty - pMin) / kTau);
                double wT = 0.0, accT = 0.0;
                for (int k = 0; k < nT; ++k) {
                    const double w = std::exp(-(pens[k] - pMin) / kTau);
                    accT += w * outs[k];
                    wT   += w;
                }
                if (wT > 1e-12 && std::isfinite(w2d)) {
                    tMean[idx]    = static_cast<float>(accT / wT);
                    shareRaw[idx] = static_cast<float>(wT / (wT + w2d));
                }
            }
        }

        // Pass 2: vertical [1,2,1] then lateral boxcar on the SHARE.
        constexpr int kLatRadius = 4;
        std::vector<float> shareV(n, 0.0f);
        for (int line = firstLine; line < lastLine; ++line) {
            const int lu = std::max(firstLine, line - 1);
            const int ld = std::min(lastLine - 1, line + 1);
            const float *su = shareRaw.data() + (size_t)lu * width;
            const float *s0 = shareRaw.data() + (size_t)line * width;
            const float *sd = shareRaw.data() + (size_t)ld * width;
            float *out = shareV.data() + (size_t)line * width;
            for (int x = 0; x < width; ++x)
                out[x] = 0.25f * (su[x] + 2.0f * s0[x] + sd[x]);
        }
        for (int line = firstLine; line < lastLine; ++line) {
            if (certifiedOneDLevel() >= 3 && certifiedDefLine(line))
                continue;   // certified cede: the 2D seed (= center) stands
            const float *sv = shareV.data() + (size_t)line * width;
            const float *tm = tMean.data() + (size_t)line * width;
            const float *rf = ref.data() + (size_t)line * width;
            for (int x = 0; x < width; ++x) {
                double acc = 0.0; int cnt = 0;
                const int a = std::max(0, x - kLatRadius);
                const int b = std::min(width - 1, x + kLatRadius);
                for (int j = a; j <= b; ++j) { acc += sv[j]; ++cnt; }
                const double s = std::clamp(acc / std::max(1, cnt), 0.0, 1.0);
                const double v = (1.0 - s) * (double)rf[x] + s * (double)tm[x];
                if (std::isfinite(v))
                    clpbuffer[2].pixel[line][left + x] = v;
            }
        }
        return;
    }

    for (int line = firstLine; line < lastLine; ++line) {
        // Certified cede (construction): temporal machinery stands down on
        // def lines; the seeded 2D value (= center) stands.
        if (certifiedOneDLevel() >= 3 && certifiedDefLine(line))
            continue;

        for (int h = left; h < right; ++h) {
            const int rel = h - left;
        
        
            qint32 bestIndex;
            double bestSample;
            TemporalCandidateSamples temporalSamples;
            
            // Pass *this as well so getBestCandidate knows context
            getBestCandidate(line, h, previousFrame, nextFrame,
                             bestIndex, bestSample, &temporalSamples);
        
            const int h0 = clampH(h);
            const int rel0 = h0 - left;
            double base1d;
            const double *lockedRow = configuration.phaseCompensation
                ? combSource1D_line(line) : nullptr;
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
                // Temporal carrier estimate: classic (Y+C) - (Y-C) / 2.
                // This is a point operation on the full-band locked scalar;
                // there is no horizontal averaging or HF roll-off here.
                const double outBest = (base1d - bestSample) * 0.5;

                // Bound the winner with estimates independent of the winner:
                // the seeded 2D result and, when available, the opposite
                // temporal direction's output. Including outBest itself would
                // make this guard vacuous in a winner-take-all election.
                const TemporalCandidateSamples::Sample *partner = nullptr;
                switch (bestIndex) {
                case CAND_PREV_FIELD: partner = &temporalSamples.nextField; break;
                case CAND_NEXT_FIELD: partner = &temporalSamples.previousField; break;
                case CAND_PREV_FRAME: partner = &temporalSamples.nextFrame; break;
                case CAND_NEXT_FRAME: partner = &temporalSamples.previousFrame; break;
                default: break;
                }

                const double ref2d = clpbuffer[1].pixel[line][h];
                double hullMin = ref2d;
                double hullMax = ref2d;
                if (partner && partner->valid) {
                    const double outPartner = (base1d - partner->value) * 0.5;
                    if (std::isfinite(outPartner)) {
                        hullMin = std::min(hullMin, outPartner);
                        hullMax = std::max(hullMax, outPartner);
                    }
                }

                const double slack = std::max(
                    0.0, configuration.tunables.TEMPORAL_HULL_SLACK_IRE) * irescale;
                if (std::isfinite(outBest) && std::isfinite(ref2d) &&
                    outBest >= hullMin - slack && outBest <= hullMax + slack)
                {
                    clpbuffer[2].pixel[line][h] = outBest;
                }
                // Outside the hull, retain the full-band 2D value seeded in
                // clpbuffer[2]. The hull can refuse a temporal replacement,
                // but it never substitutes a filtered carrier.
            }
        }
    }

}
// True when this frame holds any dG twin-certified carrier. Sampled
// rather than exhaustive: coverage is a whole-field property, so a sparse
// probe over the active lines settles it. Cached per held frame.
bool Comb::FrameBuffer::frameHasExactCoverage() const
{
    if (exactCoverageCache >= 0)
        return exactCoverageCache != 0;
    exactCoverageCache = 0;
    if (!exactCarrier_flat.empty()) {
        const int firstLine = videoParameters.firstActiveFrameLine;
        const int lastLine  = videoParameters.lastActiveFrameLine;
        const int left      = videoParameters.activeVideoStart;
        const int right     = videoParameters.activeVideoEnd;
        for (int line = firstLine; line < lastLine && !exactCoverageCache;
             ++line) {
            const float *row = exactCarrierRow(line);
            if (!row) continue;
            for (int h = left; h < right; h += 32) {
                if (std::isfinite(row[h])) { exactCoverageCache = 1; break; }
            }
        }
    }
    return exactCoverageCache != 0;
}

// 3D Election
void Comb::FrameBuffer::getBestCandidate(qint32 lineNumber, qint32 h,
                                         const FrameBuffer &previousFrame,
                                         const FrameBuffer &nextFrame,
                                         qint32 &bestIndex, double &bestSample,
                                         TemporalCandidateSamples *temporalSamples) const
{
    Candidate c[NUM_CANDIDATES];
    const FrameBuffer* src[NUM_CANDIDATES] = { nullptr };

    static constexpr double LINE_BONUS  = -2.0;
    static constexpr double FIELD_BONUS = -4.0;
    static constexpr double FRAME_BONUS = -5.0;

    auto invalidateCandidate = [&](int idx) {
        c[idx].penalty = 1000.0;
        c[idx].sample  = 0.0;
        c[idx].yPen    = 1000.0;
        c[idx].iqPen   = 0.0;
        src[idx]       = nullptr;
    };

    for (int i = 0; i < NUM_CANDIDATES; ++i)
        invalidateCandidate(i);

    // LDCD_3D_TEMPORAL_GRAMMAR=1: temporal-grammar candidate rules (see
    // getCandidate). Default OFF preserves the historical gate behaviour.
    // Temporal grammar is LIVE (user approval 2026-07-28, render-judged:
    // "it looked good - and B was less of an outlier"). The historical
    // striping was the Same/Opposite operation-class mix, now refused in
    // getCandidate; what remains is the correct fix for the contradictory
    // gates that kept the frame axis dead. LDCD_3D_TEMPORAL_GRAMMAR=0
    // restores the old pre-gates for A/B.
    static const bool temporalGrammar3D = []{
        const char *e = std::getenv("LDCD_3D_TEMPORAL_GRAMMAR");
        return !(e && std::atoi(e) == 0);
    }();

    // 1D/2D Candidates (always available via this frame)
    c[CAND_LEFT]   = getCandidate(lineNumber, h, *this, lineNumber,     h - 2, 0.0);
    src[CAND_LEFT] = this;

    c[CAND_RIGHT]   = getCandidate(lineNumber, h, *this, lineNumber,     h + 2, 0.0);
    src[CAND_RIGHT] = this;

    c[CAND_UP]   = getCandidate(lineNumber, h, *this, lineNumber - 2, h, LINE_BONUS);
    src[CAND_UP] = this;

    c[CAND_DOWN]   = getCandidate(lineNumber, h, *this, lineNumber + 2, h, LINE_BONUS);
    src[CAND_DOWN] = this;

    const bool frameVerticalAllowed = carrierFrameVerticalAllowed(lineNumber);

    // --- Previous Field ---
    //
    // Previous and next field candidates are evaluated independently. A valid
    // previous-field candidate does not require a symmetric next-field candidate.
    if (frameVerticalAllowed && lineNumber - 1 >= videoParameters.firstActiveFrameLine) {
        if (temporalGrammar3D) {
            // Cross-frame first: the temporal grammar legalizes it per line
            // with the correct sign; self-frame is the fallback.
            c[CAND_PREV_FIELD] = getCandidate(lineNumber, h,
                                              previousFrame, lineNumber - 1, h,
                                              FIELD_BONUS);
            src[CAND_PREV_FIELD] = &previousFrame;
            if (c[CAND_PREV_FIELD].penalty >= 1000.0) {
                c[CAND_PREV_FIELD] = getCandidate(lineNumber, h,
                                                  *this, lineNumber - 1, h,
                                                  FIELD_BONUS);
                src[CAND_PREV_FIELD] = this;
            }
        } else if (carrierLineFlip(lineNumber) == carrierLineFlip(lineNumber - 1)) {
            c[CAND_PREV_FIELD] = getCandidate(lineNumber, h,
                                              previousFrame, lineNumber - 1, h,
                                              FIELD_BONUS);
            src[CAND_PREV_FIELD] = &previousFrame;
        } else {
            c[CAND_PREV_FIELD] = getCandidate(lineNumber, h,
                                              *this, lineNumber - 1, h,
                                              FIELD_BONUS);
            src[CAND_PREV_FIELD] = this;
        }
    }

    // --- Next Field ---
    if (frameVerticalAllowed && lineNumber + 1 < videoParameters.lastActiveFrameLine) {
        if (temporalGrammar3D) {
            c[CAND_NEXT_FIELD] = getCandidate(lineNumber, h,
                                              nextFrame, lineNumber + 1, h,
                                              FIELD_BONUS);
            src[CAND_NEXT_FIELD] = &nextFrame;
            if (c[CAND_NEXT_FIELD].penalty >= 1000.0) {
                c[CAND_NEXT_FIELD] = getCandidate(lineNumber, h,
                                                  *this, lineNumber + 1, h,
                                                  FIELD_BONUS);
                src[CAND_NEXT_FIELD] = this;
            }
        } else if (carrierLineFlip(lineNumber) == carrierLineFlip(lineNumber + 1)) {
            c[CAND_NEXT_FIELD] = getCandidate(lineNumber, h,
                                              nextFrame, lineNumber + 1, h,
                                              FIELD_BONUS);
            src[CAND_NEXT_FIELD] = &nextFrame;
        } else {
            c[CAND_NEXT_FIELD] = getCandidate(lineNumber, h,
                                              *this, lineNumber + 1, h,
                                              FIELD_BONUS);
            src[CAND_NEXT_FIELD] = this;
        }
    }

    // --- Temporal Frame Center: Previous Frame ---
    //
    // Same-line previous/next-frame candidates are only legal when the carrier
    // line relation matches. If the phase relation differs, the temporal center
    // candidate is invalid rather than merely expensive.
    if (temporalGrammar3D ||
        carrierLineFlip(lineNumber) == previousFrame.carrierLineFlip(lineNumber)) {
        // Temporal grammar decides legality and sign inside getCandidate.
        // Historical note: the equal-lineFlip pre-gate combined with the
        // reach check's Opposite-only requirement kept this candidate
        // permanently dead; with the env unset that dead state is preserved
        // as the baseline.
        c[CAND_PREV_FRAME] = getCandidate(lineNumber, h,
                                          previousFrame, lineNumber, h,
                                          FRAME_BONUS);
        src[CAND_PREV_FRAME] = &previousFrame;
    } else {
        invalidateCandidate(CAND_PREV_FRAME);
    }

    // --- Temporal Frame Center: Next Frame ---
    if (temporalGrammar3D ||
        carrierLineFlip(lineNumber) == nextFrame.carrierLineFlip(lineNumber)) {
        c[CAND_NEXT_FRAME] = getCandidate(lineNumber, h,
                                          nextFrame, lineNumber, h,
                                          FRAME_BONUS);
        src[CAND_NEXT_FRAME] = &nextFrame;
    } else {
        invalidateCandidate(CAND_NEXT_FRAME);
    }

    // --- Uncovered-frame deference (user, 2026-07-28: "perhaps B could
    // have a small penalty in getBestCandidate for current candidates") ---
    //
    // A frame with no dG twin coverage (B and D letters, and all
    // non-cadence material) carries an estimated carrier everywhere, while
    // its A/C neighbours carry the conservation-exact one. Where such a
    // frame sits beside a covered neighbour, its own spatial candidates
    // are the weaker testimony, so they pay a small penalty and the
    // temporal candidates drawn from certified neighbours become cheaper.
    //
    // This is DIFFERENTIAL, which is why it can act where the earlier
    // per-sample "certified source bonus" could not: that bonus applied
    // uniformly to every same-frame candidate and cancelled in the argmin.
    // Inert by construction when neither neighbour is covered, when this
    // frame is itself covered, and on material with no coverage at all.
    if (!frameHasExactCoverage() &&
        (previousFrame.frameHasExactCoverage() ||
         nextFrame.frameHasExactCoverage())) {
        static constexpr double UNCOVERED_SELF_PENALTY = 1.0;
        for (int i = 0; i < NUM_CANDIDATES; ++i) {
            if (src[i] != this || c[i].penalty >= 1000.0) continue;
            c[i].penalty += UNCOVERED_SELF_PENALTY;
        }
    }

    // --- Agreement Reward Shaping ---
    //
    // Old behavior:
    //
    //     d = abs(candidate.sample - current clpbuffer[1]) / irescale
    //
    // That was a same-pixel scalar chroma/bandpass comparison. It rewarded
    // temporal candidates for matching the current 2D chroma grid and punished
    // candidates that diverged from that grid.
    //
    // New behavior:
    //
    //     d = candidate.yPen
    //
    // getCandidate() has already computed yPen from reconstructed luma:
    //
    //     Y = raw - clpbuffer[1]
    //
    // over a small cross neighborhood. Reusing yPen avoids another comparison
    // and prevents compact-color checkerboard disagreement in chroma space from
    // automatically vetoing a picture-compatible temporal candidate. Beyond
    // the reward lobe, disagreement is deliberately neutral; split3D applies
    // the independent-estimate output hull after the election.
    if (configuration.dimensions == 3 && configuration.adaptive) {
        const auto &T = configuration.tunables;

        for (int i = 0; i < NUM_CANDIDATES; ++i) {
            const FrameBuffer* s = src[i];

            // Only shape temporal candidates here. Spatial candidates already
            // carry their getCandidate() penalties and bonuses.
            if (!s || s == this || c[i].penalty >= 1000.0)
                continue;

            const double dIRE = c[i].yPen;
            if (g_distCensus.on) g_distCensus.add(dIRE);
            double delta = 0.0;

            if (dIRE <= T.AGREEMENT_REWARD_RADIUS_IRE) {
                const double x = dIRE / T.AGREEMENT_REWARD_RADIUS_IRE;
                delta = -(T.AGREEMENT_REWARD_MAX * configuration.adaptThreshold)
                        * (1.0 - x * x);
            }

            c[i].penalty += delta;

            if (c[i].penalty > configuration.candidatePenaltyHardMax)
                c[i].penalty = configuration.candidatePenaltyHardMax;
        }
    }

    // Select best candidate.
    if (configuration.adaptive) {
        int best = 0;
        for (int i = 1; i < NUM_CANDIDATES; ++i) {
            if (c[i].penalty < c[best].penalty)
                best = i;
        }
        bestIndex = best;
    } else {
        // Non-adaptive fallback: prefer Previous Frame if valid, else Next, else 2D.
        if (src[CAND_PREV_FRAME])
            bestIndex = CAND_PREV_FRAME;
        else if (src[CAND_NEXT_FRAME])
            bestIndex = CAND_NEXT_FRAME;
        else
            bestIndex = CAND_UP;
    }

    if (temporalSamples) {
        auto exportSample = [&](int idx,
                                TemporalCandidateSamples::Sample &out) {
            out.value = c[idx].sample;
            out.penalty = c[idx].penalty;
            out.valid = src[idx] != nullptr && c[idx].penalty < 1000.0 &&
                        std::isfinite(c[idx].sample);
        };
        exportSample(CAND_PREV_FIELD, temporalSamples->previousField);
        exportSample(CAND_NEXT_FIELD, temporalSamples->nextField);
        exportSample(CAND_PREV_FRAME, temporalSamples->previousFrame);
        exportSample(CAND_NEXT_FRAME, temporalSamples->nextFrame);
        double b2 = 1000.0;
        for (int i = 0; i < CAND_PREV_FIELD; ++i)
            b2 = std::min(b2, c[i].penalty);
        temporalSamples->best2DPenalty = b2;
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
// standard Y'UV colour axes. The locked path now applies the front half of its
// base hue rotation before filterIQLocked() so the axis-specific FIRs see the
// expected orientation; the output half here preserves the same total hue.
void Comb::FrameBuffer::transformIQ(double chromaGain, double chromaPhase)
{
    if (configuration.phaseCompensation) {
        const double theta =
            (LOCKED_CHROMA_OUTPUT_ROT_DEG + chromaPhase) * M_PI / 180.0;
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

    if (configuration.twoDVariant == Comb::Configuration::TwoDVariant::FieldBSimple &&
        !fieldBDecisionReason_flat.empty())
    {
        // Build the diagnostic palette through FrameCanvas.  ComponentFrame
        // stores zero-centred U/V in signal units, not unsigned 16-bit YUV;
        // writing literal 16-bit values here drove both chroma axes strongly
        // positive and collapsed every reason into the same magenta image.
        const std::array<FrameCanvas::Colour, FieldBReasonCount> reasonShades = {{
            canvas.rgb(0x8080, 0x8080, 0x8080), // none: gray
            canvas.rgb(0x0000, 0xFFFF, 0xFFFF), // blend: cyan
            canvas.rgb(0x3030, 0x3030, 0x3030), // center / no answer: dark
            canvas.rgb(0xFFFF, 0x8000, 0x0000), // explicit cede: orange
            canvas.rgb(0xFFFF, 0x0000, 0xFFFF), // one-legged comb: magenta
            canvas.rgb(0x0000, 0xFFFF, 0x0000), // physical recovery: green
            canvas.rgb(0xFFFF, 0xFFFF, 0x0000)  // repaired-center hold: yellow
        }};

        const int firstLine = videoParameters.firstActiveFrameLine;
        const int lastLine  = videoParameters.lastActiveFrameLine;
        const int left      = videoParameters.activeVideoStart;
        const int right     = videoParameters.activeVideoEnd;

        for (int line = firstLine; line < lastLine; ++line) {
            double *Y = componentFrame->y(line);
            double *U = componentFrame->u(line);
            double *V = componentFrame->v(line);
            const std::uint8_t *reasonRow = fieldBDecisionReason_line(line);
            if (!reasonRow) continue;

            for (int h = left; h < right; ++h) {
                const std::uint8_t reason = std::min<std::uint8_t>(
                    reasonRow[h - left],
                    static_cast<std::uint8_t>(reasonShades.size() - 1));
                const FrameCanvas::Colour &shade = reasonShades[reason];
                Y[h] = shade.y;
                U[h] = shade.u;
                V[h] = shade.v;
            }
        }
        return;
    }

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
