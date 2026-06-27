/******************************************************************************
 * combwitness.cpp
 * ld-chroma-decoder — Constrained multi-witness reconstruction (locked path)
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * SPDX-FileCopyrightText: 2025-2026 Joseph Burns
 *
 * Implements buildConstrainedYWitness(), which assembles the initial witness
 * outputs from the evidence already collected by buildCarrierRetracted().
 *
 * Current state: constrained witness. The stage removes impossible luma
 * candidates, sharpens the broad prior from aperture-membership evidence, and
 * permits only a bounded compact-patch correction from the 1D witness.
 *
 * Conceptual mapping (see witness-migration.md):
 *   yWitness            ← constrained Y after survivor reconciliation
 *   carrierWitness      ← raw - yWitness
 *   obstructionRisk     ← carrierImpurity plus ambiguity/conflict/exclusion risk
 *   ambiguityWidth      ← derived carrier spread across delivered four-view evidence
 *   yWitnessConfidence  ← candidate agreement/spread after impossibles
 *   carrierWitnessConfidence ← carrier band confidence after exclusions
 ******************************************************************************/

#include "comb.h"
#include "feasibleband.h"
#include <algorithm>
#include <cmath>

namespace {

bool witnessWindowDumped = false;

} // namespace


void Comb::FrameBuffer::buildConstrainedYWitness()
{
    witnessValid = false;

    if (!carrierRetractedValid)
        return;

    const int first     = videoParameters.firstActiveFrameLine;
    const int last      = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int fullWidth = videoParameters.fieldWidth;
    const int width     = right - left;

    if (width <= 0 || first >= last || demodWidth != width || demodLines < last)
        return;

    const size_t need =
        static_cast<size_t>(demodLines) * static_cast<size_t>(demodWidth);

    auto ensure = [need](std::vector<float> &v) {
        if (v.size() < need)
            v.assign(need, 0.0f);
    };

    ensure(yWitness_flat);
    ensure(yWitnessConfidence_flat);
    ensure(carrierWitness_flat);
    ensure(carrierWitnessConfidence_flat);
    ensure(carrierCorrectionMask_flat);
    ensure(carrierWitnessCombed_flat);
    ensure(obstructionRisk_flat);
    ensure(ambiguityWidth_flat);

    if ((int)scratch_lumaBaseY4.size() < width) scratch_lumaBaseY4.resize(width, 0.0);
    if ((int)scratch_lumaSmooth.size()  < width) scratch_lumaSmooth.resize(width, 0.0);
    if ((int)scratch_lineWorkA.size()   < width) scratch_lineWorkA.resize(width, 0.0);
    if ((int)scratch_lineWorkB.size()   < width) scratch_lineWorkB.resize(width, 0.0);
    if ((int)scratch_lineWorkC.size()   < width) scratch_lineWorkC.resize(width, 0.0);
    if ((int)scratch_envLine.size()     < width) scratch_envLine.resize(width, 0.0);
    if ((int)scratch_spanLine.size()    < width) scratch_spanLine.resize(width, 0.0);

    struct LocalBand {
        double lo =  1e300;
        double hi = -1e300;

        bool valid() const { return lo <= hi; }
        double width() const { return valid() ? (hi - lo) : 0.0; }
        double center() const { return 0.5 * (lo + hi); }

        void include(double v) {
            if (!std::isfinite(v))
                return;
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }

        void expand(double pad) {
            if (!valid())
                return;
            lo -= pad;
            hi += pad;
        }

        void intersect(double a, double b) {
            if (!valid()) {
                lo = a;
                hi = b;
                return;
            }
            lo = std::max(lo, a);
            hi = std::min(hi, b);
        }

        double clamp(double v) const {
            return valid() ? std::clamp(v, lo, hi) : v;
        }
    };

    const auto smoothStep01 = [](double t) -> double {
        t = std::clamp(t, 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    };

    const double ireToSamples = irescale;

    const bool haveLumaValidityRange =
        videoParameters.white16bIre > videoParameters.black16bIre;

    const double lumaValidityPad =
        haveLumaValidityRange ? (16.0 * ireToSamples) : 0.0;

    const double lumaValidityLo =
        haveLumaValidityRange
            ? ((double)videoParameters.black16bIre - lumaValidityPad)
            : 0.0;

    const double lumaValidityHi =
        haveLumaValidityRange
            ? ((double)videoParameters.white16bIre + lumaValidityPad)
            : 0.0;

    /*
     * Witness policy:
     *
     *   moving coarse    = broad luma prior
     *   four floor views = possible-value clamp
     *   lurch           = boundary sharpener inside the clamp
     *   1D              = bounded repair inside strongly certified compact patches
     *
     * The old carrier-side candidate election is intentionally absent.
     */
    const double LURCH_SUPPORT_START = 0.12;
    const double LURCH_SUPPORT_FULL  = 0.55;
    const double LURCH_DELTA_START_IRE = 0.20;
    const double LURCH_DELTA_FULL_IRE  = 2.75;

    const double FLOOR_BASE_PAD_IRE  = 0.85;
    const double FLOOR_LURCH_PAD_IRE = 2.75;

    const double PATCH_CHROMA_START_IRE = 1.25;
    const double PATCH_CHROMA_FULL_IRE  = 5.00;
    const double PATCH_FLOOR_TIGHT_IRE  = 5.50;
    const double PATCH_FLOOR_LOOSE_IRE  = 11.00;
    const double PATCH_SPAN_START_PX    = 5.0;
    const double PATCH_SPAN_FULL_PX     = 9.0;
    const double PATCH_SELECT_GATE      = 0.72;
    const double PATCH_MIN_DELTA_IRE    = 0.75;
    const double PATCH_MAX_REPAIR_IRE   = 2.0;

    // Fast-contour 1D substitution: where the comb carrier confirms low real
    // chroma (a luma edge, which does not survive interline cancellation), 1D
    // is the phase-reliable HF-luma source lurch cannot adjudicate. The
    // chroma-trust ramp aligns with the patch chroma ramp above.
    const double ONE_D_FAST_CONTOUR_START_IRE = 1.25;
    const double ONE_D_FAST_CONTOUR_FULL_IRE  = 3.50;

    double *envScratch = scratch_envLine.data();
    double *spanScratch = scratch_spanLine.data();

    // --debug-phase-legs: which gate binds the compact-patch detector?
    // Candidates are compact-chroma pixels (envelope above the start
    // threshold, span within the patch range).  "binding" counts which gate
    // was the minimum term at each candidate; "best" dumps the full gate
    // breakdown at the strongest candidate in the frame.
    struct PatchGateDiag {
        long long candidates = 0;
        long long selected = 0;
        long long bindEnv = 0, bindLegal = 0, bindSpan = 0;
        long long bindFloor = 0, bindLattice = 0, bindVeto = 0;
        double bestGate = -1.0;
        int bestLine = -1, bestX = -1;
        double bEnvIRE = 0, bEnvGate = 0, bCost = 0, bLegal = 0;
        double bSpanLen = 0, bSpanGate = 0, bFloorW = 0, bFloorGate = 0;
        double bLatticeGate = 0, bVetoGate = 0, bSharp = 0;
        double bCompact = 0, bDeltaIRE = 0;
        bool bSelected = false;

        // Strong-span coverage: contiguous env runs (>2.5 IRE, 4..24 px) are
        // patch-shaped clusters; coverage = fraction of the span's pixels
        // that received bounded repair. Solid patches should be >75%; coverage
        // in the middle bins means selection flickers spatially across the
        // patch — itself a checkerboard generator.
        long long spans = 0;
        long long spansLowCov = 0, spansMidCov = 0, spansHighCov = 0;

        // Witness comb retention at selected pixels: |combed| / |witness|.
        // ~1 means patch chroma survives cancellation; ~0.5 half-killed
        // (one-line-tall color or neighbor mode mismatch); ~0 cancelled.
        double retainCwSum = 0.0;
        double retainCombedSum = 0.0;
        long long retainN = 0;
    } patchDiag;
    const bool patchDiagEnabled = false;

    for (int line = first; line < last; ++line) {
        const quint16 *rawLine = rawbuffer.data()
                               + static_cast<size_t>(line) * fullWidth;

        const float *impurityRow = carrierImpurity_line(line);
        const auto  *evidence = coarseYEvidence_line(line);

        const double *baseY4 = nullptr;
        if (lockedLumaCacheValid &&
            !lockedLumaBaseY4_flat.empty() &&
            demodWidth == width)
        {
            baseY4 = lockedLumaBaseY4_line(line);
        } else {
            buildCompositeLumaDecompositionLine(rawLine, left, width,
                                                scratch_lumaBaseY4.data(),
                                                nullptr,
                                                nullptr);
            baseY4 = scratch_lumaBaseY4.data();
        }

        const double *smoothY = nullptr;
        if (lockedLumaCacheValid &&
            !lockedLumaSmooth_flat.empty() &&
            demodWidth == width)
        {
            smoothY = lockedLumaSmooth_line(line);
        } else {
            buildCompositeLumaDecompositionLine(rawLine, left, width,
                                                nullptr,
                                                nullptr,
                                                scratch_lumaSmooth.data());
            smoothY = scratch_lumaSmooth.data();
        }

        double *movingCoarse = scratch_lineWorkA.data();

        /*
         * Moving 4-sample coarse:
         *
         * Carrier-cancelled over one fsc aperture, but evaluated every sample.
         * This is the broad Y prior, not the final answer in narrow patches.
         */
        for (int xi = 0; xi < width; ++xi) {
            if (width >= 4) {
                int s = xi;
                if (s + 3 >= width)
                    s = width - 4;

                movingCoarse[xi] =
                    0.25 * ((double)rawLine[left + s + 0]
                           + (double)rawLine[left + s + 1]
                           + (double)rawLine[left + s + 2]
                           + (double)rawLine[left + s + 3]);
            } else {
                movingCoarse[xi] = baseY4
                    ? baseY4[xi]
                    : (double)rawLine[left + xi];
            }
        }

        /*
         * Lurch preconditioner.
         *
         * The boxcar prior smears a luma step across four columns.  Sharpen
         * it before anything downstream consumes it: confirmed same-sign
         * membership runs (phase-invariant luma evidence) snap the prior to
         * the same-side window mean.  sharpGate records where this fired so
         * the per-pixel lurch correction below stands down instead of
         * double-applying the same evidence.
         */
        double *sharpPrior = scratch_lineWorkB.data();
        double *sharpGate  = scratch_lineWorkC.data();
        for (int xi = 0; xi < width; ++xi)
            sharpPrior[xi] = movingCoarse[xi];

        if (width >= 4) {
            lurchSharpenCoarsePrior(movingCoarse, width - 3, width,
                                    sharpPrior, sharpGate);
        } else {
            std::fill(sharpGate, sharpGate + width, 0.0);
        }

        /*
         * Chroma envelope and its contiguous span.
         *
         * The patch detector's compactness question is answered directly:
         * how long is the contiguous run of envelope energy containing this
         * pixel?  A compact patch is a short run; a wide chroma region (even
         * a low-saturation one) is a long run and is the coarse/comb paths'
         * territory — compact 1D repair must not fire there, including
         * in the boundary bands where the wide-vs-narrow risk proxy lights
         * up.  Risk no longer participates in patch detection: it is a
         * compactness proxy that is wrong at the boundaries of real chroma,
         * and it has a history of relocating errors into detail.
         */
        for (int xi = 0; xi < width; ++xi) {
            const int klo = std::max(0, xi - 1);
            const int khi = std::min(width - 1, xi + 1);
            double e = 0.0;
            for (int k = klo; k <= khi; ++k) {
                e = std::max(e, std::fabs(
                    (double)rawLine[left + k] - sharpPrior[k]));
            }
            envScratch[xi] = e * invIreScale;
        }

        {
            int runStart = -1;
            for (int xi = 0; xi <= width; ++xi) {
                const bool on =
                    (xi < width) && (envScratch[xi] > PATCH_CHROMA_START_IRE);
                if (on && runStart < 0) {
                    runStart = xi;
                } else if (!on && runStart >= 0) {
                    const double runLen = (double)(xi - runStart);
                    for (int k = runStart; k < xi; ++k)
                        spanScratch[k] = runLen;
                    runStart = -1;
                }
                if (xi < width && !on)
                    spanScratch[xi] = 0.0;
            }
        }

        const float linePlausibility =
            static_cast<float>(carrierPlausibility(carrierGrammarLine(line)));

        float *yWit    = yWitness_flat.data()
                       + static_cast<size_t>(line) * width;
        float *yConf   = yWitnessConfidence_flat.data()
                       + static_cast<size_t>(line) * width;
        float *cWit    = carrierWitness_flat.data()
                       + static_cast<size_t>(line) * width;
        float *cConf   = carrierWitnessConfidence_flat.data()
                       + static_cast<size_t>(line) * width;
        float *obsRisk = obstructionRisk_flat.data()
                       + static_cast<size_t>(line) * width;
        float *ambig   = ambiguityWidth_flat.data()
                       + static_cast<size_t>(line) * width;
        float *corrMask = carrierCorrectionMask_flat.data()
                        + static_cast<size_t>(line) * width;

        // The interline-combed carrier (from buildCarrierRetracted) is the
        // honest chroma test. raw - coarse re-admits luma-step energy as false
        // carrier at sharp transitions, so 1D "sees chroma" at luma edges; the
        // comb carrier cancels that step (it does not invert line to line) and
        // keeps only real chroma. Used below to gate 1D trust.
        const float *combCarrierRow = combedCarrier_line(line);

        /*
         * Same-phase 1D luma estimate.
         *
         * This is compact-patch repair evidence. It is never a general luma
         * source and is capped after the ordinary witness path completes.
         */
        auto split1DCandidateAt = [&](int x) -> double {
            int hm2 = left + x - 2;
            if (hm2 < left)
                hm2 = left + (left - hm2 - 1);

            int hp2 = left + x + 2;
            if (hp2 >= right)
                hp2 = right - 1 - (hp2 - right);

            return 0.25 * ((double)rawLine[hm2]
                         + 2.0 * (double)rawLine[left + x]
                         + (double)rawLine[hp2]);
        };

        for (int xi = 0; xi < width; ++xi) {
            const int h = left + xi;
            const double raw = (double)rawLine[h];

            const double risk = std::clamp(
                impurityRow ? (double)impurityRow[xi] : 0.0,
                0.0,
                1.0);

            const int xm1 = std::max(0, xi - 1);
            const int xp1 = std::min(width - 1, xi + 1);
            const int xm2 = std::max(0, xi - 2);
            const int xp2 = std::min(width - 1, xi + 2);

            const double smoothPrior = smoothY
                ? smoothY[xi]
                : (baseY4 ? baseY4[xi] : movingCoarse[xi]);

            /*
             * Slope-correct aperture floors to current sample position.
             */
            const double coarseGrad =
                baseY4
                    ? 0.5 * (baseY4[xp1] - baseY4[xm1])
                    : 0.0;

            auto slopeAdj = [&](const lddecode::FourViewEvidenceView &view) {
                return coarseGrad * ((double)xi - (double)view.apertureCenter);
            };

            LocalBand floorBand;
            double bestViewCost = 1e300;
            double maxViewCost = 0.0;
            double maxYSpanIRE = 0.0;
            double maxLatticeRiskIRE = 0.0;
            double maxFitErrorIRE = 0.0;

            int viewCount = 0;
            double viewCostCache[4] = {0.0, 0.0, 0.0, 0.0};
            if (evidence && evidence[xi].viewCount > 0) {
                // views[] is fixed-size 4 in attributiondefs.h; cap defensively
                // so the stack cache below stays in bounds.
                viewCount = std::min(evidence[xi].viewCount, 4);

                for (int v = 0; v < viewCount; ++v) {
                    const auto &view = evidence[xi].views[v];

                    const double viewCost =
                        (double)view.sampleFitErrorIRE
                      + 0.50 * (double)view.remodErrorIRE
                      + 0.35 * (double)view.latticeRiskIRE
                      + 0.15 * (double)view.ySpanIRE;

                    viewCostCache[v] = viewCost;

                    bestViewCost = std::min(bestViewCost, viewCost);
                    maxViewCost = std::max(maxViewCost, viewCost);
                    maxYSpanIRE = std::max(maxYSpanIRE, (double)view.ySpanIRE);
                    maxLatticeRiskIRE = std::max(maxLatticeRiskIRE,
                                                 (double)view.latticeRiskIRE);
                    maxFitErrorIRE = std::max(maxFitErrorIRE,
                                              (double)view.sampleFitErrorIRE);
                }

                const double keepThreshold = bestViewCost + 3.0; // keepSlackIRE

                for (int v = 0; v < viewCount; ++v) {
                    if (viewCostCache[v] > keepThreshold)
                        continue;

                    const auto &view = evidence[xi].views[v];
                    floorBand.include((double)view.yFloor + slopeAdj(view));
                }
            }

            if (!floorBand.valid())
                floorBand.include(smoothPrior);

            /*
             * Lurch observations:
             *
             * These are aperture-membership movements.  They sharpen luma
             * boundaries, but they are not allowed to decide the interior of
             * compact chroma patches.
             */
            double maxMembershipSupport = 0.0;
            double maxMembershipDeltaIRE = 0.0;
            double lurchNumerator = 0.0;
            double lurchDenominator = 0.0;

            if (evidence && viewCount > 0) {
                for (int v = 0; v < viewCount; ++v) {
                    const auto &view = evidence[xi].views[v];

                    const double support =
                        std::clamp((double)view.membershipSupport, 0.0, 1.0);
                    const double deltaSample =
                        (double)view.membershipDeltaSample;
                    const double deltaIRE =
                        std::fabs((double)view.membershipDeltaIRE);
                    const double localX =
                        (double)view.membershipLocalX;

                    maxMembershipSupport =
                        std::max(maxMembershipSupport, support);
                    maxMembershipDeltaIRE =
                        std::max(maxMembershipDeltaIRE, deltaIRE);

                    if (support <= 1e-6 || deltaIRE <= 1e-6)
                        continue;

                    const double localizer =
                        std::exp(-0.5 * (localX * localX) / (1.35 * 1.35));

                    /*
                     * membershipLocalX is edge-center minus current x.
                     * For a positive step moving rightward:
                     *   left side  => localX > 0 => pull down
                     *   right side => localX < 0 => pull up
                     */
                    const double side = -std::tanh(localX / 1.25);
                    const double w = support * localizer;

                    lurchNumerator += w * side * deltaSample;
                    lurchDenominator += w;
                }
            }

            const double lurchSupportGate = smoothStep01(
                (maxMembershipSupport - LURCH_SUPPORT_START) /
                std::max(1e-9, LURCH_SUPPORT_FULL - LURCH_SUPPORT_START));

            const double lurchDeltaGate = smoothStep01(
                (maxMembershipDeltaIRE - LURCH_DELTA_START_IRE) /
                std::max(1e-9, LURCH_DELTA_FULL_IRE - LURCH_DELTA_START_IRE));

            const double lurchGate = lurchSupportGate * lurchDeltaGate;

            const double floorWidthIRE =
                floorBand.valid() ? floorBand.width() * invIreScale : 64.0;

            /*
             * Cycle-scale chroma envelope, not instantaneous residual.
             *
             * |raw - coarse| dips to zero at carrier zero-crossings every two
             * samples.  Gating the patch detector on the instantaneous value
             * made 1D repair flicker at carrier rate through a patch — the
             * authority switch itself became a checkerboard generator.  A
             * 3-sample max envelope is flat across the cycle (on the exact
             * 4fSC grid the magnitudes run A,0,A,0).
             */
            const double carrierResidIRE = envScratch[xi];

            // Real-chroma confirmation from the comb carrier, not raw - coarse.
            // 2-sample quadrature envelope is flat across the carrier cycle
            // (A,0,A,0 on the 4fSC grid). High where genuine chroma survives
            // interline cancellation; ~0 at luma edges. trust1D is its
            // complement on the patch chroma ramp: 1D is phase-reliable only
            // where the comb confirms little real chroma.
            double combChromaIRE = 0.0;
            if (combCarrierRow) {
                const int xj = std::min(xi + 1, width - 1);
                const double c0 = (double)combCarrierRow[xi];
                const double c1 = (double)combCarrierRow[xj];
                combChromaIRE = std::sqrt(c0 * c0 + c1 * c1) * invIreScale;
            }
            const double trust1D = 1.0 - smoothStep01(
                (combChromaIRE - PATCH_CHROMA_START_IRE) /
                std::max(1e-9, PATCH_CHROMA_FULL_IRE - PATCH_CHROMA_START_IRE));

            /*
             * Carrier legality from the four-view evidence.
             *
             * Each view fits its own floor plus carrier, so the floor absorbs
             * any DC offset between the witness prior and true luma — unlike
             * a residual-opposition test against sharpPrior, which the
             * blurred prior inside an unsnapped patch poisons with its own
             * lump.  A low best view cost means a legal one-cycle carrier
             * explanation of this pixel exists; barren luma-step residue has
             * no such explanation and every view prices it high.
             */
            const double viewLegality =
                (bestViewCost < 1e299)
                    ? (1.0 - smoothStep01((bestViewCost - 2.0) / 5.0))
                    : 0.0;

            const double oneDUnclamped = split1DCandidateAt(xi);

            /*
             * Compact chroma patch detector:
             *
             * A narrow saturated patch has high residual energy, while the
             * four coarse floors may still form a tight legal fence. That is
             * the only regime where the 1D estimate may offer bounded repair.
             */
            // Chroma evidence = envelope energy that also has a legal carrier
            // explanation.  Envelope alone re-admits the step residue at
            // sharp luma transitions.
            const double patchEnvGate = smoothStep01(
                (carrierResidIRE - PATCH_CHROMA_START_IRE) /
                std::max(1e-9, PATCH_CHROMA_FULL_IRE - PATCH_CHROMA_START_IRE));

            const double patchChromaGate = patchEnvGate * viewLegality;

            // Compactness, measured rather than proxied: the contiguous
            // envelope run containing this pixel must be short.  Wide chroma
            // (the gold oval) is excluded everywhere, including its boundary
            // bands, no matter what the impurity channel thinks of them.
            const double patchSpanGate = 1.0 - smoothStep01(
                (spanScratch[xi] - PATCH_SPAN_START_PX) /
                std::max(1e-9, PATCH_SPAN_FULL_PX - PATCH_SPAN_START_PX));

            const double patchFloorGate =
                1.0 - smoothStep01(
                    (floorWidthIRE - PATCH_FLOOR_TIGHT_IRE) /
                    std::max(1e-9, PATCH_FLOOR_LOOSE_IRE - PATCH_FLOOR_TIGHT_IRE));

            const double patchLatticeGate =
                1.0 - 0.35 * smoothStep01((maxLatticeRiskIRE - 2.0) / 8.0);

            /*
             * Patch authority requires chroma evidence.  Membership is luma
             * evidence — maximal at a vertical luma transition with no chroma
             * at all — and letting it substitute for the chroma gate made
             * patch mode flicker line-to-line at the threshold along such
             * edges.  The flicker is self-amplifying: adjacent lines' yWitness
             * disagree at the edge, which closes the witness comb's luma gate,
             * so the same-sign edge spikes the comb would cancel survive as
             * colored speckle.  The membership substitute existed to cover
             * carrier zero-crossing dips in the instantaneous residual; the
             * cycle-scale envelope removed those dips, so it is retired.
             *
             * A luma step confirmed by the preconditioner withdraws patch
             * authority only where no legal carrier explanation exists.  A
             * patch abutting black is both a luma step and a chroma onset at
             * the same columns: the step verdict must yield to carrier
             * legality, or the veto strips repair from exactly the
             * patch that needs it.
             */
            const double patchVetoGate =
                1.0 - 0.85 * sharpGate[xi] * (1.0 - viewLegality);

            const double compactPatchGate = std::clamp(
                patchChromaGate *
                patchSpanGate *
                patchFloorGate *
                patchLatticeGate *
                patchVetoGate,
                0.0,
                1.0);

            /*
             * The four coarse floors define possible Y. Compact detection does
             * not widen this ordinary support path; it may only apply the
             * bounded hard-feasible correction below.
             */
            const double floorPadIRE =
                FLOOR_BASE_PAD_IRE
              + FLOOR_LURCH_PAD_IRE * lurchGate
              + 0.25 * maxYSpanIRE;

            /*
             * The floor/support band is a soft support interval.  It is built
             * from aperture/coarse evidence and is useful for broad Y and
             * non-patch contour repair, but it is not allowed to become the
             * hard clamp for compact-patch 1D.
             */
            LocalBand supportBand = floorBand;
            supportBand.expand(floorPadIRE * ireToSamples);

            /*
             * Hard feasibility is accumulated only from true impossibles:
             * legal luma range plus same-phase pair-sum constraints.  Compact
             * patch 1D is clamped here, not to the coarse support band.
             */
            lddecode::FeasibleInterval hardBand;
            bool haveHardBand = false;

            if (haveLumaValidityRange) {
                hardBand.clampTo(lumaValidityLo, lumaValidityHi);
                haveHardBand = hardBand.valid();

                double pairNeighbors[2];
                int pairNeighborCount = 0;

                if (xi >= 2)
                    pairNeighbors[pairNeighborCount++] =
                        (double)rawLine[h - 2];

                if (xi + 2 < width)
                    pairNeighbors[pairNeighborCount++] =
                        (double)rawLine[h + 2];

                if (pairNeighborCount > 0) {
                    const lddecode::FeasibleInterval hardPairY =
                        lddecode::lumaFeasibleFromPairSums(
                            raw,
                            pairNeighbors,
                            pairNeighborCount,
                            lumaValidityLo,
                            lumaValidityHi);

                    if (hardPairY.valid()) {
                        hardBand.intersect(hardPairY);
                        haveHardBand = hardBand.valid();
                    }
                }
            }

            /*
             * possibleBand is the normal policy band used outside compact
             * patch 1D.  It combines soft support with hard feasibility where
             * they agree.  If they conflict, preserve the hard legality
             * interval rather than dropping the pixel.
             */
            LocalBand possibleBand = supportBand;

            if (haveHardBand)
                possibleBand.intersect(hardBand.lo, hardBand.hi);

            if (!possibleBand.valid()) {
                if (haveHardBand) {
                    possibleBand.lo = hardBand.lo;
                    possibleBand.hi = hardBand.hi;
                } else {
                    possibleBand = supportBand;
                    if (!possibleBand.valid()) {
                        possibleBand = floorBand;
                        possibleBand.expand((4.0 + 4.0 * risk) * ireToSamples);
                    }
                }
            }

            const auto hardClamp = [&](double y) -> double {
                return haveHardBand ? hardBand.clamp(y) : y;
            };

            const double y1DHard = hardClamp(oneDUnclamped);
            /*
             * Primary broad prior, lurch-preconditioned.
             */
            double yPrior = sharpPrior[xi];
            yPrior = possibleBand.clamp(yPrior);

            /*
             * Lurch sharpening.
             *
             * Compact detection does not alter the ordinary boundary model.
             * Any compact repair is applied after this path is complete.
             */
            double yLurch = yPrior;
            if (lurchDenominator > 1e-9 && lurchGate > 0.0) {
                const double rawCorrection =
                    lurchNumerator / lurchDenominator;

                // The preconditioner consumed the same membership evidence
                // upstream; where it fired, the prior is already sharp and a
                // second additive correction would double-apply the step.
                const double preconditionGate = 1.0 - sharpGate[xi];

                const double lurchGain =
                    (0.80 + 0.35 * lurchGate) * preconditionGate;

                const double maxCorrection =
                    (0.60 + 0.90 * lurchGate) *
                    std::max(0.75, maxMembershipDeltaIRE) *
                    ireToSamples;

                yLurch += std::clamp(rawCorrection * lurchGain,
                                     -maxCorrection,
                                      maxCorrection);

                if (possibleBand.valid() &&
                    lurchGate > 0.35)
                {
                    const double bandCenter = possibleBand.center();
                    const double fromCenter = yLurch - bandCenter;

                    const double signSeed =
                        (std::fabs(rawCorrection) > 1e-9)
                            ? rawCorrection
                            : fromCenter;

                    if (std::fabs(signSeed) > 1e-9) {
                        const double outward = std::copysign(1.0, signSeed);
                        const double halfBand = 0.5 * possibleBand.width();

                        const double edgePush =
                            outward *
                            halfBand *
                            0.18 *
                            smoothStep01((lurchGate - 0.35) / 0.65) *
                            preconditionGate;

                        if ((yLurch - bandCenter) * outward >= -1e-9)
                            yLurch += edgePush;
                    }
                }

                yLurch = possibleBand.clamp(yLurch);
            }

            /*
             * Patch-mode 1D candidate.
             *
             * The symmetric ±2 estimate cannot cancel carrier where the
             * chroma envelope changes inside its span: the two columns
             * flanking a patch edge retain half the carrier, leaving a
             * half-amplitude carrier deficit in the witness carrier and a
             * matching ripple in Y.  The one-sided same-phase pair sums
             * cancel carrier exactly when both samples share the patch, so
             * patch mode picks whichever candidate lands nearest the
             * carrier-free coarse estimate.  Fast-contour mode keeps the
             * symmetric form — deviating from coarse is the point there.
             */
            double y1DPatchCand = oneDUnclamped;
            {
                const double leftPair = 0.5 *
                    (raw + (double)rawLine[left + xm2]);
                const double rightPair = 0.5 *
                    (raw + (double)rawLine[left + xp2]);

                if (std::fabs(leftPair - yLurch) <
                    std::fabs(y1DPatchCand - yLurch))
                    y1DPatchCand = leftPair;
                if (std::fabs(rightPair - yLurch) <
                    std::fabs(y1DPatchCand - yLurch))
                    y1DPatchCand = rightPair;
            }
            const double y1DPatchHard = hardClamp(y1DPatchCand);

            /*
             * Compact repair.
             *
             * The 1D candidate is never a general contour source. A strong
             * compact verdict may move the completed witness toward the
             * hard-feasible 1D candidate by at most PATCH_MAX_REPAIR_IRE.
             */
            const double oneDDeltaIRE =
                std::fabs(y1DHard - yLurch) * invIreScale;
            const double patchDeltaIRE =
                std::fabs(y1DPatchHard - yLurch) * invIreScale;

            const bool compactRepairEligible =
                haveHardBand &&
                compactPatchGate >= PATCH_SELECT_GATE &&
                patchDeltaIRE >= PATCH_MIN_DELTA_IRE;

            if (patchDiagEnabled &&
                carrierResidIRE > PATCH_CHROMA_START_IRE &&
                spanScratch[xi] > 0.0 &&
                spanScratch[xi] <= PATCH_SPAN_FULL_PX)
            {
                ++patchDiag.candidates;
                if (compactRepairEligible)
                    ++patchDiag.selected;

                const double gateVals[6] = {
                    patchEnvGate, viewLegality, patchSpanGate,
                    patchFloorGate, patchLatticeGate, patchVetoGate
                };
                long long *gateBinds[6] = {
                    &patchDiag.bindEnv, &patchDiag.bindLegal,
                    &patchDiag.bindSpan, &patchDiag.bindFloor,
                    &patchDiag.bindLattice, &patchDiag.bindVeto
                };
                int minIdx = 0;
                for (int gi = 1; gi < 6; ++gi) {
                    if (gateVals[gi] < gateVals[minIdx])
                        minIdx = gi;
                }
                ++(*gateBinds[minIdx]);

                if (compactPatchGate > patchDiag.bestGate) {
                    patchDiag.bestGate = compactPatchGate;
                    patchDiag.bestLine = line;
                    patchDiag.bestX = xi;
                    patchDiag.bEnvIRE = carrierResidIRE;
                    patchDiag.bEnvGate = patchEnvGate;
                    patchDiag.bCost = (bestViewCost < 1e299) ? bestViewCost : -1.0;
                    patchDiag.bLegal = viewLegality;
                    patchDiag.bSpanLen = spanScratch[xi];
                    patchDiag.bSpanGate = patchSpanGate;
                    patchDiag.bFloorW = floorWidthIRE;
                    patchDiag.bFloorGate = patchFloorGate;
                    patchDiag.bLatticeGate = patchLatticeGate;
                    patchDiag.bVetoGate = patchVetoGate;
                    patchDiag.bSharp = sharpGate[xi];
                    patchDiag.bCompact = compactPatchGate;
                    patchDiag.bDeltaIRE = patchDeltaIRE;
                    patchDiag.bSelected = compactRepairEligible;
                }
            }

            double yOut = yLurch;
            bool patchSelected = false;
            bool contourBlended = false;

            if (compactRepairEligible) {
                const double maxRepair =
                    PATCH_MAX_REPAIR_IRE * ireToSamples * compactPatchGate;
                yOut += std::clamp(
                    y1DPatchHard - yLurch,
                    -maxRepair,
                    maxRepair);
                yOut = hardClamp(yOut);
                patchSelected = true;
            } else {
                // Fast-contour 1D: in comb-confirmed low chroma, a large gap
                // between 1D and the lurch-shaped prior means lurch is failing
                // on HF luma it cannot adjudicate (direction reverses inside the
                // span). Lean to the phase-reliable 1D, weighted by trust1D so
                // the substitution collapses as real chroma appears -- there the
                // compact patch above is the only 1D carve-out.
                const double contourGate = smoothStep01(
                    (oneDDeltaIRE - ONE_D_FAST_CONTOUR_START_IRE) /
                    std::max(1e-9,
                        ONE_D_FAST_CONTOUR_FULL_IRE - ONE_D_FAST_CONTOUR_START_IRE));
                const double fastContourBlend = trust1D * contourGate;
                if (fastContourBlend > 0.0) {
                    const double y1DSoft = possibleBand.clamp(oneDUnclamped);
                    yOut = yLurch * (1.0 - fastContourBlend)
                         + y1DSoft * fastContourBlend;
                    contourBlended = (fastContourBlend > 0.2);
                }
            }

            yOut = patchSelected ? hardClamp(yOut) : possibleBand.clamp(yOut);

            const double carrierOut = raw - yOut;

            const double possibleWidthIRE =
                possibleBand.valid() ? possibleBand.width() * invIreScale : 64.0;

            const double viewConflictIRE =
                (bestViewCost < 1e299)
                    ? std::max(0.0, maxViewCost - bestViewCost)
                    : 0.0;

            const double ambiguityIRE =
                std::max(possibleWidthIRE,
                         std::max(floorWidthIRE, oneDDeltaIRE));

            const double carrierIRE =
                std::fabs(carrierOut) * invIreScale;

            const double ambiguityNorm = std::clamp(
                ambiguityIRE /
                    std::max(4.0, 0.35 * carrierIRE + 2.0),
                0.0,
                1.0);

            const double conflictNorm =
                std::clamp(viewConflictIRE / 12.0, 0.0, 1.0);

            /*
             * Obstruction reports risk, not veto.  A compact patch selected
             * by 1D is not a failure; it is the intended authority switch.
             */
            const double obstruction = std::clamp(
                std::max({
                    risk,
                    ambiguityNorm,
                    conflictNorm,
                    0.45 * lurchGate,
                    patchSelected ? 0.20 : 0.0
                }),
                0.0,
                1.0);

            const double yBaseConfidence =
                1.0 / (1.0 + floorWidthIRE / 4.0);

            const double lurchConfidence =
                lurchGate *
                (1.0 / (1.0 + maxMembershipDeltaIRE / 12.0));

            const double riskPenalty = 1.0 - 0.45 * risk;
            const double ambiguityPenalty = 1.0 - 0.50 * ambiguityNorm;
            const double compactRepairPenalty =
                patchSelected ? (1.0 - 0.25 * compactPatchGate) : 1.0;

            double yConfidence = std::clamp(
                std::max(yBaseConfidence, lurchConfidence)
                * riskPenalty
                * ambiguityPenalty
                * compactRepairPenalty
                * (1.0 - 0.35 * conflictNorm),
                0.0,
                1.0);

            // Low-chroma 1D/lurch agreement is strong corroboration: two
            // independent witnesses concurring where the comb confirms little
            // real chroma. (Latent until a consumer weights yWitnessConfidence;
            // produceY currently commits to yWitness unconditionally.)
            const double agreementGate = 1.0 - smoothStep01(oneDDeltaIRE / 2.0);
            const double agreementBoost = trust1D * agreementGate;
            yConfidence = std::clamp(
                yConfidence + 0.35 * agreementBoost * (1.0 - yConfidence),
                0.0, 1.0);

            const double carrierConfidence = std::clamp(
                (double)linePlausibility
                * (1.0 - 0.60 * risk)
                * (1.0 - 0.55 * ambiguityNorm)
                * (1.0 - 0.35 * conflictNorm),
                0.0,
                1.0);

            yWit[xi]    = static_cast<float>(yOut);
            cWit[xi]    = static_cast<float>(carrierOut);
            yConf[xi]   = static_cast<float>(yConfidence);
            cConf[xi]   = static_cast<float>(carrierConfidence);
            obsRisk[xi] = static_cast<float>(obstruction);
            ambig[xi]   = static_cast<float>(std::max(0.0, ambiguityIRE));

            /*
             * Attribution event mask. >0.5 records the bounded compact
             * transfer so candidate construction cannot use the repair's
             * carrier complement to certify the same decision.
             */
            corrMask[xi] = patchSelected ? 1.0f
                         : (contourBlended ? 0.4f : 0.0f);
        }

        if (patchDiagEnabled) {
            int runStart = -1;
            for (int xi = 0; xi <= width; ++xi) {
                const bool on = (xi < width) && (envScratch[xi] > 2.5);
                if (on && runStart < 0) {
                    runStart = xi;
                } else if (!on && runStart >= 0) {
                    const int runLen = xi - runStart;
                    if (runLen >= 4 && runLen <= 24) {
                        int sel = 0;
                        for (int k = runStart; k < xi; ++k) {
                            if (corrMask[k] > 0.5f)
                                ++sel;
                        }
                        const double cov = (double)sel / (double)runLen;
                        ++patchDiag.spans;
                        if (cov < 0.25)
                            ++patchDiag.spansLowCov;
                        else if (cov <= 0.75)
                            ++patchDiag.spansMidCov;
                        else
                            ++patchDiag.spansHighCov;
                    }
                    runStart = -1;
                }
            }
        }
    }

    /*
     * Interline cancellation on the witness carrier.
     *
     * carrierWitness = raw - yWitness is a composite-grid residual: wherever
     * the Y witness is imperfect (HF-Y between the coarse scale and the 1D
     * patch triggers, luma pedestal inside compact patches), the error rides
     * the carrier band.  Real chroma inverts between opposite-lineFlip
     * neighbors; witness luma error does not.  Cancellation therefore rejects
     * the trespass and preserves chroma.
     *
     * Rules carried over from the resolved combedCarrier checker bug:
     *   - only neighbors with opposite lineFlip participate (the observed
     *     flip pattern can be - + + -, so a frame-adjacent neighbor may be
     *     same-polarity);
     *   - the no-neighbor fallback is full scale, never half.
     *
     * Unlike Pass 2's hard switch, cancellation strength here scales with the
     * gates, so disagreement (one-line-tall color, vertical detail) degrades
     * smoothly toward the uncancelled witness instead of snapping.
     */
    {
        auto softReachGate = [](double diffIRE, double softIRE, double hardIRE) {
            if (diffIRE <= softIRE)
                return 1.0;
            if (diffIRE >= hardIRE)
                return 0.0;
            const double t = (diffIRE - softIRE) /
                             std::max(1e-9, hardIRE - softIRE);
            return 1.0 - (t * t * (3.0 - 2.0 * t));
        };

        for (int line = first; line < last; ++line) {
            const float *cwRow = carrierWitness_flat.data()
                               + static_cast<size_t>(line) * width;
            const float *ywRow = yWitness_flat.data()
                               + static_cast<size_t>(line) * width;
            float *outRow = carrierWitnessCombed_flat.data()
                          + static_cast<size_t>(line) * width;
            const float *maskRow = carrierCorrectionMask_flat.data()
                                 + static_cast<size_t>(line) * width;

            const auto *grammar = carrierGrammarLine(line);
            if (!grammar || !grammar->grammarLocked) {
                std::copy(cwRow, cwRow + width, outRow);
                continue;
            }

            const int lineAbove = line - 1;
            const int lineBelow = line + 1;

            const auto *gAbove =
                (lineAbove >= first) ? carrierGrammarLine(lineAbove) : nullptr;
            const auto *gBelow =
                (lineBelow < last) ? carrierGrammarLine(lineBelow) : nullptr;

            const bool haveAbove = gAbove && gAbove->grammarLocked &&
                (gAbove->lineFlip != grammar->lineFlip);
            const bool haveBelow = gBelow && gBelow->grammarLocked &&
                (gBelow->lineFlip != grammar->lineFlip);

            const float *cwAbove = haveAbove
                ? (carrierWitness_flat.data()
                   + static_cast<size_t>(lineAbove) * width)
                : nullptr;
            const float *cwBelow = haveBelow
                ? (carrierWitness_flat.data()
                   + static_cast<size_t>(lineBelow) * width)
                : nullptr;
            const float *ywAbove = haveAbove
                ? (yWitness_flat.data()
                   + static_cast<size_t>(lineAbove) * width)
                : nullptr;
            const float *ywBelow = haveBelow
                ? (yWitness_flat.data()
                   + static_cast<size_t>(lineBelow) * width)
                : nullptr;

            if (!cwAbove && !cwBelow) {
                std::copy(cwRow, cwRow + width, outRow);
                continue;
            }

            /*
             * Precompute center quadrature envelope per pixel.
             *
             * The 2-sample envelope is flat across the carrier cycle and
             * avoids the per-sample gate oscillation that would inject
             * checkerboard into the comb.  Computing it once per line
             * saves a redundant sqrt per neighbor.
             */
            for (int xi = 0; xi < width; ++xi) {
                const int xj = std::min(xi + 1, width - 1);
                const double c0 = static_cast<double>(cwRow[xi]);
                const double c1 = static_cast<double>(cwRow[xj]);
                envScratch[xi] = std::sqrt(c0 * c0 + c1 * c1);
            }

            auto neighborGate = [&](int xi, const float *neighborCw,
                                    const float *neighborYw) -> double {
                if (!neighborCw || !neighborYw)
                    return 0.0;

                const double lumaDiffIRE =
                    std::fabs(static_cast<double>(ywRow[xi]) -
                              static_cast<double>(neighborYw[xi])) * invIreScale;

                const int xj = std::min(xi + 1, width - 1);
                const double c0 = static_cast<double>(cwRow[xi]);
                const double c1 = static_cast<double>(cwRow[xj]);
                const double n0 = static_cast<double>(neighborCw[xi]);
                const double n1 = static_cast<double>(neighborCw[xj]);

                // A vertical structure (vertical edge crossed at 90 deg) is a
                // horizontal HF luma event whose 1D cross-color flips sign
                // line-to-line while the luma itself is the SAME on adjacent
                // lines.  In the carrier residual that alien is NON-INVERTING
                // (c ~ +n), and 0.5(c - n) cancels it — but |c+n| alone reads
                // c ~ +n as mismatch and backs off, so the comb could not
                // cancel exactly this case.  min(|c+n|, |c-n|) also admits the
                // non-inverting relation: inverting (c ~ -n) is real interline
                // chroma the comb preserves; non-inverting is alien luma it
                // cancels.  This does NOT reach horizontal details: lumaGate
                // (unchanged below) still vetoes the risky large vertical
                // reach where adjacent-line luma genuinely differs.
                const double s0 = c0 + n0, s1 = c1 + n1;
                const double d0 = c0 - n0, d1 = c1 - n1;
                const double carrierMismatchIRE =
                    std::min(std::sqrt(s0 * s0 + s1 * s1),
                             std::sqrt(d0 * d0 + d1 * d1)) * invIreScale;
                const double carrierAmpIRE = 0.5 *
                    (envScratch[xi] +
                     std::sqrt(n0 * n0 + n1 * n1)) * invIreScale;

                const double lumaGate = softReachGate(lumaDiffIRE, 3.0, 10.0);
                const double carrierGate = softReachGate(
                    carrierMismatchIRE,
                    std::max(3.0, 0.25 * carrierAmpIRE),
                    std::max(10.0, 0.80 * carrierAmpIRE));

                return lumaGate * carrierGate;
            };

            for (int xi = 0; xi < width; ++xi) {
                const double wAbove = neighborGate(xi, cwAbove, ywAbove);
                const double wBelow = neighborGate(xi, cwBelow, ywBelow);
                const double wSum = wAbove + wBelow;

                const double center = static_cast<double>(cwRow[xi]);

                if (wSum > 1e-9) {
                    double neighborMix = 0.0;
                    if (wAbove > 0.0)
                        neighborMix += wAbove *
                            static_cast<double>(cwAbove[xi]);
                    if (wBelow > 0.0)
                        neighborMix += wBelow *
                            static_cast<double>(cwBelow[xi]);
                    neighborMix /= wSum;

                    const double cancelled = 0.5 * (center - neighborMix);
                    const double strength = std::min(1.0, wSum);

                    outRow[xi] = static_cast<float>(
                        center * (1.0 - strength) + cancelled * strength);
                } else {
                    outRow[xi] = static_cast<float>(center);
                }

                if (patchDiagEnabled && maskRow[xi] > 0.5f) {
                    patchDiag.retainCwSum += std::fabs(center);
                    patchDiag.retainCombedSum +=
                        std::fabs(static_cast<double>(outRow[xi]));
                    ++patchDiag.retainN;
                }
            }
        }
    }

    if (patchDiagEnabled) {
        // Summary prints even with zero candidates: a silent frame would be
        // indistinguishable from a logging failure, and candidates=0 is
        // itself the finding (the env/span candidate filter excludes the
        // patch entirely).
        qInfo("WitnessPatchGate: candidates=%lld selected=%lld "
              "binding[env=%lld legal=%lld span=%lld floor=%lld lattice=%lld veto=%lld]",
              patchDiag.candidates, patchDiag.selected,
              patchDiag.bindEnv, patchDiag.bindLegal, patchDiag.bindSpan,
              patchDiag.bindFloor, patchDiag.bindLattice, patchDiag.bindVeto);

        if (patchDiag.candidates > 0) {
            qInfo("WitnessPatchGate best: line=%d x=%d compact=%.3f sel=%c "
                  "envIRE=%.2f envGate=%.3f cost=%.2f legal=%.3f "
                  "spanLen=%.0f spanGate=%.3f floorW=%.2f floorGate=%.3f "
                  "lattice=%.3f sharp=%.3f veto=%.3f delta=%.2f",
                  patchDiag.bestLine, patchDiag.bestX,
                  patchDiag.bCompact, patchDiag.bSelected ? 'Y' : 'n',
                  patchDiag.bEnvIRE, patchDiag.bEnvGate,
                  patchDiag.bCost, patchDiag.bLegal,
                  patchDiag.bSpanLen, patchDiag.bSpanGate,
                  patchDiag.bFloorW, patchDiag.bFloorGate,
                  patchDiag.bLatticeGate, patchDiag.bSharp,
                  patchDiag.bVetoGate, patchDiag.bDeltaIRE);
        }

        const double combRetain = (patchDiag.retainCwSum > 1e-9)
            ? (patchDiag.retainCombedSum / patchDiag.retainCwSum)
            : -1.0;
        qInfo("WitnessPatchGate spans(env>2.5,4..24px): n=%lld "
              "cov[<25%%=%lld 25-75%%=%lld >75%%=%lld] combRetain=%.3f (n=%lld)",
              patchDiag.spans, patchDiag.spansLowCov, patchDiag.spansMidCov,
              patchDiag.spansHighCov, combRetain, patchDiag.retainN);

        /*
         * Pinpoint window dump, once per process.
         *
         * --witness-dump firstLine,lastLine,x0,x1 (frame lines, active-area
         * x) emits a per-pixel table for that window from the first frame
         * that renders it: yWitness, witness carrier, combed carrier, mask
         * state.  This is the instrument for looking at THE patch rather
         * than frame aggregates.  Window is capped at 32 lines x 64 px.
         */
        if (false) {
            int l0 = std::clamp(first, first, last - 1);
            int l1 = std::clamp(first, first, last - 1);
            int x0 = std::clamp(0, 0, width - 1);
            int x1 = std::clamp(0, 0, width - 1);
            if (l1 < l0) std::swap(l0, l1);
            if (x1 < x0) std::swap(x0, x1);
            l1 = std::min(l1, l0 + 31);
            x1 = std::min(x1, x0 + 63);

            witnessWindowDumped = true;
            qInfo("WitnessDump window lines %d..%d x %d..%d "
                  "(per pixel: yW/cw/combed/mask, mask #=patch +=contour .=none)",
                  l0, l1, x0, x1);

            for (int line = l0; line <= l1; ++line) {
                const float *cwRow = carrierWitness_flat.data()
                                   + static_cast<size_t>(line) * width;
                const float *cbRow = carrierWitnessCombed_flat.data()
                                   + static_cast<size_t>(line) * width;
                const float *mRow = carrierCorrectionMask_flat.data()
                                  + static_cast<size_t>(line) * width;
                const quint16 *rawLine = rawbuffer.data()
                                       + static_cast<size_t>(line) * fullWidth;

                QString row;
                for (int xi = x0; xi <= x1; ++xi) {
                    const double cw = static_cast<double>(cwRow[xi]) * invIreScale;
                    const double cb = static_cast<double>(cbRow[xi]) * invIreScale;
                    const double yw = (static_cast<double>(rawLine[left + xi])
                                       - static_cast<double>(cwRow[xi])
                                       - (double)videoParameters.black16bIre)
                                      * invIreScale;
                    const char m = (mRow[xi] > 0.5f) ? '#'
                                 : (mRow[xi] > 0.2f) ? '+' : '.';
                    row += QString(" %1:%2/%3/%4%5")
                        .arg(xi)
                        .arg(yw, 0, 'f', 1)
                        .arg(cw, 0, 'f', 1)
                        .arg(cb, 0, 'f', 1)
                        .arg(m);
                }
                qInfo("WitnessDump L%d:%s", line, row.toUtf8().constData());
            }
        }
    }

    witnessValid = true;
}
