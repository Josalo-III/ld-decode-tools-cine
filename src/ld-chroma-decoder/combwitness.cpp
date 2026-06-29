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
 ******************************************************************************/

#include "comb.h"
#include "feasibleband.h"
#include <algorithm>
#include <cmath>

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
    ensure(compactPatchGate_flat);

    if ((int)scratch_lumaBaseY4.size() < width) scratch_lumaBaseY4.resize(width, 0.0);
    if ((int)scratch_lumaSmooth.size()  < width) scratch_lumaSmooth.resize(width, 0.0);
    if ((int)scratch_lineWorkA.size()   < width) scratch_lineWorkA.resize(width, 0.0);
    if ((int)scratch_lineWorkB.size()   < width) scratch_lineWorkB.resize(width, 0.0);
    if ((int)scratch_lineWorkC.size()   < width) scratch_lineWorkC.resize(width, 0.0);
    if ((int)scratch_envLine.size()     < width) scratch_envLine.resize(width, 0.0);
    if ((int)scratch_spanLine.size()    < width) scratch_spanLine.resize(width, 0.0);
    if ((int)scratch_lurchGate.size()     < width) scratch_lurchGate.resize(width, 0.0);
    if ((int)scratch_lurchCurve.size()    < width) scratch_lurchCurve.resize(width, 0.0);

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

        float *yWit    = yWitness_flat.data()
                       + static_cast<size_t>(line) * width;
        float *patchGateRow = compactPatchGate_flat.data()
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

        /*
         * Lurch membership pre-pass.
         *
         * Reduce each pixel's four-view membership evidence to a transition
         * gate (support x delta). The gate is not a correction — it localizes
         * where a genuine luma transition
         * sits, so the regression below knows where to free the curve from
         * the boxcar. The displacement magnitude itself is taken from the
         * carrier-free raw difference facts in the regression, not from this
         * membership estimate (facts, not estimates).
         */
        double *lurchGateArr = scratch_lurchGate.data();

        for (int xi = 0; xi < width; ++xi) {
            double maxSupport = 0.0;
            double maxDeltaIRE = 0.0;

            if (evidence && evidence[xi].viewCount > 0) {
                for (int v = 0; v < evidence[xi].viewCount; ++v) {
                    const auto &view = evidence[xi].views[v];
                    maxSupport = std::max(maxSupport,
                        std::clamp((double)view.membershipSupport, 0.0, 1.0));
                    maxDeltaIRE = std::max(maxDeltaIRE,
                        std::fabs((double)view.membershipDeltaIRE));
                }
            }

            const double supportGate = smoothStep01(
                (maxSupport - LURCH_SUPPORT_START) /
                std::max(1e-9, LURCH_SUPPORT_FULL - LURCH_SUPPORT_START));
            const double deltaGate = smoothStep01(
                (maxDeltaIRE - LURCH_DELTA_START_IRE) /
                std::max(1e-9, LURCH_DELTA_FULL_IRE - LURCH_DELTA_START_IRE));

            lurchGateArr[xi]     = supportGate * deltaGate;
        }

        /*
         * Lurch curve regression (whole-line, banded least squares).
         *
         * The conservation law (each four-sample block sums carrier-free to
         * its coarse total) and the lurch evidence are one fact in integral
         * vs. differential form: differencing two block sums one sample apart
         * gives Y[x+4] - Y[x] = raw[x+4] - raw[x], a same-phase pair whose
         * carrier cancels exactly. These difference facts pin the curve's HF
         * shape to ~1-sample edge placement (finer than the boxcar); sharpPrior
         * anchors the LF. We SOLVE the curve rather than average a displacement,
         * so it stays associated with the composite and never deposits a
         * decoupled per-pixel step (the old additive correction's notch / fSC
         * beat).
         *
         * The anchor is sharpPrior, NOT movingCoarse. movingCoarse is the
         * phase-invariant evidence / membership reference (the boxcar the
         * difference facts and lurchGate are measured against); sharpPrior is
         * the reconstruction prior, already edge-snapped by the preconditioner
         * and purpose-built as the solved-Y anchor. Anchoring to the blurry
         * boxcar would make the anchor fight the difference facts at every
         * edge; sharpPrior and the difference facts agree there.
         *
         * Minimise over the line:
         *     wA  * sum_x ( Y[x+4] - Y[x] - d[x] )^2        d[x] = raw[x+4]-raw[x]
         *   + wB[x] * ( Y[x] - sharpPrior[x] )^2
         * subject to Y[x] within the structural carrier-amplitude band
         * (feasibility eliminates impossible carrier excursions).
         *
         * wB relaxes toward a floor where lurchGate fires (a real luma
         * transition): the curve is freed to follow the difference facts'
         * placement. At a chroma-only transition the boxcar mean does not move,
         * lurchGate stays ~0, wB stays firm, and the difference fact --
         * contaminated there by the carrier amplitude change -- is correctly
         * overruled by the anchor.
         *
         * Each Y[x] couples only to Y[x+-4], so the system decouples into four
         * independent tridiagonal chains by phase (x mod 4). Projected
         * Gauss-Seidel sweeps solve all four in place, the feasibility box
         * applied at every update. O(width) per line.
         */
        double *lurchCurve = scratch_lurchCurve.data();
        {
            constexpr double wA      = 1.0;   // carrier-free difference facts
            constexpr double wBbase  = 0.5;   // sharpPrior anchor, smooth regions
            constexpr double wBfloor = 0.04;  // anchor floor at luma edges
            constexpr int    sweeps  = 14;

            const CombCarrierGrammar *grammar = carrierGrammarLine(line);
            const double maxCarrierAmp =
                (grammar ? std::max(24.0, grammar->carrierScale * 5.0)
                         : 24.0) * ireToSamples;

            for (int xi = 0; xi < width; ++xi)
                lurchCurve[xi] = sharpPrior[xi];

            for (int it = 0; it < sweeps; ++it) {
                // Alternate sweep direction so neither end is favoured (a
                // one-way Gauss-Seidel pass would bias the curve laterally).
                const bool fwd = (it & 1) == 0;
                for (int s = 0; s < width; ++s) {
                    const int xi = fwd ? s : (width - 1 - s);
                    const double anchorW =
                        wBfloor + (wBbase - wBfloor) *
                        (1.0 - std::clamp(lurchGateArr[xi], 0.0, 1.0));

                    double diag = anchorW;
                    double rhs  = anchorW * sharpPrior[xi];

                    if (xi + 4 < width) {
                        const double d = (double)rawLine[left + xi + 4]
                                       - (double)rawLine[left + xi];
                        diag += wA;
                        rhs  += wA * (lurchCurve[xi + 4] - d);
                    }
                    if (xi - 4 >= 0) {
                        const double d = (double)rawLine[left + xi]
                                       - (double)rawLine[left + xi - 4];
                        diag += wA;
                        rhs  += wA * (lurchCurve[xi - 4] + d);
                    }

                    double y = (diag > 1e-12) ? rhs / diag : sharpPrior[xi];

                    // Feasibility projection: Y cannot imply a carrier
                    // excursion beyond the structural ceiling.
                    const double rawX = (double)rawLine[left + xi];
                    if (y > rawX + maxCarrierAmp)      y = rawX + maxCarrierAmp;
                    else if (y < rawX - maxCarrierAmp) y = rawX - maxCarrierAmp;

                    lurchCurve[xi] = y;
                }
            }
        }

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
             * Lurch observations (pre-pass values).
             *
             * The curve refinement itself was solved line-wide in the
             * regression pre-pass (lurchCurve). Here we carry the transition
             * gate forward.
             */
            const double lurchGate             = lurchGateArr[xi];

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
            patchGateRow[xi] = static_cast<float>(compactPatchGate);

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
             * Lurch-shaped prior.
             *
             * The whole-line regression already solved the carrier-free,
             * feasibility-projected curve: coarse owns the LF, the same-phase
             * difference facts own the HF edge placement. Adopt it directly,
             * clamped to the local possible band. No additive per-pixel
             * correction is applied here -- the curve is solved, not summed,
             * so it stays associated with the composite (an averaged
             * displacement would decouple and beat at fSC).
             *
             * Compact detection does not alter this boundary model; any
             * compact repair is applied after this path is complete.
             */
            double yLurch = possibleBand.clamp(lurchCurve[xi]);

            // 1D cooperation: when the curve and the phase-reliable 1D estimate
            // diverge, trust1D arbitrates — low chroma favors 1D (no carrier
            // leak), high chroma keeps the curve. Agreement leaves it untouched.
            {
                const double y1DRef = possibleBand.clamp(oneDUnclamped);
                const double divergeIRE =
                    std::fabs(yLurch - y1DRef) * invIreScale;
                const double coopGate = smoothStep01(
                    (divergeIRE - 1.0) / 3.0);
                const double coopBlend = trust1D * coopGate;
                if (coopBlend > 0.0)
                    yLurch = yLurch * (1.0 - coopBlend)
                           + y1DRef * coopBlend;
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

            double yOut = yLurch;
            bool patchSelected = false;

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
                }
            }

            yOut = patchSelected ? hardClamp(yOut) : possibleBand.clamp(yOut);

            yWit[xi] = static_cast<float>(yOut);
        }
    }

    witnessValid = true;
}
