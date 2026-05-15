/******************************************************************************
 * compositeownershipdefs.h
 * ld-decode-tools shared composite/luma/chroma ownership definitions
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This header defines the shared attribution vocabulary used by composite
 * decoding stages.  It intentionally contains evidence storage and small
 * normalization helpers only; decoder-specific scoring policy should remain
 * in the owning decoder.
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace lddecode {

// ---------------------------------------------------------------------------
// Stateful composite parser types
//
// CompositeParserState is per-line and represents the calibrated carrier
// grammar in which chroma claims are allowed to be evaluated.  The carrier
// is a line-level fact, not a per-sample discovery: one burst phasor,
// one alternation sign, one sample phase grid apply uniformly to every
// active sample on the line.  These types push that fact down to per-sample
// consumers, so downstream stages do not re-derive the carrier locally.
//
// The three-tier parser fills the state in stages:
//
//   tier 1  carrier grammar lock   - burst phasor, line-level carrier
//                                    scale and phase calibration from a
//                                    demod/remod round trip on the raw
//                                    composite (no Y subtraction needed)
//   tier 2  pattern parse          - segment the line into candidate spans
//                                    inside the calibrated grammar
//   tier 3  IQ / forward-model     - adjudicate whether a span's reading
//                                    is consistent with a remodulated
//                                    waveform; project verdict into
//                                    per-sample ownership evidence
//
// Field/frame schedule context (fieldPhaseId, lineParity,
// frameVerticalAllowed) frames the line but the line is the carrier-
// grammar unit: the larger schedule sets the line's phase, the line
// dictates the per-sample interpretation.
// ---------------------------------------------------------------------------

struct CompositeSpan {
    int x0 = 0;
    int x1 = 0;

    // Tier 2: pattern-parse evidence within the locked grammar.
    double carrierFit            = 0.0;
    double samePhaseRecurrence   = 0.0;
    double alternationCoherence  = 0.0;

    // Tier 3: IQ / forward-model adjudication.
    double iqEnvelopeCoherence   = 0.0;
    double remodAgreement        = 0.0;
    double forwardErrorIRE       = 0.0;  // mean forward-model error, IRE

    // Counter-evidence and arbitration.
    double lumaCounterEvidence   = 0.0;
    double parserConflict        = 0.0;  // carrier says yes, IQ/remod says no

    // Final claim projected back into ownership evidence by the caller.
    double chromaClaim           = 0.0;
};

struct CompositeParserState {
    // -----------------------------------------------------------------
    // Identity and schedule context.
    //
    // fieldPhaseId / lineParity / frameVerticalAllowed are the line's
    // place in the field/frame schedule.  They constrain the carrier
    // phasor but do not replace the per-line burst measurement.
    // -----------------------------------------------------------------
    int  line                    = 0;
    int  fieldPhaseId            = 0;
    int  lineParity              = 0;
    bool frameVerticalAllowed    = false;

    // -----------------------------------------------------------------
    // Tier 1: carrier grammar lock.
    //
    // burstCos / burstSin    unit-magnitude carrier phasor for the line
    //                        (raw burst direction, basis-shifted).
    //
    // carrierScale           burst amplitude in IRE.  This is the line-
    //                        level carrier scale recovered from the
    //                        pre-normalization burst magnitude.  Healthy
    //                        lines have ~7-15 IRE here.
    //
    // phaseError             systematic phase bias in carrier-phase
    //                        radians, measured by a line-level
    //                        demod/remod round trip against the raw
    //                        composite.  Capped to ±π/8.  Measured but
    //                        not yet applied to the basis LUTs; downstream
    //                        consumers may rotate IQ vectors by this
    //                        amount to achieve phase invariance.
    //
    // phaseConfidence        gated on carrierScale being plausible
    //                        (ramps in across 3-10 IRE burst magnitude).
    //
    // samplePhase0           h & 3 reference for the line's basis
    //                        (currently always 0; reserved for future
    //                        line-level sample-phase calibration).
    //
    // lineFlip               +1 or -1, alternation sign from line/field
    //                        parity.  Applied as lineScale to all
    //                        remodulation calls for the line.
    //
    // grammarLocked          true when phaseConfidence is above the
    //                        usable threshold; downstream tiers should
    //                        skip lines where this is false.
    // -----------------------------------------------------------------
    double burstCos              = 1.0;
    double burstSin              = 0.0;
    double carrierScale          = 0.0;  // IRE
    double phaseError            = 0.0;  // carrier-phase radians
    double phaseConfidence       = 0.0;  // [0,1]

    int samplePhase0             = 0;
    int lineFlip                 = +1;
    bool grammarLocked           = false;

    // -----------------------------------------------------------------
    // Tier 2 / tier 3 line-level summaries.  All in [0,1].
    // -----------------------------------------------------------------
    double linePatternConfidence   = 0.0;  // weighted mean span carrier fit
    double stableChromaConfidence  = 0.0;  // best span chromaClaim
    double iqValidationConfidence  = 0.0;  // weighted mean IQ envelope coh.
    double forwardAgreement        = 0.0;  // weighted mean remod agreement

    // Span segmentation (filled by tier 2; consumed by tier 3 and the
    // ownership projection).
    std::vector<CompositeSpan> spans;
};

// CombOwnershipEvidence is the lean, comb-decoder-facing ownership record.
// Keep this separate from the richer composite-model record below so the
// chroma decoder can manage its own memory layout and evidence vocabulary
// without pulling in parser-only fields it does not produce.
struct CombOwnershipEvidence {
    double bandpassFineIRE = 0.0;
    double bandpassMidIRE = 0.0;
    double bandpassCoarseIRE = 0.0;

    double fieldAChromaIRE = 0.0;
    double fieldBChromaIRE = 0.0;
    double frameChromaIRE = 0.0;
    double locked1DChromaIRE = 0.0;

    double frameIQCoherence = 0.0;
    double frameFieldAgreementIRE = 0.0;
    double candidateSpreadIRE = 0.0;

    double lumaExcursionIRE = 0.0;
    double residualFitErrorIRE = 0.0;
    double lumaIncursionRiskIRE = 0.0;
    double icebergAlienYFraction = 0.0;
    double lumaShapeContinuation = 0.0;

    double carrierScaleIRE = 0.0;
    double carrierPhaseErrorRad = 0.0;
    double carrierPhaseConfidence = 0.0;
    double carrierPlausibility = 0.0;

    double ownershipConflict = 0.0;
    double lumaClaim = 0.0;
    double chromaClaim = 0.0;
    double uncertainClaim = 1.0;
};


// CompositeOwnershipEvidence is the richer, parser-facing ownership record
// used by the composite decoder branch.  It carries the more decomposed
// carrier/envelope/sideband vocabulary and should stay separate from the
// comb-specific record above.
struct CompositeOwnershipEvidence {
    // ---------------------------------------------------------------------
    // Luma / residual waveform evidence
    // ---------------------------------------------------------------------

    // Generic residual evidence used by the luma-first composite decoder.
    double residualFineIRE = 0.0;
    double residualMidIRE = 0.0;
    double residualCoarseIRE = 0.0;

    double lumaExcursionIRE = 0.0;
    double residualFitErrorIRE = 0.0;
    double lumaIncursionRiskIRE = 0.0;
    double icebergAlienYFraction = 0.0;
    double lumaShapeContinuation = 0.0;

    // ---------------------------------------------------------------------
    // Composite / carrier-domain evidence
    // ---------------------------------------------------------------------

    // Historical/simple residual magnitude proxy.
    double compositeResidualIRE = 0.0;

    // Generic/PAL-safe carrier names.
    double carrierResidualIRE = 0.0;
    double carrierChromaFitIRE = 0.0;
    double carrierChromaErrorIRE = 0.0;
    double carrierChromaCoherence = 0.0;
    double carrierPhaseCoherence = 0.0;

    // Current composite-decoder NTSC names.
    double compositeChromaFitIRE = 0.0;
    double compositeChromaErrorIRE = 0.0;
    double compositeChromaCoherence = 0.0;
    double compositeCarrierCoherence = 0.0;
    double compositeLinePatternCoherence = 0.0;
    double compositeFieldCoherence = 0.0;
    double compositeBoundaryCoherence = 0.0;
    double compositeStableSideCoherence = 0.0;
    double compositePatternDepth = 0.0;
    double compositePatternLevels = 0.0;

    double carrierPlausibility = 0.0;

    // ---------------------------------------------------------------------
    // Early coarse-Y / quarter-stage evidence
    //
    // Captured immediately after the coarse Y anchor, before later Y
    // refinement mutates the original residual.  These are durable priors,
    // not final claims.
    // ---------------------------------------------------------------------

    double coarseY0IRE = 0.0;
    double quarterResidualIRE = 0.0;

    double quarterCarrierFitIRE = 0.0;
    double quarterCarrierErrorIRE = 0.0;
    double quarterCarrierCoherence = 0.0;

    double quarterIQMagIRE = 0.0;
    double quarterIQSmoothness = 0.0;
    double quarterIQChromaPrior = 0.0;
    double quarterStableChromaConstraint = 0.0;
    double quarterCheckerboardRisk = 0.0;

    double quarterLostYPeakIRE = 0.0;
    double quarterLostYPrior = 0.0;
    double quarterBoundaryPrior = 0.0;
    double quarterCoincidentTransition = 0.0;
    double quarterImpulseYPrior = 0.0;

    double quarterOffsetYDeltaIRE = 0.0;
    double quarterOffsetYPrior = 0.0;

    // Sliding 4fsc-window derivative evidence.
    //
    // For legal 4-sample means:
    //
    //     M[x]   = mean(raw[x+0], raw[x+1], raw[x+2], raw[x+3])
    //     D[x]   = 4 * (M[x+1] - M[x])
    //            = raw[x+4] - raw[x]
    //
    // This measures the same 4fsc phase entering/exiting the cancellation
    // window.  It is evidence, not a replacement Y image.
    double quarterSlidingDiffIRE = 0.0;
    double quarterSlidingCoherence = 0.0;
    double quarterSlidingPhaseDisorder = 0.0;
    double quarterSlidingYPrior = 0.0;

    // ---------------------------------------------------------------------
    // Chroma-envelope / NTSC IQ evidence
    // ---------------------------------------------------------------------

    // NTSC/IQ-specific evidence.  These names are kept because the current
    // NTSC implementation really is reasoning in I/Q space.
    double iqChromaMagIRE = 0.0;
    double iqEnvelopeSmoothness = 0.0;
    double iqExcessHF = 0.0;
    double iqRemodErrorIRE = 0.0;
    double iqChromaCoherence = 0.0;
    double iqEnvelopeCoherence = 0.0;
    double iqParserViolation = 0.0;

    // Generic/PAL-safe envelope aliases for future non-NTSC backends.
    double chromaEnvelopeMagIRE = 0.0;
    double chromaEnvelopeSmoothness = 0.0;
    double chromaEnvelopeExcessHF = 0.0;
    double chromaRemodErrorIRE = 0.0;
    double chromaEnvelopeCoherence = 0.0;

    // ---------------------------------------------------------------------
    // Sideband / carrier-centered legality evidence
    //
    // Chroma is not merely energy near the color subcarrier.  A legal chroma
    // explanation should show a structured relationship between residual
    // energy below and above the carrier.  The two sides need not have equal
    // amplitude; the useful score is whether they cohere into one plausible
    // chroma explanation.
    // ---------------------------------------------------------------------

    double sidebandLowerIRE = 0.0;
    double sidebandUpperIRE = 0.0;
    double sidebandCoherence = 0.0;
    double sidebandAsymmetry = 0.0;

    // ---------------------------------------------------------------------
    // Subtraction reconciliation evidence
    //
    // Used by model-based decoders to compare:
    //
    //     C_from_Y = rawComposite - refinedY
    //     Y_from_C = rawComposite - refinedChroma
    //
    // Agreement strengthens attribution confidence.  Disagreement should
    // generally raise uncertainty unless other evidence clearly resolves it.
    // ---------------------------------------------------------------------

    double subtractionAgreement = 0.0;
    double subtractionErrorIRE = 0.0;
    double reverseLumaIRE = 0.0;
    double forwardChromaIRE = 0.0;
    double modelAgreementClaim = 0.0;

    // Waveform-level forward-model diagnostics.
    //
    // These score the current hypothesis as a reconstructed composite:
    //
    //     rawComposite ~= Y_est + modulated(I_est, Q_est)
    //
    // The contribution support fields are not binary ownership decisions.
    // They describe how well the current Y and IQ waveform hypotheses
    // participate in explaining the local composite when judged together.
    double forwardModelResidualIRE = 0.0;
    double forwardModelErrorIRE = 0.0;
    double forwardModelAgreement = 0.0;
    double yContributionSupport = 0.0;
    double iqContributionSupport = 0.0;
    double lumaImpulseCaution = 0.0;
    double waveformClaimConflict = 0.0;

    // ---------------------------------------------------------------------
    // Attribution claims
    // ---------------------------------------------------------------------

    double ownershipConflict = 0.0;

    // Composite-model attribution claims.
    //
    // lumaClaim:
    //     Attribution that contested residual energy is better explained as
    //     luma than chroma.
    //
    // The chroma side is deliberately decomposed.  There is no generic
    // chromaClaim field here because the composite decoder should preserve
    // whether the claim came from carrier, envelope, sideband, or current
    // NTSC composite evidence.
    //
    // uncertainClaim:
    //     Remaining unattributed confidence.  The convention is:
    //
    //         uncertainClaim = 1 - max(lumaClaim, combined chroma claim)
    //
    double lumaClaim = 0.0;
    double uncertainClaim = 1.0;

    // Generic/PAL-safe decomposed chroma claims.
    double carrierChromaClaim = 0.0;
    double envelopeChromaClaim = 0.0;
    double sidebandChromaClaim = 0.0;

    // Current NTSC/composite decoder claim aliases.
    double compositeChromaClaim = 0.0;
    double iqEnvelopeClaim = 0.0;

};

inline double clamp01(double v)
{
    return std::clamp(v, 0.0, 1.0);
}

inline double combinedCarrierChromaCoherence(const CompositeOwnershipEvidence &e)
{
    return std::max({
        e.carrierChromaCoherence,
        e.carrierPhaseCoherence,
        e.compositeChromaCoherence,
        e.compositeCarrierCoherence,
        e.compositeLinePatternCoherence,
        e.compositeFieldCoherence,
        e.compositeBoundaryCoherence,
        e.compositeStableSideCoherence,
        e.carrierPlausibility,
        e.sidebandCoherence
    });
}

inline double combinedEnvelopeChromaCoherence(const CompositeOwnershipEvidence &e)
{
    return std::max({
        e.iqChromaCoherence,
        e.iqEnvelopeCoherence,
        e.chromaEnvelopeCoherence
    });
}

inline double strongestChromaCoherence(const CompositeOwnershipEvidence &e)
{
    return std::max({
        combinedCarrierChromaCoherence(e),
        combinedEnvelopeChromaCoherence(e)
    });
}

inline double combinedCompositeChromaClaim(const CompositeOwnershipEvidence &e)
{
    return std::max({
        e.carrierChromaClaim,
        e.envelopeChromaClaim,
        e.sidebandChromaClaim,
        e.compositeChromaClaim,
        e.iqEnvelopeClaim
    });
}

inline void normalizeCompositeOwnershipClaims(CompositeOwnershipEvidence &e)
{
    e.lumaClaim = clamp01(e.lumaClaim);

    e.carrierChromaClaim = clamp01(e.carrierChromaClaim);
    e.envelopeChromaClaim = clamp01(e.envelopeChromaClaim);
    e.sidebandChromaClaim = clamp01(e.sidebandChromaClaim);

    e.compositeChromaClaim = clamp01(e.compositeChromaClaim);
    e.iqEnvelopeClaim = clamp01(e.iqEnvelopeClaim);

    const double chromaClaim = clamp01(combinedCompositeChromaClaim(e));

    e.uncertainClaim = clamp01(
        1.0 - std::max(e.lumaClaim, chromaClaim));
}

inline void applyOwnershipConflictSuppression(CompositeOwnershipEvidence &e,
                                              double conflictSuppress)
{
    e.lumaClaim = clamp01(e.lumaClaim);

    const double chromaClaim = clamp01(combinedCompositeChromaClaim(e));

    const double conflict = std::sqrt(
        std::max(0.0, e.lumaClaim * chromaClaim));

    e.ownershipConflict = std::max(e.ownershipConflict, conflict);

    const double scale = std::max(
        0.0,
        1.0 - clamp01(conflictSuppress) * conflict);

    e.lumaClaim *= scale;

    e.carrierChromaClaim *= scale;
    e.envelopeChromaClaim *= scale;
    e.sidebandChromaClaim *= scale;
    e.compositeChromaClaim *= scale;
    e.iqEnvelopeClaim *= scale;

    normalizeCompositeOwnershipClaims(e);
}

inline double strongestCombChromaIRE(const CombOwnershipEvidence &e)
{
    return std::max({
        e.fieldAChromaIRE,
        e.fieldBChromaIRE,
        e.frameChromaIRE,
        e.locked1DChromaIRE
    });
}

inline void normalizeCombOwnershipClaims(CombOwnershipEvidence &e)
{
    e.lumaClaim = clamp01(e.lumaClaim);
    e.chromaClaim = clamp01(e.chromaClaim);
    e.uncertainClaim = clamp01(
        1.0 - std::max(e.lumaClaim, e.chromaClaim));
}

inline void applyOwnershipConflictSuppression(CombOwnershipEvidence &e,
                                              double conflictSuppress)
{
    e.lumaClaim = clamp01(e.lumaClaim);
    e.chromaClaim = clamp01(e.chromaClaim);

    const double conflict = std::sqrt(
        std::max(0.0, e.lumaClaim * e.chromaClaim));

    e.ownershipConflict = std::max(e.ownershipConflict, conflict);

    const double scale = std::max(
        0.0,
        1.0 - clamp01(conflictSuppress) * conflict);

    e.lumaClaim *= scale;
    e.chromaClaim *= scale;

    normalizeCombOwnershipClaims(e);
}

} // namespace lddecode
