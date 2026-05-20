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

// Immutable ownership-rule defaults.
//
// These rules define how claim sets are normalized and how competing luma
// and chroma attributions interact. Decoder-specific code may create local
// software redefinitions of these rules, but the shared header establishes
// the canonical defaults and vocabulary.
enum class OwnershipConflictRule {
    GeometricMean,
};

enum class OwnershipUncertaintyRule {
    OneMinusMaxClaim,
};

enum class OwnershipClaimMergeRule {
    Max,
};

struct OwnershipRules {
    OwnershipConflictRule conflictRule = OwnershipConflictRule::GeometricMean;
    OwnershipUncertaintyRule uncertaintyRule = OwnershipUncertaintyRule::OneMinusMaxClaim;
    OwnershipClaimMergeRule chromaMergeRule = OwnershipClaimMergeRule::Max;
    double conflictSuppress = 0.65;
};

inline constexpr OwnershipRules kDefaultOwnershipRules{};

// Comb ownership facts are narrow observed/derived measurements only. They are
// evidence, not conclusions.
struct CombOwnershipFacts {
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
};

// Comb ownership assessment is the software interpretation of the facts under
// a chosen rule set. This is where claims and decoder-local reinterpretations
// belong.
struct CombOwnershipAssessment {
    double lumaRisk = 0.0;
    double lumaResidual = 0.0;
    double baseSupport = 0.0;
    double neighborSupport = 0.0;
    double lumaShapeContinuation = 0.0;
    double chromaStrength = 0.0;
    double coherence = 0.0;
    double agreement = 0.0;
    double spreadPenalty = 0.0;
    double carrierPrior = 0.0;
    double carrierPlausibility = 0.0;

    double ownershipConflict = 0.0;
    double lumaClaim = 0.0;
    double chromaClaim = 0.0;
    double uncertainClaim = 1.0;
};

struct CombOwnershipRecord {
    CombOwnershipFacts facts;
    CombOwnershipAssessment assessment;
};

// Transitional alias: existing code may still refer to OwnershipEvidence while
// the implementation is moved over to facts + assessment explicitly.
using CombOwnershipEvidence = CombOwnershipRecord;


// Composite ownership facts are the richer, parser-facing observed/derived
// measurements used by the model-based decoder branch.
struct CompositeOwnershipFacts {
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
};

// Composite ownership assessment holds the software interpretation of the
// facts under a chosen rule set. It includes both decomposed chroma claims
// and intermediate model-interpretation terms.
struct CompositeOwnershipAssessment {
    double modelAgreementClaim = 0.0;
    double waveformClaimConflict = 0.0;
    double lumaShapeContinuation = 0.0;
    double carrierPlausibility = 0.0;
    double ownershipConflict = 0.0;
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

struct CompositeOwnershipRecord {
    CompositeOwnershipFacts facts;
    CompositeOwnershipAssessment assessment;
};

using CompositeOwnershipEvidence = CompositeOwnershipRecord;

inline double clamp01(double v)
{
    return std::clamp(v, 0.0, 1.0);
}

inline double combinedCarrierChromaCoherence(const CompositeOwnershipFacts &f)
{
    return std::max({
        f.carrierChromaCoherence,
        f.carrierPhaseCoherence,
        f.compositeChromaCoherence,
        f.compositeCarrierCoherence,
        f.compositeLinePatternCoherence,
        f.compositeFieldCoherence,
        f.compositeBoundaryCoherence,
        f.compositeStableSideCoherence,
        f.sidebandCoherence
    });
}

inline double combinedEnvelopeChromaCoherence(const CompositeOwnershipFacts &f)
{
    return std::max({
        f.iqChromaCoherence,
        f.iqEnvelopeCoherence,
        f.chromaEnvelopeCoherence
    });
}

inline double strongestChromaCoherence(const CompositeOwnershipFacts &f)
{
    return std::max({
        combinedCarrierChromaCoherence(f),
        combinedEnvelopeChromaCoherence(f)
    });
}

inline double combinedCompositeChromaClaim(const CompositeOwnershipAssessment &a,
                                           const OwnershipRules &rules = kDefaultOwnershipRules)
{
    switch (rules.chromaMergeRule) {
    case OwnershipClaimMergeRule::Max:
    default:
        return std::max({
            a.carrierChromaClaim,
            a.envelopeChromaClaim,
            a.sidebandChromaClaim,
            a.compositeChromaClaim,
            a.iqEnvelopeClaim
        });
    }
}

inline double computeOwnershipConflict(double lumaClaim,
                                       double chromaClaim,
                                       const OwnershipRules &rules = kDefaultOwnershipRules)
{
    switch (rules.conflictRule) {
    case OwnershipConflictRule::GeometricMean:
    default:
        return std::sqrt(std::max(0.0, clamp01(lumaClaim) * clamp01(chromaClaim)));
    }
}

inline double computeUncertainClaim(double lumaClaim,
                                    double chromaClaim,
                                    const OwnershipRules &rules = kDefaultOwnershipRules)
{
    switch (rules.uncertaintyRule) {
    case OwnershipUncertaintyRule::OneMinusMaxClaim:
    default:
        return clamp01(1.0 - std::max(clamp01(lumaClaim), clamp01(chromaClaim)));
    }
}

inline void normalizeCompositeOwnershipAssessment(CompositeOwnershipAssessment &a,
                                                  const OwnershipRules &rules = kDefaultOwnershipRules)
{
    a.lumaClaim = clamp01(a.lumaClaim);

    a.carrierChromaClaim = clamp01(a.carrierChromaClaim);
    a.envelopeChromaClaim = clamp01(a.envelopeChromaClaim);
    a.sidebandChromaClaim = clamp01(a.sidebandChromaClaim);
    a.compositeChromaClaim = clamp01(a.compositeChromaClaim);
    a.iqEnvelopeClaim = clamp01(a.iqEnvelopeClaim);

    const double chromaClaim = clamp01(combinedCompositeChromaClaim(a, rules));
    a.uncertainClaim = computeUncertainClaim(a.lumaClaim, chromaClaim, rules);
}

inline void applyOwnershipConflictSuppression(CompositeOwnershipAssessment &a,
                                              const OwnershipRules &rules = kDefaultOwnershipRules)
{
    a.lumaClaim = clamp01(a.lumaClaim);

    const double chromaClaim = clamp01(combinedCompositeChromaClaim(a, rules));
    const double conflict = computeOwnershipConflict(a.lumaClaim, chromaClaim, rules);

    a.ownershipConflict = std::max(a.ownershipConflict, conflict);

    const double scale = std::max(
        0.0,
        1.0 - clamp01(rules.conflictSuppress) * conflict);

    a.lumaClaim *= scale;
    a.carrierChromaClaim *= scale;
    a.envelopeChromaClaim *= scale;
    a.sidebandChromaClaim *= scale;
    a.compositeChromaClaim *= scale;
    a.iqEnvelopeClaim *= scale;

    normalizeCompositeOwnershipAssessment(a, rules);
}

inline double strongestCombChromaIRE(const CombOwnershipFacts &f)
{
    return std::max({
        f.fieldAChromaIRE,
        f.fieldBChromaIRE,
        f.frameChromaIRE,
        f.locked1DChromaIRE
    });
}

inline void normalizeCombOwnershipAssessment(CombOwnershipAssessment &a,
                                             const OwnershipRules &rules = kDefaultOwnershipRules)
{
    a.lumaClaim = clamp01(a.lumaClaim);
    a.chromaClaim = clamp01(a.chromaClaim);
    a.uncertainClaim = computeUncertainClaim(a.lumaClaim, a.chromaClaim, rules);
}

inline void applyOwnershipConflictSuppression(CombOwnershipAssessment &a,
                                              const OwnershipRules &rules = kDefaultOwnershipRules)
{
    a.lumaClaim = clamp01(a.lumaClaim);
    a.chromaClaim = clamp01(a.chromaClaim);

    const double conflict = computeOwnershipConflict(a.lumaClaim, a.chromaClaim, rules);

    a.ownershipConflict = std::max(a.ownershipConflict, conflict);

    const double scale = std::max(
        0.0,
        1.0 - clamp01(rules.conflictSuppress) * conflict);

    a.lumaClaim *= scale;
    a.chromaClaim *= scale;

    normalizeCombOwnershipAssessment(a, rules);
}

} // namespace lddecode
