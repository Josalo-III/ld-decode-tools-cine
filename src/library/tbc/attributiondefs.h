/******************************************************************************
 * attributiondefs.h
 * ld-decode-tools shared composite/luma/IQ attribution definitions
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * The Attribution system stores evidence and claims about which model update
 * is supported by the current waveform: luma, IQ/envelope chroma, composite
 * carrier-pattern chroma, sideband legality, conflict, or uncertainty.
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <cmath>

namespace lddecode {

inline double clamp01(double v)
{
    return std::clamp(v, 0.0, 1.0);
}

enum class AttributionClaimMergeRule {
    Max
};

enum class AttributionConflictRule {
    GeometricMean
};

enum class AttributionUncertaintyRule {
    OneMinusMaxClaim
};

struct AttributionRules {
    AttributionClaimMergeRule claimMergeRule = AttributionClaimMergeRule::Max;
    AttributionConflictRule conflictRule = AttributionConflictRule::GeometricMean;
    AttributionUncertaintyRule uncertaintyRule = AttributionUncertaintyRule::OneMinusMaxClaim;
    double conflictSuppress = 0.65;
};

inline constexpr AttributionRules kDefaultAttributionRules = {};

// Four legal 4fSC luma-floor views of the same raw waveform form a local
// parallax set.  The carrier model should come from the IQ component that is
// common across those views; disagreement between views is attribution evidence,
// not just a private decoder heuristic.
struct CarrierResidualConsensus {
    double lo = 0.0;
    double hi = 0.0;
    double trust = 0.0;
    bool valid = false;
};

struct FourViewCarrierView {
    double apertureCenter = 0.0;
    double yFloor = 0.0;
    double carrierSample = 0.0;
    double fittedSample = 0.0;
    double carrierI = 0.0;
    double carrierQ = 0.0;
    double sampleFitErrorIRE = 0.0;
    double remodErrorIRE = 0.0;
    double latticeRiskIRE = 0.0;
    double ySpanIRE = 0.0;
    double membershipDeltaIRE = 0.0;
    double membershipSupport = 0.0;
    double membershipLocalX = 0.0;
    double score = 0.0;
};

struct FourViewEvidenceView {
    float apertureCenter = 0.0f;
    float yFloor = 0.0f;
    float sampleFitErrorIRE = 0.0f;
    float remodErrorIRE = 0.0f;
    float latticeRiskIRE = 0.0f;
    float ySpanIRE = 0.0f;
    // Membership-change harvesting.
    //
    // These are observations, not policy.  They describe how the legal
    // carrier-cancelling luma floor moves as samples enter and leave adjacent
    // 4fSC apertures.
    float membershipDeltaIRE = 0.0f;     // same observation in IRE
    float membershipSupport = 0.0f;      // measurement quality only
};

struct FourViewPixelEvidence {
    int viewCount = 0;
    FourViewEvidenceView views[4];
};

struct FourViewCarrierAttribution {
    int viewCount = 0;
    bool valid = false;

    double ySpreadIRE = 0.0;
    double yCurvatureIRE = 0.0;

    double commonI = 0.0;
    double commonQ = 0.0;
    double commonSample = 0.0;
    double commonMagIRE = 0.0;
    double carrierSpreadIRE = 0.0;
    double carrierCoherence = 0.0;
    double sampleFitErrorIRE = 0.0;
    double sampleCoherence = 0.0;
    CarrierResidualConsensus residualConsensus;
    double latticeRiskIRE = 0.0;
};

inline FourViewCarrierAttribution buildFourViewCarrierAttribution(
    const FourViewCarrierView *views,
    int viewCount,
    double invIreScale)
{
    FourViewCarrierAttribution out;
    out.viewCount = std::clamp(viewCount, 0, 4);

    if (!views || out.viewCount <= 0)
        return out;

    out.valid = true;

    double minFloor = views[0].yFloor;
    double maxFloor = views[0].yFloor;
    for (int i = 0; i < out.viewCount; ++i) {
        minFloor = std::min(minFloor, views[i].yFloor);
        maxFloor = std::max(maxFloor, views[i].yFloor);
    }
    out.ySpreadIRE = (maxFloor - minFloor) * invIreScale;
    if (out.viewCount >= 3) {
        double curvSum = 0.0;
        int curvN = 0;
        for (int i = 1; i + 1 < out.viewCount; ++i) {
            curvSum += std::fabs(views[i - 1].yFloor -
                                 2.0 * views[i].yFloor +
                                 views[i + 1].yFloor);
            ++curvN;
        }
        out.yCurvatureIRE = (curvN > 0)
            ? (curvSum / static_cast<double>(curvN)) * invIreScale
            : 0.0;
    }

    // Precompute the symmetric IQ-distance matrix once.  With viewCount ≤ 4
    // this means at most 6 hypot calls instead of viewCount² (= 16) when
    // looped naively below, and the same matrix is reused for the spread
    // computation later — together cutting parallax hypot calls from ~20
    // per pixel to ~6 without changing any output.
    double iqDist[4][4] = {{0.0, 0.0, 0.0, 0.0},
                            {0.0, 0.0, 0.0, 0.0},
                            {0.0, 0.0, 0.0, 0.0},
                            {0.0, 0.0, 0.0, 0.0}};
    for (int i = 0; i < out.viewCount; ++i) {
        for (int j = i + 1; j < out.viewCount; ++j) {
            const double d = std::hypot(
                views[i].carrierI - views[j].carrierI,
                views[i].carrierQ - views[j].carrierQ) * invIreScale;
            iqDist[i][j] = d;
            iqDist[j][i] = d;
        }
    }

    const auto exclusionPenalty = [&](int i) -> double {
        const double localizer = std::exp(
            -0.5 * (views[i].membershipLocalX * views[i].membershipLocalX) /
            (1.35 * 1.35));
        return std::fabs(views[i].membershipDeltaIRE) *
               clamp01(views[i].membershipSupport) *
               localizer;
    };

    int best = 0;
    double bestCost = 1e300;
    for (int i = 0; i < out.viewCount; ++i) {
        double cost = 0.0;
        for (int j = 0; j < out.viewCount; ++j)
            cost += iqDist[i][j];
        cost += 0.35 * std::max(0.0, views[i].sampleFitErrorIRE);
        cost += 0.10 * exclusionPenalty(i);
        cost += 0.05 * std::max(0.0, views[i].score);
        if (cost < bestCost) {
            bestCost = cost;
            best = i;
        }
    }

    out.commonI = views[best].carrierI;
    out.commonQ = views[best].carrierQ;
    out.commonMagIRE = std::hypot(out.commonI, out.commonQ) * invIreScale;

    int sampleBest = 0;
    double sampleBestCost = 1e300;
    for (int i = 0; i < out.viewCount; ++i) {
        double cost = 0.0;
        for (int j = 0; j < out.viewCount; ++j)
            cost += std::fabs(views[i].carrierSample -
                              views[j].carrierSample) * invIreScale;
        cost += 0.25 * std::max(0.0, views[i].sampleFitErrorIRE);
        cost += 0.08 * exclusionPenalty(i);
        cost += 0.03 * std::max(0.0, views[i].score);
        if (cost < sampleBestCost) {
            sampleBestCost = cost;
            sampleBest = i;
        }
    }

    out.commonSample = views[sampleBest].carrierSample;

    double spread = 0.0;
    double sampleSpread = 0.0;
    double lattice = 0.0;
    for (int i = 0; i < out.viewCount; ++i) {
        // commonI/Q == views[best].carrierI/Q, so the IQ distance is exactly
        // iqDist[i][best] from the precomputed matrix — no extra hypot.
        spread = std::max(spread, iqDist[i][best]);
        sampleSpread = std::max(
            sampleSpread,
            std::fabs(views[i].carrierSample - out.commonSample) * invIreScale);
        lattice = std::max(lattice, views[i].latticeRiskIRE);
    }

    out.carrierSpreadIRE = spread;
    out.sampleFitErrorIRE = std::max(
        views[sampleBest].sampleFitErrorIRE,
        std::fabs(out.commonSample - views[sampleBest].fittedSample) * invIreScale);
    out.latticeRiskIRE = lattice;
    out.carrierCoherence = 1.0 - std::clamp(
        spread / std::max(3.0, 0.35 * out.commonMagIRE + 1.0),
        0.0,
        1.0);
    out.sampleCoherence = 1.0 - std::clamp(
        sampleSpread / std::max(3.0, 0.35 * std::fabs(out.commonSample) * invIreScale + 1.0),
        0.0,
        1.0);

    return out;
}

struct AttributionEvidence {
    // Luma / residual waveform evidence.
    double residualFineIRE = 0.0;
    double residualMidIRE = 0.0;
    double lumaIncursionRiskIRE = 0.0;
    double icebergAlienYFraction = 0.0;
    double lumaShapeContinuation = 0.0;

    // Carrier-free Y recovery evidence.
    double carrierFreeResidualIRE = 0.0;
    double carrierFreeYHiIRE = 0.0;
    double carrierFreeYContinuation = 0.0;
    double carrierFreeYCurvatureSupport = 0.0;
    double carrierFreeYBoundarySupport = 0.0;
    double carrierFreeYPeakSupport = 0.0;
    double carrierFreeChromaPressure = 0.0;
    double carrierFreeChromaConflict = 0.0;
    double carrierFreeCheckerRisk = 0.0;
    double carrierFreeSupport = 0.0;

    // Composite / carrier-domain evidence.
    double compositeResidualIRE = 0.0;
    double carrierResidualIRE = 0.0;
    double carrierChromaFitIRE = 0.0;
    double carrierChromaErrorIRE = 0.0;
    double carrierChromaCoherence = 0.0;
    double carrierPhaseCoherence = 0.0;

    double compositeChromaFitIRE = 0.0;
    double compositeChromaErrorIRE = 0.0;
    double compositeChromaCoherence = 0.0;
    double compositeCarrierCoherence = 0.0;
    double compositeLinePatternCoherence = 0.0;
    double compositeFieldCoherence = 0.0;
    double compositeBoundaryCoherence = 0.0;
    double compositeStableSideCoherence = 0.0;
    double compositePatternDepth = 0.0;

    double carrierPlausibility = 0.0;

    // Early coarse-Y / quarter-stage evidence.
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
    double quarterSlidingDiffIRE = 0.0;
    double quarterSlidingCoherence = 0.0;
    double quarterSlidingPhaseDisorder = 0.0;
    double quarterSlidingYPrior = 0.0;
    double carrierParallaxYSpreadIRE = 0.0;
    double carrierParallaxYCurvatureIRE = 0.0;
    double carrierParallaxSpreadIRE = 0.0;
    double carrierParallaxCoherence = 0.0;
    double carrierParallaxLatticeRiskIRE = 0.0;

    // IQ / chroma-envelope evidence.
    double iqChromaMagIRE = 0.0;
    double iqEnvelopeSmoothness = 0.0;
    double iqExcessHF = 0.0;
    double iqRemodErrorIRE = 0.0;
    double iqChromaCoherence = 0.0;
    double iqEnvelopeCoherence = 0.0;
    double iqParserViolation = 0.0;
    double iqQuadratureIncoherence = 0.0;
    double iqQuadratureTimingSupport = 0.0;
    double iqLumaTrespassClaim = 0.0;
    double chromaEnvelopeCoherence = 0.0;

    // Sideband / carrier-centered legality evidence.
    double sidebandLowerIRE = 0.0;
    double sidebandUpperIRE = 0.0;
    double sidebandCoherence = 0.0;
    double sidebandAsymmetry = 0.0;

    // Subtraction reconciliation and forward-model diagnostics.
    double reverseLumaIRE = 0.0;
    double forwardChromaIRE = 0.0;
    double modelAgreementClaim = 0.0;
    double forwardModelResidualIRE = 0.0;
    double forwardModelErrorIRE = 0.0;
    double forwardModelAgreement = 0.0;
    double yContributionSupport = 0.0;
    double iqContributionSupport = 0.0;
    double lumaImpulseCaution = 0.0;
    double waveformClaimConflict = 0.0;

    // Attribution claims.
    double attributionConflict = 0.0;
    double lumaClaim = 0.0;
    double uncertainClaim = 1.0;
    double carrierChromaClaim = 0.0;
    double envelopeChromaClaim = 0.0;
    double sidebandChromaClaim = 0.0;
    double compositeChromaClaim = 0.0;
    double iqEnvelopeClaim = 0.0;
};

inline double computeAttributionConflict(double lumaClaim,
                                         double chromaClaim,
                                         const AttributionRules &rules = kDefaultAttributionRules)
{
    switch (rules.conflictRule) {
    case AttributionConflictRule::GeometricMean:
    default:
        return std::sqrt(std::max(0.0, clamp01(lumaClaim) * clamp01(chromaClaim)));
    }
}

inline double computeUncertainAttribution(double lumaClaim,
                                          double chromaClaim,
                                          const AttributionRules &rules = kDefaultAttributionRules)
{
    switch (rules.uncertaintyRule) {
    case AttributionUncertaintyRule::OneMinusMaxClaim:
    default:
        return clamp01(1.0 - std::max(clamp01(lumaClaim), clamp01(chromaClaim)));
    }
}

inline double combinedCarrierChromaCoherence(const AttributionEvidence &e)
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

inline double combinedEnvelopeChromaCoherence(const AttributionEvidence &e)
{
    return std::max({
        e.iqChromaCoherence,
        e.iqEnvelopeCoherence,
        e.chromaEnvelopeCoherence
    });
}

inline double strongestChromaCoherence(const AttributionEvidence &e)
{
    return std::max({
        combinedCarrierChromaCoherence(e),
        combinedEnvelopeChromaCoherence(e)
    });
}

inline double combinedAttributionChromaClaim(const AttributionEvidence &e,
                                             const AttributionRules &rules = kDefaultAttributionRules)
{
    switch (rules.claimMergeRule) {
    case AttributionClaimMergeRule::Max:
    default:
        return std::max({
            e.carrierChromaClaim,
            e.envelopeChromaClaim,
            e.sidebandChromaClaim,
            e.compositeChromaClaim,
            e.iqEnvelopeClaim
        });
    }
}

inline void normalizeAttributionClaims(AttributionEvidence &e,
                                       const AttributionRules &rules = kDefaultAttributionRules)
{
    e.lumaClaim = clamp01(e.lumaClaim);

    e.carrierChromaClaim = clamp01(e.carrierChromaClaim);
    e.envelopeChromaClaim = clamp01(e.envelopeChromaClaim);
    e.sidebandChromaClaim = clamp01(e.sidebandChromaClaim);
    e.compositeChromaClaim = clamp01(e.compositeChromaClaim);
    e.iqEnvelopeClaim = clamp01(e.iqEnvelopeClaim);

    const double chromaClaim = clamp01(combinedAttributionChromaClaim(e, rules));
    e.uncertainClaim = computeUncertainAttribution(e.lumaClaim, chromaClaim, rules);
}

inline void applyAttributionConflictSuppression(AttributionEvidence &e,
                                                const AttributionRules &rules = kDefaultAttributionRules)
{
    e.lumaClaim = clamp01(e.lumaClaim);

    const double chromaClaim = clamp01(combinedAttributionChromaClaim(e, rules));
    const double conflict = computeAttributionConflict(e.lumaClaim, chromaClaim, rules);

    e.attributionConflict = std::max(e.attributionConflict, conflict);

    const double scale = std::max(
        0.0,
        1.0 - clamp01(rules.conflictSuppress) * conflict);

    e.lumaClaim *= scale;
    e.carrierChromaClaim *= scale;
    e.envelopeChromaClaim *= scale;
    e.sidebandChromaClaim *= scale;
    e.compositeChromaClaim *= scale;
    e.iqEnvelopeClaim *= scale;

    normalizeAttributionClaims(e, rules);
}

// Chroma-decoder-facing attribution schema.
struct CombAttributionFacts {
    double fieldAChromaIRE = 0.0;
    double fieldBChromaIRE = 0.0;
    double frameChromaIRE = 0.0;
    double locked1DChromaIRE = 0.0;

    double candidateSpreadIRE = 0.0;
    double frameFieldAgreementIRE = 0.0;
    double frameIQCoherence = 0.0;

    double residualFitErrorIRE = 0.0;
    double lumaIncursionRiskIRE = 0.0;
    double icebergAlienYFraction = 0.0;
    double lumaExcursionIRE = 0.0;

    double bandpassFineIRE = 0.0;
    double bandpassMidIRE = 0.0;
    double bandpassCoarseIRE = 0.0;

    double quarterCheckerboardRisk = 0.0;
    double carrierParallaxYSpreadIRE = 0.0;
    double carrierParallaxYCurvatureIRE = 0.0;
    double carrierParallaxSpreadIRE = 0.0;
    double carrierParallaxCoherence = 0.0;
    double carrierParallaxLatticeRiskIRE = 0.0;
    CarrierResidualConsensus carrierResidualConsensus;
    double carrierResidualIRE = 0.0;
    double carrierChromaFitIRE = 0.0;
    double carrierChromaErrorIRE = 0.0;
    double carrierChromaCoherence = 0.0;
    double movingResidualSample = 0.0;
    double movingResidualFitErrorIRE = 0.0;
    double movingResidualCoherence = 0.0;
    double movingResidualPull = 0.0;
    double sidebandSinResidualIRE = 0.0;
    double sidebandCosResidualIRE = 0.0;
    double sidebandAxisAsymmetry = 0.0;
    double sidebandCurvatureCoherence = 0.0;

    double lumaImpulseRisk = 0.0;
};

struct CombAttributionAssessment {
    double lumaRisk = 0.0;
    double lumaResidual = 0.0;
    double baseSupport = 0.0;
    double neighborSupport = 0.0;
    double lumaShapeContinuation = 0.0;

    double chromaStrength = 0.0;
    double sidebandChromaSupport = 0.0;
    double coherence = 0.0;
    double agreement = 0.0;
    double spreadPenalty = 0.0;
    double checkerboardRisk = 0.0;
    double carrierPrior = 0.0;
    double carrierPlausibility = 0.0;

    double lumaClaim = 0.0;
    double chromaClaim = 0.0;
    double uncertainClaim = 1.0;
    double attributionConflict = 0.0;
};

struct CombAttributionRecord {
    CombAttributionFacts facts;
    CombAttributionAssessment assessment;
};

using CombAttributionEvidence = CombAttributionRecord;

inline double strongestCombChromaIRE(const CombAttributionFacts &f)
{
    return std::max({
        f.fieldAChromaIRE,
        f.fieldBChromaIRE,
        f.frameChromaIRE,
        f.locked1DChromaIRE
    });
}

inline void normalizeCombAttributionAssessment(
    CombAttributionAssessment &a,
    const AttributionRules &rules = kDefaultAttributionRules)
{
    a.lumaClaim = clamp01(a.lumaClaim);
    a.chromaClaim = clamp01(a.chromaClaim);
    a.uncertainClaim = computeUncertainAttribution(a.lumaClaim, a.chromaClaim, rules);
}

inline void applyAttributionConflictSuppression(
    CombAttributionAssessment &a,
    const AttributionRules &rules = kDefaultAttributionRules)
{
    a.lumaClaim = clamp01(a.lumaClaim);
    a.chromaClaim = clamp01(a.chromaClaim);

    const double conflict = computeAttributionConflict(a.lumaClaim, a.chromaClaim, rules);
    a.attributionConflict = std::max(a.attributionConflict, conflict);

    const double scale = std::max(
        0.0,
        1.0 - clamp01(rules.conflictSuppress) * conflict);

    a.lumaClaim *= scale;
    a.chromaClaim *= scale;
    normalizeCombAttributionAssessment(a, rules);
}

} // namespace lddecode
