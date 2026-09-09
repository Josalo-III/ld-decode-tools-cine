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
#include <cstdint>

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

// Central, application-neutral carrier analysis.
//
// These records contain observations and compatibility results only.  They do
// not say how much carrier to render, how much colour to suppress, or which Y
// candidate should win.  In particular, the fit samples and residual bounds
// below must never be used as replacement waveforms merely because they are
// available here.
struct CarrierFitDiagnostics {
    float sourceSample = 0.0f;          // ordinary carrier-band source
    float shortSample = 0.0f;           // local coherent fit
    float wideSample = 0.0f;            // comparison-only wide coherent fit
    float shortMagnitudeIRE = 0.0f;
    float wideMagnitudeIRE = 0.0f;
    float sourceMinusShortIRE = 0.0f;
    float shortMinusWideIRE = 0.0f;
    float sourceMinusWideIRE = 0.0f;
    bool valid = false;
};

// Compatibility of a fit sample with the discrete legal coarse-residual
// options.  The interval is only the extent of the surviving option set; its
// centre is not an estimate and the options must not be averaged together.
struct CarrierResidualDiagnostics {
    // Preserve the actual discrete options.  Consumers must not reconstruct
    // them from survivorLo/Hi or treat the interval between them as observed.
    float optionSamples[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    // Kept as doubles because bounded application policy clamps the original
    // double-precision source to these exact observed survivors. Publishing
    // them preserves the observed precision.
    double survivorLo = 0.0;
    double survivorHi = 0.0;
    float movingResidualSample = 0.0f;
    float residualSpreadIRE = 0.0f;
    float maxAbsMembershipIRE = 0.0f;
    float nearestFitDistanceIRE = 0.0f;
    float movingDistanceIRE = 0.0f;
    float toleranceIRE = 0.0f;
    std::uint8_t optionCount = 0;
    std::uint8_t survivorMask = 0;
    bool movingCompatible = false;
    bool valid = false;

    int survivorCount() const {
        int n = 0;
        std::uint8_t bits = survivorMask;
        while (bits != 0) {
            n += bits & 1u;
            bits >>= 1u;
        }
        return n;
    }
};

// Compact publication of the richer four-view carrier analysis.  The complete
// per-view evidence remains in FourViewPixelEvidence where a client needs to
// inspect individual legal apertures.
struct CarrierParallaxDiagnostics {
    float commonSample = 0.0f;
    float commonMagnitudeIRE = 0.0f;
    float ySpreadIRE = 0.0f;
    float yCurvatureIRE = 0.0f;
    float carrierSpreadIRE = 0.0f;
    float carrierCoherence = 0.0f;
    float sampleFitErrorIRE = 0.0f;
    float sampleCoherence = 0.0f;
    float latticeRiskIRE = 0.0f;
    float residualLo = 0.0f;
    float residualHi = 0.0f;
    float residualTrust = 0.0f;
    bool residualValid = false;
    bool valid = false;
};

// Schedule-conformance verdict, registered at analysis time in grammar
// coordinates.  Legal carrier must invert across Opposite-relation partners
// (same-field +/-2 lines; the same line on the neighbouring frame).  Energy
// that instead matches where the schedule demands inversion is structurally
// not carrier — luma by law (near-carrier periodic luma, e.g. a fine static
// grid).  This is registration-with-rejection: the verdict is a fact of the
// table, not a downstream confidence score, and consumers convert it via
// named policy (fit input exclusion, repair abstention, election admission).
enum class CarrierScheduleConformance : std::uint8_t {
    Unresolved = 0,      // no discriminative axis had enough energy/coherence
    LegalCarrier = 1,    // at least one Opposite-relation axis inverted
    ScheduleIllegal = 2, // matched where inversion was demanded; no axis legal
};

struct CarrierAnalysisRecord {
    CarrierFitDiagnostics fit;
    CarrierResidualDiagnostics residual;
    CarrierParallaxDiagnostics parallax;
    float carrierImpurity = 0.0f;       // detector output, not transfer policy
    CarrierScheduleConformance scheduleConformance =
        CarrierScheduleConformance::Unresolved;
    // Graded conformance measurement (scanner layer). carrierConformance in
    // [-1,+1]: relation-signed correlation of carrier-band energy against
    // grammar-certified Opposite partners.  -1 = inverts like ideal carrier;
    // +1 = matches where the schedule demands inversion (luma by law).
    // conformanceUsableAxisFraction records the fraction of usable axes.
    // conformanceSupportFraction records the fraction of possible axes
    // supporting the selected sign. Availability and support remain separate:
    // one legal-looking axis is 1/3 support even when all three axes are usable.
    float carrierConformance = 0.0f;
    float conformanceUsableAxisFraction = 0.0f;
    float conformanceSupportFraction = 0.0f;
    // Contradiction is a distinct observation from absence.  This is the
    // fraction of the three possible axes that decisively voted against the
    // sign selected for carrierConformance (e.g. a matching axis when the
    // selected sign is legal-inverting).  An axis that is below the energy
    // floor or non-decisive abstains and appears in neither fraction.
    // Unobserved axes and observed contradictions are separate outcomes. Only
    // an observed contradiction can revoke a legal support vote.
    float conformanceContradictionFraction = 0.0f;
};

// Decisiveness ramp shared by schedule-compatibility licenses: input is a
// relation-signed correlation where -1 means "behaves like legal carrier".
// Zero until the legal vote threshold, saturating at a decisive inversion —
// the same -0.5 / -0.9 shape as carrierLegalProof, without any axis-count
// or corroboration input.  One observed on-schedule alternation licenses at
// any spatial scale; what the correlation is measured ON (raw bandpass, a
// fitted operand, ...) is the consumer's stated choice.
inline double scheduleAlternationLicense(double signedCorr)
{
    constexpr double kLegalVote = 0.5;
    constexpr double kLegalFull = 0.9;
    double t = (-signedCorr - kLegalVote) / (kLegalFull - kLegalVote);
    t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
    return t * t * (3.0 - 2.0 * t);
}

// Decision layer, luma side: the table-owned mapping from conformance
// measurement to a luma-attribution proof in [0,1]. This is
// conservative because a suppression consumer acting on ambiguous evidence
// desaturates genuine chroma --
// at a hue boundary the correlation windows straddle two hues, no axis can
// produce a decisive legal vote, and the stored conformance is a weakly
// positive maxCorr that means "unresolved", not "luma". The registration
// layer's own tie-break is "real chroma is never claimed as luma", so the
// proof stays zero through the ambiguous middle and engages only
// past +kIllegalVote -- the same threshold at which an axis casts a
// ScheduleIllegal vote -- ramping smoothly (no verdict flip at pixel pitch)
// and scaled by confidence so a thin axis set cannot assert a full proof.
inline double carrierIllegalProof(double conformance, double confidence)
{
    constexpr double kIllegalVote = 0.5;  // vote threshold: proof begins here
    constexpr double kIllegalFull = 0.9;  // decisive match: proof saturates

    double t = (conformance - kIllegalVote) / (kIllegalFull - kIllegalVote);
    t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
    const double p = t * t * (3.0 - 2.0 * t); // smoothstep

    // confidence arrives as supportingAxes/3. Two concurring Opposite axes
    // already constitute a full proof (the third, the frame axis, rightly
    // abstains on motion and must not discount an intra-field conviction);
    // a single supporting axis is half a proof for consumers that permit it.
    double c = 1.5 * (confidence < 0.0 ? 0.0 : confidence);
    c = c > 1.0 ? 1.0 : c;
    return c * p;
}

// Mirror image: certified-legal-carrier proof. Nonzero only when the
// bandpass decisively inverts across Opposite partners (conformance past
// the legal vote threshold), i.e. this pixel is proven genuine chroma.
// Consumers use it as spatial context: ambiguity bordering a certified
// region is a hue boundary (protect), ambiguity in a legality desert is
// actionable.
inline double carrierLegalProof(double conformance, double confidence)
{
    return carrierIllegalProof(-conformance, confidence);
}

struct CarrierResidualOption {
    double sample = 0.0;
    double membershipDeltaIRE = 0.0;
};

// Evaluate a local fit against the legal residual options without collapsing
// those options into a coarse estimate.  The moving-centred residual is an
// independent scanner: it can support or conflict with the surviving subset,
// but it is never inserted into that subset.
inline CarrierResidualDiagnostics analyzeCarrierResidualOptions(
    const CarrierResidualOption *options,
    int optionCount,
    double shortFitSample,
    double toleranceSamples,
    double toleranceIRE,
    double movingResidualSample,
    double invIreScale)
{
    CarrierResidualDiagnostics out;
    optionCount = std::clamp(optionCount, 0, 4);
    out.optionCount = static_cast<std::uint8_t>(optionCount);
    out.movingResidualSample = static_cast<float>(movingResidualSample);
    out.toleranceIRE = static_cast<float>(std::max(0.0, toleranceIRE));
    if (!options || optionCount <= 0)
        return out;

    double optionLo = options[0].sample;
    double optionHi = options[0].sample;
    double survivorLo = 1e300;
    double survivorHi = -1e300;
    double nearestDistance = 1e300;
    double maxAbsMembershipIRE = 0.0;

    for (int i = 0; i < optionCount; ++i) {
        const double sample = options[i].sample;
        out.optionSamples[i] = static_cast<float>(sample);
        optionLo = std::min(optionLo, sample);
        optionHi = std::max(optionHi, sample);
        maxAbsMembershipIRE = std::max(
            maxAbsMembershipIRE,
            std::fabs(options[i].membershipDeltaIRE));

        const double distance = std::fabs(sample - shortFitSample);
        nearestDistance = std::min(nearestDistance, distance);
        if (distance <= toleranceSamples) {
            out.survivorMask |= static_cast<std::uint8_t>(1u << i);
            survivorLo = std::min(survivorLo, sample);
            survivorHi = std::max(survivorHi, sample);
        }
    }

    out.residualSpreadIRE = static_cast<float>(
        (optionHi - optionLo) * invIreScale);
    out.maxAbsMembershipIRE = static_cast<float>(maxAbsMembershipIRE);
    out.nearestFitDistanceIRE = static_cast<float>(
        nearestDistance * invIreScale);
    out.valid = true;

    if (out.survivorMask == 0)
        return out;

    out.survivorLo = survivorLo;
    out.survivorHi = survivorHi;

    double movingDistance = 0.0;
    if (movingResidualSample < survivorLo)
        movingDistance = survivorLo - movingResidualSample;
    else if (movingResidualSample > survivorHi)
        movingDistance = movingResidualSample - survivorHi;
    out.movingDistanceIRE = static_cast<float>(movingDistance * invIreScale);
    out.movingCompatible =
        movingResidualSample >= survivorLo - toleranceSamples &&
        movingResidualSample <= survivorHi + toleranceSamples;
    return out;
}

// Encoder band-legality revocation (the bandwidth law). Chroma is bandlimited
// to the nominal 1.3 MHz legal band before modulation, so demod-envelope
// energy outside that band cannot be chroma. Revocation is one-sided: it
// removes a chroma claim but never asserts luma. A residue may move to luma
// only with affirmative luma evidence; otherwise it remains uncertain.
// Witnesses are precision-first: they may abstain, but must not claim genuine
// chroma as luma.
struct BandRevokedResidueEvidence {
    double residueIRE = 0.0;       // out-of-legal-band envelope magnitude
    double witnessMatchIRE = 0.0;  // witness luma structure at this site
    double witnessSupport = 0.0;   // witness quality [0,1], precision-first
};

// Decision layer: the luma claim a band-revoked residue may assert, in
// [0,1] of the residue.  min() caps the claim at what the witness actually
// saw (a witness cannot license more energy than it observed), and the
// support factor scales authority without ever inverting abstention into
// assertion.  Zero witness -> zero claim -> the residue abstains.
inline double bandResidueLumaClaim(const BandRevokedResidueEvidence &e)
{
    if (e.residueIRE <= 1e-9)
        return 0.0;
    const double match = std::min(e.witnessMatchIRE, e.residueIRE);
    return clamp01(match / e.residueIRE) * clamp01(e.witnessSupport);
}

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
    // this means at most 6 square roots instead of viewCount² (= 16) when
    // looped naively below, and the same matrix is reused for the spread
    // computation later — together cutting parallax distance roots from ~20
    // per pixel to ~6 without changing any output.
    double iqDist[4][4] = {{0.0, 0.0, 0.0, 0.0},
                            {0.0, 0.0, 0.0, 0.0},
                            {0.0, 0.0, 0.0, 0.0},
                            {0.0, 0.0, 0.0, 0.0}};
    for (int i = 0; i < out.viewCount; ++i) {
        for (int j = i + 1; j < out.viewCount; ++j) {
            const double dI = views[i].carrierI - views[j].carrierI;
            const double dQ = views[i].carrierQ - views[j].carrierQ;
            // Carrier-fit samples are bounded to video scale, so hypot's
            // overflow/underflow rescaling is unnecessary in this hot loop.
            const double d = std::sqrt(dI * dI + dQ * dQ) * invIreScale;
            iqDist[i][j] = d;
            iqDist[j][i] = d;
        }
    }

    // A covering four-sample view can only place its membership observation at
    // local offsets -1, 0, 1, or 2 from the current pixel.  Cache those four
    // Gaussian values process-wide, and cache each view's complete penalty for
    // reuse by the IQ and scalar medoid searches below.  The generic fallback
    // preserves this helper's contract for callers with other geometries.
    static const double membershipLocalizer[4] = {
        std::exp(-0.5 * 1.0 / (1.35 * 1.35)),
        1.0,
        std::exp(-0.5 * 1.0 / (1.35 * 1.35)),
        std::exp(-0.5 * 4.0 / (1.35 * 1.35))
    };
    double exclusionPenalties[4] = {0.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < out.viewCount; ++i) {
        const double localX = views[i].membershipLocalX;
        const int localIndex = static_cast<int>(localX) + 1;
        const double localizer =
            localIndex >= 0 && localIndex < 4 &&
            localX == static_cast<double>(localIndex - 1)
                ? membershipLocalizer[localIndex]
                : std::exp(-0.5 * (localX * localX) / (1.35 * 1.35));
        exclusionPenalties[i] =
            std::fabs(views[i].membershipDeltaIRE) *
            clamp01(views[i].membershipSupport) * localizer;
    }

    int best = 0;
    double bestCost = 1e300;
    for (int i = 0; i < out.viewCount; ++i) {
        double cost = 0.0;
        for (int j = 0; j < out.viewCount; ++j)
            cost += iqDist[i][j];
        cost += 0.35 * std::max(0.0, views[i].sampleFitErrorIRE);
        cost += 0.10 * exclusionPenalties[i];
        cost += 0.05 * std::max(0.0, views[i].score);
        if (cost < bestCost) {
            bestCost = cost;
            best = i;
        }
    }

    out.commonI = views[best].carrierI;
    out.commonQ = views[best].carrierQ;
    out.commonMagIRE =
        std::sqrt(out.commonI * out.commonI + out.commonQ * out.commonQ) *
        invIreScale;

    int sampleBest = 0;
    double sampleBestCost = 1e300;
    for (int i = 0; i < out.viewCount; ++i) {
        double cost = 0.0;
        for (int j = 0; j < out.viewCount; ++j)
            cost += std::fabs(views[i].carrierSample -
                              views[j].carrierSample) * invIreScale;
        cost += 0.25 * std::max(0.0, views[i].sampleFitErrorIRE);
        cost += 0.08 * exclusionPenalties[i];
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
        // iqDist[i][best] from the precomputed matrix — no extra square root.
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
