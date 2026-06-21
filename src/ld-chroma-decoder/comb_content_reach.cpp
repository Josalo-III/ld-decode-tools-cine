/******************************************************************************
 * comb_content_reach.cpp
 * ld-chroma-decoder shared image-content reach authority
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 ******************************************************************************/

#include "comb_content_reach.h"

#include <algorithm>
#include <cmath>

namespace CombContentReach {

namespace {

double clamp01(double v)
{
    return std::clamp(v, 0.0, 1.0);
}

double ramp(double v, double lo, double hi)
{
    if (hi <= lo)
        return (v >= hi) ? 1.0 : 0.0;
    return clamp01((v - lo) / (hi - lo));
}

double smoothGate(double valueIRE, double softIRE, double hardIRE)
{
    return 1.0 - ramp(valueIRE, softIRE, hardIRE);
}

double similarityFromDiff(double diffIRE, double clearIRE, double farIRE)
{
    if (!std::isfinite(diffIRE))
        return 0.0;
    return 1.0 - ramp(diffIRE, clearIRE, farIRE);
}

double mag(double i, double q)
{
    return std::hypot(i, q);
}

double dotIQ(double ai, double aq, double bi, double bq)
{
    return ai * bi + aq * bq;
}

double magIQ(double i, double q)
{
    return std::hypot(i, q);
}

double signedDotNormIQ(double ai, double aq, double bi, double bq)
{
    const double ma = magIQ(ai, aq);
    const double mb = magIQ(bi, bq);
    if (ma <= 1e-12 || mb <= 1e-12)
        return 0.0;
    return dotIQ(ai, aq, bi, bq) / (ma * mb);
}

double magnitudeRatioGate(double a, double b)
{
    const double hi = std::max(a, b);
    if (hi <= 1e-12)
        return 0.0;
    const double r = std::min(a, b) / hi;
    return ramp(r, 0.55, 0.88);
}

double oppositeIQFit(double centerI, double centerQ,
                     double sideI, double sideQ,
                     double minChromaIRE)
{
    const double mc = magIQ(centerI, centerQ);
    const double ms = magIQ(sideI, sideQ);

    if (mc < minChromaIRE || ms < minChromaIRE)
        return 0.0;

    const double signedDot = signedDotNormIQ(centerI, centerQ, sideI, sideQ);
    const double antiPhase = ramp(-signedDot, 0.55, 0.92);
    const double magFit = magnitudeRatioGate(mc, ms);
    const double chromaFit = ramp(std::min(mc, ms), minChromaIRE, minChromaIRE + 6.0);
    return clamp01(antiPhase * magFit * chromaFit);
}

void fillSide(Side &side,
              double iqAuthority,
              double scalarAuthority,
              double contourAuthority,
              double centerScalar,
              double sideScalar,
              double farScalar,
              bool hasFar,
              double lumaDiffIRE,
              double centerI,
              double centerQ,
              double sideI,
              double sideQ,
              bool hasIQ,
              double coherence,
              double centerCoarse,
              double sideCoarse,
              double farCoarse,
              bool hasCoarse,
              bool hasFarCoarse)
{
    side.scalarDiffIRE = std::fabs(centerScalar - sideScalar);
    side.scalarSimilarity = similarityFromDiff(side.scalarDiffIRE, 2.5, 14.0);

    side.iqCoherence = clamp01(coherence);
    if (hasIQ) {
        side.iqDistanceIRE = std::hypot(centerI - sideI, centerQ - sideQ);
        side.iqSimilarity = similarityFromDiff(side.iqDistanceIRE, 2.0, 10.0);
        side.iqSimilarity *= 0.65 + 0.35 * side.iqCoherence;
    } else {
        side.iqDistanceIRE = 0.0;
        side.iqSimilarity = 1.0;
    }

    if (hasCoarse) {
        side.contourDistanceIRE = std::fabs(centerCoarse - sideCoarse);
        side.contourSimilarity = similarityFromDiff(side.contourDistanceIRE, 2.0, 10.0);
    } else {
        side.contourDistanceIRE = 0.0;
        side.contourSimilarity = 1.0;
    }

    if (hasCoarse && hasFarCoarse && hasFar) {
        const double predicted = 2.0 * sideCoarse - centerCoarse;
        const double continuationDiff = std::fabs(farCoarse - predicted);
        side.contourContinuation = similarityFromDiff(continuationDiff, 3.0, 12.0);
    } else {
        side.contourContinuation = side.contourSimilarity;
    }

    side.materialSimilarity =
        iqAuthority * side.iqSimilarity +
        scalarAuthority * side.scalarSimilarity;

    const double contourBlend =
        0.55 * side.contourSimilarity +
        0.45 * side.contourContinuation;
    side.sameMaterial = clamp01(
        (1.0 - contourAuthority) * side.materialSimilarity +
        contourAuthority * (0.85 * side.materialSimilarity + 0.15 * contourBlend));

    side.iqSame = side.iqSimilarity >= 0.60;
    side.scalarSame = side.scalarSimilarity >= 0.60;
    side.contourSame = contourBlend >= 0.60;

    const double lumaEdge = ramp(lumaDiffIRE, 4.5, 16.0);
    const double contentMismatch = 1.0 - side.sameMaterial;
    const double edgePenalty = lumaEdge * contentMismatch;

    side.reachAuthority = clamp01(
        side.sameMaterial *
        (1.0 - 0.25 * edgePenalty) *
        (0.70 + 0.30 * side.contourContinuation));

    side.cancellationAuthority = clamp01(
        side.reachAuthority *
        (0.55 + 0.45 * side.scalarSimilarity));
}

} // namespace

double interfieldAlienCancelStrength(double centerI,
                                     double centerQ,
                                     double upI,
                                     double upQ,
                                     double downI,
                                     double downQ,
                                     bool hasUp,
                                     bool hasDown,
                                     double minChromaIRE,
                                     double columnSupport)
{
    if (!hasUp || !hasDown)
        return 0.0;

    const double minChroma = std::max(0.0, minChromaIRE);

    const double mc = magIQ(centerI, centerQ);
    const double mu = magIQ(upI, upQ);
    const double md = magIQ(downI, downQ);

    if (mc < minChroma || mu < minChroma || md < minChroma)
        return 0.0;

    const double neighborAgree =
        ramp(signedDotNormIQ(upI, upQ, downI, downQ), 0.45, 0.82);

    // The two neighbors are the local real-chroma estimate.  A center displaced
    // from their common carrier is likely alien chroma riding a luma contrast.
    const double neighborCommonI = 0.5 * (upI + downI);
    const double neighborCommonQ = 0.5 * (upQ + downQ);

    const double centerDelta =
        magIQ(centerI - neighborCommonI, centerQ - neighborCommonQ);

    const double centerDisplaced = ramp(centerDelta, 5.0, 18.0);
    const double chromaPresent =
        ramp(std::min({mc, mu, md, magIQ(neighborCommonI, neighborCommonQ)}),
             minChroma,
             minChroma + 3.0);

    // columnSupport means: a luma-contrast site with a smooth vertical column,
    // where interfield cancellation may treat alternating chroma as alien.
    const double verticalColumnSupport = 0.25 + 0.75 * clamp01(columnSupport);

    const double commonCarrierFit =
        neighborAgree * centerDisplaced * chromaPresent * verticalColumnSupport;

    // Tinted text can carry a legitimate common chroma vector while the 1D
    // sideband error alternates around it.  Use the agreeing +/-1 neighbors as
    // the shared carrier estimate, then ask whether the center is displaced
    // from that carrier.  Avoid pair-local midpoint tests here: they make any
    // center/side mismatch look like an opposite residual and promote bad
    // reaches into checker artifacts.
    const double residualIRE = 0.5 * centerDelta;
    const double residualPresent = ramp(residualIRE, 1.25, 6.25);
    const double residualChromaPresent =
        ramp(std::min({mc, mu, md}), minChroma, minChroma + 3.0);
    const double residualColumnSupport = ramp(columnSupport, 0.18, 0.75);
    const double residualTintCancel =
        neighborAgree *
        residualPresent *
        residualChromaPresent *
        residualColumnSupport;

    return clamp01(std::max(commonCarrierFit, residualTintCancel));
}

InterfieldIQReachFloor interfieldIQReachFloor(double centerI,
                                              double centerQ,
                                              double upI,
                                              double upQ,
                                              double downI,
                                              double downQ,
                                              bool hasUp,
                                              bool hasDown,
                                              double minChromaIRE,
                                              double columnSupport)
{
    InterfieldIQReachFloor out;

    const double minChroma = std::max(0.0, minChromaIRE);

    const double residualMinIRE = 1.25;

    const double upRawCancel = hasUp
        ? oppositeIQFit(centerI, centerQ, upI, upQ, minChroma)
        : 0.0;

    const double downRawCancel = hasDown
        ? oppositeIQFit(centerI, centerQ, downI, downQ, minChroma)
        : 0.0;

    double twoSidedResidualCancel = 0.0;
    if (hasUp && hasDown) {
        const double mc = magIQ(centerI, centerQ);
        const double mu = magIQ(upI, upQ);
        const double md = magIQ(downI, downQ);
        const double neighborAgree =
            ramp(signedDotNormIQ(upI, upQ, downI, downQ), 0.45, 0.82);
        const double neighborCommonI = 0.5 * (upI + downI);
        const double neighborCommonQ = 0.5 * (upQ + downQ);
        const double centerDelta =
            magIQ(centerI - neighborCommonI, centerQ - neighborCommonQ);
        const double residualPresent = ramp(0.5 * centerDelta,
                                            residualMinIRE,
                                            residualMinIRE + 5.0);
        const double chromaPresent =
            ramp(std::min({mc, mu, md}), minChroma, minChroma + 3.0);
        const double residualColumnSupport = ramp(columnSupport, 0.18, 0.75);

        twoSidedResidualCancel =
            neighborAgree *
            residualPresent *
            chromaPresent *
            residualColumnSupport;
    }

    const double upCancel =
        std::max(upRawCancel, 0.90 * twoSidedResidualCancel);

    const double downCancel =
        std::max(downRawCancel, 0.90 * twoSidedResidualCancel);

    double pairedCancel = 0.0;
    if (hasUp && hasDown) {
        pairedCancel = std::sqrt(std::clamp(upCancel * downCancel, 0.0, 1.0));
    } else {
        pairedCancel = 0.72 * std::max(upCancel, downCancel);
    }

    const double mixedLeakage = interfieldAlienCancelStrength(
        centerI, centerQ,
        upI, upQ,
        downI, downQ,
        hasUp, hasDown,
        minChroma,
        columnSupport);

    const double cleanup = clamp01(std::max(pairedCancel, mixedLeakage));

    out.up = clamp01(std::max(upCancel, 0.75 * mixedLeakage));
    out.down = clamp01(std::max(downCancel, 0.75 * mixedLeakage));
    out.cleanup = cleanup;

    return out;
}

// Fast overload: caller passes pre-computed IRE-domain magnitudes to avoid
// recomputing the ~12 sqrt/hypot calls the original version performs internally.
// Inlines interfieldAlienCancelStrength using the shared neighbor-common and
// centerDelta values, hoisting them out of both sub-computations.
InterfieldIQReachFloor interfieldIQReachFloor(double centerI,
                                              double centerQ,
                                              double upI,
                                              double upQ,
                                              double downI,
                                              double downQ,
                                              bool hasUp,
                                              bool hasDown,
                                              double minChromaIRE,
                                              double columnSupport,
                                              double centerMagIRE,
                                              double upMagIRE,
                                              double downMagIRE)
{
    InterfieldIQReachFloor out;

    const double minChroma = std::max(0.0, minChromaIRE);
    const double residualMinIRE = 1.25;

    // oppositeIQFit with pre-computed magnitudes — no sqrt.
    auto oppFitFast = [&](double cI, double cQ, double cMag,
                          double sI, double sQ, double sMag) -> double {
        if (cMag < minChroma || sMag < minChroma) return 0.0;
        const double sdot = (cMag > 1e-12 && sMag > 1e-12)
            ? std::clamp(dotIQ(cI, cQ, sI, sQ) / (cMag * sMag), -1.0, 1.0)
            : 0.0;
        const double antiPhase  = ramp(-sdot, 0.55, 0.92);
        const double magFit     = magnitudeRatioGate(cMag, sMag);
        const double chromaFit  = ramp(std::min(cMag, sMag), minChroma, minChroma + 6.0);
        return clamp01(antiPhase * magFit * chromaFit);
    };

    const double upRawCancel = hasUp
        ? oppFitFast(centerI, centerQ, centerMagIRE, upI, upQ, upMagIRE)
        : 0.0;

    const double downRawCancel = hasDown
        ? oppFitFast(centerI, centerQ, centerMagIRE, downI, downQ, downMagIRE)
        : 0.0;

    double twoSidedResidualCancel = 0.0;
    double mixedLeakage = 0.0;

    if (hasUp && hasDown) {
        const double mc = centerMagIRE;
        const double mu = upMagIRE;
        const double md = downMagIRE;

        // Neighbor correlation without sqrt: pre-computed mu and md.
        const double dotUD  = dotIQ(upI, upQ, downI, downQ);
        const double sdotUD = (mu > 1e-12 && md > 1e-12) ? dotUD / (mu * md) : 0.0;
        const double neighborAgree = ramp(sdotUD, 0.45, 0.82);

        const double neighborCommonI = 0.5 * (upI + downI);
        const double neighborCommonQ = 0.5 * (upQ + downQ);
        const double centerDelta =
            magIQ(centerI - neighborCommonI, centerQ - neighborCommonQ);

        // twoSidedResidualCancel (from base interfieldIQReachFloor)
        {
            const double residualPresent =
                ramp(0.5 * centerDelta, residualMinIRE, residualMinIRE + 5.0);
            const double chromaPresent =
                ramp(std::min({mc, mu, md}), minChroma, minChroma + 3.0);
            const double residualColumnSupport = ramp(columnSupport, 0.18, 0.75);
            twoSidedResidualCancel =
                neighborAgree * residualPresent * chromaPresent * residualColumnSupport;
        }

        // interfieldAlienCancelStrength inlined with shared vars
        if (mc >= minChroma && mu >= minChroma && md >= minChroma) {
            const double centerDisplaced = ramp(centerDelta, 5.0, 18.0);
            const double commonMag = magIQ(neighborCommonI, neighborCommonQ);
            const double chromaPresent =
                ramp(std::min({mc, mu, md, commonMag}), minChroma, minChroma + 3.0);
            const double verticalColumnSupport = 0.25 + 0.75 * clamp01(columnSupport);
            const double commonCarrierFit =
                neighborAgree * centerDisplaced * chromaPresent * verticalColumnSupport;

            const double residualIRE     = 0.5 * centerDelta;
            const double residualPresent = ramp(residualIRE, 1.25, 6.25);
            const double residualChromaPresent =
                ramp(std::min({mc, mu, md}), minChroma, minChroma + 3.0);
            const double residualColumnSupport = ramp(columnSupport, 0.18, 0.75);
            const double residualTintCancel =
                neighborAgree * residualPresent * residualChromaPresent * residualColumnSupport;
            mixedLeakage = clamp01(std::max(commonCarrierFit, residualTintCancel));
        }
    }

    const double upCancel   = std::max(upRawCancel,   0.90 * twoSidedResidualCancel);
    const double downCancel = std::max(downRawCancel, 0.90 * twoSidedResidualCancel);

    double pairedCancel = 0.0;
    if (hasUp && hasDown) {
        pairedCancel = std::sqrt(std::clamp(upCancel * downCancel, 0.0, 1.0));
    } else {
        pairedCancel = 0.72 * std::max(upCancel, downCancel);
    }

    const double cleanup = clamp01(std::max(pairedCancel, mixedLeakage));
    out.up    = clamp01(std::max(upCancel,   0.75 * mixedLeakage));
    out.down  = clamp01(std::max(downCancel, 0.75 * mixedLeakage));
    out.cleanup = cleanup;
    return out;
}

MovingCoarseContour evaluateMovingCoarseContour(double centerCoarse,
                                                double up2Coarse,
                                                double down2Coarse,
                                                double up4Coarse,
                                                double down4Coarse,
                                                bool hasUp2,
                                                bool hasDown2,
                                                bool hasUp4,
                                                bool hasDown4,
                                                double softIRE,
                                                double hardIRE)
{
    MovingCoarseContour out;

    if (!hasUp2 || !hasDown2 || !hasUp4 || !hasDown4)
        return out;

    out.valid = true;
    out.curvMidIRE = std::fabs(up2Coarse - 2.0 * centerCoarse + down2Coarse);
    out.midOk = smoothGate(out.curvMidIRE, softIRE, hardIRE);

    const double up4Pred = 2.0 * up2Coarse - centerCoarse;
    const double down4Pred = 2.0 * down2Coarse - centerCoarse;

    out.upResIRE = std::fabs(up4Coarse - up4Pred);
    out.downResIRE = std::fabs(down4Coarse - down4Pred);
    out.upSideOk = smoothGate(out.upResIRE, softIRE, hardIRE);
    out.downSideOk = smoothGate(out.downResIRE, softIRE, hardIRE);

    out.upTrust = out.midOk * out.upSideOk;
    out.downTrust = out.midOk * out.downSideOk;

    const double contourTrust =
        out.midOk * 0.5 * (out.upSideOk + out.downSideOk);
    const double contourCurvNorm =
        (hardIRE > 1e-9) ? clamp01(out.curvMidIRE / hardIRE) : 0.0;
    out.straightness = clamp01(
        0.70 * contourTrust + 0.30 * (1.0 - contourCurvNorm));

    return out;
}

Reply evaluate(const Query &query)
{
    Reply reply;

    reply.iqMagnitudeIRE = query.hasIQ
        ? mag(query.centerI, query.centerQ)
        : std::fabs(query.chromaIRE);
    reply.iqAuthority = query.hasIQ
        ? ramp(reply.iqMagnitudeIRE, 2.0, 8.0)
        : 0.0;
    reply.scalarAuthority = 1.0 - reply.iqAuthority;
    reply.contourAuthority = query.hasMovingCoarse ? 1.0 : 0.0;

    fillSide(reply.up,
             reply.iqAuthority,
             reply.scalarAuthority,
             reply.contourAuthority,
             query.centerScalar,
             query.up2Scalar,
             query.up4Scalar,
             query.hasUp4,
             query.up2LumaDiffIRE,
             query.centerI,
             query.centerQ,
             query.up2I,
             query.up2Q,
             query.hasIQ && query.hasUp2,
             query.upCoherence,
             query.centerCoarse,
             query.up2Coarse,
             query.up4Coarse,
             query.hasMovingCoarse && query.hasUp2,
             query.hasMovingCoarse && query.hasUp4);

    fillSide(reply.down,
             reply.iqAuthority,
             reply.scalarAuthority,
             reply.contourAuthority,
             query.centerScalar,
             query.down2Scalar,
             query.down4Scalar,
             query.hasDown4,
             query.down2LumaDiffIRE,
             query.centerI,
             query.centerQ,
             query.down2I,
             query.down2Q,
             query.hasIQ && query.hasDown2,
             query.downCoherence,
             query.centerCoarse,
             query.down2Coarse,
             query.down4Coarse,
             query.hasMovingCoarse && query.hasDown2,
             query.hasMovingCoarse && query.hasDown4);

    reply.upDownMaterialDifference =
        std::fabs(reply.up.sameMaterial - reply.down.sameMaterial);
    reply.upDownContourDifference =
        std::fabs(reply.up.contourContinuation - reply.down.contourContinuation);

    reply.transitionStrength = clamp01(
        1.0 - 0.5 * (reply.up.sameMaterial + reply.down.sameMaterial));
    reply.oneSidedness = clamp01(reply.upDownMaterialDifference);

    const double bestSide = std::max(reply.up.sameMaterial, reply.down.sameMaterial);
    const double balanced = 1.0 - reply.oneSidedness;
    const double bothDifferent = clamp01(
        1.0 - std::max(reply.up.sameMaterial, reply.down.sameMaterial));
    reply.centerIsIntermediate = clamp01(
        reply.transitionStrength *
        bothDifferent *
        balanced *
        (0.60 + 0.40 * (0.5 * (reply.up.contourContinuation +
                                reply.down.contourContinuation))));

    reply.bevelOrOutlineStrength = clamp01(
        reply.centerIsIntermediate *
        0.5 * (reply.up.contourContinuation + reply.down.contourContinuation));

    const double lowEvidence =
        (1.0 - reply.contourAuthority) *
        (1.0 - std::max(reply.iqAuthority, reply.scalarAuthority * 0.85));
    reply.ambiguity = clamp01(
        balanced *
        (1.0 - bestSide) *
        (0.35 + 0.65 * reply.transitionStrength));

    reply.symmetricAverageAuthority = clamp01(
        std::min(reply.up.reachAuthority, reply.down.reachAuthority) *
        (1.0 - 0.45 * reply.oneSidedness) *
        (1.0 - 0.45 * reply.centerIsIntermediate));

    reply.oneSidedAuthority = clamp01(
        std::max(reply.up.reachAuthority, reply.down.reachAuthority) *
        reply.oneSidedness *
        (0.50 + 0.50 * reply.transitionStrength));

    reply.centerFallbackAuthority = clamp01(std::max({
        0.85 * reply.centerIsIntermediate,
        0.50 * reply.bevelOrOutlineStrength,
        0.45 * reply.ambiguity,
        0.25 * lowEvidence}));

    reply.allowSymmetricReach =
        reply.symmetricAverageAuthority >= 0.16 &&
        reply.centerFallbackAuthority < 0.88;

    reply.preferUp =
        (reply.up.reachAuthority > reply.down.reachAuthority + 0.12) &&
        (reply.up.sameMaterial > 0.42);
    reply.preferDown =
        (reply.down.reachAuthority > reply.up.reachAuthority + 0.12) &&
        (reply.down.sameMaterial > 0.42);
    reply.preferCenterFallback =
        (reply.centerFallbackAuthority >=
         std::max(reply.symmetricAverageAuthority, reply.oneSidedAuthority) + 0.18) &&
        (reply.centerIsIntermediate > 0.45 ||
         reply.bevelOrOutlineStrength > 0.55);

    reply.up.selectedSide = reply.preferUp;
    reply.down.selectedSide = reply.preferDown;
    reply.up.suppressedSide = reply.preferDown;
    reply.down.suppressedSide = reply.preferUp;

    if (lowEvidence >= 0.75) {
        reply.verdict = Verdict::LowEvidence;
    } else if (reply.centerIsIntermediate >= 0.72) {
        reply.verdict = Verdict::IntermediateZone;
    } else if (reply.bevelOrOutlineStrength >= 0.60) {
        reply.verdict = Verdict::BevelOrOutline;
    } else if (reply.centerFallbackAuthority >= 0.72 &&
               reply.transitionStrength >= 0.55) {
        reply.verdict = Verdict::ClearTransition;
    } else if (reply.preferUp || reply.preferDown) {
        reply.verdict = Verdict::OneSidedContinuation;
    } else if (reply.ambiguity >= 0.60) {
        reply.verdict = Verdict::Ambiguous;
    } else {
        reply.verdict = Verdict::SmoothContinuation;
    }

    return reply;
}

} // namespace CombContentReach
