/******************************************************************************
 * comb_content_reach.h
 * ld-chroma-decoder shared image-content reach authority
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 ******************************************************************************/

#pragma once

namespace CombContentReach {

enum class Verdict {
    SmoothContinuation,
    OneSidedContinuation,
    ClearTransition,
    IntermediateZone,
    BevelOrOutline,
    ShadowOrHalo,
    Ambiguous,
    LowEvidence
};

struct Side {
    double iqDistanceIRE = 0.0;
    double iqCoherence = 0.0;
    double iqSimilarity = 1.0;

    double scalarDiffIRE = 0.0;
    double scalarSimilarity = 1.0;

    double contourDistanceIRE = 0.0;
    double contourSimilarity = 1.0;
    double contourContinuation = 1.0;

    double materialSimilarity = 1.0;
    double sameMaterial = 1.0;

    double reachAuthority = 1.0;
    double cancellationAuthority = 1.0;

    bool iqSame = false;
    bool scalarSame = false;
    bool contourSame = false;
    bool selectedSide = false;
    bool suppressedSide = false;
};

struct Reply {
    Verdict verdict = Verdict::LowEvidence;

    Side up;
    Side down;

    double iqMagnitudeIRE = 0.0;
    double iqAuthority = 0.0;
    double scalarAuthority = 0.0;
    double contourAuthority = 0.0;

    double upDownMaterialDifference = 0.0;
    double upDownContourDifference = 0.0;

    double transitionStrength = 0.0;
    double oneSidedness = 0.0;
    double centerIsIntermediate = 0.0;
    double bevelOrOutlineStrength = 0.0;
    double ambiguity = 0.0;

    double symmetricAverageAuthority = 1.0;
    double oneSidedAuthority = 0.0;
    double centerFallbackAuthority = 0.0;

    bool allowSymmetricReach = true;
    bool preferUp = false;
    bool preferDown = false;
    bool preferCenterFallback = false;
};

struct Query {
    int frameLine = 0;
    int x = 0;

    bool lockedPath = false;
    bool hasIQ = false;
    bool hasMovingCoarse = false;

    double chromaIRE = 0.0;

    double centerScalar = 0.0;
    double up2Scalar = 0.0;
    double down2Scalar = 0.0;
    double up4Scalar = 0.0;
    double down4Scalar = 0.0;

    bool hasUp2 = false;
    bool hasDown2 = false;
    bool hasUp4 = false;
    bool hasDown4 = false;

    double up2LumaDiffIRE = 0.0;
    double down2LumaDiffIRE = 0.0;

    double centerI = 0.0;
    double centerQ = 0.0;
    double up2I = 0.0;
    double up2Q = 0.0;
    double down2I = 0.0;
    double down2Q = 0.0;

    double centerCoarse = 0.0;
    double up2Coarse = 0.0;
    double down2Coarse = 0.0;
    double up4Coarse = 0.0;
    double down4Coarse = 0.0;

    double upCoherence = 1.0;
    double downCoherence = 1.0;
};

struct InterfieldIQReachFloor {
    double up = 0.0;
    double down = 0.0;
    double cleanup = 0.0;
};

struct MovingCoarseContour {
    bool valid = false;

    double curvMidIRE = 0.0;
    double upResIRE = 0.0;
    double downResIRE = 0.0;

    double midOk = 1.0;
    double upSideOk = 1.0;
    double downSideOk = 1.0;

    double upTrust = 0.0;
    double downTrust = 0.0;
    double straightness = 0.0;
};

InterfieldIQReachFloor interfieldIQReachFloor(double centerI,
                                              double centerQ,
                                              double upI,
                                              double upQ,
                                              double downI,
                                              double downQ,
                                              bool hasUp,
                                              bool hasDown,
                                              double minChromaIRE,
                                              double lumaEdgeFit);

// Confidence in [0,1] that the center IQ is alien chroma phase-displaced from
// the common carrier of its two agreeing neighbors.  This is the vector-cancel
// companion to interfieldIQReachFloor's scalar floor: a consumer that has the
// neighbor vectors can pull the center toward 0.5*(up+down) by this strength,
// removing only the displacement rather than upweighting a sign-aligned average.
// Inputs are IRE-scaled.
double interfieldAlienCancelStrength(double centerI,
                                     double centerQ,
                                     double upI,
                                     double upQ,
                                     double downI,
                                     double downQ,
                                     bool hasUp,
                                     bool hasDown,
                                     double minChromaIRE,
                                     double columnSupport);

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
                                                double hardIRE);

Reply evaluate(const Query &query);

} // namespace CombContentReach
