/******************************************************************************
 * combreach.cpp
 * ld-chroma-decoder comb reach system
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * See combreach.h for the two-arm split (lddecode reach legality /
 * CombContentReach image-content authority).
 ******************************************************************************/

#include "combreach.h"

#include <algorithm>
#include <cmath>

// ===========================================================================
// Carrier-grammar reach legality translator
// ===========================================================================
namespace lddecode {

namespace {

CombReachReply blockedReply(const CombReachRequest &request, const char *tag)
{
    CombReachReply reply;
    reply.verdict = CombReachVerdict::Blocked;
    reply.centerFrame = request.source.signFrame;
    reply.targetFrame = request.source.signFrame;
    reply.tag = tag;
    return reply;
}

CombReachReply unknownReply(const CombReachRequest &request, const char *tag)
{
    CombReachReply reply;
    reply.verdict = CombReachVerdict::Unknown;
    reply.centerFrame = request.source.signFrame;
    reply.targetFrame = request.source.signFrame;
    reply.tag = tag;
    return reply;
}

double grammarAuthority(const CarrierGrammarState *center,
                        const CarrierGrammarState *target)
{
    if (!center || !target)
        return 0.0;

    auto lineAuthority = [](const CarrierGrammarState *grammar) -> double {
        double base = grammar->grammarLocked
            ? std::clamp(grammar->phaseConfidence, 0.0, 1.0)
            : 0.65;
        base *= 1.0 - 0.5 * std::clamp(grammar->phaseScheduleConflict, 0.0, 1.0);
        return std::clamp(base, 0.0, 1.0);
    };

    return std::min(lineAuthority(center), lineAuthority(target));
}

bool scalarUse(CombReachUse use)
{
    return use == CombReachUse::FieldScalarAverage ||
           use == CombReachUse::FieldScalarCancel ||
           use == CombReachUse::FrameScalarAverage ||
           use == CombReachUse::FrameScalarCancel ||
           use == CombReachUse::ScalarSignCompare ||
           use == CombReachUse::ScalarMagnitudeCompare;
}

bool iqUse(CombReachUse use)
{
    return use == CombReachUse::IQCompare ||
           use == CombReachUse::IQAverage ||
           use == CombReachUse::IQCancel;
}

CombReachReply queryGrammarPair(const CombReachRequest &request,
                                const CarrierGrammarState *center,
                                const CarrierGrammarState *target)
{
    // Signal-class triage comes first: a phase-erased diagnostic source is
    // rejected from video-capable operations before any grammar convenience
    // can leak it into the output path.  It needs no grammar to be scored.
    if (request.source.signalClass == CombReachSignalClass::PhaseErasedDiagnostic) {
        CombReachReply reply;
        reply.verdict = CombReachVerdict::DiagnosticOnly;
        reply.valid = true;
        reply.allowDiagnosticCompare = true;
        reply.allowScalarMagnitudeCompare = true;  // magnitude is phase-free
        reply.mayBecomeVideo = false;
        reply.centerFrame = request.source.signFrame;
        reply.targetFrame = request.source.signFrame;
        reply.tag = "phase-erased-diagnostic";
        return reply;
    }

    if (!center || !target)
        return unknownReply(request, "missing-grammar");

    if (request.source.carrierFree ||
        request.source.signalClass == CombReachSignalClass::CarrierFree)
        return blockedReply(request, "carrier-free-y");

    if (request.source.kind == CombReachSourceKind::Detector)
        return blockedReply(request, "detector-not-waveform");

    if (request.source.signalClass != CombReachSignalClass::PhasePreservedCarrier)
        return blockedReply(request, "signal-class-unknown");

    CombReachReply reply;
    reply.valid = true;
    reply.centerFrame = request.source.signFrame;
    reply.targetFrame = request.source.signFrame;
    reply.authority = grammarAuthority(center, target);
    reply.carrierRelation = carrierGrammarSignedPhaseRelation(
        center,
        request.centerH,
        target,
        request.targetH);

    if (iqUse(request.use)) {
        if (!request.source.iqCarrier) {
            reply.verdict = CombReachVerdict::PriorOnly;
            reply.tag = "scalar-not-iq";
            return reply;
        }

        reply.verdict = CombReachVerdict::Green;
        reply.fastPath = true;
        reply.allowIQCompare = true;
        reply.allowIQAverage = true;
        reply.allowIQCancel =
            request.use != CombReachUse::IQCancel ||
            reply.carrierRelation == CarrierPhaseRelation::Same ||
            reply.carrierRelation == CarrierPhaseRelation::Opposite;
        reply.mayBecomeVideo = true;
        reply.tag = "iq-carrier";
        return reply;
    }

    if (!scalarUse(request.use))
        return unknownReply(request, "unknown-use");

    if (!request.source.scalarCarrier)
        return blockedReply(request, "not-scalar-carrier");

    if (reply.carrierRelation == CarrierPhaseRelation::Same ||
        reply.carrierRelation == CarrierPhaseRelation::Opposite)
    {
        reply.verdict = CombReachVerdict::Green;
        reply.fastPath = true;
        reply.allowScalarAverage = true;
        reply.allowScalarCancel = true;
        reply.allowScalarSignCompare = true;
        reply.allowScalarMagnitudeCompare = true;
        reply.mayBecomeVideo = true;
        reply.tag = "carrier-relation";
        return reply;
    }

    reply.verdict = CombReachVerdict::Blocked;
    reply.tag = "phase-relation-other";
    return reply;
}

} // namespace

void CombReachIndex::bind(const std::vector<CarrierGrammarState> *grammar,
                          int firstActiveLine,
                          int lastActiveLine)
{
    grammar_ = grammar;
    firstActiveLine_ = firstActiveLine;
    lastActiveLine_ = lastActiveLine;
}

const CarrierGrammarState *CombReachIndex::grammarLine(int line) const
{
    if (!grammar_ || line < firstActiveLine_ || line >= lastActiveLine_)
        return nullptr;
    if (line < 0 || line >= static_cast<int>(grammar_->size()))
        return nullptr;
    return &(*grammar_)[line];
}

CombReachReply CombReachIndex::query(const CombReachRequest &request) const
{
    const CarrierGrammarState *center = grammarLine(request.centerLine);
    const CarrierGrammarState *target = grammarLine(request.targetLine);

    return queryGrammarPair(request, center, target);
}

CombReachReply CombReachIndex::queryAgainst(const CombReachIndex &targetIndex,
                                            const CombReachRequest &request) const
{
    const CarrierGrammarState *center = grammarLine(request.centerLine);
    const CarrierGrammarState *target = targetIndex.grammarLine(request.targetLine);

    return queryGrammarPair(request, center, target);
}

CombReachSourceFrame makeBucketScalarReachSource()
{
    CombReachSourceFrame source;
    source.kind = CombReachSourceKind::Bucket1DScalar;
    source.signFrame = CarrierSignFrame::UnsignedBucket;
    source.signalClass = CombReachSignalClass::PhasePreservedCarrier;
    source.scalarCarrier = true;
    source.tag = "bucket-1d-scalar";
    return source;
}

// The Locked1DScalar source labels locked1DSource: physically it is
// bandpass(raw) times a flat round-trip scale (~0.994), so the raw carrier
// orientation — including the physical ±2 field-line alternation — is intact
// on every line.  It is a phase-preserved carrier and grammar legality
// decides its reach, same as the bucket scalar.
//
// The historical "common phase / polarity gone by construction" label on this
// buffer described a pre-reform pipeline and misled repeatedly.  One caveat
// from that era is still real and belongs to the source contract:
//
//  - A per-sample DEMOD of this scalar with an unsigned sample class yields
//    IQ that inherits raw signs, NOT Grid4fscIQ.  For interfield IQ use,
//    demod with carrierGrammarSignedSampleClass (lineFlip folded into the
//    phase), or read locked1DTI4fsc/TQ4fsc directly.
//
// New sites should prefer Grid4fscIQ (IQ caches) over re-deriving IQ from
// this scalar.
CombReachSourceFrame makeLocked1DScalarReachSource()
{
    CombReachSourceFrame source;
    source.kind = CombReachSourceKind::Locked1DScalar;
    source.signFrame = CarrierSignFrame::Grid4fsc;
    source.signalClass = CombReachSignalClass::PhasePreservedCarrier;
    source.scalarCarrier = true;
    source.tag = "locked-1d-scalar";
    return source;
}

CombReachSourceFrame makeGrid4fscIQReachSource()
{
    CombReachSourceFrame source;
    source.kind = CombReachSourceKind::Grid4fscIQ;
    source.signFrame = CarrierSignFrame::Grid4fsc;
    source.signalClass = CombReachSignalClass::PhasePreservedCarrier;
    source.iqCarrier = true;
    source.tag = "grid-4fsc-iq";
    return source;
}

} // namespace lddecode

// ===========================================================================
// Image-content reach authority
// ===========================================================================
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

} // namespace

IntrafieldRegionReach evaluateIntrafieldRegionReach(
    const std::complex<double> &center,
    const std::complex<double> &up,
    const std::complex<double> &down,
    lddecode::CarrierPhaseRelation upRelation,
    lddecode::CarrierPhaseRelation downRelation,
    bool allowUp,
    bool allowDown,
    double invIreScale,
    double minChromaIRE)
{
    IntrafieldRegionReach out;

    auto relationSign = [](lddecode::CarrierPhaseRelation relation) {
        switch (relation) {
        case lddecode::CarrierPhaseRelation::Same:     return 1.0;
        case lddecode::CarrierPhaseRelation::Opposite: return -1.0;
        default:                                       return 0.0;
        }
    };

    const double upSign = allowUp ? relationSign(upRelation) : 0.0;
    const double downSign = allowDown ? relationSign(downRelation) : 0.0;
    const bool haveUp = (upSign != 0.0);
    const bool haveDown = (downSign != 0.0);
    if (!haveUp && !haveDown)
        return out;

    const std::complex<double> alignedUp = upSign * up;
    const std::complex<double> alignedDown = downSign * down;
    const double scale = std::max(0.0, invIreScale);
    const double centerMagIRE = std::abs(center) * scale;
    const double upMagIRE = std::abs(alignedUp) * scale;
    const double downMagIRE = std::abs(alignedDown) * scale;
    const double chromaFloor = std::max(0.0, minChromaIRE);

    // Positive non-membership threshold at a 20-degree hue difference, with
    // a small hysteresis band below it so marginal hue drift cannot force
    // either a one-sided reach or a center-island verdict.
    constexpr double kSameHueDeg = 15.0;
    constexpr double kDifferentHueDeg = 20.0;
    constexpr double kRadiansToDegrees = 57.2957795130823208768;

    // Saturation-boundary evidence.  Hue is meaningless on a vector below the
    // chroma floor, but a clearly saturated side whose color simply does not
    // continue into the other side is a region break in its own right — this
    // is the bikini-against-skin case, where the weak side never qualifies
    // for the hue test and the pair otherwise stays Unknown while the comb
    // mixes near-complementary hues into a gray band.  The ratio gap between
    // Different and Unknown mirrors the hue hysteresis band.
    constexpr double kDifferentMagRatio = 0.35;
    constexpr double kStrongSideFloorScale = 2.0;

    auto classify = [&](const std::complex<double> &leg,
                        double legMagIRE,
                        bool haveLeg,
                        double &differenceIRE,
                        double &hueDifferenceDeg) {
        if (!haveLeg)
            return RegionRelation::Unknown;

        const double magHi = std::max(centerMagIRE, legMagIRE);
        const double magLo = std::min(centerMagIRE, legMagIRE);
        if (magHi < chromaFloor)
            return RegionRelation::Unknown;

        differenceIRE = std::abs(center - leg) * scale;

        // Saturation collapse: the weak side is a small fraction of the
        // strong side.  This is the drop-shadow band signature, and it is
        // RATIO-driven, not floor-driven: a garment fade row at 8 IRE
        // against 25 IRE is as much a band member as a 3 IRE one — gating
        // membership on an absolute weak-side floor re-created the
        // pixel-flicker at every row that straddled it.  The floor only
        // qualifies the strong side: band-grade needs a clearly saturated
        // region (>= 2.5x floor); a weaker strong side still earns a plain
        // Different verdict when its partner is sub-floor, but never band
        // membership (keeps low-chroma texture out of the band machinery).
        constexpr double kBandStrongScale = 2.5;
        if (magLo <= kDifferentMagRatio * magHi) {
            if (magHi >= kBandStrongScale * chromaFloor) {
                out.strongAsym = true;
                return RegionRelation::DifferentRegion;
            }
            if (magLo < chromaFloor &&
                magHi >= kStrongSideFloorScale * chromaFloor)
                return RegionRelation::DifferentRegion;
        }
        if (magLo < chromaFloor)
            return RegionRelation::Unknown;

        const double denom = std::abs(center) * std::abs(leg);
        if (denom <= 1e-12)
            return RegionRelation::Unknown;

        const double hueCos = std::clamp(
            std::real(center * std::conj(leg)) / denom,
            -1.0,
            1.0);
        hueDifferenceDeg = std::acos(hueCos) * kRadiansToDegrees;

        if (hueDifferenceDeg <= kSameHueDeg)
            return RegionRelation::SameRegion;

        // Anti-aligned at comparable magnitude means raw-identical content:
        // vertically coherent luma (cross-color), the comb's ideal
        // cancellation partner.  Ceding here hands the alien energy to 1D,
        // which renders it as a line-alternating rainbow.  A genuine
        // complementary-color boundary does not land in this window because
        // region breaks that matter are magnitude-asymmetric.
        constexpr double kAlienHueDeg = 165.0;
        constexpr double kAlienMagRatio = 0.60;
        if (hueDifferenceDeg >= kAlienHueDeg &&
            magLo >= kAlienMagRatio * magHi)
            return RegionRelation::AlienCancel;

        if (hueDifferenceDeg >= kDifferentHueDeg)
            return RegionRelation::DifferentRegion;
        return RegionRelation::Unknown;
    };

    out.up = classify(alignedUp, upMagIRE, haveUp,
                      out.upDifferenceIRE, out.upHueDifferenceDeg);
    out.down = classify(alignedDown, downMagIRE, haveDown,
                        out.downDifferenceIRE, out.downHueDifferenceDeg);
    out.valid = (out.up != RegionRelation::Unknown ||
                 out.down != RegionRelation::Unknown);

    if (haveUp && haveDown) {
        out.upDownDifferenceIRE = std::abs(alignedUp - alignedDown) * scale;
        const double outerDenom = std::abs(alignedUp) * std::abs(alignedDown);
        if (upMagIRE >= chromaFloor && downMagIRE >= chromaFloor &&
            outerDenom > 1e-12)
        {
            const double outerHueCos = std::clamp(
                std::real(alignedUp * std::conj(alignedDown)) / outerDenom,
                -1.0,
                1.0);
            out.upDownHueDifferenceDeg =
                std::acos(outerHueCos) * kRadiansToDegrees;
        }
        out.centerIsland =
            out.up == RegionRelation::DifferentRegion &&
            out.down == RegionRelation::DifferentRegion;
        out.threeRegion =
            out.centerIsland &&
            out.upDownHueDifferenceDeg >= kDifferentHueDeg;
    }

    return out;
}

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

} // namespace CombContentReach
