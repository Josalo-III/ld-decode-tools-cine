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

#include "combmath.h"

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
           use == CombReachUse::FieldScalarSupport ||
           use == CombReachUse::FrameScalarAverage ||
           use == CombReachUse::FrameScalarCancel ||
           use == CombReachUse::ScalarSignCompare ||
           use == CombReachUse::ScalarMagnitudeCompare;
}

bool frameScalarUse(CombReachUse use)
{
    return use == CombReachUse::FrameScalarAverage ||
           use == CombReachUse::FrameScalarCancel;
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

    // Frame scalar reach crosses the two interleaved fields.  The carrier
    // relation can still be arithmetically Same/Opposite at an edit split,
    // but the frame is not a legal vertical source there.  Keep that schedule
    // authority inside the reach translator instead of making each consumer
    // remember a second private gate.
    if (frameScalarUse(request.use) &&
        (!center->frameVerticalAllowed || !target->frameVerticalAllowed))
    {
        return blockedReply(request, "frame-vertical-blocked");
    }

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

    const bool sameRelation =
        reply.carrierRelation == CarrierPhaseRelation::Same;
    const bool oppositeRelation =
        reply.carrierRelation == CarrierPhaseRelation::Opposite;

    if (sameRelation || oppositeRelation)
    {
        const bool fieldSupport =
            request.use == CombReachUse::FieldScalarSupport;
        reply.verdict = CombReachVerdict::Green;
        reply.fastPath = true;
        reply.allowScalarAverage = sameRelation || fieldSupport;
        reply.allowScalarCancel = oppositeRelation || fieldSupport;
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

// Four-view carrierFit is a scalar remodulation in the measured burst-locked
// composite frame.  It preserves physical carrier polarity and may become
// video only through grammar-authorized scalar reach.
CombReachSourceFrame makeCarrierFitScalarReachSource()
{
    CombReachSourceFrame source;
    source.kind = CombReachSourceKind::CarrierFitScalar;
    source.signFrame = CarrierSignFrame::BurstLockedSigned;
    source.signalClass = CombReachSignalClass::PhasePreservedCarrier;
    source.scalarCarrier = true;
    source.tag = "carrier-fit-scalar";
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
    return boundedMag(i, q);
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

// Leg-vs-leg coherence.  States no policy: it reports what the pair looks
// like and whether the comparison was possible at all.  See combreach.h for
// the signal-frame requirement (callers pass ALREADY relation-aligned legs).
LegPairCoherence evaluateLegPairCoherence(
    const std::complex<double> &legA,
    const std::complex<double> &legB,
    double invIreScale,
    double minChromaIRE)
{
    constexpr double kRadiansToDegrees = 57.2957795130823208768;

    LegPairCoherence out;
    const double scale = std::max(0.0, invIreScale);
    const double floorIRE = std::max(0.0, minChromaIRE);

    const double magA = boundedMag(legA);
    const double magB = boundedMag(legB);
    const double magAIRE = magA * scale;
    const double magBIRE = magB * scale;

    out.differenceIRE = boundedMag(legA - legB) * scale;

    const double hi = std::max(magAIRE, magBIRE);
    const double lo = std::min(magAIRE, magBIRE);
    out.magRatio = (hi > 1e-12) ? std::clamp(lo / hi, 0.0, 1.0) : 0.0;

    // Hue is meaningless on a vector below the chroma floor.  Leave the angle
    // at its 0.0 default and report the pair as not comparable, so absence is
    // never read as agreement.
    const double denom = magA * magB;
    if (magAIRE >= floorIRE && magBIRE >= floorIRE && denom > 1e-12) {
        const double hueCos = std::clamp(
            std::real(legA * std::conj(legB)) / denom, -1.0, 1.0);
        out.hueDifferenceDeg = std::acos(hueCos) * kRadiansToDegrees;
        out.comparable = true;
    }
    return out;
}

IntrafieldRegionReach evaluateIntrafieldRegionReach(
    const std::complex<double> &center,
    const std::complex<double> &up,
    const std::complex<double> &down,
    lddecode::CarrierPhaseRelation upRelation,
    lddecode::CarrierPhaseRelation downRelation,
    bool allowUp,
    bool allowDown,
    double centerCarrierTrust,
    double upCarrierTrust,
    double downCarrierTrust,
    double invIreScale,
    double minChromaIRE,
    double upRawDiffIRE,
    double downRawDiffIRE,
    double centerEnergyIRE)
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
    const double centerMagRaw = boundedMag(center);
    const double upMagRaw = boundedMag(alignedUp);
    const double downMagRaw = boundedMag(alignedDown);
    const double centerMagIRE = centerMagRaw * scale;
    const double upMagIRE = upMagRaw * scale;
    const double downMagIRE = downMagRaw * scale;
    const double chromaFloor = std::max(0.0, minChromaIRE);

    // The reach query has already provided the carrier grammar: a leg only
    // arrives here with an authorized Same/Opposite relation, and the sign
    // above puts its IQ vector in center's frame. Endpoint conformance trust is
    // intentionally not a second color-admission gate here; consumers need the
    // region fact, including the refusal-to-cross fact, even where trust is not
    // high enough to command video.
    (void)centerCarrierTrust;
    (void)upCarrierTrust;
    (void)downCarrierTrust;

    constexpr double kSameHueDeg = 15.0;
    constexpr double kDifferentHueDeg = 20.0;
    constexpr double kRadiansToDegrees = 57.2957795130823208768;

    // Saturation-boundary evidence. Hue is meaningless below the chroma floor,
    // but a clearly saturated side whose color does not continue into the
    // other side is a region break in its own right.
    constexpr double kDifferentMagRatio = 0.35;
    constexpr double kStrongSideFloorScale = 2.0;

    // First-pass AlienCancel gate. ±2 same-field legs are anti-phase carriers:
    // real chroma flips the waveform and shows a LARGE raw leg difference, so a
    // near-zero raw difference on an energetic center can only be vertically
    // coherent non-carrier energy (the vertical-contrast misread) — the comb's
    // ideal cancellation partner. This is the SHARP fact; the smoothed-IQ hue
    // test below flickers on edge energy. Deciding it here means the authority
    // is right the first time, with no downstream weight revive. A magnitude-
    // asymmetric region break (shadow band) has a large raw difference and can
    // never reach this path, so it cannot override a true boundary.
    constexpr double kRawIdentIRE = 3.0;   // near-identical within noise
    constexpr double kRawEnergyIRE = 4.0;  // center clearly carries energy
    auto rawIdentAlien = [&](double legRawDiffIRE) {
        return legRawDiffIRE >= 0.0 &&
               legRawDiffIRE <= kRawIdentIRE &&
               centerEnergyIRE >= kRawEnergyIRE;
    };

    auto classify = [&](const std::complex<double> &leg,
                        double legMagRaw,
                        double legMagIRE,
                        bool haveLeg,
                        double legRawDiffIRE,
                        double &differenceIRE,
                        double &hueDifferenceDeg) {
        if (!haveLeg)
            return RegionRelation::Unknown;

        if (rawIdentAlien(legRawDiffIRE))
            return RegionRelation::AlienCancel;

        const double magHi = std::max(centerMagIRE, legMagIRE);
        const double magLo = std::min(centerMagIRE, legMagIRE);
        if (magHi < chromaFloor)
            return RegionRelation::Unknown;

        differenceIRE = boundedMag(center - leg) * scale;

        // Saturation collapse / shadow band. This is ratio-driven, not simply
        // floor-driven: a weak member beside a strong saturated member is a
        // region break even when the weak side is still above the chroma floor.
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

        const double denom = centerMagRaw * legMagRaw;
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
        // vertically coherent luma/cross-color, the comb's ideal cancellation
        // partner. This is not one-sided authority.
        constexpr double kAlienHueDeg = 165.0;
        constexpr double kAlienMagRatio = 0.60;
        if (hueDifferenceDeg >= kAlienHueDeg &&
            magLo >= kAlienMagRatio * magHi)
            return RegionRelation::AlienCancel;

        if (hueDifferenceDeg >= kDifferentHueDeg)
            return RegionRelation::DifferentRegion;

        return RegionRelation::Unknown;
    };

    out.up = classify(alignedUp, upMagRaw, upMagIRE, haveUp, upRawDiffIRE,
                      out.upDifferenceIRE, out.upHueDifferenceDeg);
    out.down = classify(alignedDown, downMagRaw, downMagIRE, haveDown, downRawDiffIRE,
                        out.downDifferenceIRE, out.downHueDifferenceDeg);
    out.valid = (out.up != RegionRelation::Unknown ||
                 out.down != RegionRelation::Unknown);

    if (haveUp && haveDown) {
        // Leg-vs-leg coherence via the shared evaluator: the same primitive
        // any comb can call on its own leg pair at its own vertical step.
        // Both operands are already relation-aligned above.
        const LegPairCoherence outer = evaluateLegPairCoherence(
            alignedUp, alignedDown, scale, chromaFloor);
        out.upDownDifferenceIRE = outer.differenceIRE;
        out.upDownHueDifferenceDeg = outer.hueDifferenceDeg;
        out.outerComparable = outer.comparable;

        const bool upDifferent =
            out.up == RegionRelation::DifferentRegion;
        const bool downDifferent =
            out.down == RegionRelation::DifferentRegion;

        out.centerIsland = upDifferent && downDifferent;
        out.threeRegion =
            out.centerIsland &&
            out.upDownHueDifferenceDeg >= kDifferentHueDeg;

        const bool upContinues =
            out.up == RegionRelation::SameRegion ||
            out.up == RegionRelation::AlienCancel;
        const bool downContinues =
            out.down == RegionRelation::SameRegion ||
            out.down == RegionRelation::AlienCancel;

        const bool centerSeparated =
            out.upDifferenceIRE >= 3.0 &&
            out.downDifferenceIRE >= 3.0;

        const bool outerCoherent =
            out.upDownDifferenceIRE <= 3.5 ||
            (out.upDownHueDifferenceDeg > 0.0 &&
             out.upDownHueDifferenceDeg <= kSameHueDeg);

        const bool outerDifferent =
            out.upDownDifferenceIRE >= 6.0 ||
            out.upDownHueDifferenceDeg >= kDifferentHueDeg;

        const bool saturatedTriplet =
            centerMagIRE >= chromaFloor &&
            std::max(upMagIRE, downMagIRE) >= chromaFloor &&
            (centerMagIRE + upMagIRE + downMagIRE) >= 3.0 * chromaFloor;
            
        const bool anyLocalContinuation =
            upContinues || downContinues;

        const bool bothAlienCancel =
            out.up == RegionRelation::AlienCancel &&
            out.down == RegionRelation::AlienCancel;
            


        // Restored Field-B cede law, now expressed as reach evidence.
        //
        // A local SameRegion result is not sufficient authority for Field B
        // when the signed chroma geometry says the center is materially
        // separated from both +/-2 legs. This is the shirt/band zipper case:
        // allowing Field B to comb here manufactures alternating output.
        //
        // AlienCancel remains protected when both legs are alien partners,
        // because that case is the intended cancellation path for coherent
        // luma/cross-color rather than a saturated chroma-region band.
        const bool destructiveFieldBTriplet =
            saturatedTriplet &&
            centerSeparated &&
            anyLocalContinuation &&
            !bothAlienCancel &&
            (outerCoherent || outerDifferent);

        if (destructiveFieldBTriplet) {
            out.up = RegionRelation::DifferentRegion;
            out.down = RegionRelation::DifferentRegion;
            out.valid = true;
            out.centerIsland = true;
            out.threeRegion = outerDifferent;
        }

        // Older conservative path: if neither side continues center, and the
        // center is separated from the two-tap neighborhood, force the missing
        // labels to Different so the cede fact is explicit.
        if (!upContinues && !downContinues &&
            centerSeparated && (outerCoherent || outerDifferent))
        {
            if (out.up == RegionRelation::Unknown)
                out.up = RegionRelation::DifferentRegion;
            if (out.down == RegionRelation::Unknown)
                out.down = RegionRelation::DifferentRegion;

            out.valid = true;
            out.centerIsland = true;
            out.threeRegion = outerDifferent;
        }
    }

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
