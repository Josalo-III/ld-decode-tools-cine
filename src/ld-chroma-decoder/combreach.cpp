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
    const double centerMagIRE = std::abs(center) * scale;
    const double upMagIRE = std::abs(alignedUp) * scale;
    const double downMagIRE = std::abs(alignedDown) * scale;
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

        differenceIRE = std::abs(center - leg) * scale;

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

    out.up = classify(alignedUp, upMagIRE, haveUp, upRawDiffIRE,
                      out.upDifferenceIRE, out.upHueDifferenceDeg);
    out.down = classify(alignedDown, downMagIRE, haveDown, downRawDiffIRE,
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

std::uint8_t intrafieldRegionCedeFlags(
    const IntrafieldRegionReach &region)
{
    if (!region.valid)
        return 0;

    const bool upDifferent =
        region.up == RegionRelation::DifferentRegion;
    const bool downDifferent =
        region.down == RegionRelation::DifferentRegion;
    const bool upSame =
        region.up == RegionRelation::SameRegion;
    const bool downSame =
        region.down == RegionRelation::SameRegion;
    const bool upAlien =
        region.up == RegionRelation::AlienCancel;
    const bool downAlien =
        region.down == RegionRelation::AlienCancel;

    std::uint8_t flags = 0;

    if (region.centerIsland)
        flags |= IntrafieldRegionCedeCenter;

    // Shadow-band membership outranks any one-sided continuation.  The band
    // should take one render, not alternate comb/center across columns.
    if (region.strongAsym)
        flags |= IntrafieldRegionCedeCenter |
                 IntrafieldRegionCedeStrongAsym;

    const bool upContinues = upSame || upAlien;
    const bool downContinues = downSame || downAlien;

    // No valid same-region partner: center cede.
    if ((upDifferent && !downContinues) ||
        (downDifferent && !upContinues))
    {
        flags |= IntrafieldRegionCedeCenter;
    }

    // Different + Same is not automatically legal one-sided Field B.
    // It is legal only when the surviving Same leg is a close center-region
    // continuation and the rejected leg is clearly worse.  Otherwise this is
    // the residual single-leg zipper: one contaminated leg survives and Field B
    // still writes an alternating bead.
    constexpr double kStrongSameMaxDiffIRE = 3.25;
    constexpr double kWeakSameMaxDiffIRE   = 4.25;
    constexpr double kRejectMinDiffIRE     = 5.25;
    constexpr double kRejectGapIRE         = 2.25;

    if (upDifferent && downSame) {
        const double sameDiff = region.downDifferenceIRE;
        const double rejectDiff = region.upDifferenceIRE;
        const bool strongSame =
            sameDiff <= kStrongSameMaxDiffIRE ||
            (sameDiff <= kWeakSameMaxDiffIRE &&
             rejectDiff >= kRejectMinDiffIRE &&
             (rejectDiff - sameDiff) >= kRejectGapIRE);

        if (!strongSame)
            flags |= IntrafieldRegionCedeCenter;
    }

    if (downDifferent && upSame) {
        const double sameDiff = region.upDifferenceIRE;
        const double rejectDiff = region.downDifferenceIRE;
        const bool strongSame =
            sameDiff <= kStrongSameMaxDiffIRE ||
            (sameDiff <= kWeakSameMaxDiffIRE &&
             rejectDiff >= kRejectMinDiffIRE &&
             (rejectDiff - sameDiff) >= kRejectGapIRE);

        if (!strongSame)
            flags |= IntrafieldRegionCedeCenter;
    }

    // Lone AlienCancel is cancellation evidence, not one-sided Field B
    // authority.  It is useful only when paired with another admitted leg.
    if ((upAlien && !downContinues) ||
        (downAlien && !upContinues))
    {
        flags |= IntrafieldRegionCedeCenter;
    }

    return flags;
}

FieldBTapPolicy resolveFieldBTapPolicy(
    const IntrafieldRegionReach &region,
    std::uint8_t cedeFlags,
    double upWeight,
    double downWeight,
    double horizontalLumaDeltaIRE,
    double horizontalLumaEdgeThresholdIRE,
    double upCoarseLumaDeltaIRE,
    double downCoarseLumaDeltaIRE,
    double bevelCede)
{
    FieldBTapPolicy out;
    out.upWeight = std::clamp(upWeight, 0.0, 1.0);
    out.downWeight = std::clamp(downWeight, 0.0, 1.0);

    // A full cede is terminal: no leg may participate and the output is the
    // 1D center. Weights and cede are still set independently so the renderer
    // never has to infer one from the other.
    auto fullCede = [&](std::uint8_t reasons) -> FieldBTapPolicy & {
        out.upWeight = 0.0;
        out.downWeight = 0.0;
        out.centerCede = 1.0;
        out.reasons |= reasons;
        return out;
    };

    // Vertical coarse-luma service, restored from the July 5 renderer. The ±2
    // legs sit two lines away; coarse-luma contrast between center and a leg
    // means the same-context premise behind vertical cancellation is failing,
    // which is exactly the vertical-comb-across-a-luma-edge alternation that
    // serrates diagonals. Graded contrast attenuates the leg's mix share and
    // contributes graded cede; a hard break invalidates Field B outright and
    // is expressed as raw IRE so no later normalization can undo it.
    constexpr double kLumaEdgeLoIRE = 6.0;
    constexpr double kLumaEdgeHiIRE = 20.0;
    constexpr double kHardVerticalBreakIRE = 14.0;
    const double dUpIRE = std::max(0.0, upCoarseLumaDeltaIRE);
    const double dDnIRE = std::max(0.0, downCoarseLumaDeltaIRE);
    const double lumaEdgeUp = std::clamp(
        (dUpIRE - kLumaEdgeLoIRE) / (kLumaEdgeHiIRE - kLumaEdgeLoIRE), 0.0, 1.0);
    const double lumaEdgeDn = std::clamp(
        (dDnIRE - kLumaEdgeLoIRE) / (kLumaEdgeHiIRE - kLumaEdgeLoIRE), 0.0, 1.0);

    if (std::max(dUpIRE, dDnIRE) >= kHardVerticalBreakIRE)
        return fullCede(FieldBPolicyReasonVerticalBreak);

    if ((cedeFlags & IntrafieldRegionCedeCenter) != 0) {
        std::uint8_t reasons = FieldBPolicyReasonRegionCede;
        if ((cedeFlags & IntrafieldRegionCedeStrongAsym) != 0)
            reasons |= FieldBPolicyReasonShadowBand;
        return fullCede(reasons);
    }

    const double edgeThresh = std::max(1.0, horizontalLumaEdgeThresholdIRE);
    const double hEdge = std::clamp(
        (horizontalLumaDeltaIRE - 0.45 * edgeThresh) /
        (0.55 * edgeThresh),
        0.0,
        1.0);

    // Applied to every non-terminal path on the way out: graded vertical
    // contrast shapes the mix per leg and cedes the output in proportion,
    // and the dedicated bevel detector's contribution is maxed in.
    auto finishPolicy = [&]() -> FieldBTapPolicy & {
        out.upWeight *= (1.0 - 0.65 * lumaEdgeUp);
        out.downWeight *= (1.0 - 0.65 * lumaEdgeDn);
        const double lumaEdgeCede = std::max(lumaEdgeUp, lumaEdgeDn);
        if (lumaEdgeCede > 0.0) {
            out.centerCede = std::max(out.centerCede, lumaEdgeCede);
            out.reasons |= FieldBPolicyReasonLumaEdgeCede;
        }
        if (bevelCede > 0.0) {
            out.centerCede = std::max(out.centerCede,
                                      std::clamp(bevelCede, 0.0, 1.0));
            out.reasons |= FieldBPolicyReasonBevelCede;
        }
        return out;
    };

    if (!region.valid) {
        if (hEdge > 0.0)
            return fullCede(FieldBPolicyReasonHEdgeGuard);
        return finishPolicy();
    }

    const bool upSame =
        region.up == RegionRelation::SameRegion;
    const bool downSame =
        region.down == RegionRelation::SameRegion;
    const bool upDifferent =
        region.up == RegionRelation::DifferentRegion;
    const bool downDifferent =
        region.down == RegionRelation::DifferentRegion;
    const bool upAlien =
        region.up == RegionRelation::AlienCancel;
    const bool downAlien =
        region.down == RegionRelation::AlienCancel;

    const bool upCancellationPartner =
        upAlien && (downSame || downAlien);
    const bool downCancellationPartner =
        downAlien && (upSame || upAlien);

    const bool upContinuesCenter =
        upSame || upCancellationPartner;
    const bool downContinuesCenter =
        downSame || downCancellationPartner;

    const bool upBreaksCenter = upDifferent;
    const bool downBreaksCenter = downDifferent;

    if (region.centerIsland || region.strongAsym) {
        std::uint8_t reasons = FieldBPolicyReasonRegionCede;
        if (region.strongAsym)
            reasons |= FieldBPolicyReasonShadowBand;
        return fullCede(reasons);
    }

    // First establish the region-authorized baseline.
    //
    // Same/Same does not mean "force both to 1.0 forever"; it means both legs
    // are eligible to participate. The relative distances below decide equal,
    // weighted, or one-legged contribution.
    if (upContinuesCenter)
        out.upWeight = 1.0;
    if (downContinuesCenter)
        out.downWeight = 1.0;

    // Clear explicit refusal-to-cross cases.
    if (downBreaksCenter && upContinuesCenter) {
        out.downWeight = 0.0;
        out.reasons |= FieldBPolicyReasonOneLeg;
    } else if (upBreaksCenter && downContinuesCenter) {
        out.upWeight = 0.0;
        out.reasons |= FieldBPolicyReasonOneLeg;
    }

    // AlienCancel is a cancellation partner, not lone one-sided authority.
    if (upAlien && !(downSame || downAlien))
        out.upWeight = 0.0;
    if (downAlien && !(upSame || upAlien))
        out.downWeight = 0.0;

    // Same/Same leg balancing.
    //
    // This handles shallow diagonal color transitions where center and one leg
    // remain cleanly in the same region, while the other leg lands on the
    // transition ramp. The transition leg may still pass SameRegion, but it
    // should not receive equal authority.
    if (upSame && downSame) {
        const double dUp = std::max(0.0, region.upDifferenceIRE);
        const double dDn = std::max(0.0, region.downDifferenceIRE);

        const double nearDiff = std::min(dUp, dDn);
        const double farDiff = std::max(dUp, dDn);
        const double gap = farDiff - nearDiff;

        const bool upNear = dUp <= dDn;

        // Do not disturb ordinary nearly-equal Same/Same reaches.
        constexpr double kEqualGapIRE = 0.85;

        // Middle zone: both legs still contribute, but the farther leg is
        // attenuated. This catches soft diagonal ramps without snapping them
        // into a binary one-leg/center pattern.
        constexpr double kWeightGapStartIRE = 1.00;
        constexpr double kWeightGapFullIRE  = 3.25;

        // One-leg zone: the farther leg is too contaminated to be averaged
        // with the clean leg.
        constexpr double kOneLegGapIRE      = 3.75;
        constexpr double kOneLegFarDiffIRE  = 4.25;
        constexpr double kOneLegRatio       = 2.35;

        if (gap > kEqualGapIRE) {
            const double denom = std::max(farDiff, 1e-9);
            const double ratio = farDiff / std::max(nearDiff, 0.50);

            const bool oneLeg =
                gap >= kOneLegGapIRE ||
                (farDiff >= kOneLegFarDiffIRE && ratio >= kOneLegRatio);

            if (oneLeg) {
                if (upNear) {
                    out.upWeight = 1.0;
                    out.downWeight = 0.0;
                } else {
                    out.upWeight = 0.0;
                    out.downWeight = 1.0;
                }
                out.reasons |= FieldBPolicyReasonOneLeg;
            } else {
                const double t = std::clamp(
                    (gap - kWeightGapStartIRE) /
                    (kWeightGapFullIRE - kWeightGapStartIRE),
                    0.0,
                    1.0);

                // At the start of the middle zone, keep the farther leg
                // almost equal. By the end, reduce it strongly but do not
                // yet make a hard one-legged choice.
                const double farScale = 1.0 - 0.70 * t;

                if (upNear) {
                    out.upWeight = 1.0;
                    out.downWeight *= farScale;
                } else {
                    out.upWeight *= farScale;
                    out.downWeight = 1.0;
                }
            }
        }
    }

    if (hEdge > 0.0 && !upContinuesCenter && !downContinuesCenter)
        return fullCede(FieldBPolicyReasonHEdgeGuard);

    // Keep the existing h-edge one-sided boundary rule, but it now applies
    // after Same/Same balancing rather than being the only source of asymmetry.
    if (hEdge > 0.0 && region.upDownDifferenceIRE > 8.0) {
        const bool preferUp =
            region.upDifferenceIRE <= region.downDifferenceIRE;
        const double diffGap =
            std::fabs(region.upDifferenceIRE - region.downDifferenceIRE);
        const double bestDiff =
            std::min(region.upDifferenceIRE, region.downDifferenceIRE);
        const double worstDiff =
            std::max(region.upDifferenceIRE, region.downDifferenceIRE);
        const double diffRatio =
            (worstDiff > 1e-9) ? (bestDiff / worstDiff) : 1.0;

        constexpr double kMatchIRE = 3.5;
        constexpr double kBetweenIRE = 6.0;

        const bool hardPreferUp =
            preferUp &&
            ((region.upDifferenceIRE < kMatchIRE &&
              region.downDifferenceIRE > kBetweenIRE) ||
             (bestDiff < 4.5 && diffGap > 2.5 && diffRatio < 0.55));

        const bool hardPreferDown =
            !preferUp &&
            ((region.downDifferenceIRE < kMatchIRE &&
              region.upDifferenceIRE > kBetweenIRE) ||
             (bestDiff < 4.5 && diffGap > 2.5 && diffRatio < 0.55));

        const bool allowOneSidedBoundary =
            (preferUp && upContinuesCenter && downBreaksCenter) ||
            (!preferUp && downContinuesCenter && upBreaksCenter);

        if ((hardPreferUp || hardPreferDown) && !allowOneSidedBoundary) {
            // A hard one-sided preference whose surviving leg cannot claim a
            // credible boundary continuation is the residual single-leg
            // zipper; the only safe render is the center.
            return fullCede(FieldBPolicyReasonHEdgeGuard);
        } else if (hardPreferUp) {
            out.downWeight = 0.0;
            out.upWeight = std::max(out.upWeight, 0.40 + 0.60 * hEdge);
            out.reasons |= FieldBPolicyReasonOneLeg;
        } else if (hardPreferDown) {
            out.upWeight = 0.0;
            out.downWeight = std::max(out.downWeight, 0.40 + 0.60 * hEdge);
            out.reasons |= FieldBPolicyReasonOneLeg;
        }
    }

    return finishPolicy();
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
