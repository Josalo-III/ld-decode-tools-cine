/******************************************************************************
 * comb_reach_index.cpp
 * ld-decode-tools shared comb reach legality translator
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 ******************************************************************************/

#include "comb_reach_index.h"

#include <algorithm>

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
    if (!center || !target)
        return unknownReply(request, "missing-grammar");

    if (request.source.carrierFree)
        return blockedReply(request, "carrier-free-y");

    if (request.source.kind == CombReachSourceKind::Detector)
        return blockedReply(request, "detector-not-waveform");

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
        reply.tag = "iq-carrier";
        return reply;
    }

    if (!scalarUse(request.use))
        return unknownReply(request, "unknown-use");

    if (!request.source.scalarCarrier)
        return blockedReply(request, "not-scalar-carrier");

    if (request.source.commonRemodulatedPhase) {
        reply.verdict = CombReachVerdict::CommonPhaseOnly;
        reply.fastPath = true;
        reply.allowScalarAverage = true;
        reply.allowScalarMagnitudeCompare = true;
        reply.tag = "locked-common-phase-scalar";
        return reply;
    }

    if (reply.carrierRelation == CarrierPhaseRelation::Same ||
        reply.carrierRelation == CarrierPhaseRelation::Opposite)
    {
        reply.verdict = CombReachVerdict::Green;
        reply.fastPath = true;
        reply.allowScalarAverage = true;
        reply.allowScalarCancel = request.source.physicalPolarityPreserved;
        reply.allowScalarSignCompare = request.source.physicalPolarityPreserved;
        reply.allowScalarMagnitudeCompare = true;
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

CombReachReply CombReachIndex::fieldBUp(int line,
                                        int h,
                                        const CombReachSourceFrame &source,
                                        CombReachUse use) const
{
    return query({line, line - 2, h, h, use, source});
}

CombReachReply CombReachIndex::fieldBDn(int line,
                                        int h,
                                        const CombReachSourceFrame &source,
                                        CombReachUse use) const
{
    return query({line, line + 2, h, h, use, source});
}

CombReachReply CombReachIndex::fieldAUp(int line,
                                        int h,
                                        const CombReachSourceFrame &source,
                                        CombReachUse use) const
{
    return query({line, line - 4, h, h, use, source});
}

CombReachReply CombReachIndex::fieldADn(int line,
                                        int h,
                                        const CombReachSourceFrame &source,
                                        CombReachUse use) const
{
    return query({line, line + 4, h, h, use, source});
}

CombReachSourceFrame makeRawCompositeReachSource()
{
    CombReachSourceFrame source;
    source.kind = CombReachSourceKind::RawCompositeScalar;
    source.signFrame = CarrierSignFrame::UnsignedBucket;
    source.physicalPolarityPreserved = true;
    source.scalarCarrier = true;
    source.tag = "raw-composite";
    return source;
}

CombReachSourceFrame makeBucketScalarReachSource()
{
    CombReachSourceFrame source;
    source.kind = CombReachSourceKind::Bucket1DScalar;
    source.signFrame = CarrierSignFrame::UnsignedBucket;
    source.physicalPolarityPreserved = true;
    source.scalarCarrier = true;
    source.tag = "bucket-1d-scalar";
    return source;
}

CombReachSourceFrame makeLockedCommonPhaseScalarReachSource()
{
    CombReachSourceFrame source;
    source.kind = CombReachSourceKind::LockedCommonPhaseScalar;
    source.signFrame = CarrierSignFrame::Grid4fsc;
    source.commonRemodulatedPhase = true;
    source.physicalPolarityPreserved = false;
    source.scalarCarrier = true;
    source.tag = "locked-common-phase-scalar";
    return source;
}

CombReachSourceFrame makeGrid4fscIQReachSource()
{
    CombReachSourceFrame source;
    source.kind = CombReachSourceKind::Grid4fscIQ;
    source.signFrame = CarrierSignFrame::Grid4fsc;
    source.physicalPolarityPreserved = true;
    source.iqCarrier = true;
    source.tag = "grid-4fsc-iq";
    return source;
}

CombReachSourceFrame makeCarrierFreeYReachSource()
{
    CombReachSourceFrame source;
    source.kind = CombReachSourceKind::CarrierFreeY;
    source.carrierFree = true;
    source.tag = "carrier-free-y";
    return source;
}

CombReachSourceFrame makeDetectorReachSource()
{
    CombReachSourceFrame source;
    source.kind = CombReachSourceKind::Detector;
    source.tag = "detector";
    return source;
}

} // namespace lddecode
