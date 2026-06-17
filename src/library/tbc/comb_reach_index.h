/******************************************************************************
 * comb_reach_index.h
 * ld-decode-tools shared comb reach legality translator
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 ******************************************************************************/

#pragma once

#include <vector>

#include "carriergrammar.h"

namespace lddecode {

enum class CombReachUse {
    FieldScalarAverage,
    FieldScalarCancel,
    FrameScalarAverage,
    FrameScalarCancel,
    ScalarSignCompare,
    ScalarMagnitudeCompare,
    IQCompare,
    IQAverage,
    IQCancel
};

enum class CombReachVerdict {
    Green,
    CommonPhaseOnly,
    IQOnly,
    PriorOnly,
    Unknown,
    Blocked
};

enum class CombReachSourceKind {
    RawCompositeScalar,
    Bucket1DScalar,
    LockedScalar,
    LockedCommonPhaseScalar,
    Grid4fscIQ,
    BurstLockedIQ,
    CarrierFreeY,
    Detector,
    Unknown
};

struct CombReachSourceFrame {
    CombReachSourceKind kind = CombReachSourceKind::Unknown;
    CarrierSignFrame signFrame = CarrierSignFrame::UnsignedBucket;
    bool commonRemodulatedPhase = false;
    bool physicalPolarityPreserved = false;
    bool scalarCarrier = false;
    bool iqCarrier = false;
    bool carrierFree = false;
    const char *tag = "unset-source";
};

struct CombReachRequest {
    int centerLine = -1;
    int targetLine = -1;
    int centerH = 0;
    int targetH = 0;
    CombReachUse use = CombReachUse::FieldScalarAverage;
    CombReachSourceFrame source = {};
};

struct CombReachReply {
    CombReachVerdict verdict = CombReachVerdict::Unknown;
    bool valid = false;
    bool fastPath = false;
    bool allowScalarAverage = false;
    bool allowScalarCancel = false;
    bool allowScalarSignCompare = false;
    bool allowScalarMagnitudeCompare = false;
    bool allowIQCompare = false;
    bool allowIQAverage = false;
    bool allowIQCancel = false;
    CarrierPhaseRelation carrierRelation = CarrierPhaseRelation::Unknown;
    CarrierSignFrame centerFrame = CarrierSignFrame::UnsignedBucket;
    CarrierSignFrame targetFrame = CarrierSignFrame::UnsignedBucket;
    double authority = 0.0;
    const char *tag = "unset";
};

class CombReachIndex {
public:
    CombReachIndex() = default;

    void bind(const std::vector<CarrierGrammarState> *grammar,
              int firstActiveLine,
              int lastActiveLine);

    CombReachReply query(const CombReachRequest &request) const;
    CombReachReply queryAgainst(const CombReachIndex &targetIndex,
                                 const CombReachRequest &request) const;

    CombReachReply fieldBUp(int line,
                            int h,
                            const CombReachSourceFrame &source,
                            CombReachUse use = CombReachUse::FieldScalarAverage) const;

    CombReachReply fieldBDn(int line,
                            int h,
                            const CombReachSourceFrame &source,
                            CombReachUse use = CombReachUse::FieldScalarAverage) const;

    CombReachReply fieldAUp(int line,
                            int h,
                            const CombReachSourceFrame &source,
                            CombReachUse use = CombReachUse::FieldScalarAverage) const;

    CombReachReply fieldADn(int line,
                            int h,
                            const CombReachSourceFrame &source,
                            CombReachUse use = CombReachUse::FieldScalarAverage) const;

private:
    const CarrierGrammarState *grammarLine(int line) const;

    const std::vector<CarrierGrammarState> *grammar_ = nullptr;
    int firstActiveLine_ = 0;
    int lastActiveLine_ = 0;
};

CombReachSourceFrame makeRawCompositeReachSource();
CombReachSourceFrame makeBucketScalarReachSource();
CombReachSourceFrame makeLockedCommonPhaseScalarReachSource();
CombReachSourceFrame makeGrid4fscIQReachSource();
CombReachSourceFrame makeCarrierFreeYReachSource();
CombReachSourceFrame makeDetectorReachSource();

} // namespace lddecode
