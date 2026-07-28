/******************************************************************************
 * combreach.h
 * ld-chroma-decoder comb reach system
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Two physical-evidence arms, kept in one translation unit:
 *
 *   namespace lddecode        — carrier-grammar reach legality translator
 *                               (CombReachIndex). Answers "is this line-to-line
 *                               operation legal in this signal frame?".
 *   namespace CombContentReach — image-content observations shared by combs.
 *                               It reports region and contour facts but never
 *                               selects legs, scales a comb, or cedes to 1D.
 *
 * Formerly comb_reach_index.{h,cpp} (in library/tbc) and combcontentreach.{h,cpp}.
 * Merged 2026-06-21: the reach index was used only by ld-chroma-decoder, so it
 * no longer earns a slot in the shared library. See the comb-reach archeology
 * note for the dead-code that was pulled in the same pass.
 ******************************************************************************/

#pragma once

#include <complex>
#include <cstdint>
#include <vector>

#include "attributiondefs.h"
#include "carriergrammar.h"

// ===========================================================================
// Carrier-grammar reach legality translator
// ===========================================================================
namespace lddecode {

enum class CombReachUse {
    FieldScalarAverage,
    FieldScalarCancel,
    FieldScalarSupport,
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
    DiagnosticOnly,
    IQOnly,
    PriorOnly,
    Unknown,
    Blocked
};

enum class CombReachSourceKind {
    RawCompositeScalar,
    Bucket1DScalar,
    LockedScalar,
    Locked1DScalar,
    CarrierFitScalar,
    Grid4fscIQ,
    BurstLockedIQ,
    CarrierFreeY,
    Detector,
    Unknown
};

// What class of signal a source is.  This is the video/diagnostic boundary:
// a client must be able to read it off the declaration alone.
//
//   PhasePreservedCarrier — physical carrier polarity intact.  Video-capable
//     operations are then authorized (or not) by carrier-grammar legality for
//     the requested line pair.
//   CarrierFree — luma/scalar evidence with no carrier semantics.  Never a
//     carrier operand.
//   PhaseErasedDiagnostic — carrier phase has been deliberately erased
//     (e.g. remodulated to a common phase for comparison convenience).  Such
//     data may inspect math and score coherence but must never become video:
//     the reach reply mechanically blocks every video-capable operation for
//     this class, regardless of grammar.  There is currently no producer of
//     this class in the tree; the class exists so any future phase-erased
//     buffer is inconvenient to misuse.
enum class CombReachSignalClass {
    Unknown,
    PhasePreservedCarrier,
    CarrierFree,
    PhaseErasedDiagnostic
};

struct CombReachSourceFrame {
    CombReachSourceKind kind = CombReachSourceKind::Unknown;
    CarrierSignFrame signFrame = CarrierSignFrame::UnsignedBucket;
    CombReachSignalClass signalClass = CombReachSignalClass::Unknown;
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

    // Diagnostic-only comparison.  May inspect math and score coherence; must
    // not produce video or authorize a carrier-bearing downstream source.
    bool allowDiagnosticCompare = false;

    // Video-capable operations.  Granted only for PhasePreservedCarrier
    // sources with a grammar-legal relation for the requested reach.
    bool allowScalarAverage = false;
    bool allowScalarCancel = false;
    bool allowScalarSignCompare = false;
    bool allowScalarMagnitudeCompare = false;
    bool allowIQCompare = false;
    bool allowIQAverage = false;
    bool allowIQCancel = false;

    // True only when the source and requested operation preserve the
    // downstream video phase contract.  Never true for PhaseErasedDiagnostic.
    bool mayBecomeVideo = false;

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

private:
    const CarrierGrammarState *grammarLine(int line) const;

    const std::vector<CarrierGrammarState> *grammar_ = nullptr;
    int firstActiveLine_ = 0;
    int lastActiveLine_ = 0;
};

CombReachSourceFrame makeBucketScalarReachSource();
CombReachSourceFrame makeLocked1DScalarReachSource();
CombReachSourceFrame makeCarrierFitScalarReachSource();
CombReachSourceFrame makeGrid4fscIQReachSource();

} // namespace lddecode

// ===========================================================================
// Image-content reach authority
// ===========================================================================
namespace CombContentReach {

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

enum class RegionRelation {
    Unknown,
    SameRegion,
    DifferentRegion,
    // Schedule-rejected partner that is anti-aligned at comparable magnitude:
    // raw-identical content, i.e. vertically coherent luma (cross-color), not
    // a chroma region.  The comb difference cancels it exactly, so this verdict
    // must neither cede nor grant one-sided authority.
    AlienCancel
};

// Leg-vs-leg coherence: does a PAIR of comb legs offer a common chroma to
// cancel against?  Published as evidence for every comb, at any vertical step
// -- intrafield (+/-2, +/-4) and interfield (+/-1) alike -- because the
// question is a property of the leg pair alone.  Each comb owns what it does
// with the answer; this evaluator states no policy.
//
// SIGNAL FRAME: both legs must ALREADY be relation-aligned into a common
// frame by the caller (grammar sign applied).  Comparing raw-frame legs here
// would read a perfect match as a maximal break.
//
// `comparable` is the validity fact and must be consulted: when either leg is
// below the chroma floor the angle is not measurable and hueDifferenceDeg
// stays 0.0, which a consumer would otherwise read as "the legs agree" --
// absence of evidence masquerading as positive evidence.
struct LegPairCoherence {
    bool comparable = false;        // both legs cleared the chroma floor
    double differenceIRE = 0.0;     // |legA - legB|, aligned frame
    double hueDifferenceDeg = 0.0;  // angle between the legs; 0 unless comparable
    double magRatio = 0.0;          // min/max magnitude, 0..1
};

LegPairCoherence evaluateLegPairCoherence(
    const std::complex<double> &legA,
    const std::complex<double> &legB,
    double invIreScale,
    double minChromaIRE);

// Shared content verdict for the two intrafield combs.  The IQ inputs retain
// their source sign; the evaluator applies the grammar relations only to local
// comparison copies.  Bandpass IQ describes an already-admitted carrier but
// never admits itself: color-region verdicts also require the registered
// schedule-conformance fact for each operand.
struct IntrafieldRegionReach {
    bool valid = false;
    RegionRelation up = RegionRelation::Unknown;
    RegionRelation down = RegionRelation::Unknown;

    double upDifferenceIRE = 0.0;
    double downDifferenceIRE = 0.0;
    double upDownDifferenceIRE = 0.0;
    double upHueDifferenceDeg = 0.0;
    double downHueDifferenceDeg = 0.0;
    double upDownHueDifferenceDeg = 0.0;
    bool outerComparable = false;

    bool centerIsland = false;
    bool threeRegion = false;
    bool strongAsym = false;

    // A local sample whose vertical chroma geometry proves that the
    // +/-2 aperture crosses a chroma-region boundary.
    bool chromaBoundarySeed = false;

    // Row-level expansion of chromaBoundarySeed. Every sample in this
    // band receives one uniform render from consumers that adopt the rule.
    bool chromaBoundaryBand = false;
};

void markIntrafieldChromaBoundaryBand(
    std::vector<IntrafieldRegionReach> &row,
    int radius);

// Vertical companion to markIntrafieldChromaBoundaryBand. The horizontal
// pass dilates chromaBoundaryBand within one line only; each line's band is
// otherwise decided independently, so a boundary sitting near the seed
// threshold can flip on/off line to line -- a jagged ("treeline") edge on
// every consumer that ceded uniformly per line (Field B's cede-to-center,
// the Y-election's band cede). Call once per field after every line's band
// has been written into `flat`, before any downstream consumer reads it.
// OR-dilates in place along the line axis per column, radius lines each
// side, using a snapshot of the input so already-dilated lines cannot
// compound into their neighbours' windows.
void markVerticalChromaBoundaryBand(
    std::vector<std::uint8_t> &flat,
    int stride,
    int width,
    int firstLine,
    int lastLine,
    int radius);

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
    // Sharp raw-scalar facts for the first-pass AlienCancel decision. ±2 same-
    // field legs are anti-phase carriers, so real chroma shows a large raw leg
    // difference; a near-zero raw difference (IRE) on an energetic center
    // (centerEnergyIRE, raw carrier-band envelope in IRE) is vertically
    // coherent non-carrier energy the comb must cancel. Pass <0 for a leg's
    // diff to disable this path (falls back to the smoothed-IQ hue test only).
    double upRawDiffIRE = -1.0,
    double downRawDiffIRE = -1.0,
    double centerEnergyIRE = 0.0);

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

} // namespace CombContentReach
