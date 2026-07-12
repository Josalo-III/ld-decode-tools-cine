/******************************************************************************
 * combreach.h
 * ld-chroma-decoder comb reach system
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Two cooperating arms, kept in one translation unit:
 *
 *   namespace lddecode        — carrier-grammar reach legality translator
 *                               (CombReachIndex). Answers "is this line-to-line
 *                               operation legal in this signal frame?".
 *   namespace CombContentReach — image-content reach authority. Answers "does
 *                               the picture content here support reaching to a
 *                               neighbour?" (interfield IQ floor, moving-coarse
 *                               contour).
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

    bool centerIsland = false;
    bool threeRegion = false;

    // True when a leg classified Different through the magnitude-asymmetry
    // branch and the strong side was independently admitted by the carrier
    // schedule.  This is the drop-shadow signature, distinct from hue-based
    // Different verdicts in ordinary low-saturation texture.
    bool strongAsym = false;
};

constexpr std::uint8_t IntrafieldRegionCedeCenter = 1u << 0;
constexpr std::uint8_t IntrafieldRegionCedeStrongAsym = 1u << 1;

// Named contributors to a Field B policy decision. Diagnostic only: the
// renderer maps these onto its reason plane, and every cede in the output
// must be traceable to at least one named source here.
enum FieldBPolicyReason : std::uint8_t {
    FieldBPolicyReasonNone          = 0,
    FieldBPolicyReasonRegionCede    = 1u << 0, // region verdicts withdrew both legs
    FieldBPolicyReasonShadowBand    = 1u << 1, // strong magnitude-asymmetry band membership
    FieldBPolicyReasonHEdgeGuard    = 1u << 2, // horizontal luma edge, no positively continuing leg
    FieldBPolicyReasonLumaEdgeCede  = 1u << 3, // graded vertical coarse-luma contrast on a ±2 leg
    FieldBPolicyReasonVerticalBreak = 1u << 4, // hard vertical context break
    FieldBPolicyReasonOneLeg        = 1u << 5, // boundary resolution excluded one leg
    FieldBPolicyReasonBevelCede     = 1u << 6, // dedicated bevel detector contribution
};

// The complete prepared verdict the Field B renderer consumes. Leg selection
// and center cede are independent policy dimensions:
//
//   upWeight/downWeight — relative leg mix plus eligibility (0 excludes the
//     leg). The renderer NORMALIZES accepted legs to full comb strength, so
//     a weight can shift the vertical estimate toward the better leg but can
//     never scale the output amplitude.
//
//   centerCede — the ONLY channel by which analysis reduces Field B output
//     toward the 1D center, graded [0,1]. Deriving output strength from the
//     sum of leg weights is the conflation this struct exists to prevent.
struct FieldBTapPolicy {
    double upWeight = 0.0;
    double downWeight = 0.0;
    double centerCede = 0.0;
    std::uint8_t reasons = FieldBPolicyReasonNone;
};

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

std::uint8_t intrafieldRegionCedeFlags(
    const IntrafieldRegionReach &region);

// Resolve the full Field B policy from prepared evidence. Consumes the region
// verdicts and cede flags, the baseline pair weights, the horizontal luma
// delta, the per-leg vertical coarse-luma deltas (|coarse0 − coarse±2| in IRE;
// pass 0 when the coarse rows are unavailable), and the dedicated bevel
// detector's cede contribution [0,1]. Emits leg mix/eligibility, explicit
// centerCede, and diagnostic reasons — the renderer performs no analysis.
FieldBTapPolicy resolveFieldBTapPolicy(
    const IntrafieldRegionReach &region,
    std::uint8_t cedeFlags,
    double upWeight,
    double downWeight,
    double horizontalLumaDeltaIRE = 0.0,
    double horizontalLumaEdgeThresholdIRE = 1.0,
    double upCoarseLumaDeltaIRE = 0.0,
    double downCoarseLumaDeltaIRE = 0.0,
    double bevelCede = 0.0);

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

// Fast overload: caller supplies pre-computed IRE-domain magnitudes to avoid
// recomputing ~12 sqrt/hypot calls per pixel inside the function.
InterfieldIQReachFloor interfieldIQReachFloor(double centerI,
                                              double centerQ,
                                              double upI,
                                              double upQ,
                                              double downI,
                                              double downQ,
                                              bool hasUp,
                                              bool hasDown,
                                              double minChromaIRE,
                                              double lumaEdgeFit,
                                              double centerMagIRE,
                                              double upMagIRE,
                                              double downMagIRE);

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

// Salutary-combine reach backoff in [0,1] for the ±1 interfield pair.  The
// signed demod aligns carrier-locked chroma across the pair but anti-aligns
// image-locked alien (luma misread into the chroma band): so real chroma makes
// the two legs point the SAME direction (their sum is salutary — it adds real
// chroma), while alien makes them point OPPOSITE (their sum extracts only the
// alien's antisymmetric part — the 2-px diagonal staircase).  Hue is invariant
// along a diagonal, so this direction test needs no registration: a real
// single-hue diagonal keeps agreeing across the shifted pair and combs, while a
// diagonal alien disagrees and cedes.  Returns 1.0 (inert, legality governs)
// when one-sided or below the chroma floor; otherwise ramps on leg agreement.
// A true chroma vertical boundary (two hues) also disagrees and cedes, which is
// the correct outcome for a ±1 average across a color break.  Both interfield
// combs (Frame A and Frame B) consume this through the shared reachGate, so the
// discrimination lives once, here, not inside either comb.
double interfieldSalutaryReach(double upI,
                               double upQ,
                               double downI,
                               double downQ,
                               bool hasUp,
                               bool hasDown,
                               double minChromaIRE);

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
