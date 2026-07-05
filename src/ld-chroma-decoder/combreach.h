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
#include <vector>

#include "carriergrammar.h"

// ===========================================================================
// Carrier-grammar reach legality translator
// ===========================================================================
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
    // Anti-aligned partner at comparable magnitude: raw-identical content,
    // i.e. vertically coherent luma (cross-color) rather than a chroma
    // region.  The comb difference cancels it exactly — the comb is the
    // cure here, so this verdict must neither cede nor grant one-sided
    // authority.  (A real complementary-color boundary is magnitude-
    // asymmetric; shared luma structure is magnitude-symmetric.)
    AlienCancel
};

// Shared content verdict for the two intrafield combs.  The IQ inputs retain
// their source sign; the evaluator applies the grammar relations only to local
// comparison copies.
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

    // True when a leg classified Different through the sub-floor
    // magnitude-asymmetry branch: a center with no bona-fide chroma of its
    // own against a clearly saturated partner.  This is the drop-shadow
    // signature, distinct from hue-based Different verdicts in ordinary
    // low-saturation texture.
    bool strongAsym = false;
};

IntrafieldRegionReach evaluateIntrafieldRegionReach(
    const std::complex<double> &center,
    const std::complex<double> &up,
    const std::complex<double> &down,
    lddecode::CarrierPhaseRelation upRelation,
    lddecode::CarrierPhaseRelation downRelation,
    bool allowUp,
    bool allowDown,
    double invIreScale,
    double minChromaIRE,
    // Carrier-free vertical luma delta per leg (|smoothLuma0 - smoothLumaLeg|,
    // in IRE). A hue-based Different verdict over vertically-IDENTICAL luma is
    // not a region boundary: the IQ difference is a carrier-phase sampling
    // artifact of fine vertical luma detail whose spatial frequency lands in
    // the subcarrier band (the Borg-cube grid) -- luma the demod misreads as
    // chroma, not chroma. That pair is exactly the comb's cancellation partner,
    // so such a leg is reclassified AlienCancel and Field B combs instead of
    // ceding to 1D center. Pass a negative value (default) when no luma
    // evidence is available -> no effect.
    double upLumaDeltaIRE = -1.0,
    double downLumaDeltaIRE = -1.0);

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
