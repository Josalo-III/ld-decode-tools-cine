/*****************************************************************************
  comb.h

  Header for Comb (NTSC/PAL comb filter / chroma decoder helper)

  This file defines the Comb class which implements the multi-stage comb/
  colour-decoding logic used by the ntsc decoders. The key public API is
  Comb::Configuration which holds runtime options and the nested Tunables
  structure used to tune scoring/penalty behaviour for the 2D/3D adaptive
  filter.

  The comments below document the purpose and effect of each tunable so
  they are easier to understand and adjust without diving into the code.
*****************************************************************************/

#ifndef COMB_H
#define COMB_H

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <QCoreApplication>
#include <QDebug>
#include <QFile>
#include <QtMath>
#include <vector>

#include "lddecodemetadata.h"
#include "attributiondefs.h"
#include "carriergrammar.h"
#include "combreach.h"
#include "combmath.h"
#include "componentframe.h"
#include "decoder.h"
#include "sourcefield.h"

class Comb
{
public:
    Comb();

    // Public configuration used by external callers (e.g. main / Decoder)
    struct Configuration {
        // Chroma transform parameters
        double chromaGain  = 1.0;  // Overall gain applied to chroma after IQ transform.
        double chromaPhase = 0.0;  // Degrees: additional phase rotation applied to chroma.

        // Dimensions: 1 = 1D only, 2 = include 2D spatial processing, 3 = 3D temporal adaptive
        qint32 dimensions  = 2;
        bool   adaptive    = true; // If true, the 3D adaptive candidate selection is used.
        bool   showMap     = false; // If true, produce a diagnostic overlay map (ntsc3d only).
        bool   debugCadence = false; // Draw cadence letter (A, B, C...) on frame
        // Demod plus Y selection: phase locked vs bucket
        // Phase locked is a coherent path that includes HF Y from composite
        bool phaseCompensation = false;

        // Per-axis product gains: multipliers applied to I and Q before filtering.
        double gi_product = 1.0;
        double gq_product = 1.1;

        // Noise reduction levels (0 = disabled)
        double cNRLevel = 0.0;
        double yNRLevel = 0.0;

        // User-facing 3D adaptive filter controls (mapped to tunables at init time).
        // adaptThreshold: multiplier on AGREEMENT_REWARD_MAX — higher values increase
        //   the reward given to temporally-agreeing candidates, biasing selection toward
        //   3D results. Range typically 0.5–2.0; default 1.0 (no scaling).
        // chromaWeight: multiplier on the chroma penalty term in getCandidate — higher
        //   values make chroma disagreement more costly, biasing selection toward 2D/1D
        //   when chroma differs between candidate and reference. Default 1.0.
        double adaptThreshold = 1.0;
        double chromaWeight   = 1.0;

        // Scoring / penalties used in candidate selection:
        enum TwoDVariant {
            Line,             // 1D only
            FieldAContour,    // Field A: contour-aware same-field +/-2 comb with +/-4 support
            FieldBSimple,     // Field B: direct same-field +/-2 comb
            FrameAAdaptiveIQ, // Frame A: column-phase-aligned adaptive interframe IQ comb
            FrameBDirectIQ,   // Frame B: direct interframe IQ comb with IRE-domain reach-floor inputs
            FieldVsFrame      // FVF (Default)
        };
        TwoDVariant twoDVariant = FieldVsFrame;

        // Opt-in for 3D temporal checks in Residual Y (getBestY)
        bool residualVideo3D = false;
        bool residualVideo = false;
        bool residualColor = false;             

        // Scoring / penalties used in candidate selection:
        bool allowOuterFrameTemporal = true;
        bool allowSpatial2DIn3D      = true;

        double candidatePenaltyHardMax = 1e6;
        double earlyOutPenaltyIRE      = 250.0;

        bool   burstFloorEnable  = false;
        double burstFloorFactor  = 800.0;

        double w2d_skip_temporal_threshold = 0.05;

        qint32 getLookBehind() const;
        qint32 getLookAhead()  const;

        struct Tunables {
            // =========================================================================
            // 1D / Lateral baseline
            // =========================================================================
            // Cross-color suppression strength (--cross-color-return).
            // alphaEff = max(0, 1 - gA*weight) applied to rendered chroma in
            // both coherent and residual paths; Y subtraction always full.
            // >1.0 drives harder suppression on ambiguous pixels.
            double CC_SUPPRESSION_WEIGHT    = 1.0;

            // =========================================================================
            // 2D Field extraction (FieldA/B) and vertical gating
            // =========================================================================
            double FIELD_K_RANGE_IRE           = 45.0; // max kScore diff (IRE) before field 2D output is penalized
            double FIELD_CONTOUR_FAR_INFLUENCE = 0.55; // weight of ±4-line neighbors relative to ±2 in contour detection
            double FIELD_CONTOUR_SOFT_IRE      = 4.0;  // contour curvature below this → no edge penalty applied
            double FIELD_CONTOUR_HARD_IRE      = 10.0; // contour curvature above this → full edge penalty applied
            double FIELD_CONTOUR_SIM_START     = 0.55; // vertical chroma similarity below which contour gate begins to open
            double FIELD_CONTOUR_SIM_FULL      = 0.85; // similarity above which contour gate is fully closed

            double FIELD_VERT_DISAGREE_THRESH_IRE = 8.0; // suppress 2D field output when ±2 line pair disagrees beyond this

            double FIELD_LUMA_EDGE_THRESH_IRE = 18.0; // horizontal luma gradient above this suppresses vertical 2D comb
            double FIELD_B_BEVEL_REACH_PENALTY = 0.45; // extra Field B reach damping in high-chroma bevels near luma edges
            double FIELD_B_BEVEL_CEDE_STRENGTH = 0.15; // extra Field B center cede in high-chroma bevels near luma edges

            // =========================================================================
            // Frame comb on phase-corrected 1D
            // =========================================================================
            double FRAME_COMB_STRENGTH        = 1.125; // interframe cancellation amplitude scale for Frame A (>1 boosts cancellation)
            double FRAME_CHROMA_MIN_IRE       = 1.5;   // Frame A minimum chroma amplitude to engage the frame IQ path
            double FRAME_IQ_RAW_MAX_DELTA_IRE = 12.0;  // Frame A max IQ mismatch between locked-1D and frame average before frame IQ is distrusted
            double FRAME_B_COMB_STRENGTH       = 1.00; // Frame B center detent: 0.5 * combStrength * reachAuthority pull
            double FRAME_B_CHROMA_MIN_IRE      = 1.5;  // Frame B IRE-domain reach-floor minimum
            double FRAME_B_RAW_MAX_DELTA_IRE   = 12.0; // Frame B IRE-domain direct-IQ delta cap
            double FRAME_B_BEVEL_REACH_PENALTY = 1.0;  // chroma-weighted bevel reach throttle on Frame B ±1; gates near a horizontal luma step where the ±1 partners straddle different bevel phases (zipper guard)

            // =========================================================================
            // FVF (Field vs Frame) scoring
            // =========================================================================
            double FVF_SMALL_DIFF_IRE = 2.0; // ±1 vs ±2 scalar diff ≤ this → field and frame agree; Frame gets a score bonus

            // Spatial neighbor shaping: pulls the FVF winner toward the local majority decision.
            double FVF_SHAPE_STRENGTH = 0.85; // 0 = disabled, 1 = full neighbor pull

            // Scale-regime biasing. The scale estimate is a horizontal proxy derived from
            // Frame B IQ magnitude, then applied as evidence for the vertical-reach election.
            double FVF_SCALE_FINE_FRAME_A_BONUS  = 0.06;
            double FVF_SCALE_FINE_FRAME_B_BONUS  = 0.12;
            double FVF_SCALE_MID_FIELD_B_BONUS   = 0.14;
            double FVF_SCALE_MID_FIELD_A_BONUS   = 0.10;
            double FVF_SCALE_MID_FRAME_A_BONUS   = 0.06;
            double FVF_SCALE_COARSE_FIELD_A_BONUS = 0.25;
            // Additional bonus when tap-line ±4 contour confirms both sides
            // (haveU4 && haveD4 && upSideOk && dnSideOk).  The ±4 evidence
            // is chroma-cancelled same-phase, so when it's vetted Field A
            // is on its strongest footing and deserves a real bonus.
            double FVF_SCALE_COARSE_DUAL4_FIELD_A_BONUS = 0.12;

            // Edge-regime biasing for the 4-member election.
            double FVF_VERT_FIELD_A_PENALTY  = 0.16; // Field A penalty under vertical contrast (±2 comb produces crosstalk)
            double FVF_VERT_FRAME_A_BONUS    = 0.16; // Frame A (precleaned) bonus under vertical contrast (unaffected by vertical luma)
            double FVF_HEDGE_FIELD_B_PENALTY = 0.0; // Field B penalty at horizontal luma edges (zipper risk)
            double FVF_HEDGE_FRAME_B_BONUS   = 0.18; // Frame B bonus at horizontal luma edges (interframe is stable there)

            // Transition sharpness reward strength.
            double FVF_TRANSITION_SHARPNESS_WEIGHT = 0.10;

            // Saturation regime biasing.
            double FVF_SAT_FIELD_A_PEN   = 0.06; // Field A penalty in high saturation
            double FVF_SAT_FIELD_B_PEN   = 0.14; // Field B penalty in high saturation (higher: zipper risk)
            double FVF_SAT_FRAME_A_BONUS = 0.10; // Frame A bonus in high saturation (smaller: boundary caution)
            double FVF_SAT_FRAME_B_BONUS = 0.12; // Frame B bonus in high saturation when coherent

            // =========================================================================
            // FVF / Model interaction tuning
            // =========================================================================
            // Penalty for a candidate being far from the regime model
            // (Frame B in progressive; Field B in interlace). Larger → more model dominance.
            double FVF_MODEL_PRIMARY_WEIGHT = 0.33;
            // Cross-domain candidates pay model-distance only past this deadband
            // (IRE).  Inside the deadband the cross-domain comb is considered
            // model-equivalent — lets the opposite-regime candidate through on
            // flat, low-contrast regions where image-shaping heuristics should
            // still be allowed to surface.
            double FVF_MODEL_PRIMARY_DEADBAND_IRE = 2.0;

            // Multiplicative cost advantage for the model-domain candidate in its own regime.
            // < 1.0 → model candidate gets a cheaper score.
            double FIELD_MODEL_BIAS = 0.50; // interlace: Field B costs less

            // > 1.0 → Frame candidate costs slightly more when regime is clearly interlace.
            double FRAME_IN_INTERLACE_PENALTY = 1.10;

            // =========================================================================
            // 3D candidate / vet / Y path
            // =========================================================================
            // Similarity curve for temporal candidate scoring in getBestCandidate.
            // Let d = |candidate − 2D reference| in IRE:
            //   d ≤ AGREEMENT_REWARD_RADIUS_IRE  → reward: −AGREEMENT_REWARD_MAX·(1−(d/r)²)
            //   d ≤ deviationThreshold            → neutral
            //   d > deviationThreshold            → veto: +AGREEMENT_VETO_BASE + deviationPenalty·(d−threshold)
            // deviationThreshold is shared with getBestY() as the global temporal-mixing veto.
            double AGREEMENT_REWARD_RADIUS_IRE = 8.5; // half-width of the reward lobe (IRE)
            double AGREEMENT_REWARD_MAX        = 3.3; // peak reward at d=0 (penalty units, scaled by adaptThreshold)
            double AGREEMENT_VETO_BASE         = 7.0; // base penalty added once d exceeds deviationThreshold
            double deviationThreshold          = 8.0; // start of veto region (IRE); shared with getBestY
            double deviationPenalty            = 3.3; // penalty slope beyond deviationThreshold (per IRE)

            // 3D Residual Y selection
            bool   RESIDUAL_Y_ELECTION     = true; // true = winner-take-all; false = median-weighted blend
            double NEIGHBOR_SHAPE_STRENGTH = 0.5;  // how strongly spatial neighbor consensus pulls the 3D Y election
            double FRAME_IQ_COLUMN_PHASE_ALIGN_MAX_DEG = 10.0; // clamp for neighbor IQ column phase alignment in frame candidates

            // Attribution-informed Y reassignment: returns bandpassed energy to Y when
            // attribution evidence says it is luma-origin rather than chroma-origin.
            bool   VET_ATTRIBUTION_ENABLE            = true;
            double VET_ATTRIBUTION_LUMA_WEIGHT       = 0.85; // blend strength for lumaClaim -> Y reassignment (0=off, 1=full)
            double VET_ATTRIBUTION_CHROMA_WEIGHT     = 0.75; // blend strength for chromaClaim -> chroma retention
            double VET_ATTRIBUTION_CONFLICT_SUPPRESS = 0.65; // attenuate attribution adjustment when luma and chroma claims both high
            double VET_ATTRIBUTION_BRIGHT_START_IRE  = 75.0; // begin bowing out of attribution reassignment above this luma level
            double VET_ATTRIBUTION_BRIGHT_FULL_IRE   = 85.0; // fully disable attribution reassignment by this luma level
            double VET_ATTRIBUTION_SAT_START_IRE     = 12.0; // begin bowing out when local chroma reaches this amplitude
            double VET_ATTRIBUTION_SAT_FULL_IRE      = 60.0; // fully disable attribution reassignment by this chroma amplitude

            // FVF attribution: attribution evidence adjusts field vs frame penalty before election.
            double FVF_ATTRIBUTION_LUMA_WEIGHT   = 0.14; // lumaClaim -> added to frame scores (field is safer when luma-owned)
            double FVF_ATTRIBUTION_CHROMA_WEIGHT = 0.18; // chromaClaim -> added to field scores (frame is safer when chroma coherent)

            // Locked-1D attribution damp: scales locked-1D IQ down when early luma attribution
            // evidence is strong, before tiRow/tqRow and downstream consumers receive it.
            double LOCKED1D_ATTRIBUTION_DAMP_WEIGHT = 0.25; // damp strength (0=off, 1=full suppression at lumaClaim=1)

            // Iceberg recovery: compensates for smooth-luma cancellation underestimating
            // alien-Y amplitude at directional edges.
            // 1.0 = no boost; 2.0–3.0 = moderate-to-strong recovery.
            double LUMA_ICEBERG_RECOVERY = 2.6;

        };
        Tunables tunables;
    };

    const Configuration &getConfiguration() const;
    void updateConfiguration(const LdDecodeMetaData::VideoParameters &videoParameters,
                             const Configuration &configuration);

    void decodeFrames(const QVector<SourceField> &inputFields,
                      qint32 startIndex, qint32 endIndex,
                      QVector<ComponentFrame> &componentFrames);

    static constexpr qint32 MAX_WIDTH  = 910;
    static constexpr qint32 MAX_HEIGHT = 525;

private:
    bool configurationSet;
    Configuration configuration;
    LdDecodeMetaData::VideoParameters videoParameters;

class FrameBuffer {
public:
	// FVF model/context metrics (not candidate-specific).
	// Stored per pixel so downstream consumers can understand why a line
	// segment was treated as frame-model vs field-model, edge-risk, etc.
	struct FvfModelMetrics {
		double intakeNyquistRiskIRE = 0.0;
		double chromaMagIRE = 0.0;
		double verticalBoundaryIRE = 0.0;   // horizontal gradient: scanline crosses a vertical edge
		double horizontalBoundaryIRE = 0.0; // vertical gradient: scanline runs along/grazes an edge
		double fieldFrameDivergenceIRE = 0.0;
		double interfieldDistinctIRE = 0.0;
		double frameToFieldModelIRE = 0.0;  // interlace regime: |FrameB - FieldB(model)|
		double frameToBestFieldIRE = 0.0;   // progressive regime: |FrameB - FieldB(candidate)|

		double iqFineFrac = 0.0;
		double iqMidFrac = 0.0;
		double iqCoarseFrac = 0.0;
		double chromaBandEnergyIRE = 0.0;
		double lumaIncursionRiskIRE = 0.0;
		double iqCoherence = 0.0;
		double residualFitErrorIRE = 0.0;

		bool frameModel = false;
		bool managementVeto = false;
		bool frameVertCoherent = false;
		bool vdisSoft = false;
		bool vdisHard = false;
		int winner = 1;
	};

	// Signal attribution evidence, collected before election. This is not a
	// scoring model; it records why bandpassed energy looks luma-owned,
	// chroma-owned, or contested so demod/admission can later act on it.
	using AttributionEvidence = lddecode::CombAttributionEvidence;
	using AttributionFacts = lddecode::CombAttributionFacts;
	using AttributionAssessment = lddecode::CombAttributionAssessment;
	using AttributionRules = lddecode::AttributionRules;

	FrameBuffer(const LdDecodeMetaData::VideoParameters &videoParameters_,
				const Configuration &configuration_);

	void loadFields(const SourceField &firstField, const SourceField &secondField);

	void split1D();
	void buildPhaseCorrected1D();
	void split2D();
	void copy2DTo3D();
	void split3D(const FrameBuffer &previousFrame,
				 const FrameBuffer &nextFrame);
	void measurePostCombImpurity();

	void setComponentFrame(ComponentFrame &_componentFrame) { componentFrame = &_componentFrame; }

	void splitIQ();         // Bucket
	void phaseLocked();  // prepares locked-path LO / basis etc.
	void splitIQlocked();   // Product (burst-locked)

	void filterIQ();
	void filterIQLocked();

	void adjustY();         // Bucket path
	void produceY();        // Product path
	void doCNR();
	void doYNR();
	void transformIQ(double chromaGain, double chromaPhase);
	void overlayMap(const FrameBuffer &previousFrame,
					const FrameBuffer &nextFrame);

	const std::vector<std::vector<FvfModelMetrics>> &getFvfMetrics() const { return fvfMetrics; }

	// Optional temporal context pointers used by Residual Y 3D election (set by decodeFrames)
	// Not owned — just references to neighboring FrameBuffer objects (may be nullptr).
	const FrameBuffer *prevFrameForVet = nullptr;
	const FrameBuffer *nextFrameForVet = nullptr;

	// Tracks if this frame is the start of a scene (edit boundary).
	bool isSceneStart = false;

private:
	struct Candidate { double penalty; double sample; };

	const LdDecodeMetaData::VideoParameters &videoParameters;
	const Configuration &configuration;

	qint32 frameHeight = 0;
	double irescale    = 1.0;
	double invIreScale = 1.0;

	 // Store cadence ID to inform combing decisions (e.g. FVF model)
	int cadenceId = -1; 

	SourceVideo::Data rawbuffer;
	qint32 firstFieldPhaseID  = 0;
	qint32 secondFieldPhaseID = 0;
	using LineAffine = lddecode::CarrierGrammarAffine;
	using CombCarrierGrammar = lddecode::CarrierGrammarState;
	struct SamplePlane {
		alignas(64) double pixel[MAX_HEIGHT][MAX_WIDTH];
	} clpbuffer[3];
	// clpbuffer[0]: 1D scalar plane, filled by split1D() (blind 1D bandpass).
	// clpbuffer[1]: 2D candidate plane, filled by split2D().
	// clpbuffer[2]: 3D temporal refinement plane, filled by split3D().

	struct CombTapScalar {
		double comp = 0.0;
		double symMag = 0.0;
	};

	struct CombTapPair {
		double diffIRE = std::numeric_limits<double>::infinity();
		double kScore = 0.0;
		double weight = 1.0;
		double reachLegalGate = 1.0;  // binary legality from reach index for this rung
		double reachGate = 1.0;       // contour-trust * legality
	};

	struct CombTapContour {
		double curvMidIRE = 0.0;
		double midOk = 1.0;
		double upResIRE = 0.0;
		double dnResIRE = 0.0;
		double upSideOk = 1.0;
		double dnSideOk = 1.0;
		double upSim = 0.0;
		double dnSim = 0.0;
		double upTrust = 0.0;
		double dnTrust = 0.0;
		double upInfluence = 0.0;
		double dnInfluence = 0.0;
	};

	enum FieldBDecisionReason : std::uint8_t {
		FieldBReasonNone = 0,
		FieldBReasonBlend = 1,
		FieldBReasonBoundaryUp = 2,
		FieldBReasonBoundaryDown = 3,
		FieldBReasonBoundaryCede = 4,
		FieldBReasonReviveCoarse = 5,
		FieldBReasonReviveScalar = 6,
		FieldBReasonCenter = 7
	};

	// Shared per-line harvest for the 2D combs. This centralizes row/tap/IQ
	// collection and reusable geometry facts only; each comb remains a
	// distinct consumer that applies its own model to this evidence.
	struct CombTapLine {
		int cacheLine = -1;
		int width = 0;
		unsigned builtFlags = 0;
		int ln0 = -1;
		int lnU1 = -1;
		int lnD1 = -1;
		int lnU2 = -1;
		int lnD2 = -1;
		int lnU4 = -1;
		int lnD4 = -1;
		bool haveU1 = false;
		bool haveD1 = false;
		bool haveU2 = false;
		bool haveD2 = false;
		bool haveU4 = false;
		bool haveD4 = false;
		std::vector<CombTapScalar> tap0;
		std::vector<CombTapScalar> tapU1;
		std::vector<CombTapScalar> tapD1;
		std::vector<CombTapScalar> tapU2;
		std::vector<CombTapScalar> tapD2;
		std::vector<CombTapScalar> tapU4;
		std::vector<CombTapScalar> tapD4;
		std::vector<double> centerEnvelope;
		std::vector<double> centerChromaT;
		std::vector<CombTapPair> pairU1;
		std::vector<CombTapPair> pairD1;
		std::vector<CombTapPair> pairU2;
		std::vector<CombTapPair> pairD2;
		std::vector<CombTapContour> contour;
		std::vector<CombContentReach::MovingCoarseContour> movingCoarseContour;
		std::vector<double> coarse0IRE;
		std::vector<double> coarseU2IRE;
		std::vector<double> coarseD2IRE;
		std::vector<double> hLumaDeltaIRE;
	};
	enum CombTapBuild : unsigned {
		TapBuildFieldB = 1u << 0, // center + +/-2, pair metrics, horizontal luma delta
		TapBuildFieldA = 1u << 1, // center + +/-2/+/-4 scalar contour facts
		TapBuildFrame  = 1u << 2, // center + +/-1 for scalar frame comb
		TapBuildAll    = TapBuildFieldB | TapBuildFieldA | TapBuildFrame
	};

	void buildCompositeLumaDecompositionLine(const quint16 *rawLine,
											 int left,
											 int width,
											 double *baseY4,
											 double *hiRaw,
											 double *lumaSmooth) const;
	ComponentFrame *componentFrame = nullptr;
	std::vector<CombCarrierGrammar> carrierGrammar;
	lddecode::CombReachIndex combReachIndex;

	// Flat/contiguous buffers (lines x width)
	// Line-local locked IQ after burst alignment and affine trim.
	std::vector<float> demodTI_flat;
	std::vector<float> demodTQ_flat;
	// Common 4fsc IQ export derived from the locked IQ. This is the seam
	// between the line-local locked domain and the cross-line 4fsc domain.
	std::vector<float> demodTI4fsc_flat;
	std::vector<float> demodTQ4fsc_flat;
	// Precomputed IQ magnitude: hypot(demodTI4fsc, demodTQ4fsc) per pixel.
	std::vector<float> demodIQMag4fsc_flat;
	// Preserved 4fsc IQ from the locked 1D demod.
	std::vector<float> locked1DTI4fsc_flat;
	std::vector<float> locked1DTQ4fsc_flat;
	// Product-scaled locked IQ prepared by splitIQLocked() for the output FIR.
	// Later stages may refine this cache, but they should not overwrite demodTI/TQ.
	std::vector<float> lockedProductI_flat;
	std::vector<float> lockedProductQ_flat;
	// Composite carrier estimate emitted by splitIQlocked() (the aligned carrier
	// remodulated to composite). produceY subtracts this from raw to form Y, so
	// the carrier removed from luma is exactly the carrier rendered as colour.
	std::vector<double> lockedCarrierComposite_flat;
	// Wide-window cross-color risk [0,1] per pixel.  Initially written by
	// buildPhaseCorrected1D() on the 1D bandpass, then overwritten by
	// measurePostCombImpurity() on the elected comb source so the value
	// reflects post-comb carrier, not per-field 1D.
	std::vector<float> carrierImpurity_flat;
	std::vector<double> scratch_lumaBaseY4;
	std::vector<double> scratch_lumaHiRaw;
	std::vector<double> scratch_lumaSmooth;
	// 3-slot ring buffers caching Field B output as shared preclean input.
	// Frame A and Frame B only need the center line and adjacent frame lines.
	std::array<std::vector<double>, 3> precleanRing;
	std::array<std::vector<double>, 3> precleanGateRing;
	std::array<int, 3> precleanRingLine = { -1, -1, -1 };
		std::vector<double> scratch_frameAAdaptiveIQComposite; // Frame A IQ candidate remodulated to composite.
		std::vector<double> scratch_frameBDirectIQComposite;   // Frame B IQ candidate remodulated to composite.
		std::vector<double> scratch_preI;          // Unscaled pre-FIR I row (per-line).
		std::vector<double> scratch_preQ;          // Unscaled pre-FIR Q row (per-line).
		std::vector<double> scratch_preI_ext;      // Edge-extended I row for FIR.
		std::vector<double> scratch_preQ_ext;      // Edge-extended Q row for FIR.
        // Per-line HP-Y and predictor demod scratch (used by splitIQlocked leakage cancellation)
		std::vector<double> scratch_yhp;   // simple HP of Y (per-line)
		std::vector<double> scratch_yI;    // demodulated HP-Y I component (post-affine)
		std::vector<double> scratch_yQ;    // demodulated HP-Y Q component (post-affine)
		std::vector<double> scratch_outMixed; // FVF elected mixed scalar output row.
		std::vector<double> scratch_lateralLine; // 1D lateral reference row / vet confidence row.
		std::vector<std::vector<float>> w2d_frame_weight;
		std::vector<std::vector<double>> w2d_fieldA_gate;
		std::vector<std::vector<FvfModelMetrics>> fvfMetrics;
	std::vector<std::complex<double>> scratch_iq; // reused per-line I/Q scratch (phase-corrected 1D)
	std::vector<std::complex<double>> scratch_centerIQ; // reused per-line preclean/locked frame IQ prep
	std::vector<std::complex<double>> scratch_upIQ;
	std::vector<std::complex<double>> scratch_dnIQ;
		// Shared line scratch planes used by split2D / produceY.
		std::vector<double> scratch_lineWorkA; // Field A scalar row; tiAdjLocked in produceY.
		std::vector<double> scratch_lineWorkB; // Field gate row; coherent carrier estimate (cHat) in produceY.
		std::vector<double> scratch_lineWorkC; // Field B scalar row; tqAdjLocked in produceY.
	// FVF divergence cluster: per-pixel condition DATA, pooled once per line and
	// consumed by the election (central management). Pure information — no
	// decisions live here; the consumer acts. Per-line now; promote to a 3-deep
	// ring when a 2D consumer needs the ±1 neighbours.
	struct CombConditionEvidence {
		double satIRE          = 0.0;  // IQ magnitude (saturation test)
		double combDivergence  = 0.0;  // |fieldB-frameB| smoothed (IQ/comb domain)
		double lumaDivergence  = 0.0;  // cached-luma interfield difference
		double contourNonLocal = 0.0;  // tap-contour: observation is more than local
	};
	std::vector<CombConditionEvidence> scratch_fvf_evidence;

	// FVF per-line scratch (avoid per-line allocations in scoreFieldVsFrame)
	std::vector<int>    scratch_fvf_winner;
	std::vector<int>    scratch_fvf_winner2;
	std::vector<double> scratch_fvf_outVal;
	std::vector<float>  scratch_fvf_outShade;
	std::vector<double> scratch_fvf_diffFVF;
	std::vector<double> scratch_fvf_satMap;
	std::vector<double> scratch_fvf_iqMag;     // per-line IQ magnitude pre-pass (scoreFieldVsFrame)
	std::vector<double> scratch_fvf_notchFieldA; // per-line Field A notch-luma pre-pass
	std::vector<double> scratch_fvf_notchFieldB; // per-line Field B notch-luma pre-pass
	std::vector<double> scratch_fvf_notchFrame;  // per-line Frame B notch-luma pre-pass
	std::vector<double> scratch_fvf_notchSource; // per-line source notch-luma pre-pass
	std::vector<double> scratch_coe_coherence;  // per-line IQ coherence pre-pass (collectCombAttributionEvidence)
	std::vector<double> scratch_coe_frameIQMag; // pre-computed |frameIQ[r]| magnitudes (collectCombAttributionEvidence)
		std::vector<double> scratch_lineWorkD; // Generic per-line filter scratch.
	std::vector<double> scratch_hpI;
	std::vector<double> scratch_hpQ;
	std::vector<double> scratch_hpY;
	std::vector<double> scratch_attrWideCarrier;
	std::vector<double> scratch_attrBandYClaim;
	std::vector<double> scratch_attrMembershipY;
	std::vector<char> scratch_vdis_flag;
	std::vector<std::vector<char>> vdisMask; // [line][rel], persistent per frame
	std::vector<std::uint8_t> fieldBDecisionReason_flat;
		// Flat per-sample locked-path buffers (line-major: demodLines x demodWidth).
		//
		// WARNING: locked1DSource_flat is a scalar that lives under the
		// LockedCommonPhaseScalar reach-source label.  That label has been a
		// repeated source of bugs and confusion:
		//
		//  - The label declares a *cross-line interpretation convention*
		//    (Grid4fsc / common phase), not a physical sign-stripping.  The
		//    stored scalar is approximately bandpass(raw) * 0.994, so within a
		//    line it preserves raw carrier orientation.
		//  - But a per-sample DEMOD of this scalar with an unsigned sample
		//    class does NOT yield Grid4fscIQ — it yields IQ that inherits raw
		//    signs.  For interfield (±1) cancels in IQ that means real chroma
		//    appears anti-phased, the combine flips it, and alien Y is
		//    preserved instead of cancelled.  This produced the 2fsc luma
		//    checker in Frame B's locked path.
		//  - The "polarity is gone by construction" framing in the buffer-flow
		//    doc has misled multiple agents (and the author) into either
		//    distrusting legitimate intrafield scalar combs or trusting
		//    interfield IQ combs that were actually wrong.
		//
		// Rules for new code:
		//  1. Prefer locked1DTI4fsc/TQ4fsc (Grid4fscIQ, physicalPolarityPreserved)
		//     for any operation that needs polarity.
		//  2. If a scalar must be demodded for interfield IQ use, demod with
		//     carrierGrammarSignedSampleClass (lineFlip folded into the phase)
		//     to land in Grid4fscIQ.  Unsigned demod here is almost always wrong.
		//  3. Intrafield (±2 same-field) scalar combs on this buffer are
		//     legitimate: same-field neighbors share lineFlip.
		//  4. Cross-line scalar averaging or magnitude compare is the only
		//     reach use the type system blesses for this source.  That is
		//     intentional and minimal — do not widen it without scrutiny.
		std::vector<double> locked1DSource_flat;
		std::vector<AttributionEvidence> attributionEvidence_flat; // Attribution facts/assessment per sample.
	std::vector<double> lockedLumaBaseY4_flat;
	std::vector<double> lockedLumaSmooth_flat;
	bool lockedLumaCacheValid = false;

	inline double *lockedLumaBaseY4_line(int line) {
		return lockedLumaBaseY4_flat.data() + size_t(line) * demodWidth;
	}
	
	inline double *lockedLumaSmooth_line(int line) {
		return lockedLumaSmooth_flat.data() + size_t(line) * demodWidth;
	}
	
	inline const double *lockedLumaBaseY4_line(int line) const {
		return lockedLumaBaseY4_flat.data() + size_t(line) * demodWidth;
	}
	
	inline const double *lockedLumaSmooth_line(int line) const {
		return lockedLumaSmooth_flat.data() + size_t(line) * demodWidth;
	}

	inline double *locked1DSource_line(int line) {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    locked1DSource_flat.empty()) return nullptr;
		return locked1DSource_flat.data() + static_cast<size_t>(line) * demodWidth;
	}

	inline const double *locked1DSource_line(int line) const {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    locked1DSource_flat.empty()) return nullptr;
		return locked1DSource_flat.data() + static_cast<size_t>(line) * demodWidth;
	}

	// Bucket-path 1D scalar: valid only after split1D().
	inline const double *bucketScalar1D_line(int line) const {
		return clpbuffer[0].pixel[line];
	}

	inline AttributionEvidence *attributionEvidence_line(int line) {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    attributionEvidence_flat.empty()) return nullptr;
		return attributionEvidence_flat.data() + static_cast<size_t>(line) * demodWidth;
	}

	inline const AttributionEvidence *attributionEvidence_line(int line) const {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    attributionEvidence_flat.empty()) return nullptr;
		return attributionEvidence_flat.data() + static_cast<size_t>(line) * demodWidth;
	}

	inline std::uint8_t *fieldBDecisionReason_line(int line) {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    fieldBDecisionReason_flat.empty()) return nullptr;
		return fieldBDecisionReason_flat.data() + static_cast<size_t>(line) * demodWidth;
	}

	inline const std::uint8_t *fieldBDecisionReason_line(int line) const {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    fieldBDecisionReason_flat.empty()) return nullptr;
		return fieldBDecisionReason_flat.data() + static_cast<size_t>(line) * demodWidth;
	}

	inline CombCarrierGrammar *carrierGrammarLine(int line) {
		return (line >= 0 && line < static_cast<int>(carrierGrammar.size()))
			? &carrierGrammar[line]
			: nullptr;
	}

	inline const CombCarrierGrammar *carrierGrammarLine(int line) const {
		return (line >= 0 && line < static_cast<int>(carrierGrammar.size()))
			? &carrierGrammar[line]
			: nullptr;
	}

	inline int carrierLineFlip(int line) const {
		const CombCarrierGrammar *grammar = carrierGrammarLine(line);
		return grammar ? grammar->lineFlip : 1;
	}

	inline bool carrierFrameVerticalAllowed(int line) const {
		const CombCarrierGrammar *grammar = carrierGrammarLine(line);
		return grammar ? grammar->frameVerticalAllowed : true;
	}

	inline int carrierLineParity(int line) const {
		const CombCarrierGrammar *grammar = carrierGrammarLine(line);
		return grammar ? grammar->lineParity : (line & 1);
	}

	inline int carrierSampleClass(int line, int h) const {
		return lddecode::carrierGrammarSampleClass(carrierGrammarLine(line), h);
	}

	inline int carrierSignedSampleClass(int line, int h) const {
		return lddecode::carrierGrammarSignedSampleClass(carrierGrammarLine(line), h);
	}

	inline int carrierOppositeSampleClass(int line, int h) const {
		return lddecode::carrierGrammarOppositeSampleClass(carrierGrammarLine(line), h);
	}

	inline double carrierPlausibility(const CombCarrierGrammar *grammar) const {
		if (!configuration.phaseCompensation)
			return 1.0;
		if (!grammar || !grammar->grammarLocked)
			return 0.0;
		double base = grammar->projectionValid
			? std::clamp(grammar->carrierFitRatio, 0.0, 1.0)
			: std::clamp(grammar->phaseConfidence, 0.0, 1.0);
		const double conflictPenalty =
			1.0 - 0.5 * std::clamp(grammar->phaseScheduleConflict, 0.0, 1.0);
		return base * conflictPenalty;
	}

	inline double remodUnsignedBucketToComposite(int line, int h,
	                                             double I, double Q) const {
		return lddecode::carrierGrammarRemod4fscToComposite(
			carrierGrammarLine(line),
			h,
			I,
			Q,
			1.0,
			lddecode::CarrierSignFrame::UnsignedBucket);
	}

	inline double remodGrid4fscToComposite(int line, int h,
	                                       double I, double Q) const {
		return lddecode::carrierGrammarRemod4fscToComposite(
			carrierGrammarLine(line),
			h,
			I,
			Q,
			1.0,
			lddecode::CarrierSignFrame::Grid4fsc);
	}

	inline double remodGrammarToComposite(int line, int h,
	                                      double I, double Q) const {
		return remodUnsignedBucketToComposite(line, h, I, Q);
	}

	inline lddecode::CombReachSourceFrame scalarReachSource() const {
		return configuration.phaseCompensation
			? lddecode::makeLockedCommonPhaseScalarReachSource()
			: lddecode::makeBucketScalarReachSource();
	}

	inline lddecode::CombReachSourceFrame iqReachSource() const {
		return lddecode::makeGrid4fscIQReachSource();
	}
		
	CombTapLine scratch_tapLine;
	std::array<CombTapLine, 3> tapLineCache;
	std::array<int, 3> tapLineCacheLine = { -1, -1, -1 };

	unsigned combTapBuildFlags_ = TapBuildAll;

	inline int precleanRingSlot(int lineNumber) const
	{
		int s = lineNumber % 3;
		if (s < 0) s += 3;
		return s;
	}

	inline bool havePrecleanLine(int lineNumber, int width) const
	{
		const int s = precleanRingSlot(lineNumber);
		if (precleanRingLine[s] != lineNumber) return false;
		return ((int)precleanRing[s].size() >= width);
	}

	inline const double *precleanLinePtr(int lineNumber, int width) const
	{
		if (!havePrecleanLine(lineNumber, width)) return nullptr;
		const int s = precleanRingSlot(lineNumber);
		return precleanRing[s].data();
	}

	inline const double *precleanGateLinePtr(int lineNumber, int width) const
	{
		if (!havePrecleanLine(lineNumber, width)) return nullptr;
		const int s = precleanRingSlot(lineNumber);
		if ((int)precleanGateRing[s].size() < width) return nullptr;
		return precleanGateRing[s].data();
	}

		inline double *precleanLinePtrMutable(int lineNumber, int width)
		{
			const int s = precleanRingSlot(lineNumber);
			if ((int)precleanRing[s].size() < width) precleanRing[s].resize(width);
			precleanRingLine[s] = lineNumber;
			return precleanRing[s].data();
		}

		inline double *precleanGateLinePtrMutable(int lineNumber, int width)
		{
			const int s = precleanRingSlot(lineNumber);
			if ((int)precleanGateRing[s].size() < width) precleanGateRing[s].resize(width);
			return precleanGateRing[s].data();
		}

	// Small helpers declared here; definitions provided after the class (in this header).
	// Per-run 4fsc shifted basis LUT for locked path (phaseCompensation=true)
	double spLUT_locked[4] = {1.0, 0.0, -1.0, 0.0};
	double cpLUT_locked[4] = {0.0, 1.0,  0.0, -1.0};
	bool   basisLockedInit = false;
	bool hasVDIS(int lineNumber, int h) const;      

	// Hybrid 2D helpers
	void invalidateCombTapCache();
	const CombTapLine &ensureCombTapLine(int lineNumber);
	void buildCombTapLine(int lineNumber, CombTapLine &tapLine);
	void computeContourFieldLine(int lineNumber,
						  double *outFieldLine,
						  double  *outGate);
	void computeContourFieldLine(const CombTapLine &tapLine,
						  double *outFieldLine,
						  double  *outGate);

	void computeSimpleFieldLine(int lineNumber,
						  double *outFieldLine,
						  std::uint8_t *outReasonLine = nullptr);
	void computeSimpleFieldLine(const CombTapLine &tapLine,
						  double *outFieldLine,
						  std::uint8_t *outReasonLine = nullptr);

	void computeFrameAAdaptiveIQLine(int line,
									std::vector<std::complex<double>> &outFrameIQ);
	void computeFrameBDirectIQCompositeLine(int line,
								   std::vector<std::complex<double>> &outFrameIQ,
								   std::vector<double> &outFrameScalar);
	void computeFrameBDirectIQLine(int line,
									std::vector<std::complex<double>> &outFrameIQ,
									const std::vector<float> *tiOverride = nullptr,
									const std::vector<float> *tqOverride = nullptr);
	void computeFrameBDirectIQFromPreparedVectors(
	    int line,
	    const std::vector<std::complex<double>> &centerIQ,
	    const std::vector<std::complex<double>> &upIQ,
	    const std::vector<std::complex<double>> &dnIQ,
	    std::vector<std::complex<double>> &outFrameIQ,
	    const CombTapLine *reachTapLine = nullptr);

	void computeFrameIQFromPreparedVectors(int line,
											   const std::vector<std::complex<double>> &centerIQ,
											   std::vector<std::complex<double>> &upIQ,
											   std::vector<std::complex<double>> &dnIQ,
											   std::vector<std::complex<double>> &outFrameIQ,
											   const std::vector<float> *tiOverride,
											   const std::vector<float> *tqOverride,
											   const CombTapLine *reachTapLine = nullptr,
											   bool allowSymmetricLeakCancel = false);
	void collectCombAttributionEvidence(int line,
									   const double *fieldA,
									   const double *fieldB,
									   const std::vector<double> &frameScalar,
									   const std::vector<std::complex<double>> *frameIQ);
	void seedCombAttributionPerLine(int line);
	void finalizeAttributionClaims(AttributionEvidence &e,
								 double neighborLumaMeanIRE = -1.0,
								 double neighborBaseMeanIRE = -1.0,
								 double lineForwardErrorIRE = 0.0) const;
	// Unified VDIS map builder: combines scalar (±2) and IQ (±1) evidence
	// into scratch_vdis_flag for a given line. Does not modify FieldA/Frame.
	void computeVDISLine(int lineNumber);
	static void consolidateVDISRegions(std::vector<std::vector<char>> &mask,
									   const LdDecodeMetaData::VideoParameters &vp);
	
	// Field-vs-Frame elects scalar bandpass candidates. IQ evidence can inform
	// the scores, but IQ-derived candidates must be remodulated before entry.
	void scoreFieldVsFrame(
	    int line,
	    const CombTapLine &tapLine,
	    const double *fieldB,
	    const std::vector<double> &fieldA,
	    const std::vector<double> *frameB,
	    double *outMixed,
	    bool writeWeights,
	    const double *lateral1D,
	    const std::vector<std::complex<double>> *frameIQ = nullptr);
	    
	static inline bool fvf_is_tri_safe(double candVal,
									   double L1,
									   double invIreScale,
									   double triSafeIre)
	{
		const double dCand1D_ire = std::fabs(candVal - L1) * invIreScale;
		return (dCand1D_ire <= triSafeIre);
	}

	static inline double getNotchLumaEven2(const double* arr, int rel, int width)
	{
		if (!arr || width <= 0) return 0.0;
		if (rel < 2) rel = 2;
		if (rel > width - 3) rel = width - 3;
		return 0.5 * (arr[rel - 2] + arr[rel + 2]);
	}

	static inline double getNotchLumaEven2Vec(const std::vector<double>& vec, int rel)
	{
		const int width = (int)vec.size();
		return (width > 0) ? getNotchLumaEven2(vec.data(), rel, width) : 0.0;
	}
		
	void getBestCandidate(qint32 lineNumber, qint32 h,
						  const FrameBuffer &previousFrame,
						  const FrameBuffer &nextFrame,
						  qint32 &bestIndex, double &bestSample) const;

	// getCandidate is declared here and implemented in comb_candidate.cpp
	Candidate getCandidate(qint32 refLineNumber, qint32 refH,
						   const FrameBuffer &frameBuffer,
						   qint32 lineNumber, qint32 h,
						   double adjustPenalty) const;

	// Dedicated 3D Residual Y selector
	double getBestY(qint32 line, qint32 h, double currentY2D, 
					const FrameBuffer &prev, const FrameBuffer &next) const;        
	
	int demodWidth  = 0;
	int demodLines  = 0;

	inline float* demodTI_line(int line) {
		return demodTI_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline float* demodTQ_line(int line) {
		return demodTQ_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline float* demodTI4fsc_line(int line) {
		return demodTI4fsc_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline float* demodTQ4fsc_line(int line) {
		return demodTQ4fsc_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline const float* demodTI_line(int line) const {
		return demodTI_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline const float* demodTQ_line(int line) const {
		return demodTQ_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline const float* demodTI4fsc_line(int line) const {
		return demodTI4fsc_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline const float* demodTQ4fsc_line(int line) const {
		return demodTQ4fsc_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline const float* demodIQMag4fsc_line(int line) const {
		if (demodIQMag4fsc_flat.empty()) return nullptr;
		return demodIQMag4fsc_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline float* locked1DTI4fsc_line(int line) {
		return locked1DTI4fsc_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline float* locked1DTQ4fsc_line(int line) {
		return locked1DTQ4fsc_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline const float* locked1DTI4fsc_line(int line) const {
		return locked1DTI4fsc_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline const float* locked1DTQ4fsc_line(int line) const {
		return locked1DTQ4fsc_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline float* lockedProductI_line(int line) {
		return lockedProductI_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline float* lockedProductQ_line(int line) {
		return lockedProductQ_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline const float* lockedProductI_line(int line) const {
		return lockedProductI_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline const float* lockedProductQ_line(int line) const {
		return lockedProductQ_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline double* lockedCarrierComposite_line(int line) {
		if (lockedCarrierComposite_flat.empty()) return nullptr;
		return lockedCarrierComposite_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline const double* lockedCarrierComposite_line(int line) const {
		if (lockedCarrierComposite_flat.empty()) return nullptr;
		return lockedCarrierComposite_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline float* carrierImpurity_line(int line) {
		return carrierImpurity_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline const float* carrierImpurity_line(int line) const {
		return carrierImpurity_flat.data() + static_cast<size_t>(line) * demodWidth;
	}

	// Vet result container (used by locked-path coherent Y rebuild).
	struct Vet1DResult {
		double composite_bandpass = 0.0;        // raw - 2D clp (IRE units)
		double leftScore         = std::numeric_limits<double>::infinity();  // smaller is better
		double rightScore        = std::numeric_limits<double>::infinity();  // smaller is better
		int    bestIndex         = -1;         //  -1 = none, 0 = left, 1 = right
		double bestScore         = std::numeric_limits<double>::infinity();
		double confidence        = 0.0;        // 0..1 confidence that replacement is safe
		bool   accept            = false;      // true => safe to apply composite substitution
		int    verticalAgree     = 0;          // number of vertical neighbors that agree (0..2)
		// diagnostics for adjacent neighbor influence
		int    adjNeighborCount  = 0;          // number of valid immediate neighbors (0..2)
		double adjNeighborSupport= 0.0;        // average agreement of h±1 with residual (0..1)
	};
};
// Inline definitions for FrameBuffer (out-of-class)
};

#endif // COMB_H
