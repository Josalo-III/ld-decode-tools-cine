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
#include <memory>
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
        bool   debugInterfieldFlip = false; // Log per-leg benefit of carrier alignment on ±1 interfield pair
        // Demod plus Y selection: phase locked vs bucket
        // Phase locked is a coherent path that includes HF Y from composite
        bool phaseCompensation = false;
        bool lumaWitness = false;
        enum class DiagnosticOutput {
            None,
            CarrierFit,
            CarrierRetracted
        };
        DiagnosticOutput diagnosticOutput = DiagnosticOutput::None;

        // Per-axis product gains: multipliers applied to locked I and Q before
        // filtering.  This is the sole authority for product tuning — do not
        // reintroduce gain constants in combmath.h.  GQ < 1.0 trims the Q axis
        // to compensate for the slight chroma ellipse of the locked demod.
        // (Defaults match the previously live combmath.h values; the old 1.1
        // here was never read.)
        double gi_product = 1.0;
        double gq_product = 0.9;

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
        bool diagnosticOnly() const {
            return diagnosticOutput != DiagnosticOutput::None;
        }
        struct Tunables {
            // =========================================================================
            // 1D / Lateral baseline
            // =========================================================================
            // Cross-color suppression strength (--cross-color-return).
			// Set in CLI to transfer cross color back to luma.  Scales how
			// much of the MEASURED contamination gets returned to luma; the
			// transfer is hard-ceilinged at the evidence itself (aperture gA,
			// or the vertical-image-detail read where gA under-reads -- see
			// splitIQlocked/filterIQLocked).  Values above 1.0 only override
			// a false-positive regionKeep rescue; they can never manufacture
			// suppression beyond what was measured, so saturated textured
			// chroma no longer grays out at strength 2. The feature is
			// opt-in; 0 leaves the ordinary residual-Y path untouched.
            double CC_SUPPRESSION_WEIGHT    = 0.0;

            // =========================================================================
            // 2D Field extraction (FieldA/B) and vertical gating
            // =========================================================================
            double FIELD_K_RANGE_IRE           = 45.0; // max kScore diff (IRE) before field 2D output is penalized
            double FIELD_CONTOUR_FAR_INFLUENCE = 0.55; // weight of ±4-line neighbors relative to ±2 in contour detection
            double FIELD_CONTOUR_SOFT_IRE      = 4.0;  // contour curvature below this → no edge penalty applied
            double FIELD_CONTOUR_HARD_IRE      = 10.0; // contour curvature above this → full edge penalty applied
            double FIELD_CONTOUR_SIM_START     = 0.55; // vertical chroma similarity below which contour gate begins to open
            double FIELD_CONTOUR_SIM_FULL      = 0.85; // similarity above which contour gate is fully closed
            double FIELD_A_BEVEL_CEDE_STRENGTH = 0.15; // Field A's own center cede in high-chroma bevels near luma edges

            double FIELD_VERT_DISAGREE_THRESH_IRE = 8.0; // suppress 2D field output when ±2 line pair disagrees beyond this

            double FIELD_LUMA_EDGE_THRESH_IRE = 18.0; // horizontal luma gradient above this suppresses vertical 2D comb
            double FIELD_B_BEVEL_REACH_PENALTY = 0.45; // RESERVED / inert: retired Field B bevel reach damping; kept for tuning compatibility

            // =========================================================================
            // Frame comb on phase-corrected 1D
            // =========================================================================
            double FRAME_COMB_STRENGTH        = 1.125; // interframe cancellation amplitude scale for Frame A (>1 boosts cancellation)
            double FRAME_CHROMA_MIN_IRE       = 1.5;   // Frame A minimum chroma amplitude to engage the frame IQ path
            double FRAME_IQ_RAW_MAX_DELTA_IRE = 70.0;  // Frame A max IQ mismatch between locked-1D and frame average before frame IQ is distrusted
            double FRAME_IQ_COH_PASS_CORR     = 0.85;  // Frame A signed center/neighbor correlation at which cohGate fully passes (firm comb); ramp starts 0.30 below
            double FRAME_B_COMB_STRENGTH       = 1.00; // Frame B ±1: fraction of the EXACT projection (pull = 0.5 * strength * reachAuthority). 1.0 = the [1,2,1] solution of the two-line alternation model (alien nulled, Y luma at unit gain). Values are capped at the projection: pull > 0.5 is not stronger cancellation, it re-injects inverted alien at (2p−1) — the 0.80-era overdrive that serrated diagonals and manufactured diagonal cross-color.
            double FRAME_B_CHROMA_MIN_IRE      = 1.5;  // Frame B IRE-domain reach-floor minimum
            double FRAME_B_RAW_MAX_DELTA_IRE   = 100.0; // Frame B IRE-domain direct-IQ delta cap; flat, no lane-specific relax — the second line of defence behind the reach throttles
            double FRAME_B_BEVEL_REACH_PENALTY = 1.0;  // chroma-weighted bevel reach throttle on Frame B ±1; gates near a horizontal luma step where the ±1 partners straddle different bevel phases (zipper guard)
            double FRAME_BEVEL_SAT_PENALTY     = 0.50; // extra reach penalty at saturated non-straight edges; squared chroma tightening on the bevel gate (0 = off, 1 = aggressive)
            double FRAME_LUMA_EDGE_THRESH_IRE  = 28.0; // horizontal luma gradient for Frame ±1 cross-color gate; higher than Field's 18 because ±1 partners are closer (one TV line) and more resilient
            double FRAME_BEVEL_XCOL_PENALTY    = 1.0;  // RESERVED / inert: the gate-side lateral-edge term was removed — a bare hLumaDeltaIRE step cannot separate a straight vertical misread column from a diagonal boundary, so it only stripped bevel protection off diagonals. Vertical-column restore now lives in the combine's partner-verified crossColorExempt (computeFrameBLine). Kept for ABI/tuning; not read by the reach gate.

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
            // Let d be the mean absolute reconstructed-luma difference over
            // the centered five-point spatial cross in getCandidate():
            //   d ≤ AGREEMENT_REWARD_RADIUS_IRE  → reward: −AGREEMENT_REWARD_MAX·(1−(d/r)²)
            //   d > AGREEMENT_REWARD_RADIUS_IRE  → neutral
            // Output safety is enforced by the independent-estimate hull at
            // the split3D write site, not by penalizing disagreement here.
            double AGREEMENT_REWARD_RADIUS_IRE = 7.5; // half-width of the reward lobe (IRE)
            double AGREEMENT_REWARD_MAX        = 3.3; // peak reward at d=0 (penalty units, scaled by adaptThreshold)
            double AGREEMENT_VETO_BASE         = 7.0; // RETIRED / inert: retained for tuning compatibility
            double deviationPenalty            = 3.3; // RETIRED / inert: retained for tuning compatibility
            double TEMPORAL_HULL_SLACK_IRE      = 1.5; // output may exceed the independent 2D/partner hull by this much

            // produceY local HF election.  A candidate earns this capped
            // advantage only when its HF magnitude is continued by a robust
            // N/S/E/W image neighbour; unsupported amplitude earns nothing.
            double PRODUCE_Y_HF_IMAGE_PREFERENCE_IRE = 1.5;
            double PRODUCE_Y_HF_CONTINUATION_IRE = 6.0;
            // Explicit policy limit on score advantage earned by actual
            // carrier reduction under measured cross-colour. Raw and smoothed
            // detector values remain separately observable in diagnostics.
            double PRODUCE_Y_CC_RETURN_EVIDENCE_CAP_IRE = 3.0;
            // Carrier-basis phase is only a hygiene/tie-break term in produceY;
            // luma image continuity owns the election.
            double PRODUCE_Y_PHASE_PENALTY_IRE = 0.75;
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
	// Single pre-render owner of the canonical locked 1D baseline and shared
	// carrier diagnostics. Application paths consume these records; they do not
	// clear or privately recreate them.
	// prevFrame: the comb pipeline's temporally previous FrameBuffer (its
	// grammar + harvested bandpass supply the frame-axis conformance test).
	// This is the comb's own 3D structure. Null when no contiguous predecessor.
	void buildCarrierAnalysis(FrameBuffer *prevFrame = nullptr);
	void buildCarrierRetractionStage(bool analysisOnly);

	// Corner-leak corrector. The locked bandpass reads luma CURVATURE as false
	// carrier: leak[x] = -0.25 * (Y[x-2] - 2Y[x] + Y[x+2]), exactly. A
	// constant-slope ramp has zero curvature and therefore leaks nothing, so
	// this stage is identically inert wherever the luma foundation does not
	// bend. It recovers the curvature by deconvolving the KNOWN notch kernel
	// ([0.25,0.5,0.25] at stride 2), gates the result by carrier-free evidence,
	// and publishes the predicted leak. DIAGNOSTIC ONLY at present: nothing
	// consumes lockedCornerLeak_flat, so the render is unchanged.
	void buildCornerLeak();

	// Carrier-retraction front end, run after shared analysis and the locked
	// local-carrier construction.
	void buildCarrierRetracted();
	void outputDiagnosticFrame();

	void lurchSharpenCoarsePrior(const double *means,
	                             int meanCount,
	                             int width,
	                             double *prior,
	                             double *gateOut,
	                             double gateGain = 1.0) const;
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

	// Tracks if this frame is the start of a scene (edit boundary).
	bool isSceneStart = false;

	// Identity of the frame this buffer currently holds, as the seqNos of the
	// two fields it was loaded from.  The triple-buffer persists across
	// decodeFrames() calls, so a buffer's contents outlive the call that
	// produced them; without an identity there is no way to tell a frame that
	// is still wanted from a stale recycled one.
	//
	// Negative means "holds nothing usable as temporal context": either never
	// loaded, explicitly invalidated after a skipped load, or loaded from the
	// pool's synthetic black padding (which carries negative seqNos by
	// construction).  Combing against any of those subtracts something that is
	// not the neighbouring picture.
	qint32 heldSeq1 = -1;
	qint32 heldSeq2 = -1;

	bool holdsRealFrame() const { return heldSeq1 >= 0 && heldSeq2 >= 0; }

	bool holds(const SourceField &a, const SourceField &b) const {
		return heldSeq1 == a.field.seqNo && heldSeq2 == b.field.seqNo;
	}

	void forgetHeldFrame() { heldSeq1 = -1; heldSeq2 = -1; }

private:
	struct Candidate {
	    double penalty;
	    double sample;
	    double yPen;
	    double iqPen;
	};
	struct TemporalCandidateSamples {
		struct Sample {
			double value = 0.0;
			bool valid = false;
		};
		Sample previousField;
		Sample nextField;
		Sample previousFrame;
		Sample nextFrame;
	};
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
		double raw = 0.0;
		double comp = 0.0;
		double symMag = 0.0;
	};

	struct CombTapPair {
		double diffIRE = std::numeric_limits<double>::infinity();
		double kScore = 0.0;
		double weight = 1.0;
		double reachLegalGate = 1.0;  // physical legality from grammar/source contract
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
			FieldBReasonCenter = 2,
			FieldBReasonCede = 3,   // explicit policy centerCede dominated the output
			FieldBReasonOneLeg = 4, // one-sided comb (a leg was excluded by policy)
			FieldBReasonRecovery = 5, // cede overridden by physically valid ±2 recovery
			FieldBReasonRepairHold = 6, // certified repaired center retained against recombing
			FieldBReasonCount = 7
		};

	// Shared per-line harvest for the 2D combs. This centralizes row/tap/IQ
	// collection and reusable geometry facts only; each comb remains a
	// distinct consumer that applies its own model to this evidence.
	struct CombTapLine {
		int cacheLine = -1;
		int width = 0;
		unsigned builtFlags = 0;
		int ln0 = -1;
		
		// Requested geometric taps.  These preserve what the comb asked for before
		// active-boundary resolution.
		int reqU1 = -1;
		int reqD1 = -1;
		int reqU2 = -1;
		int reqD2 = -1;
		int reqU4 = -1;
		int reqD4 = -1;
		
		// Resolved readable taps.  For same-field taps at the active top/bottom
		// boundary, a missing outward leg may intentionally resolve to the inward
		// same-field partner.  That is mirror resolution, not clamp fallback.
		int lnU1 = -1;
		int lnD1 = -1;
		int lnU2 = -1;
		int lnD2 = -1;
		int lnU4 = -1;
		int lnD4 = -1;
		
		// Resolved readability: true means the corresponding ln* can be read.
		bool haveU1 = false;
		bool haveD1 = false;
		bool haveU2 = false;
		bool haveD2 = false;
		bool haveU4 = false;
		bool haveD4 = false;
		
		// Geometric availability: true means the requested tap itself existed in the
		// active same-field/interfield geometry before mirror resolution.
		bool geomHaveU1 = false;
		bool geomHaveD1 = false;
		bool geomHaveU2 = false;
		bool geomHaveD2 = false;
		bool geomHaveU4 = false;
		bool geomHaveD4 = false;
		
		// Active-boundary same-field mirror state.  At the top active same-field row,
		// U2 may resolve to D2; at the bottom, D2 may resolve to U2.  Same for ±4.
		// Consumers that only need a readable support row can use have*/ln*.
		// Consumers that care about original geometry can inspect geomHave*/mirrored*.
		bool mirroredU2 = false;
		bool mirroredD2 = false;
		bool mirroredU4 = false;
		bool mirroredD4 = false;

		std::vector<CombTapScalar> tap0;
		std::vector<CombTapScalar> tapU1;
		std::vector<CombTapScalar> tapD1;
		std::vector<CombTapScalar> tapU2;
		std::vector<CombTapScalar> tapD2;
		std::vector<CombTapScalar> tapU4;
			std::vector<CombTapScalar> tapD4;
			std::vector<double> centerEnvelope;
			// Carrier amplitude interpreted as chroma only after the center sample
			// has been admitted by schedule conformance.
			std::vector<double> centerAdmittedChromaT;
		std::vector<CombTapPair> pairU1;
		std::vector<CombTapPair> pairD1;
			std::vector<CombTapPair> pairU2;
			std::vector<CombTapPair> pairD2;
			std::vector<CombContentReach::IntrafieldRegionReach> intrafieldRegionReach;
		// The shared line ends at observations. Each field comb interprets the
		// region facts independently and owns any leg exclusion or center cede.
		// Per-pixel ±4 region verdicts (center vs ±4 same-field partner).
		// Evaluated alongside the ±2 intrafieldRegionReach in
		// buildCombTapLine; consumed by the contour influence gate so
		// Field A's far-tap refinement refuses reaches across hue
		// boundaries on diagonals.
		std::vector<CombContentReach::RegionRelation> regionUp4;
		std::vector<CombContentReach::RegionRelation> regionDown4;
		std::vector<CombTapContour> contour;
		std::vector<CombContentReach::MovingCoarseContour> movingCoarseContour;
		std::vector<double> coarse0IRE;
		std::vector<double> coarseU2IRE;
		std::vector<double> coarseD2IRE;
		// True only when BOTH neighbour coarse rows carry real luma evidence
		// (a locked decomposition row, or a notch luma from a tap that exists).
		// An absent neighbour row falls back to the centre value, which reads
		// as a zero vertical delta -- i.e. "no luma break anywhere", the most
		// permissive answer there is.  Consumers that bypass a structural
		// verdict on the strength of a flat-luma context MUST consult this and
		// fail closed; vector size alone is not evidence.
		bool coarseLumaValid = false;
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
	// Line-local locked IQ after burst alignment.
	std::vector<float> demodTI_flat;
	std::vector<float> demodTQ_flat;
	// Native, sample-local 4fsc products derived from the locked scalar.
	// These retain the alternating product image and are an intermediate,
	// not the cross-line comb contract.
	std::vector<float> demodTI4fsc_flat;
	std::vector<float> demodTQ4fsc_flat;
	// Precomputed magnitude service retained for legacy local evidence.
	std::vector<float> demodIQMag4fsc_flat;
	// Canonical pre-comb IQ: separate full-scale I and Q, low-passed with a
	// symmetric integer-centred aperture and registered to locked1DSource[h].
	std::vector<float> locked1DTI4fsc_flat;
	std::vector<float> locked1DTQ4fsc_flat;
	// Per-line 7-tap smoothed signed-IQ rows, memoised for the region-reach
	// evaluator.  smoothSignedIQ(line, rel) depends only on the line and rel,
	// yet each line is referenced by up to five centers (self, +/-2, +/-4);
	// caching computes it once per line instead of once per reference.
	// Validity is cleared in invalidateCombTapCache() so the memo tracks the
	// tap cache lifecycle (i.e. is as fresh as the locked demod it reads).
	std::vector<float> smoothedLockedTI_flat;
	std::vector<float> smoothedLockedTQ_flat;
	std::vector<std::uint8_t> smoothedLockedRowValid;
	// Product-scaled locked IQ prepared by splitIQLocked() for the output FIR.
	// Later stages may refine this cache, but they should not overwrite demodTI/TQ.
	std::vector<float> lockedProductI_flat;
	std::vector<float> lockedProductQ_flat;
	// Composite carrier estimate emitted by splitIQlocked() (the aligned carrier
	// remodulated to composite). produceY subtracts this from raw to form Y, so
	// the carrier removed from luma is exactly the carrier rendered as colour.
	std::vector<double> lockedCarrierComposite_flat;
	// Render-facing cross-color impurity [0,1] per pixel. Seeded from locked 1D
	// as a provisional read; measurePostCombImpurity() overwrites it with the
	// elected-comb reading before splitIQlocked() consumes it.
	std::vector<float> carrierImpurity_flat;
	// Same-region vertical partner evidence [0,1] per pixel: 1 when two
	// schedule-admitted carrier operands positively share a ±2 chroma region
	// (relation-signed hue agreement above the amplitude floor).  Discriminates
	// a real chroma-region boundary from cross-color — both fail interline
	// verification, so gA alone cannot tell them apart.  Evidence only; the
	// suppression policy converts it at the consumption sites.
	std::vector<float> regionSamePartner_flat;
	// Cross-color suppression weight [0,1] per pixel, computed once by
	// splitIQlocked() and applied to the coherent lockedProduct chroma.
	// Raw = the per-pixel policy verdict;
	// the applied mask is its band-limited envelope (in-field ±2 vertical mix
	// + lateral boxcar): applied per-sample the raw weight carries
	// regionKeep's hard flips and gA's ring chatter at pixel pitch, which
	// amplitude-modulates the rendered chroma and beats sidebands back into
	// the passband (shredded colour on both sides of a hue boundary).  The
	// envelope may vary no faster than the chroma it gates.  Allocated only
	// when --cross-color-return is engaged.
	std::vector<float> lockedCcMaskRaw_flat;
	std::vector<float> lockedCcMask_flat;
	// Schedule-rejected (alien) vertical partner evidence [0,1] per pixel: 1
	// when a negatively conforming operand is ANTI-aligned at comparable
	// magnitude after relation signing — raw-identical content where the carrier
	// schedule demands inversion.  Legal carrier MUST invert per the lineFlip
	// schedule; energy that fails the schedule is structurally not carrier
	// and therefore luma by law (near-carrier periodic luma: the Borg-cube
	// grid).  Evidence only; consumers convert (produceY seats retractedY,
	// which keeps this energy as luma, without geometric corroboration).
	std::vector<float> regionAlienPartner_flat;

	// --- Carrier-retraction buffers. Populated by buildCarrierRetracted();
	// same geometry as the demod flats (demodLines x demodWidth).
	// carrierImpurity_flat above is shared with the current aperture detector
	// and reused here. ---
	// Per-line four-view carrier model.  Source contract:
	// CarrierFitScalar / BurstLockedSigned / PhasePreservedCarrier.  Any
	// cross-line video use must go through CombReachIndex.
	std::vector<float> carrierFit_flat;
	std::vector<float> carrierRetracted_flat;    // raw - promoted carrier model (retracted view, not final Y)
	std::vector<float> flatFloor_flat;           // 4-sample carrier-free luma floor
	std::vector<float> combedCarrier_flat;       // grammar-reached interline carrier fit
	// Envelope-scale schedule corroboration in [0,1]: how much of the
	// published fit is PROVEN carrier by relation-folded alternation
	// evidence, integrated at the aperture the encoder bandwidth law
	// defines (a legal envelope cannot vary faster than ~1.3 MHz, so
	// alternation evidence about it is only meaningful at that scale).
	// Scales the carrier the published retracted view withdraws; varies
	// no faster than a legal envelope, so the scaling manufactures no
	// out-of-band sidebands.
	std::vector<float> carrierCorroboration_flat;
	// Graded schedule-conformance participation in [0,1], from the table-owned
	// carrierTrust() on the scanner's MEASUREMENT (carrierConformance +
	// support fraction) — never the thresholded enum.  1 = full participant
	// (legal / unresolved / quiet), 0 = decisively proven schedule-illegal.
	// The old uint8 projection quantized a smooth vertical correlation to one
	// bit and made the retraction's engage/disengage flip at line pitch.
	std::vector<float> carrierEligibility_flat;
	std::vector<lddecode::FourViewPixelEvidence> coarseYEvidence_flat; // per-pixel four-view evidence
	bool carrierRetractedValid = false;
	bool carrierRetractionModelValid = false;

	std::vector<double> scratch_lumaBaseY4;
	std::vector<double> scratch_lumaHiRaw;
	std::vector<double> scratch_lumaSmooth;
	std::vector<double> scratch_envLine;
	std::vector<double> scratch_spanLine;
	// 3-slot ring buffers caching Field B output as shared preclean input.
	// Frame A and Frame B only need the center line and adjacent frame lines.
	std::array<std::vector<double>, 3> precleanRing;
	std::array<int, 3> precleanRingLine = { -1, -1, -1 };
		std::vector<double> scratch_frameAAdaptiveIQComposite; // Frame A IQ candidate remodulated to composite.
		std::vector<double> scratch_frameBDirectIQComposite;   // Frame B IQ candidate remodulated to composite.
		std::vector<double> scratch_preI;          // Unscaled pre-FIR I row (per-line).
		std::vector<double> scratch_preQ;          // Unscaled pre-FIR Q row (per-line).
		std::vector<double> scratch_preI_ext;      // Edge-extended I row for FIR.
		std::vector<double> scratch_preQ_ext;      // Edge-extended Q row for FIR.
		std::vector<double> scratch_outMixed; // FVF elected mixed scalar output row.
		std::vector<double> scratch_lateralLine; // 1D lateral/reference row.
		std::vector<std::vector<float>> w2d_frame_weight;
		std::vector<std::vector<double>> w2d_fieldA_gate;
		std::vector<std::vector<FvfModelMetrics>> fvfMetrics;
	std::vector<std::complex<double>> scratch_iq; // reused per-line I/Q scratch (phase-corrected 1D)
	std::vector<std::complex<double>> scratch_centerIQ; // reused per-line preclean/locked frame IQ prep
	std::vector<std::complex<double>> scratch_upIQ;
	std::vector<std::complex<double>> scratch_dnIQ;
	// Frame B signed-subtractor prepass (computeFrameBLine).  The ±1 pair
	// difference is a SIGNED estimator of the legs' shared image-locked alien:
	// the signed demod folds the alien with opposite signs on the Same- and
	// Opposite-relation legs, so (ZUp−ZDn)/2 recovers the aliens' SUM (≈ the
	// center's own alien, error second-order/curvature) while the midpoint
	// carries their difference (first-order on diagonals — the 2-px
	// staircase).  fbPairDiff holds the diagonal-registered, window-normalized
	// difference vector; fbAlienGate the windowed validity gate (Same leg must
	// ride with center — a real vertical chroma gradient zeroes it);
	// fbPairAgreeWinIRE the unregistered windowed pair-agreement magnitude
	// consumed by the midpoint license; fbReg the chosen registration offset
	// (diagnostic).
	std::vector<std::complex<double>> scratch_fbPairDiff;
	std::vector<double> scratch_fbAlienGate;
	std::vector<double> scratch_fbPairAgreeWinIRE;
	// Frame B leg-deviation symmetry, dSame/dOpp in [0,1].  A published FACT
	// about the ±1 pair, not a policy: 0 = asymmetric (Same leg rides centre,
	// Opposite carries ~2a -> vertically-invariant image-locked alien), 1 =
	// legs deviate together (diagonal advance or real vertical gradient).
	// Absence of evidence reports 1, never 0.
	std::vector<double> scratch_fbLegSymmetry;
	std::vector<int> scratch_fbReg;
    // Prepass working rows: edge-replicated padded copies of the three
    // demodded IQ rows (padding reproduces the clamp-to-edge indexing, so
    // the windowed sums read straight pointers), the hoisted d = 0
    // pair-difference row, and the five hoisted same-leg deviation rows
    // used by the registration search.
    std::vector<std::complex<double>> scratch_fbPadCenter;
    std::vector<std::complex<double>> scratch_fbPadUp;
    std::vector<std::complex<double>> scratch_fbPadDn;
    std::vector<std::complex<double>> scratch_fbDiff0;
    std::vector<std::complex<double>> scratch_fbDevRows;
		// Shared line scratch planes used by split2D and witness retraction.
		std::vector<double> scratch_lineWorkA; // Field A scalar / carrier-fit row.
		std::vector<double> scratch_lineWorkB; // Field gate / witness basis-I row.
		std::vector<double> scratch_lineWorkC; // Field B scalar / flattened row.
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
	std::vector<double> scratch_lurchGate;     // Lurch pre-pass: combined support*delta gate per xi (anchor relaxation)
	std::vector<double> scratch_lurchCurve;    // Lurch regression: whole-line banded-LS Y curve per xi
	std::vector<lddecode::FourViewCarrierAttribution> scratch_carrierParallax; // line-local attribution working set
	std::vector<double> scratch_hpI;
	std::vector<double> scratch_hpQ;
	std::vector<double> scratch_hpY;
	std::vector<double> scratch_attrWideCarrier;
	std::vector<double> scratch_attrBandYClaim;
	std::vector<double> scratch_attrMembershipY;
	std::vector<double> scratch_impulseExempt;
	std::vector<double> scratch_frameBReachUp;
	std::vector<double> scratch_frameBReachDown;
	std::vector<std::uint8_t> fieldBDecisionReason_flat;
		// Flat per-sample locked-path buffers (line-major: demodLines x demodWidth).
		//
		// locked1DSource_flat is the locked-path video 1D scalar, declared as
		// the Locked1DScalar reach source (PhasePreservedCarrier).  Physically
		// it is the restrained native bandpass itself: no IQ round trip and no
		// fractional resampling.  Raw carrier orientation is intact on every
		// line.  The retired
		// "common phase / polarity gone by construction" label described a
		// pre-reform pipeline and misled repeatedly.
		//
		// Rules for new code:
		//  1. Prefer locked1DTI4fsc/TQ4fsc (Grid4fscIQ, phase-preserved and
		//     integer-centred on the matching scalar sample) for any operation
		//     that needs polarity in IQ.
		//  2. If this scalar must be demodded for interfield IQ use, demod with
		//     carrierGrammarSignedSampleClass (lineFlip folded into the phase)
		//     to land in Grid4fscIQ.  An unsigned demod yields IQ that inherits
		//     raw signs: interfield (±1) IQ cancels then see real chroma as
		//     anti-phased, flip it, and preserve alien Y — the 2fsc luma
		//     checker in Frame B's locked path came from exactly this.
		//  3. Intrafield (±2 same-field) scalar combs on this buffer are
		//     legitimate: same-field neighbors share lineFlip and the physical
		//     carrier alternation is preserved in the scalar.
		//  4. Cross-line legality is not decided here: ask CombReachIndex with
		//     scalarReachSource(); grammar answers per line pair.
		std::vector<double> locked1DRawBandpass_flat; // raw pass-1 bp[x] before locked cleanup/remod
		std::vector<double> locked1DSource_flat;
		std::vector<float> locked1DParallaxRepairStrength_flat; // [0,1] actual Pass-1.5 applied repair strength
		// Signed Pass-1.5 applied repair delta (signal units) per sample.
		// Published so the retraction stage can align carrierFit with the
		// repaired 1D carrier: the fit is solved from raw BEFORE the repair
		// exists, and without this return path the witness consumes a
		// pre-repair carrier model while every other client consumes the
		// repaired 1D.
		std::vector<float> locked1DParallaxRepairDelta_flat;
	// Application-neutral carrier facts produced for every locked frame. The
	// default 1D feasibility repair, luma witness, and diagnostics all consume
	// this single shared analysis rather than privately reconstructing it.
		std::vector<lddecode::CarrierAnalysisRecord> carrierAnalysis_flat;
		std::vector<AttributionEvidence> attributionEvidence_flat; // Attribution facts/assessment per sample.
	// Default LF platform: one legal carrier-cycle mean per four input
	// samples, held over that cycle (information-rate Nyquist fSC/2).
	std::vector<double> lockedLumaBaseY4_flat;
	std::vector<double> lockedLumaSmooth_flat;
	// Centered lurch-sharpened sliding-boxcar coarse: the witness-only LF
	// platform and authority. Comb supplies middle and provisional top above
	// this same base; only the top is replaceable. lockedLumaSmooth remains a
	// geometry service.
	std::vector<double> lockedLumaSharp_flat;
	// 1D vertical-contrast service: per-sample lateral coarse-luma delta in
	// IRE (|smooth[rel+2] - smooth[rel-2]|), published once by the locked
	// decomposition pass.  The 1D is the first stage to cross a vertical
	// contrast; the high-energy step it registers here serves every later
	// client (Frame B reach exemption, hLumaDeltaIRE, cross-color, FVF
	// vertical regime) without recomputation.
	std::vector<float> lockedLumaHDeltaIRE_flat;
	// COLLECTED POOL (unfiltered): the sliding four-sample aperture means.
	//
	//   lockedApertureMean[v] = mean( raw[left+v .. left+v+3] )
	//
	// indexed by the aperture's START position, so the (up to four) legal
	// apertures covering sample x are v in {x-3, x-2, x-1, x}.
	//
	// This is a measurement, not a product: a legal carrier sums to zero over
	// ANY legal four-sample window, so each mean equals that window's luma mean
	// exactly, and the divergence between the apertures covering one sample is
	// pure luma with the carrier removed exactly (the coarse-residual parallax).
	// It is deliberately published raw -- no sharpening, no gating, no absolute
	// value -- so consumers own the decisions. `lockedLumaSharp` is derived FROM
	// this pool rather than rebuilding it privately.
	//
	// Built unconditionally (a running sum, O(1) per sample) because it has
	// default-path clients, not "just in case".
	std::vector<double> lockedApertureMean_flat;
	// Predicted 1D corner leak, raw units, same geometry as
	// locked1DRawBandpass. chroma = bp - leak, and Y = raw - chroma, so the
	// removed leak RETURNS to luma and Y + chroma == raw exactly (the
	// conservation condition a desaturating suppressor cannot satisfy).
	// DIAGNOSTIC ONLY for now: no consumer, so the render is unchanged.
	std::vector<double> lockedCornerLeak_flat;
	bool lockedLumaCacheValid = false;

	inline double *lockedLumaBaseY4_line(int line) {
		return lockedLumaBaseY4_flat.data() + size_t(line) * demodWidth;
	}

	inline double *lockedLumaSmooth_line(int line) {
		return lockedLumaSmooth_flat.data() + size_t(line) * demodWidth;
	}

	inline double *lockedLumaSharp_line(int line) {
		return lockedLumaSharp_flat.data() + size_t(line) * demodWidth;
	}

	inline const double *lockedLumaBaseY4_line(int line) const {
		return lockedLumaBaseY4_flat.data() + size_t(line) * demodWidth;
	}

	inline const double *lockedLumaSmooth_line(int line) const {
		return lockedLumaSmooth_flat.data() + size_t(line) * demodWidth;
	}

	inline const double *lockedLumaSharp_line(int line) const {
		return lockedLumaSharp_flat.data() + size_t(line) * demodWidth;
	}

	inline double *lockedApertureMean_line(int line) {
		if (lockedApertureMean_flat.empty()) return nullptr;
		return lockedApertureMean_flat.data() + size_t(line) * demodWidth;
	}

	inline double *lockedCornerLeak_line(int line) {
		if (lockedCornerLeak_flat.empty()) return nullptr;
		return lockedCornerLeak_flat.data() + size_t(line) * demodWidth;
	}

	inline const double *lockedCornerLeak_line(int line) const {
		if (lockedCornerLeak_flat.empty()) return nullptr;
		return lockedCornerLeak_flat.data() + size_t(line) * demodWidth;
	}

	inline const double *lockedApertureMean_line(int line) const {
		if (lockedApertureMean_flat.empty()) return nullptr;
		return lockedApertureMean_flat.data() + size_t(line) * demodWidth;
	}

	inline float *lockedLumaHDeltaIRE_line(int line) {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    lockedLumaHDeltaIRE_flat.empty()) return nullptr;
		return lockedLumaHDeltaIRE_flat.data() + size_t(line) * demodWidth;
	}

	inline const float *lockedLumaHDeltaIRE_line(int line) const {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    lockedLumaHDeltaIRE_flat.empty()) return nullptr;
		return lockedLumaHDeltaIRE_flat.data() + size_t(line) * demodWidth;
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

	inline double *locked1DRawBandpass_line(int line) {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    locked1DRawBandpass_flat.empty()) return nullptr;
		return locked1DRawBandpass_flat.data() + static_cast<size_t>(line) * demodWidth;
	}

	inline const double *locked1DRawBandpass_line(int line) const {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    locked1DRawBandpass_flat.empty()) return nullptr;
		return locked1DRawBandpass_flat.data() + static_cast<size_t>(line) * demodWidth;
	}

	inline float *locked1DParallaxRepairStrength_line(int line) {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    locked1DParallaxRepairStrength_flat.empty()) return nullptr;
		return locked1DParallaxRepairStrength_flat.data() + static_cast<size_t>(line) * demodWidth;
	}

	inline const float *locked1DParallaxRepairStrength_line(int line) const {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    locked1DParallaxRepairStrength_flat.empty()) return nullptr;
		return locked1DParallaxRepairStrength_flat.data() + static_cast<size_t>(line) * demodWidth;
	}

	inline float *locked1DParallaxRepairDelta_line(int line) {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    locked1DParallaxRepairDelta_flat.empty()) return nullptr;
		return locked1DParallaxRepairDelta_flat.data() + static_cast<size_t>(line) * demodWidth;
	}

	inline const float *locked1DParallaxRepairDelta_line(int line) const {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    locked1DParallaxRepairDelta_flat.empty()) return nullptr;
		return locked1DParallaxRepairDelta_flat.data() + static_cast<size_t>(line) * demodWidth;
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
		// The unqualified boundary assumes the producer preserved physical
		// carrier orientation. Operations that deliberately removed it must use
		// an explicit source-frame helper instead of changing the default.
		return lddecode::carrierGrammarRemod4fscToComposite(
			carrierGrammarLine(line),
			h,
			I,
			Q,
			1.0,
			lddecode::CarrierSignFrame::MetadataPreservedSigned);
	}

	inline lddecode::CombReachSourceFrame scalarReachSource() const {
		return configuration.phaseCompensation
			? lddecode::makeLocked1DScalarReachSource()
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

		inline double *precleanLinePtrMutable(int lineNumber, int width)
		{
			const int s = precleanRingSlot(lineNumber);
			if ((int)precleanRing[s].size() < width) precleanRing[s].resize(width);
			precleanRingLine[s] = lineNumber;
			return precleanRing[s].data();
		}

	// Small helpers declared here; definitions provided after the class (in this header).
	// Per-run 4fsc shifted basis LUT for locked path (phaseCompensation=true)
	double spLUT_locked[4] = {1.0, 0.0, -1.0, 0.0};
	double cpLUT_locked[4] = {0.0, 1.0,  0.0, -1.0};
	bool   basisLockedInit = false;

	// Hybrid 2D helpers
	void invalidateCombTapCache();
	const CombTapLine &ensureCombTapLine(int lineNumber);
	void buildCombTapLine(int lineNumber, CombTapLine &tapLine);
	// Fill the memoised 7-tap smoothed signed-IQ row for `line` if not yet
	// current this frame.  No-op when the locked IQ / smoothed buffers are
	// not sized (non-locked path).
	void ensureSmoothedLockedRow(int line);
	void computeFieldALine(const CombTapLine &tapLine,
						  double *outFieldLine,
						  double  *outGate);

	void computeFieldBLine(int lineNumber,
						  double *outFieldLine,
						  std::uint8_t *outReasonLine = nullptr);
	void computeFieldBLine(const CombTapLine &tapLine,
						  double *outFieldLine,
						  std::uint8_t *outReasonLine = nullptr);

	void computeFrameALine(int line,
									std::vector<std::complex<double>> &outFrameIQ);
	void computeFrameBLine(int line,
								   std::vector<std::complex<double>> &outFrameIQ,
								   std::vector<double> &outFrameScalar);

	void computeIQFrameAFromPreparedVectors(int line,
											   const std::vector<std::complex<double>> &centerIQ,
											   std::vector<std::complex<double>> &upIQ,
											   std::vector<std::complex<double>> &dnIQ,
											   std::vector<std::complex<double>> &outFrameIQ,
											   const CombTapLine *reachTapLine = nullptr);
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

	// Field-vs-Frame elects scalar bandpass candidates. Candidate A is the
	// same-regime buddy: Field A in interlace, Frame A in progressive.
	// IQ evidence can inform the scores, but IQ-derived candidates must be
	// remodulated before entry.
	void scoreFieldVsFrame(
	    int line,
	    const CombTapLine &tapLine,
	    const std::vector<double> &candidateA,
	    const double *fieldB,
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
						  qint32 &bestIndex, double &bestSample,
						  TemporalCandidateSamples *temporalSamples = nullptr) const;

	// getCandidate is declared here and implemented in comb_candidate.cpp
	Candidate getCandidate(qint32 refLineNumber, qint32 refH,
						   const FrameBuffer &frameBuffer,
						   qint32 lineNumber, qint32 h,
						   double adjustPenalty) const;

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
	inline const float* smoothedLockedTI_line(int line) const {
		return smoothedLockedTI_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline const float* smoothedLockedTQ_line(int line) const {
		return smoothedLockedTQ_flat.data() + static_cast<size_t>(line) * demodWidth;
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
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    carrierImpurity_flat.empty()) return nullptr;
		return carrierImpurity_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline const float* carrierImpurity_line(int line) const {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    carrierImpurity_flat.empty()) return nullptr;
		return carrierImpurity_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline float* lockedCcMaskRaw_line(int line) {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    lockedCcMaskRaw_flat.empty()) return nullptr;
		return lockedCcMaskRaw_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline const float* lockedCcMaskRaw_line(int line) const {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    lockedCcMaskRaw_flat.empty()) return nullptr;
		return lockedCcMaskRaw_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline float* lockedCcMask_line(int line) {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    lockedCcMask_flat.empty()) return nullptr;
		return lockedCcMask_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline const float* lockedCcMask_line(int line) const {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    lockedCcMask_flat.empty()) return nullptr;
		return lockedCcMask_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline float* regionSamePartner_line(int line) {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    regionSamePartner_flat.empty()) return nullptr;
		return regionSamePartner_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline const float* regionSamePartner_line(int line) const {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    regionSamePartner_flat.empty()) return nullptr;
		return regionSamePartner_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline float* regionAlienPartner_line(int line) {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    regionAlienPartner_flat.empty()) return nullptr;
		return regionAlienPartner_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline const float* regionAlienPartner_line(int line) const {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    regionAlienPartner_flat.empty()) return nullptr;
		return regionAlienPartner_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline lddecode::CarrierAnalysisRecord* carrierAnalysis_line(int line) {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    carrierAnalysis_flat.empty()) return nullptr;
		return carrierAnalysis_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline const lddecode::CarrierAnalysisRecord* carrierAnalysis_line(int line) const {
		if (demodWidth <= 0 || line < 0 || line >= demodLines ||
		    carrierAnalysis_flat.empty()) return nullptr;
		return carrierAnalysis_flat.data() + static_cast<size_t>(line) * demodWidth;
	}

	// --- Carrier-retraction accessors (guarded by carrierRetractedValid). ---
	inline float* carrierRetracted_line(int line) {
		if (!carrierRetractedValid || demodWidth <= 0 ||
		    line < 0 || line >= demodLines) return nullptr;
		return carrierRetracted_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline const float* carrierRetracted_line(int line) const {
		if (!carrierRetractedValid || demodWidth <= 0 ||
		    line < 0 || line >= demodLines) return nullptr;
		return carrierRetracted_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	// The NATIVE per-line fit, before any vertical promotion.  raw minus this
	// is the independent inverse-encoder view; raw minus combedCarrier is the
	// comb-conditioned one.  They are different products and must not be
	// confused: only the native fit is free of Field/Frame reach policy.
	inline const float* carrierFit_line(int line) const {
		if (!carrierRetractionModelValid || demodWidth <= 0 ||
		    line < 0 || line >= demodLines) return nullptr;
		return carrierFit_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline float* combedCarrier_line(int line) {
		if (!carrierRetractedValid || demodWidth <= 0 ||
		    line < 0 || line >= demodLines) return nullptr;
		return combedCarrier_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline const float* combedCarrier_line(int line) const {
		if (!carrierRetractedValid || demodWidth <= 0 ||
		    line < 0 || line >= demodLines) return nullptr;
		return combedCarrier_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline lddecode::FourViewPixelEvidence* coarseYEvidence_line(int line) {
		if (!carrierRetractedValid || demodWidth <= 0 ||
		    line < 0 || line >= demodLines ||
		    coarseYEvidence_flat.empty()) return nullptr;
		return coarseYEvidence_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline const lddecode::FourViewPixelEvidence* coarseYEvidence_line(int line) const {
		if (!carrierRetractedValid || demodWidth <= 0 ||
		    line < 0 || line >= demodLines ||
		    coarseYEvidence_flat.empty()) return nullptr;
		return coarseYEvidence_flat.data() + static_cast<size_t>(line) * demodWidth;
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

    // Persistent triple-buffer: the prev/current/next FrameBuffers are reused
    // across decodeFrames() calls instead of being allocated and zero-filled
    // on every batch.  Each FrameBuffer owns ~180 MB of per-pixel attribution
    // storage (CombAttributionRecord = ~456 B/pixel × 525 × 760), so per-batch
    // reconstruction was the single largest cost in the locked path (~25 %
    // of decode wall time on M1 Max).  decodeFrames() takes ownership into
    // locals at entry (preserving the existing std::move rotation) and
    // returns them at exit; updateConfiguration() resets them so a config
    // change re-allocates with the new geometry.
    std::unique_ptr<FrameBuffer> persistentNext;
    std::unique_ptr<FrameBuffer> persistentCurrent;
    std::unique_ptr<FrameBuffer> persistentPrevious;
};

#endif // COMB_H
