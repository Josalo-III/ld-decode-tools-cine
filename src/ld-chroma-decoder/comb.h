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
#include "compositeownershipdefs.h"
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
        bool   debugPhaseLegs = false; // Log per-(h&3) locked-demod residual stats.
        // Demod plus Y selection: phase locked vs bucket
        // Phase locked is a coherent path that includes HF Y from composite
        bool phaseCompensation = false;

        // If true, keep clpbuffer[1] burst-referenced, but feed the 2D comb stage
        // from a 4fsc-grid remodulated copy (reduces 2D Nyquist/zipper artifacts).
        bool locked2DSourceTo4fsc = true;

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
            Line,           // 1D only
            Field,          // Field A (longer reach; more rules)
            FieldB,         // Field B (Simple)
            FramePreclean,  // Frame A: scalar interfield frame from Field B-precleaned scalar
            FrameRaw,       // Frame B: simple interfield frame from locked 1D IQ
            FieldVsFrame    // FVF (Default)
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
            // Sinusoidal fit used by the phaseLocked pre-clean of clpbuffer[0].
            int    SINFIT_WIN_SAMPLES       = 16;  // samples per window; must be a multiple of 4 (one 4fsc period)
            double SINFIT_VET_THRESHOLD_IRE = 3.0; // per-sample fit residual above which that sample is rejected

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

            // =========================================================================
            // VDIS (Vertical Differential Isolation System)
            // =========================================================================
            // Conservative 2D fallback that reverts to 1D when carrier phase is incoherent
            // across lines (edit cuts, TBC errors, head-switching noise). Off by default.
            bool   VDIS_ENABLE             = false; // opt-in via --vdis
            double VDIS_PHASE_THRESH_DEG   = 20.0;  // carrier phase difference across ±2 lines that triggers VDIS
            double VDIS_MIN_CHROMA_IRE     = 5.0;   // minimum chroma amplitude for VDIS to be sensitive (below this, phase is noisy)
            bool   VDIS_HARD_FALLBACK      = true;  // true → replace 2D with 1D; false → blend by VDIS_SUPPRESS_FACTOR
            double VDIS_SUPPRESS_FACTOR    = 0.0;   // 2D output scale when soft fallback active (0 = full 1D, 1 = no change)
            bool   VDIS_USE_PLUS1          = true;  // include ±1-line phase evidence alongside ±2/±4
            double VDIS_1D_DIFF_THRESH_IRE = 2.0;   // escalate to hard fallback when |2D − 1D| exceeds this after VDIS fires

            // =========================================================================
            // Frame comb on phase-corrected 1D
            // =========================================================================
            double FRAME_COMB_STRENGTH        = 1.125; // interframe cancellation amplitude scale for Frame A (>1 boosts cancellation)
            double FRAME_B_COMB_STRENGTH      = 1.00;  // same for Frame B (raw locked-1D interframe path)
            double FRAME_CHROMA_MIN_IRE       = 2.5;   // minimum chroma amplitude to engage the frame IQ path
            double FRAME_IQ_RAW_MAX_DELTA_IRE = 8.0;   // max IQ mismatch between locked-1D and frame average before frame IQ is distrusted
            double FRAME_B_LEAK_NEG_CORR_START = 0.30; // signed corr negativity where Frame B starts treating vertical alternation as leakage
            double FRAME_B_LEAK_NEG_CORR_FULL  = 0.70; // signed corr negativity where leakage confidence is full
            double FRAME_B_LEAK_NEIGHBOR_AGREE_START = 0.20; // Up/Dn signed corr where leakage support begins
            double FRAME_B_LEAK_NEIGHBOR_AGREE_FULL  = 0.70; // Up/Dn signed corr where leakage support is full
            double FRAME_B_LEAK_CENTER_DELTA_START_IRE = 1.5; // |center - avg(Up,Dn)| where symmetric cleanup starts to look worthwhile
            double FRAME_B_LEAK_CENTER_DELTA_FULL_IRE  = 5.5; // |center - avg(Up,Dn)| where symmetric cleanup confidence is full
            double FRAME_B_LEAK_STRENGTH_BOOST = 0.45; // extra push toward full comb strength when leakage is likely

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
            double FVF_SCALE_COARSE_FIELD_A_BONUS = 0.14;
            double FVF_SCALE_COARSE_DUAL4_FIELD_A_BONUS = 0.05;

            // Edge-regime biasing for the 4-member election.
            double FVF_VERT_FIELD_A_PENALTY  = 0.16; // Field A penalty under vertical contrast (±2 comb produces crosstalk)
            double FVF_VERT_FRAME_A_BONUS    = 0.16; // Frame A (precleaned) bonus under vertical contrast (unaffected by vertical luma)
            double FVF_HEDGE_FIELD_B_PENALTY = 0.18; // Field B penalty at horizontal luma edges (zipper risk)
            double FVF_HEDGE_FRAME_B_BONUS   = 0.18; // Frame B bonus at horizontal luma edges (interframe is stable there)

            // Transition sharpness reward strength.
            double FVF_TRANSITION_SHARPNESS_WEIGHT = 0.10;

            // Saturation regime biasing.
            double FVF_SAT_FIELD_A_PEN   = 0.06; // Field A penalty in high saturation
            double FVF_SAT_FIELD_B_PEN   = 0.24; // Field B penalty in high saturation (higher: zipper risk)
            double FVF_SAT_FRAME_A_BONUS = 0.10; // Frame A bonus in high saturation (smaller: boundary caution)
            double FVF_SAT_FRAME_B_BONUS = 0.18; // Frame B bonus in high saturation when coherent

            // Cross-domain penalty derived from a spatial neighbor estimate; guards ambiguous pixels
            // where horizontal structure could alias the interfield or interframe comb.
            double NEIGHBOR_EST_WEIGHT      = 0.25; // penalty weight applied to the cross-domain estimate (IRE units)
            double NEIGHBOR_EST_SAT_MAX_IRE = 12.0; // disable neighbor estimate when local chroma exceeds this
            double NEIGHBOR_EST_EDGE_MAX_IRE = 10.0; // disable when horizontal luma edge exceeds this
            double NEIGHBOR_EST_FVF_MAX_IRE  = 4.0;  // only apply when the FVF diff itself is small / ambiguous

            // =========================================================================
            // FVF / Model interaction tuning
            // =========================================================================
            // Penalty for a candidate being far from the regime model
            // (Frame A in progressive; Field A in interlace). Larger → more model dominance.
            double FVF_MODEL_PRIMARY_WEIGHT = 0.33;

            // Multiplicative cost advantage for the model-domain candidate in its own regime.
            // < 1.0 → model candidate gets a cheaper score.
            double FRAME_MODEL_BIAS = 0.90; // progressive: Frame A costs less
            double FIELD_MODEL_BIAS = 0.50; // interlace: Field A costs less

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

            // Frame-level affine correction applied to locked-1D IQ before chroma subtraction.
            // Corrects residual carrier phase/gain drift across the frame.
            bool   Y_LOCAL_AFFINE_ENABLE = true;
            double Y_LOCAL_MAX_PHASE_DEG = 14.0; // maximum phase correction the frame affine may apply
            double Y_LOCAL_GAIN_MIN      = 0.90; // frame affine gain clamp lower bound
            double Y_LOCAL_GAIN_MAX      = 1.10; // frame affine gain clamp upper bound
            double Y_LOCAL_MAX_SHEAR     = 0.12; // maximum I/Q cross-coupling (shear) the frame affine may apply

            // Per-line affine trim: refines the frame-level fit on a line-by-line basis.
            bool   Y_LINE_AFFINE_TRIM_ENABLE     = true;
            double Y_LINE_MAX_PHASE_DEG          = 10.0;  // maximum per-line phase adjustment
            bool   Y_LINE_ALLOW_GAIN_ON_IQ       = false; // allow gain adjustment on IQ (vs phase-only trim)
            double Y_LINE_GAIN_MIN               = 0.95;  // per-line gain clamp lower bound
            double Y_LINE_GAIN_MAX               = 1.05;  // per-line gain clamp upper bound
            double Y_LINE_MAX_SHEAR              = 0.08;  // maximum per-line shear
            bool   Y_LINE_PHASE_ERROR_LUT_ENABLE = true;  // apply a LUT to correct known per-line carrier phase errors
            double Y_LINE_PHASE_ERROR_MIN_CONF   = 0.50;  // minimum burst confidence for LUT correction to engage

            // 3D Residual Y selection
            bool   RESIDUAL_Y_ELECTION     = true; // true = winner-take-all; false = median-weighted blend
            double NEIGHBOR_SHAPE_STRENGTH = 0.5;  // how strongly spatial neighbor consensus pulls the 3D Y election

            // Vet gate: screens residual Y candidates before they are applied.
            bool   VET_ENABLE_RESIDUAL_Y = true;

            // Chroma-profile correction: adjusts subtraction alpha when the vet is uncertain,
            // using 4fsc chroma/luma profile agreement as a bounded correction signal.
            double VET_Y_CHROMA_LIKE_WEIGHT = 0.12; // weight of chroma-profile match on the subtraction alpha

            // Ownership-informed Y reassignment: returns bandpassed energy to Y when
            // ownership evidence says it is luma-owned rather than chroma-owned.
            bool   VET_OWNERSHIP_ENABLE            = true;
            double VET_OWNERSHIP_LUMA_WEIGHT       = 0.75; // blend strength for lumaClaim → Y reassignment (0=off, 1=full)
            double VET_OWNERSHIP_CHROMA_WEIGHT     = 0.75; // blend strength for chromaClaim → chroma retention
            double VET_OWNERSHIP_CONFLICT_SUPPRESS = 0.65; // attenuate ownership adjustment when luma and chroma claims both high
            double VET_OWNERSHIP_BRIGHT_START_IRE  = 55.0; // begin bowing out of ownership reassignment above this luma level
            double VET_OWNERSHIP_BRIGHT_FULL_IRE   = 80.0; // fully disable ownership reassignment by this luma level
            double VET_OWNERSHIP_SAT_START_IRE     = 8.0; // begin bowing out when local chroma reaches this amplitude
            double VET_OWNERSHIP_SAT_FULL_IRE      = 20.0; // fully disable ownership reassignment by this chroma amplitude

            // FVF ownership: ownership evidence adjusts field vs frame penalty before election.
            double FVF_OWNERSHIP_LUMA_WEIGHT   = 0.14; // lumaClaim → added to frame scores (field is safer when luma-owned)
            double FVF_OWNERSHIP_CHROMA_WEIGHT = 0.18; // chromaClaim → added to field scores (frame is safer when chroma coherent)

            // Locked-1D ownership damp: scales locked-1D IQ down when early luma-ownership
            // evidence is strong, before tiRow/tqRow and downstream consumers receive it.
            double LOCKED1D_OWNERSHIP_DAMP_WEIGHT = 0.25; // damp strength (0=off, 1=full suppression at lumaClaim=1)

            // Iceberg recovery: compensates for smooth-luma cancellation underestimating
            // alien-Y amplitude at directional edges in buildPhaseCorrected1D.
            // 1.0 = no boost; 2.0–3.0 = moderate-to-strong recovery.
            double LUMA_ICEBERG_RECOVERY = 2.5;

            // Alignment check applied before residual Y to verify carrier coherence.
            int    VET_ALIGN_WIN_SAMPLES   = 16;   // samples in the alignment correlation window
            double VET_ALIGN_PHASE_MAX_DEG = 12.0; // maximum phase error the alignment correction may apply
            double VET_ALIGN_MIN_RHO       = 0.75; // minimum burst correlation for alignment to be trusted
            double VET_ALIGN_MAX_SHEAR     = 0.15; // maximum shear the alignment correction may apply
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
	enum class DemodMode { Bucket, Locked };

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
		double frameToFieldModelIRE = 0.0;  // interlace regime: |Frame - FieldA(model)|
		double frameToBestFieldIRE = 0.0;   // progressive regime: |Frame - min(FieldA,FieldB)|

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

	// Signal ownership evidence, collected before election. This is not a
	// scoring model; it records why bandpassed energy looks luma-owned,
	// chroma-owned, or contested so demod/admission can later act on it.
	using OwnershipEvidence = lddecode::CombOwnershipEvidence;
	using OwnershipFacts = lddecode::CombOwnershipFacts;
	using OwnershipAssessment = lddecode::CombOwnershipAssessment;
	using OwnershipRules = lddecode::OwnershipRules;

	FrameBuffer(const LdDecodeMetaData::VideoParameters &videoParameters_,
				const Configuration &configuration_);

	void loadFields(const SourceField &firstField, const SourceField &secondField);

	void split1D();
	void buildPhaseCorrected1D();
	void split2D();
	void copy2DTo3D(); 
	void split3D(const FrameBuffer &previousFrame,
				 const FrameBuffer &nextFrame);

	void setComponentFrame(ComponentFrame &_componentFrame) { componentFrame = &_componentFrame; }

	void splitIQ();         // Bucket
	void phaseLocked();  // prepares locked-path LO / basis etc.
	void splitIQlocked();   // Product (burst-locked)

	void filterIQ();
	void filterIQLocked();

	void adjustY();         // Bucket path
	void produceY();        // Product path
	// Build the carrier-retracted view of the raw composite waveform.
	// For each sample, fits a windowed LS carrier model to
	// (rawLine - lockedLumaBaseY4), then stores (rawLine - carrierFit)
	// in carrierRetracted_flat.
	// Must be called after phaseLocked() (which populates lockedLumaBaseY4
	// and the burst grammar) and before split2D() / buildPhaseCorrected1D().
	// The result is read-only for all downstream stages.
	void buildCarrierRetracted();
	void doCNR();
	void doYNR();
	void transformIQ(double chromaGain, double chromaPhase);
	void overlayMap(const FrameBuffer &previousFrame,
					const FrameBuffer &nextFrame);

	const std::vector<std::vector<FvfModelMetrics>> &getFvfMetrics() const { return fvfMetrics; }
	const std::vector<std::vector<OwnershipEvidence>> &getOwnershipEvidence() const { return ownershipEvidence; }

	// Optional temporal context pointers used by Residual Y 3D election (set by decodeFrames)
	// Not owned — just references to neighboring FrameBuffer objects (may be nullptr).
	const FrameBuffer *prevFrameForVet = nullptr;
	const FrameBuffer *nextFrameForVet = nullptr;

	DemodMode demodMode = DemodMode::Bucket;

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
	struct LineAffine {
		double R[2][2];  // rotation+gain (phase-clamped)
		bool   valid;
	};
	struct CombCarrierGrammar {
		double burstCos = 1.0;
		double burstSin = 0.0;
		double carrierScale = 0.0;     // IRE burst magnitude before normalization/flooring
		double phaseError = 0.0;       // measured carrier-phase error, radians
		double phaseConfidence = 0.0;  // 0..1 confidence from burst magnitude
		int    fieldPhaseId = 0;       // 1..4 from metadata
		int    lineParity = 0;         // frame-line parity
		int    fieldLine = 0;          // line / 2 in the interleaved frame
		int    lineFlip = 1;           // +1 or -1 subcarrier polarity for this line
		int    samplePhase0 = 0;       // h&3 origin for this line's carrier grid
		bool   frameVerticalAllowed = false;
		bool   grammarLocked = false;

		// Schedule conflict diagnostics (filled in loadFields; may be
		// refined by phaseLocked if burst measurement diverges from metadata).
		//
		// lineFlipAuthority records the source of lineFlip above.
		// rigidScheduleLineFlip stores the rigid-schedule derivation for
		// diagnostic comparison.  phaseScheduleConflict is non-zero when
		// they disagree.  Downstream code must not "fix" lineFlip by
		// substituting rigidScheduleLineFlip.
		lddecode::CarrierPhaseAuthority lineFlipAuthority = lddecode::CarrierPhaseAuthority::Metadata;
		int    rigidScheduleLineFlip = +1;
		double phaseScheduleConflict = 0.0;  // 0 = agreement, 1 = full conflict

		std::array<float,4> demodLUTTi = {0.0f, 0.0f, 0.0f, 0.0f};
		std::array<float,4> demodLUTTq = {0.0f, 0.0f, 0.0f, 0.0f};
		LineAffine affine;

		// Line-level carrier projection summary (filled by projectCarrierPerLine).
		double meanForwardErrorIRE = 0.0;  // mean |residual − C_model| in IRE
		double meanChromaMagIRE    = 0.0;  // mean |I,Q| from residual demod, IRE
		double carrierFitRatio     = 0.0;  // 1 − (forwardError / residualEnergy), [0,1]
		bool   projectionValid     = false;
	};
	struct SamplePlane {
		alignas(64) double pixel[MAX_HEIGHT][MAX_WIDTH];
	} clpbuffer[3]; // [0]=1D, [1]=2D, [2]=3D

	struct CombTapScalar {
		double comp = 0.0;
		double symMag = 0.0;
	};

	struct CombTapIQ {
		float ti = 0.0f;
		float tq = 0.0f;
		double iqMag = 0.0;
	};

	struct CombTapPair {
		double diffIRE = std::numeric_limits<double>::infinity();
		double iqDiffIRE = std::numeric_limits<double>::infinity();
		double coherence = 1.0;
		double kScore = 0.0;
		double weight = 1.0;
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
		bool haveIQ0 = false;
		bool haveIQU1 = false;
		bool haveIQD1 = false;
		bool haveIQU2 = false;
		bool haveIQD2 = false;
		bool haveIQU4 = false;
		bool haveIQD4 = false;
		std::vector<CombTapScalar> tap0;
		std::vector<CombTapScalar> tapU1;
		std::vector<CombTapScalar> tapD1;
		std::vector<CombTapScalar> tapU2;
		std::vector<CombTapScalar> tapD2;
		std::vector<CombTapScalar> tapU4;
		std::vector<CombTapScalar> tapD4;
		std::vector<CombTapIQ> tap0IQ;
		std::vector<CombTapIQ> tapU1IQ;
		std::vector<CombTapIQ> tapD1IQ;
		std::vector<CombTapIQ> tapU2IQ;
		std::vector<CombTapIQ> tapD2IQ;
		std::vector<CombTapIQ> tapU4IQ;
		std::vector<CombTapIQ> tapD4IQ;
		std::vector<CombTapPair> pairU1;
		std::vector<CombTapPair> pairD1;
		std::vector<CombTapPair> pairU2;
		std::vector<CombTapPair> pairD2;
		std::vector<CombTapContour> contour;
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

	// Flat/contiguous buffers (lines x width)
	// Line-local locked IQ after burst alignment and affine trim.
	std::vector<float> demodTI_flat;
	std::vector<float> demodTQ_flat;
	// Common 4fsc IQ export derived from the locked IQ. This is the seam
	// between the line-local locked domain and the cross-line 4fsc domain.
	std::vector<float> demodTI4fsc_flat;
	std::vector<float> demodTQ4fsc_flat;
	// Preserved 4fsc IQ produced by buildPhaseCorrected1D().
	// Frame B should read this earlier cache instead of depending on later
	// repurposing of the shared demodTI4fsc/TQ4fsc working buffers.
	std::vector<float> locked1DTI4fsc_flat;
	std::vector<float> locked1DTQ4fsc_flat;
	// Product-scaled locked IQ prepared by splitIQLocked() for the output FIR.
	// Later stages may refine this cache, but they should not overwrite demodTI/TQ.
	std::vector<float> lockedProductI_flat;
	std::vector<float> lockedProductQ_flat;
	std::vector<double> scratch_lumaBaseY4;
	std::vector<double> scratch_lumaHiRaw;
	std::vector<double> scratch_lumaSmooth;
	// 3-slot ring buffers caching an intrafield comb output (chroma + gate)
	// used as preclean input for locked Frame IQ demod. Only adjacent lines are needed.
	std::array<std::vector<double>, 3> precleanRing;
	std::array<std::vector<double>, 3> precleanGateRing;
	std::array<int, 3> precleanRingLine = { -1, -1, -1 };
	std::vector<double> scratch_frameBCenter;
	std::vector<double> scratch_fieldBCenter;        // raw-composite demod storage (flat contiguous)
	std::vector<float> demodTRI_flat;
	std::vector<float> demodTRQ_flat;
	std::vector<double> scratch_preI;          // unscaled pre-FIR storage (per-line)
	std::vector<double> scratch_preQ;
	std::vector<double> scratch_preI_ext;      // edge-extended for FIR (per-line)
	std::vector<double> scratch_preQ_ext;
	std::vector<double> scratch_comp_res;     // composite residual = raw - clp - chroma_est
	// Per-line HP-Y and predictor demod scratch (used by splitIQlocked leakage cancellation)
	std::vector<double> scratch_yhp;   // simple HP of Y (per-line)
	std::vector<double> scratch_yI;    // demodulated HP-Y I component (post-affine)
	std::vector<double> scratch_yQ;    // demodulated HP-Y Q component (post-affine)
	std::vector<double> scratch_outMixed;
	std::vector<double> scratch_lateralLine;
	std::vector<std::vector<float>> w2d_frame_weight;
	std::vector<std::vector<double>> w2d_fieldA_gate;
	std::vector<std::vector<FvfModelMetrics>> fvfMetrics;
	std::vector<std::complex<double>> scratch_iq; // reused per-line I/Q scratch (phase-corrected 1D)
	std::vector<std::complex<double>> scratch_centerIQ; // reused per-line preclean/locked frame IQ prep
	std::vector<std::complex<double>> scratch_upIQ;
	std::vector<std::complex<double>> scratch_dnIQ;
	std::vector<double> scratch_fieldLine;
	std::vector<double> scratch_fieldGate;
	std::vector<double> scratch_fieldBLine;
	// FVF per-line scratch (avoid per-line allocations in scoreFieldVsFrame)
	std::vector<int>    scratch_fvf_winner;
	std::vector<int>    scratch_fvf_winner2;
	std::vector<double> scratch_fvf_outVal;
	std::vector<float>  scratch_fvf_outShade;
	std::vector<double> scratch_fvf_diffFVF;
	std::vector<double> scratch_fvf_satMap;
	std::vector<double> scratch_fvf_iqMag;     // per-line IQ magnitude pre-pass (scoreFieldVsFrame)
	std::vector<double> scratch_coe_coherence; // per-line IQ coherence pre-pass (collectCombOwnershipEvidence)
	// Per-pixel precleaned Frame A value (1D-conditioned same-phase blend
	// of framePreclean). Cached during the main scoring pass so the island
	// filter and any post-processing can recover the Frame A output.
	std::vector<double> scratch_filter_temp;
	std::vector<double> scratch_hpI;
	std::vector<double> scratch_hpQ;
	std::vector<double> scratch_hpY;
	std::vector<double> scratch_sinfit_mag;    // per-line |TRI/TRQ|
	std::vector<double> scratch_sinfit_resmag; // per-line residual magnitude estimate
	std::vector<char> scratch_vdis_flag;
	std::vector<std::vector<char>> vdisMask; // [line][rel], persistent per frame
	std::vector<std::vector<double>> locked1DSource; // [line][rel], common-4fsc scalar export for locked 2D
	std::vector<std::vector<OwnershipEvidence>> ownershipEvidence; // [line][rel]
	std::vector<double> lockedLumaBaseY4_flat;
	std::vector<double> lockedLumaSmooth_flat;
	bool lockedLumaCacheValid = false;
	// Raw composite with the windowed LS carrier fit removed (per-line, same
	// geometry as the demod flat buffers).  Populated by buildCarrierRetracted()
	// after produceY(); valid when carrierRetractedValid is true.
	std::vector<float> carrierRetracted_flat;
	bool carrierRetractedValid = false;
	
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
		const CombCarrierGrammar *g = carrierGrammarLine(line);
		const int phase0 = g ? g->samplePhase0 : 0;
		return (h + phase0) & 3;
	}

	inline int carrierSignedSampleClass(int line, int h) const {
		int ph = carrierSampleClass(line, h);
		const CombCarrierGrammar *g = carrierGrammarLine(line);
		if (g && g->lineFlip < 0)
			ph = (ph + 2) & 3;
		return ph;
	}

	inline int carrierOppositeSampleClass(int line, int h) const {
		return (carrierSignedSampleClass(line, h) + 2) & 3;
	}

	inline double carrierPlausibility(const CombCarrierGrammar *grammar) const {
		if (!configuration.phaseCompensation)
			return 1.0;
		if (!grammar || !grammar->grammarLocked)
			return 0.0;
		double base = grammar->projectionValid
			? std::clamp(grammar->carrierFitRatio, 0.0, 1.0)
			: std::clamp(grammar->phaseConfidence, 0.0, 1.0);
		// Burst-vs-metadata disagreement reduces plausibility.  A conflict
		// of 1.0 (≥45° divergence) halves the score; partial conflict
		// attenuates proportionally.
		const double conflictPenalty =
			1.0 - 0.5 * std::clamp(grammar->phaseScheduleConflict, 0.0, 1.0);
		return base * conflictPenalty;
	}

	inline double remodGrammarToComposite(int line, int h,
	                                      double I, double Q) const {
		const int ph = carrierSampleClass(line, h);
		const int f = carrierLineFlip(line);
		return remod4fscToCompositePhase(I, Q, ph, f);
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
		if ((int)precleanRing[s].size() < width) precleanRing[s].assign(width, 0.0);
		precleanRingLine[s] = lineNumber;
		return precleanRing[s].data();
	}

	inline double *precleanGateLinePtrMutable(int lineNumber, int width)
	{
		const int s = precleanRingSlot(lineNumber);
		if ((int)precleanGateRing[s].size() < width) precleanGateRing[s].assign(width, 1.0);
		return precleanGateRing[s].data();
	}

	// Small helpers declared here; definitions provided after the class (in this header).
	inline qint32 getFieldID(qint32 lineNumber) const;
	// Per-run 4fsc shifted basis LUT for locked path (phaseCompensation=true)
	double spLUT_locked[4] = {1.0, 0.0, -1.0, 0.0};
	double cpLUT_locked[4] = {0.0, 1.0,  0.0, -1.0};
	bool   basisLockedInit = false;
	bool hasVDIS(int lineNumber, int h) const;      

	// Hybrid 2D helpers
	void invalidateCombTapCache();
	const CombTapLine &ensureCombTapLine(int lineNumber);
	void buildCombTapLine(int lineNumber, CombTapLine &tapLine);
	void computeField2DLine(int lineNumber,
						  double *outFieldLine,
						  double  *outGate);
	void computeField2DLine(const CombTapLine &tapLine,
						  double *outFieldLine,
						  double  *outGate);

	void computeSimpleField2DLine(int lineNumber, double *outFieldLine);
	void computeSimpleField2DLine(const CombTapLine &tapLine, double *outFieldLine);
	
	void computeFrameScalarLine(int lineNumber, double *outFrameLine);
	void computeFrameScalarLine(const CombTapLine &tapLine, double *outFrameLine);

	void computeFrameIQPrecleanLine(int line,
									std::vector<std::complex<double>> &outFrameIQ,
									bool enableLateralRefine = true);
	void computeFrameBLocked1DLine(int line,
								   std::vector<std::complex<double>> &outFrameIQ,
								   std::vector<double> &outFrameScalar);
	void computeFrameIQLocked1DLine(int line,
									std::vector<std::complex<double>> &outFrameIQ,
									const std::vector<float> *tiOverride = nullptr,
									const std::vector<float> *tqOverride = nullptr);
		void computeFrameIQFromPreparedVectors(int line,
											   const std::vector<std::complex<double>> &centerIQ,
											   std::vector<std::complex<double>> &upIQ,
											   std::vector<std::complex<double>> &dnIQ,
											   std::vector<std::complex<double>> &outFrameIQ,
											   const std::vector<float> *tiOverride,
											   const std::vector<float> *tqOverride,
											   bool enableLateralRefine,
											   bool allowSymmetricLeakCancel = false);
	void collectCombOwnershipEvidence(int line,
									   const double *fieldA,
									   const double *fieldB,
									   const std::vector<double> &frameScalar,
									   const std::vector<std::complex<double>> *frameIQ);
	void seedCombOwnershipPerLine(int line);
	void finalizeOwnershipClaims(OwnershipEvidence &e,
								 double neighborLumaMeanIRE = -1.0,
								 double neighborBaseMeanIRE = -1.0,
								 double lineForwardErrorIRE = 0.0) const;
	void reportPhaseLegStats(const char *label, int srcBufIndex, bool useLockedSource) const;
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
		const double *fieldA,
		const double *fieldB,
		const double *fieldAGate,
		const std::vector<double> &framePreclean,
		const std::vector<double> *frameRaw,
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

	// Raw-composite demod accessors
	inline float* demodTRI_line(int line) {
		return demodTRI_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline float* demodTRQ_line(int line) {
		return demodTRQ_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline const float* demodTRI_line(int line) const {
		return demodTRI_flat.data() + static_cast<size_t>(line) * demodWidth;
	}
	inline const float* demodTRQ_line(int line) const {
		return demodTRQ_flat.data() + static_cast<size_t>(line) * demodWidth;
	}

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

// Inline helper definitions for FrameBuffer (out-of-class)
inline qint32 Comb::FrameBuffer::getFieldID(qint32 lineNumber) const {
    return ((lineNumber % 2) == 0) ? firstFieldPhaseID : secondFieldPhaseID;
}

#endif // COMB_H
