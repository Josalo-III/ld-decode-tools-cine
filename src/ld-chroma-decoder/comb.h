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
            bool   BUCKET_NORM_ENABLE    = true;
            double BUCKET_NORM_GAIN_MIN  = 0.85;
            double BUCKET_NORM_GAIN_MAX  = 1.20;
        
            int    NEIGH_WIN_RADIUS      = 2;
            double NEIGH_TOL_IRE         = 1.0;
            double NEIGH_MAX_IRE         = 5.0;
            double NEIGH_MIN_WEIGHT      = 0.2;
            // Sinusoidal subcarrier fit (phaseLocked pre-clean of clpbuffer[0])
            int    SINFIT_WIN_SAMPLES        = 16;   // window for local amplitude estimate (must be multiple of 4)
            double SINFIT_VET_THRESHOLD_IRE  = 3.0;  // residual energy above which fit is rejected per-sample        
            // =========================================================================
            // 2D Field extraction (FieldA/B) and vertical gating
            // =========================================================================
            double FIELD_K_RANGE_IRE           = 45.0;
            bool   FIELD_VERT_EXTENDED_ENABLE  = true;
            double FIELD_VERT_FAR_WEIGHT       = 0.45;
            double FIELD_VERT_FAR_BLEND        = 0.60;
            double FIELD_VERT_K_RANGE_SCALE    = 1.00;
            double FIELD_CONTOUR_FAR_INFLUENCE = 0.55;
            double FIELD_CONTOUR_SOFT_IRE      = 4.0;
            double FIELD_CONTOUR_HARD_IRE      = 10.0;
            double FIELD_CONTOUR_SIM_START     = 0.55;
            double FIELD_CONTOUR_SIM_FULL      = 0.85;
        
            double FIELD_VERT_DISAGREE_THRESH_IRE = 8.0;
            double FIELD_VERT_DISAGREE_SUPPRESS   = 1.00;
            bool   FIELD_SYMMETRIC_H_NEIGHBOR     = true;
        
            double FIELD_LUMA_EDGE_THRESH_IRE       = 18.0;
            double FIELD_LUMA_EDGE_STRICT_RATIO     = 1.5;
            bool   FIELD_LUMA_EDGE_EXCLUDE_ENABLE   = true;

            
            // =========================================================================
            // VDIS (Vertical Differential Isolation System)
            // =========================================================================
            // (opt-in ultra conservative 2D via CLI)
            bool   VDIS_ENABLE                = false; // default disabled; set via --vdis
            double VDIS_PHASE_THRESH_DEG      = 20.0;
            double VDIS_MIN_CHROMA_IRE        = 5.0;
            bool   VDIS_USE_FAR               = false;
            bool   VDIS_NEIGH_CONFIRM         = false;
            bool   VDIS_HARD_FALLBACK         = true;
            double VDIS_SUPPRESS_FACTOR       = 0.0;
            int    VDIS_WIN_SAMPLES           = 4;
            bool   VDIS_CEP_NEIGH_CONFIRM     = true;
            bool   FIELD_VDIS_DISABLE_DILATION      = false;
            bool   VDIS_USE_PLUS1             = true;   // include ±1 evidence along with ±2/±4
            double VDIS_1D_DIFF_THRESH_IRE    = 2.0;    // when VDIS fires, escalate if 2D vs 1D diff > this
            double VDIS_DETAIL_GATE_THRESH_IRE = 2.0;   // If |VDIS - Std2D| < this, revert to Std2D
            double VDIS_1D_BIAS_WEIGHT         = 0.5;   // Bias towards 1D in the gating logic
        
            // =========================================================================
            // Frame comb on phase-corrected 1D
            // =========================================================================
            double FRAME_COMB_STRENGTH            = 1.125; // Frame A
            double FRAME_B_COMB_STRENGTH          = 1.00; // Frame B
            double FRAME_PHASE_DOT_COS            = 0.0;
            double FRAME_CHROMA_MIN_IRE           = 2.5;
            double FRAME_IQ_RAW_MAX_DELTA_IRE     = 8.0;
        
            // =========================================================================
            // FVF (Field vs Frame) scoring (core ±1 vs ±2 diff)
            // =========================================================================
            double FVF_SMALL_DIFF_IRE             = 2.0;  // Give Frame a bonus if diff is ≤ this
            double FVF_LARGE_DIFF_IRE             = 5.0;  // sideline Frame if diff ≥ this
            bool   FVF_BLEND_ENABLE               = false;
            double FVF_BLEND_CHROMA_MIN_IRE       = 12.0;
            double FVF_BLEND_DEV_RATIO            = 1.3;
            double FVF_BLEND_FRAME_BONUS          = 0.25;
            double FVF_BLEND_MIN_FRAME_FRACTION   = 0.20;
            double ONE_D_NEAR_THRESH_IRE          = 1.5;  
            // Neighbor Shaping for FVF
            double FVF_SHAPE_STRENGTH             = 0.75;
            // (reserved)
            // Neighbor-based cross-domain estimate shaping (FVF)
            double NEIGHBOR_EST_WEIGHT        = 0.25;  // penalty weight (in IRE units); keep small
            double NEIGHBOR_EST_SAT_MAX_IRE   = 12.0; // disable in strong saturation
            double NEIGHBOR_EST_EDGE_MAX_IRE  = 10.0; // only use when horizEdgeIRE below this
            double NEIGHBOR_EST_FVF_MAX_IRE   = 4.0;  // only when FVF diff is small / ambiguous
            // =========================================================================
            // FVF / Model interaction tuning
            // =========================================================================
            
            // How strongly a candidate is penalized for distance from the primary model
            // (Frame A in progressive; Field A in interlace). Larger => more model dominance.
            double FVF_MODEL_PRIMARY_WEIGHT   = 0.33;
            
            // Secondary model coherence weight (distance to the "other domain"):
            // - Progressive: distance to best Field
            // - Interlace:   distance to Frame (and/or best field)
            double FVF_MODEL_SECONDARY_WEIGHT = 0.3;
            
            // Small multiplicative advantage to the model domain in its regime.
            // < 1.0 => cheaper score for model candidate.
            double FRAME_MODEL_BIAS  = 0.90;  // progressive: Frame gets small edge
            double FIELD_MODEL_BIAS  = 0.50;  // interlace: Field A gets small edge
            
            // Soft handicap for Frame when we're clearly in interlace regime.
            // > 1.0 => Frame is slightly more "expensive".
            double FRAME_IN_INTERLACE_PENALTY = 1.10;
            
            // Scale-bias strengths (fine vs mid structure) per mode.
            // Progressive: keep some; Interlace: usually 0.0 to disable.
            double FRAME_SCALE_BIAS_STRENGTH_PROGRESSIVE = 0.15;
            double FRAME_SCALE_BIAS_STRENGTH_INTERLACE   = 0.0;
                                    
            // =========================================================================
            // 3D candidate / vet / Y path
            // =========================================================================
            int    CANDIDATE_SYSTEM                 = 0;
            // 2D Similarity Curve for temporal candidates (getBestCandidate).
            //
            // We compare a temporal candidate sample to the local 2D estimate (clpbuffer[1]).
            // Let d = |cand - ref2d| in IRE. We then add a delta to the candidate penalty:
            //   if d <= AGREEMENT_REWARD_RADIUS_IRE:
            //     delta = -AGREEMENT_REWARD_MAX * adaptThreshold * (1 - (d/r)^2)
            //   else if d <= deviationThreshold:
            //     delta = 0
            //   else:
            //     delta = AGREEMENT_VETO_BASE + deviationPenalty * (d - deviationThreshold)
            //
            // Notes:
            // - AGREEMENT_REWARD_MAX is in the same penalty units as getCandidate() (roughly IRE).
            // - adaptThreshold scales only the reward lobe (user-facing strength control).
            // - deviationThreshold and deviationPenalty define the veto knee and slope.
            // - deviationThreshold is shared with getBestY(): it is the
            //   global "temporal mixing veto" control. When the material disagrees
            //   beyond this point, both 3D candidate selection and residual-Y temporal
            //   mixing are suppressed consistently.
            double AGREEMENT_REWARD_RADIUS_IRE    = 5.5;  // reward region radius (IRE)
            double AGREEMENT_REWARD_MAX           = 3.3;  // max reward at d=0 (penalty units, scaled by adaptThreshold)
            double AGREEMENT_VETO_BASE            = 7.0; // base penalty once d exceeds deviationThreshold
            double deviationThreshold            = 8.0; // start of veto region (IRE)
            double deviationPenalty               = 3.3;  // penalty slope beyond deviationThreshold (per IRE)

            // We look for Y leakage in our chroma
            double LEAKAGE_ALPHA_MAX              = 0.60;
            double LEAKAGE_ALPHA_GAIN             = 1.00;
            double LEAKAGE_MIN_ENERGY_IRE         = 1.0;
            bool   LEAKAGE_GATE_BY_COHERENCE      = true;
            // We look for coherence loss to target errors
            bool   CHROMA_COHERENCE_ENABLE        = true;
            int    CHROMA_COH_WIN_SAMPLES         = 16;
            double CHROMA_COH_R0                  = 0.35;
            double CHROMA_COH_GAMMA               = 1.5;
            double CHROMA_COH_SUPPRESS_MAX        = 0.70;

            // Affine of residual Y - required to allow more detail
            bool   Y_LOCAL_AFFINE_ENABLE          = true;
            double Y_LOCAL_MAX_PHASE_DEG          = 14.0;
            double Y_LOCAL_GAIN_MIN               = 0.90;
            double Y_LOCAL_GAIN_MAX               = 1.10;
            double Y_LOCAL_MAX_SHEAR              = 0.12;
            bool   Y_LINE_AFFINE_TRIM_ENABLE      = true;
            double Y_LINE_MAX_PHASE_DEG           = 10.0;
            bool   Y_LINE_ALLOW_GAIN_ON_IQ        = false;
            double Y_LINE_GAIN_MIN                = 0.95;
            double Y_LINE_GAIN_MAX                = 1.05;
            double Y_LINE_MAX_SHEAR               = 0.08;
            
            // 3D Residual Y Logic
            bool   RESIDUAL_Y_ELECTION            = true; // false = Blend (Median-ish), true = Winner-take-all
            double NEIGHBOR_SHAPE_STRENGTH        = 0.5;   // Weight for spatial neighbor agreement     
            // The Vet removes bad residual Y from use.     
            bool   VET_ENABLE_RESIDUAL_Y          = true;
            double VET_PASS_CLOSE_Y_IRE           = 100.0;
            double SELF_INCUMBENT_BONUS           = 0.2; //default 1.0
            double VET_ADJ_NEIGH_THRESH_IRE       = 0.33;
            double VET_ADJ_NEIGH_WEIGHT_CONF      = 0.27;
            double VET_ADJ_NEIGH_WEIGHT_SCORE     = 0.30;
            double VET_Y_NEIGHBOR_WEIGHT          = 0.25;
            // Residual-Y subtraction: in uncertain cases, use 4fSC chroma/luma
            // profile agreement to make a bounded correction to the chroma
            // subtraction amount.
            double VET_Y_CHROMA_LIKE_WEIGHT       = 0.12;
            // Ownership-informed Y reassignment: when ownership evidence says
            // bandpassed energy is luma-owned, return it to Y instead of
            // subtracting it as chroma.
            bool   VET_OWNERSHIP_ENABLE            = true;
            double VET_OWNERSHIP_LUMA_WEIGHT       = 0.5;  // blend strength for lumaClaim reassignment (0=off, 1=full)
            double VET_OWNERSHIP_CHROMA_WEIGHT = 0.12;
            double VET_OWNERSHIP_CONFLICT_SUPPRESS = 0.65;
            double VET_Y_COHERENCE_ROLLBACK_WEIGHT = 0.75;
            double VET_Y_COHERENCE_BADNESS_IRE = 1.5;
			double FVF_OWNERSHIP_LUMA_WEIGHT     = 0.14; // lumaClaim favors less chroma-removal / safer field-side witnesses
			double FVF_OWNERSHIP_CHROMA_WEIGHT   = 0.18; // chromaClaim favors coherent chroma witnesses
			double LOCKED1D_OWNERSHIP_DAMP_WEIGHT = 0.25; // early positive fork: damp 1D chroma when luma-owned
            // Iceberg alien-Y cancellation in buildPhaseCorrected1D.
            // Scales the bandpass-of-smooth-luma prediction at directional edges to
            // compensate for the chroma-canceling smooth blunting the peak height.
            // 1.0 = no recovery boost, 2.0–3.0 = moderate-to-strong boost.
            double LUMA_ICEBERG_RECOVERY           = 2.0;
            double VET_Y_ADAPTIVE_SCALE           = 0.0;
            double VET_Y_ADAPTIVE_CUTOFF          = 0.05;
            bool   VET_RESIDUAL_FIR_ENABLE        = false;
            double VET_RESIDUAL_FIR_THRESH_IRE    = 1.0;
            int    VET_RESIDUAL_FIR_HALF_WIDTH    = 2;
            int    VET_ALIGN_WIN_SAMPLES          = 16;
            double VET_ALIGN_PHASE_MAX_DEG        = 10.0;
            double VET_ALIGN_MIN_RHO              = 0.75;
            double VET_ALIGN_MAX_SHEAR            = 0.15;
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

	FrameBuffer(const LdDecodeMetaData::VideoParameters &videoParameters_,
				const Configuration &configuration_);

	void loadFields(const SourceField &firstField, const SourceField &secondField);

	void split1D();
	void buildPhaseCorrected1D();
	void rebuildLockedDemodFromSelectedComb();
	void split2D();
	void copy2DTo3D(); 
	void split3D(const FrameBuffer &previousFrame,
				 const FrameBuffer &nextFrame);

	void setComponentFrame(ComponentFrame &_componentFrame) { componentFrame = &_componentFrame; }

	void splitIQ();         // Bucket
	// In FrameBuffer public section (near split1D / split2D declarations)
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
	struct SamplePlane {
		alignas(64) double pixel[MAX_HEIGHT][MAX_WIDTH];
	} clpbuffer[3]; // [0]=1D, [1]=2D, [2]=3D

	struct CombTapSample {
		double comp = 0.0;
		double symMag = 0.0;
		float ti = 0.0f;
		float tq = 0.0f;
		double iqMag = 0.0;
		bool haveComp = false;
		bool haveIQ = false;
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
		std::vector<CombTapSample> tap0;
		std::vector<CombTapSample> tapU1;
		std::vector<CombTapSample> tapD1;
		std::vector<CombTapSample> tapU2;
		std::vector<CombTapSample> tapD2;
		std::vector<CombTapSample> tapU4;
		std::vector<CombTapSample> tapD4;
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
	std::vector<float> demodBurstCos;
	std::vector<float> demodBurstSin;
	// Per-line fused locked demod LUTs for the 4 phase buckets (h&3).
	// Computed in phaseLocked() and reused by buildPhaseCorrected1D/splitIQlocked/filterIQLocked.
	std::vector<std::array<float,4>> demodLUTTi_locked;
	std::vector<std::array<float,4>> demodLUTTq_locked;

	// Flat/contiguous buffers (lines x width)
	// Line-local locked IQ after burst alignment and affine trim.
	std::vector<float> demodTI_flat;
	std::vector<float> demodTQ_flat;
	// Common 4fsc IQ export derived from the locked IQ. This is the seam
	// between the line-local locked domain and the cross-line 4fsc domain.
	std::vector<float> demodTI4fsc_flat;
	std::vector<float> demodTQ4fsc_flat;
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
	// in FrameBuffer private members (comb.h)
	std::vector<int> lineFlip;  // +1 or -1 per frame line
	std::vector<double> scratch_outMixed;
	std::vector<double> scratch_lateralLine;
	std::vector<std::vector<float>> w2d_frame_weight;
	std::vector<std::vector<double>> w2d_fieldA_gate;
	std::vector<std::vector<FvfModelMetrics>> fvfMetrics;
	std::vector<LineAffine> lineAffineLocked;
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
	// Per-pixel precleaned Frame A value (1D-conditioned same-phase blend
	// of framePreclean). Cached during the main scoring pass so the island
	// filter and any post-processing can recover the Frame A output.
	std::vector<double> scratch_fvf_frameAVal;
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
	inline bool   getLinePhase(qint32 lineNumber) const;
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
										   bool enableLateralRefine);
	void collectCombOwnershipEvidence(int line,
									   const double *fieldA,
									   const double *fieldB,
									   const std::vector<double> &frameScalar,
									   const std::vector<std::complex<double>> *frameIQ);
	void finalizeOwnershipClaims(OwnershipEvidence &e,
								 double neighborLumaMeanIRE = -1.0,
								 double neighborBaseMeanIRE = -1.0) const;
	void reportPhaseLegStats(const char *label, int srcBufIndex, bool useLockedSource) const;
	// Unified VDIS map builder: combines scalar (±2) and IQ (±1) evidence
	// into scratch_vdis_flag for a given line. Does not modify FieldA/Frame.
	void computeVDISLine(int lineNumber);
	static void consolidateVDISRegions(std::vector<std::vector<char>> &mask,
									   const LdDecodeMetaData::VideoParameters &vp);
	
	// Minimal Field-vs-Frame scorer: uses normalized FieldB and Frame plus
	// phase-corrected 1D as fallback reference only.
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

inline bool Comb::FrameBuffer::getLinePhase(qint32 lineNumber) const {
    const qint32 fieldID = getFieldID(lineNumber);
    const bool positiveOnEven = (fieldID == 1) || (fieldID == 4);
    const qint32 fieldLine = lineNumber / 2;
    const bool evenLine = (fieldLine % 2) == 0;
    return evenLine ? positiveOnEven : !positiveOnEven;
}

#endif // COMB_H
