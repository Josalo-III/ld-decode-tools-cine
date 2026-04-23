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
			Line,           // 1D only
			Field,          // Field A (Smart VDIS)
			FieldB,         // Field B (Simple)
			Frame,          // Frame (was FrameB2)
			FieldVsFrame    // FVF (Default)
		};
		TwoDVariant twoDVariant = FieldVsFrame;
	
		// Opt-in for 3D temporal checks in Residual Y (vetComposite1D)
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

		// Merged / pruned Tunables block preserving your additions and values.
		// - Keeps your HYBRID_* VDIS knobs for compatibility, and aliases VDIS_* noted below.
		// - Preserves ONE_D_NEAR_THRESH_IRE and the ANTI_COMPOSITE_* controls.
		// - Uses your boosted FVF thresholds (3.0 / 7.0) as requested.
		// - Adds VDIS_USE_PLUS1 and VDIS_1D_DIFF_THRESH_IRE with sensible defaults.
		//
		// Replace the Tunables block in comb.h with this merged version.
		// If you want the old block saved in-repo, use the backup file above.
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
		
			double FIELD_VERT_DISAGREE_THRESH_IRE = 10.0;
			double FIELD_VERT_DISAGREE_SUPPRESS   = 1.00;
			bool   FIELD_SYMMETRIC_H_NEIGHBOR     = true;
		
			double FIELD_LUMA_EDGE_THRESH_IRE       = 18.0;
			double FIELD_LUMA_EDGE_STRICT_RATIO     = 1.5;
			bool   FIELD_LUMA_EDGE_EXCLUDE_ENABLE   = true;
		
		
			// =========================================================================
			// VDIS (Vertical Differential Isolation System)
			// =========================================================================
			// (opt-in via CLI)
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
			double FRAME_COMB_STRENGTH            = 1.2;
			double FRAME_PHASE_DOT_COS            = 0.0;
			double FRAME_CHROMA_MIN_IRE           = 2.5;
		
			double FRAME_IQ_RAW_MAX_DELTA_IRE     = 8.0;
		
			// =========================================================================
			// FVF (Field vs Frame) scoring (core ±1 vs ±2 diff)
			// =========================================================================
			double FVF_SMALL_DIFF_IRE             = 2.5;  // Give Frame a bonus if diff is ≤ this
			double FVF_LARGE_DIFF_IRE             = 5.0;  // sideline Frame if diff ≥ this
			bool   FVF_BLEND_ENABLE               = false;
			double FVF_BLEND_CHROMA_MIN_IRE       = 12.0;
			double FVF_BLEND_DEV_RATIO            = 1.3;
			double FVF_BLEND_FRAME_BONUS          = 0.25;
			double FVF_BLEND_MIN_FRAME_FRACTION   = 0.20;
			double ONE_D_NEAR_THRESH_IRE          = 1.5;  
            // Neighbor Shaping for FVF (New)
            double FVF_SHAPE_STRENGTH             = 0.5;
            // Neighbor-based cross-domain estimate shaping (FVF)
			double NEIGHBOR_EST_WEIGHT        = 0.125;  // penalty weight (in IRE units); keep small
			double NEIGHBOR_EST_SAT_MAX_IRE   = 12.0; // disable in strong saturation
			double NEIGHBOR_EST_EDGE_MAX_IRE  = 10.0; // only use when horizEdgeIRE below this
			double NEIGHBOR_EST_FVF_MAX_IRE   = 4.0;  // only when FVF diff is small / ambiguous
            // =========================================================================
			// FVF / Model interaction tuning
			// =========================================================================
			
			// How strongly a candidate is penalized for distance from the primary model
			// (Frame in progressive; Field A in interlace). Larger => more model dominance.
			double FVF_MODEL_PRIMARY_WEIGHT   = 0.6;
			
			// Secondary model coherence weight (distance to the "other domain"):
			// - Progressive: distance to best Field
			// - Interlace:   distance to Frame (and/or best field)
			double FVF_MODEL_SECONDARY_WEIGHT = 0.3;
			
			// Small multiplicative advantage to the model domain in its regime.
			// < 1.0 => cheaper score for model candidate.
			double FRAME_MODEL_BIAS  = 0.90;  // progressive: Frame gets small edge
			double FIELD_MODEL_BIAS  = 0.90;  // interlace: Field A gets small edge
			
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
			// 2D Agreement Curve - prev/next get bonus for similarity to 2D
			double AGREEMENT_REWARD_RADIUS_IRE    = 4.5;
			double AGREEMENT_REWARD_MAX           = 2.5;
			double AGREEMENT_VETO_BASE            = 12.0;
			// At the other end, there is a severe deviation penalty
			double deviationThreshold            = 10.0;
			double deviationPenalty               = 3.0;

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
        // In FrameBuffer public section (near split1D / split2D declarations)
		void phaseLocked();  // new: prepares locked-path LO / basis etc.
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

        // Optional temporal context pointers used by vetComposite1D (set by decodeFrames)
        // Not owned — just references to neighboring FrameBuffer objects (may be nullptr).
        const FrameBuffer *prevFrameForVet = nullptr;
        const FrameBuffer *nextFrameForVet = nullptr;

        DemodMode demodMode = DemodMode::Bucket;

        // NEW: Tracks if this frame is the start of a scene (edit boundary).
        bool isSceneStart = false;

    private:
        struct Candidate { double penalty; double sample; };

        const LdDecodeMetaData::VideoParameters &videoParameters;
        const Configuration &configuration;

        qint32 frameHeight = 0;
        double irescale    = 1.0;
        double invIreScale = 1.0;

		 // NEW: Store cadence ID to inform combing decisions (e.g. FVF model)
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

        ComponentFrame *componentFrame = nullptr;
        std::vector<float> demodBurstCos;
        std::vector<float> demodBurstSin;

        // Flat/contiguous buffers (lines x width)
        std::vector<std::array<float,4>> lineRm_locked; // per line: [r00 r01 r10 r11]
        std::vector<float> demodBurstMagRaw;
        std::vector<float> demodTI_flat;
        std::vector<float> demodTQ_flat;
		std::vector<std::vector<double>> simpleField2D;
		std::vector<double> scratch_frameBCenter;
		std::vector<double> scratch_fieldBCenter;        // NEW: raw-composite demod storage (flat contiguous)
        std::vector<float> demodTRI_flat;
        std::vector<float> demodTRQ_flat;
        std::vector<double> scratch_preI;          // unscaled pre-FIR storage (per-line)
        std::vector<double> scratch_preQ;
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
		std::vector<LineAffine> lineAffineLocked;
		std::vector<std::complex<double>> scratch_iq; // reused per-line I/Q scratch (phase-corrected 1D)        
		std::vector<double> scratch_fieldLine;
        std::vector<double> scratch_fieldGate;
        std::vector<double> scratch_frameALine;
        std::vector<double> scratch_fieldBLine;
        std::vector<double> scratch_filter_temp;
        std::vector<double> scratch_hpI;
        std::vector<double> scratch_hpQ;
        std::vector<double> scratch_hpY;
		std::vector<char> scratch_vdis_flag;
		std::vector<std::vector<char>> vdisMask; // [line][rel], persistent per frame

        // Small helpers declared here; definitions provided after the class (in this header).
        inline qint32 getFieldID(qint32 lineNumber) const;
        inline bool   getLinePhase(qint32 lineNumber) const;
		// Per-run 4fsc shifted basis LUT for locked path (phaseCompensation=true)
		double spLUT_locked[4] = {1.0, 0.0, -1.0, 0.0};
		double cpLUT_locked[4] = {0.0, 1.0,  0.0, -1.0};
		bool   basisLockedInit = false;
		bool hasVDIS(int lineNumber, int h) const;		

		// Hybrid 2D helpers
		void computeField2DLine(int lineNumber,
							  double *outFieldLine,
							  double  *outGate);

		void computeSimpleField2DLine(int lineNumber, double *outFieldLine);

		void demodSimpleField2DLine(int line);

        // Frame B: demod/remod version, ±1, skew
		void computeFrameIQLine(int line,
								std::vector<std::complex<double>> &outFrameIQ);
		// Unified VDIS map builder: combines scalar (±2) and IQ (±1) evidence
		// into scratch_vdis_flag for a given line. Does not modify FieldA/FrameB2.
		void computeVDISLine(int lineNumber);
		
		// Minimal Field-vs-Frame scorer: uses normalized FieldB and Frame plus
		// phase-corrected 1D as fallback reference only.
        void scoreFieldVsFrame(
            int line,
            const double *fieldA,
            const double *fieldB,
            const std::vector<double> &frameB2,
            double *outMixed,
            bool writeWeights,
            const double *lateral1D,
            const std::vector<std::complex<double>> *frameIQ = nullptr);
            
            		    		    
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
        inline const float* demodTI_line(int line) const {
            return demodTI_flat.data() + static_cast<size_t>(line) * demodWidth;
        }
        inline const float* demodTQ_line(int line) const {
            return demodTQ_flat.data() + static_cast<size_t>(line) * demodWidth;
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

        // Lightweight 1D vet for a composite sample.
        // Uses only local horizontal neighbours (and optional vertical confirmation)
        // to provide a background-check style acceptance metric for applying a
        // composite-derived change to Y. This deliberately avoids running the full
        // 2D/3D candidate election (getBestCandidate) and is cheap enough to call
        // per-sample if needed.
        struct Vet1DResult {
            double composite_bandpass = 0.0;        // raw - 2D clp (IRE units)
            double leftScore         = std::numeric_limits<double>::infinity();  // smaller is better
            double rightScore        = std::numeric_limits<double>::infinity();  // smaller is better
            int    bestIndex         = -1;         //  -1 = none, 0 = left, 1 = right
            double bestScore         = std::numeric_limits<double>::infinity();
            double confidence        = 0.0;        // 0..1 confidence that replacement is safe
            bool   accept            = false;      // true => safe to apply composite substitution
            int    verticalAgree     = 0;          // number of vertical neighbors that agree (0..2)
            // NEW: diagnostics for adjacent neighbor influence
            int    adjNeighborCount  = 0;          // number of valid immediate neighbors (0..2)
            double adjNeighborSupport= 0.0;        // average agreement of h±1 with residual (0..1)
        };

        // Evaluate local 1D vet for composite at (line, h).
        Vet1DResult vetComposite1D(qint32 line, qint32 h, bool requireVerticalConfirm = false) const;

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