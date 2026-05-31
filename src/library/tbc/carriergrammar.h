/******************************************************************************
 * carriergrammar.h
 * ld-decode-tools shared composite carrier grammar definitions
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * CarrierGrammarState is the standalone line-level carrier coordinate system
 * used by attribution and decoder stages.  It records phase, sign, authority,
 * burst calibration, and span parse context; it does not decide luma/chroma
 * attribution by itself.
 ******************************************************************************/

#pragma once

#include <array>
#include <vector>

namespace lddecode {

// Shared cadence sentinels used by current decoders.
inline constexpr int kCadenceVideo = -2;
inline constexpr int kCadenceProgressive = -3;

// Carrier sign is state, not a local repair.  Call sites must know the signal
// frame of their input before deciding whether carrierLineFlip is needed.
enum class CarrierSignFrame {
    UnsignedBucket,           // (h + samplePhase0) & 3; no polarity in values
    BurstLockedSigned,        // burst phasor + lineFlip baked in; do not re-sign
    MetadataPreservedSigned,  // parser/metadata polarity preserved; do not re-sign
    Grid4fsc,                 // cross-line analysis; producer declares sign state
};

// Source priority for lineFlip and carrier polarity.
enum class CarrierPhaseAuthority {
    Metadata,       // capture metadata / field cadence; highest priority
    BurstMeasured,  // per-line burst measurement; calibration evidence
    RigidSchedule,  // rigid NTSC derivation; fallback / diagnostic only
};

struct CarrierGrammarSpan {
    int x0 = 0;
    int x1 = 0;

    // Pattern evidence within the locked grammar.
    double carrierFit            = 0.0;
    double samePhaseRecurrence   = 0.0;
    double alternationCoherence  = 0.0;

    // IQ / forward-model adjudication.
    double iqEnvelopeCoherence   = 0.0;
    double remodAgreement        = 0.0;
    double forwardErrorIRE       = 0.0;

    // Counter-evidence and arbitration.
    double lumaCounterEvidence   = 0.0;
    double parserConflict        = 0.0;

    // Final carrier-pattern claim projected into attribution evidence.
    double chromaClaim           = 0.0;
};

// Per-line locked-basis affine (phase-clamped rotation+gain).
struct CarrierGrammarAffine {
    double R[2][2] = {{1.0, 0.0}, {0.0, 1.0}};
    bool   valid   = false;
};

struct CarrierGrammarState {
    // Identity and schedule context.
    int  line                    = 0;
    int  fieldPhaseId            = 0;
    int  lineParity              = 0;
    bool frameVerticalAllowed    = false;

    // Carrier grammar lock.
    double burstCos              = 1.0;
    double burstSin              = 0.0;
    double carrierScale          = 0.0;  // IRE
    double phaseError            = 0.0;  // carrier-phase radians
    double phaseConfidence       = 0.0;  // [0,1]

    int samplePhase0             = 0;
    int lineFlip                 = +1;
    bool grammarLocked           = false;

    // Authority and schedule-conflict diagnostics.
    CarrierPhaseAuthority lineFlipAuthority  = CarrierPhaseAuthority::Metadata;
    int    rigidScheduleLineFlip             = +1;
    double phaseScheduleConflict             = 0.0;

    // Implementation-specific carrier basis for the line.
    std::array<float,4> demodLUTTi = {};
    std::array<float,4> demodLUTTq = {};
    CarrierGrammarAffine affine    = {};

    // Line-level carrier projection summary.
    double carrierModelGain   = 1.0;
    double carrierModelDCIRE  = 0.0;
    double carrierFitRatio     = 0.0;
    double meanForwardErrorIRE = 0.0;
    double meanChromaMagIRE    = 0.0;
    bool   projectionValid     = false;

    // Residual decomposition summary after the current Y estimate.
    double residualCarrierFitRatio     = 0.0;
    double residualMeanForwardErrorIRE = 0.0;
    double residualMeanChromaMagIRE    = 0.0;
    double residualForwardAgreement    = 0.0;

    // Line-level summaries.
    double linePatternConfidence   = 0.0;
    double stableChromaConfidence  = 0.0;
    double iqValidationConfidence  = 0.0;
    double forwardAgreement        = 0.0;

    std::vector<CarrierGrammarSpan> spans;
};

// Policy for merging the two per-field cadence IDs into one frame-level mode.
// Defaults preserve current chroma-decoder behavior.
struct CadenceMergePolicy {
    int videoCadenceId = kCadenceVideo;
    int progressiveCadenceId = kCadenceProgressive;
};

// Rules for converting field phase IDs into per-line grammar sign.
// Defaults preserve the current NTSC schedule interpretation.
struct CarrierGrammarSchedulePolicy {
    int positiveOnEvenPhaseIdA = 1;
    int positiveOnEvenPhaseIdB = 4;
    int defaultSamplePhase0 = 0;
    CarrierPhaseAuthority defaultAuthority = CarrierPhaseAuthority::Metadata;
};

// Frame-level helpers for cadence/model selection and grammar initialization.
int mergeCadenceIdForInterleavedFrame(int cadenceA, int cadenceB, bool editSplit);
int mergeCadenceIdForInterleavedFrame(int cadenceA,
                                      int cadenceB,
                                      bool editSplit,
                                      const CadenceMergePolicy &policy);
int phaseIdForInterleavedLine(int lineNumber,
                              int firstFieldPhaseId,
                              int secondFieldPhaseId);
CarrierGrammarState makeCarrierGrammarStateForLine(int lineNumber,
                                                   int firstFieldPhaseId,
                                                   int secondFieldPhaseId,
                                                   bool frameVerticalAllowed);
CarrierGrammarState makeCarrierGrammarStateForLine(
    int lineNumber,
    int firstFieldPhaseId,
    int secondFieldPhaseId,
    bool frameVerticalAllowed,
    const CarrierGrammarSchedulePolicy &policy);
void initializeCarrierGrammarSchedule(std::vector<CarrierGrammarState> &carrierGrammar,
                                      int firstLine,
                                      int lastLine,
                                      int firstFieldPhaseId,
                                      int secondFieldPhaseId,
                                      bool frameVerticalAllowed);
void initializeCarrierGrammarSchedule(
    std::vector<CarrierGrammarState> &carrierGrammar,
    int firstLine,
    int lastLine,
    int firstFieldPhaseId,
    int secondFieldPhaseId,
    bool frameVerticalAllowed,
    const CarrierGrammarSchedulePolicy &policy);

} // namespace lddecode
