// tools/ld-cinemap/cinemap.h
#pragma once

#include "sourcevideo.h"
#include "vbiprobe.h"

#include <array>
#include <optional>
#include <QString>
#include <QVector>
#include <tuple>
#include <utility>
#include <vector>

class CineDisc;
class LdDecodeMetaData;
class SourceVideo;

class CineMap
{
public:
    enum class Policy {
        Tv,
        Cine
    };

    explicit CineMap(CineDisc* disc, Policy policy = Policy::Tv);

    int detectCadence(const QString& tbcFilePath, double threshold);
    void setVbi(vbiProbe::ProbeResult vbi) { m_vbi = std::move(vbi); }
    void setDecisionTraceEnabled(bool enabled) { m_decisionTraceEnabled = enabled; }
    bool decisionTraceEnabled() const { return m_decisionTraceEnabled; }

private:
    // Internal structs
    struct SegmentCapture {
        int frameIndex = -1;   // 0-based frame index
        int mateField  = -1;   // 1-based field number of the other field in the frame
        bool valid     = false;
    };

    // Cache of frame/field relationships, indexed by 1-based field sequence number.
    struct SegmentCaptureCache {
        int totalFields = 0;
        std::vector<SegmentCapture> cap; // indexed by 1-based field number

        bool validSeq(int seq) const {
            return seq >= 1 && seq < static_cast<int>(cap.size()) && cap[seq].valid;
        }

        // Returns true if two fields belong to the same capture frame.
        bool sameFrame(int seqA, int seqB) const {
            if (!validSeq(seqA) || !validSeq(seqB)) return false;
            return cap[seqA].frameIndex == cap[seqB].frameIndex;
        }
    };
    
    // Per-frame mixedness / combing metric, produced by computeFrameMixedness().
    struct FrameMixedness {
        int frameIndex = -1;   // 0-based frame index within capture
        double score   = 0.0;  // combined notch+lips mixedness metric
    };

    // A candidate twin pair with its diff and confidence score.
    struct TwinEdge {
        int    seqA       = -1;   // 1-based field sequence number
        int    seqB       = -1;   // 1-based field sequence number
        double diff       = 999.0; // raw IRE difference (for reporting / cache lookup)
        double confidence = 0.0;
    };

    // Role of a field within the A or C triple.
    enum class TwinACRole {
        Unknown,
        AType,  // trailing spare: spare follows def (AA -> AB boundary)
        CType   // leading spare:  spare precedes def (BC -> CC boundary)
    };

    // Classified A/C twin triple: definitional, spare, and complement fields.
    struct TwinACInfo {
        int defSeq   = -1;
        int spareSeq = -1;
        int compSeq  = -1;
        TwinACRole role = TwinACRole::Unknown;
    };

    struct PhaseRun {
        enum class Type {
            Unknown,
            Pulldown32,
            Interlaced,
            Progressive
        };

        Type type = Type::Unknown;
        int phaseOffset = 0;
        int endField = 0;
        double confidence = 0.0;
        QString reason;

        // Per-phase mixedness evidence vector (raw scores from scanForPhaseRun).
        // Populated whenever the scan ran with informative data; consumed by
        // solveSegment's evidence-additive election. Empty/zero if not informative.
        std::array<double, 5> phaseScores = {0.0, 0.0, 0.0, 0.0, 0.0};
        bool   phaseScoresInformative = false;
    };

    struct SegmentResult {
        int startField = 0;
        int endField = 0;
        PhaseRun run;
        std::vector<FrameMixedness> mixedness;
    };

    struct Cav5Group {
        int f0 = -1;
        int f1 = -1;
        int f2 = -1;
        int f3 = -1;
        int f4 = -1;
    };

    // Result of validateCavWindowWithDG().
    // cid[frameInGroup][fieldSlot] holds the cadence ID for that field slot (0=first, 1=second).
    // ok replaces 'valid' to match the working source.
    struct CavTwinValidation {
        bool ok = false;
        int  cid[5][2] = {};   // cadence IDs per frame (0..4) per field slot (0..1)
        int  phaseOffset = 0;
        double confidence = 0.0;
    };

    // Result of analyseCavGroup().
    struct CavGroupSignature {
        bool valid             = false;
        bool invertedDomain    = false;
        int  aaTemporalFirstSeq = -1;  // seq number of the temporal-first field of the AA frame
        int  phaseOffset       = 0;
        double confidence      = 0.0;
    };

    // Output of tryLockByDgGeometry().
    struct DgLock {
        int    anchorFrame  = -1;
        int    phaseOffset  = 0;
        int    baseOffset   = 0;
        double confidence   = 0.0;

        // Per-phase brute-geometry evidence vector (normScores from the A/C
        // election). Always populated when there were any acTwins, regardless
        // of whether tryLockByDgGeometry decided to declare a lock — the
        // evidence-additive election in solveSegment consumes it directly,
        // sidestepping the per-detector margin gate.
        std::array<double, 5> phaseScores = {0.0, 0.0, 0.0, 0.0, 0.0};
        bool   phaseScoresInformative = false;
    };

    // Evidence accumulator for validatePhaseGeometry() / gatherGeometryEvidenceForPhase().
    struct GeometryEvidence {
        int typeAgree    = 0;
        int typeDisagree = 0;
        int typeSamples  = 0;
        int gapAgree     = 0;
        int gapDisagree  = 0;
        int gapSamples   = 0;

        bool hasAny()  const { return typeSamples > 0 || gapSamples > 0; }
        int  agree()   const { return typeAgree + gapAgree; }
        int  disagree() const { return typeDisagree + gapDisagree; }
    };

    // Cache keys for diff memoisation.
    struct DiffCacheKey {
        int a, b;
        bool operator==(const DiffCacheKey& o) const { return a == o.a && b == o.b; }
    };
    struct DiffCacheKeyHash {
        size_t operator()(const DiffCacheKey& k) const {
            return std::hash<int>()(k.a) ^ (std::hash<int>()(k.b) << 16);
        }
    };

    struct TwinDemodCacheKey {
        int a, b;
        bool operator==(const TwinDemodCacheKey& o) const { return a == o.a && b == o.b; }
    };
    struct TwinDemodCacheKeyHash {
        size_t operator()(const TwinDemodCacheKey& k) const {
            return std::hash<int>()(k.a) ^ (std::hash<int>()(k.b) << 16);
        }
    };

    static const char* phaseRunTypeName(PhaseRun::Type t);
    static const char* twinRoleName(TwinACRole r);
    static QString phaseRunSummary(const PhaseRun& run);

    bool isValidEvidenceField(int seq) const;
    bool boundaryBetween(int a, int b) const;

    SegmentCaptureCache buildCaptureCache(int hardMaxField);

    // Returns the last field index that should be considered for solving.
    int computeHardMaxField();

    double scoreSpecificPhase(const std::vector<FrameMixedness>& mixed,
                              int phaseOffset,
                              int startField,
                              const SegmentCaptureCache& cache);

    // CAV helpers
    void collectCavTwinPairs(std::vector<Cav5Group>& groups,
                             std::vector<std::pair<int,int>>& pairs);
    std::vector<Cav5Group> identifyCav5Groups();

    double calculateNotchScore(SourceVideo& sv,
                               int f1,
                               int f2,
                               int width,
                               int height) const;

    double calculateLipsScore(SourceVideo& sv,
                              int f1,
                              int f2,
                              int width,
                              int height) const;

    std::vector<FrameMixedness> computeFrameMixedness(SourceVideo& sv,
                                                      int segStart,
                                                      int segEnd);

    std::vector<TwinEdge> harvestTwinEdges(SourceVideo& sv,
                                           int segStart,
                                           int segEnd,
                                           int maxDist);

    int harvestTwinsByPattern(SourceVideo& sv,
                              int segStart,
                              int segEnd,
                              int phaseOffset,
                              const SegmentCaptureCache& cache);

    void collectClvTwinPairsFromMixedness(const std::vector<FrameMixedness>& mixed,
                                          const SegmentCaptureCache& cache,
                                          std::vector<std::pair<int,int>>& outPairs) const;

    void harvestClvTwinsForSegment(SourceVideo& sv,
                                   int segStart,
                                   int segEnd,
                                   const SegmentCaptureCache& cache,
                                   const std::vector<FrameMixedness>& mixedness);

    double dgDiffIre(SourceVideo& sv,
                     int seqA,
                     int seqB,
                     int width,
                     int height);

    double demodTwinDiffCached(SourceVideo& sv,
                               int seq1,
                               int seq2,
                               int width,
                               int height);

    double calculateDemodulatedFieldDiff(SourceVideo& sv,
                                         int f1,
                                         int f2,
                                         int width,
                                         int height);

    double calculateBoostedDemodDiff(SourceVideo& sv,
                                     int f1,
                                     int f2,
                                     int width,
                                     int height);

    // Strips 4fsc subcarrier by bucketing adjacent samples; result written to lumaOut.
    void computeLumaLine_Bucket(const uint16_t* rawLine,
                                std::vector<double>& lumaOut,
                                int width) const;

    double getAdaptiveTwinThreshold(int f1, int f2);

    struct TwinConfDetail {
        double diffIn = 0.0;
        double diffPre = 999.0;
        double diffPost = 999.0;
        double neighborActivity = 999.0;
        double ratio = 999.0;
        double threshAbs = 0.0;
        double ratioScore = 0.0;
        double threshScore = 0.0;
        double confidence = 0.0;
        bool silenceMatch = false;
    };

    double twinConfidence(SourceVideo& sv, int seqA, int seqB);
    double twinConfidence(SourceVideo& sv, int seqA, int seqB, TwinConfDetail& detail);
    int  fieldForFrame(int frameIdx) const;
    void detectCavCadenceBreaks(std::vector<Cav5Group>& groups, SourceVideo& sv);
    void solveCavFallback(SourceVideo& sv);
    int  enforceSteadyCadenceAcrossBoundaries(int maxSpanFields);
    // Attempts to commit a reciprocal doplGang link between fields a and b.
    // cacheOrNull: if provided, enforces strict A/C geometry before committing.
    bool tryCommitReciprocalGang(SourceVideo& sv,
                                 int a,
                                 int b,
                                 double hysteresis,
                                 const SegmentCaptureCache* cacheOrNull);

    void buildTwinEdgesForPairs(SourceVideo& sv,
                                const std::vector<std::pair<int,int>>& pairs,
                                std::vector<TwinEdge>& outEdges,
                                double minConfidence);

    bool hasReciprocalDgEdge(int a, int b) const;

    TwinACInfo classifyTwinAC_strict(int seqA,
                                     int seqB,
                                     const SegmentCaptureCache& cache) const;

    std::vector<TwinACInfo> harvestACTwinsForSegment_strict(int segStartField,
                                                            int segEndField,
                                                            const SegmentCaptureCache& cache) const;

    bool validatePhaseGeometry(int phaseOffset,
                               int segStart,
                               int segEnd,
                               const SegmentCaptureCache& cache,
                               QString* rejectReason = nullptr);

    GeometryEvidence gatherGeometryEvidenceForPhase(int phaseOffset,
                                                    int segStart,
                                                    int segEnd,
                                                    int startFrameIdx,
                                                    const SegmentCaptureCache& cache) const;

    void writeTwinEdgesToMetadata(SourceVideo& sv,
                                  const std::vector<TwinEdge>& edges,
                                  const SegmentCaptureCache* cacheOrNull = nullptr);

    bool tryLockByDgGeometry(SourceVideo& sv,
                             int segStart,
                             int segEnd,
                             const SegmentCaptureCache& cache,
                             DgLock& outLock,
                             QString* rejectReason = nullptr);

    void classifyAsInterlaced(int segStartField,
                              int segEndField,
                              const std::vector<FrameMixedness>& mixedness);

    void classifyAsProgressive(int segStartField,
                               int segEndField,
                               const std::vector<FrameMixedness>& mixedness);

    int healContinuity(std::vector<SegmentResult>& segments,
                       const SegmentCaptureCache& cache);

    void demoteCadenceRange(int startSeq, int endSeq, double newMaxConf);
    void promoteCadenceRange(int startSeq, int endSeq, double newConf);

    CavTwinValidation validateCavWindowWithDG(SourceVideo& sv,
                                              int f0,
                                              int f1,
                                              int f2,
                                              int f3,
                                              int f4);

    CavGroupSignature analyseCavGroup(Cav5Group& g, SourceVideo& sv);

    void solveSegmentCine(SourceVideo& sv,
                          int segStart,
                          int segEnd,
                          const SegmentCaptureCache& cache,
                          const std::vector<FrameMixedness>& mixedness);

    PhaseRun scanForPhaseRun(const std::vector<FrameMixedness>& mixed,
                             int startField,
                             int endField,
                             const SegmentCaptureCache& cache);

    PhaseRun solveSegment(SourceVideo& sv,
                          int segStart,
                          int segEnd,
                          const SegmentCaptureCache& cache,
                          const std::vector<FrameMixedness>& mixedness);

    void applyCadenceToSegment(int segStart,
                               int segEnd,
                               bool isLock,
                               int phaseOffset,
                               int fillCid,
                               double finalConf,
                               const SegmentCaptureCache& cache);

    std::vector<std::pair<int,int>> identifySegments(int hardMaxField);

    int  frameIndexForField(int seq) const;

    int  solveCavDisc();
    void detectAndEncodeInvertedCadenceRuns();
    int  assignPulldownRoles();
    void reconcileDoplGangWithCadence();

    // -------------------------------------------------------------------------
    // Private data members
    // -------------------------------------------------------------------------

    CineDisc*          m_disc = nullptr;  // non-owning
    Policy             m_policy = Policy::Tv;
    LdDecodeMetaData*  m_md   = nullptr;  // non-owning alias of m_disc->getMetaData()
    bool               m_decisionTraceEnabled = false;

    // Per-run sensitivity overrides (0.0 = use defaults)
    double m_notchSensitivity = 1.0;
    double m_twinSensitivity  = 1.0;
    vbiProbe::ProbeResult m_vbi;   // decoded VBI for every frame; set by setVbi()
    // Detected duplicate field links used to derive cadenceId assignments.
    std::vector<std::optional<int>> m_doplGang;        // indexed by 1-based field seq; size = totalFields + 1
    // Internal confidence scores for cadence assignments; not persisted to metadata.
    std::vector<double>             m_cadenceConfidence; // indexed by 1-based field seq; size = totalFields + 1
    // Second order memoisation caches — cleared at the start of each detectCadence() call.
    std::unordered_map<DiffCacheKey,      double, DiffCacheKeyHash>      m_diffCache;
    std::unordered_map<TwinDemodCacheKey, double, TwinDemodCacheKeyHash> m_twinDemodCache;
};
