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

    // Instrument: dump the per-pair twin measurement for every same-parity d=2
    // pair in [startField, endField] as CSV on stdout. Read-only — no solve, no
    // metadata write. Exists so the metric can be measured on real material
    // before it is wired into any decision.
    int probeDgRange(const QString& tbcFilePath, int startField, int endField);

    // Instrument: calibrate the disc's twin/noise floor and report it. If ranges is
    // non-empty ("a-b,c-d,..."), also report each range's duty at both operating
    // points. Read-only — no solve, no metadata write.
    int probeDgFloor(const QString& tbcFilePath, const QString& ranges);

    // Instrument: per-frame comb on two axes, to separate real inter-field motion
    // from vertical image detail.
    //
    // notchWithin  = notch(first(N),   second(N))   — the two fields of one frame
    // notchAcross  = notch(first(N+1), second(N))   — spans the frame boundary
    //
    // Both calls keep the same parity roles, so their vertical geometry is identical
    // and the image's own vertical detail enters both equally. It is therefore
    // common-mode and cancels in the ratio, while temporal motion does not:
    // progressive unique frames have no motion within a frame and a full step across
    // one (strong period-2 alternation), 59.94i spans one field time either way
    // (flat), and 3:2 film shows its 5-frame cycle. Read-only.
    int probeCombAxes(const QString& tbcFilePath, int startField, int endField);

    // Instrument: relative cost of notch vs lips over identical frames with a warm
    // field cache, plus the cold field-I/O cost both sit on top of.
    int benchComb(const QString& tbcFilePath, int startField, int endField);
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
        double score   = 0.0;  // lips mixedness: does this frame comb (detail masked)
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

    // Demodulated measurement of a same-parity d=2 field pair.
    //
    // At d=2 the subcarrier phase is inverted between the two fields — a property
    // of the NTSC 4-field sequence, not of content — so the pair decomposes into
    // two complementary channels (writing field2's carrier as -C2'):
    //
    //   D = f1 - f2 : broadband = luma DIFFERENCE   coherent 4fsc = C1 + C2' (mean chroma)
    //   S = f1 + f2 : broadband = mean luma         coherent 4fsc = C1 - C2' (chroma DIFFERENCE)
    //
    // A telecine twin is the same film frame scanned twice: its luma difference
    // carries no animated grain (grainIre collapses) AND its chroma cancels in the
    // sum (sCarrier collapses). The two are independent evidence — grain animation
    // and chroma disagreement are different physics with different blind spots.
    //
    // dCarrier is the POWER METER for the cancellation test: with no chroma present
    // there is nothing to cancel and q carries no information, which is a distinct
    // state from "chroma present and it failed to cancel".
    struct TwinDemod {
        double grainIre    = 1000.0; // RMS(D - carrier): broadband luma difference + animated grain
        double dCarrierIre = 0.0;    // |C1 + C2'| per-sample RMS: mean chroma amplitude
        double sCarrierIre = 0.0;    // |C1 - C2'| per-sample RMS: chroma disagreement
        double dCohIre     = 0.0;    // as dCarrier, coherently pooled over runs (rejects luma leak)
        double sCohIre     = 0.0;    // as sCarrier, coherently pooled over runs
        bool   valid       = false;

        // Cancellation defect: |tan(theta/2)| of the chroma phase disagreement
        // between the two fields. Zero for a twin. Dimensionless — it carries its
        // own scale, so unlike an IRE floor it cannot be miscalibrated.
        double q() const    { return (dCarrierIre > 0.0) ? (sCarrierIre / dCarrierIre) : 999.0; }
        double qCoh() const { return (dCohIre     > 0.0) ? (sCohIre     / dCohIre)     : 999.0; }
    };

    // The disc's own twin/noise floor, measured from its own data.
    //
    // A twin's field difference contains ONLY video noise, so the floor is a
    // property of the disc and its decode chain, not of content. It is measured,
    // never predicted: across five discs the observed floors clustered 0.71-0.96
    // IRE (1.35x spread) while bPSNR-implied noise amplitude spanned 1.01-2.24 IRE
    // (2.2x) — so scaling a threshold by bPSNR is steeper than reality by roughly a
    // squaring and makes the estimate worse than a constant would.
    //
    // Measured as the low mode of per-block minima. Block minima, not raw pairs:
    // floor-sitting content (telecine twins, static shots) is localised, so a block
    // that contains any of it reports the floor while a block of pure motion reports
    // its own motion level. concentration is the share of blocks agreeing on that
    // value — the floor is a sharp spike when it is real, so a low concentration
    // means no floor was found and the calibration must say so rather than invent one.
    struct NoiseFloor {
        double ire           = 0.0;  // the calibrated floor
        double p10           = 0.0;  // percentiles of the per-block minima
        double p25           = 0.0;
        double median        = 0.0;
        double concentration = 0.0;  // share of block minima within +/-20% of ire
        double bpsnrIre      = 0.0;  // bPSNR-implied noise, for cross-check ONLY
        int    blocks        = 0;
        int    pairs         = 0;
        bool   valid         = false;
    };

    // Operating points on the calibrated floor.
    //
    // Two different questions read the same measurement at different generosity.
    // The recall question — "are there any twins here, or may we stop looking" — is
    // deliberately liberal: progressive material stays at exactly zero duty out to
    // 3x the floor, so admitting more costs nothing, while a missed twin means
    // incorrect operations downstream in chroma decoder. Non-terminal negatives may
    // be wrong; they only trigger a harder look.
    //
    // The geometry question — "which phase" — uses the tight point, because surplus
    // hits blur the spacing-5 / alternating-parity structure that carries the phase
    // (at 1.75x a low-motion telecine shot inflates to 46% duty against the ~20%
    // that 3:2 predicts).
    static constexpr double FLOOR_MULT_RECALL   = 2.00;
    static constexpr double FLOOR_MULT_GEOMETRY = 1.35;

    // Lips owns mixedness, and unlike notch its zero is absolute: lips is a residual
    // measured after masking the image's own vertical detail per pixel and after
    // subtracting its own noise floor, so lips ~ 0 means "no comb" rather than merely
    // "no vertical structure". That is what makes these two absolute constants
    // legitimate where notch-scaled ones were not.
    //
    // Measured across five discs:
    //   progressive / clean telecine frames  0.0002 - 0.008
    //   noise-only windows (max over window) up to 0.028
    //   genuine 59.94i frames                0.065  - 2.06
    //   genuine locks (max over window)      >= 0.189
    //   telecine mixed positions (cad 2,4)   3.3    - 25.3
    //
    // LIPS_SILENCE gates a whole window: below it nothing in the window combs at all,
    // so the segment carries no phase information and must abstain rather than let
    // the percentile stretch invent one.
    // LIPS_COMB is the per-frame question "does this frame comb", used by the
    // 59.94i / progressive classifiers and by the CLV mixed-frame harvest.
    static constexpr double LIPS_SILENCE = 0.05;
    static constexpr double LIPS_COMB    = 0.05;

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

    // Full two-channel measurement (cached). demodTwinDiffCached is the grainIre
    // projection of this, kept so existing callers read unchanged.
    const TwinDemod& demodTwinCached(SourceVideo& sv,
                                     int seq1,
                                     int seq2,
                                     int width,
                                     int height);

    TwinDemod calculateDemodulatedFieldDiff(SourceVideo& sv,
                                            int f1,
                                            int f2,
                                            int width,
                                            int height);

    // Measures the disc's twin/noise floor once and caches it. See NoiseFloor.
    const NoiseFloor& calibrateTwinFloor(SourceVideo& sv);

    // Share of same-parity d=2 pairs in [startField, endField] sitting at
    // floorIre. 3:2 pulldown puts one twin per parity per 5-frame cycle, so a
    // telecined segment lands near 1/5; progressive material lands at 0; content
    // with no motion at all lands near 1, which is the absence of information
    // rather than evidence of twins.
    double twinDuty(SourceVideo& sv,
                    int startField,
                    int endField,
                    double floorIre,
                    int* outHits = nullptr,
                    int* outPairs = nullptr,
                    double* outQuietestIre = nullptr,
                    std::vector<int>* outHitFields = nullptr);

    // Share of floor hits that fall on the best pair of offsets (o, o+5) mod 10.
    //
    // 3:2 pulldown puts one twin per parity per 5-frame cycle, so EVERY twin lands on
    // exactly two offsets mod 10, five apart and of opposite parity. Real telecine
    // therefore scores ~1.0 while unstructured content scores at chance, which for
    // two of ten offsets is 0.2.
    //
    // This replaced an existence test ("are there any two hits five apart") that was
    // effectively vacuous: at 12 hits among 59 positions, chance alone supplies about
    // two such pairs, so the test passed on material with no cadence at all. Measured
    // on DS9-BTS, genuine telecine scored 1.00 while video scored 0.21-0.42.
    //
    // This matters most for footage that ORIGINATED on film but was composited
    // through a video pipeline: the cadence does not survive that path, so there is
    // nothing to recover and the material must read as video. What is being detected
    // is a recoverable field-repeat structure, not film origin.
    double cycleConcentration(const std::vector<int>& hitFields) const;

    // Minimum concentration to accept the hits as a cadence.
    //
    // Measured across four discs: telecine 0.50-1.00, chance (unstructured hits
    // spread over ten offsets) 0.20-0.21, video 0.00. The low end of the telecine
    // range is low-motion shots, where the whole pair population sits near the floor
    // so even the tight point admits some non-twins, and those scatter and dilute the
    // concentration. 0.40 clears chance by 2x and sits under the weakest measured
    // telecine with headroom.
    static constexpr double CYCLE_CONCENTRATION_MIN = 0.40;

    // A video sentinel may only be asserted over a well-sampled span. Below this
    // coverage the span is dark, not video, and darkness must bridge rather than tag.
    static constexpr double MIN_TAG_COVERAGE = 0.75;

    // True when the twin census has positively ruled film out over [segStart,segEnd]:
    // no concentrated cadence, and enough usable pairs to mean it. Only then may the
    // 59.94i / progressive fork be asked.
    //
    // The asymmetry is the whole point. Tagging a film-dominant shot as video makes
    // 24p export drop frames at effectively random positions, which destroys the shot
    // for the restorer's later composite; the reverse error is recoverable by a
    // second pass with --set-cadence. So film must be positively excluded, never
    // merely unproven, before any video sentinel is written.
    bool filmRuledOut(SourceVideo& sv, int segStart, int segEnd);

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

        // Chroma-cancellation channel (see TwinDemod). Measured for the pair and
        // its two same-parity d=2 neighbors, so the same relative reading the
        // grain channel uses is available here too.
        double qIn      = 999.0;
        double qPre     = 999.0;
        double qPost    = 999.0;
        double powerIn  = 0.0;   // dCarrier of the pair: did the test have power?
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

    // Fraction of the twin sites a phase PREDICTS that actually sit at the floor.
    //
    // A projected phase names the pairs that must be twins — roughly two per 5-frame
    // cycle — so it can be checked directly instead of asking mixedness whether it
    // looks plausible. One measurement per site against the generous floor, which for
    // a sub-60-field rescue is a dozen or so pairs, nearly all of them already cached.
    //
    // Returns -1.0 when the test had no power (too few measurable sites), which is
    // darkness and must fall back to presumption rather than reject.
    double verifyPhaseByTwins(SourceVideo& sv,
                              int segStart,
                              int segEnd,
                              int phaseOffset,
                              const SegmentCaptureCache& cache);

    // Majority of predicted sites must actually be twins. Perfect telecine gives 1.0;
    // this leaves room for dropouts and pad fields without admitting a phase whose
    // predicted sites are mostly empty.
    static constexpr double PHASE_VERIFY_MIN = 0.50;

    int healContinuity(SourceVideo& sv,
                       std::vector<SegmentResult>& segments,
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
    std::unordered_map<TwinDemodCacheKey, TwinDemod, TwinDemodCacheKeyHash> m_twinDemodCache;
    NoiseFloor m_noiseFloor;   // measured once per run by calibrateTwinFloor()
};
