#pragma once

#include <deque>
#include <map>
#include <optional>
#include <QHash>
#include <QSet>
#include <QVector>

#include "cadencedefs.h"
#include "lddecodemetadata.h"
#include "sourcefield.h"

class CadenceAssembler
{
public:
    struct Configuration {
        bool noCinemap = false;
        bool dgDiscard = false;
        bool export24p = false;
        int setCadence = 0;
		bool reverseFieldOrder = false;
		bool emitMax24p = false;
        double dgMaxOutlierFrac   = 0.10;
    };

    struct WorkItem {
        enum class Kind {
            PassthroughFrame,
            FilmFrame,
            TelecineFrame
        };

        enum class Expansion {
            None,
            Trailing,
            Leading
        };

        Kind kind = Kind::PassthroughFrame;
        Expansion expansion = Expansion::None;

        bool fieldsSwapped = false;
        bool invertedFieldOrder = false;
        bool temporalFirstIsF1  = true; // true if temporal order is f1 then f2 after any comb swap/policy

        SourceField f1;
        SourceField f2;

        char filmLabel = '?';
    };
    
    CadenceAssembler(const LdDecodeMetaData::VideoParameters& vp,
                 const Configuration& cfg,
                 std::function<void(qint32)> onBaseline = nullptr
                 );

    void push(const QVector<SourceField>& newFields);

    // Sync-tone tracker: sequential per-region alpha-beta on the certified
    // regional phase, fed at each dG merge (the anchor), twin-integrity
    // gated, cut-reset. Predictions are stamped on every emitted field as
    // INCREMENTS since the previous anchor -- convention-free at the
    // consumer, which composes them with its own in-batch anchor
    // measurement.
    struct SyncTrk {
        bool valid = false;
        double zI = 1.0, zQ = 0.0;   // phase state at last anchor
        double omega = 0.0;          // rad per field
        double missEwma = 0.5;       // rad, innovation magnitude history
    };
    void flush();
    bool hasWork() const;
    QVector<WorkItem> popWork();

private:
	std::function<void(qint32)> onFieldReleasedToBaseline;
    std::vector<SyncTrk> syncTrk;
    SyncTrk syncGlobal;
    int syncRegX = 0, syncRegY = 0;
    long syncAnchorSeq = -1;
    long syncCuts = 0;
    void syncTrackerUpdate(int anchorSeq, double twinDriftDeg,
                           const std::vector<double>& regI,
                           const std::vector<double>& regQ,
                           const std::vector<long>& regN,
                           double ireScale);
    void syncStamp(SourceField& f) const;
    LdDecodeMetaData::VideoParameters videoParameters;
    Configuration config;
    std::deque<SourceField> window;
    std::deque<WorkItem> workQueue;
	 // history model: full field timeline with consumption tracking.
	struct HistoryField {
			SourceField field;
			bool consumed = false;
			int capturePartnerSeqNo = -1;
		};
		QSet<qint32> baselineOwnedSeqNos;
		inline void releaseSeqToBaseline(qint32 seqNo) {
			if (onFieldReleasedToBaseline) onFieldReleasedToBaseline(seqNo);
		}
		inline void releaseToBaseline(SourceField&& sf) {
			releaseSeqToBaseline(sf.field.seqNo);
		}
	enum class FilmFieldRole { Def, Comp, Spare, Other };
	
	static inline FilmFieldRole classifyRole(int cid) {
		if (!cadenceKnown(cid)) return FilmFieldRole::Other;
		const int idx = cadenceIndex(cid);
		if (idx == 0 || idx == 7) return FilmFieldRole::Def;   // Adef / Cdef
		if (idx == 1 || idx == 6 || idx == 8) return FilmFieldRole::Comp; // Acomp / Ccomp / Dcomp
		if (idx == 2 || idx == 5) return FilmFieldRole::Spare; // Aspare / Cspare
		// 3,4,9 = mixed/other roles; treat them as Other here
		return FilmFieldRole::Other;
	}
		
	QHash<int, int> seqNoToHistoryIndex; // seqNo -> index in history

    QVector<HistoryField> history;
    int cursor = 0;             // index of earliest field not yet fully committed

    // Forced-cadence: counts TBC *fields* consumed in override mode
    qint64 forcedFieldIndex = 0;

    static bool boundaryBetweenFields(const SourceField& prev, const SourceField& next);
    static bool knownCadence(const SourceField& sf) { return sf.field.cinemap.cadenceId >= 0; }
	int findComplementPos(int i0) const;
    bool orderPairForComb(SourceField& a, SourceField& b) const;

    static bool isDefByRoleString(const SourceField& sf) { return sf.field.cinemap.pulldownRole == "definitional"; }
    bool mergeDgPairWithSanityWrapper(SourceField& def, SourceField& spare, SourceField& comp);

    // Auto cadence path
    int  nextUnconsumedIndex(int start) const;
    int  countUnconsumedFrom(int start, int maxCount) const;
    bool tryExtractFilmFrameAtCursor();
	bool tryEmitPassthroughAtCursor(bool flushMode, bool force = false);
    void processHistory(bool flushMode);

    // NEW: Fully disentangled forced cadence pipeline
    void processWindowForced(bool flushMode);

	    // Mapping helper: setCadence (1..5) -> starting offset in 0..9
	    int forcedStartIndex() const;
	        // Helper to mark a history index consumed and remove its seqNo mapping.
	    // Always use this instead of directly assigning history[pos].consumed = true,
	    // because the seq->index map must be updated to avoid stale lookups.
	    void markHistoryConsumed(int pos);
	    // Drop a consumed entry's sample planes. Must follow any move out of
	    // history, never be folded into markHistoryConsumed.
	    void releaseHistoryPayload(int pos);
		void handOffCaptureFrameToBaseline(int pos);
	};
