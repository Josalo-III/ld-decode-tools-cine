// tools/ld-cinemap/cinedisc.cpp
#include "cinedisc.h"

#include "cadencedefs.h"
#include "lddecodemetadata.h"
#include "tbc/logging.h"

#include <algorithm>
#include <memory>
#include <QFileInfo>
#include <unordered_set>


namespace {

    // Parse seqNo keys from comma-separated string, optionally supporting ranges (e.g., "10-20")
    static std::vector<qint32> parseSeqNoKeys(const QString& text, bool allowRanges)
    {
        std::vector<qint32> out;
        if (text.trimmed().isEmpty()) return out;
    
        const QStringList items = text.split(',', Qt::SkipEmptyParts);
        for (QString item : items) {
            item = item.trimmed();
            if (item.isEmpty()) continue;
    
            if (allowRanges && item.contains('-')) {
                const QStringList parts = item.split('-', Qt::SkipEmptyParts);
                if (parts.size() != 2) continue;
    
                bool okA = false, okB = false;
                qint32 a = parts[0].trimmed().toInt(&okA);
                qint32 b = parts[1].trimmed().toInt(&okB);
                if (!okA || !okB) continue;
                if (a <= 0 || b <= 0) continue;
    
                if (a > b) std::swap(a, b);
                for (qint32 v = a; v <= b; ++v) out.push_back(v);
            } else {
                bool ok = false;
                qint32 v = item.toInt(&ok);
                if (!ok || v <= 0) continue;
                out.push_back(v);
            }
        }
    
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }

    static QString pulldownRoleForCadenceId(int cid)
    {
        if (!cadenceKnown(cid)) return QString();
        if (isDefinitionalRole(cid)) return QStringLiteral("definitional");
        if (isSpareRole(cid)) return QStringLiteral("spare");
        return QString();
    }

    static bool parseCadenceOverrideSpec(const QString& text, qint32& firstField, qint32& lastField, int& firstCid)
    {
        const QString spec = text.trimmed();
        const int colon = spec.indexOf(':');
        if (colon <= 0 || colon == spec.size() - 1) return false;

        const QString rangeText = spec.left(colon).trimmed();
        const QString cidText = spec.mid(colon + 1).trimmed();
        const int dash = rangeText.indexOf('-');
        if (dash <= 0 || dash == rangeText.size() - 1) return false;

        bool okFirst = false, okLast = false, okCid = false;
        firstField = rangeText.left(dash).trimmed().toInt(&okFirst);
        lastField = rangeText.mid(dash + 1).trimmed().toInt(&okLast);
        firstCid = cidText.toInt(&okCid);

        if (!okFirst || !okLast || !okCid) return false;
        if (firstField <= 0 || lastField <= 0) return false;
        if (firstField > lastField) std::swap(firstField, lastField);

        if (firstCid == CADENCE_UNKNOWN ||
            firstCid == CADENCE_VIDEO ||
            firstCid == CADENCE_PROGRESSIVE) {
            return true;
        }

        return (firstCid >= 0 &&
                firstCid < CADENCE_NTSC_INVERTED_OFFSET + CADENCE_NTSC_CYCLE);
    }

    static int cadenceIdAtOffset(int firstCid, int offset)
    {
        if (!cadenceKnown(firstCid)) return firstCid;

        const int base = cadenceIsInverted(firstCid) ? CADENCE_NTSC_INVERTED_OFFSET : 0;
        int idx = cadenceIndex(firstCid);
        idx = (idx + offset) % CADENCE_NTSC_CYCLE;
        if (idx < 0) idx += CADENCE_NTSC_CYCLE;
        return base + idx;
    }

} // namespace

class CineDiscMeta : public CineDisc
{
public:
    CineDiscMeta(const QString& tbcPath, bool reverseFieldOrder)
        : m_tbcPath(tbcPath)
        , m_reverseFieldOrder(reverseFieldOrder)
        , m_md(new LdDecodeMetaData)
    {
    }

    static std::unique_ptr<CineDiscMeta> load(const QString& tbcPath,
                                              bool reverseFieldOrder)
    {
        const QString dbPath = tbcPath + ".db";
        if (!QFileInfo::exists(dbPath)) {
            qCritical() << "CineDiscMeta: metadata file not found:" << dbPath;
            return nullptr;
        }

        auto disc = std::unique_ptr<CineDiscMeta>(
            new CineDiscMeta(tbcPath, reverseFieldOrder));

        if (!disc->m_md->read(dbPath)) {
            qCritical() << "CineDiscMeta: failed to read metadata from" << dbPath;
            return nullptr;
        }

        const int nFrames = disc->m_md->getNumberOfFrames();
        if (nFrames < 2) {
            qCritical() << "CineDiscMeta: metadata contains only"
                        << nFrames << "frame(s) — too small to process.";
            return nullptr;
        }
        if (nFrames > 108000) {
            qCritical() << "CineDiscMeta: metadata contains"
                        << nFrames << "frames — exceeds maximum supported size.";
            return nullptr;
        }

        const auto vp = disc->m_md->getVideoParameters();
        disc->m_videoFieldLength = vp.fieldWidth * vp.fieldHeight;

        // Use the VideoSystem enum rather than the non-existent isSourcePal field.
        // PAL_M is a 525-line system and is treated as non-PAL for cadence purposes.
        disc->m_isDiscPal      = (vp.system == PAL);
        disc->m_numberOfFrames = nFrames;

        return disc;
    }

    // -------------------------------------------------------------------------
    // CineDisc interface implementation
    // -------------------------------------------------------------------------

    LdDecodeMetaData& getMetaData() override
    {
        return *m_md;
    }

    const LdDecodeMetaData& getMetaData() const override
    {
        return *m_md;
    }

    const QString& getTbcPath() const override
    {
        return m_tbcPath;
    }

    int getVideoFieldLength() const override
    {
        return m_videoFieldLength;
    }

    bool isDiscPal() const override
    {
        return m_isDiscPal;
    }

    bool isDiscCav() const override
    {
        return m_isDiscCav;
    }

    void setIsDiscCav(bool cav) override
    {
        m_isDiscCav = cav;
    }

    int getNumberOfFrames() const override
    {
        return m_numberOfFrames;
    }

    int getFirstFieldNumber(int frameNumber) const override
    {
        return m_md->getFirstFieldNumber(frameNumber);
    }

    int getSecondFieldNumber(int frameNumber) const override
    {
        return m_md->getSecondFieldNumber(frameNumber);
    }

    bool isPadded(int frameIndex) const override
    {
        const int frameNumber = frameIndex + 1;
        const int f1 = m_md->getFirstFieldNumber(frameNumber);
        const int f2 = m_md->getSecondFieldNumber(frameNumber);

        if (f1 <= 0 || f2 <= 0)
            return false;

        const auto a = m_md->getField(f1);
        const auto b = m_md->getField(f2);
        return a.pad || b.pad;
    }

    bool getReverseFieldOrder() const override
    {
        return m_reverseFieldOrder;
    }

    QString getFilename() const override
    {
        return m_tbcPath;
    }

    // -------------------------------------------------------------------------
    // Edit Whitelist/Blacklist Implementation
    // -------------------------------------------------------------------------

    int applyEditWhitelistSeqNoKeys(const QString& csvSeqNoList) override
    {
        const auto keys = parseSeqNoKeys(csvSeqNoList, /*allowRanges=*/false);
        return applyEditOverridesBySeqNoKeys(keys, /*value=*/true);
    }

    int applyEditBlacklistSeqNoKeys(const QString& csvSeqNoListWithRanges) override
    {
        const auto keys = parseSeqNoKeys(csvSeqNoListWithRanges, /*allowRanges=*/true);
        return applyEditOverridesBySeqNoKeys(keys, /*value=*/false);
    }

    int applyCadenceOverrideFieldRange(const QString& rangeSpec) override
    {
        qint32 firstField = 0, lastField = 0;
        int firstCid = CADENCE_UNKNOWN;
        if (!parseCadenceOverrideSpec(rangeSpec, firstField, lastField, firstCid)) {
            qWarning() << "Invalid cadence override spec:" << rangeSpec
                       << "(expected fieldStart-fieldEnd:cadenceId)";
            return -1;
        }

        const qint32 totalFields = m_md->getNumberOfFields();
        int changed = 0;
        int cadenceOffset = 0;

        for (qint32 idx = firstField; idx <= lastField && idx <= totalFields; ++idx) {
            const auto field = m_md->getField(idx);
            if (field.pad) continue;

            const int cid = cadenceIdAtOffset(firstCid, cadenceOffset++);

            auto modifiedField = field;
            modifiedField.cinemap.cadenceId = cid;
            modifiedField.cinemap.cadenceIndexPresumed = false;
            modifiedField.cinemap.pulldownRole = pulldownRoleForCadenceId(cid);
            modifiedField.cinemap.isManualOverride = true;
            m_md->updateField(modifiedField, idx);
            changed++;
        }

        return changed;
    }

private:
    // Sets isEditBoundary and isManualOverride on all fields matching seqNoKeys.
    int applyEditOverridesBySeqNoKeys(const std::vector<qint32>& seqNoKeys, bool value)
    {
        if (seqNoKeys.empty()) return 0;

        const qint32 totalFields = m_md->getNumberOfFields();
        int changed = 0;

        const std::unordered_set<qint32> wanted(seqNoKeys.begin(), seqNoKeys.end());

        for (qint32 idx = 1; idx <= totalFields; ++idx) {
            const auto field = m_md->getField(idx);

            if (wanted.find(field.seqNo) == wanted.end()) continue;

            // Skip pad fields (unusual to override but harmless)
            if (field.pad) continue;

            auto modifiedField = field;
            modifiedField.cinemap.isEditBoundary = value;
            modifiedField.cinemap.isManualOverride = true;
            m_md->updateField(modifiedField, idx);
            changed++;
        }

        return changed;
    }

    // -------------------------------------------------------------------------
    // Private member variables
    // -------------------------------------------------------------------------

    QString                           m_tbcPath;
    bool                              m_reverseFieldOrder = false;
    std::unique_ptr<LdDecodeMetaData> m_md;
    int                               m_videoFieldLength = 0;
    int                               m_numberOfFrames = 0;
    bool                              m_isDiscPal = false;
    bool                              m_isDiscCav = false;
};

// -------------------------------------------------------------------------
// Factory
// -------------------------------------------------------------------------

std::unique_ptr<CineDisc> loadCineDisc(const QString& tbcPath,
                                       bool reverseFieldOrder)
{
    return CineDiscMeta::load(tbcPath, reverseFieldOrder);
}
