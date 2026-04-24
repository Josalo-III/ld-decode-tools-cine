// tools/ld-cinemap/cinedisc.cpp
#include "cinedisc.h"

#include "lddecodemetadata.h"

#include <QDebug>
#include <QFileInfo>
#include <algorithm>
#include <unordered_set>

#include <memory>

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
        const QString jsonPath = tbcPath + ".db";
        if (!QFileInfo::exists(jsonPath)) {
            qCritical() << "CineDiscMeta: metadata file not found:" << jsonPath;
            return nullptr;
        }

        auto disc = std::unique_ptr<CineDiscMeta>(
            new CineDiscMeta(tbcPath, reverseFieldOrder));

        if (!disc->m_md->read(jsonPath)) {
            qCritical() << "CineDiscMeta: failed to read metadata from" << jsonPath;
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

private:
    // -------------------------------------------------------------------------
    // Private implementation
    // -------------------------------------------------------------------------

    // Apply edit overrides (whitelist=true for force-on, blacklist=false for force-off)
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

            // Create a modified field with the override
            auto modifiedField = field;
			modifiedField.cinemap.isEditBoundary = value;
			modifiedField.cinemap.isManualOverride = true;
            // Use updateField to persist the override
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