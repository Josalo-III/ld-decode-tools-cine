// tools/ld-cinemap/cinedisc.cpp
#include "cinedisc.h"

#include "lddecodemetadata.h"

#include <QDebug>
#include <QFileInfo>

#include <memory>

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
    // CineDisc interface
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

    // refreshFrameCache() is a no-op for CineDiscMeta: we hold no derived
    // frame-number cache of our own.  The base-class default handles this.

private:
    QString                           m_tbcPath;
    bool                              m_reverseFieldOrder = false;
    std::unique_ptr<LdDecodeMetaData> m_md;
    int                               m_videoFieldLength = 0;
    int                               m_numberOfFrames = 0;
    bool                              m_isDiscPal = false;
    bool                              m_isDiscCav = false;
};

// -----------------------------------------------------------------------------
// Factory
// -----------------------------------------------------------------------------

std::unique_ptr<CineDisc> loadCineDisc(const QString& tbcPath,
                                       bool reverseFieldOrder)
{
    return CineDiscMeta::load(tbcPath, reverseFieldOrder);
}
