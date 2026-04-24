 #pragma once

#include <QString>
#include <memory>

class LdDecodeMetaData;

class CineDisc
{
public:
    virtual ~CineDisc() = default;

    // Metadata access
    virtual LdDecodeMetaData& getMetaData() = 0;
    virtual const LdDecodeMetaData& getMetaData() const = 0;

    // IO / geometry
    virtual const QString& getTbcPath() const = 0;
    virtual int getVideoFieldLength() const = 0;

    // Disc traits
    virtual bool isDiscPal() const = 0;
    virtual bool isDiscCav() const = 0;
    virtual void setIsDiscCav(bool cav) = 0;

    // Frame / field mapping.
    // frameNumber is 1-based, matching LdDecodeMetaData conventions.
    virtual int getNumberOfFrames() const = 0;
    virtual int getFirstFieldNumber(int frameNumber) const = 0;
    virtual int getSecondFieldNumber(int frameNumber) const = 0;

    // Frame properties.
    // frameIndex is 0-based.
    virtual bool isPadded(int frameIndex) const = 0;
    virtual bool getReverseFieldOrder() const = 0;

    // Misc / compatibility
    virtual QString getFilename() const = 0;

    // Edit whitelist/blacklist - impose or remove edits
    // comma delimited lists; blacklist accepts ranges
    virtual int applyEditWhitelistSeqNoKeys(const QString& csvSeqNoList) = 0;
    virtual int applyEditBlacklistSeqNoKeys(const QString& csvSeqNoListWithRanges) = 0;

    // No-op hook for implementations that maintain a derived frame cache.
    // CineDiscMeta has no such cache; this exists so CineMap can call it
    // unconditionally after bulk metadata writes (matching DiscMap usage).
    virtual void refreshFrameCache() {}
};

// Factory: load metadata from tbcPath + ".json" and construct a CineDisc.
std::unique_ptr<CineDisc> loadCineDisc(const QString& tbcPath,
                                       bool reverseFieldOrder);