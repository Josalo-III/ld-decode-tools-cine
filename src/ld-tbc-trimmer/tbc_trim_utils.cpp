/************************************************************************

    tbc_trim_utils.cpp

    Shared TBC trimming utilities for ld-decode tools.
    Copyright (C) 2025-2026 Joseph Burns

************************************************************************/

#include "tbc_trim_utils.h"
#include "vbidecoder.h"
#include <QDebug>
#include <QFileInfo>

namespace TbcTrimUtils {

QVector<FrameInfo> buildFrameList(LdDecodeMetaData *md)
{
    QVector<FrameInfo> frames;
    if (!md) return frames;

    const int nFrames = md->getNumberOfFrames();
    frames.reserve(nFrames);

    VbiDecoder vbiDecoder;

    for (int fi = 0; fi < nFrames; ++fi) {
        const int frameNumber = fi + 1;
        FrameInfo info;
        info.firstFieldNumber  = md->getFirstFieldNumber(frameNumber);
        info.secondFieldNumber = md->getSecondFieldNumber(frameNumber);

        if (info.firstFieldNumber <= 0 || info.secondFieldNumber <= 0) {
            info.isPadded = true;
        } else {
            const auto f1 = md->getField(info.firstFieldNumber);
            const auto f2 = md->getField(info.secondFieldNumber);
            info.isPadded = f1.pad || f2.pad;

            // Detect lead-in/out via VBI
            auto vbi1 = md->getFieldVbi(info.firstFieldNumber).vbiData;
            auto vbi2 = md->getFieldVbi(info.secondFieldNumber).vbiData;
            auto vbi  = vbiDecoder.decodeFrame(vbi1[0], vbi1[1], vbi1[2],
                                               vbi2[0], vbi2[1], vbi2[2]);
            info.isLeadInOut = vbi.leadIn || vbi.leadOut;
        }

        frames.append(info);
    }

    return frames;
}

LdDecodeMetaData *sliceMetadata(LdDecodeMetaData *src,
                                const QVector<FrameInfo> &frames,
                                int startFrame,
                                int length)
{
    if (!src) return nullptr;
    if (startFrame < 0 || length <= 0 || startFrame + length > frames.size()) return nullptr;

    auto *out = new LdDecodeMetaData();
    out->setVideoParameters(src->getVideoParameters());
    out->setPcmAudioParameters(src->getPcmAudioParameters());
    out->setIsFirstFieldFirst(src->getIsFirstFieldFirst());

    for (int i = 0; i < length; ++i) {
        const int srcIdx = startFrame + i;
        const FrameInfo &fi = frames[srcIdx];

        // First field
        if (!fi.isPadded && fi.firstFieldNumber > 0) {
            LdDecodeMetaData::Field f = src->getField(fi.firstFieldNumber);
            f.seqNo = (i * 2) + 1;
            out->appendField(f);
        } else {
            LdDecodeMetaData::Field f;
            f.pad   = true;
            f.seqNo = (i * 2) + 1;
            out->appendField(f);
        }

        // Second field
        if (!fi.isPadded && fi.secondFieldNumber > 0) {
            LdDecodeMetaData::Field f = src->getField(fi.secondFieldNumber);
            f.seqNo = (i * 2) + 2;
            out->appendField(f);
        } else {
            LdDecodeMetaData::Field f;
            f.pad   = true;
            f.seqNo = (i * 2) + 2;
            out->appendField(f);
        }
    }

    return out;
}

bool findStartByCav(LdDecodeMetaData *md,
                    const QVector<FrameInfo> &frames,
                    int cavFrameNum,
                    int &outStart)
{
    if (!md) return false;

    VbiDecoder vbiDecoder;
    const int nFrames = frames.size();

    for (int i = 0; i < nFrames; ++i) {
        const FrameInfo &fi = frames[i];
        if (fi.isPadded || fi.firstFieldNumber <= 0 || fi.secondFieldNumber <= 0) continue;

        auto vbi1 = md->getFieldVbi(fi.firstFieldNumber).vbiData;
        auto vbi2 = md->getFieldVbi(fi.secondFieldNumber).vbiData;
        auto vbi  = vbiDecoder.decodeFrame(vbi1[0], vbi1[1], vbi1[2],
                                           vbi2[0], vbi2[1], vbi2[2]);

        if (vbi.picNo == cavFrameNum) {
            outStart = i;
            return true;
        }
    }

    return false;
}

bool findStartByClv(LdDecodeMetaData *md,
                    const QVector<FrameInfo> &frames,
                    LdDecodeMetaData::ClvTimecode tc,
                    int &outStart)
{
    if (!md) return false;

    VbiDecoder vbiDecoder;
    const int nFrames = frames.size();

    for (int i = 0; i < nFrames; ++i) {
        const FrameInfo &fi = frames[i];
        if (fi.isPadded || fi.firstFieldNumber <= 0 || fi.secondFieldNumber <= 0) continue;

        auto vbi1 = md->getFieldVbi(fi.firstFieldNumber).vbiData;
        auto vbi2 = md->getFieldVbi(fi.secondFieldNumber).vbiData;
        auto vbi  = vbiDecoder.decodeFrame(vbi1[0], vbi1[1], vbi1[2],
                                           vbi2[0], vbi2[1], vbi2[2]);

        LdDecodeMetaData::ClvTimecode current;
        current.hours         = vbi.clvHr;
        current.minutes       = vbi.clvMin;
        current.seconds       = vbi.clvSec;
        current.pictureNumber = vbi.clvPicNo;

        if (current.hours         == tc.hours   &&
            current.minutes       == tc.minutes  &&
            current.seconds       == tc.seconds  &&
            current.pictureNumber == tc.pictureNumber) {
            outStart = i;
            return true;
        }
    }

    return false;
}

QVector<TbcStreamWriter::WriteFrame> buildWriteFrames(const QVector<FrameInfo> &frames,
                                                      int startFrame,
                                                      int length)
{
    QVector<TbcStreamWriter::WriteFrame> writeFrames;
    writeFrames.reserve(length);

    for (int i = 0; i < length; ++i) {
        const FrameInfo &fi = frames[startFrame + i];
        writeFrames.push_back({
            fi.firstFieldNumber,
            fi.secondFieldNumber,
            fi.isPadded
        });
    }

    return writeFrames;
}

QVector<Segment> buildDecomposeSegments(const QVector<FrameInfo> &frames)
{
    QVector<Segment> segments;
    const int n = frames.size();

    int i = 0;
    while (i < n) {
        // Skip padded and lead-in/out frames
        if (frames[i].isPadded || frames[i].isLeadInOut) {
            ++i;
            continue;
        }

        // Start of a contiguous mappable run
        Segment seg;
        seg.startFrame = i;
        seg.hasOverlapAtStart = false;

        while (i < n && !frames[i].isPadded && !frames[i].isLeadInOut) {
            ++i;
        }

        seg.length = i - seg.startFrame;
        if (seg.length > 0)
            segments.append(seg);
    }

    return segments;
}

QVector<Segment> buildDecomposeEditSegments(LdDecodeMetaData *md,
                                            const QVector<FrameInfo> &frames,
                                            QString &outError)
{
    outError.clear();
    QVector<Segment> segments;
    if (!md) return segments;

    const int n = frames.size();

    // Collect cut points: each is a 0-based frame index where a new segment
    // should begin, plus a flag indicating whether it is a half-frame cut.
    struct CutPoint {
        int frameIndex;
        bool isHalfFrame;  // true = second field triggered the cut
    };
    QVector<CutPoint> cuts;

    for (int i = 0; i < n; ++i) {
        const FrameInfo &fi = frames[i];
        if (fi.isPadded || fi.isLeadInOut) continue;
        if (fi.firstFieldNumber <= 0 || fi.secondFieldNumber <= 0) continue;

        const auto &f1 = md->getField(fi.firstFieldNumber);
        const auto &f2 = md->getField(fi.secondFieldNumber);

        // Check first field: clean cut — new segment starts at this frame
        if (f1.cinemap.inUse && f1.cinemap.isEditBoundary && f1.isFirstField) {
            cuts.append({ i, false });
        }
        // Check second field: half-frame cut — new segment starts at this frame
        // (which is also the last frame of the previous segment)
        else if (f2.cinemap.inUse && f2.cinemap.isEditBoundary && !f2.isFirstField) {
            cuts.append({ i, true });
        }
    }

    if (cuts.isEmpty()) {
        outError = "No edit boundary frames found. Run ld-cinemap to establish edit points before using --decompose-edits.";
        return segments;
    }

    // Find the first non-lead-in-out, non-padded frame as the implicit start
    int contentStart = 0;
    while (contentStart < n &&
           (frames[contentStart].isLeadInOut || frames[contentStart].isPadded)) {
        ++contentStart;
    }

    // Find the last non-lead-in-out, non-padded frame as the implicit end (exclusive)
    int contentEnd = n - 1;
    while (contentEnd >= 0 &&
           (frames[contentEnd].isLeadInOut || frames[contentEnd].isPadded)) {
        --contentEnd;
    }
    ++contentEnd; // make exclusive

    if (contentStart >= contentEnd) {
        outError = "No mappable content frames found.";
        return segments;
    }

    // Build segments between cut points.
    // The implicit segment list starts at contentStart and ends at contentEnd,
    // with each cut point opening a new segment.
    // Cuts that are not at contentStart are added; contentStart is always the
    // start of segment 0.

    // Collect all segment start positions (some may coincide — deduplicate)
    QVector<CutPoint> starts;
    starts.append({ contentStart, false });
    for (const CutPoint &cp : cuts) {
        if (cp.frameIndex > contentStart && cp.frameIndex < contentEnd) {
            // Avoid duplicating if the same frame appears twice
            if (starts.isEmpty() || starts.last().frameIndex != cp.frameIndex)
                starts.append(cp);
        }
    }

    for (int s = 0; s < starts.size(); ++s) {
        const int segStart = starts[s].frameIndex;
        const bool overlap = starts[s].isHalfFrame && s > 0;

        // Segment runs to just before the next cut point, or to contentEnd
        int segEnd = (s + 1 < starts.size()) ? starts[s + 1].frameIndex : contentEnd;

        // If the next cut is a half-frame cut, the boundary frame belongs to
        // both this segment and the next — extend this segment to include it.
        if (s + 1 < starts.size() && starts[s + 1].isHalfFrame) {
            segEnd = starts[s + 1].frameIndex + 1;
        }

        Segment seg;
        seg.startFrame       = segStart;
        seg.length           = segEnd - segStart;
        seg.hasOverlapAtStart = overlap;

        if (seg.length > 0)
            segments.append(seg);
    }

    return segments;
}

DecomposeStats writeSegments(LdDecodeMetaData *src,
                             const QVector<FrameInfo> &frames,
                             const QVector<Segment> &segments,
                             const QString &outputStem,
                             const QFileInfo &inputFile,
                             int fieldWidth,
                             int fieldHeight)
{
    DecomposeStats stats;

    for (int s = 0; s < segments.size(); ++s) {
        const Segment &seg = segments[s];

        // Name: stem_001.tbc (overflows to 4+ digits gracefully beyond 999)
        const QString segName = QString("%1_%2.tbc")
                                    .arg(outputStem)
                                    .arg(s + 1, 3, 10, QChar('0'));
        const QString segDbName = segName + ".db";

        qInfo() << "Writing segment" << (s + 1) << ":" << segName
                << "(" << seg.length << "frames)";

        // Write metadata
        LdDecodeMetaData *segMd = sliceMetadata(src, frames, seg.startFrame, seg.length);
        if (!segMd) {
            qWarning() << "Failed to slice metadata for segment" << (s + 1) << "- skipping";
            continue;
        }
        segMd->write(segDbName);
        delete segMd;

        // Write video
        TbcStreamWriter::Config config;
        config.writeVideo    = true;
        config.writeAudio    = false;
        config.writeMetadata = false;  // already written above
        config.fieldWidth    = fieldWidth;
        config.fieldHeight   = fieldHeight;

        const auto writeFrames = buildWriteFrames(frames, seg.startFrame, seg.length);
        if (!TbcStreamWriter::write(writeFrames, src,
                                    inputFile,
                                    QFileInfo(segName),
                                    config)) {
            qWarning() << "Write failed for segment" << (s + 1);
        }

        ++stats.segmentCount;
        if (seg.hasOverlapAtStart) {
            ++stats.overlapSegments;
            // Each overlap event repeats one unique frame (the boundary frame)
            // across two segments. overlapFrameCount counts unique repeated frames,
            // so we increment once per overlap event regardless of which segment
            // we are on. We track this on the receiving segment (hasOverlapAtStart)
            // so we count each event exactly once.
            ++stats.overlapFrameCount;
        }
    }

    return stats;
}

} // namespace TbcTrimUtils
