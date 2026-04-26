/************************************************************************

    tbc_trim_utils.h

    Shared TBC trimming utilities for ld-decode tools.
    Operates directly on LdDecodeMetaData; no DiscMap dependency.
    Copyright (C) 2025-2026 Joseph Burns

    This file is part of ld-decode-tools.

    ld-decode-tools is free software: you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

************************************************************************/

#ifndef TBC_TRIM_UTILS_H
#define TBC_TRIM_UTILS_H

#include "lddecodemetadata.h"
#include "tbcwriter.h"

#include <QString>
#include <QVector>

namespace TbcTrimUtils {

// A lightweight frame descriptor derived directly from LdDecodeMetaData,
// sufficient for field lookup and padded-frame detection without DiscMap.
struct FrameInfo {
    int firstFieldNumber  = -1;  // 1-based source field index; -1 if padded
    int secondFieldNumber = -1;  // 1-based source field index; -1 if padded
    bool isPadded         = false;
    bool isLeadInOut      = false;
};

// A segment descriptor produced by the decompose functions.
struct Segment {
    int startFrame = 0;   // 0-based index into the FrameInfo list
    int length     = 0;   // number of frames in this segment
    bool hasOverlapAtStart = false;  // true if this segment's first frame is
                                     // shared with the preceding segment due
                                     // to a half-frame edit boundary
};

// Statistics returned by decompose operations.
struct DecomposeStats {
    int segmentCount       = 0;  // total segments written
    int overlapFrameCount  = 0;  // number of unique frames that appear in two segments
    int overlapSegments    = 0;  // number of segments that received an overlap frame
};

// Build a FrameInfo list for all frames in the source metadata.
// frameNumber is 1-based throughout, matching LdDecodeMetaData conventions.
QVector<FrameInfo> buildFrameList(LdDecodeMetaData *md);

// Produce a standalone LdDecodeMetaData covering frames [startFrame, startFrame+length).
// Fields are renumbered 1..N in the output so it can be written as a self-contained file.
// The caller owns the returned pointer.
// Returns nullptr if the range is invalid.
LdDecodeMetaData *sliceMetadata(LdDecodeMetaData *src,
                                const QVector<FrameInfo> &frames,
                                int startFrame,
                                int length);

// Find the 0-based frame index of the first frame whose CAV picture number
// (VBI picNo) matches cavFrameNum.
// Returns true and sets outStart on success.
bool findStartByCav(LdDecodeMetaData *md,
                    const QVector<FrameInfo> &frames,
                    int cavFrameNum,
                    int &outStart);

// Find the 0-based frame index of the first frame whose CLV timecode matches tc.
// Returns true and sets outStart on success.
bool findStartByClv(LdDecodeMetaData *md,
                    const QVector<FrameInfo> &frames,
                    LdDecodeMetaData::ClvTimecode tc,
                    int &outStart);

// Build a WriteFrame list for the given slice of frames, ready for TbcStreamWriter.
// Uses source field numbers directly — does not renumber.
QVector<TbcStreamWriter::WriteFrame> buildWriteFrames(const QVector<FrameInfo> &frames,
                                                      int startFrame,
                                                      int length);

// Build segment descriptors by splitting on padded/lead-in-out boundaries.
// Each contiguous run of non-padded, non-lead-in-out frames becomes one segment.
QVector<Segment> buildDecomposeSegments(const QVector<FrameInfo> &frames);

// Build segment descriptors by splitting on cinemap edit boundaries.
// Requires at least one frame whose first or second field has cinemap.inUse &&
// isEditBoundary == true; returns an empty vector and sets error if none found.
//
// When the triggering field is a second field (isFirstField == false), the
// containing frame is included in both the closing segment and the opening
// segment (hasOverlapAtStart = true on the new segment).
//
// Lead-in/out frames are excluded. Padded frames remain in their segment.
//
// Returns an empty vector and sets outError on failure.
QVector<Segment> buildDecomposeEditSegments(LdDecodeMetaData *md,
                                            const QVector<FrameInfo> &frames,
                                            QString &outError);

// Write all segments to disk. outputStem is the base path without extension
// (e.g. "/path/to/disc"); outputs are named disc_001.tbc, disc_001.tbc.db, etc.
// Returns stats for the end report.
DecomposeStats writeSegments(LdDecodeMetaData *src,
                             const QVector<FrameInfo> &frames,
                             const QVector<Segment> &segments,
                             const QString &outputStem,
                             const QFileInfo &inputFile,
                             int fieldWidth,
                             int fieldHeight);

} // namespace TbcTrimUtils

#endif // TBC_TRIM_UTILS_H
