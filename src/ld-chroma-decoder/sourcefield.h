/************************************************************************

    sourcefield.h

    ld-chroma-decoder - Colourisation filter for ld-decode
    Copyright (C) 2019 Adam Sampson

    This file is part of ld-decode-tools.

    ld-chroma-decoder is free software: you can redistribute it and/or
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

#ifndef SOURCEFIELD_H
#define SOURCEFIELD_H

#include "lddecodemetadata.h"
#include "sourcevideo.h"

// A field read from the input, with metadata and data
struct SourceField {
    LdDecodeMetaData::Field field;
    SourceVideo::Data data;
    qint32 capturePartnerSeqNo = -1;  // seqNo of the other field in the original TBC frame
    // Exact-carrier side channel from the cadence assembler's dG merge.
    // For a def field whose spare twin passed the merge sanity check, this
    // holds, per sample, the carrier of the EMITTED (merged) sample:
    //     exact = merged - (def + spare)/2
    // The twin sum is exact luma and the twin difference exact carrier by
    // conservation (same film content, opposite subcarrier), so this is a
    // measurement, not an estimate. Empty when no twin existed or sanity
    // failed; NaN outside the processed active region. Consumers must treat
    // it as evidence/calibration only -- it covers half the lines of A/C
    // film frames and would alternate if rendered directly.
    QVector<float> dgExactCarrier;

    // Sync-tone tracker payload (cadence assembler -> decoder head).
    // Per-region [predicted rotation since the previous dG anchor (rad),
    // confidence 0..1], row-major over field-raster regions of
    // kSyncRegLines x kSyncRegCols. Region row r covers frame lines
    // 32r..32r+31 (16 field lines); region col c covers active samples
    // 128c..128c+127 -- the same lattice the decoder's star/sync grids
    // use, so no remapping is needed at consumption. Empty = no tracker
    // authority for this field.
    static constexpr int kSyncRegLines = 16;
    static constexpr int kSyncRegCols  = 128;
    QVector<float> dgSyncIncrement;
    // Cadence identity remains in field.cinemap for diagnostics, but fallback
    // work must not use that identity as permission for FVF's progressive
    // frame regime. Guarded frame candidates remain part of the field regime.
    bool allowProgressiveFrameRegime = true;

    static void loadFields(SourceVideo &sourceVideo, LdDecodeMetaData &ldDecodeMetaData,
                           qint32 firstFrameNumber, qint32 numFrames,
                           qint32 lookBehindFrames, qint32 lookAheadFrames,
                           QVector<SourceField> &fields, qint32 &startIndex, qint32 &endIndex);

    qint32 getOffset() const {
        return field.isFirstField ? 0 : 1;
    }

    qint32 getFirstActiveLine(const LdDecodeMetaData::VideoParameters &videoParameters) const {
        return (videoParameters.firstActiveFrameLine + 1 - getOffset()) / 2;
    }
    qint32 getLastActiveLine(const LdDecodeMetaData::VideoParameters &videoParameters) const {
        return (videoParameters.lastActiveFrameLine + 1 - getOffset()) / 2;
    }
};

#endif
