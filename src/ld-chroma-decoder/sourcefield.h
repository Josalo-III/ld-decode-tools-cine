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
