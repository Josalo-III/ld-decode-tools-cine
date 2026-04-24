/************************************************************************

    sourcefield.cpp

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

#include "sourcefield.h"

#include "sourcevideo.h"

void SourceField::loadFields(SourceVideo &sourceVideo, LdDecodeMetaData &ldDecodeMetaData,
                             qint32 firstFrameNumber, qint32 numFrames,
                             qint32 lookBehindFrames, qint32 lookAheadFrames,
                             QVector<SourceField> &fields, qint32 &startIndex, qint32 &endIndex)
{
    const LdDecodeMetaData::VideoParameters &videoParameters = ldDecodeMetaData.getVideoParameters();

    startIndex = 2 * lookBehindFrames;
    endIndex = startIndex + (2 * numFrames);
    fields.resize(endIndex + (2 * lookAheadFrames));

    const qint32 numInputFrames = ldDecodeMetaData.getNumberOfFrames();
    qint32 frameNumber = firstFrameNumber - lookBehindFrames;
    for (qint32 i = 0; i < fields.size(); i += 2) {

        const bool useBlankFrame = frameNumber < 1 || frameNumber > numInputFrames;

        qint32 firstFieldNumber  = ldDecodeMetaData.getFirstFieldNumber(useBlankFrame ? 1 : frameNumber);
        qint32 secondFieldNumber = ldDecodeMetaData.getSecondFieldNumber(useBlankFrame ? 1 : frameNumber);

        fields[i].field     = ldDecodeMetaData.getField(firstFieldNumber);
        fields[i + 1].field = ldDecodeMetaData.getField(secondFieldNumber);

        // Store original TBC capture partner seqNo on each field.
        // This is the ground truth for reconstruction — derived once here
        // from the metadata frame map, carried forward from this point.
        fields[i].capturePartnerSeqNo     = useBlankFrame ? -1 : secondFieldNumber;
        fields[i + 1].capturePartnerSeqNo = useBlankFrame ? -1 : firstFieldNumber;

        const quint16 black = videoParameters.black16bIre;

        if (useBlankFrame) {
            fields[i].data.fill(black, sourceVideo.getFieldLength());
            fields[i + 1].data.fill(black, sourceVideo.getFieldLength());
        } else {
            fields[i].data     = sourceVideo.getVideoField(firstFieldNumber);
            fields[i + 1].data = sourceVideo.getVideoField(secondFieldNumber);

            if ((videoParameters.system == PAL || videoParameters.system == PAL_M) &&
                videoParameters.isSubcarrierLocked) {
                fields[i + 1].data.remove(0, 2);
                for (int j = 0; j < 2; j++) {
                    fields[i + 1].data.append(black);
                }
            }
        }

        frameNumber++;
    }
}
