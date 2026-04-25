/************************************************************************

    discmap.cpp

    ld-discmap - TBC and VBI alignment and correction
    Copyright (C) 2019-2025 Simon Inns

    This file is part of ld-decode-tools.

    ld-discmap is free software: you can redistribute it and/or
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

#include "discmap.h"
#include "tbc/logging.h"

DiscMap::DiscMap(const QFileInfo &metadataFileInfo, const bool reverseFieldOrder,
                 const bool noStrict)
    : m_metadataFileInfo(metadataFileInfo),
      m_reverseFieldOrder(reverseFieldOrder),
      m_noStrict(noStrict)
{
    m_tbcValid = true;
    ldDecodeMetaData = new LdDecodeMetaData;

    if (!ldDecodeMetaData->read(metadataFileInfo.filePath())) {
        tbcDebugStream() << "Cannot load metadata from" << metadataFileInfo.filePath();
        m_tbcValid = false;
        return;
    }

    if (m_reverseFieldOrder) ldDecodeMetaData->setIsFirstFieldFirst(false);
    else ldDecodeMetaData->setIsFirstFieldFirst(true);

    m_numberOfFrames = ldDecodeMetaData->getNumberOfFrames();

    if (m_numberOfFrames < 2) {
        tbcDebugStream() << "Metadata contains only" << m_numberOfFrames << "frames - too small";
        m_tbcValid = false;
        return;
    }

    if (m_numberOfFrames > 108000) {
        // 108000 frames = 60 minutes at 30fps (NTSC) — beyond any real LaserDisc side
        tbcDebugStream() << "Metadata contains" << m_numberOfFrames << "frames - too big";
        m_tbcValid = false;
        return;
    }

    m_videoFieldLength = ldDecodeMetaData->getVideoParameters().fieldWidth *
            ldDecodeMetaData->getVideoParameters().fieldHeight;

    m_frames.resize(m_numberOfFrames);

    VbiDecoder vbiDecoder;
    QVector<VbiDecoder::Vbi> vbiData(m_numberOfFrames);
    for (qint32 frameNumber = 0; frameNumber < m_numberOfFrames; frameNumber++) {
        m_frames[frameNumber].seqFrameNumber(frameNumber + 1);
        m_frames[frameNumber].firstField(ldDecodeMetaData->getFirstFieldNumber(frameNumber + 1));
        m_frames[frameNumber].secondField(ldDecodeMetaData->getSecondFieldNumber(frameNumber + 1));

        auto vbi1 = ldDecodeMetaData->getFieldVbi(ldDecodeMetaData->getFirstFieldNumber(frameNumber + 1)).vbiData;
        auto vbi2 = ldDecodeMetaData->getFieldVbi(ldDecodeMetaData->getSecondFieldNumber(frameNumber + 1)).vbiData;
        vbiData[frameNumber] = vbiDecoder.decodeFrame(vbi1[0], vbi1[1], vbi1[2], vbi2[0], vbi2[1], vbi2[2]);

        if (vbiData[frameNumber].leadIn || vbiData[frameNumber].leadOut)
            m_frames[frameNumber].isLeadInOrOut(true);
        else
            m_frames[frameNumber].isLeadInOrOut(false);
    }

    m_videoSystemDescription = ldDecodeMetaData->getVideoSystemDescription();
    if (ldDecodeMetaData->getVideoParameters().system == PAL) m_isDiscPal = true;
    else if (ldDecodeMetaData->getVideoParameters().system == NTSC) m_isDiscPal = false;
    else {
        tbcDebugStream() << "Input TBC video system" << m_videoSystemDescription << "is not supported";
        qCritical("Video system must be PAL or NTSC");
    }

    if (m_isDiscPal) {
        // PAL: 44100 Hz / 50 fields/s = 882 samples/field; 882 * 4 bytes (16-bit stereo) = 3528
        m_audioFieldByteLength   = 3528;
        m_audioFieldSampleLength = 882;
    } else {
        // NTSC: 44100 Hz / 60 fields/s = 735 samples/field; 736 * 4 bytes = 2944 (rounded up for alignment)
        m_audioFieldByteLength   = 2944;
        m_audioFieldSampleLength = 736;
    }

    qint32 framesToCheck = 100;
    if (m_numberOfFrames < framesToCheck) framesToCheck = m_numberOfFrames;
    tbcDebugStream() << "Checking first" << framesToCheck << "sequential frames for disc CAV/CLV type determination";

    qint32 cavCount = 0;
    qint32 clvCount = 0;
    for (qint32 frameNumber = 0; frameNumber < framesToCheck; frameNumber++) {
        if (vbiData[frameNumber].picNo > 0) cavCount++;
        if (vbiData[frameNumber].clvHr != -1 && vbiData[frameNumber].clvMin != -1 &&
                vbiData[frameNumber].clvSec != -1 && vbiData[frameNumber].clvPicNo != -1) clvCount++;
    }

    if (cavCount == 0 && clvCount == 0) {
        tbcDebugStream() << "Source does not seem to contain valid CAV picture numbers or CLV time-codes - cannot map";
        m_tbcValid = false;
        return;
    }

    if (cavCount > clvCount) {
        m_isDiscCav = true;
        m_discType = "CAV";
        tbcDebugStream() << "Got" << cavCount << "valid CAV picture numbers from" << framesToCheck << "frames - source disc type is CAV";
    } else {
        m_isDiscCav = false;
        m_discType = "CLV";
        tbcDebugStream() << "Got" << clvCount << "valid CLV picture numbers from" << framesToCheck << "frames - source disc type is CLV";
    }

    if (m_isDiscCav) tbcDebugStream() << "Storing VBI CAV picture numbers as frame numbers";
    else tbcDebugStream() << "Converting VBI CLV timecodes into frame numbers";

    qint32 iecOffset = -1;
    for (qint32 frameNumber = 0; frameNumber < m_numberOfFrames; frameNumber++) {
        if (!m_isDiscCav) {
            LdDecodeMetaData::ClvTimecode clvTimecode;
            clvTimecode.hours         = vbiData[frameNumber].clvHr;
            clvTimecode.minutes       = vbiData[frameNumber].clvMin;
            clvTimecode.seconds       = vbiData[frameNumber].clvSec;
            clvTimecode.pictureNumber = vbiData[frameNumber].clvPicNo;
            m_frames[frameNumber].vbiFrameNumber(ldDecodeMetaData->convertClvTimecodeToFrameNumber(clvTimecode));

            if (!m_isDiscPal) {
                // NTSC CLV discs using the IEC Amendment 2 encoding insert an extra
                // frame number at specific positions in the timecode sequence. Track
                // how many such offsets have been seen so that frame numbers can be
                // corrected back to a contiguous sequence.
                if (isNtscAmendment2ClvFrameNumber(m_frames[frameNumber].vbiFrameNumber() - iecOffset)) {
                    m_frames[frameNumber].isClvOffset(true);
                    iecOffset++;
                }
            }
        } else {
            m_frames[frameNumber].vbiFrameNumber(vbiData[frameNumber].picNo);
        }
    }

    m_numberOfPulldowns = 0;
    if (!m_isDiscPal && m_isDiscCav) {
        tbcDebugStream() << "Disc type is NTSC CAV - checking for pull-down frames";

        for (qint32 frameNumber = 0; frameNumber < m_numberOfFrames; frameNumber++) {
            bool isPulldown = false;

            if (m_frames[frameNumber].vbiFrameNumber() == -1 && !m_frames[frameNumber].isLeadInOrOut()) {
                qint32 lastPhase2 = -1;
                if (frameNumber > 0) lastPhase2 = ldDecodeMetaData->getField(
                            ldDecodeMetaData->getSecondFieldNumber(frameNumber)).fieldPhaseID;

                qint32 currentPhase1 = ldDecodeMetaData->getField(ldDecodeMetaData->getFirstFieldNumber(frameNumber + 1)).fieldPhaseID;
                qint32 currentPhase2 = ldDecodeMetaData->getField(ldDecodeMetaData->getSecondFieldNumber(frameNumber + 1)).fieldPhaseID;

                qint32 nextPhase1 = -1;
                if (frameNumber < m_numberOfFrames - 1) nextPhase1 = ldDecodeMetaData->getField(
                            ldDecodeMetaData->getFirstFieldNumber(frameNumber + 2)).fieldPhaseID;

                qint32 expectedLastPhase, expectedNextPhase, expectedIntraPhase;

                if (!m_reverseFieldOrder) {
                    expectedLastPhase  = currentPhase1 - 1;
                    if (expectedLastPhase == 0) expectedLastPhase = 4;
                    expectedNextPhase  = currentPhase2 + 1;
                    if (expectedNextPhase == 5) expectedNextPhase = 1;
                    expectedIntraPhase = currentPhase1 + 1;
                    if (expectedIntraPhase == 5) expectedIntraPhase = 1;
                } else {
                    expectedLastPhase  = currentPhase1 + 1;
                    if (expectedLastPhase == 5) expectedLastPhase = 1;
                    expectedNextPhase  = currentPhase2 - 1;
                    if (expectedNextPhase == 0) expectedNextPhase = 4;
                    expectedIntraPhase = currentPhase1 - 1;
                    if (expectedIntraPhase == 0) expectedIntraPhase = 4;
                }

                if (currentPhase2 == expectedIntraPhase) {
                    if (lastPhase2 == expectedLastPhase || lastPhase2 == -1) {
                        if (nextPhase1 == expectedNextPhase || nextPhase1 == -1) {
                            isPulldown = true;
                        } else {
                            tbcDebugStream() << "Seq. frame" << m_frames[frameNumber].seqFrameNumber() << "is not in phase sequence with the subsequent frame!";
                        }
                    } else {
                        tbcDebugStream() << "Seq. frame" << m_frames[frameNumber].seqFrameNumber() << "is not in phase sequence with the preceding frame!";
                    }
                } else {
                    tbcDebugStream() << "Seq. frame" << m_frames[frameNumber].seqFrameNumber() << "has an incorrect intra-frame phaseID!";
                }

                if (isPulldown) {
                    qint32 doubleCheckCounter = 0;
                    if (frameNumber > 5) {
                        if (vbiData[frameNumber - 5].picNo == -1) doubleCheckCounter++;
                    }
                    if (frameNumber < m_numberOfFrames - 5) {
                        if (vbiData[frameNumber + 5].picNo == -1) doubleCheckCounter++;
                    }

                    if (doubleCheckCounter < 1) {
                        if (!m_noStrict) {
                            tbcDebugStream() << "Seq. frame" << m_frames[frameNumber].seqFrameNumber()
                                        << "looks like a pull-down, but there is no pull-down sequence in the surrounding frames - marking as false-positive";
                            isPulldown = false;
                        } else {
                            tbcDebugStream() << "Seq. frame" << m_frames[frameNumber].seqFrameNumber()
                                        << "looks like a pull-down, but there is no pull-down sequence in the surrounding frames"
                                        << "- strict checking is disabled, so marking as pulldown anyway";
                            isPulldown = true;
                        }
                    }

                    m_frames[frameNumber].isPullDown(isPulldown);
                    if (m_frames[frameNumber].isPullDown()) m_numberOfPulldowns++;
                }
            }
        }
    }

    tbcDebugStream() << "Performing a frame quality analysis for each frame";
    for (qint32 frameNumber = 0; frameNumber < m_numberOfFrames; frameNumber++) {
        double penaltyPercent = 0;
        if (frameNumber < m_numberOfFrames - 1) {
            if (vbiData[frameNumber + 1].picNo < vbiData[frameNumber].picNo) penaltyPercent = 80.0;
            else penaltyPercent = 100.0;
        }

        double bsnr = (ldDecodeMetaData->getFieldVitsMetrics(ldDecodeMetaData->getFirstFieldNumber(frameNumber + 1)).bPSNR +
                ldDecodeMetaData->getFieldVitsMetrics(ldDecodeMetaData->getSecondFieldNumber(frameNumber + 1)).bPSNR) / 2.0;
        double blackSnrLinear = pow(bsnr / 20, 10);
        double snrReferenceLinear = pow(43.0 / 20, 10);
        double bsnrPercent = (100.0 / snrReferenceLinear) * blackSnrLinear;
        if (bsnrPercent > 100.0) bsnrPercent = 100.0;

        qint32 totalDotsInFrame = (ldDecodeMetaData->getVideoParameters().fieldHeight * 2) +
                                   ldDecodeMetaData->getVideoParameters().fieldWidth;
        DropOuts dropOuts1 = ldDecodeMetaData->getFieldDropOuts(ldDecodeMetaData->getFirstFieldNumber(frameNumber + 1));
        DropOuts dropOuts2 = ldDecodeMetaData->getFieldDropOuts(ldDecodeMetaData->getSecondFieldNumber(frameNumber + 1));

        qint32 frameDoLength = 0;
        for (qint32 i = 0; i < dropOuts1.size(); i++) frameDoLength += dropOuts1.endx(i) - dropOuts1.startx(i);
        for (qint32 i = 0; i < dropOuts2.size(); i++) frameDoLength += dropOuts2.endx(i) - dropOuts2.startx(i);

        double frameDoPercent = 100.0 - (static_cast<double>(frameDoLength) / static_cast<double>(totalDotsInFrame));

        qint32 syncConfPercent = (ldDecodeMetaData->getField(ldDecodeMetaData->getFirstFieldNumber(frameNumber + 1)).syncConf +
                                  ldDecodeMetaData->getField(ldDecodeMetaData->getSecondFieldNumber(frameNumber + 1)).syncConf) / 2;

        // Composite quality score weighted heavily towards dropout coverage (x1000),
        // with secondary contributions from bSNR, sequence continuity penalty, and
        // sync confidence. Normalised to a 0–100 range.
        m_frames[frameNumber].frameQuality((bsnrPercent + penaltyPercent + static_cast<double>(syncConfPercent) + (frameDoPercent * 1000.0)) / 1004.0);
    }

    for (qint32 frameNumber = 0; frameNumber < m_numberOfFrames; frameNumber++) {
        m_frames[frameNumber].firstFieldPhase(ldDecodeMetaData->getField(ldDecodeMetaData->getFirstFieldNumber(frameNumber + 1)).fieldPhaseID);
        m_frames[frameNumber].secondFieldPhase(ldDecodeMetaData->getField(ldDecodeMetaData->getSecondFieldNumber(frameNumber + 1)).fieldPhaseID);
    }
}

DiscMap::~DiscMap()
{
    delete ldDecodeMetaData;
}

QDebug operator<<(QDebug dbg, const DiscMap &discMap)
{
    dbg.nospace().noquote()
        << "DiscMap(Frames " << discMap.numberOfFrames()
        << ", disc type is " << discMap.discType()
        << ", video format is " << discMap.discFormat()
        << ", detected " << discMap.numberOfPulldowns() << " pulldown frames)";
    return dbg.maybeSpace();
}

QString DiscMap::filename() const          { return m_metadataFileInfo.filePath(); }
bool    DiscMap::valid() const             { return m_tbcValid; }
qint32  DiscMap::numberOfFrames() const    { return m_numberOfFrames; }
bool    DiscMap::isDiscCav() const         { return m_isDiscCav; }
bool    DiscMap::isDiscPal() const         { return m_isDiscPal; }
QString DiscMap::discType() const          { return m_discType; }
QString DiscMap::discFormat() const        { return m_videoSystemDescription; }
qint32  DiscMap::numberOfPulldowns() const { return m_numberOfPulldowns; }

qint32 DiscMap::vbiFrameNumber(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        tbcDebugStream() << "vbiFrameNumber out of frameNumber range";
        return -1;
    }
    return m_frames[frameNumber].vbiFrameNumber();
}

void DiscMap::setVbiFrameNumber(qint32 frameNumber, qint32 vbiFrameNumber)
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        tbcDebugStream() << "setVbiFrameNumber out of frameNumber range";
        return;
    }
    m_frames[frameNumber].vbiFrameNumber(vbiFrameNumber);
}

qint32 DiscMap::seqFrameNumber(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        tbcDebugStream() << "seqFrameNumber out of frameNumber range";
        return -1;
    }
    return m_frames[frameNumber].seqFrameNumber();
}

bool DiscMap::isPulldown(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        tbcDebugStream() << "isPulldown out of frameNumber range";
        return false;
    }
    return m_frames[frameNumber].isPullDown();
}

bool DiscMap::isPictureStop(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        tbcDebugStream() << "isPictureStop out of frameNumber range";
        return false;
    }
    return m_frames[frameNumber].isPictureStop();
}

bool DiscMap::isLeadInOut(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        tbcDebugStream() << "isLeadInOut out of frameNumber range";
        return false;
    }
    return m_frames[frameNumber].isLeadInOrOut();
}

double DiscMap::frameQuality(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        tbcDebugStream() << "frameQuality out of frameNumber range";
        return -1;
    }
    return m_frames[frameNumber].frameQuality();
}

bool DiscMap::isPadded(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        tbcDebugStream() << "isPadded out of frameNumber range";
        return false;
    }
    return m_frames[frameNumber].isPadded();
}

bool DiscMap::isClvOffset(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        tbcDebugStream() << "isClvOffset out of frameNumber range";
        return false;
    }
    return m_frames[frameNumber].isClvOffset();
}

void DiscMap::setMarkedForDeletion(qint32 frameNumber)
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        tbcDebugStream() << "setMarkedForDeletion out of frameNumber range";
        return;
    }
    m_frames[frameNumber].isMarkedForDeletion(true);
}

bool DiscMap::isPhaseCorrect(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        tbcDebugStream() << "isPhaseCorrect out of frameNumber range";
        return false;
    }

    qint32 expectedNextPhase;

    if (frameNumber > 0) {
        expectedNextPhase = m_frames[frameNumber - 1].secondFieldPhase() + 1;
        if (m_isDiscPal && expectedNextPhase == 9) expectedNextPhase = 1;
        if (!m_isDiscPal && expectedNextPhase == 5) expectedNextPhase = 1;
        if (m_frames[frameNumber].firstFieldPhase() != expectedNextPhase) {
            tbcDebugStream() << "Frame number" << frameNumber << "phase sequence does not match preceding frame! -"
                             << expectedNextPhase << "expected but got" << m_frames[frameNumber].firstFieldPhase();
            return false;
        }
    }

    if (frameNumber < m_numberOfFrames - 1) {
        expectedNextPhase = m_frames[frameNumber].secondFieldPhase() + 1;
        if (m_isDiscPal && expectedNextPhase == 9) expectedNextPhase = 1;
        if (!m_isDiscPal && expectedNextPhase == 5) expectedNextPhase = 1;
        if (m_frames[frameNumber + 1].firstFieldPhase() != expectedNextPhase) {
            tbcDebugStream() << "Frame number" << frameNumber << "phase sequence does not match following frame! -"
                             << expectedNextPhase << "expected but got" << m_frames[frameNumber].secondFieldPhase();
            return false;
        }
    }

    return true;
}

bool DiscMap::isPhaseRepeating(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        tbcDebugStream() << "isPhaseRepeating out of frameNumber range";
        return false;
    }

    if (frameNumber > 0) {
        if ((m_frames[frameNumber].firstFieldPhase()  == m_frames[frameNumber - 1].firstFieldPhase()) &&
            (m_frames[frameNumber].secondFieldPhase() == m_frames[frameNumber - 1].secondFieldPhase()))
            return true;
    } else {
        // Frame 0 has no predecessor; treat as non-repeating so it is never
        // filtered out on the basis of phase alone.
        return true;
    }

    return false;
}

qint32 DiscMap::flush()
{
    qint32 origSize = m_frames.size();
    m_frames.erase(
        std::remove_if(m_frames.begin(), m_frames.end(),
                       [](const Frame &o) { return o.isMarkedForDeletion(); }),
        m_frames.end());
    m_numberOfFrames = m_frames.size();
    return origSize - m_frames.size();
}

void DiscMap::sort()
{
    std::sort(m_frames.begin(), m_frames.end());
    m_numberOfFrames = m_frames.size();
}

void DiscMap::debugFrameDetails(qint32 frameNumber)
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        tbcDebugStream() << "debugFrameDetails out of frameNumber range";
        return;
    }
    tbcDebugStream() << m_frames[frameNumber];
}

bool DiscMap::isNtscAmendment2ClvFrameNumber(qint32 frameNumber)
{
    // The IEC Amendment 2 NTSC CLV timecode scheme inserts an extra frame number
    // at positions n = 8991*l + 899*m for l in [0,13] and m in [1,9].
    // These correspond to the rollover points where the CLV picture-number field
    // wraps within each second group.
    for (qint32 l = 0; l < 14; l++) {
        for (qint32 m = 1; m <= 9; m++) {
            qint32 n = 8991 * l + 899 * m;
            if (n == frameNumber) return true;
            if (n > frameNumber) return false;
        }
    }
    return false;
}

void DiscMap::addPadding(qint32 startFrame, qint32 numberOfFrames)
{
    m_frames.reserve(m_frames.size() + numberOfFrames);
    qint32 currentVbi = m_frames[startFrame].vbiFrameNumber() + 1;
    for (qint32 i = 0; i < numberOfFrames; i++) {
        Frame paddingFrame;
        paddingFrame.vbiFrameNumber(currentVbi + i);
        paddingFrame.seqFrameNumber(-1);
        paddingFrame.isPadded(true);
        m_frames.push_back(paddingFrame);
    }
    m_numberOfFrames = m_frames.size();
}

qint32 DiscMap::getVideoFieldLength() const            { return m_videoFieldLength; }
qint32 DiscMap::getApproximateAudioFieldLength() const { return m_audioFieldByteLength / 2; }

qint32 DiscMap::getFirstFieldNumber(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        tbcDebugStream() << "getFirstFieldNumber out of frameNumber range";
        return -1;
    }
    return m_frames[frameNumber].firstField();
}

qint32 DiscMap::getSecondFieldNumber(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        tbcDebugStream() << "getSecondFieldNumber out of frameNumber range";
        return -1;
    }
    return m_frames[frameNumber].secondField();
}

qint32 DiscMap::getFirstFieldPhase(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        tbcDebugStream() << "getFirstFieldPhase out of frameNumber range";
        return -1;
    }
    return m_frames[frameNumber].firstFieldPhase();
}

qint32 DiscMap::getSecondFieldPhase(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        tbcDebugStream() << "getSecondFieldPhase out of frameNumber range";
        return -1;
    }
    return m_frames[frameNumber].secondFieldPhase();
}

qint32 DiscMap::getFirstFieldAudioDataStart(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        tbcDebugStream() << "getFirstFieldAudioDataStart out of frameNumber range";
        return -1;
    }
    return ldDecodeMetaData->getFieldPcmAudioStart(m_frames[frameNumber].firstField());
}

qint32 DiscMap::getFirstFieldAudioDataLength(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        tbcDebugStream() << "getFirstFieldAudioDataLength out of frameNumber range";
        return -1;
    }
    return ldDecodeMetaData->getFieldPcmAudioLength(m_frames[frameNumber].firstField());
}

qint32 DiscMap::getSecondFieldAudioDataStart(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        tbcDebugStream() << "getSecondFieldAudioDataStart out of frameNumber range";
        return -1;
    }
    return ldDecodeMetaData->getFieldPcmAudioStart(m_frames[frameNumber].secondField());
}

qint32 DiscMap::getSecondFieldAudioDataLength(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        tbcDebugStream() << "getSecondFieldAudioDataLength out of frameNumber range";
        return -1;
    }
    return ldDecodeMetaData->getFieldPcmAudioLength(m_frames[frameNumber].secondField());
}

bool DiscMap::saveTargetMetadata(QFileInfo outputFileInfo)
{
    qint32 notifyInterval = m_numberOfFrames / 50;
    if (notifyInterval < 1) notifyInterval = 1;

    LdDecodeMetaData targetMetadata;
    LdDecodeMetaData::VideoParameters sourceVideoParameters = ldDecodeMetaData->getVideoParameters();
    sourceVideoParameters.isMapped = true;
    targetMetadata.setVideoParameters(sourceVideoParameters);
    targetMetadata.setPcmAudioParameters(ldDecodeMetaData->getPcmAudioParameters());
    targetMetadata.setNumberOfFields(m_numberOfFrames * 2);

    VbiDecoder vbiDecoder;

    for (qint32 frameNumber = 0; frameNumber < m_numberOfFrames; frameNumber++) {
        if (!m_frames[frameNumber].isPadded()) {
            qint32 firstFieldNumber  = m_frames[frameNumber].firstField();
            qint32 secondFieldNumber = m_frames[frameNumber].secondField();

            LdDecodeMetaData::Field firstSourceMetadata  = ldDecodeMetaData->getField(firstFieldNumber);
            LdDecodeMetaData::Field secondSourceMetadata = ldDecodeMetaData->getField(secondFieldNumber);

            if (m_isDiscCav) {
                if (!firstSourceMetadata.vbi.inUse) {
                    firstSourceMetadata.vbi.inUse = true;
                    firstSourceMetadata.vbi.vbiData[0] = 0;
                }
                firstSourceMetadata.vbi.vbiData[1] = convertFrameToVbi(m_frames[frameNumber].vbiFrameNumber());
                firstSourceMetadata.vbi.vbiData[2] = convertFrameToVbi(m_frames[frameNumber].vbiFrameNumber());

                VbiDecoder::Vbi vbi = vbiDecoder.decodeFrame(
                        firstSourceMetadata.vbi.vbiData[0], firstSourceMetadata.vbi.vbiData[1], firstSourceMetadata.vbi.vbiData[2],
                        secondSourceMetadata.vbi.vbiData[0], secondSourceMetadata.vbi.vbiData[1], secondSourceMetadata.vbi.vbiData[2]);

                if (vbi.picNo != m_frames[frameNumber].vbiFrameNumber()) {
                    // The corrected frame number has been overwritten by conflicting
                    // data already present in VBI line 0. Clear line 0 so the picture
                    // number written to lines 1 and 2 decodes cleanly.
                    qInfo() << "Warning: Updated VBI frame number for frame" << m_frames[frameNumber].vbiFrameNumber()
                            << "has been corrupted by existing VBI data - overwriting all VBI for frame";
                    firstSourceMetadata.vbi.vbiData[0] = 0;
                }
            } else {
                if (!firstSourceMetadata.vbi.inUse) firstSourceMetadata.vbi.inUse = true;
                firstSourceMetadata.vbi.vbiData[0] = convertFrameToClvPicNo(m_frames[frameNumber].vbiFrameNumber());
                firstSourceMetadata.vbi.vbiData[1] = convertFrameToClvTimeCode(m_frames[frameNumber].vbiFrameNumber());
                firstSourceMetadata.vbi.vbiData[2] = convertFrameToClvTimeCode(m_frames[frameNumber].vbiFrameNumber());
            }

            targetMetadata.appendField(firstSourceMetadata);
            targetMetadata.appendField(secondSourceMetadata);
        } else {
            LdDecodeMetaData::Field firstSourceMetadata;
            LdDecodeMetaData::Field secondSourceMetadata;
            firstSourceMetadata.isFirstField  = true;
            secondSourceMetadata.isFirstField = false;
            firstSourceMetadata.pad  = true;
            secondSourceMetadata.pad = true;
            firstSourceMetadata.fieldPhaseID  = -1;
            secondSourceMetadata.fieldPhaseID = -1;

            if (m_isDiscCav) {
                firstSourceMetadata.vbi.inUse = true;
                firstSourceMetadata.vbi.vbiData[0] = 0;
                firstSourceMetadata.vbi.vbiData[1] = convertFrameToVbi(m_frames[frameNumber].vbiFrameNumber());
                firstSourceMetadata.vbi.vbiData[2] = convertFrameToVbi(m_frames[frameNumber].vbiFrameNumber());
                firstSourceMetadata.audioSamples   = m_audioFieldSampleLength;
                secondSourceMetadata.vbi.inUse = true;
                secondSourceMetadata.vbi.vbiData[0] = 0;
                secondSourceMetadata.vbi.vbiData[1] = 0;
                secondSourceMetadata.vbi.vbiData[2] = 0;
                secondSourceMetadata.audioSamples   = m_audioFieldSampleLength;
            } else {
                firstSourceMetadata.vbi.inUse = true;
                firstSourceMetadata.vbi.vbiData[0] = convertFrameToClvPicNo(m_frames[frameNumber].vbiFrameNumber());
                firstSourceMetadata.vbi.vbiData[1] = convertFrameToClvTimeCode(m_frames[frameNumber].vbiFrameNumber());
                firstSourceMetadata.vbi.vbiData[2] = convertFrameToClvTimeCode(m_frames[frameNumber].vbiFrameNumber());
                firstSourceMetadata.audioSamples   = m_audioFieldSampleLength;
                secondSourceMetadata.vbi.inUse = true;
                secondSourceMetadata.vbi.vbiData[0] = 0;
                secondSourceMetadata.vbi.vbiData[1] = 0;
                secondSourceMetadata.vbi.vbiData[2] = 0;
                secondSourceMetadata.audioSamples   = m_audioFieldSampleLength;
            }

            targetMetadata.appendField(firstSourceMetadata);
            targetMetadata.appendField(secondSourceMetadata);
        }

        if (frameNumber % notifyInterval == 0)
            qInfo() << "Created metadata for frame" << frameNumber << "of" << m_numberOfFrames;
    }

    qInfo() << "Writing target metadata to disc...";
    targetMetadata.write(outputFileInfo.filePath());
    qInfo() << "Target metadata written";

    return true;
}

qint32 DiscMap::convertFrameToVbi(qint32 frameNumber)
{
    // CAV picture numbers are BCD-encoded in VBI with the prefix 0x00F.
    // Example: frame 12345 → "00F12345" interpreted as hex → 0x00F12345.
    QString number = "00F" + QString("%1").arg(frameNumber, 5, 10, QChar('0'));
    bool ok;
    qint32 returnValue = number.toInt(&ok, 16);
    if (!ok) returnValue = 0;
    return returnValue;
}

qint32 DiscMap::convertFrameToClvPicNo(qint32 frameNumber)
{
    LdDecodeMetaData::ClvTimecode timecode = ldDecodeMetaData->convertFrameNumberToClvTimecode(frameNumber);

    // The CLV picture-number VBI word (prefix 0x008) encodes the seconds field
    // split across two nibbles: secondsX1 carries the tens digit (offset +10, +9
    // as a hex digit) and secondsX3 carries the units digit, followed by the
    // two-digit picture number within the second.
    qint32 secondsX1 = (timecode.seconds / 10) * 10; // tens boundary
    qint32 secondsX3 = timecode.seconds - secondsX1;  // units digit
    secondsX1 = ((secondsX1 + 10) / 10) + 9;         // encode tens as offset hex digit

    QString number = "008" + QString("%1").arg(secondsX1, 1, 16, QChar('0')) + "E" +
            QString("%1").arg(secondsX3, 1, 10, QChar('0')) +
            QString("%1").arg(timecode.pictureNumber, 2, 10, QChar('0'));
    bool ok;
    qint32 returnValue = number.toInt(&ok, 16);
    if (!ok) returnValue = 0;
    return returnValue;
}

qint32 DiscMap::convertFrameToClvTimeCode(qint32 frameNumber)
{
    LdDecodeMetaData::ClvTimecode timecode = ldDecodeMetaData->convertFrameNumberToClvTimecode(frameNumber);

    // The CLV timecode VBI word (prefix 0x00F) encodes hours and minutes in BCD:
    // one digit for hours followed by two digits for minutes, with 0xDD as a
    // fixed separator between them.
    QString number = "00F" + QString("%1").arg(timecode.hours, 1, 10, QChar('0')) + "DD" +
            QString("%1").arg(timecode.minutes, 2, 10, QChar('0'));
    bool ok;
    qint32 returnValue = number.toInt(&ok, 16);
    if (!ok) returnValue = 0;
    return returnValue;
}
