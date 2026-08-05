/************************************************************************

    stackingpool.cpp

    ld-disc-stacker - Disc stacking for ld-decode
    Copyright (C) 2020-2025 Simon Inns
    Copyright (C) 2026 Joseph Burns

    This file is part of ld-decode-tools.

    ld-disc-stacker is free software: you can redistribute it and/or
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

#include "stackingpool.h"
#include "vbidecoder.h"
#include "tbc/logging.h"
#include <cmath>

StackingPool::StackingPool(QString _outputFilename, QString _outputMetadataFilename,
                           qint32 _maxThreads,
                           QVector<LdDecodeMetaData *>& _ldDecodeMetaData,
                           QVector<SourceVideo *>& _sourceVideos,
                           qint32 _mode, qint32 _smartThreshold,
                           bool _reverse, bool _noDiffDod, bool _passThrough,
                           bool _integrityCheck, bool _verbose,
                           bool _useSnrWeight, qint32 _snrWeightThreshold,
                           QObject *parent)
    : QObject(parent),
      outputFilename(_outputFilename),
      outputMetadataFilename(_outputMetadataFilename),
      maxThreads(_maxThreads),
      mode(_mode), smartThreshold(_smartThreshold),
      reverse(_reverse), noDiffDod(_noDiffDod), passThrough(_passThrough),
      integrityCheck(_integrityCheck), verbose(_verbose),
      useSnrWeight(_useSnrWeight), snrWeightThreshold(_snrWeightThreshold),
      abort(false),
      ldDecodeMetaData(_ldDecodeMetaData), sourceVideos(_sourceVideos)
{
}

bool StackingPool::process()
{
    qInfo() << "Performing final sanity checks...";

    targetVideo.setFileName(outputFilename);
    if (outputFilename == "-") {
        if (!targetVideo.open(stdout, QIODevice::WriteOnly)) {
            qInfo() << "Unable to open stdout";
            return false;
        }
    } else {
        if (!targetVideo.open(QIODevice::WriteOnly)) {
            qInfo() << "Unable to open output video file";
            return false;
        }
    }

    qInfo() << "Verifying leading fields match...";
    qint32 firstFieldNumber  = ldDecodeMetaData[0]->getFirstFieldNumber(1);
    qint32 secondFieldNumber = ldDecodeMetaData[0]->getSecondFieldNumber(1);
    if (firstFieldNumber != 1 && secondFieldNumber != 1) {
        SourceVideo::Data sourceField = sourceVideos[0]->getVideoField(1);
        if (!writeOutputField(sourceField)) {
            qInfo() << "Writing first field to the output TBC file failed";
            targetVideo.close();
            return false;
        }
    }

    qInfo() << "Scanning source videos for VBI frame number ranges...";
    if (!setMinAndMaxVbiFrames()) {
        qInfo() << "It was not possible to determine the VBI frame number range for the source video - cannot continue!";
        return false;
    }

    qInfo() << "Using" << maxThreads << "threads to process" << ldDecodeMetaData[0]->getNumberOfFrames() << "frames";

    // Write a placeholder metadata file before stacking begins so that ld-analyse
    // can open the partial output TBC if the process is interrupted.
    // A successful run overwrites this with the correct stacked metadata.
    if (outputFilename != "-") {
        ldDecodeMetaData[0]->write(outputMetadataFilename);
        qInfo() << "Placeholder metadata written to" << outputMetadataFilename;
    }

    inputFrameNumber = 1;
    outputFrameNumber = 1;
    lastFrameNumber = ldDecodeMetaData[0]->getNumberOfFrames();
    skippedFrame = 0;
    totalTimer.start();

    qInfo() << "Beginning multi-threaded disc stacking process...";
    QVector<QThread *> threads;
    threads.resize(maxThreads);
    for (qint32 i = 0; i < maxThreads; i++) {
        threads[i] = new Stacker(abort, *this);
        threads[i]->start(QThread::LowPriority);
    }
    for (qint32 i = 0; i < maxThreads; i++) {
        threads[i]->wait();
        delete threads[i];
    }

    if (abort) {
        targetVideo.close();
        return false;
    }

    const double totalSecs = static_cast<double>(totalTimer.elapsed()) / 1000.0;
    qInfo() << "Disc stacking complete -" << lastFrameNumber << "frames in" << totalSecs
            << "seconds (" << lastFrameNumber / totalSecs << "FPS )";
    if (integrityCheck)
        qInfo() << "Stacking found " << skippedFrame << "corrupted frame";

    qInfo() << "Creating metadata file for stacked TBC...";
    correctMetaData().write(outputMetadataFilename);

    targetVideo.close();
    return true;
}

bool StackingPool::getInputFrame(qint32& frameNumber,
                                  QVector<qint32>& firstFieldNumber,
                                  QVector<SourceVideo::Data>& firstFieldVideoData,
                                  QVector<LdDecodeMetaData::Field>& firstFieldMetadata,
                                  QVector<qint32>& secondFieldNumber,
                                  QVector<SourceVideo::Data>& secondFieldVideoData,
                                  QVector<LdDecodeMetaData::Field>& secondFieldMetadata,
                                  QVector<LdDecodeMetaData::VideoParameters>& videoParameters,
                                  qint32& _mode, qint32& _smartThreshold,
                                  bool& _reverse, bool& _noDiffDod, bool& _passThrough,
                                  bool& _verbose,
                                  QVector<qint32>& availableSourcesForFrame,
                                  QVector<double>& sourceSnrWeights,
                                  bool& _useSnrWeight, qint32& _snrWeightThreshold)
{
    QMutexLocker locker(&inputMutex);

    if (inputFrameNumber > lastFrameNumber)
        return false;

    frameNumber = inputFrameNumber++;

    const qint32 numberOfSources = sourceVideos.size();

    if (verbose)
        tbcDebugStream().nospace() << "Processing sequential frame number #" << frameNumber
                                   << " from " << numberOfSources << " possible source(s)";

    firstFieldNumber.resize(numberOfSources);
    firstFieldVideoData.resize(numberOfSources);
    firstFieldMetadata.resize(numberOfSources);
    secondFieldNumber.resize(numberOfSources);
    secondFieldVideoData.resize(numberOfSources);
    secondFieldMetadata.resize(numberOfSources);
    videoParameters.resize(numberOfSources);

    // Get the current VBI frame number from the timemaster's VBI map.
    // Map lookup rather than arithmetic ensures alignment is exact regardless
    // of where on the disc the timemaster starts.
    qint32 currentVbiFrame = -1;
    if (numberOfSources > 1) {
        currentVbiFrame = timemasterSeqToVbi.value(frameNumber, -1);
        if (currentVbiFrame == -1)
            currentVbiFrame = (sourceMinimumVbiFrame[0] - 1) + frameNumber;
    }

    for (qint32 sourceNo = 0; sourceNo < numberOfSources; sourceNo++) {
        firstFieldNumber[sourceNo]  = -1;
        secondFieldNumber[sourceNo] = -1;

        if (sourceNo == 0) {
            firstFieldNumber[sourceNo]  = ldDecodeMetaData[sourceNo]->getFirstFieldNumber(frameNumber);
            secondFieldNumber[sourceNo] = ldDecodeMetaData[sourceNo]->getSecondFieldNumber(frameNumber);
            if (verbose)
                tbcDebugStream().nospace() << "Source #0 fields are "
                                           << firstFieldNumber[sourceNo] << "/" << secondFieldNumber[sourceNo];
        } else if (currentVbiFrame != -1 &&
                   currentVbiFrame >= sourceMinimumVbiFrame[sourceNo] &&
                   currentVbiFrame <= sourceMaximumVbiFrame[sourceNo]) {
            const qint32 currentSourceFrameNumber = sourceVbiMap[sourceNo].value(currentVbiFrame, -1);
            if (currentSourceFrameNumber == -1) {
                if (verbose)
                    tbcDebugStream().nospace() << "Source #" << sourceNo
                                               << " does not contain VBI frame number " << currentVbiFrame;
            } else {
                firstFieldNumber[sourceNo]  = ldDecodeMetaData[sourceNo]->getFirstFieldNumber(currentSourceFrameNumber);
                secondFieldNumber[sourceNo] = ldDecodeMetaData[sourceNo]->getSecondFieldNumber(currentSourceFrameNumber);
                if (verbose)
                    tbcDebugStream().nospace() << "Source #" << sourceNo << " has VBI frame number " << currentVbiFrame
                                               << " and fields " << firstFieldNumber[sourceNo] << "/" << secondFieldNumber[sourceNo];
            }
        } else if (verbose) {
            tbcDebugStream().nospace() << "Source #" << sourceNo << " does not contain a usable frame";
        }

        if (firstFieldNumber[sourceNo] != -1 && secondFieldNumber[sourceNo] != -1) {
            if (firstFieldNumber[sourceNo] < secondFieldNumber[sourceNo]) {
                firstFieldVideoData[sourceNo]  = sourceVideos[sourceNo]->getVideoField(firstFieldNumber[sourceNo]);
                secondFieldVideoData[sourceNo] = sourceVideos[sourceNo]->getVideoField(secondFieldNumber[sourceNo]);
            } else {
                secondFieldVideoData[sourceNo] = sourceVideos[sourceNo]->getVideoField(secondFieldNumber[sourceNo]);
                firstFieldVideoData[sourceNo]  = sourceVideos[sourceNo]->getVideoField(firstFieldNumber[sourceNo]);
            }
            firstFieldMetadata[sourceNo]  = ldDecodeMetaData[sourceNo]->getField(firstFieldNumber[sourceNo]);
            secondFieldMetadata[sourceNo] = ldDecodeMetaData[sourceNo]->getField(secondFieldNumber[sourceNo]);
            videoParameters[sourceNo]     = ldDecodeMetaData[sourceNo]->getVideoParameters();
        }
    }

    availableSourcesForFrame.clear();
    if (numberOfSources > 1)
        availableSourcesForFrame = getAvailableSourcesForFrame(currentVbiFrame);
    else
        availableSourcesForFrame.append(0);

    // Tighten: remove sources that did not produce valid field numbers
    for (int j = availableSourcesForFrame.size() - 1; j >= 0; --j) {
        const qint32 s = availableSourcesForFrame[j];
        if (s < 0 || s >= numberOfSources ||
            firstFieldNumber[s] == -1 || secondFieldNumber[s] == -1)
            availableSourcesForFrame.removeAt(j);
    }

    if (integrityCheck && !availableSourcesForFrame.isEmpty()) {
        for (int j = availableSourcesForFrame.size() - 1; j >= 0; --j) {
            const qint32 s = availableSourcesForFrame[j];
            if (!isIntegrityOk(firstFieldVideoData[s], videoParameters[0])) {
                availableSourcesForFrame.removeAt(j);
                skippedFrame++;
                qInfo() << "found corrupted data at output frame :" << frameNumber << "from source (" << s << ") field 1";
            } else if (!isIntegrityOk(secondFieldVideoData[s], videoParameters[0])) {
                availableSourcesForFrame.removeAt(j);
                skippedFrame++;
                qInfo() << "found corrupted data at output frame :" << frameNumber << "from source (" << s << ") field 2";
            }
        }
    }

    // Compute per-source SNR weights (linear amplitude domain: 10^(dB/20)).
    // Sources with no VITS metrics receive weight 0.0 — they still contribute
    // via the count-based path but carry no weight in the SNR override.
    sourceSnrWeights.clear();
    sourceSnrWeights.resize(availableSourcesForFrame.size());
    if (useSnrWeight) {
        for (qint32 i = 0; i < availableSourcesForFrame.size(); i++) {
            const qint32 s = availableSourcesForFrame[i];
            const bool have1 = firstFieldMetadata[s].vitsMetrics.inUse  && firstFieldMetadata[s].vitsMetrics.bPSNR  > 0.0;
            const bool have2 = secondFieldMetadata[s].vitsMetrics.inUse && secondFieldMetadata[s].vitsMetrics.bPSNR > 0.0;
            double snrDb = -1.0;
            if      (have1 && have2) snrDb = (firstFieldMetadata[s].vitsMetrics.bPSNR + secondFieldMetadata[s].vitsMetrics.bPSNR) / 2.0;
            else if (have1)          snrDb = firstFieldMetadata[s].vitsMetrics.bPSNR;
            else if (have2)          snrDb = secondFieldMetadata[s].vitsMetrics.bPSNR;
            sourceSnrWeights[i] = (snrDb > 0.0) ? std::pow(10.0, snrDb / 20.0) : 0.0;
        }
    }
    // If useSnrWeight is false all weights stay 0.0 and the bad-consensus handler
    // in stackMode finds totalWeight == 0 and skips itself entirely.

    if (mode != -1)          _mode = mode;
    else if (numberOfSources <= 2) _mode = 0;
    else if (numberOfSources <= 4) _mode = 2;
    else                           _mode = 3;

    _smartThreshold   = smartThreshold;
    _reverse          = reverse;
    _noDiffDod        = noDiffDod;
    _passThrough      = passThrough;
    _verbose          = verbose;
    _useSnrWeight     = useSnrWeight;
    _snrWeightThreshold = snrWeightThreshold;

    return true;
}

bool StackingPool::setOutputFrame(qint32 frameNumber,
                                   SourceVideo::Data firstTargetFieldData,
                                   SourceVideo::Data secondTargetFieldData,
                                   qint32 firstFieldSeqNo, qint32 secondFieldSeqNo,
                                   DropOuts firstTargetFieldDropOuts,
                                   DropOuts secondTargetFieldDropouts)
{
    QMutexLocker locker(&outputMutex);

    OutputFrame pendingFrame;
    pendingFrame.firstTargetFieldData    = firstTargetFieldData;
    pendingFrame.secondTargetFieldData   = secondTargetFieldData;
    pendingFrame.firstFieldSeqNo         = firstFieldSeqNo;
    pendingFrame.secondFieldSeqNo        = secondFieldSeqNo;
    pendingFrame.firstTargetFieldDropOuts  = firstTargetFieldDropOuts;
    pendingFrame.secondTargetFieldDropOuts = secondTargetFieldDropouts;
    pendingOutputFrames[frameNumber] = pendingFrame;

    while (pendingOutputFrames.contains(outputFrameNumber)) {
        const OutputFrame& outputFrame = pendingOutputFrames.value(outputFrameNumber);

        bool writeFail = false;
        if (outputFrame.firstFieldSeqNo < outputFrame.secondFieldSeqNo) {
            if (!writeOutputField(outputFrame.firstTargetFieldData))  writeFail = true;
            if (!writeOutputField(outputFrame.secondTargetFieldData)) writeFail = true;
        } else {
            if (!writeOutputField(outputFrame.secondTargetFieldData)) writeFail = true;
            if (!writeOutputField(outputFrame.firstTargetFieldData))  writeFail = true;
        }

        if (writeFail) {
            qCritical() << "Writing fields to the output TBC file failed";
            targetVideo.close();
            return false;
        }

        ldDecodeMetaData[0]->clearFieldDropOuts(outputFrame.firstFieldSeqNo);
        ldDecodeMetaData[0]->clearFieldDropOuts(outputFrame.secondFieldSeqNo);
        ldDecodeMetaData[0]->updateFieldDropOuts(outputFrame.firstTargetFieldDropOuts,  outputFrame.firstFieldSeqNo);
        ldDecodeMetaData[0]->updateFieldDropOuts(outputFrame.secondTargetFieldDropOuts, outputFrame.secondFieldSeqNo);

        tbcDebugStream().nospace() << "Processed frame " << outputFrameNumber;
        if (outputFrameNumber % 100 == 0)
            qInfo() << "Processed and written frame" << outputFrameNumber;

        pendingOutputFrames.remove(outputFrameNumber);
        outputFrameNumber++;
    }

    return true;
}

bool StackingPool::setMinAndMaxVbiFrames()
{
    const qint32 numberOfSources = sourceVideos.size();

    sourceDiscTypeCav.resize(numberOfSources);
    sourceMaximumVbiFrame.resize(numberOfSources);
    sourceMinimumVbiFrame.resize(numberOfSources);
    sourceVbiMap.resize(numberOfSources);
    timemasterSeqToVbi.clear();

    for (qint32 sourceNumber = 0; sourceNumber < numberOfSources; sourceNumber++) {
        VbiDecoder vbiDecoder;
        qint32 cavCount = 0, clvCount = 0;
        qint32 cavMin = 1000000, cavMax = 0;
        qint32 clvMin = 1000000, clvMax = 0;

        sourceMinimumVbiFrame[sourceNumber] = 0;
        sourceMaximumVbiFrame[sourceNumber] = 0;
        sourceDiscTypeCav[sourceNumber]     = false;
        sourceVbiMap[sourceNumber].clear();

        for (qint32 seqFrame = 1; seqFrame <= ldDecodeMetaData[sourceNumber]->getNumberOfFrames(); seqFrame++) {
            auto vbi1 = ldDecodeMetaData[sourceNumber]->getFieldVbi(ldDecodeMetaData[sourceNumber]->getFirstFieldNumber(seqFrame)).vbiData;
            auto vbi2 = ldDecodeMetaData[sourceNumber]->getFieldVbi(ldDecodeMetaData[sourceNumber]->getSecondFieldNumber(seqFrame)).vbiData;
            VbiDecoder::Vbi vbi = vbiDecoder.decodeFrame(vbi1[0], vbi1[1], vbi1[2], vbi2[0], vbi2[1], vbi2[2]);

            if (vbi.picNo > 0) {
                cavCount++;
                if (vbi.picNo < cavMin) cavMin = vbi.picNo;
                if (vbi.picNo > cavMax) cavMax = vbi.picNo;
                sourceVbiMap[sourceNumber].insert(vbi.picNo, seqFrame);
                if (sourceNumber == 0) timemasterSeqToVbi.insert(seqFrame, vbi.picNo);
            }

            if (vbi.clvHr != -1 && vbi.clvMin != -1 && vbi.clvSec != -1 && vbi.clvPicNo != -1) {
                clvCount++;
                LdDecodeMetaData::ClvTimecode timecode;
                timecode.hours = vbi.clvHr; timecode.minutes = vbi.clvMin;
                timecode.seconds = vbi.clvSec; timecode.pictureNumber = vbi.clvPicNo;
                qint32 cvFrameNumber = ldDecodeMetaData[sourceNumber]->convertClvTimecodeToFrameNumber(timecode);
                if (cvFrameNumber < clvMin) clvMin = cvFrameNumber;
                if (cvFrameNumber > clvMax) clvMax = cvFrameNumber;
                sourceVbiMap[sourceNumber].insert(cvFrameNumber, seqFrame);
                if (sourceNumber == 0) timemasterSeqToVbi.insert(seqFrame, cvFrameNumber);
            }
        }

        tbcDebugStream() << "StackingPool::setMinAndMaxVbiFrames(): Got" << cavCount << "CAV picture codes and" << clvCount << "CLV timecodes";

        if (cavCount == 0 && clvCount == 0) {
            tbcDebugStream() << "StackingPool::setMinAndMaxVbiFrames(): Source does not seem to contain valid CAV picture numbers or CLV time-codes - cannot process";
            return false;
        }

        if (cavCount > clvCount) {
            sourceDiscTypeCav[sourceNumber] = true;
            tbcDebugStream() << "StackingPool::setMinAndMaxVbiFrames(): Got" << cavCount << "valid CAV picture numbers - source disc type is CAV";
            qInfo().nospace() << "Source #" << sourceNumber << " has a disc type of CAV (uses VBI frame numbers)";
            sourceMaximumVbiFrame[sourceNumber] = cavMax;

            // Validate minimum by monotonic walk-back from frame 10.
            // Guards against corrupt or absent VBI in opening frames producing
            // a falsely high floor that shifts alignment.
            {
                qint32 validatedMin = cavMin;
                const qint32 checkFrames = qMin(10, ldDecodeMetaData[sourceNumber]->getNumberOfFrames());
                QVector<qint32> leadIn;
                leadIn.reserve(checkFrames);
                for (qint32 f = 1; f <= checkFrames; f++) {
                    auto v1 = ldDecodeMetaData[sourceNumber]->getFieldVbi(ldDecodeMetaData[sourceNumber]->getFirstFieldNumber(f)).vbiData;
                    auto v2 = ldDecodeMetaData[sourceNumber]->getFieldVbi(ldDecodeMetaData[sourceNumber]->getSecondFieldNumber(f)).vbiData;
                    VbiDecoder::Vbi v = vbiDecoder.decodeFrame(v1[0], v1[1], v1[2], v2[0], v2[1], v2[2]);
                    leadIn.append(v.picNo > 0 ? v.picNo : -1);
                }
                qint32 anchor = leadIn.last();
                if (anchor > 0) {
                    for (qint32 i = leadIn.size() - 2; i >= 0; i--) {
                        if (leadIn[i] > 0 && leadIn[i] == anchor - 1) anchor = leadIn[i];
                        else break;
                    }
                    validatedMin = anchor;
                }
                if (validatedMin != cavMin)
                    qInfo().nospace() << "Source #" << sourceNumber << " CAV min adjusted from " << cavMin << " to " << validatedMin << " (ragged start detected)";
                sourceMinimumVbiFrame[sourceNumber] = validatedMin;
            }
        } else {
            sourceDiscTypeCav[sourceNumber] = false;
            tbcDebugStream() << "StackingPool::setMinAndMaxVbiFrames(): Got" << clvCount << "valid CLV picture numbers - source disc type is CLV";
            qInfo().nospace() << "Source #" << sourceNumber << " has a disc type of CLV (uses VBI time codes)";
            sourceMaximumVbiFrame[sourceNumber] = clvMax;

            // Same monotonic validation for CLV
            {
                qint32 validatedMin = clvMin;
                const qint32 checkFrames = qMin(10, ldDecodeMetaData[sourceNumber]->getNumberOfFrames());
                QVector<qint32> leadIn;
                leadIn.reserve(checkFrames);
                for (qint32 f = 1; f <= checkFrames; f++) {
                    auto v1 = ldDecodeMetaData[sourceNumber]->getFieldVbi(ldDecodeMetaData[sourceNumber]->getFirstFieldNumber(f)).vbiData;
                    auto v2 = ldDecodeMetaData[sourceNumber]->getFieldVbi(ldDecodeMetaData[sourceNumber]->getSecondFieldNumber(f)).vbiData;
                    VbiDecoder::Vbi v = vbiDecoder.decodeFrame(v1[0], v1[1], v1[2], v2[0], v2[1], v2[2]);
                    if (v.clvHr != -1 && v.clvMin != -1 && v.clvSec != -1 && v.clvPicNo != -1) {
                        LdDecodeMetaData::ClvTimecode tc;
                        tc.hours = v.clvHr; tc.minutes = v.clvMin;
                        tc.seconds = v.clvSec; tc.pictureNumber = v.clvPicNo;
                        leadIn.append(ldDecodeMetaData[sourceNumber]->convertClvTimecodeToFrameNumber(tc));
                    } else {
                        leadIn.append(-1);
                    }
                }
                qint32 anchor = leadIn.last();
                if (anchor > 0) {
                    for (qint32 i = leadIn.size() - 2; i >= 0; i--) {
                        if (leadIn[i] > 0 && leadIn[i] == anchor - 1) anchor = leadIn[i];
                        else break;
                    }
                    validatedMin = anchor;
                }
                if (validatedMin != clvMin)
                    qInfo().nospace() << "Source #" << sourceNumber << " CLV min adjusted from " << clvMin << " to " << validatedMin << " (ragged start detected)";
                sourceMinimumVbiFrame[sourceNumber] = validatedMin;
            }
        }

        qInfo().nospace() << "Source #" << sourceNumber << " has a VBI frame number range of "
                          << sourceMinimumVbiFrame[sourceNumber] << " to " << sourceMaximumVbiFrame[sourceNumber]
                          << " (" << sourceVbiMap[sourceNumber].size() << " frames mapped)";
    }

    return true;
}

bool StackingPool::isIntegrityOk(const SourceVideo::Data& inputFields,
                                  const LdDecodeMetaData::VideoParameters& videoParameters)
{
    qint32 count = 0;
    for (qint32 y = 0; y < videoParameters.fieldHeight; y++) {
        if (inputFields[(videoParameters.fieldWidth * y) + 4] > (videoParameters.black16bIre - (10 * 256)))
            count++;
        else
            count = 0;
        if (count == 3) return false;
    }
    return true;
}

QVector<qint32> StackingPool::getAvailableSourcesForFrame(qint32 vbiFrameNumber)
{
    QVector<qint32> availableSourcesForFrame;
    for (qint32 sourceNo = 0; sourceNo < sourceVideos.size(); sourceNo++) {
        if (vbiFrameNumber >= sourceMinimumVbiFrame[sourceNo] &&
            vbiFrameNumber <= sourceMaximumVbiFrame[sourceNo]) {

            const qint32 sequentialFrameNumber = sourceVbiMap[sourceNo].value(vbiFrameNumber, -1);
            if (sequentialFrameNumber == -1) {
                tbcDebugStream() << "VBI Frame number" << vbiFrameNumber << "not found in map for source" << sourceNo;
            } else if (ldDecodeMetaData[sourceNo]->getNumberOfFrames() < sequentialFrameNumber) {
                tbcDebugStream() << "VBI Frame number" << vbiFrameNumber << "is out of bounds for source" << sourceNo;
            } else {
                const qint32 firstFN  = ldDecodeMetaData[sourceNo]->getFirstFieldNumber(sequentialFrameNumber);
                const qint32 secondFN = ldDecodeMetaData[sourceNo]->getSecondFieldNumber(sequentialFrameNumber);
                if (!ldDecodeMetaData[sourceNo]->getField(firstFN).pad &&
                    !ldDecodeMetaData[sourceNo]->getField(secondFN).pad) {
                    availableSourcesForFrame.append(sourceNo);
                } else if (verbose) {
                    if (ldDecodeMetaData[sourceNo]->getField(firstFN).pad)
                        tbcDebugStream() << "First field number" << firstFN << "of source" << sourceNo << "is padded";
                    if (ldDecodeMetaData[sourceNo]->getField(secondFN).pad)
                        tbcDebugStream() << "Second field number" << firstFN << "of source" << sourceNo << "is padded";
                }
            }
        }
    }

    if (availableSourcesForFrame.size() != sourceVideos.size() && verbose) {
        if (availableSourcesForFrame.size() > 0)
            tbcDebugStream() << "VBI Frame number" << vbiFrameNumber << "has only" << availableSourcesForFrame.size() << "available sources";
        else
            qInfo() << "Warning: VBI Frame number" << vbiFrameNumber << "has ZERO available sources (all sources padded?)";
    }

    return availableSourcesForFrame;
}

bool StackingPool::writeOutputField(const SourceVideo::Data& fieldData)
{
    return targetVideo.write(reinterpret_cast<const char *>(fieldData.data()), 2 * fieldData.size());
}

// Lookahead-based phase correction.
// Anchors on the first non-padded field, then walks forward. When a field's
// phase diverges from expectation, it looks ahead CONFIRMATION_WINDOW
// non-padded fields to decide if this is a real cadence break (preserve) or
// isolated corruption (repair). This replaces simoninns' blind sequential
// overwrite, which destroyed real phase transitions at edit boundaries.
void StackingPool::correctPhaseIDs()
{
    constexpr qint32 PHASE_COUNT        = 4;
    constexpr qint32 CONFIRMATION_WINDOW = 4;

    const qint32 fieldCount = ldDecodeMetaData[0]->getNumberOfFields();

    qint32 pivotField = 1;
    while (pivotField <= fieldCount && ldDecodeMetaData[0]->getField(pivotField).pad)
        ++pivotField;
    if (pivotField >= fieldCount) return;

    qint32 expectedPhase = ldDecodeMetaData[0]->getField(pivotField).fieldPhaseID - 1;

    for (qint32 fieldNumber = pivotField; fieldNumber <= fieldCount; ++fieldNumber) {
        LdDecodeMetaData::Field field = ldDecodeMetaData[0]->getField(fieldNumber);

        if (field.pad) {
            field.fieldPhaseID = expectedPhase + 1;
            ldDecodeMetaData[0]->updateField(field, fieldNumber);
            expectedPhase = (expectedPhase + 1) % PHASE_COUNT;
            continue;
        }

        const qint32 actualPhase = field.fieldPhaseID - 1;

        if (actualPhase == expectedPhase) {
            expectedPhase = (expectedPhase + 1) % PHASE_COUNT;
            continue;
        }

        // Divergence: look ahead to classify as break or corruption
        qint32 confirmCount  = 0;
        qint32 lookaheadPhase = (actualPhase + 1) % PHASE_COUNT;
        for (qint32 la = fieldNumber + 1;
             la <= fieldCount && confirmCount < CONFIRMATION_WINDOW; ++la) {
            const LdDecodeMetaData::Field& laField = ldDecodeMetaData[0]->getField(la);
            if (laField.pad) continue;
            if ((laField.fieldPhaseID - 1) == lookaheadPhase) {
                ++confirmCount;
                lookaheadPhase = (lookaheadPhase + 1) % PHASE_COUNT;
            } else break;
        }

        if (confirmCount >= CONFIRMATION_WINDOW) {
            qInfo().nospace() << "correctPhaseIDs: cadence break at field " << fieldNumber
                              << " (seqNo " << field.seqNo << "): phase " << (expectedPhase + 1)
                              << " -> " << (actualPhase + 1) << " confirmed by " << confirmCount
                              << " subsequent fields — preserving";
            expectedPhase = (actualPhase + 1) % PHASE_COUNT;
        } else {
            qInfo().nospace() << "correctPhaseIDs: repairing isolated phase error at field " << fieldNumber
                              << " (seqNo " << field.seqNo << "): " << (actualPhase + 1)
                              << " -> " << (expectedPhase + 1)
                              << " (only " << confirmCount << " confirming fields found, need "
                              << CONFIRMATION_WINDOW << ")";
            field.fieldPhaseID = expectedPhase + 1;
            ldDecodeMetaData[0]->updateField(field, fieldNumber);
            expectedPhase = (expectedPhase + 1) % PHASE_COUNT;
        }
    }
}

template<int field>
void StackingPool::replaceFieldMetaData(qint32 frameNumber)
{
    const qint32 currentVbiFrame = timemasterSeqToVbi.value(frameNumber,
                                   (sourceMinimumVbiFrame[0] - 1) + frameNumber);

    qint32 fieldNumber = 0;
    if constexpr (field == 1)
        fieldNumber = ldDecodeMetaData[0]->getFirstFieldNumber(frameNumber);
    else
        fieldNumber = ldDecodeMetaData[0]->getSecondFieldNumber(frameNumber);

    const LdDecodeMetaData::Field& currentField = ldDecodeMetaData[0]->getField(fieldNumber);
    if (currentField.pad) {
        for (int sourceNo = 1; sourceNo < ldDecodeMetaData.size(); ++sourceNo) {
            if (currentVbiFrame < sourceMinimumVbiFrame[sourceNo] ||
                currentVbiFrame > sourceMaximumVbiFrame[sourceNo]) continue;
            const qint32 currentSourceFrameNumber = sourceVbiMap[sourceNo].value(currentVbiFrame, -1);
            if (currentSourceFrameNumber == -1 ||
                ldDecodeMetaData[sourceNo]->getNumberOfFrames() < currentSourceFrameNumber) continue;
            qint32 otherFieldNumber = 0;
            if constexpr (field == 1)
                otherFieldNumber = ldDecodeMetaData[sourceNo]->getFirstFieldNumber(currentSourceFrameNumber);
            else
                otherFieldNumber = ldDecodeMetaData[sourceNo]->getSecondFieldNumber(currentSourceFrameNumber);
            if (ldDecodeMetaData[sourceNo]->getField(otherFieldNumber).pad) continue;
            LdDecodeMetaData::Field potentialField = ldDecodeMetaData[sourceNo]->getField(otherFieldNumber);
            potentialField.seqNo        = currentField.seqNo;
            potentialField.fieldPhaseID = currentField.fieldPhaseID;
            potentialField.dropOuts     = currentField.dropOuts;
            ldDecodeMetaData[0]->updateField(potentialField, fieldNumber);
            break;
        }
    }
}

LdDecodeMetaData& StackingPool::correctMetaData()
{
    correctPhaseIDs();
    const qint32 frameCount = ldDecodeMetaData[0]->getNumberOfFrames();
    for (qint32 frameNumber = 1; frameNumber <= frameCount; ++frameNumber) {
        replaceFieldMetaData<1>(frameNumber);
        replaceFieldMetaData<2>(frameNumber);
    }
    return *ldDecodeMetaData[0];
}
