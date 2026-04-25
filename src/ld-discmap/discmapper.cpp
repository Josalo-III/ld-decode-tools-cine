/************************************************************************

    discmapper.cpp

    ld-discmap - TBC and VBI alignment and correction
    Copyright (C) 2019-2025 Simon Inns
    Copyright (C) 2025-2026 Joseph Burns

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

#include "discmapper.h"
#include "tbc/logging.h"

#include <iostream>

DiscMapper::DiscMapper()
{
    // This space for sale; please enquire within
}

bool DiscMapper::process(QFileInfo _inputFileInfo, QFileInfo _inputMetadataFileInfo,
                         QFileInfo _outputFileInfo, bool _reverse, bool _mapOnly,
                         bool _noStrict, bool _deleteUnmappable, bool _noAudio)
{
    inputFileInfo         = _inputFileInfo;
    inputMetadataFileInfo = _inputMetadataFileInfo;
    outputFileInfo        = _outputFileInfo;
    reverse               = _reverse;
    mapOnly               = _mapOnly;
    noStrict              = _noStrict;
    deleteUnmappable      = _deleteUnmappable;
    noAudio               = _noAudio;

    qInfo() << "LaserDisc mapping tool";
    qInfo() << "";
    qInfo() << "Please note that disc mapping is not fool-proof - if you";
    qInfo() << "have a disc that does not map correctly run ld-discmap";
    qInfo() << "with the --debug option for more details about the process";
    qInfo() << "(and the additional information required for issue reporting";
    qInfo() << "to the ld-decode project).";
    qInfo() << "";
    qInfo() << "Some early LaserDiscs do not provide frame numbering or";
    qInfo() << "time-code information and cannot be automatically mapped -";
    qInfo() << "if in doubt verify your source TBC file using the ld-analyse";
    qInfo() << "application.";
    qInfo() << "";
    qInfo() << "Note that NTSC CAV pulldown support currently only handles";
    qInfo() << "discs that follow the standard 1-in-5 pulldown pattern.";
    qInfo() << "";

    qInfo().noquote() << "Processing input metadata for" << inputFileInfo.filePath();
    if (noStrict) qInfo() << "Not enforcing strict pulldown checking - this can cause false-positive detection";
    DiscMap discMap(inputMetadataFileInfo, reverse, noStrict);
    if (!discMap.valid()) {
        qInfo() << "Could not process TBC metadata successfully - cannot map this disc";
        return false;
    }
    tbcDebugStream() << discMap;

    qInfo().noquote() << "Input TBC is a" << discMap.discType() << "disc using" << discMap.discFormat();

    removeLeadInOut(discMap);
    removeInvalidFramesByPhase(discMap);
    correctVbiFrameNumbersUsingSequenceAnalysis(discMap);
    removeDuplicateNumberedFrames(discMap);
    numberPulldownFrames(discMap);

    if (!verifyFrameNumberPresence(discMap)) {
        if (mapOnly) {
            // In map-only mode no TBC video file is written, so unmappable frames
            // cannot be deleted from the output. Offer to continue for analysis only.
            qInfo() << "";
            qInfo() << "Unmappable frames detected; deletion requires a TBC video file.";
            qInfo() << "Either cancel and re-run with -u (--delete-unmappable-frames),";
            qInfo() << "or continue to produce a metadata file for analysis only.";
            qInfo() << "";
            qInfo() << "Continue and produce analysis-only metadata? (y/n):";
            char response = 'n';
            std::cin >> response;
            if (response != 'y' && response != 'Y') {
                qInfo() << "Aborting.";
                return false;
            }
            qInfo() << "Continuing - output metadata is for analysis only and cannot be used with a TBC video file.";
        } else if (!deleteUnmappable) {
            qInfo() << "";
            qInfo() << "Disc mapping has failed as there are unmappable frames in the disc map!";
            qInfo() << "It is possible that running ld-discmap again with the --delete-unmappable-frames";
            qInfo() << "option set could rectify this issue.";
            return false;
        } else {
            qInfo() << "Verification has failed, there are unmappable frames...";
            qInfo() << "--delete-unmappable-frames is set, so the unmappable frames will be deleted";
            deleteUnmappableFrames(discMap);
        }
    }

    reorderFrames(discMap);
    padDiscMap(discMap);
    rewriteFrameNumbers(discMap);

    qInfo() << "Disc mapping process completed";

    if (mapOnly) {
        qInfo() << "--maponly selected.  No output file will be written.";
        return true;
    }

    if (noAudio) {
        qInfo() << "-no-audio selected.  No analogue audio output will be written.";
    }

    qInfo() << "Writing output video and metadata information...";
    return saveDiscMap(discMap);
}

void DiscMapper::removeLeadInOut(DiscMap &discMap)
{
    qInfo() << "Checking for lead in and out frames...";
    qint32 leadInOutCounter = 0;
    for (qint32 frameNumber = 0; frameNumber < discMap.numberOfFrames(); frameNumber++) {
        if (discMap.isLeadInOut(frameNumber)) {
            discMap.setMarkedForDeletion(frameNumber);
            leadInOutCounter++;
        }

        if (discMap.vbiFrameNumber(frameNumber) == 0) discMap.debugFrameDetails(frameNumber);

        if (!discMap.isLeadInOut(frameNumber) && discMap.isDiscCav() && discMap.vbiFrameNumber(frameNumber) == 0) {
            qInfo() << "Warning: Frame with illegal CAV frame number of 0 found... Assuming illegal lead in and deleting.";
            discMap.setMarkedForDeletion(frameNumber);
            leadInOutCounter++;
        }
    }

    qInfo() << "Removing" << leadInOutCounter << "frames marked as lead in/out";
    discMap.flush();
}

void DiscMapper::removeInvalidFramesByPhase(DiscMap &discMap)
{
    qInfo() << "Removing invalid frames by phase analysis...";

    qint32 removals = 0;

    for (qint32 frameNumber = 0; frameNumber < discMap.numberOfFrames(); frameNumber++) {
        qint32 expectedNextPhase = discMap.getFirstFieldPhase(frameNumber) + 1;
        if (discMap.isDiscPal() && expectedNextPhase == 9) expectedNextPhase = 1;
        if (!discMap.isDiscPal() && expectedNextPhase == 5) expectedNextPhase = 1;
        if (discMap.getSecondFieldPhase(frameNumber) != expectedNextPhase) {
            if (discMap.vbiFrameNumber(frameNumber) != -1) {
                tbcDebugStream() << "Marking frame" << frameNumber << "for deletion (VBI Frame#" << discMap.vbiFrameNumber(frameNumber)
                                 << ") as first and second field phases are not in sequence! -"
                                 << expectedNextPhase << "expected but got" << discMap.getSecondFieldPhase(frameNumber);
            } else {
                tbcDebugStream() << "Marking frame" << frameNumber << "for deletion (VBI Frame# invalid) as first and second field phases are not in sequence! -"
                                 << expectedNextPhase << "expected but got" << discMap.getSecondFieldPhase(frameNumber);
            }
            discMap.setMarkedForDeletion(frameNumber);
            removals++;
        }
    }

    qInfo() << "Removing" << removals << "frames marked as invalid due to incorrect phase sequence";
    discMap.flush();
}

void DiscMapper::correctVbiFrameNumbersUsingSequenceAnalysis(DiscMap &discMap)
{
    qInfo() << "Correcting frame numbers using sequence analysis...";

    qint32 scanDistance = 10;
    qint32 corrections = 0;

    for (qint32 frameNumber = 0; frameNumber < discMap.numberOfFrames() - scanDistance; frameNumber++) {
        if (!discMap.isPulldown(frameNumber) && discMap.vbiFrameNumber(frameNumber) != -1) {
            qint32 startOfSequence = discMap.vbiFrameNumber(frameNumber);
            qint32 expectedIncrement = 1;

            QVector<bool> vbiGood;
            vbiGood.resize(scanDistance);
            bool sequenceIsGood = true;

            for (qint32 i = 0; i < scanDistance; i++) {
                if (!discMap.isPulldown(frameNumber + i + 1)) {
                    if ((discMap.vbiFrameNumber(frameNumber + i + 1) == startOfSequence + expectedIncrement) ||
                            (discMap.isPulldown(frameNumber + i + 1))) {
                        sequenceIsGood = true;
                    } else {
                        sequenceIsGood = false;
                    }
                    if (sequenceIsGood) vbiGood[i] = true; else vbiGood[i] = false;
                    expectedIncrement++;
                } else {
                    if (sequenceIsGood) vbiGood[i] = true; else vbiGood[i] = false;
                }
            }

            qint32 count = 0;
            for (qint32 i = 0; i < scanDistance; i++) {
                if (vbiGood[i]) count++;
            }

            if (count != scanDistance) {
                qint32 check1 = 0;
                for (qint32 i = 0; i < scanDistance; i++) {
                    if (vbiGood[i] && !discMap.isPulldown(frameNumber + i + 1)) check1++;
                    else if (!discMap.isPulldown(frameNumber + i + 1)) break;
                }

                qint32 check2 = 0;
                for (qint32 i = scanDistance - 1; i >= 0; i--) {
                    if (vbiGood[i] && !discMap.isPulldown(frameNumber + i + 1)) check2++;
                    else if (!discMap.isPulldown(frameNumber + i + 1)) break;
                }

                if (check1 >= 2 && check2 >= 2) {
                    tbcDebugStream() << "Broken VBI frame number sequence detected:";

                    bool inError = false;
                    expectedIncrement = 1;
                    for (qint32 i = 0; i < scanDistance; i++) {
                        if (!vbiGood[i]) {
                            inError = true;
                            if (!discMap.isPulldown(frameNumber + i + 1)) {
                                if ((discMap.vbiFrameNumber(frameNumber + i + 1) != discMap.vbiFrameNumber(frameNumber + i)) &&
                                        discMap.isPhaseCorrect(frameNumber + i + 1)) {
                                    tbcDebugStream() << "  Position BAD   " << i << "Seq."
                                                << discMap.seqFrameNumber(frameNumber + i + 1)
                                                << "VBI was" << discMap.vbiFrameNumber(frameNumber + i + 1)
                                                << "now" << (startOfSequence + expectedIncrement)
                                                << "- Phase" << discMap.getFirstFieldPhase(frameNumber + i + 1) << "/"
                                                << discMap.getSecondFieldPhase(frameNumber + i + 1);
                                    discMap.setVbiFrameNumber(frameNumber + i + 1, startOfSequence + expectedIncrement);
                                    if (!discMap.isPulldown(frameNumber + i + 1)) expectedIncrement++;
                                    corrections++;
                                } else {
                                    if (discMap.isPhaseRepeating(frameNumber + i + 1)) {
                                        tbcDebugStream() << "  Position REPEAT" << i << "Seq."
                                                    << discMap.seqFrameNumber(frameNumber + i + 1)
                                                    << "VBI" << discMap.vbiFrameNumber(frameNumber + i + 1)
                                                    << "- Phase" << discMap.getFirstFieldPhase(frameNumber + i + 1) << "/"
                                                    << discMap.getSecondFieldPhase(frameNumber + i + 1);
                                        tbcDebugStream() << "  Ignoring sequence break as frame is repeating (VBI and phase) rather than out of sequence";
                                        if (inError) break;
                                    }
                                }
                            } else {
                                tbcDebugStream() << "  Position BAD   " << i << "Seq."
                                            << discMap.seqFrameNumber(frameNumber + i + 1)
                                            << "VBI pulldown"
                                            << "- Phase" << discMap.getFirstFieldPhase(frameNumber + i + 1) << "/"
                                            << discMap.getSecondFieldPhase(frameNumber + i + 1);
                            }
                        } else {
                            if (!discMap.isPulldown(frameNumber + i + 1))
                                tbcDebugStream() << "  Position GOOD  " << i << "Seq."
                                            << discMap.seqFrameNumber(frameNumber + i + 1)
                                            << "VBI" << discMap.vbiFrameNumber(frameNumber + i + 1)
                                            << "- Phase" << discMap.getFirstFieldPhase(frameNumber + i + 1) << "/"
                                            << discMap.getSecondFieldPhase(frameNumber + i + 1);
                            else tbcDebugStream() << "  Position GOOD  " << i << "Seq."
                                             << discMap.seqFrameNumber(frameNumber + i + 1)
                                             << "VBI pulldown"
                                             << "- Phase" << discMap.getFirstFieldPhase(frameNumber + i + 1) << "/"
                                             << discMap.getSecondFieldPhase(frameNumber + i + 1);

                            if (!discMap.isPulldown(frameNumber + i + 1)) expectedIncrement++;
                            if (inError) break;
                        }
                    }
                }
            }
        }
    }

    qInfo() << "Sequence analysis corrected" << corrections << "frame numbers";
}

void DiscMapper::removeDuplicateNumberedFrames(DiscMap &discMap)
{
    qInfo() << "Searching for duplicate frames";
    tbcDebugStream() << "Building list of VBIs that have more than one entry in the discmap...";
    QVector<qint32> duplicatedFrameList;
    duplicatedFrameList.reserve(discMap.numberOfFrames());
    for (qint32 frameNumber = 0; frameNumber < discMap.numberOfFrames(); frameNumber++) {
        if (!discMap.isPulldown(frameNumber)) {
            for (qint32 i = frameNumber + 1; i < discMap.numberOfFrames(); i++) {
                if (discMap.vbiFrameNumber(frameNumber) == discMap.vbiFrameNumber(i) && !discMap.isPulldown(i)) {
                    duplicatedFrameList.append(discMap.vbiFrameNumber(frameNumber));
                }
            }
        }
    }

    tbcDebugStream() << "Sorting the duplicated frame list into numerical order...";
    std::sort(duplicatedFrameList.begin(), duplicatedFrameList.end());
    tbcDebugStream() << "Removing any repeated frame numbers from the duplicated frame list...";
    auto last = std::unique(duplicatedFrameList.begin(), duplicatedFrameList.end());
    duplicatedFrameList.erase(last, duplicatedFrameList.end());

    tbcDebugStream() << "Found" << duplicatedFrameList.size() << "VBI frame numbers with more than 1 entry in the discmap";

    for (qint32 i = 0; i < duplicatedFrameList.size(); i++) {
        if (duplicatedFrameList[i] != -1) {
            tbcDebugStream() << "VBI Frame number" << duplicatedFrameList[i] << "has duplicates; searching for them...";
            QVector<qint32> discMapDuplicateAddress;
            for (qint32 frameNumber = 0; frameNumber < discMap.numberOfFrames(); frameNumber++) {
                if (discMap.vbiFrameNumber(frameNumber) == duplicatedFrameList[i]) {
                    discMapDuplicateAddress.append(frameNumber);
                }
            }

            tbcDebugStream() << "  Found" << discMapDuplicateAddress.size() << "duplicates of VBI frame" << duplicatedFrameList[i];

            qint32 bestDiscMapFrame = discMapDuplicateAddress.first();
            for (qint32 i = 0; i < discMapDuplicateAddress.size(); i++) {
                if (discMap.frameQuality(bestDiscMapFrame) < discMap.frameQuality(discMapDuplicateAddress[i])) {
                    bestDiscMapFrame = discMapDuplicateAddress[i];
                }
            }

            tbcDebugStream() << "  Highest quality duplicate of VBI" << duplicatedFrameList[i] << "is sequential frame"
                        << discMap.seqFrameNumber(bestDiscMapFrame) << "with a quality of" << discMap.frameQuality(bestDiscMapFrame);

            for (qint32 i = 0; i < discMapDuplicateAddress.size(); i++) {
                if (discMapDuplicateAddress[i] != bestDiscMapFrame) {
                    discMap.setMarkedForDeletion(discMapDuplicateAddress[i]);
                }
            }
        } else {
            if (!discMap.isDiscPal()) {
                qInfo() << "";
                qInfo() << "Warning:";
                qInfo() << "There are frames without a frame number (that are not flagged as pulldown) in the duplicate frame list";
                qInfo() << "This probably means that the disc map contains pulldown frames that do not follow the normal 1 in 5";
                qInfo() << "pulldown pattern - and disc mapping will likely fail!";
                qInfo() << "";
            } else {
                qInfo() << "";
                qInfo() << "Warning:";
                qInfo() << "There are frames without a frame number in the duplicate frame list.  Since numberless frames are";
                qInfo() << "usually unmappable, disc mapping will likely fail unless the --delete-unmappable-frames option is";
                qInfo() << "used.";
                qInfo() << "";
            }
        }
    }

    qint32 originalSize = discMap.numberOfFrames();
    discMap.flush();
    qInfo() << "Removed" << originalSize - discMap.numberOfFrames() <<
               "duplicate frames - disc map size now" << discMap.numberOfFrames() << "frames";
}

void DiscMapper::numberPulldownFrames(DiscMap &discMap)
{
    if (discMap.isDiscCav() && !discMap.isDiscPal()) {
        qInfo() << "Numbering pulldown frames in the disc map...";

        for (qint32 i = 1; i < discMap.numberOfFrames(); i++) {
            if (discMap.isPulldown(i)) discMap.setVbiFrameNumber(i, discMap.vbiFrameNumber(i - 1));
        }

        if (discMap.isPulldown(0)) {
            discMap.setVbiFrameNumber(0, discMap.vbiFrameNumber(1) - 1);
            qInfo() << "Attempted to number pulldown frames, but first frame is a pulldown.";
            qInfo() << "This probably isn't a good thing, but continuing anyway...";
        }

        qInfo() << "Numbering complete";
    }
}

bool DiscMapper::verifyFrameNumberPresence(DiscMap &discMap)
{
    qInfo() << "Verifying frame numbers are present for all frames in the disc map...";
    for (qint32 frameNumber = 0; frameNumber < discMap.numberOfFrames(); frameNumber++) {
        if (discMap.vbiFrameNumber(frameNumber) < 0) {
            qInfo() << "Verification failed - First failed frame was" << discMap.seqFrameNumber(frameNumber);
            discMap.debugFrameDetails(frameNumber);
            return false;
        }
    }
    qInfo() << "Verification successful";
    return true;
}

void DiscMapper::reorderFrames(DiscMap &discMap)
{
    qInfo() << "Sorting the disc map into numerical frame order...";

    discMap.sort();
    qInfo() << "Sorting complete";

    if (discMap.numberOfFrames() > 2) {
        qint32 frameNumber = 0;
        qint32 initialGap = discMap.vbiFrameNumber(frameNumber + 1) - discMap.vbiFrameNumber(frameNumber);

        if (initialGap > 1000) {
            qInfo() << "Warning: The gap between the first and second VBI number is" << initialGap;
            qInfo() << "this is over the 1000 frame threshold, so the first frame will be deleted to";
            qInfo() << "avoid generating a big gap.";

            discMap.setMarkedForDeletion(frameNumber);
            discMap.flush();
            qInfo() << "Removed first frame - disc map size now" << discMap.numberOfFrames() << "frames";
        }
    }
}

void DiscMapper::padDiscMap(DiscMap &discMap)
{
    qInfo() << "Looking for sequence gaps to pad in the disc map...";

    qint32 numberOfGaps = 0;
    qint32 totalMissingFrames = 0;
    qint32 clvOffsetFrames = 0;
    QVector<qint32> startFrame;
    QVector<qint32> paddingLength;

    for (qint32 frameNumber = 0; frameNumber < discMap.numberOfFrames() - 1; frameNumber++) {
        if (discMap.vbiFrameNumber(frameNumber) + 1 != discMap.vbiFrameNumber(frameNumber + 1)) {
            if (discMap.isPulldown(frameNumber)) {
                // Pulldown frames share a VBI number with the preceding frame;
                // the gap check does not apply across them.
            } else {
                if (discMap.isPulldown(frameNumber + 1)) {
                    if (discMap.vbiFrameNumber(frameNumber) + 1 != discMap.vbiFrameNumber(frameNumber + 2)) {
                        if ((discMap.vbiFrameNumber(frameNumber + 2) - discMap.vbiFrameNumber(frameNumber)) != 0) {
                            tbcDebugStream() << "Sequence break over pulldown: Current VBI frame is" << discMap.vbiFrameNumber(frameNumber)
                                        << "next frame (+1) is" << discMap.vbiFrameNumber(frameNumber + 2) << "gap of"
                                        << discMap.vbiFrameNumber(frameNumber + 2) - discMap.vbiFrameNumber(frameNumber) << "frames";

                            numberOfGaps++;
                            qint32 missingFrames = discMap.vbiFrameNumber(frameNumber + 2) - discMap.vbiFrameNumber(frameNumber);
                            totalMissingFrames += missingFrames;
                            startFrame.append(frameNumber);
                            paddingLength.append(missingFrames);
                        } else {
                            if ((discMap.vbiFrameNumber(frameNumber + 2) - discMap.vbiFrameNumber(frameNumber)) == 0) {
                               qInfo() << "Warning: zero-length gap detected across a pulldown boundary - this frame";
                               qInfo() << "will be skipped. If this causes problems, re-run with --debug for more detail.";
                            }
                        }
                    }
                } else {
                    if (!discMap.isClvOffset(frameNumber)) {
                        qint32 gapLength = discMap.vbiFrameNumber(frameNumber + 1) - discMap.vbiFrameNumber(frameNumber) - 1;
                        tbcDebugStream() << "Sequence break: Current VBI frame is" << discMap.vbiFrameNumber(frameNumber)
                                    << "next frame is" << discMap.vbiFrameNumber(frameNumber + 1) << "gap of"
                                    << gapLength << "frames";

                        if (gapLength > 1000) {
                            qInfo() << "Warning: Detected a sequence break between VBI frame" << discMap.vbiFrameNumber(frameNumber) << "and"
                                       << "VBI frame" << discMap.vbiFrameNumber(frameNumber + 1) << "representing";
                            qInfo() << "a gap of" << gapLength << "frames.  This is over the threshold of 1000 frames and could indicate that mapping";
                            qInfo() << "has failed due to badly corrupted VBI frame number data in the source TBC file.";
                        }

                        numberOfGaps++;
                        qint32 missingFrames = discMap.vbiFrameNumber(frameNumber + 1) - discMap.vbiFrameNumber(frameNumber) - 1;
                        totalMissingFrames += missingFrames;
                        startFrame.append(frameNumber);
                        paddingLength.append(missingFrames);
                    } else {
                        clvOffsetFrames++;
                    }
                }
            }
        }
    }

    for (qint32 i = 0; i < startFrame.size(); i++) {
        discMap.addPadding(startFrame[i], paddingLength[i]);
    }

    if (totalMissingFrames > 0) {
        discMap.sort();
    }

    if (numberOfGaps > 0) {
        qInfo() << "Found" << numberOfGaps << "gaps representing" << totalMissingFrames << "missing frames in the disc map";
        qInfo() << "Note: The disc map has been padded.  This means that there were missing frames";
        qInfo() << "that will be represented by black frames in the output video.";
    } else qInfo() << "No gaps found in the disc map";

    if (clvOffsetFrames > 0) qInfo() << "There were" << clvOffsetFrames << "CLV timecode offsets in the disc map";
    qInfo() << "After padding the disc map contains" << discMap.numberOfFrames() << "frames";
}

void DiscMapper::rewriteFrameNumbers(DiscMap &discMap)
{
    if (discMap.isDiscCav() && !discMap.isDiscPal()) {
        qInfo() << "Searching disc map for pulldown frames...";
        bool present = false;
        for (qint32 frameNumber = 0; frameNumber < discMap.numberOfFrames(); frameNumber++) {
            if (discMap.isPulldown(frameNumber)) {
                present = true;
                break;
            }
        }

        if (present) qInfo() << "Search complete; pulldown frames are present.  Renumbering disc map...";
        else {
            qInfo() << "Search complete; no pulldown frames present";
            return;
        }

        qint32 newVbi = discMap.vbiFrameNumber(0);
        for (qint32 frameNumber = 0; frameNumber < discMap.numberOfFrames(); frameNumber++) {
            discMap.setVbiFrameNumber(frameNumber, newVbi);
            newVbi++;
        }

        qInfo() << "Renumbering complete";
    }
}

void DiscMapper::deleteUnmappableFrames(DiscMap &discMap)
{
    qInfo() << "Deleting unmappable frames from the disc map...";
    for (qint32 frameNumber = 0; frameNumber < discMap.numberOfFrames(); frameNumber++) {
        if (discMap.vbiFrameNumber(frameNumber) < 0 && !discMap.isPulldown(frameNumber)) {
            tbcDebugStream() << "Marking frame" << frameNumber << "for deletion (unmappable)";
            discMap.setMarkedForDeletion(frameNumber);
        }
    }
    discMap.flush();
    qInfo() << "Deletion successful";
}

bool DiscMapper::saveDiscMap(DiscMap &discMap)
{
    SourceVideo sourceVideo;
    sourceVideo.open(inputFileInfo.filePath(), discMap.getVideoFieldLength());

    QFile targetVideo(outputFileInfo.filePath());
    if (!targetVideo.open(QIODevice::WriteOnly)) {
        qInfo() << "Cannot open target video file:" << outputFileInfo.filePath();
        sourceVideo.close();
        return false;
    }

    SourceAudio sourceAudio;
    QFile targetAudio;

    if (!noAudio) {
        if (!sourceAudio.open(inputFileInfo)) {
            qInfo() << "Cannot open source audio file:" << inputFileInfo.absolutePath() + "/" + inputFileInfo.completeBaseName() + ".pcm";
            sourceVideo.close();
            sourceAudio.close();
            return false;
        }

        targetAudio.setFileName(outputFileInfo.absolutePath() + "/" + outputFileInfo.completeBaseName() + ".pcm");
        if (targetAudio.exists()) {
            qInfo() << "Target audio file already exists:" << targetAudio.fileName() << "- Cannot proceed!";
            sourceVideo.close();
            sourceAudio.close();
            return false;
        }
        if (!targetAudio.open(QIODevice::WriteOnly)) {
            qInfo() << "Cannot open target audio file:" << targetAudio.fileName();
            sourceVideo.close();
            sourceAudio.close();
            return false;
        }
    }

    SourceVideo::Data missingFieldData;
    missingFieldData.fill(0, discMap.getVideoFieldLength());

    SourceAudio::Data missingFieldAudioData;
    missingFieldAudioData.fill(0, discMap.getApproximateAudioFieldLength());

    SourceVideo::Data sourceFirstField;
    SourceVideo::Data sourceSecondField;
    SourceAudio::Data sourceAudioFirstField;
    SourceAudio::Data sourceAudioSecondField;

    qInfo() << "Saving target video frames...";
    qint32 notifyInterval = discMap.numberOfFrames() / 50;
    if (notifyInterval < 1) notifyInterval = 1;

    for (qint32 frameNumber = 0; frameNumber < discMap.numberOfFrames(); frameNumber++) {
        bool writeFail = false;

        if (!discMap.isPadded(frameNumber)) {
            qint32 firstFieldNumber  = discMap.getFirstFieldNumber(frameNumber);
            qint32 secondFieldNumber = discMap.getSecondFieldNumber(frameNumber);
            sourceFirstField  = sourceVideo.getVideoField(firstFieldNumber);
            sourceSecondField = sourceVideo.getVideoField(secondFieldNumber);

            if (firstFieldNumber < secondFieldNumber) {
                if (!targetVideo.write(reinterpret_cast<const char *>(sourceFirstField.data()),
                                       sourceFirstField.size() * 2)) writeFail = true;
                if (!targetVideo.write(reinterpret_cast<const char *>(sourceSecondField.data()),
                                       sourceSecondField.size() * 2)) writeFail = true;
            } else {
                if (!targetVideo.write(reinterpret_cast<const char *>(sourceSecondField.data()),
                                       sourceSecondField.size() * 2)) writeFail = true;
                if (!targetVideo.write(reinterpret_cast<const char *>(sourceFirstField.data()),
                                       sourceFirstField.size() * 2)) writeFail = true;
            }

            if (!noAudio) {
                if ((discMap.getFirstFieldAudioDataLength(frameNumber) > 0) &&
                        (discMap.getSecondFieldAudioDataLength(frameNumber) > 0)) {
                    sourceAudioFirstField  = sourceAudio.getAudioData(discMap.getFirstFieldAudioDataStart(frameNumber),
                                                                      discMap.getFirstFieldAudioDataLength(frameNumber));
                    sourceAudioSecondField = sourceAudio.getAudioData(discMap.getSecondFieldAudioDataStart(frameNumber),
                                                                      discMap.getSecondFieldAudioDataLength(frameNumber));

                    if (!targetAudio.write(reinterpret_cast<const char *>(sourceAudioFirstField.data()),
                                           sourceAudioFirstField.size() * 2)) writeFail = true;
                    if (!targetAudio.write(reinterpret_cast<const char *>(sourceAudioSecondField.data()),
                                           sourceAudioSecondField.size() * 2)) writeFail = true;
                } else {
                    if (discMap.getFirstFieldAudioDataLength(frameNumber) < 1) {
                        qInfo() << "Warning: Input file seems to have zero audio data in the first field of frame number #" << frameNumber;
                        qInfo() << "The audio output might be corrupt.";
                    }
                    if (discMap.getSecondFieldAudioDataLength(frameNumber) < 1) {
                        qInfo() << "Warning: Input file seems to have zero audio data in the second field of frame number #" << frameNumber;
                        qInfo() << "The audio output might be corrupt.";
                    }
                }
            }
        } else {
            if (!targetVideo.write(reinterpret_cast<const char *>(missingFieldData.data()),
                                   missingFieldData.size() * 2)) writeFail = true;
            if (!targetVideo.write(reinterpret_cast<const char *>(missingFieldData.data()),
                                   missingFieldData.size() * 2)) writeFail = true;

            if (!noAudio) {
                if (!targetAudio.write(reinterpret_cast<const char *>(missingFieldAudioData.data()),
                                       missingFieldAudioData.size() * 2)) writeFail = true;
                if (!targetAudio.write(reinterpret_cast<const char *>(missingFieldAudioData.data()),
                                       missingFieldAudioData.size() * 2)) writeFail = true;
            }
        }

        if (frameNumber % notifyInterval == 0)
            qInfo() << "Written frame" << frameNumber << "of" << discMap.numberOfFrames();

        if (writeFail) {
            qWarning() << "Writing fields to the target TBC file failed on frame number" << frameNumber;
            targetVideo.close();
            sourceVideo.close();
            return false;
        }
    }
    qInfo() << discMap.numberOfFrames() << "video frames saved";

    targetVideo.close();
    sourceVideo.close();

    if (!noAudio) {
        qInfo() << "Target audio frames saved";
        targetAudio.close();
        sourceAudio.close();
    }

    qInfo() << "Saving target video metadata...";
    QFileInfo outputMetadataFileInfo(outputFileInfo.filePath() + ".db");
    if (!discMap.saveTargetMetadata(outputMetadataFileInfo)) {
        qInfo() << "Writing target metadata failed!";
        return false;
    }
    qInfo() << "Target video metadata saved";
    return true;
}
