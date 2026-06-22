/************************************************************************

    f2_stacker.cpp

    efm-stacker-f2 - EFM F2 Section stacker
    Copyright (C) 2025 Simon Inns

    This file is part of ld-decode-tools.

    This application is free software: you can redistribute it and/or
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

#include "f2_stacker.h"
#include "tbc/logging.h"

#include <QHash>
#include <QSet>

F2Stacker::F2Stacker() :
    m_noValidValueForByte(0),
    m_validValueForByte(0),
    m_usedMostCommonValue(0),
    m_tiedValueForByte(0),
    m_errorFreeFrames(0),
    m_errorFrames(0),
    m_paddedFrames(0),
    m_flatSections(0)
{}

bool F2Stacker::process(const QVector<QString> &inputFilenames, const QString &outputFilename)
{
    // Prepare the source differences statistics
    m_sourceDifferences.resize(inputFilenames.size());
    m_sourceDifferences.fill(0);
    m_sourceMissingSections.resize(inputFilenames.size());
    m_sourceMissingSections.fill(0);

    // Start by opening all the input F2 section files
    for (int index = 0; index < inputFilenames.size(); index++) {
        ReaderF2Section* reader = new ReaderF2Section();
        if (!reader->open(inputFilenames[index])) {
            qCritical() << "F2Stacker::process() - Could not open input file" << inputFilenames[index];
            delete reader;
            return false;
        }
        m_inputFiles.append(reader);
        tbcDebugStream() << "Opened input file" << inputFilenames[index];
    }

    QVector<SourceState> sourceStates;
    sourceStates.resize(m_inputFiles.size());

    qInfo() << "Preparing input sources...";
    for (int inputFileIdx = 0; inputFileIdx < m_inputFiles.size(); inputFileIdx++) {
        sourceStates[inputFileIdx].reader = m_inputFiles[inputFileIdx];
        sourceStates[inputFileIdx].reader->seekToSection(0);
        if (!readNextValidSection(sourceStates[inputFileIdx])) {
            qCritical() << "F2Stacker::process() - Input file" << inputFilenames[inputFileIdx]
                        << "does not contain any sections with valid metadata";
            return false;
        }

        qInfo().noquote() << "Input File" << inputFilenames[inputFileIdx]
                          << "- First valid section:"
                          << sourceStates[inputFileIdx].currentSection.metadata.absoluteSectionTime().toString()
                          << "- Total sections:" << m_inputFiles[inputFileIdx]->size();
    }

    // Open the output file
    if (!m_outputFile.open(outputFilename)) {
        qCritical() << "F2Stacker::process() - Could not open output file" << outputFilename;
        return false;
    }

    qint32 stackStartAddress = -1;
    qint32 stackEndAddress = -1;
    quint64 sectionsStacked = 0;

    // Process sources in timestamp order.  Each source is read forward once; missing
    // or duplicate sections affect only that source rather than shifting later votes.
    while (true) {
        qint32 address = -1;
        for (int sourceIndex = 0; sourceIndex < sourceStates.size(); sourceIndex++) {
            if (!sourceStates[sourceIndex].hasCurrentSection) {
                continue;
            }

            qint32 sourceAddress =
                    sourceStates[sourceIndex].currentSection.metadata.absoluteSectionTime().frames();
            if (address < 0 || sourceAddress < address) {
                address = sourceAddress;
            }
        }

        if (address < 0) {
            break;
        }

        if (stackStartAddress < 0) {
            stackStartAddress = address;
        }
        stackEndAddress = address;

        QVector<F2Section> sectionList;
        QVector<int> sectionSourceIndexes;

        for (int sourceIndex = 0; sourceIndex < sourceStates.size(); sourceIndex++) {
            SourceState &sourceState = sourceStates[sourceIndex];
            if (!sourceState.hasCurrentSection) {
                continue;
            }

            qint32 sourceAddress = sourceState.currentSection.metadata.absoluteSectionTime().frames();
            if (sourceAddress > address) {
                m_sourceMissingSections[sourceIndex]++;
                continue;
            }

            F2Section bestSection = sourceState.currentSection;
            quint32 bestScore = sectionScore(bestSection);
            while (readNextValidSection(sourceState, address)) {
                qint32 nextAddress =
                        sourceState.currentSection.metadata.absoluteSectionTime().frames();
                if (nextAddress != address) {
                    break;
                }

                sourceState.duplicateSections++;
                quint32 nextScore = sectionScore(sourceState.currentSection);
                if (nextScore < bestScore) {
                    bestSection = sourceState.currentSection;
                    bestScore = nextScore;
                    sourceState.replacedDuplicateSections++;
                }
            }

            sectionList.append(bestSection);
            sectionSourceIndexes.append(sourceIndex);
        }

        if (sectionList.isEmpty()) {
            qCritical().noquote() << "F2Stacker::process() - No sources available for section"
                                  << SectionTime(address).toString();
            return false;
        }

        tbcDebugStream().noquote() << "F2Stacker::process() - Stacking section" << SectionTime(address).toString();

        F2Section stackedF2Section = stackSections(sectionList, sectionSourceIndexes);

        // Write the output F2 Section
        m_outputFile.write(stackedF2Section);
        sectionsStacked++;

        // Every 2500 Sections, show progress
        if (sectionsStacked % 2500 == 0) {
            qInfo().noquote().nospace() << "Processed " << sectionsStacked
                << " sections, current section " << SectionTime(address).toString();
        }
    }

    // Close the input files
    for (int index = 0; index < m_inputFiles.size(); index++) {
        m_inputFiles[index]->close();
        delete m_inputFiles[index];
    }
    m_inputFiles.clear();
    
    // Close the output file
    m_outputFile.close();

    // Statistics
    qInfo() << "Stacking results:";
    qInfo().noquote() << "  Start time:" << SectionTime(stackStartAddress).toString();
    qInfo().noquote() << "  End time:" << SectionTime(stackEndAddress).toString();
    qInfo().noquote() << "  Sections stacked:" << sectionsStacked;
    qInfo().noquote() << "  Frames stacked:" << sectionsStacked * 98;
    qInfo().noquote() << "";
    qInfo().noquote() << "  Error free frames:" << m_errorFreeFrames;
    qInfo().noquote() << "  Error frames:" << m_errorFrames;
    qInfo().noquote().nospace() << "  Padded frames: " << m_paddedFrames << " (" << m_paddedFrames / 98 << " sections)";
    qInfo().noquote() << "  Flat source sections excluded:" << m_flatSections;
    qInfo().noquote() << "  Total frames:" << m_errorFreeFrames + m_errorFrames + m_paddedFrames;
    qInfo().noquote() << "";
    qInfo().noquote() << "  Valid bytes common to all sources:" << m_validValueForByte;
    qInfo().noquote() << "  Valid bytes that differed in value between sources:" << m_usedMostCommonValue;
    qInfo().noquote() << "  Tied valid byte values marked as errors:" << m_tiedValueForByte;
    qInfo().noquote() << "  Invalid byte in all sources:" << m_noValidValueForByte;
    qInfo().noquote() << "";
    qInfo().noquote() << "  Source differences:";
    for (int sourceIndex = 0; sourceIndex < m_sourceDifferences.size(); sourceIndex++) {
        qInfo().noquote() << "    Source" << sourceIndex << inputFilenames[sourceIndex] << ":" << m_sourceDifferences[sourceIndex];
    }
    qInfo().noquote() << "";
    qInfo().noquote() << "  Source missing sections:";
    for (int sourceIndex = 0; sourceIndex < m_sourceMissingSections.size(); sourceIndex++) {
        qInfo().noquote() << "    Source" << sourceIndex << inputFilenames[sourceIndex] << ":" << m_sourceMissingSections[sourceIndex];
    }
    qInfo().noquote() << "";
    qInfo().noquote() << "  Source irregularities:";
    for (int sourceIndex = 0; sourceIndex < sourceStates.size(); sourceIndex++) {
        qInfo().noquote() << "    Source" << sourceIndex << inputFilenames[sourceIndex]
                          << "- Invalid metadata:" << sourceStates[sourceIndex].invalidMetadataSections
                          << "- Duplicates:" << sourceStates[sourceIndex].duplicateSections
                          << "- Replaced duplicates:" << sourceStates[sourceIndex].replacedDuplicateSections
                          << "- Out of order:" << sourceStates[sourceIndex].outOfOrderSections;
    }

    return true;
}

bool F2Stacker::readNextValidSection(SourceState &sourceState, qint32 minimumAddress)
{
    while (sourceState.nextSectionNumber < sourceState.reader->size()) {
        F2Section section = sourceState.reader->read();
        sourceState.nextSectionNumber++;

        if (!section.metadata.isValid()) {
            sourceState.invalidMetadataSections++;
            continue;
        }

        qint32 address = section.metadata.absoluteSectionTime().frames();
        if (minimumAddress >= 0 && address < minimumAddress) {
            sourceState.outOfOrderSections++;
            continue;
        }
        if (sourceState.lastReadAddress >= 0 && address < sourceState.lastReadAddress) {
            sourceState.outOfOrderSections++;
            if (minimumAddress >= 0 && address <= minimumAddress) {
                continue;
            }
        }
        sourceState.lastReadAddress = address;
        sourceState.currentSection = section;
        sourceState.hasCurrentSection = true;
        return true;
    }

    sourceState.hasCurrentSection = false;
    sourceState.endOfFile = true;
    return false;
}

quint32 F2Stacker::sectionScore(const F2Section &section) const
{
    quint32 score = 0;

    if (section.metadata.isRepaired()) {
        score += 100000;
    }

    for (int frameIndex = 0; frameIndex < 98; frameIndex++) {
        const F2Frame frame = section.frame(frameIndex);
        score += frame.errorData().count(true);
        score += frame.paddedData().count(true);
    }

    if (isFlatSection(section)) {
        score += 100000;
    }

    return score;
}

bool F2Stacker::isPaddingSection(const F2Section &section) const
{
    for (int frameIndex = 0; frameIndex < 98; frameIndex++) {
        if (section.frame(frameIndex).paddedData().contains(false)) {
            return false;
        }
    }

    return true;
}

bool F2Stacker::isFlatSection(const F2Section &section) const
{
    QHash<quint8, int> byteCounts;
    QSet<QByteArray> uniqueFrames;
    int totalBytes = 0;

    for (int frameIndex = 0; frameIndex < 98; frameIndex++) {
        const F2Frame frame = section.frame(frameIndex);
        const QVector<quint8> data = frame.data();
        QByteArray frameBytes;
        frameBytes.reserve(data.size());

        for (int byteIndex = 0; byteIndex < data.size(); byteIndex++) {
            byteCounts[data[byteIndex]]++;
            frameBytes.append(static_cast<char>(data[byteIndex]));
            totalBytes++;
        }

        uniqueFrames.insert(frameBytes);
    }

    int dominantByteCount = 0;
    for (auto it = byteCounts.begin(); it != byteCounts.end(); ++it) {
        if (it.value() > dominantByteCount) {
            dominantByteCount = it.value();
        }
    }

    const int repeatedFrames = 98 - uniqueFrames.size();
    if (totalBytes == 0) {
        return true;
    }

    // A real F2 section, even for quiet audio, should have plenty of byte and
    // frame variation because parity and interleaving are still present.  This
    // catches collapsed/flat captures that arrive with clean-looking flags.
    return byteCounts.size() <= 8 ||
           repeatedFrames >= 80 ||
           dominantByteCount > (totalBytes * 85 / 100);
}

F2Section F2Stacker::stackSections(const QVector<F2Section> &f2Sections, const QVector<int> &sourceIndexes)
{
    F2Section stackedSection;
    SectionMetadata stackedMetadata;

    // Pick the first section from the list with valid, unrepaired metadata.
    bool gotValidMetadata = false;
    for (int sectionIndex = 0; sectionIndex < f2Sections.size(); sectionIndex++) {
        if (f2Sections[sectionIndex].metadata.isValid() &&
            !f2Sections[sectionIndex].metadata.isRepaired()) {
            stackedMetadata = f2Sections[sectionIndex].metadata;
            gotValidMetadata = true;
            break;
        }
    }

    // If we didn't get anything valid, try again and include repaired metadata.
    if (!gotValidMetadata) {
        for (int sectionIndex = 0; sectionIndex < f2Sections.size(); sectionIndex++) {
            if (f2Sections[sectionIndex].metadata.isValid()) {
                stackedMetadata = f2Sections[sectionIndex].metadata;
                gotValidMetadata = true;
                break;
            }
        }
    }

    if (!gotValidMetadata) {
        qFatal("F2Stacker::stackSections - No valid metadata found in the input sections");
    }

    // Remove sections that are entirely padding.
    QVector<F2Section> validF2Sections;
    QVector<int> validSourceIndexes;
    for (int sectionIndex = 0; sectionIndex < f2Sections.size(); sectionIndex++) {
        if (isPaddingSection(f2Sections[sectionIndex])) {
            tbcDebugStream().noquote()
                << "F2Stacker::stackSections - Section" << sectionIndex
                << "is just padding";
        } else if (isFlatSection(f2Sections[sectionIndex])) {
            tbcDebugStream().noquote()
                << "F2Stacker::stackSections - Section" << sectionIndex
                << "is flat data";
            m_flatSections++;
        } else {
            validF2Sections.append(f2Sections[sectionIndex]);
            validSourceIndexes.append(sourceIndexes[sectionIndex]);
        }
    }

    if (validF2Sections.isEmpty()) {
        // All candidate sections are padding. Preserve the first candidate section.
        stackedSection = f2Sections[0];
        m_paddedFrames += 98;
    } else if (validF2Sections.size() == 1) {
        // Exactly one source has real data for this section. Pass it through.
        stackedSection = validF2Sections[0];

        for (int frameIndex = 0; frameIndex < 98; frameIndex++) {
            if (stackedSection.frame(frameIndex).errorData().contains(1)) {
                m_errorFrames++;
            } else {
                m_errorFreeFrames++;
            }
        }
    } else {
        // Each section contains 98 F2Frames.
        for (int frameIndex = 0; frameIndex < 98; frameIndex++) {
            QVector<F2Frame> frameList;

            for (int sectionIndex = 0; sectionIndex < validF2Sections.size(); sectionIndex++) {
                frameList.append(validF2Sections[sectionIndex].frame(frameIndex));
            }

            F2Frame stackedFrame = stackFrames(frameList, validSourceIndexes);
            stackedSection.pushFrame(stackedFrame);

            if (stackedFrame.errorData().contains(1)) {
                m_errorFrames++;
            } else {
                m_errorFreeFrames++;
            }
        }
    }

    stackedSection.metadata = stackedMetadata;
    return stackedSection;
}

F2Frame F2Stacker::stackFrames(QVector<F2Frame> &f2Frames, const QVector<int> &sourceIndexes)
{
    F2Frame stackedFrame;

    // Process one byte at a time
    QVector<quint8> stackedFrameData;
    QVector<bool> stackedFrameErrorData;
    for (int byteIndex = 0; byteIndex < 32; byteIndex++) {
        // Make a list of the bytes to stack (i.e. those without error flags)
        QVector<quint8> validBytes;
        for (int listIndex = 0; listIndex < f2Frames.size(); ++listIndex) {
            if (f2Frames.at(listIndex).errorData().at(byteIndex) == false) {
                validBytes.append(f2Frames.at(listIndex).data().at(byteIndex));
            }
        }

        if (validBytes.size() == 0) {
            // All bytes are errors - can't correct
            stackedFrameData.append(f2Frames.at(0).data().at(byteIndex));
            stackedFrameErrorData.append(true);
            tbcDebugStream() << "F2Stacker::stackFrames - No valid byte value for index" << byteIndex;
            m_noValidValueForByte++;
        } else {
            // Are all the valid bytes the same value?
            bool allBytesSame = true;
            for (int byteIndex = 1; byteIndex < validBytes.size(); byteIndex++) {
                if (validBytes.at(byteIndex) != validBytes.at(0)) {
                    allBytesSame = false;
                    continue;
                }
            }

            // If all valid bytes are the same, use that value.
            if (allBytesSame) {
                stackedFrameData.append(validBytes.at(0));
                stackedFrameErrorData.append(false);
                m_validValueForByte++;
                continue;
            } else {
                // If all valid bytes aren't the same, calculate the most common
                // byte value from the available valid bytes and use that value
                QHash<quint8, int> byteCounts;
                for (int byteIndex = 0; byteIndex < validBytes.size(); byteIndex++) {
                    byteCounts[validBytes.at(byteIndex)]++;
                }

                // Find the most common byte value.  If there is a tie, keep the
                // byte marked as an erasure instead of blessing an arbitrary source.
                int maxCount = 0;
                int tiedMaxCount = 0;
                quint8 mostCommonByte = 0;
                for (auto it = byteCounts.begin(); it != byteCounts.end(); ++it) {
                    if (it.value() > maxCount) {
                        maxCount = it.value();
                        mostCommonByte = it.key();
                        tiedMaxCount = 1;
                    } else if (it.value() == maxCount) {
                        tiedMaxCount++;
                    }
                }

                QString validBytesString;
                for (int byteIndex = 0; byteIndex < validBytes.size(); byteIndex++) {
                    validBytesString.append(QString("%1 ").arg(validBytes.at(byteIndex), 2, 16, QChar('0')).toUpper());
                }
                QString mostCommonByteString = QString("%1").arg(mostCommonByte, 2, 16, QChar('0')).toUpper();

                if (tiedMaxCount > 1) {
                    tbcDebugStream().noquote()
                        << "F2Stacker::stackFrames - Valid byte values tie - marking byte as error from"
                        << validBytesString;
                    stackedFrameData.append(validBytes.at(0));
                    stackedFrameErrorData.append(true);
                    m_tiedValueForByte++;
                } else {
                    tbcDebugStream().noquote() << "F2Stacker::stackFrames - Valid byte values differ - using"
                        << mostCommonByteString << "from" << validBytesString;

                    stackedFrameData.append(mostCommonByte);
                    stackedFrameErrorData.append(false);
                    m_usedMostCommonValue++;
                }
            }
        }

        // Update the source differences statistics for this byte
        quint8 expectedValue = f2Frames.at(0).data().at(byteIndex);
        for (int sourceIndex = 0; sourceIndex < f2Frames.size(); sourceIndex++) {
            if (f2Frames.at(sourceIndex).data().at(byteIndex) != expectedValue) {
                m_sourceDifferences[sourceIndexes[sourceIndex]]++;
            }
        }
    }

    // Set the data for the stacked frame
    stackedFrame.setData(stackedFrameData);
    stackedFrame.setErrorData(stackedFrameErrorData);

    return stackedFrame;
}
