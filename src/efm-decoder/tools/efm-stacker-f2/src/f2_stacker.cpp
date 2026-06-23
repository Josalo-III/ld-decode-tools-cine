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

#include <array>
#include <limits>

namespace {
constexpr int kMaxStackSources = 32;
}

F2Stacker::F2Stacker() :
    m_noValidValueForByte(0),
    m_validValueForByte(0),
    m_usedMostCommonValue(0),
    m_tiedValueForByte(0),
    m_errorFreeFrames(0),
    m_errorFrames(0),
    m_paddedFrames(0),
    m_flatSections(0),
    m_divergentSections(0)
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
        QVector<SectionAnalysis> sectionAnalyses;
        QVector<int> sectionSourceIndexes;
        sectionList.reserve(sourceStates.size());
        sectionAnalyses.reserve(sourceStates.size());
        sectionSourceIndexes.reserve(sourceStates.size());

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
            SectionAnalysis bestAnalysis = sourceState.currentAnalysis;
            while (readNextValidSection(sourceState, address)) {
                qint32 nextAddress =
                        sourceState.currentSection.metadata.absoluteSectionTime().frames();
                if (nextAddress != address) {
                    break;
                }

                sourceState.duplicateSections++;
                if (sourceState.currentAnalysis.score < bestAnalysis.score) {
                    bestSection = sourceState.currentSection;
                    bestAnalysis = sourceState.currentAnalysis;
                    sourceState.replacedDuplicateSections++;
                }
            }

            sectionList.append(bestSection);
            sectionAnalyses.append(bestAnalysis);
            sectionSourceIndexes.append(sourceIndex);
        }

        if (sectionList.isEmpty()) {
            qCritical().noquote() << "F2Stacker::process() - No sources available for section"
                                  << SectionTime(address).toString();
            return false;
        }

        tbcDebugStream().noquote() << "F2Stacker::process() - Stacking section" << SectionTime(address).toString();

        F2Section stackedF2Section = stackSections(sectionList, sectionAnalyses, sectionSourceIndexes);

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
    qInfo().noquote() << "  Divergent source sections excluded:" << m_divergentSections;
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
        sourceState.currentAnalysis = analyzeSection(sourceState.currentSection);
        sourceState.hasCurrentSection = true;
        return true;
    }

    sourceState.hasCurrentSection = false;
    sourceState.endOfFile = true;
    return false;
}

F2Stacker::SectionAnalysis F2Stacker::analyzeSection(const F2Section &section) const
{
    SectionAnalysis analysis;
    analysis.isPadding = true;
    QHash<quint8, int> byteCounts;
    QSet<QByteArray> uniqueFrames;
    int totalBytes = 0;
    int dominantByteCount = 0;

    if (section.metadata.isRepaired()) {
        analysis.score += 100000;
    }

    for (int frameIndex = 0; frameIndex < 98; frameIndex++) {
        const F2Frame &frame = section.frameRef(frameIndex);
        const QVector<bool> &errorData = frame.rawErrorData();
        const QVector<bool> &paddedData = frame.rawPaddedData();
        const QVector<quint8> &data = frame.rawData();
        QByteArray frameBytes;
        frameBytes.reserve(data.size());

        analysis.score += errorData.count(true);
        analysis.score += paddedData.count(true);

        if (paddedData.contains(false)) {
            analysis.isPadding = false;
        }

        for (int byteIndex = 0; byteIndex < data.size(); byteIndex++) {
            byteCounts[data[byteIndex]]++;
            frameBytes.append(static_cast<char>(data[byteIndex]));
            totalBytes++;
        }

        uniqueFrames.insert(frameBytes);
    }

    for (auto it = byteCounts.begin(); it != byteCounts.end(); ++it) {
        if (it.value() > dominantByteCount) {
            dominantByteCount = it.value();
        }
    }

    const int repeatedFrames = 98 - uniqueFrames.size();
    analysis.isFlat = totalBytes == 0 ||
                      byteCounts.size() <= 8 ||
                      repeatedFrames >= 80 ||
                      dominantByteCount > (totalBytes * 85 / 100);

    if (analysis.isFlat) {
        analysis.score += 100000;
    }

    return analysis;
}

quint32 F2Stacker::sectionDifference(const F2Section &firstSection, const F2Section &secondSection) const
{
    quint32 differences = 0;

    for (int frameIndex = 0; frameIndex < 98; frameIndex++) {
        const F2Frame &firstFrame = firstSection.frameRef(frameIndex);
        const F2Frame &secondFrame = secondSection.frameRef(frameIndex);
        const QVector<quint8> &firstData = firstFrame.rawData();
        const QVector<quint8> &secondData = secondFrame.rawData();

        for (int byteIndex = 0; byteIndex < 32; byteIndex++) {
            if (firstData[byteIndex] != secondData[byteIndex]) {
                differences++;
            }
        }
    }

    return differences;
}

F2Section F2Stacker::stackSections(const QVector<F2Section> &f2Sections,
                                   const QVector<SectionAnalysis> &sectionAnalyses,
                                   const QVector<int> &sourceIndexes)
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
    QVector<SectionAnalysis> validSectionAnalyses;
    QVector<int> validSourceIndexes;
    for (int sectionIndex = 0; sectionIndex < f2Sections.size(); sectionIndex++) {
        if (sectionAnalyses[sectionIndex].isPadding) {
            tbcDebugStream().noquote()
                << "F2Stacker::stackSections - Section" << sectionIndex
                << "is just padding";
        } else if (sectionAnalyses[sectionIndex].isFlat) {
            tbcDebugStream().noquote()
                << "F2Stacker::stackSections - Section" << sectionIndex
                << "is flat data";
            m_flatSections++;
        } else {
            validF2Sections.append(f2Sections[sectionIndex]);
            validSectionAnalyses.append(sectionAnalyses[sectionIndex]);
            validSourceIndexes.append(sourceIndexes[sectionIndex]);
        }
    }

    if (validF2Sections.isEmpty()) {
        // All candidate sections are padding. Preserve the first candidate section.
        stackedSection = f2Sections[0];
        m_paddedFrames += 98;
        m_previousClusterSourceIndexes.clear();
    } else if (validF2Sections.size() == 1) {
        // Exactly one source has real data for this section. Pass it through.
            stackedSection = validF2Sections[0];
            m_previousClusterSourceIndexes.clear();
            m_previousClusterSourceIndexes.append(validSourceIndexes[0]);

            for (int frameIndex = 0; frameIndex < 98; frameIndex++) {
            if (stackedSection.frameRef(frameIndex).rawErrorData().contains(1)) {
                m_errorFrames++;
            } else {
                m_errorFreeFrames++;
            }
        }
    } else {
        QVector<int> bestClusterIndexes;
        quint64 bestClusterScore = std::numeric_limits<quint64>::max();
        quint64 bestClusterDifferences = std::numeric_limits<quint64>::max();
        int bestClusterPreviousOverlap = -1;
        const quint32 sameSectionDifferenceThreshold = 128;
        const int validSectionCount = validF2Sections.size();
        QVector<quint32> sectionDifferences;
        sectionDifferences.resize(validSectionCount * validSectionCount);

        for (int firstIndex = 0; firstIndex < validSectionCount; firstIndex++) {
            sectionDifferences[(firstIndex * validSectionCount) + firstIndex] = 0;
            for (int secondIndex = firstIndex + 1; secondIndex < validSectionCount; secondIndex++) {
                const quint32 differences =
                    sectionDifference(validF2Sections[firstIndex], validF2Sections[secondIndex]);
                sectionDifferences[(firstIndex * validSectionCount) + secondIndex] = differences;
                sectionDifferences[(secondIndex * validSectionCount) + firstIndex] = differences;
            }
        }

        for (int seedIndex = 0; seedIndex < validSectionCount; seedIndex++) {
            QVector<int> clusterIndexes;
            clusterIndexes.reserve(validSectionCount);
            quint64 clusterScore = 0;
            quint64 clusterDifferences = 0;
            int clusterPreviousOverlap = 0;

            for (int candidateIndex = 0; candidateIndex < validSectionCount; candidateIndex++) {
                const quint32 differences =
                    sectionDifferences[(seedIndex * validSectionCount) + candidateIndex];
                if (differences <= sameSectionDifferenceThreshold) {
                    clusterIndexes.append(candidateIndex);
                    clusterScore += validSectionAnalyses[candidateIndex].score;
                    clusterDifferences += differences;
                    if (m_previousClusterSourceIndexes.contains(validSourceIndexes[candidateIndex])) {
                        clusterPreviousOverlap++;
                    }
                }
            }

            if (clusterIndexes.size() > bestClusterIndexes.size() ||
                (clusterIndexes.size() == bestClusterIndexes.size() &&
                 clusterPreviousOverlap > bestClusterPreviousOverlap) ||
                (clusterIndexes.size() == bestClusterIndexes.size() &&
                 clusterPreviousOverlap == bestClusterPreviousOverlap &&
                 clusterScore < bestClusterScore) ||
                (clusterIndexes.size() == bestClusterIndexes.size() &&
                 clusterPreviousOverlap == bestClusterPreviousOverlap &&
                 clusterScore == bestClusterScore &&
                 clusterDifferences < bestClusterDifferences)) {
                bestClusterIndexes = clusterIndexes;
                bestClusterScore = clusterScore;
                bestClusterDifferences = clusterDifferences;
                bestClusterPreviousOverlap = clusterPreviousOverlap;
            }
        }

        if (!bestClusterIndexes.isEmpty() && bestClusterIndexes.size() < validF2Sections.size()) {
            QVector<F2Section> clusteredF2Sections;
            QVector<SectionAnalysis> clusteredSectionAnalyses;
            QVector<int> clusteredSourceIndexes;

            for (int clusterIndex = 0; clusterIndex < bestClusterIndexes.size(); clusterIndex++) {
                const int sectionIndex = bestClusterIndexes[clusterIndex];
                clusteredF2Sections.append(validF2Sections[sectionIndex]);
                clusteredSectionAnalyses.append(validSectionAnalyses[sectionIndex]);
                clusteredSourceIndexes.append(validSourceIndexes[sectionIndex]);
            }

            m_divergentSections += validF2Sections.size() - clusteredF2Sections.size();
            validF2Sections = clusteredF2Sections;
            validSectionAnalyses = clusteredSectionAnalyses;
            validSourceIndexes = clusteredSourceIndexes;
        }
        m_previousClusterSourceIndexes = validSourceIndexes;

        if (validF2Sections.size() == 1) {
            stackedSection = validF2Sections[0];

            for (int frameIndex = 0; frameIndex < 98; frameIndex++) {
                if (stackedSection.frameRef(frameIndex).rawErrorData().contains(1)) {
                    m_errorFrames++;
                } else {
                    m_errorFreeFrames++;
                }
            }

            stackedSection.metadata = stackedMetadata;
            return stackedSection;
        }

        // Each section contains 98 F2Frames.
        for (int frameIndex = 0; frameIndex < 98; frameIndex++) {
            QVector<F2Frame> frameList;
            frameList.reserve(validF2Sections.size());

            for (int sectionIndex = 0; sectionIndex < validF2Sections.size(); sectionIndex++) {
                frameList.append(validF2Sections[sectionIndex].frameRef(frameIndex));
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
    const int frameCount = f2Frames.size();
    if (frameCount > kMaxStackSources) {
        qFatal("F2Stacker::stackFrames - Too many input frames (%d > %d)",
               frameCount,
               kMaxStackSources);
    }

    // Process one byte at a time
    QVector<quint8> stackedFrameData;
    QVector<bool> stackedFrameErrorData;
    stackedFrameData.reserve(32);
    stackedFrameErrorData.reserve(32);
    for (int byteIndex = 0; byteIndex < 32; byteIndex++) {
        std::array<quint8, kMaxStackSources> validBytes{};
        int validByteCount = 0;
        bool allBytesSame = true;
        quint8 firstValidByte = 0;
        std::array<int, 256> byteCounts{};

        for (int listIndex = 0; listIndex < f2Frames.size(); ++listIndex) {
            const F2Frame &frame = f2Frames.at(listIndex);
            const QVector<bool> &errorData = frame.rawErrorData();
            if (errorData[byteIndex] == false) {
                const quint8 byteValue = frame.rawData()[byteIndex];
                if (validByteCount == 0) {
                    firstValidByte = byteValue;
                } else if (byteValue != firstValidByte) {
                    allBytesSame = false;
                }

                validBytes[validByteCount] = byteValue;
                validByteCount++;
                byteCounts[byteValue]++;
            }
        }

        if (validByteCount == 0) {
            // All bytes are errors - can't correct
            stackedFrameData.append(f2Frames.at(0).rawData()[byteIndex]);
            stackedFrameErrorData.append(true);
            tbcDebugStream() << "F2Stacker::stackFrames - No valid byte value for index" << byteIndex;
            m_noValidValueForByte++;
        } else {
            // If all valid bytes are the same, use that value.
            if (allBytesSame) {
                stackedFrameData.append(firstValidByte);
                stackedFrameErrorData.append(false);
                m_validValueForByte++;
                continue;
            } else {
                // Find the most common byte value.  If there is a tie, keep the
                // byte marked as an erasure instead of blessing an arbitrary source.
                int maxCount = 0;
                int tiedMaxCount = 0;
                quint8 mostCommonByte = 0;
                for (int value = 0; value < 256; value++) {
                    if (byteCounts[value] > maxCount) {
                        maxCount = byteCounts[value];
                        mostCommonByte = static_cast<quint8>(value);
                        tiedMaxCount = 1;
                    } else if (byteCounts[value] == maxCount && maxCount > 0) {
                        tiedMaxCount++;
                    }
                }

                QString validBytesString;
                for (int validIndex = 0; validIndex < validByteCount; validIndex++) {
                    validBytesString.append(QString("%1 ").arg(validBytes[validIndex], 2, 16, QChar('0')).toUpper());
                }
                QString mostCommonByteString = QString("%1").arg(mostCommonByte, 2, 16, QChar('0')).toUpper();

                if (tiedMaxCount > 1) {
                    tbcDebugStream().noquote()
                        << "F2Stacker::stackFrames - Valid byte values tie - marking byte as error from"
                        << validBytesString;
                    stackedFrameData.append(firstValidByte);
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
        quint8 expectedValue = f2Frames.at(0).rawData()[byteIndex];
        for (int sourceIndex = 0; sourceIndex < frameCount; sourceIndex++) {
            if (f2Frames.at(sourceIndex).rawData()[byteIndex] != expectedValue) {
                m_sourceDifferences[sourceIndexes[sourceIndex]]++;
            }
        }
    }

    // Set the data for the stacked frame
    stackedFrame.setData(stackedFrameData);
    stackedFrame.setErrorData(stackedFrameErrorData);

    return stackedFrame;
}
