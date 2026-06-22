/************************************************************************

    f2_stacker.h

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

#ifndef F2_STACKER_H
#define F2_STACKER_H

#include <QString>
#include <QVector>
#include <QDebug>
#include <QFile>

#include "reader_f2section.h"
#include "writer_f2section.h"

class F2Stacker
{
public:
    F2Stacker();

    bool process(const QVector<QString> &inputFilenames, const QString &outputFilename);

private:
    struct SourceState
    {
        ReaderF2Section *reader{nullptr};
        F2Section currentSection;
        qint64 nextSectionNumber{0};
        qint32 lastReadAddress{-1};
        bool hasCurrentSection{false};
        bool endOfFile{false};
        quint64 invalidMetadataSections{0};
        quint64 duplicateSections{0};
        quint64 replacedDuplicateSections{0};
        quint64 outOfOrderSections{0};
    };

    QVector<ReaderF2Section*> m_inputFiles;
    WriterF2Section m_outputFile;

    bool readNextValidSection(SourceState &sourceState, qint32 minimumAddress = -1);
    quint32 sectionScore(const F2Section &section) const;
    quint32 sectionDifference(const F2Section &firstSection, const F2Section &secondSection) const;
    bool isPaddingSection(const F2Section &section) const;
    bool isFlatSection(const F2Section &section) const;
    F2Section stackSections(const QVector<F2Section> &sections, const QVector<int> &sourceIndexes);
    F2Frame stackFrames(QVector<F2Frame> &f2Frames, const QVector<int> &sourceIndexes);

    // Statistics
    quint64 m_noValidValueForByte;

    quint64 m_validValueForByte;
    quint64 m_usedMostCommonValue;
    quint64 m_tiedValueForByte;

    quint64 m_errorFreeFrames;
    quint64 m_errorFrames;
    quint64 m_paddedFrames;
    quint64 m_flatSections;
    quint64 m_divergentSections;

    QVector<quint64> m_sourceDifferences;
    QVector<quint64> m_sourceMissingSections;
};

#endif // F2_STACKER_H
