/*
 * File:        f2_stacker.h
 * Module:      stacker
 * Purpose:     Multi-source F2 section stacking with per-byte majority voting
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2025 Simon Inns
 */

#ifndef F2_STACKER_H
#define F2_STACKER_H

#include <QDebug>
#include <QFile>
#include <QString>
#include <QVector>

#include "reader_f2section.h"
#include "writer_f2section.h"

class F2Stacker {
 public:
  F2Stacker();

  bool process(const QVector<QString>& inputFilenames,
               const QString& outputFilename);

 private:
  struct SectionAnalysis {
    quint32 score{0};
    bool isPadding{false};
    bool isFlat{false};
  };

  struct SourceState {
    ReaderF2Section* reader{nullptr};
    F2Section currentSection;
    SectionAnalysis currentAnalysis;
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
  QVector<int> m_previousClusterSourceIndexes;

  bool readNextValidSection(SourceState& sourceState,
                            qint32 minimumAddress = -1);
  SectionAnalysis analyzeSection(const F2Section& section) const;
  quint32 sectionDifference(const F2Section& firstSection,
                            const F2Section& secondSection) const;
  F2Section stackSections(const QVector<F2Section>& sections,
                          const QVector<SectionAnalysis>& sectionAnalyses,
                          const QVector<int>& sourceIndexes);
  F2Frame stackFrames(QVector<F2Frame>& f2Frames,
                      const QVector<int>& sourceIndexes);

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

#endif  // F2_STACKER_H
