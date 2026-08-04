/*
 * File:        reader_f2section.h
 * Module:      readers
 * Purpose:     Sequential and random-access reader for F2 section files
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2025 Simon Inns
 */

#ifndef READER_F2SECTION_H
#define READER_F2SECTION_H

#include <QDataStream>
#include <QDebug>
#include <QFile>
#include <QString>

#include "section.h"

class ReaderF2Section {
 public:
  ReaderF2Section();
  ~ReaderF2Section();

  bool open(const QString& filename);
  F2Section read();
  void close();
  qint64 size();

  void seekToSection(qint64 sectionNumber);

 private:
  QFile m_file;
  QDataStream* m_dataStream;
  qint64 m_fileSizeInSections;
  qint64 m_sectionSize;
};

#endif  // READER_F2SECTION_H
