/*
 * File:        writer_f2section.h
 * Module:      writers
 * Purpose:     Sequential writer for F2 section files
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2025 Simon Inns
 */

#ifndef WRITER_F2SECTION_H
#define WRITER_F2SECTION_H

#include <QDataStream>
#include <QDebug>
#include <QFile>
#include <QString>

#include "section.h"

class WriterF2Section {
 public:
  WriterF2Section();
  ~WriterF2Section();

  bool open(const QString& filename);
  void write(const F2Section& f2Section);
  void close();
  qint64 size() const;
  bool isOpen() const { return m_file.isOpen(); };

 private:
  QFile m_file;
  QDataStream* m_dataStream;
};

#endif  // WRITER_F2SECTION_H
