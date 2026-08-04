/*
 * File:        writer_f2section.cpp
 * Module:      writers
 * Purpose:     Sequential writer for F2 section files
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2025 Simon Inns
 */

#include "writer_f2section.h"

#include "tbc/logging.h"

WriterF2Section::WriterF2Section() {}

WriterF2Section::~WriterF2Section() {
  if (m_file.isOpen()) {
    m_file.close();
  }
}

bool WriterF2Section::open(const QString& filename) {
  m_file.setFileName(filename);
  if (!m_file.open(QIODevice::WriteOnly)) {
    qCritical() << "WriterData::open() - Could not open file" << filename
                << "for writing";
    return false;
  }

  // Create a data stream for writing
  m_dataStream = new QDataStream(&m_file);
  tbcDebugStream() << "WriterData::open() - Opened file" << filename
                   << "for data writing";
  return true;
}

void WriterF2Section::write(const F2Section& f2Section) {
  if (!m_file.isOpen()) {
    qCritical() << "WriterF2Section::write() - File is not open for writing";
    return;
  }

  *m_dataStream << f2Section;
}

void WriterF2Section::close() {
  if (!m_file.isOpen()) {
    return;
  }

  // Close the data stream
  delete m_dataStream;
  m_dataStream = nullptr;

  m_file.close();
  tbcDebugStream() << "WriterF2Section::close(): Closed the data file"
                   << m_file.fileName();
}

qint64 WriterF2Section::size() const {
  if (m_file.isOpen()) {
    return m_file.size();
  }

  return 0;
}
