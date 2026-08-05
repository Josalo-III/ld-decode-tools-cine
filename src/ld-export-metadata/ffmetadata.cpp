/*
 * File:        ffmetadata.cpp
 * Module:      export
 * Purpose:     Chapter markers as an FFmpeg ffmetadata file
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2019-2023 Adam Sampson
 */

#include "ffmetadata.h"

#include <QFile>
#include <QTextStream>
#include <QtGlobal>

#include "navigation.h"
#include "tbc/logging.h"

bool writeFfmetadata(LdDecodeMetaData &metaData, const QString &fileName) {
  const auto videoParameters = metaData.getVideoParameters();

  // Select the appropriate timebase to make 0-based field numbers work
  const QString timeBase =
      videoParameters.system == PAL ? "1/50" : "1001/60000";

  // Extract navigation information
  const NavigationInfo navInfo(metaData);

  // Open the output file
  QFile file(fileName);
  if (!file.open(QFile::WriteOnly | QFile::Text)) {
    tbcDebug(QStringLiteral("writeFfmetadata: Could not open file for output"));
    return false;
  }
  QTextStream stream(&file);
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
  stream.setCodec("UTF-8");
#endif

  // Write the header
  stream << ";FFMETADATA1\n";

  // Write the chapter changes
  for (const auto &chapter : navInfo.chapters) {
    stream << "\n";
    stream << "[CHAPTER]\n";
    stream << "TIMEBASE=" << timeBase << "\n";
    stream << "START=" << chapter.startField << "\n";
    stream << "END=" << chapter.endField - 1 << "\n";
    stream << "title=" << QString("Chapter %1").arg(chapter.number) << "\n";
  }

  if (!navInfo.stopCodes.empty()) {
    // Write the stop codes, as comments
    // XXX Is there a way to represent these properly?
    stream << "\n";
    for (qint32 field : navInfo.stopCodes) {
      stream << "; Stop code at " << field << "\n";
    }
  }

  // Done!
  file.close();
  return true;
}
