/*
 * File:        audacity.cpp
 * Module:      export
 * Purpose:     Chapter changes as an Audacity label track
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2023 Adam Sampson
 */

#include "audacity.h"

#include <QFile>
#include <QTextStream>
#include <QtGlobal>

#include "navigation.h"
#include "tbc/logging.h"

bool writeAudacityLabels(LdDecodeMetaData &metaData, const QString &fileName) {
  const auto videoParameters = metaData.getVideoParameters();

  // Positions are given in seconds, with exclusive ranges.
  // Select a scale factor to convert from 0-based field numbers to seconds.
  const double timeFactor =
      videoParameters.system == PAL ? (1.0 / 50.0) : (1001.0 / 60000.0);

  // Extract navigation information
  const NavigationInfo navInfo(metaData);

  // Open the output file
  QFile file(fileName);
  if (!file.open(QFile::WriteOnly | QFile::Text)) {
    tbcDebug(
        QStringLiteral("writeAudacityLabels: Could not open file for output"));
    return false;
  }
  QTextStream stream(&file);
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
  stream.setCodec("UTF-8");
#endif

  // Write the chapter changes
  for (const auto &chapter : navInfo.chapters) {
    stream << QString("%1\t%2\tChapter %3\n")
                  .arg(static_cast<double>(chapter.startField) * timeFactor, 0,
                       'f')
                  .arg(static_cast<double>(chapter.endField) * timeFactor, 0,
                       'f')
                  .arg(chapter.number);
  }

  // Write the stop codes
  for (qint32 field : navInfo.stopCodes) {
    stream << QString("%1\t%2\tStop code\n")
                  .arg(static_cast<double>(field) * timeFactor, 0, 'f')
                  .arg(static_cast<double>(field) * timeFactor, 0, 'f');
  }

  // Done!
  file.close();
  return true;
}
