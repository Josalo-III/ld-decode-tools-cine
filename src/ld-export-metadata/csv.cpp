/*
 * File:        csv.cpp
 * Module:      export
 * Purpose:     VITS and VBI metadata export to CSV
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2018-2019 Simon Inns
 */

#include "csv.h"

#include <QDebug>
#include <QFile>
#include <QTextStream>
#include <QtGlobal>

#include "tbc/logging.h"
#include "vbidecoder.h"

// Create an 'escaped string' for safe CSV output of QStrings
static QString escapedString(const QString &unescapedString) {
  if (!unescapedString.contains(QLatin1Char(','))) return unescapedString;
  QString escapedString = unescapedString;
  return '\"' +
         escapedString.replace(QLatin1Char('\"'), QStringLiteral("\"\"")) +
         '\"';
}

bool writeVitsCsv(LdDecodeMetaData &metaData, const QString &fileName) {
  // Open a file for the CSV output
  QFile csvFile(fileName);
  if (!csvFile.open(QFile::WriteOnly | QFile::Text)) {
    tbcDebug(
        QStringLiteral("LdDecodeMetaData::writeVitsCsv(): Could not open CSV "
                       "file for output!"));
    return false;
  }

  // Create a text stream for the CSV output
  QTextStream outStream(&csvFile);
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
  outStream.setCodec("UTF-8");
#endif

  // Write the field and VITS data
  outStream << "seqNo,isFirstField,syncConf,";
  outStream << "medianBurstIRE,fieldPhaseID,audioSamples,";

  // VITS headers
  outStream << "wSNR,bPSNR";
  outStream << '\n';

  for (qint32 fieldNumber = 1; fieldNumber <= metaData.getNumberOfFields();
       fieldNumber++) {
    const LdDecodeMetaData::Field &field = metaData.getField(fieldNumber);
    outStream << escapedString(QString::number(field.seqNo)) << ",";
    outStream << escapedString(QString::number(field.isFirstField)) << ",";
    outStream << escapedString(QString::number(field.syncConf)) << ",";
    outStream << escapedString(QString::number(field.medianBurstIRE)) << ",";
    outStream << escapedString(QString::number(field.fieldPhaseID)) << ",";
    outStream << escapedString(QString::number(field.audioSamples)) << ",";

    outStream << escapedString(QString::number(field.vitsMetrics.wSNR)) << ",";
    outStream << escapedString(QString::number(field.vitsMetrics.bPSNR)) << ",";

    outStream << '\n';
  }

  // Close the CSV file
  csvFile.close();

  return true;
}

bool writeVbiCsv(LdDecodeMetaData &metaData, const QString &fileName) {
  // Open a file for the CSV output
  QFile csvFile(fileName);
  if (!csvFile.open(QFile::WriteOnly | QFile::Text)) {
    tbcDebug(
        QStringLiteral("LdDecodeMetaData::writeVbiCsv(): Could not open CSV "
                       "file for output!"));
    return false;
  }

  // Create a text stream for the CSV output
  QTextStream outStream(&csvFile);
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
  outStream.setCodec("UTF-8");
#endif

  // Write the field and VBI data
  outStream << "frameNo,";
  outStream << "discType,pictureNumber,clvTimeCode,chapter,";
  outStream << "leadIn,leadOut,userCode,stopCode";
  outStream << '\n';

  for (qint32 frameNumber = 1; frameNumber <= metaData.getNumberOfFrames();
       frameNumber++) {
    // Get the required field numbers
    qint32 firstFieldNumber = metaData.getFirstFieldNumber(frameNumber);
    qint32 secondFieldNumber = metaData.getSecondFieldNumber(frameNumber);

    // Get the field metadata
    const LdDecodeMetaData::Field &firstField =
        metaData.getField(firstFieldNumber);
    const LdDecodeMetaData::Field &secondField =
        metaData.getField(secondFieldNumber);

    qint32 vbi16_1, vbi17_1, vbi18_1;
    qint32 vbi16_2, vbi17_2, vbi18_2;

    vbi16_1 = firstField.vbi.vbiData[0];
    vbi17_1 = firstField.vbi.vbiData[1];
    vbi18_1 = firstField.vbi.vbiData[2];
    vbi16_2 = secondField.vbi.vbiData[0];
    vbi17_2 = secondField.vbi.vbiData[1];
    vbi18_2 = secondField.vbi.vbiData[2];

    VbiDecoder vbiDecoder;
    VbiDecoder::Vbi vbi = vbiDecoder.decodeFrame(vbi16_1, vbi17_1, vbi18_1,
                                                 vbi16_2, vbi17_2, vbi18_2);

    outStream << escapedString(QString::number(frameNumber)) << ",";

    if (vbi.type != VbiDecoder::VbiDiscTypes::unknownDiscType) {
      if (vbi.type == VbiDecoder::VbiDiscTypes::cav) outStream << "CAV,";
      if (vbi.type == VbiDecoder::VbiDiscTypes::clv) outStream << "CLV,";
    } else {
      if (vbi.type == VbiDecoder::VbiDiscTypes::cav) outStream << "CAV,";
      if (vbi.type == VbiDecoder::VbiDiscTypes::clv) outStream << "CLV,";
      if (vbi.type == VbiDecoder::VbiDiscTypes::unknownDiscType)
        outStream << "unknown,";
    }

    if (vbi.picNo != -1)
      outStream << escapedString(QString::number(vbi.picNo)) << ",";
    else
      outStream << "none,";

    QString clvTimecodeString;
    if (vbi.clvHr != -1 || vbi.clvMin != -1 || vbi.clvSec != -1 ||
        vbi.clvPicNo != -1) {
      if (vbi.clvHr != -1 && vbi.clvMin != -1) {
        clvTimecodeString =
            QString("%1").arg(vbi.clvHr, 2, 10, QChar('0')) + ":" +
            QString("%1").arg(vbi.clvMin, 2, 10, QChar('0')) + ":";
      } else
        clvTimecodeString = "xx:xx:";

      if (vbi.clvSec != -1 && vbi.clvPicNo != -1) {
        clvTimecodeString += QString("%1").arg(vbi.clvSec, 2, 10, QChar('0')) +
                             "." +
                             QString("%1").arg(vbi.clvPicNo, 2, 10, QChar('0'));
      } else
        clvTimecodeString += "xx.xx";
    } else if (vbi.clvHr != -1 || vbi.clvMin != -1) {
      if (vbi.clvHr != -1 && vbi.clvMin != -1) {
        clvTimecodeString = QString("%1").arg(vbi.clvHr, 2, 10, QChar('0')) +
                            ":" +
                            QString("%1").arg(vbi.clvMin, 2, 10, QChar('0'));
      } else
        clvTimecodeString = "xx:xx";
    } else
      clvTimecodeString = "none";
    outStream << escapedString(clvTimecodeString) << ",";

    if (vbi.chNo != -1)
      outStream << escapedString(QString::number(vbi.chNo)) << ",";
    else
      outStream << "none,";

    if (vbi.leadIn)
      outStream << "true,";
    else
      outStream << "false,";

    if (vbi.leadOut)
      outStream << "true,";
    else
      outStream << "false,";

    if (vbi.userCode.isEmpty())
      outStream << "none,";
    else {
      if (vbi.userCode.isEmpty())
        outStream << "0,";
      else
        outStream << escapedString(vbi.userCode) << ",";
    }

    if (vbi.picStop)
      outStream << "true,";
    else
      outStream << "false,";

    outStream << '\n';
  }

  // Close the CSV file
  csvFile.close();

  return true;
}
