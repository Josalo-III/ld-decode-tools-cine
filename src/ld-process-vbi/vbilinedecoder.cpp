/*
 * File:        vbilinedecoder.cpp
 * Module:      vbi
 * Purpose:     Per-field VBI line decode worker
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2018-2019 Simon Inns
 */

#include "vbilinedecoder.h"

#include "biphasecode.h"
#include "closedcaption.h"
#include "decoderpool.h"
#include "fmcode.h"
#include "tbc/logging.h"
#include "videoid.h"
#include "vitccode.h"
#include "whiteflag.h"

VbiLineDecoder::VbiLineDecoder(QAtomicInt& _abort, DecoderPool& _decoderPool,
                               QObject* parent)
    : QThread(parent), abort(_abort), decoderPool(_decoderPool) {}

// Thread main processing method
void VbiLineDecoder::run() {
  qint32 fieldNumber;

  // Input data buffers
  SourceVideo::Data sourceFieldData;
  LdDecodeMetaData::Field fieldMetadata;
  LdDecodeMetaData::VideoParameters videoParameters;

  while (!abort) {
    // Get the next field to process from the input file
    if (!decoderPool.getInputField(fieldNumber, sourceFieldData, fieldMetadata,
                                   videoParameters)) {
      // No more input fields -- exit
      break;
    }

    // Show progress (for every 1000th field)
    if (fieldNumber % 1000 == 0) {
      qInfo() << "Processing field" << fieldNumber;
    }

    if (fieldMetadata.isFirstField)
      tbcDebugStream()
          << "VbiLineDecoder::process(): Getting metadata for field"
          << fieldNumber << "(first)";
    else
      tbcDebugStream()
          << "VbiLineDecoder::process(): Getting metadata for field"
          << fieldNumber << "(second)";

    // Get the 24-bit biphase-coded data from field lines 16-18
    BiphaseCode biphaseCode;
    biphaseCode.decodeLines(getFieldLine(sourceFieldData, 16, videoParameters),
                            getFieldLine(sourceFieldData, 17, videoParameters),
                            getFieldLine(sourceFieldData, 18, videoParameters),
                            videoParameters, fieldMetadata);

    // Process NTSC specific data if source type is NTSC
    if (videoParameters.system == NTSC) {
      // Get the 40-bit FM coded data from field line 10
      FmCode fmCode;
      fmCode.decodeLine(getFieldLine(sourceFieldData, 10, videoParameters),
                        videoParameters, fieldMetadata);

      // Get the white flag from field line 11
      WhiteFlag whiteFlag;
      whiteFlag.decodeLine(getFieldLine(sourceFieldData, 11, videoParameters),
                           videoParameters, fieldMetadata);

      // Get IEC 61880 data from field line 20
      VideoID videoID;
      videoID.decodeLine(getFieldLine(sourceFieldData, 20, videoParameters),
                         videoParameters, fieldMetadata);

      fieldMetadata.ntsc.inUse = true;
    }

    // Get VITC data, trying each possible line and stopping when we find a
    // valid one
    VitcCode vitcCode;
    for (qint32 lineNumber : vitcCode.getLineNumbers(videoParameters)) {
      if (vitcCode.decodeLine(
              getFieldLine(sourceFieldData, lineNumber, videoParameters),
              videoParameters, fieldMetadata)) {
        break;
      }
    }

    // Get Closed Caption data from line 21 (525-line) or 22 (625-line)
    ClosedCaption closedCaption;
    closedCaption.decodeLine(
        getFieldLine(sourceFieldData, (videoParameters.system == PAL) ? 22 : 21,
                     videoParameters),
        videoParameters, fieldMetadata);

    // Write the result to the output metadata
    if (!decoderPool.setOutputField(fieldNumber, fieldMetadata)) {
      abort = true;
      break;
    }
  }
}

// Private method to get a single scanline of greyscale data
SourceVideo::Data VbiLineDecoder::getFieldLine(
    const SourceVideo::Data& sourceField, qint32 fieldLine,
    const LdDecodeMetaData::VideoParameters& videoParameters) {
  // Range-check the field line
  if (fieldLine < startFieldLine || fieldLine > endFieldLine) {
    qWarning() << "Cannot generate field-line data, line number is out of "
                  "bounds! Scan line ="
               << fieldLine;
    return SourceVideo::Data();
  }

  qint32 startPointer =
      (fieldLine - startFieldLine) * videoParameters.fieldWidth;
  return sourceField.mid(startPointer, videoParameters.fieldWidth);
}
