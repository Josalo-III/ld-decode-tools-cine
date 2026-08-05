/*
 * File:        whiteflag.cpp
 * Module:      vbi
 * Purpose:     White flag detection
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2018-2021 Simon Inns
 */

#include "whiteflag.h"

#include "tbc/logging.h"

// Public method to read the white flag status from a field-line.
// Return true if the flag is detected, false otherwise.
bool WhiteFlag::decodeLine(
    const SourceVideo::Data& lineData,
    const LdDecodeMetaData::VideoParameters& videoParameters,
    LdDecodeMetaData::Field& fieldMetadata) {
  // Determine the 16-bit zero-crossing point
  qint32 zcPoint =
      (videoParameters.white16bIre + videoParameters.black16bIre) / 2;

  qint32 whiteCount = 0;
  for (qint32 x = videoParameters.activeVideoStart;
       x < videoParameters.activeVideoEnd; x++) {
    if (lineData[x] > zcPoint) whiteCount++;
  }

  // Mark the line as a white flag if at least 50% of the data is above the zc
  // point
  if (whiteCount >
      ((videoParameters.activeVideoEnd - videoParameters.activeVideoStart) /
       2)) {
    tbcDebugStream()
        << "WhiteFlag::getWhiteFlag(): White-flag detected: White count was"
        << whiteCount << "out of"
        << (videoParameters.activeVideoEnd - videoParameters.activeVideoStart);
    fieldMetadata.ntsc.whiteFlag = true;
    return true;
  }

  // No white flag detected
  fieldMetadata.ntsc.whiteFlag = false;
  return false;
}
