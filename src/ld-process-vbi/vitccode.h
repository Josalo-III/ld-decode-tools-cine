/*
 * File:        vitccode.h
 * Module:      vbi
 * Purpose:     VITC timecode decoder
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2022 Adam Sampson
 */

#ifndef VITCCODE_H
#define VITCCODE_H

#include <vector>

#include "lddecodemetadata.h"
#include "sourcevideo.h"

class VitcCode {
 public:
  bool decodeLine(const SourceVideo::Data& lineData,
                  const LdDecodeMetaData::VideoParameters& videoParameters,
                  LdDecodeMetaData::Field& fieldMetadata);

  std::vector<qint32> getLineNumbers(
      const LdDecodeMetaData::VideoParameters& videoParameters);
};

#endif  // VITCCODE_H
