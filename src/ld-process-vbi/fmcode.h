/*
 * File:        fmcode.h
 * Module:      vbi
 * Purpose:     40-bit FM coded signal decoder
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2018-2019 Simon Inns
 */

#ifndef FMCODE_H
#define FMCODE_H

#include "lddecodemetadata.h"
#include "sourcevideo.h"

// Decoder for NTSC LaserDisc FM code lines.
// Specified in IEC 60587-1986 section 10.2.
class FmCode {
 public:
  bool decodeLine(const SourceVideo::Data& lineData,
                  const LdDecodeMetaData::VideoParameters& videoParameters,
                  LdDecodeMetaData::Field& fieldMetadata);
};

#endif  // FMCODE_H
