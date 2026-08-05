/*
 * File:        whiteflag.h
 * Module:      vbi
 * Purpose:     White flag detection
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2018-2021 Simon Inns
 */

#ifndef WHITEFLAG_H
#define WHITEFLAG_H

#include "lddecodemetadata.h"
#include "sourcevideo.h"

// Decoder for NTSC LaserDisc white flag lines.
// Specified in IEC 60587-1986 section 10.2.4.
class WhiteFlag {
 public:
  bool decodeLine(const SourceVideo::Data& lineData,
                  const LdDecodeMetaData::VideoParameters& videoParameters,
                  LdDecodeMetaData::Field& fieldMetadata);
};

#endif  // WHITEFLAG_H
