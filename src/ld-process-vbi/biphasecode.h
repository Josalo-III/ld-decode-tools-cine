/*
 * File:        biphasecode.h
 * Module:      vbi
 * Purpose:     Manchester biphase VBI code decoder
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2018-2019 Simon Inns
 */

#ifndef BIPHASECODE_H
#define BIPHASECODE_H

#include <QVector>

#include "lddecodemetadata.h"
#include "sourcevideo.h"

// Decoder for PAL/NTSC LaserDisc biphase code lines.
// Specified in IEC 60586-1986 section 10.1 (PAL) and IEC 60587-1986
// section 10.1 (NTSC).
class BiphaseCode {
 public:
  bool decodeLines(const SourceVideo::Data& line16Data,
                   const SourceVideo::Data& line17Data,
                   const SourceVideo::Data& line18Data,
                   const LdDecodeMetaData::VideoParameters& videoParameters,
                   LdDecodeMetaData::Field& fieldMetadata);
  bool decodeLine(qint32 lineIndex, const SourceVideo::Data& lineData,
                  const LdDecodeMetaData::VideoParameters& videoParameters,
                  LdDecodeMetaData::Field& fieldMetadata);

 private:
  qint32 manchesterDecoder(const SourceVideo::Data& lineData, qint32 zcPoint,
                           LdDecodeMetaData::VideoParameters videoParameters);
};

#endif  // BIPHASECODE_H
