/*
 * File:        videoid.h
 * Module:      vbi
 * Purpose:     VIDEO ID decoder as defined in IEC 61880
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2018-2019 Simon Inns
 */

#ifndef VIDEOID_H
#define VIDEOID_H

#include "lddecodemetadata.h"
#include "sourcevideo.h"

class VideoID {
 public:
  bool decodeLine(const SourceVideo::Data& lineData,
                  const LdDecodeMetaData::VideoParameters& videoParameters,
                  LdDecodeMetaData::Field& fieldMetadata);
};

#endif  // VIDEOID_H
