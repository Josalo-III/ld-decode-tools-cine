/*
 * File:        closedcaption.h
 * Module:      vbi
 * Purpose:     NTSC line 21 closed-caption data recovery
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2018-2019 Simon Inns
 */

#ifndef CLOSEDCAPTION_H
#define CLOSEDCAPTION_H

#include "lddecodemetadata.h"
#include "sourcevideo.h"

class ClosedCaption {
 public:
  bool decodeLine(const SourceVideo::Data& lineData,
                  const LdDecodeMetaData::VideoParameters& videoParameters,
                  LdDecodeMetaData::Field& fieldMetadata);
};

#endif  // CLOSEDCAPTION_H
