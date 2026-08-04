/*
 * File:        discmapper.h
 * Module:      mapping
 * Purpose:     VBI frame-number correction, reordering and gap padding
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2019-2022 Simon Inns
 */

#ifndef DISCMAPPER_H
#define DISCMAPPER_H

#include <QCoreApplication>
#include <QDebug>
#include <QFile>
#include <QFileInfo>

// TBC library includes
#include "discmap.h"
#include "lddecodemetadata.h"
#include "sourceaudio.h"
#include "sourcevideo.h"

class DiscMapper {
 public:
  DiscMapper();

  bool process(QFileInfo _inputFileInfo, QFileInfo _inputMetadataFileInfo,
               QFileInfo _outputFileInfo, bool _reverse, bool _mapOnly,
               bool _noStrict, bool _deleteUnmappable, bool _noAudio);

 private:
  // These members hold the arguments passed to process() for use by the
  // private mapping stage methods called from it.
  QFileInfo inputFileInfo;
  QFileInfo inputMetadataFileInfo;
  QFileInfo outputFileInfo;
  bool reverse;
  bool mapOnly;
  bool noStrict;
  bool deleteUnmappable;
  bool noAudio;

  void removeLeadInOut(DiscMap& discMap);
  void removeInvalidFramesByPhase(DiscMap& discMap);
  void correctVbiFrameNumbersUsingSequenceAnalysis(DiscMap& discMap);
  void removeDuplicateNumberedFrames(DiscMap& discMap);
  void numberPulldownFrames(DiscMap& discMap);
  bool verifyFrameNumberPresence(DiscMap& discMap);
  void reorderFrames(DiscMap& discMap);
  void padDiscMap(DiscMap& discMap);
  void rewriteFrameNumbers(DiscMap& discMap);
  void deleteUnmappableFrames(DiscMap& discMap);

  bool saveDiscMap(DiscMap& discMap);
};

#endif  // DISCMAPPER_H
