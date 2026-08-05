/*
 * File:        decoderpool.h
 * Module:      threading
 * Purpose:     Field distribution and result ordering across decoder threads
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2018-2019 Simon Inns
 */

#ifndef DECODERPOOL_H
#define DECODERPOOL_H

#include <QAtomicInt>
#include <QElapsedTimer>
#include <QMutex>
#include <QThread>

#include "lddecodemetadata.h"
#include "sourcevideo.h"
#include "vbilinedecoder.h"

class DecoderPool {
 public:
  // Public methods
  explicit DecoderPool(QString _inputFilename, QString _outputMetadataFilename,
                       qint32 _maxThreads, LdDecodeMetaData &_ldDecodeMetaData);
  bool process();

  // Member functions used by worker threads
  bool getInputField(qint32 &fieldNumber, SourceVideo::Data &fieldVideoData,
                     LdDecodeMetaData::Field &fieldMetadata,
                     LdDecodeMetaData::VideoParameters &videoParameters);
  bool setOutputField(qint32 fieldNumber,
                      const LdDecodeMetaData::Field &fieldMetadata);

 private:
  QString inputFilename;
  QString outputMetadataFilename;
  qint32 maxThreads;
  QElapsedTimer totalTimer;

  // Atomic abort flag shared by worker threads; workers watch this, and shut
  // down as soon as possible if it becomes true
  QAtomicInt abort;

  // Input stream information (all guarded by inputMutex while threads are
  // running)
  QMutex inputMutex;
  qint32 inputFieldNumber;
  qint32 lastFieldNumber;
  LdDecodeMetaData &ldDecodeMetaData;
  SourceVideo sourceVideo;

  // Output stream information (all guarded by outputMutex while threads are
  // running)
  QMutex outputMutex;
  QFile targetMetadata;
};

#endif  // DECODERPOOL_H
