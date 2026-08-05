/*
 * File:        vbilinedecoder.h
 * Module:      vbi
 * Purpose:     Per-field VBI line decode worker
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2018-2019 Simon Inns
 */

#ifndef VBILINEDECODER_H
#define VBILINEDECODER_H

#include <QAtomicInt>
#include <QDebug>
#include <QObject>
#include <QThread>

#include "lddecodemetadata.h"
#include "sourcevideo.h"

class DecoderPool;

class VbiLineDecoder : public QThread {
  Q_OBJECT

 public:
  explicit VbiLineDecoder(QAtomicInt& _abort, DecoderPool& _decoderPool,
                          QObject* parent = nullptr);

  // The range of field lines needed from the input file (1-based, inclusive)
  static constexpr qint32 startFieldLine = 6;
  static constexpr qint32 endFieldLine = 22;

 protected:
  void run() override;

 private:
  // Decoder pool
  QAtomicInt& abort;
  DecoderPool& decoderPool;

  SourceVideo::Data getFieldLine(
      const SourceVideo::Data& sourceField, qint32 fieldLine,
      const LdDecodeMetaData::VideoParameters& videoParameters);
};

#endif  // VBILINEDECODER_H
