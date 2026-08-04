/*
 * File:        vitsanalyser.h
 * Module:      analysis
 * Purpose:     Per-field white SNR and black PSNR measurement from VITS lines
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2020 Simon Inns
 */

#ifndef VITSANALYSER_H
#define VITSANALYSER_H

#include <QAtomicInt>
#include <QDebug>
#include <QObject>
#include <QThread>
#include <cmath>

#include "lddecodemetadata.h"
#include "sourcevideo.h"

class ProcessingPool;

class VitsAnalyser : public QThread {
  Q_OBJECT

 public:
  explicit VitsAnalyser(QAtomicInt& _abort, ProcessingPool& _processingPool,
                        QObject* parent = nullptr);

 protected:
  void run() override;

 private:
  // Decoder pool
  QAtomicInt& abort;
  ProcessingPool& processingPool;

  // Temporary output buffer
  LdDecodeMetaData::Field outputData;

  // Other settings
  LdDecodeMetaData::VideoParameters videoParameters;

  QVector<double> getFieldLineSlice(const SourceVideo::Data& sourceField,
                                    qint32 fieldLine, qint32 startUs,
                                    qint32 lengthUs);
  double calculateSnr(QVector<double>& data, bool usePsnr);
  double calcMean(QVector<double>& data);
  double calcStd(QVector<double>& data);
  double roundDouble(double in, qint32 decimalPlaces);
};

#endif  // VITSANALYSER_H
