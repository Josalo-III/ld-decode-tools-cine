/*
 * File:        dropoutcorrect.h
 * Module:      correction
 * Purpose:     Dropout replacement from intra-field, inter-field and
 * multi-source data
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2018-2020 Simon Inns
 * SPDX-FileCopyrightText: 2019-2020 Adam Sampson
 */

#ifndef DROPOUTCORRECT_H
#define DROPOUTCORRECT_H

#include <QAtomicInt>
#include <QDebug>
#include <QElapsedTimer>
#include <QObject>
#include <QThread>

#include "lddecodemetadata.h"
#include "sourcevideo.h"

class CorrectorPool;

class DropOutCorrect : public QThread {
  Q_OBJECT
 public:
  explicit DropOutCorrect(QAtomicInt &_abort, CorrectorPool &_correctorPool,
                          QObject *parent = nullptr);

 protected:
  void run() override;

 private:
  enum Location { visibleLine, colourBurst, unknown };

  struct DropOutLocation {
    qint32 fieldLine;
    qint32 startx;
    qint32 endx;
    Location location;
  };

  struct Replacement {
    // The default value is no replacement
    Replacement() : isSameField(true), fieldLine(-1) {}

    bool isSameField;
    qint32 fieldLine;

    qint32 sourceNumber;
    double quality;

    qint32 distance;
  };

  // Statistics
  struct Statistics {
    qint32 sameSourceConcealment;
    qint32 multiSourceConcealment;
    qint32 multiSourceCorrection;
    qint32 totalReplacementDistance;
  };

  // Decoder pool
  QAtomicInt &abort;
  CorrectorPool &correctorPool;

  QVector<LdDecodeMetaData::VideoParameters> videoParameters;

  void correctField(const QVector<QVector<DropOutLocation>> &thisFieldDropouts,
                    const QVector<QVector<DropOutLocation>> &otherFieldDropouts,
                    QVector<SourceVideo::Data> &thisFieldData,
                    const QVector<SourceVideo::Data> &otherFieldData,
                    bool thisFieldIsFirst, bool intraField,
                    const QVector<qint32> &availableSourcesForFrame,
                    const QVector<double> &sourceFrameQuality,
                    Statistics &statistics);
  QVector<DropOutLocation> populateDropoutsVector(LdDecodeMetaData::Field field,
                                                  bool overCorrect);
  QVector<DropOutLocation> setDropOutLocations(
      QVector<DropOutLocation> dropOuts);
  Replacement findReplacementLine(
      const QVector<QVector<DropOutLocation>> &thisFieldDropouts,
      const QVector<QVector<DropOutLocation>> &otherFieldDropouts,
      qint32 dropOutIndex, bool thisFieldIsFirst, bool matchChromaPhase,
      bool isColourBurst, bool intraField,
      const QVector<qint32> &availableSourcesForFrame,
      const QVector<double> &sourceFrameQuality);
  void findPotentialReplacementLine(
      const QVector<QVector<DropOutLocation>> &targetDropouts,
      qint32 targetIndex,
      const QVector<QVector<DropOutLocation>> &sourceDropouts, bool isSameField,
      qint32 sourceOffset, qint32 stepAmount, qint32 sourceNo,
      const QVector<double> &sourceFrameQuality,
      QVector<Replacement> &candidates);
  void correctDropOut(const DropOutLocation &dropOut,
                      const Replacement &replacement,
                      const Replacement &chromaReplacement,
                      QVector<SourceVideo::Data> &thisFieldData,
                      const QVector<SourceVideo::Data> &otherFieldData,
                      Statistics &statistics);
};

#endif  // DROPOUTCORRECT_H
