/*
 * File:        dataconverter.h
 * Module:      conversion
 * Purpose:     Packed 10-bit to unpacked 16-bit .lds sample conversion
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2019 Simon Inns
 */

#ifndef DATACONVERTER_H
#define DATACONVERTER_H

#include <QDebug>
#include <QFile>
#include <QObject>

class DataConverter : public QObject {
  Q_OBJECT
 public:
  explicit DataConverter(QString inputFileNameParam,
                         QString outputFileNameParam, bool isPackingParam,
                         bool isRIFFParam, QObject *parent = nullptr);
  bool process(void);

 signals:

 public slots:

 private:
  QString inputFileName;
  QString outputFileName;
  bool isPacking;
  bool isRIFF;

  QFile *inputFileHandle;
  QFile *outputFileHandle;

  // Private methods
  bool openInputFile(void);
  void closeInputFile(void);
  bool openOutputFile(void);
  void closeOutputFile(void);
  void packFile(void);
  void unpackFile(void);
};

#endif  // DATACONVERTER_H
