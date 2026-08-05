/*
 * File:        exportdropouts.h
 * Module:      model
 * Purpose:     Export-side dropout run model
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2018-2020 Simon Inns
 */

// Note: Copied from the TBC library so the JSON handling code is local to the
// application

#ifndef EXPORTDROPOUTS_H
#define EXPORTDROPOUTS_H

#include <QDebug>
#include <QMetaType>
#include <QVector>
#include <QtGlobal>

class JsonReader;
class JsonWriter;

class ExportDropOuts {
 public:
  ExportDropOuts() = default;
  ExportDropOuts(int reserve);
  ~ExportDropOuts() = default;
  ExportDropOuts(const ExportDropOuts &) = default;

  ExportDropOuts(const QVector<qint32> &startx, const QVector<qint32> &endx,
                 const QVector<qint32> &fieldLine);
  ExportDropOuts &operator=(const ExportDropOuts &);

  void append(const qint32 startx, const qint32 endx, const qint32 fieldLine);
  void reserve(int size);
  void resize(qint32 size);
  void clear();
  void concatenate(const bool verbose = true);

  // Return the number of dropouts
  qint32 size() const { return m_startx.size(); }

  // Return true if there are no dropouts
  bool empty() const { return m_startx.empty(); }

  // Get position of a dropout
  qint32 startx(qint32 index) const { return m_startx[index]; }
  qint32 endx(qint32 index) const { return m_endx[index]; }
  qint32 fieldLine(qint32 index) const { return m_fieldLine[index]; }

  void read(JsonReader &reader);
  void write(JsonWriter &writer) const;

 private:
  QVector<qint32> m_startx;
  QVector<qint32> m_endx;
  QVector<qint32> m_fieldLine;

  void readArray(JsonReader &reader, QVector<qint32> &array);
  void writeArray(JsonWriter &writer, const QVector<qint32> &array) const;
};

#endif  // EXPORTDROPOUTS_H
