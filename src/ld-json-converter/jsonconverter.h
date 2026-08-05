/*
 * File:        jsonconverter.h
 * Module:      conversion
 * Purpose:     Converts .tbc.json metadata into the SQLite .tbc.db format
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2025 Simon Inns
 */

#ifndef JSONCONVERTER_H
#define JSONCONVERTER_H

#include <QSqlDatabase>
#include <QSqlError>
#include <QSqlQuery>
#include <QString>

#include "lddecodemetadata.h"

class JsonConverter {
 public:
  JsonConverter(const QString &inputJsonFilename,
                const QString &outputSqliteFilename);
  ~JsonConverter();

  bool process();

 private:
  QString m_inputJsonFilename;
  QString m_outputSqliteFilename;
  QSqlDatabase m_database;

  void reportJsonContents(LdDecodeMetaData &metaData);
  void countDropouts(const LdDecodeMetaData &metaData, qint32 &totalDropouts);
  bool createDatabase();
  bool createSchema();
  bool insertData(LdDecodeMetaData &metaData);
};

#endif  // JSONCONVERTER_H