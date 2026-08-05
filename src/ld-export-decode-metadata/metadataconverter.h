/*
 * File:        metadataconverter.h
 * Module:      conversion
 * Purpose:     Translates ld-decode metadata into the export model
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2025 Simon Inns
 */

#ifndef JSONCONVERTER_H
#define JSONCONVERTER_H

#include <QString>

#include "exportmetadata.h"
#include "lddecodemetadata.h"

class MetadataConverter {
 public:
  MetadataConverter(const QString &inputSqliteFilename,
                    const QString &outputJsonFilename);
  ~MetadataConverter();

  bool process();

 private:
  QString m_inputSqliteFilename;
  QString m_outputJsonFilename;

  void convertVideoParamters(
      const LdDecodeMetaData::VideoParameters &in_VideoParameters,
      ExportMetaData::VideoParameters &out_VideoParameters);
  void convertPcmAudioParamters(
      const LdDecodeMetaData::PcmAudioParameters &in_PcmAudioParameters,
      ExportMetaData::PcmAudioParameters &out_PcmAudioParameters);
  void convertField(const LdDecodeMetaData::Field &in_field,
                    ExportMetaData::Field &out_field);
};

#endif  // JSONCONVERTER_H