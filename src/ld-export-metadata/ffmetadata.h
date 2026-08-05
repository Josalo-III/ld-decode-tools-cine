/*
 * File:        ffmetadata.h
 * Module:      export
 * Purpose:     Chapter markers as an FFmpeg ffmetadata file
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2020 Adam Sampson
 */

#ifndef FFMETADATA_H
#define FFMETADATA_H

#include <QString>

#include "lddecodemetadata.h"

/*!
    Write an FFMETADATA1 file containing navigation information.

    This is FFmpeg's generic metadata format, and can be used to provide
    metadata for chapter-supporting formats like Matroska.
    Format description: <https://ffmpeg.org/ffmpeg-formats.html#Metadata-1>

    Returns true on success, false on failure.
*/
bool writeFfmetadata(LdDecodeMetaData &metaData, const QString &fileName);

#endif
