/*
 * File:        closedcaptions.h
 * Module:      export
 * Purpose:     Line 21 captions as Scenarist Closed Caption (SCC) V1.0
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2019-2020 Adam Sampson
 * SPDX-FileCopyrightText: 2021 Simon Inns
 */

#ifndef CLOSEDCAPTIONS_H
#define CLOSEDCAPTIONS_H

#include <QString>

#include "lddecodemetadata.h"

QString generateTimeStamp(qint32 fieldIndex);
qint32 sanityCheckData(qint32 dataByte);
bool writeClosedCaptions(LdDecodeMetaData &metaData, const QString &fileName);

#endif  // CLOSEDCAPTIONS_H
