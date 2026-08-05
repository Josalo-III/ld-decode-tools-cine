/*
 * File:        csv.h
 * Module:      export
 * Purpose:     VITS and VBI metadata export to CSV
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2018-2019 Simon Inns
 */

#ifndef CSV_H
#define CSV_H

#include <QString>

#include "lddecodemetadata.h"

/*!
    Write the per-field VITS metrics as a CSV file.

    Returns true on success, false on failure.
*/
bool writeVitsCsv(LdDecodeMetaData &metaData, const QString &fileName);

/*!
    Write the per-frame VBI information as a CSV file.

    Returns true on success, false on failure.
*/
bool writeVbiCsv(LdDecodeMetaData &metaData, const QString &fileName);

#endif
