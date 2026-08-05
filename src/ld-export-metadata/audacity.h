/*
 * File:        audacity.h
 * Module:      export
 * Purpose:     Chapter changes as an Audacity label track
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2023 Adam Sampson
 */

#ifndef AUDACITY_H
#define AUDACITY_H

#include <QString>

#include "lddecodemetadata.h"

/*!
    Write an Audacity labels file containing navigation information.

    Format description:
   <https://manual.audacityteam.org/man/importing_and_exporting_labels.html>

    Returns true on success, false on failure.
*/
bool writeAudacityLabels(LdDecodeMetaData &metaData, const QString &fileName);

#endif
