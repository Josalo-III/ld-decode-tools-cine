/************************************************************************

    stackingmodes.h

    ld-disc-stacker - Disc stacking for ld-decode
    Copyright (C) 2020-2025 Simon Inns
    Copyright (C) 2025-2026 Joseph Burns

    This file is part of ld-decode-tools.

    ld-disc-stacker is free software: you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

************************************************************************/

#ifndef STACKINGMODES_H
#define STACKINGMODES_H

#include <QString>
#include <QStringList>
#include <QVector>
#include <QtGlobal>

// ---------------------------------------------------------------------------
// Single source of truth for ld-disc-stacker's stacking modes.
//
// A mode's id, its short name, and its long description live here exactly once.
// The "-m" option summary, the "--help-mode" listing, and the "-m" range check
// are all derived from this table, so they can never disagree the way three
// separately hand-maintained copies once did (the -m summary and the --help-mode
// text had drifted apart, describing modes 5 and 6 in opposite orders).
//
// This table describes the modes; it does not dispatch them. Dispatch is the
// switch on the mode id in Stacker::stackMode(). Keep that switch and this table
// in the same order and definition — this table is the authority for what each
// case means.
// ---------------------------------------------------------------------------

struct StackMode {
    qint32      id;
    const char* name;         // short label used in the -m summary
    const char* description;  // full explanation used in --help-mode
};

// Ordered by id: AUTO (-1) first, then 0..N.
static const QVector<StackMode> STACKING_MODES = {
    { -1, "auto",
      "select mode depending on the number of frames available "
      "(2f: mean, 3~4f: smart mean, 5+f: smart neighbor)" },
    {  0, "mean",
      "average all samples not marked as dropout using the mean" },
    {  1, "median",
      "take the median of the samples not marked as dropout" },
    {  2, "smart mean",
      "take the median of the samples not marked as dropout, then average all "
      "values within smartThreshold of that median" },
    {  3, "smart neighbor",
      "take the median of each surrounding pixel's samples, find the sample "
      "closest to each neighbour median, take the one of those closest to the "
      "current sample's median, then average all values within smartThreshold of "
      "it; when only 2 sources are available, take the sample closest to the "
      "neighbour" },
    {  4, "neighbor",
      "as smart neighbor, but average the selected sample with the current "
      "sample's median rather than threshold-averaging; when only 2 sources are "
      "available, take the sample closest to the neighbour" },
    {  5, "local neighbor",
      "derive a medoid from the sample set, use it to identify and exclude "
      "outliers, then apply the neighbor process (as in mode 4) on the inlier set" },
    {  6, "smart local neighbor",
      "derive a medoid to exclude outliers, then apply the smart neighbor "
      "process (as in mode 3) on the inlier set" },
    {  7, "medoid",
      "return the sample with the shortest total distance to all other samples "
      "(falls back to median/mean for small sets)" },
};

// Default mode when none is requested on the command line (AUTO).
constexpr qint32 STACKING_MODE_AUTO = -1;

// Lowest / highest valid ids, derived from the table so the -m range check
// tracks the table automatically when modes are added or removed.
inline qint32 stackingModeMinId()
{
    qint32 lo = STACKING_MODES.first().id;
    for (const StackMode& m : STACKING_MODES) if (m.id < lo) lo = m.id;
    return lo;
}

inline qint32 stackingModeMaxId()
{
    qint32 hi = STACKING_MODES.first().id;
    for (const StackMode& m : STACKING_MODES) if (m.id > hi) hi = m.id;
    return hi;
}

// One-line "-m" option summary, e.g. "-1 = auto / 0 = mean / 1 = median / ...".
inline QString stackingModeSummary()
{
    QStringList parts;
    for (const StackMode& m : STACKING_MODES)
        parts << QStringLiteral("%1 = %2").arg(m.id).arg(QString::fromLatin1(m.name));
    return parts.join(QStringLiteral(" / "));
}

// Multi-line "--help-mode" listing, one mode per block: "(id) name : description".
inline QString stackingModeHelp()
{
    // Widest "(id) name" prefix, so the descriptions line up.
    int prefixWidth = 0;
    for (const StackMode& m : STACKING_MODES) {
        const int w = QStringLiteral("(%1) %2").arg(m.id).arg(QString::fromLatin1(m.name)).length();
        if (w > prefixWidth) prefixWidth = w;
    }

    QStringList lines;
    for (const StackMode& m : STACKING_MODES) {
        const QString prefix = QStringLiteral("(%1) %2").arg(m.id).arg(QString::fromLatin1(m.name));
        lines << QStringLiteral("%1 : %2")
                     .arg(prefix, -prefixWidth)
                     .arg(QString::fromLatin1(m.description));
    }
    return lines.join(QStringLiteral("\n"));
}

#endif // STACKINGMODES_H
