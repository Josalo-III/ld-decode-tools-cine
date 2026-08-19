/*
 * File:        visualedits.h
 * Module:      edit-detection
 * Purpose:     Visual edit detection from inter-field luma and chroma change
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 Joseph Burns
 */

#pragma once

class CineDisc;

namespace visualEdits {
int analyseVisualEdits(CineDisc& disc, double threshold, double strongFactor,
                       double peakFactor, bool traceEnabled = false,
                       int windowStart = 0, int windowEnd = 0);
}
