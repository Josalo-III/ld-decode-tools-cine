/******************************************************************************
 * fieldorder.h
 * ld-decode-tools TBC library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 Joseph Burns
 *
 * This file is part of ld-decode-tools.
 *
 * Central field order policy: a single object that answers "what is the
 * temporal order of the frame's two fields" and provides the canonical
 * cadence-id mapping for the 5-frame telecine group.
 *
 * Prefer this over local handling to avoid drift.
 ******************************************************************************/

#pragma once

#include <utility>

struct FieldOrderPolicy
{
    // false: interpret stored pair (seqField1, seqField2) as (first, second)
    // true:  interpret them swapped — (seqField2, seqField1) as (first, second)
    bool reverse = false;

    // Return the sequential field numbers in temporal order (first, second).
    inline std::pair<int,int> temporalOrder(int seqField1, int seqField2) const
    {
        return reverse ? std::pair<int,int>{ seqField2, seqField1 }
                       : std::pair<int,int>{ seqField1, seqField2 };
    }

    // Returns (cadenceId_for_temporal_first, cadenceId_for_temporal_second)
    // for the CAV 5-frame group position [0..4]. Base indices are per stored
    // slot; they are swapped when reverse=true to yield temporal order.
    inline std::pair<int,int> cavCadenceIdsForFrameInGroup(int frameInGroup) const
    {
        const int base = frameInGroup * 2; // 0,2,4,6,8 for frames 0..4
        int idStoredFirst  = base;
        int idStoredSecond = base + 1;

        if (reverse) std::swap(idStoredFirst, idStoredSecond);
        return { idStoredFirst, idStoredSecond };
    }
};
