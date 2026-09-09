/******************************************************************************
 * cadencedefs.h
 * ld-decode-tools TBC library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2025-2026 Joseph Burns
 *
 * This file is part of ld-decode-tools.
 *
 * Source of truth for telecine cadence handling.
 *
 * Shared cadenceId contract:
 *   -1  CADENCE_UNKNOWN   — unknown / no cadence assigned (video)
 *   -2  CADENCE_VIDEO     — identified as 59.94i interlaced video
 *   -3  CADENCE_PROGRESSIVE — identified as progressive
 *
 * Negative values are never given "known" flags so they are protected
 * from pulldown consolidation.
 *
 *   0..9  = normal dominance
 *  10..19 = inverted dominance
 *
 * Index layout (AA AB BC CC DD):
 *   0, 1 : AA — Pure Frame A
 *   2, 3 : AB — Mixed Frame A/B
 *   4, 5 : BC — Mixed Frame B/C
 *   6, 7 : CC — Pure Frame C
 *   8, 9 : DD — Pure Frame D
 *
 * Disambiguation: we use 3:2 pulldown counting. 2:3 differs as follows —
 * 2:3 pulldown counts from one film frame earlier, so 3:2's A is 2:3's B,
 * and 2:3's A is 3:2's D.
 ******************************************************************************/

#pragma once

static constexpr int CADENCE_UNKNOWN            = -1;
static constexpr int CADENCE_VIDEO              = -2;
static constexpr int CADENCE_PROGRESSIVE        = -3;
static constexpr int CADENCE_NTSC_CYCLE         = 10;
static constexpr int CADENCE_NTSC_INVERTED_OFFSET = 10;

static inline bool cadenceKnown(int cid) { return cid >= 0; }
static inline bool cadenceIsInverted(int cid) { return cid >= CADENCE_NTSC_INVERTED_OFFSET; }

// Normalise cadenceId into 0..9 index space.
// Every sentinel returns CADENCE_UNKNOWN: only a known id has a cadence
// position. Wrapping a negative into 0..9 would hand VIDEO and PROGRESSIVE
// the identities of DD and Cdef, defeating the protection above.
static inline int cadenceIndex(int cid)
{
    if (!cadenceKnown(cid)) return CADENCE_UNKNOWN;
    return cid % CADENCE_NTSC_CYCLE;
}

// -----------------------------------------------------------------------------
// Pulldown Role Semantics
// -----------------------------------------------------------------------------

// Returns true if this cadence position represents a Definitional field (A1 or C1).
// These are cadence anchor fields.
static inline bool isDefinitionalRole(int cid)
{
    if (!cadenceKnown(cid)) return false;
    int idx = cadenceIndex(cid);
    return (idx == 0 || idx == 7);
}

// Returns true if this cadence position represents a Spare field (A3 or C3).
// These are the duplicate fields created by telecine.
static inline bool isSpareRole(int cid)
{
    if (!cadenceKnown(cid)) return false;
    int idx = cadenceIndex(cid);
    return (idx == 2 || idx == 5);
}

// -----------------------------------------------------------------------------
// Mate & Complement Lookups
// -----------------------------------------------------------------------------

// Returns the cadenceId of the Twin mate (the identical-content field).
// Example: 0 (Adef) -> 2 (Aspare); 2 (Aspare) -> 0 (Adef).
static inline int getTwinMateCadenceId(int cid)
{
    if (!cadenceKnown(cid)) return CADENCE_UNKNOWN;
    int idx = cadenceIndex(cid);
    int base = cadenceIsInverted(cid) ? CADENCE_NTSC_INVERTED_OFFSET : 0;

    switch (idx) {
        case 0: return base + 2; // Adef  <-> Aspare
        case 2: return base + 0;
        case 5: return base + 7; // Cspare <-> Cdef
        case 7: return base + 5;
        default: return CADENCE_UNKNOWN; // B and D frames have no twins in standard 3:2
    }
}

// Returns the cadenceId of the Film Frame Complement (the opposite-parity field
// of the same film frame), used to reconstruct the progressive frame.
// Twins share a complement, but definitional fields are favoured when looking
// from the complement for the opposite field.
static inline int getFilmFrameComplementId(int cid)
{
    if (!cadenceKnown(cid)) return CADENCE_UNKNOWN;
    int idx = cadenceIndex(cid);
    int base = cadenceIsInverted(cid) ? CADENCE_NTSC_INVERTED_OFFSET : 0;

    switch (idx) {
        case 0: return base + 1; // def is favoured over spare
        case 1: return base + 0; // A2 is the complement for both A1 and A3
        case 2: return base + 1;
        case 3: return base + 4; // B1 <-> B2
        case 4: return base + 3;
        case 5: return base + 6;
        case 6: return base + 7; // C2 is the complement for both C1 and C3
        case 7: return base + 6; // def is favoured over spare
        case 8: return base + 9; // D1 <-> D2
        case 9: return base + 8;
        default: return CADENCE_UNKNOWN;
    }
}

// Name used by cadenceassembler.cpp and discmapper.cpp.
static inline int filmFrameComplement(int cid) { return getFilmFrameComplementId(cid); }

// Debug overlay helper. Uses normalised 0..9 cadence index space.
static inline char cadenceFilmLetter(int cid)
{
    if (!cadenceKnown(cid)) return '?';
    switch (cadenceIndex(cid)) {
        case 0: case 1: return 'A'; // Adef/Acomp (AA)
        case 2:         return 'A'; // Aspare (AB half)
        case 3: case 4: return 'B'; // B1/B2 (AB/BC halves)
        case 5:         return 'C'; // Cspare (BC half)
        case 6: case 7: return 'C'; // Ccomp/Cdef (CC)
        case 8: case 9: return 'D'; // D1/D2 (DD)
        default:        return '?';
    }
}

// -----------------------------------------------------------------------------
// 24p Timeline Position
// -----------------------------------------------------------------------------
//
// A field's seqNo is its 59.94 Hz time coordinate. Since 23.976 / 59.94 = 2/5,
// a frame's position on the 24p timeline is derived from its seqNo and cadence
// slot. Ten fields advance the index by exactly four at every phase.
//
// Film frames anchor on the cycle head. The head plus an ordinal frame offset
// gives an exact position at every phase and may be virtual, so short shots
// resolve even when their cycle head is outside the shot or render.
//
// Integer arithmetic is exact: 2n/5 has fractional part in {0, .2, .4, .6,
// .8}, so no tie can arise. This value gates which frames are emitted.

// Round 2n/5 half-up, including negative n. Cycle heads may be negative for
// fields in the file's first cycle, so the result remains defined across zero.
static inline long long cadenceRoundTwoFifths(long long n)
{
    const long long num = 2 * n + 2;
    return (num >= 0) ? (num / 5) : -(((-num) + 4) / 5);
}

// Ordinal position of a slot's film frame within its cycle: A=0, B=1, C=2, D=3.
// Tracks cadenceFilmLetter's slot map exactly. -1 for any unknown cadence.
static inline int cadenceFilmFrameOrdinal(int cid)
{
    if (!cadenceKnown(cid)) return -1;
    switch (cadenceIndex(cid)) {
        case 0: case 1: case 2: return 0; // A: def, comp, trailing spare
        case 3: case 4:         return 1; // B: no twin, straddles two frames
        case 5: case 6: case 7: return 2; // C: leading spare, comp, def
        case 8: case 9:         return 3; // D: no twin
        default:                return -1;
    }
}

// seqNo of the slot-0 field of this field's cadence cycle. Virtual: the head
// field need not be present in the stream, in this shot, or in this render.
static inline long long cadenceCycleHeadSeq(int cid, long long seqNo)
{
    if (!cadenceKnown(cid)) return -1;
    return seqNo - cadenceIndex(cid);
}

// 24p timeline index of the film frame a given field belongs to.
//
// Every field of a film frame returns the same value: B1 (slot 3, seq s) and
// B2 (slot 4, seq s+1) both resolve to head s-3. Callers holding both fields
// can assert that their results agree; a disagreement means the pair is not
// one film frame.
static inline long long filmFrameIndex24p(int cid, long long seqNo)
{
    const int ordinal = cadenceFilmFrameOrdinal(cid);
    if (ordinal < 0) return -1;
    const long long head = cadenceCycleHeadSeq(cid, seqNo);
    return cadenceRoundTwoFifths(head) + ordinal;
}

// 24p timeline index of an unassembled capture frame -- video, progressive, or
// telecine the assembler could not claim -- from its FIRST field's seqNo.
//
// One in every five positions collides with its predecessor, representing the
// 29.97 -> 23.976 decimation.
static inline long long captureFrameIndex24p(long long firstFieldSeqNo)
{
    return cadenceRoundTwoFifths(firstFieldSeqNo);
}
