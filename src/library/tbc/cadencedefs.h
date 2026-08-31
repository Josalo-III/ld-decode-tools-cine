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
// These are the anchor fields verified in post-production.
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

// Returns true if the cadence index points to a Mixed (Jitter) frame.
// AA(0,1) -> Progressive; AB(2,3) -> Mixed; BC(4,5) -> Mixed;
// CC(6,7) -> Progressive; DD(8,9) -> Progressive.
static inline bool cadenceIsMixed(int cid)
{
    if (!cadenceKnown(cid)) return false;
    int idx = cadenceIndex(cid);
    return (idx >= 2 && idx <= 5);
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

// Returns true if the TBC stream frame containing this cadence ID is a Pure frame
// (i.e. both fields are from the same film frame).
static inline bool isCleanFrame(int cid)
{
    if (!cadenceKnown(cid)) return false;
    int idx = cadenceIndex(cid);
    // AA(0,1), CC(6,7), DD(8,9) are clean; AB(2,3) and BC(4,5) are mixed.
    return (idx <= 1) || (idx >= 6);
}

// Aliases used by cadenceassembler.cpp and discmapper.cpp
static inline int filmFrameComplement(int cid) { return getFilmFrameComplementId(cid); }
static inline int twinMateCadence(int cid)     { return getTwinMateCadenceId(cid); }
static inline bool isDefinitionalTwin(int cid) { return isDefinitionalRole(cid); }

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
// A field's seqNo IS its 59.94 Hz time coordinate, and 23.976 / 59.94 = 2/5
// exactly, so a frame's position on the 24p timeline is a function of its own
// seqNo and nothing else: no run origin, no accumulator, no history. The
// proportional relationship re-establishes exactly once per cadence cycle --
// ten fields advance the index by exactly four, for ANY phase -- which is what
// makes a stateless answer possible at all.
//
// Film frames anchor on the CYCLE HEAD, not on their own first field. Anchoring
// per-frame rounds differently depending on (seqNo mod 5) and manufactures
// collisions and gaps inside perfectly clean cadence; the head plus an ordinal
// letter offset is exact at every phase. The head is DERIVED, never looked up,
// so it may be virtual -- a shot lasting three fields still resolves correctly
// even though its cycle head lies outside the shot, or outside the render.
//
// Integer arithmetic throughout: 2n/5 has fractional part in {0, .2, .4, .6,
// .8}, so no tie can arise. This value gates which frames are emitted, and a
// frame drop must never turn on a floating-point rounding mode -- a float form
// disagrees with the exact answer on real discs, because 0.4 is not
// representable in binary.

// round(2n/5), half-up, correct for NEGATIVE n as well: C++ integer division
// truncates toward zero, which is not floor, so the naive form is wrong below
// the origin. A cycle head is virtual and legitimately negative for a field in
// the file's first cycle (a slot-9 field at seqNo 1 heads at -8), so this must
// not be guarded away -- only differences between indices carry meaning, and
// the sequence has to stay monotonic across zero.
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
// EVERY field of a film frame returns the same value: B1 (slot 3, seq s) and
// B2 (slot 4, seq s+1) both resolve to head s-3. So a caller may ask with
// either field of an assembled pair, and the comb-ordering swap that puts B2
// first is irrelevant to the answer. Callers holding both fields should assert
// they agree; a disagreement means the pair is not one film frame.
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
// One in every five collides with its predecessor. That collision IS the
// 29.97 -> 23.976 decimation, arrived at by position and therefore evenly
// spaced, rather than timed by an accumulator that fires wherever its running
// sum happens to cross.
static inline long long captureFrameIndex24p(long long firstFieldSeqNo)
{
    return cadenceRoundTwoFifths(firstFieldSeqNo);
}
