/*
 * File:        vbiprobe.h
 * Module:      vbi
 * Purpose:     Per-frame VBI decode summary and CAV/CLV disc-type vote
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 Joseph Burns
 */

#pragma once

#include <QtGlobal>
#include <vector>

class CineDisc;

namespace vbiProbe {

// Per-frame decoded VBI summary.  One entry per frame, indexed 0-based.
struct FrameVbi {
  qint32 picNo = -1;     // > 0 if CAV picture number present; -1 if absent
  bool isClv = false;    // true if a complete CLV timecode was decoded
  bool leadIn = false;   // VBI lead-in flag
  bool leadOut = false;  // VBI lead-out flag
  bool padded = false;   // frame was marked as padded in capture
};

struct ProbeResult {
  bool isDiscCav = false;        // true if CAV wins the vote
  std::vector<FrameVbi> frames;  // one entry per frame, 0-based

  // Convenience: returns picNo for a 0-based frame index, or -1 if out of
  // range.
  qint32 picNo(int frameIdx) const {
    if (frameIdx < 0 || frameIdx >= static_cast<int>(frames.size())) return -1;
    return frames[frameIdx].picNo;
  }

  // Returns true if the frame should be excluded from cadence solving
  // (lead-in, lead-out, or padded).
  bool isExcluded(int frameIdx) const {
    if (frameIdx < 0 || frameIdx >= static_cast<int>(frames.size()))
      return true;
    const auto& f = frames[frameIdx];
    return f.leadIn || f.leadOut || f.padded;
  }
};

// Decode VBI for every frame on the disc in a single pass.
// Sets disc.setIsDiscCav() as a side effect so callers don't have to.
// Logs a critical error and returns isDiscCav=false if neither CAV nor CLV
// VBI is detected (treat as a fatal load failure).
ProbeResult probe(CineDisc& disc);

}  // namespace vbiProbe
