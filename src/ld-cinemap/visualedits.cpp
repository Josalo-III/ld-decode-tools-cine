/*
 * File:        visualedits.cpp
 * Module:      edit-detection
 * Purpose:     Visual edit detection from inter-field luma and chroma change
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 Joseph Burns
 */

#include "visualedits.h"

#include <QElapsedTimer>
#include <QString>
#include <algorithm>
#include <cmath>
#include <map>

#include "cinedisc.h"
#include "lddecodemetadata.h"
#include "sourcevideo.h"
#include "tbc/logging.h"

namespace visualEdits {

struct FieldDescriptor {
  bool valid = false;
  double cells[9] = {0.0};         // Luma (DC)
  double chromaEnergy[9] = {0.0};  // Chroma Magnitude (approx)

  // Brightest point in the title-safe area, on subcarrier-cancelled luma.
  //
  // The cell means above cannot answer "is there a picture here". A cell
  // averages some seventeen thousand samples, so a starfield — black
  // everywhere except point sources and one small bright object — reports the
  // same DC as an empty frame. That is the same error that drove mixedness
  // from notch to lips: an average is not the question. Blackness is a
  // statement that NOTHING peaks above black, and only a peak can say it.
  double peakIre = 0.0;

  // The picture itself, sampled for Pearson: every eighth sample (a 4fsc
  // pair averaged, so the subcarrier cancels) on every fourth line of the
  // title-safe window, ~5,700 points. Sampled, never averaged into cells —
  // the nine cell means are a separate measure. Pearson over these points
  // sees WHERE the edges are; Pearson over the nine means sees only where
  // the light is, and two matched close-ups on one set share that
  // (46192/46193 read 0.94 on nine cells, 0.65 on the picture). Measured
  // against twice this density on 875 commits: p90 difference 0.009,
  // worst 0.032, identical on every edit where the picture decided.
  std::vector<float> picture;
  int pictureW = 0;
  int pictureH = 0;
};

struct DeltaStats {
  double peak = 0.0;
  double total = 0.0;
  int strongCells = 0;
  double totalChroma = 0.0;  // Total change in chroma energy
};
// Compute a 3×3 descriptor over title-safe area (10–90%), in IRE.
FieldDescriptor computeFieldDescriptor(
    const uint16_t* fieldData, int width, int height,
    const LdDecodeMetaData::VideoParameters& vp) {
  FieldDescriptor desc;

  if (!fieldData || width <= 0 || height <= 0) {
    desc.valid = false;
    std::fill(std::begin(desc.cells), std::end(desc.cells), 0.0);
    return desc;
  }

  // Normalize the descriptor to IRE using the capture's black and white levels.
  const double black = (vp.black16bIre >= 0) ? vp.black16bIre : 0.0;
  const double white =
      (vp.white16bIre > vp.black16bIre) ? vp.white16bIre : 65535.0;
  const double range = white - black;

  auto sampleToIre = [&](uint16_t s16) -> double {
    double s = static_cast<double>(s16);
    if (range > 0.0) {
      return 100.0 * (s - black) / range;
    } else {
      // Fallback if metadata is weird
      return 100.0 * (s / 65535.0);
    }
  };

  // Title-safe ROI: 10–90% in both axes
  const int x0 = static_cast<int>(width * 0.10);
  const int x1 = static_cast<int>(width * 0.90);
  const int y0 = static_cast<int>(height * 0.10);
  const int y1 = static_cast<int>(height * 0.90);

  if (x1 <= x0 || y1 <= y0) {
    desc.valid = false;
    std::fill(std::begin(desc.cells), std::end(desc.cells), 0.0);
    return desc;
  }

  desc.valid = true;
  int cellIdx = 0;

  // 3×3 grid over the title-safe window
  for (int row = 0; row < 3; ++row) {
    int yStart = y0 + ((y1 - y0) * row) / 3;
    int yEnd = y0 + ((y1 - y0) * (row + 1)) / 3;

    for (int col = 0; col < 3; ++col) {
      int xStart = x0 + ((x1 - x0) * col) / 3;
      int xEnd = x0 + ((x1 - x0) * (col + 1)) / 3;

      double sumIre = 0.0;
      double sumAbsDiff = 0.0;
      int pixelCount = 0;

      // Pass 1: Calculate Mean (Luma)
      for (int y = yStart; y < yEnd; ++y) {
        const uint16_t* rowPtr = fieldData + y * width;
        for (int x = xStart; x < xEnd; ++x) {
          sumIre += sampleToIre(rowPtr[x]);
          ++pixelCount;
        }
      }

      double mean = 0.0;
      if (pixelCount > 0) {
        mean = sumIre / pixelCount;
        desc.cells[cellIdx] = mean;
      }

      // Pass 2: Calculate Chroma Energy Proxy
      // In composite video, the "Ripple" around the mean IS the chroma.
      // This is effectively a High-Pass filter removing the Luma.
      for (int y = yStart; y < yEnd; ++y) {
        const uint16_t* rowPtr = fieldData + y * width;
        for (int x = xStart; x < xEnd; ++x) {
          double val = sampleToIre(rowPtr[x]);
          sumAbsDiff += std::abs(val - mean);
        }
      }

      if (pixelCount > 0) {
        desc.chromaEnergy[cellIdx] = sumAbsDiff / pixelCount;
      }
      ++cellIdx;
    }
  }

  // Picture pass, for Pearson on the picture.
  {
    constexpr int SX = 8, SY = 4;
    desc.pictureW = (x1 - 2 - x0) / SX;
    desc.pictureH = (y1 - y0) / SY;
    desc.picture.resize(static_cast<size_t>(desc.pictureW) * desc.pictureH);
    size_t k = 0;
    for (int py = 0; py < desc.pictureH; ++py) {
      const uint16_t* rowPtr = fieldData + (y0 + py * SY) * width;
      for (int px = 0; px < desc.pictureW; ++px) {
        const int x = x0 + px * SX;
        desc.picture[k++] = static_cast<float>(
            0.5 * (sampleToIre(rowPtr[x]) + sampleToIre(rowPtr[x + 2])));
      }
    }
  }

  // Peak pass. At 4fsc, samples x and x+2 are 180 degrees apart in subcarrier
  // phase, so their mean cancels it — the peak is then a luma peak rather than
  // a chroma excursion, and a lone hot sample is halved rather than believed.
  {
    double peak = 0.0;

    for (int y = y0; y < y1; ++y) {
      const uint16_t* rowPtr = fieldData + y * width;
      for (int x = x0; x < x1 && x + 2 < width; ++x) {
        const double luma = (static_cast<double>(rowPtr[x]) +
                             static_cast<double>(rowPtr[x + 2])) *
                            0.5;
        if (luma > peak) peak = luma;
      }
    }

    desc.peakIre = sampleToIre(static_cast<uint16_t>(
        std::min(peak, static_cast<double>(UINT16_MAX))));
  }

  return desc;
}

int analyseVisualEdits(CineDisc& disc, double threshold, double strongFactor,
                       double peakFactor, bool traceEnabled) {
  auto& md = disc.getMetaData();

  // Logging controls. The noisy per-edit/per-candidate detail is gated by
  // traceEnabled (CLI flag --edit-trace). Compile-time consts below opt-in
  // to additional verbosity for source-level debugging.
  const bool LOG_COMMITS = traceEnabled;  // one line per committed edit
  constexpr bool LOG_CANDIDATES =
      false;  // every candidate that passes isChange
  constexpr bool LOG_VERBOSE_REJECT = false;  // why we rejected a candidate
  constexpr bool LOG_RAMP_VETO = false;       // when ramp veto fires
  const bool LOG_PROGRESS = true;  // heartbeat/progress line (always on)

  // Debug target: set to a field index to enable per-field trace; -1 = off.
  constexpr int DBG_FIELD = -1;

  // 1. Clamp Thresholds
  if (threshold > 8.0) threshold = 8.0;
  if (threshold < 0.8) threshold = 0.8;
  if (strongFactor < 1.0) strongFactor = 1.0;
  if (peakFactor < 1.0) peakFactor = 1.0;

  SourceVideo sourceVideo;
  if (!sourceVideo.open(disc.getTbcPath(), disc.getVideoFieldLength()))
    return 0;

  const int totalFields = md.getNumberOfFields();
  if (totalFields < 2) {
    sourceVideo.close();
    return 0;
  }

  const auto vp = md.getVideoParameters();

  // 2. Sliding Window Cache
  std::map<int, FieldDescriptor> descCache;

  int editCount = 0;
  int lastEditFrame = -100;
  int maxReadFrame = 0;

  // Tunables
  const double RAMP_TOTAL_MULT = 4.0;
  const double RAMP_RATIO_MAX = 1.3;
  const int RAMP_HALF_WINDOW = 4;

  // A flash departs and returns; a cut steps and stays. The author's rule,
  // measured 2026-08-18 on two discs: Necessary Evil's lightning leaves the
  // shot by 17-30 IRE and is back within a few fields, its cut at 3425
  // steps to 15 and holds; Civil Defense's strobes leave by 12-26 and
  // return, its cut at 88231 steps to 12.8 and holds at sigma 0.09 over
  // twelve fields. Magnitude inverts on both discs (the flash is the larger
  // excursion), so the test is never size: it is whether the picture comes
  // back. Twelve fields is the hold both cuts showed; a return to within
  // half the candidate's own departure is the picture coming back.
  const int TRANSIENT_HOLD_FIELDS = 12;
  const double TRANSIENT_RETURN_FRAC = 0.5;  // and never above threshold
  // The picture is also back when it matches a recent field as closely as
  // that field's own neighbours match it, whatever the energy says: the
  // nine-cell difference is boosted in the dark, and a 16 IRE shot coming
  // back one IRE dimmer after a bright transient reads 19 boosted IRE
  // (Civil Defense 88229) while a real cut between two dark shots can
  // read 13 (88278, itself a return). The nine cells alone cannot serve —
  // two matched close-ups on one set read 0.95 on them, and the A/B
  // dialogue cuts vetoed as returns — so the match is the picture
  // Pearson, held to the continuous level the reference field shows
  // against its own neighbour: 0.7 on a 16 IRE frame, 0.995 on a lit set.
  const double TRANSIENT_RETURN_CORR = 0.95;      // cells
  const double TRANSIENT_RETURN_SLACK = 0.05;     // below the field's own level
  const int CONTEXT_HALF_SPAN = 6;
  // How far a candidate must be the largest step: two film frames each
  // side. Camera shake is not uniform seam to seam (Emissary s1 3303-3500
  // runs 18-88 IRE), so the nearest seam alone does not speak for the
  // shot; swept 3/6/8/10/12 on that shot: 13/7/5/3/3 commits, and the
  // three at 10 are the shot's genuine outliers. Two edges of one event
  // never compete: a step into flat white is not a step (below).
  const int SIDE_REACH = 10;
  const double MOTION_STRONG_CELL_FRAC = 0.40;

  // Black-span segmentation (IRE; endpoints must be black themselves)
  const double BLACK_ENTER_IRE = 2.4;  // "definitely black"
  const double BLACK_EXIT_IRE = 2.8;   // "definitely not black" (hysteresis)

  // No area may peak above this for the field to count as empty.
  //
  // The DC thresholds above are delicate — 0.4 IRE apart, and set low enough
  // to keep a washed-out black from reading as picture. A peak needs no such
  // care: washed-out black sits a few IRE up, while a star or a lit hull is
  // tens of IRE, so anything in this region separates them and the exact value
  // does not do the work. That is the whole advantage of asking for a peak.
  const double BLACK_PEAK_IRE = 15.0;
  const int BLACK_CONFIRM = 12;  // consecutive black/not-black confirmations

  bool inBlackRun = false;
  int firstBlackField = -1;
  int lastBlackField = -1;
  int consecBlack = 0;
  int consecNotBlack = 0;

  // Cache helpers
  auto safeValid = [&](int idx) -> bool {
    auto it = descCache.find(idx);
    return (it != descCache.end()) ? it->second.valid : false;
  };

  auto getDesc = [&](int idx) -> const FieldDescriptor& {
    static FieldDescriptor dummy;
    auto it = descCache.find(idx);
    return (it != descCache.end()) ? it->second : dummy;
  };

  // Maximum cell luma, used in diagnostic logging.
  auto getMaxLuma = [&](const FieldDescriptor& d) -> double {
    if (!d.valid) return 0.0;
    double m = 0.0;
    for (double c : d.cells) m = std::max(m, c);
    return m;
  };

  // Helper: P90 Luma over 3x3 cells (robust against one bright corner/logo)
  auto getP90Luma = [&](const FieldDescriptor& d) -> double {
    if (!d.valid) return 999.0;
    double v[9];
    for (int k = 0; k < 9; ++k) v[k] = d.cells[k];
    std::nth_element(v, v + 7, v + 9);
    return v[7];
  };

  // A field is empty only if its DC is black AND nothing in it peaks above
  // black. Either condition failing is enough to call it picture: one bright
  // area is a picture no matter what the average says.
  auto isBlackField = [&](double p90Ire, double peakIre) -> bool {
    return p90Ire <= BLACK_ENTER_IRE && peakIre <= BLACK_PEAK_IRE;
  };
  auto isNotBlackField = [&](double p90Ire, double peakIre) -> bool {
    return p90Ire >= BLACK_EXIT_IRE || peakIre > BLACK_PEAK_IRE;
  };

  // Correlation (Pearson on 3x3 mean luma)
  struct CorrResult {
    double corr = 0.0;
    bool informative = false;
    bool bothFlat = false;  // neither field carries contrast: empty ground
    bool intoFlat = false;  // the later field has no contrast: into a white-out
    double pictureCorr = 0.0;  // Pearson over the sampled picture
    bool pictureInfo = false;
  };

  auto computeCorrelation = [&](const FieldDescriptor& a,
                                const FieldDescriptor& b) -> CorrResult {
    CorrResult r;
    if (!a.valid || !b.valid) return r;

    double meanA = 0.0, meanB = 0.0;
    for (int i = 0; i < 9; ++i) {
      meanA += a.cells[i];
      meanB += b.cells[i];
    }
    meanA /= 9.0;
    meanB /= 9.0;

    double num = 0.0, denA = 0.0, denB = 0.0;
    for (int i = 0; i < 9; ++i) {
      const double da = a.cells[i] - meanA;
      const double db = b.cells[i] - meanB;
      num += da * db;
      denA += da * da;
      denB += db * db;
    }

    // Flat => corr not informative (do NOT treat as "strong low corr
    // evidence"). Flat means the nine cells lack CONTRAST: a blown-out flash
    // frame with every cell at 88-97 IRE passed a 0.1 variance floor (a third
    // of an IRE of spread) and its Pearson of glow read 0.22, which the
    // structural lane took as a break. Structure is spread relative to level,
    // not spread in absolute IRE — an absolute floor took structure away
    // from dark scenes (spread 3-5 on a 10 IRE picture, a quarter of the
    // level) and left it on the white-out (spread 6-9 on 90, a twelfth).
    // Measured: the dark scene's cuts 0.20-0.25, the flash frames
    // 0.070-0.085, the real cuts around the flash 0.15-0.25 (the cut out of
    // the sparks shot reads 0.1496 on its outgoing side). 0.12 carries 1.4x
    // to the flash and 1.25x to that cut.
    constexpr double CORR_MIN_CONTRAST = 0.12;
    const double contrastA = (meanA > 1e-6) ? std::sqrt(denA / 9.0) / meanA : 0.0;
    const double contrastB = (meanB > 1e-6) ? std::sqrt(denB / 9.0) / meanB : 0.0;
    r.bothFlat = (contrastA < CORR_MIN_CONTRAST && contrastB < CORR_MIN_CONTRAST);
    r.intoFlat = (contrastB < CORR_MIN_CONTRAST);
    if (contrastA < CORR_MIN_CONTRAST || contrastB < CORR_MIN_CONTRAST)
      return r;

    r.corr = num / (std::sqrt(denA) * std::sqrt(denB));
    r.informative = true;
    return r;
  };

  // Pearson on the picture. The nine-cell Pearson answers "same lighting
  // layout"; this one answers "same picture".
  auto pictureCorrelation = [&](const FieldDescriptor& a,
                                const FieldDescriptor& b, CorrResult& r) {
    if (!a.valid || !b.valid) return;
    if (a.picture.size() != b.picture.size() || a.picture.empty()) return;
    const size_t n = a.picture.size();
    double ma = 0.0, mb = 0.0;
    for (size_t i = 0; i < n; ++i) {
      ma += a.picture[i];
      mb += b.picture[i];
    }
    ma /= n;
    mb /= n;
    double num = 0.0, da2 = 0.0, db2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
      const double da = a.picture[i] - ma;
      const double db = b.picture[i] - mb;
      num += da * db;
      da2 += da * da;
      db2 += db * db;
    }
    // A picture with under one IRE of rms structure is empty ground.
    if (da2 < n || db2 < n) return;
    r.pictureCorr = num / (std::sqrt(da2) * std::sqrt(db2));
    r.pictureInfo = true;
  };

  auto corrEvidence = [&](double corr, bool informative) -> double {
    if (!informative) return 0.0;
    // corr <= 0.50 => 1.0, corr >= 0.90 => 0.0
    double e = (0.90 - corr) / (0.90 - 0.50);
    if (e < 0.0) e = 0.0;
    if (e > 1.0) e = 1.0;
    return e;
  };

  // Shadow-Boosted Delta Stats
  auto computeBoostedStats = [&](const FieldDescriptor& a,
                                 const FieldDescriptor& b) -> DeltaStats {
    DeltaStats ds;
    if (!a.valid || !b.valid) return ds;

    double diffSum = 0.0;
    double chromaSum = 0.0;
    double maxCellDiff = 0.0;
    int strongCount = 0;

    for (int k = 0; k < 9; ++k) {
      const double lumaDiff = std::abs(a.cells[k] - b.cells[k]);
      const double chromaDiff = std::abs(a.chromaEnergy[k] - b.chromaEnergy[k]);

      const double avgCellLuma = (a.cells[k] + b.cells[k]) * 0.5;
      double boost = 1.0;
      if (avgCellLuma < 35.0) {
        boost = 1.0 + (1.5 * (35.0 - avgCellLuma) / 35.0);
      }

      const double effectiveLumaDiff = lumaDiff * boost;
      const double effectiveChromaDiff =
          chromaDiff * ((boost - 1.0) * 0.5 + 1.0);

      diffSum += effectiveLumaDiff;
      chromaSum += effectiveChromaDiff;

      if (effectiveLumaDiff > maxCellDiff) maxCellDiff = effectiveLumaDiff;

      if (effectiveLumaDiff >= threshold * strongFactor)
        ++strongCount;
      else if (effectiveChromaDiff > threshold * 2.0)
        ++strongCount;
    }

    ds.total = diffSum;
    ds.peak = maxCellDiff;
    ds.strongCells = strongCount;
    ds.totalChroma = chromaSum;
    return ds;
  };

  // Transient. A flash makes two candidates: the departure, whose picture
  // comes back to the field before it within the hold window; and the
  // return, whose picture IS a coming-back to a field within the hold
  // window behind it. Either is a transient. A cut is neither: the picture
  // after it stays away from the picture before it for the whole hold.
  // "Back" is absolute, not a fraction of the candidate's departure: the
  // field before a cut can itself be a strobe frame, which inflates the
  // departure and makes a genuine cut look like a half-return (Civil
  // Defense 88278). The picture is back when it matches a recent field as
  // closely as continuous fields match — the detector's own change
  // threshold, the level below which it calls nothing a change.
  auto isTransient = [&](int idx) -> bool {
    if (idx - 1 < 1 || !safeValid(idx - 1) || !safeValid(idx)) return false;
    const FieldDescriptor& prev = getDesc(idx - 1);
    const FieldDescriptor& cur = getDesc(idx);
    const double d0 = computeBoostedStats(prev, cur).total;
    if (d0 <= 0.0) return false;
    const double back = std::min(TRANSIENT_RETURN_FRAC * d0, threshold);
    // The picture path only speaks of a departure: the measured flashes
    // leave by 12-26 IRE (Necessary Evil's lightning, Civil Defense's
    // strobes). A motion step of 9 IRE whose picture matches its neighbour
    // is continuity, not a return.
    const bool departed_ok = d0 >= 1.5 * threshold;
    // ref is the field the picture may have come back to. The level is
    // what ref matches inside its own shot at the same distance (dist
    // fields on the far side of the transient); and a return comes back
    // CLOSER to ref than the departed picture was — a motion seam's
    // picture also matches its neighbour at the shot's level, but it never
    // left, and it is not a return.
    auto isBack = [&](int ref, int dist, const FieldDescriptor& departedPic,
                      const FieldDescriptor& other) {
      const FieldDescriptor& a = getDesc(ref);
      if (computeBoostedStats(a, other).total < back) return true;
      if (!departed_ok) return false;
      CorrResult c = computeCorrelation(a, other);
      if (!c.informative || c.corr < TRANSIENT_RETURN_CORR) return false;
      pictureCorrelation(a, other, c);
      if (!c.pictureInfo) return false;
      const int own = (ref < idx) ? ref - dist : ref + dist;
      if (own < 1 || own > totalFields || !safeValid(own)) return false;
      CorrResult level, gone;
      pictureCorrelation(a, getDesc(own), level);
      pictureCorrelation(a, departedPic, gone);
      if (!level.pictureInfo || !gone.pictureInfo) return false;
      return c.pictureCorr >= level.pictureCorr - TRANSIENT_RETURN_SLACK &&
             c.pictureCorr >= gone.pictureCorr + TRANSIENT_RETURN_SLACK;
    };
    // Departure: does the picture ahead return to prev?
    for (int off = 1; off <= TRANSIENT_HOLD_FIELDS; ++off) {
      if (idx + off > totalFields || !safeValid(idx + off)) break;
      if (isBack(idx - 1, off + 1, cur, getDesc(idx + off))) return true;
    }
    // Return: is cur a coming-back to a picture behind prev?
    for (int off = 2; off <= TRANSIENT_HOLD_FIELDS + 1; ++off) {
      if (idx - off < 1 || !safeValid(idx - off)) break;
      if (isBack(idx - off, off, prev, cur)) return true;
    }
    return false;
  };

  // Every comparison in this detector is field to previous field. An edit
  // is between two adjacent fields, so that is the measure of it; and on
  // 3:2 film the pairs inside a film frame read zero, so each film-frame
  // seam is counted exactly once and a candidate is judged against the
  // OTHER seams around it. Camera motion recurs at every seam; a cut is
  // unique. Same-parity pairs (i-2, i) were fragile here: whether such a
  // pair spanned a seam or sat inside a 3-field frame depended only on the
  // candidate's slot in the 10-field cycle, so the same motion committed at
  // one cadence position and not the next (Emissary s1 3303-3500, one
  // shaky shot, 17 commits on the slots with a zero beside them and none
  // on its heaviest seams).
  // A step INTO a flat field — picture to white-out — is a lighting event,
  // not a picture event: a flash always begins inside a shot. It is neither
  // a candidate nor a competitor. Emissary 3937's white-out is not a cut
  // (the author's ruling), and 3860's real cut lost dominance to the
  // explosion white-out six fields after it, which then committed in its
  // place. A step OUT of a flat field keeps its lanes: a shot can end in
  // white (3805 is the cut out of the 3785-3804 explosion shot; 3941 heads
  // a shot out of a decayed white-out). A decay that returns to its own
  // shot is the transient veto's, below.
  auto stepAt = [&](int j) -> DeltaStats {
    if (j < 2 || !safeValid(j) || !safeValid(j - 1)) return DeltaStats();
    if (computeCorrelation(getDesc(j - 1), getDesc(j)).intoFlat)
      return DeltaStats();
    return computeBoostedStats(getDesc(j - 1), getDesc(j));
  };
  auto isSeam = [&](const DeltaStats& d) -> bool {
    return d.total > threshold;
  };

  // Ramp: the candidate and the two nearest seams on each side are all of
  // one size — a dissolve, or motion recurring seam after seam.
  auto isRampContext = [&](int idx) -> bool {
    std::vector<double> mags;
    mags.push_back(stepAt(idx).total);
    for (int dir = -1; dir <= 1; dir += 2) {
      int found = 0;
      for (int off = 1; off <= 2 * RAMP_HALF_WINDOW && found < 2; ++off) {
        const int j = idx + dir * off;
        if (j < 2 || j > totalFields) return false;
        if (!safeValid(j) || !safeValid(j - 1)) return false;
        const DeltaStats d = stepAt(j);
        if (!isSeam(d)) continue;
        mags.push_back(d.total);
        ++found;
      }
      if (found < 2) return false;
    }

    const double maxMag = *std::max_element(mags.begin(), mags.end());
    const double minMag = *std::min_element(mags.begin(), mags.end());

    if (maxMag < threshold * RAMP_TOTAL_MULT) return false;
    if (minMag <= 0.0) return false;

    const bool ramp = ((maxMag / minMag) < RAMP_RATIO_MAX);
    if (LOG_RAMP_VETO && ramp) {
      tbcDebugStream().nospace()
          << "EditDetector: ramp veto at field " << idx << " mags=[" << mags[0]
          << "," << mags[1] << "," << mags[2] << "," << mags[3] << ","
          << mags[4] << "]";
    }
    return ramp;
  };

  // Commit helper (edit boundary)
  auto commitBoundary = [&](int fieldIdx, const char* reason, double p90Ire,
                            const QString& domMode, const QString& detReason,
                            double corr, bool corrInfo, double eCorr,
                            double evidenceScore, const DeltaStats& ds,
                            double lPrev, double lCurr, int motionFrames,
                            int motionStrong) {
    if (fieldIdx < 1 || fieldIdx > totalFields) return;
    if (fieldIdx - lastEditFrame < 3) return;  // strict echo rejection

    auto field = md.getField(fieldIdx);
    if (field.cinemap.isEditBoundary) return;
    if (field.cinemap.isEditVetoed) return;  // user shut this edit down

    field.cinemap.assertEditBoundary();
    md.updateField(field, fieldIdx);
    editCount++;
    lastEditFrame = fieldIdx;

    if (LOG_COMMITS || LOG_CANDIDATES) {
      qInfo().nospace() << "EditDetector: EDIT"
                        << " atField=" << fieldIdx << " mode=" << domMode
                        << " reason=" << detReason << " tag=" << reason
                        << " evidence=" << evidenceScore << " corr=" << corr
                        << " corrInfo=" << corrInfo << " eCorr=" << eCorr
                        << " total=" << ds.total << " peak=" << ds.peak
                        << " strong=" << ds.strongCells
                        << " chroma=" << ds.totalChroma << " lPrev=" << lPrev
                        << " lCurr=" << lCurr << " p90=" << p90Ire
                        << " motionFrames=" << motionFrames
                        << " motionStrong=" << motionStrong;
    }
  };

  // Heartbeat for progress
  QElapsedTimer hb;
  hb.start();

  // 3. Main Streaming Loop
  for (int i = 5; i <= totalFields - 2; ++i) {
    if (LOG_PROGRESS && hb.elapsed() >= 1000) {
      const double pct = 100.0 * double(i) / double(totalFields);
      qInfo().noquote()
          << QString(
                 "Edit detection: field %1/%2 (%3%) — %4 edit(s) found so far")
                 .arg(i)
                 .arg(totalFields)
                 .arg(pct, 0, 'f', 1)
                 .arg(editCount);
      hb.restart();
    }

    // Demand Paging
    const int neededUpTo = std::min(
        totalFields, i + std::max(CONTEXT_HALF_SPAN,
                                  SIDE_REACH + TRANSIENT_HOLD_FIELDS));
    while (maxReadFrame < neededUpTo) {
      maxReadFrame++;
      LdDecodeMetaData::Field field = md.getField(maxReadFrame);
      FieldDescriptor d;
      if (field.pad) {
        d.valid = false;
      } else {
        SourceVideo::Data fieldData = sourceVideo.getVideoField(maxReadFrame);
        if (fieldData.size() > 0) {
          d = computeFieldDescriptor(
              reinterpret_cast<const uint16_t*>(fieldData.constData()),
              vp.fieldWidth, vp.fieldHeight, vp);
        } else {
          d.valid = false;
        }
      }
      descCache[maxReadFrame] = d;
    }

    // Cache Pruning
    // A competitor up to SIDE_REACH away is itself tested for transience,
    // and that test looks TRANSIENT_HOLD_FIELDS beyond it on both sides.
    const int pruneThreshold =
        i - (SIDE_REACH + TRANSIENT_HOLD_FIELDS + 2);
    auto cacheIt = descCache.begin();
    while (cacheIt != descCache.end() && cacheIt->first < pruneThreshold) {
      cacheIt = descCache.erase(cacheIt);
    }

    // Fast echo pre-check
    if (i - lastEditFrame < 2) continue;
    if (!safeValid(i) || !safeValid(i - 1)) continue;

    const FieldDescriptor& d_prev = getDesc(i - 1);
    const FieldDescriptor& d_curr = getDesc(i);

    // Robust blackness metric (p90 cell mean in IRE)
    const double p90Ire = getP90Luma(d_curr);

    // --- Black-span state machine (endpoints are BLACK fields) ---
    const bool blackNow = isBlackField(p90Ire, d_curr.peakIre);
    const bool notBlackNow = isNotBlackField(p90Ire, d_curr.peakIre);

    if (blackNow) {
      consecBlack++;
      consecNotBlack = 0;
    } else if (notBlackNow) {
      consecNotBlack++;
      consecBlack = 0;
    } else {
      consecBlack = std::max(0, consecBlack - 1);
      consecNotBlack = std::max(0, consecNotBlack - 1);
    }

    if (!inBlackRun && consecBlack >= BLACK_CONFIRM) {
      inBlackRun = true;
      // Backtrack to start of confirmed sequence.
      firstBlackField = std::max(1, i - (BLACK_CONFIRM - 1));
      lastBlackField = i;
      // Do NOT commit yet — wait for end of run to determine duration/logic.

      if (LOG_COMMITS) {
        qInfo().nospace() << "EditDetector: BLACK_RUN entered at field " << i
                          << " p90=" << p90Ire << " peak=" << d_curr.peakIre;
      }
    }

    if (inBlackRun) {
      if (blackNow) lastBlackField = i;

      if (consecNotBlack >= BLACK_CONFIRM) {
        // Run ended. Analyse extent.
        const int runLengthFields = lastBlackField - firstBlackField + 1;
        DeltaStats z;

        // Short run (<= BLACK_STANDOFF * 2 fields): single edit at midpoint.
        constexpr int BLACK_STANDOFF = 50;

        if (runLengthFields < (BLACK_STANDOFF * 2)) {
          // Nothing in an empty picture can say where inside it the cut falls,
          // so the midpoint is the honest answer. What matters is that the run
          // reaching here really is empty.
          const int midpoint = firstBlackField + runLengthFields / 2;
          commitBoundary(midpoint, "blackRunMid_short", 0.0, "black",
                         "shortRunNadir", 0.0, false, 0.0, 0.0, z, 0.0, 0.0, 0,
                         0);
        } else {
          // Long run: offset 15 frames in from each end to escape fades.
          int entryPoint = firstBlackField + BLACK_STANDOFF;
          int exitPoint = lastBlackField - BLACK_STANDOFF;

          // Safety clamp (shouldn't be needed given the check above).
          if (entryPoint > exitPoint) {
            entryPoint = firstBlackField + runLengthFields / 2;
            exitPoint = entryPoint;
          }

          commitBoundary(entryPoint, "blackRunEnter_offset", 0.0, "black",
                         "longRunStart+15fr", 0.0, false, 0.0, 0.0, z, 0.0, 0.0,
                         0, 0);

          if (exitPoint != entryPoint) {
            commitBoundary(exitPoint, "blackRunExit_offset", 0.0, "black",
                           "longRunEnd-15fr", 0.0, false, 0.0, 0.0, z, 0.0, 0.0,
                           0, 0);
          }
        }

        inBlackRun = false;
        firstBlackField = -1;
        lastBlackField = -1;
      }
      // Suppress normal detection logic while inside a black run.
      continue;
    }

    // Normal edit detection
    const bool dbg = (DBG_FIELD >= 0 && i == DBG_FIELD);

    const DeltaStats ds = computeBoostedStats(d_prev, d_curr);

    CorrResult cr = computeCorrelation(d_prev, d_curr);
    pictureCorrelation(d_prev, d_curr, cr);
    // Two competing correlations. The nine-cell Pearson is the general
    // one: it does not see motion, so at a matched false-alarm rate among
    // energetic continuous fields (2%) it catches 83% of cuts where the
    // picture Pearson catches 55%. The picture Pearson sees composition,
    // and so also motion; its power is the case the cells are blind to,
    // and it votes there in its own lane below.
    const double corr = cr.corr;
    const bool corrInfo = cr.informative;

    const double lumaPrev = getMaxLuma(d_prev);
    const double lumaCurr = getMaxLuma(d_curr);

    const double eCorr = corrEvidence(corr, corrInfo);

    // No contrast ahead: empty ground, the white-out twin of a black run.
    // Nothing structural can be said across it, and an energy change into
    // it is a flash or a fade, not a cut (a flash decay committed twice on
    // energy lanes alone once its correlation was rightly declared
    // uninformative; Emissary 3937's white-out is not a cut). A cut OUT of
    // a flash has contrast ahead and keeps its lanes (3805).
    if (cr.bothFlat || cr.intoFlat) {
      if (LOG_VERBOSE_REJECT || dbg) {
        qInfo().nospace() << "EditDetector: no-contrast reject at field " << i
                          << " p90=" << p90Ire;
      }
      continue;
    }

    // -------------------------------------------------------------------------
    // Detection lane weights.
    //
    // Each lane contributes to a running evidence score.  The candidate is
    // considered a change when the total score reaches
    // EVIDENCE_COMMIT_THRESHOLD (1.0).  Setting a lane weight to 0.0 suppresses
    // it entirely; raising it above 1.0 makes it decisive on its own.
    //
    // structuralBreak  — corr-weighted total-energy test.  Fires when spatial
    //   correlation between adjacent fields is low-to-moderate (corr < 0.85)
    //   AND total luma+chroma energy exceeds threshold scaled by how bad the
    //   correlation is.  The primary lane for clean hard cuts.
    //
    // strongCorrCut    — very low corr (< 0.60) + modest energy.  Catches cuts
    //   where even a small energy signal is convincing given near-zero temporal
    //   correlation.  Overlaps structuralBreak at the low end; kept separate
    //   so it can be weighted independently.
    //
    // strongCells3     — three or more of the nine 3×3 cells each exceed the
    //   strong-factor threshold independently.  Spatially distributed signal;
    //   hard to fake with noise.  Very reliable for full-frame cuts.
    //
    // strongCells2     — two strong cells AND total energy > 3.5× threshold.
    //   Less certain than three cells; the high-total guard reduces false
    //   positives on partial-frame changes (wipes, logos, etc.).
    //
    // strongCells1     — one strong cell AND peak > 2.5× threshold.  Catches
    //   hard local transitions (splices, title cards).  Most prone to false
    //   positives in high-motion content; reduce weight if oversensitive.
    //
    // hugeTotal        — total energy across all nine cells > 6× threshold,
    //   regardless of cell distribution.  Catches diffuse whole-frame changes
    //   (fade-outs, dissolves) that spread energy too thinly to trigger cell
    //   counts.  Can fire on sustained high-motion; the dominance and
    //   continuity vetoes downstream are the main guard.
    //
    // hugeChroma       — total chroma-energy change > 4× threshold.  Catches
    //   colour-only cuts (scene changes in animated material, title-card colour
    //   fields).  Most likely to produce false positives on colour-graded or
    //   heavily saturated sources; reduce or zero if problematic.
    // -------------------------------------------------------------------------
    struct LaneWeights {
      double structuralBreak = 1.0;
      double strongCorrCut = 0.8;
      double strongCells3 = 0.6;
      double strongCells2 = 0.4;
      double strongCells1 = 0.3;
      double hugeTotal = 0.7;
      double hugeChroma = 0.5;
    };
    constexpr LaneWeights W;
    constexpr double EVIDENCE_COMMIT_THRESHOLD = 1.0;

    double evidenceScore = 0.0;
    QString reason;

    // Lane: corr-weighted structural break
    const double structuralMult = 1.5 - 0.6 * eCorr;  // [0.9..1.5]
    if (W.structuralBreak > 0.0 && corrInfo && corr < 0.85) {
      if (ds.total > threshold * structuralMult) {
        evidenceScore += W.structuralBreak;
        if (reason.isEmpty()) reason = "structuralBreak_corrWeighted";
      }
    }

    // Lane: very low corr + modest energy
    // pictureBreak — the lighting layout is kept (cells >= 0.85) and the
    //   picture is not (picture Pearson < 0.70), with energy. A cut between
    //   two matched close-ups on one set: same key, same set, different
    //   composition (Emissary 46193: cells 0.94, picture 0.66, total 55;
    //   46033: 0.91 / 0.59 / 81). The cells cannot see it by construction —
    //   nine means of the same lighting — and no other lane reaches the
    //   line on it. Measured on 25,059 energetic continuous fields: 843
    //   carry this signature, most of them motion, and the dominance and
    //   continuity vetoes downstream adjudicate them as they do every lane.
    //   And a cut is an ISOLATED decorrelation: the picture is still before
    //   it and still after it. The four A/B cuts sit between neighbouring
    //   pairs at 0.995-0.997; the battle's motion, which reads the same
    //   0.5-0.7 on the cut pair, has a median neighbour of 0.71 and only 7
    //   of 80 such candidates still on both sides at 0.95.
    if (corrInfo && corr >= 0.85 && cr.pictureInfo && cr.pictureCorr < 0.70 &&
        ds.total > threshold) {
      // Still on both sides: every adjacent pair within three fields, which
      // on 3:2 always reaches the neighbouring seam.
      bool stillAround = true;
      for (int j = i - 3; j <= i + 3 && stillAround; ++j) {
        if (j == i) continue;
        if (j < 2 || j > totalFields || !safeValid(j) || !safeValid(j - 1)) {
          stillAround = false;
          break;
        }
        CorrResult n;
        pictureCorrelation(getDesc(j - 1), getDesc(j), n);
        if (!n.pictureInfo || n.pictureCorr < 0.95) stillAround = false;
      }
      if (stillAround) {
        evidenceScore += 1.0;
        if (reason.isEmpty()) reason = "pictureBreak";
      }
    }

    if (W.strongCorrCut > 0.0 && corrInfo && corr < 0.60) {
      if (ds.total > threshold * 1.0) {
        evidenceScore += W.strongCorrCut;
        if (reason.isEmpty()) reason = "strongCorrCut";
      }
    }

    // Lane: three or more spatially distributed strong cells
    if (W.strongCells3 > 0.0 && ds.strongCells >= 3) {
      evidenceScore += W.strongCells3;
      if (reason.isEmpty()) reason = "strongCells>=3";
    }

    // Lane: two strong cells with high total energy
    if (W.strongCells2 > 0.0 && ds.strongCells == 2 &&
        ds.total > threshold * 3.5) {
      evidenceScore += W.strongCells2;
      if (reason.isEmpty()) reason = "2strongCells_highTotal";
    }

    // Lane: one strong cell with a very high peak
    if (W.strongCells1 > 0.0 && ds.strongCells == 1 &&
        ds.peak > threshold * 2.5) {
      evidenceScore += W.strongCells1;
      if (reason.isEmpty()) reason = "1strongCell_highPeak";
    }

    // The diffuse-energy lanes do not vote against an informative correlation
    // above the structural lane's own continuity line: a whole-frame change
    // on a picture whose light and dark regions stay where they are is a
    // flash, a fade or a grade, not a cut (a flash decay committed at corr
    // 0.886 on a sixth of a real cut's energy). Cuts still commit on cell
    // structure and on correlation collapse. Three or more strong cells is
    // distributed structure and keeps its vote; the block applies where the
    // diffuse lanes would carry the commit on their own.
    //
    // The continuity this reads is the PICTURE's, not the nine cells': a
    // flash keeps the picture (structure survives, only the level moves;
    // measured p50 0.80 on 136 flashes) and a cut between two matched
    // close-ups on one set loses it (46193: cells 0.94, picture 0.65) while
    // keeping the nine cells' lighting layout. Tried on the cells alone
    // with the transient veto standing in: 41 strobe commits admitted and
    // 46193 still short. On the picture the cut population reads p50 0.59
    // / p90 0.84 and the flash population p50 0.80 / p10 0.46; the line at
    // 0.75 sits between their medians, and a flash below it that reaches
    // the commit line is the transient veto's to catch by time.
    const bool continuousByCorr =
        corrInfo && corr >= 0.85 && ds.strongCells <= 1;

    // Lane: very high total energy (diffuse whole-frame change)
    if (W.hugeTotal > 0.0 && !continuousByCorr && ds.total > threshold * 6.0) {
      evidenceScore += W.hugeTotal;
      if (reason.isEmpty()) reason = "hugeTotal";
    }

    // Lane: very high chroma energy change (colour-only cut)
    if (W.hugeChroma > 0.0 && !continuousByCorr &&
        ds.totalChroma > threshold * 4.0) {
      evidenceScore += W.hugeChroma;
      if (reason.isEmpty()) reason = "hugeChroma";
    }

    const bool isChange = (evidenceScore >= EVIDENCE_COMMIT_THRESHOLD);

    if (!isChange) {
      if (LOG_VERBOSE_REJECT || dbg) {
        qInfo().nospace() << "DBG" << DBG_FIELD << " reject i=" << i
                          << " corr=" << corr << " corrInfo=" << corrInfo
                          << " eCorr=" << eCorr << " total=" << ds.total
                          << " peak=" << ds.peak << " strong=" << ds.strongCells
                          << " chroma=" << ds.totalChroma << " p90=" << p90Ire
                          << " pcorr=" << cr.pictureCorr
                          << " pinfo=" << cr.pictureInfo;
      }
      continue;
    }

    // ---- Context & Motion Analysis ----
    // Sides: the other steps within three fields (always reaches the
    // neighbouring seam on 3:2). Context: within CONTEXT_HALF_SPAN. Only
    // steps that are steps count as motion; the pairs inside a film frame
    // are silent and say nothing about motion either way.
    double sidePeak = 0.0;
    double sideTotal = 0.0;
    double sideChroma = 0.0;
    double sideCorrMin = 2.0;  // lowest informative correlation among competitors

    const int reach = std::max(CONTEXT_HALF_SPAN, SIDE_REACH);
    const int ctxStart = std::max(2, i - reach);
    const int ctxEnd = std::min(totalFields, i + reach);

    int motionFrames = 0;
    int motionStrong = 0;
    double ctxTotalMax = 0.0;

    // A flash's edges are not picture events and do not compete: the head
    // of Emissary's shaky shot (6395, 69 IRE at corr 0.52) lost dominance
    // to a strobe nine fields before it that the transient veto had
    // already thrown out. Unless the candidate is INSIDE the flash: a
    // strobe train's edges each return and are all transients, and the
    // one edge whose picture drifted too far to return is not a cut for
    // being the last one standing (Civil Defense 89129). Transients on
    // both sides of the candidate within reach mean the train is the
    // candidate's own event, and then it competes.
    bool transientBefore = false, transientAfter = false;
    std::vector<int> transientAt;
    for (int j = std::max(2, i - SIDE_REACH); j <= std::min(totalFields, i + SIDE_REACH); ++j) {
      if (j == i || !safeValid(j) || !safeValid(j - 1)) continue;
      if (!isSeam(stepAt(j)) || !isTransient(j)) continue;
      transientAt.push_back(j);
      (j < i ? transientBefore : transientAfter) = true;
    }
    const bool insideFlash = transientBefore && transientAfter;

    for (int j = ctxStart; j <= ctxEnd; ++j) {
      if (j == i) continue;  // exclude candidate itself
      if (!safeValid(j) || !safeValid(j - 1)) continue;
      const DeltaStats dj = stepAt(j);
      if (!insideFlash &&
          std::find(transientAt.begin(), transientAt.end(), j) != transientAt.end())
        continue;
      if (std::abs(j - i) <= SIDE_REACH) {
        sidePeak = std::max(sidePeak, dj.peak);
        sideTotal = std::max(sideTotal, dj.total);
        sideChroma = std::max(sideChroma, dj.totalChroma);
        if (isSeam(dj)) {
          const CorrResult cj = computeCorrelation(getDesc(j - 1), getDesc(j));
          if (cj.informative) sideCorrMin = std::min(sideCorrMin, cj.corr);
        }
      }
      if (!isSeam(dj) || std::abs(j - i) > CONTEXT_HALF_SPAN) continue;
      motionFrames++;
      if (dj.strongCells >= 2) motionStrong++;
      ctxTotalMax = std::max(ctxTotalMax, dj.total);
    }

    bool dynamicMotionContext = false;
    if (motionFrames >= 4) {
      const double fracStrong =
          static_cast<double>(motionStrong) / static_cast<double>(motionFrames);
      if (fracStrong >= MOTION_STRONG_CELL_FRAC) dynamicMotionContext = true;
    }

    // Dominance & veto
    bool dominant = false;
    QString domMode;

    if (dynamicMotionContext) {
      domMode = "motion";
      const double motionThreshold = std::max(ctxTotalMax, sideTotal) * 1.3;
      dominant = (ds.total > motionThreshold);
    } else {
      domMode = "normal";
      const double effectivePeakFactor = (peakFactor > 1.0) ? peakFactor : 1.5;
      if (corrInfo && corr < 0.7) {
        dominant =
            (ds.total >= sideTotal * 1.1) || (ds.peak >= sidePeak * 1.15);
      } else {
        dominant = (ds.total >= sideTotal * 1.3) ||
                   (ds.peak >= sidePeak * effectivePeakFactor) ||
                   (ds.totalChroma >= sideChroma * 1.4);
      }
    }

    // Dominance by correlation. The test is never size: a cut is the
    // smallest of three events inside four fields at Emissary 3764 — the
    // frame line steps 138 IRE at corr -0.06, the explosion already going
    // behind the captain steps 186 at 0.56 one field on, and its flash 370
    // at 0.85 three fields on. By energy the cut can never dominate; by
    // correlation it is the only break there is. An ANTI-correlated
    // candidate — the nine-cell light layout inverted, which no camera
    // move inside a shot produces (dark whiplash reads 0.16, a pyrotechnic
    // igniting 0.24) — is dominant when it undercuts every competitor in
    // reach by DOMINANCE_CORR_MARGIN.
    constexpr double DOMINANCE_CORR_MARGIN = 0.3;
    if (!dominant && corrInfo && corr < 0.0 && sideCorrMin <= 1.0 &&
        corr <= sideCorrMin - DOMINANCE_CORR_MARGIN) {
      dominant = true;
      domMode = "correlation";
    }

    const bool ramp = isRampContext(i);
    const bool echoReject = ((i - lastEditFrame) < 3);

    if (dbg) {
      qInfo().nospace() << "DBG" << DBG_FIELD
                        << " v[-1,i]=" << safeValid(i - 1) << ","
                        << safeValid(i) << " p90=" << p90Ire
                        << " isChange=" << isChange << " reason=" << reason
                        << " corr=" << corr << " corrInfo=" << corrInfo
                        << " eCorr=" << eCorr
                        << " ds(total,peak,strong,chroma)=" << ds.total << ","
                        << ds.peak << "," << ds.strongCells << ","
                        << ds.totalChroma << " sideTotal=" << sideTotal
                        << " ctxTotalMax=" << ctxTotalMax
                        << " motion=" << motionStrong << "/" << motionFrames
                        << " dynMotion=" << dynamicMotionContext
                        << " domMode=" << domMode << " dominant=" << dominant
                        << " ramp=" << ramp << " echoReject=" << echoReject
                        << " lastEdit=" << lastEditFrame
                        << " lPrev=" << lumaPrev << " lCurr=" << lumaCurr;
    }

    if (!dominant) continue;

    if (ramp) {
      if (LOG_RAMP_VETO && (LOG_VERBOSE_REJECT || LOG_CANDIDATES || dbg)) {
        qInfo().nospace() << "EditDetector: ramp veto commit@field " << i
                          << " reason=" << reason;
      }
      continue;
    }

    if (isTransient(i)) {
      if (LOG_COMMITS || LOG_CANDIDATES || dbg) {
        qInfo().nospace() << "EditDetector: transient veto at field " << i
                          << " reason=" << reason << " total=" << ds.total
                          << " corr=" << corr << " pcorr=" << cr.pictureCorr;
      }
      continue;
    }

    // --- CONTINUITY VETO ---
    // 1) Very-high-corr, chroma-only "edits" (grading changes, small luma
    // shifts) 2) Motion-context spikes that are not strong enough outliers vs
    // neighbors
    bool continuityVeto = false;

    // 1) Chroma-only / grading-change veto
    if (!continuityVeto && corrInfo && corr > 0.93 && ds.strongCells == 0) {
      // In a clearly continuous shot (high corr, no strong-edge cells),
      // hugeChroma/hugeTotal alone shouldn't create a hard edit boundary.
      if (motionFrames >= 6) {
        continuityVeto = true;
      }
    }

    // 2) Motion-context soft-spike veto
    if (!continuityVeto && dynamicMotionContext) {
      // In sustained motion, let only very strong structure changes through.
      const double localMax = std::max(sideTotal, ctxTotalMax);
      const bool softSpike =
          (ds.total < localMax * 1.4) && (ds.strongCells <= 1);
      const bool corrNotTerrible = corrInfo && corr > 0.6;
      if (softSpike && corrNotTerrible) {
        continuityVeto = true;
      }
    }

    if (continuityVeto) {
      if (LOG_VERBOSE_REJECT || LOG_CANDIDATES) {
        qInfo().nospace() << "EditDetector: continuity veto at field " << i
                          << " reason=" << reason << " corr=" << corr
                          << " strong=" << ds.strongCells
                          << " total=" << ds.total << " sideTotal=" << sideTotal
                          << " ctxTotalMax=" << ctxTotalMax
                          << " motionFrames=" << motionFrames
                          << " motionStrong=" << motionStrong
                          << " p90=" << p90Ire;
      }
      continue;
    }

    // Commit (field-aligned; ld-analyse cuts don't split fields)
    const int targetField = i;
    if (targetField - lastEditFrame < 3) continue;  // strict echo

    commitBoundary(targetField, "visual", p90Ire, domMode, reason, corr,
                   corrInfo, eCorr, evidenceScore, ds, lumaPrev, lumaCurr,
                   motionFrames, motionStrong);
    if (LOG_COMMITS || LOG_CANDIDATES) {
      qInfo().nospace() << "EditDetector: PCORR atField=" << targetField
                        << " cells=" << cr.corr << " picture=" << cr.pictureCorr
                        << " pictureInfo=" << cr.pictureInfo;
    }

  }

  sourceVideo.close();
  return editCount;
}

}  // namespace visualEdits
