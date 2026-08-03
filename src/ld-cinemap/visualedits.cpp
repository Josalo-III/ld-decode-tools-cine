// tools/ld-cinemap/visualedits.cpp
/******************************************************************************
 * visualedits.cpp
 * ld-cinemap — telecine cadence solver and edit detection
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 Joseph Burns
 *
 * This file is part of ld-decode-tools.
 ******************************************************************************/


#include "visualedits.h"
#include "cinedisc.h"

#include "lddecodemetadata.h"
#include "sourcevideo.h"
#include "tbc/logging.h"

#include <QElapsedTimer>
#include <QString>

#include <algorithm>
#include <cmath>
#include <map>

namespace visualEdits {

    struct FieldDescriptor {
        bool   valid = false;
        double cells[9]        = {0.0}; // Luma (DC)
        double chromaEnergy[9] = {0.0}; // Chroma Magnitude (approx)
    };
    
    struct DeltaStats {
        double peak        = 0.0;
        double total       = 0.0;
        int    strongCells = 0;
        double totalChroma = 0.0; // Total change in chroma energy
    };
    // Compute a 3×3 descriptor over title-safe area (10–90%), in IRE.
    FieldDescriptor computeFieldDescriptor(const uint16_t* fieldData,
                                           int width,
                                           int height,
                                           const LdDecodeMetaData::VideoParameters& vp)
    {
        FieldDescriptor desc;

        if (!fieldData || width <= 0 || height <= 0) {
            desc.valid = false;
            std::fill(std::begin(desc.cells), std::end(desc.cells), 0.0);
            return desc;
        }

        // Same IRE normalization as the old 38 quad descriptor
        const double black = (vp.black16bIre >= 0) ? vp.black16bIre : 0.0;
        const double white = (vp.white16bIre > vp.black16bIre) ? vp.white16bIre : 65535.0;
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
        const int x0 = static_cast<int>(width  * 0.10);
        const int x1 = static_cast<int>(width  * 0.90);
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
            int yStart = y0 + ((y1 - y0) * row)     / 3;
            int yEnd   = y0 + ((y1 - y0) * (row+1)) / 3;

            for (int col = 0; col < 3; ++col) {
                int xStart = x0 + ((x1 - x0) * col)     / 3;
                int xEnd   = x0 + ((x1 - x0) * (col+1)) / 3;

                double sumIre    = 0.0;
                double sumAbsDiff = 0.0;
                int    pixelCount = 0;

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

        return desc;
    }

    int analyseVisualEdits(CineDisc& disc,
                           double threshold,
                           double strongFactor,
                           double peakFactor,
                           bool   traceEnabled)
    {
        auto& md = disc.getMetaData();

        // Logging controls. The noisy per-edit/per-candidate detail is gated by
        // traceEnabled (CLI flag --edit-trace). Compile-time consts below opt-in
        // to additional verbosity for source-level debugging.
        const     bool LOG_COMMITS        = traceEnabled; // one line per committed edit
        constexpr bool LOG_CANDIDATES     = false;        // every candidate that passes isChange
        constexpr bool LOG_VERBOSE_REJECT = false;        // why we rejected a candidate
        constexpr bool LOG_RAMP_VETO      = false;        // when ramp veto fires
        const     bool LOG_PROGRESS       = true;          // heartbeat/progress line (always on)
    
        // Debug target: set to a field index to enable per-field trace; -1 = off.
        constexpr int DBG_FIELD = -1;
    
        // 1. Clamp Thresholds
        if (threshold > 8.0) threshold = 8.0;
        if (threshold < 0.8) threshold = 0.8;
        if (strongFactor < 1.0) strongFactor = 1.0;
        if (peakFactor   < 1.0) peakFactor   = 1.0;
    
        SourceVideo sourceVideo;
        if (!sourceVideo.open(disc.getTbcPath(), disc.getVideoFieldLength())) return 0;
    
        const int totalFields = md.getNumberOfFields();
        if (totalFields < 2) {
            sourceVideo.close();
            return 0;
        }
    
        const auto vp = md.getVideoParameters();
    
        // 2. Sliding Window Cache
        std::map<int, FieldDescriptor> descCache;
    
        int editCount     = 0;
        int lastEditFrame = -100;
        int maxReadFrame  = 0;
    
        // Tunables
        const double RAMP_TOTAL_MULT         = 4.0;
        const double RAMP_RATIO_MAX          = 1.3;
        const int    RAMP_HALF_WINDOW        = 4;
        const int    CONTEXT_HALF_SPAN       = 6;
        const double MOTION_STRONG_CELL_FRAC = 0.40;
    
        // Black-span segmentation (IRE; endpoints must be black themselves)
        const double BLACK_ENTER_IRE = 2.4; // "definitely black"
        const double BLACK_EXIT_IRE  = 2.8; // "definitely not black" (hysteresis)
        const int    BLACK_CONFIRM   = 12;  // consecutive black/not-black confirmations
    
        bool inBlackRun     = false;
        int  firstBlackField = -1;
        int  lastBlackField  = -1;
        int  consecBlack     = 0;
        int  consecNotBlack  = 0;
    
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
    
        // Helper: Max Luma (legacy; for logs)
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
    
        auto isBlackField    = [&](double p90Ire) -> bool { return p90Ire <= BLACK_ENTER_IRE; };
        auto isNotBlackField = [&](double p90Ire) -> bool { return p90Ire >= BLACK_EXIT_IRE;  };
    
        // Correlation (Pearson on 3x3 mean luma)
        struct CorrResult { double corr = 0.0; bool informative = false; };
    
        auto computeCorrelation = [&](const FieldDescriptor& a, const FieldDescriptor& b) -> CorrResult {
            CorrResult r;
            if (!a.valid || !b.valid) return r;
    
            double meanA = 0.0, meanB = 0.0;
            for (int i = 0; i < 9; ++i) { meanA += a.cells[i]; meanB += b.cells[i]; }
            meanA /= 9.0; meanB /= 9.0;
    
            double num = 0.0, denA = 0.0, denB = 0.0;
            for (int i = 0; i < 9; ++i) {
                const double da = a.cells[i] - meanA;
                const double db = b.cells[i] - meanB;
                num  += da * db;
                denA += da * da;
                denB += db * db;
            }
    
            // flat => corr not informative (do NOT treat as "strong low corr evidence")
            if (denA < 0.1 || denB < 0.1) return r;
    
            r.corr = num / (std::sqrt(denA) * std::sqrt(denB));
            r.informative = true;
            return r;
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
        auto computeBoostedStats = [&](const FieldDescriptor& a, const FieldDescriptor& b) -> DeltaStats {
            DeltaStats ds;
            if (!a.valid || !b.valid) return ds;
    
            double diffSum     = 0.0;
            double chromaSum   = 0.0;
            double maxCellDiff = 0.0;
            int    strongCount = 0;
    
            for (int k = 0; k < 9; ++k) {
                const double lumaDiff   = std::abs(a.cells[k]        - b.cells[k]);
                const double chromaDiff = std::abs(a.chromaEnergy[k] - b.chromaEnergy[k]);
    
                const double avgCellLuma = (a.cells[k] + b.cells[k]) * 0.5;
                double boost = 1.0;
                if (avgCellLuma < 35.0) {
                    boost = 1.0 + (1.5 * (35.0 - avgCellLuma) / 35.0);
                }
    
                const double effectiveLumaDiff   = lumaDiff   * boost;
                const double effectiveChromaDiff = chromaDiff * ((boost - 1.0) * 0.5 + 1.0);
    
                diffSum   += effectiveLumaDiff;
                chromaSum += effectiveChromaDiff;
    
                if (effectiveLumaDiff > maxCellDiff) maxCellDiff = effectiveLumaDiff;
    
                if (effectiveLumaDiff >= threshold * strongFactor) ++strongCount;
                else if (effectiveChromaDiff > threshold * 2.0)    ++strongCount;
            }
    
            ds.total       = diffSum;
            ds.peak        = maxCellDiff;
            ds.strongCells = strongCount;
            ds.totalChroma = chromaSum;
            return ds;
        };
    
        auto isRampContext = [&](int idx) -> bool {
            if (idx - 2 * RAMP_HALF_WINDOW < 1 || idx + 2 * RAMP_HALF_WINDOW > totalFields) return false;
            for (int off = -2 * RAMP_HALF_WINDOW; off <= 2 * RAMP_HALF_WINDOW; off += 2) {
                if (!safeValid(idx + off)) return false;
            }
    
            DeltaStats d1 = computeBoostedStats(getDesc(idx - 4), getDesc(idx - 2));
            DeltaStats d2 = computeBoostedStats(getDesc(idx - 2), getDesc(idx));
            DeltaStats d3 = computeBoostedStats(getDesc(idx),     getDesc(idx + 2));
            DeltaStats d4 = computeBoostedStats(getDesc(idx + 2), getDesc(idx + 4));
    
            const double mags[4] = { d1.total, d2.total, d3.total, d4.total };
            const double maxMag = std::max(std::max(mags[0], mags[1]), std::max(mags[2], mags[3]));
            const double minMag = std::min(std::min(mags[0], mags[1]), std::min(mags[2], mags[3]));
    
            if (maxMag < threshold * RAMP_TOTAL_MULT) return false;
            if (minMag <= 0.0) return false;
    
            const bool ramp = ((maxMag / minMag) < RAMP_RATIO_MAX);
            if (LOG_RAMP_VETO && ramp) {
                tbcDebugStream().nospace() << "EditDetector: ramp veto at field "
                                   << idx << " mags=[" << mags[0] << "," << mags[1]
                                   << "," << mags[2] << "," << mags[3] << "]";
            }
            return ramp;
        };
    
        // Commit helper (edit boundary)
        auto commitBoundary = [&](int fieldIdx, const char* reason, double p90Ire,
                                  const QString& domMode,
                                  const QString& detReason,
                                  double corr, bool corrInfo, double eCorr,
                                  double evidenceScore,
                                  const DeltaStats& ds,
                                  double lPrev, double lCurr,
                                  int motionFrames, int motionStrong)
        {
            if (fieldIdx < 1 || fieldIdx > totalFields) return;
            if (fieldIdx - lastEditFrame < 3) return; // strict echo rejection
    
            auto field = md.getField(fieldIdx);
            if (field.cinemap.isEditBoundary) return;
            if (field.cinemap.isEditVetoed) return; // user shut this edit down

            field.cinemap.assertEditBoundary();
            md.updateField(field, fieldIdx);
            editCount++;
            lastEditFrame = fieldIdx;
    
            if (LOG_COMMITS || LOG_CANDIDATES) {
                qInfo().nospace()
                    << "EditDetector: EDIT"
                    << " atField="      << fieldIdx
                    << " mode="         << domMode
                    << " reason="       << detReason
                    << " tag="          << reason
                    << " evidence="     << evidenceScore
                    << " corr="         << corr
                    << " corrInfo="     << corrInfo
                    << " eCorr="        << eCorr
                    << " total="        << ds.total
                    << " peak="         << ds.peak
                    << " strong="       << ds.strongCells
                    << " chroma="       << ds.totalChroma
                    << " lPrev="        << lPrev
                    << " lCurr="        << lCurr
                    << " p90="          << p90Ire
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
                    << QString("Edit detection: field %1/%2 (%3%) — %4 edit(s) found so far")
                           .arg(i).arg(totalFields).arg(pct, 0, 'f', 1)
                           .arg(editCount);
                hb.restart();
            }
    
            // Demand Paging
            const int neededUpTo = std::min(totalFields, i + CONTEXT_HALF_SPAN);
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
                            vp.fieldWidth, vp.fieldHeight, vp
                        );
                    } else {
                        d.valid = false;
                    }
                }
                descCache[maxReadFrame] = d;
            }
    
            // Cache Pruning
            const int pruneThreshold = i - 12;
            auto cacheIt = descCache.begin();
            while (cacheIt != descCache.end() && cacheIt->first < pruneThreshold) {
                cacheIt = descCache.erase(cacheIt);
            }
    
            // Fast echo pre-check
            if (i - lastEditFrame < 2) continue;
            if (!safeValid(i) || !safeValid(i - 2)) continue;
    
            const FieldDescriptor& d_prev = getDesc(i - 2);
            const FieldDescriptor& d_curr = getDesc(i);
    
            // Robust blackness metric (p90 cell mean in IRE)
            const double p90Ire = getP90Luma(d_curr);
    
            // --- Black-span state machine (endpoints are BLACK fields) ---
            const bool blackNow    = isBlackField(p90Ire);
            const bool notBlackNow = isNotBlackField(p90Ire);
    
            if (blackNow) { consecBlack++; consecNotBlack = 0; }
            else if (notBlackNow) { consecNotBlack++; consecBlack = 0; }
            else {
                consecBlack    = std::max(0, consecBlack    - 1);
                consecNotBlack = std::max(0, consecNotBlack - 1);
            }
    
            if (!inBlackRun && consecBlack >= BLACK_CONFIRM) {
                inBlackRun = true;
                // Backtrack to start of confirmed sequence.
                firstBlackField = std::max(1, i - (BLACK_CONFIRM - 1));
                lastBlackField  = i;
                // Do NOT commit yet — wait for end of run to determine duration/logic.
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
                        const int midpoint = firstBlackField + runLengthFields / 2;
                        commitBoundary(midpoint, "blackRunMid_short", 0.0,
                                       "black", "shortRunNadir",
                                       0.0, false, 0.0, 0.0, z, 0.0, 0.0, 0, 0);
                    } else {
                        // Long run: offset 15 frames in from each end to escape fades.
                        int entryPoint = firstBlackField + BLACK_STANDOFF;
                        int exitPoint  = lastBlackField  - BLACK_STANDOFF;
    
                        // Safety clamp (shouldn't be needed given the check above).
                        if (entryPoint > exitPoint) {
                            entryPoint = firstBlackField + runLengthFields / 2;
                            exitPoint  = entryPoint;
                        }
    
                        commitBoundary(entryPoint, "blackRunEnter_offset", 0.0,
                                       "black", "longRunStart+15fr",
                                       0.0, false, 0.0, 0.0, z, 0.0, 0.0, 0, 0);
    
                        if (exitPoint != entryPoint) {
                            commitBoundary(exitPoint, "blackRunExit_offset", 0.0,
                                           "black", "longRunEnd-15fr",
                                           0.0, false, 0.0, 0.0, z, 0.0, 0.0, 0, 0);
                        }
                    }
    
                    inBlackRun      = false;
                    firstBlackField = -1;
                    lastBlackField  = -1;
                }
                // Suppress normal detection logic while inside a black run.
                continue;
            }
    
            // Normal edit detection
            const bool dbg = (DBG_FIELD >= 0 && i == DBG_FIELD);
    
            const DeltaStats ds = computeBoostedStats(d_prev, d_curr);
    
            const CorrResult cr       = computeCorrelation(d_prev, d_curr);
            const double     corr     = cr.corr;
            const bool       corrInfo = cr.informative;
    
            const double lumaPrev = getMaxLuma(d_prev);
            const double lumaCurr = getMaxLuma(d_curr);
    
            const double eCorr = corrEvidence(corr, corrInfo);
    
            // -------------------------------------------------------------------------
            // Detection lane weights.
            //
            // Each lane contributes to a running evidence score.  The candidate is
            // considered a change when the total score reaches EVIDENCE_COMMIT_THRESHOLD
            // (1.0).  Setting a lane weight to 0.0 suppresses it entirely; raising it
            // above 1.0 makes it decisive on its own.
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
            //   counts.  Can fire on sustained high-motion; the dominance and continuity
            //   vetoes downstream are the main guard.
            //
            // hugeChroma       — total chroma-energy change > 4× threshold.  Catches
            //   colour-only cuts (scene changes in animated material, title-card colour
            //   fields).  Most likely to produce false positives on colour-graded or
            //   heavily saturated sources; reduce or zero if problematic.
            // -------------------------------------------------------------------------
            struct LaneWeights {
                double structuralBreak = 1.0;
                double strongCorrCut   = 0.8;
                double strongCells3    = 0.6;
                double strongCells2    = 0.4;
                double strongCells1    = 0.3;
                double hugeTotal       = 0.7;
                double hugeChroma      = 0.5;
            };
            constexpr LaneWeights W;
            constexpr double EVIDENCE_COMMIT_THRESHOLD = 1.0;
    
            double  evidenceScore = 0.0;
            QString reason;
    
            // Lane: corr-weighted structural break
            const double structuralMult = 1.5 - 0.6 * eCorr; // [0.9..1.5]
            if (W.structuralBreak > 0.0 && corrInfo && corr < 0.85) {
                if (ds.total > threshold * structuralMult) {
                    evidenceScore += W.structuralBreak;
                    if (reason.isEmpty()) reason = "structuralBreak_corrWeighted";
                }
            }
    
            // Lane: very low corr + modest energy
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
            if (W.strongCells2 > 0.0 && ds.strongCells == 2 && ds.total > threshold * 3.5) {
                evidenceScore += W.strongCells2;
                if (reason.isEmpty()) reason = "2strongCells_highTotal";
            }
    
            // Lane: one strong cell with a very high peak
            if (W.strongCells1 > 0.0 && ds.strongCells == 1 && ds.peak > threshold * 2.5) {
                evidenceScore += W.strongCells1;
                if (reason.isEmpty()) reason = "1strongCell_highPeak";
            }
    
            // Lane: very high total energy (diffuse whole-frame change)
            if (W.hugeTotal > 0.0 && ds.total > threshold * 6.0) {
                evidenceScore += W.hugeTotal;
                if (reason.isEmpty()) reason = "hugeTotal";
            }
    
            // Lane: very high chroma energy change (colour-only cut)
            if (W.hugeChroma > 0.0 && ds.totalChroma > threshold * 4.0) {
                evidenceScore += W.hugeChroma;
                if (reason.isEmpty()) reason = "hugeChroma";
            }
    
            const bool isChange = (evidenceScore >= EVIDENCE_COMMIT_THRESHOLD);
    
            if (!isChange) {
                if (LOG_VERBOSE_REJECT || dbg) {
                    qInfo().nospace()
                        << "DBG" << DBG_FIELD
                        << " reject i="     << i
                        << " corr="         << corr
                        << " corrInfo="     << corrInfo
                        << " eCorr="        << eCorr
                        << " total="        << ds.total
                        << " peak="         << ds.peak
                        << " strong="       << ds.strongCells
                        << " chroma="       << ds.totalChroma
                        << " p90="          << p90Ire;
                }
                continue;
            }
    
            // ---- Context & Motion Analysis ----
            const DeltaStats sideBefore = computeBoostedStats(getDesc(i - 4), getDesc(i - 2));
            const DeltaStats sideAfter  = computeBoostedStats(getDesc(i),     getDesc(i + 2));
    
            const double sidePeak   = std::max(sideBefore.peak,        sideAfter.peak);
            const double sideTotal  = std::max(sideBefore.total,       sideAfter.total);
            const double sideChroma = std::max(sideBefore.totalChroma, sideAfter.totalChroma);
    
            const int ctxStart = std::max(5,               i - CONTEXT_HALF_SPAN);
            const int ctxEnd   = std::min(totalFields - 2, i + CONTEXT_HALF_SPAN);
    
            int    motionFrames = 0;
            int    motionStrong = 0;
            double ctxTotalMax  = 0.0;
    
            for (int j = ctxStart; j <= ctxEnd; ++j) {
                if (j == i) continue; // exclude candidate itself
                if (!safeValid(j) || !safeValid(j - 2)) continue;
                const DeltaStats dj = computeBoostedStats(getDesc(j - 2), getDesc(j));
                motionFrames++;
                if (dj.strongCells >= 2) motionStrong++;
                ctxTotalMax = std::max(ctxTotalMax, dj.total);
            }
    
            bool dynamicMotionContext = false;
            if (motionFrames >= 4) {
                const double fracStrong = static_cast<double>(motionStrong)
                                        / static_cast<double>(motionFrames);
                if (fracStrong >= MOTION_STRONG_CELL_FRAC) dynamicMotionContext = true;
            }
    
            // Dominance & veto
            bool    dominant = false;
            QString domMode;
    
            if (dynamicMotionContext) {
                domMode = "motion";
                const double motionThreshold = std::max(ctxTotalMax, sideTotal) * 1.2;
                dominant = (ds.total > motionThreshold);
            } else {
                domMode = "normal";
                const double effectivePeakFactor = (peakFactor > 1.0) ? peakFactor : 1.5;
                if (corrInfo && corr < 0.7) {
                    dominant = (ds.total >= sideTotal * 1.1) ||
                               (ds.peak  >= sidePeak  * 1.15);
                } else {
                    dominant = (ds.total       >= sideTotal  * 1.3)               ||
                               (ds.peak        >= sidePeak   * effectivePeakFactor) ||
                               (ds.totalChroma >= sideChroma * 1.4);
                }
            }
    
            const bool ramp       = isRampContext(i);
            const bool echoReject = ((i - lastEditFrame) < 3);
    
            if (dbg) {
                qInfo().nospace()
                    << "DBG" << DBG_FIELD
                    << " v[-4,-2,i,+2]=" << safeValid(i-4) << "," << safeValid(i-2)
                                         << "," << safeValid(i) << "," << safeValid(i+2)
                    << " p90="       << p90Ire
                    << " isChange="  << isChange
                    << " reason="    << reason
                    << " corr="      << corr
                    << " corrInfo="  << corrInfo
                    << " eCorr="     << eCorr
                    << " ds(total,peak,strong,chroma)="
                                     << ds.total << "," << ds.peak << ","
                                     << ds.strongCells << "," << ds.totalChroma
                    << " sideTotal=" << sideTotal
                    << " ctxTotalMax=" << ctxTotalMax
                    << " motion="    << motionStrong << "/" << motionFrames
                    << " dynMotion=" << dynamicMotionContext
                    << " domMode="   << domMode
                    << " dominant="  << dominant
                    << " ramp="      << ramp
                    << " echoReject=" << echoReject
                    << " lastEdit="  << lastEditFrame
                    << " lPrev="     << lumaPrev
                    << " lCurr="     << lumaCurr;
            }
    
            if (!dominant) continue;
    
            if (ramp) {
                if (LOG_RAMP_VETO && (LOG_VERBOSE_REJECT || LOG_CANDIDATES || dbg)) {
                    qInfo().nospace() << "EditDetector: ramp veto commit@field " << i
                                      << " reason=" << reason;
                }
                continue;
            }
    
            // --- CONTINUITY VETO ---
            // 1) Very-high-corr, chroma-only "edits" (grading changes, small luma shifts)
            // 2) Motion-context spikes that are not strong enough outliers vs neighbors
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
                const double localMax  = std::max(sideTotal, ctxTotalMax);
                const bool softSpike   = (ds.total < localMax * 1.4) && (ds.strongCells <= 1);
                const bool corrNotTerrible = corrInfo && corr > 0.6;
                if (softSpike && corrNotTerrible) {
                    continuityVeto = true;
                }
            }
    
            if (continuityVeto) {
                if (LOG_VERBOSE_REJECT || LOG_CANDIDATES) {
                    qInfo().nospace()
                        << "EditDetector: continuity veto at field " << i
                        << " reason="       << reason
                        << " corr="         << corr
                        << " strong="       << ds.strongCells
                        << " total="        << ds.total
                        << " sideTotal="    << sideTotal
                        << " ctxTotalMax="  << ctxTotalMax
                        << " motionFrames=" << motionFrames
                        << " motionStrong=" << motionStrong
                        << " p90="          << p90Ire;
                }
                continue;
            }
    
            // Commit (field-aligned; ld-analyse cuts don't split fields)
            const int targetField = i;
            if (targetField - lastEditFrame < 3) continue; // strict echo
    
            commitBoundary(targetField, "visual", p90Ire,
                           domMode, reason,
                           corr, corrInfo, eCorr, evidenceScore, ds,
                           lumaPrev, lumaCurr,
                           motionFrames, motionStrong);
        }
    
        sourceVideo.close();
        return editCount;
    }

} // namespace visualEdits
