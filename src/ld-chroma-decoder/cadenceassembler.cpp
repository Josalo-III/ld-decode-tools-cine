/******************************************************************************
 * cadenceassembler.cpp
 * ld-chroma-decoder — Colourisation filter for ld-decode
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 Joseph Burns
 *
 * This file is part of ld-decode-tools.
 ******************************************************************************/
#include "cadenceassembler.h"

#include <limits>
#include <cstdio>
#include <cstdlib>
#include "cadencedefs.h"
#include "combmath.h"
#include "tbc/logging.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

// LDCD_DG_DENY=A|C selectively denies dG merging for one covered-frame
// cadence letter while leaving neighbouring covered frames intact. Covered A
// and C frames alternate in output order, so selecting one letter denies every
// other covered frame while its +-2 covered neighbours retain their merge
// facts. A denied pair follows the same path as --dg-discard for that pair:
// the spare is released to baseline, no certified plane is emitted, and no
// sync-tone anchor is captured. Unset is inert.
static bool dgDenySelective(char letter)
{
    static const int mode = []{
        const char *e = std::getenv("LDCD_DG_DENY");
        if (!e || !e[0]) return 0;
        if (e[0] == 'A' || e[0] == 'a') return 1;
        if (e[0] == 'C' || e[0] == 'c') return 2;
        return 0;
    }();
    return (mode == 1 && letter == 'A') || (mode == 2 && letter == 'C');
}

namespace {

    
    static inline quint16 clampU16(double v)
    {
        if (!(v == v)) return 0;
        if (v < 0.0) return 0;
        if (v > 65535.0) return 65535;
        return static_cast<quint16>(std::llround(v));
    }

    static inline QString pulldownRoleForCadenceId(int cid)
    {
        if (isDefinitionalRole(cid)) return QStringLiteral("definitional");
        if (isSpareRole(cid)) return QStringLiteral("spare");
        return QString();
    }

    static inline void stampForcedCadence(SourceField& field, int cadenceId)
    {
        field.field.cinemap.inUse = true;
        field.field.cinemap.cadenceId = cadenceId;
        field.field.cinemap.cadenceIndexPresumed = true;
        field.field.cinemap.pulldownRole = pulldownRoleForCadenceId(cadenceId);
    }
    
    // Merge a doplGang A/C spare field into its definitional partner.
    //
    // The def and spare are twin captures of the same film field with
    // opposite subcarrier phase, so per sample the conservation identities are
    //     luma    Lhat = (def + spare) / 2
    //     carrier chat = (def - spare) / 2   -- already in DEF phase
    // No burst detection, demodulation, or sign decision is required for this
    // split. The emitted sample is
    //     merged = Lhat + BP(chat)
    // where BP is a linear-phase carrier-band filter (taps at 0/+-2/+-4) with
    // H(fsc) = H(fsc +- 1.3 MHz) = 1 and H(DC) = H(2fsc) = 0. Carrier-band
    // luma remains in Lhat, carrier remains in BP(chat), and twin noise is
    // averaged outside the carrier band.
    //
    // Pass 1 tests twin integrity with the out-of-band residual
    // |chat - BP(chat)|. Samples above threshold are mismatches attributable
    // to motion, edits, mis-pairing, or capture defects. The whole pair is
    // rejected when their fraction exceeds cfg.dgMaxOutlierFrac.
    //
    // For a pair that passes the whole-field test, minor disagreements are
    // repaired per error run. Def, spare, and the two bracketing complement
    // rows are demodulated independently; the complement IQ rows are then
    // averaged to the def line's vertical position. The run-integrated IQ
    // distance to that complement reference decides which twin is the outlier.
    //
    // The accepted twin supplies its own composite sample. A good DEF sample
    // is already in emitted phase and is copied directly. A good SPARE sample
    // is reflected about R, the raw average of the bracketing complement rows,
    // to express its antiphase carrier in DEF phase. The repair sets
    // Lhat = R and chat = kept - R, then rebuilds BP(chat) across the line.
    // The exact-carrier side channel is denied within the BP aperture of each
    // repair because the emitted carrier there is reconstructed rather than an
    // exact twin-conservation fact.
    //
    // Where no repair aperture is present, the exact-carrier side channel is
    // exact = merged - Lhat = BP(chat).
    static bool mergeDgPairWithSanity(const LdDecodeMetaData::VideoParameters& vp,
                                     const CadenceAssembler::Configuration& cfg,
                                     SourceField& def,
                                     SourceField& spare,
                                     SourceField& comp,
                                     std::vector<double>* outRegI = nullptr,
                                     std::vector<double>* outRegQ = nullptr,
                                     std::vector<long>* outRegN = nullptr,
                                     double* outTwinDriftDeg = nullptr)
    {

        const int width = vp.fieldWidth;
        if (width <= 0) return false;

        // SourceField::data is QVector<quint16>, so size() is the sample
        // count and the geometry checks operate directly on that count.
        if (def.data.size() != spare.data.size()) return false;
        if (def.data.size() <= 0) return false;

        const int samples = def.data.size();
        if ((samples % width) != 0) return false;

        const int height = samples / width;
        if (height <= 0) return false;

        int activeLeft  = std::clamp(vp.activeVideoStart, 0, width);
        int activeRight = std::clamp(vp.activeVideoEnd,   0, width);
        if (activeLeft >= activeRight) return false;

        const int y0 = std::clamp(def.getFirstActiveLine(vp), 0, height);
        const int y1 = std::clamp(def.getLastActiveLine(vp),  0, height);
        if (y0 >= y1) return false;

        const double ireScale = (vp.white16bIre - vp.black16bIre) / 100.0;
        if (!(ireScale > 0.0)) return false;
        // Twin-disagreement threshold: samples above 6.0 IRE are repaired
        // from the accepted twin rather than averaged. The threshold is an
        // internal part of the merge policy rather than a user-facing control.
        constexpr double kDgOutlierThreshIre = 6.0;
        const double outlierThreshCode = kDgOutlierThreshIre * ireScale;
        const double maxOutlierFrac    = std::clamp(cfg.dgMaxOutlierFrac, 0.0, 1.0);
        if (!(outlierThreshCode >= 0.0)) return false;

        // Carrier-band filter: unity at fsc and at fsc +- 1.3 MHz, zero at
        // DC and 2fsc (solved exactly for 4fsc sampling).
        constexpr double kT0 = 0.676462;
        constexpr double kT2 = -0.250000;
        constexpr double kT4 = -0.088231;

        const quint16* defp   = reinterpret_cast<const quint16*>(def.data.constData());
        const quint16* sparep = reinterpret_cast<const quint16*>(spare.data.constData());

        std::vector<double> chat(width), lhat(width), bp(width);

        auto buildLine = [&](int lf) {
            const quint16* dl = defp   + (size_t)lf * width;
            const quint16* sl = sparep + (size_t)lf * width;
            for (int h = 0; h < width; ++h) {
                const double d = (double)dl[h];
                const double sv = (double)sl[h];
                lhat[h] = 0.5 * (d + sv);
                chat[h] = 0.5 * (d - sv);
            }
            auto at = [&](int h) {
                return chat[std::clamp(h, activeLeft, activeRight - 1)];
            };
            for (int h = activeLeft; h < activeRight; ++h) {
                bp[h] = kT0 * chat[h] +
                        kT2 * (at(h - 2) + at(h + 2)) +
                        kT4 * (at(h - 4) + at(h + 4));
            }
        };

        // Twin sanity counts picture only. getFirstActiveLine() maps to the
        // non-picture caption/VITC field line for this path, so exactly one
        // field line is skipped before the sanity census and merge emission.
        // The offset is relative to y0 because field parity changes the mapping
        // to exported frame rows. Non-picture data is therefore neither counted
        // as twin damage nor merged with the other capture.
        constexpr int kNonImageFieldLines = 1;   // the caption line
        const int ys = y0 + kNonImageFieldLines;
        if (ys >= y1) return false;

        // Pass 1: twin sanity on the out-of-band residual of chat.
        std::int64_t total = 0, outliers = 0;
        for (int lf = ys; lf < y1; ++lf) {
            buildLine(lf);
            for (int h = activeLeft; h < activeRight; ++h) {
                if (std::fabs(chat[h] - bp[h]) > outlierThreshCode) outliers++;
                total++;
            }
        }
        if (total <= 0) return false;
        const double outlierFrac = double(outliers) / double(total);
        if (outlierFrac > maxOutlierFrac) {
            return false;
        }

        // NOTE: no field-level refusal here. A bad cinemap solve is not
        // the merge's to salvage -- the remedy is --set-cadence or fixing
        // the edit/cadence in ld-cinemap, and a gate that refuses those
        // fields also refuses correct solves whose twins carry ordinary
        // disc damage. The per-sample repair below is what handles damage,
        // and it runs only on lines that need it.

        // Pass 2: merged = Lhat + BP(chat); emit the exact-carrier channel.
        // Minor-disagreement error correction runs per line before emission.
        def.dgExactCarrier.fill(std::numeric_limits<float>::quiet_NaN(), samples);
        quint16* defw = reinterpret_cast<quint16*>(def.data.data());

        const bool haveComp = comp.data.size() == def.data.size();
        const quint16* compp = haveComp
            ? reinterpret_cast<const quint16*>(comp.data.constData())
            : nullptr;
        // Comp field lines bracketing def line lf (interlace geometry).
        const bool defIsTop = def.field.isFirstField;

        std::vector<std::uint8_t> errMask(width), denyExact(width);
        std::vector<double> chatFix(width), lhatFix(width), bpFix(width);
        // Baseband IQ for the twin arbitration: def, spare, the two comp
        // rows, and their positional mean.
        std::vector<double> aI(width), aQ(width), bI(width), bQ(width),
                            uI(width), uQ(width), vI(width), vQ(width),
                            cI(width), cQ(width);

        // Twin phase capture feeds the sync tracker.
        // BURST-RELATIVE carrier phase per capture is (carrier phase - own burst
        // phase), which is comparable across stream positions without lattice
        // bookkeeping. Def and spare contain the same picture at two capture
        // moments, so the difference of their burst-relative phases provides a
        // motion-free sample of sequence drift directly from the conservation
        // pair.
        double phCI = 0, phCQ = 0;             // pooled chat IQ (def coords)
        double phDBI = 0, phDBQ = 0, phSBI = 0, phSBQ = 0; // burst pools
        long phN = 0;
        // Regional pools for the sync tracker (same lattice the decoder's
        // grids use: field-line row / 16, active-sample col / 128).
        const int srX = (activeRight - activeLeft +
                         SourceField::kSyncRegCols - 1) /
                        SourceField::kSyncRegCols;
        const int srY = (height + SourceField::kSyncRegLines - 1) /
                        SourceField::kSyncRegLines;
        std::vector<double> srI((size_t)srX * srY, 0.0),
                            srQ((size_t)srX * srY, 0.0);
        std::vector<long> srN((size_t)srX * srY, 0);
        const int pbL = std::clamp(vp.colourBurstStart, 0, width);
        const int pbR = std::clamp(vp.colourBurstEnd, 0, width);
        static const double phCB[4] = { 1, 0, -1, 0 };
        static const double phSB[4] = { 0, 1, 0, -1 };

        // Emission starts at the first image line, so the spare is DISCARDED
        // on the caption lines. Closed captions / VITC are data, and the two
        // captures carry different data there; merging them averages two
        // unlike waveforms into an illegible one. def's own samples are
        // already in def.data, so skipping the write keeps the caption
        // exactly as the def field carried it. dgExactCarrier stays NaN on
        // those lines, which makes certifiedDefLine() false for them, so no
        // candidate cedes to a data line either. The phase pools below
        // likewise must not read caption data as carrier.
        for (int lf = ys; lf < y1; ++lf) {
            buildLine(lf);
            quint16* out = defw + (size_t)lf * width;

            // Error localisation: broadband twin disagreement.
            bool anyErr = false;
            for (int h = activeLeft; h < activeRight; ++h) {
                errMask[h] = std::fabs(chat[h] - bp[h]) > outlierThreshCode;
                if (errMask[h]) anyErr = true;
            }

            // Phase capture: pooled carrier (clean samples) + each
            // capture's own burst. Field lines are carrier-cophased
            // (455.0 cycles per field-line pitch), so the pools are
            // coherent with the plain sample-class basis.
            {
                // Consecutive field rows are consecutive scan lines and are
                // 227.5 carrier cycles apart, so their carrier signs alternate.
                // Apply row parity before pooling so adjacent rows add
                // coherently instead of cancelling.
                const double rs = (lf & 1) ? -1.0 : 1.0;
                const quint16* dl = defp + (size_t)lf * width;
                const quint16* sl = sparep + (size_t)lf * width;
                const size_t srRow =
                    (size_t)(lf / SourceField::kSyncRegLines) * srX;
                for (int h = activeLeft; h < activeRight; ++h) {
                    if (errMask[h]) continue;
                    const int ph = h & 3;
                    const double ci = rs * chat[h] * phCB[ph];
                    const double cq = rs * chat[h] * phSB[ph];
                    phCI += ci;
                    phCQ += cq;
                    phN++;
                    const size_t r = srRow +
                        (h - activeLeft) / SourceField::kSyncRegCols;
                    srI[r] += ci; srQ[r] += cq; srN[r]++;
                }
                for (int h = pbL; h < pbR; ++h) {
                    const int ph = h & 3;
                    phDBI += rs * (double)dl[h] * phCB[ph];
                    phDBQ += rs * (double)dl[h] * phSB[ph];
                    phSBI += rs * (double)sl[h] * phCB[ph];
                    phSBQ += rs * (double)sl[h] * phSB[ph];
                }
            }

            if (anyErr && haveComp) {
                const int cUp = std::clamp(defIsTop ? lf - 1 : lf, 0, height - 1);
                const int cDn = std::clamp(defIsTop ? lf : lf + 1, 0, height - 1);
                const quint16* ru = compp + (size_t)cUp * width;
                const quint16* rd = compp + (size_t)cDn * width;
                auto R = [&](int h) {
                    h = std::clamp(h, activeLeft, activeRight - 1);
                    return 0.5 * ((double)ru[h] + (double)rd[h]);
                };
                std::copy(chat.begin(), chat.end(), chatFix.begin());
                std::copy(lhat.begin(), lhat.end(), lhatFix.begin());

                const quint16* dl = defp + (size_t)lf * width;
                const quint16* sl = sparep + (size_t)lf * width;

                // Demodulate FIRST, interpolate SECOND.
                //
                // The comp field has no sample row at the def line's vertical
                // position, so the two bracketing comp rows are interpolated
                // after demodulation. Composite-domain interpolation would also
                // cancel their antiphase carrier; demodulating each row first
                // preserves the comp chroma while the subsequent average
                // performs only the required positional interpolation.
                static const double kCB[4] = { 1, 0, -1, 0 };
                static const double kSB[4] = { 0, 1, 0, -1 };
                auto demodRow = [&](const quint16* row, int scanLine,
                                    std::vector<double>& oI,
                                    std::vector<double>& oQ) {
                    const double rs = (scanLine & 1) ? -1.0 : 1.0;
                    for (int x = activeLeft; x < activeRight; ++x) {
                        double si = 0.0, sq = 0.0;
                        int n = 0;
                        // One full carrier cycle: cancels the alternating
                        // product image and the luma pedestal together.
                        for (int k = -2; k <= 1; ++k) {
                            const int xx = x + k;
                            if (xx < activeLeft || xx >= activeRight) continue;
                            const double v = rs * (double)row[xx];
                            si += v * kCB[xx & 3];
                            sq += v * kSB[xx & 3];
                            ++n;
                        }
                        oI[x] = n ? si / n : 0.0;
                        oQ[x] = n ? sq / n : 0.0;
                    }
                };
                const int defScan = 2 * lf   + (defIsTop ? 0 : 1);
                const int upScan  = 2 * cUp  + (defIsTop ? 1 : 0);
                const int dnScan  = 2 * cDn  + (defIsTop ? 1 : 0);
                demodRow(dl, defScan, aI, aQ);
                demodRow(sl, defScan, bI, bQ);
                demodRow(ru, upScan,  uI, uQ);
                demodRow(rd, dnScan,  vI, vQ);
                for (int x = activeLeft; x < activeRight; ++x) {
                    cI[x] = 0.5 * (uI[x] + vI[x]);
                    cQ[x] = 0.5 * (vQ[x] + uQ[x]);
                }

                // No per-twin phase correction is applied. Both twins are
                // evaluated in the same demodulation basis against the same
                // interpolated comp reference, so any phase offset carried by
                // that reference is common to both distances and cancels from
                // the closer-twin decision. This comparison is only an
                // arbitration metric; the def/spare carrier relationship is
                // defined independently by the conservation identity
                // (def - spare) / 2.

                // Arbitrate per error RUN: whichever twin's chroma sits
                // closer to the comp's at that height is signal; the
                // outlier is the error and is discarded.
                int h = activeLeft;
                while (h < activeRight) {
                    if (!errMask[h]) { ++h; continue; }
                    int rEnd = h;
                    while (rEnd + 1 < activeRight && errMask[rEnd + 1]) ++rEnd;
                    double eDef = 0.0, eSpare = 0.0;
                    for (int x = h; x <= rEnd; ++x) {
                        eDef   += std::hypot(aI[x] - cI[x], aQ[x] - cQ[x]);
                        eSpare += std::hypot(bI[x] - cI[x], bQ[x] - cQ[x]);
                    }
                    const bool badIsDef = eDef > eSpare;
                    for (int x = h; x <= rEnd; ++x) {
                        // Keep the surviving twin's OWN sample -- its own
                        // carrier, not one interpolated from a cycle away.
                        // def is already in the emitted field's phase and
                        // is taken verbatim; spare is anti-phased, so it is
                        // reflected about the carrier-free local luma R to
                        // express the same reading in def phase.
                        const double kept = badIsDef
                            ? 2.0 * R(x) - (double)sl[x]
                            : (double)dl[x];
                        lhatFix[x] = R(x);
                        chatFix[x] = kept - R(x);
                    }
                    h = rEnd + 1;
                }

                // Rebuild the carrier-band filter over the repaired chat.
                auto atF = [&](int x) {
                    return chatFix[std::clamp(x, activeLeft, activeRight - 1)];
                };
                for (int x = activeLeft; x < activeRight; ++x) {
                    bpFix[x] = kT0 * chatFix[x] +
                               kT2 * (atF(x - 2) + atF(x + 2)) +
                               kT4 * (atF(x - 4) + atF(x + 4));
                }

                // The exact channel is a conservation fact; deny it within
                // the BP aperture of any repaired sample.
                std::fill(denyExact.begin(), denyExact.end(), std::uint8_t(0));
                for (int x = activeLeft; x < activeRight; ++x) {
                    if (!errMask[x]) continue;
                    for (int k = -4; k <= 4; ++k) {
                        const int xx = x + k;
                        if (xx >= activeLeft && xx < activeRight)
                            denyExact[xx] = 1;
                    }
                }

                for (int x = activeLeft; x < activeRight; ++x) {
                    const double m = lhatFix[x] + bpFix[x];
                    out[x] = clampU16(m);
                    if (!denyExact[x])
                        def.dgExactCarrier[(size_t)lf * width + x] =
                            static_cast<float>((double)out[x] - lhatFix[x]);
                }
                continue;
            }

            for (int h2 = activeLeft; h2 < activeRight; ++h2) {
                const double m = lhat[h2] + bp[h2];
                out[h2] = clampU16(m);
                def.dgExactCarrier[(size_t)lf * width + h2] =
                    static_cast<float>((double)out[h2] - lhat[h2]);
            }
        }

        // Sync tracker feed: regional pools derotated by the def capture's
        // own burst (burst-relative phase; slow burst wander must not alias
        // as carrier drift), plus the twin integrity differential.
        if (outRegI && outRegQ && outRegN) {
            const double bm = std::hypot(phDBI, phDBQ);
            const double buI = bm > 1e-9 ? phDBI / bm : 1.0;
            const double buQ = bm > 1e-9 ? phDBQ / bm : 0.0;
            outRegI->assign(srI.size(), 0.0);
            outRegQ->assign(srQ.size(), 0.0);
            *outRegN = srN;
            for (size_t r = 0; r < srI.size(); ++r) {
                (*outRegI)[r] = srI[r] * buI + srQ[r] * buQ;
                (*outRegQ)[r] = srQ[r] * buI - srI[r] * buQ;
            }
        }
        // def carrier is chat; spare carrier is -chat sample-for-sample.
        // Each referenced to its OWN burst; the difference of the two
        // burst-relative phases (mod the fixed sequence offset) is the
        // motion-free drift sample.
        if (outTwinDriftDeg && phN > 0) {
            const double aDef2 = std::atan2(phCQ, phCI);
            const double aSp2  = std::atan2(-phCQ, -phCI);
            const double bDef2 = std::atan2(phDBQ, phDBI);
            const double bSp2  = std::atan2(phSBQ, phSBI);
            double dd = ((aSp2 - bSp2) - (aDef2 - bDef2)) * 180.0 / M_PI;
            while (dd > 180.0) dd -= 360.0;
            while (dd < -180.0) dd += 360.0;
            *outTwinDriftDeg = dd;
        }

        return true;
    }

} // namespace

// CadenceAssembler — telecine pulldown consolidation for ld-chroma-decoder.
//
// CadenceAssembler accepts SourceFields with cadence metadata and assembles
// them into WorkItems for the comb decoder. It supports three operating modes:
//
// Autosolve (default): cadenceId assignments identify definitional,
//   complement, and spare roles. Definitional/complement fields are paired into
//   film frames; A-trailing and C-leading spares are conservation-merged into
//   their definitional fields when twin sanity passes, otherwise the spare is
//   released to baseline passthrough.
//
// Forced cadence (--set-cadence): field roles are assigned from the imposed
//   A-B-C-D cycle position rather than cadenceId evidence.
//
// Output: normal telecine mode emits TelecineFrame and PassthroughFrame work.
//   --export-24p emits one FilmFrame per reconstructed film frame without spare
//   expansion. Fields that cannot be placed in a film frame are released to
//   baseline so the original video stream remains complete.
//
// CadenceAssembler Implementation
CadenceAssembler::CadenceAssembler(const LdDecodeMetaData::VideoParameters& vp,
                                   const Configuration& cfg,
                                   std::function<void(qint32)> onBaseline)
    : onFieldReleasedToBaseline(std::move(onBaseline)),
      videoParameters(vp), config(cfg)
{
}

int CadenceAssembler::nextUnconsumedIndex(int start) const
{
    const int n = history.size();
    for (int i = std::max(0, start); i < n; ++i) {
        if (!history[i].consumed)
            return i;
    }
    return -1; // none
}

int CadenceAssembler::countUnconsumedFrom(int start, int maxCount) const
{
    int count = 0;
    int idx   = nextUnconsumedIndex(start);
    while (idx >= 0 && count < maxCount) {
        ++count;
        idx = nextUnconsumedIndex(idx + 1);
    }
    return count;
}

// Helper: mark history index consumed and remove seq->index map entry.
// Must be called BEFORE moving the history[pos].field out of history.
void CadenceAssembler::markHistoryConsumed(int pos)
{
    if (pos < 0 || pos >= history.size()) return;
    const int seq = history[pos].field.field.seqNo;
    seqNoToHistoryIndex.remove(seq);
    history[pos].consumed = true;
}

// Release a consumed entry's sample planes while leaving its positional slot
// and metadata intact. seqNoToHistoryIndex stores positions in history, so
// entries are not erased and those indices remain stable.
//
// No active reader needs sample planes from a consumed entry:
// nextUnconsumedIndex skips it, while complement/spare lookup refuses consumed
// entries. Consumption paths that hand ownership to baseline cause DecoderPool
// to reload the source field from TBC rather than use this stored copy.
//
// Call this only after any required move from history. markHistoryConsumed()
// runs before a move because it still needs seqNo; payload release is therefore
// a separate operation. Replacing the containers also releases their capacity.
void CadenceAssembler::releaseHistoryPayload(int pos)
{
    if (pos < 0 || pos >= history.size()) return;
    SourceField &f = history[pos].field;
    f.data            = SourceVideo::Data();
    f.dgExactCarrier  = QVector<float>();
    f.dgSyncIncrement = QVector<float>();
}

void CadenceAssembler::handOffCaptureFrameToBaseline(int pos)
{
    if (pos < 0 || pos >= history.size()) return;
    if (history[pos].consumed) return;

    const qint32 seqNo = history[pos].field.field.seqNo;
    const qint32 partnerSeqNo = history[pos].capturePartnerSeqNo;

    baselineOwnedSeqNos.insert(seqNo);
    if (partnerSeqNo >= 0) baselineOwnedSeqNos.insert(partnerSeqNo);

    // Transfer responsibility for the original capture frame to baseline before
    // retiring this field from history.  Do not consume the capture partner here:
    // in a mixed AB/BC frame that partner is still a source for the neighbouring
    // reconstructed film frame (B1 for AB, or B2 for BC).  baselineOwnedSeqNos
    // prevents a later passthrough from being queued twice without revoking that
    // field's eligibility for normal film extraction.
    releaseSeqToBaseline(seqNo);

    markHistoryConsumed(pos);
    releaseHistoryPayload(pos);
}

void CadenceAssembler::push(const QVector<SourceField>& newFields)
{
    if (config.setCadence != 0 && !config.noCinemap) {
        if (config.reverseFieldOrder) {
            // -r presents each capture frame second-field-first, so each pair
            // arrives transposed (seq 3,2,5,4,7,6...). Restore ascending order
            // before forced-cadence counting because the imposed cycle is
            // defined on cadence slots, not on transposed storage order.
            // Pushes are frame-aligned in forced mode, so each pair here is one
            // capture frame. Any unexpected odd tail is passed through rather
            // than allowed to shift all subsequent slot assignments.
            const int n = newFields.size();
            int i = 0;
            for (; i + 1 < n; i += 2) {
                window.push_back(newFields[i + 1]);
                window.push_back(newFields[i]);
            }
            if (i < n) window.push_back(newFields[i]);
        } else {
            for (const auto& f : newFields) window.push_back(f);
        }
        processWindowForced(false);
        return;
    }

    history.reserve(history.size() + newFields.size());
    for (const auto& f : newFields) {
        HistoryField hf;
        hf.field               = f;
        hf.consumed            = false;
        hf.capturePartnerSeqNo = f.capturePartnerSeqNo;
        seqNoToHistoryIndex.insert(f.field.seqNo, static_cast<int>(history.size()));
        history.push_back(std::move(hf));
    }

    processHistory(false);
}

void CadenceAssembler::flush()
{
    if (config.setCadence != 0 && !config.noCinemap) {
        processWindowForced(true);
    } else {
        processHistory(true);
    }
}

bool CadenceAssembler::hasWork() const { return !workQueue.empty(); }

// Sync-tone tracker update at a dG anchor. The alpha-beta state tracks phase
// and rate per region; twin integrity gates anchor admission, and a cut resets
// the tracker on frame-median innovation.
void CadenceAssembler::syncTrackerUpdate(int anchorSeq, double twinDriftDeg,
                                         const std::vector<double>& regI,
                                         const std::vector<double>& regQ,
                                         const std::vector<long>& regN,
                                         double ireScale)
{
    const int nReg = (int)regI.size();
    if (nReg <= 0) return;
    if ((int)syncTrk.size() != nReg) {
        syncTrk.assign(nReg, SyncTrk());
        syncAnchorSeq = -1;
    }
    if (std::fabs(twinDriftDeg) > 1.0) {
        return;
    }
    constexpr double kAlpha = 0.5, kBeta = 0.12;
    const double dt = syncAnchorSeq >= 0
        ? (double)(anchorSeq - syncAnchorSeq) : 0.0;
    const double floorRaw = 2.0 * ireScale;
    std::vector<double> misses;
    for (int r = 0; r < nReg; ++r) {
        if (regN[r] < 64) continue;
        if (std::hypot(regI[r], regQ[r]) / regN[r] < floorRaw) continue;
        SyncTrk &T = syncTrk[r];
        if (!T.valid || dt <= 0.0 || dt > 8.0) continue;
        const double th = T.omega * dt;
        const double pI = T.zI * std::cos(th) - T.zQ * std::sin(th);
        const double pQ = T.zQ * std::cos(th) + T.zI * std::sin(th);
        misses.push_back(std::fabs(std::atan2(
            regQ[r] * pI - regI[r] * pQ, regI[r] * pI + regQ[r] * pQ)));
    }
    bool cut = false;
    if (!misses.empty()) {
        std::sort(misses.begin(), misses.end());
        cut = misses[misses.size() / 2] > 35.0 * M_PI / 180.0;
    }
    if (cut) {
        for (SyncTrk &T : syncTrk) T.valid = false;
        syncCuts++;
    }
    for (int r = 0; r < nReg; ++r) {
        if (regN[r] < 64) continue;
        if (std::hypot(regI[r], regQ[r]) / regN[r] < floorRaw) continue;
        SyncTrk &T = syncTrk[r];
        const double m = std::hypot(regI[r], regQ[r]);
        if (!T.valid || dt <= 0.0 || dt > 8.0) {
            T.zI = regI[r] / m; T.zQ = regQ[r] / m;
            T.omega = 0.0; T.missEwma = 0.5; T.valid = true;
            continue;
        }
        const double th = T.omega * dt;
        const double pI = T.zI * std::cos(th) - T.zQ * std::sin(th);
        const double pQ = T.zQ * std::cos(th) + T.zI * std::sin(th);
        const double err = std::atan2(regQ[r] * pI - regI[r] * pQ,
                                      regI[r] * pI + regQ[r] * pQ);
        T.missEwma = 0.7 * T.missEwma + 0.3 * std::fabs(err);
        T.omega = std::clamp(T.omega + kBeta * err / dt, -0.10, 0.10);
        const double corr = th + kAlpha * err;
        const double cI = T.zI * std::cos(corr) - T.zQ * std::sin(corr);
        const double cQ = T.zQ * std::cos(corr) + T.zI * std::sin(corr);
        const double mz = std::hypot(cI, cQ);
        T.zI = cI / mz; T.zQ = cQ / mz;
    }
    // Global tone is pooled across all usable regions to provide a common
    // measurable phase-rate coordinate alongside the per-region tracker state.
    {
        double gI = 0, gQ = 0;
        for (int r = 0; r < nReg; ++r) {
            if (regN[r] < 64) continue;
            if (std::hypot(regI[r], regQ[r]) / regN[r] < floorRaw) continue;
            gI += regI[r]; gQ += regQ[r];
        }
        const double gm = std::hypot(gI, gQ);
        if (gm > 1e-9) {
            SyncTrk &T = syncGlobal;
            if (!T.valid || dt <= 0.0 || dt > 8.0) {
                T.zI = gI / gm; T.zQ = gQ / gm;
                T.omega = 0.0; T.missEwma = 0.2; T.valid = true;
            } else {
                const double th = T.omega * dt;
                const double pI = T.zI * std::cos(th) - T.zQ * std::sin(th);
                const double pQ = T.zQ * std::cos(th) + T.zI * std::sin(th);
                const double err = std::atan2(gQ * pI - gI * pQ,
                                              gI * pI + gQ * pQ);
                T.missEwma = 0.7 * T.missEwma + 0.3 * std::fabs(err);
                T.omega = std::clamp(T.omega + 0.25 * err / dt, -0.05, 0.05);
                const double corr2 = th + 0.6 * err;
                const double cI = T.zI * std::cos(corr2) - T.zQ * std::sin(corr2);
                const double cQ = T.zQ * std::cos(corr2) + T.zI * std::sin(corr2);
                const double mz = std::hypot(cI, cQ);
                T.zI = cI / mz; T.zQ = cQ / mz;
            }
        }
    }
    syncAnchorSeq = anchorSeq;
}

// Stamp a field with the tracker's predicted rotation since the previous
// anchor, per region, plus confidence. Increments, never absolutes: the
// consumer composes with its own in-batch anchor measurement, so a fixed
// convention offset between assembler and decoder space cancels; the
// end-to-end probe measures the handedness.
void CadenceAssembler::syncStamp(SourceField& f) const
{
    f.dgSyncIncrement.clear();
    if (syncAnchorSeq < 0 || syncTrk.empty()) return;
    const long dt = (long)f.field.seqNo - syncAnchorSeq;
    if (dt < 0 || dt > 8) return;
    // Layout header: [globalOmega (rad/field), globalConf, dtFields,
    // reserved], then per-region [omega, conf] pairs. Rates, not
    // increments: the consumer multiplies by its own dt, and validation
    // can compare rates across any sampling.
    f.dgSyncIncrement.resize(4 + (int)syncTrk.size() * 2);
    {
        const SyncTrk &G = syncGlobal;
        double gconf = 0.0;
        if (G.valid) {
            gconf = std::clamp(1.0 - G.missEwma / (15.0 * M_PI / 180.0),
                               0.0, 1.0);
            if (dt > 5) gconf *= 0.5;
        }
        f.dgSyncIncrement[0] = (float)(G.valid ? G.omega : 0.0);
        f.dgSyncIncrement[1] = (float)gconf;
        f.dgSyncIncrement[2] = (float)dt;
        f.dgSyncIncrement[3] = 0.0f;
    }
    for (size_t r = 0; r < syncTrk.size(); ++r) {
        const SyncTrk &T = syncTrk[r];
        double conf = 0.0;
        if (T.valid) {
            conf = std::clamp(1.0 - T.missEwma / (30.0 * M_PI / 180.0),
                              0.0, 1.0);
            if (dt > 5) conf *= 0.5;
        }
        f.dgSyncIncrement[4 + (int)r * 2]     =
            (float)(T.valid ? T.omega : 0.0);
        f.dgSyncIncrement[4 + (int)r * 2 + 1] = (float)conf;
    }
}

QVector<CadenceAssembler::WorkItem> CadenceAssembler::popWork()
{
    QVector<WorkItem> out;
    out.reserve((int)workQueue.size());
    while (!workQueue.empty()) {
        out.push_back(std::move(workQueue.front()));
        workQueue.pop_front();
    }
    return out;
}

bool CadenceAssembler::boundaryBetweenFields(const SourceField& prev, const SourceField& next)
{
    // Segment heads must never be paired with preceding material.
    if (next.field.cinemap.isEditBoundary) return true;

    // Also treat sequence discontinuity as a hard boundary.
    if (next.field.seqNo != prev.field.seqNo + 1) return true;

    return false;
}

bool CadenceAssembler::orderPairForComb(SourceField& a, SourceField& b) const
{
    if (!a.field.isFirstField && b.field.isFirstField) {
        std::swap(a, b);
        return true; // swapped to put Top first
    }
    return false;
}

bool CadenceAssembler::mergeDgPairWithSanityWrapper(SourceField& def, SourceField& spare, SourceField& comp)
{
    std::vector<double> regI, regQ;
    std::vector<long> regN;
    double twinDrift = 999.0;
    const bool ok = mergeDgPairWithSanity(videoParameters, config, def, spare,
                                          comp, &regI, &regQ, &regN,
                                          &twinDrift);
    if (ok) {
        const double ireScale =
            (videoParameters.white16bIre - videoParameters.black16bIre) / 100.0;
        syncTrackerUpdate(def.field.seqNo, twinDrift, regI, regQ, regN,
                          ireScale);
    }
    return ok;
}

// Forced cadence start offset helper.
//
// --set-cadence N is a FRAME count, 1..5, presented to the user 1-based over
// exactly the axis ld-cinemap reports as phaseOffset (p0..p4 in its phase
// tables): N-1 is the position, within the 5-frame pulldown cycle, of the
// FIRST FRAME OF THIS RENDER. N=1 means the render opens on the complete A
// frame. The count is relative to -s, so a scoped re-render of the same disc
// names its own opening frame -- which is what makes a multicadence
// composite workable: the autosolve keeps the dominant pattern, and each
// other pattern gets its own scoped render at its own index.
//
// The five capture frames of the cycle occupy field slots (0,1) (2,3) (4,5)
// (6,7) (8,9), so the first field's slot is simply 2*(N-1). Every value
// therefore lands on a frame boundary, which is the whole point: A starts
// clean on a frame and the user is only telling us which one.
//
// Nothing here validates the choice and no evidence may override it. Jamming
// against the grain is the feature.
int CadenceAssembler::forcedStartIndex() const
{
    const int n = std::clamp(config.setCadence, 1, 5);
    int base = 2 * (n - 1);

    // -r maps upper-field-first input to the lower-field-first cadence regime
    // represented by cadenceIds 10..19. The cadence index layout itself is
    // unchanged, but the transposed stream sits one field later, so +1 supplies
    // the stream alignment. processWindowForced() stamps the corresponding
    // inverted cadenceIds so downstream code can recover both the normalised
    // role index and the asserted dominance.
    if (config.reverseFieldOrder) base = (base + 1) % CADENCE_NTSC_CYCLE;
    return base;
}

// Assumes seqNoToHistoryIndex is maintained on push/consume.
// Returns position of the complement for history[i0], or -1 if not present.
int CadenceAssembler::findComplementPos(int i0) const {
    const auto &head = history[i0].field;
    const int cid    = head.field.cinemap.cadenceId;
    if (!cadenceKnown(cid)) return -1;

    const int idx     = cadenceIndex(cid);
    const int compCid = filmFrameComplement(cid);
    const char letter = cadenceFilmLetter(cid);

    int delta = 0;
    switch (idx) {
        case 0: delta = +1; break; // Adef -> Acomp
        case 1: delta = -1; break; // Acomp -> Adef
        case 2: delta = -1; break; // Aspare -> Acomp (if ever needed)
        case 3: delta = +1; break; // B first -> B second
        case 4: delta = -1; break; // B second -> B first
        case 5: delta = +1; break; // Cspare -> Ccomp
        case 6: delta = +1; break; // Ccomp -> Cdef
        case 7: delta = -1; break; // Cdef -> Ccomp
        case 8: delta = +1; break; // D8 -> D9
        case 9: delta = -1; break; // D9 -> D8
        default: return -1;
    }

    const int targetSeq = head.field.seqNo + delta;
    auto it = seqNoToHistoryIndex.find(targetSeq);
    if (it == seqNoToHistoryIndex.end()) return -1;

    const int pos = it.value();
    if (history[pos].consumed) return -1;

    // Guard against wrong letter/contract violations
    const auto &cand = history[pos].field;
    if (!cadenceKnown(cand.field.cinemap.cadenceId)) return -1;
    if (cadenceFilmLetter(cand.field.cinemap.cadenceId) != letter) return -1;
    if (cand.field.cinemap.cadenceId != compCid) return -1;

    return pos;
}

// Forced cadence (jam) mode: the user supplies the cadence pattern directly.
// Fields are consumed in strict A-B-C-D order from setCadence, which names the
// cycle position of the render's first frame. cadenceId evidence does not veto
// or alter the asserted role assignment.
//
// Two invariants govern consumption:
//
//   1. Every slot makes progress. A slot either begins a defined film group or
//      releases one orphan field to baseline and advances.
//
//   2. Every real field retains an output path. Anything that cannot be placed
//      in the asserted film group is released to baseline rather than dropped.
//
// A film group is consumed only when all fields required by that group are
// present, so push boundaries cannot separate a definitional field from its
// required spare.
void CadenceAssembler::processWindowForced(bool flushMode)
{
    if (config.noCinemap) return;

    const int start = forcedStartIndex();

    // Dominance regime of the stamped identity. -r asserts lower-field-first,
    // which is the inverted regime ld-cinemap records as cadenceIds 10..19; the
    // index layout is identical, so cadenceIndex() normalises it for every role
    // lookup downstream while cadenceIsInverted() keeps the dominance readable.
    // Stamping the plain 0..9 space under -r would silently claim normal
    // dominance for an inverted render.
    const int regime = config.reverseFieldOrder ? CADENCE_NTSC_INVERTED_OFFSET : 0;

    // Forced cadence is authoritative: field roles come from the asserted
    // slot sequence rather than from cadence evidence.

    auto cycleIndex = [&](qint64 consumed) -> int {
        int idx = int((start + (consumed % CADENCE_NTSC_CYCLE)) % CADENCE_NTSC_CYCLE);
        if (idx < 0) idx += CADENCE_NTSC_CYCLE;
        return idx; // 0..9
    };

    auto emitTelecine2 = [&](char label, SourceField&& f1, int cid1,
                             SourceField&& f2, int cid2,
                             WorkItem::Expansion ex = WorkItem::Expansion::None) {
        stampForcedCadence(f1, regime + cid1);
        stampForcedCadence(f2, regime + cid2);
        bool swapped = orderPairForComb(f1, f2);
        WorkItem wi;
        wi.kind               = WorkItem::Kind::TelecineFrame;
        wi.expansion          = ex;
        wi.fieldsSwapped      = swapped;
        wi.invertedFieldOrder = false;
        wi.filmLabel          = label;
        wi.f1                 = std::move(f1);
        wi.f2                 = std::move(f2);
        syncStamp(wi.f1);
        syncStamp(wi.f2);
        workQueue.push_back(std::move(wi));
    };

    auto pop1 = [&]() -> SourceField {
        SourceField f = std::move(window.front());
        window.pop_front();
        ++forcedFieldIndex;
        return f;
    };

    auto pop2 = [&]() -> std::pair<SourceField, SourceField> {
        SourceField a = pop1();
        SourceField b = pop1();
        return {std::move(a), std::move(b)};
    };

    // One loop for both push and flush. Each pass either consumes a complete
    // group for the current slot, releases a single unplaceable field to
    // baseline, or returns to wait for more input. Because every branch that
    // does not return consumes at least one field, and every slot has a
    // branch, the count always advances and no field is left unaccounted for.
    while (!window.empty()) {
        const int idx   = cycleIndex(forcedFieldIndex);
        const int avail = static_cast<int>(window.size());

        // Not enough input for this slot's group. Mid-stream that means wait;
        // at flush it means the group runs off the end of the range, so the
        // head field is baseline material and the next slot gets its turn.
        auto shortOfGroup = [&](int need) {
            if (avail >= need) return false;
            if (!flushMode) return true;
            releaseToBaseline(pop1());
            return true;
        };

        switch (idx) {
        case 0: {
            // A: def + comp, plus the trailing spare at slot 2. The spare is
            // part of this group — hold the whole thing until it has arrived,
            // or the merge loses a cover at every push boundary.
            if (shortOfGroup(flushMode ? 2 : 3)) {
                if (!flushMode) return;
                continue;
            }
            auto [def, comp] = pop2();
            stampForcedCadence(def, regime + 0);
            stampForcedCadence(comp, regime + 1);

            WorkItem::Expansion ex = WorkItem::Expansion::None;
            if (!window.empty() && cycleIndex(forcedFieldIndex) == 2) {
                SourceField spare = pop1();
                stampForcedCadence(spare, regime + 2);
                if (!config.dgDiscard && !dgDenySelective('A') &&
                    mergeDgPairWithSanityWrapper(def, spare, comp)) {
                    ex = WorkItem::Expansion::Trailing;
                } else {
                    // Twins disagreed (or the user asked for --dg-discard):
                    // no certified plane for this frame, and the spare is
                    // ordinary video. Exactly the dg-discard outcome, decided
                    // per pair by the twins themselves.
                    releaseToBaseline(std::move(spare));
                }
            }
            emitTelecine2('A', std::move(def), 0, std::move(comp), 1, ex);
            continue;
        }

        case 3: {
            // B: two ordinary fields, no twin and so no certified carrier.
            if (shortOfGroup(2)) {
                if (!flushMode) return;
                continue;
            }
            auto [f3, f4] = pop2();
            emitTelecine2('B', std::move(f3), 3, std::move(f4), 4,
                          WorkItem::Expansion::None);
            continue;
        }

        case 5: {
            // C: leading spare (slot 5) + comp + def.
            if (shortOfGroup(3)) {
                if (!flushMode) return;
                continue;
            }
            SourceField spare = pop1(); // 5
            SourceField comp  = pop1(); // 6
            SourceField def   = pop1(); // 7
            stampForcedCadence(spare, regime + 5);
            stampForcedCadence(comp, regime + 6);
            stampForcedCadence(def, regime + 7);

            WorkItem::Expansion ex = WorkItem::Expansion::None;
            if (!config.dgDiscard && !dgDenySelective('C') &&
                mergeDgPairWithSanityWrapper(def, spare, comp)) {
                ex = WorkItem::Expansion::Leading;
            } else {
                releaseToBaseline(std::move(spare));
            }
            emitTelecine2('C', std::move(comp), 6, std::move(def), 7, ex);
            continue;
        }

        case 6: {
            // C whose leading spare fell outside the range (a scoped render
            // starting here). The film frame still stands, uncovered.
            if (shortOfGroup(2)) {
                if (!flushMode) return;
                continue;
            }
            auto [comp, def] = pop2();
            emitTelecine2('C', std::move(comp), 6, std::move(def), 7,
                          WorkItem::Expansion::None);
            continue;
        }

        case 8: {
            // D
            if (shortOfGroup(2)) {
                if (!flushMode) return;
                continue;
            }
            auto [f8, f9] = pop2();
            emitTelecine2('D', std::move(f8), 8, std::move(f9), 9,
                          WorkItem::Expansion::None);
            continue;
        }

        default:
            // Slots 1, 2, 4, 7, 9: an orphan half-frame whose partner lies
            // before the start of this render (slot 4 is --set-cadence 3, and
            // -r puts every value on an odd slot), or a spare whose
            // definitional partner was already emitted. No invention of a
            // missing partner: the field is real video, so it goes to
            // baseline and the count moves on.
            releaseToBaseline(pop1());
            continue;
        }
    }
}

void CadenceAssembler::processHistory(bool flushMode)
{
    constexpr int kDefaultFallbackLookaheadFields = 4;

    auto advanceCursorPastConsumed = [&]() {
        while (cursor < (int)history.size() && history[cursor].consumed) {
            ++cursor;
        }
    };

    while (true) {
        advanceCursorPastConsumed();

        // Need at least 2 unconsumed fields to emit anything.
        int i0 = nextUnconsumedIndex(cursor);
        if (i0 < 0) break;
        int i1 = nextUnconsumedIndex(i0 + 1);
        if (i1 < 0) break;

        // 1) Try film extraction (handles spares and standard pairs)
        if (tryExtractFilmFrameAtCursor()) {
            continue;
        }

        // Default 29.97i output gives film assembly one additional frame of lookahead
        // before surrendering ownership to plain passthrough/baseline handling.
        if (!flushMode && !config.export24p &&
            countUnconsumedFrom(i0, kDefaultFallbackLookaheadFields) < kDefaultFallbackLookaheadFields) {
            break;
        }

        // 2) Normal passthrough (must be allowed to make progress)
        if (tryEmitPassthroughAtCursor(flushMode, /*force=*/false)) {
            continue;
        }

        // Can't make progress: stop (caller will feed more or we will flush later).
        break;
    }

    if (flushMode) {
        // Drain everything as passthrough. No waiting, no heuristics.
        while (true) {
            advanceCursorPastConsumed();
            int i0 = nextUnconsumedIndex(cursor);
            if (i0 < 0) break;
            int i1 = nextUnconsumedIndex(i0 + 1);
            if (i1 < 0) break;

            if (!tryEmitPassthroughAtCursor(/*flushMode=*/true, /*force=*/true)) {
                break;
            }
        }

        // Mark any residual single tail as consumed.
        while (cursor < (int)history.size()) {
            if (!history[cursor].consumed) {
                releaseToBaseline(std::move(history[cursor].field));
            }
            markHistoryConsumed(cursor);
            ++cursor;
        }
    }
}

// Autosolve path: uses cadenceId metadata written by ld-cinemap to guide
// film frame reconstruction. Fields are paired by their definitional/
// complement relationship; spare fields are merged where available.
bool CadenceAssembler::tryExtractFilmFrameAtCursor()
{
    if (config.noCinemap) return false;

    const int i0 = nextUnconsumedIndex(cursor);
    if (i0 < 0) return false;

    SourceField &head = history[i0].field;
    const int cid0 = head.field.cinemap.cadenceId;
    if (!cadenceKnown(cid0)) return false;

    const int idx0    = cadenceIndex(cid0);
    const char letter = cadenceFilmLetter(cid0);

    auto roleOf = [](int cid) {
        const int idx = cadenceIndex(cid);
        if (idx == 0 || idx == 7) return 0;             // Def
        if (idx == 1 || idx == 6 || idx == 8) return 1; // Comp (incl D-comp)
        if (idx == 2 || idx == 5) return 2;             // Spare
        return 3;                                       // Other/mixed
    };

    const int role0 = roleOf(cid0);

    // A cut can discard A-def while preserving the immediately following
    // A-comp/A-spare pair.  CineMap records the comp as a provisional slot 1
    // (rather than manufacturing the missing definition) and the spare is a
    // solved slot 2.  Together they are a real complete A frame and must use
    // the frame regime.  The spare has no B ownership; B1 remains available
    // in its own capture pair for normal B reconstruction.
    if (idx0 == 1 && head.field.cinemap.isEditBoundary) {
        const int i1 = nextUnconsumedIndex(i0 + 1);
        if (i1 >= 0) {
            const SourceField &spareRef = history[i1].field;
            const int spareCid = spareRef.field.cinemap.cadenceId;
            if (cadenceKnown(spareCid) && cadenceIndex(spareCid) == 2 &&
                cadenceFilmLetter(spareCid) == 'A' &&
                !boundaryBetweenFields(head, spareRef)) {
                markHistoryConsumed(i0);
                markHistoryConsumed(i1);

                SourceField comp = std::move(history[i0].field);
                SourceField spare = std::move(history[i1].field);
                const bool swapped = orderPairForComb(comp, spare);

                WorkItem wi;
                wi.kind = config.export24p ? WorkItem::Kind::FilmFrame
                                            : WorkItem::Kind::TelecineFrame;
                wi.expansion = WorkItem::Expansion::None;
                wi.fieldsSwapped = swapped;
                wi.invertedFieldOrder = false;
                wi.filmLabel = 'A';
                wi.f1 = std::move(comp);
                wi.f2 = std::move(spare);
                syncStamp(wi.f1);
                syncStamp(wi.f2);
                workQueue.push_back(std::move(wi));
                return true;
            }
        }
    }

    // --- HANDLE SPARES AT HEAD ---------------------------------
    if (role0 == 2) {
        // CASE 1: C-Spare (Index 5) - LEADING Spare
        if (idx0 == 5) {
            int i1 = nextUnconsumedIndex(i0 + 1);
            if (i1 >= 0 && history[i1].field.field.cinemap.isEditBoundary) {
                if (config.export24p) {
                    SourceField discarded = std::move(history[i0].field);
                    markHistoryConsumed(i0);
                    releaseToBaseline(std::move(discarded));
                } else {
                    handOffCaptureFrameToBaseline(i0);
                }
                return true;
            }
            int i2 = (i1 >= 0) ? nextUnconsumedIndex(i1 + 1) : -1;

            // Mirror of the cut-truncated A head: C's spare is leading.
            // When the cut falls between C-comp and C-def, C-spare plus
            // C-comp remain a real complete C frame.  Retain the frame
            // regime without manufacturing the missing C-def or carrying
            // the cadence into the next scene.
            const bool cutAfterCComp =
                i1 >= 0 && i2 >= 0 &&
                cadenceIndex(history[i1].field.field.cinemap.cadenceId) == 6 &&
                cadenceFilmLetter(history[i1].field.field.cinemap.cadenceId) == 'C' &&
                boundaryBetweenFields(history[i1].field, history[i2].field);
            if (cutAfterCComp) {
                markHistoryConsumed(i0);
                markHistoryConsumed(i1);

                SourceField spare = std::move(history[i0].field);
                SourceField comp  = std::move(history[i1].field);
                const bool swapped = orderPairForComb(spare, comp);

                WorkItem wi;
                wi.kind = config.export24p ? WorkItem::Kind::FilmFrame
                                            : WorkItem::Kind::TelecineFrame;
                wi.expansion = WorkItem::Expansion::None;
                wi.fieldsSwapped = swapped;
                wi.invertedFieldOrder = false;
                wi.filmLabel = 'C';
                wi.f1 = std::move(spare);
                wi.f2 = std::move(comp);
                syncStamp(wi.f1);
                syncStamp(wi.f2);
                workQueue.push_back(std::move(wi));
                return true;
            }

            bool hasCBody =
                (i1 >= 0 && i2 >= 0 &&
                 cadenceIndex(history[i1].field.field.cinemap.cadenceId) == 6 &&
                 cadenceIndex(history[i2].field.field.cinemap.cadenceId) == 7 &&
                 cadenceFilmLetter(history[i1].field.field.cinemap.cadenceId) == 'C' &&
                 cadenceFilmLetter(history[i2].field.field.cinemap.cadenceId) == 'C');

            if (hasCBody) {
                markHistoryConsumed(i1);
                markHistoryConsumed(i2);

                SourceField comp  = std::move(history[i1].field); // 6
                SourceField def   = std::move(history[i2].field); // 7
                bool spareUsed = false;

                if (config.export24p) {
                    SourceField spare = std::move(history[i0].field); // 5
                    markHistoryConsumed(i0);
                    if (!config.dgDiscard && !dgDenySelective('C') &&
                        mergeDgPairWithSanityWrapper(def, spare, comp)) {
                        spareUsed = true;
                    } else {
                        releaseToBaseline(std::move(spare));
                    }
                } else {
                    SourceField spare = history[i0].field; // work on a copy; baseline retains original ownership
                    if (!config.dgDiscard && !dgDenySelective('C') &&
                        mergeDgPairWithSanityWrapper(def, spare, comp)) {
                        spareUsed = true;
                    }
                    handOffCaptureFrameToBaseline(i0);
                }

                bool swapped = orderPairForComb(comp, def);
                WorkItem wi;
                if (config.export24p) {
                    wi.kind = WorkItem::Kind::FilmFrame;
                    wi.expansion = WorkItem::Expansion::None;
                } else {
                    wi.kind = WorkItem::Kind::TelecineFrame;
                    wi.fieldsSwapped = swapped;
                    wi.expansion = spareUsed ? WorkItem::Expansion::Leading
                                             : WorkItem::Expansion::None;
                }
                wi.filmLabel = 'C';
                wi.f1 = std::move(comp);
                wi.f2 = std::move(def);
                wi.invertedFieldOrder = false;

                syncStamp(wi.f1);
        syncStamp(wi.f2);
        workQueue.push_back(std::move(wi));

                return true;
            }

            if (config.export24p) {
                markHistoryConsumed(i0);
                return true;
            }
            return false;
        }

        // CASE 2: A-Spare (Index 2) - TRAILING Spare
        if (idx0 == 2) {
            if (config.export24p) {
                SourceField discarded = std::move(history[i0].field);
                markHistoryConsumed(i0);
                releaseToBaseline(std::move(discarded));
                return true;
            }
            return false;
        }
    }

    // --- STANDARD PAIR EXTRACTION ------------------------------------------
    auto findComplementPos = [&](int pos) -> int {
        const auto &hf = history[pos].field;
        const int cid  = hf.field.cinemap.cadenceId;
        if (!cadenceKnown(cid)) return -1;

        const int idx     = cadenceIndex(cid);
        const int compCid = filmFrameComplement(cid);
        const char let    = cadenceFilmLetter(cid);

        int delta = 0;
        switch (idx) {
            case 0: delta = +1; break;
            case 1: delta = -1; break;
            case 2: delta = -1; break;
            case 3: delta = +1; break;
            case 4: delta = -1; break;
            case 5: delta = +1; break;
            case 6: delta = +1; break;
            case 7: delta = -1; break;
            case 8: delta = +1; break;
            case 9: delta = -1; break;
            default: return -1;
        }

        const int targetSeq = hf.field.seqNo + delta;
        auto it = seqNoToHistoryIndex.find(targetSeq);
        if (it == seqNoToHistoryIndex.end()) return -1;
        const int matePos = it.value();
        if (history[matePos].consumed) return -1;
        const auto &cand = history[matePos].field;
        if (!cadenceKnown(cand.field.cinemap.cadenceId)) return -1;
        if (cadenceFilmLetter(cand.field.cinemap.cadenceId) != let) return -1;
        if (cand.field.cinemap.cadenceId != compCid) return -1;
        return matePos;
    };

    const int matePos = findComplementPos(i0);
    if (matePos < 0) return false;

    const SourceField &aRef = history[i0].field;
    const SourceField &bRef = history[matePos].field;

    int prevPos = i0;
    int nextPos = matePos;
    if (bRef.field.seqNo < aRef.field.seqNo) {
        prevPos = matePos;
        nextPos = i0;
    }

    if (boundaryBetweenFields(history[prevPos].field, history[nextPos].field)) {
        markHistoryConsumed(i0);
        markHistoryConsumed(matePos);

        SourceField a = std::move(history[i0].field);
        SourceField b = std::move(history[matePos].field);

        orderPairForComb(a, b);
        a.allowProgressiveFrameRegime = false;
        b.allowProgressiveFrameRegime = false;
        a.field.cinemap.cadenceId = -2;
        b.field.cinemap.cadenceId = -2;

        WorkItem wi;
        wi.kind               = WorkItem::Kind::PassthroughFrame;
        wi.expansion          = WorkItem::Expansion::None;
        wi.fieldsSwapped      = false;
        wi.invertedFieldOrder = false;
        wi.filmLabel          = '?';
        wi.f1                 = std::move(a);
        wi.f2                 = std::move(b);
        syncStamp(wi.f1);
        syncStamp(wi.f2);
        workQueue.push_back(std::move(wi));

        return true;
    }

    // Extract pair
    markHistoryConsumed(i0);
    markHistoryConsumed(matePos);

    SourceField fA = std::move(history[i0].field);
    SourceField fB = std::move(history[matePos].field);

    SourceField *def  = nullptr;
    SourceField *comp = nullptr;
    if (roleOf(fA.field.cinemap.cadenceId) == 0) { def = &fA; comp = &fB; }
    else if (roleOf(fB.field.cinemap.cadenceId) == 0) { def = &fB; comp = &fA; }
    else { def = &fA; comp = &fB; }

    bool spareUsed = false;
    auto tryConsumeSpare = [&](int pos) {
        if (pos < 0 || pos >= history.size()) return;
        if (history[pos].consumed) return;
        const SourceField &candRef = history[pos].field;
        if (!cadenceKnown(candRef.field.cinemap.cadenceId)) return;
        if (cadenceFilmLetter(candRef.field.cinemap.cadenceId) != letter) return;
        if (roleOf(candRef.field.cinemap.cadenceId) != 2) return;

        if (letter == 'A' && candRef.field.cinemap.isEditBoundary) return;
        if (letter == 'C' && head.field.cinemap.isEditBoundary) return;

        if (config.export24p) {
            markHistoryConsumed(pos);
            SourceField spare = std::move(history[pos].field);

            if (!config.dgDiscard && !dgDenySelective(letter)) {
                SourceField defCopy   = *def;
                SourceField spareCopy = spare;
                SourceField compCopy  = *comp;
                if (mergeDgPairWithSanityWrapper(defCopy, spareCopy, compCopy)) {
                    *def  = std::move(defCopy);
                    *comp = std::move(compCopy);
                    spareUsed = true;
                    return;
                }
            }
            releaseToBaseline(std::move(spare));
            return;
        }

        SourceField spare = history[pos].field;
        if (!config.dgDiscard && !dgDenySelective(letter)) {
            SourceField defCopy   = *def;
            SourceField spareCopy = spare;
            SourceField compCopy  = *comp;
            if (mergeDgPairWithSanityWrapper(defCopy, spareCopy, compCopy)) {
                *def  = std::move(defCopy);
                *comp = std::move(compCopy);
                spareUsed = true;
            }
        }
        handOffCaptureFrameToBaseline(pos);
    };
    
    // Normalize baseSeq to the definitional anchor for spare lookup:
    // A: always relative to Adef (idx 0); if we have Acomp (idx 1), step back 1.
    // C: always relative to Cdef (idx 7); if we have Ccomp (idx 6), step forward 1.
    const int baseSeq = (idx0 == 1) ? head.field.seqNo - 1  // Acomp -> Adef
                      : (idx0 == 6) ? head.field.seqNo + 1  // Ccomp -> Cdef
                      : head.field.seqNo;                    // Adef or Cdef already

    if (letter == 'A') {
        auto it = seqNoToHistoryIndex.find(baseSeq + 2);
        if (it != seqNoToHistoryIndex.end()) tryConsumeSpare(it.value());
    } else if (letter == 'C') {
        auto it = seqNoToHistoryIndex.find(baseSeq - 2);
        if (it != seqNoToHistoryIndex.end()) tryConsumeSpare(it.value());
    }
    // Emit
    bool swapped = orderPairForComb(fA, fB);
    WorkItem wi;
    if (config.export24p) {
        wi.kind = WorkItem::Kind::FilmFrame;
        wi.expansion = WorkItem::Expansion::None;
    } else {
        wi.kind = WorkItem::Kind::TelecineFrame;
        wi.fieldsSwapped = swapped;
        if (letter == 'A' && spareUsed)      wi.expansion = WorkItem::Expansion::Trailing;
        else if (letter == 'C' && spareUsed) wi.expansion = WorkItem::Expansion::Leading;
        else                                 wi.expansion = WorkItem::Expansion::None;
    }
    wi.filmLabel = letter;
    wi.f1 = std::move(fA);
    wi.f2 = std::move(fB);
    wi.invertedFieldOrder = false;
    syncStamp(wi.f1);
        syncStamp(wi.f2);
        workQueue.push_back(std::move(wi));

    return true;
}

// Fallback for when the film process doesn't succeed. Video frames must be
// delivered in any event, as plain video if not film.
bool CadenceAssembler::tryEmitPassthroughAtCursor(bool flushMode, bool force)
{
    (void)flushMode;

    int i0 = nextUnconsumedIndex(cursor);
    if (i0 < 0) return false;

    const qint32 seqNo = history[i0].field.field.seqNo;
    if (baselineOwnedSeqNos.contains(seqNo)) {
        markHistoryConsumed(i0);
        releaseHistoryPayload(i0);
        return true;
    }

    const int partnerSeqNo = history[i0].capturePartnerSeqNo;
    int i1 = -1;

    if (partnerSeqNo >= 0) {
        auto it = seqNoToHistoryIndex.find(partnerSeqNo);
        if (it != seqNoToHistoryIndex.end() && !history[it.value()].consumed)
            i1 = it.value();
    }

    // Partner not available — orphaned field, release to baseline.
    if (i1 < 0) {
        if (!config.export24p || force) {
            handOffCaptureFrameToBaseline(i0);
            return true;
        }
        markHistoryConsumed(i0);
        releaseToBaseline(std::move(history[i0].field));
        return true;
    }

    markHistoryConsumed(i0);
    markHistoryConsumed(i1);

    SourceField a = std::move(history[i0].field);
    SourceField b = std::move(history[i1].field);

    orderPairForComb(a, b);
    a.allowProgressiveFrameRegime = false;
    b.allowProgressiveFrameRegime = false;

    WorkItem wi;
    wi.kind               = WorkItem::Kind::PassthroughFrame;
    wi.expansion          = WorkItem::Expansion::None;
    wi.fieldsSwapped      = false;
    wi.filmLabel          = '?';
    wi.f1                 = std::move(a);
    wi.f2                 = std::move(b);
    wi.invertedFieldOrder = false;

    syncStamp(wi.f1);
        syncStamp(wi.f2);
        workQueue.push_back(std::move(wi));
    return true;
}
