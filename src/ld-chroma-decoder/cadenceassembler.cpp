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
#include "cadencedefs.h"
#include "combmath.h"
#include "tbc/logging.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

    struct BurstInfo { double bsin; double bcos; };
    
    static inline BurstInfo detectBurstForDgMerge(const quint16 *lineData,
                                                  const LdDecodeMetaData::VideoParameters &vp)
    {
        double bsin = 0.0, bcos = 0.0;
        for (int i = vp.colourBurstStart; i < vp.colourBurstEnd; ++i) {
            const double s = lineData[i];
            bsin += s * sin4fsc(i);
            bcos += s * cos4fsc(i);
        }
        const int len = vp.colourBurstEnd - vp.colourBurstStart;
        if (len > 0) {
            const double invLen = 1.0 / len;
            bsin *= invLen;
            bcos *= invLen;
        }
        double mag = std::sqrt(bsin * bsin + bcos * bcos);
        if (mag > 1e-9) {
            const double invMag = 1.0 / mag;
            bsin *= invMag;
            bcos *= invMag;
        } else {
            bsin = 0.0;
            bcos = 1.0;
        }
        return {bsin, bcos};
    }
    
    static inline BurstInfo combineDefWeightedBasis(const BurstInfo& defB,
                                                   const BurstInfo& spareB)
    {
        double bsin = 2.0 * defB.bsin + spareB.bsin;
        double bcos = 2.0 * defB.bcos + spareB.bcos;
        const double mag = std::sqrt(bsin*bsin + bcos*bcos);
        if (mag > 1e-12) { bsin /= mag; bcos /= mag; }
        else { bsin = 0.0; bcos = 1.0; }
        return {bsin, bcos};
    }
    
    static inline void demodIQ(double v, int h, double bcos, double bsin,
                               double& outI, double& outQ)
    {
        const double lsin = v * sin4fsc(h) * 2.0;
        const double lcos = v * cos4fsc(h) * 2.0;
        outI = (lsin * bcos - lcos * bsin);
        outQ = (lsin * bsin + lcos * bcos);
    }
    
    static inline double hypot2(double a, double b) { return std::sqrt(a*a + b*b); }
    
    static inline quint16 clampU16(double v)
    {
        if (!(v == v)) return 0;
        if (v < 0.0) return 0;
        if (v > 65535.0) return 65535;
        return static_cast<quint16>(std::llround(v));
    }
    
    // Merge a doplGang A/C spare field into its definitional partner.
    //
    // The merge is performed in IQ (demodulated chroma) space rather than
    // raw pixel space because the spare field carries a 180°-shifted subcarrier
    // relative to the definitional field. A direct pixel average would cancel
    // chroma; demodulating to IQ first allows the two fields' chroma to be
    // compared and averaged coherently.
    //
    // Pass 1 counts how many samples differ by more than the outlier threshold
    // in IQ distance. If the outlier fraction exceeds cfg.dgMaxOutlierFrac the
    // pair is rejected (the fields are not genuine twins) and no merge occurs.
    //
    // Pass 2 performs the actual merge: where the def and spare agree within
    // threshold, their raw pixels are averaged directly. Where they disagree,
    // the complement field lines immediately above and below are demodulated
    // and the def/spare sample closer to that reference is kept, discarding
    // the other.
    static bool mergeDgPairWithSanity(const LdDecodeMetaData::VideoParameters& vp,
                                     const CadenceAssembler::Configuration& cfg,
                                     SourceField& def,
                                     SourceField& spare,
                                     SourceField& comp)
    {
        const int width = vp.fieldWidth;
        if (width <= 0) return false;
    
        // Treat SourceField::data as a raw byte buffer of 16-bit samples.
        constexpr int bytesPerSample = int(sizeof(quint16));
    
        if (def.data.size() != spare.data.size()) return false;
        if (def.data.size() != comp.data.size()) return false;
        if (def.data.size() <= 0) return false;
        if ((def.data.size() % bytesPerSample) != 0) return false;
    
        const int samples = def.data.size() / bytesPerSample;
        if ((samples % width) != 0) return false;
    
        const int height = samples / width;
        if (height <= 0) return false;
    
        int activeLeft  = vp.activeVideoStart;
        int activeRight = vp.activeVideoEnd;
    
        // Clamp active window to buffer width.
        activeLeft  = std::clamp(activeLeft,  0, width);
        activeRight = std::clamp(activeRight, 0, width);
        if (activeLeft >= activeRight) return false;
    
        const int y0     = std::clamp(def.getFirstActiveLine(vp),   0, height);
        const int y1     = std::clamp(def.getLastActiveLine(vp),    0, height);
        const int compY0 = std::clamp(comp.getFirstActiveLine(vp),  0, height);
        const int compY1 = std::clamp(comp.getLastActiveLine(vp),   0, height);
        if (y0 >= y1 || compY0 >= compY1) return false;
    
        const double ireScale = (vp.white16bIre - vp.black16bIre) / 100.0;
        if (!(ireScale > 0.0)) return false;
    
        // Threshold expressed in sample/code units (same space as input pixels).
        const double outlierThreshCode = cfg.dgOutlierThreshIre * ireScale;
        const double maxOutlierFrac    = std::clamp(cfg.dgMaxOutlierFrac, 0.0, 1.0);
        if (!(outlierThreshCode >= 0.0)) return false;
    
        // In this pipeline's demodIQ (as used by dg merge), IQ distance is 2x code distance:
        // lsin/lcos have factor 2.0 and sin/cos are unit magnitude -> |dIQ| = 2*|dv|.
        const double outlierThreshIQ = 2.0 * outlierThreshCode;
    
        const quint16* defp   = reinterpret_cast<const quint16*>(def.data.constData());
        const quint16* sparep = reinterpret_cast<const quint16*>(spare.data.constData());
        const quint16* compp  = reinterpret_cast<const quint16*>(comp.data.constData());
    
        std::int64_t total    = 0;
        std::int64_t outliers = 0;
    
        // Pass 1: Analyze (cheap twin sanity in the same "code-derived" space via outlierThreshIQ)
        for (int lf = y0; lf < y1; ++lf) {
            const quint16* defLine   = defp   + lf * width;
            const quint16* spareLine = sparep + lf * width;
    
            const BurstInfo bDef   = detectBurstForDgMerge(defLine, vp);
            const BurstInfo bSpare = detectBurstForDgMerge(spareLine, vp);
            const BurstInfo b      = combineDefWeightedBasis(bDef, bSpare);
    
            for (int h = activeLeft; h < activeRight; ++h) {
                double ID, QD, IS, QS;
                demodIQ((double)defLine[h],   h, b.bcos, b.bsin, ID, QD);
                demodIQ((double)spareLine[h], h, b.bcos, b.bsin, IS, QS);
    
                // IQ-distance threshold (equivalent to code-distance threshold)
                if (hypot2(ID - IS, QD - QS) > outlierThreshIQ) outliers++;
                total++;
            }
        }
    
        if (total <= 0) return false;
        const double outlierFrac = double(outliers) / double(total);
        if (outlierFrac > maxOutlierFrac) return false;
    
        // Pass 2: Merge
        quint16* defw   = reinterpret_cast<quint16*>(def.data.data());
        quint16* sparew = reinterpret_cast<quint16*>(spare.data.data());
        const bool defIsFirst = def.field.isFirstField;
    
        for (int lf = y0; lf < y1; ++lf) {
            quint16* defLine   = defw   + lf * width;
            quint16* spareLine = sparew + lf * width;
    
            const BurstInfo bDef   = detectBurstForDgMerge(defLine, vp);
            const BurstInfo bSpare = detectBurstForDgMerge(spareLine, vp);
            const BurstInfo b      = combineDefWeightedBasis(bDef, bSpare);
    
            int compLfUp = defIsFirst ? (lf - 1) : lf;
            int compLfDn = defIsFirst ? lf       : (lf + 1);
            compLfUp = std::clamp(compLfUp, compY0, compY1 - 1);
            compLfDn = std::clamp(compLfDn, compY0, compY1 - 1);
    
            const quint16* compUpLine = compp + compLfUp * width;
            const quint16* compDnLine = compp + compLfDn * width;
    
            for (int h = activeLeft; h < activeRight; ++h) {
                double ID, QD, IS, QS;
                demodIQ((double)defLine[h],   h, b.bcos, b.bsin, ID, QD);
                demodIQ((double)spareLine[h], h, b.bcos, b.bsin, IS, QS);
    
                const double dDS_IQ = hypot2(ID - IS, QD - QS);
    
                quint16 outv = 0;
                if (dDS_IQ <= outlierThreshIQ) {
                    outv = clampU16(0.5 * ((double)defLine[h] + (double)spareLine[h]));
                } else {
                    double IU, QU, IDn, QDn;
                    demodIQ((double)compUpLine[h], h, b.bcos, b.bsin, IU,  QU);
                    demodIQ((double)compDnLine[h], h, b.bcos, b.bsin, IDn, QDn);
    
                    if ((IU * IDn + QU * QDn) < 0.0) { IDn = -IDn; QDn = -QDn; }
    
                    const double IR = 0.5 * (IU + IDn);
                    const double QR = 0.5 * (QU + QDn);
    
                    const double dDef = hypot2(ID - IR, QD - QR);
                    const double dSp  = hypot2(IS - IR, QS - QR);
    
                    outv = (dDef <= dSp) ? defLine[h] : spareLine[h];
                }
    
                defLine[h]   = outv;
                spareLine[h] = outv;
            }
        }
    
        return true;
    }

} // namespace

// CadenceAssembler — telecine pulldown consolidation for ld-chroma-decoder.
//
// CadenceAssembler accepts a stream of SourceFields tagged with cadence
// metadata from ld-cinemap and assembles them into WorkItems for the comb
// decoder. Three operating modes are supported:
//
// Autosolve (default): uses cadenceId assignments written by ld-cinemap to
//   identify and pair definitional, complement, and spare fields into film
//   frames. Spare fields (A-trailing, C-leading) are merged into the
//   definitional field via dG pixel averaging where the sanity check passes,
//   or released to baseline passthrough if not.
//
// Forced cadence (--set-cadence): bypasses ld-cinemap's solve entirely and
//   imposes a naive A-B-C-D pattern on the incoming field stream. Useful when
//   cadence metadata is absent or unreliable.
//
// Output path: in normal telecine mode, WorkItems carry TelecineFrame or
//   PassthroughFrame kind and are delivered to DecoderPool for comb decoding.
//   With --export-24p, film frames are emitted as FilmFrame kind (one per
//   unique film frame, no spare expansion). Fields that cannot be placed into
//   a film frame are released to baseline so DecoderPool can emit them as
//   plain video without gaps in the output stream.
//
// CadenceAssembler Implementation
CadenceAssembler::CadenceAssembler(const LdDecodeMetaData::VideoParameters& vp,
                                   const Configuration& cfg,
                                   std::function<void(qint32)> onBaseline)
    : videoParameters(vp), config(cfg),
      onFieldReleasedToBaseline(std::move(onBaseline))
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

void CadenceAssembler::push(const QVector<SourceField>& newFields)
{
    if (config.setCadence != 0 && !config.noPA) {
        for (const auto& f : newFields) window.push_back(f);
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
    if (config.setCadence != 0 && !config.noPA) {
        processWindowForced(true);
    } else {
        processHistory(true);
    }
}

bool CadenceAssembler::hasWork() const { return !workQueue.empty(); }

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
    return mergeDgPairWithSanity(videoParameters, config, def, spare, comp);
}

// Forced cadence start offset helper
int CadenceAssembler::forcedStartIndex() const
{
    int base = 0;
    switch (config.setCadence) {
        case 1: base = 0; break; // AA
        case 2: base = 2; break; // AB
        case 3: base = 4; break; // BC
        case 4: base = 6; break; // CC
        case 5: base = 8; break; // DD
        default: base = 0; break;
    }

    // -r (reverseFieldOrder) shifts the starting slot by half a cycle (5 slots).
    // In normal field order the upper (first-stored) field carries the A-definitional
    // sample; reversing field order makes the lower field the first-stored, which
    // flips which fields within each pulldown frame are the spares versus which are
    // the B-frame fields — the core distinction in pulldown consolidation.
    if (config.reverseFieldOrder) base = (base + 5) % CADENCE_NTSC_CYCLE;
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

// Forced cadence (jam) mode: the user insists on a specific cadence pattern
// regardless of what ld-cinemap detected. Fields are consumed in strict
// A-B-C-D sequence driven by setCadence (which sets the starting slot) with
// no cadenceId validation — every incoming field is assigned its position by
// counting, not by metadata. This is useful when cadence metadata is absent
// or when the user knows the disc has a simple, unbroken pulldown cadence.
void CadenceAssembler::processWindowForced(bool flushMode)
{
    if (config.noPA) return;

    const int start = forcedStartIndex();

    auto cycleIndex = [&](qint64 consumed) -> int {
        int idx = int((start + (consumed % CADENCE_NTSC_CYCLE)) % CADENCE_NTSC_CYCLE);
        if (idx < 0) idx += CADENCE_NTSC_CYCLE;
        return idx; // 0..9
    };

    auto emitTelecine2 = [&](char label, SourceField&& f1, SourceField&& f2,
                             WorkItem::Expansion ex = WorkItem::Expansion::None) {
        bool swapped = orderPairForComb(f1, f2);
        WorkItem wi;
        wi.kind               = WorkItem::Kind::TelecineFrame;
        wi.expansion          = ex;
        wi.fieldsSwapped      = swapped;
        wi.invertedFieldOrder = false;
        wi.filmLabel          = label;
        wi.f1                 = std::move(f1);
        wi.f2                 = std::move(f2);
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

    // Main loop: perform only the exact cadence-defined operation for the current slot.
    while (true) {
        if (window.empty()) break;
        if (!flushMode && window.size() < 2) break;

        const int idx = cycleIndex(forcedFieldIndex);

        // A start: 0(def),1(comp), optional 2(spare)
        if (idx == 0) {
            if (window.size() < 2) break;
            auto [def, comp] = pop2();

            WorkItem::Expansion ex = WorkItem::Expansion::None;

            // If the next slot is 2, consume it as A-spare (if present).
            if (!window.empty() && cycleIndex(forcedFieldIndex) == 2) {
                SourceField spare = pop1();
                if (!config.dgDiscard && mergeDgPairWithSanityWrapper(def, spare, comp)) {
                    ex = WorkItem::Expansion::Trailing;
                } else {
                    releaseToBaseline(std::move(spare));
                }
            }

            emitTelecine2('A', std::move(def), std::move(comp), ex);
            continue;
        }

        // A spare slot (2): consume+discard when encountered standalone.
        if (idx == 2) {
            releaseToBaseline(pop1());
            continue;
        }

        // B start: 3+4
        if (idx == 3) {
            if (window.size() < 2) break;
            auto [f3, f4] = pop2();
            emitTelecine2('B', std::move(f3), std::move(f4), WorkItem::Expansion::None);
            continue;
        }

        // C start (preferred): 5(spare)+6(comp)+7(def)
        if (idx == 5) {
            if (window.size() >= 3) {
                SourceField spare = pop1(); // 5
                SourceField comp  = pop1(); // 6
                SourceField def   = pop1(); // 7
        
                WorkItem::Expansion ex = WorkItem::Expansion::None;
                if (!config.dgDiscard && mergeDgPairWithSanityWrapper(def, spare, comp)) {
                    ex = WorkItem::Expansion::Leading;
                } else {
                    releaseToBaseline(std::move(spare));
                }
                emitTelecine2('C', std::move(comp), std::move(def), ex);
                continue;
            }
        
            if (flushMode) {
                while (!window.empty()) releaseToBaseline(pop1());
            }
            break;
        }
        
        // C start (no spare): 6+7
        if (idx == 6) {
            if (window.size() < 2) break;
            auto [comp, def] = pop2();
            emitTelecine2('C', std::move(comp), std::move(def), WorkItem::Expansion::None);
            continue;
        }

        // D start: 8+9
        if (idx == 8) {
            if (window.size() < 2) break;
            auto [f8, f9] = pop2();
            emitTelecine2('D', std::move(f8), std::move(f9), WorkItem::Expansion::None);
            continue;
        }

        // Any other slot: do nothing. No fallback, no invention.
        // In non-flush, stop and wait (because you need more input to reach a defined slot).
        // In flush, discard tail.
        if (flushMode) {
            window.clear();
        }
        break;
    }

    // Flush: drain remaining using only defined operations; discard any leftover tail.
    if (flushMode) {
        while (!window.empty()) {
            const int idx = cycleIndex(forcedFieldIndex);

            if (idx == 2) { releaseToBaseline(pop1()); continue; }

            if (idx == 0) {
                if (window.size() < 2) break;
                auto [def, comp] = pop2();

                WorkItem::Expansion ex = WorkItem::Expansion::None;
                if (!window.empty() && cycleIndex(forcedFieldIndex) == 2) {
                    SourceField spare = pop1();
                    if (!config.dgDiscard && mergeDgPairWithSanityWrapper(def, spare, comp)) {
                        ex = WorkItem::Expansion::Trailing;
                    } else {
                        releaseToBaseline(std::move(spare));
                    }
                }
                emitTelecine2('A', std::move(def), std::move(comp), ex);
                continue;
            }

            if (idx == 3) {
                if (window.size() < 2) break;
                auto [f3, f4] = pop2();
                emitTelecine2('B', std::move(f3), std::move(f4), WorkItem::Expansion::None);
                continue;
            }

            if (idx == 5) {
                if (window.size() >= 3) {
                    SourceField spare = pop1();
                    SourceField comp  = pop1();
                    SourceField def   = pop1();
            
                    WorkItem::Expansion ex = WorkItem::Expansion::None;
                    if (!config.dgDiscard && mergeDgPairWithSanityWrapper(def, spare, comp)) {
                        ex = WorkItem::Expansion::Leading;
                    } else {
                        releaseToBaseline(std::move(spare));
                    }
                    emitTelecine2('C', std::move(comp), std::move(def), ex);
                    continue;
                }
                // Incomplete triple at flush — release whatever's left
                while (!window.empty()) releaseToBaseline(pop1());
                break;
            }
            if (idx == 6) {
                if (window.size() < 2) break;
                auto [comp, def] = pop2();
                emitTelecine2('C', std::move(comp), std::move(def), WorkItem::Expansion::None);
                continue;
            }

            if (idx == 8) {
                if (window.size() < 2) break;
                auto [f8, f9] = pop2();
                emitTelecine2('D', std::move(f8), std::move(f9), WorkItem::Expansion::None);
                continue;
            }
            while (!window.empty()) releaseToBaseline(pop1());
            window.clear();
            break;
        }

        window.clear();
    }
}

void CadenceAssembler::processHistory(bool flushMode)
{
    // "Enough buffered" to conclude we're stuck and must force passthrough.
    constexpr int kStuckBufferFields = 8;

    auto haveAtLeastNUnconsumedFrom = [&](int start, int needed) -> bool {
        int count = 0;
        for (int i = std::max(0, start); i < (int)history.size(); ++i) {
            if (!history[i].consumed) {
                if (++count >= needed) return true;
            }
        }
        return false;
    };

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
    if (config.noPA) return false;

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

    // --- HANDLE SPARES AT HEAD ---------------------------------
    if (role0 == 2) {
        // CASE 1: C-Spare (Index 5) - LEADING Spare
        if (idx0 == 5) {
            int i1 = nextUnconsumedIndex(i0 + 1);
            if (i1 >= 0 && history[i1].field.field.cinemap.isEditBoundary) {
                SourceField discarded = std::move(history[i0].field);
                markHistoryConsumed(i0);
                releaseToBaseline(std::move(discarded));
                return true;
            }
            int i2 = (i1 >= 0) ? nextUnconsumedIndex(i1 + 1) : -1;
            bool hasCBody =
                (i1 >= 0 && i2 >= 0 &&
                 cadenceIndex(history[i1].field.field.cinemap.cadenceId) == 6 &&
                 cadenceIndex(history[i2].field.field.cinemap.cadenceId) == 7 &&
                 cadenceFilmLetter(history[i1].field.field.cinemap.cadenceId) == 'C' &&
                 cadenceFilmLetter(history[i2].field.field.cinemap.cadenceId) == 'C');

            if (hasCBody) {
                markHistoryConsumed(i0);
                markHistoryConsumed(i1);
                markHistoryConsumed(i2);

                SourceField spare = std::move(history[i0].field); // 5
                SourceField comp  = std::move(history[i1].field); // 6
                SourceField def   = std::move(history[i2].field); // 7

                if (!config.dgDiscard && !mergeDgPairWithSanityWrapper(def, spare, comp)) {
                    releaseToBaseline(std::move(spare));
                } else if (config.dgDiscard) {
                    releaseToBaseline(std::move(spare));
                }

                bool swapped = orderPairForComb(comp, def);
                WorkItem wi;
                if (config.export24p) {
                    wi.kind = WorkItem::Kind::FilmFrame;
                    wi.expansion = WorkItem::Expansion::None;
                } else {
                    wi.kind = WorkItem::Kind::TelecineFrame;
                    wi.fieldsSwapped = swapped;
                    wi.expansion = WorkItem::Expansion::Leading;
                }
                wi.filmLabel = 'C';
                wi.f1 = std::move(comp);
                wi.f2 = std::move(def);
                wi.invertedFieldOrder = false;

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

        markHistoryConsumed(pos);
        SourceField spare = std::move(history[pos].field);

        if (!config.dgDiscard) {
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
    workQueue.push_back(std::move(wi));

    return true;
}

// Fallback for when the film process doesn't succeed. Video frames must be
// delivered in any event, as plain video if not film.
bool CadenceAssembler::tryEmitPassthroughAtCursor(bool flushMode, bool force)
{
    (void)flushMode;
    (void)force;

    int i0 = nextUnconsumedIndex(cursor);
    if (i0 < 0) return false;

    const int partnerSeqNo = history[i0].capturePartnerSeqNo;
    int i1 = -1;

    if (partnerSeqNo >= 0) {
        auto it = seqNoToHistoryIndex.find(partnerSeqNo);
        if (it != seqNoToHistoryIndex.end() && !history[it.value()].consumed)
            i1 = it.value();
    }

    // Partner not available — orphaned field, release to baseline.
    if (i1 < 0) {
        markHistoryConsumed(i0);
        releaseToBaseline(std::move(history[i0].field));
        return true;
    }

    markHistoryConsumed(i0);
    markHistoryConsumed(i1);

    SourceField a = std::move(history[i0].field);
    SourceField b = std::move(history[i1].field);

    orderPairForComb(a, b);

    WorkItem wi;
    wi.kind               = WorkItem::Kind::PassthroughFrame;
    wi.expansion          = WorkItem::Expansion::None;
    wi.fieldsSwapped      = false;
    wi.filmLabel          = '?';
    wi.f1                 = std::move(a);
    wi.f2                 = std::move(b);
    wi.invertedFieldOrder = false;

    workQueue.push_back(std::move(wi));
    return true;
}