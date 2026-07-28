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
    // opposite subcarrier phase, so per sample the conservation identities
    // hold exactly:
    //     luma    Lhat = (def + spare) / 2
    //     carrier chat = (def - spare) / 2   -- already in DEF phase
    // No burst detection, demodulation, or sign decision is needed: chat IS
    // the def-phase carrier by construction. The merged sample is
    //     merged = Lhat + BP(chat)
    // where BP is a linear-phase carrier-band filter (taps at 0/+-2/+-4)
    // with H(fsc) = H(fsc +- 1.3 MHz) = 1 and H(DC) = H(2fsc) = 0. Signal
    // content is preserved on both sides of the split -- carrier-band luma
    // (thin lines) stays in Lhat while the carrier stays in BP(chat), which
    // a single capture cannot separate -- and twin noise is averaged (-3 dB)
    // everywhere outside the carrier band. There is deliberately no
    // per-sample branch: an earlier form selected raw def-or-spare samples
    // by IQ distance to a complement reference, which inserted 180-degree-
    // shifted spare carrier into saturated chroma (dotted tractor beam).
    //
    // Sanity (pass 1): for true twins the out-of-band part of chat is pure
    // twin noise, so |chat - BP(chat)| above threshold marks real mismatch
    // (motion, edit, mis-pairing). Reject the pair when the outlier
    // fraction exceeds cfg.dgMaxOutlierFrac and no merge occurs.
    //
    // Per-sample error correction (user design, 2026-07-28): whole-merge
    // rejection stays for GROSS mismatch (a large share of the image
    // differs -- the merge would hurt more than help). For the minor
    // disagreements that survive the gate (dropouts, disc errors), the
    // out-of-band residual localises the error and the COMPLEMENT field
    // arbitrates which twin is wrong: the twin more dissimilar to the
    // average of the comp pixels above and below is held to be the error,
    // and the other twin's data stands. Phase law of the comparison: the
    // comp lines bracketing a def line both carry carrier ANTIPHASE to the
    // def line (adjacent frame lines are 180 degrees; the two comp
    // neighbours are mutually cophased), so the raw comp average R is a
    // fair reference for SPARE directly, while DEF must be compared to R
    // with its carrier flipped -- 2*boxcar(R) - R, the boxcar being
    // carrier-free. Without the flip the test is biased toward spare
    // wherever carrier is vertically coherent, which would re-insert
    // 180-degree carrier through the arbitration (the old dotted-beam bug
    // by another door). Errors can be luma or colour; the comparison sees
    // both because it is per-sample against a phase-matched reference.
    // Repair at an error sample: the good twin supplies its half of the
    // conservation pair, the carrier is refilled from the nearest valid
    // same-phase samples (+-4k, one full cycle apart -- lawful under the
    // bandwidth law), and the exact-carrier side channel is DENIED within
    // the BP aperture of any error (no conservation fact there).
    //
    // Emits the exact-carrier side channel: exact = merged - Lhat = BP(chat),
    // the carrier of the emitted sample as a conservation fact.
    static bool mergeDgPairWithSanity(const LdDecodeMetaData::VideoParameters& vp,
                                     const CadenceAssembler::Configuration& cfg,
                                     SourceField& def,
                                     SourceField& spare,
                                     SourceField& comp)
    {

        const int width = vp.fieldWidth;
        if (width <= 0) return false;

        // SourceField::data is QVector<quint16>: size() is the SAMPLE count.
        // (A previous form treated it as a byte buffer and divided by
        // sizeof(quint16); the geometry guard below then always tripped and
        // the merge was silently dead.)
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
        const double outlierThreshCode = cfg.dgOutlierThreshIre * ireScale;
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

        // Pass 1: twin sanity on the out-of-band residual of chat.
        std::int64_t total = 0, outliers = 0;
        for (int lf = y0; lf < y1; ++lf) {
            buildLine(lf);
            for (int h = activeLeft; h < activeRight; ++h) {
                if (std::fabs(chat[h] - bp[h]) > outlierThreshCode) outliers++;
                total++;
            }
        }
        if (total <= 0) return false;
        const double outlierFrac = double(outliers) / double(total);
        static const bool dsDebug = std::getenv("LDCD_PROBE_DSREF") != nullptr;
        if (outlierFrac > maxOutlierFrac) {
            if (dsDebug)
                std::fprintf(stderr, "DSREF-MERGE reject seq=%d frac=%.3f\n",
                             def.field.seqNo, outlierFrac);
            return false;
        }
        if (dsDebug)
            std::fprintf(stderr, "DSREF-MERGE accept seq=%d frac=%.3f\n",
                         def.field.seqNo, outlierFrac);

        // Twin-agreement audit (LDCD_DUMP_TWIN_L0/L1/C0/C1, field-line and
        // FULL-raw-column coordinates; run -t 1). The (D-S)/2 channel is a
        // carrier measurement ONLY where the twins share content: the BP
        // makes the emitted value carrier-BAND by construction, so band
        // structure alone proves nothing. The out-of-band residual
        // |chat - BP(chat)| is the honest witness -- for true twins it is
        // just twin noise, and where content differs (a video-rate element
        // composited over film-rate frames, say) it is large and the
        // "carrier" reading there is a content difference in disguise.
        {
            static const auto tEnv = [](const char *n) {
                const char *s = std::getenv(n); return s ? std::atoi(s) : -1;
            };
            static const int tL0 = tEnv("LDCD_DUMP_TWIN_L0");
            static const int tL1 = tEnv("LDCD_DUMP_TWIN_L1");
            static const int tC0 = tEnv("LDCD_DUMP_TWIN_C0");
            static const int tC1 = tEnv("LDCD_DUMP_TWIN_C1");
            if (tL0 >= 0 && tC0 >= 0) {
                for (int lf = std::max(y0, tL0); lf < std::min(y1, tL1 + 1); ++lf) {
                    buildLine(lf);
                    for (int h = std::max(activeLeft, tC0);
                         h < std::min(activeRight, tC1 + 1); ++h) {
                        std::fprintf(stderr,
                            "TWIN seq=%d lf=%d h=%d lhat=%.1f chat=%.2f "
                            "bp=%.2f oob=%.2f ire=%.4f\n",
                            def.field.seqNo, lf, h, lhat[h],
                            chat[h] / ireScale, bp[h] / ireScale,
                            (chat[h] - bp[h]) / ireScale, ireScale);
                    }
                }
            }
        }

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
        long long nErrTotal = 0, nDefBad = 0, nSpareBad = 0;

        for (int lf = y0; lf < y1; ++lf) {
            buildLine(lf);
            quint16* out = defw + (size_t)lf * width;

            // Error localisation: broadband twin disagreement.
            bool anyErr = false;
            for (int h = activeLeft; h < activeRight; ++h) {
                errMask[h] = std::fabs(chat[h] - bp[h]) > outlierThreshCode;
                if (errMask[h]) anyErr = true;
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
                // Phase-balanced carrier-free mean of R at h (0.5,1,1,1,0.5)/4.
                auto Rmean = [&](int h) {
                    return (0.5 * R(h - 2) + R(h - 1) + R(h) + R(h + 1) +
                            0.5 * R(h + 2)) * 0.25;
                };

                std::copy(chat.begin(), chat.end(), chatFix.begin());
                std::copy(lhat.begin(), lhat.end(), lhatFix.begin());

                // Carrier refill at error samples from the nearest valid
                // samples one or more full cycles away (same phase class).
                auto chatRefill = [&](int h) -> double {
                    for (int k = 4; k <= 16; k += 4) {
                        const int a = h - k, b = h + k;
                        const bool va = a >= activeLeft && !errMask[a];
                        const bool vb = b < activeRight && !errMask[b];
                        if (va && vb) return 0.5 * (chat[a] + chat[b]);
                        if (va) return chat[a];
                        if (vb) return chat[b];
                    }
                    return 0.0;
                };

                // Arbitrate per error RUN (majority over the run): the twin
                // more dissimilar to the phase-matched comp reference is
                // the error.
                int h = activeLeft;
                while (h < activeRight) {
                    if (!errMask[h]) { ++h; continue; }
                    int rEnd = h;
                    while (rEnd + 1 < activeRight && errMask[rEnd + 1]) ++rEnd;
                    double eDef = 0.0, eSpare = 0.0;
                    const quint16* dl = defp + (size_t)lf * width;
                    const quint16* sl = sparep + (size_t)lf * width;
                    for (int x = h; x <= rEnd; ++x) {
                        const double r = R(x);
                        const double refSpare = r;                 // cophased
                        const double refDef = 2.0 * Rmean(x) - r;  // flipped
                        eDef += std::fabs((double)dl[x] - refDef);
                        eSpare += std::fabs((double)sl[x] - refSpare);
                    }
                    const bool badIsDef = eDef > eSpare;
                    for (int x = h; x <= rEnd; ++x) {
                        const double cf = chatRefill(x);
                        chatFix[x] = cf;
                        lhatFix[x] = badIsDef
                            ? (double)sl[x] + cf   // spare = L - c
                            : (double)dl[x] - cf;  // def   = L + c
                    }
                    nErrTotal += rEnd - h + 1;
                    if (badIsDef) nDefBad += rEnd - h + 1;
                    else          nSpareBad += rEnd - h + 1;
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

        if (dsDebug && nErrTotal > 0)
            std::fprintf(stderr,
                "DSREF-FIX seq=%d nerr=%lld defBad=%lld spareBad=%lld\n",
                def.field.seqNo, nErrTotal, nDefBad, nSpareBad);

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
    static const bool dsDebug = std::getenv("LDCD_PROBE_DSREF") != nullptr;
    const bool ok = mergeDgPairWithSanity(videoParameters, config, def, spare, comp);
    if (dsDebug)
        std::fprintf(stderr, "DSREF-WRAP def=%d spare=%d ok=%d plane=%d\n",
                     def.field.seqNo, spare.field.seqNo, (int)ok,
                     (int)def.dgExactCarrier.size());
    return ok;
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

    auto emitTelecine2 = [&](char label, SourceField&& f1, int cid1,
                             SourceField&& f2, int cid2,
                             WorkItem::Expansion ex = WorkItem::Expansion::None) {
        stampForcedCadence(f1, cid1);
        stampForcedCadence(f2, cid2);
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
            stampForcedCadence(def, 0);
            stampForcedCadence(comp, 1);

            WorkItem::Expansion ex = WorkItem::Expansion::None;

            // If the next slot is 2, consume it as A-spare (if present).
            if (!window.empty() && cycleIndex(forcedFieldIndex) == 2) {
                SourceField spare = pop1();
                stampForcedCadence(spare, 2);
                if (!config.dgDiscard && mergeDgPairWithSanityWrapper(def, spare, comp)) {
                    ex = WorkItem::Expansion::Trailing;
                } else {
                    releaseToBaseline(std::move(spare));
                }
            }

            emitTelecine2('A', std::move(def), 0, std::move(comp), 1, ex);
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
            emitTelecine2('B', std::move(f3), 3, std::move(f4), 4,
                          WorkItem::Expansion::None);
            continue;
        }

        // C start (preferred): 5(spare)+6(comp)+7(def)
        if (idx == 5) {
            if (window.size() >= 3) {
                SourceField spare = pop1(); // 5
                SourceField comp  = pop1(); // 6
                SourceField def   = pop1(); // 7
                stampForcedCadence(spare, 5);
                stampForcedCadence(comp, 6);
                stampForcedCadence(def, 7);
        
                WorkItem::Expansion ex = WorkItem::Expansion::None;
                if (!config.dgDiscard && mergeDgPairWithSanityWrapper(def, spare, comp)) {
                    ex = WorkItem::Expansion::Leading;
                } else {
                    releaseToBaseline(std::move(spare));
                }
                emitTelecine2('C', std::move(comp), 6, std::move(def), 7, ex);
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
            emitTelecine2('C', std::move(comp), 6, std::move(def), 7,
                          WorkItem::Expansion::None);
            continue;
        }

        // D start: 8+9
        if (idx == 8) {
            if (window.size() < 2) break;
            auto [f8, f9] = pop2();
            emitTelecine2('D', std::move(f8), 8, std::move(f9), 9,
                          WorkItem::Expansion::None);
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
                stampForcedCadence(def, 0);
                stampForcedCadence(comp, 1);

                WorkItem::Expansion ex = WorkItem::Expansion::None;
                if (!window.empty() && cycleIndex(forcedFieldIndex) == 2) {
                    SourceField spare = pop1();
                    stampForcedCadence(spare, 2);
                    if (!config.dgDiscard && mergeDgPairWithSanityWrapper(def, spare, comp)) {
                        ex = WorkItem::Expansion::Trailing;
                    } else {
                        releaseToBaseline(std::move(spare));
                    }
                }
                emitTelecine2('A', std::move(def), 0, std::move(comp), 1, ex);
                continue;
            }

            if (idx == 3) {
                if (window.size() < 2) break;
                auto [f3, f4] = pop2();
                emitTelecine2('B', std::move(f3), 3, std::move(f4), 4,
                              WorkItem::Expansion::None);
                continue;
            }

            if (idx == 5) {
                if (window.size() >= 3) {
                    SourceField spare = pop1();
                    SourceField comp  = pop1();
                    SourceField def   = pop1();
                    stampForcedCadence(spare, 5);
                    stampForcedCadence(comp, 6);
                    stampForcedCadence(def, 7);
            
                    WorkItem::Expansion ex = WorkItem::Expansion::None;
                    if (!config.dgDiscard && mergeDgPairWithSanityWrapper(def, spare, comp)) {
                        ex = WorkItem::Expansion::Leading;
                    } else {
                        releaseToBaseline(std::move(spare));
                    }
                    emitTelecine2('C', std::move(comp), 6, std::move(def), 7, ex);
                    continue;
                }
                // Incomplete triple at flush — release whatever's left
                while (!window.empty()) releaseToBaseline(pop1());
                break;
            }
            if (idx == 6) {
                if (window.size() < 2) break;
                auto [comp, def] = pop2();
                emitTelecine2('C', std::move(comp), 6, std::move(def), 7,
                              WorkItem::Expansion::None);
                continue;
            }

            if (idx == 8) {
                if (window.size() < 2) break;
                auto [f8, f9] = pop2();
                emitTelecine2('D', std::move(f8), 8, std::move(f9), 9,
                              WorkItem::Expansion::None);
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
                    if (!config.dgDiscard && mergeDgPairWithSanityWrapper(def, spare, comp)) {
                        spareUsed = true;
                    } else {
                        releaseToBaseline(std::move(spare));
                    }
                } else {
                    SourceField spare = history[i0].field; // work on a copy; baseline retains original ownership
                    if (!config.dgDiscard && mergeDgPairWithSanityWrapper(def, spare, comp)) {
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
            return;
        }

        SourceField spare = history[pos].field;
        if (!config.dgDiscard) {
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

    workQueue.push_back(std::move(wi));
    return true;
}
