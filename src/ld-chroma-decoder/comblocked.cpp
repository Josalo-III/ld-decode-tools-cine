/******************************************************************************
 * comblocked.cpp
 * ld-chroma-decoder — Colourisation filter for ld-decode
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * SPDX-FileCopyrightText: 2018 Chad Page
 * SPDX-FileCopyrightText: 2018-2019 Simon Inns
 * SPDX-FileCopyrightText: 2020-2021 Adam Sampson
 * SPDX-FileCopyrightText: 2021 Phillip Blucas
 * SPDX-FileCopyrightText: 2025-2026 Joseph Burns
 *
 * This file is part of ld-decode-tools.
 * A subset of comb functions used when --ntsc-phase-comp is active
 ******************************************************************************/

#include "comb.h"
#include "combmath.h"
#include "feasibleband.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <QtGlobal>

namespace {

inline double smoothGate01(double t)
{
    t = std::clamp(t, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Mean over an even effective sample weight, but with an integer centroid.
// Support is effectiveWidth+1 samples: half weight at center +/- half and
// full weight between.  The endpoints are the same carrier phase, so their
// two halves preserve the complete-cycle population of the old even window.
// Edge replication keeps the coordinate fixed instead of sliding the aperture
// away from the requested sample.
inline double centeredEvenWeightMean(const double *values,
                                     const double *prefix,
                                     int width,
                                     int center,
                                     int effectiveWidth)
{
    if (!values || !prefix || width <= 0 || effectiveWidth <= 0 ||
        (effectiveWidth & 1))
        return 0.0;

    const int half = effectiveWidth / 2;
    const int lo = center - half;
    const int hi = center + half;
    double sum = 0.0;

    if (lo >= 0 && hi < width) {
        sum = 0.5 * (values[lo] + values[hi]) +
              (prefix[hi] - prefix[lo + 1]);
    } else {
        for (int k = -half; k <= half; ++k) {
            const int x = std::clamp(center + k, 0, width - 1);
            const double w = (k == -half || k == half) ? 0.5 : 1.0;
            sum += w * values[x];
        }
    }

    return sum / static_cast<double>(effectiveWidth);
}

// produceY coarse-platform selector (witness isolation knob).
//
// Default decomposes raw on the cheap, raster-aligned four-sample coarse. The
// witness unlocks a centered, lurch-sharpened sliding-boxcar coarse as its LF
// platform. There is exactly one platform in either mode: comb supplies the
// middle band and provisional top, while the HF election may replace only that
// top band.
//
// LD_COARSE_SHARP = L (float, witness-only isolation knob):
//   unset       : witness-native lurch level (1.0).
//   <=0         : disable the witness lurch for an A/B.
//   > 0         : scale the lurch snap gate (<1 gentler, >1 snaps weaker
//                 steps too).
// Structural carrier-amplitude ceiling, measured (2026-07-28) instead of
// guessed. The measured range ledger:
//   - GGV-1069 colour bars: 36 IRE max (SMPTE f6000 and full-height
//     f22000 agree) -- the BROADCAST-LEGAL maximum.
//   - Most saturated camera-scene material (Gilgol bikini zone): 32 IRE
//     (raw envelope 31.7, conservation-exact channel 32.2).
//   - Optical-composite EFFECTS run far hotter: the tractor beam measures
//     72 IRE on the conservation-exact (D-S)/2 channel -- real carrier,
//     no estimation. A bars-derived 38 IRE ceiling visibly aliased the
//     beam (and the pre-existing 50 IRE bound was already clipping its
//     top 0.8% of samples).
// A hull may encode impossibles ONLY, so the GLOBAL bound must clear the
// hottest real carrier this material contains: 75 IRE nominal, burst-
// scaled. One global limit cannot discriminate a range this wide (legal
// carrier spans ~1 IRE on gray struts to 72 IRE on the beam); regional
// discrimination belongs to a pooled exact-channel ceiling, not to this
// constant. carrierScale is the burst correlation magnitude (~10 IRE at
// spec burst).
inline double maxCarrierAmpIREFromScale(double carrierScale)
{
    constexpr double kCarrierMaxPerBurstScale = 7.5; // 75 IRE at spec burst
    constexpr double kCarrierMaxFloorIRE      = 75.0;
    return std::max(kCarrierMaxFloorIRE,
                    carrierScale * kCarrierMaxPerBurstScale);
}

inline double coarseSharpLevel()
{
    static const double level = []{
        const char *s = std::getenv("LD_COARSE_SHARP");
        if (!s || !*s) return 1.0;
        const double v = std::atof(s);
        return std::isfinite(v) ? v : 0.0;
    }();
    return level;
}

// Transfer-knee stats (LDCD_PROBE_KNEE=1). Measurement only. Comb inherits
// 1D's bandwidth limits: fine-detail AMPLITUDE is stripped from 1D and its
// descendants -- a transfer-curve divergence at the top of the scale, HF
// only. This measures that curve directly: at pixels where the carrier-band
// energy is PROVEN not-carrier (carrierIllegalProof high), the retracted
// top is trustworthy luma amplitude; bin |comb top| against |retracted top|
// and the bin where the ratio departs from unity IS the knee -- the
// measured point at which comb's HF stops being trusted, which the coming
// roll-off keys to. Sign-flip fraction per bin rides along for the
// grammar-side sign fix (the taps own the sign).
// (D-S)/2 referee (LDCD_PROBE_DSREF=1): grades every carrier estimator
// against the assembler's exact-carrier side channel on covered lines.
// The confiscation ledger: error vs exact truth, split flat / detail
// (hLumaDelta >= 6 IRE), per frame. Thread-safe use: run -t 1.
struct DsRefProbe {
    static bool on()
    {
        static const bool v = std::getenv("LDCD_PROBE_DSREF") != nullptr;
        return v;
    }
    struct Est {
        long n = 0;
        double sumAbs = 0, sum = 0, sumSq = 0, maxAbs = 0;
        void add(double eIRE)
        {
            if (!std::isfinite(eIRE)) return;
            n++; sumAbs += std::fabs(eIRE); sum += eIRE; sumSq += eIRE * eIRE;
            maxAbs = std::max(maxAbs, std::fabs(eIRE));
        }
    };
    // [flat=0 / detail=1][estimator: 0=1D 1=comb 2=fit 3=retracted]
    // NOTE: in the default (anchor) retracted mode the retr column is
    // SELF-REFERENTIAL on covered samples (retracted == exact there);
    // grade that mode by saturation-restricted fSC-in-Y instead.
    Est e[2][4];
    long covered = 0;
    long frameIdx = 0;

    void flush()
    {
        static const char *nm[4] = { "1D", "comb", "fit", "retr" };
        if (covered > 0) {
            std::fprintf(stderr, "[DSREF f=%ld] covered px=%ld\n", frameIdx, covered);
            for (int b = 0; b < 2; ++b) {
                std::fprintf(stderr, "  %s:", b ? "detail" : "flat  ");
                for (int k = 0; k < 4; ++k) {
                    const Est &E = e[b][k];
                    if (E.n == 0) { std::fprintf(stderr, "  %s n=0", nm[k]); continue; }
                    std::fprintf(stderr,
                        "  %s |e| %.2f bias %+.2f rms %.2f max %.1f (n=%ld)",
                        nm[k], E.sumAbs / E.n, E.sum / E.n,
                        std::sqrt(E.sumSq / E.n), E.maxAbs, E.n);
                }
                std::fprintf(stderr, "\n");
            }
        }
        frameIdx++;
        covered = 0;
        for (int b = 0; b < 2; ++b) for (int k = 0; k < 4; ++k) e[b][k] = Est();
    }
};

DsRefProbe g_dsRefProbe;

// Cross-colour return referee (LDCD_PROBE_CCREF=1): grades the CC return
// against the exact-carrier side channel on covered samples.
//
// On a covered sample the ideal return fraction is not an opinion, it is
// arithmetic. With ex the certified carrier and C = raw - combY the comb's
// carrier estimate,
//
//     leak = C - ex          the luma the comb confiscated (signed IRE)
//     m*   = leak / C        the return fraction that lands on truth
//
// leak IS cross-colour, measured rather than inferred. The return publishes
// combY + m*C, and the truth is combY + leak, so its Y error is
// |m*C - leak| against the comb's own |leak|. Every covered sample therefore
// carries a VERDICT -- helped, harmed, or inert -- not a correlation.
//
// Ratios of per-sample quantities are never averaged: a bin's ideal mask is
// the properly weighted S(C*leak)/S(C*C), and its leak fraction is the
// pooled S|leak|/S|C|. Both are well conditioned where a mean of leak/C
// would be all tail.
//
// Run with -t 1 (unsynchronised accumulators). Run with LDCD_ANCHOR_1D=0 for
// the calibration read: in the default anchored mode the comb already
// consumes the retraction ladder on covered frames, so leak ~ 0 there by
// construction and the ledger measures nothing that transfers to uncovered
// lines. Default-on is still worth a run of its own -- there it answers
// whether the return correctly ABSTAINS on covered lines, or is damaging
// them today. The detector is honest in both modes: its edge read is fed by
// the true locked1DSource, which the anchoring never touches.
struct CcRefProbe {
    static bool on()
    {
        static const bool v = std::getenv("LDCD_PROBE_CCREF") != nullptr;
        return v;
    }
    // Detector components, each graded separately against the measured leak.
    // Which of these predicts the leak, and which is the noise, is the whole
    // question -- a mask driven by an uninformative read is the noisiest
    // election participant by construction.
    // RDev (2026-08-02) is the LIVE committed-verdict read: the anchored
    // delta devIRE mapped to [0,1] as devIRE / (2*cutoff), so bin center
    // 0.5 sits at the shipping cutoff and the per-bin ideal-m curve IS the
    // cutoff sweep. The gA/edge/lurch columns grade retired reads, kept for
    // contrast.
    enum Read { RgA = 0, REdge, RRegion, RGrammar, RLurch, RMask, RDev,
                NREADS };
    static constexpr int kBins = 5;
    static constexpr double kFireEps = 0.01;
    // Fixed-scalar sweep: what a constant return fraction would have scored.
    // If the adaptive mask cannot beat the best constant, the adaptation is
    // not earning its variance.
    static constexpr int kSweep = 5;
    static constexpr double kSweepM[kSweep] = { 0.0, 0.25, 0.50, 0.75, 1.0 };

    struct Ledger {
        long n = 0, helped = 0, harmed = 0;
        double sumLeak = 0, sumCarr = 0, sumMask = 0;
        double sumErrRet = 0, sumGain = 0;
        double sumCL = 0, sumCC = 0;
        double sweep[kSweep] = {};

        void add(double carr, double leak, double mask)
        {
            const double errComb = std::fabs(leak);
            const double errRet  = std::fabs(mask * carr - leak);
            n++;
            sumLeak   += errComb;
            sumCarr   += std::fabs(carr);
            sumMask   += mask;
            sumErrRet += errRet;
            sumGain   += errComb - errRet;
            sumCL     += carr * leak;
            sumCC     += carr * carr;
            for (int k = 0; k < kSweep; ++k)
                sweep[k] += std::fabs(kSweepM[k] * carr - leak);
            if (errRet < errComb - 1e-9)      helped++;
            else if (errRet > errComb + 1e-9) harmed++;
        }
        void merge(const Ledger &o)
        {
            n += o.n; helped += o.helped; harmed += o.harmed;
            sumLeak += o.sumLeak; sumCarr += o.sumCarr; sumMask += o.sumMask;
            sumErrRet += o.sumErrRet; sumGain += o.sumGain;
            sumCL += o.sumCL; sumCC += o.sumCC;
            for (int k = 0; k < kSweep; ++k) sweep[k] += o.sweep[k];
        }
        // The ideal fixed return fraction for this population.
        double mIdeal() const { return sumCC > 0.0 ? sumCL / sumCC : 0.0; }
    };
    // [flat=0 / detail=1][silent=0 / fired=1]
    Ledger led[2][2];

    struct Bin {
        long n = 0;
        double sumCL = 0, sumCC = 0, sumLeak = 0, sumCarr = 0, sumMask = 0;
        void add(double carr, double leak, double mask)
        {
            n++;
            sumCL += carr * leak; sumCC += carr * carr;
            sumLeak += std::fabs(leak); sumCarr += std::fabs(carr);
            sumMask += mask;
        }
        double mIdeal() const { return sumCC > 0.0 ? sumCL / sumCC : 0.0; }
        double leakFrac() const { return sumCarr > 0.0 ? sumLeak / sumCarr : 0.0; }
    };
    Bin bins[NREADS][kBins];

    // AMPLITUDE CONDITIONING -- without this the whole instrument lies.
    //
    // C and leak share the capture's noise. If the comb's luma were PERFECT,
    // C = c_true + n_d and leak = (n_d + n_s)/2, so
    //     S(C*leak)/S(C*C) -> sigma^2/2 / sigma^2 = 0.5
    // in the noise floor with ZERO cross-colour present. An ideal mask near
    // 0.5 at low carrier is therefore the signature of noise, not of leak,
    // and "returning" it would only push the capture's own noise into Y --
    // Lhat has half the noise of raw, so following raw there is a downgrade
    // the L1 ledger cannot see.
    //
    // Only bins whose ideal mask stands well clear of 0.5 at real carrier
    // amplitude are evidence of confiscated luma. Hence: an amplitude census,
    // and a second copy of every read table restricted to real carrier.
    static constexpr int kAmp = 5;
    static constexpr double kAmpEdgeIRE[kAmp] = { 1.0, 2.0, 4.0, 8.0, 1e9 };
    static constexpr double kHiCarrierIRE = 2.0;
    Bin ampBins[kAmp];
    Bin binsHi[NREADS][kBins];
    Ledger ledHi[2];  // [silent / fired], real carrier only
    long covered = 0, frameIdx = 0;

    // STAR LAW test (user, 2026-07-30): a black-white-black transient inside
    // 4 px cannot be legal carrier -- the encoder bandlimits chroma to
    // 1.3 MHz, so a legal envelope cannot rise and fall in under ~5.5
    // samples at 4fSC. At such a site the lawful subtrahend is ZERO and
    // composite IS the luma. Prediction, testable on covered lines: at
    // clean star sites ex ~ 0 and m* ~ 1; at the same shape WITH ringing
    // (the bail branch) truth shows carrier and m* < 1. This grades the
    // composite-domain test itself, so once certified it can run on EVERY
    // frame, covered or not.
    struct StarLed {
        long n = 0;
        double sumCL = 0, sumCC = 0, sumLeak = 0, sumCarr = 0, sumMask = 0;
        double sumExAbs = 0, sumPeak = 0, sumErrRet = 0;
        double sumE1D = 0; long n1D = 0;
        // Skirt (±3..±5) true carrier: grades the refined bail — "peak takes
        // raw, skirt goes to 1D-as-chroma". If skirt |ex| is at the noise
        // floor, 1D's chroma reading of the skirt is wrong but bail-safe; if
        // skirt |ex| is real carrier, the bail branch is doing its job.
        double sumSkirtEx = 0;
        double mIdeal() const { return sumCC > 0 ? sumCL / sumCC : 0.0; }
    };
    StarLed starLed[2]; // 0 = clean (actionable), 1 = ringing (bail branch)

    // Vertical-context census: how many of the four frame-line neighbours
    // (l±1, l±2, same column ±1) also carry the signature. A point star is
    // vertically compact (vertRun 0-1); a strut passes the horizontal test
    // on every line (vertRun 3-4). If |ex| rises with vertRun, struts carry
    // real carrier that the substitution would wrongly republish into Y.
    struct VertLed {
        long n = 0; double sumEx = 0, sumCL = 0, sumCC = 0, sumMask = 0;
        double mIdeal() const { return sumCC > 0 ? sumCL / sumCC : 0.0; }
    };
    VertLed vertLed[5];
    void vertSample(int run, double carr, double leak, double mask,
                    double exIRE)
    {
        VertLed &L = vertLed[std::clamp(run, 0, 4)];
        L.n++; L.sumEx += std::fabs(exIRE);
        L.sumCL += carr * leak; L.sumCC += carr * carr; L.sumMask += mask;
    }

    void starSample(int cls, double carr, double leak, double mask,
                    double exIRE, double peakIRE, double e1D, double skirtEx)
    {
        StarLed &L = starLed[cls];
        L.n++;
        L.sumCL += carr * leak; L.sumCC += carr * carr;
        L.sumLeak += std::fabs(leak); L.sumCarr += std::fabs(carr);
        L.sumMask += mask;
        L.sumExAbs += std::fabs(exIRE); L.sumPeak += peakIRE;
        L.sumErrRet += std::fabs(mask * carr - leak);
        L.sumSkirtEx += skirtEx;
        if (std::isfinite(e1D)) { L.sumE1D += std::fabs(e1D); L.n1D++; }
    }

    // ---- MODEL-vs-TRUTH EXPLANATION LEDGER ----
    // Not "how wrong" (the DSREF ledger already says) but WHERE and WHY:
    // which site classes carry each estimator's error mass, and how much of
    // the model's error at real carrier is amplitude (a per-window gain
    // explains it) versus shape/phase (no scalar can). Site classes are
    // assigned by priority star > legal > detail > flat, so each covered
    // sample lands in exactly one.
    enum Cls { CFlat = 0, CDetail, CStarClean, CStarRing, CLegal, NCLS };
    struct EstErr {
        long n = 0; double sumAbs = 0, sum = 0, sumSq = 0;
        void add(double e)
        {
            if (!std::isfinite(e)) return;
            n++; sumAbs += std::fabs(e); sum += e; sumSq += e * e;
        }
    };
    EstErr clsE[NCLS][3];              // estimator: 0=fit 1=comb 2=1D
    double clsExAbs[NCLS] = {};        // true carrier mass per class
    long   clsN[NCLS] = {};

    void classSample(int cls, double fitIRE, double cIRE, double oneDIRE,
                     double exIRE, int line, int xi)
    {
        clsN[cls]++;
        clsExAbs[cls] += std::fabs(exIRE);
        clsE[cls][0].add(fitIRE - exIRE);
        clsE[cls][1].add(cIRE - exIRE);
        clsE[cls][2].add(oneDIRE - exIRE);
        if (std::isfinite(fitIRE) && line >= 0 && line < pH &&
            xi >= 0 && xi < pW)
            pFit[static_cast<size_t>(line) * pW + xi] =
                static_cast<float>(fitIRE);
    }

    // Gain-explainability: non-overlapping 16-sample windows on covered
    // lines, evaluated only where the window holds real carrier
    // (mean ex^2 >= 4 IRE^2). Per window the best scalar gain g minimizes
    // sum(est - g*ex)^2; the residual after g is the SHAPE/PHASE part of the
    // error — the part no calibration could ever fix (the 857f457 law,
    // now with a number attached).
    struct GainAcc {
        long nWin = 0; double sumG = 0;
        long gBin[5] = {};             // g <0.5, 0.5-0.8, 0.8-1.25, 1.25-2, >2
        double sumErrSq = 0, sumResSq = 0;
        void addWin(double g, double errSq, double resSq)
        {
            nWin++; sumG += g;
            const int b = g < 0.5 ? 0 : g < 0.8 ? 1 : g < 1.25 ? 2
                          : g < 2.0 ? 3 : 4;
            gBin[b]++;
            sumErrSq += errSq; sumResSq += resSq;
        }
    };
    GainAcc gainFit, gainComb;
    int wLine = -1, wCount = 0;
    double wEE = 0, wFE = 0, wFF = 0, wFD = 0, wCE = 0, wCC2 = 0, wCD = 0;
    bool wFitOK = true;

    void gainWindow(int line, double fitIRE, double cIRE, double exIRE)
    {
        if (line != wLine) { wLine = line; wCount = 0; wEE = wFE = wFF =
            wFD = wCE = wCC2 = wCD = 0; wFitOK = true; }
        wEE += exIRE * exIRE;
        if (std::isfinite(fitIRE)) {
            wFE += fitIRE * exIRE; wFF += fitIRE * fitIRE;
            wFD += (fitIRE - exIRE) * (fitIRE - exIRE);
        } else wFitOK = false;
        wCE += cIRE * exIRE; wCC2 += cIRE * cIRE;
        wCD += (cIRE - exIRE) * (cIRE - exIRE);
        if (++wCount == 16) {
            if (wEE >= 16.0 * 4.0) {   // mean |ex| >= 2 IRE: real carrier
                if (wFitOK && wFD > 1e-9)
                    gainFit.addWin(wFE / wEE, wFD, wFF - wFE * wFE / wEE);
                if (wCD > 1e-9)
                    gainComb.addWin(wCE / wEE, wCD, wCC2 - wCE * wCE / wEE);
            }
            wCount = 0; wEE = wFE = wFF = wFD = wCE = wCC2 = wCD = 0;
            wFitOK = true;
        }
    }

    // RESIDUAL STRUCTURE. Magnitude is not the artifact; structure is. A
    // return that is wrong on 16% of samples is noise if those samples are
    // scattered and an artifact class if they are organised, and the L1
    // ledger above cannot tell the two apart.
    //
    // What truth can and cannot see: certified coverage is ONE PARITY of
    // every other frame, so within a covered frame the covered lines are the
    // frame lines of a single parity, pitch 2. Truth can therefore measure
    // vertical structure at FIELD pitch (l-2, l, l+2, all covered) but can
    // never see frame-pitch (l-1, l, l+1) structure in the residual -- the
    // other parity has no truth to be wrong against. The classic lineAlt
    // artifact lives at frame pitch, so it is measured here on the MASK
    // instead, which exists on every line: if the mask's frame-pitch D2
    // exceeds its field-pitch D2, the return's ACTION alternates by parity
    // even though its error cannot be graded there. Nearer neighbours should
    // disagree less; when they disagree more, the disagreement is parity.
    std::vector<float> pErrRet, pErrComb, pCarr, pMask, pFit;
    std::vector<std::uint8_t> pCov;
    int pW = 0, pH = 0;

    void ensurePlanes(int w, int h)
    {
        if (pW == w && pH == h) return;
        pW = w; pH = h;
        const size_t n = static_cast<size_t>(w) * h;
        pErrRet.assign(n, 0.0f); pErrComb.assign(n, 0.0f);
        pCarr.assign(n, 0.0f);   pMask.assign(n, 0.0f);
        pFit.assign(n, 0.0f);
        pCov.assign(n, 0);
    }
    void clearPlanes()
    {
        std::fill(pErrRet.begin(), pErrRet.end(), 0.0f);
        std::fill(pErrComb.begin(), pErrComb.end(), 0.0f);
        std::fill(pCarr.begin(), pCarr.end(), 0.0f);
        std::fill(pMask.begin(), pMask.end(), 0.0f);
        std::fill(pFit.begin(), pFit.end(), 0.0f);
        std::fill(pCov.begin(), pCov.end(), 0);
    }

    // Every sample, covered or not: the mask's own geometry needs both
    // parities.
    void note(int line, int xi, int w, int h, double carr, double mask)
    {
        ensurePlanes(w, h);
        if (line < 0 || line >= pH || xi < 0 || xi >= pW) return;
        const size_t i = static_cast<size_t>(line) * pW + xi;
        pCarr[i] = static_cast<float>(carr);
        pMask[i] = static_cast<float>(mask);
    }

    void sample(int detailBin, double carr, double leak, double mask,
                const double *reads, int line, int xi)
    {
        if (line >= 0 && line < pH && xi >= 0 && xi < pW) {
            const size_t i = static_cast<size_t>(line) * pW + xi;
            pErrRet[i]  = static_cast<float>(mask * carr - leak);
            pErrComb[i] = static_cast<float>(-leak);
            pCov[i]     = 1;
        }
        covered++;
        const int fired = mask > kFireEps ? 1 : 0;
        led[detailBin][fired].add(carr, leak, mask);

        const double aC = std::fabs(carr);
        int a = 0;
        while (a < kAmp - 1 && aC >= kAmpEdgeIRE[a]) ++a;
        ampBins[a].add(carr, leak, mask);

        const bool hi = aC >= kHiCarrierIRE;
        if (hi) ledHi[fired].add(carr, leak, mask);

        for (int r = 0; r < NREADS; ++r) {
            const int b = std::clamp(
                static_cast<int>(reads[r] * kBins), 0, kBins - 1);
            bins[r][b].add(carr, leak, mask);
            if (hi) binsHi[r][b].add(carr, leak, mask);
        }
    }

    // Residual structure at the pitches truth can reach, plus the mask's own
    // parity geometry. All restricted to real carrier: at the noise floor
    // every D2 is noise and the comparison says nothing.
    void reportStructure()
    {
        if (pW <= 0 || pH <= 0) return;
        auto at = [&](int l, int x) { return static_cast<size_t>(l) * pW + x; };
        auto real = [&](size_t i) {
            return pCov[i] && std::fabs((double)pCarr[i]) >= kHiCarrierIRE;
        };

        // 1. Vertical, FIELD pitch (l-2, l, l+2) -- adjacent covered lines.
        long nV = 0; double vRet = 0, vComb = 0;
        for (int l = 2; l + 2 < pH; ++l) {
            for (int x = 0; x < pW; ++x) {
                const size_t c = at(l, x), u = at(l - 2, x), d = at(l + 2, x);
                if (!real(c) || !pCov[u] || !pCov[d]) continue;
                vRet  += std::fabs((double)pErrRet[u]  - 2.0 * pErrRet[c]  + pErrRet[d]);
                vComb += std::fabs((double)pErrComb[u] - 2.0 * pErrComb[c] + pErrComb[d]);
                nV++;
            }
        }

        // 2. Horizontal, sample pitch -- the 2fSC speckle axis.
        long nH = 0; double hRet = 0, hComb = 0;
        for (int l = 0; l < pH; ++l) {
            for (int x = 1; x + 1 < pW; ++x) {
                const size_t c = at(l, x), a = at(l, x - 1), b = at(l, x + 1);
                if (!real(c) || !pCov[a] || !pCov[b]) continue;
                hRet  += std::fabs((double)pErrRet[a]  - 2.0 * pErrRet[c]  + pErrRet[b]);
                hComb += std::fabs((double)pErrComb[a] - 2.0 * pErrComb[c] + pErrComb[b]);
                nH++;
            }
        }

        // 3. Are the harmed samples scattered or organised? Mean run length
        // of consecutive harmed samples along a line, against the length that
        // independent coin flips at the same rate would give (1/(1-p)).
        long nElig = 0, nHarm = 0, nRuns = 0, runLenSum = 0, run = 0;
        for (int l = 0; l < pH; ++l) {
            run = 0;
            for (int x = 0; x < pW; ++x) {
                const size_t c = at(l, x);
                const bool elig = real(c) && pMask[c] > kFireEps;
                if (!elig) {
                    if (run > 0) { nRuns++; runLenSum += run; run = 0; }
                    continue;
                }
                nElig++;
                if (std::fabs((double)pErrRet[c]) > std::fabs((double)pErrComb[c])) {
                    nHarm++; run++;
                } else if (run > 0) { nRuns++; runLenSum += run; run = 0; }
            }
            if (run > 0) { nRuns++; runLenSum += run; run = 0; }
        }

        // 4. The mask's parity geometry, on ALL lines. Frame pitch (+/-1)
        // against field pitch (+/-2): nearer neighbours should disagree LESS.
        // A CONTROL rides along: |carrier| through the identical stencils.
        // For any vertically smooth quantity D2 ~ k^2, so the +/-1 stencil
        // should read about a QUARTER of the +/-2 one -- ratio ~0.25. The
        // control establishes what this material's own vertical detail does
        // to that expectation, so the mask's ratio is graded against a
        // measurement rather than against theory.
        long nM = 0; double mFrame = 0, mField = 0, cFrame = 0, cField = 0;
        for (int l = 2; l + 2 < pH; ++l) {
            for (int x = 0; x < pW; ++x) {
                const size_t c = at(l, x);
                if (std::fabs((double)pCarr[c]) < kHiCarrierIRE) continue;
                mFrame += std::fabs((double)pMask[at(l - 1, x)] -
                                    2.0 * pMask[c] + pMask[at(l + 1, x)]);
                mField += std::fabs((double)pMask[at(l - 2, x)] -
                                    2.0 * pMask[c] + pMask[at(l + 2, x)]);
                cFrame += std::fabs(std::fabs((double)pCarr[at(l - 1, x)]) -
                                    2.0 * std::fabs((double)pCarr[c]) +
                                    std::fabs((double)pCarr[at(l + 1, x)]));
                cField += std::fabs(std::fabs((double)pCarr[at(l - 2, x)]) -
                                    2.0 * std::fabs((double)pCarr[c]) +
                                    std::fabs((double)pCarr[at(l + 2, x)]));
                nM++;
            }
        }

        if (nV > 0)
            std::fprintf(stderr,
                "  struct vert(field pitch) n=%-7ld D2comb=%.3f D2ret=%.3f "
                "ratio=%.3f\n", nV, vComb / nV, vRet / nV,
                vComb > 0 ? vRet / vComb : 0.0);
        if (nH > 0)
            std::fprintf(stderr,
                "  struct horiz(sample)     n=%-7ld D2comb=%.3f D2ret=%.3f "
                "ratio=%.3f\n", nH, hComb / nH, hRet / nH,
                hComb > 0 ? hRet / hComb : 0.0);
        if (nElig > 0 && nRuns > 0) {
            const double p = (double)nHarm / (double)nElig;
            std::fprintf(stderr,
                "  struct harmed            n=%-7ld rate=%.3f runLen=%.2f "
                "indep=%.2f clustering=%.2fx\n",
                nElig, p, (double)runLenSum / nRuns, 1.0 / (1.0 - p),
                ((double)runLenSum / nRuns) * (1.0 - p));
        }
        if (nM > 0)
            std::fprintf(stderr,
                "  struct mask parity       n=%-7ld D2frame=%.4f D2field=%.4f "
                "ratio=%.3f  |ctl |C|: %.3f %.3f ratio=%.3f|\n",
                nM, mFrame / nM, mField / nM,
                mField > 0 ? mFrame / mField : 0.0,
                cFrame / nM, cField / nM,
                cField > 0 ? cFrame / cField : 0.0);

        // 5. Vertical coherence at ±2 (adjacent covered lines), real carrier
        // only. Truth's own coherence is the ceiling (doc: 0.907); the fit's
        // coherence against it says whether the model's error is per-line
        // noise (low r) or a coherent, structured misread (r near truth's,
        // but offset).
        {
            struct Corr {
                double sx=0, sy=0, sxx=0, syy=0, sxy=0; long n=0;
                void add(double x, double y){ sx+=x; sy+=y; sxx+=x*x;
                    syy+=y*y; sxy+=x*y; n++; }
                double r() const {
                    if (n < 2) return 0.0;
                    const double cx = sxx - sx*sx/n, cy = syy - sy*sy/n,
                                 cxy = sxy - sx*sy/n;
                    return (cx > 1e-9 && cy > 1e-9)
                        ? cxy / std::sqrt(cx*cy) : 0.0;
                }
            } cEx, cFit, cC;
            for (int l = 0; l + 2 < pH; ++l) {
                for (int x = 0; x < pW; ++x) {
                    const size_t a = at(l, x), b = at(l + 2, x);
                    if (!pCov[a] || !pCov[b]) continue;
                    const double exA = (double)pCarr[a] + pErrComb[a];
                    const double exB = (double)pCarr[b] + pErrComb[b];
                    if (std::fabs(exA) < kHiCarrierIRE ||
                        std::fabs(exB) < kHiCarrierIRE) continue;
                    cEx.add(exA, exB);
                    cFit.add((double)pFit[a], (double)pFit[b]);
                    cC.add((double)pCarr[a], (double)pCarr[b]);
                }
            }
            if (cEx.n > 100)
                std::fprintf(stderr,
                    "  coher ±2 (|ex|>=2)       n=%-7ld ex=%.3f fit=%.3f "
                    "comb=%.3f\n", cEx.n, cEx.r(), cFit.r(), cC.r());
        }
    }

    void flush()
    {
        static const char *dn[2] = { "flat  ", "detail" };
        static const char *fn[2] = { "silent", "fired " };
        static const char *rn[NREADS] =
            { "gA     ", "edge   ", "regKeep", "gPass  ", "lurch  ",
              "mask   ", "dev    " };
        if (covered > 0) {
            std::fprintf(stderr, "[CCREF f=%ld] covered px=%ld\n",
                         frameIdx, covered);
            Ledger all;
            for (int d = 0; d < 2; ++d) {
                for (int f = 0; f < 2; ++f) {
                    const Ledger &L = led[d][f];
                    all.merge(L);
                    if (L.n == 0) continue;
                    std::fprintf(stderr,
                        "  %s %s n=%-8ld m=%.3f |C|=%5.2f |leak|=%5.2f "
                        "errComb=%5.2f errRet=%5.2f gain=%+6.3f "
                        "helped=%5.1f%% harmed=%5.1f%% mIdeal=%+.3f\n",
                        dn[d], fn[f], L.n, L.sumMask / L.n, L.sumCarr / L.n,
                        L.sumLeak / L.n, L.sumLeak / L.n, L.sumErrRet / L.n,
                        L.sumGain / L.n, 100.0 * L.helped / L.n,
                        100.0 * L.harmed / L.n, L.mIdeal());
                }
            }
            if (all.n > 0) {
                std::fprintf(stderr,
                    "  ceiling: mIdeal=%+.3f errAdaptive=%.3f errIdealPerSample=0"
                    "  fixed-m:", all.mIdeal(), all.sumErrRet / all.n);
                for (int k = 0; k < kSweep; ++k)
                    std::fprintf(stderr, " %.2f=%.3f",
                                 kSweepM[k], all.sweep[k] / all.n);
                std::fprintf(stderr, "\n");
            }
            // Amplitude census. mIdeal ~ 0.5 in the lowest bin is the noise
            // pedestal, not evidence of leak; read the high bins.
            std::fprintf(stderr, "  |C| IRE :");
            for (int a = 0; a < kAmp; ++a) {
                const Bin &B = ampBins[a];
                if (B.n == 0) { std::fprintf(stderr, "  [<%.0f] n=0", kAmpEdgeIRE[a]); continue; }
                std::fprintf(stderr,
                    "  [<%.0f] n=%-7ld mIdeal=%+.3f leakFrac=%.3f m=%.3f",
                    kAmpEdgeIRE[a], B.n, B.mIdeal(), B.leakFrac(),
                    B.sumMask / B.n);
            }
            std::fprintf(stderr, "\n");
            for (int f = 0; f < 2; ++f) {
                const Ledger &L = ledHi[f];
                if (L.n == 0) continue;
                std::fprintf(stderr,
                    "  |C|>=%.0f %s n=%-8ld m=%.3f |C|=%5.2f |leak|=%5.2f "
                    "errComb=%5.2f errRet=%5.2f gain=%+6.3f "
                    "helped=%5.1f%% harmed=%5.1f%% mIdeal=%+.3f  fixed-m:",
                    kHiCarrierIRE, fn[f], L.n, L.sumMask / L.n,
                    L.sumCarr / L.n, L.sumLeak / L.n, L.sumLeak / L.n,
                    L.sumErrRet / L.n, L.sumGain / L.n,
                    100.0 * L.helped / L.n, 100.0 * L.harmed / L.n, L.mIdeal());
                for (int k = 0; k < kSweep; ++k)
                    std::fprintf(stderr, " %.2f=%.3f",
                                 kSweepM[k], L.sweep[k] / L.n);
                std::fprintf(stderr, "\n");
            }
            for (int r = 0; r < NREADS; ++r) {
                std::fprintf(stderr, "  %s:", rn[r]);
                for (int b = 0; b < kBins; ++b) {
                    const Bin &B = bins[r][b];
                    if (B.n == 0) { std::fprintf(stderr, "  [%d] n=0", b); continue; }
                    std::fprintf(stderr,
                        "  [%d] n=%-7ld mIdeal=%+.3f leakFrac=%.3f m=%.3f",
                        b, B.n, B.mIdeal(), B.leakFrac(), B.sumMask / B.n);
                }
                std::fprintf(stderr, "\n");
            }
            // The same reads with the noise pedestal excluded. These are the
            // only rows that can convict a detector.
            for (int r = 0; r < NREADS; ++r) {
                std::fprintf(stderr, "  %s|hi:", rn[r]);
                for (int b = 0; b < kBins; ++b) {
                    const Bin &B = binsHi[r][b];
                    if (B.n == 0) { std::fprintf(stderr, "  [%d] n=0", b); continue; }
                    std::fprintf(stderr,
                        "  [%d] n=%-7ld mIdeal=%+.3f leakFrac=%.3f m=%.3f",
                        b, B.n, B.mIdeal(), B.leakFrac(), B.sumMask / B.n);
                }
                std::fprintf(stderr, "\n");
            }
            static const char *sn[2] = { "clean", "ring " };
            for (int c = 0; c < 2; ++c) {
                const StarLed &L = starLed[c];
                if (L.n == 0) continue;
                std::fprintf(stderr,
                    "  star %s n=%-6ld peak=%5.1f |ex|=%5.2f mIdeal=%+.3f "
                    "m=%.3f |C|=%5.2f errComb=%5.2f errRet=%5.2f e1D=%5.2f "
                    "skirtEx=%.2f\n",
                    sn[c], L.n, L.sumPeak / L.n, L.sumExAbs / L.n, L.mIdeal(),
                    L.sumMask / L.n, L.sumCarr / L.n, L.sumLeak / L.n,
                    L.sumErrRet / L.n,
                    L.n1D > 0 ? L.sumE1D / L.n1D : -1.0,
                    L.sumSkirtEx / L.n);
            }
            for (int v = 0; v < 5; ++v) {
                const VertLed &L = vertLed[v];
                if (L.n == 0) continue;
                std::fprintf(stderr,
                    "  vert[%d] n=%-6ld |ex|=%5.2f mIdeal=%+.3f m=%.3f\n",
                    v, L.n, L.sumEx / L.n, L.mIdeal(), L.sumMask / L.n);
            }
            static const char *cn[NCLS] =
                { "flat  ", "detail", "star-c", "star-r", "legal " };
            double totAbs[3] = {};
            for (int c = 0; c < NCLS; ++c)
                for (int e = 0; e < 3; ++e) totAbs[e] += clsE[c][e].sumAbs;
            for (int c = 0; c < NCLS; ++c) {
                if (clsN[c] == 0) continue;
                std::fprintf(stderr,
                    "  cls %s n=%-8ld |ex|=%5.2f", cn[c], clsN[c],
                    clsExAbs[c] / clsN[c]);
                static const char *en[3] = { "fit", "comb", "1D" };
                for (int e = 0; e < 3; ++e) {
                    const EstErr &E = clsE[c][e];
                    if (E.n == 0) continue;
                    std::fprintf(stderr,
                        "  %s |e|=%5.2f bias=%+5.2f mass=%4.1f%%",
                        en[e], E.sumAbs / E.n, E.sum / E.n,
                        totAbs[e] > 0 ? 100.0 * E.sumAbs / totAbs[e] : 0.0);
                }
                std::fprintf(stderr, "\n");
            }
            static const char *gn[2] = { "fit ", "comb" };
            const GainAcc *ga[2] = { &gainFit, &gainComb };
            for (int g = 0; g < 2; ++g) {
                const GainAcc &G = *ga[g];
                if (G.nWin == 0) continue;
                std::fprintf(stderr,
                    "  gain %s nWin=%-6ld meanG=%.2f bins[<.5,.5-.8,.8-1.25,"
                    "1.25-2,>2]=%ld/%ld/%ld/%ld/%ld shapeFrac=%.3f\n",
                    gn[g], G.nWin, G.sumG / G.nWin,
                    G.gBin[0], G.gBin[1], G.gBin[2], G.gBin[3], G.gBin[4],
                    G.sumErrSq > 0 ? G.sumResSq / G.sumErrSq : 0.0);
            }
            reportStructure();
        }
        frameIdx++;
        covered = 0;
        for (int c = 0; c < 2; ++c) starLed[c] = StarLed();
        for (int v = 0; v < 5; ++v) vertLed[v] = VertLed();
        for (int c = 0; c < NCLS; ++c) {
            clsExAbs[c] = 0; clsN[c] = 0;
            for (int e = 0; e < 3; ++e) clsE[c][e] = EstErr();
        }
        gainFit = GainAcc(); gainComb = GainAcc();
        wLine = -1; wCount = 0;
        clearPlanes();
        for (int d = 0; d < 2; ++d) for (int f = 0; f < 2; ++f) led[d][f] = Ledger();
        for (int f = 0; f < 2; ++f) ledHi[f] = Ledger();
        for (int a = 0; a < kAmp; ++a) ampBins[a] = Bin();
        for (int r = 0; r < NREADS; ++r)
            for (int b = 0; b < kBins; ++b) { bins[r][b] = Bin(); binsHi[r][b] = Bin(); }
    }
};

CcRefProbe g_ccRefProbe;

// SYNC-TONE stability probe (LDCD_PROBE_SYNC=1, run -t 1): the premise test
// for the user's segment-tracked certified phase reference. On each covered
// frame, pool the def lines' working-space certified IQ (the 4fsc demod
// caches ARE certified there under stage 1) per region; report the
// amplitude-weighted phase delta against the PREVIOUS covered frame — the
// step a running tracker would have to ride. If the deltas are small and
// slowly varying, the sync tone is real and a past-only tracker is a lawful
// construction-time reference for every frame; if they are content-noise,
// the premise fails here and we say so.
struct SyncProbe {
    static bool on()
    {
        static const bool v = std::getenv("LDCD_PROBE_SYNC") != nullptr;
        return v;
    }
    static constexpr int RL = 32, RC = 128;
    struct Snap {
        std::vector<double> I, Q; std::vector<long> n;
        int parity = -1; long f = -1; bool valid = false;
    };
    Snap prev1, prev2;
    int rx = 0, ry = 0;
    long frameIdx = 0;

    static void delta(const Snap &a, const Snap &b, int nReg,
                      double ampFloorRaw,
                      double &p50, double &p90, double &glob, int &nUse)
    {
        std::vector<double> dj;
        double gI = 0, gQ = 0;
        for (int r = 0; r < nReg; ++r) {
            if (a.n[r] < 64 || b.n[r] < 64) continue;
            const double ma = std::hypot(a.I[r], a.Q[r]) / a.n[r];
            const double mb = std::hypot(b.I[r], b.Q[r]) / b.n[r];
            if (ma < ampFloorRaw || mb < ampFloorRaw) continue;
            const double cI = b.I[r] * a.I[r] + b.Q[r] * a.Q[r];
            const double cQ = b.Q[r] * a.I[r] - b.I[r] * a.Q[r];
            dj.push_back(std::fabs(std::atan2(cQ, cI)) * 180.0 / M_PI);
            gI += cI; gQ += cQ;
        }
        nUse = (int)dj.size();
        if (dj.empty()) { p50 = p90 = glob = -1; return; }
        std::sort(dj.begin(), dj.end());
        p50 = dj[dj.size() / 2];
        p90 = dj[(size_t)(dj.size() * 0.9)];
        glob = std::atan2(gQ, gI) * 180.0 / M_PI;
    }

    // ---- Anticipating tracker (validation form) ----
    // Per-region alpha-beta loop on (phase, rate). On every covered frame
    // the tracker PREDICTS the region phase from state before consuming
    // the measurement — the reported miss is genuine held-out prediction
    // error, the number the consumption design will live on. A frame-wide
    // median miss above the cut threshold resets state (scene cut: content
    // hue changed wholesale; a reference must re-seed, not average across).
    struct Trk {
        bool valid = false;
        double zI = 1.0, zQ = 0.0;   // unit phase state
        double omega = 0.0;          // rad per output frame
        long lastF = -1;
    };
    std::vector<Trk> trk;
    long cuts = 0;
    static constexpr double kAlpha = 0.5, kBeta = 0.12;
    static constexpr double kCutDeg = 35.0;

    void track(long f, int nReg, const std::vector<double> &sI,
               const std::vector<double> &sQ, const std::vector<long> &sN,
               double floorRaw)
    {
        if ((int)trk.size() != nReg) { trk.assign(nReg, Trk()); }
        std::vector<double> miss;
        std::vector<int> used;
        for (int r = 0; r < nReg; ++r) {
            if (sN[r] < 64) continue;
            const double m = std::hypot(sI[r], sQ[r]) / sN[r];
            if (m < floorRaw) continue;
            used.push_back(r);
            Trk &T = trk[r];
            if (!T.valid) continue;
            const double dt = (double)(f - T.lastF);
            const double th = T.omega * dt;
            const double pI = T.zI * std::cos(th) - T.zQ * std::sin(th);
            const double pQ = T.zQ * std::cos(th) + T.zI * std::sin(th);
            const double eI = sI[r] * pI + sQ[r] * pQ;
            const double eQ = sQ[r] * pI - sI[r] * pQ;
            miss.push_back(std::fabs(std::atan2(eQ, eI)) * 180.0 / M_PI);
        }
        double p50 = -1, p90 = -1;
        bool cut = false;
        if (!miss.empty()) {
            std::sort(miss.begin(), miss.end());
            p50 = miss[miss.size() / 2];
            p90 = miss[(size_t)(miss.size() * 0.9)];
            cut = p50 > kCutDeg;
        }
        if (cut) {
            for (Trk &T : trk) T.valid = false;
            cuts++;
        }
        double sumOm = 0; int nOm = 0;
        for (int r : used) {
            Trk &T = trk[r];
            const double aI = sI[r], aQ = sQ[r];
            if (!T.valid) {
                const double m = std::hypot(aI, aQ);
                T.zI = aI / m; T.zQ = aQ / m;
                T.omega = 0.0; T.lastF = f; T.valid = true;
                continue;
            }
            const double dt = (double)(f - T.lastF);
            const double th = T.omega * dt;
            const double pI = T.zI * std::cos(th) - T.zQ * std::sin(th);
            const double pQ = T.zQ * std::cos(th) + T.zI * std::sin(th);
            const double eI = aI * pI + aQ * pQ;
            const double eQ = aQ * pI - aI * pQ;
            const double err = std::atan2(eQ, eI);
            T.omega += kBeta * err / dt;
            T.omega = std::clamp(T.omega, -0.35, 0.35); // ~20 deg/frame hull
            const double corr = th + kAlpha * err;
            const double cI = T.zI * std::cos(corr) - T.zQ * std::sin(corr);
            const double cQ = T.zQ * std::cos(corr) + T.zI * std::sin(corr);
            const double mz = std::hypot(cI, cQ);
            T.zI = cI / mz; T.zQ = cQ / mz;
            T.lastF = f;
            sumOm += std::fabs(T.omega); nOm++;
        }
        std::fprintf(stderr,
            "[TRK f=%ld] used=%zu predMiss p50=%.1f p90=%.1f deg  "
            "|omega|=%.2f deg/f  cut=%d cuts=%ld\n",
            f, used.size(), p50, p90,
            nOm ? (sumOm / nOm) * 180.0 / M_PI : 0.0, cut ? 1 : 0, cuts);
    }

    void frame(bool covered, int parity, int nx, int ny,
               std::vector<double> &sI, std::vector<double> &sQ,
               std::vector<long> &sN,
               double subA_I, double subA_Q, double subB_I, double subB_Q,
               double irescale)
    {
        const long f = frameIdx++;
        if (!covered) return;
        const int nReg = nx * ny;
        const double floorRaw = 2.0 * irescale;
        Snap cur; cur.I = sI; cur.Q = sQ; cur.n = sN;
        cur.parity = parity; cur.f = f; cur.valid = true;
        if (rx == nx && ry == ny) {
            double p50a=-1,p90a=-1,ga=0; int na=0;
            double p50b=-1,p90b=-1,gb=0; int nb=0;
            if (prev1.valid) delta(prev1, cur, nReg, floorRaw, p50a, p90a, ga, na);
            if (prev2.valid) delta(prev2, cur, nReg, floorRaw, p50b, p90b, gb, nb);
            // within-frame consistency: alternate def lines pooled apart
            double sub = -1;
            const double mA = std::hypot(subA_I, subA_Q);
            const double mB = std::hypot(subB_I, subB_Q);
            if (mA > 1e-9 && mB > 1e-9)
                sub = std::fabs(std::atan2(
                    subB_Q * subA_I - subB_I * subA_Q,
                    subB_I * subA_I + subB_Q * subA_Q)) * 180.0 / M_PI;
            std::fprintf(stderr,
                "[SYNC2 f=%ld par=%d] gap2(n=%d): p50=%.1f p90=%.1f g=%+.1f | "
                "gap4(n=%d): p50=%.1f p90=%.1f g=%+.1f | subgrp=%.1f\n",
                f, parity, na, p50a, p90a, ga, nb, p50b, p90b, gb, sub);
        }
        rx = nx; ry = ny;
        prev2 = prev1; prev1 = cur;
        track(f, nReg, sI, sQ, sN, floorRaw);
    }
};
SyncProbe g_syncProbe;

struct KneeProbe {
    std::mutex mu;
    static constexpr int kBins = 10;
    // Bin edges on |retracted top| in IRE.
    static constexpr double kEdge[kBins] =
        {1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 1e9};
    long   n[kBins]       = {};
    double sumRef[kBins]  = {};
    double sumComb[kBins] = {};
    long   nFlip[kBins]   = {};

    static bool on()
    {
        static const bool v = []{
            const char *s = std::getenv("LDCD_PROBE_KNEE");
            return s && std::atoi(s) != 0;
        }();
        return v;
    }

    void sample(double refTopIRE, double combTopIRE)
    {
        const double aRef = std::fabs(refTopIRE);
        int b = 0;
        while (b < kBins - 1 && aRef >= kEdge[b]) ++b;
        std::lock_guard<std::mutex> lk(mu);
        ++n[b];
        sumRef[b]  += aRef;
        sumComb[b] += std::fabs(combTopIRE);
        // Sign flip only counts where both carry real amplitude.
        if (aRef > 1.0 && std::fabs(combTopIRE) > 1.0 &&
            refTopIRE * combTopIRE < 0.0)
            ++nFlip[b];
    }

    ~KneeProbe()
    {
        if (!on()) return;
        long total = 0;
        for (int b = 0; b < kBins; ++b) total += n[b];
        if (total <= 0) return;
        std::fprintf(stderr,
            "\n[KNEE] proven-luma pixels %ld  (|retrTop| bins, IRE)\n"
            "[KNEE]   bin        n     ref    comb   comb/ref  flip%%\n",
            total);
        for (int b = 0; b < kBins; ++b) {
            if (n[b] <= 0) continue;
            const double inv = 1.0 / (double)n[b];
            const double mRef = sumRef[b] * inv, mComb = sumComb[b] * inv;
            std::fprintf(stderr,
                "[KNEE]  %5.1f-%-5.1f %8ld %6.2f %7.2f %9.3f %6.1f\n",
                b ? kEdge[b - 1] : 0.0,
                kEdge[b] > 1e8 ? 99.0 : kEdge[b],
                n[b], mRef, mComb,
                mRef > 1e-9 ? mComb / mRef : 0.0,
                100.0 * (double)nFlip[b] * inv);
        }
    }
};

constexpr double KneeProbe::kEdge[KneeProbe::kBins];

KneeProbe g_kneeProbe;

// Off-grid leakage stats (LDCD_PROBE_OFFGRID=1). Measurement only. Stage 1
// of the fit reset: a carrier waveform belongs to the span of the grammar
// basis over every legal 4-sample window (the two quadrature waveforms span
// the whole fSC subspace; off-span = DC + 2fSC content, which no lawful
// carrier may carry). The per-sample scalar surgeries between the fit's
// basis exit and publication push the emitted carrier out of that span, so
// raw - fit carries OFF-GRID alternations the election cannot compare with
// comb's on-grid residue. This measures the off-span energy fraction per
// published carrier source, binned by window amplitude; the differential
// against comb under the identical operator is the honest read (lawful
// envelope motion leaks a little in any 4-sample window for every source).
struct OffGridProbe {
    std::mutex mu;
    // [source: 0=fit 1=comb 2=locked1D][amplitude bin: <5, 5-15, >=15 IRE]
    long   n[3][3]      = {};
    double sumOff[3][3] = {};
    double sumTot[3][3] = {};

    static bool on()
    {
        static const bool v = []{
            const char *s = std::getenv("LDCD_PROBE_OFFGRID");
            return s && std::atoi(s) != 0;
        }();
        return v;
    }

    void sample(int src, int bin, double off, double tot)
    {
        std::lock_guard<std::mutex> lk(mu);
        ++n[src][bin];
        sumOff[src][bin] += off;
        sumTot[src][bin] += tot;
    }

    // Phase half: position, not span. A wrong-phase carrier is perfectly
    // in-span; what damages the election is the fit's carrier sitting off
    // POSITION -- rotated against the physical scalar and jittering window
    // to window. Measured at strong windows only.
    long   nPh = 0;
    double sumDPhase = 0.0, sumAbsDPhase = 0.0;   // fit vs comb, radians
    long   nJit[3] = {};
    double sumJit[3] = {};                        // per-source |dphase/window|

    void phasePair(double dphase)
    {
        std::lock_guard<std::mutex> lk(mu);
        ++nPh;
        sumDPhase += dphase;
        sumAbsDPhase += std::fabs(dphase);
    }

    void jitter(int src, double dphase)
    {
        std::lock_guard<std::mutex> lk(mu);
        ++nJit[src];
        sumJit[src] += std::fabs(dphase);
    }

    // Dropout half: strong-window stats exclude exactly the failure the beam
    // sheet showed -- windows where the fit's amplitude COLLAPSES while comb
    // still carries the chroma. Split by the legality proof: at proven-
    // illegal energy the "dropout" is the fit correctly refusing what comb
    // wrongly models (virtuous); at legal carrier it is lost lock (the
    // defect the reset must cure).
    long nStrong[2] = {}, nDrop[2] = {};   // [0]=legal-ish, [1]=proven illegal

    void dropout(bool fitDropped, bool provenIllegal)
    {
        std::lock_guard<std::mutex> lk(mu);
        ++nStrong[provenIllegal ? 1 : 0];
        if (fitDropped) ++nDrop[provenIllegal ? 1 : 0];
    }

    ~OffGridProbe()
    {
        if (!on()) return;
        static const char *kSrc[3] = {"fit      ", "comb     ", "locked1D "};
        static const char *kBin[3] = {"<5 IRE ", "5-15   ", ">=15   "};
        bool any = false;
        for (int s = 0; s < 3 && !any; ++s)
            for (int b = 0; b < 3; ++b)
                if (n[s][b]) { any = true; break; }
        if (!any) return;
        std::fprintf(stderr,
            "\n[OFFGRID] off-span energy fraction of the published carrier "
            "(4-sample grammar-basis windows)\n");
        for (int s = 0; s < 3; ++s) {
            std::fprintf(stderr, "[OFFGRID] %s", kSrc[s]);
            for (int b = 0; b < 3; ++b) {
                if (n[s][b] > 0 && sumTot[s][b] > 1e-9)
                    std::fprintf(stderr, "  %s%5.2f%% (n=%ld)", kBin[b],
                                 100.0 * sumOff[s][b] / sumTot[s][b],
                                 n[s][b]);
                else
                    std::fprintf(stderr, "  %s    --", kBin[b]);
            }
            std::fprintf(stderr, "\n");
        }
        if (nPh > 0)
            std::fprintf(stderr,
                "[OFFGRID] phase(fit vs comb) strong windows: mean %+.1f deg  "
                "|mean| %.1f deg  (n=%ld)\n",
                sumDPhase / nPh * 180.0 / M_PI,
                sumAbsDPhase / nPh * 180.0 / M_PI, nPh);
        static const char *kSrc2[3] = {"fit", "comb", "locked1D"};
        for (int s = 0; s < 3; ++s)
            if (nJit[s] > 0)
                std::fprintf(stderr,
                    "[OFFGRID] phase jitter %s: %.1f deg/window (n=%ld)\n",
                    kSrc2[s], sumJit[s] / nJit[s] * 180.0 / M_PI, nJit[s]);
        if (nStrong[0] + nStrong[1] > 0)
            std::fprintf(stderr,
                "[OFFGRID] fit amplitude dropout at comb-strong windows: "
                "legal %.1f%% (n=%ld)  proven-illegal %.1f%% (n=%ld)\n",
                nStrong[0] ? 100.0 * (double)nDrop[0] / (double)nStrong[0] : 0.0,
                nStrong[0],
                nStrong[1] ? 100.0 * (double)nDrop[1] / (double)nStrong[1] : 0.0,
                nStrong[1]);
    }
};

OffGridProbe g_offGrid;

// Feasible-luma restraint census (LDCD_LURCH_FEASIBLE=1). Measurement only.
struct LumaFeasProbe {
    std::mutex mu;
    long nBaseIn = 0, nBaseOut = 0;
    double sumIn = 0.0, maxIn = 0.0;
    static bool on() {
        static const bool v = std::getenv("LDCD_LURCH_FEASIBLE") != nullptr;
        return v;
    }
    void hit(double dIRE, bool baseIn) {
        std::lock_guard<std::mutex> lk(mu);
        if (baseIn) {
            ++nBaseIn; sumIn += dIRE;
            if (dIRE > maxIn) maxIn = dIRE;
        } else {
            ++nBaseOut;
        }
    }
    ~LumaFeasProbe() {
        if (!on() || (nBaseIn + nBaseOut) == 0) return;
        std::fprintf(stderr,
            "[LUMAFEAS] lurch overshoot clamped: %ld samples, mean %.2f IRE, "
            "max %.2f IRE | platform already infeasible (untouched): %ld\n",
            nBaseIn, nBaseIn ? sumIn / nBaseIn : 0.0, maxIn, nBaseOut);
    }
};
LumaFeasProbe g_lumaFeas;

} // namespace

// Locked-path pre-processing: burst detection, carrier grammar, and luma cache.
void Comb::FrameBuffer::phaseLocked()
{
    if (!configuration.phaseCompensation)
        return;

    const int left       = videoParameters.activeVideoStart;
    const int right      = videoParameters.activeVideoEnd;
    const int firstLine  = videoParameters.firstActiveFrameLine;
    const int lastLine   = videoParameters.lastActiveFrameLine;
    const int fullWidth  = videoParameters.fieldWidth;
    const int width      = right - left;

    if (left >= right || firstLine >= lastLine)
        return;

    const int requiredLines = lastLine + 1;
    if ((int)carrierGrammar.size() < requiredLines)
        carrierGrammar.resize(requiredLines);
    // Basis coefficients — computed once, used by all passes
    double Ce = 1.0, Se = 0.0;
    basisCoeffs(Ce, Se);
    for (int i = 0; i < 4; ++i) {
        double sp, cp;
        shiftedBasis(i, Ce, Se, sp, cp);
        spLUT_locked[i] = sp;
        cpLUT_locked[i] = cp;
    }
    basisLockedInit = true;

    const double K   = 0.5 * M_PI;
    const double Rb0 = -K * CAL_EPS_SAMPLES + (CAL_LO_ROT_DEG * M_PI / 180.0);
    const double cRb = std::cos(Rb0);
    const double sRb = std::sin(Rb0);

    const bool   floorEnable = configuration.burstFloorEnable;
    const double floorFactor = configuration.burstFloorFactor;
    const auto  &T           = configuration.tunables;
    constexpr double MIN_PHASE_CONFIDENCE = 1e-6;

    // --- Pass 1: burst detection -> carrier grammar ---
    for (int line = firstLine; line < lastLine; ++line) {
        const quint16 *rawLine = rawbuffer.data() + line * fullWidth;
        auto burst = detectBurst(rawLine, videoParameters, floorEnable, floorFactor);
        double bcos = burst.bcos, bsin = burst.bsin;
        const double bc2 = bcos * cRb - bsin * sRb;
        const double bs2 = bcos * sRb + bsin * cRb;
        CombCarrierGrammar &grammar = carrierGrammar[line];
        grammar.burstCos = bc2;
        grammar.burstSin = bs2;
        grammar.carrierScale = burst.carrierScale * invIreScale;
        grammar.phaseConfidence =
            std::clamp((grammar.carrierScale - 3.0) / 7.0, 0.0, 1.0);
        grammar.grammarLocked = grammar.phaseConfidence > MIN_PHASE_CONFIDENCE;
        grammar.phaseError = 0.0;
        grammar.phaseScheduleConflict = 0.0;
        grammar.affine.valid = false;

        double lutTi[4], lutTq[4];
        fusedDemodLUT(bc2, bs2, spLUT_locked, cpLUT_locked, lutTi, lutTq);
        for (int i = 0; i < 4; ++i) {
            grammar.demodLUTTi[i] = (float)lutTi[i];
            grammar.demodLUTTq[i] = (float)lutTq[i];
        }
    }

    if (!lockedLumaBaseY4_flat.empty() &&
        !lockedLumaSmooth_flat.empty() &&
        demodWidth == width)
    {
        // Sharpened boxcar coarse platform for produceY. This is the
        // lurch-corrected LF authority and it lives ONLY in the --luma-witness
        // fork: default reconstructs on lockedLumaBaseY4, so the sharp buffer
        // is left unallocated (comb.cpp) and the build below is skipped. Under
        // witness it is built when the sweep knob asks for it. Same
        // construction the constrained witness used: a
        // sliding 4-sample boxcar (carrier-cancelled per aperture, evaluated
        // every sample) lurch-sharpened so a confirmed luma step lands at one
        // column instead of smearing across four. The gate is scaled by the
        // sweep level.
        static const bool lumaFeasOn = []{
            const char *e = std::getenv("LDCD_LURCH_FEASIBLE");
            return e && std::atoi(e) != 0;
        }();
        const double sharpLevel = coarseSharpLevel();
        const bool buildSharp =
            configuration.lumaWitness &&
            sharpLevel > 0.0 && !lockedLumaSharp_flat.empty();
        std::vector<double> gateScratch;
        if (buildSharp && width >= 4)
            gateScratch.assign(width, 0.0);

        for (int line = firstLine; line < lastLine; ++line) {
            const quint16 *rawLine = rawbuffer.data() + line * fullWidth;
            buildCompositeLumaDecompositionLine(rawLine, left, width,
                                                lockedLumaBaseY4_line(line),
                                                nullptr,
                                                lockedLumaSmooth_line(line));

            // Vertical-contrast service: the 1D is the first stage to cross a
            // vertical contrast, so the lateral coarse delta is registered here
            // once for every later client (Frame B reach exemption,
            // hLumaDeltaIRE, cross-color, FVF vertical regime).
            if (float *hDelta = lockedLumaHDeltaIRE_line(line)) {
                const double *smooth = lockedLumaSmooth_line(line);
                for (int rel = 0; rel < width; ++rel) {
                    const int rm = std::max(0, rel - 2);
                    const int rp = std::min(width - 1, rel + 2);
                    hDelta[rel] = static_cast<float>(
                        std::fabs(smooth[rp] - smooth[rm]) * invIreScale);
                }
            }

            // The coarse-residual aperture-mean pool is built once for every
            // path by buildApertureMeans() (from split1D, before this runs);
            // read it here for the sharpened boxcar below.
            double *apMean = lockedApertureMean_line(line);

            if (!buildSharp)
                continue;

            double *sharp = lockedLumaSharp_line(line);
            if (width < 4) {
                std::copy(lockedLumaSmooth_line(line),
                          lockedLumaSmooth_line(line) + width, sharp);
                continue;
            }
            // Derived FROM the pool above, not a private rebuild.
            const double *boxcar = apMean;
            const int lastStart = width - 4;   // last legal aperture start

            // Register the even four-sample means at integer xi by averaging
            // the two half-sample apertures on either side.  Their combination
            // is the phase-balanced five-sample support
            // (0.5,1,1,1,0.5)/4 centred exactly at xi.
            for (int xi = 0; xi < width; ++xi) {
                const int s0 = std::clamp(xi - 2, 0, lastStart);
                const int s1 = std::clamp(xi - 1, 0, lastStart);
                sharp[xi] = 0.5 * (boxcar[s0] + boxcar[s1]);
            }
            // Canonical runs (built once in split1D), edges vertically
            // corroborated (median-of-three) so the snap stops sawing
            // bright vertical contours; apply-only here.
            const std::vector<LurchStepRun> corrRuns =
                corroborateLurchEdges(line);
            std::vector<double> preSharp;
            if (lumaFeasOn) preSharp.assign(sharp, sharp + width);
            applyLurchSteps(corrRuns, boxcar, width - 3,
                            width, sharpLevel, sharp, gateScratch.data());

            // FEASIBLE-LUMA RESTRAINT (LDCD_LURCH_FEASIBLE=1, default OFF).
            //
            // Lurch earns its place because the four-sample coarse smears
            // boundaries and lurch un-smears them. So the restriction cannot
            // be "sharpen less"; it has to be "never sharpen to a value that
            // cannot be true". feasibleband.h's pair-sum law supplies exactly
            // that, and for LUMA rather than carrier: the carrier is
            // antisymmetric over +-2 samples, so
            //     composite[i] + composite[i+-2] = Y[i] + Y[i+-2]
            // is carrier-free and EXACT. With luma bounded, each available
            // neighbour pins Y[i] to an interval. A sharpened platform
            // outside that interval asserts a luma the composite forbids.
            //
            // This is the primitive's first consumer. It forbids impossibles
            // and says nothing about which surviving value is preferred --
            // no threshold, nothing fitted, no per-scene constant.
            //
            // The probe separates two cases, because only the first is about
            // lurch: BASE-IN means the unsharpened platform was feasible and
            // the sharpening pushed it out (lurch overshoot); BASE-OUT means
            // the platform was already infeasible before lurch touched it,
            // which is a different defect and not this restraint's business.
            if (lumaFeasOn) {
                const double yLo = videoParameters.black16bIre - 10.0 * irescale;
                const double yHi = videoParameters.white16bIre + 10.0 * irescale;
                for (int xi = 0; xi < width; ++xi) {
                    const int h = left + xi;
                    double nb[2];
                    int nn = 0;
                    if (h - 2 >= 0)         nb[nn++] = (double)rawLine[h - 2];
                    if (h + 2 < fullWidth)  nb[nn++] = (double)rawLine[h + 2];
                    const lddecode::FeasibleInterval f =
                        lddecode::lumaFeasibleFromPairSums(
                            (double)rawLine[h], nb, nn, yLo, yHi);
                    const double v = sharp[xi];
                    const double c = f.clamp(v);
                    static int dbg = 0;
                    if (dbg < 12 && xi > 300 && xi < 320) {
                        ++dbg;
                        std::fprintf(stderr,
                            "[FEASDBG] xi=%d sharp=%.0f pre=%.0f raw=%.0f "
                            "lo=%.0f hi=%.0f nn=%d\n",
                            xi, v, preSharp[xi], (double)rawLine[h],
                            f.lo, f.hi, nn);
                    }
                    if (c == v) continue;
                    const double base = preSharp[xi];
                    const bool baseIn = (f.clamp(base) == base);
                    g_lumaFeas.hit(std::fabs(v - c) * invIreScale, baseIn);
                    if (baseIn) sharp[xi] = c;
                }
            }
        }
        lockedLumaCacheValid = true;
    }

    return;
}

// Demodulate the blind 1D bandpass through the locked carrier grammar, publish
// the common-4fsc IQ used by the 2D candidate builders, and remodulate the
// phase-corrected scalar source used by locked 1D/2D/3D selection.
//
// This version adds a cheap local anti cross color pass:
//
//   - first pass:  demodulate the phase-locked 1D scalar into local I/Q
//   - second pass: preserve local I/Q DC, damp local AC/winding when it looks
//                  like edge-generated or dubbed cross-color
//   - publish:     write demod buffers, locked 4fsc buffers, magnitude, and
//                  lockedSource from the cleaned I/Q

void Comb::FrameBuffer::buildCarrierAnalysis(FrameBuffer *prevFrame)
{
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;
    const int fullWidth = videoParameters.fieldWidth;

    static const int crDiagLine = []{
        const char *s = std::getenv("COARSE_RESID_DIAG_LINE");
        return s ? std::atoi(s) : -1;
    }();
    static const int crDiagC0 = []{
        const char *s = std::getenv("COARSE_RESID_DIAG_C0");
        return s ? std::atoi(s) : -1;
    }();
    static const int crDiagC1 = []{
        const char *s = std::getenv("COARSE_RESID_DIAG_C1");
        return s ? std::atoi(s) : -1;
    }();
    static const double parallaxRepairTolIRE = []{
        const char *s = std::getenv("LD_1D_PARALLAX_TOL_IRE");
        return s ? std::atof(s) : 0.5;
    }();

    if (!configuration.phaseCompensation || width <= 0 || first >= last ||
        demodWidth != width || demodLines < last)
        return;

    carrierRetractionModelValid = false;

    const size_t count = static_cast<size_t>(demodLines) * demodWidth;
    if (carrierAnalysis_flat.size() < count) {
        carrierAnalysis_flat.assign(count, lddecode::CarrierAnalysisRecord{});
    }

    constexpr int kNarrowWin = 16;
    constexpr int kWideWin = 32;
    static const double cosRef[4] = { 1.0, 0.0, -1.0, 0.0 };
    static const double sinRef[4] = { 0.0, 1.0,  0.0, -1.0 };

    std::vector<double> preI(width + 1, 0.0);
    std::vector<double> preQ(width + 1, 0.0);

    for (int line = first; line < last; ++line) {
        const quint16 *rawLine =
            rawbuffer.data() + static_cast<size_t>(line) * fullWidth;
        double *baseline = locked1DRawBandpass_line(line);
        if (!baseline)
            continue;

        auto rawAtRel = [&](int rel) -> double {
            const int r = std::clamp(rel, 0, width - 1);
            return static_cast<double>(rawLine[left + r]);
        };

        // Canonical full-resolution source authority used by every later
        // carrier client. Keep this formula and its edge convention in one
        // place so analysis and rendering cannot drift apart.
        //
        // The ±2 bandpass taps use half-sample edge REFLECTION, matching the
        // bucket split1D convention ([-1]->[0], [-2]->[1] at the left; mirror
        // at the right). Edge-clamp (repeating the boundary sample) injects a
        // DC pedestal at the wrong carrier phase, so the aperture goes
        // asymmetric at the frame boundary and the bandpass fires on that
        // false step — a spurious carrier fringe along the picture edge that
        // also pollutes the interline schedule fingerprint out there.
        // Reflection continues the waveform instead. (Only the bandpass taps
        // reflect; the coarse-residual/luma-floor reads below keep rawAtRel's
        // clamp — those four-sample means never wanted reflection.)
        auto rawMirror = [&](int rel) -> double {
            if (rel < 0)              rel = -rel - 1;
            else if (rel >= width)    rel = 2 * width - 1 - rel;
            rel = std::clamp(rel, 0, width - 1);   // safety for tiny widths
            return static_cast<double>(rawLine[left + rel]);
        };
        for (int rel = 0; rel < width; ++rel) {
            const double c  = static_cast<double>(rawLine[left + rel]);
            const double m2 = rawMirror(rel - 2);
            const double p2 = rawMirror(rel + 2);
            baseline[rel] = 0.50 * c - 0.25 * (m2 + p2);
        }

        seedCombAttributionPerLine(line);
        lddecode::CarrierAnalysisRecord *analysis = carrierAnalysis_line(line);
        if (analysis) {
            std::fill(
                analysis,
                analysis + width,
                lddecode::CarrierAnalysisRecord{});
        }

        if (crDiagLine >= 0 && line == crDiagLine && crDiagC0 >= 0) {
            const int c0 = std::clamp(crDiagC0, 0, width - 1);
            const int c1 = std::clamp(
                crDiagC1 < 0 ? crDiagC0 : crDiagC1,
                c0,
                width - 1);

            std::fprintf(stderr,
                "COARSERESOPT header line rel h phase raw oldBp viewCount "
                "resLo resHi resSpreadIRE maxAbsMembershipIRE view s y4 residual "
                "residualIRE membershipDeltaIRE membershipLocalX\n");

            for (int rel = c0; rel <= c1; ++rel) {
                const int sFirst = std::max(0, rel - 3);
                const int sLast = width >= 4 ? std::min(rel, width - 4) : -1;
                double residuals[4] = {0.0, 0.0, 0.0, 0.0};
                double memberships[4] = {0.0, 0.0, 0.0, 0.0};
                double localXs[4] = {0.0, 0.0, 0.0, 0.0};
                double y4s[4] = {0.0, 0.0, 0.0, 0.0};
                int starts[4] = {0, 0, 0, 0};
                double lo = 1e300;
                double hi = -1e300;
                double maxMembership = 0.0;
                int n = 0;

                for (int s = sFirst; s <= sLast && n < 4; ++s, ++n) {
                    const double y4 = 0.25 * (
                        rawAtRel(s + 0) + rawAtRel(s + 1) +
                        rawAtRel(s + 2) + rawAtRel(s + 3));
                    const double residual = rawAtRel(rel) - y4;
                    double membership = 0.0;
                    double localX = 0.0;
                    if (s + 4 < width) {
                        membership = 0.25 *
                            (rawAtRel(s + 4) - rawAtRel(s)) * invIreScale;
                        localX = 0.5 * (static_cast<double>(s) +
                                        static_cast<double>(s + 4)) - rel;
                    }
                    starts[n] = s;
                    y4s[n] = y4;
                    residuals[n] = residual;
                    memberships[n] = membership;
                    localXs[n] = localX;
                    lo = std::min(lo, residual);
                    hi = std::max(hi, residual);
                    maxMembership = std::max(maxMembership, std::fabs(membership));
                }

                if (n == 0) {
                    std::fprintf(stderr,
                        "COARSERESOPT line=%d rel=%d h=%d phase=%d raw=%.6f "
                        "oldBp=%.6f viewCount=0 resLo=0.000000 resHi=0.000000 "
                        "resSpreadIRE=0.000000 maxAbsMembershipIRE=0.000000 "
                        "view=-1 s=-1 y4=0.000000 residual=0.000000 "
                        "residualIRE=0.000000 membershipDeltaIRE=0.000000 "
                        "membershipLocalX=0.000000\n",
                        line, rel, left + rel,
                        carrierSampleClass(line, left + rel) & 3,
                        rawAtRel(rel), baseline[rel]);
                    continue;
                }

                const double spreadIRE = (hi - lo) * invIreScale;
                for (int v = 0; v < n; ++v) {
                    std::fprintf(stderr,
                        "COARSERESOPT line=%d rel=%d h=%d phase=%d raw=%.6f "
                        "oldBp=%.6f viewCount=%d resLo=%.6f resHi=%.6f "
                        "resSpreadIRE=%.6f maxAbsMembershipIRE=%.6f "
                        "view=%d s=%d y4=%.6f residual=%.6f residualIRE=%.6f "
                        "membershipDeltaIRE=%.6f membershipLocalX=%.6f\n",
                        line, rel, left + rel,
                        carrierSampleClass(line, left + rel) & 3,
                        rawAtRel(rel), baseline[rel], n, lo, hi, spreadIRE,
                        maxMembership, v, starts[v], y4s[v], residuals[v],
                        residuals[v] * invIreScale, memberships[v], localXs[v]);
                }
            }
        }

        if (!analysis)
            continue;

        preI[0] = 0.0;
        preQ[0] = 0.0;
        for (int rel = 0; rel < width; ++rel) {
            const int p = carrierSampleClass(line, left + rel) & 3;
            preI[rel + 1] = preI[rel] + baseline[rel] * cosRef[p];
            preQ[rel + 1] = preQ[rel] + baseline[rel] * sinRef[p];
        }

        const double tolSamples =
            std::max(0.0, parallaxRepairTolIRE) * irescale;
        for (int rel = 0; rel < width; ++rel) {
            const int p = carrierSampleClass(line, left + rel) & 3;
            const int na = std::clamp(rel - kNarrowWin / 2, 0, width);
            const int nb = std::clamp(na + kNarrowWin, 0, width);
            const double nn = static_cast<double>(std::max(1, nb - na));
            const double ZnI = (preI[nb] - preI[na]) / nn;
            const double ZnQ = (preQ[nb] - preQ[na]) / nn;
            const double shortSample =
                2.0 * (ZnI * cosRef[p] + ZnQ * sinRef[p]);

            const int wa = std::clamp(rel - kWideWin / 2, 0, width);
            const int wb = std::clamp(wa + kWideWin, 0, width);
            const double wn = static_cast<double>(std::max(1, wb - wa));
            const double ZwI = (preI[wb] - preI[wa]) / wn;
            const double ZwQ = (preQ[wb] - preQ[wa]) / wn;
            const double wideSample =
                2.0 * (ZwI * cosRef[p] + ZwQ * sinRef[p]);

            lddecode::CarrierResidualOption options[4];
            int optionCount = 0;
            const int sFirst = std::max(0, rel - 3);
            const int sLast = width >= 4 ? std::min(rel, width - 4) : -1;
            for (int s = sFirst; s <= sLast && optionCount < 4; ++s) {
                const double y4 = 0.25 * (
                    rawAtRel(s + 0) + rawAtRel(s + 1) +
                    rawAtRel(s + 2) + rawAtRel(s + 3));
                options[optionCount].sample = rawAtRel(rel) - y4;
                options[optionCount].membershipDeltaIRE =
                    s + 4 < width
                        ? 0.25 * (rawAtRel(s + 4) - rawAtRel(s)) * invIreScale
                        : 0.0;
                ++optionCount;
            }

            const double movingMean = 0.25 * (
                rawAtRel(rel - 1) + rawAtRel(rel) +
                rawAtRel(rel + 1) + rawAtRel(rel + 2));
            const double movingResidual = rawAtRel(rel) - movingMean;
            const double sourceSample = baseline[rel];

            auto &record = analysis[rel];
            record.fit.sourceSample = static_cast<float>(sourceSample);
            record.fit.shortSample = static_cast<float>(shortSample);
            record.fit.wideSample = static_cast<float>(wideSample);
            record.fit.sourceMinusShortIRE = static_cast<float>(
                (sourceSample - shortSample) * invIreScale);
            record.fit.shortMinusWideIRE = static_cast<float>(
                (shortSample - wideSample) * invIreScale);
            record.fit.sourceMinusWideIRE = static_cast<float>(
                (sourceSample - wideSample) * invIreScale);
            record.fit.valid = true;
            record.residual = lddecode::analyzeCarrierResidualOptions(
                options,
                optionCount,
                shortSample,
                tolSamples,
                std::max(0.0, parallaxRepairTolIRE),
                movingResidual,
                invIreScale);
        }
    }

    // ------------------------------------------------------------------
    // Schedule-conformance registration (grammar-as-table).
    //
    // With every line's canonical bandpass harvested, register each
    // pixel's carrier-band energy against the schedule: legal carrier
    // MUST invert across Opposite-relation partners — the same-field ±2
    // lines within this frame, and the same line on the neighbouring
    // frame (the comb's own temporal structure). Energy that MATCHES where the
    // schedule demands inversion admits no legal carrier interpretation
    // and is registered ScheduleIllegal — luma by law — at entry, rather
    // than entering the carrier column with a bad grade for every
    // consumer to re-adjudicate.  Same-relation partners are
    // non-discriminative (legal and alien both match) and abstain.
    // 4-sample windows keep the correlation phase-flat; an energy floor
    // keeps noise Unresolved.  Conservative tie-break: any legal vote
    // wins — real chroma is never claimed as luma.
    if (width >= 4) {
        const bool prevUsable =
            prevFrame &&
            prevFrame->demodWidth == demodWidth &&
            prevFrame->demodLines == demodLines &&
            !prevFrame->locked1DRawBandpass_flat.empty();

        const double rmsFloor = 3.0 * irescale;           // 3 IRE RMS
        const double energyFloor = 4.0 * rmsFloor * rmsFloor;
        constexpr double kCorrVote = 0.5;

        // --- Disposable schedule-conformance instrumentation (env-gated) ---
        // Set LDCD_DUMP_CONFORMANCE=1 to print per-frame verdict statistics
        // to stderr. Zero cost when unset. Remove with the rethink.
        const bool dumpConf = std::getenv("LDCD_DUMP_CONFORMANCE") != nullptr;
        long long cBelowFloor = 0, cUnres = 0, cLegal = 0, cIllegal = 0, cNoAxis = 0;
        long long illBy[4]        = {0, 0, 0, 0}; // illegal grouped by illegal-axis-vote count (1..3)
        long long illAmp[4]       = {0, 0, 0, 0}; // illegal grouped by per-sample RMS IRE bucket
        long long usableAxisHist[4] = {0, 0, 0, 0};
        long long strongTot = 0, strongLegal = 0, strongIllegal = 0, strongUnres = 0;
        long long thirdTot[3] = {0, 0, 0}, thirdIll[3] = {0, 0, 0};

        for (int line = first; line < last; ++line) {
            lddecode::CarrierAnalysisRecord *analysis =
                carrierAnalysis_line(line);
            const double *bp0 = locked1DRawBandpass_line(line);
            const CombCarrierGrammar *g0 = carrierGrammarLine(line);
            if (!analysis || !bp0 || !g0 || !g0->grammarLocked)
                continue;

            // Opposite-relation partner rows, certified by the grammar.
            const double *axes[3];
            int nAxes = 0;
            auto addAxis = [&](const double *bpP,
                               const CombCarrierGrammar *gP) {
                if (!bpP || !gP || !gP->grammarLocked || nAxes >= 3)
                    return;
                const auto rel = lddecode::carrierGrammarSignedPhaseRelation(
                    g0, left, gP, left);
                if (rel == lddecode::CarrierPhaseRelation::Opposite)
                    axes[nAxes++] = bpP;
            };
            if (line - 2 >= first)
                addAxis(locked1DRawBandpass_line(line - 2),
                        carrierGrammarLine(line - 2));
            if (line + 2 < last)
                addAxis(locked1DRawBandpass_line(line + 2),
                        carrierGrammarLine(line + 2));
            if (prevUsable)
                addAxis(prevFrame->locked1DRawBandpass_line(line),
                        prevFrame->carrierGrammarLine(line));
            if (nAxes == 0) {
                if (dumpConf)
                    cNoAxis += width;
                continue;
            }

            for (int rel = 0; rel < width; ++rel) {
                // Centered 5-tap, half-weight ends (integer centroid): the
                // old [rel, rel+3] window registered every conformance
                // verdict 1.5 samples right of the sample it judged.
                double e0 = 0.0;
                for (int k = -2; k <= 2; ++k) {
                    const int j = std::clamp(rel + k, 0, width - 1);
                    const double w = (k == -2 || k == 2) ? 0.5 : 1.0;
                    e0 += w * bp0[j] * bp0[j];
                }
                if (e0 < energyFloor) {
                    if (dumpConf)
                        ++cBelowFloor;
                    continue;
                }

                bool legalVote = false;
                bool illegalVote = false;
                int legalAxisVotes = 0, illegalAxisVotes = 0, usableAxes = 0;
                double minCorr =  1e300;   // most-legal (negative) axis
                double maxCorr = -1e300;   // most-illegal (positive) axis
                for (int a = 0; a < nAxes; ++a) {
                    const double *bpP = axes[a];
                    double dot = 0.0;
                    double eP = 0.0;
                    for (int k = -2; k <= 2; ++k) {
                        const int j = std::clamp(rel + k, 0, width - 1);
                        const double w = (k == -2 || k == 2) ? 0.5 : 1.0;
                        dot += w * bp0[j] * bpP[j];
                        eP  += w * bpP[j] * bpP[j];
                    }
                    if (eP < energyFloor)
                        continue;
                    ++usableAxes;
                    const double corr = dot / std::sqrt(e0 * eP);
                    minCorr = std::min(minCorr, corr);
                    maxCorr = std::max(maxCorr, corr);
                    if (corr <= -kCorrVote) {
                        legalVote = true;
                        ++legalAxisVotes;
                    } else if (corr >= kCorrVote) {
                        illegalVote = true;
                        ++illegalAxisVotes;
                    }
                }

                // Scanner layer (grammar-as-table): publish the graded
                // MEASUREMENT, not a decision.  carrierConformance is the
                // relation-signed correlation, biased to the most-legal axis
                // when one inverts (real chroma is never disowned on the
                // strength of a matching neighbour) and otherwise reporting
                // the most-illegal evidence. conformanceSupportFraction records
                // the fraction of axes SUPPORTING that selected sign, not the
                // number merely available.  The old usableAxes/3 assigned a
                // lone coincidental legal vote full authority when all three
                // axes were present. Thresholding conformance at
                // -/+kCorrVote reproduces the legacy enum; action remains a
                // downstream policy.
                if (usableAxes > 0) {
                    const double conformance = legalVote ? minCorr : maxCorr;
                    const int supportingVotes = legalVote
                        ? legalAxisVotes : illegalAxisVotes;
                    // Axes decisively voting AGAINST the selected sign.
                    // Abstaining axes appear in neither fraction.
                    const int contradictingVotes = legalVote
                        ? illegalAxisVotes : legalAxisVotes;
                    analysis[rel].carrierConformance =
                        static_cast<float>(std::clamp(conformance, -1.0, 1.0));
                    analysis[rel].conformanceUsableAxisFraction =
                        static_cast<float>(std::min(1.0,
                            static_cast<double>(usableAxes) / 3.0));
                    analysis[rel].conformanceSupportFraction =
                        static_cast<float>(std::min(1.0,
                            static_cast<double>(supportingVotes) / 3.0));
                    analysis[rel].conformanceContradictionFraction =
                        static_cast<float>(std::min(1.0,
                            static_cast<double>(contradictingVotes) / 3.0));
                }

                analysis[rel].scheduleConformance = legalVote
                    ? lddecode::CarrierScheduleConformance::LegalCarrier
                    : illegalVote
                        ? lddecode::CarrierScheduleConformance::ScheduleIllegal
                        : lddecode::CarrierScheduleConformance::Unresolved;

                if (dumpConf) {
                    const double rmsIRE = std::sqrt(e0 * 0.25) * invIreScale;
                    const int third = std::clamp(
                        (line - first) * 3 / std::max(1, last - first), 0, 2);
                    const bool strong = rmsIRE >= 10.0;
                    ++thirdTot[third];
                    ++usableAxisHist[std::clamp(usableAxes, 0, 3)];
                    if (strong)
                        ++strongTot;
                    const auto verdict = analysis[rel].scheduleConformance;
                    if (verdict == lddecode::CarrierScheduleConformance::LegalCarrier) {
                        ++cLegal;
                        if (strong)
                            ++strongLegal;
                    } else if (verdict ==
                               lddecode::CarrierScheduleConformance::ScheduleIllegal) {
                        ++cIllegal;
                        ++thirdIll[third];
                        if (strong)
                            ++strongIllegal;
                        ++illBy[std::clamp(illegalAxisVotes, 0, 3)];
                        const int bkt = rmsIRE < 5.0 ? 0
                                      : rmsIRE < 10.0 ? 1
                                      : rmsIRE < 20.0 ? 2 : 3;
                        ++illAmp[bkt];
                    } else {
                        ++cUnres;
                        if (strong)
                            ++strongUnres;
                    }
                }
            }
        }

        if (dumpConf) {
            auto pct = [](long long a, long long b) {
                return b > 0 ? 100.0 * static_cast<double>(a)
                                     / static_cast<double>(b)
                             : 0.0;
            };
            const long long tested = cLegal + cIllegal + cUnres;
            std::fprintf(stderr,
                "[CONF] lines=%d..%d tested=%lld legal=%lld(%.1f%%) "
                "illegal=%lld(%.1f%%) unres=%lld(%.1f%%) belowFloor=%lld noAxisPix=%lld\n",
                first, last, tested,
                cLegal, pct(cLegal, tested),
                cIllegal, pct(cIllegal, tested),
                cUnres, pct(cUnres, tested),
                cBelowFloor, cNoAxis);
            std::fprintf(stderr,
                "[CONF] illegal-by-axisvotes 1=%lld 2=%lld 3=%lld | "
                "usableAxisHist a1=%lld a2=%lld a3=%lld\n",
                illBy[1], illBy[2], illBy[3],
                usableAxisHist[1], usableAxisHist[2], usableAxisHist[3]);
            std::fprintf(stderr,
                "[CONF] illegal-by-amp(IRE) <5=%lld 5-10=%lld 10-20=%lld >=20=%lld\n",
                illAmp[0], illAmp[1], illAmp[2], illAmp[3]);
            std::fprintf(stderr,
                "[CONF] strong(>=10IRE) tot=%lld legal=%lld(%.1f%%) "
                "illegal=%lld(%.1f%%) unres=%lld(%.1f%%)\n",
                strongTot,
                strongLegal, pct(strongLegal, strongTot),
                strongIllegal, pct(strongIllegal, strongTot),
                strongUnres, pct(strongUnres, strongTot));
            std::fprintf(stderr,
                "[CONF] illegal-by-third top=%.1f%% mid=%.1f%% bot=%.1f%%\n",
                pct(thirdIll[0], thirdTot[0]),
                pct(thirdIll[1], thirdTot[1]),
                pct(thirdIll[2], thirdTot[2]));
        }
    }

    // --- Disposable 1D-fingerprint dump (env-gated). The three orthogonal
    // views of "luma entered the bandpass", per pixel along a scanline:
    //   incoh = sourceMinusWideIRE  (horizontal: source minus wide coherent fit)
    //   lurch = maxAbsMembershipIRE (carrier-free luma movement through aperture)
    //   conf  = carrierConformance  (interline: -1 inverts like carrier,
    //                                +1 matches where schedule demands inversion
    //                                = image-locked luma leak)
    // Enable with LDCD_DUMP_FP_L (frame line) and LDCD_DUMP_FP_C0/C1 (active
    // column range = h - left). Run -t 1. Zero cost when unset.
    static const int fpLine = []{ const char *s = std::getenv("LDCD_DUMP_FP_L"); return s ? std::atoi(s) : -1; }();
    static const int fpC0   = []{ const char *s = std::getenv("LDCD_DUMP_FP_C0"); return s ? std::atoi(s) : -1; }();
    static const int fpC1   = []{ const char *s = std::getenv("LDCD_DUMP_FP_C1"); return s ? std::atoi(s) : -1; }();
    if (fpLine >= first && fpLine < last && fpC0 >= 0) {
        const lddecode::CarrierAnalysisRecord *rec = carrierAnalysis_line(fpLine);
        // Notch = raw - 1D bandpass (the CCR pixel edge read).  Recomputed
        // here from the canonical bandpass so the dump shows the SAME signal
        // splitIQlocked differences at +/-2.  IRE units.  The +/-2 difference
        // uses CCR's clamp (max/min), not the bandpass reflection, matching
        // the edge read exactly.
        const double  *bpLine  = locked1DRawBandpass_line(fpLine);
        const quint16 *rawLine =
            rawbuffer.data() + static_cast<size_t>(fpLine) * fullWidth;
        auto notchIRE = [&](int rel) -> double {
            const int r = std::clamp(rel, 0, width - 1);
            return ((double)rawLine[left + r] - bpLine[r]) * invIreScale;
        };
        if (rec && bpLine) {
            const int c1 = (fpC1 >= 0 ? fpC1 : fpC0);
            for (int rel = fpC0; rel <= c1 && rel < width; ++rel) {
                const auto &r = rec[rel];
                const char *sc = r.scheduleConformance ==
                        lddecode::CarrierScheduleConformance::LegalCarrier ? "LEG"
                    : r.scheduleConformance ==
                        lddecode::CarrierScheduleConformance::ScheduleIllegal ? "ILL"
                    : "unr";
                const int    xm     = std::max(0, rel - 2);
                const int    xp     = std::min(width - 1, rel + 2);
                const double notch  = notchIRE(rel);
                const double nedge  = std::fabs(notchIRE(xp) - notchIRE(xm));
                const double rawIRE = (double)rawLine[left + rel] * invIreScale;
                const double bpIRE  = bpLine[rel] * invIreScale;
                std::fprintf(stderr,
                    "[FP] line=%d h=%d raw=%.2f bp=%.2f src=%.2f wide=%.2f "
                    "incoh=%.2f lurch=%.2f notch=%.2f nedge=%.2f conf=%+.2f "
                    "supp=%.2f sched=%s\n",
                    fpLine, rel + left,
                    rawIRE, bpIRE,
                    r.fit.sourceSample * invIreScale,
                    r.fit.wideSample * invIreScale,
                    r.fit.sourceMinusWideIRE,
                    r.residual.maxAbsMembershipIRE,
                    notch, nedge,
                    r.carrierConformance,
                    r.conformanceSupportFraction, sc);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Corner-leak corrector.
//
// The locked bandpass bp[x] = 0.50*c - 0.25*(m2 + p2) has response sin^2(w) and
// cancels a CONSTANT luma foundation exactly (-0.25 + 0.50 - 0.25 = 0).  The
// carrier is stacked on luma, so where the foundation bends the cancellation
// fails by exactly the curvature:
//
//     leak[x] = -0.25 * (Y[x-2] - 2*Y[x] + Y[x+2])        (an identity)
//
// Consequences this stage relies on:
//   * a constant-slope ramp has ZERO curvature and therefore leaks NOTHING --
//     gradients and shading are invisible, and this stage is inert there by
//     construction rather than by a gate;
//   * only CORNERS leak, so a ramped edge is two curvature events;
//   * the alternation seen at edges arises at demod (the LUT flips sign per the
//     schedule), not in the raw leak, so one correction serves every line.
//
// Recovery.  The complementary notch is exactly
//     notch[x] = raw[x] - bp[x] = 0.25*Y[x-2] + 0.5*Y[x] + 0.25*Y[x+2],
// i.e. Y through a KNOWN 3-tap stride-2 smoother S (response cos^2(w)).  With
// m = D2{notch} we have m = S{kappa} for kappa = D2{Y}, so the curvature is
// recovered by deconvolving S.  Van Cittert (kappa += m - S{kappa}) propagates
// error as (I - S) = sin^2(w): it converges everywhere EXCEPT at fSC, where it
// is frozen at the initial guess.  Starting from kappa = 0 therefore makes NO
// claim about the one mode that is genuinely unknowable on a single line -- the
// regulariser is the physics, and there is no tuning constant in it.
//
// Gating.  Three carrier-free tests, each used for what it can actually do:
//   * lurch  -- PRESENCE of luma motion across a cycle (it cannot localise a
//               corner: full-cycle smear, one corner per cycle);
//   * schedule -- energy matching where the schedule demands inversion is luma
//               BY LAW; legal carrier is protected;
//   * parallax -- from the collected aperture-mean pool.  A legal carrier nulls
//               in EVERY legal four-sample window, so it contributes no spread
//               between the apertures covering a sample; only luma moves them.
//               The ratio spread/|carrier| is literally "what fraction of this
//               carrier-band energy fails to null in the aperture".
// All gates multiply KAPPA (a luma-domain quantity), never the carrier, so they
// cannot manufacture sidebands.
//
// DIAGNOSTIC ONLY: the published leak has no consumer yet, so the render is
// unchanged.  When it is adopted, chroma = bp - leak and Y = raw - chroma, so
// the leak returns to luma and Y + chroma == raw exactly.
// ---------------------------------------------------------------------------

// Fill the sliding four-sample aperture-mean pool for every active line.
//
// apMean[v] = mean(raw[left+v .. left+v+3]), indexed by aperture START, so the
// legal apertures covering sample x are v in {x-3, x-2, x-1, x}. Published
// unfiltered: no sharpening, no gate, no absolute value -- consumers own the
// decisions. A legal carrier sums to zero over ANY legal four-sample window, so
// each mean is that window's LUMA mean exactly; the divergence between the
// apertures covering one sample is therefore pure luma with the carrier removed
// exactly (the coarse-residual parallax), and each mean also bounds the carrier
// at every sample it covers (the feasibility hull). Running sum, O(1)/sample.
//
// Built from split1D so it exists on EVERY path (bucket and locked) without
// burst-lock rotation; phaseLocked and buildCornerLeak read the same pool.
void Comb::FrameBuffer::buildApertureMeans()
{
    if (lockedApertureMean_flat.empty()) return;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int fullWidth = videoParameters.fieldWidth;
    const int width     = right - left;
    if (left >= right || firstLine >= lastLine) return;

    const int lastStart = width - 4;
    for (int line = firstLine; line < lastLine; ++line) {
        double *apMean = lockedApertureMean_line(line);
        if (!apMean) continue;
        const quint16 *rawLine = rawbuffer.data() + size_t(line) * fullWidth;
        // dG-certified luma (user direction, 2026-07-28: "when [lurch] is
        // finding a luma edge, we have some fields where that's not in
        // doubt"). Each mean's contract is "that window's LUMA mean
        // exactly" -- true for stationary carrier, broken by the partial-
        // window leak at edges. On twin-covered lines raw - exact IS the
        // luma, so subtracting the exact carrier honours the contract
        // exactly where it used to fail: lurch edges, the coarse-residual
        // hull, and the witness coarse all read certified luma there.
        const float *exRow = exactCarrierRow(line);
        auto src = [&](int xi) -> double {
            const double r = (double)rawLine[left + xi];
            if (exRow) {
                const float e = exRow[left + xi];
                if (std::isfinite(e)) return r - (double)e;
            }
            return r;
        };
        if (width >= 4) {
            double sum4 = src(0) + src(1) + src(2) + src(3);
            for (int xi = 0; xi <= lastStart; ++xi) {
                apMean[xi] = 0.25 * sum4;
                if (xi < lastStart)
                    sum4 += src(xi + 4) - src(xi);
            }
            // Tail: no legal aperture starts here. Hold the last real mean so
            // the buffer stays readable; consumers needing "a real aperture
            // started here" must respect xi <= width-4.
            for (int xi = lastStart + 1; xi < width; ++xi)
                apMean[xi] = apMean[lastStart];
        } else {
            double avg = 0.0;
            for (int xi = 0; xi < width; ++xi)
                avg += (double)rawLine[left + xi];
            avg /= std::max(1, width);
            for (int xi = 0; xi < width; ++xi) apMean[xi] = avg;
        }
    }
}

// Clamp a carrier row into the coarse-residual feasible range, in place.
// carrierAtLeft[x] is the carrier estimate at sample left+x, x in [0,width).
// The excess leaves the carrier; the caller decides where it lands (the bucket
// path lets luma = raw - chroma absorb it downstream, the locked path folds it
// back into the leak). RESTRICTS only -- a real carrier over real luma already
// lies inside [floor, ceiling], so only impossible carrier is moved. Rotation-
// free and O(width): the feasibility hull is a default-path client.
void Comb::FrameBuffer::applyCarrierFeasibilityHull(int line,
                                                    double *carrierAtLeft)
{
    if (!carrierAtLeft) return;
    const double *apMean = lockedApertureMean_line(line);
    if (!apMean) return;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int fullWidth = videoParameters.fieldWidth;
    const int width     = right - left;
    if (width <= 0) return;
    const quint16 *rawLine = rawbuffer.data() + size_t(line) * fullWidth;
    for (int x = 0; x < width; ++x) {
        const auto rng = lddecode::carrierFeasibleRange(
            (double)rawLine[left + x], apMean, x, width);
        carrierAtLeft[x] = std::clamp(carrierAtLeft[x], rng.floor, rng.ceiling);
    }
}

// Coarse-residual parallax for one line. For each sample, demodulate the four
// covering aperture residuals (raw - apMean) on the grammar phase, envelope-
// smooth each, then take the spread across the four over their mean magnitude.
// A legal carrier is aperture-invariant (nulls in every legal four-window), so
// the four views agree and the ratio is low; moving luma shifts them apart and
// the ratio is high. Because it reads four-sample windows, it resolves compact
// colour that a wide coherent window would average away.
void Comb::FrameBuffer::apertureParallaxLine(
        int line,
        std::vector<double> &vI, std::vector<double> &vQ,
        std::vector<double> &sI, std::vector<double> &sQ,
        double *ratioOut) const
{
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int fullWidth = videoParameters.fieldWidth;
    const int width     = right - left;
    if (width <= 0 || !ratioOut) return;
    for (int x = 0; x < width; ++x) ratioOut[x] = 1.0;   // unknown => act
    const double *apMean = lockedApertureMean_line(line);
    if (!apMean) return;
    const quint16 *rawLine = rawbuffer.data() + size_t(line) * fullWidth;
    const int lastStart = width - 4;
    for (int k = 0; k < 4; ++k) {
        for (int x = 0; x < width; ++x) {
            const int v = std::clamp(x - (3 - k), 0, std::max(0, lastStart));
            const double r = (double)rawLine[left + x] - apMean[v];
            const int ph = carrierSampleClass(line, left + x);
            vI[size_t(k) * width + x] = 2.0 * r * sin4fsc(ph);
            vQ[size_t(k) * width + x] = 2.0 * r * cos4fsc(ph);
        }
    }
    // 3-sample smooth per view (envelope extraction after demod).
    for (int k = 0; k < 4; ++k) {
        const double *pi = &vI[size_t(k) * width];
        const double *pq = &vQ[size_t(k) * width];
        double *oi = &sI[size_t(k) * width];
        double *oq = &sQ[size_t(k) * width];
        for (int x = 0; x < width; ++x) {
            const int a = std::max(0, x - 1), b = std::min(width - 1, x + 1);
            oi[x] = (pi[a] + pi[x] + pi[b]) / 3.0;
            oq[x] = (pq[a] + pq[x] + pq[b]) / 3.0;
        }
    }
    for (int x = 0; x < width; ++x) {
        double mi = 0.0, mq = 0.0;
        for (int k = 0; k < 4; ++k) {
            mi += sI[size_t(k) * width + x];
            mq += sQ[size_t(k) * width + x];
        }
        mi *= 0.25; mq *= 0.25;
        double div = 0.0;
        for (int k = 0; k < 4; ++k) {
            const double di = sI[size_t(k) * width + x] - mi;
            const double dq = sQ[size_t(k) * width + x] - mq;
            div += std::hypot(di, dq);
        }
        div *= 0.25;
        const double mag = std::hypot(mi, mq);
        ratioOut[x] = div / std::max(mag, 1e-6);
    }
}

// RECOVERY PROFILE depth n, not a convergence tolerance. The Van Cittert
// iteration kappa += (m - S{kappa}) propagates error as (I - S)^n = sin^2n(w),
// so after n rounds the recovered fraction at frequency w is 1 - sin^2n(w):
// zero AT fSC always (the mode is never claimed -- the fSC null is a
// single-line law), approaching 1 elsewhere. n therefore chooses HOW MUCH of
// the near-fSC neighbourhood is claimed, and the worst-case noise gain of the
// implied inverse is bounded by n itself (G_n <= n) -- there is no tuning
// constant hiding here, only a claimed-bandwidth/noise trade. Measured: the
// strong-edge saturation DIPS then RECOVERS as n rises (a PARTIAL doublet
// subtraction leaves a residue that cancels chroma; a converged one restores
// it).
static constexpr int    kCornerRecoveryDepth  = 60;
// Outer rounds of the two-way contamination fix (see buildCornerLeak).
static constexpr int    kCornerOuterRounds    = 2;
// Parallax ratio: below soft the energy nulls in every aperture like legal
// carrier (protect); above hard it fails to null (luma, act). Measured
// populations: colour p50 0.05-0.12, pure luma p50 0.89.
static constexpr double kCornerParallaxSoft   = 0.15;
static constexpr double kCornerParallaxHard   = 0.45;

void Comb::FrameBuffer::buildCornerLeak()
{
    // No consumer yet, so the default path must not pay for it: ~11% of a
    // locked decode (the 60 Van Cittert sweeps dominate). Enable explicitly
    // with LDCD_CORNER_LEAK=1 while developing. Remove this gate when the
    // corrected source is adopted -- a stage with a real client is not
    // optional, and "on by default when it earns its place" is the rule.
    static const bool enabled = []{
        const char *s = std::getenv("LDCD_CORNER_LEAK");
        return s && std::atoi(s) != 0;
    }();
    if (!enabled)                         return;
    if (!configuration.phaseCompensation) return;
    if (lockedCornerLeak_flat.empty())    return;

    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;
    const int fullWidth = videoParameters.fieldWidth;
    if (width < 8 || first >= last) return;

    auto ramp = [](double v, double a, double b) {
        return std::clamp((v - a) / (b - a), 0.0, 1.0);
    };

    std::vector<double> notch(width), notchAdj(width),
                        mObs(width), kappa(width), sKappa(width);
    std::vector<double> ratio(width), gate(width), gateSmooth(width);
    std::vector<double> envI(width), envQ(width), sEnvI(width), sEnvQ(width),
                        envExcess(width);
    // Four aperture views of the residual, demodulated.
    std::vector<double> vI(size_t(4) * width), vQ(size_t(4) * width);
    std::vector<double> sI(size_t(4) * width), sQ(size_t(4) * width);

    // S: the notch kernel, [0.25, 0.5, 0.25] at stride 2.
    auto applyS = [&](const std::vector<double> &in, std::vector<double> &out) {
        for (int i = 0; i < width; ++i) {
            if (i >= 2 && i < width - 2)
                out[i] = 0.25 * in[i - 2] + 0.5 * in[i] + 0.25 * in[i + 2];
            else
                out[i] = in[i];
        }
    };

    for (int line = first; line < last; ++line) {
        double *leakRow = lockedCornerLeak_line(line);
        if (!leakRow) continue;
        // Clear unconditionally: a stale leak from an earlier frame must never
        // survive into this one (FrameBuffers persist across batches).
        std::fill(leakRow, leakRow + width, 0.0);
        const double *bpLine = locked1DRawBandpass_line(line);
        if (!bpLine) continue;

        const quint16 *rawLine =
            rawbuffer.data() + static_cast<size_t>(line) * fullWidth;
        const lddecode::CarrierAnalysisRecord *analysisRow =
            carrierAnalysis_line(line);
        const double *apMean = lockedApertureMean_line(line);

        for (int x = 0; x < width; ++x)
            notch[x] = (double)rawLine[left + x] - bpLine[x];

        // ---- Parallax: does this carrier-band energy null in every aperture?
        // Shared with the cross-color detector's carrier-legality veto.
        apertureParallaxLine(line, vI, vQ, sI, sQ, ratio.data());

        // ---- Gate: parallax alone. Multiplies KAPPA (a luma-domain quantity),
        // never the carrier, so it cannot manufacture sidebands.
        //
        // A lurch-presence gate and a schedule gate were both measured here and
        // both REMOVED, on evidence:
        //   * lurch changed nothing once parallax was present (identical to
        //     0.2% on every metric) -- it asks a cruder version of the same
        //     question;
        //   * the schedule THROTTLED the correction 2-4x (chevron beading
        //     -18.3% -> -7.5%, luma return +59.9% -> +15.5%) while helping
        //     neither the monochrome case nor compact colour. It is the wrong
        //     instrument for this job: the schedule reports that legal carrier
        //     is PRESENT, but the corner leak rides ON TOP of legal carrier at
        //     an edge, so it protected whole pixels. Parallax reports what
        //     FRACTION of the energy fails to null in the aperture, which is
        //     the proportional question removing a component actually poses.
        // Measured with parallax alone: compact colour (earring) +0.2%,
        // monochrome false colour -20.7% frame-wide, luma return +26.6%.
        //
        // THE GATE MUST VARY AT ENVELOPE SCALE. Multiplication in space is
        // convolution in frequency: a per-pixel gain smears kappa's spectrum
        // into fSC and MANUFACTURES carrier-rate content, which then lands in
        // luma via Y = notch + leak and reads as a garish alternating edge.
        // Measured with a raw per-pixel gate: ~28% of the injected luma change
        // sat at fSC. Van Cittert cannot produce that itself (it stalls at fSC
        // by construction), so the gate was the only possible source. Smooth
        // the gate with the encoder's own chroma-envelope kernel before
        // applying it -- a weight gathered AND applied at envelope scale cannot
        // manufacture out-of-band sidebands.
        // The schedule gate is RESTORED. It was removed earlier on chroma-side
        // metrics (beading/saturation), which the Y-alternation finding then
        // invalidated: with parallax alone the corrector injects +28.5% luma
        // alternation at thin straps versus +10.5% with the schedule in place,
        // and the straps read garishly worse. Chroma metrics could not see it
        // because strap SATURATION is flat in every configuration -- the damage
        // is entirely on the luma side.
        // Schedule side of the gate: the sanctioned table-owned luma proof,
        // not a local ramp. carrierIllegalProof() is ZERO through the entire
        // ambiguous middle by design ("real chroma is never claimed as luma"),
        // which is exactly what a leak that RETURNS energy to Y needs -- the
        // previous local form set unsupported evidence to a half-gate
        // (+ (1-supp)*0.5), desaturating genuine chroma wherever the axes could
        // not decide. Contradiction is consumed distinctly from absence, the
        // same fail-closed rule the schedule licenses use: one axis that
        // decisively votes legal carrier revokes the luma claim outright,
        // whereas an absent/abstaining axis merely fails to support it.
        for (int x = 0; x < width; ++x) {
            double g = ramp(ratio[x], kCornerParallaxSoft, kCornerParallaxHard);
            if (analysisRow) {
                const double contra = std::clamp(
                    (double)analysisRow[x].conformanceContradictionFraction,
                    0.0, 1.0);
                if (contra > 0.0)
                    g = 0.0;                 // observed legal-carrier vote: protect
                else
                    g *= lddecode::carrierIllegalProof(
                        (double)analysisRow[x].carrierConformance,
                        (double)analysisRow[x].conformanceSupportFraction);
            }
            gate[x] = g;
        }
        for (int x = 0; x < width; ++x) {
            double acc = 0.0, wsum = 0.0;
            for (int t = 0; t < lddecode::kChromaEnvelopeTaps; ++t) {
                const int o = x + t - lddecode::kChromaEnvelopeTaps / 2;
                if (o < 0 || o >= width) continue;
                const double w = lddecode::kChromaEnvelopeFilter[t];
                acc += w * gate[o]; wsum += w;
            }
            gateSmooth[x] = (wsum > 0.0) ? acc / wsum : gate[x];
        }

        // ---- One lawful estimate of the carrier-envelope curvature D2A. -----
        //
        // The bandpass assumes a constant carrier ENVELOPE as well as a constant
        // luma foundation. With c[x+-2] = -A[x+-2]*basis,
        //   B{C}[x] = basis*(0.5*A[x] + 0.25*A[x-2] + 0.25*A[x+2])
        //           = C[x] + 0.25*basis*D2{A}[x]
        // so the full error the bandpass carries is TWO curvature terms:
        //   bp = C + 0.25*basis*D2A - 0.25*D2Y
        //
        // D2A appears in BOTH corrections this loop performs, and they MUST be
        // one estimate or their difference lands in Y:
        //   * the envelope over-estimate withdrawn from the leak (+envExcess);
        //   * the chroma's own image in the notch band. N = I - B has response
        //     cos^2(w) (zero only AT fSC), and N{C} = -0.25*basis*D2A, so a
        //     compact colour feature -- fast envelope, content away from band
        //     centre -- injects curvature into D2{notch} that the luma solve
        //     would otherwise read as false leak. Removing it is notch +=
        //     0.25*basis*D2A = notch + envExcess.
        // Previously these were TWO different D2A estimates: the notch used a
        // BROADBAND N{chromaEst} off the raw (luma- and noise-contaminated)
        // chroma estimate, the leak used a lawfully projected one off bp. They
        // disagreed, and the disagreement was carrier-rate alternation in Y.
        //
        // The single estimate: demodulate the current carrier estimate, project
        // onto the encoder's own chroma-envelope band (the sanctioned P, applied
        // EXACTLY ONCE to a quantity that is never a prior P output, so it
        // cannot compound), stride-2 second difference, remodulate.
        //   envExcess[x] = 0.25 * basis * D2A[x]   (composite domain)
        //
        // Recurrence: the carrier estimate is bp cleaned of the LUMA leak
        // (bp + 0.25*gate*kappa) -- NOT of the envelope term itself, since
        // feeding the projected envelope back into its own input is exactly
        // what would compound the FIR. Round 0 has kappa = 0, so the first
        // Ahat = P{demod bp}; later rounds sharpen it with the luma solve.
        // Because envExcess no longer needs kappa, the notch correction now
        // runs from round 0 -- the envelope contamination was always present,
        // the old chicken-and-egg (chromaEst needs kappa) was the only reason
        // it waited. A wrong leak shows up as carrier-rate ALTERNATION in Y
        // (Y = notch + leak; a correct leak reconstructs the sharp luma with
        // none), so that alternation is the metric to watch.
        std::fill(kappa.begin(), kappa.end(), 0.0);
        std::fill(envExcess.begin(), envExcess.end(), 0.0);
        for (int outer = 0; outer < kCornerOuterRounds; ++outer) {
            // Carrier estimate: bp with the current (gated) luma leak removed.
            for (int x = 0; x < width; ++x) {
                const int ph = carrierSampleClass(line, left + x);
                const double cEst = bpLine[x] + 0.25 * gateSmooth[x] * kappa[x];
                envI[x] = 2.0 * cEst * sin4fsc(ph);
                envQ[x] = 2.0 * cEst * cos4fsc(ph);
            }
            // Lawful envelope: the sanctioned projection, applied once.
            lddecode::projectExpressibleChromaEnvelope(envI.data(), nullptr,
                                                       width, sEnvI.data());
            lddecode::projectExpressibleChromaEnvelope(envQ.data(), nullptr,
                                                       width, sEnvQ.data());
            // The single D2A term, in composite: envExcess = 0.25*basis*D2A.
            for (int x = 0; x < width; ++x) {
                const int m2 = std::clamp(x - 2, 0, width - 1);
                const int p2 = std::clamp(x + 2, 0, width - 1);
                const int ph = carrierSampleClass(line, left + x);
                const double d2I = sEnvI[m2] - 2.0 * sEnvI[x] + sEnvI[p2];
                const double d2Q = sEnvQ[m2] - 2.0 * sEnvQ[x] + sEnvQ[p2];
                // 0.5*(...) undoes the 2x demod gain.
                envExcess[x] = 0.25 * 0.5 * (d2I * sin4fsc(ph) + d2Q * cos4fsc(ph));
            }
            // Notch corrected for the chroma that leaks INTO it, using the SAME
            // envExcess: N{chroma} = -envExcess in composite, so notch += it.
            for (int x = 0; x < width; ++x)
                notchAdj[x] = notch[x] + envExcess[x];
            std::fill(mObs.begin(), mObs.end(), 0.0);
            for (int x = 2; x < width - 2; ++x)
                mObs[x] = notchAdj[x - 2] - 2.0 * notchAdj[x] + notchAdj[x + 2];

            // Van Cittert deconvolution of S. Error propagates as
            // (I - S) = sin^2, so this stalls at fSC by construction and never
            // claims that mode.
            std::fill(kappa.begin(), kappa.end(), 0.0);
            for (int it = 0; it < kCornerRecoveryDepth; ++it) {
                applyS(kappa, sKappa);
                for (int x = 0; x < width; ++x)
                    kappa[x] += (mObs[x] - sKappa[x]);
            }
        }
        for (int x = 0; x < width; ++x)
            kappa[x] *= gateSmooth[x];

        // Total withdrawal: envelope over-estimate MINUS the luma leak.
        //   chroma = bp - leakRow,  Y = raw - chroma.
        // kappa already estimates the stride-2 second difference D2Y, and the
        // luma leak the bandpass added is exactly -0.25*D2Y, so the luma term
        // is -0.25*kappa[x] with no further smoothing. An earlier stride-1
        // [0.25,0.5,0.25] smoother on kappa here was a SECOND regulariser on
        // the fSC neighbourhood (its response is exactly 0.5 at fSC), redundant
        // with the Van Cittert recovery profile that already owns that band;
        // it silently halved every fSC-rate leak the solve had recovered.
        for (int x = 0; x < width; ++x)
            leakRow[x] = -0.25 * kappa[x] + envExcess[x];

        // ---- Coarse-residual carrier hull (P5). -----------------------------
        //
        // The corner leak estimates carrier from a single line and, at a
        // compact feature, the encoder bandwidth law forbids the lawful
        // envelope from modelling the fast on/off -- so P2 under-corrects and
        // leaves carrier-rate content that is really luma. The coarse residuals
        // bound the carrier INDEPENDENTLY, from the luma side where compact
        // content is lawful (luma is never bandlimited): carrier must lie in
        // [raw - max_v apMean, raw - min_v apMean] over the covering apertures.
        // Clamp the emitted carrier (bp - leak) into that range and fold the
        // excess back into the leak (so it lands in luma via Y = raw - chroma);
        // both bounds are applied so an oscillating carrier is not rectified.
        // The clamp itself is the shared applyCarrierFeasibilityHull(), which
        // the bucket path also calls -- here it acts on the emitted carrier and
        // the difference is backed out into leakRow. v1 is unguarded: the dark-
        // side ceiling is clean but the pixel luma floor can be violated at a
        // lone dark sample, which would clip legal carrier -- the decomposed
        // metric (satRet/hueRot) is what reveals it.
        static const bool hullEnabled = []{
            const char *s = std::getenv("LDCD_CORNER_HULL");
            return !s || std::atoi(s) != 0;         // default ON when leak runs
        }();
        if (hullEnabled && apMean) {
            for (int x = 0; x < width; ++x)
                notchAdj[x] = bpLine[x] - leakRow[x];   // emitted carrier
            applyCarrierFeasibilityHull(line, notchAdj.data());
            for (int x = 0; x < width; ++x)
                leakRow[x] = bpLine[x] - notchAdj[x];   // excess -> leak -> luma
        }
    }

    // ---- Disposable metrics (env-gated), without changing any output. -------
    //
    // The old scalar was a stride-1 second difference of bp. For a pure fSC
    // carrier |D2_1{C}| = 2|C| and |D2_2{C}| = 4|C|: BOTH are proportional to
    // carrier amplitude pointwise, so a second-difference scalar cannot tell a
    // beading reduction from a desaturation. Retired. The correction's effect
    // decomposes EXACTLY into two disjoint channels:
    //     Y  += leak                       (luma gains the withdrawn energy)
    //     rendered chroma envelope += -P{demod(leak)}
    // so the honest instruments are (a) the fSC-rate content the leak injects
    // into luma -- zero for any pure legal carrier, hence non-tautological and
    // PRIMARY -- and (b) what the withdrawal does to the rendered envelope,
    // split into magnitude (legal-carrier retention, a cost) versus hue
    // rotation (a cross-colour signature, a defect). Conservation is demoted to
    // an assertion: Y + chroma == raw holds for ANY leak by construction, so it
    // measures arithmetic, not the model.
    static const int clDump = []{
        const char *s = std::getenv("LDCD_DUMP_CORNER"); return s ? std::atoi(s) : 0;
    }();
    if (clDump) {
        std::vector<double> lI(width), lQ(width), lIs(width), lQs(width);
        std::vector<double> c0I(width), c0Q(width), c0Is(width), c0Qs(width);
        std::vector<double> c1I(width), c1Q(width), c1Is(width), c1Qs(width);
        double yAltAll = 0.0, yAltHot = 0.0, maxLeak = 0.0;
        double satNum = 0.0, satDen = 0.0, hueW = 0.0, hueAcc = 0.0;
        long nAll = 0, nHot = 0;
        for (int line = first; line < last; ++line) {
            const double *bpLine = locked1DRawBandpass_line(line);
            const double *leakRow = lockedCornerLeak_line(line);
            if (!bpLine || !leakRow) continue;
            for (int x = 0; x < width; ++x) {
                const int ph = carrierSampleClass(line, left + x);
                const double s = sin4fsc(ph), c = cos4fsc(ph);
                lI[x]  = 2.0 * leakRow[x] * s;         lQ[x]  = 2.0 * leakRow[x] * c;
                c0I[x] = 2.0 * bpLine[x] * s;          c0Q[x] = 2.0 * bpLine[x] * c;
                const double cc = bpLine[x] - leakRow[x];
                c1I[x] = 2.0 * cc * s;                 c1Q[x] = 2.0 * cc * c;
            }
            // Lowpass each demodulated stream to the lawful envelope band; this
            // strips the 2fSC demod image so the fSC-rate part is what remains.
            lddecode::projectExpressibleChromaEnvelope(lI.data(),  nullptr, width, lIs.data());
            lddecode::projectExpressibleChromaEnvelope(lQ.data(),  nullptr, width, lQs.data());
            lddecode::projectExpressibleChromaEnvelope(c0I.data(), nullptr, width, c0Is.data());
            lddecode::projectExpressibleChromaEnvelope(c0Q.data(), nullptr, width, c0Qs.data());
            lddecode::projectExpressibleChromaEnvelope(c1I.data(), nullptr, width, c1Is.data());
            lddecode::projectExpressibleChromaEnvelope(c1Q.data(), nullptr, width, c1Qs.data());
            for (int x = 6; x < width - 6; ++x) {
                maxLeak = std::max(maxLeak, std::fabs(leakRow[x]) * invIreScale);
                // (a) fSC-rate envelope of the injected luma change = alternation.
                const double yAlt = std::hypot(lIs[x], lQs[x]) * invIreScale;
                yAltAll += yAlt; ++nAll;
                const double a0 = std::hypot(c0Is[x], c0Qs[x]);
                const double a1 = std::hypot(c1Is[x], c1Qs[x]);
                if (a0 * invIreScale > 6.0) {          // real carrier present
                    yAltHot += yAlt; ++nHot;
                    satNum += a1 * invIreScale; satDen += a0 * invIreScale;
                    // (b) hue rotation, amplitude-weighted, corrected vs raw.
                    double dth = std::atan2(c1Qs[x], c1Is[x])
                               - std::atan2(c0Qs[x], c0Is[x]);
                    while (dth >  M_PI) dth -= 2.0 * M_PI;
                    while (dth < -M_PI) dth += 2.0 * M_PI;
                    hueAcc += a0 * std::fabs(dth); hueW += a0;
                }
            }
        }
        if (nAll > 0)
            std::fprintf(stderr,
                "[CORNER] Yalt %.3f IRE (hot %.3f)  satRet %+.1f%%  hueRot %.1f deg"
                "  maxLeak %.1f IRE  nHot=%ld\n",
                yAltAll / nAll, nHot ? yAltHot / nHot : 0.0,
                100.0 * (satNum - satDen) / std::max(satDen, 1e-9),
                nHot ? (hueAcc / std::max(hueW, 1e-9)) * 180.0 / M_PI : 0.0,
                maxLeak, nHot);
    }
}

// Shared with the retraction ladder (defined near the retracted-view code):
// working-space phase snap -- est keeps amplitude, ref supplies phase.
// Optional per-sample gate [0,1] scales the snap's authority (twin
// agreement for temporal references); nullptr = full authority.
static void ldcdApplyPhaseSnap(const std::vector<double> &est,
                               const std::vector<double> &ref,
                               std::vector<double> &out,
                               int width, double irescale,
                               double ampMinIRE, double ampTauIRE,
                               bool clampRatio,
                               const double *gate = nullptr);

// Windowed vector-coherence ALPHA between two sign-aligned scalar
// carriers in the snap basis: 0 where either window is incomplete, ramped
// 0.55 -> 0.85 on the resultant coherence. A RAMP, never a cut -- binary
// per-sample gates interleave two differently-phased renders at pixel
// pitch along edges, which renders as checkerboard (the OOB-cut lesson,
// re-learned on the beach 2026-07-30).
static void ldcdSideCoherenceAlpha(const std::vector<double> &a,
                                   const std::vector<double> &b,
                                   int width, std::vector<double> &alpha)
{
    constexpr int kSnapHalf = 4;
    static const int cB[4] = { 1, 0, -1, 0 };
    static const int sB[4] = { 0, 1, 0, -1 };
    std::fill(alpha.begin(), alpha.end(), 0.0);
    // Centered 9-tap, half-weight ends (the 6f502ec integer-centroid
    // convention): the old [-3,+4] window carried a +0.5 centroid that
    // biased WHERE the alpha opens by half a sample at envelope edges.
    for (int xi = kSnapHalf; xi < width - kSnapHalf; ++xi) {
        double aI = 0, aQ = 0, bI = 0, bQ = 0;
        bool ok = true;
        for (int k = xi - kSnapHalf; k <= xi + kSnapHalf && ok; ++k) {
            if (!std::isfinite(a[k]) || !std::isfinite(b[k])) {
                ok = false; break;
            }
            const double w =
                (k == xi - kSnapHalf || k == xi + kSnapHalf) ? 0.5 : 1.0;
            const int ph = k & 3;
            aI += w * a[k] * cB[ph]; aQ += w * a[k] * sB[ph];
            bI += w * b[k] * cB[ph]; bQ += w * b[k] * sB[ph];
        }
        if (!ok) continue;
        const double ma = std::hypot(aI, aQ);
        const double mb = std::hypot(bI, bQ);
        const double mr = std::hypot(aI + bI, aQ + bQ);
        const double coh = mr / (ma + mb + 1e-12);
        alpha[xi] = std::clamp((coh - 0.55) / 0.30, 0.0, 1.0);
    }
}

// Grammar-sign-aligned certified bracket mean: the +/-1 certified rows
// brought into the center line's carrier-phase space. NaN where either
// bracket sample is absent or the grammar cannot certify the relation.
// ONE implementation -- the head snap and the retraction ladder both call
// here.
static void ldcdBuildCertBracketAligned(
    const lddecode::CarrierGrammarState *gC,
    const lddecode::CarrierGrammarState *gU,
    const lddecode::CarrierGrammarState *gD,
    const float *exU, const float *exD,
    int left, int width, std::vector<double> &bAlign,
    std::vector<double> *legUOut = nullptr,
    std::vector<double> *legDOut = nullptr)
{
    std::fill(bAlign.begin(), bAlign.end(),
              std::numeric_limits<double>::quiet_NaN());
    if (legUOut) std::fill(legUOut->begin(), legUOut->end(),
                           std::numeric_limits<double>::quiet_NaN());
    if (legDOut) std::fill(legDOut->begin(), legDOut->end(),
                           std::numeric_limits<double>::quiet_NaN());
    if (!gC || !gU || !gD || !exU || !exD) return;
    for (int xi = 0; xi < width; ++xi) {
        const int h = left + xi;
        const float eu = exU[h];
        const float ed = exD[h];
        if (!std::isfinite(eu) || !std::isfinite(ed)) continue;
        const auto rU = lddecode::carrierGrammarSignedPhaseRelation(gC, h, gU, h);
        const auto rD = lddecode::carrierGrammarSignedPhaseRelation(gC, h, gD, h);
        double sU, sD;
        if (rU == lddecode::CarrierPhaseRelation::Opposite) sU = -1.0;
        else if (rU == lddecode::CarrierPhaseRelation::Same) sU = 1.0;
        else continue;
        if (rD == lddecode::CarrierPhaseRelation::Opposite) sD = -1.0;
        else if (rD == lddecode::CarrierPhaseRelation::Same) sD = 1.0;
        else continue;
        bAlign[xi] = 0.5 * (sU * (double)eu + sD * (double)ed);
        if (legUOut) (*legUOut)[xi] = sU * (double)eu;
        if (legDOut) (*legDOut)[xi] = sD * (double)ed;
    }
}

void Comb::FrameBuffer::buildPhaseCorrected1D()
{
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;
    const int fullWidth = videoParameters.fieldWidth;

    // Phase-1 cross-color diagnostic gate (parsed once).  See the measurement
    // block in Pass 2.  A per-framebuffer heartbeat is emitted before the early
    // return so a silent dump is self-explaining: it reports whether the locked
    // path is even active and whether the requested line is in the active range.
    static const int ccDiagLine = []{ const char *s = std::getenv("CC_DIAG_LINE"); return s ? std::atoi(s) : -1; }();
    static const int ccDiagC0   = []{ const char *s = std::getenv("CC_DIAG_C0");   return s ? std::atoi(s) : -1; }();
    static const int ccDiagC1   = []{ const char *s = std::getenv("CC_DIAG_C1");   return s ? std::atoi(s) : -1; }();
    static const int crDiagLine = []{ const char *s = std::getenv("COARSE_RESID_DIAG_LINE"); return s ? std::atoi(s) : -1; }();
    static const int crDiagC0   = []{ const char *s = std::getenv("COARSE_RESID_DIAG_C0");   return s ? std::atoi(s) : -1; }();
    static const int crDiagC1   = []{ const char *s = std::getenv("COARSE_RESID_DIAG_C1");   return s ? std::atoi(s) : -1; }();
    static const double crDiagFitTolIRE = []{
        const char *s = std::getenv("COARSE_RESID_DIAG_FIT_TOL_IRE");
        return s ? std::atof(s) : 1.5;
    }();
    static const double parallaxRepairTolIRE = []{
        const char *s = std::getenv("LD_1D_PARALLAX_TOL_IRE");
        return s ? std::atof(s) : 0.5;
    }();
    static const double parallaxRepairMaxDeltaIRE = []{
        const char *s = std::getenv("LD_1D_PARALLAX_MAX_DELTA_IRE");
        return s ? std::atof(s) : 0.35;
    }();
    // Pass-1.5 mode. The repair is an EXPERIMENT and is opt-in, as the
    // buffer-flow doc has always described it:
    //   unset   -- analysis only, the source is untouched (default)
    //   report  -- analyse and log, still do not touch the source
    //   apply   -- commit the bounded move
    // It had lost its gate and was running unconditionally on every locked
    // frame, worth ~0.9 IRE rms of luma difference from bucket frame-wide.
    // Restoring the gate is a doc/code reconciliation, not a new policy.
    static const bool parallaxRepairApply = []{
        const char *s = std::getenv("LD_1D_PARALLAX_REPAIR");
        return s && std::strcmp(s, "apply") == 0;
    }();
    if (ccDiagLine >= 0) {
        std::fprintf(stderr,
            "CCDIAG-FB phaseComp=%d activeLines=[%d,%d) width=%d target=%d %s\n",
            configuration.phaseCompensation ? 1 : 0, first, last, width, ccDiagLine,
            (ccDiagLine >= first && ccDiagLine < last) ? "in-range" : "OUT-OF-RANGE");
    }
    if (crDiagLine >= 0) {
        std::fprintf(stderr,
            "COARSERES-FB phaseComp=%d activeLines=[%d,%d) width=%d target=%d %s\n",
            configuration.phaseCompensation ? 1 : 0, first, last, width, crDiagLine,
            (crDiagLine >= first && crDiagLine < last) ? "in-range" : "OUT-OF-RANGE");
    }

    if (!configuration.phaseCompensation || width <= 0 || first >= last)
        return;

    const size_t magnitudeCount =
        static_cast<size_t>(demodLines) * demodWidth;
    if (demodIQMag4fsc_flat.size() < magnitudeCount)
        demodIQMag4fsc_flat.resize(magnitudeCount, 0.0f);

    // scratch_preI carries the ordinary raw 1D bandpass line.
    // scratch_preQ carries the restrained source actually emitted downstream.
    if (static_cast<int>(scratch_preI.size()) < width)
        scratch_preI.resize(width, 0.0);
    if (static_cast<int>(scratch_preQ.size()) < width)
        scratch_preQ.resize(width, 0.0);

    auto clamp01 = [](double v) {
        return std::clamp(v, 0.0, 1.0);
    };

    // Wide-window cross-color detector (narrow-vs-wide carrier coherence).
    //
    // Real chroma is a coherent carrier sinusoid that holds phase over many
    // cycles; luma-near-fsc looks carrier-like over one cycle but is
    // incoherent over a wide aperture.
    //
    // NARROW (tighter) estimate: a rolling, current-centered mean of the
    // 2-sample carrier-fit envelope hypot(bp[x], bp[x+1]).  The point envelope
    // equals the carrier amplitude A only at perfect fsc and ripples at 2fsc
    // under real phase error; a 4-cycle centered mean nulls that 2fsc/4fsc
    // ripple, so the narrow term reads a clean A on coherent carrier.  This
    // replaces the smooth carrierFit envelope the witness rollback removed.
    //
    // WIDE estimate: an 8-cycle coherent demod sum.  On coherent carrier both
    // read A, so risk is a true ripple-free 0; on luma-near-fsc the coherent
    // wide fit cancels while the narrow mean-of-envelope does not, so risk > 0.
    //
    // The metric is published as carrierImpurity (a provisional oracle).
    // It is NEVER applied to the carrier source; the source is emitted clean.
    // Suppression happens downstream as alpha at color demod and Y subtraction.
    // measurePostCombImpurity() later replaces this provisional 1D read with
    // the elected-comb measurement that splitIQlocked() actually consumes.
    // The doc's exact form is used with no shaping: any shape made it
    // unresponsive somewhere.
    constexpr int    kNarrowWin = 16;          // 4 carrier cycles (nulls 2fsc/4fsc)
    constexpr int    kWideWin = 32;            // 8 carrier cycles
    constexpr double kImpurityFloorIRE = 2.0;  // ignore low-amplitude noise

    // Period-4 quadrature reference.  Any fixed phase-locked basis works; the
    // coherent-sum magnitude is invariant to its rotation.
    static const double cosRef[4] = { 1.0, 0.0, -1.0, 0.0 };
    static const double sinRef[4] = { 0.0, 1.0,  0.0, -1.0 };

    // Per-line scratch, allocated once and reused across lines.
    std::vector<double> demI(width), demQ(width);
    std::vector<double> preI(width + 1), preQ(width + 1);
    std::vector<double> env(width), preEnv(width + 1);
    std::vector<double> nativeI4(width), nativeQ4(width);
    std::vector<double> snapEst(width), snapOut(width), snapRef(width);

    for (int line = first; line < last; ++line) {
        const quint16 *rawLine =
            rawbuffer.data() + static_cast<size_t>(line) * fullWidth;

        double *lockedSource = locked1DSource_line(line);
        if (!lockedSource)
            continue;

        AttributionEvidence *attribution = attributionEvidence_line(line);
        lddecode::CarrierAnalysisRecord *carrierAnalysis =
            carrierAnalysis_line(line);

        CombCarrierGrammar *grammar = carrierGrammarLine(line);
        const bool grammarLocked = grammar && grammar->grammarLocked;
        const double burstCos = grammarLocked ? grammar->burstCos : 1.0;
        const double burstSin = grammarLocked ? grammar->burstSin : 0.0;

        double lutI[4];
        double lutQ[4];
        if (grammarLocked) {
            for (int i = 0; i < 4; ++i) {
                lutI[i] = static_cast<double>(grammar->demodLUTTi[i]);
                lutQ[i] = static_cast<double>(grammar->demodLUTTq[i]);
            }
        } else {
            fusedDemodLUT(
                burstCos, burstSin, spLUT_locked, cpLUT_locked, lutI, lutQ);
        }

        double i4Scale[4];
        double q4Scale[4];
        double magnitudeScale[4];

        for (int phase = 0; phase < 4; ++phase) {
            lockedTo4fsc(
                lutI[phase], lutQ[phase], burstCos, burstSin,
                i4Scale[phase], q4Scale[phase]);

            magnitudeScale[phase] =
                boundedMag(i4Scale[phase], q4Scale[phase]);
        }

        // Grammar-convention dump (LDCD_DUMP_GRAMMAR_L0/L1): per-line
        // burst phasor, samplePhase0, lineFlip -- the direct read on the
        // 180-degree alternation the sync probe found between alternate
        // def lines in the working space.
        static const int gdL0 = []{ const char *e = std::getenv("LDCD_DUMP_GRAMMAR_L0"); return e ? std::atoi(e) : -1; }();
        static const int gdL1 = []{ const char *e = std::getenv("LDCD_DUMP_GRAMMAR_L1"); return e ? std::atoi(e) : -1; }();
        if (gdL0 >= 0 && line >= gdL0 && line <= gdL1 && grammar) {
            std::fprintf(stderr,
                "GRAM line=%d locked=%d sp0=%d flip=%+d burst=(%.3f,%.3f) "
                "burstDeg=%.1f cov=%d\n",
                line, grammarLocked ? 1 : 0, grammar->samplePhase0,
                grammar->lineFlip, burstCos, burstSin,
                std::atan2(burstSin, burstCos) * 180.0 / M_PI,
                certifiedDefLine(line) ? 1 : 0);
        }

        float *demodI = demodTI_line(line);
        float *demodQ = demodTQ_line(line);
        float *demodI4 = demodTI4fsc_line(line);
        float *demodQ4 = demodTQ4fsc_line(line);
        float *lockedI4 = locked1DTI4fsc_line(line);
        float *lockedQ4 = locked1DTQ4fsc_line(line);
        float *magnitude =
            demodIQMag4fsc_flat.data() +
            static_cast<size_t>(line) * demodWidth;

        double *rawBandpass = locked1DRawBandpass_line(line);
        double *bpLine = scratch_preI.data();
        double *restrainedLine = scratch_preQ.data();

        auto rawAtRel = [&](int rel) -> double {
            const int r = std::clamp(rel, 0, width - 1);
            return static_cast<double>(rawLine[left + r]);
        };

        // Pass 1: consume the canonical full-resolution baseline harvested by
        // buildCarrierAnalysis(). The fallback preserves standalone safety but
        // normal locked orchestration has exactly one producer.
        //
        // The corner leak is withdrawn HERE, at the point the source authority
        // enters the stage, so every later consumer (repair, restraint, demod,
        // publish, and the candidates built from the published source) sees one
        // corrected carrier rather than a patched output. Because
        //     source = bp - leak   and   Y = raw - source,
        // the withdrawn leak lands in luma automatically: Y + chroma == raw
        // stays exact, which is the conservation a desaturating suppressor
        // cannot satisfy. No output-side correction pass, so the produceY
        // boundary is untouched.
        //
        // lockedCornerLeak is zero-filled unless buildCornerLeak() ran, so this
        // subtraction is inert by construction when the stage is disabled.
        const double *cornerLeak = lockedCornerLeak_line(line);
        if (rawBandpass) {
            if (cornerLeak) {
                for (int rel = 0; rel < width; ++rel)
                    bpLine[rel] = rawBandpass[rel] - cornerLeak[rel];
            } else {
                for (int rel = 0; rel < width; ++rel)
                    bpLine[rel] = rawBandpass[rel];
            }
        } else {
            for (int rel = 0; rel < width; ++rel) {
                const double c  = rawAtRel(rel);
                const double m2 = rawAtRel(rel - 2);
                const double p2 = rawAtRel(rel + 2);
                bpLine[rel] = 0.50 * c - 0.25 * (m2 + p2);
            }
        }

        // Pair class-map probe (measurement only; inert unless
        // LDCD_PROBE_DISENT). The 1D output is never touched from other
        // lines -- 1D is downstream's safe retreat.
        probeEdgePairClassMap(line);

        // Pass 1.5: coarse-residual feasibility repair for locked 1D.
        //
        // The ordinary 1D bandpass remains the source authority; every scanner
        // below only gets to justify a small bounded move toward a short-fit-
        // compatible subset of legal residual options. Coarse residuals are
        // options, not carrier estimates. The short fit is a selector, not
        // carrier authority. The moving-centered residual is support/conflict
        // evidence only. The wide fit is comparison-only and never participates
        // in survivor selection.
        if (carrierAnalysis)
        {
            const bool repairLogThisLine =
                crDiagLine >= 0 && line == crDiagLine && crDiagC0 >= 0;
            const int repairLogFirst =
                repairLogThisLine ? std::clamp(crDiagC0, 0, width - 1) : 0;
            const int repairLogLast =
                repairLogThisLine
                    ? std::clamp(crDiagC1 < 0 ? crDiagC0 : crDiagC1,
                                 repairLogFirst, width - 1)
                    : -1;

            const double maxDeltaSamples =
                std::max(0.0, parallaxRepairMaxDeltaIRE) * irescale;
            float *repairStrengthRow =
                locked1DParallaxRepairStrength_line(line);
            if (repairStrengthRow)
                std::fill(repairStrengthRow, repairStrengthRow + width, 0.0f);
            float *repairDeltaRow =
                locked1DParallaxRepairDelta_line(line);
            if (repairDeltaRow)
                std::fill(repairDeltaRow, repairDeltaRow + width, 0.0f);

            if (repairLogThisLine) {
                std::fprintf(stderr,
                    "COARSERESREPAIR header line rel h phase mode reason "
                    "sourceBp shortFit wideFit movingResidual optionCount "
                    "survivorCount survivorLo survivorHi tolIRE maxDeltaIRE "
                    "proposedDeltaIRE appliedDeltaIRE movingDistIRE "
                    "maxAbsMembershipIRE sourceMinusShortIRE shortMinusWideIRE "
                    "sourceMinusWideIRE\n");
            }

            for (int rel = 0; rel < width; ++rel) {
                const int p = carrierSampleClass(line, left + rel) & 3;
                const auto &record = carrierAnalysis[rel];
                const auto &residualDiag = record.residual;
                const int optionCount = residualDiag.optionCount;
                const double shortSample = record.fit.shortSample;
                const double wideSample = record.fit.wideSample;
                const double movingResidual = residualDiag.movingResidualSample;
                const double sourceSample = bpLine[rel];
                const int survivorCount = residualDiag.survivorCount();
                const double survivorLo = residualDiag.survivorLo;
                const double survivorHi = residualDiag.survivorHi;
                const double maxAbsMembershipIRE =
                    residualDiag.maxAbsMembershipIRE;

                const char *reason = "no-options";
                double proposedDelta = 0.0;
                double appliedDelta = 0.0;
                double movingDistIRE = 0.0;

                if (carrierAnalysis[rel].scheduleConformance ==
                    lddecode::CarrierScheduleConformance::ScheduleIllegal) {
                    // Registered as luma at analysis time: there is no
                    // carrier here to repair, and the residual options are
                    // luma interpretations that would only masquerade as
                    // survivor conflict.
                    reason = "schedule-illegal-luma";
                } else if (optionCount <= 0) {
                    reason = "no-options";
                } else if (survivorCount <= 0) {
                    reason = "conflict-no-survivors";
                } else if (survivorCount == optionCount) {
                    reason = "no-discrimination-all-survive";
                } else if (sourceSample >= survivorLo &&
                           sourceSample <= survivorHi)
                {
                    reason = "source-inside";
                } else {
                    movingDistIRE = residualDiag.movingDistanceIRE;
                    const bool movingCompatible = residualDiag.movingCompatible;

                    if (!movingCompatible) {
                        reason = "moving-conflict";
                    } else {
                        const double target =
                            std::clamp(sourceSample, survivorLo, survivorHi);
                        proposedDelta = target - sourceSample;
                        appliedDelta = std::clamp(
                            proposedDelta,
                            -maxDeltaSamples,
                            maxDeltaSamples);

                        if (!parallaxRepairApply) {
                            // Analysis complete, but the repair is opt-in:
                            // the ordinary bandpass remains source authority
                            // and no repair hold is published downstream.
                            reason = "report-only";
                            appliedDelta = 0.0;
                        } else {
                            reason = "apply";
                            bpLine[rel] = sourceSample + appliedDelta;
                            if (repairStrengthRow && maxDeltaSamples > 1e-9) {
                                repairStrengthRow[rel] = static_cast<float>(
                                    std::clamp(
                                        std::fabs(appliedDelta) / maxDeltaSamples,
                                        0.0,
                                        1.0));
                            }
                            // Publish the signed move so the retraction stage
                            // can align carrierFit with the repaired carrier.
                            if (repairDeltaRow)
                                repairDeltaRow[rel] =
                                    static_cast<float>(appliedDelta);
                        }
                    }
                }

                if (repairLogThisLine &&
                    rel >= repairLogFirst && rel <= repairLogLast)
                {
                    std::fprintf(stderr,
                        "COARSERESREPAIR line=%d rel=%d h=%d phase=%d "
                        "mode=%s reason=%s sourceBp=%.6f shortFit=%.6f "
                        "wideFit=%.6f movingResidual=%.6f optionCount=%d "
                        "survivorCount=%d survivorLo=%.6f survivorHi=%.6f "
                        "tolIRE=%.6f maxDeltaIRE=%.6f proposedDeltaIRE=%.6f "
                        "appliedDeltaIRE=%.6f movingDistIRE=%.6f "
                        "maxAbsMembershipIRE=%.6f sourceMinusShortIRE=%.6f "
                        "shortMinusWideIRE=%.6f sourceMinusWideIRE=%.6f\n",
                        line, rel, left + rel, p, "default", reason, sourceSample,
                        shortSample, wideSample, movingResidual, optionCount,
                        survivorCount, survivorLo, survivorHi,
                        std::max(0.0, parallaxRepairTolIRE),
                        std::max(0.0, parallaxRepairMaxDeltaIRE),
                        proposedDelta * invIreScale,
                        appliedDelta * invIreScale,
                        movingDistIRE,
                        maxAbsMembershipIRE,
                        (sourceSample - shortSample) * invIreScale,
                        (shortSample - wideSample) * invIreScale,
                        (sourceSample - wideSample) * invIreScale);
                }
            }
        }

        float *impurityRow = carrierImpurity_line(line);

        // Pass 2: wide-window cross-color detector.  Publishes carrierImpurity;
        // the emitted source is the bandpass under the envelope-legality
        // restraint below (an envelope-scale weight, never a carrier-rate gain).
        //
        // Demodulate the bandpass into quadrature against the period-4
        // reference (for the wide coherent fit) and form the 2-sample fit
        // envelope (for the rolling narrow fit).
        for (int rel = 0; rel < width; ++rel) {
            const int p = carrierSampleClass(line, left + rel) & 3;
            demI[rel] = bpLine[rel] * cosRef[p];
            demQ[rel] = bpLine[rel] * sinRef[p];
            const int relN = std::min(rel + 1, width - 1);
            const double b0 = bpLine[rel], b1 = bpLine[relN];
            env[rel] = std::sqrt(b0 * b0 + b1 * b1);
        }
        preI[0] = 0.0;
        preQ[0] = 0.0;
        preEnv[0] = 0.0;
        for (int rel = 0; rel < width; ++rel) {
            preI[rel + 1] = preI[rel] + demI[rel];
            preQ[rel + 1] = preQ[rel] + demQ[rel];
            preEnv[rel + 1] = preEnv[rel] + env[rel];
        }

        // Wide coherent envelope: |sum(I,Q)| over the window, normalized so a
        // coherent carrier of amplitude A returns ~A regardless of width.
        auto wideEnvIRE = [&](int center) -> double {
            const double meanI = centeredEvenWeightMean(
                demI.data(), preI.data(), width, center, kWideWin);
            const double meanQ = centeredEvenWeightMean(
                demQ.data(), preQ.data(), width, center, kWideWin);
            return 2.0 * boundedMag(meanI, meanQ) * invIreScale;
        };

        // Narrow fit: rolling, current-centered mean of the 2-sample envelope.
        // The point envelope = A on coherent carrier but ripples at 2fsc under
        // phase error; the mean over 4 cycles nulls that ripple.  env[k]
        // describes the pair (k,k+1), so its physical coordinate is k+0.5:
        // indices center-8 through center+7 are already centred on `center`.
        auto narrowEnvIRE = [&](int center) -> double {
            const int a = std::clamp(center - kNarrowWin / 2, 0, width);
            const int b = std::clamp(a + kNarrowWin, 0, width);
            const double n = static_cast<double>(std::max(1, b - a));
            return ((preEnv[b] - preEnv[a]) / n) * invIreScale;
        };

        // Envelope-legality restraint on the emitted source (encoder
        // bandwidth law, imposed at envelope scale).
        //
        // A legal chroma envelope is bandlimited to 1.3 MHz -- ~11 samples
        // at 4fSC -- so corroboration evidence about the envelope is
        // meaningful only at that scale, and the ceiling is the encoder's
        // own law: demodulated I/Q passed through the encoder's 1.3 MHz
        // chroma kernel is everything the encoder could have modulated
        // here; envelope the source holds above that (plus noise slack) is
        // inexpressible as chroma.  It is luma the blind bandpass
        // swallowed, and restraining it returns the energy to Y through
        // raw - lockedSource, where it belongs.  Both sides of the ratio
        // are smoothed by the SAME kernel, so the comparison never mixes
        // scales.  (Two falsified ceilings, kept as negative results: the
        // 2-sample POINT envelope under an envelope-scale ceiling rectifies
        // noise/sideband ripple into ~25% desaturation of legal saturated
        // bars; a coherent VECTOR mean over the law window punishes legal
        // I/Q modulation -- the law bounds envelope bandwidth, not phasor
        // constancy -- and still cost the bars ~21%.)
        //
        // The historical prohibition on any source gain ("checkerboard by
        // construction") was a prohibition on CARRIER-RATE gain: a
        // bandlimited carrier times a fast gain is amplitude modulation that
        // manufactures out-of-band sidebands.  This weight is gathered at
        // envelope scale and applied through the encoder's own envelope
        // kernel, so it varies no faster than a legal envelope may -- it
        // cannot manufacture sidebands.  Genuine chroma, including legal
        // 1.3 MHz edges, is expressible by construction and passes at
        // w = 1; the weight never exceeds 1, so it can only return energy
        // to Y, never manufacture carrier.
        const bool crDiagThisLine =
            crDiagLine >= 0 && line == crDiagLine && crDiagC0 >= 0;
        const int crDiagFirst =
            crDiagThisLine ? std::clamp(crDiagC0, 0, width - 1) : 0;
        const int crDiagLast =
            crDiagThisLine
                ? std::clamp(crDiagC1 < 0 ? crDiagC0 : crDiagC1,
                             crDiagFirst, width - 1)
                : -1;
        if (crDiagThisLine) {
            std::fprintf(stderr,
                "COARSERESFIT header line rel h phase sourceBp shortFit wideFit "
                "shortTolIRE shortLo shortHi optionCount survivorCount survivorLo "
                "survivorHi nearestShortDistIRE sourceClampDeltaIRE "
                "wideClampDeltaIRE sourceMinusShortIRE shortMinusWideIRE "
                "sourceMinusWideIRE narrowMagIRE wideMagIRE impurity\n");
            std::fprintf(stderr,
                "COARSERESFITOPT header line rel view s residual residualIRE "
                "distShortIRE distWideIRE inShortBand membershipDeltaIRE "
                "membershipLocalX\n");
        }

        // Aperture cross-color detector.  Publishes gA = aperture contamination
        // as carrierImpurity; gA is never a source gain.
        //
        // gA is NOT applied to the source.  Coherent contamination (dubbed
        // cross-color) is corroborated at envelope scale, so the legality
        // restraint above passes it untouched; discriminating it from
        // authentic chroma is gA's job, and that correction lives on the
        // COLOR side only, as the gA alpha applied in splitIQlocked().
        // A per-pixel gA gain on the source would be carrier-rate AM, and
        // removing coherent carrier from the source would strand its
        // complement in Y as checkerboard. The lockedProduct chroma path
        // therefore remains separate from luma policy.
        //
        // Winding is deliberately NOT used: control measurements on saturated
        // clothing showed coh/turn overlap authentic chroma and contamination,
        // so winding cannot discriminate without a luma-coupling guard.  gA reads
        // 0 on the uniforms and 0.2-0.3 on title cross-color, so aperture alone
        // is the correct discriminator.
        // Hoist line-invariant bounds out of the per-pixel loop. fvfMetrics is
        // a 2D vector keyed on (line, rel); both bounds are constants within
        // this line. Attribution likewise is a per-line pointer.
        const int fvfRelLimit =
            (line >= 0 && line < static_cast<int>(fvfMetrics.size()))
                ? static_cast<int>(fvfMetrics[line].size())
                : 0;
        auto *fvfLineRow =
            fvfRelLimit > 0 ? fvfMetrics[line].data() : nullptr;
        for (int rel = 0; rel < width; ++rel) {
            // Stable centre Zwide on the cycle grid (8-cycle complex mean).
            const double ZwI = centeredEvenWeightMean(
                demI.data(), preI.data(), width, rel, kWideWin);
            const double ZwQ = centeredEvenWeightMean(
                demQ.data(), preQ.data(), width, rel, kWideWin);

            const double narrowMag = narrowEnvIRE(rel);
            const double wideMag = 2.0 * boundedMag(ZwI, ZwQ) * invIreScale;

            // Aperture contamination: clamp((narrow - wide)/max(floor, narrow)).
            double gA = 0.0;
            if (narrowMag > kImpurityFloorIRE && wideMag < narrowMag) {
                gA = clamp01(
                    (narrowMag - wideMag) /
                    std::max(kImpurityFloorIRE, narrowMag));
            }

            if (crDiagThisLine && rel >= crDiagFirst && rel <= crDiagLast) {
                const int p = carrierSampleClass(line, left + rel) & 3;
                const double wideSample =
                    2.0 * (ZwI * cosRef[p] + ZwQ * sinRef[p]);

                const double ZnI = centeredEvenWeightMean(
                    demI.data(), preI.data(), width, rel, kNarrowWin);
                const double ZnQ = centeredEvenWeightMean(
                    demQ.data(), preQ.data(), width, rel, kNarrowWin);
                const double shortSample =
                    2.0 * (ZnI * cosRef[p] + ZnQ * sinRef[p]);

                const double fitTolSamples =
                    std::max(0.0, crDiagFitTolIRE) * irescale;
                const double shortLo = shortSample - fitTolSamples;
                const double shortHi = shortSample + fitTolSamples;

                const int sFirst = std::max(0, rel - 3);
                const int sLast = (width >= 4)
                    ? std::min(rel, width - 4)
                    : -1;

                double survivorLo = 1e300;
                double survivorHi = -1e300;
                double nearestDistIRE = 1e300;
                int optionCount = 0;
                int survivorCount = 0;

                for (int s = sFirst; s <= sLast && optionCount < 4; ++s) {
                    const double y4 =
                        0.25 * (rawAtRel(s + 0) +
                                rawAtRel(s + 1) +
                                rawAtRel(s + 2) +
                                rawAtRel(s + 3));
                    const double residual = rawAtRel(rel) - y4;
                    const double distShortIRE =
                        std::fabs(residual - shortSample) * invIreScale;
                    const double distWideIRE =
                        std::fabs(residual - wideSample) * invIreScale;
                    const double distBandIRE =
                        (residual < shortLo)
                            ? (shortLo - residual) * invIreScale
                            : ((residual > shortHi)
                                ? (residual - shortHi) * invIreScale
                                : 0.0);
                    nearestDistIRE = std::min(nearestDistIRE, distBandIRE);

                    const bool inBand =
                        residual >= shortLo && residual <= shortHi;
                    if (inBand) {
                        ++survivorCount;
                        survivorLo = std::min(survivorLo, residual);
                        survivorHi = std::max(survivorHi, residual);
                    }

                    double membershipDeltaIRE = 0.0;
                    double membershipLocalX = 0.0;
                    if (s + 4 < width) {
                        membershipDeltaIRE =
                            0.25 * (rawAtRel(s + 4) - rawAtRel(s)) * invIreScale;
                        membershipLocalX =
                            0.5 * (static_cast<double>(s) +
                                   static_cast<double>(s + 4)) -
                            static_cast<double>(rel);
                    }

                    std::fprintf(stderr,
                        "COARSERESFITOPT line=%d rel=%d view=%d s=%d "
                        "residual=%.6f residualIRE=%.6f distShortIRE=%.6f "
                        "distWideIRE=%.6f inShortBand=%d membershipDeltaIRE=%.6f "
                        "membershipLocalX=%.6f\n",
                        line, rel, optionCount, s, residual,
                        residual * invIreScale, distShortIRE, distWideIRE,
                        inBand ? 1 : 0, membershipDeltaIRE, membershipLocalX);
                    ++optionCount;
                }

                if (survivorCount == 0) {
                    survivorLo = 0.0;
                    survivorHi = 0.0;
                }
                if (nearestDistIRE == 1e300)
                    nearestDistIRE = 0.0;

                auto clampDeltaIRE = [&](double sample) {
                    if (survivorCount <= 0)
                        return 0.0;
                    const double clamped =
                        std::clamp(sample, survivorLo, survivorHi);
                    return (clamped - sample) * invIreScale;
                };

                const double sourceSample = bpLine[rel];
                const double sourceMinusShortIRE =
                    (sourceSample - shortSample) * invIreScale;
                const double shortMinusWideIRE =
                    (shortSample - wideSample) * invIreScale;
                const double sourceMinusWideIRE =
                    (sourceSample - wideSample) * invIreScale;
                std::fprintf(stderr,
                    "COARSERESFIT line=%d rel=%d h=%d phase=%d "
                    "sourceBp=%.6f shortFit=%.6f wideFit=%.6f "
                    "shortTolIRE=%.6f shortLo=%.6f shortHi=%.6f "
                    "optionCount=%d survivorCount=%d survivorLo=%.6f "
                    "survivorHi=%.6f nearestShortDistIRE=%.6f "
                    "sourceClampDeltaIRE=%.6f wideClampDeltaIRE=%.6f "
                    "sourceMinusShortIRE=%.6f shortMinusWideIRE=%.6f "
                    "sourceMinusWideIRE=%.6f narrowMagIRE=%.6f wideMagIRE=%.6f "
                    "impurity=%.6f\n",
                    line, rel, left + rel, p, sourceSample, shortSample,
                    wideSample, std::max(0.0, crDiagFitTolIRE), shortLo, shortHi,
                    optionCount, survivorCount, survivorLo, survivorHi,
                    nearestDistIRE, clampDeltaIRE(sourceSample),
                    clampDeltaIRE(wideSample), sourceMinusShortIRE,
                    shortMinusWideIRE, sourceMinusWideIRE, narrowMag, wideMag,
                    gA);
            }

            // The emitted source is the full-resolution ordinary carrier plus
            // any explicitly bounded 1D repair. Whitestar and the fits are
            // evidence/policy inputs only; no diagnostic projection becomes
            // picture here.
            //
            // The envelope-legality hull was REMOVED from the 1D source
            // (2026-07-25). A hull presumes a safer value to retreat to when
            // the estimate looks illegal; in produceY an outlier candidate can
            // simply lose an election. In 1D there is no such harbour -- the
            // bandpass IS the only estimate -- so bounding it is not a choice
            // between candidates but an unconditional subtraction from the sole
            // source, and whatever it removes lands in luma via Y = raw - src.
            // Worse, the ceiling is measured with the encoder's 9-tap 1.3 MHz
            // kernel, which smooths ACROSS a thin feature and therefore reads
            // its envelope as smaller than it is: the hull then cuts LEGAL
            // carrier at exactly the compact features it should protect, and
            // that carrier reappears in luma as carrier-rate alternation.
            // Measured on the beach strap (pure luma, --chroma-gain 0, vs
            // bucket): hull on 148 rms / peak 1383; hull off 83 / 230 -- the
            // strap-local dominant term while barely moving the frame-wide
            // figure, which is the signature of something that only bites on
            // thin detail. With the hull and the Pass-1.5 repair both out, the
            // locked 1D luma is BIT-IDENTICAL to bucket.
            restrainedLine[rel] = bpLine[rel];

            if (impurityRow)
                impurityRow[rel] = static_cast<float>(gA);

            if (fvfLineRow && rel < fvfRelLimit) {
                fvfLineRow[rel].intakeNyquistRiskIRE = gA * narrowMag;
            }

            if (attribution) {
                AttributionFacts &facts = attribution[rel].facts;
                facts.bandpassFineIRE = narrowMag;
                facts.bandpassCoarseIRE = wideMag;
                facts.lumaExcursionIRE = gA * narrowMag;
            }
        }

        // Optional publication is deliberately outside the render-facing loop
        // above.  When no analysis client is active, ordinary locked output has
        // no extra per-pixel branch or storage traffic.
        if (carrierAnalysis) {
            for (int rel = 0; rel < width; ++rel) {
                const int p = carrierSampleClass(line, left + rel) & 3;
                const double ZwI = centeredEvenWeightMean(
                    demI.data(), preI.data(), width, rel, kWideWin);
                const double ZwQ = centeredEvenWeightMean(
                    demQ.data(), preQ.data(), width, rel, kWideWin);
                const double wideSample =
                    2.0 * (ZwI * cosRef[p] + ZwQ * sinRef[p]);
                const double wideMag =
                    2.0 * boundedMag(ZwI, ZwQ) * invIreScale;
                const double narrowMag = narrowEnvIRE(rel);

                auto &record = carrierAnalysis[rel];
                if (!record.fit.valid) {
                    const double ZnI = centeredEvenWeightMean(
                        demI.data(), preI.data(), width, rel, kNarrowWin);
                    const double ZnQ = centeredEvenWeightMean(
                        demQ.data(), preQ.data(), width, rel, kNarrowWin);
                    const double shortSample =
                        2.0 * (ZnI * cosRef[p] + ZnQ * sinRef[p]);
                    const double sourceSample = bpLine[rel];

                    record.fit.sourceSample = static_cast<float>(sourceSample);
                    record.fit.shortSample = static_cast<float>(shortSample);
                    record.fit.wideSample = static_cast<float>(wideSample);
                    record.fit.sourceMinusShortIRE = static_cast<float>(
                        (sourceSample - shortSample) * invIreScale);
                    record.fit.shortMinusWideIRE = static_cast<float>(
                        (shortSample - wideSample) * invIreScale);
                    record.fit.sourceMinusWideIRE = static_cast<float>(
                        (sourceSample - wideSample) * invIreScale);
                    record.fit.valid = true;
                }

                record.fit.shortMagnitudeIRE = static_cast<float>(narrowMag);
                record.fit.wideMagnitudeIRE = static_cast<float>(wideMag);
                const float apertureImpurity = static_cast<float>(
                    (narrowMag > kImpurityFloorIRE && wideMag < narrowMag)
                        ? clamp01((narrowMag - wideMag) /
                                  std::max(kImpurityFloorIRE, narrowMag))
                        : 0.0);
                record.carrierImpurity = std::max(
                    record.carrierImpurity,
                    apertureImpurity);
            }
        }

        // ---- Phase-1 cross-color measurement diagnostic (gated, no output) ----
        // Touches nothing in the signal path: not the source, not the impurity
        // buffer, not alphaEff.  When CC_DIAG_LINE is set it dumps, at columns
        // [CC_DIAG_C0,CC_DIAG_C1] (active-picture-relative), the purity facts
        // plus residual-winding facts.  Purpose: see whether contamination reads
        // rotating/scattered (high turn / low coherence, or low coherence with
        // amplitude) while authentic chroma reads aligned (residual phase steady,
        // stable centre nonzero) on the actual title frame, before any
        // correction is enabled.  (Gate vars are parsed once at function top.)
        if (ccDiagLine >= 0 && line == ccDiagLine && ccDiagC0 >= 0) {
            constexpr double kPi = 3.14159265358979323846;
            const int c0 = std::clamp(ccDiagC0, 0, width - 1);
            const int c1 = std::clamp(ccDiagC1 < 0 ? ccDiagC0 : ccDiagC1, c0, width - 1);
            for (int rel = c0; rel <= c1; ++rel) {
                // Stable centre Zwide (8-cycle complex mean).
                const double ZwI = centeredEvenWeightMean(
                    demI.data(), preI.data(), width, rel, kWideWin);
                const double ZwQ = centeredEvenWeightMean(
                    demQ.data(), preQ.data(), width, rel, kWideWin);
                const double stableAmpIRE = 2.0 * std::hypot(ZwI, ZwQ) * invIreScale;

                const double narrowMag = narrowEnvIRE(rel);
                const double wideMag   = wideEnvIRE(rel);
                const double gA = (narrowMag > kImpurityFloorIRE && wideMag < narrowMag)
                    ? clamp01((narrowMag - wideMag) / std::max(kImpurityFloorIRE, narrowMag))
                    : 0.0;

                // Residual winding over the narrow support: per-cycle complex
                // envelope minus the stable centre, amplitude-weighted phase walk.
                double sw = 0.0, scos = 0.0, ssin = 0.0, sturn = 0.0;
                double sRmag = 0.0, sZmag = 0.0;
                int nk = 0;
                double prevTheta = 0.0;
                bool havePrev = false;
                const int ka = rel - kNarrowWin / 2;
                const int kb = rel + kNarrowWin / 2;
                for (int k = ka; k <= kb; ++k) {
                    auto demAt = [&](const std::vector<double> &v, int x) {
                        return v[std::clamp(x, 0, width - 1)];
                    };
                    const double ZcI = centeredCarrierCycle4Mean(
                        demAt(demI, k - 2), demAt(demI, k - 1),
                        demAt(demI, k), demAt(demI, k + 1),
                        demAt(demI, k + 2));
                    const double ZcQ = centeredCarrierCycle4Mean(
                        demAt(demQ, k - 2), demAt(demQ, k - 1),
                        demAt(demQ, k), demAt(demQ, k + 1),
                        demAt(demQ, k + 2));
                    const double Ri = ZcI - ZwI;
                    const double Rq = ZcQ - ZwQ;
                    const double Rmag = std::hypot(Ri, Rq);
                    sRmag += Rmag;
                    sZmag += std::hypot(ZcI, ZcQ);
                    ++nk;
                    const double theta = std::atan2(Rq, Ri);
                    if (havePrev) {
                        double d = theta - prevTheta;
                        while (d >  kPi) d -= 2.0 * kPi;
                        while (d < -kPi) d += 2.0 * kPi;
                        const double w = Rmag;
                        sw    += w;
                        scos  += w * std::cos(d);
                        ssin  += w * std::sin(d);
                        sturn += w * d;
                    }
                    prevTheta = theta;
                    havePrev = true;
                }
                const double incrementCoherence = sw > 0.0 ? std::hypot(scos, ssin) / sw : 0.0;
                const double netTurn = sw > 0.0 ? std::fabs(sturn) / sw : 0.0;
                const double meanRIRE = nk > 0 ? (sRmag / nk) * invIreScale : 0.0;
                const double meanZIRE = nk > 0 ? (sZmag / nk) * invIreScale : 0.0;
                const double residualStrength =
                    meanRIRE / std::max(kImpurityFloorIRE, meanZIRE);

                std::fprintf(stderr,
                    "CCDIAG line=%d col=%d narrow=%.2f wide=%.2f stable=%.2f "
                    "gA=%.3f coh=%.3f turn=%.3f rstr=%.3f\n",
                    line, rel, narrowMag, wideMag, stableAmpIRE,
                    gA, incrementCoherence, netTurn, residualStrength);
            }
        }

        // Pass 3a: publish two distinct NATIVE products without confusing
        // their phase contracts:
        //
        //   * demodI/Q and demodI/Q4 are sample-local carrier products;
        //   * restrainedLine is already a scalar carrier in physical composite
        //     sample geometry.
        //
        // The scalar must therefore be copied, not demodulated and remodulated.
        // The locked basis includes CAL_EPS while the Grid4fsc remodulator does
        // not; round-tripping the scalar through those unlike bases multiplies
        // it by cos(CAL_EPS*pi/2), leaving a small carrier residue in every
        // downstream raw-minus-carrier reconstruction.
        // Certified-field head replacement (stage 1): on def lines the 1D
        // scalar IS the conservation fact -- chat is already in def phase and
        // raw-aligned, so no basis work is needed. Per-sample repair holes
        // (NaN) keep the model's scalar; the hole boundary is a sample the
        // corrector already owns, not a policy seam. Every product derived
        // below -- demod caches, magnitude, attribution, clpbuffer[0] --
        // inherits truth from this one site.
        const float *certExRow =
            (certifiedOneDLevel() >= 1 && frameHasExactCoverage())
                ? exactCarrierRow(line)
                : nullptr;
        // Referee repair (2026-08-02): stash the PRE-head estimator so the
        // truth referees' "1D" column grades the model, not the fact it was
        // replaced by. Probe-only; never consumed by render/decisions.
        static const bool refereeStashOn = []{
            return std::getenv("LDCD_PROBE_DSREF") != nullptr ||
                   std::getenv("LDCD_PROBE_CCREF") != nullptr ||
                   std::getenv("LDCD_PROBE_ANTGRADE") != nullptr ||
                   std::getenv("LDCD_PROBE_TWEEN") != nullptr;
        }();
        float *preHeadRow = nullptr;
        if (refereeStashOn && certExRow) {
            // Reset at the first covered line so a reused buffer never
            // serves a previous frame's estimator values.
            if (probePreHead1D_flat.size() !=
                    static_cast<size_t>(demodLines) * demodWidth ||
                line == first)
                probePreHead1D_flat.assign(
                    static_cast<size_t>(demodLines) * demodWidth,
                    std::numeric_limits<float>::quiet_NaN());
            preHeadRow = probePreHead1D_flat.data() +
                static_cast<size_t>(line) * demodWidth;
        }

        // Stage 1b -- comp-line HEAD phase snap (user, 2026-07-30: "move the
        // snap earlier... Frame B is equally important"). On comp lines of
        // covered frames the 1D scalar takes the certified bracket's phase
        // and keeps its own amplitude, HERE, so Field A/B, Frame A/B, the
        // CCR reads and the chroma demod all inherit it through the derived
        // products -- no candidate-local rotation anywhere. Same shared snap
        // the retraction ladder uses (amplitude floor 1 IRE, tau 3; the
        // reference's phase was validated >= 2 IRE, coherence 0.907). The
        // scalar-publish law is honoured: the snap composes waveforms in
        // the working space, no demod/remod round-trip through unlike bases.
        // Escape LDCD_CERT_PHASE=0.
        static const bool certHeadPhaseOn = []{
            const char *e = std::getenv("LDCD_CERT_PHASE");
            return !(e && std::atoi(e) == 0);
        }();
        if (certHeadPhaseOn && certifiedOneDLevel() >= 1 &&
            frameHasExactCoverage() && !certifiedDefLine(line) &&
            line - 1 >= first && line + 1 < last) {
            std::vector<double> legU(width), legD(width), hueAlpha(width);
            ldcdBuildCertBracketAligned(
                carrierGrammarLine(line),
                carrierGrammarLine(line - 1),
                carrierGrammarLine(line + 1),
                exactCarrierRow(line - 1), exactCarrierRow(line + 1),
                left, width, snapRef, &legU, &legD);
            std::copy(restrainedLine, restrainedLine + width, snapEst.begin());
            std::fill(snapOut.begin(), snapOut.end(),
                      std::numeric_limits<double>::quiet_NaN());
            ldcdApplyPhaseSnap(snapEst, snapRef, snapOut, width, irescale,
                               1.0, 3.0, false);
            // Hue-boundary verdict as a smoothed ALPHA, never a cut: leg
            // coherence ramp, lateral boxcar ~1 carrier cycle, then blend.
            // A binary keep here interleaved snapped/unsnapped carrier at
            // pixel pitch along chroma edges -- rendered checkerboard.
            ldcdSideCoherenceAlpha(legU, legD, width, hueAlpha);
            constexpr int kAr = 4;
            for (int xi = 0; xi < width; ++xi) {
                double acc = 0.0; int n = 0;
                for (int j = std::max(0, xi - kAr);
                     j <= std::min(width - 1, xi + kAr); ++j) {
                    acc += hueAlpha[j]; ++n;
                }
                const double al = acc / std::max(1, n);
                restrainedLine[xi] = al * snapOut[xi] +
                                     (1.0 - al) * snapEst[xi];
            }
        }

        for (int rel = 0; rel < width; ++rel) {
            const int h = left + rel;
            const int phase = carrierSampleClass(line, h);
            double source = restrainedLine[rel];
            if (preHeadRow)
                preHeadRow[rel] = static_cast<float>(source);
            if (certExRow && std::isfinite(certExRow[h]))
                source = (double)certExRow[h];

            const double i = source * lutI[phase];
            const double q = source * lutQ[phase];
            const double i4 = source * i4Scale[phase];
            const double q4 = source * q4Scale[phase];
            nativeI4[rel] = i4;
            nativeQ4[rel] = q4;

            if (demodI) demodI[rel] = static_cast<float>(i);
            if (demodQ) demodQ[rel] = static_cast<float>(q);
            if (demodI4) demodI4[rel] = static_cast<float>(i4);
            if (demodQ4) demodQ4[rel] = static_cast<float>(q4);
            lockedSource[rel] = source;
        }

        // Pass 3b: the pre-comb IQ authority is a separate pair of
        // integer-centred baseband products.  The symmetric three-sample
        // aperture cancels the alternating product image while keeping both
        // axes registered at the native sample h.  It is deliberately never
        // remodulated into lockedSource: raw-minus-carrier must continue to
        // consume the physical scalar above.
        for (int rel = 0; rel < width; ++rel) {
            const int rm = std::max(0, rel - 1);
            const int rp = std::min(width - 1, rel + 1);
            const double i4 = centeredCarrierProduct3(
                nativeI4[rm], nativeI4[rel], nativeI4[rp]);
            const double q4 = centeredCarrierProduct3(
                nativeQ4[rm], nativeQ4[rel], nativeQ4[rp]);

            if (lockedI4) lockedI4[rel] = static_cast<float>(i4);
            if (lockedQ4) lockedQ4[rel] = static_cast<float>(q4);

            const double chromaMagnitude = boundedMag(i4, q4);
            magnitude[rel] = static_cast<float>(chromaMagnitude);
            if (attribution) {
                AttributionFacts &facts = attribution[rel].facts;
                facts.locked1DChromaIRE =
                    chromaMagnitude * invIreScale;
            }
        }

        if (grammar)
            grammar->projectionValid = false;

        double *published = clpbuffer[0].pixel[line];
        for (int rel = 0; rel < width; ++rel)
            published[left + rel] = lockedSource[rel];
    }

    // Trailing pass: ±2 vertical partner evidence, computed once the IQ
    // caches above are complete for every line.  Two verdicts per pixel:
    //
    //   SAME  — two schedule-admitted carrier operands positively share this
    //           pixel's chroma region (relation-signed hue agreement above
    //           the chroma floor).  A real chroma-region boundary and
    //           cross-color both fail interline carrier verification, so
    //           the gA detector alone cannot tell them apart; this is the
    //           discriminator.  Consumed by the suppression alpha.
    //
    //   ALIEN — a relation-admitted operand is ANTI-aligned at comparable
    //           magnitude after relation signing: raw-identical content where
    //           the carrier grammar says the operands are phase-comparable.
    //           That is the comb's cancellation partner, not a hue boundary
    //           to cede to 1D (near-carrier periodic luma, the Borg-cube grid).
    //           Consumed by the produceY retracted admission.
    //
    // Both are evidence only; consumers convert.  Diagnostic gate:
    // LD_REGION_KEEP=0 disables the SAME verdict (suppression veto behaves
    // exactly as before 2026-07-02) without touching the ALIEN fact — the
    // single-variable isolation for witness-render regressions.
    static const bool regionKeepEnabled = []{
        const char *s = std::getenv("LD_REGION_KEEP");
        return !(s && s[0] == '0');
    }();

    if (!regionSamePartner_flat.empty() &&
        !regionAlienPartner_flat.empty() && demodWidth >= width) {
        const lddecode::CombReachSourceFrame iqSource = iqReachSource();

        struct RegionLegReach {
            bool allow = false;
            lddecode::CarrierPhaseRelation relation =
                lddecode::CarrierPhaseRelation::Unknown;
        };

        for (int line = first; line < last; ++line) {
            float *sameRow = regionSamePartner_line(line);
            float *alienRow = regionAlienPartner_line(line);
            if (!sameRow || !alienRow)
                continue;
            std::fill(sameRow, sameRow + width, 0.0f);
            std::fill(alienRow, alienRow + width, 0.0f);

            const float *i0 = locked1DTI4fsc_line(line);
            const float *q0 = locked1DTQ4fsc_line(line);
            if (!i0 || !q0)
                continue;

            auto legReach = [&](int target) -> RegionLegReach {
                RegionLegReach leg;
                if (target < first || target >= last)
                    return leg;
                const lddecode::CombReachReply reach = combReachIndex.query(
                    {line, target, left, left,
                     lddecode::CombReachUse::IQCompare, iqSource});
                if (!reach.allowIQCompare)
                    return leg;
                leg.allow =
                    reach.carrierRelation == lddecode::CarrierPhaseRelation::Same ||
                    reach.carrierRelation == lddecode::CarrierPhaseRelation::Opposite;
                leg.relation = reach.carrierRelation;
                return leg;
            };
            const RegionLegReach upReach = legReach(line - 2);
            const RegionLegReach dnReach = legReach(line + 2);
            const float *iUp = upReach.allow
                ? locked1DTI4fsc_line(line - 2) : nullptr;
            const float *qUp = upReach.allow
                ? locked1DTQ4fsc_line(line - 2) : nullptr;
            const float *iDn = dnReach.allow
                ? locked1DTI4fsc_line(line + 2) : nullptr;
            const float *qDn = dnReach.allow
                ? locked1DTQ4fsc_line(line + 2) : nullptr;
            if (!iUp && !iDn)
                continue;

            for (int rel = 0; rel < width; ++rel) {
                // Symmetric 7-tap horizontal aggregate, matching the region
                // evaluator in buildCombTapLine.  Its input is already the
                // full, integer-centred IQ vector; divide by the complete
                // weight (6), not by the old per-axis weight (3).
                auto fullIQ = [&](const float *iR, const float *qR) {
                    static constexpr double w[7] =
                        {0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5};
                    double si = 0.0;
                    double sq = 0.0;
                    for (int k = -3; k <= 3; ++k) {
                        const int rk = std::clamp(rel + k, 0, width - 1);
                        si += w[k + 3] * static_cast<double>(iR[rk]);
                        sq += w[k + 3] * static_cast<double>(qR[rk]);
                    }
                    return std::complex<double>(si / 6.0, sq / 6.0);
                };
                const std::complex<double> z0 = fullIQ(i0, q0);
                const std::complex<double> zUp =
                    iUp ? fullIQ(iUp, qUp) : std::complex<double>(0.0, 0.0);
                const std::complex<double> zDn =
                    iDn ? fullIQ(iDn, qDn) : std::complex<double>(0.0, 0.0);
                const auto region =
                    CombContentReach::evaluateIntrafieldRegionReach(
                        z0, zUp, zDn,
                        upReach.relation,
                        dnReach.relation,
                        upReach.allow && iUp,
                        dnReach.allow && iDn,
                        0.5, 0.5, 0.5,
                        invIreScale,
                        5.0);

                if (regionKeepEnabled &&
                    (region.up == CombContentReach::RegionRelation::SameRegion ||
                     region.down == CombContentReach::RegionRelation::SameRegion))
                {
                    sameRow[rel] = 1.0f;
                }
                if (region.up == CombContentReach::RegionRelation::AlienCancel ||
                    region.down == CombContentReach::RegionRelation::AlienCancel)
                {
                    alienRow[rel] = 1.0f;
                }
            }
        }
    }
}

void Comb::FrameBuffer::measurePostCombImpurity()
{
    if (!configuration.phaseCompensation) return;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;

    if (width <= 0 || firstLine >= lastLine) return;

    constexpr int    kNarrowWin = 16;
    constexpr int    kWideWin   = 32;
    constexpr double kImpurityFloorIRE = 2.0;

    static const double cosRef[4] = { 1.0, 0.0, -1.0, 0.0 };
    static const double sinRef[4] = { 0.0, 1.0,  0.0, -1.0 };

    auto clamp01 = [](double v) {
        return std::clamp(v, 0.0, 1.0);
    };

    const int srcBuf = std::clamp(static_cast<int>(configuration.dimensions) - 1, 0, 2);

    static const int pcDiagLine = []{
        const char *s = std::getenv("CC_DIAG_LINE"); return s ? std::atoi(s) : -1;
    }();
    static const int pcDiagC0 = []{
        const char *s = std::getenv("CC_DIAG_C0"); return s ? std::atoi(s) : -1;
    }();
    static const int pcDiagC1 = []{
        const char *s = std::getenv("CC_DIAG_C1"); return s ? std::atoi(s) : -1;
    }();

    // Bandwidth-law cross-color contributor (opt-in, default off so the
    // authoritative impurity is byte-identical when disabled). Scores demod
    // envelope activity too fast to be legal carrier and MAXes it into gA; it
    // is a discriminator input for the color-side handler, never a source gain.
    static const bool bwCrossColor = []{
        const char *s = std::getenv("LDCD_BW_CROSSCOLOR");
        return s && std::atoi(s) != 0;
    }();

    // Coarse-residual carrier-legality veto (opt-in): where the aperture
    // parallax confirms legal carrier, pull gA down so the detector stops
    // desaturating compact colour it mistook for contamination. Default off so
    // the authoritative impurity is byte-identical when disabled.
    static const bool ccParallax = []{
        const char *s = std::getenv("LDCD_CC_PARALLAX");
        return s && std::atoi(s) != 0;
    }();

    std::vector<double> demI(width), demQ(width);
    std::vector<double> preI(width + 1), preQ(width + 1);
    std::vector<double> env(width), preEnv(width + 1);
    std::vector<double> lawI(width), lawQ(width);   // encoder-band envelope
    std::vector<double> pvI(size_t(4) * width), pvQ(size_t(4) * width);
    std::vector<double> psI(size_t(4) * width), psQ(size_t(4) * width);
    std::vector<double> parRatio(width);            // aperture parallax ratio

    for (int line = firstLine; line < lastLine; ++line) {
        const double *combLine = clpbuffer[srcBuf].pixel[line];
        float *impurityRow = carrierImpurity_line(line);
        if (!impurityRow) continue;

        for (int rel = 0; rel < width; ++rel) {
            const double s = combLine[left + rel];
            const int p = carrierSampleClass(line, left + rel) & 3;
            demI[rel] = s * cosRef[p];
            demQ[rel] = s * sinRef[p];
            const int relN = std::min(rel + 1, width - 1);
            const double sN = combLine[left + relN];
            env[rel] = std::sqrt(s * s + sN * sN);
        }

        preI[0] = 0.0;
        preQ[0] = 0.0;
        preEnv[0] = 0.0;
        for (int rel = 0; rel < width; ++rel) {
            preI[rel + 1] = preI[rel] + demI[rel];
            preQ[rel + 1] = preQ[rel] + demQ[rel];
            preEnv[rel + 1] = preEnv[rel] + env[rel];
        }

        // Lawful (<=1.3 MHz) envelope of the elected chroma: the most the
        // encoder could have modulated here. The demod's 2fSC image sits at
        // Nyquist, which this kernel nulls, so lawI/lawQ is the legal envelope
        // with the image already gone.
        if (bwCrossColor) {
            lddecode::projectExpressibleChromaEnvelope(
                demI.data(), nullptr, width, lawI.data());
            lddecode::projectExpressibleChromaEnvelope(
                demQ.data(), nullptr, width, lawQ.data());
        }

        if (ccParallax)
            apertureParallaxLine(line, pvI, pvQ, psI, psQ, parRatio.data());

        for (int rel = 0; rel < width; ++rel) {
            const double ZwI = centeredEvenWeightMean(
                demI.data(), preI.data(), width, rel, kWideWin);
            const double ZwQ = centeredEvenWeightMean(
                demQ.data(), preQ.data(), width, rel, kWideWin);

            // env[k] is centred at k+0.5, so this asymmetric index range is
            // the integer-centred physical aperture.
            const int na = std::clamp(rel - kNarrowWin / 2, 0, width);
            const int nb = std::clamp(na + kNarrowWin, 0, width);
            const double nn = std::max(1, nb - na);
            const double narrowMag =
                ((preEnv[nb] - preEnv[na]) / nn) * invIreScale;

            const double wideMag = 2.0 * boundedMag(ZwI, ZwQ) * invIreScale;

            double gA = 0.0;
            if (narrowMag > kImpurityFloorIRE && wideMag < narrowMag) {
                gA = clamp01(
                    (narrowMag - wideMag) /
                    std::max(kImpurityFloorIRE, narrowMag));
            }

            // Bandwidth-law contributor: the fraction of the elected chroma
            // envelope that a legal <=1.3 MHz envelope cannot reconstruct, i.e.
            // activity too fast to be carrier. imageFree = [1,2,1]/4 (stride 1)
            // removes the 2fSC image at Nyquist but keeps the 1.3 MHz-to-Nyquist
            // band; law = P{demod} keeps only the legal envelope; their
            // difference is the too-fast part. Floor-guarded so the soft 1.3 MHz
            // skirt and noise do not flag legal chroma edges. Scored, never
            // filtered -- MAXed into gA for the same color-side handler.
            if (bwCrossColor) {
                const int rm = std::max(0, rel - 1);
                const int rp = std::min(width - 1, rel + 1);
                const double ifI = 0.25 * (demI[rm] + 2.0 * demI[rel] + demI[rp]);
                const double ifQ = 0.25 * (demQ[rm] + 2.0 * demQ[rel] + demQ[rp]);
                const double crossMag = 2.0 *
                    std::hypot(ifI - lawI[rel], ifQ - lawQ[rel]) * invIreScale;
                const double actualMag =
                    2.0 * std::hypot(ifI, ifQ) * invIreScale;
                if (crossMag > kImpurityFloorIRE && actualMag > kImpurityFloorIRE)
                    gA = std::max(gA, clamp01(crossMag / actualMag));
            }

            // Carrier-legality veto: where the aperture parallax says this
            // carrier-band energy NULLS in every legal four-window (low ratio),
            // it is legal carrier -- including compact colour a wide coherent
            // window would smear -- so pull gA down. Where it fails to null
            // (high ratio), it is moving luma and gA passes. The ramp reuses the
            // corner-leak parallax thresholds (the same physical question).
            if (ccParallax) {
                const double keep = clamp01(
                    (parRatio[rel] - kCornerParallaxSoft) /
                    (kCornerParallaxHard - kCornerParallaxSoft));
                gA *= keep;
            }

            // The post-comb read is authoritative for CCR targeting. The
            // locked-1D seed published earlier is only a provisional read used
            // before the elected comb exists; once we have the elected result,
            // stale pre-comb suspicion must not linger and suppress solved
            // pixels.
            impurityRow[rel] = static_cast<float>(gA);

            if (pcDiagLine >= 0 && line == pcDiagLine &&
                rel >= pcDiagC0 && rel <= pcDiagC1) {
                std::fprintf(stderr,
                    "CCPOST line=%d col=%d srcBuf=%d narrow=%.2f wide=%.2f gA=%.3f\n",
                    line, rel, srcBuf, narrowMag, wideMag, gA);
            }
        }
    }
}

// Cross-color vertical-image-detail corroboration ramp, shared by the
// coherent (splitIQlocked) and residual (filterIQLocked) transfer policies.
// Input is the 1D vertical-contrast service (|smooth[rel+2]-smooth[rel-2]|,
// IRE): below soft the coarse field is laterally flat and the edge read is
// silent; hard matches the established FIELD_LUMA_EDGE scale (18 IRE = a
// solid vertical stroke).
static constexpr double kCcEdgeSoftIRE = 6.0;
static constexpr double kCcEdgeHardIRE = 18.0;

// Concert gate: carrier-free confirmation that the notch edge is a real luma
// transition and not saturated-carrier leak.  The notch (edgeRamp) localizes
// the corner sharply but doubles residual carrier in saturated colour;
// maxAbsMembershipIRE (the same-carrier-phase membership change = lurch) reads
// luma movement through the aperture with the carrier cancelled, so it is dark
// in a saturated-chroma interior and bright at a genuine luma edge.  Measured
// on the beach strap (frame 52100, line 150): smooth/interior lurch 0.3-1.4,
// true skin<->strap edges 5-10.  soft/hard bracket that gap.
static constexpr double kCcLurchSoftIRE = 1.0;
static constexpr double kCcLurchHardIRE = 4.0;

// Carrier-presence floor for the return. Numerically the aperture read's own
// kImpurityFloorIRE (buildPhaseCorrected1D / measurePostCombImpurity) -- the
// amplitude below which carrier-band energy is not a measurement. Named here
// so the composite verdict can obey the same law the aperture read already
// obeys internally.
static constexpr double kCcCarrierFloorIRE = 2.0;

// Committed-verdict cutoff (2026-08-02): the measured disagreement between
// the elected carrier and the anchored (anticipation-corrected) carrier, in
// IRE, below which the return bails and above which it commits fully.
// Confidence never sets a level; this is the single bail threshold. Sits at
// the noise-pedestal floor -- below kCcCarrierFloorIRE the "disagreement" is
// the capture's own noise, not confiscated luma (CCREF, 2026-07-30).
// LDCD_CC_CUTOFF overrides for A/B.
static constexpr double kCcCommitCutoffIRE = kCcCarrierFloorIRE;

// Fallback witness thresholds where no anchored plane exists (--dg-discard,
// non-cadence sources). gA's referee-endorsed regime starts at 0.40 (CCREF:
// anti-informative 0.2-0.4, superb above); regionKeep past 0.5 is a
// positively same-region vertical partner = a colour boundary, stand down.
// Binary predicates, never scales.
static constexpr double kCcFallbackGALo = 0.40;
static constexpr double kCcFallbackRegionKeepMax = 0.5;

void Comb::FrameBuffer::splitIQlocked(const FrameBuffer *prevF,
                                      const FrameBuffer *nextF)
{
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;
    const auto &T       = configuration.tunables;
    const int srcBuf    = std::clamp(static_cast<int>(configuration.dimensions) - 1, 0, 2);

    if (width <= 0 || firstLine >= lastLine) return;

    // Star classification is a split decision: establish the licensed
    // raw-domain footprint before publishing any elected carrier.  The same
    // immutable footprint is consumed by produceY and the final IQ renderer.
    buildStarFootprint(prevF, nextF);

    if (!basisLockedInit) {
        double Ce = 1.0, Se = 0.0;
        basisCoeffs(Ce, Se);
        for (int i = 0; i < 4; ++i) {
            double sp = 0.0, cp = 0.0;
            shiftedBasis(i, Ce, Se, sp, cp);
            spLUT_locked[i] = sp;
            cpLUT_locked[i] = cp;
        }
        basisLockedInit = true;
    }

    auto ensureScratch = [&](std::vector<double> &v) {
        if ((int)v.size() < width)
            v.resize(width, 0.0);
    };

    // These are consumed later by the locked IQ filter path.
    ensureScratch(scratch_preI);
    ensureScratch(scratch_preQ);

    auto finiteOrZero = [](double v) -> double {
        return std::isfinite(v) ? v : 0.0;
    };

    const double ccWeight = std::max(0.0, T.CC_SUPPRESSION_WEIGHT);
    const double giProduct = configuration.gi_product;
    const double gqProduct = configuration.gq_product;

    // Chroma-facts switch (escape LDCD_CHROMA_FACTS=0): on covered frames
    // the rendered products demodulate the anchored fact-family carrier
    // and the suppression envelope retires -- frame-uniform, so both
    // parities and every line move together.
    static const bool chromaFactsOn = []{
        const char *e = std::getenv("LDCD_CHROMA_FACTS");
        return !(e && std::atoi(e) == 0);
    }();

    // Cross-frame transfer referee (LDCD_PROBE_XFER=1, run -t 1):
    // held-out grading of the uncovered-frame successor BEFORE it is
    // built. Snapshot each covered frame's exact plane; when the next
    // covered frame arrives (pitch 2 -- twice the real transfer pitch,
    // so every number is a pessimistic bound), grade three transfer
    // grades per 32x128 region, exact against exact:
    //   raw  -- copy the neighbour's carrier verbatim
    //   rot  -- best per-region phase rotation (what the sync tone buys)
    //   full -- best gain + rotation (2-parameter LS)
    // Ledger: rms residual in IRE per grade, region percentiles, and the
    // transferable fraction (regions whose best-grade residual < 1 IRE).
    if (std::getenv("LDCD_PROBE_XFER")) {
        static std::vector<float> xferPrev, xferPrev2;
        static int xferPrevIdx = -1, xferPrev2Idx = -1;
        static long xferFrameIdx = -1;
        xferFrameIdx++;
        const bool covered = frameHasExactCoverage();
        if (covered) {
            // Pitch-4 same-parity comparison first (same line, no vertical
            // contamination): separates the chat noise floor from real
            // content drift in the static regions.
            if (xferPrev2Idx >= 0 &&
                (int)(xferFrameIdx - xferPrev2Idx) == 4 &&
                (int)xferPrev2.size() == (lastLine - firstLine) * width) {
                const double invIre = 1.0 / irescale;
                double see = 0, seeRot = 0, seeFull = 0;
                long n = 0; std::vector<double> rFull;
                const int rnx = (width + 127) / 128;
                struct R4 { double saa=0,sbb=0,sab=0,sqb=0,see=0; long n=0; };
                std::vector<R4> regs((size_t)rnx *
                                     ((lastLine - firstLine + 31) / 32));
                for (int line = firstLine; line < lastLine; ++line) {
                    const float *exN = exactCarrierRow(line);
                    if (!exN) continue;
                    const float *exP = xferPrev2.data() +
                        (size_t)(line - firstLine) * width;
                    if (!std::isfinite((double)exP[width / 2])) continue;
                    const size_t rb =
                        (size_t)((line - firstLine) / 32) * rnx;
                    for (int xi = 1; xi < width - 1; ++xi) {
                        const double b = exN[left + xi];
                        const double a = exP[xi];
                        if (!std::isfinite(b) || !std::isfinite(a) ||
                            !std::isfinite((double)exP[xi - 1]) ||
                            !std::isfinite((double)exP[xi + 1])) continue;
                        const double aq = 0.5 * ((double)exP[xi - 1] -
                                                 (double)exP[xi + 1]);
                        R4 &R = regs[rb + xi / 128];
                        R.saa += a * a + aq * aq; R.sbb += b * b;
                        R.sab += a * b; R.sqb += aq * b;
                        R.see += (b - a) * (b - a); R.n++;
                    }
                }
                for (R4 &R : regs) {
                    if (R.n < 512) continue;
                    const double half = 0.5 * R.saa;
                    const double mag =
                        std::sqrt(R.sab * R.sab + R.sqb * R.sqb);
                    const double eRot =
                        std::max(0.0, R.sbb + 2.0 * half - 2.0 * mag);
                    const double eFull = (half > 1e-9)
                        ? std::max(0.0, R.sbb - mag * mag / half) : R.sbb;
                    see += R.see; seeRot += eRot; seeFull += eFull;
                    n += R.n;
                    rFull.push_back(std::sqrt(eFull / R.n) * invIre);
                }
                if (n > 0) {
                    std::sort(rFull.begin(), rFull.end());
                    auto pc = [&](double p) {
                        return rFull.empty() ? 0.0
                            : rFull[std::min(rFull.size() - 1,
                                             (size_t)(p * rFull.size()))];
                    };
                    long xferable = 0;
                    for (double v : rFull) if (v < 1.0) xferable++;
                    std::fprintf(stderr,
                        "[XFER4 f=%ld pitch4 same-line] n=%ld "
                        "rms raw %.2f rot %.2f full %.2f IRE | "
                        "full p10/50/90 %.2f/%.2f/%.2f | "
                        "transferable %.0f%%\n",
                        xferFrameIdx, n,
                        std::sqrt(see / n) * invIre,
                        std::sqrt(seeRot / n) * invIre,
                        std::sqrt(seeFull / n) * invIre,
                        pc(0.10), pc(0.50), pc(0.90),
                        100.0 * xferable /
                            std::max((size_t)1, rFull.size()));
                }
            }
            const int gap = (xferPrevIdx >= 0)
                ? (int)(xferFrameIdx - xferPrevIdx) : -1;
            if (gap == 2 && (int)xferPrev.size() ==
                    (lastLine - firstLine) * width) {
                const double invIre =
                    1.0 / irescale;
                struct Reg { double see[3] = {0,0,0};
                             double saa = 0, sbb = 0, sab = 0, sqb = 0;
                             long n = 0; long nRaw = 0; };
                const int rnx = (width + 127) / 128;
                const int rny = (lastLine - firstLine + 31) / 32;
                std::vector<Reg> regs((size_t)rnx * rny);
                long matchSame = 0, matchOff = 0;
                for (int line = firstLine; line < lastLine; ++line) {
                    const float *exN = exactCarrierRow(line);
                    if (!exN) continue;
                    // Same line if the prev covered frame held it; else the
                    // adjacent line (coverage parity can alternate between
                    // covered frames). Off-parity rows are graded by the
                    // rotation-invariant grades only -- rot/full absorb the
                    // 180-degree row flip -- so `raw` is same-line only.
                    const float *exP = nullptr;
                    bool sameLine = false;
                    for (int cand : { line, line - 1, line + 1 }) {
                        if (cand < firstLine || cand >= lastLine) continue;
                        const float *p = xferPrev.data() +
                            (size_t)(cand - firstLine) * width;
                        if (std::isfinite((double)p[width / 2])) {
                            exP = p; sameLine = (cand == line); break;
                        }
                    }
                    if (!exP) continue;
                    if (sameLine) matchSame++; else matchOff++;
                    const size_t rb =
                        (size_t)((line - firstLine) / 32) * rnx;
                    for (int xi = 1; xi < width - 1; ++xi) {
                        const int h = left + xi;
                        const double b = exN[h];
                        const double a = exP[xi];
                        if (!std::isfinite(b) || !std::isfinite(a) ||
                            !std::isfinite((double)exP[xi - 1]) ||
                            !std::isfinite((double)exP[xi + 1]))
                            continue;
                        // Quadrature of the PREV plane at 4fsc.
                        const double aq =
                            0.5 * ((double)exP[xi - 1] -
                                   (double)exP[xi + 1]);
                        Reg &R = regs[rb + xi / 128];
                        R.saa += a * a + aq * aq;
                        R.sbb += b * b;
                        R.sab += a * b;
                        R.sqb += aq * b;
                        if (sameLine) {
                            R.see[0] += (b - a) * (b - a);
                            R.nRaw++;
                        }
                        R.n++;
                    }
                }
                // Per-region closed forms: rot = unit-gain best rotation
                // (project b onto the a/aq circle at measured amplitude);
                // full = LS onto span{a, aq}.
                std::vector<double> resRaw, resRot, resFull;
                double gEE[3] = {0,0,0}; long gN = 0, gNRaw = 0;
                long transferable = 0;
                long regCount = 0;
                for (Reg &R : regs) {
                    if (R.n < 512) continue;
                    regCount++;
                    const double half = 0.5 * R.saa; // ~ per-phase power
                    const double mag =
                        std::sqrt(R.sab * R.sab + R.sqb * R.sqb);
                    const double eRaw = R.see[0];
                    // unit-gain rotation: |b - R(a)|^2 =
                    //   sbb + 2*half - 2*mag  (gain locked to prev's own)
                    const double eRot =
                        std::max(0.0, R.sbb + 2.0 * half - 2.0 * mag);
                    const double eFull = (half > 1e-9)
                        ? std::max(0.0, R.sbb - mag * mag / half)
                        : R.sbb;
                    if (R.nRaw > 256)
                        resRaw.push_back(
                            std::sqrt(eRaw / R.nRaw) * invIre);
                    resRot.push_back(std::sqrt(eRot / R.n) * invIre);
                    resFull.push_back(std::sqrt(eFull / R.n) * invIre);
                    gEE[0] += eRaw; gEE[1] += eRot; gEE[2] += eFull;
                    gN += R.n; gNRaw += R.nRaw;
                    if (std::sqrt(eFull / R.n) * invIre < 1.0)
                        transferable++;
                }
                if (gN > 0) {
                    auto pct = [](std::vector<double> &v, double p) {
                        if (v.empty()) return 0.0;
                        std::sort(v.begin(), v.end());
                        return v[std::min(v.size() - 1,
                            (size_t)(p * v.size()))];
                    };
                    std::fprintf(stderr,
                        "[XFER f=%ld pitch2] regions=%ld n=%ld "
                        "same/off lines %ld/%ld "
                        "rms raw %.2f rot %.2f full %.2f IRE | "
                        "raw p10/50/90 %.2f/%.2f/%.2f | "
                        "rot p10/50/90 %.2f/%.2f/%.2f | "
                        "full p10/50/90 %.2f/%.2f/%.2f | "
                        "transferable %.0f%%\n",
                        xferFrameIdx, regCount, gN,
                        matchSame, matchOff,
                        gNRaw > 0 ? std::sqrt(gEE[0] / gNRaw) * invIre
                                  : -1.0,
                        std::sqrt(gEE[1] / gN) * invIre,
                        std::sqrt(gEE[2] / gN) * invIre,
                        pct(resRaw, 0.10), pct(resRaw, 0.50),
                        pct(resRaw, 0.90),
                        pct(resRot, 0.10), pct(resRot, 0.50),
                        pct(resRot, 0.90),
                        pct(resFull, 0.10), pct(resFull, 0.50),
                        pct(resFull, 0.90),
                        100.0 * transferable / std::max(1L, regCount));
                }
            }
            // LUMA TWEEN REFEREE ([LTWEEN]): grade the tween witness for
            // the uncovered-frame return. Certified luma (the retracted
            // plane -- full frame on covered frames: exact on covered
            // lines, certified-comb on comp lines) is snapshotted for the
            // last three covered frames A,B,C; tween B' = (A+C)/2 and
            // grade against B's own certified luma. Pitch 4 end-to-end vs
            // the real mechanism's pitch 2, so every number is a
            // pessimistic bound. Ledger split by band: the witness's duty
            // is the CARRIER BAND (bp = [-.25,.5,-.25] at lag 2), where
            // trapped luma lives. hit = sites |bpB| >= 2 IRE (true
            // luma-in-band, the cross-colour candidates).
            static std::vector<float> lumaA, lumaB;
            static long lumaAIdx = -1, lumaBIdx = -1;
            if (lumaBIdx >= 0 && lumaAIdx >= 0 &&
                xferFrameIdx - lumaBIdx == 2 && lumaBIdx - lumaAIdx == 2 &&
                (int)lumaA.size() == (lastLine - firstLine) * width) {
                const double invIre = 1.0 / irescale;
                const int rnx = (width + 127) / 128;
                struct LR { double seF = 0, seB = 0, sxx = 0, syy = 0,
                                   sxy = 0, hitX = 0, hitY = 0,
                                   pX = 0, pY = 0;
                            long n = 0, nHit = 0, pN = 0, pTrue = 0; };
                std::vector<LR> regs((size_t)rnx *
                                     ((lastLine - firstLine + 31) / 32));
                for (int line = firstLine; line < lastLine; ++line) {
                    const float *retr = carrierRetracted_line(line);
                    if (!retr) continue;
                    const float *pA = lumaA.data() +
                        (size_t)(line - firstLine) * width;
                    const float *pB = lumaB.data() +
                        (size_t)(line - firstLine) * width;
                    const size_t rb =
                        (size_t)((line - firstLine) / 32) * rnx;
                    for (int xi = 2; xi < width - 2; ++xi) {
                        // B true = THIS frame is C; grade the MIDDLE frame:
                        // truth is pB, tween is (pA + this frame's luma)/2.
                        const double a  = pA[xi], b = pB[xi];
                        const double c  = retr[xi];
                        if (!std::isfinite(a) || !std::isfinite(b) ||
                            !std::isfinite(c)) continue;
                        auto bp = [&](auto at, int x) {
                            return (double)at(x) -
                                   0.5 * ((double)at(x - 2) +
                                          (double)at(x + 2));
                        };
                        auto atA = [&](int x){ return pA[x]; };
                        auto atB = [&](int x){ return pB[x]; };
                        auto atC = [&](int x){ return retr[x]; };
                        const double bpA = bp(atA, xi), bpB = bp(atB, xi),
                                     bpC = bp(atC, xi);
                        if (!std::isfinite(bpA) || !std::isfinite(bpB) ||
                            !std::isfinite(bpC)) continue;
                        const double tw   = 0.5 * (a + c);
                        const double bpTw = 0.5 * (bpA + bpC);
                        LR &R = regs[rb + xi / 128];
                        R.seF += (tw - b) * (tw - b);
                        R.seB += (bpTw - bpB) * (bpTw - bpB);
                        R.sxx += bpTw * bpTw; R.syy += bpB * bpB;
                        R.sxy += bpTw * bpB;
                        if (std::fabs(bpB) >= 2.0 * irescale) {
                            R.nHit++;
                            R.hitX += std::fabs(bpTw);
                            R.hitY += std::fabs(bpB);
                        }
                        // Precision side: when the tween witness FIRES
                        // (claims trapped luma), is it right? A false
                        // fire would return carrier into Y -- the one
                        // failure mode that manufactures checker.
                        if (std::fabs(bpTw) >= 2.0 * irescale) {
                            R.pN++;
                            if (std::fabs(bpB) >= 1.0 * irescale)
                                R.pTrue++;
                            R.pX += std::fabs(bpTw);
                            R.pY += std::fabs(bpB);
                        }
                        R.n++;
                    }
                }
                double gF = 0, gB = 0; long gN = 0;
                std::vector<double> corrs, rmsB;
                double hitX = 0, hitY = 0; long hitN = 0;
                double pXg = 0, pYg = 0; long pNg = 0, pTrueg = 0;
                for (LR &R : regs) {
                    if (R.n < 512) continue;
                    gF += R.seF; gB += R.seB; gN += R.n;
                    rmsB.push_back(std::sqrt(R.seB / R.n) * invIre);
                    if (R.sxx > 1e-9 && R.syy > 1e-9)
                        corrs.push_back(
                            R.sxy / std::sqrt(R.sxx * R.syy));
                    hitX += R.hitX; hitY += R.hitY; hitN += R.nHit;
                    pXg += R.pX; pYg += R.pY;
                    pNg += R.pN; pTrueg += R.pTrue;
                }
                if (gN > 0) {
                    auto pc = [](std::vector<double> &v, double p) {
                        if (v.empty()) return 0.0;
                        std::sort(v.begin(), v.end());
                        return v[std::min(v.size() - 1,
                                          (size_t)(p * v.size()))];
                    };
                    std::fprintf(stderr,
                        "[LTWEEN f=%ld pitch4] n=%ld rms full %.2f "
                        "bp %.2f IRE | bp-rms p10/50/90 %.2f/%.2f/%.2f | "
                        "bp-corr p10/50/90 %.2f/%.2f/%.2f | "
                        "hit n=%ld cover %.2f | "
                        "fire n=%ld precision %.2f magTrue %.2f\n",
                        xferFrameIdx, gN,
                        std::sqrt(gF / gN) * invIre,
                        std::sqrt(gB / gN) * invIre,
                        pc(rmsB, 0.10), pc(rmsB, 0.50), pc(rmsB, 0.90),
                        pc(corrs, 0.10), pc(corrs, 0.50), pc(corrs, 0.90),
                        hitN, hitY > 1e-9 ? hitX / hitY : 0.0,
                        pNg,
                        pNg > 0 ? (double)pTrueg / pNg : 0.0,
                        pXg > 1e-9 ? pYg / pXg : 0.0);
                }
            }
            // FLOW-TWEEN REFEREE ([LTWEEN2]): grade TRUE tweening against
            // the averaged form above (user, 2026-07-31: "The luma can't
            // be averaged; that isn't tweening. Tweening... is creating
            // in-between animations of the boundary vectors -- think of
            // cheap slow motion that averages the frame vs optical
            // flow"). Per 32x64 block, find the symmetric half-vector d
            // minimizing SAD between A(-d) and C(+d); the midpoint is
            // B'(x) = 0.5*(A(x-d) + C(x+d)) -- boundaries land at their
            // interpolated POSITIONS instead of superimposing as a
            // double exposure. Same grades as LTWEEN for direct
            // comparison.
            if (lumaBIdx >= 0 && lumaAIdx >= 0 &&
                xferFrameIdx - lumaBIdx == 2 && lumaBIdx - lumaAIdx == 2 &&
                (int)lumaA.size() == (lastLine - firstLine) * width) {
                const double invIre = 1.0 / irescale;
                const int lb = lastLine - firstLine;
                // Current frame's certified luma as plane C.
                std::vector<float> lumaC(
                    (size_t)lb * width,
                    std::numeric_limits<float>::quiet_NaN());
                for (int line = firstLine; line < lastLine; ++line) {
                    const float *retr = carrierRetracted_line(line);
                    if (!retr) continue;
                    std::copy(retr, retr + width,
                              lumaC.data() +
                              (size_t)(line - firstLine) * width);
                }
                auto at = [&](const std::vector<float> &P, int y, int x)
                    -> double {
                    if (y < 0 || y >= lb || x < 0 || x >= width)
                        return std::numeric_limits<double>::quiet_NaN();
                    return (double)P[(size_t)y * width + x];
                };
                const int BH = 32, BW = 64;
                const int nby = (lb + BH - 1) / BH;
                const int nbx = (width + BW - 1) / BW;
                std::vector<int> flowY((size_t)nby * nbx, 0),
                                 flowX((size_t)nby * nbx, 0);
                double sumAD = 0.0; long nFlow = 0, nMoving = 0;
                for (int by = 0; by < nby; ++by)
                    for (int bx = 0; bx < nbx; ++bx) {
                        const int y0 = by * BH, x0 = bx * BW;
                        double best = 1e300, sad0 = 1e300;
                        int bdy = 0, bdx = 0;
                        for (int dy = -6; dy <= 6; ++dy)
                            for (int dx = -16; dx <= 16; ++dx) {
                                double sad = 0.0; int n = 0;
                                for (int y = y0; y < std::min(y0 + BH, lb);
                                     y += 2)
                                    for (int x = x0;
                                         x < std::min(x0 + BW, width);
                                         x += 2) {
                                        const double a =
                                            at(lumaA, y - dy, x - dx);
                                        const double c =
                                            at(lumaC, y + dy, x + dx);
                                        if (!std::isfinite(a) ||
                                            !std::isfinite(c)) continue;
                                        sad += std::fabs(a - c); n++;
                                    }
                                if (n < 64) continue;
                                const double m = sad / n;
                                if (dy == 0 && dx == 0) sad0 = m;
                                if (m < best) {
                                    best = m; bdy = dy; bdx = dx;
                                }
                            }
                        // Zero-motion preference: noise happily buys a
                        // random 1-px "improvement", and 1 px of spurious
                        // flow decorrelates the 3-px carrier band that
                        // the witness lives in. Motion must EARN its
                        // claim: keep d=0 unless the best displacement
                        // beats stillness by 20% and 0.5 IRE.
                        if (sad0 < 1e300 &&
                            best > sad0 * 0.8 - 0.5 * irescale) {
                            bdy = 0; bdx = 0;
                        }
                        flowY[(size_t)by * nbx + bx] = bdy;
                        flowX[(size_t)by * nbx + bx] = bdx;
                        sumAD += std::hypot((double)bdy, (double)bdx);
                        nFlow++;
                        if (bdy != 0 || bdx != 0) nMoving++;
                    }
                // Grade the flow-midpoint against held-out truth B.
                struct LR2 { double seF = 0, seB = 0, sxx = 0, syy = 0,
                                    sxy = 0, hitX = 0, hitY = 0,
                                    pX = 0, pY = 0;
                             long n = 0, nHit = 0, pN = 0, pTrue = 0; };
                const int rnx2 = (width + 127) / 128;
                std::vector<LR2> regs2((size_t)rnx2 * ((lb + 31) / 32));
                for (int y = 2; y < lb - 2; ++y) {
                    const int by = y / BH;
                    const float *pB = lumaB.data() + (size_t)y * width;
                    const size_t rb2 = (size_t)(y / 32) * rnx2;
                    for (int xi = 18; xi < width - 18; ++xi) {
                        const int bx = xi / BW;
                        const int dy = flowY[(size_t)by * nbx + bx];
                        const int dx = flowX[(size_t)by * nbx + bx];
                        auto mid = [&](int yy, int xx) {
                            const double a = at(lumaA, yy - dy, xx - dx);
                            const double c = at(lumaC, yy + dy, xx + dx);
                            return 0.5 * (a + c);
                        };
                        const double b = pB[xi];
                        const double tw = mid(y, xi);
                        if (!std::isfinite(b) || !std::isfinite(tw))
                            continue;
                        const double bpTw = mid(y, xi) -
                            0.5 * (mid(y, xi - 2) + mid(y, xi + 2));
                        const double bpB = (double)pB[xi] -
                            0.5 * ((double)pB[xi - 2] +
                                   (double)pB[xi + 2]);
                        if (!std::isfinite(bpTw) || !std::isfinite(bpB))
                            continue;
                        LR2 &R = regs2[rb2 + xi / 128];
                        R.seF += (tw - b) * (tw - b);
                        R.seB += (bpTw - bpB) * (bpTw - bpB);
                        R.sxx += bpTw * bpTw; R.syy += bpB * bpB;
                        R.sxy += bpTw * bpB;
                        if (std::fabs(bpB) >= 2.0 * irescale) {
                            R.nHit++;
                            R.hitX += std::fabs(bpTw);
                            R.hitY += std::fabs(bpB);
                        }
                        if (std::fabs(bpTw) >= 2.0 * irescale) {
                            R.pN++;
                            if (std::fabs(bpB) >= 1.0 * irescale)
                                R.pTrue++;
                            R.pX += std::fabs(bpTw);
                            R.pY += std::fabs(bpB);
                        }
                        R.n++;
                    }
                }
                double gF = 0, gB = 0; long gN = 0;
                std::vector<double> corrs, rmsB;
                double hitX = 0, hitY = 0; long hitN = 0;
                double pXg = 0, pYg = 0; long pNg = 0, pTrueg = 0;
                for (LR2 &R : regs2) {
                    if (R.n < 512) continue;
                    gF += R.seF; gB += R.seB; gN += R.n;
                    rmsB.push_back(std::sqrt(R.seB / R.n) * invIre);
                    if (R.sxx > 1e-9 && R.syy > 1e-9)
                        corrs.push_back(
                            R.sxy / std::sqrt(R.sxx * R.syy));
                    hitX += R.hitX; hitY += R.hitY; hitN += R.nHit;
                    pXg += R.pX; pYg += R.pY;
                    pNg += R.pN; pTrueg += R.pTrue;
                }
                if (gN > 0) {
                    auto pc = [](std::vector<double> &v, double p) {
                        if (v.empty()) return 0.0;
                        std::sort(v.begin(), v.end());
                        return v[std::min(v.size() - 1,
                                          (size_t)(p * v.size()))];
                    };
                    std::fprintf(stderr,
                        "[LTWEEN2 f=%ld flow] n=%ld rms full %.2f "
                        "bp %.2f IRE | bp-rms p10/50/90 "
                        "%.2f/%.2f/%.2f | bp-corr p10/50/90 "
                        "%.2f/%.2f/%.2f | hit n=%ld cover %.2f | "
                        "fire n=%ld precision %.2f magTrue %.2f | "
                        "flow mean|d| %.1f moving %.0f%%\n",
                        xferFrameIdx, gN,
                        std::sqrt(gF / gN) * invIre,
                        std::sqrt(gB / gN) * invIre,
                        pc(rmsB, 0.10), pc(rmsB, 0.50), pc(rmsB, 0.90),
                        pc(corrs, 0.10), pc(corrs, 0.50),
                        pc(corrs, 0.90),
                        hitN, hitY > 1e-9 ? hitX / hitY : 0.0,
                        pNg,
                        pNg > 0 ? (double)pTrueg / pNg : 0.0,
                        pXg > 1e-9 ? pYg / pXg : 0.0,
                        nFlow > 0 ? sumAD / nFlow : 0.0,
                        nFlow > 0 ? 100.0 * nMoving / nFlow : 0.0);
                }
            }

            // Rotate luma snapshots (A <- B <- current).
            lumaA.swap(lumaB); lumaAIdx = lumaBIdx;
            lumaB.assign((size_t)(lastLine - firstLine) * width,
                         std::numeric_limits<float>::quiet_NaN());
            lumaBIdx = xferFrameIdx;
            for (int line = firstLine; line < lastLine; ++line) {
                const float *retr = carrierRetracted_line(line);
                if (!retr) continue;
                float *dst = lumaB.data() +
                    (size_t)(line - firstLine) * width;
                for (int xi = 0; xi < width; ++xi)
                    dst[xi] = retr[xi];
            }

            // Snapshot this covered frame for the next comparison.
            xferPrev2.swap(xferPrev);
            xferPrev2Idx = xferPrevIdx;
            xferPrev.assign((size_t)(lastLine - firstLine) * width,
                            std::numeric_limits<float>::quiet_NaN());
            for (int line = firstLine; line < lastLine; ++line) {
                const float *exN = exactCarrierRow(line);
                if (!exN) continue;
                float *dst = xferPrev.data() +
                    (size_t)(line - firstLine) * width;
                for (int xi = 0; xi < width; ++xi)
                    dst[xi] = exN[left + xi];
            }
            xferPrevIdx = (int)xferFrameIdx;
        }
    }

    // Disposable dev-envelope census (LDCD_PROBE_CCDEV=1, engaged runs
    // only): one atomic line per frame -- the committed detector's
    // measured-disagreement distribution, to adjudicate the self-blinding
    // hypothesis (quiet frames: anchored tracks the same alias the
    // elected carrier carries, so dev sits under the cutoff with the
    // cross-colour still standing).
    static const bool ccDevOn = std::getenv("LDCD_PROBE_CCDEV") != nullptr;
    static const bool ccRefDevOn =
        std::getenv("LDCD_PROBE_CCREF") != nullptr;
    std::vector<float> ccDevSamples;
    long ccDevMaskN = 0, ccDevAnchN = 0;

    for (int line = firstLine; line < lastLine; ++line) {
        const double *src = clpbuffer[srcBuf].pixel[line];

        const CombCarrierGrammar *grammar = carrierGrammarLine(line);
        const bool grammarLocked = grammar && grammar->grammarLocked;

        const double bcos = grammarLocked ? grammar->burstCos : 1.0;
        const double bsin = grammarLocked ? grammar->burstSin : 0.0;

        double lutTi[4];
        double lutTq[4];

        if (grammarLocked) {
            for (int i = 0; i < 4; ++i) {
                lutTi[i] = finiteOrZero((double)grammar->demodLUTTi[i]);
                lutTq[i] = finiteOrZero((double)grammar->demodLUTTq[i]);
            }
        } else {
            fusedDemodLUT(bcos, bsin, spLUT_locked, cpLUT_locked, lutTi, lutTq);
            for (int i = 0; i < 4; ++i) {
                lutTi[i] = finiteOrZero(lutTi[i]);
                lutTq[i] = finiteOrZero(lutTq[i]);
            }
        }

        float  *tiRow       = demodTI_line(line);
        float  *tqRow       = demodTQ_line(line);
        float  *ti4Row      = demodTI4fsc_line(line);
        float  *tq4Row      = demodTQ4fsc_line(line);
        float  *prodIRow    = lockedProductI_line(line);
        float  *prodQRow    = lockedProductQ_line(line);
        float  *maskRawRow  = lockedCcMaskRaw_line(line);
        double *carrierComp = lockedCarrierComposite_line(line);

        const float *impRow = carrierImpurity_line(line);
        const float *sameRegionRow = regionSamePartner_line(line);
        // NOTE (renderer fact, 2026-07-31): a fact-family demod of the
        // RENDERED product planes was built here and measured inert --
        // filterIQLocked renders colour as demod(raw - Y), the exact
        // complement of emitted Y, and never reads those planes. The
        // covered-frame chroma win is carried by the Y facts plus the
        // suppression retirement (ccRetired, pass 2). Removed; do not
        // rebuild a product-side chroma path.
        //
        // CCR evidence upgrade (2026-08-02): the fact-backed carrier on
        // covered frames and the fact-corrected estimate on uncovered ones
        // are both accepted references, but their provenance remains
        // distinct at the handoff. Where either exists, disagreement with
        // locked 1D is MEASURED contamination in IRE; the notch edge read,
        // grammar proofs, and concert gate it replaced are deleted, not
        // layered under. Hard per-sample fact remains exactCarrierRow().
        const double *ccFactRow = factBackedCarrier_line(line);
        const double *ccEstimateRow =
            factCorrectedCarrierEstimate_line(line);
        const double *ccAnchRow = ccFactRow ? ccFactRow : ccEstimateRow;
        // Detector reference pair (2026-08-02, round 2): the OBSERVATION
        // side is the untouched locked 1D -- the safe retreat, which the
        // construction law guarantees no comb/echo signal ever enters --
        // never the elected carrier. Measured failure that forced this:
        // the tap fact-injection pulled the elected carrier toward the
        // anchored story, collapsing |elected - anchored| and blinding the
        // committed verdict while the alias-conforming cross-colour (which
        // combing cannot cancel) still stood: the strong frames' -53%
        // chroma fell to -4%. 1D-vs-anchored is immune to every future
        // comb change: alias and stolen luma live in the 1D and not in the
        // anchored model (fires); legal chroma lives identically in both
        // (bails).
        const double *ccObs1D = locked1DSource_line(line);

        if (prodIRow)    std::fill(prodIRow, prodIRow + width, 0.0f);
        if (prodQRow)    std::fill(prodQRow, prodQRow + width, 0.0f);
        if (carrierComp) std::fill(carrierComp, carrierComp + width, 0.0);

        for (int xi = 0; xi < width; ++xi) {
            const int h  = left + xi;
            const int ph = carrierSampleClass(line, h);

            // Plain locked demod at the native composite coordinate h.  Keep I
            // and Q as independent product streams: their common output
            // centroid is established later by the centered axis-specific
            // FIRs, not by averaging adjacent demod products here.  Such an
            // average is an extra half-sample filter on the demodulator and,
            // once remodulated, gives produceY a different carrier geometry.
            //
            // No line affine, local affine, or sliding-window carrier fit.
            // This function may suppress transfer, but it must not reshape the
            // carrier that produceY subtracts.
            const double ti = finiteOrZero(src[h] * lutTi[ph]);
            const double tq = finiteOrZero(src[h] * lutTq[ph]);

            double ti4 = 0.0;
            double tq4 = 0.0;
            lockedTo4fsc(ti, tq, bcos, bsin, ti4, tq4);

            ti4 = finiteOrZero(ti4);
            tq4 = finiteOrZero(tq4);

            if (tiRow)  tiRow[xi]  = (float)ti;
            if (tqRow)  tqRow[xi]  = (float)tq;
            if (ti4Row) ti4Row[xi] = (float)ti4;
            if (tq4Row) tq4Row[xi] = (float)tq4;

            // Critical geometry rule:
            // src is already the elected scalar carrier in physical composite
            // geometry.  It is the subtraction authority; the demodulated
            // products above are colour/evidence products, not a reason to
            // synthesize a second, numerically different scalar.  This also
            // makes the invariant exact when the locked basis or calibration
            // trim changes.
            if (carrierComp)
                carrierComp[xi] = finiteOrZero(src[h]);

            // Suppression policy below only runs when --cross-color-return
            // is engaged (the mask buffers exist); the default path pays
            // nothing.
            if (!maskRawRow) {
                const float prodI = (float)finiteOrZero(ti * giProduct);
                const float prodQ = (float)finiteOrZero(tq * gqProduct);
                if (prodIRow) prodIRow[xi] = prodI;
                if (prodQRow) prodQRow[xi] = prodQ;
                scratch_preI[xi] = prodI;
                scratch_preQ[xi] = prodQ;
                continue;
            }

            const double gA =
                impRow ? std::clamp((double)impRow[xi], 0.0, 1.0) : 0.0;

            // Named detector-to-policy conversion: a positively same-region
            // ±2 partner means this chroma continues vertically — a color
            // boundary, not cross-color — so the suppression stands down.
            const double regionKeep = sameRegionRow
                ? std::clamp((double)sameRegionRow[xi], 0.0, 1.0)
                : 0.0;

            // COMMITTED VERDICT (user law, 2026-08-02). Evidence confidence
            // cannot be expected to correlate with truth, so confidence
            // never sets a level: one cutoff below which the operation
            // bails, full commit above it. A committed operation returns a
            // clear result pro or con; a hedged one shelters the defect it
            // should expose. The evidence-ceiling cap and both analog
            // reads' level-setting retire together (research phase: data
            // over shelter). RETGRADE conviction that forced this: 77.5%
            // of the un-recovered theft budget sat where the old mask
            // idled below its own scaling (quiet/partial flat sites), and
            // gA is anti-informative in exactly that regime (CCREF bins
            // 0.2-0.4).
            //
            // Where the anchored plane exists the verdict is a MEASUREMENT:
            // the elected carrier's disagreement with the anticipation-
            // corrected carrier, in IRE. The gA aperture read remains the
            // witness only where no anchored plane exists (--dg-discard,
            // non-cadence sources: the chain-link fence regime it was
            // built for), applied as a binary predicate, never a scale.
            static const double ccCommitCutoffIRE = []{
                const char *s = std::getenv("LDCD_CC_CUTOFF");
                return s ? std::atof(s) : kCcCommitCutoffIRE;
            }();
            double lumaWeight = 0.0;
            double devIRE = -1.0;
            if (ccAnchRow && ccObs1D) {
                // |delta| oscillates at the carrier; its envelope is the
                // max over ±2 samples (1.25 cycles at 4fSC).
                const int am = std::max(0, xi - 2);
                const int ap = std::min(width - 1, xi + 2);
                double devPeak = 0.0;
                for (int k = am; k <= ap; ++k)
                    devPeak = std::max(devPeak,
                        std::fabs(finiteOrZero(ccObs1D[k]) -
                                  ccAnchRow[k]));
                devIRE = devPeak * invIreScale;
                if (devIRE >= ccCommitCutoffIRE)
                    lumaWeight = std::min(ccWeight, 1.0);
            } else if (gA >= kCcFallbackGALo &&
                       regionKeep < kCcFallbackRegionKeepMax) {
                lumaWeight = std::min(ccWeight, 1.0);
            }
            if (ccDevOn) {
                ++ccDevMaskN;
                if (devIRE >= 0.0) {
                    ++ccDevAnchN;
                    ccDevSamples.push_back((float)devIRE);
                }
            }
            // CCREF live-read plane (referee repair): the committed
            // verdict's own measurement, published so the truth referee
            // grades the read that actually decides.
            if (ccRefDevOn) {
                if (probeCcDevIRE_flat.size() !=
                    static_cast<size_t>(demodLines) * demodWidth)
                    probeCcDevIRE_flat.assign(
                        static_cast<size_t>(demodLines) * demodWidth,
                        std::numeric_limits<float>::quiet_NaN());
                probeCcDevIRE_flat[static_cast<size_t>(line) * demodWidth +
                                   xi] = (float)devIRE;
            }

            // The detector's own verdict, stashed for the off-lattice
            // audit BEFORE any fact overrides it.
            if (float *dv = ccDetectorVerdict_line(line))
                dv[xi] = (float)lumaWeight;

            // VERDICT IS THE FACT on covered samples: the false fraction of
            // the elected carrier is |C - ex| / |C| exactly. The detector is
            // not consulted where conservation already answered -- this is
            // what makes the turquoise/checker class impossible on the
            // lattice by construction. Denominator floored at 1 IRE (below
            // it the ratio is noise-on-noise; the fact verdict there is
            // simply "nothing worth returning").
            // NOTE (facts v4): per-sample fact SUPPRESSION was tried in two
            // forms (vector error, then amplitude excess) and both injected
            // luma-detail-rate structure into an envelope designed for
            // band-limited verdicts (+7.5% / +18.5% covered chroma
            // alternation). There is no separate chroma side to fix:
            // colour renders as demod(raw - Y), so chroma follows Y
            // (renderer fact, 2026-07-31). Facts reach the mask only
            // through the regional audit below; the Y side consumes the
            // fact as a VALUE in produceY.

            // Noise-pedestal restraint is inherited, not re-gated: the
            // commit cutoff sits at the pedestal floor (kCcCarrierFloorIRE
            // -- below it C and ex share the capture's noise, CCREF
            // 2026-07-30), so the committed verdict cannot fire on the
            // dark-noise class the old LDCD_CC_AMPGATE experiment guarded.

            // The verdict is NOT applied here.  Applied per-sample it carries
            // regionKeep's hard flips and gA's ring chatter at pixel pitch --
            // amplitude modulation that beats sidebands back into the chroma
            // passband and shreds both sides of a hue boundary (the residual
            // path documented this failure mode and band-limits; the coherent
            // path must too).  Pass 2 below smooths it into an envelope that
            // varies no faster than the chroma it gates, then applies.
            if (maskRawRow)
                maskRawRow[xi] = (float)lumaWeight;

            // Disposable CCR-path dump (env-gated): the committed verdict's
            // inputs at a chosen line/column range.
            static const int ccL  = []{ const char *s=std::getenv("LDCD_DUMP_CC_L");  return s?std::atoi(s):-1; }();
            static const int ccC0 = []{ const char *s=std::getenv("LDCD_DUMP_CC_C0"); return s?std::atoi(s):-1; }();
            static const int ccC1 = []{ const char *s=std::getenv("LDCD_DUMP_CC_C1"); return s?std::atoi(s):-1; }();
            if (line == ccL && (int)(h - left) >= ccC0 && (int)(h - left) <= ccC1) {
                std::fprintf(stderr,
                    "[CC] line=%d h=%d anch=%d devIRE=%.2f cutoff=%.2f "
                    "gA=%.2f regKeep=%.2f lumaW=%.2f drive=%s\n",
                    line, h, ccAnchRow ? 1 : 0, devIRE, ccCommitCutoffIRE,
                    gA, regionKeep, lumaWeight,
                    ccAnchRow ? "ANCH" : "FALLBACK");
            }

            const float prodI = (float)finiteOrZero(ti * giProduct);
            const float prodQ = (float)finiteOrZero(tq * gqProduct);

            if (prodIRow) prodIRow[xi] = prodI;
            if (prodQRow) prodQRow[xi] = prodQ;

            scratch_preI[xi] = prodI;
            scratch_preQ[xi] = prodQ;
        }
    }

    if (ccDevOn && ccDevMaskN > 0) {
        double p[5] = {0, 0, 0, 0, 0};
        double fired = 0.0;
        if (!ccDevSamples.empty()) {
            std::sort(ccDevSamples.begin(), ccDevSamples.end());
            const double q[5] = {0.25, 0.50, 0.75, 0.90, 0.99};
            for (int i = 0; i < 5; ++i)
                p[i] = ccDevSamples[std::min(
                    ccDevSamples.size() - 1,
                    (size_t)(q[i] * ccDevSamples.size()))];
            static const double cut = []{
                const char *s = std::getenv("LDCD_CC_CUTOFF");
                return s ? std::atof(s) : kCcCommitCutoffIRE;
            }();
            fired = (double)(ccDevSamples.end() -
                std::lower_bound(ccDevSamples.begin(),
                                 ccDevSamples.end(), (float)cut)) /
                (double)ccDevSamples.size();
        }
        std::fprintf(stderr,
            "[CCDEV] seq=%d covered=%d anchFrac=%.3f n=%ld "
            "p25=%.2f p50=%.2f p75=%.2f p90=%.2f p99=%.2f firedFrac=%.3f\n",
            (int)heldSeq1, frameHasExactCoverage() ? 1 : 0,
            (double)ccDevAnchN / (double)ccDevMaskN, ccDevAnchN,
            p[0], p[1], p[2], p[3], p[4], fired);
    }

    // Pass 2 (engaged when cross-color return is nonzero): band-limit the
    // suppression verdict into an envelope, then scale the published chroma
    // products.  In-field vertical mix (±2 lines, the same-parity partners
    // the verdict itself was judged against) then a lateral boxcar with a
    // radius of about one carrier cycle -- the same construction the
    // residual path uses, so suppression cannot alias in either renderer.
    if (ccWeight > 0.0 && !lockedCcMaskRaw_flat.empty() &&
        !lockedCcMask_flat.empty()) {
        constexpr int kCcMaskRadius = 4; // 9-tap, first null ~1.6 MHz

        // OFF-LATTICE FACT AUDIT (detectors-trade-for-truth, piece 3): per
        // region, the nearest covered facts grade the detector's claims --
        // auditW = (return truth honoured) / (return claimed) over fired
        // covered samples with real carrier, pooled from whichever of
        // {self, prevF, nextF} carries coverage. The lattice is one parity
        // of every other frame, so no estimate is ever more than a frame
        // from its audit. A weight from fact-agreement, never a coefficient
        // on a guess; regions without evidence stay neutral (the referee's
        // global verdict is that the return helps).
        static const bool ccFactsOn2 = []{
            const char *e = std::getenv("LDCD_CC_FACTS");
            return !(e && std::atoi(e) == 0);
        }();
        const int anx = (width + 127) / 128;
        const int any_ = (lastLine + 31) / 32;
        std::vector<double> auditW((size_t)anx * any_, 1.0);
        // SUPPRESSION-DUTY verdict (the regional court): same fact
        // grading, FAIL-CLOSED default 0. It answers one question --
        // where the detector's luma claim CONFLICTS with asserted legal
        // chroma, do the nearest facts endorse the detector? -- and a
        // region without evidence answers "no": the conflicted portion
        // of the mask abstains rather than act on an unadjudicated
        // claim. (The Y-return's auditW keeps its neutral default; that
        // duty has no opposing claim to adjudicate.)
        std::vector<double> auditSupp((size_t)anx * any_, 0.0);
        if (ccFactsOn2) {
            std::vector<double> claimed((size_t)anx * any_, 0.0),
                honoured((size_t)anx * any_, 0.0);
            // Sources: this frame if covered, else the previous covered
            // neighbour (its detector plane is complete; the NEXT frame's
            // splitIQlocked has not run yet, so it cannot serve). The
            // graded quantity is the DETECTOR plane, never the published
            // mask — on covered lines the mask is the fact and a fact
            // grading itself reads 1.0 by construction.
            const FrameBuffer *auditSrc =
                frameHasExactCoverage() ? this
                : (prevF && prevF->frameHasExactCoverage()) ? prevF
                : nullptr;
            if (auditSrc) {
                const FrameBuffer *fb = auditSrc;
                for (int line = firstLine; line < lastLine; ++line) {
                    const float *ex = fb->exactCarrierRow(line);
                    if (!ex) continue;
                    const double *cc =
                        fb->lockedCarrierComposite_line(line);
                    const float *dv = fb->ccDetectorVerdict_line(line);
                    if (!cc || !dv) continue;
                    const size_t rb = (size_t)(line / 32) * anx;
                    for (int xi = 0; xi < width; ++xi) {
                        const int h = left + xi;
                        if (!std::isfinite(ex[h])) continue;
                        const double C = cc[xi];
                        if (!std::isfinite(C) ||
                            std::fabs(C) < 2.0 * irescale) continue;
                        const double m = std::clamp((double)dv[xi], 0.0, 1.0);
                        if (m <= 0.02) continue;
                        const double claim = m * std::fabs(C);
                        const double leakAbs =
                            std::fabs(C - (double)ex[h]);
                        const size_t r = rb + xi / 128;
                        claimed[r] += claim;
                        honoured[r] += std::min(claim, leakAbs);
                    }
                }
            }
            for (size_t r = 0; r < auditW.size(); ++r)
                if (claimed[r] > 8.0 * irescale) {
                    auditW[r] = std::clamp(honoured[r] / claimed[r],
                                           0.0, 1.0);
                    auditSupp[r] = auditW[r];
                }
        }
        // Publish for produceY's Y-return consumption.
        ccAuditW_flat.assign(auditW.begin(), auditW.end());
        ccAuditNX = anx; ccAuditNY = any_;
        auto auditAt = [&](int line, int xi) {
            // Bilinear over region centers: the audit scales authority as a
            // smooth field, never a per-sample cut.
            const double ry = std::clamp((line - 16.0) / 32.0, 0.0,
                                         (double)(any_ - 1) - 1e-6);
            const double rx = std::clamp((xi - 64.0) / 128.0, 0.0,
                                         (double)(anx - 1) - 1e-6);
            const int r0 = (int)ry, c0 = (int)rx;
            const double wy = ry - r0, wx = rx - c0;
            const int r1 = std::min(r0 + 1, any_ - 1);
            const int c1 = std::min(c0 + 1, anx - 1);
            return (1 - wy) * ((1 - wx) * auditW[(size_t)r0 * anx + c0] +
                               wx * auditW[(size_t)r0 * anx + c1]) +
                   wy * ((1 - wx) * auditW[(size_t)r1 * anx + c0] +
                         wx * auditW[(size_t)r1 * anx + c1]);
        };
        auto auditSuppAt = [&](int line, int xi) {
            const double ry = std::clamp((line - 16.0) / 32.0, 0.0,
                                         (double)(any_ - 1) - 1e-6);
            const double rx = std::clamp((xi - 64.0) / 128.0, 0.0,
                                         (double)(anx - 1) - 1e-6);
            const int r0 = (int)ry, c0 = (int)rx;
            const double wy = ry - r0, wx = rx - c0;
            const int r1 = std::min(r0 + 1, any_ - 1);
            const int c1 = std::min(c0 + 1, anx - 1);
            return (1 - wy) * ((1 - wx) * auditSupp[(size_t)r0 * anx + c0] +
                               wx * auditSupp[(size_t)r0 * anx + c1]) +
                   wy * ((1 - wx) * auditSupp[(size_t)r1 * anx + c0] +
                         wx * auditSupp[(size_t)r1 * anx + c1]);
        };
        std::vector<double> vmix(width, 0.0);

        // ---- Vertical envelope stage (EXPERIMENT: LDCD_CC_VMIX) ----
        // The published mask is APPLIED on the frame raster, so its envelope
        // has to be smooth on the frame raster. The shipped mix reaches only
        // ±2 -- the same-parity partners the verdict was judged against --
        // which leaves the mask smoothed along the field axis and raw along
        // the axis it acts on. CCREF measured the consequence: mask
        // D2frame/D2field 1.18-1.27 against a |C| control at 0.73-0.82, i.e.
        // the nearer neighbour disagreeing MORE than the farther one.
        //
        //   0 field  ±2 only, [.25 .5 .25]                 (shipped baseline)
        //   1 frame  ±1 only, [.25 .5 .25]
        //   2 both   5-tap over l-2..l+2
        //   3 stage  ±2 in-field consensus, THEN ±1 across parity
        //
        // Mode 3 is the "move both together" shape: the verdict goes on being
        // formed from its own in-field evidence, and only the ACTION is made
        // parity-symmetric afterwards. This selector is experiment
        // scaffolding -- whatever wins becomes the unconditional kernel.
        static const int ccVMix = []{
            const char *s = std::getenv("LDCD_CC_VMIX");
            return s ? std::atoi(s) : 0;
        }();
        const size_t vpN = static_cast<size_t>(lastLine) * width;
        std::vector<double> ccVPlane(vpN, 0.0);
        {
            auto stage = [&](auto rowAt, int step, std::vector<double> &dst) {
                for (int l = firstLine; l < lastLine; ++l) {
                    const auto *c = rowAt(l);
                    if (!c) continue;
                    const auto *u = rowAt(l - step);
                    const auto *d = rowAt(l + step);
                    const double norm =
                        0.5 + (u ? 0.25 : 0.0) + (d ? 0.25 : 0.0);
                    double *o = dst.data() + static_cast<size_t>(l) * width;
                    for (int xi = 0; xi < width; ++xi)
                        o[xi] = (0.5 * c[xi] + (u ? 0.25 * u[xi] : 0.0) +
                                 (d ? 0.25 * d[xi] : 0.0)) / norm;
                }
            };
            auto rawRow = [&](int l) -> const float * {
                return (l >= firstLine && l < lastLine)
                    ? lockedCcMaskRaw_line(l) : nullptr;
            };
            if (ccVMix == 1) {
                stage(rawRow, 1, ccVPlane);
            } else if (ccVMix == 2) {
                for (int l = firstLine; l < lastLine; ++l) {
                    const float *c = rawRow(l);
                    if (!c) continue;
                    const float *u1 = rawRow(l - 1), *d1 = rawRow(l + 1);
                    const float *u2 = rawRow(l - 2), *d2 = rawRow(l + 2);
                    const double norm = 0.4 + (u1 ? 0.2 : 0.0)
                                            + (d1 ? 0.2 : 0.0)
                                            + (u2 ? 0.1 : 0.0)
                                            + (d2 ? 0.1 : 0.0);
                    double *o = ccVPlane.data() + static_cast<size_t>(l) * width;
                    for (int xi = 0; xi < width; ++xi)
                        o[xi] = ((0.4 * c[xi] +
                             (u1 ? 0.2 * u1[xi] : 0.0) +
                             (d1 ? 0.2 * d1[xi] : 0.0) +
                             (u2 ? 0.1 * u2[xi] : 0.0) +
                             (d2 ? 0.1 * d2[xi] : 0.0)) / norm);
                }
            } else if (ccVMix == 3) {
                std::vector<double> tmp(vpN, 0.0);
                stage(rawRow, 2, tmp);
                auto tmpRow = [&](int l) -> const double * {
                    return (l >= firstLine && l < lastLine)
                        ? tmp.data() + static_cast<size_t>(l) * width
                        : nullptr;
                };
                stage(tmpRow, 1, ccVPlane);
            } else {
                stage(rawRow, 2, ccVPlane);
            }
        }

        // SUPPRESSION RETIRES ON COVERED FRAMES (chroma-facts): the
        // rendered products are demodulated from the fact-family carrier,
        // so the false colour this envelope exists to suppress is excluded
        // at construction -- scaling those products by a fit-vague
        // envelope could only desaturate real chroma (the bikini ring).
        // Frame-uniform: the published mask reads zero on every line, so
        // produceY's estimator return stands down with it and neither
        // parity is treated alone. The detector verdict planes and the
        // regional audit above keep running: covered frames remain the
        // audit source that grades the uncovered frames' firings.
        // True coverage test, not the anchored-plane proxy: the plane now
        // publishes on uncovered frames too after two-sided refinement, and
        // the retirement premise -- false colour excluded at construction --
        // holds only where construction consumed same-frame FACTS.
        const bool ccRetired =
            chromaFactsOn && frameHasExactCoverage();

        for (int line = firstLine; line < lastLine; ++line) {
            const float *r0 = lockedCcMaskRaw_line(line);
            float *out = lockedCcMask_line(line);
            const float *exRow2 = exactCarrierRow(line);
            float *prodIRow = lockedProductI_line(line);
            float *prodQRow = lockedProductQ_line(line);
            if (!r0 || !out)
                continue;
            if (ccRetired) {
                std::fill(out, out + width, 0.0f);
                continue;
            }

            const AttributionEvidence *attributionRow =
                attributionEvidence_line(line);
            const float *impurityRow = carrierImpurity_line(line);
            const float *sameRegionRow = regionSamePartner_line(line);
            const lddecode::CarrierAnalysisRecord *analysisRow =
                carrierAnalysis_line(line);
            const lddecode::CarrierAnalysisRecord *analysisUpRow =
                (line - 2 >= firstLine) ? carrierAnalysis_line(line - 2)
                                        : nullptr;
            const lddecode::CarrierAnalysisRecord *analysisDnRow =
                (line + 2 < lastLine) ? carrierAnalysis_line(line + 2)
                                      : nullptr;

            // Delta-value return: either the fact-backed carrier or the
            // explicitly fact-corrected estimate, plus this line's demod LUT
            // (same recipe as pass 1), rebuilds a fired site's products.
            const double *factProdRow = factBackedCarrier_line(line);
            const double *estimateProdRow =
                factCorrectedCarrierEstimate_line(line);
            const double *anchProdRow =
                factProdRow ? factProdRow : estimateProdRow;
            double lutTi2[4] = {0, 0, 0, 0}, lutTq2[4] = {0, 0, 0, 0};
            const double giProduct2 = configuration.gi_product;
            const double gqProduct2 = configuration.gq_product;
            if (anchProdRow) {
                const CombCarrierGrammar *g2 = carrierGrammarLine(line);
                if (g2 && g2->grammarLocked) {
                    for (int i = 0; i < 4; ++i) {
                        lutTi2[i] = finiteOrZero((double)g2->demodLUTTi[i]);
                        lutTq2[i] = finiteOrZero((double)g2->demodLUTTq[i]);
                    }
                } else {
                    const double bc2 = g2 && g2->grammarLocked
                        ? g2->burstCos : 1.0;
                    const double bs2 = g2 && g2->grammarLocked
                        ? g2->burstSin : 0.0;
                    fusedDemodLUT(bc2, bs2, spLUT_locked, cpLUT_locked,
                                  lutTi2, lutTq2);
                    for (int i = 0; i < 4; ++i) {
                        lutTi2[i] = finiteOrZero(lutTi2[i]);
                        lutTq2[i] = finiteOrZero(lutTq2[i]);
                    }
                }
            }

            auto grammarPassAt = [&](int xi) {
                double legal = analysisRow
                    ? lddecode::carrierLegalProof(
                          (double)analysisRow[xi].carrierConformance,
                          (double)analysisRow[xi].conformanceSupportFraction)
                    : 0.0;
                legal = std::max(legal, std::max(
                    analysisUpRow
                        ? lddecode::carrierLegalProof(
                              (double)analysisUpRow[xi].carrierConformance,
                              (double)analysisUpRow[xi]
                                  .conformanceSupportFraction)
                        : 0.0,
                    analysisDnRow
                        ? lddecode::carrierLegalProof(
                              (double)analysisDnRow[xi].carrierConformance,
                              (double)analysisDnRow[xi]
                                  .conformanceSupportFraction)
                        : 0.0));
                return std::clamp(legal, 0.0, 1.0);
            };

            const double *vRow =
                ccVPlane.data() + static_cast<size_t>(line) * width;
            for (int xi = 0; xi < width; ++xi) {
                const double verticalMean = vRow[xi];
                const double impulse = attributionRow
                    ? std::clamp(
                          attributionRow[xi].facts.lumaImpulseRisk, 0.0, 1.0)
                    : 0.0;

                // A moving compact luma impulse is not expected at the same
                // column on its +/-2 partners.  Preserve its own measured raw
                // cross-colour evidence in proportion to the shared impulse
                // geometry instead of allowing the vertical envelope to halve
                // it.  This cannot invent suppression: it only raises the mix
                // toward r0 when r0 is already the stronger measurement.
                //
                // The one-sided max() encodes the FIELD kernel's geometry --
                // it assumes the neighbours never carry the impulse, so the
                // envelope can only ever be too LOW at an impulse site. Under
                // a frame-pitch kernel that premise inverts: a compact impulse
                // does land on the ±1 partners, verticalMean is raised at the
                // star, r0 - verticalMean goes negative, the max clamps the
                // correction away, and the mask sits high on the star itself
                // -- returning carrier-band energy into the impulse and
                // spreading it. Measured: star falloff +6.4% (attack shot),
                // +8.2% (cube), i.e. visibly less crisp.
                //
                // LDCD_CC_IMPULSE=1 holds the site at its OWN line's verdict
                // from either direction, so the term stops depending on which
                // neighbours happen to share the impulse. At impulse == 1 the
                // mask is exactly r0 under any kernel.
                // LDCD_CC_IMPULSE=2 removes the impulse term outright. This is
                // a MECHANISM PROBE, not a candidate: the term exists to keep
                // chroma suppression alive at stars, so dropping it trades
                // false colour for impulse crispness. It answers one question
                // -- is the impulse boost what the luma return is riding into
                // the star? -- and if so the fix is to split the mask's two
                // duties, not to delete the term.
                static const int ccImpulseMode = []{
                    const char *s = std::getenv("LDCD_CC_IMPULSE");
                    return s ? std::atoi(s) : 0;
                }();
                vmix[xi] =
                    ccImpulseMode == 2
                        ? verticalMean
                        : (ccImpulseMode == 1
                               ? verticalMean +
                                     impulse * ((double)r0[xi] - verticalMean)
                               : verticalMean +
                                     impulse * std::max(
                                         0.0,
                                         (double)r0[xi] - verticalMean));
            }

            // L4 aggregation preserves the authority of a narrow convicted
            // line without adding a second star/impulse detector. A one-pixel
            // verdict in this nine-tap aperture collapsed to 1/9 under a mean
            // and to 1/3 under RMS; L4 retains 9^(-1/4) ~= 0.58. The result is
            // still a smooth neighbourhood envelope, never exceeds the largest
            // measured verdict in its aperture, and leaves broad verdicts
            // unchanged.
            double sumFourthPowers = 0.0;
            int lo = 0, hi = -1;
            for (int xi = 0; xi < width; ++xi) {
                const int nlo = std::max(0, xi - kCcMaskRadius);
                const int nhi = std::min(width - 1, xi + kCcMaskRadius);
                while (hi < nhi) {
                    ++hi;
                    const double v2 = vmix[hi] * vmix[hi];
                    sumFourthPowers += v2 * v2;
                }
                while (lo < nlo) {
                    const double v2 = vmix[lo] * vmix[lo];
                    sumFourthPowers -= v2 * v2;
                    ++lo;
                }
                double m = std::clamp(
                    std::pow(std::max(0.0, sumFourthPowers) /
                                 (double)(nhi - nlo + 1),
                             0.25),
                    0.0, 1.0);

                // Keep the policy phase-invariant.  A complete carrier cycle
                // either retains grammar protection or admits the strong
                // aperture override as a unit; per-sample gating here would
                // amplitude-modulate chroma at fSC and recreate the very
                // transition checkerboard this envelope is meant to avoid.
                // Symmetric 5-sample cycle unit (integer centroid; the
                // old [xi-1, xi+2] unit gated half a sample right).
                double cycleLegal = 0.0;
                for (int k = -2; k <= 2; ++k)
                    cycleLegal = std::max(
                        cycleLegal,
                        grammarPassAt(std::clamp(xi + k, 0, width - 1)));

                // ADJUDICATION (replaces the strongApertureOverride,
                // 2026-07-31). Where the detector's luma claim conflicts
                // with grammar-certified legal chroma, neither side
                // overwrites: the conflicted portion's authority is the
                // REGIONAL FACT VERDICT (auditSupp -- the nearest covered
                // frames grading this detector's claims against exact
                // truth, fail-closed where unevidenced). The override let
                // an amplitude threshold cancel positive carrier-law
                // evidence -- gA reads 0.3-0.6 at genuine chroma edges,
                // so certified garment boundaries were suppressed as if
                // convicted (the bikini's loose uncovered boundaries).
                // Its legitimate client, alias-conforming cross-colour
                // that PASSES grammar (the fence/cube), is served by the
                // same court: there the facts endorse the detector and
                // full authority survives. Unconflicted claims (no
                // legality assertion) act as before.
                m *= (1.0 - cycleLegal) +
                     cycleLegal * auditSuppAt(line, xi);

                out[xi] = (float)m;
                // Delta-value return (2026-08-02), coherent renderer's half
                // of the one policy: a fired site's products come from the
                // ANCHORED (lawful) carrier, not from amplitude-killing the
                // elected one. Fallback (no anchored plane) keeps the kill.
                if (prodIRow && prodQRow) {
                    if (anchProdRow) {
                        const int ph = carrierSampleClass(line, left + xi);
                        const double a = anchProdRow[xi];
                        prodIRow[xi] = (float)((1.0 - m) * prodIRow[xi] +
                            m * a * lutTi2[ph] * giProduct2);
                        prodQRow[xi] = (float)((1.0 - m) * prodQRow[xi] +
                            m * a * lutTq2[ph] * gqProduct2);
                    } else {
                        prodIRow[xi] = (float)(prodIRow[xi] * (1.0 - m));
                        prodQRow[xi] = (float)(prodQRow[xi] * (1.0 - m));
                    }
                }
            }
        }
    }

    // ---- LEGAL-BAND CARRIER BETWEEN ANCHORS (user law, 2026-07-31) ----
    // Encoder physics: chroma was bandlimited to 1.3 MHz BEFORE modulation,
    // so demod-envelope energy beyond the legal band CANNOT be chroma -- it
    // is luma trapped in bandpass, by law, no detector consulted. On
    // uncovered frames ("between anchors") the subtraction authority is
    // redefined at construction: the composite carrier becomes the remod of
    // the legal-band I/Q envelopes, so produceY's comb Y = raw - carrier
    // keeps the illegal residue as luma automatically, and the rendered
    // products are limited to the same band so Y + remod(chroma) stays
    // conservative. Covered frames are untouched (facts own them; their
    // referees also read this plane as the comb's claim). The star law is
    // the pointwise extreme of this rule; this is the whole-spectrum form.
    // Escape LDCD_LEGAL_BAND=0. Identity probe LDCD_PROBE_LEGAL=1.
    static const bool legalBandOn = []{
        const char *e = std::getenv("LDCD_LEGAL_BAND");
        return !(e && std::atoi(e) == 0);
    }();
    // Cleared every frame regardless of path taken: this buffer is reused
    // across frames, and a covered frame (which skips the legal-band pass)
    // must not serve a stale residue plane to produceY on its comp lines.
    bandResidueY_flat.assign((size_t)demodLines * demodWidth, 0.0f);
    // True coverage test, not the anchored-plane proxy (see ccRetired).
    const bool legalFrameCovered =
        chromaFactsOn && frameHasExactCoverage();
    // Probe mode also enters on COVERED frames -- measurement only, no
    // writes: with exact truth in hand, grade whether band-limiting the
    // elected carrier moves it toward or away from ex ([LEGALREF]).
    static const bool legalProbePre =
        std::getenv("LDCD_PROBE_LEGAL") != nullptr;
    if (legalBandOn && (!legalFrameCovered || legalProbePre)) {
        // 21-tap legal-band LPF (Hann-windowed sinc, 1.3 MHz at 4fSC
        // sampling) -- same design family as filterIQLocked's FIRs.
        static std::once_flag lbInit;
        static std::vector<double> hLB;
        std::call_once(lbInit, [] {
            const double fsMHz = 14.31818, fny = fsMHz * 0.5;
            // Cutoff sweepable for the truth-referee (LDCD_LEGAL_MHZ);
            // production default 1.3 = the encoder's nominal limit.
            const double cutoff = [] {
                const char *e = std::getenv("LDCD_LEGAL_MHZ");
                return e ? std::atof(e) : 1.3;
            }();
            const double wn = cutoff / fny;
            const int N = 21, M = (N - 1) / 2;
            hLB.assign(N, 0.0);
            double sum = 0.0;
            for (int n = -M; n <= M; ++n) {
                const double x = (n == 0) ? (2.0 * wn)
                    : (std::sin(2.0 * M_PI * wn * n) / (M_PI * n));
                const double w = 0.5 * (1.0 + std::cos(M_PI * n / M));
                hLB[n + M] = x * w;
                sum += x * w;
            }
            for (double &v : hLB) v /= sum;
        });
        const int M = ((int)hLB.size() - 1) / 2;
        static const bool legalProbe =
            std::getenv("LDCD_PROBE_LEGAL") != nullptr;
        double prSum = 0, prRef = 0, prId = 0; long prN = 0;
        // Truth grading on covered frames (probe only): does the legal
        // band move the elected carrier toward exact? Split flat/detail
        // like DSREF (hLumaDelta >= 6 IRE).
        double refSrc[2] = {0, 0}, refLeg[2] = {0, 0};
        double refExOOB[2] = {0, 0};
        long refN[2] = {0, 0};
        std::vector<double> i4f(width), q4f(width);
        // Legal carrier per sample, stored for the application stage below
        // (claims are assembled regionally there; this loop only observes).
        const int lbLines = lastLine - firstLine;
        std::vector<float> legalPlane(
            (size_t)lbLines * width,
            std::numeric_limits<float>::quiet_NaN());
        for (int line = firstLine; line < lastLine; ++line) {
            const CombCarrierGrammar *grammar = carrierGrammarLine(line);
            if (!grammar || !grammar->grammarLocked) continue;
            const float *ti4Row = demodTI4fsc_line(line);
            const float *tq4Row = demodTQ4fsc_line(line);
            double *carrierComp = lockedCarrierComposite_line(line);
            if (!ti4Row || !tq4Row || !carrierComp) continue;
            const float *exRowRef = legalFrameCovered
                ? exactCarrierRow(line) : nullptr;
            const float *hdRowRef = legalFrameCovered
                ? lockedLumaHDeltaIRE_line(line) : nullptr;
            // Does TRUTH have out-of-band content? Demod ex with the same
            // line LUT, band-limit, remod, compare against itself.
            std::vector<double> exI4, exQ4;
            if (exRowRef) {
                exI4.assign(width, 0.0); exQ4.assign(width, 0.0);
                double lutTiR[4], lutTqR[4];
                for (int i = 0; i < 4; ++i) {
                    lutTiR[i] = (double)grammar->demodLUTTi[i];
                    lutTqR[i] = (double)grammar->demodLUTTq[i];
                }
                const double bc = grammar->burstCos, bs = grammar->burstSin;
                for (int xi = 0; xi < width; ++xi) {
                    const int h = left + xi;
                    const double v = std::isfinite((double)exRowRef[h])
                        ? (double)exRowRef[h] : 0.0;
                    const int ph = carrierSampleClass(line, h);
                    double i4, q4;
                    lockedTo4fsc(v * lutTiR[ph], v * lutTqR[ph],
                                 bc, bs, i4, q4);
                    exI4[xi] = i4; exQ4[xi] = q4;
                }
            }
            auto lpf = [&](const float *srcRow, std::vector<double> &dst) {
                for (int xi = 0; xi < width; ++xi) {
                    double acc = 0.0;
                    for (int k = -M; k <= M; ++k) {
                        const int xk =
                            std::clamp(xi + k, 0, width - 1);
                        acc += hLB[k + M] * (double)srcRow[xk];
                    }
                    dst[xi] = acc;
                }
            };
            lpf(ti4Row, i4f);
            lpf(tq4Row, q4f);
            for (int xi = 0; xi < width; ++xi) {
                const int h = left + xi;
                // x2: the remod's 0.5 assumes the per-sample product's
                // image terms fold back into the reconstruction. A
                // FILTERED envelope has no image terms, so its remod
                // carries half weight -- verified by the identity probe
                // (unfiltered round-trip exact at 0.03 IRE; filtered
                // reported |src|/2 before this factor).
                const double legalCar = 2.0 *
                    remodGrid4fscToComposite(line, h, i4f[xi], q4f[xi]);
                if (!std::isfinite(legalCar)) continue;
                if (legalProbe) {
                    prSum += std::fabs(carrierComp[xi] - legalCar);
                    prRef += std::fabs(carrierComp[xi]);
                    // Identity check: remod of the UNFILTERED products
                    // must reproduce src, or the split is lossy by
                    // construction.
                    const double idCar = remodGrid4fscToComposite(
                        line, h, (double)ti4Row[xi], (double)tq4Row[xi]);
                    if (std::isfinite(idCar))
                        prId += std::fabs(carrierComp[xi] - idCar);
                    prN++;
                }
                if (exRowRef && std::isfinite((double)exRowRef[h])) {
                    const int b = (hdRowRef &&
                                   hdRowRef[xi] >= 6.0f) ? 1 : 0;
                    refSrc[b] += std::fabs(
                        carrierComp[xi] - (double)exRowRef[h]);
                    refLeg[b] += std::fabs(
                        legalCar - (double)exRowRef[h]);
                    // ex vs its own legal-band self.
                    if (xi >= M && xi < width - M) {
                        double li = 0.0, lq = 0.0;
                        for (int k = -M; k <= M; ++k) {
                            li += hLB[k + M] * exI4[xi + k];
                            lq += hLB[k + M] * exQ4[xi + k];
                        }
                        const double exLegal = 2.0 *
                            remodGrid4fscToComposite(line, h, li, lq);
                        if (std::isfinite(exLegal))
                            refExOOB[b] += std::fabs(
                                (double)exRowRef[h] - exLegal);
                    }
                    refN[b]++;
                }
                legalPlane[(size_t)(line - firstLine) * width + xi] =
                    (float)legalCar;
            }
            // NOTE (renderer fact, 2026-07-31): a product-plane LPF stood
            // here and was verified bit-identical OFF vs ON -- the
            // renderer never reads the product planes (colour is
            // demod(raw - Y)). The band law reaches the render only
            // through carrierComp in the application stage below.
        }

        // ---- APPLICATION STAGE: the residue's luma claim (attribution) ----
        // The band law REVOKED the residue's chroma claim; it asserted
        // nothing. The residue moves to Y only by the luma claim the
        // temporal certified-luma witness grants (bandResidueLumaClaim,
        // attributiondefs.h): tween of the two covered neighbours'
        // certified retraction -- luma has no phase, so a tween error is a
        // soft blur, admissible as a WEIGHT where carrier transfer was
        // falsified outright. Claims are pooled per 32x128 region and
        // applied bilinearly (a smooth field, never a per-sample cut);
        // scene cuts and motion self-protect (mismatch -> claim -> 0).
        //
        // DELIVERY: published as bandResidueY (w·(src − legal) per
        // sample), consumed by produceY as a VALUE inside the returned-Y
        // candidate. NEVER written into carrierComp: the elected scalar
        // is one object consumed as both value and election evidence
        // (n.cc), and a second numerically different copy rendered as
        // dot fringing at garment edges (reach-error class, 2026-07-31).
        // Modes (LDCD_LEGAL_CARRIER): 0 off, 1 blanket (A/B; measured
        // +7.7% fsc-band Y = noise return), 2 witness-weighted (default).
        static const int legalCarrierMode = []{
            const char *e = std::getenv("LDCD_LEGAL_CARRIER");
            return e ? std::atoi(e) : 2;
        }();
        if (!legalFrameCovered && legalCarrierMode > 0) {
            const bool witnessOk = legalCarrierMode != 1 &&
                prevF && nextF &&
                prevF->frameHasExactCoverage() &&
                nextF->frameHasExactCoverage() &&
                prevF->carrierRetractedValid &&
                nextF->carrierRetractedValid;
            if (legalCarrierMode == 1) {
                for (int line = firstLine; line < lastLine; ++line) {
                    const double *carrierComp =
                        lockedCarrierComposite_line(line);
                    if (!carrierComp) continue;
                    const float *lp = legalPlane.data() +
                        (size_t)(line - firstLine) * width;
                    float *br = bandResidueY_flat.data() +
                        (size_t)line * demodWidth;
                    for (int xi = 0; xi < width; ++xi)
                        if (std::isfinite((double)lp[xi]))
                            br[xi] = (float)(carrierComp[xi] -
                                             (double)lp[xi]);
                }
            } else if (witnessOk) {
                const int rnx = (width + 127) / 128;
                const int rny = (lbLines + 31) / 32;
                std::vector<double> claimSum((size_t)rnx * rny, 0.0);
                std::vector<double> matchSum((size_t)rnx * rny, 0.0);
                // Asserted-chroma conflict (the residual garment-edge
                // fringing, 2026-07-31): at a co-located chroma+luma edge
                // the witness rightly sees luma structure, but the residue
                // there is the LEGAL chroma edge's own transient. A luma
                // claim firing against an asserted chroma claim is the
                // header's conflict case -- suppressed by carrierLegalProof
                // (same ±2-row construction as pass 2's grammarPassAt),
                // per-cycle at application to stay phase-invariant.
                std::vector<float> proofPlane(
                    (size_t)lbLines * width, 0.0f);
                for (int line = firstLine; line < lastLine; ++line) {
                    const double *carrierComp =
                        lockedCarrierComposite_line(line);
                    const float *rP = prevF->carrierRetracted_line(line);
                    const float *rN = nextF->carrierRetracted_line(line);
                    if (!carrierComp || !rP || !rN) continue;
                    const float *lp = legalPlane.data() +
                        (size_t)(line - firstLine) * width;
                    float *pf = proofPlane.data() +
                        (size_t)(line - firstLine) * width;
                    const lddecode::CarrierAnalysisRecord *anC =
                        carrierAnalysis_line(line);
                    const lddecode::CarrierAnalysisRecord *anU =
                        (line - 2 >= firstLine)
                            ? carrierAnalysis_line(line - 2) : nullptr;
                    const lddecode::CarrierAnalysisRecord *anD =
                        (line + 2 < lastLine)
                            ? carrierAnalysis_line(line + 2) : nullptr;
                    const size_t rb =
                        (size_t)((line - firstLine) / 32) * rnx;
                    for (int xi = 2; xi < width - 2; ++xi) {
                        if (!std::isfinite((double)lp[xi])) continue;
                        const double res = carrierComp[xi] - (double)lp[xi];
                        const double ares = std::fabs(res);
                        if (ares <= 0.0) continue;
                        double proof = anC
                            ? lddecode::carrierLegalProof(
                                  (double)anC[xi].carrierConformance,
                                  (double)anC[xi]
                                      .conformanceSupportFraction)
                            : 0.0;
                        if (anU) proof = std::max(proof,
                            lddecode::carrierLegalProof(
                                (double)anU[xi].carrierConformance,
                                (double)anU[xi]
                                    .conformanceSupportFraction));
                        if (anD) proof = std::max(proof,
                            lddecode::carrierLegalProof(
                                (double)anD[xi].carrierConformance,
                                (double)anD[xi]
                                    .conformanceSupportFraction));
                        pf[xi] = (float)proof;
                        // Witness fine structure: carrier-band bandpass
                        // (lag-2 [-.25,.5,-.25], the LTWEEN stencil) of
                        // EACH covered neighbour separately. TWIN
                        // CONFIRMATION, not a tween: an averaged witness
                        // carries both neighbours' edges at their own
                        // displaced positions, and under motion the claim
                        // fired at ghost geometry -- checker stamped in
                        // the NEXT frame's garment pattern (user-observed,
                        // 2026-07-31: "the checkered stripes... precisely
                        // align with those of the subsequent frame").
                        // Structure is evidence only where BOTH neighbours
                        // attest it independently -- same sign, each above
                        // the firing floor -- credited at the weaker
                        // magnitude. A moving edge has one attester per
                        // ghost position and dies; static detail has two.
                        auto bpOf = [&](const float *r, int x) {
                            return (double)r[x] -
                                   0.5 * ((double)r[x - 2] +
                                          (double)r[x + 2]);
                        };
                        const double bpP = bpOf(rP, xi);
                        const double bpN = bpOf(rN, xi);
                        // Amplitude condition (the CCREF law, third
                        // appearance): bp and residue are both noisy;
                        // min(noise, noise) reads as a match. Fire only
                        // above the 2 IRE floor -- the operating point at
                        // which the LTWEEN referee measured precision 0.70.
                        const double floorS = 2.0 * irescale;
                        const bool attested =
                            bpP * bpN > 0.0 &&
                            std::fabs(bpP) >= floorS &&
                            std::fabs(bpN) >= floorS;
                        lddecode::BandRevokedResidueEvidence ev;
                        ev.residueIRE = ares;
                        ev.witnessMatchIRE = attested
                            ? std::min(std::fabs(bpP), std::fabs(bpN))
                            : 0.0;
                        ev.witnessSupport = 1.0;
                        const size_t r = rb + xi / 128;
                        claimSum[r] += ares;
                        matchSum[r] +=
                            lddecode::bandResidueLumaClaim(ev) * ares *
                            (1.0 - proof);
                    }
                }
                std::vector<double> wReg((size_t)rnx * rny, 0.0);
                for (size_t r = 0; r < wReg.size(); ++r)
                    if (claimSum[r] > 4.0 * irescale)
                        wReg[r] = std::clamp(
                            matchSum[r] / claimSum[r], 0.0, 1.0);
                for (int line = firstLine; line < lastLine; ++line) {
                    const double *carrierComp =
                        lockedCarrierComposite_line(line);
                    if (!carrierComp) continue;
                    const float *lp = legalPlane.data() +
                        (size_t)(line - firstLine) * width;
                    float *br = bandResidueY_flat.data() +
                        (size_t)line * demodWidth;
                    for (int xi = 0; xi < width; ++xi) {
                        if (!std::isfinite((double)lp[xi])) continue;
                        const double ry = std::clamp(
                            ((line - firstLine) - 16.0) / 32.0, 0.0,
                            (double)(rny - 1) - 1e-6);
                        const double rx = std::clamp(
                            (xi - 64.0) / 128.0, 0.0,
                            (double)(rnx - 1) - 1e-6);
                        const int r0 = (int)ry, c0 = (int)rx;
                        const double wy = ry - r0, wx = rx - c0;
                        const int r1 = std::min(r0 + 1, rny - 1);
                        const int c1 = std::min(c0 + 1, rnx - 1);
                        auto W = [&](int rr, int cc) {
                            return wReg[(size_t)rr * rnx + cc];
                        };
                        const double w =
                            (1 - wy) * ((1 - wx) * W(r0, c0) +
                                        wx * W(r0, c1)) +
                            wy * ((1 - wx) * W(r1, c0) +
                                  wx * W(r1, c1));
                        if (w <= 0.0) continue;
                        // Phase-invariant conflict suppression: a complete
                        // carrier cycle acts as a unit (pass 2's rule --
                        // per-sample gating amplitude-modulates chroma at
                        // fSC and manufactures the very checker it fights).
                        const float *pfRow = proofPlane.data() +
                            (size_t)(line - firstLine) * width;
                        // Symmetric 5-sample cycle unit (see cycleLegal).
                        double cycleProof = 0.0;
                        for (int k = -2; k <= 2; ++k)
                            cycleProof = std::max(
                                cycleProof, (double)pfRow[
                                    std::clamp(xi + k, 0, width - 1)]);
                        const double wc = w * (1.0 - cycleProof);
                        if (wc <= 0.0) continue;
                        br[xi] = (float)(wc *
                            (carrierComp[xi] - (double)lp[xi]));
                    }
                }
                if (legalProbe) {
                    std::vector<double> ws;
                    for (double v : wReg) if (v > 0.0) ws.push_back(v);
                    std::sort(ws.begin(), ws.end());
                    auto pc = [&](double p) {
                        return ws.empty() ? 0.0
                            : ws[std::min(ws.size() - 1,
                                          (size_t)(p * ws.size()))];
                    };
                    std::fprintf(stderr,
                        "[WITNESS] active regions %zu/%zu "
                        "w p10/50/90 %.2f/%.2f/%.2f\n",
                        ws.size(), wReg.size(),
                        pc(0.10), pc(0.50), pc(0.90));
                }
            }
        }
        if (legalProbe && prN > 0)
            std::fprintf(stderr,
                "[LEGAL] n=%ld mean|src-legal| %.3f IRE (ref |src| %.3f, identity %.3f)\n",
                prN, prSum / prN / irescale, prRef / prN / irescale,
                prId / prN / irescale);
        if (legalProbe && (refN[0] + refN[1]) > 0)
            std::fprintf(stderr,
                "[LEGALREF covered] flat |src-ex| %.3f -> |legal-ex| %.3f "
                "exOOB %.3f (n=%ld) | detail %.3f -> %.3f exOOB %.3f "
                "(n=%ld)\n",
                refN[0] ? refSrc[0] / refN[0] / irescale : 0.0,
                refN[0] ? refLeg[0] / refN[0] / irescale : 0.0,
                refN[0] ? refExOOB[0] / refN[0] / irescale : 0.0,
                refN[0],
                refN[1] ? refSrc[1] / refN[1] / irescale : 0.0,
                refN[1] ? refLeg[1] / refN[1] / irescale : 0.0,
                refN[1] ? refExOOB[1] / refN[1] / irescale : 0.0,
                refN[1]);
    }

    // Final split authority for licensed stars.  Earlier passes may have
    // measured, returned, or legal-band-shaped this carrier; none of those
    // estimates may survive the exact-evidence license.  Keep the footprint
    // itself for complementary Y; filterIQLocked then applies its normal
    // aperture to the resulting raw - Y zero, with no special post-filter
    // shape.
    if (starFootprintBuilt &&
        starFootprint_flat.size() >=
            static_cast<size_t>(frameHeight) * demodWidth) {
        for (int line = firstLine; line < lastLine; ++line) {
            const std::uint8_t *star = starFootprint_flat.data() +
                static_cast<size_t>(line) * demodWidth;
            double *carrier = lockedCarrierComposite_line(line);
            float *prodI = lockedProductI_line(line);
            float *prodQ = lockedProductQ_line(line);
            float *maskRaw = lockedCcMaskRaw_line(line);
            float *mask = lockedCcMask_line(line);
            for (int xi = 0; xi < width; ++xi) {
                if (!star[xi]) continue;
                starSplitZeroSamples++;
                if (carrier) carrier[xi] = 0.0;
                if (prodI) prodI[xi] = 0.0f;
                if (prodQ) prodQ[xi] = 0.0f;
                // A zero split cannot subsequently request anchored-carrier
                // return through the cross-colour policy.
                if (mask && mask[xi] > 0.0f)
                    starAnchoredReturnBlocked++;
                if (maskRaw) maskRaw[xi] = 0.0f;
                if (mask) mask[xi] = 0.0f;
            }
        }
    }
}

void Comb::FrameBuffer::filterIQLocked()
{
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;

    constexpr bool   EXP_IQ_FIR_ENABLE = true;
    constexpr int    EXP_FIR_TAPS      = 21;
    constexpr double EXP_I_CUTOFF_MHZ  = 1.5;
    constexpr double EXP_Q_CUTOFF_MHZ  = 0.67;
    static_assert((EXP_FIR_TAPS & 1) == 1,
                  "locked I/Q FIRs must not add a fractional output delay");

    if (!EXP_IQ_FIR_ENABLE) return;

    static std::once_flag firInitFlag;
    static std::vector<double> hI, hQ;
    std::call_once(firInitFlag, [&](){
        auto designLPF = [&](double cutoffMHz)->std::vector<double> {
            const double fsMHz = 14.31818;
            const double fny   = fsMHz * 0.5;
            const double fc    = std::max(0.001, std::min(cutoffMHz, fny - 0.001));
            const double wn    = fc / fny;
            const int    N     = (EXP_FIR_TAPS | 1);
            const int    M     = (N - 1) / 2;
            std::vector<double> h(N, 0.0);
            double sum = 0.0;
            for (int n = -M; n <= M; ++n) {
                const double x = (n == 0) ? (2.0 * wn)
                    : (std::sin(2.0 * M_PI * wn * n) / (M_PI * n));
                const double w = 0.5 * (1.0 + std::cos(M_PI * n / (M + 1e-9)));
                const double v = x * w;
                h[n + M] = v;
                sum += v;
            }
            if (sum != 0.0) for (double &v : h) v /= sum;
            return h;
        };
        hI = designLPF(EXP_I_CUTOFF_MHZ);
        hQ = designLPF(EXP_Q_CUTOFF_MHZ);
    });

    const int NI = (int)hI.size(), NQ = (int)hQ.size();
    if (NI <= 0 || NQ <= 0) return;

    const int MI = (NI - 1) / 2;
    const int MQ = (NQ - 1) / 2;
    const double* tapsI = hI.data();
    const double* tapsQ = hQ.data();
    const int pad = std::max(MI, MQ);

    if ((int)scratch_preI.size() < width) scratch_preI.resize(width, 0.0);
    if ((int)scratch_preQ.size() < width) scratch_preQ.resize(width, 0.0);
    const int extWidth = width + 2 * pad;
    if ((int)scratch_preI_ext.size() < extWidth) scratch_preI_ext.resize(extWidth, 0.0);
    if ((int)scratch_preQ_ext.size() < extWidth) scratch_preQ_ext.resize(extWidth, 0.0);

    for (int line = firstLine; line < lastLine; ++line) {
        double* Irow = componentFrame->u(line);
        double* Qrow = componentFrame->v(line);

        // Colour is the carrier residual the elected luma left behind:
        //
        //     chroma = raw - Y
        //
        // One scalar, two outputs -- the same shape the bucket path has always
        // had (Y[h] = line[h] - val; I/Q = demod(val)).  Because Y is whatever
        // the election emitted, the published colour is its EXACT complement
        // and Y + chroma == raw holds at every pixel by construction.
        //
        // Consuming splitIQlocked's cached products instead lets the two
        // outputs drift: those products are demodulated from the elected comb
        // scalar, while produceY emits a band reassembly that only telescopes
        // back to raw - carrierComp when the comb plane wins the top band.
        // Measured on the beach, ~20% of pixels take a non-comb top band and
        // the emitted Y departs from the complement by 3.5 IRE RMS (max ~11),
        // so the colour on those pixels does not belong to the luma beside it.
        // That is exactly the population the election deviates on -- edges and
        // compact detail.
        //
        // No additional local DC follower here: that would give the colour a
        // different low-frequency convention from the luma it is derived from.
        const quint16* rawLine = rawbuffer.data() + line * videoParameters.fieldWidth;
        const double*  Yrow    = componentFrame->y(line);
        const CombCarrierGrammar *grammar = carrierGrammarLine(line);
        const bool grammarLocked = grammar && grammar->grammarLocked;
        const double bcos = grammarLocked ? grammar->burstCos : 1.0;
        const double bsin = grammarLocked ? grammar->burstSin : 0.0;
        double lutTi[4], lutTq[4];
        if (grammarLocked) {
            for (int i = 0; i < 4; ++i) {
                lutTi[i] = (double)grammar->demodLUTTi[i];
                lutTq[i] = (double)grammar->demodLUTTq[i];
            }
        } else {
            fusedDemodLUT(bcos, bsin, spLUT_locked, cpLUT_locked, lutTi, lutTq);
        }

        // Cross-colour suppression consumes the band-limited envelope
        // splitIQlocked already published (in-field +/-2 vertical mix + lateral
        // boxcar). One policy, computed once; this renderer only applies it, so
        // the suppression cannot alias here.
        //
        // Delta-value return (2026-08-02): a fired site may render from the
        // fact-backed carrier or, on uncovered frames, the separately named
        // fact-corrected estimate instead of being amplitude-killed. Without
        // either source the old suppression stands. Do not collapse these
        // accessors: exactCarrierRow() remains the hard-fact authority.
        const float *maskRow = lockedCcMask_line(line);
        const double *factRenderRow = factBackedCarrier_line(line);
        const double *estimateRenderRow =
            factCorrectedCarrierEstimate_line(line);
        const double *anchRow =
            factRenderRow ? factRenderRow : estimateRenderRow;
        const double giProduct = configuration.gi_product;
        const double gqProduct = configuration.gq_product;

        for (int i = 0; i < width; ++i) {
            const int h = left + i;
            const int ph = carrierSampleClass(line, h);
            const double chroma = (double)rawLine[h] - Yrow[h];
            const double m = maskRow
                ? std::clamp((double)maskRow[i], 0.0, 1.0)
                : 0.0;
            const double chromaEff = anchRow
                ? (1.0 - m) * chroma + m * anchRow[i]
                : (1.0 - m) * chroma;
            scratch_preI[i] = (chromaEff * lutTi[ph]) * giProduct;
            scratch_preQ[i] = (chromaEff * lutTq[ph]) * gqProduct;
        }

        // Integer-centred image cancellation (2026-08-01): the RENDER demod
        // was the last consumer left product-raw when 6f502ec registered
        // the measurement paths at the integer carrier sample.  A 4fsc
        // product carries a 2fSC image; the LPFs below reject it on flat
        // fields, but at sharp edges (and the certified family sharpens Y,
        // whose complement this is) the image's sidebands leak under the
        // stopband and render as edge-locked chroma asymmetry -- observed
        // as the smear left of the luma on covered frames.  Same symmetric
        // 3-tap the fix installed everywhere else: no fractional delay, no
        // new registration convention.  Escape LDCD_RENDER_IC=0.
        {
            static const bool renderIC = []{
                const char *e = std::getenv("LDCD_RENDER_IC");
                return !(e && std::atoi(e) == 0);
            }();
            if (renderIC && width >= 2) {
                if ((int)scratch_preI_ext.size() < width)
                    scratch_preI_ext.resize(width);
                if ((int)scratch_preQ_ext.size() < width)
                    scratch_preQ_ext.resize(width);
                std::copy(scratch_preI.begin(), scratch_preI.begin() + width,
                          scratch_preI_ext.begin());
                std::copy(scratch_preQ.begin(), scratch_preQ.begin() + width,
                          scratch_preQ_ext.begin());
                // centeredCarrierProduct3 carries the measurement paths'
                // x2 product-recovery gain (0.5a + b + 0.5c: baseband gain
                // 2, null at 2fSC).  The render chain has its own amplitude
                // convention, so the image cancellation must be UNITY at
                // baseband -- without the 0.5 the whole picture rendered
                // at double saturation (measured x1.93 flat on the GGV
                // colour bars; user-reported as excess saturation).
                for (int i = 0; i < width; ++i) {
                    const int im = std::max(0, i - 1);
                    const int ip = std::min(width - 1, i + 1);
                    scratch_preI[i] = 0.5 * centeredCarrierProduct3(
                        scratch_preI_ext[im], scratch_preI_ext[i],
                        scratch_preI_ext[ip]);
                    scratch_preQ[i] = 0.5 * centeredCarrierProduct3(
                        scratch_preQ_ext[im], scratch_preQ_ext[i],
                        scratch_preQ_ext[ip]);
                }
            }
        }

        constexpr double lockedPreRot =
            LOCKED_CHROMA_PREFILTER_ROT_DEG * M_PI / 180.0;
        const double preC = std::cos(lockedPreRot);
        const double preS = std::sin(lockedPreRot);
        for (int i = 0; i < width; ++i) {
            const double ti = scratch_preI[i];
            const double tq = scratch_preQ[i];
            scratch_preI[i] = ti * preC - tq * preS;
            scratch_preQ[i] = ti * preS + tq * preC;
        }

        double *preIext = scratch_preI_ext.data();
        double *preQext = scratch_preQ_ext.data();
        const double leftI = (width > 0) ? scratch_preI[0] : 0.0;
        const double leftQ = (width > 0) ? scratch_preQ[0] : 0.0;
        const double rightI = (width > 0) ? scratch_preI[width - 1] : 0.0;
        const double rightQ = (width > 0) ? scratch_preQ[width - 1] : 0.0;
        for (int i = 0; i < pad; ++i) { preIext[i] = leftI; preQext[i] = leftQ; }
        std::copy(scratch_preI.data(), scratch_preI.data() + width, preIext + pad);
        std::copy(scratch_preQ.data(), scratch_preQ.data() + width, preQext + pad);
        for (int i = 0; i < pad; ++i) {
            preIext[pad + width + i] = rightI;
            preQext[pad + width + i] = rightQ;
        }

        // ENCODER-MATCHED MODE (A/B: LDCD_RENDER_FEASIBLE=1). The tailored
        // FIRs below remove noise outside each component's legal trace, which
        // is their PURPOSE -- but a 21-tap kernel also spreads every chroma
        // transition (~18 px on Q), and that width is a knock-on, not design
        // (user, 2026-08-01). This mode instead applies the ENCODER'S OWN
        // 9-tap uvFilter (feasibleband.h, 1.3 MHz) to both axes: the same
        // primitive the carrier model already uses, so what it removes is
        // exactly what the encoder could not have modulated -- cleaning the
        // components mindfully of the NTSC spec rather than by a chosen
        // cutoff. Must be judged on BOTH axes at once: cross-colour admitted
        // (GGV monochrome bars) AND transition width.
        static const bool feasibleMode = []{
            const char *e = std::getenv("LDCD_RENDER_FEASIBLE");
            return e && std::atoi(e) != 0;
        }();
        if (feasibleMode) {
            std::vector<double> outI(width), outQ(width);
            lddecode::projectExpressibleChromaEnvelope(
                scratch_preI.data(), nullptr, width, outI.data());
            lddecode::projectExpressibleChromaEnvelope(
                scratch_preQ.data(), nullptr, width, outQ.data());
            for (int i = 0; i < width; ++i) {
                Irow[left + i] = outI[i];
                Qrow[left + i] = outQ[i];
            }
            continue;
        }

        // Both axes are evaluated by symmetric look-around at the same output
        // coordinate h.  The I and Q kernels intentionally have different
        // cutoffs (the oval correction), but the same odd support and centre,
        // so this final bandwidth filter adds no further relative displacement.
        // Registration belongs to the pre-comb products, not to this renderer.
        for (int i = 0; i < width; ++i) {
            double accI = tapsI[MI] * preIext[pad + i];
            double accQ = tapsQ[MQ] * preQext[pad + i];
            const double *cI = preIext + pad + i;
            const double *cQ = preQext + pad + i;
            for (int k = 1; k <= MI; ++k) {
                accI += tapsI[MI + k] * (cI[k] + cI[-k]);
            }
            for (int k = 1; k <= MQ; ++k) {
                accQ += tapsQ[MQ + k] * (cQ[k] + cQ[-k]);
            }
            const int h = left + i;
            Irow[h] = accI;
            Qrow[h] = accQ;
        }
    }
}

// ==== Exact-carrier anchor extraction + transfer-error probe ====
// LDCD_PROBE_ANCHOR=1, run -t 1. Measurement only.
//
// The (D-S)/2 exact channel certifies, on covered lines, where compact
// colour starts and stops and at what amplitude -- conservation facts,
// not estimates. Before any estimator may be TETHERED to those anchors,
// the tether needs measured margins: how far does a run's edge and
// amplitude drift per covered-line step (intra-field, the +-2-frame-line
// vertical coherence of one anchored field) and per anchored-frame step
// (cross-letter, A->C = two film frames of motion)? This probe extracts
// carrier runs from the exact envelope (hysteresis threshold, sub-pixel
// edges) and prints per-frame aggregates of both transfer errors. The
// p90s ARE the dilation margins the anchor consumers must grant.
namespace {

struct AnchorRun {
    double x0 = 0.0, x1 = 0.0;           // sub-pixel active-column edges
    double ampMean = 0.0, ampPeak = 0.0; // IRE
};

struct AnchorFrameStore {
    std::vector<std::vector<AnchorRun>> runsByLine;
    int parity = -1;
    long frameIdx = -1;
};
AnchorFrameStore g_anchorPrev;
long g_anchorFrameCounter = 0;

double anchorPct(std::vector<double> &v, double q)
{
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    const size_t k = std::min(v.size() - 1,
                              (size_t)(q * (double)(v.size() - 1) + 0.5));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

// Anchor = an EDGE (where compact colour starts or stops), so transfer
// is graded edge-to-edge: each edge matches the nearest same-polarity
// edge on the partner line within a window, or counts as unmatched (a
// topology event -- run merge/split, object end -- not a drift). Grading
// runs against runs conflated the two and blew up the tails.
void anchorMatchEdges(const std::vector<AnchorRun> &a,
                      const std::vector<AnchorRun> &b,
                      std::vector<double> &dEdge,
                      std::vector<double> &damp,
                      long &nUnmatched)
{
    constexpr double kMatchPx = 6.0;
    for (const AnchorRun &ra : a) {
        for (int pol = 0; pol < 2; ++pol) {
            const double ea = pol ? ra.x1 : ra.x0;
            double best = kMatchPx;
            const AnchorRun *bm = nullptr;
            for (const AnchorRun &rb : b) {
                const double eb = pol ? rb.x1 : rb.x0;
                const double d = std::fabs(eb - ea);
                if (d < best) { best = d; bm = &rb; }
            }
            if (bm) {
                dEdge.push_back(best);
                if (pol == 0)
                    damp.push_back(std::fabs(ra.ampMean - bm->ampMean));
            } else {
                ++nUnmatched;
            }
        }
    }
}

} // namespace

// ==== Anchor ceiling: regional carrier-amplitude bound from the exact
// channel (user direction, 2026-07-28: "I didn't want to interpolate the
// non-anchors, just constrain them with the bandwidth"). ====
//
// Per covered line the exact envelope certifies the carrier amplitude;
// pooling it over a lateral window (max-dilate, so the bound can only be
// generous) and padding by the measured transfer margins (amplitude p90
// ~1.5 IRE intra-frame) yields a REGIONAL ceiling: an amplitude no legal
// carrier in this neighbourhood exceeds. Uncovered lines inside an
// anchored frame take the max of the covered lines bracketing them
// (coverage is every other frame line there). Frames with no coverage
// publish no plane and every consumer stays inert -- cross-frame
// authority waits for the confirm-only machinery (45% of edges lapse
// across film frames; a bound must not pretend otherwise). Samples the
// error-corrector DENIED contribute nothing. The plane carries BOUNDS,
// never values -- nothing from it is ever rendered.
void Comb::FrameBuffer::buildAnchorCeiling()
{
    anchorCeiling_flat.clear();
    anchorCoveredLine.clear();
    if (exactCarrier_flat.empty()) return;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int fullWidth = videoParameters.fieldWidth;
    if (right - left <= 8 || firstLine >= lastLine || fullWidth <= 0) return;

    // Pooling radius sits just above the measured intra-frame edge-transfer
    // p90 (4 px): wide enough that anchor drift cannot fake a violation,
    // tight enough that a gray strut keeps its own low ceiling instead of
    // inheriting a neighbour's colour. Margins from the measured amplitude
    // drift (p90 1.5 IRE); floor covers the exact channel's noise envelope.
    constexpr int    kLatRadius     = 8;    // lateral pooling half-window
    constexpr double kMarginRel     = 0.10; // relative pad
    constexpr double kMarginAbsIRE  = 1.5;  // absolute pad
    constexpr double kFloorIRE      = 2.5;  // authority floor (noise env)

    const double inf = std::numeric_limits<double>::infinity();
    // Pooled per-line envelope rows, +inf where no authority.
    std::vector<std::vector<double>> pooled(lastLine);
    std::vector<double> env(fullWidth, std::numeric_limits<double>::quiet_NaN());
    bool anyCovered = false;

    for (int line = firstLine; line < lastLine; ++line) {
        const float *ex = exactCarrierRow(line);
        if (!ex) continue;
        int nFinite = 0;
        for (int h = left; h < right; ++h) {
            const float a = ex[h];
            const float b = (h + 1 < right) ? ex[h + 1] : a;
            if (std::isfinite(a) && std::isfinite(b)) {
                env[h] = std::hypot((double)a, (double)b) * invIreScale;
                ++nFinite;
            } else {
                env[h] = std::numeric_limits<double>::quiet_NaN();
            }
        }
        if (nFinite < (right - left) / 2) continue;
        anyCovered = true;

        auto &row = pooled[line];
        row.assign(fullWidth, inf);
        // Sliding max over +-kLatRadius, NaN-aware (denied samples give no
        // authority; a window with too little coverage stays +inf).
        for (int h = left; h < right; ++h) {
            double m = 0.0; int nv = 0;
            const int a = std::max(left, h - kLatRadius);
            const int b = std::min(right - 1, h + kLatRadius);
            for (int j = a; j <= b; ++j) {
                if (std::isfinite(env[j])) { m = std::max(m, env[j]); ++nv; }
            }
            if (nv >= (b - a + 1) / 2)
                row[h] = std::max(kFloorIRE,
                                  m * (1.0 + kMarginRel) + kMarginAbsIRE);
        }
    }

    if (!anyCovered) return;

    anchorCoveredLine.assign(lastLine, 0);
    for (int line = firstLine; line < lastLine; ++line)
        if (!pooled[line].empty()) anchorCoveredLine[line] = 1;

    anchorCeiling_flat.assign((size_t)lastLine * fullWidth,
                              std::numeric_limits<float>::infinity());
    for (int line = firstLine; line < lastLine; ++line) {
        float *out = anchorCeiling_flat.data() + (size_t)line * fullWidth;
        // This line's authority: its own pooled row, else the max of the
        // covered rows within +-2 (the bracketing covered lines).
        const std::vector<double> *src[3] = {nullptr, nullptr, nullptr};
        if (!pooled[line].empty()) src[0] = &pooled[line];
        else {
            if (line - 1 >= firstLine && !pooled[line - 1].empty())
                src[1] = &pooled[line - 1];
            if (line + 1 < lastLine && !pooled[line + 1].empty())
                src[2] = &pooled[line + 1];
            if (!src[1] && line - 2 >= firstLine && !pooled[line - 2].empty())
                src[1] = &pooled[line - 2];
            if (!src[2] && line + 2 < lastLine && !pooled[line + 2].empty())
                src[2] = &pooled[line + 2];
        }
        if (!src[0] && !src[1] && !src[2]) continue;
        for (int h = left; h < right; ++h) {
            double c = src[0] ? (*src[0])[h] : 0.0;
            if (!src[0]) {
                // An uncovered line is bounded only where EVERY bracketing
                // covered line grants authority; the bound is their max.
                double c1 = src[1] ? (*src[1])[h] : inf;
                double c2 = src[2] ? (*src[2])[h] : inf;
                if (!std::isfinite(c1) || !std::isfinite(c2)) { c = inf; }
                else c = std::max(c1, c2);
            }
            out[h] = (float)c;
        }
    }
}

void Comb::FrameBuffer::probeExactAnchors()
{
    static const bool anchorOn = []{
        const char *s = std::getenv("LDCD_PROBE_ANCHOR");
        return s && std::atoi(s) != 0;
    }();
    if (!anchorOn) return;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;
    if (width <= 8) return;

    constexpr double kOnIRE  = 3.0;  // run ignition
    constexpr double kOffIRE = 2.0;  // run extension (hysteresis)
    constexpr int    kMinRun = 4;

    std::vector<std::vector<AnchorRun>> runsByLine(lastLine);
    std::vector<double> env(width);
    int nCovered = 0, nRuns = 0, parity = -1;

    for (int line = firstLine; line < lastLine; ++line) {
        const float *ex = exactCarrierRow(line);
        if (!ex) continue;
        int nFinite = 0;
        for (int xi = 0; xi < width; ++xi) {
            const int h = left + xi;
            const float a = ex[h];
            const float b = (xi + 1 < width) ? ex[h + 1] : a;
            if (std::isfinite(a) && std::isfinite(b)) {
                env[xi] = std::hypot((double)a, (double)b) * invIreScale;
                ++nFinite;
            } else {
                env[xi] = 0.0;
            }
        }
        if (nFinite < width / 2) continue;
        ++nCovered;
        parity = line & 1;

        // 5-tap smooth of the envelope.
        std::vector<double> sm(width);
        for (int xi = 0; xi < width; ++xi) {
            double s = 0.0; int n = 0;
            for (int k = -2; k <= 2; ++k) {
                const int j = xi + k;
                if (j >= 0 && j < width) { s += env[j]; ++n; }
            }
            sm[xi] = s / std::max(1, n);
        }

        auto &runs = runsByLine[line];
        int xi = 0;
        while (xi < width) {
            if (sm[xi] < kOnIRE) { ++xi; continue; }
            int e = xi;
            while (e + 1 < width && sm[e + 1] >= kOffIRE) ++e;
            if (e - xi + 1 >= kMinRun) {
                AnchorRun r;
                // Sub-pixel edges: crossing of kOff below the first/above
                // the last sample when a neighbour exists.
                r.x0 = xi;
                if (xi > 0 && sm[xi] > sm[xi - 1] + 1e-9)
                    r.x0 = xi - 1 + (kOffIRE - sm[xi - 1]) /
                                        (sm[xi] - sm[xi - 1]);
                r.x1 = e;
                if (e + 1 < width && sm[e] > sm[e + 1] + 1e-9)
                    r.x1 = e + (sm[e] - kOffIRE) / (sm[e] - sm[e + 1]);
                double s = 0.0, p = 0.0;
                for (int j = xi; j <= e; ++j) {
                    s += sm[j]; p = std::max(p, sm[j]);
                }
                r.ampMean = s / (e - xi + 1);
                r.ampPeak = p;
                runs.push_back(r);
                ++nRuns;
            }
            xi = e + 1;
        }
    }

    const long fIdx = g_anchorFrameCounter++;
    if (nCovered == 0) {
        std::fprintf(stderr, "ANCHOR f=%ld no coverage\n", fIdx);
        return;
    }

    // Strong-run view: genuine compact colour, not threshold-skimming
    // fragments. Weak wide runs at the ignition threshold have noise-set
    // edges and mismatch freely; the tether will be calibrated on runs
    // whose amplitude proves an object.
    constexpr double kStrongIRE = 8.0;
    std::vector<std::vector<AnchorRun>> strongByLine(lastLine);
    int nStrong = 0;
    for (int line = firstLine; line < lastLine; ++line)
        for (const AnchorRun &r : runsByLine[line])
            if (r.ampPeak >= kStrongIRE) {
                strongByLine[line].push_back(r);
                ++nStrong;
            }

    // Intra-field transfer: covered line pairs two frame lines apart.
    auto intraStats = [&](const std::vector<std::vector<AnchorRun>> &byLine,
                          std::vector<double> &dE, std::vector<double> &damp,
                          long &unm) {
        for (int line = firstLine; line + 2 < lastLine; ++line) {
            if (byLine[line].empty() || byLine[line + 2].empty()) continue;
            anchorMatchEdges(byLine[line], byLine[line + 2], dE, damp, unm);
        }
    };
    std::vector<double> iDe, iDamp, sDe, sDamp;
    long iUnm = 0, sUnm = 0;
    intraStats(runsByLine, iDe, iDamp, iUnm);
    intraStats(strongByLine, sDe, sDamp, sUnm);
    auto c0 = iDe, ca = iDamp, s0 = sDe, sa = sDamp;
    std::fprintf(stderr,
        "ANCHOR f=%ld lines=%d runs=%d(strong %d) parity=%d  "
        "intra edges n=%zu unm=%.0f%% dE %.2f/%.2f damp %.2f/%.2f  "
        "STRONG n=%zu unm=%.0f%% dE %.2f/%.2f damp %.2f/%.2f",
        fIdx, nCovered, nRuns, nStrong, parity,
        iDe.size(), 100.0 * iUnm / std::max<size_t>(1, iDe.size() + iUnm),
        anchorPct(c0, 0.5), anchorPct(iDe, 0.9),
        anchorPct(ca, 0.5), anchorPct(iDamp, 0.9),
        sDe.size(), 100.0 * sUnm / std::max<size_t>(1, sDe.size() + sUnm),
        anchorPct(s0, 0.5), anchorPct(sDe, 0.9),
        anchorPct(sa, 0.5), anchorPct(sDamp, 0.9));

    // Cross-letter transfer vs the previous anchored frame (strong runs).
    if (g_anchorPrev.frameIdx >= 0 &&
        (int)g_anchorPrev.runsByLine.size() == lastLine) {
        std::vector<double> xDe, xDamp;
        long xUnm = 0;
        for (int line = firstLine; line < lastLine; ++line) {
            if (strongByLine[line].empty()) continue;
            // Same line if the parities match, else the covered neighbour.
            const int pl = (g_anchorPrev.parity == (line & 1))
                ? line
                : ((line + 1 < lastLine &&
                    !g_anchorPrev.runsByLine[line + 1].empty())
                       ? line + 1 : line - 1);
            if (pl < 0 || pl >= lastLine) continue;
            if (g_anchorPrev.runsByLine[pl].empty()) continue;
            anchorMatchEdges(strongByLine[line], g_anchorPrev.runsByLine[pl],
                             xDe, xDamp, xUnm);
        }
        auto d0 = xDe, da = xDamp;
        std::fprintf(stderr,
            "  cross(strong) n=%zu unm=%.0f%% dE %.2f/%.2f damp %.2f/%.2f",
            xDe.size(),
            100.0 * xUnm / std::max<size_t>(1, xDe.size() + xUnm),
            anchorPct(d0, 0.5), anchorPct(xDe, 0.9),
            anchorPct(da, 0.5), anchorPct(xDamp, 0.9));
    }
    std::fprintf(stderr, "\n");

    g_anchorPrev.runsByLine = std::move(strongByLine);
    g_anchorPrev.parity = parity;
    g_anchorPrev.frameIdx = fIdx;
}

// ---- Star law constants ----
// Signature thresholds are exactly the ones the certified referee graded
// (five-scene battery, 2026-07-30): FP ~0 on starfield and face closeups.
static constexpr int    kStarRegLines        = 32;
static constexpr int    kStarRegCols         = 128;
static constexpr double kStarPeakMinIRE      = 10.0;
static constexpr double kStarFlankAgreeIRE   = 6.0;
// The decoder's IRE convention already places black at 0.  7.5 IRE is the
// historical NTSC setup pedestal and is itself a visibly faded black here;
// retain it only as a generous noise/fade ceiling.  The former 20 IRE limit
// admitted low-luminance coloured regions (the Borg tractor beam) as space.
static constexpr double kStarBlackCeilIRE    = 7.5;
// A dark coloured background can satisfy the luma-black test.  Two raw
// composite cycles on each side of the compact event provide a phase-free
// continuity veto: a real carrier run persists through the star, while black
// space has no phasor to agree.  This is abstention only, never carrier source.
static constexpr double kStarCarrierRunMinIRE = 2.0;
static constexpr double kStarCarrierRunCosMin = 0.80;
// License thresholds: certified |ex| at signature sites. Measured noise
// floor 0.4-0.9 IRE where the law holds absolutely (battle 0.64, Kira 0.53);
// the compact-colour failure scenes pool 2.4-3.0 (temple, dark interior).
static constexpr double kStarLicenseRegionIRE = 1.5;
static constexpr double kStarLicenseFrameIRE  = 1.2;
static constexpr int    kStarLicenseRegionMinN = 3;
static constexpr int    kStarLicenseFrameMinN  = 20;

int Comb::FrameBuffer::certifiedOneDLevel()
{
    // DEFAULT ON, full family (promoted 2026-08-01, user directive on the
    // Emissary title bevel). The contract: certified defs CEDE to center —
    // their separation is already perfect and any comb can only mix truth
    // with a model — and comp pixels comb vigorously with the perfect legs
    // the defs give them. Measured on the title bevel: covered-frame
    // chroma alternation 31.4/29.4 -> 8.3/8.0 (-73%), visually near-clean.
    // The former promotion gate (beach 2026-07-30: anchors-vs-between-
    // frames contrast reads as discrediting alternation) was re-adjudicated
    // by the user: the uncovered frames are the same if not better — the
    // certified frames just show them up more — and that contrast does not
    // justify hiding the feature. Between-frame closure (cross-frame
    // transfer of the certified split on static content) remains the open
    // successor arc, pursued WITH the family live, not ahead of it.
    // LDCD_CERT_1D=0..3 remains as the A/B escape only.
    static const int level = []{
        const char *s = std::getenv("LDCD_CERT_1D");
        if (!s) return 3;
        const int v = std::atoi(s);
        return std::clamp(v, 0, 3);
    }();
    return level;
}

bool Comb::FrameBuffer::certifiedDefLine(int line) const
{
    if (certifiedOneDLevel() == 0) return false;
    if (line < 0 || line >= frameHeight) return false;
    if ((int)certifiedLineCache.size() != frameHeight)
        certifiedLineCache.assign(frameHeight, -1);
    qint8 &c = certifiedLineCache[line];
    if (c >= 0) return c != 0;
    c = 0;
    const float *ex = exactCarrierRow(line);
    if (ex) {
        const int left  = videoParameters.activeVideoStart;
        const int right = videoParameters.activeVideoEnd;
        for (int h = left; h < right; ++h)
            if (std::isfinite(ex[h])) { c = 1; break; }
    }
    return c != 0;
}

// Composite-shape star signature at h. Returns peak IRE over the flank mean
// (0 = no signature); flank outputs are raw-scale. The single source of the
// shape test: the actuator, the evidence build and the CCREF probe all call
// here, so the certified grading and the shipped behaviour cannot drift.
double Comb::FrameBuffer::starSignatureAt(const quint16 *rawLine, int h,
                                          double *flankOut,
                                          double *flankLOut,
                                          double *flankROut,
                                          bool *carrierRunVetoOut) const
{
    if (carrierRunVetoOut) *carrierRunVetoOut = false;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    if (h - 5 < left || h + 5 >= right) return 0.0;
    const double inv = 1.0 / irescale;
    auto rw = [&](int k) { return (double)rawLine[h + k]; };
    const double flankL = (rw(-5) + rw(-4) + rw(-3)) / 3.0;
    const double flankR = (rw(3) + rw(4) + rw(5)) / 3.0;
    const double flank  = 0.5 * (flankL + flankR);
    const double peakIRE = (rw(0) - flank) * inv;
    if (peakIRE < kStarPeakMinIRE) return 0.0;
    if (std::fabs(flankL - flankR) * inv > kStarFlankAgreeIRE) return 0.0;
    const double dev2IRE = std::max(std::fabs(rw(-2) - flankL),
                                    std::fabs(rw(2) - flankR)) * inv;
    if (dev2IRE > 0.4 * peakIRE) return 0.0;
    if ((flank - (double)videoParameters.black16bIre) * inv >
        kStarBlackCeilIRE) return 0.0;   // stars in black space, narrowly

    // Raw-domain carrier continuity.  Each side is eight samples (two whole
    // 4fSC cycles), outside both the five-sample star event and its black
    // return triplets.  Full cycles cancel DC exactly.  Absolute sample class
    // is used only to compare the two sides on this same line; no burst hue,
    // 1D estimate, demodulated colour, or output phase enters the decision.
    if (h - 13 >= left && h + 13 < right) {
        auto sidePhasor = [&](int k0, double &i, double &q) {
            static constexpr double c4[4] = { 1.0, 0.0, -1.0, 0.0 };
            static constexpr double s4[4] = { 0.0, 1.0, 0.0, -1.0 };
            i = 0.0;
            q = 0.0;
            for (int k = k0; k < k0 + 8; ++k) {
                const int ph = (h + k) & 3;
                const double v = rw(k);
                i += v * c4[ph];
                q += v * s4[ph];
            }
            // 2/N recovers the sinusoid's peak amplitude for N=8.
            i *= 0.25;
            q *= 0.25;
        };
        double li = 0.0, lq = 0.0, ri = 0.0, rq = 0.0;
        sidePhasor(-13, li, lq); // h-13 .. h-6
        sidePhasor(6, ri, rq);   // h+6  .. h+13
        const double la = std::hypot(li, lq);
        const double ra = std::hypot(ri, rq);
        if (la * inv >= kStarCarrierRunMinIRE &&
            ra * inv >= kStarCarrierRunMinIRE) {
            const double coherence =
                (li * ri + lq * rq) / std::max(1e-12, la * ra);
            if (coherence >= kStarCarrierRunCosMin) {
                if (carrierRunVetoOut) *carrierRunVetoOut = true;
                return 0.0;
            }
        }
    }
    if (flankOut)  *flankOut  = flank;
    if (flankLOut) *flankLOut = flankL;
    if (flankROut) *flankROut = flankR;
    return peakIRE;
}

// Certified star evidence, pooled regionally. Covered frames only produce
// evidence; every frame builds the (possibly empty) grid so neighbours can
// pool without shape checks. Idempotent per load.
void Comb::FrameBuffer::buildStarEvidence() const
{
    if (starEvidenceBuilt) return;
    starEvidenceBuilt = true;
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    if (right - left <= 0 || lastLine <= firstLine) return;
    starRegionsX = (right - left + kStarRegCols - 1) / kStarRegCols;
    starRegionsY = (lastLine + kStarRegLines - 1) / kStarRegLines;
    starEvidenceSum.assign((size_t)starRegionsX * starRegionsY, 0.0f);
    starEvidenceCnt.assign((size_t)starRegionsX * starRegionsY, 0);
    if (!frameHasExactCoverage()) return;
    const double inv = 1.0 / irescale;
    for (int line = firstLine; line < lastLine; ++line) {
        const float *ex = exactCarrierRow(line);
        if (!ex) continue;
        const quint16 *rawLine = rawbuffer.data() +
            (size_t)line * videoParameters.fieldWidth;
        for (int h = left + 5; h < right - 5; ++h) {
            if (!std::isfinite(ex[h])) continue;
            if (starSignatureAt(rawLine, h, nullptr) <= 0.0) continue;
            const size_t r = (size_t)(line / kStarRegLines) * starRegionsX +
                             (h - left) / kStarRegCols;
            starEvidenceSum[r] += (float)(std::fabs((double)ex[h]) * inv);
            starEvidenceCnt[r]++;
        }
    }
}

// Build the one authoritative star footprint before the elected carrier is
// published.  Exact evidence licenses the raw-domain signature; neither the
// comb result nor a demodulated hue participates in the decision.  The mask is
// subsequently consumed by both halves of the split: zero carrier here and
// raw luma in produceY().
void Comb::FrameBuffer::buildStarFootprint(const FrameBuffer *prevF,
                                           const FrameBuffer *nextF)
{
    if (starFootprintBuilt) return;
    starFootprintBuilt = true;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;

    starFootprint_flat.assign(
        static_cast<size_t>(frameHeight) * demodWidth, 0);
    starLicensedRegions = 0;
    starSignatureCenters = 0;
    starLicensedCenters = 0;
    starSubstitutedSamples = 0;
    starExtendedCenters = 0;
    starAddedSamples = 0;
    starAddedExactN = 0;
    starCarrierRunVetoes = 0;
    starSplitZeroSamples = 0;
    starAnchoredReturnBlocked = 0;
    starUsedPrevEvidence = false;
    starUsedNextEvidence = false;
    starAddedExactSum = 0.0;
    starAddedExactMax = 0.0;

    static const bool starFixOn = []{
        const char *s = std::getenv("LDCD_STAR_FIX");
        return !(s && std::atoi(s) == 0);
    }();
    static const bool starFixDump =
        std::getenv("LDCD_DUMP_STARFIX") != nullptr;
    if (width <= 0 || firstLine >= lastLine || demodWidth < width)
        return;

    // Evidence is a per-load cache, so it is const-safe to prepare on the
    // temporal witnesses while deciding the current frame's license.
    buildStarEvidence();
    if (prevF) prevF->buildStarEvidence();
    if (nextF) nextF->buildStarEvidence();

    std::vector<std::uint8_t> license;
    bool anyLicense = false;
    if (starFixOn && starRegionsX > 0) {
        const int nReg = starRegionsX * starRegionsY;
        std::vector<double> evSum(nReg, 0.0);
        std::vector<long> evCnt(nReg, 0);
        auto addFrom = [&](const FrameBuffer *f) -> bool {
            if (!f || !f->starEvidenceBuilt ||
                f->starRegionsX != starRegionsX ||
                f->starRegionsY != starRegionsY) return false;
            for (int r = 0; r < nReg; ++r) {
                evSum[r] += f->starEvidenceSum[r];
                evCnt[r] += f->starEvidenceCnt[r];
            }
            return true;
        };
        if (frameHasExactCoverage()) addFrom(this);
        else if (!hasSceneSplit) {
            // Edit flags mark the first field of the new scene.  Never let a
            // regional class license borrow from the opposite side of that
            // boundary: frame 165 of the Borg excerpt inherited tractor-beam
            // carrier from the preceding shot and lost every black-space star
            // for exactly one frame.  Within a shot the original two-sided
            // rule remains; at a boundary the one same-shot witness suffices.
            if (!isSceneStart)
                starUsedPrevEvidence = addFrom(prevF);
            if (!(nextF && nextF->isSceneStart))
                starUsedNextEvidence = addFrom(nextF);
        }

        double total = 0.0;
        long totalN = 0;
        for (int r = 0; r < nReg; ++r) {
            total += evSum[r];
            totalN += evCnt[r];
        }
        const bool frameLicense =
            totalN >= kStarLicenseFrameMinN &&
            total / totalN <= kStarLicenseFrameIRE;
        license.assign(nReg, 0);
        for (int r = 0; r < nReg; ++r) {
            const bool licensed =
                evCnt[r] >= kStarLicenseRegionMinN
                    ? evSum[r] / evCnt[r] <= kStarLicenseRegionIRE
                    : frameLicense;
            license[r] = licensed ? 1 : 0;
            if (licensed) {
                anyLicense = true;
                starLicensedRegions++;
            }
        }
    }

    if (!anyLicense && !starFixDump) return;

    for (int line = firstLine; line < lastLine; ++line) {
        const quint16 *rawLine = rawbuffer.data() +
            static_cast<size_t>(line) * videoParameters.fieldWidth;
        const float *exact = exactCarrierRow(line);
        std::uint8_t *footprint = starFootprint_flat.data() +
            static_cast<size_t>(line) * demodWidth;
        for (int h = left + 5; h < right - 5; ++h) {
            double flankL = 0.0, flankR = 0.0;
            bool carrierRunVeto = false;
            const double peakIRE = starSignatureAt(
                rawLine, h, nullptr, &flankL, &flankR, &carrierRunVeto);
            if (carrierRunVeto) {
                starCarrierRunVetoes++;
                continue;
            }
            if (peakIRE <= 0.0) continue;
            starSignatureCenters++;

            const size_t region =
                static_cast<size_t>(line / kStarRegLines) * starRegionsX +
                (h - left) / kStarRegCols;
            if (!anyLicense || region >= license.size() || !license[region])
                continue;
            starLicensedCenters++;

            int leftReturn = -3, rightReturn = 3;
            double leftErr = std::fabs(
                static_cast<double>(rawLine[h + leftReturn]) - flankL);
            double rightErr = std::fabs(
                static_cast<double>(rawLine[h + rightReturn]) - flankR);
            for (int k = -4; k >= -5; --k) {
                const double e = std::fabs(
                    static_cast<double>(rawLine[h + k]) - flankL);
                if (e < leftErr) { leftErr = e; leftReturn = k; }
            }
            for (int k = 4; k <= 5; ++k) {
                const double e = std::fabs(
                    static_cast<double>(rawLine[h + k]) - flankR);
                if (e < rightErr) { rightErr = e; rightReturn = k; }
            }
            const int runLo = leftReturn + 1;
            const int runHi = rightReturn - 1;
            if (runLo < -2 || runHi > 2) starExtendedCenters++;

            for (int k = runLo; k <= runHi; ++k) {
                const int xi = h + k - left;
                footprint[xi] = 1;
                starSubstitutedSamples++;
                if (k < -2 || k > 2) {
                    starAddedSamples++;
                    if (exact && std::isfinite(exact[h + k])) {
                        const double exIRE =
                            std::fabs(static_cast<double>(exact[h + k])) /
                            irescale;
                        starAddedExactSum += exIRE;
                        starAddedExactMax = std::max(starAddedExactMax,
                                                     exIRE);
                        starAddedExactN++;
                    }
                }
            }
        }
    }
}

void Comb::FrameBuffer::produceY(const FrameBuffer *prevF,
                                 const FrameBuffer *nextF)
{
    if (!configuration.phaseCompensation) return;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int fullWidth = videoParameters.fieldWidth;
    const int width     = right - left;
    if (width <= 0) return;

    const int srcBuf = std::clamp((int)configuration.dimensions - 1, 0, 2);
    const bool showMap = configuration.showMap;

    // splitIQlocked owns the star decision.  produceY only consumes its
    // published zero-carrier footprint as the complementary raw-luma value.
    static const bool starFixDump =
        std::getenv("LDCD_DUMP_STARFIX") != nullptr;
    buildStarFootprint(prevF, nextF); // idempotent safety for direct callers

    // Sync-tone stability probe: pool def lines' working-space certified IQ
    // per region and hand the frame to the tracker premise test.
    if (SyncProbe::on()) {
        const int nx = (width + SyncProbe::RC - 1) / SyncProbe::RC;
        const int ny = (lastLine + SyncProbe::RL - 1) / SyncProbe::RL;
        std::vector<double> sI((size_t)nx * ny, 0.0), sQ((size_t)nx * ny, 0.0);
        std::vector<long> sN((size_t)nx * ny, 0);
        double aI = 0, aQ = 0, bI = 0, bQ = 0;
        int defParity = -1, defIdx = 0;
        const bool cov = frameHasExactCoverage();
        if (cov) {
            for (int line = firstLine; line < lastLine; ++line) {
                if (!certifiedDefLine(line)) continue;
                if (defParity < 0) defParity = line & 1;
                const float *i4 = locked1DTI4fsc_line(line);
                const float *q4 = locked1DTQ4fsc_line(line);
                if (!i4 || !q4) continue;
                // The working space is hue-common across lines UP TO
                // lineFlip (BurstLockedSigned: polarity baked in). Pooling
                // without it measured a fictitious 180-degree alternation;
                // apply the convention.
                const CombCarrierGrammar *g = carrierGrammarLine(line);
                const double fl = (g && g->lineFlip < 0) ? -1.0 : 1.0;
                const size_t rb = (size_t)(line / SyncProbe::RL) * nx;
                double li = 0, lq = 0;
                for (int xi = 0; xi < width; ++xi) {
                    const size_t r = rb + xi / SyncProbe::RC;
                    sI[r] += fl * (double)i4[xi];
                    sQ[r] += fl * (double)q4[xi];
                    sN[r]++;
                    li += fl * (double)i4[xi]; lq += fl * (double)q4[xi];
                }
                if ((defIdx++ & 1) == 0) { aI += li; aQ += lq; }
                else                     { bI += li; bQ += lq; }
            }
        }
        // Snapshot the PREVIOUS covered frame before frame() replaces it —
        // the first build compared the frame against itself (act = 0 by
        // construction).
        SyncProbe::Snap syncPrevSnap = g_syncProbe.prev1;
        g_syncProbe.frame(cov, defParity, nx, ny, sI, sQ, sN,
                          aI, aQ, bI, bQ, irescale);

        // END-TO-END SHIPMENT CHECK (covered frames, -t 1, stateful in the
        // probe only): compare the ASSEMBLER-SHIPPED increment against the
        // increment the decoder itself measures between consecutive covered
        // frames in ITS OWN working space. This is the user's demanded
        // certainty that rotating the certifieds into 4fsc needs no
        // adjustment of the tone: corr +1 = conventions aligned, corr -1 =
        // handedness flip (fix at one defined boundary), drifting offset =
        // contract broken. Reported, never assumed.
        if (cov && syncPrevSnap.valid) {
            const QVector<float> &inc =
                !syncIncFirst.isEmpty() ? syncIncFirst : syncIncSecond;
            const int nReg = nx * ny;
            if (inc.size() >= 4 + nReg * 2) {
                // Global handedness/offset check: shipped global increment
                // vs the decoder's own global pooled rotation since the
                // previous covered frame. Accumulated across the run; the
                // sign of the running correlation IS the handedness verdict.
                static double gSS = 0, gAA = 0, gSA = 0;
                static long gN = 0;
                double aI2 = 0, aQ2 = 0, pI2 = 0, pQ2 = 0;
                for (int r = 0; r < nReg; ++r) {
                    if (syncPrevSnap.n[r] < 64 || sN[r] < 64) continue;
                    aI2 += sI[r]; aQ2 += sQ[r];
                    pI2 += syncPrevSnap.I[r];
                    pQ2 += syncPrevSnap.Q[r];
                }
                // RAW REPLICA (user: "DecoderPool and CadenceAssembler do
                // their work with composite before split1D... be sure we
                // are comparing apples to apples"). Reproduce the
                // assembler's measurement EXACTLY, decoder-side: certified
                // carrier (exactCarrierRow IS the shipped chat) + raw burst
                // samples of the same def lines, same k&3 basis, same
                // field-row signing (frame line L of the def parity is
                // field line L/2), same burst derotation. Three
                // trajectories then separate the question: ship vs rawAct
                // proves the CONTRACT; rawAct vs lutAct measures what the
                // decoder's own burst-locking adds on top.
                double rcI = 0, rcQ = 0, rbI = 0, rbQ = 0;
                {
                    static const int kcB[4] = { 1, 0, -1, 0 };
                    static const int ksB[4] = { 0, 1, 0, -1 };
                    const int bL = std::clamp(
                        videoParameters.colourBurstStart, 0,
                        videoParameters.fieldWidth);
                    const int bR = std::clamp(
                        videoParameters.colourBurstEnd, 0,
                        videoParameters.fieldWidth);
                    for (int line = firstLine; line < lastLine; ++line) {
                        if (!certifiedDefLine(line)) continue;
                        const double rs2 = ((line / 2) & 1) ? -1.0 : 1.0;
                        const float *ex = exactCarrierRow(line);
                        const quint16 *rw = rawbuffer.data() +
                            (size_t)line * videoParameters.fieldWidth;
                        if (ex)
                            for (int h = left; h < left + width; ++h) {
                                if (!std::isfinite(ex[h])) continue;
                                const int ph = h & 3;
                                rcI += rs2 * (double)ex[h] * kcB[ph];
                                rcQ += rs2 * (double)ex[h] * ksB[ph];
                            }
                        for (int h = bL; h < bR; ++h) {
                            const int ph = h & 3;
                            rbI += rs2 * (double)rw[h] * kcB[ph];
                            rbQ += rs2 * (double)rw[h] * ksB[ph];
                        }
                    }
                }
                static double prevRawI = 0, prevRawQ = 0;
                static bool prevRawValid = false;
                double rawAct = 0; bool rawActValid = false;
                {
                    const double bm = std::hypot(rbI, rbQ);
                    if (bm > 1e-9 && std::hypot(rcI, rcQ) > 1e-9) {
                        const double buI = rbI / bm, buQ = rbQ / bm;
                        const double rI = rcI * buI + rcQ * buQ;
                        const double rQ = rcQ * buI - rcI * buQ;
                        if (prevRawValid) {
                            rawAct = std::atan2(rQ * prevRawI - rI * prevRawQ,
                                                rI * prevRawI + rQ * prevRawQ);
                            rawActValid = true;
                        }
                        std::fprintf(stderr,
                            "[RAWABS] par=%d phase=%+.2f deg\n", defParity,
                            std::atan2(rQ, rI) * 180.0 / M_PI);
                        prevRawI = rI; prevRawQ = rQ; prevRawValid = true;
                    }
                }
                if (std::hypot(aI2, aQ2) > 1e-9 &&
                    std::hypot(pI2, pQ2) > 1e-9 && inc[1] > 0.2) {
                    const double act = std::atan2(aQ2 * pI2 - aI2 * pQ2,
                                                  aI2 * pI2 + aQ2 * pQ2);
                    // Rate comparison: shipped omega (rad/field) vs the
                    // actual anchor-to-anchor rate (5 fields per covered
                    // gap in 3:2). LUT-space actual is CONJUGATED by the
                    // measured handedness before comparison.
                    const double shipRate = inc[0];
                    const double actRawRate =
                        rawActValid ? rawAct / 5.0 : 0.0;
                    if (rawActValid) {
                        gSS += shipRate * shipRate;
                        gAA += actRawRate * actRawRate;
                        gSA += shipRate * actRawRate;
                        gN++;
                    }
                    const double gc = (gSS > 1e-12 && gAA > 1e-12)
                        ? gSA / std::sqrt(gSS * gAA) : 0.0;
                    std::fprintf(stderr,
                        "[SYNCSHIP-G] n=%ld shipRate=%+.3f rawRate=%+.3f "
                        "lutAct=%+.3f deg runCorr=%+.3f\n",
                        gN, shipRate * 180.0 / M_PI,
                        actRawRate * 180.0 / M_PI,
                        act * 180.0 / M_PI, gc);
                }
            }
            if (inc.size() >= 2 + nReg * 2) {
                double sPP = 0, sSS = 0, sPS = 0; int nCmp = 0;
                double sumShip = 0, sumAct = 0;
                for (int r = 0; r < nReg; ++r) {
                    const double conf = inc[4 + r * 2 + 1];
                    if (conf < 0.3) continue;
                    if (syncPrevSnap.n[r] < 64 || sN[r] < 64) continue;
                    const double a0 = std::hypot(syncPrevSnap.I[r],
                                                 syncPrevSnap.Q[r]) /
                                      syncPrevSnap.n[r];
                    const double a1 = std::hypot(sI[r], sQ[r]) / sN[r];
                    if (a0 < 2.0 * irescale || a1 < 2.0 * irescale) continue;
                    const double act = std::atan2(
                        sQ[r] * syncPrevSnap.I[r] -
                            sI[r] * syncPrevSnap.Q[r],
                        sI[r] * syncPrevSnap.I[r] +
                            sQ[r] * syncPrevSnap.Q[r]);
                    const double ship = inc[4 + r * 2];
                    sPP += ship * ship; sSS += act * act; sPS += ship * act;
                    sumShip += ship; sumAct += act; nCmp++;
                }
                if (nCmp > 8) {
                    const double corr =
                        (sPP > 1e-12 && sSS > 1e-12)
                            ? sPS / std::sqrt(sPP * sSS) : 0.0;
                    std::fprintf(stderr,
                        "[SYNCSHIP] n=%d corr=%+.3f shipMean=%+.3f "
                        "actMean=%+.3f deg\n",
                        nCmp, corr,
                        (sumShip / nCmp) * 180.0 / M_PI,
                        (sumAct / nCmp) * 180.0 / M_PI);
                }
            }
        }
    }
    // Off-grid leakage probe (measurement only; inert unless
    // LDCD_PROBE_OFFGRID). All published carriers exist by this point.
    probeOffGrid();

    // Exact-carrier anchor probe (measurement only; inert unless
    // LDCD_PROBE_ANCHOR). Extracts compact-colour runs from the exact
    // channel and grades their transfer error line-to-line and
    // frame-to-frame -- the margins the anchor tether needs.
    probeExactAnchors();

    // Anchor ceiling: pooled regional amplitude bound from the exact
    // channel. Cheap no-op on frames without coverage.
    buildAnchorCeiling();

    // ---- HF-election diagnostic probe (no output influence) ----
    // Column/line gated per-pixel dump of the produceY HF luma election:
    // candidate roster + planes, self/neighbor/decision anchors, the winner,
    // the regime vertical step, and whether the clean 1D candidate was
    // roster-excluded.  Enable with LDCD_PY_L0/L1 (frame line range) and
    // LDCD_PY_C0/C1 (active-picture column range, i.e. h - left).  Run -t 1.
    // ---- LDCD_PROBE_YCAND: per-CANDIDATE grading against certified luma ----
    // User question (2026-08-01): "we've established that there's still some
    // HF Y left on the table -- do you see low-hanging fruit in pulling more
    // HF Y from comb, witness or return?"  At covered samples Ltrue = raw -
    // exact is fact, so every candidate can be graded individually, and the
    // ORACLE bound (per-pixel best candidate) sizes exactly how much fidelity
    // the election is leaving behind.  Run HELD OUT (LDCD_CERT_1D=0).
    static const bool yCandOn = std::getenv("LDCD_PROBE_YCAND") != nullptr;
    double candErr[5] = {0,0,0,0,0};
    long   candN[5]   = {0,0,0,0,0};
    double candSlopeY[5] = {0,0,0,0,0}, candSlopeT[5] = {0,0,0,0,0};
    long   candSlopeN[5] = {0,0,0,0,0};
    double electErr = 0.0, oracleErr = 0.0; long electN = 0;
    long oracleWin[5] = {0,0,0,0,0};
    // ---- LDCD_PROBE_RETHULL: is the return hull clamping GOOD values? ----
    // User challenge (2026-08-02): "Retracted has such better stats than
    // comb that I would need a tiebreaker to award it to comb before I'd
    // take it from retracted. Do you have a confirmation that retracted is
    // in the wrong on those differences?" -- correct: the hull's segment
    // [comb, raw] has BOTH endpoints defined by comb, so it encodes "comb's
    // carrier claim is right", which is an assumption, not an impossible
    // (the carrier is signed; if comb's sign or magnitude is wrong the true
    // luma lies outside). Graded here against certified truth on def lines:
    // pre-hull vs post-hull distance to raw - exact, over all fired samples
    // and over the binding subset. Run in SHIPPING config with
    // LDCD_CHROMA_FACTS=0 (so return is not retired on covered frames).
    static const bool retHullProbeOn =
        std::getenv("LDCD_PROBE_RETHULL") != nullptr;
    double hfPickErr = 0, hfChanceSum = 0;
    long hfPickN = 0, hfPickHit = 0;
    double rhPre = 0, rhPost = 0, rhBindPre = 0, rhBindPost = 0;
    long rhN = 0, rhBindN = 0, rhBindDef = 0;
    // Return-delivery sub-ledger (see the split below).
    constexpr double kRetDeliverIRE = 0.5;
    double ractElect = 0, ractRet = 0, ractComb = 0, ractRetr = 0;
    double ractPos = 0, ractProxRet = 0, ractEvRet = 0;
    long ractProxN = 0;
    long ractN = 0, ractRetrN = 0, ractPosN = 0, ractRetBetter = 0;

    // ---- LDCD_PROBE_RETIMPACT: quantify return's impact on UNCOVERED
    // frames, where no certified fact exists to grade against (YCAND/YCERT
    // are structurally blind there). User (2026-08-01): "you have tested
    // uncovered frames with other tools and held back certified at other
    // times. Please quantify return's impact." Reference: the antRefLuma
    // chain (Pass 1.7's tween, validated TRUTH-GRADE on static content
    // earlier this session -- ANTGRADE box: 0.78 IRE mean|err|, corr +0.997
    // vs certified truth at the title bevel), read DIRECTLY here since
    // produceY and buildCarrierRetractionStage share the same FrameBuffer.
    // Direct-tier samples only (antRefLuma_flat[line] itself finite, not
    // the vertical-bracket fallback) -- the higher-confidence tier. This is
    // an APPROXIMATE reference, not fact: report it as such, and expect
    // motion to degrade it (the ghost lesson) rather than treat 0 error as
    // certification. Independent of LDCD_CERT_1D/_TONE (reads the chain
    // plane directly); meaningful only where antRefAge>=1 (chain reached
    // this frame) and NOT frameHasExactCoverage() (this frame is itself
    // uncovered -- the population the fact-based probes cannot reach).
    static const bool retImpactOn =
        std::getenv("LDCD_PROBE_RETIMPACT") != nullptr;
    double riErrSum = 0.0; long riN = 0;
    double riAbsYSum = 0.0;   // mean |Y - raw| at graded sites: content scale

    // ---- LDCD_PROBE_RETGRADE: grade return against CERTIFIED TRUTH, on
    // the OLD (uncertified) carrier -- user's design (2026-08-01): "if
    // return is run on the old dg-discard carrier of the def, we have our
    // measure against certified." True --dg-discard is an assembler-time
    // (ld-cinemap) flag, not available at decode time; the equivalent here
    // is LDCD_CERT_1D=0 (comb computes its own estimate instead of ceding
    // to exact) PLUS LDCD_CHROMA_FACTS=0 (the suppression mask does not
    // retire to zero on covered frames -- ccRetired depends on frame
    // coverage, not on CERT_1D, so without this second gate return is a
    // silent no-op here regardless of CERT_1D, as measured on the cube).
    // Both held out together put return in EXACTLY the regime it operates
    // in everywhere else (ordinary carrier, no certification) -- but on
    // def lines, where truth exists to grade it. Run this probe TWICE
    // (--cross-color-return 0 and 1.0) and pair by frame index externally:
    // eOff - eOn > 0 means return recovered luma 1D's bandpass stole;
    // < 0 is a false positive (return moved AWAY from truth).
    // dsExactRow must be read directly (certifiedDefLine() short-circuits
    // to false whenever certifiedOneDLevel()==0).
    static const bool retGradeOn =
        std::getenv("LDCD_PROBE_RETGRADE") != nullptr;
    double rgErrSum = 0.0, rgErrSqSum = 0.0, rgSignedSum = 0.0;
    long rgN = 0;
    // Residual-budget decomposition (user, 2026-08-01: "figure out what the
    // best thread to pull on might be to get at that 88%"): cells of
    // [mask state x site class]. maskBin: 0 = quiet (<0.05), 1 = partial
    // (0.05-0.5), 2 = strong (>0.5) -- with return OFF the mask plane is
    // all-zero (not built), so bins 1-2 populate only in the ON run; the
    // OFF/ON comparison per SITE CLASS gives recovery, the ON run's mask
    // bins split the residual into fired-but-undershot vs never-fired.
    // siteClass: 2 = impulse (lumaImpulseRisk>0.3, wins over edge),
    // 1 = edge (hLumaDelta >= 6 IRE, the cc edge-read soft knee),
    // 0 = flat/texture. Cell order [maskBin*3 + siteClass].
    double rgCellErr[9] = {0,0,0,0,0,0,0,0,0};
    long rgCellN[9] = {0,0,0,0,0,0,0,0,0};
    double rgCellBand[9] = {0,0,0,0,0,0,0,0,0};
    long rgCellBandN[9] = {0,0,0,0,0,0,0,0,0};

    // Election bypass: see the per-pixel site. Diagnostic A/B only.
    static const bool electBypass = []{
        const char *s = std::getenv("LDCD_ELECT_BYPASS");
        return s && std::atoi(s) != 0;
    }();
    static const int pyDiagL0 = []{ const char *s = std::getenv("LDCD_PY_L0"); return s ? std::atoi(s) : -1; }();
    static const int pyDiagL1 = []{ const char *s = std::getenv("LDCD_PY_L1"); return s ? std::atoi(s) : -1; }();
    static const int pyDiagC0 = []{ const char *s = std::getenv("LDCD_PY_C0"); return s ? std::atoi(s) : -1; }();
    static const int pyDiagC1 = []{ const char *s = std::getenv("LDCD_PY_C1"); return s ? std::atoi(s) : -1; }();
    const bool pyDiag = pyDiagL0 >= 0 && pyDiagC0 >= 0;
    const bool consDump = std::getenv("LDCD_DUMP_CONS") != nullptr;
    long long consN = 0, consP0N = 0, consOtherN = 0;
    double consSum = 0.0, consP0Sum = 0.0, consOtherSum = 0.0, consMax = 0.0;

    // --- Disposable emission-hull instrumentation (env-gated). Set
    // LDCD_DUMP_YHULL=1 to print, per frame, how many emitted pixels fail
    // |raw - Y| <= maxCarrierAmpSamples -- the legality test the election
    // applies to CANDIDATES but never re-applies to the emitted band splice
    // reconstructTop(top) = coarse + combMiddle + combPlatform + top.
    // Broken out by which top band won so a fix can be targeted. Set
    // LDCD_DUMP_YHULL=2 to also print each individual violation. Zero cost
    // when unset. Remove with the rethink.
    static const char *dumpHullEnv = std::getenv("LDCD_DUMP_YHULL");
    const bool dumpHull = dumpHullEnv != nullptr;
    const bool dumpHullVerbose = dumpHullEnv && std::atoi(dumpHullEnv) >= 2;
    long long hullTotal = 0, hullOver = 0;
    long long hullOverByPlane[7] = {0, 0, 0, 0, 0, 0, 0}; // 0,1,3,4 planes; 5=blend; 6=clamped
    double hullOverSumIRE = 0.0, hullMaxOvershootIRE = 0.0;
    double hullMismatchSumIRE = 0.0, hullMaxMismatchIRE = 0.0;

    // --- Disposable retracted-win schedule attribution (env-gated). Set
    // LDCD_DUMP_RETR=1. Answers the decisive question: when the retracted
    // plane wins the election, is its kept near-carrier energy schedule-LEGAL
    // (comb was right, retracted is passthrough failure) or schedule-ILLEGAL
    // (real grid luma comb destroyed, retracted correctly kept it)? Split by
    // the "bright passthrough" signature: retracted sits at raw while comb
    // removed a full carrier lobe. Zero cost when unset. Remove with the
    // rethink.
    const bool dumpRetr = std::getenv("LDCD_DUMP_RETR") != nullptr;
    // [conformance 0=Unresolved 1=Legal 2=Illegal][passthrough 0/1]
    long long retrWinBySchedule[3][2] = {{0,0},{0,0},{0,0}};
    long long retrWinTotal = 0, retrPassthroughTotal = 0;
    double retrLicenseSumOnPass = 0.0;
    // Very-bright retracted wins (the ones the eye spots): output above a high
    // IRE threshold. Split by schedule so we learn whether the visible bright
    // specks are legal carrier (passthrough failure) or illegal grid luma.
    long long retrBrightBySchedule[3] = {0, 0, 0};
    long long retrBrightTotal = 0;

    // produceY is a pure consumer. splitIQlocked() exports the selected comb
    // scalar at the same physical integer sample as raw, and that exact scalar
    // is the coherent subtraction authority. In 3D, clpbuffer[2] has already
    // been selected by split3D/getBestCandidate before splitIQlocked() runs, so
    // coherent 3D high-frequency luma follows the same raw-minus-carrier path;
    // no separate post-comb Y channel or temporal Y election is required.

    for (int line = firstLine; line < lastLine; ++line) {
        if (line >= demodLines) continue;

        const quint16 *rawLine = rawbuffer.data() + line * fullWidth;
        double *Y = componentFrame->y(line);
        const double *clpLine = clpbuffer[srcBuf].pixel[line];
        const double *carrierComp = lockedCarrierComposite_line(line);

        // The witness/retraction path remains an independent luma hypothesis.
        // Its existence is a fact about the completed analysis, not a chroma
        // rendering mode.
        const float *retractedRow =
            carrierRetractedValid ? carrierRetracted_line(line) : nullptr;
        const float *ccMaskRow = lockedCcMask_line(line);
        const float *ccMaskRawRow = lockedCcMaskRaw_line(line);
        // Plane 4 intentionally accepts either derived carrier family, but
        // keeps the fact-backed construction distinct from the uncovered
        // fact-corrected estimate. Hard fact itself is exactCarrierRow().
        const double *returnedFactCarrierRow =
            factBackedCarrier_line(line);
        const double *returnedEstimateCarrierRow =
            factCorrectedCarrierEstimate_line(line);
        const bool returnedHasDerivedCarrier =
            returnedFactCarrierRow || returnedEstimateCarrierRow;

        // Diagnostic view export (A/B only, same family as LD_RETRACTED_ADMIT).
        // LDCD_YVIEW publishes ONE election contestant AS Y so each candidate
        // can be rendered and compared in the identical pipeline geometry.
        // Pair with --chroma-gain 0 for a pure grayscale read.  This is an
        // inspection port, not a policy path: it bypasses the election
        // entirely and must never be a default.
        //
        //   mono      raw composite (carrier included) -- ground reference
        //   comb      plane 0, raw - lockedCarrierComposite
        //   retracted plane 1, the published retracted product (default:
        //             raw - w·carrierFit; LDCD_RETRACTED_SOURCE overrides)
        //   oned      plane 3, raw - locked1DSource
        //   returned  plane 4, combY + ccMask*(raw - combY)
        //   (unset)   the elected output
        //
        // The plane views route through planeY() below so they are EXACTLY
        // the values the election sees -- no second implementation to drift.
        static const int yViewMode = []{
            const char *s = std::getenv("LDCD_YVIEW");
            if (!s) return -1;
            // "retracted" and "returned" share the "ret" prefix; discriminate
            // on the fourth character rather than the first three.
            if (std::strcmp(s, "mono") == 0)      return 100;
            // The NATIVE inverse-encoder view: raw minus the per-line fit,
            // with no vertical promotion.  "retracted" below is raw minus
            // combedCarrier, i.e. the promoted, comb-conditioned product.
            if (std::strcmp(s, "native") == 0)    return 101;
            if (std::strcmp(s, "comb") == 0)      return 0;
            if (std::strcmp(s, "retracted") == 0) return 1;
            if (std::strcmp(s, "oned") == 0)      return 3;
            if (std::strcmp(s, "returned") == 0)  return 4;
            return -1;
        }();
        if (yViewMode == 100) {
            for (int h = left; h < right; ++h)
                Y[h] = (double)rawLine[h];
            continue;
        }
        if (yViewMode == 101) {
            const float *fitRow = carrierFit_line(line);
            for (int h = left; h < right; ++h)
                Y[h] = fitRow ? ((double)rawLine[h] - (double)fitRow[h - left])
                              : (double)rawLine[h];
            continue;
        }

        if (retractedRow || ccMaskRow) {
            // ================= HF luma election =================
            //
            // One selected coarse owns LF. Comb owns the middle band and is
            // the provisional top; a per-pixel election adjudicates only that
            // top among complete raw-carrier luma candidates. This replaces the prior
            // gate = max(gA, deltaGate, wGate); yOut = combY + ccReturn*gate*delta
            // blend, which produced a 2fSC checkerboard at chroma-amplitude
            // transitions: wGate (a DISTANCE between candidates, |combY - wY|)
            // and delta (= retractedY - combY) are both carrier-band, and the
            // witness lurch leaks Δchroma at carrier rate where chroma amplitude
            // is changing, so wGate*delta = fsc*fsc = DC + 2fSC. The cure is to
            // make every contributor a complete luma VALUE and let direct,
            // named measurements about each candidate (never a distance between
            // candidates) caution -- never override -- a selection anchored by
            // robust consensus and geometry.
            // Pattern: ld-disc-stacker neighbor modes (medoid center + inlier DQ
            // + capped quality penalty + neighbor selection) and the FVF
            // neighbor anchor (comb.cpp:1398).
            //
            // At most three contestants are active:
            //   0 coherentY  = raw - carrierComp     (phase-solved selected comb)
            //   1 retractedY = carrierRetracted      (raw - combedCarrier)
            //   3 1D         = raw - locked1DSource  (replaces comb only if DQ'd)
            //   4 returnedY  = combY + ccMask*(raw - combY), derived and
            //                  admitted only after the base population is fixed.
            // Coarse-platform selector. Default owns one cheap raster-aligned
            // four-sample coarse. --luma-witness unlocks the heavier centered,
            // lurch-sharpened platform. The selected platform is the sole LF
            // authority and defines the top-band coordinate. No second coarse
            // is mixed into reconstruction.
            const bool useSharpCoarse =
                configuration.lumaWitness && coarseSharpLevel() > 0.0 &&
                !lockedLumaSharp_flat.empty();
            auto coarseFloor_line = [&](int l) -> const double * {
                return useSharpCoarse ? lockedLumaSharp_line(l)
                                      : lockedLumaBaseY4_line(l);
            };
            const double *coarseRow =
                (lockedLumaCacheValid && demodWidth == width)
                    ? coarseFloor_line(line)
                    : nullptr;
            const double *oneDRow = locked1DSource_line(line); // may be null
            const std::uint8_t *bandRow = chromaBoundaryBand_line(line);
            const float *dsExactRow = exactCarrierRow(line);
            const float *anchorRow = anchorCeilingRow(line);
            const float *dsHDeltaRow = lockedLumaHDeltaIRE_line(line);
            // CCREF referee inputs: the CC detector's own components, read
            // back here so each can be graded against the measured leak. The
            // ±2 analysis rows reproduce the grammar-pass max the detector
            // itself takes (splitIQlocked), so we grade the real read and not
            // a centre-row approximation of it.
            const bool ccRefOn = CcRefProbe::on();
            const float *ccImpurityRow = ccRefOn ? carrierImpurity_line(line)
                                                 : nullptr;
            const float *ccRegionRow = ccRefOn ? regionSamePartner_line(line)
                                               : nullptr;
            const lddecode::CarrierAnalysisRecord *ccAnalysisUpRow =
                (ccRefOn && line - 2 >= firstLine) ? carrierAnalysis_line(line - 2)
                                                   : nullptr;
            const lddecode::CarrierAnalysisRecord *ccAnalysisDnRow =
                (ccRefOn && line + 2 < lastLine) ? carrierAnalysis_line(line + 2)
                                                 : nullptr;
            const lddecode::CarrierAnalysisRecord *analysisRow =
                carrierAnalysis_line(line);
            const float *alienRow = regionAlienPartner_line(line);
            // Star/impulse facts (single producer: compactLumaExcursionEvidence)
            // for the impulse-seniority bias in the election scoring.
            const AttributionEvidence *attribRow = attributionEvidence_line(line);
            // Retracted-admission mode (isolation switch for the cube/beach
            // A/B).  Default: conflicted fits admit retracted only with
            // neighbour spatial support.  LD_RETRACTED_ADMIT=trust restores
            // the residualTrust-only hard gate; =all seats unconditionally
            // (pre-2026-07-02 behaviour).
            static const int retractedAdmitMode = []{
                const char *s = std::getenv("LD_RETRACTED_ADMIT");
                if (!s) return 0;              // 0 = spatial (default)
                if (s[0] == 't') return 1;     // trust-only hard gate
                if (s[0] == 'a') return 2;     // admit all
                return 0;
            }();
            const bool retractedAdmitSpatial = (retractedAdmitMode == 0);
            const bool retractedAdmitAll = (retractedAdmitMode == 2);

            // Structural carrier-amplitude ceiling (samples): I/Q are bounded
            // sinusoids, so apparent carrier beyond this must be luma. Used as
            // the feasibility DQ. Same bound buildCarrierRetracted clamps with.
            // Measured limits (bars/beach) -- see maxCarrierAmpIREFromScale.
            const CombCarrierGrammar *grammarLine =
                carrierGrammarLine(line);
            const double maxCarrierAmpSamples =
                maxCarrierAmpIREFromScale(
                    grammarLine ? grammarLine->carrierScale : 0.0) * irescale;

            // Checks the emitted band splice against the same hull the
            // election applies to candidates -- candidate feasibility says
            // nothing about the reconstructTop() value actually written to
            // Y[h], since only plane 0 telescopes back to combY exactly.
            auto tallyHull = [&](double rawSample, double emittedY, int plane, int h,
                                 double fourMismatch, double planeComplete) {
                if (!dumpHull) return;
                ++hullTotal;
                const double c = rawSample - emittedY;
                if (c > maxCarrierAmpSamples || c < -maxCarrierAmpSamples) {
                    ++hullOver;
                    const int p = std::clamp(plane, 0, 6);
                    ++hullOverByPlane[p];
                    const double overIRE =
                        (std::fabs(c) - maxCarrierAmpSamples) * invIreScale;
                    hullOverSumIRE += overIRE;
                    hullMaxOvershootIRE = std::max(hullMaxOvershootIRE, overIRE);
                    hullMismatchSumIRE += std::fabs(fourMismatch) * invIreScale;
                    hullMaxMismatchIRE = std::max(hullMaxMismatchIRE,
                                                  std::fabs(fourMismatch) * invIreScale);
                    if (dumpHullVerbose)
                        std::fprintf(stderr,
                            "[YHULL-EV] line=%d h=%d plane=%d raw=%.2f Y=%.2f "
                            "c=%.2f maxAmp=%.2f overIRE=%.2f "
                            "four0-fourP=%.2f planeComplete=%.2f "
                            "cIfOwnBands=%.2f\n",
                            line, h, plane, rawSample * invIreScale,
                            emittedY * invIreScale, c * invIreScale,
                            maxCarrierAmpSamples * invIreScale, overIRE,
                            fourMismatch * invIreScale,
                            planeComplete * invIreScale,
                            (rawSample - planeComplete) * invIreScale);
                }
            };

            // Retracted-win schedule attribution. Called at each emission with
            // the winning plane; only plane 1 (retracted) is tallied. The
            // "bright passthrough" signature is retracted sitting at raw while
            // comb removed a full carrier lobe -- the failure the reframe
            // targets. The schedule enum then says whether that kept energy is
            // legal carrier (comb right, retracted wrong) or illegal grid luma
            // (retracted right).
            const double passRetrTolSamp = 6.0 * irescale;
            const double passLobeSamp    = 12.0 * irescale;
            auto tallyRetr = [&](double rawSample, int winnerPlane,
                                 double combVal, double retrVal, int h,
                                 const lddecode::CarrierAnalysisRecord *rec) {
                if (!dumpRetr || winnerPlane != 1) return;
                ++retrWinTotal;
                const bool passthrough =
                    std::fabs(retrVal - rawSample) < passRetrTolSamp &&
                    std::fabs(combVal - rawSample) > passLobeSamp;
                if (passthrough) ++retrPassthroughTotal;
                int sc = 0; // Unresolved
                double lic = 0.0;
                if (rec) {
                    switch (rec->scheduleConformance) {
                        case lddecode::CarrierScheduleConformance::LegalCarrier:
                            sc = 1; break;
                        case lddecode::CarrierScheduleConformance::ScheduleIllegal:
                            sc = 2; break;
                        default: sc = 0; break;
                    }
                    lic = lddecode::carrierScheduleLicense(
                        (double)rec->carrierConformance,
                        (double)rec->conformanceSupportFraction,
                        (double)rec->conformanceContradictionFraction);
                }
                ++retrWinBySchedule[sc][passthrough ? 1 : 0];
                if (passthrough) retrLicenseSumOnPass += lic;
                // retrVal is a sample-domain luma; convert to IRE for the
                // brightness gate the same way the hull dump does.
                if (retrVal * invIreScale > 100.0) {
                    ++retrBrightTotal;
                    ++retrBrightBySchedule[sc];
                }
            };

            // Election tolerances (IRE -> samples).
            const double inlierTol  = 4.0 * irescale; // medoid inlier gate
            const double phasePenSamp =
                std::max(0.0, configuration.tunables.PRODUCE_Y_PHASE_PENALTY_IRE)
                * irescale; // capped phase hygiene penalty

            // Carrier-basis window norms (constant per line: the 4-sample window
            // always spans the full set of phases regardless of start).
            double basisSN = 0.0, basisCN = 0.0;
            for (int i = 0; i < 4; ++i) {
                basisSN += spLUT_locked[i] * spLUT_locked[i];
                basisCN += cpLUT_locked[i] * cpLUT_locked[i];
            }

            // Candidate plane sampler: complete raw - carrier luma at sample hh.
            // Any non-finite source falls back to comb so a plane never poisons
            // a neighbor probe with NaN.
            auto planeY = [&](int plane, int hh) -> double {
                const int xx = hh - left;
                if (plane == 1 && retractedRow) {
                    const double r = (double)retractedRow[xx];
                    if (std::isfinite(r)) return r;
                } else if (plane == 3 && oneDRow) {
                    const double o = oneDRow[xx];
                    if (std::isfinite(o)) return (double)rawLine[hh] - o;
                } else if (plane == 4 && ccMaskRow) {
                    const double c = carrierComp ? carrierComp[xx] : clpLine[hh];
                    const double comb = (double)rawLine[hh] -
                        (std::isfinite(c) ? c : 0.0);
                    const double m = std::clamp((double)ccMaskRow[xx], 0.0, 1.0);
                    // Delta-value return (2026-08-02): the returned VALUE is
                    // the measured contamination -- the elected carrier's
                    // excess over the anchored carrier -- so lawful chroma
                    // stays in chroma. raw - anchored == retracted luma by
                    // construction, so a full commit hands the site the
                    // anticipation-anchored luma. Lab conviction: the
                    // committed verdict with the old full-residual value
                    // recovered 20.8% of the stolen-luma gap but paid
                    // 1.2-2.3 IRE at fired sites on colored content (legal
                    // carrier dumped into Y). Where no anchored plane
                    // exists the full-residual return stands -- there the
                    // verdict itself came from the fallback witness.
                    //
                    // WITHDRAWN 2026-08-02 -- a segment hull [comb, raw] was
                    // installed here and removed the same day. It was NOT a
                    // hull on an impossible: both endpoints are defined by
                    // comb, so its width IS the comb's carrier claim, and
                    // passing raw only means the residual carrier changes
                    // SIGN -- ordinary for a signed oscillation whenever
                    // comb's sign or magnitude is wrong. It therefore
                    // encoded "comb's carrier claim is correct" as if it
                    // were a law, and handed the tie to comb (3.22 IRE at
                    // return-delivering sites) over retracted (2.32 IRE).
                    // Measured: it bound on 5.2% of pixels but on 0.00% of
                    // COVERED def samples -- i.e. exclusively where no exact
                    // channel exists to confirm it -- and moved the 2fSC
                    // speckle class it was meant to fix by +0.0..+0.3%.
                    // A restriction that acts only where it cannot be
                    // verified, against the better estimator, needs a
                    // tiebreaker; none exists. The genuine impossible (a
                    // carrier larger than any legal carrier) is already
                    // enforced by feasible() on the roster.
                    if (returnedHasDerivedCarrier && retractedRow) {
                        const double r = (double)retractedRow[xx];
                        if (std::isfinite(r)) return comb + m * (r - comb);
                    }
                    return comb + m * ((double)rawLine[hh] - comb);
                }
                const double c = carrierComp ? carrierComp[xx] : clpLine[hh];
                return (double)rawLine[hh] - (std::isfinite(c) ? c : 0.0);
            };

            // Candidate export port (see LDCD_YVIEW above): publish one
            // contestant's COMPLETE luma, exactly as the election samples it.
            if (yViewMode >= 0) {
                for (int h = left; h < right; ++h)
                    Y[h] = planeY(yViewMode, h);
                continue;
            }

            // Three-band composition. The selected coarse is the LF platform.
            // The cheap block coarse publishes one value per carrier cycle, so
            // its information-rate Nyquist is fSC/2.  A centered two-cycle
            // (8-sample) mean of comb-minus-platform is therefore deliberately
            // omitted: that band belongs to the platform. Comb supplies the
            // middle between the 8- and 4-sample apertures. Election contestants
            // contribute only the remainder above the legal four-sample
            // aperture, whose response is zero at fSC; the election consequently
            // retains full authority over carrier-rate cross-colour.
            auto candidateResidualAt = [&](int plane, int hh) -> double {
                const int xx = hh - left;
                return planeY(plane, hh) - coarseRow[xx];
            };
            auto candidateEvenMeanAt = [&](int plane, int h0,
                                           int effectiveWidth) -> double {
                double sum = 0.0;
                const int half = effectiveWidth / 2;
                for (int k = -half; k <= half; ++k) {
                    const int hh = std::clamp(h0 + k, left, right - 1);
                    const double w =
                        (k == -half || k == half) ? 0.5 : 1.0;
                    sum += w * candidateResidualAt(plane, hh);
                }
                return sum / (double)effectiveWidth;
            };
            auto candidateFourMeanAt = [&](int plane, int h0) -> double {
                return candidateEvenMeanAt(plane, h0, 4);
            };
            auto candidatePlatformResidualAt = [&](int plane,
                                                    int h0) -> double {
                return candidateEvenMeanAt(plane, h0, 8);
            };
            auto candidateMiddleAt = [&](int plane, int h0) -> double {
                return candidateFourMeanAt(plane, h0) -
                       candidatePlatformResidualAt(plane, h0);
            };
            auto candidateTopAt = [&](int plane, int h0) -> double {
                return candidateResidualAt(plane, h0) -
                       candidateFourMeanAt(plane, h0);
            };
            auto completeTopAt = [&](auto completeAt,
                                     const double *platform,
                                     int h0) -> double {
                if (!platform)
                    return std::numeric_limits<double>::quiet_NaN();
                const int x0 = h0 - left;
                const double center = completeAt(h0) - platform[x0];
                if (!std::isfinite(center))
                    return std::numeric_limits<double>::quiet_NaN();
                double middle = 0.0;
                for (int k = -2; k <= 2; ++k) {
                    const int hh = std::clamp(h0 + k, left, right - 1);
                    const int xx = hh - left;
                    const double complete = completeAt(hh);
                    if (!std::isfinite(complete))
                        return std::numeric_limits<double>::quiet_NaN();
                    const double w = (k == -2 || k == 2) ? 0.5 : 1.0;
                    middle += w * (complete - platform[xx]);
                }
                return center - middle * 0.25;
            };

            // Carrier-basis cleanliness: 1 - (AC energy explained by the carrier
            // basis). This is a direct waveform measurement, not an aggregate
            // "confidence" whose provenance is hidden.
            // Cycle-integrated over four effective samples on the symmetric
            // half-endpoint support x-2..x+2. The aperture is registered at
            // the pixel it judges and does not flicker at carrier rate. Remove
            // the weighted mean before both projection and norm: DC cannot
            // project onto a complete carrier cycle, and it must not inflate
            // the denominator and make a DC-displaced candidate look clean.
            auto carrierCleanlinessOf = [&](int plane, int h0) -> double {
                double hf5[5], s5[5], c5[5], w5[5];
                double meanHF = 0.0;
                for (int j = 0; j < 5; ++j) {
                    const int k = j - 2;
                    const int hh = std::clamp(h0 + k, left, right - 1);
                    const double w = (j == 0 || j == 4) ? 0.5 : 1.0;
                    const double dc = coarseRow[hh - left];
                    hf5[j] = planeY(plane, hh) - dc;
                    // Index the carrier basis by the grammar sample class, NOT
                    // hh & 3. The locked demod (the basis these LUTs were built
                    // for) uses carrierSampleClass(line, h); a raw-position
                    // index applies a per-line rotation, making cleanliness
                    // line-dependent -> a line-alternating election penalty
                    // (checkerboard) on luma transitions.
                    const int idx = carrierSampleClass(line, hh);
                    s5[j] = spLUT_locked[idx];
                    c5[j] = cpLUT_locked[idx];
                    w5[j] = w;
                    meanHF += w * hf5[j];
                }
                meanHF *= 0.25;
                double dotS = 0.0, dotC = 0.0, nrm = 0.0;
                for (int j = 0; j < 5; ++j) {
                    const double ac = hf5[j] - meanHF;
                    dotS += w5[j] * ac * s5[j];
                    dotC += w5[j] * ac * c5[j];
                    nrm  += w5[j] * ac * ac;
                }
                const double carrierE =
                    (basisSN > 1e-9 ? dotS * dotS / basisSN : 0.0) +
                    (basisCN > 1e-9 ? dotC * dotC / basisCN : 0.0);
                return std::clamp(1.0 - carrierE / (nrm + 1e-9), 0.0, 1.0);
            };

            // Cycle-integrated carrier remaining in raw - candidate Y. This is
            // the amount that candidate would still publish as chroma, measured
            // on the same integer-centred four-effective-sample cycle.
            // It is an amplitude measurement with explicit provenance, not a
            // candidate label or an aggregate quality judgment.
            auto remainingCarrierMagnitudeOf = [&](int plane, int h0) -> double {
                double dotS = 0.0, dotC = 0.0;
                for (int k = -2; k <= 2; ++k) {
                    const int hh = std::clamp(h0 + k, left, right - 1);
                    const double w = (k == -2 || k == 2) ? 0.5 : 1.0;
                    const double residualCarrier =
                        (double)rawLine[hh] - planeY(plane, hh);
                    const int idx = carrierSampleClass(line, hh);
                    dotS += w * residualCarrier * spLUT_locked[idx];
                    dotC += w * residualCarrier * cpLUT_locked[idx];
                }
                const double carrierE =
                    (basisSN > 1e-9 ? dotS * dotS / basisSN : 0.0) +
                    (basisCN > 1e-9 ? dotC * dotC / basisCN : 0.0);
                return std::sqrt(std::max(0.0, carrierE));
            };

            const bool coarseLines = lockedLumaCacheValid && demodWidth == width;
            struct ProduceYNeighborRows {
                int line = -1;
                bool have = false;
                const quint16 *raw = nullptr;
                const double *cc = nullptr;
                const double *clp = nullptr;
                const float *ret = nullptr;
                const double *coarse = nullptr;
                const lddecode::CarrierAnalysisRecord *analysis = nullptr;
                const CombCarrierGrammar *grammar = nullptr;
            };
            auto makeNeighborRows = [&](int l) {
                ProduceYNeighborRows n;
                n.line = l;
                n.have = coarseLines && l >= firstLine &&
                         l < lastLine && l < demodLines;
                if (!n.have) return n;
                n.raw = rawbuffer.data() + l * fullWidth;
                n.cc = lockedCarrierComposite_line(l);
                n.clp = clpbuffer[srcBuf].pixel[l];
                n.ret = carrierRetracted_line(l);
                n.coarse = coarseFloor_line(l);
                n.analysis = carrierAnalysis_line(l);
                n.grammar = carrierGrammarLine(l);
                return n;
            };
            ProduceYNeighborRows north1 = makeNeighborRows(line - 1);
            ProduceYNeighborRows south1 = makeNeighborRows(line + 1);
            ProduceYNeighborRows north2 = makeNeighborRows(line - 2);
            ProduceYNeighborRows south2 = makeNeighborRows(line + 2);

            // Regime-sensitive vertical anchors. Frame/progressive evidence
            // uses adjacent picture lines (±1); field/interlace evidence uses
            // same-field neighbours (±2). FVF can vary this per pixel.
            const bool variantFrameRegime =
                configuration.twoDVariant ==
                    Comb::Configuration::TwoDVariant::FrameAAdaptiveIQ ||
                configuration.twoDVariant ==
                    Comb::Configuration::TwoDVariant::FrameBDirectIQ;
            const bool variantFvf =
                configuration.twoDVariant ==
                    Comb::Configuration::TwoDVariant::FieldVsFrame;
            const bool haveFvfLine =
                line >= 0 && line < (int)fvfMetrics.size() &&
                (int)fvfMetrics[line].size() >= width;
            auto verticalStepAt = [&](int xi) {
                // In Y the carrier phase no longer matters, so vertical
                // evidence uses the IMMEDIATELY adjacent picture line whenever
                // the cadence says the frame is progressive (telecine) --
                // spacing was a carrier-relation concern, not a luma one.
                if (carrierFrameVerticalAllowed(line))
                    return 1;
                if (variantFvf && haveFvfLine)
                    return fvfMetrics[line][xi].frameModel ? 1 : 2;
                return variantFrameRegime ? 1 : 2;
            };
            auto northRowsForStep = [&](int step) -> const ProduceYNeighborRows& {
                return (step == 1) ? north1 : north2;
            };
            auto southRowsForStep = [&](int step) -> const ProduceYNeighborRows& {
                return (step == 1) ? south1 : south2;
            };
            auto legalRel = [&](const CombCarrierGrammar *gNbr) {
                if (!grammarLine || !gNbr)
                    return false;
                const auto rel = lddecode::carrierGrammarSignedPhaseRelation(
                    grammarLine, left, gNbr, left);
                return rel == lddecode::CarrierPhaseRelation::Same ||
                       rel == lddecode::CarrierPhaseRelation::Opposite;
            };

            // Robust top band at a neighbour pixel: median of that pixel's
            // complete luma planes after removing its one selected coarse and
            // the centered four-sample middle band.
            // Returns false where the neighbour lacks a usable coarse.
            auto neighborHFAt = [&](const quint16 *rawP, const double *ccP,
                                    const double *clpP,
                                    const float *retP,
                                    const double *coaP, int hh, double &out) -> bool {
                if (!rawP || !coaP) return false;
                double v[2]; int n = 0;
                const double identityTol = 1e-6 * irescale;
                auto appendUnique = [&](double value) {
                    if (!std::isfinite(value)) return;
                    for (int i = 0; i < n; ++i)
                        if (std::fabs(v[i] - value) <= identityTol)
                            return;
                    v[n++] = value;
                };
                auto combComplete = [&](int hk) {
                    const int xk = hk - left;
                    const double c = ccP ? ccP[xk]
                                         : (clpP ? clpP[hk] : 0.0);
                    return std::isfinite(c)
                        ? (double)rawP[hk] - c
                        : std::numeric_limits<double>::quiet_NaN();
                };
                if (ccP || clpP)
                    appendUnique(completeTopAt(combComplete, coaP, hh));
                if (retP) {
                    auto retractedComplete = [&](int hk) {
                        return (double)retP[hk - left];
                    };
                    appendUnique(completeTopAt(retractedComplete, coaP, hh));
                }
                if (n == 0) return false;
                if (n == 1) out = v[0];
                else out = 0.5 * (v[0] + v[1]);
                return true;
            };

            // ld-disc-stacker primitives (mode 3/6), specialised for the small
            // candidate set. medoid = robust self-center (agreement fast path
            // and the no-neighbour fallback); closest reconciles a nomination
            // to it. The full election scoring lives inline below.
            auto medoidD = [](const double *a, int n) -> double {
                if (n == 1) return a[0];
                if (n == 2) return 0.5 * (a[0] + a[1]);
                int best = 0; double bestTot = 1e300;
                for (int i = 0; i < n; ++i) {
                    double t = 0.0;
                    for (int j = 0; j < n; ++j) t += std::fabs(a[i] - a[j]);
                    if (t < bestTot) { bestTot = t; best = i; }
                }
                return a[best];
            };
            auto closestD = [](const double *a, int n, double target) -> double {
                double best = a[0];
                for (int i = 1; i < n; ++i)
                    if (std::fabs(target - a[i]) < std::fabs(target - best))
                        best = a[i];
                return best;
            };

            for (int h = left; h < right; ++h) {
                const int xi = h - left;
                const double rawH = (double)rawLine[h];

                // combY -- senior comb candidate.
                double combY;
                if (carrierComp) {
                    const double c = carrierComp[xi];
                    combY = std::isfinite(c)
                        ? rawH - c
                        : (std::isfinite(clpLine[h])
                               ? rawH - clpLine[h]
                               : rawH);
                } else {
                    const double c = clpLine[h];
                    combY = std::isfinite(c) ? rawH - c : rawH;
                }

                // (D-S)/2 referee: grade every estimator against the exact
                // carrier wherever the side channel covers this sample.
                // Before any early-out so coverage is complete.
                if (DsRefProbe::on() && dsExactRow &&
                    std::isfinite(dsExactRow[h])) {
                    const double ex = dsExactRow[h];
                    const double hd = dsHDeltaRow ? dsHDeltaRow[xi] : 0.0;
                    const int bin = hd >= 6.0 ? 1 : 0;
                    g_dsRefProbe.covered++;
                    // Referee repair (2026-08-02): the certified head and
                    // factFit stamp locked1DSource / carrierFit with the
                    // exact fact at precisely the samples this referee
                    // grades, so the raw planes read |e|=0 by construction.
                    // Grade the PRE-fact estimator stashes instead.
                    const float *preHead = probePreHead1D_flat.empty()
                        ? nullptr
                        : probePreHead1D_flat.data() +
                              static_cast<size_t>(line) * demodWidth;
                    const float *preFit = probePreFactFit_flat.empty()
                        ? nullptr
                        : probePreFactFit_flat.data() +
                              static_cast<size_t>(line) * demodWidth;
                    const double oneDEst =
                        (preHead && std::isfinite(preHead[xi]))
                            ? (double)preHead[xi]
                            : (oneDRow ? oneDRow[xi]
                                       : std::numeric_limits<
                                             double>::quiet_NaN());
                    if (std::isfinite(oneDEst))
                        g_dsRefProbe.e[bin][0].add((oneDEst - ex) * invIreScale);
                    g_dsRefProbe.e[bin][1].add((rawH - combY - ex) * invIreScale);
                    const float *fitRowDs = carrierFit_line(line);
                    const double fitEst =
                        (preFit && std::isfinite(preFit[xi]))
                            ? (double)preFit[xi]
                            : ((fitRowDs && std::isfinite(fitRowDs[xi]))
                                   ? (double)fitRowDs[xi]
                                   : std::numeric_limits<
                                         double>::quiet_NaN());
                    if (std::isfinite(fitEst))
                        g_dsRefProbe.e[bin][2].add((fitEst - ex) * invIreScale);
                    if (retractedRow && std::isfinite(retractedRow[xi]))
                        g_dsRefProbe.e[bin][3].add(
                            (rawH - (double)retractedRow[xi] - ex) * invIreScale);
                    // Zone-restricted per-sample dump (LDCD_DUMP_DSZONE_*,
                    // run -t 1): the exact value and each estimator's signed
                    // error in IRE, one text line per covered sample, for
                    // offline per-line grading of a region (pillar work).
                    static const auto zEnv = [](const char *n) {
                        const char *s = std::getenv(n);
                        return s ? std::atoi(s) : -1;
                    };
                    static const int zL0 = zEnv("LDCD_DUMP_DSZONE_L0");
                    static const int zL1 = zEnv("LDCD_DUMP_DSZONE_L1");
                    static const int zC0 = zEnv("LDCD_DUMP_DSZONE_C0");
                    static const int zC1 = zEnv("LDCD_DUMP_DSZONE_C1");
                    if (zL0 >= 0 && line >= zL0 && line <= zL1 &&
                        xi >= zC0 && xi <= zC1) {
                        const double e1 = std::isfinite(oneDEst)
                                ? (oneDEst - ex) * invIreScale
                                : std::numeric_limits<double>::quiet_NaN();
                        const double ef = std::isfinite(fitEst)
                                ? (fitEst - ex) * invIreScale
                                : std::numeric_limits<double>::quiet_NaN();
                        const double er =
                            (retractedRow && std::isfinite(retractedRow[xi]))
                                ? (rawH - (double)retractedRow[xi] - ex) *
                                      invIreScale
                                : std::numeric_limits<double>::quiet_NaN();
                        std::fprintf(stderr,
                            "DSZONE line=%d xi=%d hd=%.1f ex=%.2f "
                            "e1D=%.2f eComb=%.2f eFit=%.2f eRetr=%.2f\n",
                            line, xi, hd, ex * invIreScale,
                            e1, (rawH - combY - ex) * invIreScale, ef, er);
                    }
                }

                // Cross-colour return referee: grade the RETURN, and each
                // detector component that drives it, against the exact
                // carrier. Before any early-out so coverage is complete.
                if (ccRefOn) {
                    // The mask's own parity geometry needs every line, not
                    // just the covered ones: the return ACTS on both parities
                    // while truth can only grade one.
                    g_ccRefProbe.note(line, xi, width, lastLine,
                                      (rawH - combY) * invIreScale,
                                      ccMaskRow
                                          ? std::clamp((double)ccMaskRow[xi],
                                                       0.0, 1.0)
                                          : 0.0);
                }
                if (ccRefOn && dsExactRow && std::isfinite(dsExactRow[h])) {
                    const double exIRE = (double)dsExactRow[h] * invIreScale;
                    const double carr  = (rawH - combY) * invIreScale;
                    const double leak  = carr - exIRE;
                    const double mask  = ccMaskRow
                        ? std::clamp((double)ccMaskRow[xi], 0.0, 1.0) : 0.0;

                    const double gA = ccImpurityRow
                        ? std::clamp((double)ccImpurityRow[xi], 0.0, 1.0) : 0.0;
                    const double regionKeep = ccRegionRow
                        ? std::clamp((double)ccRegionRow[xi], 0.0, 1.0) : 0.0;
                    // Retired notch edge read, rebuilt for CONTRAST from the
                    // PRE-head estimator stash (referee repair 2026-08-02:
                    // locked1DSource IS the exact fact on covered samples, so
                    // the old rebuild here was grading certified luma under a
                    // detector label).
                    const float *ccPreHead = probePreHead1D_flat.empty()
                        ? nullptr
                        : probePreHead1D_flat.data() +
                              static_cast<size_t>(line) * demodWidth;
                    auto oneDEstAt = [&](int rel) -> double {
                        if (ccPreHead && std::isfinite(ccPreHead[rel]))
                            return (double)ccPreHead[rel];
                        return oneDRow ? oneDRow[rel]
                                       : std::numeric_limits<
                                             double>::quiet_NaN();
                    };
                    double ccHDeltaIRE = 0.0;
                    if (oneDRow || ccPreHead) {
                        const int xm = std::max(0, xi - 2);
                        const int xp = std::min(width - 1, xi + 2);
                        const double nM =
                            (double)rawLine[left + xm] - oneDEstAt(xm);
                        const double nP =
                            (double)rawLine[left + xp] - oneDEstAt(xp);
                        ccHDeltaIRE = std::fabs(nP - nM) * invIreScale;
                    }
                    const double edgeRamp = std::clamp(
                        (ccHDeltaIRE - kCcEdgeSoftIRE) /
                            (kCcEdgeHardIRE - kCcEdgeSoftIRE), 0.0, 1.0);
                    double gPass = analysisRow
                        ? lddecode::carrierLegalProof(
                              (double)analysisRow[xi].carrierConformance,
                              (double)analysisRow[xi].conformanceSupportFraction)
                        : 0.0;
                    gPass = std::max(gPass, std::max(
                        ccAnalysisUpRow
                            ? lddecode::carrierLegalProof(
                                  (double)ccAnalysisUpRow[xi].carrierConformance,
                                  (double)ccAnalysisUpRow[xi]
                                      .conformanceSupportFraction)
                            : 0.0,
                        ccAnalysisDnRow
                            ? lddecode::carrierLegalProof(
                                  (double)ccAnalysisDnRow[xi].carrierConformance,
                                  (double)ccAnalysisDnRow[xi]
                                      .conformanceSupportFraction)
                            : 0.0));
                    const double lurchIRE = analysisRow
                        ? (double)analysisRow[xi].residual.maxAbsMembershipIRE
                        : 0.0;
                    const double lurchRamp = std::clamp(
                        (lurchIRE - kCcLurchSoftIRE) /
                            (kCcLurchHardIRE - kCcLurchSoftIRE), 0.0, 1.0);

                    // Live committed-verdict read: devIRE / (2*cutoff), so
                    // bin 0.5 = the shipping cutoff. -1/NaN (no anchored
                    // plane) pins to 0 = the bail regime.
                    double devRamp = 0.0;
                    if (!probeCcDevIRE_flat.empty()) {
                        const float dv = probeCcDevIRE_flat[
                            static_cast<size_t>(line) * demodWidth + xi];
                        if (std::isfinite(dv) && dv >= 0.0f)
                            devRamp = std::clamp(
                                (double)dv / (2.0 * kCcCommitCutoffIRE),
                                0.0, 1.0);
                    }
                    const double ccReads[CcRefProbe::NREADS] = {
                        gA, edgeRamp, regionKeep, gPass, lurchRamp, mask,
                        devRamp };
                    const double ccHd = dsHDeltaRow ? dsHDeltaRow[xi] : 0.0;
                    g_ccRefProbe.sample(ccHd >= 6.0 ? 1 : 0, carr, leak, mask,
                                        ccReads, line, xi);

                    // Star-signature classification, from the COMPOSITE'S OWN
                    // SHAPE (no estimator input): white peak over agreeing
                    // dark flanks, down by ±2 (b-w-b inside 4 px), classified
                    // clean vs ringing by the ±3..±5 skirt. Thresholds are
                    // what the actuator would use; truth grades them here.
                    int starCls = -1;
                    double starPeakIRE = 0.0, starSkirtEx = 0.0;
                    {
                        const double inv = invIreScale;
                        double flank = 0.0, flankL = 0.0, flankR = 0.0;
                        const double peakIRE = starSignatureAt(
                            rawLine, h, &flank, &flankL, &flankR);
                        auto rw = [&](int k) { return (double)rawLine[h + k]; };
                        double ringIRE = 0.0;
                        if (peakIRE > 0.0)
                            for (int k = 3; k <= 5; ++k)
                                ringIRE = std::max(ringIRE, std::max(
                                    std::fabs(rw(-k) - flankL),
                                    std::fabs(rw(k) - flankR)) * inv);
                        if (peakIRE > 0.0) {
                            const bool ringing =
                                ringIRE > std::max(3.0, 0.2 * peakIRE);
                            starCls = ringing ? 1 : 0;
                            starPeakIRE = peakIRE;
                            // Skirt truth: mean |ex| over ±3..±5 where the
                            // side channel covers them.
                            double se = 0.0; int sn_ = 0;
                            for (int k = 3; k <= 5; ++k)
                                for (int s = -1; s <= 1; s += 2) {
                                    const float v = dsExactRow[h + s * k];
                                    if (std::isfinite(v)) {
                                        se += std::fabs((double)v) * inv;
                                        sn_++;
                                    }
                                }
                            starSkirtEx = sn_ > 0 ? se / sn_ : 0.0;
                            const double oneDEstStar = oneDEstAt(xi);
                            const double e1D = std::isfinite(oneDEstStar)
                                    ? (oneDEstStar -
                                       (double)dsExactRow[h]) * inv
                                    : std::numeric_limits<
                                          double>::quiet_NaN();
                            g_ccRefProbe.starSample(starCls, carr, leak, mask,
                                                    exIRE, starPeakIRE, e1D,
                                                    starSkirtEx);
                            // Vertical context: signature on the four frame
                            // neighbours at this column (±1 col tolerance).
                            int vertRun = 0;
                            for (int vk : { -2, -1, 1, 2 }) {
                                const int vl = line + vk;
                                if (vl < firstLine || vl >= lastLine)
                                    continue;
                                const quint16 *vRaw = rawbuffer.data() +
                                    (size_t)vl * videoParameters.fieldWidth;
                                bool hit = false;
                                for (int dc = -1; dc <= 1 && !hit; ++dc)
                                    hit = starSignatureAt(vRaw, h + dc,
                                                          nullptr) > 0.0;
                                if (hit) vertRun++;
                            }
                            g_ccRefProbe.vertSample(vertRun, carr, leak,
                                                    mask, exIRE);
                        }
                    }

                    // Explanation ledger: class assignment (star > legal >
                    // detail > flat), per-estimator errors, gain windows.
                    {
                        // Referee repair: pre-stamp fit / pre-head 1D, so
                        // the explanation ledger grades estimators.
                        const float *fitRowCc = carrierFit_line(line);
                        const float *preFitCc = probePreFactFit_flat.empty()
                            ? nullptr
                            : probePreFactFit_flat.data() +
                                  static_cast<size_t>(line) * demodWidth;
                        const double fitIRE =
                            (preFitCc && std::isfinite(preFitCc[xi]))
                                ? (double)preFitCc[xi] * invIreScale
                                : (fitRowCc && std::isfinite(fitRowCc[xi]))
                                    ? (double)fitRowCc[xi] * invIreScale
                                    : std::numeric_limits<
                                          double>::quiet_NaN();
                        const double oneDEstCls = oneDEstAt(xi);
                        const double oneDIRE = std::isfinite(oneDEstCls)
                                ? oneDEstCls * invIreScale
                                : std::numeric_limits<double>::quiet_NaN();
                        const int cls =
                            starCls == 0 ? CcRefProbe::CStarClean
                            : starCls == 1 ? CcRefProbe::CStarRing
                            : gPass >= 0.7 ? CcRefProbe::CLegal
                            : ccHd >= 6.0 ? CcRefProbe::CDetail
                                          : CcRefProbe::CFlat;
                        g_ccRefProbe.classSample(cls, fitIRE, carr, oneDIRE,
                                                 exIRE, line, xi);
                        g_ccRefProbe.gainWindow(line, fitIRE, carr, exIRE);
                    }
                }

                double ccReturn = ccMaskRow
                    ? std::clamp((double)ccMaskRow[xi], 0.0, 1.0)
                    : 0.0;
                // Fact-audit on the Y-RETURN duty only (regional, smooth;
                // built in splitIQlocked pass 2 from the nearest covered
                // facts grading the detector's claims).
                if (ccReturn > 0.0 && !ccAuditW_flat.empty() &&
                    ccAuditNX > 0) {
                    const double ry = std::clamp(
                        (line - 16.0) / 32.0, 0.0,
                        (double)(ccAuditNY - 1) - 1e-6);
                    const double rx = std::clamp(
                        (xi - 64.0) / 128.0, 0.0,
                        (double)(ccAuditNX - 1) - 1e-6);
                    const int r0 = (int)ry, c0 = (int)rx;
                    const double wy = ry - r0, wx = rx - c0;
                    const int r1 = std::min(r0 + 1, ccAuditNY - 1);
                    const int c1 = std::min(c0 + 1, ccAuditNX - 1);
                    auto A = [&](int rr, int cc2) {
                        return (double)ccAuditW_flat[
                            (size_t)rr * ccAuditNX + cc2];
                    };
                    ccReturn *= (1 - wy) * ((1 - wx) * A(r0, c0) +
                                            wx * A(r0, c1)) +
                                wy * ((1 - wx) * A(r1, c0) +
                                      wx * A(r1, c1));
                }
                const double ccMeasuredHere = ccMaskRawRow
                    ? std::clamp((double)ccMaskRawRow[xi], 0.0, 1.0)
                    : 0.0;
                // A mask buffer exists for the whole frame whenever the feature
                // is enabled. Do not let that allocation alone activate the
                // larger election at an unmarked pixel when no retracted luma
                // plane otherwise requires it.
                if (!retractedRow && ccReturn <= 0.0) {
                    Y[h] = combY;
                    continue;
                }

                // Coarse floor unavailable -> emit combY.
                if (!coarseRow) {
                    Y[h] = combY;
                    continue;
                }
                const double coarse = coarseRow[xi];
                const double combMiddle = candidateMiddleAt(0, h);
                // The comb's platform-residual band (the centred 8-sample mean
                // of comb-minus-platform) must be carried, not dropped.  It was
                // omitted on the theory that the selected coarse already owns
                // everything below the 8-sample aperture, but the coarse is a
                // raster-aligned BLOCK mean -- piecewise constant across each
                // 4fSC cycle -- not a centred 8-sample mean.  Its block-pitch
                // stairstep is exactly what this band corrects, so discarding
                // it published that stairstep as luma: +20% line-alternation at
                // Gilgol Beach chroma transitions (the bead crawl) and inflated
                // GGV false saturation, because the discarded remainder is
                // carrier-band.  Carrying it makes the identity explicit --
                //   Y = combY + (electedTop - combTop)
                // -- which IS the stated design: coarse owns LF, comb owns the
                // middle and the provisional top, and the election swaps only
                // the top band.
                const double combPlatform = candidatePlatformResidualAt(0, h);
                auto reconstructTop = [&](int plane, double top) {
                    // Non-comb winners keep their own four-mean so the top
                    // band is rebuilt in the same candidate geometry that won.
                    const double fourMean =
                        (plane == 0)
                            ? (combMiddle + combPlatform)
                            : candidateFourMeanAt(plane, h);
                    return coarse + fourMean + top;
                };
                // Transfer-knee probe (measurement only; inert unless
                // LDCD_PROBE_KNEE): at proven-luma pixels the retracted top
                // is trustworthy amplitude; bin comb's top against it.
                if (KneeProbe::on() && retractedRow && analysisRow) {
                    const double proofK = lddecode::carrierIllegalProof(
                        (double)analysisRow[xi].carrierConformance,
                        (double)analysisRow[xi].conformanceSupportFraction);
                    if (proofK >= 0.7) {
                        const double rT = candidateTopAt(1, h);
                        const double cT = candidateTopAt(0, h);
                        if (std::isfinite(rT) && std::isfinite(cT))
                            g_kneeProbe.sample(rT * invIreScale,
                                               cT * invIreScale);
                    }
                }

                const int verticalStep = verticalStepAt(xi);
                const ProduceYNeighborRows &northRows =
                    northRowsForStep(verticalStep);
                const ProduceYNeighborRows &southRows =
                    southRowsForStep(verticalStep);
                const bool requireVerticalCarrierRelation =
                    (verticalStep == 2);
                const bool relLegalN =
                    !requireVerticalCarrierRelation ||
                    legalRel(northRows.grammar);
                const bool relLegalS =
                    !requireVerticalCarrierRelation ||
                    legalRel(southRows.grammar);

                auto feasible = [&](double y) {
                    const double c = rawH - y;
                    return c <= maxCarrierAmpSamples && c >= -maxCarrierAmpSamples;
                };

                // Roster with structural feasibility DQ. Coherent comb is the
                // senior hypothesis; 1D replaces it only if comb is infeasible.
                // Retracted Y may join as the one independent base challenger.
                double candY[3]; // top-band values; name retained locally
                int    candPlane[3];
                int    nCand = 0;
                const double identityTol = 1e-6 * irescale;
                auto addBaseCandidate = [&](double completeY, int plane) {
                    const double y = candidateTopAt(plane, h);
                    if (!std::isfinite(completeY) || !feasible(completeY) ||
                        !std::isfinite(y))
                        return;
                    // Population statistics describe distinct numerical
                    // hypotheses, not the number of pipelines which happened
                    // to publish one. The plane provenance remains on the
                    // retained value; duplicate hypotheses must not count
                    // coherent Y twice.
                    for (int k = 0; k < nCand; ++k)
                        if (std::fabs(candY[k] - y) <= identityTol)
                            return;
                    candY[nCand] = y;
                    candPlane[nCand] = plane;
                    ++nCand;
                };
                const bool combOK = std::isfinite(combY) && feasible(combY);

                if (combOK) {
                    addBaseCandidate(combY, 0);
                }

                if (retractedRow) {
                    // Evidence admission for retractedY (raw - combedCarrier),
                    // the leg that keeps near-carrier HF luma the comb strips.
                    //
                    // A low four-view residualTrust marks a conflicted fit —
                    // but that conflict is present for BOTH broadband texture
                    // noise (beach: retracted is junk, comb should stand) AND
                    // real near-carrier periodic structure (Borg-cube grid:
                    // retracted carries the detail comb destroyed).  Trust
                    // alone cannot separate them, so it must not hard-DQ:
                    // that is a fit-quality measure vetoing geometry, which
                    // the election forbids.
                    //
                    // The separator is SPATIAL COHERENCE. Real structure
                    // agrees with its regime-appropriate vertical neighbours:
                    // frame/progressive uses adjacent picture lines (±1),
                    // field/interlace uses same-field partners (±2). Per-pixel
                    // texture noise agrees with neither. So: a clean fit is
                    // admitted outright; a conflicted fit is admitted only when
                    // its HF is corroborated by a neighbour. Feasibility
                    // remains the only true DQ.
                    const double r = retractedRow ? (double)retractedRow[xi] : combY;
                    const double ry = std::isfinite(r) ? r : combY;
                    bool retractedAdmitted =
                        retractedAdmitAll ||
                        !analysisRow ||
                        (analysisRow[xi].parallax.residualValid &&
                         analysisRow[xi].parallax.residualTrust >= 0.5f);
                    // Schedule-illegality admits by LAW, not corroboration:
                    // an alien ±2 partner means this pixel's carrier-band
                    // energy is raw-identical where the schedule demands
                    // inversion — structurally not carrier, hence luma.
                    // retractedY is the plane that keeps it as luma, and it
                    // is exactly the line-pitch detail (Borg-cube grid) that
                    // a same-field agreement check can never certify, because
                    // that detail IS the same-field disagreement.
                    if (!retractedAdmitted && alienRow && alienRow[xi] > 0.5f)
                        retractedAdmitted = true;
                    // Same law from the analysis-time registration, which
                    // adds the FRAME axis (static line-decorrelated detail
                    // the ±2 tests cannot reach).
                    if (!retractedAdmitted && analysisRow &&
                        analysisRow[xi].scheduleConformance ==
                            lddecode::CarrierScheduleConformance::ScheduleIllegal)
                        retractedAdmitted = true;
                    if (!retractedAdmitted && retractedAdmitSpatial &&
                        retractedRow && std::isfinite(r)) {
                        auto currentRetracted = [&](int hh) {
                            return (double)retractedRow[hh - left];
                        };
                        const double rTop = completeTopAt(
                            currentRetracted, coarseRow, h);
                        // Frame/progressive vertical checks are luma-image
                        // checks and do not require carrier grammar. Field/
                        // interlace checks use ±2 same-field partners; there
                        // the schedule must certify phase-comparable lines
                        // before matched HF is accepted as structure.
                        if (relLegalN && northRows.ret && northRows.coarse) {
                            auto northRetracted = [&](int hh) {
                                return (double)northRows.ret[hh - left];
                            };
                            const double nTop = completeTopAt(
                                northRetracted, northRows.coarse, h);
                            if (std::isfinite(nTop) &&
                                std::fabs(rTop - nTop) <= inlierTol)
                                retractedAdmitted = true;
                        }
                        if (!retractedAdmitted && relLegalS &&
                            southRows.ret && southRows.coarse) {
                            auto southRetracted = [&](int hh) {
                                return (double)southRows.ret[hh - left];
                            };
                            const double sTop = completeTopAt(
                                southRetracted, southRows.coarse, h);
                            if (std::isfinite(sTop) &&
                                std::fabs(rTop - sTop) <= inlierTol)
                                retractedAdmitted = true;
                        }
                    }
                    if (retractedAdmitted)
                        addBaseCandidate(ry, 1);
                }
                if (!combOK && oneDRow) {
                    const double o = oneDRow[xi];
                    if (std::isfinite(o)) {
                        const double y1 = rawH - o;
                        addBaseCandidate(y1, 3);
                    }
                }

                // Returned Y is derived from combY, so it is a selectable
                // challenger but not another independent observation when the
                // base candidates establish their center, subset, or scoring
                // scales. It joins only after those quantities are fixed.
                // VERDICT IS THE FACT, as a VALUE (detectors-trade-for-
                // truth): on covered samples the returned candidate IS
                // certified luma, raw - ex. The mask formalism (m·C) can
                // only correct ALONG C; the true correction leak = C - ex
                // opposes C on roughly half the fired samples (referee:
                // helped 45% under the magnitude-only override). A fact
                // enters as the value, or not at all.
                static const bool ccFactsY = []{
                    const char *e = std::getenv("LDCD_CC_FACTS");
                    return !(e && std::atoi(e) == 0);
                }();
                const bool haveFactY = ccFactsY && dsExactRow &&
                                       std::isfinite(dsExactRow[h]);
                // Band-revoked residue with an affirmed luma claim
                // (splitIQlocked application stage): joins the returned
                // candidate as a VALUE, so the election's hull and
                // feasibility machinery contest it like any other claim.
                // Facts outrank it (the fact branch is already exact).
                const float *bandRow = haveFactY
                    ? nullptr : bandResidueY_line(line);
                const double bandVal =
                    (bandRow && std::isfinite((double)bandRow[xi]))
                        ? (double)bandRow[xi] : 0.0;
                // NOTE (open, 2026-08-02): this admission formula and the
                // value the election actually samples (planeY plane 4) are
                // NOT the same expression -- admission tests
                // comb + ccReturn*(raw - comb) + bandVal while the election
                // samples the delta form comb + m*(retracted - comb). The
                // roster therefore judges feasibility on a claim it will not
                // evaluate. Left as found pending a measured decision; do
                // not "fix" it by clamping, which was tried and withdrawn
                // (see planeY plane 4).
                const double returnedY = haveFactY
                    ? rawH - (double)dsExactRow[h]
                    : combY + ccReturn * (rawH - combY) + bandVal;
                const bool returnedFeasible =
                    (haveFactY || ccReturn > 0.0 || bandVal != 0.0) &&
                    std::isfinite(returnedY) && feasible(returnedY);

                if (nCand == 0) {
                    // No legal top band exists here. Publishing the illegal
                    // one truncated to the hull edge (Y = raw -/+ maxAmp) is
                    // not a rejection: it moves Y TOWARD raw, so it keeps a
                    // signed slice of the carrier waveform as luma. raw
                    // alternates about the luma at carrier rate, so that
                    // slice renders as a 2fSC speckle of up to ~20 IRE at
                    // exactly the near-peak samples where the estimate broke.
                    // Reject instead: keep the bands below the top aperture,
                    // which are 4- and 8-sample centred means and therefore
                    // cancel carrier by construction, and publish no top.
                    Y[h] = reconstructTop(0, 0.0);
                    tallyHull(rawH, Y[h], 6 /* no legal top */, h, 0.0, combY);
                    continue;
                }
                // Feasibility is the only DQ (stated doctrine, now enforced).
                // The former inlier gate (|cand - center| <= inlierTol around
                // a medoid that ties to comb on a 2-candidate roster) was a
                // second centrist filter: it ejected the detailed candidate at
                // exactly the pixels where it differed enough to matter, so
                // the election's scoring never saw the contest. Outliers are
                // bounded by the scoring itself -- a candidate far from every
                // adjacent neighbour pays its full distance as cost.
                //
                // A prior "base agreement" early-out used to commit outright
                // to the medoid-nearest candidate's own plane whenever the
                // base tops sat within agreeTol of each other -- bypassing
                // the blend on a coarser, differently-scaled metric than the
                // one the blend actually scores on (user, 2026-07-28: "the
                // medoid isn't that great, we shouldn't allow it to govern").
                // Removed: every base candidate now always reaches the cost
                // loop and the confidence-alpha blend below, agreeing or not.
                int inIdx[3];
                int nIn = 0;
                for (int k = 0; k < nCand; ++k)
                    inIdx[nIn++] = k;
                // The named cross-colour mask, rather than distance from the
                // base center, admits returned Y. A distance gate here removes
                // exactly the strong HF that the return exists to recover.
                const bool returnedAdmitted = returnedFeasible;
                // ELECTION BYPASS (diagnostic A/B only, LDCD_ELECT_BYPASS=1;
                // measurement instrument for the chroma-lean hunt, 2026-08-01).
                // Forces the top band to the PURE COMB candidate, so emitted Y
                // is the comb complement everywhere comb is feasible and the
                // election's edge-population departure (measured 3.5 IRE RMS on
                // ~20% of edge pixels) cannot express. If the locked-vs-bucket
                // chroma lean (I -0.39 / Q -0.20 px) collapses toward the
                // bucket referee under this, the election's complement
                // departure is the source; if it survives, the election is
                // exonerated and the bias is upstream of the roster.
                // NOT a shipping mode: it discards every other candidate.
                if (electBypass && combOK) {
                    Y[h] = reconstructTop(0, candidateTopAt(0, h));
                    continue;
                }

                if (nIn == 1 && !returnedAdmitted) {
                    Y[h] = reconstructTop(candPlane[inIdx[0]], candY[inIdx[0]]);
                    if (dumpHull) {
                        const int wp = candPlane[inIdx[0]];
                        tallyHull(rawH, Y[h], wp, h,
                                  candidateFourMeanAt(0, h) - candidateFourMeanAt(wp, h),
                                  planeY(wp, h));
                    }
                    tallyRetr(rawH, candPlane[inIdx[0]], combY, planeY(1, h), h,
                              analysisRow ? &analysisRow[xi] : nullptr);
                    continue;
                }

                // Inlier HF set + per-inlier carrier-basis cleanliness. This is
                // a cautionary term, not the positive reason to select HF.
                double inHF[3], inCarrierCleanliness[3];
                double inCrossColorReturnEvidence[3] = {0.0, 0.0, 0.0};
                for (int k = 0; k < nIn; ++k) {
                    inHF[k] = candY[inIdx[k]];
                    inCarrierCleanliness[k] =
                        carrierCleanlinessOf(candPlane[inIdx[k]], h);
                }
                const int baseNIn = nIn;
                auto planeForTop = [&](double top) {
                    for (int k = 0; k < baseNIn; ++k)
                        if (inHF[k] == top)
                            return candPlane[inIdx[k]];
                    for (int k = baseNIn; k < nIn; ++k)
                        if (inHF[k] == top)
                            return 4;
                    return 0;
                };

                // Single self-anchor: medoid of the BASE inlier HFs (mode 6).
                // The derived return may be selected, but does not move this
                // population statistic.
                const double selfAnchor = medoidD(inHF, baseNIn);
                if (returnedAdmitted) {
                    inHF[nIn] = candidateTopAt(4, h);
                    inCarrierCleanliness[nIn] =
                        carrierCleanlinessOf(4, h);
                    ++nIn;
                }

                // Let the named cross-colour evidence affect scoring according
                // to what each candidate actually does. The comb plane defines
                // zero return. A candidate earns only the cycle-integrated
                // carrier reduction it delivers relative to comb, and never
                // more than either the measured false-colour amount or the
                // explicit policy cap reported by the tunable.
                // Retracted Y can therefore receive this evidence when it
                // already outperforms nominal returned
                // Y; a label cannot win an advantage its samples did not earn.
                if (ccReturn > 0.0) {
                    const double combCarrierMagnitude =
                        remainingCarrierMagnitudeOf(0, h);
                    const double measuredFalseCarrier =
                        ccReturn * combCarrierMagnitude;
                    const double crossColorReturnCap =
                        std::max(0.0,
                            configuration.tunables
                                .PRODUCE_Y_CC_RETURN_EVIDENCE_CAP_IRE) *
                        irescale;
                    for (int k = 0; k < nIn; ++k) {
                        const int plane = (k < baseNIn)
                            ? candPlane[inIdx[k]] : 4;
                        const double deliveredReduction = std::max(
                            0.0,
                            combCarrierMagnitude -
                                remainingCarrierMagnitudeOf(plane, h));
                        inCrossColorReturnEvidence[k] = std::min(
                            crossColorReturnCap,
                            std::min(measuredFalseCarrier,
                                     deliveredReduction));
                    }
                }

                // Four independent image neighbours (N/S at regime-sensitive
                // vertical step, E/W at ±1 sample). Keep them separate: a line
                // or edge may continue in one direction while legitimately
                // crossing a transition in another.
                //
                // NOTE on twin pinning (user direction, 2026-07-28): a
                // distance-scaling weight on the pinned N/S testimony was
                // built and measured INERT here -- it only re-ranks which
                // neighbour anchors the boost, and cannot move the blend
                // toward the pinned value when no candidate already sits
                // near it (referee unchanged; comp-vs-bracket residual
                // 1.93->1.96 IRE). Pinning needs a value-level lever;
                // anchorLinePinned() is the infrastructure awaiting it.
                double dirHF[4], dirImageHF[4]; int nDir = 0;
                auto appendDirection = [&](const quint16 *rawP,
                                           const double *ccP,
                                           const double *clpP,
                                           const float *retP,
                                           const double *geometryFloor,
                                           const double *dcFloor,
                                           int hh) {
                    if (nDir >= 4 || !dcFloor) return;
                    double geometryHF, imageHF;
                    if (neighborHFAt(rawP, ccP, clpP, retP,
                                     geometryFloor, hh, geometryHF) &&
                        neighborHFAt(rawP, ccP, clpP, retP,
                                     dcFloor, hh, imageHF)) {
                        dirHF[nDir] = geometryHF;
                        dirImageHF[nDir] = imageHF;
                        ++nDir;
                    }
                };
                appendDirection(northRows.raw, northRows.cc, northRows.clp,
                                northRows.ret,
                                northRows.coarse, northRows.coarse, h);
                appendDirection(southRows.raw, southRows.cc, southRows.clp,
                                southRows.ret,
                                southRows.coarse, southRows.coarse, h);
                if (h - 1 >= left)
                    appendDirection(rawLine, carrierComp, clpLine,
                                    retractedRow,
                                    coarseRow, coarseRow, h - 1);
                if (h + 1 < right)
                    appendDirection(rawLine, carrierComp, clpLine,
                                    retractedRow,
                                    coarseRow, coarseRow, h + 1);

                // ---- Election scoring: neighbour boost + legality. ----
                // (2026-07-26 redesign, user-directed.)
                //
                // The anchor machinery (medoid self-anchor averaged with an
                // inlier-MEAN neighbour anchor) was a CENTRIST instrument:
                // after boldness was removed (correctly -- it selected
                // carrier) no pro-detail term remained, and softness became
                // the election's default. The elected output left the sharp
                // candidates' quality off to the side.
                //
                //   * NEIGHBOUR BOOST, not anchor. Each candidate's base cost
                //     is its distance to the NEAREST immediately adjacent
                //     neighbour top (E/W +-1; N/S +-1 wherever the cadence
                //     allows frame-vertical -- in Y, carrier phase no longer
                //     matters, so there is no spacing). Proximity to a real
                //     neighbour value lowers cost; no mean is formed anywhere.
                //   * LEGALITY is the pro-detail term, replacing boldness.
                //     Retained HF above comb's top earns preference only in
                //     proportion to the per-pixel PROOF that the energy
                //     cannot be carrier (carrierIllegalProof on the
                //     registered conformance). Detail wins only where it is
                //     provably luma; dot crawl is provably carrier and earns
                //     nothing. Legal-schedule errors remain -- some errors
                //     are legal -- but the rest are cut down.
                //   * Carrier-basis cleanliness stays as the capped caution.
                //   * IMPULSE SENIORITY: at star/impulse pixels (the single
                //     lumaImpulseRisk producer, kept as-is -- its job is
                //     rescuing stars mangled by comb) retracted outranks comb
                //     by a small bias.
                //   * HIGH-CHROMA DEMOTION: retracted is demoted where the
                //     measured chroma is strong (the tractor-beam checkers),
                //     until the retracted product is clean there.
                const double imagePrefCap =
                    std::max(0.0, configuration.tunables.PRODUCE_Y_HF_IMAGE_PREFERENCE_IRE)
                    * irescale;
                const double proxTol =
                    std::max(0.5, configuration.tunables.PRODUCE_Y_HF_CONTINUATION_IRE)
                    * irescale;
                constexpr double kImpulseRetractedBiasIRE = 1.0;
                constexpr double kHighChromaSoftIRE      = 10.0;
                constexpr double kHighChromaHardIRE      = 20.0;
                constexpr double kHighChromaDemoteIRE    = 1.5;

                // Diagnostics for the pyDiag dump: proximity01 and the earned
                // legality reward (IRE) take the retired continuation/retained
                // slots; the decision anchor is retired (no anchor exists).
                double diagContinuation[3] = {
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN()
                };
                double diagRetained[3] = {
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN()
                };
                double diagNeighborAnchor = std::numeric_limits<double>::quiet_NaN();
                const double diagDecisionAnchor = std::numeric_limits<double>::quiet_NaN();

                // Capped-caution reference: median cleanliness of the base set.
                double sw[3];
                const int refN = std::clamp(baseNIn, 1, nIn);
                for (int i = 0; i < refN; ++i)
                    sw[i] = inCarrierCleanliness[i];
                std::sort(sw, sw + refN);
                const double medianW = sw[refN / 2];

                const double combTopHere = candidateTopAt(0, h);
                const double illegalProof = analysisRow
                    ? lddecode::carrierIllegalProof(
                          (double)analysisRow[xi].carrierConformance,
                          (double)analysisRow[xi].conformanceSupportFraction)
                    : 0.0;
                const double impulseT = attribRow
                    ? std::clamp(attribRow[xi].facts.lumaImpulseRisk, 0.0, 1.0)
                    : 0.0;
                // High chroma means LEGAL carrier energy. The raw remaining-
                // carrier magnitude also counts confiscated illegal luma (the
                // grid), which would demote retracted exactly where it should
                // win; the illegal-proof share is excluded.
                const double chromaT = std::clamp(
                    (remainingCarrierMagnitudeOf(0, h) * invIreScale -
                     kHighChromaSoftIRE) /
                        (kHighChromaHardIRE - kHighChromaSoftIRE),
                    0.0, 1.0) * (1.0 - illegalProof);

                // Darkest-choice penalty: comb's weak impulse highlights.
                // At a genuine luma peak comb rounds AND darkens; retracted
                // keeps the peak, so the roster's DARKEST candidate is the
                // wrong one there, in proportion to the evidence that the
                // peak is real luma (illegal-proof or the star/impulse
                // channel) and bright (raw's own top band) -- and it pays
                // NOTHING where the energy is legal carrier, where the
                // most-subtracted candidate is usually the correct one
                // (otherwise this term would be reverse-boldness and
                // re-select carrier).
                //
                // Weight-space reform (user, 2026-07-28, at the beam
                // highlight join): the argmin-era form charged the gate
                // TIMES the roster spread -- big enough to always flip the
                // winner, and safe when only ordering mattered. Feeding
                // alpha, spread is an inter-candidate DISTANCE modulating
                // the mixing proportion, and at a highlight over dense
                // chroma that distance is the comb/retracted divergence
                // itself -- distance x difference in the render, the exact
                // product the blend-weight doctrine forbids, visible as the
                // join band. The evidence gate is already normalised, so it
                // enters as a multiplicative weight factor (1 - gate) on
                // the darkest candidate instead: smooth demotion along the
                // evidence ramp, a true veto at full proof, and the
                // candidates' distances nowhere in the weight.
                constexpr double kPeakSoftIRE = 8.0;
                constexpr double kPeakHardIRE = 20.0;
                int darkestIdx = 0;
                double darkestVetoGate = 0.0;
                if (baseNIn > 1) {
                    for (int k = 1; k < baseNIn; ++k) {
                        if (inHF[k] < inHF[darkestIdx]) darkestIdx = k;
                    }
                    auto rawComplete = [&](int hk) {
                        return (double)rawLine[hk];
                    };
                    const double rawTopIRE = std::fabs(
                        completeTopAt(rawComplete, coarseRow, h)) * invIreScale;
                    const double peakT = std::clamp(
                        (rawTopIRE - kPeakSoftIRE) /
                            (kPeakHardIRE - kPeakSoftIRE),
                        0.0, 1.0);
                    darkestVetoGate =
                        std::max(illegalProof, impulseT) * peakT;
                }

                double resultHF = inHF[0];
                // The Y-election band cede is REMOVED (user, 2026-07-28).
                // It hard-picked the comb candidate inside chromaBoundaryBand
                // to kill the bikini-diagonal per-column winner interleave
                // (2026-07-27), an artifact of the winner-take-all era that
                // the confidence-alpha blend obviates: alpha varies smoothly
                // where candidates tie, so there is no per-column flip left
                // to suppress. And the cede was measured blanketing the
                // entire cube face (band seeds fire on dense alternating
                // detail everywhere -- thin struts against shadow meet the
                // DifferentRegion definition), silencing the blend and the
                // retracted candidate exactly where retracted is the solid
                // image (probe 2026-07-28: bandCede 99-100% of face px,
                // comb top D2-RMS letter-alternating 7.8/4.9, retracted
                // stable ~6.2). The band plane stays published (Field B's
                // own settled cede is upstream and unaffected).
                // Confidence-alpha blend (user direction): the per-pixel
                // WINNER flip between candidates -- comb/retracted measured
                // 46/54 at boundary teeth, and frame-to-frame flips render
                // as strobing on detail -- is the same per-pixel-decision
                // artifact family as the Field B beading. The election's
                // cost terms stay exactly as they are (they carry the image
                // shaping: neighbour boost, legality, impulse seniority,
                // darkest-peak penalty), but they now shape ALPHA instead of
                // electing one winner: Y = sum w_k * reconstruct_k with
                // w = exp(-(cost-min)/tau). A decisive cost gap still yields
                // alpha ~= 1 (tau matched to the cost scale), so commitment
                // survives where evidence is clear and blending concentrates
                // where ambiguity -- and hence the strobing -- lives.
                // Vetoes remain binary and upstream: admission and the
                // feasibility DQ.
                double blendNum = 0.0, blendDen = 0.0;
                constexpr double kBlendTauIRE = 0.75;
                const double blendTau = kBlendTauIRE * irescale;
                double costs[4];
                double wDiag[4] = {0.0, 0.0, 0.0, 0.0};
                // Per-plane capture for the proximity-nulling measurement
                // (user, 2026-08-02). proximity01 falls to 0 as a candidate
                // departs from its neighbours -- which is the definition of
                // return DELIVERING -- so the evidence term it multiplies
                // may be switched off at exactly the sites it exists for.
                double proxByPlane[5] = {-1, -1, -1, -1, -1};
                double evByPlane[5]   = {0, 0, 0, 0, 0};
                // ELECTION v2 (user-approved shape, 2026-08-02):
                //  1. HF-surplus reward made CANDIDATE-NEUTRAL -- it no
                //     longer needs illegalProof to vouch for the site.
                //     Truth-validated as a general score, never a selector:
                //     argmax|HF| is the truth-best candidate 46-48% vs 33%
                //     chance across cube/title/bridge, yet a pure max-HF
                //     pick LOSES to the election (0.50 vs 0.46). A score,
                //     not a rule.
                //  2. The named plane==1 (retracted) adjustment retires --
                //     the neutral HF term is what it was approximating.
                //  3. The return-evidence term stops being multiplied by
                //     proximity01 (see above).
                //  4. Blend moves from an absolute-difference exponential
                //     to a ratio/log curve, so a candidate that is much
                //     better pedestals regardless of the local cost scale.
                // BITMASK so the four changes can be bisected:
                //   1 = candidate-neutral HF-surplus reward
                //   2 = retire the named plane==1 (retracted) term
                //   4 = drop proximity01 scaling on return evidence
                //   8 = ratio/log blend curve
                //  15 = all four (the bundle first measured)
                //
                // DEFAULT 7 (= 1|2|4, everything except the ratio blend),
                // promoted 2026-08-02 on a truth-graded bisection, held out
                // (CERT_1D=0 + CHROMA_FACTS=0 + native retraction), cube and
                // title, frame-wide elected error vs certified luma:
                //     HF neutral   -0.74% / -0.22%
                //     retire named +0.06% / -1.29%
                //     un-gate ev   -1.88% / -0.64%   <- largest single win
                //     ratio blend  +3.33% / +2.77%   <- the only loser
                //     all four     +0.41% / +0.14%   (blend cancels the rest)
                //     1|2|4        -2.56% / -2.14%   <- promoted
                // At return-delivering sites 1|2|4 is -11.4% / -4.4%.
                // The ratio/log blend is REJECTED BY MEASUREMENT: pedestaling
                // harder only pays where the pedestaled candidate is right,
                // and return is the best candidate at its delivering sites on
                // the cube but NOT on the title (1.82 vs a 1.70 blend). A
                // sharper curve cannot fix a score that cannot tell those two
                // regimes apart. LDCD_ELECT_V2=8 keeps it available for A/B.
                static const int electV2 = []{
                    const char *e = std::getenv("LDCD_ELECT_V2");
                    return e ? std::atoi(e) : 7;
                }();
                const bool ev2Hf    = (electV2 & 1) != 0;
                const bool ev2Named = (electV2 & 2) != 0;
                const bool ev2Prox  = (electV2 & 4) != 0;
                const bool ev2Blend = (electV2 & 8) != 0;
                static const double kHfBaseShare = []{
                    const char *e = std::getenv("LDCD_HF_BASE");
                    return e ? std::atof(e) : 0.40;
                }();
                {
                    double bestCost = 1e300;
                    for (int k = 0; k < nIn; ++k) {
                        const int plane =
                            (k < baseNIn) ? candPlane[inIdx[k]] : 4;
                        // Neighbour boost: nearest adjacent top. With no
                        // usable neighbour, the medoid stands in as the sole
                        // fallback reference.
                        double nd = 1e300, nv = selfAnchor;
                        for (int d = 0; d < nDir; ++d) {
                            const double dd = std::fabs(inHF[k] - dirHF[d]);
                            if (dd < nd) { nd = dd; nv = dirHF[d]; }
                        }
                        if (nDir == 0)
                            nd = std::fabs(inHF[k] - selfAnchor);
                        const double proximity01 =
                            1.0 - std::clamp(nd / proxTol, 0.0, 1.0);

                        double cost = nd;
                        if (medianW > 0.0)
                            cost += (std::max(0.0,
                                         medianW - inCarrierCleanliness[k]) /
                                     medianW) * phasePenSamp;
                        const double extra = std::isfinite(combTopHere)
                            ? std::max(0.0, std::fabs(inHF[k]) -
                                            std::fabs(combTopHere))
                            : 0.0;
                        // Early-transition slope (user, 2026-07-27): the
                        // comb-to-challenger handoff must occur while the
                        // candidates still nearly agree, hiding the seam. At
                        // unit slope the reward only overcame comb's
                        // neighbour advantage after the surplus was already
                        // large, so the flip landed mid-divergence and drew a
                        // visible step along rising edges. Doubling the slope
                        // halves the surplus at which the flip happens; the
                        // cap is unchanged, so large-surplus behaviour is
                        // identical.
                        constexpr double kEarlyHandoffSlope = 2.0;
                        // v2: a baseline share of the HF reward is paid
                        // without the estimator's proof; the proof still
                        // tops it up. Candidate-neutral by construction --
                        // it names only |HF| against comb's |HF|.
                        const double hfVouch = ev2Hf
                            ? (kHfBaseShare + (1.0 - kHfBaseShare) *
                                                  illegalProof)
                            : illegalProof;
                        const double legality =
                            hfVouch * std::min(kEarlyHandoffSlope * extra,
                                               imagePrefCap);
                        cost -= legality;
                        // Cross-colour return evidence. v1 scaled it by
                        // proximity01, which vanishes exactly where the
                        // candidate departs -- i.e. where return delivers.
                        if (plane >= 0 && plane <= 4) {
                            proxByPlane[plane] = proximity01;
                            evByPlane[plane] =
                                std::max(0.0, inCrossColorReturnEvidence[k]);
                        }
                        cost -= (ev2Prox ? 1.0 : proximity01) *
                            std::max(0.0, inCrossColorReturnEvidence[k]);
                        if (plane == 1 && !ev2Named)
                            cost += chromaT * kHighChromaDemoteIRE * irescale -
                                    impulseT * kImpulseRetractedBiasIRE *
                                        irescale;
                        if (k < 3) {
                            diagContinuation[k] = proximity01;
                            diagRetained[k] = legality * invIreScale;
                        }
                        if (k < 4) costs[k] = cost;
                        // Strict < keeps roster order as the neutral
                        // tie-break: coherent comb stays senior on ties.
                        if (cost < bestCost) {
                            bestCost = cost;
                            resultHF = inHF[k];
                            diagNeighborAnchor = nv;
                        }
                    }
                    // Anchor authority (user direction, 2026-07-28): where
                    // the exact channel granted a regional amplitude
                    // ceiling, a candidate whose implied carrier
                    // (raw - complete Y) exceeds it is claiming carrier the
                    // anchors certify cannot exist there. The excess enters
                    // as a normalised weight factor -- authority in the
                    // scoring, per the weight doctrine, never a hard cut --
                    // and the bound was max-pooled + margin-padded, so
                    // legal carrier never pays. Inert wherever no plane or
                    // no authority (+inf).
                    constexpr double kAnchorTauIRE = 1.5;
                    const double anchorCeilIRE =
                        anchorRow ? (double)anchorRow[h]
                                  : std::numeric_limits<double>::infinity();
                    for (int k = 0; k < nIn && k < 4; ++k) {
                        const int plane =
                            (k < baseNIn) ? candPlane[inIdx[k]] : 4;
                        const double yk = reconstructTop(plane, inHF[k]);
                        if (!std::isfinite(yk)) continue;
                        // v1: exp(-dCost/tau), an ABSOLUTE difference
                        // against a fixed 0.75 IRE scale -- where local
                        // costs are compressed it cannot separate anyone.
                        // v2: ratio/log curve w = (1 + dCost/floor)^-p, so
                        // the pedestal follows the RATIO of the scores.
                        static const double kElectFloorIRE = []{
                            const char *e = std::getenv("LDCD_ELECT_FLOOR");
                            return e ? std::atof(e) : 0.15;
                        }();
                        static const double kElectPow = []{
                            const char *e = std::getenv("LDCD_ELECT_POW");
                            return e ? std::atof(e) : 2.5;
                        }();
                        double w;
                        if (ev2Blend) {
                            const double d = std::max(0.0,
                                costs[k] - bestCost) /
                                (kElectFloorIRE * irescale);
                            w = std::exp(-kElectPow * std::log1p(d));
                        } else {
                            w = std::exp(-(costs[k] - bestCost) / blendTau);
                        }
                        if (k == darkestIdx && k < baseNIn && baseNIn > 1)
                            w *= 1.0 - darkestVetoGate;
                        if (std::isfinite(anchorCeilIRE)) {
                            const double complete = (k < baseNIn)
                                ? planeY(candPlane[inIdx[k]], h)
                                : returnedY;
                            const double cIRE =
                                std::fabs(rawH - complete) * invIreScale;
                            const double excess = cIRE - anchorCeilIRE;
                            if (excess > 0.0)
                                w *= std::exp(-excess / kAnchorTauIRE);
                        }
                        wDiag[k] = w;
                        blendNum += w * yk;
                        blendDen += w;
                    }
                }

                const int winnerPlane = planeForTop(resultHF);
                Y[h] = (blendDen > 1e-12)
                    ? blendNum / blendDen
                    : reconstructTop(winnerPlane, resultHF);

                if (retImpactOn && antRefAge >= 1 &&
                    !frameHasExactCoverage() &&
                    antRefLuma_flat.size() ==
                        (size_t)frameHeight * demodWidth) {
                    const float ref = antRefLuma_flat[
                        (size_t)line * demodWidth + xi];
                    if (std::isfinite(ref)) {
                        riErrSum += std::fabs(Y[h] - (double)ref) * invIreScale;
                        riAbsYSum += std::fabs(rawH - Y[h]) * invIreScale;
                        ++riN;
                    }
                }

                if (yCandOn && dsExactRow &&
                    std::isfinite(dsExactRow[h])) {
                    const double lTrue = rawH - (double)dsExactRow[h];
                    // truth's own local gradient, for the slope figures
                    double gT = 0.0; bool haveG = false;
                    if (h - 1 >= left && h + 1 < right &&
                        std::isfinite(dsExactRow[h - 1]) &&
                        std::isfinite(dsExactRow[h + 1])) {
                        const double lm = (double)rawLine[h - 1] -
                                          (double)dsExactRow[h - 1];
                        const double lp = (double)rawLine[h + 1] -
                                          (double)dsExactRow[h + 1];
                        gT = std::fabs(lp - lm) * invIreScale;
                        haveG = gT > 8.0;
                    }
                    if (retHullProbeOn && ccMaskRow && carrierComp) {
                        const double cv = carrierComp[xi];
                        const double combV =
                            rawH - (std::isfinite(cv) ? cv : 0.0);
                        const double mv = std::clamp(
                            (double)ccMaskRow[xi], 0.0, 1.0);
                        double pre = std::numeric_limits<double>::quiet_NaN();
                        if (returnedHasDerivedCarrier && retractedRow &&
                            std::isfinite(retractedRow[xi]))
                            pre = combV + mv *
                                ((double)retractedRow[xi] - combV);
                        else
                            pre = combV + mv * (rawH - combV);
                        if (mv > 0.0 && std::isfinite(pre)) {
                            const double post = std::clamp(pre,
                                std::min(combV, rawH),
                                std::max(combV, rawH));
                            const double ePre =
                                std::fabs(pre - lTrue) * invIreScale;
                            const double ePost =
                                std::fabs(post - lTrue) * invIreScale;
                            ++rhN; rhPre += ePre; rhPost += ePost;
                            if (std::fabs(post - pre) > 1e-9) {
                                ++rhBindN;
                                rhBindPre += ePre;
                                rhBindPost += ePost;
                                ++rhBindDef;
                            }
                        }
                    }
                    double bestE = 1e300;
                    int bestK = -1;
                    // HF-AS-QUALITY-PROXY TEST (user proposal, 2026-08-02:
                    // "Rewarding HF content is a nice, candidate neutral
                    // score that would give retracted a boost without
                    // calling it by name"). Before wiring any such term
                    // into the cost, ask truth whether it is a valid
                    // proxy: does the candidate carrying the MOST HF
                    // actually stand closest to certified luma? Reported
                    // as (a) how often argmax|HF| == the truth-best
                    // candidate, against the 1/n chance baseline, and
                    // (b) what a pure "always pick max HF" selector would
                    // have scored versus the shipping election and the
                    // per-pixel oracle.
                    int hfArgPlane = -1; double hfMaxAbs = -1.0;
                    double hfPickErrHere = std::numeric_limits<
                        double>::quiet_NaN();
                    // RETURN-DELIVERY SPLIT (user, 2026-08-02: return must
                    // "win on the merits where it's delivering", since comb
                    // and retracted do not address bandpass luma at all).
                    // Delivering = where the returned candidate actually
                    // departs from comb, i.e. where it is claiming stolen
                    // bandpass luma. Everywhere else it is comb by
                    // construction and grading it says nothing.
                    double ykByPlane[5];
                    for (int q = 0; q < 5; ++q)
                        ykByPlane[q] = std::numeric_limits<double>::quiet_NaN();
                    for (int k = 0; k < nIn && k < 4; ++k) {
                        const int plane =
                            (k < baseNIn) ? candPlane[inIdx[k]] : 4;
                        if (plane < 0 || plane > 4) continue;
                        const double yk = reconstructTop(plane, inHF[k]);
                        if (!std::isfinite(yk)) continue;
                        const double e = std::fabs(yk - lTrue) * invIreScale;
                        candErr[plane] += e; candN[plane]++;
                        ykByPlane[plane] = yk;
                        if (e < bestE) { bestE = e; bestK = plane; }
                        const double hfAbs = std::fabs(inHF[k]);
                        if (hfAbs > hfMaxAbs) {
                            hfMaxAbs = hfAbs;
                            hfArgPlane = plane;
                            hfPickErrHere = e;
                        }
                        if (haveG) {
                            // candidate gradient via its own plane values
                            const double ym = planeY(plane, h - 1);
                            const double yp = planeY(plane, h + 1);
                            if (std::isfinite(ym) && std::isfinite(yp)) {
                                candSlopeY[plane] +=
                                    std::fabs(yp - ym) * invIreScale;
                                candSlopeT[plane] += gT;
                                candSlopeN[plane]++;
                            }
                        }
                    }
                    if (bestK >= 0) {
                        electErr += std::fabs(Y[h] - lTrue) * invIreScale;
                        oracleErr += bestE;
                        electN++;
                        oracleWin[bestK]++;
                        if (hfArgPlane >= 0 && std::isfinite(hfPickErrHere)) {
                            hfPickErr += hfPickErrHere;
                            hfPickN++;
                            hfChanceSum += 1.0 / (double)std::max(1, nIn);
                            if (hfArgPlane == bestK) hfPickHit++;
                        }
                        const double yRet = ykByPlane[4];
                        const double yComb = ykByPlane[0];
                        if (std::isfinite(yRet) && std::isfinite(yComb) &&
                            std::fabs(yRet - yComb) * invIreScale >=
                                kRetDeliverIRE) {
                            ++ractN;
                            ractElect += std::fabs(Y[h] - lTrue) * invIreScale;
                            ractRet   += std::fabs(yRet - lTrue) * invIreScale;
                            ractComb  += std::fabs(yComb - lTrue) * invIreScale;
                            const double yRetr = ykByPlane[1];
                            if (std::isfinite(yRetr)) {
                                ractRetr +=
                                    std::fabs(yRetr - lTrue) * invIreScale;
                                ++ractRetrN;
                            }
                            // Where does the elected Y actually sit between
                            // the two? 0 = on comb, 1 = on return.
                            const double span = yRet - yComb;
                            if (std::fabs(span) > 1e-9) {
                                ractPos += std::clamp(
                                    (Y[h] - yComb) / span, -1.0, 2.0);
                                ++ractPosN;
                            }
                            if (std::fabs(yRet - lTrue) <
                                std::fabs(yComb - lTrue))
                                ++ractRetBetter;
                            if (proxByPlane[4] >= 0.0) {
                                ractProxRet += proxByPlane[4];
                                ractEvRet   += evByPlane[4];
                                ++ractProxN;
                            }
                        }
                    }
                }
                if (consDump) {
                    // Bucket writes Y and chroma from ONE scalar:
                    //   Y = raw - val, chroma = demod(val)  => Y + chroma == raw.
                    // Locked demodulates chroma from carrierComp but emits Y
                    // from a band reassembly, so the identity only survives
                    // where the winner telescopes back to raw - carrierComp.
                    const double bucketY = rawH - (carrierComp ? carrierComp[xi]
                                                               : clpLine[h]);
                    const double d = Y[h] - bucketY;
                    ++consN; consSum += d * d;
                    consMax = std::max(consMax, std::fabs(d));
                    if (winnerPlane == 0) { ++consP0N; consP0Sum += d * d; }
                    else                  { ++consOtherN; consOtherSum += d * d; }
                }

                tallyRetr(rawH, winnerPlane, combY, planeY(1, h), h,
                          analysisRow ? &analysisRow[xi] : nullptr);

                if (dumpHull) {
                    int winnerPlaneHull = 5; // fallback: unattributed blend
                    for (int k = 0; k < nIn; ++k) {
                        if (inHF[k] == resultHF) {
                            winnerPlaneHull = (k < baseNIn) ? candPlane[inIdx[k]] : 4;
                            break;
                        }
                    }
                    const int wp = (winnerPlaneHull == 5) ? 0 : winnerPlaneHull;
                    tallyHull(rawH, Y[h], winnerPlaneHull, h,
                              candidateFourMeanAt(0, h) - candidateFourMeanAt(wp, h),
                              planeY(wp, h));
                }

                if (pyDiag && line >= pyDiagL0 && line <= pyDiagL1 &&
                    xi >= pyDiagC0 && xi <= pyDiagC1) {
                    // Winner plane: resultHF is one of the inHF entries.
                    int winnerPlane = -1;
                    for (int k = 0; k < nIn; ++k)
                        if (inHF[k] == resultHF) {
                            winnerPlane = (k < baseNIn) ? candPlane[inIdx[k]] : 4;
                            break;
                        }
                    // Roster string: planes of the admitted base candidates.
                    char roster[64]; int rp = 0;
                    for (int k = 0; k < baseNIn && rp < 60; ++k)
                        rp += std::snprintf(roster + rp, sizeof(roster) - rp,
                                            "%d ", candPlane[inIdx[k]]);
                    if (rp == 0) { roster[0] = '-'; roster[1] = 0; }
                    else roster[rp ? rp - 1 : 0] = 0;
                    // Neighbour image-HF continuation values (N,S,E,W order).
                    char nbr[96]; int np = 0;
                    for (int d = 0; d < nDir && np < 90; ++d)
                        np += std::snprintf(nbr + np, sizeof(nbr) - np,
                                            "%.2f ", dirImageHF[d] / irescale);
                    if (np == 0) { nbr[0] = '-'; nbr[1] = 0; }
                    else nbr[np ? np - 1 : 0] = 0;
                    // Per-candidate elected TOP (IRE), including 1D even when
                    // excluded, plus the comb-owned middle. Keeping the bands
                    // separate makes it impossible for this diagnostic to hide
                    // a coarse or middle-band substitution under an HF label.
                    const double combTop = candidateTopAt(0, h) / irescale;
                    const double combPlatformResidual =
                        candidatePlatformResidualAt(0, h) / irescale;
                    const double retrTop = retractedRow
                        ? candidateTopAt(1, h) / irescale
                        : std::numeric_limits<double>::quiet_NaN();
                    const double oneDTop = oneDRow
                        ? candidateTopAt(3, h) / irescale
                        : std::numeric_limits<double>::quiet_NaN();
                    const double retnTop = candidateTopAt(4, h) / irescale;
                    auto rawComplete = [&](int hh) {
                        return (double)rawLine[hh];
                    };
                    // Mono composite top: ground-reference with carrier still
                    // included. Its line alternation exposes raw passthrough.
                    const double monoTop = completeTopAt(
                        rawComplete, coarseRow, h) / irescale;
                    const int returnedIndex = returnedAdmitted
                        ? nIn - 1 : -1;
                    const double returnedImagePref =
                        (returnedIndex >= 0 && returnedIndex < 3)
                        ? diagContinuation[returnedIndex]
                        : std::numeric_limits<double>::quiet_NaN();
                    const double returnedCcEvidence = returnedIndex >= 0
                        ? inCrossColorReturnEvidence[returnedIndex] / irescale
                        : std::numeric_limits<double>::quiet_NaN();
                    // Diagnostic-only: nativeTop is NOT a roster candidate.
                    // Measured here to answer the separability question the
                    // full-field-Y-runoff plan raised -- how far apart are
                    // native (raw-carrierFit, no vertical promotion) and
                    // promoted (retrTop, raw-combedCarrier) in the top band
                    // -- without seating native and paying its election cost.
                    const float *nativeCarrierRow = carrierFit_line(line);
                    double nativeTop = std::numeric_limits<double>::quiet_NaN();
                    if (nativeCarrierRow) {
                        auto nativeComplete = [&](int hh) {
                            return (double)rawLine[hh] -
                                (double)nativeCarrierRow[hh - left];
                        };
                        nativeTop = completeTopAt(
                            nativeComplete, coarseRow, h) / irescale;
                    }
                    // Per-admitted-candidate cleanliness and cc-return
                    // evidence, paired positionally with roster=[...]
                    // (baseNIn entries; the derived return is reported
                    // separately above as it is not a base candidate).
                    char cleanStr[64]; int cp = 0;
                    for (int k = 0; k < baseNIn && cp < 60; ++k)
                        cp += std::snprintf(cleanStr + cp, sizeof(cleanStr) - cp,
                                            "%.2f ", inCarrierCleanliness[k]);
                    if (cp == 0) { cleanStr[0] = '-'; cleanStr[1] = 0; }
                    else cleanStr[cp ? cp - 1 : 0] = 0;
                    char ccEvStr[64]; int cep = 0;
                    for (int k = 0; k < baseNIn && cep < 60; ++k)
                        cep += std::snprintf(ccEvStr + cep, sizeof(ccEvStr) - cep,
                                            "%.2f ", inCrossColorReturnEvidence[k] / irescale);
                    if (cep == 0) { ccEvStr[0] = '-'; ccEvStr[1] = 0; }
                    else ccEvStr[cep ? cep - 1 : 0] = 0;
                    // Image evidence per admitted candidate: proximity01 (the
                    // neighbour-boost factor) now fills the imgPref slot.
                    char imgStr[64]; int ip = 0;
                    for (int k = 0; k < baseNIn && ip < 60 && k < 3; ++k)
                        ip += std::snprintf(imgStr + ip, sizeof(imgStr) - ip,
                                            "%.3f ", diagContinuation[k]);
                    if (ip == 0) { imgStr[0] = '-'; imgStr[1] = 0; }
                    else imgStr[ip ? ip - 1 : 0] = 0;
                    char contStr[64]; int cnp = 0;
                    for (int k = 0; k < baseNIn && cnp < 60; ++k)
                        cnp += std::snprintf(contStr + cnp, sizeof(contStr) - cnp,
                                            "%.3f ", diagContinuation[k]);
                    if (cnp == 0) { contStr[0] = '-'; contStr[1] = 0; }
                    else contStr[cnp ? cnp - 1 : 0] = 0;
                    char retnStr[64]; int rtp = 0;
                    for (int k = 0; k < baseNIn && rtp < 60; ++k)
                        rtp += std::snprintf(retnStr + rtp, sizeof(retnStr) - rtp,
                                            "%.3f ", diagRetained[k]);
                    if (rtp == 0) { retnStr[0] = '-'; retnStr[1] = 0; }
                    else retnStr[rtp ? rtp - 1 : 0] = 0;
                    // Blend-weight view: normalised alpha per roster slot
                    // (positional with roster=[...]; the derived return is
                    // the trailing slot when admitted), plus the two
                    // election downgrade gates acting on this pixel.
                    char wStr[64]; int wp2 = 0;
                    const double wSumDiag =
                        wDiag[0] + wDiag[1] + wDiag[2] + wDiag[3];
                    for (int k = 0; k < nIn && k < 4 && wp2 < 60; ++k)
                        wp2 += std::snprintf(wStr + wp2, sizeof(wStr) - wp2,
                                            "%.3f ",
                                            wSumDiag > 1e-12
                                                ? wDiag[k] / wSumDiag
                                                : 0.0);
                    if (wp2 == 0) { wStr[0] = '-'; wStr[1] = 0; }
                    else wStr[wp2 ? wp2 - 1 : 0] = 0;
                    std::fprintf(stderr,
                        "PYDIAG line=%d h=%d xi=%d vstep=%d combOK=%d "
                        "1Dexcl=%d nCand=%d roster=[%s] clean=[%s] "
                        "ccEv=[%s] imgPref=[%s] cont=[%s] retn=[%s] "
                        "selfA=%.2f nbrA=%.2f decA=%.2f winPlane=%d "
                        "winTop=%.2f coarseIRE=%.2f combMiddle=%.2f "
                        "combPlatformResidual=%.2f ccRaw=%.2f ccRet=%.2f "
                        "monoTop=%.2f combTop=%.2f retrTop=%.2f nativeTop=%.2f "
                        "oneDTop=%.2f retnTop=%.2f "
                        "retnImg=%.3f retnCcEv=%.2f "
                        "alpha=[%s] chromaT=%.3f dkGate=%.3f dkPlane=%d "
                        "band=%d "
                        "nbrImgTop=[%s]\n",
                        line, h, xi, verticalStep, combOK ? 1 : 0,
                        (combOK && oneDRow) ? 1 : 0, nCand, roster, cleanStr,
                        ccEvStr, imgStr, contStr, retnStr,
                        selfAnchor / irescale, diagNeighborAnchor / irescale,
                        diagDecisionAnchor / irescale, winnerPlane,
                        resultHF / irescale, coarse / irescale,
                        combMiddle / irescale, combPlatformResidual,
                        ccMeasuredHere, ccReturn, monoTop, combTop, retrTop,
                        nativeTop, oneDTop,
                        retnTop, returnedImagePref, returnedCcEvidence,
                        wStr, chromaT, darkestVetoGate,
                        (baseNIn > 1) ? candPlane[inIdx[darkestIdx]] : -1,
                        (bandRow && bandRow[xi]) ? 1 : 0,
                        nbr);
                }
            }
        } else if (carrierComp) {
            for (int h = left; h < right; ++h) {
                const double c = carrierComp[h - left];
                Y[h] = std::isfinite(c)
                    ? (double)rawLine[h] - c
                    : (double)rawLine[h] - clpLine[h];
            }
        } else {
            for (int h = left; h < right; ++h) {
                const double c = clpLine[h];
                Y[h] = std::isfinite(c) ? (double)rawLine[h] - c : (double)rawLine[h];
            }
        }

        // The split has already declared zero carrier on this footprint.
        // Honour that complement after every Y branch: Y = raw - 0.  This is
        // not a second decision and does not reshape the footprint.
        if (starFootprintBuilt &&
            starFootprint_flat.size() >=
                static_cast<size_t>(frameHeight) * demodWidth) {
            const std::uint8_t *star = starFootprint_flat.data() +
                static_cast<size_t>(line) * demodWidth;
            for (int xi = 0; xi < width; ++xi)
                if (star[xi]) Y[left + xi] = (double)rawLine[left + xi];
        }

        // RETGRADE post-pass (per line, Y row now complete): grade emitted
        // Y against certified truth, classed by mask x site, and split each
        // error into its CARRIER-BAND vs out-of-band parts. The in-band
        // share is the theft signature: luma stolen by the 1D bandpass
        // lives near fSC, so residual error that is IN-band at a
        // quiet-mask site is recoverable theft the detector missed, while
        // broadband error there was never stolen luma at all and return
        // cannot fix it.
        if (retGradeOn) {
            // Self-contained refetch (this scope is outside the demod
            // block's row declarations).
            const float *dsExactRow = exactCarrierRow(line);
            const float *ccMaskRow = lockedCcMask_line(line);
            const float *dsHDeltaRow = lockedLumaHDeltaIRE_line(line);
            const AttributionEvidence *attribRow =
                attributionEvidence_line(line);
            const quint16 *rawLinePtr = rawbuffer.data() +
                static_cast<size_t>(line) * videoParameters.fieldWidth;
            const double *Y = componentFrame->y(line);
            if (dsExactRow && Y) {
            constexpr double bT0 = 0.676462;
            constexpr double bT2 = -0.250000;
            constexpr double bT4 = -0.088231;
            static thread_local std::vector<double> errRow;
            if ((int)errRow.size() < width)
                errRow.resize(width);
            for (int xi2 = 0; xi2 < width; ++xi2) {
                const int hh = left + xi2;
                errRow[xi2] = std::isfinite(dsExactRow[hh])
                    ? Y[hh] - ((double)rawLinePtr[hh] -
                               (double)dsExactRow[hh])
                    : std::numeric_limits<double>::quiet_NaN();
            }
            for (int xi2 = 0; xi2 < width; ++xi2) {
                const double e0 = errRow[xi2];
                if (!std::isfinite(e0)) continue;
                const double aerr = std::fabs(e0) * invIreScale;
                rgErrSum += aerr;
                rgErrSqSum += e0 * e0 * invIreScale * invIreScale;
                rgSignedSum += e0 * invIreScale;
                ++rgN;
                const double m = ccMaskRow
                    ? std::clamp((double)ccMaskRow[xi2], 0.0, 1.0) : 0.0;
                const int mb = (m < 0.05) ? 0 : (m <= 0.5 ? 1 : 2);
                const double impT = attribRow
                    ? std::clamp(
                          attribRow[xi2].facts.lumaImpulseRisk, 0.0, 1.0)
                    : 0.0;
                const double hd = dsHDeltaRow
                    ? (double)dsHDeltaRow[xi2] : 0.0;
                const int sc = (impT > 0.3) ? 2 : (hd >= 6.0 ? 1 : 0);
                const int c = mb * 3 + sc;
                rgCellErr[c] += aerr;
                rgCellN[c]++;
                // carrier-band part of the error at this sample
                bool ok = true; double taps[5];
                static const int off5[5] = { 0, -2, 2, -4, 4 };
                for (int k = 0; k < 5 && ok; ++k) {
                    const int j = std::clamp(xi2 + off5[k], 0, width - 1);
                    taps[k] = errRow[j];
                    if (!std::isfinite(taps[k])) ok = false;
                }
                if (ok) {
                    const double bp = bT0 * taps[0] +
                                      bT2 * (taps[1] + taps[2]) +
                                      bT4 * (taps[3] + taps[4]);
                    rgCellBand[c] += std::fabs(bp) * invIreScale;
                    rgCellBandN[c]++;
                }
            }
            }
        }

        if (showMap) {
            std::fill(w2d_frame_weight[line].begin(),
                      w2d_frame_weight[line].end(), 0.0f);
        }
    }

    if (retGradeOn && rgN > 0) {
        std::fprintf(stderr,
            "[RETGRADE] n=%ld meanAbsErr=%.4f IRE rms=%.4f IRE "
            "meanSignedErr=%+.4f IRE\n",
            rgN, rgErrSum / rgN,
            std::sqrt(rgErrSqSum / rgN), rgSignedSum / rgN);
        static const char *mbn[3] = {"quiet", "partial", "strong"};
        static const char *scn[3] = {"flat", "edge", "impulse"};
        for (int mb = 0; mb < 3; ++mb)
            for (int sc = 0; sc < 3; ++sc) {
                const int c = mb * 3 + sc;
                if (!rgCellN[c]) continue;
                std::fprintf(stderr,
                    "[RETGRADE-CELL] mask=%s site=%s n=%ld "
                    "meanAbsErr=%.4f errShare=%.1f%% pxShare=%.1f%% "
                    "inBandShare=%.1f%%\n",
                    mbn[mb], scn[sc], rgCellN[c],
                    rgCellErr[c] / rgCellN[c],
                    100.0 * rgCellErr[c] / rgErrSum,
                    100.0 * rgCellN[c] / rgN,
                    rgCellBandN[c]
                        ? 100.0 * (rgCellBand[c] / rgCellBandN[c]) /
                              (rgCellErr[c] / rgCellN[c])
                        : 0.0);
            }
    }

    if (retImpactOn && riN > 0) {
        std::fprintf(stderr,
            "[RETIMPACT] uncovered n=%ld  Y-vs-antRefLuma mean|err| %.3f IRE "
            "(approx reference, NOT certified fact -- content scale "
            "mean|Y-raw| %.2f IRE)\n",
            riN, riErrSum / riN, riAbsYSum / riN);
    }

    if (consDump && consN > 0) {
        std::fprintf(stderr,
            "[CONS] Y vs (raw-carrierComp): rms=%.3f IRE max=%.2f IRE  "
            "plane0 n=%lld rms=%.3f | other n=%lld rms=%.3f (%.1f%% of px)\n",
            std::sqrt(consSum / consN) * invIreScale,
            consMax * invIreScale,
            consP0N, consP0N ? std::sqrt(consP0Sum / consP0N) * invIreScale : 0.0,
            consOtherN, consOtherN ? std::sqrt(consOtherSum / consOtherN) * invIreScale : 0.0,
            100.0 * (double)consOtherN / (double)consN);
    }

    if (dumpHull) {
        auto pct = [](long long a, long long b) {
            return b > 0 ? 100.0 * static_cast<double>(a)
                                 / static_cast<double>(b)
                         : 0.0;
        };
        std::fprintf(stderr,
            "[YHULL] tested=%lld over=%lld(%.4f%%) sumOverIRE=%.1f maxOverIRE=%.2f\n",
            hullTotal, hullOver, pct(hullOver, hullTotal),
            hullOverSumIRE, hullMaxOvershootIRE);
        std::fprintf(stderr,
            "[YHULL] over-by-plane comb=%lld retracted=%lld oneD=%lld "
            "returned=%lld blend=%lld clamped=%lld | "
            "sumMismatchIRE=%.1f maxMismatchIRE=%.2f\n",
            hullOverByPlane[0], hullOverByPlane[1], hullOverByPlane[3],
            hullOverByPlane[4], hullOverByPlane[5], hullOverByPlane[6],
            hullMismatchSumIRE, hullMaxMismatchIRE);
    }

    if (dumpRetr) {
        auto pct = [](long long a, long long b) {
            return b > 0 ? 100.0 * static_cast<double>(a)
                                 / static_cast<double>(b) : 0.0;
        };
        std::fprintf(stderr,
            "[RETR] wins=%lld passthrough=%lld(%.2f%%) meanLicenseOnPass=%.3f\n",
            retrWinTotal, retrPassthroughTotal,
            pct(retrPassthroughTotal, retrWinTotal),
            retrPassthroughTotal > 0
                ? retrLicenseSumOnPass / (double)retrPassthroughTotal : 0.0);
        std::fprintf(stderr,
            "[RETR] all-wins   bySchedule unres=%lld legal=%lld illegal=%lld\n",
            retrWinBySchedule[0][0] + retrWinBySchedule[0][1],
            retrWinBySchedule[1][0] + retrWinBySchedule[1][1],
            retrWinBySchedule[2][0] + retrWinBySchedule[2][1]);
        std::fprintf(stderr,
            "[RETR] passthrough bySchedule unres=%lld legal=%lld illegal=%lld"
            " <- LEGAL here = passthrough failure; ILLEGAL = real grid luma\n",
            retrWinBySchedule[0][1], retrWinBySchedule[1][1],
            retrWinBySchedule[2][1]);
        std::fprintf(stderr,
            "[RETR] bright(>100IRE) total=%lld bySchedule unres=%lld legal=%lld"
            " illegal=%lld <- the specks the eye spots\n",
            retrBrightTotal, retrBrightBySchedule[0],
            retrBrightBySchedule[1], retrBrightBySchedule[2]);
    }

    if (starFixDump)
        std::fprintf(stderr,
            "[STARFIX] covered=%d licensedRegions=%ld/%d "
            "signatures=%ld licensedCenters=%ld substitutedSamples=%ld "
            "blackToBlack=%d extendedCenters=%ld addedSamples=%ld "
            "addedExactN=%ld addedExactMean=%.3f addedExactMax=%.3f "
            "carrierRunVetoes=%ld splitZeroSamples=%ld "
            "anchoredReturnBlocked=%ld sceneStart=%d sceneSplit=%d "
            "usedPrevEvidence=%d usedNextEvidence=%d\n",
            frameHasExactCoverage() ? 1 : 0, starLicensedRegions,
            starRegionsX * starRegionsY, starSignatureCenters,
            starLicensedCenters, starSubstitutedSamples,
            1, starExtendedCenters,
            starAddedSamples, starAddedExactN,
            starAddedExactN > 0
                ? starAddedExactSum / (double)starAddedExactN : 0.0,
            starAddedExactMax, starCarrierRunVetoes, starSplitZeroSamples,
            starAnchoredReturnBlocked, isSceneStart ? 1 : 0,
            hasSceneSplit ? 1 : 0, starUsedPrevEvidence ? 1 : 0,
            starUsedNextEvidence ? 1 : 0);

    if (DsRefProbe::on())
        g_dsRefProbe.flush();

    if (CcRefProbe::on())
        g_ccRefProbe.flush();

    if (retHullProbeOn && rhN > 0) {
        std::fprintf(stderr,
            "[RETHULL] fired n=%ld  preHull %.4f  postHull %.4f IRE | "
            "binding n=%ld (%.2f%%)  preHull %.4f  postHull %.4f IRE  "
            "verdict=%s\n",
            rhN, rhPre / rhN, rhPost / rhN,
            rhBindN, 100.0 * (double)rhBindN / (double)rhN,
            rhBindN ? rhBindPre / rhBindN : 0.0,
            rhBindN ? rhBindPost / rhBindN : 0.0,
            (rhBindN && rhBindPost > rhBindPre) ? "HULL HARMFUL"
                : (rhBindN && rhBindPost < rhBindPre) ? "hull helps"
                : "inert");
    }
    if (yCandOn && hfPickN > 0) {
        std::fprintf(stderr,
            "[HFPROXY] n=%ld  maxHF-is-truth-best %.1f%% (chance %.1f%%)  "
            "maxHF-pick %.4f IRE  elected %.4f IRE  oracle %.4f IRE  "
            "verdict=%s\n",
            hfPickN, 100.0 * (double)hfPickHit / (double)hfPickN,
            100.0 * hfChanceSum / (double)hfPickN,
            hfPickErr / hfPickN,
            electN ? electErr / electN : 0.0,
            electN ? oracleErr / electN : 0.0,
            (electN && hfPickErr / hfPickN < electErr / electN)
                ? "HF beats the election" : "HF does not beat the election");
    }
    if (yCandOn && ractN > 0) {
        std::fprintf(stderr,
            "[YRET] delivering n=%ld (%.1f%% of graded)  elected %.4f  "
            "return %.4f  comb %.4f  retracted %.4f IRE  "
            "returnBetterThanComb %.1f%%  electedPos %.3f "
            "(0=comb 1=return)\n",
            ractN, 100.0 * (double)ractN / (double)std::max(1L, electN),
            ractElect / ractN, ractRet / ractN, ractComb / ractN,
            ractRetrN ? ractRetr / ractRetrN : 0.0,
            100.0 * (double)ractRetBetter / (double)ractN,
            ractPosN ? ractPos / ractPosN : 0.0);
        std::fprintf(stderr,
            "[YRETPROX] delivering n=%ld  proximity01(return) %.4f  "
            "evidence %.4f  effective(v1 prox*ev) %.6f IRE\n",
            ractProxN,
            ractProxN ? ractProxRet / ractProxN : 0.0,
            ractProxN ? (ractEvRet / ractProxN) * invIreScale : 0.0,
            ractProxN ? (ractProxRet / ractProxN) *
                        (ractEvRet / ractProxN) * invIreScale : 0.0);
    }
    if (yCandOn && electN > 0) {
        static const char *pn[5] = {"comb","retracted","?2","oneD","returned"};
        std::fprintf(stderr,
            "[YCAND] n=%ld  elected %.3f IRE  ORACLE %.3f IRE  "
            "headroom %.1f%%\n", electN, electErr / electN,
            oracleErr / electN,
            (electErr > 0.0)
                ? 100.0 * (electErr - oracleErr) / electErr : 0.0);
        for (int k = 0; k < 5; ++k) {
            if (!candN[k]) continue;
            std::fprintf(stderr,
                "         %-9s fid %.3f IRE (n=%ld)  slope %.3f  "
                "oracle-wins %.1f%%\n",
                pn[k], candErr[k] / candN[k], candN[k],
                (candSlopeT[k] > 0.0) ? candSlopeY[k] / candSlopeT[k] : 0.0,
                100.0 * oracleWin[k] / electN);
        }
    }

    // ---- LDCD_PROBE_YCERT: grade emitted Y against CERTIFIED LUMA ----
    // User direction (2026-08-01): "use the certified/covered truth to compare
    // excess sharpening. We want the sharpening, but not to the point of
    // disruption of color."  On covered def lines the conservation fact
    // Ltrue = raw - exact IS the luma, at full band, per sample.  So the
    // witness's sharpening can be graded against truth instead of against
    // another decoder's convention:
    //   fidelity   mean |Y - Ltrue| over the line (IRE)
    //   slope      mean |dY| / mean |dLtrue| at strong truth edges
    //              (1.0 = matches truth; >1 = sharpened PAST the source)
    //   overshoot  mean of (|Y-Ltrue| at edge-adjacent samples) in IRE, the
    //              part that must leave through chroma (chroma = raw - Y)
    //   band       fraction of |Y - Ltrue| energy above 1.3 MHz -- excess that
    //              is not even expressible as a chroma correction
    // Measurement only; -t 1 for a stable read.
    {
        static const bool yCertOn =
            std::getenv("LDCD_PROBE_YCERT") != nullptr;
        // NOTE: does NOT use certifiedDefLine() -- that helper is gated on
        // the certified family, so with LDCD_CERT_1D=0 (the held-out
        // configuration this probe needs) it reports no def lines at all.
        // Read the exact plane directly: truth exists independently of
        // whether the decoder is allowed to consume it.
        if (yCertOn && exactCarrier_flat.size() ==
                (size_t)frameHeight * videoParameters.fieldWidth) {
            double fSum = 0.0; long fN = 0;
            double dY = 0.0, dT = 0.0; long dN = 0;
            double ovSum = 0.0; long ovN = 0;
            double eLo = 0.0, eHi = 0.0;
            for (int line = firstLine; line < lastLine; ++line) {
                const float *ex = exactCarrierRow(line);
                if (!ex) continue;
                bool anyExact = false;
                for (int xi = 0; xi < width && !anyExact; ++xi)
                    anyExact = std::isfinite(ex[left + xi]);
                if (!anyExact) continue;
                const quint16 *raw = rawbuffer.data() +
                    static_cast<size_t>(line) * videoParameters.fieldWidth;
                const double *Yr = componentFrame->y(line);
                std::vector<double> Lt(width,
                    std::numeric_limits<double>::quiet_NaN());
                for (int xi = 0; xi < width; ++xi) {
                    const float e = ex[left + xi];
                    if (std::isfinite(e))
                        Lt[xi] = (double)raw[left + xi] - (double)e;
                }
                for (int xi = 1; xi < width - 1; ++xi) {
                    if (!std::isfinite(Lt[xi])) continue;
                    const double err = (Yr[left + xi] - Lt[xi]) * invIreScale;
                    fSum += std::fabs(err); ++fN;
                    if (!std::isfinite(Lt[xi - 1]) ||
                        !std::isfinite(Lt[xi + 1])) continue;
                    const double gT =
                        std::fabs(Lt[xi + 1] - Lt[xi - 1]) * invIreScale;
                    const double gY = std::fabs(Yr[left + xi + 1] -
                                                Yr[left + xi - 1]) * invIreScale;
                    if (gT > 8.0) {          // a real luma edge in TRUTH
                        dY += gY; dT += gT; ++dN;
                        ovSum += std::fabs(err); ++ovN;
                    }
                    // crude LF/HF split of the error at the carrier scale
                    const double e0 = (Yr[left + xi] - Lt[xi]) * invIreScale;
                    const double eM = 0.5 *
                        (((std::isfinite(Lt[xi - 1]))
                            ? (Yr[left + xi - 1] - Lt[xi - 1]) : 0.0) +
                         ((std::isfinite(Lt[xi + 1]))
                            ? (Yr[left + xi + 1] - Lt[xi + 1]) : 0.0)) *
                        invIreScale;
                    eLo += eM * eM;
                    eHi += (e0 - eM) * (e0 - eM);
                }
            }
            // ---- CHROMA quality against the CERTIFIED CARRIER ----
            // User (2026-08-01): "Chroma side cost should look to the
            // certified carrier. It's sparse, but nothing else is in the same
            // league."  At a covered sample the TRUE carrier is exact, and the
            // render's implied carrier is (raw - Y) -- the exact complement of
            // emitted luma.  Both are composite-domain carriers, so their
            // 4fSC quadrature envelopes are directly comparable:
            //   env(x) = hypot(c[x], c[x+1])     (adjacent samples are 90 deg)
            // Reported (all truth-referenced, no bucket anywhere):
            //   amp     mean envRender / mean envTruth at strong truth chroma
            //           -- SATURATION fidelity against fact (1.0 = correct)
            //   width   10-90% transition width of each envelope at truth
            //           chroma edges -- the width question, re-asked against
            //           truth after the bucket-referenced version was
            //           falsified by the user
            //   err     mean |envRender - envTruth| in IRE
            {
                double aR = 0.0, aT = 0.0; long aN = 0;
                double eSum = 0.0; long eN = 0;
                std::vector<double> wR, wT;
                for (int line = firstLine; line < lastLine; ++line) {
                    const float *ex = exactCarrierRow(line);
                    if (!ex) continue;
                    const quint16 *raw = rawbuffer.data() +
                        static_cast<size_t>(line) * videoParameters.fieldWidth;
                    const double *Yr = componentFrame->y(line);
                    std::vector<double> eT(width, std::nan("")),
                                        eR(width, std::nan(""));
                    for (int xi = 0; xi + 1 < width; ++xi) {
                        const float e0 = ex[left + xi], e1 = ex[left + xi + 1];
                        if (!std::isfinite(e0) || !std::isfinite(e1)) continue;
                        eT[xi] = std::hypot((double)e0, (double)e1) *
                                 invIreScale;
                        const double r0 = (double)raw[left + xi] -
                                          Yr[left + xi];
                        const double r1 = (double)raw[left + xi + 1] -
                                          Yr[left + xi + 1];
                        eR[xi] = std::hypot(r0, r1) * invIreScale;
                    }
                    for (int xi = 0; xi < width; ++xi) {
                        if (!std::isfinite(eT[xi]) || !std::isfinite(eR[xi]))
                            continue;
                        eSum += std::fabs(eR[xi] - eT[xi]); ++eN;
                        if (eT[xi] > 10.0) {
                            aR += eR[xi]; aT += eT[xi]; ++aN;
                        }
                    }
                    // Transition widths at truth chroma edges.
                    auto widthAt = [&](const std::vector<double> &v,
                                       int xi) -> double {
                        const int lo = xi - 12, hi = xi + 12;
                        if (lo < 0 || hi >= width) return std::nan("");
                        double a = 0, b = 0; int na = 0, nb = 0;
                        for (int k = lo; k < lo + 4; ++k) {
                            if (!std::isfinite(v[k])) return std::nan("");
                            a += v[k]; ++na;
                        }
                        for (int k = hi - 3; k <= hi; ++k) {
                            if (!std::isfinite(v[k])) return std::nan("");
                            b += v[k]; ++nb;
                        }
                        a /= na; b /= nb;
                        if (std::fabs(b - a) < 12.0) return std::nan("");
                        const double t1 = std::min(a, b) +
                            0.1 * std::fabs(b - a);
                        const double t2 = std::min(a, b) +
                            0.9 * std::fabs(b - a);
                        int first = -1, last = -1;
                        for (int k = lo; k <= hi; ++k) {
                            if (!std::isfinite(v[k])) return std::nan("");
                            if (v[k] >= t1 && v[k] <= t2) {
                                if (first < 0) first = k;
                                last = k;
                            }
                        }
                        return (first >= 0 && last > first)
                            ? (double)(last - first) : std::nan("");
                    };
                    for (int xi = 13; xi + 13 < width; ++xi) {
                        if (!std::isfinite(eT[xi]) ||
                            !std::isfinite(eT[xi + 1])) continue;
                        if (std::fabs(eT[xi + 1] - eT[xi]) < 4.0) continue;
                        const double wt = widthAt(eT, xi);
                        const double wr = widthAt(eR, xi);
                        if (std::isfinite(wt) && std::isfinite(wr)) {
                            wT.push_back(wt); wR.push_back(wr);
                        }
                    }
                }
                auto mean = [](const std::vector<double> &v) {
                    if (v.empty()) return 0.0;
                    double s = 0.0; for (double x : v) s += x;
                    return s / v.size();
                };
                std::fprintf(stderr,
                    "[CCERT] n=%ld env-err %.2f IRE | strong n=%ld amp "
                    "render/truth %.3f | edges n=%zu width truth %.2f "
                    "render %.2f (x%.2f)\n",
                    eN, eN ? eSum / eN : 0.0, aN,
                    (aT > 0.0) ? aR / aT : 0.0, wT.size(),
                    mean(wT), mean(wR),
                    (mean(wT) > 0.0) ? mean(wR) / mean(wT) : 0.0);
            }

            std::fprintf(stderr,
                "[YCERT] n=%ld fidelity |Y-Ltrue| %.2f IRE | edges n=%ld "
                "slope Y/truth %.3f | edge-err %.2f IRE | HF share %.1f%%\n",
                fN, fN ? fSum / fN : 0.0, dN,
                (dT > 0.0) ? dY / dT : 0.0,
                ovN ? ovSum / ovN : 0.0,
                (eLo + eHi > 0.0) ? 100.0 * eHi / (eLo + eHi) : 0.0);
        }
    }
}
// Lurch preconditioner for the coarse luma prior.
//
// The legal 4-sample means cancel carrier, but they are boxcars: a luma step
// smears across the four windows that straddle it, and every pre-2D consumer
// of the coarse prior (the carrier fit, the witness, the patch gates) then
// sees step energy stranded in the carrier band.  Membership movement between
// adjacent windows,
//
//     D[s] = M[s+1] - M[s] = (raw[s+4] - raw[s]) / 4,
//
// compares samples of identical carrier phase, so a luma step produces a
// same-sign run of D across the straddling windows — phase-invariant — while
// a chroma envelope edge alternates sign window to window and is rejected by
// the run test.  Where a step is confirmed, the prior takes the nearest
// same-side window mean instead of the boxcar ramp: the transition lands at
// one column instead of four.  One steep transition per aperture, by design.
//
// `prior` is blended in place; `gateOut` (optional) reports per-pixel
// sharpening activity so a consumer can stand down its own edge correction.
// (LurchStepRun itself lives in comb.h -- it is production data now.)

// Detect luma step runs in a coarse mean sequence -- the difference facts
// that own HF (a legal carrier is aperture-invariant, so a same-sign run of
// D across straddling windows is a luma step; a chroma envelope edge
// alternates sign and is rejected). Gates are stored at UNIT gain; consumers
// scale (clamp(gate*gain,0,1) reproduces any detection-time gain exactly).
// Canonical runs on the shared aperture pool come from buildLurchStepRuns();
// this stays callable directly for OTHER mean sequences (e.g. the carrier
// fit's winFloor), which are different quantities, not duplication.
static void detectLurchSteps(const double *means, int meanCount,
                             double irescale, double invIreScale,
                             std::vector<LurchStepRun> &runs)
{
    runs.clear();
    if (!means || meanCount < 6)
        return;

    const auto smoothStep01 = [](double t) {
        t = std::clamp(t, 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    };

    // Per-window movement floor: a confirmed step of >= ~1.2 IRE moves each
    // straddling window by >= ~0.3 IRE.
    const double dThreshSamples = 0.30 * irescale;
    const int dCount = meanCount - 1;

    int s = 0;
    while (s < dCount) {
        const double d0 = means[s + 1] - means[s];
        if (std::fabs(d0) < dThreshSamples) {
            ++s;
            continue;
        }

        const bool positive = d0 > 0.0;
        const int a = s;
        int b = s;
        while (b + 1 < dCount) {
            const double dn = means[b + 2] - means[b + 1];
            if (std::fabs(dn) < dThreshSamples || (dn > 0.0) != positive)
                break;
            ++b;
        }
        s = b + 1;

        // An ideal step moves exactly four windows; allow slack for noise
        // and local gradient, but a long run is a ramp, not a step.
        const int runLength = b - a + 1;
        if (runLength > 6)
            continue;

        const double stepSamples =
            means[std::min(b + 1, meanCount - 1)] - means[a];
        const double stepIRE = std::fabs(stepSamples) * invIreScale;
        const double gate = smoothStep01((stepIRE - 1.25) / 2.75);
        if (gate <= 0.0)
            continue;

        // Amplitude-weighted centroid of |D| locates the edge even when the
        // threshold trimmed the run asymmetrically (an ideal step's run is
        // s in [e-4, e-1], centroid e-2.5).
        double wSum = 0.0;
        double wPos = 0.0;
        for (int k = a; k <= b; ++k) {
            const double w = std::fabs(means[k + 1] - means[k]);
            wSum += w;
            wPos += w * (double)k;
        }
        const double centroid =
            (wSum > 1e-12) ? (wPos / wSum) : 0.5 * (double)(a + b);

        LurchStepRun run;
        run.a = a;
        run.b = b;
        run.edge = centroid + 2.5;
        run.stepSamples = stepSamples;
        run.stepAbsIRE = stepIRE;
        run.gate = gate;
        runs.push_back(run);
    }

    if (runs.empty())
        return;

    // Ringing suppression.  Sharp analog edges overshoot, which reverses the
    // membership movement and shows up as a smaller opposite run right beside
    // the true step.  Snapping to those fragments puts line-varying plateaus
    // into raw - prior at exactly the edges this pass exists to clean — the
    // fringes survive every comb stage because the contamination is upstream
    // of all of them.  A run within a few windows of a materially stronger
    // run is overshoot, not a second edge.
    for (size_t i = 0; i < runs.size(); ++i) {
        for (size_t j = 0; j < runs.size(); ++j) {
            if (i == j)
                continue;
            const int gap = (runs[i].a > runs[j].b)
                ? runs[i].a - runs[j].b
                : runs[j].a - runs[i].b;
            if (gap <= 3 && runs[j].stepAbsIRE >= 2.5 * runs[i].stepAbsIRE) {
                runs[i].suppressed = true;
                break;
            }
        }
    }
}

// Sample-resolution step detector on CERTIFIED luma (Lhat = raw - exact,
// covered lines only). The aperture detector above fuses same-sign corners
// closer than ~4 px and lets ringing suppression eat close opposite pairs
// -- the one-corner-per-4-px limit. Lhat needs no aperture: it is already
// carrier-free by conservation, so steps resolve at sample resolution and
// the limit does not exist here. Detection thresholds mirror the aperture
// detector's IRE scale; the per-sample difference threshold is higher
// (0.60 vs 0.30 IRE) because samples do not enjoy the window's 4x
// averaging. NaN-tolerant: a NaN sample ends any run.
static void detectCertifiedLurchSteps(const double *lhat, int count,
                                      double irescale, double invIreScale,
                                      std::vector<LurchStepRun> &runs)
{
    runs.clear();
    if (!lhat || count < 6)
        return;

    const auto smoothStep01 = [](double t) {
        t = std::clamp(t, 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    };

    const double dThresh = 0.60 * irescale;
    const int dCount = count - 1;

    int s = 0;
    while (s < dCount) {
        if (!std::isfinite(lhat[s]) || !std::isfinite(lhat[s + 1])) {
            ++s;
            continue;
        }
        const double d0 = lhat[s + 1] - lhat[s];
        if (std::fabs(d0) < dThresh) {
            ++s;
            continue;
        }
        const bool positive = d0 > 0.0;
        const int a = s;
        int b = s;
        while (b + 1 < dCount) {
            if (!std::isfinite(lhat[b + 2])) break;
            const double dn = lhat[b + 2] - lhat[b + 1];
            if (std::fabs(dn) < dThresh || (dn > 0.0) != positive)
                break;
            ++b;
        }
        s = b + 1;

        // A sample-resolution step spans 1-2 differences; longer is a ramp.
        const int runLength = b - a + 1;
        if (runLength > 3)
            continue;

        const double stepSamples = lhat[b + 1] - lhat[a];
        const double stepIRE = std::fabs(stepSamples) * invIreScale;
        const double gate = smoothStep01((stepIRE - 1.25) / 2.75);
        if (gate <= 0.0)
            continue;

        double wSum = 0.0, wPos = 0.0;
        for (int k = a; k <= b; ++k) {
            const double w = std::fabs(lhat[k + 1] - lhat[k]);
            wSum += w;
            wPos += w * (double)k;
        }
        const double centroid =
            (wSum > 1e-12) ? (wPos / wSum) : 0.5 * (double)(a + b);

        LurchStepRun run;
        run.a = a;
        run.b = b;
        run.edge = centroid + 0.5;   // d[k] sits between samples k, k+1
        run.stepSamples = stepSamples;
        run.stepAbsIRE = stepIRE;
        run.gate = gate;
        run.certified = true;
        runs.push_back(run);
    }

    // Same ringing rule as the aperture detector, in sample units.
    for (size_t i = 0; i < runs.size(); ++i) {
        for (size_t j = 0; j < runs.size(); ++j) {
            if (i == j) continue;
            const int gap = (runs[i].a > runs[j].b)
                ? runs[i].a - runs[j].b
                : runs[j].a - runs[i].b;
            if (gap <= 3 && runs[j].stepAbsIRE >= 2.5 * runs[i].stepAbsIRE) {
                runs[i].suppressed = true;
                break;
            }
        }
    }
}

// Fill the canonical per-line lurch run lists from the shared aperture pool:
// ONE detection per line per frame, unit gain, meanCount = width-3 (the real
// aperture starts). Every consumer -- the witness coarse-sharpener, the edge
// probes, and the coming 2D threshold work -- reads these instead of privately
// re-running the scan. Runs on every path from split1D, right after the pool
// itself is built; O(width) per line, so the default path pays noise.
void Comb::FrameBuffer::buildLurchStepRuns()
{
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int width     = right - left;

    if ((int)lurchStepRuns.size() < lastLine)
        lurchStepRuns.resize(lastLine);
    for (auto &rowRuns : lurchStepRuns)
        rowRuns.clear();

    if (width < 8 || firstLine >= lastLine || lockedApertureMean_flat.empty())
        return;

    // FALSIFIED -- do not re-propose: SUB-PIXEL EDGE REFINEMENT FROM THE
    // DISCARDED HALF-SAMPLE TERM.
    //
    // The moving coarse registers to integer xi by AVERAGING the two
    // apertures whose centres straddle xi at +-0.5, so only their sum
    // survives; their difference is a carrier-free luma gradient that is
    // thrown away. The idea was to recover it as a sub-pixel correction to
    // the detected edge.
    //
    // Measured against certified luma with LDCD_PROBE_LURCHVET, using only
    // run-time-available quantities, over 9 scenes on 4 sources
    // (n = 180,485 steps above 10 IRE):
    //
    //   scene       p50 placement err change     native slope
    //   cube            -14.6%                      -2.14
    //   s2x10            -1.2%                      -0.48
    //   vol@520          +5.4%                      -0.18
    //   ggv@18k          +7.7%                      +0.05
    //   ggv@1.3k         +9.3%                      -0.08
    //   beach           +13.8%                      +0.19
    //   s1x11@45k       +17.5%                      -0.81
    //   s1x11@60k       +17.6%                      -5.06
    //   POOLED           +7.6%                      -0.23
    //
    // The relationship is not a physical constant: the slope CHANGES SIGN
    // between sources, and on GGV -- 150k of those samples -- it is
    // indistinguishable from zero. A coefficient fitted on the cube (-1.8)
    // makes placement WORSE on eight of nine scenes. No fixed coefficient can
    // exist, and a per-scene fit would be calibration, which this project
    // does not do. The apparent effect was one shot's worth of frames
    // mistaken for a mechanism.
    //
    // What DID survive is the census that killed it: the detector's
    // reliability curve (blind below ~2.5 IRE, ~1 px at 5-10, 0.7-0.8 px
    // above 10, 15-18% of runs with no certified counterpart), which is
    // stable across all four sources and is the basis for restricting where
    // these runs are allowed to act.

    for (int line = firstLine; line < lastLine; ++line) {
        const double *apMean = lockedApertureMean_line(line);
        if (!apMean) continue;
        detectLurchSteps(apMean, width - 3, irescale, invIreScale,
                         lurchStepRuns[line]);
    }

    // CERTIFIED LUMA FACTS (user, 2026-07-29: "Lurch: yes, this is a big
    // priority, the 1 corner per 4 pixel limit can be broken"). On covered
    // lines Lhat = raw - exact is per-sample certified luma; its steps are
    // facts, consumed as facts (never calibration). Three parity-lawful
    // moves -- the parity law says exact consumption must reach both
    // parities together, and it has bitten twice this arc:
    //   1. REPOSITION: a covered line's aperture run within tolerance of a
    //      certified step takes the certified sub-sample edge.
    //   2. (in corroborateLurchEdges) comp lines bracketed by two
    //      certified partners snap to their MEAN -- the certified
    //      interpolation -- instead of median-of-three.
    //   3. INSERTION, always as a vertical TRIPLE: a certified step the
    //      aperture missed on line L, matched by a certified step on
    //      L+2, inserts runs on L, L+1 (interpolated) and L+2 together --
    //      corner density rises on both parities at once, never on one.
    // A/B escape LDCD_LURCH_CERT=0; default ON. Uncovered frames untouched.
    static const bool lurchCert = []{
        const char *e = std::getenv("LDCD_LURCH_CERT");
        return !(e && std::atoi(e) == 0);
    }();
    if (lurchCert && frameHasExactCoverage()) {
        constexpr double kCertMatchPx = 1.5;
        std::vector<std::vector<LurchStepRun>> certMissed(lastLine);
        // Pending common-mode corrections per (line, run index): a comp
        // run bracketed by two covered lines receives a delta from each;
        // they must AVERAGE, not stack.
        std::vector<std::vector<std::pair<double, int>>> pend(lastLine);
        std::vector<double> lhat(width);
        std::vector<LurchStepRun> certRuns;
        // Detector vetting census (LDCD_PROBE_LURCHVET=1). READ-ONLY.
        // A covered line carries BOTH the aperture means the detector sees
        // and the certified luma saying what is actually there, so the
        // detector can be scored against truth instead of against its own
        // math -- which was impossible before the exact channel existed.
        // Scores the detection as it stood BEFORE any certified
        // repositioning. Alters nothing: not truth, not the runs, not
        // conduct.
        static const bool vetProbe =
            std::getenv("LDCD_PROBE_LURCHVET") != nullptr;
        // Stratified by certified step magnitude, because the two detectors
        // are NOT equally sensitive: this one trips at 0.60 IRE per sample on
        // noise-free truth, while the aperture detector works on 4-sample
        // means with a ~1.2 IRE step floor. An unstratified "miss" count
        // mostly measures that design gap, not detector error. Position error
        // is measured to the NEAREST aperture run of the same sign at any
        // distance, so the 1.5 px match window does not truncate it.
        std::vector<char> vetLine(lastLine, 0);
        std::vector<std::vector<LurchStepRun>> certAll(lastLine);
        struct VetBin { long cert = 0, found = 0;
            std::vector<double> err, res, grad; };
        static constexpr double kVetEdges[5] = { 0.6, 1.2, 2.5, 5.0, 10.0 };
        std::vector<VetBin> vetBin(6);
        std::vector<double> vetDelta;
        for (int line = firstLine; line < lastLine; ++line) {
            const float *ex = exactCarrierRow(line);
            if (!ex) continue;
            const quint16 *rawLine = rawbuffer.data()
                + static_cast<size_t>(line) * videoParameters.fieldWidth;
            int nFinite = 0;
            for (int xi = 0; xi < width; ++xi) {
                const float e = ex[left + xi];
                if (std::isfinite(e)) {
                    lhat[xi] = static_cast<double>(rawLine[left + xi]) -
                               static_cast<double>(e);
                    ++nFinite;
                } else {
                    lhat[xi] = std::numeric_limits<double>::quiet_NaN();
                }
            }
            if (nFinite < 16) continue;
            detectCertifiedLurchSteps(lhat.data(), width, irescale,
                                      invIreScale, certRuns);
            if (vetProbe) { vetLine[line] = 1; certAll[line] = certRuns; }
            for (const LurchStepRun &cr : certRuns) {
                if (cr.suppressed) continue;
                int vetB = 0;
                if (vetProbe) {
                    while (vetB < 5 && cr.stepAbsIRE >= kVetEdges[vetB])
                        ++vetB;
                    ++vetBin[vetB].cert;
                    // Nearest same-sign aperture run at ANY distance.
                    double best = -1.0, bestSigned = 0.0;
                    for (const LurchStepRun &ar : lurchStepRuns[line]) {
                        if (ar.suppressed) continue;
                        if ((ar.stepSamples > 0.0) != (cr.stepSamples > 0.0))
                            continue;
                        const double d = std::fabs(ar.edge - cr.edge);
                        if (best < 0.0 || d < best) {
                            best = d;
                            bestSigned = cr.edge - ar.edge;
                        }
                    }
                    if (best >= 0.0) {
                        ++vetBin[vetB].found;
                        vetBin[vetB].err.push_back(best);
                        // The term the moving coarse DISCARDS. sharp[xi]
                        // averages the two apertures whose centres straddle
                        // xi at +-0.5; only their sum survives. Their
                        // DIFFERENCE is a carrier-free luma gradient at
                        // half-sample offset. Does it predict where the
                        // detector's placement is wrong?
                        // OPERATIONAL quantities only: the gradient is
                        // located from the APERTURE run's own edge and
                        // normalised by its own step height, because that is
                        // all a live correction can see. Locating from
                        // cr.edge or scaling by cr.stepAbsIRE would fit a
                        // coefficient on certified data the implementation
                        // does not have.
                        const double *ap = lockedApertureMean_line(line);
                        const LurchStepRun *nr = nullptr;
                        for (const LurchStepRun &ar : lurchStepRuns[line]) {
                            if (ar.suppressed) continue;
                            if ((ar.stepSamples > 0.0) != (cr.stepSamples > 0.0))
                                continue;
                            if (std::fabs(std::fabs(ar.edge - cr.edge) - best)
                                < 1e-9) { nr = &ar; break; }
                        }
                        const int si = nr ? (int)std::lround(nr->edge) : -1;
                        if (best <= 3.0 && ap && nr && si >= 0
                            && si + 1 < width - 3 && nr->stepAbsIRE > 0.0) {
                            const double g = (ap[si + 1] - ap[si]) *
                                             invIreScale;
                            vetBin[vetB].res.push_back(bestSigned);
                            vetBin[vetB].grad.push_back(g / nr->stepAbsIRE);
                        }
                    }
                }
                bool matched = false;
                for (LurchStepRun &ar : lurchStepRuns[line]) {
                    if (ar.suppressed) continue;
                    if ((ar.stepSamples > 0.0) != (cr.stepSamples > 0.0))
                        continue;
                    if (std::fabs(ar.edge - cr.edge) > kCertMatchPx)
                        continue;
                    // COHERENT repositioning (the parity law's third
                    // lesson this arc, caught by the straightness dump):
                    // moving only the covered line to truth breaks
                    // line-to-line consistency, because adjacent aperture
                    // detections share systematic bias -- truth on one
                    // parity + bias on the other IS alternation (scatter
                    // 0.384 -> 0.437 px, strut lineAlt +3%). So the
                    // certified CORRECTION delta propagates to the
                    // vertically-matched comp runs above and below at the
                    // same time: every line moves together, common-mode,
                    // and the shared detector bias is what the delta
                    // cancels.
                    const double delta = cr.edge - ar.edge;
                    if (vetProbe) vetDelta.push_back(std::fabs(delta));
                    ar.edge = cr.edge;          // the certified position
                    ar.gate = std::max(ar.gate, cr.gate);
                    ar.certified = true;
                    for (int nl = line - 1; nl <= line + 1; nl += 2) {
                        if (nl < firstLine || nl >= lastLine) continue;
                        auto &rowRuns = lurchStepRuns[nl];
                        if (pend[nl].size() < rowRuns.size())
                            pend[nl].resize(rowRuns.size(), {0.0, 0});
                        for (size_t ri = 0; ri < rowRuns.size(); ++ri) {
                            LurchStepRun &mr = rowRuns[ri];
                            if (mr.suppressed || mr.certified) continue;
                            if ((mr.stepSamples > 0.0) !=
                                (cr.stepSamples > 0.0)) continue;
                            if (std::fabs(mr.edge - (cr.edge - delta)) >
                                kCertMatchPx) continue;
                            pend[nl][ri].first += delta;
                            pend[nl][ri].second += 1;
                            break;
                        }
                    }
                    matched = true;
                    break;
                }
                if (!matched)
                    certMissed[line].push_back(cr);
            }
        }
        if (vetProbe) {
            long vetAp = 0, vetFalse = 0;
            // Stratified by the APERTURE run's own step height -- the
            // quantity a restriction can key on at run time. Asks: of the
            // runs the detector emits at this size, what fraction has NO
            // certified step of the same sign within 1.5 px? On a covered
            // line, truth says nothing is there.
            long apCert[6] = {0}, apTot[6] = {0};
            // Is a "false" run genuinely spurious, or is the CLAIM FLAG the
            // artefact? ar.certified is set only on the first match and only
            // within 1.5 px, so several runs on one wide feature leave all
            // but one flagged false BY CONSTRUCTION. Measure distance to the
            // nearest same-sign certified step regardless of the flag, and
            // the run's own width, so the three readings separate.
            std::vector<double> apNear[6];
            std::vector<int> apWide[6];
            std::vector<double> apCarOK[6], apCarBad[6];
            for (int line = firstLine; line < lastLine; ++line) {
                if (!vetLine[line]) continue;
                for (const LurchStepRun &ar : lurchStepRuns[line]) {
                    if (ar.suppressed) continue;
                    ++vetAp;
                    if (!ar.certified) ++vetFalse;
                    int ab = 0;
                    while (ab < 5 && ar.stepAbsIRE >= kVetEdges[ab]) ++ab;
                    ++apTot[ab];
                    if (ar.certified) ++apCert[ab];
                    double nd = -1.0;
                    for (const LurchStepRun &cr : certAll[line]) {
                        if (cr.suppressed) continue;
                        if ((cr.stepSamples > 0.0) != (ar.stepSamples > 0.0))
                            continue;
                        const double d = std::fabs(cr.edge - ar.edge);
                        if (nd < 0.0 || d < nd) nd = d;
                    }
                    if (nd >= 0.0) apNear[ab].push_back(nd);
                    apWide[ab].push_back(ar.b - ar.a);
                    // Characterise the population truth does NOT corroborate.
                    // If those runs sit where the CARRIER is strong, they are
                    // leakage through imperfect four-sample cancellation and
                    // the bandwidth law has a claim on them. If their carrier
                    // looks like everyone else's, they are not a carrier
                    // phenomenon and no bandwidth constraint will find them.
                    const float *ex2 = exactCarrierRow(line);
                    const int xe = (int)std::lround(ar.edge) + 2;
                    if (ex2 && xe >= 0 && xe < width) {
                        const float c = ex2[left + xe];
                        if (std::isfinite(c)) {
                            const double cIRE = std::fabs((double)c) *
                                                invIreScale;
                            if (nd >= 0.0 && nd <= 3.0) apCarOK[ab].push_back(cIRE);
                            else                        apCarBad[ab].push_back(cIRE);
                        }
                    }
                }
            }
            static const char *kBinName[6] = {
                "  <0.6", "0.6-1.2", "1.2-2.5", "2.5-5.0", "5.0-10 ", " >10  " };
            std::fprintf(stderr,
                "[LURCHVET] lines=%ld  aperture runs=%ld  uncertified=%ld "
                "(%.1f%%)\n",
                (long)std::count(vetLine.begin(), vetLine.end(), 1),
                vetAp, vetFalse, vetAp ? 100.0 * vetFalse / vetAp : 0.0);
            for (int b = 0; b < 6; ++b) {
                if (!apTot[b]) continue;
                std::sort(apNear[b].begin(), apNear[b].end());
                std::sort(apWide[b].begin(), apWide[b].end());
                const double nmed = apNear[b].empty() ? -1.0
                    : apNear[b][apNear[b].size() / 2];
                const int wmed = apWide[b].empty() ? 0
                    : apWide[b][apWide[b].size() / 2];
                const double w15 = apNear[b].empty() ? 0.0 : 100.0 *
                    std::count_if(apNear[b].begin(), apNear[b].end(),
                        [](double d){ return d <= 1.5; }) / apNear[b].size();
                const double w30 = apNear[b].empty() ? 0.0 : 100.0 *
                    std::count_if(apNear[b].begin(), apNear[b].end(),
                        [](double d){ return d <= 3.0; }) / apNear[b].size();
                // Uncorroborated = no certified step of the same sign within
                // 3 px. Distance-based, NOT the ar.certified claim flag: that
                // flag is set on the first match only, so several runs on one
                // feature leave all but one flagged by construction.
                auto med = [](std::vector<double> &v) {
                    if (v.empty()) return -1.0;
                    std::sort(v.begin(), v.end());
                    return v[v.size() / 2];
                };
                const double carOK = med(apCarOK[b]), carBad = med(apCarBad[b]);
                std::fprintf(stderr,
                    "[LURCHVET]   RUNS %s IRE emitted=%-6ld uncorrob=%.0f%% "
                    "| nearest cert px med=%.2f <=1.5px=%.0f%% <=3px=%.0f%% "
                    "| width med=%d | carrier IRE corrob=%.2f uncorrob=%.2f "
                    "(n=%zu)\n",
                    kBinName[b], apTot[b], 100.0 - w30,
                    nmed, w15, w30, wmed, carOK, carBad, apCarBad[b].size());
            }
            for (int b = 0; b < 6; ++b) {
                VetBin &v = vetBin[b];
                if (!v.cert) continue;
                std::sort(v.err.begin(), v.err.end());
                double m = 0.0;
                for (double d : v.err) m += d;
                if (!v.err.empty()) m /= v.err.size();
                const double p50 = v.err.empty() ? 0.0
                    : v.err[v.err.size() / 2];
                const double p90 = v.err.empty() ? 0.0
                    : v.err[std::min(v.err.size() - 1,
                        (size_t)(0.9 * v.err.size()))];
                // Pearson r between the discarded half-sample gradient and
                // the SIGNED placement residual, over near matches.
                double rP = 0.0, slope = 0.0;
                const size_t nc = v.res.size();
                if (nc >= 32) {
                    double mx = 0, my = 0;
                    for (size_t i = 0; i < nc; ++i) { mx += v.grad[i]; my += v.res[i]; }
                    mx /= nc; my /= nc;
                    double sxy = 0, sxx = 0, syy = 0;
                    for (size_t i = 0; i < nc; ++i) {
                        const double dx = v.grad[i] - mx, dy = v.res[i] - my;
                        sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
                    }
                    if (sxx > 1e-12 && syy > 1e-12) {
                        rP = sxy / std::sqrt(sxx * syy);
                        slope = sxy / sxx;
                    }
                }
                std::fprintf(stderr,
                    "[LURCHVET]   step %s IRE  certified=%-6ld nearest-run "
                    "err px mean=%.2f p50=%.2f p90=%.2f  within1.5px=%.1f%%"
                    "  | subpx n=%-5zu r=%+.3f slope=%+.2f\n",
                    kBinName[b], v.cert, m, p50, p90,
                    v.err.empty() ? 0.0 : 100.0 *
                        std::count_if(v.err.begin(), v.err.end(),
                            [](double d){ return d <= 1.5; }) / v.err.size(),
                    nc, rP, slope);
            }
        }

        // Apply the averaged common-mode corrections.
        for (int line = firstLine; line < lastLine; ++line) {
            auto &rowRuns = lurchStepRuns[line];
            for (size_t ri = 0; ri < pend[line].size() &&
                                ri < rowRuns.size(); ++ri) {
                if (pend[line][ri].second > 0)
                    rowRuns[ri].edge += pend[line][ri].first /
                                        pend[line][ri].second;
            }
        }

        // Insertion as vertical triples (move 3).
        for (int line = firstLine; line + 2 < lastLine; ++line) {
            for (LurchStepRun &r : certMissed[line]) {
                if (r.suppressed) continue;   // reused as consumed marker
                LurchStepRun *o = nullptr;
                for (LurchStepRun &c : certMissed[line + 2]) {
                    if (c.suppressed) continue;
                    if ((c.stepSamples > 0.0) != (r.stepSamples > 0.0))
                        continue;
                    if (std::fabs(c.edge - r.edge) > kCertMatchPx) continue;
                    o = &c;
                    break;
                }
                if (!o) continue;
                // The comp line between them must not already carry an
                // aperture run here (it would double-report the corner).
                bool compHasIt = false;
                for (const LurchStepRun &m : lurchStepRuns[line + 1]) {
                    if (m.suppressed) continue;
                    if (std::fabs(m.edge - 0.5 * (r.edge + o->edge)) <=
                        kCertMatchPx) { compHasIt = true; break; }
                }
                LurchStepRun mid;
                mid.edge = 0.5 * (r.edge + o->edge);
                mid.stepSamples = 0.5 * (r.stepSamples + o->stepSamples);
                mid.stepAbsIRE = 0.5 * (r.stepAbsIRE + o->stepAbsIRE);
                mid.gate = std::min(r.gate, o->gate);
                mid.certified = true;
                mid.a = std::max(0, (int)mid.edge - 2);
                mid.b = std::min(width - 4, (int)mid.edge + 1);
                lurchStepRuns[line].push_back(r);
                if (!compHasIt)
                    lurchStepRuns[line + 1].push_back(mid);
                lurchStepRuns[line + 2].push_back(*o);
                o->suppressed = true;         // consumed
                r.suppressed = true;
            }
        }
    }

    // Disposable per-line edge dump (LDCD_DUMP_LURCH_L0/L1/C0/C1, run -t 1):
    // raw pre-corroboration edges with line phase class, for measuring
    // vertical edge scatter and its phase correlation. Zero cost when unset.
    static const int luL0 = []{ const char *s = std::getenv("LDCD_DUMP_LURCH_L0"); return s ? std::atoi(s) : -1; }();
    static const int luL1 = []{ const char *s = std::getenv("LDCD_DUMP_LURCH_L1"); return s ? std::atoi(s) : -1; }();
    static const int luC0 = []{ const char *s = std::getenv("LDCD_DUMP_LURCH_C0"); return s ? std::atoi(s) : -1; }();
    static const int luC1 = []{ const char *s = std::getenv("LDCD_DUMP_LURCH_C1"); return s ? std::atoi(s) : -1; }();
    if (luL0 >= 0 && luC0 >= 0) {
        for (int line = std::max(firstLine, luL0);
             line < std::min(lastLine, luL1 + 1); ++line) {
            for (const LurchStepRun &run : lurchStepRuns[line]) {
                if (run.edge < luC0 || run.edge > luC1) continue;
                std::fprintf(stderr,
                    "LURCH line=%d flip=%d edge=%.3f stepIRE=%.2f gate=%.2f "
                    "a=%d b=%d sup=%d cert=%d\n",
                    line, carrierLineFlip(line), run.edge,
                    run.stepSamples * invIreScale, run.gate,
                    run.a, run.b, run.suppressed ? 1 : 0,
                    run.certified ? 1 : 0);
            }
        }
    }
}

// See comb.h. The vertical partner step mirrors the election's Y-geometry
// rule: immediately adjacent lines wherever the cadence allows the frame-
// vertical model, same-field +-2 otherwise.
std::vector<LurchStepRun> Comb::FrameBuffer::corroborateLurchEdges(int line) const
{
    std::vector<LurchStepRun> runs = lurchStepRuns_line(line);
    if (runs.empty()) return runs;

    constexpr double kLurchMatchPx = 1.5;
    const int step = carrierFrameVerticalAllowed(line) ? 1 : 2;
    const auto &up = lurchStepRuns_line(line - step);
    const auto &dn = lurchStepRuns_line(line + step);

    auto matchEdge = [&](const std::vector<LurchStepRun> &nbr,
                         const LurchStepRun &run, double &edgeOut,
                         bool &certOut) -> bool {
        double bestD = kLurchMatchPx;
        bool found = false;
        for (const LurchStepRun &o : nbr) {
            if (o.suppressed) continue;
            if ((o.stepSamples > 0.0) != (run.stepSamples > 0.0)) continue;
            const double d = std::fabs(o.edge - run.edge);
            if (d <= bestD) {
                bestD = d; edgeOut = o.edge; certOut = o.certified;
                found = true;
            }
        }
        return found;
    };

    for (LurchStepRun &run : runs) {
        if (run.suppressed) continue;
        double eu, ed;
        bool cu = false, cd = false;
        if (!matchEdge(up, run, eu, cu) || !matchEdge(dn, run, ed, cd))
            continue;                       // no full vertical company
        if (cu && cd && !run.certified) {
            // Both partners carry certified positions: their mean is the
            // certified interpolation, strictly better information than
            // this line's own aperture centroid. The comp line inherits
            // the anchors -- both parities improve together.
            run.edge = 0.5 * (eu + ed);
            continue;
        }
        const double e = run.edge;
        // median of three, by selection
        run.edge = std::max(std::min(eu, e), std::min(std::max(eu, e), ed));
    }
    return runs;
}

// Reverse-engineering stats for the pair disentangle (LDCD_PROBE_DISENT=1).
// Measurement only. The per-class map is the point: the leak's demod hue is
// set by (x mod 4, lineFlip), so the SAME luma edge presents a different
// error per line class -- Y colliding with different conventions along the
// phase sequence. These counters expose that map and the field asymmetry.
namespace {

struct DisentProbe {
    std::mutex mu;
    long nRuns = 0, nNoPartner = 0;
    long partnerUp[2] = {0, 0}, partnerDn[2] = {0, 0};
    // [line parity][lineFlip < 0]: samples and IRE sums.
    long   cn[2][2]   = {{0, 0}, {0, 0}};
    double cAbs[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    double cI[2][2]   = {{0.0, 0.0}, {0.0, 0.0}};
    double cQ[2][2]   = {{0.0, 0.0}, {0.0, 0.0}};
    // Scale law: per-run (step height h, footprint peak |cm|) regression.
    long   nScale = 0;
    double sH = 0.0, sP = 0.0, sHP = 0.0, sHH = 0.0;
    long peakHist[5] = {0, 0, 0, 0, 0};   // <1, 1-2, 2-4, 4-8, >=8 IRE
    // Shape conformance: per-run corr r of measured cm against the
    // anticipated doublet, and amplitude ratio beta where the shape holds.
    long rHist[5] = {0, 0, 0, 0, 0};      // <0, 0-.3, .3-.6, .6-.8, >=.8
    long   nBeta = 0;
    double sBeta = 0.0;

    static bool on()
    {
        static const bool v = []{
            const char *s = std::getenv("LDCD_PROBE_DISENT");
            return s && std::atoi(s) != 0;
        }();
        return v;
    }

    void run(int parity, int partnerDelta)
    {
        if (!on()) return;
        std::lock_guard<std::mutex> lk(mu);
        ++nRuns;
        if (partnerDelta < 0) ++partnerUp[parity & 1];
        else                  ++partnerDn[parity & 1];
    }

    void noPartner(long gatedRuns)
    {
        if (!on()) return;
        std::lock_guard<std::mutex> lk(mu);
        nRuns += gatedRuns;
        nNoPartner += gatedRuns;
    }

    void sample(int parity, int flipNeg, double cmIRE, double iIRE, double qIRE)
    {
        if (!on()) return;
        std::lock_guard<std::mutex> lk(mu);
        ++cn[parity & 1][flipNeg & 1];
        cAbs[parity & 1][flipNeg & 1] += cmIRE;
        cI[parity & 1][flipNeg & 1]   += iIRE;
        cQ[parity & 1][flipNeg & 1]   += qIRE;
    }

    void runProfile(double hIRE, double peakIRE, double r, double beta,
                    bool shapeValid)
    {
        if (!on()) return;
        std::lock_guard<std::mutex> lk(mu);
        ++nScale;
        sH += hIRE; sP += peakIRE; sHP += hIRE * peakIRE; sHH += hIRE * hIRE;
        peakHist[peakIRE < 1.0 ? 0 : peakIRE < 2.0 ? 1
               : peakIRE < 4.0 ? 2 : peakIRE < 8.0 ? 3 : 4]++;
        if (shapeValid) {
            rHist[r < 0.0 ? 0 : r < 0.3 ? 1 : r < 0.6 ? 2 : r < 0.8 ? 3 : 4]++;
            if (r >= 0.6) { ++nBeta; sBeta += beta; }
        }
    }

    ~DisentProbe()
    {
        if (!on() || nRuns <= 0) return;
        std::fprintf(stderr,
            "\n[DISENT] runs %ld  noPartner %.1f%%  partner up/dn: "
            "even %ld/%ld  odd %ld/%ld\n",
            nRuns, 100.0 * (double)nNoPartner / (double)nRuns,
            partnerUp[0], partnerDn[0], partnerUp[1], partnerDn[1]);
        for (int p = 0; p < 2; ++p)
            for (int f = 0; f < 2; ++f) {
                if (cn[p][f] <= 0) continue;
                const double inv = 1.0 / (double)cn[p][f];
                std::fprintf(stderr,
                    "[DISENT] class parity=%d flip=%c: n=%ld  |cm| %.2f IRE"
                    "  |I| %.2f  |Q| %.2f\n",
                    p, f ? '-' : '+', cn[p][f],
                    cAbs[p][f] * inv, cI[p][f] * inv, cQ[p][f] * inv);
            }
        if (nScale > 1) {
            const double det = nScale * sHH - sH * sH;
            const double slope = (det > 1e-9)
                ? (nScale * sHP - sH * sP) / det : 0.0;
            const double icept = (sP - slope * sH) / nScale;
            std::fprintf(stderr,
                "[DISENT] scale: n=%ld  peak|cm| = %.3f*h %+.2f IRE  "
                "(h mean %.1f, peak mean %.2f)\n",
                nScale, slope, icept, sH / nScale, sP / nScale);
            std::fprintf(stderr,
                "[DISENT] peak|cm| IRE: <1:%.1f%% 1-2:%.1f%% 2-4:%.1f%% "
                "4-8:%.1f%% >=8:%.1f%%\n",
                100.0 * peakHist[0] / nScale, 100.0 * peakHist[1] / nScale,
                100.0 * peakHist[2] / nScale, 100.0 * peakHist[3] / nScale,
                100.0 * peakHist[4] / nScale);
            long nR = 0;
            for (int i = 0; i < 5; ++i) nR += rHist[i];
            if (nR > 0)
                std::fprintf(stderr,
                    "[DISENT] shape r: <0:%.1f%% 0-.3:%.1f%% .3-.6:%.1f%% "
                    ".6-.8:%.1f%% >=.8:%.1f%%   beta(r>=.6) %.2f (n=%ld)\n",
                    100.0 * rHist[0] / nR, 100.0 * rHist[1] / nR,
                    100.0 * rHist[2] / nR, 100.0 * rHist[3] / nR,
                    100.0 * rHist[4] / nR,
                    nBeta ? sBeta / nBeta : 0.0, nBeta);
        }
    }
};

DisentProbe g_disentProbe;

// Downstream-fate stats (LDCD_PROBE_EDGEFATE=1). Measurement only. Every
// render judged so far was ntsc1d -- pure 1D, no comb ever touched the edge
// bands. This probe asks what the REAL pipeline does at lurch footprints:
// how far 2D moves the carrier off its 1D source there (vs a control of all
// other pixels), how often it effectively passes 1D through (the fallback),
// and what 3D adds. The answer decides WHERE edge evidence should be
// delivered: into the fallback conditioning, into the comb's own gates, or
// into the election.
struct EdgeFateProbe {
    std::mutex mu;
    // [isEdgeFootprint]: pixels, |clp1-clp0| and |clp2-clp1| sums (IRE),
    // and counts of |clp1-clp0| under the pass-through thresholds.
    long   n[2]     = {0, 0};
    double d21[2]   = {0.0, 0.0};
    double d32[2]   = {0.0, 0.0};
    long   nEx[2]   = {0, 0};      // < 0.1 IRE: effectively 1D
    long   nNear[2] = {0, 0};      // < 0.5 IRE
    bool   have3D   = false;

    static bool on()
    {
        static const bool v = []{
            const char *s = std::getenv("LDCD_PROBE_EDGEFATE");
            return s && std::atoi(s) != 0;
        }();
        return v;
    }

    void merge(const long *ln, const double *ld21, const double *ld32,
               const long *lex, const long *lnear, bool dims3)
    {
        std::lock_guard<std::mutex> lk(mu);
        for (int e = 0; e < 2; ++e) {
            n[e] += ln[e]; d21[e] += ld21[e]; d32[e] += ld32[e];
            nEx[e] += lex[e]; nNear[e] += lnear[e];
        }
        have3D = have3D || dims3;
    }

    ~EdgeFateProbe()
    {
        if (!on() || (n[0] + n[1]) <= 0) return;
        static const char *kName[2] = {"control", "edge   "};
        std::fprintf(stderr, "\n");
        for (int e = 1; e >= 0; --e) {
            if (n[e] <= 0) continue;
            const double inv = 1.0 / (double)n[e];
            std::fprintf(stderr,
                "[EDGEFATE] %s n=%ld  |2D-1D| %.3f IRE  1D-passthru "
                "<0.1:%.1f%% <0.5:%.1f%%%s%.3f IRE\n",
                kName[e], n[e], d21[e] * inv,
                100.0 * (double)nEx[e] * inv,
                100.0 * (double)nNear[e] * inv,
                have3D ? "  |3D-2D| " : "  (no 3D) ",
                have3D ? d32[e] * inv : 0.0);
        }
    }
};

EdgeFateProbe g_edgeFate;

// Post-law hull-violation stats (LDCD_PROBE_RETRHULL=1). Measurement only.
// Pass 1 clamps the carrier fit into the residual-consensus feasible range
// (region-pure four-view complements + the rolling witness), but the encoder
// bandwidth law is imposed AFTER that clamp at publication, and its 9-tap
// FIR mixes neighbours -- the published fit can leave the per-sample range
// the clamp enforced. This measures how often and by how much, deciding
// whether re-imposing the hull after the law is load-bearing or insurance.
struct RetrHullProbe {
    std::mutex mu;
    long   n = 0, nOut = 0;
    double sumExcess = 0.0, maxExcess = 0.0, sumWidth = 0.0;

    static bool on()
    {
        static const bool v = []{
            const char *s = std::getenv("LDCD_PROBE_RETRHULL");
            return s && std::atoi(s) != 0;
        }();
        return v;
    }

    void sample(double fit, double lo, double hi, double invIre)
    {
        std::lock_guard<std::mutex> lk(mu);
        ++n;
        sumWidth += (hi - lo) * invIre;
        double ex = 0.0;
        if (fit < lo)      ex = (lo - fit) * invIre;
        else if (fit > hi) ex = (fit - hi) * invIre;
        if (ex > 0.0) {
            ++nOut;
            sumExcess += ex;
            maxExcess = std::max(maxExcess, ex);
        }
    }

    ~RetrHullProbe()
    {
        if (!on() || n <= 0) return;
        std::fprintf(stderr,
            "\n[RETRHULL] n=%ld  outside %.1f%%  excess mean %.3f IRE "
            "(of violators)  max %.2f  hull width mean %.2f IRE\n",
            n, 100.0 * (double)nOut / (double)n,
            nOut ? sumExcess / nOut : 0.0, maxExcess,
            sumWidth / n);
    }
};

RetrHullProbe g_retrHull;

} // namespace

// Pair class-map probe at luma steps. MEASUREMENT ONLY -- writes nothing.
//
// LAW (1D safe retreat, 2026-07-26): the 1D stage may contain no comb,
// blend, or influence from other lines. Downstream 2D is programmed to
// fall back to 1D precisely when its vertical machinery produces errors;
// 1D is the safe retreat, and the fallback hierarchy only works if 1D is
// structurally incapable of vertical error. Adjacent-line data may CONFIRM
// (phase relations, luma contrasts) as verdict inputs; rejection comes
// from the line's own model; no cross-line sample may contribute signal.
//
// P12 briefly subtracted the pair common mode
//     cm = 0.5*(B{raw_n} + B{raw_p}) = Lbar + 0.5*dC
// from the emitted carrier at lurch footprints. That is a 2-tap interfield
// comb inside 1D -- a violation of the law above -- and it injected 0.5*dC
// at vertical colour boundaries. WITHDRAWN. What the measurement itself
// established is kept (and this probe reproduces it on demand): the leak
// is CLASS-INVARIANT in composite -- |cm| and its I/Q split are uniform
// across all four parity x flip cells -- so the field asymmetry seen in
// renders was manufactured by per-line verdict actuators, never present in
// the error. The complement-pair tiling ([+ - - +]; even lines pair down,
// odd pair up, swapping per frame) is confirmed against the grammar here,
// not presumed.
void Comb::FrameBuffer::probeEdgePairClassMap(int line)
{
    if (!DisentProbe::on()) return;

    const double *apMean = lockedApertureMean_line(line);
    if (!apMean) return;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int fullWidth = videoParameters.fieldWidth;
    const int width     = right - left;
    if (width < 16) return;

    const std::vector<LurchStepRun> &runs = lurchStepRuns_line(line);
    long gated = 0;
    for (const LurchStepRun &run : runs)
        if (!run.suppressed && run.gate > 0.0) ++gated;
    if (gated == 0) return;

    // Complement partner: the schedule tiles [+ - - +] down the frame, so
    // exactly one adjacent frame line carries the opposite signed class.
    // The grammar names it; nothing is presumed from the schedule.
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int want = (carrierSignedSampleClass(line, left) + 2) & 3;
    int partner = -1;
    if (line - 1 >= firstLine
            && carrierSignedSampleClass(line - 1, left) == want)
        partner = line - 1;
    else if (line + 1 < lastLine
            && carrierSignedSampleClass(line + 1, left) == want)
        partner = line + 1;
    if (partner < 0) {
        g_disentProbe.noPartner(gated);
        return;
    }

    const auto sstep = [](double t) {
        t = std::clamp(t, 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    };

    const quint16 *rawLine = rawbuffer.data() + size_t(line)    * fullWidth;
    const quint16 *rawP    = rawbuffer.data() + size_t(partner) * fullWidth;
    auto bpAt = [&](const quint16 *rl, int x) {
        const int m2 = std::clamp(x - 2, 0, width - 1);
        const int p2 = std::clamp(x + 2, 0, width - 1);
        return 0.50 * (double)rl[left + x]
             - 0.25 * ((double)rl[left + m2] + (double)rl[left + p2]);
    };

    // Footprint weights: max-combine overlapping runs so each sample is
    // counted once in the class-map stats regardless of run overlap.
    std::vector<double> wAcc(width, 0.0);
    bool any = false;
    for (const LurchStepRun &run : runs) {
        if (run.suppressed || run.gate <= 0.0) continue;
        g_disentProbe.run(line & 1, partner - line);
        const int x0 = (int)std::floor(run.edge) - 6;
        const int x1 = (int)std::ceil(run.edge) + 6;
        for (int x = std::max(0, x0); x <= std::min(width - 1, x1); ++x) {
            const double tIn =
                std::min((double)(x - x0), (double)(x1 - x)) / 3.0;
            const double w = run.gate * sstep(tIn);
            if (w > wAcc[x]) { wAcc[x] = w; any = true; }
        }

        // Per-run error profile against the ANTICIPATED doublet: a chain-
        // sharp (w = 2) ramp with lurch's pinned plateau levels at lurch's
        // edge, through the bandpass's own operator -0.25*D2_2. r says
        // whether the anticipation carries the SHAPE (evidence can predict
        // the waveform) or only a location and magnitude bound; beta says
        // whether lurch's amplitude scales it correctly where the shape
        // holds. peak|cm| vs step height h feeds the scale law.
        const int xa = std::max(0, x0), xb = std::min(width - 1, x1);
        if (xb - xa < 6) continue;
        const double lo = apMean[std::clamp(run.a, 0, width - 1)];
        const double hi = apMean[std::clamp(run.b + 1, 0, width - 1)];
        auto rampAt = [&](double j) {
            return lo + (hi - lo)
                 * std::clamp((j - run.edge) / 2.0 + 0.5, 0.0, 1.0);
        };
        double peak = 0.0, sCC = 0.0, sPP = 0.0, sCP = 0.0;
        for (int x = xa; x <= xb; ++x) {
            const double cm = 0.5 * (bpAt(rawLine, x) + bpAt(rawP, x));
            const double d2 = rampAt((double)(x - 2)) - 2.0 * rampAt((double)x)
                            + rampAt((double)(x + 2));
            const double pred = -0.25 * d2;
            peak = std::max(peak, std::fabs(cm));
            sCC += cm * cm; sPP += pred * pred; sCP += cm * pred;
        }
        const bool shapeValid = sPP > 1e-9 && sCC > 1e-9;
        const double r    = shapeValid ? sCP / std::sqrt(sCC * sPP) : 0.0;
        const double beta = (sPP > 1e-9) ? sCP / sPP : 0.0;
        g_disentProbe.runProfile(run.stepAbsIRE, peak * invIreScale,
                                 r, beta, shapeValid);
    }
    if (!any) return;

    const int parity  = line & 1;
    const int flipNeg = carrierLineFlip(line) < 0 ? 1 : 0;
    for (int x = 0; x < width; ++x) {
        if (wAcc[x] <= 0.0) continue;
        const double cm = 0.5 * (bpAt(rawLine, x) + bpAt(rawP, x));
        const int ph = carrierSampleClass(line, left + x);
        g_disentProbe.sample(parity, flipNeg,
            std::fabs(cm) * invIreScale,
            std::fabs(2.0 * cm * sin4fsc(ph)) * invIreScale,
            std::fabs(2.0 * cm * cos4fsc(ph)) * invIreScale);
    }
}

// Off-grid leakage of the published carriers (LDCD_PROBE_OFFGRID).
// Measurement only. See OffGridProbe for what the numbers decide.
void Comb::FrameBuffer::probeOffGrid()
{
    if (!OffGridProbe::on()) return;
    // Referee repair (2026-08-02): on covered frames the certified head and
    // factFit make the "fit" and "1D" sources here the exact carrier, so
    // "off-grid leakage" and "fit-vs-comb dropout" would be truth-vs-comb
    // measurements wearing estimator labels (the pre-repair "95.7% lock"
    // calibration note is void for the same reason). The probe's question
    // is about estimator behaviour; ask it only where estimators answer.
    if (frameHasExactCoverage()) return;

    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int width     = right - left;
    if (width < 8 || firstLine >= lastLine) return;

    const double floorTot = (0.5 * irescale) * (0.5 * irescale) * 4.0;

    for (int line = firstLine; line < lastLine && line < demodLines; ++line) {
        const CombCarrierGrammar *grammar = carrierGrammarLine(line);
        if (!grammar || !grammar->grammarLocked) continue;
        double bI4[4], bQ4[4];
        for (int p = 0; p < 4; ++p) {
            bI4[p] = (double)grammar->demodLUTTi[p];
            bQ4[p] = (double)grammar->demodLUTTq[p];
        }

        const float  *fitRow  = carrierRetractedValid
            ? carrierFit_line(line) : nullptr;
        const double *combRow = lockedCarrierComposite_line(line);
        const double *oneDRow = locked1DSource_line(line);
        const lddecode::CarrierAnalysisRecord *anRow =
            carrierAnalysis_line(line);

        const double strongIRE = 8.0;
        double lastPhase[3];
        bool   havePhase[3] = {false, false, false};

        for (int x = 0; x + 3 < width; x += 2) {
            double Sii = 0, Siq = 0, Sqq = 0;
            for (int k = 0; k < 4; ++k) {
                const int cls = carrierSampleClass(line, left + x + k);
                Sii += bI4[cls] * bI4[cls];
                Siq += bI4[cls] * bQ4[cls];
                Sqq += bQ4[cls] * bQ4[cls];
            }
            const double det = Sii * Sqq - Siq * Siq;
            if (std::fabs(det) < 1e-12) continue;

            double phase[3];
            bool   strong[3] = {false, false, false};
            double amp[3] = {0.0, 0.0, 0.0};
            auto solve = [&](int src, auto at) {
                double SiY = 0, SqY = 0, Stt = 0;
                for (int k = 0; k < 4; ++k) {
                    const int cls = carrierSampleClass(line, left + x + k);
                    const double v = at(x + k);
                    SiY += bI4[cls] * v; SqY += bQ4[cls] * v; Stt += v * v;
                }
                amp[src] = std::sqrt(Stt * 0.25) * invIreScale;
                if (Stt < floorTot) return;
                const double p = ( Sqq * SiY - Siq * SqY) / det;
                const double q = (-Siq * SiY + Sii * SqY) / det;
                const double off = std::max(0.0, Stt - (p * SiY + q * SqY));
                g_offGrid.sample(src, amp[src] < 5.0 ? 0 : amp[src] < 15.0 ? 1 : 2,
                                 off, Stt);
                if (amp[src] >= strongIRE) {
                    phase[src] = std::atan2(q, p);
                    strong[src] = true;
                }
            };
            if (fitRow)  solve(0, [&](int xi) { return (double)fitRow[xi]; });
            if (combRow) solve(1, [&](int xi) { return combRow[xi]; });
            if (oneDRow) solve(2, [&](int xi) { return oneDRow[xi]; });

            auto wrapPi = [](double a) {
                while (a >  M_PI) a -= 2.0 * M_PI;
                while (a < -M_PI) a += 2.0 * M_PI;
                return a;
            };
            if (strong[0] && strong[1])
                g_offGrid.phasePair(wrapPi(phase[0] - phase[1]));
            if (fitRow && combRow && amp[1] >= strongIRE) {
                const bool provenIllegal = anRow &&
                    lddecode::carrierIllegalProof(
                        (double)anRow[x + 1].carrierConformance,
                        (double)anRow[x + 1].conformanceSupportFraction)
                        >= 0.7;
                g_offGrid.dropout(amp[0] < 0.5 * amp[1], provenIllegal);
            }
            for (int s = 0; s < 3; ++s) {
                if (!strong[s]) { havePhase[s] = false; continue; }
                if (havePhase[s])
                    g_offGrid.jitter(s, wrapPi(phase[s] - lastPhase[s]));
                lastPhase[s] = phase[s];
                havePhase[s] = true;
            }
        }
    }
}

// Downstream fate of the edge footprints (LDCD_PROBE_EDGEFATE). Measurement
// only -- reads the finished scalar planes after 2D/3D have run, writes
// nothing. See EdgeFateProbe above for what the numbers decide.
void Comb::FrameBuffer::probeEdgeFate(int dimensions)
{
    if (!EdgeFateProbe::on()) return;

    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int width     = right - left;
    if (width < 16 || firstLine >= lastLine) return;

    const bool dims3 = dimensions == 3;
    long   ln[2]    = {0, 0};
    double ld21[2]  = {0.0, 0.0};
    double ld32[2]  = {0.0, 0.0};
    long   lex[2]   = {0, 0};
    long   lnear[2] = {0, 0};

    std::vector<std::uint8_t> mask(width);
    for (int line = firstLine; line < lastLine; ++line) {
        // clpbuffer[0] is not the untouched 1D baseline on certified or
        // fact-corrected frames, so this particular diagnostic has no
        // comparable reference there.
        if (certifiedDefLine(line) ||
            factCorrectedCarrierEstimate_line(line))
            continue;
        const double *c0 = clpbuffer[0].pixel[line];
        const double *c1 = clpbuffer[1].pixel[line];
        const double *c2 = dims3 ? clpbuffer[2].pixel[line] : nullptr;
        if (!c0 || !c1) continue;

        const std::vector<LurchStepRun> &runs = lurchStepRuns_line(line);
        std::fill(mask.begin(), mask.end(), 0);
        for (const LurchStepRun &run : runs) {
            if (run.suppressed || run.gate <= 0.0) continue;
            const int x0 = std::max(0, (int)std::floor(run.edge) - 6);
            const int x1 = std::min(width - 1, (int)std::ceil(run.edge) + 6);
            for (int x = x0; x <= x1; ++x) mask[x] = 1;
        }

        for (int x = 0; x < width; ++x) {
            const int e = mask[x];
            const double d21 =
                std::fabs(c1[left + x] - c0[left + x]) * invIreScale;
            ++ln[e];
            ld21[e] += d21;
            if (c2)
                ld32[e] += std::fabs(c2[left + x] - c1[left + x]) * invIreScale;
            if (d21 < 0.1) ++lex[e];
            if (d21 < 0.5) ++lnear[e];
        }
    }
    g_edgeFate.merge(ln, ld21, ld32, lex, lnear, dims3);
}

// ==== LDCD_PROBE_INVENT: carrier-invention ledger ====
//
// Reconciliation (user, 2026-08-02: "we want all the data we can get").
// The premise behind the lawful-subspace push was that the anchored model
// admits alias on the cube body -- but that reading was rendered |C|
// (chroma magnitude after demod gain and transformIQ), which is NOT
// carrier IRE and cannot be compared with the truth ledger, where the fit
// grades 0.39-0.89 IRE against exact. One of those two readings misleads.
//
// This settles it in one unit: every carrier claimant vs the exact channel
// in IRE, on covered def samples, full frame and inside a box.
// Claimants: 1D (locked1DSource), comb (the elected composite carrier),
// fit (carrierFit), retracted (raw - carrierRetracted).
// Per claimant: mean |claim|, mean |claim - exact|, and two rates --
//   invent  = P(|claim| > |exact| + 1 IRE)   claiming carrier truth lacks
//   confisc = P(|exact| > |claim| + 1 IRE)   missing carrier truth has
// against mean |exact|, the true carrier mass in that population.
// Run with LDCD_CERT_1D=0: exact is read directly (truth survives) while
// every claimant stays an estimator.
void Comb::FrameBuffer::probeCarrierInvention()
{
    static const bool inventOn =
        std::getenv("LDCD_PROBE_INVENT") != nullptr;
    if (!inventOn) return;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int width     = videoParameters.activeVideoEnd - left;
    if (width <= 0 || demodWidth <= 0 || lastLine <= firstLine) return;

    static const auto iEnv = [](const char *n, int d) {
        const char *s = std::getenv(n);
        return s ? std::atoi(s) : d;
    };
    // Box in y4m/export coordinates: line = firstActiveFrameLine + row.
    static const int bR0 = iEnv("LDCD_INVENT_L0", -1);
    static const int bR1 = iEnv("LDCD_INVENT_L1", -1);
    static const int bC0 = iEnv("LDCD_INVENT_C0", -1);
    static const int bC1 = iEnv("LDCD_INVENT_C1", -1);
    const bool haveBox = bR0 >= 0 && bC0 >= 0;

    struct Acc {
        long n = 0;
        double ex = 0;
        double mag[5] = {0, 0, 0, 0, 0};
        double err[5] = {0, 0, 0, 0, 0};
        long   inv[5] = {0, 0, 0, 0, 0};
        long   con[5] = {0, 0, 0, 0, 0};
        long   cn[5]  = {0, 0, 0, 0, 0};
    };
    Acc full, box;

    for (int line = firstLine; line < lastLine; ++line) {
        const float *ex = exactCarrierRow(line);
        if (!ex) continue;
        const quint16 *raw = rawbuffer.data()
            + static_cast<size_t>(line) * videoParameters.fieldWidth;
        const double *oneD = locked1DSource_line(line);
        const double *comp = lockedCarrierComposite_line(line);
        const float  *fit  = carrierFit_line(line);
        const float  *retr = carrierRetracted_line(line);
        const double *Yrow = componentFrame ? componentFrame->y(line)
                                            : nullptr;
        const int row = line - firstLine;

        for (int xi = 0; xi < width; ++xi) {
            const float exv = ex[left + xi];
            if (!std::isfinite(exv)) continue;
            const double exIRE = std::fabs((double)exv) * invIreScale;

            double claim[5];
            bool have[5];
            claim[0] = oneD ? oneD[xi]
                            : std::numeric_limits<double>::quiet_NaN();
            claim[1] = comp ? comp[xi]
                            : std::numeric_limits<double>::quiet_NaN();
            claim[2] = fit ? (double)fit[xi]
                           : std::numeric_limits<double>::quiet_NaN();
            claim[3] = retr ? ((double)raw[left + xi] - (double)retr[xi])
                            : std::numeric_limits<double>::quiet_NaN();
            // The claimant the RENDERER actually consumes: rendered chroma
            // is demod(raw - Y) of the ELECTED Y, not of any carrier plane.
            // If the planes are clean and this is not, the invention lives
            // in the election / Y assembly, not in the carrier model.
            claim[4] = Yrow ? ((double)raw[left + xi] - Yrow[left + xi])
                            : std::numeric_limits<double>::quiet_NaN();
            for (int k = 0; k < 5; ++k) have[k] = std::isfinite(claim[k]);

            auto feed = [&](Acc &A) {
                ++A.n;
                A.ex += exIRE;
                for (int k = 0; k < 5; ++k) {
                    if (!have[k]) continue;
                    const double cIRE = std::fabs(claim[k]) * invIreScale;
                    const double eIRE =
                        std::fabs(claim[k] - (double)exv) * invIreScale;
                    ++A.cn[k];
                    A.mag[k] += cIRE;
                    A.err[k] += eIRE;
                    if (cIRE > exIRE + 1.0) ++A.inv[k];
                    if (exIRE > cIRE + 1.0) ++A.con[k];
                }
            };
            feed(full);
            if (haveBox && row >= bR0 && row <= bR1 &&
                xi >= bC0 && xi <= bC1)
                feed(box);
        }
    }

    static const char *cname[5] =
        { "oneD", "comb", "fit ", "retr", "elecY" };
    auto report = [&](const Acc &A, const char *tag) {
        if (!A.n) return;
        for (int k = 0; k < 5; ++k) {
            if (!A.cn[k]) continue;
            std::fprintf(stderr,
                "[INVENT] seq=%d zone=%-4s claim=%s n=%-8ld "
                "trueIRE=%.3f claimIRE=%.3f errIRE=%.3f "
                "invent=%.3f confisc=%.3f\n",
                (int)heldSeq1, tag, cname[k], A.cn[k],
                A.ex / A.n, A.mag[k] / A.cn[k], A.err[k] / A.cn[k],
                (double)A.inv[k] / A.cn[k], (double)A.con[k] / A.cn[k]);
        }
    };
    report(full, "full");
    report(box, "box");
}

// ==== LDCD_PROBE_TWEEN: two-sided certified-luma tween, held out ====
//
// User design call (2026-08-02): amplitude work in COMPOSITE is out -- an
// envelope needs a window (a smoother on composite blurs, never cancels)
// and the per-cycle alternative scores at carrier rate, which amplitude-
// modulates the very thing it judges. The luma route is an identity
// instead: carrier = raw - L, per sample, no window anywhere. Because
// alias-conforming cross-colour IS luma, a true L excludes it from the
// carrier model BY CONSTRUCTION rather than by bound -- the only form of
// "lawful subspace" that can reject energy no spectral law can touch.
//
// The shipped anticipated rung is ONE-SIDED (causal copy from the previous
// cover) = copying, not tweening, and it renders as the ghost the beach
// exposed. The cadence makes the two-sided form available for free: every
// uncovered frame sits at the midpoint between two covers.
//
// HELD OUT HONESTLY: target = the MIDDLE of three consecutive covers,
// predicted from the outer two, graded per sample against its own exact
// channel. That straddle spans 4 film frames with the target at the
// midpoint -- exactly twice the real job's distance -- so every number is
// a conservative bound on the real tween.
//
// Columns, all on one population:
//   tween = 0.5*(L_prev2 + L_current)   the proposal
//   copy  = L_prev2                     what ships today (one-sided)
//   fit   = raw - fit                   the incumbent, from the PRE-STAMP
//                                       fit stash so it grades an
//                                       estimator, not the fact factFit
//                                       writes over it
// Split by DIRECT (both predictors land on their own def lines -- the
// parity-union question) vs BRACKET, and by straddle disagreement
// |L_prev2 - L_current| (the motion axis: where the two covers agree the
// tween should be truth; where they disagree the disagreement IS the
// out-of-band residual, NOT a witness -- see the revocation note in
// refineRetractedTemporal).
//
// Run: LDCD_PROBE_TWEEN=1 LDCD_CERT_1D=0 -t 1. Exact is read directly
// (certifiedDefLine short-circuits under CERT_1D=0), so truth survives the
// hold-out while every estimator stays an estimator.
void Comb::FrameBuffer::probeLumaTween()
{
    static const bool tweenOn =
        std::getenv("LDCD_PROBE_TWEEN") != nullptr;
    if (!tweenOn) return;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int width     = videoParameters.activeVideoEnd - left;
    if (width <= 0 || demodWidth <= 0 || lastLine <= firstLine) return;
    const size_t planeSz =
        static_cast<size_t>(demodLines) * demodWidth;
    if (planeSz == 0) return;
    const float nanF = std::numeric_limits<float>::quiet_NaN();

    // Coverage test independent of the certified-family gates, so the
    // probe still sees covers under LDCD_CERT_1D=0.
    bool isCover = false;
    for (int line = firstLine; line < lastLine && !isCover; ++line) {
        const float *ex = exactCarrierRow(line);
        if (!ex) continue;
        for (int xi = 0; xi < width; ++xi)
            if (std::isfinite(ex[left + xi])) { isCover = true; break; }
    }
    if (!isCover) return;

    // This cover's certified luma (raw - exact) and fit-implied luma.
    std::vector<float> cert(planeSz, nanF), fitL(planeSz, nanF);
    for (int line = firstLine; line < lastLine; ++line) {
        const float *ex = exactCarrierRow(line);
        if (!ex) continue;
        const quint16 *raw = rawbuffer.data()
            + static_cast<size_t>(line) * videoParameters.fieldWidth;
        const float *preFit = (probePreFactFit_flat.size() == planeSz)
            ? probePreFactFit_flat.data() +
                  static_cast<size_t>(line) * demodWidth
            : nullptr;
        const float *fitRow = (carrierFit_flat.size() == planeSz)
            ? carrierFit_flat.data() +
                  static_cast<size_t>(line) * demodWidth
            : nullptr;
        float *cRow = cert.data() + static_cast<size_t>(line) * demodWidth;
        float *fRow = fitL.data() + static_cast<size_t>(line) * demodWidth;
        for (int xi = 0; xi < width; ++xi) {
            const double rawV = (double)raw[left + xi];
            if (std::isfinite(ex[left + xi]))
                cRow[xi] = (float)(rawV - (double)ex[left + xi]);
            double f = std::numeric_limits<double>::quiet_NaN();
            if (preFit && std::isfinite(preFit[xi])) f = (double)preFit[xi];
            else if (fitRow && std::isfinite(fitRow[xi]))
                f = (double)fitRow[xi];
            if (std::isfinite(f)) fRow[xi] = (float)(rawV - f);
        }
    }

    // Rolling stash of the two previous covers. Function-local statics,
    // the established diagnostic pattern in this file: -t 1 only.
    static std::vector<float> certPrev, certPrev2, fitPrev;
    static qint32 seqPrev = -1, seqPrev2 = -1;

    if (certPrev.size() == planeSz && certPrev2.size() == planeSz &&
        fitPrev.size() == planeSz) {
        auto sampleAt = [&](const std::vector<float> &P, int line, int xi,
                            bool &direct) -> double {
            const float v = P[static_cast<size_t>(line) * demodWidth + xi];
            if (std::isfinite(v)) { direct = true; return (double)v; }
            direct = false;
            if (line - 1 >= firstLine && line + 1 < lastLine) {
                const float a =
                    P[static_cast<size_t>(line - 1) * demodWidth + xi];
                const float b =
                    P[static_cast<size_t>(line + 1) * demodWidth + xi];
                if (std::isfinite(a) && std::isfinite(b))
                    return 0.5 * ((double)a + (double)b);
            }
            return std::numeric_limits<double>::quiet_NaN();
        };

        double sE[2][4][3] = {};
        long sN[2][4] = {};
        long sNf[2][4] = {};
        long tgtN = 0, twAny = 0, twDirect = 0, cpAny = 0, cpDirect = 0;

        for (int line = firstLine; line < lastLine; ++line) {
            for (int xi = 0; xi < width; ++xi) {
                const size_t idx =
                    static_cast<size_t>(line) * demodWidth + xi;
                const float tgtF = certPrev[idx];
                if (!std::isfinite(tgtF)) continue;
                const double tgt = (double)tgtF;
                ++tgtN;
                bool dA = false, dB = false;
                const double a = sampleAt(certPrev2, line, xi, dA);
                const double b = sampleAt(cert, line, xi, dB);
                if (std::isfinite(a)) {
                    ++cpAny;
                    if (dA) ++cpDirect;
                }
                if (!std::isfinite(a) || !std::isfinite(b)) continue;
                ++twAny;
                if (dA && dB) ++twDirect;

                const int cls = (dA && dB) ? 0 : 1;
                const double dis = std::fabs(a - b) * invIreScale;
                const int mb = dis < 0.5 ? 0 : dis < 2.0 ? 1
                             : dis < 8.0 ? 2 : 3;
                ++sN[cls][mb];
                sE[cls][mb][0] +=
                    std::fabs(0.5 * (a + b) - tgt) * invIreScale;
                sE[cls][mb][1] += std::fabs(a - tgt) * invIreScale;
                const float ft = fitPrev[idx];
                if (std::isfinite(ft)) {
                    sE[cls][mb][2] +=
                        std::fabs((double)ft - tgt) * invIreScale;
                    ++sNf[cls][mb];
                }
            }
        }

        // CONSECUTIVE-COVER UNION CENSUS. certPrev2 and certPrev are
        // adjacent covers, i.e. exactly the pair that straddles ONE
        // uncovered frame -- the real application geometry, as opposed to
        // the ±2 straddle this probe must grade on (truth only exists at
        // covers). The question the tween lives or dies on: does the
        // yin-yang def-parity mirror make those two covers
        // parity-complementary, so their union delivers DIRECT certified
        // luma on every line of the frame between them (no vertical
        // bracket -- no smoother)? Measured here over the whole active
        // area, needing no truth.
        long unionN = 0, unionDirect = 0, p2Only = 0, p1Only = 0, both = 0;
        for (int line = firstLine; line < lastLine; ++line)
            for (int xi = 0; xi < width; ++xi) {
                const size_t idx =
                    static_cast<size_t>(line) * demodWidth + xi;
                const bool a = std::isfinite(certPrev2[idx]);
                const bool b = std::isfinite(certPrev[idx]);
                ++unionN;
                if (a || b) ++unionDirect;
                if (a && b) ++both;
                else if (a) ++p2Only;
                else if (b) ++p1Only;
            }
        auto defParOf = [&](const std::vector<float> &P) {
            for (int line = firstLine; line < lastLine; ++line)
                for (int xi = 0; xi < width; ++xi)
                    if (std::isfinite(
                            P[static_cast<size_t>(line) * demodWidth + xi]))
                        return line & 1;
            return -1;
        };
        std::fprintf(stderr,
            "[TWEENUNION] coverA=%d coverB=%d parA=%d parB=%d "
            "unionDirect=%.3f bothDirect=%.3f aOnly=%.3f bOnly=%.3f\n",
            (int)seqPrev2, (int)seqPrev,
            defParOf(certPrev2), defParOf(certPrev),
            unionN ? (double)unionDirect / unionN : 0.0,
            unionN ? (double)both / unionN : 0.0,
            unionN ? (double)p2Only / unionN : 0.0,
            unionN ? (double)p1Only / unionN : 0.0);

        std::fprintf(stderr,
            "[TWEENCOV] seqMid=%d seqPrev=%d seqNext=%d targetN=%ld "
            "tweenAny=%.3f tweenDirect=%.3f copyAny=%.3f copyDirect=%.3f\n",
            (int)seqPrev, (int)seqPrev2, (int)heldSeq1, tgtN,
            tgtN ? (double)twAny / tgtN : 0.0,
            tgtN ? (double)twDirect / tgtN : 0.0,
            tgtN ? (double)cpAny / tgtN : 0.0,
            tgtN ? (double)cpDirect / tgtN : 0.0);
        static const char *cn[2] = { "direct", "bracket" };
        static const char *mn[4] = { "<0.5", "0.5-2", "2-8", ">=8" };
        for (int c = 0; c < 2; ++c)
            for (int m = 0; m < 4; ++m) {
                if (!sN[c][m]) continue;
                std::fprintf(stderr,
                    "[TWEEN] seqMid=%d cls=%-7s dis=%-5s n=%-8ld "
                    "tween=%.4f copy=%.4f fit=%.4f\n",
                    (int)seqPrev, cn[c], mn[m], sN[c][m],
                    sE[c][m][0] / sN[c][m],
                    sE[c][m][1] / sN[c][m],
                    sNf[c][m] ? sE[c][m][2] / sNf[c][m] : 0.0);
            }
    }

    certPrev2 = std::move(certPrev);
    seqPrev2  = seqPrev;
    certPrev  = cert;
    fitPrev   = std::move(fitL);
    seqPrev   = heldSeq1;
}

void Comb::FrameBuffer::lurchSharpenCoarsePrior(const double *means,
                                                int meanCount,
                                                int width,
                                                double *prior,
                                                double *gateOut,
                                                double gateGain) const
{
    if (!means || meanCount < 6 || width <= 0) {
        if (gateOut && width > 0)
            std::fill(gateOut, gateOut + width, 0.0);
        return;
    }

    std::vector<LurchStepRun> runs;
    detectLurchSteps(means, meanCount, irescale, invIreScale, runs);
    applyLurchSteps(runs, means, meanCount, width, gateGain, prior, gateOut);
}

// The application half of the lurch sharpener, split out so consumers of the
// CANONICAL run lists (buildLurchStepRuns) apply them without re-detecting.
// gateGain scales the stored unit-gain gates: clamp(gate*gain,0,1) is exactly
// the value detection at that gain would have produced.
void Comb::FrameBuffer::applyLurchSteps(const std::vector<LurchStepRun> &runs,
                                        const double *means, int meanCount,
                                        int width, double gateGain,
                                        double *prior, double *gateOut) const
{
    if (gateOut && width > 0)
        std::fill(gateOut, gateOut + width, 0.0);

    if (!means || !prior || meanCount < 6 || width <= 0 || runs.empty())
        return;

    using StepRun = LurchStepRun;

    // Apply the strongest surviving run per pixel, always blending from the
    // unsharpened base so overlapping runs never compound.
    std::vector<double> base(prior, prior + width);
    std::vector<double> localGate(width, 0.0);

    // A side anchor is only trustworthy if its window is clear of every
    // detected run.  A short bar (e.g. a colored patch abutting black) has
    // two close opposite runs; an anchor that crosses into the other run's
    // span would snap the bar interior to the far plateau.  With no clean
    // anchor on the required side, leave the pixel on the base prior — an
    // honest blur beats a confident wrong plateau.
    const auto anchorContaminated = [&runs](int s) {
        for (const StepRun &other : runs) {
            if (other.suppressed)
                continue;
            if (s >= other.a && s <= other.b + 1)
                return true;
        }
        return false;
    };

    for (const StepRun &run : runs) {
        if (run.suppressed)
            continue;
        const double g = std::clamp(run.gate * gateGain, 0.0, 1.0);

        const int xiFirst =
            std::clamp((int)std::floor(run.edge) - 4, 0, width - 1);
        const int xiLast =
            std::clamp((int)std::ceil(run.edge) + 3, 0, width - 1);

        for (int xi = xiFirst; xi <= xiLast; ++xi) {
            if (g <= localGate[xi])
                continue;

            // One window of margin beyond the detected run: threshold
            // trimming can leave the run's end windows still straddling the
            // step, and snapping to a straddling window is worse than not
            // snapping at all.
            const int side = ((double)xi < run.edge)
                ? std::clamp(std::min(xi, run.a - 1), 0, meanCount - 1)
                : std::clamp(std::max(xi - 3, run.b + 2), 0, meanCount - 1);

            if (anchorContaminated(side))
                continue;

            localGate[xi] = g;
            prior[xi] = base[xi] * (1.0 - g) + means[side] * g;
        }
    }

    if (gateOut) {
        for (int xi = 0; xi < width; ++xi)
            gateOut[xi] = localGate[xi];
    }
}

// Build the carrier-retracted view from the shared carrier model.
//
// The locked orchestration is intentionally single-pass here:
//   1. buildCarrierAnalysis() harvests canonical bandpass and schedule
//      conformance data.
//   2. buildPhaseCorrected1D() builds the corrected 1D baseline.
//   3. buildCarrierRetracted() calls buildCarrierRetractionStage(false)
//      once, after the corrected 1D baseline exists.
//
// buildCarrierRetractionStage() then performs four-view carrier/Y attribution,
// line-to-line cancellation on carrierFit, and raw - combedCarrier to produce
// the flattened carrier-retracted view.
// Retracted-view mode resolution and the shared working-space phase snap.
// File-scope because two stages consume them: the load-time retraction
// build (vertical certified reference, covered frames) and the output-time
// temporal refinement (two-sided certified reference, uncovered frames).
static int ldcdRetractedSourceMode()
{
    static const int mode = []{
        const char *s = std::getenv("LDCD_RETRACTED_SOURCE");
        if (!s)
            return 3;
        if (s[0] == 'a')
            return 3;
        if (s[0] == 'c')
            return 0;
        if (s[0] == 'n')
            return 1;
        if (s[0] == 'p')
            return 2;
        return 3;
    }();
    return mode;
}

static bool ldcdPhaseSnapOn()
{
    static const bool on = []{
        const char *e = std::getenv("LDCD_PHASE_SNAP");
        return !(e && std::atoi(e) == 0);
    }();
    return on;
}

static bool ldcdPhaseSnapTemporalOn()
{
    static const bool on = []{
        const char *e = std::getenv("LDCD_PHASE_SNAP_T");
        return !(e && std::atoi(e) == 0);
    }();
    return on;
}

// Working-space snap: per W=8 window the estimator keeps its amplitude and
// takes the reference's phase, fading in above the amplitude floor (the
// phase of a near-zero carrier is meaningless -- a clamp on an
// impossible). est and out must not alias: windows read est after earlier
// out writes. clampRatio bounds |est|/|ref| to [0.5, 2.0] -- used when est
// is the fit alone, whose window amplitude is wild (ratio p10 0.45 / p90
// 4.1 vs truth): the reference is CERTIFIED carrier, so an estimator
// amplitude far outside its neighbourhood is estimator failure, not
// content. A hull on the impossible, not a value substitution.
static void ldcdApplyPhaseSnap(const std::vector<double> &est,
                               const std::vector<double> &ref,
                               std::vector<double> &out,
                               int width, double irescale,
                               double ampMinIRE, double ampTauIRE,
                               bool clampRatio,
                               const double *gate)
{
    // Centered 9-tap, half-weight ends (integer centroid; the old [-3,+4]
    // W=8 window estimated phase/amplitude half a sample right of the
    // sample it snapped). Endpoint classes coincide at 4fSC, so the class
    // population and the 0.25 normalisation are unchanged.
    constexpr int kSnapHalf = 4;
    for (int xi = kSnapHalf; xi < width - kSnapHalf; ++xi) {
        double eI = 0.0, eQ = 0.0, rI = 0.0, rQ = 0.0;
        bool ok = true;
        for (int k = xi - kSnapHalf; k <= xi + kSnapHalf && ok; ++k) {
            const double b = ref[k];
            if (!std::isfinite(b)) { ok = false; break; }
            static const int cB[4] = { 1, 0, -1, 0 };
            static const int sB[4] = { 0, 1, 0, -1 };
            const double w =
                (k == xi - kSnapHalf || k == xi + kSnapHalf) ? 0.5 : 1.0;
            const int ph = k & 3;
            eI += w * est[k] * cB[ph]; eQ += w * est[k] * sB[ph];
            rI += w * b * cB[ph];      rQ += w * b * sB[ph];
        }
        if (!ok) continue;
        // I/Q sums over W=8 carry ~4x the waveform amplitude; 0.25
        // normalizes for the IRE thresholds. The ratio needs no
        // normalization.
        const double aRefRaw = std::hypot(rI, rQ);
        const double aRef = aRefRaw * 0.25;
        if (aRef < ampMinIRE * irescale) continue;
        const double aEstRaw = std::hypot(eI, eQ);
        double ratio = aEstRaw / aRefRaw;
        if (clampRatio) ratio = std::clamp(ratio, 0.5, 2.0);
        const double snapped = ratio * ref[xi];
        // Ramp from the floor, continuous at it -- a step in a blend
        // weight is a contour.
        double sf = 1.0 - std::exp(
            -(aRef - ampMinIRE * irescale) /
             (ampTauIRE * irescale));
        if (gate) sf *= std::clamp(gate[xi], 0.0, 1.0);
        out[xi] = sf * snapped + (1.0 - sf) * est[xi];
    }
    for (int xi = 0; xi < width; ++xi)
        if (!std::isfinite(out[xi]))
            out[xi] = est[xi];
}

// Two-sided certified temporal reference, factored from
// refineRetractedTemporal so the elected-scalar refinement consumes the
// identical construction. Each covered neighbour contributes a sign-aligned
// reference (direct same-line where its parity covers it, bracket mean
// otherwise; grammar relations exact cross-frame from absolute
// fieldPhaseIDs); the mean across sides IS the interpolation.
void Comb::FrameBuffer::buildTemporalCertReference(
    const FrameBuffer *prevF, const FrameBuffer *nextF, int line,
    std::vector<double> &tAlign,
    std::vector<double> *sidePrev, std::vector<double> *sideNext) const
{
    if (sidePrev)
        std::fill(sidePrev->begin(), sidePrev->end(),
                  std::numeric_limits<double>::quiet_NaN());
    if (sideNext)
        std::fill(sideNext->begin(), sideNext->end(),
                  std::numeric_limits<double>::quiet_NaN());
    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int width     = videoParameters.activeVideoEnd - left;
    std::fill(tAlign.begin(), tAlign.end(),
              std::numeric_limits<double>::quiet_NaN());
    const CombCarrierGrammar *gC = carrierGrammarLine(line);
    const FrameBuffer *nb[2] = {
        (prevF && prevF->frameHasExactCoverage()) ? prevF : nullptr,
        (nextF && nextF->frameHasExactCoverage()) ? nextF : nullptr,
    };
    const auto relSign = [](const CombCarrierGrammar *ga, int h,
                            const CombCarrierGrammar *gb) {
        if (!ga || !gb) return 0.0;
        const auto r =
            lddecode::carrierGrammarSignedPhaseRelation(ga, h, gb, h);
        if (r == lddecode::CarrierPhaseRelation::Opposite) return -1.0;
        if (r == lddecode::CarrierPhaseRelation::Same) return 1.0;
        return 0.0;
    };
    for (int side = 0; side < 2; ++side) {
        const FrameBuffer *fb = nb[side];
        if (!fb) continue;
        const float *pex0 = fb->exactCarrierRow(line);
        const float *pexU = (line - 1 >= firstLine)
            ? fb->exactCarrierRow(line - 1) : nullptr;
        const float *pexD = (line + 1 < lastLine)
            ? fb->exactCarrierRow(line + 1) : nullptr;
        const CombCarrierGrammar *pg0 = fb->carrierGrammarLine(line);
        const CombCarrierGrammar *pgU = fb->carrierGrammarLine(line - 1);
        const CombCarrierGrammar *pgD = fb->carrierGrammarLine(line + 1);
        for (int xi = 0; xi < width; ++xi) {
            const int h = left + xi;
            double ref = std::numeric_limits<double>::quiet_NaN();
            if (pex0 && std::isfinite(pex0[h])) {
                const double s0 = relSign(gC, h, pg0);
                if (s0 != 0.0)
                    ref = s0 * static_cast<double>(pex0[h]);
            }
            if (!std::isfinite(ref) && pexU && pexD &&
                std::isfinite(pexU[h]) && std::isfinite(pexD[h])) {
                const double sU = relSign(gC, h, pgU);
                const double sD = relSign(gC, h, pgD);
                if (sU != 0.0 && sD != 0.0)
                    ref = 0.5 * (sU * static_cast<double>(pexU[h]) +
                                 sD * static_cast<double>(pexD[h]));
            }
            if (!std::isfinite(ref)) continue;
            if (side == 0 && sidePrev) (*sidePrev)[xi] = ref;
            if (side == 1 && sideNext) (*sideNext)[xi] = ref;
            // Mean of available neighbour references = the interpolation
            // (one-sided hold at stream edges).
            tAlign[xi] = std::isfinite(tAlign[xi])
                ? 0.5 * (tAlign[xi] + ref) : ref;
        }
    }
}

// TEMPORAL PHASE INTERPOLATION for uncovered frames (user, 2026-07-29:
// "if we're correcting phase every other frame the other frames should
// come along by an interpolated amount; phase has a slow enough rate
// change that we should be getting close with that"). The uncovered half
// of the stream is where the fit stands alone -- the frames the vertical
// certified snap never reached. Runs at current-time with BOTH temporal
// neighbours loaded: each covered neighbour contributes a sign-aligned
// reference (direct same-line where its parity covers it, bracket mean
// otherwise; grammar relations are exact cross-frame from absolute
// fieldPhaseIDs), and their mean IS the interpolation. Amplitude stays
// local; only PHASE transfers, the coordinate that survives content
// change best. Escapes: inert under LDCD_PHASE_SNAP=0 or
// LDCD_PHASE_SNAP_T=0.
bool Comb::FrameBuffer::refineRetractedTemporal(const FrameBuffer *prevF,
                                                const FrameBuffer *nextF)
{
    if (!carrierRetractedValid) return false;
    if (ldcdRetractedSourceMode() != 3) return false;
    if (!ldcdPhaseSnapOn() || !ldcdPhaseSnapTemporalOn()) return false;
    if (frameHasExactCoverage()) return false;
    const FrameBuffer *nb[2] = {
        (prevF && prevF->frameHasExactCoverage()) ? prevF : nullptr,
        (nextF && nextF->frameHasExactCoverage()) ? nextF : nullptr,
    };
    // Publication is all-or-nothing: an uncovered estimate may acquire
    // fact-corrected authority only after two independent covered witnesses
    // have participated.  At stream/batch edges the provisional retracted
    // candidate remains available to the Y election, but the named estimate
    // accessor stays null and no comb construction can mistake a one-sided
    // hold for completed temporal refinement.
    if (!nb[0] || !nb[1]) return false;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;
    if (width <= 0) return false;

    // Temporal fade is stricter than vertical: the reference is a film
    // frame away, so demand more of its amplitude before trusting its
    // phase; and the local estimator's amplitude rides the certified
    // ratio hull.
    constexpr double kTSnapAmpMinIRE = 2.0;
    constexpr double kTSnapAmpTauIRE = 4.0;

    // A/B escape only (LDCD_TEMPORAL_COMB=0); default ON.
    static const bool temporalComb = []{
        const char *e = std::getenv("LDCD_TEMPORAL_COMB");
        return !(e && std::atoi(e) == 0);
    }();

    static const bool tcwProbeOn =
        std::getenv("LDCD_PROBE_TCW") != nullptr;
    long tcwN[4] = {0, 0, 0, 0};
    double tcwRes[4] = {0, 0, 0, 0};
    double tcwWc[4] = {0, 0, 0, 0};

    std::vector<double> tAlign(width), est(width), out(width);
    std::vector<double> sideP(width), sideN(width), snapGate(width);

    for (int line = firstLine; line < lastLine; ++line) {
        const quint16 *rawLine = rawbuffer.data()
            + static_cast<size_t>(line) * videoParameters.fieldWidth;
        float *retractedRow = carrierRetracted_flat.data()
            + static_cast<size_t>(line) * demodWidth;

        buildTemporalCertReference(nb[0], nb[1], line, tAlign,
                                   &sideP, &sideN);

        // TWIN-AGREEMENT GATE on the interpolated phase (the ghost law,
        // 2026-07-31): the mean of two references is only meaningful
        // where the two ENDPOINTS agree. Under motion each neighbour's
        // phase field carries its own edge geometry, their mean is a
        // phase between two truths that this frame holds at neither --
        // and the snap stamped neighbour-edge ghosts into uncovered
        // frames (measured: 15 of the 24-unit uncovered ghost excess;
        // both orientations, "precisely align with the subsequent
        // frame", user). Per snap window: both sides' vectors above the
        // temporal floor AND phase-agreeing (cos of their difference
        // ramped 0.5->0.9) or the snap abstains and the estimator keeps
        // its own phase. Window-smooth, never a per-sample cut.
        for (int xi = 0; xi < width; ++xi) snapGate[xi] = 0.0;
        // Centered 9-tap, half-weight ends (see ldcdApplyPhaseSnap).
        constexpr int kSnapHalf = 4;
        for (int xi = kSnapHalf; xi < width - kSnapHalf; ++xi) {
            double pI = 0, pQ = 0, nI = 0, nQ = 0;
            bool ok = true;
            for (int k = xi - kSnapHalf; k <= xi + kSnapHalf && ok; ++k) {
                const double p = sideP[k], n = sideN[k];
                if (!std::isfinite(p) || !std::isfinite(n)) {
                    ok = false; break;
                }
                static const int cB[4] = { 1, 0, -1, 0 };
                static const int sB[4] = { 0, 1, 0, -1 };
                const double w =
                    (k == xi - kSnapHalf || k == xi + kSnapHalf) ? 0.5 : 1.0;
                const int ph = k & 3;
                pI += w * p * cB[ph]; pQ += w * p * sB[ph];
                nI += w * n * cB[ph]; nQ += w * n * sB[ph];
            }
            if (!ok) continue;
            const double ap = std::hypot(pI, pQ) * 0.25;
            const double an = std::hypot(nI, nQ) * 0.25;
            // AMPLITUDE RAMP, NOT A CUT -- the uncovered-letter checkerboard.
            //
            // The original form was
            //     if (ap < kTSnapAmpMinIRE || an < kTSnapAmpMinIRE) continue;
            // which holds the gate at exactly 0 on one side of a threshold and
            // lets it reach 1 on the other: a BINARY PER-SAMPLE gate on a
            // carrier-rate quantity. That is the shape ldcdSideCoherenceAlpha's
            // own comment warns about a few lines above -- "binary per-sample
            // gates interleave two differently-phased renders at pixel pitch
            // along edges, which renders as checkerboard (the OOB-cut lesson,
            // re-learned on the beach 2026-07-30)" -- and this stage is where
            // the uncovered-letter checkerboard was bisected to (62bc3bf).
            //
            // MEASURED (per-line gate census, since removed; 5 uncovered
            // frames, s1x11 2795). Attribution of every zero gate: per-side
            // amplitude 91.4%, cosD twin-disagreement 0.7%, window bail 0.4%;
            // the gate is on for 7.5% of samples. cosD RAMPS PROPERLY (mean
            // step 0.019 across its 0.40-wide window, 0.5% of steps crossing it
            // in one sample) -- the twin-agreement law is not implicated. The
            // amplitude cut did not ramp at all, and since out = g*snapped +
            // (1-g)*est, its boundary swapped the sample OUTRIGHT between the
            // reference's phase and the estimator's own:
            //     cut : mean gate step 0.545, 3763 steps >0.5, largest 1.000
            //     ramp: mean gate step 0.074,  121 steps >0.5, largest 0.953
            // -- 3763 full phase swaps per frame, on the rims of the high-
            // carrier islands. Patches, which is how the user sees them.
            //
            // The ramp turns on the IDENTICAL SET of samples (27494 either
            // way); it changes only how sharply they arrive. Confirmed by eye
            // 2026-08-05 (cube_ampramp.mov, user: "Improved").
            //
            // WHY THIS WAS MISSED FOR A DAY: the frame-mean output difference
            // is 0.009 IRE and the max is 28.6 IRE, a 3000:1 ratio. An earlier
            // pass reported the mean and recorded the cut as EXONERATED. The
            // effect lives on 3763 boundaries, not on 364720 samples; a mean
            // over the frame cannot see a boundary error class.
            //
            // The weaker side ramps over kTSnapAmpTauIRE, the amplitude tau
            // this stage already carries and passes to ldcdApplyPhaseSnap. The
            // floor is unchanged: below kTSnapAmpMinIRE the gate is still zero,
            // it is simply reached smoothly instead of by a step. No escape
            // hatch -- a step in a blend weight is an error class, and the one
            // remaining hard zero here is the window bail, which is a genuine
            // absence (no reference exists) rather than a threshold.
            const double aWeak = std::min(ap, an) * invIreScale;
            const double ta = std::clamp(
                (aWeak - kTSnapAmpMinIRE) / kTSnapAmpTauIRE, 0.0, 1.0);
            const double ampAlpha = ta * ta * (3.0 - 2.0 * ta);
            if (ampAlpha <= 0.0) continue;
            const double cosD = (pI * nI + pQ * nQ) /
                                (16.0 * ap * an);
            const double t = std::clamp((cosD - 0.5) / 0.4, 0.0, 1.0);
            snapGate[xi] = ampAlpha * t * t * (3.0 - 2.0 * t);
        }

        for (int xi = 0; xi < width; ++xi) {
            est[xi] = static_cast<double>(rawLine[left + xi]) -
                      static_cast<double>(retractedRow[xi]);
            out[xi] = std::numeric_limits<double>::quiet_NaN();
        }

        // TEMPORAL CERTIFIED-LUMA COMB (user, 2026-07-29: approved as the
        // second estimator for the uncovered comp lines -- "giving
        // uncovered comp lines a real second estimator"). The cadence
        // geometry makes it DIRECT: the two covered neighbours carry
        // opposite parities, so every line here has a same-row certified
        // luma Lhat = raw_nb - exact_nb in exactly one neighbour. Then
        //     R = raw_this - Lhat_nb = C_this + (L_this - L_nb)
        // -- the luma error is purely temporal (motion), no vertical
        // mixing enters at any point. The merge FIR takes the carrier and
        // the out-of-band residual |R - BP(R)| is a CORRELATE ONLY (it is
        // not a witness -- see the revocation note at the merge below),
        // feeding the same confidence alpha as the vertical comb (a
        // penalty to alpha, never a cutoff).
        if (temporalComb) {
            // PARITY LAW: exact consumption must reach both parities
            // together. A direct-only reference alternates source film
            // frames line-by-line (even lines from prev, odd from next)
            // and any motion becomes a LINE-ALTERNATING luma error --
            // measured +9.4% pillar lineAlt in the v1 build. So every
            // line's Lhat draws from BOTH neighbours symmetrically:
            // direct same-line from the parity that covers it, bracket
            // mean from the other; motion error goes common-mode.
            //
            // THE BRACKET IS NOT A COMPROMISE -- IT BUYS UNIFORM
            // PROVENANCE, AND THAT IS THE WHOLE MECHANISM. Every line
            // here carries the SAME mixture of prev/next and the SAME
            // mixture of direct/bracket, so no term in Lhat depends on
            // line parity and there is nothing for an error to alternate
            // with.
            //
            // FALSIFIED 2026-08-02, do not attempt a third time: the
            // "sharp direct + smooth delta" build. With D = L_prev -
            // L_next it is algebraically exact to keep the SHARP direct
            // value on every line and reach the true midpoint via
            //     M = L_prev - D/2   (lines the prev cover owns)
            //     M = L_next + D/2   (lines the next cover owns)
            // smoothing only D, which is a motion difference field and
            // therefore lawful to smooth. It measured +9.1%
            // parity-locked luma (LHAT A/B on the cube, the same
            // signature as v1's +9.4%) and chroma +2.5% AWAY from the
            // certified truth level. The reason generalises:
            //
            //   ANY TERM WHOSE SIGN OR PROVENANCE DEPENDS ON LINE PARITY
            //   CONVERTS ITS OWN ERROR INTO LINE ALTERNATION.
            //
            // D is applied as -D/2 on one parity and +D/2 on the other,
            // so wherever the smoothed D departs from the true D -- i.e.
            // wherever D itself carries vertical structure -- the residual
            // enters with alternating sign and IS alternation. Exactness
            // of the algebra does not help; only exactness of D would,
            // and that is motion compensation, not smoothing. Sharpness
            // at this site has to come from a better estimate of the
            // OTHER parity's luma (uniform provenance preserved), never
            // from a parity-signed correction.
            std::vector<double> R(width,
                std::numeric_limits<double>::quiet_NaN());
            std::vector<double> lhatSum(width, 0.0);
            std::vector<int> lhatN(width, 0);
            // Each cover's own value, kept apart: their DISAGREEMENT is the
            // occlusion reading (see the guard below). Both are filled on
            // every line by direct-or-bracket, so provenance stays uniform
            // across parities and the guard cannot alternate.
            std::vector<double> sideVal[2];
            sideVal[0].assign(width, std::numeric_limits<double>::quiet_NaN());
            sideVal[1].assign(width, std::numeric_limits<double>::quiet_NaN());
            for (int side = 0; side < 2; ++side) {
                const FrameBuffer *fb = nb[side];
                if (!fb) continue;
                const float *pex0 = fb->exactCarrierRow(line);
                const float *pexU = (line - 1 >= firstLine)
                    ? fb->exactCarrierRow(line - 1) : nullptr;
                const float *pexD = (line + 1 < lastLine)
                    ? fb->exactCarrierRow(line + 1) : nullptr;
                const quint16 *rawNb = fb->rawbuffer.data()
                    + static_cast<size_t>(line) * videoParameters.fieldWidth;
                const quint16 *rawNbU = fb->rawbuffer.data()
                    + static_cast<size_t>(std::max(line - 1, 0)) *
                      videoParameters.fieldWidth;
                const quint16 *rawNbD = fb->rawbuffer.data()
                    + static_cast<size_t>(std::min(line + 1,
                            videoParameters.lastActiveFrameLine - 1)) *
                      videoParameters.fieldWidth;
                for (int xi = 0; xi < width; ++xi) {
                    const int h = left + xi;
                    if (pex0 && std::isfinite(pex0[h])) {
                        const double v = static_cast<double>(rawNb[h]) -
                                         static_cast<double>(pex0[h]);
                        lhatSum[xi] += v; lhatN[xi] += 1;
                        sideVal[side][xi] = v;
                    } else if (pexU && pexD &&
                               std::isfinite(pexU[h]) &&
                               std::isfinite(pexD[h])) {
                        const double lu = static_cast<double>(rawNbU[h]) -
                                          static_cast<double>(pexU[h]);
                        const double ld = static_cast<double>(rawNbD[h]) -
                                          static_cast<double>(pexD[h]);
                        const double v = 0.5 * (lu + ld);
                        lhatSum[xi] += v; lhatN[xi] += 1;
                        sideVal[side][xi] = v;
                    }
                }
            }
            // Both sides or nothing: a one-sided reference re-creates the
            // frame-source alternation the symmetric build exists to
            // prevent (stream edges lose the comb, keeping snap-only).
            // BUG FIXED 2026-08-02: this read `(nb[0] && nb[1]) ? 2 : 1`,
            // i.e. where only ONE neighbour was covered it happily built the
            // reference from that one side -- exactly the one-sided
            // direct-reference shape the parity law falsified (+9.4%
            // line-alternating luma), and precisely the opposite of what the
            // comment above it promises. Measured consequence: the frames
            // carrying the WORST temporal excursions (peaks 88-94 IRE) were
            // the batch-boundary frames running one-sided, and the occlusion
            // guard could not act on them either, because it needs two covers
            // to form a disagreement. Both sides or nothing, as stated.
            const int needSides = 2;
            for (int xi = 0; xi < width; ++xi) {
                if (lhatN[xi] < needSides) continue;
                R[xi] = static_cast<double>(rawLine[left + xi]) -
                        lhatSum[xi] / lhatN[xi];
            }

            constexpr double kOobTauIRE = 2.0;

            // ---- OCCLUSION GUARD (2026-08-02) ----
            // Replaces the out-of-band residual, whose term was revoked
            // above: it reads only the out-of-band share of the temporal
            // luma error while the ghost lives IN-band, and measured it
            // INVERTED -- reading lower and trusting more at the largest
            // merges. Visible cost: a light unmasking in the NEXT frame
            // showing through a face.
            //
            // The two covers bracket this frame in time, so their
            // disagreement is a DIRECT reading of temporal instability --
            // largest exactly at an occlusion, no band assumption, and
            // nothing for a smooth error to hide behind. Smoothed
            // laterally (evidence may be smoothed; the VALUE it gates never
            // is), then applied as a cutoff: commit fully below, bail
            // entirely above, no floor.
            static const bool tcGuard = []{
                const char *e = std::getenv("LDCD_TC_GUARD");
                return !(e && std::atoi(e) == 0);
            }();
            static const double kTcCutoffIRE = []{
                const char *e = std::getenv("LDCD_TC_CUTOFF");
                return e ? std::atof(e) : 3.0;
            }();
            const bool tcGuardLive = tcGuard && nb[0] && nb[1];
            std::vector<double> occIRE(width, 0.0);
            if (tcGuardLive) {
                std::vector<double> draw(width,
                    std::numeric_limits<double>::quiet_NaN());
                for (int xi = 0; xi < width; ++xi)
                    if (std::isfinite(sideVal[0][xi]) &&
                        std::isfinite(sideVal[1][xi]))
                        draw[xi] = std::fabs(sideVal[0][xi] -
                                             sideVal[1][xi]) * invIreScale;
                constexpr int kOccHalf = 4;
                for (int xi = 0; xi < width; ++xi) {
                    double acc = 0.0; int n = 0;
                    for (int k = std::max(0, xi - kOccHalf);
                         k <= std::min(width - 1, xi + kOccHalf); ++k) {
                        if (!std::isfinite(draw[k])) continue;
                        acc += draw[k]; ++n;
                    }
                    occIRE[xi] = n ? acc / n : 0.0;
                }
            }

            for (int xi = 0; xi < width; ++xi) {
                bool ok = true;
                double taps[5];
                static const int off[5] = { 0, -2, 2, -4, 4 };
                for (int k = 0; k < 5 && ok; ++k) {
                    const int j = std::clamp(xi + off[k], 0, width - 1);
                    taps[k] = R[j];
                    if (!std::isfinite(taps[k])) ok = false;
                }
                if (!ok) continue;
                const double bp = 0.676462 * taps[0] +
                                  -0.250000 * (taps[1] + taps[2]) +
                                  -0.088231 * (taps[3] + taps[4]);
                // ---- TERM REVOKED, 2026-08-02 (user instruction) ----
                // This quantity was called a "motion witness" throughout
                // this file. It is not a witness and never was: I coined the
                // label, then reasoned from my own label. What it measures
                // is the OUT-OF-BAND part of R = C_this + (L_this - L_nb).
                // C_this is in-band by law, so `res` sees only the
                // out-of-band share of the temporal luma error -- while the
                // share that actually becomes a ghost is the IN-BAND part,
                // which bp passes straight through and `res` cannot see.
                // Lhat is also half bracket (a vertical mean), so `res`
                // reads vertical luma detail even on a perfectly static
                // frame: an HF-content reading with motion as one input.
                //
                // MEASURED (LDCD_PROBE_TCW, 40 frames), binned by how far
                // the merge moves the estimate:
                //     <1 IRE   93.07%  res  1.51  wc 0.885
                //     1-5       6.47%  res  7.97  wc 0.655
                //     5-20      0.45%  res 14.10  wc 0.589
                //     >=20     0.002%  res  6.35  wc 0.716   <-- INVERTS
                // At the largest, most damaging merges the reading FALLS and
                // the trust RISES. It is blind exactly where it would have
                // to see: a smooth occlusion -- a light unmasking behind a
                // face -- is low-frequency, so it passes at near full
                // weight. That is the cheek artifact.
                //
                // The 0.5 floor below compounds it: even a firing reading
                // can only ever halve the damage.
                //
                // Nothing here has evidentiary standing until something
                // grades it against truth.
                const double res = std::fabs(R[xi] - bp);
                double wc;
                if (tcGuardLive) {
                    const double t = std::clamp(
                        (occIRE[xi] - kTcCutoffIRE) / (0.5 * kTcCutoffIRE),
                        0.0, 1.0);
                    wc = 1.0 - t * t * (3.0 - 2.0 * t);
                } else {
                    wc = 0.5 + 0.5 * std::exp(
                        -res / (kOobTauIRE * irescale));
                }
                // LDCD_PROBE_TCW: is the out-of-band residual actually a
                // witness for THIS failure? User challenge (2026-08-02):
                // "How is the out of band residual a motion witness? Isn't
                // it just comb's HF extension?" R = C_this + (L_this -
                // L_nb); C_this is in-band by law, so res sees only the
                // OUT-OF-BAND part of the temporal luma error -- while the
                // part that becomes the ghost is the IN-BAND part bp passes
                // straight through. Binned by how far the merge actually
                // MOVES the estimate: if the big moves carry SMALL res and
                // wc near 1, the witness is blind exactly where it matters.
                if (tcwProbeOn) {
                    const double moved =
                        std::fabs(wc * (bp - est[xi])) * invIreScale;
                    const int b = moved < 1.0 ? 0 : moved < 5.0 ? 1
                                : moved < 20.0 ? 2 : 3;
                    ++tcwN[b];
                    tcwRes[b] += res * invIreScale;
                    tcwWc[b]  += wc;
                }
                est[xi] = wc * bp + (1.0 - wc) * est[xi];
            }
        }

        ldcdApplyPhaseSnap(est, tAlign, out, width, irescale,
                           kTSnapAmpMinIRE, kTSnapAmpTauIRE, true,
                           snapGate.data());
        for (int xi = 0; xi < width; ++xi)
            retractedRow[xi] = static_cast<float>(
                static_cast<double>(rawLine[left + xi]) - out[xi]);
    }
    if (tcwProbeOn) {
        static const char *bn[4] = {"<1", "1-5", "5-20", ">=20"};
        for (int b = 0; b < 4; ++b) {
            if (!tcwN[b]) continue;
            std::fprintf(stderr,
                "[TCW] moveIRE=%-5s n=%-9ld meanOobRes=%.3f IRE  meanWc=%.3f\n",
                bn[b], tcwN[b], tcwRes[b] / tcwN[b], tcwWc[b] / tcwN[b]);
        }
    }
    publishAnchoredCarrierFromRetracted(
        AnchoredCarrierProvenance::FactCorrectedEstimate);
    return factCorrectedCarrierEstimate_line(firstLine) != nullptr;
}

void Comb::FrameBuffer::buildCarrierRetracted(const FrameBuffer *prevF)
{
    buildCarrierRetractionStage(false, prevF);
}


// ---------------------------------------------------------------------------
// Certified twin-bracket carrier hull. See feasibleband.h for the law, the
// measurement record and the falsified alternatives.
//
// Placement: CURRENT time, because it needs a covered NEIGHBOUR; the fit and
// the retraction ladder are built at load time from this frame alone, which is
// why the uncovered letters currently ride an unbounded fit ("B soft ... the
// open cross-frame transfer work", buildCarrierRetractionStage). Measured:
// denying the merge wholesale (--dg-discard) leaves an uncovered frame's
// carrier bit-identical, so no certified fact reaches this object by any other
// route.
//
// ONE bracket per line parity is the ordinary 3:2 geometry -- the two nearest
// twins certify OPPOSITE parities, so each line of an uncovered frame has a
// single certified side, and kMarginOneSided is what applies there. A single
// neighbour is therefore sufficient and the guard below asks only for one.
// An earlier draft demanded both, because the (since falsified) inter-bracket
// motion precondition was measured BETWEEN them; that gate is gone and its
// requirement went with it.
//
// It clamps the PUBLISHED carrier rather than the fit. The certified statement
// is about carrier magnitude and does not care which stage produced the value,
// and clamping the publication keeps one story for every consumer instead of
// bounding a base that later stages re-inflate.
//
// Structurally inert without the inputs: no cadence roles -> no spare -> no
// exact channel -> no bracket -> unbounded interval everywhere. Under
// --dg-discard nothing exists to read and this cannot fire.
// ---------------------------------------------------------------------------
bool Comb::FrameBuffer::applyCertifiedCarrierHull(const FrameBuffer *prevF,
                                                  const FrameBuffer *nextF)
{
    // OPT-IN, not a default: one of four measured sources violates the bound
    // at 9.8% and the regime is not yet characterised (see feasibleband.h).
    // A false forbid confiscates real colour, so this does not ship on.
    static const bool hullOn = []{
        const char *e = std::getenv("LDCD_CARRIER_HULL");
        return e && std::atoi(e) != 0;
    }();
    static const bool hullProbe = std::getenv("LDCD_PROBE_HULL") != nullptr;
    static const auto hullEnvD = [](const char *n, double d) {
        const char *e = std::getenv(n); return e ? std::atof(e) : d;
    };
    // One bracket per line parity is the ordinary 3:2 geometry for the
    // uncovered letters (the two neighbouring twins certify OPPOSITE
    // parities), and one-sided needs the wider margin: 2.3% violation at 2.0
    // against 0.90% two-sided, 0.95% at 3.0.
    static const double kMarginOneSided = hullEnvD("LDCD_HULL_MARGIN", 3.0);
    static const double kMarginTwoSided = hullEnvD("LDCD_HULL_MARGIN2", 2.0);
    // Block mean-square -> per-sample amplitude the carrier may legitimately
    // reach. 0.58% per-sample violation at 2.0; 4.50% at 1.5.
    static const double kCrest        = hullEnvD("LDCD_HULL_CREST", 2.0);
    constexpr int kBlockLines   = 8;   // same-parity lines per block
    constexpr int kBlockSamples = 16;

    if (!hullOn) return false;
    if (!carrierRetractedValid) return false;
    if (ldcdRetractedSourceMode() != 3) return false;
    // Covered frames carry their own twin; measured, their fit already tracks
    // the certified carrier at r = +0.99 and has nothing to be bounded from.
    if (frameHasExactCoverage()) return false;
    // One bracket per line parity is the ordinary 3:2 geometry, so a single
    // neighbour is enough; whichever certifies a line supplies that line.
    if (!prevF && !nextF) return false;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int width     = videoParameters.activeVideoEnd - left;
    if (width <= 0 || firstLine >= lastLine) return false;

    const int fw = videoParameters.fieldWidth;
    const double invIre = invIreScale;

    // The carrier-antisymmetric projection of a composite line, active span.
    // Shared definition; the bracket and this frame are measured identically.
    auto projectBand = [&](const quint16 *rawLine, std::vector<double> &out) {
        out.assign(width, 0.0);
        for (int xi = 0; xi < width; ++xi) {
            const int h = left + xi;
            const double hm2 = (h - 2 >= 0) ? static_cast<double>(rawLine[h - 2])
                                            : static_cast<double>(rawLine[h]);
            out[xi] = lddecode::carrierBandProjection(
                static_cast<double>(rawLine[h]), hm2);
        }
    };
    const FrameBuffer *nb[2] = { prevF, nextF };
    long clampedSamples = 0, evaluatedSamples = 0, bracketedBlocks = 0;
    long fractionOpen = 0;
    double removedIRE = 0.0;
    bool changed = false;

    // Per-block-row scratch, allocated once. Every projection below is taken
    // exactly once per line and then read by every block column.
    const size_t rowStride = static_cast<size_t>(width);
    std::vector<double> selfBand(rowStride * kBlockLines);
    std::vector<double> nbBand[2] = {
        std::vector<double>(rowStride * kBlockLines),
        std::vector<double>(rowStride * kBlockLines) };
    std::vector<double> scratchA(width);
    const float *nbExact[2][kBlockLines] = {};

    for (int base = firstLine; base < lastLine; ++base) {
        // Blocks run over SAME-PARITY lines: which neighbour certifies a line
        // is a parity fact, so a block that mixed parities would mix brackets.
        if (base + 2 * (kBlockLines - 1) >= lastLine) continue;
        if ((base - firstLine) % (2 * kBlockLines) >= 2) continue;

        for (int li = 0; li < kBlockLines; ++li) {
            const int line = base + 2 * li;
            double *sb = selfBand.data() + rowStride * li;
            projectBand(rawbuffer.data() + static_cast<size_t>(line) * fw,
                        scratchA);
            std::copy(scratchA.begin(), scratchA.end(), sb);

            for (int s = 0; s < 2; ++s) {
                nbExact[s][li] = nullptr;
                const FrameBuffer *n = nb[s];
                if (!n || !n->frameHasExactCoverage()) continue;
                const float *ex = n->exactCarrierRow(line);
                if (!ex) continue;
                nbExact[s][li] = ex;
                projectBand(n->rawbuffer.data() + static_cast<size_t>(line) * fw,
                            scratchA);
                std::copy(scratchA.begin(), scratchA.end(),
                          nbBand[s].data() + rowStride * li);
            }
        }

        for (int x0 = 0; x0 + kBlockSamples <= width; x0 += kBlockSamples) {
            lddecode::CertifiedBracketBlock stat[2];
            double selfEnergy = 0.0;
            long   selfN = 0;

            for (int li = 0; li < kBlockLines; ++li) {
                const double *sb = selfBand.data() + rowStride * li;
                for (int xi = x0; xi < x0 + kBlockSamples; ++xi) {
                    selfEnergy += sb[xi] * sb[xi];
                    ++selfN;
                }
                for (int s = 0; s < 2; ++s) {
                    const float *ex = nbExact[s][li];
                    if (!ex) continue;
                    const double *band = nbBand[s].data() + rowStride * li;
                    for (int xi = x0; xi < x0 + kBlockSamples; ++xi) {
                        const double c = static_cast<double>(ex[left + xi]);
                        if (!std::isfinite(c)) continue;   // not certified here
                        stat[s].bracketCarrierEnergy += c * c;
                        stat[s].bracketBandEnergy += band[xi] * band[xi];
                        ++stat[s].samples;
                    }
                }
            }

            if (selfN <= 0) continue;
            const double localBand = selfEnergy / static_cast<double>(selfN);

            // The loosest interval any bracket admits: a bracket that reports
            // more carrier forbids less, and forbidding is the only thing this
            // is allowed to do.
            lddecode::FeasibleInterval iv;   // unbounded
            int bracketsUsed = 0;
            for (int s = 0; s < 2; ++s) {
                if (stat[s].samples <= 0) continue;
                ++bracketsUsed;
            }
            if (bracketsUsed == 0) continue;

            const double margin = (bracketsUsed >= 2) ? kMarginTwoSided
                                                      : kMarginOneSided;
            bool bounded = false;
            for (int s = 0; s < 2; ++s) {
                if (stat[s].samples <= 0) continue;
                lddecode::CertifiedBracketBlock b = stat[s];
                b.bracketCarrierEnergy /= static_cast<double>(b.samples);
                b.bracketBandEnergy    /= static_cast<double>(b.samples);
                b.localBandEnergy       = localBand;
                const lddecode::FeasibleInterval one =
                    lddecode::carrierFeasibleFromCertifiedBracket(
                        b, margin, kCrest);
                if (one.hi > 1e299) continue;      // this bracket forbids nothing
                if (!bounded) { iv = one; bounded = true; }
                else {
                    // Loosest of the available brackets, never the tightest:
                    // each is an independent claim and only the weakest is
                    // safe to impose.
                    if (one.hi > iv.hi) { iv.hi = one.hi; iv.lo = one.lo; }
                }
            }
            if (!bounded) { ++fractionOpen; continue; }
            ++bracketedBlocks;

            // carrierRetracted_flat holds raw - carrier (the retracted LUMA);
            // the carrier is what raw minus it publishes. Bound the carrier
            // and store the luma the bounded carrier implies, so whatever the
            // hull refuses returns to Y rather than vanishing.
            for (int li = 0; li < kBlockLines; ++li) {
                const int line = base + 2 * li;
                const quint16 *rawLine = rawbuffer.data()
                    + static_cast<size_t>(line) * fw;
                float *retractedRow = carrierRetracted_flat.data()
                    + static_cast<size_t>(line) * demodWidth;
                for (int xi = x0; xi < x0 + kBlockSamples; ++xi) {
                    ++evaluatedSamples;
                    const double lum = static_cast<double>(retractedRow[xi]);
                    if (!std::isfinite(lum)) continue;
                    const double raw = static_cast<double>(rawLine[left + xi]);
                    const double carrier = raw - lum;
                    const double bounded = iv.clamp(carrier);
                    if (bounded == carrier) continue;
                    retractedRow[xi] = static_cast<float>(raw - bounded);
                    removedIRE += std::fabs(carrier - bounded) * invIre;
                    ++clampedSamples;
                    changed = true;
                }
            }
        }
    }

    if (hullProbe)
        std::fprintf(stderr,
            "[HULL] blocks bounded=%ld openFraction=%ld  "
            "clamped %ld/%ld (%.1f%%)  mean removed %.2f IRE\n",
            bracketedBlocks, fractionOpen,
            clampedSamples, evaluatedSamples,
            evaluatedSamples ? 100.0 * clampedSamples / evaluatedSamples : 0.0,
            clampedSamples ? removedIRE / clampedSamples : 0.0);

    if (!changed) return false;
    // Republish through the ordinary path so the plane's stated provenance
    // matches what the carrier now says. combSource1D_line() reads this plane
    // directly, so republication is what reaches the comb.
    publishAnchoredCarrierFromRetracted(
        AnchoredCarrierProvenance::FactCorrectedEstimate);
    return true;
}

// Publish a derived full-parity carrier only at the point where its stated
// provenance is actually true. Covered frames arrive here from the load-time
// same-frame construction. Uncovered frames arrive only after the two-sided
// current-time refinement has completed; the causal placeholder is never
// exposed through the fact-corrected accessor.
void Comb::FrameBuffer::publishAnchoredCarrierFromRetracted(
    AnchoredCarrierProvenance provenance)
{
    static const bool anchor1D = []{
        const char *e = std::getenv("LDCD_ANCHOR_1D");
        return !(e && std::atoi(e) == 0);
    }();
    if (!anchor1D || !carrierRetractedValid ||
        ldcdRetractedSourceMode() != 3 ||
        provenance == AnchoredCarrierProvenance::None)
        return;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int width     = videoParameters.activeVideoEnd - left;
    if (width <= 0 || firstLine >= lastLine) return;

    anchored1DSource_flat.assign(
        static_cast<size_t>(demodLines) * demodWidth, 0.0);
    for (int line = firstLine; line < lastLine; ++line) {
        const quint16 *rawLine = rawbuffer.data()
            + static_cast<size_t>(line) * videoParameters.fieldWidth;
        const float *retr = carrierRetracted_flat.data()
            + static_cast<size_t>(line) * demodWidth;
        double *dst = anchored1DSource_flat.data()
            + static_cast<size_t>(line) * demodWidth;
        for (int xi = 0; xi < width; ++xi)
            dst[xi] = static_cast<double>(rawLine[left + xi]) -
                      static_cast<double>(retr[xi]);
    }
    anchoredCarrierProvenance = provenance;
}

// Sync-tone actuator. See comb.h. Runs between fit construction and its
// first consumer inside buildCarrierRetractionStage.
void Comb::FrameBuffer::applyToneToFit(const FrameBuffer *prevF)
{
    // DEFAULT ON (promoted 2026-08-01 with the anchoring family, user
    // directive): the research verdict is that the certified carrier has
    // an exact repeating cycle, and the anticipated phase lands almost
    // exactly on the measured phase at the next cover -- so the curve is
    // usable CAUSALLY, from the previous covered frame alone.  This is
    // the anticipation actuator: prev cover's certified regional phase,
    // advanced by the tracked global rate, corrects the uncovered fit's
    // trend at load time (before Pass 2 and before split2D consumes the
    // ladder).  A tone corrects trend, never performs surgery (+-25 deg
    // hull, confidence-scaled, bilinear-smooth).  Escape LDCD_CERT_TONE=0
    // for A/B only.
    static const bool toneOn = []{
        const char *e = std::getenv("LDCD_CERT_TONE");
        return !(e && std::atoi(e) == 0);
    }();
    if (!toneOn || certifiedOneDLevel() == 0) return;
    if (frameHasExactCoverage()) return;                // anchors stand
    if (!prevF || !prevF->frameHasExactCoverage()) return;
    const QVector<float> &pay =
        !syncIncFirst.isEmpty() ? syncIncFirst : syncIncSecond;
    if (pay.size() < 4) return;
    const double gOmega = pay[0];
    const double gConf  = pay[1];
    const double dtF    = pay[2];
    if (gConf < 0.2 || dtF <= 0.0 || dtF > 8.0) return;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int width     = videoParameters.activeVideoEnd - left;
    if (width <= 0) return;
    const int nx = (width + 127) / 128;
    const int ny = (lastLine + 31) / 32;
    const int nRegPay = (pay.size() - 4) / 2;
    static const int kcB[4] = { 1, 0, -1, 0 };
    static const int ksB[4] = { 0, 1, 0, -1 };

    // Anchor: prevF's certified regional phase, RAW convention (the
    // convention proven identical to the assembler's to 0.01 deg). Def
    // lines only, field-row signing (line/2)&1.
    std::vector<double> aI((size_t)nx * ny, 0.0), aQ((size_t)nx * ny, 0.0);
    std::vector<long>   aN((size_t)nx * ny, 0);
    int defPar = -1;
    for (int line = firstLine; line < lastLine; ++line) {
        if (!prevF->certifiedDefLine(line)) continue;
        if (defPar < 0) defPar = line & 1;
        const float *ex = prevF->exactCarrierRow(line);
        if (!ex) continue;
        const double rs = ((line / 2) & 1) ? -1.0 : 1.0;
        const size_t rb = (size_t)(line / 32) * nx;
        for (int h = left; h < left + width; ++h) {
            if (!std::isfinite(ex[h])) continue;
            const int ph = h & 3;
            const size_t r = rb + (h - left) / 128;
            aI[r] += rs * (double)ex[h] * kcB[ph];
            aQ[r] += rs * (double)ex[h] * ksB[ph];
            aN[r]++;
        }
    }
    if (defPar < 0) return;

    // Local fit phase, pooled over the SAME parity lines with the SAME
    // signing so the delta lives in one convention and any lattice
    // constant cancels.
    std::vector<double> fI((size_t)nx * ny, 0.0), fQ((size_t)nx * ny, 0.0);
    std::vector<long>   fN((size_t)nx * ny, 0);
    for (int line = firstLine; line < lastLine; ++line) {
        if ((line & 1) != defPar) continue;
        const float *fit = carrierFit_flat.data()
                           + (size_t)line * demodWidth;
        const double rs = ((line / 2) & 1) ? -1.0 : 1.0;
        const size_t rb = (size_t)(line / 32) * nx;
        for (int xi = 0; xi < width; ++xi) {
            const float v = fit[xi];
            if (!std::isfinite(v)) continue;
            const int ph = (left + xi) & 3;
            const size_t r = rb + xi / 128;
            fI[r] += rs * (double)v * kcB[ph];
            fQ[r] += rs * (double)v * ksB[ph];
            fN[r]++;
        }
    }

    // Per-region delta: (anchor advanced by the global rate) minus fit,
    // confidence-scaled, hulled at +-25 deg (a tone corrects trend, never
    // performs surgery).
    const double floorRaw = 2.0 * irescale;
    constexpr double kMaxDelta = 25.0 * M_PI / 180.0;
    std::vector<double> delta((size_t)nx * ny, 0.0);
    for (int r = 0; r < nx * ny; ++r) {
        if (aN[r] < 64 || fN[r] < 64) continue;
        if (std::hypot(aI[r], aQ[r]) / aN[r] < floorRaw) continue;
        if (std::hypot(fI[r], fQ[r]) / fN[r] < floorRaw) continue;
        double conf = gConf;
        if (r < nRegPay) conf *= std::clamp((double)pay[4 + r * 2 + 1],
                                            0.0, 1.0);
        if (conf <= 0.05) continue;
        const double aPh = std::atan2(aQ[r], aI[r]) + gOmega * dtF;
        const double d0 = std::atan2(
            std::sin(aPh) * fI[r] - std::cos(aPh) * fQ[r],
            std::cos(aPh) * fI[r] + std::sin(aPh) * fQ[r]);
        delta[r] = std::clamp(conf * d0, -kMaxDelta, kMaxDelta);
    }

    // Bilinear delta field over region centers, then the quadrature
    // rotation per line. Smooth by construction: no seams, no per-sample
    // policy.
    std::vector<double> rot(width), src(width);
    for (int line = firstLine; line < lastLine; ++line) {
        float *fit = carrierFit_flat.data() + (size_t)line * demodWidth;
        const double ry = std::clamp(
            (line - 16.0) / 32.0, 0.0, (double)(ny - 1) - 1e-6);
        const int r0 = (int)ry;
        const double wy = ry - r0;
        for (int xi = 0; xi < width; ++xi) {
            const double rx = std::clamp(
                (xi - 64.0) / 128.0, 0.0, (double)(nx - 1) - 1e-6);
            const int c0 = (int)rx;
            const double wx = rx - c0;
            const double d =
                (1 - wy) * ((1 - wx) * delta[(size_t)r0 * nx + c0] +
                            wx * delta[(size_t)r0 * nx + c0 + 1 < (size_t)nx * ny ? (size_t)r0 * nx + std::min(c0 + 1, nx - 1) : (size_t)r0 * nx + c0]) +
                wy * ((1 - wx) * delta[(size_t)std::min(r0 + 1, ny - 1) * nx + c0] +
                      wx * delta[(size_t)std::min(r0 + 1, ny - 1) * nx + std::min(c0 + 1, nx - 1)]);
            src[xi] = fit[xi];
            rot[xi] = d;
        }
        for (int xi = 0; xi < width; ++xi) {
            const double sm = xi > 0 ? src[xi - 1] : src[xi];
            const double sp = xi + 1 < width ? src[xi + 1] : src[xi];
            const double q = 0.5 * (sm - sp);
            fit[xi] = (float)(std::cos(rot[xi]) * src[xi] +
                              std::sin(rot[xi]) * q);
        }
    }
}

void Comb::FrameBuffer::buildCarrierRetractionStage(bool analysisOnly,
                                                    const FrameBuffer *prevF)
{
    carrierRetractedValid = false;

    if (!configuration.phaseCompensation ||
        !configuration.lumaWitness)
        return;

    const int firstLine = videoParameters.firstActiveFrameLine;
    const int lastLine  = videoParameters.lastActiveFrameLine;
    const int left      = videoParameters.activeVideoStart;
    const int right     = videoParameters.activeVideoEnd;
    const int width     = right - left;
    const auto &T       = configuration.tunables;

    if (width <= 0 || firstLine >= lastLine)
        return;
    if (demodWidth != width || demodLines < lastLine)
        return;
    if (!basisLockedInit)
        return;

    const size_t need = static_cast<size_t>(demodLines) * static_cast<size_t>(demodWidth);
    if (carrierFit_flat.size() < need)
        carrierFit_flat.assign(need, 0.0f);
    if (carrierRetracted_flat.size() < need)
        carrierRetracted_flat.assign(need, 0.0f);
    if (flatFloor_flat.size() < need)
        flatFloor_flat.assign(need, 0.0f);
    if (combedCarrier_flat.size() < need)
        combedCarrier_flat.assign(need, 0.0f);
    if (carrierCorroboration_flat.size() < need)
        carrierCorroboration_flat.assign(need, 0.0f);
    if (carrierEligibility_flat.size() < need)
        carrierEligibility_flat.assign(need, 0.0f);
    if (coarseYEvidence_flat.size() < need)
        coarseYEvidence_flat.assign(need, lddecode::FourViewPixelEvidence{});
    if (carrierImpurity_flat.size() < need)
        carrierImpurity_flat.assign(need, 0.0f);
    if (carrierAnalysis_flat.size() < need)
        return; // shared analysis must already have been produced

    // --- Disposable dead-zone instrumentation (env-gated). Set
    // LDCD_DUMP_DEADZONE=1 to print the amplified footprint of the schedule
    // DQ downstream: how many pixels lose their carrier estimate, split by
    // cause. Zero cost when unset. Remove with the rethink. ---
    const bool dumpDead = std::getenv("LDCD_DUMP_DEADZONE") != nullptr;
    long long dzActive = 0, dzIneligible = 0;
    long long dzDead = 0, dzDeadIllegal = 0, dzDeadFitStarved = 0;
    long long dzWinTotal = 0, dzWinInvalid = 0, dzWinRank = 0, dzWinDet = 0;

    if ((int)scratch_preI.size()        < width) scratch_preI.resize(width, 0.0);
    if ((int)scratch_preQ.size()        < width) scratch_preQ.resize(width, 0.0);
    if ((int)scratch_lineWorkA.size()   < width) scratch_lineWorkA.resize(width, 0.0);
    if ((int)scratch_lineWorkB.size()   < width) scratch_lineWorkB.resize(width, 0.0);
    if ((int)scratch_lineWorkC.size()   < width) scratch_lineWorkC.resize(width, 0.0);
    if ((int)scratch_lineWorkD.size()   < width) scratch_lineWorkD.resize(width, 0.0);
    if ((int)scratch_lumaBaseY4.size()  < width) scratch_lumaBaseY4.resize(width, 0.0);
    if ((int)scratch_lumaSmooth.size()  < width) scratch_lumaSmooth.resize(width, 0.0);
    if ((int)scratch_lateralLine.size() < width) scratch_lateralLine.resize(width, 0.0);

    double *rawWhole   = scratch_preI.data();
    double *coarseY    = scratch_preQ.data();
    double *carrierFit = scratch_lineWorkA.data();
    double *basisI     = scratch_lineWorkB.data();
    double *flattened  = scratch_lineWorkC.data();
    double *basisQ     = scratch_lineWorkD.data();
    double *refinedY   = scratch_lumaSmooth.data();
    double *slideMean4 = scratch_lateralLine.data();

    std::vector<double> winFloor;
    std::vector<double> winI;
    std::vector<double> winQ;
    std::vector<double> winErrorIRE;
    std::vector<double> winLatticeIRE;
    std::vector<double> winYSpanIRE;
    std::vector<double> winScore;
    std::vector<std::uint8_t> winFitValid;
    std::vector<std::uint8_t> boundaryMark;
    std::vector<double> winEnvScratch;
    std::vector<double> envI, envQ, envTmp;
    std::vector<double> partWeight(static_cast<size_t>(width), 1.0);

    auto median3 = [](double a, double b, double c) -> double {
        if (a > b) std::swap(a, b);
        if (b > c) std::swap(b, c);
        if (a > b) std::swap(a, b);
        return b;
    };

    // Flat-region fast path: when all four views agree to within ~0.05 IRE,
    // the cost matrix is dominated by arithmetic noise and the medoid is
    // within the same tolerance of the plain mean. Skip 16 abs + compares
    // (twice per pixel across the two call sites) and return the mean. The
    // anchor only matters as a tiebreak when views disagree, so it does not
    // need to participate in the flat case.
    const double medoidFlatTol = 0.05 * irescale;
    auto medoid4Anchored = [medoidFlatTol](double a, double b, double c, double d,
                                            double anchor) -> double {
        const double lo = std::min(std::min(a, b), std::min(c, d));
        const double hi = std::max(std::max(a, b), std::max(c, d));
        if (hi - lo < medoidFlatTol)
            return 0.25 * (a + b + c + d);

        double v[4] = {a, b, c, d};
        double best = v[0];
        double bestCost = 1e300;
        double bestAnchorDist = 1e300;

        for (int i = 0; i < 4; ++i) {
            double cost = 0.0;
            for (int j = 0; j < 4; ++j)
                cost += std::fabs(v[i] - v[j]);

            const double anchorDist = std::fabs(v[i] - anchor);
            if (cost < bestCost - 1e-9 ||
                (std::fabs(cost - bestCost) <= 1e-9 &&
                 anchorDist < bestAnchorDist)) {
                best = v[i];
                bestCost = cost;
                bestAnchorDist = anchorDist;
            }
        }

        return best;
    };

    auto smoothStep01 = [](double t) {
        t = std::clamp(t, 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    };

    if (!carrierRetractionModelValid) {
        // ---------------------------------------------------------------
        // Analysis/model promotion: per-line carrier withdrawal.
        //
        // This version does not start from the ±2 complement estimate.  It first
        // derives a per-sample Y prior from the legal 4-sample luma-floor views,
        // then projects raw - refinedY into locked IQ.  The four carrier windows
        // touching the current sample are treated as attribution evidence rather
        // than being blended as an immediate heuristic result.
        //
        // In other words:
        //
        //     raw = Y + C
        //
        // is resolved by asking which legal Y floor leaves the most coherent C,
        // rather than by subtracting a complement-estimated C and trusting whatever
        // remains as Y.
        // ---------------------------------------------------------------

    for (int line = firstLine; line < lastLine; ++line) {
        float *fitRow       = carrierFit_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        float *floorRow     = flatFloor_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        auto *evidenceRow   = coarseYEvidence_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        auto *analysisRow   = carrierAnalysis_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        float *eligibilityRow = carrierEligibility_flat.data()
                              + static_cast<size_t>(line) * demodWidth;

        for (int xi = 0; xi < width; ++xi)
            analysisRow[xi].parallax = lddecode::CarrierParallaxDiagnostics{};

        const CombCarrierGrammar *grammar = carrierGrammarLine(line);
        const bool grammarLocked = grammar && grammar->grammarLocked;

        const quint16 *rawLine =
            rawbuffer.data() + static_cast<size_t>(line) * videoParameters.fieldWidth;

        const double *baseY4Src;
        if (lockedLumaCacheValid && demodWidth == width &&
            !lockedLumaBaseY4_flat.empty())
        {
            baseY4Src = lockedLumaBaseY4_line(line);
        } else {
            buildCompositeLumaDecompositionLine(rawLine, left, width,
                                                scratch_lumaBaseY4.data(),
                                                nullptr, nullptr);
            baseY4Src = scratch_lumaBaseY4.data();
        }

        for (int xi = 0; xi < width; ++xi) {
            rawWhole[xi] = static_cast<double>(rawLine[left + xi]);
            coarseY[xi]  = baseY4Src[xi];
            refinedY[xi] = coarseY[xi];
        }

        if (!grammarLocked) {
            for (int xi = 0; xi < width; ++xi) {
                fitRow[xi]       = 0.0f;
                floorRow[xi]     = static_cast<float>(coarseY[xi]);
                evidenceRow[xi].viewCount = 0;
                eligibilityRow[xi] = 0.0f;
            }
            continue;
        }

        // Graded schedule participation, consumed everywhere this stage used
        // to consume the thresholded enum.  carrierTrust() is the table-owned
        // mapping from the conformance MEASUREMENT (+ support fraction) to a
        // carrier-trust weight; participation doubles it and saturates so the
        // legacy keep-set (Legal, Unresolved, quiet — trust >= 0.5) keeps its
        // old FULL weight and only the illegal side grades: one axis vote
        // ~0.67, two ~0.33, decisive proof 0.  The old per-sample enum test
        // was a 1-bit sampler on a smoothly drifting vertical correlation —
        // half of all Illegal verdicts rested on a single axis vote, and the
        // bit flipping at line pitch was the dominant vertical raggedness of
        // the retracted view (Borg-cube study, 2026-07-20).
        for (int xi = 0; xi < width; ++xi) {
            const auto &a = analysisRow[xi];
            partWeight[xi] = std::min(1.0, 2.0 * lddecode::carrierTrust(
                static_cast<double>(a.carrierConformance),
                static_cast<double>(a.conformanceSupportFraction)));
            eligibilityRow[xi] = static_cast<float>(partWeight[xi]);
        }

        const double bcos = grammar->burstCos;
        const double bsin = grammar->burstSin;
        const double maxCarrierSamples =
            maxCarrierAmpIREFromScale(grammar->carrierScale) * irescale;

        // basisI/Q at position h depend only on the carrier sample class
        // given the line's burst phasor and locked basis.  The class is the
        // GRAMMAR sample class, carrierSampleClass(line, h) = (h+samplePhase0)
        // & 3 — not a raw h & 3.  These are identical while samplePhase0 == 0
        // (its current value everywhere), but every other locked-path site
        // (Pass-3 demod, splitIQlocked, filterIQLocked, the HF-election
        // phaseConf) indexes by the grammar class; hardcoding the raw index
        // here was the lone site that would silently diverge the instant the
        // schedule set a nonzero sample phase.  Consult the schedule, do not
        // assume it.  Precompute the four phase values and fill by lookup.
        double basisI4[4];
        double basisQ4[4];
        for (int p = 0; p < 4; ++p) {
            basisI4[p] = remodLockedToShiftedComposite(
                1.0, 0.0, p, bcos, bsin,
                spLUT_locked, cpLUT_locked);
            basisQ4[p] = remodLockedToShiftedComposite(
                0.0, 1.0, p, bcos, bsin,
                spLUT_locked, cpLUT_locked);
        }
        for (int xi = 0; xi < width; ++xi) {
            const int idx = carrierSampleClass(line, left + xi);
            basisI[xi] = basisI4[idx];
            basisQ[xi] = basisQ4[idx];
        }

        if (width >= 4) {
            const int meanCount = width - 3;
            if ((int)winFloor.size() < meanCount) {
                winFloor.resize(meanCount, 0.0);
                winI.resize(meanCount, 0.0);
                winQ.resize(meanCount, 0.0);
                winErrorIRE.resize(meanCount, 0.0);
                winLatticeIRE.resize(meanCount, 0.0);
                winYSpanIRE.resize(meanCount, 0.0);
                winScore.resize(meanCount, 0.0);
                winFitValid.resize(meanCount, std::uint8_t{0});
            }
            if ((int)boundaryMark.size() < width)
                boundaryMark.resize(width, 0);

            // winFloor IS the shared aperture pool: 0.25*(raw[s..s+3]) is
            // exactly what buildApertureMeans() published (both are exact
            // integer sums, so the copy is byte-identical to the old private
            // rebuild). One scan, many readers.
            {
                const double *apRow = lockedApertureMean_line(line);
                if (apRow) {
                    std::copy(apRow, apRow + meanCount, winFloor.begin());
                } else {
                    for (int s = 0; s < meanCount; ++s)
                        winFloor[s] =
                            0.25 * (rawWhole[s + 0] + rawWhole[s + 1] +
                                    rawWhole[s + 2] + rawWhole[s + 3]);
                }
            }

            // Luma prior: the integer-centred moving coarse, not a medoid of the
            // four covering means.  The medoid was robust but is still a
            // boxcar statistic with half the smear baked in; the prior is
            // the rolling legal mean, and the lurch preconditioner restores
            // the step placement the boxcar blurs.
            for (int xi = 0; xi < width; ++xi) {
                const int s0 = std::clamp(xi - 2, 0, meanCount - 1);
                const int s1 = std::clamp(xi - 1, 0, meanCount - 1);
                refinedY[xi] = 0.5 * (winFloor[s0] + winFloor[s1]);
            }

            // Lurch preconditioner: sharpen the prior before the carrier fit
            // consumes it, so step energy stays out of raw - refinedY and
            // never enters the carrier band. Canonical runs (built once in
            // split1D on the same pool), edges vertically corroborated;
            // apply-only here.
            const std::vector<LurchStepRun> corrRuns =
                corroborateLurchEdges(line);
            applyLurchSteps(corrRuns, winFloor.data(),
                            meanCount, width, 1.0, refinedY, nullptr);

            for (int s = 0; s < meanCount; ++s) {
                double sII = 0.0, sIQ = 0.0, sQQ = 0.0;
                double sIY = 0.0, sQY = 0.0;
                double sampleWeight = 0.0;

                double refinedMean = 0.0;
                double minRefined = 1e300;
                double maxRefined = -1e300;

                for (int k = 0; k < 4; ++k) {
                    const int xi = s + k;
                    refinedMean += refinedY[xi];
                    minRefined = std::min(minRefined, refinedY[xi]);
                    maxRefined = std::max(maxRefined, refinedY[xi]);

                    // Registration doubt DOWNWEIGHTS, it does not remove.
                    // The old hard removal changed the normal matrix's sample
                    // population whenever the per-pixel verdict bit flipped,
                    // so adjacent lines solved structurally different systems
                    // and their fits decorrelated (which Pass 2's license then
                    // read as noise).  A weight leaves both sides of the solve
                    // in the same geometry and lets doubt fade the sample
                    // smoothly; w == 0 (decisive proof) still removes exactly.
                    const double w = partWeight[xi];
                    if (w <= 0.0)
                        continue;

                    const double bI = basisI[xi];
                    const double bQ = basisQ[xi];
                    const double residual = rawWhole[xi] - refinedY[xi];

                    sII += w * bI * bI;
                    sIQ += w * bI * bQ;
                    sQQ += w * bQ * bQ;
                    sIY += w * bI * residual;
                    sQY += w * bQ * residual;
                    sampleWeight += w;
                }

                refinedMean *= 0.25;

                double fitI = 0.0;
                double fitQ = 0.0;
                const double det = sII * sQQ - sIQ * sIQ;
                // Effective-sample floor: a 2-parameter solve needs more than
                // two samples' worth of participating evidence (the old rule
                // was >= 3 of 4 hard samples; the graded analogue crosses the
                // same boundary smoothly instead of at one sample's bit).
                const bool fitValid =
                    sampleWeight >= 2.5 && std::fabs(det) > 1e-9;
                if (fitValid) {
                    const double inv = 1.0 / det;
                    fitI = ( sQQ * sIY - sIQ * sQY) * inv;
                    fitQ = (-sIQ * sIY + sII * sQY) * inv;
                }

                winI[s] = fitI;
                winQ[s] = fitQ;
                winFitValid[s] = fitValid ? std::uint8_t{1} : std::uint8_t{0};

                if (dumpDead) {
                    ++dzWinTotal;
                    if (!fitValid) {
                        ++dzWinInvalid;
                        if (sampleWeight < 2.5)
                            ++dzWinRank;   // starved of participating weight
                        else
                            ++dzWinDet;    // killed by singular normal matrix
                    }
                }

                double errSq = 0.0;
                double basis01 = 0.0; // +-+-
                double basis02 = 0.0; // ++--
                double basis03 = 0.0; // +--+
                double fitAbs = 0.0;
                double gradedWeight = 0.0;

                for (int k = 0; k < 4; ++k) {
                    const int xi = s + k;
                    const double w = partWeight[xi];
                    if (w <= 0.0)
                        continue;

                    const double bI = basisI[xi];
                    const double bQ = basisQ[xi];

                    const double fit = fitI * bI + fitQ * bQ;
                    const double residual = rawWhole[xi] - refinedY[xi];
                    const double e = residual - fit;

                    errSq += w * e * e;
                    fitAbs += w * std::fabs(fit);
                    gradedWeight += w;

                    const double we = w * e;
                    if (k == 0) {
                        basis01 += we;
                        basis02 += we;
                        basis03 += we;
                    } else if (k == 1) {
                        basis01 -= we;
                        basis02 += we;
                        basis03 -= we;
                    } else if (k == 2) {
                        basis01 += we;
                        basis02 -= we;
                        basis03 -= we;
                    } else {
                        basis01 -= we;
                        basis02 -= we;
                        basis03 += we;
                    }
                }

                const double gradeInv = gradedWeight > 0.0
                    ? 1.0 / gradedWeight
                    : 0.0;
                const double errIRE =
                    std::sqrt(gradeInv * errSq) * invIreScale;
                const double latticeIRE =
                    gradeInv * std::max({std::fabs(basis01),
                                         std::fabs(basis02),
                                         std::fabs(basis03)}) * invIreScale;
                const double floorDriftIRE =
                    std::fabs(winFloor[s] - refinedMean) * invIreScale;
                const double ySpanIRE =
                    (maxRefined - minRefined) * invIreScale;
                const double ampIRE =
                    gradeInv * fitAbs * invIreScale;

                winErrorIRE[s] = errIRE;
                winLatticeIRE[s] = latticeIRE;
                winYSpanIRE[s] = ySpanIRE;

                // The score prefers:
                //   - residual that remodulates cleanly through locked IQ,
                //   - little remaining +-+- / ++-- / +--+ lattice residue,
                //   - a legal floor that agrees with the refined-Y prior,
                //   - low refined-Y span inside the 4-sample cell.
                //
                // The small amplitude term prevents the all-Y interpretation
                // from winning merely because it is conservative.
                winScore[s] =
                    errIRE +
                    0.75 * latticeIRE +
                    0.25 * floorDriftIRE +
                    0.15 * ySpanIRE -
                    0.10 * std::min(ampIRE, 24.0);
            }

            // NOTE: the encoder bandwidth law is NOT applied to winI/winQ here.
            // It is imposed ONCE, at the model boundary below, after the
            // four-view attribution and residual clamp have had their say.
            // Filtering here as well would be a second forward application of
            // the same FIR, and a forward FIR is not a projection: P(P(x)) !=
            // P(x) (measured 0.25 relative error, and 4.5 dB of EXTRA loss at
            // 1.3 MHz for a third application).  Legal chroma near the top of
            // the encoder's own passband would be progressively attenuated and
            // dumped into Y — the opposite of the intent.

            // Residual-side chroma-boundary discovery (the lurch dual).
            // The per-window LS fits are the carrier profile along the
            // line, and they were solved against the lurch-sharpened Y
            // prior, so luma steps are already out of this profile: what
            // steps here is chroma.  A boundary is a step between two
            // internally COHERENT runs — d(p) compares the disjoint
            // adjacent windows ending at p and starting at p+1, so a mark
            // at p places the transition between samples p and p+1.  The
            // side-coherence requirement leaves broadband texture
            // (incoherent profile) unmarked: there the four-view spread is
            // noise, not geometry, and filtering would only cost the
            // attribution its robustness.
            std::fill(boundaryMark.begin(),
                      boundaryMark.begin() + width, std::uint8_t{0});
            {
                const double stepFloor = 3.0 * irescale;
                for (int p = 7; p + 5 < meanCount + 3; ++p) {
                    const int sl = p - 3;       // window [p-3 .. p]
                    const int sr = p + 1;       // window [p+1 .. p+4]
                    const int sll = sl - 4;
                    const int srr = sr + 4;
                    if (srr >= meanCount)
                        break;
                    if (!winFitValid[sl] || !winFitValid[sr] ||
                        !winFitValid[sll] || !winFitValid[srr])
                    {
                        continue;
                    }
                    const double dI = winI[sr] - winI[sl];
                    const double dQ = winQ[sr] - winQ[sl];
                    const double stepSq = dI * dI + dQ * dQ;
                    const double magLSq =
                        winI[sl] * winI[sl] + winQ[sl] * winQ[sl];
                    const double magRSq =
                        winI[sr] * winI[sr] + winQ[sr] * winQ[sr];
                    if (stepSq < stepFloor * stepFloor ||
                        stepSq < (0.35 * 0.35) * std::max(magLSq, magRSq))
                        continue;
                    const double cI = winI[sl] - winI[sll];
                    const double cQ = winQ[sl] - winQ[sll];
                    const double dI2 = winI[srr] - winI[sr];
                    const double dQ2 = winQ[srr] - winQ[sr];
                    if (cI * cI + cQ * cQ > 0.25 * stepSq ||
                        dI2 * dI2 + dQ2 * dQ2 > 0.25 * stepSq)
                        continue;
                    boundaryMark[p] = 1;
                }
            }

            for (int xi = 0; xi < width; ++xi) {
                const int sFirst = std::max(0, xi - 3);
                const int sLast  = std::min(xi, meanCount - 1);

                lddecode::FourViewCarrierView views[4];
                int viewCount = 0;
                // The pixel's own schedule doubt no longer withholds the fit:
                // the covering windows were solved with that doubt already
                // downweighted, so the model's opinion AT this position is
                // well-formed evidence.  How much of the resulting carrier may
                // ACT is the eligibility weight published above — authority is
                // graded downstream, not amputated here.  (The old gate also
                // published fitRow = 0 at every proven-illegal pixel, which
                // made those samples unobservable to Pass 2's license and
                // partner correlations.)

                // Region-pure aperture law: a window straddling a
                // discovered chroma boundary is not evidence for any pixel.
                // A boundary at p splits samples p | p+1, so window
                // [s .. s+3] straddles iff a mark lies in [s .. s+2]; the
                // window ENDING at the boundary stays pure for the left
                // side and the one STARTING after it for the right, so
                // every pixel keeps at least one same-side view unless
                // marks are pathologically dense — then fall back to the
                // unfiltered set rather than starve attribution.
                auto windowStraddles = [&](int s) {
                    const int pHi = std::min(s + 2, width - 1);
                    for (int p = s; p <= pHi; ++p)
                        if (boundaryMark[p])
                            return true;
                    return false;
                };

                for (int pass = 0;
                     pass < 2 && viewCount == 0;
                     ++pass) {
                for (int s = sFirst; s <= sLast; ++s) {
                    if (viewCount >= 4)
                        break;
                    if (!winFitValid[s])
                        continue;
                    if (pass == 0 && windowStraddles(s))
                        continue;
                    const double carrierSample = rawWhole[xi] - winFloor[s];
                    const double fittedSample =
                        winI[s] * basisI[xi] + winQ[s] * basisQ[xi];
                    views[viewCount].apertureCenter = static_cast<double>(s) + 1.5;
                    views[viewCount].yFloor = winFloor[s];
                    views[viewCount].carrierSample = carrierSample;
                    views[viewCount].fittedSample = fittedSample;
                    views[viewCount].carrierI = winI[s];
                    views[viewCount].carrierQ = winQ[s];
                    views[viewCount].sampleFitErrorIRE =
                        std::fabs(carrierSample - fittedSample) * invIreScale;
                    views[viewCount].remodErrorIRE = winErrorIRE[s];
                    views[viewCount].latticeRiskIRE = winLatticeIRE[s];
                    views[viewCount].ySpanIRE = winYSpanIRE[s];
                    views[viewCount].score = winScore[s];
                    {
                        double mDeltaIRE = 0.0;
                        double mSupport = 0.0;
                        double mLocalX = 0.0;
                        const int s0 = s;
                        if (s0 + 4 < width) {
                            const double enterLeaveSample =
                                rawWhole[s0 + 4] - rawWhole[s0];
                            mDeltaIRE = 0.25 * enterLeaveSample * invIreScale;
                            const double deltaMagIRE = std::fabs(mDeltaIRE);
                            const double deltaGate =
                                smoothStep01((deltaMagIRE - 0.35) / 4.0);
                            const double fitGate =
                                1.0 - smoothStep01((winErrorIRE[s0] - 1.5) / 5.0);
                            const double latticeGate =
                                1.0 - smoothStep01((winLatticeIRE[s0] - 1.0) / 5.0);
                            mSupport = std::clamp(
                                deltaGate * fitGate * latticeGate, 0.0, 1.0);
                            mLocalX = 0.5 * ((double)s0 + (double)(s0 + 4))
                                    - (double)xi;
                        }
                        views[viewCount].membershipDeltaIRE    = mDeltaIRE;
                        views[viewCount].membershipSupport     = mSupport;
                        views[viewCount].membershipLocalX      = mLocalX;
                    }
                    ++viewCount;
                }
                }  // pass: region-pure first, unfiltered fallback

                evidenceRow[xi].viewCount = viewCount;

                if (dumpDead) {
                    ++dzActive;
                    if (partWeight[xi] <= 0.0)
                        ++dzIneligible;
                    if (viewCount == 0) {
                        ++dzDead;
                        // Pass 1 harvests any covering winFitValid window, so
                        // viewCount==0 means the fit was starved (no covering
                        // window survived weight/det).
                        if (partWeight[xi] <= 0.0)
                            ++dzDeadIllegal;
                        else
                            ++dzDeadFitStarved;
                    }
                }

                for (int v = 0; v < viewCount; ++v) {
                    auto &dst = evidenceRow[xi].views[v];
                    const auto &src = views[v];
                    dst.apertureCenter = static_cast<float>(src.apertureCenter);
                    dst.yFloor = static_cast<float>(src.yFloor);
                    dst.sampleFitErrorIRE = static_cast<float>(src.sampleFitErrorIRE);
                    dst.remodErrorIRE = static_cast<float>(src.remodErrorIRE);
                    dst.latticeRiskIRE = static_cast<float>(src.latticeRiskIRE);
                    dst.ySpanIRE = static_cast<float>(src.ySpanIRE);
                    dst.membershipDeltaIRE    = static_cast<float>(src.membershipDeltaIRE);
                    dst.membershipSupport     = static_cast<float>(src.membershipSupport);
                }

                // ---- Stage-0 line-solve harness dump (env-gated, disposable).
                // Emits everything the offline event solver needs, at the one
                // point where the four-view feasible band is already built:
                //   raw, bp        -> notch = raw-bp, and the leak lives in bp
                //   lurch          -> carrier-free luma-motion evidence
                //   conf/sched     -> interline grammar (luma by law)
                //   yFloor[0..3]   -> the FOUR LEGAL VIEWS as a feasible band
                //                     (constraints, never collapsed here)
                // LDCD_DUMP_SOLVE_L / _C0 / _C1.  Run -t 1.
                {
                    static const int slL  = []{ const char *s=std::getenv("LDCD_DUMP_SOLVE_L");  return s?std::atoi(s):-1; }();
                    static const int slC0 = []{ const char *s=std::getenv("LDCD_DUMP_SOLVE_C0"); return s?std::atoi(s):-1; }();
                    static const int slC1 = []{ const char *s=std::getenv("LDCD_DUMP_SOLVE_C1"); return s?std::atoi(s):-1; }();
                    if (line == slL && xi >= slC0 && xi <= slC1) {
                        const double *bpL = locked1DRawBandpass_line(line);
                        const double rawI = (double)rawLine[left + xi] * invIreScale;
                        const double bpI  = bpL ? bpL[xi] * invIreScale : 0.0;
                        const double *apM = lockedApertureMean_line(line);
                        const double *clk = lockedCornerLeak_line(line);
                        std::fprintf(stderr, "[LEAK] h=%d leak=%.4f\n",
                                     left + xi, clk ? clk[xi] * invIreScale : 0.0);
                        std::fprintf(stderr,
                            "[SOLVE] h=%d raw=%.3f bp=%.3f am0=%.3f am1=%.3f "
                            "am2=%.3f am3=%.3f lurch=%.3f conf=%+.2f "
                            "supp=%.2f nviews=%d f0=%.2f f1=%.2f f2=%.2f f3=%.2f\n",
                            left + xi, rawI, bpI,
                            apM ? apM[std::max(0,xi-3)] * invIreScale : 0.0,
                            apM ? apM[std::max(0,xi-2)] * invIreScale : 0.0,
                            apM ? apM[std::max(0,xi-1)] * invIreScale : 0.0,
                            apM ? apM[xi] * invIreScale : 0.0,
                            (double)analysisRow[xi].residual.maxAbsMembershipIRE,
                            (double)analysisRow[xi].carrierConformance,
                            (double)analysisRow[xi].conformanceSupportFraction,
                            viewCount,
                            viewCount > 0 ? evidenceRow[xi].views[0].yFloor * invIreScale : 0.0f,
                            viewCount > 1 ? evidenceRow[xi].views[1].yFloor * invIreScale : 0.0f,
                            viewCount > 2 ? evidenceRow[xi].views[2].yFloor * invIreScale : 0.0f,
                            viewCount > 3 ? evidenceRow[xi].views[3].yFloor * invIreScale : 0.0f);
                    }
                }

                auto parallax = lddecode::buildFourViewCarrierAttribution(
                    views,
                    viewCount,
                    invIreScale);

                double modelI = parallax.valid ? parallax.commonI : 0.0;
                double modelQ = parallax.valid ? parallax.commonQ : 0.0;

                double residualCarrierLo = -1e300;
                double residualCarrierHi =  1e300;
                double residualTightenSupport = 0.0;
                double residualTightenSpreadIRE = 0.0;
                double residualTightenFitErrorIRE = 0.0;
                if (viewCount > 0) {
                    residualCarrierLo =  1e300;
                    residualCarrierHi = -1e300;
                    double sumFitError = 0.0;

                    int residualN = 0;

                    for (int v = 0; v < viewCount; ++v) {
                        const auto &view = views[v];
                        residualCarrierLo = std::min(residualCarrierLo, view.carrierSample);
                        residualCarrierHi = std::max(residualCarrierHi, view.carrierSample);
                        sumFitError += view.sampleFitErrorIRE;
                        ++residualN;
                    }

                    // Fifth residual witness: a centered/rolling legal 4fSC
                    // cancellation complement.  The four aperture views tell
                    // which legal windows including xi say what carrier remains;
                    // this rolling witness is aperture-independent in the sense
                    // that it is not one of the four ownership views being
                    // scored.  It is still a carrier-cancelling Y estimate, so
                    // raw - moving floor is mostly carrier plus whatever HF-Y
                    // the rolling window could not preserve.
                    double movingResidualSample = 0.0;
                    // The rolling witness obeys the same region-pure
                    // aperture law as the four ownership views.
                    const int movingS0 =
                        std::clamp(xi - 2, 0, meanCount - 1);
                    const int movingS1 =
                        std::clamp(xi - 1, 0, meanCount - 1);
                    if (meanCount > 0 &&
                        !windowStraddles(movingS0) &&
                        !windowStraddles(movingS1)) {
                        const double movingFloor =
                            0.5 * (winFloor[movingS0] + winFloor[movingS1]);
                        movingResidualSample = rawWhole[xi] - movingFloor;
                        residualCarrierLo = std::min(residualCarrierLo, movingResidualSample);
                        residualCarrierHi = std::max(residualCarrierHi, movingResidualSample);
                        const double movingFitError = parallax.valid
                            ? std::fabs(movingResidualSample - parallax.commonSample) * invIreScale
                            : 0.0;
                        sumFitError += movingFitError;
                        ++residualN;
                    }

                    residualTightenFitErrorIRE =
                        sumFitError / static_cast<double>(std::max(1, residualN));
                    residualTightenSpreadIRE =
                        (residualCarrierHi - residualCarrierLo) * invIreScale;

                    const double spreadGate = 1.0 - smoothStep01(
                        (residualTightenSpreadIRE - 2.0) / 8.0);
                    const double fitGate = 1.0 - smoothStep01(
                        (residualTightenFitErrorIRE - 1.5) / 6.0);
                    residualTightenSupport = std::clamp(
                        0.35 + 0.65 * spreadGate * fitGate,
                        0.0,
                        1.0);

                    lddecode::CarrierResidualConsensus consensus;
                    consensus.lo = residualCarrierLo;
                    consensus.hi = residualCarrierHi;
                    consensus.trust = residualTightenSupport;
                    consensus.valid = true;
                    parallax.residualConsensus = consensus;
                }

                {
                    auto &dst = analysisRow[xi].parallax;
                    dst.commonSample = static_cast<float>(parallax.commonSample);
                    dst.commonMagnitudeIRE = static_cast<float>(parallax.commonMagIRE);
                    dst.ySpreadIRE = static_cast<float>(parallax.ySpreadIRE);
                    dst.yCurvatureIRE = static_cast<float>(parallax.yCurvatureIRE);
                    dst.carrierSpreadIRE = static_cast<float>(parallax.carrierSpreadIRE);
                    dst.carrierCoherence = static_cast<float>(parallax.carrierCoherence);
                    dst.sampleFitErrorIRE = static_cast<float>(parallax.sampleFitErrorIRE);
                    dst.sampleCoherence = static_cast<float>(parallax.sampleCoherence);
                    dst.latticeRiskIRE = static_cast<float>(parallax.latticeRiskIRE);
                    dst.valid = parallax.valid;
                    if (parallax.residualConsensus.valid) {
                        dst.residualLo = static_cast<float>(
                            parallax.residualConsensus.lo);
                        dst.residualHi = static_cast<float>(
                            parallax.residualConsensus.hi);
                        dst.residualTrust = static_cast<float>(
                            parallax.residualConsensus.trust);
                        dst.residualValid = true;
                    }
                }

                // The model sample is the four-view model remodulated on the
                // grammar basis, then RESTRICTED: the residual-consensus
                // range and the amplitude ceiling remove impossible values
                // and nothing else. Two operations that used to sit between
                // remod and publication were deleted as UNLAWFUL under the
                // coarse-residual law (statistics and range-restriction only
                // -- never blend, never average, never output):
                //   * the commonSample anchor blend pulled the waveform
                //     toward a consensus of the coarse-residual VALUES --
                //     averaged coarse residuals becoming output;
                //   * the survivor-extent move treated the interval between
                //     discrete survivors as a legal continuum, which the
                //     analysis contract forbids in as many words ("the
                //     unobserved values between them do not become legal").
                // Measured before removal (LDCD_PROBE_OFFGRID): the fit's
                // span, rotation, and jitter were already the cleanest in
                // the tree -- nothing real leaned on either operation.
                double cf = modelI * basisI[xi] + modelQ * basisQ[xi];
                if (residualTightenSupport > 0.0)
                    cf = std::clamp(cf, residualCarrierLo, residualCarrierHi);
                cf = std::clamp(cf, -maxCarrierSamples, maxCarrierSamples);

                carrierFit[xi] = cf;
                flattened[xi] = rawWhole[xi] - cf;

                fitRow[xi] = static_cast<float>(cf);
            }

            // ---------------------------------------------------------------
            // ENCODER BANDWIDTH LAW, enforced at publication.
            //
            // Projecting winI/winQ above is not sufficient on its own:
            // finalizeCarrierSample() then blends the sample toward
            // parallax.commonSample and clamps it into the residual-consensus
            // bounds, and BOTH of those are built from raw residuals
            // (rawWhole - winFloor), which are full-band by construction.
            // Those steps are right on their own terms -- they pull a bad
            // window fit back toward what every legal Y floor says at this
            // sample -- but they re-admit exactly the out-of-band energy the
            // encoder could never have modulated.  Measured: the early
            // projection alone moved the fit only 23.5% -> 20.7% out of band.
            //
            // So the law is applied last, to the published model, where
            // nothing downstream can violate it.  Demodulate the finished fit
            // against the line's own grammar basis, bandlimit the envelope,
            // remodulate.  The 2x2 normal matrix is CONSTANT along the line:
            // basisI/basisQ depend only on the carrier sample class, and any
            // four consecutive samples span all four classes, so it is
            // inverted once per line rather than per sample.
            // ---------------------------------------------------------------
            {
                if ((int)envI.size() < width) {
                    envI.resize(width, 0.0);
                    envQ.resize(width, 0.0);
                    envTmp.resize(width, 0.0);
                }

                double sII = 0.0, sIQ = 0.0, sQQ = 0.0;
                for (int k = 0; k < 4 && k < width; ++k) {
                    sII += basisI[k] * basisI[k];
                    sIQ += basisI[k] * basisQ[k];
                    sQQ += basisQ[k] * basisQ[k];
                }
                const double det = sII * sQQ - sIQ * sIQ;

                if (std::fabs(det) > 1e-9) {
                    const double inv = 1.0 / det;
                    // Centered 5-tap, half-weight ends (integer centroid;
                    // the old [xi-1, xi+2] window solved for the envelope
                    // half a sample right of the sample it published).  At
                    // 4fSC the endpoint classes coincide, so each carrier
                    // class keeps total weight 1 and the normal matrix is
                    // unchanged.
                    for (int xi = 0; xi < width; ++xi) {
                        double sIY = 0.0, sQY = 0.0;
                        for (int k = -2; k <= 2; ++k) {
                            const int j = std::clamp(xi + k, 0, width - 1);
                            const double w = (k == -2 || k == 2) ? 0.5 : 1.0;
                            const double v = w * carrierFit[j];
                            sIY += basisI[j] * v;
                            sQY += basisQ[j] * v;
                        }
                        envI[xi] = ( sQQ * sIY - sIQ * sQY) * inv;
                        envQ[xi] = (-sIQ * sIY + sII * sQY) * inv;
                    }

                    lddecode::projectExpressibleChromaEnvelope(
                        envI.data(), nullptr, width, envTmp.data());
                    std::copy(envTmp.begin(), envTmp.begin() + width,
                              envI.begin());
                    lddecode::projectExpressibleChromaEnvelope(
                        envQ.data(), nullptr, width, envTmp.data());
                    std::copy(envTmp.begin(), envTmp.begin() + width,
                              envQ.begin());

                    for (int xi = 0; xi < width; ++xi) {
                        const double cf = std::clamp(
                            envI[xi] * basisI[xi] + envQ[xi] * basisQ[xi],
                            -maxCarrierSamples, maxCarrierSamples);
                        carrierFit[xi] = cf;
                        flattened[xi] = rawWhole[xi] - cf;
                        fitRow[xi] = static_cast<float>(cf);
                    }
                }
            }

            // Post-law hull-violation probe (measurement only; inert unless
            // LDCD_PROBE_RETRHULL). Reports what the re-clamp below is about
            // to correct: how far the law's FIR moved the published fit back
            // outside the per-sample feasible range the Pass-1 clamp had
            // enforced.
            if (RetrHullProbe::on()) {
                for (int xi = 0; xi < width; ++xi) {
                    const auto &pp = analysisRow[xi].parallax;
                    if (!pp.residualValid) continue;
                    g_retrHull.sample(carrierFit[xi],
                                      static_cast<double>(pp.residualLo),
                                      static_cast<double>(pp.residualHi),
                                      invIreScale);
                }
            }

            // Re-impose the residual-consensus hull AFTER the law. The law's
            // FIR mixes neighbours and re-manufactures carrier the per-sample
            // feasible range forbids (measured: cube 45.4% of samples
            // outside, mean excess 0.56 IRE, max 44; beach 37.9%). The two
            // constraints cannot be imposed by one pass each in either order
            // -- but they are not symmetric: a clamp is a TRUE projection
            // (idempotent, per-sample), so it cannot compound the FIR, and
            // what it removes returns to Y via flattened = raw - cf, which
            // is lawful -- carrier outside the hull IS luma by the
            // conservation facts. The residual envelope kink this leaves is
            // bounded by the clamp delta; publishing impossible carrier is
            // the greater crime (it is the cube face's manufactured Y). Law
            // once, hull last. Kill switch for A/B only: LDCD_RETRHULL=0.
            {
                static const bool retrHullOn = []{
                    const char *s = std::getenv("LDCD_RETRHULL");
                    return !s || std::atoi(s) != 0;
                }();
                if (retrHullOn) {
                    for (int xi = 0; xi < width; ++xi) {
                        const auto &pp = analysisRow[xi].parallax;
                        if (!pp.residualValid) continue;
                        const double cf = std::clamp(
                            carrierFit[xi],
                            static_cast<double>(pp.residualLo),
                            static_cast<double>(pp.residualHi));
                        if (cf != carrierFit[xi]) {
                            carrierFit[xi] = cf;
                            flattened[xi]  = rawWhole[xi] - cf;
                            fitRow[xi]     = static_cast<float>(cf);
                        }
                    }
                }
            }
        } else {
            for (int xi = 0; xi < width; ++xi) {
                const double cf = 0.0;
                carrierFit[xi] = cf;
                flattened[xi] = rawWhole[xi];
                fitRow[xi] = 0.0f;
                evidenceRow[xi].viewCount = 0;
            }
        }
/*
        // Optional LS refit at luma edges, retained from the previous design.
        // It is now a secondary edge repair, not the primary saturated-fill
        // carrier estimator. 
        {
            constexpr double EDGE_SOFT_IRE  = 3.0;
            constexpr double EDGE_HARD_IRE  = 10.0;
            constexpr double DISC_SOFT_IRE  = 1.0;
            constexpr double DISC_HARD_IRE  = 4.0;
            constexpr int    LS_HALF_WIN    = 2;
            constexpr bool   LS_BRIGHT_COLOR_GUARD = true;
            constexpr double LS_BRIGHT_SIDE_SOFT_IRE = 3.0;
            constexpr double LS_BRIGHT_SIDE_HARD_IRE = 10.0;
            constexpr double LS_BRIGHT_COLOR_START_IRE = 8.0;
            constexpr double LS_BRIGHT_COLOR_FULL_IRE = 24.0;

            double *edgeGate = slideMean4;
            bool anyEdge = false;

            for (int xi = 0; xi < width; ++xi) {
                const int xm = std::max(0, xi - 2);
                const int xp = std::min(width - 1, xi + 2);
                const double gradIRE =
                    std::fabs(refinedY[xp] - refinedY[xm]) * invIreScale;
                const double gate = std::clamp(
                    (gradIRE - EDGE_SOFT_IRE) /
                    std::max(1e-9, EDGE_HARD_IRE - EDGE_SOFT_IRE),
                    0.0, 1.0);
                edgeGate[xi] = gate;
                if (gate > 0.0)
                    anyEdge = true;
            }

            if (anyEdge) {
                for (int xi = 0; xi < width; ++xi) {
                    if (edgeGate[xi] <= 0.0)
                        continue;

                    const int a = std::max(0, xi - LS_HALF_WIN);
                    const int b = std::min(width - 1, xi + LS_HALF_WIN);

                    double sII = 0.0, sIQ = 0.0, sQQ = 0.0;
                    double sIY = 0.0, sQY = 0.0;

                    for (int k = a; k <= b; ++k) {
                        const double obs = rawWhole[k] - refinedY[k];
                        const double bI = basisI[k];
                        const double bQ = basisQ[k];

                        const double dist = std::fabs(static_cast<double>(k - xi));
                        const double w = 1.0 - 0.65 * std::min(
                            1.0, dist / std::max(1.0, static_cast<double>(LS_HALF_WIN)));

                        sII += w * bI * bI;
                        sIQ += w * bI * bQ;
                        sQQ += w * bQ * bQ;
                        sIY += w * bI * obs;
                        sQY += w * bQ * obs;
                    }

                    double fitI = 0.0, fitQ = 0.0;
                    const double det = sII * sQQ - sIQ * sIQ;
                    if (std::fabs(det) > 1e-9) {
                        const double inv = 1.0 / det;
                        fitI = ( sQQ * sIY - sIQ * sQY) * inv;
                        fitQ = (-sIQ * sIY + sII * sQY) * inv;
                    }

                    double lsFit = fitI * basisI[xi] + fitQ * basisQ[xi];
                    lsFit = std::clamp(lsFit,
                        -maxCarrierSamples, maxCarrierSamples);

                    const double discrepancyIRE =
                        std::fabs(lsFit - carrierFit[xi]) * invIreScale;
                    const double discGate = std::clamp(
                        (discrepancyIRE - DISC_SOFT_IRE) /
                        std::max(1e-9, DISC_HARD_IRE - DISC_SOFT_IRE),
                        0.0, 1.0);

                    double brightColorProtect = 0.0;
                    if (LS_BRIGHT_COLOR_GUARD) {
                        const int xm = std::max(0, xi - 2);
                        const int xp = std::min(width - 1, xi + 2);
                        const double lumaM = refinedY[xm];
                        const double lumaP = refinedY[xp];
                        const double dir = (lumaP >= lumaM) ? 1.0 : -1.0;
                        const double lumaMid = 0.5 * (lumaM + lumaP);
                        const double brightOffsetIRE =
                            dir * (refinedY[xi] - lumaMid) * invIreScale;
                        const double brightSideGate = smoothStep01(
                            (brightOffsetIRE - LS_BRIGHT_SIDE_SOFT_IRE) /
                            std::max(1e-9, LS_BRIGHT_SIDE_HARD_IRE -
                                           LS_BRIGHT_SIDE_SOFT_IRE));

                        const int brightIdx = (dir > 0.0) ? xp : xm;
                        const int brightJ = (brightIdx + 1 < width)
                            ? (brightIdx + 1)
                            : (brightIdx > 0 ? brightIdx - 1 : brightIdx);
                        const int xiJ = (xi + 1 < width)
                            ? (xi + 1)
                            : (xi > 0 ? xi - 1 : xi);
                        const double brightAmpIRE =
                            boundedMag(carrierFit[brightIdx],
                                       carrierFit[brightJ]) * invIreScale;
                        const double localAmpIRE =
                            boundedMag(carrierFit[xi],
                                       carrierFit[xiJ]) * invIreScale;
                        const double coloredBrightIRE =
                            std::max(brightAmpIRE, localAmpIRE);
                        const double brightColorGate = smoothStep01(
                            (coloredBrightIRE - LS_BRIGHT_COLOR_START_IRE) /
                            std::max(1e-9, LS_BRIGHT_COLOR_FULL_IRE -
                                           LS_BRIGHT_COLOR_START_IRE));

                        brightColorProtect = brightSideGate * brightColorGate;
                    }

                    double parallaxProtect = 0.0;
                    if (parallaxRow[xi].valid) {
                        const auto &p = parallaxRow[xi];
                        const double spreadT = std::clamp(
                            p.carrierSpreadIRE /
                            std::max(3.0, 0.35 * p.commonMagIRE + 1.0),
                            0.0,
                            1.0);
                        const double latticeT = std::clamp(
                            p.latticeRiskIRE /
                            std::max(3.0, 0.35 * p.commonMagIRE + 1.0),
                            0.0,
                            1.0);
                        parallaxProtect = std::max(
                            spreadT,
                            std::max(latticeT, 1.0 - p.carrierCoherence));
                    }

                    const double g =
                        edgeGate[xi] * discGate *
                        (1.0 - brightColorProtect) *
                        (1.0 - parallaxProtect);
                    if (g > 0.0) {
                        const double blended =
                            carrierFit[xi] * (1.0 - g) + lsFit * g;
                        carrierFit[xi] = blended;
                        flattened[xi] = rawWhole[xi] - blended;
                        fitRow[xi] = static_cast<float>(blended);
                    }
                }
            }
        }
*/
        // Build the carrier-cancelled floor from every legal 4-sample mean of
        // the final flattened waveform.
        if (width >= 4) {
            const int meanCount = width - 3;
            for (int s = 0; s < meanCount; ++s) {
                slideMean4[s] =
                    0.25 * (flattened[s + 0] +
                            flattened[s + 1] +
                            flattened[s + 2] +
                            flattened[s + 3]);
            }

            for (int xi = 0; xi < width; ++xi) {
                double v[4] = {0.0, 0.0, 0.0, 0.0};
                int n = 0;
                const int sFirst = std::max(0, xi - 3);
                const int sLast  = std::min(xi, meanCount - 1);

                for (int s = sFirst; s <= sLast && n < 4; ++s)
                    v[n++] = slideMean4[s];

                double floor = refinedY[xi];
                if (n >= 4)
                    floor = medoid4Anchored(v[0], v[1], v[2], v[3], refinedY[xi]);
                else if (n == 3)
                    floor = median3(v[0], v[1], v[2]);
                else if (n == 2)
                    floor = 0.5 * (v[0] + v[1]);
                else if (n == 1)
                    floor = v[0];

                floorRow[xi] = static_cast<float>(floor);
            }
        } else {
            double mean = 0.0;
            for (int xi = 0; xi < width; ++xi)
                mean += flattened[xi];
            mean /= static_cast<double>(std::max(1, width));

            for (int xi = 0; xi < width; ++xi)
                floorRow[xi] = static_cast<float>(mean);
        }
    }
        carrierRetractionModelValid = true;
    }

    if (analysisOnly)
        return;
    if (!carrierRetractionModelValid)
        return;

    // ---------------------------------------------------------------
    // ---------------------------------------------------------------
    // Pass 1.7: FACT ANCHOR (user-approved 2026-08-01; position reversed
    // from the original facts-last placement on user direction: "put it
    // early enough to benefit the line cancellation").  At certified
    // samples the fit IS the exact carrier -- facts as facts, no gain,
    // no blend, no hull.  Stamped BEFORE the cross-line passes, so the
    // 1.75 bridge, Pass 1.9's +-2 evidence legs, and Pass 2's interline
    // cancellation all comb against FACT legs: same contract as the comb
    // candidates' cede/comb pair in a different shape -- def lines
    // idempotent/const, comp lines improved by combing against truth.
    // PERSISTENCE LAW: the passes downstream carry const-guards (the
    // bridge never adopts into a certified line; 1.9 never repairs a
    // certified sample; Pass 2 writes combedCarrier, not the fit; the
    // tone no-ops on covered frames), so a stamped fact survives to
    // publication structurally, not by re-stamping.
    //
    // REGIONAL FACT AUDIT (the return's architecture, tailored to the
    // witness): before stamping, the fit's pre-stamp error |fit - exact|
    // at certified samples is pooled per region (128x32, the tone's
    // geometry).  This is the truth ledger as a live instrument: it
    // grades the SOLVE where truth exists, and conditions trust in the
    // fit where truth doesn't (the between-anchor alpha merges consume
    // it bilinearly, seam-free).  Regions with no certified samples
    // carry NaN = no verdict = conduct unchanged.
    // Escape LDCD_FACT_FIT=0 (also inert at LDCD_CERT_1D=0 and under
    // --dg-discard, where no facts exist).
    // ---------------------------------------------------------------
    fitFactAuditNx = 0;
    fitFactAuditNy = 0;
    {
        static const bool factFitOn = []{
            const char *e = std::getenv("LDCD_FACT_FIT");
            return !(e && std::atoi(e) == 0);
        }();
        if (factFitOn && certifiedOneDLevel() >= 1 &&
            frameHasExactCoverage()) {
            const int nx = (width + 127) / 128;
            const int ny = (lastLine + 31) / 32;
            fitFactAuditNx = nx;
            fitFactAuditNy = ny;
            fitFactAuditIRE.assign((size_t)nx * ny,
                std::numeric_limits<float>::quiet_NaN());
            std::vector<double> aSum((size_t)nx * ny, 0.0);
            std::vector<long> aN((size_t)nx * ny, 0);
            // Referee repair (2026-08-02): keep the PRE-stamp fit so the
            // truth referees' "fit" column grades the estimator, not the
            // fact it was replaced by. Probe-only.
            static const bool fitStashOn = []{
                return std::getenv("LDCD_PROBE_DSREF") != nullptr ||
                       std::getenv("LDCD_PROBE_CCREF") != nullptr ||
                       std::getenv("LDCD_PROBE_ANTGRADE") != nullptr ||
                       std::getenv("LDCD_PROBE_TWEEN") != nullptr;
            }();
            float *fitStash = nullptr;
            if (fitStashOn) {
                probePreFactFit_flat.assign(
                    static_cast<size_t>(demodLines) * demodWidth,
                    std::numeric_limits<float>::quiet_NaN());
                fitStash = probePreFactFit_flat.data();
            }
            for (int line = firstLine; line < lastLine; ++line) {
                if (!certifiedDefLine(line)) continue;
                const float *exRow = exactCarrierRow(line);
                if (!exRow) continue;
                float *fitRow = carrierFit_flat.data()
                    + static_cast<size_t>(line) * demodWidth;
                const size_t rb = (size_t)(line / 32) * nx;
                for (int xi = 0; xi < width; ++xi) {
                    const float ex = exRow[left + xi];
                    if (!std::isfinite(ex)) continue;
                    const size_t r = rb + xi / 128;
                    aSum[r] += std::fabs((double)fitRow[xi] - (double)ex) *
                               invIreScale;
                    aN[r]++;
                    if (fitStash)
                        fitStash[static_cast<size_t>(line) * demodWidth +
                                 xi] = fitRow[xi];
                    fitRow[xi] = ex;
                }
            }
            for (size_t r = 0; r < aSum.size(); ++r)
                if (aN[r] >= 16)
                    fitFactAuditIRE[r] =
                        static_cast<float>(aSum[r] / aN[r]);
        }
    }

    // Pass 1.75: vertical amplitude continuity for the fit (the beam fix).
    //
    // Measured (LDCD_PROBE_OFFGRID): on LEGAL carrier the fit holds lock on
    // 95.7% (cube) / 99.9% (beach) of comb-strong windows -- the dropouts
    // cluster on fast diagonal envelopes crossing texture (the tractor
    // beam), where one 4-sample window's normal equations lose the carrier
    // while the SAME COLUMN one line away holds it. Each dropout leaves the
    // full lobe in retracted Y as a checker patch.
    //
    // The bridge is SELECTION UNDER EVIDENCE, never averaging: at a window
    // whose fit amplitude collapsed while the canonical raw bandpass is
    // strong and the energy is NOT proven-illegal (the grid's virtuous
    // refusals stay refused), each grammar-locked vertical neighbour's fit
    // is converted along the compatible rotation path -- partner locked
    // frame -> common 4fsc -> this line's locked frame, through the burst
    // phasors -- and remodulated on THIS line's basis. The candidate that
    // explains this line's own raw bandpass window better than the
    // collapsed fit does (and better than the other partner) is adopted
    // whole. Adopted samples are then RESTRICTED as always: residual-
    // consensus range and amplitude ceiling. No qualifying partner => the
    // dropout stands -- an honest hole, never a synthetic patch.
    //
    // flatFloor at bridged windows still reflects the pre-bridge fit
    // (sparse; revisit if Pass 2's gates misbehave at bridged columns).
    // Kill switch for A/B only: LDCD_FIT_BRIDGE=0.
    {
        static const bool bridgeOn = []{
            const char *s = std::getenv("LDCD_FIT_BRIDGE");
            return !s || std::atoi(s) != 0;
        }();
        static const double kBridgeRawStrongIRE = 8.0;
        static const double kBridgeCollapse     = 0.4;
        static const double kBridgePartnerHold  = 0.6;

        std::vector<std::uint8_t> bridged(width);
        auto lineBasis = [&](const CombCarrierGrammar *g, double *bI, double *bQ) {
            for (int p = 0; p < 4; ++p) {
                bI[p] = remodLockedToShiftedComposite(
                    1.0, 0.0, p, g->burstCos, g->burstSin,
                    spLUT_locked, cpLUT_locked);
                bQ[p] = remodLockedToShiftedComposite(
                    0.0, 1.0, p, g->burstCos, g->burstSin,
                    spLUT_locked, cpLUT_locked);
            }
        };
        // 4-sample window LS demod on a line basis; every window holds each
        // class once, so the normal matrix is the same for all windows.
        auto windowIQ = [&](const float *row, int x, int line,
                            const double *bI, const double *bQ,
                            double Sii, double Siq, double Sqq,
                            double &io, double &qo) -> bool {
            double SiY = 0.0, SqY = 0.0;
            for (int k = 0; k < 4; ++k) {
                const int cls = carrierSampleClass(line, left + x + k);
                SiY += bI[cls] * (double)row[x + k];
                SqY += bQ[cls] * (double)row[x + k];
            }
            const double det = Sii * Sqq - Siq * Siq;
            if (std::fabs(det) < 1e-12) return false;
            io = ( Sqq * SiY - Siq * SqY) / det;
            qo = (-Siq * SiY + Sii * SqY) / det;
            return true;
        };
        auto winRmsIRE = [&](const auto *row, int x) {
            double e = 0.0;
            for (int k = 0; k < 4; ++k)
                e += (double)row[x + k] * (double)row[x + k];
            return std::sqrt(e * 0.25) * invIreScale;
        };

        long bridgedWindows = 0;
        auto bridgeLine = [&](int line) {
            const CombCarrierGrammar *gL = carrierGrammarLine(line);
            if (!gL || !gL->grammarLocked) return;
            // Fact persistence: a certified line is const -- the bridge
            // never adopts into it (it serves as a fact PARTNER instead).
            if (certifiedDefLine(line)) return;
            const double *bpL = locked1DRawBandpass_line(line);
            float *fitL = carrierFit_flat.data()
                          + static_cast<size_t>(line) * demodWidth;
            const auto *anL = carrierAnalysis_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
            const double maxCarrierL =
                maxCarrierAmpIREFromScale(gL->carrierScale) * irescale;
            if (!bpL) return;

            double bIL[4], bQL[4];
            lineBasis(gL, bIL, bQL);
            double SiiL = 0, SiqL = 0, SqqL = 0;
            for (int p = 0; p < 4; ++p) {
                SiiL += bIL[p] * bIL[p];
                SiqL += bIL[p] * bQL[p];
                SqqL += bQL[p] * bQL[p];
            }

            const int partnerStep = carrierFrameVerticalAllowed(line) ? 1 : 2;
            std::fill(bridged.begin(), bridged.end(), std::uint8_t{0});

            for (int x = 0; x + 3 < width; ++x) {
                if (bridged[x]) continue;
                const double rawAmp = winRmsIRE(bpL, x);
                if (rawAmp < kBridgeRawStrongIRE) continue;
                if (winRmsIRE(fitL, x) >= kBridgeCollapse * rawAmp) continue;
                if (lddecode::carrierIllegalProof(
                        (double)anL[x + 1].carrierConformance,
                        (double)anL[x + 1].conformanceSupportFraction) >= 0.7)
                    continue;

                double errOwn = 0.0;
                for (int k = 0; k < 4; ++k) {
                    const double d = bpL[x + k] - (double)fitL[x + k];
                    errOwn += d * d;
                }

                double bestErr = errOwn;
                double bestC[4];
                bool haveBest = false;
                for (int dp = -partnerStep; dp <= partnerStep;
                     dp += 2 * partnerStep) {
                    const int lp = line + dp;
                    if (lp < firstLine || lp >= lastLine) continue;
                    const CombCarrierGrammar *gP = carrierGrammarLine(lp);
                    if (!gP || !gP->grammarLocked) continue;
                    const double *bpP = locked1DRawBandpass_line(lp);
                    const float *fitP = carrierFit_flat.data()
                                        + static_cast<size_t>(lp) * demodWidth;
                    const auto *anP = carrierAnalysis_flat.data()
                                      + static_cast<size_t>(lp) * demodWidth;
                    if (!bpP) continue;
                    // Partner must HOLD lock on its own raw band, lawfully.
                    const double rawAmpP = winRmsIRE(bpP, x);
                    if (rawAmpP < kBridgeRawStrongIRE) continue;
                    if (winRmsIRE(fitP, x) < kBridgePartnerHold * rawAmpP)
                        continue;
                    if (lddecode::carrierIllegalProof(
                            (double)anP[x + 1].carrierConformance,
                            (double)anP[x + 1].conformanceSupportFraction)
                            >= 0.7)
                        continue;

                    double bIP[4], bQP[4];
                    lineBasis(gP, bIP, bQP);
                    double SiiP = 0, SiqP = 0, SqqP = 0;
                    for (int p = 0; p < 4; ++p) {
                        SiiP += bIP[p] * bIP[p];
                        SiqP += bIP[p] * bQP[p];
                        SqqP += bQP[p] * bQP[p];
                    }
                    double iP, qP;
                    if (!windowIQ(fitP, x, lp, bIP, bQP,
                                  SiiP, SiqP, SqqP, iP, qP))
                        continue;
                    // Compatible rotation path: partner locked -> 4fsc ->
                    // this line's locked frame.
                    double i4, q4, iL, qL;
                    lockedTo4fsc(iP, qP, gP->burstCos, gP->burstSin, i4, q4);
                    fourfscToLocked(i4, q4, gL->burstCos, gL->burstSin,
                                    iL, qL);
                    double c[4];
                    double err = 0.0;
                    for (int k = 0; k < 4; ++k) {
                        const int cls = carrierSampleClass(line, left + x + k);
                        c[k] = iL * bIL[cls] + qL * bQL[cls];
                        const double d = bpL[x + k] - c[k];
                        err += d * d;
                    }
                    if (err < bestErr) {
                        bestErr = err;
                        for (int k = 0; k < 4; ++k) bestC[k] = c[k];
                        haveBest = true;
                    }
                }
                if (!haveBest) continue;

                for (int k = 0; k < 4; ++k) {
                    double nv = bestC[k];
                    const auto &pp = anL[x + k].parallax;
                    if (pp.residualValid)
                        nv = std::clamp(nv, (double)pp.residualLo,
                                            (double)pp.residualHi);
                    nv = std::clamp(nv, -maxCarrierL, maxCarrierL);
                    fitL[x + k] = static_cast<float>(nv);
                    bridged[x + k] = 1;
                }
                ++bridgedWindows;
            }
        };
        if (bridgeOn) {
            // Two sweeps, top-down then bottom-up: a window bridged in one
            // sweep becomes a QUALIFYING PARTNER for the adjacent line later
            // in sweep order, so cures cascade through multi-line dropout
            // clusters from both intact boundary lines inward. Still
            // selection under evidence at every step -- the cascade extends
            // only where each adoption beats the collapsed fit on this
            // line's own raw bandpass window.
            for (int line = firstLine; line < lastLine; ++line)
                bridgeLine(line);
            for (int line = lastLine - 1; line >= firstLine; --line)
                bridgeLine(line);
        }
        if (bridgeOn && std::getenv("LDCD_PROBE_OFFGRID"))
            std::fprintf(stderr, "[BRIDGE] windows adopted: %ld\n",
                         bridgedWindows);
    }

    // ---------------------------------------------------------------
    // RETIRED: the locked-1D repair return path.
    //
    // This used to add `locked1DParallaxRepairDelta` into carrierFit, to keep
    // the retraction tracking the repaired locked-1D carrier.  That
    // requirement belonged to the era when the fitted carrier was itself
    // promoted into the 1D intake; for an INDEPENDENT inverse-encoder view it
    // is backwards.  It made the dependency
    //
    //     raw analysis -> 1D decision and repair -> carrier-retracted view
    //
    // where the intended shape is a shared, application-neutral analysis
    // feeding ordinary locked 1D and the native retraction as SIBLINGS.  With
    // the delta folded in, a change made to improve 1D silently altered the
    // supposedly independent view, and the two could no longer serve as
    // checks on one another.
    //
    // Removing it also resolves a second inconsistency: flatFloor is built
    // from `raw - carrierFit` inside the model block above, while these
    // deltas were applied afterwards, so Pass 2's reach gates were evaluating
    // floors derived from a different carrier operand than the one they were
    // promoting.  Both operands are now the same model.
    //
    // Shared inputs remain legitimate and are unaffected: burst calibration,
    // carrier grammar, raw samples, the coarse-residual options in
    // carrierAnalysis, and the encoder bandwidth law are all
    // application-neutral.  What is gone is the correction selected
    // specifically for the locked-1D OUTPUT.  If a promoted product that
    // deliberately tracks the repaired 1D source is ever wanted, it should be
    // derived as its own buffer rather than by mutating the native model.
    // ---------------------------------------------------------------
    const lddecode::CombReachSourceFrame carrierFitSource =
        lddecode::makeCarrierFitScalarReachSource();

    // Shared soft gate for the vertical passes below (Pass 1.9 evidence
    // admission and Pass 2's reach/content-break guards read the same ramp).
    auto softReachGate = [](double diffIRE, double softIRE, double hardIRE) {
        if (diffIRE <= softIRE)
            return 1.0;
        if (diffIRE >= hardIRE)
            return 0.0;
        const double t = (diffIRE - softIRE) /
                         std::max(1e-9, hardIRE - softIRE);
        return 1.0 - (t * t * (3.0 - 2.0 * t));
    };

    // ---------------------------------------------------------------
    // Pass 1.9: vertical schedule law on the fit — proven-standing
    // exclusion (the title-bevel fix, 2026-08-01).
    //
    // The encoder bandwidth law (imposed once, at the Pass-1 model
    // boundary) is a LATERAL law: legal carrier lives in fSC±1.3 MHz.
    // The schedule adds a VERTICAL law the fit never honoured: legal
    // carrier INVERTS against every grammar-certified Opposite partner.
    // A component that MATCHES its Opposite partners is image-locked
    // luma standing in the carrier band — the bandpass leak of a strong
    // vertical image edge (same D²Y ripple on every line).  The fit
    // follows raw and absorbs that leak; every full-subtraction product
    // (native, the anchor ladder's fit rung) then confiscates it from Y,
    // and the renderer's complement demodulates it as line-alternating
    // false colour (the demod axis flips per line, so standing composite
    // paints alternating green/magenta).  Measured on the Emissary title
    // bevel: retracted Y sat a mean 5.6 IRE below comb Y at the leak
    // columns; every election share given the fit-derived candidate
    // rendered as the alternation.
    //
    // Shape of the law, and why it is imposed on the MODEL:
    //   * The pedestal law forbids scaling the withdrawal (raw − w·fit
    //     was the retired floating-pedestal product).  So the fit itself
    //     is repaired: the PROVEN-standing component is removed from the
    //     model, and what the encoder could not have modulated never
    //     enters the carrier — Y keeps it by construction, upstream of
    //     every election and every consumer.
    //   * Proof, not detection: the signed schedule correlation of the
    //     fit against FIELD-PURE legs only (±2 same-field Opposite; the
    //     ±1 interfield leg is 20 ms away and motion scrambles its
    //     relation — the beach garment lesson), integrated VERTICALLY
    //     down the column, pointwise laterally.  The lateral envelope
    //     aperture Pass 2's corroboration uses is WRONG for this law and
    //     was measured failing on the bevel itself: a standing leak is a
    //     1-6 px COLUMN (it is luma; only chroma is bound by the 1.3 MHz
    //     envelope limit), and lateral smoothing diluted its pointwise
    //     +0.9 match with the surrounding legal letter chroma down to
    //     sc ~0.34, under the knee.  Vertically the aperture and the
    //     evidence agree: the leak persists down the image edge (a
    //     letter stem is 50+ lines), so the proof integrates where the
    //     phenomenon lives.  A minimum evidence run (6 contributing leg
    //     lines) keeps transient flukes out.  The license is the mirror
    //     of scheduleAlternationLicense: it opens only where the fit
    //     provably MATCHES where the schedule demands inversion
    //     (sc ≥ +0.5, full at +0.9).  Everything short of proof leaves
    //     the model untouched — no confidence hedging, the unproven fit
    //     stays committed.
    //   * The standing estimator is the waveform mid-sum s = ½(fit +
    //     opp): exact for a chroma+standing mix (c+s with partner −c+s
    //     gives s exactly), zero for pure legal chroma.  The license is
    //     envelope-smooth by construction, so lic·s cannot manufacture
    //     out-of-band sidebands (no fast-gain AM).
    //   * HULL direction: the subtraction may only REMOVE carrier —
    //     the repaired sample is clamped to the same sign and no larger
    //     magnitude than the fit (a true per-sample projection; what it
    //     removes returns to Y via raw − fit, which is lawful).
    //   * flatFloor keeps the pre-law fit (same sparse-staleness caveat
    //     Pass 1.75 carries; revisit if Pass 2's gates misbehave at
    //     repaired columns).
    // Known cost, accepted with eyes open: sustained diagonal chroma at
    // slopes that rotate the carrier 180° per 2 lines reads as matching
    // at ±2 and would be confiscated toward Y where the envelope
    // correlation stays ≥ +0.5 over a full coherence length — the same
    // verdict Pass 2's ownership license already renders on that
    // content ("energy stays in Y").  Kill switch for A/B only:
    // LDCD_STANDING_LAW=0.  Census under LDCD_PROBE_STANDING=1.
    // ---------------------------------------------------------------
    {
        static const bool standingLawOn = []{
            const char *s = std::getenv("LDCD_STANDING_LAW");
            return !s || std::atoi(s) != 0;
        }();
        static const bool standingProbe =
            std::getenv("LDCD_PROBE_STANDING") != nullptr;
        // Windowed per-pixel dump (frame line / active column, same
        // convention as LDCD_PY_*): sc, lic, s and the operands.
        static const auto stEnvInt = [](const char *n, int f) {
            const char *s = std::getenv(n); return s ? std::atoi(s) : f;
        };
        static const int stL0 = stEnvInt("LDCD_STANDING_L0", -1);
        static const int stL1 = stEnvInt("LDCD_STANDING_L1", -1);
        static const int stC0 = stEnvInt("LDCD_STANDING_C0", -1);
        static const int stC1 = stEnvInt("LDCD_STANDING_C1", -1);
        const bool stDump = stL0 >= 0 && stC0 >= 0;
        if (standingLawOn) {
            const double ampFloorS = 3.0 * irescale;    // 3 IRE
            const double powFloorS = ampFloorS * ampFloorS;
            constexpr int kVertHalf = 8;   // vertical window: ±8 lines
            constexpr int kMinLegLines = 6; // minimum evidence run

            // Sweep 1: per-line pointwise leg products against the
            // ORIGINAL fit (nothing is modified until sweep 2, so plane
            // reads are order-safe).  Five planes over the active span.
            const int nLines = lastLine - firstLine;
            const size_t planeN = static_cast<size_t>(nLines) * width;
            std::vector<float> pP(planeN, 0.0f);   // Σ g·fit·leg
            std::vector<float> pS(planeN, 0.0f);   // Σ g·fit²
            std::vector<float> pL(planeN, 0.0f);   // Σ g·leg²
            std::vector<float> pO(planeN, 0.0f);   // Σ g·leg
            std::vector<float> pW(planeN, 0.0f);   // Σ g

            for (int line = firstLine; line < lastLine; ++line) {
                const CombCarrierGrammar *gL = carrierGrammarLine(line);
                if (!gL || !gL->grammarLocked)
                    continue;
                const float *fitRow = carrierFit_flat.data()
                                      + static_cast<size_t>(line) * demodWidth;
                const float *floorRow = flatFloor_flat.data()
                                        + static_cast<size_t>(line) * demodWidth;
                float *rP = pP.data() +
                    static_cast<size_t>(line - firstLine) * width;
                float *rS = pS.data() +
                    static_cast<size_t>(line - firstLine) * width;
                float *rL = pL.data() +
                    static_cast<size_t>(line - firstLine) * width;
                float *rO = pO.data() +
                    static_cast<size_t>(line - firstLine) * width;
                float *rW = pW.data() +
                    static_cast<size_t>(line - firstLine) * width;

                for (int dir = -1; dir <= 1; dir += 2) {
                    const int targetLine = line + 2 * dir;
                    if (targetLine < firstLine || targetLine >= lastLine)
                        continue;
                    const CombCarrierGrammar *gT =
                        carrierGrammarLine(targetLine);
                    if (!gT || !gT->grammarLocked)
                        continue;
                    const lddecode::CombReachReply reach =
                        combReachIndex.query(
                            {line, targetLine, left, left,
                             lddecode::CombReachUse::FieldScalarCancel,
                             carrierFitSource});
                    if (!(reach.allowScalarCancel && reach.mayBecomeVideo &&
                          reach.carrierRelation ==
                              lddecode::CarrierPhaseRelation::Opposite))
                        continue;
                    const float *legFit = carrierFit_flat.data()
                        + static_cast<size_t>(targetLine) * demodWidth;
                    const float *legFloor = flatFloor_flat.data()
                        + static_cast<size_t>(targetLine) * demodWidth;
                    for (int xi = 0; xi < width; ++xi) {
                        const double legBreakIRE =
                            std::fabs(static_cast<double>(floorRow[xi]) -
                                      static_cast<double>(legFloor[xi])) *
                            invIreScale;
                        const double g = softReachGate(legBreakIRE, 3.0, 10.0);
                        if (g <= 0.0)
                            continue;
                        const double c = static_cast<double>(fitRow[xi]);
                        const double n = static_cast<double>(legFit[xi]);
                        rP[xi] += static_cast<float>(g * c * n);
                        rS[xi] += static_cast<float>(g * c * c);
                        rL[xi] += static_cast<float>(g * n * n);
                        rO[xi] += static_cast<float>(g * n);
                        rW[xi] += static_cast<float>(g);
                    }
                }
            }

            // Sweep 2: sliding vertical sums per column, license, repair.
            // In-place emit is safe: line L is only ever modified at step
            // L, and every windowed read comes from the sweep-1 planes.
            long provenPx = 0;
            double confiscatedIRE = 0.0;

            std::vector<double> vP(width), vS(width), vL(width), vN(width);
            auto addLine = [&](int l, double sgn) {
                if (l < firstLine || l >= lastLine) return;
                const size_t r0 = static_cast<size_t>(l - firstLine) * width;
                for (int xi = 0; xi < width; ++xi) {
                    vP[xi] += sgn * pP[r0 + xi];
                    vS[xi] += sgn * pS[r0 + xi];
                    vL[xi] += sgn * pL[r0 + xi];
                    if (pW[r0 + xi] > 0.0f)
                        vN[xi] += sgn;
                }
            };
            std::fill(vP.begin(), vP.end(), 0.0);
            std::fill(vS.begin(), vS.end(), 0.0);
            std::fill(vL.begin(), vL.end(), 0.0);
            std::fill(vN.begin(), vN.end(), 0.0);
            for (int l = firstLine; l <= firstLine + kVertHalf &&
                                    l < lastLine; ++l)
                addLine(l, +1.0);

            for (int line = firstLine; line < lastLine; ++line) {
                float *fitRow = carrierFit_flat.data()
                                + static_cast<size_t>(line) * demodWidth;
                const size_t r0 =
                    static_cast<size_t>(line - firstLine) * width;
                const CombCarrierGrammar *gL = carrierGrammarLine(line);
                // Fact persistence: never repair a certified sample (the
                // stamped fact is const; its legs still serve as evidence).
                const float *exGuard = certifiedDefLine(line)
                    ? exactCarrierRow(line) : nullptr;

                if (gL && gL->grammarLocked) {
                    for (int xi = 0; xi < width; ++xi) {
                        if (exGuard &&
                            std::isfinite(exGuard[left + xi]))
                            continue;
                        const double selfE = vS[xi];
                        const double legE = vL[xi];
                        const bool observable =
                            vN[xi] >= kMinLegLines &&
                            selfE > 0.0 && legE > 0.0 &&
                            selfE / std::max(1.0, vN[xi]) >= powFloorS;
                        const double sc = observable
                            ? vP[xi] / std::sqrt(selfE * legE)
                            : std::numeric_limits<double>::quiet_NaN();
                        const double licHere = (observable && sc == sc)
                            ? lddecode::scheduleAlternationLicense(-sc)
                            : 0.0;
                        const double wHere =
                            static_cast<double>(pW[r0 + xi]);
                        const double fit = static_cast<double>(fitRow[xi]);
                        double s = std::numeric_limits<double>::quiet_NaN();
                        double nf = fit;
                        if (licHere > 0.0 && wHere > 0.0) {
                            s = 0.5 * (fit +
                                static_cast<double>(pO[r0 + xi]) / wHere);
                            nf = fit - licHere * s;
                            // Removal-only hull: same sign, no larger
                            // magnitude — a true per-sample projection.
                            if (fit >= 0.0)
                                nf = std::clamp(nf, 0.0, fit);
                            else
                                nf = std::clamp(nf, fit, 0.0);
                        }
                        if (stDump && line >= stL0 && line <= stL1 &&
                            xi >= stC0 && xi <= stC1) {
                            std::fprintf(stderr,
                                "STDUMP line=%d xi=%d fit=%.2f sc=%.3f "
                                "lic=%.3f s=%.2f nRun=%.0f\n",
                                line, xi, fit * invIreScale, sc,
                                licHere, s * invIreScale, vN[xi]);
                        }
                        if (nf != fit) {
                            fitRow[xi] = static_cast<float>(nf);
                            if (standingProbe) {
                                ++provenPx;
                                confiscatedIRE +=
                                    std::fabs(fit - nf) * invIreScale;
                            }
                        }
                    }
                }

                // Slide the vertical window.
                addLine(line - kVertHalf, -1.0);
                addLine(line + kVertHalf + 1, +1.0);
            }
            if (standingProbe)
                std::fprintf(stderr,
                    "[STANDING] repaired px=%ld mean removed %.3f IRE\n",
                    provenPx,
                    provenPx > 0 ? confiscatedIRE / provenPx : 0.0);
        }
    }

    // ---------------------------------------------------------------
    // Pass 2: interline comb on carrierFit → combedCarrier.
    //
    // Leg roster, not a fixed pairing.  The previous form admitted only
    // grammar-Opposite ±1 partners; on the NTSC schedule those alternate
    // sides by line parity (even lines pair down, odd lines pair up, never
    // both), so the stage degenerated into fixed disjoint 2-line couples —
    // no comparison ever crossed a couple boundary, and every decision
    // rendered at a 2-line quantum (measured on the Borg cube: state runs
    // of median 1 line, boundary Y-steps 2x the picture's natural line
    // variation).  A vertical structure was combed through different
    // arithmetic on adjacent lines.
    //
    // The comb-like replacement gathers every grammar-certified Opposite
    // partner among {±1, ±2}: the ±1 partner (whichever side the schedule
    // makes Opposite, interfield) and both ±2 partners (same-field, always
    // Opposite on the NTSC schedule — relations compose Opp∘Same = Opp).
    // Every line now has both-side vertical support, so the couple quantum
    // dissolves; which legs exist is the grammar's answer per pair, never a
    // structural assumption here.  Legality: FrameScalarCancel for the
    // interfield rung (carries the edit-split frame-vertical block),
    // FieldScalarCancel for the same-field rung.  Only Opposite relations
    // cancel — a Same partner sees chroma and image-locked alien with the
    // same sign and cannot discriminate, so it contributes nothing here.
    // (softReachGate is shared with Pass 1.9, defined above.)
    // ---------------------------------------------------------------

    // Diagnostic gate-chain dump (evidence-only, same family as LDCD_YVIEW).
    // LDCD_PASS2_DUMP=<prefix> writes every per-pixel term of the Pass-2
    // engage/disengage decision to <prefix>_NNN.bin, one file per frame
    // processed, so the mechanism can be studied offline instead of inferred
    // from the rendered output.  Channel plan (float32, [line][ch][x]):
    //   0 eligCenter    1 legCount(elig at xi)  2 legMask(b0 up2,b1 up1,
    //                                             b2 dn1,b3 dn2)
    //   3 gUp2raw       4 gUp1raw       5 gDn1raw       6 gDn2raw
    //   7 wSumRaw       8 neighborFit   9 strength(min(1,wSum))
    //  10 ownedFallback 11 corrCode(+3 quiet, +4 unobservable, else signed corr)
    //  12 fitRow       13 combRow      14 corroboration(envelope-scale w)
    // Sync-tone actuator: fit is complete here; every consumer below
    // (comb pass, legs, retraction output) reads the rotated fit.
    applyToneToFit(prevF);

    static const char *pass2DumpPrefix = std::getenv("LDCD_PASS2_DUMP");
    constexpr int kP2D = 15;
    std::vector<float> p2dump;
    if (pass2DumpPrefix)
        p2dump.assign(static_cast<size_t>(lastLine - firstLine) * kP2D * width,
                      0.0f);
    auto p2rec = [&](int line, int ch, int xi, double v) {
        p2dump[(static_cast<size_t>(line - firstLine) * kP2D + ch) * width +
               xi] = static_cast<float>(v);
    };

    // Per-leg raw-gate scratch, hoisted across lines.  Slot order is fixed
    // (up2, up1, dn1, dn2) so the dump channels stay identifiable; an
    // absent leg keeps null pointers and a zeroed gate row.
    constexpr int kNLegs = 4;
    std::vector<double> legGateScratch[kNLegs];
    for (int k = 0; k < kNLegs; ++k)
        legGateScratch[k].assign(static_cast<size_t>(width), 0.0);

    // Corroboration evidence scratch (see the envelope-scale corroboration
    // block inside the loop), hoisted across lines like the gate scratch.
    std::vector<double> corrSelf(static_cast<size_t>(width), 0.0);
    std::vector<double> corrNum(static_cast<size_t>(width), 0.0);
    std::vector<double> corrDen(static_cast<size_t>(width), 0.0);
    std::vector<double> corrScr0(static_cast<size_t>(width), 0.0);
    std::vector<double> corrScr1(static_cast<size_t>(width), 0.0);
    std::vector<double> corrScr2(static_cast<size_t>(width), 0.0);

    for (int line = firstLine; line < lastLine; ++line) {
        const float *fitRow = carrierFit_flat.data()
                              + static_cast<size_t>(line) * demodWidth;
        const float *floorRow = flatFloor_flat.data()
                                + static_cast<size_t>(line) * demodWidth;
        float *combRow = combedCarrier_flat.data()
                         + static_cast<size_t>(line) * demodWidth;
        float *wRow = carrierCorroboration_flat.data()
                      + static_cast<size_t>(line) * demodWidth;

        const CombCarrierGrammar *grammar = carrierGrammarLine(line);
        if (!grammar || !grammar->grammarLocked) {
            std::fill(combRow, combRow + width, 0.0f);
            // No grammar, no corroboration: the published product withdraws
            // nothing here (fail closed, same term as the license).
            std::fill(wRow, wRow + width, 0.0f);
            continue;
        }

        const float *eligRow =
            carrierEligibility_flat.data() + static_cast<size_t>(line) * demodWidth;

        // Build the leg roster for this line.
        struct FitLeg {
            const float *fit = nullptr;
            const float *lumaFloor = nullptr;
            const float *elig = nullptr;   // graded participation [0,1]
            double *wRaw = nullptr;
            bool present = false;
        };
        FitLeg legs[kNLegs];
        constexpr int legOffset[kNLegs] = {-2, -1, +1, +2};
        constexpr lddecode::CombReachUse legUse[kNLegs] = {
            lddecode::CombReachUse::FieldScalarCancel,
            lddecode::CombReachUse::FrameScalarCancel,
            lddecode::CombReachUse::FrameScalarCancel,
            lddecode::CombReachUse::FieldScalarCancel,
        };
        for (int k = 0; k < kNLegs; ++k) {
            const int targetLine = line + legOffset[k];
            legs[k].wRaw = legGateScratch[k].data();
            if (targetLine < firstLine || targetLine >= lastLine)
                continue;
            const CombCarrierGrammar *g = carrierGrammarLine(targetLine);
            if (!g || !g->grammarLocked)
                continue;
            const lddecode::CombReachReply reach = combReachIndex.query(
                {line, targetLine, left, left, legUse[k], carrierFitSource});
            if (!(reach.allowScalarCancel && reach.mayBecomeVideo &&
                  reach.carrierRelation ==
                      lddecode::CarrierPhaseRelation::Opposite))
                continue;
            legs[k].fit = carrierFit_flat.data() +
                          static_cast<size_t>(targetLine) * demodWidth;
            legs[k].lumaFloor = flatFloor_flat.data() +
                                static_cast<size_t>(targetLine) * demodWidth;
            legs[k].elig = carrierEligibility_flat.data() +
                           static_cast<size_t>(targetLine) * demodWidth;
            legs[k].present = true;
        }

        // -------------------------------------------------------------
        // Envelope-scale schedule corroboration for the PUBLISHED product.
        //
        // Same evidence family as the ownership license below — relation-
        // folded alternation of this line's fit against its certified-
        // Opposite legs, mapped through scheduleAlternationLicense() — but
        // integrated at the aperture the encoder bandwidth law defines.
        // The license's one-cycle window returns chance-level verdicts on
        // dithered content (cube box: 43% of pixels "inverting", a coin
        // flip per pixel); the same correlation integrated at envelope
        // scale is decisive, because a legal envelope cannot vary faster
        // than ~1.3 MHz and so its alternation cannot flicker at pixel
        // pitch.
        //
        // LP(fit·leg) extracts the relation-folded envelope inner product
        // directly in the raw domain — the carrier-rate cross terms land
        // at 2fSC and are rejected by the smoothing — so no basis or sign
        // convention is applied a second time.  The smoother is the
        // encoder's own envelope filter (feasibleband.h), applied twice
        // for one full coherence length of support: the weight therefore
        // cannot vary faster than the envelope it scales, and w·fit stays
        // inside the legal band by construction — the fast-gain AM defect
        // documented at the Pass-2 emit cannot occur on this product.
        //
        // The proof standard is uniform: an unproven fit withdraws NOTHING,
        // quiet included.  This deliberately breaks with the ownership
        // license's quiet-licenses-at-1 convention: that convention is
        // about confiscating a ~0 REMAINDER, but here w scales a published
        // withdrawal, and the Pass-2 mechanism study already measured that
        // subtracting 2-3 IRE of absorbed alien on scattered quiet lines
        // is not harmless to column straightness.  Measured on the cube
        // box: quiet-at-1 put 0.65 IRE of unlawful withdrawal through
        // (48% lawful share); quiet-at-0 withdraws less but lawfully.
        // Sub-floor legal chroma stays in Y at <= 3 IRE — the loud lawful
        // carrier is what the corroboration exists to prove.  Eligibility
        // is NOT consulted here: schedule-conformance already downweighted
        // the solve that produced the fit, and counting the same evidence
        // twice is how verdicts start flipping at one line's bit.
        // -------------------------------------------------------------
        {
            const double ampFloorC = 3.0 * irescale;   // 3 IRE envelope
            const double powFloorC = 0.5 * ampFloorC * ampFloorC;

            // Two passes of the encoder envelope filter; out == in is fine
            // (the intermediate lives in corrScr2), out == corrScr2 is not.
            auto smoothEnv = [&](const double *in, double *out) {
                lddecode::projectExpressibleChromaEnvelope(
                    in, nullptr, width, corrScr2.data());
                lddecode::projectExpressibleChromaEnvelope(
                    corrScr2.data(), nullptr, width, out);
            };

            for (int xi = 0; xi < width; ++xi) {
                const double c = static_cast<double>(fitRow[xi]);
                corrScr0[xi] = c * c;
            }
            smoothEnv(corrScr0.data(), corrSelf.data());

            std::fill(corrNum.begin(), corrNum.begin() + width, 0.0);
            std::fill(corrDen.begin(), corrDen.begin() + width, 0.0);

            for (int k = 0; k < kNLegs; ++k) {
                if (!legs[k].present)
                    continue;
                // FIELD-PURE evidence only: the ±2 legs are the same field,
                // simultaneous with this line.  The ±1 interfield leg is
                // 20 ms away — under vertical motion it presents the other
                // field's displaced content, whose carrier relation is
                // scrambled by the motion phase (measured on the beach
                // garment edges: ±2 corr −1 while ±1 corr +1 at the same
                // pixel).  Folding it into the aggregate made w flicker at
                // line pitch across moving chroma edges, and a line-rate w
                // on a strong carrier renders a checkerboard.  Grammar
                // legality is not content simultaneity; temporal evidence
                // belongs to the temporal machinery, not this weight.
                if (legOffset[k] == -1 || legOffset[k] == +1)
                    continue;
                const float *legFit = legs[k].fit;
                for (int xi = 0; xi < width; ++xi) {
                    const double c = static_cast<double>(fitRow[xi]);
                    const double n = static_cast<double>(legFit[xi]);
                    corrScr0[xi] = c * n;
                    corrScr1[xi] = n * n;
                }
                smoothEnv(corrScr0.data(), corrScr0.data());
                smoothEnv(corrScr1.data(), corrScr1.data());
                for (int xi = 0; xi < width; ++xi) {
                    if (corrScr1[xi] < powFloorC)
                        continue;   // this leg is unobservable here
                    // Content-break guard, same carrier-free criterion the
                    // cancellation trusts (flatFloor delta): a leg across a
                    // luma content boundary is a straddling window, and a
                    // straddling window is not evidence for any pixel — its
                    // own corr is exactly the noisy verdict that wandered at
                    // edge strips.  Graded, so authority fades instead of
                    // flipping; both legs faded -> unobservable -> w=0,
                    // uniformly across the strip.
                    const double legBreakIRE =
                        std::fabs(static_cast<double>(floorRow[xi]) -
                                  static_cast<double>(legs[k].lumaFloor[xi])) *
                        invIreScale;
                    const double g = softReachGate(legBreakIRE, 3.0, 10.0);
                    if (g <= 0.0)
                        continue;
                    corrNum[xi] += g * corrScr0[xi];
                    corrDen[xi] += g * std::sqrt(corrSelf[xi] * corrScr1[xi]);
                }
            }

            for (int xi = 0; xi < width; ++xi) {
                double w;
                if (corrSelf[xi] < powFloorC)
                    w = 0.0;    // quiet: unproven, withdraw nothing
                else if (corrDen[xi] <= 0.0)
                    w = 0.0;    // loud, no observable partner: fail closed
                else
                    w = lddecode::scheduleAlternationLicense(
                        corrNum[xi] / corrDen[xi]);
                wRow[xi] = static_cast<float>(w);
                if (pass2DumpPrefix)
                    p2rec(line, 14, xi, w);
            }
        }

        // Reach gate: per-pixel cancellation weight toward one leg.
        //
        // Mismatch and amplitude are evaluated as 2-sample quadrature
        // envelopes, never instantaneous samples: |fit + neighbor| dips to
        // zero twice per carrier cycle, so a per-sample gate oscillates at
        // carrier rate.  On slanted chroma (~2 px/line drift = 180° carrier
        // rotation per line) the grammar-Opposite neighbor presents
        // SAME-SIGN chroma; the envelope gate sees a constant 2A mismatch
        // and stands the cancellation down consistently, where the
        // per-sample gate cancelled real chroma in carrier-rate bursts —
        // a checkerboard manufactured inside combedCarrier itself.
        auto reachGate = [&](int xi, const FitLeg &leg,
                             double *lumaGateOut = nullptr,
                             double *carrGateOut = nullptr) {
            if (!leg.present)
                return 0.0;
            // No per-pixel eligibility test here: the fit now exists at every
            // grammar-locked sample (schedule doubt downweights the solve, it
            // no longer amputates the estimate), and participation is applied
            // once, at the weight/emit stage — never twice.

            const double lumaDiffIRE =
                std::fabs(static_cast<double>(floorRow[xi]) -
                          static_cast<double>(leg.lumaFloor[xi])) * invIreScale;

            const int xj = (xi + 1 < width) ? xi + 1
                         : (xi > 0 ? xi - 1 : xi);
            const double c0 = static_cast<double>(fitRow[xi]);
            const double c1 = static_cast<double>(fitRow[xj]);
            const double n0 = static_cast<double>(leg.fit[xi]);
            const double n1 = static_cast<double>(leg.fit[xj]);

            // Direct sqrt, not std::hypot: these are bounded carrier-fit
            // samples (IRE-scale, no overflow risk), so hypot's IEEE
            // over/underflow guarding is pure cost in a 6-call-per-pixel
            // inner loop.
            const double sum0 = c0 + n0, sum1 = c1 + n1;
            const double carrierMismatchIRE =
                std::sqrt(sum0 * sum0 + sum1 * sum1) * invIreScale;
            const double carrierAmpIRE = 0.5 *
                (std::sqrt(c0 * c0 + c1 * c1) +
                 std::sqrt(n0 * n0 + n1 * n1)) * invIreScale;

            double lumaGate = softReachGate(lumaDiffIRE, 3.0, 10.0);

            const double carrierSoftIRE = std::max(3.0, 0.25 * carrierAmpIRE);
            const double carrierHardIRE = std::max(10.0, 0.80 * carrierAmpIRE);
            double carrierGate =
                softReachGate(carrierMismatchIRE,
                              carrierSoftIRE,
                              carrierHardIRE);

            // A/B switch (LD_P2_CARRGATE=0) for the carrier-mismatch gate.
            // It tests |fit + neighbour|, which is small only when the partner
            // INVERTS.  Image-locked alien luma matches its partner instead,
            // so the mismatch reads ~2A and the gate closes -- measured at
            // 0.001 on the cube face, i.e. fully shut on 75% of loud pixels,
            // which is exactly the content 0.5*(fit - neighbour) resolves
            // correctly.  lumaGate remains as the independent, carrier-free
            // content-break guard when this is disabled.
            static const bool carrGateEnabled = []{
                const char *s = std::getenv("LD_P2_CARRGATE");
                return !(s && s[0] == '0');
            }();
            if (!carrGateEnabled)
                carrierGate = 1.0;

            if (lumaGateOut) *lumaGateOut = lumaGate;
            if (carrGateOut) *carrGateOut = carrierGate;
            return lumaGate * carrierGate;
        };

        // -----------------------------------------------------------------
        // Fused reach-gate sweep: Pass A materializes each leg's raw gates
        // (the stencil below reads ±2 horizontal neighbors), Pass B applies
        // the 5-tap smooth, the decision blend, the license, and emits.
        // -----------------------------------------------------------------

        // Pass A: raw gates per leg.
        for (int xi = 0; xi < width; ++xi) {
            for (int k = 0; k < kNLegs; ++k)
                legs[k].wRaw[xi] = reachGate(xi, legs[k]);
            if (pass2DumpPrefix) {
                p2rec(line, 0, xi, eligRow[xi]);
                int mask = 0, count = 0;
                for (int k = 0; k < kNLegs; ++k) {
                    if (legs[k].present && legs[k].elig[xi] > 0.5f) {
                        mask |= 1 << k;
                        ++count;
                    }
                    p2rec(line, 3 + k, xi, legs[k].wRaw[xi]);
                }
                p2rec(line, 1, xi, count);
                p2rec(line, 2, xi, mask);
            }
        }

        // Pass B: inline 5-tap smooth + decision blend + combRow output.
        // The smooth kernel is [1,2,3,2,1] (sum = 9 in the interior).
        // At edges, the kernel is clamped and the divisor adjusts.
        constexpr double kWeights[5] = {1.0, 2.0, 3.0, 2.0, 1.0};

        for (int xi = 0; xi < width; ++xi) {
            // A refused center sample owns no carrier column.  Keep the DQ
            // explicit at the interline publication boundary as well as in
            // the fit, so smoothing of neighboring reach gates cannot give
            // it a route back into video.  Participation is graded now: the
            // emit below scales by it, so a decisively proven-illegal centre
            // (participation 0) still publishes no carrier, while doubt short
            // of proof fades the carrier's authority instead of flipping it
            // at one line's verdict bit.
            const double centerParticipation =
                static_cast<double>(eligRow[xi]);
            if (centerParticipation <= 0.0) {
                combRow[xi] = 0.0f;
                continue;
            }

            // Ownership weight for the un-cancelled remainder (see the emit
            // block below): the OPERAND schedule-compatibility license.  The
            // confiscation operand is this line's carrierFit; the license
            // tests the operand itself against each certified-Opposite
            // leg's fit.  On-schedule chroma inverts (signed corr -> -1:
            // licensed, keep the confiscation, microscopic runs included); a
            // fit that absorbed alien luma MATCHES its partner where the
            // schedule demands inversion (cube grid: corr -> +1, license 0,
            // energy stays in Y).  A quiet operand licenses at 1
            // (confiscating ~nothing is harmless); a loud operand with no
            // observable partner fails closed.
            //
            // The window is one full 4-sample carrier cycle — the same
            // geometry as the residual-carrier operand license in produceY —
            // not a 2-sample pair.  The 2-sample dot product was measured
            // moving 0.145 per line against a license band only 0.4 wide,
            // parking 73% of loud cube-face pixels in the ±0.5 dead zone:
            // an aperture problem, not an evidence problem.
            double ownedFallback;
            double p2corrCode = 4.0;       // dump-only: +4 = unobservable
            {
                // Centered 5-tap, half-weight ends (integer centroid);
                // total weight stays 4 so the cycle-energy floor keeps
                // its meaning.
                double e0 = 0.0;
                for (int k = -2; k <= 2; ++k) {
                    const int j = std::clamp(xi + k, 0, width - 1);
                    const double w = (k == -2 || k == 2) ? 0.5 : 1.0;
                    const double c = static_cast<double>(fitRow[j]);
                    e0 += w * c * c;
                }
                const double ampFloor = 3.0 * irescale;      // 3 IRE envelope
                // Energy of a carrier of envelope A over one full cycle is
                // 2*A^2 (cos^2 + sin^2 pairs), so the 3 IRE envelope floor
                // keeps its meaning on the 4-sample window.
                const double eFloor = 2.0 * ampFloor * ampFloor;
                if (e0 < eFloor) {
                    ownedFallback = 1.0;   // harmless confiscation
                    p2corrCode = 3.0;      // dump-only: +3 = quiet operand
                } else {
                    // Integrate the legs into ONE energy-weighted signed
                    // correlation rather than taking the most-legal single
                    // observation.  With a one-partner roster the two forms
                    // coincide; with three noisy legs, min() lets any single
                    // draw below -0.5 license the confiscation (measured:
                    // open rate doubled on the cube face), which inverts the
                    // proof standard — confiscation requires the observations
                    // TOGETHER to prove carrier-law inversion.  Real chroma
                    // inverts against every Opposite leg, so integration
                    // costs it nothing; a decorrelated alien fit cannot
                    // manufacture a negative aggregate from one lucky leg.
                    double dotSum = 0.0, normSum = 0.0;
                    for (int k = 0; k < kNLegs; ++k) {
                        if (!legs[k].present)
                            continue;
                        const float *pf = legs[k].fit;
                        double dot = 0.0, eP = 0.0;
                        for (int j = -2; j <= 2; ++j) {
                            const int jj = std::clamp(xi + j, 0, width - 1);
                            const double w = (j == -2 || j == 2) ? 0.5 : 1.0;
                            const double c = static_cast<double>(fitRow[jj]);
                            const double n = static_cast<double>(pf[jj]);
                            dot += w * c * n;
                            eP += w * n * n;
                        }
                        if (eP < eFloor)
                            continue;
                        dotSum += dot;
                        normSum += std::sqrt(e0 * eP);
                    }
                    if (normSum > 0.0) {
                        const double sc = dotSum / normSum;
                        ownedFallback =
                            lddecode::scheduleAlternationLicense(sc);
                        p2corrCode = sc;
                    } else {
                        ownedFallback = 0.0;  // loud, unobservable: fail closed
                    }
                }

                // Ownership FLOOR (LDCD_OWNERSHIP_FLOOR, default 0 = no
                // change).  A single scalar interpolating the standdown
                // remainder between the conservative promoted product (floor
                // 0: unproven carrier is left in Y) and a more aggressive
                // withdrawal (floor 1: the fit's full standdown remainder is
                // always confiscated, matching native's unconditional
                // withdrawal in that band).  Measured: promoted manufactures
                // ~2x less illegal energy than native at every Pass-2
                // strength band, so this is a swept aggression/cleanliness
                // trade, not a correctness fix -- it moves along the same
                // fast-varying-gain amplitude-modulation defect the encoder
                // bandwidth law names, it does not resolve it.
                static const double ownershipFloor = []{
                    const char *s = std::getenv("LDCD_OWNERSHIP_FLOOR");
                    const double v = s ? std::atof(s) : 0.0;
                    return std::clamp(v, 0.0, 1.0);
                }();
                if (ownershipFloor > 0.0)
                    ownedFallback = std::max(ownedFallback, ownershipFloor);
            }

            // Inline 5-tap smooth of each leg's raw gates.
            double legW[kNLegs];
            {
                double sumW = 0.0;
                double sums[kNLegs] = {0.0, 0.0, 0.0, 0.0};
                for (int dx = -2; dx <= 2; ++dx) {
                    const int xx = std::clamp(xi + dx, 0, width - 1);
                    const double w = kWeights[dx + 2];
                    sumW += w;
                    for (int k = 0; k < kNLegs; ++k)
                        sums[k] += w * legs[k].wRaw[xx];
                }

                // Decision blend by local registered amplitude: raw gates at
                // low amplitude (cheap, exact), smoothed gates where carrier
                // is loud (a per-pixel gate step on loud carrier is itself a
                // rendered artifact).
                const int xm = std::max(0, xi - 1);
                const int xp = std::min(width - 1, xi + 1);
                double amp = 0.0;
                auto includeRegisteredAmp = [&](const float *fit,
                                                const float *elig,
                                                int x) {
                    if (fit && elig && elig[x] > 0.0f)
                        amp = std::max(amp,
                                       std::fabs(static_cast<double>(fit[x])));
                };
                includeRegisteredAmp(fitRow, eligRow, xi);
                includeRegisteredAmp(fitRow, eligRow, xm);
                includeRegisteredAmp(fitRow, eligRow, xp);
                for (int k = 0; k < kNLegs; ++k) {
                    if (!legs[k].present)
                        continue;
                    includeRegisteredAmp(legs[k].fit, legs[k].elig, xi);
                    includeRegisteredAmp(legs[k].fit, legs[k].elig, xm);
                    includeRegisteredAmp(legs[k].fit, legs[k].elig, xp);
                }
                const double blend =
                    smoothStep01((amp * invIreScale - 8.0) / 10.0);

                // The leg's graded participation multiplies its weight, so
                // the horizontal smoother cannot resurrect a proven-illegal
                // partner sample (x0 stays 0), and partial doubt fades the
                // leg instead of flipping it at the verdict threshold.
                for (int k = 0; k < kNLegs; ++k) {
                    legW[k] = legs[k].present
                        ? (legs[k].wRaw[xi] * (1.0 - blend) +
                              (sums[k] / sumW) * blend) *
                          static_cast<double>(legs[k].elig[xi])
                        : 0.0;
                }
            }

            double wSum = 0.0;
            for (int k = 0; k < kNLegs; ++k)
                wSum += legW[k];

            if (wSum > 1e-9) {
                double neighborFit = 0.0;
                for (int k = 0; k < kNLegs; ++k) {
                    if (legW[k] > 0.0)
                        neighborFit += legW[k] *
                            static_cast<double>(legs[k].fit[xi]);
                }
                neighborFit /= wSum;

                // Strength-scaled cancellation, not a hard switch.  For
                // genuinely inverting chroma 0.5*(C - (-C)) = C at any
                // strength, so this is amplitude-neutral where the comb is
                // right; where the neighbor does NOT invert (slanted chroma,
                // pedestal) a partial gate now yields partial retention
                // instead of full cancellation of real signal.
                const double cancelled =
                    0.5 * (static_cast<double>(fitRow[xi]) - neighborFit);
                const double strength = std::min(1.0, wSum);

                // Ownership rule: the un-cancelled remainder is carrier this
                // stage would confiscate from Y.  It may only be confiscated
                // where carrier is owned.  The license is nonzero ONLY where
                // an observed on-schedule alternation proves the operand
                // behaves as carrier; it is zero through the entire ambiguous
                // middle (so genuine/slanted chroma is untouched).  Withdraw
                // only the standdown remainder, never the active
                // cancellation.
                //
                // Schedule participation joins the LICENCE on that remainder,
                // for the same reason and on the same term.  It must NOT
                // scale the cancelled term: 0.5*(fit - neighbour) is
                // self-proving by construction -- it yields C where the
                // partner inverts (real chroma, correctly kept as carrier)
                // and 0 where the partner matches (image-locked alien,
                // correctly left in Y).  Scaling that by doubt defeats the
                // separation the comb exists to perform, and on
                // alien-dominated mixed content (the cube face) it withholds
                // the very chroma the cancellation had already isolated.
                combRow[xi] = static_cast<float>(
                    static_cast<double>(fitRow[xi]) * (1.0 - strength) *
                        ownedFallback * centerParticipation +
                    cancelled * strength);

                if (pass2DumpPrefix)
                    p2rec(line, 8, xi, neighborFit);
            } else {
                // Reach fully closed: the whole sample is un-cancelled carrier.
                // Confiscate it only where carrier is owned.
                combRow[xi] = static_cast<float>(
                    centerParticipation *
                    static_cast<double>(fitRow[xi]) * ownedFallback);
            }

            if (pass2DumpPrefix) {
                double wSumRaw = 0.0;
                for (int k = 0; k < kNLegs; ++k)
                    wSumRaw += legs[k].wRaw[xi];
                p2rec(line, 7, xi, wSumRaw);
                p2rec(line, 9, xi, std::min(1.0, wSum));
                p2rec(line, 10, xi, ownedFallback);
                p2rec(line, 11, xi, p2corrCode);
                p2rec(line, 12, xi, static_cast<double>(fitRow[xi]));
                p2rec(line, 13, xi, static_cast<double>(combRow[xi]));
            }
        }
    }

    // NOTE: the encoder bandwidth law is imposed ONCE, on the native
    // carrierFit at the Pass-1 model boundary.  It is deliberately NOT
    // re-applied to combedCarrier here.  A forward FIR is not a projection, so
    // a second application would attenuate legal chroma near the top of the
    // encoder's own passband (measured 4.5 dB of extra loss at 1.3 MHz for a
    // third pass) and leave it in Y.
    //
    // That does leave a real, separate defect on the promoted product: the
    // emit scales the fit by per-pixel factors (strength x ownedFallback x
    // participation), and a bandlimited carrier times a fast-varying gain is
    // amplitude modulation, which manufactures out-of-band sidebands.  The
    // cure for that is to stop scaling a carrier-band signal by a
    // carrier-rate control, not to filter the damage afterwards.

    if (pass2DumpPrefix) {
        static std::atomic<int> p2DumpCounter{0};
        const int n = p2DumpCounter.fetch_add(1);
        char path[512];
        std::snprintf(path, sizeof(path), "%s_%03d.bin", pass2DumpPrefix, n);
        if (FILE *fp = std::fopen(path, "wb")) {
            const qint32 hdr[6] = {0x50325644, firstLine, lastLine,
                                   width, kP2D, left};
            std::fwrite(hdr, sizeof(qint32), 6, fp);
            std::fwrite(p2dump.data(), sizeof(float), p2dump.size(), fp);
            std::fclose(fp);
            std::fprintf(stderr, "PASS2DUMP wrote %s heldSeq=%d/%d\n",
                         path, heldSeq1, heldSeq2);
        }
    }

    // ---------------------------------------------------------------
    // Final publication: retracted Y derives from the promoted carrier model,
    // not from the workprint fit. flatFloor has already served Pass 2 and has
    // no downstream consumer.
    // ---------------------------------------------------------------
    // Which carrier model the published view withdraws:
    //
    // PEDESTAL LAW (user, 2026-07-26): every election candidate rides the
    // SAME BASIS — raw minus a FULL, COMMITTED carrier subtraction.  The
    // candidates differ by WHICH carrier model they subtract, never by HOW
    // MUCH of it.  The previous default (raw - w·carrierFit) scaled the
    // subtraction by the corroboration, so the candidate's baseline floated
    // by (1-w)·fit relative to the roster — at w→0 it stood a full legal-
    // carrier lobe above every other candidate and won bright pixels on
    // that pedestal alone (the >100 IRE speck population, ~40% of it
    // schedule-LEGAL kept energy).  Feasibility cannot catch it: |raw - Y|
    // ~= 0 at passthrough is maximally feasible.  Scaling the withdrawal
    // was also a second confidence hedge on a promoted model: the fit is
    // already schedule-aware through the graded participation weights, and
    // since P14 it is hull-bounded, so full subtraction cannot over-
    // withdraw beyond what the legal Y floors permit.  Corroboration stays
    // EVIDENCE (admission/scoring); it is no longer a subtraction gain.
    //
    //   native (default) — raw - carrierFit: the full hull-bounded lawful
    //       fit.  In-band luma the schedule refused never entered the fit
    //       (graded participation), so the grid detail survives HERE, in
    //       the model, not via a downstream haircut.
    //   corr (LDCD_RETRACTED_SOURCE=corr) — raw - w·carrierFit: the retired
    //       floating-pedestal product, kept as an A/B escape only.
    //   promoted (LDCD_RETRACTED_SOURCE=promoted) — raw - combedCarrier:
    //       Pass 2's per-pixel confiscation policy (measured: the most
    //       lawful share of the old products, 71%, but rendered as a
    //       per-pixel patchwork and nearly inert on the cube face).
    //
    // One default product; the env selections are A/B escapes.  Every
    // downstream consumer — centre and vertical neighbours alike — sees the
    // same selection, so the election is fed a consistent product rather
    // than a mixture.
    //   anchor (DEFAULT, promoted 2026-07-28 on render judgement: A/C
    //       great, D good, B soft -- the remaining softness is the
    //       no-coverage letters riding the fit, the open cross-frame
    //       transfer work) -- the certified-carrier ladder below:
    //       covered sample -> (def-spare)/2 as itself; comp sample ->
    //       certified-luma vertical comb where its out-of-band residual
    //       passes; fit only where no certified product exists. On
    //       material without dG coverage every sample falls through to
    //       the fit, i.e. exactly the old default.
    //   native (LDCD_RETRACTED_SOURCE=native) -- raw - carrierFit
    //       everywhere: the pre-anchor default, kept as the A/B escape.
    const int retractedSource = ldcdRetractedSourceMode();

    // Certified-luma vertical comb for the COMP lines in anchor mode (user,
    // 2026-07-28: the reform used the certified CARRIER on covered lines and
    // left the comps with the fit -- "I wonder if the reform of retracted
    // didn't make full use of the 'real luma' we have available from the
    // certified fields"). The conservation pair supplies both halves: on
    // covered lines Lhat = raw - exact is certified luma, CARRIER-FREE, so
    // a comp line combed against its two certified brackets has no phase
    // relation to honour at all:
    //     R = raw_comp - (Lhat_up + Lhat_dn)/2 = C_comp + (L_comp - L_avg)
    // All shared luma -- the strong lateral structure whose partial-window
    // leak drives every estimator's error -- cancels EXACTLY; what remains
    // is the carrier plus only the comp line's own vertical-detail
    // deviation. The merge's own carrier-band filter (unity at fSC and
    // +-1.3 MHz, zero at DC/2fSC) then takes the carrier, and the
    // out-of-band residual |R - BP(R)| is the same correlate the twin
    // merge uses: where it is large the bracket assumption failed (real
    // vertical detail / chroma boundary) and the fit stands for that
    // sample. This narrows the covered/uncovered quality gap from the comp
    // side -- the convergent direction -- rather than special-casing
    // covered lines further.
    std::vector<double> certComp;
    if (retractedSource == 3)
        certComp.assign(width, std::numeric_limits<double>::quiet_NaN());

    // Anticipated rung (uncovered frames, 2026-08-01, user-approved
    // consumption level: ESTIMATOR with confidence alpha, NEVER
    // certification -- no cede, no HQ-leg status; facts are facts,
    // predictions are estimators).  The certified carrier's curve is
    // causal (the anticipated phase lands almost exactly at the next
    // cover), so the PREVIOUS cover's certified separation serves the
    // uncovered frame at load time: its certified LUMA (raw - exact,
    // direct where its parity covers the line, bracket mean otherwise)
    // is the tween reference -- luma tweens where carrier cannot -- and
    //     R = raw_this - Lhat_prev = C_this + (L_this - L_prev)
    // keeps the carrier NATIVE to this frame (phase correct by
    // construction; nothing is transferred as carrier).  The merge FIR
    // takes the carrier; the out-of-band residual is the motion/
    // staleness CORRELATE feeding alpha.  Unlike the covered comp rung
    // there is NO parity floor on alpha: that floor is justified by a
    // same-frame reference measured to beat the fit even where the
    // residual fires; a reference one film frame away under motion is
    // worse than the fit, so alpha must be able to reach zero and the
    // rung fails toward the (tone-corrected) fit.  Escape: the rung
    // rides LDCD_ANCHOR_1D / LDCD_CERT_1D family escapes -- with
    // LDCD_CERT_1D=0 no exact data exists and the rung never forms.
    // Chain maintenance: a covered frame publishes its own certified luma
    // (raw - exact on def lines, NaN elsewhere); an uncovered frame
    // inherits the previous frame's plane with age+1, capped at the same
    // horizon the tone trusts (dtF <= 8), so every uncovered frame within
    // the horizon of a cover holds the reference -- not only the
    // immediate successor.
    antRefAge = -1;
    if (retractedSource == 3 && certifiedOneDLevel() >= 1) {
        if (frameHasExactCoverage()) {
            antRefLuma_flat.assign(
                static_cast<size_t>(frameHeight) * demodWidth,
                std::numeric_limits<float>::quiet_NaN());
            for (int line = firstLine; line < lastLine; ++line) {
                if (!certifiedDefLine(line)) continue;
                const float *ex = exactCarrierRow(line);
                if (!ex) continue;
                const quint16 *raw = rawbuffer.data()
                    + static_cast<size_t>(line) * videoParameters.fieldWidth;
                float *dst = antRefLuma_flat.data()
                    + static_cast<size_t>(line) * demodWidth;
                for (int xi = 0; xi < width; ++xi) {
                    const int h = left + xi;
                    if (std::isfinite(ex[h]))
                        dst[xi] = static_cast<float>(
                            (double)raw[h] - (double)ex[h]);
                }
            }
            antRefAge = 0;
        } else if (prevF && prevF->antRefAge >= 0 &&
                   prevF->antRefAge < 8 &&
                   prevF->antRefLuma_flat.size() ==
                       static_cast<size_t>(frameHeight) * demodWidth) {
            antRefLuma_flat = prevF->antRefLuma_flat;
            antRefAge = prevF->antRefAge + 1;
        }
    }

    // Held-out grading (measurement only, LDCD_PROBE_ANTGRADE=1): on a
    // COVERED frame, build the anticipated construction as if this frame
    // were uncovered -- from the PREDECESSOR's chained reference plane --
    // and grade per-sample against this frame's OWN exact channel.  The
    // grade lattice is the frame's def lines (where exact exists), classed
    // by what the reference plane offers THERE: DIRECT (tween luma known
    // same-line) vs BRACKET (vertical mean of the plane's adjacent lines).
    // Three estimators per class: FULL (raw - Lhat, full band, no FIR),
    // FIR (the shipped rung's bp), and FIT (the baseline competitor).
    // Reported: mean|err| / rms in IRE, and the signed correlation with
    // exact (phase agreement -- the quantity the magnitude means were
    // blind to).  Optional box via LDCD_ANTGRADE_L0/L1/C0/C1 (frame line /
    // active column).
    {
        static const bool antGradeOn =
            std::getenv("LDCD_PROBE_ANTGRADE") != nullptr;
        if (antGradeOn && retractedSource == 3 && frameHasExactCoverage() &&
            prevF && prevF->antRefAge >= 0 &&
            prevF->antRefLuma_flat.size() ==
                static_cast<size_t>(frameHeight) * demodWidth) {
            static const auto agEnv = [](const char *n, int d) {
                const char *s = std::getenv(n); return s ? std::atoi(s) : d;
            };
            static const int agL0 = agEnv("LDCD_ANTGRADE_L0", -1);
            static const int agL1 = agEnv("LDCD_ANTGRADE_L1", -1);
            static const int agC0 = agEnv("LDCD_ANTGRADE_C0", -1);
            static const int agC1 = agEnv("LDCD_ANTGRADE_C1", -1);
            const bool haveBox = agL0 >= 0 && agC0 >= 0;

            struct AntAcc {
                long n = 0;
                double ae = 0, se = 0, dxe = 0, e2 = 0, x2 = 0;
                void add(double est, double ex, double inv) {
                    const double e = (est - ex) * inv;
                    ++n; ae += std::fabs(e); se += e * e;
                    dxe += est * ex; e2 += est * est; x2 += ex * ex;
                }
                void print(const char *tag) const {
                    const double corr = (e2 > 0 && x2 > 0)
                        ? dxe / std::sqrt(e2 * x2) : 0.0;
                    std::fprintf(stderr,
                        "  %-8s n=%-7ld mean|e|=%6.2f rms=%6.2f corr=%+.3f\n",
                        tag, n, n ? ae / n : 0.0,
                        n ? std::sqrt(se / n) : 0.0, corr);
                }
            };
            AntAcc dFull, dFir, dFit, bFull, bFir, bFit;
            AntAcc wFull, wFir, wFit;

            std::vector<double> Rg(width,
                std::numeric_limits<double>::quiet_NaN());
            std::vector<std::uint8_t> clsRow(width);
            const float *refBase = prevF->antRefLuma_flat.data();

            for (int line = firstLine; line < lastLine; ++line) {
                const float *ex = exactCarrierRow(line);
                if (!ex) continue;
                const quint16 *raw = rawbuffer.data()
                    + static_cast<size_t>(line) * videoParameters.fieldWidth;
                const float *fit = carrierFit_flat.data()
                    + static_cast<size_t>(line) * demodWidth;
                // Referee repair (2026-08-02): factFit stamped the fit with
                // the exact fact at exactly the graded samples; the FIT
                // baseline must read the pre-stamp estimator.
                const float *preFitAg = probePreFactFit_flat.empty()
                    ? nullptr
                    : probePreFactFit_flat.data() +
                          static_cast<size_t>(line) * demodWidth;
                const float *r0 = refBase
                    + static_cast<size_t>(line) * demodWidth;
                const float *rU = (line - 1 >= firstLine)
                    ? refBase + static_cast<size_t>(line - 1) * demodWidth
                    : nullptr;
                const float *rD = (line + 1 < lastLine)
                    ? refBase + static_cast<size_t>(line + 1) * demodWidth
                    : nullptr;

                std::fill(Rg.begin(), Rg.end(),
                          std::numeric_limits<double>::quiet_NaN());
                for (int xi = 0; xi < width; ++xi) {
                    clsRow[xi] = 0;
                    double lhat =
                        std::numeric_limits<double>::quiet_NaN();
                    if (std::isfinite(r0[xi])) {
                        lhat = (double)r0[xi];
                        clsRow[xi] = 1;             // DIRECT
                    } else if (rU && rD && std::isfinite(rU[xi]) &&
                               std::isfinite(rD[xi])) {
                        lhat = 0.5 * ((double)rU[xi] + (double)rD[xi]);
                        clsRow[xi] = 2;             // BRACKET
                    }
                    if (std::isfinite(lhat))
                        Rg[xi] = (double)raw[left + xi] - lhat;
                }

                for (int xi = 0; xi < width; ++xi) {
                    const int h = left + xi;
                    if (!std::isfinite(ex[h]) || clsRow[xi] == 0)
                        continue;
                    bool ok = true;
                    double taps[5];
                    static const int off[5] = { 0, -2, 2, -4, 4 };
                    for (int k = 0; k < 5 && ok; ++k) {
                        const int j = std::clamp(xi + off[k], 0, width - 1);
                        taps[k] = Rg[j];
                        if (!std::isfinite(taps[k])) ok = false;
                    }
                    const double bp = ok
                        ? 0.676462 * taps[0] +
                          -0.250000 * (taps[1] + taps[2]) +
                          -0.088231 * (taps[3] + taps[4])
                        : std::numeric_limits<double>::quiet_NaN();
                    const double exd = (double)ex[h];
                    const double fitd =
                        (preFitAg && std::isfinite(preFitAg[xi]))
                            ? (double)preFitAg[xi]
                            : (double)fit[xi];
                    const bool inBox = haveBox &&
                        line >= agL0 && line <= agL1 &&
                        xi >= agC0 && xi <= agC1;
                    if (clsRow[xi] == 1) {
                        dFull.add(Rg[xi], exd, invIreScale);
                        if (std::isfinite(bp))
                            dFir.add(bp, exd, invIreScale);
                        dFit.add(fitd, exd, invIreScale);
                    } else {
                        bFull.add(Rg[xi], exd, invIreScale);
                        if (std::isfinite(bp))
                            bFir.add(bp, exd, invIreScale);
                        bFit.add(fitd, exd, invIreScale);
                    }
                    if (inBox) {
                        wFull.add(Rg[xi], exd, invIreScale);
                        if (std::isfinite(bp))
                            wFir.add(bp, exd, invIreScale);
                        wFit.add(fitd, exd, invIreScale);
                    }
                }
            }
            // One ATOMIC line per frame (threaded runs interleave stderr
            // between fprintf calls, never mid-line), tagged with the frame
            // identity and this frame's own def parity so the two anchor
            // legs (A->C vs C->A) and any longer cycle can be separated
            // offline -- the recalibration axis (user, 2026-08-02): the
            // original "lands almost exactly" verdict was one aggregate,
            // never stratified by leg or cycle.
            {
                int ownDefPar = -1;
                for (int l = firstLine; l < lastLine; ++l)
                    if (certifiedDefLine(l)) { ownDefPar = l & 1; break; }
                const auto m = [](const AntAcc &a) {
                    return a.n ? a.ae / a.n : 0.0;
                };
                const auto c = [](const AntAcc &a) {
                    return (a.e2 > 0 && a.x2 > 0)
                        ? a.dxe / std::sqrt(a.e2 * a.x2) : 0.0;
                };
                std::fprintf(stderr,
                    "[ANTGRADE1] seq=%d age=%d defPar=%d "
                    "dN=%ld dFull=%.3f/%+.3f dFir=%.3f/%+.3f "
                    "dFit=%.3f/%+.3f "
                    "bN=%ld bFull=%.3f/%+.3f bFir=%.3f/%+.3f "
                    "bFit=%.3f/%+.3f\n",
                    (int)heldSeq1, prevF->antRefAge + 1, ownDefPar,
                    dFull.n, m(dFull), c(dFull), m(dFir), c(dFir),
                    m(dFit), c(dFit),
                    bFull.n, m(bFull), c(bFull), m(bFir), c(bFir),
                    m(bFit), c(bFit));
            }
            if (haveBox) {
                std::fprintf(stderr, " BOX (L%d-%d C%d-%d):\n",
                             agL0, agL1, agC0, agC1);
                wFull.print("full"); wFir.print("fir"); wFit.print("fit");
            }
        }
    }

    // MOTION-WITNESSED CONSTRUCTION (default; the full-commit data run is
    // kept as LDCD_ANT_COMMIT=1 for A/B).  The full-commit experiment
    // rendered GHOSTS on motion (user verdict 2026-08-01, beach: "a
    // reversion on the 'averaging is not tweening' front"): the chain is
    // one-sided by load-time causality, so under motion Lhat carries the
    // previous cover's edges at their displaced positions, the FIR passes
    // that ghost geometry as carrier, and the cede makes it roster-
    // unanimous -- the election cannot reject it.  The causal-anticipation
    // license was for the PHASE CURVE (global, rides the cycle); per-pixel
    // luma content moves, and a one-sided reference under motion is
    // averaging, not tweening.  So the ceded center's VALUE is built with
    // the out-of-band residual as its motion CORRELATE -- alpha =
    // exp(-|R-bp|/tau) between tween carrier and fit, per sample, no
    // floor -- the same shape as the shipped covered comp rung.  Weight is
    // candidate confidence, never inter-candidate distance; unanimity
    // stays uniform (no license flapping); static content keeps the
    // anticipation, motion degrades to current-frame data.
    static const bool antCommit = []{
        const char *e = std::getenv("LDCD_ANT_COMMIT");
        return e && std::atoi(e) == 1;
    }();
    std::vector<double> antComp;
    // A/B escape (LDCD_ANT_RUNG=0, user-authorised 2026-08-04). Stands the
    // anticipated rung down so the ladder falls through to certComp/fit.
    // Default ON: inert unless set. Under test as the remaining source of the
    // uncovered letters' checkerboard -- measured by the RLAD rung census,
    // this rung stands on 100% of samples in the affected window and departs
    // from the fit by 1.8-3.0 IRE, a departure that reads as STANDING (image-
    // locked luma) rather than as carrier. Every other stage in this campaign
    // carries an escape; this one did not, so it could not be tested.
    static const bool antRungOff = []{
        const char *e = std::getenv("LDCD_ANT_RUNG");
        return e && std::atoi(e) == 0;
    }();
    const bool antRungOn = !antRungOff && retractedSource == 3 &&
        !frameHasExactCoverage() && antRefAge >= 1;
    if (antRungOn)
        antComp.assign(width, std::numeric_limits<double>::quiet_NaN());

    // Region-local ladder dump (LDCD_DUMP_RLAD=1, box via _R0/_R1/_C0/_C1
    // in output-row / active-column coords; run -t 1). One line per sample:
    // which rung stood, what each estimator said, and the certified
    // brackets with their grammar signs. Diagnostic only -- the user's
    // direction (2026-07-29): the certified fields' exact shade is the
    // truth reference for diagnosing the comp-line estimators in thorny
    // areas where carrier and HF luma resist separation.
    static const bool dumpRlad = []{
        const char *e = std::getenv("LDCD_DUMP_RLAD");
        return e && std::atoi(e) != 0;
    }();
    static const auto rladEnvInt = [](const char *name, int dflt) {
        const char *e = std::getenv(name);
        return e ? std::atoi(e) : dflt;
    };
    static const int rladR0 = rladEnvInt("LDCD_DUMP_RLAD_R0", 0);
    static const int rladR1 = rladEnvInt("LDCD_DUMP_RLAD_R1", 1 << 20);
    static const int rladC0 = rladEnvInt("LDCD_DUMP_RLAD_C0", 0);
    static const int rladC1 = rladEnvInt("LDCD_DUMP_RLAD_C1", 1 << 20);
    static long g_rladFrame = 0;
    const long rladFrame = dumpRlad ? g_rladFrame++ : 0;
    if (dumpRlad)
        std::fprintf(stderr, "RLADH f=%ld firstLine=%d left=%d ire=%.5f\n",
                     rladFrame, firstLine, left, irescale);
    std::vector<double> certBpDbg;  // BP(R) BEFORE the witness, diag only
    if (dumpRlad)
        certBpDbg.assign(width, std::numeric_limits<double>::quiet_NaN());

    // Escapes for A/B only, both default ON:
    //   LDCD_OOB_ALPHA=0  -- restore the binary OOB cut
    //   LDCD_PHASE_SNAP=0 -- disable the working-space snap
    static const bool oobAlpha = []{
        const char *e = std::getenv("LDCD_OOB_ALPHA");
        return !(e && std::atoi(e) == 0);
    }();
    const bool phaseSnap = ldcdPhaseSnapOn();
    // The phase reference was validated at >=2 IRE (vertical coherence
    // 0.907); below that its phase is noise and snapping toward it costs
    // Y (measured: fading from 0.25 IRE gave back the alpha reform's D2
    // win).
    constexpr double kSnapAmpTauIRE = 3.0;
    constexpr double kSnapAmpMinIRE = 1.0;

    for (int line = firstLine; line < lastLine; ++line) {
        const quint16 *rawLine =
            rawbuffer.data() + static_cast<size_t>(line) * videoParameters.fieldWidth;
        const float *fitRowPub = carrierFit_flat.data()
                                 + static_cast<size_t>(line) * demodWidth;
        const float *combRowPub = combedCarrier_flat.data()
                                  + static_cast<size_t>(line) * demodWidth;
        const float *wRowPub = carrierCorroboration_flat.data()
                               + static_cast<size_t>(line) * demodWidth;
        float *retractedRow = carrierRetracted_flat.data()
                              + static_cast<size_t>(line) * demodWidth;

        const float *exRowPub = exactCarrierRow(line);
        const float *exU = (line - 1 >= firstLine)
            ? exactCarrierRow(line - 1) : nullptr;
        const float *exD = (line + 1 < lastLine)
            ? exactCarrierRow(line + 1) : nullptr;

        if (retractedSource == 3) {
            std::fill(certComp.begin(), certComp.end(),
                      std::numeric_limits<double>::quiet_NaN());
            if (dumpRlad)
                std::fill(certBpDbg.begin(), certBpDbg.end(),
                          std::numeric_limits<double>::quiet_NaN());
            if (exU && exD) {
                const quint16 *rawU = rawbuffer.data()
                    + static_cast<size_t>(line - 1) * videoParameters.fieldWidth;
                const quint16 *rawD = rawbuffer.data()
                    + static_cast<size_t>(line + 1) * videoParameters.fieldWidth;
                // R = comp composite minus the certified-luma bracket mean.
                std::vector<double> R(width,
                    std::numeric_limits<double>::quiet_NaN());
                for (int xi = 0; xi < width; ++xi) {
                    const float eu = exU[left + xi];
                    const float ed = exD[left + xi];
                    if (!std::isfinite(eu) || !std::isfinite(ed)) continue;
                    const double lu = (double)rawU[left + xi] - (double)eu;
                    const double ld = (double)rawD[left + xi] - (double)ed;
                    R[xi] = (double)rawLine[left + xi] - 0.5 * (lu + ld);
                }
                // Merge-identical carrier-band FIR. The out-of-band residual
                // is now EVIDENCE, not a router (user, 2026-07-29: "if we're
                // doing a blend, that should be a penalty that goes to
                // alpha, not a binary cutoff"). The old cut interleaved comb
                // and fit per sample along the strut -- two ~5 IRE-
                // disagreeing renders alternating down a vertical feature
                // WAS the pillar wobble; and on the very samples where the
                // residual fired, the comb still beat the fit (4.72 vs 5.30
                // IRE vs certified truth), so the comb's weight floors at
                // parity: the residual can say "trust me less", never
                // "trust the fit more".
                constexpr double kT0 = 0.676462;
                constexpr double kT2 = -0.250000;
                constexpr double kT4 = -0.088231;
                constexpr double kOobTauIRE = 2.0;   // e-fold of the alpha
                constexpr double kOobCutIRE = 4.0;   // escape-mode old cut
                std::vector<double> bpV(width,
                    std::numeric_limits<double>::quiet_NaN());
                std::vector<double> resV(width,
                    std::numeric_limits<double>::quiet_NaN());
                for (int xi = 0; xi < width; ++xi) {
                    bool ok = true;
                    double taps[5];
                    static const int off[5] = { 0, -2, 2, -4, 4 };
                    for (int k = 0; k < 5 && ok; ++k) {
                        const int j = std::clamp(xi + off[k], 0, width - 1);
                        taps[k] = R[j];
                        if (!std::isfinite(taps[k])) ok = false;
                    }
                    if (!ok) continue;
                    const double bp = kT0 * taps[0] +
                                      kT2 * (taps[1] + taps[2]) +
                                      kT4 * (taps[3] + taps[4]);
                    if (dumpRlad) certBpDbg[xi] = bp;  // pre-alpha, diag only
                    bpV[xi] = bp;
                    resV[xi] = std::fabs(R[xi] - bp);
                }

                // Confidence-alpha merge of the two comp-line estimators.
                // The fit partner's share is CONDITIONED by the regional
                // fact audit (Pass 1.7): where the solve graded poorly
                // against the exact truth in this neighbourhood, the fit
                // is a proven-poor witness and its share fades toward the
                // certified-bracket comb. Bilinear over region centers --
                // a weight must never step at a region seam. No verdict
                // (NaN / no audit) leaves conduct unchanged.
                constexpr double kAuditTauIRE = 4.0;
                auto factTrustAt = [&](int xi) -> double {
                    if (fitFactAuditNx <= 0) return 1.0;
                    const int nx = fitFactAuditNx, ny = fitFactAuditNy;
                    const double rx = std::clamp(
                        (xi - 64.0) / 128.0, 0.0, (double)(nx - 1));
                    const double ry = std::clamp(
                        (line - 16.0) / 32.0, 0.0, (double)(ny - 1));
                    const int x0 = std::min((int)rx, nx - 1);
                    const int y0 = std::min((int)ry, ny - 1);
                    const int x1 = std::min(x0 + 1, nx - 1);
                    const int y1 = std::min(y0 + 1, ny - 1);
                    const double fx = rx - x0, fy = ry - y0;
                    auto v = [&](int yy, int xx) -> double {
                        const float a =
                            fitFactAuditIRE[(size_t)yy * nx + xx];
                        return std::isfinite(a) ? (double)a
                            : std::numeric_limits<double>::quiet_NaN();
                    };
                    const double v00 = v(y0, x0), v01 = v(y0, x1);
                    const double v10 = v(y1, x0), v11 = v(y1, x1);
                    if (!(std::isfinite(v00) && std::isfinite(v01) &&
                          std::isfinite(v10) && std::isfinite(v11)))
                        return 1.0;   // no full verdict: unchanged
                    const double a =
                        (v00 * (1 - fx) + v01 * fx) * (1 - fy) +
                        (v10 * (1 - fx) + v11 * fx) * fy;
                    return std::exp(-a / kAuditTauIRE);
                };
                std::vector<double> est(width);
                for (int xi = 0; xi < width; ++xi) {
                    const double fit = static_cast<double>(fitRowPub[xi]);
                    if (!std::isfinite(bpV[xi])) { est[xi] = fit; continue; }
                    double wc;
                    if (oobAlpha)
                        wc = 0.5 + 0.5 * std::exp(-resV[xi] /
                                                  (kOobTauIRE * irescale));
                    else
                        wc = (resV[xi] <= kOobCutIRE * irescale) ? 1.0 : 0.0;
                    wc = 1.0 - (1.0 - wc) * factTrustAt(xi);
                    est[xi] = wc * bpV[xi] + (1.0 - wc) * fit;
                }

                // Phase alignment in the 4fsc working space (user: "we
                // generally rotate all candidates in an election into the
                // same working space... the candidates in the Y election
                // need to have their phases aligned before it gets baked
                // in"). MEASURED basis (LDCD_DUMP_RLAD, 2026-07-29): the
                // true carrier holds phase down the strut (vertical
                // coherence 0.907 across covered lines), while the fit's
                // phase error is spatially incoherent (rms 52 deg at real
                // carrier, line-to-line ANTI-correlated) -- so the
                // certified brackets are the phase reference and the local
                // estimator keeps only its amplitude:
                //     carrier = |est| * bAlign / |bAlign|
                // per short window, where bAlign is the grammar-sign-
                // aligned bracket mean. The snap fades in with reference
                // amplitude (phase of a near-zero carrier is meaningless,
                // a clamp on an impossible) and is skipped where the
                // bracket window is incomplete.
                if (phaseSnap) {
                    std::vector<double> bAlign(width);
                    ldcdBuildCertBracketAligned(
                        carrierGrammarLine(line),
                        carrierGrammarLine(line - 1),
                        carrierGrammarLine(line + 1),
                        exU, exD, left, width, bAlign);
                    ldcdApplyPhaseSnap(est, bAlign, certComp,
                                       width, irescale,
                                       kSnapAmpMinIRE, kSnapAmpTauIRE,
                                       false);
                } else {
                    for (int xi = 0; xi < width; ++xi)
                        certComp[xi] = est[xi];
                }
            }

            // Anticipated rung: see the declaration comment above.  Built
            // per line from the chained cover's certified luma; carrier
            // stays native to this frame.
            if (antRungOn) {
                std::fill(antComp.begin(), antComp.end(),
                          std::numeric_limits<double>::quiet_NaN());
                const float *ref0 = antRefLuma_flat.data()
                    + static_cast<size_t>(line) * demodWidth;
                const float *refU = (line - 1 >= firstLine)
                    ? antRefLuma_flat.data()
                      + static_cast<size_t>(line - 1) * demodWidth
                    : nullptr;
                const float *refD = (line + 1 < lastLine)
                    ? antRefLuma_flat.data()
                      + static_cast<size_t>(line + 1) * demodWidth
                    : nullptr;

                std::vector<double> R(width,
                    std::numeric_limits<double>::quiet_NaN());
                for (int xi = 0; xi < width; ++xi) {
                    double lhat = std::numeric_limits<double>::quiet_NaN();
                    if (std::isfinite(ref0[xi])) {
                        lhat = (double)ref0[xi];
                    } else if (refU && refD &&
                               std::isfinite(refU[xi]) &&
                               std::isfinite(refD[xi])) {
                        lhat = 0.5 * ((double)refU[xi] + (double)refD[xi]);
                    }
                    if (std::isfinite(lhat))
                        R[xi] = (double)rawLine[left + xi] - lhat;
                }

                constexpr double kT0a = 0.676462;
                constexpr double kT2a = -0.250000;
                constexpr double kT4a = -0.088231;
                constexpr double kAntTauIRE = 2.0;

                // MOTION WITNESS: the out-of-band residual |R − bp|, the
                // best-measured of three instruments (2026-08-01 battery,
                // title-bevel / beach-Calt):
                //   OOB residual            13.9/17.8 | 5.66  <- shipped
                //   retracted-LF as GATE    20.1/21.9 | 5.58  (sharp LP;
                //     boxcar was 22.5/26.5 -- bracket lines vertically
                //     interpolate the reference across diagonals, a real
                //     LF displacement the band-limited product never
                //     carries, so gating on it taxes working sites)
                //   retracted-LF as CORRECTOR (R' = R − LP(R − fit))
                //                           19.5/24.2 | 5.95  (the LF of
                //     R − fit includes the carrier-difference envelope,
                //     and subtracting it polluted both scenes)
                // NEGATIVE RESULT, do not re-propose the retracted-LF
                // witness in either form without new evidence. Known
                // remaining blind spot of the OOB residual: the in-band
                // component of a displaced edge's leak passes the FIR at
                // small residual and survives at partial alpha.
                for (int xi = 0; xi < width; ++xi) {
                    bool ok = true;
                    double taps[5];
                    static const int off[5] = { 0, -2, 2, -4, 4 };
                    for (int k = 0; k < 5 && ok; ++k) {
                        const int j = std::clamp(xi + off[k], 0, width - 1);
                        taps[k] = R[j];
                        if (!std::isfinite(taps[k])) ok = false;
                    }
                    if (!ok) continue;
                    const double bp = kT0a * taps[0] +
                                      kT2a * (taps[1] + taps[2]) +
                                      kT4a * (taps[3] + taps[4]);
                    if (antCommit) {
                        // A/B escape: the full-commit data run (ghosts on
                        // motion; kept for comparison renders only).
                        antComp[xi] = bp;
                        continue;
                    }
                    const double fit = static_cast<double>(fitRowPub[xi]);
                    // Post-correction motion correlate: the surviving error
                    // in R' is HF (the LF ghost body was corrected away by
                    // construction), so the OOB residual is back to being
                    // the right instrument for what remains.
                    const double res = std::fabs(R[xi] - bp);
                    const double wc =
                        std::exp(-res * invIreScale / kAntTauIRE);
                    antComp[xi] = wc * bp + (1.0 - wc) * fit;
                }
            }
        }

        for (int xi = 0; xi < width; ++xi) {
            double carrier;
            switch (retractedSource) {
            case 1:
                carrier = static_cast<double>(fitRowPub[xi]);
                break;
            case 2:
                carrier = static_cast<double>(combRowPub[xi]);
                break;
            case 3: {
                // CERTIFIED CARRIER, used as itself (user: "we already have
                // a carrier... the merged fields can cancel to luma or
                // carrier with a sign flip"). Covered sample: (def-spare)/2
                // IS the carrier, a conservation fact. Comp sample: the
                // certified-luma vertical comb above, where its out-of-band
                // witness passed. The fit stands only where neither
                // certified product exists.
                const float ex = exRowPub ? exRowPub[left + xi]
                                          : std::numeric_limits<float>::quiet_NaN();
                char rung;
                if (std::isfinite(ex)) {
                    carrier = static_cast<double>(ex);
                    rung = 'e';
                } else if (antRungOn && std::isfinite(antComp[xi])) {
                    // Anticipated rung: estimator with alpha, never
                    // certification (see the antComp declaration).
                    // Checked BEFORE certComp: on an uncovered frame the
                    // exact rows exist as all-NaN planes, so the covered
                    // comp construction above degenerates to a fit copy
                    // in certComp -- rung order alone keeps the covered
                    // frames' real cc rung senior (antRungOn is false
                    // there).
                    carrier = antComp[xi];
                    rung = 'a';
                } else if (std::isfinite(certComp[xi])) {
                    carrier = certComp[xi];
                    rung = 'c';
                } else {
                    carrier = static_cast<double>(fitRowPub[xi]);
                    rung = 'f';
                }
                if (dumpRlad) {
                    const int row = line - firstLine;
                    if (row >= rladR0 && row < rladR1 &&
                        xi >= rladC0 && xi < rladC1) {
                        const int h = left + xi;
                        int sU = 0, sD = 0;
                        const CombCarrierGrammar *gC = carrierGrammarLine(line);
                        const CombCarrierGrammar *gU = carrierGrammarLine(line - 1);
                        const CombCarrierGrammar *gD = carrierGrammarLine(line + 1);
                        if (gC && gU) {
                            const auto r = lddecode::carrierGrammarSignedPhaseRelation(gC, h, gU, h);
                            if (r == lddecode::CarrierPhaseRelation::Opposite) sU = -1;
                            else if (r == lddecode::CarrierPhaseRelation::Same) sU = 1;
                        }
                        if (gC && gD) {
                            const auto r = lddecode::carrierGrammarSignedPhaseRelation(gC, h, gD, h);
                            if (r == lddecode::CarrierPhaseRelation::Opposite) sD = -1;
                            else if (r == lddecode::CarrierPhaseRelation::Same) sD = 1;
                        }
                        const double inv = 1.0 / irescale;
                        std::fprintf(stderr,
                            "RLAD f=%ld r=%d x=%d g=%c car=%.3f fit=%.3f "
                            "cc=%.3f bp=%.3f ex=%.3f eu=%.3f ed=%.3f su=%d sd=%d "
                            "raw=%.2f\n",
                            rladFrame, row, xi, rung,
                            carrier * inv, (double)fitRowPub[xi] * inv,
                            certComp[xi] * inv, certBpDbg[xi] * inv,
                            (double)ex * inv,
                            exU ? (double)exU[h] * inv
                                : std::numeric_limits<double>::quiet_NaN(),
                            exD ? (double)exD[h] * inv
                                : std::numeric_limits<double>::quiet_NaN(),
                            sU, sD,
                            (double)rawLine[h] * inv);
                    }
                }
                break;
            }
            default:
                carrier = static_cast<double>(wRowPub[xi]) *
                          static_cast<double>(fitRowPub[xi]);
                break;
            }
            retractedRow[xi] = static_cast<float>(
                static_cast<double>(rawLine[left + xi]) - carrier);
        }
    }

    carrierRetractedValid = true;

    // Load time may publish the covered construction because its same-frame
    // provenance is complete.  The uncovered construction is still causal
    // here, so keep it private in carrierRetracted_flat; current-time
    // refineRetractedTemporal() will publish it only after both covered
    // neighbours have participated.
    anchoredCarrierProvenance = AnchoredCarrierProvenance::None;
    anchored1DSource_flat.clear();
    if (!analysisOnly && retractedSource == 3 && frameHasExactCoverage())
        publishAnchoredCarrierFromRetracted(
            AnchoredCarrierProvenance::FactBacked);

    if (dumpDead) {
        auto pct = [](long long a, long long b) {
            return b > 0 ? 100.0 * static_cast<double>(a)
                                 / static_cast<double>(b)
                         : 0.0;
        };
        std::fprintf(stderr,
            "[DEAD] active=%lld ineligible(illegal)=%lld(%.1f%%) "
            "dead(viewCount==0)=%lld(%.1f%%) [illegal=%lld fitStarved=%lld]\n",
            dzActive,
            dzIneligible, pct(dzIneligible, dzActive),
            dzDead, pct(dzDead, dzActive),
            dzDeadIllegal, dzDeadFitStarved);
        std::fprintf(stderr,
            "[DEAD] windows=%lld invalid=%lld(%.1f%%) [rank(illegal)=%lld det=%lld] "
            "amplification dead/illegal=%.2fx\n",
            dzWinTotal, dzWinInvalid, pct(dzWinInvalid, dzWinTotal),
            dzWinRank, dzWinDet,
            dzIneligible > 0 ? static_cast<double>(dzDead)
                                   / static_cast<double>(dzIneligible)
                             : 0.0);
    }
}
