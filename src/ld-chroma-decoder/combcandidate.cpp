/******************************************************************************
 * combcandidate.cpp
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
 *
 * Implements Comb::FrameBuffer::getCandidate(),
 * separated from comb.cpp so that candidate selection and 2D helpers
 * live in a single translation unit.
 ******************************************************************************/

#include "comb.h"
#include "combmath.h"

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <limits>

// -------------------------------------------------------------------------

static inline int reflectCombRel(int rel, int width)
{
    if (width <= 0) return 0;
    if (rel < 0) return -rel;
    if (rel >= width) return (width - 1) - (rel - (width - 1));
    return rel;
}

static inline double combKMetric(double cc, double symC, double cn, double symN)
{
    // |comp| carries a 2fsc (2-sample period) Nyquist ripple from the chroma
    // subcarrier envelope: |comp[rel]| peaks anti-phase to symMag (the average
    // of the rel±1 magnitudes). Comparing the raw rectified samples therefore
    // injects that 2px ripple into the weight, modulating the vertical comb
    // gain and producing the bevel zipper. Summing |comp| with its anti-phase
    // neighbour-average yields a 2px-flat envelope magnitude (peak+trough ≈ A
    // at every sample), so the disagreement/credit metric below is computed on
    // a ripple-free magnitude.
    const double magC = std::fabs(cc) + symC;
    const double magN = std::fabs(cn) + symN;
    double k = std::fabs(magC - magN);
    k -= (magC + magN) * 0.10;
    return std::max(0.0, k);
}

static inline double combSmoothGate(double xIRE, double softIRE, double hardIRE)
{
    if (xIRE <= softIRE) return 1.0;
    if (xIRE >= hardIRE) return 0.0;
    double t = (xIRE - softIRE) / std::max(1e-9, hardIRE - softIRE);
    return 1.0 - std::clamp(t, 0.0, 1.0);
}

static inline double combSimilarityFactor(double sim, double start, double full)
{
    if (sim <= start) return 0.0;
    if (sim >= full) return 1.0;
    double t = (sim - start) / std::max(1e-9, full - start);
    return std::clamp(t, 0.0, 1.0);
}

// Geometry-only evidence for a compact luma excursion.  This replaces the
// old fixed +/-2 comparison: an optically broadened star can still be sitting
// on its own shoulder at two samples, so that aperture measured little or
// nothing.  Each wider read must show the same background on both sides and
// stable outer flanks; a step or an extended texture therefore does not become
// an impulse merely because one radius happens to be symmetric.
//
// Carrier legality is intentionally absent here.  This service publishes the
// luma shape once as lumaImpulseRisk; each downstream consumer combines that
// named fact with its own carrier evidence and policy.
static inline double compactLumaExcursionEvidence(
    const double *luma, int x, int width, double invIreScale)
{
    if (!luma || width <= 0)
        return 0.0;

    constexpr std::array<int, 4> radii = { 2, 4, 6, 8 };
    constexpr int outerStep = 2;
    constexpr double flatSoftIRE = 2.0;
    constexpr double flatHardIRE = 4.0;
    constexpr double excursionSoftIRE = 5.0;
    constexpr double excursionHardIRE = 15.0;

    double best = 0.0;
    for (const int radius : radii) {
        if (x - radius - outerStep < 0 ||
            x + radius + outerStep >= width)
            continue;

        const double left = luma[x - radius];
        const double right = luma[x + radius];
        const double outerLeft = luma[x - radius - outerStep];
        const double outerRight = luma[x + radius + outerStep];
        const double surroundSpanIRE = std::max({
            std::fabs(left - right),
            std::fabs(left - outerLeft),
            std::fabs(right - outerRight)
        }) * invIreScale;
        const double flatSupport = combSmoothGate(
            surroundSpanIRE, flatSoftIRE, flatHardIRE);
        if (flatSupport <= 0.0)
            continue;

        const double excursionIRE = std::fabs(
            luma[x] - 0.5 * (left + right)) * invIreScale;
        const double excursion = std::clamp(
            (excursionIRE - excursionSoftIRE) /
                (excursionHardIRE - excursionSoftIRE),
            0.0, 1.0);
        best = std::max(best, excursion * flatSupport);
    }
    return best;
}

// -------------------------------------------------------------------------
// Comb-owned policy.  combreach publishes grammar legality and measured
// content relationships only; it never selects a leg or cedes output to 1D.

namespace {

constexpr std::uint8_t FieldACedeCenter       = 1u << 0;
constexpr std::uint8_t FieldACedeStrongAsym   = 1u << 1;

std::uint8_t fieldARegionCedeFlags(
    const CombContentReach::IntrafieldRegionReach &region)
{
    using R = CombContentReach::RegionRelation;
    if (!region.valid) return 0;

    const bool upDifferent = region.up == R::DifferentRegion;
    const bool downDifferent = region.down == R::DifferentRegion;
    const bool upSame = region.up == R::SameRegion;
    const bool downSame = region.down == R::SameRegion;
    const bool upAlien = region.up == R::AlienCancel;
    const bool downAlien = region.down == R::AlienCancel;
    const bool upContinues = upSame || upAlien;
    const bool downContinues = downSame || downAlien;

    std::uint8_t flags = 0;
    if (region.centerIsland) flags |= FieldACedeCenter;
    if (region.strongAsym)
        flags |= FieldACedeCenter | FieldACedeStrongAsym;
    if ((upDifferent && !downContinues) ||
        (downDifferent && !upContinues))
        flags |= FieldACedeCenter;

    // Preserve Field A's prior refusal thresholds, but keep their ownership
    // here: Field B is free to evolve a different one-leg policy.
    constexpr double strongSameMax = 3.25;
    constexpr double weakSameMax = 4.25;
    constexpr double rejectMin = 5.25;
    constexpr double rejectGap = 2.25;
    auto credibleOneSide = [&](double sameDiff, double rejectDiff) {
        return sameDiff <= strongSameMax ||
            (sameDiff <= weakSameMax && rejectDiff >= rejectMin &&
             rejectDiff - sameDiff >= rejectGap);
    };
    if (upDifferent && downSame &&
        !credibleOneSide(region.downDifferenceIRE, region.upDifferenceIRE))
        flags |= FieldACedeCenter;
    if (downDifferent && upSame &&
        !credibleOneSide(region.upDifferenceIRE, region.downDifferenceIRE))
        flags |= FieldACedeCenter;
    if ((upAlien && !downContinues) || (downAlien && !upContinues))
        flags |= FieldACedeCenter;
    return flags;
}

double fieldContourGate(const CombContentReach::MovingCoarseContour &mc,
                        bool up)
{
    if (!mc.valid) return 1.0;
    const double trust = up ? mc.upTrust : mc.downTrust;
    return 0.25 + 0.75 * std::clamp(trust, 0.0, 1.0);
}

// ========================= LDCD_PROBE_CEDE =================================
// Zone-scoped Field B cede-attribution probe for the 2D threshold revisit.
// Measurement only: per pixel, which gate stood the comb down (region cause
// bits + policy branch), and whether a vertically corroborated lurch step run
// covers the column -- i.e. whether the coarse delta the gate keyed on is an
// EXPLAINED luma edge. Reports P(cede | evidence) vs P(cede | bare) per frame
// over an optional line/column window (LDCD_PROBE_CEDE_L0/L1/C0/C1).
// Single accumulator: run with -t 1.
struct FieldBCedeProbe {
    bool enabled = false;
    int l0 = 0, l1 = 1 << 30, c0 = 0, c1 = 1 << 30;
    long frameIdx = 0;
    int lastLine = -1;

    long px = 0;
    long reasonCounts[8] = {};
    long cededPx = 0;
    long regionCauseCounts[8] = {};   // indexed by cause-bit position 1..6
    long policyCauseCounts[6] = {};
    long coveredPx = 0;
    long cededCovered = 0;
    long cededBare = 0;
    double cededCoveredHSum = 0.0, cededCoveredHMax = 0.0;
    double wSumCovered = 0.0, wSumBare = 0.0;
    long rolloffPx = 0, rolloffCovered = 0, rolloffHardPx = 0;
    long recovPx = 0, recovCovered = 0;
    // Seed/verdict attribution vs lurch coverage: is the region evaluator's
    // Different/seed density itself keyed to explained luma edges?
    long seedCovered = 0, seedBare = 0;
    long diffCovered = 0, diffBare = 0;
    long asymCovered = 0, asymBare = 0;

    // Leak-transfer (kappa) regression: per chroma bin (center envelope IRE
    // <4 / 4-12 / >=12), the region evaluator's measured per-leg differenceIRE
    // against corroborated step height h at covered columns, vs the bare
    // baseline. Samples are (leg, pixel) pairs where the evaluator actually
    // measured a difference (differenceIRE > 0).
    struct KappaBin {
        long nBare = 0; double sumDBare = 0.0;
        long nCov = 0; double sumDCov = 0.0, sumH = 0.0, sumHD = 0.0, sumH2 = 0.0;
        long diffVCov = 0, diffVBare = 0;  // Different-verdict count per class
        void reset() { *this = KappaBin(); }
    };
    KappaBin kappaBins[3];

    static int chromaBin(double envIRE)
    { return envIRE < 4.0 ? 0 : (envIRE < 12.0 ? 1 : 2); }

    FieldBCedeProbe()
    {
        enabled = std::getenv("LDCD_PROBE_CEDE") != nullptr;
        auto envInt = [](const char *name, int fallback) {
            const char *s = std::getenv(name);
            return s ? std::atoi(s) : fallback;
        };
        l0 = envInt("LDCD_PROBE_CEDE_L0", 0);
        l1 = envInt("LDCD_PROBE_CEDE_L1", 1 << 30);
        c0 = envInt("LDCD_PROBE_CEDE_C0", 0);
        c1 = envInt("LDCD_PROBE_CEDE_C1", 1 << 30);
    }

    // Band-uniformity: inside a chroma-boundary band the doctrine requires ONE
    // render for the whole region (grail chain #3). Column-to-column decision
    // changes inside a band ARE the per-column interleave that manufactures
    // edge beading, so the switching rate is the direct instrument for it --
    // unlike alternation energy, which cannot see a manufactured colour.
    long bandPx = 0, bandSwitches = 0, bandRuns = 0;
    long freePx = 0, freeSwitches = 0;

    void noteBandUniformity(const std::uint8_t *reason,
                            const std::uint8_t *inBand,
                            int width)
    {
        if (!enabled || !reason || !inBand) return;
        const int lo = std::max(0, c0);
        const int hi = std::min(width - 1, c1);
        for (int x = lo; x <= hi; ++x) {
            const bool band = inBand[x] != 0;
            if (band) {
                bandPx++;
                if (x == lo || inBand[x - 1] == 0) bandRuns++;
                else if (reason[x] != reason[x - 1]) bandSwitches++;
            } else {
                freePx++;
                if (x > lo && inBand[x - 1] == 0 &&
                    reason[x] != reason[x - 1]) freeSwitches++;
            }
        }
    }

    void flush()
    {
        if (px <= 0 && bandPx <= 0) { frameIdx++; return; }
        if (px <= 0) {
            // Rebuilt path: only the shared band-uniformity instrument runs.
            std::fprintf(stderr,
                "[CEDE f=%ld ln=%d-%d col=%d-%d] (rebuilt)\n"
                "  band uniformity: inBand %.1f%% | switches/inBandPx %.3f "
                "(%.2f per band run) | outside-band switches/px %.3f\n",
                frameIdx, l0, std::min(l1, 9999), c0, std::min(c1, 9999),
                100.0 * bandPx / std::max(1L, bandPx + freePx),
                double(bandSwitches) / bandPx,
                bandRuns > 0 ? double(bandSwitches) / bandRuns : 0.0,
                freePx > 0 ? double(freeSwitches) / freePx : 0.0);
            frameIdx++;
            bandPx = bandSwitches = bandRuns = 0;
            freePx = freeSwitches = 0;
            return;
        }
        const double inv = 100.0 / px;
        const double invCede = cededPx > 0 ? 100.0 / cededPx : 0.0;
        std::fprintf(stderr,
            "[CEDE f=%ld ln=%d-%d col=%d-%d] px=%ld\n"
            "  reasons%%: ctr %.1f cede %.1f blnd %.1f 1leg %.1f recv %.1f hold %.1f\n"
            "  region causes (%%ceded): band %.1f island %.1f asym %.1f diff %.1f "
            "1legfail %.1f hue %.1f\n"
            "  policy causes (%%ceded): flags %.1f invEdge %.1f islasym %.1f "
            "edgeNoCont %.1f hard %.1f\n",
            frameIdx, l0, std::min(l1, 9999), c0, std::min(c1, 9999), px,
            reasonCounts[2] * inv, reasonCounts[3] * inv, reasonCounts[1] * inv,
            reasonCounts[4] * inv, reasonCounts[5] * inv, reasonCounts[6] * inv,
            regionCauseCounts[1] * invCede, regionCauseCounts[2] * invCede,
            regionCauseCounts[3] * invCede, regionCauseCounts[4] * invCede,
            regionCauseCounts[5] * invCede, regionCauseCounts[6] * invCede,
            policyCauseCounts[1] * invCede, policyCauseCounts[2] * invCede,
            policyCauseCounts[3] * invCede, policyCauseCounts[4] * invCede,
            policyCauseCounts[5] * invCede);
        const long barePx = px - coveredPx;
        std::fprintf(stderr,
            "  lurch: covered %.1f%% | P(cede|cov) %.1f%% vs P(cede|bare) %.1f%% | "
            "ceded&cov h mean %.2f max %.2f IRE\n"
            "  wSum(mean): cov %.2f bare %.2f | rolloff>0.25: %.1f%% "
            "(P(cov|rolloff)=%.1f%%) hard: %.1f%% | recovery %.1f%% (cov %.1f%%)\n",
            coveredPx * inv,
            coveredPx > 0 ? 100.0 * cededCovered / coveredPx : 0.0,
            barePx > 0 ? 100.0 * cededBare / barePx : 0.0,
            cededCovered > 0 ? cededCoveredHSum / cededCovered : 0.0,
            cededCoveredHMax,
            coveredPx > 0 ? wSumCovered / coveredPx : 0.0,
            barePx > 0 ? wSumBare / barePx : 0.0,
            rolloffPx * inv,
            rolloffPx > 0 ? 100.0 * rolloffCovered / rolloffPx : 0.0,
            rolloffHardPx * inv,
            recovPx * inv,
            recovPx > 0 ? 100.0 * recovCovered / recovPx : 0.0);
        std::fprintf(stderr,
            "  verdict density: P(seed|cov) %.1f%% vs P(seed|bare) %.1f%% | "
            "P(diff|cov) %.1f%% vs bare %.1f%% | P(asym|cov) %.1f%% vs bare %.1f%%\n",
            coveredPx > 0 ? 100.0 * seedCovered / coveredPx : 0.0,
            barePx > 0 ? 100.0 * seedBare / barePx : 0.0,
            coveredPx > 0 ? 100.0 * diffCovered / coveredPx : 0.0,
            barePx > 0 ? 100.0 * diffBare / barePx : 0.0,
            coveredPx > 0 ? 100.0 * asymCovered / coveredPx : 0.0,
            barePx > 0 ? 100.0 * asymBare / barePx : 0.0);
        std::fprintf(stderr,
            "  band uniformity: inBand %.1f%% | switches/inBandPx %.3f "
            "(%.2f per band run) | outside-band switches/px %.3f\n",
            100.0 * bandPx / std::max(1L, bandPx + freePx),
            bandPx > 0 ? double(bandSwitches) / bandPx : 0.0,
            bandRuns > 0 ? double(bandSwitches) / bandRuns : 0.0,
            freePx > 0 ? double(freeSwitches) / freePx : 0.0);
        static const char *binNames[3] = { "env<4", "env4-12", "env>=12" };
        for (int bi = 0; bi < 3; ++bi) {
            const KappaBin &K = kappaBins[bi];
            double slope = 0.0, meanH = 0.0;
            if (K.nCov > 1) {
                meanH = K.sumH / K.nCov;
                const double meanD = K.sumDCov / K.nCov;
                const double varH = K.sumH2 / K.nCov - meanH * meanH;
                if (varH > 1e-9)
                    slope = (K.sumHD / K.nCov - meanH * meanD) / varH;
            }
            std::fprintf(stderr,
                "  kappa[%s]: bare n=%ld meanD %.2f (diffV %.1f%%) | "
                "cov n=%ld meanD %.2f meanH %.2f slope %.3f (diffV %.1f%%)\n",
                binNames[bi],
                K.nBare, K.nBare > 0 ? K.sumDBare / K.nBare : 0.0,
                K.nBare > 0 ? 100.0 * K.diffVBare / K.nBare : 0.0,
                K.nCov, K.nCov > 0 ? K.sumDCov / K.nCov : 0.0,
                meanH, slope,
                K.nCov > 0 ? 100.0 * K.diffVCov / K.nCov : 0.0);
        }
        frameIdx++;
        px = 0;
        std::fill(std::begin(reasonCounts), std::end(reasonCounts), 0L);
        cededPx = 0;
        std::fill(std::begin(regionCauseCounts), std::end(regionCauseCounts), 0L);
        std::fill(std::begin(policyCauseCounts), std::end(policyCauseCounts), 0L);
        coveredPx = cededCovered = cededBare = 0;
        cededCoveredHSum = cededCoveredHMax = 0.0;
        wSumCovered = wSumBare = 0.0;
        rolloffPx = rolloffCovered = rolloffHardPx = 0;
        recovPx = recovCovered = 0;
        seedCovered = seedBare = 0;
        diffCovered = diffBare = 0;
        asymCovered = asymBare = 0;
        bandPx = bandSwitches = bandRuns = 0;
        freePx = freeSwitches = 0;
        for (KappaBin &K : kappaBins) K.reset();
    }
};

FieldBCedeProbe gCedeProbe;

// ========================= LDCD_PROBE_FRAMEB ===============================
// Frame B engagement/throttle census. Frame B's job is to cancel the
// vertically-invariant image-locked alien -- the 1D debris that stands in
// columns -- and its authority is a product of four terms:
//     pull = clamp(0.5 * combStrength * reachAuthority, 0, 0.5) * midLicense
// so any one of them at zero silences it. This reports which term is binding,
// split by leg symmetry: sym -> 0 IS the signature class (alien vertically
// invariant, midpoint safe and needed); sym -> 1 is diagonal advance, where
// refusal is correct. Windowed with the LDCD_PROBE_CEDE envs.
struct FrameBProbe {
    bool enabled = false;
    int l0 = 0, l1 = 1 << 30, c0 = 0, c1 = 1 << 30;
    long frameIdx = 0;
    int lastLine = -1;

    // bucket 0: sym < 0.35 (signature class), 1: 0.35-0.70, 2: >= 0.70
    struct Bucket {
        long n = 0, noPartner = 0, capBound = 0;
        double sumReachBase = 0, sumExempt = 0, sumReach = 0;
        double sumMidLic = 0, sumPull = 0, sumEffective = 0;
        double sumDeltaIRE = 0, sumMovedIRE = 0;
        long midLicZero = 0, reachZero = 0;
    } b[3];

    FrameBProbe()
    {
        enabled = std::getenv("LDCD_PROBE_FRAMEB") != nullptr;
        auto envInt = [](const char *n, int f) {
            const char *s = std::getenv(n); return s ? std::atoi(s) : f;
        };
        l0 = envInt("LDCD_PROBE_CEDE_L0", 0);
        l1 = envInt("LDCD_PROBE_CEDE_L1", 1 << 30);
        c0 = envInt("LDCD_PROBE_CEDE_C0", 0);
        c1 = envInt("LDCD_PROBE_CEDE_C1", 1 << 30);
    }

    Bucket alienB[3];   // same split, restricted to a clear alien signature
    long noSymEvidence = 0, noSymEvidenceAlien = 0;

    // Restricted to STRONG IMAGE VERTICALS (large carrier-free lateral luma
    // step): the columns where 1D debris stands and interfield is supposed to
    // cancel it. This is also exactly where the bevel/cross-colour throttle
    // collapses reach, so the terms are reported separately here.
    Bucket vertB[3];
    double vSumReachBase = 0, vSumExempt = 0, vSumReach = 0;
    long vN = 0;

    // Standing vs alternating decomposition of the blind 1D bandpass at the
    // Frame B (+-1) and Field B (+-2) geometries. Real chroma ALTERNATES
    // between carrier-opposite lines; the bandpass leak of a vertical luma
    // edge is IDENTICAL on every line (same D^2 Y), i.e. STANDING -- and a
    // standing carrier-band component is invisible to a difference comb:
    // center - neighbor = 0. If standing energy dominates at strong image
    // verticals, the un-cancelled 74% is not an authority problem at all.
    double vAlt1 = 0, vStand1 = 0, vAlt2 = 0, vStand2 = 0; long vBpN = 0;
    double oAlt1 = 0, oStand1 = 0, oAlt2 = 0, oStand2 = 0; long oBpN = 0;

    // Up/down leak asymmetry at verticals, from the carrier-free aperture
    // means: the alternating alien per line follows that line's lateral
    // luma curvature (leak = -0.25 * D^2_2 Y), so the +-1 midpoint's
    // residual fraction is |D2up - D2dn| / (D2up + D2dn). If this ratio is
    // large, a_up != a_dn and the midpoint CANNOT null the alien -- the
    // 29% delivery is then a model limit, not a licence problem.
    double vAsymSum = 0, vD2Sum = 0, vPairSum = 0; long vAsymN = 0, vAsymHi = 0;
    double acSum = 0, acMax = 0; long acN = 0, acAllN = 0;

    void noteAsymCorr(double ire)
    {
        acAllN++;
        if (ire > 0.0) { acN++; acSum += ire; acMax = std::max(acMax, ire); }
    }

    // kappa_FB regression: |pairDiff| (IQ IRE, = the alien SUM the +-1 pair
    // exposes) against (D2u + D2d) (composite IRE, the luma prediction of
    // that same sum). The slope carries BOTH the leak transfer and the
    // fullSignedIQ-vs-composite scale factor, which is exactly what the
    // residual predictor needs. Restricted to strong verticals with a clear
    // alien signature so real vertical chroma difference does not dominate.
    double kX = 0, kY = 0, kXX = 0, kXY = 0, kYY = 0; long kN = 0;

    // Sub-sample REGISTRATION between the centre line and each +-1 leg,
    // estimated from carrier-free aperture means only. At a true image
    // vertical the luma at a given x is the same on every line, so any
    // difference is a horizontal shift: delta ~= (Yleg - Yc) / (dY/dx).
    // Split by centre-line parity because under progressive telecine the
    // +-1 legs are the OTHER field -- a field-to-field registration error
    // must therefore alternate sign with parity, while a genuine image
    // slope does not.
    double rgUpSum[2] = {0,0}, rgDnSum[2] = {0,0};
    double rgUpAbs = 0, rgDnAbs = 0;
    long rgN[2] = {0,0};
    // Does leg disagreement track the lateral GRADIENT (registration) or the
    // CURVATURE (leak)? Correlate |pairDiff| against each.
    double rgGX = 0, rgGY = 0, rgGXX = 0, rgGXY = 0, rgGYY = 0; long rgGN = 0;

    void noteReg(int parity, double dUp, double dDn)
    {
        const int p = parity & 1;
        rgUpSum[p] += dUp; rgDnSum[p] += dDn;
        rgUpAbs += std::fabs(dUp); rgDnAbs += std::fabs(dDn);
        rgN[p]++;
    }

    void noteGrad(double absGrad, double pairIRE)
    {
        rgGX += absGrad; rgGY += pairIRE; rgGXX += absGrad * absGrad;
        rgGXY += absGrad * pairIRE; rgGYY += pairIRE * pairIRE; rgGN++;
    }

    void noteKappa(double d2sum, double pairIRE)
    {
        kX += d2sum; kY += pairIRE; kXX += d2sum * d2sum;
        kXY += d2sum * pairIRE; kYY += pairIRE * pairIRE; kN++;
    }

    void noteAsym(double r, double d2SumIRE, double pairIRE)
    {
        vAsymSum += r; vD2Sum += d2SumIRE; vPairSum += pairIRE;
        vAsymN++;
        if (r > 0.5) vAsymHi++;
    }

    void noteBp(bool vertical, double alt1, double stand1,
                double alt2, double stand2)
    {
        if (vertical) { vAlt1 += alt1; vStand1 += stand1;
                        vAlt2 += alt2; vStand2 += stand2; vBpN++; }
        else          { oAlt1 += alt1; oStand1 += stand1;
                        oAlt2 += alt2; oStand2 += stand2; oBpN++; }
    }

    void note(double sym, bool havePartner, double reachBase, double exempt,
              double reach, double midLic, double pull, double deltaIRE,
              double movedIRE, bool capBound, double aGate, bool symMeasured,
              double hLumaIRE)
    {
        const int bi = sym < 0.35 ? 0 : (sym < 0.70 ? 1 : 2);
        if (hLumaIRE > 14.0) {
            Bucket &V = vertB[bi];
            V.n++;
            if (havePartner) {
                vN++;
                vSumReachBase += reachBase; vSumExempt += exempt;
                vSumReach += reach;
                V.sumMidLic += midLic; V.sumPull += pull;
                V.sumEffective += pull * midLic;
                V.sumDeltaIRE += deltaIRE; V.sumMovedIRE += movedIRE;
                if (midLic <= 1e-9) V.midLicZero++;
                if (reach <= 1e-9) V.reachZero++;
            } else {
                V.noPartner++;
            }
        }
        if (!symMeasured) {
            noSymEvidence++;
            if (aGate > 0.5) noSymEvidenceAlien++;
        }
        if (aGate > 0.5) {
            Bucket &A = alienB[bi];
            A.n++;
            if (havePartner) {
                A.sumMidLic += midLic; A.sumPull += pull;
                A.sumEffective += pull * midLic;
                A.sumDeltaIRE += deltaIRE; A.sumMovedIRE += movedIRE;
                if (midLic <= 1e-9) A.midLicZero++;
            } else {
                A.noPartner++;
            }
        }
        Bucket &B = b[bi];
        B.n++;
        if (!havePartner) { B.noPartner++; return; }
        B.sumReachBase += reachBase; B.sumExempt += exempt; B.sumReach += reach;
        B.sumMidLic += midLic; B.sumPull += pull;
        B.sumEffective += pull * midLic;
        B.sumDeltaIRE += deltaIRE; B.sumMovedIRE += movedIRE;
        if (midLic <= 1e-9) B.midLicZero++;
        if (reach <= 1e-9) B.reachZero++;
        if (capBound) B.capBound++;
    }

    void flush()
    {
        long tot = b[0].n + b[1].n + b[2].n;
        if (tot <= 0) { frameIdx++; return; }
        static const char *names[3] = { "sym<0.35 SIGNATURE", "sym 0.35-0.70   ",
                                        "sym>=0.70 diagonal" };
        std::fprintf(stderr, "[FRAMEB f=%ld ln=%d-%d col=%d-%d] px=%ld\n",
                     frameIdx, l0, std::min(l1, 9999), c0, std::min(c1, 9999), tot);
        for (int i = 0; i < 3; ++i) {
            const Bucket &B = b[i];
            if (B.n == 0) continue;
            const long eng = B.n - B.noPartner;
            const double e = eng > 0 ? 1.0 / eng : 0.0;
            std::fprintf(stderr,
                "  %s %5.1f%% of px | noPartner %5.1f%% | reachBase %.2f "
                "exempt %.2f reach %.2f | midLic %.2f (zero %5.1f%%) | "
                "pull %.3f eff %.3f | delta %5.2f moved %5.2f IRE (%4.1f%%) "
                "cap %4.1f%%\n",
                names[i], 100.0 * B.n / tot, 100.0 * B.noPartner / B.n,
                B.sumReachBase * e, B.sumExempt * e, B.sumReach * e,
                B.sumMidLic * e, 100.0 * B.midLicZero * e,
                B.sumPull * e, B.sumEffective * e,
                B.sumDeltaIRE * e, B.sumMovedIRE * e,
                B.sumDeltaIRE > 0 ? 100.0 * B.sumMovedIRE / B.sumDeltaIRE : 0.0,
                100.0 * B.capBound * e);
        }
        const long alienTot = alienB[0].n + alienB[1].n + alienB[2].n;
        std::fprintf(stderr,
            "  [alien signature aGate>0.5] %.1f%% of px | "
            "no-sym-evidence %.1f%% of frame (%.1f%% of alien px)\n",
            100.0 * alienTot / tot,
            100.0 * noSymEvidence / tot,
            alienTot > 0 ? 100.0 * noSymEvidenceAlien / alienTot : 0.0);
        for (int i = 0; i < 3; ++i) {
            const Bucket &A = alienB[i];
            if (A.n == 0) continue;
            const long eng = A.n - A.noPartner;
            const double e = eng > 0 ? 1.0 / eng : 0.0;
            std::fprintf(stderr,
                "    %s %5.1f%% | midLic %.2f (zero %5.1f%%) | eff %.3f | "
                "delta %5.2f moved %5.2f IRE (%4.1f%%)\n",
                names[i], 100.0 * A.n / std::max(1L, alienTot),
                A.sumMidLic * e, 100.0 * A.midLicZero * e, A.sumEffective * e,
                A.sumDeltaIRE * e, A.sumMovedIRE * e,
                A.sumDeltaIRE > 0 ? 100.0 * A.sumMovedIRE / A.sumDeltaIRE : 0.0);
        }
        const long vTot = vertB[0].n + vertB[1].n + vertB[2].n;
        if (vTot > 0) {
            const double ve = vN > 0 ? 1.0 / vN : 0.0;
            std::fprintf(stderr,
                "  [STRONG IMAGE VERTICAL hLuma>14 IRE] %.1f%% of px | "
                "reachBase %.2f exempt %.2f reach %.2f\n",
                100.0 * vTot / tot,
                vSumReachBase * ve, vSumExempt * ve, vSumReach * ve);
            for (int i = 0; i < 3; ++i) {
                const Bucket &V = vertB[i];
                if (V.n == 0) continue;
                const long eng = V.n - V.noPartner;
                const double e = eng > 0 ? 1.0 / eng : 0.0;
                std::fprintf(stderr,
                    "    %s %5.1f%% | midLic %.2f (zero %5.1f%%) | eff %.3f | "
                    "delta %5.2f moved %5.2f IRE (%4.1f%%) | reachZero %4.1f%%\n",
                    names[i], 100.0 * V.n / vTot,
                    V.sumMidLic * e, 100.0 * V.midLicZero * e,
                    V.sumEffective * e, V.sumDeltaIRE * e, V.sumMovedIRE * e,
                    V.sumDeltaIRE > 0 ? 100.0 * V.sumMovedIRE / V.sumDeltaIRE : 0.0,
                    100.0 * V.reachZero * e);
            }
        }
        if (acAllN > 0) {
            std::fprintf(stderr,
                "  leak-asymmetry correction: fired on %.1f%% of px, "
                "mean %.2f max %.2f IRE\n",
                100.0 * acN / acAllN,
                acN > 0 ? acSum / acN : 0.0, acMax);
        }
        if (rgN[0] + rgN[1] > 2) {
            const long n0 = std::max(1L, rgN[0]), n1 = std::max(1L, rgN[1]);
            const long nt = rgN[0] + rgN[1];
            std::fprintf(stderr,
                "  registration @verticals (samples, carrier-free luma):\n"
                "    parity0 up %+.4f dn %+.4f (n=%ld) | parity1 up %+.4f dn %+.4f (n=%ld)\n"
                "    mean |shift| up %.4f dn %.4f | parity-alternating component up %+.4f dn %+.4f\n",
                rgUpSum[0]/n0, rgDnSum[0]/n0, rgN[0],
                rgUpSum[1]/n1, rgDnSum[1]/n1, rgN[1],
                rgUpAbs/nt, rgDnAbs/nt,
                0.5*(rgUpSum[0]/n0 - rgUpSum[1]/n1),
                0.5*(rgDnSum[0]/n0 - rgDnSum[1]/n1));
        }
        if (rgGN > 2) {
            const double mx = rgGX/rgGN, my = rgGY/rgGN;
            const double vxx = rgGXX/rgGN - mx*mx, vyy = rgGYY/rgGN - my*my;
            const double vxy = rgGXY/rgGN - mx*my;
            std::fprintf(stderr,
                "  |pairDiff| vs lateral GRADIENT: slope %.3f r=%.3f (mean |grad| %.1f IRE/sample)\n",
                vxx > 1e-9 ? vxy/vxx : 0.0,
                (vxx > 1e-9 && vyy > 1e-9) ? vxy/std::sqrt(vxx*vyy) : 0.0, mx);
        }
        if (kN > 2) {
            const double mx = kX / kN, my = kY / kN;
            const double vxx = kXX / kN - mx * mx;
            const double vyy = kYY / kN - my * my;
            const double vxy = kXY / kN - mx * my;
            const double slope = vxx > 1e-9 ? vxy / vxx : 0.0;
            const double r = (vxx > 1e-9 && vyy > 1e-9)
                ? vxy / std::sqrt(vxx * vyy) : 0.0;
            std::fprintf(stderr,
                "  kappa_FB regression (n=%ld): |pairDiff| = %.3f * (D2u+D2d) "
                "+ %.2f  | r=%.3f  | meanD2sum %.1f meanPair %.2f IRE\n",
                kN, slope, my - slope * mx, r, mx, my);
        }
        if (vAsymN > 0) {
            std::fprintf(stderr,
                "  leak asymmetry @verticals: mean |D2u-D2d|/(D2u+D2d) %.2f "
                "(r>0.5: %.0f%%) | mean D2 sum %.1f IRE | mean |pairDiff| %.2f IRE | "
                "predicted midpoint delivery %.0f%%\n",
                vAsymSum / vAsymN, 100.0 * vAsymHi / vAsymN,
                vD2Sum / vAsymN, vPairSum / vAsymN,
                100.0 * (1.0 - vAsymSum / vAsymN));
        }
        if (vBpN > 0 && oBpN > 0) {
            std::fprintf(stderr,
                "  bandpass split @verticals: +-1 alt %.2f stand %.2f (%.0f%% standing) | "
                "+-2 alt %.2f stand %.2f (%.0f%% standing)\n"
                "  bandpass split off-vert:   +-1 alt %.2f stand %.2f (%.0f%% standing) | "
                "+-2 alt %.2f stand %.2f (%.0f%% standing)\n",
                vAlt1 / vBpN, vStand1 / vBpN,
                100.0 * vStand1 / std::max(1e-9, vAlt1 + vStand1),
                vAlt2 / vBpN, vStand2 / vBpN,
                100.0 * vStand2 / std::max(1e-9, vAlt2 + vStand2),
                oAlt1 / oBpN, oStand1 / oBpN,
                100.0 * oStand1 / std::max(1e-9, oAlt1 + oStand1),
                oAlt2 / oBpN, oStand2 / oBpN,
                100.0 * oStand2 / std::max(1e-9, oAlt2 + oStand2));
        }
        frameIdx++;
        for (Bucket &B : b) B = Bucket();
        for (Bucket &A : alienB) A = Bucket();
        for (Bucket &V : vertB) V = Bucket();
        vSumReachBase = vSumExempt = vSumReach = 0.0; vN = 0;
        vAsymSum = vD2Sum = vPairSum = 0.0; vAsymN = vAsymHi = 0;
        acSum = acMax = 0.0; acN = acAllN = 0;
        kX = kY = kXX = kXY = kYY = 0.0; kN = 0;
        rgUpSum[0]=rgUpSum[1]=rgDnSum[0]=rgDnSum[1]=0.0;
        rgUpAbs=rgDnAbs=0.0; rgN[0]=rgN[1]=0;
        rgGX=rgGY=rgGXX=rgGXY=rgGYY=0.0; rgGN=0;
        vAlt1 = vStand1 = vAlt2 = vStand2 = 0.0; vBpN = 0;
        oAlt1 = oStand1 = oAlt2 = oStand2 = 0.0; oBpN = 0;
        noSymEvidence = noSymEvidenceAlien = 0;
    }
};

FrameBProbe gFrameBProbe;

} // namespace

void Comb::FrameBuffer::invalidateCombTapCache()
{
    tapLineCacheLine = { -1, -1, -1 };
    for (auto &tapLine : tapLineCache) {
        tapLine.cacheLine = -1;
        tapLine.builtFlags = 0;
    }
    // The memoised smoothed signed-IQ rows read the locked demod, which is
    // rebuilt per frame; clear their validity in lockstep with the tap cache.
    std::fill(smoothedLockedRowValid.begin(),
              smoothedLockedRowValid.end(),
              std::uint8_t{0});
}

// Fill the 7-tap smoothed signed-IQ row for `line` once per frame.  The
// balanced end-weighted aperture (0.5,1,1,1,1,1,0.5) now smooths the
// integer-centred baseband products published by buildPhaseCorrected1D.
// Normalize by its total weight (6): the input already has the full-signed-IQ
// scale, so no carrier-lattice compensation remains to be done here.
void Comb::FrameBuffer::ensureSmoothedLockedRow(int line)
{
    if (line < 0 || line >= demodLines)
        return;
    if ((int)smoothedLockedRowValid.size() != demodLines)
        return; // non-locked path: buffers not sized
    if (smoothedLockedRowValid[line])
        return;

    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;
    if (width <= 0)
        return;

    const float *iRow = locked1DTI4fsc_line(line);
    const float *qRow = locked1DTQ4fsc_line(line);
    float *sI = smoothedLockedTI_flat.data() + static_cast<size_t>(line) * demodWidth;
    float *sQ = smoothedLockedTQ_flat.data() + static_cast<size_t>(line) * demodWidth;

    static constexpr double w[7] = {0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5};
    for (int rel = 0; rel < width; ++rel) {
        double si = 0.0;
        double sq = 0.0;
        for (int k = -3; k <= 3; ++k) {
            const int rk = reflectCombRel(rel + k, width);
            si += w[k + 3] * static_cast<double>(iRow[rk]);
            sq += w[k + 3] * static_cast<double>(qRow[rk]);
        }
        sI[rel] = static_cast<float>(si / 6.0);
        sQ[rel] = static_cast<float>(sq / 6.0);
    }
    smoothedLockedRowValid[line] = std::uint8_t{1};
}

const Comb::FrameBuffer::CombTapLine &Comb::FrameBuffer::ensureCombTapLine(int lineNumber)
{
    const int slot = precleanRingSlot(lineNumber);
    CombTapLine &tapLine = tapLineCache[slot];
    if (tapLineCacheLine[slot] != lineNumber || tapLine.cacheLine != lineNumber) {
        tapLineCacheLine[slot] = lineNumber;
        tapLine.cacheLine = -1;
        tapLine.builtFlags = 0;
        buildCombTapLine(lineNumber, tapLine);
    }
    return tapLine;
}

void Comb::FrameBuffer::buildCombTapLine(int lineNumber, CombTapLine &tapLine)
{
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;
    const int fieldWidth = videoParameters.fieldWidth;

    tapLine.width = std::max(0, width);
    if (width <= 0 || lineNumber < first || lineNumber >= last) return;

    tapLine.cacheLine = lineNumber;
    tapLine.width = width;

    const unsigned flags = combTapBuildFlags_;
    const bool wantFieldA = (flags & TapBuildFieldA) != 0;
    const bool wantFieldB = (flags & TapBuildFieldB) != 0;
    const bool wantFrame  = (flags & TapBuildFrame)  != 0;

    // Contour is not only an intrafield-comb helper.  Frame B uses the same
    // same-column evidence as support for +/-1 interfield cancellation at
    // luma-contrast sites.
    const bool wantContour = wantFieldA || wantFieldB || wantFrame;

    auto activeLine = [&](int ln)->bool {
        return ln >= first && ln < last;
    };
    
    auto sameFieldActiveLine = [&](int ln)->bool {
        return activeLine(ln) &&
               carrierLineParity(ln) == carrierLineParity(lineNumber);
    };
    
    struct ResolvedTap {
        int req = -1;
        int ln = -1;
        bool geomHave = false;
        bool have = false;
        bool mirrored = false;
    };
    
    // Opposite-field/frame taps are not mirrored here.  If a ±1 line does not
    // exist, the frame consumer sees missing support.
    auto resolveFrameTap = [&](int offset)->ResolvedTap {
        ResolvedTap r;
        r.req = lineNumber + offset;
        r.geomHave = activeLine(r.req);
        r.have = r.geomHave;
        r.ln = r.have ? r.req : -1;
        return r;
    };
    
    // Same-field taps use active-boundary mirror resolution.  If the requested
    // outward partner is missing but the inward same-field partner exists, resolve
    // to that inward partner.  This restores the old top/bottom behavior:
    //
    //   top active row:    U2 resolves to D2
    //   bottom active row: D2 resolves to U2
    //
    // The same rule is applied to ±4 so Field A's wider same-field context does
    // not silently clamp to arbitrary active lines.
    auto resolveSameFieldTap = [&](int primaryOffset, int mirrorOffset)->ResolvedTap {
        ResolvedTap r;
        r.req = lineNumber + primaryOffset;
    
        if (sameFieldActiveLine(r.req)) {
            r.geomHave = true;
            r.have = true;
            r.ln = r.req;
            return r;
        }
    
        const int mirror = lineNumber + mirrorOffset;
        if (sameFieldActiveLine(mirror)) {
            r.geomHave = false;
            r.have = true;
            r.mirrored = true;
            r.ln = mirror;
            return r;
        }
    
        r.geomHave = false;
        r.have = false;
        r.mirrored = false;
        r.ln = -1;
        return r;
    };
    
    const ResolvedTap u1 = resolveFrameTap(-1);
    const ResolvedTap d1 = resolveFrameTap(+1);
    const ResolvedTap u2 = resolveSameFieldTap(-2, +2);
    const ResolvedTap d2 = resolveSameFieldTap(+2, -2);
    const ResolvedTap u4 = resolveSameFieldTap(-4, +4);
    const ResolvedTap d4 = resolveSameFieldTap(+4, -4);
    
    tapLine.ln0 = lineNumber;
    
    tapLine.reqU1 = u1.req;
    tapLine.reqD1 = d1.req;
    tapLine.reqU2 = u2.req;
    tapLine.reqD2 = d2.req;
    tapLine.reqU4 = u4.req;
    tapLine.reqD4 = d4.req;
    
    tapLine.lnU1 = u1.ln;
    tapLine.lnD1 = d1.ln;
    tapLine.lnU2 = u2.ln;
    tapLine.lnD2 = d2.ln;
    tapLine.lnU4 = u4.ln;
    tapLine.lnD4 = d4.ln;
    
    tapLine.geomHaveU1 = u1.geomHave;
    tapLine.geomHaveD1 = d1.geomHave;
    tapLine.geomHaveU2 = u2.geomHave;
    tapLine.geomHaveD2 = d2.geomHave;
    tapLine.geomHaveU4 = u4.geomHave;
    tapLine.geomHaveD4 = d4.geomHave;
    
    tapLine.mirroredU2 = u2.mirrored;
    tapLine.mirroredD2 = d2.mirrored;
    tapLine.mirroredU4 = u4.mirrored;
    tapLine.mirroredD4 = d4.mirrored;
    
    tapLine.haveU1 = wantFrame && u1.have;
    tapLine.haveD1 = wantFrame && d1.have;
    tapLine.haveU2 = wantContour && u2.have;
    tapLine.haveD2 = wantContour && d2.have;
    tapLine.haveU4 = wantContour && u4.have;
    tapLine.haveD4 = wantContour && d4.have;
    
    auto ensureWidth = [&](auto &v) {
        if ((int)v.size() != width) v.resize(width);
    };

    ensureWidth(tapLine.tap0);

    if (wantFrame) {
        ensureWidth(tapLine.tapU1);
        ensureWidth(tapLine.tapD1);
        ensureWidth(tapLine.pairU1);
        ensureWidth(tapLine.pairD1);
    }

    if (wantContour) {
        ensureWidth(tapLine.tapU2);
        ensureWidth(tapLine.tapD2);
        ensureWidth(tapLine.pairU2);
        ensureWidth(tapLine.pairD2);
        ensureWidth(tapLine.intrafieldRegionReach);
        ensureWidth(tapLine.regionUp4);
        ensureWidth(tapLine.regionDown4);
    }

    // Cleared on every build; the contour pass below re-establishes it only
    // where both neighbour rows carry real luma evidence.  A stale true from a
    // previous line would be exactly the false authority this flag exists to
    // prevent.
    tapLine.coarseLumaValid = false;

    if (wantContour) {
        ensureWidth(tapLine.tapU4);
        ensureWidth(tapLine.tapD4);
        ensureWidth(tapLine.contour);
        ensureWidth(tapLine.movingCoarseContour);
        ensureWidth(tapLine.coarse0IRE);
        ensureWidth(tapLine.coarseU2IRE);
        ensureWidth(tapLine.coarseD2IRE);
    }

    if (wantFieldB || wantFrame) {
        ensureWidth(tapLine.hLumaDeltaIRE);
    }

    auto getCompRow = [&](int ln)->const double* {
        if (ln < first || ln >= last) return nullptr;
        if (configuration.phaseCompensation)
            return locked1DSource_line(ln);
        return bucketScalar1D_line(ln) + left;
    };

    auto getRawRow = [&](int ln)->const quint16* {
        if (ln < first || ln >= last) return nullptr;
        if (ln < 0 || ln >= demodLines) return nullptr;
        if (rawbuffer.isEmpty()) return nullptr;
        return rawbuffer.data() + static_cast<size_t>(ln) * fieldWidth + left;
    };

    struct RowRefs {
        int ln = -1;
        const quint16 *raw = nullptr;
        const double *comp = nullptr;
        bool haveLine = false;
    };

    auto rowRefs = [&](int ln, bool haveLine)->RowRefs {
        RowRefs r;
        r.ln = ln;
        r.haveLine = haveLine;
        r.raw = haveLine ? getRawRow(ln) : nullptr;
        r.comp = haveLine ? getCompRow(ln) : nullptr;
        return r;
    };

    RowRefs r0  = rowRefs(tapLine.ln0,  true);
    RowRefs rU1 = rowRefs(tapLine.lnU1, tapLine.haveU1);
    RowRefs rD1 = rowRefs(tapLine.lnD1, tapLine.haveD1);
    RowRefs rU2 = rowRefs(tapLine.lnU2, tapLine.haveU2);
    RowRefs rD2 = rowRefs(tapLine.lnD2, tapLine.haveD2);
    RowRefs rU4 = rowRefs(tapLine.lnU4, tapLine.haveU4);
    RowRefs rD4 = rowRefs(tapLine.lnD4, tapLine.haveD4);

    auto getLumaRow = [&](int ln)->const double* {
        if (!configuration.phaseCompensation ||
            !lockedLumaCacheValid ||
            demodWidth < width ||
            lockedLumaSmooth_flat.empty() ||
            ln < 0 || ln >= demodLines)
        {
            return nullptr;
        }
        return lockedLumaSmooth_line(ln);
    };

    const double *luma0  = getLumaRow(tapLine.ln0);
    const double *lumaU2 = getLumaRow(tapLine.lnU2);
    const double *lumaD2 = getLumaRow(tapLine.lnD2);
    const double *lumaU4 = getLumaRow(tapLine.lnU4);
    const double *lumaD4 = getLumaRow(tapLine.lnD4);

    auto fillTap = [&](const RowRefs &r,
                       std::vector<CombTapScalar> &dst) {
        if (r.comp == nullptr) {
            for (int rel = 0; rel < width; ++rel)
                dst[rel] = CombTapScalar();
            return;
        }

        for (int rel = 0; rel < width; ++rel) {
            const int rm1 = reflectCombRel(rel - 1, width);
            const int rp1 = reflectCombRel(rel + 1, width);

            CombTapScalar &s = dst[rel];
            s.raw = r.raw ? static_cast<double>(r.raw[rel]) : 0.0;
            s.comp = r.comp[rel];
            s.symMag = 0.5 * (std::fabs(r.comp[rm1]) + std::fabs(r.comp[rp1]));
        }
    };

    {
        fillTap(r0, tapLine.tap0);

        if (wantFrame) {
            fillTap(rU1, tapLine.tapU1);
            fillTap(rD1, tapLine.tapD1);
        }

        if (wantContour) {
            fillTap(rU2, tapLine.tapU2);
            fillTap(rD2, tapLine.tapD2);
            fillTap(rU4, tapLine.tapU4);
            fillTap(rD4, tapLine.tapD4);
        }
    }

    const auto &T = configuration.tunables;
    const double kRange = T.FIELD_K_RANGE_IRE * irescale;
    const double invK = (kRange > 1e-9) ? (1.0 / kRange) : 0.0;
    const double invI = invIreScale;
    const bool wantCenterEnvelope = wantFieldA || wantFieldB || wantFrame;
    if (wantCenterEnvelope) {
        ensureWidth(tapLine.centerEnvelope);
        ensureWidth(tapLine.centerAdmittedChromaT);
        const auto *centerAnalysis = carrierAnalysis_line(tapLine.ln0);
        for (int rel = 0; rel < width; ++rel) {
            const CombTapScalar &s = tapLine.tap0[rel];
            const double envC = boundedMag(s.comp, s.symMag);
            tapLine.centerEnvelope[rel] = envC;
            const double carrierTrust = centerAnalysis
                ? lddecode::carrierTrust(
                    centerAnalysis[rel].carrierConformance,
                    centerAnalysis[rel].conformanceUsableAxisFraction)
                : 0.5;
            const double admission =
                std::clamp(2.0 * (carrierTrust - 0.5), 0.0, 1.0);
            tapLine.centerAdmittedChromaT[rel] = admission *
                std::clamp((envC * invI - 2.0) / 8.0, 0.0, 1.0);
        }
    }

    const lddecode::CombReachSourceFrame scalarSource = scalarReachSource();
    const lddecode::CombReachSourceFrame iqSource = iqReachSource();

    auto legalGateForReachUse = [](const lddecode::CombReachReply &reach,
                                   lddecode::CombReachUse reachUse)->double {
        switch (reachUse) {
        case lddecode::CombReachUse::FieldScalarAverage:
        case lddecode::CombReachUse::FrameScalarAverage:
            return reach.allowScalarAverage ? 1.0 : 0.0;

        case lddecode::CombReachUse::FieldScalarCancel:
        case lddecode::CombReachUse::FrameScalarCancel:
            return reach.allowScalarCancel ? 1.0 : 0.0;

        case lddecode::CombReachUse::FieldScalarSupport:
            return reach.allowScalarSignCompare ? 1.0 : 0.0;

        case lddecode::CombReachUse::ScalarSignCompare:
            return reach.allowScalarSignCompare ? 1.0 : 0.0;

        case lddecode::CombReachUse::ScalarMagnitudeCompare:
            return reach.allowScalarMagnitudeCompare ? 1.0 : 0.0;

        case lddecode::CombReachUse::IQCompare:
            return reach.allowIQCompare ? 1.0 : 0.0;

        case lddecode::CombReachUse::IQAverage:
            return reach.allowIQAverage ? 1.0 : 0.0;

        case lddecode::CombReachUse::IQCancel:
            return reach.allowIQCancel ? 1.0 : 0.0;
        }

        return 0.0;
    };

    // carrierGrammarSignedPhaseRelation uses (h + phase0 + flip) & 3 for both
    // center and target with the same h, so h cancels — the reply is identical
    // for every pixel on this line pair. Hoisted once per call.
    auto fillPair = [&](const std::vector<CombTapScalar> &nbr,
                        int targetLine,
                        bool haveNbr,
                        std::vector<CombTapPair> &dst,
                        const lddecode::CombReachSourceFrame &reachSource,
                        lddecode::CombReachUse reachUse) {
        if (!haveNbr) {
            for (int rel = 0; rel < width; ++rel)
                dst[rel] = CombTapPair();
            return;
        }

        const lddecode::CombReachReply lineReach = combReachIndex.query(
            {lineNumber, targetLine, left, left, reachUse, reachSource});
        const double lineReachLegalGate =
            legalGateForReachUse(lineReach, reachUse);

        const bool needScalarWeight =
            (reachUse != lddecode::CombReachUse::IQCancel);

        if (!needScalarWeight) {
            for (int rel = 0; rel < width; ++rel) {
                CombTapPair &p = dst[rel];
                p = CombTapPair();
                p.reachLegalGate = lineReachLegalGate;
            }
            return;
        }

        for (int rel = 0; rel < width; ++rel) {
            CombTapPair &p = dst[rel];
            p = CombTapPair();
            p.reachLegalGate = lineReachLegalGate;

            const CombTapScalar &c = tapLine.tap0[rel];
            const CombTapScalar &n = nbr[rel];
            p.diffIRE = std::fabs(c.comp - n.comp) * invI;
            p.kScore = combKMetric(c.comp, c.symMag, n.comp, n.symMag);
            p.weight = (kRange > 1e-9) ? (1.0 - p.kScore * invK) : 1.0;
            p.weight = std::clamp(p.weight, 0.0, 1.0);
        }
    };

    {
        if (wantFrame) {
            fillPair(tapLine.tapU1,
                     tapLine.lnU1,
                     tapLine.haveU1,
                     tapLine.pairU1,
                     iqSource,
                     lddecode::CombReachUse::IQCancel);

            fillPair(tapLine.tapD1,
                     tapLine.lnD1,
                     tapLine.haveD1,
                     tapLine.pairD1,
                     iqSource,
                     lddecode::CombReachUse::IQCancel);
        }
    }

    {
        if (wantContour) {
            fillPair(tapLine.tapU2,
                     tapLine.lnU2,
                     tapLine.haveU2,
                     tapLine.pairU2,
                     scalarSource,
                     lddecode::CombReachUse::FieldScalarSupport);

            fillPair(tapLine.tapD2,
                     tapLine.lnD2,
                     tapLine.haveD2,
                     tapLine.pairD2,
                     scalarSource,
                     lddecode::CombReachUse::FieldScalarSupport);
        }
    }

    if (wantContour) {
        std::fill(tapLine.intrafieldRegionReach.begin(),
                  tapLine.intrafieldRegionReach.end(),
                  CombContentReach::IntrafieldRegionReach{});
        std::fill(tapLine.regionUp4.begin(),
                  tapLine.regionUp4.end(),
                  CombContentReach::RegionRelation::Unknown);
        std::fill(tapLine.regionDown4.begin(),
                  tapLine.regionDown4.end(),
                  CombContentReach::RegionRelation::Unknown);

        const size_t iqCount =
            static_cast<size_t>(demodLines) * static_cast<size_t>(demodWidth);
        const bool haveSignedIQ =
            configuration.phaseCompensation &&
            demodWidth >= width &&
            tapLine.ln0 >= 0 && tapLine.ln0 < demodLines &&
            tapLine.lnU2 >= 0 && tapLine.lnU2 < demodLines &&
            tapLine.lnD2 >= 0 && tapLine.lnD2 < demodLines &&
            locked1DTI4fsc_flat.size() >= iqCount &&
            locked1DTQ4fsc_flat.size() >= iqCount &&
            carrierAnalysis_flat.size() >= iqCount;

        if (haveSignedIQ) {
            // ±2 grammar reach (hoisted per line).
            const lddecode::CombReachReply upReach = combReachIndex.query(
                {lineNumber, tapLine.lnU2, left, left,
                 lddecode::CombReachUse::IQCompare, iqSource});
            const lddecode::CombReachReply downReach = combReachIndex.query(
                {lineNumber, tapLine.lnD2, left, left,
                 lddecode::CombReachUse::IQCompare, iqSource});

            // Balanced 7-tap horizontal aggregate, memoised per line.  Even
            // offsets carry one carrier axis and odd offsets the other, so the
            // 0.5 end weights equalize the axis sums (3:3) and keep the vector
            // phase-flat.  The narrow 3-tap vector was noise-limited in
            // low-saturation skin (hue sigma ~25 deg), so the verdicts
            // flickered at pixel pitch and one-sided combing toggled column to
            // column — the beaded fringe on garment edges.  Each line's
            // smoothed row is computed once (ensureSmoothedLockedRow) and
            // reused by every center that references it as a ±2/±4 partner.
            ensureSmoothedLockedRow(tapLine.ln0);
            ensureSmoothedLockedRow(tapLine.lnU2);
            ensureSmoothedLockedRow(tapLine.lnD2);
            const float *sI0  = smoothedLockedTI_line(tapLine.ln0);
            const float *sQ0  = smoothedLockedTQ_line(tapLine.ln0);
            const float *sIUp = smoothedLockedTI_line(tapLine.lnU2);
            const float *sQUp = smoothedLockedTQ_line(tapLine.lnU2);
            const float *sIDn = smoothedLockedTI_line(tapLine.lnD2);
            const float *sQDn = smoothedLockedTQ_line(tapLine.lnD2);
            const auto *analysis0 = carrierAnalysis_line(tapLine.ln0);
            const auto *analysisUp = carrierAnalysis_line(tapLine.lnU2);
            const auto *analysisDn = carrierAnalysis_line(tapLine.lnD2);
            auto trustAt = [](const lddecode::CarrierAnalysisRecord *row,
                              int rel) {
                return row
                    ? lddecode::carrierTrust(row[rel].carrierConformance,
                                             row[rel].conformanceUsableAxisFraction)
                    : 0.5;
            };

            // ±4 grammar reach and smoothed rows for contour-influence gating.
            const bool have4IQ =
                tapLine.lnU4 >= 0 && tapLine.lnU4 < demodLines &&
                tapLine.lnD4 >= 0 && tapLine.lnD4 < demodLines;
            lddecode::CombReachReply up4Reach, dn4Reach;
            bool haveUp4 = false, haveDn4 = false;
            const float *sIUp4 = nullptr, *sQUp4 = nullptr;
            const float *sIDn4 = nullptr, *sQDn4 = nullptr;
            const lddecode::CarrierAnalysisRecord *analysisUp4 = nullptr;
            const lddecode::CarrierAnalysisRecord *analysisDn4 = nullptr;
            if (have4IQ && tapLine.haveU4) {
                up4Reach = combReachIndex.query(
                    {lineNumber, tapLine.lnU4, left, left,
                     lddecode::CombReachUse::IQCompare, iqSource});
                if (up4Reach.allowIQCompare) {
                    ensureSmoothedLockedRow(tapLine.lnU4);
                    sIUp4 = smoothedLockedTI_line(tapLine.lnU4);
                    sQUp4 = smoothedLockedTQ_line(tapLine.lnU4);
                    analysisUp4 = carrierAnalysis_line(tapLine.lnU4);
                    haveUp4 = true;
                }
            }
            if (have4IQ && tapLine.haveD4) {
                dn4Reach = combReachIndex.query(
                    {lineNumber, tapLine.lnD4, left, left,
                     lddecode::CombReachUse::IQCompare, iqSource});
                if (dn4Reach.allowIQCompare) {
                    ensureSmoothedLockedRow(tapLine.lnD4);
                    sIDn4 = smoothedLockedTI_line(tapLine.lnD4);
                    sQDn4 = smoothedLockedTQ_line(tapLine.lnD4);
                    analysisDn4 = carrierAnalysis_line(tapLine.lnD4);
                    haveDn4 = true;
                }
            }
            const bool want4Region = (haveUp4 || haveDn4);

            for (int rel = 0; rel < width; ++rel) {
                const std::complex<double> z0(sI0[rel], sQ0[rel]);

                tapLine.intrafieldRegionReach[rel] =
                    CombContentReach::evaluateIntrafieldRegionReach(
                        z0,
                        std::complex<double>(sIUp[rel], sQUp[rel]),
                        std::complex<double>(sIDn[rel], sQDn[rel]),
                        upReach.carrierRelation,
                        downReach.carrierRelation,
                        upReach.allowIQCompare,
                        downReach.allowIQCompare,
                        trustAt(analysis0, rel),
                        trustAt(analysisUp, rel),
                        trustAt(analysisDn, rel),
                        invI,
                        5.0,
                        // Sharp raw ±2 scalar facts: the first-pass AlienCancel
                        // decision (replaces the former Field B weight revive).
                        (rel < (int)tapLine.pairU2.size())
                            ? tapLine.pairU2[rel].diffIRE : -1.0,
                        (rel < (int)tapLine.pairD2.size())
                            ? tapLine.pairD2[rel].diffIRE : -1.0,
                        (rel < (int)tapLine.centerEnvelope.size())
                            ? tapLine.centerEnvelope[rel] * invI : 0.0);

                if (want4Region) {
                    const auto region4 =
                        CombContentReach::evaluateIntrafieldRegionReach(
                            z0,
                            haveUp4
                                ? std::complex<double>(sIUp4[rel], sQUp4[rel])
                                : std::complex<double>(0.0, 0.0),
                            haveDn4
                                ? std::complex<double>(sIDn4[rel], sQDn4[rel])
                                : std::complex<double>(0.0, 0.0),
                            up4Reach.carrierRelation,
                            dn4Reach.carrierRelation,
                            haveUp4,
                            haveDn4,
                            trustAt(analysis0, rel),
                            trustAt(analysisUp4, rel),
                            trustAt(analysisDn4, rel),
                            invI,
                            5.0);
                    tapLine.regionUp4[rel] = region4.up;
                    tapLine.regionDown4[rel] = region4.down;
                }
            }
            CombContentReach::markIntrafieldChromaBoundaryBand(
                tapLine.intrafieldRegionReach,
                4);
        }
    }

    if (wantFieldB || wantFrame) {
        // Carrier-free lateral luma delta.
        //
        // Composite level is never a luma witness in high color.  `comp` is a
        // CARRIER estimate in both modes -- split1D's +/-2 notch chroma in
        // bucket, the locked 1D carrier scalar under phase compensation -- so
        // differencing `comp` reads a chroma-AMPLITUDE transition as a luma
        // edge.  That is maximal at exactly the saturated garment boundaries
        // where the comb is needed most, and every hEdge consumer (Field A
        // cede, Field B leg admission, Frame, FVF) then cedes to 1D,
        // which renders back what the comb was cancelling.
        //
        // The dual of that fact is the fix: `raw - comp` is a LUMA estimate in
        // both modes, and both tap fields are already populated, so the
        // carrier-free reading costs nothing extra and needs no decomposition
        // pass on the bucket fast path.  Every rung below differences luma;
        // there is deliberately no helper left in this file that differences
        // composite level.
        //
        // Rung order is by aperture quality, not by mode: the locked
        // decomposition's dedicated vertical-contrast row first, its smooth
        // luma row next, then the mode's own notch luma.  hd0/luma0 are null
        // whenever the locked cache is absent, so the bucket path falls
        // through to the notch without a mode branch.  All three rungs sample
        // +/-2, so the shared hEdge constants mean one thing in every mode.
        const float *hd0 = lockedLumaCacheValid && demodWidth >= width
            ? lockedLumaHDeltaIRE_line(tapLine.ln0)
            : nullptr;
        auto notchLuma = [&](int rel)->double {
            const int r = std::clamp(rel, 0, width - 1);
            return tapLine.tap0[r].raw - tapLine.tap0[r].comp;
        };
        for (int rel = 0; rel < width; ++rel) {
            if (hd0 && width >= 5) {
                tapLine.hLumaDeltaIRE[rel] = hd0[rel];
            } else if (luma0 && width >= 5) {
                const int rm = std::clamp(rel - 2, 0, width - 1);
                const int rp = std::clamp(rel + 2, 0, width - 1);
                tapLine.hLumaDeltaIRE[rel] =
                    std::fabs(luma0[rp] - luma0[rm]) * invI;
            } else {
                // The +/-2 aperture is REQUIRED here, not chosen to match the
                // locked rows above.  The notch cancels a stationary carrier
                // exactly, but where the chroma envelope has curvature (i.e. at
                // a transition) it leaves a residual
                //     L[h] = -0.25 * cos(theta_h) * curvature(A)[h]
                // which alternates at carrier rate.  The differencing aperture
                // then decides whether that residual cancels or compounds:
                //   +/-1: cos(theta_h+1) = -sin, cos(theta_h-1) = +sin
                //         -> opposite signs, the residuals ADD, and the result
                //            swings with sin(theta) -> carrier-rate flicker in
                //            hEdge, localised to exactly the chroma
                //            transitions.  Neighbouring samples then land on
                //            opposite sides of the cede thresholds: some cede
                //            (1D line-alternation along vertical garment
                //            edges) while others comb across the boundary
                //            (zipper into the adjacent colour).  Both failure
                //            directions at once is the signature.
                //   +/-2: cos(theta_h+2) = cos(theta_h-2) = -cos
                //         -> equal, the residuals SUBTRACT to a third
                //            difference of the envelope.
                // Do not "simplify" this back to a tighter aperture.
                tapLine.hLumaDeltaIRE[rel] =
                    std::fabs(notchLuma(rel + 2) - notchLuma(rel - 2)) * invI;
            }
        }
    }

    // There is deliberately no amplitude/compactness-based "chroma shape"
    // exception here.  High-frequency energy defaults to luma.  Physical
    // appearance (large envelope, narrow span, isolation, proximity to a luma
    // edge) may limit a reach that already has carrier authority, but it may
    // never register that energy as carrier or grant an escape from the comb.
    // Carrier privileges come from the grammar/conformance tables and their
    // named consumers.  The retired irrationalChroma path violated this rule
    // by promoting exactly the fine luma structures the decoder must protect.

    {
    if (wantContour) {
        // Hoist tunables and data() pointers out of the hot loop.  Reading 11
        // vectors per pixel via std::vector<T>::operator[] generates redundant
        // address math; raw pointers + cached tunables let the loop become a
        // tight straight-line block.
        const double soft = T.FIELD_CONTOUR_SOFT_IRE;
        const double hard = T.FIELD_CONTOUR_HARD_IRE;
        const double simStart = T.FIELD_CONTOUR_SIM_START;
        const double simFull  = T.FIELD_CONTOUR_SIM_FULL;
        const double farInf   = T.FIELD_CONTOUR_FAR_INFLUENCE;
        const bool kValid = (kRange > 1e-9);
        const bool hasLuma = (luma0 && lumaU2 && lumaD2 && lumaU4 && lumaD4);
        const bool haveU2 = tapLine.haveU2;
        const bool haveD2 = tapLine.haveD2;
        const bool haveU4 = tapLine.haveU4;
        const bool haveD4 = tapLine.haveD4;

        const CombTapScalar *t0  = tapLine.tap0.data();
        const CombTapScalar *tU2 = tapLine.tapU2.data();
        const CombTapScalar *tD2 = tapLine.tapD2.data();
        const CombTapScalar *tU4 = tapLine.tapU4.data();
        const CombTapScalar *tD4 = tapLine.tapD4.data();
        const CombTapPair   *pU2 = tapLine.pairU2.data();
        const CombTapPair   *pD2 = tapLine.pairD2.data();
        CombTapContour      *outContour = tapLine.contour.data();
        CombContentReach::MovingCoarseContour *outMCC = tapLine.movingCoarseContour.data();
        double *outCoarse0 = tapLine.coarse0IRE.data();
        double *outCoarseU2 = tapLine.coarseU2IRE.data();
        double *outCoarseD2 = tapLine.coarseD2IRE.data();

        // ---- Coarse luma rows -------------------------------------------
        // Locked mode publishes a dedicated carrier-free decomposition row.
        // Bucket mode's equivalent is the notch, raw - comp, laterally
        // 4-mean'd -- notch IS bucket's coarse.
        //
        // The mean is not smoothing for its own sake.  The notch cancels a
        // STATIONARY carrier exactly, but where the chroma envelope has
        // curvature (a transition) it leaves
        //     -0.25 * cos(theta_h) * curvature(envelope)[h]
        // which alternates at carrier rate.  The +/-2 same-field legs these
        // rows get differenced against are ANTI-PHASE, so a bare per-sample
        // notch luma would make those residuals ADD across the vertical
        // difference -- the same trap the lateral aperture falls into at +/-1.
        // Four consecutive samples span a full carrier cycle, so the mean
        // cancels the cos(theta) factor and leaves a genuine luma coarse.
        auto notchLumaAt = [&](const CombTapScalar *tap, int i)->double {
            const int c = std::clamp(i, 0, width - 1);
            return tap[c].raw - tap[c].comp;
        };
        auto fillNotchCoarse = [&](const CombTapScalar *tap, double *out) {
            // One carrier cycle with integer centroid rel.  The former
            // [rel-1, rel+2] box was centred at rel+0.5, so locked luma and
            // fallback notch luma described different horizontal positions.
            for (int rel = 0; rel < width; ++rel) {
                out[rel] = centeredCarrierCycle4Mean(
                    notchLumaAt(tap, rel - 2),
                    notchLumaAt(tap, rel - 1),
                    notchLumaAt(tap, rel),
                    notchLumaAt(tap, rel + 1),
                    notchLumaAt(tap, rel + 2)) * invI;
            }
        };
        auto fillLockedCoarse = [&](const double *luma, double *out) {
            for (int rel = 0; rel < width; ++rel)
                out[rel] = luma[rel] * invI;
        };

        // The centre tap is always built, so centre luma is always available.
        if (luma0) fillLockedCoarse(luma0, outCoarse0);
        else       fillNotchCoarse(t0, outCoarse0);

        // A neighbour needs its tap to actually exist: fillTap zero-fills an
        // absent row, and a zero "luma" differenced against a real centre would
        // read as a huge false vertical break.  Absent neighbour luma falls
        // back to the centre (delta 0) and is reported through coarseLumaValid.
        const bool coarseU2Real = (lumaU2 != nullptr) || haveU2;
        const bool coarseD2Real = (lumaD2 != nullptr) || haveD2;

        if (lumaU2)           fillLockedCoarse(lumaU2, outCoarseU2);
        else if (coarseU2Real) fillNotchCoarse(tU2, outCoarseU2);
        else std::copy(outCoarse0, outCoarse0 + width, outCoarseU2);

        if (lumaD2)           fillLockedCoarse(lumaD2, outCoarseD2);
        else if (coarseD2Real) fillNotchCoarse(tD2, outCoarseD2);
        else std::copy(outCoarse0, outCoarse0 + width, outCoarseD2);

        tapLine.coarseLumaValid = coarseU2Real && coarseD2Real;

        for (int rel = 0; rel < width; ++rel) {
            const CombTapScalar &sC  = t0[rel];
            const CombTapScalar &sU2 = tU2[rel];
            const CombTapScalar &sD2 = tD2[rel];
            const CombTapScalar &sU4 = tU4[rel];
            const CombTapScalar &sD4 = tD4[rel];

            const double aC  = std::fabs(sC.comp);
            const double aU2 = std::fabs(sU2.comp);
            const double aD2 = std::fabs(sD2.comp);
            const double aU4 = std::fabs(sU4.comp);
            const double aD4 = std::fabs(sD4.comp);

            CombTapContour c;
            c.curvMidIRE = std::fabs(aU2 - 2.0 * aC + aD2) * invI;
            c.midOk = combSmoothGate(c.curvMidIRE, soft, hard);

            const double u4Pred = 2.0 * aU2 - aC;
            const double d4Pred = 2.0 * aD2 - aC;

            c.upResIRE = std::fabs(aU4 - u4Pred) * invI;
            c.dnResIRE = std::fabs(aD4 - d4Pred) * invI;
            c.upSideOk = combSmoothGate(c.upResIRE, soft, hard);
            c.dnSideOk = combSmoothGate(c.dnResIRE, soft, hard);

            const double upK = combKMetric(sU2.comp, sU2.symMag, sU4.comp, sU4.symMag);
            const double dnK = combKMetric(sD2.comp, sD2.symMag, sD4.comp, sD4.symMag);

            c.upSim = (pU2[rel].weight > 0.0)
                ? std::clamp(kValid ? (1.0 - upK * invK) : 1.0, 0.0, 1.0)
                : 0.0;
            c.dnSim = (pD2[rel].weight > 0.0)
                ? std::clamp(kValid ? (1.0 - dnK * invK) : 1.0, 0.0, 1.0)
                : 0.0;

            c.upTrust = c.midOk * c.upSideOk;
            c.dnTrust = c.midOk * c.dnSideOk;

            c.upInfluence = farInf * c.upTrust *
                            combSimilarityFactor(c.upSim, simStart, simFull);
            c.dnInfluence = farInf * c.dnTrust *
                            combSimilarityFactor(c.dnSim, simStart, simFull);

            if (tapLine.regionUp4[rel] == CombContentReach::RegionRelation::DifferentRegion)
                c.upInfluence = 0.0;
            if (tapLine.regionDown4[rel] == CombContentReach::RegionRelation::DifferentRegion)
                c.dnInfluence = 0.0;

            outContour[rel] = c;

            outMCC[rel] = hasLuma
                ? CombContentReach::evaluateMovingCoarseContour(
                    luma0[rel] * invI,
                    lumaU2[rel] * invI,
                    lumaD2[rel] * invI,
                    lumaU4[rel] * invI,
                    lumaD4[rel] * invI,
                    haveU2, haveD2, haveU4, haveD4,
                    soft, hard)
                : CombContentReach::MovingCoarseContour();
        }
    }
    }

    tapLine.builtFlags = flags;
}

static inline double clampCarrierToInputCarrierLimits(
    double v,
    std::initializer_list<double> inputs,
    double fallback)
{
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    double maxAbs = 0.0;

    for (double x : inputs) {
        if (!std::isfinite(x))
            continue;
        lo = std::min(lo, x);
        hi = std::max(hi, x);
        maxAbs = std::max(maxAbs, std::fabs(x));
    }

    if (!std::isfinite(lo) || !std::isfinite(hi))
        return std::isfinite(fallback) ? fallback : 0.0;

    double out = std::isfinite(v) ? v : fallback;
    if (!std::isfinite(out))
        out = 0.0;

    out = std::clamp(out, -maxAbs, maxAbs);
    return std::clamp(out, lo, hi);
}

static inline double clampScalarTowardCenterHalf(
    double v,
    double center,
    double leg)
{
    if (!std::isfinite(v))
        return std::isfinite(center) ? center : 0.0;
    if (!std::isfinite(center) || !std::isfinite(leg))
        return v;

    const double mid = 0.5 * (center + leg);
    if (center <= leg)
        return std::clamp(v, center, mid);
    return std::clamp(v, mid, center);
}

static inline double clampCarrierToInputLumaRange(
    double carrier,
    double centerRaw,
    std::initializer_list<double> inputLuma,
    double fallbackCarrier)
{
    if (!std::isfinite(centerRaw))
        return std::isfinite(fallbackCarrier) ? fallbackCarrier : 0.0;

    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    for (double y : inputLuma) {
        if (!std::isfinite(y))
            continue;
        lo = std::min(lo, y);
        hi = std::max(hi, y);
    }

    if (!std::isfinite(lo) || !std::isfinite(hi))
        return std::isfinite(fallbackCarrier) ? fallbackCarrier : 0.0;

    double out = std::isfinite(carrier) ? carrier : fallbackCarrier;
    if (!std::isfinite(out))
        out = 0.0;

    const double yOut = std::clamp(centerRaw - out, lo, hi);
    return centerRaw - yOut;
}

static inline double clampCarrierToInputLimits(
    double carrier,
    double centerRaw,
    std::initializer_list<double> inputCarrier,
    std::initializer_list<double> inputLuma,
    double fallbackCarrier)
{
    // Carrier limits keep the comb estimate plausible in its own signal
    // domain.  Luma limits are applied last because visible failures are
    // judged after reconstructing Y = rawCenter - carrier.
    carrier = clampCarrierToInputCarrierLimits(
        carrier, inputCarrier, fallbackCarrier);
    return clampCarrierToInputLumaRange(
        carrier, centerRaw, inputLuma, fallbackCarrier);
}


// Field A - we sample 2 and 4 lines above and below, with the 4s asymmetrically
// influencing the 2s,and 2s then influencing the evaluated pixel. Strictly intra-field.

void Comb::FrameBuffer::computeFieldALine(const CombTapLine &tapLine,
                                           double *outFieldLine,
                                           double *outGate)
{
    const int width = tapLine.width;
    if (width <= 0 || !outFieldLine || (int)tapLine.tap0.size() < width) return;

    if (outGate) std::fill(outGate, outGate + width, 1.0f);

    const auto  &T   = configuration.tunables;
    const double invI = invIreScale;
    const double hEdgeThreshIRE = std::max(1.0, T.FIELD_LUMA_EDGE_THRESH_IRE);

    for (int rel = 0; rel < width; ++rel) {
        const double rawC  = tapLine.tap0[rel].raw;
        const double rawU2 = tapLine.tapU2[rel].raw;
        const double rawD2 = tapLine.tapD2[rel].raw;
        const double C    = tapLine.tap0[rel].comp;
        const double Cup2 = tapLine.tapU2[rel].comp;
        const double Cdn2 = tapLine.tapD2[rel].comp;
        const double Cup4 = tapLine.tapU4[rel].comp;
        const double Cdn4 = tapLine.tapD4[rel].comp;

        const auto &movingContour = tapLine.movingCoarseContour[rel];
        const double reachUp2 = tapLine.pairU2[rel].reachLegalGate *
                                fieldContourGate(movingContour, true);
        const double reachDn2 = tapLine.pairD2[rel].reachLegalGate *
                                fieldContourGate(movingContour, false);

        double wUp2 = tapLine.pairU2[rel].weight * reachUp2;
        double wDn2 = tapLine.pairD2[rel].weight * reachDn2;

        const CombTapContour &curve = tapLine.contour[rel];

        // Per-side coarse-luma-edge facts.  Mild mismatch can shape Field A,
        // but a real vertical context break invalidates the same-context
        // vertical comb premise.  Do not let one surviving side normalize into
        // a full-strength ordered comb artifact.
        const double chromaT =
            (rel < (int)tapLine.centerAdmittedChromaT.size())
                ? tapLine.centerAdmittedChromaT[rel]
                : 0.0;

        double eUp = 0.0;
        double eDn = 0.0;
        double verticalContextBreak = 0.0;
        bool hardVerticalBreak = false;

        // Size is not evidence -- see coarseLumaValid.  Without it an absent
        // neighbour row reads as a flat zero delta, i.e. "no vertical break",
        // which is the permissive answer.
        if (tapLine.coarseLumaValid &&
            rel < (int)tapLine.coarse0IRE.size() &&
            rel < (int)tapLine.coarseU2IRE.size() &&
            rel < (int)tapLine.coarseD2IRE.size())
        {
            const double LUMA_EDGE_LO_IRE = 6.0;
            const double LUMA_EDGE_HI_IRE = 20.0;

            const double dUpIRE = std::fabs(
                tapLine.coarse0IRE[rel] - tapLine.coarseU2IRE[rel]);
            const double dDnIRE = std::fabs(
                tapLine.coarse0IRE[rel] - tapLine.coarseD2IRE[rel]);

            eUp = std::clamp(
                (dUpIRE - LUMA_EDGE_LO_IRE) /
                (LUMA_EDGE_HI_IRE - LUMA_EDGE_LO_IRE),
                0.0,
                1.0);

            eDn = std::clamp(
                (dDnIRE - LUMA_EDGE_LO_IRE) /
                (LUMA_EDGE_HI_IRE - LUMA_EDGE_LO_IRE),
                0.0,
                1.0);

            verticalContextBreak = std::max(eUp, eDn);

            // Same starting point as Field B.  If this cedes too much, try
            // 16.0; avoid 20.0 as the first stop because the ordered-column
            // failure is already eligible before the old HI threshold.
            const double HARD_VERTICAL_BREAK_IRE = 14.0;
            hardVerticalBreak = (std::max(dUpIRE, dDnIRE) >= HARD_VERTICAL_BREAK_IRE);

            if (!hardVerticalBreak) {
                // Mild contrast remains flexible, but do not fully kill a side
                // here.  Full side loss plus sc2 normalization is the failure.
                wUp2 *= (1.0 - 0.80 * eUp);
                wDn2 *= (1.0 - 0.80 * eDn);
            }
        }

        double boundaryCede = 0.0;

        if (!hardVerticalBreak &&
            rel < (int)tapLine.pairU2.size() &&
            rel < (int)tapLine.pairD2.size() &&
            rel < (int)tapLine.hLumaDeltaIRE.size())
        {
            const double hEdge = std::clamp(
                (tapLine.hLumaDeltaIRE[rel] - 0.45 * hEdgeThreshIRE) /
                (0.55 * hEdgeThreshIRE),
                0.0,
                1.0);

            if (hEdge > 0.0) {
                const double dUp0IRE = tapLine.pairU2[rel].diffIRE;
                const double dDn0IRE = tapLine.pairD2[rel].diffIRE;
                const double scalarUpDn = std::fabs(Cup2 - Cdn2) * invI;

                const double lumaUpDn =
                    (rel < (int)tapLine.coarseU2IRE.size() &&
                     rel < (int)tapLine.coarseD2IRE.size())
                        ? std::fabs(tapLine.coarseU2IRE[rel] - tapLine.coarseD2IRE[rel])
                        : scalarUpDn;

                const double dUpDnIRE =
                    scalarUpDn * (1.0 - chromaT) + lumaUpDn * chromaT;

                const double diffGapIRE = std::fabs(dUp0IRE - dDn0IRE);
                const double bestDiffIRE = std::min(dUp0IRE, dDn0IRE);
                const double worstDiffIRE = std::max(dUp0IRE, dDn0IRE);
                const double diffRatio =
                    (worstDiffIRE > 1e-9) ? (bestDiffIRE / worstDiffIRE) : 1.0;

                const double wBest = std::min(tapLine.pairU2[rel].weight,
                                              tapLine.pairD2[rel].weight);
                const double wWorst = std::max(tapLine.pairU2[rel].weight,
                                               tapLine.pairD2[rel].weight);
                const double wRatio =
                    (wWorst > 1e-9) ? (wBest / wWorst) : 1.0;

                const double matchIRE = 3.5;
                const double betweenIRE = 6.0;
                const double edgeUdIRE = 8.0;

                if (dUpDnIRE > edgeUdIRE) {
                    const bool preferUp = (dUp0IRE <= dDn0IRE);
                    const bool hardPreferUp =
                        preferUp &&
                        ((dUp0IRE < matchIRE && dDn0IRE > betweenIRE) ||
                         (bestDiffIRE < 4.5 && diffGapIRE > 2.5 && diffRatio < 0.55) ||
                         (bestDiffIRE < 5.0 && wRatio < 0.45));
                    const bool hardPreferDn =
                        !preferUp &&
                        ((dDn0IRE < matchIRE && dUp0IRE > betweenIRE) ||
                         (bestDiffIRE < 4.5 && diffGapIRE > 2.5 && diffRatio < 0.55) ||
                         (bestDiffIRE < 5.0 && wRatio < 0.45));

                    bool allowOneSidedBoundary = false;
                    if (rel < (int)tapLine.intrafieldRegionReach.size()) {
                        const auto &region = tapLine.intrafieldRegionReach[rel];
                        allowOneSidedBoundary = region.valid &&
                            ((preferUp &&
                              region.up == CombContentReach::RegionRelation::SameRegion &&
                              region.down == CombContentReach::RegionRelation::DifferentRegion) ||
                             (!preferUp &&
                              region.down == CombContentReach::RegionRelation::SameRegion &&
                              region.up == CombContentReach::RegionRelation::DifferentRegion));
                    }

                    if ((hardPreferUp || hardPreferDn) && !allowOneSidedBoundary) {
                        wUp2 = 0.0;
                        wDn2 = 0.0;
                        boundaryCede = 1.0;
                    } else if (hardPreferUp) {
                        wDn2 = 0.0;
                        wUp2 = std::max(wUp2, 0.40 + 0.60 * hEdge);
                    } else if (hardPreferDn) {
                        wUp2 = 0.0;
                        wDn2 = std::max(wDn2, 0.40 + 0.60 * hEdge);
                    } else {
                        const double sideGapT =
                            std::clamp((diffGapIRE - 1.0) / 4.0, 0.0, 1.0);
                        const double cedeT =
                            std::clamp((bestDiffIRE - 3.0) / 6.0, 0.0, 1.0);

                        if (preferUp)
                            wDn2 *= std::max(0.0, 1.0 - 0.85 * hEdge * (0.35 + 0.65 * sideGapT));
                        else
                            wUp2 *= std::max(0.0, 1.0 - 0.85 * hEdge * (0.35 + 0.65 * sideGapT));

                        boundaryCede = std::max(boundaryCede, 0.70 * hEdge * cedeT);
                    }
                }
            }
        }

        // Windowed OR of the no-valid-partner flag: three-region cedes
        // (drop shadows) must hold a uniform height, not flicker column to
        // column, so cede outranks one-sided anywhere in the window.
        // Bit1 marks the drop-shadow (strong magnitude-asymmetry) islands
        // that qualify for the zero-chroma render.
        bool centerIsland = false;
        bool shadowIsland = false;
        if ((int)tapLine.intrafieldRegionReach.size() >= width) {
            const int lo = std::max(rel - 4, 0);
            const int hi = std::min(rel + 4, width - 1);
            for (int k = lo; k <= hi; ++k) {
                const std::uint8_t v =
                    fieldARegionCedeFlags(tapLine.intrafieldRegionReach[k]);
                if (v & FieldACedeCenter)
                    centerIsland = true;
                if (v & FieldACedeStrongAsym)
                    shadowIsland = true;
            }
        }

        double sc2 = 1.0;

        if (hardVerticalBreak || centerIsland) {
            // A hard vertical context break means Field A has no valid
            // same-context answer here.  Signed-IQ center islands carry the
            // same verdict: neither +/-2 leg belongs to the center region.
            // Both cases also block the revive path.
            wUp2 = 0.0;
            wDn2 = 0.0;
            boundaryCede = 1.0;
        } else if ((wUp2 > 0.0) || (wDn2 > 0.0)) {
            // Keep the old strong-asymmetry handling only when the hard
            // vertical context veto has not fired.  Otherwise this becomes
            // the ordered one-sided comb failure.
            if (wDn2 > 3.0 * wUp2)      wUp2 = 0.0;
            else if (wUp2 > 3.0 * wDn2) wDn2 = 0.0;

            const double denom = wUp2 + wDn2;
            if (denom > 1e-9) {
                sc2 = 2.0 / denom;
                if (sc2 < 1.0) sc2 = 1.0;
            } else {
                wUp2 = 0.0;
                wDn2 = 0.0;
            }
        } else {
            double dMag  = std::fabs(std::fabs(Cup2) - std::fabs(Cdn2));
            double sumUD = std::fabs(Cup2 + Cdn2);

            if (dMag - std::fabs(sumUD * 0.2) <= 0.0) {
                wUp2 = reachUp2;
                wDn2 = reachDn2;
                sc2 = 1.0;
            } else {
                wUp2 = 0.0;
                wDn2 = 0.0;
            }
        }

        auto refineNearWithFar = [&](double nearS, double farS, double influence)->double {
            if (influence <= 0.0) return nearS;
            if (nearS == 0.0) return nearS;
            if ((nearS > 0.0) != (farS > 0.0)) return nearS;

            const double nearMag = std::fabs(nearS);
            const double farMag  = std::fabs(farS);
            const double mag = (nearMag + influence * farMag) / (1.0 + influence);
            return std::copysign(mag, nearS);
        };

        const double Cup2Adj = refineNearWithFar(Cup2, Cup4, curve.upInfluence);
        const double Cdn2Adj = refineNearWithFar(Cdn2, Cdn4, curve.dnInfluence);

        double tc = 0.0;
        const bool combed = (wUp2 > 0.0 || wDn2 > 0.0);

        if (combed) {
            double t2  = ((C - Cup2Adj) * wUp2 * sc2);
            t2        += ((C - Cdn2Adj) * wDn2 * sc2);
            tc        += 0.25 * t2;
        } else if (shadowIsland && chromaT < 0.375) {
            // Same law as Field B: at a drop-shadow island the 1D holds
            // genuine partial chroma unless horizontal luma structure can
            // generate cross-color, so the render follows the carrier-free
            // hEdge ramp — 1D fade where flat, zero where a luma edge
            // would bead through the 1D.
            const double hEdgeT =
                (rel < (int)tapLine.hLumaDeltaIRE.size())
                    ? std::clamp(
                        (tapLine.hLumaDeltaIRE[rel] - 0.35 * hEdgeThreshIRE) /
                        (0.65 * hEdgeThreshIRE),
                        0.0,
                        1.0)
                    : 0.0;
            tc = C * (1.0 - hEdgeT);
        } else {
            tc = C;
        }

        // Zipper defense: cede on either participating vertical luma break,
        // not only the both-sides case.  min(eUp,eDn) asked "are both sides
        // bad?"  Field A's model validity needs "did either side cross a real
        // context boundary?"
        if (combed) {
            const double lumaEdgeCede = std::max(eUp, eDn);

            double bevelCede = 0.0;
            if (rel < (int)tapLine.movingCoarseContour.size() &&
                tapLine.movingCoarseContour[rel].valid &&
                rel < (int)tapLine.hLumaDeltaIRE.size())
            {
                const auto &mc = tapLine.movingCoarseContour[rel];
                const double hEdge = std::clamp(
                    (tapLine.hLumaDeltaIRE[rel] - 0.30 * hEdgeThreshIRE) /
                    (0.70 * hEdgeThreshIRE),
                    0.0,
                    1.0);

                const double sideBalance =
                    1.0 - std::fabs(
                        std::clamp(mc.upTrust, 0.0, 1.0) -
                        std::clamp(mc.downTrust, 0.0, 1.0));

                const double bevelRisk =
                    chromaT *
                    hEdge *
                    (1.0 - std::clamp(mc.straightness, 0.0, 1.0)) *
                    (0.35 + 0.65 * std::clamp(sideBalance, 0.0, 1.0));

                bevelCede =
                    std::clamp(T.FIELD_A_BEVEL_CEDE_STRENGTH * bevelRisk, 0.0, 1.0);
            }

            const double totalCede = std::max({lumaEdgeCede, bevelCede, boundaryCede});
            if (totalCede > 0.0)
                tc = tc * (1.0 - totalCede) + C * totalCede;
        }

        if (combed) {
            // Scalar comb outputs may not be brighter, darker, or larger in
            // magnitude than the raw input pixels that formed the comb.
            // Candidate/output clusters and excluded legs are not a legal range.
            const double yC = rawC - C;
            const double yU = rawU2 - Cup2;
            const double yD = rawD2 - Cdn2;
            if (wUp2 > 0.0 && wDn2 > 0.0)
                tc = clampCarrierToInputLimits(
                    tc, rawC, { C, Cup2, Cdn2 }, { yC, yU, yD }, C);
            else if (wUp2 > 0.0) {
                tc = clampCarrierToInputLimits(
                    tc, rawC, { C, Cup2 }, { yC, yU }, C);
                tc = clampScalarTowardCenterHalf(tc, C, Cup2);
            } else {
                tc = clampCarrierToInputLimits(
                    tc, rawC, { C, Cdn2 }, { yC, yD }, C);
                tc = clampScalarTowardCenterHalf(tc, C, Cdn2);
            }
        }
        
        if (!std::isfinite(tc))
            tc = C;
        
        outFieldLine[rel] = tc;
        
        double gateA = std::max(wUp2, wDn2);
        gateA = std::clamp(gateA, 0.0, 1.0);
        if (outGate) outGate[rel] = gateA;
    }

    // (FieldAStats per-line logging removed: Field A is no longer in the
    // election, and the per-line spam buried the active diagnostics.)
}

// Field B
// Simplified Field comb as a FrameBuffer member:
// - uses only 2 vertical neighbours
void Comb::FrameBuffer::computeFieldBLine(int lineNumber,
                                               double *outFieldLine,
                                               std::uint8_t *outReasonLine)
{
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int width = videoParameters.activeVideoEnd - videoParameters.activeVideoStart;

    if (width <= 0 || lineNumber < first || lineNumber >= last || !outFieldLine) {
        if (outFieldLine) std::fill(outFieldLine, outFieldLine + std::max(width, 0), 0.0);
        if (outReasonLine) std::fill(outReasonLine, outReasonLine + std::max(width, 0), FieldBReasonNone);
        return;
    }

    const CombTapLine &tapLine = ensureCombTapLine(lineNumber);
    computeFieldBLine(tapLine, outFieldLine, outReasonLine);
}


// ============================== FIELD B ====================================
// Ground-up rebuild, promoted 2026-07-27 after the beach sentinel battery.
// It replaced a policy that ceded 79-93% of pixels to 1D on program material
// and therefore rendered ~1D everywhere; this one combs by default and stands
// down only on named physical evidence.
//
// Measured premises (2026-07-26, beach s1x11 battery):
//   - The predecessor ceded 79-93% of pixels through the chroma-boundary
//     band blanket and rendered ~1D (alternation within 2-3% of line).
//   - The old adaptive comb cancels 25-45% of line-alternating chroma error
//     but never stands down (two-leg 100%), paying +40% Y-alternation at
//     boundaries and luma detail.
//   - Region verdicts at lurch-covered columns carry a leak term of
//     kappa*h (kappa ~= 0.35 measured, 0.45 conservative) and the bare-column
//     noise floor sits at 4-5 IRE, at the legacy thresholds themselves.
//
// Stance: COMB BY DEFAULT (the interiors were never the problem). A leg is
// excluded only by:
//   - grammar illegality (absolute, reachLegalGate), or
//   - a Different-region verdict whose measured differenceIRE clears the
//     evidence floor kBase PLUS the leak bound kappa*h*gate at columns under
//     a vertically corroborated lurch run, or
//   - a per-leg coarse-luma break (the Y-alternation guard), from which an
//     AlienCancel leg is exempt: raw-identical legs share the center's leak
//     and cancel it exactly.
// Legs proven different from EACH OTHER (no shared chroma to cancel) render
// one-sided with the nearer leg instead of blending a colour present on
// neither line. Full cede to 1D only when no leg survives.
//
// Vertical taps come mirror-resolved from the tap system, so the active
// top/bottom lines comb against the inward partner instead of black --
// no jagged first/last bands.
//
// RENDER GRANULARITY: per column. The band-uniform alternative (one verdict
// for a whole chroma-boundary band, which by construction cannot interleave)
// was built and measured -- it removes the 5.6-11.3 decision switches per
// band run that manufacture edge beading -- but the per-column render was
// preferred on the canonical sentinel and showed no crawl in motion. If edge
// beading ever returns, the switching rate inside a band is the instrument
// that sees it (LDCD_PROBE_CEDE), and band-uniform verdicts are the remedy;
// alternation energy is blind to that class. See git history for the pass.
void Comb::FrameBuffer::computeFieldBLine(const CombTapLine &tapLine,
                                          double *outFieldLine,
                                          std::uint8_t *outReasonLine)
{
    const int width =
        videoParameters.activeVideoEnd - videoParameters.activeVideoStart;
    const int first = videoParameters.firstActiveFrameLine;
    const int last = videoParameters.lastActiveFrameLine;
    const int lineNumber = tapLine.cacheLine;

    if (width <= 0 || !outFieldLine)
        return;

    if (lineNumber < first || lineNumber >= last ||
        static_cast<int>(tapLine.tap0.size()) < width) {
        std::fill(outFieldLine, outFieldLine + width, 0.0);
        if (outReasonLine)
            std::fill(outReasonLine, outReasonLine + width, FieldBReasonNone);
        return;
    }

    const bool haveU =
        tapLine.haveU2 &&
        static_cast<int>(tapLine.tapU2.size()) >= width &&
        static_cast<int>(tapLine.pairU2.size()) >= width;
    const bool haveD =
        tapLine.haveD2 &&
        static_cast<int>(tapLine.tapD2.size()) >= width &&
        static_cast<int>(tapLine.pairD2.size()) >= width;

    if (!haveU && !haveD) {
        for (int rel = 0; rel < width; ++rel)
            outFieldLine[rel] = tapLine.tap0[rel].comp;
        if (outReasonLine)
            std::fill(outReasonLine, outReasonLine + width, FieldBReasonCenter);
        return;
    }

    // Boundary-evidence floor and leak transfer.
    //
    // kBaseIRE sits above the region evaluator's measured bare-column
    // difference noise floor (4-5 IRE on program material) so a verdict must
    // clear the noise it is made of. kKappa is the conservative end of the
    // measured leak-transfer slope (0.26-0.46 across five zones and three
    // chroma bins, pooled ~0.35): each IRE of corroborated along-line step
    // height contributes this much spurious separation to the evaluator's
    // per-leg difference, so it must be cleared on top of the floor.
    constexpr double kBaseIRE = 6.0;
    constexpr double kKappa = 0.45;

    // Per-column leak bound contribution from vertically corroborated
    // along-line step runs: max h*gate over covering runs (means coordinate
    // v spans rel v..v+3).
    std::vector<float> hgAt(width, 0.0f);
    for (const LurchStepRun &run : corroborateLurchEdges(lineNumber)) {
        if (run.suppressed) continue;
        const double g = std::clamp(run.gate, 0.0, 1.0);
        if (g <= 0.0) continue;
        const float hg = static_cast<float>(run.stepAbsIRE * g);
        const int xa = std::max(0, run.a);
        const int xb = std::min(width - 1, run.b + 3);
        for (int x = xa; x <= xb; ++x)
            hgAt[x] = std::max(hgAt[x], hg);
    }

    const float *centerRepairStrength =
        locked1DParallaxRepairStrength_line(lineNumber);

    // Decision-mix counters under the shared probe envs ([FB2] report).
    const bool probeLine = gCedeProbe.enabled &&
        lineNumber >= gCedeProbe.l0 && lineNumber <= gCedeProbe.l1;
    if (gCedeProbe.enabled) {
        if (lineNumber < gCedeProbe.lastLine)
            gCedeProbe.flush();
        gCedeProbe.lastLine = lineNumber;
    }
    static long fb2Px, fb2TwoLeg, fb2OneLeg, fb2Cede, fb2Ctr, fb2Hold;
    static long fb2BoundaryLegs, fb2LumaZeroLegs, fb2IllegalLegs, fb2OuterSplit;
    static long fb2StandPx; static double fb2StandSumIRE, fb2StandMaxIRE;
    static int fb2LastLine = -1;
    if (gCedeProbe.enabled) {
        if (lineNumber < fb2LastLine && fb2Px > 0) {
            std::fprintf(stderr,
                "[FB2 ln=%d-%d col=%d-%d] px=%ld  two %.1f%% one %.1f%% "
                "cede %.1f%% ctr %.1f%% hold %.1f%%  legs: boundary %.1f%% "
                "lumaZero %.1f%% illegal %.1f%% outerSplit %.1f%%\n"
                "  standing-sub at ceded px: %.1f%% of px, mean %.2f max %.2f IRE\n",
                gCedeProbe.l0, std::min(gCedeProbe.l1, 9999),
                gCedeProbe.c0, std::min(gCedeProbe.c1, 9999), fb2Px,
                100.0 * fb2TwoLeg / fb2Px, 100.0 * fb2OneLeg / fb2Px,
                100.0 * fb2Cede / fb2Px, 100.0 * fb2Ctr / fb2Px,
                100.0 * fb2Hold / fb2Px,
                50.0 * fb2BoundaryLegs / fb2Px, 50.0 * fb2LumaZeroLegs / fb2Px,
                50.0 * fb2IllegalLegs / fb2Px, 100.0 * fb2OuterSplit / fb2Px,
                100.0 * fb2StandPx / fb2Px,
                fb2StandPx > 0 ? fb2StandSumIRE / fb2StandPx : 0.0,
                fb2StandMaxIRE);
            fb2Px = fb2TwoLeg = fb2OneLeg = fb2Cede = fb2Ctr = fb2Hold = 0;
            fb2BoundaryLegs = fb2LumaZeroLegs = fb2IllegalLegs = fb2OuterSplit = 0;
            fb2StandPx = 0; fb2StandSumIRE = fb2StandMaxIRE = 0.0;
        }
        fb2LastLine = lineNumber;
    }

    using RR = CombContentReach::RegionRelation;
    const CombContentReach::IntrafieldRegionReach unknownRegion;

    std::vector<std::uint8_t> probeReason, probeBand;
    if (probeLine) {
        probeReason.assign(width, 0);
        probeBand.assign(width, 0);
    }

    // ---- Multicolumn (band-uniform) verdicts -- DEFAULT ON ----------------
    // User verdict 2026-07-27 (second pass): per-column decisions inside a
    // chroma-boundary band manufacture the bikini fishboning / vertical
    // lapel error class -- neighbouring columns answering the same boundary
    // question differently render as an interleave of manufactured colours,
    // and the errors reach the election. Dealbreaker. Field B is therefore
    // the ERROR COMB: each contiguous boundary-band run takes ONE verdict
    // per leg (aggregate evidence vs the aggregate leak bound, band-wide
    // legality), clean over detailed. Aggregation also lifts the verdict
    // out of the per-column noise floor (bare-column meanD 4-5 IRE sits AT
    // the thresholds; a run average does not).
    //
    // Per-column stays reachable as LDCD_FB2_BANDUNIFORM=0: it is the
    // candidate "detail comb" policy, intended to move to a SEPARATE
    // election candidate (Field A taking the per-column baton) so detail
    // and error-freedom compete in the election, not inside one comb.
    static const bool useBandUniform = []{
        const char *s = std::getenv("LDCD_FB2_BANDUNIFORM");
        return !s || std::atoi(s) != 0;
    }();

    std::vector<std::uint8_t> inBandRun, bandUpAdmit, bandDownAdmit;
    if (useBandUniform) {
        inBandRun.assign(width, 0);
        bandUpAdmit.assign(width, 0);
        bandDownAdmit.assign(width, 0);
        int x = 0;
        while (x < width) {
            const auto &r0 =
                x < static_cast<int>(tapLine.intrafieldRegionReach.size())
                    ? tapLine.intrafieldRegionReach[x] : unknownRegion;
            if (!r0.chromaBoundaryBand) { ++x; continue; }
            int xe = x;
            while (xe < width) {
                const auto &r =
                    xe < static_cast<int>(tapLine.intrafieldRegionReach.size())
                        ? tapLine.intrafieldRegionReach[xe] : unknownRegion;
                if (!r.chromaBoundaryBand) break;
                ++xe;
            }
            double sumUp = 0.0, sumDown = 0.0, sumBound = 0.0;
            int nUp = 0, nDown = 0, n = 0;
            bool upLegalAll = true, downLegalAll = true;
            for (int i = x; i < xe; ++i) {
                const auto &r =
                    i < static_cast<int>(tapLine.intrafieldRegionReach.size())
                        ? tapLine.intrafieldRegionReach[i] : unknownRegion;
                sumBound += kBaseIRE + kKappa * hgAt[i];
                ++n;
                if (r.upDifferenceIRE > 0.0) { sumUp += r.upDifferenceIRE; ++nUp; }
                if (r.downDifferenceIRE > 0.0) { sumDown += r.downDifferenceIRE; ++nDown; }
                if (!(haveU && tapLine.pairU2[i].reachLegalGate > 0.0))
                    upLegalAll = false;
                if (!(haveD && tapLine.pairD2[i].reachLegalGate > 0.0))
                    downLegalAll = false;
            }
            const double meanBound = n > 0 ? sumBound / n : kBaseIRE;
            const bool upAdmit = upLegalAll &&
                !(nUp > 0 && sumUp / nUp >= meanBound);
            const bool downAdmit = downLegalAll &&
                !(nDown > 0 && sumDown / nDown >= meanBound);
            for (int i = x; i < xe; ++i) {
                inBandRun[i] = 1;
                bandUpAdmit[i] = upAdmit ? 1 : 0;
                bandDownAdmit[i] = downAdmit ? 1 : 0;
            }
            x = xe;
        }
    }

    for (int rel = 0; rel < width; ++rel) {
        const double rawCenter = tapLine.tap0[rel].raw;
        const double center = tapLine.tap0[rel].comp;
        const double rawUp = haveU ? tapLine.tapU2[rel].raw : 0.0;
        const double up = haveU ? tapLine.tapU2[rel].comp : 0.0;
        const double rawDown = haveD ? tapLine.tapD2[rel].raw : 0.0;
        const double down = haveD ? tapLine.tapD2[rel].comp : 0.0;

        const auto &region =
            rel < static_cast<int>(tapLine.intrafieldRegionReach.size())
                ? tapLine.intrafieldRegionReach[rel] : unknownRegion;

        const bool upLegal =
            haveU && tapLine.pairU2[rel].reachLegalGate > 0.0;
        const bool downLegal =
            haveD && tapLine.pairD2[rel].reachLegalGate > 0.0;

        const double bound = kBaseIRE + kKappa * hgAt[rel];

        // A Different verdict stands only above the evidence floor plus the
        // leak bound; below it the measured difference is explainable as the
        // 1D bandpass leak of a corroborated luma step (or texture noise),
        // and the comb proceeds.
        const bool upBoundary =
            region.up == RR::DifferentRegion &&
            region.upDifferenceIRE >= bound;
        const bool downBoundary =
            region.down == RR::DifferentRegion &&
            region.downDifferenceIRE >= bound;

        // Per-leg coarse-luma break: the Y-alternation guard. A leg whose
        // carrier-free luma departs from center carries a different bandpass
        // leak, and combing it moves that difference into Y. AlienCancel legs
        // are exempt (raw-identical content shares the leak).
        const bool haveCoarse =
            tapLine.coarseLumaValid &&
            rel < static_cast<int>(tapLine.coarse0IRE.size()) &&
            rel < static_cast<int>(tapLine.coarseU2IRE.size()) &&
            rel < static_cast<int>(tapLine.coarseD2IRE.size());
        constexpr double kLumaLo = 6.0;
        constexpr double kLumaHi = 20.0;
        constexpr double kLumaHardBreak = 14.0;
        auto lumaGate = [&](double deltaIRE, bool alien) {
            if (alien) return 1.0;
            if (deltaIRE >= kLumaHardBreak) return 0.0;
            const double t = std::clamp(
                (deltaIRE - kLumaLo) / (kLumaHi - kLumaLo), 0.0, 1.0);
            return 1.0 - 0.65 * t;
        };
        const double upCoarseDelta = haveCoarse
            ? std::fabs(tapLine.coarse0IRE[rel] - tapLine.coarseU2IRE[rel])
            : 0.0;
        const double downCoarseDelta = haveCoarse
            ? std::fabs(tapLine.coarse0IRE[rel] - tapLine.coarseD2IRE[rel])
            : 0.0;

        double wUp = (upLegal && !upBoundary)
            ? lumaGate(upCoarseDelta, region.up == RR::AlienCancel)
            : 0.0;
        double wDown = (downLegal && !downBoundary)
            ? lumaGate(downCoarseDelta, region.down == RR::AlienCancel)
            : 0.0;

        // Legs proven different from each other offer no common chroma to
        // cancel: render one-sided with the nearer leg. Magnitude-bounded so
        // a leak-manufactured outer difference cannot split a real pair.
        if (wUp > 0.0 && wDown > 0.0 &&
            region.outerComparable &&
            region.upDownHueDifferenceDeg >= 20.0 &&
            region.upDownDifferenceIRE >= bound)
        {
            if (region.upDifferenceIRE <= region.downDifferenceIRE)
                wDown = 0.0;
            else
                wUp = 0.0;
        }

        // Inside a boundary band the run's single verdict replaces every
        // per-column content decision -- a column may not hold a private
        // opinion along an edge.
        if (useBandUniform && !inBandRun.empty() && inBandRun[rel]) {
            wUp = bandUpAdmit[rel] ? 1.0 : 0.0;
            wDown = bandDownAdmit[rel] ? 1.0 : 0.0;
        }

        const bool useUp = wUp > 1e-9;
        const bool useDown = wDown > 1e-9;
        double standingSubIRE = 0.0;

        double output;
        std::uint8_t reason;
        if (!useUp && !useDown) {
            // Ceded output: the 1D carrier -- MINUS its provably-illegal
            // standing component, bounded by the leak evidence.
            //
            // Any leg difference cancels a line-invariant (standing) carrier
            // component by construction, so two-leg and one-leg outputs never
            // carry it. Only this path can hand it through. But standing
            // carrier at the anti-phase +-2 geometry is not expressible as
            // encoder chroma (it would be vertical chroma at the alternation
            // frequency itself); measured on the beach sentinel it is the
            // bandpass leak of vertical luma edges -- 38% of carrier-band
            // energy at strong image verticals vs 12% elsewhere -- and the
            // +-1 midpoint downstream is structurally blind to it, so
            // whatever passes here survives to the render.
            //
            // The naive standing estimate 0.5*(center + mean(legs)) also
            // reads a REAL vertical chroma boundary as standing (chroma that
            // fails to continue is line-variant in truth but not in this
            // scalar), so the subtraction is capped by the same measured
            // leak-transfer bound used for the verdicts: kappa*h*gate at
            // corroborated lurch columns, zero where there is no luma-step
            // evidence. A chroma boundary keeps everything the luma step
            // cannot explain.
            // LDCD_FB_NOSTAND=1: diagnostic A/B, disables the standing-leak
            // subtraction so its contribution to render texture can be judged
            // in one variable.
            static const bool noStand = []{
                const char *s = std::getenv("LDCD_FB_NOSTAND");
                return s && std::atoi(s) != 0;
            }();
            output = center;
            if (!noStand && hgAt[rel] > 0.0f && haveU && haveD) {
                const double sEst = 0.5 * (center + 0.5 * (up + down));
                const double cap = kKappa * hgAt[rel] * irescale;
                const double sSub = std::clamp(sEst, -cap, cap);
                output = center - sSub;
                standingSubIRE = sSub * invIreScale;
            }
            reason = (upBoundary || downBoundary ||
                      (useBandUniform && !inBandRun.empty() && inBandRun[rel]))
                ? FieldBReasonCede : FieldBReasonCenter;
        } else {
            const double denom = wUp + wDown;
            const double neighbor = (up * wUp + down * wDown) / denom;
            const double combed = 0.5 * (center - neighbor);

            // Physical feasibility only: the half-difference is amplitude-
            // bounded by construction; retain the reconstructed-luma bound
            // from the raw samples that participated.
            const double yCenter = rawCenter - center;
            const double yUp = rawUp - up;
            const double yDown = rawDown - down;
            output = (useUp && useDown)
                ? clampCarrierToInputLumaRange(
                    combed, rawCenter, { yCenter, yUp, yDown }, center)
                : (useUp
                    ? clampCarrierToInputLumaRange(
                        combed, rawCenter, { yCenter, yUp }, center)
                    : clampCarrierToInputLumaRange(
                        combed, rawCenter, { yCenter, yDown }, center));
            reason = (useUp && useDown)
                ? FieldBReasonBlend : FieldBReasonOneLeg;
        }

        if (!std::isfinite(output)) {
            output = center;
            reason = FieldBReasonCenter;
        }

        // Certified Pass 1.5 repairs are source authority at this sample;
        // recombining them with unrepaired legs would reinstate the rejected
        // component. Same law as the legacy path.
        const double appliedRepair = centerRepairStrength
            ? std::clamp(static_cast<double>(centerRepairStrength[rel]), 0.0, 1.0)
            : 0.0;
        if (appliedRepair > 0.0) {
            output = center;
            reason = FieldBReasonRepairHold;
        }

        outFieldLine[rel] = output;
        if (outReasonLine)
            outReasonLine[rel] = reason;

        if (probeLine) {
            probeReason[rel] = reason;
            probeBand[rel] = region.chromaBoundaryBand ? 1 : 0;
        }

        if (probeLine && rel >= gCedeProbe.c0 && rel <= gCedeProbe.c1) {
            fb2Px++;
            if (reason == FieldBReasonBlend) fb2TwoLeg++;
            else if (reason == FieldBReasonOneLeg) fb2OneLeg++;
            else if (reason == FieldBReasonCede) fb2Cede++;
            else if (reason == FieldBReasonRepairHold) fb2Hold++;
            else fb2Ctr++;
            if (upBoundary) fb2BoundaryLegs++;
            if (downBoundary) fb2BoundaryLegs++;
            if (upLegal && !upBoundary &&
                lumaGate(upCoarseDelta, region.up == RR::AlienCancel) <= 0.0)
                fb2LumaZeroLegs++;
            if (downLegal && !downBoundary &&
                lumaGate(downCoarseDelta, region.down == RR::AlienCancel) <= 0.0)
                fb2LumaZeroLegs++;
            if (!upLegal) fb2IllegalLegs++;
            if (!downLegal) fb2IllegalLegs++;
            if ((useUp != useDown) &&
                region.outerComparable &&
                region.upDownHueDifferenceDeg >= 20.0 &&
                region.upDownDifferenceIRE >= bound)
                fb2OuterSplit++;
            if (standingSubIRE != 0.0) {
                fb2StandPx++;
                fb2StandSumIRE += std::fabs(standingSubIRE);
                fb2StandMaxIRE = std::max(fb2StandMaxIRE,
                                          std::fabs(standingSubIRE));
            }
        }
    }

    if (probeLine)
        gCedeProbe.noteBandUniformity(probeReason.data(), probeBand.data(), width);
}


static inline double cmag(const std::complex<double> &z) { return boundedMag(z); }
static inline double cmag2(const std::complex<double> &z) { return z.real() * z.real() + z.imag() * z.imag(); }
static inline double dotIQ(const std::complex<double> &a, const std::complex<double> &b) { return a.real()*b.real() + a.imag()*b.imag(); }

struct ColumnPhaseAlignmentLimits {
    double minFitIRE = 0.0;
    double pMax = 0.0;
    double tanPMax = 0.0;
};

static inline std::complex<double> applyColumnPhaseAlignment(
    const std::complex<double> &center,
    const std::complex<double> &neighbor,
    double invI,
    const ColumnPhaseAlignmentLimits &limits)
{
    const double a0 = cmag(center);
    const double an = cmag(neighbor);
    if (a0 * invI < limits.minFitIRE || an * invI < limits.minFitIRE)
        return neighbor;

    const double dot = dotIQ(neighbor, center);
    const double cross = neighbor.real() * center.imag() - neighbor.imag() * center.real();

    // The rotation we want is by phase = atan2(cross, dot).  When the clamp
    // doesn't bind, cos(phase) = dot/h and sin(phase) = cross/h where
    // h = |(dot, cross)| — so one magnitude replaces atan2 + cos + sin.
    // The clamp binds iff |phase| > pMax, equivalent to
    //   dot < 0  (phase in second/third quadrant, always > pMax for pMax<90°)
    //   or  |cross| > dot * tan(pMax).
    // Only then do we fall back to atan2+cos+sin to honor the clamp.
    double c, s;
    if (dot > 0.0 && std::fabs(cross) <= dot * limits.tanPMax) {
        const double h = boundedMag(dot, cross);
        if (h <= 1e-18) return neighbor;
        c = dot / h;
        s = cross / h;
    } else {
        double phase = std::atan2(cross, dot);
        phase = std::clamp(phase, -limits.pMax, limits.pMax);
        c = std::cos(phase);
        s = std::sin(phase);
    }

    return std::complex<double>(
        c * neighbor.real() - s * neighbor.imag(),
        s * neighbor.real() + c * neighbor.imag());
}

void Comb::FrameBuffer::computeIQFrameAFromPreparedVectors(
    int line,
    const std::vector<std::complex<double>> &centerIQ,
    std::vector<std::complex<double>> &upIQ,
    std::vector<std::complex<double>> &dnIQ,
    std::vector<std::complex<double>> &outFrameIQ,
    const CombTapLine *reachTapLine)
{
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;

    if (width <= 0) {
        outFrameIQ.clear();
        return;
    }
    outFrameIQ.resize(width);

    auto clearFrameIQ = [&]() {
        std::fill(outFrameIQ.begin(), outFrameIQ.end(), std::complex<double>(0.0, 0.0));
    };
    if (line < first || line >= last) {
        clearFrameIQ();
        return;
    }
    if (line >= demodLines || demodWidth <= 0) {
        clearFrameIQ();
        return;
    }

    const auto  &T    = configuration.tunables;
    const double invI = invIreScale;

    const bool verticalAllowed = carrierFrameVerticalAllowed(line);
    const bool haveUpLine = verticalAllowed && (line - 1 >= first);
    const bool haveDnLine = verticalAllowed && (line + 1 <  last);

    // Signed correlation in [-1..1].  Caller supplies pre-computed magnitudes
    // to avoid recomputing hypot per call site.
    auto corrSignedMags = [&](const std::complex<double> &a,
                              const std::complex<double> &b,
                              double ma, double mb)->double {
        if (ma <= 1e-12 || mb <= 1e-12) return 0.0;
        return dotIQ(a, b) / (ma*mb + 1e-12);
    };

    // Soft signed contribution and its weight, computed together so the shared
    // (c, ac, w) calculation is not run twice per neighbor.  Magnitudes are
    // supplied by the caller (precomputed once per pixel).
    auto softAlignBoth = [&](const std::complex<double> &Z0,
                             const std::complex<double> &Zn,
                             double a0, double an,
                             std::complex<double> &contribOut,
                             double &weightOut)
    {
        if (a0 <= 1e-12 || an <= 1e-12) {
            contribOut = {0.0, 0.0};
            weightOut  = 0.0;
            return;
        }
        const double c  = dotIQ(Z0, Zn) / (a0*an + 1e-12); // signed corr [-1..1]
        const double ac = std::fabs(c);

        const double t0 = 0.55;
        const double t1 = 0.85;

        double w = (ac - t0) / (t1 - t0);
        w = std::clamp(w, 0.0, 1.0);

        // No floor: if correlation is below the ramp start, the neighbor
        // does not contribute.  Edge safety is reach's job; a wFloor here
        // leaks badly-correlated material through throttled-but-nonzero reach.

        const double s = (c >= 0.0) ? 1.0 : -1.0;
        contribOut = Zn * (w * s);
        weightOut  = w;
    };

    const double COMB_STRENGTH  = std::max(0.0, T.FRAME_COMB_STRENGTH);
    const double MAX_DELTA_IRE  = T.FRAME_IQ_RAW_MAX_DELTA_IRE;
    const double MIN_CHROMA_IRE = T.FRAME_CHROMA_MIN_IRE;
    const double phaseAlignMax = T.FRAME_IQ_COLUMN_PHASE_ALIGN_MAX_DEG * M_PI / 180.0;
    const ColumnPhaseAlignmentLimits phaseAlignLimits{
        std::max(2.0, 0.5 * MIN_CHROMA_IRE),
        phaseAlignMax,
        std::tan(phaseAlignMax)
    };

    // ------------------------------------------------------------
    // Build IQ vectors for center and 1 neighbors
    // ------------------------------------------------------------
    if ((int)centerIQ.size() < width || (int)upIQ.size() < width || (int)dnIQ.size() < width) {
        clearFrameIQ();
        return;
    }

    std::copy(centerIQ.begin(), centerIQ.begin() + width, outFrameIQ.begin());

    // Column-local phase alignment.
    for (int x = 0; x < width; ++x) {
        upIQ[x] = applyColumnPhaseAlignment(centerIQ[x], upIQ[x], invI, phaseAlignLimits);
        dnIQ[x] = applyColumnPhaseAlignment(centerIQ[x], dnIQ[x], invI, phaseAlignLimits);
    }

    // ------------------------------------------------------------
    // Combine (soft signed contributions + boundary-aware asymmetry)
    // ------------------------------------------------------------
    for (int x = 0; x < width; ++x) {
        const std::complex<double> Z0 = centerIQ[x];
        const double a0 = cmag(Z0);
        const double a0_ire = a0 * invI;

        if (a0_ire <= MIN_CHROMA_IRE) {
            outFrameIQ[x] = Z0;
            continue;
        }

        std::complex<double> ZUpRaw = upIQ[x];
        std::complex<double> ZDnRaw = dnIQ[x];

        const double aUp = cmag(ZUpRaw);
        const double aDn = cmag(ZDnRaw);

        // Reach contributes physical legality only. Frame A owns the method:
        // select the center-continuing side at a vertical transition, retain a
        // small contribution from the other side, then let signed correlation
        // and phase protection decide how strongly to use what remains.
        double upReach = 0.0;
        double dnReach = 0.0;
        if (reachTapLine &&
            x < (int)reachTapLine->pairU1.size() &&
            x < (int)reachTapLine->pairD1.size())
        {
            upReach = haveUpLine
                ? std::clamp(reachTapLine->pairU1[x].reachLegalGate, 0.0, 1.0)
                : 0.0;
            dnReach = haveDnLine
                ? std::clamp(reachTapLine->pairD1[x].reachLegalGate, 0.0, 1.0)
                : 0.0;
        }

        const double dUp0IRE = cmag(ZUpRaw - Z0) * invI;
        const double dDn0IRE = cmag(ZDnRaw - Z0) * invI;
        const double dUpDnIRE = cmag(ZUpRaw - ZDnRaw) * invI;
        constexpr double kTransitionPairIRE = 8.0;
        constexpr double kTransitionMatchIRE = 4.0;
        constexpr double kTransitionSuppress = 0.35;
        if (dUpDnIRE > kTransitionPairIRE) {
            if (dUp0IRE + 1.0 < dDn0IRE && dUp0IRE < kTransitionMatchIRE)
                dnReach *= kTransitionSuppress;
            else if (dDn0IRE + 1.0 < dUp0IRE && dDn0IRE < kTransitionMatchIRE)
                upReach *= kTransitionSuppress;
        }

        const bool useUp = (upReach > 0.0) && (aUp > 1e-9);
        const bool useDn = (dnReach > 0.0) && (aDn > 1e-9);

        if (!useUp && !useDn) {
            outFrameIQ[x] = Z0;
            continue;
        }

        // Per-side signed correlation, hoisted for reuse by both the phase-
        // protection block and cohGate below.
        const double corrUp = useUp
            ? std::fabs(corrSignedMags(Z0, ZUpRaw, a0, aUp)) : 0.0;
        const double corrDn = useDn
            ? std::fabs(corrSignedMags(Z0, ZDnRaw, a0, aDn)) : 0.0;

        // Phase protection: when center and one neighbor agree on phase,
        // the other neighbor must not rotate the hue.  Project the
        // disagreeing neighbor onto center's phase axis so it can still
        // contribute amplitude (luma-slope influence) but not alter hue.
        // The threshold for "agrees" is the cohGate pass point — the same
        // correlation that fully passes the adaptive-strength gate.
        {
            const double agreeThresh = std::clamp(T.FRAME_IQ_COH_PASS_CORR, 0.0, 1.0);
            const bool upAgrees = useUp && (corrUp >= agreeThresh);
            const bool dnAgrees = useDn && (corrDn >= agreeThresh);
            const double a0sq = a0 * a0;

            if (upAgrees && !dnAgrees && useDn && a0sq > 1e-18) {
                const double proj = dotIQ(ZDnRaw, Z0) / a0sq;
                ZDnRaw = Z0 * proj;
            } else if (dnAgrees && !upAgrees && useUp && a0sq > 1e-18) {
                const double proj = dotIQ(ZUpRaw, Z0) / a0sq;
                ZUpRaw = Z0 * proj;
            }
        }

        // Combine neighbors with soft signed contributions, weighted by reach
        // (no integer dilution).  softAlignBoth returns contribution and weight
        // together so the shared correlation calc isn't repeated per side; the
        // reach gate scales both, so a throttled side pulls less without any
        // separate boundary heuristic.
        std::complex<double> Zsum = Z0;
        double wsum = 1.0;

        if (useUp) {
            std::complex<double> contrib;
            double w;
            softAlignBoth(Z0, ZUpRaw, a0, aUp, contrib, w);
            w *= upReach;
            if (w > 0.0) {
                Zsum += contrib * upReach;
                wsum += w;
            }
        }
        if (useDn) {
            std::complex<double> contrib;
            double w;
            softAlignBoth(Z0, ZDnRaw, a0, aDn, contrib, w);
            w *= dnReach;
            if (w > 0.0) {
                Zsum += contrib * dnReach;
                wsum += w;
            }
        }

        std::complex<double> Zframe = Zsum / wsum;

        std::complex<double> delta = Zframe - Z0;
        const double deltaMagIRE = cmag(delta) * invI;

        if (MAX_DELTA_IRE > 0.0 && deltaMagIRE > MAX_DELTA_IRE && deltaMagIRE > 1e-9)
            delta *= (MAX_DELTA_IRE / deltaMagIRE);

        // --------------------------------------------------------
        // Adaptive comb strength: COMB_STRENGTH_LO .. COMB_STRENGTH.
        // This is Frame A's signature: signed correlation sets confidence and
        // up/down disagreement backs off the candidate at a context break.
        // --------------------------------------------------------
        const double COMB_STRENGTH_HI = COMB_STRENGTH;
        const double COMB_STRENGTH_LO = std::min(0.8, COMB_STRENGTH_HI);

        // Coherence: reuse the hoisted per-side correlations.
        const double coh = std::max(corrUp, corrDn);

        // Map coherence -> [0..1].  cohPass is the correlation required to pass
        // cohGate fully; the ramp starts a fixed 0.30 below it.
        const double cohPass  = std::clamp(T.FRAME_IQ_COH_PASS_CORR, 0.0, 1.0);
        const double cohStart = std::max(0.0, cohPass - 0.30);
        double cohGate = (cohPass > cohStart)
            ? (coh - cohStart) / (cohPass - cohStart)
            : (coh >= cohPass ? 1.0 : 0.0);
        cohGate = std::clamp(cohGate, 0.0, 1.0);

        const double disagreementStart = 5.0;
        const double disagreementFull = 18.0;
        const double disGate = 1.0 - std::clamp(
            (dUpDnIRE - disagreementStart) /
                (disagreementFull - disagreementStart),
            0.0, 1.0);
        double localStrength =
            (COMB_STRENGTH_LO +
             (COMB_STRENGTH_HI - COMB_STRENGTH_LO) * cohGate) * disGate;

        std::complex<double> Zcandidate = Z0 + (delta * localStrength);
        
        // No bound in IQ.  Frame A's failures are judged as luma, and the
        // reconstructed-luma feasibility bound is applied to its composite
        // scalar in split2D() where the remod puts it one subtraction from Y.
        // The box that used to sit here was over {Z0, ZupCand, ZdnCand}: two
        // candidates re-derived from Frame A's own strength law, so a wrong
        // strength moved the candidate and its bound together and the box
        // admitted exactly the excursions it existed to catch.
        outFrameIQ[x] = Zcandidate;
    }
}
// Frame A: adaptive interframe IQ comb fed by the Field B preclean ring.
void Comb::FrameBuffer::computeFrameALine(
    int line,
    std::vector<std::complex<double>> &outFrameIQ)
{
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;
    const bool verticalAllowed = carrierFrameVerticalAllowed(line);

	    if (width <= 0) {
	        outFrameIQ.clear();
	        return;
	    }
	    outFrameIQ.resize(width);
	    auto clearFrameIQ = [&]() {
	        std::fill(outFrameIQ.begin(), outFrameIQ.end(), std::complex<double>(0.0, 0.0));
	    };
	    if (line < first || line >= last) {
	        clearFrameIQ();
	        return;
	    }
	    if (line >= demodLines || demodWidth <= 0) {
	        clearFrameIQ();
	        return;
	    }

    // Frame A reads only the shared physical tap facts. Its transition,
    // disagreement, correlation, and phase policy lives in
    // computeIQFrameAFromPreparedVectors; Frame B cannot shape this candidate.
    // The tap line is already built with TapBuildFrame, so ensure is a cache hit.
    const CombTapLine &reachTapLine = ensureCombTapLine(line);

    auto tiLine = [&](int ln)->const float* { return demodTI4fsc_line(ln); };
    auto tqLine = [&](int ln)->const float* { return demodTQ4fsc_line(ln); };

    const float *ti0_raw  = tiLine(line);
    const float *tq0_raw  = tqLine(line);
    const float *tiUp_raw = (verticalAllowed && line - 1 >= first) ? tiLine(line - 1) : nullptr;
    const float *tqUp_raw = (verticalAllowed && line - 1 >= first) ? tqLine(line - 1) : nullptr;
    const float *tiDn_raw = (verticalAllowed && line + 1 <  last)  ? tiLine(line + 1) : nullptr;
    const float *tqDn_raw = (verticalAllowed && line + 1 <  last)  ? tqLine(line + 1) : nullptr;

    auto scalarLine = [&](int ln)->const double* {
        if (ln < first || ln >= last) return nullptr;
        return configuration.phaseCompensation
            ? locked1DSource_line(ln)
            : bucketScalar1D_line(ln);
    };

    const double *preclean0  = precleanLinePtr(line, width);
    const double *precleanUp = verticalAllowed ? precleanLinePtr(line - 1, width) : nullptr;
    const double *precleanDn = verticalAllowed ? precleanLinePtr(line + 1, width) : nullptr;

    const double *scalar0  = scalarLine(line);
    const double *scalarUp = verticalAllowed ? scalarLine(line - 1) : nullptr;
    const double *scalarDn = verticalAllowed ? scalarLine(line + 1) : nullptr;

    auto phaseCursor = [&](int ln) {
        return carrierGrammarSignedSampleCursor(
            configuration.phaseCompensation ? carrierGrammarLine(ln) : nullptr,
            left);
    };
    auto preclean0Cursor  = phaseCursor(line);
    auto precleanUpCursor = phaseCursor(line - 1);
    auto precleanDnCursor = phaseCursor(line + 1);
    auto scalar0Cursor    = phaseCursor(line);
    auto scalarUpCursor   = phaseCursor(line - 1);
    auto scalarDnCursor   = phaseCursor(line + 1);

    if ((int)scratch_centerIQ.size() != width) scratch_centerIQ.resize(width);
    if ((int)scratch_upIQ.size() != width) scratch_upIQ.resize(width);
    if ((int)scratch_dnIQ.size() != width) scratch_dnIQ.resize(width);
    for (int x = 0; x < width; ++x) {
        if (preclean0)
            scratch_centerIQ[x] =
                carrierGrammarDemodSignedCompositeTo4fsc(preclean0Cursor, preclean0[x]);
        else if (ti0_raw && tq0_raw)
            scratch_centerIQ[x] = std::complex<double>((double)ti0_raw[x], (double)tq0_raw[x]);
        else if (scalar0)
            scratch_centerIQ[x] =
                carrierGrammarDemodSignedCompositeTo4fsc(scalar0Cursor, scalar0[x]);
        else scratch_centerIQ[x] = std::complex<double>(0.0, 0.0);

        if (precleanUp) {
            scratch_upIQ[x] =
                carrierGrammarDemodSignedCompositeTo4fsc(precleanUpCursor, precleanUp[x]);
        } else {
            if (tiUp_raw && tqUp_raw)
                scratch_upIQ[x] = std::complex<double>((double)tiUp_raw[x], (double)tqUp_raw[x]);
            else if (scalarUp)
                scratch_upIQ[x] =
                    carrierGrammarDemodSignedCompositeTo4fsc(scalarUpCursor, scalarUp[x]);
            else
                scratch_upIQ[x] = std::complex<double>(0.0, 0.0);
        }

        if (precleanDn) {
            scratch_dnIQ[x] =
                carrierGrammarDemodSignedCompositeTo4fsc(precleanDnCursor, precleanDn[x]);
        } else {
            if (tiDn_raw && tqDn_raw)
                scratch_dnIQ[x] = std::complex<double>((double)tiDn_raw[x], (double)tqDn_raw[x]);
            else if (scalarDn)
                scratch_dnIQ[x] =
                    carrierGrammarDemodSignedCompositeTo4fsc(scalarDnCursor, scalarDn[x]);
            else
                scratch_dnIQ[x] = std::complex<double>(0.0, 0.0);
        }
    }

    // Product demodulation is sample-local and still carries its 2fSC image.
    // Marshal every Frame-A operand onto the native integer coordinate before
    // the comb sees it.  The symmetric 3-tap aperture is zero at 2fSC and has
    // centroid x; I and Q remain independent components of the complex row.
    centerCarrierProductRowInPlace(scratch_centerIQ.data(), width);
    centerCarrierProductRowInPlace(scratch_upIQ.data(), width);
    centerCarrierProductRowInPlace(scratch_dnIQ.data(), width);

    computeIQFrameAFromPreparedVectors(line, scratch_centerIQ, scratch_upIQ, scratch_dnIQ,
                                      outFrameIQ, &reachTapLine);
}

// Frame B: direct interframe IQ comb.
// Sources from the Field B preclean ring, with locked-1D IQ as optional
// diagnostic center fallback.
//
// Frame B's intended strong case is opposed-field contamination:
//
//     current:        Z0 = C + E
//     frame partners: Zp = C - E
//     neutral:        C  = 0.5 * (Z0 + Zp)
//
// Therefore the correction remains an exact midpoint projection with maximum
// pull 0.5.  The relaxation here is not overdrive; it is permission for Frame B
// to complete that midpoint projection when both ±1 frame legs agree.
void Comb::FrameBuffer::computeFrameBLine(
    int line,
    std::vector<std::complex<double>> &outFrameIQ,
    std::vector<double> &outFrameScalar)
{
    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;

    if (width <= 0) {
        outFrameIQ.clear();
        outFrameScalar.clear();
        return;
    }

    outFrameIQ.resize(width);
    outFrameScalar.resize(width);

    auto clearFrameOutputs = [&]() {
        std::fill(outFrameIQ.begin(), outFrameIQ.end(),
                  std::complex<double>(0.0, 0.0));
        std::fill(outFrameScalar.begin(), outFrameScalar.end(), 0.0);
    };

    if (line < first || line >= last) {
        clearFrameOutputs();
        return;
    }

    if (line >= demodLines || demodWidth <= 0) {
        clearFrameOutputs();
        return;
    }

    const bool verticalAllowed = carrierFrameVerticalAllowed(line);

    // Native product rows are centred together with the signed-preclean rows
    // below.  Do not start from locked1DTI4fsc/TQ4fsc here: those are already
    // the canonical centred pre-comb products, and filtering them again would
    // give the forced 1D diagnostic a different aperture.
    auto tiLine = [&](int ln)->const float* { return demodTI4fsc_line(ln); };
    auto tqLine = [&](int ln)->const float* { return demodTQ4fsc_line(ln); };

    const float *ti0_raw = tiLine(line);
    const float *tq0_raw = tqLine(line);

    const bool haveUpLine = verticalAllowed && (line - 1 >= first);
    const bool haveDnLine = verticalAllowed && (line + 1 < last);

    const float *tiUp_raw = haveUpLine ? tiLine(line - 1) : nullptr;
    const float *tqUp_raw = haveUpLine ? tqLine(line - 1) : nullptr;
    const float *tiDn_raw = haveDnLine ? tiLine(line + 1) : nullptr;
    const float *tqDn_raw = haveDnLine ? tqLine(line + 1) : nullptr;

    Q_UNUSED(tiUp_raw);
    Q_UNUSED(tqUp_raw);
    Q_UNUSED(tiDn_raw);
    Q_UNUSED(tqDn_raw);

    if (!ti0_raw || !tq0_raw) {
        clearFrameOutputs();
        return;
    }

    const CombTapLine &reachTapLine = ensureCombTapLine(line);

    const double *preclean0  = precleanLinePtr(line, width);
    const double *precleanUp = haveUpLine ? precleanLinePtr(line - 1, width) : nullptr;
    const double *precleanDn = haveDnLine ? precleanLinePtr(line + 1, width) : nullptr;

    // LDCD_FB_RAW_LEGS=1: diagnostic A/B. Feed the +-1 legs from the locked
    // 1D source instead of the Field B preclean, to test whether the measured
    // leg asymmetry (3.8 IRE, ~10x what luma curvature predicts) is
    // manufactured by the preclean's own per-column decisions rather than
    // being a property of the image.
    static const bool rawLegs = []{
        const char *s = std::getenv("LDCD_FB_RAW_LEGS");
        return s && std::atoi(s) != 0;
    }();
    if (rawLegs) {
        if (haveUpLine) {
            const double *r = locked1DSource_line(line - 1);
            if (r) precleanUp = r;
        }
        if (haveDnLine) {
            const double *r = locked1DSource_line(line + 1);
            if (r) precleanDn = r;
        }
    }

    // Raw rows for the reconstructed-luma feasibility bound at the output.
    const quint16 *rawCenterRow = rawbuffer.constData() + line * videoParameters.fieldWidth;
    const quint16 *rawUpRow = haveUpLine
        ? rawbuffer.constData() + (line - 1) * videoParameters.fieldWidth : nullptr;
    const quint16 *rawDnRow = haveDnLine
        ? rawbuffer.constData() + (line + 1) * videoParameters.fieldWidth : nullptr;

    auto phaseCursor = [&](int ln) {
        return carrierGrammarSignedSampleCursor(
            configuration.phaseCompensation ? carrierGrammarLine(ln) : nullptr,
            left);
    };

    auto phase0Cursor  = phaseCursor(line);
    auto phaseUpCursor = phaseCursor(haveUpLine ? line - 1 : line);
    auto phaseDnCursor = phaseCursor(haveDnLine ? line + 1 : line);

    // Preclean and cached locked 1D have different scalar round-trip
    // contracts.  Keep both cursors live so an explicit forced-center override
    // cannot desync the carrier phase of every pixel that follows it.
    auto signedRemodCursor = carrierGrammarSignedSampleCursor(
        configuration.phaseCompensation ? carrierGrammarLine(line) : nullptr,
        left);

    auto gridRemodCursor = lddecode::carrierGrammarCompositeRemodCursor(
        configuration.phaseCompensation ? carrierGrammarLine(line) : nullptr,
        left,
        1.0,
        lddecode::CarrierSignFrame::Grid4fsc);

    if (gFrameBProbe.enabled) {
        if (line < gFrameBProbe.lastLine)
            gFrameBProbe.flush();
        gFrameBProbe.lastLine = line;
    }

    const auto &T = configuration.tunables;

    const double combStrength =
        std::clamp(std::max(0.0, T.FRAME_B_COMB_STRENGTH), 0.0, 1.0);

    // Ordinary ambiguous-case cap remains tunable.  Verified cancellation
    // pairs get a wider local cap so the midpoint projection is not clipped.
    const double maxDeltaIRE =
        std::max(0.0, T.FRAME_B_RAW_MAX_DELTA_IRE);

    static const bool forceFrameBLocked1D = [] {
        const char *s = std::getenv("LD_FRAME_B_FORCE_LOCKED_1D");
        return s && std::atoi(s) != 0;
    }();

    // Per-pixel decision probe (diagnostic only, no output influence).
    static const int fbDiagLine = []{ const char *s = std::getenv("FRAMEB_DIAG_LINE"); return s ? std::atoi(s) : -1; }();
    static const int fbDiagC0   = []{ const char *s = std::getenv("FRAMEB_DIAG_C0");   return s ? std::atoi(s) : -1; }();
    static const int fbDiagC1   = []{ const char *s = std::getenv("FRAMEB_DIAG_C1");   return s ? std::atoi(s) : -1; }();
    const bool fbDiagThisLine = fbDiagLine >= 0 && line == fbDiagLine && fbDiagC0 >= 0;
    const int fbDiagFirst = fbDiagThisLine ? std::clamp(fbDiagC0, 0, width - 1) : 0;
    const int fbDiagLast = fbDiagThisLine
        ? std::clamp(fbDiagC1 < 0 ? fbDiagC0 : fbDiagC1, fbDiagFirst, width - 1)
        : -1;
    const float *fbDiagImp0 = fbDiagThisLine ? carrierImpurity_line(line) : nullptr;
    const float *fbDiagImpU = (fbDiagThisLine && haveUpLine) ? carrierImpurity_line(line - 1) : nullptr;
    const float *fbDiagImpD = (fbDiagThisLine && haveDnLine) ? carrierImpurity_line(line + 1) : nullptr;
    if (fbDiagThisLine) {
        std::fprintf(stderr,
            "FRAMEBDIAG header line x haveUp haveDn legalUp legalDn "
            "reachUp reachDn pairAgreeIRE dUp0IRE dDn0IRE reachAuthority "
            "pull deltaIRE z0MagIRE targetMagIRE imp0 impU1 impD1 "
            "sigma dReg aGate corrIRE midLic\n");
    }

    // Demod the center and ±1 legs to signed 4fsc IQ, aligned to center's
    // carrier frame.
    if ((int)scratch_centerIQ.size() < width) scratch_centerIQ.resize(width);
    if ((int)scratch_upIQ.size() < width) scratch_upIQ.resize(width);
    if ((int)scratch_dnIQ.size() < width) scratch_dnIQ.resize(width);

    for (int x = 0; x < width; ++x) {
        // Always consume the signed preclean cursors.  Conditional consumption
        // shifts all later samples onto the wrong carrier leg after the first
        // forced-center override.
        const std::complex<double> Z0Preclean = preclean0
            ? carrierGrammarDemodSignedCompositeTo4fsc(phase0Cursor, preclean0[x])
            : std::complex<double>(0.0, 0.0);

        scratch_centerIQ[x] = forceFrameBLocked1D
            ? std::complex<double>((double)ti0_raw[x], (double)tq0_raw[x])
            : Z0Preclean;

        scratch_upIQ[x] = precleanUp
            ? carrierGrammarDemodSignedCompositeTo4fsc(phaseUpCursor, precleanUp[x])
            : std::complex<double>(0.0, 0.0);

        scratch_dnIQ[x] = precleanDn
            ? carrierGrammarDemodSignedCompositeTo4fsc(phaseDnCursor, precleanDn[x])
            : std::complex<double>(0.0, 0.0);
    }

    // Put all three operands on the same native integer coordinate before
    // either Frame-B estimator sees them.  This cancels the carrier-product
    // image without the h-0.5 delay of a previous/current average.
    centerCarrierProductRowInPlace(scratch_centerIQ.data(), width);
    centerCarrierProductRowInPlace(scratch_upIQ.data(), width);
    centerCarrierProductRowInPlace(scratch_dnIQ.data(), width);

    // Frame B takes its reach from the shared tap service's physical legality
    // and throttles the midpoint at close luma bevels.
    //
    // There was an alien-IQ reach policy here that could only ever RAISE that
    // baseline (max'd against it).  Measured over 7.37M samples of interfield
    // material, reachLegalGate was 1.0 everywhere and the policy never once
    // exceeded it, so every one of its per-pixel evaluations was discarded.
    // Lowering the baseline instead is not open to it: Frame B's +/-1 comb
    // must run wherever a legal partner exists, so an IQ observation may not
    // veto a legal leg.  A gate that can only raise a saturated baseline has
    // no expressible effect, and it is gone.
    scratch_frameBReachUp.resize(width);
    scratch_frameBReachDown.resize(width);
    scratch_impulseExempt.resize(width);
    const double *frameLuma0 = lockedLumaCacheValid
        ? lockedLumaSmooth_line(line) : nullptr;
    const double *frameLumaU1 = frameLuma0 && haveUpLine
        ? lockedLumaSmooth_line(line - 1) : nullptr;
    const double *frameLumaD1 = frameLuma0 && haveDnLine
        ? lockedLumaSmooth_line(line + 1) : nullptr;
    const double *frameLumaU2 = frameLuma0 && line - 2 >= first
        ? lockedLumaSmooth_line(line - 2) : nullptr;
    const double *frameLumaD2 = frameLuma0 && line + 2 < last
        ? lockedLumaSmooth_line(line + 2) : nullptr;
    const bool haveCloseLuma = frameLuma0 && frameLumaU1 && frameLumaD1 &&
                               frameLumaU2 && frameLumaD2;
    const double bevelPenalty =
        std::clamp(T.FRAME_B_BEVEL_REACH_PENALTY, 0.0, 1.0);
    const double satPenalty =
        std::clamp(T.FRAME_BEVEL_SAT_PENALTY, 0.0, 1.0);

    // One producer for the existing lumaImpulseRisk channel.  Field B reads
    // this scratch below, and collectCombAttributionEvidence publishes the
    // same samples for FVF and later cross-colour consumers.
    for (int x = 0; x < width; ++x) {
        scratch_impulseExempt[x] = compactLumaExcursionEvidence(
            frameLuma0, x, width, invIreScale);
    }

    for (int x = 0; x < width; ++x) {
        const double legalUp = x < (int)reachTapLine.pairU1.size()
            ? reachTapLine.pairU1[x].reachLegalGate : 0.0;
        const double legalDown = x < (int)reachTapLine.pairD1.size()
            ? reachTapLine.pairD1[x].reachLegalGate : 0.0;
        double upGate = haveUpLine ? legalUp : 0.0;
        double downGate = haveDnLine ? legalDown : 0.0;

        const double impulseExempt = scratch_impulseExempt[x];

        if (haveCloseLuma && bevelPenalty > 0.0) {
            const auto contour = CombContentReach::evaluateMovingCoarseContour(
                frameLuma0[x] * invIreScale,
                frameLumaU1[x] * invIreScale,
                frameLumaD1[x] * invIreScale,
                frameLumaU2[x] * invIreScale,
                frameLumaD2[x] * invIreScale,
                true, true, true, true,
                T.FIELD_CONTOUR_SOFT_IRE, T.FIELD_CONTOUR_HARD_IRE);
            if (contour.valid) {
                const double chromaWeight =
                    x < (int)reachTapLine.centerAdmittedChromaT.size()
                        ? reachTapLine.centerAdmittedChromaT[x] : 0.0;
                const double curvature =
                    1.0 - std::clamp(contour.straightness, 0.0, 1.0);
                double bevelGate = std::clamp(
                    1.0 - bevelPenalty * chromaWeight * curvature, 0.0, 1.0);
                bevelGate *= std::clamp(
                    1.0 - satPenalty * chromaWeight * chromaWeight *
                          curvature * curvature,
                    0.0, 1.0);
                const double effective =
                    bevelGate + (1.0 - bevelGate) * impulseExempt;
                upGate *= effective;
                downGate *= effective;
            }
        }
        scratch_frameBReachUp[x] = std::clamp(upGate, 0.0, 1.0);
        scratch_frameBReachDown[x] = std::clamp(downGate, 0.0, 1.0);
    }

    // =====================================================================
    // Signed-subtractor prepass.
    //
    // The signed demod folds image-locked alien with OPPOSITE signs on the
    // Same- and Opposite-relation ±1 legs (carrier-locked chroma reads
    // identically from both — that is the point of signed demod).  The two
    // quadratic forms of the pair therefore split cleanly:
    //
    //   midpoint  (ZUp+ZDn)/2 : chroma + (a_up − a_dn)/2  — the aliens'
    //     image-space DIFFERENCE: zero on verticals, first-order in the
    //     diagonal slope, parity-alternating — the 2-px staircase.
    //   difference (ZUp−ZDn)/2 : ±(a_up + a_dn)/2 — the aliens' SUM ≈ the
    //     center's own alien, with only a second-order (vertical curvature)
    //     error.  σ (alienSign) unfolds the parity.
    //
    // So the pair difference is a direct signed estimator of exactly the
    // contamination Frame B exists to remove; the combine subtracts it from
    // center BEFORE the midpoint projection.  σ comes from grammar lineFlip
    // polarity — a scalar-domain fact, which is the part of alignment the
    // grammar genuinely owns.  Evidence requires exactly one Same + one
    // Opposite relation across the pair; σ = +1 iff up is the Same leg.
    //
    // Registration: a thin feature advancing ~1 px/line decorrelates across
    // the ±1 pair, and an unregistered difference subtracts a straddled
    // double-image (partial correction is the worst geometry).  Each column
    // searches d ∈ [−2,+2] along the local diagonal for the offset that
    // maximizes the windowed difference magnitude — the offset where the two
    // legs' aliens add coherently.  Non-zero d must clear an 8% margin over
    // d=0 so noise cannot steer the registration.  All gates are windowed.
    // The canonical input is already a full integer-centred IQ vector; the
    // wider 7-tap windows below provide decision stability, not registration
    // or missing-axis reconstruction.
    // =====================================================================
    const bool havePairIQ =
        !forceFrameBLocked1D && haveUpLine && haveDnLine &&
        precleanUp && precleanDn;

    bool haveSignedAlien = false;
    double alienSign = 0.0;
    if (havePairIQ && configuration.phaseCompensation) {
        const auto *g0 = carrierGrammarLine(line);
        const auto *gU = carrierGrammarLine(line - 1);
        const auto *gD = carrierGrammarLine(line + 1);
        if (g0 && gU && gD) {
            const bool upSame = (gU->lineFlip == g0->lineFlip);
            const bool dnSame = (gD->lineFlip == g0->lineFlip);
            if (upSame != dnSame) {
                haveSignedAlien = true;
                alienSign = upSame ? 1.0 : -1.0;
            }
        }
    }

    if (havePairIQ) {
        if ((int)scratch_fbPairDiff.size() < width)
            scratch_fbPairDiff.resize(width);
        if ((int)scratch_fbAlienGate.size() < width)
            scratch_fbAlienGate.resize(width);
        if ((int)scratch_fbPairAgreeWinIRE.size() < width)
            scratch_fbPairAgreeWinIRE.resize(width);
        if ((int)scratch_fbLegSymmetry.size() < width)
            scratch_fbLegSymmetry.resize(width);
        if ((int)scratch_fbReg.size() < width)
            scratch_fbReg.resize(width);

        static constexpr double kWin[7] = {0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5};
        constexpr double kWinSum = 6.0;
        constexpr double kRegMargin = 1.08;

        // The windowed sums reach at most |k| + |d| = 3 + 2 = 5 samples past
        // a column, and the previous clampX indexing clamped every access to
        // [0, width - 1] — edge replication.  Padded copies reproduce that
        // exactly, so every window below reads straight pointers (no per-tap
        // clamp, no vector bounds check), and each window keeps its tap
        // order, so all sums are bit-identical to the clamped form.
        constexpr int kPad = 5;
        const int paddedWidth = width + 2 * kPad;
        if ((int)scratch_fbPadCenter.size() < paddedWidth) {
            scratch_fbPadCenter.resize(paddedWidth);
            scratch_fbPadUp.resize(paddedWidth);
            scratch_fbPadDn.resize(paddedWidth);
        }
        auto padRow = [&](std::vector<std::complex<double>> &dst,
                          const std::vector<std::complex<double>> &src) {
            std::copy(src.begin(), src.begin() + width, dst.begin() + kPad);
            std::fill(dst.begin(), dst.begin() + kPad, src[0]);
            std::fill(dst.begin() + kPad + width,
                      dst.begin() + paddedWidth, src[width - 1]);
        };
        padRow(scratch_fbPadCenter, scratch_centerIQ);
        padRow(scratch_fbPadUp, scratch_upIQ);
        padRow(scratch_fbPadDn, scratch_dnIQ);

        // pX[j] == scratch_xIQ[clamp(j, 0, width-1)] for j in
        // [-kPad, width - 1 + kPad].
        const std::complex<double> *pC = scratch_fbPadCenter.data() + kPad;
        const std::complex<double> *pU = scratch_fbPadUp.data() + kPad;
        const std::complex<double> *pD = scratch_fbPadDn.data() + kPad;

        // Hoisted d = 0 pair-difference row: every column takes the
        // unregistered windowed pair agreement, so the per-tap subtraction
        // moves out of the window loop.  Rows carry the window overhang:
        // index j in [-3, width + 2].
        constexpr int kRowPad = 3;
        const int rowWidth = width + 2 * kRowPad;
        if ((int)scratch_fbDiff0.size() < rowWidth)
            scratch_fbDiff0.resize(rowWidth);
        std::complex<double> *f0 = scratch_fbDiff0.data() + kRowPad;
        for (int j = -kRowPad; j < width + kRowPad; ++j)
            f0[j] = pU[j] - pD[j];

        // Leg roles are line-level facts (alienSign is per line), so the
        // Same/Opposite selection and the deviation rows the registration
        // search reads at every column hoist out of the pixel loop:
        // g_s[j] = same[j + s] - center[j] for s in [-2, 2].
        const bool sameIsUp = (alienSign > 0.0);
        const std::complex<double> *pSame = sameIsUp ? pU : pD;
        const std::complex<double> *pOpp = sameIsUp ? pD : pU;
        std::complex<double> *devRows[5] = {nullptr, nullptr, nullptr,
                                            nullptr, nullptr};
        if (haveSignedAlien) {
            if ((int)scratch_fbDevRows.size() < 5 * rowWidth)
                scratch_fbDevRows.resize(5 * rowWidth);
            for (int si = 0; si < 5; ++si) {
                const int s = si - 2;
                std::complex<double> *g =
                    scratch_fbDevRows.data() + si * rowWidth + kRowPad;
                for (int j = -kRowPad; j < width + kRowPad; ++j)
                    g[j] = pSame[j + s] - pC[j];
                devRows[si] = g;
            }
        }

        for (int x = 0; x < width; ++x) {
            // Unregistered (d = 0) windowed pair agreement.  Diagnostic only:
            // this is |ZUp - ZDn|, the aliens' SUM, which is NOT the midpoint's
            // error term -- see the licence in the combine below.
            std::complex<double> S0(0.0, 0.0);
            for (int k = -3; k <= 3; ++k)
                S0 += kWin[k + 3] * f0[x + k];
            scratch_fbPairAgreeWinIRE[x] =
                (cmag(S0) / kWinSum) * invIreScale;

            if (!haveSignedAlien) {
                scratch_fbPairDiff[x] = std::complex<double>(0.0, 0.0);
                scratch_fbAlienGate[x] = 0.0;
                scratch_fbReg[x] = 0;
                // No signed evidence: no symmetry fact either.  Report
                // symmetric so the midpoint licence cannot read the absence
                // of evidence as permission.
                scratch_fbLegSymmetry[x] = 1.0;
                continue;
            }

            // Registration anchored to STRUCTURE, not difference energy:
            // find the shift s* that best aligns the Same-relation leg to
            // center (the observable local diagonal advance).  Maximizing
            // |S(d)| directly is steered by chroma texture — misregistering
            // real chroma inflates the difference — so the search would
            // wander on textured content.  The Same leg rides with center by
            // the estimator's own premise, so its best-alignment shift IS the
            // diagonal advance, and the pair difference is then taken at the
            // registration that advance implies (up aligns to center at −d,
            // down at +d, so d = −s* when the Same leg is up, +s* when down).
            // s* ≠ 0 must clear an 8% improvement margin so noise cannot
            // steer the registration off the d=0 default.
            double devMag[5];
            for (int si = 0; si < 5; ++si) {
                const std::complex<double> *g = devRows[si];
                std::complex<double> devAcc(0.0, 0.0);
                for (int k = -3; k <= 3; ++k)
                    devAcc += kWin[k + 3] * g[x + k];
                devMag[si] = cmag(devAcc) / kWinSum;
            }

            int bestSi = 2;
            double bestDev = devMag[2] / kRegMargin;
            for (int si = 0; si < 5; ++si) {
                if (si == 2) continue;
                if (devMag[si] < bestDev) {
                    bestDev = devMag[si];
                    bestSi = si;
                }
            }

            const int sStar = bestSi - 2;
            const int d = sameIsUp ? -sStar : sStar;

            // Registration and the Same/Opposite discriminator need a
            // multi-axis aperture, but the correction waveform must retain
            // the sample that was measured.  Using the seven-tap average here
            // moved energy away from narrow vertical details: carrier-like
            // peaks were under-subtracted while adjacent columns received a
            // correction belonging to their neighbours.  Read the registered
            // pair pointwise; the windowed facts above still decide whether
            // this sample is an image-locked alien before it can be applied.
            const std::complex<double> pairDiff =
                pU[x - d] - pD[x + d];
            scratch_fbPairDiff[x] = pairDiff;
            scratch_fbReg[x] = d;

            // Validity gate on the SAME/OPPOSITE deviation asymmetry, both
            // legs read at their registered positions (up aligns to center at
            // −d, down at +d).  The signed fold makes the two failure modes
            // separable: image-locked alien leaves the Same leg riding center
            // (dSame ≈ 0) while displacing the Opposite leg by twice the
            // alien (dOpp ≈ 2a) — gate → 1.  A real vertical chroma gradient
            // displaces both legs symmetrically (dSame ≈ dOpp ≈ g) — gate →
            // 0, so real chroma structure is never subtracted.  Comparing
            // dSame against the pair difference instead (the earlier form)
            // conflates these, because the pair difference carries both the
            // alien sum and the gradient.
            const int oppShift = sameIsUp ? d : -d;
            std::complex<double> oppAcc(0.0, 0.0);
            for (int k = -3; k <= 3; ++k) {
                oppAcc += kWin[k + 3] *
                    (pOpp[x + k + oppShift] - pC[x + k]);
            }
            const double dSame = devMag[bestSi];
            const double dOpp = cmag(oppAcc) / kWinSum;
            // Commit once the signature is decisive (dSame ≤ 0.4·dOpp reads
            // as pure alien — full retraction), proportional below.  A gate
            // that lingers at half strength subtracts half the alien and
            // leaves a parity-alternating residue: partial correction is the
            // worst geometry, per the round-1 lesson.
            const double ratioGate =
                (dOpp > 1e-12)
                    ? std::clamp((1.0 - dSame / dOpp) / 0.6, 0.0, 1.0)
                    : 0.0;
            // "Rides with center" is an ABSOLUTE condition, not merely
            // relative: the ratio alone still commits at horizontal content
            // boundaries where BOTH legs deviate hugely and the Opposite one
            // happens to deviate 2.5× more — subtracting the neighbours'
            // carrier content there paints negative vertical ghosts of the
            // ±1 lines onto center (measured on the bridge figures,
            // 2026-07-12).  Full retraction requires the Same leg near
            // center in IRE terms; fades out by 5 IRE of windowed deviation.
            constexpr double kSameRideFullIRE = 1.25;
            constexpr double kSameRideZeroIRE = 5.0;
            const double dSameIRE = dSame * invIreScale;
            const double rideGate = std::clamp(
                (kSameRideZeroIRE - dSameIRE) /
                    (kSameRideZeroIRE - kSameRideFullIRE),
                0.0, 1.0);

            scratch_fbAlienGate[x] = ratioGate * rideGate;

            // Publish the leg-deviation SYMMETRY as a fact for the midpoint
            // estimator downstream.  This is the same pair of measurements the
            // ratioGate above consumes, but published raw: the two estimators
            // have different failure geometry and must own their own policy,
            // so the subtractor's commit ramp is NOT reused as a licence.
            //
            //   symmetry -> 0 : legs asymmetric.  The Same leg rides centre
            //                   while the Opposite carries ~2a -- the signature
            //                   of a vertically-invariant image-locked alien
            //                   (a_up == a_dn).
            //   symmetry -> 1 : legs deviate together.  Diagonal advance or a
            //                   real vertical gradient (a_up != a_dn).
            //
            // Below the noise floor there is nothing to arbitrate; report
            // symmetric so no consumer reads absence as positive evidence.
            constexpr double kSymNoiseFloorIRE = 0.75;
            scratch_fbLegSymmetry[x] =
                (dOpp * invIreScale > kSymNoiseFloorIRE)
                    ? std::clamp(dSame / dOpp, 0.0, 1.0)
                    : 1.0;
        }
    }

    for (int x = 0; x < width; ++x) {
        const bool useLockedCenter = forceFrameBLocked1D;

        const std::complex<double> &ZUp = scratch_upIQ[x];
        const std::complex<double> &ZDn = scratch_dnIQ[x];

        const std::complex<double> &Z0 = scratch_centerIQ[x];
        const std::complex<double> &Z0Preclean = Z0;

        double upReachRaw = 0.0;
        double dnReachRaw = 0.0;

        if (x < (int)reachTapLine.pairU1.size() &&
            x < (int)reachTapLine.pairD1.size())
        {
            upReachRaw = haveUpLine ? scratch_frameBReachUp[x] : 0.0;

            dnReachRaw = haveDnLine ? scratch_frameBReachDown[x] : 0.0;
        }

        const bool haveUpSignal =
            haveUpLine && precleanUp;

        const bool haveDnSignal =
            haveDnLine && precleanDn;

        // Estimator (1): signed alien retraction.  Subtract the registered
        // pair-difference alien estimate from center before any midpoint
        // geometry. Scaled by physical legality only; Frame B's midpoint
        // reach policy models a different estimator and must not leak into
        // this signed-subtractor authority.
        std::complex<double> Zc = Z0;

        // Probe capture (assigned along the path; printed only when active).
        double diagPull = 0.0;
        double diagDeltaIRE = 0.0;
        double diagTargetMagIRE = 0.0;
        double diagReachAuthority = 0.0;
        double diagSigma = 0.0;
        double diagAlienGate = 0.0;
        double diagCorrIRE = 0.0;
        double diagMidLic = 1.0;

        if (!useLockedCenter && haveSignedAlien &&
            haveUpSignal && haveDnSignal &&
            x < (int)reachTapLine.pairU1.size() &&
            x < (int)reachTapLine.pairD1.size())
        {
            const double pairLegalGate = std::min(
                std::clamp(reachTapLine.pairU1[x].reachLegalGate, 0.0, 1.0),
                std::clamp(reachTapLine.pairD1[x].reachLegalGate, 0.0, 1.0));
            const double aGate = scratch_fbAlienGate[x];

            if (pairLegalGate > 0.0 && aGate > 0.0) {
                const std::complex<double> corr =
                    ((0.5 * alienSign) * scratch_fbPairDiff[x]) *
                    (combStrength * aGate * pairLegalGate);
                if (std::isfinite(corr.real()) && std::isfinite(corr.imag()))
                    Zc = Z0 - corr;
                diagSigma = alienSign;
                diagAlienGate = aGate;
                diagCorrIRE = cmag(corr) * invIreScale;

            }
        }

        std::complex<double> Zout = Zc;

        // LDCD_PROBE_FRAMEB census (measurement only).
        const bool fbProbeThis = gFrameBProbe.enabled &&
            line >= gFrameBProbe.l0 && line <= gFrameBProbe.l1 &&
            x >= gFrameBProbe.c0 && x <= gFrameBProbe.c1;
        double probeReachBase = 0.0, probeExempt = 0.0, probeReach = 0.0;
        double probeMidLic = 0.0, probePull = 0.0, probeDeltaIRE = 0.0;
        bool probeEngaged = false, probeCapBound = false;

        // Estimator (2): plain ±1 interfield midpoint — the grail law, one
        // combine for both regimes, now operating from the alien-retracted
        // center.  Per project_frameb_comb_must_run: when a legal partner
        // exists this MUST run — never gate to bare center via confidence
        // correlations.  The regime discrimination does not live here: the
        // reach gates decide WHERE (bevel/cross-color throttles collapse reach
        // at horizontal color edges; verticals keep full reach), the flat
        // delta cap decides HOW MUCH, and the midpoint license reserves full
        // pull for the verified center-owned-alien geometry (partners agree,
        // center differs).
        if (!useLockedCenter && (upReachRaw > 0.0 || dnReachRaw > 0.0)) {
            const bool haveUp = haveUpSignal && upReachRaw > 0.0;
            const bool haveDn = haveDnSignal && dnReachRaw > 0.0;

            double wsum = 0.0;
            std::complex<double> target(0.0, 0.0);

            if (haveUp) {
                target += ZUp * upReachRaw;
                wsum += upReachRaw;
            }

            if (haveDn) {
                target += ZDn * dnReachRaw;
                wsum += dnReachRaw;
            }

            if (wsum > 1e-12) {
                target /= wsum;

                std::complex<double> delta = target - Zc;
                const double deltaIRE = cmag(delta) * invIreScale;

                // Midpoint license: the midpoint estimator's own premise is
                // partners carrying common C ± E (agree with each other,
                // differ from center).  Pair disagreement measures the
                // premise violation directly — the antisymmetric alien the
                // midpoint would inject — so the pull is licensed only where
                // the partners agree within a quarter of the delta being
                // pulled: the same 0.25 agreement fraction as the delta-cap
                // verify band and crossColorExempt, so all three
                // verified-cancellation tests share one geometry.  The
                // canonical opposed-field case (pairAgree ≈ 0, delta large)
                // keeps license ≈ 1.  Attribution measured 2026-07-12 on the
                // bridge braces: with retraction active, the residual Y-plane
                // staircase was ENTIRELY midpoint-injected — Zc-only decoded
                // at the intrafield parity floor.
                // Diagnostic A/B only (LD_RETRACTED_ADMIT family):
                // LD_FRAMEB_MIDLIC=1 forces the midpoint license fully open so
                // a decode can attribute an artifact to license refusal in one
                // variable.
                static const bool forceMidLicense = []{
                    const char *s = std::getenv("LD_FRAMEB_MIDLIC");
                    return s && s[0] == '1';
                }();
                double midLicense = 1.0;
                if (!forceMidLicense &&
                    havePairIQ && haveUp && haveDn &&
                    x < (int)scratch_fbLegSymmetry.size())
                {
                    // The midpoint is licensed on the LEG-DEVIATION SYMMETRY,
                    // because that is the quadratic form carrying its error.
                    //
                    // From the prepass algebra at the head of this function:
                    //   midpoint  (ZUp+ZDn)/2 = chroma + (a_up - a_dn)/2
                    //   pair diff (ZUp-ZDn)/2 = +-(a_up + a_dn)/2
                    // The midpoint's error is the aliens' DIFFERENCE; the pair
                    // disagreement is their SUM.  The previous licence gated on
                    // the SUM (scratch_fbPairAgreeWinIRE), which is large in
                    // BOTH the cases it needed to separate, so it could only
                    // ever refuse:
                    //   vertical/lateral edge - alien vertically invariant,
                    //     a_up == a_dn, DIFFERENCE ~ 0, midpoint clean and
                    //     NEEDED; SUM = 2a large -> wrongly refused.  This is
                    //     Frame B's signature class (limb and garment edges),
                    //     and refusing it is what stopped Frame B cancelling
                    //     there: measured 93% of columns carrying a clear
                    //     image-locked-alien signature had licence == 0
                    //     (Sisko's arm, 2026-07-19).
                    //   diagonal advance - a_up != a_dn, DIFFERENCE non-zero,
                    //     midpoint injects the 2-px staircase; correctly
                    //     refused, but by accident of the SUM also being large.
                    //
                    // Leg symmetry answers the DIFFERENCE question directly:
                    // asymmetric legs mean the alien is vertically invariant
                    // (midpoint safe), symmetric deviation means the legs carry
                    // genuinely different content (midpoint injects).
                    //
                    // Note also that the old form normalised by deltaIRE - the
                    // very distance being pulled.  Per the blend-weight
                    // doctrine a licence is candidate CONFIDENCE and must not
                    // be a function of inter-candidate distance; the form below
                    // is a property of the measurement alone.
                    constexpr double kMidSymOpen  = 0.35; // below: alien, licence
                    constexpr double kMidSymClose = 0.70; // above: differing legs, refuse
                    const double sym = scratch_fbLegSymmetry[x];
                    midLicense = std::clamp(
                        (kMidSymClose - sym) / (kMidSymClose - kMidSymOpen),
                        0.0, 1.0);
                }
                diagMidLic = midLicense;

                double effectiveMaxDeltaIRE = maxDeltaIRE;
                if (haveUp && haveDn &&
                    maxDeltaIRE > 0.0 && deltaIRE > maxDeltaIRE)
                {
                    constexpr double kVerifyAgreeFrac = 0.25;

                    const double pairAgreeIRE =
                        cmag(ZUp - ZDn) * invIreScale;

                    const double verifyBand =
                        kVerifyAgreeFrac * deltaIRE;

                    const double t = (verifyBand > 1e-9)
                        ? std::clamp(
                              1.0 - pairAgreeIRE / verifyBand,
                              0.0,
                              1.0)
                        : 0.0;

                    effectiveMaxDeltaIRE =
                        maxDeltaIRE +
                        (deltaIRE - maxDeltaIRE) * t;
                }

                if (effectiveMaxDeltaIRE > 0.0 &&
                    deltaIRE > effectiveMaxDeltaIRE &&
                    deltaIRE > 1e-9)
                {
                    delta *= effectiveMaxDeltaIRE / deltaIRE;
                    probeCapBound = true;
                }

                // Reach selected and weighted the target above. Do not apply the
                // same authority a second time to the exact midpoint projection.
                // Preserve full midpoint authority when either interfield leg is fully
                // admitted, while allowing a common reach throttle to suppress bevel
                // zippers at horizontal colour transitions.
                // Normal Frame B authority supplied by reach policy.
                const double baseReachAuthority = std::max(
                    haveUp ? upReachRaw : 0.0,
                    haveDn ? dnReachRaw : 0.0);

                // A strong lateral luma transition is a likely source of cross-colour.
                // Do not treat it as an exemption by itself: require both precleaned
                // interfield partners to agree with one another while differing from center.
                double crossColorExempt = 0.0;

                if (haveUp && haveDn &&
                    x < static_cast<int>(reachTapLine.hLumaDeltaIRE.size()))
                {
                    const double hEdgeThreshIRE =
                        std::max(1.0, T.FRAME_LUMA_EDGE_THRESH_IRE);

                    const double hEdge = std::clamp(
                        (reachTapLine.hLumaDeltaIRE[x] -
                         0.30 * hEdgeThreshIRE) /
                        (0.70 * hEdgeThreshIRE),
                        0.0,
                        1.0);

                    const double pairAgreeIRE =
                        cmag(ZUp - ZDn) * invIreScale;

                    const double centerDisagreeIRE =
                        0.5 * (cmag(ZUp - Z0) + cmag(ZDn - Z0)) *
                        invIreScale;

                    // The partners must agree within one quarter of their average
                    // disagreement with center. This is the same geometry already used
                    // to recognize a verified cancellation pair for the delta cap.
                    const double verifyBand =
                        0.25 * centerDisagreeIRE;

                    const double verifiedCancellation =
                        (verifyBand > 1e-9)
                            ? std::clamp(
                                  1.0 - pairAgreeIRE / verifyBand,
                                  0.0,
                                  1.0)
                            : 0.0;

                    crossColorExempt =
                        hEdge * verifiedCancellation;
                }

                // Verified edge-generated cross-colour restores authority withdrawn by
                // the generic horizontal-transition/bevel throttle. It cannot create a
                // Frame B engagement where reach supplied no partner at all.
                const double reachAuthority =
                    baseReachAuthority +
                    (1.0 - baseReachAuthority) * crossColorExempt;

                const double pull = std::clamp(
                    0.5 * combStrength * reachAuthority,
                    0.0,
                    0.5);
                Zout = Zc + delta * (pull * midLicense);

                probeEngaged = true;
                probeReachBase = baseReachAuthority;
                probeExempt = crossColorExempt;
                probeReach = reachAuthority;
                probeMidLic = midLicense;
                probePull = pull;
                probeDeltaIRE = deltaIRE;

                diagPull = pull;
                diagDeltaIRE = deltaIRE;
                diagTargetMagIRE = cmag(target) * invIreScale;
                diagReachAuthority = reachAuthority;

                if (!std::isfinite(Zout.real()) ||
                    !std::isfinite(Zout.imag()))
                {
                    Zout = Zc;
                }
            }
        }

        static const bool dumpFbMap = std::getenv("LDCD_DUMP_FBMAP") != nullptr;
        if (fbProbeThis && dumpFbMap) {
            std::fprintf(stderr, "FBMAP %d %d %.3f %.1f %.2f %.3f\n",
                line, x,
                x < (int)scratch_fbLegSymmetry.size()
                    ? scratch_fbLegSymmetry[x] : 1.0,
                x < (int)reachTapLine.hLumaDeltaIRE.size()
                    ? reachTapLine.hLumaDeltaIRE[x] : 0.0,
                x < (int)scratch_fbAlienGate.size()
                    ? scratch_fbAlienGate[x] : 0.0,
                probeMidLic);
        }
        if (fbProbeThis) {
            // Blind-bandpass standing/alternating split at both comb
            // geometries. Absolute h coordinates; guarded to active lines.
            const int h = left + x;
            const int lastL = videoParameters.lastActiveFrameLine;
            const int firstL = videoParameters.firstActiveFrameLine;
            if (line - 2 >= firstL && line + 2 < lastL) {
                const double b0 = clpbuffer[0].pixel[line][h];
                const double m1 = 0.5 * (clpbuffer[0].pixel[line - 1][h] +
                                         clpbuffer[0].pixel[line + 1][h]);
                const double m2 = 0.5 * (clpbuffer[0].pixel[line - 2][h] +
                                         clpbuffer[0].pixel[line + 2][h]);
                const double hIRE =
                    x < (int)reachTapLine.hLumaDeltaIRE.size()
                        ? reachTapLine.hLumaDeltaIRE[x] : 0.0;
                gFrameBProbe.noteBp(hIRE > 14.0,
                    0.5 * std::fabs(b0 - m1) * invIreScale,
                    0.5 * std::fabs(b0 + m1) * invIreScale,
                    0.5 * std::fabs(b0 - m2) * invIreScale,
                    0.5 * std::fabs(b0 + m2) * invIreScale);
                if (hIRE > 14.0) {
                    const double *amU = lockedApertureMean_line(line - 1);
                    const double *amD = lockedApertureMean_line(line + 1);
                    const int meansCount = width - 3;
                    if (amU && amD && x >= 2 && x + 2 < meansCount) {
                        const double d2u = std::fabs(
                            amU[x - 2] - 2.0 * amU[x] + amU[x + 2]) * invIreScale;
                        const double d2d = std::fabs(
                            amD[x - 2] - 2.0 * amD[x] + amD[x + 2]) * invIreScale;
                        if (d2u + d2d > 2.0) {
                            const double pairIRE =
                                x < (int)scratch_fbPairDiff.size()
                                    ? cmag(scratch_fbPairDiff[x]) * invIreScale
                                    : 0.0;
                            gFrameBProbe.noteAsym(
                                std::fabs(d2u - d2d) / (d2u + d2d),
                                d2u + d2d, pairIRE);
                            const double aGk = x < (int)scratch_fbAlienGate.size()
                                ? scratch_fbAlienGate[x] : 0.0;
                            if (aGk > 0.5)
                                gFrameBProbe.noteKappa(d2u + d2d, pairIRE);

                            const double *amCr = lockedApertureMean_line(line);
                            if (amCr && x >= 1 && x + 1 < meansCount) {
                                const double grad =
                                    0.5 * (amCr[x + 1] - amCr[x - 1]) * invIreScale;
                                if (std::fabs(grad) > 3.0) {
                                    const double yU = amU[x] * invIreScale;
                                    const double yD = amD[x] * invIreScale;
                                    const double yC = amCr[x] * invIreScale;
                                    gFrameBProbe.noteReg(line,
                                                         (yU - yC) / grad,
                                                         (yD - yC) / grad);
                                }
                                gFrameBProbe.noteGrad(std::fabs(grad), pairIRE);
                            }
                        }
                    }
                }
            }
        }
        if (fbProbeThis) {
            const double sym = x < (int)scratch_fbLegSymmetry.size()
                ? scratch_fbLegSymmetry[x] : 1.0;
            const double aG = x < (int)scratch_fbAlienGate.size()
                ? scratch_fbAlienGate[x] : 0.0;
            const double dOppIRE = x < (int)scratch_fbPairDiff.size()
                ? cmag(scratch_fbPairDiff[x]) * invIreScale : 0.0;
            gFrameBProbe.note(sym, probeEngaged, probeReachBase, probeExempt,
                              probeReach, probeMidLic, probePull, probeDeltaIRE,
                              cmag(Zout - Zc) * invIreScale, probeCapBound,
                              aG, dOppIRE > 0.75,
                              x < (int)reachTapLine.hLumaDeltaIRE.size()
                                  ? reachTapLine.hLumaDeltaIRE[x] : 0.0);
        }

        // Graceful failure at highlights.  No magnitude clamp here: cmag(Zout)
        // is fullSignedIQ scale, not composite-scalar scale.
        if (!std::isfinite(Zout.real()) || !std::isfinite(Zout.imag())) {
            Zout = (std::isfinite(Z0Preclean.real()) &&
                    std::isfinite(Z0Preclean.imag()))
                       ? Z0Preclean
                       : std::complex<double>(0.0, 0.0);
        }

        if (fbDiagThisLine && x >= fbDiagFirst && x <= fbDiagLast) {
            const double legalUp = (x < (int)reachTapLine.pairU1.size())
                ? reachTapLine.pairU1[x].reachLegalGate : -1.0;
            const double legalDn = (x < (int)reachTapLine.pairD1.size())
                ? reachTapLine.pairD1[x].reachLegalGate : -1.0;
            const double pairAgreeIRE =
                (haveUpSignal && haveDnSignal)
                    ? cmag(ZUp - ZDn) * invIreScale : -1.0;
            std::fprintf(stderr,
                "FRAMEBDIAG line=%d x=%d haveUp=%d haveDn=%d legalUp=%.3f "
                "legalDn=%.3f reachUp=%.3f reachDn=%.3f pairAgreeIRE=%.3f "
                "dUp0IRE=%.3f dDn0IRE=%.3f "
                "reachAuthority=%.3f pull=%.3f "
                "deltaIRE=%.3f z0MagIRE=%.3f "
                "targetMagIRE=%.3f imp0=%.3f impU1=%.3f impD1=%.3f "
                "sigma=%.0f dReg=%d aGate=%.3f corrIRE=%.3f midLic=%.3f\n",
                line, x, haveUpSignal ? 1 : 0, haveDnSignal ? 1 : 0,
                legalUp, legalDn, upReachRaw, dnReachRaw,
                pairAgreeIRE,
                cmag(ZUp - Z0) * invIreScale, cmag(ZDn - Z0) * invIreScale,
                diagReachAuthority,
                diagPull, diagDeltaIRE,
                cmag(Z0) * invIreScale, diagTargetMagIRE,
                fbDiagImp0 ? fbDiagImp0[x] : -1.0f,
                fbDiagImpU ? fbDiagImpU[x] : -1.0f,
                fbDiagImpD ? fbDiagImpD[x] : -1.0f,
                diagSigma,
                havePairIQ && x < (int)scratch_fbReg.size()
                    ? scratch_fbReg[x] : 0,
                diagAlienGate, diagCorrIRE, diagMidLic);
        }

        outFrameIQ[x] = Zout;

        if (useLockedCenter) {
            outFrameScalar[x] =
                lddecode::carrierGrammarRemod4fscToComposite(
                    gridRemodCursor, Zout.real(), Zout.imag());

            lddecode::carrierGrammarAdvanceSignedSampleCursor(
                signedRemodCursor);
        } else {
            outFrameScalar[x] =
                carrierGrammarRemodSigned4fscToComposite(
                    signedRemodCursor, Zout.real(), Zout.imag());

            lddecode::carrierGrammarAdvanceRemodCursor(gridRemodCursor);
        }

        // Reconstructed-luma feasibility on the composite Frame B just made.
        //
        // Deliberately not a bound in IQ.  Frame B's signed-subtractor path
        // removes an alien present in ALL of its legs, so the correct carrier
        // routinely sits outside the legs' carrier range; clamping there was
        // measured binding on 53% of pixels, which is an under-comb, not a
        // net.  In luma the statement is true again: whatever the comb does
        // to the chroma, the luma it implies must be a luma its own legs
        // could have produced.  Unsaturated light/dark zippers are the
        // failure this is here to catch.
        if (preclean0) {
            const int h = left + x;
            const double yC = (double)rawCenterRow[h] - preclean0[x];
            const double yU = (rawUpRow && precleanUp)
                ? (double)rawUpRow[h] - precleanUp[x]
                : std::numeric_limits<double>::quiet_NaN();
            const double yD = (rawDnRow && precleanDn)
                ? (double)rawDnRow[h] - precleanDn[x]
                : std::numeric_limits<double>::quiet_NaN();
            outFrameScalar[x] = clampCarrierToInputLumaRangeShared(
                outFrameScalar[x], (double)rawCenterRow[h],
                { yC, yU, yD }, preclean0[x]);
        }
    }

    // Interfield-flip diagnostic: per-leg center-relative benefit of the
    // carrier alignment.  For each leg independently:
    //   rawError     = ||Z0 - RawLeg||     (leg without carrier sign correction)
    //   alignedError = ||Z0 - AlignedLeg|| (leg as the grammar aligned it)
    //   benefit      = (rawError - alignedError) / (rawError + alignedError + ε)
    // Positive = flip helped (carrier-locked chroma).
    // Negative = flip hurt  (image-locked structure).
    if (configuration.debugInterfieldFlip && configuration.phaseCompensation) {
        static const int flipProbeStep = [] {
            const char *s = std::getenv("FLIP_DIAG_STEP");
            return s ? std::max(1, std::atoi(s)) : 8;
        }();
        const int probeFirst = first + (last - first) / 4;
        const int probeLast  = first + (last - first) * 3 / 4;
        const bool lineIsProbe =
            line >= probeFirst && line < probeLast &&
            ((line - probeFirst) % flipProbeStep == 0);

        if (lineIsProbe && (haveUpLine || haveDnLine)) {
            const auto *grammar0 =
                carrierGrammarLine(line);
            const auto *grammarUp = haveUpLine
                ? carrierGrammarLine(line - 1) : nullptr;
            const auto *grammarDn = haveDnLine
                ? carrierGrammarLine(line + 1) : nullptr;

            const int flip0 = grammar0 ? grammar0->lineFlip : +1;
            const bool upFlipped =
                grammarUp && (grammarUp->lineFlip != flip0);
            const bool dnFlipped =
                grammarDn && (grammarDn->lineFlip != flip0);

            constexpr double eps = 0.1;

            double sumBenefitUp = 0.0, sumBenefitDn = 0.0;
            double sumMinBenefit = 0.0;
            double sumLegDisagree = 0.0;
            double sumSamePhaseDistIRE = 0.0;
            int countUp = 0, countDn = 0, countBoth = 0;
            int countUpPositive = 0, countDnPositive = 0;
            int countSamePhase = 0;
            double sumCenterMag = 0.0;

            for (int x = 0; x < width; ++x) {
                const std::complex<double> &Z0x = scratch_centerIQ[x];
                const double z0mag = cmag(Z0x);

                if (haveUpLine && precleanUp) {
                    const std::complex<double> &ZUpx = scratch_upIQ[x];
                    if (upFlipped) {
                        const std::complex<double> rawUp = -ZUpx;
                        const double rawErr = cmag(Z0x - rawUp);
                        const double alignedErr = cmag(Z0x - ZUpx);
                        const double benefit =
                            (rawErr - alignedErr) /
                            (rawErr + alignedErr + eps);
                        sumBenefitUp += benefit;
                        if (benefit > 0.0) ++countUpPositive;
                    } else {
                        sumSamePhaseDistIRE += cmag(Z0x - ZUpx) * invIreScale;
                        ++countSamePhase;
                    }
                    ++countUp;
                    sumCenterMag += z0mag;
                }

                if (haveDnLine && precleanDn) {
                    const std::complex<double> &ZDnx = scratch_dnIQ[x];
                    if (dnFlipped) {
                        const std::complex<double> rawDn = -ZDnx;
                        const double rawErr = cmag(Z0x - rawDn);
                        const double alignedErr = cmag(Z0x - ZDnx);
                        const double benefit =
                            (rawErr - alignedErr) /
                            (rawErr + alignedErr + eps);
                        sumBenefitDn += benefit;
                        if (benefit > 0.0) ++countDnPositive;
                    } else {
                        sumSamePhaseDistIRE += cmag(Z0x - ZDnx) * invIreScale;
                        ++countSamePhase;
                    }
                    ++countDn;
                    if (!haveUpLine || !precleanUp)
                        sumCenterMag += z0mag;
                }

                if (haveUpLine && precleanUp && haveDnLine && precleanDn) {
                    const std::complex<double> &ZUpx = scratch_upIQ[x];
                    const std::complex<double> &ZDnx = scratch_dnIQ[x];
                    const double upMag = cmag(ZUpx);
                    const double dnMag = cmag(ZDnx);
                    const double minMag = std::min(upMag, dnMag);
                    if (minMag > eps) {
                        const double dot =
                            (ZUpx.real() * ZDnx.real() +
                             ZUpx.imag() * ZDnx.imag()) /
                            (upMag * dnMag);
                        sumLegDisagree += (1.0 - dot);
                    }

                    if (upFlipped || dnFlipped) {
                        const std::complex<double> rawUp =
                            upFlipped ? -ZUpx : ZUpx;
                        const std::complex<double> rawDn =
                            dnFlipped ? -ZDnx : ZDnx;
                        const double rawErrUp = cmag(Z0x - rawUp);
                        const double alignedErrUp = cmag(Z0x - ZUpx);
                        const double bUp =
                            (rawErrUp - alignedErrUp) /
                            (rawErrUp + alignedErrUp + eps);
                        const double rawErrDn = cmag(Z0x - rawDn);
                        const double alignedErrDn = cmag(Z0x - ZDnx);
                        const double bDn =
                            (rawErrDn - alignedErrDn) /
                            (rawErrDn + alignedErrDn + eps);
                        sumMinBenefit += std::min(bUp, bDn);
                    }
                    ++countBoth;
                }
            }

            const int flippedLegs =
                (upFlipped ? countUp : 0) + (dnFlipped ? countDn : 0);
            const double avgBenUp = (upFlipped && countUp > 0)
                ? sumBenefitUp / countUp : 0.0;
            const double avgBenDn = (dnFlipped && countDn > 0)
                ? sumBenefitDn / countDn : 0.0;
            const double avgFlippedBenefit = flippedLegs > 0
                ? (sumBenefitUp + sumBenefitDn) / flippedLegs : 0.0;
            const double avgMin = countBoth > 0
                ? sumMinBenefit / countBoth : 0.0;
            const double avgDisagree = countBoth > 0
                ? sumLegDisagree / countBoth : 0.0;
            const double avgCenterMagIRE =
                countUp > 0
                    ? (sumCenterMag / countUp) * invIreScale
                    : 0.0;
            const double avgSamePhaseDistIRE = countSamePhase > 0
                ? sumSamePhaseDistIRE / countSamePhase : -1.0;
            const double fracUpPos = (upFlipped && countUp > 0)
                ? double(countUpPositive) / countUp : -1.0;
            const double fracDnPos = (dnFlipped && countDn > 0)
                ? double(countDnPositive) / countDn : -1.0;

            std::fprintf(stderr,
                "FLIPDIAG line=%d flipUp=%d flipDn=%d "
                "benefitUp=%.4f benefitDn=%.4f "
                "avgFlipBenefit=%.4f minBenefit=%.4f "
                "legDisagree=%.4f samePhaseDistIRE=%.2f "
                "centerMagIRE=%.2f "
                "fracUpPos=%.3f fracDnPos=%.3f "
                "nUp=%d nDn=%d nBoth=%d\n",
                line, upFlipped ? 1 : 0, dnFlipped ? 1 : 0,
                avgBenUp, avgBenDn,
                avgFlippedBenefit, avgMin,
                avgDisagree, avgSamePhaseDistIRE,
                avgCenterMagIRE,
                fracUpPos, fracDnPos,
                countUp, countDn, countBoth);
        }
    }
}

// 3D Section
// getCandidate - prescreen for 3D election
// 3D Section
// getCandidate - prescreen for 3D election
Comb::FrameBuffer::Candidate Comb::FrameBuffer::getCandidate(
    qint32 refLineNumber, qint32 refH,
    const FrameBuffer &frameBuffer, qint32 lineNumber, qint32 h,
    double adjustPenalty) const
{
    Candidate result;
    result.penalty = configuration.candidatePenaltyHardMax;
    result.sample  = 0.0;
    result.yPen    = configuration.candidatePenaltyHardMax;
    result.iqPen   = 0.0;

    const int firstLine  = videoParameters.firstActiveFrameLine;
    const int lastLine   = videoParameters.lastActiveFrameLine;
    const int left       = videoParameters.activeVideoStart;
    const int right      = videoParameters.activeVideoEnd;
    const int fieldWidth = videoParameters.fieldWidth;

    auto clampH = [&](int x) -> int {
        return std::clamp(x, left, right - 1);
    };

    // Bounds check.
    if ((unsigned)(lineNumber - firstLine) >= (unsigned)(lastLine - firstLine) ||
        (unsigned)(refLineNumber - firstLine) >= (unsigned)(lastLine - firstLine)) {
        result.penalty = 1000.0;
        result.yPen    = 1000.0;
        result.iqPen   = 0.0;
        return result;
    }

    // Cross-frame ScalarSignCompare on the mode's actual 1D scalar.  Both the
    // bucket scalar and the locked 1D scalar are PhasePreservedCarrier, so
    // grammar legality answers identically for either; the historical Bucket
    // mislabel (a workaround for the retired common-phase classification) is
    // no longer needed.  The sample read below follows the same mode switch.
    const lddecode::CombReachReply phaseReach = combReachIndex.queryAgainst(
        frameBuffer.combReachIndex,
        {refLineNumber,
         lineNumber,
         refH,
         h,
         lddecode::CombReachUse::ScalarSignCompare,
         scalarReachSource()});

    if (!phaseReach.allowScalarSignCompare ||
        phaseReach.carrierRelation != lddecode::CarrierPhaseRelation::Opposite) {
        result.penalty = 1000.0;
        result.yPen    = 1000.0;
        result.iqPen   = 0.0;
        return result;
    }

    const int hh = clampH(h);

    // 1D sample: locked path reads the phase-corrected blind bandpass;
    // bucket path reads clpbuffer[0] directly.
    const double *lockedRow = frameBuffer.configuration.phaseCompensation
        ? frameBuffer.locked1DSource_line(lineNumber) : nullptr;

    if (lockedRow && (hh - left) >= 0 && (hh - left) < (right - left)) {
        result.sample = lockedRow[hh - left];
    } else {
        result.sample = frameBuffer.bucketScalar1D_line(lineNumber)[hh];
    }

    // --- Luma Penalty with Neighbor Shaping ---
    //
    // This is the already-paid luma-domain evidence:
    //
    //     reconstructed Y = raw - 2D chroma/composite estimate
    //
    // It compares current/reference against candidate over a small cross:
    //
    //     center line: x-1, x, x+1
    //     vertical:    y-1, y+1 at x
    //
    // getBestCandidate can now reuse result.yPen directly instead of
    // re-deriving a same-pixel scalar chroma distance from result.sample.
    const quint16 *refRawC  = rawbuffer.data() + refLineNumber * fieldWidth;
    const double  *refClpC  = clpbuffer[1].pixel[refLineNumber];

    const quint16 *candRawC = frameBuffer.rawbuffer.data() + lineNumber * fieldWidth;
    const double  *candClpC = frameBuffer.clpbuffer[1].pixel[lineNumber];

    const bool verticalAllowed =
        carrierFrameVerticalAllowed(refLineNumber) &&
        frameBuffer.carrierFrameVerticalAllowed(lineNumber);

    const bool haveUp = verticalAllowed &&
                        (refLineNumber - 1 >= firstLine) &&
                        (lineNumber - 1 >= firstLine);

    const bool haveDn = verticalAllowed &&
                        (refLineNumber + 1 < lastLine) &&
                        (lineNumber + 1 < lastLine);

    const quint16 *refRawU = haveUp
        ? (rawbuffer.data() + (refLineNumber - 1) * fieldWidth)
        : refRawC;
    const double *refClpU = haveUp
        ? clpbuffer[1].pixel[refLineNumber - 1]
        : refClpC;

    const quint16 *refRawD = haveDn
        ? (rawbuffer.data() + (refLineNumber + 1) * fieldWidth)
        : refRawC;
    const double *refClpD = haveDn
        ? clpbuffer[1].pixel[refLineNumber + 1]
        : refClpC;

    const quint16 *candRawU = haveUp
        ? (frameBuffer.rawbuffer.data() + (lineNumber - 1) * fieldWidth)
        : candRawC;
    const double *candClpU = haveUp
        ? frameBuffer.clpbuffer[1].pixel[lineNumber - 1]
        : candClpC;

    const quint16 *candRawD = haveDn
        ? (frameBuffer.rawbuffer.data() + (lineNumber + 1) * fieldWidth)
        : candRawC;
    const double *candClpD = haveDn
        ? frameBuffer.clpbuffer[1].pixel[lineNumber + 1]
        : candClpC;

    auto getLuma = [&](const quint16 *raw, const double *chroma, int idx) -> double {
        return static_cast<double>(raw[idx]) - chroma[idx];
    };

    const int r0 = clampH(refH - 1);
    const int r1 = clampH(refH);
    const int r2 = clampH(refH + 1);

    const int c0 = clampH(h - 1);
    const int c1 = hh;
    const int c2 = clampH(h + 1);

    const double dC0 = std::fabs(getLuma(refRawC, refClpC, r0) -
                                 getLuma(candRawC, candClpC, c0));
    const double dC1 = std::fabs(getLuma(refRawC, refClpC, r1) -
                                 getLuma(candRawC, candClpC, c1));
    const double dC2 = std::fabs(getLuma(refRawC, refClpC, r2) -
                                 getLuma(candRawC, candClpC, c2));

    const double dU = std::fabs(getLuma(refRawU, refClpU, r1) -
                                getLuma(candRawU, candClpU, c1));
    const double dD = std::fabs(getLuma(refRawD, refClpD, r1) -
                                getLuma(candRawD, candClpD, c1));

    const double yPen = ((dC0 + dC1 + dC2 + dU + dD) / 5.0) * invIreScale;

    // --- Chroma/2D Penalty ---
    //
    // Preserve the existing chroma disagreement evidence separately instead of
    // collapsing it into result.penalty only. This lets getBestCandidate treat
    // low-yPen/high-iqPen as "picture-compatible but chroma-grid divergent",
    // which is the compact-checkerboard repair case.
    const int fRef  = carrierLineFlip(refLineNumber);
    const int fCand = carrierLineFlip(lineNumber);

    double iqPen =
        (std::fabs((fRef * refClpC[r0]) - (fCand * candClpC[c0])) * 0.5 +
         std::fabs((fRef * refClpC[r1]) - (fCand * candClpC[c1])) * 1.0 +
         std::fabs((fRef * refClpC[r2]) - (fCand * candClpC[c2])) * 0.5) / 2.0;

    iqPen = (iqPen * invIreScale) * 0.28 * configuration.chromaWeight;

    double penalty = yPen + iqPen + adjustPenalty;
    if (penalty > configuration.candidatePenaltyHardMax)
        penalty = configuration.candidatePenaltyHardMax;

    result.yPen    = yPen;
    result.iqPen   = iqPen;
    result.penalty = penalty;

    return result;
}
