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

// ---------------------------------------------------------------------------
// Retained record from the removed LDCD_PROBE_FRAMEB census.
//
// The Frame B engagement/throttle census that gathered the notes below has
// been removed; the notes themselves are the physical/measured record it
// produced and are kept here.
//
// Frame B's job is to cancel the vertically-invariant image-locked alien --
// the 1D debris that stands in columns -- and its authority is a product of
// four terms:
//     pull = clamp(0.5 * combStrength * reachAuthority, 0, 0.5) * midLicense
// so any one of them at zero silences it. The census split by leg symmetry:
// sym -> 0 IS the signature class (alien vertically invariant, midpoint safe
// and needed); sym -> 1 is diagonal advance, where refusal is correct.
//
// STRONG IMAGE VERTICALS (large carrier-free lateral luma step) are the
// columns where 1D debris stands and interfield is supposed to cancel it.
// That is also exactly where the bevel/cross-colour throttle collapses reach,
// so the authority terms were reported separately there.
//
// Standing vs alternating decomposition of the blind 1D bandpass at the
// Frame B (+-1) and Field B (+-2) geometries. Real chroma ALTERNATES
// between carrier-opposite lines; the bandpass leak of a vertical luma
// edge is IDENTICAL on every line (same D^2 Y), i.e. STANDING -- and a
// standing carrier-band component is invisible to a difference comb:
// center - neighbor = 0. If standing energy dominates at strong image
// verticals, the un-cancelled 74% is not an authority problem at all.
//
// Up/down leak asymmetry at verticals, from the carrier-free aperture
// means: the alternating alien per line follows that line's lateral
// luma curvature (leak = -0.25 * D^2_2 Y), so the +-1 midpoint's
// residual fraction is |D2up - D2dn| / (D2up + D2dn). If this ratio is
// large, a_up != a_dn and the midpoint CANNOT null the alien -- the
// 29% delivery is then a model limit, not a licence problem.
//
// kappa_FB regression: |pairDiff| (IQ IRE, = the alien SUM the +-1 pair
// exposes) against (D2u + D2d) (composite IRE, the luma prediction of
// that same sum). The slope carries BOTH the leak transfer and the
// fullSignedIQ-vs-composite scale factor, which is exactly what the
// residual predictor needs. Restricted to strong verticals with a clear
// alien signature so real vertical chroma difference does not dominate.
//
// Sub-sample REGISTRATION between the centre line and each +-1 leg,
// estimated from carrier-free aperture means only. At a true image
// vertical the luma at a given x is the same on every line, so any
// difference is a horizontal shift: delta ~= (Yleg - Yc) / (dY/dx).
// Split by centre-line parity because under progressive telecine the
// +-1 legs are the OTHER field -- a field-to-field registration error
// must therefore alternate sign with parity, while a genuine image
// slope does not. Leg disagreement was correlated against both the
// lateral GRADIENT (registration) and the CURVATURE (leak) to separate
// the two.
// ---------------------------------------------------------------------------

// TEMPORARY INSTRUMENT (LDCD_PROBE_FBREG=1): grade Frame B's own notch aim
// against the certified aim on covered comp lines, where both exist. The
// question it answers is whether the disciplined luma search reproduces the
// fact -- including how often it correctly STAYS at the detent -- before that
// search is trusted on the lines that carry no fact. Strip when it closes.
struct LdcdFbRegStat {
    long both = 0, exact = 0, missed = 0, falseAim = 0,
         offByOne = 0, offMore = 0, certNonZero = 0, aimNonZero = 0;
    // Detent distribution on the lines the search actually serves (no fact
    // available). The shape to look for is monotonically decreasing; a bin
    // pinned at the adoptable limit is a saturating search, not a picture.
    long dist[3] = {0, 0, 0};   // |d| = 0, 1, 2
};
thread_local LdcdFbRegStat gFbRegStat;

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
        // Fact-family injection is covered-only. On uncovered frames this is
        // the ordinary 1D observation; no two-sided estimate may become a tap
        // base.
        // LDCD_TAP_FACTS=0 restores the pre-injection base for A/B.
        static const bool tapFactsOn = []{
            const char *s = std::getenv("LDCD_TAP_FACTS");
            return !(s && std::atoi(s) == 0);
        }();
        if (configuration.phaseCompensation && !tapFactsOn)
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

    // Certified cede (CONSTRUCTION): on a def line the Field A candidate IS
    // the center -- the certified 1D. The twin capture already separated
    // this field; a comb can only mix truth with a model. Upstream of every
    // election, so preclean consumers and attribution read the same story.
    if (certifiedOneDLevel() >= 2 && certifiedDefLine(tapLine.cacheLine)) {
        const double *center = locked1DSource_line(tapLine.cacheLine);
        if (center) {
            std::copy(center, center + width, outFieldLine);
            return;
        }
    }
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

    // Certified cede (construction): see computeFieldALine.
    if (certifiedOneDLevel() >= 2 && certifiedDefLine(lineNumber)) {
        const double *center = locked1DSource_line(lineNumber);
        if (center) {
            std::copy(center, center + width, outFieldLine);
            if (outReasonLine)
                std::fill(outReasonLine, outReasonLine + width,
                          (std::uint8_t)FieldBReasonCede);
            return;
        }
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
// that sees it (the LDCD_PROBE_CEDE census, since removed), and band-uniform
// verdicts are the remedy; alternation energy is blind to that class. See
// git history for the pass.
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

    // Certified cede (construction): see computeFieldALine.
    if (certifiedOneDLevel() >= 2 && lineNumber >= first &&
        lineNumber < last && certifiedDefLine(lineNumber)) {
        const double *center = locked1DSource_line(lineNumber);
        if (center) {
            std::copy(center, center + width, outFieldLine);
            if (outReasonLine)
                std::fill(outReasonLine, outReasonLine + width,
                          (std::uint8_t)FieldBReasonCede);
            return;
        }
    }
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

    using RR = CombContentReach::RegionRelation;
    const CombContentReach::IntrafieldRegionReach unknownRegion;

    // Publish band membership for downstream band-uniform laws (Y election).
    std::uint8_t *bandOut = chromaBoundaryBand_line(lineNumber);

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

        if (bandOut) bandOut[rel] = region.chromaBoundaryBand ? 1 : 0;

        // A chroma-boundary band cedes to center. Both evidence-driven
        // reclaims of band territory were built and failed on real renders:
        // per-column verdicts manufacture the fishboning/lapel interleave,
        // and band-wide admission manufactures lateral mixing across the
        // boundary (bikini diagonal, 2026-07-27). The band's original law
        // stands -- one region, one render, and that render is the 1D.
        // Detail at boundaries is a job for a separate election candidate,
        // not for this comb.
        if (region.chromaBoundaryBand) {
            wUp = 0.0;
            wDown = 0.0;
        }

        const bool useUp = wUp > 1e-9;
        const bool useDown = wDown > 1e-9;

        double output;
        std::uint8_t reason;
        if (!useUp && !useDown) {
            output = center;
            reason = (upBoundary || downBoundary || region.chromaBoundaryBand)
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
    }
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

    // HQ-leg treatment (user, 2026-07-30: "certified legs are ideally
    // positioned for comb as no other"). A certified leg removes half the
    // comb's unknowns: its luma AND carrier are conservation facts, so every
    // trust-based discount below -- phase alignment toward the center, the
    // correlation ramp, hue projection, the coherence strength gate -- is
    // vestigial against it. Those cautions exist because a MODEL leg's
    // carrier can lie; a certified leg cannot. Content-based gates (the
    // vertical-transition selector, reach legality, disGate) remain: they
    // answer "is the image continuous here", which certification does not
    // decide. Escape LDCD_CERT_LEGS=0 (active only with LDCD_CERT_1D >= 2).
    static const bool certLegsOn = []{
        const char *e = std::getenv("LDCD_CERT_LEGS");
        return !(e && std::atoi(e) == 0);
    }();
    const bool upCert = certLegsOn && certifiedOneDLevel() >= 2 &&
                        haveUpLine && certifiedDefLine(line - 1);
    const bool dnCert = certLegsOn && certifiedOneDLevel() >= 2 &&
                        haveDnLine && certifiedDefLine(line + 1);
    // GILGOL BAND LAW (user, 2026-08-21: "when the comp combs with
    // certified, there's no limit and a few small zippers show").
    // Inside the discovered chroma-boundary band, certification stops
    // being an exemption: a certified leg is a perfect VALUE from the
    // wrong REGION, and the trust limits it bypasses (the correlation
    // ramp, cohGate, the phase-protection agreement grant) are exactly
    // the limits that would have caught the crossing. In-band, certified
    // legs face the same conduct as model legs.
    const std::uint8_t *gilgolBandRow = chromaBoundaryBand_line(line);

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

    // Column-local phase alignment. A certified leg is never aligned toward
    // the center: its phase is exact, and rotating truth toward a model
    // estimate is the alignment running backwards.
    for (int x = 0; x < width; ++x) {
        if (!upCert)
            upIQ[x] = applyColumnPhaseAlignment(centerIQ[x], upIQ[x], invI, phaseAlignLimits);
        if (!dnCert)
            dnIQ[x] = applyColumnPhaseAlignment(centerIQ[x], dnIQ[x], invI, phaseAlignLimits);
    }

    // ------------------------------------------------------------
    // Combine (soft signed contributions + boundary-aware asymmetry)
    // ------------------------------------------------------------
    // NOTE (user, 2026-07-30): the comp-line bracket phase rotation was
    // briefly candidate-local here and was MOVED TO THE HEAD (stage 1b in
    // buildPhaseCorrected1D) -- "Frame B is equally important" and a
    // candidate-local rotation cannot reach it. The centerIQ arriving here
    // already carries bracket phase on comp lines of covered frames.

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

        // In-band demotion of certified legs (band law above).
        const bool gilgolBand = gilgolBandRow && gilgolBandRow[x] != 0;
        const bool upCertX = upCert && !gilgolBand;
        const bool dnCertX = dnCert && !gilgolBand;

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
            const bool upAgrees = useUp && (upCertX || corrUp >= agreeThresh);
            const bool dnAgrees = useDn && (dnCertX || corrDn >= agreeThresh);
            const double a0sq = a0 * a0;

            if (upAgrees && !dnAgrees && useDn && !dnCertX && a0sq > 1e-18) {
                const double proj = dotIQ(ZDnRaw, Z0) / a0sq;
                ZDnRaw = Z0 * proj;
            } else if (dnAgrees && !upAgrees && useUp && !upCertX && a0sq > 1e-18) {
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

        auto legContrib = [&](bool cert, const std::complex<double> &Zn,
                              double an, double reach) {
            std::complex<double> contrib;
            double w;
            if (cert) {
                // Certified leg: the correlation RAMP is a trust instrument
                // and does not apply; the sign is content (a genuine hue
                // reversal must not be averaged as agreement).
                const double c = corrSignedMags(Z0, Zn, a0, an);
                contrib = (c >= 0.0) ? Zn : -Zn;
                w = 1.0;
            } else {
                softAlignBoth(Z0, Zn, a0, an, contrib, w);
            }
            if (w * reach > 0.0) {
                Zsum += contrib * reach;
                wsum += w * reach;
            }
        };
        if (useUp) legContrib(upCertX, ZUpRaw, aUp, upReach);
        if (useDn) legContrib(dnCertX, ZDnRaw, aDn, dnReach);

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
        // A certified leg's participation is not a confidence question;
        // only the content gate (disGate) may back the comb off it.
        if ((useUp && upCertX) || (useDn && dnCertX))
            cohGate = 1.0;

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
        // Covered frames may carry their certified construction here;
        // uncovered frames retain the ordinary 1D observation. +left keeps
        // this branch's rel-indexing convention.
        return configuration.phaseCompensation
            ? bucketScalar1D_line(ln) + left
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

    static const bool fbRegProbe = []{
        const char *e = std::getenv("LDCD_PROBE_FBREG");
        return e && std::atoi(e) == 1;
    }();
    // Reported at the head of the next frame's first line, so the print
    // happens whether or not this frame's last line ceded.
    if (fbRegProbe && line == first) {
        const long nd = gFbRegStat.dist[0] + gFbRegStat.dist[1] +
                        gFbRegStat.dist[2];
        if (gFbRegStat.both > 0) {
            const double pc = 100.0 / (double)gFbRegStat.both;
            std::fprintf(stderr,
                "[FBREG] graded %ld cols vs fact: exact %.1f%% | "
                "missed %.1f%% | falseAim %.1f%% | off1 %.1f%% | "
                "offMore %.1f%% || cert nonzero %.1f%%, aim nonzero %.1f%%\n",
                gFbRegStat.both,
                gFbRegStat.exact * pc, gFbRegStat.missed * pc,
                gFbRegStat.falseAim * pc, gFbRegStat.offByOne * pc,
                gFbRegStat.offMore * pc,
                gFbRegStat.certNonZero * pc, gFbRegStat.aimNonZero * pc);
        }
        if (nd > 0) {
            const double pd = 100.0 / (double)nd;
            std::fprintf(stderr,
                "[FBAIM] searched %ld cols: |d|=0 %.1f%% | |d|=1 %.1f%% | "
                "|d|=2 %.1f%%\n",
                nd, gFbRegStat.dist[0] * pd, gFbRegStat.dist[1] * pd,
                gFbRegStat.dist[2] * pd);
        }
        if (gFbRegStat.both > 0 || nd > 0)
            gFbRegStat = LdcdFbRegStat();
    }

    // Certified cede (construction): on a def line the Frame B candidate IS
    // the center; IQ from the stage-1 locked products of the same scalar.
    if (certifiedOneDLevel() >= 2 && certifiedDefLine(line)) {
        const double *center = locked1DSource_line(line);
        const float *cI4 = locked1DTI4fsc_line(line);
        const float *cQ4 = locked1DTQ4fsc_line(line);
        for (int rel = 0; rel < width; ++rel) {
            outFrameScalar[rel] = center ? center[rel] : 0.0;
            outFrameIQ[rel] = std::complex<double>(
                cI4 ? (double)cI4[rel] : 0.0,
                cQ4 ? (double)cQ4[rel] : 0.0);
        }
        return;
    }
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

        // LDCD_FB_IQ_REG=1 restores the retired precleaned-IQ argmin as the
        // registration, so the change of MATERIAL can be graded on its own.
        static const bool iqRegistration = []{
            const char *e = std::getenv("LDCD_FB_IQ_REG");
            return e && std::atoi(e) == 1;
        }();

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

        // THE LOCATOR, ON LUMA (author, 2026-08-08: "the vertical comb needs
        // to be vertical first. If frame B is going off and searching and
        // mis-locking it's targets, then it's off mission").
        //
        // The registration asks a LUMA-GEOMETRY question -- where did this
        // feature go between the line above and the line below -- and the
        // search below it used to ask that of precleaned IQ. By its own
        // account that search "would wander on textured content", which is
        // not a caveat but the material being wrong for the question: in the
        // IQ domain a lattice is nothing but chroma texture, most of it
        // manufactured by the very confusion Frame B exists to undo. The
        // search was steered by its own quarry.
        //
        // The elementary notch [1,0,1]/2 on raw is the carrier-free luma view
        // that keeps the detail. Its magnitude response is |cos w|: at 4fSC
        // the carrier sits at w = pi/2 where that response is exactly zero,
        // and it is unity at DC and unity again at Nyquist. So it passes the
        // finest horizontal detail in the raster untouched in magnitude and
        // pays only in the band immediately around the carrier -- the
        // opposite trade from a smoothed platform, which keeps the carrier
        // out by throwing the detail away.
        //
        // Its own faults -- summits inside the band are absent from it, and
        // it carries the leak doublets -- are the cheap ones here, because
        // the SAME operator runs on both bracket lines and those artefacts
        // CORRELATE rather than corrupt. It locates; it never supplies shape.
        // Nothing downstream reads these rows: the correction waveform is
        // still the pointwise IQ pair difference at the registered positions.
        constexpr int kNotchPad = 6;   // window |k|=3 plus sampled |s|=3
        const int notchWidth = width + 2 * kNotchPad;
        const double *nU = nullptr;
        const double *nD = nullptr;
        if (rawUpRow && rawDnRow && left >= 1 &&
            left + width < videoParameters.fieldWidth) {
            if ((int)scratch_fbNotchUp.size() < notchWidth) {
                scratch_fbNotchUp.resize(notchWidth);
                scratch_fbNotchDn.resize(notchWidth);
            }
            auto notchAt = [](const quint16 *r, int i) {
                return 0.5 * (static_cast<double>(r[i - 1]) +
                              static_cast<double>(r[i + 1]));
            };
            for (int x = 0; x < width; ++x) {
                scratch_fbNotchUp[kNotchPad + x] = notchAt(rawUpRow, left + x);
                scratch_fbNotchDn[kNotchPad + x] = notchAt(rawDnRow, left + x);
            }
            // Edge replication, matching the IQ rows' convention exactly.
            for (int p = 0; p < kNotchPad; ++p) {
                scratch_fbNotchUp[p] = scratch_fbNotchUp[kNotchPad];
                scratch_fbNotchDn[p] = scratch_fbNotchDn[kNotchPad];
                scratch_fbNotchUp[kNotchPad + width + p] =
                    scratch_fbNotchUp[kNotchPad + width - 1];
                scratch_fbNotchDn[kNotchPad + width + p] =
                    scratch_fbNotchDn[kNotchPad + width - 1];
            }
            nU = scratch_fbNotchUp.data() + kNotchPad;
            nD = scratch_fbNotchDn.data() + kNotchPad;
        }

        // THREE DISCIPLINES, LIFTED FROM THE CERTIFIED SEARCH, which measured
        // them necessary: the aim census read 30/37/33 across s=0 / |s|=1 /
        // |s|=2 with the largest bin PINNED at the search limit before they
        // were added, and 59.7/26.4/13.9 -- monotonically decreasing, the
        // shape a real slope population has -- after. All three push toward
        // the detent.
        //
        //   SAMPLE WIDER THAN YOU ADOPT. Search runs to +-3, only +-2 may be
        //     adopted, so every adoptable shift has neighbours on both sides
        //     and can be REQUIRED to be a genuine interior minimum. The old
        //     search sampled +-2 and adopted +-2, so the limit shifts could
        //     not be tested at all and collected the search's failures.
        //   INTERIOR MINIMUM. A shift stands only if it is strictly better
        //     than the positions either side of it. A real diagonal makes a
        //     trough; the near-ties dense fine texture produces do not, and
        //     this is what separates them.
        //   IDENTIFIABILITY FLOOR. An aim is only answerable where the two
        //     rows actually DISAGREE at the detent. Below the floor there is
        //     nothing for a shift to explain, every shift scores the same to
        //     within noise, and letting the margin pick among them is reading
        //     a coin toss as geometry. A clamp on an impossible, not a lever.
        //
        // The 8% margin stands on top of all three: d = 0 is the default and
        // the vertical is Frame A's whole job, so a non-zero aim displaces the
        // detent only on proof.
        //
        // ONE implementation, per the no-duplicate-math law: the fallback
        // registration and the census below both call this.
        auto notchAim = [&](int x) -> int {
            constexpr int kFbRegMax = 2;      // adoptable
            constexpr int kFbRegSearch = 3;   // sampled
            constexpr double kFbIdentifyIRE = 1.0;
            double ndev[2 * kFbRegSearch + 1];
            for (int si = 0; si <= 2 * kFbRegSearch; ++si) {
                const int s = si - kFbRegSearch;
                double acc = 0.0;
                for (int k = -3; k <= 3; ++k)
                    acc += kWin[k + 3] *
                           std::fabs(nU[x + k - s] - nD[x + k + s]);
                ndev[si] = acc / kWinSum;
            }
            if (ndev[kFbRegSearch] * invIreScale < kFbIdentifyIRE)
                return 0;
            int bestD = 0;
            double bestDev = ndev[kFbRegSearch] / kRegMargin;
            for (int si = 0; si <= 2 * kFbRegSearch; ++si) {
                const int s = si - kFbRegSearch;
                if (s == 0 || std::abs(s) > kFbRegMax) continue;
                if (ndev[si] >= ndev[si - 1] ||
                    ndev[si] >= ndev[si + 1]) continue;
                if (ndev[si] < bestDev) {
                    bestDev = ndev[si];
                    bestD = s;
                }
            }
            return bestD;
        };

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

            // CERTIFIED REGISTRATION FIRST (2026-08-08). Where the frame
            // carries a fact-grade aim for this column -- measured on the
            // two bracketing lines' certified, carrier-free luma, published
            // by buildCertifiedCarrierStage -- take it instead of searching
            // here. Same quantity, same sign convention (see comb.h); the
            // difference is the material it was measured on. This search
            // reads precleaned IQ, and by its own account above is "steered
            // by chroma texture", which is precisely the wander certified
            // luma cannot have. The fact exists only on the comp lines of a
            // covered frame, so the IQ search below still serves every other
            // line and the un-covered case is unchanged.
            //
            // bestSi is set FROM the adopted d rather than left where the
            // search put it: dSame = devMag[bestSi] feeds the ratio and ride
            // gates below, and a gate read at a different shift from the
            // correction it licenses is the incoherence this whole change is
            // about.
            const qint8 *certRegRow = certRegistration_line(line);
            const qint8 certReg = certRegRow ? certRegRow[x] : kCertRegNone;
            int bestSi;
            if (certReg != kCertRegNone) {
                const int sStarCert = sameIsUp ? -(int)certReg : (int)certReg;
                bestSi = std::clamp(sStarCert + 2, 0, 4);
                // GRADE THE ESTIMATOR WHERE THE FACT EXISTS (temporary,
                // LDCD_PROBE_FBREG=1). Covered comp lines carry both aims, so
                // the notch search can be scored against certified truth
                // before it is trusted on the lines that have no truth. Strip
                // when the question closes.
                if (fbRegProbe && nU && nD) {
                    const int aim = notchAim(x);
                    ++gFbRegStat.both;
                    if (aim == (int)certReg) ++gFbRegStat.exact;
                    else if (aim == 0) ++gFbRegStat.missed;
                    else if (certReg == 0) ++gFbRegStat.falseAim;
                    else if (std::abs(aim - (int)certReg) == 1)
                        ++gFbRegStat.offByOne;
                    else ++gFbRegStat.offMore;
                    if (certReg != 0) ++gFbRegStat.certNonZero;
                    if (aim != 0) ++gFbRegStat.aimNonZero;
                }
            } else if (nU && nD && !iqRegistration) {
                // FRAME B'S OWN AIM. Same question, same pairing and the same
                // sign convention as certRegistration -- the aligned pair is
                // (up[x-s], dn[x+s]) -- measured on the notch rows above, so
                // this is the certified search's form running on material
                // Frame B can compute for itself on any frame. Where the fact
                // exists it is preferred above; where it does not, this is the
                // same search rather than a different one, which is what makes
                // the certified aim a corroborator instead of a crutch. The
                // three disciplines it carries are documented at notchAim.
                const int bestD = notchAim(x);
                if (fbRegProbe)
                    ++gFbRegStat.dist[std::min(std::abs(bestD), 2)];
                const int sStarNotch = sameIsUp ? -bestD : bestD;
                bestSi = std::clamp(sStarNotch + 2, 0, 4);
            } else if (iqRegistration) {
                // A/B only (LDCD_FB_IQ_REG=1): the retired IQ argmin, kept so
                // the material change can be measured in one variable.
                bestSi = 2;
                double bestDev = devMag[2] / kRegMargin;
                for (int si = 0; si < 5; ++si) {
                    if (si == 2) continue;
                    if (devMag[si] < bestDev) {
                        bestDev = devMag[si];
                        bestSi = si;
                    }
                }
                if (fbRegProbe)
                    ++gFbRegStat.dist[std::min(std::abs(bestSi - 2), 2)];
            } else {
                // No luma locator here (frame edge, or the notch's taps fall
                // outside the raster). Standing at the detent is the honest
                // answer: the vertical is what Frame B is for, and a search
                // with nothing to search on is exactly the mis-locking this
                // change exists to stop.
                bestSi = 2;
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
            }
        }

        std::complex<double> Zout = Zc;

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

                if (!std::isfinite(Zout.real()) ||
                    !std::isfinite(Zout.imag()))
                {
                    Zout = Zc;
                }
            }
        }

        // Graceful failure at highlights.  No magnitude clamp here: cmag(Zout)
        // is fullSignedIQ scale, not composite-scalar scale.
        if (!std::isfinite(Zout.real()) || !std::isfinite(Zout.imag())) {
            Zout = (std::isfinite(Z0Preclean.real()) &&
                    std::isfinite(Z0Preclean.imag()))
                       ? Z0Preclean
                       : std::complex<double>(0.0, 0.0);
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

    // Temporal grammar: the signed relation is absolute (lineFlip derives
    // from metadata fieldPhaseIDs), so cross-frame replies are temporally
    // correct. BOTH definite relations are usable by the temporal comb:
    //   Opposite -> partner scalar carries -C: (base - s)/2 combs.
    //   Same     -> partner scalar carries +C: negate the stored sample so
    //               the same downstream math (base - (-s))/2 averages.
    // Quadrature/unknown remains illegal. The previous form demanded
    // Opposite while getBestCandidate's frame pre-gate demanded equal
    // lineFlips (= Same): mutually exclusive, so the frame-axis temporal
    // comb never ran, and cross-frame field work existed only on the
    // alternating lines where the heuristics happened to align -- the
    // line-parity striping on cadence material.
    // LDCD_3D_TEMPORAL_GRAMMAR=1 enables the temporal-grammar candidate
    // rules (both relations legal, sign-folded samples, Same-relation bias).
    // Default OFF: the chroma gain is real but per-pixel temporal ACCEPTANCE
    // (the hull guard) still varies line-to-line at dense verticals, which
    // measured net-worse on Y than the old dead-gate state on A/C frames.
    // The acceptance-uniformity pass is the remaining design work.
    static const bool temporalGrammar = []{
        const char *e = std::getenv("LDCD_3D_TEMPORAL_GRAMMAR");
        return e && std::atoi(e) != 0;
    }();
    double relationSign;
    if (!phaseReach.allowScalarSignCompare) {
        result.penalty = 1000.0;
        result.yPen    = 1000.0;
        result.iqPen   = 0.0;
        return result;
    }
    // A Same-relation partner cannot yield carrier AT ALL: with equal phase
    // the difference cancels the CARRIER and leaves a luma difference, so
    // the sign-folded form is a temporal AVERAGE, not a comb -- it keeps the
    // mean 1D leak that combing cancels. Admitting it alongside Opposite
    // therefore runs two different physical operations on alternating lines,
    // which is the line-parity striping that kept this whole path parked
    // (measured 2026-07-28: acceptance was already uniform at 99.8%/share
    // 0.95 and smoothing it changed nothing -- the stripes were in the
    // VALUES, i.e. in the operation class). Opposite is the only path to a
    // carrier product; content agreement is already gated by the 2D
    // similarity curve (AGREEMENT_REWARD_*) in getBestCandidate, which is
    // what makes a cross-frame Opposite partner honest where the picture is
    // static. LDCD_3D_SAME_RELATION=1 restores the old averaging branch for
    // A/B only.
    static const bool allowSameRelation = []{
        const char *e = std::getenv("LDCD_3D_SAME_RELATION");
        return e && std::atoi(e) != 0;
    }();
    if (phaseReach.carrierRelation == lddecode::CarrierPhaseRelation::Opposite) {
        relationSign = 1.0;
    } else if (temporalGrammar && allowSameRelation &&
               phaseReach.carrierRelation == lddecode::CarrierPhaseRelation::Same) {
        relationSign = -1.0;
        adjustPenalty += 3.0;
    } else {
        result.penalty = 1000.0;
        result.yPen    = 1000.0;
        result.iqPen   = 0.0;
        return result;
    }

    const int hh = clampH(h);

    // 1D sample: locked path reads the anchored-aware source (the same
    // plane split3D's base1d consumes -- fact injection 2026-08-02; falls
    // back to the phase-corrected blind bandpass when no anchored plane);
    // bucket path reads clpbuffer[0] directly. relationSign folds the
    // carrier relation in so every consumer keeps Opposite-form math.
    const double *lockedRow = frameBuffer.configuration.phaseCompensation
        ? frameBuffer.combSource1D_line(lineNumber) : nullptr;

    if (lockedRow && (hh - left) >= 0 && (hh - left) < (right - left)) {
        result.sample = relationSign * lockedRow[hh - left];
    } else {
        result.sample = relationSign * frameBuffer.bucketScalar1D_line(lineNumber)[hh];
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

    // CARRIER-FREE LUMA FOR THE SIMILARITY DISTANCE.
    //
    // This distance used to be built from raw - clpbuffer[1] -- a per-frame 2D
    // chroma ESTIMATE. Where that estimate carries a carrier-locked misread it
    // inverts frame to frame, so a reference and a temporal candidate hold it
    // with opposite sign and the distance reads about TWICE the misread. The
    // artifact inflates the very measurement used to judge the candidate that
    // would cancel it, which is why the old deviation veto self-defeated and
    // was retired. The defect was in the INPUT, not in the penalty.
    //
    // The coarse platform cannot carry it: a legal four-sample mean cancels the
    // carrier exactly, so the platform is carrier-free by construction. The
    // lurch-sharpened form is preferred over the plain boxcar because lurch
    // un-smears the four-sample placement limit, and the smoothness that
    // remains is wanted here rather than tolerated -- this is a SIMILARITY
    // metric, and a little smoothing is what stops it reacting to per-frame
    // noise instead of to real content change.
    //
    // Falls back to the old reconstruction only where no coarse exists (bucket
    // mode), which leaves those paths exactly as they were.
    const int coarseLeft = videoParameters.activeVideoStart;
    const int coarseW    = demodWidth;
    auto coarseRow = [](const FrameBuffer &fb, int line) -> const double * {
        if (!fb.lockedLumaCacheValid) return nullptr;
        if (!fb.lockedLumaSharp_flat.empty())
            return fb.lockedLumaSharp_line(line);
        if (!fb.lockedLumaBaseY4_flat.empty())
            return fb.lockedLumaBaseY4_line(line);
        return nullptr;
    };

    const double *refCoarseC  = coarseRow(*this, refLineNumber);
    const double *candCoarseC = coarseRow(frameBuffer, lineNumber);
    const double *refCoarseU  = haveUp ? coarseRow(*this, refLineNumber - 1)
                                       : refCoarseC;
    const double *refCoarseD  = haveDn ? coarseRow(*this, refLineNumber + 1)
                                       : refCoarseC;
    const double *candCoarseU = haveUp ? coarseRow(frameBuffer, lineNumber - 1)
                                       : candCoarseC;
    const double *candCoarseD = haveDn ? coarseRow(frameBuffer, lineNumber + 1)
                                       : candCoarseC;

    auto getLuma = [&](const double *coarse, const quint16 *raw,
                       const double *chroma, int idx) -> double {
        if (coarse) {
            const int rel = idx - coarseLeft;
            if (rel >= 0 && rel < coarseW) {
                const double v = coarse[rel];
                if (std::isfinite(v)) return v;
            }
        }
        return static_cast<double>(raw[idx]) - chroma[idx];
    };

    const int r0 = clampH(refH - 1);
    const int r1 = clampH(refH);
    const int r2 = clampH(refH + 1);

    const int c0 = clampH(h - 1);
    const int c1 = hh;
    const int c2 = clampH(h + 1);

    const double dC0 = std::fabs(getLuma(refCoarseC, refRawC, refClpC, r0) -
                                 getLuma(candCoarseC, candRawC, candClpC, c0));
    const double dC1 = std::fabs(getLuma(refCoarseC, refRawC, refClpC, r1) -
                                 getLuma(candCoarseC, candRawC, candClpC, c1));
    const double dC2 = std::fabs(getLuma(refCoarseC, refRawC, refClpC, r2) -
                                 getLuma(candCoarseC, candRawC, candClpC, c2));

    const double dU = std::fabs(getLuma(refCoarseU, refRawU, refClpU, r1) -
                                getLuma(candCoarseU, candRawU, candClpU, c1));
    const double dD = std::fabs(getLuma(refCoarseD, refRawD, refClpD, r1) -
                                getLuma(candCoarseD, candRawD, candClpD, c1));

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
