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

#include <string>
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

// Geometry-only evidence for a compact luma excursion. The detector tests several
// symmetric radii because an optically broadened impulse may still sit on its own
// shoulder at a single radius. Each accepted radius requires the same background
// on both sides and stable outer flanks, excluding steps and extended texture.
// Carrier legality is intentionally absent: this service publishes only the luma
// shape evidence for downstream consumers to combine with their own carrier facts.
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

// ---------------------------------------------------------------------------
// Frame B ±1 operand admission.
//
// Frame B is a subtractive estimator, so an operand from a different chroma region
// can drive the result outside the hull of the observed lines. Reach therefore
// controls operand admission rather than scaling correction strength.
//
// One floor governs both legs. The ±1 partners are one TV line from center, so
// kFrameBMajorIRE is set above the intrafield region floor. A leg marked
// DifferentRegion at or above that difference is not admitted to the refusal-band
// seed; both legs are acted on symmetrically at the band level.
constexpr double kFrameBMajorIRE = 12.0;
constexpr double kFrameBRegionChromaFloorIRE = 5.0;

// Independent carrier-free luma evidence. One differing leg is an ordinary
// coverage transition; two same-signed departures place center outside the
// rational luma range of its vertical neighbourhood. Each leg must exceed the
// 5 IRE near-departure floor and the combined departure must reach 12 IRE, which
// admits asymmetric islands without classifying a one-sided moving edge as one.
constexpr double kFrameBLumaIslandNearIRE = 5.0;
constexpr double kFrameBLumaIslandTotalIRE = 12.0;

inline bool frameBLumaIslandSeed(double centerIRE,
                                 double upIRE,
                                 double downIRE)
{
    if (!std::isfinite(centerIRE) || !std::isfinite(upIRE) ||
        !std::isfinite(downIRE))
        return false;
    const double dUp = upIRE - centerIRE;
    const double dDown = downIRE - centerIRE;
    const bool sameDirection =
        (dUp > 0.0 && dDown > 0.0) ||
        (dUp < 0.0 && dDown < 0.0);
    if (!sameDirection)
        return false;

    const double nearDeparture =
        std::min(std::abs(dUp), std::abs(dDown));
    const double totalDeparture = std::abs(dUp) + std::abs(dDown);
    return nearDeparture >= kFrameBLumaIslandNearIRE &&
           totalDeparture >= kFrameBLumaIslandTotalIRE;
}

// SYMMETRY LAW.
//
// Same and Opposite roles alternate by column, so per-leg refusal would produce
// different operations on adjacent carrier phases. Refusing Same substitutes the
// center and preserves a full alien estimate; refusing Opposite removes the alien
// measurement entirely. The verdict is therefore band input only: a failure
// contributes to frameBBandSeed, where both legs are refused together by any
// consumer that chooses to cede.
inline bool frameBLegAdmitted(CombContentReach::RegionRelation measured,
                              double differenceIRE)
{
    return !(measured == CombContentReach::RegionRelation::DifferentRegion &&
             differenceIRE >= kFrameBMajorIRE);
}

// RELATION LOCKING separates an alien term from a chroma-region transition.
//
// An alien term rides center on the Same leg while displacing the Opposite leg by
// approximately twice the alien amplitude. A real chroma-region transition has no
// such relation preference. kFrameBRelationLockRatio identifies this asymmetric
// difference pattern and exempts relation-locked alien cancellation from the
// region-boundary interpretation.
//
// The measurement is used for binary admission/band evidence only; it never scales
// Frame B's subtraction strength.
constexpr double kFrameBRelationLockRatio = 0.5;

// True when the column's vertical difference belongs to the carrier rather
// than to the picture. Consumes only facts already published on the row.
inline bool frameBRelationLocked(
    const CombContentReach::IntrafieldRegionReach &r, bool sameIsUp)
{
    const double dSame = sameIsUp ? r.upDifferenceIRE : r.downDifferenceIRE;
    const double dOpp  = sameIsUp ? r.downDifferenceIRE : r.upDifferenceIRE;
    return dOpp > 1e-6 && dSame <= kFrameBRelationLockRatio * dOpp;
}

// The chroma branch of the band seed: up, down and centre in three different
// regions, every pairwise separation major. Deliberately NOT the evaluator's
// own `threeRegion`, which carries Field B's triplet promotions and fires with
// one leg still continuing centre -- the one-legged case, which combs rather
// than cedes.  The independent luma-island branch is ORed below.
//
// Left on the region evaluator deliberately. "Changes and stays" IS the region
// signature, and the three-way seed demands all three pairwise separations at
// once -- corroboration the single-leg test never had.
inline bool frameBBandSeed(const CombContentReach::IntrafieldRegionReach &r,
                           bool sameIsUp,
                           bool lumaIsland)
{
// A relation-locked triplet is treated as alien cancellation rather than a region
// boundary. The exemption is applied at the shared refusal-band seed so the Same
// and Opposite legs remain symmetric.
    const bool strictThreeRegionViolation =
        !frameBLegAdmitted(r.upMeasured, r.upDifferenceIRE) &&
        !frameBLegAdmitted(r.downMeasured, r.downDifferenceIRE) &&
        r.outerComparable &&
        r.upDownDifferenceIRE >= kFrameBMajorIRE &&
        r.upDownHueDifferenceDeg >=
            CombContentReach::kRegionDifferentHueDeg;

    // Cancellable alternation may be imperfectly balanced, but it still
    // carries loosely comparable saturation.  A decisive magnitude collapse
    // on BOTH legs is therefore contrary evidence in its own right: center is
    // outside the rational saturation range of its vertical neighbourhood.
    // One collapsed leg is only a coverage transition; making it cede both
    // operands causes moving edges to switch Frame B off and strobe against
    // uncovered frames.  Less extreme chroma differences retain the existing
    // relation-lock exemption and strict three-region requirement.
    const bool twoSidedStrongAsym =
        r.upStrongAsym && r.downStrongAsym;
    const bool chromaViolation =
        twoSidedStrongAsym ||
        (!frameBRelationLocked(r, sameIsUp) &&
         strictThreeRegionViolation);

    // The two facts are independent.  Relation locking can explain an IQ
    // difference as alien, but it cannot erase a carrier-free luma island.
    return chromaViolation || lumaIsland;
}


// ---------------------------------------------------------------------------
// Frame B registration model.
//
// Frame B cancels vertically invariant image-locked colour while retaining a
// registered ±1 pair. Registration is derived from luma geometry, not from the IQ
// material being cancelled. A diagonal advance is published once per line and
// Frame B rounds that fact to the integer shift its ±1 aperture can justify.
//
// The registered pair is interpreted in the center line's carrier relation before
// forming the signed alien estimate. The correction remains pointwise at the
// adopted coordinates; registration changes where the pair is sampled, not the
// estimator's gain.
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

    // Fill the 7-tap smoothed signed-IQ row for `line` once per frame. The balanced
    // end-weighted aperture (0.5,1,1,1,1,1,0.5) smooths the integer-centred baseband
    // products and normalizes by total weight 6, preserving the full signed-IQ scale.
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
    
    // Same-field taps use active-boundary mirror resolution. If the requested outward
    // partner is missing and the inward same-field partner exists, resolve to that
    // partner:
    //   top active row:    U2 resolves to D2
    //   bottom active row: D2 resolves to U2
    // The same rule applies to ±4 taps.
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
        ensureWidth(tapLine.interfieldRegionReach);
    }
    // Cleared on every build: a stale true would hand Frame B a previous
    // line's admission verdicts as if they were this line's.
    tapLine.interfieldRegionValid = false;

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
        ensureWidth(tapLine.coarseU4IRE);
        ensureWidth(tapLine.coarseD4IRE);
        ensureWidth(tapLine.vReachResid1IRE);
        ensureWidth(tapLine.vReachResid2IRE);
        ensureWidth(tapLine.vReachResid2UpIRE);
        ensureWidth(tapLine.vReachResid2DownIRE);
        ensureWidth(tapLine.vReachResid4IRE);
        ensureWidth(tapLine.lateralCornerIRE);
        ensureWidth(tapLine.notchCoarse0IRE);
    }

    if (wantFieldB || wantFrame) {
        ensureWidth(tapLine.hLumaDeltaIRE);
    }

    auto getCompRow = [&](int ln)->const double* {
        if (ln < first || ln >= last) return nullptr;
    // Fact-family injection is covered-only. On uncovered frames the tap base is the
    // ordinary 1D observation; no two-sided certified estimate may replace it.
    // Covered positions use the certified source selected by the tap-source rules.
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

    // The comb's coarse rows are the luma estimate its own decisions react to.
    // lockedLumaSmooth is a block-centre scaffold: one mean per four samples,
    // anchored at the block centre and LINEARLY INTERPOLATED between anchors
    // (comb.cpp).  A piecewise-linear row has zero second difference inside
    // every segment and a knot at every anchor, so lateralCornerIRE -- the
    // second difference of coarse0IRE -- reports the block grid as much as the
    // picture, and luma detail finer than a block is invisible to it.
    //
    // The solved platform is the single coarse estimate shared by every client;
    // private alternate coarse construction is not permitted here.
    const bool combUsesSolved = !lockedLumaSolved_flat.empty();
    auto getLumaRow = [&](int ln)->const double* {
        if (!configuration.phaseCompensation ||
            !lockedLumaCacheValid ||
            demodWidth < width ||
            lockedLumaSmooth_flat.empty() ||
            ln < 0 || ln >= demodLines)
        {
            return nullptr;
        }
        if (combUsesSolved) return lockedLumaSolved_line(ln);
        return lockedLumaSmooth_line(ln);
    };

    const double *luma0  = getLumaRow(tapLine.ln0);
    const double *lumaU1 = tapLine.haveU1 ? getLumaRow(tapLine.lnU1) : nullptr;
    const double *lumaD1 = tapLine.haveD1 ? getLumaRow(tapLine.lnD1) : nullptr;
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
            const double admission = centerAnalysis &&
                centerAnalysis[rel].scheduleConformance ==
                    lddecode::CarrierScheduleConformance::LegalCarrier
                ? 1.0 : 0.0;
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

    // Balanced 7-tap horizontal aggregate, memoised per line. Even offsets carry one
    // carrier axis and odd offsets the other, so 0.5 end weights equalize the axis
    // sums (3:3) and keep the vector phase-flat. The wider aperture supplies stable
    // region-reach evidence while remaining integer-centred.
            ensureSmoothedLockedRow(tapLine.ln0);
            ensureSmoothedLockedRow(tapLine.lnU2);
            ensureSmoothedLockedRow(tapLine.lnD2);
            const float *sI0  = smoothedLockedTI_line(tapLine.ln0);
            const float *sQ0  = smoothedLockedTQ_line(tapLine.ln0);
            const float *sIUp = smoothedLockedTI_line(tapLine.lnU2);
            const float *sQUp = smoothedLockedTQ_line(tapLine.lnU2);
            const float *sIDn = smoothedLockedTI_line(tapLine.lnD2);
            const float *sQDn = smoothedLockedTQ_line(tapLine.lnD2);
            // ±4 grammar reach and smoothed rows for contour-influence gating.
            const bool have4IQ =
                tapLine.lnU4 >= 0 && tapLine.lnU4 < demodLines &&
                tapLine.lnD4 >= 0 && tapLine.lnD4 < demodLines;
            lddecode::CombReachReply up4Reach, dn4Reach;
            bool haveUp4 = false, haveDn4 = false;
            const float *sIUp4 = nullptr, *sQUp4 = nullptr;
            const float *sIDn4 = nullptr, *sQDn4 = nullptr;
            if (have4IQ && tapLine.haveU4) {
                up4Reach = combReachIndex.query(
                    {lineNumber, tapLine.lnU4, left, left,
                     lddecode::CombReachUse::IQCompare, iqSource});
                if (up4Reach.allowIQCompare) {
                    ensureSmoothedLockedRow(tapLine.lnU4);
                    sIUp4 = smoothedLockedTI_line(tapLine.lnU4);
                    sQUp4 = smoothedLockedTQ_line(tapLine.lnU4);
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
                        invI,
                        5.0,
    // Sharp raw ±2 scalar facts feed the first-pass AlienCancel decision.
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

    // ±1 interfield content census, for Frame B's operand admission.  The same
    // evaluator at a different vertical step: center against the ±1 legs,
    // relation-aligned by the reach grammar exactly as the ±2 pass above.
    //
    // The raw-scalar AlienCancel shortcut is deliberately NOT passed here.  It
    // reads a near-zero raw leg difference as vertically coherent non-carrier
    // energy, and that inference is specific to ±2 same-field legs, which are
    // ANTI-PHASE carriers: there, raw-identical can only mean the carrier
    // cancelled.  ±1 legs are one TV line apart in the opposite field and
    // carry no such guarantee.  Omitting the argument falls the evaluator back
    // to the smoothed-IQ hue test, which is frame-agnostic.
    if (wantFrame) {
        const size_t iqCount =
            static_cast<size_t>(demodLines) * static_cast<size_t>(demodWidth);
        const bool haveSignedIQ =
            configuration.phaseCompensation &&
            demodWidth >= width &&
            tapLine.haveU1 && tapLine.haveD1 &&
            tapLine.ln0 >= 0 && tapLine.ln0 < demodLines &&
            tapLine.lnU1 >= 0 && tapLine.lnU1 < demodLines &&
            tapLine.lnD1 >= 0 && tapLine.lnD1 < demodLines &&
            locked1DTI4fsc_flat.size() >= iqCount &&
            locked1DTQ4fsc_flat.size() >= iqCount &&
            carrierAnalysis_flat.size() >= iqCount;

        if (haveSignedIQ) {
            const lddecode::CombReachReply up1Reach = combReachIndex.query(
                {lineNumber, tapLine.lnU1, left, left,
                 lddecode::CombReachUse::IQCompare, iqSource});
            const lddecode::CombReachReply dn1Reach = combReachIndex.query(
                {lineNumber, tapLine.lnD1, left, left,
                 lddecode::CombReachUse::IQCompare, iqSource});

            ensureSmoothedLockedRow(tapLine.ln0);
            ensureSmoothedLockedRow(tapLine.lnU1);
            ensureSmoothedLockedRow(tapLine.lnD1);
            const float *sI0  = smoothedLockedTI_line(tapLine.ln0);
            const float *sQ0  = smoothedLockedTQ_line(tapLine.ln0);
            const float *sIUp = smoothedLockedTI_line(tapLine.lnU1);
            const float *sQUp = smoothedLockedTQ_line(tapLine.lnU1);
            const float *sIDn = smoothedLockedTI_line(tapLine.lnD1);
            const float *sQDn = smoothedLockedTQ_line(tapLine.lnD1);
            // Relation sign is a LINE fact -- one grammar lookup per leg,
            // hoisted. Zero means the grammar does not authorise the
            // comparison, which the evaluator reports as Unknown and no
            // policy here can convict.
            auto legSign = [](const lddecode::CombReachReply &rep) {
                if (!rep.allowIQCompare) return 0.0;
                if (rep.carrierRelation ==
                    lddecode::CarrierPhaseRelation::Same) return 1.0;
                if (rep.carrierRelation ==
                    lddecode::CarrierPhaseRelation::Opposite) return -1.0;
                return 0.0;
            };
            const double upSgn = legSign(up1Reach);
            const double dnSgn = legSign(dn1Reach);
            const bool sameIsUp = up1Reach.carrierRelation ==
                lddecode::CarrierPhaseRelation::Same;

            for (int rel = 0; rel < width; ++rel) {
                auto &r = tapLine.interfieldRegionReach[rel];
                const std::complex<double> z0(sI0[rel], sQ0[rel]);

                const bool lumaIsland = luma0 && lumaU1 && lumaD1 &&
                    frameBLumaIslandSeed(
                        luma0[rel] * invI,
                        lumaU1[rel] * invI,
                        lumaD1[rel] * invI);

                const std::complex<double> zUp(sIUp[rel], sQUp[rel]);
                const std::complex<double> zDown(sIDn[rel], sQDn[rel]);
                const double centerMagIRE = boundedMag(z0) * invI;
                const bool possibleStrongAsym =
                    CombContentReach::isStrongRegionMagnitudeAsym(
                        centerMagIRE, boundedMag(zUp) * invI,
                        kFrameBRegionChromaFloorIRE) ||
                    CombContentReach::isStrongRegionMagnitudeAsym(
                        centerMagIRE, boundedMag(zDown) * invI,
                        kFrameBRegionChromaFloorIRE);

                // CHEAP CHROMA GUARD.  Below the major IQ floor no hue geometry
                // can create the strict three-region branch, so skip its
                // evaluator unless a strong saturation asymmetry can create
                // the independent chroma branch.  The carrier-free luma branch
                // has already been decided above and must survive this path.
                const double dU = (upSgn != 0.0)
                    ? boundedMag(z0 - upSgn * zUp) * invI
                    : 0.0;
                const double dD = (dnSgn != 0.0)
                    ? boundedMag(z0 - dnSgn * zDown) * invI
                    : 0.0;
                if (dU < kFrameBMajorIRE && dD < kFrameBMajorIRE &&
                    !possibleStrongAsym) {
                    r = CombContentReach::IntrafieldRegionReach{};
                    r.chromaBoundarySeed = lumaIsland;
                    r.chromaBoundaryBand = lumaIsland;
                    continue;
                }

                r = CombContentReach::evaluateIntrafieldRegionReach(
                        z0,
                        zUp,
                        zDown,
                        up1Reach.carrierRelation,
                        dn1Reach.carrierRelation,
                        up1Reach.allowIQCompare,
                        dn1Reach.allowIQCompare,
                        invI,
                        kFrameBRegionChromaFloorIRE);

    // Frame B publishes its own refusal-band seed on the shared row. The shared
    // dilation machinery consumes that seed without changing Frame B's law.
                r.chromaBoundarySeed =
                    frameBBandSeed(r, sameIsUp, lumaIsland);
                r.chromaBoundaryBand = r.chromaBoundarySeed;
            }

            // Band-uniform, same radius as Field B's.  A per-column cede at a
            // chroma boundary is the beading mechanism; the run gets one
            // verdict or the switching inside it becomes the artifact.
            CombContentReach::markIntrafieldChromaBoundaryBand(
                tapLine.interfieldRegionReach, 4);
            tapLine.interfieldRegionValid = true;

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

    // There is deliberately no amplitude/compactness-based "chroma shape" exception
    // here. High-frequency energy defaults to luma. Physical appearance may limit a
    // reach that already has carrier authority, but it cannot itself register energy
    // as carrier or grant an escape from the comb. Carrier privileges come only from
    // the grammar/conformance tables and their named consumers.

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
        double *outCoarseU4 = tapLine.coarseU4IRE.data();
        double *outCoarseD4 = tapLine.coarseD4IRE.data();

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
            // One carrier-cycle aperture with integer centroid at rel, so locked luma and the
            // fallback notch luma refer to the same horizontal coordinate.
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

        // Same rules at +-4: an absent row falls back to the centre so a
        // zero row can never read as a huge false gradient, and the validity
        // flag reports whether either neighbour was real.
        const bool coarseU4Real = (lumaU4 != nullptr) || tapLine.haveU4;
        const bool coarseD4Real = (lumaD4 != nullptr) || tapLine.haveD4;
        if (lumaU4)            fillLockedCoarse(lumaU4, outCoarseU4);
        else if (coarseU4Real) fillNotchCoarse(tU4, outCoarseU4);
        else std::copy(outCoarse0, outCoarse0 + width, outCoarseU4);
        if (lumaD4)            fillLockedCoarse(lumaD4, outCoarseD4);
        else if (coarseD4Real) fillNotchCoarse(tD4, outCoarseD4);
        else std::copy(outCoarse0, outCoarse0 + width, outCoarseD4);
        // ---- Shared trigger facts (see comb.h) --------------------------
        {
            const CombCarrierGrammar *tgGram = carrierGrammarLine(lineNumber);
            const int sp0 = tgGram ? (tgGram->samplePhase0 & 1) : 0;
            // The tap's `comp` IS the carrier at that sample -- the locked
            // path publishes it and leaves `raw` unfetched, so reconstructing
            // a bandpass here would both duplicate the math and (in locked
            // mode) read an empty row.  Take the published product.
            auto carrierRow = [&](const CombTapScalar *tap,
                                  std::vector<double> &out) {
                out.assign(width, 0.0);
                for (int i = 0; i < width; ++i) out[i] = tap[i].comp * invI;
            };
            // Magnitude of the residual VECTOR: canonical non-overlapping
            // stride-2 pair, lattice-aligned, held over the pair.
            auto envelope = [&](const std::vector<double> &v,
                                std::vector<double> &out) {
                out.assign(width, 0.0);
                for (int i = sp0; i + 1 < width; i += 2) {
                    const double e = std::hypot(v[i], v[i + 1]);
                    out[i] = e; out[i + 1] = e;
                }
                if (sp0 == 1 && width > 1) out[0] = out[1];
            };
            std::vector<double> bp0, bpN, res, env;
            carrierRow(t0, bp0);
            // The fold is ASKED of the grammar per side; nothing here assumes
            // a relation from the distance.
            auto foldSide = [&](const CombTapScalar *tap, bool have, int dLine,
                                std::vector<double> &dst) {
                if (!have) return;
                const CombCarrierGrammar *gN =
                    carrierGrammarLine(lineNumber + dLine);
                if (!tgGram || !gN) return;
                const int h0 = videoParameters.activeVideoStart;
                const lddecode::CarrierPhaseRelation rel =
                    lddecode::carrierGrammarSignedPhaseRelation(
                        tgGram, h0, gN, h0);
                double sgn;
                if (rel == lddecode::CarrierPhaseRelation::Opposite) sgn = +1.0;
                else if (rel == lddecode::CarrierPhaseRelation::Same) sgn = -1.0;
                else return;
                carrierRow(tap, bpN);
                res.assign(width, 0.0);
                for (int i = 0; i < width; ++i) res[i] = bp0[i] + sgn * bpN[i];
                envelope(res, env);
                for (int i = 0; i < width; ++i)
                    dst[i] = std::max(dst[i], env[i]);
            };
            tapLine.vReachResid1IRE.assign(width, 0.0);
            tapLine.vReachResid2IRE.assign(width, 0.0);
            tapLine.vReachResid2UpIRE.assign(width, 0.0);
            tapLine.vReachResid2DownIRE.assign(width, 0.0);
            tapLine.vReachResid4IRE.assign(width, 0.0);
            foldSide(tapLine.tapU1.data(), tapLine.haveU1, -1, tapLine.vReachResid1IRE);
            foldSide(tapLine.tapD1.data(), tapLine.haveD1, +1, tapLine.vReachResid1IRE);
            foldSide(tU2, haveU2, -2, tapLine.vReachResid2IRE);
            foldSide(tD2, haveD2, +2, tapLine.vReachResid2IRE);
            // The same folds kept apart.  The combined row above stays for
            // the readers that already have it; admission uses these.
            foldSide(tU2, haveU2, -2, tapLine.vReachResid2UpIRE);
            foldSide(tD2, haveD2, +2, tapLine.vReachResid2DownIRE);
            foldSide(tU4, tapLine.haveU4, -4, tapLine.vReachResid4IRE);
            foldSide(tD4, tapLine.haveD4, +4, tapLine.vReachResid4IRE);
            // Lateral cornering: second difference of the CARRIER-FREE luma
            // at stride 2, taken from the coarse row this same pass already
            // published rather than re-derived.
            for (int i = 0; i < width; ++i) {
                const int m  = std::clamp(i - 2, 0, width - 1);
                const int p2 = std::clamp(i + 2, 0, width - 1);
                tapLine.lateralCornerIRE[i] = std::fabs(
                    outCoarse0[i] - 0.5 * (outCoarse0[m] + outCoarse0[p2]));
            }
            tapLine.triggerFacts1Valid = tapLine.haveU1 || tapLine.haveD1;
        }

        // ---- Narrow-notch coarse (centre row only) ----------------------
        // A luma question wants luma specificity, so the carrier concern is
        // secondary here and the notch must be the NARROWEST kind. The
        // two-stride form used elsewhere, 0.5 * (raw[-2] + raw[+2]), never
        // reads the centre at all: at a transition it interpolates ACROSS
        // the feature it is being asked to report. This is the Y election's
        // plane-5 notch instead, which keeps the centre at half weight over
        // the same support, so the value belongs to THIS sample.
        //
        // The reduction is a MEDOID over the four complete carrier cycles
        // the sample is a member of, not a mean of them. The mean above
        // (centeredCarrierCycle4Mean) is a fifth, centred construction that
        // is none of the memberships, and it blends in cycles skewed by an
        // outlier sitting at their far edge. Choosing one membership leaves
        // no mixing, just a preferred coarse. Within three samples of the
        // active edge a sample has fewer than four complete memberships, so
        // only the complete ones vote.
        auto narrowNotchAt = [&](const CombTapScalar *tap, int i)->double {
            const int c  = std::clamp(i,     0, width - 1);
            const int cm = std::clamp(i - 2, 0, width - 1);
            const int cp = std::clamp(i + 2, 0, width - 1);
            return 0.25 * (tap[cm].raw + 2.0 * tap[c].raw + tap[cp].raw);
        };
        {
            double *outNotch0 = tapLine.notchCoarse0IRE.data();
            for (int rel = 0; rel < width; ++rel) {
                double cycles[4];
                int nCycles = 0;
                for (int k = 0; k < 4; ++k) {
                    const int a = rel - 3 + k;
                    if (a < 0 || a + 3 > width - 1) continue;
                    cycles[nCycles++] =
                        0.25 * (narrowNotchAt(t0, a) +
                                narrowNotchAt(t0, a + 1) +
                                narrowNotchAt(t0, a + 2) +
                                narrowNotchAt(t0, a + 3));
                }
                outNotch0[rel] =
                    (nCycles > 0 ? coarseCycleMedoid(cycles, nCycles)
                                 : narrowNotchAt(t0, rel)) * invI;
            }
        }

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

void Comb::FrameBuffer::computeFieldALine(const CombTapLine &tapLine,
                                           double *outFieldLine,
                                           double *outGate)
{
    const int width = tapLine.width;
    if (width <= 0 || !outFieldLine || (int)tapLine.tap0.size() < width)
        return;

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
    for (int rel = 0; rel < width; ++rel) {
        const double C    = tapLine.tap0[rel].comp;
        const double Cup2 = tapLine.tapU2[rel].comp;
        const double Cdn2 = tapLine.tapD2[rel].comp;
        const double Cup4 = tapLine.tapU4[rel].comp;
        const double Cdn4 = tapLine.tapD4[rel].comp;

        // Construction and safety have separate jobs. Field A keeps the sharp near-leg
        // estimator; only hard source/grammar legality removes an operand here. The ±4
        // contour shapes the value directly instead of throttling the filter at ordinary
        // edges.
        const double reachUp2 = tapLine.pairU2[rel].reachLegalGate;
        const double reachDn2 = tapLine.pairD2[rel].reachLegalGate;

        double wUp2 = tapLine.pairU2[rel].weight * reachUp2;
        double wDn2 = tapLine.pairD2[rel].weight * reachDn2;

        double sc2 = 1.0;

        if ((wUp2 > 0.0) || (wDn2 > 0.0)) {
            if (wDn2 > 3.0 * wUp2)      wUp2 = 0.0;
            else if (wUp2 > 3.0 * wDn2) wDn2 = 0.0;

            const double denom = wUp2 + wDn2;
            if (denom > 1e-9) {
                sc2 = 2.0 / denom;
                if (sc2 < 1.0) sc2 = 1.0;
                sc2 = 1.0;
            } else {
                wUp2 = 0.0;
                wDn2 = 0.0;
            }
        }

        double tc = 0.0;
        const bool combed = (wUp2 > 0.0 || wDn2 > 0.0);

        if (combed) {
            const CombTapContour &curve = tapLine.contour[rel];
            auto refineNearWithFar = [](double nearS,
                                        double farS,
                                        double influence) {
                if (influence <= 0.0 || nearS == 0.0)
                    return nearS;
                if ((nearS > 0.0) != (farS > 0.0))
                    return nearS;

                const double nearMag = std::fabs(nearS);
                const double farMag = std::fabs(farS);
                const double mag =
                    (nearMag + influence * farMag) / (1.0 + influence);
                return std::copysign(mag, nearS);
            };

            const double Cup2Adj =
                refineNearWithFar(Cup2, Cup4, curve.upInfluence);
            const double Cdn2Adj =
                refineNearWithFar(Cdn2, Cdn4, curve.dnInfluence);

            double t2 = (C - Cup2Adj) * wUp2 * sc2;
            t2 += (C - Cdn2Adj) * wDn2 * sc2;
            tc = 0.25 * t2;
        } else {
            tc = C;
        }
        
        if (!std::isfinite(tc))
            tc = C;
        
        outFieldLine[rel] = tc;
        
        double gateA = std::max(wUp2, wDn2);
        gateA = std::clamp(gateA, 0.0, 1.0);
        if (outGate) outGate[rel] = gateA;
    }

    // FieldAStats per-line logging is disabled; active diagnostics remain concise.
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
// Sharp adaptive three-line constructor: center against same-field ±2. Pair
// similarity chooses one or both legs; a 3:1 decision avoids averaging a weak,
// laterally displaced partner into a sharp edge. The normalized half-difference
// preserves full comb authority for admitted operands.
static bool ldcdFieldBClean()
{
    static const bool v = []{
        const char *e = std::getenv("LDCD_FIELDB_CLEAN");
        return e && std::atoi(e) != 0;
    }();
    return v;
}

// Refusal threshold on a leg's own measured chroma step.
static double ldcdFieldBCleanStepIRE()
{
    static const double v = []{
        const char *e = std::getenv("LDCD_FIELDB_CLEAN_STEP");
        return e ? std::atof(e) : 11.0;
    }();
    return v;
}

// ISOLATION HARNESS.  The clean baseline admits a leg on grammar LEGALITY
// alone; LDCD_FIELDB_CLEAN_ADD names ONE mechanism to add on top, so each can
// be judged by itself rather than inside the stack that grew around it.
// Broad band cede, revive, and 3x leg-selection policies are deliberately not
// offered here; this isolates the per-column mechanisms.
static bool ldcdFieldBCleanAdd(const char *who)
{
    static const std::string spec = []{
        const char *e = std::getenv("LDCD_FIELDB_CLEAN_ADD");
        return std::string(e ? e : "");
    }();
    if (spec.empty()) return false;
    return spec.find(who) != std::string::npos;
}

void Comb::FrameBuffer::computeFieldBLineClean(const CombTapLine &tapLine,
                                               double *outFieldLine,
                                               std::uint8_t *outReasonLine)
{
    const int width =
        videoParameters.activeVideoEnd - videoParameters.activeVideoStart;
    const int lineNumber = tapLine.cacheLine;
    if (width <= 0 || !outFieldLine) return;

    const bool haveU = tapLine.haveU2 &&
        static_cast<int>(tapLine.tapU2.size()) >= width;
    const bool haveD = tapLine.haveD2 &&
        static_cast<int>(tapLine.tapD2.size()) >= width;
    const bool have4 =
        static_cast<int>(tapLine.coarseU4IRE.size()) >= width &&
        static_cast<int>(tapLine.coarseD4IRE.size()) >= width &&
        static_cast<int>(tapLine.tapU4.size()) >= width &&
        static_cast<int>(tapLine.tapD4.size()) >= width;

    const double stepIRE = ldcdFieldBCleanStepIRE();
    constexpr double kSlopeTolIRE = 3.0;
    constexpr double kGradMinIRE  = 1.0;
    constexpr double kBaseIRE = 6.0;
    constexpr double kKappa = 0.45;
    constexpr double kLumaHardBreakIRE = 14.0;
    const bool addRegion    = ldcdFieldBCleanAdd("region");
    const bool addStep      = ldcdFieldBCleanAdd("step");
    const bool addBoundary  = ldcdFieldBCleanAdd("boundary");
    const bool addLumaBreak = ldcdFieldBCleanAdd("lumabreak");
    const bool haveCoarse =
        tapLine.coarseLumaValid &&
        static_cast<int>(tapLine.coarse0IRE.size()) >= width &&
        static_cast<int>(tapLine.coarseU2IRE.size()) >= width &&
        static_cast<int>(tapLine.coarseD2IRE.size()) >= width;
    // The lurch step gate the boundary bound rides on: a published service,
    // read here, not rebuilt.
    std::vector<float> hgAt(width, 0.0f);
    for (const LurchStepRun &run : corroborateLurchEdges(lineNumber)) {
        if (run.suppressed) continue;
        const double gate = std::clamp(run.gate, 0.0, 1.0);
        if (gate <= 0.0) continue;
        const float hg = static_cast<float>(run.stepAbsIRE * gate);
        const int xa = std::max(0, run.a);
        const int xb = std::min(width - 1, run.b + 3);
        for (int x = xa; x <= xb; ++x) hgAt[x] = std::max(hgAt[x], hg);
    }
    using RR = CombContentReach::RegionRelation;
    const CombContentReach::IntrafieldRegionReach unknownRegion;
    std::uint8_t *bandOut = chromaBoundaryBand_line(lineNumber);

    for (int rel = 0; rel < width; ++rel) {
        const double center = tapLine.tap0[rel].comp;
        const auto &region =
            rel < static_cast<int>(tapLine.intrafieldRegionReach.size())
                ? tapLine.intrafieldRegionReach[rel] : unknownRegion;
        if (bandOut) bandOut[rel] = region.chromaBoundaryBand ? 1 : 0;

        // Per-side admission reads the measured evidence directly. Promoted region
        // verdicts are not used to refuse the opposite side.
        const double bound = kBaseIRE + kKappa * hgAt[rel];
        auto sideOk = [&](bool have, bool legal, RR measured, double diffIRE,
                          double coarseNeighbour) {
            if (!have || !legal) return false;
            if (addRegion && measured == RR::DifferentRegion) return false;
            if (addStep && diffIRE >= stepIRE) return false;
            if (addBoundary && measured == RR::DifferentRegion &&
                diffIRE >= bound) return false;
            if (addLumaBreak && haveCoarse && measured != RR::AlienCancel &&
                std::fabs(tapLine.coarse0IRE[rel] - coarseNeighbour) >=
                    kLumaHardBreakIRE) return false;
            return true;
        };
        const bool upOk = sideOk(haveU,
            haveU && tapLine.pairU2[rel].reachLegalGate > 0.0,
            region.upMeasured, region.upDifferenceIRE,
            haveCoarse ? tapLine.coarseU2IRE[rel] : 0.0);
        const bool downOk = sideOk(haveD,
            haveD && tapLine.pairD2[rel].reachLegalGate > 0.0,
            region.downMeasured, region.downDifferenceIRE,
            haveCoarse ? tapLine.coarseD2IRE[rel] : 0.0);

        double output = center;
        std::uint8_t reason = FieldBReasonCenter;

        if (upOk || downOk) {
            lddecode::CombReachCancelLeg legs[4];
            int n = 0;
            if (upOk)   legs[n++] = { -2 };
            if (downOk) legs[n++] = {  2 };
            if (have4 && !(upOk && downOk)) {
                if (tapLine.haveU4) legs[n++] = { -4 };
                if (tapLine.haveD4) legs[n++] = {  4 };
            }
            const lddecode::CombReachCancelPlan plan =
                combReachIndex.planCancel(
                    lineNumber, videoParameters.activeVideoStart + rel,
                    legs, n);
            if (plan.valid) {
                bool take = true;
                if (plan.leg4 != 0 && (plan.leg4 % 4) == 0 && have4) {
                    const double c0  = tapLine.coarse0IRE[rel];
                    const double cL2 = (plan.leg2 > 0) ? tapLine.coarseD2IRE[rel]
                                                       : tapLine.coarseU2IRE[rel];
                    const double cL4 = (plan.leg4 > 0) ? tapLine.coarseD4IRE[rel]
                                                       : tapLine.coarseU4IRE[rel];
                    const double gradient = c0 - cL4;
                    const bool nearSide = (plan.leg4 * plan.leg2) > 0;
                    const double predicted = nearSide ? 0.5 * gradient
                                                      : -0.5 * gradient;
                    const double mismatch = std::fabs((cL2 - c0) - predicted);
                    take = std::fabs(gradient) >= kGradMinIRE &&
                           (nearSide || mismatch < kSlopeTolIRE);
                }
                if (take) {
                    auto legValue = [&](int off) {
                        switch (off) {
                            case -2: return tapLine.tapU2[rel].comp;
                            case  2: return tapLine.tapD2[rel].comp;
                            case -4: return tapLine.tapU4[rel].comp;
                            case  4: return tapLine.tapD4[rel].comp;
                        }
                        return 0.0;
                    };
                    const double v = plan.wCenter * center +
                                     plan.wLeg2 * legValue(plan.leg2) +
                                     plan.wLeg4 * legValue(plan.leg4);
                    if (std::isfinite(v)) {
                        output = v;
                        reason = (upOk && downOk) ? FieldBReasonBlend
                                                  : FieldBReasonOneLeg;
                    }
                } else if (upOk != downOk) {
                    // Gradient refused: the plain one-sided cancel stands.
                    const double leg = upOk ? tapLine.tapU2[rel].comp
                                            : tapLine.tapD2[rel].comp;
                    output = 0.5 * (center - leg);
                    reason = FieldBReasonOneLeg;
                }
            }
        } else {
            reason = FieldBReasonCede;
        }

        if (!std::isfinite(output)) { output = center; reason = FieldBReasonCenter; }
        outFieldLine[rel] = output;
        if (outReasonLine) outReasonLine[rel] = reason;
    }
}

void Comb::FrameBuffer::computeFieldBLine(const CombTapLine &tapLine,
                                          double *outFieldLine,
                                          std::uint8_t *outReasonLine)
{
    if (ldcdFieldBClean()) {
        computeFieldBLineClean(tapLine, outFieldLine, outReasonLine);
        return;
    }
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

    // Region admission has a measured noise floor plus the luma-leak carried
    // by a corroborated lateral step.  This is a binary operand test, not a
    // strength control: admitted samples retain the original filter exactly.
    constexpr double kBaseIRE = 6.0;
    constexpr double kKappa = 0.45;
    std::vector<float> hgAt(width, 0.0f);
    for (const LurchStepRun &run : corroborateLurchEdges(lineNumber)) {
        if (run.suppressed) continue;
        const double gate = std::clamp(run.gate, 0.0, 1.0);
        if (gate <= 0.0) continue;
        const float hg = static_cast<float>(run.stepAbsIRE * gate);
        const int xa = std::max(0, run.a);
        const int xb = std::min(width - 1, run.b + 3);
        for (int x = xa; x <= xb; ++x)
            hgAt[x] = std::max(hgAt[x], hg);
    }

    const float *centerRepairStrength =
        locked1DParallaxRepairStrength_line(lineNumber);

    // At ±2 the grammar relation is Opposite, so legal carrier should cancel.
    // vReachResid2IRE measures the remainder the fold cannot explain; above the IRE
    // threshold that operand no longer satisfies the comb premise. The threshold is
    // material-independent because the residual is expressed in IRE.
    // LDCD_FIELDB_LEGACY_GATES=1 selects the alternate region-gate diagnostic path.
    static const double kResidCedeIRE = []{
        const char *e = std::getenv("LDCD_FIELDB_RESID_IRE");
        return e ? std::atof(e) : 3.0;
    }();
    static const bool residLaw = []{
        const char *e = std::getenv("LDCD_FIELDB_LEGACY_GATES");
        return !(e && std::atoi(e) != 0);
    }();
    const bool haveResidUp =
        static_cast<int>(tapLine.vReachResid2UpIRE.size()) >= width;
    const bool haveResidDown =
        static_cast<int>(tapLine.vReachResid2DownIRE.size()) >= width;

    using RR = CombContentReach::RegionRelation;
    const CombContentReach::IntrafieldRegionReach unknownRegion;

    // THE CARRIER LICENSE APPLIES TO FIELD B'S AVERAGING HALF.
    //
    // Field B first estimates a neighbour from the admitted legs, then cancels that
    // estimate from center. A leg whose band energy is luma-owned should not be
    // averaged into the carrier estimate. The license therefore scales leg weights
    // before the 3:1 selection. It is operand selection, not a cede or a correction
    // strength control: a weak leg yields to its partner, and center is used only when
    // both leg weights vanish.
    const float *licUpRow = carrierLicense_line(tapLine.lnU2);
    const float *licDnRow = carrierLicense_line(tapLine.lnD2);
    const bool licenseOnComb = ldcdCombLicenseEnabled() && licUpRow && licDnRow;

    // Publish band membership for downstream band-uniform laws (Y election).
    std::uint8_t *bandOut = chromaBoundaryBand_line(lineNumber);

    for (int rel = 0; rel < width; ++rel) {
        const double center = tapLine.tap0[rel].comp;
        const double up = haveU ? tapLine.tapU2[rel].comp : 0.0;
        const double down = haveD ? tapLine.tapD2[rel].comp : 0.0;

        const auto &region =
            rel < static_cast<int>(tapLine.intrafieldRegionReach.size())
                ? tapLine.intrafieldRegionReach[rel] : unknownRegion;

        const bool upLegal =
            haveU && tapLine.pairU2[rel].reachLegalGate > 0.0;
        const bool downLegal =
            haveD && tapLine.pairD2[rel].reachLegalGate > 0.0;

        const double bound = kBaseIRE + kKappa * hgAt[rel];
        const bool upBoundary =
            region.up == RR::DifferentRegion &&
            region.upDifferenceIRE >= bound;
        const bool downBoundary =
            region.down == RR::DifferentRegion &&
            region.downDifferenceIRE >= bound;

        // A real carrier-free vertical context break is also binary.  The
        // AlienCancel relation is exempt because raw-identical content shares
        // the center's leak and is precisely what the comb should cancel.
        constexpr double kLumaHardBreakIRE = 14.0;
        const bool haveCoarse =
            tapLine.coarseLumaValid &&
            rel < static_cast<int>(tapLine.coarse0IRE.size()) &&
            rel < static_cast<int>(tapLine.coarseU2IRE.size()) &&
            rel < static_cast<int>(tapLine.coarseD2IRE.size());
        const bool upLumaBreak =
            haveCoarse && region.up != RR::AlienCancel &&
            std::fabs(tapLine.coarse0IRE[rel] -
                      tapLine.coarseU2IRE[rel]) >= kLumaHardBreakIRE;
        const bool downLumaBreak =
            haveCoarse && region.down != RR::AlienCancel &&
            std::fabs(tapLine.coarse0IRE[rel] -
                      tapLine.coarseD2IRE[rel]) >= kLumaHardBreakIRE;

        const bool bandCede = region.chromaBoundaryBand;
        // ADMISSION ONLY. Residual-law evidence decides whether each operand is available;
        // the leg-selection rule below is independent.
        const bool residCedeUp =
            residLaw && haveResidUp &&
            tapLine.vReachResid2UpIRE[rel] >= kResidCedeIRE;
        const bool residCedeDown =
            residLaw && haveResidDown &&
            tapLine.vReachResid2DownIRE[rel] >= kResidCedeIRE;
        const bool upAdmitted = residLaw
            ? (upLegal && !residCedeUp)
            : (upLegal && !upBoundary && !upLumaBreak && !bandCede);
        const bool downAdmitted = residLaw
            ? (downLegal && !residCedeDown)
            : (downLegal && !downBoundary && !downLumaBreak && !bandCede);

        // Field B is a sharp three-line estimator. Content-region evidence changes only
        // operand admission; it never scales an admitted leg or the resulting
        // half-difference.
        double wUp = upAdmitted ? tapLine.pairU2[rel].weight : 0.0;
        double wDown = downAdmitted ? tapLine.pairD2[rel].weight : 0.0;
        if (licenseOnComb) {
            wUp   *= std::clamp((double)licUpRow[rel], 0.0, 1.0);
            wDown *= std::clamp((double)licDnRow[rel], 0.0, 1.0);
        }

        // When one local match is more than three times stronger, select that leg rather
        // than averaging a laterally displaced weaker partner into the edge. This is
        // selection, not a strength throttle.
        if (wDown > 3.0 * wUp)
            wUp = 0.0;
        else if (wUp > 3.0 * wDown)
            wDown = 0.0;

        bool recovered = false;
        if (wUp <= 0.0 && wDown <= 0.0) {
            const double dMag =
                std::fabs(std::fabs(up) - std::fabs(down));
            const double sumUD = std::fabs(up + down);
            constexpr double kReviveCarrierFloorIRE = 2.0;
            if (sumUD * invIreScale > kReviveCarrierFloorIRE &&
                dMag - std::fabs(sumUD * 0.2) <= 0.0)
            {
                wUp = upAdmitted ? 1.0 : 0.0;
                wDown = downAdmitted ? 1.0 : 0.0;
                recovered = wUp > 0.0 || wDown > 0.0;
            }
        }

        // When both operands remain legal but demonstrably belong to
        // different regions, use the one closer to center rather than mixing
        // a carrier no participating line contained.
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

        // Publish the same Gilgol/region fact for downstream band and election
        // laws; the constructor has used it only as the hard admission above.
        if (bandOut) bandOut[rel] = region.chromaBoundaryBand ? 1 : 0;

        const bool useUp = wUp > 1e-9;
        const bool useDown = wDown > 1e-9;

        double output;
        std::uint8_t reason;
        if (!useUp && !useDown) {
            output = center;
            reason = (residCedeUp || residCedeDown ||
                      bandCede || upBoundary || downBoundary ||
                      upLumaBreak || downLumaBreak)
                ? FieldBReasonCede : FieldBReasonCenter;
        } else {
            const double denom = wUp + wDown;
            const double neighbor = (up * wUp + down * wDown) / denom;
            output = 0.5 * (center - neighbor);
            reason = recovered
                ? FieldBReasonRecovery
                : ((useUp && useDown)
                    ? FieldBReasonBlend : FieldBReasonOneLeg);

        }

        if (!std::isfinite(output)) {
            output = center;
            reason = FieldBReasonCenter;
        }

        // Certified Pass 1.5 repairs are source authority at this sample; recombining
        // them with unrepaired legs would reinstate the rejected component.
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
// Frame A is a plain reach-gated midpoint comb.  The preclean stage has
// already sharpened each field and placed all three operands in the same IQ
// coordinate.  Reach decides which complementary observations are safe; it
// does not scale the midpoint itself.
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

    outFrameIQ.assign(width, std::complex<double>(0.0, 0.0));
    if (line < first || line >= last ||
        line >= demodLines || demodWidth <= 0 ||
        (int)centerIQ.size() < width ||
        (int)upIQ.size() < width ||
        (int)dnIQ.size() < width)
        return;

    const bool verticalAllowed = carrierFrameVerticalAllowed(line);
    const bool haveUpLine = verticalAllowed && (line - 1 >= first);
    const bool haveDnLine = verticalAllowed && (line + 1 < last);

    for (int x = 0; x < width; ++x) {
        const std::complex<double> center = centerIQ[x];
        double upReach = 0.0;
        double dnReach = 0.0;
        if (reachTapLine &&
            x < (int)reachTapLine->pairU1.size() &&
            x < (int)reachTapLine->pairD1.size())
        {
            if (haveUpLine)
                upReach = std::clamp(
                    reachTapLine->pairU1[x].reachLegalGate, 0.0, 1.0);
            if (haveDnLine)
                dnReach = std::clamp(
                    reachTapLine->pairD1[x].reachLegalGate, 0.0, 1.0);
        }

        const double legWeight = upReach + dnReach;
        if (legWeight <= 0.0) {
            outFrameIQ[x] = center;
            continue;
        }

        const std::complex<double> complement =
            (upIQ[x] * upReach + dnIQ[x] * dnReach) / legWeight;
        outFrameIQ[x] = 0.5 * (center + complement);
    }
}
// Frame A: reach-gated midpoint comb fed by the Field B preclean ring.
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

    // Frame A reads only the shared physical reach facts.  The tap line is
    // already built with TapBuildFrame, so ensure is a cache hit.
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
        // uncovered frames retain the ordinary 1D observation.  Both branches
        // are rel-indexed: the locked export already is, and the bucket row
        // takes +left to match.  (Frame B is locked-only -- needFrameBCompute
        // requires phaseCompensation -- so the second branch is a guard, not
        // a path.)
        return configuration.phaseCompensation
            ? locked1DSource_line(ln)
            : bucketScalar1D_line(ln) + left;
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

// Frame B performs sharp signed cancellation of image-locked colour from the
// precleaned 1D center. The registered ±1 pair estimates the alien term; Frame B
// subtracts that estimate without blending either neighbour into the picture.
// Frame A owns the interfield midpoint candidate.
//
// THE DIAGONAL FACT IS MEASURED ONCE BEFORE CANDIDATE CONSTRUCTION.
// measureDiagonalAdvanceLine finds the same signed lateral crossing on center and
// both ±2 same-field neighbours and requires monotonic progression through the
// rows. Raster-locked carrier residue does not shift its crossing from row to row,
// so it cannot create a diagonal advance.
//
// The published advance is in samples per frame line. crossingUp and crossingDown
// span four frame-line steps, so their total displacement is divided by four.
// Frame B rounds that per-line advance to the integer shift used by its ±1 pair.
// advance > 0 aligns the up leg at x-d and the down leg at x+d.
void Comb::FrameBuffer::measureDiagonalAdvanceLine(int line)
{
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;
    if (width <= 0) return;

    if ((int)scratch_fbDiagAdvance.size() != width) {
        scratch_fbDiagAdvance.assign(width, 0.0);
        scratch_fbDiagStrength.assign(width, 0.0);
    } else {
        std::fill(scratch_fbDiagAdvance.begin(),
                  scratch_fbDiagAdvance.end(), 0.0);
        std::fill(scratch_fbDiagStrength.begin(),
                  scratch_fbDiagStrength.end(), 0.0);
    }

    const CombTapLine &tapLine = ensureCombTapLine(line);
    if ((int)tapLine.notchCoarse0IRE.size() < width ||
        (int)tapLine.coarseU2IRE.size() < width ||
        (int)tapLine.coarseD2IRE.size() < width ||
        (int)tapLine.hLumaDeltaIRE.size() < width)
        return;

    const auto &T = configuration.tunables;
    const double edgeThreshIRE = T.FIELD_LUMA_EDGE_THRESH_IRE;
    constexpr int kPlateauSamples = 5;
    constexpr int kPlateauSearchMax = 16;
    constexpr double kPlateauJitterMaxIRE = 1.2;
    const double stepThresholdIRE = std::max(2.0, 0.9 * edgeThreshIRE);
    const double seedThreshIRE = 0.75 * edgeThreshIRE;

    // Everything here is already in IRE, so no rescaling is needed: the
    // election converts because its rows are composite, not because the
    // measurement wants composite units.
    const double *src = tapLine.notchCoarse0IRE.data();
    const double *rowU = tapLine.coarseU2IRE.data();
    const double *rowD = tapLine.coarseD2IRE.data();

    auto findPlateauInner = [&](int from, int dir) -> int {
        for (int step = 1; step <= kPlateauSearchMax; ++step) {
            const int inner = from + dir * step;
            const int outer = inner + dir * (kPlateauSamples - 1);
            if (inner < 0 || inner >= width) break;
            if (outer < 0 || outer >= width) break;
            const int b = std::min(inner, outer);
            const int e = std::max(inner, outer);
            double lo = src[b], hi = lo;
            for (int r = b + 1; r <= e; ++r) {
                lo = std::min(lo, src[r]);
                hi = std::max(hi, src[r]);
            }
            if (hi - lo <= kPlateauJitterMaxIRE) return inner;
        }
        return -1;
    };

    // A plateau is reduced by CHOOSING one of its samples, never by averaging
    // them, so the reference is a value that was actually observed.
    auto medoidRange = [&](const double *row, int b, int e) {
        double buf[kPlateauSamples];
        int n = 0;
        for (int r = b; r <= e && n < kPlateauSamples; ++r) buf[n++] = row[r];
        return coarseCycleMedoid(buf, n);
    };

    int x = 0;
    while (x < width) {
        if (tapLine.hLumaDeltaIRE[x] < seedThreshIRE) { ++x; continue; }
        const int seedBegin = x;
        while (x + 1 < width && tapLine.hLumaDeltaIRE[x + 1] >= seedThreshIRE)
            ++x;
        const int seedEnd = x;
        ++x;

        // No settled region on a side means this was texture, not a
        // transition between two things. Decline.
        const int leftPlateauEnd = findPlateauInner(seedBegin, -1);
        const int rightPlateauBegin = findPlateauInner(seedEnd, +1);
        if (leftPlateauEnd < 0 || rightPlateauBegin < 0) continue;
        const int runBegin = leftPlateauEnd - (kPlateauSamples - 1);
        const int runEnd   = rightPlateauBegin + (kPlateauSamples - 1);
        if (runBegin < 0 || runEnd >= width) continue;

        const double sourceLeft  = medoidRange(src, runBegin, leftPlateauEnd);
        const double sourceRight = medoidRange(src, rightPlateauBegin, runEnd);
        const double sourceDelta = sourceRight - sourceLeft;
        if (std::fabs(sourceDelta) < stepThresholdIRE) continue;

        auto crossingFromRow = [&](const double *row) -> double {
            double lbuf[kPlateauSamples], rbuf[kPlateauSamples];
            int ln = 0, rn = 0;
            for (int r = runBegin; r <= leftPlateauEnd && ln < kPlateauSamples; ++r)
                lbuf[ln++] = row[r];
            for (int r = rightPlateauBegin; r <= runEnd && rn < kPlateauSamples; ++r)
                rbuf[rn++] = row[r];
            const double lo = coarseCycleMedoid(lbuf, ln);
            const double hi = coarseCycleMedoid(rbuf, rn);
            const double delta = hi - lo;
            // Same step, same SIGN, or it is not the same edge.
            if (std::fabs(delta) < stepThresholdIRE ||
                delta * sourceDelta <= 0.0)
                return std::numeric_limits<double>::quiet_NaN();
            double prevT = (row[leftPlateauEnd] - lo) / delta;
            for (int r = leftPlateauEnd + 1; r <= rightPlateauBegin; ++r) {
                const double t = (row[r] - lo) / delta;
                if (prevT < 0.5 && t >= 0.5) {
                    const double frac = std::clamp(
                        (0.5 - prevT) / std::max(1e-12, t - prevT), 0.0, 1.0);
                    return (r - 1) + frac;
                }
                prevT = t;
            }
            return std::numeric_limits<double>::quiet_NaN();
        };

        const double c0 = crossingFromRow(src);
        const double cU = crossingFromRow(rowU);
        const double cD = crossingFromRow(rowD);
        if (!std::isfinite(c0) || !std::isfinite(cU) || !std::isfinite(cD))
            continue;

        const double upStep = c0 - cU;
        const double downStep = cD - c0;
        const double totalShift = std::fabs(cD - cU);
        const bool monotone = upStep * downStep >= -0.20;
        const bool local = std::fabs(upStep) <= 4.0 &&
                           std::fabs(downStep) <= 4.0;
        if (!monotone || !local || totalShift < 0.75) continue;

        const double strength = std::clamp((totalShift - 0.75) / 3.25, 0.0, 1.0);
        const double advance = (cD - cU) / 4.0;
        for (int r = runBegin; r <= runEnd; ++r) {
            scratch_fbDiagAdvance[r] = advance;
            scratch_fbDiagStrength[r] = strength;
        }
    }
}

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
	// The diagonal fact first: the registration below reads it instead of
	// searching for a shift of its own.
	measureDiagonalAdvanceLine(line);

	if ((int)scratch_frameBReachUnsafe.size() != width)
		scratch_frameBReachUnsafe.assign(width, 0);
	else
		std::fill(scratch_frameBReachUnsafe.begin(),
		          scratch_frameBReachUnsafe.end(), 0);

	// The refusal band is election evidence, not a normal Frame B construction
	// actuator. Frame B publishes full registered subtraction plus
	// scratch_frameBReachUnsafe; FVF can disqualify that candidate over the transition
	// support. LDCD_FB_BAND_CEDE=1 enables pair cede as a diagnostic A/B mode.
	static const bool frameBBandCede = []{
		const char *e = std::getenv("LDCD_FB_BAND_CEDE");
		return e && std::atoi(e) != 0;
	}();
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

    // Close Frame B's refusal band over the aperture that can consume it.
    // A boundary verdict on line L protects more than a Frame-B candidate
    // centred on L: candidates centred on L-1 and L+1 would otherwise still
    // take L as one of their operands and pull that region into the picture
    // on the other side.  Those are exactly the dark-band reaches visible at
    // a skin / narrow-band / turquoise transition.
    //
    // The horizontal band has already been dilated for registration.  Taking
    // the OR of the adjacent centre-line bands here is the corresponding
    // vertical dilation by Frame B's physical +/-1 reach -- no farther.  The
    // three cache slots are line-mod-3, so current/up/down remain resident
    // together while this candidate is built.
    const CombTapLine *reachTapUp = haveUpLine
        ? &ensureCombTapLine(line - 1) : nullptr;
    const CombTapLine *reachTapDn = haveDnLine
        ? &ensureCombTapLine(line + 1) : nullptr;
    auto frameBBandAt = [](const CombTapLine *tap, int x) {
        return tap && tap->interfieldRegionValid &&
               x >= 0 && x < (int)tap->interfieldRegionReach.size() &&
               tap->interfieldRegionReach[x].chromaBoundaryBand;
    };

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

    const double cancelStrength =
        std::clamp(std::max(0.0, T.FRAME_B_COMB_STRENGTH), 0.0, 1.0);

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
    // the Frame-B estimator sees them.  This cancels the carrier-product
    // image without the h-0.5 delay of a previous/current average.
    centerCarrierProductRowInPlace(scratch_centerIQ.data(), width);
    centerCarrierProductRowInPlace(scratch_upIQ.data(), width);
    centerCarrierProductRowInPlace(scratch_dnIQ.data(), width);

    // Preserve the shared luma-impulse observation consumed later by the
    // attribution/election path.  It is evidence only and does not soften or
    // throttle Frame B's signed correction.
    scratch_impulseExempt.resize(width);
    // The luma-impulse evidence reads the solved platform too: the block
    // scaffold cannot resolve an impulse shorter than its own block, which is
    // exactly what this observation is looking for.
    const double *frameLuma0 = nullptr;
    if (lockedLumaCacheValid) {
        frameLuma0 = ldcdReachUsesSolvedLuma() ? lockedLumaSolved_line(line)
                                               : nullptr;
        if (!frameLuma0) frameLuma0 = lockedLumaSmooth_line(line);
    }
    for (int x = 0; x < width; ++x) {
        scratch_impulseExempt[x] = compactLumaExcursionEvidence(
            frameLuma0, x, width, invIreScale);
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
    // center and publishes that sharp corrected center.  σ comes from grammar
    // lineFlip polarity — a scalar-domain fact, which is the part of alignment the
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
        if ((int)scratch_fbReg.size() < width)
            scratch_fbReg.resize(width);

        static constexpr double kWin[7] = {0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5};
        constexpr double kWinSum = 6.0;
        constexpr double kRegMargin = 1.08;

        // LDCD_FB_IQ_REG=1 selects the precleaned-IQ argmin registration for diagnostic
        // A/B comparison with the luma-geometry registration.
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

        // Working rows carry the registration window overhang: index j in
        // [-3, width + 2].
        constexpr int kRowPad = 3;
        const int rowWidth = width + 2 * kRowPad;

        // Leg roles are line-level facts (alienSign is per line), so the
        // Same/Opposite selection and the deviation rows the registration
        // search reads at every column hoist out of the pixel loop:
        // g_s[j] = same[j + s] - center[j] for s in [-2, 2].
        const bool sameIsUp = (alienSign > 0.0);
        const std::complex<double> *pSame = sameIsUp ? pU : pD;
        std::complex<double> *devRows[5] = {nullptr, nullptr, nullptr,
                                            nullptr, nullptr};
        // The precleaned-IQ argmin rows are required only when LDCD_FB_IQ_REG=1.
        if (haveSignedAlien && iqRegistration) {
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

        // THE LOCATOR IS LUMA GEOMETRY, NOT A CARRIER/IQ SEARCH.
        //
        // measureDiagonalAdvanceLine publishes a monotone crossing progression across the
        // ±2 rows. The registration consumes that measured advance and rounds it to the
        // integer aim justified by a ±1 comb. Raster-locked carrier residue has the same
        // horizontal phase on each row and therefore contributes zero crossing advance.
        //
        // The adopted aim is clamped to |d| <= 1 because the detector's maximum published
        // advance is one sample per frame line. d = 0 is the detent whenever no qualifying
        // diagonal transition is present.
        //
        // One implementation supplies both candidate construction and diagnostics so the
        // registration fact has a single producer.
        auto diagAim = [&](int x) -> int {
            if (x < 0 || x >= (int)scratch_fbDiagAdvance.size()) return 0;
            const long r = std::lround(scratch_fbDiagAdvance[x]);
            return (int)std::clamp<long>(r, -1, 1);
        };


        for (int x = 0; x < width; ++x) {
            if (!haveSignedAlien) {
                scratch_fbPairDiff[x] = std::complex<double>(0.0, 0.0);
                scratch_fbReg[x] = 0;
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
            double devMag[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
            if (iqRegistration) {
                for (int si = 0; si < 5; ++si) {
                    const std::complex<double> *g = devRows[si];
                    std::complex<double> devAcc(0.0, 0.0);
                    for (int k = -3; k <= 3; ++k)
                        devAcc += kWin[k + 3] * g[x + k];
                    devMag[si] = cmag(devAcc) / kWinSum;
                }
            }

            // THE AIM. Normal operation reads the per-line diagonal fact; LDCD_FB_IQ_REG=1
            // selects the precleaned-IQ argmin diagnostic. bestSi is derived from the adopted
            // shift so registration metadata and the correction waveform use the same aim.
            int bestSi;
            if (!iqRegistration) {
                // FRAME B'S NORMAL AIM reads the diagonal fact measured before candidate
                // construction. The aligned pair is (up[x-d], dn[x+d]), matching the published
                // advance convention.
                const int bestD = diagAim(x);
                const int sStarDiag = sameIsUp ? -bestD : bestD;
                bestSi = std::clamp(sStarDiag + 2, 0, 4);
            } else {
                // A/B mode (LDCD_FB_IQ_REG=1): use the precleaned-IQ argmin.
                bestSi = 2;
                double bestDev = devMag[2] / kRegMargin;
                for (int si = 0; si < 5; ++si) {
                    if (si == 2) continue;
                    if (devMag[si] < bestDev) {
                        bestDev = devMag[si];
                        bestSi = si;
                    }
                }
            }

            const int sStar = bestSi - 2;
            // The pin overrides every aim above, including the certified one:
            // the question it asks is whether leaving the column is the fault
            // at all, and a fact-grade off-column aim is still off-column.
            const int d = ldcdFrameBPinColumn()
                ? 0
                : (sameIsUp ? -sStar : sStar);

			// REACH EVIDENCE (see frameBBandSeed).
			//
			// Frame B constructs Zsame - Zopp at full strength. If the registered aperture
			// touches a refusal band, scratch_frameBReachUnsafe publishes that fact to FVF.
			// Progressive FVF can give the support to Frame A; interlace FVF can disqualify
			// Frame B while retaining its field alternatives.
			//
			// There is no one-leg refusal rung because refusing Same and refusing Opposite
			// produce physically different estimators. LDCD_FB_BAND_CEDE=1 is the only mode
			// that cedes both legs together. The 0.5 combine coefficient is the reciprocal of
			// the sigma fold, not a leg-count-dependent gain.
            bool upAdmit = true;
            bool dnAdmit = true;
            const bool refusePair =
                frameBBandAt(&reachTapLine, x) ||
                frameBBandAt(reachTapUp, x) ||
                frameBBandAt(reachTapDn, x);
			scratch_frameBReachUnsafe[x] = refusePair ? 1 : 0;
			if (frameBBandCede && refusePair) {
                upAdmit = false;
                dnAdmit = false;
            }
			// Without LDCD_FB_BAND_CEDE, both registered legs stand and the unsafe bit remains
			// separate election evidence.
			//
			// Registration may use a multi-axis aperture, but the correction waveform is
			// sampled pointwise at the registered coordinates so narrow vertical carrier
			// structure is not spread laterally.
            const std::complex<double> pairDiff =
                (upAdmit ? pU[x - d] : pC[x]) -
                (dnAdmit ? pD[x + d] : pC[x]);
            scratch_fbPairDiff[x] = pairDiff;
            scratch_fbReg[x] = d;


        }
    }

    for (int x = 0; x < width; ++x) {
        const bool useLockedCenter = forceFrameBLocked1D;

        const std::complex<double> &Z0 = scratch_centerIQ[x];
        const std::complex<double> &Z0Preclean = Z0;

        const bool haveUpSignal =
            haveUpLine && precleanUp;

        const bool haveDnSignal =
            haveDnLine && precleanDn;

        // Subtract the registered pair-difference alien estimate from center.
        // Reach supplies physical legality; no picture-forming blend follows.
        std::complex<double> Zc = Z0;

        if (!useLockedCenter && haveSignedAlien &&
            haveUpSignal && haveDnSignal &&
            x < (int)reachTapLine.pairU1.size() &&
            x < (int)reachTapLine.pairD1.size())
        {
            const double pairLegalGate = std::min(
                std::clamp(reachTapLine.pairU1[x].reachLegalGate, 0.0, 1.0),
                std::clamp(reachTapLine.pairD1[x].reachLegalGate, 0.0, 1.0));
			if (pairLegalGate <= 0.0)
				scratch_frameBReachUnsafe[x] = 1;

            if (pairLegalGate > 0.0) {
                const std::complex<double> corr =
                    ((0.5 * alienSign) * scratch_fbPairDiff[x]) *
                    (cancelStrength * pairLegalGate);
                if (std::isfinite(corr.real()) && std::isfinite(corr.imag()))
                    Zc = Z0 - corr;
            }
        }

        std::complex<double> Zout = Zc;

        // Frame B publishes the corrected center directly.  Its ±1 rows are
        // evidence for the alien term, never picture material to average into
        // the result; the plain midpoint is Frame A's distinct candidate.
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

    }

}

// ---------------------------------------------------------------------------
// Frame C: covered-frame comp-line bootstrap.
//
// Certified def lines are conservation facts and cede to center. Comp lines
// between them are bootstrapped toward the def lines rather than entered into the
// uncovered-frame election. Frame C is a plain ±1 comb whose legs are the
// certified defs' published 1D sources.
//
// Frame C is not a scored candidate and does not use Frame A/B correlation,
// reach, license, or delta-cap policy. With both certified legs it uses p = 0.67;
// with one certified leg it uses the midpoint p = 0.5:
//     out = (C + E) + p*(C - (C + E)) = C + (1 - p)*E
//
// Composite scalars retain each line's carrier orientation, so relationSign folds
// each leg into the center line's orientation before averaging.
//
// Returns false when Frame C does not own the line; the caller then follows the
// ordinary uncovered-frame path. LDCD_FRAME_C=0 disables this bootstrap.
// ---------------------------------------------------------------------------
bool Comb::FrameBuffer::computeFrameCLine(int line, std::vector<double> &out)
{
    static const bool frameCOn = []{
        const char *e = std::getenv("LDCD_FRAME_C");
        return !(e && std::atoi(e) == 0);
    }();
    if (!frameCOn) return false;
    if (!configuration.phaseCompensation) return false;
    if (certifiedOneDLevel() < 2) return false;
    if (!frameHasExactCoverage()) return false;
    if (certifiedDefLine(line)) return false;

    const int first = videoParameters.firstActiveFrameLine;
    const int last  = videoParameters.lastActiveFrameLine;
    const int left  = videoParameters.activeVideoStart;
    const int right = videoParameters.activeVideoEnd;
    const int width = right - left;
    if (width <= 0 || line < first || line >= last) return false;
    if (line >= demodLines || demodWidth <= 0) return false;

    const double *sC = combSource1D_line(line);
    const auto *g0 = carrierGrammarLine(line);
    if (!sC || !g0) return false;

    auto legSource = [&](int ln, double &sign) -> const double * {
        if (ln < first || ln >= last) return nullptr;
        if (!certifiedDefLine(ln)) return nullptr;
        const auto *g = carrierGrammarLine(ln);
        if (!g) return nullptr;
        sign = (g->lineFlip == g0->lineFlip) ? 1.0 : -1.0;
        return combSource1D_line(ln);
    };

    double sgnU = 0.0, sgnD = 0.0;
    const double *sU = legSource(line - 1, sgnU);
    const double *sD = legSource(line + 1, sgnD);
    if (!sU && !sD) return false;

    constexpr double kFrameCPull = 0.67;
    constexpr double kFrameCPullOneSided = 0.5;
    const double p = (sU && sD) ? kFrameCPull : kFrameCPullOneSided;

    if ((int)out.size() < width) out.resize(width);
    for (int rel = 0; rel < width; ++rel) {
        const double c = sC[rel];
        const double target = (sU && sD)
            ? 0.5 * (sgnU * sU[rel] + sgnD * sD[rel])
            : (sU ? sgnU * sU[rel] : sgnD * sD[rel]);
        out[rel] = c + p * (target - c);
    }
    return true;
}

// 3D Section
// getCandidate - prescreen for 3D election
// 3D Section
// getCandidate - prescreen for 3D election
Comb::FrameBuffer::Candidate Comb::FrameBuffer::getCandidate(
    qint32 refLineNumber, qint32 refH,
    const FrameBuffer &frameBuffer, qint32 lineNumber, qint32 h,
    double adjustPenalty,
    const lddecode::CombReachReply *prefilledReach) const
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

    // Cross-frame ScalarSignCompare uses the mode's actual 1D scalar. Bucket and
    // locked 1D scalars are PhasePreservedCarrier, so grammar legality is identical;
    // the sample source follows the same mode switch.
    const lddecode::CombReachReply phaseReach = prefilledReach
        ? *prefilledReach
        : combReachIndex.queryAgainst(
            frameBuffer.combReachIndex,
            {refLineNumber,
             lineNumber,
             refH,
             h,
             lddecode::CombReachUse::ScalarSignCompare,
             scalarReachSource()});

    if (!phaseReach.allowScalarSignCompare) {
        result.penalty = 1000.0;
        result.yPen    = 1000.0;
        result.iqPen   = 0.0;
        return result;
    }
    // Only an Opposite relation yields a carrier cancellation. A Same
    // relation is an average of two different pictures, not a comb.
    if (phaseReach.carrierRelation != lddecode::CarrierPhaseRelation::Opposite) {
        result.penalty = 1000.0;
        result.yPen    = 1000.0;
        result.iqPen   = 0.0;
        return result;
    }
    constexpr double relationSign = 1.0;

    const int hh = clampH(h);

    // 1D sample: locked mode reads the anchored-aware source, falling back to the
    // phase-corrected blind bandpass when no anchored plane exists; bucket mode reads
    // clpbuffer[0]. relationSign folds the carrier relation so consumers use the same
    // Opposite-form math.
    const double *lockedRow = frameBuffer.configuration.phaseCompensation
        ? frameBuffer.combSource1D_line(lineNumber) : nullptr;

    if (lockedRow && (hh - left) >= 0 && (hh - left) < (right - left)) {
        result.sample = relationSign * lockedRow[hh - left];
    } else {
        result.sample = relationSign * frameBuffer.bucketScalar1D_line(lineNumber)[hh];
    }

    // --- Luma Penalty with Neighbor Shaping ---
    //
    // getCandidate computes luma-domain evidence from reconstructed
    // Y = raw - 2D chroma/composite estimate over a five-point spatial cross:
    //   center line: x-1, x, x+1
    //   vertical:    y-1, y+1 at x
    // getBestCandidate consumes result.yPen directly.
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
    // The similarity metric uses the coarse luma platform because a legal four-sample
    // mean cancels carrier exactly. Lurch sharpening improves edge placement while
    // retaining the smoothing useful for a similarity metric, so per-frame noise does
    // not dominate content change.
    //
    // Where no coarse platform exists, the bucket path uses its reconstructed-luma
    // fallback.
    const int coarseLeft = videoParameters.activeVideoStart;
    const int coarseW    = demodWidth;
    auto coarseRow = [](const FrameBuffer &fb, int line) -> const double * {
        if (!fb.lockedLumaCacheValid) return nullptr;
        if (!fb.lockedLumaSolved_flat.empty())
            return fb.lockedLumaSolved_line(line);
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
