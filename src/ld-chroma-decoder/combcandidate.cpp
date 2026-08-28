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

// ---------------------------------------------------------------------------
// Frame B's ±1 operand admission (user, 2026-08-24).
//
// Frame A's good behaviour is inherent to being an average: its output cannot
// leave the hull of its operands, so a leg from the wrong chroma region costs
// half its error and no more. Frame B is the opposite estimator by design --
// FVF needs the choice -- and a subtractor leaves that hull as a matter of
// course. With centre at 0, a Same leg at +10 sitting in another region and
// the Opposite leg at 0, Frame B publishes -5: colour of a sign no
// participating line carried.
//
// The one thing that DOES port from Frame A is its separation of powers --
// "reach decides which complementary observations are safe; it does not scale
// the midpoint itself". So this is operand ADMISSION, hard, and never a
// strength scaler. Frame B has already falsified the continuous form: every
// gate it used to carry (ratioGate, rideGate, midLicense) multiplied the
// correction, and its own note records the outcome -- "a gate that lingers at
// half strength subtracts half the alien and leaves a parity-alternating
// residue: partial correction is the worst geometry".
//
// ONE floor governs both rungs below. They answer the same question, and a
// second threshold would only be a second thing to be wrong. It sits at twice
// Field B's evidence floor because ±1 partners are one TV line apart and share
// more content than the ±2 pair, so a separation that convicts at the
// intrafield step is not yet major here -- the same reasoning that puts
// FRAME_LUMA_EDGE_THRESH_IRE at 28 against Field's 18.
constexpr double kFrameBMajorIRE = 12.0;
constexpr double kFrameBRegionChromaFloorIRE = 5.0;

// Independent carrier-free luma evidence.  One differing leg is an ordinary
// coverage transition; two same-signed departures prove that center itself is
// outside the rational luma range of its vertical neighbourhood.  Keep each
// leg beyond the measured 4-5 IRE noise floor, while allowing the island to be
// asymmetric when the two departures together retain the former 2 * 6 IRE
// corroboration.  This catches a partial middle band without turning a moving
// one-sided edge into a Frame B cancellation outage.
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

// SYMMETRY LAW (author, 2026-08-25): "If same and opposite alternate per
// column, they cannot behave differently or the zippers are inevitable."
//
// The two legs cannot be refused INDIVIDUALLY, because refusing one is not the
// same operation as refusing the other and never can be. Only the Opposite leg
// carries the alien with a flipped sign, so it is the only leg that can expose
// it: refuse the Same leg and centre stands in for it exactly (centre is
// Same-relation to itself) and the correction keeps full strength; refuse the
// Opposite leg and there is no alien measurement left at all. Full correction
// versus none -- and which leg is refused varies per column, so adjacent
// columns alternate between the two. That alternation is the zipper, and it
// cannot be symmetrised away: the asymmetry is physical, not a policy choice.
//
// So this verdict is a BAND INPUT ONLY. It decides nothing on its own; it
// contributes to frameBBandSeed, where both legs are refused together and the
// operation is identical whichever leg fails. Measured on the beach: as a
// standalone rung it fired on 2.9% of columns, 43-69% of them at a 20-40 IRE
// lateral luma step -- a per-column asymmetry concentrated exactly along the
// garment edges where the zippers were seen.
inline bool frameBLegAdmitted(CombContentReach::RegionRelation measured,
                              double differenceIRE)
{
    return !(measured == CombContentReach::RegionRelation::DifferentRegion &&
             differenceIRE >= kFrameBMajorIRE);
}

// RELATION-LOCKING: what separates the alien from a chroma region.
//
// The author, 2026-08-24: "the error alternates hue, real chroma alternates
// phase" -- and of the cheek case, "not regional at all". A region evaluator
// cannot make that distinction, because it is built to answer "regional?" and
// the thing it must never convict is not regional at all. On the Opposite leg
// the two are literally the same measurement: centre reads (C + a) against
// (C - a), so no hue or magnitude threshold can separate a region break from
// the very alien the interfield comb exists to cancel.
//
// The separation this file already recorded is that the alien is RELATION-
// LOCKED: it "leaves the Same leg riding center (dSame ~ 0) while displacing
// the Opposite leg by twice the alien (dOpp ~ 2a)". Real chroma has no such
// preference. Measured at the columns being convicted, mean dSame/dOpp:
//
//     lateral luma delta   <5    5-10   10-20   20-40   40+
//     dSame/dOpp          0.83-  0.48-  0.24-   0.10-   0.08-
//                          1.00   0.67   0.41    0.24    0.11
//
// Flat material convicts symmetric differences -- real regions. Jennifer's
// cheek convicts at 10:1 -- the alien, every time.
//
// This is the measurement the retired ratioGate made. What was wrong with that
// gate was never the measurement but the ACTUATOR: it multiplied the
// correction, and a lingering half-strength subtraction "leaves a
// parity-alternating residue: partial correction is the worst geometry". Here
// the same fact admits or refuses an OPERAND, which is Frame A's separation of
// powers and the one thing that ports.
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
    // The alien is never a region boundary. A relation-locked triplet -- Same
    // leg riding centre while the Opposite leg is displaced by twice the alien
    // -- is the cross-colour this comb exists to cancel, whatever its hue and
    // magnitude say. This exemption used to sit on the per-leg rung; it lives
    // here now, because the band is the only refusal left and it must not be
    // the thing that stands the comb down at a luma transition.
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

double fieldContourGate(const CombContentReach::MovingCoarseContour &mc,
                        bool up)
{
    if (!mc.valid) return 1.0;
    const double trust = up ? mc.upTrust : mc.downTrust;
    return 0.25 + 0.75 * std::clamp(trust, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Retained physical record from the removed LDCD_PROBE_FRAMEB census.
//
// Frame B's job is to cancel vertically-invariant image-locked colour left by
// 1D without softening the picture.  Its registered signed pair difference
// estimates that alien directly; Frame A owns the separate midpoint model.
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
            const auto *analysis0  = carrierAnalysis_line(tapLine.ln0);
            const auto *analysisUp = carrierAnalysis_line(tapLine.lnU1);
            const auto *analysisDn = carrierAnalysis_line(tapLine.lnD1);
            auto trustAt = [](const lddecode::CarrierAnalysisRecord *row,
                              int rel) {
                return row
                    ? lddecode::carrierTrust(row[rel].carrierConformance,
                                             row[rel].conformanceUsableAxisFraction)
                    : 0.5;
            };

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
                        trustAt(analysis0, rel),
                        trustAt(analysisUp, rel),
                        trustAt(analysisDn, rel),
                        invI,
                        kFrameBRegionChromaFloorIRE);

                // Frame B's own seed replaces the evaluator's Field-B one on
                // this row.  Both name the same field because the dilation
                // below is the shared one; the LAW written into it is Frame
                // B's, and nothing else reads this row.
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
    for (int rel = 0; rel < width; ++rel) {
        const double C    = tapLine.tap0[rel].comp;
        const double Cup2 = tapLine.tapU2[rel].comp;
        const double Cdn2 = tapLine.tapD2[rel].comp;
        const double Cup4 = tapLine.tapU4[rel].comp;
        const double Cdn4 = tapLine.tapD4[rel].comp;

        // Construction and safety have separate jobs.  Field A keeps the
        // legacy near-leg estimator intact; only hard source/grammar legality
        // can remove an operand here.  The +/-4 contour below shapes the value
        // directly instead of throttling the whole filter at ordinary edges.
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
            } else {
                wUp2 = 0.0;
                wDn2 = 0.0;
            }
        } else {
            double dMag  = std::fabs(std::fabs(Cup2) - std::fabs(Cdn2));
            double sumUD = std::fabs(Cup2 + Cdn2);

            // The legacy agreement recovery remains, but it cannot override a
            // physically illegal leg and it cannot self-certify at the 1D
            // noise floor.
            constexpr double kReviveCarrierFloorIRE = 2.0; // 1D noise floor
            if (sumUD * invIreScale > kReviveCarrierFloorIRE &&
                dMag - std::fabs(sumUD * 0.2) <= 0.0) {
                wUp2 = reachUp2;
                wDn2 = reachDn2;
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

    // FieldAStats per-line logging remains removed; the per-line spam buried
    // the active diagnostics.
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
// Sharp adaptive three-line constructor: center against same-field +/-2.
// Pair similarity chooses one or both legs; the 3:1 decision preserves the
// legacy split2D edge shape, and the normalized half-difference preserves full
// comb authority.  Content-region facts act only as binary admission/cede
// decisions so unsafe preclean never reaches either Frame constructor.
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

    using RR = CombContentReach::RegionRelation;
    const CombContentReach::IntrafieldRegionReach unknownRegion;

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
        const bool upAdmitted =
            upLegal && !upBoundary && !upLumaBreak && !bandCede;
        const bool downAdmitted =
            downLegal && !downBoundary && !downLumaBreak && !bandCede;

        // Field B is the sharp legacy three-line estimator.  Content-region
        // evidence changes only operand admission; it never scales an admitted
        // leg or the resulting half-difference.
        double wUp = upAdmitted ? tapLine.pairU2[rel].weight : 0.0;
        double wDown = downAdmitted ? tapLine.pairD2[rel].weight : 0.0;

        // Legacy split2D's decisive shape: when one local match is more than
        // three times stronger, do not average a laterally displaced second
        // leg into the edge.  This is selection, not a strength throttle.
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
            reason = (bandCede || upBoundary || downBoundary ||
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

// Frame B: sharp signed cancellation of image-locked colour inherited from
// the precleaned 1D center.  The registered ±1 pair estimates the alien term;
// Frame B subtracts that estimate without blending either neighbour into the
// picture.  Frame A owns the interfield midpoint candidate.
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
	if ((int)scratch_frameBReachUnsafe.size() != width)
		scratch_frameBReachUnsafe.assign(width, 0);
	else
		std::fill(scratch_frameBReachUnsafe.begin(),
		          scratch_frameBReachUnsafe.end(), 0);

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

	// The refusal band is now election evidence, not a Frame-B construction
	// actuator.  Frame B's purpose is the aggressive registered subtraction;
	// suppressing it here merely republishes the contaminated center.  FVF sees
	// the exact aperture-level verdict through scratch_frameBReachUnsafe and can
	// give the whole transition run to Frame A.  Keep the former construction
	// cede as an explicit A/B mode only.
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

    // Raw bracket rows feed the carrier-free luma registration locator.
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
    const double *frameLuma0 = lockedLumaCacheValid
        ? lockedLumaSmooth_line(line) : nullptr;
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
            // bestSi is set from the adopted d so the correction waveform and
            // the registration fact always use the same shift.
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

			// REACH EVIDENCE (see frameBBandSeed).
			//
			// Frame B now always constructs Zsame - Zopp at full strength.
			// If this registered aperture touches a refusal band, the exact
			// fact is published separately for FVF: progressive gives Frame A
			// the unsafe pixel/run, while interlace disqualifies Frame B and
			// leaves its two field seats live.
			//
			// There is still no one-legged rung. Refusing a single leg is not
			// one operation but two different ones -- centre substitutes for
			// a refused SAME leg exactly and the correction keeps full
			// strength, while a refused OPPOSITE leg leaves no alien
			// measurement at all -- and which leg fails varies per column.
			// The old band cede therefore remains available only under
			// LDCD_FB_BAND_CEDE=1 for comparison; it is no longer the
			// production actuator.
            //
            // The coefficient never moves either: the 0.5 applied at the
            // combine is the reciprocal of the sigma FOLD, not a leg count.
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
			// Outside the opt-in legacy cede BOTH legs always stand. Per-leg
			// refusal remains forbidden; the unsafe bit above is the separate
			// election fact.

            // Registration needs a multi-axis aperture, but the correction waveform must retain
            // the sample that was measured.  Using the seven-tap average here
            // moved energy away from narrow vertical details: carrier-like
            // peaks were under-subtracted while adjacent columns received a
            // correction belonging to their neighbours.  Read the registered
            // pair pointwise; physical pair reach decides whether it applies.
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
// Frame C: the covered-frame comp-line bootstrap (user, 2026-08-22).
//
// Covered frames are handled unlike everything else in this file. The
// certified def lines are conservation fact: they run nothing, and cede to
// center in any system they are routed through. The comp lines between them
// are not an election problem -- they are bootstrapped toward the defs.
// Frame C is that bootstrap: a plain +-1 comb whose legs ARE the certified
// defs' own published 1D (combSource1D serves the exact fact where the
// anchored plane is live, the certified construction otherwise).
//
// It is deliberately not a candidate. No --two-d-variant access, no seat in
// scoreFieldVsFrame, and none of Frame A/B's trust machinery: correlation
// ramps, reach, licences and delta caps exist because a MODEL leg's carrier
// can lie, and a certified leg cannot. The uncovered midpoint law does not
// apply either: Frame A's 0.5 pull cap solves center C+E against partners
// C-E, where past-midpoint re-injects inverted alien. Against certified
// legs the partners carry C, so
//
//     out = (C + E) + p*(C - (C + E)) = C + (1 - p)*E
//
// and p past 0.5 is not overdrive -- it is the ratchet, driving the comp
// line only one way: toward the certified defs. p = 0.67 with both legs;
// a lone certified leg (frame edge) combs at the plain midpoint 0.5.
//
// Sign frame: composite scalars carry each line's raw carrier orientation,
// so a leg whose grammar lineFlip differs from center's holds matched
// chroma NEGATED; the relation sign folds it into center's orientation
// before the average (the intrafield sign-frame lesson, 2026-07-02).
//
// Returns false when it does not own the line (uncovered frame, def line,
// certified family off, no grammar, no certified leg); the caller proceeds
// to the ordinary election. LDCD_FRAME_C=0 is the A/B escape.
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
