/******************************************************************************
 * combmath.h
 * ld-decode-tools TBC library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 Joseph Burns
 *
 * This file is part of ld-decode-tools.
 *
 * Shared mathematics for comb filtering and related demodulation operations.
 ******************************************************************************/

#ifndef COMBMATH_H
#define COMBMATH_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

#include "lddecodemetadata.h"

// 4fSC sine/cosine helpers shared across translation units.
static constexpr double sin4fsc_data_global[] = { 1.0, 0.0, -1.0, 0.0 };

inline double sin4fsc(int i) { return sin4fsc_data_global[i & 3]; }
inline double cos4fsc(int i) { return sin4fsc((i + 1) & 3); }

// ---------------------------------------------------------------------------
// Carrier lanes are the two interleaved coordinate streams preserved by an
// even-tap carrier-band filter. Consecutive samples in each lane are two
// 4fSC samples apart, so de-alternating by (-1)^(h>>1) recovers the stream.
//
// Writing c[x] = I*cos(a+33deg) + Q*sin(a+33deg) with a advancing 90deg per
// sample and theta = a(0) + 33deg, the de-alternated lanes are
//
//     laneA =  I*cos(theta) + Q*sin(theta)
//     laneB = -I*sin(theta) + Q*cos(theta)
//
// The lanes are the colour coordinates rotated by theta. No information is
// created or destroyed, and no sample moves, so the decomposition is
// invertible.
//
// The 4fSC demod computes
// envI[x] = 2*c[x]*sin4fsc(ph) and envQ[x] = 2*c[x]*cos4fsc(ph). Because
// sin4fsc is {1,0,-1,0}, envI is non-zero only on even ph and equals twice the
// de-alternated even lane; envQ likewise carries the odd lane. The low-pass
// following demodulation both enforces the bandwidth law and interpolates
// across the punched zeros.
//
// Splitting the 9-tap envelope kernel by tap parity gives:
//
//   even offsets  0.0021  0.0903  0.3153  0.0903  0.0021   sum 0.5001
//   odd offsets       0.0191  0.2308  0.2308  0.0191       sum 0.4998
//
// Even taps land on real lane samples; odd taps interpolate the punched zeros.
// Renormalised, the lane kernel is:
//
//   [0.004199, 0.180564, 0.630474, 0.180564, 0.004199]
//
// which reproduces the full-grid response through the passband to within
// 0.08 dB (-2.21 vs -2.26 at 1.30 MHz, -2.93 vs -3.01 at 1.50).
//
// A full-grid kernel with even-only taps is therefore a lane kernel with every
// other tap. The lane sample spacing is two samples on the 4fSC grid; this is
// a structural property of the interleaving, not a separate rate conversion.
//
// The decomposition has exactly two lanes because the carrier has two
// components. Splitting a lane again would be ordinary decimation and would
// discard half of one coordinate.
//
// hypot(c[x], c[x+1]) likewise discards the lane split. Both lanes are useful
// evidence and must remain separate.
//
// These are lanes, not I and Q. Recovering colour axes requires the rotation
// by theta, which is derived from the line burst. Callers that need I/Q supply
// that angle explicitly.
// ---------------------------------------------------------------------------

// De-alternation sign for absolute sample index h. Self-inverse.
inline double carrierLaneSign(int h) { return ((h >> 1) & 1) ? -1.0 : 1.0; }

// Split `carrier` (width samples starting at absolute index h0) into its two
// lanes. laneA collects even absolute indices, laneB odd. Non-finite samples
// are carried through, never patched: an absent sample is not a zero.
inline void decomposeCarrierLanes(const double *carrier, int h0, int width,
                                  std::vector<double> &laneA,
                                  std::vector<double> &laneB)
{
    laneA.clear();
    laneB.clear();
    if (!carrier || width <= 0) return;
    laneA.reserve((width + 1) / 2);
    laneB.reserve((width + 1) / 2);
    for (int i = 0; i < width; ++i) {
        const int h = h0 + i;
        const double d = carrier[i] * carrierLaneSign(h);
        ((h & 1) == 0 ? laneA : laneB).push_back(d);
    }
}

// Magnitude of a bounded 2-vector: direct sqrt, not std::hypot.  Comb
// magnitudes are video-domain quantities (sample/IRE scale) whose squares
// cannot over- or underflow a double, so hypot's IEEE range guarding is
// pure per-call cost on the per-pixel paths.
inline double boundedMag(double a, double b) { return std::sqrt(a * a + b * b); }
inline double boundedMag(const std::complex<double> &z) { return boundedMag(z.real(), z.imag()); }

// Integer-centred reconstruction of a carrier product stream.
//
// Product demodulation at 4fSC contains the wanted baseband vector plus an
// alternating 2fSC image.  The symmetric binomial
// aperture below has centroid exactly h and a zero at 2fSC.  Its gain of two
// preserves this decoder's full-signed-IQ convention (composite remodulation
// applies the reciprocal 0.5).  Apply it to I and Q independently; it is a
// registration/image-rejection primitive, not an I/Q merge and not a
// scalar-carrier replacement.
template <typename T>
inline T centeredCarrierProduct3(const T &previous,
                                 const T &current,
                                 const T &next)
{
    return previous * 0.5 + current + next * 0.5;
}

// In-place row form of centeredCarrierProduct3.  Reads each original sample
// before overwriting it and uses edge replication, matching the locked 1D
// producer's boundary convention.
template <typename T>
inline void centerCarrierProductRowInPlace(T *row, int width)
{
    if (!row || width <= 0)
        return;

    T previous = row[0];
    T current = row[0];
    for (int x = 0; x < width; ++x) {
        const T next = row[std::min(x + 1, width - 1)];
        row[x] = centeredCarrierProduct3(previous, current, next);
        previous = current;
        current = next;
    }
}

// One complete four-sample carrier cycle, registered at an integer sample.
// The half-weight endpoints are the same carrier phase, so together they
// contribute one ordinary phase sample.  Total weight is four and the
// centroid is the middle argument.
template <typename T>
inline T centeredCarrierCycle4Mean(const T &minus2,
                                   const T &minus1,
                                   const T &center,
                                   const T &plus1,
                                   const T &plus2)
{
    return (minus2 + plus2) * 0.125 +
           (minus1 + center + plus1) * 0.25;
}

// Choose one complete carrier-cycle estimate for a sample. A medoid returns an
// observed membership instead of averaging across memberships. Ties resolve
// to the lowest index, so the choice is deterministic.
inline double coarseCycleMedoid(const double *cycles, int count)
{
    if (count <= 0) return 0.0;
    int best = 0;
    double bestCost = std::numeric_limits<double>::infinity();
    for (int i = 0; i < count; ++i) {
        double cost = 0.0;
        for (int j = 0; j < count; ++j)
            cost += std::fabs(cycles[i] - cycles[j]);
        if (cost < bestCost) { bestCost = cost; best = i; }
    }
    return cycles[best];
}

// Weighted medoid: the same selection, with the voters carrying unequal
// weight.  argmin_i sum_j W_j * |c_i - c_j|.
//
// A member whose window straddles a luma change reports two parts of the
// picture. Its opinion should count for less, not be discarded, so weighting
// keeps all members while reducing the influence of that estimate.
//
// With every W equal this is exactly coarseCycleMedoid, so the unlurched case
// is not a special path.
inline double coarseCycleMedoidWeighted(const double *cycles,
                                        const double *w, int count)
{
    if (count <= 0) return 0.0;
    if (!w) return coarseCycleMedoid(cycles, count);
    int best = 0;
    double bestCost = std::numeric_limits<double>::infinity();
    for (int i = 0; i < count; ++i) {
        double cost = 0.0;
        for (int j = 0; j < count; ++j)
            cost += w[j] * std::fabs(cycles[i] - cycles[j]);
        if (cost < bestCost) { bestCost = cost; best = i; }
    }
    return cycles[best];
}

// Select the coarse estimate from the complete carrier cycles containing the
// sample. Four covering apertures and an optional moving coarse provide the
// candidates.
//
//   four offset apertures   starting at v in {x-3 .. x}, chord centres at
//                           x-1.5, x-0.5, x+0.5, x+1.5
//   the moving coarse       centeredCarrierCycle4Mean over [x-2 .. x+2],
//                           centroid exactly x
//
// Each candidate is a mean over one complete cycle. Candidates are not
// averaged together because their disagreement is the sub-block luma.
//
// They share one weakness: every window reaches four samples wide, so any of
// them may include picture from a very different part of the image. The medoid
// is the defence available to a selection -- it excludes the extreme outliers
// and returns the membership the others agree with.
//
// Five is also the right count: with an even set the medoid is degenerate --
// for sorted a<=b<=c<=d the costs of b and c are both c+d-a-b, identically --
// so the tie-break rather than the evidence would decide. An odd set has a
// strict winner. The moving coarse is the member that both breaks the tie and
// supplies a reading centred on the sample.
//
// Out-of-range members are skipped rather than clamped; a clamped duplicate
// would vote twice and bias the selection toward the edge.
//
// The moving coarse participates as one candidate but is not otherwise
// involved in selection. `trust`, when supplied, carries one weight per
// candidate slot in collection order: covering apertures v = x-3 .. x that
// are in range, followed by the moving coarse. Null means an unweighted vote.
inline double coveringCycleMedoid(const double *apMean, int x, int lastStart,
                                  bool haveMovingCoarse, double movingCoarse,
                                  const double *trust = nullptr)
{
    if (!apMean || lastStart < 0) return 0.0;
    double cycles[5];
    int count = 0;
    for (int k = 0; k < 4; ++k) {
        const int v = x - 3 + k;
        if (v < 0 || v > lastStart) continue;
        cycles[count++] = apMean[v];
    }
    if (haveMovingCoarse) cycles[count++] = movingCoarse;
    if (count == 0)
        return apMean[x < 0 ? 0 : (x > lastStart ? lastStart : x)];
    return coarseCycleMedoidWeighted(cycles, trust, count);
}

// Shared fractional-basis demod helpers. These are tiny math utilities used by
// both the locked demod path and candidate generation.
inline constexpr double CAL_EPS_SAMPLES = -0.07;

// Locked-path per-axis chroma gain lives in Comb::Configuration
// (gi_product / gq_product, comb.h).  combmath.h provides reusable
// arithmetic only; it does not own decoder product tuning.

// Locked-path hue basis split.  The total locked IQ->UV rotation is preserved,
// but the locked-specific correction is applied before the axis-specific FIRs
// so the final output step can keep the canonical 33 degree IQ->UV rotation.
inline constexpr double LOCKED_CHROMA_TOTAL_ROT_DEG = 70.0;
inline constexpr double LOCKED_CHROMA_OUTPUT_ROT_DEG = 33.0;
inline constexpr double LOCKED_CHROMA_PREFILTER_ROT_DEG =
    LOCKED_CHROMA_TOTAL_ROT_DEG - LOCKED_CHROMA_OUTPUT_ROT_DEG;

// Per-bucket carrier-excursion gain.  Indexed by carrierSampleClass & 3.
// Set to 1.0 (neutral): per-bucket magnitude asymmetry was shown to be
// scene-dependent and does not track the visible checkerboard artifact.
inline constexpr double CARRIER_BUCKET_GAIN[4] = { 1.0, 1.0, 1.0, 1.0 };

inline void basisCoeffs(double& Ce, double& Se)
{
    const double K = 0.5 * M_PI;
    Ce = std::cos(K * CAL_EPS_SAMPLES);
    Se = std::sin(K * CAL_EPS_SAMPLES);
}

// Basis projection for sample h shifted by ε (CAL_EPS_SAMPLES).
// sp = sin((h + ε) · π/2),  cp = cos((h + ε) · π/2)
inline void shiftedBasis(int h, double Ce, double Se, double& sp, double& cp)
{
    const int idx = (h & 3);
    const double s4 = sin4fsc(idx);
    const double c4 = cos4fsc(idx);
    sp = Ce * s4 + Se * c4;
    cp = Ce * c4 - Se * s4;
}

// Fuse burst rotation into the 4-phase locked basis:
//   ti = c * 2 * (sp*bcos - cp*bsin)
//   tq = c * 2 * (sp*bsin + cp*bcos)
// (where (bcos,bsin) is the per-line burst phasor).
inline void fusedDemodLUT(double bcos, double bsin,
                          const double spLUT[4], const double cpLUT[4],
                          double outTi[4], double outTq[4])
{
    for (int i = 0; i < 4; ++i) {
        const double sp = spLUT[i];
        const double cp = cpLUT[i];
        outTi[i] = 2.0 * (sp * bcos - cp * bsin);
        outTq[i] = 2.0 * (sp * bsin + cp * bcos);
    }
}

// Rotate line-local locked IQ into the common 4fsc frame.
inline void lockedTo4fsc(double iLocked, double qLocked,
                         double bcos, double bsin,
                         double &i4fsc, double &q4fsc)
{
    i4fsc = iLocked * bcos + qLocked * bsin;
    q4fsc = -iLocked * bsin + qLocked * bcos;
}

// Demodulate scalar composite already aligned to the common 4fsc grid.
inline void demod4fscFromComposite(double v, int h, double &i4fsc, double &q4fsc)
{
    i4fsc = v * sin4fsc(h) * 2.0;
    q4fsc = v * cos4fsc(h) * 2.0;
}

// Remodulate common 4fsc IQ back into composite sample space at position h.
inline double remod4fscToComposite(double i4fsc, double q4fsc, int h,
                                   double lineScale = 1.0)
{
    return lineScale * 0.5 * (i4fsc * sin4fsc(h) + q4fsc * cos4fsc(h));
}

// Remodulate using a grammar-derived phase index rather than a raw sample position.
inline double remod4fscToCompositePhase(double i4fsc, double q4fsc,
                                        int phase, double lineScale = 1.0)
{
    return lineScale * 0.5 * (i4fsc * sin4fsc(phase) + q4fsc * cos4fsc(phase));
}

// Remodulate common 4fsc IQ back into the shifted sample basis used by the
// burst-locked demod path. This preserves the fractional basis choice while
// staying explicit about the fact that the IQ itself is already canonical 4fsc.
inline double remod4fscToShiftedComposite(double i4fsc, double q4fsc, int h,
                                          const double spLUT[4], const double cpLUT[4],
                                          double lineScale = 1.0)
{
    const int idx = (h & 3);
    return lineScale * 0.5 * (i4fsc * spLUT[idx] + q4fsc * cpLUT[idx]);
}

// Remodulate line-local locked IQ back into composite sample space using the
// shifted locked basis. The seam is explicit: locked IQ is rotated once into
// common 4fsc, then remodulated on the chosen sample basis.
inline double remodLockedToShiftedComposite(double iLocked, double qLocked, int h,
                                            double bcos, double bsin,
                                            const double spLUT[4], const double cpLUT[4],
                                            double lineScale = 1.0)
{
    double i4fsc = 0.0, q4fsc = 0.0;
    lockedTo4fsc(iLocked, qLocked, bcos, bsin, i4fsc, q4fsc);
    return remod4fscToShiftedComposite(i4fsc, q4fsc, h, spLUT, cpLUT, lineScale);
}

// Small median-of-3 helper, used in several places.
inline double median3(double a, double b, double c)
{
    if (a > b) { if (b > c) return b; else if (a > c) return c; else return a; }
    else       { if (a > c) return a; else if (b > c) return c; else return b; }
}

// Colour burst measurement result for a single line.
// bcos/bsin: normalised unit-magnitude phasor (post-floor-clamp).
// carrierScale: raw burst magnitude before normalisation (in sample units).
struct BurstInfo {
    double bsin        = 0.0;
    double bcos        = 1.0;
    double carrierScale = 0.0;
};

// Measure the colour burst in the horizontal blanking interval to derive a
// normalised phasor (bcos, bsin). If floorEnable is true and the measured
// magnitude falls below floorFactor, the phasor is clamped to floorFactor
// so very noisy lines still produce a usable reference.
inline BurstInfo detectBurst(const quint16 *lineData,
                             const LdDecodeMetaData::VideoParameters &vp,
                             bool floorEnable,
                             double floorFactor)
{
    double bsin = 0.0, bcos = 0.0;
    for (int i = vp.colourBurstStart; i < vp.colourBurstEnd; ++i) {
        const double s = lineData[i];
        bsin += s * sin4fsc(i);
        bcos += s * cos4fsc(i);
    }
    const int len = vp.colourBurstEnd - vp.colourBurstStart;
    if (len > 0) { const double invLen = 1.0 / len; bsin *= invLen; bcos *= invLen; }
    const double carrierScale = std::sqrt(bsin * bsin + bcos * bcos);
    double mag = carrierScale;

    if (floorEnable && mag < floorFactor && mag > 1e-9) {
        const double s = floorFactor / mag;
        bsin *= s; bcos *= s; mag = floorFactor;
    }
    if (mag > 1e-9) { const double invMag = 1.0 / mag; bsin *= invMag; bcos *= invMag; }
    else { bsin = 0.0; bcos = 1.0; }
    return {bsin, bcos, carrierScale};
}

// Demodulation result for a single line: separated Y, I, Q arrays and the
// per-line burst phasor (bsin, bcos). Used by helper functions that return
// a fully separated line without writing into a FrameBuffer.
struct DemodResult {
    std::vector<double> Y;
    std::vector<double> I;
    std::vector<double> Q;
    double bsin = 0.0;
    double bcos = 1.0;
};

#endif // COMBMATH_H
