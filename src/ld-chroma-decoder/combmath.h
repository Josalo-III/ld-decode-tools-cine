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
// CARRIER LANES — the two coordinate streams a 4fSC carrier plane really is.
//
// A carrier-band filter on a 4fSC grid has taps at EVEN offsets only, so it
// never mixes the two sample lattices. A plane it produced is therefore not
// one waveform: it is TWO INDEPENDENT COLOUR-COORDINATE STREAMS AT 2fSC,
// interleaved. Consecutive samples of one lane are 2 samples of 4fSC apart,
// hence 180 degrees, so de-alternating by (-1)^(h>>1) -- the same expression
// for both parities -- recovers each stream directly.
//
// Writing c[x] = I*cos(a+33deg) + Q*sin(a+33deg) with a advancing 90deg per
// sample and theta = a(0) + 33deg, the de-alternated lanes are
//
//     laneA =  I*cos(theta) + Q*sin(theta)
//     laneB = -I*sin(theta) + Q*cos(theta)
//
// i.e. THE LANES ARE THE COLOUR COORDINATES ROTATED BY theta, and nothing
// else. No information is created or destroyed and no sample moves, so the
// decomposition is exactly invertible.
//
// WHY THIS PRIMITIVE EXISTS. The 4fSC demod already in this decoder computes
// envI[x] = 2*c[x]*sin4fsc(ph) and envQ[x] = 2*c[x]*cos4fsc(ph). Because
// sin4fsc is {1,0,-1,0}, envI is NON-ZERO ONLY ON EVEN ph and equals twice
// the de-alternated even lane; envQ likewise carries the odd lane. They are
// the lanes with zeros punched into the other parity, and the low-pass that
// follows is doing two jobs at once: enforcing the bandwidth law AND
// interpolating across those zeros. Entangling them is why the law's kernel
// shape could not be changed independently -- sharpen the law and you alter
// the interpolation, relax the interpolation and you alter the law.
//
// Working in lanes separates the two, and the separation is visible in the
// coefficients rather than being a matter of interpretation. Split the 9-tap
// envelope kernel by tap parity:
//
//   even offsets  0.0021  0.0903  0.3153  0.0903  0.0021   sum 0.5001
//   odd offsets       0.0191  0.2308  0.2308  0.0191       sum 0.4998
//
// The EVEN taps land on real lane samples: that is the law. The ODD taps land
// on the punched zeros: that is the interpolation. (The near-exact 0.5/0.5
// split is also where the demod's factor of 2 comes from.) Renormalised, the
// law a lane actually receives is the 5-tap
//
//   [0.004199, 0.180564, 0.630474, 0.180564, 0.004199]
//
// which reproduces the full-grid response through the passband to within
// 0.08 dB (-2.21 vs -2.26 at 1.30 MHz, -2.93 vs -3.01 at 1.50).
//
// So a full-grid kernel whose taps are even-only IS a lane kernel with every
// other tap taken, and a lane kernel needs about half the taps for that
// mechanical reason -- not because a sample rate changed underneath. (A lane
// does carry one sample per two of 4fSC, but nothing here rests on saying so,
// and 2fSC is numerically the 4fSC grid's own Nyquist, which invites exactly
// the rate/Nyquist confusion this note is written to avoid.)
//
// THE DECOMPOSITION BOTTOMS OUT AT TWO, and not by convention: there are two
// lanes because the carrier is a two-component object. Four samples carry
// both coordinates twice, once positive and once negative; de-alternation
// undoes the sign and parity separates the coordinates, which exhausts the
// structure. Splitting a lane again is ordinary decimation with nothing
// behind it -- two streams with no distinction, half the samples of one
// coordinate discarded. The licence is spent after one use.
//
// hypot(c[x], c[x+1]) is the same conflation in the other direction: a sound
// magnitude, but it discards the lane split, which is real evidence -- on the
// certified carrier both lanes are separately FACT.
//
// NOTE ON AXES: these are LANES, not I and Q. Recovering the colour axes
// needs the rotation by theta above, and theta is a property of the line's
// burst. This primitive deliberately stops short of naming the axes; a
// caller that needs I/Q must supply the angle, because handing a lane to a
// per-axis law under an axis name is the one error neither can detect.
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

// Exact inverse of decomposeCarrierLanes for the same (h0, width).
inline void recomposeCarrierLanes(const std::vector<double> &laneA,
                                  const std::vector<double> &laneB,
                                  int h0, int width, double *carrier)
{
    if (!carrier || width <= 0) return;
    size_t ia = 0, ib = 0;
    for (int i = 0; i < width; ++i) {
        const int h = h0 + i;
        const bool even = ((h & 1) == 0);
        const std::vector<double> &lane = even ? laneA : laneB;
        size_t &idx = even ? ia : ib;
        if (idx >= lane.size()) return;      // caller mismatched the geometry
        carrier[i] = lane[idx++] * carrierLaneSign(h);
    }
}

// Rotate a lane pair into a colour-axis pair, and back. theta is the angle
// the lanes are rotated by (see the derivation above); pass the line's own
// value. These are the only places an axis convention enters.
inline void laneToAxis(double a, double b, double cosT, double sinT,
                       double &axWide, double &axNarrow)
{
    axWide   = a * cosT - b * sinT;
    axNarrow = a * sinT + b * cosT;
}
inline void axisToLane(double axWide, double axNarrow,
                       double cosT, double sinT, double &a, double &b)
{
    a =  axWide * cosT + axNarrow * sinT;
    b = -axWide * sinT + axNarrow * cosT;
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
// alternating 2fSC image.  A previous/current average rejects that image only
// by placing the result at h-0.5; applying it at carrier phase h then mixes
// two different composite coordinates.  The symmetric binomial
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

// Rotate common 4fsc IQ back into the line-local locked frame.
inline void fourfscToLocked(double i4fsc, double q4fsc,
                            double bcos, double bsin,
                            double &iLocked, double &qLocked)
{
    iLocked = i4fsc * bcos - q4fsc * bsin;
    qLocked = i4fsc * bsin + q4fsc * bcos;
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

inline void eig2_sym(const double S[2][2], double &l1, double &l2, double V[2][2]);

// General 2x2 polar/affine helpers. The retired residual-Y estimator no longer
// calls these, but they remain available as representation math; removing that
// one policy path is not a reason to erase the general primitive.
inline void clamp_rotation_gain_shear(double R[2][2], double U[2][2],
                                      double phaseMaxRad, bool allowGain,
                                      double gMin, double gMax, double shearMax)
{
    double phase = std::atan2(R[1][0], R[0][0]);
    if (std::fabs(phase) > phaseMaxRad) {
        const double p = (phase < 0.0 ? -phaseMaxRad : phaseMaxRad);
        const double c = std::cos(p), s = std::sin(p);
        R[0][0] = c; R[0][1] = -s; R[1][0] = s; R[1][1] = c;
    }

    double l1, l2, V[2][2];
    eig2_sym(U, l1, l2, V);
    double s1 = std::max(0.0, l1), s2 = std::max(0.0, l2);
    double g = 0.5 * (s1 + s2);
    const double shear =
        (g > 1e-12) ? std::fabs(s1 - s2) / g : 0.0;

    if (shear > shearMax && (s1 > 0.0 || s2 > 0.0)) {
        const double target = g * shearMax;
        const double avg = 0.5 * (s1 + s2);
        s1 = avg + 0.5 * target;
        s2 = avg - 0.5 * target;
        const double VD[2][2] = {
            {V[0][0] * s1, V[0][1] * s2},
            {V[1][0] * s1, V[1][1] * s2}
        };
        U[0][0] = VD[0][0] * V[0][0] + VD[0][1] * V[0][1];
        U[0][1] = VD[0][0] * V[1][0] + VD[0][1] * V[1][1];
        U[1][0] = VD[1][0] * V[0][0] + VD[1][1] * V[0][1];
        U[1][1] = VD[1][0] * V[1][0] + VD[1][1] * V[1][1];
        g = 0.5 * (s1 + s2);
    }

    if (!allowGain) g = 1.0;
    else g = std::clamp(g, gMin, gMax);
    R[0][0] *= g; R[0][1] *= g;
    R[1][0] *= g; R[1][1] *= g;
}

inline void mat2_mul(const double A[2][2], const double B[2][2],
                     double C[2][2])
{
    C[0][0] = A[0][0] * B[0][0] + A[0][1] * B[1][0];
    C[0][1] = A[0][0] * B[0][1] + A[0][1] * B[1][1];
    C[1][0] = A[1][0] * B[0][0] + A[1][1] * B[1][0];
    C[1][1] = A[1][0] * B[0][1] + A[1][1] * B[1][1];
}

inline void mat2_T_mul(const double A[2][2], const double B[2][2],
                       double C[2][2])
{
    C[0][0] = A[0][0] * B[0][0] + A[1][0] * B[1][0];
    C[0][1] = A[0][0] * B[0][1] + A[1][0] * B[1][1];
    C[1][0] = A[0][1] * B[0][0] + A[1][1] * B[1][0];
    C[1][1] = A[0][1] * B[0][1] + A[1][1] * B[1][1];
}

inline bool mat2_inv(const double M[2][2], double Minv[2][2])
{
    const double det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    if (std::fabs(det) < 1e-12) return false;
    const double inv = 1.0 / det;
    Minv[0][0] =  M[1][1] * inv; Minv[0][1] = -M[0][1] * inv;
    Minv[1][0] = -M[1][0] * inv; Minv[1][1] =  M[0][0] * inv;
    return true;
}

inline void eig2_sym(const double S[2][2], double &l1, double &l2,
                     double V[2][2])
{
    const double a = S[0][0], b = S[0][1], d = S[1][1];
    const double tr = a + d;
    const double det = a * d - b * b;
    const double rt = std::sqrt(std::max(0.0, tr * tr / 4.0 - det));
    l1 = tr / 2.0 + rt;
    l2 = tr / 2.0 - rt;
    if (std::fabs(b) > 1e-12) {
        V[0][0] = l1 - d; V[1][0] = b;
        V[0][1] = l2 - d; V[1][1] = b;
    } else {
        V[0][0] = 1.0; V[1][0] = 0.0;
        V[0][1] = 0.0; V[1][1] = 1.0;
    }
    for (int j = 0; j < 2; ++j) {
        const double n = boundedMag(V[0][j], V[1][j]);
        if (n > 1e-12) {
            V[0][j] /= n;
            V[1][j] /= n;
        }
    }
}

inline void sym_inv_sqrt(const double S[2][2], double Sminushalf[2][2])
{
    double l1, l2, V[2][2];
    eig2_sym(S, l1, l2, V);
    const double d1 = (l1 > 1e-12) ? 1.0 / std::sqrt(l1) : 0.0;
    const double d2 = (l2 > 1e-12) ? 1.0 / std::sqrt(l2) : 0.0;
    const double VD[2][2] = {
        {V[0][0] * d1, V[0][1] * d2},
        {V[1][0] * d1, V[1][1] * d2}
    };
    Sminushalf[0][0] = VD[0][0] * V[0][0] + VD[0][1] * V[0][1];
    Sminushalf[0][1] = VD[0][0] * V[1][0] + VD[0][1] * V[1][1];
    Sminushalf[1][0] = VD[1][0] * V[0][0] + VD[1][1] * V[0][1];
    Sminushalf[1][1] = VD[1][0] * V[1][0] + VD[1][1] * V[1][1];
}

inline void polar_decompose_2x2(const double A[2][2],
                                double R[2][2], double U[2][2])
{
    double AtA[2][2];
    mat2_T_mul(A, A, AtA);
    double AtA_mhalf[2][2];
    sym_inv_sqrt(AtA, AtA_mhalf);
    mat2_mul(A, AtA_mhalf, R);
    const double Rt[2][2] = {
        {R[0][0], R[1][0]},
        {R[0][1], R[1][1]}
    };
    mat2_mul(Rt, A, U);
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
