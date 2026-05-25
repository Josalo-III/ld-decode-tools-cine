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
#include <cstdint>
#include <vector>

#include "lddecodemetadata.h"

// 4fSC sine/cosine helpers shared across translation units.
static constexpr double sin4fsc_data_global[] = { 1.0, 0.0, -1.0, 0.0 };

inline double sin4fsc(int i) { return sin4fsc_data_global[i & 3]; }
inline double cos4fsc(int i) { return sin4fsc((i + 1) & 3); }

// Shared fractional-basis demod helpers. These are tiny math utilities used by
// both the locked demod path and candidate generation.
inline constexpr double CAL_EPS_SAMPLES = -0.07;

// Global LO trim in degrees applied during burst rotation in phaseLocked().
// Negative values counteract a slight green bias. Zero = no trim.
inline constexpr double CAL_LO_ROT_DEG = 0.0;

// Locked-path per-axis chroma gain. GQ < 1.0 trims the Q axis to compensate
// for the slight chroma ellipse produced by the locked demod.
inline constexpr double GI_PRODUCT = 1.0;
inline constexpr double GQ_PRODUCT = 0.9;

// Bucket-smooth strength applied to locked1DSource in buildPhaseCorrected1D.
// Zero disables the smooth entirely (current default).
inline constexpr double FIELD_BUCKET_SMOOTH_STRENGTH = 0.0;

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

// Demodulate using a grammar-derived phase index rather than a raw sample position.
inline void demod4fscFromCompositePhase(double v, int phase, double &i4fsc, double &q4fsc)
{
    i4fsc = v * sin4fsc(phase) * 2.0;
    q4fsc = v * cos4fsc(phase) * 2.0;
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

// Polar-decompose a 2x2 affine matrix A = RU into a rotation R and symmetric U,
// then clamp R to a maximum phase rotation, clamp U's shear metric, and optionally
// clamp the gain (mean singular value). The clamped gain is folded into R so callers
// can apply a single matrix.
inline void clamp_rotation_gain_shear(double R[2][2], double U[2][2],
                                      double phaseMaxRad, bool allowGain,
                                      double gMin, double gMax, double shearMax)
{
    double phase = std::atan2(R[1][0], R[0][0]);
    if (std::fabs(phase) > phaseMaxRad) {
        double p = (phase < 0.0 ? -phaseMaxRad : phaseMaxRad);
        double c = std::cos(p), s = std::sin(p);
        R[0][0] = c; R[0][1] = -s; R[1][0] = s; R[1][1] = c;
    }

    double l1, l2, V[2][2];
    eig2_sym(U, l1, l2, V);
    double s1 = std::max(0.0, l1), s2 = std::max(0.0, l2);
    double g  = 0.5 * (s1 + s2);
    double shear = (g > 1e-12) ? std::fabs(s1 - s2) / g : 0.0;

    if (shear > shearMax && (s1 > 0.0 || s2 > 0.0)) {
        double target = g * shearMax;
        double avg = 0.5 * (s1 + s2);
        s1 = avg + 0.5 * target;
        s2 = avg - 0.5 * target;
        double VD[2][2] = { { V[0][0] * s1, V[0][1] * s2 },
                            { V[1][0] * s1, V[1][1] * s2 } };
        U[0][0] = VD[0][0] * V[0][0] + VD[0][1] * V[0][1];
        U[0][1] = VD[0][0] * V[1][0] + VD[0][1] * V[1][1];
        U[1][0] = VD[1][0] * V[0][0] + VD[1][1] * V[0][1];
        U[1][1] = VD[1][0] * V[1][0] + VD[1][1] * V[1][1];
        g = 0.5 * (s1 + s2);
    }

    if (!allowGain) g = 1.0;
    else            g = std::min(std::max(g, gMin), gMax);
    R[0][0] *= g; R[0][1] *= g; R[1][0] *= g; R[1][1] *= g;
}

// 2x2 matrix helpers used by locked demod, LS, and vetComposite1D.

inline void mat2_mul(const double A[2][2], const double B[2][2], double C[2][2])
{
    C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0];
    C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1];
    C[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0];
    C[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1];
}

inline void mat2_T_mul(const double A[2][2], const double B[2][2], double C[2][2])
{
    // C = A^T * B
    C[0][0] = A[0][0]*B[0][0] + A[1][0]*B[1][0];
    C[0][1] = A[0][0]*B[0][1] + A[1][0]*B[1][1];
    C[1][0] = A[0][1]*B[0][0] + A[1][1]*B[1][0];
    C[1][1] = A[0][1]*B[0][1] + A[1][1]*B[1][1];
}

inline bool mat2_inv(const double M[2][2], double Minv[2][2])
{
    const double det = M[0][0]*M[1][1] - M[0][1]*M[1][0];
    if (std::fabs(det) < 1e-12) return false;
    const double inv = 1.0 / det;
    Minv[0][0] =  M[1][1]*inv; Minv[0][1] = -M[0][1]*inv;
    Minv[1][0] = -M[1][0]*inv; Minv[1][1] =  M[0][0]*inv;
    return true;
}

inline void eig2_sym(const double S[2][2], double &l1, double &l2, double V[2][2])
{
    // Symmetric 2x2: S = [a b; b d]
    const double a = S[0][0], b = S[0][1], d = S[1][1];
    const double tr   = a + d;
    const double det  = a*d - b*b;
    const double disc = std::max(0.0, tr*tr/4 - det);
    const double rt   = std::sqrt(disc);
    l1 = tr/2 + rt;
    l2 = tr/2 - rt;
    // Eigenvectors
    if (std::fabs(b) > 1e-12) {
        V[0][0] = l1 - d; V[1][0] = b;
        V[0][1] = l2 - d; V[1][1] = b;
    } else {
        V[0][0] = 1.0; V[1][0] = 0.0;
        V[0][1] = 0.0; V[1][1] = 1.0;
    }
    // Normalise columns
    for (int j = 0; j < 2; ++j) {
        double n = std::hypot(V[0][j], V[1][j]);
        if (n > 1e-12) { V[0][j] /= n; V[1][j] /= n; }
    }
}

inline void sym_inv_sqrt(const double S[2][2], double Sminushalf[2][2])
{
    double l1, l2, V[2][2];
    eig2_sym(S, l1, l2, V);
    double d1 = (l1 > 1e-12) ? 1.0 / std::sqrt(l1) : 0.0;
    double d2 = (l2 > 1e-12) ? 1.0 / std::sqrt(l2) : 0.0;
    // S^{-1/2} = V diag(d1,d2) V^T
    double VD[2][2] = { { V[0][0]*d1, V[0][1]*d2 }, { V[1][0]*d1, V[1][1]*d2 } };
    Sminushalf[0][0] = VD[0][0]*V[0][0] + VD[0][1]*V[0][1];
    Sminushalf[0][1] = VD[0][0]*V[1][0] + VD[0][1]*V[1][1];
    Sminushalf[1][0] = VD[1][0]*V[0][0] + VD[1][1]*V[0][1];
    Sminushalf[1][1] = VD[1][0]*V[1][0] + VD[1][1]*V[1][1];
}

inline void polar_decompose_2x2(const double A[2][2], double R[2][2], double U[2][2])
{
    // A = R U, with R orthogonal, U symmetric positive definite
    double AtA[2][2];
    mat2_T_mul(A, A, AtA);
    double AtA_mhalf[2][2];
    sym_inv_sqrt(AtA, AtA_mhalf);
    mat2_mul(A, AtA_mhalf, R); // R = A (A^T A)^{-1/2}
    // U = R^T A
    double Rt[2][2] = { { R[0][0], R[1][0] }, { R[0][1], R[1][1] } };
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
