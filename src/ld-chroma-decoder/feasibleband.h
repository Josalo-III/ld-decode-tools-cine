/******************************************************************************
 * feasibleband.h
 * ld-decode-tools shared feasible-interval (hard clamp) primitive
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * A FeasibleInterval is a hard range limiter accumulated from established
 * impossibles. It is not a policy or an estimator: it forbids values that
 * cannot be true, and says nothing about which surviving value is preferred.
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <cstdint>

namespace lddecode {

struct FeasibleInterval {
    double lo = -1e300;
    double hi =  1e300;

    bool valid() const { return lo <= hi; }
    bool empty() const { return lo > hi; }
    double width() const { return hi - lo; }
    double center() const { return 0.5 * (lo + hi); }

    void clampTo(double a, double b) {
        if (a > lo)
            lo = a;
        if (b < hi)
            hi = b;
    }

    void intersect(const FeasibleInterval &o) {
        clampTo(o.lo, o.hi);
    }

    double clamp(double v) const {
        return std::clamp(v, lo, hi);
    }
};

// Luma feasibility from the +-2 carrier-cancelling sum facts.
//
// For a 4fSC composite, the carrier is antisymmetric over +-2, so:
//
//     composite[i] + composite[i+-2] = Y[i] + Y[i+-2]
//
// This yields a carrier-free luma fact. With every luma value constrained to
// [yLo, yHi], the center sample must lie in [S2 - yHi, S2 - yLo] for each
// available neighbor S2 = composite[i] + composite[i+-2].
inline FeasibleInterval lumaFeasibleFromPairSums(
    double composite_i,
    const double *neighborComposite,
    int neighborCount,
    double yLo,
    double yHi)
{
    FeasibleInterval f;
    f.clampTo(yLo, yHi);

    if (!neighborComposite || neighborCount <= 0)
        return f;

    for (int n = 0; n < neighborCount; ++n) {
        const double s2 = composite_i + neighborComposite[n];
        f.clampTo(s2 - yHi, s2 - yLo);
    }

    return f;
}

// ---------------------------------------------------------------------------
// ENCODER BANDWIDTH LAW — the carrier envelope is a bandlimited signal.
//
// This is a feasibility fact of the same family as the pair-sum interval
// above: it forbids what the encoder could not have produced, and says
// nothing about which surviving value is preferred.  It differs only in
// living on a SEQUENCE rather than a scalar, so it is expressed as a
// projection onto the expressible sub-space instead of as an interval.
//
// From the NTSC encoder (ld-chroma-encoder/encoder/ntscencoder.cpp):
//
//   * chroma is LOW-PASS FILTERED before modulation --
//         uvFilter.apply(C1);  uvFilter.apply(C2);      // 9 taps, 1.3 MHz
//     and only then modulated
//         chroma = C2*sin(a + 33deg) + C1*cos(a + 33deg);
//   * luma receives NO filter of any kind: Y[x] goes straight to the output,
//     gated only at the active-region edges.  Full band to Nyquist.
//
// So every legal carrier is  Re{ C(x) * e^{i*a(x)} }  in which the ENVELOPE
// C(x) = (C1, C2) is bandlimited to ~1.3 MHz.  At 4fSC sampling that confines
// all legal carrier energy to fSC +- 1.3 MHz (2.28 - 4.88 MHz).  An envelope
// component faster than that is INEXPRESSIBLE: no encoder input could produce
// it, so energy a fit attributes there is luma the fit has taken.
//
// The asymmetry is the whole point.  Luma is unrestricted and may sit at, or
// beside, the subcarrier; chroma may not vary quickly.  Fine luma detail near
// fSC (a dense grid, fine line art) can therefore be excluded from carrier by
// LAW rather than by any threshold, vertical partner, or discriminator --
// expressing it as carrier would require an envelope the encoder could not
// have modulated.
//
// Caveats this primitive does NOT resolve:
//   * The law is an EXCLUSION.  Carrier cannot live outside the band, but luma
//     certainly lives inside it, so this constrains without fully separating.
//   * Real mastering chains are not this encoder.  A ~1.3 MHz chroma bandlimit
//     is standard NTSC practice rather than an implementation quirk, but
//     multi-generation or dubbed sources may present a somewhat wider
//     effective band; treat the projection as the conservative floor it is.
// ---------------------------------------------------------------------------

// The encoder's own chroma bandlimit (ntscencoder.cpp uvFilterCoeffs):
// 0 dB at DC, >= -2 dB at 1.3 MHz, < -20 dB at 3.6 MHz, at 4fSC.
inline constexpr int kChromaEnvelopeTaps = 9;
inline constexpr double kChromaEnvelopeFilter[kChromaEnvelopeTaps] = {
    0.0021, 0.0191, 0.0903, 0.2308, 0.3153,
    0.2308, 0.0903, 0.0191, 0.0021
};

// Project one channel of a fitted carrier envelope onto the expressible
// sub-space, sample-aligned (`in` and `out` may not alias).
//
// `valid` (optional) marks positions carrying a real fit.  Invalid positions
// are neither read nor written: an absent fit is not evidence of a zero
// envelope, and smearing one across its neighbours would manufacture carrier.
// Where taps are dropped -- at line ends or beside invalid positions -- the
// remaining weight is renormalised so the passband gain stays 1.0 and legal
// carrier is not attenuated at the margins; the cutoff is correspondingly
// approximate there.
//
// The residue (in - out) is not returned or stored anywhere: whatever the
// encoder could not have modulated simply never enters the carrier model, and
// therefore remains in luma by construction.
inline void projectExpressibleChromaEnvelope(const double *in,
                                             const std::uint8_t *valid,
                                             int n,
                                             double *out)
{
    if (!in || !out || n <= 0)
        return;

    constexpr int half = kChromaEnvelopeTaps / 2;

    for (int i = 0; i < n; ++i) {
        if (valid && !valid[i]) {
            out[i] = in[i];
            continue;
        }

        double acc = 0.0;
        double used = 0.0;
        for (int k = 0; k < kChromaEnvelopeTaps; ++k) {
            const int j = i + k - half;
            if (j < 0 || j >= n)
                continue;
            if (valid && !valid[j])
                continue;
            const double w = kChromaEnvelopeFilter[k];
            acc += w * in[j];
            used += w;
        }

        out[i] = (used > 1e-12) ? (acc / used) : in[i];
    }
}

} // namespace lddecode
