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
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

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
// ENCODER BANDWIDTH LAW — the carrier envelope is bandlimited.
//
// This is a feasibility constraint, not an estimator: it excludes envelope
// variation the encoder cannot express and does not choose among surviving
// carrier values. Because the constraint acts on a sequence, it is represented
// by filtering rather than a scalar FeasibleInterval.
//
// The 9-tap envelope filter below is a forward FIR, not an idempotent
// projection. Reapplying it compounds its attenuation, so callers using
// projectExpressibleChromaEnvelope must treat each application as part of the
// signal path rather than as a cost-free assertion of the same law.
//
// The reference NTSC encoder low-pass filters chroma before modulation:
//
//     uvFilter.apply(C1);  uvFilter.apply(C2);      // 9 taps, 1.3 MHz
//     chroma = C2*sin(a + 33deg) + C1*cos(a + 33deg);
//
// Its NARROWBAND_Q synthesis mode uses a separate 0.46 MHz kernel for one
// rotated coordinate. That test-generation mode is not a mastering-law source.
// Decoder render filtering remains a separate mechanism from feasibility.
//
// Luma is not filtered by the encoder model: Y[x] passes directly to output
// inside the active region. Therefore legal carrier has the form
//
//     Re{ C(x) * e^{i*a(x)} }
//
// with a bandlimited chroma envelope C(x), while luma may contain energy at or
// near fSC. Envelope energy outside the permitted chroma bandwidth is therefore
// excluded from the carrier model and remains in luma.
//
// The law is only an exclusion: luma can also occupy the legal carrier band, so
// bandwidth alone cannot fully separate Y from chroma. Axis-specific legality
// kernels are defined below for callers that have the corresponding coordinate
// frame.
// ---------------------------------------------------------------------------

// The encoder's 9-tap chroma creation filter (ntscencoder.cpp
// uvFilterCoeffs) has this response at 4fSC:
//
//        DC   0.46   1.00   1.30   1.50   2.00   2.50   3.58 MHz
//      0.00  -0.28  -1.34  -2.26  -3.01  -5.35  -8.37 -17.15 dB
//
// It is suitable for reproducing the encoder's creation filter, but not as a
// hard legality projector: its passband droops and its stopband remains finite.
// The flatter legality kernels below serve the latter role.

inline constexpr int kChromaEnvelopeTaps = 9;
inline constexpr double kChromaEnvelopeFilter[kChromaEnvelopeTaps] = {
    0.0021, 0.0191, 0.0903, 0.2308, 0.3153,
    0.2308, 0.0903, 0.0191, 0.0021
};

// Filter one fitted carrier-envelope channel with the current encoder-shaped
// envelope kernel, sample-aligned (`in` and `out` may not alias).
//
// `valid` marks positions carrying a real fit. Invalid positions are neither
// read nor replaced by zero; dropped taps at invalid samples or line ends are
// renormalised so the surviving kernel weights retain unit DC gain.
//
// LDCD_ENV_TAPS selects an odd resampled aperture for both channels;
// LDCD_ENV_TAPS_I and LDCD_ENV_TAPS_Q override it per channel. Values below 3
// leave that selector unset, and an unset selector uses the base 9-tap kernel.
// Resampling preserves unit DC gain while changing the effective aperture.

inline std::vector<double> buildResampledEnvelopeKernel(int ov)
{
    std::vector<double> k;
    if (ov < 3) return k;
    k.resize(ov);
    const double scale = double(kChromaEnvelopeTaps - 1) / double(ov - 1);
    double sum = 0.0;
    for (int i = 0; i < ov; ++i) {
        const double src = i * scale;
        const int    j   = (int)src;
        const double f   = src - j;
        const double a   = kChromaEnvelopeFilter[j];
        const double b   = (j + 1 < kChromaEnvelopeTaps)
                             ? kChromaEnvelopeFilter[j + 1] : 0.0;
        k[i] = a + f * (b - a);
        sum += k[i];
    }
    for (double &v : k) v /= sum;
    return k;
}

inline int envelopeTapsFor(int chan)
{
    auto rd = [](const char *n) {
        const char *e = std::getenv(n);
        int v = e ? std::atoi(e) : 0;
        return v < 3 ? 0 : (v | 1);
    };
    static const int base = rd("LDCD_ENV_TAPS");
    static const int tI   = rd("LDCD_ENV_TAPS_I");
    static const int tQ   = rd("LDCD_ENV_TAPS_Q");
    if (chan == 0 && tI >= 3) return tI;
    if (chan == 1 && tQ >= 3) return tQ;
    return base;
}

// chan 0 = the I-side channel, 1 = the Q-side channel, -1 = unattributed
// (magnitude observables, cascades).  Per-channel apertures exist because the
// two channels are NOT bandlimited alike: the encoder gave them different
// allowances and the material fills them differently.
inline const double *envelopeKernel(int &taps, int chan = -1)
{
    static const std::vector<double> kDef =
        buildResampledEnvelopeKernel(envelopeTapsFor(-1));
    static const std::vector<double> kI =
        buildResampledEnvelopeKernel(envelopeTapsFor(0));
    static const std::vector<double> kQ =
        buildResampledEnvelopeKernel(envelopeTapsFor(1));
    const std::vector<double> *k =
        (chan == 0) ? &kI : (chan == 1) ? &kQ : &kDef;
    if (!k->empty()) { taps = (int)k->size(); return k->data(); }
    taps = kChromaEnvelopeTaps;
    return kChromaEnvelopeFilter;
}

inline void projectExpressibleChromaEnvelope(const double *in,
                                             const std::uint8_t *valid,
                                             int n,
                                             double *out,
                                             int chan = -1)
{
    if (!in || !out || n <= 0)
        return;

    int kChromaEnvelopeTaps_ = 0;
    const double *kern = envelopeKernel(kChromaEnvelopeTaps_, chan);
    const int half = kChromaEnvelopeTaps_ / 2;

    for (int i = 0; i < n; ++i) {
        if (valid && !valid[i]) {
            out[i] = in[i];
            continue;
        }

        double acc = 0.0;
        double used = 0.0;
        for (int k = 0; k < kChromaEnvelopeTaps_; ++k) {
            const int j = i + k - half;
            if (j < 0 || j >= n)
                continue;
            if (valid && !valid[j])
                continue;
            const double w = kern[k];
            acc += w * in[j];
            used += w;
        }

        out[i] = (used > 1e-12) ? (acc / used) : in[i];
    }
}

// ---------------------------------------------------------------------------
// LEGALITY-LAW KERNELS.
//
// These FIRs are flat through their admitted passbands and steep in the stop,
// so repeated application is approximately idempotent and legal in-band energy
// is not materially attenuated.
//
//                    in band        at the stop      extra-pass loss
//   wide law         -0.01 dB @1.5   -59 dB @2.2     ~0.009 dB
//   narrow law       -0.01 dB @0.8   -88 dB @1.5     ~0.008 dB
//
// WIDE is the 1.50 MHz legality ceiling used by the current coarse-feasibility
// path. Applying the same wide law to both orthogonal quadrature coordinates is
// rotation-invariant.
//
// NARROW is an axis-specific 0.80 MHz kernel. It is available through
// projectLawfulChromaCoordinate but has no current decode-path caller; any use
// must supply the actual narrow chroma axis rather than an arbitrary lattice
// coordinate.
// ---------------------------------------------------------------------------

enum class ChromaLawAxis { Wide, Narrow };

inline constexpr int kChromaLawWideTaps = 75;
inline constexpr double kChromaLawWide[kChromaLawWideTaps] = {
    -0.000172157, -0.000216309, -0.000052499, +0.000319393, +0.000673212,
    +0.000650387, +0.000037160, -0.000954748, -0.001665972, -0.001360526,
    +0.000173559, +0.002222995, +0.003366078, +0.002355885, -0.000817816,
    -0.004485000, -0.006027750, -0.003587675, +0.002276636, +0.008269144,
    +0.010004789, +0.004946670, -0.005193938, -0.014504787, -0.015983451,
    -0.006276025, +0.010902650, +0.025400201, +0.025895632, +0.007397957,
    -0.023421518, -0.048952329, -0.048233909, -0.008148624, +0.067696001,
    +0.157745539, +0.230515396, +0.258411496, +0.230515396, +0.157745539,
    +0.067696001, -0.008148624, -0.048233909, -0.048952329, -0.023421518,
    +0.007397957, +0.025895632, +0.025400201, +0.010902650, -0.006276025,
    -0.015983451, -0.014504787, -0.005193938, +0.004946670, +0.010004789,
    +0.008269144, +0.002276636, -0.003587675, -0.006027750, -0.004485000,
    -0.000817816, +0.002355885, +0.003366078, +0.002222995, +0.000173559,
    -0.001360526, -0.001665972, -0.000954748, +0.000037160, +0.000650387,
    +0.000673212, +0.000319393, -0.000052499, -0.000216309, -0.000172157
};

inline constexpr int kChromaLawNarrowTaps = 87;
inline constexpr double kChromaLawNarrow[kChromaLawNarrowTaps] = {
    +0.000142502, +0.000214411, +0.000239181, +0.000172641, -0.000011973,
    -0.000305192, -0.000649802, -0.000943503, -0.001059383, -0.000882418,
    -0.000353991, +0.000488651, +0.001492946, +0.002403798, +0.002913359,
    +0.002741818, +0.001731759, -0.000069799, -0.002369908, -0.004651116,
    -0.006264315, -0.006582279, -0.005183165, -0.002018322, +0.002486059,
    +0.007435369, +0.011601380, +0.013674906, +0.012592748, +0.007868721,
    -0.000153808, -0.010203333, -0.020183911, -0.027486889, -0.029467403,
    -0.023998482, -0.009988168, +0.012257890, +0.040921596, +0.072850966,
    +0.104016519, +0.130168745, +0.147572959, +0.153676469, +0.147572959,
    +0.130168745, +0.104016519, +0.072850966, +0.040921596, +0.012257890,
    -0.009988168, -0.023998482, -0.029467403, -0.027486889, -0.020183911,
    -0.010203333, -0.000153808, +0.007868721, +0.012592748, +0.013674906,
    +0.011601380, +0.007435369, +0.002486059, -0.002018322, -0.005183165,
    -0.006582279, -0.006264315, -0.004651116, -0.002369908, -0.000069799,
    +0.001731759, +0.002741818, +0.002913359, +0.002403798, +0.001492946,
    +0.000488651, -0.000353991, -0.000882418, -0.001059383, -0.000943503,
    -0.000649802, -0.000305192, -0.000011973, +0.000172641, +0.000239181,
    +0.000214411, +0.000142502
};

// Shared FIR body: symmetric kernel, taps dropped at line ends and beside
// invalid positions with the remaining weight renormalised so passband gain
// stays 1.0 (the cutoff is correspondingly approximate there). `in` and `out`
// may not alias.
inline void applyRenormalisedSymmetricFir(const double *in,
                                          const std::uint8_t *valid,
                                          int n,
                                          double *out,
                                          const double *k,
                                          int ktaps)
{
    if (!in || !out || !k || n <= 0 || ktaps <= 0)
        return;

    const int half = ktaps / 2;

    for (int i = 0; i < n; ++i) {
        if (valid && !valid[i]) {
            out[i] = in[i];
            continue;
        }

        double acc = 0.0;
        double used = 0.0;
        for (int t = 0; t < ktaps; ++t) {
            const int j = i + t - half;
            if (j < 0 || j >= n)
                continue;
            if (valid && !valid[j])
                continue;
            const double w = k[t];
            acc += w * in[j];
            used += w;
        }

        out[i] = (used > 1e-12) ? (acc / used) : in[i];
    }
}

// Apply the selected legality kernel to one chroma-envelope coordinate.
// WIDE may be applied to both members of any orthogonal quadrature basis because
// the same bandwidth is imposed on both axes. NARROW is axis-specific and
// requires the caller to supply the corresponding physical chroma coordinate.
inline void projectLawfulChromaCoordinate(const double *in,
                                          const std::uint8_t *valid,
                                          int n,
                                          double *out,
                                          ChromaLawAxis axis)
{
    if (axis == ChromaLawAxis::Narrow)
        applyRenormalisedSymmetricFir(in, valid, n, out,
                                      kChromaLawNarrow, kChromaLawNarrowTaps);
    else
        applyRenormalisedSymmetricFir(in, valid, n, out,
                                      kChromaLawWide, kChromaLawWideTaps);
}

// Coarse-residual feasibility bounds on the COMPOSITE carrier at one sample.
//
// A legal carrier sums to zero over every legal four-sample window, so each
// coarse mean apMean[v] = mean(raw[v..v+3]) is that window's LUMA mean exactly
// -- the carrier is removed with no filter and no assumption. The four windows
// covering sample x (starts v in {x-3 .. x}) therefore share x's carrier and
// differ only in their luma. Bounding the luma at x bounds the carrier:
//
//   Y[x] >= min_v apMean[v]   (the darkest covering coarse is the BRIGHTEST the
//                              luma floor could be; the bright side is less
//                              knowable) => carrier <= raw - min_v apMean[v]
//   Y[x] <= max_v apMean[v]                => carrier >= raw - max_v apMean[v]
//
// The ceiling (dark side) is the clean bound; the floor (bright side) is
// ambiguous. These RESTRICT the emitted carrier -- a consumer clamps into the
// range and the excess returns to luma -- they are never averaged into it, and
// the primitive publishes them unfiltered so the consumer owns the decision.
struct CarrierFeasibleRange { double floor; double ceiling; };
inline CarrierFeasibleRange carrierFeasibleRange(double rawSample,
                                                 const double *apMean,
                                                 int x, int width)
{
    const int lastStart = width - 4;      // last legal aperture START
    double lo = 0.0, hi = 0.0;
    bool any = false;
    for (int d = 0; d < 4; ++d) {
        const int v = x - (3 - d);        // covering starts x-3, x-2, x-1, x
        if (v < 0 || v > lastStart) continue;
        const double m = apMean[v];
        if (!any) { lo = hi = m; any = true; }
        else { lo = m < lo ? m : lo; hi = m > hi ? m : hi; }
    }
    if (!any)                             // no legal aperture: no restriction
        return { -1e300, 1e300 };
    return { rawSample - hi, rawSample - lo };   // { floor, ceiling }
}

// ---------------------------------------------------------------------------
// CERTIFIED TWIN-BRACKET CARRIER BOUND.
//
// A 3:2 telecine definitional field and its spare are two captures of the same
// film image with opposite carrier polarity:
//
//     (def + spare)/2 = image-locked band content
//     (def - spare)/2 = certified carrier
//
// For an uncovered frame, a certified neighbouring frame can therefore supply
// a regional carrier FRACTION. The fraction transfers as a magnitude ceiling,
// not as a pointwise carrier value: local sample values still come from the
// frame being bounded.
//
// CertifiedBracketBlock stores mean-square carrier energy from the twin split,
// mean-square carrier-band energy from that certified bracket, and local
// carrier-band energy from the frame being bounded. The interval constructor
// converts their ratio into a symmetric per-sample carrier bound:
//
//     fraction  = bracketCarrierEnergy / bracketBandEnergy
//     ceiling   = margin * fraction
//     amplitude = crest * sqrt(ceiling * localBandEnergy)
//
// `margin` supplies cross-frame headroom and `crest` converts block RMS energy
// to an allowed sample excursion. A ceiling at or above unity is non-binding.
// Missing, invalid, or non-positive evidence likewise returns an unbounded
// interval: absence of evidence forbids nothing.
//
// The fraction is a regional/block statistic and is applied uniformly within
// that block; it is not estimated from instantaneous band samples. A caller
// using only one certifying bracket should provide the one-sided margin required
// by its cadence policy.
//
// If a supposed twin contains picture change, the twin difference grows and
// the resulting fraction loosens the ceiling toward inaction rather than
// creating a tighter false bound.
// ---------------------------------------------------------------------------

// The carrier-antisymmetric projection at 4fSC: unity at fSC, null at DC and
// at 2fSC.  Same +-2 antisymmetry lumaFeasibleFromPairSums sums over, taken
// as a difference so it selects the carrier band instead of cancelling it.
inline double carrierBandProjection(double sample_h, double sample_hMinus2)
{
    return 0.5 * (sample_h - sample_hMinus2);
}

// Block statistics feeding the bound.  Energies are mean-square over the same
// block; the two bracket terms come from the certified frame, the local term
// from the frame being bounded.
struct CertifiedBracketBlock {
    double bracketCarrierEnergy = 0.0;  // mean of ((def-spare)/2)^2
    double bracketBandEnergy    = 0.0;  // mean of projection^2, bracket frame
    double localBandEnergy      = 0.0;  // mean of projection^2, this frame
    long   samples              = 0;
};

// Carrier-magnitude interval for every sample in the block.  An empty or
// unusable bracket returns the unbounded interval: absence of evidence
// forbids nothing.
//
// margin scales the certified fraction before it is believed (it is another
// frame's fraction, not this one's); crest converts a block mean-square into a
// per-sample amplitude that the carrier's own excursions may legitimately
// reach.  A fraction that reaches unity forbids nothing at all -- the bracket
// is reporting a fully chromatic band -- and the interval opens.
inline FeasibleInterval carrierFeasibleFromCertifiedBracket(
    const CertifiedBracketBlock &b, double margin, double crest)
{
    FeasibleInterval f;                       // unbounded
    if (b.samples <= 0 || b.bracketBandEnergy <= 0.0 ||
        b.localBandEnergy <= 0.0 || margin <= 0.0 || crest <= 0.0)
        return f;

    const double fraction = b.bracketCarrierEnergy / b.bracketBandEnergy;
    if (!(fraction >= 0.0))                   // NaN-safe
        return f;

    const double ceiling = margin * fraction;
    if (ceiling >= 1.0)                       // nothing is excluded
        return f;

    const double amplitude = crest * std::sqrt(ceiling * b.localBandEnergy);
    f.clampTo(-amplitude, amplitude);
    return f;
}

} // namespace lddecode
