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
// ENCODER BANDWIDTH LAW — the carrier envelope is a bandlimited signal.
//
// This is a feasibility fact of the same family as the pair-sum interval
// above: it forbids what the encoder could not have produced, and says
// nothing about which surviving value is preferred.  It differs in living on
// a SEQUENCE rather than a scalar, so it cannot be an interval.
//
// IT IS NOT A PROJECTION, despite what this header said until 2026-08-11.
// A forward FIR is idempotent only if its response is 0 or 1 everywhere, and
// this one's is neither: applying it twice costs a further 3.01 dB at 1.5 MHz
// (measured; the same defect a third-party review caught in 2026-07 when it
// was being applied at three layers, H(f)^3).  Callers must therefore know
// how many times it has run, which is a fragile contract for something
// calling itself a law.  See the kernel note below for why the shape is
// wrong for this job in the first place.
//
// From the NTSC encoder (ld-chroma-encoder/encoder/ntscencoder.cpp):
//
//   * chroma is LOW-PASS FILTERED before modulation --
//         uvFilter.apply(C1);  uvFilter.apply(C2);      // 9 taps, 1.3 MHz
//     and only then modulated
//         chroma = C2*sin(a + 33deg) + C1*cos(a + 33deg);
//
//     A SECOND ENCODER MODE EXISTS, and it is easy to over-read. Under
//     NARROWBAND_Q (ntscencoder.cpp:300) C2 is rotated 33 degrees out of U/V
//     and filtered by a 23-tap 0.46 MHz kernel instead, so the two
//     coordinates are bandlimited very differently. But NARROWBAND_Q is an
//     ENCODER symbol for synthesising test material -- it is NOT evidence
//     about how any disc was mastered, and it must not be cited as such.
//     What the DECODER already believes about the axes is in filterIQLocked,
//     which renders I at 1.5 MHz and Q at 0.67 MHz -- a mechanism chosen for
//     spec-mindful noise removal, not a law (see the locked-Q record).
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
//   * Real mastering chains are not this encoder.  This caveat used to warn
//     that multi-generation or dubbed sources might present a WIDER effective
//     band, making the law unsafe as a bound.  Measured on disc 2026-08-11
//     against certified twins, and it points the other way: nothing measured
//     anywhere exceeded its design filter, so the law is confirmed as a
//     ceiling.  The result is PER AXIS and pooling the axes gives a confident
//     wrong number -- the narrow axis is at spec everywhere (0.46-0.55 MHz),
//     while the wide axis reaches spec only on electronically-generated
//     content (1.05-1.17 MHz on a title, 0.62-0.65 on anything photographed).
//     That last figure is a property of CONTENT through a lens, never a
//     legality bound: a law must forbid only what the encoder could not emit,
//     so the wide axis stays at its 1.50 MHz design and must not be tightened
//     toward what one scene happened to contain.
// ---------------------------------------------------------------------------

// The encoder's own chroma bandlimit (ntscencoder.cpp uvFilterCoeffs).
//
// MEASURED RESPONSE at 4fSC (2026-08-11). The two figures this comment used
// to carry -- ">= -2 dB at 1.3 MHz, < -20 dB at 3.6 MHz" -- were BOTH wrong,
// and wrong in the reassuring direction:
//
//        DC   0.46   1.00   1.30   1.50   2.00   2.50   3.58 MHz
//      0.00  -0.28  -1.34  -2.26  -3.01  -5.35  -8.37 -17.15 dB
//
// So it takes 2.26 dB at 1.3 MHz (not "at most 2") and reaches only
// -17.15 dB at Nyquist (not "below -20").
//
// THIS IS THE ENCODER'S CREATION FILTER, NOT A LEGALITY TEST, and the two
// jobs want opposite shapes. Legal chroma reaching us has ALREADY been
// through this response once at the encoder; applying it again in the name
// of legality attenuates what is by definition legal -- a third of the
// amplitude at the encoder's own passband edge -- while admitting
// out-of-band energy at only -17 dB. A law wants a flat passband and a steep
// stop; this has a drooping passband and a lazy stop. It fails in both
// directions at once.
//
// Verified replacements exist (Kaiser designs, flat to 0.01 dB in band,
// -59/-88 dB in stop, idempotent to 0.009 dB so the run-count contract
// disappears) but are NOT wired: they must be applied on the true I/Q axes,
// and the demodulated channels in this decoder are on the raw sample-class
// lattice (combmath.h sin4fsc/cos4fsc), which is the LANE frame, not I/Q.
// Handing a narrow Q kernel to a channel merely NAMED Q would clip legal I
// by 26 dB at 1.3 MHz.
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
// TEMPORARY SWEEP KNOB (LDCD_ENV_TAPS, 2026-08-17): resample the encoder
// kernel to an arbitrary odd tap count so the envelope law's effective
// aperture can be swept against certified truth. 0/unset = the shipped 9-tap
// kernel, untouched. The resampled kernel keeps unit DC gain, so only the
// cutoff moves. Instrument only -- strip with the sweep.
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
// REPLACEMENT LAW KERNELS (designed and verified 2026-08-11, NOT YET WIRED).
//
// The kernel above is the encoder's CREATION filter. These two are LEGALITY
// TESTS, which is a different job and wants the opposite shape: flat wherever
// the encoder could legally have put energy, steep immediately above it.
//
//                    in band      at the stop      idempotency
//   encoder kernel   -3.01 dB @1.5   -17 dB @Nyq   -3.01 dB per extra pass
//   wide law         -0.01 dB @1.5   -59 dB @2.2   -0.009 dB
//   narrow law       -0.01 dB @0.8   -88 dB @1.5   -0.008 dB
//
// Being idempotent to 0.01 dB, these retire the "must be applied exactly
// once" contract that made the old kernel fragile: a second pass costs
// nothing measurable, so no caller has to know the run count.
//
// TWO AXES. Read the landscape before wiring either, because most of this
// ground is already surveyed and the survey is better than a fresh guess:
//
//   * filterIQLocked ALREADY filters per axis (1.5 / 0.67 MHz). Per-axis
//     treatment is not a new capability to introduce here.
//   * Swapping the encoder 9-tap onto both render axes was already measured
//     (LDCD_RENDER_FEASIBLE): cross-colour 3% BETTER, width and saturation
//     unchanged -- free, and it does not move the chroma smear.
//   * Naively widening Q to 1.3 MHz costs +14.2% cross-colour on GGV
//     monochrome bars. Any Q change must be measured on BOTH axes at once:
//     cross-colour admitted AND transition width.
//
//   WIDE -- passband 1.50 MHz, the encoder's uvFilter design point, which is
//   what a law must bound. Do NOT retune it toward the 0.62-0.65 MHz measured
//   on photographed material: that is a lens and a film chain, not a legality
//   limit, and a title on the same disc legally reached 1.05-1.17 MHz.
//
//   NARROW -- passband 0.80 MHz. THIS NUMBER IS NOT YET JUSTIFIED. It was
//   derived from the encoder's narrowband-Q design point plus margin, i.e.
//   from a test-synthesis mode misread as a mastering fact. It also sits
//   LOOSER than the 0.67 MHz the renderer already applies, so as written it
//   would never bind. Settle what the narrow axis's legal bound actually is
//   before wiring this; the number here is a placeholder, not a finding.
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

// Bound one COLOUR COORDINATE to what the encoder could have modulated on
// that axis. The caller must supply a genuine I or Q sequence: the axis is
// the whole point, and the demodulated channels in this decoder are on the
// raw sample-class lattice (the lane frame), which is 33 degrees away.
// Handing a lane to this function under an axis name is the one mistake it
// cannot detect.
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
// CERTIFIED TWIN-BRACKET CARRIER BOUND — the temporal sibling of
// lumaFeasibleFromPairSums above.
//
// That primitive uses the carrier's antisymmetry over +-2 SAMPLES to obtain a
// carrier-free luma fact.  A 3:2 telecine hands us the same antisymmetry over
// +-2 FIELDS: the definitional field and its spare are one film frame captured
// twice, so with the carrier inverted between them
//
//     (def + spare)/2 = the image-locked band content   (carrier cancels)
//     (def - spare)/2 = the carrier, exactly            (picture cancels)
//
// The second is a certified per-sample carrier for that film frame, in native
// composite units.  Measured against the band-projected form it correlates at
// r = +0.98, so nothing needs extracting to obtain it: the twin difference IS
// the carrier.
//
// An uncovered frame has no twin of its own.  What it does have is a covered
// frame on either side whose certified split says what FRACTION of the carrier
// band at this location is genuinely carrier.  That fraction is a scene
// property and it transfers; the sample values do not (measured: r = +0.08
// pointwise, so this must never be used as a value -- see FALSIFIED below).
// So the bracket yields a CEILING on carrier magnitude and nothing else, which
// is exactly this file's contract: it forbids what cannot be true and says
// nothing about which surviving value is preferred.
//
// Measured across 4 footage sources / 13 scenes / ~1.5M samples: the carrier
// fit on a frame with no certified legs over-claims by 2.7-3.1x where the band
// is luma-dominated and by 1.03-1.06x where it is genuinely chromatic, so the
// bound is self-targeting -- it binds on the content class that produces
// cross-colour and is inert on real chroma.
//
// FALSIFIED — none of these may enter a construction:
//   * A fixed global ceiling.  49-62% of certified blocks exceed any constant
//     in 0.3/0.5/0.7; carrier amplitude admits no single discriminator.
//   * margin 1.0.  25.6% violation two-sided.  The margin is what makes this a
//     feasibility bound rather than an estimator.
//   * Scaling the bound by the INSTANTANEOUS band value.  It collapses at band
//     zero-crossings, where the carrier need not be zero; 11.4% violation even
//     at 3x headroom, and it does not converge.  The bound must be block-flat.
//   * Estimating the fraction per sample.  15.0% violation at 1x1 against
//     0.90% at 8x16.  The ratio is noisy at fine scale; estimate regionally,
//     apply per sample.
//   * Gating on the two brackets' agreement with each other.  Violations
//     CONCENTRATE where they agree (2.2%) and vanish where they disagree
//     (0.0%): disagreement inflates the max() and loosens the ceiling, so the
//     statistic is anti-correlated with the failure, not independent of it.
//   * Using the spare's carrier at the spare rotation as a value.  It is a
//     deterministic negation of what the merge already holds.
//   * An inter-bracket "motion" gate on rms(luma_prev - luma_next).  It looked
//     decisive pooled (9.1% violation above 20 IRE against 0.1-0.4% below) and
//     is an ARTEFACT: excluding one scene, that same bin runs at 0.24% while
//     the excluded scene alone runs at 42.9%.  The statistic contains no
//     displacement -- a bright object moving one sample outscores a dim one
//     moving ten, and a lighting change with no movement outscores both -- so
//     it was never a motion measurement, and pooling let one scene wear it as
//     a law.  Any precondition here must be derived from an image
//     transformation, not from a raw frame difference.
//   * An amplitude floor under the bound.  It does not reach the failure
//     regime below (8.6% at 5 IRE) and costs 9 points of engagement
//     elsewhere.
//
// REQUIRED:
//   * margin 3.0 where only ONE bracket certifies the line's parity, which is
//     the ordinary 3:2 geometry for the uncovered letters (2.3% violation at
//     margin 2.0 one-sided against 0.90% two-sided).
//
// OPEN — why this is not yet a default:
//   On one of four sources the bound is unsafe at 9.8% and nothing tried so
//   far separates it.  That material carries a carrier fraction of 0.003-0.008
//   -- the band is ~99.5% image-locked -- and the violating blocks hold a
//   genuine 0.07-0.11 against brackets at 0.002, with band energy identical
//   across the bracket.  A MULTIPLICATIVE margin on a near-zero base forbids
//   almost everything, so real chroma appearing in the middle film frame
//   breaks it.  Every other scene measured runs at 0.00-0.09% with the bound
//   active on 67-77% of blocks.  Until that regime is characterised this stays
//   opt-in: a false forbid confiscates real colour, which outranks the gain.
//
// A mislabelled twin fails SAFE: if the pair is not one film frame, the
// difference carries picture change as well as carrier, the fraction rises,
// and the ceiling loosens toward inaction.  So this needs no validity test of
// its own and never second-guesses the cadence autosolve.
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
