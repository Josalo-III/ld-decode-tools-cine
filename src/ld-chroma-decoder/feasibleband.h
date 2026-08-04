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
