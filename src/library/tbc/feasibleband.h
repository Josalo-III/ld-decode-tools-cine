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

} // namespace lddecode
