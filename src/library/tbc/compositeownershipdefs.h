/******************************************************************************
 * compositeownershipdefs.h
 * ld-decode-tools shared composite/luma/chroma ownership definitions
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This header defines the shared attribution vocabulary used by composite
 * decoding stages.  It intentionally contains evidence storage and small
 * normalization helpers only; decoder-specific scoring policy should remain
 * in the owning decoder.
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <cmath>

namespace lddecode {

struct CompositeOwnershipEvidence {
    // ---------------------------------------------------------------------
    // Luma / residual waveform evidence
    // ---------------------------------------------------------------------

    // Generic residual evidence used by the luma-first composite decoder.
    double residualFineIRE = 0.0;
    double residualMidIRE = 0.0;
    double residualCoarseIRE = 0.0;

    double lumaExcursionIRE = 0.0;
    double residualFitErrorIRE = 0.0;
    double lumaIncursionRiskIRE = 0.0;
    double icebergAlienYFraction = 0.0;
    double lumaShapeContinuation = 0.0;

    // ---------------------------------------------------------------------
    // Composite / carrier-domain evidence
    // ---------------------------------------------------------------------

    // Historical/simple residual magnitude proxy.
    double compositeResidualIRE = 0.0;

    // Generic/PAL-safe carrier names.
    double carrierResidualIRE = 0.0;
    double carrierChromaFitIRE = 0.0;
    double carrierChromaErrorIRE = 0.0;
    double carrierChromaCoherence = 0.0;
    double carrierPhaseCoherence = 0.0;

    // Current composite-decoder NTSC names.
    double compositeChromaFitIRE = 0.0;
    double compositeChromaErrorIRE = 0.0;
    double compositeChromaCoherence = 0.0;
    double compositeCarrierCoherence = 0.0;

    double carrierPlausibility = 0.0;

    // ---------------------------------------------------------------------
    // Early coarse-Y / quarter-stage evidence
    //
    // Captured immediately after the coarse Y anchor, before later Y
    // refinement mutates the original residual.  These are durable priors,
    // not final claims.
    // ---------------------------------------------------------------------

    double coarseY0IRE = 0.0;
    double quarterResidualIRE = 0.0;

    double quarterCarrierFitIRE = 0.0;
    double quarterCarrierErrorIRE = 0.0;
    double quarterCarrierCoherence = 0.0;

    double quarterIQMagIRE = 0.0;
    double quarterIQSmoothness = 0.0;
    double quarterIQChromaPrior = 0.0;

    double quarterLostYPeakIRE = 0.0;
    double quarterLostYPrior = 0.0;
    double quarterBoundaryPrior = 0.0;
    double quarterCoincidentTransition = 0.0;
    double quarterImpulseYPrior = 0.0;

    // Difference between the two legal 4fsc-cycle coarse-Y explanations after
    // expansion to the full sample grid:
    //
    //     quarterOffsetYDeltaIRE = abs(expandedB - expandedA)
    //
    // This is not final luma.  It marks where one 4fsc cancellation phase
    // strands different material in the residual than the other.
    double quarterOffsetYDeltaIRE = 0.0;
    double quarterOffsetYPrior = 0.0;

    // ---------------------------------------------------------------------
    // Chroma-envelope / NTSC IQ evidence
    // ---------------------------------------------------------------------

    // NTSC/IQ-specific evidence.  These names are kept because the current
    // NTSC implementation really is reasoning in I/Q space.
    double iqChromaMagIRE = 0.0;
    double iqEnvelopeSmoothness = 0.0;
    double iqExcessHF = 0.0;
    double iqRemodErrorIRE = 0.0;
    double iqChromaCoherence = 0.0;
    double iqEnvelopeCoherence = 0.0;

    // Generic/PAL-safe envelope aliases for future non-NTSC backends.
    double chromaEnvelopeMagIRE = 0.0;
    double chromaEnvelopeSmoothness = 0.0;
    double chromaEnvelopeExcessHF = 0.0;
    double chromaRemodErrorIRE = 0.0;
    double chromaEnvelopeCoherence = 0.0;

    // ---------------------------------------------------------------------
    // Sideband / carrier-centered legality evidence
    //
    // Chroma is not merely energy near the color subcarrier.  A legal chroma
    // explanation should show a structured relationship between residual
    // energy below and above the carrier.  The two sides need not have equal
    // amplitude; the useful score is whether they cohere into one plausible
    // chroma explanation.
    // ---------------------------------------------------------------------

    double sidebandLowerIRE = 0.0;
    double sidebandUpperIRE = 0.0;
    double sidebandCoherence = 0.0;
    double sidebandAsymmetry = 0.0;

    // ---------------------------------------------------------------------
    // Subtraction reconciliation evidence
    //
    // Used by model-based decoders to compare:
    //
    //     C_from_Y = rawComposite - refinedY
    //     Y_from_C = rawComposite - refinedChroma
    //
    // Agreement strengthens attribution confidence.  Disagreement should
    // generally raise uncertainty unless other evidence clearly resolves it.
    // ---------------------------------------------------------------------

    double subtractionAgreement = 0.0;
    double subtractionErrorIRE = 0.0;
    double reverseLumaIRE = 0.0;
    double forwardChromaIRE = 0.0;
    double modelAgreementClaim = 0.0;

    // ---------------------------------------------------------------------
    // Attribution claims
    // ---------------------------------------------------------------------

    double ownershipConflict = 0.0;

    // Composite-model attribution claims.
    //
    // lumaClaim:
    //     Attribution that contested residual energy is better explained as
    //     luma than chroma.
    //
    // The chroma side is deliberately decomposed.  There is no generic
    // chromaClaim field here because the composite decoder should preserve
    // whether the claim came from carrier, envelope, sideband, or current
    // NTSC composite evidence.
    //
    // uncertainClaim:
    //     Remaining unattributed confidence.  The convention is:
    //
    //         uncertainClaim = 1 - max(lumaClaim, combined chroma claim)
    //
    double lumaClaim = 0.0;
    double uncertainClaim = 1.0;

    // Generic/PAL-safe decomposed chroma claims.
    double carrierChromaClaim = 0.0;
    double envelopeChromaClaim = 0.0;
    double sidebandChromaClaim = 0.0;

    // Current NTSC/composite decoder claim aliases.
    double compositeChromaClaim = 0.0;
    double iqEnvelopeClaim = 0.0;
};

inline double clamp01(double v)
{
    return std::clamp(v, 0.0, 1.0);
}

inline double combinedCarrierChromaCoherence(const CompositeOwnershipEvidence &e)
{
    return std::max({
        e.carrierChromaCoherence,
        e.carrierPhaseCoherence,
        e.compositeChromaCoherence,
        e.compositeCarrierCoherence,
        e.carrierPlausibility,
        e.sidebandCoherence
    });
}

inline double combinedEnvelopeChromaCoherence(const CompositeOwnershipEvidence &e)
{
    return std::max({
        e.iqChromaCoherence,
        e.iqEnvelopeCoherence,
        e.chromaEnvelopeCoherence
    });
}

inline double strongestChromaCoherence(const CompositeOwnershipEvidence &e)
{
    return std::max({
        combinedCarrierChromaCoherence(e),
        combinedEnvelopeChromaCoherence(e)
    });
}

inline double combinedCompositeChromaClaim(const CompositeOwnershipEvidence &e)
{
    return std::max({
        e.carrierChromaClaim,
        e.envelopeChromaClaim,
        e.sidebandChromaClaim,
        e.compositeChromaClaim,
        e.iqEnvelopeClaim
    });
}

inline void normalizeCompositeOwnershipClaims(CompositeOwnershipEvidence &e)
{
    e.lumaClaim = clamp01(e.lumaClaim);

    e.carrierChromaClaim = clamp01(e.carrierChromaClaim);
    e.envelopeChromaClaim = clamp01(e.envelopeChromaClaim);
    e.sidebandChromaClaim = clamp01(e.sidebandChromaClaim);

    e.compositeChromaClaim = clamp01(e.compositeChromaClaim);
    e.iqEnvelopeClaim = clamp01(e.iqEnvelopeClaim);

    const double chromaClaim = clamp01(combinedCompositeChromaClaim(e));

    e.uncertainClaim = clamp01(
        1.0 - std::max(e.lumaClaim, chromaClaim));
}

inline void applyOwnershipConflictSuppression(CompositeOwnershipEvidence &e,
                                              double conflictSuppress)
{
    e.lumaClaim = clamp01(e.lumaClaim);

    const double chromaClaim = clamp01(combinedCompositeChromaClaim(e));

    const double conflict = std::sqrt(
        std::max(0.0, e.lumaClaim * chromaClaim));

    e.ownershipConflict = std::max(e.ownershipConflict, conflict);

    const double scale = std::max(
        0.0,
        1.0 - clamp01(conflictSuppress) * conflict);

    e.lumaClaim *= scale;

    e.carrierChromaClaim *= scale;
    e.envelopeChromaClaim *= scale;
    e.sidebandChromaClaim *= scale;
    e.compositeChromaClaim *= scale;
    e.iqEnvelopeClaim *= scale;

    normalizeCompositeOwnershipClaims(e);
}

} // namespace lddecode