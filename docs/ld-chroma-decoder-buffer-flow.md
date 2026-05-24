# ld-chroma-decoder Buffer Flow Diagram

This document maps the current decoder buffer flow by buffer family: scalar composite-domain buffers, IQ buffers, Y buffers, carrier grammar state, candidate scratch buffers, ownership diagnostics, and final component output. It describes what each buffer is, where it is filled, where it is consumed downstream, and why that buffer exists as a distinct varietal rather than being folded into another plane.

## Top-Level Flow

```text
SourceField pairs
    |
    v
rawbuffer                          [raw composite frame, interleaved fields]
    |
    +--> split1D()
    |       fills clpbuffer[0]      [1D scalar bandpass]
    |
    +--> phaseLocked()              [locked path only]
    |       fills carrierGrammar
    |       fills demodTRI/TRQ      [raw-composite IQ fit input]
    |       usually fills locked Y caches   [4fsc-cancelled Y / smoothed Y]
    |       (buildPhaseCorrected1D() can also backfill these if needed)
    |
    +--> split2D()
    |       may call buildPhaseCorrected1D()
    |       builds Field / Frame / FVF scalar candidates
    |       fills clpbuffer[1]      [selected 2D scalar bandpass]
    |
    +--> copy2DTo3D(), split3D()    [3D mode only]
    |       fills clpbuffer[2]      [selected 3D scalar bandpass]
    |
    +--> demod path
            bucket path:
                splitIQ() -> componentFrame U/V
                adjustY() -> componentFrame Y
                filterIQ()
            locked path:
                splitIQlocked() -> demodTI/TQ + lockedProductI/Q
                produceY()      -> componentFrame Y + refined lockedProductI/Q
                filterIQLocked()-> componentFrame U/V
    |
    v
transformIQ()
    componentFrame Y/U/V output
```

The current architecture has three broad buffer families:

```text
scalar buffers      composite-domain bandpass or comb candidate planes
IQ buffers          demodulated chroma-envelope analysis/render planes
Y buffers           luma anchors, luma structure references, and final luma
```

The complexity is mostly justified by preserving those domains instead of collapsing them too early.

---

# Persistent Frame Buffers

## `rawbuffer`

```text
Kind: raw composite
Shape: full interleaved frame
Filled by: loadFields()
Used by: split1D(), phaseLocked(), produceY(), filterIQLocked(residualColor), diagnostics
Why unique:
    This is the only untouched source waveform. Every scalar, Y, and IQ estimate
    is ultimately an explanation of rawbuffer.
```

`loadFields()` interleaves the two source fields into frame-line order and seeds field phase, cadence, edit-boundary, and carrier grammar schedule identity. The raw buffer is therefore not just samples; it is the source frame’s temporal and phase context.

---

## `clpbuffer[0]`

```text
Kind: scalar composite-bandpass
Name/role: 1D scalar bandpass plane
Filled by: split1D()
Also read/reinterpreted by: buildPhaseCorrected1D(), 2D tap builders, bucket splitIQ(), 3D candidate selection
Used downstream:
    - bucket demod: splitIQ()
    - locked 1D source construction: buildPhaseCorrected1D()
    - candidate baseline: getCandidate()
    - VDIS scalar leg
Why unique:
    This is the raw horizontal comb result: cheap, local, phase-grid scalar.
    It is the baseline “there is carrier-band energy here” plane before any
    vertical, temporal, or locked IQ interpretation.
```

Flow:

```text
rawbuffer
    -> split1D()
        clpbuffer[0]
```

This is not Y, not final chroma, and not IQ. It is a scalar composite-domain residual around the subcarrier. Bucket mode consumes it directly; locked mode uses it as an intermediate source for phase-corrected exports.

---

## `clpbuffer[1]`

```text
Kind: scalar composite-bandpass
Name/role: 2D-selected scalar plane
Filled by: split2D()
Used downstream:
    - copy2DTo3D()
    - splitIQ() in bucket mode when dimensions == 2
    - splitIQlocked() in locked mode when dimensions == 2
    - getBestCandidate agreement shaping
Why unique:
    This is the spatially adjudicated scalar chroma/bandpass estimate.
    It may represent Field, FieldB, FramePreclean, FrameRaw, or FVF output,
    depending on twoDVariant.
```

Flow:

```text
clpbuffer[0]
    -> FieldA / FieldB / FrameA / FrameB / FVF election
        -> clpbuffer[1]
```

---

## `clpbuffer[2]`

```text
Kind: scalar composite-bandpass
Name/role: 3D-selected scalar plane
Filled by:
    copy2DTo3D() seeds from clpbuffer[1]
    split3D() selectively overwrites temporal wins
Used downstream:
    - splitIQ() / splitIQlocked() when dimensions == 3
    - produceY() as selected scalar source in locked mode
Why unique:
    This is the temporal refinement layer. It starts as the 2D decision and
    only changes where temporal candidates beat the 2D baseline.
```

Flow:

```text
clpbuffer[1]
    -> copy2DTo3D()
        clpbuffer[2] initially equals clpbuffer[1]
            -> split3D()
                temporal candidate wins overwrite selected pixels
```

---

# Carrier Grammar and Phase-State Buffers

## `carrierGrammar`

```text
Kind: per-line carrier state, not image data
Filled by:
    loadFields(): schedule identity, line parity, fieldLine, lineFlip, samplePhase0
    phaseLocked(): burst phasor, carrierScale, phaseConfidence, demod LUTs
    phaseLocked()/buildPhaseCorrected1D(): affine, phase/error metrics, projection metrics
Used by:
    carrierSampleClass()
    carrierSignedSampleClass()
    carrierOppositeSampleClass()
    carrierLineFlip()
    remodGrammarToComposite()
    splitIQ(), adjustY(), splitIQlocked(), buildPhaseCorrected1D(), 2D/FVF, 3D candidate phase gates
Why unique:
    This is the line-level NTSC grammar. It prevents each consumer from
    rediscovering line phase and sample phase locally.
```

Flow:

```text
metadata fieldPhaseID + line index + burst measurement
    -> carrierGrammar[line]
        -> all phase-indexed demod/remod/scoring paths
```

The important split is:

```text
carrierSampleClass(line,h)
    -> unsigned 4fsc bucket
    -> used for LUT indexing and cancellation-window math

carrierSignedSampleClass(line,h)
    -> bucket plus lineFlip
    -> used for carrier-domain comparisons

carrierOppositeSampleClass(line,h)
    -> explicit opposite-carrier phase
    -> used by 3D candidate phase gate
```

---

## `spLUT_locked`, `cpLUT_locked`

```text
Kind: 4-phase shifted carrier basis
Filled by: phaseLocked(), lazily also splitIQlocked()
Used by:
    fusedDemodLUT()
    fallback locked demod
    shifted remod helpers
Why unique:
    This holds the fractional 4fsc basis trim (CAL_EPS_SAMPLES), separate from
    burst orientation. Preserving this trim is part of how the locked path reduces
    the chroma ellipse; additional per-axis gain trim happens later (e.g. GQ_PRODUCT).
    The burst phasor rotates the basis per line; the LUTs define the shared
    sample basis.
```

These are not picture buffers; they are per-run math tables.

---

# Locked IQ Buffers

## `demodTRI_flat`, `demodTRQ_flat`

```text
Kind: IQ, raw-composite demod
Filled by: phaseLocked() pass 2
Source: rawbuffer minus local DC
Used by:
    phaseLocked() sinusoidal fit / affine solve
    VDIS IQ leg indirectly via demodTI4fsc later, not normally these directly
Why unique:
    These are raw composite IQ observations before the selected comb result
    exists. They are used to calibrate the carrier/affine model, not to render
    final chroma.
```

Flow:

```text
rawbuffer - line DC
    -> demod through burst-locked LUT
        demodTRI/TRQ
            -> line affine / phase-error solve
```

This is an input-to-calibration IQ buffer.

---

## `demodTI_flat`, `demodTQ_flat`

```text
Kind: IQ, line-local locked basis
Filled by:
    buildPhaseCorrected1D() from clpbuffer[0]
    splitIQlocked() from selected clpbuffer[srcBuf]
Used by:
    produceY()
    lockedProduct derivation
    diagnostics
Why unique:
    This is the current line-local locked IQ representation. It is still in
    the burst-referenced locked basis, so it preserves the line’s measured
    carrier orientation.
```

There are two important moments:

```text
before 2D:
    buildPhaseCorrected1D()
        clpbuffer[0] -> demodTI/TQ as phase-corrected 1D basis

after 2D/3D:
    splitIQlocked()
        clpbuffer[srcBuf] -> demodTI/TQ refreshed from final selected scalar
```

The second refresh matters: once 2D/3D has selected the scalar bandpass, the IQ state must be regenerated from the actual selected scalar, not from the earlier 1D source.

---

## `locked1DSource`

```text
Kind: scalar, sign-carrying locked 1D source
Filled by: buildPhaseCorrected1D()
Source:
    line-local locked IQ after affine/iceberg adjustment
    remodulated through the shifted locked basis
Used by:
    buildCombTapLine() scalar taps in phase-compensated Field A/B
    lateral 1D baseline for FVF
Why unique:
    This is the scalar path that must keep the physical alternation the
    intrafield comb expects. It is deliberately not regenerated from the
    normalized demodTI4fsc/TQ4fsc export, because doing that collapses the
    sign relationship same-field subtraction needs.
```

Flow:

```text
clpbuffer[0]
    -> locked demod LUT + line affine / luma correction
        demodTI/TQ
            -> remodLockedToShiftedComposite()
                locked1DSource
```

---

## `demodTI4fsc_flat`, `demodTQ4fsc_flat`

```text
Kind: IQ, locked 4fsc-grid export
Filled by:
    buildPhaseCorrected1D()
    splitIQlocked()
Used by:
    frame IQ construction
    VDIS IQ leg
    clients that need a stable locked export before choosing a comparison frame
Why unique:
    This is the locked-path “4fsc-grid export” used by downstream analysis and
    candidate logic. This is not the scalar source for intrafield comb polarity;
    that role belongs to `locked1DSource`, which is built before this export
    normalizes the line-local locked basis into its 4fsc-grid IQ representation.
    Consumers that need a different sign convention must project explicitly where
    the comb geometry is known.

    The locked demod path bakes the fractional basis trim (`CAL_EPS_SAMPLES`)
    into its demod basis; `lockedTo4fsc()` preserves that trim. Any additional
    normalization or “always-common-frame” projection belongs in a separate
    helper at the call site for clients that need it.
```

Flow:

```text
line-local locked IQ
    -> lockedTo4fsc()  (pure rotation; no magnitude normalization)
        demodTI4fsc/TQ4fsc
            -> frame IQ, VDIS, tap IQ, FVF evidence
```

This is why it must exist separately from `demodTI/TQ`: it provides an explicit
locked 4fsc export without pretending it is the scalar physical alternation
carrier. `locked1DSource` carries that scalar obligation for intrafield combs.

---

## `locked1DTI4fsc_flat`, `locked1DTQ4fsc_flat`

```text
Kind: IQ, preserved 4fsc-grid export from phase-corrected 1D
Filled by: buildPhaseCorrected1D()
Used by:
    computeFrameIQLocked1DLine()
    computeFrameBLocked1DLine()
    Frame B / raw locked-1D frame path
Why unique:
    This preserves the pre-2D phase-corrected 1D IQ snapshot. Later stages
    repurpose demodTI4fsc/TQ4fsc, so Frame B needs a stable earlier cache.
```

Flow:

```text
clpbuffer[0]
    -> buildPhaseCorrected1D()
        -> locked1DTI4fsc/TQ4fsc
            -> computeFrameIQLocked1DLine()
                -> computeFrameBLocked1DLine()
                    -> frameIQ + frameScalar
```

This is a “do not overwrite my baseline” buffer. Without it, Frame B can accidentally read whatever the later final-selected demod path wrote.

---

## `lockedProductI_flat`, `lockedProductQ_flat`

```text
Kind: IQ, product-scaled locked output-prep cache
Filled by:
    splitIQlocked()
    refined by produceY()
Used by:
    filterIQLocked()
    eventually copied into componentFrame U/V after FIR
Why unique:
    This is the chroma product that should go to output filtering. It is not
    the 4fsc-grid analysis/export IQ and not the pre-2D 1D IQ. It is the selected,
    scaled (GI_PRODUCT/GQ_PRODUCT), potentially Y-vetted chroma estimate.
```

Flow:

```text
selected scalar clpbuffer[srcBuf]
    -> splitIQlocked()
        -> lockedProductI/Q
            -> produceY() may attenuate/refine per pixel
                -> filterIQLocked()
                    -> componentFrame.u/v
```

This buffer is the locked path’s chroma handoff to rendering.

---

# Locked Y Buffers

## `lockedLumaBaseY4_flat`

```text
Kind: Y
Name/role: 4fsc-cycle-cancelled coarse Y anchor
Filled by:
    phaseLocked()
    buildPhaseCorrected1D() lazy cache path
Used by:
    buildPhaseCorrected1D()
    produceY()
Why unique:
    This is the trusted low-resolution luma scaffold. It cancels the carrier
    cycle but necessarily loses high-frequency Y. It anchors residual analysis.
```

Flow:

```text
rawbuffer
    -> buildCompositeLumaDecompositionLine()
        -> lockedLumaBaseY4
            -> residual = raw - baseY4
                -> produceY() recovers contested high-frequency content
```

This is not final Y. It is “definitely Y, but incomplete.”

---

## `lockedLumaSmooth_flat`

```text
Kind: Y
Name/role: smoothed luma structure
Filled by:
    phaseLocked()
    buildPhaseCorrected1D()
Used by:
    buildPhaseCorrected1D() luma incursion / iceberg support
Why unique:
    This is a structural luma reference used to tell whether carrier-band
    residual has plausible Y continuation. It is not the output luma.
```

Flow:

```text
baseY4
    -> smoothed/structural Y
        -> luma edge / iceberg / incursion evidence
            -> ownershipEvidence + adjusted locked IQ
```

---

## `scratch_lumaBaseY4`, `scratch_lumaHiRaw`, `scratch_lumaSmooth`

```text
Kind: Y scratch
Filled by: buildCompositeLumaDecompositionLine()
Used by:
    buildPhaseCorrected1D()
    produceY()
Why unique:
    Per-line scratch equivalents of the persistent locked Y caches. Used when
    the flat cache is not valid or when a local call needs temporary Y components.
```

These are working buffers, not long-lived state.

---

# Component Output Buffers

## `componentFrame->y(line)`

```text
Kind: final/intermediate Y plane
Filled by:
    splitIQ() initializes to raw in bucket mode
    adjustY() subtracts remodulated bucket chroma
    produceY() writes locked-path luma
    doYNR() may modify
Used by:
    filterIQLocked(residualColor)
    output writer
Why unique:
    This is the actual component luma shipped downstream.
```

Bucket flow:

```text
rawbuffer
    -> splitIQ(): Y = raw
    -> adjustY(): Y -= reconstructed chroma
    -> doYNR()
    -> output
```

Locked flow:

```text
baseY4 + high-frequency residual adjudication
    -> produceY()
        componentFrame Y
            -> doYNR()
            -> output
```

---

## `componentFrame->u(line)`, `componentFrame->v(line)`

```text
Kind: final/intermediate chroma planes
Filled by:
    bucket path: splitIQ(), then filterIQ(), doCNR(), transformIQ()
    locked path: filterIQLocked() from lockedProductI/Q, then doCNR(), transformIQ()
Used by:
    transformIQ()
    output writer
Why unique:
    These are the final component chroma planes. They are no longer merely
    analysis IQ once transformIQ() has rotated/scaled them for output.
```

Before `transformIQ()`, these are decoder-internal I/Q-like planes. After `transformIQ()`, they are the output chroma representation expected by the rest of the pipeline.

---

# 2D Scalar Candidate Buffers

## `scratch_fieldLine`

```text
Kind: scalar
Name/role: Field A candidate
Filled by: computeField2DLine()
Used by:
    split2D() direct Field variant
    scoreFieldVsFrame()
    collectCombOwnershipEvidence()
Why unique:
    This is the longer-reach intrafield candidate, using same-field vertical
    context and contour/gate logic. It tends to preserve interlace-safe vertical
    detail but must be gated around vertical/horizontal structure.
```

Flow:

```text
CombTapLine ±2/±4 scalar/IQ evidence
    -> computeField2DLine()
        scratch_fieldLine + scratch_fieldGate
```

---

## `scratch_fieldGate`

```text
Kind: scalar confidence/gate
Filled by: computeField2DLine()
Used by: scoreFieldVsFrame()
Why unique:
    It is not a picture signal. It says how much the Field A candidate should
    be trusted at that pixel.
```

---

## `scratch_fieldBLine`

```text
Kind: scalar
Name/role: Field B / simple intrafield candidate
Filled by:
    computeSimpleField2DLine()
    or copied from precleanRing
Used by:
    split2D() direct FieldB variant
    preclean source for frame IQ
    scoreFieldVsFrame()
    fallback when FVF output is non-finite
Why unique:
    This is the conservative intrafield candidate. It is also used as the
    scalar preclean material for frame-comb construction.
```

Flow:

```text
same-field ±2 simple cancellation
    -> scratch_fieldBLine
        -> precleanRing
        -> FramePreclean path
        -> FVF fallback/reference
```

---

## `scratch_lateralLine`

```text
Kind: scalar
Name/role: lateral/1D selected baseline for FVF
Filled by:
    locked mode: locked1DSource
    bucket mode: clpbuffer[0]
Used by: scoreFieldVsFrame()
Why unique:
    This gives FVF a same-line baseline independent of vertical/frame claims.
    It is the “do not get worse than local 1D unless evidence supports it”
    reference.
```

---

## `scratch_frameBCenter`

```text
Kind: scalar, overloaded by context
Main roles:
    1. FrameRaw / Frame B scalar candidate
    2. temporary baseY4 or hiRaw holder in produceY()
Filled by:
    computeFrameBLocked1DLine() in split2D()
    produceY() local copies
Used by:
    FrameRaw variant
    scoreFieldVsFrame()
    collectCombOwnershipEvidence()
    produceY() local Y residual math
Why unique:
    In split2D it represents the raw locked-1D frame-comb scalar candidate.
    In produceY it is reused as per-line scratch for Y decomposition.
```

This one is a naming hazard. In the 2D path it means “Frame B center scalar.” In `produceY()` it is just storage reused for `baseY4`.

---

## `scratch_fieldBCenter`

```text
Kind: scalar, overloaded by context
Main roles:
    1. remodulated FramePreclean IQ scalar
    2. temporary hiRaw holder in produceY()
Filled by:
    computeFrameIQPrecleanLine() + remod in split2D()
    produceY() local raw-base residual setup
Used by:
    FramePreclean variant
    FVF framePreclean input
    collectCombOwnershipEvidence()
Why unique:
    In split2D it is the scalar version of the precleaned frame-IQ path.
    It differs from scratch_frameBCenter because it is built after FieldB
    precleaning, not directly from locked 1D.
```

---

## `scratch_outMixed`

```text
Kind: scalar
Name/role: FVF elected output line
Filled by: scoreFieldVsFrame()
Used by:
    split2D() writes it into clpbuffer[1]
    produceY() may reuse as local matrix scratch
Why unique:
    This is the result of policy, not a raw candidate. It is where Field A,
    Field B, FramePreclean, FrameRaw, and lateral 1D are adjudicated.
```

Flow:

```text
Field A + Field B + FramePreclean/FrameRaw + lateral1D + ownership/FVF metrics
    -> scoreFieldVsFrame()
        scratch_outMixed
            -> clpbuffer[1]
```

---

# Frame IQ Candidate Buffers

## `frameIQ`

```text
Kind: IQ, temporary line vector
Filled by:
    computeFrameBLocked1DLine()
    computeFrameIQPrecleanLine()
Used by:
    collectCombOwnershipEvidence()
    scoreFieldVsFrame()
Why unique:
    It carries the IQ explanation behind a frame candidate, while the scalar
    remodulated version is passed separately. FVF can use IQ coherence even
    though the final scalar election writes scalar bandpass.
```

---

## `frameIQPreclean`

```text
Kind: IQ, temporary line vector
Filled by: computeFrameIQPrecleanLine()
Used by:
    remodulated into scratch_fieldBCenter
Why unique:
    This is FramePreclean’s IQ before it becomes scalar. It tests whether
    precleaned FieldB material gives a better interfield chroma envelope.
```

Flow:

```text
precleanRing / demodTI4fsc fallback
    -> computeFrameIQPrecleanLine()
        frameIQPreclean
            -> remod4fscToCompositePhase()
                scratch_fieldBCenter
```

---

## `scratch_centerIQ`, `scratch_upIQ`, `scratch_dnIQ`

```text
Kind: IQ scratch
Filled by:
    computeFrameIQPrecleanLine()
    computeFrameIQLocked1DLine()
Used by:
    computeFrameIQFromPreparedVectors()
Why unique:
    These are per-line staging vectors for center/up/down IQ before the frame
    comb decides how to blend or reject vertical-frame chroma evidence.
```

These are 4fsc IQ vectors, not scalar composite samples. They are used when the
code wants to reason about chroma as an envelope. Any extra projection into a
comparison frame should happen in the consumer that needs it.

---

# Preclean Buffers

## `precleanRing[3]`

```text
Kind: scalar ring cache
Name/role: FieldB preclean scalar cache
Filled by:
    split2D() via computeSimpleField2DLine()
Used by:
    computeFrameIQPrecleanLine()
Why unique:
    FramePreclean needs adjacent lines after FieldB precleaning. The ring keeps
    only the current/adjacent line neighborhood instead of storing another full
    frame plane.
```

Flow:

```text
FieldB scalar
    -> precleanRing[line % 3]
        -> computeFrameIQPrecleanLine()
```

---

## `precleanGateRing[3]`

```text
Kind: scalar gate ring cache
Filled by: split2D()
Used by: currently paired with precleanRing; mostly simple 1.0 fill in this path
Why unique:
    Intended to carry trust/gating alongside the FieldB preclean scalar so
    precleaned frame logic can know whether the source scalar was trustworthy.
```

At present the shown path fills it as `1.0`, so it is structurally present but not heavily exploited.

---

# Tap Cache Buffers

## `tapLineCache[3]`, `scratch_tapLine`

```text
Kind: scalar + IQ + metrics cache
Filled by:
    ensureCombTapLine()
    buildCombTapLine()
Used by:
    computeField2DLine()
    computeSimpleField2DLine()
    computeFrameScalarLine()
    scoreFieldVsFrame()
Why unique:
    This centralizes reusable vertical-neighbor harvest. Instead of every
    candidate re-reading raw/clp/IQ neighbors differently, the tap cache gathers
    center/up/down ±1/±2/±4 scalar and IQ facts once.
```

---

## `CombTapLine.tap0`, `tapU1`, `tapD1`, `tapU2`, `tapD2`, `tapU4`, `tapD4`

```text
Kind: scalar tap vectors
Filled by: buildCombTapLine()
Used by:
    FieldB: center ±2
    FieldA: center ±2 ±4
    Frame: center ±1
Why unique:
    They separate neighbor geometry by semantic role:
        ±1 = opposite field / frame-comb neighborhood
        ±2 = same field immediate vertical neighborhood
        ±4 = same field wider contour neighborhood
```

---

## `CombTapLine.tap*IQ`

```text
Kind: IQ tap vectors
Filled by: buildCombTapLine() when phaseCompensation is enabled
Used by:
    pair coherence
    frame/field evidence
    FVF scoring support
Why unique:
    Scalar similarity is not enough in saturated color. These buffers retain
    IQ-vector agreement so vertical/frame candidates can be judged by chroma
    envelope coherence, not just scalar amplitude.
```

---

## `CombTapLine.pairU1/pairD1/pairU2/pairD2`

```text
Kind: scalar/IQ metric vectors
Filled by: buildCombTapLine()
Used by:
    Field and Frame scoring
Why unique:
    These are already-reduced agreement facts: diff, IQ diff, coherence, kScore,
    weight. They prevent every candidate path from re-deriving the same local
    geometry differently.
```

---

## `CombTapLine.contour`

```text
Kind: scalar contour evidence
Filled by: buildCombTapLine()
Used by: computeField2DLine()
Why unique:
    Field A needs wider same-field context to avoid combing across horizontal
    or vertical structure. Contour evidence gates vertical reach.
```

---

## `CombTapLine.hLumaDeltaIRE`

```text
Kind: Y-ish scalar metric
Filled by: buildCombTapLine()
Used by: Field/FVF gating
Why unique:
    It records local horizontal luma change as an edge-risk signal. It is not
    image Y; it is a metric that protects comb decisions from luma edges.
```

---

# Ownership and FVF Diagnostics

## `ownershipEvidence[line][rel]`

```text
Kind: evidence records, not image
Filled by:
    seedCombOwnershipPerLine()
    buildPhaseCorrected1D()
    collectCombOwnershipEvidence()
    finalizeOwnershipClaims()
Used by:
    scoreFieldVsFrame()
    produceY()
    diagnostics/UI accessors
Why unique:
    This preserves why energy is believed to be luma-owned, chroma-owned, or
    uncertain. It keeps policy separate from raw candidate generation.
```

Flow:

```text
carrier projection + bandpass shape + field/frame candidates + IQ evidence
    -> ownershipEvidence facts/assessment
        -> FVF scoring
        -> produceY alpha ownership adjustment
```

---

## `fvfMetrics[line][rel]`

```text
Kind: diagnostic/model metrics
Filled by:
    buildPhaseCorrected1D()
    scoreFieldVsFrame()
Used by:
    getFvfMetrics()
    map/debug reasoning
Why unique:
    This records model-context facts such as Nyquist risk, boundary risk,
    field/frame divergence, IQ coherence, and winner class. It is explanatory
    state, not render state.
```

---

## `w2d_frame_weight[line][rel]`

```text
Kind: diagnostic/display scalar
Filled by:
    split2D()
    scoreFieldVsFrame()
    produceY()
Used by:
    overlay/showMap
Why unique:
    This is a visualization side channel. It explains candidate weighting or
    Y-subtraction alpha; it should not be treated as signal.
```

---

## `w2d_fieldA_gate[line][rel]`

```text
Kind: scalar gate/diagnostic
Filled by constructor allocation; FieldA logic may populate via field gate paths
Used by FVF/showMap-related paths
Why unique:
    This persists FieldA trust/gate information separate from the selected
    scalar output.
```

---

# VDIS Buffers

## `scratch_vdis_flag`

```text
Kind: per-line mask scratch
Filled by: computeVDISLine()
Used by: split2D() to populate vdisMask
Why unique:
    Temporary per-line detection result. It combines scalar ±2 disagreement
    and IQ ±1 phase disagreement before consolidation.
```

---

## `vdisMask[line][rel]`

```text
Kind: persistent per-frame mask
Values:
    0 = no VDIS
    1 = soft VDIS
    2 = hard VDIS
Filled by:
    split2D(): computeVDISLine() per line
    consolidateVDISRegions()
Used by:
    split3D()
    getCandidate()
    FVF/2D protection paths
Why unique:
    This is a veto/isolation map for regions where vertical reasoning is unsafe.
    It is not chroma, Y, or IQ; it is a safety mask.
```

VDIS uses both scalar disagreement from `clpbuffer[0]` and IQ phase disagreement from `demodTI4fsc/TQ4fsc`, then consolidates local clusters.

---

# Locked-Path Local Scratch Buffers

## `scratch_preI`, `scratch_preQ`

```text
Kind: IQ scratch
Filled by:
    buildPhaseCorrected1D()
    splitIQlocked()
    filterIQLocked()
Used by:
    local FIR prep
    local IQ analysis
Why unique:
    These are reusable per-line staging buffers. They avoid allocations and
    keep intermediate IQ separate from persistent demod caches.
```

---

## `scratch_preI_ext`, `scratch_preQ_ext`

```text
Kind: IQ scratch, edge-extended
Filled by: filterIQLocked()
Used by: filterIQLocked() FIR convolution
Why unique:
    FIR filtering needs padded edge samples. These buffers prevent edge handling
    from contaminating the actual product IQ cache.
```

---

## `scratch_comp_res`

```text
Kind: scalar residual
Filled by: produceY()
Meaning: composite residual / high-frequency residual
Used by:
    produceY() vet and alpha solve
Why unique:
    This is the contested residual after coarse Y. It is the local battlefield
    where chroma subtraction and lost-Y recovery are negotiated.
```

---

## `scratch_yhp`, `scratch_yI`, `scratch_yQ`

```text
Kind: Y-derived / IQ-derived scratch
Filled by: produceY()
Used by:
    local matrix/affine solve and HP-Y leakage logic
Why unique:
    These represent high-frequency Y structure projected into IQ-like reasoning.
    They exist because some carrier-band energy is actually luma structure.
```

---

## `scratch_hpI`, `scratch_hpQ`, `scratch_hpY`

```text
Kind: high-pass scratch
Filled by:
    doCNR()
    doYNR()
    produceY()
Used by:
    chroma NR
    luma NR
    local produceY matrices
Why unique:
    They hold high-frequency components temporarily so NR and vet logic can
    subtract/cap detail without overwriting the base planes prematurely.
```

---

## `scratch_filter_temp`

```text
Kind: generic line scratch
Filled/used by:
    filterIQ()
    buildPhaseCorrected1D()
    produceY()
    FIR/temp operations
Why unique:
    General-purpose reusable single-line work buffer. It has no stable signal
    meaning outside the function currently using it.
```

---

## `scratch_sinfit_mag`, `scratch_sinfit_resmag`

```text
Kind: scalar metric scratch
Filled by: phaseLocked()
Used by: phaseLocked() sinusoidal fit / affine solve
Why unique:
    These support carrier calibration. They are not picture buffers.
```

---

# Output Transform and NR Buffers

## `scratch_hpI`, `scratch_hpQ` in `doCNR()`

```text
Kind: high-pass chroma scratch
Filled by: FIR high-pass of componentFrame U/V
Used by: doCNR() to subtract clamped chroma noise
Why unique:
    Keeps chroma noise reduction non-destructive until the clamp decision is made.
```

---

## `scratch_hpY` in `doYNR()`

```text
Kind: high-pass Y scratch
Filled by: FIR high-pass of componentFrame Y
Used by: doYNR() to subtract clamped luma noise
Why unique:
    Same concept as chroma NR, but on luma.
```

---

# The Two Main Buffer Families, Simplified

## Scalar Family

```text
rawbuffer
    |
    v
clpbuffer[0]          1D scalar bandpass
    |
    +--> locked1DSource       phase-corrected scalar export, locked only
    |
    v
2D candidates:
    scratch_fieldLine         Field A
    scratch_fieldBLine        Field B / preclean
    scratch_fieldBCenter      FramePreclean scalar
    scratch_frameBCenter      FrameRaw scalar
    scratch_lateralLine       1D/lateral baseline
    scratch_outMixed          FVF elected scalar
    |
    v
clpbuffer[1]          selected 2D scalar
    |
    v
clpbuffer[2]          selected 3D scalar, if enabled
    |
    v
selected scalar source for demod:
    clpbuffer[dimensions - 1]
```

Why these scalar varietals exist:

```text
clpbuffer[0]      local horizontal evidence
FieldA            richer same-field vertical estimate
FieldB            safer same-field vertical estimate
FramePreclean     interfield estimate after FieldB scalar cleanup
FrameRaw          interfield estimate from locked 1D IQ
FVF mixed         policy arbitration among the above
clpbuffer[1]      selected 2D scalar
clpbuffer[2]      selected 3D scalar
```

## IQ Family

```text
rawbuffer
    |
    +--> demodTRI/TRQ             raw-composite IQ for carrier/affine solve
    |
clpbuffer[0]
    |
    +--> buildPhaseCorrected1D()
            demodTI/TQ           line-local locked 1D IQ
            demodTI4fsc/TQ4fsc   locked 4fsc-grid export
            locked1DTI4fsc/TQ4fsc preserved pre-2D locked 4fsc-grid export
            locked1DSource       remodulated scalar export
    |
selected clpbuffer[srcBuf]
    |
    +--> splitIQlocked()
            demodTI/TQ           refreshed from final selected scalar
            demodTI4fsc/TQ4fsc   refreshed selected locked export
            lockedProductI/Q     scaled output-prep IQ
    |
produceY()
    |
    +--> lockedProductI/Q refined/attenuated
    |
filterIQLocked()
    |
    +--> componentFrame U/V
```

Why these IQ varietals exist:

```text
demodTRI/TRQ          calibrate from raw composite before comb selection
demodTI/TQ            line-local burst-locked IQ for current selected scalar
demodTI4fsc/TQ4fsc    locked 4fsc-grid export
locked1DTI4fsc/TQ4fsc preserved early 1D 4fsc-grid export for Frame B
lockedProductI/Q      output-prep IQ after product scaling and Y arbitration
componentFrame U/V    final chroma planes after FIR and output transform
```

## Y Family

```text
rawbuffer
    |
    +--> lockedLumaBaseY4       coarse 4fsc-cancelled Y
    +--> lockedLumaSmooth       structural/smoothed Y
    |
    v
produceY()
    residual = raw - baseY4
    cHat     = remodulated IQ chroma hypothesis
    Y        = baseY4 + (hiRaw - alpha * cHat)
    |
    v
componentFrame Y
```

Why these Y varietals exist:

```text
lockedLumaBaseY4      trusted but incomplete Y anchor
lockedLumaSmooth      structure reference for luma-continuation evidence
scratch_luma*         local versions of the above
componentFrame Y      final rendered luma
scratch_comp_res      contested residual used to decide how much chroma to subtract
```

---

# Practical Cleanup Observations

The current buffer architecture is mostly coherent, but several names now hide changed responsibilities.

Most useful renames or comments would be:

```text
scratch_frameBCenter
    split2D meaning: frameRawScalar
    produceY meaning: baseY4Line
    Recommendation: split into separate named scratch aliases or add local references
    that make the reuse explicit.

scratch_fieldBCenter
    split2D meaning: framePrecleanScalar
    produceY meaning: hiRawLine
    Recommendation: same as above.

demodTI4fsc/TQ4fsc
    current selected locked 4fsc-grid export

locked1DTI4fsc/TQ4fsc
    preserved pre-2D 4fsc-grid export
    Recommendation: keep both. This distinction is important and justified.

lockedProductI/Q
    output-prep IQ
    Recommendation: keep separate from demodTI/TQ. This prevents render-path
    scaling/vetting from contaminating analysis IQ.

locked1DSource
    scalar remodulated early locked 1D
    Recommendation: keep. It is the scalar sibling of locked1DTI4fsc/TQ4fsc,
    and gives FVF/lateral logic a stable locked 1D baseline.
```

The central architectural rule is sound: scalar planes are candidate/output bandpass estimates; IQ planes are analysis/render chroma envelopes; Y buffers are either coarse anchors or final component luma. The buffer complexity comes from preserving those domains instead of collapsing them too early.
