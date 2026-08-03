# ld-cinemap
**Telecine Cadence Solver and Edit Detection**

## Overview
ld-cinemap detects film-originated content within LaserDisc TBC streams and solves the underlying telecine cadence pattern. It identifies edit boundaries (structural phase discontinuities and visual cuts), classifies content as 3:2 pulldown, interlaced, or progressive, and records the relationship between film frames and interlaced video fields — enabling high-quality film reconstruction, improved comb filtering, and optional 24p output.

## Usage

```bash
ld-cinemap [OPTIONS] <input.tbc> [output]
```

## Modes

If no mode flag is given, the **full pipeline** runs.

### Full Pipeline (default)
Segmentation → visual edit detection → cadence solving, written in one pass.

```bash
ld-cinemap <input.tbc> [output]
```

### Edit Detection Only (`--detect-edits-only`)
Segments by phase structure and runs visual edit detection; does not solve cadence. Use to vet edit detection or apply whitelist/blacklist overrides before solving.

```bash
ld-cinemap --detect-edits-only <input.tbc> [output]
```

Output: metadata with `isEditBoundary` flags set; cadence fields unset.

### Cadence Solve Only (`--skip-edits`)
Skips edit detection and solves cadence using existing edit boundaries. Use to re-solve after refining detection without re-running the full pipeline.

```bash
ld-cinemap --skip-edits --cine <input.tbc> [output]
```

Requires `isEditBoundary` flags already present in input metadata.

### Override Only (`--override-only`)
Applies manual edit whitelist/blacklist and cadence overrides, then writes metadata. Does not run segmentation, visual edit detection, or cadence solving.

```bash
ld-cinemap --override-only --cadence-override 12340-12419:0 <input.tbc> [output]
```

## Options

### Cadence Policy

- **`--cine`**: Film-edited policy. Optimised for discs where film was edited before telecine; produces 3:2 pulldown locks with cadence breaks only at reel changes.
- **`--tv`** (default): Video-edited policy. Optimised for footage telecined then edited in video; solves each shot separately.
- **`--cadence-override <start-end:cid>`**: Manually write cadence IDs over a 1-based field-number range matching ld-analyse's field display. `cid` is the first field's `cadenceId`; known cadence IDs (`0..9`, `10..19`) advance through the NTSC sequence, while `-1`, `-2`, and `-3` fill the range flat. May be repeated.

### Visual Edit Detection
Ignored with `--skip-edits` and `--override-only`.

- **`--sensitivity <value>`** (default: 8.0) — Overall sensitivity. Lower flags more edits; higher is more conservative. Typical range: 4.0–12.0.
- **`--strong <value>`** (default: 1.5) — Multiplier for strong discontinuities.
- **`--peak <value>`** (default: 1.6) — Multiplier for peak luminance differences.
- **`--edit-whitelist <keys>`** — Comma-separated field seqNo keys to force-mark as edit boundaries.
- **`--edit-blacklist <keys>`** — Comma-separated field seqNo keys (ranges accepted) to veto as edit boundaries. A veto is permanent and survives later runs: no detection or solver pass can re-assert a boundary on a vetoed field, and the veto travels with the metadata into any output file. Clear one with `--edit-whitelist` on the same field, or all of them with `--clear-all-flags`.

### Input/Output

- **`<input.tbc>`** — Input TBC file. Metadata read from `<input.tbc>.tbc.db`.
- **`[output]`** (optional) — Output base path; metadata written to `<output>.tbc.db`. If omitted, input metadata is overwritten.

### Metadata Management

- **`--clear-all-flags`** — Clears all solver-owned flags (`isEditBoundary`, `cadenceId`, `cadenceIndexPresumed`, `pulldownRole`) **and every manual edit veto** before running. This is the only command that discards a veto. Applied before any other processing.
- **`--clear-edits`** — Clears edit boundaries only, keeping manual edit vetoes standing. Use this for a fresh detection pass that still honours the edits you have already shut down. Mutually exclusive with `--clear-all-flags`.
- **`-r, --reverse`** — Swap first and second fields during processing.
- **`-y, --yes`** — Assume yes for all prompts (used by `--clear-all-flags` confirmation).
- **`-h, --help`** / **`-v, --version`**

## Examples

```bash
# Full pipeline, film-edited policy
ld-cinemap --cine source.tbc output.tbc

# Preview edit detection with tighter sensitivity
ld-cinemap --detect-edits-only --sensitivity 6.0 source.tbc

# Re-solve with different policy, keeping existing boundaries
ld-cinemap --skip-edits --tv source.tbc refined.tbc

# Re-solve, then manually force a cadence sequence over a problem range
ld-cinemap --skip-edits --cadence-override 12340-12419:0 source.tbc

# Patch only the manual override metadata, leaving detection and solve output untouched
ld-cinemap --override-only --cadence-override 12340-12419:0 source.tbc

# Start fresh, aggressive detection, auto-confirm
ld-cinemap --clear-all-flags --sensitivity 4.0 --cine --yes source.tbc

# Write output to a separate file
ld-cinemap --cine source.tbc /archive/restored
```

## Output

### Metadata Fields

Written to the SQLite `.tbc.db` at field level:

**Edit detection** (full pipeline and `--detect-edits-only`):
- `isEditBoundary` — true at phase discontinuities and visual cuts.
- `isEditVetoed` — true where the user has vetoed a boundary via `--edit-blacklist`.

These two are the tri-state `is_edit_boundary` column, not two columns: NULL means no determination, `1` means a boundary was asserted, `0` means a manual veto. A veto outranks every automatic determination — detection and solver passes reach the boundary flag only through an accessor that refuses to overwrite one — so false positives can be shut down once and stay down.

**Cadence solving** (full pipeline and `--skip-edits`):
- `cadenceId` — film frame identity within the cadence group; −1 if unsolved.
- `cadenceIndexPresumed` — true if the assignment was interpolated.
- `pulldownRole` — field's role in the 3:2 pattern (`"definitional"`, `"spare"`, or empty).

**Manual overrides** (`--override-only`, or after detection/solve in other modes):
- `--edit-whitelist` — forces `isEditBoundary` and lifts any standing veto on that field.
- `--edit-blacklist` — vetoes the field. Permanent until `--edit-whitelist` or `--clear-all-flags`.
- `--cadence-override` — writes `cadenceId`, clears `cadenceIndexPresumed`, and refreshes `pulldownRole`.

**Global**:
- `isCinemapped` — set in video parameters after any solve run, or after a cadence override writes at least one field.

Input and output metadata are SQLite 3 databases conforming to the ld-decode schema. See ld-decode documentation for full schema details.

### Logging

Progress is logged to stderr:
```
vbiProbe::probe: scanned 54000 frame(s) | CAV hits = 54000 | CLV hits = 0
segmenter::segmentDisc: marked 12 boundary(s).
Visual edit detection committed 8 edit boundary(s).
Cadence solver locked 108000 field(s).
Output metadata written to output.tbc.db
```

Enable `--cinemap-trace` to include the detailed CineMap decision explainer and the per-segment summary lines.

## Workflow Integration

1. **Capture** — Digitise with ld-decode, producing `.tbc` and `.tbc.db`.
2. **VBI** — Run ld-process-vbi to decode timecodes and picture numbers.
3. **Disc mapping** — Run ld-discmap to establish frame sequence.
4. **VITS** (recommended) — Run ld-process-vits for per-field SNR data; improves cadence confidence on stacked sources.
5. **Cinemap** — Run ld-cinemap to detect edits and solve cadence.
   ```bash
   ld-cinemap --cine source.tbc source.tbc
   ```
6. **Decompose** (optional) — Split at edit boundaries with ld-tbc-trim:
   ```bash
   ld-tbc-trim --decompose-edits source.tbc /segments/disc
   ```
7. **Reconstruction** — Run ld-chroma-decoder using cadence data for film frame extraction.

Steps 3 and 4 improve results but are not mandatory; ld-cinemap will operate on any valid TBC/metadata pair from ld-decode and ld-process-vbi.

## Disc Support

- **NTSC (525-line)**: Full support.
- **PAL (625-line)**: Edit segmentation only; cadence solving not yet implemented.
- **CAV**: Full support; uses VBI picture numbers for cadence lock.
- **CLV**: Full support; uses VBI timecodes for alignment.

## Performance

- **VBI scanning**: Negligible.
- **Visual edit detection**: 1–5 minutes per disc on modern hardware.
- **Cadence solving**: 5–30 minutes depending on disc length and complexity.

For large discs (>100K frames) or frequent re-solves, use `--detect-edits-only` then `--skip-edits` to iterate on cadence parameters without re-running detection.

## Troubleshooting

**"no CAV or CLV VBI detected"** — Run ld-process-vbi on the source first.

**No cadence locks found** — Content may be shot-on-video or heavily noisy. Try `--tv` policy; check that edit boundaries are correctly placed.

**Spurious edit boundaries** — Increase `--sensitivity` (try 10.0–12.0); use `--detect-edits-only` to review, then `--edit-blacklist` to suppress false positives.

**Metadata not written** — Check output directory exists and is writable.

## See Also
- **ld-decode** — LaserDisc digitisation and TBC generation
- **ld-process-vbi** — VBI timecode and picture number decoding
- **ld-tbc-trim** — TBC trimming and decomposition at cadence boundaries
- **ld-chroma-decoder** — Chroma reconstruction using cadence data
- **VideoForge** — Film frame compositor with integrated cadence solver UI

## References
- NTSC 3:2 pulldown and field phase: *SMPTE RP 202* and related standards
- LaserDisc VBI encoding: *LaserDisc Technical Handbook*
- Telecine and cadence analysis: Internal ld-decode design documentation
