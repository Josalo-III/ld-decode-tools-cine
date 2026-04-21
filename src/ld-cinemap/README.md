# ld-cinemap
**Telecine Cadence Solver and Edit Detection**

## Overview
ld-cinemap detects film-originated content within LaserDisc TBC streams and solves the underlying telecine cadence pattern. It identifies edit boundaries (both structural phase discontinuities and visual cuts), detects 3:2 pulldown cadence patterns or interlaced/ progressive video, and records the relationship between film frames and interlaced video fields.

The tool is designed for restoration workflows where accurate cadence information enables high-quality film reconstruction, increasing SNR, improving error correction, allowing more informed comb in chroma decoder, and optionally 24p output.

## Usage

### Basic Syntax

```bash
ld-cinemap [OPTIONS] <input.tbc> [output]
```

## Modes

ld-cinemap supports three operational modes, selected by flags. If no mode flag is given, the **full pipeline** runs (default).

### Full Pipeline (default)
Runs segmentation → visual edit detection → cadence solving in sequence, writing all metadata in one pass.

```bash
ld-cinemap <input.tbc> [output]
```

### Edit Detection Only (`--detect-edits-only`)
Segments the disc by phase structure and runs visual edit detection, but does not solve cadence. Useful for vetting edit detection or for applying manual whitelist/blacklist overrides before solving.

```bash
ld-cinemap --detect-edits-only <input.tbc> [output]
```

Output: metadata with `isEditBoundary` flags set; cadence fields remain unset.

### Cadence Solve Only (`--skip-edits`)
Skips all edit detection and accepts any existing edit boundaries already present in the input metadata. Solves cadence using pre-annotated boundaries. Useful for re-solving after refining edit detection.

```bash
ld-cinemap --skip-edits --cine <input.tbc> [output]
```

Requires: input metadata must have `isEditBoundary` flags already set (by a prior run or manual annotation).

## Options

### Cadence Policy
Exactly one policy must be selected to control how cadence is resolved:

- **`--cine`**: Implement film-edited telecine reconstruction policy.  
  Optimised for discs where film was edited before telecining, producing cadence breaks only at reel changes. Produces 3:2 pulldown locks for material with occasional resets.

- **`--tv`** (default): Implement video-edited telecine policy.  
  Optimised for discs where the entire telecine sequence was recorded and then edited in video. Allows more flexible cadence interpretation and is more robust to noise.

### Visual Edit Detection
These options control the sensitivity and behaviour of visual edit detection (ignored with `--skip-edits`):

- **`--sensitivity <value>`** (default: 8.0)  
  Overall visual edit sensitivity. Lower values flag more potential edits; higher values are more conservative. Typical range: 4.0–12.0.

- **`--strong <value>`** (default: 1.5)  
  Multiplier for strong visual discontinuities. Used to weight obvious cuts more heavily in the detection pipeline.

- **`--peak <value>`** (default: 1.6)  
  Multiplier for peak visual discontinuities. Applied to the most extreme luminance differences.

- **`--edit-whitelist <keys>`**  
  Comma-separated or space-separated list of sequential frame keys to force-mark as edit boundaries, overriding sensitivity settings.  
  *Note: Implementation pending.*

- **`--edit-blacklist <keys>`**  
  Comma-separated or space-separated list of sequential frame keys to force-exclude from edit boundaries, overriding detection results.  
  *Note: Implementation pending.*

### Input/Output

- **`<input.tbc>`**: Path to input TBC file. Metadata expected at `<input.tbc>.tbc.db`.

- **`[output]`** (optional): Base path for output metadata. If omitted, output overwrites input metadata at `<input.tbc>.tbc.db`. If provided, output is written to `<output>.tbc.db`.  
  Special values: `-` means stdout (metadata only; TBC not duplicated).

### Metadata Management

- **`--clear-all-flags`**  
  Clears all solver-owned flags from the input metadata before running the selected mode:
  - `isEditBoundary`
  - `cadenceId`, `cadenceIndexPresumed`
  - `pulldownRole`

  Useful for starting fresh after a prior run, or for resetting between experimental parameter sweeps. The flag is applied first, before any other processing.

- **`-r, --reverse`**  
  Reverse field order (swap first and second fields) during processing. Used if the capture TBC has fields in the opposite order from the metadata.

### Prompts and Automation

- **`-y, --yes`**  
  Assume 'yes' for all interactive prompts. Useful for automated batch processing.  
  Currently used by: `--clear-all-flags` confirmation.

### Help and Version

- **`-h, --help`**: Display help on command-line options.
- **`-v, --version`**: Display version information (branch and commit).

## Examples

### Basic full pipeline on NTSC CAV disc
```bash
ld-cinemap --cine source.tbc output.tbc
```
Segments, detects visual edits, solves cadence using the film-edited policy, and writes metadata to `output.tbc.db`.

### Quick edit detection preview
```bash
ld-cinemap --detect-edits-only --sensitivity 6.0 source.tbc
```
Runs segmentation and visual edit detection with tighter sensitivity (more conservative), overwrites source metadata.

### Re-solve with different policy
```bash
ld-cinemap --skip-edits --tv source.tbc refined.tbc
```
Assumes edit boundaries are already in `source.tbc.tbc.db`, solves cadence using the video-edited policy, writes to `refined.tbc.db`.

### Start fresh, aggressive edit detection
```bash
ld-cinemap --clear-all-flags --sensitivity 4.0 --cine --yes source.tbc
```
Clears prior results (auto-confirm), runs full pipeline with aggressive edit detection and film-edited policy, overwrites source metadata.

### Export to separate file
```bash
ld-cinemap --cine source.tbc /archive/restored
```
Writes output metadata to `/archive/restored.tbc.db` (source TBC unmodified; metadata is separate).

## Output

### Metadata Fields
ld-cinemap writes the following fields to the SQLite metadata at frame/field level:

#### Edit Detection (all modes except `--skip-edits`)
- **`isEditBoundary`**: Boolean flag; true if a phase discontinuity or visual cut is detected at this field.

#### Cadence Solving (all modes except `--detect-edits-only`)
- **`cadenceId`**: Integer ≥ 0 identifying which film frame this field belongs to (within its cadence group). −1 if not locked.
- **`cadenceIndexPresumed`**: Boolean; true if the cadence assignment was interpolated rather than directly measured.
- **`pulldownRole`**: String identifier for the field's role within the 3:2 pulldown pattern (e.g. `"A"`, `"B"`, `"C"`), or empty if not in a pulldown pattern.

#### Global Metadata
- **`isCinemapped`**: Boolean flag set to true in the video parameters after any solve run, indicating that cadence data is present.

### Logging

ld-cinemap logs progress to stderr via Qt's logging framework:
- VBI scanning results (CAV vs CLV detection)
- Segmentation summary (boundaries found)
- Visual edit detection summary (edits committed)
- Cadence solver summary (fields locked)
- Output file paths

Example:
```
vbiProbe::probe: scanned 54000 frame(s) | CAV hits = 54000 | CLV hits = 0
segmenter::segmentDisc: marked 12 boundary(s).
Visual edit detection committed 8 edit boundary(s).
Cadence solver locked 108000 field(s).
Output metadata written to output.tbc.db
```

## Workflow Integration

### Typical Restoration Pipeline
1. **Capture and TBC generation**: Digitise LaserDisc with ld-decode, producing `.tbc` and `.tbc.db`.
2. **VBI processing**: Run ld-process-vbi to decode timecodes and picture numbers into metadata.
3. **Disc mapping**: Run ld-discmap to establish frame sequence and handle multi-sided or multi-disc sources.
4. **VITS processing**: Run ld-process-vits to measure per-field SNR and signal quality. While not strictly required, the stacking and SNR data written by this step is useful to ld-cinemap's cadence detection and solving — fields from a multi-source stack carry better signal quality metrics that can inform phase confidence.
5. **Cinemap analysis**: Run ld-cinemap to detect edits and solve cadence.
   ```bash
   ld-cinemap --cine source.tbc source.tbc
   ```
6. **Decompose** (optional): Use ld-tbc-trim to split at edit boundaries:
   ```bash
   ld-tbc-trim --decompose-edits source.tbc /segments/disc
   ```
7. **Frame reconstruction**: Run ld-chroma-decoder using cadence data to extract film frames at full quality.

### Running on Any TBC
ld-cinemap requires only a TBC file and its associated `.tbc.db` metadata. Steps 3 and 4 above improve results but are not mandatory — ld-cinemap will operate on any valid TBC/metadata pair produced by ld-decode and ld-process-vbi.

### Manual Oversight
- Inspect `isEditBoundary` flags with a metadata viewer or custom tooling.
- Use `--edit-whitelist` / `--edit-blacklist` (when available) to correct false positives/negatives.
- Re-run with different `--sensitivity` values to tune detection threshold.
- Use `--skip-edits` to keep boundary corrections and re-solve cadence with alternative policies.

## Disc Support

- **NTSC (525-line)**: Full support. Phase structure validation for 4-phase NTSC pulldown.
- **PAL (625-line)**: Edit segmentation only (VBI probe sets `isDiscPal`). Cadence solving deferred; 8-phase PAL structure not yet implemented.
- **CAV (Constant Angular Velocity)**: Full support. Uses VBI picture numbers for cadence lock.
- **CLV (Constant Linear Velocity)**: Full support. Uses VBI timecodes for alignment; picture numbers absent.

## Performance Notes

- **VBI scanning**: One pass over all frames; negligible cost.
- **Visual edit detection**: Frame-level luminance analysis; typically 1–5 minutes for a full disc on modern hardware.
- **Cadence solving**: Field-pair correlation and twin demodulation; typically 5–30 minutes depending on disc length and complexity.
- **Full pipeline**: Sequential; total time is sum of components plus metadata I/O.

For very large discs (>100K frames) or when re-solving frequently, use `--detect-edits-only` first, then `--skip-edits` for rapid iteration on cadence parameters.

## Metadata Format

Input and output metadata are SQLite 3 databases named `<tbcPath>.tbc.db`, conforming to the ld-decode metadata schema. The schema includes fields for:
- VBI data (picture numbers, timecodes, lead-in/out flags)
- Field phase ID and padding status
- Solver-owned columns: `isEditBoundary`, `cadenceId`, `cadenceIndexPresumed`, `pulldownRole`

See the ld-decode documentation for full schema details.

## Troubleshooting

### "no CAV or CLV VBI detected"
**Cause**: Input metadata lacks valid VBI decoding.  
**Solution**: Ensure ld-process-vbi was run on the source. Check that the TBC is a valid LaserDisc capture.

### No cadence locks found
**Cause**: Disc content is heavily interlaced, shot-on-video, or has extreme noise.  
**Solution**: Try `--tv` policy first (more permissive). Inspect visual edit detection results; edit boundaries may be misaligned.

### Spurious edit boundaries
**Cause**: Visual edit sensitivity too aggressive for noisy content.  
**Solution**: Increase `--sensitivity` value (try 10.0–12.0). Review detected boundaries with `--detect-edits-only` first. Use `--edit-blacklist` to suppress false positives (when available).

### Metadata not written
**Cause**: Output path is invalid or read-only.  
**Solution**: Check that the output directory exists and has write permissions. Ensure the output base path does not conflict with the input.

## See Also
- **ld-decode**: LaserDisc digitisation and TBC generation
- **ld-process-vbi**: VBI timecode and picture number decoding
- **ld-tbc-trim**: TBC trimming and decomposition at cadence boundaries
- **ld-chroma-decoder**: Chroma reconstruction using cadence data for film frame extraction
- **VideoForge**: Real-time film frame compositor with integrated cadence solver UI

## References
- NTSC 3:2 pulldown and field phase: *SMPTE RP 202* and related standards
- LaserDisc VBI encoding: *LaserDisc Technical Handbook*
- Telecine and cadence analysis: Internal ld-decode design documentation
