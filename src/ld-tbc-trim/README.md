# ld-tbc-trim
**TBC Trimmer and Decomposer**

## Overview
ld-tbc-trim trims and decomposes TBC files. In trim mode it extracts a specific range of frames from a source TBC, writing a new self-contained TBC and metadata pair. In decompose mode it splits a source TBC into multiple output pairs automatically, either at padded-frame boundaries or at edit points established by ld-cinemap.

## Usage

### Basic Syntax

**Trim mode** (exactly one of `-s`, `-cav`, or `-clv` required):
```bash
ld-tbc-trim -s <start_frame> [-l <length>] <input.tbc> <output.tbc>
ld-tbc-trim -cav <frame_number> [-l <length>] <input.tbc> <output.tbc>
ld-tbc-trim -clv <hh:mm:ss:ff> [-l <length>] <input.tbc> <output.tbc>
```

**Decompose mode** (output argument is used as the output name stem):
```bash
ld-tbc-trim --decompose <input.tbc> <stem>
ld-tbc-trim --decompose-edits <input.tbc> <stem>
```

## Options

#### Trim Selectors
- `-s, --seq <start_frame>`: Start at sequential frame number (1-based)
- `--cav <frame_number>`: Start at CAV picture number (VBI frame number)
- `--clv <hh:mm:ss:ff>`: Start at CLV timecode (hours:minutes:seconds:frame)

#### Trim Length
- `-l, --length <length>`: Number of frames to include (default: to end of source)

#### Decompose Modes
- `--decompose`: Split into one TBC/metadata pair per contiguous mappable span; padded and lead-in/out frames are excluded from output
- `--decompose-edits`: Split at edit boundaries established by ld-cinemap; padded frames are included within segments; lead-in/out frames excluded

#### Common Options
- `-h, --help`: Display help on command-line options
- `-v, --version`: Display version information

## Examples

```bash
# Trim by sequential frame number, 2400 frames from frame 12001
ld-tbc-trim -s 12001 -l 2400 input.tbc output.tbc

# Trim from a CAV picture number to end of source
ld-tbc-trim -cav 26000 input.tbc output.tbc

# Trim by CLV timecode, 2400 frames
ld-tbc-trim -clv 00:42:15:05 -l 2400 input.tbc output.tbc

# Decompose into one pair per contiguous content span
ld-tbc-trim --decompose input.tbc /path/to/disc

# Decompose at ld-cinemap edit points
ld-tbc-trim --decompose-edits input.tbc /path/to/disc
```

## Output Naming

### Trim mode
The output filename is specified directly as the positional argument:
- `output.tbc` and `output.tbc.db`

### Decompose modes
Output files are named from the stem argument with a zero-padded three-digit segment number:
- `stem_001.tbc` and `stem_001.tbc.db`
- `stem_002.tbc` and `stem_002.tbc.db`
- etc.

The segment counter overflows naturally beyond 999.

## Decompose Mode Details

### --decompose
Splits on padded frame boundaries. Each contiguous run of non-padded, non-lead-in/out frames becomes one segment. Use this to extract the mappable content from a disc map that contains padding gaps.

### --decompose-edits
Requires edit boundary data written by ld-cinemap. Splits wherever a field has `cinemap.inUse` and `isEditBoundary` set. The boundary frame opens the new segment.

If the edit boundary falls on a second field (a half-frame cut), the frame containing that field is included in both the closing segment and the opening segment. This ensures restorers receive complete frames at every boundary.

If no edit boundary frames are found, an error is emitted directing the user to run ld-cinemap first.

### End report
Both decompose modes print a summary on completion:
```
27 segment(s) created with 3 frame(s) repeated across 6 segment(s).
```

## Input/Output

### Input Format
- TBC file with associated SQLite metadata (`.tbc.db`)
- Metadata must have been processed by ld-process-vbi
- For `--decompose-edits`, metadata must have been processed by ld-cinemap

### Output Format
- Self-contained TBC and SQLite metadata pair(s)
- Fields are renumbered 1..N in each output so it stands alone without reference to the source

## Compatibility

The following options cannot be combined with `--decompose` or `--decompose-edits`:
- `-s`, `--seq`
- `--cav`
- `--clv`
- `-l`, `--length`

## Troubleshooting

### No edit boundary frames found
- Run ld-cinemap on the source TBC before using `--decompose-edits`
- Verify that ld-cinemap completed successfully and wrote edit points to the metadata

### Timecode not found
- Confirm the disc type is CLV; CAV discs use `--cav` instead
- Check the timecode format is exactly `hh:mm:ss:ff` (e.g. `00:42:15:05`)
- Verify ld-process-vbi was run and VBI data is present

### Start frame out of range
- Sequential frame numbers are 1-based
- Check the total frame count with ld-analyse or by inspecting the metadata

### Output length clamped
- If the requested length extends past the end of the source, it is automatically reduced and a warning is printed
