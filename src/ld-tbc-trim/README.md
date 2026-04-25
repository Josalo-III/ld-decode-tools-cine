# ld-tbc-trim
**TBC Trimmer and Decomposer**

## Overview
ld-tbc-trim trims and decomposes TBC files. In trim mode it extracts a specific range of frames from a source TBC, writing a new self-contained TBC and metadata pair. In decompose mode it splits a source TBC into multiple output pairs automatically, either at padded-frame boundaries or at edit points established by ld-cinemap.

## Usage

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
- `--cav <frame_number>`: Start at CAV picture number
- `--clv <hh:mm:ss:ff>`: Start at CLV timecode (hours:minutes:seconds:frame)
- `-l, --length <length>`: Number of frames to include (default: to end of source)

#### Decompose Modes
- `--decompose`: Split into one TBC/metadata pair per contiguous mappable span; padded and lead-in/out frames are excluded
- `--decompose-edits`: Split at edit boundaries established by ld-cinemap; padded frames are included within segments; lead-in/out frames are excluded. Requires ld-cinemap to have been run on the source first.

## Output Naming

### Trim mode
The output filename is specified directly: `output.tbc` and `output.tbc.db`.

### Decompose modes
Files are named from the stem with a zero-padded three-digit segment number:
`stem_001.tbc` / `stem_001.tbc.db`, `stem_002.tbc` / `stem_002.tbc.db`, etc.

## Notes on `--decompose-edits`

If the edit boundary falls on a second field (a half-frame cut), the containing frame is included in both the closing segment and the opening segment, ensuring complete frames at every boundary. The end-of-run summary reports how many frames were repeated and across how many segments.

## Examples

```bash
# Trim by sequential frame, 2400 frames from frame 12001
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
