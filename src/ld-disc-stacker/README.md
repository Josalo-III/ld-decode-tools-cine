# ld-disc-stacker

**Multi-Source TBC Stacking and Combination**

## Overview

ld-disc-stacker combines multiple TBC captures of the same LaserDisc to produce a superior output. By analyzing corresponding fields from multiple sources, it selects the best data for each field, effectively reducing dropouts and improving overall signal quality.

## Usage

### Basic Syntax
```bash
ld-disc-stacker [options] <source1.tbc> <source2.tbc> [...] <output.tbc>
```

## Options

#### Common Options
- `-h, --help`: Display help on command-line options
- `-v, --version`: Display version information
- `-d, --debug`: Show debug information
- `-q, --quiet`: Suppress info and warning messages
- `--help-mode`: Show detailed info about stacking modes
- `-V, --verbose`: Show more info during stacking

#### Input Sources
- `inputs`: Input TBC files — first can be `-` for piped input (required, 2+ sources)
- `output`: Combined output TBC (omit or `-` for piped output)

#### Metadata
- `--input-metadata <filename>`: Specify the input metadata file for the first input file (default input.db)
- `--output-metadata <filename>`: Specify the output metadata file (default output.db)

#### Stacking Mode
- `-m, --mode <number>`: Specify the stacking mode to use (default is Auto)
  - -1 = Auto
  - 0 = Mean
  - 1 = Median
  - 2 = Smart mean
  - 3 = Smart neighbor
  - 4 = Neighbor
  - 5 = Local neighbor
  - 6 = Smart local neighbor
  - 7 = Medoid (akin to mode avg; aka "local")
- `--st, --smart-threshold <number>`: Range of value in 8-bit (0-128) used by smart modes (default 15)

#### Processing Options
- `-r, --reverse`: Reverse the field order to second/first (default first/second)
- `-t, --threads <number>`: Specify the number of concurrent threads (default is the number of logical CPUs)
- `--no-diffdod`: Do not use differential dropout detection on low source pixels
- `--no-map`: Disable mapping requirement
- `--passthrough`: Pass-through dropouts present on every source
- `--no-snr-weight`: Disable SNR-weighted bad-consensus override (default: enabled when VITS metrics are present)
- `--it, --integrity`: Check if frames contain skip or sample drop and discard bad source for specific frame

## Examples

```bash
# Basic two-source stack
ld-disc-stacker capture1.tbc capture2.tbc combined.tbc

# Use median mode
ld-disc-stacker -m 1 capture1.tbc capture2.tbc capture3.tbc combined.tbc

# Use smart mean with custom threshold
ld-disc-stacker -m 2 --st 20 capture1.tbc capture2.tbc combined.tbc

# With integrity checking
ld-disc-stacker --it capture1.tbc capture2.tbc capture3.tbc combined.tbc
```

## Input/Output

### Input Requirements
- **Same Disc**: All sources must be from the same physical disc
- **VBI Data**: Sources need ld-process-vbi run first
- **Same Standard**: PAL sources with PAL, NTSC with NTSC
- **Quality Metadata**: Requires SNR and dropout data

### Output
- **Best-Of TBC**: Combined output with best fields
- **Statistics**: Quality improvement metrics
- **Metadata**: Complete combined metadata

## Troubleshooting

### Alignment Issues
- Ensure all sources are from the same physical disc
- Check that ld-process-vbi was run on all sources
- Verify VBI frame numbers are present in metadata

### Quality Issues
- Use `--verbose` to see selection decisions
- Try different stacking modes (`-m` option)
- Check individual source quality with ld-analyse

### Performance Issues
- Reduce thread count if memory usage is high
- Process sources in smaller batches for large files
