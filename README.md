# ld-decode Tools

This is the complete suite of tools for processing LaserDisc captures and TBC (Time Base Corrected) files. The ld-decode project provides professional-grade tools for digitizing, processing, and analyzing analog video sources with exceptional quality and accuracy.

### "Cine?"

This repo is the home of some experimental variations on the ld-decode tools, the chief of which is the addition of internal telecine pulldown detection and consolidation. 

Also of note: the world's first TBC trimmer, and a stacker able to use such trims.

The new stacker will align samples that have discontinuities and still properly incorporate them, no matter the start point or duration. The timemaster now obeys VBI frame numbers like the others, so a custom-trimmed timemaster can be used to target a patch in the middle of a set of full-length samples, producing a stack output of selects easily. A hash map is maintained of the VBI range so that each frame moves to the VBI numbered position, breaking with the requirements of linearity. 

We also have 3 new stacking modes, mode 5: local neighbor (SNR is highest here, punchy but soft), mode 6 : smart local neighbor (superior detail overall, good SNR), and medoid, a close analog to a mode average, is mode 7. Medoid picks the sample with the shortest total distance to all other samples for the current pixel - this is the "local" constraint on modes 5 and 6, removing local outliers from the neighbor process. Arguably the best approximation of the master for taller stacks.

We detect edits and telecine pulldown in a new tool, ld-cinemap, best run after updating vits on a stack output. This tool can work in Cine mode, where we expect just a few cadence breaks, or in TV mode, where we expect it per shot. Since telecine detection in TV mode is dependent on successful edit detection, overrides are available via whitelist/blacklist commands. The results are written to the tbc's metadata, now in .db sqlite form.

The new ld-chroma-decoder expands the options when --ntsc-phase-comp is enabled, including both coherently solved or optionally high frequency residually derived Y/IQ, with a post demod 3D option for the latter. Then I and Q pass through bandwidth-tailored FIRs before rotation. Before comb is where the Cadence Assembler will merge telecine's duplicate fields for improving SNR, and also diff the duplicates for error correction. The process can follow metadata written by ld-cinemap, or alternatively the user can force a telecine pattern in the cli with --set-cadence. The default sends out the same frames that came in, but is processed as 24p where ever possible, so that every field that can be is upgraded. For --export-24p mode, Assembler resyncs each segment by default to compensate for excess frames due to 29.97 edits offset from film frames boundaries. User can override to get every frame and go long with --emit-max24p. Alternatively, all telecine handling can be disabled with --no-PA (PA is for pulldown awareness).

The new 2D comb section is a complex election between 3 different combs: the interfield comb, Frame is favored for progressive footage (including telecine), while the intrafield combs, Field A and Field B, are favored for interlaced material (as detected in ld-cinemap, or otherwise via field differencing). The user can pick the Field vs Frame election or any of its constituent combs via --two-d-variant. 3D now has a 2D similarity curve, favoring next and previous as they near current, but vetoing them if they drift too far; this removes many checkerboard errors and sharpens the comb. With the rebase, the new 3D controls in 7.2 have been rigged to this mechanism.

## Tool Categories

### Core Processing Tools
- **ld-process-vbi** - Decode Vertical Blanking Interval data
- **ld-process-vits** - Process Vertical Interval Test Signals
- **ld-process-ac3** - Extract Dolby Digital AC3 audio tracks

### EFM Decoder Suite
*Replaces deprecated ld-process-efm with staged decoding and stacking capabilities*
- **efm-decoder-f2** - Convert EFM T-values to F2 sections
- **efm-decoder-d24** - Convert F2 sections to Data24 format
- **efm-decoder-audio** - Convert EFM Data24 sections to 16-bit stereo PCM audio
- **efm-decoder-data** - Convert EFM Data24 sections to ECMA-130 binary data
- **efm-stacker-f2** - Combine multiple F2 captures for improved quality

### Analysis and Quality Tools
- **ld-analyse** - GUI tool for TBC file analysis and visualization
- **ld-discmap** - TBC and VBI alignment and correction tool
- **ld-dropout-correct** - Advanced dropout detection and correction
- **ld-chroma-decoder** - Color decoder for TBC LaserDisc video to RGB/YUV conversion
- **ld-disc-stacker** - Combine multiple TBC captures for improved quality

### Export and Conversion Tools
- **ld-export-metadata** - Export TBC metadata to external formats
- **ld-lds-converter** - Convert between 10-bit and 16-bit LaserDisc sample formats
- **ld-json-converter** - Convert between old internal JSON and new internal SQLite metadata formats

### Utility Scripts
- **ld-compress** - Compress TBC files for storage (in scripts/)
- **filtermaker** - Create custom filtering profiles (in scripts/)
- **tbc-video-export-legacy** - Legacy TBC to video conversion (archived)

## Getting Started

1. **Capture Processing**: Start with `ld-decode` to convert raw RF captures to TBC format
2. **Quality Analysis**: Use `ld-analyse` to assess capture quality and identify issues
3. **Correction**: Apply `ld-dropout-correct` for dropout repair if needed
4. **Chroma Decoding**: Process composite sources with `ld-chroma-decoder`
5. **Export**: Convert to final formats using `tbc-video-export`

## Important Notes

- **SQLite Format**: All tools now use SQLite format for metadata storage instead of JSON
- **File Extensions**: TBC files use `.tbc` extension, metadata uses `.tbc.db` (SQLite format)
- **Dependencies**: Most tools require FFmpeg and other multimedia libraries
- **Performance**: Many tools support multi-threading for faster processing

> [!WARNING]  
> The SQLite metadata format is **internal to ld-decode tools only** and subject to change without notice. External tools and scripts should **not** access this database directly. Instead, use `ld-export-metadata` or similar tools to export metadata in stable, documented formats.

## Documentation

Each tool directory contains detailed README.md files with:
- Comprehensive usage instructions
- Complete option references
- Practical examples
- Input/output format specifications
- Troubleshooting guides

See individual tool directories for specific documentation.

