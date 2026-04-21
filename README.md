# ld-decode Tools

This is the complete suite of tools for processing LaserDisc captures and TBC (Time Base Corrected) files. The ld-decode project provides professional-grade tools for digitizing, processing, and analyzing analog video sources with exceptional quality and accuracy.

### "Cine?"

This repo is the home of some experimental variations on the ld-decode tools, the chief of which is the addition of internal pulldown detection and consolidation. Also of note: the world's first TBC trimmer, and a stacker able to use such trims (or other samples lacking the usual alignment) and still properly incorporate them, no matter the start point or duration. We also have 3 new stacking modes, which are a revelation themselves, particularly mode 6, smart local neighbor. We detect edits and pulldown in a new tool, ld-cinemap, best run after updating vits on a stack output. This tool can work in Cine mode, where we expect just a few cadence breaks, or in TV mode, where we expect it per shot. Since telecine detection in TV mode is dependent on successful edit detection, overrides are available via whitelist/blacklist commands. With metadata updated, files can then be sent through the new ld-chroma-decoder, where the CadenceAssembler will merge duplicate fields for improving SNR, and also diff the duplicates for error correction. Assembler resyncs each segment by default (user can override to get every frame and go long). User can export as 24p, or have the improved handling upgrade the existing fields for 29.97 fps output. The new 2D section is a complex election between 3 different combs, and the interfield comb Frame is favored for progressive footage (including telecine), while the intrafield combs Field A and Field B, are favored for interlaced material. The user can pick the Field vs Frame election or any of its constituent combs via --two-d-variant. 3D now has a 2D similarity curve, favoring next and previous as they near current, but vetoing them if they drift too far; this removes many checkerboard errors and sharpens the comb nicely. Setting --ntsc-phase-comp opens up a much more heavily featured "Locked path," including coherently solved Y, high freqency residual Y/IQ, and bandwidth tailored filters for I and Q. Residual video can be disabled or sent through a post-demod "3D" election. 

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

