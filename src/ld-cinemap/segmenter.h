// tools/ld-cinemap/segmenter.h
#pragma once

class CineDisc;

namespace segmenter {

    // Marks phase-discontinuity edit boundaries in disc metadata.
    // Does NOT clear any existing boundaries — call clearEditBoundaries()
    // explicitly before this if a fresh run is needed.
    int segmentDisc(CineDisc& disc);

    // Clears all isEditBoundary flags from disc metadata.
    // Never called implicitly by segmentDisc() or any other pass.
    void clearEditBoundaries(CineDisc& disc);

    // Clears all solver-owned flags from disc metadata:
    //   - isEditBoundary
    //   - cadenceId / cadenceConfidence / cadenceIndexPresumed
    //   - pulldownRole
    // This is the backing implementation for --clear-all-flags.
    // Never called implicitly by any detection pass.
    void clearAllFlags(CineDisc& disc);

} // namespace segmenter