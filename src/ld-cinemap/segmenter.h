// tools/ld-cinemap/segmenter.h
#pragma once

class CineDisc;

namespace segmenter {

    // Marks phase-discontinuity edit boundaries in disc metadata.
    // Does NOT clear any existing boundaries — call clearEditBoundaries()
    // explicitly before this if a fresh run is needed.
    int segmentDisc(CineDisc& disc);

    // Clears all isEditBoundary flags from disc metadata, leaving manual edit
    // vetoes standing. This is the backing implementation for --clear-edits.
    // Never called implicitly by segmentDisc() or any other pass.
    void clearEditBoundaries(CineDisc& disc);

    // Clears all solver-owned flags from disc metadata:
    //   - isEditBoundary
    //   - cadenceId / cadenceIndexPresumed
    //   - pulldownRole
    // and additionally discards every manual edit veto (isEditVetoed).
    // This is the backing implementation for --clear-all-flags, and the only
    // path in the tool that destroys a user veto.
    // Never called implicitly by any detection pass.
    void clearAllFlags(CineDisc& disc);

} // namespace segmenter