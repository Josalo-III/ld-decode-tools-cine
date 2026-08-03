// tools/ld-cinemap/segmenter.cpp
#include "segmenter.h"
#include "cinedisc.h"
#include "lddecodemetadata.h"
#include "tbc/logging.h"

namespace segmenter {

// Marks phase-discontinuity boundaries in disc metadata.
// Does NOT clear existing boundaries — call clearEditBoundaries()
// or clearAllFlags() explicitly before this if a fresh run is needed.
// PAL discs are not processed (phase structure is 8-phase; not yet supported).
int segmentDisc(CineDisc& disc)
{
    if (disc.isDiscPal()) return 0;

    auto& md = disc.getMetaData();
    const int totalFields = md.getNumberOfFields();

    int  boundaryCount = 0;
    int  prevPhase     = -1;
    bool prevValid     = false;

    for (int i = 1; i <= totalFields; ++i) {
        auto field = md.getField(i);

        const int  phase    = field.fieldPhaseID;
        const bool curValid = (phase >= 0) && !field.pad;

        if (!curValid) {
            // Invalid or padded field: reset phase tracking and clear solver flags.
            prevPhase            = -1;
            prevValid            = false;
            field.cinemap.clearEditBoundary();
            md.updateField(field, i);
            continue;
        }

        bool discontinuity = false;
        if (prevValid) {
            const int delta = phase - prevPhase;
            // Valid NTSC phase steps: +1 (normal advance) or -3 (wrap 4->0).
            // delta == 0 is tolerated for duplicate-phase fields.
            if (!(delta == 1 || delta == -3 || delta == 0)) {
                discontinuity = true;
            }
        }

        if (discontinuity || !prevValid) {
            // A vetoed field absorbs the discontinuity without becoming a
            // boundary, and is not counted as one.
            field.cinemap.assertEditBoundary();
            if (field.cinemap.isEditBoundary) boundaryCount++;
            md.updateField(field, i);
        }

        prevPhase = phase;
        prevValid = true;
    }

    qInfo() << "segmenter::segmentDisc: marked" << boundaryCount << "boundary(s).";
    return boundaryCount;
}

// Clears all isEditBoundary flags, leaving manual vetoes standing.
// Never called implicitly.
void clearEditBoundaries(CineDisc& disc)
{
    auto& md = disc.getMetaData();
    const int totalFields = md.getNumberOfFields();
    for (int i = 1; i <= totalFields; ++i) {
        auto f = md.getField(i);
        if (!f.cinemap.isEditBoundary) continue; // skip write if already clear
        f.cinemap.clearEditBoundary();
        md.updateField(f, i);
    }
    qInfo() << "segmenter::clearEditBoundaries: cleared all edit boundaries (vetoes kept).";
}

// Clears all solver-owned flags AND manual edit vetoes. Never called implicitly.
// Covers: isEditBoundary, isEditVetoed, cadenceId, cadenceIndexPresumed,
// pulldownRole. This is the only path that discards a user veto, which is why
// it is a separate, explicitly confirmed command from clearEditBoundaries().
void clearAllFlags(CineDisc& disc)
{
    auto& md = disc.getMetaData();
    const int totalFields = md.getNumberOfFields();
    int vetoesCleared = 0;
    for (int i = 1; i <= totalFields; ++i) {
        auto f = md.getField(i);
        if (f.cinemap.isEditVetoed) vetoesCleared++;
        f.cinemap.clearEditState();
        f.cinemap.cadenceId            = -1;
        f.cinemap.cadenceIndexPresumed = false;
        f.cinemap.pulldownRole         = QString();
        md.updateField(f, i);
    }
    qInfo() << "segmenter::clearAllFlags: cleared all solver flags and"
            << vetoesCleared << "manual edit veto(s).";
}

} // namespace segmenter
