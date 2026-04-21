// tools/ld-cinemap/vbiprobe.cpp
#include "vbiprobe.h"
#include "cinedisc.h"
#include "lddecodemetadata.h"
#include "vbidecoder.h"
#include "tbc/logging.h"

namespace vbiProbe {

ProbeResult probe(CineDisc& disc)
{
    ProbeResult result;

    auto& md      = disc.getMetaData();
    const int nFrames = md.getNumberOfFrames();

    result.frames.resize(static_cast<size_t>(nFrames));

    VbiDecoder vbiDecoder;
    int cavCount = 0;
    int clvCount = 0;

    for (int fi = 0; fi < nFrames; ++fi) {
        FrameVbi& fv = result.frames[static_cast<size_t>(fi)];

        // Mark padded frames; still decode their VBI for lead-in/out flags.
        fv.padded = disc.isPadded(fi);

        const int frameNumber = fi + 1; // 1-based
        const int f1 = md.getFirstFieldNumber (frameNumber);
        const int f2 = md.getSecondFieldNumber(frameNumber);

        if (f1 < 1 || f2 < 1) continue;

        const auto vbi1 = md.getFieldVbi(f1).vbiData;
        const auto vbi2 = md.getFieldVbi(f2).vbiData;
        const auto v    = vbiDecoder.decodeFrame(vbi1[0], vbi1[1], vbi1[2],
                                                 vbi2[0], vbi2[1], vbi2[2]);

        fv.picNo   = v.picNo;
        fv.leadIn  = v.leadIn;
        fv.leadOut = v.leadOut;
        fv.isClv   = (v.clvHr != -1 && v.clvMin != -1 &&
                      v.clvSec != -1 && v.clvPicNo != -1);

        if (v.picNo > 0)  cavCount++;
        if (fv.isClv)     clvCount++;
    }

    qInfo() << "vbiProbe::probe: scanned" << nFrames << "frame(s)"
            << "| CAV hits =" << cavCount
            << "| CLV hits =" << clvCount;

    if (cavCount == 0 && clvCount == 0) {
        qCritical() << "vbiProbe::probe: no CAV or CLV VBI detected in"
                    << nFrames << "frame(s) - cannot determine disc type."
                    << "Check that the source is a valid LaserDisc TBC.";
        result.isDiscCav = false;
    } else {
        // CAV only if strictly greater, matching DiscMap tie-break behaviour.
        result.isDiscCav = (cavCount > clvCount);
    }

    disc.setIsDiscCav(result.isDiscCav);
    return result;
}

} // namespace vbiProbe
