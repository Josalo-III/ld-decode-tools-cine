// tools/ld-cinemap/visualedits.h
#pragma once

class CineDisc;

namespace visualEdits {
    int analyzeVisualEdits(CineDisc& disc,
                           double threshold,
                           double strongFactor,
                           double peakFactor);
}
