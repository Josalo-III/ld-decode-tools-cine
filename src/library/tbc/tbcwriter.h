/******************************************************************************
 * tbcwriter.h
 * ld-decode-tools TBC library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 Joseph Burns
 *
 * This file is part of ld-decode-tools.
 *
 * Shared TBC stream writer for standalone ld-decode tools.
 ******************************************************************************/

#ifndef TBCWRITER_H
#define TBCWRITER_H

#include <QFileInfo>
#include <QVector>
#include "lddecodemetadata.h"

class TbcStreamWriter
{
public:
    struct WriteFrame {
        int firstFieldSourceIndex;  // 1-based source field index
        int secondFieldSourceIndex; // 1-based source field index
        bool isPadded;              // If true, write black frames
    };

    struct Config {
        bool writeVideo    = true;
        bool writeAudio    = true;
        bool writeMetadata = true;
        int fieldWidth  = 0; // Required if writing video
        int fieldHeight = 0; // Required if writing video
    };

    // Write A/V streams based on a mapped frame list.
    //
    // sourceMetadata is a non-const pointer because getFieldPcmAudioStart and
    // getFieldPcmAudioLength are not yet const. getVideoParameters() is const.
    static bool write(const QVector<WriteFrame> &frames,
                      LdDecodeMetaData *sourceMetadata,
                      const QFileInfo &inputFile,
                      const QFileInfo &outputFile,
                      const Config &config);
};

#endif // TBCWRITER_H
