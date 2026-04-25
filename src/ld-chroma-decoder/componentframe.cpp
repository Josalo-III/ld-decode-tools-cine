/******************************************************************************
 * componentframe.cpp
 * ld-chroma-decoder — Colourisation filter for ld-decode
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2021 Adam Sampson
 *
 * This file is part of ld-decode-tools.
 ******************************************************************************/

#include "componentframe.h"

ComponentFrame::ComponentFrame()
    : width(-1), height(-1)
{
}

void ComponentFrame::init(const LdDecodeMetaData::VideoParameters &videoParameters, bool mono)
{
    width = videoParameters.fieldWidth;
    height = (videoParameters.fieldHeight * 2) - 1;

    const qint32 size = width * height;

    yData.resize(size);
    yData.fill(0.0);

    if(!mono) {
        uData.resize(size);
        uData.fill(0.0);

        vData.resize(size);
        vData.fill(0.0);
    } else {
        // Clear and deallocate U/V if they're not used.
        uData.clear();
        uData.squeeze();

        vData.clear();
        vData.squeeze();
    }
}
