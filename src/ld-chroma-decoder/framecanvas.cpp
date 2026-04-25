/******************************************************************************
 * framecanvas.cpp
 * ld-chroma-decoder — Colourisation filter for ld-decode
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2019-2021 Adam Sampson
 *
 * This file is part of ld-decode-tools.
 ******************************************************************************/

#include "framecanvas.h"

FrameCanvas::FrameCanvas(ComponentFrame &_componentFrame, const LdDecodeMetaData::VideoParameters &_videoParameters)
    : yData(_componentFrame.y(0)), uData(_componentFrame.u(0)), vData(_componentFrame.v(0)),
      width(_componentFrame.getWidth()), height(_componentFrame.getHeight()),
      ireRange(_videoParameters.white16bIre - _videoParameters.black16bIre), blackIre(_videoParameters.black16bIre),
      videoParameters(_videoParameters)
{
}

qint32 FrameCanvas::top()
{
    return videoParameters.firstActiveFrameLine;
}

qint32 FrameCanvas::bottom()
{
    return videoParameters.lastActiveFrameLine;
}

qint32 FrameCanvas::left()
{
    return videoParameters.activeVideoStart;
}

qint32 FrameCanvas::right()
{
    return videoParameters.activeVideoEnd;
}

FrameCanvas::Colour FrameCanvas::rgb(quint16 r, quint16 g, quint16 b)
{
    // Scale R'G'B' to match the IRE range
    const double sr = (r / 65535.0) * ireRange;
    const double sg = (g / 65535.0) * ireRange;
    const double sb = (b / 65535.0) * ireRange;

    // Convert to Y'UV form [Poynton eq 28.5 p337]
    return Colour {
        ((sr * 0.299)    + (sg * 0.587)     + (sb * 0.114))    + blackIre,
        (sr * -0.147141) + (sg * -0.288869) + (sb * 0.436010),
        (sr * 0.614975)  + (sg * -0.514965) + (sb * -0.100010)
    };
}

FrameCanvas::Colour FrameCanvas::grey(quint16 value)
{
    // Scale Y to match the IRE range
    return Colour {((value / 65535.0) * ireRange) + blackIre, 0.0, 0.0};
}

void FrameCanvas::drawPoint(qint32 x, qint32 y, const Colour& colour)
{
    if (x < 0 || x >= width || y < 0 || y >= height) {
        // Outside the frame
        return;
    }

    const qint32 offset = (y * width) + x;
    yData[offset] = colour.y;
    uData[offset] = colour.u;
    vData[offset] = colour.v;
}

void FrameCanvas::drawRectangle(qint32 xStart, qint32 yStart, qint32 w, qint32 h, const Colour& colour)
{
    for (qint32 y = yStart; y < yStart + h; y++) {
        drawPoint(xStart, y, colour);
        drawPoint(xStart + w - 1, y, colour);
    }
    for (qint32 x = xStart + 1; x < xStart + w - 1; x++) {
        drawPoint(x, yStart, colour);
        drawPoint(x, yStart + h - 1, colour);
    }
}

void FrameCanvas::fillRectangle(qint32 xStart, qint32 yStart, qint32 w, qint32 h, const Colour& colour)
{
    for (qint32 y = yStart; y < yStart + h; y++) {
        for (qint32 x = xStart; x < xStart + w; x++) {
            drawPoint(x, y, colour);
        }
    }
}
