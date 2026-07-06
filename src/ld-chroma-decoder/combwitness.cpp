/******************************************************************************
 * combwitness.cpp
 * ld-chroma-decoder — carrier-retraction diagnostics
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * SPDX-FileCopyrightText: 2025-2026 Joseph Burns
 ******************************************************************************/

#include "comb.h"

void Comb::FrameBuffer::outputDiagnosticFrame()
{
    if (!componentFrame) {
        qWarning("Diagnostic output requested without a destination frame");
        return;
    }

    const int first = videoParameters.firstActiveFrameLine;
    const int last = videoParameters.lastActiveFrameLine;
    const int left = videoParameters.activeVideoStart;
    const int width = videoParameters.activeVideoEnd - left;

    if (width <= 0 || demodWidth != width || demodLines < last) {
        qWarning("Diagnostic output geometry is invalid");
        return;
    }

    const double black = videoParameters.black16bIre;
    const double white = videoParameters.white16bIre;
    const double mid = black + 0.5 * (white - black);

    switch (configuration.diagnosticOutput) {
    case Configuration::DiagnosticOutput::None:
        qWarning("Diagnostic output requested with no diagnostic mode selected");
        return;

    case Configuration::DiagnosticOutput::CarrierFit:
        if (!carrierRetractionModelValid) {
            qWarning("Carrier-fit output requested without a valid carrier model");
            return;
        }
        for (int line = first; line < last; ++line) {
            const float *source = carrierFit_flat.data()
                                + static_cast<size_t>(line) * demodWidth;
            double *destination = componentFrame->y(line) + left;
            for (int x = 0; x < width; ++x)
                destination[x] = mid + static_cast<double>(source[x]);
        }
        return;

    case Configuration::DiagnosticOutput::CarrierRetracted:
        if (!carrierRetractedValid) {
            qWarning("Carrier-retracted output requested without a valid retracted view");
            return;
        }
        for (int line = first; line < last; ++line) {
            const float *source = carrierRetracted_line(line);
            double *destination = componentFrame->y(line) + left;
            for (int x = 0; x < width; ++x)
                destination[x] = source[x];
        }
        return;
    }
}
