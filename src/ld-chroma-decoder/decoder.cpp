/******************************************************************************
 * decoder.cpp
 * ld-chroma-decoder — Colourisation filter for ld-decode
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2019-2021 Adam Sampson
 * SPDX-FileCopyrightText: 2022-2026 Joseph Burns
 *
 * This file is part of ld-decode-tools.
 ******************************************************************************/

#include "decoder.h"

#include "decoderpool.h"

qint32 Decoder::getLookBehind() const
{
    return 0;
}

qint32 Decoder::getLookAhead() const
{
    return 0;
}

DecoderThread::DecoderThread(QAtomicInt& _abort, DecoderPool& _decoderPool, QObject *parent)
    : QThread(parent), abort(_abort), decoderPool(_decoderPool), outputWriter(_decoderPool.getOutputWriter())
{
}

void DecoderThread::run()
{
    QVector<SourceField> inputFields;
    QVector<ComponentFrame> componentFrames;
    QVector<OutputFrame> outputFrames;

    while (!abort) {
        qint32 startFrameNumber, startIndex, endIndex;
        if (!decoderPool.getInputFrames(startFrameNumber, inputFields, startIndex, endIndex)) {
            break;
        }

        const qint32 numFrames = (endIndex - startIndex) / 2;
        componentFrames.resize(numFrames);
        outputFrames.resize(numFrames);

        decodeFrames(inputFields, startIndex, endIndex, componentFrames);

        for (qint32 i = 0; i < numFrames; i++) {
            outputWriter.convert(componentFrames[i], outputFrames[i]);
        }

        if (!decoderPool.getCadenceConfig().noPA &&
            !decoderPool.getCadenceConfig().export24p) {
            if (!decoderPool.putOutputFrames(startFrameNumber, componentFrames, outputFrames)) {
                abort = true;
                break;
            }
        } else {
            if (!decoderPool.putOutputFrames(startFrameNumber, outputFrames)) {
                abort = true;
                break;
            }
        }
    }
}
