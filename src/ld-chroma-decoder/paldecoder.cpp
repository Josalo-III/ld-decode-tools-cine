/******************************************************************************
 * paldecoder.cpp
 * ld-chroma-decoder — Colourisation filter for ld-decode
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2018-2019 Simon Inns
 * SPDX-FileCopyrightText: 2019-2021 Adam Sampson
 *
 * This file is part of ld-decode-tools.
 ******************************************************************************/

#include "paldecoder.h"

#include "decoderpool.h"

PalDecoder::PalDecoder(const PalColour::Configuration &palConfig)
{
    config.pal = palConfig;
}

bool PalDecoder::configure(const LdDecodeMetaData::VideoParameters &videoParameters) {
    // Ensure the source video is PAL
    if (videoParameters.system != PAL && videoParameters.system != PAL_M) {
        qCritical() << "This decoder is for PAL video sources only";
        return false;
    }

    config.videoParameters = videoParameters;

    return true;
}

qint32 PalDecoder::getLookBehind() const
{
    return config.pal.getLookBehind();
}

qint32 PalDecoder::getLookAhead() const
{
    return config.pal.getLookAhead();
}

QThread *PalDecoder::makeThread(QAtomicInt& abort, DecoderPool& decoderPool) {
    return new PalThread(abort, decoderPool, config);
}

PalThread::PalThread(QAtomicInt& _abort, DecoderPool& _decoderPool,
                     const PalDecoder::Configuration &_config, QObject *parent)
    : DecoderThread(_abort, _decoderPool, parent), config(_config)
{
    // Configure PALcolour
    palColour.updateConfiguration(config.videoParameters, config.pal);
}

void PalThread::decodeFrames(const QVector<SourceField> &inputFields, qint32 startIndex, qint32 endIndex,
                             QVector<ComponentFrame> &componentFrames)
{
    palColour.decodeFrames(inputFields, startIndex, endIndex, componentFrames);
}
