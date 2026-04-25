/******************************************************************************
 * ntscdecoder.cpp
 * ld-chroma-decoder — Colourisation filter for ld-decode
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2018 Chad Page
 * SPDX-FileCopyrightText: 2018-2019 Simon Inns
 * SPDX-FileCopyrightText: 2019-2021 Adam Sampson
 *
 * This file is part of ld-decode-tools.
 ******************************************************************************/

#include "ntscdecoder.h"

#include "decoderpool.h"

NtscDecoder::NtscDecoder(const Comb::Configuration &combConfig)
{
    config.combConfig = combConfig;
}

bool NtscDecoder::configure(const LdDecodeMetaData::VideoParameters &videoParameters) {
    // Ensure the source video is NTSC
    if (videoParameters.system != NTSC) {
        qCritical() << "This decoder is for NTSC video sources only";
        return false;
    }

    config.videoParameters = videoParameters;

    return true;
}

qint32 NtscDecoder::getLookBehind() const
{
    return config.combConfig.getLookBehind();
}

qint32 NtscDecoder::getLookAhead() const
{
    return config.combConfig.getLookAhead();
}

QThread *NtscDecoder::makeThread(QAtomicInt& abort, DecoderPool& decoderPool)
{
    return new NtscThread(abort, decoderPool, config);
}

NtscThread::NtscThread(QAtomicInt& _abort, DecoderPool &_decoderPool,
                       const NtscDecoder::Configuration &_config, QObject *parent)
    : DecoderThread(_abort, _decoderPool, parent), config(_config)
{
    // Configure NTSC decoder
    comb.updateConfiguration(config.videoParameters, config.combConfig);
}

void NtscThread::decodeFrames(const QVector<SourceField> &inputFields, qint32 startIndex, qint32 endIndex,
                              QVector<ComponentFrame> &componentFrames)
{
    // Decode fields to frames
    comb.decodeFrames(inputFields, startIndex, endIndex, componentFrames);
}
