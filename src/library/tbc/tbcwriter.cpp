/******************************************************************************
 * tbcwriter.cpp
 * ld-decode-tools TBC library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 Joseph Burns
 *
 * This file is part of ld-decode-tools.
 ******************************************************************************/

#include "tbcwriter.h"
#include "sourcevideo.h"
#include "sourceaudio.h"
#include <QDebug>
#include <QFile>

bool TbcStreamWriter::write(const QVector<WriteFrame> &frames,
                            LdDecodeMetaData *sourceMetadata,
                            const QFileInfo &inputFile,
                            const QFileInfo &outputFile,
                            const Config &config)
{
    // 1. Write metadata (SQLite)
    if (config.writeMetadata && sourceMetadata
            && !outputFile.filePath().isEmpty()
            && outputFile.filePath() != "-") {
        if (!sourceMetadata->write(outputFile.filePath() + ".tbc.db")) {
            qWarning() << "Failed to write metadata to" << outputFile.filePath() + ".tbc.db";
        } else {
            qInfo() << "Metadata written to" << outputFile.filePath() + ".tbc.db";
        }
    }

    if (!config.writeVideo && !config.writeAudio) {
        return true;
    }

    if (outputFile.filePath().isEmpty()) {
        qCritical() << "No output filename specified for video/audio writing.";
        return false;
    }

    if (config.writeVideo && (config.fieldWidth <= 0 || config.fieldHeight <= 0)) {
        qCritical() << "TbcStreamWriter: Invalid field dimensions provided.";
        return false;
    }

    const int fieldSize = config.fieldWidth * config.fieldHeight;

    // 2. Open source video
    SourceVideo sourceVideo;
    if (config.writeVideo) {
        if (!sourceVideo.open(inputFile.filePath(), fieldSize)) {
            qCritical() << "Could not open source video:" << inputFile.filePath();
            return false;
        }
    }

    // 3. Open source audio
    SourceAudio sourceAudio;
    bool processAudio = config.writeAudio;
    if (processAudio) {
        if (!sourceAudio.open(inputFile)) {
            qWarning() << "Could not open source audio - disabling audio output.";
            processAudio = false;
        }
    }

    // 4. Open target files
    QFile targetVideo;
    QFile targetAudio;

    const bool isPipe = (outputFile.filePath() == "-");

    if (!isPipe) {
        targetVideo.setFileName(outputFile.filePath());
        targetAudio.setFileName(outputFile.filePath() + ".pcm");
    }

    if (config.writeVideo) {
        if (isPipe) {
            if (!targetVideo.open(stdout, QIODevice::WriteOnly)) {
                qCritical() << "Could not open stdout for video";
                return false;
            }
        } else {
            if (!targetVideo.open(QIODevice::WriteOnly)) {
                qCritical() << "Could not open output video:" << outputFile.filePath();
                return false;
            }
        }
    }

    if (processAudio) {
        if (isPipe) {
            qWarning() << "Piped output does not support audio writing.";
            processAudio = false;
        } else {
            if (!targetAudio.open(QIODevice::WriteOnly)) {
                qWarning() << "Could not open output audio:" << targetAudio.fileName();
                processAudio = false;
            }
        }
    }

    // 5. Processing loop
    qInfo() << "Writing output stream...";
    const qint32 frameCount     = frames.size();
    const qint32 notifyInterval = std::max(1, frameCount / 20);

    for (int i = 0; i < frameCount; ++i) {
        const WriteFrame &frame = frames[i];

        // Video
        if (config.writeVideo) {
            SourceVideo::Data d1, d2;

            if (frame.isPadded) {
                d1.resize(fieldSize * 2); d1.fill(0);
                d2.resize(fieldSize * 2); d2.fill(0);
            } else {
                d1 = sourceVideo.getVideoField(frame.firstFieldSourceIndex);
                d2 = sourceVideo.getVideoField(frame.secondFieldSourceIndex);
            }

            if (targetVideo.write(reinterpret_cast<const char *>(d1.constData()), d1.size() * 2) != d1.size() * 2) {
                qCritical() << "Write error on video frame" << i;
                return false;
            }
            if (targetVideo.write(reinterpret_cast<const char *>(d2.constData()), d2.size() * 2) != d2.size() * 2) {
                qCritical() << "Write error on video frame" << i;
                return false;
            }
        }

        // Audio
        if (processAudio) {
            auto writeFieldAudio = [&](int fieldIdx) {
                SourceAudio::Data audioData;

                if (!frame.isPadded && fieldIdx > 0 && sourceMetadata) {
                    // getFieldPcmAudioStart/Length are not yet const
                    const int start = sourceMetadata->getFieldPcmAudioStart(fieldIdx);
                    const int len   = sourceMetadata->getFieldPcmAudioLength(fieldIdx);
                    if (start >= 0 && len > 0) {
                        audioData = sourceAudio.getAudioData(start, len);
                    }
                }

                if (audioData.isEmpty()) {
                    // Fallback silence
                    int samples = 736;
                    if (sourceMetadata
                            && sourceMetadata->getVideoParameters().system == PAL)
                        samples = 882;
                    audioData.resize(samples * 2);
                    audioData.fill(0);
                }

                targetAudio.write(reinterpret_cast<const char *>(audioData.constData()),
                                  audioData.size() * sizeof(qint16));
            };

            writeFieldAudio(frame.firstFieldSourceIndex);
            writeFieldAudio(frame.secondFieldSourceIndex);
        }

        if (i % notifyInterval == 0) {
            qInfo() << "Writing frame" << i << "/" << frameCount;
        }
    }

    qInfo() << "Write complete.";
    return true;
}
