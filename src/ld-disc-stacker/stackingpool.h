/************************************************************************

    stackingpool.h

    ld-disc-stacker - Disc stacking for ld-decode
    Copyright (C) 2020-2025 Simon Inns
    Copyright (C) 2026 Joseph Burns

    This file is part of ld-decode-tools.

    ld-disc-stacker is free software: you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

************************************************************************/

#ifndef STACKINGPOOL_H
#define STACKINGPOOL_H

#include <QObject>
#include <QAtomicInt>
#include <QElapsedTimer>
#include <QHash>
#include <QMutex>
#include <QThread>

#include "sourcevideo.h"
#include "lddecodemetadata.h"
#include "stacker.h"

class StackingPool : public QObject
{
    Q_OBJECT
public:
    explicit StackingPool(QString _outputFilename, QString _outputMetadataFilename,
                          qint32 _maxThreads,
                          QVector<LdDecodeMetaData *>& _ldDecodeMetaData,
                          QVector<SourceVideo *>& _sourceVideos,
                          qint32 _mode, qint32 _smartThreshold,
                          bool _reverse, bool _noDiffDod, bool _passThrough,
                          bool _integrityCheck, bool _verbose,
                          bool _useSnrWeight, qint32 _snrWeightThreshold,
                          QObject *parent = nullptr);

    bool process();

    // Member functions used by worker threads
    bool getInputFrame(qint32& frameNumber,
                       QVector<qint32>& firstFieldNumber,
                       QVector<SourceVideo::Data>& firstFieldVideoData,
                       QVector<LdDecodeMetaData::Field>& firstFieldMetadata,
                       QVector<qint32>& secondFieldNumber,
                       QVector<SourceVideo::Data>& secondFieldVideoData,
                       QVector<LdDecodeMetaData::Field>& secondFieldMetadata,
                       QVector<LdDecodeMetaData::VideoParameters>& videoParameters,
                       qint32& _mode, qint32& _smartThreshold,
                       bool& _reverse, bool& _noDiffDod, bool& _passThrough,
                       bool& _verbose,
                       QVector<qint32>& availableSourcesForFrame,
                       QVector<double>& sourceSnrWeights,
                       bool& _useSnrWeight, qint32& _snrWeightThreshold);

    bool setOutputFrame(qint32 frameNumber,
                        SourceVideo::Data firstTargetFieldData,
                        SourceVideo::Data secondTargetFieldData,
                        qint32 firstFieldSeqNo, qint32 secondFieldSeqNo,
                        DropOuts firstTargetFieldDropOuts,
                        DropOuts secondTargetFieldDropouts);

private:
    QString outputFilename;
    QString outputMetadataFilename;
    qint32 maxThreads;
    qint32 mode;
    qint32 smartThreshold;
    bool reverse;
    bool noDiffDod;
    bool passThrough;
    bool integrityCheck;
    bool verbose;
    bool useSnrWeight;
    qint32 snrWeightThreshold;

    QElapsedTimer totalTimer;
    qint32 skippedFrame;

    QAtomicInt abort;

    QMutex inputMutex;
    qint32 inputFrameNumber;
    qint32 lastFrameNumber;
    QVector<LdDecodeMetaData *>& ldDecodeMetaData;
    QVector<SourceVideo *>& sourceVideos;

    QMutex outputMutex;

    struct OutputFrame {
        SourceVideo::Data firstTargetFieldData;
        SourceVideo::Data secondTargetFieldData;
        qint32 firstFieldSeqNo;
        qint32 secondFieldSeqNo;
        DropOuts firstTargetFieldDropOuts;
        DropOuts secondTargetFieldDropOuts;
    };

    qint32 outputFrameNumber;
    QMap<qint32, OutputFrame> pendingOutputFrames;
    QFile targetVideo;

    QVector<bool>    sourceDiscTypeCav;
    QVector<qint32>  sourceMinimumVbiFrame;
    QVector<qint32>  sourceMaximumVbiFrame;

    // Direct VBI → sequential frame number lookup per source.
    // Replaces arithmetic conversion, which was fragile when sources started
    // at different points on the disc (e.g. trimmed vs full-length captures).
    QVector<QHash<qint32, qint32>> sourceVbiMap;

    // Reverse map for the timemaster only: sequential → VBI.
    QHash<qint32, qint32> timemasterSeqToVbi;

    bool setMinAndMaxVbiFrames();
    QVector<qint32> getAvailableSourcesForFrame(qint32 vbiFrameNumber);
    bool writeOutputField(const SourceVideo::Data& fieldData);
    void correctPhaseIDs();
    bool isIntegrityOk(const SourceVideo::Data& inputFields,
                       const LdDecodeMetaData::VideoParameters& videoParameters);
    template<int field>
    void replaceFieldMetaData(qint32 frameNumber);
    LdDecodeMetaData& correctMetaData();
};

#endif // STACKINGPOOL_H
