/******************************************************************************
 * decoderpool.cpp
 * ld-chroma-decoder — Colourisation filter for ld-decode
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2018-2019 Simon Inns
 * SPDX-FileCopyrightText: 2021 Adam Sampson
 * SPDX-FileCopyrightText: 2021 Phillip Blucas
 * SPDX-FileCopyrightText: 2025-2026 Joseph Burns
 *
 * This file is part of ld-decode-tools.
 ******************************************************************************/

#include "decoderpool.h"

#include <algorithm>
#include <unistd.h>

DecoderPool::DecoderPool(Decoder &_decoder, QString _inputFileName,
                         LdDecodeMetaData &_ldDecodeMetaData,
                         OutputWriter::Configuration &_outputConfig,
                         CadenceAssembler::Configuration &_cadenceConfig,
                         QString _outputFileName,
                         qint32 _startFrame, qint32 _length, qint32 _maxThreads)
    : decoder(_decoder), inputFileName(_inputFileName),
      outputConfig(_outputConfig), cadenceConfig(_cadenceConfig), outputFileName(_outputFileName),
      startFrame(_startFrame), length(_length), maxThreads(_maxThreads),
      abort(false), ldDecodeMetaData(_ldDecodeMetaData)
{
}

Decoder &DecoderPool::getDecoder() { return decoder; }

SourceField DecoderPool::createBlackField(bool isFirst, int seqNo) const
{
    SourceField sf;
    int size = ldDecodeMetaData.getVideoParameters().fieldWidth *
               ldDecodeMetaData.getVideoParameters().fieldHeight;
    sf.data.resize(size * sizeof(quint16));
    sf.data.fill(0);
    sf.field.isFirstField = isFirst;
    sf.field.seqNo = seqNo;
    sf.field.cinemap.isEditBoundary = false; // ensure padding never blocks passthrough
    sf.field.cinemap.cadenceId = -1;         // neutral cadence for padding
    sf.allowProgressiveFrameRegime = false;
    return sf;
}

namespace {

    static inline int cadenceRoleIndex(int cid)
    {
        if (!cadenceKnown(cid)) return -1;
        const int idx = cadenceIndex(cid);
        if (idx == 0 || idx == 7) return 0;           // Def
        if (idx == 1 || idx == 6 || idx == 8) return 1; // Comp
        if (idx == 2 || idx == 5) return 2;           // Spare
        return -1;
    }
    
} // namespace

bool DecoderPool::process()
{
    LdDecodeMetaData::VideoParameters videoParameters = ldDecodeMetaData.getVideoParameters();

    outputWriter.updateConfiguration(videoParameters, outputConfig);
    outputWriter.printOutputInfo();
    
    effectiveVideoParameters = videoParameters;

    if (!decoder.configure(videoParameters)) {
        return false;
    }

    cadenceAssembler = std::make_unique<CadenceAssembler>(
        videoParameters,
        cadenceConfig,
        [this](qint32 seqNo) {
            // Called under inputMutex (from pumpAssembler inside getInputFrames).
            if (!cadenceConfig.export24p && !cadenceConfig.noPA)
                enqueueBaselinePassthrough(seqNo);
        }
    );
    
    decoderLookBehind = decoder.getLookBehind();
    decoderLookAhead  = decoder.getLookAhead();

    if (!sourceVideo.open(inputFileName, videoParameters.fieldWidth * videoParameters.fieldHeight)) {
        qInfo() << "Unable to open ld-decode video file";
        return false;
    }

    if (startFrame == -1) startFrame = 1;

    if (startFrame > ldDecodeMetaData.getNumberOfFrames()) {
        qInfo() << "Specified start frame is out of bounds, only" << ldDecodeMetaData.getNumberOfFrames() << "frames available";
        return false;
    }

    if (length == -1) {
        length = ldDecodeMetaData.getNumberOfFrames() - (startFrame - 1);
    } else {
        if (length + (startFrame - 1) > ldDecodeMetaData.getNumberOfFrames()) {
            qInfo() << "Specified length of" << length << "exceeds the number of available frames, setting to"
                    << ldDecodeMetaData.getNumberOfFrames() - (startFrame - 1);
            length = ldDecodeMetaData.getNumberOfFrames() - (startFrame - 1);
        }
    }

    if (outputFileName == "-") {
        if (!targetVideo.open(STDOUT_FILENO, QIODevice::WriteOnly)) {
            qCritical() << "Could not open stdout for output";
            sourceVideo.close();
            return false;
        }
        qInfo() << "Writing output to stdout";
    } else {
        targetVideo.setFileName(outputFileName);
        if (!targetVideo.open(QIODevice::WriteOnly)) {
            qCritical() << "Could not open" << outputFileName << "for output";
            sourceVideo.close();
            return false;
        }
    }

    const QByteArray streamHeader = outputWriter.getStreamHeader();
    if (!streamHeader.isEmpty() && targetVideo.write(streamHeader) == -1) {
        qCritical() << "Writing to the output video file failed";
        return false;
    }

    qInfo() << "Using" << maxThreads << "threads";
    qInfo() << "Processing from start frame #" << startFrame << "with a length of" << length << "frames";

    inputFrameNumber  = startFrame;
    outputFrameNumber = startFrame;
    servedFrameNumber = startFrame;
    lastFrameNumber   = length + (startFrame - 1);

    workItems.clear();
    baselineFramesQueued.clear();
    resolvedOutputFrames.clear();
    upgradedTbcFrames.clear();
    outstandingUpgradeSeqNos.clear();
    nextOutputKey24p = startFrame;
    writeCursor24p   = startFrame;

    paddingHistory.clear();
    paddingHistory.push_back(createBlackField(true, -4));
    paddingHistory.push_back(createBlackField(false, -3));
    paddingHistory.push_back(createBlackField(true, -2));
    paddingHistory.push_back(createBlackField(false, -1));

    
    scheduled24p.clear();
    buildingSeg24p.clear();
    initialFieldSeq = -1;
    framesScheduled24p = 0;
    lastCadenceIndex24p = -1; 
    
    totalTimer.start();

    QList<QThread *> threads;
    threads.resize(maxThreads);
    for (qint32 i = 0; i < maxThreads; i++) {
        threads[i] = decoder.makeThread(abort, *this);
        threads[i]->start(QThread::LowPriority);
    }

    for (qint32 i = 0; i < maxThreads; i++) {
        threads[i]->wait();
        delete threads[i];
    }

    if (abort) {
        sourceVideo.close();
        targetVideo.close();
        return false;
    }
    if (!cadenceConfig.noPA && !cadenceConfig.export24p) {
        if (outputFrameNumber <= lastFrameNumber) {
            const qint32 s1 = ldDecodeMetaData.getFirstFieldNumber(outputFrameNumber);
            const qint32 s2 = ldDecodeMetaData.getSecondFieldNumber(outputFrameNumber);
    
            qCritical() << "29.97 registry stalled at frame"
                        << outputFrameNumber
                        << "waiting for seqNos"
                        << s1 << "and" << s2;
    
            auto check = [&](qint32 s){
                bool haveBase = baselineFieldsBySeq.contains(s);
                bool haveUp   = upgradedFieldsBySeq.contains(s);
                qCritical() << "seq" << s
                            << "baseline:" << haveBase
                            << "upgrade:"  << haveUp
                            << "upgrade outstanding:"
                            << outstandingUpgradeSeqNos.contains(s);
            };
    
            check(s1);
            check(s2);
    
            abort = true;
        }
    }

    double totalSecs = (static_cast<double>(totalTimer.elapsed()) / 1000.0);
    qInfo() << "Processing complete -" << length << "frames in" << totalSecs << "seconds (" <<
               length / totalSecs << "FPS )";

    sourceVideo.close();
    targetVideo.close();
    return true;
}

void DecoderPool::registerFieldUpgrade(qint32 fieldSequenceNumber,
                                       const QVector<quint16> &upgradedField)
{
    QMutexLocker locker(&metaDataMutex);
    upgradedFieldsBySeq[fieldSequenceNumber] = { upgradedField, true };
}

// Split a fully converted OutputFrame into two field payloads (even/odd lines)
void DecoderPool::splitOutputFrameToFields(const OutputFrame& frame,
                                            QVector<quint16>& field1,
                                            QVector<quint16>& field2) const
{
    const qint32 width = outputWriter.getOutputWidth();
    const qint32 height = outputWriter.getOutputHeight();
    const qint32 pixelFormat = outputWriter.getPixelFormat();

    const qint32 field1Lines = (height + 1) / 2;
    const qint32 field2Lines = height / 2;

    if (pixelFormat == OutputWriter::YUV444P16) {
        // Planar: three separate planes of width * height each
        const qint32 planeSize = width * height;
        const qint32 f1PlaneSize = width * field1Lines;
        const qint32 f2PlaneSize = width * field2Lines;

        field1.resize(f1PlaneSize * 3);
        field2.resize(f2PlaneSize * 3);

        for (int plane = 0; plane < 3; ++plane) {
            const quint16* src = frame.constData() + plane * planeSize;
            quint16* dst1 = field1.data() + plane * f1PlaneSize;
            quint16* dst2 = field2.data() + plane * f2PlaneSize;

            for (qint32 line = 0; line < height; ++line) {
                if ((line & 1) == 0)
                    memcpy(dst1 + (line / 2) * width, src + line * width, width * sizeof(quint16));
                else
                    memcpy(dst2 + (line / 2) * width, src + line * width, width * sizeof(quint16));
            }
        }
    } else {
        // Packed (RGB48 or GRAY16)
        qint32 samplesPerPixel = (pixelFormat == OutputWriter::RGB48) ? 3 : 1;
        const qint32 lineStride = width * samplesPerPixel;

        field1.resize(field1Lines * lineStride);
        field2.resize(field2Lines * lineStride);

        for (qint32 line = 0; line < height; ++line) {
            const quint16* src = frame.constData() + line * lineStride;
            if ((line & 1) == 0)
                memcpy(field1.data() + (line / 2) * lineStride, src, lineStride * sizeof(quint16));
            else
                memcpy(field2.data() + (line / 2) * lineStride, src, lineStride * sizeof(quint16));
        }
    }
}

OutputFrame DecoderPool::interleaveFieldPayloads(const QVector<quint16>& field1,
                                                  const QVector<quint16>& field2) const
{
    const qint32 width = outputWriter.getOutputWidth();
    const qint32 height = outputWriter.getOutputHeight();
    const qint32 pixelFormat = outputWriter.getPixelFormat();

    const qint32 field1Lines = (height + 1) / 2;
    const qint32 field2Lines = height / 2;

    if (pixelFormat == OutputWriter::YUV444P16) {
        const qint32 planeSize = width * height;
        const qint32 f1PlaneSize = width * field1Lines;
        const qint32 f2PlaneSize = width * field2Lines;

        OutputFrame frame;
        frame.resize(planeSize * 3);

        for (int plane = 0; plane < 3; ++plane) {
            quint16* dst = frame.data() + plane * planeSize;
            const quint16* src1 = field1.constData() + plane * f1PlaneSize;
            const quint16* src2 = field2.constData() + plane * f2PlaneSize;

            for (qint32 line = 0; line < height; ++line) {
                if ((line & 1) == 0)
                    memcpy(dst + line * width, src1 + (line / 2) * width, width * sizeof(quint16));
                else
                    memcpy(dst + line * width, src2 + (line / 2) * width, width * sizeof(quint16));
            }
        }

        return frame;
    } else {
        qint32 samplesPerPixel = (pixelFormat == OutputWriter::RGB48) ? 3 : 1;
        const qint32 lineStride = width * samplesPerPixel;

        OutputFrame frame;
        frame.resize(height * lineStride);

        for (qint32 line = 0; line < height; ++line) {
            quint16* dst = frame.data() + line * lineStride;
            if ((line & 1) == 0)
                memcpy(dst, field1.constData() + (line / 2) * lineStride, lineStride * sizeof(quint16));
            else
                memcpy(dst, field2.constData() + (line / 2) * lineStride, lineStride * sizeof(quint16));
        }

        return frame;
    }
}

void DecoderPool::submitBaselineField(qint32 seqNo, const QVector<quint16>& data)
{
    if (seqNo < 0 || data.isEmpty()) return;
    baselineFieldsBySeq[seqNo] = { data, true };
}

void DecoderPool::submitUpgradedField(qint32 seqNo, const QVector<quint16>& data)
{
    if (seqNo < 0 || data.isEmpty()) return;
    upgradedFieldsBySeq[seqNo] = { data, true };
}

qint32 DecoderPool::frameNumberForSeq(qint32 seqNo) const
{
    const int nFrames = ldDecodeMetaData.getNumberOfFrames();
    for (int fn = 1; fn <= nFrames; ++fn) {
        if (ldDecodeMetaData.getFirstFieldNumber(fn) == seqNo ||
            ldDecodeMetaData.getSecondFieldNumber(fn) == seqNo)
            return fn;
    }
    return -1;
}

void DecoderPool::enqueueBaselinePassthrough(qint32 seqNo)
{
    // Called under outputMutex (from callback) or inputMutex (from getInputFrames).
    // Both are held by the caller — do not re-acquire.

    const qint32 frameNumber = frameNumberForSeq(seqNo);
    if (frameNumber < 1) return;
    if (baselineFramesQueued.contains(frameNumber)) return;
    baselineFramesQueued.insert(frameNumber);

    const qint32 seq1 = ldDecodeMetaData.getFirstFieldNumber(frameNumber);
    const qint32 seq2 = ldDecodeMetaData.getSecondFieldNumber(frameNumber);
    if (seq1 < 1 || seq2 < 1) return;

    // Load the raw fields from TBC.
    QVector<SourceField> rawVec;
    qint32 dummyStart = 0, dummyEnd = 0;
    SourceField::loadFields(sourceVideo, ldDecodeMetaData,
                            frameNumber, 1,
                            0, 0,
                            rawVec, dummyStart, dummyEnd);
    if (rawVec.size() < 2) return;

    CadenceAssembler::WorkItem wi;
    wi.kind = CadenceAssembler::WorkItem::Kind::PassthroughFrame;
    wi.expansion = CadenceAssembler::WorkItem::Expansion::None;
    wi.fieldsSwapped = false;
    wi.invertedFieldOrder = false;
    wi.filmLabel = '?';
    wi.f1 = std::move(rawVec[0]);
    wi.f2 = std::move(rawVec[1]);
    wi.f1.allowProgressiveFrameRegime = false;
    wi.f2.allowProgressiveFrameRegime = false;

    // Insert in disc order: find first workItem whose min seqNo exceeds ours.
    const qint32 minSeq = std::min(seq1, seq2);
    auto it = workItems.begin();
    while (it != workItems.end()) {
        const qint32 itMin = std::min(it->f1.field.seqNo, it->f2.field.seqNo);
        if (itMin > minSeq) break;
        ++it;
    }
    workItems.insert(it, std::move(wi));
}

bool DecoderPool::tryEmitNextOriginalPair()
{
    const qint32 seq1 = ldDecodeMetaData.getFirstFieldNumber(outputFrameNumber);
    const qint32 seq2 = ldDecodeMetaData.getSecondFieldNumber(outputFrameNumber);

    OutputFrame finalFrame;
    if (!assembleResolvedPairToFrame(seq1, seq2, finalFrame)) {
        return false;
    }

    const QByteArray frameHeader = outputWriter.getFrameHeader();
    if (!frameHeader.isEmpty() && targetVideo.write(frameHeader) == -1) return false;
    if (targetVideo.write(reinterpret_cast<const char *>(finalFrame.data()),
                          finalFrame.size() * 2) == -1) return false;

    baselineFieldsBySeq.remove(seq1);
    baselineFieldsBySeq.remove(seq2);
    upgradedFieldsBySeq.remove(seq1);
    upgradedFieldsBySeq.remove(seq2);
    frameReconstructionMap.remove(outputFrameNumber);
    resolvedOutputFrames.remove(outputFrameNumber);
    upgradedTbcFrames.remove(outputFrameNumber);

    const qint32 outputCount = outputFrameNumber - startFrame + 1;
    if ((outputCount % 32) == 0) {
        double fps = outputCount /
                     (static_cast<double>(totalTimer.elapsed()) / 1000.0);
        qInfo() << outputCount << "frames processed -" << fps << "FPS";
    }

    ++outputFrameNumber;
    return true;
}

bool DecoderPool::getInputFrames(qint32 &startFrameNumber, QList<SourceField> &fields,
                                 qint32 &startIndex, qint32 &endIndex)
{
    QMutexLocker locker(&inputMutex);

    auto getCadenceIdx = [](const CadenceAssembler::WorkItem& wi) -> int {
        if (wi.f1.field.cinemap.cadenceId < 0) return -1;
        return cadenceIndex(wi.f1.field.cinemap.cadenceId);
    };

    auto seqStart = [](const CadenceAssembler::WorkItem& wi) -> int {
        return std::min(wi.f1.field.seqNo, wi.f2.field.seqNo);
    };

    auto seqEndExclusive = [](const CadenceAssembler::WorkItem& wi) -> int {
        return std::max(wi.f1.field.seqNo, wi.f2.field.seqNo) + 1;
    };

    auto isFilmWork = [](const CadenceAssembler::WorkItem& wi) -> bool {
        return wi.kind == CadenceAssembler::WorkItem::Kind::TelecineFrame &&
               wi.f1.field.cinemap.cadenceId >= 0 &&
               wi.f2.field.cinemap.cadenceId >= 0;
    };

    auto isSegStart = [&](const CadenceAssembler::WorkItem& wi) -> bool {
        const bool currIsFilm = isFilmWork(wi);
        if (!currIsFilm) return true;
        if (buildingSeg24p.empty()) return true;

        const auto& prev = buildingSeg24p.back();
        const bool prevIsFilm = isFilmWork(prev);
        if (!prevIsFilm) return true;

        if (wi.f1.field.cinemap.isEditBoundary || wi.f2.field.cinemap.isEditBoundary) return true;

        char pL = prev.filmLabel;
        char cL = wi.filmLabel;

        bool clean = false;
        if (pL == 'A' && cL == 'B') clean = true;
        if (pL == 'B' && cL == 'C') clean = true;
        if (pL == 'C' && cL == 'D') clean = true;
        if (pL == 'D' && cL == 'A') clean = true;

        return !clean;
    };

    auto dropScore24p = [&](const CadenceAssembler::WorkItem& wi) -> int {
        int s = 0;
        if (wi.f1.field.cinemap.cadenceId == -2 || wi.f2.field.cinemap.cadenceId == -2) s += 100;
        if (wi.f1.field.cinemap.cadenceId < 0 || wi.f2.field.cinemap.cadenceId < 0) s += 4;
        if (wi.f1.field.cinemap.isEditBoundary || wi.f2.field.cinemap.isEditBoundary) s += 50;
        return s;
    };

    auto queueAssemblerWorkAheadOfBaselines =
        [&](QVector<CadenceAssembler::WorkItem> produced) {
            // Baseline callbacks run inside CadenceAssembler::push/flush, so
            // they reach workItems before popWork() returns the cadence work
            // that caused them.  Keep the fallback jobs, but decode the
            // planned A/B/C/D reconstructions first.
            std::deque<CadenceAssembler::WorkItem> baselines;
            baselines.swap(workItems);

            for (auto &wi : produced) workItems.push_back(std::move(wi));
            while (!baselines.empty()) {
                workItems.push_back(std::move(baselines.front()));
                baselines.pop_front();
            }
        };

    auto pumpAssembler = [&]() -> bool {
        if (inputFrameNumber > lastFrameNumber) {
            if (cadenceAssembler) {
                cadenceAssembler->flush();
                queueAssemblerWorkAheadOfBaselines(cadenceAssembler->popWork());
            }
            return false;
        }

        const qint32 fetchFrames = qMin(DEFAULT_BATCH_SIZE, lastFrameNumber + 1 - inputFrameNumber);
        const qint32 fetchStart  = inputFrameNumber;
        inputFrameNumber += fetchFrames;

        QVector<SourceField> rawVec;
        qint32 dummyStart = 0, dummyEnd = 0;

        SourceField::loadFields(sourceVideo, ldDecodeMetaData,
                                fetchStart, fetchFrames,
                                0, 0,
                                rawVec, dummyStart, dummyEnd);
                                
        // Trim batch to end on a cadence cycle boundary (cadenceId 9 = D2)
        const int rawSize = static_cast<int>(rawVec.size());
        int trimEnd = rawSize;
        for (int i = rawSize - 1; i >= std::max(0, rawSize - 10); --i) {
            const int cid = rawVec[i].field.cinemap.cadenceId;
            if (cid == 9 || (cid >= 0 && cid % 10 == 9)) {
                trimEnd = i + 1;
                break;
            }
            if (rawVec[i].field.cinemap.isEditBoundary) {
                trimEnd = i;
                break;
            }
        }
        
        if (trimEnd < rawSize) {
            const int fieldsReturned = rawSize - trimEnd;
            inputFrameNumber -= fieldsReturned / 2;
            rawVec.resize(trimEnd);
        }
        if (cadenceAssembler) {
            cadenceAssembler->push(rawVec);
            queueAssemblerWorkAheadOfBaselines(cadenceAssembler->popWork());
        } else {
            for (int i = 0; i + 1 < rawVec.size(); i += 2) {
                CadenceAssembler::WorkItem wi;
                wi.kind = CadenceAssembler::WorkItem::Kind::PassthroughFrame;
                wi.f1 = std::move(rawVec[i]);
                wi.f2 = std::move(rawVec[i + 1]);
                wi.f1.allowProgressiveFrameRegime = false;
                wi.f2.allowProgressiveFrameRegime = false;
                workItems.push_back(std::move(wi));
            }
        }
        return true;
    };

    auto ensureWorkItems = [&]() -> bool {
        while (workItems.empty()) {
            const bool pumped = pumpAssembler();
            if (!pumped && workItems.empty()) return false;
        }
        return true;
    };

    auto finalizeSegment24p = [&]() {
        if (buildingSeg24p.empty()) return;

        const int Nhave = (int)buildingSeg24p.size();

        if (cadenceConfig.emitMax24p) {
            qInfo() << "emit-max-24p: preserving" << Nhave << "frames (no drops)";
            for (auto &w : buildingSeg24p) scheduled24p.push_back(std::move(w));
            framesScheduled24p += Nhave;
            buildingSeg24p.clear();
            return;
        }

        const int segStartSeq = seqStart(buildingSeg24p.front());
        const int segEndSeq   = seqEndExclusive(buildingSeg24p.back());

        if (initialFieldSeq == -1) initialFieldSeq = segStartSeq;

        long long inputDuration = segEndSeq - initialFieldSeq;
        double targetFrames = (double)inputDuration * 0.4;
        long long projectedFrames = framesScheduled24p + Nhave;
        double diff = (double)projectedFrames - targetFrames;

        int dropsNeeded = 0;
        if (diff >= 0.95) dropsNeeded = (int)diff;
        if (dropsNeeded > Nhave) dropsNeeded = Nhave;

        if (dropsNeeded <= 0) {
            for (auto &w : buildingSeg24p) scheduled24p.push_back(std::move(w));
            framesScheduled24p += Nhave;
            buildingSeg24p.clear();
            return;
        }

        const int headZone = std::min(2, Nhave);
        const int tailZone = std::min(2, Nhave);

        std::vector<int> candidates;
        candidates.reserve(headZone + tailZone);
        for (int i = Nhave - tailZone; i < Nhave; ++i) candidates.push_back(i);
        for (int i = 0; i < headZone; ++i) candidates.push_back(i);

        std::stable_sort(candidates.begin(), candidates.end(),
            [&](int a, int b) { return dropScore24p(buildingSeg24p[a]) > dropScore24p(buildingSeg24p[b]); });

        std::vector<char> dropMask(Nhave, 0);
        int dropped = 0;

        if (dropsNeeded > 0) {
            for (int idx : candidates) {
                if (dropped >= dropsNeeded) break;
                if (!dropMask[idx]) {
                    dropMask[idx] = 1;
                    ++dropped;
                }
            }
        }

        for (int i = 0; i < Nhave && dropped < dropsNeeded; ++i) {
            if (!dropMask[i]) {
                dropMask[i] = 1;
                ++dropped;
            }
        }

        int actualAdded = 0;
        for (int i = 0; i < Nhave; ++i) {
            if (dropMask[i]) continue;
            scheduled24p.push_back(std::move(buildingSeg24p[i]));
            actualAdded++;
        }
        framesScheduled24p += actualAdded;
        buildingSeg24p.clear();
    };

    auto ensureScheduled24p = [&]() -> bool {
        while (scheduled24p.empty()) {
            if (!ensureWorkItems()) {
                finalizeSegment24p();
                return !scheduled24p.empty();
            }

            CadenceAssembler::WorkItem in = std::move(workItems.front());
            workItems.pop_front();

            if (isSegStart(in) && !buildingSeg24p.empty()) {
                finalizeSegment24p();
            }

            lastCadenceIndex24p = getCadenceIdx(in);
            buildingSeg24p.push_back(std::move(in));
        }
        return true;
    };

    // Serve a RUN of consecutive frames, not one.
    //
    // The rolling triple-buffer needs {F-1, F, F+1} to emit F, so a call that
    // emits a single frame must analyse three: measured 3.00 analysed per 1
    // output in locked 3D (2.00 in 2D), i.e. two thirds of all locked analysis
    // computed and discarded.  A batch of N amortises the same pre-roll over N
    // outputs -- (N + 2) / N -- without weakening the estimator anywhere.
    //
    // Everything downstream was already batch-shaped: decoder.cpp derives
    // numFrames from (endIndex - startIndex) and putOutputFrames() walks
    // startFrameNumber + i.  Only this producer served one at a time.
    //
    // Determinism: batch composition depends only on the work-item queue
    // order, never on which thread asks or how many exist, so the decode stays
    // reproducible under --threads.  (This is why per-call buffer REUSE was
    // rejected earlier: round-robin meant a thread rarely held its own
    // predecessor, and where it did the picture changed with thread count.)
    fields.clear();
    for (const auto &pad : paddingHistory) fields.push_back(pad);
    const int paddingSize = fields.size();

    qint32 servedThisCall = 0;

    for (qint32 batchSlot = 0; batchSlot < decoderBatchFrames; ++batchSlot) {
    CadenceAssembler::WorkItem wi;
    qint32 thisFrameNumber = 0;

    if (cadenceConfig.export24p) {
        if (!ensureScheduled24p()) break;

        wi = std::move(scheduled24p.front());
        scheduled24p.pop_front();

        thisFrameNumber = nextOutputKey24p++;
    } else {
        if (!ensureWorkItems()) break;

        wi = std::move(workItems.front());
        workItems.pop_front();

        // Non-24p path: servedFrameNumber is only a decode ticket, not slot ownership.
        thisFrameNumber = servedFrameNumber;
    }

    if (servedThisCall == 0)
        startFrameNumber = thisFrameNumber;

    // Keep existing reconstruction metadata if you still want it for debugging / future use.
    if (!cadenceConfig.export24p &&
        wi.kind == CadenceAssembler::WorkItem::Kind::TelecineFrame) {
        FrameReconstructionInfo info;
        info.expansion          = wi.expansion;
        info.swapped            = wi.fieldsSwapped;
        info.invertedFieldOrder = wi.invertedFieldOrder;

        auto roleOf = [&](const SourceField& sf) {
            int cid = sf.field.cinemap.cadenceId;
            if (!cadenceKnown(cid)) return FrameReconstructionInfo::Role::Neutral;
            int idx = cadenceIndex(cid);
            if (idx == 0 || idx == 7) return FrameReconstructionInfo::Role::Def;
            if (idx == 1 || idx == 6 || idx == 8) return FrameReconstructionInfo::Role::Comp;
            return FrameReconstructionInfo::Role::Neutral;
        };

        info.f1Role = roleOf(wi.f1);
        info.f2Role = roleOf(wi.f2);
        info.spareIsTopDef = (info.f1Role == FrameReconstructionInfo::Role::Def);

        QMutexLocker metaLock(&metaDataMutex);
        frameReconstructionMap[thisFrameNumber] = info;
    }
    // Build decode ticket for non-24p.
    if (!cadenceConfig.export24p) {
        DecodeTicket ticket;
        ticket.homeSeq1 = wi.f1.field.seqNo;
        ticket.homeSeq2 = wi.f2.field.seqNo;

        if (wi.kind == CadenceAssembler::WorkItem::Kind::PassthroughFrame) {
            // Cadence resolved to passthrough, so this is baseline fill for exactly these homes.
            ticket.kind = DecodeTicket::Kind::BaselinePair;
        } else {
            // Telecine work only supplies upgrades for home seqs it is actually handling.
            ticket.kind = DecodeTicket::Kind::UpgradePair;

            const bool mergedA =
                (wi.filmLabel == 'A' &&
                 wi.expansion == CadenceAssembler::WorkItem::Expansion::Trailing);

            const bool mergedC =
                (wi.filmLabel == 'C' &&
                 wi.expansion == CadenceAssembler::WorkItem::Expansion::Leading);

            if (mergedA || mergedC) {
                const int role1 = cadenceRoleIndex(wi.f1.field.cinemap.cadenceId);
                const int role2 = cadenceRoleIndex(wi.f2.field.cinemap.cadenceId);
                if (role1 == 0) {
                    ticket.duplicateTwin = true;
                    ticket.twinHomeSeq = wi.f1.field.seqNo + (mergedA ? 2 : -2);
                    ticket.twinSource = 1;
                } else if (role2 == 0) {
                    ticket.duplicateTwin = true;
                    ticket.twinHomeSeq = wi.f2.field.seqNo + (mergedA ? 2 : -2);
                    ticket.twinSource = 2;
                }
            }
        }

        {
            QMutexLocker outLock(&outputMutex);
            if (ticket.kind == DecodeTicket::Kind::UpgradePair) {
                outstandingUpgradeSeqNos.insert(ticket.homeSeq1);
                outstandingUpgradeSeqNos.insert(ticket.homeSeq2);
                if (ticket.duplicateTwin)
                    outstandingUpgradeSeqNos.insert(ticket.twinHomeSeq);
            }
            decodeTicketsByFrameNumber[thisFrameNumber] = ticket;
        }
        
    }

    fields.push_back(std::move(wi.f1));
    fields.push_back(std::move(wi.f2));

    servedFrameNumber += 1;
    ++servedThisCall;

    }   // end batch loop

    if (servedThisCall == 0)
        return false;

    paddingHistory.clear();
    const int total = fields.size();
    if (total >= 4) {
        paddingHistory.push_back(fields[total - 4]);
        paddingHistory.push_back(fields[total - 3]);
        paddingHistory.push_back(fields[total - 2]);
        paddingHistory.push_back(fields[total - 1]);
    } else {
        for (const auto &f : fields) paddingHistory.push_back(f);
    }

    startIndex = paddingSize;
    endIndex   = fields.size();

    // Look-ahead tail.  The window contract is
    //
    //     {lookbehind... [startIndex] real... [endIndex] lookahead...}
    //
    // Decoder derives its frame count from (endIndex - startIndex), so fields
    // appended BEYOND endIndex change neither the number of decoded frames nor
    // their order -- they exist purely so the temporal comb can see the frame
    // that follows the one being decoded.
    //
    // The next work item is PEEKED, never popped.  Serving order, decode
    // tickets, 24p scheduling keys, and output ordering are all untouched; the
    // only cost is copying two fields.  paddingHistory is deliberately built
    // above, from the served window, so the look-ahead frame does not
    // double-advance the look-behind history.
    //
    // At end of stream there is no tail: `next` stays unloaded, the comb sees
    // no temporal context, and that final frame keeps its 2D result.
    if (decoderLookAhead > 0) {
        const CadenceAssembler::WorkItem *ahead = nullptr;
        if (cadenceConfig.export24p) {
            if (ensureScheduled24p() && !scheduled24p.empty())
                ahead = &scheduled24p.front();
        } else {
            if (ensureWorkItems() && !workItems.empty())
                ahead = &workItems.front();
        }
        if (ahead) {
            fields.push_back(ahead->f1);
            fields.push_back(ahead->f2);
        }
    }

    return true;
}

bool DecoderPool::assembleResolvedPairToFrame(qint32 seq1, qint32 seq2, OutputFrame& frame) const
{
    auto getFieldData = [&](qint32 seq) -> const QVector<quint16>* {
        auto upIt = upgradedFieldsBySeq.constFind(seq);
        if (upIt != upgradedFieldsBySeq.constEnd() && !upIt->data.isEmpty())
            return &upIt->data;

        // A fallback decode may finish before an earlier cadence decode on a
        // different worker.  Once an upgrade ticket exists, baseline data is
        // not sufficient to finalize this field until that upgrade arrives.
        if (outstandingUpgradeSeqNos.contains(seq)) return nullptr;

        auto baseIt = baselineFieldsBySeq.constFind(seq);
        if (baseIt != baselineFieldsBySeq.constEnd() && !baseIt->data.isEmpty())
            return &baseIt->data;
        return nullptr;
    };

    const QVector<quint16>* f1data = getFieldData(seq1);
    const QVector<quint16>* f2data = getFieldData(seq2);

    if (!f1data || !f2data) return false;

    frame = interleaveFieldPayloads(*f1data, *f2data);
    return true;
}

bool DecoderPool::putOutputFrames(qint32 startFrameNumber,
                                  const QList<OutputFrame> &outputFrames)
{
    for (int i = 0; i < outputFrames.size(); ++i) {
        if (!putOutputFrame(startFrameNumber + i, outputFrames[i])) {
            return false;
        }
    }
    return true;
}

bool DecoderPool::putOutputFrames(qint32 startFrameNumber,
                                  const QList<ComponentFrame> &componentFrames,
                                  const QList<OutputFrame> &outputFrames)
{
    if (componentFrames.size() != outputFrames.size()) {
        qCritical() << "putOutputFrames: component/output frame count mismatch";
        return false;
    }

    for (int i = 0; i < outputFrames.size(); ++i) {
        if (!putOutputFrame(startFrameNumber + i, &componentFrames[i], outputFrames[i])) {
            return false;
        }
    }
    return true;
}

bool DecoderPool::putOutputFrame(qint32 frameNumber, const OutputFrame &outputFrame)
{
    return putOutputFrame(frameNumber, nullptr, outputFrame);
}

bool DecoderPool::putOutputFrame(qint32 frameNumber,
                                 const ComponentFrame *componentFrame,
                                 const OutputFrame &outputFrame)
{
    QMutexLocker locker(&outputMutex);

    if (cadenceConfig.noPA) {
        pendingOutputFrames[frameNumber] = outputFrame;

        while (pendingOutputFrames.contains(outputFrameNumber)) {
            const OutputFrame &outputData = pendingOutputFrames.value(outputFrameNumber);
            const QByteArray frameHeader = outputWriter.getFrameHeader();
            if (!frameHeader.isEmpty() && targetVideo.write(frameHeader) == -1) return false;
            if (targetVideo.write(reinterpret_cast<const char *>(outputData.data()),
                                  outputData.size() * 2) == -1) return false;

            pendingOutputFrames.remove(outputFrameNumber);
            outputFrameNumber++;

            const qint32 outputCount = outputFrameNumber - startFrame;
            if ((outputCount % 32) == 0) {
                double fps = outputCount /
                             (static_cast<double>(totalTimer.elapsed()) / 1000.0);
                qInfo() << outputCount << "frames processed -" << fps << "FPS";
            }
        }
        return true;
    }

    if (cadenceConfig.export24p) {
        pendingOutputFrames[frameNumber] = outputFrame;

        while (pendingOutputFrames.contains(writeCursor24p)) {
            const OutputFrame &outputData = pendingOutputFrames.value(writeCursor24p);
            const QByteArray frameHeader = outputWriter.getFrameHeader();
            if (!frameHeader.isEmpty() && targetVideo.write(frameHeader) == -1) return false;
            if (targetVideo.write(reinterpret_cast<const char *>(outputData.data()),
                                  outputData.size() * 2) == -1) return false;

            pendingOutputFrames.remove(writeCursor24p);
            writeCursor24p++;

            const qint32 outputCount = writeCursor24p;
            if ((outputCount % 32) == 0) {
                double fps = outputCount /
                             (static_cast<double>(totalTimer.elapsed()) / 1000.0);
                qInfo() << outputCount << "frames processed -" << fps << "FPS";
            }
        }
        return true;
    }

    auto ticketIt = decodeTicketsByFrameNumber.find(frameNumber);
    if (ticketIt == decodeTicketsByFrameNumber.end()) return true;

    const DecodeTicket ticket = ticketIt.value();
    decodeTicketsByFrameNumber.erase(ticketIt);

    const bool isUpgrade = (ticket.kind == DecodeTicket::Kind::UpgradePair);
    auto releaseUpgradeClaims = [&]() {
        if (!isUpgrade) return;
        outstandingUpgradeSeqNos.remove(ticket.homeSeq1);
        outstandingUpgradeSeqNos.remove(ticket.homeSeq2);
        if (ticket.duplicateTwin)
            outstandingUpgradeSeqNos.remove(ticket.twinHomeSeq);
    };

    if (!componentFrame) {
        // This ticket can no longer produce.  Relinquish every claim before
        // aborting so a failed producer never leaves an orphaned registry hold.
        releaseUpgradeClaims();
        qWarning() << "putOutputFrame: 29.97 path called without ComponentFrame"
                   << "for frameNumber" << frameNumber
                   << "homeSeq1" << ticket.homeSeq1
                   << "homeSeq2" << ticket.homeSeq2;
        return false;
    }

    QVector<quint16> f1data, f2data;
    splitOutputFrameToFields(outputFrame, f1data, f2data);

    if (f1data.isEmpty() || f2data.isEmpty()) {
        releaseUpgradeClaims();
        qWarning() << "putOutputFrame: decoded field payload is empty"
                   << "for frameNumber" << frameNumber
                   << "homeSeq1" << ticket.homeSeq1
                   << "homeSeq2" << ticket.homeSeq2;
        return false;
    }

    if (isUpgrade) {
        submitUpgradedField(ticket.homeSeq1, f1data);
        submitUpgradedField(ticket.homeSeq2, f2data);

        if (ticket.duplicateTwin) {
            const QVector<quint16>& twinFieldData =
                (ticket.twinSource == 1) ? f1data : f2data;
            submitUpgradedField(ticket.twinHomeSeq, twinFieldData);
        }

        // The payloads are installed.  This ticket has fulfilled every claim;
        // release the holds before asking the writer to make progress.
        releaseUpgradeClaims();
    } else {
        if (!upgradedFieldsBySeq.contains(ticket.homeSeq1))
            submitBaselineField(ticket.homeSeq1, f1data);
        if (!upgradedFieldsBySeq.contains(ticket.homeSeq2))
            submitBaselineField(ticket.homeSeq2, f2data);
    }

    while (tryEmitNextOriginalPair()) {
    }

    return true;
}
