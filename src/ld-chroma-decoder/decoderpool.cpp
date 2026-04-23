/************************************************************************

    decoderpool.cpp

    ld-chroma-decoder - Colourisation filter for ld-decode
    Copyright (C) 2018-2019 Simon Inns
    Copyright (C) 2021 Phillip Blucas
    Copyright (C) 2021 Adam Sampson
    Copyright (C) 2025-2026 Joseph Burns
    
    This file is part of ld-decode-tools.

    ld-chroma-decoder is free software: you can redistribute it and/or
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

    // Disable padding. 
    outputConfig.paddingAmount = 1;

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
    baselineFramesQueued.clear();
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
                            << "upgrade:"  << haveUp;
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

    auto looksUnclean24p = [](const CadenceAssembler::WorkItem& wi) -> bool {
        if (wi.filmLabel == 'B') return true;
        if (wi.f1.field.cinemap.cadenceId < 0 || wi.f2.field.cinemap.cadenceId < 0) return true;
        return false;
    };

    auto dropScore24p = [&](const CadenceAssembler::WorkItem& wi) -> int {
        int s = 0;
        if (wi.f1.field.cinemap.cadenceId == -2 || wi.f2.field.cinemap.cadenceId == -2) s += 100;
        if (wi.f1.field.cinemap.cadenceId < 0 || wi.f2.field.cinemap.cadenceId < 0) s += 4;
        if (wi.f1.field.cinemap.isEditBoundary || wi.f2.field.cinemap.isEditBoundary) s += 50;
        if (looksUnclean24p(wi)) s += 2;
        return s;
    };

    auto pumpAssembler = [&]() -> bool {
        if (inputFrameNumber > lastFrameNumber) {
            if (cadenceAssembler) {
                cadenceAssembler->flush();
                const auto flushed = cadenceAssembler->popWork();
                for (auto wi : flushed) workItems.push_back(std::move(wi));
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

        if (cadenceAssembler) {
            cadenceAssembler->push(rawVec);
            const auto produced = cadenceAssembler->popWork();
            for (auto wi : produced) workItems.push_back(std::move(wi));
        } else {
            for (int i = 0; i + 1 < rawVec.size(); i += 2) {
                CadenceAssembler::WorkItem wi;
                wi.kind = CadenceAssembler::WorkItem::Kind::PassthroughFrame;
                wi.f1 = std::move(rawVec[i]);
                wi.f2 = std::move(rawVec[i + 1]);
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

    CadenceAssembler::WorkItem wi;

    if (cadenceConfig.export24p) {
        if (!ensureScheduled24p()) return false;

        wi = std::move(scheduled24p.front());
        scheduled24p.pop_front();

        startFrameNumber = nextOutputKey24p++;
    } else {
        if (!ensureWorkItems()) return false;

        wi = std::move(workItems.front());
        workItems.pop_front();

        // Non-24p path: servedFrameNumber is only a decode ticket, not slot ownership.
        startFrameNumber = servedFrameNumber;
    }

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
        frameReconstructionMap[servedFrameNumber] = info;
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
            decodeTicketsByFrameNumber[startFrameNumber] = ticket;
        }
        
	}

    fields.clear();

    for (const auto &pad : paddingHistory) fields.push_back(pad);
    const int paddingSize = fields.size();

    fields.push_back(std::move(wi.f1));
    fields.push_back(std::move(wi.f2));

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

    servedFrameNumber += 1;
    return true;
}

void DecoderPool::submitProcessedFieldPair(qint32 seq1, qint32 seq2, bool isUpgrade)
{
    if (isUpgrade) {
        upgradedFieldsBySeq[seq1] = { {}, true };
        upgradedFieldsBySeq[seq2] = { {}, true };
    } else {
        if (!upgradedFieldsBySeq.contains(seq1))
            baselineFieldsBySeq[seq1] = { {}, true };
        if (!upgradedFieldsBySeq.contains(seq2))
            baselineFieldsBySeq[seq2] = { {}, true };
    }
}

bool DecoderPool::assembleResolvedPairToFrame(qint32 seq1, qint32 seq2, OutputFrame& frame) const
{
    // Check seqNo completion — registry unchanged.
    auto seqReady = [&](qint32 seq) -> bool {
        if (upgradedFieldsBySeq.contains(seq)) return true;
        if (baselineFieldsBySeq.contains(seq)) return true;
        return false;
    };
    if (!seqReady(seq1) || !seqReady(seq2)) return false;

    // Pull the full OutputFrame stored at decode time.
    const qint32 tbcFrame = frameNumberForSeq(seq1);
    auto it = resolvedOutputFrames.constFind(tbcFrame);
    if (it == resolvedOutputFrames.constEnd() || it->isEmpty()) return false;

    frame = *it;
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
    if (ticketIt == decodeTicketsByFrameNumber.end()) {
        return true;
    }

    const DecodeTicket ticket = ticketIt.value();
    decodeTicketsByFrameNumber.erase(ticketIt);

    const bool isUpgrade = (ticket.kind == DecodeTicket::Kind::UpgradePair);

    if (!componentFrame) {
        qWarning() << "putOutputFrame: 29.97 path called without ComponentFrame"
                   << "for frameNumber" << frameNumber
                   << "homeSeq1" << ticket.homeSeq1
                   << "homeSeq2" << ticket.homeSeq2;
        return false;
    }

    // Store the full OutputFrame — no half-frame split.
    // Upgrade caps baseline: once an upgrade is stored the slot is closed.
    const qint32 tbcFrame = frameNumberForSeq(ticket.homeSeq1);
    if (tbcFrame >= 1) {
        if (isUpgrade) {
            resolvedOutputFrames[tbcFrame] = outputFrame;
            upgradedTbcFrames.insert(tbcFrame);
        } else if (!upgradedTbcFrames.contains(tbcFrame)) {
            resolvedOutputFrames[tbcFrame] = outputFrame;
        }
    }

    // Register seqNo completion for registry tracking — no data stored.
    submitProcessedFieldPair(ticket.homeSeq1, ticket.homeSeq2, isUpgrade);

    // Twin writeback: the spare TBC frame gets the same OutputFrame as the A/C frame.
    if (isUpgrade && ticket.duplicateTwin) {
        const qint32 twinTbcFrame = frameNumberForSeq(ticket.twinHomeSeq);
        if (twinTbcFrame >= 1 && !upgradedTbcFrames.contains(twinTbcFrame)) {
            resolvedOutputFrames[twinTbcFrame] = outputFrame;
            upgradedTbcFrames.insert(twinTbcFrame);
        }
        // Register twin seqNo completion.
        upgradedFieldsBySeq[ticket.twinHomeSeq] = { {}, true };
    }

    while (tryEmitNextOriginalPair()) {
    }

    return true;
}