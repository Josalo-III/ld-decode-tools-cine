/************************************************************************

    decoderpool.h

    ld-chroma-decoder - Colourisation filter for ld-decode
    Copyright (C) 2018-2019 Simon Inns

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
#ifndef DECODERPOOL_H
#define DECODERPOOL_H

#include <QObject>
#include <QAtomicInt>
#include <QElapsedTimer>
#include <QFile>
#include <QList>
#include <QMap>
#include <QMutex>
#include <QThread>
#include <QString>
#include <memory>
#include <deque>
#include <vector>

#include "lddecodemetadata.h"
#include "sourcevideo.h"

#include "decoder.h"
#include "outputwriter.h"
#include "sourcefield.h"
#include "cadenceassembler.h"

class DecoderPool
{
public:
    explicit DecoderPool(Decoder &decoder, QString inputFileName,
                         LdDecodeMetaData &ldDecodeMetaData,
                         OutputWriter::Configuration &outputConfig,
                         CadenceAssembler::Configuration &cadenceConfig,
                         QString outputFileName,
                         qint32 startFrame, qint32 length, qint32 maxThreads);

    bool process();

    OutputWriter &getOutputWriter() { return outputWriter; }
    Decoder& getDecoder();

    bool getInputFrames(qint32 &startFrameNumber, QList<SourceField> &fields, qint32 &startIndex, qint32 &endIndex);

	bool putOutputFrame(qint32 frameNumber,
                                 const ComponentFrame *componentFrame,
                                 const OutputFrame &outputFrame);
                                 
    bool putOutputFrames(qint32 startFrameNumber, const QList<OutputFrame> &outputFrames);
    
    bool putOutputFrames(qint32 startFrameNumber,
                     const QList<ComponentFrame> &componentFrames,
                     const QList<OutputFrame> &outputFrames);


	const CadenceAssembler::Configuration& getCadenceConfig() const { return cadenceConfig; }

private:
    bool putOutputFrame(qint32 frameNumber, const OutputFrame &outputFrame);
    
    SourceField createBlackField(bool isFirst, int seqNo) const;

	QSet<qint32> baselineFramesQueued; // TBC frame numbers already queued for baseline passthrough
	
	qint32 frameNumberForSeq(qint32 seqNo) const;
	void enqueueBaselinePassthrough(qint32 seqNo);
	QMap<qint32, OutputFrame> resolvedOutputFrames; // keyed by TBC frame number
	QSet<qint32> upgradedTbcFrames;                 // cap: frames with upgrade already stored
    static constexpr qint32 DEFAULT_BATCH_SIZE = 20;

    qint32 nextOutputKey24p = 0;
    qint32 writeCursor24p   = 0;
    
    struct FrameReconstructionInfo {
        CadenceAssembler::WorkItem::Expansion expansion;
        bool swapped;
        bool invertedFieldOrder = false;
        bool temporalFirstIsF1  = true;
        enum class Role { Neutral, Def, Comp };
        Role f1Role = Role::Neutral;
        Role f2Role = Role::Neutral;
        bool spareIsTopDef = true;

        // Capture-time metadata for 30i reconstruction
        int  origSeqFirst      = -1;
        int  origSeqSecond     = -1;
        bool origIsFirstFirst  = true;
        bool origIsFirstSecond = false;
    };
    
	QMap<qint32, FrameReconstructionInfo> frameReconstructionMap;
	QMutex metaDataMutex; // Protects frameReconstructionMap

	
struct StoredFieldPayload
{
    QVector<quint16> data;
    bool ready = false;
};

struct DecodeTicket
{
    enum class Kind {
        BaselinePair,   // passthrough / no-pa style result for exactly these homes
        UpgradePair     // telecine result that may improve these homes
    };

    Kind kind = Kind::BaselinePair;

    qint32 homeSeq1 = -1;
    qint32 homeSeq2 = -1;

    // Optional twin writeback for merged A/C
    bool duplicateTwin = false;
    qint32 twinHomeSeq = -1;
    int twinSource = 0; // 1 => use field 1 payload, 2 => use field 2 payload
};
    
	void registerFieldUpgrade(qint32 fieldSequenceNumber,
										   const QVector<quint16> &upgradedField);
										   
	void submitBaselineField(qint32 seqNo, const QVector<quint16>& data);
	void submitUpgradedField(qint32 seqNo, const QVector<quint16>& data);
	bool tryEmitNextOriginalPair();

	void submitProcessedFieldPair(qint32 seq1, qint32 seq2, bool isUpgrade);
	bool assembleResolvedPairToFrame(qint32 seq1, qint32 seq2, OutputFrame& frame) const;
	// transitional worker bookkeeping; ownership remains seq-based
	QMap<qint32, DecodeTicket> decodeTicketsByFrameNumber;

    Decoder &decoder;
    QString inputFileName;
    OutputWriter::Configuration outputConfig;
    CadenceAssembler::Configuration cadenceConfig;
    QString outputFileName;
    qint32 startFrame;
    qint32 length;
    qint32 maxThreads;

    QAtomicInt abort;

    QMutex inputMutex;
    qint32 decoderLookBehind = 0;
    qint32 decoderLookAhead  = 0;
    qint32 inputFrameNumber  = 1;
    qint32 lastFrameNumber   = 1;
    qint32 servedFrameNumber = 1;
    
    LdDecodeMetaData &ldDecodeMetaData;
    LdDecodeMetaData::VideoParameters effectiveVideoParameters;

    SourceVideo sourceVideo;

    QMutex outputMutex;
    qint32 outputFrameNumber = 1;
    QMap<qint32, OutputFrame> pendingOutputFrames;
    OutputWriter outputWriter;
    QFile targetVideo;
    QElapsedTimer totalTimer;
	QMap<qint32, StoredFieldPayload> baselineFieldsBySeq;
	QMap<qint32, StoredFieldPayload> upgradedFieldsBySeq;
    std::deque<CadenceAssembler::WorkItem> scheduled24p;
    std::vector<CadenceAssembler::WorkItem> buildingSeg24p;

    // --- Absolute Clock State ---
    long long initialFieldSeq = -1;
    long long framesScheduled24p = 0;
    int lastCadenceIndex24p = -1; 

	std::unique_ptr<CadenceAssembler> cadenceAssembler;

    std::deque<CadenceAssembler::WorkItem> workItems;
    
    std::deque<SourceField> paddingHistory;
};

#endif // DECODERPOOL_H