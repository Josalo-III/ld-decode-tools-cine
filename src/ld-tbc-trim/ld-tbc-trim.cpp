/************************************************************************

    ld-tbc-trim.cpp

    Simple TBC/metadata trimmer and decomposer for the ld-decode ecosystem.
    Copyright (C) 2025-2026 Joseph Burns

************************************************************************/

#include <QCoreApplication>
#include <QFileInfo>
#include <QDebug>
#include <QString>
#include <QStringList>
#include <QCommandLineParser>
#include <QCommandLineOption>
#include <QRegularExpression>

#include "lddecodemetadata.h"
#include "tbcwriter.h"
#include "tbc-trim-utils.h"

static void printHelp()
{
    printf(
        "ld-tbc-trim - TBC trimmer and decomposer\n"
        "\n"
        "Trim mode (exactly one of -s, -cav, -clv required):\n"
        "  ld-tbc-trim -s <start_frame> [-l <length>] <input.tbc> <output.tbc>\n"
        "  ld-tbc-trim -cav <frame_number> [-l <length>] <input.tbc> <output.tbc>\n"
        "  ld-tbc-trim -clv <hh:mm:ss:ff> [-l <length>] <input.tbc> <output.tbc>\n"
        "\n"
        "Decompose mode (output argument is used as the output name stem):\n"
        "  ld-tbc-trim --decompose <input.tbc> <stem>\n"
        "  ld-tbc-trim --decompose-edits <input.tbc> <stem>\n"
        "\n"
        "Trim options:\n"
        "  -s <start_frame>     Start at sequential frame number (1-based)\n"
        "  -cav <frame_number>  Start at CAV picture number\n"
        "  -clv <hh:mm:ss:ff>   Start at CLV timecode (hours:minutes:seconds:frame)\n"
        "  -l <length>          Number of frames (default: to end of source)\n"
        "\n"
        "Decompose options:\n"
        "  --decompose          Split into one file pair per contiguous mappable span;\n"
        "                       padded and lead-in/out frames are excluded\n"
        "  --decompose-edits    Split at ld-cinemap edit boundaries; padded frames\n"
        "                       are included within segments; lead-in/out excluded\n"
        "\n"
        "Output naming for decompose modes:\n"
        "  <stem>_001.tbc and <stem>_001.tbc.db, <stem>_002.tbc, etc.\n"
        "\n"
        "Examples:\n"
        "  ld-tbc-trim -s 12001 -l 2400 input.tbc output.tbc\n"
        "  ld-tbc-trim -cav 26000 input.tbc output.tbc\n"
        "  ld-tbc-trim --decompose input.tbc /path/to/disc\n"
        "  ld-tbc-trim --decompose-edits input.tbc /path/to/disc\n"
    );
}

// Parses hh:mm:ss:ff as CLV timecode
static bool parseClvTimecode(const QString &str, LdDecodeMetaData::ClvTimecode &tc)
{
    QRegularExpression re("^(\\d{2}):(\\d{2}):(\\d{2}):(\\d{2})$");
    auto match = re.match(str);
    if (!match.hasMatch()) return false;
    tc.hours         = match.captured(1).toInt();
    tc.minutes       = match.captured(2).toInt();
    tc.seconds       = match.captured(3).toInt();
    tc.pictureNumber = match.captured(4).toInt();
    return true;
}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    QCommandLineParser parser;
    parser.setApplicationDescription("ld-tbc-trim - TBC trimmer and decomposer");
    parser.addHelpOption();

    // Trim selectors
    parser.addOption(QCommandLineOption(QStringList() << "s" << "seq",
        "Sequential start frame number (1-based)", "start"));
    parser.addOption(QCommandLineOption("cav",
        "CAV picture number", "frame"));
    parser.addOption(QCommandLineOption("clv",
        "CLV timecode (hh:mm:ss:ff)", "timecode"));
    parser.addOption(QCommandLineOption(QStringList() << "l" << "length",
        "Number of frames to trim", "length"));

    // Decompose modes
    parser.addOption(QCommandLineOption("decompose",
        "Split into one file pair per contiguous mappable span"));
    parser.addOption(QCommandLineOption("decompose-edits",
        "Split at ld-cinemap edit boundaries"));

    parser.addPositionalArgument("input",  "Input TBC file");
    parser.addPositionalArgument("output", "Output TBC file (trim) or stem (decompose)");

    parser.process(a);

    if (parser.positionalArguments().size() < 2) {
        printHelp();
        return 0;
    }

    const QString inputTbcPath  = parser.positionalArguments().at(0);
    const QString outputArg     = parser.positionalArguments().at(1);
    const QString inputDbPath   = inputTbcPath + ".db";

    const bool bySeq        = parser.isSet("s");
    const bool byCav        = parser.isSet("cav");
    const bool byClv        = parser.isSet("clv");
    const bool doDecompose  = parser.isSet("decompose");
    const bool doDecomposeEdits = parser.isSet("decompose-edits");

    // Validate mode combinations
    if (doDecompose && doDecomposeEdits) {
        qCritical("Error: --decompose and --decompose-edits are mutually exclusive.");
        return -1;
    }

    const bool isDecomposeMode = doDecompose || doDecomposeEdits;
    const int selectorCount = int(bySeq) + int(byCav) + int(byClv);

    if (isDecomposeMode && selectorCount > 0) {
        qCritical("Error: --decompose and --decompose-edits cannot be combined with -s, -cav, or -clv.");
        return -1;
    }

    if (isDecomposeMode && parser.isSet("l")) {
        qCritical("Error: --decompose and --decompose-edits cannot be combined with -l.");
        return -1;
    }

    if (!isDecomposeMode && selectorCount != 1) {
        qCritical("Error: Must provide exactly one of -s, -cav, or -clv.");
        printHelp();
        return -1;
    }

    // --- Load source metadata ---
    LdDecodeMetaData srcMd;
    if (!srcMd.read(inputDbPath)) {
        qCritical("Error: Cannot read input metadata from %s", qPrintable(inputDbPath));
        return -1;
    }

    const int inTotalFrames = srcMd.getNumberOfFrames();
    if (inTotalFrames < 1) {
        qCritical("Error: Input metadata contains no frames.");
        return -1;
    }

    const auto vp = srcMd.getVideoParameters();

    // Guard: --decompose-edits requires ld-cinemap to have been run first.
    // isCinemapped is set by ld-cinemap on successful completion and is the
    // O(1) handshake that cinemap field data is present in the metadata.
    if (doDecomposeEdits && !vp.isCinemapped) {
        qCritical("Error: --decompose-edits requires ld-cinemap to have been run on this source first.");
        return -1;
    }

    // --- Build frame list ---
    const QVector<TbcTrimUtils::FrameInfo> frames = TbcTrimUtils::buildFrameList(&srcMd);

    // =========================================================================
    // Decompose modes
    // =========================================================================

    if (isDecomposeMode) {
        QVector<TbcTrimUtils::Segment> segments;

        if (doDecompose) {
            segments = TbcTrimUtils::buildDecomposeSegments(frames);
            if (segments.isEmpty()) {
                qCritical("Error: No mappable content spans found in input.");
                return -1;
            }
        } else {
            // doDecomposeEdits
            QString error;
            segments = TbcTrimUtils::buildDecomposeEditSegments(&srcMd, frames, error);
            if (!error.isEmpty()) {
                qCritical("Error: %s", qPrintable(error));
                return -1;
            }
            if (segments.isEmpty()) {
                qCritical("Error: No segments produced.");
                return -1;
            }
        }

        qInfo() << "Found" << segments.size() << "segment(s). Writing...";

        const TbcTrimUtils::DecomposeStats stats =
            TbcTrimUtils::writeSegments(&srcMd, frames, segments,
                                        outputArg,
                                        QFileInfo(inputTbcPath),
                                        vp.fieldWidth, vp.fieldHeight);

        // End report
        qInfo() << "";
        if (stats.overlapFrameCount == 0) {
            qInfo("%d segment(s) created.", stats.segmentCount);
        } else {
            qInfo("%d segment(s) created with %d frame(s) repeated across %d segment(s).",
                  stats.segmentCount,
                  stats.overlapFrameCount,
                  stats.overlapSegments);
        }

        return 0;
    }

    // =========================================================================
    // Trim mode
    // =========================================================================

    const QString outputTbcPath = outputArg;
    const QString outputDbPath  = outputTbcPath + ".db";

    // --- Resolve start index ---
    int startIdx = -1;

    if (bySeq) {
        startIdx = parser.value("s").toInt() - 1;  // convert 1-based to 0-based
        if (startIdx < 0 || startIdx >= inTotalFrames) {
            qCritical("Error: Start frame %d out of range (input has %d frames).",
                      startIdx + 1, inTotalFrames);
            return -1;
        }
    } else if (byCav) {
        const int cavNum = parser.value("cav").toInt();
        if (!TbcTrimUtils::findStartByCav(&srcMd, frames, cavNum, startIdx)) {
            qCritical("Error: CAV frame number %d not found in input.", cavNum);
            return -1;
        }
    } else if (byClv) {
        LdDecodeMetaData::ClvTimecode tc = {0, 0, 0, 0};
        if (!parseClvTimecode(parser.value("clv"), tc)) {
            qCritical("Error: Invalid CLV timecode format. Use hh:mm:ss:ff (e.g., 00:42:15:05)");
            return -1;
        }
        if (!TbcTrimUtils::findStartByClv(&srcMd, frames, tc, startIdx)) {
            qCritical("Error: CLV timecode %s not found in input.",
                      qPrintable(parser.value("clv")));
            return -1;
        }
    }

    // --- Resolve length ---
    int length = -1;
    if (parser.isSet("l")) {
        length = parser.value("l").toInt();
        if (length <= 0) {
            qCritical("Error: Length must be a positive integer.");
            return -1;
        }
    }

    if (length == -1) {
        length = inTotalFrames - startIdx;
        qInfo("No length specified - trimming from frame %d to end of source (%d frames).",
              startIdx + 1, length);
    }

    if (startIdx + length > inTotalFrames) {
        const int clamped = inTotalFrames - startIdx;
        qWarning("Warning: Requested %d frames from frame %d but only %d are available - "
                 "clamping to %d frames.", length, startIdx + 1, clamped, clamped);
        length = clamped;
    }

    qInfo() << "Trimming frames" << (startIdx + 1) << "to" << (startIdx + length) << "inclusive.";

    // --- Write output metadata ---
    LdDecodeMetaData *trimmedMd = TbcTrimUtils::sliceMetadata(&srcMd, frames, startIdx, length);
    if (!trimmedMd) {
        qCritical("Error: Failed to slice metadata.");
        return -1;
    }
    trimmedMd->write(outputDbPath);
    delete trimmedMd;
    qInfo() << "Output metadata written to" << outputDbPath;

    // --- Write TBC video ---
    const auto writeFrames = TbcTrimUtils::buildWriteFrames(frames, startIdx, length);

    TbcStreamWriter::Config config;
    config.writeVideo    = true;
    config.writeAudio    = false;
    config.writeMetadata = false;  // already written above
    config.fieldWidth    = vp.fieldWidth;
    config.fieldHeight   = vp.fieldHeight;

    if (!TbcStreamWriter::write(writeFrames, &srcMd,
                                QFileInfo(inputTbcPath),
                                QFileInfo(outputTbcPath),
                                config)) {
        qCritical("Error: Write failed.");
        return -1;
    }

    qInfo() << "Trim successful!";
    qInfo() << "Output files:" << outputTbcPath << "and" << outputDbPath;
    return 0;
}
