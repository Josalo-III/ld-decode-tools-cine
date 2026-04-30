/******************************************************************************
 * main.cpp
 * ld-chroma-decoder — Colourisation filter for ld-decode
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2018-2020 Simon Inns
 * SPDX-FileCopyrightText: 2019-2022 Adam Sampson
 * SPDX-FileCopyrightText: 2021 Chad Page
 * SPDX-FileCopyrightText: 2021 Phillip Blucas
 * SPDX-FileCopyrightText: 2025 Joseph Burns
 *
 * This file is part of ld-decode-tools.
 ******************************************************************************/

#include <QCoreApplication>
#include <QDebug>
#include <QtGlobal>
#include <QCommandLineParser>
#include <QThread>
#include <fstream>
#include <memory>

#include "decoderpool.h"
#include "lddecodemetadata.h"
#include "logging.h"

#include "comb.h"
#include "monodecoder.h"
#include "ntscdecoder.h"
#include "outputwriter.h"
#include "palcolour.h"
#include "paldecoder.h"
#include "transformpal.h"
#include "cadenceassembler.h"

// Enable fast FP mode (flush denormals) on supported CPUs
#if defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86)
  #include <xmmintrin.h>
  #include <pmmintrin.h>
#endif
#if defined(__aarch64__)
  #include <stdint.h>
#endif

static inline void enableFastFPUMode()
{
#if defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  #if defined(_MM_DENORMALS_ZERO_MASK)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  #endif
#endif
#if defined(__aarch64__)
    // Set FPCR: FZ (bit 24). DN (bit 25) controls Default NaN; we leave it unchanged.
    uint64_t fpcr;
    asm volatile("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= (1ull << 24); // FZ: Flush-to-zero for subnormals
    asm volatile("msr fpcr, %0" : : "r"(fpcr));
#endif
}

// Load the thresholds file for the Transform decoders, if specified.
static bool loadTransformThresholds(QCommandLineParser &parser, QCommandLineOption &transformThresholdsOption, PalColour::Configuration &palConfig)
{
    if (!parser.isSet(transformThresholdsOption)) {
        return true;
    }

    QString filename = parser.value(transformThresholdsOption);
    std::ifstream thresholdsFile(filename.toStdString());
    if (thresholdsFile.fail()) {
        qCritical() << "Transform thresholds file could not be opened:" << filename;
        return false;
    }

    palConfig.transformThresholds.clear();
    while (true) {
        double value;
        thresholdsFile >> value;
        if (thresholdsFile.eof()) break;
        if (value < 0.0 || value > 1.0) {
            qCritical() << "Values in Transform thresholds file must be between 0 and 1:" << filename;
            return false;
        }
        if (thresholdsFile.fail()) {
            qCritical() << "Couldn't parse Transform thresholds file:" << filename;
            return false;
        }
        palConfig.transformThresholds.push_back(value);
    }

    if (palConfig.transformThresholds.size() != palConfig.getThresholdsSize()) {
        qCritical() << "Transform thresholds file contained" << palConfig.transformThresholds.size()
                    << "values, expecting" << palConfig.getThresholdsSize() << "values:" << filename;
        return false;
    }

    thresholdsFile.close();
    return true;
}

int main(int argc, char *argv[])
{
    // Set 'binary mode' for stdin and stdout on Windows
    setBinaryMode();

    // Critical on Apple Silicon and other CPUs: avoid denormal slowdowns
    enableFastFPUMode();

    // Install the local debug message handler
    setDebug(true);
    qInstallMessageHandler(debugOutputHandler);

    QCoreApplication a(argc, argv);

    // Set application name and version
    QCoreApplication::setApplicationName("ld-chroma-decoder");
    QCoreApplication::setApplicationVersion(QString("Branch: %1 / Commit: %2").arg(APP_BRANCH, APP_COMMIT));
    QCoreApplication::setOrganizationDomain("domesday86.com");

    // Set up the command line parser
    QCommandLineParser parser;
    parser.setApplicationDescription(
                "ld-chroma-decoder - Colourisation filter for ld-decode\n"
                "\n"
                "(c)2018-2020 Simon Inns\n"
                "(c)2019-2021 Adam Sampson\n"
                "(c)2018-2021 Chad Page\n"
                "(c)2021 Phillip Blucas\n"
                "(c)2025 Joseph Burns\n"
                "Contains PALcolour: Copyright (c)2018 William Andrew Steer\n"
                "Contains Transform PAL: Copyright (c)2014 Jim Easterbrook\n"
                "GPLv3 Open-Source - github: https://github.com/happycube/ld-decode");
    parser.addHelpOption();
    parser.addVersionOption();

    // -- General options --

    // Add the standard debug options --debug and --quiet
    addStandardDebugOptions(parser);

    // Option to specify a different JSON input file
    QCommandLineOption inputJsonOption(QStringList() << "input-metadata",
                                       QCoreApplication::translate("main", "Specify the input metadata file (default input.tbc.db)"),
                                       QCoreApplication::translate("main", "filename"));
    parser.addOption(inputJsonOption);

    // Option to select start frame (sequential) (-s)
    QCommandLineOption startFrameOption(QStringList() << "s" << "start",
                                        QCoreApplication::translate("main", "Specify the start frame number"),
                                        QCoreApplication::translate("main", "number"));
    parser.addOption(startFrameOption);

    // Option to select length (-l)
    QCommandLineOption lengthOption(QStringList() << "l" << "length",
                                        QCoreApplication::translate("main", "Specify the length (number of frames to process)"),
                                        QCoreApplication::translate("main", "number"));
    parser.addOption(lengthOption);

    // Option to reverse the field order (-r)
    QCommandLineOption setReverseOption(QStringList() << "r" << "reverse",
                                       QCoreApplication::translate("main", "Reverse the field order to second/first (default first/second)"));
    parser.addOption(setReverseOption);

    // Option to specify chroma gain
    QCommandLineOption chromaGainOption(QStringList() << "chroma-gain",
                                        QCoreApplication::translate("main", "Gain factor applied to chroma components (default 1.0)"),
                                        QCoreApplication::translate("main", "number"));
    parser.addOption(chromaGainOption);

    // Option to specify chroma phase
    QCommandLineOption chromaPhaseOption(QStringList() << "chroma-phase",
                                        QCoreApplication::translate("main", "Phase rotation applied to chroma components (degrees; default 0.0)"),
                                        QCoreApplication::translate("main", "number"));
    parser.addOption(chromaPhaseOption);

    // Option to select the output format (-p)
    QCommandLineOption outputFormatOption(QStringList() << "p" << "output-format",
                                       QCoreApplication::translate("main", "Output format (rgb, yuv, y4m, or explicit pixel formats RGB48, YUV444P16, GRAY16; default rgb)"),
                                       QCoreApplication::translate("main", "output-format"));
    parser.addOption(outputFormatOption);

    // Option to set the black and white output flag (-b)
    QCommandLineOption setBwModeOption(QStringList() << "b" << "blackandwhite",
                                       QCoreApplication::translate("main", "Output in black and white"));
    parser.addOption(setBwModeOption);

    // Option to select output padding (-pad)
    QCommandLineOption outputPaddingOption(QStringList() << "pad" << "output-padding",
                                       QCoreApplication::translate("main", "Pad the output frame to a multiple of this many pixels on both axes (1 means no padding, maximum is 32)"),
                                       QCoreApplication::translate("main", "number"));
    parser.addOption(outputPaddingOption);

    // Option to select which decoder to use (-f)
    QCommandLineOption decoderOption(QStringList() << "f" << "decoder",
                                     QCoreApplication::translate("main", "Decoder to use (pal2d, transform2d, transform3d, ntsc1d, ntsc2d, ntsc3d, ntsc3dnoadapt, mono; default automatic)"),
                                     QCoreApplication::translate("main", "decoder"));
    parser.addOption(decoderOption);

    // Option to select the number of threads (-t)
    QCommandLineOption threadsOption(QStringList() << "t" << "threads",
                                     QCoreApplication::translate("main", "Specify the number of concurrent threads (default number of logical CPUs)"),
                                     QCoreApplication::translate("main", "number"));
    parser.addOption(threadsOption);

    // Options to override line parameters
    QCommandLineOption firstFieldLineOption(QStringList() << "ffll" << "first_active_field_line",
                                            QCoreApplication::translate("main", "The first visible line of a field. Range 1-259 for NTSC (default: 20), 2-308 for PAL (default: 22)"),
                                            QCoreApplication::translate("main", "number"));
    parser.addOption(firstFieldLineOption);

    QCommandLineOption lastFieldLineOption(QStringList() << "lfll" << "last_active_field_line",
                                           QCoreApplication::translate("main", "The last visible line of a field. Range 1-259 for NTSC (default: 259), 2-308 for PAL (default: 308)"),
                                           QCoreApplication::translate("main", "number"));
    parser.addOption(lastFieldLineOption);

    QCommandLineOption firstFrameLineOption(QStringList() << "ffrl" << "first_active_frame_line",
                                            QCoreApplication::translate("main", "The first visible line of a frame. Range 1-525 for NTSC (default: 40), 1-620 for PAL (default: 44)"),
                                            QCoreApplication::translate("main", "number"));
    parser.addOption(firstFrameLineOption);

    QCommandLineOption lastFrameLineOption(QStringList() << "lfrl" << "last_active_frame_line",
                                           QCoreApplication::translate("main", "The last visible line of a frame. Range 1-525 for NTSC (default: 525), 1-620 for PAL (default: 620)"),
                                           QCoreApplication::translate("main", "number"));
    parser.addOption(lastFrameLineOption);

    // -- NTSC decoder options --
    QCommandLineOption showMapOption(QStringList() << "o" << "oftest",
                                            QCoreApplication::translate("main", "NTSC: Overlay the adaptive filter map (only used for testing)"));
    parser.addOption(showMapOption);

    QCommandLineOption chromaNROption(QStringList() << "chroma-nr",
                                      QCoreApplication::translate("main", "NTSC: Chroma noise reduction level in dB (default 0.0)"),
                                      QCoreApplication::translate("main", "number"));
    parser.addOption(chromaNROption);

    QCommandLineOption lumaNROption(QStringList() << "luma-nr",
                                    QCoreApplication::translate("main", "Luma noise reduction level in dB (default 0.0)"),
                                    QCoreApplication::translate("main", "number"));
    parser.addOption(lumaNROption);

    QCommandLineOption ntscPhaseCompOption(QStringList() << "ntsc-phase-comp",
                                           QCoreApplication::translate("main", "NTSC: Adjust phase per-line using burst phase -"
                                           "also enables the advanced 2D interfield comb and election, high frequency Y from composite residual, and discrete filtering of I and Q"));
    parser.addOption(ntscPhaseCompOption);

    QCommandLineOption vdisOption(QStringList() << "vdis",
                                            QCoreApplication::translate("main", "NTSC: Enable VDIS (Vertical Differential Isolation System) " 
                                            "This restricts the 2d section very substantially - for use when regular output exhibits artifacts near horizontal color boundaries"));
    parser.addOption(vdisOption);

    QCommandLineOption noResidualVideoOption(QStringList() << "no-residual-video",
                                             QCoreApplication::translate("main", "NTSC (locked mode): Disable composite-derived residual video (Y and color)"));
    parser.addOption(noResidualVideoOption);

    QCommandLineOption residualVideo3DOption(QStringList() << "residual-video-3d",
                                         QCoreApplication::translate("main", "Enable temporal enhancement for residual video"));
    parser.addOption(residualVideo3DOption);

    QCommandLineOption noResidualColorOption(QStringList() << "no-residual-color",
                                         QCoreApplication::translate("main", "NTSC (locked mode): Keep residual Y but disable residual color refinement"));
    parser.addOption(noResidualColorOption);

    QCommandLineOption adaptThresholdOption(QStringList() << "adapt-threshold",
        QCoreApplication::translate("main",
            "NTSC: 3D adaptive filter threshold (default 1.0). "
            "Higher values increase the reward for temporally-agreeing candidates, "
            "biasing selection toward 3D results. Lower values are more conservative, "
            "preferring 2D/1D on motion. Veto for large deviations is unaffected."),
        QCoreApplication::translate("main", "number"));
    parser.addOption(adaptThresholdOption);

    QCommandLineOption chromaWeightOption(QStringList() << "chroma-weight",
        QCoreApplication::translate("main",
            "NTSC: Chroma weight for 3D adaptive filter (default 1.0). "
            "Higher values make chroma disagreement between candidate and reference "
            "more costly, biasing selection toward 2D/1D when colours differ."),
        QCoreApplication::translate("main", "number"));
    parser.addOption(chromaWeightOption);

    QCommandLineOption twoDVariantOption(
        QStringList() << QCoreApplication::translate("main", "two-d-variant"),
        QCoreApplication::translate("main",
            "2D comb variant: line | field | fieldb | frame | fvf (default) Select between 2D comb filters: Frame is an interfield comb; Field and Field B are intrafield combs, B is simpler; Line is 1D - FVF (Field Vs Frame) intelligently selects per pixel from Frame and Field A/B "),
        QCoreApplication::translate("main", "variant"),
        QCoreApplication::translate("main", "fvf"));
    parser.addOption(twoDVariantOption);

    // -- Cadence / PA options --
    QCommandLineOption setCadenceOption(
        QStringList() << "set-cadence",
        QCoreApplication::translate("main",
            "Impose a global pulldown interpretation, selecting video start frames for the 5 -frame cycle, overriding cadenceId metadata from CineMap -"
            "Values are: "
            "1=AA, 2=AB, 3=BC, 4=CC, 5=DD"),
        QCoreApplication::translate("main", "number"));
    parser.addOption(setCadenceOption);

    QCommandLineOption export24pOption(QStringList() << "export-24p",
        QCoreApplication::translate("main", "Export 23.976 fps direct from consolidated telecine"
        "- this version re-syncs to the original timing, trimming as needed)"));
    parser.addOption(export24pOption);

    // Max 24p option: emit every possible film frame (can be very long)
    QCommandLineOption emitMax24pOption(QStringList() << "emit-max-24p",
        QCoreApplication::translate("main",
            "Use with --export-24p: outputs every film frame for which two opposite fields are available, "
            "bypassing the per-segment resync (with trims) pass; thus produces longer, asynchronous output."));
    parser.addOption(emitMax24pOption);

    // Option to enable visual cadence debugging
    QCommandLineOption debugCadenceOption(QStringList() << "debug-cadence",
                                    QCoreApplication::translate("main", "Overlay the detected film frame (A, B, C, D) as well as edit boundaries on the image. For assessing ld-cinemap errors"));
    parser.addOption(debugCadenceOption);
    
    QCommandLineOption noPAOption(QStringList() << "no-pa",
        QCoreApplication::translate("main", "Disable pulldown awareness - reverts to original 29.97 video process"));
    parser.addOption(noPAOption);

    QCommandLineOption dgDiscardOption(QStringList() << "dg-discard",
        QCoreApplication::translate("main", "Skip pulldown consolidation and discard spare fields"
        " - trades quality for speed"));
    parser.addOption(dgDiscardOption);

    QCommandLineOption dgOutlierOption(QStringList() << "dg-outlier-thresh",
        QCoreApplication::translate("main", "When duplicate fields are merged, pixels are corrected instead of averaged when difference between twins is above this threshold (default 6 IRE)"),
        QCoreApplication::translate("main", "number"));
    parser.addOption(dgOutlierOption);

    // ---- PAL decoder options ----------

    QCommandLineOption simplePALOption(QStringList() << "simple-pal",
                                       QCoreApplication::translate("main", "Transform: Use 1D UV filter (default 2D)"));
    parser.addOption(simplePALOption);

    QCommandLineOption transformThresholdOption(QStringList() << "transform-threshold",
                                                QCoreApplication::translate("main", "Transform: Uniform similarity threshold (default 0.4)"),
                                                QCoreApplication::translate("main", "number"));
    parser.addOption(transformThresholdOption);

    QCommandLineOption transformThresholdsOption(QStringList() << "transform-thresholds",
                                                 QCoreApplication::translate("main", "Transform: File containing per-bin similarity thresholds"),
                                                 QCoreApplication::translate("main", "file"));
    parser.addOption(transformThresholdsOption);

    QCommandLineOption showFFTsOption(QStringList() << "show-ffts",
                                      QCoreApplication::translate("main", "Transform: Overlay the input and output FFTs"));
    parser.addOption(showFFTsOption);

    // Positional args
    parser.addPositionalArgument("input", QCoreApplication::translate("main", "Specify input TBC file (- for piped input)"));
    parser.addPositionalArgument("output", QCoreApplication::translate("main", "Specify output file (omit or - for piped output)"));

    // Process the command line options and arguments
    parser.process(a);

    // Standard logging options
    processStandardDebugOptions(parser);
    emitDeprecatedToolWarning();

    // Get the arguments from the parser
    QString inputFileName;
    QString outputFileName = "-";
    QStringList positionalArguments = parser.positionalArguments();
    if (positionalArguments.count() == 2) {
        inputFileName = positionalArguments.at(0);
        outputFileName = positionalArguments.at(1);
    } else if (positionalArguments.count() == 1) {
        inputFileName = positionalArguments.at(0);
    } else {
        qCritical("You must specify the input TBC and output files");
        return -1;
    }

    if (inputFileName == "-" && !parser.isSet(inputJsonOption)) {
        qCritical("With piped input, you must also specify the input metadata file");
        return -1;
    }
    if (inputFileName == outputFileName && outputFileName != "-") {
        qCritical("Input and output files cannot be the same");
        return -1;
    }

    qint32 startFrame = -1;
    qint32 length = -1;
    qint32 maxThreads = QThread::idealThreadCount();
    PalColour::Configuration palConfig;
    Comb::Configuration combConfig;
    MonoDecoder::MonoConfiguration monoConfig;
    OutputWriter::Configuration outputConfig;
    CadenceAssembler::Configuration cadenceConfig;

    // Parse Cadence / PA options
    if (parser.isSet(export24pOption)) {
        cadenceConfig.export24p = true;
        outputConfig.export24p = true; // Tell output writer to use 24p mode
    }
    if (parser.isSet(noPAOption)) cadenceConfig.noPA = true;
    if (parser.isSet(dgDiscardOption)) cadenceConfig.dgDiscard = true;
    if (parser.isSet(dgOutlierOption)) {
        cadenceConfig.dgOutlierThreshIre = parser.value(dgOutlierOption).toDouble();
    }
    if (parser.isSet(emitMax24pOption)) {
        cadenceConfig.emitMax24p = true;
        if (!parser.isSet(export24pOption)) {
            qWarning() << "--emit-max-24p has no effect without --export-24p";
        }
    }

    if (parser.isSet(debugCadenceOption)) {
        combConfig.debugCadence = true;
    }
    if (cadenceConfig.noPA && cadenceConfig.setCadence != 0) {
    qCritical() << "--set-cadence is incompatible with --no-pa (pulldown processing disabled)";
    return -1;
    }
    
    QString v = parser.value(twoDVariantOption).toLower();
    if (v == "line") {
        combConfig.twoDVariant = Comb::Configuration::TwoDVariant::Line;
    } else if (v == "field") {
        combConfig.twoDVariant = Comb::Configuration::TwoDVariant::Field;
    } else if (v == "fieldb") {
        combConfig.twoDVariant = Comb::Configuration::TwoDVariant::FieldB;
    } else if (v == "frame" || v == "frameb2") {
        combConfig.twoDVariant = Comb::Configuration::TwoDVariant::Frame;
    } else if (v == "fvf" || v == "fieldvframe") {
        combConfig.twoDVariant = Comb::Configuration::TwoDVariant::FieldVsFrame;
    } else {
        qWarning() << "Unknown two-d-variant" << v << " defaulting to fvf";
        combConfig.twoDVariant = Comb::Configuration::TwoDVariant::FieldVsFrame;
    }

    if (parser.isSet(startFrameOption)) {
        startFrame = parser.value(startFrameOption).toInt();
        if (startFrame < 1) {
            qCritical("Specified startFrame must be at least 1");
            return -1;
        }
    }

    if (parser.isSet(lengthOption)) {
        length = parser.value(lengthOption).toInt();
        if (length < 1) {
            qCritical("Specified length must be greater than zero frames");
            return -1;
        }
    }

    if (parser.isSet(threadsOption)) {
        maxThreads = parser.value(threadsOption).toInt();
        if (maxThreads < 1) {
            qCritical("Specified number of threads must be greater than zero");
            return -1;
        }
    }

    if (parser.isSet(chromaGainOption)) {
        const double value = parser.value(chromaGainOption).toDouble();
        palConfig.chromaGain = value;
        combConfig.chromaGain = value;
        if (value < 0.0) {
            qCritical("Chroma gain must not be less than 0");
            return -1;
        }
    }

    if (parser.isSet(chromaPhaseOption)) {
        const double value = parser.value(chromaPhaseOption).toDouble();
        palConfig.chromaPhase = value;
        combConfig.chromaPhase = value;
    }

    bool bwMode = parser.isSet(setBwModeOption);
    if (bwMode) {
        palConfig.chromaGain = 0.0;
        combConfig.chromaGain = 0.0;
    }

    if (parser.isSet(showMapOption)) {
        combConfig.showMap = true;
    }

    if (parser.isSet(chromaNROption)) {
        combConfig.cNRLevel = parser.value(chromaNROption).toDouble();
        if (combConfig.cNRLevel < 0.0) {
            qCritical("Chroma noise reduction cannot be negative");
            return -1;
        }
    }

    if (parser.isSet(lumaNROption)) {
        double yNR = parser.value(lumaNROption).toDouble();
        combConfig.yNRLevel = yNR;
        palConfig.yNRLevel = yNR;
        monoConfig.yNRLevel = yNR;
        if (combConfig.yNRLevel < 0.0) {
            qCritical("Luma noise reduction cannot be negative");
            return -1;
        }
    }

    if (parser.isSet(ntscPhaseCompOption)) {
        combConfig.phaseCompensation = true;
    }

    if (parser.isSet(vdisOption)) {
        combConfig.tunables.VDIS_ENABLE = true;
    }

    // Residual video: enabled by default when phase compensation is active
    if (parser.isSet(noResidualVideoOption)) {
        combConfig.tunables.VET_ENABLE_RESIDUAL_Y = false;
        combConfig.residualColor = false;
    } else {
        combConfig.tunables.VET_ENABLE_RESIDUAL_Y = combConfig.phaseCompensation ? true : combConfig.tunables.VET_ENABLE_RESIDUAL_Y;
        combConfig.residualColor = combConfig.phaseCompensation ? true : false;
    }

    if (parser.isSet(noResidualColorOption)) {
        combConfig.residualColor = false;
    }

    if (parser.isSet(adaptThresholdOption)) {
        const double v = parser.value(adaptThresholdOption).toDouble();
        if (v <= 0.0) {
            qCritical("Adapt threshold must be greater than 0");
            return -1;
        }
        combConfig.adaptThreshold = v;
    }

    if (parser.isSet(chromaWeightOption)) {
        const double v = parser.value(chromaWeightOption).toDouble();
        if (v < 0.0) {
            qCritical("Chroma weight must be greater than or equal to 0");
            return -1;
        }
        combConfig.chromaWeight = v;
    }

    if (parser.isSet(transformThresholdOption)) {
        palConfig.transformThreshold = parser.value(transformThresholdOption).toDouble();
        if (palConfig.transformThreshold < 0.0 || palConfig.transformThreshold > 1.0) {
            qCritical("Transform threshold must be between 0 and 1");
            return -1;
        }
    }

    if (parser.isSet(showFFTsOption)) {
        palConfig.showFFTs = true;
    }

    if (parser.isSet(simplePALOption)) {
        palConfig.simplePAL = true;
    }

    // Optional overrides for line parameters
    LdDecodeMetaData::LineParameters lineParameters;
    if (parser.isSet(firstFieldLineOption)) {
        lineParameters.firstActiveFieldLine = parser.value(firstFieldLineOption).toInt();
    }
    if (parser.isSet(lastFieldLineOption)) {
        lineParameters.lastActiveFieldLine = parser.value(lastFieldLineOption).toInt();
    }
    if (parser.isSet(firstFrameLineOption)) {
        lineParameters.firstActiveFrameLine = parser.value(firstFrameLineOption).toInt();
    }
    if (parser.isSet(lastFrameLineOption)) {
        lineParameters.lastActiveFrameLine = parser.value(lastFrameLineOption).toInt();
    }

    // Work out the metadata filename
    QString inputJsonFileName = inputFileName + ".db";
    if (parser.isSet(inputJsonOption)) {
        inputJsonFileName = parser.value(inputJsonOption);
    }

    // Load the source video metadata
    LdDecodeMetaData metaData;
    if (!metaData.read(inputJsonFileName)) {
        qCritical() << "Unable to open ld-decode metadata file:" << inputJsonFileName;
        return -1;
    }

    metaData.processLineParameters(lineParameters);

    // Reverse field order if required
    if (parser.isSet(setReverseOption)) {
        qInfo() << "Expected field order is reversed to second field/first field";
        metaData.setIsFirstFieldFirst(false);
        cadenceConfig.reverseFieldOrder = true;
    }

    // Select decoder
    QString decoderName;
    if (parser.isSet(decoderOption)) {
        decoderName = parser.value(decoderOption);
    } else if (metaData.getVideoParameters().system == NTSC) {
        decoderName = "ntsc2d";
    } else {
        decoderName = "pal2d";
    }
    
    // Require ntsc3d if the map overlay is selected
    if (combConfig.showMap && decoderName != "ntsc3d") {
        qCritical() << "Can only show adaptive filter map with the ntsc3d decoder";
        return -1;
    }

    // Require transform2d/3d if the FFT overlay is selected
    if (palConfig.showFFTs && decoderName != "transform2d" && decoderName != "transform3d") {
        qCritical() << "Can only show FFTs with the transform2d/transform3d decoders";
        return -1;
    }

    // Create decoder
    std::unique_ptr<Decoder> decoder;
    if (decoderName == "pal2d") {
        decoder = std::make_unique<PalDecoder>(palConfig);
    } else if (decoderName == "transform2d") {
        palConfig.chromaFilter = PalColour::transform2DFilter;
        if (!loadTransformThresholds(parser, transformThresholdsOption, palConfig)) return -1;
        decoder = std::make_unique<PalDecoder>(palConfig);
    } else if (decoderName == "transform3d") {
        palConfig.chromaFilter = PalColour::transform3DFilter;
        if (!loadTransformThresholds(parser, transformThresholdsOption, palConfig)) return -1;
        decoder = std::make_unique<PalDecoder>(palConfig);
    } else if (decoderName == "ntsc1d") {
        combConfig.dimensions = 1;
        decoder = std::make_unique<NtscDecoder>(combConfig);
    } else if (decoderName == "ntsc2d") {
        combConfig.dimensions = 2;
        decoder = std::make_unique<NtscDecoder>(combConfig);
    } else if (decoderName == "ntsc3d") {
        combConfig.dimensions = 3;
        decoder = std::make_unique<NtscDecoder>(combConfig);
    } else if (decoderName == "ntsc3dnoadapt") {
        combConfig.dimensions = 3;
        combConfig.adaptive = false;
        decoder = std::make_unique<NtscDecoder>(combConfig);
    } else if (decoderName == "mono") {
        decoder = std::make_unique<MonoDecoder>(monoConfig);
    } else {
        qCritical() << "Unknown decoder" << decoderName;
        return -1;
    }

    if (parser.isSet(residualVideo3DOption)) {
        combConfig.residualVideo3D = true;
    }
    
    // Select output format
    QString outputFormatName;
    if (parser.isSet(outputFormatOption)) {
        outputFormatName = parser.value(outputFormatOption);
    } else {
        outputFormatName = "rgb";
    }

    // Accept explicit pixel format strings as well as legacy names.
    QString fmt = outputFormatName.trimmed().toLower();
    if (fmt == "y4m") {
        outputConfig.outputY4m = true;
        fmt = "yuv";
    }

    if (fmt == "yuv" || fmt == "yuv444p16") {
        if (bwMode || decoderName == "mono") {
            outputConfig.pixelFormat = OutputWriter::PixelFormat::GRAY16;
        } else {
            outputConfig.pixelFormat = OutputWriter::PixelFormat::YUV444P16;
        }
    } else if (fmt == "rgb" || fmt == "rgb48") {
        outputConfig.pixelFormat = OutputWriter::PixelFormat::RGB48;
    } else if (fmt == "gray16" || fmt == "grey16" || fmt == "gray") {
        outputConfig.pixelFormat = OutputWriter::PixelFormat::GRAY16;
    } else {
        qCritical() << "Unknown output format" << outputFormatName;
        return -1;
    }
    
    if (parser.isSet(outputPaddingOption)) {
        outputConfig.paddingAmount = parser.value(outputPaddingOption).toInt();
        if (outputConfig.paddingAmount < 1 || outputConfig.paddingAmount > 32) {
            qInfo() << "Invalid value" << outputConfig.paddingAmount << "specified for padding amount, defaulting to 8.";
            outputConfig.paddingAmount = 8;
        }
    }
    // After other cadenceConfig parsing in main.cpp
    if (parser.isSet(setCadenceOption)) {
        int val = parser.value(setCadenceOption).toInt();
        if (val < 1 || val > 5) {
            qCritical() << "--set-cadence must be in the range 1..5";
            return -1;
        }
        cadenceConfig.setCadence = val;
            // Send the notice to stderr so it doesnt corrupt piped stdout (y4m/raw)
            fprintf(stderr,
                    "Info: Forcing global cadence start at index %d (AA=1, AB=2, BC=3, CC=4, DD=5).\n",
                    val);
            std::fflush(stderr);
        }
    // Perform the processing
    DecoderPool decoderPool(*decoder, inputFileName, metaData, outputConfig, cadenceConfig, outputFileName, startFrame, length, maxThreads);
    if (!decoderPool.process()) {
        return -1;
    }

    return 0;
}
