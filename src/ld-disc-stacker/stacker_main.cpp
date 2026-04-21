/************************************************************************

    main.cpp

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

#include <QCoreApplication>
#include <QDebug>
#include <QtGlobal>
#include <QCommandLineParser>
#include <QThread>
#include <QFile>
#include <QFileInfo>

#include "tbc/logging.h"
#include "lddecodemetadata.h"
#include "sourcevideo.h"
#include "stackingpool.h"

int main(int argc, char *argv[])
{
    setBinaryMode();
    setDebug(true);
    qInstallMessageHandler(debugOutputHandler);

    QCoreApplication a(argc, argv);

    QCoreApplication::setApplicationName("ld-disc-stacker");
    QCoreApplication::setApplicationVersion(QString("ld-decode-tools - Branch: %1 / Commit: %2").arg(APP_BRANCH, APP_COMMIT));
    QCoreApplication::setOrganizationDomain("domesday86.com");

    QCommandLineParser parser;
    parser.setApplicationDescription(
                "ld-disc-stacker - Disc stacking for ld-decode\n"
                "\n"
                "(c)2020-2025 Simon Inns\n"
                "GPLv3 Open-Source - github: https://github.com/happycube/ld-decode\n"
                "For more info on stacking mode, use --help-mode");
    parser.addVersionOption();

    addStandardDebugOptions(parser);

    QCommandLineOption HelpOption(QStringList() << "?" << "h" << "help",
        QCoreApplication::translate("main", "Displays help on commandline options."));
    parser.addOption(HelpOption);

    QCommandLineOption helpModeOption(QStringList() << "help-mode",
        QCoreApplication::translate("main", "Show info about stacking mode"));
    parser.addOption(helpModeOption);

    QCommandLineOption verboseOption(QStringList() << "V" << "verbose",
        QCoreApplication::translate("main", "Show more info during stacking"));
    parser.addOption(verboseOption);

    QCommandLineOption inputMetadataOption(QStringList() << "input-metadata",
        QCoreApplication::translate("main", "Specify the input metadata file for the first input file (default input.db)"),
        QCoreApplication::translate("main", "filename"));
    parser.addOption(inputMetadataOption);

    QCommandLineOption outputMetadataOption(QStringList() << "output-metadata",
        QCoreApplication::translate("main", "Specify the output metadata file (default output.db)"),
        QCoreApplication::translate("main", "filename"));
    parser.addOption(outputMetadataOption);

    QCommandLineOption setReverseOption(QStringList() << "r" << "reverse",
        QCoreApplication::translate("main", "Reverse the field order to second/first (default first/second)"));
    parser.addOption(setReverseOption);

    QCommandLineOption threadsOption(QStringList() << "t" << "threads",
        QCoreApplication::translate("main", "Specify the number of concurrent threads (default is the number of logical CPUs)"),
        QCoreApplication::translate("main", "number"));
    parser.addOption(threadsOption);

    QCommandLineOption modeOption(QStringList() << "m" << "mode",
        QCoreApplication::translate("main",
            "Specify the stacking mode to use (default is Auto) "
            "-1 = auto / 0 = mean / 1 = median / 2 = smart mean / "
            "3 = smart neighbor / 4 = neighbor / 5 = local neighbor / "
            "6 = smart local neighbor / 7 = medoid"),
        QCoreApplication::translate("main", "number"));
    parser.addOption(modeOption);

    QCommandLineOption smartThresholdOption(QStringList() << "st" << "smart-threshold",
        QCoreApplication::translate("main",
            "Specify the range of value in 8 bit (0~128) for selecting sample where the distance to "
            "the median didn't exceed the selected value for applying mean (default is 15)"),
        QCoreApplication::translate("main", "number"));
    parser.addOption(smartThresholdOption);

    QCommandLineOption noDiffDodOption(QStringList() << "no-diffdod",
        QCoreApplication::translate("main", "Do not use differential dropout detection on low source pixels"));
    parser.addOption(noDiffDodOption);

    QCommandLineOption noMapOption(QStringList() << "no-map",
        QCoreApplication::translate("main", "Disable mapping requirement"));
    parser.addOption(noMapOption);

    QCommandLineOption passthroughOption(QStringList() << "passthrough",
        QCoreApplication::translate("main", "Pass-through dropouts present on every source"));
    parser.addOption(passthroughOption);

    QCommandLineOption integrityOption(QStringList() << "it" << "integrity",
        QCoreApplication::translate("main", "Check if frames contain skip or sample drop and discard bad source for specific frame"));
    parser.addOption(integrityOption);

    QCommandLineOption noSnrWeightOption(QStringList() << "no-snr-weight",
        QCoreApplication::translate("main",
            "Disable SNR-weighted bad-consensus override (default: enabled when VITS metrics are present)"));
    parser.addOption(noSnrWeightOption);

    QCommandLineOption snrWeightThresholdOption(QStringList() << "swt" << "snr-weight-threshold",
        QCoreApplication::translate("main",
            "Minimum divergence in 8-bit units (0~128) between the SNR-weighted mean and the "
            "count-consensus result before the bad-consensus override can fire (default: same as smart-threshold)"),
        QCoreApplication::translate("main", "number"));
    parser.addOption(snrWeightThresholdOption);

    parser.addPositionalArgument("inputs", QCoreApplication::translate("main",
        "Specify input TBC files (- as first source for piped input)"));
    parser.addPositionalArgument("output", QCoreApplication::translate("main",
        "Specify output TBC file (omit or - for piped output)"));

    parser.process(a);

    if (parser.isSet(helpModeOption)) {
        qInfo() << "ld-disc-stacker - Disc stacking for ld-decode\n";
        qInfo() << "(c)2020-2025 Simon Inns";
        qInfo() << "GPLv3 Open-Source - github: https://github.com/happycube/ld-decode";
        qInfo() << "For more info on stacking mode, use --help-mode\n";
        qInfo() << "Mode:\n";
        qInfo() << "(-1) auto              : select mode depending on the number of frames available (2f: mean, 3~4f: smart mean, 5+f: smart-neighbor)\n";
        qInfo() << " (0) mean              : average all samples not marked as dropouts using mean\n";
        qInfo() << " (1) median            : find the median from samples not marked as dropout\n";
        qInfo() << " (2) smart mean        : find the median from samples not marked as dropout then average all value within (median +/- smartThreshold) using mean\n";
        qInfo() << " (3) smart neighbor    : find the median for every surrounding pixel not marked as dropout then find the closest sample to the surrounding median for each neighbour";
        qInfo() << "                        then take the closest value to the median of the current sample from the different closest values found";
        qInfo() << "                        then average all values within (selectedSample +/- smartThreshold) using mean\n";
        qInfo() << " (4) neighbor          : find the median for every surrounding pixel not marked as dropout then find the closest sample to the surrounding median for each neighbour";
        qInfo() << "                        then take the closest value to the median of the current sample from the different closest values found, then average the selected sample with the median\n";
        qInfo() << " (5) local neighbor    : first apply medoid as an anchor (medoid +/- smartThreshold) to pre-filter outliers.";
        qInfo() << "                        Then for each available neighbour direction, find the inlier closest to that neighbour median estimate;";
        qInfo() << "                        from this set, select the one closest to the inlier median, then blend with the inlier median.";
        qInfo() << "                        If no inliers survive, returns the medoid.\n";
        qInfo() << " (6) smart local       : extends mode 5 with a medoid pass on the inlier set to produce a refined local anchor,";
        qInfo() << "     neighbor            then builds the neighbour anchor using mode 3 logic (smart mean within smartThreshold of neighbour-guided selection).";
        qInfo() << "                        The final result blends the local medoid and the neighbour smart mean equally.\n";
        qInfo() << " (7) medoid            : finds the sample that minimises total absolute distance to all other samples.";
        qInfo() << "                        Always returns a real source sample. Tiered fallback: N=1 passthrough, N=2 mean, N>=3 O(N^2) pairwise minimisation.";
        qInfo() << "                        No threshold parameter used. Makes no structural assumption about cluster shape or contiguity.";
        return 0;
    }

    if (parser.isSet(HelpOption))
        parser.showHelp();

    processStandardDebugOptions(parser);
    emitDeprecatedToolWarning();

    bool reverse        = parser.isSet(setReverseOption);
    bool verbose        = parser.isSet(verboseOption);
    bool noDiffDod      = parser.isSet(noDiffDodOption);
    bool noMap          = parser.isSet(noMapOption);
    bool passThrough    = parser.isSet(passthroughOption);
    bool integrityCheck = parser.isSet(integrityOption);
    bool useSnrWeight   = !parser.isSet(noSnrWeightOption);

    qint32 mode = -1;
    if (parser.isSet(modeOption)) {
        mode = parser.value(modeOption).toInt();
        if (mode > 7 || mode < 0) {
            qInfo() << "Specified mode (" << mode << ") is unknown - using auto mode instead";
            mode = -1;
        }
    }

    qint32 smartThreshold = 15 * 256;
    if (parser.isSet(smartThresholdOption)) {
        smartThreshold = parser.value(smartThresholdOption).toInt();
        if (smartThreshold > 128 || smartThreshold < 0) {
            qInfo() << "Specified threshold (" << smartThreshold << ") is out of range - using 15 instead";
            smartThreshold = 15 * 256;
        } else {
            smartThreshold *= 256;
        }
    }

    qint32 snrWeightThreshold = smartThreshold;
    if (parser.isSet(snrWeightThresholdOption)) {
        qint32 snrWtRaw = parser.value(snrWeightThresholdOption).toInt();
        if (snrWtRaw > 128 || snrWtRaw < 0)
            qInfo() << "Specified SNR weight threshold (" << snrWtRaw << ") is out of range (0~128) - using smart-threshold value instead";
        else
            snrWeightThreshold = snrWtRaw * 256;
    }

    qint32 maxThreads = QThread::idealThreadCount();
    if (parser.isSet(threadsOption)) {
        maxThreads = parser.value(threadsOption).toInt();
        if (maxThreads < 1) {
            qCritical("Specified number of threads must be greater than zero");
            return -1;
        }
    }

    QVector<QString> inputFilenames;
    QString outputFilename = "-";
    QStringList positionalArguments = parser.positionalArguments();
    qint32 totalNumberOfInputFiles = positionalArguments.count() - 1;

    if (totalNumberOfInputFiles > 32) {
        qCritical() << "A maximum of 32 input TBC files are supported";
        return -1;
    }

    if (positionalArguments.count() >= 3) {
        inputFilenames.resize(totalNumberOfInputFiles);
        for (qint32 i = 0; i < positionalArguments.count() - 1; i++)
            inputFilenames[i] = positionalArguments.at(i);
        if (positionalArguments.count() == 3)
            qInfo() << "Only 2 input sources specified - stack will be only based on averaging (3 or more sources are recommended)";
    } else {
        qCritical("You must specify at least 2 input and 1 output TBC file");
        return -1;
    }

    outputFilename = positionalArguments.at(positionalArguments.count() - 1);

    if (inputFilenames[0] == "-" && !parser.isSet(inputMetadataOption)) {
        qCritical("With piped input, you must also specify the input metadata file with --input-metadata");
        return -1;
    }

    if (outputFilename == "-" && !parser.isSet(outputMetadataOption)) {
        qCritical("With piped output, you must also specify the output metadata file with --output-metadata");
        return -1;
    }

    for (qint32 i = 0; i < totalNumberOfInputFiles; i++) {
        if (inputFilenames[i] == outputFilename) {
            qCritical("Input and output files cannot have the same filenames");
            return -1;
        }
    }

    for (qint32 i = 0; i < totalNumberOfInputFiles; i++) {
        for (qint32 j = 0; j < totalNumberOfInputFiles; j++) {
            if (i != j && inputFilenames[i] == inputFilenames[j]) {
                qCritical("Each input file should only be specified once - some filenames were repeated");
                return -1;
            }
        }
    }

    if (outputFilename != "-") {
        QFileInfo outputFileInfo(outputFilename);
        if (outputFileInfo.exists()) {
            qCritical("Specified output file already exists - will not overwrite");
            return -1;
        }
    }

    QString outputMetadataFilename = outputFilename + ".db";
    if (parser.isSet(outputMetadataOption))
        outputMetadataFilename = parser.value(outputMetadataOption);

    qInfo() << "Starting preparation for disc stacking processes...";
    tbcDebugStream() << "main(): Opening source video metadata files..";

    QVector<LdDecodeMetaData *> ldDecodeMetaData;
    ldDecodeMetaData.resize(totalNumberOfInputFiles);
    for (qint32 i = 0; i < totalNumberOfInputFiles; i++)
        ldDecodeMetaData[i] = new LdDecodeMetaData;

    for (qint32 i = 0; i < totalNumberOfInputFiles; i++) {
        QString metadataFilename = inputFilenames[i] + ".db";
        if (parser.isSet(inputMetadataOption) && i == 0) metadataFilename = parser.value(inputMetadataOption);
        qInfo().nospace().noquote() << "Reading input #" << i << " metadata from " << metadataFilename;
        if (!ldDecodeMetaData[i]->read(metadataFilename)) {
            qCritical() << "Unable to open TBC metadata file - cannot continue";
            return -1;
        }
    }

    if (reverse) {
        qInfo() << "Expected field order is reversed to second field/first field";
        for (qint32 i = 0; i < totalNumberOfInputFiles; i++)
            ldDecodeMetaData[i]->setIsFirstFieldFirst(false);
    }

    if (noDiffDod)   qInfo() << "Differential Dropout Detection is disabled";
    if (passThrough) qInfo() << "Passing through dropouts present on every input source";

    tbcDebugStream() << "Opening source video files...";
    QVector<SourceVideo *> sourceVideos;
    sourceVideos.resize(totalNumberOfInputFiles);
    for (qint32 i = 0; i < totalNumberOfInputFiles; i++)
        sourceVideos[i] = new SourceVideo;

    for (qint32 i = 0; i < totalNumberOfInputFiles; i++) {
        LdDecodeMetaData::VideoParameters videoParameters = ldDecodeMetaData[i]->getVideoParameters();
        qInfo().nospace() << "Opening input #" << i << ": " << videoParameters.fieldWidth << "x" << videoParameters.fieldHeight
                          << " - input filename is " << inputFilenames[i];

        if (!sourceVideos[i]->open(inputFilenames[i], videoParameters.fieldWidth * videoParameters.fieldHeight)) {
            qInfo() << "Unable to open input source" << i;
            qInfo() << "Please verify that the specified source video files exist with the correct file permissions";
            return 1;
        }

        if (sourceVideos[i]->getNumberOfAvailableFields() != ldDecodeMetaData[i]->getNumberOfFields()) {
            qInfo() << "Warning: TBC file contains" << sourceVideos[i]->getNumberOfAvailableFields()
                    << "fields but the metadata indicates" << ldDecodeMetaData[i]->getNumberOfFields()
                    << "fields - some fields will be ignored";
            qInfo() << "Update your copy of ld-decode and try again, this shouldn't happen unless the metadata has been corrupted";
        }

        if (!ldDecodeMetaData[i]->getFieldVbi(1).inUse) {
            qInfo() << "Source video" << i << "does not appear to have valid VBI data in the metadata.";
            qInfo() << "Please try running ld-process-vbi on the source video and then try again";
            return 1;
        }

        if (ldDecodeMetaData[0]->getVideoParameters().system != videoParameters.system) {
            qInfo() << "All additional input sources must have the same video system as the initial source!";
            qInfo() << "Initial source is" << ldDecodeMetaData[0]->getVideoSystemDescription()
                    << "and current source is" << ldDecodeMetaData[i]->getVideoSystemDescription();
            return 1;
        }

        if (!videoParameters.isMapped && !noMap) {
            qInfo() << "Source video" << i << "has not been mapped - run ld-discmap on all source videos and try again or use option \"no-map\"";
            qInfo() << "Disc stacking relies on accurate VBI frame numbering to match source frames together";
            return 1;
        } else if (noMap && !videoParameters.isMapped) {
            qInfo() << "Source video" << i << "has not been mapped - be careful using option no-map";
        }
    }

    qInfo() << "Initial source checks are ok and sources are loaded";
    qint32 result = 0;
    StackingPool stackingPool(outputFilename, outputMetadataFilename, maxThreads,
                               ldDecodeMetaData, sourceVideos,
                               mode, smartThreshold, reverse, noDiffDod, passThrough,
                               integrityCheck, verbose,
                               useSnrWeight, snrWeightThreshold);
    if (!stackingPool.process()) result = 1;

    for (qint32 i = 0; i < totalNumberOfInputFiles; i++) sourceVideos[i]->close();
    for (qint32 i = 0; i < totalNumberOfInputFiles; i++) delete ldDecodeMetaData[i];
    for (qint32 i = 0; i < totalNumberOfInputFiles; i++) delete sourceVideos[i];

    return result;
}
