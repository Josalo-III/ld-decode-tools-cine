/*
 * File:        main.cpp
 * Module:      cli
 * Purpose:     Command-line entry point for disc mapping and correction
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2019-2022 Simon Inns
 * SPDX-FileCopyrightText: 2025-2026 Joseph Burns
 */

#include <QCommandLineParser>
#include <QCoreApplication>
#include <QDebug>
#include <QFileInfo>
#include <QtGlobal>

#include "discmapper.h"
#include "tbc/logging.h"

int main(int argc, char* argv[]) {
  // Set 'binary mode' for stdin and stdout on Windows
  setBinaryMode();

  // Install the local debug message handler
  setDebug(true);
  qInstallMessageHandler(debugOutputHandler);

  QCoreApplication a(argc, argv);

  // Set application name and version
  QCoreApplication::setApplicationName("ld-discmap");
  QCoreApplication::setApplicationVersion(
      QString("ld-decode-tools - Branch: %1 / Commit: %2")
          .arg(APP_BRANCH, APP_COMMIT));
  QCoreApplication::setOrganizationDomain("domesday86.com");

  // Set up the command line parser
  QCommandLineParser parser;
  parser.setApplicationDescription(
      "ld-discmap - TBC and VBI alignment and correction\n"
      "\n"
      "(c)2019-2022 Simon Inns\n"
      "GPLv3 Open-Source - github: https://github.com/happycube/ld-decode");
  parser.addHelpOption();
  parser.addVersionOption();

  // Add the standard debug options --debug and --quiet
  addStandardDebugOptions(parser);

  // Option to reverse the field order (-r / --reverse)
  QCommandLineOption setReverseOption(
      QStringList() << "r" << "reverse",
      QCoreApplication::translate(
          "main",
          "Reverse the field order to second/first (default first/second)"));
  parser.addOption(setReverseOption);

  // Option to only perform mapping (without saving) (-m / --maponly)
  QCommandLineOption setMapOnlyOption(
      QStringList() << "m" << "maponly",
      QCoreApplication::translate(
          "main", "Only perform mapping - No output TBC file required"));
  parser.addOption(setMapOnlyOption);

  // Option to remove strict checking on pulldown frames (-s / --nostrict)
  QCommandLineOption setNoStrictOption(
      QStringList() << "s" << "nostrict",
      QCoreApplication::translate("main",
                                  "No strict checking on pulldown frames"));
  parser.addOption(setNoStrictOption);

  // Option to delete unmappable frames (-u / --delete-unmappable-frames)
  QCommandLineOption setDeleteUnmappableOption(
      QStringList() << "u" << "delete-unmappable-frames",
      QCoreApplication::translate("main", "Delete unmappable frames"));
  parser.addOption(setDeleteUnmappableOption);

  // Option to not process analogue audio (-n / --no-audio)
  QCommandLineOption setNoAudioOption(
      QStringList() << "n" << "no-audio",
      QCoreApplication::translate("main", "Do not process analogue audio"));
  parser.addOption(setNoAudioOption);

  // Positional argument to specify input TBC file
  parser.addPositionalArgument(
      "input", QCoreApplication::translate("main", "Specify input TBC file"));

  // Positional argument to specify output TBC file
  parser.addPositionalArgument(
      "output", QCoreApplication::translate("main", "Specify output TBC file"));

  // Process the command line options and arguments given by the user
  parser.process(a);

  // Standard logging options
  processStandardDebugOptions(parser);
  emitDeprecatedToolWarning();

  // Get the options from the parser
  bool reverse = parser.isSet(setReverseOption);
  bool mapOnly = parser.isSet(setMapOnlyOption);
  bool noStrict = parser.isSet(setNoStrictOption);
  bool deleteUnmappable = parser.isSet(setDeleteUnmappableOption);
  bool noAudio = parser.isSet(setNoAudioOption);

  // Process the command line options
  QString inputFilename;
  QString outputFilename;
  QStringList positionalArguments = parser.positionalArguments();
  if (!mapOnly) {
    // Require source and target filenames
    if (positionalArguments.count() == 2) {
      inputFilename = positionalArguments.at(0);
      outputFilename = positionalArguments.at(1);
    } else {
      qCritical("You must specify input and output TBC files");
      return -1;
    }

    if (inputFilename == outputFilename) {
      qCritical("Input and output TBC files cannot have the same file names");
      return -1;
    }
  } else {
    // Require only source filename
    if (positionalArguments.count() > 0) {
      inputFilename = positionalArguments.at(0);
      outputFilename = "";
    } else {
      qCritical("You must specify the input TBC file");
      return -1;
    }
  }

  // Put the input and output file names into QFileInfo for portability
  QFileInfo inputFileInfo(inputFilename);
  QFileInfo outputFileInfo(outputFilename);

  // Check that required input TBC file exists
  if (!inputFileInfo.exists()) {
    qCritical("The specified input file does not exist");
    return -1;
  }

  // Check that the required input TBC metadata file exists
  QFileInfo inputMetadataFileInfo(inputFileInfo.filePath() + ".db");
  if (!inputMetadataFileInfo.exists()) {
    qCritical("The specified input file metadata does not exist");
    return -1;
  }

  // Check that the required output TBC file isn't overwriting something
  if (!mapOnly) {
    if (outputFileInfo.exists()) {
      qCritical(
          "The specified output file already exists - please delete the "
          "existing file or use another output file name");
      return -1;
    }
  }

  // Perform disc mapping
  DiscMapper discMapper;
  if (!discMapper.process(inputFileInfo, inputMetadataFileInfo, outputFileInfo,
                          reverse, mapOnly, noStrict, deleteUnmappable,
                          noAudio)) {
    return 1;
  }

  // Quit with success
  return 0;
}
