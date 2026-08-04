/*
 * File:        main.cpp
 * Module:      cli
 * Purpose:     Command-line entry point for the EFM F2 section stacker
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2025 Simon Inns
 */

#include <QCommandLineParser>
#include <QCoreApplication>
#include <QDebug>
#include <QFileInfo>
#include <QLoggingCategory>
#include <QThread>
#include <QtGlobal>

#include "f2_stacker.h"
#include "tbc/logging.h"

int main(int argc, char* argv[]) {
  // Set 'binary mode' for stdin and stdout on Windows
  setBinaryMode();

  // Install the local debug message handler
  setDebug(true);
  qInstallMessageHandler(debugOutputHandler);

  // Enable Qt debug logging if debug mode is enabled (as Qt 5.2+ suppresses
  // qDebug by default) Not sure how wide this effect is but without it Fedora
  // 43 shows no qDebug output at all
  QLoggingCategory::setFilterRules("*.debug=true");

  QCoreApplication app(argc, argv);

  // Set application name and version
  QCoreApplication::setApplicationName("efm-stacker-f2");
  QCoreApplication::setApplicationVersion(
      QString("ld-decode-tools - Branch: %1 / Commit: %2")
          .arg(APP_BRANCH, APP_COMMIT));
  QCoreApplication::setOrganizationDomain("domesday86.com");

  // Set up the command line parser
  QCommandLineParser parser;
  parser.setApplicationDescription(
      "efm-stacker-f2 - EFM F2 Section stacker\n"
      "\n"
      "(c)2025 Simon Inns\n"
      "GPLv3 Open-Source - github: https://github.com/happycube/ld-decode");
  parser.addHelpOption();
  parser.addVersionOption();

  // Add the standard debug options --debug and --quiet
  addStandardDebugOptions(parser);

  // Positional arguments
  parser.addPositionalArgument(
      "inputs",
      QCoreApplication::translate("main", "Specify input F2 section files"));
  parser.addPositionalArgument(
      "output",
      QCoreApplication::translate("main", "Specify output F2 section file"));

  // Process the command line options and arguments given by the user
  parser.process(app);

  // Standard logging options
  processStandardDebugOptions(parser);
  emitDeprecatedToolWarning();

  // Get the filename arguments from the parser
  QVector<QString> inputFilenames;
  QString outputFilename;
  QStringList positionalArguments = parser.positionalArguments();
  qint32 totalNumberOfInputFiles = positionalArguments.count() - 1;

  // Ensure we don't have more than 32 sources
  if (totalNumberOfInputFiles > 32) {
    qCritical() << "A maximum of 32 input F2 section files are supported";
    return -1;
  }

  // Get the input F2 section sources
  if (positionalArguments.count() >= 3) {
    // Resize the input filenames vector according to the number of input files
    // supplied
    inputFilenames.resize(totalNumberOfInputFiles);

    for (qint32 i = 0; i < positionalArguments.count() - 1; i++) {
      inputFilenames[i] = positionalArguments.at(i);
    }

    // Warn if only 2 sources are used
    if (positionalArguments.count() == 3) {
      qInfo() << "Only 2 input sources specified (3 or more sources are "
                 "recommended)";
    }
  } else {
    // Quit with error
    qCritical(
        "You must specify at least 2 input F2 section files and 1 output F2 "
        "section file");
    return -1;
  }

  // Get the output F2 section file (should be the last argument of the command
  // line)
  outputFilename = positionalArguments.at(positionalArguments.count() - 1);

  // Check that none of the input filenames are used as the output file
  for (qint32 i = 0; i < totalNumberOfInputFiles; i++) {
    if (inputFilenames[i] == outputFilename) {
      // Quit with error
      qCritical("Input and output files cannot have the same filenames");
      return -1;
    }
  }

  // Check that none of the input filenames are repeated
  for (qint32 i = 0; i < totalNumberOfInputFiles; i++) {
    for (qint32 j = 0; j < totalNumberOfInputFiles; j++) {
      if (i != j) {
        if (inputFilenames[i] == inputFilenames[j]) {
          // Quit with error
          qCritical(
              "Each input file should only be specified once - some F2 section "
              "files were repeated");
          return -1;
        }
      }
    }
  }

  // Perform the processing
  qInfo() << "Beginning F2 Section stacking...";

  F2Stacker f2Stacker;
  if (!f2Stacker.process(inputFilenames, outputFilename)) {
    // Quit with error
    qCritical("F2 Section stacking failed");
    return -1;
  }

  // Quit with success
  return 0;
}
