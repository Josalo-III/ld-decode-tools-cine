/*
 * File:        processingpool.cpp
 * Module:      threading
 * Purpose:     Worker thread pool and field dispatch for VITS analysis
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2020-2025 Simon Inns
 */

#include "processingpool.h"

#include "tbc/logging.h"

ProcessingPool::ProcessingPool(QString _inputFilename,
                               QString _outputMetadataFilename,
                               qint32 _maxThreads,
                               LdDecodeMetaData& _ldDecodeMetaData)
    : inputFilename(_inputFilename),
      outputMetadataFilename(_outputMetadataFilename),
      maxThreads(_maxThreads),
      ldDecodeMetaData(_ldDecodeMetaData) {}

bool ProcessingPool::process() {
  // Get the metadata for the video parameters
  LdDecodeMetaData::VideoParameters videoParameters =
      ldDecodeMetaData.getVideoParameters();
  qInfo().noquote() << "Input TBC source dimensions are"
                    << videoParameters.fieldWidth << "x"
                    << videoParameters.fieldHeight;

  // Open the source video
  if (!sourceVideo.open(
          inputFilename,
          videoParameters.fieldWidth * videoParameters.fieldHeight,
          videoParameters.fieldWidth)) {
    // Could not open source video file
    qCritical() << "Source TBC file could not be opened";
    return false;
  }

  // Check TBC and metadata field numbers match
  if (sourceVideo.getNumberOfAvailableFields() !=
      ldDecodeMetaData.getNumberOfFields()) {
    qWarning() << "Warning: TBC file contains"
               << sourceVideo.getNumberOfAvailableFields()
               << "fields but the metadata indicates"
               << ldDecodeMetaData.getNumberOfFields()
               << "fields - some fields will be ignored";
  }

  // Show some information for the user
  qInfo() << "Using" << maxThreads << "threads to process"
          << ldDecodeMetaData.getNumberOfFields() << "fields";

  // Initialise processing state
  inputFieldNumber = 1;
  lastFieldNumber = ldDecodeMetaData.getNumberOfFields();
  totalTimer.start();

  // Start a vector of decoding threads to process the video
  QVector<QThread*> threads;
  threads.resize(maxThreads);
  for (qint32 i = 0; i < maxThreads; i++) {
    threads[i] = new VitsAnalyser(abort, *this);
    threads[i]->start(QThread::LowPriority);
  }

  // Wait for the workers to finish
  for (qint32 i = 0; i < maxThreads; i++) {
    threads[i]->wait();
    delete threads[i];
  }

  // Did any of the threads abort?
  if (abort) {
    sourceVideo.close();
    return false;
  }

  // Show the processing speed to the user
  double totalSecs = (static_cast<double>(totalTimer.elapsed()) / 1000.0);
  qInfo() << "VITS Processing complete -" << lastFieldNumber << "fields in"
          << totalSecs << "seconds (" << lastFieldNumber / totalSecs << "FPS )";

  // Write the metadata file
  qInfo() << "Writing metadata file...";
  ldDecodeMetaData.write(outputMetadataFilename);
  qInfo() << "VITS processing complete";

  // Close the source video
  sourceVideo.close();

  return true;
}

// Get the next field that needs processing from the input.
//
// Returns true if a field was returned, false if the end of the input has been
// reached.
bool ProcessingPool::getInputField(
    qint32& fieldNumber, SourceVideo::Data& fieldVideoData,
    LdDecodeMetaData::Field& fieldMetadata,
    LdDecodeMetaData::VideoParameters& videoParameters) {
  QMutexLocker locker(&inputMutex);

  if (inputFieldNumber > lastFieldNumber) {
    // No more input fields
    return false;
  }

  fieldNumber = inputFieldNumber;
  inputFieldNumber++;

  // Show what we are about to process
  // tbcDebugStream() << "Processing field number" << fieldNumber;

  // Fetch the input data
  fieldVideoData = sourceVideo.getVideoField(fieldNumber);
  fieldMetadata = ldDecodeMetaData.getField(fieldNumber);
  videoParameters = ldDecodeMetaData.getVideoParameters();

  return true;
}

// Put a decoded field into the output stream.
//
// Returns true on success, false on failure.
bool ProcessingPool::setOutputField(qint32 fieldNumber,
                                    LdDecodeMetaData::Field fieldMetadata) {
  QMutexLocker locker(&outputMutex);

  // Save the field data to the metadata (only VITS metrics metadata is
  // affected)
  ldDecodeMetaData.updateFieldVitsMetrics(fieldMetadata.vitsMetrics,
                                          fieldNumber);

  return true;
}
