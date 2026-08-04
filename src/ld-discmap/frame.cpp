/*
 * File:        frame.cpp
 * Module:      model
 * Purpose:     Per-frame descriptor: VBI number, quality and status flags
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2019-2022 Simon Inns
 */

#include "frame.h"

Frame::Frame(const qint32 seqFrameNumber, const qint32 vbiFrameNumber,
             const bool isPictureStop, const bool isPullDown,
             const bool isLeadInOrOut, const bool isMarkedForDeletion,
             const double frameQuality, const bool isPadded,
             const bool isClvOffset, const qint32 firstField,
             const qint32 secondField, const qint32 firstFieldPhase,
             const qint32 secondFieldPhase)
    : m_seqFrameNumber(seqFrameNumber),
      m_vbiFrameNumber(vbiFrameNumber),
      m_isPictureStop(isPictureStop),
      m_isPullDown(isPullDown),
      m_isLeadInOrOut(isLeadInOrOut),
      m_isMarkedForDeletion(isMarkedForDeletion),
      m_frameQuality(frameQuality),
      m_isPadded(isPadded),
      m_isClvOffset(isClvOffset),
      m_firstField(firstField),
      m_secondField(secondField),
      m_firstFieldPhase(firstFieldPhase),
      m_secondFieldPhase(secondFieldPhase) {}

// Custom streaming operator (for debug)
QDebug operator<<(QDebug dbg, const Frame& frame) {
  dbg.nospace().noquote() << "Frame("
                          << "seqFrameNumber " << frame.seqFrameNumber()
                          << ", vbiFrameNumber " << frame.vbiFrameNumber()
                          << ", isPictureStop " << frame.isPictureStop()
                          << ", isLeadInOrOut " << frame.isLeadInOrOut()
                          << ", isMarkedForDeletion "
                          << frame.isMarkedForDeletion() << ", frameQuality "
                          << frame.frameQuality() << ", isPadded "
                          << frame.isPadded() << ", isClvOffset "
                          << frame.isClvOffset() << ", firstField "
                          << frame.firstField() << ", secondField"
                          << frame.secondField() << ", firstFieldPhase "
                          << frame.firstFieldPhase() << ", secondFieldPhase "
                          << frame.secondFieldPhase() << ")";

  return dbg.maybeSpace();
}

// Get methods
qint32 Frame::seqFrameNumber() const { return m_seqFrameNumber; }
qint32 Frame::vbiFrameNumber() const { return m_vbiFrameNumber; }
bool Frame::isPictureStop() const { return m_isPictureStop; }
bool Frame::isPullDown() const { return m_isPullDown; }
bool Frame::isLeadInOrOut() const { return m_isLeadInOrOut; }
bool Frame::isMarkedForDeletion() const { return m_isMarkedForDeletion; }
double Frame::frameQuality() const { return m_frameQuality; }
bool Frame::isPadded() const { return m_isPadded; }
bool Frame::isClvOffset() const { return m_isClvOffset; }
qint32 Frame::firstField() const { return m_firstField; }
qint32 Frame::secondField() const { return m_secondField; }
qint32 Frame::firstFieldPhase() const { return m_firstFieldPhase; }
qint32 Frame::secondFieldPhase() const { return m_secondFieldPhase; }

// Set methods
void Frame::seqFrameNumber(qint32 value) { m_seqFrameNumber = value; }
void Frame::vbiFrameNumber(qint32 value) { m_vbiFrameNumber = value; }
void Frame::isPictureStop(bool value) { m_isPictureStop = value; }
void Frame::isPullDown(bool value) { m_isPullDown = value; }
void Frame::isLeadInOrOut(bool value) { m_isLeadInOrOut = value; }
void Frame::isMarkedForDeletion(bool value) { m_isMarkedForDeletion = value; }
void Frame::frameQuality(double value) { m_frameQuality = value; }
void Frame::isPadded(bool value) { m_isPadded = value; }
void Frame::isClvOffset(bool value) { m_isClvOffset = value; }
void Frame::firstField(qint32 value) { m_firstField = value; }
void Frame::secondField(qint32 value) { m_secondField = value; }
void Frame::firstFieldPhase(qint32 value) { m_firstFieldPhase = value; }
void Frame::secondFieldPhase(qint32 value) { m_secondFieldPhase = value; }

// Override less than operator for sorting.
// Primary: ascending VBI frame number.
// Secondary: non-pulldown before pulldown.
// Tertiary: ascending sequential frame number for determinism when both match.
bool Frame::operator<(const Frame& other) const {
  if (m_vbiFrameNumber != other.m_vbiFrameNumber)
    return m_vbiFrameNumber < other.m_vbiFrameNumber;
  if (m_isPullDown != other.m_isPullDown)
    return m_isPullDown < other.m_isPullDown;
  return m_seqFrameNumber < other.m_seqFrameNumber;
}
