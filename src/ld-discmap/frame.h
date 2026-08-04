/*
 * File:        frame.h
 * Module:      model
 * Purpose:     Per-frame descriptor: VBI number, quality and status flags
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2019-2022 Simon Inns
 */

#ifndef FRAME_H
#define FRAME_H

#include <QCoreApplication>
#include <QDebug>

class Frame {
 public:
  Frame(const qint32 seqFrameNumber = -1, const qint32 vbiFrameNumber = -1,
        const bool isPictureStop = false, const bool isPullDown = false,
        const bool isLeadInOrOut = false,
        const bool isMarkedForDeletion = false, const double frameQuality = 0,
        const bool isPadded = false, const bool isClvOffset = false,
        const qint32 firstField = -1, const qint32 secondField = -1,
        const qint32 firstFieldPhase = -1, const qint32 secondFieldPhase = -1);
  ~Frame() = default;
  Frame(const Frame&) = default;
  Frame& operator=(const Frame&) = default;

  // Get
  qint32 seqFrameNumber() const;
  qint32 vbiFrameNumber() const;
  bool isPictureStop() const;
  bool isPullDown() const;
  bool isLeadInOrOut() const;
  bool isMarkedForDeletion() const;
  double frameQuality() const;
  bool isPadded() const;
  bool isClvOffset() const;
  qint32 firstField() const;
  qint32 secondField() const;
  qint32 firstFieldPhase() const;
  qint32 secondFieldPhase() const;

  // Set
  void seqFrameNumber(qint32 value);
  void vbiFrameNumber(qint32 value);
  void isPictureStop(bool value);
  void isPullDown(bool value);
  void isLeadInOrOut(bool value);
  void isMarkedForDeletion(bool value);
  void frameQuality(double value);
  void isPadded(bool value);
  void isClvOffset(bool value);
  void firstField(qint32 value);
  void secondField(qint32 value);
  void firstFieldPhase(qint32 value);
  void secondFieldPhase(qint32 value);

  // Operators
  bool operator<(const Frame&) const;

 private:
  qint32 m_seqFrameNumber;
  qint32 m_vbiFrameNumber;
  bool m_isPictureStop;
  bool m_isPullDown;
  bool m_isLeadInOrOut;
  bool m_isMarkedForDeletion;
  double m_frameQuality;
  bool m_isPadded;
  bool m_isClvOffset;
  qint32 m_firstField;
  qint32 m_secondField;
  qint32 m_firstFieldPhase;
  qint32 m_secondFieldPhase;
};

// Custom streaming operator for debug
QDebug operator<<(QDebug dbg, const Frame& frame);

// Custom meta-type declaration
Q_DECLARE_METATYPE(Frame)

#endif  // FRAME_H
