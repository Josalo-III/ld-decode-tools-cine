// tools/ld-cinemap/cinemap.cpp
/******************************************************************************
 * cinemap.cpp
 * ld-cinemap — telecine cadence solver and edit detection
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 Joseph Burns
 *
 * This file is part of ld-decode-tools.
 ******************************************************************************/
#include "cinemap.h"
#include "cinedisc.h"

#include "lddecodemetadata.h"
#include "cadencedefs.h"
#include "fieldorder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <QElapsedTimer>
#include <limits>
#include <optional>
#include "tbc/logging.h"
#include <vector>
#include <unordered_map>


static inline int normalizePhase(long long val, int mod) {
    int res = val % mod;
    return (res < 0) ? res + mod : res;
}

const char* CineMap::phaseRunTypeName(PhaseRun::Type t)
{
    switch (t) {
        case PhaseRun::Type::Unknown:     return "unknown";
        case PhaseRun::Type::Pulldown32:  return "pulldown32";
        case PhaseRun::Type::Interlaced:  return "interlaced";
        case PhaseRun::Type::Progressive: return "progressive";
    }
    return "invalid";
}

const char* CineMap::twinRoleName(TwinACRole r)
{
    switch (r) {
        case TwinACRole::Unknown: return "unknown";
        case TwinACRole::AType:   return "AType";
        case TwinACRole::CType:   return "CType";
    }
    return "invalid";
}

static QString phaseArrayString(const std::array<double, 5>& values,
                                int bestPhase = -1)
{
    QString out;
    for (int p = 0; p < 5; ++p) {
        if (!out.isEmpty()) out += " ";
        out += QString("p%1=%2%3")
               .arg(p)
               .arg(values[p], 0, 'f', 4)
               .arg(p == bestPhase ? "*" : "");
    }
    return out;
}

static QString phaseIntArrayString(const std::array<int, 5>& values)
{
    QString out;
    for (int p = 0; p < 5; ++p) {
        if (!out.isEmpty()) out += " ";
        out += QString("p%1=%2").arg(p).arg(values[p]);
    }
    return out;
}

QString CineMap::phaseRunSummary(const PhaseRun& run)
{
    if (run.type == PhaseRun::Type::Pulldown32) {
        return QString("lock phase=%1 conf=%2")
            .arg(run.phaseOffset)
            .arg(run.confidence, 0, 'f', 3);
    }

    if (run.type != PhaseRun::Type::Unknown) {
        return QString(phaseRunTypeName(run.type));
    }

    QString out = QString("unknown");
    if (!run.reason.isEmpty()) {
        out += QString(" reason=%1").arg(run.reason);
    }
    return out;
}
CineMap::CineMap(CineDisc* disc, Policy policy)
    : m_disc(disc)
    , m_policy(policy)
    , m_md(disc ? &disc->getMetaData() : nullptr)
{
}

int CineMap::detectCadence(const QString& tbcFilePath, double threshold)
{
    Q_UNUSED(threshold);

    if (!m_disc || !m_md || m_disc->isDiscPal()) {
        qInfo() << "Skipping cadence detection (PAL or invalid)";
        return 0;
    }

    // ------------------------------------------------------------------
    // Reset solver-owned state: cadence + doplGang.
    // We treat every CineMap run as non-iterative and from-scratch.
    // ------------------------------------------------------------------
    const int totalFields = m_md->getNumberOfFields();
    for (int i = 1; i <= totalFields; ++i) {
        auto f = m_md->getField(i);

        // Cadence solution state
        f.cinemap.cadenceId = CADENCE_UNKNOWN;
        f.cinemap.cadenceIndexPresumed = false;
        f.cinemap.pulldownRole.clear();

        m_md->updateField(f, i);
    }
    m_doplGang.assign(totalFields + 1, std::nullopt);
    m_cadenceConfidence.assign(totalFields + 1, 0.0);

    // CAV fast-path
    if (m_disc->isDiscCav()) {
        qInfo() << "Running CAV cadence solver (picNo + dG based)";
        int cavLocked = solveCavDisc();
        if (cavLocked > 0) return cavLocked;
        qInfo() << "CAV fast-path found 0 groups/locks; falling back to CLV solver.";
    }

    // CLV path
    SourceVideo sv;
    if (!sv.open(tbcFilePath, m_disc->getVideoFieldLength())) {
        qWarning() << "Failed to open TBC file";
        return 0;
    }

    const int hardMaxField = computeHardMaxField();

    // 1. Initial segmentation by edit boundaries
    auto segments = identifySegments(hardMaxField);
    qInfo() << "Initially identified" << segments.size() << "segments";

    // 2. Build frame/field capture cache
    auto cache = buildCaptureCache(hardMaxField);

    // 3. Initial solve loop (per segment)
    std::vector<SegmentResult> solvedSegments;
    solvedSegments.reserve(segments.size());

    const size_t segCount = segments.size();
    for (size_t segIdx = 0; segIdx < segCount; ++segIdx) {
        const auto& [segStart, segEnd] = segments[segIdx];

        if (segIdx == 0 || (segIdx % 10) == 0 || segIdx + 1 == segCount) {
            const double pct = (segCount > 0)
                ? (100.0 * double(segIdx + 1) / double(segCount))
                : 100.0;
            qInfo().noquote()
                << QString("Solving segment %1/%2 (%3%) fields [%4..%5]")
                       .arg(segIdx + 1)
                       .arg(segCount)
                       .arg(pct, 0, 'f', 1)
                       .arg(segStart)
                       .arg(segEnd);
        }

        auto mixedness = computeFrameMixedness(sv, segStart, segEnd);

        if (m_policy == Policy::Cine) {
            solveSegmentCine(sv, segStart, segEnd, cache, mixedness);

            PhaseRun dummyRun;
            dummyRun.type        = PhaseRun::Type::Pulldown32;
            dummyRun.phaseOffset = 0;
            dummyRun.endField    = segEnd;
            dummyRun.confidence  = 0.8;

            solvedSegments.push_back({segStart, segEnd, dummyRun, mixedness});
        } else {
            PhaseRun run = solveSegment(sv, segStart, segEnd, cache, mixedness);

            if (run.type != PhaseRun::Type::Pulldown32) {
                // Failing to lock a phase is not evidence of video. Ask the twin
                // census to positively rule film out before writing either video
                // sentinel; where it cannot, leave the span unknown for the healer
                // to bridge, which is what "solve in the dark" is for.
                if (filmRuledOut(sv, segStart, segEnd)) {
                    classifyAsInterlaced(segStart, segEnd, mixedness);
                    classifyAsProgressive(segStart, segEnd, mixedness);
                }
            }

            solvedSegments.push_back({segStart, segEnd, run, mixedness});

            if (run.type == PhaseRun::Type::Pulldown32 && run.confidence > 0.5) {
                applyCadenceToSegment(segStart, segEnd,
                                      /*isLock=*/true,
                                      run.phaseOffset,
                                      /*fillCid=*/CADENCE_UNKNOWN,
                                      /*finalConf=*/run.confidence,
                                      cache);
            }
        }
    }

    // 4. Continuity & intra-segment healing pass
    qInfo() << "Running Continuity Healer...";
    int healedCount = healContinuity(sv, solvedSegments, cache);
    qInfo() << "Healer refined" << healedCount << "segments/spans.";

    // 5. Final paint using healed results
    int totalFieldsLocked = 0;
    for (const auto& seg : solvedSegments) {
        if (seg.run.type == PhaseRun::Type::Pulldown32 && seg.run.confidence > 0.5) {
            applyCadenceToSegment(seg.startField, seg.endField,
                                  /*isLock=*/true,
                                  seg.run.phaseOffset,
                                  /*fillCid=*/CADENCE_UNKNOWN,
                                  /*finalConf=*/seg.run.confidence,
                                  cache);
            totalFieldsLocked += (seg.endField - seg.startField + 1);
        }
    }

    sv.close();

    // 6. Post-processing
    detectAndEncodeInvertedCadenceRuns();
    assignPulldownRoles();

    // ld-cinemap does not write back to DiscMap frame flags; that is DiscMap's domain.

    // 7. Reconcile doplGang with final cadence geometry
    reconcileDoplGangWithCadence();

    qInfo() << "Solver locked" << totalFieldsLocked << "fields.";
    return totalFieldsLocked;
}

bool CineMap::isValidEvidenceField(int seq) const
{
    if (!m_md || seq < 1 || seq > m_md->getNumberOfFields()) return false;
    auto f = m_md->getField(seq);
    return !f.pad && f.fieldPhaseID >= 0;
}

bool CineMap::boundaryBetween(int a, int b) const
{
    if (! m_md) return true;

    int lo = std::min(a, b);
    int hi = std::max(a, b);

    for (int k = lo + 1; k <= hi; ++k) {
        if (m_md->getField(k).cinemap.isEditBoundary) return true;
    }
    return false;
}

std::vector<std::pair<int,int>> CineMap::identifySegments(int hardMaxField)
{
    std::vector<std::pair<int,int>> segs;
    if (!m_md || hardMaxField < 1) return segs;

    int start = 1;

    for (int i = 2; i <= hardMaxField; ++i) {
        if (m_md->getField(i).cinemap.isEditBoundary) {
            if (i - 1 >= start)
                segs.push_back({start, i - 1});
            start = i;
        }
    }

    if (start <= hardMaxField)
        segs.push_back({start, hardMaxField});

    return segs;
}

CineMap::SegmentCaptureCache CineMap::buildCaptureCache(int hardMaxField)
{
    SegmentCaptureCache cache;
    cache.totalFields = hardMaxField;
    cache.cap.resize(hardMaxField + 1);

    const int nFrames = m_disc->getNumberOfFrames();
    for (int fi = 0; fi < nFrames; ++fi) {
        if (m_disc->isPadded(fi)) continue;

        const int frame1 = fi + 1;
        const int f1 = m_disc->getFirstFieldNumber(frame1);
        const int f2 = m_disc->getSecondFieldNumber(frame1);

        if (f1 >= 1 && f1 <= hardMaxField) {
            cache.cap[f1] = {fi, f2, true};
        }
        if (f2 >= 1 && f2 <= hardMaxField) {
            cache.cap[f2] = {fi, f1, true};
        }
    }

    return cache;
}

int CineMap::computeHardMaxField()
{
    if (!m_disc || !m_md) return 0;

    int hardMax = m_md->getNumberOfFields();

    // Walk backwards to find the last frame that is neither padded nor
    // lead-in/out.  m_vbi.isExcluded() covers all three conditions.
    for (int i = m_disc->getNumberOfFrames() - 1; i >= 0; --i) {
        if (!m_vbi.isExcluded(i)) {
            hardMax = m_disc->getSecondFieldNumber(i + 1); // 1-based frameNumber
            break;
        }
    }

    return hardMax;
}

double CineMap::scoreSpecificPhase(const std::vector<FrameMixedness>& mixed, int phaseOffset, int startField, const SegmentCaptureCache& cache)
{
    if (mixed.empty()) return 0.0;
    
    int startFrameIdx = -1;
    if (cache.validSeq(startField)) startFrameIdx = cache.cap[startField].frameIndex;
    if (startFrameIdx < 0) return 0.0;

    double cleanSum = 0.0;
    double mixedSum = 0.0;
    int cleanCount = 0;
    int mixedCount = 0;

    for (const auto& m : mixed) {
        int relFrame = m.frameIndex - startFrameIdx;
        int pos = (relFrame + phaseOffset) % 5;
        if (pos < 0) pos += 5;

        // AB(1) and BC(2) should be mixed
        if (pos == 1 || pos == 2) {
            mixedSum += m.score;
            mixedCount++;
        } else {
            cleanSum += m.score;
            cleanCount++;
        }
    }

    double avgMixed = (mixedCount > 0) ? (mixedSum / mixedCount) : 0.0;
    double avgClean = (cleanCount > 0) ? (cleanSum / cleanCount) : 0.0;

    // Normalised contrast in [-1, 1]. Positive = good fit.
    //
    // This was a raw difference, which silently carried the units of whatever metric
    // fed it — so the healer's fixed acceptance bar meant one thing under notch and
    // something else entirely under lips. As a ratio it means the same thing under
    // any metric: the share of the total that separates the expected-mixed positions
    // from the expected-clean ones. A perfect 3:2 fit approaches 1, a wrong phase 0.
    const double total = avgMixed + avgClean;
    if (total <= 1e-9) return 0.0;
    return (avgMixed - avgClean) / total;
}

void CineMap::computeLumaLine_Bucket(const uint16_t* rawLine,
                                     std::vector<double>& lumaOut,
                                     int width) const
{
    if (width <= 0) return;
    if (static_cast<int>(lumaOut.size()) != width) lumaOut.resize(width);

    // Simple 2-tap averaging notch: at 4fsc, samples x and x+2 are 180°
    // apart in subcarrier phase, so averaging them cancels the subcarrier.
    for (int x = 0; x < width - 2; ++x) {
        lumaOut[x] = (static_cast<double>(rawLine[x]) +
                      static_cast<double>(rawLine[x + 2])) * 0.5;
    }

    // Handle the two tail samples that can't be averaged.
    if (width >= 2) {
        lumaOut[width - 2] = static_cast<double>(rawLine[width - 2]);
        lumaOut[width - 1] = static_cast<double>(rawLine[width - 1]);
    }
}

// CAV helpers

CineMap::CavTwinValidation
CineMap::validateCavWindowWithDG(SourceVideo& sv,
                                                   int f0, int f1, int f2, int f3, int f4)
{
    CavTwinValidation res;
    if (!m_disc || !m_md) return res;

    const int total = m_md->getNumberOfFields();
    FieldOrderPolicy fo;
    fo.reverse = m_disc->getReverseFieldOrder();

    auto getPair = [&](int frameIdx) -> std::pair<int,int> {
        return {m_disc->getFirstFieldNumber(frameIdx + 1),
                m_disc->getSecondFieldNumber(frameIdx + 1)};
    };

    auto F0 = getPair(f0);
    auto F1 = getPair(f1);
    auto F2 = getPair(f2);
    auto F3 = getPair(f3);
    auto F4 = getPair(f4);

    auto validPair = [&](std::pair<int,int> p) {
        return p.first >= 1 && p.second >= 1 &&
               p.first <= total && p.second <= total;
    };
    if (!validPair(F0) || !validPair(F1) || !validPair(F2) || !validPair(F3) || !validPair(F4)) {
        return res;
    }

    auto isTwin = [&](int a, int b) -> bool {
        if (a < 1 || b < 1 || a > total || b > total) return false;
        auto fa = m_md->getField(a);
        auto fb = m_md->getField(b);
        if (fa.pad || fb.pad) return false;
        if (fa.isFirstField != fb.isFirstField) return false;
        double conf = twinConfidence(sv, a, b);
        return conf > 0.6; // strict-ish gate
    };

    auto tryOrder = [&](bool swapped) -> bool {
        auto pf = [&](std::pair<int,int> p, int idx) { return swapped ? (idx ? p.first : p.second)
                                                                      : (idx ? p.second : p.first); };

        int A_def = pf(F0,0);
        int A_spa = pf(F1,0);
        int C_spa = pf(F2,1);
        int C_def = pf(F3,1);

        if (!isTwin(A_def, A_spa)) return false;
        if (!isTwin(C_spa, C_def)) return false;

        // Fill cadence IDs for AA AB BC CC DD
        for (int i=0;i<5;i++){res.cid[i][0]=res.cid[i][1]=CADENCE_UNKNOWN;}

        // Frame 0 (AA)
        res.cid[0][0] = 0; // def
        res.cid[0][1] = 1; // comp

        // Frame 1 (AB)
        res.cid[1][0] = 2; // Aspare
        res.cid[1][1] = 3; // B1

        // Frame 2 (BC)
        res.cid[2][0] = 4; // B2
        res.cid[2][1] = 5; // Cspare

        // Frame 3 (CC)
        res.cid[3][0] = 6; // Ccomp
        res.cid[3][1] = 7; // Cdef

        // Frame 4 (DD)
        res.cid[4][0] = 8; // D1
        res.cid[4][1] = 9; // D2

        if (swapped) {
            for (int i=0;i<5;i++) std::swap(res.cid[i][0], res.cid[i][1]);
        }

        return true;
    };

    if (tryOrder(false) || tryOrder(true)) {
        res.ok = true;
    }
    return res;
}

CineMap::CavGroupSignature
CineMap::analyseCavGroup(Cav5Group& g, SourceVideo& sv)
{
    CavGroupSignature sig;
    if (!m_disc || !m_md) return sig;

    const auto& vp = m_md->getVideoParameters();
    const int W = vp.fieldWidth;
    const int H = vp.fieldHeight;
    const int totalFields = m_md->getNumberOfFields();

    FieldOrderPolicy fo;
    fo.reverse = m_disc->getReverseFieldOrder();

    auto getFields = [&](int frameIdx) -> std::pair<int,int> {
        int a = m_disc->getFirstFieldNumber(frameIdx + 1);
        int b = m_disc->getSecondFieldNumber(frameIdx) + 1;
        if (a < 1 || b < 1 || a > totalFields || b > totalFields) {
            return std::make_pair(-1, -1);
        }
        return {a, b};
    };

    auto AA = getFields(g.f1);
    auto P1 = getFields(g.f2);
    auto P2 = getFields(g.f3);
    auto CC = getFields(g.f4);

    int A1 = AA.first;
    int A2 = AA.second;
    int P1a = P1.first;
    int P1b = P1.second;
    int P2a = P2.first;
    int P2b = P2.second;
    int C1  = CC.first;
    int C2  = CC.second;

    if (A1 < 1 || A2 < 1 || P1a < 1 || P1b < 1 ||
        P2a < 1 || P2b < 1 || C1  < 1 || C2  < 1) {
        return sig;
    }

    auto twinScore = [&](int x, int y) -> double {
        double diff = dgDiffIre(sv, x, y, W, H);
        double thr  = getAdaptiveTwinThreshold(x, y);
        if (!(thr > 0.0) || diff >= thr) return 0.0;
        return 1.0 - (diff / thr);
    };

    struct FieldSide {
        int seq = -1;
        double confToA = 0.0;
        double confToC = 0.0;
    };

    FieldSide pf1[2] = { {P1a}, {P1b} };
    FieldSide pf2[2] = { {P2a}, {P2b} };

    const int A1_seq = A1;
    const int A2_seq = A2;
    const int C1_seq = C1;
    const int C2_seq = C2;

    auto gather = [&](FieldSide& s) {
        if (s.seq < 1) return;
        s.confToA = std::max(twinScore(s.seq, A1_seq),
                             twinScore(s.seq, A2_seq));
        s.confToC = std::max(twinScore(s.seq, C1_seq),
                             twinScore(s.seq, C2_seq));
    };

    gather(pf1[0]);
    gather(pf1[1]);
    gather(pf2[0]);
    gather(pf2[1]);

    enum class Side { Unknown, A, C, Both };

    auto classify = [](const FieldSide& s, double minConf) -> Side {
        bool a = (s.confToA >= minConf);
        bool c = (s.confToC >= minConf);
        if (a && c) return Side::Both;
        if (a)      return Side::A;
        if (c)      return Side::C;
        return Side::Unknown;
    };

    constexpr double MIN_TWIN_CONF = 0.6;
    Side f1Side[2] = {
        classify(pf1[0], MIN_TWIN_CONF),
        classify(pf1[1], MIN_TWIN_CONF)
    };
    Side f2Side[2] = {
        classify(pf2[0], MIN_TWIN_CONF),
        classify(pf2[1], MIN_TWIN_CONF)
    };

    bool hasAInF1 = (f1Side[0] == Side::A || f1Side[1] == Side::A ||
                     f1Side[0] == Side::Both || f1Side[1] == Side::Both);
    bool hasCInF2 = (f2Side[0] == Side::C || f2Side[1] == Side::C ||
                     f2Side[0] == Side::Both || f2Side[1] == Side::Both);

    if (!hasAInF1 || !hasCInF2) {
        return sig; // weak geometry for this group
    }

    bool invertedDomain = false;
    auto dominantSide = [](const FieldSide sArr[2]) -> Side {
        double a = std::max(sArr[0].confToA, sArr[1].confToA);
        double c = std::max(sArr[0].confToC, sArr[1].confToC);
        if (a >= c + 0.1) return Side::A;
        if (c >= a + 0.1) return Side::C;
        return Side::Both;
    };

    {
        Side dom1 = dominantSide(pf1);
        Side dom2 = dominantSide(pf2);
        if (dom1 == Side::C && dom2 == Side::A) {
            invertedDomain = true;
        }
    }

    auto [tFirst, tSecond] = fo.temporalOrder(A1, A2);
    (void)tSecond;

    sig.valid = true;
    sig.invertedDomain = invertedDomain;
    sig.aaTemporalFirstSeq = tFirst;
    return sig;
}

void CineMap::collectCavTwinPairs(std::vector<Cav5Group>& groups,
                                  std::vector<std::pair<int,int>>& pairs)
{
    if (!m_disc) return;
    const int nFrames = m_disc->getNumberOfFrames();

    for (const auto& g : groups) {
        if (g.f1 < 0 || g.f4 < 0) continue;
        if (g.f1 >= nFrames || g.f2 >= nFrames ||
            g.f3 >= nFrames || g.f4 >= nFrames) continue;

        int a1 = m_disc->getFirstFieldNumber(g.f1 + 1);
        int a2 = m_disc->getSecondFieldNumber(g.f1 + 1);
        int ab1 = m_disc->getFirstFieldNumber(g.f2 + 1);
        int ab2 = m_disc->getSecondFieldNumber(g.f2 + 1);

        int bc1 = m_disc->getFirstFieldNumber(g.f3 + 1);
        int bc2 = m_disc->getSecondFieldNumber(g.f3 + 1);
        int c1  = m_disc->getFirstFieldNumber(g.f4 + 1);
        int c2  = m_disc->getSecondFieldNumber(g.f4 + 1);

        pairs.emplace_back(a1, ab1);
        pairs.emplace_back(a1, ab2);
        pairs.emplace_back(a2, ab1);
        pairs.emplace_back(a2, ab2);

        pairs.emplace_back(bc1, c1);
        pairs.emplace_back(bc1, c2);
        pairs.emplace_back(bc2, c1);
        pairs.emplace_back(bc2, c2);
    }
}

void CineMap::detectCavCadenceBreaks(std::vector<Cav5Group>& groups,
                                     SourceVideo& sv)
{
    if (!m_md || !m_disc) return;
    if (groups.size() < 2) return;

    std::vector<CavGroupSignature> sigs;
    sigs.reserve(groups.size());
    for (auto& g : groups)
        sigs.push_back(analyseCavGroup(g, sv));

    FieldOrderPolicy fo;
    fo.reverse = m_disc->getReverseFieldOrder();

    const int totalFields = m_md->getNumberOfFields();
    const int nFrames     = m_disc->getNumberOfFrames();

    auto markBoundaryAtFrame = [&](int frameIdx) {
        if (frameIdx < 0 || frameIdx >= nFrames) return;
        if (m_disc->isPadded(frameIdx)) return;

        const int frameNumber = frameIdx + 1;
        int f1 = m_disc->getFirstFieldNumber (frameNumber);
        int f2 = m_disc->getSecondFieldNumber(frameNumber);
        if (f1 < 1 || f2 < 1) return;

        auto [tFirst, tSecond] = fo.temporalOrder(f1, f2);
        (void)tSecond;

        if (tFirst < 1 || tFirst > totalFields) return;
        auto fld = m_md->getField(tFirst);
        if (!fld.cinemap.isEditBoundary) {
            fld.cinemap.isEditBoundary = true;
            m_md->updateField(fld, tFirst);
            qInfo() << "CAV: inserted cadence break at field" << tFirst
                    << "(frame" << frameIdx << ")";
        }
    };

    // VBI picture number for a frame index, or -1 if unavailable.
    auto pic = [&](int frameIdx) -> qint32 {
        if (frameIdx < 0 || frameIdx >= nFrames) return -1;
    return m_vbi.picNo(frameIdx);
    };

    for (size_t i = 1; i < groups.size(); ++i) {
        const auto& sp = sigs[i - 1];
        const auto& sc = sigs[i];
        if (!sp.valid || !sc.valid) continue;

        bool breakHere = false;

        // Domain flip is the strongest signal of a cadence regime change.
        if (sp.invertedDomain != sc.invertedDomain) {
            breakHere = true;
        } else {
            // AA temporal-first sequence continuity: a normal CAV increment
            // should advance by roughly 10 fields per group (5 frames × 2).
            // Very large jumps or regressions indicate a cut.
            const int prevAA = sp.aaTemporalFirstSeq;
            const int currAA = sc.aaTemporalFirstSeq;
            if (prevAA > 0 && currAA > 0) {
                const int delta = currAA - prevAA;
                if (delta < 4 || delta > 20)
                    breakHere = true;
            }
        }

        if (!breakHere) continue;

        const auto& gPrev = groups[i - 1];
        const auto& gCurr = groups[i];

        // Default: new regime starts at the AA frame of the current group.
        int candidateFrame = gCurr.f0;

        // PicNo refinement: if there is a large CC→D jump in the previous
        // group followed by a small D→AA step, the regime actually started
        // on D of the previous group.
        const qint32 picCCprev = pic(gPrev.f3); // CC frame (f3)
        const qint32 picDprev  = pic(gPrev.f4); // DD frame (f4)
        const qint32 picAA     = pic(gCurr.f0); // AA frame of next group

        if (picCCprev > 0 && picDprev > 0 && picAA > 0) {
            constexpr qint32 BIG_JUMP   = 5;
            constexpr qint32 SMALL_STEP = 2;

            if (std::abs(picDprev - picCCprev) > BIG_JUMP &&
                std::abs(picAA   - picDprev)   <= SMALL_STEP) {
                candidateFrame = gPrev.f4; // break lives on D of previous group
            }
        }

        markBoundaryAtFrame(candidateFrame);
    }

    m_disc->refreshFrameCache(); // no-op on CineDiscMeta; present for correctness
}

// CAV Solver

int CineMap::solveCavDisc()
{
    if (!m_disc || !m_md || !m_disc->isDiscCav()) return 0;

    const int nFrames = m_disc->getNumberOfFrames();

    // ------------------------------------------------------------------
    // 1. Derive phase from picNo using the cached VBI data.
    //    No VbiDecoder call here — m_vbi was populated by main.cpp.
    // ------------------------------------------------------------------
    static constexpr bool CANONICAL[5] = { true, false, false, true, true };

    std::array<int, 5> phaseVotes = {};
    int picNoHits    = 0;
    int windowsScored = 0;

    for (int f = 0; f + 4 < nFrames; ++f) {
        bool anyExcluded = false;
        for (int k = 0; k < 5 && !anyExcluded; ++k)
            if (m_vbi.isExcluded(f + k)) anyExcluded = true;
        if (anyExcluded) continue;

        for (int p = 0; p < 5; ++p) {
            bool match = true;
            for (int k = 0; k < 5 && match; ++k) {
                const bool hasNum   = (m_vbi.picNo(f + k) > 0);
                const bool expected = CANONICAL[(k + p) % 5];
                if (hasNum != expected) match = false;
            }
            if (match) phaseVotes[p]++;
        }
        windowsScored++;
    }

    // Count picNo hits for the threshold check.
    for (int fi = 0; fi < nFrames; ++fi)
        if (m_vbi.picNo(fi) > 0) picNoHits++;

    const bool hasPicNo = (picNoHits >= std::max(5, nFrames / 20));

    int bestPhase = 0, bestVotes = phaseVotes[0];
    for (int p = 1; p < 5; ++p) {
        if (phaseVotes[p] > bestVotes) { bestVotes = phaseVotes[p]; bestPhase = p; }
    }

    constexpr int MIN_SAMPLES = 10;
    const bool confident = hasPicNo &&
                           (bestVotes >= MIN_SAMPLES) &&
                           (bestVotes > windowsScored / 2);

    qInfo() << "CAV picNo hits" << picNoHits << "/ frames" << nFrames
            << "| best phase" << bestPhase << "with" << bestVotes
            << "votes of" << windowsScored << "| confident:" << confident;

    SourceVideo sv;
    if (!sv.open(m_disc->getTbcPath(), m_disc->getVideoFieldLength())) {
        qWarning() << "CAV solver: failed to open TBC file";
        return 0;
    }

    if (!hasPicNo) {
        qInfo() << "CAV: no picNo found — applying CLV-policy fallback";
        solveCavFallback(sv);
        sv.close();
        return 0;
    }

    if (!confident) {
        qInfo() << "CAV: picNo present but phase not confident"
                << "(only" << bestVotes << "agreeing windows)"
                << "— applying CLV-policy fallback";
        solveCavFallback(sv);
        sv.close();
        return 0;
    }

    // ------------------------------------------------------------------
    // 2. Confirm the picNo-derived phase by running pattern dG harvest
    //    across the whole side. picNo replaces mixedness on CAV, but we
    //    have to corroborate it. Pattern is the test we already plan to
    //    run for harvest, so it doubles as the gate: 3:2 telecine yields
    //    ~1 AA-twin per 5 frames, so observed coverage well below that
    //    means picNo's canonical {numbered,absent,absent,numbered,numbered}
    //    isn't actually riding on real twin pairs → untrust the side and
    //    fall through to the CLV solver, which runs mixedness + pattern
    //    + brute per segment.
    //
    //    Pattern's commits to doplGang here aren't a permanent scar:
    //    redo runs clear flags, and the fallback path re-elects per
    //    segment with the same evidence-additive ranking.
    // ------------------------------------------------------------------
    const int hardMax = computeHardMaxField();
    const SegmentCaptureCache cache = buildCaptureCache(hardMax);

    const int observedPairs = harvestTwinsByPattern(sv, 1, hardMax, bestPhase, cache);
    const int expectedPairs = nFrames / 5;
    const double coverage   = (expectedPairs > 0)
                              ? double(observedPairs) / double(expectedPairs)
                              : 0.0;
    constexpr double MIN_CAV_COVERAGE = 0.30;

    qInfo().nospace() << "CAV: pattern dG harvest at phase " << bestPhase
                      << " produced " << observedPairs << " pairs"
                      << " (expected ~" << expectedPairs
                      << ", coverage " << QString::number(coverage, 'f', 3) << ")";

    if (coverage < MIN_CAV_COVERAGE) {
        qInfo() << "CAV: pattern coverage below" << MIN_CAV_COVERAGE
                << "— picNo phase not corroborated by twins — applying CLV-policy fallback";
        solveCavFallback(sv);
        sv.close();
        return 0;
    }

    qInfo() << "CAV: picNo phase" << bestPhase
            << "corroborated by pattern dG — proceeding";

    applyCadenceToSegment(1, hardMax,
                          /*isLock=*/true,
                          bestPhase,
                          /*fillCid=*/CADENCE_UNKNOWN,
                          /*finalConf=*/0.85,
                          cache);

    detectAndEncodeInvertedCadenceRuns();

    // ------------------------------------------------------------------
    // 4. Group-level validation and final paint.
    // ------------------------------------------------------------------
    auto groups = identifyCav5Groups();
    detectCavCadenceBreaks(groups, sv);

    std::vector<std::pair<int,int>> cavPairs;
    collectCavTwinPairs(groups, cavPairs);
    std::vector<TwinEdge> cavEdges;
    buildTwinEdgesForPairs(sv, cavPairs, cavEdges, /*minConfidence=*/0.0);
    if (!cavEdges.empty())
        writeTwinEdgesToMetadata(sv, cavEdges);

    int validated = 0;
    for (const auto& g : groups) {
        CavTwinValidation v = validateCavWindowWithDG(sv, g.f0, g.f1, g.f2, g.f3, g.f4);
        if (!v.ok)
            v = validateCavWindowWithDG(sv, g.f0+1, g.f1+1, g.f2+1, g.f3+1, g.f4+1);
        if (!v.ok) continue;
        ++validated;

        auto assignFrame = [&](int frameIdx, const int cidPair[2]) {
            const int fn = frameIdx + 1;
            int fa = m_disc->getFirstFieldNumber(fn);
            int fb = m_disc->getSecondFieldNumber(fn);
            if (fa < 1 || fb < 1) return;
            auto fldA = m_md->getField(fa); fldA.cinemap.cadenceId = cidPair[0];
            auto fldB = m_md->getField(fb); fldB.cinemap.cadenceId = cidPair[1];
            m_md->updateField(fldA, fa);
            m_md->updateField(fldB, fb);
        };

        assignFrame(g.f0, v.cid[0]);
        assignFrame(g.f1, v.cid[1]);
        assignFrame(g.f2, v.cid[2]);
        assignFrame(g.f3, v.cid[3]);
        assignFrame(g.f4, v.cid[4]);
    }
    qInfo() << "CAV: validated windows painted" << validated;

    sv.close();

    int totalLocked = 0;
    const int totalFields = m_md->getNumberOfFields();
    for (int i = 1; i <= totalFields; ++i)
        if (cadenceKnown(m_md->getField(i).cinemap.cadenceId)) totalLocked++;

    qInfo() << "CAV solver completed, total cadenced fields:" << totalLocked;
    return totalLocked;
}

std::vector<CineMap::Cav5Group> CineMap::identifyCav5Groups()
{
    std::vector<Cav5Group> result;
    if (!m_disc) return result;

    const int n = m_disc->getNumberOfFrames();
    if (n < 5) return result;

    int paddedSkip = 0, patternMismatch = 0;

    for (int f = 0; f + 4 < n; ++f) {
        // Use isExcluded() which covers padded, lead-in, and lead-out.
        if (m_vbi.isExcluded(f)   || m_vbi.isExcluded(f+1) ||
            m_vbi.isExcluded(f+2) || m_vbi.isExcluded(f+3) ||
            m_vbi.isExcluded(f+4)) { paddedSkip++; continue; }

        const qint32 v0 = m_vbi.picNo(f);
        const qint32 v1 = m_vbi.picNo(f+1);
        const qint32 v2 = m_vbi.picNo(f+2);
        const qint32 v3 = m_vbi.picNo(f+3);
        const qint32 v4 = m_vbi.picNo(f+4);

        // Canonical telecine: numbered, absent, absent, numbered, numbered
        if (v0 > 0 && v1 == -1 && v2 == -1 && v3 > 0 && v4 > 0) {
            result.push_back({f, f+1, f+2, f+3, f+4});
        } else {
            patternMismatch++;
        }
    }

    qInfo() << "CAV identifyCav5Groups: windows checked"
            << (n >= 5 ? n - 4 : 0)
            << "paddedSkip" << paddedSkip
            << "patternMismatch" << patternMismatch
            << "accepted" << result.size();

    return result;
}

void CineMap::solveCavFallback(SourceVideo& sv)
{
    qInfo() << "CAV: running CLV-policy fallback (Tv policy)";

    const int hardMax = computeHardMaxField();
    const SegmentCaptureCache cache = buildCaptureCache(hardMax);

    // Treat the disc as a single segment and run the standard Tv solver.
    // detectCavCadenceBreaks will have already inserted any boundaries it
    // can find, so identifySegments will subdivide if needed.
    const auto segments = identifySegments(hardMax);

    for (const auto& [segStart, segEnd] : segments) {
        const auto mixed = computeFrameMixedness(sv, segStart, segEnd);

        PhaseRun run;
        if (m_policy == Policy::Cine) {
            // Cine path: use the change-point detector.
            solveSegmentCine(sv, segStart, segEnd, cache, mixed);
            continue; // solveSegmentCine paints directly
        } else {
            // Tv path (default for now).
            run = solveSegment(sv, segStart, segEnd, cache, mixed);
        }

        if (run.type == PhaseRun::Type::Pulldown32) {
            applyCadenceToSegment(segStart, segEnd,
                                  /*isLock=*/true,
                                  run.phaseOffset,
                                  /*fillCid=*/CADENCE_UNKNOWN,
                                  run.confidence,
                                  cache);
        } else if (run.type == PhaseRun::Type::Interlaced) {
            classifyAsInterlaced(segStart, segEnd, mixed);
        } else if (run.type == PhaseRun::Type::Progressive) {
            classifyAsProgressive(segStart, segEnd, mixed);
        }
        // PhaseRun::Type::Unknown: leave fields as CADENCE_UNKNOWN
    }

    m_disc->refreshFrameCache();
}

// 2 in 5 pulldown/mixed frame detection

// We try to provide the solver with the location of the AB and BC frames using two field comparisons
// We create a per-frame mixedness score from these and compare the scores,
// seeking a 2-high, 3-low pattern. Lips owns that score: it masks the image's own
// vertical detail per pixel, so it answers "does this frame comb" rather than "does
// this frame have vertical structure". Notch is retained for instruments only — it
// has no production caller, and measured, it was also the SLOWER of the two.

double CineMap::calculateNotchScore(SourceVideo& sv, int f1, int f2, int width, int height) const
{
    if (f1 < 1 || f2 < 1 || width <= 0 || height <= 0) return 0.0;

    auto d1 = sv.getVideoField(f1);
    auto d2 = sv.getVideoField(f2);
    // Relaxed size check: allow some headroom for short buffers
    if (d1.size() < (width * height * 2) / 2 || d2.size() < (width * height * 2) / 2) return 0.0;

    const uint16_t* p1 = reinterpret_cast<const uint16_t*>(d1.constData());
    const uint16_t* p2 = reinterpret_cast<const uint16_t*>(d2.constData());

    const auto& vp = m_md->getVideoParameters();
    const double black = (vp.black16bIre > 0) ? vp.black16bIre : 0.0;
    const double white = (vp.white16bIre > black) ? vp.white16bIre : 65535.0;
    const double scaleToIre = (white > black) ? (100.0 / (white - black)) : (100.0 / 65535.0);

    // Adaptive noise floor derived from field bPSNR.
    double psnr1 = m_md->getFieldVitsMetrics(f1).bPSNR;
    double psnr2 = m_md->getFieldVitsMetrics(f2).bPSNR;
    if (psnr1 <= 0.1) psnr1 = 38.0;
    if (psnr2 <= 0.1) psnr2 = 38.0;
    double avgPsnr = (psnr1 + psnr2) / 2.0;

    double noiseRmsRaw = 65535.0 / std::pow(10.0, avgPsnr / 20.0);
    double noiseRmsIre = noiseRmsRaw * scaleToIre;
    double adaptiveFloor = std::max(2.5, noiseRmsIre * 6.0 * 2.0); // ~2 of filtered noise

    // Global sensitivity: >1.0 lowers the floor (more sensitive), <1.0 raises it.
    if (m_notchSensitivity > 0.0) {
        adaptiveFloor /= m_notchSensitivity;
    }

    // --- ROI setup ---
    const int startX = static_cast<int>(width * 0.10);
    const int endX   = static_cast<int>(width * 0.90);
    const int startY = static_cast<int>(height * 0.10);
    const int endY   = static_cast<int>(height * 0.90);

    const int strideX = 2; // safe for 4fsc
    const int strideY = 2;

    // We want to give weight to *columns* of alternating comb, not just scattered pixels.
    // Strategy:
    //   - For each sampled x, count vertical comb hits.
    //   - A "column" is any x where hits >= MIN_COLUMN_HITS.
    //   - The frame score is proportional to number of such columns.
    std::vector<int> columnHits(width, 0);

    // Small margin above floor to ignore micronoise
    const double MIN_EXCESS_IRE = 1.0;

    for (int y = startY; y < endY; y += strideY) {
        if (y <= 0 || y >= height - 1) continue;

        const uint16_t* line_outer_up   = p2 + (y - 1) * width;
        const uint16_t* line_neigh_up   = p1 + (y)     * width;
        const uint16_t* line_center     = p2 + (y)     * width;
        const uint16_t* line_neigh_down = p1 + (y + 1) * width;
        const uint16_t* line_outer_down = p2 + (y + 1) * width;

        for (int x = startX; x < endX - 2; x += strideX) {
            // Average (x,x+2) to cancel chroma and make vertical filter cleaner
            double v_nu = (double(line_neigh_up[x])   + double(line_neigh_up[x+2])) * 0.5;
            double v_nd = (double(line_neigh_down[x]) + double(line_neigh_down[x+2])) * 0.5;
            double v_c  = (double(line_center[x])     + double(line_center[x+2])) * 0.5;
            double v_ou = (double(line_outer_up[x])   + double(line_outer_up[x+2])) * 0.5;
            double v_od = (double(line_outer_down[x]) + double(line_outer_down[x+2])) * 0.5;

            // 5tap vertical comb/notch: isolates temporal combing
            double rawVal = std::abs(4.0 * v_c - 3.0 * (v_nu + v_nd) + (v_ou + v_od));
            double valIre = rawVal * scaleToIre;

            if (valIre <= adaptiveFloor) continue;

            double excess = valIre - adaptiveFloor;
            if (excess >= MIN_EXCESS_IRE) {
                // Count a vertical "hit" for this x column
                columnHits[x]++;
            }
        }
    }

    // Aggregate percolumn hits into a score.
    // Columns with more vertical hits contribute more; isolated hits are downweighted.
    int activeColumns = 0;
    double columnScore = 0.0;

    const int MIN_COLUMN_HITS = std::max(3, (endY - startY) / 80); // require several line hits
    for (int x = startX; x < endX - 2; x += strideX) {
        int h = columnHits[x];
        if (h >= MIN_COLUMN_HITS) {
            activeColumns++;
            // Slightly favor long columns, but sublinearly so we don't overblow
            columnScore += std::sqrt(static_cast<double>(h));
        }
    }

    if (activeColumns == 0) return 0.0;

    // Scale into a "roughly 0..few" range. This is tunable; 0.02 is conservative.
    // With ~50100 active columns we want scores in the 13 range.
    const double COL_SCALE = 0.02;
    double score = columnScore * COL_SCALE;

    return score;
}

// This excludes the top and bottom quarters of title-safe and takes the middle half where a 
// speaking person's lips might be, and do a more expensive alternate method of comparing fields
double CineMap::calculateLipsScore(SourceVideo& sv, int f1, int f2, int width, int height) const
{
    if (f1 < 1 || f2 < 1 || width <= 0 || height <= 0) return 0.0;

    auto d1 = sv.getVideoField(f1);
    auto d2 = sv.getVideoField(f2);
    if (d1.size() < width * height || d2.size() < width * height) return 0.0;

    const uint16_t* p1 = reinterpret_cast<const uint16_t*>(d1.constData());
    const uint16_t* p2 = reinterpret_cast<const uint16_t*>(d2.constData());

    const auto& vp = m_md->getVideoParameters();
    const double black = (vp.black16bIre > 0) ? vp.black16bIre : 0.0;
    const double white = (vp.white16bIre > black) ? vp.white16bIre : 65535.0;
    const double scaleToIre = (white > black) ? (100.0 / (white - black)) : (100.0 / 65535.0);

    // Lips ROI (Center-Lower)
    const int startX = static_cast<int>(width * 0.20);
    const int endX   = static_cast<int>(width * 0.80);
    const int startY = static_cast<int>(height * 0.40);
    const int endY   = static_cast<int>(height * 0.80);

    double totalLipsEnergy = 0.0;
    
    // Reuse bucket buffers
    std::vector<double> Y1_m1(width), Y1_p1(width);
    std::vector<double> Y2_0(width);

    // Stride 1 for maximum detail resolution on lips
    for (int y = startY; y < endY; y += 2) {
        if (y < 1 || y >= height - 1) continue;

        // Get Demodulated Luma (removes 4fsc pattern)
        computeLumaLine_Bucket(p1 + (y - 1) * width, Y1_m1, width); // Field 1 (Upper)
        computeLumaLine_Bucket(p1 + (y + 1) * width, Y1_p1, width); // Field 1 (Lower)
        computeLumaLine_Bucket(p2 + y * width,       Y2_0,  width); // Field 2 (Center)

        for (int x = startX; x < endX; x++) {
            
            // 1. Calculate "Spatial Mask" (Complexity of Field 1)
            // If F1 has a vertical edge here, prediction is hard.
            double spatialDetail = std::abs(Y1_m1[x] - Y1_p1[x]) * scaleToIre;

            // 2. Calculate "Temporal Error" (Combing)
            // Predict F2 as average of F1 lines
            double pred = (Y1_m1[x] + Y1_p1[x]) * 0.5;
            double temporalDiff = std::abs(Y2_0[x] - pred) * scaleToIre;

            // 3. The Filter (Difference - Mask)
            // We only care if the Temporal Error is significantly larger than Spatial Detail.
            // This allows us to detect motion *on* edges, provided the motion artifact 
            // is stronger than the static edge gradient.
            
            double noiseFloor = 2.0; // Basic noise floor for demodulated luma
            
            // If temporal diff is huge (combing) and spatial is small (flat area), Metric is huge.
            // If temporal diff is moderate and spatial is moderate (static edge), Metric is ~0.
            double metric = temporalDiff - std::max(spatialDetail, noiseFloor);

            if (metric > 0.0) {
                // Square it to emphasize the deviation
                totalLipsEnergy += (metric * metric);
            }
        }
    }

    // Normalize similar to Notch
    return totalLipsEnergy * 0.00002; 
}

std::vector<CineMap::FrameMixedness>
CineMap::computeFrameMixedness(SourceVideo& sv, int segStart, int segEnd)
{
    std::vector<FrameMixedness> results;
    if (!m_md || !m_disc) return results;
    const auto& vp = m_md->getVideoParameters();

    int startFrame = frameIndexForField(segStart);
    int endFrame   = frameIndexForField(segEnd);
    if (startFrame < 0 || endFrame < 0) return results;

    tbcDebugStream() << "  Computing mixedness (Lips) for frames"
             << startFrame << "-" << endFrame;

    // Lips owns mixedness.
    //
    // It masks the image's own vertical detail per pixel — metric = temporalDiff -
    // max(spatialDetail, noiseFloor), accumulated only where positive — so it answers
    // "does this frame comb" rather than "does this frame have vertical structure".
    // A detailed progressive frame reads ~zero here where notch read whatever its
    // edges amounted to, which is what used to paint such shots as 59.94i.
    //
    // Notch previously ran first as a cheap prefilter, consulting lips only in a
    // middle band and skipping it entirely once notch exceeded 0.10 — i.e. bypassing
    // the detail mask in exactly the case where detail was the likely cause of the
    // large reading. Measured, lips is also the CHEAPER operator (0.69-0.84x notch
    // over three discs), so the tiering cost accuracy and bought nothing.
    for (int fi = startFrame; fi <= endFrame; ++fi) {
        if (m_disc->isPadded(fi)) continue;

        int f1 = m_disc->getFirstFieldNumber(fi + 1);
        int f2 = m_disc->getSecondFieldNumber(fi + 1);
        if (f1 < 1 || f2 < 1) continue;

        results.push_back({fi,
            calculateLipsScore(sv, f1, f2, vp.fieldWidth, vp.fieldHeight)});
    }

    return results;
}

// doplGang twin detection & helpers
// We identify duplicate fields by comparing the drop in difference (twins will subtract to near zero)
// We try using identified pulldown frames to scope the checks by pattern, else we harvest wholesale
// This is the wholesale operation
std::vector<CineMap::TwinEdge> CineMap::harvestTwinEdges(SourceVideo& sv, int segStart, int segEnd, int maxDist)
{
    std::vector<TwinEdge> edges;
    if (!m_md || !m_disc) return edges;
    const auto& vp = m_md->getVideoParameters();

    tbcDebugStream() << "  Harvesting twin edges (Dip-Based), maxDist =" << maxDist << "Range:" << segStart << "-" << segEnd;

    int pairsEvaluated = 0;
    int belowSilence = 0;
    int ratioPassCount = 0;
    int threshPassCount = 0;
    TwinConfDetail bestDetail;
    int bestA = -1, bestB = -1;
    TwinConfDetail bestRatioDetail;
    int bestRatioA = -1, bestRatioB = -1;
    double sumRatio = 0.0;
    int ratioCount = 0;

    for (int a = segStart; a <= segEnd; ++a) {
        if (!isValidEvidenceField(a)) continue;

        auto fa = m_md->getField(a);

        for (int d = 2; d <= maxDist; d += 2) {
            int b = a + d;
            if (b > segEnd) break;
            if (boundaryBetween(a, b)) continue;
            if (!isValidEvidenceField(b)) continue;

            auto fb = m_md->getField(b);
            if (fa.isFirstField != fb.isFirstField) continue;

            pairsEvaluated++;

            TwinConfDetail det;
            double confidence = twinConfidence(sv, a, b, det);

            // Sparse-scorer admit: any pair quieter than its neighbors is a clue.
            // Downstream election (geometry / cleanup) decides which clues add up.
            if (confidence > 0.0) {
                double diff = dgDiffIre(sv, a, b, vp.fieldWidth, vp.fieldHeight);
                edges.push_back({a, b, diff, confidence});
            }

            if (det.silenceMatch) belowSilence++;
            if (det.ratio < 0.85) ratioPassCount++;
            if (det.diffIn < det.threshAbs) threshPassCount++;

            if (det.ratio < 900.0) {
                sumRatio += det.ratio;
                ratioCount++;
                if (det.ratio < bestRatioDetail.ratio) {
                    bestRatioDetail = det;
                    bestRatioA = a; bestRatioB = b;
                }
            }

            if (confidence > bestDetail.confidence) {
                bestDetail = det;
                bestA = a; bestB = b;
            }
        }
    }

    tbcDebugStream() << "  Found" << edges.size() << "twin edges in brute force pass.";

    if (m_decisionTraceEnabled) {
        double avgRatio = (ratioCount > 0) ? (sumRatio / ratioCount) : 999.0;
        qInfo().noquote()
            << QString("CineMap decision: TWIN_HARVEST fields [%1..%2] pairs=%3 accepted=%4 belowSilence=%5 ratioPass=%6 threshPass=%7 bestConf=%8 avgRatio=%9 bestRatio=%10")
               .arg(segStart).arg(segEnd)
               .arg(pairsEvaluated)
               .arg(edges.size())
               .arg(belowSilence)
               .arg(ratioPassCount)
               .arg(threshPassCount)
               .arg(bestDetail.confidence, 0, 'f', 4)
               .arg(avgRatio, 0, 'f', 4)
               .arg(bestRatioDetail.ratio < 900.0 ? bestRatioDetail.ratio : 0.0, 0, 'f', 4);
        if (bestA >= 0) {
            qInfo().noquote()
                << QString("CineMap decision: TWIN_HARVEST_BEST fields [%1..%2] pair=(%3,%4) diffIn=%5 neighbor=%6 ratio=%7 conf=%8 ratioScore=%9 threshScore=%10 threshAbs=%11")
                   .arg(segStart).arg(segEnd)
                   .arg(bestA).arg(bestB)
                   .arg(bestDetail.diffIn, 0, 'f', 4)
                   .arg(bestDetail.neighborActivity < 900.0 ? bestDetail.neighborActivity : -1.0, 0, 'f', 4)
                   .arg(bestDetail.ratio < 900.0 ? bestDetail.ratio : -1.0, 0, 'f', 4)
                   .arg(bestDetail.confidence, 0, 'f', 4)
                   .arg(bestDetail.ratioScore, 0, 'f', 4)
                   .arg(bestDetail.threshScore, 0, 'f', 4)
                   .arg(bestDetail.threshAbs, 0, 'f', 4);
        }
        if (bestRatioA >= 0 && bestRatioA != bestA) {
            qInfo().noquote()
                << QString("CineMap decision: TWIN_HARVEST_BEST_RATIO fields [%1..%2] pair=(%3,%4) diffIn=%5 neighbor=%6 ratio=%7 conf=%8 ratioScore=%9 threshScore=%10")
                   .arg(segStart).arg(segEnd)
                   .arg(bestRatioA).arg(bestRatioB)
                   .arg(bestRatioDetail.diffIn, 0, 'f', 4)
                   .arg(bestRatioDetail.neighborActivity, 0, 'f', 4)
                   .arg(bestRatioDetail.ratio, 0, 'f', 4)
                   .arg(bestRatioDetail.confidence, 0, 'f', 4)
                   .arg(bestRatioDetail.ratioScore, 0, 'f', 4)
                   .arg(bestRatioDetail.threshScore, 0, 'f', 4);
        }
    }

    // Merge-write doplGang with conflict resolution vs existing JSON
    writeTwinEdgesToMetadata(sv, edges);

    return edges;
}

// This harvest uses mixedness to bracket its checks around the pulldown frames instead of brute forcing
int CineMap::harvestTwinsByPattern(SourceVideo& sv, int segStart, int segEnd,
                                   int phaseOffset, const SegmentCaptureCache& cache)
{
    int pairsFound = 0;
    if (!m_md || !m_disc) return 0;
    if (segStart >= segEnd) return 0;

    // Stable anchor frame index from cache
    int startFrameIdx = -1;
    for (int s = segStart; s <= segEnd; ++s) {
        if (cache.validSeq(s)) { startFrameIdx = cache.cap[s].frameIndex; break; }
    }
    if (startFrameIdx < 0) return 0;

    // Frame scan bounds from cache (single source of truth once present)
    int fStart = -1, fEnd = -1;
    for (int s = segStart; s <= segEnd; ++s) {
        if (cache.validSeq(s)) { fStart = cache.cap[s].frameIndex; break; }
    }
    for (int s = segEnd; s >= segStart; --s) {
        if (cache.validSeq(s)) { fEnd = cache.cap[s].frameIndex; break; }
    }
    if (fStart < 0 || fEnd < 0) return 0;

    // Pattern-credibility threshold: at predicted twin sites we require a
    // noticeable preponderance (relative-quietness ratio ≲ 0.85). This is
    // NOT a back-off gate — pattern's failure to admit enough pairs at the
    // segment level escalates to brute-force harvest downstream. Threshold
    // here exists so the count of admitted pairs is a meaningful signal
    // (the CAV coverage gate, for one, reads it).
    constexpr double MIN_PATTERN_CONF = 0.15;
    constexpr double HYSTERESIS       = 0.10;

    auto tryCommitPatternPair = [&](int a, int b) {
        if (a < 1 || b < 1) return;
        if (a == b) return;

        // stay inside the segment
        if (a < segStart || a > segEnd || b < segStart || b > segEnd) return;

        // Pattern is the gate
        const double conf = twinConfidence(sv, a, b);
        if (conf <= MIN_PATTERN_CONF) return;

        // Commit directly with conflict resolution; cache is provided for any strict geometry checks
        // implemented inside tryCommitReciprocalGang.
        if (tryCommitReciprocalGang(sv, a, b, HYSTERESIS, &cache)) {
            pairsFound++;
        }
    };

    // Pattern-permitted twin sites:
    //  pos 0 (AA): first field twin across next frame (A def/spare)
    //  pos 2 (BC): second field twin across next frame (C spare/def)
    for (int fi = fStart; fi <= fEnd; ++fi) {
        if (m_disc->isPadded(fi)) continue;

        const int relFrame = fi - startFrameIdx;
        int pos = (relFrame + phaseOffset) % 5;
        if (pos < 0) pos += 5;

        if (pos != 0 && pos != 2) continue;

        // Need mate frame
        if (fi + 1 > fEnd || m_disc->isPadded(fi + 1)) continue;

        if (pos == 0) {
            const int a = m_disc->getFirstFieldNumber(fi + 1);
            const int b = m_disc->getFirstFieldNumber(fi + 2);
            tryCommitPatternPair(a, b);
        } else { // pos == 2
            const int a = m_disc->getSecondFieldNumber(fi + 1);
            const int b = m_disc->getSecondFieldNumber(fi + 2);
            tryCommitPatternPair(a, b);
        }
    }

    return pairsFound;
}

void CineMap::collectClvTwinPairsFromMixedness(
        const std::vector<FrameMixedness>& mixed,
        const SegmentCaptureCache& cache,
        std::vector<std::pair<int,int>>& pairs) const
{
    if (!m_disc || !m_md) return;
    if (mixed.size() < 2) return;

    // "Clearly mixed" frames. Same per-frame comb question the classifiers ask, so
    // it reads the same constant rather than carrying its own notch-scaled one.
    constexpr double THRESH_MIXED = LIPS_COMB;

    const int nFrames = m_disc->getNumberOfFrames();

    // Map frameIndex -> index in mixed vector for convenience if needed,
    // but here we just use mixed[i].frameIndex directly.
    for (size_t i = 0; i < mixed.size(); ++i) {
        if (mixed[i].score < THRESH_MIXED) continue;
        int f1 = mixed[i].frameIndex;
        int f0 = f1 - 1;
        int f2 = f1 + 1;
        int f3 = f1 + 2;
        if (f0 < 0 || f3 >= nFrames) continue;
        if (m_disc->isPadded(f0) || m_disc->isPadded(f1) ||
            m_disc->isPadded(f2) || m_disc->isPadded(f3)) continue;

        // AA vs AB: f0 vs f1
        int a1 = m_disc->getFirstFieldNumber(f0 + 1);
        int a2 = m_disc->getSecondFieldNumber(f0 + 1);
        int ab1 = m_disc->getFirstFieldNumber(f1 + 1);
        int ab2 = m_disc->getSecondFieldNumber(f1 + 1);

        // BC vs CC: f2 vs f3
        int bc1 = m_disc->getFirstFieldNumber(f2 + 1);
        int bc2 = m_disc->getSecondFieldNumber(f2 + 1);
        int c1  = m_disc->getFirstFieldNumber(f3 + 1);
        int c2  = m_disc->getSecondFieldNumber(f3 + 1);

        pairs.emplace_back(a1, ab1);
        pairs.emplace_back(a1, ab2);
        pairs.emplace_back(a2, ab1);
        pairs.emplace_back(a2, ab2);

        pairs.emplace_back(bc1, c1);
        pairs.emplace_back(bc1, c2);
        pairs.emplace_back(bc2, c1);
        pairs.emplace_back(bc2, c2);
    }
}

void CineMap::harvestClvTwinsForSegment(SourceVideo& sv,
                                        int segStart,
                                        int segEnd,
                                        const SegmentCaptureCache& cache,
                                        const std::vector<FrameMixedness>& mixedness)
{
    if (!m_md || !m_disc) return;
    if (segStart >= segEnd) return;
    if (mixedness.empty()) return;

    // 1) Generate candidate twin pairs from the AB/BC mixedness pattern
    std::vector<std::pair<int,int>> pairs;
    collectClvTwinPairsFromMixedness(mixedness, cache, pairs);
    if (pairs.empty()) return;

    // 2) Score them. Sparse scorer: any pair quieter than its neighbors is a clue.
    //    tryCommitReciprocalGang still gates writes with its own absolute sanity check.
    std::vector<TwinEdge> edges;
    constexpr double MIN_CLV_CONF = 0.0;
    buildTwinEdgesForPairs(sv, pairs, edges, MIN_CLV_CONF);
    if (edges.empty()) return;

    // 3) Write doplGang to metadata with conflict resolution
    writeTwinEdgesToMetadata(sv, edges);
}

double CineMap::dgDiffIre(SourceVideo& sv, int seqA, int seqB, int width, int height)
{
    DiffCacheKey key{std::min(seqA, seqB), std::max(seqA, seqB)};
    auto it = m_diffCache.find(key);
    if (it != m_diffCache.end()) return it->second;

    auto dA = sv.getVideoField(seqA);
    auto dB = sv.getVideoField(seqB);

    // Safety check for empty/short buffers
    size_t required = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(uint16_t);
    if (dA.size() < (int)required || dB.size() < (int)required) return 999.0;

    const uint16_t* pA = reinterpret_cast<const uint16_t*>(dA.constData());
    const uint16_t* pB = reinterpret_cast<const uint16_t*>(dB.constData());

    // ROI: center region (50% of image)
    int yStart = height / 4;
    int yEnd   = height * 3 / 4;
    int xStart = width / 4;
    int xEnd   = width * 3 / 4;

    double diffSum = 0.0;
    int count = 0;

    const auto& vp = m_md->getVideoParameters();
    const double black = (vp.black16bIre > 0) ? vp.black16bIre : 0.0;
    const double white = (vp.white16bIre > black) ? vp.white16bIre : 65535.0;
    const double scaleToIre = (white > black) ? (100.0 / (white - black)) : (100.0 / 65535.0);

    // Use a small stride for performance, but 2 is better than 4 for accuracy
    std::vector<double> lumaA, lumaB;
    
    // We compute Luma per line to strip subcarrier
    for (int y = yStart; y < yEnd; y += 4) {
        computeLumaLine_Bucket(pA + y * width, lumaA, width);
        computeLumaLine_Bucket(pB + y * width, lumaB, width);

        for (int x = xStart; x < xEnd; x += 2) {
            double diff = std::abs(lumaA[x] - lumaB[x]);
            diffSum += diff;
            count++;
        }
    }

    double result = (count > 0) ? ((diffSum / count) * scaleToIre) : 999.0;
    m_diffCache[key] = result;
    return result;
}

const CineMap::TwinDemod& CineMap::demodTwinCached(SourceVideo& sv, int seq1, int seq2, int width, int height)
{
    static const TwinDemod invalid{};

    if (!m_md) return invalid;
    if (seq1 < 1 || seq2 < 1) return invalid;

    TwinDemodCacheKey key{ std::min(seq1, seq2), std::max(seq1, seq2) };

    auto it = m_twinDemodCache.find(key);
    if (it != m_twinDemodCache.end()) return it->second;

    TwinDemod d = calculateDemodulatedFieldDiff(sv, key.a, key.b, width, height);
    return m_twinDemodCache.emplace(key, d).first->second;
}

double CineMap::demodTwinDiffCached(SourceVideo& sv, int seq1, int seq2, int width, int height)
{
    return demodTwinCached(sv, seq1, seq2, width, height).grainIre;
}

int CineMap::benchComb(const QString& tbcFilePath, int startField, int endField)
{
    if (!m_md || !m_disc) return 0;

    SourceVideo sv;
    if (!sv.open(tbcFilePath, m_disc->getVideoFieldLength())) {
        qWarning() << "Failed to open TBC file";
        return 0;
    }

    const auto& vp = m_md->getVideoParameters();
    const int startFrame = frameIndexForField(startField);
    const int endFrame   = frameIndexForField(endField);
    if (startFrame < 0 || endFrame < 0) return 0;

    // Collect the frames first so both operators see exactly the same work.
    std::vector<std::pair<int,int>> frames;
    for (int fi = startFrame; fi < endFrame; ++fi) {
        if (m_disc->isPadded(fi)) continue;
        const int f1 = m_disc->getFirstFieldNumber(fi + 1);
        const int f2 = m_disc->getSecondFieldNumber(fi + 1);
        if (f1 < 1 || f2 < 1) continue;
        frames.emplace_back(f1, f2);
    }
    if (frames.empty()) return 0;

    // SourceVideo caches fields (QCache, maxCost 100), so whichever operator ran
    // first would otherwise absorb all the field I/O and look slow. Warm the cache
    // first, and keep the frame count inside the cache so neither pass pays eviction.
    const size_t maxFrames = 45;   // 90 fields, under the 100-field cache
    if (frames.size() > maxFrames) frames.resize(maxFrames);
    for (const auto& fr : frames) {
        (void)sv.getVideoField(fr.first);
        (void)sv.getVideoField(fr.second);
    }

    // Two timed passes over identical work, plus a repeat to expose run-to-run noise.
    QElapsedTimer t;
    double notchNs = 0.0, lipsNs = 0.0;
    constexpr int REPS = 3;

    for (int rep = 0; rep < REPS; ++rep) {
        t.start();
        for (const auto& fr : frames)
            (void)calculateNotchScore(sv, fr.first, fr.second, vp.fieldWidth, vp.fieldHeight);
        notchNs += t.nsecsElapsed();

        t.start();
        for (const auto& fr : frames)
            (void)calculateLipsScore(sv, fr.first, fr.second, vp.fieldWidth, vp.fieldHeight);
        lipsNs += t.nsecsElapsed();
    }

    const double n = static_cast<double>(frames.size()) * REPS;
    const double notchUs = notchNs / n / 1000.0;
    const double lipsUs  = lipsNs  / n / 1000.0;

    printf("frames=%d reps=%d (field cache warm)\n"
           "notch  %8.1f us/frame\n"
           "lips   %8.1f us/frame\n"
           "ratio  %8.2fx  (lips / notch)\n",
           static_cast<int>(frames.size()), REPS, notchUs, lipsUs,
           (notchUs > 0.0) ? (lipsUs / notchUs) : 0.0);

    // Cold-cache cost of the field reads alone, for scale: this is what BOTH
    // operators sit on top of, and it is paid once per field regardless of which
    // one runs, so it dilutes any compute-side difference between them.
    sv.close();
    SourceVideo sv2;
    if (sv2.open(tbcFilePath, m_disc->getVideoFieldLength())) {
        t.start();
        for (const auto& fr : frames) {
            (void)sv2.getVideoField(fr.first);
            (void)sv2.getVideoField(fr.second);
        }
        const double ioUs = t.nsecsElapsed() / static_cast<double>(frames.size()) / 1000.0;
        printf("field I/O %7.1f us/frame (2 fields, cold)\n", ioUs);
        printf("  => notch+I/O %.1f us, lips+I/O %.1f us, effective ratio %.2fx\n",
               notchUs + ioUs, lipsUs + ioUs,
               (notchUs + ioUs > 0.0) ? ((lipsUs + ioUs) / (notchUs + ioUs)) : 0.0);
    }

    fflush(stdout);
    return 1;
}

int CineMap::probeCombAxes(const QString& tbcFilePath, int startField, int endField)
{
    if (!m_md || !m_disc) return 0;

    SourceVideo sv;
    if (!sv.open(tbcFilePath, m_disc->getVideoFieldLength())) {
        qWarning() << "Failed to open TBC file";
        return 0;
    }

    const auto& vp = m_md->getVideoParameters();

    int startFrame = frameIndexForField(startField);
    int endFrame   = frameIndexForField(endField);
    if (startFrame < 0 || endFrame < 0) return 0;

    // lips alongside notch: lips already discounts the image's own vertical detail
    // per-pixel (metric = temporalDiff - max(spatialDetail, noiseFloor), accumulated
    // only where positive), so it is the sparse operator for this axis while notch is
    // the dense one. computeFrameMixedness skips lips whenever notch >= 0.10 on the
    // assumption that a large notch is self-evident — the case this column tests.
    printf("frame,f1,f2,cad,notchWithin,notchAcross,ratio,lipsWithin,lipsAcross\n");

    int rows = 0;
    for (int fi = startFrame; fi < endFrame; ++fi) {
        if (m_disc->isPadded(fi) || m_disc->isPadded(fi + 1)) continue;

        const int f1 = m_disc->getFirstFieldNumber(fi + 1);
        const int f2 = m_disc->getSecondFieldNumber(fi + 1);
        const int n1 = m_disc->getFirstFieldNumber(fi + 2);
        if (f1 < 1 || f2 < 1 || n1 < 1) continue;

        const double within = calculateNotchScore(sv, f1, f2, vp.fieldWidth, vp.fieldHeight);
        const double across = calculateNotchScore(sv, n1, f2, vp.fieldWidth, vp.fieldHeight);
        const double lipsW  = calculateLipsScore(sv, f1, f2, vp.fieldWidth, vp.fieldHeight);
        const double lipsA  = calculateLipsScore(sv, n1, f2, vp.fieldWidth, vp.fieldHeight);

        printf("%d,%d,%d,%d,%.5f,%.5f,%.4f,%.5f,%.5f\n",
               fi, f1, f2, m_md->getField(f1).cinemap.cadenceId,
               within, across,
               (across > 1e-9) ? (within / across) : 0.0,
               lipsW, lipsA);
        rows++;
    }

    fflush(stdout);
    return rows;
}

const CineMap::NoiseFloor& CineMap::calibrateTwinFloor(SourceVideo& sv)
{
    if (m_noiseFloor.valid || !m_md || !m_disc) return m_noiseFloor;

    const auto& vp = m_md->getVideoParameters();
    const int hardMax = computeHardMaxField();
    if (hardMax < 8) return m_noiseFloor;

    // Blocks of consecutive pairs, spread evenly. Consecutive pairs within a block
    // share fields, so a block costs far less than its pair count suggests, and a
    // block is what gives floor-sitting content a chance to show itself: a telecined
    // block contains ~1 twin per 5 frames, a static block is all floor, a block of
    // pure motion contains none and reports its own level instead.
    constexpr int WANT_BLOCKS = 48;
    constexpr int BLOCK_PAIRS = 16;

    const int stride = std::max(BLOCK_PAIRS + 2, (hardMax - 2) / WANT_BLOCKS);

    std::vector<double> blockMin;
    blockMin.reserve(WANT_BLOCKS);
    int pairsMeasured = 0;

    for (int start = 1; start + 2 <= hardMax; start += stride) {
        double lo = std::numeric_limits<double>::max();

        for (int a = start; a < start + BLOCK_PAIRS && a + 2 <= hardMax; ++a) {
            const int b = a + 2;

            auto fa = m_md->getField(a);
            auto fb = m_md->getField(b);
            if (fa.pad || fb.pad) continue;
            if (fa.isFirstField != fb.isFirstField) continue;
            if (boundaryBetween(a, b)) continue;

            const TwinDemod m = demodTwinCached(sv, a, b, vp.fieldWidth, vp.fieldHeight);
            if (!m.valid) continue;

            pairsMeasured++;
            if (m.grainIre < lo) lo = m.grainIre;
        }

        if (lo < std::numeric_limits<double>::max()) blockMin.push_back(lo);
    }

    if (blockMin.size() < 8) return m_noiseFloor;

    std::sort(blockMin.begin(), blockMin.end());
    auto pct = [&](double p) {
        const size_t k = static_cast<size_t>(std::llround(p * (blockMin.size() - 1)));
        return blockMin[std::min(k, blockMin.size() - 1)];
    };

    m_noiseFloor.blocks = static_cast<int>(blockMin.size());
    m_noiseFloor.pairs  = pairsMeasured;
    m_noiseFloor.p10    = pct(0.10);
    m_noiseFloor.p25    = pct(0.25);
    m_noiseFloor.median = pct(0.50);
    m_noiseFloor.ire    = m_noiseFloor.p10;

    int agree = 0;
    for (double v : blockMin) {
        if (v >= m_noiseFloor.ire * 0.8 && v <= m_noiseFloor.ire * 1.2) agree++;
    }
    m_noiseFloor.concentration = static_cast<double>(agree) / blockMin.size();

    // bPSNR cross-check only. Recorded so a wildly-off calibration is visible; it
    // must never scale the floor (measured to make the estimate worse).
    const double black = (vp.black16bIre > 0) ? vp.black16bIre : 0.0;
    const double white = (vp.white16bIre > black) ? vp.white16bIre : 65535.0;
    double bp = 0.0;
    int bpN = 0;
    for (int f = 1; f <= hardMax; f += std::max(1, hardMax / 400)) {
        const double v = m_md->getFieldVitsMetrics(f).bPSNR;
        if (v > 0.1) { bp += v; bpN++; }
    }
    if (bpN > 0) {
        m_noiseFloor.bpsnrIre = (65535.0 / std::pow(10.0, (bp / bpN) / 20.0))
                              * (100.0 / (white - black));
    }

    m_noiseFloor.valid = (m_noiseFloor.ire > 0.0);
    return m_noiseFloor;
}

bool CineMap::filmRuledOut(SourceVideo& sv, int segStart, int segEnd)
{
    const NoiseFloor& nf = calibrateTwinFloor(sv);
    if (!nf.valid) return false;          // no floor measured: cannot rule anything out

    int hits = 0, pairs = 0;
    double quiet = 0.0;
    std::vector<int> hitFields;

    // Geometry operating point: the tight floor. The generous one deliberately
    // over-admits so a twin is never missed, but those extra hits are not twins and
    // scatter across offsets, diluting the very concentration being measured.
    const double dutyG = twinDuty(sv, segStart, segEnd,
                                  nf.ire * FLOOR_MULT_GEOMETRY,
                                  &hits, &pairs, &quiet, &hitFields);

    const int possible = std::max(0, segEnd - segStart - 1);
    const double coverage = (possible > 0)
                          ? (static_cast<double>(pairs) / possible) : 0.0;
    if (coverage < MIN_TAG_COVERAGE) return false;   // dark, not video

    // Everything at the floor is the absence of information, not evidence of video:
    // a static passage's field difference is pure noise whether it is film or not.
    if (dutyG >= 0.80) return false;

    const double conc = cycleConcentration(hitFields);
    const bool ruledOut = (conc < CYCLE_CONCENTRATION_MIN);

    if (m_decisionTraceEnabled) {
        qInfo().noquote()
            << QString("CineMap decision: FILM_CENSUS fields [%1..%2] floor=%3 pairs=%4 coverage=%5 dutyGeo=%6 hits=%7 conc=%8 filmRuledOut=%9")
               .arg(segStart).arg(segEnd)
               .arg(nf.ire, 0, 'f', 4).arg(pairs)
               .arg(coverage, 0, 'f', 3)
               .arg(dutyG, 0, 'f', 3).arg(hits)
               .arg(conc, 0, 'f', 3)
               .arg(ruledOut ? "yes" : "no");
    }

    return ruledOut;
}

double CineMap::cycleConcentration(const std::vector<int>& hitFields) const
{
    if (!m_md || hitFields.size() < 4) return 0.0;

    // Histogram the hits by offset within the 10-field (5-frame) cadence cycle.
    int bucket[10] = {0};
    for (int f : hitFields) {
        int o = f % 10;
        if (o < 0) o += 10;
        bucket[o]++;
    }

    // Best pair of offsets five apart. Opposite parity follows from a 5-field
    // offset in an undisrupted sequence; pads can break that, so confirm rather
    // than assume by checking a representative hit from each half.
    double best = 0.0;
    for (int o = 0; o < 5; ++o) {
        const int paired = bucket[o] + bucket[o + 5];
        if (paired == 0) continue;

        if (bucket[o] > 0 && bucket[o + 5] > 0) {
            int a = -1, b = -1;
            for (int f : hitFields) {
                const int m = ((f % 10) + 10) % 10;
                if (m == o && a < 0) a = f;
                if (m == o + 5 && b < 0) b = f;
            }
            if (a > 0 && b > 0
                && m_md->getField(a).isFirstField == m_md->getField(b).isFirstField) {
                continue;   // same parity: not a 3:2 twin pair
            }
        }

        best = std::max(best, static_cast<double>(paired) / hitFields.size());
    }

    return best;
}

double CineMap::twinDuty(SourceVideo& sv,
                         int startField,
                         int endField,
                         double floorIre,
                         int* outHits,
                         int* outPairs,
                         double* outQuietestIre,
                         std::vector<int>* outHitFields)
{
    if (outHits) *outHits = 0;
    if (outPairs) *outPairs = 0;
    if (outQuietestIre) *outQuietestIre = 0.0;
    if (outHitFields) outHitFields->clear();
    if (!m_md || floorIre <= 0.0) return 0.0;

    const auto& vp = m_md->getVideoParameters();
    const int total = m_md->getNumberOfFields();
    startField = std::max(1, startField);
    endField   = std::min(total, endField);

    int hits = 0, pairs = 0;
    double quietest = std::numeric_limits<double>::max();

    for (int a = startField; a + 2 <= endField; ++a) {
        const int b = a + 2;

        auto fa = m_md->getField(a);
        auto fb = m_md->getField(b);
        if (fa.pad || fb.pad) continue;
        if (fa.isFirstField != fb.isFirstField) continue;

        const TwinDemod m = demodTwinCached(sv, a, b, vp.fieldWidth, vp.fieldHeight);
        if (!m.valid) continue;

        pairs++;
        if (m.grainIre < quietest) quietest = m.grainIre;
        if (m.grainIre < floorIre) {
            hits++;
            if (outHitFields) outHitFields->push_back(a);
        }
    }

    if (outHits) *outHits = hits;
    if (outPairs) *outPairs = pairs;
    if (outQuietestIre && quietest < std::numeric_limits<double>::max())
        *outQuietestIre = quietest;

    return (pairs > 0) ? (static_cast<double>(hits) / pairs) : 0.0;
}

int CineMap::probeDgFloor(const QString& tbcFilePath, const QString& ranges)
{
    if (!m_md || !m_disc) return 0;

    SourceVideo sv;
    if (!sv.open(tbcFilePath, m_disc->getVideoFieldLength())) {
        qWarning() << "Failed to open TBC file";
        return 0;
    }

    const NoiseFloor& nf = calibrateTwinFloor(sv);
    if (!nf.valid) {
        qWarning() << "Twin floor calibration failed (too few usable blocks).";
        return 0;
    }

    printf("floor=%.4f IRE  (blockMin p10=%.4f p25=%.4f median=%.4f)  "
           "concentration=%.3f  blocks=%d  pairs=%d  bpsnrIre=%.4f  ratio=%.3f\n",
           nf.ire, nf.p10, nf.p25, nf.median, nf.concentration,
           nf.blocks, nf.pairs, nf.bpsnrIre,
           (nf.bpsnrIre > 0.0) ? (nf.ire / nf.bpsnrIre) : 0.0);

    if (nf.concentration < 0.20) {
        printf("WARNING: block minima do not concentrate — no sharp floor found; "
               "treat this calibration as unproven\n");
    }

    if (ranges.isEmpty()) {
        fflush(stdout);
        return 1;
    }

    printf("\n%-16s %6s %6s %6s %8s %8s %8s %8s %5s  %s\n",
           "range", "pairs", "cover", "quiet", "q/floor", "dutyGeo", "dutyRec",
           "hitsRec", "conc", "read");

    int reported = 0;
    for (const QString& spec : ranges.split(',', Qt::SkipEmptyParts)) {
        const auto parts = spec.split('-', Qt::SkipEmptyParts);
        if (parts.size() != 2) continue;
        bool okA = false, okB = false;
        const int lo = parts.at(0).toInt(&okA);
        const int hi = parts.at(1).toInt(&okB);
        if (!okA || !okB || hi <= lo) continue;

        int hitsG = 0, hitsR = 0, pairs = 0;
        double quiet = 0.0;
        std::vector<int> hitFieldsG, hitFieldsR;
        const double dutyG = twinDuty(sv, lo, hi, nf.ire * FLOOR_MULT_GEOMETRY,
                                      &hitsG, &pairs, &quiet, &hitFieldsG);
        const double dutyR = twinDuty(sv, lo, hi, nf.ire * FLOOR_MULT_RECALL,
                                      &hitsR, &pairs, &quiet, &hitFieldsR);

        // Each question reads its own operating point. "All floor, so no
        // information" reads the TIGHT point: there, static sits at 100% while
        // telecine sits near the 1/5 that 3:2 predicts, whereas the generous point
        // inflates a low-motion telecine shot to ~57% and crowds that boundary.
        // Presence reads the generous point plus the cycle-concentration test.
        // margin (quietest pair over the floor) is reported for confidence but no
        // longer gates anything: under preponderance the tag is decided by which
        // regime dominates, not by how emphatic the negative is.

        // A denial must never be confused with an absence of data. For real film,
        // motion cannot hide a twin — the twin's difference is noise by construction,
        // so motion only raises the NON-twin pairs and improves the contrast. The one
        // way genuine film shows no twins is that the twins are missing from the data:
        // pads, dropouts, boundaries. That is darkness, and darkness must bridge
        // rather than deny, so an under-sampled range is not allowed to deny however
        // large its margin looks.
        constexpr double MIN_DENY_COVERAGE = 0.75;

        const int possible = std::max(0, hi - lo - 1);
        const double coverage = (possible > 0)
                              ? (static_cast<double>(pairs) / possible) : 0.0;

        const double margin = (nf.ire > 0.0) ? (quiet / nf.ire) : 0.0;
        // Concentration reads the TIGHT point, per the operating-point split: the
        // generous floor deliberately over-admits so a twin is never missed, but the
        // extra hits are not twins and scatter across offsets, diluting the very
        // structure being measured. Geometry gets the clean hits, recall gets the
        // liberal ones.
        const double conc = cycleConcentration(hitFieldsG);
        const bool cadenced = conc >= CYCLE_CONCENTRATION_MIN;

        // "All floor" must be tested at BOTH points. The tight point catches dead
        // static (every pair literally at the floor); the generous point catches
        // near-static, whose pairs sit just above the floor at 1.2-1.9x and would
        // otherwise slip through and read as twins on the strength of an incidental
        // cycle. Low-motion telecine reaches only ~57% at the generous point, so an
        // 80% line still separates it from near-static's 83-98%.
        const bool allFloor = (dutyG >= 0.80) || (dutyR >= 0.80);

        // Solve for the preponderance: the output regime is whichever covers the
        // BIGGEST PICTURE AREA. A composite carries more than one regime at once and
        // no single tag can serve all of it, so the dominant one wins outright — if
        // that is a film cadence we solve for that cadence, even though video
        // elements are present in the same shot. Video is not a safe default here; it
        // is simply what holds the most area when no film cadence dominates.
        //
        // The remaining elements are recovered by the user in second and third passes
        // via chroma decoder's --set-cadence, for which this solve is the background
        // plate. De-compositing inside the decoder is future work.
        //
        // No uncertain state survives here. Escalation belongs to the solve ladder,
        // not to the tag: every terminal read is an assertion.
        const char* read =
              (coverage < MIN_DENY_COVERAGE && !cadenced) ? "dark (insufficient sample)"
            : allFloor                                    ? "no information (all floor)"
            : cadenced                                    ? "CADENCE (solve for it)"
            : "video (preponderance)";

        printf("%-16s %6d %5.0f%% %6.2f %8.2f %7.1f%% %7.1f%% %8d %6.2f  %s\n",
               qPrintable(spec), pairs, coverage * 100.0, quiet, margin,
               dutyG * 100.0, dutyR * 100.0, hitsR, conc, read);
        reported++;
    }

    fflush(stdout);
    return reported;
}

int CineMap::probeDgRange(const QString& tbcFilePath, int startField, int endField)
{
    if (!m_md || !m_disc) return 0;

    SourceVideo sv;
    if (!sv.open(tbcFilePath, m_disc->getVideoFieldLength())) {
        qWarning() << "Failed to open TBC file";
        return 0;
    }

    const auto& vp = m_md->getVideoParameters();
    const int total = m_md->getNumberOfFields();

    startField = std::max(1, startField);
    endField   = std::min(total, endField);

    // bPSNR and the field phase IDs travel with each row: the twin floor is set by
    // the disc's noise, not by content, and the d=2 180-degree premise is a
    // recorded fact (phaseB - phaseA == 2 mod 4) rather than an assumption.
    printf("seqA,seqB,frameA,parity,cadA,roleA,phA,phB,bpsnr,noiseIre,"
           "grainIre,dCarr,sCarr,q,dCoh,sCoh,qCoh\n");

    const double black = (vp.black16bIre > 0) ? vp.black16bIre : 0.0;
    const double white = (vp.white16bIre > black) ? vp.white16bIre : 65535.0;
    const double scaleToIre = 100.0 / (white - black);

    int rows = 0;
    for (int a = startField; a + 2 <= endField; ++a) {
        const int b = a + 2;

        auto fa = m_md->getField(a);
        auto fb = m_md->getField(b);
        if (fa.pad || fb.pad) continue;
        if (fa.isFirstField != fb.isFirstField) continue;   // d=2 is same-parity by construction

        const TwinDemod m = demodTwinCached(sv, a, b, vp.fieldWidth, vp.fieldHeight);
        if (!m.valid) continue;

        // Local single-field video-noise amplitude implied by bPSNR. A twin's field
        // difference contains only noise, so this sets the floor the twin state sits at.
        double bpsnr = 0.5 * (m_md->getFieldVitsMetrics(a).bPSNR
                            + m_md->getFieldVitsMetrics(b).bPSNR);
        const double noiseIre = (bpsnr > 0.1)
                              ? (65535.0 / std::pow(10.0, bpsnr / 20.0)) * scaleToIre
                              : 0.0;

        printf("%d,%d,%d,%s,%d,%s,%d,%d,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
               a, b,
               (a + 1) / 2,
               fa.isFirstField ? "first" : "second",
               fa.cinemap.cadenceId,
               fa.cinemap.pulldownRole.isEmpty() ? "-" : qPrintable(fa.cinemap.pulldownRole),
               fa.fieldPhaseID, fb.fieldPhaseID,
               bpsnr, noiseIre,
               m.grainIre, m.dCarrierIre, m.sCarrierIre, m.q(),
               m.dCohIre, m.sCohIre, m.qCoh());
        rows++;
    }

    fflush(stdout);
    return rows;
}

CineMap::TwinDemod CineMap::calculateDemodulatedFieldDiff(SourceVideo& sv, int f1, int f2, int width, int height)
{
    TwinDemod out;
    if (!m_md || f1 < 1 || f2 < 1 || width <= 0 || height <= 0) return out;
    const auto vp = m_md->getVideoParameters();

    auto d1 = sv.getVideoField(f1);
    auto d2 = sv.getVideoField(f2);
    if (d1.size() < width * height || d2.size() < width * height) return out;

    const uint16_t* p1 = reinterpret_cast<const uint16_t*>(d1.constData());
    const uint16_t* p2 = reinterpret_cast<const uint16_t*>(d2.constData());

    const int startX = width / 4;
    const int endX   = (width * 3) / 4;
    const int yStart = height / 4;
    const int yEnd   = (height * 3) / 4;
    const int yStep  = 4;   // subsample lines for speed; x stays stride 1

    // dG twin metric. A telecine twin is the same film frame scanned twice, so its
    // film grain is *identical* in both fields; a non-twin same-parity pair carries
    // independently animated grain. Video noise animates in both cases (common-mode
    // floor). For a same-parity d=2 pair the field difference D = field1 - field2 is,
    // in a static region, a clean coherent 4fsc tone (the static chroma's subcarrier,
    // opposite-phase between the two fields so it *doubles*) plus grain plus noise.
    // The subcarrier swamps the grain, so we strip it: over a short sliding window the
    // 4fsc quadrature basis is just {1,0,-1,0}/{0,1,0,-1}, so the local tone is two
    // alternating sums (I on even samples, Q on odd). Grain is broadband and does not
    // survive the windowed I/Q estimate, so it stays in the residual. We then measure
    // the ENERGY (RMS) of D minus that tone: a true twin -> grain cancels -> residual
    // collapses to the noise floor; animated grain survives as extra energy.
    //
    // The 180-degree relation that makes the tone double in D also makes it CANCEL
    // in the sum S = field1 + field2, so the same pass measures a second, independent
    // channel (see TwinDemod). D's coherent tone is the MEAN chroma C1+C2' — until now
    // discarded as nuisance, it is the power meter for the test. S's coherent tone is
    // the chroma DIFFERENCE C1-C2', which is zero for a twin no matter how much colour
    // is present. Both magnitudes come from the same quadrature estimate, so the second
    // channel costs one more prefix-sum pass over data already in hand.
    constexpr int N = 8;        // window length in samples (2 carrier cycles)
    constexpr int H = N / 2;

    // Coherent pooling run length. Chroma phase is near-constant across a run this
    // short, so its (I,Q) vectors add; fine luma detail that lands in the subcarrier
    // band has rapidly varying phase and averages down. That matters most for S,
    // whose broadband part is the MEAN luma (large) rather than a luma difference.
    constexpr int R = 16;

    const double black = (vp.black16bIre > 0) ? vp.black16bIre : 0.0;
    const double white = (vp.white16bIre > 0) ? vp.white16bIre : 65535.0;
    const double scaleToIre = 100.0 / (white - black);

    double sumSq = 0.0;
    uint64_t count = 0;

    double dCarrSq = 0.0, sCarrSq = 0.0;   // per-sample RMS accumulators
    double dCohSq  = 0.0, sCohSq  = 0.0;   // coherently-pooled accumulators
    uint64_t cohCount = 0;

    std::vector<double> D(width), S(width);
    std::vector<double> sI(width), sQ(width);   // signed quadrature projections of D
    std::vector<double> tI(width), tQ(width);   // ... and of S
    std::vector<double> PI(width + 1), PQ(width + 1); // prefix sums for sliding window
    std::vector<double> QI(width + 1), QQ(width + 1);

    for (int y = yStart; y < yEnd; y += yStep) {
        const uint16_t* l1 = p1 + y * width;
        const uint16_t* l2 = p2 + y * width;

        // Field difference and its 4fsc quadrature samples. cos(pi*x/2) is nonzero
        // only on even x (+/-1); sin only on odd x. So sI carries even samples, sQ odd.
        for (int x = 0; x < width; ++x) {
            const double a = static_cast<double>(l1[x]);
            const double b = static_cast<double>(l2[x]);
            const double d = a - b;
            const double s = a + b;
            D[x] = d;
            S[x] = s;
            switch (x & 3) {
                case 0: sI[x] =  d; sQ[x] = 0.0; tI[x] =  s; tQ[x] = 0.0; break;
                case 1: sI[x] = 0.0; sQ[x] =  d; tI[x] = 0.0; tQ[x] =  s; break;
                case 2: sI[x] = -d; sQ[x] = 0.0; tI[x] = -s; tQ[x] = 0.0; break;
                default: sI[x] = 0.0; sQ[x] = -d; tI[x] = 0.0; tQ[x] = -s; break;
            }
        }

        PI[0] = 0.0; PQ[0] = 0.0; QI[0] = 0.0; QQ[0] = 0.0;
        for (int x = 0; x < width; ++x) {
            PI[x + 1] = PI[x] + sI[x];
            PQ[x + 1] = PQ[x] + sQ[x];
            QI[x + 1] = QI[x] + tI[x];
            QQ[x + 1] = QQ[x] + tQ[x];
        }

        for (int x = startX; x < endX; ++x) {
            const int lo = std::max(0, x - H);
            const int hi = std::min(width, x + H);

            // Basis power in the (edge-clamped) window: count of even / odd indices.
            const int evens = ((hi + 1) >> 1) - ((lo + 1) >> 1);
            const int odds  = (hi - lo) - evens;

            const double Ihat = (evens > 0) ? (PI[hi] - PI[lo]) / evens : 0.0;
            const double Qhat = (odds  > 0) ? (PQ[hi] - PQ[lo]) / odds  : 0.0;

            double carrier;
            switch (x & 3) {
                case 0: carrier =  Ihat; break;
                case 1: carrier =  Qhat; break;
                case 2: carrier = -Ihat; break;
                default: carrier = -Qhat; break;
            }

            const double grain = D[x] - carrier;
            sumSq += grain * grain;
            count++;

            // Sum-channel quadrature over the same window: the chroma difference.
            const double IhatS = (evens > 0) ? (QI[hi] - QI[lo]) / evens : 0.0;
            const double QhatS = (odds  > 0) ? (QQ[hi] - QQ[lo]) / odds  : 0.0;

            dCarrSq += Ihat  * Ihat  + Qhat  * Qhat;
            sCarrSq += IhatS * IhatS + QhatS * QhatS;
        }

        // Coherent pooling: average the (I,Q) vectors over a run, then take the
        // magnitude. Chroma survives; random-phase luma leak averages down.
        for (int x0 = startX; x0 + R <= endX; x0 += R) {
            double dIsum = 0.0, dQsum = 0.0, sIsum = 0.0, sQsum = 0.0;
            for (int x = x0; x < x0 + R; ++x) {
                const int lo = std::max(0, x - H);
                const int hi = std::min(width, x + H);
                const int evens = ((hi + 1) >> 1) - ((lo + 1) >> 1);
                const int odds  = (hi - lo) - evens;

                dIsum += (evens > 0) ? (PI[hi] - PI[lo]) / evens : 0.0;
                dQsum += (odds  > 0) ? (PQ[hi] - PQ[lo]) / odds  : 0.0;
                sIsum += (evens > 0) ? (QI[hi] - QI[lo]) / evens : 0.0;
                sQsum += (odds  > 0) ? (QQ[hi] - QQ[lo]) / odds  : 0.0;
            }
            const double inv = 1.0 / static_cast<double>(R);
            dIsum *= inv; dQsum *= inv; sIsum *= inv; sQsum *= inv;

            dCohSq += dIsum * dIsum + dQsum * dQsum;
            sCohSq += sIsum * sIsum + sQsum * sQsum;
            cohCount++;
        }
    }

    if (count == 0) return out;

    out.grainIre    = std::sqrt(sumSq   / count) * scaleToIre;  // RMS carrier-stripped grain, IRE
    out.dCarrierIre = std::sqrt(dCarrSq / count) * scaleToIre;
    out.sCarrierIre = std::sqrt(sCarrSq / count) * scaleToIre;
    if (cohCount > 0) {
        out.dCohIre = std::sqrt(dCohSq / cohCount) * scaleToIre;
        out.sCohIre = std::sqrt(sCohSq / cohCount) * scaleToIre;
    }
    out.valid = true;
    return out;
}

double CineMap::calculateBoostedDemodDiff(SourceVideo& sv, int f1, int f2, int width, int height)
{
    if (!m_md || f1 < 1 || f2 < 1 || width <= 0 || height <= 0) return 1000.0;
    const auto vp = m_md->getVideoParameters();

    auto d1 = sv.getVideoField(f1);
    auto d2 = sv.getVideoField(f2);
    if (d1.size() < width * height || d2.size() < width * height) return 1000.0;

    const uint16_t* p1 = reinterpret_cast<const uint16_t*>(d1.constData());
    const uint16_t* p2 = reinterpret_cast<const uint16_t*>(d2.constData());

    const int yStart = height / 4;
    const int yEnd   = height * 3 / 4;
    const int xStart = width / 4;
    const int xEnd   = width * 3 / 4;

    std::vector<double> Y1, Y2;

    const double black = (vp.black16bIre > 0) ? vp.black16bIre : 0.0;
    const double white = (vp.white16bIre > black) ? vp.white16bIre : 65535.0;
    const double scaleToIre = 100.0 / (white - black);

    double sum = 0.0;
    uint64_t count = 0;

    for (int y = yStart; y < yEnd; y += 4) {
        computeLumaLine_Bucket(p1 + y * width, Y1, width);
        computeLumaLine_Bucket(p2 + y * width, Y2, width);

        for (int x = xStart; x < xEnd; x += 2) {
            double L1 = (Y1[x] - black) * scaleToIre;
            double L2 = (Y2[x] - black) * scaleToIre;
            double avgL = 0.5 * (L1 + L2);
            double diff = std::abs(L1 - L2);

            // --- BOOST FOR SCORING ONLY ---
            // Emphasize differences where we expect real picture content
            // and small raw deltas might still be important.
            //
            // We never downweight; boost is >= 1.0.
            double boost = 1.0;

            // Low/mid luma: dark shadows / midtones – where grain and detail live.
            if (avgL < 60.0) {
                // Map 0..60 IRE -> boost 1.0..2.0
                double t = std::clamp(avgL / 60.0, 0.0, 1.0);
                boost = 1.0 + (1.0 - t); // 2.0 at 0 IRE, 1.0 at 60 IRE
            } else {
                // High luma (bright regions): leave ~unity or a slight boost if desired
                boost = 1.0;
            }

            sum += diff * boost;
            count++;
        }
    }

    if (count == 0) return 1000.0;
    return sum / count;
}

double CineMap::getAdaptiveTwinThreshold(int f1, int f2)
{
    if (!m_md) return 2.5; 

    // Retrieve bSNR (Black Signal-to-Noise Ratio) from metadata
    double bpsnr1 = m_md->getFieldVitsMetrics(f1).bPSNR;
    double bpsnr2 = m_md->getFieldVitsMetrics(f2).bPSNR;

    // Sanity check: if metrics are missing (0.0), assume a noisy disc (38dB) to be safe
    if (bpsnr1 <= 0.1) bpsnr1 = 38.0;
    if (bpsnr2 <= 0.1) bpsnr2 = 38.0;

    const double avgBpsnr = (bpsnr1 + bpsnr2) * 0.5;

    // Base Reference: 45dB is "Clean". Below that is noisy.
    constexpr double BASE_THRESHOLD_IRE = 3.0; 
    constexpr double BASE_SNR_REF_DB    = 40.0;

    // Scale threshold up for noisy discs (logarithmic dB scale)
    // If SNR drops by 6dB, noise amplitude doubles.
    const double dbDiff = BASE_SNR_REF_DB - avgBpsnr;
    double noiseScale = std::pow(10.0, dbDiff / 20.0);
    
    // Clamp: Don't let threshold go below 1.2 (too strict) or above 5.0 (too loose)
    noiseScale = std::clamp(noiseScale, 0.6, 2.5);

    double thr = BASE_THRESHOLD_IRE * noiseScale;
    // Apply global twin/dG sensitivity: >1.0  more sensitive (lower threshold)
    if (m_twinSensitivity > 0.0) {
        thr /= m_twinSensitivity;
    }

    return thr;
}

double CineMap::twinConfidence(SourceVideo& sv, int seqA, int seqB)
{
    TwinConfDetail d;
    return twinConfidence(sv, seqA, seqB, d);
}

double CineMap::twinConfidence(SourceVideo& sv, int seqA, int seqB, TwinConfDetail& d)
{
    d = TwinConfDetail{};
    if (!m_md) return 0.0;
    const auto& vp = m_md->getVideoParameters();

    {
        // Copy out by value: demodTwinCached returns a reference into the cache,
        // which the neighbour lookups below may rehash.
        const TwinDemod in = demodTwinCached(sv, seqA, seqB, vp.fieldWidth, vp.fieldHeight);
        d.diffIn  = in.grainIre;
        d.qIn     = in.qCoh();
        d.powerIn = in.dCohIre;
    }
    d.threshAbs = getAdaptiveTwinThreshold(seqA, seqB);

    const int totalFields = m_md->getNumberOfFields();
    if (seqA - 2 >= 1 && !boundaryBetween(seqA - 2, seqA)) {
        const TwinDemod pre = demodTwinCached(sv, seqA - 2, seqA, vp.fieldWidth, vp.fieldHeight);
        d.diffPre = pre.grainIre;
        d.qPre    = pre.qCoh();
    }
    if (seqB + 2 <= totalFields && !boundaryBetween(seqB, seqB + 2)) {
        const TwinDemod post = demodTwinCached(sv, seqB, seqB + 2, vp.fieldWidth, vp.fieldHeight);
        d.diffPost = post.grainIre;
        d.qPost    = post.qCoh();
    }

    if (d.diffPre < 900.0 && d.diffPost < 900.0) d.neighborActivity = std::min(d.diffPre, d.diffPost);
    else if (d.diffPre < 900.0) d.neighborActivity = d.diffPre;
    else if (d.diffPost < 900.0) d.neighborActivity = d.diffPost;

    // Sparse scorer: don't prove twinship, serve clues. Confidence is the pure RELATIVE
    // quietness of this pair vs its same-parity d=2 neighbors. A pair more twin-like than
    // its neighbors yields a positive score; downstream election picks the phase whose
    // A/C pattern collects the most score. No absolute floors at this layer.
    if (d.neighborActivity < 900.0) {
        d.ratio = d.diffIn / (d.neighborActivity + 0.01);
        if (d.ratio < 1.0) {
            d.ratioScore = 1.0 - d.ratio;
            d.confidence = d.ratioScore;
        }
        if (d.diffIn < 0.5 && d.neighborActivity < 0.5) {
            d.silenceMatch = true;   // trace only; not a confidence override
        }
    } else if (d.threshAbs > 0.0 && d.diffIn < d.threshAbs) {
        // Segment-edge fallback: no neighbor available, lean on absolute.
        d.threshScore = 1.0 - (d.diffIn / d.threshAbs);
        d.confidence = d.threshScore;
    }

    return d.confidence;
}

bool CineMap::tryCommitReciprocalGang(SourceVideo& sv,
                                      int a,
                                      int b,
                                      double hysteresis,
                                      const SegmentCaptureCache* cacheOrNull)
{
    if (!m_md) return false;
    if (a < 1 || b < 1) return false;
    if (a == b) return false;

    const int total = m_md->getNumberOfFields();
    if (a > total || b > total) return false;

    auto fa = m_md->getField(a);
    auto fb = m_md->getField(b);

    // Must be same parity (telecine twins are same-parity)
    if (fa.isFirstField != fb.isFirstField) return false;

    // If we have cache, enforce A/C strict-only for doplGang semantics (outer-edge only)
    if (cacheOrNull) {
        const auto& cache = *cacheOrNull;
        if (!cache.validSeq(a) || !cache.validSeq(b)) return false;
        TwinACInfo ac = classifyTwinAC_strict(a, b, cache);
        if (ac.role == TwinACRole::Unknown) return false;

        // Boosted sanity gate for A/C twins:
        const auto& vp = m_md->getVideoParameters();
        double dSanity = calculateBoostedDemodDiff(sv, a, b, vp.fieldWidth, vp.fieldHeight);
        double thr     = getAdaptiveTwinThreshold(a, b);
        if (!(thr > 0.0) || dSanity >= thr) {
            // Even after boosting, these two fields are not close enough
            // in demodulated luma to be reliable twins.
            return false;
        }
    }

    // Helper to break an existing reciprocal link x<->y if present.
    auto breakIfReciprocal = [&](int x, int y) {
        if (x < 1 || y < 1 || x > total || y > total) return;
        if (m_doplGang[x] == y && m_doplGang[y] == x) {
            m_doplGang[x] = std::nullopt;
            m_doplGang[y] = std::nullopt;
        }
    };

    // If already linked correctly, done
    if (m_doplGang[a].has_value() && m_doplGang[a].value() == b &&
        m_doplGang[b].has_value() && m_doplGang[b].value() == a) {
        return true;
    }

    // Resolve conflicts by recomputing merits on demand (no persisted conf).
    const double sNew = twinConfidence(sv, a, b);

    // If A has another mate, candidate must beat it
    if (m_doplGang[a].has_value() && m_doplGang[a].value() != b) {
        int a2 = m_doplGang[a].value();
        if (a2 >= 1 && a2 <= total) {
            double sOld = twinConfidence(sv, a, a2);
            if (sNew < sOld + hysteresis) return false;
            breakIfReciprocal(a, a2);
        }
    }

    // If B has another mate, candidate must beat it
    if (m_doplGang[b].has_value() && m_doplGang[b].value() != a) {
        int b2 = m_doplGang[b].value();
        if (b2 >= 1 && b2 <= total) {
            double sOld = twinConfidence(sv, b, b2);
            if (sNew < sOld + hysteresis) return false;
            breakIfReciprocal(b, b2);
        }
    }

    // Commit reciprocal
    m_doplGang[a] = b;
    m_doplGang[b] = a;
    return true;
}

// Edges & Geometry - if image is static, pulldown frames will be unclear in CLV; this is the fallback
// Once we identify twins, we can see which triple they belong to by inspecting the geometry
// A has a trailing spare, C has a leading spare; capture frames reveal this structure
// Differentiating A and C gets us our cadence offset, and twin location pins field order as well

void CineMap::buildTwinEdgesForPairs(SourceVideo& sv,
                                     const std::vector<std::pair<int,int>>& pairs,
                                     std::vector<TwinEdge>& outEdges,
                                     double minConf)
{
    if (!m_md) return;
    const int totalFields = m_md->getNumberOfFields();

    for (auto [a,b] : pairs) {
        if (a < 1 || b < 1 || a > totalFields || b > totalFields) continue;
        double conf = twinConfidence(sv, a, b);
        if (conf <= minConf) continue;

        // diff was cached by dgDiffIre inside twinConfidence
        DiffCacheKey key{std::min(a,b), std::max(a,b)};
        auto it = m_diffCache.find(key);
        double diff = (it != m_diffCache.end()) ? it->second : 999.0;

        outEdges.push_back({a, b, diff, conf});
    }
}

bool CineMap::hasReciprocalDgEdge(int a, int b) const
{
    if (!m_md || a == b) return false;
    if (a < 1 || b < 1) return false;

    int total = m_md->getNumberOfFields();
    if (a > total || b > total) return false;

    if (a >= (int)m_doplGang.size() || b >= (int)m_doplGang.size()) return false;
    return m_doplGang[a] == b && m_doplGang[b] == a;
}

CineMap::TwinACInfo CineMap::classifyTwinAC_strict(
    int seqA, int seqB,
    const SegmentCaptureCache& cache) const
{
    TwinACInfo info;

    if (!m_md) return info;
    if (!cache.validSeq(seqA) || !cache.validSeq(seqB)) return info;

    int lo = std::min(seqA, seqB);
    int hi = std::max(seqA, seqB);

    // twins separated by exactly 2 fields
    if (hi != lo + 2) {
        return info;
    }

    int compSeq = lo + 1;
    if (!cache.validSeq(compSeq)) {
        return info;
    }

    auto sameFrame = [&](int s1, int s2) -> bool {
        return cache.sameFrame(s1, s2);
    };

    bool compWithLo = sameFrame(lo, compSeq);
    bool compWithHi = sameFrame(hi, compSeq);

    // Must share the capture frame with exactly ONE twin
    if (compWithLo == compWithHi) {
        return info;
    }

    int defSeq   = compWithLo ? lo : hi;
    int spareSeq = compWithLo ? hi : lo;

    info.compSeq  = compSeq;
    info.defSeq   = defSeq;
    info.spareSeq = spareSeq;

    int minDC = std::min(defSeq, compSeq);
    int maxDC = std::max(defSeq, compSeq);

    if (spareSeq > maxDC) {
        info.role = TwinACRole::AType;  // trailing spare (A)
    } else if (spareSeq < minDC) {
        info.role = TwinACRole::CType;  // leading spare (C)
    } else {
        info.role = TwinACRole::Unknown;
    }

    return info;
}

std::vector<CineMap::TwinACInfo>
CineMap::harvestACTwinsForSegment_strict(
    int segStartField,
    int segEndField,
    const SegmentCaptureCache& cache) const
{
    std::vector<TwinACInfo> out;
    if (!m_md) return out;

    const int total = m_md->getNumberOfFields();

    for (int s = segStartField; s <= segEndField; ++s) {
        if (!m_doplGang[s].has_value()) continue;

        int mate = m_doplGang[s].value();
        if (mate <= s || mate < segStartField || mate > segEndField || mate > total) continue;

        auto f  = m_md->getField(s);
        auto fm = m_md->getField(mate);
        if (m_doplGang[mate] != s) continue;
        if (f.isFirstField != fm.isFirstField) continue;

        TwinACInfo ac = classifyTwinAC_strict(s, mate, cache);
        if (ac.role != TwinACRole::Unknown) {
            out.push_back(ac);
        }
    }

    return out;
}

bool CineMap::validatePhaseGeometry(int phaseOffset,
                                    int segStart,
                                    int segEnd,
                                    const SegmentCaptureCache& cache,
                                    QString* rejectReason)
{
    if (!m_md) return false;
    if (rejectReason) rejectReason->clear();

    int startFrameIdx = -1;
    for (int s = segStart; s <= segEnd; ++s) {
        if (cache.validSeq(s)) {
            startFrameIdx = cache.cap[s].frameIndex;
            break;
        }
    }

    if (startFrameIdx < 0) {
        if (rejectReason) *rejectReason = "no-start-frame";
        if (m_decisionTraceEnabled) {
            qInfo().noquote()
                << QString("CineMap decision: GEOMETRY_VALIDATE fields [%1..%2] phase=%3 result=false reason=no-start-frame")
                   .arg(segStart).arg(segEnd).arg(phaseOffset);
        }
        return false;
    }

    GeometryEvidence ev = gatherGeometryEvidenceForPhase(phaseOffset,
                                                        segStart,
                                                        segEnd,
                                                        startFrameIdx,
                                                        cache);

    if (!ev.hasAny()) {
        if (rejectReason) *rejectReason = "no-evidence";
        if (m_decisionTraceEnabled) {
            qInfo().noquote()
                << QString("CineMap decision: GEOMETRY_VALIDATE fields [%1..%2] phase=%3 startFrame=%4 result=false reason=no-evidence type=%5/%6 gap=%7/%8")
                   .arg(segStart).arg(segEnd).arg(phaseOffset).arg(startFrameIdx)
                   .arg(ev.typeAgree).arg(ev.typeSamples)
                   .arg(ev.gapAgree).arg(ev.gapSamples);
        }
        return false;
    }

    const int agree = ev.agree();
    const int disagree = ev.disagree();

    if (agree <= 0) {
        if (rejectReason) *rejectReason = "no-agree";
        if (m_decisionTraceEnabled) {
            qInfo().noquote()
                << QString("CineMap decision: GEOMETRY_VALIDATE fields [%1..%2] phase=%3 startFrame=%4 result=false reason=no-agree agree=%5 disagree=%6 typeAgree=%7 typeDisagree=%8 typeSamples=%9 gapAgree=%10 gapDisagree=%11 gapSamples=%12")
                   .arg(segStart).arg(segEnd).arg(phaseOffset).arg(startFrameIdx)
                   .arg(agree).arg(disagree)
                   .arg(ev.typeAgree).arg(ev.typeDisagree).arg(ev.typeSamples)
                   .arg(ev.gapAgree).arg(ev.gapDisagree).arg(ev.gapSamples);
        }
        return false;
    }

    if (disagree > agree + 2) {
        if (rejectReason) *rejectReason = "contradiction-dominates";
        if (m_decisionTraceEnabled) {
            qInfo().noquote()
                << QString("CineMap decision: GEOMETRY_VALIDATE fields [%1..%2] phase=%3 startFrame=%4 result=false reason=contradiction-dominates agree=%5 disagree=%6 typeAgree=%7 typeDisagree=%8 typeSamples=%9 gapAgree=%10 gapDisagree=%11 gapSamples=%12")
                   .arg(segStart).arg(segEnd).arg(phaseOffset).arg(startFrameIdx)
                   .arg(agree).arg(disagree)
                   .arg(ev.typeAgree).arg(ev.typeDisagree).arg(ev.typeSamples)
                   .arg(ev.gapAgree).arg(ev.gapDisagree).arg(ev.gapSamples);
        }
        return false;
    }

    if (m_decisionTraceEnabled) {
        qInfo().noquote()
            << QString("CineMap decision: GEOMETRY_VALIDATE fields [%1..%2] phase=%3 startFrame=%4 result=true agree=%5 disagree=%6 typeAgree=%7 typeDisagree=%8 typeSamples=%9 gapAgree=%10 gapDisagree=%11 gapSamples=%12")
               .arg(segStart).arg(segEnd).arg(phaseOffset).arg(startFrameIdx)
               .arg(agree).arg(disagree)
               .arg(ev.typeAgree).arg(ev.typeDisagree).arg(ev.typeSamples)
               .arg(ev.gapAgree).arg(ev.gapDisagree).arg(ev.gapSamples);
    }

    return true;
}

void CineMap::writeTwinEdgesToMetadata(SourceVideo& sv,
                                      const std::vector<TwinEdge>& edges,
                                      const SegmentCaptureCache* cacheOrNull)
{
    if (!m_md) return;
    if (edges.empty()) return;

    // Sort by confidence: this is only a processing order hint.
    // Actual accept/replace is decided by recomputing merits inside tryCommitReciprocalGang.
    std::vector<TwinEdge> sorted = edges;
    std::sort(sorted.begin(), sorted.end(),
              [](const TwinEdge& a, const TwinEdge& b) { return a.confidence > b.confidence; });

    // Sparse scorer: process every clue in order of relative quietness.
    // tryCommitReciprocalGang's internal absolute sanity gate (boosted demod vs
    // adaptive threshold) filters false positives at the actual write.
    constexpr double MIN_EDGE_HINT = 0.0;
    constexpr double HYSTERESIS    = 0.10;  // prevents thrash

    for (const auto& e : sorted) {
        if (e.confidence < MIN_EDGE_HINT) break;

        const int a = e.seqA;
        const int b = e.seqB;

        if (a < 1 || b < 1) continue;
        if (a == b) continue;

        (void)tryCommitReciprocalGang(sv, a, b, HYSTERESIS, cacheOrNull);
    }
}

CineMap::GeometryEvidence
CineMap::gatherGeometryEvidenceForPhase(int phaseOffset,
                                       int segStart,
                                       int segEnd,
                                       int startFrameIdx,
                                       const SegmentCaptureCache& cache) const
{
    GeometryEvidence ev;
    if (!m_md) return ev;
    if (startFrameIdx < 0) return ev;

    // 1) Gather classified A/C twin trios from current doplGang
    auto trios = harvestACTwinsForSegment_strict(segStart, segEnd, cache);
    if (trios.empty()) return ev;

    // Sort by definitional capture time
    std::sort(trios.begin(), trios.end(), [&](const TwinACInfo& a, const TwinACInfo& b) {
        int fa = cache.validSeq(a.defSeq) ? cache.cap[a.defSeq].frameIndex : 1e9;
        int fb = cache.validSeq(b.defSeq) ? cache.cap[b.defSeq].frameIndex : 1e9;
        return fa < fb;
    });

    // Helper: compute pos in 5-cycle under thesis
    auto posForDef = [&](const TwinACInfo& t) -> int {
        if (!cache.validSeq(t.defSeq)) return -1;
        int df = cache.cap[t.defSeq].frameIndex;
        int rel = df - startFrameIdx;
        int pos = rel + phaseOffset;
        pos %= 5;
        if (pos < 0) pos += 5;
        return pos;
    };

    // 2) Primary: compact type-placement evidence (AType->pos0, CType->pos3)
    for (const auto& t : trios) {
        int pos = posForDef(t);
        if (pos < 0) continue;

        ev.typeSamples++;
        if (t.role == TwinACRole::AType) {
            if (pos == 0) ev.typeAgree++; else ev.typeDisagree++;
        } else if (t.role == TwinACRole::CType) {
            if (pos == 3) ev.typeAgree++; else ev.typeDisagree++;
        }
    }

    // 3) Secondary: B/D gap evidence between successive twin events
    // We only score when we observe a clean 2-field gap between consecutive twin trios.
    for (size_t i = 0; i + 1 < trios.size(); ++i) {
        const auto& t1 = trios[i];
        const auto& t2 = trios[i + 1];

        // Determine the end of first trio and start of next trio in seq space.
        // Use min/max of {spare, comp, def} since this is a triple-field group.
        int t1_min = std::min({t1.defSeq, t1.compSeq, t1.spareSeq});
        int t1_max = std::max({t1.defSeq, t1.compSeq, t1.spareSeq});
        int t2_min = std::min({t2.defSeq, t2.compSeq, t2.spareSeq});
        int t2_max = std::max({t2.defSeq, t2.compSeq, t2.spareSeq});
        (void)t2_max;

        int gapStart = t1_max + 1;
        int gapEnd   = t2_min - 1;
        int gapLen = gapEnd - gapStart + 1;
        if (gapLen != 2) continue;
        if (!cache.validSeq(gapStart) || !cache.validSeq(gapEnd)) continue;

        bool observedIsD = cache.sameFrame(gapStart, gapEnd);

        // Predict gap type from thesis: C->A transition has D gap, A->C transition has B gap.
        // We infer whether each event is A-position or C-position from its type-placement (pos0 vs pos3).
        int p1 = posForDef(t1);
        int p2 = posForDef(t2);
        if (p1 < 0 || p2 < 0) continue;

        // Only score if events land on the canonical twin positions under this thesis.
        // (If they don't, that's already captured in typeDisagree; no need to double-count.)
        bool t1_isA = (t1.role == TwinACRole::AType && p1 == 0);
        bool t1_isC = (t1.role == TwinACRole::CType && p1 == 3);
        bool t2_isA = (t2.role == TwinACRole::AType && p2 == 0);
        bool t2_isC = (t2.role == TwinACRole::CType && p2 == 3);

        if (!((t1_isA || t1_isC) && (t2_isA || t2_isC))) continue;

        bool predictedIsD = false;
        if (t1_isC && t2_isA) predictedIsD = true;   // C -> A crosses D
        if (t1_isA && t2_isC) predictedIsD = false;  // A -> C crosses B

        ev.gapSamples++;
        if (predictedIsD == observedIsD) ev.gapAgree++;
        else ev.gapDisagree++;
    }

    return ev;
}

bool CineMap::tryLockByDgGeometry(SourceVideo& sv,
                                  int segStartField,
                                  int segEndField,
                                  const SegmentCaptureCache& cache,
                                  DgLock& outLock,
                                  QString* rejectReason)
{
    Q_UNUSED(sv);
    if (!m_md || !m_disc) return false;
    if (rejectReason) rejectReason->clear();

    auto acTwins = harvestACTwinsForSegment_strict(segStartField, segEndField, cache);

    if (acTwins.empty()) {
        if (rejectReason) *rejectReason = "no-strict-ac-twins";
        if (m_decisionTraceEnabled) {
            qInfo().noquote()
                << QString("CineMap decision: DG_GEOMETRY_LOCK fields [%1..%2] result=false reason=no-strict-ac-twins")
                   .arg(segStartField).arg(segEndField);
        }
        return false;
    }

    int startFrameIdx = -1;
    for (int s = segStartField; s <= segEndField; ++s) {
        if (cache.validSeq(s)) {
            startFrameIdx = cache.cap[s].frameIndex;
            break;
        }
    }

    if (startFrameIdx < 0) {
        if (rejectReason) *rejectReason = "no-start-frame";
        if (m_decisionTraceEnabled) {
            qInfo().noquote()
                << QString("CineMap decision: DG_GEOMETRY_LOCK fields [%1..%2] result=false reason=no-start-frame acTwins=%3")
                   .arg(segStartField).arg(segEndField).arg(acTwins.size());
        }
        return false;
    }

    std::array<double, 5> score = {0, 0, 0, 0, 0};
    std::array<int,    5> count = {0, 0, 0, 0, 0};

    int aTypeCount = 0;
    int cTypeCount = 0;

    for (const auto& t : acTwins) {
        if (t.role == TwinACRole::AType) aTypeCount++;
        else if (t.role == TwinACRole::CType) cTypeCount++;

        if (!cache.validSeq(t.defSeq)) continue;

        int df  = cache.cap[t.defSeq].frameIndex;
        int rel = df - startFrameIdx;

        if (m_decisionTraceEnabled) {
            qInfo().noquote()
                << QString("CineMap decision: DG_GEOMETRY_TWIN fields [%1..%2] role=%3 def=%4 comp=%5 spare=%6 defFrame=%7 relFrame=%8")
                   .arg(segStartField).arg(segEndField)
                   .arg(twinRoleName(t.role))
                   .arg(t.defSeq)
                   .arg(t.compSeq)
                   .arg(t.spareSeq)
                   .arg(df)
                   .arg(rel);
        }

        for (int p = 0; p < 5; ++p) {
            int pos = normalizePhase(rel + p, 5);
            bool ok = false;

            if (t.role == TwinACRole::AType) {
                ok = (pos == 0);
            } else if (t.role == TwinACRole::CType) {
                ok = (pos == 3);
            }

            if (ok) score[p] += 1.0;
            else    score[p] -= 0.25;

            count[p]++;
        }
    }

    int    bestP = -1;
    double bestS = -1e9;

    std::array<double, 5> normScore = {0, 0, 0, 0, 0};

    for (int p = 0; p < 5; ++p) {
        if (count[p] == 0) continue;

        double s = score[p] / count[p];
        normScore[p] = s;

        if (s > bestS) {
            bestS = s;
            bestP = p;
        }
    }

    // Expose the per-phase evidence vector unconditionally so the
    // evidence-additive election in solveSegment can consume it even when
    // brute geometry alone wouldn't declare a lock (no clear winner).
    bool anyCounts = false;
    for (int p = 0; p < 5; ++p) if (count[p] > 0) { anyCounts = true; break; }
    outLock.phaseScores = normScore;
    outLock.phaseScoresInformative = anyCounts;

    // Sparse-scorer election: don't compare bestScore against an absolute floor —
    // that's the same vestigial gate we removed at the harvest layer. The real
    // preponderance-of-evidence signal is the MARGIN between the winning phase
    // and its runner-up. A clear winner with positive net agreement beats a
    // near-tie even if the absolute count is modest.
    double secondBestS = -1e9;
    for (int p = 0; p < 5; ++p) {
        if (count[p] == 0) continue;
        if (p == bestP) continue;
        if (normScore[p] > secondBestS) secondBestS = normScore[p];
    }
    constexpr double MIN_GEOMETRY_MARGIN = 0.15;
    const bool hasClearWinner = (bestP != -1) && (bestS > 0.0)
                                && (bestS - secondBestS >= MIN_GEOMETRY_MARGIN);

    if (!hasClearWinner) {
        if (rejectReason) *rejectReason = "no-clear-winner";
        if (m_decisionTraceEnabled) {
            qInfo().noquote()
                << QString("CineMap decision: DG_GEOMETRY_LOCK fields [%1..%2] result=false reason=no-clear-winner startFrame=%3 acTwins=%4 AType=%5 CType=%6 bestPhase=%7 bestScore=%8 secondBest=%9 margin=%10 rawScores={%11} normScores={%12} counts={%13}")
                   .arg(segStartField).arg(segEndField)
                   .arg(startFrameIdx)
                   .arg(acTwins.size())
                   .arg(aTypeCount)
                   .arg(cTypeCount)
                   .arg(bestP)
                   .arg(bestS, 0, 'f', 4)
                   .arg(secondBestS > -1e8 ? secondBestS : 0.0, 0, 'f', 4)
                   .arg(secondBestS > -1e8 ? (bestS - secondBestS) : 0.0, 0, 'f', 4)
                   .arg(phaseArrayString(score, bestP))
                   .arg(phaseArrayString(normScore, bestP))
                   .arg(phaseIntArrayString(count));
        }
        return false;
    }

    outLock.anchorFrame = startFrameIdx;
    outLock.phaseOffset = bestP;
    outLock.baseOffset  = 0;
    outLock.confidence  = 0.90;

    if (m_decisionTraceEnabled) {
        qInfo().noquote()
            << QString("CineMap decision: DG_GEOMETRY_LOCK fields [%1..%2] result=lock source=twin-geometry startFrame=%3 phase=%4 conf=%5 acTwins=%6 AType=%7 CType=%8 bestScore=%9 secondBest=%13 margin=%14 rawScores={%10} normScores={%11} counts={%12}")
               .arg(segStartField).arg(segEndField)
               .arg(startFrameIdx)
               .arg(bestP)
               .arg(outLock.confidence, 0, 'f', 3)
               .arg(acTwins.size())
               .arg(aTypeCount)
               .arg(cTypeCount)
               .arg(bestS, 0, 'f', 4)
               .arg(phaseArrayString(score, bestP))
               .arg(phaseArrayString(normScore, bestP))
               .arg(phaseIntArrayString(count))
               .arg(secondBestS > -1e8 ? secondBestS : 0.0, 0, 'f', 4)
               .arg(secondBestS > -1e8 ? (bestS - secondBestS) : 0.0, 0, 'f', 4);
    }

    return true;
}

CineMap::PhaseRun
CineMap::scanForPhaseRun(const std::vector<FrameMixedness>& mixed,
                         int startField,
                         int endField,
                         const SegmentCaptureCache& cache)
{
    PhaseRun run;
    run.endField    = endField;
    run.type        = PhaseRun::Type::Unknown;
    run.confidence  = 0.0;
    run.phaseOffset = 0;

    if (mixed.empty()) {
        run.reason = "empty";
        if (m_decisionTraceEnabled) {
            qInfo().noquote()
                << QString("CineMap decision: MIXEDNESS_SCAN fields [%1..%2] result=unknown reason=empty")
                   .arg(startField).arg(endField);
        }
        return run;
    }

    int startFrameIdx = -1;
    if (cache.validSeq(startField)) {
        startFrameIdx = cache.cap[startField].frameIndex;
    }
    if (startFrameIdx == -1) {
        run.reason = "no-start-frame";
        if (m_decisionTraceEnabled) {
            qInfo().noquote()
                << QString("CineMap decision: MIXEDNESS_SCAN fields [%1..%2] result=unknown reason=no-start-frame")
                   .arg(startField).arg(endField);
        }
        return run;
    }

    const int numFrames = static_cast<int>(mixed.size());

    // ---------------------------------------------------------------------
    // Composite reality check:
    // - Absolute mixedness amplitude is not reliable.
    // - We care about relative separability of phase hypotheses.
    // ---------------------------------------------------------------------

    std::vector<double> vals;
    vals.reserve(static_cast<size_t>(numFrames));
    for (int i = 0; i < numFrames; ++i) vals.push_back(mixed[i].score);

    auto percentile = [&](double q) -> double {
        if (vals.empty()) return 0.0;
        q = std::clamp(q, 0.0, 1.0);
        std::vector<double> tmp = vals;
        const size_t k = static_cast<size_t>(q * static_cast<double>(tmp.size() - 1));
        std::nth_element(tmp.begin(), tmp.begin() + static_cast<std::ptrdiff_t>(k), tmp.end());
        return tmp[k];
    };

    const double p10 = percentile(0.10);
    const double p50 = percentile(0.50);
    const double p90 = percentile(0.90);
    const double denom = std::max(1e-9, (p90 - p10));

    // ABSOLUTE silence gate, and it must come BEFORE the percentile stretch.
    //
    // The stretch below rescales whatever spread exists into [0,1], so it cannot
    // tell "no comb anywhere" from "comb varies" — it manufactures a pattern from a
    // noise floor. That was survivable while mixedness was notch, whose floor is the
    // image's own vertical structure and therefore never small. Lips goes properly to
    // zero on clean content, so without this gate the stretch amplifies pure
    // numerical noise into confident locks.
    //
    // Lips is what makes an absolute test legitimate here: it is a residual measured
    // AFTER masking vertical detail and after subtracting its own noise floor, so its
    // zero means "no comb", not merely "no structure". Measured over 40 windows on
    // four discs, locks conjured from noise topped out at max lips 0.028 while every
    // genuine lock had max lips >= 0.189 — this sits in that gap.
    if (percentile(1.0) < LIPS_SILENCE) {
        run.type = PhaseRun::Type::Unknown;
        run.reason = "silence";

        if (m_decisionTraceEnabled) {
            qInfo().noquote()
                << QString("CineMap decision: MIXEDNESS_SCAN fields [%1..%2] frames=%3 result=unknown reason=silence maxLips=%4 threshold=%5")
                   .arg(startField).arg(endField).arg(numFrames)
                   .arg(percentile(1.0), 0, 'f', 6)
                   .arg(LIPS_SILENCE, 0, 'f', 4);
        }

        return run;
    }

    if ((p90 - p10) < 1e-6) {
        run.type = PhaseRun::Type::Unknown;
        run.reason = "flat-mixedness";

        if (m_decisionTraceEnabled) {
            qInfo().noquote()
                << QString("CineMap decision: MIXEDNESS_SCAN fields [%1..%2] frames=%3 result=unknown reason=flat-mixedness p10=%4 p50=%5 p90=%6")
                   .arg(startField).arg(endField).arg(numFrames)
                   .arg(p10, 0, 'f', 6)
                   .arg(p50, 0, 'f', 6)
                   .arg(p90, 0, 'f', 6);
        }

        return run;
    }

    std::array<double, 5> phaseScores = {0.0, 0.0, 0.0, 0.0, 0.0};

    double totalW = 0.0;
    int activeFrames = 0;

    constexpr double ACTIVE_W = 0.25;

    for (int i = 0; i < numFrames; ++i) {
        const double val = mixed[i].score;

        double w = (val - p10) / denom;
        w = std::clamp(w, 0.0, 1.0);

        totalW += w;
        if (w >= ACTIVE_W) activeFrames++;

        const int relFrame = mixed[i].frameIndex - startFrameIdx;

        for (int p = 0; p < 5; ++p) {
            int pos = (relFrame + p) % 5;
            if (pos < 0) pos += 5;

            // Canonical 3:2 cycle: positions 1 (AB) and 2 (BC) are mixed; 0/3/4
            // are clean. We reward mixedness at the "expected mixed" positions
            // AND penalize mixedness at the "expected clean" positions — the
            // anti-pattern term. Without it, 3-of-5-mixed content (e.g. content
            // that animates three positions in a row, not a true 3:2 cycle)
            // produces a misleading mixedness winner: the argmax phase puts two
            // of the three high spots on positions 1+2, scoring nearly as well
            // as a real lock. With the anti-pattern, the third high spot lands
            // on an "expected clean" position and subtracts, collapsing the
            // winner toward zero (and producing a tie between competing phases
            // where appropriate). Real 3:2 stays at the same score because its
            // clean positions are actually clean (w ≈ 0 there).
            if (pos == 1 || pos == 2) {
                phaseScores[p] += w;
            } else {
                phaseScores[p] -= w;
            }
        }
    }

    int bestP = -1;
    double bestScore = -1e9;
    double worstScore = 1e9;

    for (int p = 0; p < 5; ++p) {
        if (phaseScores[p] > bestScore) {
            bestScore = phaseScores[p];
            bestP = p;
        }
        if (phaseScores[p] < worstScore) {
            worstScore = phaseScores[p];
        }
    }

    // Expose the per-phase evidence vector unconditionally so the
    // evidence-additive election in solveSegment can consume it even when
    // mixedness alone wouldn't declare a lock (silence, busy-no-phase,
    // below-confidence).
    run.phaseScores = phaseScores;
    run.phaseScoresInformative = true;

    const double spread = bestScore - worstScore;
    const double avgW = (numFrames > 0) ? (totalW / numFrames) : 0.0;
    const double activeSpread = (activeFrames > 0) ? (spread / activeFrames) : 0.0;

    if (activeFrames == 0 && avgW < 0.05) {
        run.type = PhaseRun::Type::Unknown;
        run.reason = "silence";

        if (m_decisionTraceEnabled) {
            qInfo().noquote()
                << QString("CineMap decision: MIXEDNESS_SCAN fields [%1..%2] frames=%3 result=unknown reason=silence p10=%4 p50=%5 p90=%6 avgW=%7 activeFrames=%8 scores={%9}")
                   .arg(startField).arg(endField).arg(numFrames)
                   .arg(p10, 0, 'f', 6)
                   .arg(p50, 0, 'f', 6)
                   .arg(p90, 0, 'f', 6)
                   .arg(avgW, 0, 'f', 4)
                   .arg(activeFrames)
                   .arg(phaseArrayString(phaseScores, bestP));
        }

        return run;
    }

    if (avgW > 0.35 && activeSpread < (avgW * 0.15)) {
        run.type = PhaseRun::Type::Unknown;
        run.confidence = 0.0;
        run.reason = "busy-no-phase";

        if (m_decisionTraceEnabled) {
            qInfo().noquote()
                << QString("CineMap decision: MIXEDNESS_SCAN fields [%1..%2] frames=%3 result=unknown reason=busy-no-phase p10=%4 p50=%5 p90=%6 avgW=%7 activeFrames=%8 activeSpread=%9 threshold=%10 scores={%11}")
                   .arg(startField).arg(endField).arg(numFrames)
                   .arg(p10, 0, 'f', 6)
                   .arg(p50, 0, 'f', 6)
                   .arg(p90, 0, 'f', 6)
                   .arg(avgW, 0, 'f', 4)
                   .arg(activeFrames)
                   .arg(activeSpread, 0, 'f', 4)
                   .arg(avgW * 0.15, 0, 'f', 4)
                   .arg(phaseArrayString(phaseScores, bestP));
        }

        return run;
    }

    // Tie / ambiguity detection. Under anti-pattern, 3-mixed-in-a-row content
    // produces an INHERENT tie between two phases (both "off by one" placements
    // score the same — the math is symmetric in the three-consecutive case).
    // If the runner-up is too close to the winner, refuse to lock from
    // mixedness alone — the additive election still sees both vector entries
    // tied at 1.0 in mixN and lets dG break the tie correctly.
    double secondScore = -1e9;
    for (int p = 0; p < 5; ++p) {
        if (p == bestP) continue;
        if (phaseScores[p] > secondScore) secondScore = phaseScores[p];
    }
    const double phaseMargin = (bestP != -1 && secondScore > -1e8)
                               ? (bestScore - secondScore)
                               : bestScore;
    // Tie ratio: how distinctive is the winner vs the runner-up, scaled by
    // best. <0.25 means runner-up is within 25% of winner → ambiguous.
    const double relMargin = (bestScore > 1e-9) ? (phaseMargin / bestScore) : 0.0;

    double calculatedConf = std::min(0.95, 0.38 + (activeSpread * 10.0));

    if (activeFrames < 3) calculatedConf *= 0.85;

    if (bestP != -1 && relMargin < 0.25) {
        run.type = PhaseRun::Type::Unknown;
        run.confidence = 0.0;
        run.reason = "ambiguous-pattern";

        if (m_decisionTraceEnabled) {
            qInfo().noquote()
                << QString("CineMap decision: MIXEDNESS_SCAN fields [%1..%2] frames=%3 result=unknown reason=ambiguous-pattern bestPhase=%4 bestScore=%5 secondBest=%6 relMargin=%7 avgW=%8 activeFrames=%9 scores={%10}")
                   .arg(startField).arg(endField).arg(numFrames)
                   .arg(bestP)
                   .arg(bestScore, 0, 'f', 4)
                   .arg(secondScore, 0, 'f', 4)
                   .arg(relMargin, 0, 'f', 4)
                   .arg(avgW, 0, 'f', 4)
                   .arg(activeFrames)
                   .arg(phaseArrayString(phaseScores, bestP));
        }

        return run;
    }

    if (bestP != -1 && calculatedConf > 0.45) {
        run.type = PhaseRun::Type::Pulldown32;
        run.phaseOffset = bestP;
        run.confidence = calculatedConf;
        run.reason.clear();

        if (m_decisionTraceEnabled) {
            qInfo().noquote()
                << QString("CineMap decision: MIXEDNESS_SCAN fields [%1..%2] frames=%3 result=lock phase=%4 conf=%5 p10=%6 p50=%7 p90=%8 avgW=%9 activeFrames=%10 activeSpread=%11 scores={%12}")
                   .arg(startField).arg(endField).arg(numFrames)
                   .arg(bestP)
                   .arg(calculatedConf, 0, 'f', 3)
                   .arg(p10, 0, 'f', 6)
                   .arg(p50, 0, 'f', 6)
                   .arg(p90, 0, 'f', 6)
                   .arg(avgW, 0, 'f', 4)
                   .arg(activeFrames)
                   .arg(activeSpread, 0, 'f', 4)
                   .arg(phaseArrayString(phaseScores, bestP));
        }

        return run;
    }

    run.reason = "below-confidence";
    if (m_decisionTraceEnabled) {
        qInfo().noquote()
            << QString("CineMap decision: MIXEDNESS_SCAN fields [%1..%2] frames=%3 result=unknown reason=below-confidence phase=%4 conf=%5 p10=%6 p50=%7 p90=%8 avgW=%9 activeFrames=%10 activeSpread=%11 scores={%12}")
               .arg(startField).arg(endField).arg(numFrames)
               .arg(bestP)
               .arg(calculatedConf, 0, 'f', 3)
               .arg(p10, 0, 'f', 6)
               .arg(p50, 0, 'f', 6)
               .arg(p90, 0, 'f', 6)
               .arg(avgW, 0, 'f', 4)
               .arg(activeFrames)
               .arg(activeSpread, 0, 'f', 4)
               .arg(phaseArrayString(phaseScores, bestP));
    }

    return run;
}

CineMap::PhaseRun CineMap::solveSegment(SourceVideo& sv,
                                        int segStartField,
                                        int segEndField,
                                        const SegmentCaptureCache& cache,
                                        const std::vector<FrameMixedness>& mixedness)
{
    PhaseRun run;
    run.type        = PhaseRun::Type::Unknown;
    run.endField    = segEndField;
    run.confidence  = 0.0;
    run.phaseOffset = 0;

    if (!m_md || !m_disc || segStartField >= segEndField) {
        run.reason = "invalid-input";
        if (m_decisionTraceEnabled) {
            qInfo().noquote()
                << QString("CineMap decision: SEGMENT fields [%1..%2] result=unknown reason=invalid-input")
                   .arg(segStartField).arg(segEndField);
            qInfo().noquote()
                << QString("CineMap summary: SEGMENT fields [%1..%2] mixedness=skip geometry=skip final=unknown reason=invalid-input")
                   .arg(segStartField).arg(segEndField);
        }
        return run;
    }

    if (m_decisionTraceEnabled) {
        qInfo().noquote()
            << QString("CineMap decision: SEGMENT fields [%1..%2] begin mixednessFrames=%3")
               .arg(segStartField).arg(segEndField).arg(mixedness.size());
    }

    // 1. Mixedness signal — produces a per-phase score vector (mixVec).
    run = scanForPhaseRun(mixedness, segStartField, segEndField, cache);
    const QString mixednessSummary = phaseRunSummary(run);
    const auto mixVec = run.phaseScores;
    const bool mixInformative = run.phaseScoresInformative;
    const bool mixednessLocked = (run.type == PhaseRun::Type::Pulldown32);
    const int  mixedPhase = mixednessLocked ? run.phaseOffset : -1;
    const double mixedConf = mixednessLocked ? run.confidence : 0.0;

    // 2. Pattern harvest — the cheap fast-out. It checks only the ~2 sites per
    //    5-frame cycle that a candidate phase predicts, instead of every d=2 pair
    //    in the segment, and writes what it confirms to doplGang.
    //
    //    It runs on any phase mixedness can NAME, not only one it could lock.
    //    Checking a proposed phase cheaply is exactly how an unsure proposal should
    //    be adjudicated, and that is the case where a fast-out is worth the most.
    //    Previously this was gated on a lock, so the fast path could never fire when
    //    it was needed and brute force ran regardless.
    int patternCandidatePhase = mixedPhase;
    if (patternCandidatePhase < 0 && mixInformative) {
        int bestMix = -1;
        double bestMixScore = 0.0;
        for (int p = 0; p < 5; ++p) {
            if (mixVec[p] > bestMixScore) { bestMixScore = mixVec[p]; bestMix = p; }
        }
        patternCandidatePhase = bestMix;   // stays -1 if nothing scored positive
    }

    int patternPairs = 0;
    if (patternCandidatePhase >= 0) {
        patternPairs = harvestTwinsByPattern(sv, segStartField, segEndField,
                                             patternCandidatePhase, cache);
    }

    // 3. Brute force is the FALLBACK, not the default. It is the wholesale
    //    operation — every d=2 pair in the segment — so it runs only when pattern
    //    came up short of the ~1-twin-per-5-frames that 3:2 predicts. A pattern
    //    that found its twins has already established the cadence, and paying for
    //    the wholesale pass on top of it buys nothing.
    //
    //    Restricted to d=2: classifyTwinAC_strict requires hi==lo+2, so d=4/6
    //    cannot form geometry.
    const int segFramesForPattern = std::max(1, (segEndField - segStartField + 1) / 2);
    const int expectedPatternPairs = std::max(1, segFramesForPattern / 5);
    const bool patternSufficed =
        (patternPairs >= std::max(1, (expectedPatternPairs + 1) / 2));

    int harvestedEdges = 0;
    if (!patternSufficed) {
        harvestedEdges = static_cast<int>(
            harvestTwinEdges(sv, segStartField, segEndField, /*maxDist=*/2).size());
    }

    DgLock lock;
    QString geomRejectReason;
    (void)tryLockByDgGeometry(sv, segStartField, segEndField,
                              cache, lock, &geomRejectReason);
    const auto dgVec = lock.phaseScores;
    const bool dgInformative = lock.phaseScoresInformative;

    // 4. Evidence-additive election.
    //    We have two per-phase evidence vectors with different scales:
    //      - mixVec: raw mixedness count (unbounded above)
    //      - dgVec:  normalized A/C agreement (~[-0.25, 0.5])
    //    Max-normalize each (clamping negatives to 0 so phases with net
    //    disagreement contribute nothing), then sum. Highest score wins
    //    the segment. We never veto: a phase with both signals voting for
    //    it beats a phase one signal disfavors, but a strong single-signal
    //    candidate can still win if the other signal is silent.
    auto maxNormalize = [](const std::array<double,5>& v) {
        double mx = 0.0;
        for (double x : v) if (x > mx) mx = x;
        std::array<double,5> n = {0.0, 0.0, 0.0, 0.0, 0.0};
        if (mx > 0.0) {
            for (int i = 0; i < 5; ++i) n[i] = std::max(0.0, v[i]) / mx;
        }
        return n;
    };

    const auto mixN = mixInformative ? maxNormalize(mixVec) : std::array<double,5>{};
    const auto dgN  = dgInformative  ? maxNormalize(dgVec)  : std::array<double,5>{};

    std::array<double,5> combined = {0.0, 0.0, 0.0, 0.0, 0.0};
    int signalSources = 0;
    if (mixInformative) { for (int i = 0; i < 5; ++i) combined[i] += mixN[i]; signalSources++; }
    if (dgInformative)  { for (int i = 0; i < 5; ++i) combined[i] += dgN[i];  signalSources++; }

    // Pattern bonus at the phase pattern actually tested: confirmatory evidence at
    // predicted sites adds weight to that phase, calibrated against the
    // ~1-per-5-frames expected twin rate. Capped at 1.0 (same scale as one
    // normalized signal contribution).
    //
    // Keyed to patternCandidatePhase rather than to a locked mixedness phase, since
    // pattern now also runs on named-but-unlocked candidates. The bonus remains
    // one-sided — only the tested phase can earn it — which is why pattern is a
    // fast-out and not a vote: the election proper is mixVec + dgVec.
    if (patternCandidatePhase >= 0 && patternPairs > 0) {
        const double patternBonus =
            std::min(1.0, double(patternPairs) / double(expectedPatternPairs));
        combined[patternCandidatePhase] += patternBonus;
    }

    int bestP = -1;
    double bestC = -1e9;
    for (int p = 0; p < 5; ++p) {
        if (combined[p] > bestC) { bestC = combined[p]; bestP = p; }
    }
    double secondC = -1e9;
    for (int p = 0; p < 5; ++p) {
        if (p == bestP) continue;
        if (combined[p] > secondC) secondC = combined[p];
    }
    const double margin = (bestP != -1 && secondC > -1e8) ? (bestC - secondC) : bestC;

    // Margin floor: 0.15 per contributing signal source (same shape as the
    // per-detector margin gate). Two signals → 0.30, one → 0.15.
    constexpr double MARGIN_PER_SOURCE = 0.15;
    const double marginFloor = MARGIN_PER_SOURCE * std::max(1, signalSources);
    const bool hasWinner = (bestP != -1) && (bestC > 0.0) && (margin >= marginFloor);

    if (hasWinner) {
        run.type        = PhaseRun::Type::Pulldown32;
        run.phaseOffset = bestP;
        run.endField    = segEndField;
        run.reason.clear();
        // Confidence: 0.80 base + scaled by how much the margin exceeds the
        // floor (caps at 0.95). Both signals agreeing produces a strong margin.
        double conf = 0.80;
        if (marginFloor > 0.0) {
            conf += 0.15 * std::min(1.0, (margin - marginFloor) / marginFloor);
        }
        run.confidence = std::clamp(conf, 0.80, 0.95);
    } else {
        run.type = PhaseRun::Type::Unknown;
        run.confidence = 0.0;
        run.reason = (signalSources == 0) ? QString("no-signals")
                                          : QString("no-clear-combined-winner");
    }

    if (m_decisionTraceEnabled) {
        auto vecStr = [](const std::array<double,5>& v, int bestIdx) {
            QString out;
            for (int p = 0; p < 5; ++p) {
                if (p > 0) out += " ";
                out += QString("p%1=%2%3").arg(p)
                                          .arg(v[p], 0, 'f', 3)
                                          .arg(p == bestIdx ? "*" : "");
            }
            return out;
        };
        QString sources;
        if (mixInformative) sources += "mix";
        if (dgInformative)  sources += sources.isEmpty() ? "dg" : "+dg";
        if (patternCandidatePhase >= 0 && patternPairs > 0)
            sources += sources.isEmpty() ? "pattern" : "+pattern";
        if (sources.isEmpty()) sources = "none";

        qInfo().noquote()
            << QString("CineMap decision: SEGMENT_ELECT fields [%1..%2] sources=%3 patternPhase=%4 mixedConf=%5 patternPairs=%6/%17 patternSufficed=%18 bruteEdges=%7 mixN={%8} dgN={%9} combined={%10} bestPhase=%11 bestScore=%12 secondBest=%13 margin=%14 floor=%15 result=%16")
               .arg(segStartField).arg(segEndField)
               .arg(sources)
               .arg(patternCandidatePhase)
               .arg(mixedConf, 0, 'f', 3)
               .arg(patternPairs)
               .arg(harvestedEdges)
               .arg(vecStr(mixN, bestP))
               .arg(vecStr(dgN,  bestP))
               .arg(vecStr(combined, bestP))
               .arg(bestP)
               .arg(bestC, 0, 'f', 4)
               .arg(secondC > -1e8 ? secondC : 0.0, 0, 'f', 4)
               .arg(margin, 0, 'f', 4)
               .arg(marginFloor, 0, 'f', 4)
               .arg(hasWinner ? "lock" : (signalSources == 0 ? "no-signals" : "no-clear-combined-winner"))
               .arg(expectedPatternPairs)
               .arg(patternSufficed ? "yes(brute skipped)" : "no(brute ran)");

        qInfo().noquote()
            << QString("CineMap summary: SEGMENT fields [%1..%2] mixedness=%3 final=%4 conf=%5 source=evidence-additive(%6)")
               .arg(segStartField).arg(segEndField)
               .arg(mixednessSummary)
               .arg(phaseRunSummary(run))
               .arg(run.confidence, 0, 'f', 3)
               .arg(sources);
    }

    return run;
}

void CineMap::solveSegmentCine(SourceVideo& sv,
                               int segStartField,
                               int segEndField,
                               const SegmentCaptureCache& cache,
                               const std::vector<FrameMixedness>& mixedness)
{
    Q_UNUSED(sv);
    if (!m_md || !m_disc || segStartField >= segEndField) return;
    if (mixedness.empty()) return;

    FieldOrderPolicy fo;
    fo.reverse = m_disc->getReverseFieldOrder();

    // First valid frame index in this segment.
    int startFrameIdx = -1;
    for (int s = segStartField; s <= segEndField; ++s) {
        if (cache.validSeq(s)) { startFrameIdx = cache.cap[s].frameIndex; break; }
    }
    if (startFrameIdx < 0) return;

    // Convert a 0-based frame-index span to a field sequence span.
    auto frameRangeToFieldRange =
        [&](int frameStartIdx, int frameEndIdx,
            int& outFieldStart, int& outFieldEnd) -> bool
    {
        if (frameStartIdx < 0 || frameEndIdx < frameStartIdx) return false;
        if (frameEndIdx >= m_disc->getNumberOfFrames())        return false;

        // 1-based frameNumber = frameIdx + 1
        int fStart1 = m_disc->getFirstFieldNumber (frameStartIdx + 1);
        int fStart2 = m_disc->getSecondFieldNumber(frameStartIdx + 1);
        int fEnd1   = m_disc->getFirstFieldNumber (frameEndIdx   + 1);
        int fEnd2   = m_disc->getSecondFieldNumber(frameEndIdx   + 1);
        if (fStart1 < 1 || fStart2 < 1 || fEnd1 < 1 || fEnd2 < 1) return false;

        auto [tStartFirst, tStartSecond] = fo.temporalOrder(fStart1, fStart2);
        auto [tEndFirst,   tEndSecond]   = fo.temporalOrder(fEnd1,   fEnd2);
        (void)tStartSecond;

        outFieldStart = tStartFirst;
        outFieldEnd   = tEndSecond;
        return (outFieldStart >= 1 && outFieldEnd >= outFieldStart);
    };

    // -----------------------------------------------------------------
    // 1. Per-frame local phase preferences using a 5-frame sliding window.
    // -----------------------------------------------------------------
    struct PhaseSample {
        int    frameIndex = -1;
        int    bestPhase  = -1;
        double strength   = 0.0;  // separation between best and runner-up
    };

    std::vector<PhaseSample> series;
    series.reserve(mixedness.size());

    for (size_t i = 0; i < mixedness.size(); ++i) {
        const int frameIdx = mixedness[i].frameIndex;
        const int relFrame = frameIdx - startFrameIdx;

        int    bestP   = -1;
        double bestS   = -1e9;
        double secondS = -1e9;

        for (int p = 0; p < 5; ++p) {
            double score = 0.0;

            for (int k = -2; k <= 2; ++k) {
                const long long ii = static_cast<long long>(i) + k;
                if (ii < 0 || ii >= static_cast<long long>(mixedness.size())) continue;

                const int    pos = normalizePhase(static_cast<long long>(relFrame) + k + p, 5);
                const double val = mixedness[static_cast<size_t>(ii)].score;
                const bool shouldBeMixed = (pos == 1 || pos == 2);

                if (shouldBeMixed)
                    score += (k == 0 ? 1.5 : 1.0) * val;
                else
                    score -= 0.3 * val;
            }

            if (score > bestS)   { secondS = bestS; bestS = score; bestP = p; }
            else if (score > secondS) { secondS = score; }
        }

        series.push_back({frameIdx, bestP, bestS - secondS});
    }

    // -----------------------------------------------------------------
    // 2. Find stable phase runs and change-points.
    // -----------------------------------------------------------------
    constexpr double MIN_LOCAL_STRENGTH = 0.5;
    constexpr int    STABILITY_NEED     = 3;
    constexpr int    MIN_RUN_FRAMES     = 10;

    struct Run { int frameStart, frameEnd, phase; };
    std::vector<Run> runs;

    int currentPhase  = -1;
    int runStartFrame = -1;
    int runLength     = 0;
    int stablePhase   = -1;
    int stableCtr     = 0;

    for (size_t i = 0; i < series.size(); ++i) {
        const auto& s = series[i];

        if (s.bestPhase < 0 || s.strength < MIN_LOCAL_STRENGTH) {
            stablePhase = -1; stableCtr = 0; continue;
        }

        if (s.bestPhase == stablePhase) stableCtr++;
        else { stablePhase = s.bestPhase; stableCtr = 1; }

        if (stableCtr >= STABILITY_NEED) {
            if (currentPhase < 0) {
                currentPhase  = stablePhase;
                runStartFrame = s.frameIndex;
                runLength     = 1;
            } else if (stablePhase == currentPhase) {
                runLength++;
            } else {
                if (runLength >= MIN_RUN_FRAMES)
                    runs.push_back({runStartFrame, series[i - 1].frameIndex, currentPhase});
                currentPhase  = stablePhase;
                runStartFrame = s.frameIndex;
                runLength     = 1;
            }
        }
    }
    if (currentPhase >= 0 && runLength >= MIN_RUN_FRAMES)
        runs.push_back({runStartFrame, series.back().frameIndex, currentPhase});

    if (runs.empty()) return;

    // -----------------------------------------------------------------
    // 3. Insert cadence boundaries at run transitions and paint cadence.
    // -----------------------------------------------------------------
    auto setCadenceBoundaryAtFrame = [&](int frameIdx) {
        if (frameIdx < 0 || frameIdx >= m_disc->getNumberOfFrames()) return;
        if (m_disc->isPadded(frameIdx)) return;

        const int frameNumber = frameIdx + 1;
        int f1 = m_disc->getFirstFieldNumber (frameNumber);
        int f2 = m_disc->getSecondFieldNumber(frameNumber);
        if (f1 < 1 || f2 < 1) return;

        auto [tFirst, tSecond] = fo.temporalOrder(f1, f2);
        (void)tSecond;

        if (tFirst < 1 || tFirst > m_md->getNumberOfFields()) return;
        auto fld = m_md->getField(tFirst);
        if (!fld.cinemap.isEditBoundary) {
            fld.cinemap.isEditBoundary = true;
            m_md->updateField(fld, tFirst);
        }
    };

    for (size_t ri = 0; ri < runs.size(); ++ri) {
        const Run& r = runs[ri];

        int fs = 0, fe = 0;
        if (!frameRangeToFieldRange(r.frameStart, r.frameEnd, fs, fe)) continue;

        fs = std::max(fs, segStartField);
        fe = std::min(fe, segEndField);
        if (fs >= fe) continue;

        if (ri > 0) setCadenceBoundaryAtFrame(r.frameStart);

        constexpr double CINE_RUN_CONF = 0.80;
        applyCadenceToSegment(fs, fe,
                              /*isLock=*/true,
                              r.phase,
                              /*fillCid=*/CADENCE_UNKNOWN,
                              CINE_RUN_CONF,
                              cache);
    }
}

void CineMap::classifyAsInterlaced(int segStartField,
                                   int segEndField,
                                   const std::vector<FrameMixedness>& mixedness)
{
    if (!m_md) return;
    if (segStartField >= segEndField) return;
    if (mixedness.empty()) return;

    const int spanFields = segEndField - segStartField + 1;

    // Fraction of frames that actually comb. On lips this is a real question about
    // inter-field motion: the image's own vertical detail has already been masked
    // out per pixel, so a detailed progressive frame scores ~zero here where notch
    // scored whatever its edges amounted to.
    int highMixedFrames = 0;
    for (const auto& m : mixedness) {
        if (m.score > LIPS_COMB) {
            highMixedFrames++;
        }
    }
    const double fracHigh = (mixedness.empty())
        ? 0.0
        : static_cast<double>(highMixedFrames) / static_cast<double>(mixedness.size());

    // Require more than ~2/5 frames with significant mixedness
    if (fracHigh < 0.4) {
        // Not "busy" enough to confidently call interlaced
        return;
    }

    // At this point, mixedness suggests "lots of comb, no clear 3:2".
    // We assume genuine 59.94i. Paint cadenceId = -2 across the span
    // but only where cadenceId is still Unknown (-1).
    const int total = m_md->getNumberOfFields();
    int start = std::max(1, segStartField);
    int end   = std::min(total, segEndField);

    for (int s = start; s <= end; ++s) {
        auto f = m_md->getField(s);
        if (f.pad) continue;

        // Only paint where we don't already have a known cadence (film or otherwise).
        if (cadenceKnown(f.cinemap.cadenceId)) continue;

        f.cinemap.cadenceId = -2; // 59.94i
        // Confidence: not absolute, but stronger than "unsolved"
        if (m_cadenceConfidence[s] < 0.7)
            m_cadenceConfidence[s] = 0.7;
        f.cinemap.cadenceIndexPresumed = false;

        m_md->updateField(f, s);
    }
}

void CineMap::classifyAsProgressive(int segStartField,
                                    int segEndField,
                                    const std::vector<FrameMixedness>& mixedness)
{
    if (!m_md) return;
    if (segStartField >= segEndField) return;
    if (mixedness.empty()) return;

    // Fraction of frames that comb — the mirror of classifyAsInterlaced's test, and
    // deliberately a fraction rather than a maximum. A max-based test lets a single
    // frame veto a whole span, and the frame it usually trips on is the cut at the
    // shot boundary, whose two fields genuinely straddle different content. A shot of
    // unique progressive film frames should read ~zero on lips throughout its body
    // regardless of what happens at its ends.
    int combedFrames = 0;
    for (const auto& m : mixedness) {
        if (m.score > LIPS_COMB) combedFrames++;
    }
    const double fracCombed = mixedness.empty()
        ? 0.0
        : static_cast<double>(combedFrames) / static_cast<double>(mixedness.size());

    // Require the span to be overwhelmingly comb-free. The gap either side is wide:
    // classifyAsInterlaced needs >= 0.40 to call 59.94i.
    if (fracCombed > 0.10) {
        return;
    }

    // In addition, check that we don't see existing film structure (cadenceKnown)
    // across much of the span; if some fields are already film, leave them.
    const int total = m_md->getNumberOfFields();
    int start = std::max(1, segStartField);
    int end   = std::min(total, segEndField);

    int knownFilmCount = 0;
    int validCount = 0;
    for (int s = start; s <= end; ++s) {
        auto f = m_md->getField(s);
        if (f.pad) continue;
        validCount++;

        if (cadenceKnown(f.cinemap.cadenceId) && f.cinemap.cadenceId >= 0) {
            // 0+ is your film / pulldown domain; adjust if needed
            knownFilmCount++;
        }
    }

    // If we already solved most of it as film, don't try to override.
    if (validCount > 0 &&
        static_cast<double>(knownFilmCount) / static_cast<double>(validCount) > 0.2) {
        return;
    }

    // Now paint as 29.97p (-3) where cadence is still unknown.
    for (int s = start; s <= end; ++s) {
        auto f = m_md->getField(s);
        if (f.pad) continue;

        if (cadenceKnown(f.cinemap.cadenceId)) continue;

        f.cinemap.cadenceId = -3; // 29.97p progressive
        if (m_cadenceConfidence[s] < 0.7)
            m_cadenceConfidence[s] = 0.7;
        f.cinemap.cadenceIndexPresumed = false;

        m_md->updateField(f, s);
    }
}

void CineMap::detectAndEncodeInvertedCadenceRuns()
{
    if (!m_disc || !m_md || m_disc->isDiscPal()) return;

    const int totalFields = m_md->getNumberOfFields();
    if (totalFields < 4) return;

    // We need a capture cache to classify twins.  Build a fresh one covering
    // the full field range (same call as detectCadence uses).
    const int hardMax = computeHardMaxField();
    const SegmentCaptureCache cache = buildCaptureCache(hardMax);

    // -------------------------------------------------------------------------
    // Pass 1: for every reciprocal doplGang pair that classifies as A- or
    // C-type, record a vote: +1 = normal, -1 = inverted.
    // We index votes by the TBC frame that owns the spare field so we can
    // apply corrections run-by-run later.
    // -------------------------------------------------------------------------

    // votes[frameIndex] accumulates evidence for that frame's capture position.
    std::vector<int> votes(static_cast<size_t>(m_disc->getNumberOfFrames()), 0);

    for (int s = 1; s <= hardMax; ++s) {
        if (!m_doplGang[s].has_value()) continue;

        int mate = m_doplGang[s].value();
        if (mate <= s || mate > totalFields) continue; // process each pair once

        auto f  = m_md->getField(s);
        auto fm = m_md->getField(mate);
        if (m_doplGang[mate] != s) continue;
        if (f.isFirstField != fm.isFirstField) continue; // twins are same parity

        TwinACInfo ac = classifyTwinAC_strict(s, mate, cache);
        if (ac.role == TwinACRole::Unknown) continue;

        // Determine which field is the spare and what parity it has.
        const int spareSeq = ac.spareSeq;
        if (spareSeq < 1 || spareSeq > totalFields) continue;

        const bool spareIsFirst = m_md->getField(spareSeq).isFirstField;

        // Normal dominance expectation per role:
        //   AType spare (index 2) → stored-first → isFirstField == true
        //   CType spare (index 5) → stored-second → isFirstField == false
        bool expectIsFirst = (ac.role == TwinACRole::AType);
        bool isInverted    = (spareIsFirst != expectIsFirst);

        // Attribute the vote to the frame that contains the spare.
        if (!cache.validSeq(spareSeq)) continue;
        int frameIdx = cache.cap[spareSeq].frameIndex;
        if (frameIdx >= 0 && frameIdx < static_cast<int>(votes.size())) {
            votes[frameIdx] += isInverted ? -1 : +1;
        }
    }

    // -------------------------------------------------------------------------
    // Pass 2: walk the field sequence and find contiguous runs of cadence-known
    // fields.  For each run, sum the votes of the frames it covers.
    // If the run is net-inverted, shift every cadenceId by
    // CADENCE_NTSC_INVERTED_OFFSET (but only if it is currently in the normal
    // domain, i.e. cadenceId < CADENCE_NTSC_INVERTED_OFFSET).
    // -------------------------------------------------------------------------

    int runsFlipped = 0;
    int fieldsFlipped = 0;

    int runStart = -1;

    auto processRun = [&](int start, int end) {
        if (start < 0 || end < start) return;

        // Tally votes for all frames in this run.
        int voteSum = 0;
        for (int s = start; s <= end; ++s) {
            if (!cache.validSeq(s)) continue;
            int fi = cache.cap[s].frameIndex;
            if (fi >= 0 && fi < static_cast<int>(votes.size())) {
                voteSum += votes[fi];
            }
        }

        // Require a net-inverted signal (negative sum) AND at least one piece
        // of direct evidence (|voteSum| > 0) before flipping anything.
        if (voteSum >= 0) return;

        // Flip: add offset to every normal-domain cadenceId in this run.
        runsFlipped++;
        for (int s = start; s <= end; ++s) {
            auto fld = m_md->getField(s);
            if (!cadenceKnown(fld.cinemap.cadenceId)) continue;
            if (cadenceIsInverted(fld.cinemap.cadenceId)) continue; // already inverted

            fld.cinemap.cadenceId += CADENCE_NTSC_INVERTED_OFFSET;
            m_md->updateField(fld, s);
            fieldsFlipped++;
        }
    };

    for (int s = 1; s <= hardMax; ++s) {
        auto fld = m_md->getField(s);
        bool inRun = cadenceKnown(fld.cinemap.cadenceId) && !fld.pad;

        if (inRun) {
            if (runStart < 0) runStart = s;
        } else {
            if (runStart >= 0) {
                processRun(runStart, s - 1);
                runStart = -1;
            }
        }
    }
    if (runStart >= 0) processRun(runStart, hardMax);

    if (runsFlipped > 0) {
        qInfo() << "detectAndEncodeInvertedCadenceRuns: flipped"
                << fieldsFlipped << "field(s) across"
                << runsFlipped << "run(s) to inverted domain.";
    } else {
        qInfo() << "detectAndEncodeInvertedCadenceRuns: all runs normal dominance.";
    }
}

double CineMap::verifyPhaseByTwins(SourceVideo& sv,
                                   int segStart,
                                   int segEnd,
                                   int phaseOffset,
                                   const SegmentCaptureCache& cache)
{
    if (!m_md || !m_disc) return -1.0;

    const NoiseFloor& nf = calibrateTwinFloor(sv);
    if (!nf.valid) return -1.0;

    int startFrameIdx = -1, fStart = -1, fEnd = -1;
    for (int s = segStart; s <= segEnd; ++s) {
        if (cache.validSeq(s)) { startFrameIdx = fStart = cache.cap[s].frameIndex; break; }
    }
    for (int s = segEnd; s >= segStart; --s) {
        if (cache.validSeq(s)) { fEnd = cache.cap[s].frameIndex; break; }
    }
    if (startFrameIdx < 0 || fEnd < 0) return -1.0;

    const auto& vp = m_md->getVideoParameters();
    const double floorIre = nf.ire * FLOOR_MULT_RECALL;

    int sites = 0, hits = 0;

    // The same site geometry pattern harvest uses: within the 5-frame cycle, pos 0
    // carries the AA first-field twin and pos 2 the BC second-field twin.
    for (int fi = fStart; fi <= fEnd; ++fi) {
        if (m_disc->isPadded(fi)) continue;
        if (fi + 1 > fEnd || m_disc->isPadded(fi + 1)) continue;

        int pos = (fi - startFrameIdx + phaseOffset) % 5;
        if (pos < 0) pos += 5;
        if (pos != 0 && pos != 2) continue;

        const int a = (pos == 0) ? m_disc->getFirstFieldNumber(fi + 1)
                                 : m_disc->getSecondFieldNumber(fi + 1);
        const int b = (pos == 0) ? m_disc->getFirstFieldNumber(fi + 2)
                                 : m_disc->getSecondFieldNumber(fi + 2);
        if (a < 1 || b < 1) continue;
        if (a < segStart || a > segEnd || b < segStart || b > segEnd) continue;

        const TwinDemod m = demodTwinCached(sv, a, b, vp.fieldWidth, vp.fieldHeight);
        if (!m.valid) continue;

        sites++;
        if (m.grainIre < floorIre) hits++;
    }

    if (m_decisionTraceEnabled) {
        qInfo().noquote()
            << QString("CineMap decision: PHASE_VERIFY fields [%1..%2] phase=%3 frames=[%4..%5] sites=%6 hits=%7 floor=%8")
               .arg(segStart).arg(segEnd).arg(phaseOffset)
               .arg(fStart).arg(fEnd).arg(sites).arg(hits)
               .arg(floorIre, 0, 'f', 4);
    }

    if (sites < 2) return -1.0;   // no power: darkness, not a rejection
    return static_cast<double>(hits) / sites;
}

int CineMap::healContinuity(SourceVideo& sv,
                            std::vector<SegmentResult>& segments,
                            const SegmentCaptureCache& cache)
{
    int changes = 0;
    if (segments.empty()) return 0;

    auto getFrameIdx = [&](int fieldSeq) -> int {
        if (cache.validSeq(fieldSeq)) return cache.cap[fieldSeq].frameIndex;
        return -1;
    };

    // Helper: demote entire segment to allow a new solution to repaint it
    auto demoteSegmentRange = [&](SegmentResult& seg, double newMaxConf) {
        demoteCadenceRange(seg.startField, seg.endField, newMaxConf);
        if (seg.run.confidence > newMaxConf) {
            seg.run.confidence = newMaxConf;
        }
    };

    // ---------------------------
    // Phase 1: segment-to-segment continuity (original CASE 1–4)
    // ---------------------------
    for (size_t i = 0; i < segments.size() - 1; ++i) {
        SegmentResult& curr = segments[i];
        SegmentResult& next = segments[i + 1];

        // Ensure contiguity (sanity check)
        if (curr.endField + 1 != next.startField) continue;

        bool currLocked = (curr.run.type == PhaseRun::Type::Pulldown32);
        bool nextLocked = (next.run.type == PhaseRun::Type::Pulldown32);

        // -----------------------------------------------------------------
        // CASE 1: Seamless Stitching (Both Locked)
        // If phases line up perfectly, just boost confidence.
        // -----------------------------------------------------------------
        if (currLocked && nextLocked) {
            int startA = getFrameIdx(curr.startField);
            int startB = getFrameIdx(next.startField);
            if (startA < 0 || startB < 0) continue;

            // Project A's phase to B's start
            int expectedNextPhase =
                (curr.run.phaseOffset + (startB - startA)) % 5;
            if (expectedNextPhase < 0) expectedNextPhase += 5;

            if (next.run.phaseOffset == expectedNextPhase) {
                const double BOOST = 0.95;
                curr.run.confidence = std::max(curr.run.confidence, BOOST);
                next.run.confidence = std::max(next.run.confidence, BOOST);
                changes++;
            }
        }

        // -----------------------------------------------------------------
        // CASE 2: Rescue Short Segment (Curr is Short/Unknown, Next is Locked)
        // Try to extend 'Next' BACKWARDS into 'Curr'.
        // -----------------------------------------------------------------
        if (!currLocked && nextLocked) {
            int lenFields = curr.endField - curr.startField + 1;
            if (lenFields < 60) { // Only heal short gaps (< 1 sec)
                int startA = getFrameIdx(curr.startField);
                int startB = getFrameIdx(next.startField);
                if (startA < 0 || startB < 0) continue;

                int projectedPhase =
                    (next.run.phaseOffset - (startB - startA)) % 5;
                if (projectedPhase < 0) projectedPhase += 5;

                // Verify the projection against the twin sites it names. Where that
                // test has power it decides, because it checks the actual claim; the
                // mixedness fit only asks whether the phase looks plausible, and it
                // was the criterion that let cadences extend into non-film. Where it
                // has no power the span is dark, and darkness presumes — that is what
                // solve-in-the-dark is for — so the fit still carries it.
                const double twinFit = verifyPhaseByTwins(sv,
                                                          curr.startField,
                                                          curr.endField,
                                                          projectedPhase,
                                                          cache);

                double fitScore =
                    scoreSpecificPhase(curr.mixedness,
                                       projectedPhase,
                                       curr.startField,
                                       cache);

                const bool accept = (twinFit >= 0.0) ? (twinFit >= PHASE_VERIFY_MIN)
                                                     : (fitScore > 0.15);

                if (accept) {
                    qInfo() << "Healer: Back-projected phase"
                            << projectedPhase
                            << "into segment" << (i + 1)
                            << (twinFit >= 0.0 ? "(twins:" : "(mixedness fit:")
                            << (twinFit >= 0.0 ? twinFit : fitScore) << ")";

                    demoteSegmentRange(curr, 0.4);

                    curr.run.type        = PhaseRun::Type::Pulldown32;
                    curr.run.phaseOffset = projectedPhase;
                    curr.run.confidence  = 0.85; // Healed!
                    changes++;
                    currLocked = true;
                }
            }
        }

        // -----------------------------------------------------------------
        // CASE 3: Rescue Short Segment (Curr is Locked, Next is Short/Unknown)
        // Try to extend 'Curr' FORWARDS into 'Next'.
        // -----------------------------------------------------------------
        if (currLocked && !nextLocked) {
            int lenFields = next.endField - next.startField + 1;
            if (lenFields < 60) {
                int startA = getFrameIdx(curr.startField);
                int startB = getFrameIdx(next.startField);
                if (startA < 0 || startB < 0) continue;

                int projectedPhase =
                    (curr.run.phaseOffset + (startB - startA)) % 5;
                if (projectedPhase < 0) projectedPhase += 5;

                // Same rule as the back-projection: the twin sites the phase names
                // decide where they can be measured, mixedness presumes where dark.
                const double twinFit = verifyPhaseByTwins(sv,
                                                          next.startField,
                                                          next.endField,
                                                          projectedPhase,
                                                          cache);

                double fitScore =
                    scoreSpecificPhase(next.mixedness,
                                       projectedPhase,
                                       next.startField,
                                       cache);

                const bool accept = (twinFit >= 0.0) ? (twinFit >= PHASE_VERIFY_MIN)
                                                     : (fitScore > 0.15);

                if (accept) {
                    qInfo() << "Healer: Forward-projected phase"
                            << projectedPhase
                            << "into segment" << (i + 2)
                            << (twinFit >= 0.0 ? "(twins:" : "(mixedness fit:")
                            << (twinFit >= 0.0 ? twinFit : fitScore) << ")";

                    demoteSegmentRange(next, 0.4);

                    next.run.type        = PhaseRun::Type::Pulldown32;
                    next.run.phaseOffset = projectedPhase;
                    next.run.confidence  = 0.85;
                    changes++;
                    nextLocked = true;
                }
            }
        }

        // -----------------------------------------------------------------
        // CASE 4: Bridge Gap (Curr=Locked, Middle=Short/Unknown, Next=Locked)
        // Look ahead to i+2; if the far segment matches the phase projection
        // from curr, we treat the middle as a dropout/glitch and bridge it.
        // -----------------------------------------------------------------
        if (currLocked && !nextLocked && (i + 2) < segments.size()) {
            SegmentResult& far = segments[i + 2];
            bool farLocked = (far.run.type == PhaseRun::Type::Pulldown32);
            
            if (farLocked) {
                int startA = getFrameIdx(curr.startField);
                int startC = getFrameIdx(far.startField);
                if (startA < 0 || startC < 0) continue;
                
                int expectedFarPhase =
                    (curr.run.phaseOffset + (startC - startA)) % 5;
                if (expectedFarPhase < 0) expectedFarPhase += 5;
                
                if (far.run.phaseOffset == expectedFarPhase) {
                    int startB = getFrameIdx(next.startField);
                    if (startB < 0) continue;

                    int bridgePhase =
                        (curr.run.phaseOffset + (startB - startA)) % 5;
                    if (bridgePhase < 0) bridgePhase += 5;
                    
                    qInfo() << "Healer: Bridged gap segment"
                            << (i + 2) << "between locked neighbors.";

                    demoteSegmentRange(next, 0.4);

                    next.run.type        = PhaseRun::Type::Pulldown32;
                    next.run.phaseOffset = bridgePhase;
                    next.run.confidence  = 0.90; // Bridge is very trustworthy
                    changes++;
                    nextLocked = true;
                }
            }
        }
    }

    // ---------------------------
    // Phase 2: Intra-segment steady-cadence enforcement
    // ---------------------------
    FieldOrderPolicy fo;
    fo.reverse = m_disc->getReverseFieldOrder();

    auto enforceWithinSegment =
        [&](int segStartField, int segEndField, int phaseOffset) -> int
    {
        if (segStartField >= segEndField) return 0;

        // Find anchor frame index (same as applyCadenceToSegment)
        int startFrameIdx = -1;
        for (int s = segStartField; s <= segEndField; ++s) {
            if (cache.validSeq(s)) {
                startFrameIdx = cache.cap[s].frameIndex;
                break;
            }
        }
        if (startFrameIdx < 0) return 0;

        auto predictCid = [&](int seq) -> int {
            if (!cache.validSeq(seq)) return CADENCE_UNKNOWN;

            const int frameIdx = cache.cap[seq].frameIndex;
            if (frameIdx < 0 || frameIdx >= m_disc->getNumberOfFrames()) return CADENCE_UNKNOWN;
            if (m_disc->isPadded(frameIdx)) return CADENCE_UNKNOWN;

            const int relFrame = frameIdx - startFrameIdx;
            int pos = normalizePhase((long long)relFrame + (long long)phaseOffset, 5);

            auto [cidFirst, cidSecond] = fo.cavCadenceIdsForFrameInGroup(pos);

            int f1 = m_disc->getFirstFieldNumber(frameIdx + 1);
            int f2 = m_disc->getSecondFieldNumber(frameIdx + 1);
            if (f1 < 1 || f2 < 1) return CADENCE_UNKNOWN;

            auto [tFirst, tSecond] = fo.temporalOrder(f1, f2);

            if (seq == tFirst)  return cidFirst;
            if (seq == tSecond) return cidSecond;
            return CADENCE_UNKNOWN;
        };

        struct MismatchSpan { int startSeq = -1; int endSeq = -1; };
        std::vector<MismatchSpan> spans;
        bool inSpan = false;
        int spanStart = -1;

        // Find contiguous regions where existing cadenceId disagrees with predicted
        for (int s = segStartField; s <= segEndField; ++s) {
            auto f = m_md->getField(s);
            if (f.pad || !cadenceKnown(f.cinemap.cadenceId)) {
                if (inSpan) {
                    spans.push_back({spanStart, s - 1});
                    inSpan = false;
                }
                continue;
            }

            int expectedCid = predictCid(s);
            bool mismatch = (expectedCid != CADENCE_UNKNOWN &&
                             expectedCid != f.cinemap.cadenceId);

            if (mismatch) {
                if (!inSpan) {
                    inSpan = true;
                    spanStart = s;
                }
            } else {
                if (inSpan) {
                    spans.push_back({spanStart, s - 1});
                    inSpan = false;
                }
            }
        }
        if (inSpan) {
            spans.push_back({spanStart, segEndField});
        }

        if (spans.empty()) return 0;

        int fieldsFixed = 0;
        const int MAX_SPAN_FIELDS = 12;  // at most ~0.5 sec

        for (const auto& span : spans) {
            const int s0 = span.startSeq;
            const int s1 = span.endSeq;
            const int spanLen = s1 - s0 + 1;
            if (spanLen > MAX_SPAN_FIELDS) continue;

            bool leftOk  = false;
            bool rightOk = false;

            if (s0 > segStartField) {
                int leftSeq = s0 - 1;
                auto lf = m_md->getField(leftSeq);
                if (!lf.pad && cadenceKnown(lf.cinemap.cadenceId)) {
                    int exp = predictCid(leftSeq);
                    if (exp != CADENCE_UNKNOWN && exp == lf.cinemap.cadenceId) {
                        leftOk = true;
                    }
                }
            }

            if (s1 < segEndField) {
                int rightSeq = s1 + 1;
                auto rf = m_md->getField(rightSeq);
                if (!rf.pad && cadenceKnown(rf.cinemap.cadenceId)) {
                    int exp = predictCid(rightSeq);
                    if (exp != CADENCE_UNKNOWN && exp == rf.cinemap.cadenceId) {
                        rightOk = true;
                    }
                }
            }

            if (!leftOk && !rightOk) continue;

            for (int s = s0; s <= s1; ++s) {
                int expCid = predictCid(s);
                if (expCid == CADENCE_UNKNOWN) continue;

                auto f = m_md->getField(s);
                f.cinemap.cadenceId = expCid;
                if (m_cadenceConfidence[s] < 0.8) {
                    m_cadenceConfidence[s] = 0.8;
                }
                f.cinemap.cadenceIndexPresumed = false;

                // Clear internal isEditBoundary; we want cadence monotone inside segment.
            /* Whitelist needs to have the last word, no after the fact removals    
            if (f.cinemap.isEditBoundary && s != segStartField && s != segEndField) {
                    f.cinemap.isEditBoundary = false;
                }*/

                m_md->updateField(f, s);
                fieldsFixed++;
            }
        }

        return fieldsFixed;
    };

    // Intra-segment pass over all segments
    for (const auto& seg : segments) {
        if (seg.run.type != PhaseRun::Type::Pulldown32) continue;
        if (seg.run.confidence < 0.7) continue; // only strong film segments

        int fixed = enforceWithinSegment(seg.startField,
                                         seg.endField,
                                         seg.run.phaseOffset);
        changes += fixed;
    }

    if (changes > 0) {
        m_disc->refreshFrameCache();
    }

    return changes;
}

int CineMap::enforceSteadyCadenceAcrossBoundaries(int maxSpanFields)
{
    if (!m_md) return 0;
    const int total = m_md->getNumberOfFields();
    if (total < 3) return 0;

    int fixed = 0;

    // Project a cadenceId from srcSeq to targetSeq by arithmetic step.
    // The 10-position cycle is preserved; inversion domain is inherited.
    auto predictCid = [](int srcSeq, int targetSeq, int srcCid) -> int {
        if (!cadenceKnown(srcCid)) return CADENCE_UNKNOWN;
        const int delta   = targetSeq - srcSeq;
        const bool inv    = cadenceIsInverted(srcCid);
        int normIdx = cadenceIndex(srcCid);
        normIdx = ((normIdx + delta) % CADENCE_NTSC_CYCLE + CADENCE_NTSC_CYCLE)
                  % CADENCE_NTSC_CYCLE;
        return (inv ? CADENCE_NTSC_INVERTED_OFFSET : 0) + normIdx;
    };

    for (int i = 2; i <= total - 1; ++i) {
        if (!m_md->getField(i).cinemap.isEditBoundary) continue;

        // Left anchor: field immediately before the boundary.
        const int leftSeq = i - 1;
        auto fL = m_md->getField(leftSeq);
        if (!cadenceKnown(fL.cinemap.cadenceId) || m_cadenceConfidence[leftSeq] < 0.8) continue;

        // Right anchor: first high-confidence field within maxSpanFields.
        int rightSeq = -1;
        for (int j = i; j <= std::min(total, i + maxSpanFields); ++j) {
            auto fR = m_md->getField(j);
            if (cadenceKnown(fR.cinemap.cadenceId) && m_cadenceConfidence[j] >= 0.8) {
                rightSeq = j;
                break;
            }
        }
        if (rightSeq < 0) continue;

        // Both anchors must project onto each other with zero error.
        auto fR = m_md->getField(rightSeq);
        if (predictCid(leftSeq, rightSeq, fL.cinemap.cadenceId) != fR.cinemap.cadenceId) continue;

        // Fill the gap between the two anchors.
        for (int s = i; s < rightSeq; ++s) {
            auto fm = m_md->getField(s);
            if (fm.pad) continue;

            const int cid = predictCid(leftSeq, s, fL.cinemap.cadenceId);
            if (!cadenceKnown(cid)) continue;

            if (!cadenceKnown(fm.cinemap.cadenceId) ||
                m_cadenceConfidence[s] < m_cadenceConfidence[leftSeq]) {
                fm.cinemap.cadenceId             = cid;
                fm.cinemap.cadenceIndexPresumed  = true;
                m_cadenceConfidence[s]           = std::max(m_cadenceConfidence[s], 0.7);
                m_md->updateField(fm, s);
                ++fixed;
            }
            // isEditBoundary is intentionally not cleared.
        }

        i = rightSeq; // skip ahead past the filled span
    }

    if (fixed > 0)
        m_disc->refreshFrameCache();

    return fixed;
}

int CineMap::frameIndexForField(int seq) const
{
    if (!m_disc || seq < 1) return -1;

    const int nFrames = m_disc->getNumberOfFrames();
    for (int fi = 0; fi < nFrames; ++fi) {
        // frameNumber is 1-based
        if (m_disc->getFirstFieldNumber(fi + 1)  == seq ||
            m_disc->getSecondFieldNumber(fi + 1) == seq) {
            return fi;
        }
    }
    return -1;
}

int CineMap::fieldForFrame(int frameIdx) const
{
    if (!m_disc || frameIdx < 0 || frameIdx >= m_disc->getNumberOfFrames())
        return -1;
    return m_disc->getFirstFieldNumber(frameIdx + 1);
}


void CineMap::demoteCadenceRange(int startSeq, int endSeq, double newMaxConf)
{
    if (!m_md) return;
    int total = m_md->getNumberOfFields();
    startSeq = std::max(1, startSeq);
    endSeq   = std::min(total, endSeq);
    for (int i = startSeq; i <= endSeq; ++i) {
        auto f = m_md->getField(i);
        if (!cadenceKnown(f.cinemap.cadenceId)) continue;
        if (m_cadenceConfidence[i] > newMaxConf) {
            m_cadenceConfidence[i] = newMaxConf;
            m_md->updateField(f, i);
        }
    }
}

void CineMap::promoteCadenceRange(int startSeq, int endSeq, double newConf)
{
    if (!m_md) return;
    int total = m_md->getNumberOfFields();
    startSeq = std::max(1, startSeq);
    endSeq   = std::min(total, endSeq);
    for (int i = startSeq; i <= endSeq; ++i) {
        auto f = m_md->getField(i);
        if (!cadenceKnown(f.cinemap.cadenceId)) continue;
        if (m_cadenceConfidence[i] < newConf) {
            m_cadenceConfidence[i] = newConf;
            m_md->updateField(f, i);
        }
    }
}

void CineMap::applyCadenceToSegment(int segStart,
                                    int segEnd,
                                    bool isLock,
                                    int phaseOffset,
                                    int fillCid,
                                    double finalConf,
                                    const SegmentCaptureCache& cache)
{
    if (!m_disc || !m_md) return;

    // Find first valid frame index in segment (handles padded/weird starts).
    int startFrameIdx = -1;
    for (int s = segStart; s <= segEnd; ++s) {
        if (cache.validSeq(s)) { startFrameIdx = cache.cap[s].frameIndex; break; }
    }

    FieldOrderPolicy fo;
    fo.reverse = m_disc->getReverseFieldOrder();

    const int totalFields = m_md->getNumberOfFields();

    for (int i = segStart; i <= segEnd; ++i) {
        if (i < 1 || i > totalFields) continue;

        auto fld = m_md->getField(i);
        if (fld.pad) continue;

        // Don't overwrite a higher-confidence lock already on this field.
        if (cadenceKnown(fld.cinemap.cadenceId) && m_cadenceConfidence[i] > finalConf) continue;

        if (isLock) {
            // Need a valid frame reference for 3:2 position arithmetic.
            if (!cache.validSeq(i) || startFrameIdx < 0) continue;

            const int frameIdx = cache.cap[i].frameIndex;
            const int relFrame = frameIdx - startFrameIdx;

            // Map into 0..4 position: 0=AA, 1=AB, 2=BC, 3=CC, 4=DD
            const int pos = normalizePhase(static_cast<long long>(relFrame) + phaseOffset, 5);

            // cavCadenceIdsForFrameInGroup returns (cidFirst, cidSecond) in
            // temporal order.  temporalOrder maps stored (f1,f2) → (first,second).
            auto [c1, c2] = fo.cavCadenceIdsForFrameInGroup(pos);

            // frameIdx is 0-based; getFirstFieldNumber / getSecondFieldNumber
            // expect 1-based frameNumber.
            const int frameNumber = frameIdx + 1;
            int f1 = m_disc->getFirstFieldNumber(frameNumber);
            int f2 = m_disc->getSecondFieldNumber(frameNumber);
            auto [t1, t2] = fo.temporalOrder(f1, f2);

            if      (i == t1) fld.cinemap.cadenceId = c1;
            else if (i == t2) fld.cinemap.cadenceId = c2;

        } else if (fillCid != CADENCE_UNKNOWN) {
            fld.cinemap.cadenceId = fillCid;
        }

        if (fld.cinemap.cadenceId != CADENCE_UNKNOWN || fillCid != CADENCE_UNKNOWN) {
            m_cadenceConfidence[i]         = finalConf;
            fld.cinemap.cadenceIndexPresumed = false;
            m_md->updateField(fld, i);
        }
    }
}

void CineMap::reconcileDoplGangWithCadence()
{
    if (!m_md) return;
    const int total = m_md->getNumberOfFields();
    if (total <= 0) return;

    auto setReciprocal = [&](int a, int b) {
        if (a < 1 || b < 1 || a > total || b > total) return;
        m_doplGang[a] = b;
        m_doplGang[b] = a;
    };

    for (int i = 1; i <= total; ++i) {
        auto f = m_md->getField(i);
        if (!cadenceKnown(f.cinemap.cadenceId)) continue;

        const int idx = cadenceIndex(f.cinemap.cadenceId);
        int partnerSeq  = -1;
        int expectedIdx = -1;

        // Only process the definitional half of each pair to avoid writing
        // each link twice (the spare will be covered when we reach it, but
        // setReciprocal is idempotent so a double-write is harmless).
        switch (idx) {
            case 0: partnerSeq = i + 2; expectedIdx = 2; break; // Adef  → Aspare
            case 5: partnerSeq = i + 2; expectedIdx = 7; break; // Cspare → Cdef
            default: continue;
        }

        if (partnerSeq < 1 || partnerSeq > total) continue;

        auto g = m_md->getField(partnerSeq);
        if (!cadenceKnown(g.cinemap.cadenceId))               continue;
        if (cadenceIndex(g.cinemap.cadenceId) != expectedIdx) continue;
        if (f.isFirstField != g.isFirstField)                 continue; // must be same parity

        char lf = cadenceFilmLetter(f.cinemap.cadenceId);
        char lg = cadenceFilmLetter(g.cinemap.cadenceId);
        if (!(lf == lg && (lf == 'A' || lf == 'C')))  continue;

        setReciprocal(i, partnerSeq);
    }
}

int CineMap::assignPulldownRoles()
{
    if (!m_md) return 0;

    int count = 0;
    int total = m_md->getNumberOfFields();

    for (int i = 1; i <= total; ++i) {
        auto f = m_md->getField(i);
        if (!cadenceKnown(f.cinemap.cadenceId)) continue;

        if (isDefinitionalRole(f.cinemap.cadenceId)) {
            f.cinemap.pulldownRole = "definitional";
            count++;
        } else if (isSpareRole(f.cinemap.cadenceId)) {
            f.cinemap.pulldownRole = "spare";
            count++;
        } else {
            f.cinemap.pulldownRole = "";
        }
        m_md->updateField(f, i);
    }

    return count;
}
