/******************************************************************************
 * CadenceAssembler regression tests
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 Joseph Burns
 ******************************************************************************/

#include <cstdio>
#include <cstdlib>

#include <QVector>

#include "cadenceassembler.h"

namespace {

void check(bool condition, const char *message)
{
    if (condition) return;
    std::fprintf(stderr, "testcadenceassembler: %s\n", message);
    std::abort();
}

SourceField makeField(int cadenceId)
{
    SourceField field;
    field.field.seqNo = cadenceId + 1;
    field.field.isFirstField = ((cadenceId & 1) == 0);
    field.field.cinemap.inUse = true;
    field.field.cinemap.cadenceId = cadenceId;
    field.field.cinemap.isEditBoundary = false;
    field.capturePartnerSeqNo = (cadenceId & 1)
        ? field.field.seqNo - 1
        : field.field.seqNo + 1;
    return field;
}

void testLockedCycleKeepsMixedFramePartnersAvailable()
{
    LdDecodeMetaData::VideoParameters videoParameters;
    CadenceAssembler::Configuration configuration;

    QVector<qint32> releasedToBaseline;
    CadenceAssembler assembler(
        videoParameters,
        configuration,
        [&](qint32 seqNo) { releasedToBaseline.push_back(seqNo); });

    QVector<SourceField> fields;
    for (int cadenceId = 0; cadenceId < 10; ++cadenceId) {
        fields.push_back(makeField(cadenceId));
    }

    assembler.push(fields);
    assembler.flush();
    const QVector<CadenceAssembler::WorkItem> work = assembler.popWork();

    // The A-spare and C-spare release their original mixed capture frames to
    // baseline.  Their B-field capture partners must nevertheless remain
    // available to form reconstructed B1/B2.
    check(releasedToBaseline == QVector<qint32>({3, 6}),
          "only A-spare and C-spare should release capture frames");
    check(work.size() == 4, "locked cycle should emit four film frames");

    check(work[0].filmLabel == 'A', "first film frame should be A");
    check(work[0].f1.field.cinemap.cadenceId == 0, "A top should be Adef");
    check(work[0].f2.field.cinemap.cadenceId == 1, "A bottom should be Acomp");

    check(work[1].filmLabel == 'B', "second film frame should be B");
    check(work[1].f1.field.cinemap.cadenceId == 4, "B top should be B2");
    check(work[1].f2.field.cinemap.cadenceId == 3, "B bottom should be B1");

    check(work[2].filmLabel == 'C', "third film frame should be C");
    check(work[2].f1.field.cinemap.cadenceId == 6, "C top should be Ccomp");
    check(work[2].f2.field.cinemap.cadenceId == 7, "C bottom should be Cdef");

    check(work[3].filmLabel == 'D', "fourth film frame should be D");
    check(work[3].f1.field.cinemap.cadenceId == 8, "D top should be D1");
    check(work[3].f2.field.cinemap.cadenceId == 9, "D bottom should be D2");
}

void testPassthroughPreservesIdentityButForbidsFrameComb()
{
    LdDecodeMetaData::VideoParameters videoParameters;
    CadenceAssembler::Configuration configuration;
    configuration.noCinemap = true;

    CadenceAssembler assembler(videoParameters, configuration);
    assembler.push(QVector<SourceField>({makeField(4), makeField(5)}));
    assembler.flush();

    const QVector<CadenceAssembler::WorkItem> work = assembler.popWork();
    check(work.size() == 1, "passthrough pair should emit one work item");
    check(work[0].kind == CadenceAssembler::WorkItem::Kind::PassthroughFrame,
          "no-PA work should be passthrough");
    check(work[0].f1.field.cinemap.cadenceId == 4,
          "passthrough should preserve top cadence identity");
    check(work[0].f2.field.cinemap.cadenceId == 5,
          "passthrough should preserve bottom cadence identity");
    check(!work[0].f1.allowProgressiveFrameRegime &&
              !work[0].f2.allowProgressiveFrameRegime,
          "passthrough cadence identity must not grant progressive FVF permission");
}

void testCutTruncatedACompSpareUsesFrameRegime()
{
    LdDecodeMetaData::VideoParameters videoParameters;
    CadenceAssembler::Configuration configuration;
    CadenceAssembler assembler(videoParameters, configuration);

    // The edit discarded A-def (slot 0).  The scene begins with the valid
    // A-comp/A-spare pair (slots 1,2), which straddles a capture-frame edge.
    SourceField comp = makeField(1);
    SourceField spare = makeField(2);
    comp.field.seqNo = 101;
    spare.field.seqNo = 102;
    comp.capturePartnerSeqNo = 100;
    spare.capturePartnerSeqNo = 103;
    comp.field.cinemap.isEditBoundary = true;
    comp.field.cinemap.cadenceIndexPresumed = true;

    assembler.push(QVector<SourceField>({comp, spare}));
    assembler.flush();
    const QVector<CadenceAssembler::WorkItem> work = assembler.popWork();

    check(work.size() == 1, "cut-truncated A should emit one frame work item");
    check(work[0].kind == CadenceAssembler::WorkItem::Kind::TelecineFrame,
          "cut-truncated A must use the frame regime, not passthrough");
    check(work[0].filmLabel == 'A', "cut-truncated pair should retain A identity");
    const int a0 = work[0].f1.field.cinemap.cadenceId;
    const int a1 = work[0].f2.field.cinemap.cadenceId;
    check((a0 == 1 && a1 == 2) || (a0 == 2 && a1 == 1),
          "cut head should retain A-comp and A-spare identities");
}

void testCutTruncatedCSpareCompUsesFrameRegime()
{
    LdDecodeMetaData::VideoParameters videoParameters;
    CadenceAssembler::Configuration configuration;
    CadenceAssembler assembler(videoParameters, configuration);

    // The cut removes C-def (slot 7), leaving C-spare/C-comp (5,6) at
    // the tail of the outgoing scene.  They still form a complete C frame.
    SourceField spare = makeField(5);
    SourceField comp = makeField(6);
    SourceField nextScene = makeField(-1);
    spare.field.seqNo = 201;
    comp.field.seqNo = 202;
    nextScene.field.seqNo = 203;
    spare.capturePartnerSeqNo = 200;
    comp.capturePartnerSeqNo = 203;
    nextScene.capturePartnerSeqNo = 204;
    nextScene.field.cinemap.isEditBoundary = true;

    assembler.push(QVector<SourceField>({spare, comp, nextScene}));
    assembler.flush();
    const QVector<CadenceAssembler::WorkItem> work = assembler.popWork();

    check(!work.isEmpty(), "cut-truncated C should emit a frame work item");
    check(work[0].kind == CadenceAssembler::WorkItem::Kind::TelecineFrame,
          "cut-truncated C must use the frame regime, not passthrough");
    check(work[0].filmLabel == 'C', "cut-truncated pair should retain C identity");
    const int c0 = work[0].f1.field.cinemap.cadenceId;
    const int c1 = work[0].f2.field.cinemap.cadenceId;
    check((c0 == 5 && c1 == 6) || (c0 == 6 && c1 == 5),
          "cut tail should retain C-spare and C-comp identities");
}

// An imposed cadence is the user's assertion against the evidence, so the only
// thing the assembler owes them is that it acts on every value and loses
// nothing. Both invariants are checked for all five positions, in each field
// order: every pushed field is either placed in a film frame or released to
// baseline, and the run produces work.
//
// The regression this pins: forcedStartIndex mapped the five values onto field
// slots {0,2,4,6,8} while the loop only had handlers for {0,2,3,5,6,8}. A start
// on slot 4 (--set-cadence 3) consumed nothing forever, and flush discarded the
// whole window without releasing it, so the render wrote a zero-byte file and
// reported success.
void testForcedCadenceActsOnEveryPositionAndLosesNoField()
{
    for (int reversed = 0; reversed <= 1; ++reversed) {
        for (int setCadence = 1; setCadence <= 5; ++setCadence) {
            LdDecodeMetaData::VideoParameters videoParameters;
            CadenceAssembler::Configuration configuration;
            configuration.setCadence        = setCadence;
            configuration.reverseFieldOrder = (reversed != 0);

            QVector<qint32> releasedToBaseline;
            CadenceAssembler assembler(
                videoParameters,
                configuration,
                [&](qint32 seqNo) { releasedToBaseline.push_back(seqNo); });

            // Three full cycles, pushed in two batches so a group that
            // straddles a push boundary is exercised too.
            QVector<SourceField> first, second;
            for (int i = 0; i < 12; ++i) first.push_back(makeField(i % 10));
            for (int i = 12; i < 30; ++i) second.push_back(makeField(i % 10));
            // makeField keys seqNo off cadenceId, which repeats every cycle;
            // the accounting below needs distinct fields, so renumber.
            for (int i = 0; i < first.size(); ++i) first[i].field.seqNo = i + 1;
            for (int i = 0; i < second.size(); ++i)
                second[i].field.seqNo = 12 + i + 1;

            assembler.push(first);
            assembler.push(second);
            assembler.flush();
            const QVector<CadenceAssembler::WorkItem> work = assembler.popWork();

            char message[128];
            std::snprintf(message, sizeof(message),
                          "forced cadence %d (reversed=%d) produced no work",
                          setCadence, reversed);
            check(!work.isEmpty(), message);

            QVector<qint32> accounted = releasedToBaseline;
            for (const auto &item : work) {
                accounted.push_back(item.f1.field.seqNo);
                accounted.push_back(item.f2.field.seqNo);

                // -r asserts lower-field-first, which is ld-cinemap's inverted
                // regime (cadenceIds 10..19). Stamping the normal 0..9 space
                // there would claim normal dominance for an inverted render.
                for (const SourceField *f : { &item.f1, &item.f2 }) {
                    const int cid = f->field.cinemap.cadenceId;
                    std::snprintf(message, sizeof(message),
                                  "forced cadence %d (reversed=%d) stamped cid "
                                  "%d in the wrong dominance regime",
                                  setCadence, reversed, cid);
                    check(cadenceIsInverted(cid) == (reversed != 0), message);
                }
            }

            // -r drops the stream's leading field when it regroups, so it can
            // account for one field fewer than were pushed.
            const int pushed   = static_cast<int>(first.size() + second.size());
            const int expected = pushed - (reversed ? 1 : 0);
            std::snprintf(message, sizeof(message),
                          "forced cadence %d (reversed=%d) accounted for %d of "
                          "%d fields", setCadence, reversed,
                          static_cast<int>(accounted.size()), expected);
            check(accounted.size() >= expected, message);
        }
    }
}

} // namespace

int main()
{
    testLockedCycleKeepsMixedFramePartnersAvailable();
    testPassthroughPreservesIdentityButForbidsFrameComb();
    testCutTruncatedACompSpareUsesFrameRegime();
    testCutTruncatedCSpareCompUsesFrameRegime();
    testForcedCadenceActsOnEveryPositionAndLosesNoField();
    return 0;
}
