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
    configuration.noPA = true;

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

} // namespace

int main()
{
    testLockedCycleKeepsMixedFramePartnersAvailable();
    testPassthroughPreservesIdentityButForbidsFrameComb();
    testCutTruncatedACompSpareUsesFrameRegime();
    testCutTruncatedCSpareCompUsesFrameRegime();
    return 0;
}
