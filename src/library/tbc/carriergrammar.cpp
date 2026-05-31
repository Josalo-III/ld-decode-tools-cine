/******************************************************************************
 * carriergrammar.cpp
 * ld-decode-tools shared composite carrier grammar helpers
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 ******************************************************************************/

#include "carriergrammar.h"

namespace lddecode {

int mergeCadenceIdForInterleavedFrame(int cadenceA, int cadenceB, bool editSplit)
{
    return mergeCadenceIdForInterleavedFrame(
        cadenceA, cadenceB, editSplit, CadenceMergePolicy{});
}

int mergeCadenceIdForInterleavedFrame(int cadenceA,
                                      int cadenceB,
                                      bool editSplit,
                                      const CadenceMergePolicy &policy)
{
    // If the edit split happens between these two fields, force Video mode.
    if (editSplit) return policy.videoCadenceId;

    const bool aFilm = (cadenceA >= 0);
    const bool bFilm = (cadenceB >= 0);
    if (aFilm && bFilm) return (cadenceA < cadenceB) ? cadenceA : cadenceB;
    if (aFilm) return cadenceA;
    if (bFilm) return cadenceB;
    if (cadenceA == policy.progressiveCadenceId ||
        cadenceB == policy.progressiveCadenceId)
    {
        return policy.progressiveCadenceId;
    }
    return policy.videoCadenceId;
}

int phaseIdForInterleavedLine(int lineNumber,
                              int firstFieldPhaseId,
                              int secondFieldPhaseId)
{
    return ((lineNumber % 2) == 0) ? firstFieldPhaseId : secondFieldPhaseId;
}

CarrierGrammarState makeCarrierGrammarStateForLine(int lineNumber,
                                                   int firstFieldPhaseId,
                                                   int secondFieldPhaseId,
                                                   bool frameVerticalAllowed)
{
    return makeCarrierGrammarStateForLine(
        lineNumber,
        firstFieldPhaseId,
        secondFieldPhaseId,
        frameVerticalAllowed,
        CarrierGrammarSchedulePolicy{});
}

CarrierGrammarState makeCarrierGrammarStateForLine(
    int lineNumber,
    int firstFieldPhaseId,
    int secondFieldPhaseId,
    bool frameVerticalAllowed,
    const CarrierGrammarSchedulePolicy &policy)
{
    CarrierGrammarState grammar{};
    grammar.line = lineNumber;
    grammar.fieldPhaseId =
        phaseIdForInterleavedLine(lineNumber, firstFieldPhaseId, secondFieldPhaseId);
    grammar.lineParity = lineNumber & 1;

    const int fieldLine = lineNumber / 2;
    const bool positiveOnEven =
        (grammar.fieldPhaseId == policy.positiveOnEvenPhaseIdA) ||
        (grammar.fieldPhaseId == policy.positiveOnEvenPhaseIdB);
    const bool evenFieldLine = ((fieldLine & 1) == 0);
    const bool linePhase = evenFieldLine ? positiveOnEven : !positiveOnEven;

    grammar.lineFlip = linePhase ? -1 : +1;
    grammar.samplePhase0 = policy.defaultSamplePhase0;
    // lineFlip is derived from fieldPhaseId which comes from capture metadata,
    // so it holds Metadata authority.
    grammar.lineFlipAuthority = policy.defaultAuthority;
    grammar.rigidScheduleLineFlip = grammar.lineFlip;
    grammar.phaseScheduleConflict = 0.0;
    grammar.frameVerticalAllowed = frameVerticalAllowed;
    return grammar;
}

void initializeCarrierGrammarSchedule(std::vector<CarrierGrammarState> &carrierGrammar,
                                      int firstLine,
                                      int lastLine,
                                      int firstFieldPhaseId,
                                      int secondFieldPhaseId,
                                      bool frameVerticalAllowed)
{
    initializeCarrierGrammarSchedule(
        carrierGrammar,
        firstLine,
        lastLine,
        firstFieldPhaseId,
        secondFieldPhaseId,
        frameVerticalAllowed,
        CarrierGrammarSchedulePolicy{});
}

void initializeCarrierGrammarSchedule(
    std::vector<CarrierGrammarState> &carrierGrammar,
    int firstLine,
    int lastLine,
    int firstFieldPhaseId,
    int secondFieldPhaseId,
    bool frameVerticalAllowed,
    const CarrierGrammarSchedulePolicy &policy)
{
    if (lastLine <= firstLine)
        return;

    if ((int)carrierGrammar.size() < lastLine)
        carrierGrammar.resize(lastLine);

    for (int line = firstLine; line < lastLine; ++line) {
        carrierGrammar[line] = makeCarrierGrammarStateForLine(
            line,
            firstFieldPhaseId,
            secondFieldPhaseId,
            frameVerticalAllowed,
            policy);
    }
}

} // namespace lddecode
