// tools/ld-cinemap/main.cpp

#include <QCoreApplication>
#include <QCommandLineParser>
#include <QFileInfo>
#include <QTextStream>
#include <QtGlobal>

#include "tbc/logging.h"
#include "cinedisc.h"
#include "lddecodemetadata.h"
#include "cinemap.h"
#include "segmenter.h"
#include "visualedits.h"
#include "vbiprobe.h"

// -----------------------------------------------------------------------------
// Prompt helper — returns true if the user confirms, false otherwise.
// Bypassed when autoConfirm is true (--yes flag).
// -----------------------------------------------------------------------------
static bool confirmPrompt(const QString& message, bool autoConfirm)
{
    if (autoConfirm) return true;

    QTextStream out(stdout);
    QTextStream in(stdin);
    out << message << " [Y/N]: ";
    out.flush();

    QString response;
    in >> response;
    return response.trimmed().compare("y", Qt::CaseInsensitive) == 0;
}

// -----------------------------------------------------------------------------
// Mode: edit detection only
//   - Segments the disc by phase structure
//   - Runs visual edit detection
//   - Applies whitelist / blacklist
//   - Writes updated JSON; no TBC output
// -----------------------------------------------------------------------------
static bool runDetectEditsOnly(CineDisc& disc,
                               const QFileInfo& outputFileInfo,
                               double editSensitivity,
                               double editStrong,
                               double editPeak,
                               const QString& editWhitelistArg,
                               const QString& editBlacklistArg)
{
    qInfo() << "ld-cinemap: running edit detection only.";

    // 1) Structural segmentation
    const int segCount = segmenter::segmentDisc(disc);
    qInfo() << "Segmenter identified" << segCount << "segment(s).";

    // 2) Visual edit detection
    const int editCount = visualEdits::analyzeVisualEdits(disc,
                                                          editSensitivity,
                                                          editStrong,
                                                          editPeak);
    qInfo() << "Visual edit detection committed" << editCount << "edit boundary(s).";

    // 3) Whitelist / blacklist overrides
    if (!editWhitelistArg.trimmed().isEmpty()) {
        // TODO: port applyEditWhitelistSeqNoKeys
        qInfo() << "Applying edit whitelist:" << editWhitelistArg;
    }
    if (!editBlacklistArg.trimmed().isEmpty()) {
        // TODO: port applyEditBlacklistSeqNoKeys
        qInfo() << "Applying edit blacklist:" << editBlacklistArg;
    }

    // 4) Determine output metadata path
    const QString sourcePath = disc.getTbcPath() + ".db";
    const QString jsonOutPath =
        (!outputFileInfo.filePath().isEmpty()
         && outputFileInfo.filePath() != "-"
         && outputFileInfo.filePath() != sourcePath)
        ? outputFileInfo.filePath() + ".db"
        : sourcePath;

    if (!disc.getMetaData().write(jsonOutPath)) {
        qWarning() << "Failed to save metadata with edit annotations.";
        return false;
    }

    qInfo() << "Edit annotations written to" << jsonOutPath;
    return true;
}

// -----------------------------------------------------------------------------
// Mode: cadence solve only (--skip-edits)
//   - Does not run segmenter or visual edit detection
//   - Runs CineMap cadence solve using existing edit boundaries
//   - Sets isCinemapped flag and writes metadata back to tbcPath + ".db"
// -----------------------------------------------------------------------------
static bool runSolveOnly(CineDisc& disc,
                         CineMap::Policy policy,
                         double threshold)
{
    qInfo() << "ld-cinemap: running cadence solver (--skip-edits; using pre-annotated boundaries).";

    CineMap solver(&disc, policy);
    const int locked = solver.detectCadence(disc.getTbcPath(), threshold);
    qInfo() << "Cadence solver locked" << locked << "field(s).";

    auto vp = disc.getMetaData().getVideoParameters();
    vp.isCinemapped = true;
    disc.getMetaData().setVideoParameters(vp);

    if (!disc.getMetaData().write(disc.getTbcPath() + ".db")) {
        qWarning() << "Failed to save cadence metadata.";
        return false;
    }

    return true;
}

// -----------------------------------------------------------------------------
// Mode: full pipeline (default)
//   - Segment → visual edits
//   - Optional whitelist / blacklist
//   - Cadence solve
//   - Write JSON
// -----------------------------------------------------------------------------
static bool runFullPipeline(CineDisc& disc,
                            const QFileInfo& outputFileInfo,
                            CineMap::Policy policy,
                            double threshold,
                            double editSensitivity,
                            double editStrong,
                            double editPeak,
                            const QString& editWhitelistArg,
                            const QString& editBlacklistArg)
{
    qInfo() << "ld-cinemap: running full pipeline.";

    // 1) Structural segmentation
    const int segCount = segmenter::segmentDisc(disc);
    qInfo() << "Segmenter identified" << segCount << "segment(s).";

    // 2) Visual edit detection
    const int editCount = visualEdits::analyzeVisualEdits(disc,
                                                          editSensitivity,
                                                          editStrong,
                                                          editPeak);
    qInfo() << "Visual edit detection committed" << editCount << "edit boundary(s).";

    // 3) Whitelist / blacklist overrides
    if (!editWhitelistArg.trimmed().isEmpty()) {
        // TODO: port applyEditWhitelistSeqNoKeys
        qInfo() << "Applying edit whitelist:" << editWhitelistArg;
    }
    if (!editBlacklistArg.trimmed().isEmpty()) {
        // TODO: port applyEditBlacklistSeqNoKeys
        qInfo() << "Applying edit blacklist:" << editBlacklistArg;
    }

    // 4) Cadence / twin / mixedness solve
    CineMap solver(&disc, policy);
    const int locked = solver.detectCadence(disc.getTbcPath(), threshold);
    qInfo() << "Cadence solver locked" << locked << "field(s).";

    // 5) Set isCinemapped flag so downstream tools know cadence data is present
    auto vp = disc.getMetaData().getVideoParameters();
    vp.isCinemapped = true;
    disc.getMetaData().setVideoParameters(vp);

    // 6) Write metadata
    const QString sourcePath = disc.getTbcPath() + ".db";
    const QString jsonOutPath =
        (!outputFileInfo.filePath().isEmpty()
         && outputFileInfo.filePath() != "-"
         && outputFileInfo.filePath() != sourcePath)
        ? outputFileInfo.filePath() + ".db"
        : sourcePath;

    if (!disc.getMetaData().write(jsonOutPath)) {
        qWarning() << "Failed to save cadence metadata.";
        return false;
    }

    qInfo() << "Output metadata written to" << jsonOutPath;
    return true;
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    QCoreApplication app(argc, argv);
    QCoreApplication::setApplicationName("ld-cinemap");
    QCoreApplication::setApplicationVersion(QString("ld-decode-tools - Branch: %1 / Commit: %2").arg(APP_BRANCH, APP_COMMIT));

    QCommandLineParser parser;
    parser.setApplicationDescription("ld-cinemap — telecine cadence solver + edit detection");
    parser.addHelpOption();
    parser.addVersionOption();

    // Options
    QCommandLineOption reverseOpt(
        QStringList() << "r" << "reverse",
        "Reverse field order (swap first/second fields).");

    QCommandLineOption cineOpt(
        QStringList() << "cine",
        "Implement film edited telecine reconstruction policy.");

    QCommandLineOption tvOpt(
        QStringList() << "tv",
        "Implement video edited telecine policy.");

    QCommandLineOption thresholdOpt(
        QStringList() << "threshold",
        "Cadence confidence threshold.",
        "threshold",
        "0.0");

    QCommandLineOption detectEditsOnlyOpt(
        QStringList() << "detect-edits-only",
        "Only run segmentation + visual edit detection and write JSON; do not solve cadence.");

    QCommandLineOption skipEditsOpt(
        QStringList() << "skip-edits",
        "Skip segmentation + visual edit detection; solve cadence using existing edit boundaries.");

    QCommandLineOption clearAllFlagsOpt(
        QStringList() << "clear-all-flags",
        "Clear all edit/cadence related flags in metadata before doing anything.");

    QCommandLineOption yesOpt(
        QStringList() << "y" << "yes",
        "Assume 'yes' for prompts.");

    QCommandLineOption sensitivityOpt(
        QStringList() << "sensitivity",
        "Visual edit sensitivity.",
        "sensitivity",
        "8.0");

    QCommandLineOption strongOpt(
        QStringList() << "strong",
        "Visual edit strong factor.",
        "strong",
        "1.5");

    QCommandLineOption peakOpt(
        QStringList() << "peak",
        "Visual edit peak factor.",
        "peak",
        "1.6");

    QCommandLineOption editWhitelistOpt(
        QStringList() << "edit-whitelist",
        "Edit whitelist (seqNo keys).",
        "whitelist",
        "");

    QCommandLineOption editBlacklistOpt(
        QStringList() << "edit-blacklist",
        "Edit blacklist (seqNo keys).",
        "blacklist",
        "");

    parser.addOption(reverseOpt);
    parser.addOption(cineOpt);
    parser.addOption(tvOpt);
    parser.addOption(thresholdOpt);
    parser.addOption(detectEditsOnlyOpt);
    parser.addOption(skipEditsOpt);
    parser.addOption(clearAllFlagsOpt);
    parser.addOption(yesOpt);
    parser.addOption(sensitivityOpt);
    parser.addOption(strongOpt);
    parser.addOption(peakOpt);
    parser.addOption(editWhitelistOpt);
    parser.addOption(editBlacklistOpt);

    parser.addPositionalArgument("tbc", "Input TBC file (metadata expected at tbc + \".db\").");
    parser.addPositionalArgument("output", "Optional output base path (metadata will be written to output + \".db\").", "[output]");

    parser.process(app);

    const auto positional = parser.positionalArguments();
    if (positional.isEmpty()) {
        parser.showHelp(1);
    }

    const QString inputFilename  = positional.at(0);
    const QString outputFilename = (positional.size() >= 2) ? positional.at(1) : QString();

    QFileInfo inputFileInfo(inputFilename);
    QFileInfo outputFileInfo(outputFilename);

    if (!inputFileInfo.exists()) {
        qCritical("Input file does not exist");
        return 1;
    }

    // Mutual exclusion: --detect-edits-only vs --skip-edits
    if (parser.isSet(detectEditsOnlyOpt) && parser.isSet(skipEditsOpt)) {
        qCritical("Error: --detect-edits-only and --skip-edits are mutually exclusive.");
        return 1;
    }

    const bool autoConfirm = parser.isSet(yesOpt);

    // Policy selection
    CineMap::Policy policy = CineMap::Policy::Tv;
    if (parser.isSet(cineOpt)) policy = CineMap::Policy::Cine;
    if (parser.isSet(tvOpt))   policy = CineMap::Policy::Tv;

    const double threshold       = parser.value(thresholdOpt).toDouble();
    const double editSensitivity = parser.value(sensitivityOpt).toDouble();
    const double editStrong      = parser.value(strongOpt).toDouble();
    const double editPeak        = parser.value(peakOpt).toDouble();

    const QString editWhitelistArg = parser.value(editWhitelistOpt);
    const QString editBlacklistArg = parser.value(editBlacklistOpt);

    // -------------------------------------------------------------------------
    // Construct CineDisc
    // -------------------------------------------------------------------------
    std::unique_ptr<CineDisc> disc = loadCineDisc(inputFilename,
                                                  parser.isSet(reverseOpt));
    if (!disc) {
        qCritical("Error: failed to load disc metadata.");
        return 1;
    }

	const auto vbi = vbiProbe::probe(*disc);
	//   probe() calls disc->setIsDiscCav() internally.
	qInfo() << "Disc type:" << (vbi.isDiscCav ? "CAV" : "CLV");

    // -------------------------------------------------------------------------
    // --clear-all-flags
    // -------------------------------------------------------------------------
    if (parser.isSet(clearAllFlagsOpt)) {
        const bool modeFollows = parser.isSet(detectEditsOnlyOpt)
                              || parser.isSet(skipEditsOpt);
        const bool runPipeline = modeFollows
            || confirmPrompt(
                   QString("Clearing all flags (edit boundaries/cadenceId). Continue?"),
                   autoConfirm);

        if (!runPipeline) {
            qInfo() << "Aborted.";
            return 0;
        }

        segmenter::clearAllFlags(*disc);
    }

    // -------------------------------------------------------------------------
    // Modes
    // -------------------------------------------------------------------------
    if (parser.isSet(detectEditsOnlyOpt)) {
        const bool ok = runDetectEditsOnly(*disc,
                                           outputFileInfo,
                                           editSensitivity,
                                           editStrong,
                                           editPeak,
                                           editWhitelistArg,
                                           editBlacklistArg);
        return ok ? 0 : 1;
    }

    if (parser.isSet(skipEditsOpt)) {
        const bool ok = runSolveOnly(*disc, policy, threshold);
        return ok ? 0 : 1;
    }

    // Default: full pipeline
    const bool ok = runFullPipeline(*disc,
                                    outputFileInfo,
                                    policy,
                                    threshold,
                                    editSensitivity,
                                    editStrong,
                                    editPeak,
                                    editWhitelistArg,
                                    editBlacklistArg);
    return ok ? 0 : 1;
}