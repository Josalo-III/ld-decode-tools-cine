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

static int applyCadenceOverrides(CineDisc& disc, const QStringList& cadenceOverrideArgs)
{
    int total = 0;
    for (const QString& spec : cadenceOverrideArgs) {
        if (spec.trimmed().isEmpty()) continue;
        const int changed = disc.applyCadenceOverrideFieldRange(spec);
        if (changed < 0) return -1;
        total += changed;
    }
    if (total > 0) {
        auto vp = disc.getMetaData().getVideoParameters();
        vp.isCinemapped = true;
        disc.getMetaData().setVideoParameters(vp);
    }
    return total;
}

static void applyEditOverrides(CineDisc& disc,
                               const QString& editWhitelistArg,
                               const QString& editBlacklistArg)
{
    if (!editWhitelistArg.trimmed().isEmpty()) {
        const int n = disc.applyEditWhitelistSeqNoKeys(editWhitelistArg);
        qInfo() << "Applied edit whitelist to" << n << "field(s) (seqNo keys).";
    }
    if (!editBlacklistArg.trimmed().isEmpty()) {
        const int n = disc.applyEditBlacklistSeqNoKeys(editBlacklistArg);
        qInfo() << "Applied edit blacklist to" << n << "field(s) (seqNo keys).";
    }
}

static QString metadataOutputPath(const CineDisc& disc, const QFileInfo& outputFileInfo)
{
    const QString sourcePath = disc.getTbcPath() + ".db";
    return (!outputFileInfo.filePath().isEmpty()
            && outputFileInfo.filePath() != "-"
            && outputFileInfo.filePath() != sourcePath)
        ? outputFileInfo.filePath() + ".db"
        : sourcePath;
}

static void markCinemapInUse(CineDisc& disc)
{
    auto& metadata = disc.getMetaData();
    const int totalFields = metadata.getNumberOfFields();

    for (int seqNo = 1; seqNo <= totalFields; ++seqNo) {
        auto field = metadata.getField(seqNo);
        field.cinemap.inUse = true;
        metadata.updateField(field, seqNo);
    }
}

static bool writeMetadata(CineDisc& disc,
                          const QFileInfo& outputFileInfo,
                          const char *failureMessage)
{
    markCinemapInUse(disc);

    const QString dbOutPath = metadataOutputPath(disc, outputFileInfo);
    if (!disc.getMetaData().write(dbOutPath)) {
        qWarning() << failureMessage;
        return false;
    }

    qInfo() << "Output metadata written to" << dbOutPath;
    return true;
}

static bool applyManualOverrides(CineDisc& disc,
                                 const QString& editWhitelistArg,
                                 const QString& editBlacklistArg,
                                 const QStringList& cadenceOverrideArgs)
{
    applyEditOverrides(disc, editWhitelistArg, editBlacklistArg);

    const int cadenceOverrides = applyCadenceOverrides(disc, cadenceOverrideArgs);
    if (cadenceOverrides < 0) return false;
    if (cadenceOverrides > 0) {
        qInfo() << "Applied manual cadence override to" << cadenceOverrides << "field(s).";
    }

    return true;
}

// -----------------------------------------------------------------------------
// Mode: override only
//   - Applies edit whitelist / blacklist and cadence overrides
//   - Does not run structural segmentation, visual edit detection, or cadence solve
// -----------------------------------------------------------------------------
static bool runOverrideOnly(CineDisc& disc,
                            const QFileInfo& outputFileInfo,
                            const QString& editWhitelistArg,
                            const QString& editBlacklistArg,
                            const QStringList& cadenceOverrideArgs)
{
    qInfo() << "ld-cinemap: running override-only metadata patch.";

    if (!applyManualOverrides(disc,
                              editWhitelistArg,
                              editBlacklistArg,
                              cadenceOverrideArgs)) {
        return false;
    }

    return writeMetadata(disc, outputFileInfo, "Failed to save metadata with manual overrides.");
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
                               const QString& editBlacklistArg,
                               const QStringList& cadenceOverrideArgs)
{
    qInfo() << "ld-cinemap: running edit detection only.";

    // 1) Structural segmentation
    const int segCount = segmenter::segmentDisc(disc);
    qInfo() << "Segmenter identified" << segCount << "segment(s).";

    // 2) Visual edit detection
    const int editCount = visualEdits::analyseVisualEdits(disc,
                                                          editSensitivity,
                                                          editStrong,
                                                          editPeak);
    qInfo() << "Visual edit detection committed" << editCount << "edit boundary(s).";

    if (!applyManualOverrides(disc,
                              editWhitelistArg,
                              editBlacklistArg,
                              cadenceOverrideArgs)) {
        return false;
    }

    return writeMetadata(disc, outputFileInfo, "Failed to save metadata with edit annotations.");
}

// -----------------------------------------------------------------------------
// Mode: cadence solve only (--skip-edits)
//   - Does not run segmenter or visual edit detection
//   - Runs CineMap cadence solve using existing edit boundaries
//   - Sets isCinemapped flag and writes metadata back to tbcPath + ".db"
// -----------------------------------------------------------------------------
static bool runSolveOnly(CineDisc& disc,
                         CineMap::Policy policy,
                         double threshold,
                         const QStringList& cadenceOverrideArgs)
{
    qInfo() << "ld-cinemap: running cadence solver (--skip-edits; using pre-annotated boundaries).";

    CineMap solver(&disc, policy);
    const int locked = solver.detectCadence(disc.getTbcPath(), threshold);
    qInfo() << "Cadence solver locked" << locked << "field(s).";

    if (!applyManualOverrides(disc, QString(), QString(), cadenceOverrideArgs)) return false;

    auto vp = disc.getMetaData().getVideoParameters();
    vp.isCinemapped = true;
    disc.getMetaData().setVideoParameters(vp);

    return writeMetadata(disc, QFileInfo(), "Failed to save cadence metadata.");
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
                            const QString& editBlacklistArg,
                            const QStringList& cadenceOverrideArgs)
{
    qInfo() << "ld-cinemap: running full pipeline.";

    // 1) Structural segmentation
    const int segCount = segmenter::segmentDisc(disc);
    qInfo() << "Segmenter identified" << segCount << "segment(s).";

    // 2) Visual edit detection
    const int editCount = visualEdits::analyseVisualEdits(disc,
                                                          editSensitivity,
                                                          editStrong,
                                                          editPeak);
    qInfo() << "Visual edit detection committed" << editCount << "edit boundary(s).";

    applyEditOverrides(disc, editWhitelistArg, editBlacklistArg);
        
    // 4) Cadence / twin / mixedness solve
    CineMap solver(&disc, policy);
    const int locked = solver.detectCadence(disc.getTbcPath(), threshold);
    qInfo() << "Cadence solver locked" << locked << "field(s).";

    if (!applyManualOverrides(disc, QString(), QString(), cadenceOverrideArgs)) return false;

    // 5) Set isCinemapped flag so downstream tools know cadence data is present
    auto vp = disc.getMetaData().getVideoParameters();
    vp.isCinemapped = true;
    disc.getMetaData().setVideoParameters(vp);

    return writeMetadata(disc, outputFileInfo, "Failed to save cadence metadata.");
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

    QCommandLineOption overrideOnlyOpt(
        QStringList() << "override-only",
        "Only apply edit whitelist/blacklist and cadence overrides; do not run segmentation, visual edit detection, or cadence solving.");

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

    QCommandLineOption cadenceOverrideOpt(
        QStringList() << "cadence-override",
        "Manual cadence override as fieldStart-fieldEnd:cadenceId, using field numbers as shown by ld-analyse. May be repeated. cadenceId is the first field's cadenceId; known IDs advance through the NTSC cadence sequence.",
        "override");

    parser.addOption(reverseOpt);
    parser.addOption(cineOpt);
    parser.addOption(tvOpt);
    parser.addOption(thresholdOpt);
    parser.addOption(detectEditsOnlyOpt);
    parser.addOption(skipEditsOpt);
    parser.addOption(overrideOnlyOpt);
    parser.addOption(clearAllFlagsOpt);
    parser.addOption(yesOpt);
    parser.addOption(sensitivityOpt);
    parser.addOption(strongOpt);
    parser.addOption(peakOpt);
    parser.addOption(editWhitelistOpt);
    parser.addOption(editBlacklistOpt);
    parser.addOption(cadenceOverrideOpt);

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

    const int modeCount = (parser.isSet(detectEditsOnlyOpt) ? 1 : 0)
                        + (parser.isSet(skipEditsOpt) ? 1 : 0)
                        + (parser.isSet(overrideOnlyOpt) ? 1 : 0);
    if (modeCount > 1) {
        qCritical("Error: --detect-edits-only, --skip-edits, and --override-only are mutually exclusive.");
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
    const QStringList cadenceOverrideArgs = parser.values(cadenceOverrideOpt);

    // -------------------------------------------------------------------------
    // Construct CineDisc
    // -------------------------------------------------------------------------
    std::unique_ptr<CineDisc> disc = loadCineDisc(inputFilename,
                                                  parser.isSet(reverseOpt));
    if (!disc) {
        qCritical("Error: failed to load disc metadata.");
        return 1;
    }

    if (!parser.isSet(overrideOnlyOpt)) {
        const auto vbi = vbiProbe::probe(*disc);
        // probe() calls disc->setIsDiscCav() internally.
        qInfo() << "Disc type:" << (vbi.isDiscCav ? "CAV" : "CLV");
    }

    // -------------------------------------------------------------------------
    // --clear-all-flags
    // -------------------------------------------------------------------------
    if (parser.isSet(clearAllFlagsOpt)) {
        const bool modeFollows = parser.isSet(detectEditsOnlyOpt)
                              || parser.isSet(skipEditsOpt)
                              || parser.isSet(overrideOnlyOpt);
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
                                           editBlacklistArg,
                                           cadenceOverrideArgs);
        return ok ? 0 : 1;
    }

    if (parser.isSet(overrideOnlyOpt)) {
        const bool ok = runOverrideOnly(*disc,
                                        outputFileInfo,
                                        editWhitelistArg,
                                        editBlacklistArg,
                                        cadenceOverrideArgs);
        return ok ? 0 : 1;
    }

    if (parser.isSet(skipEditsOpt)) {
        const bool ok = runSolveOnly(*disc, policy, threshold, cadenceOverrideArgs);
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
                                    editBlacklistArg,
                                    cadenceOverrideArgs);
    return ok ? 0 : 1;
}
