/************************************************************************

    stacker.cpp

    ld-disc-stacker - Disc stacking for ld-decode
    Copyright (C) 2020-2022 Simon Inns
    Copyright (C) 2025-2026 Joseph Burns

    This file is part of ld-decode-tools.

    ld-disc-stacker is free software: you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

************************************************************************/

#include "stacker.h"
#include "stackingpool.h"
#include "tbc/logging.h"

Stacker::Stacker(QAtomicInt& _abort, StackingPool& _stackingPool, QObject *parent)
    : QThread(parent), abort(_abort), stackingPool(_stackingPool)
{
}

void Stacker::run()
{
    qint32 frameNumber;
    QVector<qint32> firstFieldSeqNo;
    QVector<qint32> secondFieldSeqNo;
    QVector<SourceVideo::Data> firstSourceField;
    QVector<SourceVideo::Data> secondSourceField;
    QVector<LdDecodeMetaData::Field> firstFieldMetadata;
    QVector<LdDecodeMetaData::Field> secondFieldMetadata;
    qint32 mode;
    qint32 smartThreshold;
    bool reverse;
    bool noDiffDod;
    bool passThrough;
    bool verbose;
    QVector<qint32> availableSourcesForFrame;
    QVector<double> sourceSnrWeights;
    bool useSnrWeight;
    qint32 snrWeightThreshold;

    while (!abort) {
        if (!stackingPool.getInputFrame(frameNumber, firstFieldSeqNo, firstSourceField, firstFieldMetadata,
                                        secondFieldSeqNo, secondSourceField, secondFieldMetadata,
                                        videoParameters, mode, smartThreshold, reverse, noDiffDod, passThrough,
                                        verbose, availableSourcesForFrame, sourceSnrWeights,
                                        useSnrWeight, snrWeightThreshold)) {
            break;
        }

        SourceVideo::Data outputFirstField(firstSourceField[0].size());
        SourceVideo::Data outputSecondField(secondSourceField[0].size());
        DropOuts outputFirstFieldDropOuts;
        DropOuts outputSecondFieldDropOuts;

        stackField(frameNumber, firstSourceField, videoParameters[0], firstFieldMetadata,
                   availableSourcesForFrame, sourceSnrWeights, noDiffDod, passThrough,
                   outputFirstField, outputFirstFieldDropOuts, mode, smartThreshold, verbose,
                   useSnrWeight, snrWeightThreshold);
        stackField(frameNumber, secondSourceField, videoParameters[0], secondFieldMetadata,
                   availableSourcesForFrame, sourceSnrWeights, noDiffDod, passThrough,
                   outputSecondField, outputSecondFieldDropOuts, mode, smartThreshold, verbose,
                   useSnrWeight, snrWeightThreshold);

        stackingPool.setOutputFrame(frameNumber, outputFirstField, outputSecondField,
                                    firstFieldSeqNo[0], secondFieldSeqNo[0],
                                    outputFirstFieldDropOuts, outputSecondFieldDropOuts);
    }
}

void Stacker::stackField(const qint32 frameNumber,
                         const QVector<SourceVideo::Data>& inputFields,
                         const LdDecodeMetaData::VideoParameters& videoParameters,
                         const QVector<LdDecodeMetaData::Field>& fieldMetadata,
                         const QVector<qint32> availableSourcesForFrame,
                         const QVector<double>& sourceSnrWeights,
                         const bool& noDiffDod, const bool& passThrough,
                         SourceVideo::Data& outputField, DropOuts& dropOuts,
                         const qint32& mode, const qint32& smartThreshold,
                         const bool& verbose, const bool& useSnrWeight,
                         const qint32& snrWeightThreshold)
{
    quint16 prevGoodValue = videoParameters.black16bIre;
    bool forceDropout = false;

    // Sparse cache of pixel sample vectors for neighbour lookup in modes >= 3.
    // QHash avoids allocating ~150,000 empty QVector objects upfront; only
    // slots that are actually written incur any allocation.
    QHash<qint32, QVector<WeightedSample>> tmpField;
    tmpField.reserve(videoParameters.fieldWidth * 4);

    if (availableSourcesForFrame.size() > 0) {
        const qint32 nSrc = availableSourcesForFrame.size();

        // Pre-compute per-line dropout interval maps for each available source.
        // Replaces O(n) linear scan in isDropout() with O(k) scan over only the
        // intervals on the current line. Also correctly scopes haveAllDropout()
        // to availableSourcesForFrame rather than all fieldMetadata entries.
        using Interval = QPair<qint32, qint32>;
        QVector<QVector<QVector<Interval>>> srcDropMap(nSrc);
        for (qint32 si = 0; si < nSrc; si++) {
            const DropOuts& d = fieldMetadata[availableSourcesForFrame[si]].dropOuts;
            srcDropMap[si].resize(videoParameters.fieldHeight);
            for (qint32 k = 0; k < d.size(); k++) {
                const qint32 line = d.fieldLine(k) - 1;
                if (line >= 0 && line < videoParameters.fieldHeight)
                    srcDropMap[si][line].append({d.startx(k), d.endx(k)});
            }
        }

        auto fastIsDropout = [&](qint32 si, qint32 x, qint32 y) -> bool {
            if (y < 0 || y >= videoParameters.fieldHeight) return false;
            for (const auto& iv : srcDropMap[si][y])
                if (x >= iv.first && x <= iv.second) return true;
            return false;
        };

        auto fastHaveAllDropout = [&](qint32 x, qint32 y) -> bool {
            if (y < 0 || y >= videoParameters.fieldHeight) return true;
            for (qint32 si = 0; si < nSrc; si++)
                if (!fastIsDropout(si, x, y)) return false;
            return true;
        };

        // Hoist per-pixel working vectors outside the inner loop to avoid
        // repeated heap allocation. Each sample carries its source's SNR weight
        // inline, so a value and its weight can never be separated downstream.
        QVector<WeightedSample> inputValues, valuesN, valuesS, valuesE, valuesW;
        inputValues.reserve(nSrc);
        valuesN.reserve(nSrc); valuesS.reserve(nSrc);
        valuesE.reserve(nSrc); valuesW.reserve(nSrc);

        for (qint32 y = 0; y < videoParameters.fieldHeight; y++) {
            const qint32 rowOffset = videoParameters.fieldWidth * y;
            for (qint32 x = 0; x < videoParameters.fieldWidth; x++) {
                inputValues.clear();
                valuesN.clear();         valuesS.clear();
                valuesE.clear();         valuesW.clear();
                QVector<bool> isAllDropout = {true, true, true, true, true};

                if (mode >= 3) {
                    getProcessedSample(x, y, availableSourcesForFrame, inputFields, sourceSnrWeights,
                                       tmpField, srcDropMap, videoParameters, fieldMetadata,
                                       inputValues, valuesN, valuesS, valuesE, valuesW,
                                       isAllDropout, noDiffDod, verbose);
                } else {
                    for (qint32 i = 0; i < availableSourcesForFrame.size(); i++) {
                        const quint16 pixelValue = inputFields[availableSourcesForFrame[i]][rowOffset + x];
                        const bool sampleIsDropout = fastIsDropout(i, x, y);
                        const double w = (i < sourceSnrWeights.size()) ? sourceSnrWeights[i] : 0.0;

                        if (!sampleIsDropout)
                            inputValues.append({pixelValue, w});
                        else if (pixelValue > 0 && !noDiffDod)
                            inputValues.append({pixelValue, w});

                        if (!sampleIsDropout)
                            isAllDropout[0] = false;
                    }

                    if (isAllDropout[0] && availableSourcesForFrame.size() >= 3 && !noDiffDod) {
                        if (x > videoParameters.colourBurstStart) {
                            if (inputValues.size() >= 3) {
                                const double medianValue = static_cast<double>(median(inputValues));
                                const double minValue = qMax(0.0,     medianValue - (medianValue / 100.0) * 10.0);
                                const double maxValue = qMin(65535.0, medianValue + (medianValue / 100.0) * 10.0);
                                QVector<WeightedSample> filteredValues;
                                filteredValues.reserve(inputValues.size());
                                for (qint32 pi = 0; pi < inputValues.size(); pi++) {
                                    if (inputValues[pi].value > minValue && inputValues[pi].value < maxValue)
                                        filteredValues.append(inputValues[pi]);
                                }
                                inputValues.swap(filteredValues);
                            }

                            if (verbose) {
                                if (inputValues.size() > 0) {
                                    QVector<quint16> recovered; recovered.reserve(inputValues.size());
                                    for (const auto& s : inputValues) recovered.append(s.value);
                                    qInfo().nospace() << "Frame #" << frameNumber << ": DiffDOD recovered " << inputValues.size()
                                                      << " values: " << recovered << " for field location (" << x << ", " << y << ")";
                                }
                                else if (x > videoParameters.colourBurstStart)
                                    qInfo().nospace() << "Frame #" << frameNumber << ": DiffDOD failed, no values recovered for field location (" << x << ", " << y << ")";
                                else
                                    qInfo().nospace() << "Frame #" << frameNumber << ": Values 0 recovered for field location (" << x << ", " << y << ")";
                            }
                        }
                    }
                }

                forceDropout = false;
                if (availableSourcesForFrame.size() > 0 && passThrough) {
                    if (x > videoParameters.colourBurstStart) {
                        if (inputValues.size() == 0) {
                            forceDropout = true;
                            if (verbose)
                                qInfo().nospace() << "Frame #" << frameNumber << ": All sources for field location (" << x << ", " << y << ") are marked as dropout, passing through";
                        }
                    }
                }

                if (inputValues.size() == 0) {
                    outputField[rowOffset + x] = prevGoodValue;
                    if (x > videoParameters.colourBurstStart) dropOuts.append(x, x, y + 1);
                } else if (inputValues.size() == 1) {
                    outputField[rowOffset + x] = inputValues[0].value;
                    prevGoodValue = outputField[rowOffset + x];
                    if (forceDropout) dropOuts.append(x, x, y + 1);
                } else {
                    outputField[rowOffset + x] = stackMode(inputValues,
                                                            valuesN, valuesS, valuesE, valuesW,
                                                            isAllDropout, mode, smartThreshold,
                                                            snrWeightThreshold);
                    prevGoodValue = outputField[rowOffset + x];
                    // Cache the stacked result for neighbour lookups by later pixels/rows.
                    // Only ever read back via neighbourEstimate (value only), so the
                    // synthesised weight is immaterial.
                    tmpField[rowOffset + x] = QVector<WeightedSample>{ {prevGoodValue, 0.0} };
                    if (forceDropout) dropOuts.append(x, x, y + 1);
                }
            }
        }

        if (dropOuts.size() != 0) dropOuts.concatenate(verbose);
    } else {
        for (qint32 y = 0; y < videoParameters.fieldHeight; y++) {
            const qint32 rowOffset = videoParameters.fieldWidth * y;
            for (qint32 x = videoParameters.colourBurstStart; x < videoParameters.fieldWidth; x++)
                outputField[rowOffset + x] = videoParameters.black16bIre;
        }
    }
}

quint16 Stacker::stackMode(const QVector<WeightedSample>& elements,
                           const QVector<WeightedSample>& elementsN,
                           const QVector<WeightedSample>& elementsS,
                           const QVector<WeightedSample>& elementsE,
                           const QVector<WeightedSample>& elementsW,
                           const QVector<bool>& isAllDropout,
                           const qint32& mode,
                           const qint32& smartThreshold,
                           const qint32& snrWeightThreshold)
{
    const qint32 nbOfElements = elements.size();
    qint32 nbSelected = 0;
    quint32 result = 0;
    QVector<quint16> closestList;

    const double maxSnrPenalty = (nbOfElements > 1)
                                  ? static_cast<double>(snrWeightThreshold) * 0.5
                                  : 0.0;

    qint32 resultN = 0, resultS = 0, resultE = 0, resultW = 0;
    quint32 resultNeighbor = 0;
    qint32 nbNeighbor = 0;

    auto neighborEstimate = [&](const QVector<WeightedSample>& v) -> qint32 {
        if (v.size() <= 0) return -1;
        return Stacker::median(v);
    };

    // Dispatch on mode id. The user-facing name and description of each case are
    // defined once in STACKING_MODES (stackingmodes.h); keep these cases in step
    // with that table.
    switch (mode) {
        case 0: // mean
        {
            result = Stacker::mean(elements);
            break;
        }

        case 1: // median
        {
            result = Stacker::median(elements);
            break;
        }

        case 2: // smart mean
        {
            const qint32 med = Stacker::median(elements);
            for (int i = 0; i < nbOfElements; i++) {
                const qint32 v = static_cast<qint32>(elements[i].value);
                if (v < med + smartThreshold && v > med - smartThreshold) {
                    nbSelected++;
                    result += static_cast<quint32>(elements[i].value);
                }
            }
            result = (nbSelected == 0) ? static_cast<quint32>(med) : (result / static_cast<quint32>(nbSelected));
            break;
        }

        case 3: // smart neighbor
        {
            const qint32 med = Stacker::median(elements);

            resultN = neighborEstimate(elementsN);
            resultS = neighborEstimate(elementsS);
            if (!isAllDropout[0]) {
                resultE = neighborEstimate(elementsE);
                resultW = neighborEstimate(elementsW);
            } else {
                resultE = -1; resultW = -1;
            }

            (resultN != -1) ? nbNeighbor++ : resultN = 0;
            (resultS != -1) ? nbNeighbor++ : resultS = 0;
            (resultE != -1) ? nbNeighbor++ : resultE = 0;
            (resultW != -1) ? nbNeighbor++ : resultW = 0;

            closestList.clear();
            if (nbNeighbor > 0) {
                if (resultN > 0) closestList.append(Stacker::closestSnr(elements, resultN, maxSnrPenalty));
                if (resultS > 0) closestList.append(Stacker::closestSnr(elements, resultS, maxSnrPenalty));
                if (resultE > 0) closestList.append(Stacker::closestSnr(elements, resultE, maxSnrPenalty));
                if (resultW > 0) closestList.append(Stacker::closestSnr(elements, resultW, maxSnrPenalty));
                resultNeighbor = Stacker::closest(closestList, med);
            } else {
                resultNeighbor = static_cast<quint32>(med);
            }

            if (nbOfElements > 2) {
                result = 0; nbSelected = 0;
                for (int i = 0; i < nbOfElements; i++) {
                    const qint32 v = static_cast<qint32>(elements[i].value);
                    if (v < static_cast<qint32>(resultNeighbor) + smartThreshold &&
                        v > static_cast<qint32>(resultNeighbor) - smartThreshold) {
                        nbSelected++;
                        result += static_cast<quint32>(elements[i].value);
                    }
                }
                result = (nbSelected == 0) ? resultNeighbor : (result / static_cast<quint32>(nbSelected));
            } else {
                result = resultNeighbor;
            }
            break;
        }

        case 4: // neighbor
        {
            const qint32 med = Stacker::median(elements);

            resultN = neighborEstimate(elementsN);
            resultS = neighborEstimate(elementsS);
            if (!isAllDropout[0] || (isAllDropout[1] && isAllDropout[2])) {
                resultE = neighborEstimate(elementsE);
                resultW = neighborEstimate(elementsW);
            } else {
                resultE = -1; resultW = -1;
            }

            (resultN != -1) ? nbNeighbor++ : resultN = 0;
            (resultS != -1) ? nbNeighbor++ : resultS = 0;
            (resultE != -1) ? nbNeighbor++ : resultE = 0;
            (resultW != -1) ? nbNeighbor++ : resultW = 0;

            closestList.clear();
            if (nbNeighbor > 0) {
                if (resultN > 0) closestList.append(Stacker::closestSnr(elements, resultN, maxSnrPenalty));
                if (resultS > 0) closestList.append(Stacker::closestSnr(elements, resultS, maxSnrPenalty));
                if (resultE > 0) closestList.append(Stacker::closestSnr(elements, resultE, maxSnrPenalty));
                if (resultW > 0) closestList.append(Stacker::closestSnr(elements, resultW, maxSnrPenalty));
                result = Stacker::closest(closestList, med);
                if (nbOfElements > 2)
                    result = (static_cast<quint32>(med) + result) / 2;
            } else {
                result = static_cast<quint32>(med);
            }
            break;
        }

        case 5: // local neighbor: medoid inlier gate + mode 4 on inliers
        {
            const qint32 center = static_cast<qint32>(Stacker::medoid(elements));
            QVector<WeightedSample> inliers;
            inliers.reserve(nbOfElements);
            for (int i = 0; i < nbOfElements; i++) {
                const qint32 v = static_cast<qint32>(elements[i].value);
                if (v < center + smartThreshold && v > center - smartThreshold)
                    inliers.append(elements[i]);
            }
            if (inliers.isEmpty()) { result = static_cast<quint32>(center); break; }

            const qint32 inlierMedian = Stacker::median(inliers);

            resultN = neighborEstimate(elementsN);
            resultS = neighborEstimate(elementsS);
            if (!isAllDropout[0] || (isAllDropout[1] && isAllDropout[2])) {
                resultE = neighborEstimate(elementsE);
                resultW = neighborEstimate(elementsW);
            } else { resultE = -1; resultW = -1; }

            (resultN != -1) ? nbNeighbor++ : resultN = 0;
            (resultS != -1) ? nbNeighbor++ : resultS = 0;
            (resultE != -1) ? nbNeighbor++ : resultE = 0;
            (resultW != -1) ? nbNeighbor++ : resultW = 0;

            closestList.clear();
            if (nbNeighbor > 0) {
                if (resultN > 0) closestList.append(Stacker::closestSnr(inliers, resultN, maxSnrPenalty));
                if (resultS > 0) closestList.append(Stacker::closestSnr(inliers, resultS, maxSnrPenalty));
                if (resultE > 0) closestList.append(Stacker::closestSnr(inliers, resultE, maxSnrPenalty));
                if (resultW > 0) closestList.append(Stacker::closestSnr(inliers, resultW, maxSnrPenalty));
                result = Stacker::closest(closestList, inlierMedian);
                if (inliers.size() > 2)
                    result = (static_cast<quint32>(inlierMedian) + result) / 2;
            } else {
                result = static_cast<quint32>(inlierMedian);
            }
            break;
        }

        case 6: // smart local neighbor: medoid inlier gate + medoid on inliers + mode 3 neighbour anchor
        {
            const qint32 center = static_cast<qint32>(Stacker::medoid(elements));
            QVector<WeightedSample> inliers;
            inliers.reserve(nbOfElements);
            for (int i = 0; i < nbOfElements; i++) {
                const qint32 v = static_cast<qint32>(elements[i].value);
                if (v < center + smartThreshold && v > center - smartThreshold)
                    inliers.append(elements[i]);
            }
            if (inliers.isEmpty()) { result = static_cast<quint32>(center); break; }

            const quint32 smartLocalAnchor = static_cast<quint32>(Stacker::medoid(inliers));

            resultN = neighborEstimate(elementsN);
            resultS = neighborEstimate(elementsS);
            if (!isAllDropout[0]) {
                resultE = neighborEstimate(elementsE);
                resultW = neighborEstimate(elementsW);
            } else { resultE = -1; resultW = -1; }

            (resultN != -1) ? nbNeighbor++ : resultN = 0;
            (resultS != -1) ? nbNeighbor++ : resultS = 0;
            (resultE != -1) ? nbNeighbor++ : resultE = 0;
            (resultW != -1) ? nbNeighbor++ : resultW = 0;

            quint32 neighborAnchor = smartLocalAnchor;
            closestList.clear();
            if (nbNeighbor > 0) {
                if (resultN > 0) closestList.append(Stacker::closestSnr(inliers, resultN, maxSnrPenalty));
                if (resultS > 0) closestList.append(Stacker::closestSnr(inliers, resultS, maxSnrPenalty));
                if (resultE > 0) closestList.append(Stacker::closestSnr(inliers, resultE, maxSnrPenalty));
                if (resultW > 0) closestList.append(Stacker::closestSnr(inliers, resultW, maxSnrPenalty));
                const quint32 neighborSelection = Stacker::closest(closestList, static_cast<qint32>(smartLocalAnchor));
                if (inliers.size() > 2) {
                    quint32 neighborSum = 0; qint32 neighborCount = 0;
                    for (int i = 0; i < inliers.size(); i++) {
                        const qint32 v = static_cast<qint32>(inliers[i].value);
                        if (v < static_cast<qint32>(neighborSelection) + smartThreshold &&
                            v > static_cast<qint32>(neighborSelection) - smartThreshold) {
                            neighborCount++;
                            neighborSum += static_cast<quint32>(inliers[i].value);
                        }
                    }
                    neighborAnchor = (neighborCount == 0) ? neighborSelection : (neighborSum / static_cast<quint32>(neighborCount));
                } else {
                    neighborAnchor = neighborSelection;
                }
            }
            result = (smartLocalAnchor + neighborAnchor) / 2;
            break;
        }

        case 7: // medoid
        {
            result = static_cast<quint32>(Stacker::medoid(elements));
            break;
        }

        default:
        {
            result = static_cast<quint32>(Stacker::median(elements));
            break;
        }
    }

    // Bad-consensus SNR override:
    // If the SNR-weighted mean disagrees with the count-consensus result by more
    // than snrWeightThreshold, and the high-SNR minority holds >= 35% of total
    // SNR weight, replace result with the SNR-weighted mean of that minority.
    {
        constexpr double BAD_CONSENSUS_MIN_WEIGHT_FRACTION = 0.35;

        if (nbOfElements >= 3) {
            double totalWeight = 0.0, weightedSum = 0.0;
            for (int i = 0; i < nbOfElements; i++) {
                totalWeight += elements[i].weight;
                weightedSum += elements[i].weight * static_cast<double>(elements[i].value);
            }

            if (totalWeight > 0.0) {
                const double snrWeightedMean = weightedSum / totalWeight;
                const double divergence = snrWeightedMean - static_cast<double>(result);

                if (divergence > static_cast<double>(snrWeightThreshold) ||
                    divergence < -static_cast<double>(snrWeightThreshold)) {

                    double agreeingWeight = 0.0, agreeingWeightedSum = 0.0;
                    for (int i = 0; i < nbOfElements; i++) {
                        const double v = static_cast<double>(elements[i].value);
                        if (v >= snrWeightedMean - static_cast<double>(snrWeightThreshold) &&
                            v <= snrWeightedMean + static_cast<double>(snrWeightThreshold)) {
                            agreeingWeight      += elements[i].weight;
                            agreeingWeightedSum += elements[i].weight * v;
                        }
                    }

                    if (agreeingWeight >= totalWeight * BAD_CONSENSUS_MIN_WEIGHT_FRACTION)
                        result = static_cast<quint32>(agreeingWeightedSum / agreeingWeight + 0.5);
                }
            }
        }
    }

    return static_cast<quint16>(result);
}

inline quint16 Stacker::median(QVector<WeightedSample> elements)
{
    const qint32 noOfElements = elements.size();
    const auto byValue = [](const WeightedSample& a, const WeightedSample& b) { return a.value < b.value; };
    if (noOfElements % 2 == 0) {
        std::nth_element(elements.begin(), elements.begin() + noOfElements / 2, elements.end(), byValue);
        std::nth_element(elements.begin(), elements.begin() + (noOfElements - 1) / 2, elements.end(), byValue);
        return static_cast<quint16>((elements[(noOfElements - 1) / 2].value + elements[noOfElements / 2].value) / 2.0);
    } else {
        std::nth_element(elements.begin(), elements.begin() + noOfElements / 2, elements.end(), byValue);
        return static_cast<quint16>(elements[noOfElements / 2].value);
    }
}

// The medoid is the sample minimising the sum of absolute distances to all
// other samples — the most globally central real observation.
// Fallback: N=0 → 0, N=1 → passthrough, N=2 → mean, N≥3 → O(N²) pairwise minimisation.
// Ties broken by first-wins (stable and deterministic).
inline quint16 Stacker::medoid(const QVector<WeightedSample>& elements)
{
    const qint32 n = elements.size();
    if (n <= 0) return 0;
    if (n == 1) return elements[0].value;
    if (n == 2) return static_cast<quint16>((static_cast<quint32>(elements[0].value) +
                                             static_cast<quint32>(elements[1].value)) / 2);

    quint32 bestTotalDist = std::numeric_limits<quint32>::max();
    quint16 bestValue     = elements[0].value;
    for (qint32 i = 0; i < n; i++) {
        quint32 totalDist = 0;
        for (qint32 j = 0; j < n; j++)
            totalDist += static_cast<quint32>(std::abs(static_cast<qint32>(elements[i].value) -
                                                        static_cast<qint32>(elements[j].value)));
        if (totalDist < bestTotalDist) {
            bestTotalDist = totalDist;
            bestValue     = elements[i].value;
        }
    }
    return bestValue;
}

inline qint32 Stacker::mean(const QVector<WeightedSample>& elements)
{
    quint32 result = 0;
    const qint32 nbElements = elements.size();
    if (nbElements > 1) {
        for (int i = 0; i < nbElements; i++) result += elements[i].value;
        return result / nbElements;
    } else if (nbElements == 1) {
        return elements[0].value;
    }
    return -1;
}

inline quint16 Stacker::closest(const QVector<quint16>& elements, const qint32 target)
{
    const qint32 nbOfElements = elements.size();
    qint32 best = 0;
    if (nbOfElements > 0) {
        best = elements[0];
        for (int i = 1; i < nbOfElements; i++)
            if (abs(target - elements[i]) < abs(target - best))
                best = elements[i];
    }
    return best;
}

// Find the closest value to target, with a distance penalty for sources below
// the median SNR weight. The penalty rises linearly from zero at the median weight
// to maxPenalty at weight zero, capped so SNR never overrides a large distance gap.
inline quint16 Stacker::closestSnr(const QVector<WeightedSample>& elements,
                                    const qint32 target, const double maxPenalty)
{
    const qint32 n = elements.size();
    if (n == 0) return 0;

    QVector<double> sorted;
    sorted.reserve(n);
    for (const auto& e : elements) sorted.append(e.weight);
    std::nth_element(sorted.begin(), sorted.begin() + n / 2, sorted.end());
    const double medianWeight = sorted[n / 2];

    qint32 bestValue = elements[0].value;
    double bestCost  = std::numeric_limits<double>::max();
    for (int i = 0; i < n; i++) {
        double dist = static_cast<double>(std::abs(target - static_cast<qint32>(elements[i].value)));
        if (medianWeight > 0.0) {
            const double deficit = qMax(0.0, medianWeight - elements[i].weight) / medianWeight;
            dist += deficit * maxPenalty;
        }
        if (dist < bestCost) { bestCost = dist; bestValue = elements[i].value; }
    }
    return static_cast<quint16>(bestValue);
}

void Stacker::getProcessedSample(const qint32 x, const qint32 y,
                                  const QVector<qint32>& availableSourcesForFrame,
                                  const QVector<SourceVideo::Data>& inputFields,
                                  const QVector<double>& sourceSnrWeights,
                                  QHash<qint32, QVector<WeightedSample>>& tmpField,
                                  const QVector<QVector<QVector<QPair<qint32,qint32>>>>& srcDropMap,
                                  const LdDecodeMetaData::VideoParameters& videoParameters,
                                  const QVector<LdDecodeMetaData::Field>& fieldMetadata,
                                  QVector<WeightedSample>& sample,
                                  QVector<WeightedSample>& sampleN, QVector<WeightedSample>& sampleS,
                                  QVector<WeightedSample>& sampleE, QVector<WeightedSample>& sampleW,
                                  QVector<bool>& isAllDropout,
                                  const bool& noDiffDod, const bool& verbose)
{
    quint16 pixelValue = 0;
    qint32 source = 0;
    const qint32 fieldWidth  = videoParameters.fieldWidth;
    const qint32 fieldHeight = videoParameters.fieldHeight;
    const qint32 rowOffset     = fieldWidth * y;
    const qint32 rowOffsetNext = fieldWidth * (y + 1);
    const qint32 rowOffsetPrev = fieldWidth * (y - 1);
    bool sampleIsDropout = true;

    const qint32 nSrc = availableSourcesForFrame.size();
    auto fastIsDropout = [&](qint32 si, qint32 px, qint32 py) -> bool {
        if (py < 0 || py >= fieldHeight) return false;
        for (const auto& iv : srcDropMap[si][py])
            if (px >= iv.first && px <= iv.second) return true;
        return false;
    };
    auto fastHaveAllDropout = [&](qint32 px, qint32 py) -> bool {
        if (py < 0 || py >= fieldHeight) return true;
        for (qint32 si = 0; si < nSrc; si++)
            if (!fastIsDropout(si, px, py)) return false;
        return true;
    };

    for (qint32 i = 0; i < availableSourcesForFrame.size(); i++) {
        source = availableSourcesForFrame[i];
        // Weight of this source, bound to each value it contributes at the moment
        // of the append so a skipped (dropout) source drops its weight with it.
        const double w = (i < sourceSnrWeights.size()) ? sourceSnrWeights[i] : 0.0;
        if (y == 0) {
            if (x == 0) {
                pixelValue = inputFields[source][rowOffset + x];
                sampleIsDropout = fastIsDropout(i, x, y);
                if (!sampleIsDropout) sample.append({pixelValue, w});
                else if (pixelValue > 0 && !noDiffDod) sample.append({pixelValue, w});
                if (!sampleIsDropout) isAllDropout[0] = false;

                pixelValue = inputFields[source][rowOffset + x + 1];
                sampleIsDropout = fastIsDropout(i, x+1, y);
                if (!sampleIsDropout) sampleE.append({pixelValue, w});
                else if (pixelValue > 0 && !noDiffDod) sampleE.append({pixelValue, w});
                if (!sampleIsDropout) isAllDropout[3] = false;

                pixelValue = inputFields[source][rowOffsetNext + x];
                sampleIsDropout = fastIsDropout(i, x, y+1);
                if (!sampleIsDropout) sampleS.append({pixelValue, w});
                else if (pixelValue > 0 && !noDiffDod) sampleS.append({pixelValue, w});
                if (!sampleIsDropout) isAllDropout[2] = false;
            } else if (x == fieldWidth - 1) {
                pixelValue = inputFields[source][rowOffsetNext + x];
                sampleIsDropout = fastIsDropout(i, x, y+1);
                if (!sampleIsDropout) sampleS.append({pixelValue, w});
                else if (pixelValue > 0 && !noDiffDod) sampleS.append({pixelValue, w});
                if (!sampleIsDropout) isAllDropout[2] = false;
            } else {
                pixelValue = inputFields[source][rowOffset + x + 1];
                sampleIsDropout = fastIsDropout(i, x+1, y);
                if (!sampleIsDropout) sampleE.append({pixelValue, w});
                else if (pixelValue > 0 && !noDiffDod) sampleE.append({pixelValue, w});
                if (!sampleIsDropout) isAllDropout[3] = false;

                pixelValue = inputFields[source][rowOffsetNext + x];
                sampleIsDropout = fastIsDropout(i, x, y+1);
                if (!sampleIsDropout) sampleS.append({pixelValue, w});
                else if (pixelValue > 0 && !noDiffDod) sampleS.append({pixelValue, w});
                if (!sampleIsDropout) isAllDropout[2] = false;
            }
        } else if (y != fieldHeight - 1) {
            pixelValue = inputFields[source][rowOffsetNext + x];
            sampleIsDropout = fastIsDropout(i, x, y+1);
            if (!sampleIsDropout) sampleS.append({pixelValue, w});
            else if (pixelValue > 0 && !noDiffDod) sampleS.append({pixelValue, w});
            if (!sampleIsDropout) isAllDropout[2] = false;
        }
    }

    if (y == 0) {
        if (x == 0) {
            if (!noDiffDod && x > videoParameters.colourBurstStart) {
                if (isAllDropout[0] && availableSourcesForFrame.size() >= 3) sample  = diffDod(sample,  videoParameters, verbose);
                if (isAllDropout[3] && availableSourcesForFrame.size() >= 3) sampleE = diffDod(sampleE, videoParameters, verbose);
                if (isAllDropout[2] && availableSourcesForFrame.size() >= 3) sampleS = diffDod(sampleS, videoParameters, verbose);
            }
            tmpField[rowOffset + x]     = sample;
            tmpField[rowOffset + x + 1] = sampleE;
            tmpField[rowOffsetNext + x] = sampleS;
        } else if (x == fieldWidth - 1) {
            if (!noDiffDod && x > videoParameters.colourBurstStart)
                if (isAllDropout[2] && availableSourcesForFrame.size() >= 3) sampleS = diffDod(sampleS, videoParameters, verbose);
            tmpField[rowOffsetNext + x] = sampleS;
            sample  = tmpField[rowOffset + x];
            sampleW = tmpField[rowOffset + x - 1];
            isAllDropout[4] = fastHaveAllDropout(x-1, y);
        } else {
            if (!noDiffDod && x > videoParameters.colourBurstStart) {
                if (isAllDropout[3] && availableSourcesForFrame.size() >= 3) sampleE = diffDod(sampleE, videoParameters, verbose);
                if (isAllDropout[2] && availableSourcesForFrame.size() >= 3) sampleS = diffDod(sampleS, videoParameters, verbose);
            }
            tmpField[rowOffset + x + 1] = sampleE;
            tmpField[rowOffsetNext + x] = sampleS;
            sample  = tmpField[rowOffset + x];
            sampleW = tmpField[rowOffset + x - 1];
            isAllDropout[4] = fastHaveAllDropout(x-1, y);
        }
    } else if (y != fieldHeight - 1) {
        if (!noDiffDod && x > videoParameters.colourBurstStart)
            if (isAllDropout[2] && availableSourcesForFrame.size() >= 3) sampleS = diffDod(sampleS, videoParameters, verbose);
        tmpField[rowOffsetNext + x] = sampleS;
        if (x == 0) {
            sample  = tmpField[rowOffset + x];
            sampleE = tmpField[rowOffset + x + 1];
            sampleN = tmpField[rowOffsetPrev + x];
            isAllDropout[1] = fastHaveAllDropout(x,   y-1);
            isAllDropout[3] = fastHaveAllDropout(x+1, y);
        } else if (x == fieldWidth - 1) {
            sample  = tmpField[rowOffset + x];
            sampleW = tmpField[rowOffset + x - 1];
            sampleN = tmpField[rowOffsetPrev + x];
            isAllDropout[1] = fastHaveAllDropout(x,   y-1);
            isAllDropout[4] = fastHaveAllDropout(x-1, y);
        } else {
            sample  = tmpField[rowOffset + x];
            sampleW = tmpField[rowOffset + x - 1];
            sampleE = tmpField[rowOffset + x + 1];
            sampleN = tmpField[rowOffsetPrev + x];
            isAllDropout[1] = fastHaveAllDropout(x,   y-1);
            isAllDropout[3] = fastHaveAllDropout(x+1, y);
            isAllDropout[4] = fastHaveAllDropout(x-1, y);
        }
    } else {
        if (x == 0) {
            sample  = tmpField[rowOffset + x];
            sampleE = tmpField[rowOffset + x + 1];
            sampleN = tmpField[rowOffsetPrev + x];
            isAllDropout[1] = fastHaveAllDropout(x,   y-1);
            isAllDropout[3] = fastHaveAllDropout(x+1, y);
        } else if (x == fieldWidth - 1) {
            sample  = tmpField[rowOffset + x];
            sampleW = tmpField[rowOffset + x - 1];
            sampleN = tmpField[rowOffsetPrev + x];
            isAllDropout[1] = fastHaveAllDropout(x,   y-1);
            isAllDropout[4] = fastHaveAllDropout(x-1, y);
        } else {
            sample  = tmpField[rowOffset + x];
            sampleW = tmpField[rowOffset + x - 1];
            sampleE = tmpField[rowOffset + x + 1];
            sampleN = tmpField[rowOffsetPrev + x];
            isAllDropout[1] = fastHaveAllDropout(x,   y-1);
            isAllDropout[3] = fastHaveAllDropout(x+1, y);
            isAllDropout[4] = fastHaveAllDropout(x-1, y);
        }
    }
}

// Legacy per-pixel dropout check. The hot path in stackField uses the inlined
// fastIsDropout lambda over a pre-built interval map instead. These functions
// are retained for any future callers outside that loop.
inline bool Stacker::isDropout(const DropOuts& dropOuts, const qint32 fieldX, const qint32 fieldY)
{
    for (qint32 i = 0; i < dropOuts.size(); i++) {
        if ((dropOuts.fieldLine(i) - 1) == fieldY)
            if (fieldX >= dropOuts.startx(i) && fieldX <= dropOuts.endx(i))
                return true;
    }
    return false;
}

inline bool Stacker::haveAllDropout(const QVector<LdDecodeMetaData::Field>& fieldMetadata,
                                     const qint32 x, const qint32 y)
{
    const qint32 size = fieldMetadata.size();
    for (qint32 i = 0; i < size; i++)
        if (!isDropout(fieldMetadata[i].dropOuts, x, y))
            return false;
    return true;
}

QVector<Stacker::WeightedSample> Stacker::diffDod(const QVector<WeightedSample>& inputValues,
                                   const LdDecodeMetaData::VideoParameters& videoParameters,
                                   const bool& verbose)
{
    QVector<WeightedSample> outputValues;
    if (inputValues.size() < 3) return inputValues;

    const double medianValue = static_cast<double>(median(inputValues));
    const double threshold   = 10.0; // %
    double maxValueD = medianValue + ((medianValue / 100.0) * threshold);
    double minValueD = medianValue - ((medianValue / 100.0) * threshold);
    if (minValueD < 0)      minValueD = 0;
    if (maxValueD > 65535)  maxValueD = 65535;
    const quint16 minValue = static_cast<quint16>(minValueD);
    const quint16 maxValue = static_cast<quint16>(maxValueD);

    for (qint32 i = 0; i < inputValues.size(); i++)
        if (inputValues[i].value > minValue && inputValues[i].value < maxValue)
            outputValues.append(inputValues[i]);

    if (verbose) {
        QVector<quint16> inVals; inVals.reserve(inputValues.size());
        for (const auto& s : inputValues) inVals.append(s.value);
        tbcDebugStream() << "diffDOD:  Input" << inVals;
        if (outputValues.size() == 0)
            tbcDebugStream().nospace() << "diffDOD: Empty output... Range was " << minValue << "-" << maxValue << " with a median of " << medianValue;
        else {
            QVector<quint16> outVals; outVals.reserve(outputValues.size());
            for (const auto& s : outputValues) outVals.append(s.value);
            tbcDebugStream() << "diffDOD: Output" << outVals;
        }
    }

    return outputValues;
}
