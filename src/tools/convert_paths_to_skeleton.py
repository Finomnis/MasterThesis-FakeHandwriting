import matplotlib.pyplot as plt

import numpy as np
from sklearn import mixture

import sys
import argparse
from PIL import Image
from PIL import ImageDraw
import scipy

from skimage.morphology import skeletonize, thin

from ext.deepwriting.dataset_hw import HandWritingDatasetConditional
from ext.deepwriting import preprocessing
from algorithms.skeleton_to_graph import skeleton_to_graph
from algorithms.graph_to_strokes import graph_to_strokes
from algorithms.resolve_strokes import resolve_strokes
from algorithms.scaling_errors import find_scaling_errors, fix_scaling_error
from algorithms.sample_to_penpositions import sample_to_penpositions
from algorithms.penpositions_to_skeletonimages import penpositions_to_skeletonimages
from algorithms.strokes_to_penpositions import strokes_to_penpositions
from algorithms.resample_strokes import resample_strokes_smooth
from datastructures.PenPosition import PenPosition

import math


# input database consists of:

# - subject_labels: no idea

# - strokes: the actual dataset of strokes. everything else is metadata.
# - alphabet: all possible labels that a character can have
# - sow_labels: start of word labels
# - eow_labels: end of word labels
# - soc_labels: start of character labels
# - eoc_labels: end of character labels
# - char_labels: current character we are about to write. first stroke is labeled 0-character, every further
#                stroke in ascii, last stroke of the caracter aditionally the eoc-label
# - word_labels: enumerates the current word, starting with 1
# - texts: content of the sample, as text
# - max: the maximum value of 'scale' preprocessing
# - min: the minimum value of 'scale' preprocessing
# - mean: the mean value of 'normalization' preprocessing
# - std: the std derivation of 'normalization' preprocessing
# - preprocessing: preprocessing steps. in our case 'scale', 'origin_translation',
#                  'relative_representation' and 'normalization'


def annotateStrokes(strokesGraph, char_bitmap, eoc_bitmap, bow_bitmap):
    # print(np.shape(char_bitmap))

    # Annotate
    for strokeId, stroke in strokesGraph.strokes.items():
        charLabels = list()
        eocLabels = list()
        bowLabels = list()
        for point in stroke.points:
            pX = int(round(point.pos[0]))
            pY = int(round(point.pos[1]))
            currentChar = char_bitmap[pY, pX]
            currentEoc = eoc_bitmap[pY, pX]
            currentBow = bow_bitmap[pY, pX]
            charLabels.append(currentChar)
            eocLabels.append(currentEoc)
            bowLabels.append(currentBow)

        # Remove small ripples
        charLabels = scipy.ndimage.filters.median_filter(charLabels, 7)
        eocLabels = scipy.ndimage.filters.median_filter(eocLabels, 7)
        bowLabels = scipy.ndimage.filters.median_filter(bowLabels, 7)

        # Store in stroke
        for point, eocLabel, bowLabel, charLabel in zip(stroke.points, eocLabels, bowLabels, charLabels):
            point.eocLabel = eocLabel
            point.bowLabel = bowLabel
            point.charLabel = charLabel


def generateSubdivisionPoint(stroke, subDivisionPosition):
    pos1Weight, pos0 = math.modf(subDivisionPosition)
    pos0 = int(pos0)
    pos1 = pos0 + 1
    pos0Weight = 1 - pos1Weight
    np.testing.assert_almost_equal(pos0 * pos0Weight + pos1 * pos1Weight, subDivisionPosition)

    if pos0 < 0:
        pos0 = 0
    if pos1 >= len(stroke.points):
        pos1 = len(stroke.points) - 1

    # print(pos0, len(stroke.points))
    assert (pos1 >= 0)
    assert (pos0 < len(stroke.points))

    point0 = stroke.points[pos0]
    point1 = stroke.points[pos1]

    pX = point0[0] * pos0Weight + point1[0] * pos1Weight
    pY = point0[1] * pos0Weight + point1[1] * pos1Weight

    additionalDataPos = pos0
    if pos0Weight < 0.5:
        additionalDataPos = pos1

    return PenPosition(pX, pY, False,
                       stroke.extraData['charLabels'][additionalDataPos],
                       stroke.extraData['eocLabels'][additionalDataPos],
                       stroke.extraData['bowLabels'][additionalDataPos])


def pointDistance(p0, p1):
    dX = p1[0] - p0[0]
    dY = p1[1] - p0[1]

    return math.sqrt(dX * dX + dY * dY)


def addSampleToDataset(dataset, penPositions, textContent):
    eocLabels = list()
    bowLabels = list()
    charLabels = list()
    stroke = list()
    for penPosition in penPositions:
        eocLabels.append(penPosition.eocLabel)
        bowLabels.append(penPosition.bowLabel)
        charLabels.append(penPosition.charLabel)
        stroke.append([penPosition.pos[0], penPosition.pos[1], penPosition.penUp])

    eocLabels = np.array(eocLabels, dtype='i4')
    bowLabels = np.array(bowLabels, dtype='i4')
    charLabels = np.array(charLabels, dtype='i4')
    stroke = np.array(stroke, dtype='f4')
    # print(stroke)

    dataset.get('eoc_labels').append(eocLabels)
    dataset.get('sow_labels').append(bowLabels)
    dataset.get('char_labels').append(charLabels)
    dataset.get('samples').append(stroke)
    dataset.get('texts').append(textContent)


def generatePenPositions(strokesGraph, penSpeed):
    penPositions = list()

    for strokeId, stroke in strokesGraph.strokes.items():
        strokeLength = stroke.length
        numSubdivisions = int(round(stroke.length / penSpeed))
        if numSubdivisions < 1:
            numSubdivisions = 1

        subdivisionPointPositions = np.array(range(numSubdivisions + 1)) * strokeLength / numSubdivisions

        subdivisionPoints = list()

        previousPoint = None
        distanceSum = 0.0
        previousDistanceSum = -1.0  # important to prevent division through zero
        nextSubdivisionId = 0
        nextSubdivision = subdivisionPointPositions[nextSubdivisionId]
        for pointId, point in enumerate(stroke.points):
            if previousPoint is not None:
                previousDistanceSum = distanceSum
                distanceSum += pointDistance(previousPoint, point)

            while distanceSum > nextSubdivision:
                # print('---\n', previousDistanceSum, nextSubdivision, distanceSum)

                # Compute sub-integer array position of the point we are searching for
                distanceStep = distanceSum - previousDistanceSum
                distanceSubstep = nextSubdivision - previousDistanceSum
                distancePct = distanceSubstep / distanceStep
                targetArrayPosition = pointId - 1 + distancePct

                # Compute the interpolated value
                subdivisionPoint = generateSubdivisionPoint(stroke, targetArrayPosition)
                subdivisionPoints.append(subdivisionPoint)

                nextSubdivisionId += 1
                if nextSubdivisionId >= len(subdivisionPointPositions):
                    break
                nextSubdivision = subdivisionPointPositions[nextSubdivisionId]

            previousPoint = point

        if len(subdivisionPoints) < len(subdivisionPointPositions):
            subdivisionPoints.append(generateSubdivisionPoint(stroke, len(stroke.points) - 1))

        assert (len(subdivisionPoints) == len(subdivisionPointPositions))

        # insert pen-up event
        subdivisionPoints[-1].penUp = True

        penPositions.extend(subdivisionPoints)

    # Create eoc signals
    seenEocLabels = set()
    for penPos in reversed(penPositions):
        if penPos.eocLabel in seenEocLabels:
            penPos.eocLabel = 0
        else:
            if penPos.penUp:
                penPos.eocLabel = 0
                penPos.charLabel = 0
            else:
                seenEocLabels.add(penPos.eocLabel)
                penPos.eocLabel = 1

    # Create bow signals
    seenBowLabels = set()
    for penPos in penPositions:
        if penPos.bowLabel in seenBowLabels:
            penPos.bowLabel = 0
        else:
            seenBowLabels.add(penPos.bowLabel)
            penPos.bowLabel = 1

    # for penPos in penPositions:
    #    print(penPos)

    return penPositions


def plotPenPositions(penPositions):
    xCoords = list()
    yCoords = list()
    for penPosition in penPositions:
        xCoords.append(penPosition.pX)
        yCoords.append(penPosition.pY)
        if penPosition.penUp:
            plt.plot(xCoords, yCoords, '-')
            xCoords = list()
            yCoords = list()


def createOutputDataset(inputDataset):
    outputDataset = dict()
    outputDataset['alphabet'] = inputDataset.alphabet
    outputDataset['eoc_labels'] = list()
    outputDataset['sow_labels'] = list()
    outputDataset['char_labels'] = list()
    outputDataset['samples'] = list()
    outputDataset['texts'] = list()
    outputDataset['preprocessing'] = list()

    # Add necessary but unused data
    outputDataset['subject_labels'] = None
    outputDataset['soc_labels'] = None
    outputDataset['eow_labels'] = None

    return outputDataset


def applyPreProcessing(dataset, validationSet=None):
    preprocessing.scale_dataset(dataset, validation_data=validationSet)
    preprocessing.translate_to_origin(dataset)
    preprocessing.convert_to_diff_representation(dataset)
    if validationSet:
        preprocessing.translate_to_origin(validationSet)
        preprocessing.convert_to_diff_representation(validationSet)
    preprocessing.standardize_dataset(dataset, validation_data=validationSet)


def convertDataset(inputDataset, DRAW_STEPS=False):
    outputDataset = createOutputDataset(inputDataset)

    preprocessingIsIncorrect = find_scaling_errors(inputDataset)

    for sampleId, rawSample in enumerate(inputDataset.samples):
        if sampleId % 50 == 0:
            print()
            sys.stdout.write('Adding sample ' + str(sampleId) + ' / ' + str(len(inputDataset.samples)) + ' ')
            sys.stdout.flush()
        else:
            sys.stdout.write('.')
            sys.stdout.flush()

        sample = inputDataset.undo_preprocess(rawSample)

        if preprocessingIsIncorrect[sampleId]:
            sample = fix_scaling_error(sample, inputDataset.scale_max, inputDataset.scale_min)

        penPositions = sample_to_penpositions(sample,
                                              inputDataset.char_labels[sampleId],
                                              inputDataset.eoc_labels[sampleId],
                                              inputDataset.bow_labels[sampleId])

        skeletonImage, skeletonCharImage, skeletonEocImage, skeletonBowImage, skeletonMetadata = penpositions_to_skeletonimages(penPositions)

        if False:
            figure = plt.figure('SkeletonImages')
            plt.subplot(4, 1, 1)
            plt.imshow(skeletonImage)
            plt.subplot(4, 1, 2)
            plt.imshow(skeletonCharImage, cmap='nipy_spectral', vmin=40, vmax=70)
            plt.subplot(4, 1, 3)
            plt.imshow(skeletonEocImage, cmap='nipy_spectral')
            plt.subplot(4, 1, 4)
            plt.imshow(skeletonBowImage, cmap='nipy_spectral')
            plt.show()
            exit(1)

        thinnedImage = skeletonize(skeletonImage)

        graph = skeleton_to_graph(thinnedImage)

        if DRAW_STEPS:
            print("Drawing ...")
            figure = plt.figure("Graphs")
            plt.subplot(4, 1, 1)
            plt.imshow(thinnedImage, cmap='binary', vmax=10)
            graph.plot()

        resolve_strokes(graph)

        strokes = graph_to_strokes(graph)

        strokes.sort()
        annotateStrokes(strokes, skeletonCharImage, skeletonEocImage, skeletonBowImage)

        smoothStrokes = resample_strokes_smooth(strokes)

        if DRAW_STEPS:
            plt.subplot(4, 1, 2)
            plt.imshow(thinnedImage, cmap='binary', vmax=10)
            graph.plot()
            plt.subplot(4, 1, 3)
            plt.imshow(thinnedImage, cmap='binary', vmax=10)
            strokes.plot()
            plt.subplot(4, 1, 4)
            plt.imshow(thinnedImage, cmap='binary', vmax=10)
            smoothStrokes.plot()
            plt.show()
            exit(1)

        fakePenPositions = strokes_to_penpositions(smoothStrokes)

        addSampleToDataset(outputDataset, fakePenPositions, inputDataset.texts[sampleId])

        if False:
            figure = plt.figure('PenPositionsImages')
            plt.subplot(2, 1, 1)
            plt.imshow(thinnedImage, cmap='binary', vmax=10)
            smoothStrokes.plot()
            plt.subplot(2, 1, 2)
            plt.imshow(skeletonImage, cmap='binary', vmax=10)
            currentStrokeX = list()
            currentStrokeY = list()
            for penPosition in fakePenPositions:
                currentStrokeX.append(penPosition.pX)
                currentStrokeY.append(penPosition.pY)
                if penPosition.penUp:
                    plt.plot(currentStrokeX, currentStrokeY, '.-')
                    currentStrokeX = list()
                    currentStrokeY = list()
            plt.show()
            exit(1)

        if False:
            sample = outputDataset.get('samples')[0]
            penPositions = sample_to_penpositions(sample,
                                                  outputDataset.get('char_labels')[sampleId],
                                                  outputDataset.get('eoc_labels')[sampleId],
                                                  outputDataset.get('sow_labels')[sampleId])

            skeletonImage, skeletonCharImage, skeletonEocImage, skeletonBowImage, skeletonMetadata = penpositions_to_skeletonimages(
                penPositions)

            figure = plt.figure('OutputSkeletonImages')
            plt.subplot(4, 1, 1)
            plt.imshow(skeletonImage)
            plt.subplot(4, 1, 2)
            plt.imshow(skeletonCharImage, cmap='nipy_spectral', vmin=40, vmax=70)
            plt.subplot(4, 1, 3)
            plt.imshow(skeletonEocImage, cmap='nipy_spectral')
            plt.subplot(4, 1, 4)
            plt.imshow(skeletonBowImage, cmap='nipy_spectral')
            plt.show()
            exit(1)

    print()

    return outputDataset


def main():
    parser = argparse.ArgumentParser(description='Converts the DeepWriting dataset to a skeleton version.')
    parser.add_argument('input', help='The input stroke database')
    parser.add_argument('output', help='The output skeleton database')
    parser.add_argument('--validation-in', help='The input stroke verification database')
    parser.add_argument('--validation-out', help='The output skeleton verification database')
    args = parser.parse_args()
    print(args)

    hasValidationDataset = False
    if args.validation_in and args.validation_out:
        hasValidationDataset = True

    outputValidationDataset = None
    if hasValidationDataset:
        inputValidationDataset = HandWritingDatasetConditional(args.validation_in)
        outputValidationDataset = convertDataset(inputValidationDataset)

    inputDataset = HandWritingDatasetConditional(args.input)
    outputDataset = convertDataset(inputDataset)

    applyPreProcessing(outputDataset, outputValidationDataset)

    np.savez_compressed(args.output, **outputDataset)

    if hasValidationDataset:
        np.savez_compressed(args.validation_out, **outputValidationDataset)

    verifiedOutputDataset = HandWritingDatasetConditional(args.output)


if __name__ == "__main__":
    main()
