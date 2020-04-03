import matplotlib.pyplot as plt

import numpy as np
from sklearn import mixture

import sys
import argparse

from ext.deepwriting.dataset_hw import HandWritingDatasetConditional
from algorithms.scaling_errors import find_scaling_errors, fix_scaling_error
from algorithms.sample_to_penpositions import sample_to_penpositions
from algorithms.penpositions_to_strokes import penpositions_to_strokes
from algorithms.penpositions_to_skeletonimages import penpositions_to_skeletonimages
from algorithms.strokes_to_penpositions import strokes_to_penpositions
from algorithms.resample_strokes import resample_strokes_smooth, analyse_strokes_acceleration

from datastructures.ConvertedDeepwritingDataset import ConvertedDataset

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


def convertDataset(inputDataset, resample=False):
    outputDataset = ConvertedDataset(inputDataset)

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

        strokes = penpositions_to_strokes(penPositions)
        skeletonImage,_,_,_,_ = penpositions_to_skeletonimages(penPositions)

        if False:
            strokeAccelerations = analyse_strokes_acceleration(strokes)
            figure = plt.figure('StrokeAccelerations')
            plt.hist(strokeAccelerations, bins=30)
            plt.show()
            #exit(1)

        smoothStrokes = resample_strokes_smooth(strokes)

        fakePenPositions = strokes_to_penpositions(smoothStrokes)

        outputDataset.addSample(fakePenPositions, inputDataset.texts[sampleId])

        if True:
            figure = plt.figure('PenPositionsImages')
            plt.subplot(3, 1, 1)
            plt.imshow(skeletonImage, cmap='binary', vmax=10)
            strokes.plot()
            plt.subplot(3, 1, 2)
            plt.imshow(skeletonImage, cmap='binary', vmax=10)
            smoothStrokes.plot()
            plt.subplot(3, 1, 3)
            plt.imshow(skeletonImage, cmap='binary', vmax=10)
            currentStrokeX = list()
            currentStrokeY = list()
            for penPosition in fakePenPositions:
                currentStrokeX.append(penPosition.pos[0])
                currentStrokeY.append(penPosition.pos[1])
                if penPosition.penUp:
                    plt.plot(currentStrokeX, currentStrokeY, '.-')
                    currentStrokeX = list()
                    currentStrokeY = list()
            if currentStrokeX:
                plt.plot(currentStrokeX, currentStrokeY, '.-')
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

    print('Applying preprocessing ...')
    outputDataset.applyPreProcessing(outputValidationDataset)

    print('Compressing databases ...')
    outputDataset.save(args.output)

    if hasValidationDataset:
        outputValidationDataset.save(args.validation_out)

    print('Verifying ...')
    verifiedOutputDataset = HandWritingDatasetConditional(args.output)

    print('Done.')


if __name__ == "__main__":
    main()
