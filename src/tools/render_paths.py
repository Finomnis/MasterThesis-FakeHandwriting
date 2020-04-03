import matplotlib.pyplot as plt

import numpy as np
from sklearn import mixture

import os
import sys
import argparse
from PIL import Image

from ext.deepwriting.dataset_hw import HandWritingDatasetConditional
from algorithms.scaling_errors import find_scaling_errors, fix_scaling_error
from algorithms.sample_to_penpositions import sample_to_penpositions
from algorithms.penpositions_to_skeletonimages import penpositions_to_skeletonimages
from algorithms.penpositions_to_text import penpositions_to_text


def renderDataset(inputDataset, outputFolder):

    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    preprocessingIsIncorrect = find_scaling_errors(inputDataset)

    for sampleId, rawSample in enumerate(inputDataset.samples):
        if sampleId % 50 == 0:
            print()
            sys.stdout.write('Rendering sample ' + str(sampleId) + ' / ' + str(len(inputDataset.samples)) + ' ')
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
        img = Image.fromarray(255-skeletonImage.astype('uint8')*255, mode='L')

        img.save(os.path.join(outputFolder, str(sampleId) + '.png'), 'PNG')
        text = penpositions_to_text(penPositions)
        with open(os.path.join(outputFolder, str(sampleId) + '.txt'), 'w') as fil:
            fil.write(text)


def main():

    parser = argparse.ArgumentParser(description='Converts the DeepWriting dataset to a skeleton version.')
    parser.add_argument('input', help='The input stroke database')
    parser.add_argument('output', help='The output skeleton database folder')
    parser.add_argument('--validation-in', help='The input stroke verification database')
    parser.add_argument('--validation-out', help='The output skeleton verification database folder')
    args = parser.parse_args()
    print(args)

    hasValidationDataset = False
    if args.validation_in and args.validation_out:
        hasValidationDataset = True

    if hasValidationDataset:
        inputValidationDataset = HandWritingDatasetConditional(args.validation_in)
        renderDataset(inputValidationDataset, args.validation_out)

    inputDataset = HandWritingDatasetConditional(args.input)
    renderDataset(inputDataset, args.output)


if __name__ == "__main__":
    main()
