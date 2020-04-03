#!/usr/bin/env python3

import argparse
import os

from tqdm import tqdm
from PIL import Image, ImageOps
import numpy as np

from pipeline.sampling import sample_to_penpositions
from datastructures.ConvertedGravesDataset import ConvertedGravesDataset


def main():
    parser = argparse.ArgumentParser(description='Converts a dataset of skeletons to a graves dataset.')
    parser.add_argument('input', help='The input skeletons folder')
    parser.add_argument('labels', help='The text labels as .npy file')
    parser.add_argument('output', help='The output dataset')
    parser.add_argument('--sort-by', default='mean', help='The sorting order. Can be one of \'first\', \'last\', \'leftmost\', \'rightmost\', \'mean\'.')
    args = parser.parse_args()
    print(args)

    fileNames = [f for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f))]
    textAnnotations = np.load(args.labels, allow_pickle=True).item()

    outputDataset = ConvertedGravesDataset()

    for fileName in tqdm(fileNames):
        #print(fileName)
        strokeId = os.path.splitext(os.path.basename(fileName))[0]
        lineText = textAnnotations[strokeId]
        # print(strokeId, lineText)

        fullName = os.path.join(args.input, fileName)
        lineImg = ImageOps.invert(Image.open(fullName).convert('L'))

        # print('SkeletonImg:', lineId, lineImg, lineText)
        penPositions = sample_to_penpositions(lineImg, args.sort_by)
        outputDataset.addSample(penPositions, lineText)

    outputDataset.save(args.output)


if __name__ == "__main__":
    main()
