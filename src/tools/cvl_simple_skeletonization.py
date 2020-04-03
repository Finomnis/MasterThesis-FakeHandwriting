#!/usr/bin/env python3

import argparse
import os

from PIL import Image, ImageFilter
import numpy as np

import matplotlib.pyplot as plt
from scipy.ndimage import morphology
from skimage import morphology as skmorph

from tqdm import tqdm


def skeletonize(inFile, outFile, blur=True):

    #if not inFile.endswith("0002-1-10.tif"):
    #    return

    # Open File and convert to array
    inputImg = Image.open(inFile).convert(mode='RGB')
    inputArray = np.asarray(inputImg)

    img = np.any(inputArray < 240, 2)

    openStruct = [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    closeStruct = [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ]

    openedImg = morphology.binary_opening(img, structure=openStruct, iterations=1)
    closedImg = morphology.binary_closing(openedImg, structure=closeStruct, iterations=1)

    outputImgArray = skmorph.skeletonize(closedImg)
    outputImg = Image.fromarray(255 - outputImgArray.astype("uint8") * 255)

    if False:
        figure = plt.figure('CVL Skeletonization')
        plt.subplot(3, 2, 1)
        plt.imshow(inputArray)
        plt.subplot(3, 2, 2)
        plt.imshow(img)
        plt.subplot(3, 2, 3)
        plt.imshow(closedImg)
        plt.subplot(3, 2, 4)
        plt.imshow(skmorph.thin(closedImg))
        plt.subplot(3, 2, 5)
        plt.imshow(skmorph.skeletonize(closedImg))
        plt.subplot(3, 2, 6)
        plt.imshow(skmorph.skeletonize_3d(closedImg))

        #outputImg.show()
        plt.show()

        #exit(1)

    if blur:
        outputImg = outputImg.filter(ImageFilter.GaussianBlur(1))

    outputImg.save(outFile)


def main():

    parser = argparse.ArgumentParser(description='Converts the CVL dataset to a skeleton version.')
    parser.add_argument('input', help='The input cvl dataset folder')
    parser.add_argument('output', help='The output skeleton database folder')
    parser.add_argument('--no_blur', action='store_true', help='Disables the blurring')
    args = parser.parse_args()
    print(args)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    fileNames = [f for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f))]
    for relativeFilename in tqdm(fileNames):
        inFile = os.path.join(args.input, relativeFilename)
        if not os.path.isfile(inFile):
            continue
        outFile = os.path.join(args.output, relativeFilename)

        skeletonize(inFile, outFile, not args.no_blur)

if __name__ == "__main__":
    main()
