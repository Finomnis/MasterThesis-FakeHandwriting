#!/usr/bin/env python3

import argparse
import os

from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from pipeline.skeletonization import Skeletonizer
from pipeline.sampling import sample_to_penpositions
from pipeline.graves import GravesWriter
from pipeline.align import align
from pipeline.render_skeleton import render_skeleton
from pipeline.colorization import Colorizer
from pipeline.split_foreground_background import ForegroundExtractor, BackgroundFiller
from pipeline.pen_style_transfer import PenStyleTransfer

from datastructures.PenPosition import plotPenPositions


def main():

    parser = argparse.ArgumentParser(description='The first working version (without pen style transfer). Modifies the content of an image.')
    parser.add_argument('--text-in', help='The input text', required=True)
    parser.add_argument('--text-out', help='The output text', required=True)
    parser.add_argument('input', help='The input file')
    parser.add_argument('--output', help='The output file')
    args = parser.parse_args()
    print(args)

    inputImg = Image.open(args.input)

    #with ForegroundExtractor() as foregroundExtractor:
    #    foreground = foregroundExtractor.extract_foreground(inputImg)
    #    backgroundMask = foregroundExtractor.create_background_mask(foreground)
    #    #print(np.asarray(foreground) / 255.0)
    #    #print(np.asarray(backgroundMask))
    #    background_masked = (np.asarray(inputImg)/255.0) * backgroundMask

    #with BackgroundFiller() as backgroundFiller:
    #    background = backgroundFiller.fill(inputImg, backgroundMask)

    with Skeletonizer() as skeletonizer:
        skeletonBlurImg = skeletonizer.skeletonize_blurred(inputImg)
        skeletonImg = skeletonizer.skeletonize_sharp(skeletonBlurImg)

    penPositions = sample_to_penpositions(skeletonImg)

    with GravesWriter() as writer:
        newPenPositions = writer.write(args.text_out, args.text_in, penPositions)

    newPenPositions = align(newPenPositions, penPositions)

    newSkeletonBlurImg, newSkeletonImg = render_skeleton(newPenPositions, inputImg.size)

    #with Colorizer() as colorizer:
    #    outputImg = colorizer.colorize(newSkeletonBlurImg)

    with PenStyleTransfer() as penStyleTransfer:
        outputImg = penStyleTransfer.transferStyle(newSkeletonBlurImg, inputImg)

    print("Done. Displaying results ...")

    plt.figure('Full Pipeline', figsize=(16, 9))
    plt.subplot(3, 2, 1)
    plt.imshow(inputImg)

    #plt.subplot(3, 3, 2)
    #plt.imshow(foreground)
    #plt.subplot(3, 3, 5)
    #plt.imshow(background_masked)
    #plt.subplot(3, 3, 8)
    #plt.imshow(background)


    plt.subplot(3, 2, 3)
    plt.imshow(skeletonBlurImg)
    plt.subplot(3, 2, 5)
    plt.imshow(skeletonImg, cmap='binary', vmax=10)
    plotPenPositions(penPositions)
    plt.subplot(3, 2, 6)
    plt.imshow(newSkeletonImg, cmap='binary', vmax=256*10)
    plotPenPositions(newPenPositions)
    plt.subplot(3, 2, 4)
    plt.imshow(newSkeletonBlurImg)
    plt.subplot(3, 2, 2)
    plt.imshow(outputImg)
    plt.show()

    if args.output:
        Image.fromarray(255 - skeletonImg.astype(np.uint8)*255).save(args.output)


if __name__ == "__main__":
    main()
