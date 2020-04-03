#!/usr/bin/env python3

import argparse
import os

from PIL import Image

import matplotlib.pyplot as plt

from pipeline.skeletonization import Skeletonizer
from pipeline.sampling import sample_to_penpositions
from pipeline.graves import GravesWriter
from pipeline.align import align
from pipeline.render_skeleton import render_skeleton
from pipeline.pen_style_transfer import PenStyleTransfer

from datastructures.PenPosition import plotPenPositions

import numpy as np

import json


def main():

    parser = argparse.ArgumentParser(description='Generates the sentence \'Thanks for your attention!\', for the presentation')
    parser.add_argument('input', help='The input folder')
    parser.add_argument('output', help='The output folder')
    args = parser.parse_args()
    print(args)

    text_out = "Thanks for your attention!"

    skeletonizer = Skeletonizer()
    writer = GravesWriter()
    penStyleTransfer = PenStyleTransfer()

    fileNames = [f for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f)) and not f.endswith(".json")]

    with open(os.path.join(args.input, "text.json")) as fil:
        textContent = json.load(fil)
    print(textContent)

    for fileName in fileNames:
        inputImgRaw = Image.open(os.path.join(args.input, fileName))

        inputImg = Image.new('RGB', (2048,256), (255,255,255))
        inputImg.paste(inputImgRaw, (int(1024 - inputImgRaw.width/2), int(128-inputImgRaw.height/2)))

        skeletonBlurImg = skeletonizer.skeletonize_blurred(inputImg)
        skeletonImg = skeletonizer.skeletonize_sharp(skeletonBlurImg)

        penPositions = sample_to_penpositions(skeletonImg)

        text_in = textContent[fileName]
        newPenPositions = writer.write(text_out, text_in, penPositions)

        newPenPositions = align(newPenPositions, penPositions)

        newSkeletonBlurImg, newSkeletonImg = render_skeleton(newPenPositions, inputImg.size)

        outputImg = penStyleTransfer.transferStyle(newSkeletonBlurImg, inputImg)

        print("Done. Displaying results ...")

        # plt.figure('Full Pipeline', figsize=(16, 9))
        # plt.subplot(3, 2, 1)
        # plt.imshow(inputImg)
#
        # #plt.subplot(3, 3, 2)
        # #plt.imshow(foreground)
        # #plt.subplot(3, 3, 5)
        # #plt.imshow(background_masked)
        # #plt.subplot(3, 3, 8)
        # #plt.imshow(background)
#
#
        # plt.subplot(3, 2, 3)
        # plt.imshow(skeletonBlurImg)
        # plt.subplot(3, 2, 5)
        # plt.imshow(skeletonImg, cmap='binary', vmax=10)
        # plotPenPositions(penPositions)
        # plt.subplot(3, 2, 6)
        # plt.imshow(newSkeletonImg, cmap='binary', vmax=256*10)
        # plotPenPositions(newPenPositions)
        # plt.subplot(3, 2, 4)
        # plt.imshow(newSkeletonBlurImg)
        # plt.subplot(3, 2, 2)
        # plt.imshow(outputImg)
        # plt.show()

        outputImg.save(os.path.join(args.output, fileName))


if __name__ == "__main__":
    main()
