#!/usr/bin/env python3

import argparse
import os

from tqdm import tqdm

from PIL import Image, ImageOps, ImageFilter

import random
import numpy as np

# import matplotlib.pyplot as plt
from scipy.ndimage import morphology


def renderImg(inputImg):

    lineWidth = random.randint(2, 7)
    blur = random.uniform(0, 3)
    magenta = random.uniform(0.5, 0.9)
    yellow = random.uniform(0.3, 0.7)
    cyan = random.uniform(0.5, 0.9)

    stencil = np.zeros((2*lineWidth+1, 2*lineWidth+1), dtype=np.uint8)
    for y in range(stencil.shape[0]):
        pY = (y - lineWidth) / lineWidth
        for x in range(stencil.shape[1]):
            pX = (x - lineWidth) / lineWidth
            if pX*pX+pY*pY < 1:
                stencil[y, x] = 1

    imgArray = np.asarray(ImageOps.invert(inputImg))
    morphedArray = morphology.binary_dilation(imgArray, structure=stencil)

    morphedArray = morphedArray.astype(np.uint8) * 255
    morphedArray = np.expand_dims(morphedArray, 2)
    morphedArray = np.matmul(morphedArray, np.array([[[cyan, magenta, yellow]]]))
    morphedImg = ImageOps.invert(Image.fromarray(morphedArray.astype(np.uint8)))
    filteredImg = morphedImg.filter(ImageFilter.GaussianBlur(blur))

    return filteredImg


def main():
    parser = argparse.ArgumentParser(description='Converts offline skeletons to thick skeletons.')
    parser.add_argument('input', help='The input dataset')
    parser.add_argument('output', help='The output dataset')
    args = parser.parse_args()
    print(args)

    fileNames = [f for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f))]
    if not fileNames:
        raise RuntimeError("No input files found!")

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    for fileName in tqdm(fileNames):
        inputImg = Image.open(os.path.join(args.input, fileName))
        outputImg = renderImg(inputImg)
        #inputImg.show()
        #outputImg.show()
        outputImg.save(os.path.join(args.output, fileName))


if __name__ == "__main__":
    main()
