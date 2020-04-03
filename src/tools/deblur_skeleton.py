#!/usr/bin/env python3

import argparse
import os

from PIL import Image

from pipeline.skeletonization import Skeletonizer

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tqdm import tqdm

def main():

    parser = argparse.ArgumentParser(description='Produces sharp skeleton images from blurred ones')
    parser.add_argument('input', help='The input folder')
    parser.add_argument('output', help='The output folder')
    args = parser.parse_args()

    fileNames = [f for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f))]
    if not fileNames:
        raise RuntimeError("No input files found!")

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    for fileName in tqdm(fileNames):

        img = Image.open(os.path.join(args.input, fileName)).convert('L')
        img_sharp_raw = Skeletonizer.skeletonize_sharp(img)

        img_sharp = Image.fromarray(255-img_sharp_raw.astype(np.uint8)*255, mode='L')
        img_sharp.save(os.path.join(args.output, fileName))

        #for thresh in range(100, 255, 5):
        #    img_sharp_raw, img_thresh_raw = Skeletonizer.skeletonize_sharp(img, threshold=thresh, return_threshold_img=True)
        #    img_sharp = Image.fromarray(255 - img_sharp_raw.astype(np.uint8) * 255, mode='L')
        #    img_thresh = Image.fromarray(255 - img_thresh_raw.astype(np.uint8) * 255, mode='L')
        #
        #    img_sharp.save(os.path.join(args.output, str(thresh) + "_" + fileName))
        #    img_thresh.save(os.path.join(args.output, str(thresh) + "_th_" + fileName))


if __name__ == "__main__":
    main()
