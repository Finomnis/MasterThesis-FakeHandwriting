#!/usr/bin/env python3

import argparse
import os

from tqdm import tqdm
from PIL import Image, ImageOps
import numpy as np

from pipeline.skeletonization import Skeletonizer
from pipeline.render_skeleton import blur_skeleton


def main():
    parser = argparse.ArgumentParser(description='Takes an image folder and creates skeletons from it')
    parser.add_argument('input', help='The input folder')
    parser.add_argument('output', help='The output folder')
    args = parser.parse_args()
    print(args)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    fileNames = [f for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f))]

    with Skeletonizer() as skeletonizer:
        for fileName in tqdm(fileNames):
            #print(fileName)

            fullName = os.path.join(args.input, fileName)
            colorImg = Image.open(fullName).convert('RGB')
            lineImg_array = skeletonizer.skeletonize(colorImg)

            lineImg = ImageOps.invert(Image.fromarray((lineImg_array*255).astype(np.uint8), mode='L'))

            lineImgBlurred = blur_skeleton(lineImg)

            #colorImg.show()
            #lineImg.show()
            #lineImgBlurred.show()

            lineImgBlurred.save(os.path.join(args.output, fileName))

            #exit(0)




if __name__ == "__main__":
    main()
