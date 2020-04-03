#!/usr/bin/env python3

import argparse
import os

from tqdm import tqdm
from PIL import Image, ImageOps
import numpy as np

from pipeline.colorization import Colorizer
from pipeline.skeletonization import Skeletonizer
from pipeline.render_skeleton import blur_skeleton


def main():
    parser = argparse.ArgumentParser(description='Takes a skeleton folder, creates CVL images from it and then re-skeletonizes it.')
    parser.add_argument('input', help='The input skeletons folder')
    parser.add_argument('output', help='The output folder')
    parser.add_argument('--image-folder', help='The folder for the intermediate images, speeds up computaton')
    args = parser.parse_args()
    print(args)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    if args.image_folder and not os.path.isdir(args.image_folder):
        os.makedirs(args.image_folder)

    fileNames = [f for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f))]

    with Colorizer() as colorizer:
        with Skeletonizer() as skeletonizer:
            for fileName in tqdm(fileNames):

                #print(fileName)
                fullName = os.path.join(args.input, fileName)
                lineImg = Image.open(fullName).convert('L')

                blurredLineImg = blur_skeleton(lineImg)

                colorImg = colorizer.colorize(blurredLineImg)
                #colorImg.show()

                if args.image_folder:
                    colorImg.save(os.path.join(args.image_folder, fileName))

                newLineImg_array = skeletonizer.skeletonize(colorImg)
                newLineImg = ImageOps.invert(Image.fromarray(newLineImg_array.astype(np.uint8)*255, mode='L')).convert('1')
                #newLineImg.show()

                newLineImg.save(os.path.join(args.output, fileName))

                #exit(0)




if __name__ == "__main__":
    main()
