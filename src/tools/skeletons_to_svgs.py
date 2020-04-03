#!/usr/bin/env python3

import argparse
import os

from tqdm import tqdm
from PIL import Image, ImageOps

from pipeline.sampling import sample_to_penpositions
from algorithms.render_penpositions_to_svg import render_penpositions_to_svg

import svgwrite


def main():
    parser = argparse.ArgumentParser(description='Converts a dataset of skeletons to a graves dataset.')
    parser.add_argument('input', help='The input skeletons folder')
    parser.add_argument('output', help='The output folder')
    parser.add_argument('--sort-by', default='mean', help='The sorting order. Can be one of \'first\', \'last\', \'leftmost\', \'rightmost\', \'mean\'.')
    args = parser.parse_args()
    print(args)

    fileNames = [f for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f))]

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    for fileName in tqdm(fileNames):
        #print(fileName)
        strokeId = os.path.splitext(os.path.basename(fileName))[0]
        # print(strokeId, lineText)

        fullName = os.path.join(args.input, fileName)
        lineImg = ImageOps.invert(Image.open(fullName).convert('L'))

        # print('SkeletonImg:', lineId, lineImg, lineText)
        penPositions = sample_to_penpositions(lineImg, args.sort_by)

        svg = svgwrite.Drawing(os.path.join(args.output, strokeId + '.svg'),
                               profile='tiny',
                               size=(str(lineImg.width) + 'px', str(lineImg.height) + 'px'))
        svg.add(svg.rect(size=('100%', '100%'), fill='white'))

        render_penpositions_to_svg(penPositions, svg)

        svg.save()

        #exit(1)

if __name__ == "__main__":
    main()
