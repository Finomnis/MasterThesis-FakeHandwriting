#!/usr/bin/env python3

import argparse
import os

from tqdm import tqdm
from PIL import Image

from pipeline.skeletonization import Skeletonizer
from pipeline.sampling import sample_to_penpositions

from datastructures.IamOfflineDataset import IamOfflineDataset
from datastructures.ConvertedGravesDataset import ConvertedGravesDataset


def main():
    parser = argparse.ArgumentParser(description='Converts the IAM-Offline dataset to a graves version.')
    parser.add_argument('input', help='The input dataset')
    parser.add_argument('output', help='The output dataset')
    parser.add_argument('--output-skeletons', required=True, help='An output folder for skeletonization results')
    args = parser.parse_args()
    print(args)

    iamOfflineDataset = IamOfflineDataset(args.input)

    if args.output_skeletons and not os.path.isdir(args.output_skeletons):
        os.makedirs(args.output_skeletons)

    skeletonDataset = dict()

    with Skeletonizer() as skeletonizer:
        for formId, formImg in tqdm(iamOfflineDataset.formsIterator()):
            outFile = os.path.join(args.output_skeletons, formId + '.png')
            outSkelFile = os.path.join(args.output_skeletons, formId + '_skel.png')

            skeletonBlurImg = None
            skeletonImg = None

            # Use cacheing if skeleton dir is provided
            if os.path.isfile(outFile):
                skeletonBlurImg = Image.open(outFile)
                if os.path.isfile(outSkelFile):
                    skeletonImg = Image.open(outSkelFile)
                    print('Found cached file: ' + outSkelFile)

            if not skeletonBlurImg:
                skeletonBlurImg = skeletonizer.skeletonize_blurred(formImg)
                skeletonBlurImg.save(outFile)

            if not skeletonImg:
                skeletonImgRaw = skeletonizer.skeletonize_sharp(skeletonBlurImg)
                skeletonImg = Image.fromarray(skeletonImgRaw.astype('uint8')*255)
                skeletonImg.save(outSkelFile)

            skeletonDataset[formId] = outSkelFile

    outputDataset = ConvertedGravesDataset()

    def imgLoader(form_id):
        return Image.open(skeletonDataset[form_id])

    for (lineId, lineImg, lineText) in tqdm(iamOfflineDataset.linesIterator(imgLoader)):
        # print('SkeletonImg:', lineId, lineImg, lineText)
        penPositions = sample_to_penpositions(lineImg)
        outputDataset.addSample(penPositions, lineText)

    outputDataset.save(args.output)


if __name__ == "__main__":
    main()
