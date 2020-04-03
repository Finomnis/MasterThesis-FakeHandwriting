#!/usr/bin/env python3

import argparse
import random
import os
import shutil

from tqdm import tqdm


def main():

    parser = argparse.ArgumentParser(description='Splits a folder of images in a train and test set')
    parser.add_argument('input', help='The input image folder')
    parser.add_argument('train', help='The output training folder')
    parser.add_argument('test', help='The output test folder')
    parser.add_argument('--testSize', type=int, default=100, help='The size of the test set')
    args = parser.parse_args()
    print(args)

    if not os.path.isdir(args.train):
        os.makedirs(args.train)
    if not os.path.isdir(args.test):
        os.makedirs(args.test)

    fileNames = [f for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f))]

    testSet = set(random.sample(fileNames, args.testSize))

    for fileName in tqdm(fileNames):
        inFile = os.path.join(args.input, fileName)
        outFile = os.path.join(args.train, fileName)
        if fileName in testSet:
            outFile = os.path.join(args.test, fileName)

        shutil.copyfile(inFile, outFile)


if __name__ == "__main__":
    main()
