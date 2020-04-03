#!/usr/bin/env python3

from zipfile import ZipFile

from tqdm import tqdm

import re
import argparse
import os
import sys


def main():

    parser = argparse.ArgumentParser(description='Extracts the cvl dataset from its zip file, exporting the train and test images to respective folders and extracting their text labels')
    parser.add_argument('input', help='The input cvl dataset zip file')
    parser.add_argument('output', help='The output folder, where all the data will be stored')
    args = parser.parse_args()
    print(args, file=sys.stderr)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    trainSet_regex = re.compile(r"[\w\-]+/trainset/lines/\d+/(([\w\-]+)\.tif)")
    testSet_regex = re.compile(r"[\w\-]+/testset/lines/\d+/(([\w\-]+)\.tif)")
    trainSet = []
    testSet = []

    with ZipFile(args.input) as zipFile:

        print("Reading file list ...")
        for info in zipFile.infolist():

            trainSet_match = trainSet_regex.match(info.filename)
            if trainSet_match:
                trainSet.append((info.filename, trainSet_match.group(1), trainSet_match.group(2)))

            testSet_match = testSet_regex.match(info.filename)
            if testSet_match:
                testSet.append((info.filename, testSet_match.group(1), testSet_match.group(2)))

        def process_set(name, items):
            print("Extracting '" + name + "' ...")

            extractionPath = os.path.join(args.output, name)
            if not os.path.isdir(extractionPath):
                os.makedirs(extractionPath)

            for item_zipName, item_fileName, item_id in tqdm(items, file=sys.stdout):
                with open(os.path.join(extractionPath, item_fileName), 'wb') as outFile:
                    outFile.write(zipFile.open(item_zipName).read())

        process_set('train', trainSet)
        process_set('test', testSet)

if __name__ == "__main__":
    main()
