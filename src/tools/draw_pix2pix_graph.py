#!/usr/bin/env python3

import argparse
import re

from tqdm import tqdm

from datastructures.TrainingGraphPlotter import TrainingGraphPlotter

regexMatcher = re.compile(r"^(\w*): (False|True|-?[\d.]*),?( .*)?")


class LogEntry:
    def __init__(self, epoch, iters, data):
        self.epoch = epoch
        self.iters = iters
        self.data = data


def parseToDict(line):
    if not line:
        return {}
    line = line.strip()
    if not line:
        return {}

    # print(line)
    result = regexMatcher.match(line)
    # print(result.groups())

    results = parseToDict(result.group(3))

    try:
        value = int(result.group(2))
    except ValueError:
        try:
            value = float(result.group(2))
        except ValueError:
            if result.group(2) == 'True':
                value = True
            elif result.group(2) == 'False':
                value = False
            else:
                value = 0
                print("Unable to parse '" + result.group(2) + "' from line '" + line + "'!")
                exit(1)

    results[result.group(1)] = value

    return results


def parseLine(line):

    if not line.startswith('('):
        return None
    line = line[1:]

    header, data = line.split(") ")

    header = header.strip()
    data = data.strip()

    headerDict = parseToDict(header)
    dataDict = parseToDict(data)

    return LogEntry(headerDict['epoch'], headerDict['iters'], dataDict)


def main():
    parser = argparse.ArgumentParser(description='Reads a pix2pix-like loss_log.txt file and creates a graph from it.')
    parser.add_argument('input', help='The loss_log.txt file')
    args = parser.parse_args()
    print(args)

    log = list()
    logColumns = set()
    epochLen = 0
    with open(args.input, 'r') as lossFile:
        for line in tqdm(lossFile.readlines()):
            logEntry = parseLine(line.strip())
            if not logEntry:
                continue

            log.append(logEntry)

            # Track which columns we have
            for logColumn in logEntry.data:
                logColumns.add(logColumn)

            # Track how long our longest epoch is, to convert epoch+iters to an epoch float
            epochLen = max(logEntry.iters+1, epochLen)

    graphPlotter = TrainingGraphPlotter(logColumns, figureTitle=args.input, maxNumberNodes=400)

    for logElem in log:
        # Convert epoch+iters to epoch float
        epoch = logElem.epoch + logElem.iters / epochLen

        # Add data to plotter
        graphPlotter.addOrderedDataPoint(epoch, logElem.data)

    graphPlotter.plot()


if __name__ == "__main__":
    main()
