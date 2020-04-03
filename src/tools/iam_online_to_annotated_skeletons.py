#!/usr/bin/env python3

from tqdm import tqdm

from PIL import Image, ImageDraw
import numpy as np

import tarfile
from xml.etree import ElementTree
from html import unescape

import argparse
import os
import sys
import svgwrite

from datastructures.PenPosition import PenPosition
from algorithms.render_penpositions_to_svg import render_penpositions_to_svg

def main():

    parser = argparse.ArgumentParser(description="""\
Extracts the iam-online dataset from its zip file and renders it to skeletons.
It further exports the text labels of the skeletons and stores them in an npz.\
""")
    parser.add_argument('input_xml', help='The file \'original-xml-part.tar.gz\'')
    parser.add_argument('input_lineStrokes', help='The file \'lineStrokes-all.tar.gz\'')
    parser.add_argument('output', help='The output folder, where all the data will be stored')
    args = parser.parse_args()
    #print(args, file=sys.stderr)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    img_dir = os.path.join(args.output, 'all')
    svg_dir = os.path.join(args.output, 'svg')

    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    if not os.path.isdir(svg_dir):
        os.makedirs(svg_dir)

    textLabels = dict()

    print('Opening \'' + args.input_xml + '\'...')
    with tarfile.open(args.input_xml) as xmlFile:
        members_list = xmlFile.getmembers()
        print('Extracting labels ...')
        for member in tqdm(members_list, file=sys.stdout):

            etree = ElementTree.parse(xmlFile.extractfile(member))

            for elem in etree.findall("./Transcription/TextLine"):
                elemId = elem.attrib['id']
                elemText = unescape(elem.attrib['text'])

                textLabels[elemId] = elemText

    print('Opening \'' + args.input_lineStrokes + '\'...')
    with tarfile.open(args.input_lineStrokes) as strokesFile:
        members_list = strokesFile.getmembers()
        print('Extracting and rendering strokes ...')
        for member in tqdm(members_list, file=sys.stdout):

            strokeId = os.path.splitext(os.path.basename(member.name))[0]

            strokeLabel = textLabels.get(strokeId)
            if not strokeLabel:
                continue

            #print(strokeId, strokeLabel)

            x_max = None
            x_min = None
            y_max = None
            y_min = None

            etree = ElementTree.parse(strokesFile.extractfile(member))
            strokes = []
            for elem in etree.findall("./StrokeSet/Stroke"):
                stroke = []
                for point in elem.findall("./Point"):
                    x = int(point.attrib['x'])
                    y = int(point.attrib['y'])
                    if x_max is None or x > x_max:
                        x_max = x
                    if y_max is None or y > y_max:
                        y_max = y
                    if x_min is None or x < x_min:
                        x_min = x
                    if y_min is None or y < y_min:
                        y_min = y

                    stroke.append((x, y))
                strokes.append(stroke)

            resize_factor = 5

            size = ((x_max-x_min+200)//resize_factor, (y_max-y_min+200)//resize_factor)
            img = Image.new(mode='1', size=size, color=1)

            penPositions = []

            draw = ImageDraw.Draw(img)
            for stroke in strokes:
                finalPositions = [((x-x_min+100)//resize_factor, (y-y_min+100)//resize_factor) for x, y in stroke]
                for pos in finalPositions:
                    penPositions.append(PenPosition(float(pos[0]), float(pos[1]), 0, " ", 0, 0))
                penPositions[-1].penUp = 1
                draw.line(finalPositions, fill=0)

            img.save(os.path.join(img_dir, strokeId + '.png'))

            svg = svgwrite.Drawing(os.path.join(svg_dir, strokeId + '.svg'),
                                   profile='tiny',
                                   size=(str(size[0]) + 'px', str(size[1]) + 'px'))
            svg.add(svg.rect(size=('100%', '100%'), fill='white'))

            render_penpositions_to_svg(penPositions, svg)

            svg.save()

    print('Saving labels ...')
    np.save(os.path.join(args.output, 'labels.npy'), textLabels)

    #print(textLabels)
    exit(0)


if __name__ == "__main__":
    main()
