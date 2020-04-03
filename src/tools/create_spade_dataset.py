
import argparse
import os
import random

from multiprocessing import Pool

from tqdm import tqdm
from PIL import Image, ImageOps

from torchvision import transforms

import numpy as np

from algorithms.image_cut_and_fill import generateCutOffset, cutImageWithOffset
from pipeline.skeletonization import Skeletonizer
from algorithms.thicken_lines import thicken_lines

from matplotlib import pyplot as plt


class BackgroundGenerator:

    def __init__(self, sizeX, sizeY, folderName):
        self.imgs = [Image.open(os.path.join(folderName, f)) for f in os.listdir(folderName)
                     if os.path.isfile(os.path.join(folderName, f))]
        self.sizeX = sizeX
        self.sizeY = sizeY

    def __iter__(self):
        return self

    def __next__(self):
        img = random.choice(self.imgs)

        size = max(self.sizeX, self.sizeY)

        imgTransforms = transforms.Compose([
            transforms.RandomResizedCrop((int(1.5*size), int(1.5*size))),
            # transforms.RandomRotation(180),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.CenterCrop((self.sizeY, self.sizeX)),
            transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.3, 1.3], saturation=[0, 1.0], hue=0.2)
        ])

        return imgTransforms(img)


class TextGenerator:
    def __init__(self, sizeX, sizeY, folderName, skeletonFolderName):
        self.imgs = [f for f in os.listdir(folderName) if os.path.isfile(os.path.join(folderName, f))]
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.folderName = folderName
        self.skeletonFolderName = skeletonFolderName

    def __iter__(self):
        return self

    def __next__(self):
        fileName = random.choice(self.imgs)
        imgName = os.path.join(self.folderName, fileName)
        skeletonName = os.path.join(self.skeletonFolderName, fileName)

        img = Image.open(imgName)
        skelImg = Image.open(skeletonName)

        assert(img.size == skelImg.size)

        cutOffset = generateCutOffset((self.sizeX, self.sizeY), img.size)

        img = cutImageWithOffset((self.sizeX, self.sizeY), img, cutOffset)
        skelImg = cutImageWithOffset((self.sizeX, self.sizeY), skelImg, cutOffset)

        img = ImageOps.invert(img)
        imgTransforms = transforms.Compose([
            # transforms.RandomResizedCrop((int(1.5*self.sizeY), int(1.5*self.sizeX))),
            # transforms.RandomRotation(180),
            # transforms.RandomCrop((self.sizeY, self.sizeX), pad_if_needed=True),
            transforms.ColorJitter(brightness=[0.5, 1.5], saturation=[0, 1.0], hue=0.5)
        ])
        img = ImageOps.invert(imgTransforms(img))

        skelImgSpade = Skeletonizer.skeletonize_sharp(skelImg)
        skelImgSpade = Image.fromarray(thicken_lines(skelImgSpade, 6).astype('uint32'))

        if False:
            figure1 = plt.figure('Test')

            plt.subplot(3, 1, 1)
            plt.imshow(img)
            plt.subplot(3, 1, 2)
            plt.imshow(skelImg, cmap='binary', vmax=10)
            plt.subplot(3, 1, 3)
            plt.imshow(skelImgSpade, cmap='binary', vmax=10)

            plt.show()
            exit(1)

        return img, skelImgSpade, skelImg


def blend(foreground, background):
    foreground = np.asarray(ImageOps.invert(foreground)).astype(float)
    background = np.asarray(ImageOps.invert(background)).astype(float)

    result = foreground + background

    result = np.clip(result, 0.0, 255.0).astype('uint8')

    return ImageOps.invert(Image.fromarray(result))


threadBackgroundGenerator = None
threadTextGenerator = None
threadOutputImgPath = None
threadOutputSkelPath = None
threadOutputPix2pixSkelPath = None


def initThread(args, outputImgPath, outputSkelPath, outputPix2pixSkelPath):
    print("Initializing thread ...")

    global threadBackgroundGenerator
    global threadTextGenerator
    global threadOutputImgPath
    global threadOutputSkelPath
    global threadOutputPix2pixSkelPath

    threadBackgroundGenerator = BackgroundGenerator(args.out_size[0], args.out_size[1], args.backgrounds)
    threadTextGenerator = TextGenerator(args.out_size[0], args.out_size[1], args.input, args.skeletons)

    threadOutputImgPath = outputImgPath
    threadOutputSkelPath = outputSkelPath
    threadOutputPix2pixSkelPath = outputPix2pixSkelPath


def generateImage(i):

    global threadBackgroundGenerator
    global threadTextGenerator
    global threadOutputImgPath
    global threadOutputSkelPath
    global threadOutputPix2pixSkelPath

    (text, skel, skelPix2Pix) = next(threadTextGenerator)
    bg = next(threadBackgroundGenerator)
    img = blend(text, bg)

    skel = skel.convert('RGB')

    img.save(os.path.join(threadOutputImgPath, str(i) + '.png'))
    skel.save(os.path.join(threadOutputSkelPath, str(i) + '.png'))
    skelPix2Pix.save(os.path.join(threadOutputPix2pixSkelPath, str(i) + '.png'))


def main():

    parser = argparse.ArgumentParser(description='Takes a bunch of input images and .')
    parser.add_argument('input', help='The offline handwriting input folder')
    parser.add_argument('skeletons', help='The skeletonized version of the handwriting input folder')
    parser.add_argument('output', default=None, help='The output folder')
    parser.add_argument('--backgrounds', help='The backgrounds folder', required=True)
    parser.add_argument('--out-size', nargs=2, type=int, default=[256, 256], metavar=('SIZE_X', 'SIZE_Y'),
                        help='The size of the generated images')
    parser.add_argument('--samples', type=int, default=50000, help='The number of samples to generate')
    args = parser.parse_args()
    print(args)

    outputImgPath = os.path.join(args.output, 'img')
    outputSkelPath = os.path.join(args.output, 'skel')
    outputPix2pixSkelPath = os.path.join(args.output, 'skel_pix2pix')

    if not os.path.isdir(outputImgPath):
        os.makedirs(outputImgPath)
    if not os.path.isdir(outputSkelPath):
        os.makedirs(outputSkelPath)
    if not os.path.isdir(outputPix2pixSkelPath):
        os.makedirs(outputPix2pixSkelPath)

    # backgroundGenerator = BackgroundGenerator(args.out_size[0], args.out_size[1], args.backgrounds)
    # textGenerator = TextGenerator(args.out_size[0], args.out_size[1], args.input, args.skeletons)
    # (text, skel) = next(textGenerator)
    # bg = next(backgroundGenerator)
    # blend(text, bg).show()
    # skel.show()

    with Pool(None, initThread, (args, outputImgPath, outputSkelPath, outputPix2pixSkelPath)) as pool:

        tasks = list()

        for i in range(args.samples):
            tasks.append(pool.apply_async(generateImage, (i,)))

        for task in tqdm(tasks):
            task.wait()


if __name__ == "__main__":
    main()
