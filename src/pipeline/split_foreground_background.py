
from ext.inpaint.Models import PConvInfilNet
from ext.inpaint.DataLoaders import MaskedImageDataset

import os
import gc

from PIL import Image
from skimage import morphology

from utils.add_path import add_path
from algorithms.thicken_lines import thicken_lines

import numpy as np

with add_path(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'ext', 'pix2pix')):
    from options.test_options import TestOptions
    from models import create_model
    from data.single_item_dataset import SingleItemDataset
    from util.util import tensor2im


class ForegroundExtractor:

    def __init__(self):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.pix2pixDir = os.path.join(self.dir_path, '..', 'ext', 'pix2pix')

        self.opts = TestOptions().parse(['--dataroot', '',
                                   '--model', 'pix2pix',
                                   '--checkpoints_dir',
                                   str(os.path.join(self.dir_path, '..', '..', 'results', 'pix2pix')),
                                   '--name', '1560510053_comb_to_pen'
                                   # '--name', '1556985661_skeletonization_improved'
                                   ])  # set model options
        # hard-code some parameters
        self.opts.num_threads = 0  # test code only supports num_threads = 1
        self.opts.batch_size = 1  # test code only supports batch_size = 1
        self.opts.preprocess = 'none'
        self.opts.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        self.opts.no_flip = True  # no flip; comment this line if results on flipped images are needed.
        self.opts.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

        self.model = create_model(self.opts)  # create a model given opt.model and other options
        self.model.setup(self.opts)  # regular setup: load and print networks; create schedulers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.model
        del self.opts
        gc.collect()

    def __createSingleImageDataset(self, img):
        with add_path(self.pix2pixDir):
            dataset = SingleItemDataset(self.opts, img)
            return dataset

    def extract_foreground(self, img):
        dataset = self.__createSingleImageDataset(img)

        self.model.set_input(dataset[0])
        self.model.test()
        visuals = self.model.get_current_visuals()

        result_img_array = tensor2im(visuals['fake_B'])

        result_img = Image.fromarray(result_img_array)
        return result_img

    @staticmethod
    def create_background_mask(foreground):

        foreground_array = np.asarray(foreground)
        means = foreground_array.mean(2, keepdims=True)
        background_pixels = means < 230

        #dilated = morphology.binary_dilation(background_pixels, selem=stencil)
        dilated = thicken_lines(background_pixels, 6)

        return (1.0 - dilated*1.0).repeat(foreground_array.shape[2], axis=2).astype(np.float32)


class BackgroundFiller:
    def __init__(self):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.model = PConvInfilNet(os.path.join(self.dir_path, '..', '..', 'results', 'inpaint-pconv', '1561256576_places2'), 'latest', use_dropout=False, ngf=64)

        # self.model = PConvInfilNet(os.path.join(self.dir_path, '..', '..', 'results', 'inpaint-pconv', '1562185514_places2_dropout'), 'latest', use_dropout=True, ngf=128)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.model
        gc.collect()

    def fill(self, img, mask):

        img_prepared, mask_prepared = MaskedImageDataset.single_image_prepare(img, mask)
        img_out, comp_out, mask_out = self.model.forward(img_prepared, mask_prepared)

        return MaskedImageDataset.to_img(img_out)
