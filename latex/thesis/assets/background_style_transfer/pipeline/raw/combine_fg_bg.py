#!/usr/bin/env python3

from PIL import Image
import numpy as np

#bg = Image.open('9166.png')
#fg = Image.open('input_2048.png')
bg = Image.open('background_infilled.png')
fg = Image.open('output_2048.png')


bg_arr = np.asarray(bg).astype(np.int32)
fg_arr = np.asarray(fg).astype(np.int32)

comb_arr = 255 - ((255 - bg_arr) + (255 - fg_arr))

comb_arr = np.maximum(0, np.minimum(255, comb_arr))

comb = Image.fromarray(comb_arr.astype(np.uint8))

comb.save('output_with_background.png')
