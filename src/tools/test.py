import matplotlib.pyplot as plt

import numpy as np

import sys
import argparse
from PIL import Image
from PIL import ImageDraw

from skimage.morphology import skeletonize, thin

from ext.deepwriting.dataset_hw import HandWritingDataset



#img = np.array([
#    [0, 0, 1, 0, 0],
#    [0, 0, 1, 0, 0],
#    [1, 1, 1, 1, 1],
#    [0, 0, 1, 0, 0],
#    [0, 0, 1, 0, 0],
#])
img = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
])
img = np.array([
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
])
img = np.array([
    [0, 1, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 0, 0],
    [1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0],
])
img = np.array([
    [0, 1, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1],
])
img = np.array([
    [1, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 1, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 1],
])

print(img, '\n')

print(thin(img).astype(int), '\n')

plt.subplot(121)
plt.imshow(img)
plt.plot([2,2],[5,4])
plt.subplot(122)
plt.imshow(thin(img))
plt.plot([2,2],[5,4])
plt.show()