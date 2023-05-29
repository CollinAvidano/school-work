import os
import random

import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from skimage.color import rgb2gray

from selectRegion import roipoly

# specific frame dir and matdir
framesdir = "PS4Frames/frames/"
siftdir = "PS4SIFT/sift/"
fname = "twoFrameData.mat"

mat = scipy.io.loadmat(os.path.join(matdir, fname))

# read the associated image
im = mat["im1"]

# now show how to select a subset of the features using polygon drawing.
print(
    "now use the mouse to draw a polygon, right click or double click to end it",
    flush=True,
)

plt.imshow(im)
roi = roipoly(color="r")
roi_verticies = np.array([roi.all_x_points, roi.all_y_points]).transpose()

np.save("region.npy", roi_verticies)

indices = roi.get_indices(im, mat["positions1"])

np.save("points.npy", indices)