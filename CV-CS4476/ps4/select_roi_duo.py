import os
import random
import glob

import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from skimage.color import rgb2gray

from selectRegion import roipoly

# specific frame dir and matdir
framesdir = "PS4Frames/frames/"
siftdir = "PS4SIFT/sift/"

fnames = glob.glob(siftdir + "*.mat")
fnames = [os.path.basename(name) for name in fnames]
num_fnames = len(fnames)

# fname = fnames[60]
# fname = fnames[800]
# fname = fnames[1200]
fname = fnames[4600]


mat = scipy.io.loadmat(os.path.join(siftdir, fname))

# read the associated image
im = imageio.imread(os.path.join(framesdir, str(mat["imname"])[2:-2]))

# now show how to select a subset of the features using polygon drawing.
print(
    "now use the mouse to draw a polygon, right click or double click to end it",
    flush=True,
)

plt.imshow(im)
roi = roipoly(color="r")
roi_verticies = np.array([roi.all_x_points, roi.all_y_points]).transpose()

np.save("region4.npy", roi_verticies)

indices = roi.get_indices(im, mat["positions"])

np.save("points4.npy", indices)
