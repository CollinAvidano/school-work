#export
import math
import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2gray
from skimage.feature import canny
from scipy import ndimage, signal
from typing import Tuple
import matplotlib.pyplot as plt
from imageio import imread, imsave

def compute_gradients(img):
    gx = gy = np.zeros_like(img)

    dx = [[1, 0, -1],[ 1, 0, -1], [1, 0, -1]]
    dy = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]

    gx = ndimage.correlate(img, dx, mode='constant', cval=0.0)
    gy = ndimage.correlate(img, dy, mode='constant', cval=0.0)
  
    return gx, gy


# centers = detect_circles(img, radius, use_gradient) - Given an RGB image img, a target
# radius that specifies the size of circle we are looking for, and a flag use_gradient that allows the user
# to optionally exploit the gradient direction measured at the edge points. The output centers is an N x
# 2 matrix in which each row lists the (x, y) position of a detected circles’ center. Save this function
# in a file called submissionDetectCircles.py and submit it.

h_sigma = 4
h_low_threshold = 0.2
h_vote_percent = 0.8

def detect_circles(img : np.ndarray, radius : int, use_gradient : bool):
    
    if img.ndim > 2:
        img = rgb2gray(img)
    
    edges = canny(img, sigma = h_sigma, low_threshold = h_low_threshold)
    
    gradient_dir = np.zeros_like(img)
    if use_gradient:
        gx, gy = compute_gradients(img)
        gradient_dir = np.arctan2(gy, gx)
        
    
    accumulator = np.zeros(img.shape,dtype='int')

    for iy,ix in np.ndindex(edges.shape):
        if edges[iy, ix] == True:
            if use_gradient:
                theta = gradient_dir[iy, ix]
                a = int(round(ix - radius * math.cos(theta)))
                b = int(round(iy - radius * math.sin(theta)))
                # if in image. vote
                if a > 0 and a < accumulator.shape[1] and b > 0 and b < accumulator.shape[0]:
                    accumulator[b, a] += 1
            else:
                for theta in range(360):
                    a = int(round(ix - radius * math.cos((theta * math.pi) / 180)))
                    b = int(round(iy - radius * math.sin((theta * math.pi) / 180)))
                    # if in image. vote
                    if a > 0 and a < accumulator.shape[1] and b > 0 and b < accumulator.shape[0]:
                        accumulator[b, a] += 1

    # ATAN2 is for gradient angle if use gradient is set
    # if use gradient is set you  only test the rounded
    # theta value from arctan 2 at each edge pixel
             

    vote_threshold = int(round(np.max(accumulator) * h_vote_percent))
    
    results_b , results_a = np.where(accumulator > vote_threshold)

    cartesian_results = np.stack((results_a, results_b), axis=1)

    return cartesian_results

                        