import os
import glob
import imageio
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
 
#############################################################################
# TODO: Add additional imports
#############################################################################
from sklearn.cluster import KMeans

def dist2(x, c):
    """ Calculates the pairwise Euclidian distance between every row in x and every row in c.

    Inputs:
    - x: NxD matrix where each row represents a feature vector
    - c: MxD matrix where each row represents a feature vector

    Outputs:
    - d: NxM where the value at (i, j) represents the Euclidian distance between features x_i and c_j 
    """
    ...
    ndata, dimx = x.shape
    ncentres, dimc = c.shape
    if dimx != dimc:
        raise NameError("Data dimension does not match dimension of centres")

    n2 = (
        np.transpose(
            np.dot(
                np.ones((ncentres, 1)),
                np.transpose(np.sum(np.square(x), 1).reshape(ndata, 1)),
            )
        )
        + np.dot(
            np.ones((ndata, 1)),
            np.transpose(np.sum(np.square(c), 1).reshape(ncentres, 1)),
        )
        - 2 * np.dot(x, np.transpose(c))
    )

    n2[n2 < 0] = 0
    return n2


def match_descriptors(desc1, desc2):
    """ Finds the `descriptors2` that best match `descriptors1`
    
    Inputs:
    - desc1: NxD matrix of feature descriptors
    - desc2: MxD matrix of feature descriptors

    Returns:
    - indices: the index of N descriptors from `desc2` that 
               best match each descriptor in `desc1`
    """
    N = desc1.shape[0]
    indices = np.zeros((N,), dtype="int64")
    
    ############################
    # TODO: Add your code here #
    ############################
    # To generate candidate matches, find patches that have the most similar appearance (e.g., lowest SSD)
    # Simplest approach: compare them all, take the closest (or closest k, or within a thresholded distance)
    distance_matrix = dist2(desc1, desc2)
    indices = np.argmin(distance_matrix, axis=1)

    ############################
    #     END OF YOUR CODE     #
    ############################
    
    return indices


def calculate_bag_of_words_histogram(vocabulary, descriptors):
    """ Calculate the bag-of-words histogram for the given frame descriptors.
    
    Inputs:
    - vocabulary: kxd array representing a visual vocabulary
    - descriptors: nxd array of frame descriptors
    
    Outputs:
    - histogram: k-dimensional bag-of-words histogram
    """
    k = vocabulary.shape[0]
    histogram = np.zeros((k,), dtype="int64")

    ############################
    # TODO: Add your code here #
    ############################
    
    if descriptors.shape[0] == 0:
        print("empty descriptors")
        return histogram
    
    closest_words = match_descriptors(descriptors, vocabulary)
    unique, counts = np.unique(closest_words, return_counts = True)

    it = np.nditer(unique, flags=['f_index'])
    for x in it:
        histogram[x] = counts[it.index]
    
    ############################
    #     END OF YOUR CODE     #
    ############################

    return histogram

def calculate_normalized_scalar_product(hist1, hist2):
    """ Caculate the normalized scalar product between two histograms.
    
    Inputs:
    - hist1: k-dimensional array
    - hist2: k-dimensional array
    
    Outputs:
    - score: the normalized scalar product described above
    """
    score = 0
    
    ############################
    # TODO: Add your code here #
    ############################

    score = hist1.dot(hist2) / (np.sqrt(hist1.dot(hist1)) * np.sqrt(hist2.dot(hist2)))
    
    if np.isnan(score):
        score = 0
    
    ############################
    #     END OF YOUR CODE     #
    ############################
    
    return score