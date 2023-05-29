import skimage
import math
import numpy as np
import matplotlib.pyplot as plot
from imageio import imread, imsave
from scipy.ndimage.filters import convolve
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.draw import circle_perimeter

def detectCircles(img, radius, useGradient):
	edges = canny(skimage.img_as_float(rgb2gray(img)), sigma = 2) #sigma is a parameter to change depending on image
	image = skimage.img_as_float(rgb2gray(img))
	rows, columns = edges.shape
	accumulator = np.zeros((rows, columns))
	for row in range(0, rows):
		for column in range(0, columns):
			if edges[row, column] == 1:
				if useGradient == 0:
					for theta in np.arange(0, 2 * math.pi, .01):
						a = int(round(row + radius * math.sin(theta)))
						b = int(round(column + radius * math.cos(theta)))
						if a >= 0 and b >= 0 and b < (columns - 1) and a < (rows - 1): 
							accumulator[a, b] = accumulator[a, b] + 1
				else:
						gx = convolve(image, np.array([[1,-1]]), mode="wrap") #np.gradient was getting errors for me so I stole the old convolution from lecture for gradient detection haha
						gx[gx == 0] = .00000001 #take that divide by 0!
						gy = (convolve(image, np.array([[1],[-1]]),mode="wrap"))
						gradient_direction = np.arctan(gy/gx)
						theta = gradient_direction[row, column] #right side of circle
						a = int(round(row + radius * math.sin(theta)))
						b = int(round(column + radius * math.cos(theta)))
						if a >= 0 and b >= 0 and b < (columns - 1) and a < (rows - 1):
							accumulator[a, b] = accumulator[a, b] + 1
						theta = gradient_direction[row, column] - math.pi #left side of circle (must do because arctan be this way)
						a = int(round(row + radius * math.sin(theta)))
						b = int(round(column + radius * math.cos(theta)))
						if a >= 0 and b >= 0 and b < (columns - 1) and a < (rows - 1):
							accumulator[a, b] = accumulator[a, b] + 1
	plot.imshow(accumulator)
	plot.show()
	return np.column_stack(np.where(accumulator >= .8 * np.amax(accumulator))) #The .8 is an arbitrary acceptance threshhold
