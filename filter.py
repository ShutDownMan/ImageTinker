import cv2
import numpy as np
from scipy import ndimage

class Filter():
	def threshold(image, thresh, maxval=255, thresh_type=cv2.THRESH_BINARY):

		ret, imthresh = cv2.threshold(image, thresh, maxval, thresh_type)

		return imthresh

	def highpass(image, hp_type=1, sigma=3, reinforcement_factor=1.7):
		if hp_type == 0: # basic
			return cv2.GaussianBlur(image, (0, 0), sigma)

		if hp_type == 1: # high reinforcement
			return (image * (reinforcement_factor - 1)).astype('uint8')  + cv2.GaussianBlur(image, (0, 0), sigma)

		return None

	def lowpass(image, lp_type=0, sigma=3):
		if lp_type == 0: # average
			return cv2.GaussianBlur(image, (0, 0), sigma)

		if lp_type == 1: # median
			return cv2.medianBlur(image, sigma)

		return None

	def roberts(image):
		roberts_cross_v = np.array([[ 0, 0, 0 ], [ 0, 1, 0 ], [ 0, 0,-1 ]])

		roberts_cross_h = np.array([[ 0, 0, 0 ], [ 0, 0, 1 ], [ 0,-1, 0 ]])

		vertical = ndimage.convolve(image.astype('float'), roberts_cross_v)
		horizontal = ndimage.convolve(image.astype('float'), roberts_cross_h)

		return cv2.sqrt(cv2.pow(horizontal, 2) + cv2.pow(vertical, 2)).astype('uint8')

	def prewitt(image):
		kernelx = np.array([[1,1,1], [0,0,0], [-1,-1,-1]])
		kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

		vertical = ndimage.convolve(image.astype('double'), kernelx)
		horizontal = ndimage.convolve(image.astype('double'), kernely)

		return cv2.sqrt(cv2.pow(horizontal, 2) + cv2.pow(vertical, 2)).astype('uint8')

	def sobel(image, kernel_size=3):
		sobel_v = cv2.Sobel(image, cv2.CV_8U, 0, 1, kernel_size)
		sobel_h = cv2.Sobel(image, cv2.CV_8U, 1, 0, kernel_size)

		return sobel_h + sobel_v

	def canny(image, lower_thresh=100, upper_thresh=200):
		return cv2.Canny(image, lower_thresh, upper_thresh)

	def laplace(image, kernel_size=3):
		return cv2.Laplacian(image, cv2.CV_8U, kernel_size)

	def zerocross(image, kernel_size=3):
		LoG = cv2.Laplacian(image, cv2.CV_16S)

		minLoG = cv2.morphologyEx(LoG, cv2.MORPH_ERODE, np.ones((3,3)))
		maxLoG = cv2.morphologyEx(LoG, cv2.MORPH_DILATE, np.ones((3,3)))

		result = (np.logical_or(np.logical_and(minLoG < 0,  LoG > 0), np.logical_and(maxLoG > 0, LoG < 0)) * 255).astype('uint8')

		return result