import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

input_image = None
result_image = None

# Setting up arguments
parser = argparse.ArgumentParser(description='Tinkering with images.')

parser.add_argument('-i', '--input', metavar='FILE_PATH', required=True)

parser.add_argument('-t', '--threshold', metavar='ARG_VALUE', dest='threshold_args', type=int, nargs='+')

parser.add_argument('-g', '--grayscale', action='store_const', default=0, const=1, help='output as grayscale')

parser.add_argument('-hp', '--highpass', choices=['basic', 'high_reinforcement'], nargs='?', const='basic')
parser.add_argument('-hp_s', '--hp_sigma', metavar='sigma', type=int, nargs='?', default=3)
parser.add_argument('-hp_hf', '--hp_reinforcement_factor', metavar='reinforcement_factor', type=int, nargs='?', default=1.7)

parser.add_argument('-lp', '--lowpass', choices=['average', 'median'], nargs='?', const='average')
parser.add_argument('-lp_s', '--lp_sigma', metavar='sigma', type=int, nargs='?', default=3)

parser.add_argument('-fr', '--filter_roberts', action='store_const', default=0, const=1)

parser.add_argument('-fp', '--filter_prewitt', action='store_const', default=0, const=1)

parser.add_argument('-fs', '--filter_sobel',  type=int, nargs='?', const=3)

parser.add_argument('-fc', '--filter_canny', dest='canny_args', type=int, nargs='+')

parser.add_argument('-fl', '--filter_laplace', type=int, nargs='?', const=3)

parser.add_argument('-fz', '--filter_zerocross', type=int, nargs='?', const=3)

parser.add_argument('-v', '--verbose', action='count', default=0)
parser.add_argument('--test', action='store_true')

args = parser.parse_args()
if args.verbose >= 1:
	print(args)
# print(parser.parse_args('--input="./rose.png" --threshold 127 255 0 -g'.split()))

# ./image_tinker.py --input="./rose.png" --threshold 127 255 0 -g -hp basic -hp_s=3

def showImage(image, image_name='image'):
	cv2.imshow(image_name, image)

def threshold(image):
	ret = None
	imthresh = None
	
	# use available arguments
	if len(args.threshold_args) > 1:
		# all arguments were passed
		ret, imthresh = cv2.threshold(image, args.threshold_args[0], args.threshold_args[1], args.threshold_args[2])
	else:
		# only thresh passed, infer max and algo type
		ret, imthresh = cv2.threshold(image, args.threshold_args[0], 255, cv2.THRESH_BINARY)

	return imthresh

def highpass(image, hp_type, sigma, reinforcement_factor):
	if hp_type == 'basic':
		return cv2.GaussianBlur(image, (0, 0), sigma)

	if hp_type == 'high_reinforcement':
		return (image * (reinforcement_factor - 1)).astype('uint8')  + cv2.GaussianBlur(image, (0, 0), sigma)

	return None

def lowpass(image, lp_type, sigma):
	if lp_type == 'average':
		return cv2.GaussianBlur(image, (0, 0), sigma)

	if lp_type == 'median':
		return cv2.medianBlur(image, sigma)

	return None

def roberts(image):
	roberts_cross_v = np.array([[ 0, 0, 0 ], [ 0, 1, 0 ], [ 0, 0,-1 ]])

	roberts_cross_h = np.array([[ 0, 0, 0 ], [ 0, 0, 1 ], [ 0,-1, 0 ]])

	vertical = ndimage.convolve(image.astype('double'), roberts_cross_v)
	horizontal = ndimage.convolve(image.astype('double'), roberts_cross_h)

	return cv2.sqrt(cv2.pow(horizontal, 2) + cv2.pow(vertical, 2)).astype('uint8')

def prewitt(image):
	kernelx = np.array([[1,1,1], [0,0,0], [-1,-1,-1]])
	kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

	vertical = ndimage.convolve(image.astype('double'), kernelx)
	horizontal = ndimage.convolve(image.astype('double'), kernely)

	return cv2.sqrt(cv2.pow(horizontal, 2) + cv2.pow(vertical, 2)).astype('uint8')

def sobel(image, kernel_size):
	return cv2.Sobel(image, cv2.CV_8U, 1, 1, kernel_size)

def canny(image, lower_thresh, upper_thresh):
	return cv2.Canny(image, lower_thresh, upper_thresh)

def laplace(image, kernel_size):
	return cv2.Laplacian(image, cv2.CV_8U, kernel_size)

def zerocross(image, kernel_size):
	laplace_im = laplace(image, kernel_size)

	return (laplace_im/laplace_im.max())

if __name__=='__main__':
	args = parser.parse_args()
	input_image = result_image = cv2.imread(args.input)

	showImage(input_image, image_name='original')

	# if grayscale convertion is needed
	if args.grayscale != 0:
		result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

	# if threshold is to be used
	if args.threshold_args is not None:
		result_image = threshold(result_image)

	if args.highpass is not None:
		result_image = highpass(result_image, args.highpass, args.hp_sigma, args.hp_reinforcement_factor)

	if args.lowpass is not None:
		result_image = lowpass(result_image, args.lowpass, args.lp_sigma)

	if args.filter_roberts != 0:
		result_image = roberts(result_image)

	if args.filter_prewitt != 0:
		result_image = prewitt(result_image)

	if args.filter_sobel is not None:
		result_image = sobel(result_image, args.filter_sobel)

	if args.canny_args is not None:
		result_image = canny(result_image, args.canny_args[0], args.canny_args[1])

	if args.filter_laplace is not None:
		result_image = laplace(result_image, args.filter_laplace)

	if args.filter_zerocross is not None:
		result_image = zerocross(result_image, args.filter_zerocross)

	showImage(result_image, image_name='result')

	cv2.waitKey(0)
	cv2.destroyAllWindows()
