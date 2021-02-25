from tkinter import *
from tkinter import messagebox, filedialog
from filter import Filter
from PIL import Image, ImageTk
from mbox import MessageBox
import cv2
import numpy as np
from scipy import ndimage
from skimage.util import random_noise
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import imutils

class Window(Frame):
	def __init__(self, master=None):
		Frame.__init__(self, master)
		self.master = master
		#self.master.resizable(width=False, height=False)
		self.pack(fill=BOTH, expand=1)

		self.undo_stack = []
		self.save_path = None
		
		self.img_label = None

		self.setup_menubar()
		
		self.input_image = self.result_image = cv2.imread("lena_std.tif")
		self.undo_stack.append(self.input_image)

		self.show_image()

	def setup_menubar(self):
		menubar = Menu(self.master, background='#ff8000', foreground='black', activebackground='white', activeforeground='black')  
		file = Menu(menubar, tearoff=1, background='#139ddd', foreground='black')
		file.add_command(label="Open", command=self.new_image)
		file.add_command(label="Save", command=self.save_image)
		file.add_command(label="Save as")
		file.add_separator()
		file.add_command(label="Exit", command=self.master.quit)
		menubar.add_cascade(label="File", menu=file)

		edit = Menu(menubar, tearoff=0)
		edit.add_command(label="Undo", command=self.undo)
		menubar.add_cascade(label="Edit", menu=edit)

		self.stack_filters = BooleanVar()
		self.stack_filters.set(True)

		preferences = Menu(menubar, tearoff=0)
		preferences.add_checkbutton(label="Stack Filters", onvalue=1, offvalue=0, variable=self.stack_filters)
		menubar.add_cascade(label='Preferences', menu=preferences)

		filters = Menu(menubar, tearoff=0)
		filters.add_command(label="Grayscale", command=self.grayscale)
		filters.add_command(label="Threshold", command=self.threshold)
		filters.add_command(label="High Pass", command=self.highpass)
		filters.add_command(label="Low Pass", command=self.lowpass)
		filters.add_command(label="Roberts", command=self.roberts)
		filters.add_command(label="Prewitt", command=self.prewitt)
		filters.add_command(label="Sobel", command=self.sobel)
		filters.add_command(label="Canny", command=self.canny)
		filters.add_command(label="Laplace", command=self.laplace)
		filters.add_command(label="Zero Cross", command=self.zerocross)
		filters.add_command(label="Invert", command=self.invert)
		menubar.add_cascade(label="Filters", menu=filters)

		noises = Menu(menubar, tearoff=0)
		noises.add_command(label="Salt & Pepper", command=self.saltnpepper)
		noises.add_command(label="Gauss", command=self.gauss)
		noises.add_command(label="Speckle", command=self.speckle)
		menubar.add_cascade(label="Noises", menu=noises)

		segmentation = Menu(menubar, tearoff=0)
		segmentation.add_command(label="Watershed Counting", command=self.watershed_counting)
		segmentation.add_command(label="Histogram", command=self.histogram)
		menubar.add_cascade(label="Misc", menu=segmentation)

		help = Menu(menubar, tearoff=0)
		help.add_command(label="About", command=about)
		menubar.add_cascade(label="Help", menu=help)

		self.master.config(menu=menubar)

	def show_image(self):
		# get image dimensions
		width, height, _ = self.input_image.shape

		# convert from opencv to TKinter image
		image_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(self.result_image, cv2.COLOR_RGB2BGR)))
		if self.img_label:
			self.img_label.image = None
		self.img_label = Label(self, image=image_tk)
		
		# set current image
		self.img_label.image = image_tk
		self.img_label.place(x=0, y=0)

		# TODO: rescale image to sane resolution
		# TODO: set window as image resolution
		self.master.geometry('{}x{}'.format(height+5, width+5))

	def new_image(self):
		filename = filedialog.askopenfilename(initialdir = "./",
											title = "Select a File",
											filetypes = (("All files", "*.*"),
														("Images", "*.jpg*")))

		self.undo_stack = []
		self.input_image = self.result_image = cv2.imread(filename)
		self.show_image()

	def save_image(self):
		if not self.save_path:
			filename = filedialog.asksaveasfilename(initialdir = "/",
										title = "Select a File",
										filetypes = (("Images", "*.jpg*"),
													("all files", "*.*")))

			self.save_path = filename

		cv2.imwrite(self.save_path, self.result_image)

	def undo(self):
		self.result_image = self.undo_stack.pop()

		self.show_image()

	def grayscale(self):
		current_image = self.result_image if self.stack_filters else self.input_image

		self.undo_stack.append(self.result_image)

		self.result_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)

		self.show_image()

	""" FILTERS """

	def threshold(self):
		current_image = self.result_image if self.stack_filters else self.input_image

		threshold_args = {}
		threshold_args['thresh_value'] = mbox('Threshold cuttof value:', entry=True)
		if threshold_args['thresh_value']:
			threshold_args['defaults'] = mbox('Use defaults?', ('yes', 'y'), ('no', 'n'))
			if threshold_args['defaults'] == 'n':
				threshold_args['max_value'] = mbox('Max value:', entry=True)
				threshold_args['thresh_type'] = mbox('Threshold type:', entry=True)
			# mbox(threshold_args, frame=False)

		self.undo_stack.append(self.result_image)

		if threshold_args['defaults'] == 'n':
			self.result_image = Filter.threshold(current_image, int(threshold_args['thresh_value']), int(threshold_args['max_value']), int(threshold_args['thresh_type']))
		else:
			self.result_image = Filter.threshold(current_image, int(threshold_args['thresh_value']))

		self.show_image()
	
	def highpass(self):
		current_image = self.result_image if self.stack_filters else self.input_image
		self.undo_stack.append(self.result_image)

		highpass_args = {}
		highpass_args['defaults'] = mbox('Use defaults?', ('yes', 'y'), ('no', 'n'))
		if highpass_args['defaults'] == 'n':
			highpass_args['type'] = mbox('High Pass Type', ('Basic', 0), ('Original Factor', 1))
			highpass_args['sigma'] = mbox('Sigma:', entry=True)
			if highpass_args['type'] == 1:
				highpass_args['reinforcement_factor'] = mbox('Reinforcement Factor:', entry=True)
		# mbox(highpass_args, frame=False)

		if highpass_args['defaults'] == 'n' and not highpass_args['sigma']:
			return

		if highpass_args['defaults'] == 'n':
			self.result_image = Filter.highpass(current_image, highpass_args['type'], int(highpass_args['sigma']), float(highpass_args['reinforcement_factor']))
		else:
			self.result_image = Filter.highpass(current_image)

		self.show_image()

	def lowpass(self):
		current_image = self.result_image if self.stack_filters else self.input_image
		self.undo_stack.append(self.result_image)

		lowpass_args = {}
		lowpass_args['defaults'] = mbox('Use defaults?', ('yes', 'y'), ('no', 'n'))
		if lowpass_args['defaults'] == 'n':
			lowpass_args['type'] = mbox('Low Pass Type', ('Average', 0), ('Median', 1))
			lowpass_args['sigma'] = mbox('Sigma:', entry=True)
		# mbox(lowpass_args, frame=False)

		if lowpass_args['defaults'] == 'n' and not lowpass_args['sigma']:
			return

		if lowpass_args['defaults'] == 'n':
			self.result_image = Filter.lowpass(current_image, lowpass_args['type'], int(lowpass_args['sigma']))
		else:
			self.result_image = Filter.lowpass(current_image)

		self.show_image()

	def roberts(self):
		current_image = self.result_image if self.stack_filters else self.input_image

		if len(self.result_image.shape) == 3:
			img_r, img_g, img_b = cv2.split(self.result_image)
			img_r = Filter.roberts(img_r)
			img_g = Filter.roberts(img_g)
			img_b = Filter.roberts(img_b)
			self.result_image = cv2.merge((img_r, img_g, img_b))
		else:
			self.result_image = Filter.roberts(self.result_image)

		self.show_image()

	def prewitt(self):
		current_image = self.result_image if self.stack_filters else self.input_image

		if len(self.result_image.shape) == 3:
			img_r, img_g, img_b = cv2.split(self.result_image)
			img_r = Filter.prewitt(img_r)
			img_g = Filter.prewitt(img_g)
			img_b = Filter.prewitt(img_b)
			self.result_image = cv2.merge((img_r, img_g, img_b))
		else:
			self.result_image = Filter.prewitt(current_image)

		self.show_image()
	
	def sobel(self):
		current_image = self.result_image if self.stack_filters else self.input_image

		sobel_args = {}
		sobel_args['defaults'] = mbox('Use defaults?', ('yes', 'y'), ('no', 'n'))
		if sobel_args['defaults'] == 'n':
			sobel_args['ksize'] = mbox('Kernel Size:', entry=True)
		# mbox(sobel_args, frame=False)

		self.undo_stack.append(self.result_image)

		if sobel_args['defaults'] == 'n':
			self.result_image = Filter.sobel(current_image, float(sobel_args['ksize']))
		else:
			self.result_image = Filter.sobel(current_image)

		self.show_image()

	def canny(self):
		current_image = self.result_image if self.stack_filters else self.input_image

		canny_args = {}
		canny_args['defaults'] = mbox('Use defaults?', ('yes', 'y'), ('no', 'n'))
		if canny_args['defaults'] == 'n':
			canny_args['lower_thesh'] = mbox('Lower Threshold:', entry=True)
			canny_args['upper_thesh'] = mbox('Upper Threshold:', entry=True)
		# mbox(canny_args, frame=False)

		self.undo_stack.append(self.result_image)

		if canny_args['defaults'] == 'n':
			self.result_image = Filter.canny(current_image, float(canny_args['lower_thesh']), float(canny_args['upper_thesh']))
		else:
			self.result_image = Filter.canny(current_image)

		self.show_image()

	def laplace(self):
		current_image = self.result_image if self.stack_filters else self.input_image

		laplace_args = {}
		laplace_args['defaults'] = mbox('Use defaults?', ('yes', 'y'), ('no', 'n'))
		if laplace_args['defaults'] == 'n':
			laplace_args['ksize'] = mbox('Kernel Size:', entry=True)
		# mbox(laplace_args, frame=False)

		self.undo_stack.append(self.result_image)

		if laplace_args['defaults'] == 'n':
			self.result_image = Filter.laplace(current_image, float(laplace_args['ksize']))
		else:
			self.result_image = Filter.laplace(current_image)

		self.show_image()

	def zerocross(self):
		current_image = self.result_image if self.stack_filters else self.input_image
		self.undo_stack.append(self.result_image)

		zerocross_args = {}
		zerocross_args['defaults'] = mbox('Use defaults?', ('yes', 'y'), ('no', 'n'))
		if zerocross_args['defaults'] == 'n':
			zerocross_args['ksize'] = mbox('Kernel Size:', entry=True)
		# mbox(zerocross_args, frame=False)

		if zerocross_args['defaults'] == 'n':
			self.result_image = Filter.zerocross(current_image, int(zerocross_args['ksize']))
		else:
			self.result_image = Filter.zerocross(current_image)

		self.show_image()

	def invert(self):
		current_image = self.result_image if self.stack_filters else self.input_image
		self.undo_stack.append(self.result_image)

		self.result_image = cv2.bitwise_not(current_image)

		self.show_image()

	""" NOISES """

	def saltnpepper(self):
		self.undo_stack.append(self.result_image)
		# Convert img to 0 to 1 float to avoid wrapping that occurs with uint8
		current_image = self.result_image if self.stack_filters else self.input_image

		saltnpepper_args = {}
		saltnpepper_args['defaults'] = mbox('Use defaults?', ('yes', 'y'), ('no', 'n'))
		if saltnpepper_args['defaults'] == 'n':
			saltnpepper_args['amount'] = mbox('Amount:', entry=True)
		
		amount = float(saltnpepper_args['amount']) if saltnpepper_args['defaults'] == 'n' else 0.3
		
		noise_img = random_noise(current_image, mode='s&p', amount=0.3)
		self.result_image = np.array(255*noise_img, dtype = 'uint8')

		self.show_image()

	def gauss(self):
		self.undo_stack.append(self.result_image)
		# Convert img to 0 to 1 float to avoid wrapping that occurs with uint8
		current_image = self.result_image if self.stack_filters else self.input_image
		
		gauss = np.random.normal(0, 1, current_image.size)
		gauss = gauss.reshape(current_image.shape[0], current_image.shape[1], current_image.shape[2]).astype('uint8')
		# Add the Gaussian noise to the image
		self.result_image = cv2.add(current_image, gauss)

		self.show_image()

	def speckle(self):
		# Convert img to 0 to 1 float to avoid wrapping that occurs with uint8
		current_image = self.result_image if self.stack_filters else self.input_image
		self.undo_stack.append(self.result_image)

		gauss = np.random.normal(0, 1, current_image.size)
		gauss = gauss.reshape(current_image.shape[0], current_image.shape[1], current_image.shape[2]).astype('uint8')
		self.result_image = current_image + current_image * gauss

		self.show_image()

	""" SEGMENTATION """

	def watershed_counting(self):
		self.undo_stack.append(self.result_image)

		# compute the exact Euclidean distance from every binary
		# pixel to the nearest zero pixel, then find peaks in this
		# distance map
		D = ndimage.distance_transform_edt(self.result_image)
		localMax = peak_local_max(D, indices=False, min_distance=20,
									labels=self.result_image)
		# perform a connected component analysis on the local peaks,
		# using 8-connectivity, then appy the Watershed algorithm
		markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
		labels = watershed(-D, markers, mask=self.result_image)
		# print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

		original_image = self.input_image
		# loop over the unique labels returned by the Watershed
		# algorithm
		for label in np.unique(labels):
			# if the label is zero, we are examining the 'background'
			# so simply ignore it
			if label == 0:
				continue
			# otherwise, allocate memory for the label region and draw
			# it on the mask
			mask = np.zeros(self.result_image.shape, dtype="uint8")
			mask[labels == label] = 255
			# detect contours in the mask and grab the largest one
			cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
				cv2.CHAIN_APPROX_SIMPLE)
			cnts = imutils.grab_contours(cnts)
			c = max(cnts, key=cv2.contourArea)
			# draw a circle enclosing the object
			((x, y), r) = cv2.minEnclosingCircle(c)
			cv2.circle(original_image, (int(x), int(y)), int(r), (0, 255, 0), 2)
			cv2.putText(original_image, "#{}".format(label), (int(x) - 10, int(y)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
		
		self.result_image = original_image

		self.show_image()

	def histogram(self):
		current_image = self.result_image if self.stack_filters else self.input_image
		self.undo_stack.append(self.result_image)

		bgr_planes = cv2.split(self.result_image)
		
		histSize = self.result_image.shape[1]
		
		histRange = (0, self.result_image.shape[0]) # the upper boundary is exclusive
		
		accumulate = False

		b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
		g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
		r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)

		hist_h = self.result_image.shape[0]
		hist_w = self.result_image.shape[1]
		bin_w = int(round( hist_w/histSize ))
		histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

		for i in range(1, histSize):
			cv2.line(histImage, ( bin_w*(i-1), hist_h - int(b_hist[i-1]) ),
					( bin_w*(i), hist_h - int(b_hist[i]) ),
					( 255, 0, 0), thickness=2)
			cv2.line(histImage, ( bin_w*(i-1), hist_h - int(g_hist[i-1]) ),
					( bin_w*(i), hist_h - int(g_hist[i]) ),
					( 0, 255, 0), thickness=2)
			cv2.line(histImage, ( bin_w*(i-1), hist_h - int(r_hist[i-1]) ),
					( bin_w*(i), hist_h - int(r_hist[i]) ),
					( 0, 0, 255), thickness=2)

		self.result_image = histImage

		self.show_image()

	def temp(self):
		pass

def about():
	messagebox.showinfo('Image Tinkerer', 'Tinker with images.\nMade with Sz by Jedson Gabriel')

def mbox(msg, b1='OK', b2='Cancel', frame=True, t=False, entry=False):
    """Create an instance of MessageBox, and get data back from the user.
    msg = string to be displayed
    b1 = text for left button, or a tuple (<text for button>, <to return on press>)
    b2 = text for right button, or a tuple (<text for button>, <to return on press>)
    frame = include a standard outerframe: True or False
    t = time in seconds (int or float) until the msgbox automatically closes
    entry = include an entry widget that will have its contents returned: True or False
    """
    msgbox = MessageBox(msg, b1, b2, frame, t, entry)
    msgbox.root.mainloop()
    # the function pauses here until the mainloop is quit
    msgbox.root.destroy()
    return msgbox.returning

root = Tk()
app = Window(root)
root.wm_title("Image Tinkerer")
root.mainloop()

"""
1. Grayscale
2. low pass
3. threshold (0, 255, 8)
4. watershed
"""