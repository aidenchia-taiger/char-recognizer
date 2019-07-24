import cv2
import numpy as np
import os
import argparse
from scipy.ndimage.filters import maximum_filter
from PIL import Image
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet
from scipy.signal import convolve2d
import math
import pdb

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--i', help="path to input image", default="test2.png")
	parser.add_argument('--d', help="display denoised image", action='store_true')
	parser.add_argument('--o', help="save binarized image", default=None)
	args = parser.parse_args()

	img = cv2.imread(args.i, cv2.IMREAD_GRAYSCALE)
	denoiser = Denoiser()
	auto_denoised = denoiser.denoise(img, userconfig=False)
	user_denoised = denoiser.denoise(img, userconfig=True)

	if args.d:
		display(img, auto_denoised, True)
		display(img, user_denoised, True)

	if args.o:
		cv2.imwrite('out/{}_auto.png'.format(args.o), auto_denoised)
		cv2.imwrite('out/{}_user.png'.format(args.o), user_denoised)

	print(np.array_equal(auto_denoised, user_denoised))

def display(img, denoised, hist=False):
	plt.rcParams["figure.figsize"] = [12, 9]
	
	if hist:
		fig, axes = plt.subplots(2,2, tight_layout=True)
		ax0, ax1, ax2, ax3 = axes.flatten()
		ax0.imshow(img, 'gray')
		ax1.imshow(denoised, 'gray')
		ax2.hist(img.ravel(), 256, [0,256])
		ax3.hist(denoised.ravel(), 256, [0,256])

	else:
		fig, axes = plt.subplots(1,2, tight_layout=True)
		ax0, ax1 = axes.flatten()
		ax0.imshow(img, 'gray')
		ax1.imshow(denoised, 'gray')

	plt.show()

class Denoiser:
	def denoise_by_user_config(self, img):
		self.config = self.read_config(open('userconfig.txt'))
		if self.config['DESHADOW']:
			img = self.deshadow(img, self.config['MAX KERNEL'], self.config['MEDIAN KERNEL'])

		if self.config['CONTRAST']:
			img = self.increaseContrast(img)

		if self.config['CROPBACKGROUND']:
			img = self.cropBackground(img, self.config['MIN AREA PERCENTAGE'])

		if self.config['SHARPEN']:
			img = self.sharpen(img)

		if self.config['WAVELET']:
			img = self.wavelet(img, self.config['WAVELET SIGMA'])

		if self.config['TOPHAT']:
			img = self.tophat(img, self.config['TOPHAT KERNEL SIZE'])

		if self.config['GRADIENT']:
			img = self.gradient(img, self.config['GRADIENT KERNEL SIZE'])

		if self.config['CLOSING']:
			img = self.closing(img, self.config['CLOSING KERNEL SIZE'])

		if self.config['BLUR']:
			img = self.blur(img, self.config['BLURRING METHOD'])

		if self.config['OPENING']:
			img = self.opening(img, self.config['OPENING KERNEL SIZE'])

		if self.config['EROSION']:
			img = self.erosion(img, self.config['EROSION KERNEL SIZE'], \
									self.config['EROSION ITERATIONS'])

		if self.config['DILATION']:
			img = self.dilation(img, self.config['DILATION KERNEL SIZE'], \
									 self.config['DILATION ITERATIONS'])

		if self.config['BINARIZE']:
			img = self.binarize(img, self.config['BINARIZATION METHOD'])

		if self.config['CROPTEXT']:
			img = self.cropText(img)

		if self.config['BINARIZE']:
			img = self.binarize(img, self.config['BINARIZATION METHOD'])

		return img

	def read_config(self, config):
		dic = {}
		for line in config:
			line = line.strip().split()

			if line[0] == 'BINARIZE':
				dic[line[0]] = True if line[1] == 'T' else False
				if line[2] == str(0): 
					dic['BINARIZATION METHOD'] = 'global'

				elif line[2] == str(1):
					dic['BINARIZATION METHOD'] = 'adaptive'

				elif line[2] == str(2):
					dic['BINARIZATION METHOD'] = 'otsu'

				elif line[2] == str(3):
					dic['BINARIZATION METHOD'] = 'inv'
				self.printIfTrue(line[0], dic,'BINARIZATION METHOD')

			elif line[0] == 'DESHADOW':
				dic[line[0]] = True if line[1] == 'T' else False
				dic['MAX KERNEL'] = int(line[2])
				dic['MEDIAN KERNEL'] = int(line[3])
				self.printIfTrue(line[0], dic, 'MAX KERNEL', 'MEDIAN KERNEL')
			
			elif line[0] == 'BLUR':
				dic[line[0]] = True if line[1] == 'T' else False
				dic['BLURRING KERNEL SIZE'] = int(line[2])

				if line[3] == str(0):
					dic['BLURRING METHOD'] = 'average'
				elif line[3] == str(1):
					dic['BLURRING METHOD'] = 'median'
				elif line[3] == str(2):
					dic['BLURRING METHOD'] = 'gaussian'
				elif line[3] == str(3):
					dic['BLURRING METHOD'] = 'bilateral'
				elif line[3] == str(4):
					dic['BLURRING METHOD'] = 'max'
				self.printIfTrue(line[0], dic, 'BLURRING METHOD')
			
			elif line[0] == 'DISPLAY':
				dic[line[0]] = True if line[1] == 'T' else False
				dic['HIST'] = True if line[2] == 'T' else False
				self.printIfTrue(line[0], dic, 'HIST')

			elif line[0] == 'GRADIENT':
				dic[line[0]] = True if line[1] == 'T' else False
				dic['GRADIENT KERNEL SIZE'] = int(line[2])
				self.printIfTrue(line[0], dic, 'GRADIENT KERNEL SIZE')

			elif line[0] == 'CLOSING':
				dic[line[0]] = True if line[1] == 'T' else False
				dic['CLOSING KERNEL SIZE'] = int(line[2])
				self.printIfTrue(line[0], dic, 'CLOSING KERNEL SIZE')

			elif line[0] == 'OPENING':
				dic[line[0]] = True if line[1] == 'T' else False
				dic['OPENING KERNEL SIZE'] = int(line[2])
				self.printIfTrue(line[0], dic, 'OPENING KERNEL SIZE')

			elif line[0] =='EROSION':
				dic[line[0]] = True if line[1] == 'T' else False
				dic['EROSION KERNEL SIZE'] = int(line[2])
				dic['EROSION ITERATIONS'] = int(line[3])
				self.printIfTrue(line[0], dic, 'EROSION KERNEL SIZE', 'EROSION ITERATIONS')

			elif line[0] =='DILATION':
				dic[line[0]] = True if line[1] == 'T' else False
				dic['DILATION KERNEL SIZE'] = int(line[2])
				dic['DILATION ITERATIONS'] = int(line[3])
				self.printIfTrue(line[0], dic, 'DILATION KERNEL SIZE', 'DILATION ITERATIONS')

			elif line[0] == 'CROPBACKGROUND':
				dic[line[0]] = True if line[1] == 'T' else False
				dic['MIN AREA PERCENTAGE'] = int(line[2])
				self.printIfTrue(line[0], dic, 'MIN AREA PERCENTAGE')

			elif line[0] == 'TOPHAT':
				dic[line[0]] = True if line[1] == 'T' else False
				dic['TOPHAT KERNEL SIZE'] = int(line[2])
				self.printIfTrue(line[0], dic, 'TOPHAT KERNEL SIZE')

			elif line[0] == 'SHARPEN':
				dic[line[0]] = True if line[1] == 'T' else False
				dic['SHARPEN IMG'] = float(line[2])
				dic['BLUR IMG'] = float(line[3])
				self.printIfTrue(line[0], dic, 'SHARPEN IMG', 'BLUR IMG')

			elif line[0] == 'CONTRAST':
				dic[line[0]] = True if line[1] == 'T' else False
				dic['CONTRAST METHOD'] = 'global' if line[2] == str(0) else 'adaptive'
				self.printIfTrue(line[0], dic, 'CONTRAST METHOD')

			else:
				dic[line[0]] = True if line[1] == 'T' else False
				self.printIfTrue(line[0], dic)
			
		return dic

	def printIfTrue(self, method, dic, *params):
		if dic[method]:
			print('{}: {}'.format(method, dic[method]))
			for param in params:
				print('{}: {}'.format(param, dic[param]))

	def is_binary(self, img):
		return np.array_equal(np.unique(img), np.array([0, 255]))

	def percentageBlack(self, img):
		img = self.binarize(img, 'global' ,127)
		numBlack = (img == 0).sum()
		numWhite = (img == 255).sum()
		return numBlack * 100 / numWhite

	def denoise(self, img, userconfig=False):
		if userconfig:
			return self.denoise_by_user_config(img)

		f = open('config.txt', 'w+')

		'TODO: if (shadow): deshadow'
		'TODO: how to distinguish other bg object?'
		'TODO: how to remove black border?'

		if 100 - self.percentageBlack(img) > 55: # there is likely a large white border surrounding area of interest
			f.write('CROPBACKGROUND T 5\n')
			img = self.cropBackground(img)

		if self.percentageBlack(img) > 40: # img likely has high density of pepper noise
			f.write('CLOSING T 3\n')
			img = self.closing(img, 3)

		if self.percentageBlack(img) > 15: # img has moderate density of pepper noise
			f.write('BLUR T 12 3\n') # bilateral blur
			f.write('BINARIZE T 2\n') # Otsu binarization
			img = self.blur(img, 'bilateral', 12)
			img = self.binarize(img, 'otsu')

		f.write('CROPTEXT T\n')
		img = self.cropText(img)

		if not self.is_binary(img):
			f.write('BINARIZE T 2\n')
			img = self.binarize(img, 'otsu', 21)
		
		f.close()
		return img

	###############################################
	def binarize(self, img, method='otsu', gthreshold=127):
		# adaptive and Otsu requires image to be grayscale
		if method == 'global':
			_, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

		elif method == 'adaptive':
			img = cv2.adaptiveThreshold(src=img, dst=img, maxValue=255, \
										adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
								   		thresholdType=cv2.THRESH_BINARY, blockSize=21, C=2)

		elif method == 'otsu':
			_, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

		return img

	###############################################
	def cropBackground(self, img, minArea=3):
		imgArea = img.shape[0] * img.shape[1]
		th, threshed = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)

		## (2) Morph-op to remove noise
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
		morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

		## (3) Find the max-area contour
		cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
		cnt = sorted(cnts, key=cv2.contourArea, reverse=True)
		rois = []
		print(len(rois))
		for idx in range(len(cnt)):
			x,y,w,h = cv2.boundingRect(cnt[idx])
			if w*h / imgArea < minArea / 100:
				continue
			roi = img[y:y+h, x:x+w]
			rois.append(roi)
			#cv2.imwrite('out/crop{}.png'.format(idx), roi)

		if len(rois) != 0:
			(minH, minW) = rois[-1].shape
			img = np.vstack([Image.fromarray(roi).resize((minW, minH)) for roi in rois])
			#cv2.imwrite('out/crop100.png', img)
		return img

	###############################################
	def gradient(self, img, kernelSize):
		kernel = np.ones((kernelSize,kernelSize),np.uint8)
		img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
		return img

	###############################################
	def wavelet(self, img, sigma=None):
		return denoise_wavelet(img, sigma)

	###############################################
	def sharpen(self, img, sharpenImg=1.5, blurImg=-0.5):
		blur = self.blur(img, 'gaussian')
		img = cv2.addWeighted(img, sharpenImg, blur, blurImg, 0)
		return img

	###############################################
	def closing(self, img, kernelSize):
		kernel = np.ones((kernelSize,kernelSize),np.uint8)
		img =  cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
		return img

	###############################################
	def erosion(self, img, eKernel,eIterations):
		eKernel = np.ones((eKernel, eKernel), np.uint8)
		return cv2.erode(img, eKernel, eIterations)

	###############################################
	def dilation(self, img, dKernel,dIterations):
		dKernel = np.ones((dKernel, dKernel), np.uint8)
		return cv2.dilate(img, dKernel, dIterations)

	###############################################
	def cropText(self, img):
		'Finds the texts in img and returns an image with the texts against a white background'
		rgb = cv2.pyrDown(img)
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
		grad = cv2.morphologyEx(rgb, cv2.MORPH_GRADIENT, kernel)

		_, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
		connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

		# using RETR_EXTERNAL instead of RETR_CCOMP
		contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		#For opencv 3+ comment the previous line and uncomment the following line
		#_, contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

		mask = np.zeros(bw.shape, dtype=np.uint8)
		white = np.zeros_like(rgb)
		white[white==0] = 255

		# If fail to find any contours, return the original image
		#print('[INFO] No. of contours found: {}'.format(len(contours)))
		if len(contours) == 0:
			return bw

		for idx in range(len(contours)):
		    x, y, w, h = cv2.boundingRect(contours[idx])
		    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
		    aspectRatio = float(w / h)
		    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h) # the 'size' of the contour / area of contour 

		    if r > 0.45 and w > 8 and h > 8:
		        #cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (255, 255, 255), 2)
		        out = rgb[y:y+h, x:x+w]
		        #plt.imshow(out, 'gray')
		        #plt.show()
		        white[y:y+h, x:x+w] = out

		#cv2.imwrite('cropped2.png', white)
		return white

	###############################################
	def tophat(self, img, kernelSize):
		kernel = np.ones((kernelSize, kernelSize), np.uint8)
		return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

	#############################	##################
	def opening(self, img, kernelSize):
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelSize, kernelSize))
		return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
	
	###############################################
	def deshadow(self, img, maxKernel=10, medianKernel=17):

	    bg_img = maximum_filter(img, size =(maxKernel,maxKernel)) # Max Filter

	    bg_img = cv2.medianBlur(bg_img, medianKernel) # Median Filter

	    diff_img = 255 - cv2.absdiff(img, bg_img) # Extract foreground

	    norm_img = np.empty(diff_img.shape)
	    norm_img = cv2.normalize(diff_img, dst=norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1) # Normalize pixels
	    
	    return norm_img

	###############################################
	def blur(self, img, method, kernelSize=3):
		if method == 'average':
			img = cv2.blur(img, (kernelSize, kernelSize))

		elif method == 'median':
			print(kernelSize)
			img = cv2.medianBlur(img, kernelSize)

		elif method == 'gaussian':
			img = cv2.GaussianBlur(img, (kernelSize, kernelSize), 0)

		elif method == 'bilateral':
			img = cv2.bilateralFilter(img, 9, 150, 150)

		elif method == 'max':
			img = cv2.maximum_filter(img, (kernelSize, kernelSize))

		return img
	###############################################
	def increaseContrast(self, img, method='global'):
		if method == 'global':
			return cv2.equalizeHist(img)
		elif method == 'adaptive':
			clahe = cv2.createCLAHE()
			img = clahe.apply(img)
			return img

if __name__ == '__main__':
	main()
