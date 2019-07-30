# Reference: http://uksim.info/uksim2017/CD/data/2735a050.pdf
import os
import cv2
import random
import numpy as np
import pandas as pd
from collections import Counter
from skimage.feature import peak_local_max
from sklearn.svm import SVC
from Utils import display, printInfo

class Distinguisher:
	def __init__(self):
		pass

	def process(self, img):
		self.orig = img
		self.getStrokeInfo(img)

	def pixelVariance(self, img):
		return np.var(img, axis=None)

	def pixelMean(self, img):
		return np.mean(img, axis=None)

	def pixelStdev(self, img):
		return np.std(img, axis=None)

	def pixelOtsu(self, img):
		return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]

	def pixelPeaks(self, img):
		return len(peak_local_max(img, min_distance=1))

	def getStrokeInfo(self, img):
		img = cv2.blur(img, (3,3))
		img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[-1]
		if cv2.__version__[0] == '3':
			contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
		elif cv2.__version__[0] == '2' or cv2.__version__[0] == '4':
			contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

		# Sort by x coordinate
		contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

		self.nbCharacters = 0
		self.strokeWidths = []
		self.strokeHeights = []
		self.charDensities = []
		self.nbStraightLines = 0

		for ctr in contours:
			self.nbCharacters += 1
			x, y, w, h = cv2.boundingRect(ctr)

			self.strokeWidths.append(w)
			self.strokeHeights.append(y)
			subimg = self.orig[y:y+h, x:x+w]
			
			# Uncomment below line to draw the bboxes
			#cv2.rectangle(self.orig, (x,y), (x+w, y+h), (0,255,0), 1)
			
			# Accumulate the no. of straight lines per character to get no. of straight lines for the entire word
			self.nbStraightLines += self.getNbStraightLines(subimg)

			#display(subimg)

	def getStraightness(self, img):
		return self.nbStraightLines * 100 / self.nbCharacters


	def getNbStraightLines(self, img):
		"""
		Machine printed text should have more straight lines in each character compared to handwritten text.
		This algorithm uses Hough Lines Transform and returns the number of horizontal and vertical lines detected.
		"""

		img = cv2.blur(img, (3,3))
		edge = cv2.Canny(img, 50, 150, apertureSize=3)
		lines = cv2.HoughLines(edge, rho=1, theta=np.pi / 180, threshold=1)

		if lines is None:
			return 0

		nbStraightLines = 0
		for rho, theta in lines[0]:
			# Check to see if the line is horizontal or vertical
			#print(theta)
			if abs(theta) < 1e-3  or abs(theta - np.pi/2) < 1e-3:
				#print('Straight line detected')
				nbStraightLines += 1

			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a * rho
			y0 = b * rho
			x1 = int(x0 + 1000 * (-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))

			# Uncomment to draw the line
			#cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 2)

		return nbStraightLines

	def getAlignment(self, img):
		nbOccurrences = dict(Counter(self.strokeHeights))
		nbRepeating = 0
		for k,v in nbOccurrences.items():
			if v > 1:
				nbRepeating += v

		return nbRepeating * 100 / self.nbCharacters

	def getUniformity(self, img):
		"""
		Machine printed text should have very similar stroke widths compared to handwritten text
		"""
		nbOccurrences = dict(Counter(self.strokeWidths))
		nbRepeating = 0
		for k,v in nbOccurrences.items():
			if v > 1:
				nbRepeating += v

		return nbRepeating * 100 / self.nbCharacters

	def extractFeatures(self, img):
		features = [self.getAlignment(img), 
					self.getUniformity(img),
					self.pixelOtsu(img),
					self.pixelStdev(img),
					self.pixelMean(img),
					self.pixelPeaks(img),
					self.getStraightness(img)]

		return features


def trainSVM(X, y):
	clf = SVC()
	clf.fit(X, y)


def loadData(dist, train_path):
	X = []
	y = []
	for root, _, files in os.walk(train_path):
		for file in files:
			if 'handwritten' in root:
				class_label = 0
			else:
				class_label = 1

			fullpath = os.path.join(root, file)
			img = cv2.imread(fullpath, cv2.IMREAD_GRAYSCALE)
			dist.process(img)
			featureVector = dist.extractFeatures(img)
			featureVector.append(class_label)
			X.append(featureVector)

	print(X)


def main(dist, imgpath):
	print('[INFO] Image processed: {}'.format(imgpath))
	img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
	#display(img)

	printInfo(img)
	dist.process(img)
	print('No. Repeating Stroke Heights: ', dist.getAlignment(img))
	print('No. Repeating Stroke Widths: ', dist.getUniformity(img))
	print('Otsu Threshold: ', dist.pixelOtsu(img))
	print('Stdev of Pixel Values: ', dist.pixelStdev(img))
	print('Mean of Pixel Values: ', dist.pixelMean(img))
	print('No. of Local Peaks: ',dist.pixelPeaks(img))
	print('Percentage of Straight Lines Per Character: ', dist.getStraightness(img))
	print('\n')
	#display(dist.orig)

if __name__ == "__main__":
	dist = Distinguisher()
	#for img in ['../sample_imgs/handwritten_date.png', '../sample_imgs/digital_title.png']:
	#	main(dist, img)

	loadData(dist, train_path='../svm_imgs')