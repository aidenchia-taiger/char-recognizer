# Reference: http://uksim.info/uksim2017/CD/data/2735a050.pdf
import os
import cv2
import numpy as np
import pandas as pd
import pickle
from pdb import set_trace
from collections import Counter
from skimage.feature import peak_local_max
from sklearn import model_selection
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from Utils import display, printInfo

class Distinguisher:
	def __init__(self):
		pass

	def process(self, img):
		self.orig = img
		self.getStrokeInfo(img)
		self.idx_to_class = {0: 'handwritten', 1: 'digital'}

	def pixelDensity(self, img):
		img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[-1]
		numWhite = np.count_nonzero(img)
		return numWhite / img.size

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
					self.pixelVariance(img),
					self.pixelDensity(img),
					self.pixelMean(img),
					self.pixelPeaks(img),
					self.getStraightness(img)]

		return features


	def trainSVM(self, X, y, modelpath='distinguisher'):
		svm = SVC(kernel='poly', gamma='auto', probability=True)
		print('[INFO] Training SVM model..')
		self.svm = svm.fit(X, y)
		print('Training Accuracy: {}'.format(self.svm.score(X,y)))
		# Save the model
		pickle.dump(self.svm, open(modelpath, 'wb'))

	def distinguish(self, img, modelpath='distinguisher'):
		if modelpath:
			self.svm = pickle.load(open(modelpath, 'rb'))
			#print('[INFO] Loaded distinguisher model: {}'.format(modelpath))

		self.process(img)
		# Rule-based predictions: Please run Feature Visualisation.ipynb under 'src' directory.
		if self.getAlignment(img) <= 30 \
		or self.getStraightness(img) <= 5 \
		or self.getUniformity(img) <= 15 \
		or self.pixelDensity(img) <= 0.55:
			y = 0

		else:
			x = np.array(self.extractFeatures(img)).reshape(1, -1)
			y = self.svm.predict(x)[0]
		
		print('[INFO] Word distinguished as: {}'.format(self.idx_to_class[y]))
		return self.idx_to_class[y]

def testSVM(dist, path, modelpath='distinguisher'):
	[X, y], paths = loadData(dist, path, getPaths=True)

	if modelpath:
		print('[INFO] Loading distinguisher model: {}'.format(modelpath))
		dist.svm = pickle.load(open(modelpath, 'rb'))

	y_pred = dist.svm.predict(X)
	for i in range(len(y_pred)):
		if y[i] != y_pred[i]:
			print('[INFO] Wrongly Classfied: {}'.format(paths[i]))

	print('[INFO] Accuracy: {}'.format(dist.svm.score(X, y)))
	print('[INFO] Precision Score: {}'.format(precision_score(y, y_pred)))
	print('[INFO] Recall Score: {}'.format(recall_score(y, y_pred)))
	


def loadData(dist, path, getPaths=False):
	X = []
	y = []
	paths = []
	for root, _, files in os.walk(path):
		for file in files:
			# Discard any non-images
			if '.png' not in file and '.jpg' not in file:
				continue

			# Set the ground truth label according to parent dir
			if 'handwritten' in root:
				class_label = 0
			else:
				class_label = 1

			# X is a matrix where each row is the feature vector of a single image, and y is the corresponding gt label
			fullpath = os.path.join(root, file)

			# This is useful for error analysis when we want to see which images have been classified wrongly
			if getPaths:
				paths.append(fullpath)

			img = cv2.imread(fullpath, cv2.IMREAD_GRAYSCALE)
			dist.process(img)
			featureVector = dist.extractFeatures(img)
			X.append(featureVector)
			y.append(class_label)

	if not getPaths:
		X, y = shuffle(X, y, random_state=0)
	
	X = np.array(X)
	#X = normalize(X, axis=1)
	y = np.array(y)
	print('[INFO] X.shape: {}'.format(X.shape))
	print(X)
	print('[INFO] y.shape: {}'.format(y.shape))
	print(y)

	if getPaths:
		return [X, y], paths

	return [X, y]


def runExample(dist, imgpath):
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
	print('Pixel Density: ', dist.pixelDensity(img))
	print('Percentage of Straight Lines Per Character: ', dist.getStraightness(img))
	print('\n')
	#display(dist.orig)

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help='select which model to use', default="distinguisher")
	parser.add_argument('--train', help='train the model', action='store_true')
	parser.add_argument('--view', help='view what each individual function returns', default=None)
	parser.add_argument('--test', help='test the model', action='store_true')
	parser.add_argument('--infer', help='infer an image on the model', default=None)
	args = parser.parse_args()

	dist = Distinguisher()
	modelpath = 'distinguisher'

	# View what each individual function returns
	if args.view:
		for img in ['../dist_imgs/test/handwritten/Charis.png', '../dist_imgs/test/digital/num1.png']:
			runExample(dist, img)

	# Train the model
	if args.train:
		X, y = loadData(dist, path='../dist_imgs/train')
		dist.trainSVM(X, y, modelpath=args.model)

	# Test the model
	if args.test:
		testSVM(dist, path='../dist_imgs/test', modelpath=args.model)

	# Do inference
	if args.infer:
		imgpath = args.infer
		img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
		print('[INFO] Reading image: {}'.format(imgpath))
		dist.distinguish(img, modelpath=args.model)