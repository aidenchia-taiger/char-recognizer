from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
import pdb
from Utils import display, save

class TextDetector:
	def __init__(self, newW=960, newH=960, minConfidence=0.001, modelpath='../models/frozen_east_text_detection.pb'):
		self.newW = newW
		self.newH = newH
		self.minConfidence = minConfidence
		self.net = cv2.dnn.readNet(modelpath)

	def detect(self, img):
		self.orig = img.copy()
		resized = self.resize(img)
		bboxes = self.getBBoxes(resized)
		cropped = self.cropBBoxes(bboxes)

		return cropped

	def resize(self, img):
		(self.H, self.W) = img.shape[:2]
		self.rW = self.W / float(self.newW)
		self.rH = self.H / float(self.newH)
		resized = cv2.resize(img, (self.newW, self.newH))
		(self.H, self.W) = resized.shape[:2]
		
		return resized

	def getBBoxes(self, img):
		layerNames = [ "feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
		blob = cv2.dnn.blobFromImage(img, 1.0, (self.W, self.H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
		self.net.setInput(blob)
		(scores, geometry) = self.net.forward(layerNames)
		(numRows, numCols) = scores.shape[2:4]
		rects = []
		confidences = []

		for y in range(0, numRows):
			scoresData = scores[0, 0, y]
			xData0 = geometry[0, 0, y]
			xData1 = geometry[0, 1, y]
			xData2 = geometry[0, 2, y]
			xData3 = geometry[0, 3, y]
			anglesData = geometry[0, 4, y]

			for x in range(0, numCols):
				if scoresData[x] < self.minConfidence:
					continue

				# compute the offset factor as our resulting feature maps will be 4x smaller than the input image
				(offsetX, offsetY) = (x * 4.0, y * 4.0)

				# extract the rotation angle for the prediction and then compute the sin and cosine
				angle = anglesData[x]
				cos = np.cos(angle)
				sin = np.sin(angle)

				# use the geometry volume to derive the width and height of the bounding box
				h = xData0[x] + xData2[x]
				w = xData1[x] + xData3[x]

				# compute both the starting and ending (x, y)-coordinates for the text prediction bounding box
				endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
				endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
				startX = int(endX - w)
				startY = int(endY - h)


				rects.append((startX, startY, endX, endY))
				confidences.append(scoresData[x])


		bboxes = non_max_suppression(np.array(rects), probs=confidences)
		return bboxes

	def cropBBoxes(self, bboxes):
		cropped = []
		for (startX, startY, endX, endY) in bboxes:
			# scale the bounding box coordinates based on the respective ratios
			startX = int(startX * self.rW)
			startY = int(startY * self.rH)
			endX = int(endX * self.rW)
			endY = int(endY * self.rH)

			cv2.rectangle(self.orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
			cropped.append(self.orig[startY:endY,startX:endX])

		# show the output image
		cv2.imshow("Text Detection", self.orig)
		cv2.waitKey(0)
		return cropped

if __name__ == "__main__":
	td = TextDetector()
	img = cv2.imread('../sample_imgs/otp.png')
	td.detect(img)
