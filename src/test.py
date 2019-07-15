import argparse
import os
import keras
import numpy as np 
import cv2
from Model import ModelFactory

# TODO: Supply ground truth

def testModel():
	'Test the model on a directory of character images'
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help='select which model to use', default="beta")
	parser.add_argument('--test', help="path to dir")
	args = parser.parse_args()

	mf = ModelFactory(modelName=args.model)
	model = mf.load()

	imgFiles = []
	gt = []
	for root, _, files in os.walk(args.test):
		for file in files:
			if file[-4:] != '.png':
					continue
			imgFiles.append(os.path.join(root, file))

	imgFiles = sorted(imgFiles)

	for file in imgFiles:
		img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
		charImg = mf.preprocess(img)
		prediction = mf.predictChar(model, charImg)


if __name__ == "__main__":
	testModel()