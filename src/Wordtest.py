import argparse
import os
import keras
import numpy as np 
import cv2
from Model import ModelFactory
from Segmenter import Segmenter

# TODO: Supply ground truth

def testModel():
	'Test the model on a directory of character images'
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help='select which model to use', default="model")
	parser.add_argument('--test', help="path to dir", default="../../data/sample_data/IAM/")
	args = parser.parse_args()

	mf = ModelFactory(modelName=args.model)
	model = mf.load()

	CER = []
	ground_truth = []
	predicted = []
	labels = ["nominating", "any", "move", "meeting", "been", "texts", "ready", "one"]
	ctr = 0


	imgFiles = []
	gt = []
	for root, _, files in os.walk(args.test):
		for file in files:
			# if file[-4:] != '.jpg':
			# 		continue
			imgFiles.append(os.path.join(root, file))

	imgFiles = sorted(imgFiles)
	print("Hello")

	for file in imgFiles:
		print(file)
		img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
		#print(img.shape)
		#charImg = mf.preprocess(img, invert=True)
		#print("Image processed")
		#prediction = mf.predictChar(model, charImg)
		segmenter = Segmenter()
		pred = mf.predictWord(model, segmenter, img, show=False)
		print(pred)
		print(file)
		gt = file.split('/')[-1]
		print(gt)
		gt = gt.split('.')[0]
		print(gt)
		gt = labels[ctr]
		print("labels", gt)
		ctr += 1
		CER.append(mf.getCER(pred, gt))
		ground_truth.append(gt)
		predicted.append(pred)

	for i in range(len(predicted)):
		print(ground_truth[i], predicted[i], CER[i])

	print(np.average(np.array(CER)))


if __name__ == "__main__":
	testModel()