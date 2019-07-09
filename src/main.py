import argparse
from Model import ModelFactory
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
import editdistance
import numpy as np
import cv2
import pdb
from Utils import display, percentageBlack
from keras.models import load_model
from Segmenter import Segmenter

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', help='train the NN', action='store_true')
	parser.add_argument('--test', help='test the NN with a batch of images')
	parser.add_argument('--infer', help="infer a directory of images or single image")
	parser.add_argument('--model', help='select the model to use')
	args = parser.parse_args()

	if args.model == None:
		args.model = 'model'
	else:
		args.model = os.path.join('../models', args.model) + '.h5'

	mf = ModelFactory(modelName=args.model, batchSize=32, numClasses=53, imgSize=(32,32,1), dropoutRatio=0.6,
				 numFilters=[8,16,32,64], kernelVals=[4,4,4,4], poolVals=[2,2,2,1], strideVals=[1,1,1,1],
				 learningRate=0.01, numEpochs=50)
	
	if args.train:
		model = mf.build()
		mf.train(model)
		mf.save(model)

	elif args.infer:
		model = mf.build()
		model.load_weights(args.model)
		print('[INFO] Load model from {}'.format(args.model))

		segmenter = Segmenter()

		pred = ""
		if os.path.isdir(args.infer):
			gt = args.infer.split('/')[-2]
			imgFiles = [os.path.join(args.infer, x) for x in os.listdir(args.infer) if x[-4:]=='.png']
			imgFiles = sorted(imgFiles)
			for img in imgFiles:
				img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
				img = mf.preprocess(img)
				prediction = predict(mf.mapping, model, img)
				pred += prediction

			getCER(pred, gt)
		
		elif os.path.isfile(args.infer):
			imgs = segmenter.segment(args.infer)
			for img in imgs:
				img = mf.preprocess(img)
				prediction = predict(mf.mapping, model, img)
		
	elif args.test:
		model = mf.build()
		model.load_weights(args.model)
		print('[INFO] Load model from {}'.format(args.model))

		imgFiles = []
		gt = []
		for root, _, files in os.walk(args.test):
			for file in files:
				if file[-4:] != '.png':
					continue
				imgFiles.append(os.path.join(root, file))
		imgFiles = sorted(imgFiles)
		
		for img in imgFiles:
			img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
			img = mf.preprocess(img)
			prediction = predict(mf.mapping, model, img)


def predict(mapping, model, img, batch_size=1, verbose=1):
	prediction = mapping[np.argmax(model.predict(img, batch_size=1, verbose=1))]
	print('[INFO] Predicted: {}'.format(prediction))
	return prediction

def getCER(pred, gt):
	cer = editdistance.eval(pred, gt) * 100/ len(gt)
	print("[INFO] Ground Truth: {} | Predicted: {}".format(pred, gt))
	print("[INFO] Character Error Rate: {:.1f}%".format(cer))

	return getCER

if __name__ == "__main__":
	main()
