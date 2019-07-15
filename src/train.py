import cv2
import numpy as np
import keras
import os
from Model import ModelFactory
import argparse

# TODO: Implement restore training from a checkpoint

def trainModel():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help='name the model')
	parser.add_argument('--dropout', help="dropout ratio", default=0.0, type=float)
	parser.add_argument('--batchSize', help="batchSize", default=32, type=int)
	parser.add_argument('--epochs', help="no. of epochs", default=50, type=int)
	parser.add_argument('--lr', help="initial learning rate", default=0.01, type=float)
	args = parser.parse_args()

	mf = ModelFactory(modelName=args.model, batchSize=args.batchSize, numClasses=53, imgSize=(28,28,1), dropoutRatio=args.dropout,
			 numFilters=[8,16,32,64,128,256], kernelVals=[3,3,3,5,5,5], poolVals=[2,2,2,1,1,1], strideVals=[1,1,1,1,1,1],
			 learningRate=args.lr, numEpochs=args.epochs)

	model = mf.build()
	mf.train(model)
	mf.save(model)


if __name__== "__main__":
	trainModel()