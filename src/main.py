import argparse
import os
import keras
from keras.models import load_model
import editdistance
import numpy as np
import tensorflow as tf
import cv2
import pdb
from Tesseract_TextDetector import TextDetector
from Utils import display, save, outputHOCR
from Model import ModelFactory
from Segmenter import Segmenter

def main():
	'Infer on a single word / character image or entire document'

	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help='select which model to use', default="beta")
	parser.add_argument('--type', help="set your input type to either doc, word, or char", default="word")
	parser.add_argument('--infer', help="path to input file")
	parser.add_argument('--show', help="display texts detected & characters segmented by detector & segmenter", action="store_true")
	parser.add_argument('--gt', help="supply ground truth of word or char image")
	args = parser.parse_args()

	mf = ModelFactory(modelName=args.model)
	model = mf.load()
	segmenter = Segmenter()
	textDetector = TextDetector()

	if args.type == "char":
		# Infer a char image - no need to do segmentation
		charImg = cv2.imread(args.infer, cv2.IMREAD_GRAYSCALE)
		charImg = mf.preprocess(charImg)
		prediction = mf.predictChar(model, charImg)

	elif args.type == "word":
		# Infer a word image - segment the word image then predict char by char 
		wordImg = cv2.imread(args.infer, cv2.IMREAD_GRAYSCALE)
		pred = mf.predictWord(model,segmenter, wordImg)

		if args.gt:
			mf.getCER(pred, args.gt)

	elif args.type == "doc":
		# Infer a doc image - detect texts, classify if handwritten or digital, segment handwritten words, predict char by char
		docImg = cv2.imread(args.infer) # EAST text detector requires 3 channels
		textPreds, lineBoxes = mf.predictDoc(model, segmenter, textDetector, docImg, showCrop=args.show, showChar=args.show)
		outputHOCR(textPreds, lineBoxes, 'out.hocr')


if __name__ == "__main__":
	main()
