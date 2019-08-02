import argparse
import os
import keras
import numpy as np 
import cv2
from Model import ModelFactory
from Segmenter.Segmenter import Segmenter
from SpellCorrector.SimpleSpellCorrector import SpellCorrector

def testModel():
	'Test the model on a directory of character or word images'
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help='select which model to use', default="beta")
	parser.add_argument('--test', help="path to dir")
	parser.add_argument('--type', help="input type to either doc, word, or char")
	args = parser.parse_args()

	mf = ModelFactory(modelName=args.model)
	segmenter = Segmenter()
	spellCorrector = SpellCorrector(lexicon="SpellCorrector/lexicon.txt", misclassify="SpellCorrector/misclassify.json")
	
	model = mf.load()

	imgFiles = []

	for root, _, files in os.walk(args.test):
		for file in files:
			if file[-4:] != '.png':
					continue
			imgFiles.append(os.path.join(root, file))

	imgFiles = sorted(imgFiles)

	for file in imgFiles:
		img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
		img = mf.preprocess(img)

		if args.type == 'char':
			prediction = mf.predictChar(model, img)

		elif args.type == 'word':
			prediction = mf.predictWord(model, segmenter, img, spellCorrector)




if __name__ == "__main__":
	testModel()