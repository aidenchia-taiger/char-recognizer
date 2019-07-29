import argparse
import os
import pdb
import json
import cv2
import keras
import numpy as np
import tensorflow as tf
from keras.models import load_model
from Utils import display, save, output
from Model import ModelFactory
from Deslanter.Deslanter import Deslanter
from Segmenter.Segmenter import Segmenter
from SpellCorrector.SimpleSpellCorrector import SpellCorrector
from TextDetector.TesseractTextDetector import TextDetector
#from TextDetector.EASTTextDetector import TextDetector

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help='select which model to use', default="beta")
	parser.add_argument('--type', help=" input type is either doc, word, or char", default="doc")
	parser.add_argument('--infer', help="path to input file")
	parser.add_argument('--show', help="display texts detected & characters segmented by detector & segmenter", action="store_true")
	parser.add_argument('--gt', help="supply ground truth of word or char image", default="../sample_imgs/gt.json")
	args = parser.parse_args()

	mf = ModelFactory()
	model = mf.load()
	segmenter = Segmenter()
	textDetector = TextDetector()
	deslanter = Deslanter()
	spellCorrector = SpellCorrector(lexicon="SpellCorrector/lexicon.txt", misclassify="SpellCorrector/misclassify.json")

	if args.type == "char":
		# Infer a char image - no need to do segmentation
		charImg = cv2.imread(args.infer, cv2.IMREAD_GRAYSCALE)
		charImg = mf.preprocess(charImg)
		prediction = mf.predictChar(model, charImg)

	elif args.type == "word":
		# Infer a word image - segment the word image then predict char by char 
		wordImg = cv2.imread(args.infer, cv2.IMREAD_GRAYSCALE)
		pred = mf.predictWord(model,segmenter, wordImg, spellCorrector, show=args.show)

		if args.gt:
			mf.getCER(pred, args.gt)

	elif args.type == "doc":
		# Infer a doc image - detect texts, segment char by char, predict each char
		docImg = cv2.imread(args.infer)
		docImg = deslanter.deslant(docImg, args.show)
		textPreds, lineBoxes = mf.predictDoc(model, segmenter, textDetector, docImg, spellCorrector, \
											 showCrop=args.show, showChar=args.show)

		pred = output(textPreds, lineBoxes, 'out.hocr', 'out.txt')
		
		if args.gt:
			# Check that the gt.json file actually exists
			assert os.path.isfile(args.gt)
			# If the ground truth is in another folder, grab just the filename itself instead of the whole path
			key = args.infer.split('/')[-1] if '/' in args.infer else args.infer
			gt = json.load(open(args.gt))[key]
			# We upper case everything so that the char error rate doesn't count case errors
			mf.getCER(pred.upper(), gt)


if __name__ == "__main__":
	main()
