import argparse
from Model import ModelFactory
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import editdistance
import numpy as np
import tensorflow as tf
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
	parser.add_argument('--serve', help='launch web UI for demo', action='store_true')
	args = parser.parse_args()

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	K.set_session(sess)

	if args.model[0:2] != '..':
		args.model = os.path.join('../models', args.model)

	if args.model[-3:] != '.h5':
		args.model += '.h5'

	mf = ModelFactory(modelName=args.model, batchSize=32, numClasses=53, imgSize=(28,28,1), dropoutRatio=0.0,
				 numFilters=[8,16,32,64,128,256], kernelVals=[3,3,3,5,5,5], poolVals=[2,2,2,1,1,1], strideVals=[1,1,1,1,1,1],
				 learningRate=0.01, numEpochs=50)
	
	segmenter = Segmenter()

	if args.train:
		model = mf.build()
		mf.train(model)
		mf.save(model)

	elif args.infer:
		'Infer on a single word / character image'
		model = load_model(args.model)
		print('[INFO] Load model from {}'.format(args.model))

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
		'Test the model on a directory of character images'
		model = load_model(args.model)
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

	elif args.serve:
		from flask import Flask, render_template, request
		from PIL import Image
		import io
		
		app = Flask(__name__)

		model = load_model(args.model)
		print('[INFO] Load model from {}'.format(args.model))
		graph = tf.get_default_graph()

		@app.route('/', methods=['GET', 'POST'])
		def upload():
			return render_template('upload.html')

		@app.route('/predict', methods=['GET', 'POST'])
		def predict_display():
			if request.method == 'POST' or request.method=='GET':
				with graph.as_default():
					img = np.array(Image.open(io.BytesIO(request.files['image'].read())).convert('L'))
					imgs = segmenter.segment(img)
					pred = ""
					for img in imgs:
						img = mf.preprocess(img, False)
						prediction = predict(mf.mapping, model, img)
						pred += prediction
					return render_template('predict.html', pred=pred, orig='../sample_imgs/ABISHEK.png')
			return 'Please upload an image'

		app.run(debug=True)

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
