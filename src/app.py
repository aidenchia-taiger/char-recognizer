from flask import Flask, render_template, request
from PIL import Image
import io
import argparse
import os
import keras
from keras import backend as K
import numpy as np
import tensorflow as tf
import cv2
import pdb
from Utils import display, percentageBlack
from Model import ModelFactory
from Segmenter import Segmenter

# Command line args
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='select which model to use', default="beta")
parser.add_argument('--debug', help="turn on debug mode", action="store_true")
args = parser.parse_args()

# Prevent CUDA OUT OF MEMORY error
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


app = Flask(__name__)

mf = ModelFactory(modelName=args.model)
model = mf.load()
graph = tf.get_default_graph()
segmenter = Segmenter()

@app.route('/', methods=['GET', 'POST'])
def upload():
	return render_template('upload.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	if request.method == 'POST' or request.method=='GET':
		# Use global var graph otherwise Flask loads new graph for each new request
		global graph
		with graph.as_default():
			wordImg = np.array(Image.open(io.BytesIO(request.files['image'].read())).convert('L'))
			pred = mf.predictWord(model, segmenter, wordImg, show=False)
			return render_template('predict.html', pred=pred)

	return "Please upload an image"

if __name__ == "__main__":
	app.run(debug=args.debug)





