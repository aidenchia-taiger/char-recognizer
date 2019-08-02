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
import shutil
from Utils import display, output, makeDir
from Model import ModelFactory
from Segmenter.Segmenter import Segmenter
from TextDetector.TesseractTextDetector import TextDetector
from SpellCorrector.SimpleSpellCorrector import SpellCorrector
from Deslanter.Deslanter import Deslanter
from tessTest import TesseractRunner

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

mf = ModelFactory()
model = mf.load()
graph = tf.get_default_graph()
segmenter = Segmenter()
textDetector = TextDetector()
deslanter = Deslanter()
spellCorrector = SpellCorrector(lexicon="SpellCorrector/lexicon.txt", misclassify="SpellCorrector/misclassify.json")
tr = TesseractRunner()

@app.route('/', methods=['GET', 'POST'])
def upload():
	return render_template('upload.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	if request.method == 'POST' or request.method=='GET':
		# Use global var graph otherwise Flask loads new graph for each new request
		global graph
		with graph.as_default():
			# Convert from PIL img to numpy array
			img = np.array(Image.open(io.BytesIO(request.files['image'].read())).convert('L'))

			# Get the image filename
			filename = request.files.getlist("image")[0].filename

			# Copy the image over to static folder so we can display it on the UI
			shutil.copy(os.path.join('../sample_imgs', filename), os.path.join("static", filename))
			
			docImg = deslanter.deslant(img)

			# Handwritten recognition model prediction
			textPreds, lineBoxes = mf.predictDoc(model, segmenter, textDetector, docImg, spellCorrector, \
											 	 showCrop=False, showChar=False)
			pred = output(textPreds, lineBoxes, 'out.hocr', 'out.txt')

			# Tesseract prediction
			tesspred = tr.run(docImg)

			return render_template('predict.html', pred=pred, tesspred=tesspred, filename=filename)

	return "Please upload an image"

if __name__ == "__main__":
	app.run(debug=args.debug)





