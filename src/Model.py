import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Input, MaxPooling2D, Dense, Conv2D, Flatten, Dropout, Activation, BatchNormalization
from keras.activations import softmax, relu, sigmoid
from keras.losses import categorical_crossentropy
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import numpy as np
import cv2
import pandas as pd
import datetime
import pickle
import editdistance
import os
import pdb
import json
from Utils import display, save, percentageBlack
from Segmenter import Segmenter
import warnings
from Tesseract_TextDetector import TextDetector

class ModelFactory:
	def __init__(self, configFile):
		configs = json.load(open(configFile))

		self.modelName = configs['Model Name']
		self.numClasses = configs['Num Classes']
		self.mapping = self.getMapping(configs['Mapping File'])
		self.trainDir = configs['Train Dir']
		self.validDir = configs['Valid Dir']
		self.testDir = configs['Test Dir']
		self.dropoutRatio = configs['Dropout Ratio']
		self.learningRate = configs['Learning Rate']
		self.numEpochs = configs['Num Epochs']
		self.imgSize = tuple(configs['Image Size'])
		self.batchSize = configs['Batch Size']
		self.numFilters = configs['Num Filters']
		self.kernelVals = configs['Kernel Values']
		self.poolVals = configs['Pool Values']
		self.strideVals = configs['Stride Values']
		self.minConfidence = configs['Min Confidence']
		self.logpath = os.path.join('../logs', datetime.datetime.now().strftime("Time_%H%M_Date_%d-%m")) + "Model_" + self.modelName 
		self.savepath = os.path.join('../models', self.modelName) + '.h5'

		[print('[INFO] {}: {}'.format(k,v)) for k, v in dict(vars(self)).items()]
		assert len(self.kernelVals) == len(self.poolVals) == len(self.numFilters)
		assert len(self.imgSize) == 3

	def build(self):
		numCNNlayers = len(self.kernelVals)
		
		inputs = Input(shape=(self.imgSize))
		x = Conv2D(filters=self.numFilters[0], padding='SAME', kernel_size=self.kernelVals[0], strides=self.strideVals[0])(inputs)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = MaxPooling2D(self.poolVals[0])(x)

		for i in range(1, numCNNlayers):
			x = Conv2D(filters=self.numFilters[i], padding='SAME', kernel_size=self.kernelVals[i], strides=self.strideVals[i])(x)
			x = BatchNormalization()(x)
			x = Activation('relu')(x)
			x = MaxPooling2D(pool_size=self.poolVals[i])(x)

		x = Flatten()(x)
		x = Dense(128, activation='relu')(x)
		x = Dropout(self.dropoutRatio)(x)
		outputs = Dense(self.numClasses, activation='softmax')(x)

		model = Model(inputs=inputs, outputs=outputs)

		print(model.summary())
		model.compile(optimizer=Adam(self.learningRate), loss='categorical_crossentropy', metrics=['accuracy'])
		return model


	def train(self, model):
		model.compile(optimizer=Adam(self.learningRate), loss='categorical_crossentropy', metrics=['accuracy'])
		train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.2, height_shift_range=0.2, 
										   shear_range=0.2)
		train_generator = train_datagen.flow_from_directory(self.trainDir, target_size = (self.imgSize[0], self.imgSize[1]), 
													  batch_size=self.batchSize, color_mode='grayscale', 
													  class_mode='categorical', shuffle=True)

		valid_datagen = ImageDataGenerator(rescale=1./255)
		valid_generator = valid_datagen.flow_from_directory(self.validDir, target_size = (self.imgSize[0], self.imgSize[1]), 
													  batch_size=self.batchSize, color_mode='grayscale', 
													  class_mode='categorical', shuffle=False) 

		checkpoint = ModelCheckpoint(self.savepath, monitor='val_acc', verbose=1, save_best_only=True)
		tensorboard = TensorBoard(self.logpath, batch_size=self.batchSize)
		reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001, min_delta=0.0001)
		callbacks = [checkpoint, tensorboard, reduceLR]
		model.fit_generator(train_generator, steps_per_epoch= train_generator.n // self.batchSize, 
							validation_data=valid_generator, validation_steps= valid_generator.n // self.batchSize,
							epochs=self.numEpochs, callbacks=callbacks)

	def save(self, model):
		model.save(self.savepath)
		print('[INFO] Saved model to: {}'.format(self.savepath))

	def load(self):
		model = load_model(self.savepath)
		print('[INFO] Loading model: {}'.format(self.savepath))
		model.compile(optimizer=Adam(self.learningRate), loss='categorical_crossentropy', metrics=['accuracy'])

		return model

	def getMapping(self, mappingFile):
		if os.path.exists(mappingFile):
			return pickle.load(open(mappingFile, 'rb'))

		# If mapping file doesn't exist, obtain mapping from validation dataset
		valid_datagen = ImageDataGenerator(rescale=1./255)
		valid_generator = valid_datagen.flow_from_directory(self.validDir, target_size = (self.imgSize[0], self.imgSize[1]), 
													  batch_size=self.batchSize, color_mode='grayscale', 
													  class_mode='categorical', shuffle=False)
		mapping = valid_generator.class_indices
		inv_mapping = {v:k for k,v in mapping.items()}
		return inv_mapping

	def preprocess(self, img, show=True, minBlack=30, invert=True):
		# invert colours to make image a white text on black bg
		if invert:
			img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[-1]

		if percentageBlack(img) < minBlack:
			warnings.warn("PLEASE ENSURE DISPLAYED IMAGE IS BLACK TEXT ON WHITE BG.")
			#display(img)
		img = cv2.resize(img, (self.imgSize[0], self.imgSize[1]))
		if show:
			display(img)
		img = np.expand_dims(img, axis=0)
		img = np.expand_dims(img, axis=-1)
		img = img.astype(np.float32) / 255.0

		return img

	def predictChar(self, model, charImg):
		predictions = model.predict(charImg, batch_size=1, verbose=1)
		prob = max(max(predictions))
		# Replace prediction with empty str if below a certain confidence threshold
		predLabel = self.mapping[np.argmax(predictions)] if prob > self.minConfidence else ""
		print('[INFO] Predicted: {} | Probability: {}'.format(predLabel, prob))
		return predLabel

	def predictWord(self, model, segmenter, wordImg, show=True):
		if show:
			display(wordImg)
		charImgs = segmenter.segment(wordImg)
		pred = ""
		for charImg in charImgs:
			charImg = self.preprocess(charImg, show)
			prediction = self.predictChar(model, charImg)
			pred += prediction

		return pred

	def predictDoc(self, model, segmenter, textDetector, docImg, showCrop=False, showChar=False):
		textPreds = {"text": [], "top": [], "left": [], "width": [], "height": []}
		lineBoxes, textBoxes = textDetector.detect(docImg, show=showCrop)
		for textBox in textBoxes:
			wordImg, coord = textBox
			pred = self.predictWord(model, segmenter, wordImg, show=showChar)
			print('Pred: {} | Coord: {}'.format(pred, coord))
			# only include those predictions which are non-empty strings. Empty string means model's prediction probabiliy is below prob threshold
			if pred != "":
				textPreds["text"].append(pred)
				textPreds["top"].append(coord['y'])
				textPreds["left"].append(coord['x'])
				textPreds["width"].append(coord['w'])
				textPreds["height"].append(coord['h'])

		textPreds = pd.DataFrame.from_dict(textPreds)

		return textPreds, lineBoxes

	def getCER(self, pred, gt):
		cer = editdistance.eval(pred, gt) * 100/ len(gt)
		print("[INFO] Ground Truth: {} | Predicted: {}".format(pred, gt))
		print("[INFO] Character Error Rate: {:.1f}%".format(cer))

		return cer

if __name__ == '__main__':
	mf = ModelFactory('betaconfig.json')
	model = mf.build()
