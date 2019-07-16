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
from Utils import display, save, percentageBlack
from Segmenter import Segmenter
from TextDetector import TextDetector

class ModelFactory:
	def __init__(self, modelName="model", batchSize=32, numClasses=53, imgSize=(28,28,1), dropoutRatio=0.0,
				 numFilters=[8,16,32,64,128,256], kernelVals=[3,3,3,5,5,5], poolVals=[2,2,2,1,1,1], strideVals=[1,1,1,1,1,1],
				 learningRate=0.01, numEpochs=50):
		self.dropoutRatio = dropoutRatio
		self.learningRate = learningRate
		self.numEpochs = numEpochs
		self.imgSize = imgSize
		self.batchSize = batchSize
		self.numClasses = numClasses
		self.numFilters = numFilters
		self.kernelVals = kernelVals
		self.poolVals = poolVals
		self.strideVals = strideVals
		self.modelName = modelName
		self.mapping = self.getMapping()
		self.logpath = os.path.join('../logs', datetime.datetime.now().strftime("Time_%H%M_Date_%d-%m")) + "_" + self.modelName.split('/')[-1]
		self.savepath = os.path.join('../models', modelName) + '.h5'
		[print('[INFO] {}: {}'.format(k,v)) for k, v in dict(vars(self)).items()]

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
		train_generator = train_datagen.flow_from_directory('../imgs/train', target_size = (self.imgSize[0], self.imgSize[1]), 
													  batch_size=self.batchSize, color_mode='grayscale', 
													  class_mode='categorical', shuffle=True)

		valid_datagen = ImageDataGenerator(rescale=1./255)
		valid_generator = valid_datagen.flow_from_directory('../imgs/validation', target_size = (self.imgSize[0], self.imgSize[1]), 
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

	def getMapping(self):
		if os.path.exists('../models/mapping.pkl'):
			return pickle.load(open('../models/mapping.pkl', 'rb'))

		valid_datagen = ImageDataGenerator(rescale=1./255)
		valid_generator = valid_datagen.flow_from_directory('../imgs/validation', target_size = (self.imgSize[0], self.imgSize[1]), 
													  batch_size=self.batchSize, color_mode='grayscale', 
													  class_mode='categorical', shuffle=False)
		mapping = valid_generator.class_indices
		inv_mapping = {v:k for k,v in mapping.items()}
		return inv_mapping

	def preprocess(self, img, show=True):
		# invert colours if black text on white bg
		img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[-1] if percentageBlack(img) < 50 else img
		img = cv2.resize(img, (self.imgSize[0], self.imgSize[1]))
		if show:
			display(img)
		img = np.expand_dims(img, axis=0)
		img = np.expand_dims(img, axis=-1)
		img = img.astype(np.float32) / 255.0

		return img

	def predictChar(self, model, charImg, threshold=0.5):
		predictions = model.predict(charImg, batch_size=1, verbose=1)
		prob = max(max(predictions))
		predLabel = self.mapping[np.argmax(predictions)] if prob > threshold else ""
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
		result = {"text": [], "top": [], "left": [], "width": [], "height": []}
		croptexts = textDetector.detect(docImg, show=showCrop)
		for crop in croptexts:
			wordImg, coord = crop
			left, top, right, bot = coord 
			pred = self.predictWord(model, segmenter, wordImg, show=showChar)
			print('Pred: {} | Coord: {}'.format(pred, coord))
			result["text"].append(pred)
			result["top"].append(top)
			result["left"].append(left)
			result["width"].append(right - left)
			result["height"].append(bot - top)

		result = pd.DataFrame.from_dict(result)
		return result

	def getCER(self, pred, gt):
		cer = editdistance.eval(pred, gt) * 100/ len(gt)
		print("[INFO] Ground Truth: {} | Predicted: {}".format(pred, gt))
		print("[INFO] Character Error Rate: {:.1f}%".format(cer))

		return cer

if __name__ == '__main__':
	mf = ModelFactory()
	model = mf.build()
