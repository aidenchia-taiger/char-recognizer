import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, MaxPooling2D, Dense, Conv2D, Flatten, Dropout, Activation
from keras.activations import softmax, relu, sigmoid
from keras.losses import categorical_crossentropy
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from Utils import display, save
from keras.optimizers import Adam
import numpy as np
import datetime
import os

class ModelFactory:
	def __init__(self, batchSize=32, numClasses=47, imgSize=(28,28,1), dropoutRatio=0.6, modelName='model',
				 filterVals=[16,32,64], kernelVals=[4,4,4], poolVals=[2,2,2], strideVals=[1,1,1],
				 learningRate=0.01, validation_split=0.1, numEpochs=50):
		self.dropoutRatio = dropoutRatio
		self.learningRate = learningRate
		self.numEpochs = numEpochs
		self.imgSize = imgSize
		self.batchSize = batchSize
		self.numClasses = numClasses
		self.filterVals = filterVals
		self.kernelVals = kernelVals
		self.poolVals = poolVals
		self.strideVals = strideVals
		self.validation_split = validation_split
		self.opt = Adam(self.learningRate)
		self.modelName = modelName
		self.logpath = os.path.join('../models/logs', datetime.datetime.now().strftime("Time_%H%M_Date_%d-%m")) + "_" + self.modelName
		self.savepath = os.path.join('../models', self.modelName)
		

	def build(self):
		numCNNlayers = len(self.kernelVals)
		
		inputs = Input(shape=(self.imgSize))
		x = Conv2D(filters=self.filterVals[0], kernel_size=self.kernelVals[0], strides=self.strideVals[0])(inputs)
		for i in range(1, numCNNlayers):
			x = Conv2D(filters=self.filterVals[i], kernel_size=self.kernelVals[i], strides=self.strideVals[i])(x)
			x = Activation('relu')(x)
			x = MaxPooling2D(pool_size=self.poolVals[i])(x)

		x = Flatten()(x)
		x = Dense(512, activation='relu')(x)
		x = Dropout(self.dropoutRatio)(x)
		outputs = Dense(self.numClasses, activation='softmax')(x)

		model = Model(inputs=inputs, outputs=outputs)

		print(model.summary())
		model.compile(optimizer=self.opt, loss='categorical_crossentropy', metrics=['accuracy'])
		return model


	def train(self, model):
		train_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.2, height_shift_range=0.2, 
										   validation_split=self.validation_split)
		train_generator = train_datagen.flow_from_directory('../imgs/', target_size = (self.imgSize[0], self.imgSize[1]), 
													  batch_size=self.batchSize, color_mode='grayscale', 
													  class_mode='categorical', shuffle=True)

		checkpoint = ModelCheckpoint(self.savepath, monitor='val_acc', verbose=1, save_best_only=True)
		tensorboard = TensorBoard(self.logpath, batch_size=self.batchSize)
		reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001, min_delta=0.0001)
		callbacks = [checkpoint, tensorboard, reduceLR]
		model.fit_generator(generator, epochs=self.numEpochs, callbacks=callbacks)

	def save(self, model):
		model_json = model.to_json()
		with open(self.savepath, 'w') as json_file:
			json_file.write(model_json)

		model.save_weights(self.savepath + '.h5')
		print('[INFO] Saved model to: {}'.format(self.savepath + '.h5'))

	def load(self):
		with open(self.savepath, 'r') as json_file:
			loaded_model_json = json_file.read()
		
		model = model_from_json(loaded_model_json)
		model.load_weights(self.savepath + '.h5')
		print('[INFO] Loading model: {}'.format(self.savepath + '.h5'))
		model.compile(optimizer=self.opt, loss='categorical_crossentropy', metrics=['accuracy'])

		return model


if __name__ == '__main__':
	model = ModelFactory().build()
		
