import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, MaxPooling2D, Dense, Conv2D, Flatten, Dropout, Activation, BatchNormalization
from keras.activations import softmax, relu, sigmoid
from keras.losses import categorical_crossentropy
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from Utils import display, save, percentageBlack
from keras.optimizers import Adam
import numpy as np
import cv2
from Utils import display, save
import datetime
import os

class ModelFactory:
	def __init__(self, modelName='model', batchSize=32, numClasses=53, imgSize=(28,28,1), dropoutRatio=0.6,
				 numFilters=[16,32,64], kernelVals=[4,4,4], poolVals=[2,2,2], strideVals=[1,1,1],
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
		self.logpath = os.path.join('../logs', datetime.datetime.now().strftime("Time_%H%M_Date_%d-%m")) + "_" + self.modelName
		self.savepath = os.path.join('../models', self.modelName) + '.h5'
		[print('[INFO] {}: {}'.format(k,v)) for k, v in dict(vars(self)).items()]

	def build(self):
		numCNNlayers = len(self.kernelVals)
		
		inputs = Input(shape=(self.imgSize))
		x = Conv2D(filters=self.numFilters[0], padding='SAME', kernel_size=self.kernelVals[0], strides=self.strideVals[0])(inputs)
		for i in range(1, numCNNlayers):
			x = Conv2D(filters=self.numFilters[i], padding='SAME', kernel_size=self.kernelVals[i], strides=self.strideVals[i])(x)
			x = BatchNormalization()(x)
			x = Activation('relu')(x)
			x = MaxPooling2D(pool_size=self.poolVals[i])(x)

		x = Flatten()(x)
		x = Dense(128, activation='relu')(x)
		x = Dropout(0.0)(x)
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
		model.fit_generator(train_generator, steps_per_epoch= len(train_generator) / self.batchSize, 
							validation_data=valid_generator, validation_steps= len(valid_generator),
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
		valid_datagen = ImageDataGenerator(rescale=1./255)
		valid_generator = valid_datagen.flow_from_directory('../imgs/validation', target_size = (self.imgSize[0], self.imgSize[1]), 
													  batch_size=self.batchSize, color_mode='grayscale', 
													  class_mode='categorical', shuffle=False)
		mapping = valid_generator.class_indices
		inv_mapping = {v:k for k,v in mapping.items()}
		return inv_mapping

	def preprocess(self, img):
		# invert colours if black text on white bg
		img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[-1] if percentageBlack(img) < 50 else img
		img = cv2.resize(img, (self.imgSize[0], self.imgSize[1]))
		display(img)
		img = np.expand_dims(img, axis=0)
		img = np.expand_dims(img, axis=-1)
		img = img.astype(np.float32) / 255.0

		return img


if __name__ == '__main__':
	mf = ModelFactory()
	model = mf.build()
