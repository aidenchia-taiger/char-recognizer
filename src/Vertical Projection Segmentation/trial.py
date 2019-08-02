sys.path.append("../")
from Model import ModelFactory
import cv2
import os
import numpy as np

mf = ModelFactory()
model = mf.load()


files = os.listdir('../imgs/trial')
for file in files:
	patch = cv2.imread('../imgs/trial/'+file, cv2.IMREAD_GRAYSCALE)
	print('../imgs/trial/'+file)

	cv2.imshow("image", patch)
	cv2.waitKey(1000)
	patch = np.expand_dims(patch, 0)
	patch = np.expand_dims(patch, -1)
	#patch = patch.reshape(1, 28, 28, 1)
	#patch 

	pred_char, prob = mf.predictChar(model, patch)
	print(pred_char, prob)
	enter = input("enter")