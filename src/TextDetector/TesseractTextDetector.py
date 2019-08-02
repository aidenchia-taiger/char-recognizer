import os
import pdb
import cv2
import sys
import numpy as np
from PIL import Image
from tesserocr import PyTessBaseAPI, RIL
import sys
sys.path.append("..")
from Utils import display, save

class TextDetector:
	def detect(self, img, show=True, write_to_dir=False, filename=None):
		self.orig = img
		PIL_img = Image.fromarray(img)
		lineBoxes = self.detectLines(PIL_img)
		textBoxes = self.detectTexts(PIL_img, show, write_to_dir, filename)
		return lineBoxes, textBoxes
	
	def detectLines(self, img):
		with PyTessBaseAPI() as api:
			api.SetImage(img)
			# GetComponentImages returns a list of tuples (PIL img, bbox, _, _). For lines, we only care about the bbox.
			lineBoxes = [x[1] for x in api.GetComponentImages(RIL.TEXTLINE, True)]
			print('[INFO] Found {} textline image components.'.format(len(lineBoxes)))
		return lineBoxes

	def detectTexts(self, img, show=True, write_to_dir=False, filename=None):
		with PyTessBaseAPI() as api:
			api.SetImage(img)
			# GetComponentImages returns a list of tuples (PIL img, bbox, _, _)
			textBoxes = [(np.array(x[0]),x[1]) for x in api.GetComponentImages(RIL.WORD, True)]
			print('[INFO] Found {} word image components.'.format(len(textBoxes)))

			for i, textBox in enumerate(textBoxes):
				x = textBox[1]['x']
				w = textBox[1]['w']
				y = textBox[1]['y']
				h = textBox[1]['h']
				cv2.rectangle(self.orig, (x, y), (x+w, y+h), (255, 255, 255), 1)

				if write_to_dir:
					# Crop out the individual words, add in some margin
					subimg = self.orig[y:y+h+5, x:x+w+5]
					#display(subimg)
					savename = filename + '_' + str(i)
					save(subimg, name=savename, prefix=None, suffix='.png')

			if show:
				display(self.orig)

				
		return textBoxes

if __name__ == "__main__":	
	td = TextDetector()
	#img = cv2.imread('../../digital_para/ss.png')
	#td.detect(img, show=True, write_to_dir=True)

	## Extract out words from digital paragraph
	for root, _, files in os.walk('../../para_imgs'):
		for i, file in enumerate(files):
			if '.png' not in file and '.jpg' not in file:
				continue

			fullpath = os.path.join(root, file)
			img = cv2.imread(fullpath)
			td.detect(img, show=False, write_to_dir=True, filename=file[:-4])

			if i == 30:
				break