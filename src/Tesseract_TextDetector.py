from PIL import Image
from tesserocr import PyTessBaseAPI, RIL
from Utils import display
import pdb
import cv2
import numpy as np

class TextDetector:
	def detect(self, img, show=True):
		self.orig = img
		PIL_img = Image.fromarray(img)
		lineBoxes = self.detectLines(PIL_img)
		textBoxes = self.detectTexts(PIL_img, show)
		return lineBoxes, textBoxes
	
	def detectLines(self, img):
		with PyTessBaseAPI() as api:
			api.SetImage(img)
			# GetComponentImages returns a list of tuples (PIL img, bbox, _, _). For lines, we only care about the bbox.
			lineBoxes = [x[1] for x in api.GetComponentImages(RIL.TEXTLINE, True)]
			print('[INFO] Found {} textline image components.'.format(len(lineBoxes)))
		return lineBoxes

	def detectTexts(self, img, show=True):
		with PyTessBaseAPI() as api:
			api.SetImage(img)
			# GetComponentImages returns a list of tuples (PIL img, bbox, _, _)
			textBoxes = [(np.array(x[0]),x[1]) for x in api.GetComponentImages(RIL.WORD, True)]
			print('[INFO] Found {} word image components.'.format(len(textBoxes)))

		if show:
			for textBox in textBoxes:
				x = textBox[1]['x']
				w = textBox[1]['w']
				y = textBox[1]['y']
				h = textBox[1]['h']
				cv2.rectangle(self.orig, (x, y), (x+w, y+h), (0, 255, 0), 2)
			display(self.orig)
		return textBoxes

if __name__ == "__main__":
	td = TextDetector()
	img = cv2.imread('../sample_imgs/cleandoc.png')
	td.detect(img)
