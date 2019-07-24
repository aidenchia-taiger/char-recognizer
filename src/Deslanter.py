# Reference: https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/

import numpy as np 
import cv2
import argparse
from pdb import set_trace
from Utils import display, save

class Deslanter:
	def __init__(self):
		pass

	def detectSlant(self, img):
		# grayscale the image if it isn't already
		if len(img.shape) == 3:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# make the document white text on black bg
		img = cv2.bitwise_not(img)
		img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		# find all (x,y) coordinates in the image that are part of the foreground
		coords = np.column_stack(np.where(img > 0))

		# pass all the coords into a cv2 function that finds the min rotated rectangle that contains the entire text region
		angle = cv2.minAreaRect(coords)[-1]

		# the cv2.minAreaRect function returns values [-90, 0)
		# https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned
		angle = -(90 + angle) if angle < -45 else -angle
		
		print('[INFO] Rotate by: {:.2f} degrees'.format(angle))
		return angle

	def deslantImg(self, img, angle, show=False):
		(h, w) = img.shape[:2]
		center = (w // 2, h // 2)
		rm = cv2.getRotationMatrix2D(center, angle, 1.0)	
		rotated = cv2.warpAffine(img, rm, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
		
		if show:
			display_img = [img, rotated]
			display(display_img, multiple=True, rows=1, cols=2)
		
		#save(rotated, 'rotated', prefix='../out', suffix=".png")

		return rotated

	def deslant(self, img):
		angle = self.detectSlant(img)
		rotated_img = self.deslantImg(img, angle)

		return rotated_img


if __name__ == "__main__":
	deslanter = Deslanter()
	img = cv2.imread('../sample_imgs/slant.jpg')
	deslanter.deslant(img)