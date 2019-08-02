'''
This file is used to run a tesseract inference on a document, and output the CER and word accuracy of the prediction.
Use the metrics as a performance benchmark against the model you've built.
'''
import cv2
import argparse
import json
import editdistance
from PIL import Image
from Deslanter.Deslanter import Deslanter
from tesserocr import PyTessBaseAPI

class TesseractRunner:
	def __init__(self):
		pass

	def run(self, img):
		self.orig = img
		img = Image.fromarray(img)

		with PyTessBaseAPI() as api:
			api.SetImage(img)
			pred = api.GetUTF8Text()

		f = open('tessout.txt', 'w+')
		f.write(pred)
		print('[INFO] Written to tessout.txt')

		pred = ' '.join(pred.split('\n'))

		return pred

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--infer', help="path to input file")
	parser.add_argument('--gt', help="path to gt json file", default=None)
	args = parser.parse_args()

	tr = TesseractRunner()
	deslanter = Deslanter()

	img = cv2.imread(args.infer)
	img = deslanter.deslant(img)
	
	pred = tr.run(img)

	# If the ground truth is in another folder, grab just the filename itself instead of the whole path
	if args.gt:
		key = args.infer.split('/')[-1] if '/' in args.infer else args.infer
		gt = json.load(open(args.gt))[key]

		# View prediction vs ground truth
		print("[INFO] Ground Truth: \n{}\n".format(gt))
		print("[INFO] Prediction: \n{}\n".format(pred))

		# Calculate the CER
		cer = editdistance.eval(pred, gt) * 100/ len(gt)
		print("[INFO] Character Error Rate: {:.1f}%".format(cer))

		# Calculate Word Accuracy
		pred = pred.split(' ')
		gt = gt.split(' ')
		wa = len(list(set(pred).intersection(gt))) * 100 / len(gt)
		print("[INFO] Word Accuracy: {:.1f}%".format(wa))


if __name__ == "__main__":
	main()