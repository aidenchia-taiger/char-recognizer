import sys
import cv2
import os

sys.path.append("../")
from Model import ModelFactory
from probableSegmentsMixed import *

def searchModel():

	parser = argparse.ArgumentParser()
	parser.add_argument('--input_path', type=str, help='Input images path', default="")
	args = parser.parse_args()
	

	mf = ModelFactory()
	model = mf.load()

	CER = []
	ground_truth = []
	predicted = []
	labels = ["nominating", "any", "move", "meeting", "been", "texts", "ready", "one"]
	ctr = 0

	#beam search widrth not being used now
	#beam_width = 3

	files = os.listdir(args.input_path)
	files = sorted(files)
	for file in files:
		file_name = os.path.join(args.input_path, file)

		#get segment boundaries
		results = get_segments(file_name)
		plt.close('all')


		#search algorithm
		probables = [] #[segments, last, cost, str]
		image_len = len(results)-1 #no of image segments
		last = 0
		print("Current working directory", os.getcwd())
		img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
		# img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[-1]

		#preprocessing
		# img = cv2.threshold(img, 200, 1, cv2.THRESH_BINARY_INV)[-1]
		# img = skeletonize(img)
		# img = np.float32(img)


		h, w = img.shape
		final_results = []
		while last<=image_len:
			if len(probables) == 0:
				patch = img[0:h, results[0]:results[0+1]]
				#cv2.imshow("images23", patch)
				#cv2.waitKey(1000)
				patch = mf.preprocess(patch, invert=True)
				pred_char, prob = mf.predictChar(model, patch)
				probables.append([[[0]], 1, prob, pred_char])
				print('segments', 0, 0)
				
				patch = img[0:h, results[0]:results[0+2]]
				patch = mf.preprocess(patch, invert=True)
				pred_char, prob = mf.predictChar(model, patch)
				probables.append([[[0, 1]], 2, prob, pred_char])
				print('segments', 0, 1)

				last = 0
			else:
				options = []
				last_element_reached = image_len+1
				for i in range(len(probables)):
					segments, last, cost, word = probables[i]
					if last + 1 <= image_len:
						new_segments = segments.copy()
						new_segments.append([last])

						new_last = last + 1
						last_element_reached = min(last_element_reached, new_last)

						patch = img[0:h, results[last]:results[new_last]]
						patch = mf.preprocess(patch, invert=True)
						pred_char, prob = mf.predictChar(model, patch)
						print('segments', last, last)

						new_cost = cost + prob
						new_word = word + pred_char

						options.append([new_segments, new_last, new_cost, new_word])

					if last + 2 <= image_len:
						new_segments = segments.copy()
						new_segments.append([last, last+1])

						new_last = last + 2
						last_element_reached = min(last_element_reached, new_last)

						patch = img[0:h, results[last]:results[new_last]]
						patch = mf.preprocess(patch, invert=True)
						pred_char, prob = mf.predictChar(model, patch)
						print('segments', last, last+1)

						new_cost = cost + prob
						new_word = word + pred_char

						options.append([new_segments, new_last, new_cost, new_word])

				probables = sorted(options, key=lambda x: x[2], reverse=True)
				last = max(last, last_element_reached)
				for i in probables:
					if(i[1]==image_len):
						final_results.append(i)

		for ele in final_results:
			print(ele)

		
		pred = final_results[-1][-1]
		print(pred)

		print(file)
		gt = file.split('/')[-1]
		print(gt)
		gt = gt.split('.')[0]
		print(gt)
		gt = labels[ctr]
		print("labels", gt)
		ctr += 1
		CER.append(mf.getCER(pred, gt))
		ground_truth.append(gt)
		predicted.append(pred)

	for i in range(len(predicted)):
		print(ground_truth[i], predicted[i], CER[i])

	print(np.average(np.array(CER)))

if __name__ == "__main__":
	searchModel()