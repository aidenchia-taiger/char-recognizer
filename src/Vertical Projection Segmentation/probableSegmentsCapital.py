import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.morphology import skeletonize
from numpy import diff
import argparse

def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3

def remove_noise_and_smooth(file_name):
    img = cv2.imread(file_name, 0)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    cv2.imshow("or_image", or_image)
    return or_image

def oneD_NMS(black_ctr):
	#Simple NMS
	i = 1
	nms_results = []
	while i+1 < len(black_ctr):
		if black_ctr[i] > black_ctr[i+1]:
			if black_ctr[i] >= black_ctr[i-1]:
				nms_results.append(i)
		else:
			i = i + 1
			while i+1 < len(black_ctr) and black_ctr[i] <= black_ctr[i+1]:
				i = i+1
			if i+1 < len(black_ctr):
				nms_results.append(i)
		i = i + 2
	print(nms_results)

def twoD_NMS(black_ctr, start, end, p_max):
	p_max[end] = black_ctr[end]
	best = end
	while end > start:
		end = end - 1
		if black_ctr[end] <= black_ctr[best]:
			p_max[end] = black_ctr[best]
		else:
			p_max[end] = black_ctr[end]
			best = end
	return best, p_max

def threeD_NMS(black_ctr, n):
	i = n
	p_max = np.zeros(len(black_ctr)) 
	something, p_max = twoD_NMS(black_ctr, 0, i-1, p_max)
	chkpt = -1
	W = len(black_ctr)
	nms_results = []
	while i < W - 2 * n:
		j, p_max = twoD_NMS(black_ctr, i, i+n, p_max)
		k, p_max = twoD_NMS(black_ctr, i+n+1, j+n, p_max)
		if i == j or black_ctr[j] > black_ctr[k]:
			if (chkpt <= j-n or black_ctr[j] >  p_max[chkpt]) and (j - n == 1 or black_ctr[j] >= p_max[j-n]):
				nms_results.append(j)
			if i < j:
				chkpt = i + n + 1
			i = j + n + 1
		else:
			i = k
			chkpt = j + n + 1
			while i < W - n:
				j, p_max = twoD_NMS(black_ctr, chkpt, i+n, p_max)
				if black_ctr[i] > black_ctr[j]:
					nms_results.append(i)
					i = i + n - 1
					break
				else:
					chkpt = i + n - 1
					i = j
	return nms_results

def readFile(location):
	print(location)
	img = cv2.imread(location, cv2.IMREAD_GRAYSCALE)
	h, w = img.shape
	print("Image shape ",h, w)
	return h, w, img

def pre_process_image(img, dilation=False, inversion=True, skeleton=True, show=False):

	if inversion:
		img = cv2.threshold(img, 200, 1, cv2.THRESH_BINARY_INV)[-1]
		print(type(img))
		if show:
			cv2.imshow("Inverse image", img)

	if skeleton:
		skeleton_image = skeletonize(img)
		img = np.float32(skeleton_image)
		if show:
			cv2.imshow("Skeleton image", img)

	if dilation:
		d_kernel = np.ones((1, 1),np.uint8)
		e_kernel = np.ones((2, 2),np.uint8)

		#inv = cv2.dilate(inv, d_kernel, iterations=1)
		#inv = cv2.erode(inv, e_kernel, iterations=2)
		#inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, d_kernel)
		# cv2.imshow("image", d_kernel)
		# cv2.imshow("inverse", e_kernel)
		# cv2.imshow("difference", gradient)

		# inv = cv2.dilate(inv, d_kernel,iterations = 1)
		# inv = cv2.erode(inv, e_kernel,iterations = 2)
		# inv = cv2.dilate(inv, d_kernel,iterations = 1)
		# inv = cv2.erode(inv, e_kernel,iterations = 2)
		# inv = cv2.dilate(inv,kernel,iterations = 3)

	return img


def vertical_projection(img, h, w):
	black_ctr = []
	for x in range(0, w):
	    temp = img[0:h, x:x+1]
	    #black_ctr.append(len(temp.flatten()) - np.count_nonzero(temp))
	    black_ctr.append(np.count_nonzero(temp))
	#print(black_ctr)
	plt.plot(black_ctr)
	#plt.show()
	print("Black ctr done")
	return black_ctr

def get_segment_boundaries(black_ctr, inv, h, w):
	results = []
	upper_results = []
	lower_results = []
	# for i in range(len(nms_results)-1):
	# 	results.append(int(nms_results[i] + (nms_results[i+1]-nms_results[i])/2))
	# results.insert(0, 0)
	# results.append(len(black_ctr))
	i = 0
	while i<len(black_ctr):
		if black_ctr[i] == 0:
			start = i
			while i<len(black_ctr) and black_ctr[i] <= 0:
			# while i<len(black_ctr) and black_ctr[i] <= 1:
				i += 1
			end = i-1
			results.append(start+int((end-start)/2))
			lower_results.append(start)
			upper_results.append(end)
		# elif black_ctr[i] == 1:
		# 	start = i
		# 	while i<len(black_ctr) and black_ctr[i] <= 1:
		# 		i += 1
		# 	end = i-1
		# 	if end-start >= 5:
		# 		results.append(start+int((end-start)/2))
		# 		lower_results.append(start)
		# 		upper_results.append(end)
		else:
			i += 1


	if results[0] is not 0:
		results.insert(0, 0)
		lower_results.insert(0, 0)
		upper_results.insert(0, 0)

	if results[-1] is not len(black_ctr):
		results.append(len(black_ctr))
		lower_results.append(len(black_ctr))
		upper_results.append(len(black_ctr))
	print("pre processed results")
	print(lower_results)
	print(results)
	print(upper_results)

	#removing closeby boundaries
	new_results = []
	for i in range(len(results)-1):
		if(lower_results[i+1]-upper_results[i]>=3):
			new_results.append(results[i])
	if new_results[0] is not 0:
		new_results.insert(0, 0)
	if new_results[-1] is not len(black_ctr):
		new_results.append(len(black_ctr))

	
	a = np.size(inv.flatten())
	b = np.count_nonzero(inv.flatten())
	print("Total # of pixels ", a)
	print("Total # of black pixels", b)
	print("Percentage ", b*100/a)

	#combining segments with very less black pixels
	results = new_results
	new_results = []
	new_results.append(0)
	i = 0
	while i < (len(results)-1):
		patch = inv[0:h, results[i]:results[i+1]]
		blacks = np.count_nonzero(patch)
		total = np.size(patch.flatten())
		if blacks*100/total > 1 or results[i+1]-results[i]>20:
			print(i, i+1, blacks*100/total)
			print(0, h, results[i], results[i+1])
			new_results.append(results[i+1])
			i += 1
		else:
			if i+1 is not len(results)-1:
				results[i+1] = results[i]
			else:
				new_results[-1] = results[i+1] #when last segment doesnt have the character combine it with the previous segment
			i += 1

	print("results", results)

	# if new_results[-1] is not len(black_ctr):


	results = new_results
	print("split results", results)
	return results


def show_boundary_boxes(img, results, h, w):

	white_bg = 255*np.ones_like(img)
	print(results)
	for i in range(len(results)-1):
			# Get bounding box
		
		roi = img[0:h, results[i]:results[i+1]]
		#t = cv2.rectangle(inv, (0, 0), (40, h), (90, 90, 90), 3)

		cv2.rectangle(img, (results[i], 0), (results[i+1], h), (90, 90, 90), 3)        
		#cv2.imshow('{}.png'.format(i), roi)

		#--- paste ROIs on image with white background 
		white_bg[0:h, results[i]:results[i+1]] = roi

	cv2.imshow('white_bg_new', white_bg)
	cv2.waitKey(10000)
	return

def get_segments(file_location):
	h, w, img = readFile(file_location)
	inv = pre_process_image(img, False, True, True, True)
	black_ctr = vertical_projection(inv, h, w)
	results = get_segment_boundaries(black_ctr, inv, h, w)
	show_boundary_boxes(img, results, h, w)
	return results


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_file', type=str,
						help='Input image')
	args = parser.parse_args()
	get_segments(args.input_file)

# ../../data/sample_data/export1.png
# ../../data/sample_data/IAM/d01-056-06-05.png