import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb
import argparse
from Utils import display, save, is_binary, printInfo, makeDir, percentageBlack
from Denoiser import Denoiser

class Segmenter:
    def segment(self, imgpath, minArea= 10, minHeightWidthRatio=1.1, write_to_dir=False):
        print('[INFO] Segmenting img: {}'.format(imgpath))
        img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        inv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[-1]
        if cv2.__version__[0] == '3':
            contours = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
        elif cv2.__version__[0] == '2' or cv2.__version__[0] == '4':
            contours = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
        
        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0]) # sort by x coordinate

        numCharacters = 0
        rois = []
        for ctr in contours:
            x, y, w, h = cv2.boundingRect(ctr)
            roi = img[y:y+h, x:x+w]
            if w*h > minArea:
                numCharacters += 1
                rois.append(roi)
                if write_to_dir:
                    save(roi, name= str(numCharacters), prefix='../out/' + imgpath[:-4].split('/')[-1], suffix='.png')

        #save(img, name='original', prefix='../out/' + imgpath[:-4].split('/')[-1], suffix='.png')
        print('[INFO] No. of Characters Found: {}'.format(numCharacters))
        return rois

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', help="path to image", default="../sample_imgs")
    parser.add_argument('--o', help="save segmented characters", action='store_true')
    args = parser.parse_args()
    segmenter = Segmenter()

    if os.path.isdir(args.i):
        imgFiles = [os.path.join(args.i, x) for x in os.listdir(args.i) if x[-4:]=='.png']
        [segmenter.segment(x) for x in imgFiles]
    
    else:
        segmenter.segment(imgpath=args.i, write_to_dir=args.o)

if __name__ == '__main__':
    main()