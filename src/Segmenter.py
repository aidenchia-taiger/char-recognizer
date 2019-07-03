import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb
import argparse
from Utils import display, save, is_binary, printInfo, makeDir, percentageBlack
from Denoiser import Denoiser

class Segmenter:
    def segment(self, imgpath, minArea= 50, minHeightWidthRatio=1.1):
        print('[INFO] Segmenting img: {}'.format(imgpath))
        img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        inv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[-1]
        contours = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0]) # sort by x coordinate
        
        numCharacters = 0
        white = np.zeros(img.shape)
        for ctr in contours:
            x, y, w, h = cv2.boundingRect(ctr)
            roi = img[y:y+h, x:x+w]
            numCharacters += 1
            save(roi, name= str(numCharacters), prefix='../out/' + imgpath[:-4].split('/')[-1], suffix='.png')
            cv2.rectangle(img, (x,y), (x+w, y+h), 0, 1)

        save(img, name='original', prefix='../out/' + imgpath[:-4].split('/')[-1], suffix='.png')

        print('[INFO] No. of Characters Found: {}'.format(numCharacters))
        return numCharacters

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', help="path to folder of images", default="../imgs")
    args = parser.parse_args()

    makeDir('../out')
    imgFiles = [os.path.join(args.i, x) for x in os.listdir(args.i) if x[-4:]=='.png']

    segmenter = Segmenter()
    [segmenter.segment(x) for x in imgFiles]

if __name__ == '__main__':
    main()