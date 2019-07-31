import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb
import argparse
from Utils import display, save, is_binary, printInfo, makeDir, percentageBlack
from Denoiser import Denoiser


class Segmenter:
    def __init__(self):
        self.numCharacters = 0

    def segment(self, img, w_min=100, w_max=2000, h_min=20, h_max=1000, extra_pixel=3, area_thres =0.3,write_to_dir=True):
        if type(img) == str:
            print('[INFO] Segmenting img: {}'.format(img))
            img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        inv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[-1]
        cv2.imshow('image', img)
        cv2.waitKey(500)
        cv2.imshow('image', inv)
        cv2.waitKey(500)
        # Convert to monochrome image as findContours requires 2-dim image
        if len(inv.shape) == 3:
            inv = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)

        if cv2.__version__[0] == '3':
            contours = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
        elif cv2.__version__[0] == '2' or cv2.__version__[0] == '4':
            contours = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        inv = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        cv2.imshow('image', inv)
        cv2.waitKey(5000)

        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])  # sort by x coordinate
        inp_org = inv
        rois = []
        for i, ctr in enumerate(contours):
            discard = 0
            # get bounding box
            x, y, w, h = cv2.boundingRect(ctr)
            if w_min < w < w_max and h_min < h < h_max:

                for j, temp_ctrs in enumerate(contours):
                    temp_x, temp_y, temp_w, temp_h = cv2.boundingRect(temp_ctrs)

                    if j != i and temp_w < w_max and temp_h < h_max:

                        # if there is overlap and intersection area is more than threshold then discard the bounding box
                        if (temp_x < x < temp_x + temp_w and temp_y < y < temp_y + temp_h) or (
                                temp_x < x + w < temp_x + temp_w and temp_y < y + h < temp_y + temp_h):
                            dx = min((temp_x + temp_w), (x + w)) - max(temp_x, x)
                            dy = min((temp_y + temp_h), (y + h)) - max(temp_y, y)
                            if (dx * dy > 0):
                                intersection_area = dx * dy
                            else:
                                intersection_area = 0
                            if (intersection_area / (w * h) > area_thres):
                                discard += 1
                                break

                # if not discarded then add bounding box in list
                if discard == 0:
                    roi = inp_org[y:y + h - extra_pixel, x:x + w]
                    # if roi_resize != None:
                    #     roi = cv2.resize(roi, roi_resize)
                    self.numCharacters += 1
                    rois.append(roi)
                    if write_to_dir:
                        save(roi, name=str(self.numCharacters), prefix='../out/', suffix='.png')

        print('[INFO] No. of Characters Found: {}'.format(self.numCharacters))
        return rois

    def show(self, inp, string=""):
        print(string)
        if inp.shape != 3:
            plt.imshow(inp, cmap='gray')
        else:
            plt.imshow(inp)
        plt.show()
        print()

    def multi_show(self, inp, string="", fig_size=(18, 18), row=10, col=20):
        print(string)
        fig = plt.figure(figsize=fig_size)
        for i, r in enumerate(inp):
            fig.add_subplot(row, col, i + 1)
            if r.shape != 3:
                plt.imshow(r, cmap='gray')
            else:
                plt.imshow(r)
        plt.show()
        print(len(inp))
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', help="path to image", default="../sample_imgs")
    parser.add_argument('--o', help="save segmented characters")
    args = parser.parse_args()
    segmenter = Segmenter()

    if os.path.isdir(args.i):
        imgFiles = [os.path.join(args.i, x) for x in os.listdir(args.i) if x[-4:] == '.png']
        [segmenter.segment(x, w_min=10, w_max=500, h_min=30, h_max=500, extra_pixel=10) for x in imgFiles]

    else:
        segmenter.segment(img=args.i, write_to_dir=args.o)


if __name__ == '__main__':
    main()