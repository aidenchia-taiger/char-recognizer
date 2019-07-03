import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb

def display(img):
    img = np.array(img)
    plt.rcParams["figure.figsize"] = [9, 6]
    plt.imshow(img,'gray')
    plt.show()

def save(img, name, prefix=None, suffix=None):
    if prefix:
        makeDir(prefix)
        name = os.path.join(prefix, name)
    if suffix:
        name += suffix

    print('[INFO] Img saved as {}'.format(name))
    cv2.imwrite(name, img)

def is_binary(img):
    return np.array_equal(np.unique(img), np.array([0, 255])) or np.array_equal(np.unique(img), np.array([0, 1]))

def printInfo(img):
    print('Width x Height: {} x {}'.format(img.shape[1], img.shape[0]))
    print('Aspect Ratio: {}'.format(float(img.shape[1] / img.shape[0])))
    print('Binarized: {}'.format(is_binary(img)))

def makeDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def percentageBlack(img):
    numWhite = np.count_nonzero(img)
    return 1 - numWhite * 100 / img.size