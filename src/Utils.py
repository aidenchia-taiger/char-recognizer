import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb

def display(img, multiple=False,rows=3,cols=3):
    plt.rcParams["figure.figsize"] = [9, 6]
    if multiple: # img is a list of imgs
        fig=plt.figure(figsize=(8, 8))
        for i in range(1, cols * rows + 1):
            fig.add_subplot(rows, cols, i)
            image = img[i]
            if image.ndim == 3:
                image = np.squeeze(image)
            plt.imshow(image)
        plt.show()

    else:
        img = np.array(img)
        if img.ndim == 4:
            img = np.squeeze(img, axis=0)
            img = np.squeeze(img, axis=-1)
        elif img.ndim == 3:
            img = np.squeeze(img)
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
    if type(img) == str:
        img = cv2.imread(img)
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


