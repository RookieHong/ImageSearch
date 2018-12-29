# For every image in Data/ImageNet/ILSVRC2012/img-val/, generate feature files using pickle, they are saved in Data/ImageNet/ILSVRC2012/val-features/
from predictors import resnet152
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import os

def getImgReady(img, show=False):
    if img is None:
         return None
    if show:
         plt.imshow(img)
         plt.axis('off')
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img

imgsDir = '../Data/ImageNet/ILSVRC2012/img_val/'
imgNames = os.listdir(imgsDir)
for imgName in imgNames:
    imgPath = imgsDir + imgName
    img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
    img = getImgReady(img)

    feature, label = resnet152.predictionAndFeature(img)
    imgDict = {'imgPath': imgName, 'feature': feature}

    featureFile = open('../Data/ImageNet/ILSVRC2012/val-wholeImage-features/{}'.format(label), 'ab+')
    pickle.dump(imgDict, featureFile)
    featureFile.close()