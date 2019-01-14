# For every image in imgsDirs, generate feature files using pickle, they are saved in Data/wholeImage-features/
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from utils import addImageToDB

count = 0

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

imgsDirs = [
    '../Data/ImageNet/ILSVRC2012/img_val/',
    '../Data/VOCdevkit/VOC2012/JPEGImages/',
    '../Data/userImages/'
    ]
for imgsDir in imgsDirs:
    imgNames = os.listdir(imgsDir)
    for imgName in imgNames:
        imgPath = imgsDir + imgName
        addImageToDB.addImageToDB(imgPath, 'resnet18')
        count = count + 1
        if count % 1000 == 0:
            print('{} images have been processed.'.format(count))

print('All processed images amount is {}'.format(count))