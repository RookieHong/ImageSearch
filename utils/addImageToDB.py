# Add a image to Database so it can be searched
import numpy as np
import cv2
import pickle
import importlib

def getImgReady(img):
    if img is None:
         return None
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img

def addImageToDB(imgPath, selectedPredictor):
    DBdir = '../Data/wholeImage-features-{}/'.format(selectedPredictor)
    predictor = importlib.import_module('predictors.{}'.format(selectedPredictor))

    img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
    img = getImgReady(img)

    feature, label = predictor.predictionAndFeature(img)
    imgDict = {'imgPath': imgPath, 'feature': feature}

    featureFile = open(DBdir + label, 'ab+')
    pickle.dump(imgDict, featureFile)
    featureFile.close()

    return label