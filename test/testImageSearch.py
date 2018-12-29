from utils import selectors
import pickle
from scipy.spatial.distance import pdist
from predictors import resnet152
import cv2
import matplotlib.pyplot as plt
import numpy as np
import operator

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

def pickle_load(f):
    try:
        data = pickle.load(f)
    except EOFError:
        data = None
    finally:
        return data

imgsDir = '../Data/ImageNet/ILSVRC2012/img_val/'
imgName = selectors.selectImg(imgsDir)
imgPath = imgsDir + imgName
img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)

plt.figure(imgName)
plt.imshow(img)
plt.axis('off')

img = getImgReady(img)

inputFeature, label = resnet152.predictionAndFeature(img)
featureFile = open('../Data/ImageNet/ILSVRC2012/val-wholeImage-features/{}'.format(label))

distances = {}
data = pickle_load(featureFile)
while data:
    inputFeature = np.array(inputFeature)
    dataFeature = np.array(data['feature'])
    distance = pdist(np.vstack([inputFeature,dataFeature]),'cosine')
    distances[data['imgPath']] = distance
    data = pickle_load(featureFile)

distances = sorted(distances.items(),key = operator.itemgetter(1))
for i in range(len(distances)-1,-1,-1):
    imgName = distances[i][0]
    img = cv2.cvtColor(cv2.imread(imgsDir + imgName), cv2.COLOR_BGR2RGB)
    plt.figure(imgName)
    plt.imshow(img)

plt.show()