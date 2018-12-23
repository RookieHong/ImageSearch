import cv2
import matplotlib.pyplot as plt
import numpy as np
import selectivesearch
from predictors import resnet152
from utils import selectors, nms
import xml.etree.ElementTree as ET
import random

boxes = {}

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

def addBox(x, y, w, h, prob, label):
    x1 = x
    x2 = x + w
    y1 = y
    y2 = y + h
    if not boxes.has_key(label):
        boxes[label] = [[x1, y1, x2, y2, prob]]
    else:
        boxes[label].append([x1, y1, x2, y2, prob])

def drawGroundTruth(imgName, img):
    xmlTree = ET.parse('../Data/VOCdevkit/VOC2012/Annotations/{}.xml'.format(imgName.split('.')[0]))  # reads corresponding XML file
    for object in xmlTree.findall('object'):
        name = object.find('name').text

        bndbox = object.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))     #reads coordinates
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))

        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=False, edgecolor=(0, 1, 0), linewidth=3.5)
        plt.gca().add_patch(rect)
        plt.gca().text(xmin, ymin - 2, 'Ground Truth:{:s}'.format(name),
                       bbox=dict(facecolor=(0, 1, 0), alpha=0.5), fontsize=12, color='white')

imgsDir = '../Data/VOCdevkit/VOC2012/JPEGImages/'
imgName = selectors.selectImg(imgsDir)
imgPath = imgsDir + imgName
img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)

plt.figure(imgName.split('.')[0])
plt.imshow(img)

img_label, regions = selectivesearch.selective_search(img, scale = 500, sigma = 0.9, min_size = 500)
for i, region in enumerate(regions):    #rect:x y w h
    x = region['rect'][0]
    y = region['rect'][1]
    w = region['rect'][2]
    h = region['rect'][3]

    croppedImg = img[y:y + h,x:x + w]
    croppedImg = getImgReady(croppedImg)
    prob, label = resnet152.predict(croppedImg)

    if prob < 0.2:  #ignore low probability boxes
        continue

    addBox(x, y, w, h, prob, label)

    # plt.figure('{}-{}'.format(prob, label))
    # plt.imshow(img[y:y + h,x:x + w])
    # plt.show()

for label in boxes:
    color = (random.random(), random.random(), random.random())
    indexes = nms.nms(np.array(boxes[label]), 0.3)
    for i in indexes:
        x1 = boxes[label][i][0]
        y1 = boxes[label][i][1]
        x2 = boxes[label][i][2]
        y2 = boxes[label][i][3]
        prob = boxes[label][i][4]

        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             fill=False, edgecolor=color, linewidth=3.5)
        plt.gca().add_patch(rect)
        plt.gca().text(x1, y1 - 2, '{:s} {:.3f}'.format(label, prob),
                       bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')

drawGroundTruth(imgName, img)

plt.show()