import cv2
import matplotlib.pyplot as plt
import numpy as np
import selectivesearch
from predictors import resnet152
from utils import selectors, nms

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

imgsDir = '../Data/VOCdevkit/VOC2012/JPEGImages/'
imgPath = imgsDir + selectors.selectImg(imgsDir)
img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)

img_label, regions = selectivesearch.selective_search(img, scale = 500, sigma = 0.9, min_size = 1024)
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
    indexes = nms.nms(np.array(boxes[label]), 0.3)
    for i in indexes:
        x1 = boxes[label][i][0]
        y1 = boxes[label][i][1]
        x2 = boxes[label][i][2]
        y2 = boxes[label][i][3]
        prob = boxes[label][i][4]

        #draw rectangle
        # parameters are image, left-top (x,y), right-bottom(x,y), color(RGB), thickness
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

        #put text
        font = cv2.FONT_HERSHEY_SIMPLEX
        # parameters are image, text, position, font, size, color(RGB), thickness
        cv2.putText(img, label + ' ' + ('%.2f' % prob), (x1, y1), font, 0.3, (255, 255, 255), 1)

plt.figure(imgPath)
plt.imshow(img)
plt.show()