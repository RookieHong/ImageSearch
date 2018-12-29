#!/usr/bin/env python

import sys

sys.stderr = sys.stdout

print("Content-Type: text/html\n\n")

try:
    projectPath = '/home/hongyigeng/PycharmProjects/ImageSearch/'  # to be imported, all file paths in this script must be absolute path
    sys.path.append(projectPath)  # to import the modules defined by me, it's necessary to add project path as a sysPath

    toRet = {}  #The json to return to front page

    import traceback
    import cgi
    import cgitb;cgitb.enable()
    import cv2
    import numpy as np
    import selectivesearch
    from predictors import resnet152
    from utils import nms
    import random
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pickle
    import operator
    from scipy.spatial.distance import pdist
    import json

    def getImgReady(img):
        if img is None:
            return None
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

    def pickle_load(f):
        try:
            data = pickle.load(f)
        except EOFError:
            data = None
        finally:
            return data

    def matchImages(readyImg):   #Param readyImg means the param img must be the output of getImgReady()
        inputFeature, label = resnet152.predictionAndFeature(readyImg)
        featureFile = open(projectPath + 'Data/ImageNet/ILSVRC2012/val-wholeImage-features/{}'.format(label))

        distances = {}
        data = pickle_load(featureFile)
        while data:
            inputFeature = np.array(inputFeature)
            dataFeature = np.array(data['feature'])
            distance = pdist(np.vstack([inputFeature, dataFeature]), 'cosine')
            distances[data['imgPath']] = float(distance)
            data = pickle_load(featureFile)

        matchList = sorted(distances.items(), key=operator.itemgetter(1))

        return matchList

    form = cgi.FieldStorage()
    fileitem = form['file']
    ifSearch = True if form['ifSearch'].file.read() == 'true' else False
    ifWholeImage = True if form['ifWholeImage'].file.read() == 'true' else False
    ifBoundingBoxRegression = True if form['ifBoundingBoxRegression'].file.read() == 'true' else False

    boxes = {}

    if fileitem.filename:
        open('./input.jpg', 'wb').write(fileitem.file.read())

        img = cv2.cvtColor(cv2.imread('input.jpg'), cv2.COLOR_BGR2RGB)
        plt.figure('image')
        plt.imshow(img)
        plt.axis('off')

        if ifWholeImage:
            img = getImgReady(img)
            prob, label = resnet152.predict(img)
            rect = plt.Rectangle((0, 0), img.shape[0], img.shape[1],
                                 fill=False, edgecolor=(0, 1, 0), linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.gca().text(0, 0 - 2, '{:s} {:.3f}'.format(label, prob),
                           bbox=dict(facecolor=(0, 1, 0), alpha=0.5), fontsize=12, color='white')

            if ifSearch:
                matchList = matchImages(img)

                toRet['matchList'] = matchList

        else:
            img_label, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=500)
            for i, region in enumerate(regions):  # rect:x y w h
                x = region['rect'][0]
                y = region['rect'][1]
                w = region['rect'][2]
                h = region['rect'][3]

                croppedImg = img[y:y + h, x:x + w]
                croppedImg = getImgReady(croppedImg)
                prob, label = resnet152.predict(croppedImg)

                if prob < 0.2:  # ignore low probability boxes
                    continue

                addBox(x, y, w, h, prob, label)

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

        plt.savefig('output.jpg',bbox_inches='tight', pad_inches=0)
        message = 'image upload successful and processed in output.jpg'

    else:
        message = 'image upload failed'

    toRet['message'] = message
    json_toRet_str = json.dumps(toRet)
    print(json_toRet_str)

except:
    traceback.print_exc()
