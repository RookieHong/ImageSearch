#!/usr/bin/env python

import sys

sys.stderr = sys.stdout

print("Content-Type: text/html\n\n")

projectPath = '/home/hongyigeng/PycharmProjects/ImageSearch/'  # to be imported, every file path in this script must be absolute path
sys.path.append(projectPath)  # to import the modules defined by me, it's necessary to add project path as a sysPath

toRet = {}  #The json to return to front page
toRet['status'] = 'success' #set this to 'error' and front page will print message in danger mode
message = ''

try:
    import traceback
    import cgi
    import cgitb;cgitb.enable()
    import cv2
    import numpy as np
    import selectivesearch
    from utils import nms, addImageToDB
    import random
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pickle
    import operator
    from scipy.spatial.distance import pdist
    import json
    import time
    import os
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
        inputFeature, label = predictor.predictionAndFeature(readyImg)
        featureFile = open(featuresDir + label)

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

    def rcnn(img):
        predictTime = time.time()
        img_label, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=500)
        for i, region in enumerate(regions):  # rect:x y w h
            x = region['rect'][0]
            y = region['rect'][1]
            w = region['rect'][2]
            h = region['rect'][3]

            croppedImg = img[y:y + h, x:x + w]
            croppedImg = getImgReady(croppedImg)
            prob, label = predictor.predict(croppedImg)

            if prob < 0.6:  # ignore low probability boxes
                continue

            addBox(x, y, w, h, prob, label)

        predictTime = time.time() - predictTime

        return predictTime

    def fasterRcnn():
        from predictors import resnet101_fasterRcnn
        global boxes
        predictTime = time.time()
        boxes = resnet101_fasterRcnn.predict('./input.{}'.format(ext))
        predictTime = time.time() - predictTime

        return predictTime


    form = cgi.FieldStorage()
    fileitem = form['file']
    ext = form['ext'].file.read()
    ifAddImage = True if form['ifAddImage'].file.read() == 'true' else False    #ifAddImage diverges the program

    if not ifAddImage:      #If ifAddImage is false, then this program should process the input image instead of adding it to database
        ifWholeImage = True if form['searchType'].file.read() == 'wholeImage' else False

        selectedPredictor = form['predictor'].file.read()  # Import the selected predictor module and set the features directory
        predictor = importlib.import_module('predictors.{}'.format(selectedPredictor))
        featuresDir = projectPath + 'Data/wholeImage-features-{}/'.format(selectedPredictor)

        boxes = {}

        if fileitem.filename:
            savedImagePath = './input.{}'.format(ext)
            open(savedImagePath, 'wb').write(fileitem.file.read())

            img = cv2.cvtColor(cv2.imread(savedImagePath), cv2.COLOR_BGR2RGB)
            plt.figure('image')
            plt.imshow(img)
            plt.axis('off')

            if ifWholeImage:
                img = getImgReady(img)
                predictTime = time.time()
                prob, label = predictor.predict(img)
                predictTime = time.time() - predictTime
                message = message + 'prediction time cost:{}s\n'.format(predictTime)

                rect = plt.Rectangle((0, 0), img.shape[0], img.shape[1],        #draw prediction
                                     fill=False, edgecolor=(0, 1, 0), linewidth=3.5)
                plt.gca().add_patch(rect)
                plt.gca().text(0, 0 - 2, '{:s} {:.3f}'.format(label, prob),
                               bbox=dict(facecolor=(0, 1, 0), alpha=0.5), fontsize=12, color='white')

                searchTime = time.time()
                matchList = matchImages(img)
                searchTime = time.time() - searchTime
                message = message + 'search time cost:{}s\n'.format(searchTime)

                toRet['matchList'] = matchList

            else:   #In this diverge, image will be cropped into many bounding boxes using selective search and every box will be predicted using predictor
                algorithm = form['algorithm'].file.read()
                predictTime = rcnn(img) if algorithm == 'rcnn' else fasterRcnn()

                message = message + 'prediction time cost:{}s\n'.format(predictTime)
                for label in boxes:
                    color = (random.random(), random.random(), random.random())
                    indexes = nms.nms(np.array(boxes[label]), 0.3)  #Nms threshold is 0.3
                    for i in indexes:
                        x1 = boxes[label][i][0]
                        y1 = boxes[label][i][1]
                        x2 = boxes[label][i][2]
                        y2 = boxes[label][i][3]
                        prob = boxes[label][i][4]

                        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,    #draw predictions
                                             fill=False, edgecolor=color, linewidth=3.5)
                        plt.gca().add_patch(rect)
                        plt.gca().text(x1, y1 - 2, '{:s} {:.3f}'.format(label, prob),
                                       bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')

            plt.savefig('output.jpg',bbox_inches='tight', pad_inches=0)

        else:
            message = 'image upload failed\n'

    else:   #If ifAddImage is true, the uploaded image should be added to dataBase, this image is stored in Data/userImages/
        if fileitem.filename:
            imgPath = '../Data/userImages/{}.{}'.format(time.asctime(), ext)
            open(imgPath, 'wb').write(fileitem.file.read())

            img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
            img = getImgReady(img)
            matchList = matchImages(img)
            imageExist = False
            for match in matchList: #Check if the uploaded image is already saved in database
                if match[1] < 0.0001:
                    imageExist = True
                    break
            if imageExist:
                os.remove(imgPath)
                toRet['status'] = 'error'
                message = message + 'This image is already saved in database!\n'
            else:
                label = addImageToDB.addImageToDB(imgPath, selectedPredictor)
                message = message + 'Image has been successfully added to {} database, it belongs to "{}"\n'.format(selectedPredictor, label)

    toRet['message'] = message

except:
    toRet['status'] = 'error'
    message = message + traceback.format_exc()
    toRet['message'] = message
finally:
    json_toRet_str = json.dumps(toRet)
    print(json_toRet_str)