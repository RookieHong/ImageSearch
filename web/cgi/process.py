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
    from utils import nms, addImageToDB_wholeImage
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

    def pickle_load(f):
        try:
            data = pickle.load(f)
        except EOFError:
            data = None
        finally:
            return data

    def matchImages_wholeImage(imgPath):
        prob, label, inputFeature = predictor.predictionAndFeature(imgPath)
        featureFile = open(featuresDir + label)

        distances = {}
        data = pickle_load(featureFile)
        inputFeature = np.array(inputFeature)
        while data:
            dataFeature = np.array(data['feature'])
            distance = pdist(np.vstack([inputFeature, dataFeature]), 'cosine')
            distances[data['imgPath']] = float(distance)
            data = pickle_load(featureFile)
        featureFile.close()

        matchList = sorted(distances.items(), key=operator.itemgetter(1))

        return matchList

    def matchImages(inputFeature, label):
        featureFile = open(featuresDir + label)

        distances = {}
        data = pickle_load(featureFile)
        inputFeature = np.array(inputFeature)
        while data:
            dataFeature = np.array(data['feature'])
            distance = pdist(np.vstack([inputFeature, dataFeature]), 'cosine')
            distances[data['imgPath']] = float(distance)
            data = pickle_load(featureFile)
        featureFile.close()

        matchList = sorted(distances.items(), key=operator.itemgetter(1))

        return matchList

    def matchImages_objects(inputFeature, label):   #Match objects using database splited by kmeans
        centroidsFile = open(projectPath + 'Data/splited-objects-features-resnet101_fasterRcnn/centroids')

        centroids_distances = {}
        centroid = pickle_load(centroidsFile)
        inputFeature = np.array(inputFeature)
        while centroid:     #Find which centroid this obejct is allocated to
            centroidFileName = centroid.keys()[0]

            if label not in centroidFileName or os.path.getsize(projectPath + 'Data/splited-objects-features-resnet101_fasterRcnn/' + centroidFileName) == 0:    #No object allocated into this file
                centroid = pickle_load(centroidsFile)
                continue

            centroidFeature = np.array(centroid[centroidFileName])
            distance = pdist(np.vstack([inputFeature, centroidFeature]), 'cosine')
            centroids_distances[centroidFileName] = float(distance)
            centroid = pickle_load(centroidsFile)
        centroidsFile.close()

        allocatedTo = min(centroids_distances, key=centroids_distances.get)

        featureFile = open(projectPath + 'Data/splited-objects-features-resnet101_fasterRcnn/' + allocatedTo)

        distances = {}
        data = pickle_load(featureFile)
        while data:
            dataFeature = np.array(data['feature'])
            distance = pdist(np.vstack([inputFeature, dataFeature]), 'cosine')
            distances[data['imgPath']] = float(distance)
            data = pickle_load(featureFile)
        featureFile.close()

        matchList = sorted(distances.items(), key=operator.itemgetter(1))

        return matchList

    form = cgi.FieldStorage()
    fileitem = form['file']
    ext = form['ext'].file.read()
    ifWholeImage = True if form['searchType'].file.read() == 'wholeImage' else False
    ifAddImage = True if form['ifAddImage'].file.read() == 'true' else False

    if not fileitem.filename:
        raise Exception('image upload failed\n')

    savedImagePath = './input.{}'.format(ext)
    open(savedImagePath, 'wb').write(fileitem.file.read())

    if ifWholeImage:
        selectedPredictor = form['predictor'].file.read()  # Import the selected predictor module and set the features directory
        predictor = importlib.import_module('predictors.{}'.format(selectedPredictor))
        featuresDir = projectPath + 'Data/wholeImage-features-{}/'.format(selectedPredictor)

        if not ifAddImage:  # If ifAddImage is false, then this program should process the input image instead of adding it to database

            predictTime = time.time()
            prob, label, feature = predictor.predictionAndFeature(savedImagePath)
            predictTime = time.time() - predictTime
            toRet['classification'] = 'Class:{} confidence:{}'.format(label, prob)
            message = message + 'prediction time cost:{}s\n'.format(predictTime)

            searchTime = time.time()
            matchList = matchImages(feature, label)
            searchTime = time.time() - searchTime
            message = message + 'search time cost:{}s\n'.format(searchTime)

            toRet['matchList'] = matchList

        else:  # If ifAddImage is true, the uploaded image should be added to dataBase, this image is stored in Data/userImages/
            imgPath = '../Data/userImages/{}.{}'.format(time.asctime(), ext)
            open(imgPath, 'wb').write(fileitem.file.read())

            matchList = matchImages_wholeImage(imgPath)
            imageExist = False
            for match in matchList:  # Check if the uploaded image is already saved in database
                if match[1] < 0.0001:
                    imageExist = True
                    break
            if imageExist:
                os.remove(imgPath)
                toRet['status'] = 'error'
                message = message + 'This image is already saved in database!\n'
            else:
                label = addImageToDB_wholeImage.addImageToDB(imgPath, selectedPredictor)
                message = message + 'Image has been successfully added to {} database, it belongs to "{}"\n'.format(selectedPredictor, label)

    else:   #In this diverge, the image will be processed using RCNN series algorithms to detect objects in it and search images in objects level.
        selectedPredictor = form['predictor'].file.read() + '_fasterRcnn'  # Import the selected predictor module and set the features directory
        predictor = importlib.import_module('predictors.{}'.format(selectedPredictor))
        featuresDir = projectPath + 'Data/objects-features-{}/'.format(selectedPredictor)

        if not ifAddImage:
            img = cv2.cvtColor(cv2.imread(savedImagePath), cv2.COLOR_BGR2RGB)
            plt.figure('image')
            plt.imshow(img)
            plt.axis('off')

            predictTime = time.time()
            predictions, features = predictor.predictionAndFeature('./input.{}'.format(ext))
            predictTime = time.time() - predictTime

            objects = []
            matchList = []

            message = message + 'prediction time cost:{}s\n'.format(predictTime)

            searchTime = time.time()
            for i, [x1, y1, x2, y2, label, conf] in enumerate(predictions):
                color = (random.random(), random.random(), random.random())

                objects.append({
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'label': label,
                    'conf': conf,
                    'num': i
                })

                # matchList.append(matchImages(features[i], label))  # Search objects for every object
                matchList.append(matchImages_objects(features[i], label))   #Search objects for every object

                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,    #draw predictions
                                     fill=False, edgecolor=color, linewidth=3.5)
                plt.gca().add_patch(rect)
                plt.gca().text(x1, y1 - 2, '{:s} {:.3f}'.format(label, conf),
                               bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')

            searchTime = time.time() - searchTime
            message = message + 'search time cost:{}s\n'.format(searchTime)
            plt.savefig('output.jpg',bbox_inches='tight', pad_inches=0)

            toRet['matchList'] = matchList
            toRet['objects'] = objects

    toRet['message'] = message

except:
    toRet['status'] = 'error'
    message = message + traceback.format_exc()
    toRet['message'] = message
finally:
    json_toRet_str = json.dumps(toRet)
    print(json_toRet_str)