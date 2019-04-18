import os
import pickle
import numpy as np
from scipy.spatial.distance import pdist
import operator
import importlib
import time

def pickle_load(f):
    try:
        data = pickle.load(f)
    except EOFError:
        data = None
    finally:
        return data

def matchImages(inputFeature, featuresDir, QEsize=0):
    featureFiles = os.listdir(featuresDir)
    distances = {}
    features = {}

    for featureFile in featureFiles:
        featureFile = open(featuresDir + featureFile)
        data = pickle_load(featureFile)
        while data:
            dataFeature = np.array(data['feature'])
            distance = pdist(np.vstack([inputFeature, dataFeature]), 'cosine')

            imgName = os.path.splitext(os.path.basename(data['imgPath']))[0]
            distances[imgName] = float(distance)
            features[imgName] = dataFeature

            data = pickle_load(featureFile)
        featureFile.close()

    matchList = sorted(distances.items(), key=operator.itemgetter(1))

    if QEsize > 0:  # Query Expansion
        for i in range(0, QEsize):
            inputFeature = inputFeature + features[matchList[i][0]]
        inputFeature = inputFeature / QEsize

        matchList = matchImages(inputFeature, featuresDir)

    return matchList

predictorNames = ['custom']
logFile = open('../log/Test/test_Oxford-5k_mAP.log', 'a')
logFile.write('\n{}:\n'.format(time.asctime()))

ifcropped = False    # Use cropped query images or not
logFile.write('Using cropped query images: {}\n'.format(ifcropped))

QEsize = 0  # Use top k retrieved image features' mean to re-retrieve
logFile.write('QE size = {}\n'.format(QEsize))

for predictorName in predictorNames:
    print('{}: \n'.format(predictorName))
    featuresDir = '../Data/Oxford-5k/{}/'.format(predictorName)
    predictor = importlib.import_module('predictors.{}'.format(predictorName))

    query_images = os.listdir('../Data/Oxford-5k/resized_cropped_query_images/') if ifcropped else os.listdir('../Data/Oxford-5k/resized_query_images/')
    aps = []
    gtObject_aps = {}
    for query_image in query_images:
        query_name = os.path.splitext(query_image)[0]

        gtObject = query_name.split('_')
        gtObject = '_'.join(gtObject[0:len(gtObject) - 1])
        if not gtObject_aps.has_key(gtObject):
            gtObject_aps[gtObject] = []

        prob, label, inputFeature = predictor.predictionAndFeature('../Data/Oxford-5k/resized_cropped_query_images/' + query_image) if ifcropped else predictor.predictionAndFeature('../Data/Oxford-5k/resized_query_images/' + query_image)
        matchList = matchImages(inputFeature, featuresDir, QEsize=QEsize)

        rankFilePath = '/home/hongyigeng/PycharmProjects/ImageSearch/Data/Oxford-5k/temp.txt'
        rankFile = open(rankFilePath, 'w')
        rankFile.writelines(match[0] + '\n' for match in matchList)
        rankFile.close()

        gt_file = '/home/hongyigeng/PycharmProjects/ImageSearch/Data/Oxford-5k/gt_files/' + query_name
        cmd = '../Data/Oxford-5k/compute_ap %s %s' % (gt_file, rankFilePath)
        ap = float(os.popen(cmd).read().strip())
        os.remove(rankFilePath)
        aps.append(ap)
        gtObject_aps[gtObject].append(ap)
        print("{}, {}".format(query_name, ap))
    for key in gtObject_aps.keys():
        logFile.write('\t{}:{}\n'.format(key, np.mean(gtObject_aps[key])))

    logFile.write('{} mAP: {}\n'.format(predictorName, np.mean(aps)))

logFile.close()
