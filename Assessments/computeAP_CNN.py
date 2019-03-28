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

def matchImages(inputFeature, featuresDir):
    featureFiles = os.listdir(featuresDir)
    distances = {}

    for featureFile in featureFiles:
        featureFile = open(featuresDir + featureFile)
        data = pickle_load(featureFile)
        while data:
            dataFeature = np.array(data['feature'])
            distance = pdist(np.vstack([inputFeature, dataFeature]), 'cosine')
            imgName = os.path.splitext(os.path.basename(data['imgPath']))[0]
            distances[imgName] = float(distance)
            data = pickle_load(featureFile)
        featureFile.close()

    matchList = sorted(distances.items(), key=operator.itemgetter(1))

    return matchList

predictorNames = ['resnet18', 'resnet101', 'resnet152']
logFile = open('../log/Test/test_Oxford-5k_mAP.log', 'a')
logFile.write('\n{}:\n'.format(time.asctime()))
ifcropped = False    # Use cropped query images or not
logFile.write('Using cropped query images: {}\n'.format(ifcropped))

for predictorName in predictorNames:
    print('{}: \n'.format(predictorName))
    featuresDir = '../Data/Oxford-5k/{}/'.format(predictorName)
    predictor = importlib.import_module('predictors.{}'.format(predictorName))

    query_images = os.listdir('../Data/Oxford-5k/cropped_query_images/') if ifcropped else os.listdir('../Data/Oxford-5k/query_images/')
    aps = []
    for query_image in query_images:
        query_name = os.path.splitext(os.path.basename(query_image))[0]
        prob, label, inputFeature = predictor.predictionAndFeature('../Data/Oxford-5k/cropped_query_images/' + query_image) if ifcropped else predictor.predictionAndFeature('../Data/Oxford-5k/query_images/' + query_image)
        matchList = matchImages(inputFeature, featuresDir)

        rankFilePath = '../Data/Oxford-5k/temp.txt'
        rankFile = open(rankFilePath, 'w')
        rankFile.writelines(match[0] + '\n' for match in matchList)
        rankFile.close()

        gt_file = '../Data/Oxford-5k/gt_files/' + query_name
        cmd = '../Data/Oxford-5k/gt_files/compute_ap %s %s' % (gt_file, rankFilePath)
        ap = os.popen(cmd).read()
        os.remove(rankFilePath)
        aps.append(float(ap.strip()))
        print("{}, {}".format(query_name, ap.strip()))
    logFile.write('{} mAP: {}\n'.format(predictorName, np.mean(aps)))

logFile.close()
