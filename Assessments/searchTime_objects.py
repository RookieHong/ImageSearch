#Calculate search time
from scipy.spatial.distance import pdist
import os
import pickle
import numpy as np
import operator
import time

def pickle_load(f):
    try:
        data = pickle.load(f)
    except EOFError:
        data = None
    finally:
        return data

predictorName = 'resnet101_fasterRcnn'
featuresDir = '../Data/objects-features-{}/'.format(predictorName)
logFile = open('../log/test_{}-objects.log'.format(predictorName), 'a')
featureFileNames = os.listdir(featuresDir)

logFile.write('Assessment {}\n'.format(time.asctime()))
totalCount = 0

for featureFileName in featureFileNames:
    searchTime = time.time()
    featureFile = open(featuresDir + featureFileName)

    distances = {}
    data = pickle_load(featureFile)
    inputFeature = np.array(data['feature'])
    count = 0
    while data:
        count = count + 1
        dataFeature = np.array(data['feature'])
        distance = pdist(np.vstack([inputFeature, dataFeature]), 'cosine')
        distances[data['imgPath']] = float(distance)
        data = pickle_load(featureFile)

    matchList = sorted(distances.items(), key=operator.itemgetter(1))
    searchTime = time.time() - searchTime

    logFile.write('Class:{:30s}\t\t\t\tCount:{}\t\t\t\tTime:{:5f}\t\t\t\tTime/Count:{}\n'.format(featureFileName, count, searchTime, searchTime/count))
    totalCount = totalCount + count

logFile.write('total objects:{}\n'.format(totalCount))
logFile.close()