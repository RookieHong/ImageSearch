#Calculate search time
from scipy.spatial.distance import pdist
import os
import pickle
import numpy as np
import operator
import time
import matplotlib.pyplot as plt

def pickle_load(f):
    try:
        data = pickle.load(f)
    except EOFError:
        data = None
    finally:
        return data

predictorName = 'resnet18'
featuresDir = '../Data/wholeImage-features-{}/'.format(predictorName)
logFile = open('../log/Test/test_{}-wholeImage.log'.format(predictorName), 'a')
featureFileNames = os.listdir(featuresDir)

logFile.write('Assessment {}\n'.format(time.asctime()))
totalCount = 0

searchTimes = []
counts = []

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
    featureFile.close()

    matchList = sorted(distances.items(), key=operator.itemgetter(1))
    searchTime = time.time() - searchTime
    searchTimes.append(searchTime)

    logFile.write('Class:{:30s}\t\t\t\tCount:{}\t\t\t\tTime:{:5f}\t\t\t\tTime/Count:{}\n'.format(featureFileName, count, searchTime, searchTime/count))
    totalCount = totalCount + count
    counts.append(count)

logFile.write('total images:{}\n\n'.format(totalCount))
logFile.close()

plt.figure("{} search time statistics".format(predictorName))
plt.bar(range(len(searchTimes)), searchTimes)
plt.xlabel("Categories")
plt.ylabel("Search time(s)")
plt.title("{} search time statistics".format(predictorName))

plt.figure("{} quantity statistics".format(predictorName))
plt.bar(range(len(counts)), counts)
plt.xlabel("Categories")
plt.ylabel("Quantity")
plt.title("{} quantity statistics".format(predictorName))

plt.show()