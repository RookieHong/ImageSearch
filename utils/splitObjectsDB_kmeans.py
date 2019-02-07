import numpy as np
from scipy.spatial.distance import pdist
import os
import pickle

def pickle_load(f):
    try:
        data = pickle.load(f)
    except EOFError:
        data = None
    finally:
        return data

def randomCentroids(data, k):
    n = np.shape(data)[1]   #n is the dimension of data
    centroids = np.zeros((k, n))
    for j in range(n):
        min = np.min(data[:, j])
        max = np.max(data[:, j])
        range_j = max - min
        for i in range(k):
            centroids[i][j] = min + range_j * np.random.rand()
    return centroids

def setCentroids(dataAllocation, data, centroids):
    k = np.shape(centroids)[0]
    for centroidNum in range(k):
        indexes = [i for i in range(np.shape(dataAllocation)[0]) if dataAllocation[i][1] == centroidNum]
        centroids[centroidNum] = np.mean(data[indexes], axis = 0)
    return centroids

def kmeans(data, k):
    m = np.shape(data)[0]
    dataAllocation = np.zeros((m, 2))   #First column: distance, second column: centroid
    centroids = randomCentroids(data, k)    #centroids is a k x n matrix
    epoch = 0
    notConverged = True
    while notConverged:
        if epoch >= 100:    #Max iteration times is 100
            break
        notConverged = False
        epoch = epoch + 1
        for i in range(m):
            dataFeature = data[i]
            minDistance = 2     #Cosine distance implemented in pdist is reduced by 1, so its between [0, 2] and the smaller, the closer
            minIndex = -1
            for j in range(k):
                centroidFeature = centroids[j]
                distance = pdist(np.vstack([dataFeature, centroidFeature]), 'cosine')
                if distance < minDistance:
                    minDistance = distance
                    minIndex = j
            if dataAllocation[i][1] != minIndex:
                notConverged = True
            dataAllocation[i][0] = minDistance
            dataAllocation[i][1] = minIndex
        print('Epoch %d:' % epoch)
        #for centroidNum in range(k):
        #    pointsCount = len([i for i in range(np.shape(dataAllocation)[0]) if dataAllocation[i][1] == centroidNum])
        #    print('Centroid %d has %d points' % (centroidNum, pointsCount))
        if notConverged:
            centroids = setCentroids(dataAllocation, data, centroids)
    return dataAllocation, centroids

targetDBDir = '../Data/objects-features-resnet101_fasterRcnn/'
newDBDir = '../Data/splited-objects-features-resnet101_fasterRcnn/'
featureFileNames = os.listdir(targetDBDir)

for featureFileName in featureFileNames:
    print('Now processing %s:' % featureFileName)
    featureFile = open(targetDBDir + featureFileName)
    features = []
    allData = []
    data = pickle_load(featureFile)
    count = 0
    while data:
        count = count + 1
        features.append(data['feature'])
        allData.append(data)
        data = pickle_load(featureFile)
    featureFile.close()
    features = np.array(features)

    centroidAmount = count / 400    #Expect about 300 images allocated to a centroid
    print('Count: {} Centroids: {}'.format(count, centroidAmount))
    dataAllocation, centroids = kmeans(features, centroidAmount)

    centroidFile = open(newDBDir + 'centroids', 'ab')   #centroidFile records every centroid feature in it
    for i in range(centroidAmount):
        pickle.dump({'{}_{}'.format(featureFileName, i): centroids[i]}, centroidFile)
    centroidFile.close()

    files = [(open('{}_{}'.format(newDBDir + featureFileName, i), 'wb')) for i in range(centroidAmount)]    #Save image dicts into every file
    for i in range(count):
        allocatedTo = int(dataAllocation[i][1])
        pickle.dump(allData[i], files[allocatedTo])

    for i in range(len(files)):
        files[i].close()