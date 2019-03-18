import numpy as np
import pickle
import time
from scipy.spatial.distance import pdist
import random
import os
import operator
import matplotlib.pyplot as plt

def pickle_load(f):
    try:
        data = pickle.load(f)
    except EOFError:
        data = None
    finally:
        return data

class_names = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

logFile = open('../log/Test/test_resnet101_fasterRcnn-objects_splitedDB.log', 'a')

logFile.write('Assessment {}\n'.format(time.asctime()))
totalCount = 0

inputFeature = np.random.rand(2048)

centroidsFile = open('../Data/splited-objects-features-resnet101_fasterRcnn/centroids')

findCentroidTime = time.time()

centroid = pickle_load(centroidsFile)
centroids_distances = {}
label = random.sample(class_names, 1)[0]   #Randomly choose a class_name as label

while centroid:  # Find which centroid this obejct is allocated to
    centroidFileName = centroid.keys()[0]

    if label not in centroidFileName or os.path.getsize(
            '../Data/splited-objects-features-resnet101_fasterRcnn/' + centroidFileName) == 0:  # No object allocated into this file
        centroid = pickle_load(centroidsFile)
        continue

    centroidFeature = np.array(centroid[centroidFileName])
    distance = pdist(np.vstack([inputFeature, centroidFeature]), 'cosine')
    centroids_distances[centroidFileName] = float(distance)
    centroid = pickle_load(centroidsFile)
centroidsFile.close()

allocatedTo = min(centroids_distances, key=centroids_distances.get)

findCentroidTime = time.time() - findCentroidTime

logFile.write('Random class: {}\t\tAllocated to: {}\t\tFinding centroid costs time: {}s\n'.format(label, allocatedTo, findCentroidTime))

featureFileNames = os.listdir('../Data/splited-objects-features-resnet101_fasterRcnn/')
searchTimes = []
counts = []
totalCount = 0

for featureFileName in featureFileNames:
    if os.path.getsize('../Data/splited-objects-features-resnet101_fasterRcnn/' + featureFileName) == 0 or featureFileName == 'centroids':    # No object allocated into this file or this is centroids file
        continue
    searchTime = time.time()
    featureFile = open('../Data/splited-objects-features-resnet101_fasterRcnn/' + featureFileName)

    distances = {}
    data = pickle_load(featureFile)
    inputFeature = np.array(data['feature'])
    objectsCount = 0
    while data:
        objectsCount = objectsCount + 1
        dataFeature = np.array(data['feature'])
        distance = pdist(np.vstack([inputFeature, dataFeature]), 'cosine')
        distances[data['imgPath']] = float(distance)
        data = pickle_load(featureFile)
    featureFile.close()

    matchList = sorted(distances.items(), key=operator.itemgetter(1))
    searchTime = time.time() - searchTime + findCentroidTime    # add findCentroidTime to each search time
    searchTimes.append(searchTime)

    logFile.write('Class:{:30s}\t\t\t\tCount:{}\t\t\t\tTime:{:5f}\t\t\t\tTime/Count:{}\n'.format(featureFileName, objectsCount,searchTime,searchTime / objectsCount))
    totalCount = totalCount + objectsCount
    counts.append(objectsCount)

logFile.write('total objects:{}\n\n'.format(totalCount))
logFile.close()

plt.figure("splitted-objects-features-resnet101_fasterRcnn search time statistics")
plt.bar(range(len(searchTimes)), searchTimes)
plt.xlabel("Categories")
plt.ylabel("Search time(s)")
plt.title("splitted-objects-features-resnet101_fasterRcnn search time statistics")

plt.figure("splitted-objects-features-resnet101_fasterRcnn quantity statistics")
plt.bar(range(len(counts)), counts)
plt.xlabel("Categories")
plt.ylabel("Quantity")
plt.title("splitted-objects-features-resnet101_fasterRcnn quantity statistics")

plt.show()