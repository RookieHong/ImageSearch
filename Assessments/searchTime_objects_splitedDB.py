import numpy as np
import pickle
import time
from scipy.spatial.distance import pdist
import random
import os
import operator

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

logFile = open('../log/test_resnet101_fasterRcnn-objects_splitedDB.log', 'a')

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

logFile.write('Random class: {}\t\tAllocated to: {}\t\tFinding centroid cost time: {}s\n'.format(label, allocatedTo, findCentroidTime))

searchTime = time.time()
objectsCount = 0

featureFile = open('../Data/splited-objects-features-resnet101_fasterRcnn/' + allocatedTo)

distances = {}
data = pickle_load(featureFile)
while data:
    objectsCount = objectsCount + 1
    dataFeature = np.array(data['feature'])
    distance = pdist(np.vstack([inputFeature, dataFeature]), 'cosine')
    distances[data['imgPath']] = float(distance)
    data = pickle_load(featureFile)
featureFile.close()

matchList = sorted(distances.items(), key=operator.itemgetter(1))

searchTime = time.time() - searchTime

logFile.write('{} has {} objects\t\tSearch cost time: {}s\t\tTime/Count: {}\n'.format(allocatedTo, objectsCount, searchTime, searchTime / objectsCount))
logFile.write('Totally cost time: {}s\n\n'.format(findCentroidTime + searchTime))
logFile.close()