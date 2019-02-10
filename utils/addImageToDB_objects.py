# Add a image to Database so it can be searched(based on objects)
import importlib
import pickle
import numpy as np
import os
from scipy.spatial.distance import pdist

def pickle_load(f):
    try:
        data = pickle.load(f)
    except EOFError:
        data = None
    finally:
        return data

def addImageToDB(imgPath, selectedPredictor, predictions = None, features = None):
    predictor = importlib.import_module('predictors.{}'.format(selectedPredictor))

    if not predictions and not features:    #For quickly adding image from web page, provide inputting predictions and features to avoid calculating them again
        predictions, features = predictor.predictionAndFeature(imgPath)
    for i, [x1, y1, x2, y2, label, conf] in enumerate(predictions):
        centroidsFile = open('../Data/splited-objects-features-{}/centroids'.format(selectedPredictor))

        imgDict = {'imgPath': imgPath,
                   'feature': features[i],
                   'x1': x1,
                   'y1': y1,
                   'x2': x2,
                   'y2': y2,
                   'conf': conf}

        centroids_distances = {}
        centroid = pickle_load(centroidsFile)
        inputFeature = np.array(features[i])
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

        featureFile = open('../Data/splited-objects-features-{}/{}'.format(selectedPredictor, allocatedTo), 'ab+')
        pickle.dump(imgDict, featureFile)
        featureFile.close()

    return predictions