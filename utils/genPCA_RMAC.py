import mxnet as mx
import numpy as np
import importlib
import os
import pickle
from sklearn import preprocessing
from sklearn import decomposition
from rmacRegions import rmac_regions

if __name__ == '__main__':

    predictorName = 'resnet101'
    predictor = importlib.import_module('predictors.{}'.format(predictorName))
    dimension = 2048
    PCAfile = open('../Data/PCA/{}_{}_PCA'.format(predictorName, dimension), 'wb')

    imgsDirs = [
        '../Data/Paris-6k/all_images/',
        '../Data/Oxford-5k/oxbuild_images/'
    ]
    imgNames  = []
    for imgsDir in imgsDirs:
        images = os.listdir(imgsDir)
        for image in images:
            imgNames.append(imgsDir + image)
    all_features = []
    count = 0
    for i, imgName in enumerate(imgNames):
        featureMap = predictor.getFeatureMap(imgName)[0]
        featureMap = featureMap[np.newaxis, :]
        Wmap, Hmap = featureMap.shape[3], featureMap.shape[2]

        all_regions = []
        regions = rmac_regions(Wmap, Hmap, 3)
        for region in regions:
            x = region[0]
            y = region[1]
            w = region[2]
            h = region[3]

            x1 = x
            x2 = x + w - 1
            y1 = y
            y2 = y + h - 1

            all_regions.append([0, x1, y1, x2, y2])

        featureMap = mx.nd.array(featureMap)
        all_regions = mx.nd.array(all_regions)

        x = mx.nd.ROIPooling(data=featureMap, rois=all_regions, pooled_size=(1, 1), spatial_scale=1.0)

        x = np.squeeze(x.asnumpy())
        x = preprocessing.normalize(x, norm='l2', axis=1)

        if i == 0:
            all_features = x
        else:
            all_features = np.concatenate((all_features, x), axis=0)

        count += 1
        if count % 500 == 0:
            print('{} images have been processed.'.format(count))

     # PCA should be trained on whole dataset
    pca = decomposition.PCA(n_components=dimension, whiten=True)
    pca.fit(all_features)
    pickle.dump(pca, PCAfile)
    PCAfile.close()