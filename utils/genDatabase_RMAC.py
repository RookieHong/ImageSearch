import mxnet as mx
import numpy as np
import importlib
import os
import pickle
from sklearn import preprocessing
from rmacRegions import rmac_regions

if __name__ == '__main__':

    featureDim = {
        'vgg16': 512,
        'resnet18': 512,
        'resnet101': 512,
        'resnet152': 512,
        'custom': 512
    }

    predictorName = 'resnet18'
    predictor = importlib.import_module('predictors.{}'.format(predictorName))
    featuresFile = open('../Data/Oxford-5k/R-MAC/{}_{}_RMAC'.format(predictorName, featureDim[predictorName]), 'wb')

    PCAfile = open('../Data/PCA/{}_{}_PCA'.format(predictorName, featureDim[predictorName]))
    pca = pickle.load(PCAfile)

    imgsDir = '../Data/Oxford-5k/oxbuild_images/'
    imgNames = os.listdir(imgsDir)
    features = {}
    count = 0
    for i, imgName in enumerate(imgNames):
        featureMap = predictor.getFeatureMap(imgsDir + imgName)[0]
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

        x = mx.nd.ROIPooling(data=featureMap, rois=all_regions, pooled_size=(1,1), spatial_scale=1.0)

        x = np.squeeze(x.asnumpy())
        x = preprocessing.normalize(x, norm='l2', axis=1)

        features[imgName] = x

        count += 1
        if count % 500 == 0:
            print('{} images have been processed'.format(count))

    for imgName in features.keys():
        x = pca.transform(features[imgName])
        x = preprocessing.normalize(x, norm='l2', axis=1)
        x = np.sum(x, axis=0)
        x = preprocessing.normalize(x.reshape(1,-1), norm='l2', axis=1)[0]
        features[imgName] = x

    pickle.dump(features, featuresFile)
    featuresFile.close()