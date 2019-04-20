import mxnet as mx
import numpy as np
import importlib
import os
import pickle
from sklearn import preprocessing
import time
from scipy.spatial.distance import pdist
import operator
from utils.rmacRegions import rmac_regions

def matchImages(inputFeature, QEsize=0):
    distances = {}
    for key in features.keys():
        dataFeature = features[key]
        distance = pdist(np.vstack([inputFeature, dataFeature]), 'cosine')

        imgName = os.path.splitext(os.path.basename(key))[0]
        distances[imgName] = float(distance)

    matchList = sorted(distances.items(), key=operator.itemgetter(1))

    if QEsize > 0:  # Query Expansion
        for i in range(0, QEsize):
            inputFeature = inputFeature + features[matchList[i][0]]
        inputFeature = inputFeature / QEsize

        matchList = matchImages(inputFeature)

    return matchList

if __name__ == '__main__':

    logFile = open('../log/Test/test_Oxford-5k_mAP.log', 'a')
    logFile.write('\n{}:\n'.format(time.asctime()))

    ifcropped = False  # Use cropped query images or not
    logFile.write('Using cropped query images: {}\n'.format(ifcropped))

    QEsize = 0  # Use top k retrieved image features' mean to re-retrieve
    logFile.write('QE size = {}\n'.format(QEsize))

    predictorName = 'resnet18'
    dimension = 512
    print('{}: \n'.format(predictorName))
    predictor = importlib.import_module('predictors.{}'.format(predictorName))
    featuresFile = open('../Data/Oxford-5k/R-MAC/{}_{}_RMAC'.format(predictorName, dimension), 'rb')
    features = pickle.load(featuresFile)

    logFile.write('R-MAC, {}d, no resized\n'.format(dimension))

    PCAfile = open('../Data/PCA/{}_{}_PCA'.format(predictorName, dimension))
    pca = pickle.load(PCAfile)

    imgsDir = '../Data/Oxford-5k/cropped_query_images/' if ifcropped else '../Data/Oxford-5k/query_images/'

    query_images = os.listdir(imgsDir)
    aps = []
    gtObject_aps = {}
    for i, query_image in enumerate(query_images):
        query_name = os.path.splitext(query_image)[0]

        gtObject = query_name.split('_')
        gtObject = '_'.join(gtObject[0:len(gtObject) - 1])
        if not gtObject_aps.has_key(gtObject):
            gtObject_aps[gtObject] = []

        featureMap = predictor.getFeatureMap(imgsDir + query_image)[0]
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

        x = pca.transform(x)
        x = preprocessing.normalize(x, norm='l2', axis=1)
        x = np.sum(x, axis=0)
        x = preprocessing.normalize(x.reshape(1,-1), norm='l2', axis=1)[0]

        matchList = matchImages(x, QEsize=QEsize)

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

