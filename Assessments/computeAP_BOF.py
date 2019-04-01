import cv2
import numpy as np
from sklearn.externals import joblib
import os
import time
from scipy.cluster.vq import *
from sklearn import preprocessing

logFile = open('../log/Test/test_Oxford-5k_mAP.log', 'a')
logFile.write('\n{}:\n'.format(time.asctime()))
ifcropped = False    # Use cropped query images or not
logFile.write('Using cropped query images: {}\n'.format(ifcropped))

query_images = os.listdir('../Data/Oxford-5k/cropped_query_images/') if ifcropped else os.listdir('../Data/Oxford-5k/query_images/')
im_features, image_paths, idf, numWords, voc, nfeatures = joblib.load('../Data/Oxford-5k/BOF/BOF_256features.pkl')
sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures)
aps = []
for query_image in query_images:
    query_name = os.path.splitext(os.path.basename(query_image))[0]
    img = cv2.imread('../Data/Oxford-5k/cropped_query_images/' + query_image) if ifcropped else cv2.imread('../Data/Oxford-5k/query_images/' + query_image)
    kps, des = sift.detectAndCompute(img, None)
    inputFeature = np.zeros((1, numWords), "float32")
    words, distance = vq(des, voc)
    for w in words:
        inputFeature[0][w] += 1

    # Perform L2 normalization
    inputFeature = inputFeature * idf
    inputFeature = preprocessing.normalize(inputFeature, norm='l2')

    score = np.dot(inputFeature, im_features.T)
    rank_ID = np.argsort(-score)
    matchList = []
    for i in range(len(rank_ID[0])):
        imgName = os.path.splitext(os.path.basename(image_paths[rank_ID[0][i]]))[0]
        matchList.append(imgName)

    rankFilePath = '../Data/Oxford-5k/temp.txt'
    rankFile = open(rankFilePath, 'w')
    rankFile.writelines(match + '\n' for match in matchList)
    rankFile.close()

    gt_file = '../Data/Oxford-5k/gt_files/' + query_name
    cmd = '../Data/Oxford-5k/gt_files/compute_ap %s %s' % (gt_file, rankFilePath)
    ap = os.popen(cmd).read()
    os.remove(rankFilePath)
    aps.append(float(ap.strip()))
    print("{}, {}".format(query_name, ap.strip()))

logFile.write('BOF_{}features mAP: {}\n'.format(nfeatures, np.mean(aps)))
logFile.close()
