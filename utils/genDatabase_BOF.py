import cv2
import numpy as np
import os
from sklearn.externals import joblib
from scipy.cluster.vq import *
import gc

from sklearn import preprocessing

# Get the training classes names and store them in a list
train_path = '../Data/Oxford-5k/oxbuild_images/'

training_names = os.listdir(train_path)

numWords = 1000

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
for training_name in training_names:
    image_path = os.path.join(train_path, training_name)
    image_paths += [image_path]

# Create feature extraction and keypoint detector objects
nfeatures = 512
sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures)

# List where all the descriptors are stored
des_list = []
#des_list = joblib.load('../Data/Oxford-5k/BOF_des_list.pkl')

for i, image_path in enumerate(image_paths):
    img = cv2.imread(image_path)
    if i % 100 == 0:
        print("Have extracted {} images's SIFT, total {} images".format(i, len(image_paths)))
    kps, des = sift.detectAndCompute(img, None)
    des_list.append((image_path, des))

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
del des_list[0]
gc.collect()

while len(des_list) > 0:
    image_path = des_list[0][0]
    descriptor = des_list[0][1]

    del des_list[0]
    gc.collect()

    if descriptor is None:
        print('{} has none descriptor'.format(image_path))
        continue
    descriptors = np.vstack((descriptors, descriptor))

print('descriptors shape: {}'.format(descriptors.shape))

# Perform k-means clustering
print("Start k-means: %d words, %d key points" %(numWords, descriptors.shape[0]))
voc, variance = kmeans(descriptors, numWords, 1)

# Calculate the histogram of features
im_features = np.zeros((len(image_paths), numWords), "float32")
for i in xrange(len(image_paths)):
    if des_list[i][1] is None:
        continue
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum((im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Perform L2 normalization
im_features = im_features*idf
im_features = preprocessing.normalize(im_features, norm='l2')

joblib.dump((im_features, image_paths, idf, numWords, voc, nfeatures), "../Data/Oxford-5k/BOF/BOF_{}features.pkl".format(nfeatures), compress=3)