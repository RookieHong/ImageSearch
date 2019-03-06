import pickle
import numpy as np
import os
from PIL import Image
import json
import matplotlib.pyplot as plt

def pickle_load(f):
    try:
        data = pickle.load(f)
    except EOFError:
        data = None
    finally:
        return data

DBDir = '../Data/objects-features-resnet101_fasterRcnn/'
output_path = '../Data/ResizedObjects/'
featureFileNames = os.listdir(DBDir)

count = np.zeros(20)
count.dtype = 'int'

with open('../Classes.json', 'r') as json_f:    #open json file that includes classes-label info
    classes = json.load(json_f)
    labels = dict(zip(classes.values(), classes.keys()))  # reverse json info to label-classes

for featureFileName in featureFileNames:
    print('Now processing %s:' % featureFileName)
    featureFile = open(DBDir + featureFileName)
    data = pickle_load(featureFile)
    label = classes[featureFileName]    # number of class
    while data:
        img = Image.open(data['imgPath'])

        x1 = int(data['x1'])
        x2 = int(data['x2'])
        y1 = int(data['y1'])
        y2 = int(data['y2'])

        height = y2 - y1
        width = x2 - x1
        if height < 32 or width < 32:
            data = pickle_load(featureFile)
            continue

        cropped = img.crop((x1, y1, x2, y2))
        cropped = cropped.resize((224, 224), Image.ANTIALIAS)

        # plt.figure(str(height) + 'x' + str(width) + ' ' + featureFileName)
        # plt.imshow(cropped)
        # plt.show()

        save_path = output_path + featureFileName + '-' + str(count[label]) + '_' + str(label) + '.jpg'
        cropped.save(save_path)
        count[label] = count[label] + 1
        if count[label] % 1000 == 0:
            print('{} has processed {} objects'.format(featureFileName, count[label]))

        data = pickle_load(featureFile)

outputCount_f = open('../Data/' + 'objectCount.txt', 'w')
for i in range(0, 20):
    outputCount_f.write('{} has {} images\n'.format(labels[i], count[i]))
outputCount_f.close()