# coding=utf-8
import xml.etree.ElementTree as ET
from PIL import Image
import os
import json
import numpy as np
import matplotlib.pyplot as plt

imgs_path = '../Data/VOCdevkit/VOC2012/JPEGImages/'
output_path = '../Data/ResizedObjects/'
filenames = os.listdir(imgs_path)

count = np.zeros(20)
count.dtype = 'int'

with open('../Classes.json', 'r') as json_f:    #open json file that includes classes-label info
    classes = json.load(json_f)
    labels = dict(zip(classes.values(), classes.keys()))  # reverse json info to label-classes

for i, filename in enumerate(filenames):
    filepath = os.sep.join([imgs_path, filename])
    img = Image.open(filepath)
    xmlTree = ET.parse('../Data/VOCdevkit/VOC2012/Annotations/' + filename.split('.')[0] + '.xml')  # reads corresponding XML file

    for object in xmlTree.findall('object'):
        name = object.find('name').text
        label = classes[name]

        bndbox = object.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))     #reads coordinates
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))

        height = ymax - ymin
        width = xmax - xmin
        if height < 32 or width < 32:   #skip boxes that are too small
            continue

        cropped = img.crop((xmin, ymin, xmax, ymax))
        cropped = cropped.resize((224, 224), Image.ANTIALIAS)

        #plt.figure(str(height) + 'x' + str(width) + ' ' + name)
        #plt.imshow(cropped)
        #plt.show()

        save_path = output_path + name + '-' + str(count[label]) + '_' + str(label) + '.jpg'
        cropped.save(save_path)
        count[label] = count[label] + 1

outputCount_f = open('../Data/' + 'objectCount.txt', 'w')
for i in range(0, 20):
    outputCount_f.write('{} has {} images\n'.format(labels[i], count[i]))
outputCount_f.close()