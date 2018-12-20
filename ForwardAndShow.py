# coding=utf-8
import sys
import os
import cv2
import numpy as np
import mxnet as mx
from collections import namedtuple
import json

with open('Classes.json', 'r') as json_f:    ##open json file that includes classes-label info
    classes = json.load(json_f)
    classes = dict(zip(classes.values(), classes.keys()))   ## reverse json info to label-classes

Batch = namedtuple('Batch', ['data'])
input_path = 'Data/ResizedObjects'
mod = mx.mod.Module.load('params/PascalVOC_AlexNet', 35, context=mx.gpu(0))
mod.bind(
    data_shapes=[('data', (1, 3, 227, 227))],
    for_training=False
)
filenames = os.listdir(input_path)

for i, filename in enumerate(filenames):
    filepath = os.sep.join([input_path, filename])
    img = cv2.imread(filepath, cv2.COLOR_BGR2RGB)

    img = (img.astype(np.float) - 128) * 0.00390625
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]

    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)
    pred_label = np.argmax(prob)

    #img = cv2.imread(filepath)
    #cv2.imshow('Recognition result:{}'.format(i + 1, pred_label), img)
    #cv2.waitKey(100)
    print('Predicted class for {} is {}'.format(filename, classes[pred_label]))