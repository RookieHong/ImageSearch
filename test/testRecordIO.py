# coding=utf-8
import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np
import json

train_iter = mx.io.ImageRecordIter(
    path_imgrec="../Data/RecordIO/mini-train.rec",  # rec file path
    #path_imgrec="Data/RecordIO/train.rec",  # # rec file path
    data_shape=(3, 227, 227),     # 3x227x227 is the required AlexNet input image size

    batch_size=4,  # batch size=4

    mean_r = 128,   #mean RGB
    mean_g = 128,
    mean_b = 128,

    scale = 0.00390625  #scale image pixels' value to [-0.5, 0.5]
)

with open('../Classes.json', 'r') as json_f:    #open json file that includes classes-label info
    classes = json.load(json_f)
    labels = dict(zip(classes.values(), classes.keys()))   # reverse json info to label-classes

for batch in train_iter:
    data = batch.data[0]
    data = data + 0.5   #scale data to [0, 1]
    for i in range(4):
        label = int(batch.label[0].asnumpy()[i])
        plt.figure(labels[label])
        plt.imshow(data[i].asnumpy().transpose((1, 2, 0)))
        plt.show()