# coding=utf-8
import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np
import json

train_iter = mx.io.ImageRecordIter(
    path_imgrec="../Data/RecordIO/mini-train.rec",  # rec文件路径
    #path_imgrec="Data/RecordIO/train.rec",  # rec文件路径
    data_shape=(3, 227, 227),     # 期望的数据形状，注意：
                                # 即使图片不是这个尺寸，也可以在此被自动转换
    batch_size=4,  # 每次传入4条数据

    mean_r = 128,   #三个通道的均值
    mean_g = 128,
    mean_b = 128,

    scale = 0.00390625  #减去均值后归一化到[-0.5, 0.5]之间
)

with open('../Classes.json', 'r') as json_f:    #载入记录类别和对应编号的json文件
    classes = json.load(json_f)
    classes = dict(zip(classes.values(), classes.keys()))   #把键值对颠倒方便输出

for batch in train_iter:
    data = batch.data[0]
    for i in range(4):
        label = int(batch.label[0].asnumpy()[i])
        plt.figure(classes[label])
        plt.imshow(data[i].asnumpy().transpose((1, 2, 0)))
        plt.show()