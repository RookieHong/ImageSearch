# coding=utf-8
import mxnet as mx
import os
import re
from utils import selectors

param, epoch = selectors.selectParam('../params/')

test_iter = mx.io.ImageRecordIter(
    path_imgrec = '../Data/RecordIO/val.rec',
    data_shape = (3, 224, 224),

    batch_size=50,  # batch size=50

    mean_r = 128,   #mean RGB
    mean_g = 128,
    mean_b = 128,

    scale = 0.00390625  #scale the pixels to [-0.5, 0.5]
)

mod = mx.mod.Module.load(param, epoch, context = mx.gpu())
mod.bind(
    data_shapes = test_iter.provide_data,
    label_shapes = test_iter.provide_label,
    for_training = False
)
metric = mx.metric.create('acc')
mod.score(test_iter, metric)
for name, val in metric.get_name_value():
    print('{}={:4f}%'.format(name, val * 100))