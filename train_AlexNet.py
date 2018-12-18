# coding=utf-8
import numpy as np
import mxnet as mx
import os
import cv2
from matplotlib import pyplot as plt
import logging
import time

data = mx.sym.Variable('data')

#stage 1
conv1 = mx.sym.Convolution(data = data, kernel = (11, 11), stride = (4, 4), num_filter = 96)
relu1 = mx.sym.Activation(data = conv1, act_type = 'relu')
lrn1 = mx.sym.LRN(data = relu1, alpha = 0.0001, beta = 0.75, knorm = 2, nsize = 5)
pool1 = mx.sym.Pooling(data = lrn1, kernel = (3, 3), stride = (2, 2), pool_type = 'max')

#stage 2
conv2 = mx.sym.Convolution(data = pool1, kernel = (5, 5), pad = (2, 2), num_filter = 256)
relu2 = mx.sym.Activation(data = conv2, act_type = 'relu')
lrn2 = mx.sym.LRN(data = relu2, alpha = 0.0001, beta = 0.75, knorm = 2, nsize = 5)
pool2 = mx.sym.Pooling(data = lrn2, kernel = (3, 3), stride = (2, 2), pool_type = 'max')

#stage 3
conv3 = mx.sym.Convolution(data = pool2, kernel = (3, 3), pad = (1, 1), num_filter = 384)
relu3 = mx.sym.Activation(data = conv3, act_type = 'relu')
conv4 = mx.sym.Convolution(data = relu3, kernel = (3, 3), pad = (1, 1), num_filter = 384)
relu4 = mx.sym.Activation(data = conv4, act_type = 'relu')
conv5 = mx.sym.Convolution(data = relu4, kernel = (3, 3), pad = (1, 1), num_filter = 256)
relu5 = mx.sym.Activation(data = conv5, act_type = 'relu')
pool3 = mx.sym.Pooling(data = relu5, kernel = (3, 3), stride = (2, 2), pool_type = 'max')

#stage 4
flatten = mx.sym.Flatten(data = pool3)
fc1 = mx.sym.FullyConnected(data = flatten, num_hidden = 4096)
relu6 = mx.sym.Activation(data = fc1, act_type = 'relu')
dropout1 = mx.sym.Dropout(data = relu6, p = 0.5)

#stage 5
fc2 = mx.sym.FullyConnected(data = dropout1, num_hidden = 4096)
relu7 = mx.sym.Activation(data = fc2, act_type = 'relu')
dropout2 = mx.sym.Dropout(data = relu7, p = 0.5)

#stage 6
fc3 = mx.sym.FullyConnected(data = dropout2, num_hidden = 20)   #num_hidden should equal Pascal VOC2012 data's classes=20
softmax = mx.sym.SoftmaxOutput(data = fc3, name = 'softmax')

AlexNet = softmax
mod = mx.mod.Module(AlexNet, context = mx.gpu(0))

#mx.viz.plot_network(AlexNet,title='AlexNet',save_format='jpg',hide_weights=True).view()    #visulize network

train_iter = mx.io.ImageRecordIter(
    path_imgrec="Data/RecordIO/mini-train.rec",  # rec file path
    #path_imgrec="Data/RecordIO/train.rec",  # rec file path
    data_shape=(3, 227, 227),     # 3x227x227 is the required AlexNet input image size

    batch_size=10,  # batch size=10 for mini dataset
    #batch_size=64,  # batch size=64 for large dataset

    mean_r = 128,   #mean RGB
    mean_g = 128,
    mean_b = 128,

    scale = 0.00390625  #scale image pixels' value to [-0.5, 0.5]
)

# validation dataset iter
val_iter = mx.io.ImageRecordIter(
    path_imgrec="Data/RecordIO/mini-val.rec",
    #path_imgrec="Data/RecordIO/val.rec",
    data_shape=(3, 227, 227),

    batch_size=10,  # batch size=10 for mini dataset
    #batch_size=64,  # batch size=64 for large dataset

    mean_r=128,  # mean RGB
    mean_g=128,
    mean_b=128,

    scale=0.00390625  # scale image pixel value to [-0.5, 0.5]
)

logging.getLogger().setLevel(logging.DEBUG)
#fh = logging.FileHandler('log/train_AlexNet.log')
fh = logging.FileHandler('log/train_AlexNet-miniDataset.log')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logging.getLogger().addHandler(fh)
logging.getLogger().addHandler(ch)

lr_scheduler = mx.lr_scheduler.FactorScheduler(500, factor = 0.90)
optimizer_params = {
    'learning_rate': 0.01,
    'momentum': 0.9,
    'wd': 0.0005,   #weight decay
    'lr_scheduler': lr_scheduler
}
num_epoch = 40  #train epochs
checkpoint = mx.callback.do_checkpoint('params/miniPascalVOC_AlexNet', period = 10)
#checkpoint = mx.callback.do_checkpoint('params/PascalVOC_AlexNet', period = 5)

#set eval_metrics
eval_metrics = mx.metric.CompositeEvalMetric()
metric1 = mx.metric.Accuracy()
metric2 = mx.metric.CrossEntropy()
metric3 = mx.metric.MSE()
for child_metric in [metric1, metric2, metric3]:
    eval_metrics.add(child_metric)

logging.debug('num_epoch={}'.format(num_epoch))

start = time.time() #start time
mod.fit(
    train_iter,
    eval_data = val_iter,
    optimizer_params = optimizer_params,
    num_epoch = num_epoch,
    epoch_end_callback = checkpoint,
    eval_metric = eval_metrics
)
time_elapsed = time.time() - start
print('total training time:{}s'.format(time_elapsed))