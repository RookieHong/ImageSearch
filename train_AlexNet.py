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
fc3 = mx.sym.FullyConnected(data = dropout2, num_hidden = 20)   #num_hidden应当等于分类的类别数，这里因为是使用的Pascal VOC2012数据集，所以为20
softmax = mx.sym.SoftmaxOutput(data = fc3, name = 'softmax')

AlexNet = softmax
mod = mx.mod.Module(AlexNet, context = mx.gpu(0))

#mx.viz.plot_network(AlexNet,title='AlexNet',save_format='jpg',hide_weights=True).view()    #画出网络结构

train_iter = mx.io.ImageRecordIter(
    path_imgrec="Data/RecordIO/mini-train.rec",  # rec文件路径
    data_shape=(3, 227, 227),     # 期望的数据形状，注意：
                                # 即使图片不是这个尺寸，也可以在此被自动转换
    batch_size=10,  # 每次传入100条数据

    mean_r = 128,   #三个通道的均值
    mean_g = 128,
    mean_b = 128,

    scale = 0.00390625  #减去均值后归一化到[-0.5, 0.5]之间
)

# 创建内部测试集iter
val_iter = mx.io.ImageRecordIter(
    path_imgrec="Data/RecordIO/mini-val.rec",
    data_shape=(3, 227, 227),
    batch_size=10,  # 必须与上面的batch_size相等，否则不能对应

    mean_r=128,  # 三个通道的均值
    mean_g=128,
    mean_b=128,

    scale=0.00390625  # 减去均值后归一化到[-0.5, 0.5]之间
)

logging.getLogger().setLevel(logging.DEBUG)
fh = logging.FileHandler('log/train_AlexNet-miniDataset.log')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logging.getLogger().addHandler(fh)
logging.getLogger().addHandler(ch)

lr_scheduler = mx.lr_scheduler.FactorScheduler(500, factor = 0.95)
optimizer_params = {
    'learning_rate': 0.01,
    'momentum': 0.9,
    'wd': 0.0005,
    'lr_scheduler': lr_scheduler
}
checkpoint = mx.callback.do_checkpoint('params/miniPascalVOC_AlexNet', period = 5)

#定义eval_metrics以在训练时输出相关信息
eval_metrics = mx.metric.CompositeEvalMetric()
metric1 = mx.metric.Accuracy()
metric2 = mx.metric.CrossEntropy()
metric3 = mx.metric.MSE()
for child_metric in [metric1, metric2, metric3]:
    eval_metrics.add(child_metric)

start = time.time() #设置起始时间以记录训练用时
mod.fit(
    train_iter,
    eval_data = val_iter,
    optimizer_params = optimizer_params,
    num_epoch = 36,
    epoch_end_callback = checkpoint,
    eval_metric = eval_metrics
)
time_elapsed = time.time() - start
print('训练总用时：{}s'.format(time_elapsed))