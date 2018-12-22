# coding=utf-8
import numpy as np
import mxnet as mx
import logging
import time
from importlib import import_module

netName = 'LeNet'   #the symbol file name in symbols directory
net = import_module('symbols.'+ netName)
sym = net.get_symbol(num_classes=20)
mod = mx.mod.Module(sym, context = mx.gpu())

usingMiniDataset = True
trainFile = "../Data/RecordIO/mini-train.rec" if usingMiniDataset else "../Data/RecordIO/train.rec"
valFile = "../Data/RecordIO/mini-val.rec" if usingMiniDataset else "../Data/RecordIO/val.rec"
logFileName = '../log/train_' + netName + '-miniDataset.log' if usingMiniDataset else '../log/train_' + netName + '.log'
checkpointName = '../params/miniPascalVOC_' + netName if usingMiniDataset else '../params/PascalVOC_' + netName
batch_size = 10 if usingMiniDataset else 64

#mx.viz.plot_network(net,title=netName,save_format='jpg',hide_weights=True).view()    #visulize network

train_iter = mx.io.ImageRecordIter(
    path_imgrec=trainFile,
    data_shape=(3, 227, 227),     # 3x227x227 is the required AlexNet input image size

    batch_size=batch_size,

    mean_r = 128,   #mean RGB
    mean_g = 128,
    mean_b = 128,

    scale = 0.00390625  #scale image pixels' value to [-0.5, 0.5]
)

# validation dataset iter
val_iter = mx.io.ImageRecordIter(
    path_imgrec=valFile,
    data_shape=(3, 227, 227),

    batch_size=batch_size,

    mean_r=128,  # mean RGB
    mean_g=128,
    mean_b=128,

    scale=0.00390625  # scale image pixel value to [-0.5, 0.5]
)

logging.getLogger().setLevel(logging.DEBUG)
fh = logging.FileHandler(logFileName)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logging.getLogger().addHandler(fh)
logging.getLogger().addHandler(ch)

lr_scheduler = mx.lr_scheduler.FactorScheduler(500, factor = 0.90)
optimizer_params = {
    'learning_rate': 0.001,
    'momentum': 0.9,
    'wd': 0.0005,   #weight decay
    'lr_scheduler': lr_scheduler
}
num_epoch = 40  #train epochs
checkpoint = mx.callback.do_checkpoint(checkpointName, period = num_epoch)

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