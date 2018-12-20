import numpy as np
import os
import mxnet as mx
import logging
import cPickle
import time
from importlib import import_module

def unpickle(file):
    with open(file,'rb') as fo:
        dict = cPickle.load(fo)
    return np.array(dict['data']).reshape(10000,3072),np.array(dict['labels']).reshape(10000)


def to4d(img):
    return img.reshape(img.shape[0],3,32,32).astype(np.float32)/255


def fit(batch_num,model,val_iter,batch_size):
    (train_img, train_lbl) = unpickle('../Data/cifar-10-batches-py/data_batch_'+str(batch_num))
    train_iter = mx.io.NDArrayIter(to4d(train_img), train_lbl, batch_size, shuffle=True)
    model.fit(
        train_iter,
        eval_data=val_iter,
        optimizer_params=optimizer_params,
        num_epoch=num_epoch,
        batch_end_callback=mx.callback.Speedometer(batch_size,200),
        eval_metric=eval_metrics
    )


(val_img, val_lbl) = unpickle('../Data/cifar-10-batches-py/test_batch')

batch_size = 100
val_iter = mx.io.NDArrayIter(to4d(val_img),val_lbl,batch_size)

netName = 'vggNet'   #the symbol file name in symbols directory
net = import_module('symbols.'+ netName)
sym = net.get_symbol(num_classes=10)
mod = mx.mod.Module(sym, context = mx.gpu(0))

logFileName = '../log/train_{}-CIFAR10.log'.format(netName)

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
    'learning_rate': 0.01,
    'momentum': 0.9,
    'wd': 0.0005,   #weight decay
    'lr_scheduler': lr_scheduler
}
num_epoch = 20

#set eval_metrics
eval_metrics = mx.metric.CompositeEvalMetric()
metric1 = mx.metric.Accuracy()
metric2 = mx.metric.CrossEntropy()
metric3 = mx.metric.MSE()
for child_metric in [metric1, metric2, metric3]:
    eval_metrics.add(child_metric)

for batch_num in range(1,6):
    start = time.time()  # start time
    logging.debug('num_epoch={}'.format(num_epoch))
    fit(batch_num, mod, val_iter, batch_size)
    time_elapsed = time.time() - start
    print('total training time:{}s'.format(time_elapsed))