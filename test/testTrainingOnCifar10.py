import numpy as np
import os
import mxnet as mx
import logging
import cPickle
import time

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

# data = mx.sym.Variable('data')
# cv1 = mx.sym.Convolution(data=data,name='cv1',num_filter=32,kernel=(3,3))
# act1 = mx.sym.Activation(data=cv1,name='relu1',act_type='relu')
# poing1 = mx.sym.Pooling(data=act1,name='poing1',kernel=(2,2),pool_type='max')
# do1 = mx.sym.Dropout(data=poing1,name='do1',p=0.25)
# cv2 = mx.sym.Convolution(data=do1,name='cv2',num_filter=32,kernel=(3,3))
# act2 = mx.sym.Activation(data=cv2,name='relu2',act_type='relu')
# poing2 = mx.sym.Pooling(data=act2,name='poing2',kernel=(2,2),pool_type='avg')
# do2 = mx.sym.Dropout(data=poing2,name='do2',p=0.25)
# cv3 = mx.sym.Convolution(data=do2,name='cv3',num_filter=64,kernel=(3,3))
# act3 = mx.sym.Activation(data=cv3,name='relu3',act_type='relu')
# poing3 = mx.sym.Pooling(data=act3,name='poing3',kernel=(2,2),pool_type='avg')
# do3 = mx.sym.Dropout(data=poing3,name='do3',p=0.25)
# data = mx.sym.Flatten(data=do3)
# fc1 = mx.sym.FullyConnected(data=data,name='fc1',num_hidden=64)
# act4 = mx.sym.Activation(data=fc1,name='relu4',act_type='relu')
# do4 = mx.sym.Dropout(data=act4,name='do4',p=0.25)
# fc2 = mx.sym.FullyConnected(data=do4,name='fc2',num_hidden=10)
# mlp = mx.sym.SoftmaxOutput(data=fc2,name='softmax')

data = mx.symbol.Variable('data')

conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",kernel=(2,2), stride=(2,2))
# second conv
conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",kernel=(2,2), stride=(2,2))
# first fullc
flatten = mx.symbol.Flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
# loss
lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')

mod = mx.mod.Module(lenet, context = mx.gpu(0))

logging.getLogger().setLevel(logging.DEBUG)
fh = logging.FileHandler('../log/train-CIFAR10.log')
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
num_epoch = 10

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