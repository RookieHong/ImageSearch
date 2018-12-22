#load a param in params/ and forward compute a image in ResizedObjects then visualize all conv layers
import mxnet as mx
import matplotlib.pyplot as plt
import os
import re
import cv2
import numpy as np
from collections import namedtuple
import math
import json
from utils import selectors

param, epoch = selectors.selectParam('../params/')
imgName = selectors.selectImg('../Data/ResizedObjects')

sym, arg_params, aux_params = mx.model.load_checkpoint('../params/{}'.format(param), epoch)
args = sym.get_internals().list_outputs()
internals = sym.get_internals()

convs = []  #gather all conv layers in convs
for key in args:
    if 'convolution' in key and 'output' in key:    #only visualize convolution output layer
        convs.append(internals[key])

if not convs:
    raise ValueError('no convolution layer to visualize!!')
group = mx.symbol.Group(convs)

mod = mx.mod.Module(symbol=group, context=mx.gpu())
mod.bind(for_training=False, data_shapes=[('data', (1, 3, 227, 227))])  #shape of Resized objects is (1, 3, 227, 227)
mod.set_params(arg_params, aux_params)

img = plt.imread('../Data/ResizedObjects/{}'.format(imgName))
plt.figure(imgName)
plt.imshow(img)
img = img.transpose(2, 0, 1)
img = (img.astype(np.float) - 128) * 0.00390625
img = img.reshape((1,) + img.shape)

Batch = namedtuple('Batch', 'data')
mod.forward(Batch([mx.nd.array(img)]))

for i in range(0, len(mod.get_outputs())):
    output = mod.get_outputs()[i].asnumpy()[0]
    nrows = int(math.floor(math.sqrt(len(output))))
    ncols = int(math.floor(len(output) / nrows))
    plt.figure(str(convs[i]))
    for j in range(0, nrows * ncols):
        plt.subplot(nrows, ncols, j + 1)
        plt.imshow(output[j])

plt.show()

#this part below simply outputs the prediction of the input img
output = mx.symbol.Group([internals['softmax_output']])
mod = mx.mod.Module(symbol=output, context=mx.gpu())
mod.bind(for_training=False, data_shapes=[('data', (1, 3, 227, 227))])  #shape of Resized objects is (1, 3, 227, 227)
mod.set_params(arg_params, aux_params)
mod.forward(Batch([mx.nd.array(img)]))
prob = mod.get_outputs()[0].asnumpy()
prob = np.squeeze(prob)
pred_label = np.argmax(prob)
with open('../Classes.json', 'r') as json_f:    #open json file that includes classes-label info
    classes = json.load(json_f)
    classes = dict(zip(classes.values(), classes.keys()))   # reverse json info to label-classes
print('Predicted class is {}'.format(classes[pred_label]))
