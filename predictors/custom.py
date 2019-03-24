import cv2
import numpy as np
import mxnet as mx
from collections import namedtuple
import json

def getImgReady(img):
    if img is None:
        return None
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img

projectPath = '/home/hongyigeng/PycharmProjects/ImageSearch/'   #to be imported, all file paths in this script must be absolute path

with open(projectPath + 'Classes.json', 'r') as json_f:    # open json file that includes classes-label info
    classes = json.load(json_f)
    classes = dict(zip(classes.values(), classes.keys()))   # reverse json info to label-classes

Batch = namedtuple('Batch', ['data'])
sym, arg_params, aux_params = mx.model.load_checkpoint(projectPath + 'params/PascalVOC_custom', 200)
mod = mx.mod.Module(symbol=sym, context=mx.gpu(), label_names=None)
#input data size must be 224x224
mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))],
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)

def predict(imgPath):
    img = cv2.imread(imgPath, cv2.COLOR_BGR2RGB)
    img = getImgReady(img)

    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)
    label = np.argmax(prob)

    return prob, label

def predictionAndFeature(imgPath):
    img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
    img = getImgReady(img)

    # list the last 10 layers
    all_layers = sym.get_internals()
    symbols = []
    symbols.append(all_layers['flatten0_output'])
    symbols.append(all_layers['fullyconnected2_output'])
    symbols = mx.symbol.Group(symbols)

    fe_mod = mx.mod.Module(symbol=symbols, context=mx.gpu(), label_names=None)
    fe_mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
    fe_mod.set_params(arg_params, aux_params)

    fe_mod.forward(Batch([mx.nd.array(img)]))

    feature = fe_mod.get_outputs()[0].asnumpy()
    probs = fe_mod.get_outputs()[1].asnumpy()

    probs = np.squeeze(probs)
    a = np.argsort(probs)[::-1]
    index = a[0]
    prob = probs[index]
    label = classes[index].encode('utf8')

    return prob, label, feature