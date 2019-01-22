import mxnet as mx
import numpy as np
from collections import namedtuple
# define a simple data batch
Batch = namedtuple('Batch', ['data'])

projectPath = '/home/hongyigeng/PycharmProjects/ImageSearch/'   #to be imported, all file paths in this script must be absolute path

sym, arg_params, aux_params = mx.model.load_checkpoint(projectPath + 'params/resnet101', 0)
mod = mx.mod.Module(symbol=sym, context=mx.gpu(), label_names=None)
#input data size must be 224x224
mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))],
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)
with open(projectPath + 'synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

def predict(img):
    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    probs = mod.get_outputs()[0].asnumpy()
    probs = np.squeeze(probs)
    a = np.argsort(probs)[::-1]
    index = a[0]
    prob = probs[index]
    label = labels[index]
    label = label.split(' ')[1]

    return prob, label

def predictionAndFeature(img):
    # list the last 10 layers
    all_layers = sym.get_internals()
    symbols = []
    symbols.append(all_layers['flatten0_output']) #An often used layer for feature extraction is the one before the last fully connected layer.
                                # For ResNet, and also Inception, it is the flattened layer with name flatten0 which reshapes the 4-D convolutional layer output into 2-D for the fully connected layer.
    symbols.append(all_layers['fc1_output'])
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
    label = labels[index].split(' ')[1]

    return feature, label