import mxnet as mx
import numpy as np
from collections import namedtuple
# define a simple data batch
Batch = namedtuple('Batch', ['data'])

sym, arg_params, aux_params = mx.model.load_checkpoint('../params/resnet152', 0)
mod = mx.mod.Module(symbol=sym, context=mx.gpu(), label_names=None)
#input data size must be 224x224
mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))],
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)
with open('../synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

def predict(img):
    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    probs = mod.get_outputs()[0].asnumpy()
    # print the top-5
    probs = np.squeeze(probs)
    a = np.argsort(probs)[::-1]
    index = a[0]
    prob = probs[index]
    label = labels[index]
    label = label.split(' ')[1]

    return prob, label