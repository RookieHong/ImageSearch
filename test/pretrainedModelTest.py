import mxnet as mx
import matplotlib.pyplot as plt
import cv2
import numpy as np
from collections import namedtuple
from utils import selectors
# define a simple data batch
Batch = namedtuple('Batch', ['data'])

sym, arg_params, aux_params = mx.model.load_checkpoint('../params/resnet152', 0)
mod = mx.mod.Module(symbol=sym, context=mx.gpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))],
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)
with open('../synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]
imgsDir = '../Data/ResizedObjects/'

def get_image(imgPath, show=False):
    # download and show the image
    #fname = mx.test_utils.download(url)
    img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
    if img is None:
         return None
    if show:
         plt.imshow(img)
         plt.axis('off')
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img

def predict(imgPath):
    img = get_image(imgPath, show=False)
    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    for i in a[0:5]:
        print('probability=%f, class=%s' %(prob[i], labels[i]))

img = imgsDir + selectors.selectImg(imgsDir)
predict(img)

plt.figure(img)
plt.imshow(plt.imread(img))
plt.show()