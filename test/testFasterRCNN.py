#There is only pretrained model on VOC
import mxnet as mx
import cv2
import numpy as np
from utils import selectors
from symbols import resnet
from utils.image import imdecode, resize, transform, get_image, tensor_vstack
from utils.bbox import im_detect
import matplotlib.pyplot as plt
import random

params = ''
imgsDir = '../Data/VOCdevkit/VOC2012/JPEGImages/'

def infer_param_shape(symbol, data_shapes):
    arg_shape, _, aux_shape = symbol.infer_shape(**dict(data_shapes))
    arg_shape_dict = dict(zip(symbol.list_arguments(), arg_shape))
    aux_shape_dict = dict(zip(symbol.list_auxiliary_states(), aux_shape))
    return arg_shape_dict, aux_shape_dict

def infer_data_shape(symbol, data_shapes):
    _, out_shape, _ = symbol.infer_shape(**dict(data_shapes))
    data_shape_dict = dict(data_shapes)
    out_shape_dict = dict(zip(symbol.list_outputs(), out_shape))
    return data_shape_dict, out_shape_dict

def check_shape(symbol, data_shapes, arg_params, aux_params):
    arg_shape_dict, aux_shape_dict = infer_param_shape(symbol, data_shapes)
    data_shape_dict, out_shape_dict = infer_data_shape(symbol, data_shapes)
    for k in symbol.list_arguments():
        if k in data_shape_dict or 'label' in k:
            continue
        assert k in arg_params, '%s not initialized' % k
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for %s inferred %s provided %s' % (k, arg_shape_dict[k], arg_params[k].shape)
    for k in symbol.list_auxiliary_states():
        assert k in aux_params, '%s not initialized' % k
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for %s inferred %s provided %s' % (k, aux_shape_dict[k], aux_params[k].shape)

def load_test(filename, short, max_size, mean, std):
    # read and transform image
    im_orig = imdecode(filename)
    im, im_scale = resize(im_orig, short, max_size)
    height, width = im.shape[:2]
    im_info = mx.nd.array([height, width, im_scale])

    # transform into tensor and normalize
    im_tensor = transform(im, mean, std)

    # for 1-batch inference purpose, cannot use batchify (or nd.stack) to expand dims
    im_tensor = mx.nd.array(im_tensor).expand_dims(0)
    im_info = mx.nd.array(im_info).expand_dims(0)

    # transform cv2 BRG image to RGB for matplotlib
    im_orig = im_orig[:, :, (2, 1, 0)]
    return im_tensor, im_info, im_orig

def generate_batch(im_tensor, im_info):
    """return batch"""
    data = [im_tensor, im_info]
    data_shapes = [('data', im_tensor.shape), ('im_info', im_info.shape)]
    data_batch = mx.io.DataBatch(data=data, label=None, provide_data=data_shapes, provide_label=None)
    return data_batch

def load_param(params, ctx=None):
    """same as mx.model.load_checkpoint, but do not load symnet and will convert context"""
    if ctx is None:
        ctx = mx.cpu()
    save_dict = mx.nd.load(params)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v.as_in_context(ctx)
        if tp == 'aux':
            aux_params[name] = v.as_in_context(ctx)
    return arg_params, aux_params

def get_resnet101():
    global params
    params = '../params/resnet_voc0712-0010.params'
    return resnet.get_resnet(anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2),
                           rpn_feature_stride=16, rpn_pre_topk=6000,
                           rpn_post_topk=300, rpn_nms_thresh=0.7,
                           rpn_min_size=16,
                           num_classes=21, rcnn_feature_stride=16,
                           rcnn_pooled_size=(14, 14), rcnn_batch_size=1,
                           units=(3, 4, 23, 3), filter_list=(256, 512, 1024, 2048))

def vis_detection(im_orig, detections, class_names, thresh=0.7):
    """visualize [cls, conf, x1, y1, x2, y2]"""
    plt.imshow(im_orig)
    colors = [(random.random(), random.random(), random.random()) for _ in class_names]
    for [cls, conf, x1, y1, x2, y2] in detections:
        cls = int(cls)
        if cls > 0 and conf > thresh:
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 fill=False, edgecolor=colors[cls], linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.gca().text(x1, y1 - 2, '{:s} {:.3f}'.format(class_names[cls], conf),
                           bbox=dict(facecolor=colors[cls], alpha=0.5), fontsize=12, color='white')
    plt.show()

class_names = ['__background__',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

sym = get_resnet101()

# load params
arg_params, aux_params = load_param(params, ctx=mx.gpu())

# produce shape max possible
data_names = ['data', 'im_info']
label_names = None
data_shapes = [('data', (1, 3, 1000, 600)), ('im_info', (1, 3))]
label_shapes = None

# check shapes
check_shape(sym, data_shapes, arg_params, aux_params)

# create and bind module
mod = mx.module.Module(sym, data_names, label_names, context=mx.gpu())
mod.bind(data_shapes, label_shapes, for_training=False)
mod.init_params(arg_params=arg_params, aux_params=aux_params)

imgPath = imgsDir + selectors.selectImg(imgsDir)

# load single test
im_tensor, im_info, im_orig = load_test(imgPath, short=600, max_size=1000,
                                        mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))

# generate data batch
data_batch = generate_batch(im_tensor, im_info)

# forward
mod.forward(data_batch)
rois, scores, bbox_deltas = mod.get_outputs()
rois = rois[:, 1:]
scores = scores[0]
bbox_deltas = bbox_deltas[0]
im_info = im_info[0]

# decode detection
det = im_detect(rois, scores, bbox_deltas, im_info,
                bbox_stds=(0.1, 0.1, 0.2, 0.2), nms_thresh=0.3,
                conf_thresh=1e-3)

# print out
for [cls, conf, x1, y1, x2, y2] in det:
    if cls > 0 and conf > 0.7:
        print(class_names[int(cls)], conf, [x1, y1, x2, y2])

vis_detection(im_orig, det, class_names, thresh=0.7)