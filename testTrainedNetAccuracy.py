# coding=utf-8
import mxnet as mx

test_iter = mx.io.ImageRecordIter(
    path_imgrec = 'Data/RecordIO/test.rec',
    data_shape = (3, 227, 227),

    batch_size=50,  # 每次传入50条数据

    mean_r = 128,   #三个通道的均值
    mean_g = 128,
    mean_b = 128,

    scale = 0.00390625  #减去均值后归一化到[-0.5, 0.5]之间
)
mod = mx.mod.Module.load('params/PascalVOC_AlexNet', 35, context = mx.gpu(0))
mod.bind(
    data_shapes = test_iter.provide_data,
    label_shapes = test_iter.provide_label,
    for_training = False
)
metric = mx.metric.create('acc')
mod.score(test_iter, metric)
for name, val in metric.get_name_value():
    print('{}={:4f}%'.format(name, val * 100))