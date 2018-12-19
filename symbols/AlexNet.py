import mxnet as mx

def get_symbol(num_classes=10):
    data = mx.sym.Variable('data')

    # stage 1
    conv1 = mx.sym.Convolution(data=data, kernel=(11, 11), stride=(4, 4), num_filter=96)
    relu1 = mx.sym.Activation(data=conv1, act_type='relu')
    lrn1 = mx.sym.LRN(data=relu1, alpha=0.0001, beta=0.75, knorm=2, nsize=5)
    pool1 = mx.sym.Pooling(data=lrn1, kernel=(3, 3), stride=(2, 2), pool_type='max')

    # stage 2
    conv2 = mx.sym.Convolution(data=pool1, kernel=(5, 5), pad=(2, 2), num_filter=256)
    relu2 = mx.sym.Activation(data=conv2, act_type='relu')
    lrn2 = mx.sym.LRN(data=relu2, alpha=0.0001, beta=0.75, knorm=2, nsize=5)
    pool2 = mx.sym.Pooling(data=lrn2, kernel=(3, 3), stride=(2, 2), pool_type='max')

    # stage 3
    conv3 = mx.sym.Convolution(data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu3 = mx.sym.Activation(data=conv3, act_type='relu')
    conv4 = mx.sym.Convolution(data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu4 = mx.sym.Activation(data=conv4, act_type='relu')
    conv5 = mx.sym.Convolution(data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256)
    relu5 = mx.sym.Activation(data=conv5, act_type='relu')
    pool3 = mx.sym.Pooling(data=relu5, kernel=(3, 3), stride=(2, 2), pool_type='max')

    # stage 4
    flatten = mx.sym.Flatten(data=pool3)
    fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=4096)
    relu6 = mx.sym.Activation(data=fc1, act_type='relu')
    dropout1 = mx.sym.Dropout(data=relu6, p=0.5)

    # stage 5
    fc2 = mx.sym.FullyConnected(data=dropout1, num_hidden=4096)
    relu7 = mx.sym.Activation(data=fc2, act_type='relu')
    dropout2 = mx.sym.Dropout(data=relu7, p=0.5)

    # stage 6
    fc3 = mx.sym.FullyConnected(data=dropout2, num_hidden=num_classes)
    softmax = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

    return softmax