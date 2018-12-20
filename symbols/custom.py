import mxnet as mx

def get_symbol(num_classes=10):
    data = mx.symbol.Variable('data')

    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=16)
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    pool1 = mx.symbol.Pooling(data=relu1, pool_type="max", kernel=(2,2), stride=(2,2))

    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=16)
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, pool_type="max", kernel=(2,2), stride=(2,2))

    conv3 = mx.symbol.Convolution(data=pool2, kernel=(3,3),num_filter=16)
    relu3 = mx.symbol.Activation(data=conv3, act_type='relu')
    pool3 = mx.symbol.Pooling(data=relu3, pool_type='max', kernel=(2,2), stride=(2,2))

    flatten = mx.symbol.Flatten(data=pool3)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    relu3 = mx.symbol.Activation(data=fc1, act_type="relu")

    fc2 = mx.symbol.FullyConnected(data=relu3, num_hidden=num_classes)

    custom = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return custom