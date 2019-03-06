import mxnet as mx

num_classes = 20

data = mx.symbol.Variable('data')

conv1 = mx.symbol.Convolution(data=data, kernel=(8,8), stride=(4,4),num_filter=20)
bn1 = mx.symbol.BatchNorm(data=conv1)
relu1 = mx.symbol.Activation(data=bn1, act_type="relu")
pool1 = mx.symbol.Pooling(data=relu1, pool_type="max", kernel=(3,3), stride=(2,2))
dropout1 = mx.symbol.Dropout(data=pool1, p=0.5)

conv2 = mx.symbol.Convolution(data=dropout1, kernel=(5,5), pad=(2, 2), num_filter=50)
bn2 = mx.symbol.BatchNorm(data=conv2)
relu2 = mx.symbol.Activation(data=bn2, act_type="relu")
pool2 = mx.symbol.Pooling(data=relu2, pool_type="max", kernel=(3,3), stride=(2,2))
dropout2 = mx.symbol.Dropout(data=pool2, p=0.5)

conv3 = mx.symbol.Convolution(data=dropout2, kernel=(5, 5), pad=(2, 2), num_filter=50)
bn3 = mx.symbol.BatchNorm(data=conv3)
relu3 = mx.symbol.Activation(data=bn3, act_type="relu")
pool3 = mx.symbol.Pooling(data=relu3, pool_type="max", kernel=(3, 3), stride=(2, 2))
dropout3 = mx.symbol.Dropout(data=pool3, p=0.5)

flatten = mx.symbol.Flatten(data=dropout3)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
relu3 = mx.symbol.Activation(data=fc1, act_type="relu")
dropout3 = mx.symbol.Dropout(data=relu3, p=0.5)

fc2 = mx.symbol.FullyConnected(data=dropout3, num_hidden=500)
relu4 = mx.symbol.Activation(data=fc2, act_type="relu")
dropout4 = mx.symbol.Dropout(data=relu4, p=0.5)

fc2 = mx.symbol.FullyConnected(data=dropout4, num_hidden=num_classes)

sym = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')

arg_shape, out_shape, aux_shape = sym.infer_shape(data=(1, 3, 224, 224))
print(dict(zip(sym.list_arguments(), arg_shape)))