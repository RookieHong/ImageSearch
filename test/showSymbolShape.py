import mxnet as mx

num_classes = 20

data = mx.symbol.Variable('data')

conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=64)
bn1 = mx.symbol.BatchNorm(data=conv1)
relu1 = mx.symbol.Activation(data=bn1, act_type="relu")
pool1 = mx.symbol.Pooling(data=relu1, pool_type="max", kernel=(2, 2), stride=(2, 2))
# dropout1 = mx.symbol.Dropout(data=pool1, p=0.5)

conv2 = mx.symbol.Convolution(data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=64)
bn2 = mx.symbol.BatchNorm(data=conv2)
relu2 = mx.symbol.Activation(data=bn2, act_type="relu")
pool2 = mx.symbol.Pooling(data=relu2, pool_type="max", kernel=(2, 2), stride=(2, 2))
# dropout2 = mx.symbol.Dropout(data=pool2, p=0.5)

conv3 = mx.symbol.Convolution(data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=128)
bn3 = mx.symbol.BatchNorm(data=conv3)
relu3 = mx.symbol.Activation(data=bn3, act_type="relu")
pool3 = mx.symbol.Pooling(data=relu3, pool_type="max", kernel=(2, 2), stride=(2, 2))
#dropout3 = mx.symbol.Dropout(data=pool3, p=0.5)

conv4 = mx.symbol.Convolution(data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=128)
bn4 = mx.symbol.BatchNorm(data=conv4)
relu4 = mx.symbol.Activation(data=bn4, act_type="relu")
pool4 = mx.symbol.Pooling(data=relu4, pool_type="max", kernel=(2, 2), stride=(2, 2))
#dropout4 = mx.symbol.Dropout(data=pool4, p=0.5)

conv5 = mx.symbol.Convolution(data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=256)
bn5 = mx.symbol.BatchNorm(data=conv5)
relu5 = mx.symbol.Activation(data=bn5, act_type="relu")
pool5 = mx.symbol.Pooling(data=relu5, pool_type="max", kernel=(2, 2), stride=(2, 2))
#dropout5 = mx.symbol.Dropout(data=pool5, p=0.5)

conv6 = mx.symbol.Convolution(data=pool5, kernel=(3, 3), pad=(1, 1), num_filter=256)
bn6 = mx.symbol.BatchNorm(data=conv6)
relu6 = mx.symbol.Activation(data=bn6, act_type="relu")
pool6 = mx.symbol.Pooling(data=relu6, pool_type="max", kernel=(2, 2), stride=(2, 2))
#dropout6 = mx.symbol.Dropout(data=pool6, p=0.5)

# conv7 = mx.symbol.Convolution(data=pool6, kernel=(3, 3), pad=(1, 1), num_filter=512)
# bn7 = mx.symbol.BatchNorm(data=conv7)
# relu7 = mx.symbol.Activation(data=bn7, act_type="relu")
# pool7 = mx.symbol.Pooling(data=relu7, pool_type="max", kernel=(2, 2), stride=(2, 2))
# #dropout7 = mx.symbol.Dropout(data=pool7, p=0.5)
#
# conv8 = mx.symbol.Convolution(data=pool7, kernel=(3, 3), pad=(1, 1), num_filter=512)
# bn8 = mx.symbol.BatchNorm(data=conv8)
# relu8 = mx.symbol.Activation(data=bn8, act_type="relu")
# pool8 = mx.symbol.Pooling(data=relu8, pool_type="max", kernel=(2, 2), stride=(2, 2))
# #dropout8 = mx.symbol.Dropout(data=pool8, p=0.5)

# conv9 = mx.symbol.Convolution(data=dropout8, kernel=(3, 3), pad=(2, 2), num_filter=128)
# bn9 = mx.symbol.BatchNorm(data=conv9)
# relu9 = mx.symbol.Activation(data=bn9, act_type="relu")
# pool9 = mx.symbol.Pooling(data=relu9, pool_type="max", kernel=(2, 2), stride=(2, 2))
# dropout9 = mx.symbol.Dropout(data=pool9, p=0.5)

flatten = mx.symbol.Flatten(data=pool6)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096)
reluA = mx.symbol.Activation(data=fc1, act_type="relu")
dropoutA = mx.symbol.Dropout(data=reluA, p=0.5)

fc2 = mx.symbol.FullyConnected(data=dropoutA, num_hidden=4096)
reluB = mx.symbol.Activation(data=fc2, act_type="relu")
dropoutB = mx.symbol.Dropout(data=reluB, p=0.5)

fc3 = mx.symbol.FullyConnected(data=dropoutB, num_hidden=num_classes)

sym = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')

data_shape=(64,3,224,224)
mx.viz.plot_network(sym, save_format='jpg', shape={"data":data_shape}, node_attrs={"shape":'oval',"fixedsize":'false'}).view()