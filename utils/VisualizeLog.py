# encoding=utf8
#这个脚本是用来可视化训练的log文件的内容的，以图表形式画出训练过程的loss曲线和acc曲线
import re
import matplotlib.pyplot as plt
import numpy as np

train_acc = np.zeros(36)
train_crossEntropy = np.zeros(36)
train_mse = np.zeros(36)

time_cost = np.zeros(36)

val_acc = np.zeros(36)
val_crossEntropy = np.zeros(36)
val_mse = np.zeros(36)

logFileName = 'train_AlexNet-miniDataset.log'
f =  open('../log/' + logFileName)
line = f.readline()
while line:             #逐行读取log文件，直到最后一次训练的迭代
    for i in range(0, 36):
        train_acc[i] = float(re.search('accuracy=(.*)', line).group(1))
        line = f.readline()
        train_crossEntropy[i] = float(re.search('entropy=(.*)', line).group(1))
        line = f.readline()
        train_mse[i] = float(re.search('mse=(.*)', line).group(1))
        line = f.readline()

        time_cost[i] = float(re.search('cost=(.*)', line).group(1))
        line = f.readline()

        if re.search('checkpoint', line) is not None:   #   跳过保存检查点参数的那一行，通常都在time_cost那行之后
            line = f.readline()

        val_acc[i] = float(re.search('accuracy=(.*)', line).group(1))
        line = f.readline()
        val_crossEntropy[i] = float(re.search('entropy=(.*)', line).group(1))
        line = f.readline()
        val_mse[i] = float(re.search('mse=(.*)', line).group(1))
        line = f.readline()

x = np.arange(0, 36)

plt.figure(logFileName)

plt.subplot(311)
plt.title('Training accuracy')
plt.plot(x, train_acc, 'r', label = 'train acc')
plt.plot(x, val_acc, 'b', label = 'val acc')
plt.legend()    #显示上面的label
plt.ylabel('accuracy')

plt.subplot(312)
plt.title('Training cross entropy')
plt.plot(x, train_crossEntropy, 'r', label = 'train cross entropy')
plt.plot(x, val_crossEntropy, 'b', label = 'val cross entropy')
plt.legend()    #显示上面的label
plt.ylabel('cross entropy')

plt.subplot(313)
plt.title('Training MSE')
plt.plot(x, train_mse, 'r', label = 'train mse')
plt.plot(x, val_mse, 'b', label = 'val mse')
plt.legend()    #显示上面的label
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()
