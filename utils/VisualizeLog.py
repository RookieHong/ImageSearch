# encoding=utf8
#这个脚本是用来可视化训练的log文件的内容的，以图表形式画出训练过程的loss曲线和acc曲线
import re
import matplotlib.pyplot as plt
import numpy as np
import os

def showFigure(num, train_acc, train_crossEntropy, train_mse, time_cost, val_acc, val_crossEntropy, val_mse):
    x = np.arange(0, epochs)

    plt.figure(logFileName + '-' + str(num))

    plt.subplot(311)
    plt.title('Training accuracy')
    plt.plot(x, train_acc, 'r', label='train acc')
    plt.plot(x, val_acc, 'b', label='val acc')
    plt.legend()  # 显示上面的label
    plt.ylabel('accuracy')

    plt.subplot(312)
    plt.title('Training cross entropy')
    plt.plot(x, train_crossEntropy, 'r', label='train cross entropy')
    plt.plot(x, val_crossEntropy, 'b', label='val cross entropy')
    plt.legend()  # 显示上面的label
    plt.ylabel('cross entropy')

    plt.subplot(313)
    plt.title('Training MSE')
    plt.plot(x, train_mse, 'r', label='train mse')
    plt.plot(x, val_mse, 'b', label='val mse')
    plt.legend()  # 显示上面的label
    plt.xlabel('Epoch')
    plt.ylabel('MSE')

logFiles = os.listdir('../log/')
for i, logFile in enumerate(logFiles):
    print('{}: {}'.format(i, logFile))
logFileName = logFiles[input('输入log文件编号:')]
showAll = True if raw_input('是否画出每次训练的曲线?(y确认否则只画出最后一次):') == 'y' else False

f =  open('../log/' + logFileName)
line = f.readline()

num = 0 #记录这是log文件中的第几次训练

while line:             #逐行读取log文件，直到最后一次训练
    epochs = int(re.search('num_epoch=(.*)', line).group(1))
    line = f.readline()

    train_acc = np.zeros(epochs)
    train_crossEntropy = np.zeros(epochs)
    train_mse = np.zeros(epochs)

    time_cost = np.zeros(epochs)

    val_acc = np.zeros(epochs)
    val_crossEntropy = np.zeros(epochs)
    val_mse = np.zeros(epochs)

    num = num + 1

    for i in range(0, epochs):
        train_acc[i] = float(re.search('accuracy=(.*)', line).group(1))
        line = f.readline()
        train_crossEntropy[i] = float(re.search('entropy=(.*)', line).group(1))
        line = f.readline()
        train_mse[i] = float(re.search('mse=(.*)', line).group(1))
        line = f.readline()

        time_cost[i] = float(re.search('cost=(.*)', line).group(1))
        line = f.readline()

        if re.search('checkpoint', line) is not None:   # 跳过保存检查点参数的那一行，通常都在time_cost那行之后
            line = f.readline()

        val_acc[i] = float(re.search('accuracy=(.*)', line).group(1))
        line = f.readline()
        val_crossEntropy[i] = float(re.search('entropy=(.*)', line).group(1))
        line = f.readline()
        val_mse[i] = float(re.search('mse=(.*)', line).group(1))
        line = f.readline()

        if re.search('Change learning rate', line) is not None: # 跳过改变学习率的那一行， 通常都在validation_mse那行之后
            line = f.readline()

    if showAll:
        showFigure(num, train_acc, train_crossEntropy, train_mse, time_cost, val_acc, val_crossEntropy, val_mse)

    if not showAll and not line:
        showFigure(num, train_acc, train_crossEntropy, train_mse, time_cost, val_acc, val_crossEntropy, val_mse)

plt.show()