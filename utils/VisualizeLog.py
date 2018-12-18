# encoding=utf8
#This script is to visualize log file to show loss and accuracy
import re
import matplotlib.pyplot as plt
import numpy as np
import os

def readLogLine(file):  #keep reading till reads something useful
    line = file.readline()
    while line \
        and not re.search('num_epoch=(.*)', line) \
        and not re.search('accuracy=(.*)', line) \
        and not re.search('entropy=(.*)', line) \
        and not re.search('mse=(.*)', line) \
        and not re.search('cost=(.*)', line):
            line = file.readline()
    return line

def showFigure(num, train_acc, train_crossEntropy, train_mse, time_cost, val_acc, val_crossEntropy, val_mse):
    x = np.arange(0, epochs)

    plt.figure(logFileName + '-' + str(num))

    plt.subplot(311)
    plt.title('Training accuracy')
    plt.plot(x, train_acc, 'r', label='train acc')
    plt.plot(x, val_acc, 'b', label='val acc')
    plt.legend()  # print label
    plt.ylabel('accuracy')

    plt.subplot(312)
    plt.title('Training cross entropy')
    plt.plot(x, train_crossEntropy, 'r', label='train cross entropy')
    plt.plot(x, val_crossEntropy, 'b', label='val cross entropy')
    plt.legend()  # print label
    plt.ylabel('cross entropy')

    plt.subplot(313)
    plt.title('Training MSE')
    plt.plot(x, train_mse, 'r', label='train mse')
    plt.plot(x, val_mse, 'b', label='val mse')
    plt.legend()  # print label
    plt.xlabel('Epoch')
    plt.ylabel('MSE')

logFiles = os.listdir('../log/')
for i, logFile in enumerate(logFiles):
    print('{}: {}'.format(i, logFile))
logFileName = logFiles[input('input log file No.:')]
showAll = True if raw_input('Visualize all training logs?(input y or will visualize only the last training log):') == 'y' else False

f =  open('../log/' + logFileName)
line = readLogLine(f)

num = 0 #records the number of log files

logDict = {
    'Train-accuracy=(.*)': 'train_acc[i]',
    'Train-cross-entropy=(.*)': 'train_crossEntropy[i]',
    'Train-mse=(.*)': 'train_mse[i]',
    'cost=(.*)': 'time_cost[i]',
    'Validation-accuracy=(.*)': 'val_acc[i]',
    'Validation-cross-entropy=(.*)': 'val_crossEntropy[i]',
    'Validation-mse=(.*)': 'val_mse[i]'
}

while line:             #keep reading till log file ends
    epochs = int(re.search('num_epoch=(.*)', line).group(1))
    line = readLogLine(f)

    train_acc = np.zeros(epochs)
    train_crossEntropy = np.zeros(epochs)
    train_mse = np.zeros(epochs)

    time_cost = np.zeros(epochs)

    val_acc = np.zeros(epochs)
    val_crossEntropy = np.zeros(epochs)
    val_mse = np.zeros(epochs)

    num = num + 1

    while line and not re.search('num_epoch=(.*)', line):    #check if reached the end of training
        for key in logDict: #check if this line has info that contained in logDict
            i = int(re.search('Epoch\[(.*)\]', line).group(1))
            value = re.search(key, line)
            if value:
                exec('{}={}'.format(logDict[key], value.group(1)))
                break
        line = readLogLine(f)

    if showAll:
        showFigure(num, train_acc, train_crossEntropy, train_mse, time_cost, val_acc, val_crossEntropy, val_mse)

    if not showAll and not line:
        showFigure(num, train_acc, train_crossEntropy, train_mse, time_cost, val_acc, val_crossEntropy, val_mse)

plt.show()