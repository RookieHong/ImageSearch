# encoding=utf8
#This script is to visualize log file to show loss and accuracy
import re
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import selectors

def readLogLine(file):  #keep reading till reads something useful
    line = file.readline()
    while line \
        and not re.search('num_epoch=(.*)', line) \
        and not re.search('accuracy=(.*)', line) \
        and not re.search('entropy=(.*)', line):
            line = file.readline()
    return line

def showFigure(num, train_acc, train_crossEntropy, val_acc, val_crossEntropy, epoch):
    x = np.arange(0, epoch)

    plt.figure(logFileName + '-' + str(num))

    plt.subplot(211)
    plt.title('Accuracy')
    plt.plot(x, train_acc, 'r', label='train acc')
    plt.plot(x, val_acc, 'b', label='val acc')
    plt.legend()  # print label
    plt.ylabel('accuracy')

    plt.subplot(212)
    plt.title('Cross entropy')
    plt.plot(x, train_crossEntropy, 'r', label='train cross entropy')
    plt.plot(x, val_crossEntropy, 'b', label='val cross entropy')
    plt.legend()  # print label
    plt.ylabel('cross entropy')

logFileName = selectors.selectLog('../log/Training/')
showLogs = input('Input n to show last n training logs:')

f =  open('../log/Training/' + logFileName)
line = readLogLine(f)

num = 0 #records the number of log files

logDict = {
    'Train-accuracy=(.*)': 'train_acc[i]',
    'Train-cross-entropy=(.*)': 'train_crossEntropy[i]',
    'Validation-accuracy=(.*)': 'val_acc[i]',
    'Validation-cross-entropy=(.*)': 'val_crossEntropy[i]'
}

epochs = []

train_accs = []
train_crossEntropys = []

val_accs = []
val_crossEntropys = []

while line:             #keep reading till log file ends
    epoch = int(re.search('num_epoch=(.*)', line).group(1))
    line = readLogLine(f)

    train_acc = np.zeros(epoch)
    train_crossEntropy = np.zeros(epoch)

    val_acc = np.zeros(epoch)
    val_crossEntropy = np.zeros(epoch)

    num = num + 1

    while line and not re.search('num_epoch=(.*)', line):    #check if reached the end of training
        for key in logDict: #check if this line has info that contained in logDict
            i = int(re.search('Epoch\[(.*)\]', line).group(1))
            value = re.search(key, line)
            if value:
                exec('{}={}'.format(logDict[key], value.group(1)))
                break
        line = readLogLine(f)

    train_accs.append(train_acc)
    train_crossEntropys.append(train_crossEntropy)

    val_accs.append(val_acc)
    val_crossEntropys.append(val_crossEntropy)

    epochs.append(epoch)

if showLogs > num:
    showLogs = num

for i in range(showLogs):
    toBeShownNo = num - 1 - i
    showFigure(toBeShownNo, train_accs[toBeShownNo], train_crossEntropys[toBeShownNo], val_accs[toBeShownNo], val_crossEntropys[toBeShownNo], epochs[toBeShownNo])

plt.show()