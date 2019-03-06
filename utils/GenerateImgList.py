# coding=utf-8
import os
import random

trainLst_f = open('../Data/RecordIO/train.lst', 'w')
valLst_f = open('../Data/RecordIO/val.lst', 'w')

trainNo = valNo = 0

imgs_path = '../Data/ResizedObjects'
filenames = os.listdir(imgs_path)

for i, filename in enumerate(filenames):
    filepath = os.sep.join(['Data/ResizedObjects', filename])
    label = filename[:filename.rfind('.')].split('_')[1]

    rand = random.randint(1, 10)
    if rand <= 9:   #training set
        line = '{}\t{}\t{}\n'.format(trainNo, label, filepath)
        trainNo = trainNo + 1
        trainLst_f.write(line)
    else:   #validation set
        line = '{}\t{}\t{}\n'.format(valNo, label, filepath)
        valNo = valNo + 1
        valLst_f.write(line)

trainLst_f.close()
valLst_f.close()

print('{}\t{}'.format(trainNo, valNo))