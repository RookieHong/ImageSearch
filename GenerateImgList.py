# coding=utf-8
import os
import random

trainLst_f = open('Data/RecordIO/train.lst', 'w')
valLst_f = open('Data/RecordIO/val.lst', 'w')
testLst_f = open('Data/RecordIO/test.lst', 'w')

trainNo = valNo = testNo = 0

imgs_path = 'Data/VOCdevkit/VOC2012/ResizedObjects'
filenames = os.listdir(imgs_path)

for i, filename in enumerate(filenames):
    filepath = os.sep.join([imgs_path, filename])
    label = filename[:filename.rfind('.')].split('_')[1]

    rand = random.randint(1, 10)
    if rand <= 7:   #分到训练集中
        line = '{}\t{}\t{}\n'.format(trainNo, label, filepath)
        trainNo = trainNo + 1
        trainLst_f.write(line)
    elif rand >= 8 and rand <= 9:   #分到验证集中
        line = '{}\t{}\t{}\n'.format(valNo, label, filepath)
        valNo = valNo + 1
        valLst_f.write(line)
    else:   #分到测试集
        line = '{}\t{}\t{}\n'.format(testNo, label, filepath)
        testNo = testNo + 1
        testLst_f.write(line)

trainLst_f.close()
valLst_f.close()
testLst_f.close()

print('{}\t{}\t{}'.format(trainNo, valNo, testNo))