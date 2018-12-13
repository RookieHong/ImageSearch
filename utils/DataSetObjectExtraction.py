# coding=utf-8
import xml.etree.ElementTree as ET
from PIL import Image
import os
import json

imgs_path = '../Data/VOCdevkit/VOC2012/JPEGImages/'
output_path = '../Data/VOCdevkit/VOC2012/ResizedObjects/'
filenames = os.listdir(imgs_path)

count = 0

with open('../Classes.json', 'r') as json_f:    #载入记录类别和对应编号的json文件
    classes = json.load(json_f)

for i, filename in enumerate(filenames):
    filepath = os.sep.join([imgs_path, filename])
    img = Image.open(filepath)
    xmlTree = ET.parse('../Data/VOCdevkit/VOC2012/Annotations/' + filename.split('.')[0] + '.xml')  # 读取并解析该图片所对应的xml文件

    for object in xmlTree.findall('object'):
        count = count + 1

        name = object.find('name').text
        label = classes[name]

        bndbox = object.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))     #似乎有些坐标并不是整数
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))

        cropped = img.crop((xmin, ymin, xmax, ymax))
        cropped = cropped.resize((227, 227), Image.ANTIALIAS)

        save_path = output_path + name + '-' + str(count) + '_' + str(label) + '.jpg'
        cropped.save(save_path)
