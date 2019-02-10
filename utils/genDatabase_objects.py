# For every image in imgsDirs, generate feature files using pickle, they are saved in Data/objects-features-resnet101_fasterRcnn/
import os
from utils import addImageToDB_objects
import time

usedTime = time.time()

count = 0

imgsDirs = [
    '../Data/ImageNet/ILSVRC2012/img_val/'
    '../Data/VOCdevkit/VOC2012/JPEGImages/',
    '../Data/userImages/'
    ]
for imgsDir in imgsDirs:
    imgNames = os.listdir(imgsDir)
    for imgName in imgNames:
        imgPath = imgsDir + imgName
        addImageToDB_objects.addImageToDB(imgPath, 'resnet101_fasterRcnn')
        count = count + 1
        if count % 1000 == 0:
            print('{} images have been processed.'.format(count))

usedTime = time.time() - usedTime

print('All processed images amount is {}'.format(count))
print('Used time {}s'.format(usedTime))