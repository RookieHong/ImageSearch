# For every image in imgsDirs, generate feature files using pickle, they are saved in Data/wholeImage-features/
import os
from utils import addImageToDB_wholeImage

count = 0

imgsDirs = [
    '../Data/ImageNet/ILSVRC2012/img_val/',
    '../Data/VOCdevkit/VOC2012/JPEGImages/',
    '../Data/userImages/'
    ]
for imgsDir in imgsDirs:
    imgNames = os.listdir(imgsDir)
    for imgName in imgNames:
        imgPath = imgsDir + imgName
        addImageToDB_wholeImage.addImageToDB(imgPath, 'custom')
        count = count + 1
        if count % 1000 == 0:
            print('{} images have been processed.'.format(count))

print('All processed images amount is {}'.format(count))