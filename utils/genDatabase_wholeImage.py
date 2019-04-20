# For every image in imgsDirs, generate feature files using pickle, they are saved in Data/wholeImage-features/
import os
from utils import addImageToDB_wholeImage

count = 0

predictor = 'vgg16'
#DBdir = '../Data/wholeImage-features-{}/'.format(predictor)
DBdir = '../Data/Oxford-5k/{}/'.format(predictor)
imgsDirs = [
    # '../Data/ImageNet/ILSVRC2012/img_val/',
    # '../Data/VOCdevkit/VOC2012/JPEGImages/',
    # '../Data/userImages/'
    '../Data/Oxford-5k/oxbuild_images/'
    ]
for imgsDir in imgsDirs:
    imgNames = os.listdir(imgsDir)
    for imgName in imgNames:
        imgPath = imgsDir + imgName
        addImageToDB_wholeImage.addImageToDB(imgPath, predictor, DBdir) # select a predictor here
        count = count + 1
        if count % 1000 == 0:
            print('{} images have been processed.'.format(count))

print('All processed images amount is {}'.format(count))