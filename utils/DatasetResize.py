from PIL import Image
import os

imgs_path = '../Data/VOCdevkit/VOC2012/JPEGImages/'
output_path = '../Data/ResizedImages/'
filenames = os.listdir(imgs_path)

for i, filename in enumerate(filenames):
    img = Image.open(imgs_path + filename)
    out = img.resize((227, 227), Image.ANTIALIAS)
    out.save(output_path + filename)