import cv2
import selectivesearch

img = cv2.imread('Data/VOCdevkit/VOC2012/JPEGImages/2007_000039.jpg')
img_label, regions = selectivesearch.selective_search(img, scale = 500, sigma = 0.9, min_size = 200)
for i, region in enumerate(regions):    #rect:x y w h
    x = region['rect'][0]
    y = region['rect'][1]
    w = region['rect'][2]
    h = region['rect'][3]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
cv2.imshow('result', img)
cv2.waitKey(0)