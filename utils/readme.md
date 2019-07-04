## utils

这个文件夹下是各种工具，有直接运行的也有被引用的，比较混乱，写的时候都一股脑把这类脚本扔在里面了。。

`addImageToDB_wholeImage.py`：被调用，把一张图片加入到对应模型整图检索的数据库中

`addImageToDB_objects.py`：被调用，把一张图片加入到物体检索数据库中

`bbox.py`：Faster R-CNN运行必备的一个工具文件，从[MX-RCNN](https://github.com/ijkguo/mx-rcnn)的库中直接拷贝下来的

`DataSetObjectExtraction.py`：将Pascal VOC数据集中图片里的object根据Annotation裁剪并resize作为单独的图片

`extractOxford5kQueryImages.py`：提取出Oxford5k数据集中的query images并保存，同时也会保存其中经过裁剪的部分

`genDatabase_xxx.py`：这些都是生成对应图像数据库用的脚本

`GenerateImgList.py`：对从数据集中提取出来的物体图片进行随机9:1的分类，生成lst文件以生成rec文件

`genPCA_RMAC.py`：用于R-MAC特征测试的脚本，在其中使用相应的数据集以生成PCA文件

`genResizedObjectsFromObjectsDB.py`：这是我为了获得更多Pascal标注的数据时写的一个脚本，从物体检索的图像数据库中（因为物体检索使用的预训练Faster R-CNN模型是基于Pascal VOC数据集训练的）把分割出来的物体图片加入到训练的数据集中。当然我实验之后发现这样是行不通的，训练的效果会更差。。

`im2rec.py`：这是MXNET安装时自带的脚本，用来根据lst文件把图片制作成rec文件

`image.py`：同`bbox.py`是[MX-RCNN](https://github.com/ijkguo/mx-rcnn)中必须的一个工具文件

`nms.py`：非极大值抑制脚本，用于去除目标检测时多余的bbox

`rmacRegions.py`：被调用，产生提取R-MAC特征时必须的各个区域

`selectors.py`：被调用，调试时很好用的，用来选择某个参数文件或者图片等等，会列出对应目录下的文件供选择

`splitDB_kmeans.py`：主动使用，用kmeans方法“分裂”对应的图像数据库使检索更快，我使用了这个分裂物体检索的图像数据库，因为其对于六万多张图片只有20个分类

`VisualizeConvs.py`：主动使用，用来可视化一个卷积神经网络模型中间的卷积层

`VisualizeLog.py`：主动使用，可视化一个训练日志的误差以及准确率的波动，很好用