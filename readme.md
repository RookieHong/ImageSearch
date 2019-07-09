# 一个图像检索系统

这是一个基于MXNET实现的图像检索系统，从头到尾地实现了从训练卷积神经网络模型，提取特征，生成图像数据库，检索耗时评估，检索准确率评估到前端网页的全步骤

在该图像检索系统中实现了两种图像检索模式，一种是基于整张图像提取特征，另一种是基于目标检测，对物体部分提取特征。

使用了多种模型，包括Resnet18，Resnet101，Resnet152，VGG16等等实现了图像检索并对比它们的表现

使用的目标检测算法来自Faster R-CNN， 算法代码来自[MX-RCNN](https://github.com/ijkguo/mx-rcnn)

![整图检索](rdImgs/ScreenCapture-wholeImg.gif "整图检索")

![物体检索](rdImgs/ScreenCapture-object.gif "物体检索")

其中每个文件夹下存放的内容如下：

`Assessments`：评估相关的脚本，包括准确率和耗时的评估

`Data`：数据，包括训练使用的数据，生成图像数据库的数据以及做准确率评估的数据

`log`：各种日志，包括测试以及训练的

`params`：各类模型的参数

`predictors`：封装好的各种模型

`symbols`：各类卷积神经网络的结构

`test`：各种测试时用的脚本，不充当项目中的成分

`utils`：各类工具脚本

`web`：前端相关的文件

`训练记录.docx`是在训练卷积神经网络时做的记录，里面记录了每次调整网络结构带来的训练变化

`记录.docx`是整个项目的记录，从一开始的设计思路到最后的完成的整个过程的记录，也是我作为一个小白从头开始学习深度学习和图像检索的经验记录

## 运行该项目的步骤：

*这里就写我完成这个项目的过程吧*

**1. 训练卷积神经网络模型**

首先我下载了Pascal VOC 2012数据集，放在了`Data/VOCdevkit`

然后使用`utils/DataSetObjectExtraction.py`脚本对数据集中的物体图片进行裁剪并resize，输出的图片在`Data/ResizedObjects`

运行`utils/GenerateImgList.py`脚本按照9:1的比例将输出的图片分成训练集和验证集，结果的lst文件放在`Data/RecordIO`中

使用MXNET自带的`utils/im2rec.py`根据lst文件把数据集里的图片转换成rec文件，执行以下指令`sudo python utils/im2rec.py --num-thread=4 Data/RecordIO/train.lst  `，转换之后的rec文件在`Data/RecordIO`中

然后运行`trainings/trainOnResizedObjects.py`脚本开始训练，训练的日志放在`log`文件夹中，训练之后的模型放在`params`文件夹中

运行`utils/VisualizeLog.py`可以可视化训练日志，生成准确率和误差的折线图

**2. 生成图像数据库**

运行`utils/genDatabase_wholeImage.py`脚本可以生成基于整图检索的图像数据库

运行`utils/genDatabase_objects.py`脚本可以生成基于物体检索的图像数据库

运行`utils/genDatabase_RMAC.py`脚本可以生成RMAC特征的图像数据库

其中的图像数据库存储位置以及使用的图像位置需要在脚本中修改以指定，图像数据库以pickle的方式写入

整图检索的数据库中保存了图像路径、图像特征

物体检索的数据库中保存了物体在原图的坐标、图像特征以及路径

**3. 在前端操作**

我用的是apache2配置的本地服务器，具体配置过程可以百度

`web/cgi/process.py`脚本接收前端请求并处理，将结果返回前端，是最核心的一个脚本

**4. 耗时评估**

`Assessments/searchTime_wholeImage.py`脚本运行后会统计出基于整图检索的耗时统计柱状图

`Assessments/searchTime_objects.py`脚本运行后会统计出基于物体检索的耗时统计柱状图

我在实现过程中发现基于物体检索的耗时实在是太高，于是便写了`utils/splitDB_kmeans.py`，使用kmeans算法把物体检索的数据库“分裂”

分裂后的耗时统计柱状图由`Assessments/searchTime_objects_splitedDB.py`运行得到

**5. 检索准确率评估**

这部分检索准确率评估使用的数据集来自`Oxford Buildings Dataset`，俗称Oxford5k

首先运行`utils/extractOxford5kQueryImages.py`脚本提取出Oxford5k数据集中的query images,同时也会得到裁剪后的以及resize后的query images

然后需要使用之前生成图像数据库的步骤把Oxford5k数据集中的图片通过每个模型生成它们的特征，我是放在了`Data/Oxford-5k`下对应模型名字的文件夹中

修改`Assessments/computeAP_CNN.py`脚本中的各个参数并运行，可以得到对应各个模型的检索准确率结果，存放在`log/Test/test_Oxford-5k_mAP.log`日志中

这时评估使用的都是CNN的朴素图像特征，你可能会觉得准确率不尽如人意，于是我就又实现了R-MAC特征进行实验

[R-MAC论文链接](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=2ahUKEwiPrsKuppjjAhVCTrwKHfkNAScQFjABegQIAhAB&url=https%3A%2F%2Farxiv.org%2Fabs%2F1511.05879&usg=AOvVaw31QNrzPsuacjaHaCls81-t)

*我找过了网上，居然没有MXNET版本的R-MAC特征实现，这可能是MXNET版本的第一个R-MAC特征实现了*

生成R-MAC特征的过程要稍微麻烦一些，因为过程中使用了PCA，而产生PCA的数据集在后面的实验中也会发现，会大大地影响最后的评估结果

修改`utils/genPCA_RMAC.py`脚本中的参数，包括使用的数据集，维度以及模型，运行后会在`Data/PCA`目录下得到相应的PCA文件

我这里使用的产生PCA数据集的数据集包括Oxford5k和Paris6k，还有它们混合之后的数据集

之后运行`Assessments/computeAP_RMAC.py`脚本可得到R-MAC特征的检索准确率评估结果，同样放在上面所说的日志文件中

另外，提取图片的Feature Map时，是否对图片resize也会对结果影响很大，每个predictor里面都有getFeatureMap_resized和getFeatureMap两个函数，其中带resized后缀的是会对输入图片resize的函数

在`Assessments/computeAP_RMAC.py`中提取featureMap的那一行可以修改是否将输入图片resize

传统的图像检索方法：

| **检索方法** | **mAP** |
| :----: | :----:|
| BoW 200k-D | 0.364 |
| BoW 20k-D | 0.354 |
| VLAD 64D | 0.555 |
| Improved Fisher 64D | 0.418 |

以上数据来自相应的论文

朴素CNN图像特征测试结果：

| **模型** | **mAP** |
| :----: | :----:|
| Resnet18 512D | 0.424 |
| Resnet101 2048D | 0.439 |
| Resnet152 2048D | 0.425 |
| VGG16 25088D | 0.421 |
| Custom 3136D | 0.081 |

使用R-MAC特征，resize输入图片的测试结果：

<table>
   <tr>
      <th></th>
      <th colspan="3">PCA所用数据集</th>
   </tr>
   <tr>
      <td>模型</td>
      <td>Paris6k</td>
      <td>Oxford5k</td>
      <td>Paris6k+Oxford5k</td>
   </tr>
   <tr>
      <td>VGG16 512D</td>
      <td>0.587</td>
      <td>0.538</td>
      <td>0.568</td>
   </tr>
   <tr>
      <td>Resnet18 512D</td>
      <td>0.439</td>
      <td>0.395</td>
      <td>0.417</td>
   </tr>
   <tr>
      <td>Resnet101 2048D</td>
      <td>0.102</td>
      <td>0.072</td>
      <td>0.087</td>
   </tr>
   <tr>
      <td>Resnet101 512D</td>
      <td>0.157</td>
      <td>0.138</td>
      <td>0.148</td>
   </tr>
   <tr>
      <td>Resnet152 2048D</td>
      <td>0.158</td>
      <td>0.117</td>
      <td>0.010</td>
   </tr>
   <tr>
      <td>Resnet152 512D</td>
      <td>0.181</td>
      <td>0.165</td>
      <td>0.174</td>
   </tr>
   <tr>
      <td>自定义网络 512D</td>
      <td>0.129</td>
      <td>0.128</td>
      <td>0.129</td>
   </tr>
</table>

使用R-MAC特征，不resize输入图片的测试结果：

<table>
   <tr>
      <th></th>
      <th colspan="3">PCA所用数据集</th>
   </tr>
   <tr>
      <td>模型</td>
      <td>Paris6k</td>
      <td>Oxford5k</td>
      <td>Paris6k+Oxford5k</td>
   </tr>
   <tr>
      <td>VGG16 512D</td>
      <td>0.722</td>
      <td>0.670</td>
      <td>0.705</td>
   </tr>
   <tr>
      <td>Resnet18 512D</td>
      <td>0.574</td>
      <td>0.500</td>
      <td>0.541</td>
   </tr>
   <tr>
      <td>Resnet101 2048D</td>
      <td>0.075</td>
      <td>0.075</td>
      <td>0.105</td>
   </tr>
   <tr>
      <td>Resnet101 512D</td>
      <td>0.220</td>
      <td>0.196</td>
      <td>0.205</td>
   </tr>
   <tr>
      <td>Resnet152 2048D</td>
      <td>0.245</td>
      <td>0.193</td>
      <td>0.168</td>
   </tr>
   <tr>
      <td>Resnet152 512D</td>
      <td>0.271</td>
      <td>0.255</td>
      <td>0.264</td>
   </tr>
   <tr>
      <td>自定义网络 512D</td>
      <td>0.140</td>
      <td>0.138</td>
      <td>0.140</td>
   </tr>
</table>

可以在`Assessments/computeAP_RMAC.py`中修改QEsize以设置Query Expansion，提高最后的表现，以下是使用VGG16，不resize输入图片下各种QEsize的测试结果：

| k   | 0     | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     | 9     | 10    | 11    | 12    | 13    | 14    | 15    |
|-----|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| mAP | 0.722 | 0.723 | 0.744 | 0.761 | 0.767 | 0.774 | 0.775 | 0.776 | 0.776 | 0.775 | 0.774 | 0.767 | 0.757 | 0.753 | 0.748 | 0.742 |

可以看到k=7和8的时候mAP有0.776，已经远远地超出传统的图像检索方法了

这是我作为一个小白从零开始学习深度学习和图像检索做的项目，自己感觉做的比较完整，也踩过很多坑，希望这个项目能够帮到更多同样的在学习的人吧！

还要强力推荐一下[willard-yuan](https://github.com/willard-yuan)！在他的博客里面学到了很多，在学习图像检索的同学们可以多去看看这位前辈的博客！
