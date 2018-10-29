# tracker 
森哥目标追踪

## 目标跟踪背景
目标跟踪，是通用单目标跟踪，第一帧给个矩形框，这个框在数据库里面是人工标注的，在实际情况下大多是检测算法的结果，然后需要跟踪算法在后续帧紧跟住这个框。

外观变形，光照变化，快速运动和运动模糊，背景相似干扰
![avatar](https://pic2.zhimg.com/80/v2-1169ca84d569b5f8aff728d0de563869_hd.jpg)
正因为这些情况才让tracking变得很难，目前比较常用的数据库除了OTB，还有前面找到的VOT竞赛数据库(类比ImageNet)

## 数据集
OTB和VOT区别：OTB包括25%的灰度序列，但VOT都是彩色序列，这也是造成很多颜色特征算法性能差异的原因；两个库的评价指标不一样，具体请参考论文；VOT库的序列分辨率普遍较高，这一点后面分析会提到。

## 传统方法
目标视觉跟踪(Visual Object Tracking)，大家比较公认分为两大类：生成(generative)模型方法和判别(discriminative)模型方法
### 生成类
生成类方法，在当前帧对目标区域建模，下一帧寻找与模型最相似的区域就是预测位置，比较著名的有卡尔曼滤波，粒子滤波，mean-shift等。举个例子，从当前帧知道了目标区域80%是红色，20%是绿色，然后在下一帧，搜索算法就像无头苍蝇，到处去找最符合这个颜色比例的区域
### 判别类
判别类方法，OTB50里面的大部分方法都是这一类，CV中的经典套路图像特征+机器学习， 当前帧以目标区域为正样本，背景区域为负样本，机器学习方法训练分类器，下一帧用训练好的分类器找最优区域
与生成类方法最大的区别是，分类器采用机器学习，训练中用到了背景信息，这样分类器就能专注区分前景和背景，所以判别类方法普遍都比生成类好。
### 相关滤波
介绍最经典的高速相关滤波类跟踪算法CSK, KCF/DCF, CN



## 目标跟踪生成示意图第一版  
随机生成训练验本，解决目标跟踪训练集不足的问题  
生成训练集解决方案:  
 
### 背景图  
从网络下载n张，偏大的照片  
![avatar](https://github.com/wenxingsen/tracker/blob/master/images/backgrounds.jpg)

### 前景目标图  
从网络下载n张，偏小的照片  
![avatar](https://github.com/wenxingsen/tracker/blob/master/images/fronts.jpg)

### 生成标签算法：
随机从背景库里面挑选一张背景，随机从前景目标库里面挑选一张前景，
通过随机生成一个xy坐标，前景和背景融合起来，然后随机产生xy的一定距离偏差，（后期加一些形变），
重新生成前景和背景融合的图片，与此同时，同时生成黑框。

![avatar](https://github.com/wenxingsen/tracker/blob/master/images/demo1.jpg)

**生成跟踪 模板帧图像**
![avatar](https://github.com/wenxingsen/tracker/blob/master/images/img1.jpg)
**生成跟踪 模板帧box**
![avatar](https://github.com/wenxingsen/tracker/blob/master/images/img1_box.jpg)

**生成跟踪 预测帧图像**
![avatar](https://github.com/wenxingsen/tracker/blob/master/images/img2.jpg)
**生成跟踪预测帧box**
![avatar](https://github.com/wenxingsen/tracker/blob/master/images/img2_box.jpg)

### 前景和背景加上一定的抖动
![avatar](https://github.com/wenxingsen/tracker/blob/master/images/generate.gif)

### 网络拓扑
网络的输入：模板帧图像，模板帧box，预测帧图像  
网络的输出：预测帧box  
![avatar](https://github.com/wenxingsen/tracker/blob/master/images/trakernet.jpg)


### 预测效果图
追踪的图
![avatar](https://github.com/wenxingsen/tracker/blob/master/images/pillow_img.jpg)

追踪的预测图
![avatar](https://github.com/wenxingsen/tracker/blob/master/images/pillow_box.jpg)
