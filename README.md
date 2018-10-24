# tracker
森哥目标追踪

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

**生成跟踪模板帧图像**
![avatar](https://github.com/wenxingsen/tracker/blob/master/images/img1.jpg)
**生成跟踪模板帧图像标签**
![avatar](https://github.com/wenxingsen/tracker/blob/master/images/img1_box.jpg)

**生成跟踪预测帧图像**
![avatar](https://github.com/wenxingsen/tracker/blob/master/images/img2.jpg)
**生成跟踪预测帧图像标签**
![avatar](https://github.com/wenxingsen/tracker/blob/master/images/img2_box.jpg)


