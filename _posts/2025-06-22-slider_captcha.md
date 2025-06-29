---
title: 基于OpenCV的滑块验证码识别
date: 2025-06-22 22:00:00 +0800
categories: [CV, OpenCV]
tags: [implement]     # TAG names should always be lowercase
author: momochi
# authors: [xx,xx]
description: 介绍简单的滑块验证码如何通过OpenCV解决的思路，以及实际操作流程和示例代码
comments: true # 评论
pin: false # top 
math: true
---

## 目标

首先，基本所有的滑块验证码都是由拼图(target)和背景(background)两个图片构成。如下：

拼图:

![alt text](/assets/img/slider_captcha/target.png)

背景：

![alt text](/assets/img/slider_captcha/background.png)

所以我们需要使用OpenCV完成以下任务：
- 加载拼图和背景
- 图像预处理
- 在背景中找到正确的target位置
- 输出拼图到正确缺口的偏移量(忽略实际浏览器的缩放)

## 图像预处理

在简单的滑块验证码中，拼图都是水平移动的，也就是说我们可以忽略拼图可移动范围之外的背景，以此来缩小查找范围。

观察拼图，可以知道其图片高度与背景一致，除去拼图部分其余为透明部分，因此我们可以加载拼图(包含透明度)，然后找到第一个非透明的像素位置和最后一个非透明的像素位置，此为拼图左上角所在背景中的坐标(以图片左上角为坐标原点)，然后拼图多余的透明部分可以裁剪掉，方便后续的处理，代码示例如下：

```python
image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
# 获取透明度通道
alpha_channel = image[:, :, 3]
# 找到不透明像素的边界
rows = np.any(alpha_channel > 0, axis=1)
cols = np.any(alpha_channel > 0, axis=0)
start_y, end_y = np.where(rows)[0][[0, -1]]
start_x, end_x = np.where(cols)[0][[0, -1]]
# 裁剪图像
cropped_image = image[start_y:end_y + 1, start_x:end_x + 1]
```

裁剪后的拼图如下：

![alt text](/assets/img/slider_captcha/target_cat.png)

然后，我们需要裁剪背景，因为我们已经知道了拼图的左上角所在背景中的坐标(`start_x`和`start_y`)，于是我们可以裁剪出背景中`(0,start_y)`到`(max_weight, start_y + h)`部分。其中`max_weight`为背景图片的宽，`h`为拼图的高。`h=end_y-start_y`，背景图片的宽可以通过输出`shape`获得。裁剪背景的示例代码如下：

```python
background = cv2.imread(background_path, cv2.IMREAD_ANYCOLOR)
background = background[target_y:target_y + h, 0:background.shape[1]]  # 裁剪背景图片，缩小匹配范围
```

以上，基础的图像处理已经结束，接着需要根据最终的匹配方法来采取不同的处理方法。

本篇最终使用OpenCV的模板匹配matchTemplate。可以在背景中找到与拼图最相似的部分，然而直接使用该方法肯定是找不到的，因为背景中对应部分是缺口，而并非原图。所以我们现在的目的应该是让背景和拼图忽略掉颜色细节，也就是说只有黑白灰，同时尽量让缺口和拼图颜色统一，都为白色或黑色，这样方便匹配。以下以白色为例。

拼图很好处理，只需要将透明部分设置为黑色，其余部分设置为白色，处理就结束了，示例代码如下：

```python
target = cv2.imread(target_path, cv2.IMREAD_UNCHANGED) # 读取裁剪后的拼图
mask = target[:, :, 3]  # 提取透明度通道
mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]  # 将透明度通道转换为二值掩码
target[mask == 255] = [255, 255, 255]
target[mask == 0] = [0, 0, 0]
```

处理后的拼图如下：

![alt text](/assets/img/slider_captcha/res_target.png)

接着，我们将会处理背景，而背景中并不像拼图这样，有明显的区分(透明和非透明)，所以无法固定`mask`来直接让缺口部分为白色，其余部分为黑色。通过观察背景，可以发现两点：

1. 缺口部分相较于其余非缺口部分颜色较暗
2. 缺口部分和缺口周围部分有明显的色差

根据以上两点，可以有不同的处理方法：

1. 根据亮度差异动态确定阈值，二值化背景，与拼图一样使用`mask`，将阈值之外部分设置为黑色，阈值之内设置为白色
2. 使用边缘检测，然后将滑块部分设置为白色，其余设置为黑色

### 亮度差异+阈值化

通过读取背景灰度图，分析其像素亮度分布，最终提取出“较暗”区域，然后将这些区域设置为白色，其他区域设置为黑色。

该方法最关键的步骤是找到合适的阈值来区分“较暗”和“较亮”区域。

这里经过多次实践，发现缺口部分的颜色(RGB)波动较大，很难找到一个合适的固定阈值来区分缺口和背景。因此这里使用动态阈值的方法，OTSU适合图像直方图有两个峰的情况：

```python
cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
```

接着只需要利用这个`mask`将255对应原图片的位置设置为白色，其余设置为黑色。

上述裁剪过后的背景图，经过该处理后的效果图如下：

![alt text](/assets/img/slider_captcha/res_background.png)

可以直接找出与处理后的拼图较匹配的位置，最终我们使用模板匹配方法，在背景图中找到与拼图最佳位置：

```python
res = cv2.matchTemplate(background, target, cv2.TM_CCOEFF_NORMED)
# 找到最佳匹配位置 其中max_loc是匹配度最高的位置
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
```

如此，找到了最佳位置，这里将匹配位置绘制在裁剪后的背景图中：

![alt text](/assets/img/slider_captcha/res.png)

该方法的提升空间在于阈值化步骤，只要阈值确定的好，就能较好的区分出缺口和背景，如此再使用模板匹配时就能更好的找到正确位置。

### 边缘检测

通过边缘检测，我们希望该方法能够较好的通过缺口和其余部分的颜色梯度变化识别出缺口位置。最终找位置都是使用模板匹配的方式。

这里，我们依旧使用上面图像预处理后的拼图(未二值化)和背景。直接使用`cv2.Canny`看一下边缘检测后的图片效果：

```python
background = cv2.Canny(background, 100, 200)
cv2.imwrite("test/88_/background_canny.png", background)
```

最终输出的拼图和背景如下：

![alt text](/assets/img/slider_captcha/target_canny.png)

![alt text](/assets/img/slider_captcha/background_canny.png)

可以发现效果并不好，因为拼图是不规则图片，而图片都是矩形的方式，对拼图进行边缘检测容易受到非拼图区域的影响。

我们先忽略边缘检测方法中两个参数的影响，先处理拼图图片中非拼图区域对检测结果的影响，我们需要先对拼图进行边缘检测，因为其色彩差异很大，能够很好的区分出拼图和非拼图。然后我们需要在最终的输出结果中，去掉这个边缘。

```python

```
