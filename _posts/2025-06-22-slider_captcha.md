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




