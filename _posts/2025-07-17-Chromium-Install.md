---
title: Chromium 下载
date: 2025-07-14 10:00:00 +0800
categories: [C++, 浏览器]
tags: [impl]     # TAG names should always be lowercase
author: momochi
# authors: [xx,xx]
description: Chromium install 
comments: true # 评论
pin: false # top 
math: true
toc: true
content: true
---

## Chromium版本说明

该篇使用Chromium版本为140.0.7300.1，注意在下载过程中需要在命令行中设置代理，假设某vpn代理端口为5555。那么应在命令行中输入如下设置：

```shell
set http_proxy=http://127.0.0.1:5555
set https_proxy=http://127.0.0.1:5555
```

## git 下载

```shell
git version
```

查看`git`版本，保证在`2.16.1`或更新的版本。

然后在命令行中进行如下设置，其中名称等自定义填写：

```shell
git config --global user.name "My Name"
git config --global user.email "my-name@chromium.org"
git config --global core.autocrlf false
git config --global core.filemode false
git config --global core.preloadindex true
git config --global core.fscache true
git config --global branch.autosetuprebase always
git config --global core.longpaths true
```

## depot_tools下载

这里假设将会把`depot_tools`下载在`D:\src`中。

几乎整个流程需要使用`vpn`

```shell
cd /d D:\src
git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git
```

接着将`D:\src\depot_tools`加入系统环境变量`Path`中。并将其优先级设置为最高。

## Visual Studio 2022下载

下载Visual Studio后，在安装界面选择：

![alt text](/assets/img/chromium/vs_install.png)

在系统环境中新增环境变量：`vs2022_install=vsstudio安装的地方`

## Windows11 SDK

选择版本为 10.0.26100.3323版本。

[Windows11 SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/)

在安装界面，一定要选择上Debugging Tools。

然后设置环境变量：`WINDOWSSDKDIR=C:\Program Files (x86)\Windows Kits\10`，路径选择自己的即可。

## 拉取源码

先进入下载源码的文件夹中。

例如：

```shell
mkdir chromium && cd chromium
```

然后使用`fetch`拉取源码，可以使用`--no-history`只拉取最新版本。

```shell
fetch --no-history chromium
```

在拉取过程中，可以看见循环`still working on src`。等待拉取结束即可。

如果中途出现错误，或其他原因导致的中断下载，可以进入当前目录的`src`文件夹，然后打开命令行输入：

```shell
gclient sync
```

在接下来的步骤中，需要进入`src`文件夹中。以下假设都以`src`为当前位置。

## 构建

先要配置并生成Ninja构建文件，以下为最小构建：

```shell
gn gen out\Default --args="is_component_build = true is_debug = false enable_nacl = false  blink_symbol_level = 0 v8_symbol_level = 0 symbol_level = 0"
```

- `is_component_build=true`：这使用了更多、更小的dll，并且可以避免在每次更改后重新链接chrome.dll。
- `enable_nacl=false`：这将禁用本地构建通常不需要的本机客户端。
- `blink_symbol_level=0`：关闭blink的源代码级调试以减少构建时间，如果您不打算调试blink，这是合适的。
- `v8_symbol_level = 0`：关闭v8的源代码级调试以减少构建时间，如果你不打算调试v8，这是合适的。
- `symbol_level=0`：减少了编译器和链接器必须做的工作。

然后使用Ninja编译Chromium，生成可执行的浏览器chrome.exe

```shell
autoninja -C out\Default chrome
```

或者设置最大并行任务数来限制编译过程中开启的并行任务数量，防止因为内存不足导致的编译中断。以下为开启并行任务数为4的示例：

```shell
autoninja -C out\Default chrome -j4
```

## 生成安装包

```shell
$ autoninja -C out\Default mini_installer
```
