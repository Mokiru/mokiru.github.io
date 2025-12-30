---
title: WPF中ListView流式分页
date: 2025-12-30 20:00:00 +0800
categories: [C#, WPF]
tags: [study]     # TAG names should always be lowercase
author: momochi
# authors: [xx,xx]
description: 介绍在ListView控件中，如何进行滚动翻页
comments: true # 评论
pin: false # top 
math: true
---

## ListView

`ListView`控件中内置了一个`ScrollViewer`，所以我们可以利用内置的`ScrollViewer`控件的事件来触发列表加载下一页：

```csharp
public static T GetChild<T>(DependencyObject d) where T : DependencyObject
{
    if (d == null)
    {
        return null;
    }
    if (d is T result)
    {
        return result;
    }
    for (int i = 0; i < VisualTreeHelper.GetChildrenCount(d); i++)
    {
        T child = GetChild<T>(VisualTreeHelper.GetChild(d, i));
        if (child != null)
        {
            return child;
        }
    }
    return null;
}
```

```csharp
ScrollViewer scrollViewer = VisualHelper.GetChild<ScrollViewer>(你的ListView);
scrollViewer.ScrollToTop();
scrollViewer.PreviewMouseWheel -= 函数;
scrollViewer.PreviewMouseWheel += 函数;
```

接着在这个`PreviewMouseWheel`调用的函数中来进行校验，例如判断当前是否滑动到底部了：

```csharp
private async void ScrollViewer_PreviewMouseWheel(object sender, MouseWheelEventArgs e)
{
    ScrollViewer scrollViewer = sender as ScrollViewer;
    if (e.Delta < 0 && scrollViewer.VerticalOffset == scrollViewer.ScrollableHeight)
    {
        await _viewModel.LoadNextPage();
    }
}
```