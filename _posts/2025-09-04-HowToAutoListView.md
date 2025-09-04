---
title: 如何让ListView中每列自适应
date: 2025-04-25 20:00:00 +0800
categories: [C#, WPF]
tags: [study]     # TAG names should always be lowercase
author: momochi
# authors: [xx,xx]
description: 介绍在WPF中如何让ListView中每列随着窗口变化而自适应大小
comments: true # 评论
pin: false # top 
math: true
---

使用Grid来定义一个网格布局：

```c#
<Grid Name="dummygrid" Visibility="Hidden" Margin="20,0,20,20">
    <Grid.ColumnDefinitions>
        <ColumnDefinition Width="0.4*"></ColumnDefinition>
        <ColumnDefinition Width="0.3*"></ColumnDefinition>
        <ColumnDefinition Width="0.3*"></ColumnDefinition>
    </Grid.ColumnDefinitions>
    <Border Grid.Column="0" Name="dummywidth1"></Border>
    <Border Grid.Column="1" Name="dummywidth2"></Border>
    <Border Grid.Column="2" Name="dummywidth3"></Border>
</Grid>
```

然后`ListView`每列可以绑定指定的`Border Name`

```c#
<ListView x:Name="listView" Grid.Row="1" Margin="20,0,20,20" ItemsSource="{Binding DataList}">
    <ListView.View>
        <GridView>
            <GridViewColumn  Header="xx" Width="{Binding ElementName=dummywidth1, Path=ActualWidth}"/>
            <GridViewColumn  Header="xxx" Width="{Binding ElementName=dummywidth2, Path=ActualWidth}"/>
            <GridViewColumn  Header="xxxx" Width="{Binding ElementName=dummywidth3, Path=ActualWidth}"/>
        </GridView>
    </ListView.View>
</ListView>
```
