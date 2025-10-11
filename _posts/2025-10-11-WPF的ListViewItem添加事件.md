---
title: WPF中如何给ListViewItem添加事件
date: 2025-10-11 20:00:00 +0800
categories: [C#, WPF]
tags: [study]     # TAG names should always be lowercase
author: momochi
# authors: [xx,xx]
description: 介绍在WPF中如何给ListViewItem添加事件
comments: true # 评论
pin: false # top 
math: true
---

## ListViewItem

我们有时候需要给`ListView`中每一项添加事件，以下在MVVM下，给`Grid`添加鼠标左键双击事件。一般来说只要继承了`UIElement`都可以使用该方法。

```xaml
<ListView ItemsSource="{Binding Nodes}" 
        SelectedItem="{Binding DataContext.CurrentApp.SelectedFlow.SelectedNode, RelativeSource={RelativeSource AncestorType=Window}}"
        HorizontalContentAlignment="Stretch">
    <ListView.ItemTemplate>
        <DataTemplate>
            <Grid Margin="2">
                <Grid.ColumnDefinitions>
                    <ColumnDefinition Width="Auto"/>
                    <ColumnDefinition Width="*"/>
                    <ColumnDefinition Width="Auto"/>
                </Grid.ColumnDefinitions>
                <TextBlock Text="{Binding DisplayName}" Background="LightBlue" Padding="4 2" FontWeight="Bold"/>
                <TextBlock Grid.Column="1" Text="{Binding Title}" Margin="8,0,0,0"/>
                <StackPanel Grid.Column="2" Orientation="Horizontal">
                    <Button Content="↑" Command="{Binding DataContext.MoveNodeUpCommand, RelativeSource={RelativeSource AncestorType=Window}}" CommandParameter="{Binding}" Padding="4,0" Margin="2,0"/>
                    <Button Content="↓" Command="{Binding DataContext.MoveNodeDownCommand, RelativeSource={RelativeSource AncestorType=Window}}" CommandParameter="{Binding}" Padding="4,0" Margin="2,0"/>
                    <Button Content="×" Command="{Binding DataContext.RemoveSelectedNodeCommand, RelativeSource={RelativeSource AncestorType=Window}}" Padding="4,0" Margin="2,0"/>
                </StackPanel>
                <Grid.InputBindings>
                    <MouseBinding Command="{Binding DataContext.DoubleClickFlowNodeCommand, RelativeSource={RelativeSource AncestorType=Window}}" CommandParameter="{Binding}" MouseAction="LeftDoubleClick"/>
                </Grid.InputBindings>
            </Grid>
        </DataTemplate>
    </ListView.ItemTemplate>
</ListView>
```

主要看`Grid.InputBindings`。如此给ListViewItem添加了鼠标左键双击事件。