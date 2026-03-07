---
title: Python装饰器
date: 2026-03-07 20:00:00 +0800
categories: [Python]
tags: [study]     # TAG names should always be lowercase
author: momochi
# authors: [xx,xx]
comments: true # 评论
pin: false # top 
math: true
---


## 简介

装饰器是修改其他函数的功能的函数，有助于让代码更简短，类似Java代理方法。

## 一切皆对象

我们可以将定义的函数赋值给变量，如下：

```python
def a():
    print('a')

b = a
b()
# output a
```

我们也可以在函数中定义函数：

```python
def a():
    def b():
        print('b')
    print('a')
    b()
a()
# output a b
```

那么由以上两点，我们可以在函数中定义一个函数，并将这个函数作为返回值返回：

```python
def a():
    def b():
        print('b')

    return b
x = a()
x()
# output b
```

上面相当于将一个函数赋值给x，我们也可以将函数作为参数传递，并在函数内部使用：

```python
def test():
    print('test')

def a(func):
    print('pre')
    func()
    print('after')
a(test)

# output pre test after
```

上面这个虽然实现了“代理”的功能，从调用函数名来看和原方法名是不一样的，所以我们可以内部定义一个函数封装，然后将这个封装函数返回：

```python
def test():
    print('test')

def a(func):
    def b():
        print('pre')
        func()
        print('after')
    return b

x = a(test)
x()
# output pre test after
```

但是，一般函数都有参数，所以我们可以使用可变参数接收：

```python
def test(c):
    print(c)

def a(func):
    def b(*args, **kwargs):
        print('pre')
        func(*args, **kwargs)
        print('after')
    return b

x = a(test)
x(1)
# output pre 1 after
```

Python中，我们可以将最后的赋值操作使用`@`来代替，如下(注意定义函数顺序)：

```python
def a(func):
    def b(*args, **kwargs):
        print('pre')
        func(*args, **kwargs)
        print('after')
    return b

@a
def test(c):
    print(c)

test(1)
# output pre 1 after
```
