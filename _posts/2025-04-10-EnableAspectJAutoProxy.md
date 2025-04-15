---
title: EnableAspectJAutoProxy使用
date: 2025-04-10 21:30:00 +0800
categories: [Java, Spring Boot]
tags: [study]     # TAG names should always be lowercase
author: momochi
# authors: [xx,xx]
description: EnableAspectJAutoProxy介绍
comments: true # 评论
pin: false # top 
math: true
---

## 简介

如果遇到自身需要调用内部方法，并且需要代理生效的情况下，首先`this.xxx`的调用不是代理对象调用，而是原对象调用，这样AOP将不会生效，于是我们需要在内部调用代理对象的方法，于是`EnableAspectJAutoProxy`就能解决这种问题

## 使用


### Configuration指定

```java
@Configuration
@EnableAspectJAutoProxy(exposeProxy = true, proxyTargetClass = true)
@ComponentScan("扫描包")
public class TestAspectJAutoConfig {
}
```

这样便能让指定包下的Component启动，然后我们可以看见这个注解有两个参数：
- exposeProxy:true则使用CGLIB代理
- proxyTargetClass:true则让AOP框架将代理公开，这样便能通过`AopContext.currentProxy()`获取代理对象

### Component直接指定

上面可以一次性指定多个Component，现在可以直接在Component直接加上该注解，则可以直接指定当前Component

```java
@Component
public class TestAutoAspectJ {

    @XXXX
    public void aspectIn() {

    }

    public void testProxy() {
        Object proxy = AopContext.currentProxy();
        System.out.println(proxy);
    }
}
```

## 注意

以上一切方法都需要基于原本的类会被Spring代理，而不是创建原始对象。因此可以如上，在目标类中加入AOP，这样类便会被代理，于是`AopContext`才能获取当前类的代理对象，然后才能进行方法调用，否则将会出现上一个被代理对象，或者在类型转换时出现ClassCastException异常。