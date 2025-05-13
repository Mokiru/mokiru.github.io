---
title: SpringBoot加载配置文件
date: 2025-05-14 00:00:00 +0800
categories: [Java, Spring Boot]
tags: [study]     # TAG names should always be lowercase
author: momochi
# authors: [xx,xx]
description: 介绍SpringBoot中灵活加载配置文件的方法
comments: true # 评论
pin: false # top 
math: true
---

## 目的

将properties或yaml配置文件中需求的配置加载为Map类型或其他制定的类型

## 实现

首先SpringBoot提供了将配置转为指定类型的方法，如下：

在SpringBoot2.x之前版本可以使用`org.springframework.boot.bind.RelaxedPropertyResolver`进行绑定，之后则使用`org.springframework.boot.context.properties.bind.Binder`。

以下主要以2.x之后版本为例，在`Binder`中主要使用`get`和`bind`方法，作用分别是根据`Environment`创建`Binder`对象和根据prefix字符串将配置转为指定类型。例如ShardingSphere中读取配置文件，对于每一个dataSource都需要指定url，password等等属性，在创建数据源时，会依次读取这些数据源的配置信息，例如某数据源名称为test，那么`prefix=test`，接着指定为Map类型，将会得到`{'url':'xxx', ....}`结果。

具体实现如下：

```java
String prefix = "spring.shardingsphere.datasource.";
String name = "md-configuration";
try {
    Class<?> binderClass = Class.forName("org.springframework.boot.context.properties.bind.Binder");
    Method getMethod = binderClass.getDeclaredMethod("get", Environment.class);
    Method bindMethod = binderClass.getDeclaredMethod("bind", String.class, Class.class);
    Object binder = getMethod.invoke((Object) null, environment);
    Object bindResult = bindMethod.invoke(binder, prefix + name, Map.class);
    Method resultGetMethod = bindResult.getClass().getDeclaredMethod("get");
    Map<String, Object> map = (Map) resultGetMethod.invoke(bindResult);
} catch (ClassNotFoundException | IllegalAccessException | InvocationTargetException | NoSuchMethodException e) {
    throw new RuntimeException(e);
}
```

`application.properties`文件中相应配置如下:

```properties
spring.shardingsphere.datasource.md-configuration.type= com.zaxxer.hikari.HikariDataSource
spring.shardingsphere.datasource.md-configuration.driver-class-name= com.mysql.cj.jdbc.Driver
spring.shardingsphere.datasource.md-configuration.jdbc-url= jdbc:mysql://xxxx:3306/md_configuration?serverTimezone=Asia/Shanghai&useLegacyDatetimeCode=false
spring.shardingsphere.datasource.md-configuration.username= xxxxx
spring.shardingsphere.datasource.md-configuration.password= xxxxx
```

最终得到的`map`结构如下:

```java
{
    "jdbc-url":"jdbc:mysql://xxxx:3306/md_configuration?serverTimezone=Asia/Shanghai&useLegacyDatetimeCode=false",
    "type":com.zaxxer.hikari.HikariDataSource,
    .....
}
```

以下将给出`ShardingSphere`中使用的相关绑定的工具类:

```java
public final class PropertyUtil {
    private static int springBootVersion = 1;

    public static boolean containPropertyPrefix(Environment environment, String prefix) {
        try {
            Map<String, Object> properties = (Map)(1 == springBootVersion ? v1(environment, prefix, false) : v2(environment, prefix, Map.class));
            return !properties.isEmpty();
        } catch (Exception var3) {
            return false;
        }
    }

    public static <T> T handle(Environment environment, String prefix, Class<T> targetClass) {
        switch (springBootVersion) {
            case 1:
                return (T)v1(environment, prefix, true);
            default:
                return (T)v2(environment, prefix, targetClass);
        }
    }

    private static Object v1(Environment environment, String prefix, boolean handlePlaceholder) {
        try {
            Class<?> resolverClass = Class.forName("org.springframework.boot.bind.RelaxedPropertyResolver");
            Constructor<?> resolverConstructor = resolverClass.getDeclaredConstructor(PropertyResolver.class);
            Method getSubPropertiesMethod = resolverClass.getDeclaredMethod("getSubProperties", String.class);
            Object resolverObject = resolverConstructor.newInstance(environment);
            String prefixParam = prefix.endsWith(".") ? prefix : prefix + ".";
            Method getPropertyMethod = resolverClass.getDeclaredMethod("getProperty", String.class);
            Map<String, Object> dataSourceProps = (Map)getSubPropertiesMethod.invoke(resolverObject, prefixParam);
            Map<String, Object> propertiesWithPlaceholderResolved = new HashMap();

            for(Map.Entry<String, Object> entry : dataSourceProps.entrySet()) {
                String key = (String)entry.getKey();
                Object value = entry.getValue();
                if (handlePlaceholder && value instanceof String && ((String)value).contains("${")) {
                    String resolvedValue = (String)getPropertyMethod.invoke(resolverObject, prefixParam + key);
                    propertiesWithPlaceholderResolved.put(key, resolvedValue);
                } else {
                    propertiesWithPlaceholderResolved.put(key, value);
                }
            }

            return Collections.unmodifiableMap(propertiesWithPlaceholderResolved);
        } catch (Throwable $ex) {
            throw $ex;
        }
    }

    private static Object v2(Environment environment, String prefix, Class<?> targetClass) {
        try {
            Class<?> binderClass = Class.forName("org.springframework.boot.context.properties.bind.Binder");
            Method getMethod = binderClass.getDeclaredMethod("get", Environment.class);
            Method bindMethod = binderClass.getDeclaredMethod("bind", String.class, Class.class);
            Object binderObject = getMethod.invoke((Object)null, environment);
            String prefixParam = prefix.endsWith(".") ? prefix.substring(0, prefix.length() - 1) : prefix;
            Object bindResultObject = bindMethod.invoke(binderObject, prefixParam, targetClass);
            Method resultGetMethod = bindResultObject.getClass().getDeclaredMethod("get");
            return resultGetMethod.invoke(bindResultObject);
        } catch (Throwable $ex) {
            throw $ex;
        }
    }

    private PropertyUtil() {
    }

    static {
        try {
            Class.forName("org.springframework.boot.bind.RelaxedPropertyResolver");
        } catch (ClassNotFoundException var1) {
            springBootVersion = 2;
        }

    }
}
```
