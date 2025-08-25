---
title: HyperLogLog-估计流中元素数量(去重)
date: 2025-04-09 21:30:00 +0800
categories: [Math, 元素数量统计]
tags: [study]     # TAG names should always be lowercase
author: momochi
# authors: [xx,xx]
description: HyperLogLog介绍
comments: true # 评论
pin: false # top 
math: true
---

## 简介&背景

HyperLogLog是一种高效算法，用于估算大型数据集合中不同元素的数量，这种数据集合被称为多重集，通常是海量数据流(只读序列)。显然，多重集的基数可以通过存储复杂度与元素数量基本成比例的算法精确计算，然而在大多数实际应用中，待处理的多重集规模很大，根本无法完全存入核心内存，此时关键思路是放宽对基数n进行精确计算的要求，转而开发专门用于近似估计n的概率算法，目前已经发展出多种仅需亚线性内存的算法，或最坏情况下仅需较小常数的线性内存算法。

所有已知的高效基数估计器都依赖于随机化技术，这通过哈希函数实现，假设待统计元素属于数据域$$D$$，我们给定哈希函数:$$h:D\rightarrow \{0,1\}^\infty$$，即将哈希值视作 $$\{0,1\}^\infty$$ 的无限二进制串，或等价于单位区间内的实数(实际应用中，32位哈希足以估计超过$$10^9$$的基数)，我们要求哈希函数的设计满足：哈希值能够高度模拟均匀随机模型，即哈希值的各个比特位相互独立且出现概率均为 $$\frac{1}{2}$$

最著名的基数估计器依赖于对输入多重集 $$M$$的哈希值 $$h(M)$$进行适当且简洁的观测，进而推断未知基数 $$n$$的合理估计值，对于由 $$\{0,1\}^\infty$$ 字符串构成的多重集(或者，$$[0,1]$$实数集) $$S\equiv h(M)$$，定义其可观测量为仅取决于 $$S$$底层集合的函数（即与元素重复无关的量）目前已研究的两大类基数可观测量包括：

- 位模式可观测量
- 顺序统计可观测量

### 位模式可观察量

位模式可观察量是基于哈希值的二进制表示中的特定位模式。具体来说，它关注的是哈希值的二进制表示中 "0" 的连续序列长度。例如，假设我们有一个哈希函数 $$h$$，它将数据元素映射到一个无限长的二进制字符串。对于一个哈希值 $$x$$，我们定义 $$\rho (x)$$ 为 $$x$$ 中最左边的 "1" 出现的位置 (从 $$1$$ 开始计数)。如果 $$\rho (x)=k$$，这意味着 $$x$$ 的前 $$k-1$$ 位都是 "0"，而第 $$k$$ 位是 "1"。

例如：$$ x=00101101...$$ 则 $$\rho (x)=3$$，$$x=101101...$$ 则 $$\rho (x)=1$$

使用位模式可观察量的目的是通过观察哈希值中 "0" 的连续序列的长度来估计基数。具体来说，如果一个数据集合中有 $$n$$个不同的元素，那么这些元素的哈希值中，最左边的 "1" 出现的位置 $$\rho (x)$$ 的最大值 $$max\ \rho (x)$$可以用来估计 $$n$$

假设数据集合中的元素是均匀分布的，那么对于一个随机的哈希值， $$\rho (x)=k$$ 的概率大约是 $$2^{-k}$$。因此，如果 $$max\ \rho(x)=k$$，我们可以估计基数 $$n$$ 大约为 $$2^{k}$$。这种方法的优点是计算简单，只需要观察哈希值的前几位即可。

例子：假设我们有一个数据集合，其哈希值的 $$\rho (x)$$ 分布如下：
- \$$\rho(x_1)=3$$
- \$$\rho(x_2)=5$$
- \$$\rho(x_3)=2$$
- \$$\rho(x_4)=4$$

那么 $$max \ \rho(x)=5$$，我们可以估计基数 $$n$$大约为 $$2^5=32$$

### 顺序统计可观察量

顺序统计可观察量是基于哈希值的顺序统计特性，例如哈希值中的最小值。假设我们有一个哈希函数 $$h$$，它将数据元素映射到单位区间 $$[0,1]$$ 内的实数，对于一个数据集合 $$M$$ ，我们定义 $$X=min\{h(x)\vert x\in M\}$$ 为哈希值中的最小值

使用顺序统计可观察量的目的是通过观察哈希值中的最小值来估计基数。具体来说，如果一个数据集合中有 $$n$$ 个不同的元素，那么这些元素的哈希值中的最小值 $$X$$ 可以用来估计 $$n$$

假设哈希值是均匀分布的，那么 $$X$$ 的期望值 $$E(X)$$ 大约为 $$\frac{1}{n+1}$$。因此，如果观察到 $$X$$ 的值为 $$x$$，我们可以估算基数 $$n$$大约为 $$\frac{1}{x} - 1$$。这种方法的优点是它利用了哈希值的顺序特性，可以提供更准确的估计。

## 总结

这些可观察量可以通过一个或几个寄存器来维护。然而它们本身只能提供对所求基数 $$n$$ 的粗略估计，通过 $$log_{2}n$$ 或 $$\frac{1}{n}$$ 来表示。一个困难是由于较高的变异性，因此单次观察(对应于维护单个变量)不足以获得准确的预测，一个直接的想法是进行多次实验：如果 $$m$$ 个随机变量的每个标准差为 $$\sigma$$，则它们的算术平均值的标准差为 $$\frac{\sigma}{\sqrt{m}}$$，通过增加 $$m$$ 可以使其任意小。然而，这种简单策略有两个主要缺点：计算成本高(我们需要为每个扫描的元素计算 $$m$$ 个哈希值)，而且糟糕的是它需要大量独立的哈希函数，而目前没有已知的构造方法。于是引入随机平均方法解决这一问题，它通过单个哈希函数来模拟 $$m$$ 次实验的效果。简而言之，我们将输入流 $$h(M)$$ 分成 $$m$$ 个子流，对应于将单位区间内的哈希值划分为 $$[\frac{1}{m}, \frac{2}{m}[,...,[\frac{m-1}{m}, 1]$$，然后我们维护每个子流对应的 $$m$$ 个可观察量 $$O_1,...,O_m$$ 然后，对 $$\{O_j\}$$ 进行适当的平均，期望产生一个随着 $$m$$ 增加而质量提高(由于平均效应)的基数估计值。这种方法的好处是，它只需要对多重集 $$M$$的每个元素执行常数数量的基本操作(而不是与 $$m$$ 成比例的数量)，并且现在只需要一个哈希函数

## HyperLogLog

基于 LOGLOG相同的可观察量，即最大 $$\rho$$ 值，其中 $$\rho(x)$$ 是二进制字符串 $$x$$ 中最左边的 1 的位置。它使用随机平均，但与标准算法不同的是，它的评估函数基于调和平均值，而标准算法使用的是几何平均值。

$$
\begin{aligned}
&Let\ h:D \rightarrow [0,1] \equiv \{0,1\}^{\infty} hash\ data\ from \ domain\ D\ to\ the\ binary\ domain.\\
&Let\ \rho(s),for s\in \{0,1\}^{\infty},be\ the\ position\ of\ the\ leftmost\ 1-bit\ (\rho(0001...)=4).\\
&Algorithm\ HyperLogLog\ (input M:multiset \ of \ items\ from\ domain\ D).\\
&assume\ m=2^b\ with\ b\in Z\gt 0;\\
&initialize\ a\ collection\ of\ m\ reisters,M[1],...,M[m],to\ -\infty;\\
&for\ v\in M\ do:\\
& \ \ \ \ \ \ set\ x:= h(v);\\
& \ \ \ \ \ \ set\ j=1+\langle x_1x_2...x_b \rangle _{2}; \set{the\ binary\ address\ determined\ by\ the\ first\ b\ bits\ of\ x}\\
& \ \ \ \ \ \ set\ \omega := x_{b+1}x_{b+2}...; set\ M[j] := max(M[j], \rho(\omega));\\
&compute\ Z:=(\sum_{j=1}^{m}2^{-M[j]});\set{the\ “indicator”\ function}\\
&return\ E:=\alpha_{m}m^2Z\
\end{aligned}
$$

输入一个多重集 $$M$$(即顺序读取数据流)，输出为 $$M$$ 中不同元素的数量，给定一个字符串 $$ S\in \set{0,1}^{\infty}$$，令 $$\rho(s)$$ 代表最左边的 $$1$$ 的位置，将流 $$M$$ 分为子流 $$M_1,...,M_m$$，基于哈希值的前 $$b$$位计算，其中 $$m=2^b$$ ，每个子流独立处理，对于 $$ N\equiv M_j$$ 这样的子流(被视为由去掉了初始 $$b$$ 位的哈希值组成)，相应的观测值为:

$$
Max(N):=\max_{x\in N}\rho(x)
$$

通常 $$Max(\emptyset)=-\infty$$。$$M(j)=Max(M_j)$$ ，当所有的元素都被遍历完，这个算法的指标为：

$$
Z:=(\sum_{j=1}^{m}2^{-M[j]})
$$

然后，会以下面的形式返回 $$2^{M(j)}$$ 的调和平均值

$$
E:=\alpha_{m}m^2Z,with \ \alpha:=(m\int_{0}^{\infty}(log_{2}(\frac{2+u}{1+u}))^{m}du)^{-1}.
$$

## Discussion

$$
\begin{aligned}
&Let\ h\ :\ D\rightarrow\set{0,1}^{32}\ hash\ data\ from\ D\ to\ binary\ 32-bit\ words.\\
&Let\ \rho(s)\ be\ the\ position\ of\ the\ leftmost\ 1-bit\ of\ s\ :\ e.g.\ \rho(1...)=1,\rho(0001...)=4,\rho(0^{K})=K+1.\\
&define\ \alpha_{16}=0.673;\ \alpha_{32}=0.697;\ \alpha_{64}=0.709;\ \alpha_{m}=0.7213/(1+1.079/m)\ for\ m\geq 128;\\
&Program\ HYPERLOGLOG\ (input\ M:multiset\ of\ items\ from\ domain\ D).\\
&initialize\ a\ collection\ of\ m\ registers,\ M[1],...,M[m],to\ 0;\\
\\
&for\ v\in M\ do\\
&\ \ \ \ set\ x:= \ h(v);\\
&\ \ \ \ set\ j=1+(x_{1}x_{2}...x_{b})_{2}; \ \ \ \ \set{the\ binary\ address\ determined\ by\ the\ first\ b\ bits\ of\ x}\\
&\ \ \ \ set\ \omega:=x_{b+1}x_{b+2}...;\\
&\ \ \ \ set\ M[j]:=\max(M[j],\rho(\omega));\\
&compute\ E:=\alpha_{m}m^{2}\sdot(\sum_{j=1}^{m}2^{-M[j]})^{-1};\ \ \ \ \set{the\ “row”\ HyperLogLog\ estimate}
\\
&if\ E\leq \frac{5}{2}m\ then\\
&\ \ \ \ let\ V\ be\ the\ number\ of\ registers\ equal\ to\ 0;\\
&\ \ \ \ if\ V\neq 0\ then\ set\ E^{*}:=mlog(m/V)\ else\ set\ E^{*}:=E;\ \ \ \ \set{small\ range\ correction}\\
&if\ E\leq\frac{1}{30}2^{32}\ then\\
&\ \ \ \ set\ E^{*}:=E;\ \ \ \ \set{intermediate\ range-no\ correction}\\
&if\ E\gt\frac{1}{30}2^{32}\ then\\
&\ \ \ \ set\ E^{*}:=-2^{32}log(1-E/2^{32});\ \ \ \ \set{large\ range\ correction}\\
&return\ candinality\ estimate\ E^{*}\ with\ typical\ relative\ error\pm 1.04/\sqrt{m}.
\end{aligned}
$$


以下是一个`HyperLogLog`的`Java`实现

```java
import java.util.Arrays;

public class HyperLogLog {
    private final int[] registers;
    private final int m;
    private final double alphaMM;

    public HyperLogLog(int b) {
        if (b < 4 || b > 16) {
            throw new IllegalArgumentException("b must be in the range [4, 16]");
        }
        this.m = 1 << b; // m = 2^b
        this.registers = new int[m];
        Arrays.fill(registers, 0);
        this.alphaMM = getAlphaMM(m);
    }

    private double getAlphaMM(int m) {
        switch (m) {
            case 16:
                return 0.673 * m * m;
            case 32:
                return 0.697 * m * m;
            case 64:
                return 0.709 * m * m;
            default:
                return (0.7213 / (1 + 1.079 / m)) * m * m;
        }
    }

    private int rho(long value, int maxBits) {
        int position = 1;
        while ((value & 1) == 0 && position <= maxBits) {
            value >>>= 1;
            position++;
        }
        return position;
    }

    public void add(long hash) {
        int index = (int) (hash >>> (Long.SIZE - Integer.numberOfTrailingZeros(m)));
        long w = hash & ((1L << (Long.SIZE - Integer.numberOfTrailingZeros(m))) - 1);
        registers[index] = Math.max(registers[index], rho(w, Long.SIZE - Integer.numberOfTrailingZeros(m)));
    }

    public double estimate() {
        double z = 0.0;
        int zeroCount = 0;

        for (int register : registers) {
            z += 1.0 / (1 << register);
            if (register == 0) {
                zeroCount++;
            }
        }

        double estimate = alphaMM / z;

        if (estimate <= 2.5 * m) {
            // Small range correction
            return zeroCount > 0 ? m * Math.log((double) m / zeroCount) : estimate;
        } else if (estimate > (1.0 / 30.0) * (1L << 32)) {
            // Large range correction
            return -(1L << 32) * Math.log(1 - (estimate / (1L << 32)));
        } else {
            // No correction
            return estimate;
        }
    }

    public static void main(String[] args) {
        HyperLogLog hll = new HyperLogLog(10); // b = 10, m = 2^10 = 1024
        for (int i = 1; i <= 1000000; i++) {
            hll.add(hash(i));
        }
        System.out.println("Estimated cardinality: " + hll.estimate());
    }

    private static long hash(int value) {
        // A simple hash function (can be replaced with a better one)
        return value * 0x9E3779B97F4A7C15L;
    }
}
```