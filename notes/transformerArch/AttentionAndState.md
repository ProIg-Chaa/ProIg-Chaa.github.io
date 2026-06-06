---
date: 2026-06-06T09:35:24-07:00
updated: 2026-06-06T09:35:24-07:00
---

# 核函数为什么能把 Attention 写成状态方程

*By Chaa*

这一部分的核心问题是：

> 为什么使用可分解核函数后，attention 可以写成类似 RNN 的状态递推形式，而原版 softmax attention 不可以？

关键不在于“核函数”这个名字，而在于：

$$
\kappa(q,k)
$$

能不能被拆成：

$$
\kappa(q,k)
=
\phi(q)^\top \phi(k)
$$

如果能拆，而且 $\phi$ 是有限维特征映射，那么历史 key-value 信息就可以提前聚合成一个固定大小状态。

---

## 1. 状态方程需要什么条件

我们希望把 attention 写成：

$$
S_t
=
S_{t-1}
+
\text{update}(k_t,v_t)
$$

然后当前输出只依赖当前 query 和状态：

$$
y_t
=
\text{read}(q_t,S_t)
$$

这要求一个很关键的条件：

> 历史 token 写入状态时，不能依赖未来的 query。

也就是说，第 $j$ 个历史 token 写入状态时，只能用自己的：

$$
k_j,\ v_j
$$

不能等某个未来 query $q_i$ 出现之后，才知道这个历史 token 应该怎么参与计算。

因此，要想提前维护状态，attention 的相似度必须能拆成：

$$
\kappa(q_i,k_j)
=
\phi(q_i)^\top \phi(k_j)
$$

这样，和 query 有关的部分是：

$$
\phi(q_i)
$$

和历史 key/value 有关的部分是：

$$
\phi(k_j),\ v_j
$$

二者就可以分离。

---

## 2. Kernel Attention 的一般形式

把 attention 写成 kernel 形式：

$$
y_i
=
\frac{
\sum_{j=1}^{i}\kappa(q_i,k_j)v_j
}{
\sum_{j=1}^{i}\kappa(q_i,k_j)
}
$$

在 causal language modeling 中，第 $i$ 个 token 只能看前面和当前位置，所以求和范围是：

$$
j=1,\ldots,i
$$

如果 kernel 可以分解：

$$
\kappa(q_i,k_j)
=
\phi(q_i)^\top \phi(k_j)
$$

那么代入可得：

$$
y_i
=
\frac{
\sum_{j=1}^{i}
\phi(q_i)^\top \phi(k_j)v_j
}{
\sum_{j=1}^{i}
\phi(q_i)^\top \phi(k_j)
}
$$

因为：

$$
\phi(q_i)
$$

和求和下标 $j$ 无关，所以可以提出求和：

$$
y_i
=
\frac{
\phi(q_i)^\top
\left(
\sum_{j=1}^{i}\phi(k_j)v_j^\top
\right)
}{
\phi(q_i)^\top
\left(
\sum_{j=1}^{i}\phi(k_j)
\right)
}
$$

---

## 3. 状态形式从哪里来

定义状态矩阵：

$$
S_i
=
\sum_{j=1}^{i}\phi(k_j)v_j^\top
$$

定义归一化状态：

$$
z_i
=
\sum_{j=1}^{i}\phi(k_j)
$$

那么输出可以写成：

$$
y_i
=
\frac{
\phi(q_i)^\top S_i
}{
\phi(q_i)^\top z_i
}
$$

更重要的是，$S_i$ 和 $z_i$ 都可以递推维护：

$$
S_i
=
S_{i-1}
+
\phi(k_i)v_i^\top
$$

$$
z_i
=
z_{i-1}
+
\phi(k_i)
$$

这就是 linear attention 能写成状态方程的原因。

它本质上把历史信息压缩到了两个状态里：

$$
S_i
$$

和：

$$
z_i
$$

之后每个 query 只需要从状态中读取，而不需要重新扫描全部历史 token。

---

## 4. 为什么原版 Softmax Attention 不能这样写

原版 softmax attention 是：

$$
y_i
=
\frac{
\sum_{j=1}^{i}\exp(q_i^\top k_j)v_j
}{
\sum_{j=1}^{i}\exp(q_i^\top k_j)
}
$$

这里的相似度是：

$$
\kappa(q_i,k_j)
=
\exp(q_i^\top k_j)
$$

问题在于：

$$
\exp(q_i^\top k_j)
$$

把 $q_i$ 和 $k_j$ 强烈耦合在一起。

对于每一个新的 query $q_i$，都要重新计算：

$$
\exp(q_i^\top k_1),\ 
\exp(q_i^\top k_2),\ 
\ldots,\ 
\exp(q_i^\top k_i)
$$

然后才能得到分母：

$$
\sum_{\ell=1}^{i}\exp(q_i^\top k_\ell)
$$

也就是说，softmax 的每一行归一化分布都依赖当前 query 和所有历史 key 的比较结果。

所以原版 softmax attention 必须为每个 query 重新构造一行 attention distribution：

$$
[\alpha_{i1},\alpha_{i2},\ldots,\alpha_{ii}]
$$

这就无法提前把历史 key-value 精确压缩成一个固定大小状态。

---

## 5. 更准确地说：Softmax Kernel 可以拆，但需要无限维

需要注意，softmax 的非归一化 kernel：

$$
\exp(q^\top k)
$$

并不是完全不能写成内积形式。

理论上，它可以写成：

$$
\exp(q^\top k)
=
\phi(q)^\top \phi(k)
$$

但问题是，精确的 $\phi$ 通常是无限维的。

看一维情况最清楚。设 $q,k$ 都是标量：

$$
\exp(qk)
$$

泰勒展开为：

$$
\exp(qk)
=
\sum_{r=0}^{\infty}
\frac{(qk)^r}{r!}
$$

也就是：

$$
\exp(qk)
=
1
+
qk
+
\frac{q^2k^2}{2!}
+
\frac{q^3k^3}{3!}
+
\cdots
$$

这可以写成两个无限维向量的内积：

$$
\exp(qk)
=
\left[
1,\ q,\ \frac{q^2}{\sqrt{2!}},\ \frac{q^3}{\sqrt{3!}},\ldots
\right]^\top
\left[
1,\ k,\ \frac{k^2}{\sqrt{2!}},\ \frac{k^3}{\sqrt{3!}},\ldots
\right]
$$

所以：

$$
\phi(q)
=
\left[
1,\ q,\ \frac{q^2}{\sqrt{2!}},\ \frac{q^3}{\sqrt{3!}},\ldots
\right]
$$

这是无限维特征映射。

多维情况也是类似的，会包含所有阶数的交互特征：

$$
1,\quad q_a,\quad q_aq_b,\quad q_aq_bq_c,\quad \ldots
$$

也就是一阶、二阶、三阶，一直到无限阶。

---

## 6. 为什么无限维会导致无法状态化

linear attention 想维护的是有限大小状态：

$$
S_i
=
\sum_{j=1}^{i}\phi(k_j)v_j^\top
$$

如果：

$$
\phi(k_j)\in \mathbb{R}^{d_\phi}
$$

那么状态大小是：

$$
S_i \in \mathbb{R}^{d_\phi \times d_v}
$$

只要 $d_\phi$ 固定，状态大小就不随上下文长度增长。

但如果 softmax kernel 的精确 $\phi$ 是无限维，那么状态会变成：

$$
S_i \in \mathbb{R}^{\infty \times d_v}
$$

这在实际计算中不可行。

所以原版 softmax attention 不能被精确地写成可计算的有限维状态方程。

---

## 7. Linear Attention 实际做了什么

实际 linear attention 通常有两种路线。

第一种是直接换一个有限维可分解 kernel：

$$
\kappa(q,k)
=
\phi(q)^\top \phi(k)
$$

其中：

$$
\phi(q),\phi(k)\in\mathbb{R}^{d_\phi}
$$

这样可以精确得到状态递推：

$$
S_i
=
S_{i-1}
+
\phi(k_i)v_i^\top
$$

$$
z_i
=
z_{i-1}
+
\phi(k_i)
$$

$$
y_i
=
\frac{
\phi(q_i)^\top S_i
}{
\phi(q_i)^\top z_i
}
$$

第二种是近似 softmax kernel：

$$
\exp(q^\top k)
\approx
\phi(q)^\top \phi(k)
$$

其中 $\phi$ 是有限维的近似特征映射。

这种方法不是和原版 softmax 完全等价，而是在做近似。

---

## 8. 核函数和状态方程的关系

可以把整个逻辑压缩成三步。

第一步，选择一个可分解 kernel：

$$
\kappa(q,k)
=
\phi(q)^\top \phi(k)
$$

第二步，把历史 key-value 聚合成状态：

$$
S_i
=
\sum_{j=1}^{i}\phi(k_j)v_j^\top
$$

$$
z_i
=
\sum_{j=1}^{i}\phi(k_j)
$$

第三步，用当前 query 从状态中读取：

$$
y_i
=
\frac{
\phi(q_i)^\top S_i
}{
\phi(q_i)^\top z_i
}
$$

因此，kernel 让 attention 状态化的根本原因是：

> 它把 query 部分和 history 部分拆开了。

---

## 9. 与原版 Softmax 的本质区别

原版 softmax attention：

$$
y_i
=
\frac{
\sum_{j=1}^{i}\exp(q_i^\top k_j)v_j
}{
\sum_{j=1}^{i}\exp(q_i^\top k_j)
}
$$

它的权重必须在当前 query 出现后，通过当前 query 与所有历史 key 的比较来决定。

可分解 kernel attention：

$$
y_i
=
\frac{
\phi(q_i)^\top
\left(
\sum_{j=1}^{i}\phi(k_j)v_j^\top
\right)
}{
\phi(q_i)^\top
\left(
\sum_{j=1}^{i}\phi(k_j)
\right)
}
$$

它可以把所有只和历史有关的部分提前累积。

所以二者的差别是：

$$
\text{softmax attention}
=
\text{query-specific full scan}
$$

$$
\text{linear attention}
=
\text{state update + state read}
$$

---

## 10. 最终总结

简单来说：

$$
\exp(q^\top k)
$$

可以被看成 kernel 点积，但精确分解通常需要无限维特征：

$$
\exp(q^\top k)
=
\phi(q)^\top \phi(k),
\qquad
\phi \text{ is infinite-dimensional}
$$

而 linear attention 需要的是有限维状态：

$$
S_t
\in
\mathbb{R}^{d_\phi \times d_v}
$$

所以原版 softmax attention 不能被精确地转化为可计算的有限维状态方程。

linear attention 的做法是：

$$
\exp(q^\top k)
\approx
\phi(q)^\top \phi(k)
$$

或者直接换用一个有限维可分解 kernel。

这带来的结果是：

$$
S_t
=
S_{t-1}
+
\phi(k_t)v_t^\top
$$

$$
z_t
=
z_{t-1}
+
\phi(k_t)
$$

$$
y_t
=
\frac{
\phi(q_t)^\top S_t
}{
\phi(q_t)^\top z_t
}
$$

因此，linear attention 能状态化，不是因为“核函数”这个概念本身神奇，而是因为它使用了有限维可分解的特征映射，把 query-dependent 的相似度计算拆成了：

$$
\text{query part}
\times
\text{history part}
$$

从而让历史部分可以提前累积成状态。
