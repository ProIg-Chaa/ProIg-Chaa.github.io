---
date: 2026-06-02
updated: 2026-06-02
---

# 线性注意力：从 Softmax Attention 到 Kernel Attention 的本质变化

线性注意力不是“把标准 attention 免费等价加速”，而是把标准 attention 的记忆机制从：每个 query 显式检索所有历史 token

改成了：
先把历史 key-value 压缩进一个固定大小状态，再让 query 从这个状态中读取

这个变化带来了线性复杂度，也带来了精确检索能力的损失。

---

## 1. 标准 Attention 在做什么

标准 attention 可以写成：

$$
\operatorname{Attn}(Q,K,V)
=
\operatorname{softmax}(QK^\top)V
$$

设序列长度为：

$$
n
$$

单个 attention head 的 key/query 维度为：

$$
d_k
$$

value 维度为：

$$
d_v
$$

则有：

$$
Q \in \mathbb{R}^{n \times d_k}
$$

$$
K \in \mathbb{R}^{n \times d_k}
$$

$$
V \in \mathbb{R}^{n \times d_v}
$$

标准 attention 会先计算：

$$
QK^\top
$$

其形状是：

$$
QK^\top \in \mathbb{R}^{n \times n}
$$

这个矩阵的含义是：

> 每一个 query token，都和每一个 key token 算一次相似度。

第 $i$ 行表示第 $i$ 个 query 对所有 key 的打分：

$$
[q_i^\top k_1,\ q_i^\top k_2,\ \ldots,\ q_i^\top k_n]
$$

经过 softmax 后得到注意力权重：

$$
[\alpha_{i1},\alpha_{i2},\ldots,\alpha_{in}]
$$

然后输出：

$$
y_i
=
\sum_{j=1}^{n}\alpha_{ij}v_j
$$

所以标准 attention 的核心是：

> 每个 token 都有一次独立的全文检索机会。

这也是它强大的地方。第 $i$ 个 token 可以决定自己主要读取第 10 个 token、第 500 个 token，还是第 10000 个 token。

---

## 2. 标准 Attention 为什么是二次复杂度

标准 attention 的主要成本来自：

$$
QK^\top
$$

因为：

$$
Q \in \mathbb{R}^{n \times d_k}
$$

$$
K^\top \in \mathbb{R}^{d_k \times n}
$$

所以：

$$
QK^\top \in \mathbb{R}^{n \times n}
$$

计算这个矩阵需要：

$$
O(n^2d_k)
$$

然后还要计算：

$$
\operatorname{softmax}(QK^\top)V
$$

其中：

$$
\operatorname{softmax}(QK^\top) \in \mathbb{R}^{n \times n}
$$

$$
V \in \mathbb{R}^{n \times d_v}
$$

这一步复杂度为：

$$
O(n^2d_v)
$$

所以标准 attention 的总复杂度大约是：

$$
O(n^2d_k+n^2d_v)
$$

当 $n$ 很大时，瓶颈来自：

$$
n^2
$$

这不是实现问题，而是标准 attention 的能力本身就依赖一个 token-token 关系矩阵。

---

## 3. 如果没有 Softmax，确实可以换括号

先暂时忽略 softmax，只看：

$$
QK^\top V
$$

矩阵乘法满足结合律：

$$
(QK^\top)V
=
Q(K^\top V)
$$

所以从代数结果上看，两者完全一样。

但是计算路径不同。

原始路径是先算：

$$
QK^\top
$$

得到：

$$
n \times n
$$

然后再乘 $V$。

另一种路径是先算：

$$
K^\top V
$$

其中：

$$
K^\top \in \mathbb{R}^{d_k \times n}
$$

$$
V \in \mathbb{R}^{n \times d_v}
$$

所以：

$$
K^\top V \in \mathbb{R}^{d_k \times d_v}
$$

这个中间矩阵的大小不再依赖 $n^2$。

然后再算：

$$
Q(K^\top V)
$$

其中：

$$
Q \in \mathbb{R}^{n \times d_k}
$$

$$
K^\top V \in \mathbb{R}^{d_k \times d_v}
$$

复杂度大约是：

$$
O(nd_kd_v)
$$

如果 $d_k$ 和 $d_v$ 是固定 head dimension，比如 64 或 128，那么复杂度对序列长度 $n$ 是线性的。

所以这里的关键不是代数结果变了，而是：

> 括号改变了中间对象的形状，从 $n \times n$ 变成了 $d_k \times d_v$。

---

## 4. 但标准 Softmax Attention 不能直接换括号

标准 attention 是：

$$
\operatorname{softmax}(QK^\top)V
$$

不能改成：

$$
Q(K^\top V)
$$

原因是 softmax 不是线性操作。

对第 $i$ 个 query，标准 attention 是：

$$
y_i
=
\sum_{j=1}^{n}
\frac{
\exp(q_i^\top k_j)
}{
\sum_{\ell=1}^{n}\exp(q_i^\top k_\ell)
}
v_j
$$

注意分母：

$$
\sum_{\ell=1}^{n}\exp(q_i^\top k_\ell)
$$

这个分母依赖当前 query：

$$
q_i
$$

也依赖所有 key：

$$
k_1,k_2,\ldots,k_n
$$

也就是说，每个 query 都有自己的一套归一化分布。

第 1 个 query 的 softmax 分布和第 2 个 query 的 softmax 分布不同。  
第 $i$ 个 query 必须先和所有 key 比较，才能得到自己的读取权重。

因此，softmax 阻止了我们提前把所有 $K$ 和 $V$ 合并成一个统一状态。

这就是标准 attention 很强但很贵的核心原因：

> 每个 query 都要现场构造一行长度为 $n$ 的检索分布。

---

## 5. QK 内积已经是相似度，Softmax 还有什么意义？

QK 内积当然可以作为相似度：

$$
s_{ij}
=
q_i^\top k_j
$$

它表示第 $i$ 个 query 和第 $j$ 个 key 的匹配程度。

但是，只有相似度分数还不够。最后 attention 要用这些分数去混合 value：

$$
y_i
=
\sum_{j=1}^{n} w_{ij}v_j
$$

如果直接令：

$$
w_{ij}
=
q_i^\top k_j
$$

那么会出现几个问题。

---

## 6. Softmax 的意义一：把分数变成非负权重

内积可以是负数：

$$
q_i^\top k_j < 0
$$

如果直接用内积作为权重，那么某些 value 会被负权重乘上：

$$
y_i
=
\sum_{j=1}^{n}(q_i^\top k_j)v_j
$$

这不再像“读取信息”，而更像任意线性组合。

softmax 之后：

$$
\alpha_{ij}
=
\frac{
\exp(q_i^\top k_j)
}{
\sum_{\ell=1}^{n}\exp(q_i^\top k_\ell)
}
$$

一定满足：

$$
\alpha_{ij} > 0
$$

所以 softmax 的第一个作用是：

> 把任意相似度分数变成非负读取权重。

---

## 7. Softmax 的意义二：控制输出尺度

如果直接用内积权重：

$$
y_i
=
\sum_{j=1}^{n}(q_i^\top k_j)v_j
$$

随着上下文长度 $n$ 增加，求和项变多，输出尺度可能系统性变大。

softmax 会保证：

$$
\sum_{j=1}^{n}\alpha_{ij}
=
1
$$

因此输出是 value 的加权平均：

$$
y_i
=
\sum_{j=1}^{n}\alpha_{ij}v_j
$$

这让输出尺度更加稳定。

所以 softmax 的第二个作用是：

> 把相似度分数归一化成总和为 1 的读取分布。

这不是简单的“数值稳定”。  
数值稳定通常指计算 softmax 时减去最大值：

$$
\operatorname{softmax}(s_i)
=
\operatorname{softmax}(s_i-\max_j s_{ij})
$$

而 softmax 本身的归一化意义要更根本。

---

## 8. Softmax 的意义三：引入竞争

softmax 权重是：

$$
\alpha_{ij}
=
\frac{
\exp(q_i^\top k_j)
}{
\sum_{\ell=1}^{n}\exp(q_i^\top k_\ell)
}
$$

注意分母包含所有 key：

$$
\sum_{\ell=1}^{n}\exp(q_i^\top k_\ell)
$$

这意味着第 $j$ 个 key 的权重不只取决于它和 query 有多像，还取决于其他 key 和 query 有多像。

所以 softmax 问的不是：

> 这个 key 像不像 query？

而是：

> 在所有 key 里面，这个 key 相对来说是不是最像 query？

这就是竞争。

例如分数为：

$$
[10,\ 9,\ 1]
$$

第一个分数虽然最高，但第二个分数也很高，所以第一个不会独占全部权重。

如果分数是：

$$
[10,\ 1,\ 1]
$$

第一个就会拿走绝大部分权重。

所以 softmax 的第三个作用是：

> 把绝对相似度变成相对竞争式的读取分布。

---

## 9. Softmax 的意义四：制造尖锐选择

softmax 里有指数函数：

$$
\exp(q_i^\top k_j)
$$

指数会放大分数差异。

如果两个分数分别是：

$$
s_1 = 10
$$

$$
s_2 = 8
$$

它们只差 2。

但指数之后的比例是：

$$
\frac{\exp(10)}{\exp(8)}
=
\exp(2)
\approx 7.39
$$

所以小的分数差异会被放大成明显的权重差异。

这使得 attention 可以形成非常尖锐的分布：

$$
[\alpha_{i1},\alpha_{i2},\ldots,\alpha_{in}]
\approx
[0,0,\ldots,1,\ldots,0]
$$

这意味着：

> 当前 token 可以近似只读取某一个历史 token。

这就是标准 attention 具备强检索能力的原因。

---

## 10. Softmax 的本质

QK 内积只是相似度分数：

$$
s_{ij}
=
q_i^\top k_j
$$

softmax 把这一组分数变成读取权重：

$$
\alpha_{i1},\alpha_{i2},\ldots,\alpha_{in}
$$

并满足：

$$
\alpha_{ij} > 0
$$

$$
\sum_{j=1}^{n}\alpha_{ij}=1
$$

所以 softmax 的本质是：

> 对每个 query，在所有 key 上构造一个 query-specific 的、竞争性的、归一化的、可尖锐选择的读取分布。

它不是 attention 的数值修饰，而是 attention 的核心建模机制。

---

## 11. 什么叫“把 Q 和 K 映射到别的空间计算相似度”

linear attention 的关键想法是：

不要直接使用 softmax 形式的相似度，而是换成一种可以分解的相似度函数。

先把 query 和 key 映射到另一个特征空间：

$$
q
\rightarrow
\phi(q)
$$

$$
k
\rightarrow
\phi(k)
$$

然后用：

$$
\phi(q)^\top \phi(k)
$$

作为相似度。

也就是说，相似度函数写成：

$$
\kappa(q,k)
=
\phi(q)^\top \phi(k)
$$

这就是 kernel method 的思想。

它的意思是：

> 不一定在原始空间里直接比较 q 和 k，而是先映射到一个特征空间，再在那个空间里做内积。

例如原始内积是：

$$
q^\top k
$$

而更复杂的相似度可以是：

$$
\kappa(q,k)
=
(q^\top k)^2
$$

它等价于在包含二阶交互特征的新空间里做内积。

所以映射 $\phi$ 的目的，是让相似度函数变成可以分解的内积形式。

---

## 12. Softmax Attention 也可以看成 Kernel Attention

标准 softmax attention 的非归一化相似度是：

$$
\exp(q_i^\top k_j)
$$

这其实也是一种 kernel：

$$
\kappa(q_i,k_j)
=
\exp(q_i^\top k_j)
$$

于是 softmax attention 可以写成：

$$
y_i
=
\frac{
\sum_{j=1}^{n}\kappa(q_i,k_j)v_j
}{
\sum_{j=1}^{n}\kappa(q_i,k_j)
}
$$

其中：

$$
\kappa(q_i,k_j)
=
\exp(q_i^\top k_j)
$$

代入后得到：

$$
y_i
=
\frac{
\sum_{j=1}^{n}\exp(q_i^\top k_j)v_j
}{
\sum_{j=1}^{n}\exp(q_i^\top k_j)
}
$$

这比简单写成：

$$
\operatorname{softmax}(QK^\top)V
$$

更能揭示 softmax attention 的本质：

> 分子是相似度加权的 value 求和，分母是归一化项。

---

## 13. Linear Attention 如何利用 Kernel 形式

如果 kernel 可以分解成：

$$
\kappa(q,k)
=
\phi(q)^\top \phi(k)
$$

那么：

$$
y_i
=
\frac{
\sum_{j=1}^{n}
\phi(q_i)^\top \phi(k_j)v_j
}{
\sum_{j=1}^{n}
\phi(q_i)^\top \phi(k_j)
}
$$

由于 $\phi(q_i)$ 不依赖 $j$，可以把它提出去：

$$
y_i
=
\frac{
\phi(q_i)^\top
\left(
\sum_{j=1}^{n}\phi(k_j)v_j^\top
\right)
}{
\phi(q_i)^\top
\left(
\sum_{j=1}^{n}\phi(k_j)
\right)
}
$$

令：

$$
S
=
\sum_{j=1}^{n}\phi(k_j)v_j^\top
$$

再令：

$$
z
=
\sum_{j=1}^{n}\phi(k_j)
$$

那么：

$$
y_i
=
\frac{
\phi(q_i)^\top S
}{
\phi(q_i)^\top z
}
$$

这就是 normalized linear attention 的核心形式。

这里的关键是：

> $S$ 和 $z$ 都可以提前累积，不需要为每个 query 显式遍历所有 key。

---

## 14. Causal 推理中的递推形式

在 causal language modeling 中，第 $t$ 个 token 只能看前面和当前位置的信息。

因此状态可以逐步更新：

$$
S_t
=
S_{t-1}
+
\phi(k_t)v_t^\top
$$

归一化向量也可以逐步更新：

$$
z_t
=
z_{t-1}
+
\phi(k_t)
$$

当前输出为：

$$
y_t
=
\frac{
\phi(q_t)^\top S_t
}{
\phi(q_t)^\top z_t
}
$$

这说明 linear attention 可以像 RNN 一样工作：

> 每来一个 token，就更新一次固定大小状态，然后从状态中读取输出。

标准 attention 推理时要保存完整 KV cache：

$$
K_{1:t},V_{1:t}
$$

而 linear attention 推理时只需要保存：

$$
S_t
$$

以及：

$$
z_t
$$

所以它的状态大小不随上下文长度线性增长。

---

## 15. 为什么复杂度变成线性

对于每个 token，linear attention 主要做两件事。

更新状态：

$$
S_t
=
S_{t-1}
+
\phi(k_t)v_t^\top
$$

这一步复杂度大约是：

$$
O(d_\phi d_v)
$$

读取状态：

$$
y_t
=
\frac{
\phi(q_t)^\top S_t
}{
\phi(q_t)^\top z_t
}
$$

主要复杂度也是：

$$
O(d_\phi d_v)
$$

所以每个 token 的成本不随历史长度 $t$ 增长。

整个长度为 $n$ 的序列，总复杂度是：

$$
O(nd_\phi d_v)
$$

如果 $d_\phi$ 和 $d_v$ 是固定的，那么复杂度对 $n$ 是线性的：

$$
O(n)
$$

这就是 linear attention 的“linear”来源。

它不是说矩阵乘法神奇变少了，而是说：

> 它不再显式构造 $n \times n$ 的 token-token 关系矩阵。

---

## 16. 为什么不精确分解 Softmax Kernel

softmax 的非归一化 kernel 是：

$$
\exp(q^\top k)
$$

理论上，它确实可以被看作某个高维甚至无限维特征空间中的内积：

$$
\exp(q^\top k)
=
\phi(q)^\top \phi(k)
$$

但问题是：这个精确特征空间通常维度太高，不能直接计算。

所以实际 linear attention 往往使用有限维的、便宜的特征映射：

$$
\exp(q^\top k)
\approx
\phi(q)^\top \phi(k)
$$

或者干脆不用精确近似 softmax，而是选择另一种更容易线性化的 kernel。

因此 linear attention 不是免费等价替代，而是在做 trade-off：

> 用可分解 kernel 换取线性复杂度，但会损失 softmax attention 的部分建模能力。

---

## 17. Linear Attention 损失了什么

标准 attention 保留的是：

$$
P
=
\operatorname{softmax}(QK^\top)
$$

其中：

$$
P \in \mathbb{R}^{n \times n}
$$

这个矩阵表示所有 token 之间的显式关系。

linear attention 保留的是固定大小状态：

$$
S
\in
\mathbb{R}^{d_\phi \times d_v}
$$

也就是说，信息结构从：

$$
\text{n 个独立 token 地址}
$$

变成了：

$$
\text{一个固定大小压缩状态}
$$

这是检索能力下降的根源。

标准 attention 可以近似只读取某个历史 token：

$$
[\alpha_{i1},\alpha_{i2},\ldots,\alpha_{in}]
\approx
[0,0,\ldots,1,\ldots,0]
$$

而 linear attention 是从压缩状态中读取：

$$
y_i
=
\frac{
\phi(q_i)^\top S
}{
\phi(q_i)^\top z
}
$$

它不再显式拥有 $n$ 个可寻址槽位。

---

## 18. 信息混叠：Linear Attention 的核心弱点

linear attention 的状态是：

$$
S
=
\sum_{j=1}^{n}\phi(k_j)v_j^\top
$$

如果两个 key 的特征相似：

$$
\phi(k_1)
\approx
\phi(k_2)
$$

那么它们写入状态的方向也相似：

$$
S
=
\phi(k_1)v_1^\top
+
\phi(k_2)v_2^\top
+
\cdots
$$

之后某个 query 读取状态时：

$$
\phi(q)^\top S
$$

可能会同时读出 $v_1$ 和 $v_2$ 的混合信息。

这就是信息混叠，也就是 memory interference。

标准 attention 中，即使两个 key 很像，它们仍然属于两个不同 token 位置。模型可以通过 attention 分布在位置之间选择。

linear attention 中，这些信息可能已经提前混进同一个状态矩阵里。

---

## 19. 一个直观例子

假设上下文中有：

> Alice's ID is 48291.  
> Bob's ID is 73921.  
> Carol's ID is 15603.

后面问：

> Bob's ID is?

标准 attention 的理想行为是：

> 当前 query 对 “Bob” 和 “73921” 附近的位置产生高权重。

它可以显式选择相关 token。

linear attention 的状态则类似：

$$
S
\approx
\phi(k_{\text{Alice}})v_{\text{48291}}^\top
+
\phi(k_{\text{Bob}})v_{\text{73921}}^\top
+
\phi(k_{\text{Carol}})v_{\text{15603}}^\top
+
\cdots
$$

所有人名和 ID 的绑定关系都被写进同一个状态矩阵。

如果上下文很短，模型也许能读出来。  
但如果上下文里有几百个类似绑定关系，这些信息就容易互相干扰。

所以 linear attention 更像：

> 保存一份不断更新的压缩摘要。

而标准 attention 更像：

> 保存所有原始 token，并在需要时按内容寻址检索。

---

## 20. 位置分辨率也会下降

标准 attention 显式保留每个位置：

$$
k_1,k_2,\ldots,k_n
$$

每个 key 都有自己的 token index。

linear attention 的状态是累加形式：

$$
S_t
=
S_{t-1}
+
\phi(k_t)v_t^\top
$$

如果没有额外的位置编码、衰减或门控机制，它更难区分：

- 第 100 个 token 写入的信息
- 第 10000 个 token 写入的信息
- 最近的信息
- 很久以前的信息
- 哪个 value 对应哪个 key
- 哪个实体对应哪个属性

因此很多后续模型会加入门控、衰减、选择性更新等机制。

例如可以写成：

$$
S_t
=
\gamma_t S_{t-1}
+
\phi(k_t)v_t^\top
$$

其中：

$$
\gamma_t
$$

控制旧状态保留多少。

这类设计已经开始接近 RetNet、Mamba、Gated DeltaNet 等结构的思想：模型不只是简单累加，而是学会保留、遗忘、覆盖和更新。

---

## 21. 标准 Attention 与 Linear Attention 的能力边界

标准 attention 更像内容寻址内存：

$$
\text{query}
\rightarrow
\text{match keys}
\rightarrow
\text{retrieve values}
$$

它适合：

- 精确复制
- 长程检索
- 变量绑定
- 代码依赖追踪
- needle-in-a-haystack
- 多实体对应关系
- 从上下文中寻找具体证据

linear attention 更像压缩状态记忆：

$$
S_t
=
S_{t-1}
+
\text{update}_t
$$

它适合：

- 长序列低成本建模
- 连续状态传播
- 主题、风格、局部语义维持
- 不需要精确逐 token 回看的任务
- 极长上下文中的粗粒度信息流

---

## 22. 为什么现代模型常用 Hybrid

如果全部使用标准 attention，长上下文成本太高：

$$
O(n^2)
$$

如果全部使用 linear attention，又容易损失精确检索能力。

所以更实际的路线是混合：

$$
\text{多数层使用 linear attention 或 state-space 结构}
$$

$$
\text{少数层保留 full attention}
$$

这样模型大部分时候用便宜的状态传播信息，同时仍然保留部分层用于精确 token 检索。

这类 hybrid 结构的核心思想是：

> 用 linear/state-space 层负责长程低成本传播，用 full attention 层负责高分辨率检索。

---

## 23. 最终总结

标准 attention 是：

$$
\operatorname{softmax}(QK^\top)V
$$

它的核心能力来自：

$$
\operatorname{softmax}(QK^\top)
\in
\mathbb{R}^{n \times n}
$$

这个矩阵为每个 query 提供了一行独立的 token-level 检索分布。

linear attention 把相似度改成可分解 kernel：

$$
\kappa(q,k)
=
\phi(q)^\top \phi(k)
$$

于是可以把历史提前压缩成状态：

$$
S_t
=
S_{t-1}
+
\phi(k_t)v_t^\top
$$

归一化项也可以递推维护：

$$
z_t
=
z_{t-1}
+
\phi(k_t)
$$

当前输出变成：

$$
y_t
=
\frac{
\phi(q_t)^\top S_t
}{
\phi(q_t)^\top z_t
}
$$

因此复杂度从：

$$
O(n^2)
$$

变成：

$$
O(n)
$$

但代价是：

$$
\text{逐 token 可寻址记忆}
\rightarrow
\text{固定大小压缩状态}
$$

所以 linear attention 的本质不是“更快的标准 attention”，而是：

> 一种用压缩状态替代显式 token-token 检索图的注意力近似。

它快，是因为不再构造 $n \times n$ 关系矩阵。  
它弱，是因为不再保留每个历史 token 的独立可寻址位置。  
它有价值，是因为在超长上下文场景中，很多信息并不需要逐 token 精确检索，而只需要低成本状态传播。
