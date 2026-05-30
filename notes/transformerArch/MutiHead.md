# 注意力头的本质：为什么 Multi-Head Attention 不只是把大矩阵切成小矩阵

*By Chaa*

## 0.写这篇的灵感

在我刚开始学习注意力机制的时候，就了解到了多头注意力。在那时就有一个疑问：明明从代数角度上来看（甚至权重数量也是一样的），好像多头注意力只是把同一件事情拆开来做罢了。为什么会有这么好的效果和区别呢？其实一个重要的答案就藏在**softmax**内

## 1. 核心结论

多头注意力的关键贡献，不只是“有多个头”，也不只是“每个头有自己的 Q/K/V 投影”。

更准确地说：

> 在 Q/K/V 可以由同一个大矩阵分块得到的前提下，Multi-Head Attention 相比 Single-Head Attention 的关键非等价性，主要来自每个 head 独立执行 softmax 归一化。

也就是说：

$$
\boxed{
\text{Multi-Head 的关键不是简单切块，而是切块后每个 head 各自 softmax}
}
$$

如果没有 softmax，多头结构很大程度上可以看成大矩阵运算的分块实现；但一旦每个 head 各自做 softmax，结果就不再等价于一个大单头 attention。

---

## 2. 单头注意力的形式

设输入为：

$$
X \in \mathbb{R}^{L \times d}
$$

其中：

- $L$ 是序列长度；
- $d$ 是 hidden size，也就是 $d_{\text{model}}$。

单头 attention 中：

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

其中：

$$
Q,K,V \in \mathbb{R}^{L \times d}
$$

标准 scaled dot-product attention 是：

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right)V
$$

其中 attention logits 为：

$$
S =
\frac{QK^\top}{\sqrt{d}}
\in \mathbb{R}^{L \times L}
$$

softmax 后得到 attention matrix：

$$
A =
\operatorname{softmax}(S)
$$

输出为：

$$
O = AV
$$

对于第 $i$ 个 token：

$$
O_i
=
\sum_{j=1}^{L} A_{ij}V_j
$$

所以单头 attention 的含义是：

> 每个 query token 只有一套 attention 分布，用这一套分布从所有 value token 中加权取信息。

---

## 3. 多头注意力的形式

Multi-Head Attention 把 hidden dimension 切成多个 head。

设 head 数为 $h$，每个 head 的维度为：

$$
d_h = \frac{d}{h}
$$

把 Q/K/V 按 hidden dimension 拆开：

$$
Q = [Q_1,Q_2,\ldots,Q_h]
$$

$$
K = [K_1,K_2,\ldots,K_h]
$$

$$
V = [V_1,V_2,\ldots,V_h]
$$

其中：

$$
Q_r,K_r,V_r \in \mathbb{R}^{L \times d_h}
$$

第 $r$ 个 head 的输出是：

$$
O_r
=
\operatorname{softmax}
\left(
\frac{Q_rK_r^\top}{\sqrt{d_h}}
\right)V_r
$$

最后拼接所有 head：

$$
O_{\text{cat}}
=
[O_1,O_2,\ldots,O_h]
$$

再经过输出投影：

$$
Y = O_{\text{cat}}W_O^\top
$$

所以 multi-head attention 是：

$$
\operatorname{MHA}(X)
=
W_O
[
\operatorname{head}_1;
\operatorname{head}_2;
\ldots;
\operatorname{head}_h
]
$$

其中：

$$
\operatorname{head}_r
=
\operatorname{softmax}
\left(
\frac{Q_rK_r^\top}{\sqrt{d_h}}
\right)V_r
$$

---

## 4. 关键问题：如果没有 softmax，多头是不是就接近大单头？

这是讨论的核心。

假设我们暂时去掉 softmax，看一个简化版 attention：

$$
O = QK^\top V
$$

把大 Q/K/V 拆成多个 head：

$$
Q=[Q_1,\ldots,Q_h]
$$

$$
K=[K_1,\ldots,K_h]
$$

$$
V=[V_1,\ldots,V_h]
$$

那么：

$$
QK^\top
=
[Q_1,\ldots,Q_h]
\begin{bmatrix}
K_1^\top \\
\vdots \\
K_h^\top
\end{bmatrix}
$$

所以：

$$
QK^\top
=
\sum_{r=1}^{h}Q_rK_r^\top
$$

如果没有 softmax，那么 attention 剩下的主要就是矩阵乘法、分块、拼接。这时多头结构的特殊性会明显下降，因为它更像是在做分块双线性运算。

严格地说，标准单头的：

$$
QK^\top V
$$

和多头分块的：

$$
[
Q_1K_1^\top V_1,
Q_2K_2^\top V_2,
\ldots,
Q_hK_h^\top V_h
]
$$

不一定完全相等。

因为单头中：

$$
QK^\top
=
\sum_{r=1}^{h}Q_rK_r^\top
$$

这个总 attention logits 会作用到整个 $V$ 上；而多头分块中，第 $r$ 个 value 只被第 $r$ 个 head 的 logits 作用。

但是这两者都还处在“可以通过线性代数分块理解”的范围内，真正使 multi-head 和 single-head 难以互相吸收的，是 softmax 的非线性归一化位置。

---

## 5. Softmax 让两者不再等价

单头大 attention 是：

$$
A_{\text{single}}
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right)
$$

由于：

$$
QK^\top
=
\sum_{r=1}^{h}Q_rK_r^\top
$$

所以单头可以写成：

$$
A_{\text{single}}
=
\operatorname{softmax}
\left(
\frac{\sum_{r=1}^{h}Q_rK_r^\top}{\sqrt{d}}
\right)
$$

也就是说，单头是：

> 先把所有子空间的 logits 加起来，再做一次 softmax。

多头则是：

$$
A_r
=
\operatorname{softmax}
\left(
\frac{Q_rK_r^\top}{\sqrt{d_h}}
\right)
$$

然后：

$$
O_{\text{multi}}
=
[
A_1V_1,
A_2V_2,
\ldots,
A_hV_h
]
$$

也就是说，多头是：

> 每个子空间各自做 softmax，再各自聚合 value。

两者的顺序完全不同。

关键数学事实是：

$$
\operatorname{softmax}(A+B)
\neq
\operatorname{softmax}(A)+\operatorname{softmax}(B)
$$

也不等价于：

$$
[
\operatorname{softmax}(A_1),
\operatorname{softmax}(A_2),
\ldots,
\operatorname{softmax}(A_h)
]
$$

所以：

$$
\boxed{
\text{单头：先合并 logits，再归一化}
}
$$

$$
\boxed{
\text{多头：先分别归一化，再拼接输出}
}
$$

这就是 multi-head attention 和 single-head attention 的核心非等价性。

---

## 6. 一个极简例子

假设有两个 head，两个 key。

第一个 head 的 logits 是：

$$
S_1 = [10,0]
$$

第二个 head 的 logits 是：

$$
S_2 = [0,10]
$$

多头分别 softmax：

$$
\operatorname{softmax}(S_1)
\approx
[1,0]
$$

$$
\operatorname{softmax}(S_2)
\approx
[0,1]
$$

这表示：

- head 1 强烈关注第一个 token；
- head 2 强烈关注第二个 token。

如果合成一个单头 logits：

$$
S = S_1 + S_2 = [10,10]
$$

那么：

$$
\operatorname{softmax}(S)
=
[0.5,0.5]
$$

这时单头变成同时平均关注两个 token。

这个例子说明：

> 多头可以保留“不同子空间分别强烈支持不同 token”的结构；单头在 logits 合并后，这种结构会被抹平。

所以 multi-head attention 的关键价值之一是：

$$
\boxed{
\text{每个 head 有自己独立的 attention 分布}
}
$$

---

## 7. Softmax 的本质作用：独立注意力预算

softmax 会把每一行 logits 变成概率分布。

对于单头：

$$
\sum_{j=1}^{L} A_{ij}=1
$$

也就是说，对于第 $i$ 个 query token，单头只有一份注意力预算。

如果这个 token 同时需要关注：

- 主语；
- 谓语；
- 指代对象；
- 位置 token；
- 分隔符；
- 数字条件；
- 视觉证据 token；

这些关系都必须挤在同一个 attention 分布里竞争概率质量。

多头则不同。

对于 multi-head：

$$
\sum_{j=1}^{L} A_{ij}^{(r)}=1,
\quad
r=1,\ldots,h
$$

也就是说，第 $i$ 个 query token 拥有 $h$ 份独立的注意力预算。

每个 head 都可以形成一套自己的路由图：

$$
A^{(1)},A^{(2)},\ldots,A^{(h)}
$$

因此：

$$
\boxed{
\text{Multi-Head Attention 给同一个 token 提供了多套独立的信息检索分布}
}
$$

这比“把 hidden dimension 切成几块”更重要。

---

## 8. 多头不是简单地“看不同 token”，而是“用不同标准看 token”

一个 head 的 attention logits 是：

$$
S_{ij}^{(r)}
=
\frac{
(x_iW_Q^{(r)})(x_jW_K^{(r)})^\top
}{\sqrt{d_h}}
$$

这说明第 $r$ 个 head 有自己的 query/key 投影矩阵：

$$
W_Q^{(r)}, W_K^{(r)}
$$

不同 head 对应不同的相似度函数。

所以多头不只是“不同 head 看不同位置”，更准确地说是：

> 不同 head 用不同的匹配标准，构造不同的 token-to-token 信息路由图。

每个 head 的 attention matrix：

$$
A^{(r)} \in \mathbb{R}^{L \times L}
$$

可以看成一张有向加权图：

- 节点是 token；
- 边表示 token 之间的信息流动；
- 边权是 attention weight。

因此：

$$
\boxed{
\text{一个 head 是一张 learned token interaction graph}
}
$$

多头就是多张 token interaction graph 并行。

---

## 9. 为什么单个更大的 head 不能完全替代多个小 head

假设 hidden size 固定：

$$
d_{\text{model}}=768
$$

如果使用一个大 head：

$$
d_h=768
$$

如果使用 12 个 heads：

$$
d_h=64
$$

一个大 head 的相似度空间更大，但它仍然只有一套 softmax 分布：

$$
A \in \mathbb{R}^{L \times L}
$$

而 12 个 heads 有 12 套分布：

$$
A^{(1)},A^{(2)},\ldots,A^{(12)}
$$

所以两者的差别不是单纯的维度大小，而是：

$$
\boxed{
\text{单大头增强的是单个相似度空间}
}
$$

$$
\boxed{
\text{多头增加的是多个独立路由分布}
}
$$

单头只能回答一次：

> 我这个 token 应该看谁？

多头可以并行回答多次：

> 从语法角度我应该看谁？
>
> 从指代角度我应该看谁？
>
> 从位置角度我应该看谁？
>
> 从格式角度我应该看谁？
>
> 从语义角度我应该看谁？

这就是 multi-head 的结构优势。

---

## 10. 从梯度角度看：多头隔离了 softmax 竞争

softmax 的梯度有竞争性。

对于一行 softmax：

$$
A_i=\operatorname{softmax}(S_i)
$$

其雅可比为：

$$
\frac{\partial A_{ij}}{\partial S_{ik}}
=
A_{ij}
(
\mathbf{1}_{j=k}
-
A_{ik}
)
$$

这意味着，在同一个 softmax 分布内部，提高某个 key 的概率，会相对压低其他 key 的概率。

单头中，所有关系都在同一个 softmax 分布里竞争。

多头中，不同关系可以分散到不同 head：

- 语法关系在一个 head 内竞争；
- 指代关系在另一个 head 内竞争；
- 局部关系在另一个 head 内竞争；
- 特殊符号关系在另一个 head 内竞争。

所以：

$$
\boxed{
\text{多头把不同类型关系的 softmax 竞争隔离开了}
}
$$

这也是 head-wise softmax 的深层意义。

---

## 11. 从信息混叠角度看：多头减少加权平均的信息损失

attention 输出本质上是加权平均：

$$
O_i = \sum_j A_{ij}V_j
$$

单头只有一次加权平均。如果一个 token 同时需要从多个位置取不同类型的信息，这些信息会被混合进同一个向量中。

多头则是分别聚合：

$$
O_i^{(1)}
=
\sum_j A_{ij}^{(1)}V_j^{(1)}
$$

$$
O_i^{(2)}
=
\sum_j A_{ij}^{(2)}V_j^{(2)}
$$

$$
\cdots
$$

然后拼接：

$$
O_i
=
[
O_i^{(1)},O_i^{(2)},\ldots,O_i^{(h)}
]
$$

这使得不同信息源可以先保持分离，再由输出矩阵 $W_O$ 统一混合。

因此：

$$
\boxed{
\text{多头减少了单一加权平均带来的信息混叠}
}
$$

---

## 12. $W_O$ 的作用

多头输出拼接后：

$$
O_{\text{cat}}
=
[
O_1,O_2,\ldots,O_h
]
$$

再经过输出投影：

$$
Y=O_{\text{cat}}W_O^\top
$$

$W_O$ 的作用不是装饰性的。

它负责把多个 head 取回来的信息重新混合回 residual stream。

如果没有 $W_O$，多个 head 只是机械拼接；有了 $W_O$，模型可以学习：

- 哪些 head 的信息更重要；
- 不同 head 的信息如何组合；
- 如何把多个路由分布取回来的信息变成下一层可用的 hidden state。

所以一个完整 attention head 的贡献链条是：

$$
\boxed{
\text{独立 Q/K/V 投影}
\rightarrow
\text{独立 softmax 路由}
\rightarrow
\text{独立 value 聚合}
\rightarrow
\text{输出投影混合}
}
$$

---

## 13. 和 MQA / GQA 的关系

这个理解也可以解释 MQA 和 GQA。

标准 Multi-Head Attention 中：

$$
H_q = H_{kv}
$$

每个 query head 有自己的 key/value head。

Multi-Query Attention 是：

$$
H_{kv}=1
$$

也就是多个 query heads 共享同一组 key/value。

Grouped-Query Attention 是：

$$
1 < H_{kv} < H_q
$$

也就是一组 query heads 共享一组 key/value。

从“head-wise softmax”的角度看：

- query heads 仍然可以有不同 softmax 分布；
- 但 K/V 子空间的多样性减少；
- KV cache 显存和访存下降；
- 表达能力可能略受影响。

所以多头的贡献不只在 softmax，也依赖 Q/K/V 子空间。

但在“为什么拆头后和单头不等价”这个问题上，最关键的非线性差异仍然是：

$$
\boxed{
\text{每个 head 独立 softmax}
}
$$

---

## 14. 最终总结

单头 attention 可以概括为：

$$
\boxed{
\text{一个相似度标准}
+
\text{一套 softmax 路由}
+
\text{一次 value 聚合}
}
$$

多头 attention 可以概括为：

$$
\boxed{
\text{多套相似度标准}
+
\text{多套 softmax 路由}
+
\text{多路 value 聚合}
+
\text{输出混合}
}
$$

如果只看 Q/K/V 矩阵的分块，多头似乎只是把一个大矩阵切成多个小矩阵。

但由于每个 head 都在自己的 logits 上独立执行 softmax，所以：

$$
\operatorname{softmax}
\left(
\sum_r S_r
\right)
\neq
[
\operatorname{softmax}(S_1),
\ldots,
\operatorname{softmax}(S_h)
]
$$

因此 multi-head attention 不是 single-head attention 的简单分块实现。

最准确的结论是：

$$
\boxed{
\text{Multi-Head Attention 的关键非等价性，来自 head-wise softmax。}
}
$$

进一步说：



$$
\text{独立 softmax 是 multi-head 的路由核心；独立 Q/K/V 子空间是它能路由出不同信息的前提。}
$$

