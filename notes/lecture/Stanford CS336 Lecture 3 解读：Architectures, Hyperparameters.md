# Stanford CS336 Lecture 3 解读：Architectures, Hyperparameters

视频链接：<https://www.youtube.com/watch?v=ptFiH_bHnJw&list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_&index=3>

这节课对应 **Stanford CS336: Language Modeling from Scratch, Spring 2025, Lecture 3: Architectures, Hyperparameters**。

第 2 讲主要讲的是：

$$
\text{Resource Accounting}
$$

也就是怎么估算显存、FLOPs、训练时间、GPU 利用率。

第 3 讲则转向：

$$
\text{Modern Transformer Architecture}
$$

也就是现代大语言模型为什么大多采用：

- decoder-only Transformer
- Pre-Norm
- RMSNorm
- RoPE
- SwiGLU
- no bias
- MQA / GQA
- QK-Norm
- z-loss
- 特定的超参数比例

这节课的核心不是“Transformer 有哪些模块”，而是解释：

$$
\boxed{
\text{现代 LLM 架构}
=
\text{训练稳定性}
+
\text{参数预算}
+
\text{访存效率}
+
\text{推理 KV cache 约束}
}
$$

---

## 1. 原始 Transformer 和现代 LLM 的差别

原始 Transformer 的结构大致是：

$$
x \rightarrow \text{MHA} \rightarrow \text{Add \& Norm}
\rightarrow \text{FFN} \rightarrow \text{Add \& Norm}
$$

也就是 **Post-Norm**：

$$
x_{l+1}
=
\text{LN}(x_l + F(x_l))
$$

其中 $F$ 可以是 attention 或 FFN。

但是现代 LLM 通常采用 **Pre-Norm**：

$$
x_{l+1}
=
x_l + F(\text{Norm}(x_l))
$$

对于一个完整 Transformer block，通常写成：

$$
x'_l
=
x_l
+
\text{Attention}(\text{Norm}(x_l))
$$

$$
x_{l+1}
=
x'_l
+
\text{MLP}(\text{Norm}(x'_l))
$$

这看起来只是把 LayerNorm 的位置换了一下，但对深层模型训练非常关键。

---

## 2. 为什么现代 LLM 更喜欢 Pre-Norm

Post-Norm 是：

$$
x_{l+1}
=
\text{LN}(x_l + F(x_l))
$$

Pre-Norm 是：

$$
x_{l+1}
=
x_l + F(\text{LN}(x_l))
$$

区别在于 residual path 是否保持干净。

Residual connection 的核心价值是提供一条近似恒等映射路径：

$$
\frac{\partial x_{l+1}}{\partial x_l}
\approx
I
+
\frac{\partial F}{\partial x_l}
$$

这样梯度可以更容易从深层传回浅层。

如果使用 Post-Norm：

$$
x_{l+1}
=
\text{LN}(x_l + F(x_l))
$$

那么 LayerNorm 会直接作用在 residual add 之后，等于把 residual stream 也归一化了。这样会破坏那条干净的 identity path。

Pre-Norm 的好处是：

$$
x_{l+1}
=
x_l + F(\text{LN}(x_l))
$$

其中 $x_l$ 这条主路径没有被 LayerNorm 直接打断。LayerNorm 只作用在非残差分支的输入上。

所以可以理解为：

> Pre-Norm 的本质是保护 residual stream，让深层模型更容易训练。

不过 Pre-Norm 也不是完美的。现代一些模型还会加 extra norm 或 double norm，例如在 block 前后都加 normalization，用来进一步控制激活尺度。

---

## 3. LayerNorm 到 RMSNorm

LayerNorm 的公式是：

$$
\text{LN}(x)
=
\gamma
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
+
\beta
$$

其中：

$$
\mu
=
\frac{1}{d}
\sum_{i=1}^{d}x_i
$$

$$
\sigma^2
=
\frac{1}{d}
\sum_{i=1}^{d}(x_i-\mu)^2
$$

RMSNorm 的公式是：

$$
\text{RMSNorm}(x)
=
\gamma
\frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}}
$$

RMSNorm 和 LayerNorm 的主要区别是：

- LayerNorm 会减均值；
- RMSNorm 不减均值；
- RMSNorm 只根据均方根缩放；
- RMSNorm 通常没有 bias；
- RMSNorm 更简单，访存和计算更少。

LayerNorm：

$$
x
\rightarrow
x-\mu
\rightarrow
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
\rightarrow
\gamma(\cdot)+\beta
$$

RMSNorm：

$$
x
\rightarrow
\frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon}}
\rightarrow
\gamma(\cdot)
$$

在大模型里，Norm 层的 FLOPs 不是主导项，但 runtime 可能并不低，因为它通常是 memory-bound 操作。

也就是说：

$$
\text{runtime}
\not\propto
\text{FLOPs only}
$$

更准确地说，很多操作的耗时接近：

$$
\text{runtime}
\approx
\max
\left(
\frac{\text{FLOPs}}{\text{compute throughput}},
\frac{\text{bytes moved}}{\text{memory bandwidth}}
\right)
$$

所以 RMSNorm 的意义不只是“少算几个数”，而是：

> 减少 reduction、减少访存、减少参数移动，更适合高吞吐实现。

---

## 4. 为什么现代 LLM 经常去掉 bias

传统线性层是：

$$
y = Wx + b
$$

现代 LLM 里很多 linear layer 直接写成：

$$
y = Wx
$$

比如 FFN 可以从：

$$
\text{FFN}(x)
=
W_2 \phi(W_1x+b_1)+b_2
$$

变成：

$$
\text{FFN}(x)
=
W_2 \phi(W_1x)
$$

去掉 bias 的原因主要有：

- 参数更少；
- 读写更少；
- kernel 更容易融合；
- 对大模型效果通常影响不大；
- 某些情况下训练更稳定。

这背后体现了现代 LLM 的一种架构哲学：

> 尽量保留大矩阵乘法，减少小而碎的额外操作。

因为大矩阵乘法可以高效利用 Tensor Core，而 bias、norm、activation、softmax 这类操作往往更容易被访存限制。

---

## 5. FFN：从 ReLU / GeLU 到 SwiGLU

传统 FFN 是：

$$
\text{FFN}(x)
=
W_2 \phi(W_1x)
$$

其中 $\phi$ 可以是 ReLU 或 GeLU。

GeLU 的形式大致是：

$$
\text{GeLU}(x)
=
x \Phi(x)
$$

其中 $\Phi(x)$ 是标准正态分布的 CDF。

GeLU 比 ReLU 更平滑，但现代 LLM 越来越多使用 GLU 变体，尤其是 SwiGLU。

GLU 的基本思想是：

$$
\text{GLU}(x)
=
(W_{\text{up}}x)
\odot
\sigma(W_{\text{gate}}x)
$$

SwiGLU 可以写成：

$$
\text{SwiGLU}(x)
=
(W_{\text{up}}x)
\odot
\text{SiLU}(W_{\text{gate}}x)
$$

其中：

$$
\text{SiLU}(z)
=
z\sigma(z)
$$

完整的 SwiGLU FFN 是：

$$
\text{FFN}(x)
=
W_{\text{down}}
\left[
(W_{\text{up}}x)
\odot
\text{SiLU}(W_{\text{gate}}x)
\right]
$$

这里有三组矩阵：

$$
W_{\text{up}},\quad W_{\text{gate}},\quad W_{\text{down}}
$$

相比普通 FFN，SwiGLU 多了一个 gate branch。

它的直觉是：

$$
\text{output}
=
\text{content branch}
\odot
\text{gate branch}
$$

也就是说，模型不只是做非线性变换，还可以学习哪些通道该通过、哪些该压制。

---

## 6. 为什么 SwiGLU 的 hidden size 通常是 $\frac{8}{3}d_{\text{model}}$

普通 FFN 的参数量大约是：

$$
d_{\text{model}}d_{\text{ff}}
+
d_{\text{ff}}d_{\text{model}}
=
2d_{\text{model}}d_{\text{ff}}
$$

传统配置通常是：

$$
d_{\text{ff}}
=
4d_{\text{model}}
$$

所以普通 FFN 参数量大约是：

$$
2d_{\text{model}}
\cdot
4d_{\text{model}}
=
8d_{\text{model}}^2
$$

SwiGLU 有三组矩阵：

$$
W_{\text{up}},
W_{\text{gate}},
W_{\text{down}}
$$

所以参数量大约是：

$$
3d_{\text{model}}d_{\text{ff}}
$$

如果希望 SwiGLU 的参数量和普通 FFN 大致相同，需要满足：

$$
3d_{\text{model}}d_{\text{ff}}
=
8d_{\text{model}}^2
$$

两边除以 $3d_{\text{model}}$：

$$
d_{\text{ff}}
=
\frac{8}{3}d_{\text{model}}
$$

所以现代 LLM 中常见：

$$
\boxed{
d_{\text{ff}}
=
\frac{8}{3}d_{\text{model}}
}
$$

这个比例不是玄学，而是为了让 SwiGLU 和普通 $4d$ FFN 的参数量大致匹配。

---

## 7. Serial Block 和 Parallel Block

标准 Transformer block 是串行结构：

$$
x'
=
x
+
\text{Attn}(\text{Norm}(x))
$$

$$
x''
=
x'
+
\text{MLP}(\text{Norm}(x'))
$$

也就是 attention 后面接 MLP。

也有一些模型采用 parallel block：

$$
x'
=
x
+
\text{Attn}(\text{Norm}(x))
+
\text{MLP}(\text{Norm}(x))
$$

二者区别：

| 结构           | 特点                                        |
| -------------- | ------------------------------------------- |
| Serial Block   | attention 和 MLP 串行组合，表达力可能更强   |
| Parallel Block | attention 和 MLP 并行计算，系统效率可能更好 |

Serial 更像增加深度：

$$
x \rightarrow \text{Attn} \rightarrow \text{MLP}
$$

Parallel 更像增加宽度：

$$
x \rightarrow
\begin{cases}
\text{Attn}\\
\text{MLP}
\end{cases}
\rightarrow \text{Add}
$$

这说明架构设计不是纯数学问题，而是效果、稳定性和系统效率之间的 trade-off。

---

## 8. 位置编码：为什么 RoPE 成为主流

语言模型必须知道 token 的位置。

常见位置编码有：

- sinusoidal absolute position embedding
- learned absolute position embedding
- relative position embedding
- RoPE

原始 Transformer 用 sinusoidal embedding。GPT 早期使用 learned absolute position embedding。现代 LLM 则大量使用 RoPE。

RoPE 的核心目标是：

> 让 attention score 自然依赖相对位置，而不是绝对位置。

我们希望：

$$
\langle f(x_i,i), f(y_j,j) \rangle
=
g(x_i,y_j,i-j)
$$

也就是说，位置影响应该主要体现为 $i-j$。

RoPE 使用旋转矩阵。对于二维子空间：

$$
R_{\theta i}
=
\begin{bmatrix}
\cos(\theta i) & -\sin(\theta i) \\
\sin(\theta i) & \cos(\theta i)
\end{bmatrix}
$$

对 query 和 key 做旋转：

$$
q_i'
=
R_{\theta i}q_i
$$

$$
k_j'
=
R_{\theta j}k_j
$$

attention score 是：

$$
(q_i')^\top k_j'
=
(R_{\theta i}q_i)^\top(R_{\theta j}k_j)
$$

利用旋转矩阵性质：

$$
R_{\theta i}^{\top}R_{\theta j}
=
R_{\theta(j-i)}
$$

所以：

$$
(q_i')^\top k_j'
=
q_i^\top R_{\theta(j-i)}k_j
$$

最终 attention score 只通过 $j-i$ 引入相对位置。

这就是 RoPE 的优雅之处：

$$
\boxed{
\text{RoPE 把相对位置编码进了 } QK^\top \text{ 的几何结构中}
}
$$

它不是简单把 position embedding 加到 token embedding 上，而是在 Q/K 空间里改变 attention score。

---

## 9. 超参数：$d_{\text{ff}}$ 怎么选

普通 FFN 是：

$$
x \in \mathbb{R}^{d_{\text{model}}}
$$

$$
W_1 \in \mathbb{R}^{d_{\text{model}}\times d_{\text{ff}}}
$$

$$
W_2 \in \mathbb{R}^{d_{\text{ff}}\times d_{\text{model}}}
$$

经典配置是：

$$
d_{\text{ff}}
=
4d_{\text{model}}
$$

这个比例来自长期经验，不是严格理论。

对于 GLU / SwiGLU，因为有 gate branch，通常使用：

$$
d_{\text{ff}}
=
\frac{8}{3}d_{\text{model}}
$$

本质是为了控制参数预算。

所以两类 FFN 可以这样对比：

| FFN 类型   |                      中间维度 |              参数量级 |
| ---------- | ----------------------------: | --------------------: |
| 普通 FFN   |           $4d_{\text{model}}$ | $8d_{\text{model}}^2$ |
| SwiGLU FFN | $\frac{8}{3}d_{\text{model}}$ | $8d_{\text{model}}^2$ |

---

## 10. 超参数：vocab size 不是越小越好

vocab size 记为 $V$，hidden size 记为 $d$。

Embedding 参数量大约是：

$$
Vd
$$

如果 output LM head 不和 embedding 共享权重，还会有：

$$
Vd
$$

所以 vocab size 越大，embedding 和输出层参数越多。

但是 vocab size 太小，会导致文本被切成更多 token。

例如中文、多语言、代码、符号文本，如果 tokenizer 不友好，序列长度会明显膨胀。

这会影响：

- 上下文窗口利用率；
- 推理成本；
- KV cache 大小；
- 长文本处理效率。

因为 KV cache 大小近似和 sequence length 成正比：

$$
M_{\text{KV}}
\propto
L
$$

其中 $L$ 是 token 序列长度。

所以 vocab size 的 trade-off 是：

$$
\text{larger vocab}
\Rightarrow
\text{larger embedding/head}
$$

但：

$$
\text{larger vocab}
\Rightarrow
\text{fewer tokens for multilingual/code data}
$$

对于中文和多语言模型，这一点尤其重要。

---

## 11. Dropout 为什么在大规模预训练里不流行

传统机器学习里，dropout 用于防止过拟合：

$$
h' = m \odot h
$$

其中 $m$ 是随机 mask。

但 LLM pretraining 里通常：

- 数据极大；
- 训练 token 数极多；
- 很多模型甚至没有完整遍历所有数据很多轮；
- train/val gap 不是最主要矛盾。

所以 dropout 在现代大规模预训练里不太常用。

可以粗略理解为：

$$
\text{small data regime}
\Rightarrow
\text{dropout useful}
$$

$$
\text{large data pretraining}
\Rightarrow
\text{dropout less necessary}
$$

不过这不代表所有场景都不用 dropout。小模型、小数据、finetuning、特定任务中 dropout 仍可能有用。

---

## 12. Weight Decay 为什么还在用

AdamW 的更新可以理解为：

$$
\theta_{t+1}
=
\theta_t
-
\eta
\left(
\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
+
\lambda \theta_t
\right)
$$

其中：

- $\eta$ 是学习率；
- $\lambda$ 是 weight decay 系数；
- $\hat{m}_t$ 是一阶动量；
- $\hat{v}_t$ 是二阶动量。

传统理解中，weight decay 是 regularization，用来防止过拟合。

但在 LLM pretraining 里，weight decay 不只是防过拟合。它还会影响 optimization dynamics，尤其和 learning rate schedule 共同作用。

例如 cosine learning rate decay：

$$
\eta_t
=
\eta_{\min}
+
\frac{1}{2}
(\eta_{\max}-\eta_{\min})
\left(
1+\cos\frac{\pi t}{T}
\right)
$$

训练后期学习率变小，weight decay 对参数范数和 loss 下降会产生复杂影响。

所以更准确地说：

$$
\boxed{
\text{LLM 中的 weight decay 不只是正则项，也是一种优化动态控制}
}
$$

---

## 13. Softmax 是训练稳定性的危险区

Transformer 中有两个重要 softmax：

1. output softmax；
2. attention softmax。

Output softmax：

$$
p_i
=
\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Attention softmax：

$$
\text{Attention}(Q,K,V)
=
\text{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

softmax 的问题在于 exponential：

$$
e^{z_i}
$$

如果 logits 太大，会 overflow；如果某些 logits 过大，softmax 会变得极端尖锐，导致训练不稳定。

所以现代 LLM 会使用一些技巧控制 softmax 的数值尺度。

---

## 14. z-loss：稳定 output softmax

Output softmax 的归一化项是：

$$
Z(x)
=
\sum_j e^{z_j}
$$

z-loss 的思想是鼓励：

$$
\log Z(x)
\approx 0
$$

也就是：

$$
Z(x)
\approx 1
$$

辅助损失可以写成：

$$
\mathcal{L}_z
=
\alpha
\left(
\log Z(x)
\right)^2
$$

其中 $\alpha$ 是一个很小的系数。

它的作用不是提高模型表达能力，而是避免 output softmax 的 normalizer 变得过大或过小。

可以理解为：

$$
\boxed{
\text{z-loss 用来稳定 output softmax 的尺度}
}
$$

---

## 15. QK-Norm：稳定 attention softmax

Attention logits 是：

$$
S
=
\frac{QK^\top}{\sqrt{d_k}}
$$

如果 $Q$ 或 $K$ 的范数很大，那么 $QK^\top$ 会变大，softmax 会变尖：

$$
\text{softmax}(S)
$$

可能变成接近 one-hot 的分布。

QK-Norm 的想法是在计算 attention score 前先归一化 query 和 key：

$$
\tilde{Q}
=
\text{Norm}(Q)
$$

$$
\tilde{K}
=
\text{Norm}(K)
$$

然后：

$$
\text{Attention}
=
\text{softmax}
\left(
\frac{\tilde{Q}\tilde{K}^{\top}}{\sqrt{d_k}}
\right)V
$$

它的目标是控制进入 softmax 的 logits 尺度。

可以理解为：

$$
\boxed{
\text{QK-Norm 用来稳定 attention softmax}
}
$$

这对多模态模型尤其重要。因为图像 token、文本 token、latent token、anchor token 混在一起时，attention logits 的分布可能更容易失控。

如果你做 visual anchor injection 或 latent reasoning，应该特别关注：

$$
QK^\top
$$

的尺度变化，而不只是看 attention mass 是否变大。

---

## 16. Arithmetic Intensity：为什么 FLOPs 不是全部

Arithmetic intensity 定义为：

$$
\text{Arithmetic Intensity}
=
\frac{\text{arithmetic operations}}{\text{memory accesses}}
$$

或者更常见地写成：

$$
\text{AI}
=
\frac{\text{FLOPs}}{\text{Bytes moved}}
$$

它衡量的是：

> 每搬运一个 byte 的数据，能做多少浮点计算。

GPU 上 compute 很强，但显存带宽有限。因此很多操作不是 compute-bound，而是 memory-bound。

如果：

$$
\frac{\text{FLOPs}}{\text{compute throughput}}
>
\frac{\text{Bytes}}{\text{memory bandwidth}}
$$

则操作偏 compute-bound。

如果：

$$
\frac{\text{Bytes}}{\text{memory bandwidth}}
>
\frac{\text{FLOPs}}{\text{compute throughput}}
$$

则操作偏 memory-bound。

大矩阵乘法通常 arithmetic intensity 高，更容易吃满 GPU。

Norm、softmax、KV cache 读写通常 arithmetic intensity 低，更容易 memory-bound。

这就是为什么：

$$
\text{FLOPs 少}
\not\Rightarrow
\text{一定快}
$$

真正要看：

$$
\boxed{
\text{wall-clock time}
=
\text{compute cost}
+
\text{memory movement cost}
+
\text{kernel launch / sync overhead}
}
$$

---

## 17. 推理时为什么 attention 变成 memory-bound

训练时，整个 sequence 可以并行计算。

假设：

$$
Q,K,V \in \mathbb{R}^{B \times H \times L \times d}
$$

attention score 是：

$$
QK^\top
\in
\mathbb{R}^{B \times H \times L \times L}
$$

这时可以使用大矩阵乘法，GPU 利用率比较高。

但是推理时是 autoregressive decoding：

$$
x_1 \rightarrow x_2 \rightarrow x_3 \rightarrow \cdots
$$

每一步只能生成一个新 token。

对于第 $t$ 个 token，只需要当前 query：

$$
q_t
$$

但要 attend 到过去所有 key/value：

$$
K_{\leq t}, V_{\leq t}
$$

所以：

$$
o_t
=
\text{softmax}
\left(
\frac{q_t K_{\leq t}^{\top}}{\sqrt{d_k}}
\right)
V_{\leq t}
$$

这就需要 KV cache。

KV cache 保存过去 token 的 key 和 value：

$$
\text{KV cache}
=
\{K_1,V_1,K_2,V_2,\ldots,K_t,V_t\}
$$

每生成一个新 token：

1. 计算当前 token 的 $q_t,k_t,v_t$；
2. 读取历史 $K_{\leq t},V_{\leq t}$；
3. 计算 attention；
4. 把 $k_t,v_t$ 追加进 cache。

问题在于：

> 长上下文推理时，每一步都要从显存读取大量 KV cache。

所以 decoding 阶段很容易 memory-bound。

---

## 18. KV cache 显存公式

设：

- batch size 为 $B$；
- sequence length 为 $L$；
- layer 数为 $N_{\text{layers}}$；
- KV head 数为 $H_{\text{kv}}$；
- head dimension 为 $d_{\text{head}}$；
- 每个元素 bytes 为 $s$；
- K 和 V 两份，所以乘以 $2$。

那么 KV cache 大小近似是：

$$
M_{\text{KV}}
=
2
\cdot
B
\cdot
L
\cdot
N_{\text{layers}}
\cdot
H_{\text{kv}}
\cdot
d_{\text{head}}
\cdot
s
$$

可以看到 KV cache 与 $L$ 线性相关：

$$
M_{\text{KV}}
\propto L
$$

也与 KV heads 数量线性相关：

$$
M_{\text{KV}}
\propto H_{\text{kv}}
$$

这就是为什么 MQA 和 GQA 对推理很重要。

---

## 19. MHA、MQA、GQA

标准 Multi-Head Attention 中，每个 query head 都有自己的 key/value head：

$$
H_q = H_{kv}
$$

这叫 MHA。

MQA，Multi-Query Attention，是：

$$
H_q > 1,\quad H_{kv}=1
$$

也就是多个 query heads 共享一组 key/value。

GQA，Grouped-Query Attention，是折中方案：

$$
1 < H_{kv} < H_q
$$

也就是一组 query heads 共享一组 key/value。

三者对比：

| 类型 | KV heads | KV cache |   表达力 | 推理效率 |
| ---- | -------: | -------: | -------: | -------: |
| MHA  |    $H_q$ |     最大 |       强 |       慢 |
| MQA  |      $1$ |     最小 | 可能下降 |       快 |
| GQA  |   中间值 |     中等 |     较强 |     较快 |

KV cache 大小与 $H_{\text{kv}}$ 成正比：

$$
M_{\text{KV}}
\propto
H_{\text{kv}}
$$

所以从 MHA 到 MQA，KV cache 可以明显减少。

这对推理加速非常关键，因为 decoding 常常不是算不动，而是 KV cache 搬不动。

---

## 20. 这节课和 LLM 系统优化的关系

这节课最重要的不是背架构名词，而是形成判断框架。

现代 LLM 的很多设计都可以归入以下几类账本。

| 设计                          | 主要目的                              |
| ----------------------------- | ------------------------------------- |
| Pre-Norm                      | 稳定深层梯度传播                      |
| RMSNorm                       | 降低访存和计算，效果接近 LayerNorm    |
| 去 bias                       | 简化结构，减少额外参数和潜在不稳定    |
| SwiGLU                        | 提高 MLP 表达能力                     |
| $\frac{8}{3}d_{\text{model}}$ | 让 SwiGLU 和普通 FFN 参数量匹配       |
| RoPE                          | 在 attention score 中自然编码相对位置 |
| Weight decay                  | 不只是正则，也影响优化动态            |
| z-loss                        | 稳定 output softmax                   |
| QK-Norm                       | 稳定 attention softmax                |
| MQA / GQA                     | 减少 KV cache 读写压力，提高推理效率  |

可以总结成：

$$
\boxed{
\text{架构细节}
=
\text{稳定性}
+
\text{表达力}
+
\text{参数量}
+
\text{访存效率}
+
\text{推理成本}
}
$$

---

## 21. 对 inference acceleration 的启发

如果目标是做 LLM 推理加速，不能只问：

> 这个方法减少了多少 FLOPs？

还要问：

$$
\text{是否减少 KV cache 读写？}
$$

$$
\text{是否提高 arithmetic intensity？}
$$

$$
\text{是否减少 kernel launch 和调度开销？}
$$

$$
\text{是否破坏 batch 形状？}
$$

$$
\text{是否引入额外 CPU-GPU 同步？}
$$

很多方法理论上减少 token 或减少 FLOPs，但实际不一定加速。

因为推理阶段的瓶颈可能是：

- KV cache memory bandwidth；
- 小 batch 下 GPU 利用率不足；
- dynamic shape 导致 kernel 不稳定；
- 控制逻辑太复杂；
- attention hook 或监控引入同步；
- prefix cache 命中率不足；
- cache locality 差。

所以更合理的判断是：

$$
\boxed{
\text{真实加速}
\neq
\text{理论 FLOPs 减少}
}
$$

而是：

$$
\boxed{
\text{真实加速}
=
\text{更少访存}
+
\text{更高并行度}
+
\text{更稳定 batch}
+
\text{更低调度开销}
}
$$

---

## 22. 对 visual anchor / latent reasoning 的启发

如果你做的是多模态推理、latent reasoning、visual anchor injection，那么这节课里最重要的是 QK-Norm 和 attention stability。

因为这类方法往往会改变 attention 分布。

例如你注入 visual anchor hidden state，本质上可能改变：

$$
QK^\top
$$

的分布。

这会影响：

- attention entropy；
- visual attention mass；
- attention logits scale；
- softmax sharpness；
- decoding stability；
- 是否出现过度依赖某些视觉 token 的问题。

所以不能只观察：

$$
\text{visual attention mass}
$$

还应该观察：

$$
\|Q\|,\quad \|K\|,\quad QK^\top,\quad \text{softmax entropy}
$$

尤其是：

$$
\max(QK^\top),\quad
\operatorname{std}(QK^\top),\quad
H(\text{softmax}(QK^\top))
$$

否则可能出现一种情况：

> attention mass 变大了，但模型不是更会看图，而是 attention logits 尺度失控了。

QK-Norm 的思想可以作为一种稳定性参考。

---

## 23. 总结

第 3 讲的核心不是介绍 Transformer 模块，而是解释现代 LLM 架构为什么逐渐收敛到当前形态。

可以用一句话概括：

$$
\boxed{
\text{现代 LLM 架构不是原始 Transformer 的简单放大，}
\text{而是在训练稳定性、参数预算、访存效率和推理约束下演化出来的系统工程结果。}
}
$$

第 2 讲教的是：

$$
\text{怎么算账}
$$

第 3 讲讲的是：

$$
\text{为什么现代架构要这样设计}
$$

对应关系是：

| 架构选择     | 背后的账                   |
| ------------ | -------------------------- |
| Pre-Norm     | 梯度传播账                 |
| RMSNorm      | 访存账                     |
| SwiGLU       | 表达力与参数账             |
| RoPE         | 位置建模账                 |
| Weight Decay | 优化动态账                 |
| z-loss       | output softmax 稳定性账    |
| QK-Norm      | attention softmax 稳定性账 |
| MQA / GQA    | KV cache 推理账            |

最终应该形成的判断是：

$$
\boxed{
\text{不要把 Transformer 当成固定模板，}
\text{要把它看成一组围绕稳定性和硬件效率的 trade-off。}
}
$$