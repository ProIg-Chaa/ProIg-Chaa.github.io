# Stanford CS336 Lecture 4 解读：Attention Alternatives and Mixtures of Experts

*PS:这一讲可以看成第 3 讲架构取舍的进一步延伸。第 3 讲主要讲现代 dense Transformer 为什么采用 Pre-Norm、RMSNorm、RoPE、SwiGLU、GQA 等设计；第 4 讲则继续追问：如果标准 Transformer 还是太贵怎么办？尤其当上下文越来越长、参数规模越来越大时，仅仅优化标准 attention 和 dense MLP 可能已经不够了。所以这一讲的主线就是：如何让模型的计算从“全量使用资源”变成“按需使用资源”。*

视频链接：<https://www.youtube.com/watch?v=LPv1KfUXLCo&list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_&index=4>

这节课对应 **Stanford CS336: Language Modeling from Scratch, Spring 2025, Lecture 4: Attention Alternatives and Mixtures of Experts**。

第 3 讲主要讲的是：

$$
\text{Modern Transformer Architecture}
$$

也就是现代 LLM 为什么大多采用：

- Pre-Norm
- RMSNorm
- RoPE
- SwiGLU
- no bias
- MQA / GQA
- QK-Norm
- z-loss

第 4 讲则转向：

$$
\text{Attention Alternatives and Mixtures of Experts}
$$

也就是当标准 dense Transformer 面对更长上下文和更大参数规模时，如何进一步改造架构。

这节课的核心不是简单介绍几个新架构名字，而是在解释：

$$
\boxed{
\text{未来 LLM 架构优化}
=
\text{更便宜的上下文建模}
+
\text{更稀疏的参数激活}
+
\text{更复杂的系统调度}
}
$$

可以先用一句话概括：

> Attention alternatives 解决的是：每个 token 要看多少历史？
>
> MoE 解决的是：每个 token 要激活多少参数？

---

## 1. 标准 Transformer 的两个扩展瓶颈

标准 decoder-only Transformer block 大致是：

$$
x
\rightarrow
\text{Self-Attention}
\rightarrow
\text{MLP}
$$

更具体地写，是：

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

这两个部分各自有不同的扩展瓶颈。

Attention 的瓶颈主要来自序列长度。

标准 scaled dot-product attention 是：

$$
\text{Attention}(Q,K,V)
=
\text{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

设序列长度为 $n$，则：

$$
Q,K,V \in \mathbb{R}^{n \times d}
$$

那么：

$$
QK^\top \in \mathbb{R}^{n \times n}
$$

也就是说，attention matrix 的大小是：

$$
n^2
$$

所以 full attention 的核心问题是：

$$
\boxed{
\text{context length 增长}
\Rightarrow
\text{attention cost 二次增长}
}
$$

如果上下文从 $4K$ 增长到 $64K$，长度增长 $16$ 倍，但 attention 相关成本近似增长：

$$
16^2=256
$$

这就是为什么长上下文模型不能只靠原始 full attention 硬撑。

MLP 的瓶颈则来自参数激活。

Dense MLP 中，每个 token 都要经过同一套完整 FFN 参数：

$$
y
=
\text{MLP}(x)
$$

如果模型参数越来越大，每个 token 激活的参数量也会跟着变大。

所以标准 Transformer 的两个主要问题可以写成：

$$
\text{Attention}:
\quad
\text{每个 token 看所有历史 token}
$$

$$
\text{MLP}:
\quad
\text{每个 token 激活所有 FFN 参数}
$$

Lecture 4 的两条主线正好对应这两个问题：

$$
\boxed{
\text{Attention Alternatives}
\Rightarrow
\text{让每个 token 不必看所有历史}
}
$$

$$
\boxed{
\text{Mixture of Experts}
\Rightarrow
\text{让每个 token 不必激活所有参数}
}
$$

---

## 2. Attention Alternatives 的基本动机

标准 attention 的语义是：

> 对每个 query token，和所有 key token 做相似度，然后通过 softmax 得到一个动态分布，再加权汇聚 value。

公式是：

$$
\alpha_{ij}
=
\frac{
\exp(q_i^\top k_j / \sqrt{d_k})
}{
\sum_m \exp(q_i^\top k_m / \sqrt{d_k})
}
$$

输出是：

$$
o_i
=
\sum_j \alpha_{ij}v_j
$$

这很强，因为每个 token 都可以动态选择它想看的历史 token。

比如模型需要从很远的上下文中找回某个名字、代码变量、数学条件，full attention 理论上可以直接把 attention mass 放到对应位置。

但代价也很明显：

$$
\text{每个 query}
\rightarrow
\text{看所有 key}
$$

所以成本是：

$$
O(n^2)
$$

Lecture 4 开头提到一些比较基础的工具：

- local attention
- global attention
- sparse pattern
- FlashAttention
- systems engineering
- 更高效的 kernel

这些方法本质上仍然保留 attention 的基本形式，只是在 attention pattern 或系统实现上做优化。

但这一讲真正关注的是更激进的路线：

$$
\text{full attention}
\rightarrow
\text{linear attention}
\rightarrow
\text{recurrent attention}
\rightarrow
\text{Mamba / Gated DeltaNet}
\rightarrow
\text{sparse attention}
$$

它们都在试图回答一个问题：

$$
\boxed{
\text{能不能不显式构造 } n \times n \text{ 的 attention matrix？}
}
$$

---

## 3. Linear Attention：把 $QK^\top V$ 改写成 $Q(K^\top V)$

标准 attention 可以抽象成：

$$
\text{Attn}(Q,K,V)
=
\rho(QK^\top)V
$$

其中 $\rho$ 通常是 softmax。

如果暂时忽略 softmax，或者把 $\rho$ 换成某种可以分解的 kernel function，那么可以利用矩阵乘法结合律：

$$
QK^\top V
=
Q(K^\top V)
$$

这一步看起来非常简单，但它改变了计算结构。

原始写法是先算：

$$
QK^\top:
\mathbb{R}^{n \times d_k}
\times
\mathbb{R}^{d_k \times n}
\rightarrow
\mathbb{R}^{n \times n}
$$

也就是先产生 token-token 关系矩阵。

再乘以：

$$
V \in \mathbb{R}^{n \times d_v}
$$

所以复杂度里会出现：

$$
n^2d_k+n^2d_v
$$

如果改成：

$$
K^\top V:
\mathbb{R}^{d_k \times n}
\times
\mathbb{R}^{n \times d_v}
\rightarrow
\mathbb{R}^{d_k \times d_v}
$$

则先把所有历史 key-value 信息压缩成一个固定大小的矩阵。

然后再算：

$$
Q(K^\top V)
$$

复杂度大约变成：

$$
2nd_kd_v
$$

于是序列长度相关复杂度从：

$$
O(n^2)
$$

变成：

$$
O(n)
$$

这就是 linear attention 的核心。

但必须注意，这不是免费的午餐。

标准 attention 是：

$$
\text{每个 token 动态选择历史 token}
$$

linear attention 是：

$$
\text{把历史压缩进固定大小状态}
$$

这意味着：

$$
\boxed{
\text{linear attention 用效率换掉了一部分精确检索能力}
}
$$

---

## 4. Linear Attention 的 recurrent form

在 causal language modeling 中，linear attention 可以写成递推形式。

定义状态矩阵：

$$
S_t
=
S_{t-1}
+
k_tv_t^\top
$$

输出为：

$$
y_t
=
q_t^\top S_t
$$

其中：

$$
S_t \in \mathbb{R}^{d_k \times d_v}
$$

这时候 $S_t$ 可以理解为截至第 $t$ 个 token 的历史压缩状态。

每来一个 token，就往状态里写入：

$$
k_tv_t^\top
$$

当前 query 则通过：

$$
q_t^\top S_t
$$

读取历史。

这和 RNN 非常相似。

RNN 的形式是：

$$
h_t
=
f(h_{t-1},x_t)
$$

linear attention 的形式是：

$$
S_t
=
f(S_{t-1},k_t,v_t)
$$

区别在于，RNN 的 hidden state 通常是向量，而这里的状态 $S_t$ 是矩阵。

这个 recurrent form 有一个重要优势：

> 训练时可以用并行形式，推理时可以用递推形式。

标准 attention 推理时需要保存所有历史 KV：

$$
K_{1:t},V_{1:t}
$$

也就是 KV cache。

而 linear attention 推理时只需要保存：

$$
S_t
$$

其状态大小和上下文长度无关。

所以从推理角度看，linear attention 非常适合长上下文，因为它避免了 KV cache 随上下文长度无限增长。

---

## 5. Linear Attention 的根本缺陷

Linear attention 的问题在于状态更新太简单：

$$
S_t
=
S_{t-1}
+
k_tv_t^\top
$$

它只会累加，不会主动判断：

- 哪些信息应该保留？
- 哪些信息应该遗忘？
- 当前 token 是否应该写入？
- 新旧信息冲突时怎么处理？
- 远处的某个 token 如何被精确找回？

标准 attention 的 softmax 有一个强能力：

$$
\text{每个 query 可以重新选择历史}
$$

也就是说，对于不同 query，模型可以产生不同的 attention distribution。

但 linear attention 把历史统一压缩成 $S_t$，这会削弱 token-level 的动态选择能力。

所以后续很多 attention alternatives 的核心都可以看成：

$$
\boxed{
\text{如何在保留 linear/recurrent 效率的同时，增强状态更新能力？}
}
$$

这就引出了 Mamba-2 和 Gated DeltaNet。

---

## 6. Mamba-2：给状态加入输入相关的遗忘门

纯 linear attention 的状态更新是：

$$
S_t
=
S_{t-1}
+
k_tv_t^\top
$$

Mamba-2 可以理解为给这个状态更新加入 gate：

$$
S_t
=
\gamma_tS_{t-1}
+
k_tv_t^\top
$$

输出为：

$$
y_t
=
q_t^\top S_t
+
v_t^\top D
$$

其中：

$$
\gamma_t=f(x_t)
$$

$\gamma_t$ 是由当前输入决定的衰减因子。

当：

$$
\gamma_t \approx 1
$$

模型倾向于保留旧状态。

当：

$$
\gamma_t \approx 0
$$

模型倾向于忘掉旧状态。

所以 Mamba-2 的核心不是单纯线性累加，而是：

$$
\text{fixed recurrent state}
+
\text{input-dependent forgetting}
$$

可以用一句话理解：

$$
\boxed{
\text{Mamba-2}
\approx
\text{linear attention}
+
\text{动态遗忘门}
}
$$

它试图在 full attention 和 RNN/SSM 之间做折中：

- 比 full attention 更便宜；
- 比普通 RNN 更适合并行训练；
- 比纯 linear attention 多了输入相关的状态控制能力。

但它仍然不是 full attention。

它擅长连续状态建模，但不一定擅长从很远的上下文中精确复制某个 token。

---

## 7. Gated DeltaNet：不仅遗忘，还能擦除特定方向

Gated DeltaNet 进一步增强了状态更新。

它可以写成：

$$
S_t
=
\gamma_t
\left(
I-\beta_t k_tk_t^\top
\right)
S_{t-1}
+
\beta_t k_tv_t^\top
$$

输出为：

$$
y_t
=
q_t^\top S_t
$$

其中：

$$
\gamma_t=f(x_t)
$$

$$
\beta_t=f(x_t)
$$

这里有两个 gate。

$\gamma_t$ 控制旧状态整体保留多少。

$\beta_t$ 控制当前 token 写入强度。

更关键的是：

$$
I-\beta_t k_tk_t^\top
$$

这一项可以理解为：在写入当前 key-value 信息之前，先擦除旧状态中和当前 key 方向相关的部分。

普通 linear attention 是：

$$
S_t
=
S_{t-1}
+
k_tv_t^\top
$$

它只会堆积信息。

Gated DeltaNet 则是：

$$
\text{erase old information}
+
\text{write new information}
$$

也就是说，它让固定大小状态更像一个可编辑的外部记忆。

这一点和 fast weight programming、test-time training 的思想有关系。

因为：

$$
q_t^\top S_t
$$

可以看成当前 query 去读取一个在推理过程中不断更新的临时矩阵 $S_t$。

从这个角度看，$S_t$ 有点像一种动态形成的 fast weight。

---

## 8. 为什么很多模型采用 Hybrid，而不是完全替代 Attention

虽然 linear attention、Mamba-2、Gated DeltaNet 都很有吸引力，但完全抛弃 full attention 风险很大。

原因是：

$$
\boxed{
\text{full attention 提供 token-level 精确检索能力}
}
$$

这对很多任务非常重要：

- 长程引用
- 代码补全
- 数学推理
- 字符串复制
- needle-in-a-haystack
- 多段落证据对齐
- retrieval-like reasoning

如果完全使用 recurrent / linear state，历史信息被压缩进固定大小状态，可能无法稳定保留所有细节。

所以很多新模型采用 hybrid 结构：

$$
\text{若干 linear / recurrent layers}
+
\text{少量 full attention layers}
$$

直觉上：

- linear / recurrent 层负责便宜地吸收和压缩长上下文；
- full attention 层负责必要时做精确 token 检索；
- hybrid ratio 决定效率和能力之间的平衡。

所以一个更现实的判断是：

> 未来不一定是 full attention 被彻底消灭，而是 full attention 从“每层都有”变成“少量关键层保留”。

这也是架构设计里的典型 trade-off：

$$
\text{效率}
\leftrightarrow
\text{表达力}
$$

---

## 9. Sparse Attention：不压缩历史，而是选择历史

除了 linear / recurrent 路线，还有 sparse attention 路线。

Linear attention 的做法是：

$$
\text{全部历史}
\rightarrow
\text{固定大小状态}
$$

Sparse attention 的做法是：

$$
\text{全部历史 token}
\rightarrow
\text{少量候选 token}
$$

也就是说，sparse attention 不一定把历史压缩成一个状态，而是先选择一部分历史 token，然后只对这些 token 做 attention。

设被选中的 token 集合为：

$$
\mathcal{I}
$$

那么 sparse attention 可以写成：

$$
y_t
=
\text{Attention}(q_t,K_{\mathcal{I}},V_{\mathcal{I}})
$$

如果：

$$
|\mathcal{I}|=k
$$

并且：

$$
k \ll n
$$

那么每个 token 的 attention 成本可以从：

$$
O(n)
$$

降到：

$$
O(k)
$$

总体复杂度也从：

$$
O(n^2)
$$

变成近似：

$$
O(nk)
$$

Sparse attention 和 linear attention 的哲学不同。

Linear attention 牺牲的是 token-level 可访问性：

$$
\text{history}
\rightarrow
\text{compressed state}
$$

Sparse attention 牺牲的是覆盖率：

$$
\text{all tokens}
\rightarrow
\text{selected tokens}
$$

如果任务需要精确找回某处信息，sparse attention 可能比 pure recurrent 更自然。

因为它仍然保留 token-to-token attention 的形式，只是把候选集合变小。

但 sparse attention 的问题也很明显：

> 如果 token 选择机制选错了，后面的 attention 再强也没用。

所以 sparse attention 的关键不是 attention 本身，而是 selector / indexer 是否可靠。

---

## 10. Attention Alternatives 的统一理解

可以按照“历史信息怎么存、怎么读”来理解 attention alternatives。

### 10.1 Full Attention

Full attention 保存所有历史 KV：

$$
K_{1:t},V_{1:t}
$$

读取时，每个 query 可以看所有历史 token：

$$
q_t
\rightarrow
K_{1:t},V_{1:t}
$$

优点是表达力强，支持精确检索。

缺点是上下文越长，成本越高。

---

### 10.2 Linear / Recurrent Attention

Linear / recurrent attention 不保存所有 KV，而是保存固定状态：

$$
S_t
$$

读取时：

$$
y_t
=
q_t^\top S_t
$$

优点是推理状态大小不随上下文长度增长。

缺点是历史被压缩，精确检索能力弱。

---

### 10.3 Gated Recurrent Models

Mamba-2、Gated DeltaNet 等模型在 recurrent state 上加入 gate：

$$
S_t
=
\text{Update}(S_{t-1},x_t,k_t,v_t)
$$

它们试图增强固定状态的动态更新能力。

优点是比纯 linear attention 更强。

缺点是仍然不能完全替代 full attention 的精确 token access。

---

### 10.4 Sparse Attention

Sparse attention 仍然保留 token-level attention，但只选择一部分 token：

$$
\mathcal{I}
=
\text{Select}(q_t,K_{1:t})
$$

$$
y_t
=
\text{Attention}(q_t,K_{\mathcal{I}},V_{\mathcal{I}})
$$

优点是保留精确检索形式。

缺点是依赖选择机制；选错 token 就会损失信息。

---

## 11. Mixture of Experts：从 dense activation 到 sparse activation

Lecture 4 的后半部分讲 MoE。

标准 dense MLP 是：

$$
y
=
\text{MLP}(x)
$$

每个 token 都经过同一套完整 MLP 参数。

MoE 则把一个大 MLP 换成多个 expert：

$$
E_1,E_2,\ldots,E_N
$$

然后用 router 决定每个 token 应该送给哪些 expert：

$$
s(x)
=
\text{Router}(x)
$$

$$
\mathcal{E}(x)
=
\text{TopK}(s(x),k)
$$

最终输出是选中 experts 的加权和：

$$
y
=
\sum_{i \in \mathcal{E}(x)}
g_i(x)E_i(x)
$$

其中：

- $E_i$ 是第 $i$ 个 expert；
- $g_i(x)$ 是 router 给第 $i$ 个 expert 的权重；
- $\mathcal{E}(x)$ 是被选中的 expert 集合。

MoE 的关键不是“有很多 expert”，而是：

$$
\text{total parameters} \uparrow
$$

但：

$$
\text{active parameters per token}
\approx
\text{constant}
$$

这就是 MoE 的核心思想：

$$
\boxed{
\text{模型总容量变大，但每个 token 只激活一小部分参数}
}
$$

---

## 12. 为什么 MoE 有吸引力

MoE 的吸引力主要来自三个方面。

### 12.1 更大的总容量

Dense model 中，如果扩大参数量，每个 token 的计算量通常也会变大。

MoE 中，可以通过增加 expert 数量扩大总参数量，但每个 token 仍然只选择 top-$k$ experts。

因此 MoE 可以实现：

$$
\text{model capacity} \uparrow
$$

同时尽量控制：

$$
\text{FLOPs per token}
$$

---

### 12.2 更好的 compute-performance trade-off

同样 FLOPs 下，sparse parameters 往往可以带来更好的 loss。

直觉上，不同 experts 可以学习不同类型的模式。

例如：

- 某些 expert 更擅长代码；
- 某些 expert 更擅长数学；
- 某些 expert 更擅长自然语言；
- 某些 expert 更擅长特定语种；
- 某些 expert 更擅长结构化文本。

当然，这种 specialization 不是人工指定的，而是 router 和训练动态共同形成的。

MoE 的目标不是让模型“少参数”，而是让模型有更多参数可以条件化调用。

---

### 12.3 更适合多设备扩展

MoE 的 experts 可以分布在不同 GPU 上。

例如：

$$
E_1,E_2,E_3,E_4
$$

放在 GPU 0 上，

$$
E_5,E_6,E_7,E_8
$$

放在 GPU 1 上。

当 token 被 router 分配给某个 expert 时，就把 token dispatch 到对应 GPU 上计算。

这就是 expert parallelism。

它让 MoE 具备天然的多设备扩展能力。

但同时也带来了通信问题。

---

## 13. MoE 的真实难点：routing、load balancing、communication

MoE 看起来很简单：

$$
\text{token}
\rightarrow
\text{router}
\rightarrow
\text{expert}
$$

但真正困难的是：

$$
\boxed{
\text{routing}
+
\text{load balancing}
+
\text{all-to-all communication}
}
$$

---

## 14. Routing：谁来决定 token 去哪个 expert？

Router 的作用是给每个 expert 一个分数：

$$
s
=
xW_g
$$

然后选择 top-$k$ experts：

$$
\mathcal{E}
=
\text{TopK}(s,k)
$$

如果 $k=2$，每个 token 只会送给两个 experts。

这种方式很高效，但也有问题。

如果 router 总是选择少数几个 experts，那么：

$$
\text{少数 experts 过载}
$$

而：

$$
\text{多数 experts 几乎不用}
$$

这会导致 expert collapse。

---

## 15. Load Balancing：MoE 稳定训练的核心

MoE 最大的问题之一就是 expert collapse。

也就是：

$$
\text{少数 experts 被大量选择}
$$

而：

$$
\text{多数 experts 几乎收不到 token}
$$

这会带来很多问题：

- expert 训练不均；
- GPU 负载不均；
- 某些 expert 过载；
- 某些 token 被 drop；
- all-to-all 通信恶化；
- 训练不稳定。

所以 MoE 通常需要 load balancing loss 或其他 balancing mechanism。

一种直觉目标是：

$$
\text{每个 expert 接收的 token 数量尽量接近}
$$

但这和 expert specialization 有冲突。

如果强行均匀，expert 很难形成专门化能力。

如果完全自由，router 又可能塌缩到少数 expert。

所以 MoE 的核心矛盾是：

> router 要足够自由，才能让 expert 专门化；
>
> router 又不能太自由，否则负载会崩。

这个矛盾贯穿了几乎所有 MoE 设计。

---

## 16. Communication：MoE 把 compute 问题转移成通信问题

在多 GPU 上，MoE 通常需要 all-to-all communication。

大致过程是：

1. 每张 GPU 上有一批 token；
2. router 判断每个 token 要去哪些 experts；
3. 如果 expert 在其他 GPU 上，就把 token 发过去；
4. expert 计算完后，再把结果发回原来的位置；
5. 最后把多个 expert 输出 combine 起来。

这个过程可以抽象为：

$$
\text{token dispatch}
\rightarrow
\text{expert computation}
\rightarrow
\text{result combine}
$$

所以 MoE 并不是单纯减少计算。

更准确地说，它是把一部分 dense compute 压力转移成：

$$
\text{routing}
+
\text{communication}
+
\text{load balancing}
$$

如果网络带宽不足，或者 token 分布不均，MoE 的通信成本可能抵消计算收益。

所以 MoE 是模型结构和分布式系统的结合，而不是单纯的神经网络模块。

---

## 17. DeepSeek MoE 的设计意义

Lecture 4 中 DeepSeek MoE 是一个重要案例。

它的设计里有几个关键点：

- shared experts；
- routed experts；
- fine-grained experts；
- expert-level load balancing；
- device-level load balancing。

---

## 18. Shared Experts

Shared experts 是所有 token 都会经过的公共专家。

可以写成：

$$
y_{\text{shared}}
=
\sum_j E_j^{\text{shared}}(x)
$$

它的作用是提供稳定的通用能力。

如果所有能力都依赖 router 动态选择，那么训练早期 router 不稳定时，模型可能很难学好。

Shared experts 提供了一条公共路径，相当于给 MoE 一个 dense backbone。

---

## 19. Routed Experts

Routed experts 是由 router 动态选择的专家：

$$
y_{\text{routed}}
=
\sum_{i \in \text{TopK}}
g_i(x)E_i(x)
$$

它们负责更细粒度的 token specialization。

最终输出可以理解成：

$$
y
=
y_{\text{shared}}
+
y_{\text{routed}}
$$

Shared experts 负责保底，routed experts 负责动态能力。

---

## 20. Fine-Grained Experts

Fine-grained experts 是把 expert 切得更细。

相比少量大 expert，更多小 expert 的好处是组合更灵活。

如果有 $N$ 个 experts，每个 token 选 $k$ 个，那么 token 可以形成很多不同组合。

这有点像：

$$
\text{many small modules}
\rightarrow
\text{combinatorial capacity}
$$

但代价是 routing 和通信更复杂。

所以 fine-grained experts 的本质也是 trade-off：

$$
\text{更细粒度的专家组合}
\leftrightarrow
\text{更复杂的调度和通信}
$$

---

## 21. Expert-Level 和 Device-Level Balancing

MoE 不只要考虑 expert-level balance，还要考虑 device-level balance。

Expert-level balance 是：

$$
\text{每个 expert 接收 token 数尽量合理}
$$

Device-level balance 是：

$$
\text{每个 GPU 承担的总 token 数尽量合理}
$$

为什么 device-level balance 重要？

因为如果某些 GPU 上放的 experts 被频繁选中，即使 expert 本身没有完全 collapse，系统层面仍然会出现负载不均。

所以大规模 MoE 要同时考虑：

$$
\text{expert balance}
$$

和：

$$
\text{device balance}
$$

这说明 MoE 不是纯算法问题，而是模型结构和分布式系统的结合。

---

## 22. Dense Model 和 MoE 的关键差别

Dense model 是：

$$
\text{每个 token 激活所有参数}
$$

MoE 是：

$$
\text{每个 token 只激活 top-}k\text{ experts}
$$

Dense model 的扩展方式是：

$$
\text{parameters} \uparrow
\Rightarrow
\text{FLOPs per token} \uparrow
$$

MoE 的扩展方式是：

$$
\text{total parameters} \uparrow
$$

但：

$$
\text{active parameters per token}
\approx
\text{constant}
$$

所以 MoE 的目标不是简单减少参数，而是改变参数使用方式。

它让模型具有更大的总容量，但保持相对可控的每 token 计算成本。

---

## 23. Attention Alternatives 和 MoE 的统一视角

这节课看似有两个主题：

1. Attention alternatives
2. Mixture of Experts

但它们其实都在做同一件事：

$$
\boxed{
\text{让每个 token 不再使用全部资源}
}
$$

标准 dense Transformer 是：

$$
\text{每个 token 看所有历史 token}
$$

并且：

$$
\text{每个 token 激活所有 MLP 参数}
$$

Attention alternatives 改的是第一句：

$$
\text{每个 token 不一定看所有历史 token}
$$

MoE 改的是第二句：

$$
\text{每个 token 不一定激活所有参数}
$$

所以这节课的深层主题是：

$$
\boxed{
\text{sparse computation}
}
$$

也就是：

$$
\text{selective context}
+
\text{selective parameters}
$$

---

## 24. 从系统角度看这节课

Lecture 4 和前几节课联系非常紧。

Lecture 2 讲 resource accounting，强调训练和推理都要算：

- compute
- memory
- bandwidth
- wall-clock time

Lecture 3 讲现代 Transformer 架构，解释 RoPE、RMSNorm、SwiGLU、GQA 等设计背后的 trade-off。

Lecture 4 则进一步问：

> 如果标准 dense Transformer 即使用了这些现代组件，还是太贵怎么办？

答案不是简单地“优化代码”，而是改变计算图本身。

Attention alternatives 改变的是 attention 的计算图。

MoE 改变的是 FFN 的计算图。

二者都不是单纯减少 FLOPs，而是在不同资源之间做 trade-off。

Attention alternatives 的 trade-off 是：

$$
\text{compute / memory cost}
\downarrow
$$

但可能导致：

$$
\text{precise retrieval ability}
\downarrow
$$

MoE 的 trade-off 是：

$$
\text{active compute}
\downarrow
$$

但会增加：

$$
\text{routing complexity}
+
\text{communication cost}
+
\text{load balancing difficulty}
$$

所以这节课真正训练的是一种系统判断：

$$
\boxed{
\text{架构设计不是只看数学表达力，也不是只看 FLOPs，}
\text{而是看 compute、memory、communication、optimization stability 和 final quality 的综合平衡。}
}
$$

---

## 25. 对 inference acceleration 的启发

如果目标是做 LLM 推理加速，Lecture 4 有很强的启发意义。

不能只问：

> 这个方法减少了多少 FLOPs？

还要问：

$$
\text{是否减少 KV cache 读写？}
$$

$$
\text{是否降低长上下文 attention 成本？}
$$

$$
\text{是否提高 arithmetic intensity？}
$$

$$
\text{是否引入额外通信？}
$$

$$
\text{是否破坏 batch shape？}
$$

$$
\text{是否让调度变复杂？}
$$

很多方法理论上减少计算，但实际不一定加速。

例如 MoE 虽然减少 active parameters，但可能增加 all-to-all communication。

Sparse attention 虽然减少 attention tokens，但 selector / indexer 可能引入额外开销。

Linear attention 虽然减少 KV cache，但可能损失精确检索能力。

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
\text{更低通信开销}
+
\text{可接受的精度损失}
}
$$

---

## 26. 对 long-context modeling 的启发

长上下文建模有三条主要路线。

### 26.1 保留 full attention，但优化实现

例如：

- FlashAttention
- paged attention
- KV cache optimization
- local/global attention pattern

这类方法的优点是保留 full attention 能力，缺点是上下文极长时仍然压力很大。

---

### 26.2 用 recurrent / linear state 替代 KV cache

例如：

- linear attention
- Mamba-2
- Gated DeltaNet

这类方法的优点是推理状态不随上下文长度增长。

缺点是历史被压缩，精确检索能力可能下降。

---

### 26.3 用 sparse attention 选择历史 token

例如：

- learned sparse attention
- indexer-based sparse attention
- DeepSeek Sparse Attention 类思路

这类方法的优点是保留 token-level attention 形式。

缺点是依赖选择机制，选错 token 就会丢信息。

所以长上下文建模的本质不是单一技术选择，而是：

$$
\boxed{
\text{compression}
\quad
\text{vs}
\quad
\text{selection}
\quad
\text{vs}
\quad
\text{full retrieval}
}
$$

---

## 27. 对 MoE scaling 的启发

MoE 的关键不是“专家越多越好”。

真正要看：

$$
\text{routing 是否稳定}
$$

$$
\text{experts 是否形成有效分工}
$$

$$
\text{负载是否均衡}
$$

$$
\text{all-to-all 通信是否可控}
$$

$$
\text{batch size 是否足够支撑 expert parallelism}
$$

如果 batch 太小，每个 expert 分到的 token 太少，MoE 的矩阵乘法可能很碎，GPU 利用率会下降。

如果网络通信慢，expert dispatch 的 all-to-all 会成为瓶颈。

如果 router 训练不好，expert 可能 collapse。

所以 MoE scaling 的本质是：

$$
\boxed{
\text{model scaling}
+
\text{routing optimization}
+
\text{distributed systems}
}
$$

这也是为什么 MoE 更像一个系统工程问题，而不是单纯的 layer replacement。

---

## 28. 总结

Lecture 4 的核心可以浓缩成一句话：

$$
\boxed{
\text{现代 LLM 架构优化，不是让每个 token 更暴力地使用全部资源，}
\text{而是让每个 token 动态选择它真正需要的历史信息和参数子集。}
}
$$

Attention alternatives 解决的是：

$$
\boxed{
\text{每个 token 要看多少历史？}
}
$$

MoE 解决的是：

$$
\boxed{
\text{每个 token 要激活多少参数？}
}
$$

标准 dense Transformer 是：

$$
\text{all tokens attend to all tokens}
$$

并且：

$$
\text{all tokens use all FFN parameters}
$$

Lecture 4 讨论的是更稀疏、更动态、更系统依赖的模型形态：

$$
\text{selective context}
+
\text{selective parameters}
+
\text{routing}
+
\text{state}
+
\text{communication-aware design}
$$

第 3 讲告诉我们：

$$
\text{现代 Transformer 是在稳定性、参数预算、访存效率和推理约束下演化出来的。}
$$

第 4 讲进一步告诉我们：

$$
\text{当 dense Transformer 继续扩展遇到瓶颈时，必须引入更强的动态资源分配机制。}
$$

所以这节课最终形成的判断是：

$$
\boxed{
\text{未来 LLM 的能力扩展，不只是堆参数和堆上下文，}
\text{而是设计更好的 memory、routing、sparsity 和 communication trade-off。}
}
$$