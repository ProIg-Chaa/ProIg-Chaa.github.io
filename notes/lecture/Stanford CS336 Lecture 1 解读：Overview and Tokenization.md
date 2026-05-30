# Stanford CS336 Lecture 1 解读：Overview and Tokenization

**ps:其实这节课还有一些前瞻的，抑或是关于gpu架构的闲谈，但是我就没有记录。之后我应该会出专门按gpu/训练/推理等支线形式更新的系列**

视频链接：<https://www.youtube.com/watch?v=SQ3fZ1sAqXI&list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_&index=1>

这节课对应 **Stanford CS336: Language Modeling from Scratch, Lecture 1: Overview and Tokenization**。

这一讲的核心是：

$$
\text{Language Modeling from Scratch}
$$

也就是从零理解并搭建语言模型系统，而不是只停留在调用现成 API 或背 Transformer 结构图。

这节课主要建立两个判断：

1. 语言模型不是一个孤立的神经网络，而是一整套端到端系统；
2. tokenizer 不是普通预处理工具，而是决定模型输入空间、训练效率、上下文利用率和多语言能力的基础组件。

可以用一句话概括：

> LLM 的所有系统成本，都是从 token 开始计账的。

---

## 1. CS336 的课程定位

CS336 的标题是：

$$
\text{Language Modeling from Scratch}
$$

这里的 “from scratch” 不是说完全不用 PyTorch，也不是说所有底层 kernel 都要手写，而是指：

> 不把语言模型当成黑盒，而是把数据、tokenizer、Transformer、loss、optimizer、training loop、evaluation、inference 全部拆开理解。

普通 NLP 课程更关注：

- Transformer 是什么；
- attention 怎么算；
- BERT/GPT 怎么用；
- 下游任务怎么 fine-tune。

但 CS336 更像一门 **LLM 系统构建课**。

它关心的问题是：

> 如果让我自己造一个小型 GPT，我需要实现哪些部件？每个部件为什么这样设计？系统瓶颈在哪里？

整条主线可以写成：

$$
\text{raw text} \rightarrow \text{tokens} \rightarrow \text{Transformer} \rightarrow \text{loss} \rightarrow \text{optimizer} \rightarrow \text{training loop} \rightarrow \text{generation}
$$

Lecture 1 是这条链路的入口。

---

## 2. 语言模型到底在建模什么

语言模型的目标是给一段文本序列赋概率。

设 token 序列为：

$$
x_1,x_2,\ldots,x_T
$$

语言模型要建模的是联合概率：

$$
p(x_1,x_2,\ldots,x_T)
$$

根据概率链式法则：

$$
p(x_1,x_2,\ldots,x_T)=\prod_{t=1}^{T}p(x_t\mid x_1,x_2,\ldots,x_{t-1})
$$

也可以简写为：

$$
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t})
$$

所以语言模型可以转化为一个不断预测下一个 token 的模型：

$$
p(x_t\mid x_{<t})
$$

这就是 **next-token prediction**。

现代 decoder-only LLM 的训练目标，本质上就是：

> 给定前面的 token，预测下一个 token。

例如一句话：

~~~text
The cat sat on the mat.
~~~

经过 tokenizer 之后变成：

$$
x_1,x_2,x_3,\ldots,x_T
$$

训练时模型看到：

$$
x_1,\ldots,x_{t-1}
$$

然后预测：

$$
x_t
$$

所以语言模型的监督信号不是人工标注出来的，而是文本本身天然提供的。

这就是自监督学习的核心：

> 输入文本本身就提供了训练标签。

---

## 3. 为什么语言模型需要 tokenization

神经网络不能直接处理字符串。

原始文本是：

~~~text
hello world
~~~

模型真正需要的是整数序列：

$$
[15496,995]
$$

或者更一般地：

$$
\text{text}\rightarrow\text{token IDs}\rightarrow\text{embedding vectors}\rightarrow\text{Transformer}
$$

tokenizer 的作用就是把字符串映射成整数序列：

$$
\text{Tokenizer}:\text{string}\rightarrow\mathbb{N}^{T}
$$

然后 embedding layer 把整数映射成向量：

$$
\text{Embedding}:\mathbb{N}^{T}\rightarrow\mathbb{R}^{T\times d_{\text{model}}}
$$

所以 tokenizer 是语言模型的入口。

如果 tokenizer 设计得不好，后面的 Transformer 再强也会受影响。

比如：

- token 序列过长，训练和推理成本变高；
- 中文、代码、数学符号被切得很碎，上下文窗口浪费；
- rare word 处理不好，泛化能力下降；
- special token 处理不好，文档边界和控制符会混乱；
- decode 不可逆，生成结果可能损坏。

可以把 tokenizer 理解为：

> tokenizer 决定了模型“看见文本”的方式。

---

## 4. 字符、Unicode 和 byte 的区别

很多人第一次学 tokenizer 会以为：

> 一个字符就是一个 token。

这个想法不够准确。

在计算机里，文本至少涉及三层概念：

| 层次               | 含义                   | 例子             |
| ------------------ | ---------------------- | ---------------- |
| 字符               | 人类看到的符号         | `牛`, `a`, `🙂`   |
| Unicode code point | 字符对应的整数编号     | `牛` 对应 U+725B |
| byte encoding      | 字符在内存中的字节表示 | UTF-8 bytes      |

Unicode 解决的问题是：

> 给世界上不同语言、符号、emoji 分配统一编号。

例如：

$$
\operatorname{ord}(\text{“牛”})=29275
$$

但是模型不适合直接在 Unicode code point 上训练 tokenizer。

原因是 Unicode code point 空间太大，而且非常稀疏。

如果直接把 Unicode 字符当基础词表，词表会非常大：

$$
|\mathcal{V}_{\text{unicode}}|\approx 10^5
$$

但大量字符极少出现。

这会导致：

- embedding 参数浪费；
- 低频字符训练不足；
- 多语言处理复杂；
- OOV 或稀疏问题明显。

所以现代 tokenizer 常用一个更底层的方案：

$$
\text{Unicode string}\rightarrow\text{UTF-8 bytes}\rightarrow\text{subword tokens}
$$

---

## 5. 为什么使用 UTF-8 bytes

UTF-8 会把任意 Unicode 字符编码成一个或多个 byte。

byte 的取值范围是：

$$
0,1,2,\ldots,255
$$

所以 byte-level tokenizer 的基础词表大小只有：

$$
256
$$

这有一个极其重要的性质：

> 任何文本都可以表示成 byte 序列。

也就是说，只要 tokenizer 以 byte 为基础，就不会真正遇到 unknown token。

这比 word-level tokenizer 更稳健。

word-level tokenizer 的问题是：

- 词表外单词无法处理；
- 拼写变化会造成 OOV；
- 多语言和代码符号很麻烦。

character-level tokenizer 也有问题：

- token 序列太长；
- 英文单词被拆得太碎；
- 长程依赖更难学。

byte-level tokenizer 的优点是：

$$
\text{byte-level tokenizer}=\text{no OOV}+\text{small base vocabulary}+\text{universal text coverage}
$$

但 byte-level 本身也有缺点：

$$
\text{sequence length is long}
$$

例如英文单词：

~~~text
hello
~~~

如果按 byte 切，就是：

$$
[h,e,l,l,o]
$$

一个词变成 5 个 token。

如果是中文字符，一个字符通常会变成 3 个 UTF-8 bytes。

所以仅用 byte 不够。

需要进一步压缩。

这就引出 BPE。

---

## 6. tokenizer 的核心矛盾：词表大小 vs 序列长度

tokenizer 设计里有一个基本 trade-off：

$$
\text{vocab size}\leftrightarrow\text{sequence length}
$$

如果词表很小，比如 byte-level 的 256 个基础 token：

$$
|\mathcal{V}|=256
$$

那么任何文本都能表示，但是序列会很长。

如果词表很大，比如 word-level tokenizer：

$$
|\mathcal{V}|\gg 10^5
$$

那么常见词可以一个 token 表示，序列更短，但 rare word、拼写变化、多语言会很难处理。

Subword tokenizer 处在中间：

$$
\text{byte/character}\longleftrightarrow\text{subword}\longleftrightarrow\text{word}
$$

它的思路是：

> 常见片段合成大 token，罕见片段退回小 token。

例如：

~~~text
unbelievable
~~~

可能被切成：

~~~text
un + believable
~~~

或者：

~~~text
un + believe + able
~~~

这样既能压缩常见模式，又能处理新词。

---

## 7. BPE 的核心思想

BPE，全称 **Byte-Pair Encoding**。

它原本是一种压缩算法，后来被用于 subword tokenization。

核心过程很简单：

1. 从 byte-level vocabulary 开始；
2. 统计语料中相邻 token pair 的频率；
3. 找出最高频 pair；
4. 把这个 pair 合并成一个新 token；
5. 重复这个过程，直到达到目标 vocab size。

可以写成：

$$
(a,b)\rightarrow ab
$$

如果某个 pair：

$$
(a,b)
$$

在语料中出现次数最高，就把它合并成新 token：

$$
c=ab
$$

然后加入词表：

$$
\mathcal{V}\leftarrow\mathcal{V}\cup\{c\}
$$

反复执行后，词表从：

$$
256
$$

逐渐增长到：

$$
|\mathcal{V}|=\text{target vocab size}
$$

BPE 的本质是：

> 用频率最高的相邻片段合并来压缩文本。

它不是根据语义手工切词，而是根据 corpus statistics 自动学习 subword units。

---

## 8. 一个 BPE 合并例子

假设语料里有：

~~~text
low low low low low
lower lower
widest widest widest
newest newest newest newest newest newest
~~~

经过初始切分后，可以把每个词看成 byte 或 character 序列：

~~~text
low    -> l o w
lower  -> l o w e r
widest -> w i d e s t
newest -> n e w e s t
~~~

统计相邻 pair 的频率：

~~~text
l o
o w
w e
e r
w i
i d
d e
e s
s t
n e
e w
~~~

如果最高频 pair 是：

$$
(s,t)
$$

就合并为：

$$
st
$$

于是：

~~~text
widest -> w i d e st
newest -> n e w e st
~~~

下一轮可能发现：

$$
(e,st)
$$

频率最高，于是合并：

$$
est
$$

得到：

~~~text
widest -> w i d est
newest -> n e w est
~~~

再继续：

$$
(o,w)\rightarrow ow
$$

$$
(l,ow)\rightarrow low
$$

于是：

~~~text
low -> low
lower -> low e r
~~~

这种过程会把高频片段逐渐变成独立 token。

最终效果是：

- 常见词可能成为单 token；
- 常见词缀可能成为 token；
- 常见空格模式也可能进入 token；
- 罕见词可以由更小片段组合出来。

---

## 9. BPE 不是“理解语义”，而是“压缩频率模式”

BPE 很容易被误解成一种语言学分词。

它不是。

BPE 不知道什么是词根、词缀、语法、实体。

它只知道：

$$
\operatorname{count}(a,b)
$$

也就是哪个相邻 pair 出现得最多。

所以 BPE 学到的 token 可能和语言学直觉不一致。

比如英文里常见 token 可能包括：

~~~text
 the
ing
ion
ed
~~~

注意很多 token 会带前导空格：

~~~text
" the"
" and"
" of"
~~~

这是因为英文文本中单词边界通常由空格体现。

在 GPT 系列 tokenizer 中，前导空格是非常重要的模式。

例如：

~~~text
"hello"
~~~

和：

~~~text
" hello"
~~~

可能是不同 token。

这说明 tokenizer 学到的不是抽象语言学单位，而是：

> 文本字节序列中的高频压缩单元。

---

## 10. 为什么需要 pre-tokenization

如果直接在整个 corpus 的 byte 序列上跑 BPE，会有一些问题。

最明显的问题是：BPE 可能跨越不应该跨越的边界。

例如两个文档拼在一起：

~~~text
[Doc 1]<|endoftext|>[Doc 2]
~~~

如果不处理特殊边界，BPE 可能学到跨文档的 token。

这没有意义。

还有一个问题是标点和空格。

如果完全不做粗粒度切分，BPE 可能学出很多带标点的变体：

~~~text
dog
dog.
dog!
dog,
~~~

这些 token 在语义上接近，但会被分成完全不同的 token ID。

所以 BPE 训练前通常会做 pre-tokenization。

pre-tokenization 的作用不是最终分词，而是：

> 限制 BPE merge 的边界。

也就是说，BPE 只在 pre-token 内部合并，不跨 pre-token 合并。

例如文本：

~~~text
some text that i'll pre-tokenize
~~~

可能先被粗切成：

~~~text
some
 text
 that
 i
'll
 pre
-
tokenize
~~~

然后每个 pre-token 内部再做 byte-level BPE。

这能减少不合理合并，也让统计更高效。

---

## 11. special tokens 的作用

语言模型训练中经常需要特殊 token。

例如：

~~~text
<|endoftext|>
~~~

它可以表示文档结束。

special token 有两个角色：

1. 编码时，它应该作为整体 token 保留；
2. 训练 BPE merge 时，它应该作为硬边界，阻止跨文档合并。

也就是说，遇到：

~~~text
[Doc 1]<|endoftext|>[Doc 2]
~~~

应该把它视为：

$$
\text{Doc 1}\mid\text{boundary}\mid\text{Doc 2}
$$

BPE 不应该在 Doc 1 末尾和 Doc 2 开头之间合并。

所以 special token 不是普通字符串。

它是一种控制信号：

$$
\text{special token}=\text{structural marker in model input}
$$

如果 special token 处理错，后续训练会出问题。

比如文档边界消失后，模型会看到很多不自然的跨文档上下文。

---

## 12. BPE training 和 BPE encoding 的区别

这里要区分两个过程。

### BPE training

BPE training 是学习词表和 merge 规则：

$$
\text{corpus}\rightarrow(\mathcal{V},\mathcal{M})
$$

其中：

- $\mathcal{V}$ 是 vocab；
- $\mathcal{M}$ 是 merge list。

merge list 是有顺序的：

$$
\mathcal{M}=[(a_1,b_1),(a_2,b_2),\ldots,(a_K,b_K)]
$$

越早出现的 merge 优先级越高。

### BPE encoding

BPE encoding 是使用已经学到的 vocab 和 merges，把新文本转成 token IDs：

$$
\text{text}\rightarrow[x_1,x_2,\ldots,x_T]
$$

编码时不是重新统计频率，而是按训练得到的 merge 顺序应用规则。

例如 merges 是：

$$
(t,h),(th,e)
$$

输入：

~~~text
the
~~~

初始 byte 序列：

$$
[t,h,e]
$$

先合并：

$$
(t,h)\rightarrow th
$$

得到：

$$
[th,e]
$$

再合并：

$$
(th,e)\rightarrow the
$$

得到：

$$
[the]
$$

所以 encoding 是一个 deterministic process：

$$
(\mathcal{V},\mathcal{M},\text{text})\rightarrow\text{token IDs}
$$

---

## 13. tokenizer 的压缩率为什么重要

tokenizer 的一个核心指标是 compression ratio。

通常可以写成：

$$
\text{compression ratio}=\frac{\text{bytes}}{\text{tokens}}
$$

如果一个 tokenizer 的 compression ratio 更高，说明平均每个 token 承载更多 byte 信息。

这通常意味着：

$$
\text{same text}\rightarrow\text{fewer tokens}
$$

这对语言模型非常重要。

因为 Transformer 的训练和推理成本很大程度上与 token 数有关。

尤其是 attention：

$$
\text{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

对于序列长度 $L$，attention matrix 大小是：

$$
L\times L
$$

所以训练时 attention 的复杂度近似为：

$$
O(L^2)
$$

如果 tokenizer 把文本切得更碎，$L$ 变大，计算成本会显著上升。

推理时虽然每步只生成一个 token，但长上下文需要保存 KV cache：

$$
M_{\text{KV}}\propto L
$$

所以 tokenizer 影响的不只是文本表示，也影响系统成本：

$$
\text{tokenizer}\rightarrow\text{sequence length}\rightarrow\text{attention cost}+\text{KV cache cost}
$$

---

## 14. vocab size 的 trade-off

BPE tokenizer 有一个关键超参数：

$$
|\mathcal{V}|
$$

也就是 vocab size。

如果 vocab size 太小：

- token 序列更长；
- 压缩率低；
- 上下文窗口浪费；
- 多语言文本更容易被切碎。

如果 vocab size 太大：

- embedding 参数变多；
- LM head 参数变多；
- softmax 输出维度变大；
- 低频 token 训练不足。

Embedding 参数量约为：

$$
|\mathcal{V}|\cdot d_{\text{model}}
$$

如果 output embedding 不共享权重，LM head 也有：

$$
|\mathcal{V}|\cdot d_{\text{model}}
$$

所以 vocab size 变大，会直接增加参数量：

$$
M_{\text{embed/head}}\propto|\mathcal{V}|d_{\text{model}}
$$

但 vocab size 变小，会增加序列长度：

$$
L\uparrow
$$

进而增加 attention 和 KV cache 成本。

所以 vocab size 的本质 trade-off 是：

$$
\text{embedding/head parameters}\leftrightarrow\text{sequence length and context efficiency}
$$

---

## 15. Tokenization 对不同语言并不公平

一个非常重要但容易忽略的问题是：

> 同一个 tokenizer 对不同语言的压缩率可能差异很大。

例如英文语料上训练的 BPE tokenizer，通常对英文压缩很好。

但对中文、日文、代码、数学符号、罕见 Unicode 字符，可能切得更碎。

假设同样是表达一句话：

~~~text
I like machine learning.
~~~

英文 tokenizer 可能只需要少量 token。

但中文：

~~~text
我喜欢机器学习。
~~~

如果 tokenizer 对中文支持不好，可能每个汉字都被拆成多个 byte token。

这会造成：

$$
\text{same semantic content}\rightarrow\text{more tokens}\rightarrow\text{higher cost}
$$

这不是语言本身更复杂，而是 tokenizer 对它不友好。

所以 tokenizer 也会影响模型的语言公平性和多语言能力。

可以粗略理解为：

$$
\text{token cost}=\frac{\text{tokens}}{\text{semantic content}}
$$

压缩率低的语言或格式，需要为同样的信息付出更多 token 成本。

---

## 16. Assignment 1 从 tokenizer 开始，都做了啥？

CS336 Assignment 1 要实现：

- BPE tokenizer；
- Transformer LM；
- cross-entropy；
- AdamW；
- training loop；
- checkpointing；
- generation。

这个顺序不是随便排的。

因为语言模型训练的第一步不是写 Transformer，而是把原始文本变成 token IDs。

完整链路是：

$$
\text{raw corpus}\rightarrow\text{train tokenizer}\rightarrow\text{encode corpus}\rightarrow\text{token IDs}\rightarrow\text{data loader}\rightarrow\text{Transformer LM}\rightarrow\text{loss}\rightarrow\text{optimizer}
$$

如果 tokenizer 错了，后面所有东西都建立在错误输入上。

例如：

- encode/decode 不一致；
- special token 被拆碎；
- Unicode 解码错误；
- BPE merge 顺序错误；
- 不该跨文档合并；
- 大文件 tokenization 内存爆炸。

这些问题可能不会立刻报错，但会污染训练数据。

这类 bug 比模型代码 bug 更隐蔽。

因为模型仍然能训练，只是训练在错误分布上。

---

## 17. 实现 BPE 时真正难的地方

BPE 的算法思想很简单，但工程实现并不轻松。

最朴素的实现是：

1. 每轮扫描整个 corpus；
2. 统计所有 pair；
3. 找最高频 pair；
4. 全量替换；
5. 重复几万次。

这个复杂度很高。

假设要做 $K$ 次 merge，每次都全量扫描 corpus，那么训练会非常慢：

$$
O(KN)
$$

其中 $N$ 是 corpus 的 token 长度。

所以高效实现需要考虑：

- 预先统计 pre-token frequency；
- 用 `dict[tuple[bytes, ...], int]` 表示 pre-token；
- 每次 merge 后增量更新 pair counts；
- 避免反复扫描全部文本；
- pre-tokenization 可以 multiprocessing；
- 大文件读取要避免一次性全部塞入内存；
- special token 要先切开，不能参与 merge 统计。

这说明 tokenizer 不是“简单字符串处理”。

它本质上是一个小型数据系统问题：

$$
\text{BPE training}=\text{text processing}+\text{frequency counting}+\text{incremental updates}+\text{memory management}
$$

---

## 18. encode_iterable 为什么重要

Assignment 里要求 tokenizer 支持：

~~~python
encode_iterable(iterable)
~~~

这个接口的意义是处理大文件。

如果训练语料有几十 GB 或几百 GB，不可能直接：

~~~python
text = open(path).read()
~~~

然后一次性 tokenize。

正确做法是流式读取：

$$
\text{file stream}\rightarrow\text{chunks}\rightarrow\text{token IDs}
$$

但这里有一个微妙问题：

> chunk 边界不能改变 tokenization 结果。

如果把文本任意切块，可能一个 token 被切断。

例如：

~~~text
unbelievable
~~~

如果 chunk 变成：

~~~text
unbel
ievable
~~~

tokenization 可能和整段文本一次性 tokenize 不同。

所以流式 tokenization 需要在安全边界切分，例如特殊 token、换行、文档边界等。

这体现了语言模型系统里的一个常见原则：

$$
\text{correctness}+\text{memory efficiency}+\text{boundary handling}
$$

三者必须同时成立。

---

## 19. tokenizer 和模型训练目标的连接

tokenizer 输出 token IDs：

$$
x_1,x_2,\ldots,x_T
$$

Transformer LM 输入一段上下文：

$$
x_{1:t}
$$

输出 logits：

$$
o_t\in\mathbb{R}^{|\mathcal{V}|}
$$

softmax 后得到：

$$
p(x_{t+1}\mid x_{\leq t})=\operatorname{softmax}(o_t)
$$

训练损失是 cross-entropy：

$$
\mathcal{L}_t=-\log p(x_{t+1}\mid x_{\leq t})
$$

对于整段序列：

$$
\mathcal{L}=-\frac{1}{T}\sum_{t=1}^{T}\log p(x_{t+1}\mid x_{\leq t})
$$

所以 tokenizer 直接决定了模型预测的基本单位。

如果 tokenizer 把一个单词切成多个 token，那么模型需要一步一步预测这些片段。

例如：

~~~text
tokenization
~~~

被切成：

~~~text
token + ization
~~~

模型预测的是：

$$
p(\text{token})\cdot p(\text{ization}\mid\text{token})
$$

而不是一次预测整个单词。

这会改变学习难度。

tokenizer 越细，局部预测更容易，但序列更长；

tokenizer 越粗，序列更短，但每个 token 的类别空间更大。

---

## 20. 为什么 tokenizer 会影响 perplexity 的可比性

语言模型常用 perplexity 评价：

$$
\text{PPL}=\exp(\mathcal{L})
$$

其中 $\mathcal{L}$ 是平均 cross-entropy。

但注意：

> perplexity 是按 token 计算的，不是天然按字符或 byte 计算的。

如果两个模型 tokenizer 不同，它们的 token 粒度不同，那么 perplexity 不能直接公平比较。

例如 tokenizer A 把文本切成 100 个 token，tokenizer B 切成 150 个 token。

即使它们建模的是同一段文本，平均每 token loss 的含义也不同。

更合理的比较方式可能是：

$$
\text{bits per byte}
$$

或者：

$$
\text{negative log likelihood per byte}
$$

这就是为什么 tokenizer 不只是预处理，而会影响评估指标解释。

可以总结为：

> 不同 tokenizer 下的 token-level PPL 不一定可直接比较。

---

## 21. 从 Lecture 1 看整个 CS336 的学习路线

Lecture 1 把语言模型 pipeline 的入口讲清楚。

后面的课程会沿着这条链路继续展开：

| 阶段                  | 关键问题                           |
| --------------------- | ---------------------------------- |
| Tokenization          | 文本如何变成 token IDs             |
| Resource Accounting   | 训练一个模型需要多少显存和 FLOPs   |
| Architecture          | Transformer 为什么这样设计         |
| Attention / MoE       | 模型结构如何扩展                   |
| GPU / Kernel / Triton | 如何让算子跑得快                   |
| Parallelism           | 多 GPU 如何训练                    |
| Scaling Laws          | 模型、数据、计算量如何配比         |
| Inference             | 如何高效生成                       |
| Data                  | 预训练数据如何收集、清洗、去重     |
| Alignment             | 如何让模型更符合人类偏好和任务需求 |

所以 Lecture 1 不是单纯讲 tokenizer，而是在告诉你：

> LLM 不是一个模型文件，而是一套端到端系统。

---

## 22. 我对Assignment 1的一点想法

如果现在要做 Assignment 1，不应该一上来就写 Transformer。

更合理的顺序是：

### 阶段 1：把 tokenizer 原理吃透

你需要明确：

$$
\text{Unicode}\rightarrow\text{UTF-8 bytes}\rightarrow\text{pre-token}\rightarrow\text{BPE merges}\rightarrow\text{token IDs}
$$

尤其要理解：

- byte 和 character 不一样；
- 一个 Unicode 字符可能对应多个 UTF-8 bytes；
- special token 是硬边界；
- BPE merge 不能跨 pre-token；
- merge list 的顺序决定 encoding；
- decode 时要从 token IDs 还原 bytes，再 decode 成字符串。

### 阶段 2：先写慢但正确的 BPE

不要一开始追求极致优化。

先实现一个朴素版本，能通过小样例。

你要验证：

$$
\text{train}\rightarrow\text{vocab, merges}\rightarrow\text{encode}\rightarrow\text{decode}
$$

是否闭环。

### 阶段 3：再优化 pre-tokenization 和 merge

当正确性稳定后，再做：

- multiprocessing pre-tokenization；
- pair count cache；
- merge 增量更新；
- profiling；
- 减少重复构造 tuple；
- 避免无意义全量扫描。

### 阶段 4：用 TinyStories 做端到端验证

TinyStories 的好处是文本简单，模型较小，生成结果容易肉眼判断。

如果 tokenizer、data loader、loss、training loop 都没大问题，小模型应该能生成类似儿童故事的文本。

这比只看 unit test 更能暴露系统问题。

---

## 23. 对 LLM systems 的更深启发

Lecture 1 讲 tokenization，但它实际上已经埋下了系统优化的主线。

很多后续系统问题都和 token 有关。

### 训练成本

训练 token 数是核心单位：

$$
\text{training compute}\propto\text{number of tokens}
$$

如果 tokenizer 压缩率差，同样语料会产生更多 token，训练成本直接变高。

### 上下文窗口

模型 context length 是按 token 计数，不是按字符或单词计数：

$$
\text{context window}=L\text{ tokens}
$$

tokenizer 越碎，同样窗口能容纳的信息越少。

### 推理延迟

自回归生成一次生成一个 token：

$$
x_1\rightarrow x_2\rightarrow\cdots\rightarrow x_T
$$

如果同样回答需要更多 token，推理步数也更多。

### KV cache

KV cache 大小与序列长度线性相关：

$$
M_{\text{KV}}\propto L
$$

tokenizer 影响 $L$，也就影响推理显存。

所以 tokenizer 的影响链条是：

$$
\text{tokenizer}\rightarrow\text{token count}\rightarrow\text{training compute}+\text{context efficiency}+\text{generation latency}+\text{KV cache memory}
$$

---

## 24. 对中文、多模态和代码模型的启发

如果研究目标涉及中文、多模态、代码或数学推理，tokenizer 问题不能忽略。

### 中文

中文没有天然空格边界。

英文 tokenizer 常常学到带空格的 subword：

~~~text
" the"
" and"
"ing"
~~~

但中文文本中没有这种空格结构。

如果 tokenizer 主要在英文语料训练，中文可能被切得很碎。

这会导致：

$$
\text{Chinese text}\rightarrow\text{more tokens}\rightarrow\text{higher cost}\rightarrow\text{shorter effective context}
$$

### 代码

代码里有很多符号模式：

~~~python
def foo(x):
    return x + 1
~~~

一个好的代码 tokenizer 需要有效处理：

- indentation；
- brackets；
- operators；
- camelCase；
- snake_case；
- rare identifiers；
- long strings。

如果 tokenizer 对代码不友好，代码模型的上下文效率会下降。

### 多模态

多模态模型里，图像也会被转成 token-like representations。

虽然视觉 token 不一定来自 BPE，但系统问题类似：

$$
\text{raw modality}\rightarrow\text{discrete or continuous tokens}\rightarrow\text{Transformer context}
$$

所以 token budget 会变成统一资源。

文本 token 多了，能放的图像 token 就少；

图像 token 多了，能放的文本上下文就少。

这对 visual reasoning、visual anchor、latent reasoning 都很关键。

---

## 25. 这节课最应该记住的判断框架

Lecture 1 不是让你背 BPE 细节，而是建立一种系统性判断：

| 问题                     | 低水平理解       | 高水平理解                                     |
| ------------------------ | ---------------- | ---------------------------------------------- |
| tokenizer 是什么         | 把文本切成 token | 定义模型输入空间                               |
| BPE 是什么               | 高频 pair 合并   | 用统计压缩文本序列                             |
| byte-level 有什么用      | 可以处理字符     | 避免 OOV，统一任意文本                         |
| vocab size 怎么选        | 越大越好         | 参数量和序列长度 trade-off                     |
| special token 是什么     | 特殊字符串       | 训练和推理中的结构控制符                       |
| compression ratio 是什么 | 压缩指标         | 影响训练成本、上下文效率、推理延迟             |
| token-level perplexity   | 模型指标         | 受 tokenizer 粒度影响，跨 tokenizer 不一定可比 |

可以用一句话概括：

> tokenization 是语言模型系统的输入层设计，不是简单的数据清洗步骤。

---

## 26. 总结

第 1 讲的核心是把语言模型从“神经网络模型”重新放回“完整系统”里理解。

它讲的是：

$$
\text{raw text}\rightarrow\text{tokens}\rightarrow\text{language model}
$$

但真正要理解的是：

> tokenizer 决定了模型如何看见世界。

BPE 的核心逻辑是：

$$
\text{byte vocabulary}\rightarrow\text{frequent pair merges}\rightarrow\text{subword vocabulary}
$$

它解决了三个问题：

1. 避免 OOV；
2. 压缩 byte 序列；
3. 在词表大小和序列长度之间取得折中。

但它也引入了新的系统 trade-off：

$$
\text{vocab size}\leftrightarrow\text{embedding/head parameters}\leftrightarrow\text{sequence length}\leftrightarrow\text{attention and KV cache cost}
$$

> 