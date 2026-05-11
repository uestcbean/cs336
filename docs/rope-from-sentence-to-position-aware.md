# RoPE 入门：从一句话到位置感知的 Attention

> 这份文档只做一件事：把 RoPE（Rotary Position Embedding）讲清楚。
> 我们会从一个句子出发，先建立直觉，再看公式，最后用代码把完整流程串起来。

---

## 目录

1. [先给结论：RoPE 在做什么](#0-先给结论rope-在做什么)
2. [为什么 Transformer 需要位置信息](#1-为什么-transformer-需要位置信息)
3. [先理解二维旋转](#2-先理解二维旋转)
4. [RoPE 的核心想法](#3-rope-的核心想法)
5. [从一个向量到一整段序列](#4-从一个向量到一整段序列)
6. [为什么 RoPE 表达的是相对位置](#5-为什么-rope-表达的是相对位置)
7. [为什么要有很多频率](#6-为什么要有很多频率)
8. [完整代码：把 RoPE 接进 Attention](#7-完整代码把-rope-接进-attention)
9. [常见疑问](#8-常见疑问)
10. [总结](#9-总结)

---

## 0. 先给结论：RoPE 在做什么

一句话：

> **RoPE 把 token 的位置变成一个旋转角度，然后用这个角度去旋转 Q 和 K。**

在 Transformer 里，一句话会经历这样的流程：

```text
"The cat sat on the mat"
        |
        v
token ids
        |
        v
embedding X
        |
        v
Q, K, V
        |
        v
只旋转 Q 和 K
        |
        v
attention(Q_rope, K_rope, V)
```

RoPE 不是改 token id，也不是直接改 embedding，而是在 attention 前改 `Q` 和 `K`。

为什么只改 `Q` 和 `K`？

- `Q` 和 `K` 决定“这个位置应该看哪里”。
- `V` 是被读取的内容，不负责寻址。
- 位置应该影响“看哪里”，而不是把内容本身扭来扭去。

先记住这个直觉：**RoPE 给 attention 的寻址过程加上位置感。**

---

## 1. 为什么 Transformer 需要位置信息

我们用一句话作为贯穿例子：

```text
The cat sat on the mat
```

经过 tokenizer 后，它变成一串 token id：

```python
token_ids = [464, 3797, 3332, 319, 262, 2603]
```

这串数字本身有顺序：

```text
位置 0: The
位置 1: cat
位置 2: sat
位置 3: on
位置 4: the
位置 5: mat
```

接着模型用 embedding matrix 查表，把每个 token id 变成一个向量：

```text
X shape = (seq_len, d_model)

X[0] = "The" 的向量
X[1] = "cat" 的向量
X[2] = "sat" 的向量
...
```

到这里，顺序还保存在矩阵的“第几行”里。但 attention 的核心计算是：

```text
scores = Q @ K.T
```

它只看向量之间的相似度。假如我们把整句话的行顺序一起打乱，attention 得到的分数矩阵也只是跟着重排，模型并不会天然知道“谁在前、谁在后”。

一个最小演示：

```python
import torch

torch.manual_seed(0)

seq_len = 4
d_model = 8

X = torch.randn(seq_len, d_model)
W_q = torch.randn(d_model, d_model)
W_k = torch.randn(d_model, d_model)

Q = X @ W_q
K = X @ W_k
scores = Q @ K.T

perm = torch.tensor([3, 1, 2, 0])
X2 = X[perm]
Q2 = X2 @ W_q
K2 = X2 @ W_k
scores2 = Q2 @ K2.T

print(scores[3, 1])
print(scores2[0, 1])  # 同一个 token 对，只是位置被重排了
```

这说明一个问题：

> **attention 本身很擅长比较内容，但不擅长记住位置。**

所以我们需要一种方式，把“第几个 token”写进 `Q` 和 `K` 的向量内容里。RoPE 就是其中一种非常优雅的方式。

---

## 2. 先理解二维旋转

RoPE 的数学核心只有一个动作：**二维旋转**。

如果有一个二维向量：

```text
(x, y)
```

把它逆时针旋转角度 `theta` 后，会得到：

```text
x' = x * cos(theta) - y * sin(theta)
y' = x * sin(theta) + y * cos(theta)
```

写成矩阵就是：

```text
| x' |   | cos(theta)  -sin(theta) | | x |
| y' | = | sin(theta)   cos(theta) | | y |
```

可以把 `cos` 和 `sin` 理解成单位圆上的横坐标和纵坐标：

```text
              y
              ^
              |
              |     (cos(theta), sin(theta))
              |    /
              |   /
              |  / theta
--------------+----------------> x
```

代码验证一下：

```python
import math

def rotate_2d(x, y, theta):
    return (
        x * math.cos(theta) - y * math.sin(theta),
        x * math.sin(theta) + y * math.cos(theta),
    )

print(rotate_2d(1, 0, math.pi / 2))  # 约等于 (0, 1)
print(rotate_2d(1, 0, math.pi))      # 约等于 (-1, 0)
print(rotate_2d(1, 0, 2 * math.pi))  # 约等于 (1, 0)
```

这里有两个重要性质。

第一，旋转不会改变向量长度：

```text
旋转前长度 = sqrt(x^2 + y^2)
旋转后长度 = sqrt(x'^2 + y'^2)
两者相同
```

也就是说，RoPE 不会粗暴放大或缩小 `Q`、`K`，它主要改变方向。

第二，连续旋转可以合并：

```text
先转 a，再转 b = 一次性转 a + b
```

这个性质后面会变成 RoPE 的关键：**两个位置的旋转角度相减，就会留下相对距离。**

---

## 3. RoPE 的核心想法

现在把二维旋转放进高维向量里。

假设一个 `Q` 向量有 8 维：

```text
q = [q0, q1, q2, q3, q4, q5, q6, q7]
```

RoPE 会把它两两分组：

```text
[q0, q1] | [q2, q3] | [q4, q5] | [q6, q7]
  pair 0    pair 1    pair 2    pair 3
```

每一组都当成一个二维向量，然后根据当前位置旋转。

如果当前位置是 `m`，第 `i` 组的旋转角度是：

```text
angle(m, i) = m * inv_freq[i]
```

其中：

```text
inv_freq[i] = 1 / base^(2i / d)
base 通常是 10000
d 是向量维度
```

所以第 `i` 组会这样更新：

```text
even' = even * cos(angle) - odd * sin(angle)
odd'  = even * sin(angle) + odd * cos(angle)
```

这就是 RoPE 的全部主体。

一个只处理单个向量的版本：

```python
import torch

def apply_rope_to_vector(x, position, base=10000.0):
    """
    x: shape (d,)
    position: 当前 token 的位置，例如 0, 1, 2, ...
    """
    d = x.shape[-1]
    assert d % 2 == 0, "RoPE 需要偶数维度，方便两两配对"

    half = d // 2
    inv_freq = 1.0 / (base ** (torch.arange(half) * 2 / d))
    angles = position * inv_freq

    cos = angles.cos()
    sin = angles.sin()

    x_even = x[0::2]
    x_odd = x[1::2]

    out = torch.empty_like(x)
    out[0::2] = x_even * cos - x_odd * sin
    out[1::2] = x_even * sin + x_odd * cos
    return out


x = torch.tensor([1.0, 2.0, 3.0, 4.0])

print(apply_rope_to_vector(x, position=0))  # 位置 0，不旋转
print(apply_rope_to_vector(x, position=1))  # 位置 1，开始旋转
print(apply_rope_to_vector(x, position=5))  # 位置 5，旋转更多
```

注意 `position=0` 时，所有角度都是 0：

```text
cos(0) = 1
sin(0) = 0
```

所以输出等于原向量。这符合直觉：第 0 个位置不需要额外转动。

---

## 4. 从一个向量到一整段序列

真实模型里不是只有一个向量，而是一整段序列：

```text
Q shape = (seq_len, d_model)
K shape = (seq_len, d_model)
```

比如 6 个 token，每个向量 64 维：

```text
Q[0] 用 position=0 旋转
Q[1] 用 position=1 旋转
Q[2] 用 position=2 旋转
...

K[0] 用 position=0 旋转
K[1] 用 position=1 旋转
K[2] 用 position=2 旋转
...
```

矩阵版本如下：

```python
import torch

def apply_rope(x, positions, base=10000.0):
    """
    x: shape (seq_len, d)
    positions: shape (seq_len,)
    """
    seq_len, d = x.shape
    assert d % 2 == 0

    half = d // 2
    inv_freq = 1.0 / (base ** (torch.arange(half, device=x.device) * 2 / d))
    angles = positions[:, None].to(x.dtype) * inv_freq[None, :]

    cos = angles.cos()
    sin = angles.sin()

    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]

    out = torch.empty_like(x)
    out[:, 0::2] = x_even * cos - x_odd * sin
    out[:, 1::2] = x_even * sin + x_odd * cos
    return out
```

接入 attention 时，只做三步：

```python
positions = torch.arange(seq_len)

Q_rope = apply_rope(Q, positions)
K_rope = apply_rope(K, positions)

scores = Q_rope @ K_rope.T
```

`V` 不需要 RoPE：

```python
output = F.softmax(scores, dim=-1) @ V
```

到这里已经可以理解 RoPE 的工程流程了：

> **生成 Q/K/V，旋转 Q/K，用旋转后的 Q/K 算 attention，最后仍然聚合原来的 V。**

---

## 5. 为什么 RoPE 表达的是相对位置

这一节先讲直觉，再给公式。

假设某个 `Q` 在位置 `m`，某个 `K` 在位置 `n`。

RoPE 会做：

```text
Q 按角度 m * freq 旋转
K 按角度 n * freq 旋转
```

attention score 要看它们的内积，也就是“旋转后的 Q 和旋转后的 K 有多对齐”。

两个向量都旋转后，它们之间真正重要的是两个角度的差：

```text
角度差 = m * freq - n * freq
       = (m - n) * freq
```

也就是相对距离 `m - n`。

这就是 RoPE 最重要的性质：

> **RoPE 让 attention score 天然看到相对位置，而不是只看到绝对位置。**

用二维旋转写出来就是：

```text
dot(R_m q, R_n k)
= dot(q, R_{n-m} k)
```

这里的 `R_m` 表示“按位置 m 对应的角度旋转”。公式右边写成了 `n - m`，是因为我们把 `R_m` 从左边移到了右边；如果站在 `Q` 的角度看，也可以说是 `m - n`。符号方向不重要，重点是：**它只依赖两个位置的差，而不依赖 m 和 n 各自的绝对值。**

用代码验证：

```python
import torch

def apply_rope_to_vector(x, position, base=10000.0):
    d = x.shape[-1]
    half = d // 2
    inv_freq = 1.0 / (base ** (torch.arange(half) * 2 / d))
    angles = position * inv_freq
    cos = angles.cos()
    sin = angles.sin()

    x_even = x[0::2]
    x_odd = x[1::2]

    out = torch.empty_like(x)
    out[0::2] = x_even * cos - x_odd * sin
    out[1::2] = x_even * sin + x_odd * cos
    return out


torch.manual_seed(42)
q = torch.randn(8)
k = torch.randn(8)

# 三组位置不同，但相对距离都是 3
a = (apply_rope_to_vector(q, 5) * apply_rope_to_vector(k, 2)).sum()
b = (apply_rope_to_vector(q, 10) * apply_rope_to_vector(k, 7)).sum()
c = (apply_rope_to_vector(q, 100) * apply_rope_to_vector(k, 97)).sum()

# 相对距离变成 1
d = (apply_rope_to_vector(q, 5) * apply_rope_to_vector(k, 4)).sum()

print(a)
print(b)
print(c)
print(d)
```

前 3 个值应该非常接近，因为它们的相对距离都是 3。第 4 个值通常不同，因为相对距离变了。

这对语言模型很自然。语言里的很多关系不是“第 47 个 token 和第 52 个 token”，而是：

- 前一个词
- 后两个词
- 三步前的主语
- 当前 token 到上一个换行符的距离

RoPE 正好把这种“相对距离”放进了 attention score。

---

## 6. 为什么要有很多频率

如果只用一个频率，会有一个问题：旋转会绕圈。

比如频率是 `1.0`，位置每增加 1，角度就增加 1 弧度。大约每 `2π ≈ 6.28` 个 token 就转完一圈。

这对短距离很敏感，但长距离会混淆：

```text
位置 0 和位置 1：角度差明显
位置 0 和位置 1000：已经绕了很多圈，不容易判断真实距离
```

所以 RoPE 不只用一个频率，而是给不同维度对分配不同频率：

```text
pair 0: 高频，适合分辨近距离
pair 1: 稍低频
pair 2: 更低频
...
最后的 pair: 低频，适合分辨远距离
```

可以用“钟表”类比：

```text
秒针：变化快，适合看短时间差
分针：变化慢，适合看中等时间差
时针：更慢，适合看更长时间差
```

RoPE 的多频率也是这个思想：不同维度负责不同尺度的位置。

看一个小表：

```python
import math

base = 10000.0
d = 64

for i in [0, 1, 5, 10, 20, 31]:
    inv_freq = 1.0 / (base ** (2 * i / d))
    wavelength = 2 * math.pi / inv_freq
    print(f"pair {i:>2}: inv_freq={inv_freq:.6f}, wavelength={wavelength:.1f}")
```

你会看到：

```text
前面的 pair 频率高，波长短
后面的 pair 频率低，波长长
```

波长可以理解成“转完一圈需要走多少个 token”：

```text
wavelength = 2π / inv_freq
```

这就是 RoPE 能兼顾短距离和长距离的原因。

---

## 7. 完整代码：把 RoPE 接进 Attention

下面是一个完整、可运行的最小版本。

```python
import math
import torch
import torch.nn.functional as F


def apply_rope(x, positions, base=10000.0):
    """
    对序列矩阵应用 RoPE。

    x: shape (seq_len, d)
    positions: shape (seq_len,)
    """
    seq_len, d = x.shape
    assert d % 2 == 0

    half = d // 2
    inv_freq = 1.0 / (base ** (torch.arange(half, device=x.device) * 2 / d))
    angles = positions[:, None].to(x.dtype) * inv_freq[None, :]

    cos = angles.cos()
    sin = angles.sin()

    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]

    out = torch.empty_like(x)
    out[:, 0::2] = x_even * cos - x_odd * sin
    out[:, 1::2] = x_even * sin + x_odd * cos
    return out


def attention_with_rope(X, W_q, W_k, W_v):
    """
    X: shape (seq_len, d_model)
    W_q/W_k/W_v: shape (d_model, d_model)
    """
    seq_len, d_model = X.shape

    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    positions = torch.arange(seq_len, device=X.device)
    Q = apply_rope(Q, positions)
    K = apply_rope(K, positions)

    scores = Q @ K.T / math.sqrt(d_model)
    attn = F.softmax(scores, dim=-1)
    output = attn @ V
    return output, attn


torch.manual_seed(0)

vocab_size = 50000
d_model = 64
token_ids = torch.tensor([464, 3797, 3332, 319, 262, 2603])

E = torch.randn(vocab_size, d_model)
X = E[token_ids]

W_q = torch.randn(d_model, d_model) * 0.1
W_k = torch.randn(d_model, d_model) * 0.1
W_v = torch.randn(d_model, d_model) * 0.1

output, attn = attention_with_rope(X, W_q, W_k, W_v)

print("output shape:", output.shape)
print("attention shape:", attn.shape)
print(attn.detach().round(decimals=3))
```

这段代码对应的流程是：

```text
token ids
  -> embedding X
  -> Q, K, V
  -> RoPE(Q), RoPE(K)
  -> softmax(Q_rope @ K_rope.T)
  -> attention weights @ V
```

---

## 8. 常见疑问

### 8.1 RoPE 是不是一种 positional embedding？

是，但它和传统“加一个位置向量”的方式不同。

传统做法通常是：

```text
X = token_embedding + position_embedding
```

RoPE 的做法是：

```text
Q_rope = rotate(Q, position)
K_rope = rotate(K, position)
```

也就是说，RoPE 不是把位置向量加进去，而是用位置去旋转 `Q` 和 `K`。

### 8.2 为什么不对 V 做 RoPE？

因为 `V` 是内容。

attention 可以粗略理解成：

```text
用 Q/K 算“去哪儿取”
用 V 表示“取到什么内容”
```

位置应该影响“去哪儿取”，所以进入 `Q/K`。如果旋转 `V`，等于把被读取的内容也按位置变形了，通常不是我们想要的。

### 8.3 RoPE 会不会丢失原来的语义？

旋转不会改变每个二维 pair 的长度，只改变方向。因此它不是随便破坏向量，而是用一种结构化方式把位置混进方向里。

当然，旋转后向量数值会变，但模型训练时会适应这种表示。

### 8.4 为什么公式里叫 `inv_freq`？

很多实现里会写：

```python
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
```

名字叫 `inv_freq` 是历史习惯。直观上它就是每个 pair 的角速度：位置每增加 1，角度增加多少。

### 8.5 RoPE 和绝对位置编码最大的区别是什么？

绝对位置编码直接告诉模型“这是第 m 个 token”。

RoPE 更巧妙：它让 `Q_m` 和 `K_n` 做内积时自然出现 `m - n`，所以 attention 更容易看到相对距离。

---

## 9. 总结

把 RoPE 压缩成 5 句话：

1. attention 只看内容相似度，本身不可靠地知道 token 顺序。
2. RoPE 把位置 `m` 变成旋转角度 `m * inv_freq`。
3. 高维向量被两两分组，每组按自己的频率做二维旋转。
4. RoPE 只作用在 `Q` 和 `K` 上，因为它影响 attention 的寻址。
5. `Q_m` 和 `K_n` 旋转后再做内积，会自然依赖 `m - n`，所以 RoPE 擅长表达相对位置。

最核心的一句话：

> **RoPE 把“位置”变成“旋转角度”，再让 attention 通过角度差看见 token 之间的相对距离。**

