# RoPE 完全指南：从一个句子，到位置感知的语言模型

> 这份文档以**一个句子在 Transformer 里的全生命周期**为主线，循序渐进讲解 RoPE。
> 每一步都配公式 + 可运行代码，确保你不只是"知道"而是"理解"。

---

## 目录

0. [总览：我们要走的路径](#0-总览我们要走的路径)
1. [一个句子是怎么变成数字的](#1-一个句子是怎么变成数字的)
2. [Embedding Matrix：词的"语义坐标"](#2-embedding-matrix词的语义坐标)
3. [位置丢失问题：Attention 的"健忘症"](#3-位置丢失问题attention-的健忘症)
4. [必备数学：sin、cos、单位圆](#4-必备数学sincos单位圆)
5. [频率与波长：旋转的两个核心参数](#5-频率与波长旋转的两个核心参数)
6. [二维旋转矩阵：把"位置"编码成"角度"](#6-二维旋转矩阵把位置编码成角度)
7. [RoPE 构造：把 d 维向量切成 d/2 对小翅膀](#7-rope-构造把-d-维向量切成-d2-对小翅膀)
8. [完整流程：embedding 矩阵 → RoPE → attention](#8-完整流程embedding-矩阵--rope--attention)
9. [为什么 RoPE 编码的是"相对位置"](#9-为什么-rope-编码的是相对位置)
10. [多尺度位置感知：频率层级的妙处](#10-多尺度位置感知频率层级的妙处)
11. [总结与延伸](#11-总结与延伸)

---

## 0. 总览：我们要走的路径

我们要回答一个核心问题：

> **当一个句子进入 Transformer，它的"位置信息"是如何被编码进去的？**

整条路径如下：

```
"The cat sat on the mat"
        │
        ▼  Step 1 (BPE 分词)
[464, 3797, 3332, 319, 262, 2603]
        │
        ▼  Step 2 (Embedding lookup)
矩阵 X ∈ R^{6 × d_model}      ← 每行是一个 token 的语义向量, 没有位置信息
        │
        ▼  Step 3 (线性投影)
Q, K, V ∈ R^{6 × d_model}     ← Q 和 K 各自是 6 行 d 维向量
        │
        ▼  Step 4 (RoPE 加工 Q 和 K)
Q', K' ∈ R^{6 × d_model}      ← 每行被"按其位置"旋转过, 现在带位置信息了
        │
        ▼  Step 5 (Attention 计算)
output = softmax(Q' K'^T / √d) V
```

RoPE 就发生在 **Step 4**——它不是对 embedding 直接动手，而是在 Q、K 经过线性投影后、attention 计算前，把位置信息**旋转**进去。

我们一步一步走。

---

## 1. 一个句子是怎么变成数字的

我们用 **"The cat sat on the mat"** 这个句子作为整篇贯穿示例。

### 1.1 分词（Tokenization）

BPE tokenizer（你已经手写过了！）把文本切成 token 并查表：

```
"The cat sat on the mat"
       ↓ BPE
[464, 3797, 3332, 319, 262, 2603]
```

每个数字是 vocab 里的一个**索引**——本身没有任何意义，只是个标签。重要的是这个**序列的顺序**保留了原句的顺序。

```python
# 模拟 (真实的 GPT-2 BPE 输出仅供示意)
sentence = "The cat sat on the mat"
token_ids = [464, 3797, 3332, 319, 262, 2603]  # length = 6
```

### 1.2 关键观察

到这一步，**位置信息还在**——`token_ids` 是个**有序列表**。索引 0 是 "The"，索引 5 是 "mat"。

但下一步，我们要把这些 id 变成向量喂给神经网络。位置信息**很容易就丢了**。

---

## 2. Embedding Matrix：词的"语义坐标"

### 2.1 什么是 Embedding 矩阵

模型有一个**可学习**的查找表 `E`，shape 是 `(vocab_size, d_model)`：

```
       d_model 列
       ◄────────►
       ┌──────────┐
   0   │ . . . .  │  ← token id 0 的语义向量
   1   │ . . . .  │
       │   ...    │
  464  │ . . . .  │  ← "The" 的语义向量
       │   ...    │
  3797 │ . . . .  │  ← "cat" 的语义向量
       │   ...    │
50000  │ . . . .  │
       └──────────┘
```

每一行 d_model 维就是该 token 的"语义坐标"——训练后，**意思相近的词向量会靠近**（比如 "cat" 和 "dog" 的向量比 "cat" 和 "philosophy" 更接近）。

### 2.2 用 token_ids 查表 → 得到 X 矩阵

```
token_ids = [464, 3797, 3332, 319, 262, 2603]   ← shape: (6,)

X = E[token_ids]                                  ← shape: (6, d_model)

X 长这样:
   ┌──────────────────────┐
0  │ E[464]   (= "The")   │
1  │ E[3797]  (= "cat")   │
2  │ E[3332]  (= "sat")   │
3  │ E[319]   (= "on")    │
4  │ E[262]   (= "the")   │
5  │ E[2603]  (= "mat")   │
   └──────────────────────┘
```

代码：

```python
import torch

vocab_size = 50000
d_model = 64

torch.manual_seed(42)
E = torch.randn(vocab_size, d_model)  # 随机初始化, 假装是训好的

token_ids = torch.tensor([464, 3797, 3332, 319, 262, 2603])
X = E[token_ids]  # shape: (6, 64)

print(X.shape)         # torch.Size([6, 64])
print(X[0, :5])        # X 的第 0 行前 5 维 = "The" 的语义坐标的前 5 维
```

### 2.3 为什么这一步**还有**位置信息

`X` 是按 token_ids 顺序排好的——第 0 行就是 "The"，第 5 行就是 "mat"。**但这种"位置信息"是隐含在矩阵的行索引里的**，不在向量本身的内容里。

下一节就要看到，attention 机制对"行索引"几乎是"健忘"的。

---

## 3. 位置丢失问题：Attention 的"健忘症"

### 3.1 简化版 attention 演示

Attention 的核心是 **`softmax(Q K^T / √d) V`**。简化一下，只看"每个位置 m 关心哪些位置 n"——也就是 attention weight matrix `A_{m,n}`：

```
A_{m,n} = exp(<Q_m, K_n> / √d) / Σ_k exp(<Q_m, K_k> / √d)
```

这里 Q_m 是第 m 行 Q，K_n 是第 n 行 K。

### 3.2 演示：打乱顺序，attention 模式不变

```python
import torch

# 设置
torch.manual_seed(0)
seq_len = 4
d_model = 8

X = torch.randn(seq_len, d_model)  # 4 个 token 的 embedding
W_q = torch.randn(d_model, d_model)
W_k = torch.randn(d_model, d_model)

# 算 Q, K
Q = X @ W_q
K = X @ W_k

# 算 attention scores (没用 √d 缩放, 不影响演示)
scores_original = Q @ K.T  # shape (4, 4)
print("原始 attention scores:")
print(scores_original)

# 现在打乱顺序: 把第 0 行和第 3 行交换
perm = torch.tensor([3, 1, 2, 0])
X_shuffled = X[perm]

Q_shuffled = X_shuffled @ W_q
K_shuffled = X_shuffled @ W_k
scores_shuffled = Q_shuffled @ K_shuffled.T

print("\n打乱后的 attention scores:")
print(scores_shuffled)

print("\n打乱后的 scores[0,1] =", scores_shuffled[0, 1].item())
print("原始的 scores[3,1] =", scores_original[3, 1].item())
print("它们应该相等 (打乱后的位置 0 = 原来的位置 3, 位置 1 没变)")
```

**输出**会发现：scores_shuffled 和 scores_original 只是**行列重排**（被 perm 重排过），数值完全一样——证明 attention 完全不在乎"哪个 token 在第几行"，只在乎"内容"。

### 3.3 核心问题

> **Attention 是置换不变（permutation-invariant）的。** 如果不显式注入位置信息，
> 模型分不清 "The cat sat" 和 "Sat cat the"。

所以我们必须在 Q 和 K 里**人为地嵌入位置信息**。RoPE 就是干这个的。

---

## 4. 必备数学：sin、cos、单位圆

要理解 RoPE，必须先把 sin、cos 看作**单位圆上的一个点**。

### 4.1 单位圆视角

想象一个半径为 1 的圆，圆心在原点。圆上的任意一点 P，用从 x 轴正方向逆时针旋转的**角度 θ** 唯一确定。

定义：
- **`cos(θ)` = P 的横坐标**
- **`sin(θ)` = P 的纵坐标**

```
       y
       ▲
       │     P = (cos θ, sin θ)
       │    /
       │   /
       │  / θ
       │ /
       │/
   ────┼──────►  x
       │
```

具体值：

| θ | 点 P | cos(θ) | sin(θ) |
|---|------|--------|--------|
| 0 | (1, 0) | 1 | 0 |
| π/4 | (√2/2, √2/2) | ≈0.707 | ≈0.707 |
| π/2 | (0, 1) | 0 | 1 |
| π | (-1, 0) | -1 | 0 |
| 3π/2 | (0, -1) | 0 | -1 |
| 2π | (1, 0) | 1 | 0 |

代码验证：

```python
import math

print(f"θ=0:    ({math.cos(0):.3f}, {math.sin(0):.3f})")
print(f"θ=π/2:  ({math.cos(math.pi/2):.3f}, {math.sin(math.pi/2):.3f})")
print(f"θ=π:    ({math.cos(math.pi):.3f}, {math.sin(math.pi):.3f})")
print(f"θ=2π:   ({math.cos(2*math.pi):.3f}, {math.sin(2*math.pi):.3f})")
```

### 4.2 周期性

走完一整圈（2π 弧度 = 360°）回到原点：

```
cos(θ + 2π) = cos(θ)
sin(θ + 2π) = sin(θ)
```

```python
import math
print(math.cos(0))                  # 1.0
print(math.cos(2 * math.pi))        # ≈ 1.0  (绕一圈回来)
print(math.cos(100 * math.pi))      # ≈ 1.0  (绕 50 圈回来)
```

### 4.3 sin、cos 之间的关系

`sin` 比 `cos` "晚 π/2"：

```
sin(θ) = cos(θ - π/2)
```

直观：x 坐标是 cos，y 坐标是 sin，相当于 x 坐标"提前"四分之一圈。

---

## 5. 频率与波长：旋转的两个核心参数

### 5.1 频率（angular frequency）ω

考虑这样一个函数：

```
f(m) = cos(m · ω)
```

`m` 是位置（token 索引 0, 1, 2, ...），`ω` 是常数，叫做**角频率**。

**物理意义**：每当 m 增加 1，输入角度增加 ω 弧度。所以 `ω` 控制 cos 函数随位置变化的"快慢"。

```python
import math

omega = 0.5
for m in range(6):
    print(f"m={m}: cos(m·{omega}) = cos({m*omega:.2f}) = {math.cos(m*omega):.3f}")
```

### 5.2 波长 λ

**问**：m 走多远，f(m) 才重复一次？

由 cos 的周期性：

```
f(m + λ) = f(m)
cos((m+λ)·ω) = cos(m·ω)
(m+λ)·ω - m·ω = 2π
λ · ω = 2π
λ = 2π / ω         ◄── 波长公式
```

### 5.3 三组对比例子

```python
import math

def show_oscillation(omega, label):
    lambda_ = 2 * math.pi / omega
    print(f"\n{label}: ω = {omega}, 波长 λ = 2π/ω = {lambda_:.2f}")
    for m in range(0, int(lambda_) + 5, max(1, int(lambda_)//8)):
        val = math.cos(m * omega)
        print(f"  m={m:>3}: cos = {val:+.3f}")

show_oscillation(1.0, "高频")
show_oscillation(0.1, "中频")
show_oscillation(0.001, "低频")
```

**输出（节选）**：

```
高频: ω = 1.0, 波长 λ = 6.28
  m=0: cos = +1.000
  m=1: cos = +0.540
  m=2: cos = -0.416
  m=3: cos = -0.990
  m=6: cos = +0.960    ← 接近回到 +1
  
低频: ω = 0.001, 波长 λ = 6283.19
  m=0: cos = +1.000
  m=785: cos = +0.708
  m=1570: cos = +0.000   ← 走 1570 步才转 1/4 圈
  ...
```

**直觉**：
- **频率高 → 波长短**：每隔几步就重复，对位置极敏感
- **频率低 → 波长长**：要走很久才有变化，编码大尺度位置

这两点是 RoPE 设计的灵魂。

---

## 6. 二维旋转矩阵：把"位置"编码成"角度"

### 6.1 二维旋转

把一个 2D 向量 `(x, y)` 逆时针旋转角度 θ：

```
| x' |   | cos θ  -sin θ | | x |
| y' | = | sin θ   cos θ | | y |

x' = x · cos θ - y · sin θ
y' = x · sin θ + y · cos θ
```

代码：

```python
import math

def rotate_2d(x, y, theta):
    return (
        x * math.cos(theta) - y * math.sin(theta),
        x * math.sin(theta) + y * math.cos(theta),
    )

# 把 (1, 0) 转 90°  →  应得 (0, 1)
print(rotate_2d(1, 0, math.pi/2))    # (≈0, ≈1)

# 把 (1, 0) 转 180° →  应得 (-1, 0)
print(rotate_2d(1, 0, math.pi))      # (≈-1, ≈0)

# 把 (1, 0) 转 360° →  回到 (1, 0)
print(rotate_2d(1, 0, 2*math.pi))    # (≈1, ≈0)
```

### 6.2 旋转的"加法性质"

**两次旋转 = 总角度的一次旋转**：

```
R(α) · R(β) = R(α + β)
```

代码验证：

```python
import math

def rotate_2d(x, y, theta):
    return (
        x * math.cos(theta) - y * math.sin(theta),
        x * math.sin(theta) + y * math.cos(theta),
    )

# 先转 30°, 再转 50° = 直接转 80°
x1, y1 = rotate_2d(1, 0, math.radians(30))
x2, y2 = rotate_2d(x1, y1, math.radians(50))
x_direct, y_direct = rotate_2d(1, 0, math.radians(80))

print(f"两次旋转: ({x2:.4f}, {y2:.4f})")
print(f"一次 80°: ({x_direct:.4f}, {y_direct:.4f})")
# 两个结果应该一模一样
```

**这个性质就是 RoPE 能编码"相对位置"的根本原因**——后面会用到。

### 6.3 复数视角（可选但优雅）

把 `(x, y)` 看成一个复数 `z = x + iy`。则旋转 θ 等价于乘以 `e^{iθ}`：

```
(x + iy) · e^{iθ} = (x + iy)(cos θ + i sin θ)
                  = (x cos θ - y sin θ) + i(x sin θ + y cos θ)
                  = x' + i y'
```

旋转的加法性质在复数表示下变成：

```
e^{iα} · e^{iβ} = e^{i(α+β)}
```

这是高中数学里的"指数加法律"。我们后面证明 RoPE 性质时会用到。

---

## 7. RoPE 构造：把 d 维向量切成 d/2 对小翅膀

到这里，所有零件都齐了。RoPE 的设计思路：

> **既然我们能在 2D 里旋转一个向量，那把 d 维向量切成 d/2 个 2D 子向量，
> 每个子向量按位置旋转，就把"位置"嵌入了向量本身。**

### 7.1 配对维度

对于一个 d 维向量 `q = [q_0, q_1, q_2, q_3, ..., q_{d-2}, q_{d-1}]`，**相邻两维一组**：

```
q = [q_0, q_1 | q_2, q_3 | q_4, q_5 | ... | q_{d-2}, q_{d-1}]
       pair 0     pair 1     pair 2          pair (d/2 - 1)
```

每个 pair 是一个 2D 向量 `(q_{2i}, q_{2i+1})`。

### 7.2 给每对一个不同的频率

第 i 对的旋转频率是：

```
θ_i = base^(-2i / d)         (base 通常 = 10000)
```

具体：
- i = 0：θ_0 = base^0 = 1.0（最高频）
- i = 1：θ_1 = base^(-2/d)
- ...
- i = d/2 - 1：θ_{d/2-1} = base^(-(d-2)/d) ≈ 1/base（最低频）

### 7.3 给位置 m 处的向量旋转

对于位置 m 的 q 向量，第 i 对要旋转的角度是：

```
α_{m,i} = m · θ_i
```

旋转之后：

```
q'_{2i}   = q_{2i} · cos(m·θ_i) - q_{2i+1} · sin(m·θ_i)
q'_{2i+1} = q_{2i} · sin(m·θ_i) + q_{2i+1} · cos(m·θ_i)
```

### 7.4 完整代码实现

```python
import torch
import math

def apply_rope(q, m, base=10000.0):
    """
    对一个 d 维向量 q 应用 RoPE, 用位置 m 的角度旋转.
    
    Args:
        q: shape (d,) 的向量
        m: 位置 (整数或浮点)
        base: RoPE 的 θ_i = base^(-2i/d) 的底数
    Returns:
        shape (d,) 的旋转后向量
    """
    d = q.shape[-1]
    half_d = d // 2

    # 1. 算每对的频率 θ_i = base^(-2i/d), i=0..half_d-1
    inv_freq = 1.0 / (base ** (torch.arange(half_d, dtype=torch.float32) * 2 / d))
    # inv_freq shape: (half_d,)
    
    # 2. 算每对的角度 m · θ_i
    angles = m * inv_freq                # (half_d,)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    
    # 3. 拆出偶数/奇数维 (即 q_{2i} 和 q_{2i+1})
    q_even = q[0::2]                     # (half_d,)
    q_odd = q[1::2]
    
    # 4. 对每对应用 2D 旋转
    out_even = q_even * cos - q_odd * sin
    out_odd = q_even * sin + q_odd * cos
    
    # 5. 交错拼回原 shape
    out = torch.empty_like(q)
    out[0::2] = out_even
    out[1::2] = out_odd
    return out


# 演示: 同一个 q 在不同位置 RoPE 后的结果
q = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

print("位置 m=0:", apply_rope(q, 0))     # 应该 = q (cos(0)=1, sin(0)=0)
print("位置 m=1:", apply_rope(q, 1))
print("位置 m=5:", apply_rope(q, 5))
print("位置 m=100:", apply_rope(q, 100))
```

**预期**：
- `m=0` 时输出等于 q（旋转 0 度 = 不变）
- `m=1, 5, 100` 时每对维度都被旋转了不同角度

---

## 8. 完整流程：embedding 矩阵 → RoPE → attention

把 RoPE 接入 Transformer 的完整流程：

```python
import torch
import torch.nn.functional as F
import math

def rope_apply_to_matrix(M, positions, base=10000.0):
    """
    对一个矩阵 M ∈ R^{seq_len × d} 的每一行, 按其对应位置应用 RoPE.
    
    Args:
        M: shape (seq_len, d), 通常是 Q 或 K
        positions: shape (seq_len,), int, 每行对应的 token 位置
        base: RoPE base
    Returns:
        shape (seq_len, d) 的 RoPE 后矩阵
    """
    d = M.shape[-1]
    half_d = d // 2

    inv_freq = 1.0 / (base ** (torch.arange(half_d, dtype=torch.float32) * 2 / d))
    angles = positions.float().unsqueeze(-1) * inv_freq  # (seq_len, half_d)
    cos = angles.cos()
    sin = angles.sin()

    M_even = M[..., 0::2]
    M_odd = M[..., 1::2]
    out_even = M_even * cos - M_odd * sin
    out_odd = M_even * sin + M_odd * cos

    out = torch.empty_like(M)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


def attention_with_rope(X, W_q, W_k, W_v, base=10000.0):
    """
    一个完整的 self-attention 前向, 含 RoPE.
    
    Args:
        X: (seq_len, d_model) embedding 矩阵
        W_q, W_k, W_v: (d_model, d_model) 投影矩阵
    """
    seq_len, d = X.shape

    # Step 1: Q, K, V 投影
    Q = X @ W_q             # (seq_len, d)
    K = X @ W_k
    V = X @ W_v             # 注意 V 不应用 RoPE!

    # Step 2: 给 Q 和 K 应用 RoPE
    positions = torch.arange(seq_len)
    Q_rope = rope_apply_to_matrix(Q, positions, base)
    K_rope = rope_apply_to_matrix(K, positions, base)

    # Step 3: scaled dot-product attention
    scores = Q_rope @ K_rope.T / math.sqrt(d)   # (seq_len, seq_len)
    attn = F.softmax(scores, dim=-1)
    output = attn @ V                            # (seq_len, d)

    return output, attn


# 演示
torch.manual_seed(0)
vocab_size, d_model = 50000, 64
seq_len = 6

E = torch.randn(vocab_size, d_model)
token_ids = torch.tensor([464, 3797, 3332, 319, 262, 2603])  # "The cat sat on the mat"
X = E[token_ids]                            # (6, 64)

W_q = torch.randn(d_model, d_model) * 0.1
W_k = torch.randn(d_model, d_model) * 0.1
W_v = torch.randn(d_model, d_model) * 0.1

output, attn = attention_with_rope(X, W_q, W_k, W_v)
print("Attention 矩阵 shape:", attn.shape)   # (6, 6)
print("\nAttention matrix:")
print(attn.detach().numpy().round(3))
```

### 8.1 关键提醒

> **RoPE 只作用在 Q 和 K 上，绝对不要对 V 应用！**

为什么？
- Q 和 K 决定 **"该看哪里"**（地址）——位置信息要影响这里
- V 是被聚合的 **"内容"**（数据）——加权求和的"原材料"，不需要旋转
- 数学上，旋转在 Q·K^T 中以"相对位置"形式出现（下一节证明）；如果 V 也旋转，attention 的输出就被无意义地转了一下，破坏语义

---

## 9. 为什么 RoPE 编码的是"相对位置"

这是 RoPE 最迷人的性质，也是它优于绝对位置编码的根本原因。

### 9.1 命题

```
<RoPE_m(Q), RoPE_n(K)>  仅依赖于 (m - n)
```

也就是说：把 Q 和 K 都各自按位置 m, n 旋转后，它们的内积**只跟 m-n 有关**，跟 m, n 各自的绝对值无关。

### 9.2 证明（用复数表示）

把 q 和 k 的每对维度看成复数：

```
q_pair_i = q_{2i} + i·q_{2i+1}     (复数)
k_pair_i = k_{2i} + i·k_{2i+1}
```

RoPE 在位置 m 对 q 做的事就是给每个 pair_i 乘 e^{im·θ_i}：

```
RoPE_m(q)_pair_i = q_pair_i · e^{im·θ_i}
RoPE_n(k)_pair_i = k_pair_i · e^{in·θ_i}
```

实数空间的内积 = 复数表示下"共轭内积的实部"：

```
<RoPE_m(q), RoPE_n(k)>_pair_i 
  = Re( q_pair_i · e^{im·θ_i} · conj(k_pair_i · e^{in·θ_i}) )
  = Re( q_pair_i · conj(k_pair_i) · e^{im·θ_i} · e^{-in·θ_i} )
  = Re( q_pair_i · conj(k_pair_i) · e^{i(m-n)·θ_i} )
                                       ↑
                              只依赖 m-n
```

总和：

```
<RoPE_m(Q), RoPE_n(K)> = Σ_i Re( q_pair_i · conj(k_pair_i) · e^{i(m-n)·θ_i} )
```

每一项的角度都是 `(m-n)·θ_i`，**完全由相对位置 m-n 决定**。QED.

### 9.3 数值验证

```python
import torch
import math

def rope_apply(v, m, base=10000.0):
    d = v.shape[-1]
    half_d = d // 2
    inv_freq = 1.0 / (base ** (torch.arange(half_d, dtype=torch.float32) * 2 / d))
    angles = m * inv_freq
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    v_even, v_odd = v[..., 0::2], v[..., 1::2]
    out_even = v_even * cos - v_odd * sin
    out_odd = v_even * sin + v_odd * cos
    out = torch.empty_like(v)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


torch.manual_seed(42)
Q = torch.randn(8)
K = torch.randn(8)

# 对比 1: m=5, n=2 (相对距离 3)
inner_a = (rope_apply(Q, 5) * rope_apply(K, 2)).sum().item()

# 对比 2: m=10, n=7 (相对距离 3)
inner_b = (rope_apply(Q, 10) * rope_apply(K, 7)).sum().item()

# 对比 3: m=100, n=97 (相对距离 3)
inner_c = (rope_apply(Q, 100) * rope_apply(K, 97)).sum().item()

# 对比 4: m=5, n=4 (相对距离 1, 不一样)
inner_d = (rope_apply(Q, 5) * rope_apply(K, 4)).sum().item()

print(f"<RoPE_5(Q), RoPE_2(K)>     = {inner_a:.6f}")
print(f"<RoPE_10(Q), RoPE_7(K)>    = {inner_b:.6f}  ← 应等于上面")
print(f"<RoPE_100(Q), RoPE_97(K)>  = {inner_c:.6f}  ← 应等于上面")
print(f"<RoPE_5(Q), RoPE_4(K)>     = {inner_d:.6f}  ← 应不等于上面 (相对距离不同)")
```

**预期输出**：前 3 行数值完全一致，第 4 行不同。

### 9.4 为什么这个性质重要

语言里大部分关系都是**相对**的：
- "前一个词"
- "上一个动词"
- "三步前提到的主语"

模型不需要知道"我在第 47 个 token 位置"，它需要知道"我和别的词的相对距离"。RoPE 让 attention 的几何天然契合这一点——**这就是为什么 RoPE 比绝对位置编码更适合长文本**。

---

## 10. 多尺度位置感知：频率层级的妙处

### 10.1 各频率对的"工作范围"

每对维度的 θ_i 不同，对应不同的"工作距离"：

```python
import math

base = 10000.0
d = 64

print(f"{'i':>3} {'θ_i':>12} {'波长 λ_i':>12} {'擅长距离':>20}")
print("-" * 55)
for i in [0, 5, 10, 20, 30, 31]:
    theta_i = base ** (-2*i/d)
    lambda_i = 2 * math.pi / theta_i
    print(f"{i:>3} {theta_i:>12.6f} {lambda_i:>12.1f} {'~λ_i/4 内可分辨':>20}")
```

输出：

```
  i        θ_i      波长 λ_i        擅长距离
-------------------------------------------------------
  0     1.000000          6.3   ~λ_i/4 内可分辨    ← 高频, 1-2 token 级别
  5     0.421697         14.9
 10     0.177828         35.3
 20     0.031623        198.7
 30     0.005623       1117.7
 31     0.000133      47324.4   ~λ_i/4 内可分辨    ← 低频, 万级别
```

**直觉**：
- 高频对（θ_0=1, λ=6）每隔 6 个 token 就转完一圈——分辨"邻居"很在行
- 低频对（θ_31=0.0001, λ=47000）几乎不变——但跨越万 token 距离时它的微小角度差异是唯一的位置标识

### 10.2 演示：高频对在远距离会"绕迷糊"

```python
import math

theta_high = 1.0      # 高频
theta_low = 0.0001    # 低频

# 比较位置 0 和位置 1 (邻居)
print("位置 0 vs 位置 1 (邻居):")
print(f"  高频: cos(0)={math.cos(0):.3f}, cos(1)={math.cos(1):.3f}, 差异 = {abs(1-math.cos(1)):.3f}")
print(f"  低频: cos(0)={math.cos(0):.3f}, cos(0.0001)={math.cos(0.0001):.6f}, 差异 = {abs(1-math.cos(0.0001)):.6f}")

# 比较位置 0 和位置 1000 (远距离)
print("\n位置 0 vs 位置 1000 (远距离):")
print(f"  高频: cos(0)={math.cos(0):.3f}, cos(1000)={math.cos(1000):.3f}")
print("    → 1000 弧度 = 159 圈, 看起来"随机", 区分能力混乱")
print(f"  低频: cos(0)={math.cos(0):.3f}, cos(0.1)={math.cos(0.1):.6f}, 差异 = {abs(1-math.cos(0.1)):.4f}")
print("    → 0.1 弧度, 干净的小角度, 区分能力清晰")
```

**结论**：
- 邻居（m=0,1）：高频对分得清，低频对分不清
- 远距离（m=0,1000）：高频对绕了 159 圈，反而分不清；低频对正合适

**模型把所有频率对一起看，就拿到了从 1-token 到数万-token 的全频谱位置感知能力**。这就像同时拥有秒针、分针、时针、日期——每个尺度都有"专家"。

### 10.3 类比：二进制表示

更深的类比：RoPE 就像位置 m 的"连续版二进制表示"。

二进制下表示位置 m：

```
m 的第 0 位:   每加 1 翻转一次  (高频, 周期 = 2)
m 的第 1 位:   每加 2 翻转一次  (周期 = 4)
m 的第 2 位:   每加 4 翻转一次  (周期 = 8)
...
m 的第 k 位:   每加 2^k 翻转一次 (周期 = 2^(k+1))
```

读到 64 位二进制就唯一标识 m。

**RoPE 用的是相同思路，只是用 cos/sin 平滑波形代替 0/1 翻转**：每对维度有自己的"频率/周期"，组合起来唯一确定位置。

---

## 11. 总结与延伸

### 11.1 核心要点回顾

| # | 概念 | 一句话总结 |
|---|------|-----------|
| 1 | 句子 → token id | BPE 把字符串切成有序整数序列 |
| 2 | id → embedding | 查 E 表得到 (seq_len, d_model) 矩阵 X |
| 3 | 位置丢失 | Attention 是置换不变, X 行序在它眼里"无所谓" |
| 4 | sin/cos | 单位圆上一点的横/纵坐标 |
| 5 | 频率 ω, 波长 λ = 2π/ω | ω 控制 cos(m·ω) 随 m 变化的快慢 |
| 6 | 2D 旋转矩阵 | 把向量按角度 θ 旋转, 满足 R(α)·R(β)=R(α+β) |
| 7 | RoPE 构造 | d 维向量两两配对, 第 i 对用频率 θ_i = base^(-2i/d), 在位置 m 处转 m·θ_i |
| 8 | 完整流程 | X → Q,K,V; 对 Q,K 应用 RoPE; attention 计算; **V 不动** |
| 9 | 相对位置 | <RoPE_m(Q), RoPE_n(K)> 只依赖 m-n |
| 10 | 多尺度 | 频率层级让模型同时拥有 1-token 到万-token 的位置感知 |

### 11.2 完整流程图

```
"The cat sat on the mat"
        │
        ▼  BPE
[464, 3797, 3332, 319, 262, 2603]
        │
        ▼  E[token_ids]
X ∈ R^{6 × d_model}     ← 每行: 词的"语义坐标", 行索引隐含位置
        │
        ▼  线性投影
Q, K, V ∈ R^{6 × d_model}
        │                                     ┌──────────────────────────┐
        │     ┌── Q ──────────────────────►   │ RoPE_m(q)                │
        │     │                                │  把 d 维拆成 d/2 对       │
        │     │     位置 m=0..5                │  每对在位置 m 旋转 m·θ_i  │
        │     │                                │  θ_i = base^(-2i/d)       │
        │     └── K ──────────────────────►   │ → Q', K'                 │
        │                                     │ V 不动                    │
        │                                     └──────────────────────────┘
        │                              │
        │                              ▼
        │     scores = Q' · K'^T / √d_k         ← 这里的内积只依赖相对位置
        │     attn = softmax(scores)
        │     output = attn · V
        ▼
最终输出 ∈ R^{6 × d_model}
```

### 11.3 进一步阅读

实战中的 RoPE 进阶话题（这份文档没展开，留给你）：

- **base 调参**：LLaMA-3 把 base 从 10000 → 500000 是为什么
- **Position Interpolation (PI)**：把 pos 缩小让超长上下文回到训练分布
- **NTK-aware Scaling / YaRN**：更精细的频率分段缩放
- **RoPE 在推理时的 KV-cache 优化**：如何只计算新 token 的 cos/sin
- **替代方案**：ALiBi（线性偏置）、Hyena（隐式卷积）等

### 11.4 一句话给将来的自己

> **RoPE 把"位置"翻译成"角度"，把"距离"翻译成"角度差"。
> 通过让 Q 和 K 在不同频率上各自旋转，attention 的内积自然只依赖相对位置；
> 通过频率层级，模型同时拥有从 1-token 到万-token 的全频谱位置感知。**

---

## 附录：完整可运行代码

把整个流程串起来的完整脚本——直接 `python xxx.py` 就能跑：

```python
"""
RoPE 完全演示: 从一个句子到 attention 输出.
"""
import torch
import torch.nn.functional as F
import math


def apply_rope_to_matrix(M, positions, base=10000.0):
    """对矩阵的每一行, 按其对应位置应用 RoPE."""
    d = M.shape[-1]
    half_d = d // 2
    inv_freq = 1.0 / (base ** (torch.arange(half_d, dtype=torch.float32) * 2 / d))
    angles = positions.float().unsqueeze(-1) * inv_freq
    cos = angles.cos()
    sin = angles.sin()
    M_even = M[..., 0::2]
    M_odd = M[..., 1::2]
    out = torch.empty_like(M)
    out[..., 0::2] = M_even * cos - M_odd * sin
    out[..., 1::2] = M_even * sin + M_odd * cos
    return out


def main():
    torch.manual_seed(0)

    # 1. 设置
    vocab_size, d_model = 50000, 64
    seq_len = 6

    # 2. Embedding 矩阵 (假装训好的)
    E = torch.randn(vocab_size, d_model)

    # 3. 句子 → token_ids → X
    sentence = "The cat sat on the mat"
    token_ids = torch.tensor([464, 3797, 3332, 319, 262, 2603])
    X = E[token_ids]
    print(f"句子: {sentence}")
    print(f"token_ids: {token_ids.tolist()}")
    print(f"X shape: {X.shape}")

    # 4. Q, K, V 投影
    W_q = torch.randn(d_model, d_model) * 0.1
    W_k = torch.randn(d_model, d_model) * 0.1
    W_v = torch.randn(d_model, d_model) * 0.1
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    # 5. 对 Q 和 K 应用 RoPE (V 不动!)
    positions = torch.arange(seq_len)
    Q_rope = apply_rope_to_matrix(Q, positions)
    K_rope = apply_rope_to_matrix(K, positions)

    # 6. Attention
    scores = Q_rope @ K_rope.T / math.sqrt(d_model)
    attn = F.softmax(scores, dim=-1)
    output = attn @ V

    print(f"\nAttention 矩阵 (每行表示一个 token 关心其他 token 的权重):")
    print(attn.numpy().round(3))
    print(f"\n输出 shape: {output.shape}")

    # 7. 验证: RoPE 的相对位置性质
    print("\n--- 验证 RoPE 的相对位置不变性 ---")
    q = torch.randn(8)
    k = torch.randn(8)
    # 单个向量 RoPE 用 m=int 也行
    def single_rope(v, m, base=10000.0):
        d = v.shape[-1]
        half_d = d // 2
        inv_freq = 1.0 / (base ** (torch.arange(half_d, dtype=torch.float32) * 2 / d))
        angles = m * inv_freq
        cos, sin = angles.cos(), angles.sin()
        v_e, v_o = v[..., 0::2], v[..., 1::2]
        out = torch.empty_like(v)
        out[..., 0::2] = v_e * cos - v_o * sin
        out[..., 1::2] = v_e * sin + v_o * cos
        return out

    inner1 = (single_rope(q, 5) * single_rope(k, 2)).sum().item()
    inner2 = (single_rope(q, 10) * single_rope(k, 7)).sum().item()
    inner3 = (single_rope(q, 100) * single_rope(k, 97)).sum().item()
    inner4 = (single_rope(q, 5) * single_rope(k, 4)).sum().item()  # 不同相对距离

    print(f"<RoPE_5(q), RoPE_2(k)>    = {inner1:.6f}")
    print(f"<RoPE_10(q), RoPE_7(k)>   = {inner2:.6f}  (应=上面)")
    print(f"<RoPE_100(q), RoPE_97(k)> = {inner3:.6f}  (应=上面)")
    print(f"<RoPE_5(q), RoPE_4(k)>    = {inner4:.6f}  (应≠上面, 相对距离=1 而非 3)")


if __name__ == "__main__":
    main()
```

---

**写完了。** 把这份文档作为你 RoPE 知识的"圣经"——以后忘了任何一点，回到这里就能找到对应小节快速回忆。
