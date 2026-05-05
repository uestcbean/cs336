# Transformer Block 深度解读 + Mask 机制

> 这份文档以**具体数据维度变化**为骨架, 把 Transformer Block 的内部机制讲到底.
> 包括: 整体架构, residual stream 视角, pre-norm 选择, attention/FFN 分工, mask 机制, 以及现代 LLM 变体.

---

## 0. 贯穿全文的具体配置

为了让讨论"踩在地上", 我固定一组参数:

| 参数 | 值 | 含义 |
|------|----|----|
| `batch` | 4 | 同时处理 4 个样本 |
| `seq` | 12 | 每个样本 12 个 token |
| `d_model` | 64 | 主隐藏维度 |
| `num_heads` | 4 | 注意力头数 |
| `d_head` | 16 | 每头维度 (= d_model / num_heads) |
| `d_ff` | 128 | FFN 内部高维空间 |
| `vocab_size` | 10000 | 词表大小 |

---

## 目录

1. [一个 Transformer Block 完整数据流](#1-一个-transformer-block-完整数据流)
2. [Residual Stream 视角](#2-residual-stream-视角现代解读)
3. [Pre-norm vs Post-norm](#3-pre-norm-vs-post-norm)
4. [Attention 和 FFN 的"分工"](#4-attention-和-ffn-的分工)
5. [Mask 机制深度解读](#5-mask-机制深度解读)
6. [Sublayer 1: Pre-norm + Attn + Residual 的 step-by-step](#6-sublayer-1-pre-norm--attn--residual-的-step-by-step)
7. [Sublayer 2: Pre-norm + FFN + Residual 的 step-by-step](#7-sublayer-2-pre-norm--ffn--residual-的-step-by-step)
8. [设计选择的"为什么"](#8-设计选择的为什么)
9. [现代 LLM 变体](#9-现代-llm-变体)
10. [一句话总结](#10-一句话总结)

---

## 1. 一个 Transformer Block 完整数据流

```
输入 x   shape: (4, 12, 64)        ← (batch, seq, d_model)
       │
       ├─────────────────────┐  ← 残差直连 (主路径)
       │                     │
       ▼                     │
   RMSNorm (ln1)              │   ← 给 attention 的"读取镜头"
   shape 不变: (4, 12, 64)     │
       │                     │
       ▼                     │
   Multi-Head Attention       │   ← 跨位置通信
   (with RoPE & causal mask)  │
   shape 不变: (4, 12, 64)     │
       │                     │
       └──── + ──────────────┘  ← 残差合并
       │
       │  shape: (4, 12, 64)
       │
       ├─────────────────────┐  ← 残差直连
       │                     │
       ▼                     │
   RMSNorm (ln2)              │   ← 给 FFN 的"读取镜头"
       │                     │
       ▼                     │
   SwiGLU FFN                 │   ← 单位置非线性变换 (内部升到 128 维)
       │                     │
       └──── + ──────────────┘  ← 残差合并
       │
输出 x   shape: (4, 12, 64)        ← shape 完全不变!
```

**关键事实**: 整个 block 输入输出 shape 完全相同 `(4, 12, 64)`. 这是 **残差结构** 的特征——
让多个 block 能"无缝堆叠"成 N 层.

---

## 2. Residual Stream 视角(现代解读)

把残差路径上的 `x` 想象成一条 **信息高速公路** (residual stream),
sublayers 是一个个 **加油站**:

```
  ┌─────────────────────────────────────────────────────► output
  │                                              ▲
  │  ┌──────────┐                                 │
  │  │ Attn 0   │  读 x → 计算 → 写回                │
  ▼  │          │═════════════════════════════════╪══►
  │  └──────────┘                                 │
  │  ┌──────────┐                                 │
  │  │ FFN 0    │                                 │
  │  │          │═════════════════════════════════╪══►
  │  └──────────┘                                 │
  │  ┌──────────┐                                 │
  │  │ Attn 1   │                                 │
  │  │          │═════════════════════════════════╪══►
  │  └──────────┘                                 │
                            ...
```

**每个 sublayer 的工作模式**:

```python
h = rmsnorm(weights["ln1.weight"], x, eps)   # 1. 读取一份归一化的 x
h = multihead_self_attention_with_rope(...)  # 2. 计算
x = x + h                                    # 3. 写回到残差流
```

代码层面 `x = x + h` **不是"叠加输出"**, 而是 **sublayer 把它的"贡献"写入残差流**.

### 残差流视角解释了什么

1. **为什么残差不被 norm 改变**: 信息一旦写进残差流, 不会被后续的 norm 抹掉
   (norm 只发生在"读取镜头", 主路径上的 x 始终保留)

2. **为什么 sublayers 之间能互相协作**:
   层 N 写的信息, 所有后续层都能读到——形成"通信网络"

3. **为什么"层是函数式叠加"**:
   每层就是一个"可选的小贡献", 把它们的作用累加起来

---

## 3. Pre-norm vs Post-norm

### 两种架构

```
原始 Transformer (Post-norm):              现代 LLM (Pre-norm):

x' = LayerNorm(x + Sublayer(x))            x' = x + Sublayer(LayerNorm(x))
     ↑ norm 在残差合并 之后                       ↑ norm 在 sublayer 入口
```

### 为什么 post-norm 训不深

反向传播时, 从 loss 到第 0 层要经过 N 个 LayerNorm. 每个 LayerNorm 会**轻微衰减**梯度.

**Post-norm 的梯度链**:
```
∂L/∂x_0 = (LN_衰减) × (LN_衰减) × ... × (LN_衰减) × ∂L/∂x_N
                       ↑ 12+ 个相乘后, 梯度近似为 0
```

N=12 还能撑, N=96 (GPT-3) 就崩了.

### Pre-norm 的"残差直通车"

Pre-norm 里残差路径是恒等映射, 梯度可以**直接从顶层"高速公路"传到底层**:

```
∂L/∂x_0 = ∂L/∂x_N + (sublayer 项的梯度, 加性贡献)
            ↑ 主导项, 几乎不衰减
```

**这就是为什么 pre-norm 能训 100+ 层**——梯度有一条"绕过所有 norm"的直通车.

### 代价: 需要 ln_final

主路径上 x 累加了 N 次 sublayer 输出, 幅度会爆.
所以 final 出口加一道 RMSNorm (`ln_final`) 拉回标准尺度.

> 这是经典工程权衡: post-norm 不需要 ln_final 但训不深;
> pre-norm 训得深但要 ln_final 收尾.
> 所有现代 LLM (LLaMA、Qwen、Mistral) 都选 pre-norm.

---

## 4. Attention 和 FFN 的"分工"

### 一个清晰的二分法

| | Attention | FFN |
|---|----------|-----|
| 作用 | **跨位置通信** (mixing) | **单位置计算** (processing) |
| "看哪里" | 看其他位置的内容 | 只看自己 |
| 是否依赖 seq 长度 | 是 (O(seq²)) | 否 (每个位置独立处理) |
| 参数量 | ~4 × d_model² = ~16k | ~3 × d_model × d_ff = ~24k |
| 数据流 | 信息**横向**流动 | 信息**纵向**深化 |

### 形象比喻

把每个 token 想象成一个"细胞", 它们排成一列:

```
[The] [cat] [sat] [on] [the] [mat]
```

- **Attention** = 让每个细胞**伸出触角**到其他细胞那里收集信息
- **FFN** = 每个细胞**关起门来**对自己手头的信息做精细加工 (非线性变换)

**两者必须搭配**:
- 只有 attn 没有 FFN: 能跨位置传信息, 但不能"思考"——表达力弱
- 只有 FFN 没有 attn: 能"深思", 但不知道其他位置——退化为逐位置 MLP

### 为什么 FFN 要先升维到 d_ff 再降回来?

```python
# SwiGLU 内部:
h1 = x @ W1.T              # (4, 12, 64) → (4, 12, 128)        升维 (d_ff = 2×d_model 在我们例子里, 真实模型 ~4×)
h3 = x @ W3.T              # (4, 12, 64) → (4, 12, 128)        升维
gated = silu(h1) * h3      # (4, 12, 128)                       高维空间做非线性
out = gated @ W2.T         # (4, 12, 128) → (4, 12, 64)        降维
```

**为什么先升维**? FFN 的"思考能力"主要来自高维空间的非线性映射.
- 64 维 ReLU 后再降 = 一个粗糙的非线性
- 64 → 128 ReLU → 64 = 一个精细得多的非线性

> 这就是为什么 FFN 占 Transformer 总参数的 ~2/3 ——大部分容量都在 FFN 的升降维里.

---

## 5. Mask 机制深度解读

### 5.1 Mask 在代码里出现的两个地方

**构造** ([model.py:165-167](../tests/algo/model.py#L165-L167)):
```python
def _causal_mask(seq_len: int, device) -> Tensor:
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
```

**使用** ([model.py:139-140](../tests/algo/model.py#L139-L140)):
```python
if mask is not None:
    scores = scores.masked_fill(~mask, float("-inf"))
```

### 5.2 Mask 长什么样

`_causal_mask(seq_len=6, ...)` 返回的张量 (用 `T`/`F` 表示 True/False):

```
       key_0  key_1  key_2  key_3  key_4  key_5
query_0  T      F      F      F      F      F      ← 位置 0 只能看自己
query_1  T      T      F      F      F      F      ← 位置 1 能看 0, 1
query_2  T      T      T      F      F      F
query_3  T      T      T      T      F      F
query_4  T      T      T      T      T      F
query_5  T      T      T      T      T      T      ← 最后一个能看到所有历史
```

shape: `(6, 6)` bool tensor, 下三角.

### 5.3 为什么必须有 causal mask?

LLM 训练做的是 "next token prediction":

```
输入: "The cat sat on the mat"
       0     1    2   3   4   5

模型对每个位置同时预测下一个 token:
  位置 0 → 预测位置 1 (target = "cat")
  位置 1 → 预测位置 2 (target = "sat")
  ...
```

**关键约束**: 位置 m 在做"预测下一个 token"时, 只能用位置 0..m 的信息——不能"偷看"未来.

如果没 mask, 位置 1 可以直接看到位置 2 ("sat"),
那它的"预测任务"就成了**作弊**——模型啥也学不到.

### 5.4 数学上 mask 怎么生效

#### 关键技巧: 把要屏蔽的位置设成 `-inf`

```python
scores = scores.masked_fill(~mask, float("-inf"))
```

走具体数字 (一行 6 个 key 的 attention scores):

```
原 scores (Q · K^T / √d_k 的一行):
  [0.5, 0.3, 0.8, 0.2, 0.7, 0.1]   ← query_2 跟 6 个 key 的得分

mask 行 (query_2 的): [T, T, T, F, F, F]
~mask 行:             [F, F, F, T, T, T]

masked_fill 后:
  [0.5, 0.3, 0.8, -inf, -inf, -inf]
```

#### softmax 后 -inf 自动变 0

```
减最大值 0.8: [-0.3, -0.5, 0.0, -inf, -inf, -inf]
exp:          [ 0.74, 0.61, 1.00, 0.0, 0.0, 0.0]   ← exp(-inf) = 0
和:           = 2.35
softmax:      [ 0.31, 0.26, 0.43, 0.0, 0.0, 0.0]
                                  ↑↑↑↑↑↑↑↑↑↑↑
                          被屏蔽的 key 完全不获得权重
```

`weights @ V`:
```
output_2 = 0.31·V[0] + 0.26·V[1] + 0.43·V[2] + 0·V[3] + 0·V[4] + 0·V[5]
                                                   ↑ 完全不参与
```

V[3]、V[4]、V[5] 的内容**根本进不来**. 位置 2 永远看不到未来.

### 5.5 为什么用 `-inf` 而不是 `0`?

非常常见的初学者疑问.

#### 错误的"用 0"思路

```python
scores = scores * mask    # ❌ 错误!
```

```
原 scores 行: [0.5, 0.3, 0.8, 0.2, 0.7, 0.1]
mask 后:      [0.5, 0.3, 0.8, 0.0, 0.0, 0.0]   ← 后 3 个被设成 0
```

然后 softmax:
```
exp:    [1.65, 1.35, 2.23, 1.00, 1.00, 1.00]   ← exp(0) = 1, 不是 0!!
权重:    [0.18, 0.15, 0.25, 0.11, 0.11, 0.11]   ← 被屏蔽位置仍有 ~11% 权重!
```

**问题**: `exp(0) = 1`. 被屏蔽的位置仍然有非零权重. 屏蔽失败.

#### `-inf` 才是正确解

只有 `-inf` 通过 `exp(-inf) = 0` 才能保证被屏蔽位置的权重**精确为 0**:

```
exp(-inf) = 0     ← 真正的"屏蔽"
exp(0)    = 1     ← 反而成了"标准权重"
```

### 5.6 不同类型的 mask

| Mask 类型 | 用途 | 形状 |
|---------|------|------|
| **Causal** (我们用的) | 自回归 LM (GPT, LLaMA) | 下三角 |
| **Bidirectional** | 双向 (BERT) | 全 True |
| **Padding** | 不等长序列 | 把 pad 位置标 False |
| **Sliding window** | 长上下文 (Mistral) | 仅最近 N 个为 True |
| **Custom** | 多模态、视觉等 | 任意定制 |

我们这里只用 causal. 真实 batch 训练时通常 **causal AND padding** 取逻辑与.

### 5.7 训练 vs 推理的 mask

| | 训练 | 推理 |
|---|------|------|
| 是否需要 causal mask | 是 | 是 |
| 一次处理多少 token | 整个 seq | 一个 (或在 KV-cache 里只新增一个) |

推理时虽然每步只算 1 个新 token, **但生成完后续 token 时仍然要 causal**——因为不能让早先生成的 token 看到后来的.

> KV-cache 是个性能优化: 把之前算过的 K/V 缓存起来, 每步只算"新 query".
> 这种情况下显式 mask 反而不需要 (只有 1 个 query, 自动只能看历史的 K).
> 但逻辑上仍然是 causal.

---

## 6. Sublayer 1: Pre-norm + Attn + Residual 的 step-by-step

### 完整代码

```python
h = rmsnorm(weights["ln1.weight"], x, eps)
h = multihead_self_attention_with_rope(
    h,
    w_q=weights["attn.q_proj.weight"],
    w_k=weights["attn.k_proj.weight"],
    w_v=weights["attn.v_proj.weight"],
    w_o=weights["attn.output_proj.weight"],
    num_heads=num_heads,
    max_seq_len=max_seq_len,
    theta=theta,
    token_positions=pos,
)
x = x + h
```

### 数据流追踪 (具体维度)

```
入口:
  x:                        (4, 12, 64)         "残差流当前状态"

Step 1: RMSNorm (读取镜头)
  ln1.weight:               (64,)
  h = rmsnorm(ln1.weight, x, eps)
  h:                        (4, 12, 64)         归一化后的副本, 主路径 x 不变

Step 2: Q, K, V 投影 (一次大 matmul)
  q_proj_weight:            (64, 64)
  Q = h @ q_proj_weight.T:  (4, 12, 64)
  K = h @ k_proj_weight.T:  (4, 12, 64)
  V = h @ v_proj_weight.T:  (4, 12, 64)

Step 3: 拆头
  Q.reshape((4, 12, 4, 16)):                    (4, 12, 4, 16)
  Q.transpose(-3, -2):                           (4, 4, 12, 16)   ← (batch, head, seq, d_head)
  K, V 同样处理

Step 4: 给 Q, K 应用 RoPE
  Q = rope(Q, positions, d_head=16, theta=10000):  (4, 4, 12, 16)
  K = rope(K, positions, d_head=16, theta=10000):  (4, 4, 12, 16)
  V 不动

Step 5: 因果 mask
  mask = _causal_mask(12):                       (12, 12) bool, 下三角

Step 6: SDPA
  scores = Q @ K.transpose(-2,-1) / √16:         (4, 4, 12, 12)
  scores = scores.masked_fill(~mask, -inf):      (4, 4, 12, 12)   ← mask broadcast 到 (4, 4, 12, 12)
  weights = softmax(scores, dim=-1):              (4, 4, 12, 12)   ← 每行和为 1, 上三角全 0
  out = weights @ V:                              (4, 4, 12, 16)

Step 7: 合并头
  out.transpose(-3, -2):                         (4, 12, 4, 16)
  out.reshape((4, 12, 64)):                      (4, 12, 64)

Step 8: 输出投影
  o_proj_weight:                                  (64, 64)
  h = out @ o_proj_weight.T:                      (4, 12, 64)

Step 9: 残差合并
  x = x + h:                                      (4, 12, 64)      ← 写回残差流
```

### 信息含义变化

| 时刻 | x 的语义 |
|------|---------|
| 进入 sublayer 前 | "位置 m 的当前认知" |
| RMSNorm 后 (h) | "归一化后的快照, 准备用于 attention 计算" |
| Attention 后 | "位置 m 看了一眼上下文, 收集到的相关信息" |
| 残差合并后 | "原认知 + 上下文增强" |

---

## 7. Sublayer 2: Pre-norm + FFN + Residual 的 step-by-step

### 完整代码

```python
h = rmsnorm(weights["ln2.weight"], x, eps)
h = swiglu(
    w1=weights["ffn.w1.weight"],
    w2=weights["ffn.w2.weight"],
    w3=weights["ffn.w3.weight"],
    x=h,
)
x = x + h
```

### 数据流追踪

```
入口:
  x:                           (4, 12, 64)      "已被 attn 加工过的认知"

Step 1: RMSNorm
  h = rmsnorm(ln2.weight, x, eps):
  h:                           (4, 12, 64)

Step 2: SwiGLU (内部 3 步)
  ffn.w1.weight:               (128, 64)
  ffn.w2.weight:               (64, 128)
  ffn.w3.weight:               (128, 64)

  Step 2a: 升维
    h1 = h @ w1.T:             (4, 12, 128)     "激活流"
    h3 = h @ w3.T:             (4, 12, 128)     "门控信号"

  Step 2b: 高维非线性
    silu(h1):                  (4, 12, 128)     SiLU 激活
    gated = silu(h1) * h3:     (4, 12, 128)     element-wise 门控相乘

  Step 2c: 降维
    h = gated @ w2.T:          (4, 12, 64)      回到 d_model 空间

Step 3: 残差合并
  x = x + h:                   (4, 12, 64)      写回残差流
```

### 信息含义变化

| 时刻 | x 的语义 |
|------|---------|
| 进入 sublayer 前 | "经过 attn 后的认知 (含上下文)" |
| RMSNorm 后 (h) | "归一化的副本" |
| 升维到 d_ff (h1, h3) | "在更高维空间展开, 给非线性更大的施展空间" |
| SiLU + 门控 | "做精细的非线性变换" |
| 降维回 d_model | "提炼后的精华回到主路径维度" |
| 残差合并后 | "原认知 + 非线性增强" |

---

## 8. 设计选择的"为什么"

### 8.1 为什么 RMSNorm 不用 LayerNorm

```
LayerNorm:  y = γ · (x - mean(x)) / sqrt(var(x) + eps) + β
RMSNorm:    y = γ · x / sqrt(mean(x²) + eps)
```

**差别**: RMSNorm 省掉了 "减均值" 和 "加偏置 β".

**为什么省掉无害**:
- 实证: LLM 里"均值居中"对效果几乎没贡献
- 计算少 ~30%
- 参数少 ~50% (少了 β)
- 训练精度无影响

LLaMA、Qwen、Mistral 全用 RMSNorm.

### 8.2 为什么 Linear 没有 bias

PyTorch 默认 `nn.Linear(in, out, bias=True)`, 但现代 LLM 普遍 `bias=False`:
- 实验证明 bias 对效果影响 < 0.1%
- 省下的参数可以投到更大的 d_model

### 8.3 为什么 SwiGLU 不用 ReLU

旧 Transformer: `FFN(x) = W2 · ReLU(W1 · x)`
新版 SwiGLU: `FFN(x) = (SiLU(W1·x) ⊙ W3·x) · W2`

**门控的强大之处**:
- `SiLU(W1·x)` 是"激活"
- `W3·x` 是"门控信号"
- element-wise 乘 = 门控信号能动态地"选择性放大或抑制激活的不同维度"

实证: SwiGLU 在 LLM 上比 ReLU 普遍提升 1-2 个 perplexity 点,
相当于免费送 10% 的训练计算量. 代价: 多 50% 的 FFN 参数.

---

## 9. 现代 LLM 变体

我们的实现是"标准 GPT-2 + 现代化升级". 实际生产 LLM 里你会看到:

### 9.1 GQA / MQA (节省 KV cache 内存)

```
原版 MHA:        所有 head 都有独立 K, V
GQA (Grouped):   多个 head 共享一组 K, V (e.g., 32 heads / 8 groups)
MQA (Multi):     所有 head 共享同一组 K, V
```

LLaMA-2 70B 用 GQA, 推理时 KV cache 内存降到 1/8.

### 9.2 MoE (Mixture of Experts)

把 FFN 替换成"多个 FFN 专家 + router": 每个 token 只激活 2-4 个专家.

```
原版 FFN:        所有 token 走同一个 FFN
MoE FFN:         有 8 个 FFN, router 选择最相关的 2 个
```

DeepSeek-V3、Mixtral 用 MoE. **总参数大幅增长, 但激活参数不变**——容量大但推理便宜.

### 9.3 Parallel Block (PaLM)

把 attn 和 FFN 改成**并行**而不是串行:

```python
# 串行 (我们这版):
x = x + attn(norm(x))
x = x + ffn(norm(x))

# 并行 (PaLM):
x = x + attn(norm(x)) + ffn(norm(x))
```

省一次 RMSNorm 的开销, 速度更快, 效果接近.

### 9.4 PostNorm 的"复活"

最近一些工作发现, 在小心初始化下 post-norm 也能训深——但这是研究前沿, 不是主流.

---

## 10. 一句话总结

> **每个 Transformer Block 是一个"先看再想"的单元:
> 先用 attention 跨位置通信 (看别人), 再用 FFN 单位置加工 (自己想).
> 两个 sublayer 都通过"残差直通 + pre-norm"的模板嵌入到一条信息高速公路 (residual stream) 里——
> 不破坏直通路径就保证梯度能传到深层;
> 每次写入都是 sublayer 对残差流的一次"贡献".
> Causal mask 通过 `-inf` + softmax 的组合, 确保位置 m 永远只能看 0..m 的历史,
> 让自回归"next token prediction"不作弊.
> 这个模板重复 N 次, 就构成了 Transformer 的全部表达力.**

---

## 附录: 完整数据流 (3 层 Transformer LM)

```
in_indices                    (4, 12)              ← 整数 token id
       │
       ▼  embedding lookup
x                             (4, 12, 64)          ← 连续语义向量
       │
       ▼  TransformerBlock_0  (写入 attn 的贡献 + 写入 ffn 的贡献)
x                             (4, 12, 64)          ← shape 不变
       │
       ▼  TransformerBlock_1
x                             (4, 12, 64)
       │
       ▼  TransformerBlock_2
x                             (4, 12, 64)
       │
       ▼  ln_final (拉回标准尺度)
x                             (4, 12, 64)
       │
       ▼  lm_head (从 64 投到 10000)
logits                        (4, 12, 10000)       ← 每个位置预测下一个 token 的得分
```

每个 TransformerBlock 内部更细的展开见 §6, §7.
