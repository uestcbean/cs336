"""
Block 4: Transformer 模型基础组件 (无可学习参数版本, 供 adapters 直接调用).

各函数签名严格对齐 tests/adapters.py 里的 run_xxx.
所有 weights 的 shape 约定都是 (d_out, d_in), 且没有 bias.
"""

from __future__ import annotations

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Linear: y = x @ W^T  (PyTorch 约定, weights shape 是 (d_out, d_in))
# ---------------------------------------------------------------------------
def linear(weights: Tensor, x: Tensor) -> Tensor:
    return x @ weights.T


# ---------------------------------------------------------------------------
# Embedding Lookup: 整数 token id → 连续语义向量
#
# weights: (vocab_size, d_model)   "语义查询表", 每行一个 token 的 d_model 维向量
# token_ids: 任意 shape, int       一批 token id
#
# weights[token_ids] 是 PyTorch 的 fancy indexing:
#   把 token_ids 每个整数当作 weights 的行号, 取出对应行 (d_model 维)替换它.
#   输出 shape = token_ids.shape + (d_model,)
#
# 例: token_ids[0,0]=3797 ("cat") → out[0,0] = weights[3797]
# 这是"符号世界 → 连续向量世界"的入口, 后续所有 Transformer 操作
# (attention/FFN/norm) 都在 d_model 维浮点空间里进行.
# ---------------------------------------------------------------------------
def embedding(weights: Tensor, token_ids: Tensor) -> Tensor:
    return weights[token_ids]


# ---------------------------------------------------------------------------
# Softmax: 沿 dim 轴做 softmax. 减去最大值保证数值稳定 (避免 exp 溢出).
# softmax(x) = exp(x - max) / sum(exp(x - max))
# ---------------------------------------------------------------------------
def softmax(x: Tensor, dim: int) -> Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    e = torch.exp(x - x_max)
    return e / e.sum(dim=dim, keepdim=True)


# ---------------------------------------------------------------------------
# SiLU (Swish): x * sigmoid(x)
# ---------------------------------------------------------------------------
def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


# ---------------------------------------------------------------------------
# RMSNorm: y = x / sqrt(mean(x^2) + eps) * gain
# 关键: 必须在 fp32 计算分母, 否则 fp16/bf16 时 mean(x^2) 会溢出/精度爆炸.
# ---------------------------------------------------------------------------
def rmsnorm(weights: Tensor, x: Tensor, eps: float) -> Tensor:
    in_dtype = x.dtype
    x32 = x.to(torch.float32)
    rms = torch.sqrt((x32 * x32).mean(dim=-1, keepdim=True) + eps)
    out = (x32 / rms) * weights.to(torch.float32)
    return out.to(in_dtype)


# ---------------------------------------------------------------------------
# SwiGLU FFN: out = (SiLU(x W1^T)  ⊙  x W3^T)  W2^T
# weights shape:
#   w1: (d_ff, d_model)
#   w3: (d_ff, d_model)
#   w2: (d_model, d_ff)
# ---------------------------------------------------------------------------
def swiglu(
    w1: Tensor, w2: Tensor, w3: Tensor, x: Tensor
) -> Tensor:
    h1 = linear(w1, x)              # (..., d_ff)
    h3 = linear(w3, x)              # (..., d_ff)
    gated = silu(h1) * h3           # (..., d_ff)
    return linear(w2, gated)        # (..., d_model)


# ---------------------------------------------------------------------------
# RoPE (Rotary Position Embedding)
#
# 思路:
#   把 d_k 维向量两两配对成 d_k/2 个 2D 子向量,
#   第 i 对 (x_{2i}, x_{2i+1}) 在位置 m 处旋转角度 m * θ_i,
#   其中 θ_i = base^(-2i/d_k), base = `theta` 参数 (通常 10000.0).
#
# 旋转公式:
#   x'_{2i}   = x_{2i} * cos(m·θ_i) - x_{2i+1} * sin(m·θ_i)
#   x'_{2i+1} = x_{2i} * sin(m·θ_i) + x_{2i+1} * cos(m·θ_i)
# ---------------------------------------------------------------------------
def rope(
    x: Tensor,                # (..., seq_len, d_k)
    token_positions: Tensor,  # (..., seq_len), int
    d_k: int,
    theta: float,
    max_seq_len: int,         # 仅用于可选的 cos/sin 预缓存, 当前实现按需计算
) -> Tensor:
    half_dim = d_k // 2

    # 1. 算每对维度的频率 θ_i = base^(-2i/d_k), i = 0..d_k/2-1
    #    用 1/theta^(...) 等价但避免负指数
    inv_freq = 1.0 / (
        theta ** (torch.arange(half_dim, dtype=torch.float32, device=x.device) * 2 / d_k)
    )  # (d_k/2,)

    # 2. 算每个 (位置, 频率对) 的角度: angle[..., m, i] = pos[m] * θ_i
    pos = token_positions.to(torch.float32)            # (..., seq_len)
    angles = pos.unsqueeze(-1) * inv_freq              # (..., seq_len, d_k/2)
    cos = angles.cos()                                  # (..., seq_len, d_k/2)
    sin = angles.sin()                                  # (..., seq_len, d_k/2)

    # 3. 把 x 按"交错"方式拆成偶数/奇数维: (x_0,x_2,x_4,...) 和 (x_1,x_3,x_5,...)
    x_even = x[..., 0::2]                              # (..., seq_len, d_k/2)
    x_odd = x[..., 1::2]                               # (..., seq_len, d_k/2)

    # 4. 对每对应用 2D 旋转
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    # 5. 把偶数/奇数维交错地放回原 shape
    out = torch.empty_like(x)
    out[..., 0::2] = rotated_even
    out[..., 1::2] = rotated_odd
    return out


# ---------------------------------------------------------------------------
# Scaled Dot-Product Attention
#   Attention(Q,K,V) = softmax((Q K^T + mask) / sqrt(d_k)) · V
# 支持任意 leading dims (batch, num_heads, ...).
# mask: bool, True=keep, False=mask out (变 -inf).
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    import math
    d_k = Q.shape[-1]
    # K.transpose(-2, -1) 把 (..., n_keys, d_k) 转成 (..., d_k, n_keys)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = softmax(scores, dim=-1)
    return weights @ V


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention 的 head 拆/合工具
# ---------------------------------------------------------------------------
def _split_heads(t: Tensor, num_heads: int) -> Tensor:
    """(..., seq, d_model) → (..., num_heads, seq, d_head)"""
    *leading, seq, d_model = t.shape
    d_head = d_model // num_heads
    t = t.reshape(*leading, seq, num_heads, d_head)
    # 把 num_heads 维提到 seq 前面: ..., seq, head, d → ..., head, seq, d
    return t.transpose(-3, -2)


def _combine_heads(t: Tensor) -> Tensor:
    """(..., num_heads, seq, d_head) → (..., seq, d_model)"""
    # 先 transpose: ..., head, seq, d → ..., seq, head, d
    t = t.transpose(-3, -2).contiguous()
    *leading, seq, num_heads, d_head = t.shape
    return t.reshape(*leading, seq, num_heads * d_head)


def _causal_mask(seq_len: int, device) -> Tensor:
    """生成 (seq_len, seq_len) 的下三角 bool mask. True=keep, False=mask."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention (无 RoPE)
# ---------------------------------------------------------------------------
def multihead_self_attention(
    x: Tensor,                # (..., seq, d_model)
    w_q: Tensor, w_k: Tensor, w_v: Tensor, w_o: Tensor,
    num_heads: int,
) -> Tensor:
    seq = x.shape[-2]

    # 1. 一次大矩阵乘做完所有 head 的 Q/K/V 投影
    Q = linear(w_q, x)                                # (..., seq, d_model)
    K = linear(w_k, x)
    V = linear(w_v, x)

    # 2. 拆头: (..., seq, d_model) → (..., num_heads, seq, d_head)
    Q = _split_heads(Q, num_heads)
    K = _split_heads(K, num_heads)
    V = _split_heads(V, num_heads)

    # 3. 因果 mask (下三角): 第 i 个 query 只能看到 0..i 的 key
    mask = _causal_mask(seq, device=x.device)

    # 4. 每个 head 并行做 SDPA
    out = scaled_dot_product_attention(Q, K, V, mask=mask)
    # out: (..., num_heads, seq, d_head)

    # 5. 合并头: (..., num_heads, seq, d_head) → (..., seq, d_model)
    out = _combine_heads(out)

    # 6. 输出投影
    return linear(w_o, out)


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention with RoPE
# ---------------------------------------------------------------------------
def multihead_self_attention_with_rope(
    x: Tensor,
    w_q: Tensor, w_k: Tensor, w_v: Tensor, w_o: Tensor,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    token_positions: Tensor,   # (..., seq)
) -> Tensor:
    seq = x.shape[-2]
    d_model = x.shape[-1]
    d_head = d_model // num_heads

    Q = linear(w_q, x)
    K = linear(w_k, x)
    V = linear(w_v, x)

    Q = _split_heads(Q, num_heads)         # (..., num_heads, seq, d_head)
    K = _split_heads(K, num_heads)
    V = _split_heads(V, num_heads)

    # 关键: 对 Q, K 的最后一维 (d_head) 应用 RoPE.
    # token_positions shape (..., seq) 会 broadcast 到 (..., num_heads, seq).
    Q = rope(Q, token_positions, d_k=d_head, theta=theta, max_seq_len=max_seq_len)
    K = rope(K, token_positions, d_k=d_head, theta=theta, max_seq_len=max_seq_len)

    mask = _causal_mask(seq, device=x.device)
    out = scaled_dot_product_attention(Q, K, V, mask=mask)
    out = _combine_heads(out)
    return linear(w_o, out)


# ===========================================================================
# Block 7: Transformer Block + Transformer LM
# ===========================================================================

def transformer_block(
    x: Tensor,                # (..., seq, d_model)
    weights: dict[str, Tensor],
    num_heads: int,
    d_ff: int,                # 实际未直接用 (从 weights 推断), 保留接口对齐
    max_seq_len: int,
    theta: float,
    eps: float = 1e-5,
) -> Tensor:
    """
    Pre-norm Transformer block:
        x = x + Attn(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    weights 期望的 key (无 "layers.X." 前缀):
      attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight, attn.output_proj.weight,
      ln1.weight, ffn.w1.weight, ffn.w2.weight, ffn.w3.weight, ln2.weight
    """
    seq = x.shape[-2]
    # 标准位置: 0..seq-1, 用 (1, seq) 形状好和任意 leading dims broadcast
    pos = torch.arange(seq, device=x.device).unsqueeze(0)

    # ---- Sublayer 1: pre-norm + Multi-Head Self-Attn (with RoPE) + residual ----
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

    # ---- Sublayer 2: pre-norm + SwiGLU FFN + residual ----
    h = rmsnorm(weights["ln2.weight"], x, eps)
    h = swiglu(
        w1=weights["ffn.w1.weight"],
        w2=weights["ffn.w2.weight"],
        w3=weights["ffn.w3.weight"],
        x=h,
    )
    x = x + h

    return x


def transformer_lm(
    in_indices: Tensor,            # (..., seq), int
    weights: dict[str, Tensor],
    vocab_size: int,
    context_length: int,           # 用作 RoPE 的 max_seq_len
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    eps: float = 1e-5,
) -> Tensor:
    """
    完整的因果语言模型前向:
        token_embeddings → N×TransformerBlock → ln_final → lm_head

    weights 期望的 key:
      token_embeddings.weight,
      layers.{i}.attn.{q,k,v,output}_proj.weight,
      layers.{i}.ln1.weight, layers.{i}.ln2.weight,
      layers.{i}.ffn.{w1,w2,w3}.weight,
      ln_final.weight, lm_head.weight
    """
    # 1. Token Embedding Lookup —— 把整数 id 翻译成连续语义向量
    #    in_indices              : (..., seq)              整数 token id
    #    token_embeddings.weight : (vocab_size, d_model)   "语义查询表"
    #    weights[in_indices]     : (..., seq, d_model)     每个 id 替换成对应行向量
    #    例: id=3797 ("cat") → 一个 d_model 维向量, 表示 "cat" 的语义坐标
    x = embedding(weights["token_embeddings.weight"], in_indices)

    # 2. 堆叠 N 层 Transformer block —— 每层独立权重, 反复加工 x
    #
    #    每层 shape 不变 (batch, seq, d_model), 但内容越来越抽象:
    #      Layer 0 ~ 处理局部模式 (词性、短语)
    #      Layer 1 ~ 中等距离语法关系 (主谓宾)
    #      Layer N ~ 高层语义 (指代、主题一致性)
    #
    #    单层 attention 已能跨整个 seq (causal mask 内) "看到"所有历史位置;
    #    层数加深 = 让每个位置基于"已被反复加工的高层表示"再做一次 attention,
    #    多层堆叠 + 残差结构 = Transformer 表达力的来源.
    #
    #    每层有独立 weights (layers.0.*, layers.1.*, ...), 不共享.
    for i in range(num_layers):
        prefix = f"layers.{i}."
        # 把 "layers.0.attn.q_proj.weight" 剥成 "attn.q_proj.weight" 喂给 block
        block_weights = {
            k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)
        }
        x = transformer_block(
            x,
            weights=block_weights,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            eps=eps,
        )

    # 3. ln_final —— 最后一道"稳定器" (现代 pre-norm LLM 标配)
    #    Pre-norm 架构里, 残差路径上的 x 一路被 N 层 sublayer 输出累加,
    #    幅度持续增长 (虽然每个 sublayer 入口的 RMSNorm 处理了输入但没处理这条主路).
    #    不归一化直接进 lm_head, 会让 logits 幅度漂移, softmax 输出坍缩成 one-hot.
    #    ln_final 把每个位置的向量重新拉回 RMS≈1 的标准尺度.
    x = rmsnorm(weights["ln_final.weight"], x, eps)

    # 4. lm_head —— 从 d_model 内部表示翻译到 vocab 空间的"分类器"
    #    x:           (..., seq, d_model)
    #    lm_head.weight: (vocab_size, d_model)  每行 = 一个 token 的"模板"
    #    logits:      (..., seq, vocab_size)    每个位置一个 vocab 长度的"得分向量"
    #
    #    logits[b, m, v] = "模型认为位置 m 之后下一个 token 是 v 的得分"
    #      (尚未归一化为概率, softmax 之后才是)
    #
    #    每个位置都输出 logits 是因为 causal mask 保证位置 m 只用到 0..m 的信息,
    #    一次前向就能给所有位置的"下一 token 预测"打分 → 训练时高效, 多任务并行.
    logits = linear(weights["lm_head.weight"], x)
    return logits
