"""
TransformerLM 的 nn.Module 包装器.

为什么需要这层包装?
  - 我们之前的 transformer_lm(in_indices, weights, ...) 是纯函数, 接收 dict.
  - 训练时需要 PyTorch 自动管理参数、算梯度、保存 state_dict.
  - 解决: 用 nn.Module 持有 nn.Parameter, forward 时把它们组装成 dict
    喂给已有的 transformer_lm 函数. 函数本身完全不用改.

权重命名严格对齐已有测试约定:
  token_embeddings.weight
  layers.{i}.attn.{q,k,v,output}_proj.weight
  layers.{i}.ln1.weight, layers.{i}.ln2.weight
  layers.{i}.ffn.{w1,w2,w3}.weight
  ln_final.weight, lm_head.weight
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tests.algo.model import transformer_lm


class _RMSNormParam(nn.Module):
    """只持有 weight 参数的 RMSNorm 占位 module, 用来对齐 ln1.weight 这种命名."""
    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))


class _LinearParam(nn.Module):
    """只持有 weight 参数, 命名为 'weight' 的占位 module (vocab_size, d_model 等矩阵)."""
    def __init__(self, out_features: int, in_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))


class _AttnBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q_proj = _LinearParam(d_model, d_model)
        self.k_proj = _LinearParam(d_model, d_model)
        self.v_proj = _LinearParam(d_model, d_model)
        self.output_proj = _LinearParam(d_model, d_model)


class _FFNBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = _LinearParam(d_ff, d_model)
        self.w2 = _LinearParam(d_model, d_ff)
        self.w3 = _LinearParam(d_ff, d_model)


class _TransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.attn = _AttnBlock(d_model)
        self.ln1 = _RMSNormParam(d_model)
        self.ffn = _FFNBlock(d_model, d_ff)
        self.ln2 = _RMSNormParam(d_model)


class TransformerLM(nn.Module):
    """完整的因果语言模型, 可训练 (autograd 跟踪所有 nn.Parameter)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        context_length: int,
        rope_theta: float = 10000.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.context_length = context_length
        self.rope_theta = rope_theta
        self.eps = eps

        self.token_embeddings = _LinearParam(vocab_size, d_model)  # 实际是 (vocab, d_model) 表
        self.layers = nn.ModuleList([
            _TransformerLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.ln_final = _RMSNormParam(d_model)
        self.lm_head = _LinearParam(vocab_size, d_model)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """简化版 GPT-2 初始化: Linear 层 N(0, 0.02), RMSNorm gain = 1."""
        std = 0.02
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=std)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)
        # RMSNorm 的 weight 已经在 _RMSNormParam.__init__ 里初始化为 1
        for layer in self.layers:
            for w in (layer.attn.q_proj.weight, layer.attn.k_proj.weight,
                      layer.attn.v_proj.weight, layer.attn.output_proj.weight,
                      layer.ffn.w1.weight, layer.ffn.w2.weight, layer.ffn.w3.weight):
                nn.init.normal_(w, mean=0.0, std=std)

    # ------------------------------------------------------------------
    def _build_weights_dict(self) -> dict[str, Tensor]:
        """
        组装 transformer_lm 函数期望的权重 dict.
        关键: 直接传 nn.Parameter 引用, 不要 detach/clone, 否则 autograd 会断.
        """
        weights = {
            "token_embeddings.weight": self.token_embeddings.weight,
            "ln_final.weight": self.ln_final.weight,
            "lm_head.weight": self.lm_head.weight,
        }
        for i, layer in enumerate(self.layers):
            p = f"layers.{i}."
            weights[p + "attn.q_proj.weight"] = layer.attn.q_proj.weight
            weights[p + "attn.k_proj.weight"] = layer.attn.k_proj.weight
            weights[p + "attn.v_proj.weight"] = layer.attn.v_proj.weight
            weights[p + "attn.output_proj.weight"] = layer.attn.output_proj.weight
            weights[p + "ln1.weight"] = layer.ln1.weight
            weights[p + "ln2.weight"] = layer.ln2.weight
            weights[p + "ffn.w1.weight"] = layer.ffn.w1.weight
            weights[p + "ffn.w2.weight"] = layer.ffn.w2.weight
            weights[p + "ffn.w3.weight"] = layer.ffn.w3.weight
        return weights

    # ------------------------------------------------------------------
    def forward(self, in_indices: Tensor) -> Tensor:
        """
        in_indices: (batch, seq) int
        return:     (batch, seq, vocab_size) logits
        """
        return transformer_lm(
            in_indices=in_indices,
            weights=self._build_weights_dict(),
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            rope_theta=self.rope_theta,
            eps=self.eps,
        )

    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
