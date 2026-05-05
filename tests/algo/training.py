"""
Block 8: 训练循环里的零件.

包含:
  - cross_entropy:  loss 函数
  - gradient_clipping: 限制梯度全局 L2 范数
  - get_batch: 从 token 数组里随机采样输入/目标对
  - get_lr_cosine_schedule: 学习率三段式调度
  - save_checkpoint / load_checkpoint
  - AdamW 优化器 (从零实现)
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import IO, BinaryIO

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


# ===========================================================================
# Cross Entropy Loss
#
# 公式: CE(logits, target) = -log(softmax(logits)[target])
# 用 log-sum-exp 写法避免数值溢出:
#   -log(exp(logit_target) / sum(exp(logits)))
#   = -logit_target + log(sum(exp(logits)))
#   = -logit_target + logsumexp(logits)
# 进一步用 max-subtraction 让 logsumexp 稳定:
#   logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
# ===========================================================================
def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """
    logits:  (batch, vocab_size)  - 未归一化的得分
    targets: (batch,)              - 每个样本的正确类别 id
    返回:    标量, 平均 cross-entropy loss
    """
    # 数值稳定: 减最大值
    x_max = logits.max(dim=-1, keepdim=True).values
    x_shifted = logits - x_max
    log_sum_exp = torch.log(torch.exp(x_shifted).sum(dim=-1)) + x_max.squeeze(-1)
    # gather 出目标位置的 logit
    logit_target = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    loss_per_sample = log_sum_exp - logit_target
    return loss_per_sample.mean()


# ===========================================================================
# Gradient Clipping (全局 L2 范数)
#
# 算所有参数梯度的全局 L2 范数 ||g||.
# 如果 ||g|| > max_norm, 把所有梯度缩小为原来的 max_norm / ||g|| 倍.
# 否则不动.
# ===========================================================================
def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    """In-place 修改 parameter.grad."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    # 全局范数 = sqrt(Σ ||g_i||²)
    total_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))

    clip_coef = max_l2_norm / (total_norm + eps)
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)


# ===========================================================================
# Get Batch
#
# 从 1D token 数组里随机采样: 每个样本是连续的 context_length 个 token.
# input  = tokens[i : i + context_length]
# target = tokens[i+1 : i + context_length + 1]   (input 错位 1)
# ===========================================================================
def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    n = len(dataset)
    # 合法起点范围: [0, n - context_length - 1] 才能保证 target 不越界
    # 因为 target 取 i+1 .. i+context_length, 最大需要的索引是 i+context_length, 需 ≤ n-1
    max_start = n - context_length
    starts = np.random.randint(0, max_start, size=batch_size)

    # 用 numpy stack 取片段, 一次性构造所有样本
    x = np.stack([dataset[s : s + context_length] for s in starts])
    y = np.stack([dataset[s + 1 : s + 1 + context_length] for s in starts])

    return (
        torch.from_numpy(x).to(dtype=torch.long, device=device),
        torch.from_numpy(y).to(dtype=torch.long, device=device),
    )


# ===========================================================================
# Cosine LR Schedule (with warmup)
#
# 三段:
#   1. 0 <= it < warmup_iters:        线性从 0 升到 max_lr
#   2. warmup_iters <= it <= cosine_cycle_iters: cosine 从 max_lr 降到 min_lr
#   3. it > cosine_cycle_iters:       常数 min_lr
# ===========================================================================
def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        # Warmup: 线性
        return max_learning_rate * it / warmup_iters
    if it <= cosine_cycle_iters:
        # Cosine decay: progress 0..1
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        # 0.5 * (1 + cos(π·progress)) 从 1 平滑降到 0
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine
    # 之后保持 min_lr
    return min_learning_rate


# ===========================================================================
# Checkpoint Save / Load
# ===========================================================================
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | "os.PathLike" | BinaryIO | IO[bytes],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | "os.PathLike" | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    ckpt = torch.load(src, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["iteration"]


# ===========================================================================
# AdamW Optimizer (从零实现, 跟 PyTorch 的 torch.optim.AdamW 等价)
#
# 算法 (每个参数 θ 独立维护 m, v):
#   m_t = β1 · m_{t-1} + (1 - β1) · g_t              (一阶矩 = 动量平滑梯度)
#   v_t = β2 · v_{t-1} + (1 - β2) · g_t²             (二阶矩 = 平滑梯度方差)
#   bias_correction1 = 1 - β1^t
#   bias_correction2 = 1 - β2^t
#   step_size = lr / bias_correction1
#   denom = sqrt(v_t / bias_correction2) + ε
#
#   θ ← θ · (1 - lr · weight_decay)                  (decoupled weight decay)
#   θ ← θ - step_size · m_t / denom                  (Adam 主更新)
# ===========================================================================
class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                # 第一次 step 时初始化状态
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)        # m
                    state["exp_avg_sq"] = torch.zeros_like(p)     # v

                state["step"] += 1
                m, v = state["exp_avg"], state["exp_avg_sq"]
                t = state["step"]

                # 更新一阶矩 m: m = β1·m + (1-β1)·g
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # 更新二阶矩 v: v = β2·v + (1-β2)·g²
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                # 1. Decoupled weight decay: θ *= (1 - lr·wd)
                p.mul_(1 - lr * wd)

                # 2. Adam 主更新: θ -= step_size · m / (sqrt(v)/sqrt(bc2) + eps)
                denom = (v.sqrt() / bias_correction2_sqrt).add_(eps)
                p.addcdiv_(m, denom, value=-step_size)

        return loss
