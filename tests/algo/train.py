"""
Block 10-11: 训练循环.

流程:
  1. 读 train.npy / valid.npy (np.memmap, 不全部加载到内存).
  2. 构建 TransformerLM + AdamW + cosine LR schedule.
  3. 训练循环: get_batch → forward → cross-entropy → backward → grad-clip → step.
  4. 周期性打印 train loss, 跑 eval, 保存 checkpoint.

跑法: uv run python -m tests.algo.train
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from tests.algo.config import Config
from tests.algo.lm_module import TransformerLM
from tests.algo.training import (
    AdamW,
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    save_checkpoint,
)


@torch.no_grad()
def evaluate(model: TransformerLM, valid: np.ndarray, cfg: Config, num_batches: int = 20) -> float:
    """跑 num_batches 个 valid batch, 返回平均 loss (perplexity 用 exp(loss))."""
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = get_batch(valid, cfg.batch_size, cfg.context_length, cfg.device)
        logits = model(x)
        loss = cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
        )
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main() -> None:
    cfg = Config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ---- 加载数据 ----
    if not cfg.train_npy.exists():
        raise FileNotFoundError(
            f"{cfg.train_npy} not found. Run `uv run python -m tests.algo.prepare_data` first."
        )
    train = np.load(cfg.train_npy, mmap_mode="r")
    valid = np.load(cfg.valid_npy, mmap_mode="r")
    print(f"Loaded train: {len(train):,} tokens, valid: {len(valid):,} tokens")

    # ---- 构建模型 + 优化器 ----
    model = TransformerLM(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        context_length=cfg.context_length,
        rope_theta=cfg.rope_theta,
        eps=cfg.eps,
    ).to(cfg.device)
    print(f"Model parameters: {model.num_parameters():,}")

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr_max,
        betas=cfg.betas,
        eps=cfg.adamw_eps,
        weight_decay=cfg.weight_decay,
    )

    # ---- 训练循环 ----
    print(f"\nTraining for {cfg.max_iters} iterations (batch={cfg.batch_size}, ctx={cfg.context_length})...")
    print(f"{'iter':>6} {'lr':>10} {'train_loss':>11} {'valid_loss':>11} {'tok/s':>9}")
    print("-" * 60)

    t_start = time.time()
    last_log_time = t_start
    last_log_tokens = 0

    for it in range(cfg.max_iters + 1):
        # 1. 调整 learning rate
        lr = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=cfg.lr_max,
            min_learning_rate=cfg.lr_min,
            warmup_iters=cfg.warmup_iters,
            cosine_cycle_iters=cfg.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 2. 取一个 batch
        x, y = get_batch(train, cfg.batch_size, cfg.context_length, cfg.device)

        # 3. 前向 + loss
        logits = model(x)                                    # (B, T, V)
        loss = cross_entropy(
            logits.reshape(-1, logits.shape[-1]),            # (B*T, V)
            y.reshape(-1),                                    # (B*T,)
        )

        # 4. 反向 + 梯度裁剪 + 更新
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # 5. 日志 / eval / checkpoint
        if it % cfg.log_every == 0:
            now = time.time()
            tokens_processed = (it - last_log_tokens) * cfg.batch_size * cfg.context_length
            tok_per_sec = tokens_processed / max(now - last_log_time, 1e-9)
            valid_str = ""
            if it > 0 and it % cfg.eval_every == 0:
                v_loss = evaluate(model, valid, cfg)
                valid_str = f"{v_loss:11.4f}"
            else:
                valid_str = " " * 11
            print(f"{it:>6} {lr:>10.2e} {loss.item():>11.4f} {valid_str} {tok_per_sec:>9.0f}")
            last_log_time = now
            last_log_tokens = it

        if it > 0 and it % cfg.save_every == 0:
            save_checkpoint(model, optimizer, iteration=it, out=cfg.ckpt_path)

    # ---- 最终保存 ----
    save_checkpoint(model, optimizer, iteration=cfg.max_iters, out=cfg.ckpt_path)
    elapsed = time.time() - t_start
    final_valid = evaluate(model, valid, cfg)
    print(
        f"\nDone in {elapsed:.0f}s. Final valid loss = {final_valid:.4f} "
        f"(perplexity = {math.exp(final_valid):.1f})"
    )
    print(f"Checkpoint saved to {cfg.ckpt_path}")
    print("Now run: uv run python -m tests.algo.generate")


if __name__ == "__main__":
    main()
