"""
Block 12: 用训好的 LM 生成文本.

实现 3 种采样:
  - greedy:     argmax (deterministic, 经常退化为重复)
  - temperature:  按 logits / T 后 softmax 采样 (T 大 → 多样, T 小 → 接近 greedy)
  - top-k:        只保留 top-k 个最高 logit, 其余设 -inf 后采样

跑法:
  uv run python -m tests.algo.generate
  uv run python -m tests.algo.generate "Once upon a time"
"""

from __future__ import annotations

import pickle
import sys

import torch

from tests.algo.bpe import Tokenizer
from tests.algo.config import Config
from tests.algo.lm_module import TransformerLM
from tests.algo.training import load_checkpoint, AdamW


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int | None = 50,
    eos_token: str | None = "<|endoftext|>",
    device: str = "cpu",
) -> str:
    """
    给一段 prompt, 续写最多 max_new_tokens 个 token.

    采样策略:
      - temperature == 0: argmax (greedy)
      - temperature > 0:  按 logits/T softmax 采样
      - top_k != None:    采样前先把非 top-k 位置设 -inf
    """
    model.eval()

    eos_id = None
    if eos_token is not None:
        # 假设 special token 在 vocab 里, 单独 encode 找 id
        eos_ids = tokenizer.encode(eos_token)
        eos_id = eos_ids[0] if len(eos_ids) == 1 else None

    prompt_ids = tokenizer.encode(prompt)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # (1, T)

    for _ in range(max_new_tokens):
        # context 截断 (防止超过模型 ctx)
        ctx_ids = ids[:, -model.context_length:]
        logits = model(ctx_ids)              # (1, T, V)
        next_logits = logits[0, -1]          # (V,)

        if temperature == 0:
            next_id = int(next_logits.argmax())
        else:
            scaled = next_logits / temperature
            if top_k is not None and top_k > 0:
                topk_vals, _ = torch.topk(scaled, k=min(top_k, scaled.size(-1)))
                threshold = topk_vals[-1]
                scaled = torch.where(scaled < threshold, torch.full_like(scaled, float("-inf")), scaled)
            probs = torch.softmax(scaled, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

        if eos_id is not None and next_id == eos_id:
            break
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

    return tokenizer.decode(ids[0].tolist())


def main() -> None:
    cfg = Config()

    # ---- 加载 tokenizer ----
    if not cfg.vocab_path.exists():
        raise FileNotFoundError(
            f"{cfg.vocab_path} not found. Run `uv run python -m tests.algo.prepare_data` first."
        )
    with open(cfg.vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(cfg.merges_path, "rb") as f:
        merges = pickle.load(f)
    tokenizer = Tokenizer(vocab, merges, special_tokens=list(cfg.special_tokens))

    # ---- 加载模型 ----
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

    if cfg.ckpt_path.exists():
        # load_checkpoint 也会读 optimizer state, 但我们不需要; 给它一个临时 optimizer
        tmp_opt = AdamW(model.parameters(), lr=1e-4)
        iters = load_checkpoint(cfg.ckpt_path, model, tmp_opt)
        print(f"Loaded checkpoint @ iter {iters} from {cfg.ckpt_path.name}")
    else:
        print(f"WARNING: no checkpoint at {cfg.ckpt_path}, using random weights "
              f"(output will be gibberish)")

    # ---- 生成 ----
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
    print(f"\nPrompt: {prompt!r}")
    print("-" * 60)

    for label, kwargs in [
        ("greedy",       dict(temperature=0)),
        ("T=0.8 top-k=50", dict(temperature=0.8, top_k=50)),
        ("T=1.0 top-k=20", dict(temperature=1.0, top_k=20)),
    ]:
        print(f"\n[{label}]")
        text = generate(model, tokenizer, prompt, max_new_tokens=120, device=cfg.device, **kwargs)
        print(text)


if __name__ == "__main__":
    main()
