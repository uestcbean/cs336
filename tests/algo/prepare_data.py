"""
Block 9: 准备训练数据.

流程:
  1. 在 corpus_path 上训练 BPE (vocab_size 在 Config 里).
  2. 把 vocab + merges 序列化到 artifact_dir/.
  3. 用训好的 tokenizer 编码整段语料 → 一串 token id.
  4. 按 valid_split 切成训练 / 验证两部分, 各自存为 .npy (uint16, vocab < 65k).

跑法: uv run python -m tests.algo.prepare_data
"""

from __future__ import annotations

import pickle
import time

import numpy as np

from tests.algo.bpe import Tokenizer, train_bpe
from tests.algo.config import Config


def main() -> None:
    cfg = Config()
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 训 BPE ----
    print(f"[1/4] Training BPE (vocab={cfg.vocab_size}) on {cfg.corpus_path.name}...")
    t0 = time.time()
    vocab, merges = train_bpe(
        input_path=cfg.corpus_path,
        vocab_size=cfg.vocab_size,
        special_tokens=list(cfg.special_tokens),
    )
    print(f"      done in {time.time() - t0:.1f}s, {len(merges)} merges learned.")

    # ---- 2. 保存 tokenizer 工件 ----
    with open(cfg.vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(cfg.merges_path, "wb") as f:
        pickle.dump(merges, f)
    print(f"[2/4] Saved vocab → {cfg.vocab_path.name}, merges → {cfg.merges_path.name}")

    # ---- 3. 编码全部语料 ----
    print(f"[3/4] Encoding corpus...")
    tokenizer = Tokenizer(vocab, merges, special_tokens=list(cfg.special_tokens))
    with open(cfg.corpus_path, encoding="utf-8") as f:
        text = f.read()
    t0 = time.time()
    ids = tokenizer.encode(text)
    print(f"      encoded {len(ids):,} tokens in {time.time() - t0:.1f}s "
          f"({len(text) / len(ids):.2f} chars/token)")

    # vocab_size < 65536 用 uint16 节省一半内存
    arr = np.asarray(ids, dtype=np.uint16 if cfg.vocab_size < 2**16 else np.int32)

    # ---- 4. 切分 + 保存 ----
    split = int(len(arr) * (1 - cfg.valid_split))
    np.save(cfg.train_npy, arr[:split])
    np.save(cfg.valid_npy, arr[split:])
    print(f"[4/4] Train: {split:,} tokens → {cfg.train_npy.name}")
    print(f"      Valid: {len(arr) - split:,} tokens → {cfg.valid_npy.name}")
    print("\nDone. Now run: uv run python -m tests.algo.train")


if __name__ == "__main__":
    main()
