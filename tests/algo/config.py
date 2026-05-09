"""共享的训练配置 (dataclass), 同时给 prepare_data / train / generate 用."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # ---- Tokenizer ----
    vocab_size: int = 10000
    special_tokens: tuple[str, ...] = ("<|endoftext|>",)
    corpus_path: Path = field(default_factory=lambda: _REPO_ROOT / "data/TinyStoriesV2-GPT4-train.txt")

    # ---- Model architecture ----
    d_model: int = 512
    num_layers: int = 8
    num_heads: int = 8
    d_ff: int = 2048
    context_length: int = 512
    rope_theta: float = 10000.0
    eps: float = 1e-5

    # ---- Training ----
    batch_size: int = 64
    max_iters: int = 20000
    lr_max: float = 3e-4
    lr_min: float = 3e-5
    warmup_iters: int = 500
    cosine_cycle_iters: int = 20000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-8

    # ---- Logging / IO ----
    log_every: int = 100
    eval_every: int = 500
    save_every: int = 1000
    valid_split: float = 0.1   # 数据末尾 10% 当验证集
    artifact_dir: Path = field(default_factory=lambda: _REPO_ROOT / "artifacts")

    # ---- Runtime ----
    # 自动检测: MPS (Apple Silicon) > CUDA > CPU
    device: str = (
        "mps" if __import__("torch").backends.mps.is_available()
        else "cuda" if __import__("torch").cuda.is_available()
        else "cpu"
    )
    seed: int = 42

    # ---- 派生路径 (artifact_dir 下的固定布局) ----
    @property
    def vocab_path(self) -> Path:
        return self.artifact_dir / "vocab.pkl"

    @property
    def merges_path(self) -> Path:
        return self.artifact_dir / "merges.pkl"

    @property
    def train_npy(self) -> Path:
        return self.artifact_dir / "train.npy"

    @property
    def valid_npy(self) -> Path:
        return self.artifact_dir / "valid.npy"

    @property
    def ckpt_path(self) -> Path:
        return self.artifact_dir / "checkpoint.pt"
