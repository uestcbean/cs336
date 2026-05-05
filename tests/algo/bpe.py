"""
GPT-2 风格的字节级 BPE (Byte-Pair Encoding) tokenizer
适配 CS336 Assignment 1 的接口
"""

from __future__ import annotations

import os
import regex as re
from collections import Counter
from multiprocessing import Pool
from typing import BinaryIO

# 小于这个字节数走串行——多进程的启动开销 > 并行收益
PARALLEL_THRESHOLD = 1_000_000  # 1 MB

# ---------------------------------------------------------------------------
# GPT-2 的 pretokenization 正则
# - 'll / 've / 're / 's / 'd / 'm / 't  : 英文常见缩略
# -  ?\p{L}+   : 可选前导空格 + 一串字母
# -  ?\p{N}+   : 可选前导空格 + 一串数字
# -  ?[^\s\p{L}\p{N}]+ : 可选前导空格 + 一串"非空白非字母非数字"(主要是标点)
# - \s+(?!\S)  : 行尾空白 (后面没有非空白字符)
# - \s+        : 其他空白
# ---------------------------------------------------------------------------
GPT2_PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    把文件按字节切成若干段, 切点对齐到 `split_special_token` 上.
    实际段数可能少于 desired_num_chunks (边界撞车时去重).

    源自 cs336_basics/pretokenization_example.py, 复制到这避免跨文件 import.
    """
    assert isinstance(split_special_token, bytes)

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def _pretokenize_chunk(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[str],
) -> Counter[tuple[bytes, ...]]:
    """
    Worker 函数: 处理文件中 [start, end) 这段字节范围.

    必须放在模块顶层 (不能是闭包/嵌套函数), 否则 multiprocessing 用 spawn 启动 worker 时
    无法 pickle 这个函数.
    """
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    if special_tokens:
        special_pat = "|".join(re.escape(s) for s in special_tokens)
        docs = re.split(special_pat, chunk)
    else:
        docs = [chunk]

    counts: Counter[tuple[bytes, ...]] = Counter()
    for doc in docs:
        for match in GPT2_PAT.finditer(doc):
            pretoken_bytes = match.group(0).encode("utf-8")
            symbols = tuple(bytes([b]) for b in pretoken_bytes)
            counts[symbols] += 1

    return counts


def pretokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> Counter[tuple[bytes, ...]]:
    """
    读文件 → 按 special_tokens 切段 → 每段做 GPT-2 pretokenization → 统计字节元组频次.

    并行策略:
    - 文件 < PARALLEL_THRESHOLD: 串行 (多进程启动开销 > 收益)
    - 没 special_tokens 可用作切点: 串行 (没法安全切分)
    - 否则: 用 find_chunk_boundaries 按 special token 对齐切, 多进程并行处理
    """
    if num_processes is None:
        num_processes = os.cpu_count() or 1

    file_size = os.path.getsize(input_path)

    # 串行 fallback
    if file_size < PARALLEL_THRESHOLD or num_processes <= 1 or not special_tokens:
        return _pretokenize_chunk(input_path, 0, file_size, special_tokens)

    # 并行: 用第一个 special token 做切点对齐
    split_token = special_tokens[0].encode("utf-8")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token)

    chunk_args = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    # 实际段数可能比 num_processes 少, Pool 自适应
    with Pool(processes=min(num_processes, len(chunk_args))) as pool:
        partial_counts = pool.starmap(_pretokenize_chunk, chunk_args)

    # 合并各 worker 的 Counter
    total: Counter[tuple[bytes, ...]] = Counter()
    for c in partial_counts:
        total.update(c)
    return total


# ===========================================================================
# 阶段 B: 初始 vocab + pair 频次统计 + 应用合并
# ===========================================================================

def build_initial_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """
    初始 vocab:
    - id 0..255 -> 对应的单字节 b'\\x00'..b'\\xff'
    - id 256..  -> special tokens (按传入顺序)

    后面每学到一个 merge, 就追加一个新 id -> 合并后的 bytes
    """
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, tok in enumerate(special_tokens):
        vocab[256 + i] = tok.encode("utf-8")
    return vocab


def get_pair_counts(
    pretoken_counts: Counter[tuple[bytes, ...]],
) -> Counter[tuple[bytes, bytes]]:
    """
    扫描所有 pretoken, 统计每对相邻 token 的加权出现次数.

    例: 如果 (b'h', b'e', b'l', b'l', b'o') 出现 3 次, 那么这一项贡献:
        (b'h', b'e') += 3
        (b'e', b'l') += 3
        (b'l', b'l') += 3
        (b'l', b'o') += 3
    """
    pairs: Counter[tuple[bytes, bytes]] = Counter()
    for symbols, freq in pretoken_counts.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq
    return pairs


def apply_merge(
    pretoken_counts: Counter[tuple[bytes, ...]],
    pair: tuple[bytes, bytes],
) -> Counter[tuple[bytes, ...]]:
    """
    把所有 pretoken 里出现的 `pair` 合并成一个新 token (= pair[0] + pair[1]),
    返回新的 pretoken_counts.

    注意"重叠"问题: 比如 pair=(b'a',b'a'), pretoken=(b'a',b'a',b'a').
    我们一次扫描应该只合并一次, 得到 (b'aa', b'a') 而不是 (b'aaa',).
    用一个 i 指针, 命中 pair 时跳 2 步, 否则跳 1 步, 即可.
    """
    a, b = pair
    merged = a + b
    new_counts: Counter[tuple[bytes, ...]] = Counter()

    for symbols, freq in pretoken_counts.items():
        new_symbols: list[bytes] = []
        i = 0
        n = len(symbols)
        while i < n:
            if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_counts[tuple(new_symbols)] = freq

    return new_counts


# ===========================================================================
# 阶段 C: 训练主循环
# ===========================================================================

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练 GPT-2 风格的字节级 BPE.

    返回:
        vocab:  dict[int, bytes]  - id → token bytes
        merges: list[tuple[bytes, bytes]] - 按学习顺序的合并规则
    """
    # ---- 1. pretokenize: 把语料压成"字节元组 → 频次"的 Counter
    pretoken_counts = pretokenize(input_path, special_tokens)

    # ---- 2. 初始 vocab (256 字节 + special tokens)
    vocab = build_initial_vocab(special_tokens)

    # ---- 3. 算总共要做多少次合并
    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        # 用户给的 vocab_size 太小, 一次合并都不做
        return vocab, []

    merges: list[tuple[bytes, bytes]] = []

    # ---- 4. 训练循环
    for _ in range(num_merges):
        pair_counts = get_pair_counts(pretoken_counts)
        if not pair_counts:
            # 已经无 pair 可合并 (语料里所有 pretoken 都被合成单 token 了)
            break

        # ⚠️ Tie-breaking: 频次最大优先, 频次相同时 pair 字典序最大优先
        best_pair = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

        # 记录这条 merge 规则
        merges.append(best_pair)

        # 把合并后的 token 加进 vocab, 分配下一个可用 id
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        # 更新 pretoken_counts (把所有出现的 best_pair 合并成一个新 token)
        pretoken_counts = apply_merge(pretoken_counts, best_pair)

    return vocab, merges


# ===========================================================================
# Block 3 阶段 A: Tokenizer 类 (encode 部分)
# ===========================================================================

class Tokenizer:
    """
    GPT-2 风格的字节级 BPE Tokenizer.
    给定 vocab + merges, 提供 encode / decode / encode_iterable.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        # 复制 vocab 一份, 防止误改外部传入的
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges: list[tuple[bytes, bytes]] = list(merges)
        self.special_tokens: list[str] = list(special_tokens) if special_tokens else []

        # 防御性: 如果 special token 不在 vocab 里, 自动追加
        existing_values = set(self.vocab.values())
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in existing_values:
                self.vocab[len(self.vocab)] = tok_bytes
                existing_values.add(tok_bytes)

        # bytes → id 的反向映射, encode 时用
        self.inv_vocab: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        # merge_rank: pair → 它在 merges 列表里的位置 (rank 越小 = 越早学到, 越优先合并)
        self.merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }

        # special token 必须按长度降序匹配 (重叠时长的优先, 见 test_overlapping_special_tokens)
        sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
        if sorted_specials:
            self._special_pat = re.compile(
                "|".join(re.escape(s) for s in sorted_specials)
            )
        else:
            self._special_pat = None

    # ----- encode 主入口 ---------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """把文本编码为 token id 列表."""
        if not text:
            return []

        ids: list[int] = []

        if self._special_pat is None:
            # 没 special tokens, 整段直接走普通编码
            ids.extend(self._encode_ordinary(text))
            return ids

        # 有 special tokens: 用 finditer 找出所有 special 位置,
        # 把文本切成 [normal, special, normal, special, ..., normal] 交替结构
        pos = 0
        for m in self._special_pat.finditer(text):
            # m.start() 之前是普通文本
            if m.start() > pos:
                ids.extend(self._encode_ordinary(text[pos:m.start()]))
            # special token 整体作为一个 id
            special_bytes = m.group(0).encode("utf-8")
            ids.append(self.inv_vocab[special_bytes])
            pos = m.end()
        # 末尾的普通文本 (如果有)
        if pos < len(text):
            ids.extend(self._encode_ordinary(text[pos:]))

        return ids

    # ----- 编码不含 special token 的"纯文本" -------------------------------

    def _encode_ordinary(self, text: str) -> list[int]:
        """对一段保证不含 special token 的文本做 GPT-2 pretokenize → BPE merge → 查 id."""
        ids: list[int] = []
        for match in GPT2_PAT.finditer(text):
            pretoken_bytes = match.group(0).encode("utf-8")
            symbols = [bytes([b]) for b in pretoken_bytes]
            symbols = self._apply_bpe_merges(symbols)
            for sym in symbols:
                ids.append(self.inv_vocab[sym])
        return ids

    # ----- 对一个 pretoken 应用 BPE merges --------------------------------

    def _apply_bpe_merges(self, symbols: list[bytes]) -> list[bytes]:
        """
        反复合并: 找当前所有相邻 pair 中 merge_rank 最小的, 合并它. 直到无可合并.
        rank 最小 = 训练时最早学到的 merge, 应该最先应用.
        """
        while len(symbols) >= 2:
            min_rank = None
            min_idx = -1
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.merge_rank.get(pair)
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_rank = rank
                    min_idx = i
            if min_idx == -1:
                break
            # 在 min_idx 处合并相邻两个 token
            symbols = (
                symbols[:min_idx]
                + [symbols[min_idx] + symbols[min_idx + 1]]
                + symbols[min_idx + 2:]
            )
        return symbols

    # ----- decode ---------------------------------------------------------

    def decode(self, ids: list[int]) -> str:
        """
        把 id 列表解码回字符串.
        对未知 id 用空 bytes 跳过 (理论上不该出现, 但防御一下).
        非法 UTF-8 字节用 U+FFFD (replacement char) 替换.
        """
        bytes_seq = b"".join(self.vocab.get(i, b"") for i in ids)
        return bytes_seq.decode("utf-8", errors="replace")

    # ----- encode_iterable (流式) -----------------------------------------

    def encode_iterable(self, iterable):
        """
        流式 encode: 接收一个产出 str 块的可迭代对象 (通常是文件), yield 出 id.

        关键: 不能直接把每个 chunk 独立 encode, 因为:
        - special token 可能跨 chunk (chunk 1 末尾 "<|endof", chunk 2 开头 "text|>")
        - pretoken 可能跨 chunk

        策略: 维护一个 buffer, 只在"安全切点"(找到完整的 special token 之后) 才 flush.
        没 special token 的情况下, 每个 chunk 独立 encode (可能切坏 pretoken,
        但 roundtrip 仍然正确, 因为字节序列不变, decode 还原一样的字符串).
        """
        if self._special_pat is None:
            # 简单路径: 没 special tokens, 直接逐块 encode
            for chunk in iterable:
                yield from self.encode(chunk)
            return

        buffer = ""
        for chunk in iterable:
            buffer += chunk
            # 找 buffer 里最后一个 special token 的结束位置
            last_end = -1
            for m in self._special_pat.finditer(buffer):
                last_end = m.end()
            if last_end > 0:
                # 把 [0, last_end) flush 出去, 剩余 buffer 等下一个 chunk
                yield from self.encode(buffer[:last_end])
                buffer = buffer[last_end:]
        # 收尾: flush 剩余的 buffer
        if buffer:
            yield from self.encode(buffer)


# ---------------------------------------------------------------------------
# 自检: 用作业测试用的 corpus.en, vocab_size=500 跑一次, 看是否能完成
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    from pathlib import Path

    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    input_path = fixtures / "corpus.en"

    t0 = time.time()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    elapsed = time.time() - t0

    print(f"训练耗时: {elapsed:.2f}s (作业要求 < 1.5s, 朴素实现可能更慢, 不影响正确性测试)")
    print(f"最终 vocab 大小: {len(vocab)} (期望 500)")
    print(f"merges 数量: {len(merges)} (期望 {500 - 257} = 243)")
    print(f"\n前 10 条 merge 规则:")
    for i, (a, b) in enumerate(merges[:10]):
        print(f"  {i:3d}  ({a!r:>12}, {b!r:>12}) -> {(a+b)!r}")
    print(f"\n最后 5 条 merge 规则:")
    for i, (a, b) in enumerate(merges[-5:], start=len(merges) - 5):
        print(f"  {i:3d}  ({a!r:>12}, {b!r:>12}) -> {(a+b)!r}")
