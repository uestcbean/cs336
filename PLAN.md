# CS336 Assignment 1 — 2 天速通计划

> 目标：2 天内完成作业，并且**彻底理解每个组件**（不是死记硬背抄答案）。

---

## 总体策略

**理解锚点是 PDF 的具体小节** — 每写一段代码前，先读对应小节 + 推导一遍公式。如果某段公式推不出来，就停下来问 / 查资料，**不要硬写代码**。这是这份计划能保证"理解"的关键。

**测试驱动**：每实现一个函数就跑对应测试，绿了再走下一个。

```sh
uv run pytest tests/test_xxx.py::test_yyy -x -v
```

**两个心态**：
- 跳过 Leaderboard 的所有事 —— 那是 1 周作业，不是 2 天
- TinyStories 上做 1 次小训练（验证能跑 + 生成几句故事）就够了，不追求好效果

---

## Day 1：Tokenizer + 模型零件（约 10–11 小时）

### 🌅 上午：BPE Tokenizer（4–5 小时）—— 全作业最难的部分

#### Block 1（30 分钟）：搭环境 + 看 PDF
- [ ] `uv run pytest -x` 跑一下，看到 NotImplementedError 就对了
- [ ] 下载数据（README 里的 wget 命令）
- [ ] **必读**：PDF 第 2 节（BPE Tokenization）整节，特别是：
  - "naive BPE" 算法步骤
  - pretokenization 的 GPT-2 正则表达式
  - special tokens 处理
  - 合并的 tie-breaking 规则（取字典序更大的 pair）

#### Block 2（2.5 小时）：`run_train_bpe`
**理解先行**：在纸上手算一个 5 词小语料的 BPE 训练过程，至少做 3 次合并。**做不出来就别开始写**。

实现要点（在 `cs336_basics/bpe.py` 新建）：
1. 用 special token 做 split（先切再 pretokenize，避免合并跨边界）
2. 用 GPT-2 正则做 pretokenization（`cs336_basics/pretokenization_example.py` 已有例子）
3. 把每个 pretoken 表示成 bytes 序列
4. 反复找最高频的相邻 pair，合并，更新计数
5. tie-breaking 用字典序大的 pair（注意是 bytes 比较）

跑通：`uv run pytest tests/test_train_bpe.py -x -v`

**理解检查**：能口头解释为什么要先 pretokenize 而不是直接对原文 BPE 吗？(提示：效率 + 防止跨词合并)

#### Block 3（1.5 小时）：`get_tokenizer`（encode/decode）
实现一个 `Tokenizer` 类（PDF 里有 API 规范）：
- `encode(text) -> list[int]`：split special tokens → pretokenize → 应用 merges
- `decode(ids) -> str`：拼 bytes → UTF-8 解码（注意非法 bytes 的处理）
- `encode_iterable`：流式处理大文件

跑通：`uv run pytest tests/test_tokenizer.py -x -v`

**理解检查**：encode 时合并的顺序怎么决定？(按 merges 列表顺序贪心，**不是**按当前 pair 频率)

---

### ☀️ 下午：模型积木（4–5 小时）

#### Block 4（1 小时）：基础层
按顺序实现，每个 5–15 分钟：
- `run_linear` — 注意权重 shape 是 `(d_out, d_in)`，没有 bias
- `run_embedding` — 就是查表
- `run_softmax` — 用 max-subtraction 数值稳定
- `run_silu` — `x * sigmoid(x)`
- `run_rmsnorm` — **在 fp32 下计算**（PDF 提示），最后 cast 回原 dtype
- `run_swiglu` — `W2(SiLU(W1·x) ⊙ W3·x)`

**全部写在 `cs336_basics/model.py`**，建议都做成 `nn.Module` 子类，再在 adapters 里实例化加载权重。

跑通对应测试。

#### Block 5（2 小时）：RoPE —— 最容易写错
**理解先行**：在纸上推一遍二维旋转矩阵，理解为什么 RoPE 等价于把 `(x_{2i}, x_{2i+1})` 当成一个复数乘上 `e^{i·θ_i·pos}`。

实现 `run_rope`：
- 预计算 `cos` / `sin` 缓存（形状 `(max_seq_len, d_k/2)`）
- 用 `token_positions` 做 gather
- 注意 PDF 1.0.3 修过 off-by-one，仔细看公式

跑通：`uv run pytest tests/test_model.py -k rope -x -v`

**理解检查**：为什么 RoPE 只作用在 Q、K，不作用在 V？

#### Block 6（1.5–2 小时）：Attention（含 multihead）
- `run_scaled_dot_product_attention`：`softmax(QK^T/√d_k + mask) · V`，mask 用 `-inf` 填
- `run_multihead_self_attention`：reshape 成 `(batch, num_heads, seq, d_head)` 一次算完
- `run_multihead_self_attention_with_rope`：在 reshape 后、attention 前对 Q/K 应用 RoPE

跑通所有 attention 相关测试。

**理解检查**：因果 mask（causal mask）长什么样？为什么要用 `-inf` 而不是 0？

---

## Day 2：组装 + 训练 + 真正跑起来（约 9–10 小时）

### 🌅 上午：完整 Transformer + 训练工具（4–5 小时）

#### Block 7（1.5 小时）：Transformer Block + LM
- `run_transformer_block`：pre-norm 结构 → `x + Attn(RMSNorm(x))`，再 `x + FFN(RMSNorm(x))`
- `run_transformer_lm`：embedding → N 层 block → 最后一个 RMSNorm → lm_head（线性投到 vocab）

跑通：`uv run pytest tests/test_model.py -x -v`（**全绿**）

**理解检查**：pre-norm vs post-norm 区别？(pre-norm 训练更稳定，现代 LLM 都用)

#### Block 8（2 小时）：训练数学
按顺序：
- `run_cross_entropy` — **从 logits 直接算**，用 log-sum-exp 数值稳定，不要先 softmax 再 log
- `run_gradient_clipping` — 算所有参数梯度的全局 L2 范数，超了就缩
- `get_adamw_cls` — **重头戏**，自己继承 `torch.optim.Optimizer` 实现。在纸上推一遍 m、v、bias correction、weight decay 的更新公式
- `run_get_lr_cosine_schedule` — 三段：linear warmup / cosine decay / 常数 min_lr
- `run_get_batch` — 从 1D token array 里随机采样起点，input 和 target 错位 1
- `run_save_checkpoint` / `run_load_checkpoint` — `torch.save({...})` 三件套

跑通 test_nn_utils / test_optimizer / test_data / test_serialization。

**理解检查**：AdamW 和 Adam 的差别究竟在哪？(weight decay 不再混进梯度，而是直接 `θ -= lr·wd·θ`)

#### Block 9（1 小时）：用你的 tokenizer 编码数据集
写个小脚本 `cs336_basics/encode_dataset.py`：
1. 加载训好的 BPE
2. 流式 encode TinyStories train/valid 文件
3. 存成 `.npy` 给训练用

如果嫌训 BPE 慢，可以先用 `vocab_size=10000` 跑 BPE。

---

### ☀️ 下午：训练循环 + 真的跑一次（4–5 小时）

#### Block 10（2 小时）：写 train.py
在 `cs336_basics/train.py` 把所有零件接起来：

```
配置 → 加载数据 (np.memmap) → 构建模型 → AdamW → 训练循环：
  for it in range(max_iters):
    x, y = get_batch(...)
    logits = model(x)
    loss = cross_entropy(logits, y)
    loss.backward()
    grad_clip()
    optimizer.step()
    update_lr()
    每 N 步：valid loss + 打印 + checkpoint
```

#### Block 11（1.5 小时）：小规模训练 TinyStories
**目标不是好效果，是验证管道通**：
- 模型小一点：`d_model=128, n_layers=2, n_heads=4, d_ff=512, ctx=128`
- 跑 ~500–1000 步，loss 应该从 ~10 降到 5 以下
- **没 GPU 就用 CPU 跑 100 步**，loss 下降即可

#### Block 12（1 小时）：生成 + 收尾理解
写个 generate 函数：argmax / top-k / temperature sampling。喂个 "Once upon a time" 让模型续写。

**最后 30 分钟做"理解 review"**：
- 关上 IDE，能口头复述：从一个原始字符串到模型输出 logits，每一步发生了什么？
  (BPE → input ids → embedding → +RoPE in attn → N×(attn+FFN with RMSNorm) → ln_final → lm_head)
- 不能流畅复述的环节，回去重读对应 PDF 小节。

---

## 底线建议

1. **第一天结束时如果 BPE 没搞定，DON'T PANIC** — BPE 是真的难。可以先跳过 `run_train_bpe`，用别人提供的 vocab/merges 跑 tokenizer 测试，第二天有空再补。
2. **不要陷进数值精度地狱** — 测试通常 atol=1e-4 ~ 1e-3 都给得很宽，别为了 1e-6 浪费 1 小时。
3. **每个 Block 卡超过 30 分钟就停下来求助** — debug / 概念解释比硬磕高效。
4. **PDF 的 "Problem X.Y" 框框是你的圣经** — 比 README 重要 100 倍，每个函数的精确规范都在那里。

---

## 进度跟踪

### Day 1
- [ ] Block 1：环境 + PDF 第 2 节
- [ ] Block 2：`run_train_bpe`
- [ ] Block 3：Tokenizer encode/decode
- [ ] Block 4：Linear / Embedding / Softmax / SiLU / RMSNorm / SwiGLU
- [ ] Block 5：RoPE
- [ ] Block 6：Attention（SDPA + MHA + MHA with RoPE）

### Day 2
- [ ] Block 7：TransformerBlock + TransformerLM
- [ ] Block 8：CrossEntropy / GradClip / AdamW / Cosine / GetBatch / Checkpoint
- [ ] Block 9：编码数据集
- [ ] Block 10：train.py
- [ ] Block 11：训练 TinyStories
- [ ] Block 12：生成 + 理解 review
