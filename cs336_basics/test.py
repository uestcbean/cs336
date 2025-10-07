import Embedding
import torch
import torch.nn as nn
import numpy as np

# 实例1：基本使用
print("=" * 50)
print("实例1：基本使用")
print("=" * 50)

# 假设词汇表大小为10，嵌入维度为4
vocab_size = 10
d_model = 4
embedding = Embedding.Embedding(vocab_size, d_model)

# 输入token索引 (batch_size=2, seq_len=3)
tokens = torch.LongTensor([[1, 2, 3],
                           [4, 5, 6]])

print(f"输入tokens形状: {tokens.shape}")
print(f"输入tokens:\n{tokens}\n")

# 前向传播
output = embedding(tokens)
print(f"输出形状: {output.shape}")
print(f"输出 (batch_size=2, seq_len=3, d_model=4):\n{output}\n")

# 查看权重矩阵
print(f"权重矩阵形状: {embedding.weights.shape}")
print(f"权重矩阵前3行:\n{embedding.weights[:3]}\n")


# 实例2：带padding的情况
print("=" * 50)
print("实例2：带padding的情况")
print("=" * 50)

# 假设0是padding token
padding_idx = 0
embedding_with_pad = Embedding.Embedding(vocab_size=10, d_model=4, padding_idx=padding_idx)

# 输入包含padding的tokens
tokens_with_pad = torch.LongTensor([[1, 2, 0],   # 最后一个是padding
                                     [3, 0, 0]])  # 后两个是padding

print(f"输入tokens (0表示padding):\n{tokens_with_pad}\n")

output_with_pad = embedding_with_pad(tokens_with_pad)
print(f"输出 (padding位置应该全为0):\n{output_with_pad}\n")

# 验证padding_idx对应的权重确实是0
print(f"权重矩阵中padding_idx={padding_idx}的嵌入向量:")
print(f"{embedding_with_pad.weights[padding_idx]}\n")


# 实例3：展示查表机制
print("=" * 50)
print("实例3：理解查表机制")
print("=" * 50)

# 创建一个小的embedding
small_embed = Embedding.Embedding(vocab_size=5, d_model=3)

print("权重矩阵 (每行代表一个token的嵌入):")
for i in range(5):
    print(f"Token {i}: {small_embed.weights[i].data}")

print("\n查询token 2的嵌入:")
token_2 = torch.LongTensor([[2]])
result = small_embed(token_2)
print(f"结果: {result.squeeze()}")
print(f"应该等于权重矩阵的第2行: {small_embed.weights[2].data}")