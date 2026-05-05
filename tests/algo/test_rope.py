import torch
import math

# 配置
d_k = 4
base = 10000.0

# 频率对 i=0 用 θ_0=1, 频率对 i=1 用 θ_1=10000^(-0.5)≈0.01
theta_0 = base ** (-0/d_k * 2)  # = 1.0
theta_1 = base ** (-2/d_k * 2)  # ≈ 0.01

# 一个 4 维向量 (无论是 Q 还是 K, 算法一样)
v = torch.tensor([1.0, 2.0, 3.0, 4.0])

def apply_rope(v, m, theta_0, theta_1):
    """对位置 m 处的 4 维向量应用 RoPE"""
    a0 = m * theta_0  # 第 0 对的旋转角
    a1 = m * theta_1  # 第 1 对的旋转角
    return torch.tensor([
        v[0]*math.cos(a0) - v[1]*math.sin(a0),
        v[0]*math.sin(a0) + v[1]*math.cos(a0),
        v[2]*math.cos(a1) - v[3]*math.sin(a1),
        v[2]*math.sin(a1) + v[3]*math.cos(a1),
    ])

# 让 Q 在位置 m=5, K 在位置 n=2 (相对距离 = 3)
Q_orig = torch.tensor([1.0, 2.0, 3.0, 4.0])
K_orig = torch.tensor([0.5, -1.0, 1.5, 2.0])

Q5 = apply_rope(Q_orig, 5, theta_0, theta_1)
K2 = apply_rope(K_orig, 2, theta_0, theta_1)
print("内积 1:", torch.dot(Q5, K2).item())

# 现在让 Q 在位置 m=10, K 在位置 n=7 (相对距离仍然 = 3)
Q10 = apply_rope(Q_orig, 10, theta_0, theta_1)
K7 = apply_rope(K_orig, 7, theta_0, theta_1)
print("内积 2:", torch.dot(Q10, K7).item())


base = 10000.0
d_k = 64
trained_len = 2048
deploy_len = 8192

# 看看每个频率对在训练 vs 部署时见过的"最大角度"
print(f"{'i':>3} {'theta':>12} {'训练时最大角度(rad)':>22} {'部署时最大角度':>20}")
print("-" * 70)
for i in [0, 5, 10, 20, 30, 31]:
    theta = base ** (-2*i/d_k)
    train_max = trained_len * theta
    deploy_max = deploy_len * theta
    print(f"{i:>3} {theta:>12.6f} {train_max:>22.2f} {deploy_max:>20.2f}")

