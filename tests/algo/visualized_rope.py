import math
import matplotlib.pyplot as plt

d_k = 64
base = 10000.0
positions = list(range(0, 200))

fig, ax = plt.subplots(figsize=(10, 5))

for i in [0, 5, 15, 25, 31]:
    theta = base ** (-2*i/d_k)
    angles = [m * theta for m in positions]
    cos_vals = [math.cos(a) for a in angles]
    ax.plot(positions, cos_vals, label=f"i={i}, θ={theta:.4f}")

ax.set_xlabel("token position m")
ax.set_ylabel("cos(m·θ_i)")
ax.set_title("RoPE 不同频率对的 cos 值")
ax.legend()
ax.grid()
plt.show()
