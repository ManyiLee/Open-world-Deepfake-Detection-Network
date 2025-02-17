import matplotlib.pyplot as plt
import numpy as np

# 月份和数据
months = ['ProGAN', 'StyleGAN', 'StyleGAN2', 'BigGAN', 'CycleGAN', 'StarGAN', 'GauGAN', 'Deepfake', 'DALLE', 'Glide100_10', 'Glide100_27', 'Glide50_27', 'ADM', 'LDM_100', 'LDM_200', 'LDM_200_cfg']
ours = [98.65, 91.95, 89.05, 95.7, 92.4, 97.45, 96.3, 73.9, 91.8, 93.65, 93.4, 94.45, 86.35, 95.55, 95.5, 84.35]
angles = np.linspace(0, 2*np.pi, len(months), endpoint=False)

fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

# 画线
ax.plot(angles, ours + ours[:1], label='UADNet(ours)', marker='o', color='red', linewidth=2)

# 添加标签
for angle, label in zip(angles, months):
    x = np.deg2rad(angle)
    y = max(ours) + 5  # 调整标签距离圆心的距离
    ax.text(x, y, label, rotation=angle, ha='center', va='center')

# 设置图形
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_rlabel_position(180)  # 调整标签的位置

# 其他设置
ax.set_yticklabels([])
ax.set_ylim(50, 100)
ax.legend(loc='upper right', bbox_to_anchor=(1.12, 1.12), frameon=False)

plt.savefig('figure1_avg_leida.png')
