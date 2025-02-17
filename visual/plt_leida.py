import matplotlib.pyplot as plt
import numpy as np

# 两年的数据示例：每个月份的降雨量
months = ['ProGAN', 'StyleGAN', 'StyleGAN2', 'BigGAN', 'CycleGAN', 'StarGAN', 'GauGAN', 'Deepfake', 'DALLE', 'G100_10', 'G100_27', 'G50_27', 'ADM', 'LDM_100', 'LDM_200', 'LDM_200_cfg']
# FrePGAN = [99.0 , 80.7 , 84.1 , 69.2 , 71.1 , 99.9 , 60.3 , 70.9 , 67.2 , 51.2 , 51.1 , 51.7 , 49.6 , 54.7 , 54.9 , 53.8]
# BiHPF = [90.7 , 76.9 , 76.2 , 84.9 , 81.9 , 94.4 , 69.5 , 54.4 , 52.4 , 58.8 , 59.4 , 64.2 , 58.3 , 53.0 , 52.6 , 51.9]
# 
# 
# High_Freq = [98.9 , 74.4 , 68.8 , 75.2 , 71.0 , 92.7 , 75.5 , 57.0 , 57.0 , 53.6 , 50.4 , 52.0 , 53.4 , 56.6 , 56.4 , 56.5]
#GAN training
# CNNDetection = [91.4 , 63.8 , 76.4 , 52.9 , 72.7 , 63.8 , 63.9 , 51.7 , 51.8 , 53.3 , 53.0 , 54.2 , 54.9 , 51.9 , 52.0 , 51.6]
# Frank = [90.3, 74.5, 73.1, 88.7, 75.5, 99.5, 69.2, 60.7, 57.0, 53.6, 50.4, 52, 53.4, 56.6, 56.4, 56.5]
# Durall = [90.3 , 74.5 , 73.1 , 88.7 , 75.5 , 99.5 , 69.2 , 60.7 , 55.9 , 54.9 , 48.9 , 51.7 , 40.6 , 62.0 , 61.7 , 58.4]
# Patchfor =[81.1 , 54.4 , 66.8 , 60.1 , 69.0 , 98.1 , 61.9 , 50.2 , 79.8 , 87.3 , 82.8 , 84.9 , 74.2 , 95.8 , 95.6 , 94.0]
# F3Net = [99.4 , 92.6 , 88.0 , 65.3 , 76.4 , 100.0, 58.1 , 63.5 , 71.6 , 88.3 , 87.0 , 88.5 , 69.2 , 74.1 , 73.4 , 80.7]
# LGrad = [99.9, 94.8, 96.0, 82.9, 85.3, 99.6, 72.4, 58.0, 88.5, 89.4, 87.4, 90.7, 86.6, 94.8, 94.2, 95.9]
# Ojha = [99.7, 89.0, 83.9, 90.5, 87.9, 91.4, 89.9, 80.2, 89.5, 90.1, 90.7, 91.1, 75.7, 90.5, 90.2, 77.3]
# ours = [99.8, 90.5, 87.7, 96.6, 92.4, 95.6, 95.0, 83.6, 93.1, 90.1, 89.7, 91.4, 78.4, 93.8, 93.5, 81.8]

# # diffusion training
# CNNDetection = [86.5, 81.7, 80.3, 79.6, 59.7, 92.8, 73.9, 52.4, 88.5, 94.2, 94.3, 95.5, 90.1, 93.8, 91.8, 86.3]
# Frank = [53.2, 68.9, 71.1, 73.0, 56.2, 99.5, 59.3, 56.2, 84.3, 84.5, 83.5, 84.1, 72.1, 84.1, 84.2, 83.1]
# Durall = [51.1, 62.5, 52.7, 56.0, 40.0, 35.1, 60.8, 51.0, 55.1, 81.7, 79.4, 80.7, 72.9, 59.8, 60.4, 61.7]
# Patchfor = [66.4, 72.3, 70.8, 66.4, 62.3, 86.1, 55.1, 51.9, 97.2, 97.4, 96.6, 96.9, 66.3, 97.0, 96.7, 96.9]
# F3Net = [60.8, 50.5, 51.6, 57.5, 48.8, 41.3, 50.6, 58.4, 51.3, 79.8, 78.3, 81.6, 56.4, 50.1, 50.8, 54.4]
# LGrad = [66.7, 81.5, 76.8, 60.5, 65.0, 91.2, 54.6, 73.3, 93.2, 96.3, 96.3, 96.4, 78.6, 96.3, 96.3, 96.0]
# Ojha = [92.2, 87.1, 85.5, 84.5, 80.3, 90.2, 87.5, 52.9, 81.5, 90.4, 90.6, 90.6, 86.6, 88.8, 89.8, 77.3]
# ours = [97.5, 93.4, 90.4, 94.8, 92.4, 99.3, 97.6, 64.2, 90.5, 97.2, 97.1, 97.5, 94.3, 97.3, 97.5, 84.9]
#avg
CNNDetection = [88.95, 72.75, 78.35, 66.25, 66.2, 78.3, 68.9, 52.05, 70.15, 73.6, 73.65, 74.85, 72.5, 74.85, 74.9, 73.95]
Frank = [71.75, 71.7, 72.1, 80.85, 65.35, 99.5, 64.25, 58.45, 70.65, 69.05, 66.95, 68.05, 62.7, 70.35, 70.3, 69.8]
Durall = [70.7, 68.75, 62.8, 70.25, 57.875, 51.725, 65.0, 55.85, 55.5, 68.3, 64.15, 66.2, 56.75, 60.9, 61.05, 60.525]
Patchfor = [73.75, 63.35, 68.8, 70.25, 65.65, 92.1, 58.5, 51.05, 88.5, 90.35, 89.7, 90.9, 70.25, 96.4, 96.15, 95.975]
F3Net = [80.1, 71.55, 69.8, 71.925, 62.6, 70.75, 54.35, 61.95, 61.45, 84.05, 82.65, 85.05, 62.8, 62.1, 62.1, 67.975]
LGrad = [83.3, 88.15, 86.4, 71.7, 75.15, 95.4, 63.5, 65.65, 91.2, 93.85, 92.35, 93.55, 80.65, 95.575, 95.5, 95.475]
Ojha = [94.95, 88.05, 84.7, 87.5, 84.3, 90.8, 88.7, 66.55, 85.5, 90.25, 90.65, 90.85, 81.15, 89.65, 90.0, 77.3]
ours = [98.65, 91.95, 89.05, 95.7, 92.4, 97.45, 96.3, 73.9, 91.8, 93.65, 93.4, 94.45, 86.35, 95.55, 95.5, 84.35]


# 找到'Glide100_10'在列表中的索引
index_glide100_10 = months.index('G100_10')

# 将数据循环移位，使'Glide100_10'位于新的列表开头
months = months[index_glide100_10:] + months[:index_glide100_10]
CNNDetection = CNNDetection[index_glide100_10:] + CNNDetection[:index_glide100_10]
Frank = Frank[index_glide100_10:] + Frank[:index_glide100_10]
Durall = Durall[index_glide100_10:] + Durall[:index_glide100_10]
Patchfor = Patchfor[index_glide100_10:] + Patchfor[:index_glide100_10]
F3Net = F3Net[index_glide100_10:] + F3Net[:index_glide100_10]
LGrad = LGrad[index_glide100_10:] + LGrad[:index_glide100_10]
Ojha = Ojha[index_glide100_10:] + Ojha[:index_glide100_10]
ours = ours[index_glide100_10:] + ours[:index_glide100_10]

# 将12个月份分成360度
angles = np.linspace(0, 2*np.pi, len(months), endpoint=False)

# 将雷达图的第一个点重复以形成一个封闭的图形
Ojha += Ojha[:1]
LGrad += LGrad[:1]
Patchfor += Patchfor[:1]
F3Net += F3Net[:1]
Durall += Durall[:1]
Frank += Frank[:1]
CNNDetection += CNNDetection[:1]
ours += ours[:1]
# High_Freq += High_Freq[:1]
# FrePGAN += FrePGAN[:1]
# BiHPF += BiHPF[:1]
angles = np.concatenate((angles, [angles[0]]))

# 创建雷达图
fig, ax = plt.subplots(figsize=(17, 12), subplot_kw=dict(polar=True))

# 绘制两年的降雨数据的折线

ax.plot(angles, Ojha, label='Ojha(CVPR 23\'), 86.6', marker='o', color='lightgreen', linewidth=2, linestyle='dashed')
ax.plot(angles, LGrad, label='LGrad(CVPR 23\'), 82.3', marker='o', color='lightblue',linewidth=2, linestyle='dashed')
ax.plot(angles, Patchfor, label='Patchfor(ECCV 20\'), 76.3', marker='o', color=(1.0, 0.8, 0.6),linewidth=2, linestyle='dashed')
# ax.plot(angles, FrePGAN, label='FrePGAN', marker='o', color=(1.0, 0.8, 0.6))  # Light orange as RGB tuple
# ax.plot(angles, BiHPF, label='BiHPF', marker='o', color=(0.8, 0.6, 1.0))
ax.plot(angles, F3Net, label='F3Net(ECCV 20\'), 69.1', marker='o', color=(0.7, 0.5, 0.3),linewidth=2, linestyle='dashed')
ax.plot(angles, Durall, label='Durall(CVPR 20\'), 62.6', marker='o', color='lightpink',linewidth=2, linestyle='dashed')
ax.plot(angles, Frank, label='Frank(ICML 20\'), 73.9', marker='o', color=(0.5, 0.4, 0.5),linewidth=2, linestyle='dashed')
# ax.plot(angles, High_Freq, label='High_Freq', marker='o', color=(1.0, 0.6, 1.0))
ax.plot(angles, CNNDetection, label='CNNDet(CVPR 20\'), 70.3', marker='o', color='lightgray',linewidth=2, linestyle='dashed')
ax.plot(angles, ours, label='UADNet(ours), 92.7', marker='o', color='red', linewidth=2)
# 设置标签
# ax.set_thetagrids(angles[:-1] * 180/np.pi, months)
# ax.set_thetagrids(angles[:-1] * 180/np.pi, labels=months, ha="center", va="center", rotation=angles[:-1] * 180/np.pi)
ax.set_thetagrids(angles[:-1] * 180/np.pi, labels=months)
for label, angle in zip(ax.get_xticklabels(), angles[:-1] * 180/np.pi):
    label.set_horizontalalignment('center')
    label.set_verticalalignment('center')
    label.set_position((angle, -0.09))  # 调整位置
    label.set_rotation(angle)



ax.set_yticklabels([])  # 隐藏半径刻度标签
ax.tick_params(axis='x', labelsize=17)  # 设置x轴刻度标签大小为12
ax.tick_params(axis='y', labelsize=17)  # 设置y轴刻度标签大小为12

# 设置y轴刻度范围
ax.set_ylim(50, 100)

# 添加图例
# legend = ax.legend(loc='upper right', bbox_to_anchor=(1.12, 1.12), frameon=False)
# 添加图例
legend = ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.12), frameon=False)
for label in legend.get_texts():
    label.set_fontsize(18)  # 设置图例中字体的大小为12

# 添加标题
# plt.title('Monthly Rainfall Radar Chart for Two Years', size=20, y=1.1)

# 显示图形
plt.rc('font', size=18)  # 设置字体大小为14

plt.savefig('figure1_avg_leida.png')
