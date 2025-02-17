import matplotlib.pyplot as plt
import numpy as np

datasets = ['AttGAN', 'BEGAN', 'CramerGAN', 'InfoMaxGAN', 'MMDGAN', 'RelGAN', 'S3GAN', 'SNGGAN', 'STGGAN', 'ProGAN', 'StyleGAN', 'StyleGAN2', 'BigGAN', 'CycleGAN', 'StarGAN', 'GauGAN', 'Deepfake']

#2-class quality-agnostic
#MesoNet = [60.0, 44.3, 59.7, 46.3, 59.8, 58.7, 47.8, 56.3, 69.1, 55.0, 51.0, 49.9, 53.9, 60.5, 64.8, 49.9, 51.1]
#FF = [56.6, 37.7, 79.4, 66.9, 77.1, 60.5, 55.0, 69.2, 79.8, 87.8, 55.1, 59.8, 57.1, 79.9, 75.6, 71.6, 52.0]
#F3Net = [51.5, 48.8, 61.9, 58.0, 59.3, 53.2, 52.0, 54.9, 61.1, 83.4, 52.7, 54.9, 55.0, 73.7, 65.9, 66.7, 52.4]
#MAT = [50.5, 49.9, 59.6, 54.2, 57.6, 51.2, 52.1, 52.7, 57.8, 86.1, 52.3, 53.0, 52.7, 70.3, 58.2, 68.0, 51.3]
#SBIs = [50.1, 51.9, 63.4, 56.6, 59.3, 50.6, 62.2, 52.1, 53.0, 88.6, 51.3, 52.4, 55.7, 76.0, 53.9, 78.1, 51.2]
#ADD = [50.7, 50.9, 59.0, 51.8, 57.1, 52.8, 45.0, 52.3, 52.9, 70.2, 48.0, 48.7, 51.8, 71.9, 55.5, 65.1, 51.3]
#QAD = [61.5, 55.2, 80.0, 72.3, 78.3, 65.5, 54.5, 76.5, 79.2, 86.4, 56.4, 58.0, 57.4, 82.6, 77.8, 63.5, 56.5]
#ours = [68.1, 44.1, 76.8, 72.1, 76.5, 73.3, 58.0, 75.6, 83.5, 90.8, 61.1, 65.9, 63.9, 83.5, 77.0, 72.9, 55.0]

#2-class quality-agnostic
MesoNet = [62.9, 45.4, 63.5, 58.7, 62.0, 50.2, 48.7, 58.4, 64.1, 55.4, 52.0, 48.1, 53.7, 63.2, 62.0, 49.6, 51.8]
FF = [63.3, 29.9, 82.0, 68.9, 80.4, 67.2, 55.5, 75.4, 82.0, 93.0, 61.1, 59.8, 57.9, 80.1, 78.6, 67.3, 51.9]
F3Net = [53.2, 43.4, 65.8, 62.0, 64.1, 56.7, 55.4, 58.8, 67.7, 92.5, 76.6, 62.3, 56.8, 60.5, 71.0, 71.3, 51.1]
MAT = [50.6, 49.3, 62.5, 52.2, 60.3, 51.7, 53.3, 53.9, 58.6, 92.2, 54.4, 54.9, 54.0, 76.5, 59.4, 68.4, 51.0 ]
SBIs = [50.3, 57.4, 74.8, 61.3, 67.5, 54.6, 61.5, 53.2, 57.1, 95.9, 57.2, 52.9, 55.4, 78.3, 59.3, 74.6, 50.7 ]
ADD = [50.3, 50.2, 54.4, 51.0, 53.4, 50.7, 46.2, 50.5, 50.9, 75.8, 51.4, 51.6, 52.7, 72.6, 52.3, 66.4, 50.7 ]
QAD = [68.5, 46.4, 79.6, 76.7, 77.1, 73.6, 58.3, 76.3, 81.0, 90.2, 65.3, 71.3, 64.6, 81.8, 77.1, 66.7, 55.1 ]
ours = [68.7, 35.1, 81.0, 80.4, 78.2, 74.5, 62.2, 77.5, 81.7, 91.7, 69.2, 70.4, 68.0, 78.8, 73.4, 73.8, 55.3 ]


# 将12个月份分成360度
angles = np.linspace(0, 2*np.pi, len(datasets), endpoint=False)

# 将雷达图的第一个点重复以形成一个封闭的图形
MesoNet += MesoNet[:1]
FF += FF[:1]
F3Net += F3Net[:1]
MAT += MAT[:1]
SBIs += SBIs[:1]
ADD += ADD[:1]
QAD += QAD[:1]
ours += ours[:1]

angles = np.concatenate((angles, [angles[0]]))
print(ours)
# 创建雷达图
fig, ax = plt.subplots(figsize=(15, 15), subplot_kw=dict(polar=True))

ax.plot(angles, ours, label='ours', marker='o', color='red', linewidth=2)
ax.plot(angles, MesoNet, label='MesoNet', marker='o', color='lightgreen', linewidth=2, linestyle='dashed')
ax.plot(angles, FF, label='FF++', marker='o', color='lightblue',linewidth=2, linestyle='dashed')
ax.plot(angles, F3Net, label='F3Net', marker='o', color=(1.0, 0.8, 0.6),linewidth=2, linestyle='dashed')
ax.plot(angles, MAT, label='MAT', marker='o', color=(0.7, 0.5, 0.3),linewidth=2, linestyle='dashed')
ax.plot(angles, SBIs, label='SBIs', marker='o', color='lightpink',linewidth=2, linestyle='dashed')
ax.plot(angles, ADD, label='ADD', marker='o', color=(0.5, 0.4, 0.5),linewidth=2, linestyle='dashed')
ax.plot(angles, QAD, label='QAD', marker='o', color='lightgray',linewidth=2, linestyle='dashed')

label_positions = np.linspace(0, 2 * np.pi, len(datasets), endpoint=False)
# 下面两行代码调整了月份的位置
ax.set_xticks(label_positions)
ax.set_xticklabels(datasets, rotation=45, ha='center',size = 18)


# 其余代码保持不变
ax.set_yticklabels([])
ax.set_ylim(0, 100)
ax.legend(loc='upper right', bbox_to_anchor=(1.12, 1.12), frameon=False, prop = {'size':18})


label_positions_rad = label_positions * 180 / np.pi  # 将弧度转换为角度
for label, pos in zip(ax.get_xticklabels(), label_positions_rad):
    label.set_rotation(pos)
    #label.set_horizontalalignment('center')
    """
    if pos >= 0 and pos < 180:
        label.set_position((0.5, 1.1))
        label.set_horizontalalignment('center')
    else:
        label.set_position((0.5, -0.1))
        label.set_horizontalalignment('center')
    """
plt.savefig('figure2_avg_leida.png', bbox_inches='tight')



