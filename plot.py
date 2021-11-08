import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

path = r'../results/'

# ####################high and low quality with residual Unet on test images (30) ############
# df = pd.read_excel(path + r'high_low_2/r1.xlsx', sheet_name='low-high-nodropout')
# df = df[:30]
# print(df)

# ################## over Iou between high and low quality datasets ############
# plt.scatter(x=np.arange(0, 30), y=df['o_iou1'],
#             marker='o', c='c', alpha=0.3, s=80,
#             label='low quality')
# plt.axhline(y=np.mean(df['o_iou1']), c='c', linestyle='--', label='mean_iou_low')
#
# plt.scatter(x=np.arange(0, 30), y=df['o_iou2'],
#             marker='o', c='r', alpha=0.3, s=80,
#             label='high quality')
# plt.axhline(y=np.mean(df['o_iou2']), c='r', linestyle='--', label='mean_iou_high')
# for index, row in df.iterrows():
#     if np.abs(row['o_iou1'] - row['o_iou2']) > 0.02:
#         plt.text(index+0.3, y=row['o_iou1'], s=int(row['N']), fontsize=8)
#         plt.text(index+0.3, y=row['o_iou2'], s=int(row['N']), fontsize=8)
#     else:
#         plt.text(index+0.3, y=row['o_iou1'], s=int(row['N']), fontsize=8)
# plt.yticks([0.5, 0.6, 0.7, 0.8, np.mean(df['o_iou1']), np.mean(df['o_iou2']), 0.9, 1.0])
# plt.ylabel('Iou')
# plt.xticks([])
# plt.xlabel('image')
# plt.legend()
# plt.show()

# ################## tree Iou between high and low quality datasets ############
# plt.scatter(x=np.arange(0, 30), y=df['tree_iou1'],
#             marker='o', c='c', alpha=0.3, s=80,
#             label='low quality')
# plt.axhline(y=np.mean(df['tree_iou1']), c='c', linestyle='--', label='mean_iou_low')
#
# plt.scatter(x=np.arange(0, 30), y=df['tree_iou2'],
#             marker='o', c='r', alpha=0.3, s=80,
#             label='high quality')
# plt.axhline(y=np.mean(df['tree_iou2']), c='r', linestyle='--', label='mean_iou_high')
# for index, row in df.iterrows():
#     if np.abs(row['tree_iou1'] - row['tree_iou2']) > 0.02:
#         plt.text(index+0.3, y=row['tree_iou1'], s=int(row['N']), fontsize=8)
#         plt.text(index+0.3, y=row['tree_iou2'], s=int(row['N']), fontsize=8)
#     else:
#         plt.text(index+0.3, y=row['tree_iou1'], s=int(row['N']), fontsize=8)
# plt.yticks([0.5, 0.6, 0.7, 0.8, np.mean(df['tree_iou1']), np.mean(df['tree_iou2'])])
# plt.ylabel('Iou')
# plt.xticks([])
# plt.xlabel('image')
# plt.legend()
# plt.show()
# ####################### mixture datasets on test images (30) #################
# m_tree_iou, m_o_iou = [], []
# plt.figure(figsize=(12, 6))
# for i in np.arange(1, 10):
#     # print(i)
#     # print(f'mix_mask_{i}')
#     df = pd.read_excel(path + 'mix/r-mix.xlsx', sheet_name=f'mix_mask_{i}')
#     df = df[:-1]
#     # tree_iou = df['tree_iou1']
#     n = df['N']
#     o_iou = df['tree_iou1']
#     plt.scatter(np.arange(0, 30), o_iou, s=50, alpha=0.5, label=f'{(i*0.1):.0%}')
#     # print(df)
#     # m_t, m_o = np.mean(df['tree_iou1']), np.mean(df['o_iou1'])
#     # # print(m_t, m_o)
#     # m_tree_iou.append(m_t)
#     # m_o_iou.append(m_o)
#     # break
#     plt.xticks(np.arange(0, 30), n, fontsize=10, rotation=45)
#     plt.ylabel('Tree Iou', fontweight='bold')
#     plt.xlabel('Image Id', fontweight='bold')
# plt.legend()
# plt.show()

# #########################group bar plot for mix datasets on test images (30) #########
# plt.figure(figsize=(12, 6))
# barWidth = 0.25
# bar1 = m_tree_iou
# bar2 = m_o_iou
#
# r1 = np.arange(0, 9, 1)
# r2 = [x + barWidth for x in r1]
#
# plt.bar(r1, bar1, color='r', width=barWidth, edgecolor='white', label='Tree_iou', alpha=0.4)
# plt.bar(r2, bar2, color='c', width=barWidth, edgecolor='white', label='Overall_iou', alpha=0.4)
#
# plt.xlabel('Dataset', fontweight='bold')
# plt.ylabel("Iou", fontweight='bold')
# plt.xticks([r+0.15 for r in np.arange(0, 9, 1)], [f'{p:.0%} high \n {(1-p):.0%} low' for p in np.arange(0.1, 1.0, 0.1)])
# plt.ylim(0.5, 0.9)
# plt.legend(loc=2, prop={'size': 8})
# plt.show()
