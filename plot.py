import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

path = r'../results/'

# #################### high and low quality with residual Unet on test images (30) ############
# df = pd.read_excel(path + r'high_low_2/r1.xlsx', sheet_name='low-high-nodropout')
# df = df[:30]
# print(df)
#
# # ################## over Iou between high and low quality datasets ############
# plt.figure(figsize=(12, 6))
# plt.scatter(x=np.arange(0, 30), y=df['tree_iou1'],
#             marker='o', c='c', alpha=0.3, s=80,
#             label='low quality')
# plt.axhline(y=np.mean(df['tree_iou1']), c='c', linestyle='--', label='mean_iou_low')
#
# plt.scatter(x=np.arange(0, 30), y=df['tree_iou2'],
#             marker='o', c='r', alpha=0.3, s=80,
#             label='high quality')
# plt.axhline(y=np.mean(df['tree_iou2']), c='r', linestyle='--', label='mean_iou_high')
# # for index, row in df.iterrows():
# #     if np.abs(row['o_iou1'] - row['o_iou2']) > 0.02:
# #         plt.text(index+0.3, y=row['o_iou1'], s=int(row['N']), fontsize=8)
# #         plt.text(index+0.3, y=row['o_iou2'], s=int(row['N']), fontsize=8)
# #     else:
# #         plt.text(index+0.3, y=row['o_iou1'], s=int(row['N']), fontsize=8)
# plt.yticks([0.5, 0.6, 0.7, 0.8, np.mean(df['tree_iou1']), np.mean(df['tree_iou2']), 0.9, 1.0])
# plt.ylabel('Tree Iou')
# plt.xticks(np.arange(0, 30), [int(x) for x in df['N']], rotation=45)
# plt.xlabel('Image Id')
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

# #########################group bar plot for mix datasets on test images (30) ###########
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

# ########### model uncertainty analysis #############
# fig, axis = plt.subplots(ncols=3, nrows=2, figsize=(12, 6))
# plt.subplots_adjust(wspace=0.3, hspace=0.5)
# # deltas = np.arange(1, 7) * 0.01
#
# for i, ax in zip(range(2, 8), axis.flat):
#     # print(i)
#     df = pd.read_excel(path + 'active/high/r.xlsx', sheet_name=f'active_e_0.0_{i}')
#     df = df[:30]
#     # print(df)
#     ax.scatter(df['Entropy1'], df['O_iou'], marker='o', c='g', alpha=0.4, s=50)
#     ax.axvspan(-0.001, 0., alpha=0.2, color='r')
#     ax.set_title(f'active_{i-1}_epoch_{0.}')
#     ax.set_xlabel(r'Entropy')
#     ax.set_ylabel(r'overall Iou')
#
# plt.show()

# #################### entropy and variance dual plot, acc as marker size ###################
# fig, axis = plt.subplots(ncols=3, nrows=2, figsize=(12, 6))
# plt.subplots_adjust(wspace=0.3, hspace=0.5)
# # deltas = np.arange(1, 7) * 0.01
#
# for i, ax in zip(range(2, 8), axis.flat):
#     # print(i)
#     df = pd.read_excel(path + 'active/high/r_high_fixed.xlsx', sheet_name=f'active_e_0.01_{i}')
#     df = df[:30]
#     # print(df)
#     # iou_min = np.min(df['O_iou'])
#     # iou_max = np.max(df['O_iou'])
#     # iou_mean = np.mean(df['O_iou'])
#     # s = 2*(df['O_iou'] - iou_min) / (iou_max - iou_min)
#     # ax.scatter(df['Entropy1'], df['Variance'], marker='o', c='g', alpha=0.4, s=s*10)
#     ax.scatter(df['Entropy1'], df['Variance'], marker='o', c='g', alpha=0.4, s=df['O_iou']*100)
#     # ax.axvspan(0, 0.03, alpha=0.2, color='r')
#     # ax.set_title(f'active_{i-1}_epoch_{delta}')
#     ax.set_xlabel(r'Entropy')
#     ax.set_ylabel(r'Variance')
#
# plt.show()
# ################### active learning + entropy selection + accuracy trend plot ###########################
# e = np.arange(0, 7) * 0.01
# e = np.append(e, [0.1, 0.5])
# plt.figure(figsize=(12, 6))
# for i in e:
#     # print(i)
#     data = pd.read_excel(path + r'active/high/r.xlsx', sheet_name=f'active_data_{i}')
#     # print(data)
#     o_iou = data['tree iou']
#     plt.plot(o_iou, marker='o', label=f'entropy {i}', linestyle='dashed', linewidth=2, markersize=5, alpha=0.7)
#
# data1 = pd.read_excel(path + r'active/high/r_decay.xlsx', sheet_name=f'active_data')
# o_iou = data1['tree iou']
# plt.plot(o_iou, marker='o', label=f'entropy_decay', linestyle='dashed', linewidth=2, markersize=5, alpha=0.7)
# plt.xlabel('Epochs')
# plt.ylabel('tree Iou')
# plt.legend()
# plt.show()
# ################## high quality fine tuning model trained on low quality datasets #########
mean_t, mean_o = [], []
plt.figure(figsize=(12, 6))
for p in range(0, 12, 2):
    print(p)
    df = pd.read_excel(path + 'fine/r.xlsx', sheet_name=f'fine_tune_{p}')
    print(df)
    n = df['N']
    o_iou = df['o_iou1']
    plt.scatter(np.arange(0, 30), o_iou, s=50, alpha=0.5, label=f'{(p*0.1):.0%}')
    plt.xticks(np.arange(0, 30), n, fontsize=10, rotation=45)
    plt.ylabel('Overall Iou', fontweight='bold')
    plt.xlabel('Image Id', fontweight='bold')

    mean_tree_iou = df['tree_iou1'].mean()
    mean_o_iou = df['o_iou1'].mean()
    mean_t.append(mean_tree_iou)
    mean_o.append(mean_o_iou)
# plt.figure(figsize=(12, 6))
# barWidth = 0.25
# bar1 = mean_t
# bar2 = mean_o
#
# r1 = np.arange(0, 6, 1)
# r2 = [x + barWidth for x in r1]
#
# plt.bar(r1, bar1, color='r', width=barWidth, edgecolor='white', label='Tree_iou', alpha=0.4)
# plt.bar(r2, bar2, color='c', width=barWidth, edgecolor='white', label='Overall_iou', alpha=0.4)
#
#
# plt.xticks([r+0.15 for r in np.arange(0, 6, 1)], ['no fine tune',
#                                                   '10%', '20%', '30%', '40%', '50%'])
# plt.ylim(0.5, 0.9)
# plt.xlabel('percentage of high quality datasets')
# plt.ylabel('Iou')
plt.legend()
plt.show()
