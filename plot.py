import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utility import *
# ##################### three masks plotting
# path = r'../results/'
image_path = r'../quality/'
# i = 174
# image_sample_high = get_image(image_path + f'high/mask_{i}.tif'
# image_sample_low = get_image(image_path + f'low/mask_{i}.tif')
# image_sample_image = get_image(image_path + f'images/tile_{i}.tif')
#
# plt.figure(figsize=(12, 4))
# plt.subplot(131)
# plt.imshow(image_sample_image[:, :, [4, 3, 2]])
# plt.imshow(rgb_mask(image_sample_high[:, :, 1]), alpha=0.5)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(r'High cost mask')
#
# plt.subplot(132)
# plt.imshow(image_sample_image[:, :, [4, 3, 2]])
# plt.imshow(rgb_mask(image_sample_low[:, :, 1]), alpha=0.5)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(r'Low cost mask')
#
#
# plt.subplot(133)
# plt.imshow(image_sample_image[:, :, [4, 3, 2]])
# active_201 = get_mat_info(image_path + f'active/active_{i}')
# active_201 = active_201[:, :, 1] > 0.5
# plt.imshow(rgb_mask(active_201), alpha=0.5)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(r'Model predicted mask')


i = 259
###############################################################################
raw_image = get_image(image_path + f'images/tile_{i}.tif')
ground_truth = get_image(image_path + f'high/mask_{i}.tif')
plt.figure(figsize=(12, 6))
plt.subplots_adjust(wspace=0.1, hspace=0.2)
plt.subplot(241)
plt.imshow(raw_image[:, :, [4, 3, 2]])
plt.xticks([])
plt.yticks([])
plt.xlabel(f'Image_{i}')
#
plt.subplot(242)
plt.imshow(rgb_mask(ground_truth[:, :, 1]))
plt.xticks([])
plt.yticks([])
plt.xlabel(r'Ground truth')

plt.subplot(243)
low_174 = get_mat_info(image_path + f'mix/{i}_l.mat')
acc = iou(ground_truth, low_174)
low_174 = low_174[:, :, 1] > 0.5
plt.imshow(rgb_mask(low_174))
plt.xticks([])
plt.yticks([])
plt.xlabel(f'100% Low acc:{acc:.2%}')

plt.subplot(244)
low_174_s = np.zeros((333, 333, 2))
for p in range(3, 8, 1):
    low_174 = get_mat_info(image_path + f'mix/{i}_mix_2_{p}.mat')
    low_174_s = low_174_s + low_174
low_174 = low_174_s / 5
acc = iou(ground_truth, low_174)

low_174 = low_174[:, :, 1] > 0.5
plt.imshow(rgb_mask(low_174))
plt.xticks([])
plt.yticks([])
plt.xlabel(f'20% High acc:{acc:.2%}')

plt.subplot(245)
low_174_s = np.zeros((333, 333, 2))
for p in range(3, 8, 1):
    low_174 = get_mat_info(image_path + f'mix/{i}_mix_4_{p}.mat')
    low_174_s = low_174_s + low_174
low_174 = low_174_s / 5
acc = iou(ground_truth, low_174)

low_174 = low_174[:, :, 1] > 0.5
plt.imshow(rgb_mask(low_174))
plt.xticks([])
plt.yticks([])
plt.xlabel(f'40% High acc:{acc:.2%}')

plt.subplot(246)
low_174_s = np.zeros((333, 333, 2))
for p in range(3, 8, 1):
    low_174 = get_mat_info(image_path + f'mix/{i}_mix_6_{p}.mat')
    low_174_s = low_174_s + low_174
low_174 = low_174_s / 5
acc = iou(ground_truth, low_174)

low_174 = low_174[:, :, 1] > 0.5
plt.imshow(rgb_mask(low_174))
plt.xticks([])
plt.yticks([])
plt.xlabel(f'60% High acc:{acc:.2%}')

plt.subplot(247)
low_174_s = np.zeros((333, 333, 2))
for p in range(3, 8, 1):
    low_174 = get_mat_info(image_path + f'mix/{i}_mix_8_{p}.mat')
    low_174_s = low_174_s + low_174
low_174 = low_174_s / 5
acc = iou(ground_truth, low_174)

low_174 = low_174[:, :, 1] > 0.5
plt.imshow(rgb_mask(low_174))
plt.xticks([])
plt.yticks([])
plt.xlabel(f'80% High acc:{acc:.2%}')

plt.subplot(248)
high_174 = get_mat_info(image_path + f'mix/{i}_h.mat')
acc = iou(ground_truth, high_174)
high_174 = high_174[:, :, 1] > 0.5
plt.imshow(rgb_mask(high_174))
plt.xticks([])
plt.yticks([])
plt.xlabel(f'100% High acc:{acc:.2%}')
# plt.show()
########################################################################################
############################### active learning plot ###################
# raw_image = get_image(image_path + f'images/tile_{i}.tif')
# ground_truth = get_image(image_path + f'high/mask_{i}.tif')
# plt.figure(figsize=(9, 9))
# plt.subplots_adjust(wspace=0.1, hspace=0.2)
# plt.subplot(331)
# plt.imshow(raw_image[:, :, [4, 3, 2]])
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(f'Image_{i}')
# #
# plt.subplot(332)
# plt.imshow(rgb_mask(ground_truth[:, :, 1]))
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(r'Ground truth')
#
# plt.subplot(333)
# low_174 = get_mat_info(image_path + f'active/low/{i}_initial.mat')
# acc = iou(ground_truth, low_174)
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(f'Initial model acc:{acc:.2%}')
#
# plt.subplot(334)
# low_174 = get_mat_info(image_path + f'active/low/{i}_active_2.mat')
# acc = iou(ground_truth, low_174)
#
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(f'Step 1 acc:{acc:.2%}')
#
# plt.subplot(335)
# low_174 = get_mat_info(image_path + f'active/low/{i}_active_3.mat')
# acc = iou(ground_truth, low_174)
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(f'Step 2 acc:{acc:.2%}')
#
# plt.subplot(336)
# low_174 = get_mat_info(image_path + f'active/low/{i}_active_4.mat')
# acc = iou(ground_truth, low_174)
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(f'Step 3 acc:{acc:.2%}')
#
# plt.subplot(337)
# low_174 = get_mat_info(image_path + f'active/low/{i}_active_5.mat')
# acc = iou(ground_truth, low_174)
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(f'Step 4 acc:{acc:.2%}')
#
# plt.subplot(338)
# high_174 = get_mat_info(image_path + f'active/low/{i}_active_6.mat')
# acc = iou(ground_truth, high_174)
# high_174 = high_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(high_174))
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(f'Step 5 acc:{acc:.2%}')
#
# plt.subplot(339)
# high_174 = get_mat_info(image_path + f'active/low/{i}_active_7.mat')
# acc = iou(ground_truth, high_174)
# high_174 = high_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(high_174))
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(f'Step 6 acc:{acc:.2%}')
######################################### active learning plot ############################
plt.show()

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
# # plt.figure(figsize=(12, 6))
# for i in np.arange(1, 10):
#     # print(i)
#     # print(f'mix_mask_{i}')
#     df = pd.read_excel(path + 'mix/r_mix_1.xlsx', sheet_name=f'mix_mask_{i}')
#     # df = df[:-1]
#     # tree_iou = df['tree_iou1']
#     # n = df['N']
#     # o_iou = df['tree_iou1']
#     # plt.scatter(np.arange(0, 30), o_iou, s=50, alpha=0.5, label=f'{(i*0.1):.0%}')
# #     # print(df)
#     m_t, m_o = np.mean(df['tree_iou1']), np.mean(df['o_iou1'])
#     # # print(m_t, m_o)
#     m_tree_iou.append(m_t)
#     m_o_iou.append(m_o)
#     # break
#     # plt.xticks(np.arange(0, 30), n, fontsize=10, rotation=45)
#     # plt.ylabel('Tree Iou', fontweight='bold')
#     # plt.xlabel('Image Id', fontweight='bold')
# # plt.legend()
# # plt.show()
#
# # #########################group bar plot for mix datasets on test images (30) ########### graph 1
# df = pd.read_excel(path + 'new_results.xlsx', sheet_name=f'mix')
# plt.figure(figsize=(10, 6))
# barWidth = 0.25
# bar1 = df['mean_tree_iou1']
# bar2 = df['mean_o_iou1']
# y_error1 = df['std1']
# y_error2 = df['std2']
#
# r1 = np.arange(0, 6, 1)
# r2 = [x + barWidth for x in r1]
#
# plt.bar(r1, bar1, yerr=y_error1, align='center', ecolor='blue', capsize=5, color='r', width=barWidth,
#         edgecolor='white', label='Tree_iou', alpha=0.4)
# plt.bar(r2, bar2, yerr=y_error2, align='center', ecolor='blue', capsize=5, color='c', width=barWidth,
#         edgecolor='white', label='Overall_iou', alpha=0.4)
#
# plt.xlabel('Dataset', fontweight='bold')
# plt.ylabel("Iou", fontweight='bold')
# plt.xticks([r+0.15 for r in np.arange(0, 6, 1)], [f'{p:.0%} high\n{(1-p):.0%} low' for p in np.arange(0.0, 1.2, 0.2)])
# plt.ylim(0.5, 0.9)
# plt.legend(loc=2, prop={'size': 8})
# plt.show()

# ########### model uncertainty analysis #############
# fig, axis = plt.subplots(ncols=3, nrows=2, figsize=(12, 6))
# plt.subplots_adjust(wspace=0.5, hspace=0.5)
# # deltas = np.arange(1, 7) * 0.01
#
# for i, ax in zip(range(2, 8), axis.flat):
#     # print(i)
#     df = pd.read_excel(path + 'active/high/r_fixed.xlsx', sheet_name=f'active_e_0.05_{i}')
#     df = df[:30]
#     # print(df)
#     ax.scatter(df['Entropy1'], df['O_iou'], marker='o', c='g', alpha=0.4, s=50)
#     ax.axvspan(0, 0.05, alpha=0.2, color='r')
#     ax.set_title(f'Step {i-1}')
#     ax.set_xlabel(r'Entropy')
#     ax.set_ylabel(r'Overall Iou')
#
# plt.show()

# ########### model uncertainty analysis #############
# fig, axis = plt.subplots(ncols=3, nrows=2, figsize=(12, 6))
# plt.subplots_adjust(wspace=0.5, hspace=0.5)
# # deltas = np.arange(1, 7) * 0.01
#
# for i, ax in zip(range(2, 8), axis.flat):
#     # print(i)
#     df = pd.read_excel(path + 'active/high/r_decay.xlsx', sheet_name=f'active_e_{i}')
#     df = df[:30]
#     # print(df)
#     ax.scatter(df['Entropy1'], df['O_iou'], marker='o', c='g', alpha=0.4, s=50)
#     ax.axvspan(0, (8-i)*0.01, alpha=0.2, color='r')
#     ax.set_title(f'Step {i-1}')
#     ax.set_xlabel(r'Entropy')
#     ax.set_ylabel(r'Overall Iou')

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
#     # ax.set_title(f'active_{i-1}_iteration_{delta}')
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
#     data = pd.read_excel(path + r'active/low/r_fixed.xlsx', sheet_name=f'active_data_{i}')
#     # print(data)
#     o_iou = data['overall iou']
#     plt.plot(o_iou, marker='o', label=f'entropy {i}', linestyle='dashed', linewidth=2, markersize=5, alpha=0.7)
#
# data1 = pd.read_excel(path + r'active/low/r_decay.xlsx', sheet_name=f'active_data')
# o_iou = data1['overall iou']
# plt.plot(o_iou, marker='o', label=f'entropy_decay', linestyle='dashed', linewidth=2, markersize=5, alpha=0.7)
# plt.xlabel('Iteration', fontweight='bold')
# plt.ylabel('Overall Iou', fontweight='bold')
# plt.legend()
# plt.show()
# ################## high quality fine tuning model trained on low quality datasets #########
# e_mean_t, e_mean_o = [], []
# d_mean_t, d_mean_o = [], []
# n_mean_t, n_mean_o = [], []
#
# for p in range(0, 12, 2):
#     df = pd.read_excel(path + 'fine/encoder_freeze/r_encoder.xlsx', sheet_name=f'fine_{p}')
#     df2 = pd.read_excel(path + 'fine/decoder_freeze/r_decoder.xlsx', sheet_name=f'fine_{p}')
#     df3 = pd.read_excel(path + 'fine/no_freeze/no_freeze.xlsx', sheet_name=f'fine_{p}')
#
#     mean_t = df['tree_iou1'].mean()
#     mean_o = df['o_iou1'].mean()
#
#     mean_t_2 = df2['tree_iou1'].mean()
#     mean_o_2 = df2['o_iou1'].mean()
#
#     mean_t_3 = df3['tree_iou1'].mean()
#     mean_o_3 = df3['o_iou1'].mean()
#
#     e_mean_t.append(mean_t)
#     e_mean_o.append(mean_o)
#     d_mean_t.append(mean_t_2)
#     d_mean_o.append(mean_o_2)
#     n_mean_t.append(mean_t_3)
#     n_mean_o.append(mean_o_3)
# print(e_mean_t, e_mean_o,
#       d_mean_t, d_mean_o)
# df1 = pd.read_excel(path + 'new_results.xlsx', sheet_name=f'fine_en_freeze')
# df2 = pd.read_excel(path + 'new_results.xlsx', sheet_name=f'fine_de_freeze')
# df3 = pd.read_excel(path + 'new_results.xlsx', sheet_name=f'fine_no_freeze')
#
# plt.errorbar(np.arange(0, 6, 1), df1['mean_o_iou1'], yerr=df1['std2'], capsize=3, marker='o',
#              label=f'encoder freeze',
#              linestyle='dashed', linewidth=2, markersize=5, alpha=0.7)
# plt.errorbar(np.arange(0, 6, 1), df2['mean_o_iou1'], yerr=df2['std2'], capsize=3, marker='o',
#              label=f'decoder freeze',
#              linestyle='dashed', linewidth=2, markersize=5, alpha=0.7)
# plt.errorbar(np.arange(0, 6, 1), df3['mean_o_iou1'], yerr=df3['std2'], capsize=3, marker='o',
#              label=f'no freeze',
#              linestyle='dashed', linewidth=2, markersize=5, alpha=0.7)

#
# plt.errorbar(np.arange(0, 6, 1), df1['tree_iou1'], yerr=df1['std1'], capsize=3, marker='o',
#              label=f'encoder freeze',
#              linestyle='dashed', linewidth=2, markersize=5, alpha=0.7)
# plt.errorbar(np.arange(0, 6, 1), df2['tree_iou1'], yerr=df2['std1'], capsize=3, marker='o',
#              label=f'decoder freeze',
#              linestyle='dashed', linewidth=2, markersize=5, alpha=0.7)
# plt.errorbar(np.arange(0, 6, 1), df3['tree_iou1'], yerr=df3['std1'], capsize=3, marker='o',
#              label=f'no freeze',
#              linestyle='dashed', linewidth=2, markersize=5, alpha=0.7)

# mean_t, mean_o = [], []
# plt.figure(figsize=(12, 6))
# for p in range(0, 12, 2):
#     print(p)
#     df = pd.read_excel(path + 'fine/decoder_freeze/r_decoder.xlsx', sheet_name=f'fine_{p}')
#     print(df)
#     n = df['N']
#     o_iou = df['o_iou1']
#     plt.scatter(np.arange(0, 30), o_iou, s=50, alpha=0.5, label=f'{(p*0.1):.0%}')
#     plt.xticks(np.arange(0, 30), n, fontsize=10, rotation=45)
#     plt.ylabel('Overall Iou', fontweight='bold')
#     plt.xlabel('Image Id', fontweight='bold')
#
# #     mean_tree_iou = df['tree_iou1'].mean()
# #     mean_o_iou = df['o_iou1'].mean()
# #     mean_t.append(mean_tree_iou)
# #     mean_o.append(mean_o_iou)
# # plt.figure(figsize=(12, 6))
# # barWidth = 0.25
# # bar1 = mean_t
# # bar2 = mean_o
# #
# # r1 = np.arange(0, 6, 1)
# # r2 = [x + barWidth for x in r1]
# #
# # plt.bar(r1, bar1, color='r', width=barWidth, edgecolor='white', label='Tree_iou', alpha=0.4)
# # plt.bar(r2, bar2, color='c', width=barWidth, edgecolor='white', label='Overall_iou', alpha=0.4)
# #
# #
# plt.xticks([r+0.15 for r in np.arange(0, 6, 1)], ['no fine tune',
#                                                   '10%', '20%', '30%', '40%', '50%'])
# # # # plt.ylim(0.5, 0.9)
# plt.xlabel('percentage of high quality datasets', fontweight='bold')
# plt.ylabel('Overall Iou', fontweight='bold')
# plt.legend()
# plt.show()
# ######################## active learning + high and low quality ####################
# df_high_active_initial = pd.read_excel(path + r'new_results.xlsx', sheet_name='high_active_initial')
# df_high_active_decay = pd.read_excel(path + r'new_results.xlsx', sheet_name='high_active_decay_overall')
# df_high_active_fixed_overall = pd.read_excel(path + r'new_results.xlsx', sheet_name='high_active_fixed_overall')

# df_low_active_initial = pd.read_excel(path + r'new_results.xlsx', sheet_name='low_active_initial')
# df_low_active_decay = pd.read_excel(path + r'new_results.xlsx', sheet_name='low_active_decay_overall')
# df_low_active_fixed_overall = pd.read_excel(path + r'new_results.xlsx', sheet_name='low_active_fixed_overall')
# #
# # # print(df_high_active_initial)
# # # print(df_high_active_decay)
# # # print(df_high_active_fixed_overall)
# #
# df_decay = df_low_active_initial.append(df_low_active_decay)
# # # print(df_decay)
# #
# E = pd.unique(df_low_active_fixed_overall['E'])
# plt.figure(figsize=(10, 5))
# plt.subplot(121)
# for e in [0.0, 0.03, 0.05, 0.1, 0.5]:
#     df_fixed = df_low_active_initial.append(df_low_active_fixed_overall.loc[df_low_active_fixed_overall['E'] == e])
#     # print(df_fixed)
#     plt.plot(np.arange(0, 7, 1), df_fixed['mean_o_iou1'],
#              marker='o', label=f'Entropy {e}', linestyle='dashed', linewidth=2, markersize=5, alpha=0.7)
#
# plt.plot(np.arange(0, 7, 1), df_decay['mean_o_iou1'],
#          marker='o', label=f'Entropy decay', linestyle='dashed', linewidth=2, markersize=5, alpha=0.7)
#
# plt.xlabel('Steps', fontweight='bold')
# plt.ylabel('Overall Iou', fontweight='bold')
# plt.legend()
#
# plt.subplot(122)
# for e in [0.0, 0.03, 0.05, 0.1, 0.5]:
#     df_fixed = df_low_active_initial.append(df_low_active_fixed_overall.loc[df_low_active_fixed_overall['E'] == e])
#     # print(df_fixed)
#     plt.plot(np.arange(0, 7, 1), df_fixed['mean_tree_iou1'],
#              marker='o', label=f'Entropy {e}', linestyle='dashed', linewidth=2, markersize=5, alpha=0.7)
#
# plt.plot(np.arange(0, 7, 1), df_decay['mean_tree_iou1'],
#          marker='o', label=f'Entropy decay', linestyle='dashed', linewidth=2, markersize=5, alpha=0.7)
#
# plt.xlabel('Steps', fontweight='bold')
# plt.ylabel('Tree Iou', fontweight='bold')
# plt.legend()

# plt.show()
# plt.show()
# plt.savefig(r'../results/graph/fixed_entropy_05.png', dpi=300)
