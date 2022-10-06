import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utility import *
###################### Three masks plotting ##########################
path = r'../results/'
image_path = r'../quality/'

# i = 71
# image_sample_high = get_image(image_path + f'high/mask_{i}.tif')
# image_sample_low = get_image(image_path + f'low/mask_{i}.tif')
# image_sample_image = get_image(image_path + f'images/tile_{i}.tif')

# plt.figure(figsize=(12, 4))
# plt.subplot(131)
# plt.imshow(image_sample_image[:, :, [4, 3, 2]])
# plt.imshow(rgb_mask(image_sample_high[:, :, 1]), alpha=0.5)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(r'High cost mask')

# plt.subplot(132)
# plt.imshow(image_sample_image[:, :, [4, 3, 2]])
# plt.imshow(rgb_mask(image_sample_low[:, :, 1]), alpha=0.5)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(r'Low cost mask')

# plt.subplot(133)
# plt.imshow(image_sample_image[:, :, [4, 3, 2]])
# active_201 = get_mat_info(image_path + f'active/active_{i}')
# active_201 = active_201[:, :, 1] > 0.5
# plt.imshow(rgb_mask(active_201), alpha=0.5)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(r'Model predicted mask')
i = 71
shuffle = [0]
P = 0
e = 90
# quality = 'low'

##################### Figure 7 #####################################
# raw_image = get_image(image_path + f'images/tile_{i}.tif')
# ground_truth = get_image(image_path + f'high/mask_{i}.tif')
#
# plt.figure(figsize=(12, 6))
# plt.subplots_adjust(wspace=0.1, hspace=0.2)
# plt.subplot(241)
# plt.imshow(raw_image[:, :, [4, 3, 2]])
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(f'Image_{i}')
#
# plt.subplot(242)
# plt.imshow(rgb_mask(ground_truth[:, :, 1]))
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(r'Ground truth')
#
# plt.subplot(243)
# low_174 = get_mat_info(image_path + f'mix/{i}_l.mat')
# acc = iou(ground_truth, low_174)
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# # plt.xlabel(f'0% P-100% R acc:{acc:.2%}')
# plt.xlabel(f'0% P-100% R')
#
# plt.subplot(244)
# low_174_s = np.zeros((333, 333, 2))
# for p in range(3, 8, 1):
#     low_174 = get_mat_info(image_path + f'mix/{i}_mix_2_{p}.mat')
#     low_174_s = low_174_s + low_174
# low_174 = low_174_s / 5
# acc = iou(ground_truth, low_174)
#
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# # plt.xlabel(f'20% P-80% R acc:{acc:.2%}')
# plt.xlabel(f'20% P-80% R')
#
# plt.subplot(245)
# low_174_s = np.zeros((333, 333, 2))
# for p in range(3, 8, 1):
#     low_174 = get_mat_info(image_path + f'mix/{i}_mix_4_{p}.mat')
#     low_174_s = low_174_s + low_174
# low_174 = low_174_s / 5
# acc = iou(ground_truth, low_174)
#
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# # plt.xlabel(f'40% P-60% R acc:{acc:.2%}')
# plt.xlabel(f'40% P-60% R')
#
#
# plt.subplot(246)
# low_174_s = np.zeros((333, 333, 2))
# for p in range(3, 8, 1):
#     low_174 = get_mat_info(image_path + f'mix/{i}_mix_6_{p}.mat')
#     low_174_s = low_174_s + low_174
# low_174 = low_174_s / 5
# acc = iou(ground_truth, low_174)
#
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# # plt.xlabel(f'60% P-40% R acc:{acc:.2%}')
# plt.xlabel(f'60% P-40% R')
#
#
# plt.subplot(247)
# low_174_s = np.zeros((333, 333, 2))
# for p in range(3, 8, 1):
#     low_174 = get_mat_info(image_path + f'mix/{i}_mix_8_{p}.mat')
#     low_174_s = low_174_s + low_174
# low_174 = low_174_s / 5
# acc = iou(ground_truth, low_174)
#
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# # plt.xlabel(f'80% P-20% R acc:{acc:.2%}')
# plt.xlabel(f'80% P-20% R')
#
#
# plt.subplot(248)
# high_174 = get_mat_info(image_path + f'mix/{i}_h.mat')
# acc = iou(ground_truth, high_174)
# high_174 = high_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(high_174))
# plt.xticks([])
# plt.yticks([])
# # plt.xlabel(f'100% P-0% R acc:{acc:.2%}')
# plt.xlabel(f'100% P-0% R')

# plt.show()

########################################################################################
############################### active learning plot ###################################
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
# # low_174 = get_mat_info(image_path + f'active/percent/high/{i}_initial.mat')
# low_174 = np.zeros((333, 333, 2))
# for s in shuffle:
#     low_174_s = get_mat_info(image_path + f'active_percent/{quality}/shuffle_{s}/{i}_initial.mat')
#     low_174 = low_174 + low_174_s
# low_174 = low_174 / len(shuffle)
#
# acc = iou(ground_truth, low_174)
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(f'Initial model acc:{acc:.2%}')
#
# plt.subplot(334)
# # low_174 = get_mat_info(image_path + f'active/percent/high/{e}/{i}_active_2.mat')
# low_174 = np.zeros((333, 333, 2))
# for s in shuffle:
#     low_174_s = get_mat_info(image_path + f'active_percent/{quality}/shuffle_{s}/no_scratch/{P}/{i}_active_2.mat')
#     low_174 = low_174 + low_174_s
# low_174 = low_174 / len(shuffle)
# acc = iou(ground_truth, low_174)
#
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(f'Step 1 acc:{acc:.2%}')
#
# plt.subplot(335)
# # low_174 = get_mat_info(image_path + f'active/percent/high/{e}/{i}_active_3.mat')
# low_174 = np.zeros((333, 333, 2))
# for s in shuffle:
#     low_174_s = get_mat_info(image_path + f'active_percent/{quality}/shuffle_{s}/no_scratch/{P}/{i}_active_3.mat')
#     low_174 = low_174 + low_174_s
# low_174 = low_174 / len(shuffle)
# acc = iou(ground_truth, low_174)
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(f'Step 2 acc:{acc:.2%}')
#
# plt.subplot(336)
# # low_174 = get_mat_info(image_path + f'active/percent/high/{e}/{i}_active_4.mat')
# low_174 = np.zeros((333, 333, 2))
# for s in shuffle:
#     low_174_s = get_mat_info(image_path + f'active_percent/{quality}/shuffle_{s}/no_scratch/{P}/{i}_active_4.mat')
#     low_174 = low_174 + low_174_s
# low_174 = low_174 / len(shuffle)
# acc = iou(ground_truth, low_174)
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(f'Step 3 acc:{acc:.2%}')
#
# plt.subplot(337)
# low_174 = np.zeros((333, 333, 2))
# for s in shuffle:
#     low_174_s = get_mat_info(image_path + f'active_percent/{quality}/shuffle_{s}/no_scratch/{P}/{i}_active_5.mat')
#     low_174 = low_174 + low_174_s
# low_174 = low_174 / len(shuffle)
# # low_174 = get_mat_info(image_path + f'active/percent/high/{e}/{i}_active_5.mat')
# acc = iou(ground_truth, low_174)
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(f'Step 4 acc:{acc:.2%}')
#
# plt.subplot(338)
# low_174 = np.zeros((333, 333, 2))
# for s in shuffle:
#     low_174_s = get_mat_info(image_path + f'active_percent/{quality}/shuffle_{s}/no_scratch/{P}/{i}_active_6.mat')
#     low_174 = low_174 + low_174_s
# low_174 = low_174 / len(shuffle)
# # low_174 = get_mat_info(image_path + f'active/percent/high/{e}/{i}_active_6.mat')
# acc = iou(ground_truth, low_174)
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(f'Step 5 acc:{acc:.2%}')
#
# plt.subplot(339)
# low_174 = np.zeros((333, 333, 2))
# for s in shuffle:
#     low_174_s = get_mat_info(image_path + f'active_percent/{quality}/shuffle_{s}/no_scratch/{P}/{i}_active_7.mat')
#     low_174 = low_174 + low_174_s
# low_174 = low_174 / len(shuffle)
# # low_174 = get_mat_info(image_path + f'active/percent/high/{e}/{i}_active_7.mat')
# acc = iou(ground_truth, low_174)
# high_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(high_174))
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(f'Step 6 acc:{acc:.2%}')

########################################### Figure 11 ######################################
# raw_image = get_image(image_path + f'images/tile_{i}.tif')
# ground_truth = get_image(image_path + f'high/mask_{i}.tif')
#
# plt.figure(figsize=(12, 9))
# plt.subplots_adjust(wspace=0.1, hspace=0.2)
#
# plt.subplot(3, 4, 1)
# plt.imshow(raw_image[:, :, [4, 3, 2]])
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(f'Image_{i}')
#
# plt.subplot(3, 4, 2)
# plt.imshow(rgb_mask(ground_truth[:, :, 1]))
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(r'Ground truth')
#
# plt.subplot(3, 4, 3)
# low_174 = get_mat_info(image_path + f'mix/{i}_l.mat')
# acc = iou(ground_truth, low_174)
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# # plt.xlabel(f'0% P-100% R acc:{acc:.2%}')
# plt.xlabel(f'Model (R)')
#
# plt.subplot(3, 4, 4)
# high_174 = get_mat_info(image_path + f'mix/{i}_h.mat')
# acc = iou(ground_truth, high_174)
# high_174 = high_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(high_174))
# plt.xticks([])
# plt.yticks([])
# # plt.xlabel(f'100% P-0% R acc:{acc:.2%}')
# plt.xlabel(f'Model (P)')
#
# plt.subplot(3, 4, 5)
# # low_174 = get_mat_info(image_path + f'active/percent/high/{i}_initial.mat')
# low_174 = np.zeros((333, 333, 2))
# for s in shuffle:
#     low_174_s = get_mat_info(image_path + f'active_percent/high/shuffle_{s}/{i}_initial.mat')
#     low_174 = low_174 + low_174_s
# low_174 = low_174 / len(shuffle)
#
# acc = iou(ground_truth, low_174)
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# # plt.xlabel(f'Initial model acc:{acc:.2%}')
# plt.xlabel(f'Initial model (P)')
#
# plt.subplot(3, 4, 6)
# # low_174 = get_mat_info(image_path + f'active/percent/high/{e}/{i}_active_2.mat')
# low_174 = np.zeros((333, 333, 2))
# for s in shuffle:
#     low_174_s = get_mat_info(image_path + f'active_percent/high/shuffle_{s}/no_scratch/{P}/{i}_active_2.mat')
#     low_174 = low_174 + low_174_s
# low_174 = low_174 / len(shuffle)
# acc = iou(ground_truth, low_174)
#
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# # plt.xlabel(f'Step 1 acc:{acc:.2%}')
# plt.xlabel(f'Model (P)+AL Step 1')
#
#
# plt.subplot(3, 4, 7)
# # low_174 = get_mat_info(image_path + f'active/percent/high/{e}/{i}_active_3.mat')
# low_174 = np.zeros((333, 333, 2))
# for s in shuffle:
#     low_174_s = get_mat_info(image_path + f'active_percent/high/shuffle_{s}/no_scratch/{P}/{i}_active_4.mat')
#     low_174 = low_174 + low_174_s
# low_174 = low_174 / len(shuffle)
# acc = iou(ground_truth, low_174)
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# # plt.xlabel(f'Step 3 acc:{acc:.2%}')
# plt.xlabel(f'Model (P)+AL Step 3')
#
# plt.subplot(3, 4, 8)
# # low_174 = get_mat_info(image_path + f'active/percent/high/{e}/{i}_active_4.mat')
# low_174 = np.zeros((333, 333, 2))
# for s in shuffle:
#     low_174_s = get_mat_info(image_path + f'active_percent/high/shuffle_{s}/no_scratch/{P}/{i}_active_7.mat')
#     low_174 = low_174 + low_174_s
# low_174 = low_174 / len(shuffle)
# acc = iou(ground_truth, low_174)
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# # plt.xlabel(f'Step 6 acc:{acc:.2%}')
# plt.xlabel(f'Model (P)+AL')
#
# plt.subplot(3, 4, 9)
# # low_174 = get_mat_info(image_path + f'active/percent/high/{i}_initial.mat')
# low_174 = np.zeros((333, 333, 2))
# for s in shuffle:
#     low_174_s = get_mat_info(image_path + f'active_percent/low/shuffle_{s}/{i}_initial.mat')
#     low_174 = low_174 + low_174_s
# low_174 = low_174 / len(shuffle)
#
# acc = iou(ground_truth, low_174)
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# # plt.xlabel(f'Initial model acc:{acc:.2%}')
# plt.xlabel(f'Initial model (R)')
#
#
# plt.subplot(3, 4, 10)
# # low_174 = get_mat_info(image_path + f'active/percent/high/{e}/{i}_active_2.mat')
# low_174 = np.zeros((333, 333, 2))
# for s in shuffle:
#     low_174_s = get_mat_info(image_path + f'active_percent/low/shuffle_{s}/no_scratch/{P}/{i}_active_2.mat')
#     low_174 = low_174 + low_174_s
# low_174 = low_174 / len(shuffle)
# acc = iou(ground_truth, low_174)
#
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# # plt.xlabel(f'Step 1 acc:{acc:.2%}')
# plt.xlabel(f'Model (R)+AL Step 1')
#
#
# plt.subplot(3, 4, 11)
# # low_174 = get_mat_info(image_path + f'active/percent/high/{e}/{i}_active_3.mat')
# low_174 = np.zeros((333, 333, 2))
# for s in shuffle:
#     low_174_s = get_mat_info(image_path + f'active_percent/low/shuffle_{s}/no_scratch/{P}/{i}_active_4.mat')
#     low_174 = low_174 + low_174_s
# low_174 = low_174 / len(shuffle)
# acc = iou(ground_truth, low_174)
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# # plt.xlabel(f'Step 3 acc:{acc:.2%}')
# plt.xlabel(f'Model (R)+AL Step 3')
#
# plt.subplot(3, 4, 12)
# # low_174 = get_mat_info(image_path + f'active/percent/high/{e}/{i}_active_4.mat')
# low_174 = np.zeros((333, 333, 2))
# for s in shuffle:
#     low_174_s = get_mat_info(image_path + f'active_percent/low/shuffle_{s}/no_scratch/{P}/{i}_active_7.mat')
#     low_174 = low_174 + low_174_s
# low_174 = low_174 / len(shuffle)
# acc = iou(ground_truth, low_174)
# low_174 = low_174[:, :, 1] > 0.5
# plt.imshow(rgb_mask(low_174))
# plt.xticks([])
# plt.yticks([])
# # plt.xlabel(f'Step 6 acc:{acc:.2%}')
# plt.xlabel(f'Model (R)+AL')



###########################################################################################
# # #########################group bar plot for mix datasets on test images (30) ########### graph 1
df = pd.read_excel(path + 'new_results.xlsx', sheet_name=f'mix')
plt.figure(figsize=(10, 6))
barWidth = 0.25
bar1 = df['mean_tree_iou1']
bar2 = df['mean_o_iou1']
y_error1 = df['std1']
y_error2 = df['std2']

r1 = np.arange(0, 6, 1)
r2 = [x + barWidth for x in r1]

plt.bar(r1, bar1, yerr=y_error1, align='center', ecolor='blue', capsize=5, color='r', width=barWidth,
        edgecolor='white', label='Tree Iou', alpha=0.4)
plt.bar(r2, bar2, yerr=y_error2, align='center', ecolor='blue', capsize=5, color='c', width=barWidth,
        edgecolor='white', label='Overall Iou', alpha=0.4)

plt.xlabel('Datasets', fontweight='bold')
plt.ylabel("Iou", fontweight='bold')
plt.xticks([r+0.15 for r in np.arange(0, 6, 1)], [f'{p:.0%} P\n{(1-p):.0%} R' for p in np.arange(0.0, 1.2, 0.2)])
plt.ylim(0.5, 0.9)
plt.legend(loc=2, prop={'size': 8})
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
# df_high_active = pd.read_excel(path + r'new_trains.xlsx', sheet_name='Sheet2')
# # df_high_active_decay = pd.read_excel(path + r'new_trains.xlsx', sheet_name='sheet1')
##### data loading area ########################################################################################
df_high_initial = pd.read_excel(path + r'new_results.xlsx', sheet_name='high_active_initial')
df_high_active_fixed_overall = pd.read_excel(path + r'new_results.xlsx', sheet_name='high_active_fixed_overall')
df_high_active_decay_overall = pd.read_excel(path + r'new_results.xlsx', sheet_name='high_active_decay_overall')

df_low_initial = pd.read_excel(path + r'new_results.xlsx', sheet_name='low_active_initial')
df_low_active_fixed_overall = pd.read_excel(path + r'new_results.xlsx', sheet_name='low_active_fixed_overall')
df_low_active_decay_overall = pd.read_excel(path + r'new_results.xlsx', sheet_name='low_active_decay_overall')

# label cost vs accuracy
df_label_cost = pd.read_excel(path + r'new_results.xlsx', sheet_name='label_cost_1')
df_new_train = pd.read_excel(path + r'new_trains.xlsx', sheet_name='continue-training-1')
df_new_train_l = pd.read_excel(path + r'new_trains.xlsx', sheet_name='low-continue-training-1')
##### data loading area ########################################################################################

# # df_low_active_initial = pd.read_excel(path + r'new_results.xlsx', sheet_name='low_active_initial')
# # df_low_active_decay = pd.read_excel(path + r'new_results.xlsx', sheet_name='low_active_decay_overall')
# # df_low_active_fixed_overall = pd.read_excel(path + r'new_results.xlsx', sheet_name='low_active_fixed_overall')

# # print(df_high_active_initial)
# # print(df_high_active_decay)
# # print(df_high_active_fixed_overall)

# # df_decay = df_low_active_initial.append(df_low_active_decay)
# # # print(df_decay)

# E = pd.unique(df_high_active_fixed_overall['E'])
######## Figure 9 active learning image selection process ##########################################
# fig, axis = plt.subplots(ncols=3, nrows=2, figsize=(12, 6))
# plt.subplots_adjust(wspace=0.5, hspace=0.5)
# p = 0.6
# for i, ax in zip(range(2, 8), axis.flat):
#     df = pd.read_excel(path + r'active/high/new_percent/shuffle_0/r.xlsx',
#                        sheet_name=f'active_e_{p}_{i}')
#     E = df['Entropy1'].to_numpy()
#     h = np.argsort(E)[:int(40*p)]
#     h_b = E[h[-1]]
#
#     l = np.argsort(E)[-int(40*(1-p)):]
#     l_b = E[l[0]]
#
#     ax.scatter(df['Entropy1'], df['O_iou'], marker='o', c='g', alpha=0.4, s=40)
#     ax.axvspan(E[h[0]], h_b, alpha=0.3, color='c', label='Model Labeled')
#     ax.axvspan(l_b, E[l[-1]], alpha=0.3, color='r', label='Human Labeled')
#
#     ax.set_title(f'Step {i-1}', fontweight='bold')
#     ax.set_xlabel(r'Entropy', fontweight='bold')
#     ax.set_yticks(np.arange(0, 1.1, 0.2))
#     ax.set_ylabel(r'Overall Iou', fontweight='bold')
#     ax.legend(fontsize=7,
#               loc='upper right' if i == 2 else 'lower left')
####################################################################################################

######## Figure 10 active learning on precisely datasets with entropy percent strategy #############
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# for p in [0, 20, 40, 60, 80, 100]:
#     df_high_active_p = df_new_train.loc[df_new_train['percent'] == p]
#     plt.plot(df_high_active_p['labeling cost'],
#              df_high_active_p['overall_iou'],
#              marker='o', label=f'Entropy {p}%', linestyle='dashed', linewidth=2,
#              markersize=5, alpha=0.5)
#     plt.fill_between(df_high_active_p['labeling cost'],
#                      df_high_active_p['overall_iou'] - df_high_active_p['std2'],
#                      df_high_active_p['overall_iou'] + df_high_active_p['std2'],
#                      alpha=0.1, linewidth=1)
# plt.axhline(y=0.8806,
#             c='r',
#             linewidth=1,
#             label='Without AL')
#
# plt.xlabel('Labeling cost (min)', fontweight='bold')
# plt.xticks(np.arange(40, 300, 40)*6)
# plt.yticks(np.arange(0.82, 0.89, 0.02))
# # plt.yticks(np.arange(0.71, 0.83, 0.01))
# plt.ylabel('Overall IoU', fontweight='bold')
# plt.title('(a) Precisely delineated dataset', fontweight='bold')
# plt.legend(loc='lower right')
#
# plt.subplot(122)
# for p in [0, 20, 40, 60, 80, 100]:
#     df_low_active_p = df_new_train_l.loc[df_new_train_l['percent'] == p]
#     plt.plot(df_low_active_p['labeling cost'],
#              df_low_active_p['overall_iou'],
#              marker='*', label=f'Entropy {p}%', linestyle='dashed', linewidth=2,
#              markersize=8, alpha=0.5)
#     plt.fill_between(df_low_active_p['labeling cost'],
#                      df_low_active_p['overall_iou'] - df_low_active_p['std2'],
#                      df_low_active_p['overall_iou'] + df_low_active_p['std2'],
#                      alpha=0.1, linewidth=1)
#
# plt.axhline(y=0.8113,
#             c='r',
#             linewidth=1,
#             label='Without AL')
#
# plt.xlabel('Labeling cost (min)', fontweight='bold')
# plt.xticks(np.arange(40, 300, 40)*2)
# plt.yticks(np.arange(0.76, 0.83, 0.02))
# # plt.yticks(np.arange(0.71, 0.83, 0.01))
# plt.ylabel('Overall IoU', fontweight='bold')
# plt.title('(b) Roughly delineated dataset', fontweight='bold')
# plt.legend(loc='lower right')
###Figure 10.2 #######################################################################################
# plt.figure(figsize=(8, 8))
# # plt.figure(figsize=(12, 6))
# # plt.subplot(121)
# for p in [0, 20, 40, 60, 80, 100]:
#     df_high_active_p = df_new_train.loc[df_new_train['percent'] == p]
#     plt.plot(df_high_active_p['labeling cost'],
#              df_high_active_p['overall_iou'],
#              marker='o', label=f'Without AL' if p == 0 else f'Entropy {p}%', linestyle='dashed', linewidth=2,
#              markersize=5, alpha=0.5)
#     plt.fill_between(df_high_active_p['labeling cost'],
#                      df_high_active_p['overall_iou'] - df_high_active_p['std2'],
#                      df_high_active_p['overall_iou'] + df_high_active_p['std2'],
#                      alpha=0.1, linewidth=1)
#
# # plt.xlabel('Labeling cost (min)', fontweight='bold')
# # plt.xticks(np.arange(40, 300, 40)*6)
# # plt.yticks(np.arange(0.82, 0.89, 0.02))
# # # plt.yticks(np.arange(0.71, 0.83, 0.01))
# # plt.ylabel('IoU', fontweight='bold')
# # plt.title('(a) Precisely delineated dataset', fontweight='bold')
# # plt.legend(loc='lower right')
#
# # plt.subplot(122)
# for p in [0, 20, 40, 60, 80, 100]:
#     df_low_active_p = df_new_train_l.loc[df_new_train_l['percent'] == p]
#     plt.plot(df_low_active_p['labeling cost'],
#              df_low_active_p['overall_iou'],
#              marker='*', label=f'Without AL' if p == 0 else f'Entropy {p}%', linestyle='dashed', linewidth=2,
#              markersize=8, alpha=0.5)
#     plt.fill_between(df_low_active_p['labeling cost'],
#                      df_low_active_p['overall_iou'] - df_low_active_p['std2'],
#                      df_low_active_p['overall_iou'] + df_low_active_p['std2'],
#                      alpha=0.1, linewidth=1)
#
# plt.xlabel('Labeling cost (min)', fontweight='bold')
# plt.xticks(np.arange(40, 300, 40)*6)
# plt.yticks(np.arange(0.76, 0.89, 0.02))
# # plt.yticks(np.arange(0.71, 0.83, 0.01))
# plt.ylabel('IoU', fontweight='bold')
# plt.title('(b) Roughly delineated dataset', fontweight='bold')
# plt.legend(loc='outside right')
########################################################################################################
#################### active learning + quality datasets ################################################
# plt.figure(figsize=(12, 10))
# plt.subplot(211)
# # for e in [0, 20, 40, 60, 80, 100]:
# for e in [0.0, 0.03, 0.05, 0.1, 0.5]:
#     # df_fixed = df_low_active_initial.append(df_low_active_fixed_overall.loc[df_low_active_fixed_overall['E'] == e])
#     # print(df_fixed)
#     df_high_active_e = df_high_active_fixed_overall.loc[df_high_active_fixed_overall['E'] == e]
#     plt.plot(df_high_initial['N'].append(df_high_active_e['L']),
#              df_high_initial['mean_o_iou1'].append(df_high_active_e['mean_o_iou1']),
#              marker='o', label=f'Without AL' if e == 0.0 else f'Entropy {e}', linestyle='dashed', linewidth=2,
#              markersize=5, alpha=0.7)
# #
# # plt.plot(np.arange(40, 320, 40), df_high_initial['mean_tree_iou1'].append(df_high_active_decay_overall['mean_tree_iou1']),
# #          marker='o', label=f'Entropy decay', linestyle='dashed', linewidth=2, markersize=5, alpha=0.7)
# #
# plt.xlabel('Labeling cost (min)', fontweight='bold')
# plt.xticks(np.arange(40, 300, 40)*6)
# plt.yticks(np.arange(0.76, 0.89, 0.03))
# # plt.yticks(np.arange(0.58, 0.8, 0.03))
# plt.ylabel('IoU', fontweight='bold')
# plt.title('(a) Precisely delineated dataset', fontweight='bold')
# plt.legend(loc='lower right')
# # #
# plt.subplot(212)
# # for e in [0, 20, 40, 60, 80, 100]:
# for e in [0.0, 0.03, 0.05, 0.1, 0.5]:
# #     # df_fixed = df_low_active_initial.append(df_low_active_fixed_overall.loc[df_low_active_fixed_overall['E'] == e])
# #     # print(df_fixed)
#     df_low_active_e = df_low_active_fixed_overall.loc[df_low_active_fixed_overall['E'] == e]
#     plt.plot(df_low_initial['N'].append(df_low_active_e['L']),
#              df_low_initial['mean_o_iou1'].append(df_low_active_e['mean_o_iou1']),
#              marker='o', label=f'Without AL' if e == 0.0 else f'Entropy {e}', linestyle='dashed', linewidth=2,
#              markersize=5, alpha=0.7)
# # plt.plot(np.arange(40, 320, 40), df_low_initial['mean_tree_iou1'].append(df_low_active_decay_overall['mean_tree_iou1']),
# #          marker='o', label=f'Entropy decay', linestyle='dashed', linewidth=2, markersize=5, alpha=0.7)
# #
# plt.xlabel('Labeling cost (min)', fontweight='bold')
# plt.xticks(np.arange(40, 300, 40)*6)
# # plt.yticks(np.arange(0.58, 0.8, 0.03))
# plt.yticks(np.arange(0.73, 0.89, 0.03))
# plt.ylabel('IoU', fontweight='bold')
# plt.title('(b) Roughly delineated dataset', fontweight='bold')
# plt.legend()

# plt.figure(figsize=(12, 6))
# df_high_label_cost = df_label_cost.loc[df_label_cost['quality'] == 1]
# df_low_label_cost = df_label_cost.loc[df_label_cost['quality'] == 2]
#
# df_high_initial_model = df_label_cost.loc[df_label_cost['quality'] == 3]
# df_low_initial_model = df_label_cost.loc[df_label_cost['quality'] == 4]
#
# plt.scatter(df_high_label_cost['label_cost'], df_high_label_cost['overall_acc'], label='Active learning (P)',
#             marker='o', c='green', s=150, alpha=0.4)
# for l, a, e in zip(df_high_label_cost['label_cost'], df_high_label_cost['overall_acc'], df_high_label_cost['entropy']):
#     plt.annotate(e,
#                  (l, a),
#                  textcoords='offset points',
#                  xytext=(0, -15),
#                  ha='center')
# plt.scatter(df_low_label_cost['label_cost'], df_low_label_cost['overall_acc'], label='Active learning (R)',
#             marker='o', c='red', s=150, alpha=0.4)
# for l, a, e in zip(df_low_label_cost['label_cost'], df_low_label_cost['overall_acc'], df_low_label_cost['entropy']):
#     plt.annotate(e,
#                  (l, a),
#                  textcoords='offset points',
#                  xytext=(0, -15),
#                  ha='center')
#
# plt.scatter(df_high_initial_model['label_cost'], df_high_initial_model['overall_acc'], label='Initial model (P)',
#             marker='^', c='green', s=150, alpha=0.4)
# plt.scatter(df_low_initial_model['label_cost'], df_low_initial_model['overall_acc'], label='Initial model (R)',
#             marker='^', c='red', s=150, alpha=0.4)
# plt.grid(visible=None, which='both', axis='both', color='grey', linestyle='-', linewidth=.5)
# # plt.yticks(np.arange(0.57, 0.80, 0.03))
# plt.yticks(np.arange(0.72, 0.88, 0.03))
# plt.xlabel('labeling cost', fontweight='bold')
# plt.ylabel('IoU', fontweight='bold')
# plt.legend()
# plt.show()
# plt.show()
plt.savefig(r'../results/graph/graph/Figure 6.png', dpi=300)
