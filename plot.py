import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


df = pd.read_excel(r'../results/train.xlsx')
print(df)
# plt.scatter(x=np.arange(0, 30), y=df['Low'],
#             marker='d', c='r', alpha=0.5, s=80,
#             label='low quality')
# plt.axhline(y=np.mean(df['Low']), c='r', linestyle='--', label='average_low')
#
# plt.scatter(x=np.arange(0, 30), y=df['High'],
#             marker='^', c='b', alpha=0.5, s=80,
#             label='high quality')
# plt.axhline(y=np.mean(df['High']), c='b', linestyle='--', label='average_high')
# for index, row in df.iterrows():
#     plt.text(index+0.3, y=row['Low'], s=int(row['N']), fontsize=7)
# plt.ylabel('acc')
# plt.xticks([])
# plt.xlabel('image')
# plt.legend()
# plt.show()
# iou = []
# for index, row in df.iterrows():
#     # print(row['Iou'])
#     acc = row['Iou']
#     # print(type(acc))
#     acc = float(acc.split(', ')[0].split('(')[-1])
#     # print(acc)
#     iou.append(acc)
# df['iou'] = iou
# df.to_excel(r'../results/train.xlsx')

