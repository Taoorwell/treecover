import pandas as pd
import numpy as np
from loss import dice, iou
from utility import load_path, get_image
from unets import U_Net
from test_pre import predict_on_array

width = 256
path = [r'../quality/images', r'../quality/high/']

model = U_Net(input_shape=(256, 256, 7), n_classes=2, recurrent=True, residual=True, attention=True)
model.load_weights(r'checkpoints/ckpt-unet_rec_res_att_300_softmax')
print('model load successfully')

paths = load_path(path=path, mode='train')
image_ids, accs = [], []
for image_path, high_mask_path in zip(paths[0], paths[1]):
    image_id = image_path.split('_')[-1].split('.')[0]
    # print(image_id)
    image_arr, mask_arr = get_image(image_path), get_image(high_mask_path)
    mask_arr = np.eye(2)[np.array(mask_arr[:, :, 0], dtype=np.int32)]
    # print(image_arr.shape, mask_arr.shape)
    output, _ = predict_on_array(model=model,
                                 arr=image_arr,
                                 in_shape=(256, 256, 7),
                                 out_bands=2,
                                 stride=200,
                                 batchsize=20,
                                 augmentation=True,
                                 verbose=False,
                                 report_time=True)
    acc = iou(mask_arr, output)
    print(acc.numpy())
    image_ids.append(image_id)
    accs.append(acc.numpy())
    print(image_id, 'predicted successfully')
    # break
df = pd.DataFrame({'ID': image_ids, 'Iou': accs})
df.to_excel('../results/train_1.xlsx')
