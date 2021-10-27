import pandas as pd
import numpy as np
from loss import dice, iou
from dataloder import get_path, dataset
from unets import U_Net
from test_pre import predict_on_array

if __name__ == '__main__':
    # parameter define
    width = 256
    path = r'../quality/high'
    n_inference = 10

    # trained model reload
    model = U_Net(input_shape=(256, 256, 7), n_classes=2, recurrent=True, residual=True, attention=True)
    model.load_weights(r'checkpoints/ckpt-unet_r2_att_softmax_dice_loss')
    print('model load successfully')

    # datasets preparation
    image_path_active1, mask_path_active1, image_id_active1 = get_path(path=path,
                                                                       mode='train',
                                                                       seed=1,
                                                                       active=1)
    active1_datasets = dataset(image_path_active1,
                               mask_path_active1,
                               mode='test',
                               image_shape=(256, 256),
                               batch_size=1,
                               n_classes=2)

    e1, e2, var = [], [], []
    # model prediction
    for image_arr, mask_arr in active1_datasets:
        image_arr, mask_arr = image_arr[0], mask_arr[0]
        print(image_arr.shape, mask_arr.shape)
        outputs = np.zeros((n_inference, ) + mask_arr.shape, dtype=np.float32)
        for i in range(n_inference):
            output, _ = predict_on_array(model=model,
                                         arr=image_arr,
                                         in_shape=(256, 256, 7),
                                         out_bands=2,
                                         stride=200,
                                         batchsize=20,
                                         augmentation=True,
                                         verbose=False,
                                         report_time=True)
            outputs[i] = output
        # output prediction uncertainty estimation
        # categorical first cross entropy
        # first
        a = outputs[..., 0] * np.log2(outputs[..., 0]) + outputs[..., 1] * np.log2(outputs[..., 1])
        E1 = np.mean(a, axis=0)
        print(E1.shape)
        E1 = np.sum(E1)
        print(E1)

        # second
        b1, b2 = np.mean(outputs[..., 0], axis=0), np.mean(outputs[..., 1], axis=0)
        E2 = b1 * np.log2(b1) + b2 * np.log2(b2)
        print(E2.shape)
        E2 = np.sum(E2)
        print(E2)

        # third
        v1, v2 = np.var(outputs[..., 0], axis=0), np.var(outputs[..., 1], axis=0)
        v = v1 + v2
        print(v.shape)
        v = np.sum(v)
        print(v)

        e1.append(E1)
        e2.append(E2)
        var.append(v)
        # break
    df = pd.DataFrame({'ID': image_id_active1, 'Entropy1': e1, 'Entropy2': e2, 'Variance': var})
    print(df)
    # df.to_excel('../results/train_1.xlsx')
