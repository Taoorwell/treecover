import os
import pandas as pd
import numpy as np
import tensorflow as tf
from glob import glob
from dataloder import dataset
from unets import U_Net
from test_pre import predict_on_array
from loss import dice_loss, iou, tree_iou


def get_mix_path(path, mode='train', seed=2, p=0.0):
    # get image and mask path according to the mode (train, valid, test)
    images_path = sorted(glob(os.path.join(path, r"images/*.tif")))
    high_masks_path = sorted(glob(os.path.join(path, r'high/*.tif')))
    low_masks_path = sorted(glob(os.path.join(path, r'low/*.tif')))

    length = len(images_path)

    np.random.seed(seed)
    idx = np.random.permutation(length)

    train_idx, test_idx = idx[:-30], idx[-30:]
    if mode == 'train':
        idx = train_idx[:280]
    elif mode == 'valid':
        idx = train_idx[280:]
    else:
        idx = test_idx

    image_path = [images_path[i] for i in idx]
    high_mask_path = [high_masks_path[i] for i in idx]
    low_mask_path = [low_masks_path[i] for i in idx]
    if p != 0:
        high_mask_path[int(p * len(idx)):] = low_mask_path[int(p * len(idx)):]
    mask_path = high_mask_path
    return image_path, mask_path


if __name__ == '__main__':
    path = r'../quality/'
    # test_image_path, test_mask_path = get_mix_path(path, mode='test', seed=2)
    # print(test_mask_path)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    path = r'../quality/'
    # p = 0.9
    epochs = 300
    n_classes = 2
    loss_fn = dice_loss
    initial_learning_rate = 0.0001

    valid_image_path, valid_mask_path = get_mix_path(path, mode='valid', seed=2)
    test_image_path, test_mask_path = get_mix_path(path, mode='test', seed=2)
    image_id_test = [int(p.split('_')[-1].split('.')[0]) for p in test_mask_path]

    valid_datasets = dataset(valid_image_path,
                             valid_mask_path,
                             mode='train',
                             batch_size=10)

    test_datasets = dataset(test_image_path,
                            test_mask_path,
                            mode='train',
                            batch_size=1)

    for p in np.arange(0.1, 1.0, 0.1):
        print(f'{p:.0%} high quality mask training...')
        train_image_path, train_mask_path = get_mix_path(path, mode='train', seed=2, p=p)
        train_datasets = dataset(train_image_path,
                                 train_mask_path,
                                 mode='train',
                                 batch_size=4)
        optimizer = tf.optimizers.Adam(learning_rate=initial_learning_rate)


        def lr_cosine_decay(e):
            cosine_decay = 0.5 * (1 + np.cos(np.pi * e / epochs))
            return initial_learning_rate * cosine_decay


        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = U_Net(input_shape=(256, 256, 7),
                          n_classes=n_classes,
                          rate=0.0,
                          mc=False,
                          residual=True)
            model.compile(optimizer=optimizer, loss=[loss_fn], metrics=[iou, tree_iou])
        # model.summary()

        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_cosine_decay, verbose=0)
        # tensorboard
        tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir=f'tb_callback_dir/unet_mix_mask_{int(p*10)}',
                                                               histogram_freq=1)

        model.fit(train_datasets,
                  steps_per_epoch=len(train_datasets),
                  epochs=epochs,
                  validation_data=valid_datasets,
                  validation_steps=len(valid_datasets),
                  callbacks=[learning_rate_scheduler, tensorboard_callbacks])
        model.save(f'checkpoints/unet_mix_mask_{int(p*10)}')
        # model prediction on test dataset
        acc1, acc2 = [], []
        for (im, ms), i in zip(test_datasets, image_id_test):
            image_arr, mask_arr = im.numpy(), ms.numpy()
            output_1, _ = predict_on_array(model=model,
                                           arr=image_arr[0],
                                           in_shape=(256, 256, 7),
                                           out_bands=2,
                                           stride=200,
                                           batchsize=20,
                                           augmentation=True,
                                           verbose=False,
                                           report_time=True)

            acc_iou_1 = iou(mask_arr[0][:, :, 1], output_1[:, :, 1])
            acc1.append(acc_iou_1.numpy())

            acc_iou_2 = iou(mask_arr[0], output_1)
            acc2.append(acc_iou_2.numpy())

        df = pd.DataFrame({'N': image_id_test, 'tree_iou1': acc1, 'o_iou1': acc2})
        print(df)
        print(np.mean(acc1), np.mean(acc2))
        with pd.ExcelWriter(r'checkpoints/active/r.xlsx', mode='a',
                            engine='openpyxl', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=f'mix_mask_{int(p*10)}')
        del model
