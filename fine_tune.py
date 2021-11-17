import os
import pandas as pd
import numpy as np
import tensorflow as tf
from glob import glob
from dataloder import dataset
from unets import U_Net
from test_pre import predict_on_array
from loss import dice_loss, iou, tree_iou


def get_fine_path(path, mode='train', seed=2, p=0.0):
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
    if p != 0.0:
        low_image_path, low_mask_path = image_path[:140], low_mask_path[:140]
        high_image_path, high_mask_path = image_path[140: int(140+p*140)], high_mask_path[140: int(140+p*140)]
        return low_image_path, low_mask_path, high_image_path, high_mask_path
    else:
        return image_path, high_mask_path


def lr_cosine_decay(e):
    initial_learning_rate = 0.0001
    cosine_decay = 0.5 * (1 + np.cos(np.pi * e / epochs))
    return initial_learning_rate * cosine_decay


def initial_model(initial_dataset, validation_dataset):
    epochs = 200
    initial_learning_rate = 0.0001
    if os.path.exists(r'checkpoints/fine/unet_140_0'):
        model = tf.keras.models.load_model(f'checkpoints/fine/unet_140_0',
                                           custom_objects={'dice_loss': dice_loss,
                                                           'iou': iou,
                                                           'tree_iou': tree_iou})
    else:
        optimizer = tf.optimizers.Adam(learning_rate=initial_learning_rate)

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            # Monte Carlo Dropout
            model = U_Net(input_shape=(256, 256, 7), n_classes=2, rate=.0, mc=False, residual=True)
            model.compile(optimizer=optimizer, loss=[dice_loss], metrics=[iou, tree_iou])
        # model.summary()

        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_cosine_decay, verbose=0)

        # tensorboard
        tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir=f'tb_callback_dir/fine/unet_140_0',
                                                               histogram_freq=1)

        model.fit(initial_dataset,
                  steps_per_epoch=len(initial_dataset),
                  epochs=epochs,
                  validation_data=validation_dataset,
                  validation_steps=len(validation_dataset),
                  verbose=0,
                  callbacks=[learning_rate_scheduler, tensorboard_callbacks])
        model.save(f'checkpoints/fine/unet_140_0')
    return model


def model_test(model, test_datasets, image_id_test, p):
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
    with pd.ExcelWriter(r'checkpoints/fine/r.xlsx', mode='a',
                        engine='openpyxl', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=f'fine_tune_{int(p * 10)}')


if __name__ == '__main__':
    path = r'../quality/'
    freeze = False
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    epochs = 100
    initial_learning_rate = 0.00001

    valid_image_path, valid_mask_path = get_fine_path(path, mode='valid', seed=2)
    test_image_path, test_mask_path = get_fine_path(path, mode='test', seed=2)
    image_id_test = [int(p.split('_')[-1].split('.')[0]) for p in test_mask_path]

    valid_datasets = dataset(valid_image_path,
                             valid_mask_path,
                             mode='train',
                             batch_size=10)

    test_datasets = dataset(test_image_path,
                            test_mask_path,
                            mode='train',
                            batch_size=1)

    for p in np.arange(0.2, 1.2, 0.2):
        print(f'{p:.0%} high quality mask fine tuning...')
        train_image_path, train_mask_path, fine_image_path, fine_mask_path = get_fine_path(path,
                                                                                           mode='train',
                                                                                           seed=2,
                                                                                           p=p)
        initial_datasets = dataset(train_image_path,
                                   train_mask_path,
                                   mode='train',
                                   batch_size=4)
        fine_datasets = dataset(fine_image_path,
                                fine_mask_path,
                                mode='train',
                                batch_size=4)

        model = initial_model(initial_datasets, valid_datasets)
        print('initial model training and loading successful')

        # initial model prediction on test dataset
        # if p == 0.1:
        #     model_test(model, test_datasets, image_id_test, p=0.0)

        # model fine tuning phrase
        # freeze the encoder part of trained Unet
        if freeze is True:
            for layer in model.layers[:48]:
                layer.trainable = False

        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
        model.compile(optimizer=model.optimizer, loss=model.loss, metrics=[iou, tree_iou])
        model.summary()

        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_cosine_decay, verbose=0)
        # tensorboard
        tensorboard_callbacks = tf.keras.callbacks.TensorBoard(
            log_dir=f'tb_callback_dir/fine/no_freeze/unet_140_fine_{int(p*10)}',
            histogram_freq=1)
        model.fit(fine_datasets,
                  steps_per_epoch=len(fine_datasets),
                  epochs=epochs,
                  validation_data=valid_datasets,
                  validation_steps=len(valid_datasets),
                  verbose=0,
                  callbacks=[learning_rate_scheduler, tensorboard_callbacks]
                  )

        model.save(f'checkpoints/fine/no_freeze/unet_140_fine_{int(p*10)}')
        print(f'unet_140_fine_{int(p*10)} saved!')
        model_test(model, test_datasets, image_id_test, p=p)

        del model