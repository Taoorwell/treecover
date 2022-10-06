import pandas as pd
import numpy as np
import tensorflow as tf
from utility import get_image
from dataloder import dataset, get_path
from unets import U_Net
from test_pre import predict_on_array
from loss import dice_loss, iou, tree_iou


def random_pick(mask_path=r'../quality/high/', seed=3, t=2):
    # get image and mask path according to the mode (train, valid, test)
    images_path, masks_path, _ = get_path(path=mask_path, mode='train', seed=2)

    # generate random index to select mask
    np.random.seed(seed)
    s = np.random.choice(len(images_path), size=int(t*40), replace=False)

    images_path = np.array(images_path)
    masks_path = np.array(masks_path)

    images_path = images_path[s]
    masks_path = masks_path[s]

    # train, valid, test consistent
    images_path = images_path.tolist()
    masks_path = masks_path.tolist()

    return images_path, masks_path


def model_test(model, images_path, masks_path):
    true_masks_cat, pre_masks_cat = [], []
    for im, ms in zip(images_path, masks_path):
        image_arr, mask_arr = get_image(im), get_image(ms)
        output = predict_on_array(model=model,
                                  arr=image_arr,
                                  in_shape=(256, 256, 7),
                                  out_bands=2,
                                  stride=200,
                                  batchsize=20,
                                  augmentation=True)
        true_masks_cat.append(mask_arr)
        pre_masks_cat.append(output)
    true_mask = np.concatenate(true_masks_cat, axis=0)
    pre_mask = np.concatenate(pre_masks_cat, axis=0)
    mean_tree_iou = iou(true_mask[:, :, 1], pre_mask[:, :, 1]).numpy()
    mean_o_iou = iou(true_mask, pre_mask).numpy()

    return mean_tree_iou, mean_o_iou


if __name__ == '__main__':
    mask_path = r'../quality/high/'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    epochs = 150
    n_classes = 2
    loss_fn = dice_loss
    initial_learning_rate = 0.0001
    #
    valid_image_path, valid_mask_path, _ = get_path(mask_path, mode='valid', seed=2)
    test_image_path, test_mask_path, _ = get_path(path=r'../quality/high/', mode='test', seed=2)

    valid_datasets = dataset(valid_image_path,
                             valid_mask_path,
                             mode='train',
                             batch_size=10)

    Numbers, seeds, tree_ious, o_ious = [], [], [], []
    for t in np.arange(2, 7, 1):
        for se in np.arange(3, 6):
            print(f'{t*40} {se} masks training...')
            train_image_path, train_mask_path = random_pick(mask_path=mask_path, seed=se, t=t)
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
            tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir=f'tb_callback_dir/random/'
                                                                           f'unet_random_{int(t*40)}_{se}',
                                                                   histogram_freq=1)

            model.fit(train_datasets,
                      steps_per_epoch=len(train_datasets),
                      epochs=epochs,
                      validation_data=valid_datasets,
                      validation_steps=len(valid_datasets),
                      verbose=0,
                      callbacks=[learning_rate_scheduler, tensorboard_callbacks])
            model.save(f'checkpoints/random/unet_random_{int(t*40)}_{se}')
            # model prediction on test dataset
            acc1, acc2 = model_test(model,
                                    images_path=test_image_path,
                                    masks_path=test_mask_path)
            del model
            print(f'{t*40} {se} masks training Finished...')
            Numbers.append(t*40)
            seeds.append(se)
            tree_ious.append(acc1)
            o_ious.append(acc2)
    df = pd.DataFrame({'N': Numbers, 'Seed': seeds, 'tree_iou': tree_ious, 'o_iou': o_ious})
    with pd.ExcelWriter(r'checkpoints/random/random.xlsx', mode='a',
                        engine='openpyxl', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=f'high_mask_random')
