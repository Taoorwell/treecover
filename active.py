# import math
import os
# from utility import rgb_mask
# import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
from loss import dice_loss, iou, tree_iou
from dataloder import get_path, get_image, dataset
from unets import U_Net
from test_pre import predict_on_array


def get_active_image_mask_array_list(path_dataset):
    active_dataset_image, active_dataset_mask = np.array(list(map(get_image, path_dataset[0]))),\
                                                np.array(list(map(get_image, path_dataset[1])))
    return active_dataset_image, active_dataset_mask


# initial_dataset training for model 1
def lr_cosine_decay(e):
    initial_learning_rate = 0.0001
    cosine_decay = 0.5 * (1 + np.cos(np.pi * e / epochs))
    return initial_learning_rate * cosine_decay


def initial_model_train():
    if os.path.exists(r'checkpoints/active/unet_active_1.h5'):
        model = tf.keras.models.load_model(r'checkpoints/active/unet_active_1.h5',
                                           custom_objects={'dice_loss': dice_loss,
                                                           'iou': iou,
                                                           'tree_iou': tree_iou})
    else:
        optimizer = tf.optimizers.Adam(learning_rate=initial_learning_rate)

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            # Monte Carlo Dropout
            model = U_Net(input_shape=(256, 256, 7), n_classes=n_classes, rate=.1, mc=True, residual=True)
            model.compile(optimizer=optimizer, loss=[loss_fn], metrics=[iou, tree_iou])
        # model.summary()

        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_cosine_decay, verbose=0)

        # tensorboard
        # tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir='tb_callback_dir/active/unet_active_1',
        #                                                        histogram_freq=1)

        model.fit(initial_dataset,
                  steps_per_epoch=len(initial_dataset),
                  epochs=epochs,
                  validation_data=validation_dataset,
                  validation_steps=len(validation_dataset),
                  callbacks=[learning_rate_scheduler])
        model.save(r'checkpoints/active/unet_active_1.h5')
    return model


def model_test(model, dataset, inf, n):
    acc1, acc2 = [], []
    for (im, ms), i in zip(dataset, test_path_dataset[2]):
        image_arr, mask_arr = im.numpy(), ms.numpy()
        # print(image_arr.shape, type(image_arr))
        outputs = np.zeros((inf, ) + mask_arr[0].shape, dtype=np.float32)
        for i in range(inf):
            output_1 = predict_on_array(model=model,
                                        arr=image_arr[0],
                                        in_shape=(256, 256, 7),
                                        out_bands=2,
                                        stride=200,
                                        batchsize=20,
                                        augmentation=True)
            outputs[i] = output_1
        # print(outputs.shape)
        outputs = np.mean(outputs, axis=0)
        # print(outputs.shape)
        acc_iou_1 = iou(mask_arr[0][:, :, 1], outputs[:, :, 1])
        acc_iou_2 = iou(mask_arr[0], outputs)
        acc1.append(acc_iou_1.numpy())
        acc2.append(acc_iou_2.numpy())

    df = pd.DataFrame({'N': test_path_dataset[2],
                       'tree_iou': acc1,
                       'o_iou': acc2})
    print(df)
    mean_tree_iou, mean_o_iou = np.mean(acc1), np.mean(acc2)
    print(mean_tree_iou, mean_o_iou)
    with pd.ExcelWriter(r'checkpoints/active/r.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=f'active_{n}')
    return mean_tree_iou, mean_o_iou


def model_pred(model, images, masks, images_id, inf, delta):
    def uncertainty(outputs):
        eps = 1e-15
        # first
        a = -(outputs[..., 0] * np.log2(outputs[..., 0] + eps) + outputs[..., 1] * np.log2(outputs[..., 1] + eps))
        E1 = np.mean(a, axis=0)
        E1 = np.mean(E1)
        # second
        b1, b2 = np.mean(outputs[..., 0], axis=0), np.mean(outputs[..., 1], axis=0)
        E2 = -(b1 * np.log2(b1+eps) + b2 * np.log2(b2+eps))
        E2 = np.mean(E2)
        # # third
        v1, v2 = np.mean(outputs[..., 0], axis=0), np.mean(outputs[..., 1], axis=0)
        v = np.var(np.array([v1, v2]), axis=0)
        v = np.mean(v)
        return E1, E2, v

    prob, entropy1, entropy2, variance = [], [], [], []
    for im, ms, ids in zip(images, masks, images_id):
        outputs = np.zeros((inf,) + im.shape[:-1] + (2,), dtype=np.float32)
        for i in range(inf):
            mask_prob = predict_on_array(model=model,
                                         arr=im,
                                         in_shape=(256, 256, 7),
                                         out_bands=2,
                                         stride=100,
                                         batchsize=20,
                                         augmentation=False)
            outputs[i] = mask_prob
        e1, e2, v = uncertainty(outputs=outputs)
        # plt.subplot(131)
        # plt.imshow(im[:, :, :3])
        # plt.xlabel(f'image_{ids}')
        #
        # plt.subplot(132)
        # plt.imshow(rgb_mask(np.argmax(ms, axis=-1)))
        # plt.xlabel(f'mask_{ids}')
        #
        # plt.subplot(133)
        # plt.imshow(rgb_mask(np.argmax((np.mean(outputs, axis=0) > 0.5) * 1, axis=-1)))
        # plt.title(f'Entropy:{e1}')
        #
        # plt.show()
        entropy1.append(e1)
        entropy2.append(e2)
        variance.append(v)
        prob.append((np.mean(outputs, axis=0) > 0.5) * 1)

    df = pd.DataFrame({'Image_id': images_id,
                       'Entropy1': entropy1,
                       'Entropy2': entropy2,
                       'Variance': variance})
    print(df)
    image_id_selected = np.array(images_id)[np.array(entropy1) < delta]
    print(f'number of high: {len(image_id_selected)}, '
          f'high confidence index:{image_id_selected}')
    # replace mask from model prediction
    masks[np.array(entropy1) < delta] = np.array(prob)[np.array(entropy1) < delta]
    print(f'mask replacing finished!')

    return images, masks, prob, df, image_id_selected

# put new images and masks with previous datasets together.


if __name__ == '__main__':
    # some parameters
    seed = 2
    path = r'../quality/high/'
    initial_learning_rate = 0.0001
    epochs = 100
    n_classes = 2
    loss_fn = dice_loss
    delta = 0.21

    # initial datasets, validation and test datasets
    initial_path_dataset = get_path(path=path,
                                    mode='train',
                                    seed=seed,
                                    active=1)

    validation_path_dataset = get_path(path=path,
                                       mode='valid',
                                       seed=seed,
                                       active=0)

    test_path_dataset = get_path(path=path,
                                 mode='test',
                                 seed=seed,
                                 active=0)

    print(f'initial datasets length: {len(initial_path_dataset[2])}')
    print('initial datasets id:')
    print(initial_path_dataset[-1])

    # initial, validation and test tensorflow datasets
    initial_dataset_image, initial_dataset_mask = get_active_image_mask_array_list(initial_path_dataset)
    print(f'initial dataset image and mask loading successfully')

    initial_dataset = dataset(initial_dataset_image, initial_dataset_mask, mode='train', batch_size=4)
    validation_dataset = dataset(validation_path_dataset[0], validation_path_dataset[1], mode='valid', batch_size=10)
    test_dataset = dataset(test_path_dataset[0], test_path_dataset[1], mode='test', batch_size=1)
    print(f'initial, validation and test tensorflow datasets loading successfully')

    tree_ious, o_ious = [], []

    model = initial_model_train()
    print('initial model loaded successfully')
    print('initial model prediction on test datasets')
    i_tree_iou, i_o_iou = model_test(model, test_dataset, inf=5, n=1)
    tree_ious.append(i_tree_iou)
    o_ious.append(i_o_iou)
    model_labeled_r, human_labeled_r = [0], [40]
    for i in np.arange(2, 8):
        print(f'{i-1} Active learning starting! ')
        # Get active 2 path dataset
        active_path_dataset = get_path(path=path,
                                       mode='train',
                                       seed=seed,
                                       active=i)
        active_dataset_image, active_dataset_mask = get_active_image_mask_array_list(active_path_dataset)

        print(f'{i} new batch active datasets loading successfully')
        print(f'new batch active datasets length: {len(active_path_dataset[2])}')
        print(f'new batch active datasets id:{active_path_dataset[2]}')

        # model_test(initial_model, test_dataset, inf=5)
        print(f'model prediction on new batch active datasets')
        images, masks, prob, df, image_id_selected = model_pred(model,
                                                                active_dataset_image,
                                                                active_dataset_mask,
                                                                active_path_dataset[2],
                                                                inf=5,
                                                                delta=delta)
        model_labeled = len(image_id_selected)
        human_labeled = 40 - model_labeled
        model_labeled_r.append(model_labeled)
        human_labeled_r.append(human_labeled)

        new_images = np.concatenate([initial_dataset_image, images], axis=0)
        new_masks = np.concatenate([initial_dataset_mask, masks], axis=0)
        new_dataset = dataset(new_images, new_masks, mode='train', batch_size=4)
        print(f'Concatenate datasets built for re-train model')
        print(f'Concatenate datasets length: {len(new_dataset) * 4}')

        print(f'Re-train model...')
        # if os.path.exists(r'')
        another_strategy = tf.distribute.MirroredStrategy()
        with another_strategy.scope():
            # optimizer = tf.optimizers.Adam(learning_rate=initial_learning_rate)
            model = tf.keras.models.load_model(f'checkpoints/active/unet_active_{i-1}.h5',
                                               custom_objects={'dice_loss': dice_loss,
                                                               'iou': iou,
                                                               'tree_iou': tree_iou})
            # model.compile(optimizer=optimizer, loss=[loss_fn], metrics=[iou, tree_iou])
            learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_cosine_decay, verbose=0)

            model.fit(new_dataset,
                      steps_per_epoch=len(new_dataset),
                      epochs=epochs,
                      validation_data=validation_dataset,
                      validation_steps=len(validation_dataset),
                      callbacks=[learning_rate_scheduler])

            model.save(f'checkpoints/active/unet_active_{i}.h5')
            print(f'unet_active_{i} saved!')

        initial_dataset_image = new_images
        initial_dataset_mask = new_masks
        # new model for prediction
        print(f'Active {i} prediction on test datasets')
        tree_iou, o_iou = model_test(model, test_dataset, inf=5, n=i)
        tree_ious.append(tree_iou)
        o_ious.append(o_iou)

    data = pd.DataFrame({'active epoch': np.arange(1, 8),
                         'human label sample': human_labeled_r,
                         'model label sample': model_labeled_r,
                         'tree iou': tree_ious,
                         'overall iou': o_ious})
    print(data)
    # for im, ms, p, (index, rows), ids in zip(images, masks, prob, df.iterrows(), active2_path_dataset[2]):
    #     if rows['Entropy1'] < 0.21:
    #         fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    #         axs[0].imshow(im[:, :, :3])
    #         axs[0].set_xlabel(f'image_{ids}')
    #
    #         axs[1].imshow(rgb_mask(np.argmax(ms, axis=-1)))
    #         axs[1].set_xlabel(f'mask_{ids}')
    #
    #         axs[2].imshow(rgb_mask(np.argmax(p, axis=-1)))
    #         axs[2].set_xlabel(f'prob_{ids} \n model labeled' if ids in image_id_selected else f'drop')
    #         axs[2].set_title(f"E1:{rows['Entropy1']:.4f} \n Var:{rows['Variance']:.4f}")
    #         plt.show()
        # break

