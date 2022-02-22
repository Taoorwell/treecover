import os
from utility import get_image
import pandas as pd
import tensorflow as tf
import numpy as np
from loss import dice_loss, iou, tree_iou
from dataloder import get_path, dataset
from unets import U_Net
from test_pre import predict_on_array


def get_active_image_mask_array_list(image_path, mask_path):
    images, masks = np.array(list(map(get_image, image_path))),\
                    np.array(list(map(get_image, mask_path)))
    return images, masks


# initial_dataset training for model 1
def lr_cosine_decay(e):
    initial_learning_rate = 0.0001
    cosine_decay = 0.5 * (1 + np.cos(np.pi * e / epochs))
    return initial_learning_rate * cosine_decay


def initial_model_train(initial_dataset, validation_dataset):
    if os.path.exists(r'checkpoints/active/high/unet_active_1'):
        model = tf.keras.models.load_model(r'checkpoints/active/high/unet_active_1',
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
                  verbose=0,
                  callbacks=[learning_rate_scheduler])
        model.save(r'checkpoints/active/high/unet_active_1')
    return model


def retrain_model(new_dataset, validation_dataset, i):
    optimizer = tf.optimizers.Adam(learning_rate=initial_learning_rate)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # Monte Carlo Dropout
        model = U_Net(input_shape=(256, 256, 7), n_classes=n_classes, rate=.1, mc=True, residual=True)
        model.compile(optimizer=optimizer, loss=[loss_fn], metrics=[iou, tree_iou])
    model.compile(optimizer=model.optimizer, loss=model.loss, metrics=[iou, tree_iou])
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_cosine_decay, verbose=0)
    # tensorboard
    tensorboard_callbacks = tf.keras.callbacks.TensorBoard(
        log_dir=f'tb_callback_dir/active/high/percent/decay/unet_active_{i}',
        histogram_freq=1)
    model.fit(new_dataset,
              steps_per_epoch=len(new_dataset),
              epochs=epochs,
              validation_data=validation_dataset,
              validation_steps=len(validation_dataset),
              verbose=0,
              callbacks=[learning_rate_scheduler, tensorboard_callbacks]
              )
    model.save(f'checkpoints/active/high/percent/decay/unet_active_{i}')
    print(f'unet_active_{i} saved!')

    return model


def model_test(model, images, masks, images_ids, inf, n):
    acc1, acc2 = [], []
    for im, ms in zip(images, masks):
        image_arr, mask_arr = im, ms
        # print(image_arr.shape, type(image_arr))
        outputs = np.zeros((inf, ) + mask_arr.shape, dtype=np.float32)
        for f in range(inf):
            output_1 = predict_on_array(model=model,
                                        arr=image_arr,
                                        in_shape=(256, 256, 7),
                                        out_bands=2,
                                        stride=200,
                                        batchsize=20,
                                        augmentation=True)
            outputs[f] = output_1
        # print(outputs.shape)
        outputs = np.mean(outputs, axis=0)
        # print(outputs.shape)
        acc_iou_1 = iou(mask_arr[:, :, 1], outputs[:, :, 1])
        acc_iou_2 = iou(mask_arr, outputs)
        acc1.append(acc_iou_1.numpy())
        acc2.append(acc_iou_2.numpy())

    df = pd.DataFrame({'N': images_ids,
                       'tree_iou': acc1,
                       'o_iou': acc2})
    print(df)
    mean_tree_iou, mean_o_iou = np.mean(acc1), np.mean(acc2)
    print(mean_tree_iou, mean_o_iou)
    with pd.ExcelWriter(r'checkpoints/active/high/decay/r.xlsx',
                        engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=f'active_{n}')
    return mean_tree_iou, mean_o_iou


def model_test_1(model, images, masks, inf):
    true_masks_cat, pre_masks_cat = [], []
    for im, ms in zip(images, masks):
        image_arr, mask_arr = im, ms
        outputs = np.zeros((inf, ) + mask_arr.shape, dtype=np.float32)
        for f in range(inf):
            output_1 = predict_on_array(model=model,
                                        arr=image_arr,
                                        in_shape=(256, 256, 7),
                                        out_bands=2,
                                        stride=200,
                                        batchsize=20,
                                        augmentation=True)
            outputs[f] = output_1
        outputs = np.mean(outputs, axis=0)
        true_masks_cat.append(mask_arr)
        pre_masks_cat.append(outputs)
    true_mask = np.concatenate(true_masks_cat, axis=0)
    pre_mask = np.concatenate(pre_masks_cat, axis=0)
    mean_tree_iou = iou(true_mask[:, :, 1], pre_mask[:, :, 1]).numpy()
    mean_o_iou = iou(true_mask, pre_mask).numpy()

    return mean_tree_iou, mean_o_iou


def model_pred(model, images, masks, images_ids, inf, delta):
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
    t_c, o_c = [], []
    for im, ms in zip(images, masks):
        outputs = np.zeros((inf,) + im.shape[:-1] + (2,), dtype=np.float32)
        for f in range(inf):
            mask_prob = predict_on_array(model=model,
                                         arr=im,
                                         in_shape=(256, 256, 7),
                                         out_bands=2,
                                         stride=100,
                                         batchsize=20,
                                         augmentation=False)
            outputs[f] = mask_prob
        e1, e2, v = uncertainty(outputs=outputs)

        entropy1.append(e1)
        entropy2.append(e2)
        variance.append(v)
        prob.append((np.mean(outputs, axis=0) > 0.5) * 1)

        t_c.append(iou(ms[:, :, 1], np.mean(outputs, axis=0)[:, :, 1]).numpy())
        o_c.append(iou(ms, np.mean(outputs, axis=0)).numpy())

    df = pd.DataFrame({'Image_id': images_ids,
                       'Entropy1': entropy1,
                       'Entropy2': entropy2,
                       'Variance': variance,
                       'Tree_iou': t_c,
                       'O_iou': o_c})
    print(df)
    if delta >= 10:
        print(f'first {0.01*delta:.2%} Percentage as model labelled masks')

        image_id_selected = np.argsort(np.array(entropy1))[:int(len(entropy1)*delta*0.01)]
        print(f'number of high: {len(image_id_selected)}, '
              f'high confidence index:{np.array(images_ids)[image_id_selected]}')
        # replace mask from model prediction
        masks[image_id_selected] = np.array(prob)[image_id_selected]
        print(f'mask replacing finished!')
    else:
        image_id_selected = np.array(images_ids)[np.array(entropy1) < delta]
        print(f'number of high: {len(image_id_selected)}, '
              f'high confidence index:{image_id_selected}')
        # replace mask from model prediction
        masks[np.array(entropy1) < delta] = np.array(prob)[np.array(entropy1) < delta]
        print(f'mask replacing finished!')

    return images, masks, prob, df, image_id_selected

# put new images and masks with previous datasets together.


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # some parameters
    seed = 2
    path = r'../quality/high/'
    initial_learning_rate = 0.0001
    epochs = 100
    n_classes = 2
    loss_fn = dice_loss
    # deltas = [0.1, 0.08, 0.06, 0.04, 0.02, 0.01]
    # deltas = [0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    deltas = [20, 30, 40, 50, 60, 70, 80]
    # initial datasets, validation and test datasets
    initial_image_path, initial_mask_path, initial_image_id = get_path(path=path,
                                                                       mode='train',
                                                                       seed=seed,
                                                                       active=1)

    valid_image_path, valid_mask_path, valid_image_id = get_path(path=path,
                                                                 mode='valid',
                                                                 seed=seed,
                                                                 active=0)

    test_image_path, test_mask_path, test_image_id = get_path(path=r'../quality/high/',
                                                              mode='test',
                                                              seed=seed,
                                                              active=0)

    print(f'initial datasets length: {len(initial_image_id)}')
    print('initial datasets id:')
    print(initial_image_id)

    # initial, validation and test tensorflow datasets
    initial_dataset_image, initial_dataset_mask = get_active_image_mask_array_list(initial_image_path,
                                                                                   initial_mask_path)
    print(f'initial dataset image and mask loading successfully')

    initial_dataset = dataset(initial_dataset_image, initial_dataset_mask, mode='train', batch_size=4)
    validation_dataset = dataset(valid_image_path, valid_mask_path, mode='valid', batch_size=10)
    # test_dataset = dataset(test_path_dataset[0], test_path_dataset[1], mode='test', batch_size=1)
    test_dataset_image, test_dataset_mask = get_active_image_mask_array_list(test_image_path, test_mask_path)
    print(f'initial, validation and test tensorflow datasets loading successfully')

    tree_ious, o_ious = [], []

    model = initial_model_train(initial_dataset, validation_dataset)
    print('initial model loaded successfully')
    print('initial model prediction on test datasets')
    i_tree_iou, i_o_iou = model_test(model, test_dataset_image, test_dataset_mask, test_image_id, inf=5, n=1)
    tree_ious.append(i_tree_iou)
    o_ious.append(i_o_iou)
    model_labeled_r, human_labeled_r = [0], [40]
    for i, de in zip(np.arange(2, 8), deltas):
        print(f'{i-1} Active learning starting! ')
        # Get active 2 path dataset
        active_image_path, active_mask_path, active_image_id = get_path(path=path,
                                                                        mode='train',
                                                                        seed=seed,
                                                                        active=i)
        active_dataset_image, active_dataset_mask = get_active_image_mask_array_list(active_image_path,
                                                                                     active_mask_path)

        print(f'{i} new batch active datasets loading successfully')
        print(f'new batch active datasets length: {len(active_image_id)}')
        print(f'new batch active datasets id:{active_image_id}')

        # model_test(initial_model, test_dataset, inf=5)
        print(f'model prediction on new batch active datasets')
        images, masks, prob, df, image_id_selected = model_pred(model,
                                                                active_dataset_image,
                                                                active_dataset_mask,
                                                                active_image_id,
                                                                inf=5,
                                                                delta=de)
        with pd.ExcelWriter(r'checkpoints/active/high/percent/decay/r.xlsx', engine='openpyxl', mode='a',
                            if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=f'active_e_{i}')

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
        if os.path.exists(f'checkpoints/active/high/percent/decay/unet_active_{i}'):
            model = tf.keras.models.load_model(f'checkpoints/active/high/percent/decay/unet_active_{i}',
                                               custom_objects={'dice_loss': dice_loss,
                                                               'iou': iou,
                                                               'tree_iou': tree_iou},
                                               compile=True)
        else:
            model = retrain_model(new_dataset, validation_dataset, i)
            # pass
            # if i == 2:
            #     model = tf.keras.models.load_model(f'checkpoints/active/high/unet_active_{i-1}',
            #                                        custom_objects={'dice_loss': dice_loss,
            #                                                        'iou': iou,
            #                                                        'tree_iou': tree_iou},
            #                                        compile=True)
            # else:
            #     model = tf.keras.models.load_model(f'checkpoints/active/high/decay/unet_active_{i-1}',
            #                                        custom_objects={'dice_loss': dice_loss,
            #                                                        'iou': iou,
            #                                                        'tree_iou': tree_iou},
            #                                        compile=True)
            #
            # model.compile(optimizer=model.optimizer, loss=model.loss, metrics=[iou, tree_iou])
            # learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_cosine_decay, verbose=0)
            # # tensorboard
            # tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir=
            #                                                        f'tb_callback_dir/active/high/decay/unet_active_{i}',
            #                                                        histogram_freq=1)
            #
            # model.fit(new_dataset,
            #           steps_per_epoch=len(new_dataset),
            #           epochs=epochs,
            #           validation_data=validation_dataset,
            #           validation_steps=len(validation_dataset),
            #           verbose=0,
            #           callbacks=[learning_rate_scheduler, tensorboard_callbacks]
            #           )
            #
            # model.save(f'checkpoints/active/high/decay/unet_active_{i}')
            # print(f'unet_active_{i} saved!')

        initial_dataset_image = new_images
        initial_dataset_mask = new_masks
        # new model for prediction
        print(f'Active {i} prediction on test datasets')
        tree_iou_1, o_iou_1 = model_test_1(model, test_dataset_image, test_dataset_mask, inf=5)
        tree_ious.append(tree_iou_1)
        o_ious.append(o_iou_1)

    data = pd.DataFrame({'active epoch': np.arange(1, 8),
                         'human label sample': human_labeled_r,
                         'model label sample': model_labeled_r,
                         'tree iou': tree_ious,
                         'overall iou': o_ious,
                         'delta': [20, 30, 40, 50, 60, 70, 80]})
    with pd.ExcelWriter(r'checkpoints/active/high/percent/decay/r.xlsx',
                        engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        data.to_excel(writer, sheet_name=f'active_data_decay')
    print(data)

