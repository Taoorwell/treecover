# import math
from utility import rgb_mask
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
# import pandas as pd
import numpy as np
from loss import dice_loss, tree_iou
from dataloder import get_path, get_image, dataset
from unets import U_Net
from test_pre import predict_on_array


def iou(y_true, y_pred):
    numerator = np.sum(y_true * y_pred)
    denominator = np.sum(y_true + y_pred)
    return numerator / (denominator - numerator)


def get_active_image_mask_array_list(path_dataset):
    active_dataset_image, active_dataset_mask = np.array(list(map(get_image, path_dataset[0]))),\
                                                np.array(list(map(get_image, path_dataset[1])))
    return active_dataset_image, active_dataset_mask


# initial_dataset training for model 1
def initial_model_train():
    optimizer = tf.optimizers.Adam(learning_rate=initial_learning_rate)

    def lr_cosine_decay(e):
        cosine_decay = 0.5 * (1 + np.cos(np.pi * e / epochs))
        return initial_learning_rate * cosine_decay

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # Monte Carlo Dropout
        model = U_Net(input_shape=(256, 256, 7), dropout=.1, n_classes=n_classes, residual=True)
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

    # model.save_weights('checkpoints/active/ckpt-unet_active_1')
    model.save(r'checkpoints/active/ckpt-unet_active_1.h5')
    return model


def model_test(model, dataset, inf):
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
        acc1.append(acc_iou_1)
        acc2.append(acc_iou_2)

    df = pd.DataFrame({'N': test_path_dataset[2],
                       'tree_iou': acc1,
                       'o_iou': acc2})
    print(df)
    print(np.mean(acc1), np.mean(acc2))
    with pd.ExcelWriter(r'../results/r.xlsx', mode='a') as writer:
        df.to_excel(writer, sheet_name='active_1')


def model_pred(model, images, masks, images_id, inf, delta):
    def uncertainty(outputs):
        eps = 1e-15
        # first
        a = -(outputs[..., 0] * np.log2(outputs[..., 0] + eps) + outputs[..., 1] * np.log2(outputs[..., 1] + eps))
        E1 = np.mean(a, axis=0)
        E1 = np.mean(E1)
        # second
        # b1, b2 = np.mean(outputs[..., 0], axis=0), np.mean(outputs[..., 1], axis=0)
        # E2 = -(b1 * np.log2(b1+eps) + b2 * np.log2(b2+eps))
        # E2 = np.mean(E2)
        # # third
        v1, v2 = np.var(outputs[..., 0], axis=0), np.var(outputs[..., 1], axis=0)
        v = v1 + v2
        v = np.sum(v)
        return E1, v

    prob, entropy1, variance = [], [], []
    for im, ms, ids in zip(images, masks, images_id):
        outputs = np.zeros((inf,) + im.shape[:-1] + (2,), dtype=np.float32)
        for i in range(inf):
            mask_prob = predict_on_array(model=model,
                                         arr=im,
                                         in_shape=(256, 256, 7),
                                         out_bands=2,
                                         stride=100,
                                         batchsize=20)
            outputs[i] = mask_prob
        e1, v = uncertainty(outputs=outputs)
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
        # entropy2.append(e2)
        variance.append(v)
        prob.append((np.mean(outputs, axis=0) > 0.5) * 1)

    df = pd.DataFrame({'Image_id': images_id,
                       'Entropy1': entropy1,
                       'Variance': variance})
    print(df)
    image_id_selected = np.array(images_id)[np.array(entropy1) < delta]
    print(f'number of high: {len(image_id_selected)}, '
          f'high confidence index:{image_id_selected}')
    # replace mask from model prediction
    # masks[np.array(entropy1) < delta] = np.array(prob)[np.array(entropy1) < delta]
    # plot something
    return images, masks, prob, image_id_selected

# put new images and masks with previous datasets together.


if __name__ == '__main__':
    # some parameters
    seed = 2
    path = r'../quality/high'
    initial_learning_rate = 0.0001
    epochs = 200
    n_classes = 2
    loss_fn = dice_loss

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
    validation_dataset = dataset(validation_path_dataset[0], validation_path_dataset[1], mode='valid', batch_size=1)
    test_dataset = dataset(test_path_dataset[0], test_path_dataset[1], mode='test', batch_size=1)
    print(f'initial, validation and test tensorflow datasets loading successfully')
    # Get active 2 path dataset
    active2_path_dataset = get_path(path=path,
                                    mode='train',
                                    seed=seed,
                                    active=2)
    active2_dataset_image, active2_dataset_mask = get_active_image_mask_array_list(active2_path_dataset)
    print(f'new active datasets loading successfully')

    model = initial_model_train()

    # trained model loading
    # initial_model = U_Net(input_shape=(256, 256, 7), dropout=0.9, n_classes=n_classes, residual=True)
    # initial_model.load_weights(r'checkpoints/active/ckpt-unet_active_1')
    # initial_model = tf.keras.models.load_model(r'checkpoints/active/unet_active_1.h5',
    #                                            custom_objects={'dice_loss': dice_loss,
    #                                                            'iou': iou,
    #                                                            'tree_iou': tree_iou})
    # initial_model.summary()
    # print('model loaded successfully')
    # # model validation part
    # for im, ms in validation_dataset:
    #     # print(im.shape)
    #     outputs = np.zeros((10, ) + ms.shape[1:], dtype=np.float32)
    #     for i in range(10):
    #         output = initial_model.predict(x=im)
    #         print(output[0, 0, 0, :])
    #         outputs[i] = output
    #     v1, v2 = np.var(outputs[..., 0], axis=0), np.var(outputs[..., 1], axis=0)
    #     v = v1 + v2
    #     v = np.sum(v)
    #     print(v)
    # model test
    # model_test(initial_model, test_dataset, inf=10)
    # model prediction on active2 datasets
    # images, masks, prob, image_id_selected = model_pred(initial_model,
    #                                                     active2_dataset_image,
    #                                                     active2_dataset_mask,
    #                                                     active2_path_dataset[2],
    #                                                     3,
    #                                                     0.05)
    # for im, ms, p, ids in zip(images, masks, prob, active2_path_dataset[2]):
    #     fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    #     axs[0].imshow(im[:, :, :3])
    #     axs[0].set_xlabel(f'image_{ids}')
    #
    #     axs[1].imshow(rgb_mask(np.argmax(ms, axis=-1)))
    #     axs[1].set_xlabel(f'mask_{ids}')
    #
    #     axs[2].imshow(rgb_mask(np.argmax(p, axis=-1)))
    #     axs[2].set_xlabel(f'prob_{ids}')
    #     axs[2].set_title(f'model labeled' if ids in image_id_selected else f'drop')
    #     plt.show()

# if __name__ == '__main__':
#     # parameter define
#     width = 256
#     path = r'../quality/high'
#     n_classes = 2
#     train_batch_size = 4
#     valid_batch_size = 10
#     n_inference = 10
#     eps = 10e-10
#     initial_learning_rate = 0.0001
#     epochs = 200
#     # validation datasets always
#     image_path_valid, mask_path_valid, image_i_valid = get_path(path=path,
#                                                                 mode='valid',
#                                                                 seed=1,
#                                                                 active=0)
#     valid_datasets = dataset(image_path_valid,
#                              mask_path_valid,
#                              mode='valid',
#                              image_shape=(width, width),
#                              batch_size=valid_batch_size,
#                              n_classes=n_classes)
#     print('Validation datasets loading')
#     print(len(image_i_valid), image_i_valid)
#     # active datasets preparation for training
#     image_path_active1, mask_path_active1, image_id_active1 = get_path(path=path,
#                                                                        mode='train',
#                                                                        seed=1,
#                                                                        active=1)
#
#     active1_datasets = dataset(image_path_active1,
#                                mask_path_active1,
#                                mode='train',
#                                image_shape=(width, width),
#                                batch_size=train_batch_size,
#                                n_classes=n_classes)
#     print('active train datasets loading')
#     print(len(image_id_active1), image_id_active1)
#     # train model
#
#     optimizer = tf.optimizers.Adam(learning_rate=initial_learning_rate)
#
#     def lr_cosine_decay(e):
#         cosine_decay = 0.5 * (1 + math.cos(math.pi * e / epochs))
#         return initial_learning_rate * cosine_decay
#
#
#     strategy = tf.distribute.MirroredStrategy()
#     with strategy.scope():
#         # model = build_res_unet(input_shape=(width, width, 7))
#         model = U_Net(input_shape=(width, width, 7),
#                       n_classes=n_classes,
#                       dropout=.5,
#                       residual=True)
#         model.compile(optimizer=optimizer, loss=[dice_loss], metrics=[iou, tree_iou])
#     model.summary()
#
#     learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_cosine_decay, verbose=0)
#     # tensorboard
#     tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir='tb_callback_dir/active/unet_res_active1',
#                                                            histogram_freq=1)
#     print('active seed model training')
#     model.fit(active1_datasets,
#               steps_per_epoch=len(active1_datasets),
#               epochs=epochs,
#               validation_data=valid_datasets,
#               validation_steps=len(valid_datasets),
#               callbacks=[learning_rate_scheduler, tensorboard_callbacks])
#     # model.save('model.h5')
#     model.save_weights('checkpoints/active/ckpt-unet_res_active1')
#     # model.load_weights(r'checkpoints/ckpt-unet_res_softmax_dice_4_500')
#     print('active seed model save successfully')
#
#     # uncertainty datasets preparation
#     image_path_active2, mask_path_active2, image_id_active2 = get_path(path=path,
#                                                                        mode='train',
#                                                                        seed=1,
#                                                                        active=2)
#     active2_datasets = dataset(image_path_active2,
#                                mask_path_active2,
#                                mode='test',
#                                image_shape=(256, 256),
#                                batch_size=1,
#                                n_classes=2)
#     print(len(image_id_active2), image_id_active2)
#     print('Uncertainty prediction')
#
#     e1, e2, var = [], [], []
#     # model prediction
#     for image_arr, mask_arr in active2_datasets:
#         image_arr, mask_arr = image_arr[0], mask_arr[0]
#         # print(image_arr.shape, mask_arr.shape)
#         outputs = np.zeros((n_inference, ) + mask_arr.shape, dtype=np.float32)
#         for i in range(n_inference):
#             output, _ = predict_on_array(model=model,
#                                          arr=image_arr,
#                                          in_shape=(256, 256, 7),
#                                          out_bands=2,
#                                          stride=200,
#                                          batchsize=20,
#                                          augmentation=False,
#                                          verbose=False,
#                                          report_time=True)
#             outputs[i] = output
#         # output prediction uncertainty estimation
#         # categorical first cross entropy
#         # first
#         a = -(outputs[..., 0] * np.log2(outputs[..., 0]+eps) + outputs[..., 1] * np.log2(outputs[..., 1]+eps))
#         E1 = np.mean(a, axis=0)
#         # print(E1.shape)
#         E1 = np.mean(E1)
#         # print(E1)
#
#         # second
#         b1, b2 = np.mean(outputs[..., 0], axis=0), np.mean(outputs[..., 1], axis=0)
#         E2 = -(b1 * np.log2(b1+eps) + b2 * np.log2(b2+eps))
#         # print(E2.shape)
#         E2 = np.mean(E2)
#         # print(E2)
#
#         # third
#         v1, v2 = np.var(outputs[..., 0], axis=0), np.var(outputs[..., 1], axis=0)
#         v = v1 + v2
#         # print(v.shape)
#         v = np.sum(v)
#         # print(v)
#
#         e1.append(E1)
#         e2.append(E2)
#         var.append(v)
#         # break
#     df = pd.DataFrame({'ID': image_id_active2, 'Entropy1': e1, 'Entropy2': e2, 'Variance': var})
#     print(df)
    # df.to_excel('../results/train_1.xlsx')
