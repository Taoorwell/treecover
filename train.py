import tensorflow as tf
from utility import *
from residual_unet import *
from matplotlib import pyplot as plt


# Datasets construction
def image_dataset(path, mode, width, batch_size):
    # image path and mask path dataset
    images_path, masks_path = load_data(path, mode)
    datasets = tf.data.Dataset.from_tensor_slices((images_path, masks_path))

    # parse path into full image and then into patches
    # define parse function
    def parse_fun(x, y):
        def f(x, y):
            x1 = x.decode()
            y1 = y.decode()

            x2 = get_raster(x1)
            y2 = get_raster(y1)
            return x2, y2

        image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
        image.set_shape([width, width, 7])
        mask.set_shape([width, width, 1])
        return image, mask
    datasets = datasets.map(parse_fun)
    datasets = datasets.batch(batch_size)
    datasets = datasets.repeat()
    return datasets


if __name__ == '__main__':
    width = 333
    batch_size = 3
    train_steps = 270 // batch_size
    valid_steps = 25 // batch_size
    # image_path, mask_path = load_data(path='../', mode='test')
    train_dataset = image_dataset(path='../', mode='train',
                                  width=width, batch_size=batch_size)
    valid_dataset = image_dataset(path='../', mode='valid',
                                  width=width, batch_size=batch_size)
    # test_dataset = image_dataset(path='../', mode='test',
    #                              width=width, batch_size=1)
    # model restore
    # model = build_res_unet(input_shape=(width, width, 7))
    # model.load_weights('checkpoints/ckpt')

    # for i, (image, mask) in enumerate(test_dataset):
    #     mask_pred = model.predict(image)
    #     acc = dice(mask, mask_pred)
    #     mask_pred = (model.predict(image)[0] > 0.5) * 1
    #     image_id = image_path[i].split('_')[-1].split('.')[0]
    #
    #     plt.subplot(131)
    #     plt.imshow(image.numpy()[0][:, :, [4, 3, 2]])
    #     plt.xlabel('Image_{}'.format(image_id))
    #     plt.xticks([])
    #     plt.yticks([])
    #
    #     plt.subplot(132)
    #     plot_mask(mask.numpy()[0][:, :, 0])
    #     plt.xlabel('mask_{}'.format(image_id))
    #     plt.xticks([])
    #     plt.yticks([])
    #
    #     plt.subplot(133)
    #     plot_mask(mask_pred[:, :, 0])
    #     plt.xlabel('mask_{}_pre'.format(image_id))
    #     plt.xticks([])
    #     plt.yticks([])
    #
    #     plt.title('Accuracy:{:.2%}'.format(acc))
    #     # plt.show()
    #     plt.savefig('pre/treecover/Image_{}_pre'.format(image_id))
    #     print('finish: {}'.format(i))
    #     if i == 34:
    #         break
    #
    # val_dataset = image_dataset(path='../..', mode='eval',
    #                             width=width, batch_size=batch_size)

    # model construction
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_res_unet(input_shape=(width, width, 7))
    # model.summary()

    # model compile
        initial_learning_rate = 0.01
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr_schedule),
                      loss=dice_loss, metrics=[dice])

    # tensorboard
    tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir='tb_callback_dir/1m_cedice_lr_decay',
                                                           histogram_freq=1)

    model.fit(train_dataset,
              steps_per_epoch=train_steps,
              epochs=100,
              validation_data=valid_dataset,
              validation_steps=valid_steps,
              callbacks=[tensorboard_callbacks])
    # model.save('model.h5')
    model.save_weights('checkpoints/ckpt-1m_cedice_lr_decay')








