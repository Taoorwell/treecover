import math
import json
import tensorflow as tf
from utility import *
from residual_unet import *
# from matplotlib import pyplot as plt
# tf_config = json.loads(os.environ['TF_CONFIG'])
# Datasets construction


def image_dataset(path, mode, width, batch_size):
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
        image.set_shape(image.shape)
        mask.set_shape(mask.shape)
        return image, mask
    datasets = datasets.map(parse_fun)

    def augment(image_mask, seed):
        image, mask = image_mask
        image = tf.image.stateless_random_crop(image, size=(width, width, 7), seed=seed)
        mask = tf.image.stateless_random_crop(mask, size=(width, width, 1), seed=seed)
        if np.random.uniform((0)) > 0.5:
            new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
            image = tf.image.stateless_random_flip_left_right(image, seed=new_seed)
            mask = tf.image.stateless_random_flip_left_right(mask, seed=new_seed)
        return image, mask

    rng = tf.random.Generator.from_seed(123, alg='philox')

    def f(x, y):
        seed = rng.make_seeds(2)[0]
        image, label = augment((x, y), seed)
        return image, label

    datasets = datasets.map(f, num_parallel_calls=tf.data.AUTOTUNE)
    datasets = datasets.batch(batch_size)
    datasets = datasets.repeat()

    datasets = datasets.prefetch(buffer_size=tf.data.AUTOTUNE)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    datasets = datasets.with_options(options)

    return datasets


if __name__ == '__main__':
    width = 256
    batch_size = 3
    train_steps = (270 // batch_size) * 4
    valid_steps = (25 // batch_size) * 4
    epochs = 100
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
        initial_learning_rate = 0.0001

        model.compile(optimizer=tf.optimizers.Adam(learning_rate=initial_learning_rate),
                      loss=combined_log_loss, metrics=[Iou])

        def lr_exponential_decay(epoch):
            # something happen
            decay_rate = 0.04
            return initial_learning_rate * math.pow(decay_rate, epoch / epochs)

        def lr_cosine_decay(epoch):
            cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / epochs))
            return initial_learning_rate * cosine_decay

    # tensorboard
    tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir='tb_callback_dir/1m_combined_log_cosine_aug',
                                                           histogram_freq=1)
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_cosine_decay, verbose=1)

    model.fit(train_dataset,
              steps_per_epoch=train_steps,
              epochs=epochs,
              validation_data=valid_dataset,
              validation_steps=valid_steps,
              callbacks=[tensorboard_callbacks, learning_rate_scheduler])
    # model.save('model.h5')
    model.save_weights('checkpoints/ckpt-1m_combined_log_cosine_aug')








