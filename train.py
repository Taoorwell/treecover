import math
import json
import tensorflow as tf
from utility import *
from residual_unet import *
# from matplotlib import pyplot as plt
# tf_config = json.loads(os.environ['TF_CONFIG'])
# Datasets construction


def image_dataset(path, mode, width, batch_size):
    # image path and mask path dataset
    images_path, masks_path = load_data(path, mode)
    datasets = tf.data.Dataset.from_tensor_slices((images_path, masks_path))

    # parse path into full image and then into patches
    # define parse function
    rng = tf.random.Generator.from_seed(123, alg='philox')

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

        seed = rng.make_seeds(2)[0]
        image = tf.image.stateless_random_crop(image, size=(width, width, 7), seed=seed)
        mask = tf.image.stateless_random_crop(mask, size=(width, width, 1), seed=seed)
        new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
        image = tf.image.stateless_random_flip_left_right(image, seed=new_seed)
        mask = tf.image.stateless_random_flip_left_right(mask, seed=new_seed)

        return image, mask
    datasets = datasets.map(parse_fun, num_parallel_calls=tf.data.AUTOTUNE)
    datasets = datasets.batch(batch_size)
    datasets = datasets.repeat()
    datasets = datasets.prefetch(buffer_size=tf.data.AUTOTUNE)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    datasets = datasets.with_options(options)

    return datasets


if __name__ == '__main__':
    width = 250
    batch_size = 5
    train_steps = (280 // batch_size) * 6
    valid_steps = (30 // batch_size) * 6
    epochs = 300
    # image_path, mask_path = load_data(path='../', mode='test')
    train_dataset = image_dataset(path='../', mode='train',
                                  width=width, batch_size=batch_size)
    valid_dataset = image_dataset(path='../', mode='valid',
                                  width=width, batch_size=batch_size)
    # test_dataset = image_dataset(path='../', mode='test',
    #                              width=width, batch_size=1)
    # model construction
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_res_unet(input_shape=(width, width, 7))
    # model.summary()

    # model compile
        initial_learning_rate = 0.0001

        model.compile(optimizer=tf.optimizers.Adam(learning_rate=initial_learning_rate),
                      loss=combined_log_loss, metrics=[dice])

        def lr_exponential_decay(epoch):
            # something happen
            decay_rate = 0.04
            return initial_learning_rate * math.pow(decay_rate, epoch / epochs)

        def lr_cosine_decay(epoch):
            cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / epochs))
            return initial_learning_rate * cosine_decay
        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_cosine_decay, verbose=0)

    # tensorboard
    tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir='tb_callback_dir/1m_combined_log_cosine_aug_279',
                                                           histogram_freq=1)

    model.fit(train_dataset,
              steps_per_epoch=train_steps,
              epochs=epochs,
              validation_data=valid_dataset,
              validation_steps=valid_steps,
              callbacks=[tensorboard_callbacks, learning_rate_scheduler])
    # model.save('model.h5')
    model.save_weights('checkpoints/ckpt-280')








