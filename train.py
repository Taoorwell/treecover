import math
from unets import U_Net
from loss import *
from dataloder import dataset, get_path

if __name__ == '__main__':
    # some parameters
    path = r'../quality/high/'
    seed = 2
    width = 256
    n_classes = 2
    train_batch_size = 4
    valid_batch_size = 10
    epochs = 300
    initial_learning_rate = 0.0001
    loss_fn = dice_loss
    rate = 0.0

    # get path and image
    image_path_train, mask_path_train, image_i_train = get_path(path=path,
                                                                mode='train',
                                                                seed=seed,
                                                                active=0)

    image_path_valid, mask_path_valid, image_i_valid = get_path(path=path,
                                                                mode='valid',
                                                                seed=seed,
                                                                active=0)
    # train and valid datasets
    train_datasets = dataset(image_path_train, mask_path_train, mode='train',
                             image_shape=(width, width), batch_size=train_batch_size)
    valid_datasets = dataset(image_path_valid, mask_path_valid, mode='valid',
                             image_shape=(width, width), batch_size=valid_batch_size)

    optimizer = tf.optimizers.Adam(learning_rate=initial_learning_rate)

    def lr_cosine_decay(e):
        cosine_decay = 0.5 * (1 + math.cos(math.pi * e / epochs))
        return initial_learning_rate * cosine_decay

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = U_Net(input_shape=(width, width, 7),
                      n_classes=n_classes,
                      rate=rate,
                      mc=False,
                      residual=True)
        model.compile(optimizer=optimizer, loss=[loss_fn], metrics=[iou, tree_iou])
    model.summary()

    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_cosine_decay, verbose=0)
    # tensorboard
    tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir=f'tb_callback_dir/unet_res_high_2',
                                                           histogram_freq=1)

    model.fit(train_datasets,
              steps_per_epoch=len(train_datasets),
              epochs=epochs,
              validation_data=valid_datasets,
              validation_steps=len(valid_datasets),
              callbacks=[learning_rate_scheduler, tensorboard_callbacks])
    model.save('checkpoints/ckpt-unet_res_high_2.h5')
    # model.save_weights('checkpoints/ckpt-unet_res_high_2')
