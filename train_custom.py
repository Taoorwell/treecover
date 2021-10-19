import math
import tensorflow as tf
from unets import U_Net
from loss import *
from dataloder import dataset

if __name__ == '__main__':
    # some parameters
    width = 256
    batch_size = 6
    epochs = 50
    initial_learning_rate = 0.0001
    loss_fn = combined_loss
    # train datasets
    train_datasets = dataset(path=r'../quality/', mode='train', image_shape=(width, width), batch_size=batch_size)
    valid_datasets = dataset(path=r'../quality/', mode='valid', image_shape=(width, width), batch_size=batch_size)

    optimizer = tf.optimizers.Adam(learning_rate=initial_learning_rate)
    # def lr_exponential_decay(e):
    #     # something happen
    #     decay_rate = 0.04
    #     return initial_learning_rate * math.pow(decay_rate, e / epochs)

    def lr_cosine_decay(e):
        cosine_decay = 0.5 * (1 + math.cos(math.pi * e / epochs))
        return initial_learning_rate * cosine_decay

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # model = build_res_unet(input_shape=(width, width, 7))
        model = U_Net(input_shape=(width, width, 7), n_classes=1, recurrent=True, residual=True, attention=True)
        model.compile(optimizer=optimizer, loss=[loss_fn], metrics=[iou])
    model.summary()

    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_cosine_decay, verbose=0)
    # tensorboard
    # tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir='tb_callback_dir/1m_combined_log_cosine_aug_279',
    #                                                        histogram_freq=1)

    model.fit(train_datasets,
              steps_per_epoch=len(train_datasets),
              epochs=epochs,
              validation_data=valid_datasets,
              validation_steps=len(valid_datasets),
              callbacks=[learning_rate_scheduler])
    # model.save('model.h5')
    # model.save_weights('checkpoints/ckpt-280')
