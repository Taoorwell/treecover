import math
from unets import U_Net
from loss import *
from dataloder import dataset

if __name__ == '__main__':
    # some parameters
    width = 256
    train_batch_size = 8
    valid_batch_size = 10
    epochs = 300
    initial_learning_rate = 0.0001
    loss_fn = dice_loss
    # train datasets
    train_datasets = dataset(path=r'../quality/', mode='train', image_shape=(width, width), batch_size=train_batch_size)
    valid_datasets = dataset(path=r'../quality/', mode='valid', image_shape=(width, width), batch_size=valid_batch_size)

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
        model = U_Net(input_shape=(width, width, 7), n_classes=1, recurrent=False, residual=True, attention=True)
        model.compile(optimizer=optimizer, loss=[loss_fn], metrics=[iou])
    model.summary()

    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_cosine_decay, verbose=0)
    # tensorboard
    tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir='tb_callback_dir/unet_res_att_300',
                                                           histogram_freq=1)

    model.fit(train_datasets,
              steps_per_epoch=len(train_datasets),
              epochs=epochs,
              validation_data=valid_datasets,
              validation_steps=len(valid_datasets),
              callbacks=[learning_rate_scheduler, tensorboard_callbacks])
    # model.save('model.h5')
    model.save_weights('checkpoints/ckpt-unet_res_att_300')
