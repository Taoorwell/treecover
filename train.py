# import math
import tensorflow as tf
from residual_unet import build_res_unet, combined_log_loss, dice_loss, cross_entropy, iou
from dataloder import dataset
from tqdm import tqdm
# Datasets construction
# def image_dataset(path, mode, width, batch_size):
#     # image path and mask path dataset
#     images_path, masks_path = load_data(path, mode)
#     datasets = tf.data.Dataset.from_tensor_slices((images_path, masks_path))
#
#     # parse path into full image and then into patches
#     # define parse function
#     rng = tf.random.Generator.from_seed(123, alg='philox')
#
#     def parse_fun(x, y):
#         def f(x, y):
#             x1 = x.decode()
#             y1 = y.decode()
#
#             x2 = get_raster(x1)
#             y2 = get_raster(y1)
#             return x2, y2
#
#         image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
#         image.set_shape(image.shape)
#         mask.set_shape(mask.shape)
#
#         seed = rng.make_seeds(2)[0]
#         image = tf.image.stateless_random_crop(image, size=(width, width, 7), seed=seed)
#         mask = tf.image.stateless_random_crop(mask, size=(width, width, 1), seed=seed)
#         new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
#         image = tf.image.stateless_random_flip_left_right(image, seed=new_seed)
#         mask = tf.image.stateless_random_flip_left_right(mask, seed=new_seed)
#
#         return image, mask
#     datasets = datasets.map(parse_fun, num_parallel_calls=tf.data.AUTOTUNE)
#     datasets = datasets.batch(batch_size)
#     datasets = datasets.repeat()
#     datasets = datasets.prefetch(buffer_size=tf.data.AUTOTUNE)
#
#     options = tf.data.Options()
#     options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
#     datasets = datasets.with_options(options)
#
#     return datasets


if __name__ == '__main__':
    # some parameters
    width = 256
    batch_size = 5
    epochs = 10
    initial_learning_rate = 0.0001
    # train dataloader
    # train_dataloader = Dataloader(path='../', mode='train', image_shape=(width, width, 7), batch_size=batch_size)
    # valid_dataloader = Dataloader(path='../', mode='valid', image_shape=(width, width, 7), batch_size=batch_size)
    train_datasets = dataset(path='../', mode='train', image_shape=(width, width), batch_size=batch_size)
    valid_datasets = dataset(path='../', mode='valid', image_shape=(width, width), batch_size=batch_size)
    # model compile
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_res_unet(input_shape=(width, width, 7))
        optimizer = tf.optimizers.Adam(learning_rate=initial_learning_rate)
    # model.compile(optimizer=tf.optimizers.Adam(learning_rate=initial_learning_rate),
    #               loss=combined_log_loss, metrics=[dice])

    # def lr_exponential_decay(e):
    #     # something happen
    #     decay_rate = 0.04
    #     return initial_learning_rate * math.pow(decay_rate, e / epochs)
    #
    # def lr_cosine_decay(e):
    #     cosine_decay = 0.5 * (1 + math.cos(math.pi * e / epochs))
    #     return initial_learning_rate * cosine_decay
    # learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_cosine_decay, verbose=0)
    dist_train_datasets = strategy.experimental_distribute_dataset(train_datasets)
    dist_valid_datasets = strategy.experimental_distribute_dataset(valid_datasets)

    # @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = dice_loss(y, logits)
            loss_value = tf.nn.compute_average_loss(loss, global_batch_size=batch_size)
        # gradients and optimizer
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc = tf.reduce_mean(iou(y, logits))
        return loss_value, train_acc

    def valid_step(x, y):
        predictions = model(x, training=False)
        valid_loss = tf.reduce_mean(dice_loss(y, predictions))
        valid_acc = tf.reduce_mean(iou(y, predictions))
        return valid_loss, valid_acc

    @tf.function
    def dist_train_step(x, y):
        per_replica_loss, _ = strategy.run(train_step, args=(x, y))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None), _

    @tf.function
    def dist_valid_step(x, y):
        return strategy.run(valid_step, args=(x, y))

    for epoch in range(epochs):
        train_acc, train_loss, valid_loss, valid_acc = [], [], [], []
        # optimizer = tf.optimizers.Adam(learning_rate=lr_cosine_decay(epoch))
        for batch_image, batch_mask in tqdm(dist_train_datasets):
            batch_loss, _ = dist_train_step(batch_image, batch_mask)
            train_loss.append(batch_loss)
            train_acc.append(_)
            # print('train loss: {}, train acc:{}'.format(batch_loss, _))
        for v_batch_image, v_batch_mask in tqdm(dist_valid_datasets):
            valid_los, valid_ac = dist_valid_step(v_batch_image, v_batch_mask)
            valid_loss.append(valid_los)
            valid_acc.append(valid_ac)
            # print('valid loss: {}, valid acc:{}'.format(valid_los, valid_ac))
        print('Epoch: {}, Train loss:{}, acc:{}, Valid loss:{}, acc:{}'.format(epoch+1,
                                                                               tf.reduce_mean(train_loss),
                                                                               tf.reduce_mean(train_acc),
                                                                               tf.reduce_mean(valid_loss),
                                                                               tf.reduce_mean(valid_acc)))
        # validation datasets
        # for batch_image_valid, batch_mask_valid in valid_dataloader.load_batch(batch_size=batch_size):
        #     val_logits = model(batch_image_valid, training=False)
        #     valid_acc.append(dice(batch_mask_valid, val_logits))
        # print('Epoch:{}, train acc:{}, train loss:{}, valid acc:{}'.format(epoch+1,
        #                                                                    tf.reduce_mean(train_acc),
        #                                                                    tf.reduce_mean(train_loss),
        #                                                                    tf.reduce_mean(valid_acc)))

    # tensorboard
    # tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir='tb_callback_dir/1m_combined_log_cosine_aug_279',
    #                                                        histogram_freq=1)

    # model.fit(train_dataset,
    #           steps_per_epoch=train_steps,
    #           epochs=epochs,
    #           validation_data=valid_dataset,
    #           validation_steps=valid_steps,
    #           callbacks=[tensorboard_callbacks, learning_rate_scheduler])
    # model.save('model.h5')
    # model.save_weights('checkpoints/ckpt-280')








