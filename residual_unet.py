import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Activation, add, concatenate


def res_block(x, nb_filters, strides):
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(x)
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='relu')(res_path)

    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='relu')(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(3, 3), strides=strides[0], padding='same')(x)
    shortcut = BatchNormalization()(shortcut)

    res_path = add([shortcut, res_path])
    return res_path


def encoder(x):
    to_decoder = []

    main_path = res_block(x, [32, 32], [(1, 1), (1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [64, 64], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [128, 128], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    return to_decoder


def slice_concatenate(x1, x2):
    if x1.shape != x2.shape:
        x1 = x1[:, :x2.shape[1], :x2.shape[2], :]
    x = concatenate([x1, x2], axis=3)
    return x


def decoder(x, from_encoder):
    main_path = UpSampling2D(size=(2, 2))(x)
    main_path = slice_concatenate(main_path, from_encoder[2])
    # main_path = concatenate((main_path, from_encoder[2]), axis=3)
    main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = slice_concatenate(main_path, from_encoder[1])
    # main_path = concatenate((main_path, from_encoder[1]), axis=3)
    main_path = res_block(main_path, [64, 64], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = slice_concatenate(main_path, from_encoder[0])
    # main_path = concatenate((main_path, from_encoder[0]), axis=3)
    main_path = res_block(main_path, [32, 32], [(1, 1), (1, 1)])

    return main_path


def build_res_unet(input_shape):
    inputs = Input(shape=input_shape)
    '''encoder'''
    to_decoder = encoder(inputs)
    # print(to_decoder[0].shape, to_decoder[1].shape, to_decoder[2].shape)
    '''bridge'''
    path = res_block(to_decoder[2], [128, 128], [(2, 2), (1, 1)])
    # print('bridge output shape:', path.shape)
    '''decoder'''
    path = decoder(path, from_encoder=to_decoder)
    # print('decoder:', path.shape)
    '''output'''
    path = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(path)
    # print('final output shape:', path.shape)
    return Model(inputs=inputs, outputs=path)


def Iou(y_true, y_pred):
    numerator = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return numerator / (denominator - numerator)


def dice(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return numerator / denominator


def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - (numerator / denominator)


def cross_entropy(y_true, y_pred):
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(loss)


def combined_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    loss = cross_entropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def combined_log_loss(y_true, y_pred):
    eps = 1E-15
    y_true = tf.cast(y_true, tf.float32)
    loss = cross_entropy(y_true, y_pred) - tf.math.log(Iou(y_true, y_pred) + eps)
    return loss



