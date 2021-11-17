import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Dropout, \
    add, Input, MaxPool2D, UpSampling2D, concatenate, multiply


# Monte Carlo dropout
def mc_dropout(rate, x, mc=False):
    if mc is True:
        return Dropout(rate)(x, training=True)
    else:
        return Dropout(rate)(x)


def conv_block(x, n_filters, rate, mc=False, recurrent=False, residual=False):
    """
    :param mc: Monte carlo dropout or not
    :param x: input tensor, like (256, 256, 7)
    :param n_filters: number of filters in convolution layer
    :param rate: dropout rate, from 0 to 1, avoid over fitting
    :param recurrent: recurrent unit, if True, convolution operation recurrent one time
    :param residual: residual block, if True, add shortcut with convolution to output
    :return: output, after 2 times convolution block at least (recurrent or residual optional)
    """
    filter_size = 3

    def conv_b_a(x_, re=recurrent):
        conv_op = Conv2D(n_filters, (filter_size, filter_size), padding='same')
        conv_ = conv_op(x_)
        conv_ = BatchNormalization()(conv_)
        conv_ = Activation('relu')(conv_)
        if re is True:
            conv_ = add([x_, conv_])
            conv_ = conv_op(conv_)
            # conv_ = conv_b_a(conv_, re=False)
        return conv_

    if recurrent is True:
        x = Conv2D(n_filters, (1, 1), padding='same')(x)

    conv = conv_b_a(x, recurrent)

    conv = conv_b_a(conv, recurrent)

    if rate > 0:
        # conv = Dropout(dropout)(conv)
        conv = mc_dropout(rate, conv, mc=mc)

    if residual is True:
        shortcut = Conv2D(n_filters, (1, 1), padding='same')(x)
        shortcut = BatchNormalization()(shortcut)
        conv = add([shortcut, conv])
        # conv = Activation('relu')(conv)
    return conv


def attention_block(x, g, n_filters):
    """
    :param x: skip connect from encoder part
    :param g: output in decoder part + up sampling
    :param n_filters: number of filter
    :return: new skip connect but with attention
    """
    x = Conv2D(n_filters, (1, 1))(x)
    g = Conv2D(n_filters, (1, 1))(g)

    query = add([x, g])

    f = Activation('relu')(query)
    f = Conv2D(1, (1, 1))(f)
    sigmoid_f = Activation('sigmoid')(f)

    x_att = multiply([x, sigmoid_f])

    return x_att


def U_Net(input_shape, n_classes, rate=0., mc=False, recurrent=False, residual=False, attention=False):
    """
    :param mc: Mont Carlo or not
    :param rate: if larger 0, apply mc dropout with rate dropout
    :param input_shape: input tensor, image like (256, 256, 7)
    :param n_classes: output classes, binary here, 1
    :param recurrent: if True, recurrent once in convolution operation
    :param residual: if True, add shortcut in convolution block
    :param attention: if True, apply attention gate in decoder part
    :return: mask tensor, shape like (256, 256, 1)
    """

    # network structure parameters
    NUM_FILTER = [14, 28, 56, 112, 140]
    # FILTER_SIZE = 3
    # dropout = .5

    inputs = Input(input_shape, dtype=tf.float32)

    # encoder part
    # down 1
    down_conv_1 = conv_block(x=inputs, n_filters=NUM_FILTER[0], rate=rate, mc=mc,
                             recurrent=recurrent, residual=residual)
    down_pool_1 = MaxPool2D(pool_size=(2, 2))(down_conv_1)

    # down 2
    down_conv_2 = conv_block(x=down_pool_1, n_filters=NUM_FILTER[1], rate=rate, mc=mc,
                             recurrent=recurrent, residual=residual)
    down_pool_2 = MaxPool2D(pool_size=(2, 2))(down_conv_2)

    # down 3
    down_conv_3 = conv_block(x=down_pool_2, n_filters=NUM_FILTER[2], rate=rate, mc=mc,
                             recurrent=recurrent, residual=residual)
    down_pool_3 = MaxPool2D(pool_size=(2, 2))(down_conv_3)

    # down 4
    down_conv_4 = conv_block(x=down_pool_3, n_filters=NUM_FILTER[3], rate=rate, mc=mc,
                             recurrent=recurrent, residual=residual)
    down_pool_4 = MaxPool2D(pool_size=(2, 2))(down_conv_4)

    # bridge
    bridge_conv_5 = conv_block(x=down_pool_4, n_filters=NUM_FILTER[4], rate=rate, mc=mc,
                               recurrent=recurrent, residual=residual)

    # decoder part
    for i, down_conv in enumerate([down_conv_4, down_conv_3, down_conv_2, down_conv_1]):
        up = UpSampling2D(size=(2, 2))(bridge_conv_5)
        if attention is True:
            down_conv = attention_block(down_conv, up, n_filters=NUM_FILTER[(3-i)])
        up = concatenate([up, down_conv], axis=-1)
        up = conv_block(x=up, n_filters=NUM_FILTER[(3-i)], rate=rate, mc=mc,
                        recurrent=recurrent, residual=residual)
        bridge_conv_5 = up
    # # up 4
    # up_4 = UpSampling2D(size=(2, 2))(bridge_conv_5)
    # up_4 = concatenate([up_4, down_conv_4], axis=-1)
    # up_conv_4 = conv_block(x=up_4, n_filters=NUM_FILTER[3], filter_size=FILTER_SIZE,
    #                        dropout=0, recurrent=recurrent, residual=residual)
    #
    # # up 3
    # up_3 = UpSampling2D(size=(2, 2))(up_conv_4)
    # up_3 = concatenate([up_3, down_conv_3], axis=-1)
    # up_conv_3 = conv_block(x=up_3, n_filters=NUM_FILTER[2], filter_size=FILTER_SIZE,
    #                        dropout=0, recurrent=recurrent, residual=residual)
    #
    # # up 2
    # up_2 = UpSampling2D(size=(2, 2))(up_conv_3)
    # up_2 = concatenate([up_2, down_conv_2], axis=-1)
    # up_conv_2 = conv_block(x=up_2, n_filters=NUM_FILTER[1], filter_size=FILTER_SIZE,
    #                        dropout=0, recurrent=recurrent, residual=residual)
    #
    # # up 1
    # up_1 = UpSampling2D(size=(2, 2))(up_conv_2)
    # up_1 = concatenate([up_1, down_conv_1], axis=-1)
    # up_conv_1 = conv_block(x=up_1, n_filters=NUM_FILTER[0], filter_size=FILTER_SIZE,
    #                        dropout=0, recurrent=recurrent, residual=residual)

    # Output part
    if n_classes == 1:
        conv_res = Conv2D(n_classes, kernel_size=(1, 1), activation='sigmoid')(bridge_conv_5)
    else:
        conv_res = Conv2D(n_classes, kernel_size=(1, 1), activation='softmax')(bridge_conv_5)
    # conv_res = Conv2D(n_classes, kernel_size=(1, 1), activation='sigmoid')(up_conv_1)

    # Model
    model = tf.keras.models.Model(inputs, conv_res)
    return model


if __name__ == '__main__':
    unet = U_Net(input_shape=(256, 256, 7), n_classes=2, rate=0.0, mc=False, residual=True)
    unet.summary()
    # for i, layer in enumerate(unet.layers[:48]):
    #     print(f'{i}th layer:')
    #     print(layer)
    #     layer.trainable=False
    # unet.summary()

