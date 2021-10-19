import tensorflow as tf
import numpy as np


# Metrics-accuracy
def iou(y_true, y_pred):
    # y_true and y_pred shape: batch_size, image_width, image_width, 1 or none.
    # reduce_sum and axis [1, 2], get each image accuracy.
    y_true = tf.cast(y_true, tf.float32)
    numerator = tf.reduce_sum(y_true * y_pred, [1, 2])
    denominator = tf.reduce_sum(y_true + y_pred, [1, 2])
    return numerator / (denominator - numerator)


def dice(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    numerator = 2 * tf.reduce_sum(y_true * y_pred, [1, 2])
    denominator = tf.reduce_sum(y_true + y_pred, [1, 2])
    return numerator / denominator


def tversky(y_true, y_pred, beta=0.99):
    y_true = tf.cast(y_true, tf.float32)
    numerator = tf.reduce_sum(y_true * y_pred, [1, 2])
    denominator = numerator + beta * tf.reduce_sum((1 - y_true) * y_pred, [1, 2]) \
                  + (1 - beta) * tf.reduce_sum(y_true * (1 - y_pred), [1, 2])

    return numerator / denominator


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gama=0.25):
    tv_loss = tversky_loss(y_true, y_pred)
    return tf.pow(tv_loss, gama)


# Loss function, dice_loss and cross entropy, and combined loss function
def dice_loss(y_true, y_pred):
    return 1 - dice(y_true, y_pred)


def cross_entropy(y_true, y_pred, weight=False):
    y_true = tf.cast(y_true, tf.float32)
    bce_func = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                  reduction=tf.keras.losses.Reduction.NONE)
    bce = bce_func(y_true, y_pred)  # no reduction shape: (batch_size, width, width), pixel level
    loss = tf.expand_dims(bce, axis=-1)
    if weight:
        loss = log_conv(y_true) * loss
    return tf.reduce_mean(loss, [1, 2])


# weight map generating by LoG filter convolution
def log_conv(y_true):
    # Laplacian for edge extraction
    laplacian_filter = tf.constant([[0, .25, 0], [.25, -1, .25], [0, .25, 0]],
                                   dtype=tf.float32)
    laplacian_filter = tf.reshape(laplacian_filter, (3, 3, 1, 1))

    output = tf.nn.conv2d(y_true,
                          filters=laplacian_filter,
                          strides=1,
                          padding=[[0, 0], [1, 1], [1, 1], [0, 0]])

    edge = output != 0
    edge = tf.cast(edge, tf.float32)

    # Gaussian blur for pixel weight
    def gaussian_2d(ksize, sigma=1):
        m = (ksize - 1) / 2
        y, x = np.ogrid[-m:m+1, -m:m+1]
        value = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
        # value[value < np.finfo(value.dtype).eps * value.max()] = 0
        sum_v = value.sum()
        if sum_v != 0:
            value /= sum_v
        return value

    gaussian_filter = gaussian_2d(ksize=3, sigma=1)
    gaussian_filter = np.reshape(gaussian_filter, (3, 3, 1, 1))

    pixel_weight = 10 * tf.nn.conv2d(edge,
                                     filters=gaussian_filter,
                                     strides=1,
                                     padding='SAME') + 1

    return pixel_weight


# combine cross entropy and focal tversky loss
def combined_loss(y_true, y_pred, weight=True):
    loss = cross_entropy(y_true, y_pred, weight=weight) + 0.5 * focal_tversky_loss(y_true, y_pred)
    return loss


# combine cross entropy and logarithm of iou
def combined_log_loss(y_true, y_pred, weight=True):
    eps = 1E-15
    loss = cross_entropy(y_true, y_pred, weight=weight) - tf.math.log(iou(y_true, y_pred) + eps)
    return loss


if __name__ == '__main__':
    a = tf.sigmoid(tf.random.normal((4, 50, 50, 1), 0, 1, dtype=tf.float32))
    b = tf.sigmoid(tf.random.normal((4, 50, 50, 1), 0, 1, dtype=tf.float32))

    final_loss = combined_log_loss(a, b, weight=True)
    print(final_loss)
    # combined = combined_loss(a, b)
    # combined_1 = combined_log_loss(a, b)
    # print(tf.reduce_mean(combined), combined_1)

    # dice1 = dice_loss(a, b)
    # print(dice1)
    # cb = combined_log_loss(a, b)
    # print(cb)
    # ce = cross_entropy(a, b)
    # print(ce)
    # iou = iou(a, b)
    # print(iou)
    # path = '../'
    # images_path, masks_path = load_path(path=path, mode='test')
    # images = map(lambda x: get_image(x), images_path)
    # masks = map(lambda x: get_image(x), masks_path)
    # predictions = get_mat_info('loss/predictions1.mat')
    #
    # for i, (image, mask, pre) in enumerate(zip(images, masks, predictions)):
    #     plt.subplot(131)
    #     plt.imshow(image[:, :, [4, 3, 2]])
    #     plt.xlabel('Image')
    #     plt.xticks([])
    #     plt.yticks([])
    #
    #     plt.subplot(132)
    #     plot_mask(mask[:, :, 0])
    #     plt.xlabel('mask')
    #     plt.xticks([])
    #     plt.yticks([])
    #
    #     plt.subplot(133)
    #     plot_mask(pre[:, :] > 0.5)
    #     plt.xlabel('pred')
    #     # plt.title('dice_loss:{:.4f}, ce:{:.4f}'.format(d_loss[0], ce[0]))
    #     plt.xticks([])
    #     plt.yticks([])
    #
    #     m = np.expand_dims(mask[:, :, 0], axis=0)
    #     p = np.expand_dims(pre, axis=0)
    #     d_loss, ce, combine = dice_loss(m, p), cross_entropy(m, p), combined_log_loss(m, p)
    #     iou, dice = iou(m, p), dice(m, p)
    #     plt.suptitle('dice_loss:{:.4f}, ce:{:.4f}, '
    #                  'combine:{:.4f}, iou:{:.4f}, '
    #                  'dice:{:.4f}'.format(d_loss[0], ce[0], combine[0], iou[0], dice[0]))
    #     plt.show()
    #     break
