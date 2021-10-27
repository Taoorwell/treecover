import tensorflow as tf
import numpy as np
from utility import get_image, rgb_mask
import matplotlib.pyplot as plt
# Metrics-accuracy
eps = 1e-10


def iou(y_true, y_pred):
    # y_true and y_pred shape: batch_size, image_width, image_width, 1 or none.
    # reduce_sum and axis [1, 2], get each image accuracy.
    # y_true = tf.cast(y_true, tf.float32)
    numerator = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return (numerator + eps) / (denominator - numerator + eps)


def tree_iou(y_true, y_pred):
    y_true = y_true[..., 1]
    y_pred = y_pred[..., 1]
    # y_true and y_pred shape: batch_size, image_width, image_width, 1 or none.
    # reduce_sum and axis [1, 2], get each image accuracy.
    # y_true = tf.cast(y_true, tf.float32)
    numerator = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return (numerator + eps) / (denominator - numerator + eps)


def dice(y_true, y_pred):
    # y_true = tf.cast(y_true, tf.float32)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    # numerator = 2 * tf.reduce_sum(y_true == (y_pred >= 0.5))
    # denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    # return numerator / denominator
    return (numerator + eps) / (denominator + eps)


def tversky(y_true, y_pred, beta=0.99):
    # y_true = tf.cast(y_true, tf.float32)
    numerator = tf.reduce_sum(y_true * y_pred)
    denominator = numerator + beta * tf.reduce_sum((1 - y_true) * y_pred) \
                  + (1 - beta) * tf.reduce_sum(y_true * (1 - y_pred))

    return (numerator + eps) / (denominator + eps)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gama=0.25):
    tv_loss = tversky_loss(y_true, y_pred)
    return tf.pow(tv_loss, gama)


# Loss function, dice_loss and cross entropy, and combined loss function
def dice_loss(y_true, y_pred):
    return 1 - dice(y_true, y_pred)


def cross_entropy(y_true, y_pred, weight=False):
    # y_true = tf.cast(y_true, tf.float32)
    bce_func = tf.keras.losses.BinaryCrossentropy(from_logits=False,
                                                  reduction=tf.keras.losses.Reduction.NONE)
    bce = bce_func(y_true, y_pred)  # no reduction shape: (batch_size, width, width), pixel level
    loss = tf.expand_dims(bce, axis=-1)
    if weight:
        loss = log_conv(y_true) * loss
    return tf.reduce_mean(loss)


def categorical_cross_entropy(y_true, y_pred):
    cce_func = tf.keras.losses.CategoricalCrossentropy()
    cce = cce_func(y_true, y_pred)
    return cce


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
def combined_loss(y_true, y_pred, weight=False):
    loss = cross_entropy(y_true, y_pred, weight=weight) + 0.5 * focal_tversky_loss(y_true, y_pred)
    return loss


# combine cross entropy and logarithm of iou
def combined_log_loss(y_true, y_pred, weight=False):
    loss = cross_entropy(y_true, y_pred, weight=weight) - tf.math.log(iou(y_true, y_pred) + eps)
    return loss


if __name__ == '__main__':
    y_true_2 = tf.constant([[1, 0], [0, 1]], dtype=tf.float32)
    y_pre_2 = tf.constant([[.8, .2], [.2, .8]], dtype=tf.float32)
    cce1 = cross_entropy(y_true_2, y_pre_2)
    print(cce1)

    y_true_1 = tf.constant([[1, 0]], dtype=tf.float32)
    y_pre_1 = tf.constant([[.8, .2]], dtype=tf.float32)
    ce = cross_entropy(y_true_1, y_pre_1)
    print(ce)

    # # 39, 8, 137, 62
    # n = 137
    # path = r'loss/'
    # # for sigmoid single band output
    # mask_39 = get_image(path + 'mask_{}.tif'.format(n))
    # mask_39_1 = get_image(path + 'mask_{}_1.tif'.format(n))
    # mask_39_2 = get_image(path + 'mask_{}_2.tif'.format(n))
    #
    # # for softmax double band output
    # # mask_39 = np.eye(2)[np.array(mask_39[:, :, 0], np.int32)]
    # # mask_39_1 = np.eye(2)[np.array(mask_39_1[:, :, 0], np.int32)]
    # # mask_39_2 = np.eye(2)[np.array(mask_39_2[:, :, 0], np.int32)]
    #
    # # accuracy computing
    # acc1 = iou(mask_39, mask_39_1)
    # acc2 = iou(mask_39, mask_39_2)
    #
    # # loss computing
    # loss_1 = dice_loss(mask_39, mask_39_1)
    # loss_2 = dice_loss(mask_39, mask_39_2)
    #
    # plt.subplot(131)
    # plt.imshow(rgb_mask(mask_39[:, :, -1]))
    # plt.xlabel('True mask')
    # plt.xticks([])
    # plt.yticks([])
    #
    # plt.subplot(132)
    # plt.imshow(rgb_mask(mask_39_1[:, :, -1]))
    # plt.xlabel('Prediction 1')
    # plt.title('Iou:{:.2%}\nloss:{:.4f}'.format(acc1, loss_1))
    # plt.xticks([])
    # plt.yticks([])
    #
    # plt.subplot(133)
    # plt.imshow(rgb_mask(mask_39_2[:, :, -1]))
    # plt.xlabel('Prediction 2')
    # plt.title('Iou:{:.2%}\nloss:{:.4f}'.format(acc2, loss_2))
    # plt.xticks([])
    # plt.yticks([])
    #
    # plt.show()
    # final_loss = combined_log_loss(a, b, weight=False)
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
