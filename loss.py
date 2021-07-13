import tensorflow as tf
import numpy as np
from utility import *


def iou(y_true, y_pred):
    numerator = tf.reduce_sum(y_true * y_pred, [1, 2])
    denominator = tf.reduce_sum(y_true + y_pred, [1, 2])
    return numerator / (denominator - numerator)

# class Iou(tf.keras.metrics):
#     def __init__(self, y_true, y_pred, axis):
#         super(Iou, self).__init__()
#         self.y_true = y_true
#         self.y_pred = y_pred
#         self.axis = axis
#
#     def __call__(self):
#         numerator = tf.reduce_sum(self.y_true * self.y_pred, self.axis)
#         denominator = tf.reduce_sum(self.y_true + self.y_pred, self.axis)
#         return numerator / (denominator - numerator)


def dice(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, [1, 2])
    denominator = tf.reduce_sum(y_true + y_pred, [1, 2])
    return numerator / denominator


def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, [1, 2])
    denominator = tf.reduce_sum(y_true + y_pred, [1, 2])
    return 1 - (numerator / denominator)


def cross_entropy(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                             reduction=tf.keras.losses.Reduction.NONE)
    loss = bce(y_true, y_pred)
    return tf.reduce_mean(loss, 1)


def combined_loss(y_true, y_pred):
    # y_true = tf.cast(y_true, tf.float32)
    loss = cross_entropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def combined_log_loss(y_true, y_pred):
    eps = 1E-10
    # y_true = tf.cast(y_true, tf.float32)
    loss = cross_entropy(y_true, y_pred) - tf.math.log(iou(y_true, y_pred) + eps)
    return loss


if __name__ == '__main__':
    path = '../'
    images_path, masks_path = load_path(path=path, mode='test')
    images = map(lambda x: get_image(x), images_path)
    masks = map(lambda x: get_image(x), masks_path)
    predictions = get_mat_info('loss/predictions1.mat')

    for i, (image, mask, pre) in enumerate(zip(images, masks, predictions)):
        plt.subplot(131)
        plt.imshow(image[:, :, [4, 3, 2]])
        plt.xlabel('Image')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(132)
        plot_mask(mask[:, :, 0])
        plt.xlabel('mask')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(133)
        plot_mask(pre[:, :] > 0.5)
        plt.xlabel('pred')
        m = np.expand_dims(mask[:, :, 0], axis=0)
        p = np.expand_dims(pre, axis=0)
        # iou, dice = iou(m, p), dice(m, p)
        d_loss, ce = dice_loss(m, p), combined_log_loss(m, p)
        plt.title('dice_loss:{:.4f}, ce:{:.4f}'.format(d_loss[0], ce[0]))
        plt.xticks([])
        plt.yticks([])
        plt.show()


