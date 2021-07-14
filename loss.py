import tensorflow as tf


# Metrics-accuracy
def iou(y_true, y_pred):
    # y_true and y_pred shape: batch_size, image_width, image_width, 1 or none.
    # reduce_sum and axis [1, 2], get each image accuracy.
    y_true = tf.cast(y_true, tf.float32)
    numerator = tf.reduce_sum(y_true * y_pred, [1, 2])
    denominator = tf.reduce_sum(y_true + y_pred, [1, 2])
    return tf.divide(numerator, tf.subtract(denominator, numerator))


def dice(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    numerator = 2 * tf.reduce_sum(y_true * y_pred, [1, 2])
    denominator = tf.reduce_sum(y_true + y_pred, [1, 2])
    return tf.divide(numerator, denominator)


# Loss function, dice_loss and cross entropy, and combined loss function
def dice_loss(y_true, y_pred):
    return tf.subtract(1, dice(y_true, y_pred))


def cross_entropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                             reduction=tf.keras.losses.Reduction.NONE)
    loss = tf.expand_dims(bce(y_true, y_pred), axis=-1)
    return tf.reduce_mean(loss, [1, 2])


def combined_loss(y_true, y_pred):
    loss = cross_entropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def combined_log_loss(y_true, y_pred):
    eps = 1E-15
    loss = tf.subtract(cross_entropy(y_true, y_pred), tf.math.log(iou(y_true, y_pred) + eps))
    return loss


if __name__ == '__main__':
    a = tf.sigmoid(tf.random.normal((4, 50, 50, 1), 0, 1, dtype=tf.float32))
    b = tf.sigmoid(tf.random.normal((4, 50, 50, 1), 0, 1, dtype=tf.float32))
    cb = combined_log_loss(a, b)
    print(cb)
    ce = cross_entropy(a, b)
    print(ce)
    iou = iou(a, b)
    print(iou)
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


