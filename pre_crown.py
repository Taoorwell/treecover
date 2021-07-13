from utility import *
from residual_unet import *
from matplotlib import pyplot as plt
import scipy


def parse_fun(x, y):
    def f(x, y):
        x1 = x.decode()
        y1 = y.decode()

        x2 = get_image(x1)
        y2 = get_image(y1)
        return x2, y2

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
    image.set_shape([width, width, 7])
    mask.set_shape([width, width, 1])
    return image, mask


if __name__ == '__main__':
    path = '../'
    width = 333
    batch_size = 1
    # images_path = sorted(glob(os.path.join(path, "tiles/*.tif")))
    # masks_path = sorted(glob(os.path.join(path, "masks/*.tif")))
    images_path, masks_path = load_path(path=path, mode='test')
    datasets = tf.data.Dataset.from_tensor_slices((images_path, masks_path))
    datasets = datasets.map(parse_fun)
    datasets = datasets.batch(batch_size)
    # datasets = datasets.repeat()
    model = build_res_unet((333, 333, 7))
    model.load_weights('checkpoints/checkpoints/ckpt-1m_combined_log_cosine_aug_279')
    predictions = np.zeros((len(images_path),) + (width, width))
    predictions1 = np.zeros((len(images_path),) + (width, width))

    masks = np.zeros((len(images_path),) + (width, width))
    for i, (image, mask) in enumerate(datasets):
        mask_pred = model.predict(image)
        predictions1[i] = mask_pred[0, :, :, 0]

        acc = tf.reduce_mean(iou(mask, mask_pred))
        mask_pred = (model.predict(image)[0] > 0.5) * 1
        image_id = images_path[i].split('_')[-1].split('.')[0]

        plt.subplot(131)
        plt.imshow(image.numpy()[0][:, :, [4, 3, 2]])
        plt.xlabel('Image_{}'.format(image_id))
        plt.xticks([])
        plt.yticks([])

        plt.subplot(132)
        plot_mask(mask.numpy()[0][:, :, 0])
        plt.xlabel('mask_{}'.format(image_id))
        plt.xticks([])
        plt.yticks([])

        plt.subplot(133)
        plot_mask(mask_pred[:, :, 0])
        plt.xlabel('mask_{}_pre'.format(image_id))
        plt.xticks([])
        plt.yticks([])

        plt.title('Accuracy:{:.2%}'.format(acc))
        plt.show()
        # plt.savefig('pre/treecover/Image_{}_pre_5'.format(image_id))
        # masks[i] = mask.numpy()[0][:, :, 0]
        # predictions[i] = mask_pred[:, :, 0]
        # save_array_to_mat(masks, 'loss/masks.mat')
        # save_array_to_mat(predictions, 'loss/predictions.mat')
        print('finish: {}'.format(i))
    save_array_to_mat(predictions1, 'loss/predictions1.mat')

