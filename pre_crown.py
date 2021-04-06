from utility import *
from residual_unet import *
from matplotlib import pyplot as plt


def parse_fun(x, y):
    def f(x, y):
        x1 = x.decode()
        y1 = y.decode()

        x2 = get_raster(x1)
        y2 = get_raster(y1)
        return x2, y2

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
    image.set_shape([width, width, 7])
    mask.set_shape([width, width, 1])
    return image, mask


if __name__ == '__main__':
    path = '../../../treecrown'
    width = 1000
    batch_size = 1
    images_path = sorted(glob(os.path.join(path, "tiles/*.tif")))
    masks_path = sorted(glob(os.path.join(path, "masks/*.tif")))
    datasets = tf.data.Dataset.from_tensor_slices((images_path, masks_path))
    datasets = datasets.map(parse_fun)
    datasets = datasets.batch(batch_size)
    datasets = datasets.repeat()
    model = build_res_unet((1000, 1000, 7))
    model.load_weights('checkpoints/ckpt')
    for i, (image, mask) in enumerate(datasets):
        mask_pred = model.predict(image)
        acc = dice(mask, mask_pred)
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
        # plt.show()
        plt.savefig('pre/treecrown/Image_{}_pre'.format(image_id))
        print('finish: {}'.format(i))
        if i == 34:
            break
