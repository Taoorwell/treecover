import tensorflow as tf
from utility import *


def image_dataset(path, mode, width, batch_size):
    # image path and mask path dataset
    images_path, masks_path = load_data(path, mode)
    datasets = tf.data.Dataset.from_tensor_slices((images_path, masks_path))

    # parse path into full image and then into patches
    # define parse function
    rng = tf.random.Generator.from_seed(123, alg='philox')

    def parse_fun(x, y):
        def f(x, y):
            x1 = x.decode()
            y1 = y.decode()

            x2 = get_raster(x1)
            y2 = get_raster(y1)
            return x2, y2

        image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
        image.set_shape(image.shape)
        mask.set_shape(mask.shape)
        seed = rng.make_seeds(2)[0]
        image = tf.image.stateless_random_crop(image, size=(width, width, 7), seed=seed)
        mask = tf.image.stateless_random_crop(mask, size=(width, width, 1), seed=seed)
        return image, mask
    datasets = datasets.map(parse_fun, num_parallel_calls=tf.data.AUTOTUNE)
    datasets = datasets.batch(batch_size)
    datasets = datasets.repeat()
    datasets = datasets.prefetch(buffer_size=tf.data.AUTOTUNE)
    return datasets


if __name__ == '__main__':
    path = '../'
    mode = 'train'
    width = 250
    batch_size = 1
    train_dataset = image_dataset(path, mode, width, batch_size)
    for i, (image, mask) in enumerate(train_dataset):
        if i == 0 or i % 279 == 0:
            image1, mask1 = image[0], mask[0]
            plt.subplot(1, 2, 1)
            plt.imshow(image1.numpy()[:, :, [4, 3, 2]])

            plt.subplot(1, 2, 2)
            plt.imshow(mask1.numpy()[:, :, 0])
            plt.show()
