import numpy as np
import albumentations as a
from glob import glob
import os
from utility import get_image
import tensorflow as tf
# from matplotlib import pyplot as plt


class Dataloader(tf.keras.utils.Sequence):
    def __init__(self, path, mode, image_shape, batch_size):
        self.path = path
        self.mode = mode
        self.image_shape = image_shape
        self.batch_size = batch_size

    @property
    def load_path(self):
        # train, valid, test: 235, 29, 30
        images_path = sorted(glob(os.path.join(self.path, "tiles_north/*.tif")))
        masks_path = sorted(glob(os.path.join(self.path, "masks_north/*.tif")))
        length = len(images_path)
        np.random.seed(1)
        idx = np.random.permutation(length)
        if self.mode == 'train':
            idx = idx[:int(0.8 * length)]
        elif self.mode == 'valid':
            idx = idx[int(0.8 * length):int(0.9 * length)]
        else:
            idx = idx[int(0.9 * length):]

        image_path = [images_path[i] for i in idx]
        mask_path = [masks_path[i] for i in idx]
        return image_path, mask_path

    def __len__(self):
        le = int(np.ceil(len(self.load_path[0]) / self.batch_size))
        return le

    @property
    def load_batch(self):
        im_path, ms_path = self.load_path
        n_batches = int(len(im_path) / self.batch_size)
        for i in range(n_batches):
            img_path, mas_path = im_path[i*self.batch_size: (i+1)*self.batch_size], \
                                 ms_path[i*self.batch_size: (i+1)*self.batch_size]
            images, masks = [], []
            for img, mas in zip(img_path, mas_path):
                image, mask = get_image(img), get_image(mas)

                if self.mode != 'test':
                    aug = a.Compose([a.RandomCrop(height=self.image_shape[0],
                                     width=self.image_shape[1],
                                     p=1),
                                     a.VerticalFlip(p=.5),
                                     a.HorizontalFlip(p=.5),
                                     a.RandomRotate90(p=0.5)])
                    augmented = aug(image=image, mask=mask)
                    image, mask = augmented['image'], augmented['mask']
                images.append(image)
                masks.append(mask)
            yield np.stack(images, axis=0), np.stack(masks, axis=0)

    def __getitem__(self, item):
        im_path, ms_path = self.load_path
        i = item * self.batch_size
        img_path, mas_path = im_path[i: (i + self.batch_size)],\
                             ms_path[i: (i + self.batch_size)]

        images = np.zeros((len(img_path), ) + self.image_shape + (7,))
        masks = np.zeros((len(mas_path), ) + self.image_shape + (1,))
        for index, (img, mas) in enumerate(zip(img_path, mas_path)):
            image, mask = get_image(img), get_image(mas)

            if self.mode != 'test':
                aug = a.Compose([a.RandomCrop(height=self.image_shape[0],
                                              width=self.image_shape[1],
                                              p=1),
                                 a.VerticalFlip(p=.5),
                                 a.HorizontalFlip(p=.5),
                                 a.RandomRotate90(p=0.5)])
                augmented = aug(image=image, mask=mask)
                image, mask = augmented['image'], augmented['mask']
            images[index] = image
            masks[index] = mask
        return images, masks


def dataset(path, mode, image_shape, batch_size):

    # get image and mask path according to the mode (train, valid, test)
    images_path = sorted(glob(os.path.join(path, "tiles_north/*.tif")))
    masks_path = sorted(glob(os.path.join(path, "masks_north/*.tif")))
    length = len(images_path)
    np.random.seed(1)
    idx = np.random.permutation(length)
    if mode == 'train':
        idx = idx[:int(0.8 * length)]
    elif mode == 'valid':
        idx = idx[int(0.8 * length):int(0.9 * length)]
    else:
        idx = idx[int(0.9 * length):]

    image_path = [images_path[i] for i in idx]
    mask_path = [masks_path[i] for i in idx]

    # creat path datasets using tf.data.Dataset
    datasets = tf.data.Dataset.from_tensor_slices((image_path, mask_path))
    # creat parse function to parse path to image

    def parse_function(x, y):
        def parse(x, y):
            x1, y1 = x.decode(), y.decode()
            x2, y2 = get_image(x1), get_image(y1)
            return x2, y2
        x3, y3 = tf.numpy_function(parse, inp=(x, y), Tout=[tf.float32, tf.float32])
        x3.set_shape((333, 333, 7))
        y3.set_shape((333, 333, 1))
        return x3, y3

    def augment_function(x, y):
        def augment(x, y):
            aug = a.Compose([a.RandomCrop(height=image_shape[0],
                                          width=image_shape[1],
                                          p=1),
                             a.VerticalFlip(p=.5),
                             a.HorizontalFlip(p=.5),
                             a.RandomRotate90(p=0.5)])
            augmented = aug(image=x, mask=y)
            x2, y2 = augmented['image'], augmented['mask']
            return x2, y2
        x3, y3 = tf.numpy_function(augment, inp=(x, y), Tout=[tf.float32, tf.float32])
        x3.set_shape(image_shape + (7,))
        y3.set_shape(image_shape + (1,))
        return x3, y3

    datasets = datasets.map(parse_function)
    if mode != 'test':
        datasets = datasets.map(augment_function)
    datasets = datasets.batch(batch_size)
    datasets = datasets.repeat()
    return datasets


if __name__ == '__main__':
    valid_datasets = dataset(path='../', mode='valid', image_shape=(256, 256), batch_size=10)
    # for b_image, b_mask in valid_datasets:
    #     b_im, b_ms = b_image[0, :, :, :3], b_mask[0, :, :, 0]
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(b_im)
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(b_ms)
    #     plt.show()

    # dataloader = Dataloader(path='../', mode='valid', image_shape=(256, 256), batch_size=10)
    # for i, j in dataloader:
    #     print(i.shape, j.shape)
    strategy = tf.distribute.MirroredStrategy()
    dist_train_dataloader = strategy.experimental_distribute_dataset(valid_datasets)

