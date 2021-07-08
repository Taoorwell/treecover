import numpy as np
import albumentations as a
from glob import glob
import os
from utility import get_image
import tensorflow as tf
# from matplotlib import pyplot as plt


class Dataloader:
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
        return len(self.load_path[0])

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


def make_generator_callable(gen):
    def generator():
        for i, j in gen:
            yield i, j
    return generator


if __name__ == '__main__':
    dataloader = Dataloader(path='../', mode='valid', image_shape=(256, 256, 7), batch_size=2)