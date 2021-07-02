import numpy as np
import albumentations as a
from glob import glob
import os
from utility import get_image
# from matplotlib import pyplot as plt


class Dataloader:
    def __init__(self, path, mode, image_shape):
        self.path = path
        self.mode = mode
        self.image_shape = image_shape

    @property
    def load_path(self):
        # train, valid, test: 235, 29, 30
        images_path = glob(os.path.join(self.path, "tiles_north/*.tif"))
        masks_path = glob(os.path.join(self.path, "masks_north/*.tif"))
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

    # @property
    # def load_index(self):
    #     image_path, mask_path = self.load_path
    #     image_index = [x.split('_')[-1].split('.')[0] for x in image_path]
    #     mask_index = [x.split('_')[-1].split('.')[0] for x in mask_path]
    #     return image_index, mask_index

    def load_batch(self, batch_size):
        im_path, ms_path = self.load_path
        n_batches = int(len(im_path) / batch_size)
        for i in range(n_batches):
            img_path, mas_path = im_path[i*batch_size: (i+1)*batch_size], \
                                 ms_path[i*batch_size: (i+1)*batch_size]
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


if __name__ == '__main__':
    dataloader = Dataloader(path='../', mode='valid', image_shape=(256, 256, 7))
    print(len(dataloader))
    # index = dataloader.load_index
    # for i, (image, mask) in enumerate(dataloader.load_batch(batch_size=1)):
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(image[0][:, :, [4, 3, 2]])
    #     plt.title('%s' % index[0][i])
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(mask[0][:, :, 0])
    #     plt.title('%s' % index[1][i])
    #     plt.show()


