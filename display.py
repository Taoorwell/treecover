from utility import load_path, get_image, plot_mask
from matplotlib import pyplot as plt
import time

if __name__ == '__main__':
    path = '../'
    images_path, masks_path = load_path(path=path, mode='train')
    print(len(images_path))
    for image_path, mask_path in zip(images_path, masks_path):
        image_id = image_path.split('_')[-1].split('.')[0]
        image = get_image(image_path)
        mask = get_image(mask_path)
        figure, axs = plt.subplots(1, 2)
        ax1 = axs[0]
        ax1.imshow(image[:, :, [4, 3, 2]])
        ax1.set_xlabel('Image_{}'.format(image_id))
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = axs[1]
        plot_mask(mask[:, :, 0])
        ax2.set_xlabel('mask_{}'.format(image_id))
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.show()
        time.sleep(5)
