from utility import load_path, get_image, plot_mask
from matplotlib import pyplot as plt
import time

if __name__ == '__main__':
    path = ['../quality/images', '../quality/high', '../quality/low']
    path = load_path(path=path, mode='train')
    print(len(path), len(path[0]), len(path[1]), len(path[2]))
    for image_path, high_mask_path, low_mask_path in zip(path[0], path[1], path[2]):
        image_id = image_path.split('_')[-1].split('.')[0]
        image = get_image(image_path)
        high_mask = get_image(high_mask_path)
        low_mask = get_image(low_mask_path)

        figure, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 9))
        ax1 = axs.flat[0]
        ax1.imshow(image[:, :, [4, 3, 2]])
        ax1.set_xlabel('Image_{}'.format(image_id))
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = axs.flat[1]
        high_mask = plot_mask(high_mask[:, :, 0])
        ax2.imshow(high_mask)
        ax2.set_xlabel('high_mask_{}'.format(image_id))
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax3 = axs.flat[2]
        low_mask = plot_mask(low_mask[:, :, 0])
        ax3.imshow(low_mask)
        ax3.set_xlabel('low_mask_{}'.format(image_id))
        ax3.set_xticks([])
        ax3.set_yticks([])

        plt.show()

        time.sleep(5)
