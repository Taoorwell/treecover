from utility import *
from residual_unet import *
from tqdm import tqdm

if __name__ == '__main__':
    path = r'../large_scale/'
    width = 256
    image_16 = get_raster(raster_path=path + 'strip_2020_0.tif')
    # image_16 = image_16[:, :, :-1]
    model = build_res_unet(input_shape=(width, width, 7))
    model.load_weights('checkpoints/ckpt-1m_combined_log_cosine_aug_309')
    pre_images = np.zeros(((image_16.shape[0] // width) * width, (image_16.shape[1] // width) * width, 1),
                          dtype=np.float32)
    # image_20 = get_raster(raster_path=path + '2020_sub.tif')
    for r in tqdm(range(image_16.shape[0] // width)):
        for c in range(image_16.shape[1] // width):
            # print([r*width, (r+1)*width, c*width, (c+1)*width])
            image = image_16[r*width: (r+1)*width, c*width: (c+1)*width, :]
            pre = model.predict(np.expand_dims(image, axis=0))
            pre_images[r*width: (r+1)*width, c*width: (c+1)*width, :] = pre
    # plt.imshow(pre_images[:, :, 0])
    # plt.show()
    write_geotiff(name='../large_scale/strip_2020_0_pre_256.tif',
                  prediction=pre_images[:, :, 0],
                  original_path=path + 'strip_2020_0.tif')
