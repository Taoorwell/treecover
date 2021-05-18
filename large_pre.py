from utility import *
from residual_unet import *
from tqdm import tqdm

if __name__ == '__main__':
    path = r'../large_scale/subplots/2020_m2/2020_m2_1/'
    width = 250
    subplots = sorted(glob(os.path.join(path, "*.tif")))
    model = build_res_unet(input_shape=(width, width, 7))
    model.load_weights('checkpoints/checkpoints/ckpt-1m_combined_log_cosine_aug_309')
    # print(subplots)
    for plot in tqdm(subplots):
        image = get_raster(raster_path=plot)
        pre_images = np.zeros(((image.shape[0] // width) * width, (image.shape[1] // width) * width, 1),
                              dtype=np.float32)
        for r in tqdm(range(image.shape[0] // width)):
            for c in range(image.shape[1] // width):
                # print([r*width, (r+1)*width, c*width, (c+1)*width])
                image1 = image[r*width: (r+1)*width, c*width: (c+1)*width, :]
                pre = model.predict(np.expand_dims(image1, axis=0))
                pre_images[r*width: (r+1)*width, c*width: (c+1)*width, :] = pre
        write_geotiff(name=path + plot.split('\\')[-1].split('.')[0] + '_pre.tif',
                      prediction=pre_images[:, :, 0],
                      original_path=plot)
        print('Prediction file check: {}'.format(path + plot.split('\\')[-1].split('.')[0] + '_pre.tif'))
