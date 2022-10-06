import os
import numpy as np
from osgeo import gdal
from matplotlib import pyplot as plt
from glob import glob
import scipy.io as sio


def get_path(path, mode='train', seed=1, active=0):
    # get image and mask path according to the mode (train, valid, test)
    images_path = sorted(glob(os.path.join(r'../quality/', r"images/*.tif")))
    masks_path = sorted(glob(os.path.join(path, '*.tif')))
    length = len(images_path)
    np.random.seed(seed)
    idx = np.random.permutation(length)
    train_idx, test_idx = idx[:-30], idx[-30:]
    if mode == 'train':
        idx = train_idx[:280]
    elif mode == 'valid':
        idx = train_idx[280:]
    else:
        idx = test_idx

    image_path = [images_path[i] for i in idx]
    mask_path = [masks_path[i] for i in idx]
    image_id = [int(im.split('_')[-1].split('.')[0]) for im in image_path]
    if active != 0:
        image_path = image_path[(active-1)*40:active*40]
        mask_path = mask_path[(active-1)*40:active*40]
        image_id = image_id[(active-1)*40:active*40]
    return image_path, mask_path, image_id


def get_image(raster_path):
    ds = gdal.Open(raster_path)
    image = np.empty((ds.RasterYSize, ds.RasterXSize, ds.RasterCount), dtype=np.float32)
    for b in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(b).ReadAsArray()
        image[:, :, b-1] = band
    if image.shape[-1] == 1:
        image = image[:, :, 0].astype(int)
        image = np.eye(2)[image]
        image = image.astype(np.float32)
    else:
        image = norma_data(image, norma_methods='min-max')
    return image


def write_geotiff(name, prediction, original_path):
    ds = gdal.Open(original_path)
    geo = ds.GetGeoTransform()
    proj = ds.GetProjectionRef()

    driver = gdal.GetDriverByName('GTiff')
    rows, cols = prediction.shape
    dataset = driver.Create(name, cols, rows, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform(geo)
    dataset.SetProjection(proj)
    band = dataset.GetRasterBand(1)
    band.WriteArray(prediction)


def norma_data(data, norma_methods="z-score"):
    arr = np.empty(data.shape, dtype=np.float32)
    for i in range(data.shape[-1]):
        array = data[:, :, i]
        mi, ma, mean, std = np.percentile(array, 1), np.percentile(array, 99), array.mean(), array.std()
        if norma_methods == "z-score":
            new_array = (array-mean)/std
        else:
            new_array = (2*(array-mi)/(ma-mi)).clip(0, 1)
        arr[:, :, i] = new_array
    return arr


def rgb_mask(result):
    arr_2d = result
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i
    return arr_3d


def get_mat_info(mat_data_path):
    bands_data_dict = sio.loadmat(mat_data_path)
    bands_data = bands_data_dict[list(bands_data_dict.keys())[-1]]
    return bands_data


def save_array_to_mat(array, filename):
    dict = {"pre": array}
    sio.savemat(filename, dict)


palette = {0: (255, 255, 255),  # White
           6: (0, 191, 255),  # DeepSkyBlue
           1: (34, 139, 34),  # ForestGreen
           3: (255, 0, 255),  # Magenta
           2: (0, 255, 0),  # Lime
           5: (255, 127, 80),  # Coral
           4: (255, 0, 0),  # Red
           7: (0, 255, 255),  # Cyan
           8: (0, 255, 0),  # Lime
           9: (0, 128, 128),
           10: (128, 128, 0),
           11: (255, 128, 128),
           12: (128, 128, 255),
           13: (128, 255, 128),
           14: (255, 128, 255),
           15: (165, 42, 42),
           16: (175, 238, 238)}


def iou(y_true, y_pred):
    numerator = np.sum(y_true[:, :, :] * y_pred[:, :, :])
    denominator = np.sum(y_true[:, :, :] + y_pred[:, :, :])
    return numerator / (denominator - numerator)


if __name__ == '__main__':
    # path = r'../quality/high/'
    # images_path, masks_path, images_id = get_path(path, mode='train', seed=2, active=0)
    # print(len(images_path))
    # for im, ms, ids in zip(images_path[80:120], masks_path[80:120], images_id[80:120]):
    #     image = get_image(im)
    #     mask = get_image(ms)
    #     plt.subplot(121)
    #     plt.imshow(image[:, :, [4, 3, 2]])
    #     plt.xlabel(f'image_{int(ids)}')
    #     plt.xticks([])
    #     plt.yticks([])
    #
    #     plt.subplot(122)
    #     plt.imshow(rgb_mask(mask[:, :, 1]))
    #     plt.xlabel(f'mask_{int(ids)}')
    #     plt.xticks([])
    #     plt.yticks([])
    #
    #     plt.show()
    #     break
    image_201 = r'../quality/images/tile_201.tif'
    ds = gdal.Open(image_201)
    geo = ds.GetGeoTransform()
    proj = ds.GetProjectionRef()
    print(geo)
    print(proj)

    # high_201 = r'../quality/high/mask_201.tif'
    # low_201 = r'../quality/low/mask_201.tif'
    # image_201, high_201, low_201 = get_image(image_201), get_image(high_201), get_image(low_201)
    #
    # plt.imshow(image_201[:, :, [4, 3, 2]])
    # # plt.imshow(rgb_mask(high_201[:, :, 1]), alpha=1)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
