import os
from glob import glob
import numpy as np
from osgeo import gdal
from matplotlib import pyplot as plt


def load_data(path, mode):
    images_path = sorted(glob(os.path.join(path, "tiles_north/*.tif")))
    masks_path = sorted(glob(os.path.join(path, "masks_north/*.tif")))
    if mode == 'train':
        image_path, mask_path = images_path[:-35], masks_path[0:-35]
    else:
        image_path, mask_path = images_path[-35:], masks_path[-35:]
    return image_path, mask_path


def get_raster(raster_path):
    ds = gdal.Open(raster_path)
    data = np.empty((ds.RasterYSize, ds.RasterXSize, ds.RasterCount), dtype=np.float32)
    for b in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(b).ReadAsArray()
        data[:, :, b-1] = band
    if data.shape[-1] > 1:
        data = norma_data(data, norma_methods='min-max')
    return data


def norma_data(data, norma_methods="z-score"):
    arr = np.empty(data.shape, dtype=np.float32)
    for i in range(data.shape[-1]):
        array = data.transpose(2, 0, 1)[i, :, :]
        mins, maxs, mean, std= np.percentile(array, 1), np.percentile(array, 99), np.mean(array), np.std(array)
        if norma_methods == "z-score":
            new_array = (array-mean)/std
        else:
            new_array = np.clip(2*(array-mins)/(maxs-mins), 0, 1)
        arr[:, :, i] = new_array
    return arr


def plot_mask(result):
    arr_2d = result
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i
    plt.imshow(arr_3d)


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