from sys import stdout
from dataloder import dataset
from utility import rgb_mask
# from utility import get_image, write_geotiff
from unets import U_Net
import time
import numpy as np
from matplotlib import pyplot as plt


def iou(y_true, y_pred):
    # y_true and y_pred shape: batch_size, image_width, image_width, 1 or none.
    # reduce_sum and axis [1, 2], get each image accuracy.
    numerator = np.sum(y_true * y_pred)
    denominator = np.sum(y_true + y_pred)
    return numerator / (denominator - numerator)


def compute_pyramid_patch_weight_loss(width, height):
    """Compute a weight matrix that assigns bigger weight on pixels in
center and
    less weight to pixels on image boundary.
    This weight matrix is then used for merging individual tile
predictions and helps dealing
    with prediction artifacts on tile boundaries.

    Taken from & credit to:

https://github.com/BloodAxe/pytorch-toolbelt/blob/f3acfca5da05cd7ccdd85e8d343d75fa40fb44d9/pytorch_toolbelt/inference/tiles.py#L16-L50

    Args:
        width: Tile width
        height: Tile height
    Returns:
        The weight mask as ndarray
    """
    xc = width * 0.5
    yc = height * 0.5
    xl = 0
    xr = width
    yb = 0
    yt = height

    Dcx = np.square(np.arange(width) - xc + 0.5)
    Dcy = np.square(np.arange(height) - yc + 0.5)
    Dc = np.sqrt(Dcx[np.newaxis].transpose() + Dcy)

    De_l = np.square(np.arange(width) - xl + 0.5) + np.square(0.5)
    De_r = np.square(np.arange(width) - xr + 0.5) + np.square(0.5)
    De_b = np.square(0.5) + np.square(np.arange(height) - yb + 0.5)
    De_t = np.square(0.5) + np.square(np.arange(height) - yt + 0.5)

    De_x = np.sqrt(np.minimum(De_l, De_r))
    De_y = np.sqrt(np.minimum(De_b, De_t))
    De = np.minimum(De_x[np.newaxis].transpose(), De_y)

    alpha = (width * height) / np.sum(np.divide(De, np.add(Dc, De)))
    W = alpha * np.divide(De, np.add(Dc, De))
    return W


def predict_on_array(model,
                     arr,
                     in_shape,
                     out_bands,
                     stride=None,
                     drop_border=0,
                     batchsize=64,
                     dtype="float32",
                     augmentation=False,
                     no_data=None,
                     verbose=False,
                     report_time=False):
    """
    Applies a pytorch segmentation model to an array in a strided
manner.

    Call model.eval() before use!

    Args:
        model: pytorch model - make sure to call model.eval() before
using this function!
        arr: HWC array for which the segmentation should be created
        stride: stride with which the model should be applied. Default:
output size
        batchsize: number of images to process in parallel
        dtype: desired output type (default: float32)
        augmentation: whether to average over rotations and mirrorings
of the image or not. triples computation time.
        no_data: a no-data vector. its length must match the number of
layers in the input array.
        verbose: whether or not to display progress
        report_time: if true, returns (result, execution time)

    Returns:
        An array containing the segmentation.
    """
    t0 = None

    if augmentation:
        operations = (lambda x: x,
                      lambda x: np.rot90(x, 1),
                      # lambda x: np.rot90(x, 2),
                      # lambda x: np.rot90(x, 3),
                      # lambda x: np.flip(x,0),
                      lambda x: np.flip(x, 1))

        inverse = (lambda x: x,
                   lambda x: np.rot90(x, -1),
                   # lambda x: np.rot90(x, -2),
                   # lambda x: np.rot90(x, -3),
                   # lambda x: np.flip(x,0),
                   lambda x: np.flip(x, 1))
    else:
        operations = (lambda x: x,)
        inverse = (lambda x: x,)

    assert in_shape[0] == in_shape[1], "Input shape must be equal in first two dims."
    out_shape = (in_shape[0] - 2 * drop_border, in_shape[1] - 2 * drop_border, out_bands)
    in_size = in_shape[0]
    out_size = out_shape[0]
    stride = stride or out_size
    pad = (in_size - out_size)//2
    assert pad % 2 == 0, "Model input and output shapes have to be divisible by 2."

    weight_mask = compute_pyramid_patch_weight_loss(out_size, out_size)

    original_size = arr.shape
    ymin = 0
    xmin = 0

    if no_data is not None:
        # assert arr.shape[-1]==len(no_data_vec), "Length of no_data_vec must match number of channels."
        # data_mask = np.all(arr[:,:,0].reshape( (-1,arr.shape[-1]) ) != no_data, axis=1).reshape(arr.shape[:2])
        nonzero = np.nonzero(arr[:, :, 0]-no_data)
        ymin = np.min(nonzero[0])
        ymax = np.max(nonzero[0])
        xmin = np.min(nonzero[1])
        xmax = np.max(nonzero[1])
        img = arr[ymin:ymax, xmin:xmax]

    else:
        img = arr

    final_output = np.zeros(img.shape[:2]+(out_shape[-1],), dtype=dtype)

    op_cnt = 0
    for op, inv in zip(operations, inverse):
        img = op(img)
        img_shape = img.shape
        x_tiles = int(np.ceil(img.shape[1]/stride))
        y_tiles = int(np.ceil(img.shape[0]/stride))

        y_range = range(0, (y_tiles+1)*stride-out_size, stride)
        x_range = range(0, (x_tiles+1)*stride-out_size, stride)

        y_pad_after = y_range[-1]+in_size-img.shape[0]-pad
        x_pad_after = x_range[-1]+in_size-img.shape[1]-pad

        output = np.zeros((img.shape[0]+y_pad_after-pad, img.shape[1]+x_pad_after-pad)+(out_shape[-1],), dtype=dtype)
        division_mask = np.zeros(output.shape[:2], dtype=dtype) + 1E-7
        img = np.pad(img, ((pad, y_pad_after), (pad, x_pad_after), (0, 0)), mode='reflect')

        patches = len(y_range)*len(x_range)

        def patch_generator():
            for y in y_range:
                for x in x_range:
                    yield img[y:y+in_size, x:x+in_size]

        patch_gen = patch_generator()

        y = 0
        x = 0
        patch_idx = 0
        batchsize_ = batchsize

        t0 = time.time()

        while patch_idx < patches:
            batchsize_ = min(batchsize_, patches, patches - patch_idx)
            patch_idx += batchsize_
            if verbose:
                stdout.write("\r%.2f%%" % (100 * (patch_idx + op_cnt * patches) / (len(operations) * patches)))

            batch = np.zeros((batchsize_,) + in_shape, dtype=dtype)

            for j in range(batchsize_):
                batch[j] = next(patch_gen)

            # EDIT HERE
            prediction = model.predict(batch)
            if drop_border > 0:
                prediction = prediction[:, drop_border:-drop_border, drop_border:-drop_border, :]

            for j in range(batchsize_):
                output[y:y + out_size, x:x + out_size] += prediction[j] * weight_mask[..., None]
                division_mask[y:y + out_size, x:x + out_size] += weight_mask
                x += stride
                if x + out_size > output.shape[1]:
                    x = 0
                    y += stride

        output = output / division_mask[..., None]
        output = inv(output[:img_shape[0], :img_shape[1]])
        final_output += output
        img = arr[ymin:ymax, xmin:xmax] if no_data is not None else arr
        op_cnt += 1
        if verbose:
            stdout.write("\rAugmentation step %d/%d done.\n" % (op_cnt, len(operations)))

    if verbose:
        stdout.flush()

    final_output = final_output/len(operations)

    if no_data is not None:
        final_output = np.pad(final_output, ((ymin, original_size[0] - ymax), (xmin, original_size[1]-xmax), (0, 0)),
                              mode='constant', constant_values=0)

    if report_time:
        return final_output, time.time() - t0

    else:
        return final_output


if __name__ == '__main__':
    # Parameter define
    width, height = 256, 256
    # image_path = r'../large_scale/subplots/2020_m2/2020_m2_1/2020_11_north_urban_m2_1_0.tif'
    # output_path = r'../large_scale/subplots/2020_m2/2020_m2_1/2020_11_north_urban_m2_1_0_pre.tif'
    # Trained Model loading
    # model = build_res_unet(input_shape=(width, width, 7))
    model = U_Net(input_shape=(width, width, 7), n_classes=1, recurrent=False, residual=True, attention=True)
    model.load_weights('checkpoints/ckpt-unet_res_att_300_low')

    # Image loading for further prediction
    # large_image = get_image(raster_path=image_path)
    dataset = dataset(path=r'../quality/', mode='test', image_shape=(256, 256), batch_size=1)
    acc = []
    for im, ms in dataset:
        # print(im.shape, ms.shape)
        image_arr, mask_arr = im.numpy(), ms.numpy()
        # print(image_arr.shape, mask_arr.shape)
        # Prediction on large Image
        output, _ = predict_on_array(model=model,
                                     arr=image_arr[0],
                                     in_shape=(256, 256, 7),
                                     out_bands=1,
                                     stride=200,
                                     batchsize=20,
                                     augmentation=True,
                                     verbose=False,
                                     report_time=True)
        # output = (output > 0.5) * 1
        acc_iou = iou(mask_arr[0], output)
        acc.append(acc_iou)
        output = (output > 0.5) * 1
        # Display the results
        # plt.subplot(131)
        # plt.imshow(image_arr[0, :, :, :3])
        # plt.xlabel('image')
        # plt.xticks([])
        # plt.yticks([])

        # plt.subplot(132)
        # plt.imshow(rgb_mask(mask_arr[0, :, :, 0]))
        # plt.xlabel('mask')
        # plt.xticks([])
        # plt.yticks([])

        # plt.subplot(133)
        # plt.imshow(rgb_mask(output[:, :, 0]))
        # plt.xlabel('mask_pre')
        # plt.title('Iou:{:.2%}'.format(acc_iou))
        # plt.xticks([])
        # plt.yticks([])

        # plt.show()
        # Write out prediction to Tif file with coordinates
        # write_geotiff(output_path, output, image_path)
        # break
        # print('Writing out finish!')
    print(acc)
    print('Average Iou acc:{:.2%}'.format(np.mean(acc)))




