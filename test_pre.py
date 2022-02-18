import tensorflow as tf
from sys import stdout
from dataloder import dataset, get_path
from utility import rgb_mask, get_image, save_array_to_mat
import time
import numpy as np
import pandas as pd
from loss import dice_loss, iou, tree_iou
from matplotlib import pyplot as plt


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
            prediction = model.predict_on_batch(batch)
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
    ks = [259, 215, 210, 66, 150]
    for k in ks:

    #########################################################################################################
        image_201_path = f'../quality/images/tile_{k}.tif'
        image_201 = get_image(image_201_path)
        # model_high = tf.keras.models.load_model(f'../results/high_low_2/unet_res_high_2',
        #                                         custom_objects={'dice_loss': dice_loss,
        #                                                         'iou': iou,
        #                                                         'tree_iou': tree_iou})
        # pre_201 = predict_on_array(model=model_high,
        #                            arr=image_201,
        #                            in_shape=(256, 256, 7),
        #                            out_bands=2,
        #                            stride=200,
        #                            batchsize=20,
        #                            augmentation=True,
        #                            verbose=False,
        #                            report_time=False)
        # # print(pre_201.shape)
        # save_array_to_mat(pre_201, f'../quality/mix/{k}_h.mat')
        #
        # model_low = tf.keras.models.load_model(f'../results/high_low_2/unet_res_low_2.h5',
        #                                        custom_objects={'dice_loss': dice_loss,
        #                                                        'iou': iou,
        #                                                        'tree_iou': tree_iou})
        # pre_201 = predict_on_array(model=model_low,
        #                            arr=image_201,
        #                            in_shape=(256, 256, 7),
        #                            out_bands=2,
        #                            stride=200,
        #                            batchsize=20,
        #                            augmentation=True,
        #                            verbose=False,
        #                            report_time=False)
        # # print(pre_201.shape)
        # save_array_to_mat(pre_201, f'../quality/mix/{k}_l.mat')

        # print(image_201.shape)
        for i in np.arange(2, 9, 2):
            for j in np.arange(4, 8, 1):
                model_1 = tf.keras.models.load_model(f'../results/mix/random/unet_mix_mask_{i}_{j}',
                                                     custom_objects={'dice_loss': dice_loss,
                                                                     'iou': iou,
                                                                     'tree_iou': tree_iou})

                pre_201 = predict_on_array(model=model_1,
                                           arr=image_201,
                                           in_shape=(256, 256, 7),
                                           out_bands=2,
                                           stride=200,
                                           batchsize=20,
                                           augmentation=True,
                                           verbose=False,
                                           report_time=False)
                # print(pre_201.shape)
                save_array_to_mat(pre_201, f'../quality/mix/{k}_mix_{i}_{j}.mat')
                print(f'mix model {k} {i} {j}, prediction finish')
    #################################################################################################################
    ############################### active learning ############################
    # image_201_path = f'../quality/images/tile_{k}.tif'
    # image_201 = get_image(image_201_path)
    # model_initial = tf.keras.models.load_model(f'../results/active/low/unet_active_1',
    #                                            custom_objects={'dice_loss': dice_loss,
    #                                                            'iou': iou,
    #                                                            'tree_iou': tree_iou})
    # pre_201 = predict_on_array(model=model_initial,
    #                            arr=image_201,
    #                            in_shape=(256, 256, 7),
    #                            out_bands=2,
    #                            stride=200,
    #                            batchsize=20,
    #                            augmentation=True,
    #                            verbose=False,
    #                            report_time=False)
    # save_array_to_mat(pre_201, f'../quality/active/low/{k}_initial.mat')
    #
    # for i in np.arange(2, 8, 1):
    #     model_1 = tf.keras.models.load_model(f'../results/active/low/decay/old_decay/unet_active_{i}',
    #                                          custom_objects={'dice_loss': dice_loss,
    #                                                          'iou': iou,
    #                                                          'tree_iou': tree_iou})
    #     outputs_1 = np.zeros((5,) + pre_201.shape, dtype=np.float32)
    #     for j in range(5):
    #         pre_201 = predict_on_array(model=model_1,
    #                                    arr=image_201,
    #                                    in_shape=(256, 256, 7),
    #                                    out_bands=2,
    #                                    stride=200,
    #                                    batchsize=20,
    #                                    augmentation=True,
    #                                    verbose=False,
    #                                    report_time=False)
    #         outputs_1[j] = pre_201
    #     output_1 = np.mean(outputs_1, axis=0)
    #     # print(pre_201.shape)
    #     save_array_to_mat(output_1, f'../quality/active/low/{k}_active_{i}.mat')
    #     print(f'active model {i}, prediction finish')
    ####################################################################################################
    # path = r'../quality/high/'
    # width, height = 256, 256
    # seed = 2
    #
    # # Trained Model loading
    # # model_1 = tf.keras.models.load_model(r'../results/active/high/decay/unet_active_2',
    # #                                      custom_objects={'dice_loss': dice_loss,
    # #                                                      'iou': iou,
    # #                                                      'tree_iou': tree_iou})
    #
    # # model_2 = tf.keras.models.load_model(r'../results/high_low_2/unet_res_high_2.h5',
    # #                                      custom_objects={'dice_loss': dice_loss,
    # #                                                      'iou': iou,
    # #                                                      'tree_iou': tree_iou})
    #
    # # Image loading for further prediction
    #
    # # image_158_path = r'../quality/images/tile_26.tif'
    # # mask_158_path = r'../quality/high/mask_26.tif'
    # #
    # # image_158 = get_image(image_158_path)
    # # print(image_158.shape)
    # # mask_158 = get_image(mask_158_path)
    #
    # image_path_test, mask_path_test, image_id_test = get_path(path=path,
    #                                                           mode='test',
    #                                                           seed=seed,
    #                                                           active=0)
    # mean_t, mean_o, name = [], [], []
    # for i in range(0, 1, 1):
    #     for j in range(2, 8, 1):
    #         print(f'{i}_{j} start predicting....')
    #         model = tf.keras.models.load_model(f'../results/active/low/percent/decay/unet_active_{j}',
    #                                            custom_objects={'dice_loss': dice_loss,
    #                                                            'iou': iou,
    #                                                            'tree_iou': tree_iou})
    #
    #         true_masks_cat, pre_masks_cat = [], []
    #         for image_path, mask_path, image_id in zip(image_path_test, mask_path_test, image_id_test):
    #             image = get_image(image_path)
    #             mask = get_image(mask_path)
    #             pre_158 = predict_on_array(model=model,
    #                                        arr=image,
    #                                        in_shape=(256, 256, 7),
    #                                        out_bands=2,
    #                                        stride=200,
    #                                        batchsize=20,
    #                                        augmentation=True,
    #                                        verbose=False,
    #                                        report_time=False)
    #             true_masks_cat.append(mask)
    #             pre_masks_cat.append(pre_158)
    #         # #         # acc1, acc2, acc3, acc4 = [], [], [], []
    #         # #         # for (im, ms), i in zip(dataset, image_id_test):
    #         # #         #     # print(im.shape, ms.shape)
    #         # #         #     image_arr, mask_arr = im.numpy(), ms.numpy()
    #         # #         #     # print(image_arr.shape, mask_arr.shape)
    #         # #         #     # Prediction on large Image
    #         # #         #     outputs_1, outputs_2 = np.zeros((5, ) + mask_arr[0].shape, dtype=np.float32), \
    #         # #         #                            np.zeros((5, ) + mask_arr[0].shape, dtype=np.float32)
    #         # #         #     for inf in range(5):
    #         # #         #         output_1, _ = predict_on_array(model=model_1,
    #         # #         #                                        arr=image_arr[0],
    #         # #         #                                        in_shape=(256, 256, 7),
    #         # #         #                                        out_bands=2,
    #         # #         #                                        stride=200,
    #         # #         #                                        batchsize=20,
    #         # #         #                                        augmentation=True,
    #         # #         #                                        verbose=False,
    #         # #         #                                        report_time=True)
    #         # #         #
    #         # #         #         output_2, _ = predict_on_array(model=model_2,
    #         # #         #                                        arr=image_arr[0],
    #         # #         #                                        in_shape=(256, 256, 7),
    #         # #         #                                        out_bands=2,
    #         # #         #                                        stride=200,
    #         # #         #                                        batchsize=20,
    #         # #         #                                        augmentation=True,
    #         # #         #                                        verbose=False,
    #         # #         #                                        report_time=True)
    #         # #         #         outputs_1[inf] = output_1
    #         # #         #         outputs_2[inf] = output_2
    #         # #         #
    #         # #         #     # output_1 = (output_1 > 0.5) * 1
    #         # #         #     output_1 = np.mean(outputs_1, axis=0)
    #         # #         #     output_2 = np.mean(outputs_2, axis=0)
    #         # #         #
    #         # #         #     acc_iou_1 = iou(mask_arr[0][:, :, 1], output_1[:, :, 1])
    #         # #         #     acc_iou_2 = iou(mask_arr[0], output_1)
    #         # #         #     acc1.append(acc_iou_1.numpy())
    #         # #         #     acc2.append(acc_iou_2.numpy())
    #         # #         #
    #         # #         #     # output_1 = (output_1 > 0.5) * 1
    #         # #         #
    #         # #         #     # output_2 = (output_2 > 0.5) * 1
    #         # #         #     acc_iou_3 = iou(mask_arr[0][:, :, 1], output_2[:, :, 1])
    #         # #         #     acc_iou_4 = iou(mask_arr[0], output_2)
    #         # #         #     acc3.append(acc_iou_3.numpy())
    #         # #         #     acc4.append(acc_iou_4.numpy())
    #         # #         #     # output_2 = np.argmax(output_2, axis=-1)
    #         # #         #
    #         # #         #     # Display the results
    #         # #         #     plt.figure(figsize=(12, 3))
    #         # #             acc_iou_1 = iou(mask[:, :, 1], pre_158[:, :, 1]).numpy()
    #         # #             acc_iou_2 = iou(mask, pre_158).numpy()
    #         # #             # print(acc_iou_1, acc_iou_2)
    #         # #             acc1.append(acc_iou_1)
    #         # #             acc2.append(acc_iou_2)
    #         true_mask_cat = np.concatenate(true_masks_cat, axis=0)
    #         pre_mask_cat = np.concatenate(pre_masks_cat, axis=0)
    #
    #         acc_iou_1 = iou(true_mask_cat[:, :, 1], pre_mask_cat[:, :, 1]).numpy()
    #         acc_iou_2 = iou(true_mask_cat, pre_mask_cat).numpy()
    #         print(f'tree iou for all 30 images:{acc_iou_1:.4f}')
    #         print(f'Overall iou for all 30 images:{acc_iou_2:.4f}')
    #         del model
    #         mean_t.append(acc_iou_1)
    #         mean_o.append(acc_iou_2)
    #         name.append((i, j))
    # #         # break
    # df_o = pd.DataFrame({'N': name, 'mean_tree_iou1': mean_t, 'mean_o_iou1': mean_o})
    # with pd.ExcelWriter(r'../results/new_results.xlsx', mode='a',
    #                     engine='openpyxl', if_sheet_exists='replace') as writer:
    #     df_o.to_excel(writer, sheet_name=f'low_active_percent_decay')
    # t_iou, o_iou, name = [], [], []
    # entropy = [0, 1, 2, 3, 4, 5, 6, 10, 50]
    # for i in entropy:
    #     for j in range(2, 8, 1):
    #         print(f'fixed entropy {i} active iteration {j} start predicting')
    #         model = tf.keras.models.load_model(f'../results/active/low/fixed/{i}/unet_active_{j}',
    #                                            custom_objects={'dice_loss': dice_loss,
    #                                                            'iou': iou,
    #                                                            'tree_iou': tree_iou})
    #         true_masks_cat, pre_masks_cat = [], []
    #         for image_path, mask_path, image_id in zip(image_path_test, mask_path_test, image_id_test):
    #             image = get_image(image_path)
    #             mask = get_image(mask_path)
    #             pre_158 = predict_on_array(model=model,
    #                                        arr=image,
    #                                        in_shape=(256, 256, 7),
    #                                        out_bands=2,
    #                                        stride=200,
    #                                        batchsize=20,
    #                                        augmentation=True,
    #                                        verbose=False,
    #                                        report_time=False)
    #             true_masks_cat.append(mask)
    #             pre_masks_cat.append(pre_158)
    #
    #         true_mask_cat = np.concatenate(true_masks_cat, axis=0)
    #         pre_mask_cat = np.concatenate(pre_masks_cat, axis=0)
    #
    #         # print(f'true mask cat shape:{true_mask_cat.shape}')
    #         # print(f'pre mask cat shape:{pre_mask_cat.shape}')
    #
    #         acc_iou_1 = iou(true_mask_cat[:, :, 1], pre_mask_cat[:, :, 1]).numpy()
    #         acc_iou_2 = iou(true_mask_cat, pre_mask_cat).numpy()
    #         print(f'tree iou for all 30 images:{acc_iou_1:.4f}')
    #         print(f'Overall iou for all 30 images:{acc_iou_2:.4f}')
    #         t_iou.append(acc_iou_1)
    #         o_iou.append(acc_iou_2)
    #         name.append((i, j))
    #
    # df_o = pd.DataFrame({'N': name, 'mean_tree_iou1': t_iou, 'mean_o_iou1': o_iou})
    # with pd.ExcelWriter(r'../results/new_results.xlsx', mode='a',
    #                     engine='openpyxl', if_sheet_exists='replace') as writer:
    #     df_o.to_excel(writer, sheet_name=f'low_active_fixed_overall')
    # with pd.ExcelWriter(r'../results/fine/random/de_freeze/random/de_freeze_random.xlsx', mode='a',
    #                     engine='openpyxl', if_sheet_exists='replace') as writer:
    #     df.to_excel(writer, sheet_name=f'fine_10_3')
    # del model
        # ax.imshow(rgb_mask(np.argmax(pre_158, axis=-1)))
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_xlabel(f'{(0.1*(i+1)):.0%} high')
        # ax.set_title(f'tree: {acc_iou_1:.2%}\n overall: {acc_iou_2:.2%}')
        # break
    # plt.show()
    #     plt.subplot(143)
    #     plt.imshow(rgb_mask(np.argmax(output_1, axis=-1)))
    #     plt.xlabel('mask_pre_low')
    #     plt.title('tree_iou:{:.2%}\n Iou:{:.2%}'.format(acc_iou_1, acc_iou_2))
    #     plt.xticks([])
    #     plt.yticks([])
    #
    #     plt.subplot(144)
    #     plt.imshow(rgb_mask(np.argmax(output_2, axis=-1)))
    #     plt.xlabel('mask_pre_high')
    #     plt.title('tree_iou:{:.2%}\n Iou:{:.2%}'.format(acc_iou_3, acc_iou_4))
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.savefig(f'../results/high_low_2/fig/image_{i}')
    #     # plt.show()
    #     # Write out prediction to Tif file with coordinates
    #     # write_geotiff(output_path, output, image_path)
    #     # break
    #     print(f'{i} Writing out finish!')
    # print(np.mean(acc1), np.mean(acc2))
    # print(acc1, acc2)
    # df = pd.DataFrame({'N': image_id_test, 'tree_iou1': acc1, 'o_iou1': acc2, 'tree_iou2': acc3, 'o_iou2': acc4})
    # print(df)
    # print(np.mean(acc1), np.mean(acc2), np.mean(acc3), np.mean(acc4))
    # with pd.ExcelWriter(r'../results/r.xlsx', mode='a',
    #                     engine='openpyxl', if_sheet_exists='replace') as writer:
    #     df.to_excel(writer, sheet_name='high-low_mc_dropout')




