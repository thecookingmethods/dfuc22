import math

import numpy as np
from imgaug import augmenters as iaa
from tensorflow import keras
from skimage import color

from utils import norm_img


class DataGenerator(keras.utils.Sequence):
    def __init__(self, images, masks, batch_size, network_input_wh, training=False):
        self._batch_size = batch_size
        self._images = images
        self._masks = masks

        self._network_input_wh = network_input_wh
        self._image_wh = self._images[0].shape[:-1]
        self._image_channels = self._images[0].shape[-1] * 2  # rgb + lab

        self._no_of_examples = len(self._images)

        translation_ratio = [(y-x)//4 for x, y in zip(self._network_input_wh, self._image_wh)]

        self._training = training
        if self._training:
            self._aug = iaa.Sequential([

                iaa.Fliplr(0.25),
                iaa.Flipud(0.25),
                iaa.Sometimes(0.5, iaa.Multiply((0.9, 1.1), per_channel=0.2)),
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-90, 90),
                    shear=(-8, 8)
                )

            ], random_order=True)

    def __len__(self):
        return int(math.ceil(self._no_of_examples / self._batch_size))

    def __getitem__(self, idx):
        slice_idx_from = idx * self._batch_size
        slice_idx_to = (idx + 1) * self._batch_size
        if slice_idx_to > self._no_of_examples:
            slice_idx_to = self._no_of_examples

        current_batch_size = slice_idx_to - slice_idx_from

        x = np.array(self._images[slice_idx_from:slice_idx_to, :, :, :])
        y = np.array(self._masks[slice_idx_from:slice_idx_to, :, :, :])

        if self._training:
            x, y = self._aug(images=x, segmentation_maps=y)

        x_out = np.zeros((current_batch_size, *self._network_input_wh, self._image_channels), dtype=np.float32)
        y_out = np.zeros((current_batch_size, *self._network_input_wh, 1), dtype=np.float32)
        for a in range(x_out.shape[0]):
            if not self._training:
                rows_offset, cols_offset = [(y - x) // 2 for x, y in zip(self._network_input_wh, self._image_wh)]
            else:
                rows_offset = np.random.randint(0, self._image_wh[0] - self._network_input_wh[0])
                cols_offset = np.random.randint(0, self._image_wh[1] - self._network_input_wh[1])
            x_rgb = x[a, rows_offset:rows_offset + self._network_input_wh[0],
                                     cols_offset:cols_offset + self._network_input_wh[1], :]

            x_lab = color.rgb2lab(x_rgb)

            x_rgb = norm_img(x_rgb.astype(np.float32))
            x_lab[:, :, 0] = x_lab[:, :, 0] / 100.0
            x_lab[:, :, 1] = (x_lab[:, :, 1] + 127.0) / 255.0
            x_lab[:, :, 2] = (x_lab[:, :, 2] + 127.0) / 255.0

            x_out[a, :, :, :3] = x_rgb
            x_out[a, :, :, 3:] = x_lab

            mask = y[a, rows_offset:rows_offset + self._network_input_wh[0],
                   cols_offset:cols_offset + self._network_input_wh[1], :]
            mask = norm_img(mask.astype(np.float32))
            y_out[a, :, :, :] = mask

        return x_out, y_out
