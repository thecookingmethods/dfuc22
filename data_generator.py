import math

import numpy as np
from imgaug import augmenters as iaa
from tensorflow import keras

from utils import norm_img


class DataGenerator(keras.utils.Sequence):
    def __init__(self, images, masks, batch_size, network_input_wh, training=False):
        self._batch_size = batch_size
        self._images = images
        self._masks = masks

        self._network_input_wh = network_input_wh
        self._image_wh = self._images[0].shape[:-1]
        self._image_channels = self._images[0].shape[-1]

        self._no_of_examples = len(self._images)

        translation_ratio = [(y-x)//4 for x, y in zip(self._network_input_wh, self._image_wh)]

        self._training = training
        if self._training:
            self._aug = iaa.Sequential([

                iaa.Fliplr(0.25),  # horizontal flips
                iaa.Flipud(0.25),  # vertical flip
                # Small gaussian blur with random sigma between 0 and 0.5.
                # But we only blur about 10% of all images.
                iaa.Sometimes(0.1, iaa.GaussianBlur(sigma=(0, 0.5))),
                # Strengthen or weaken the contrast in each image.
                iaa.Sometimes(0.2, iaa.LinearContrast((0.95, 1.1))),
                # Add gaussian noise.
                # For 50% of all images, we sample the noise once per pixel.
                # For the other 50% of all images, we sample the noise per pixel AND
                # channel. This can change the color (not only brightness) of the
                # pixels.
                #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                # Make some images brighter and some darker.
                # In 20% of all cases, we sample the multiplier once per channel,
                # which can end up changing the color of the images.
                iaa.Sometimes(0.5, iaa.Multiply((0.9, 1.1), per_channel=0.2)),
                # Apply affine transformations to each image.
                # Scale/zoom them, translate/move them, rotate them and shear them.
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
            rows_offset, cols_offset = [(y - x) // 2 for x, y in zip(self._network_input_wh, self._image_wh)]
            x_out[a, :, :, :] = x[a, rows_offset:rows_offset + self._network_input_wh[0],
                                     cols_offset:cols_offset + self._network_input_wh[1], :]
            y_out[a, :, :, :] = y[a, rows_offset:rows_offset + self._network_input_wh[0],
                                     cols_offset:cols_offset + self._network_input_wh[1], :]

        x_out = x_out.astype(np.float32)
        y_out = y_out.astype(np.float32)
        x_out = norm_img(x_out)
        y_out = norm_img(y_out)

        return x_out, y_out
