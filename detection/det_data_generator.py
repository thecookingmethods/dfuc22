import math

import numpy as np
from imgaug import augmenters as iaa
from tensorflow import keras
from skimage.measure import regionprops, label

from utils import norm_img


class DetDataGenerator(keras.utils.Sequence):
    def __init__(self, images, masks, batch_size, network_input_hw, channels, training=False):
        assert batch_size % 2 == 0, "batch size must be even"
        self._batch_size = batch_size
        self._images = images
        self._masks = masks

        self._network_input_hw = network_input_hw
        self._network_input_channels = channels
        self._image_hw = self._images[0].shape[:-1]
        self._image_channels = self._images[0].shape[-1]

        self._no_of_examples = len(self._images)

        translation_ratio = [(y - x) // 4 for x, y in zip(self._network_input_hw, self._image_hw)]

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
        masks = np.array(self._masks[slice_idx_from:slice_idx_to, :, :, :])

        if self._training:
            x, masks = self._aug(images=x, segmentation_maps=masks)

        x_out = np.zeros((current_batch_size, *self._network_input_hw, self._network_input_channels), dtype=np.float32)
        y_out = np.zeros((current_batch_size, *self._network_input_hw, 3), dtype=np.float32)

        a = 0
        while a < x_out.shape[0]:
            rows_offset = np.random.randint(0, self._image_hw[0] - self._network_input_hw[0])
            cols_offset = np.random.randint(0, self._image_hw[1] - self._network_input_hw[1])

            x_out[a, :, :, :] = x[a, rows_offset:rows_offset + self._network_input_hw[0],
                                  cols_offset:cols_offset + self._network_input_hw[1], :]
            mask_aug = masks[a, rows_offset:rows_offset + self._network_input_hw[0],
                             cols_offset:cols_offset + self._network_input_hw[1], :]

            m_c = np.zeros((*self._network_input_hw, 1), dtype=np.float32)
            m_hw = np.zeros((*self._network_input_hw, 2), dtype=np.float32)

            mask = mask_aug.astype(np.float32) / 255.0
            labeled_mask = label(mask[:, :, 0])
            regions = regionprops(labeled_mask)
            for region in regions:
                from_h, from_w, to_h, to_w = region.bbox
                h = to_h - from_h
                w = to_w - from_w
                c_h_offset = np.random.randint(-h // 10, h // 10)
                c_w_offset = np.random.randint(-w // 10, w // 10)

                if to_h + c_h_offset > self._network_input_hw[0]:
                    to_h = self._network_input_hw[0]
                    from_h += c_h_offset
                elif from_h + c_h_offset < 0:
                    to_h += c_h_offset
                    from_h = 0
                else:
                    to_h += c_h_offset
                    from_h += c_h_offset

                if to_w + c_w_offset > self._network_input_hw[1]:
                    to_w += self._network_input_hw[1]
                    from_w += c_w_offset
                elif from_w + c_w_offset < 0:
                    to_w += c_w_offset
                    from_w = 0
                else:
                    to_w += c_w_offset
                    from_w += c_w_offset

                c_h = (from_h + to_h) // 2
                c_w = (from_w + to_w) // 2

                m_c[c_h, c_w, 0] = 1.0

                h = to_h - from_h
                w = to_w - from_w

                m_hw[c_h, c_w, 0] = h
                m_hw[c_h, c_w, 1] = w

            y_out[a, :, :, 0] = m_c[:, :, 0]
            y_out[a, :, :, 1] = m_hw[:, :, 1]
            y_out[a, :, :, 2] = m_hw[:, :, 2]

        x_out = x_out.astype(np.float32)
        x_out = norm_img(x_out)

        return x_out, y_out

    @staticmethod
    def _unit_circle(r, hw):
        d = hw
        rx, ry = d / 2, d / 2
        x, y = np.indices((d, d))
        rx_x = rx - x
        ry_y = ry - y
        hypot = np.hypot(rx_x, ry_y)
        abs = np.abs(hypot)
        ones = np.zeros([hw, hw], dtype=np.float32)
        ones[abs < r] = 1.0
        return ones

    @staticmethod
    def _unit_square(hw_square, hw_image):

        x, y = np.indices((hw_image, hw_image))
        rx, ry = hw_image / 2, hw_image / 2
        rx_x = np.abs(rx - x)
        ry_y = np.abs(ry - y)
        ones = (rx_x < hw_square/2).astype(int) * (ry_y < hw_square/2).astype(int)
        return ones

