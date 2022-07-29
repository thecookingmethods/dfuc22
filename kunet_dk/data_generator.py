import math

import numpy as np
from imgaug import augmenters as iaa
from tensorflow import keras
from skimage import color
from skimage.measure import label, regionprops

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
        return int(math.floor(self._no_of_examples / self._batch_size))

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

            mask = y[a].astype(np.float32)
            mask[mask > 0.5] = 1.0
            mask[mask <= 0.5] = 0.0
            labeled_mask = label(mask[:, :, 0])
            regions = regionprops(labeled_mask)
            if len(regions) > 0:
                region_idx = np.random.randint(len(regions))
                region = regions[region_idx]
                center_h, center_w = int(region.centroid[0]), int(region.centroid[1])
                from_h, to_h = region.bbox[0], region.bbox[2]
                from_w, to_w = region.bbox[1], region.bbox[3]

                rnd = np.random.randint(0, 100)
                if rnd < 75:
                    center_h_offset = np.random.randint(from_h + (center_h - from_h) // 4, to_h - (to_h - center_h) // 4)
                    center_w_offset = np.random.randint(from_w + (center_w - from_w) // 4, to_w - (to_w - center_w) // 4)
                elif 75 < rnd < 90:
                    center_h_offset = np.random.randint(center_h - self._network_input_wh[0], center_h + self._network_input_wh[0])
                    center_w_offset = np.random.randint(center_w - self._network_input_wh[1], center_w + self._network_input_wh[1])
                else:
                    center_h_offset = np.random.randint(self._network_input_wh[0] // 2,
                                                        self._image_wh[0] - self._network_input_wh[0] // 2)
                    center_w_offset = np.random.randint(self._network_input_wh[1] // 2,
                                                        self._image_wh[1] - self._network_input_wh[1] // 2)
            else:
                center_h_offset = np.random.randint(self._network_input_wh[0] // 2,
                                                    self._image_wh[0] - self._network_input_wh[0] // 2)
                center_w_offset = np.random.randint(self._network_input_wh[1] // 2,
                                                    self._image_wh[1] - self._network_input_wh[1] // 2)

            from_h = center_h_offset - self._network_input_wh[0]//2
            to_h = center_h_offset + self._network_input_wh[0]//2
            if from_h < 0:
                from_h = 0
                to_h = self._network_input_wh[0]
            elif to_h > self._image_wh[0]:
                from_h = self._image_wh[0] - self._network_input_wh[0]
                to_h = self._image_wh[0]

            from_w = center_w_offset - self._network_input_wh[1] // 2
            to_w = center_w_offset + self._network_input_wh[1] // 2
            if from_w < 0:
                from_w = 0
                to_w = self._network_input_wh[1]
            elif to_w > self._image_wh[1]:
                from_w = self._image_wh[1] - self._network_input_wh[1]
                to_w = self._image_wh[1]

            x_rgb = x[a, from_h:to_h, from_w:to_w, :]

            x_lab = color.rgb2lab(x_rgb)

            x_rgb = norm_img(x_rgb.astype(np.float32))
            x_lab[:, :, 0] = x_lab[:, :, 0] / 100.0
            x_lab[:, :, 1] = (x_lab[:, :, 1] + 127.0) / 255.0
            x_lab[:, :, 2] = (x_lab[:, :, 2] + 127.0) / 255.0

            x_out[a, :, :, :3] = x_rgb
            x_out[a, :, :, 3:] = x_lab

            mask = y[a, from_h:to_h, from_w:to_w, :]
            mask = norm_img(mask.astype(np.float32))
            y_out[a, :, :, :] = mask

        return x_out, y_out
