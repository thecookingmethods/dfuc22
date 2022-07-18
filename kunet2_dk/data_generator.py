import math

import numpy as np
from imgaug import augmenters as iaa
from tensorflow import keras
from skimage import color
from skimage.measure import regionprops, label
from skimage.transform import rescale, resize, downscale_local_mean

from utils import norm_img


class DataGenerator(keras.utils.Sequence):
    def __init__(self, images, masks, batch_size, network_input_wh, training=False):
        self._batch_size = batch_size
        self._images = images
        self._masks = masks

        self._network_input_wh = network_input_wh
        self._image_wh = self._images[0].shape[:-1]
        self._network_input_channels = self._images[0].shape[-1] * 2  # rgb + lab
        self._images_channels = self._images[0].shape[-1]

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
                    #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
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

        images = np.zeros((current_batch_size, *self._network_input_wh, self._images_channels), dtype=np.uint8)
        masks = np.zeros((current_batch_size, *self._network_input_wh, 1), dtype=np.uint8)

        for i, idx in enumerate(range(slice_idx_from, slice_idx_to)):
            img_tmp = self._images[idx] / 255.0
            img_tmp_resized = resize(img_tmp, self._network_input_wh)
            images[i] = (img_tmp_resized * 255).astype(np.uint8)

            mask_tmp = self._masks[idx] / 255.0
            mask_tmp_resized = resize(mask_tmp, self._network_input_wh)
            masks[i] = (mask_tmp_resized * 255.0).astype(np.uint8)

        if self._training:
            images, masks = self._aug(images=images, segmentation_maps=masks)

        x_out = np.zeros((current_batch_size, *self._network_input_wh, self._network_input_channels), dtype=np.float32)
        y_out = np.zeros((current_batch_size, 3), dtype=np.float32)

        for a in range(current_batch_size):
            image = images[a]
            mask = masks[a]
            labeled_mask = label(mask[:, :, 0] / 255.0)
            regions = regionprops(labeled_mask)
            if len(regions) == 0:
                continue
            region_idx = np.random.randint(len(regions))
            for r_idx in range(len(regions)):
                if r_idx == region_idx:
                    continue
                for ch in range(image.shape[2]):
                    image[regions[r_idx].bbox[0]:regions[r_idx].bbox[2], regions[r_idx].bbox[1]:regions[r_idx].bbox[3], ch] = np.mean(image[regions[r_idx].bbox[0]:regions[r_idx].bbox[2], regions[r_idx].bbox[1]:regions[r_idx].bbox[3], ch])
            region = regions[region_idx]

            center_h, center_w = [int(region.centroid[0]), int(region.centroid[1])]
            h, w = region.bbox[2] - region.bbox[0], region.bbox[3] - region.bbox[1]
            r = np.sqrt(h**2 + w**2)

            y_out[a, 0] = center_h / self._network_input_wh[0]
            y_out[a, 1] = center_w / self._network_input_wh[1]
            y_out[a, 2] = r / ((self._network_input_wh[0] + self._network_input_wh[1]) / 2)

            x_rgb = image
            x_lab = color.rgb2lab(x_rgb)

            x_rgb = norm_img(x_rgb.astype(np.float32))
            x_lab[:, :, 0] = x_lab[:, :, 0] / 100.0
            x_lab[:, :, 1] = (x_lab[:, :, 1] + 127.0) / 255.0
            x_lab[:, :, 2] = (x_lab[:, :, 2] + 127.0) / 255.0

            x_out[a, :, :, :3] = x_rgb
            x_out[a, :, :, 3:] = x_lab

        return x_out, y_out
