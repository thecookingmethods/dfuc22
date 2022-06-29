#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import glob

from unetlike import Unetlike
from utils import norm_img

import imageio.v3 as iio
import numpy as np

DATASET_DIR = '/home/darekk/dev/dfuc2022_challenge/DFUC2022_val_release'
DEST_MASKS_DIR = '/home/darekk/dev/dfuc2022_challenge/DFUC2022_val_release_preds_masks'


def main():
    folds_count = 5
    experiment_name = 'long_train_less_aug'

    models = []
    for fold_no in range(folds_count):
        model = Unetlike([320, 480, 3], f'{experiment_name}_{fold_no}')
        model.load(f'{experiment_name}_{fold_no}.h5')
        models.append(model)

    files_paths = load_files(DATASET_DIR)
    imgs = read_images(files_paths)

    save_masks(imgs, DEST_MASKS_DIR, models)


def save_masks(imgs, dest_masks_dir, models):
    os.makedirs(dest_masks_dir, exist_ok=True)
    folds_count = len(models)
    for i, (code, img) in enumerate(imgs.items()):
        print(f'{i + 1}/{len(imgs)}: {code}')
        img = img.astype(np.float32)
        img = norm_img(img)
        top_right = img[:320, :480, :]
        top_left = img[:320, img.shape[1] - 480:, :]
        bottom_right = img[img.shape[0] - 320:, :480, :]
        bottom_left = img[img.shape[0] - 320:, img.shape[1] - 480:, :]

        top_right.shape = [1, *top_right.shape]
        top_left.shape = [1, *top_left.shape]
        bottom_right.shape = [1, *bottom_right.shape]
        bottom_left.shape = [1, *bottom_left.shape]

        top_right_probabs = 0.0
        top_left_probabs = 0.0
        bottom_right_probabs = 0.0
        bottom_left_probabs = 0.0

        for model in models:
            top_right_probabs += model.model.predict(top_right)[0, :, :, :] / folds_count
            top_left_probabs += model.model.predict(top_left)[0, :, :, :] / folds_count
            bottom_right_probabs += model.model.predict(bottom_right)[0, :, :, :] / folds_count
            bottom_left_probabs += model.model.predict(bottom_left)[0, :, :, :] / folds_count

        probabs = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        probabs_overlap_counter = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

        probabs[:320, :480] += top_right_probabs[:, :, 0]
        probabs_overlap_counter[:320, :480] += 1.0

        probabs[:320, img.shape[1] - 480:] += top_left_probabs[:, :, 0]
        probabs_overlap_counter[:320, img.shape[1] - 480:] += 1.0

        probabs[img.shape[0] - 320:, :480] += bottom_right_probabs[:, :, 0]
        probabs_overlap_counter[img.shape[0] - 320:, :480] += 1.0

        probabs[img.shape[0] - 320:, img.shape[1] - 480:] += bottom_left_probabs[:, :, 0]
        probabs_overlap_counter[img.shape[0] - 320:, img.shape[1] - 480:] += 1.0

        probabs /= probabs_overlap_counter

        preds = np.round(probabs)
        preds *= 255.0
        preds = preds.astype(np.uint8)

        iio.imwrite(os.path.join(dest_masks_dir, f'{code}.png'), preds)


def load_files(dataset_dir):
    img_file_pattern = "*.jpg"
    imgs_files_paths = list(sorted(glob.glob(os.path.join(dataset_dir, img_file_pattern))))
    return imgs_files_paths


def read_images(files_paths):
    imgs = {}

    for img_path in files_paths:
        img = iio.imread(img_path)

        code = os.path.split(img_path)[-1].split('.')[0]

        imgs[code] = img
    return imgs


if __name__ == "__main__":
    main()
