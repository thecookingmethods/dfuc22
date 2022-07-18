#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import glob

from kunet_dk.unetlike import Unetlike
from utils import norm_img, get_experiment_model_name, get_experiment_dir, read_images, load_files

from skimage import color
import imageio.v3 as iio
import numpy as np

DATASET_DIR = '/home/darekk/dev/dfuc2022_challenge/DFUC2022_val_release'
DEST_MASKS_DIR = '/home/darekk/dev/dfuc2022_challenge/DFUC2022_val_release_preds_masks'


def main(config):
    folds_count = config['kunet_dk']['folds_count']
    imgs_dir = config['kunet_dk']['imgs_dir']
    masks_dir = config['kunet_dk']['masks_dir']
    experiment_name = config['kunet_dk']['experiment_name']
    experiment_type = config['experiment_type']
    experiment_artifacts_dir = config['experiment_artifacts_root_dir']
    print(f'folds_count: {folds_count}')
    print(f'imgs_dir: {imgs_dir}')
    print(f'masks_dir: {masks_dir}')
    print(f'experiment_name: {experiment_name}')
    print(f'experiment_type: {experiment_type}')
    print(f'experiment_artifacts_dir: {experiment_artifacts_dir}')

    models = []
    for fold_no in range(folds_count):
        model = Unetlike([320, 480, 6], get_experiment_model_name(experiment_name, fold_no), '')
        try:
            model_file_dir = get_experiment_dir(experiment_artifacts_dir, experiment_name, experiment_type)
            model_file_name = get_experiment_model_name(experiment_name, fold_no)
            model.load(os.path.join(model_file_dir, model_file_name))
            models.append(model)
        except OSError:
            print(f'No model for fold {fold_no}.')

    files_paths = load_files(DATASET_DIR)
    imgs = read_images(files_paths)

    save_masks(imgs, DEST_MASKS_DIR, models)


def save_masks(imgs, dest_masks_dir, models):
    os.makedirs(dest_masks_dir, exist_ok=True)
    folds_count = len(models)
    for i, (code, img) in enumerate(imgs.items()):
        print(f'{i + 1}/{len(imgs)}: {code}')

        img_out = np.zeros([*img.shape[:2], 6], dtype=np.float32)

        img_lab = color.rgb2lab(img)

        img = img.astype(np.float32)
        img = norm_img(img)

        img_lab[:, :, 0] = img_lab[:, :, 0] / 100.0
        img_lab[:, :, 1] = (img_lab[:, :, 1] + 127.0) / 255.0
        img_lab[:, :, 2] = (img_lab[:, :, 2] + 127.0) / 255.0

        img_out[:, :, :3] = img
        img_out[:, :, 3:] = img_lab

        img = img_out

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


if __name__ == "__main__":
    main()
