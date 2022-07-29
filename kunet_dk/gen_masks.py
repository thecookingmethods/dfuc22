#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import glob

from kunet_dk.unetlike import Unetlike
from utils import norm_img, get_experiment_model_name, get_experiment_dir, read_images, load_files

from skimage import color
import imageio.v3 as iio
import numpy as np
from skimage.measure import label, regionprops


def main(config):
    folds_count = config['kunet_dk']['folds_count']
    imgs_dir = config['kunet_dk']['imgs_dir']
    dest_masks_dir = config['kunet_dk']['dest_masks_dir']
    experiment_name = config['kunet_dk']['experiment_name']
    experiment_type = config['experiment_type']
    experiment_artifacts_dir = config['experiment_artifacts_root_dir']
    net_input_size = config["kunet_dk"]["net_input_size"]
    step_ratio = config["kunet_dk"]["step_ratio"]

    print(f'folds_count: {folds_count}')
    print(f'imgs_dir: {imgs_dir}')
    print(f'masks_dir: {dest_masks_dir}')
    print(f'experiment_name: {experiment_name}')
    print(f'experiment_type: {experiment_type}')
    print(f'experiment_artifacts_dir: {experiment_artifacts_dir}')

    models = []
    for fold_no in range(folds_count):
        model = Unetlike([*net_input_size, 6], get_experiment_model_name(experiment_name, fold_no), '')
        try:
            model_file_dir = get_experiment_dir(experiment_artifacts_dir, experiment_name, experiment_type)
            model_file_name = get_experiment_model_name(experiment_name, fold_no)
            model.load(os.path.join(model_file_dir, model_file_name))
            models.append(model)
        except OSError:
            print(f'No model for fold {fold_no}.')

    files_paths = load_files(imgs_dir)
    imgs = read_images(files_paths)

    save_masks(imgs, dest_masks_dir, models, net_input_size, step_ratio)


def save_masks(imgs, dest_masks_dir, models, net_input_size, step_ratio):
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

        size_h, size_w = net_input_size
        step_h, step_w = (img.shape[0] - size_h) // step_ratio, (img.shape[1] - size_w) // step_ratio

        preds_votes = []
        probabs_votes = []

        for model in models:
            probabs = np.zeros(img.shape[:2], dtype=np.float32)
            probabs_overlap_counter = np.zeros(img.shape[:2], dtype=np.float32)

            h = 0
            while h + size_h <= img.shape[0]:
                w = 0
                while w + size_w <= img.shape[1]:
                    patch = img_out[h:h + size_h, w:w + size_w, :]
                    patch.shape = [1, *patch.shape]
                    result = model._model.predict(patch)[0, :, :, 0]
                    probabs[h:h + size_h, w:w + size_w] += result
                    probabs_overlap_counter[h:h + size_h, w:w + size_w] += 1.0

                    w += step_w
                    if step_w == 0:
                        break
                h += step_h
                if step_h == 0:
                    break

            probabs_overlap_counter[probabs_overlap_counter == 0.0] = 1.0
            probabs /= probabs_overlap_counter
            preds = np.round(probabs)

            preds_votes.append(preds)
            probabs_votes.append(probabs)

        preds_votes = np.array(preds_votes)
        preds_votes = np.mean(preds_votes, axis=0)
        preds_votes = np.round(preds_votes)

        probabs_votes = np.array(probabs_votes)
        probabs_votes = np.mean(probabs_votes, axis=0)

        preds = np.round(probabs_votes)

        preds = preds.astype(np.float32)
        preds[preds > 0.5] = 1.0
        preds[preds <= 0.5] = 0.0
        labeled_mask = label(preds)
        regions = regionprops(labeled_mask)
        for region in regions:
            area = region.area
            height = region.bbox[2] - region.bbox[0]
            width = region.bbox[3] - region.bbox[1]
            if area < 91 or height < 9 or width < 8:
                preds[region.bbox[0] - 1:region.bbox[2] + 2, region.bbox[1] - 1:region.bbox[3] + 2] = 0.0

        preds *= 255.0
        preds = preds.astype(np.uint8)

        iio.imwrite(os.path.join(dest_masks_dir, f'{code}.png'), preds)


if __name__ == "__main__":
    main()
