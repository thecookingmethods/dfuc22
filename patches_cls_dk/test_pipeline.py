#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
from matplotlib import pyplot as plt

from patches_cls_dk.resnet_cls import ResnetCls
from patches_cls_dk.unetlike_segm import UnetlikeSegm
from utils import load_files_paths, read_imgs_with_masks, get_foldwise_split, norm_img, \
    get_experiment_dir, get_experiment_model_name


def main(config):
    folds_count = config['patches_cls_dk']['folds_count']
    imgs_dir = config['patches_cls_dk']['imgs_dir']
    masks_dir = config['patches_cls_dk']['masks_dir']
    patch_size = config['patches_cls_dk']['patch_size']
    stride = config['patches_cls_dk']['stride']
    experiment_cls_name = config['patches_cls_dk']['experiment_cls_name']
    experiment_segm_name = config['patches_cls_dk']['experiment_segm_name']
    experiment_type = config['experiment_type']
    experiment_artifacts_dir = config['experiment_artifacts_root_dir']

    print(f'folds_count: {folds_count}')
    print(f'imgs_dir: {imgs_dir}')
    print(f'masks_dir: {masks_dir}')
    print(f'experiment_cls_name: {experiment_cls_name}')
    print(f'experiment_segm_name: {experiment_segm_name}')
    print(f'experiment_type: {experiment_type}')
    print(f'experiment_artifacts_dir: {experiment_artifacts_dir}')

    patch_h, patch_w = patch_size
    stride_h, stride_w = stride

    imgs_masks_pairs = load_files_paths(imgs_dir, masks_dir)

    _, _, test_set = get_foldwise_split(0, folds_count, imgs_masks_pairs)

    models_cls = []
    for fold_no in range(folds_count):
        model_cls = ResnetCls([patch_h, patch_w, 3], get_experiment_model_name(experiment_cls_name, fold_no), '')
        try:
            model_file_dir = get_experiment_dir(experiment_artifacts_dir, experiment_cls_name, experiment_type)
            model_file_name = get_experiment_model_name(experiment_cls_name, fold_no)
            model_cls.load(os.path.join(model_file_dir, model_file_name))
            print(f'Loaded cls model for fold {fold_no}')
            models_cls.append(model_cls)
        except IOError:
            print(f'No cls model for fold {fold_no}')

    models_segm = []
    for fold_no in range(folds_count):
        model_segm = UnetlikeSegm([patch_h, patch_w, 3], get_experiment_model_name(experiment_segm_name, fold_no), '')
        try:
            model_file_dir = get_experiment_dir(experiment_artifacts_dir, experiment_segm_name, experiment_type)
            model_file_name = get_experiment_model_name(experiment_segm_name, fold_no)
            model_segm.load(os.path.join(model_file_dir, model_file_name))
            print(f'Loaded segm model for fold {fold_no}')
            models_segm.append(model_segm)
        except IOError:
            print(f'No segm model for fold {fold_no}')

    test_imgs, test_masks = read_imgs_with_masks(test_set)

    img_h, img_w = test_imgs.shape[:2]

    dice_sum = 0.0
    for i, (img, mask) in enumerate(zip(test_imgs, test_masks)):
        print(f'calculating {i + 1}/{len(test_imgs)}')
        img = norm_img(img.astype(np.float32))
        mask_probabs = np.zeros(img.shape[:2], dtype=np.float32)
        mask_overlap = np.zeros(img.shape[:2], dtype=np.uint8)
        h = 0
        while h + patch_h < img_h:
            w = 0
            while w + patch_w < img_w:
                patch = img[h:h + patch_h, w:w + patch_w]
                patch.shape = [1, patch_h, patch_w, 3]
                probab = 0.0
                for model_cls in models_cls:
                    probab += model_cls.model.predict(patch) / len(models_cls)
                if probab >= 0.5:
                    segm_probabs = 0.0
                    for model_segm in models_segm:
                        segm_probabs += model_segm.model.predict(patch) / len(models_segm)
                        mask_probabs[h:h + patch_h, w:w + patch_w] += segm_probabs[0, :, :, 0]
                        mask_overlap[h:h + patch_h, w:w + patch_w] += 1

                w += stride_w

            h += stride_h

        mask_overlap[mask_overlap == 0] = 1
        mask_probabs /= mask_overlap

        mask_preds = np.round(mask_probabs)

        up = (2 * mask[:,:,0] * mask_preds).sum()
        down = (mask[:,:,0] + mask_preds).sum()
        dice = up / down
        dice_sum += dice

    dice_per_image = dice_sum / len(test_imgs)

    print(f'dice_per_image: {dice_per_image}')

