#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
from matplotlib import pyplot as plt

from patches_cls_dk.resnet_cls import ResnetCls
from utils import load_files_paths, read_imgs_with_masks, get_foldwise_split, norm_img, \
    get_experiment_dir, get_experiment_model_name


def main(config):
    folds_count = config['patches_cls_dk']['folds_count']
    imgs_dir = config['patches_cls_dk']['imgs_dir']
    masks_dir = config['patches_cls_dk']['masks_dir']
    patch_size = config['patches_cls_dk']['patch_size']
    stride = config['patches_cls_dk']['stride']
    experiment_name = config['patches_cls_dk']['experiment_cls_name']
    experiment_type = config['experiment_type']
    experiment_artifacts_dir = config['experiment_artifacts_root_dir']
    print(f'folds_count: {folds_count}')
    print(f'imgs_dir: {imgs_dir}')
    print(f'masks_dir: {masks_dir}')
    print(f'experiment_name: {experiment_name}')
    print(f'experiment_type: {experiment_type}')
    print(f'experiment_artifacts_dir: {experiment_artifacts_dir}')

    patch_h, patch_w = patch_size
    stride_h, stride_w = stride

    imgs_masks_pairs = load_files_paths(imgs_dir, masks_dir)

    _, _, test_set = get_foldwise_split(0, folds_count, imgs_masks_pairs)

    models = []
    for fold_no in range(folds_count):
        model = ResnetCls([patch_h, patch_w, 3], get_experiment_model_name(experiment_name, fold_no), '')
        try:
            model_file_dir = get_experiment_dir(experiment_artifacts_dir, experiment_name, experiment_type)
            model_file_name = get_experiment_model_name(experiment_name, fold_no)
            model.load(os.path.join(model_file_dir, model_file_name))
            print(f'Loaded model for fold {fold_no}')
            models.append(model)
        except IOError:
            print(f'No model for fold {fold_no}')

    test_imgs, test_masks = read_imgs_with_masks(test_set)

    img_h, img_w = test_imgs.shape[:2]

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i, (img, mask) in enumerate(zip(test_imgs, test_masks)):
        print(f'calculating {i+1}/{len(test_imgs)}')
        img = norm_img(img.astype(np.float32))
        mask_probabs = np.zeros(img.shape[:2], dtype=np.float32)
        mask_overlap = np.zeros(img.shape[:2], dtype=np.uint8)
        h = 0
        while h+patch_h < img_h:
            w = 0
            while w+patch_w < img_w:
                patch = img[h:h+patch_h, w:w+patch_w]
                patch.shape = [1, patch_h, patch_w, 3]
                probab = 0.0
                for model in models:
                    probab += model.model.predict(patch) / len(models)
                mask_probabs[h:h + patch_h, w:w + patch_w] += probab
                mask_overlap[h:h + patch_h, w:w + patch_w] += 1

                if np.round(probab) == 1.0 and mask[h:h + patch_h, w:w + patch_w].sum() > 0.0:
                    tp += 1
                elif np.round(probab) == 1.0 and mask[h:h + patch_h, w:w + patch_w].sum() == 0.0:
                    fp += 1
                elif np.round(probab) == 0.0 and mask[h:h + patch_h, w:w + patch_w].sum() == 0.0:
                    tn += 1
                elif np.round(probab) == 0.0 and mask[h:h + patch_h, w:w + patch_w].sum() >= 1.0:
                    fn += 1
                else:
                    raise ArithmeticError('Should not happened')

                w += stride_w

            h += stride_h

        mask_overlap[mask_overlap == 0] = 1
        mask_probabs /= mask_overlap

        mask_preds = np.round(mask_probabs)

        '''
        fig, axs = plt.subplots(1, 3)
        
        mask_preds_3ch = np.zeros(img.shape, np.float32)
        mask_preds_3ch[:, :, 0] = mask[:, :, 0] / 255.0
        mask_preds_3ch[:, :, 2] = np.round(mask_probabs)

        mask_probabs_3ch = np.zeros(img.shape, dtype=np.float32)
        mask_probabs_3ch[:, :, 0] = mask[:, :, 0] / 255.0
        mask_probabs_3ch[:, :, 2] = mask_probabs

        axs[0].imshow(img)
        axs[1].imshow(mask_probabs_3ch)
        axs[2].imshow(mask_preds_3ch)
        plt.show()
        pass
        '''
    se = tp / (tp + fn)
    ppv = tp / (tp + fp)
    sp = tn / (tn + fp)
    npv = tn / (tn + fn)

    print(f'se: {se}')
    print(f'pp: {ppv}')
    print(f'se: {sp}')
    print(f'pp: {npv}')

    #dice = calculate_dice(test_imgs, test_masks, pred_func, (models, ))
    #print(dice)


def pred_func(img, mask, *pred_func_additional_args):
    models = pred_func_additional_args[0][0]

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
        top_right_probabs += model.model.predict(top_right)[0, :, :, :] / len(models)
        top_left_probabs += model.model.predict(top_left)[0, :, :, :] / len(models)
        bottom_right_probabs += model.model.predict(bottom_right)[0, :, :, :] / len(models)
        bottom_left_probabs += model.model.predict(bottom_left)[0, :, :, :] / len(models)

    probabs = np.zeros(mask.shape, dtype=np.float32)
    probabs_overlap_counter = np.zeros(mask.shape, dtype=np.float32)

    probabs[:320, :480] += top_right_probabs
    probabs_overlap_counter[:320, :480] += 1.0

    probabs[:320, img.shape[1] - 480:] += top_left_probabs
    probabs_overlap_counter[:320, img.shape[1] - 480:] += 1.0

    probabs[img.shape[0] - 320:, :480] += bottom_right_probabs
    probabs_overlap_counter[img.shape[0] - 320:, :480] += 1.0

    probabs[img.shape[0] - 320:, img.shape[1] - 480:] += bottom_left_probabs
    probabs_overlap_counter[img.shape[0] - 320:, img.shape[1] - 480:] += 1.0

    probabs /= probabs_overlap_counter

    preds = np.round(probabs)

    return preds


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('folds_count', type=int, help='folds count in experiment')
    arg_parser.add_argument('imgs_dir', type=str, help='Directory with images.')
    arg_parser.add_argument('masks_dir', type=str, help='Directory with masks.')
    arg_parser.add_argument('--experiment_name', type=str, default='cls',
                            help='needed to define model name, it will be like experiment_name_fold_no.h5')

    args = arg_parser.parse_args()
    main(args.folds_count,
         args.imgs_dir,
         args.masks_dir,
         args.experiment_name)
