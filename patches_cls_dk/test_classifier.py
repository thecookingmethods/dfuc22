#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse

import numpy as np
from matplotlib import pyplot as plt

from resnet_cls import ResnetCls
from utils import load_files_paths, read_imgs_with_masks, get_foldwise_split, calculate_dice, norm_img


def main(folds_count, imgs_dir, masks_dir, experiment_name):
    print(f'folds_count: {folds_count}')
    print(f'imgs_dir: {imgs_dir}')
    print(f'masks_dir: {masks_dir}')
    print(f'experiment_name: {experiment_name}')
    imgs_masks_pairs = load_files_paths(imgs_dir, masks_dir)

    _, _, test_set = get_foldwise_split(0, folds_count, imgs_masks_pairs)

    models = []
    for fold_no in range(folds_count):
        model = ResnetCls([96, 96, 3], f'{experiment_name}_{fold_no}')
        try:
            model.load(f'{experiment_name}_{fold_no}.h5')
            models.append(model)
        except IOError:
            print(f'No model for fold {fold_no}')

    test_imgs, test_masks = read_imgs_with_masks(test_set)

    img_h, img_w = test_imgs.shape[:2]
    patch_h, patch_w = [96, 96]
    stride_h, stride_w = [96, 96]

    for img, mask in zip(test_imgs, test_masks):
        img = norm_img(img.astype(np.float32))
        mask_preds = np.zeros(img.shape[:2], dtype=np.float32)
        h = 0
        while h < img_h:
            w = 0
            while w < img_w:
                patch = img[h:h+patch_h, w:w+patch_w]
                patch.shape = [1, 96, 96, 3]
                probab = 0.0
                for model in models:
                    probab += model.model.predict(patch) / len(models)
                mask_preds[h:h + patch_h, w:w + patch_w] = probab

                w += stride_w

            h += stride_h

        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(img)
        axs[1].imshow(mask, cmap='gray')
        axs[2].imshow(mask_preds, cmap='gray')
        plt.show()


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
