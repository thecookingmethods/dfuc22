#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
from utils import load_files_paths, get_foldwise_split, read_imgs_with_masks, calculate_dice


def main(folds_count, imgs_dir, masks_dir):
    print(f'folds_count: {folds_count}')
    print(f'imgs_dir: {imgs_dir}')
    print(f'masks_dir: {masks_dir}')

    # load paths
    imgs_masks_pairs = load_files_paths(imgs_dir, masks_dir)

    # fold number does not matter here, test set is the same always for specified folds count
    _, _, test_set = get_foldwise_split(fold_no=0, number_of_folds=folds_count, imgs_masks_pairs=imgs_masks_pairs)

    # read each model for each fold
    models = []
    for fold_no in range(folds_count):
        model = None  # load model for each fold
        models.append(model)

    # read images and masks
    test_imgs, test_masks = read_imgs_with_masks(test_set)

    # calculate dice
    dice = calculate_dice(test_imgs, test_masks, pred_func, (models,))
    print(dice)


# This func is used by calculate_dice, needs to be implemented (the way how prediction mask is calculated)
def pred_func(img, mask, *pred_func_additional_args):
    raise NotImplementedError()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('folds_count', type=int, help='folds count in experiment')
    arg_parser.add_argument('imgs_dir', type=str, help='Directory with images.')
    arg_parser.add_argument('masks_dir', type=str, help='Directory with masks.')
    args = arg_parser.parse_args()
    main(args.folds_count,
         args.imgs_dir,
         args.masks_dir,)
