#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse

from utils import load_files_paths, get_foldwise_split


def main(fold_no, folds_count, imgs_dir, masks_dir):
    print(f'fold_no: {fold_no}')
    print(f'folds_count: {folds_count}')
    print(f'imgs_dir: {imgs_dir}')
    print(f'masks_dir: {masks_dir}')
    # loading paths from directories. Dirs given by programs args
    # should be absolute and look like /path/to/imgs/DFUC2022_train_images
    # and /path/to/masks/DFUC2022_train_masks
    imgs_masks_pairs = load_files_paths(imgs_dir, masks_dir)

    # split dataset fold-wise
    train_set, val_set, test_set = get_foldwise_split(fold_no, folds_count, imgs_masks_pairs, save_debug_file=True)

    # then train model on train_set, with validation on val_set (specific to given fold)
    # and test on test_set (same for every fold)
    # model.fit(train_set, validation=val_set)
    # evaluate(test_set)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('fold_no', type=int, help='fold number to train.')
    arg_parser.add_argument('folds_count', type=int, help='folds count in experiment')
    arg_parser.add_argument('imgs_dir', type=str, help='Directory with images.')
    arg_parser.add_argument('masks_dir', type=str, help='Directory with masks.')

    args = arg_parser.parse_args()
    main(args.fold_no,
         args.folds_count,
         args.imgs_dir,
         args.masks_dir)
