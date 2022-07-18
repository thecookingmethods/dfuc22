import functools
import glob
import os
import json

import imageio.v3 as iio
import numpy as np

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def norm_img(img):
    img /= 255.0
    return img


def load_files_paths(imgs_dir, masks_dir):
    img_file_pattern = "*.jpg"
    mask_file_patter = "*.png"

    imgs_files = list(sorted(glob.glob(os.path.join(imgs_dir, img_file_pattern))))
    masks_files = list(sorted(glob.glob(os.path.join(masks_dir, mask_file_patter))))

    assert len(imgs_files) == len(masks_files), "number of imgs does not match number of masks"

    imgs_masks_pairs = zip(imgs_files, masks_files)

    imgs_masks_pairs = [(img_path, mask_path)
                            if os.path.basename(img_path).split(".")[0] == os.path.basename(mask_path).split(".")[0]
                            else None
                        for img_path, mask_path in imgs_masks_pairs]\

    assert None not in imgs_masks_pairs, "None in paths list means that there is a difference in some img_path, mask_path pair (does not match)"

    return imgs_masks_pairs


def read_imgs_with_masks(imgs_masks_pairs):
    imgs = []
    masks = []

    for img_path, mask_path in imgs_masks_pairs:
        img = iio.imread(img_path)
        mask = iio.imread(mask_path)

        imgs.append(img)
        masks.append(mask)

    imgs = np.array(imgs, dtype=np.uint8)
    masks = np.array(masks, dtype=np.uint8)
    masks.shape = [masks.shape[0], masks.shape[1], masks.shape[2], 1]
    return imgs, masks


def get_foldwise_split(fold_no, number_of_folds, imgs_masks_pairs, save_debug_file=False):
    rest, test_set = train_test_split(imgs_masks_pairs, test_size=0.2, random_state=12345)
    assert fold_no < number_of_folds
    val_sets = []
    for i in range(number_of_folds):
        val_sets.append(rest[i::number_of_folds])

    current_val_set = val_sets[fold_no]
    del val_sets[fold_no]
    current_train_set = list(functools.reduce(lambda x, y: x + y, val_sets, []))

    # check if sets are disjunctive
    check_if_sets_are_disjunctive(current_train_set, current_val_set, test_set)

    if save_debug_file:
        debug_train_set_list = [(os.path.split(train_example[0])[-1], os.path.split(train_example[1])[-1]) for train_example in current_train_set]
        debug_val_set_list = [(os.path.split(val_example[0])[-1], os.path.split(val_example[1])[-1]) for val_example in current_val_set]
        debug_test_set_list = [(os.path.split(test_example[0])[-1], os.path.split(test_example[1])[-1]) for test_example in test_set]

        with open(f'fold_{fold_no}_out_of_{number_of_folds}_debug_files_list.json', 'w') as f:
            json.dump({'training': debug_train_set_list, 'validation': debug_val_set_list, 'test': debug_test_set_list}, f, indent=4)

    return current_train_set, current_val_set, test_set


def check_if_sets_are_disjunctive(current_train_set, current_val_set, test_set):
    for elem in current_val_set:
        if elem in current_train_set:
            raise AssertionError("Element from val set exists in training set.")
    for elem in test_set:
        if elem in current_train_set:
            raise AssertionError("Element from test set exists in training set.")
    for elem in current_val_set:
        if elem in test_set:
            raise AssertionError("Element from val set exists in test set.")


def plot_and_save_fig(metrics, legend, xlabel, ylabel, name):
    fig = plt.figure()
    for metric in metrics:
        plt.plot(metric)
    plt.legend(legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('name')
    plt.savefig(f'./{name}.png')
    plt.close(fig)


def calculate_dice(imgs, masks, pred_func, *pred_func_additional_args):
    dice_sum = 0.0
    examples_counter = 0
    for i, (img, mask) in enumerate(zip(imgs, masks)):
        print(f'{i + 1}/{len(imgs)}')
        preds_masks = pred_func(img, mask, *pred_func_additional_args)
        up = (2 * mask * preds_masks).sum()
        down = (mask + preds_masks).sum()
        dice = up / down
        dice_sum += dice
        examples_counter += 1.0
    dice_per_image = dice_sum / examples_counter
    return dice_per_image


def get_experiment_dir(experiment_artifacts_root_dir, experiment_name, experiment_type):
    return os.path.join(experiment_artifacts_root_dir, f'{experiment_name}-{experiment_type}_experiment')


def get_experiment_model_name(experiment_name, fold_no):
    return f'{experiment_name}_{fold_no}.h5'


def read_images(files_paths):
    imgs = {}

    for img_path in files_paths:
        img = iio.imread(img_path)

        code = os.path.split(img_path)[-1].split('.')[0]

        imgs[code] = img
    return imgs


def load_files(dataset_dir):
    img_file_pattern = "*.jpg"
    imgs_files_paths = list(sorted(glob.glob(os.path.join(dataset_dir, img_file_pattern))))
    return imgs_files_paths