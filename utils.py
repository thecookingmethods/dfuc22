import functools
import glob
import os

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


def get_foldwise_split(fold_no, number_of_folds, imgs_masks_pairs):
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
