import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import cv2

from kunet_dk.unetlike import Unetlike
from utils import load_files_paths, read_imgs_with_masks, get_foldwise_split, calculate_dice, norm_img, \
    get_experiment_dir, get_experiment_model_name


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
    imgs_masks_pairs = load_files_paths(imgs_dir, masks_dir)

    _, _, test_set = get_foldwise_split(0, folds_count, imgs_masks_pairs)

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

    test_imgs, test_masks = read_imgs_with_masks(test_set)

    dice = calculate_dice(test_imgs, test_masks, pred_func, (models, ))
    print(dice)


def pred_func(img, mask, *pred_func_additional_args):
    models = pred_func_additional_args[0][0]
    ulcer_mask = segm_ulcer(img, mask, models)
    skin_mask = segm_skin(img)

    preds = skin_mask * ulcer_mask

    #fig, axs = plt.subplots(2, 3)
    #axs[0, 0].imshow(img)
    #axs[0, 1].imshow(mask, cmap='gray')
    #axs[1, 0].imshow(ulcer_mask, cmap='gray')
    #axs[1, 1].imshow(skin_mask, cmap='gray')
    #axs[1, 2].imshow(preds, cmap='gray')
    #plt.show()

    return preds


def segm_skin(img):
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    skin_mask = cv2.inRange(hsv_img, lower, upper)

    #plt.figure()
    #plt.imshow(skin_mask, cmap='gray')
    #plt.show()

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

    #plt.figure()
    #plt.imshow(skin_mask, cmap='gray')
    #plt.show()

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)

    #plt.figure()
    #plt.imshow(skin_mask, cmap='gray')
    #plt.show()

    skin = cv2.bitwise_and(img, img, mask=skin_mask)

    #fig, axs = plt.subplots(1, 3)
    #axs[0].imshow(img)
    #axs[1].imshow(skin_mask)
    #axs[2].imshow(skin)
    #plt.show()

    skin_mask.shape = [skin_mask.shape[0], skin_mask.shape[1], 1]
    return skin_mask


def segm_ulcer(img, mask, models):
    img_out = np.zeros([*img.shape[:2], 6], dtype=np.float32)

    img_lab = color.rgb2lab(img)

    img = img.astype(np.float32)
    img = norm_img(img)

    img_lab[:, :, 0] = img_lab[:, :, 0] / 100.0
    img_lab[:, :, 1] = (img_lab[:, :, 1] + 127.0) / 255.0
    img_lab[:, :, 2] = (img_lab[:, :, 2] + 127.0) / 255.0

    img_out[:, :, :3] = img
    img_out[:, :, 3:] = img_lab

    top_right = img_out[:320, :480, :]
    top_left = img_out[:320, img.shape[1] - 480:, :]
    bottom_right = img_out[img.shape[0] - 320:, :480, :]
    bottom_left = img_out[img.shape[0] - 320:, img.shape[1] - 480:, :]

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
