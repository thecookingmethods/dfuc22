import argparse
import os

import numpy as np
from skimage import color
import cv2

from kunet_dk.unetlike import Unetlike
from utils import load_files_paths, read_imgs_with_masks, get_foldwise_split, calculate_dice, norm_img, \
    get_experiment_dir, get_experiment_model_name
from skimage.measure import label, regionprops


def main(config):
    folds_count = config['kunet_dk']['folds_count']
    imgs_dir = config['kunet_dk']['imgs_dir']
    masks_dir = config['kunet_dk']['masks_dir']
    experiment_name = config['kunet_dk']['experiment_name']
    experiment_type = config['experiment_type']
    experiment_artifacts_dir = config['experiment_artifacts_root_dir']
    debug_imgs = config["kunet_dk"]["debug_imgs"]
    net_input_size = config["kunet_dk"]["net_input_size"]
    step_ratio = config["kunet_dk"]["step_ratio"]

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
        model = Unetlike([*net_input_size, 6], get_experiment_model_name(experiment_name, fold_no), '')
        try:
            model_file_dir = get_experiment_dir(experiment_artifacts_dir, experiment_name, experiment_type)
            model_file_name = get_experiment_model_name(experiment_name, fold_no)
            model.load(os.path.join(model_file_dir, model_file_name))
            models.append(model)
            print(f'loaded model for fold {fold_no}')
        except OSError:
            print(f'No model for fold {fold_no}.')

    test_imgs, test_masks = read_imgs_with_masks(test_set)

    dice_sum = 0.0
    examples_counter = 0
    for i, (img, mask) in enumerate(zip(test_imgs, test_masks)):
        print(f'{i + 1}/{len(test_imgs)}', end='. ')
        preds, ulcer_mask = pred_func(img, mask, models, net_input_size, step_ratio)

        mask = mask / 255.0

        preds = ulcer_mask

        preds = preds.astype(np.float32)
        preds[preds > 0.5] = 1.0
        preds[preds <= 0.5] = 0.0
        labeled_mask = label(preds[:, :, 0])
        regions = regionprops(labeled_mask)
        for region in regions:
            area = region.area
            height = region.bbox[2] - region.bbox[0]
            width = region.bbox[3] - region.bbox[1]
            if area < 91 or height < 9 or width < 8:
                preds[region.bbox[0]-1:region.bbox[2]+2, region.bbox[1]-1:region.bbox[3]+2, :] = 0.0

        if debug_imgs:
            from matplotlib import pyplot as plt
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].imshow(img)
            axs[0, 1].imshow(mask, cmap='gray')
            axs[1, 0].imshow(ulcer_mask, cmap='gray')
            axs[1, 1].imshow(skin_mask, cmap='gray')
            os.makedirs('debug_imgs', exist_ok=True)
            plt.savefig(f'./debug_imgs/{i}.png')

        up = (2 * mask * preds).sum()
        down = (mask + preds).sum()
        dice = up / down
        print(f'dice = {dice}')
        dice_sum += dice
        examples_counter += 1.0
    dice = dice_sum / examples_counter

    print(dice)


def pred_func(img, mask, models, net_input_size, step_ratio):
    ulcer_mask, probabs = segm_ulcer2(img, mask, models, net_input_size, step_ratio)

    preds = ulcer_mask

    #fig, axs = plt.subplots(2, 3)
    #axs[0, 0].imshow(img)
    #axs[0, 1].imshow(mask, cmap='gray')
    #axs[1, 0].imshow(ulcer_mask, cmap='gray')
    #axs[1, 1].imshow(skin_mask, cmap='gray')
    #axs[1, 2].imshow(preds, cmap='gray')
    #plt.show()

    return preds, ulcer_mask


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
        top_right_probabs += model._model.predict(top_right)[0, :, :, :] / len(models)
        top_left_probabs += model._model.predict(top_left)[0, :, :, :] / len(models)
        bottom_right_probabs += model._model.predict(bottom_right)[0, :, :, :] / len(models)
        bottom_left_probabs += model._model.predict(bottom_left)[0, :, :, :] / len(models)

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

    return preds, probabs


def segm_ulcer2(img, mask, models, net_input_size, step_ratio):
    preds_votes = []
    probabs_votes = []

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

    for model in models:
        probabs = np.zeros(mask.shape, dtype=np.float32)
        probabs_overlap_counter = np.zeros(mask.shape, dtype=np.float32)

        h = 0
        while h + size_h <= img.shape[0]:
            w = 0
            while w + size_w <= img.shape[1]:
                patch = img_out[h:h+size_h, w:w+size_w, :]
                patch.shape = [1, *patch.shape]
                result = model._model.predict(patch)[0, :, :, :]
                probabs[h:h+size_h, w:w+size_w] += result
                probabs_overlap_counter[h:h+size_h, w:w+size_w] += 1.0

                w += step_w
            h += step_h

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

    #return preds_votes, probabs_votes
    return np.round(probabs_votes), probabs_votes
