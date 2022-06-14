#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from utils import norm_img


class Evaluator:
    def __init__(self, models):
        self._models = models
        self._folds_count = len(models)

    def eval_set(self, test_imgs, test_masks):
        dice_sum = 0.0
        examples_counter = 0
        for img, mask in zip(test_imgs, test_masks):
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

            for model in self._models:
                top_right_probabs += model.model.predict(top_right)[0, :, :, :] / self._folds_count
                top_left_probabs += model.model.predict(top_left)[0, :, :, :] / self._folds_count
                bottom_right_probabs += model.model.predict(bottom_right)[0, :, :, :] / self._folds_count
                bottom_left_probabs += model.model.predict(bottom_left)[0, :, :, :] / self._folds_count

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

            mask = mask.astype(np.float32)
            mask /= 255.0
            up = (2 * mask * preds).sum()
            down = (mask + preds).sum()
            if down == 0.0:
                print('mianownik zero, brak zmiany na masce?')
            dice = up / down
            dice_sum += dice
            examples_counter += 1.0

        dice_per_image = dice_sum / examples_counter

        return dice_per_image