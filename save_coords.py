#!/usr/bin/python
# -*- coding: utf-8 -*-
import os

import numpy as np
from utils import get_foldwise_split, read_imgs_with_masks, load_files_paths
from skimage.measure import regionprops, label
from skimage.io import imsave


def main():
    imgs_dir = "/home/darekk/dev/dfuc2022_challenge/DFUC2022_train_release/DFUC2022_train_images"
    masks_dir = "/home/darekk/dev/dfuc2022_challenge/DFUC2022_train_release/DFUC2022_train_masks"

    result_dir = './yolo_db_test'

    imgs_masks_pairs = load_files_paths(imgs_dir, masks_dir)
    train_set, val_set, test_set = get_foldwise_split(0, 5, imgs_masks_pairs, save_debug_file=True)

    #train_imgs, train_masks = read_imgs_with_masks(train_set)
    #val_imgs, val_masks = read_imgs_with_masks(val_set)
    test_imgs, test_masks = read_imgs_with_masks(test_set)
    imgs, masks = test_imgs, test_masks

    #imgs = np.concatenate([train_imgs, val_imgs], axis=0)
    #masks = np.concatenate([train_masks, val_masks], axis=0)

    os.makedirs(result_dir, exist_ok=True)

    for i, (img, mask) in enumerate(zip(imgs, masks)):
        labeled_mask = label(mask[:, :, 0] / 255.0)
        regions = regionprops(labeled_mask)
        label_file_content = []
        for j, region in enumerate(regions):

            if len(regions) > 1:
                print('elo')

            center_y, center_x = region.centroid[0], region.centroid[1]
            y, x = region.bbox[2] - region.bbox[0], region.bbox[3] - region.bbox[1]

            center_y /= img.shape[0]
            center_x /= img.shape[1]

            y /= img.shape[0]
            x /= img.shape[1]


            label_content = f'0 {center_x} {center_y} {x} {y}'
            label_file_content.append(label_content)

        img_path = os.path.join(result_dir, f'{i}.jpg')
        label_path = os.path.join(result_dir, f'{i}.txt')

        with open(label_path, 'w') as f:
            f.write('\n'.join(label_file_content))
        imsave(img_path, img)





if __name__ == "__main__":
    main()
