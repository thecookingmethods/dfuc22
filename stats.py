#!/usr/bin/python
# -*- coding: utf-8 -*-

from skimage.measure import regionprops, label
from utils import load_files_paths, read_imgs_with_masks
from collections import defaultdict
import numpy as np

def main():
    imgs_dir = "/home/darekk/dev/dfuc2022_challenge/DFUC2022_train_release/DFUC2022_train_images"
    masks_dir = "/home/darekk/dev/dfuc2022_challenge/DFUC2022_train_release/DFUC2022_train_masks"

    imgs_masks_pairs = load_files_paths(imgs_dir, masks_dir)
    imgs, masks = read_imgs_with_masks(imgs_masks_pairs)


    ile_na_obrazie = defaultdict(int)
    smallest = 99999
    areas = []
    for i, (img, mask) in enumerate(zip(imgs, masks)):
        print(f'{i}/')
        labeled_mask = label(mask[:, :, 0]/255.0)
        regions = regionprops(labeled_mask)
        ile_na_obrazie[len(regions)] += 1
        for region in regions:
            areas.append((region.area))
    print(ile_na_obrazie)
    print(f'50 smallest: {list(sorted(areas))[:50]}')
    hist, bin_edges = np.histogram(areas)
    print(f'hist: '
          f'{hist}, bin_edges: {bin_edges}')



if __name__ == "__main__":
    main()
