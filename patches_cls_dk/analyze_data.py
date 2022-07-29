#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
import imageio.v3 as iio
from skimage.measure import regionprops, label
import numpy as np
from utils import load_files_paths, read_imgs_with_masks
from collections import defaultdict
import json


def main(imgs_dir, masks_dir):
    print(f'imgs_dir: {imgs_dir}')
    print(f'masks_dir: {masks_dir}')

    imgs_masks_pairs = load_files_paths(imgs_dir, masks_dir)

    metadata = defaultdict(list)
    for img_path, mask_path in imgs_masks_pairs:
        code = os.path.split(img_path)[-1].split(".")[0]
        mask = iio.imread(mask_path).astype(np.float32) / 255.0

        labeled_mask = label(mask)
        regions = regionprops(labeled_mask)

        for region in regions:
            center_hw = [int(region.centroid[0]), int(region.centroid[1])]
            from_hw = [int(region.bbox[0]), int(region.bbox[1])]
            to_hw = [int(region.bbox[2]), int(region.bbox[3])]
            area = int(region.area)

            metadata[code].append({'center_hw': center_hw, 'from_hw': from_hw, 'to_hw': to_hw, 'area': area})

    with open('metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('imgs_dir', type=str, help='Directory with images.')
    arg_parser.add_argument('masks_dir', type=str, help='Directory with masks.')

    args = arg_parser.parse_args()
    main(args.imgs_dir,
         args.masks_dir)
