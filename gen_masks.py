#!/usr/bin/python
# -*- coding: utf-8 -*-
from kunet_dk.gen_masks import main as gen_masks_kunet
from yolo_det_unet_segm.gen_mask_yolo import main as gen_mask_yolo
import json
import argparse


def main(config):
    #gen_masks_kunet(config)
    gen_mask_yolo(config)


def read_config(path):
    with open(path, 'r') as f:
        config = json.load(f)
        return config


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-c", "--config", type=str, required=True)
    args = argparser.parse_args()

    config = read_config(args.config)

    main(config)
