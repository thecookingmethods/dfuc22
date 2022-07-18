#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import json

from kunet_dk.train import main as kunet_training
from patches_cls_dk.train_classifier import main as patches_cls_training
from patches_cls_dk.train_segm import main as patches_segm_training
from detection.train_detection import main as detection_training
from kunet2_dk.train import main as kunet2_training


def main(config_file_path):
    config = read_config(config_file_path)

    if config['experiment_type'] == 'kunet_dk':
        kunet_training(config)
    elif config['experiment_type'] == 'patches_cls_dk':
        if config['patches_cls_dk']['phase'] == 'cls':
            patches_cls_training(config)
        elif config['patches_cls_dk']['phase'] == 'segm':
            patches_segm_training(config)
    elif config['experiment_type'] == 'detection':
        detection_training(config)
    elif config['experiment_type'] == 'kunet2_dk':
        kunet2_training(config)

    exit(0)


def read_config(path):
    with open(path, 'r') as f:
        config = json.load(f)
        return config


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-c", "--config", type=str, required=True)
    args = argparser.parse_args()
    main(args.config)
