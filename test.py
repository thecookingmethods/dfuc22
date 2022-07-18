#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import json

from kunet_dk.test import main as kunet_testing
from patches_cls_dk.test_classifier import main as patches_cls_testing
from patches_cls_dk.test_segm import main as patches_segm_testing
from patches_cls_dk.test_pipeline import main as patches_pipeline_testing
from yolo_det_unet_segm.eval_flow import main as yolo_eval_flow


def main(config_file_path):
    config = read_config(config_file_path)

    if config['experiment_type'] == 'kunet_dk':
        kunet_testing(config)
    elif config['experiment_type'] == 'patches_cls_dk':
        if config['patches_cls_dk']['phase'] == 'cls':
            patches_cls_testing(config)
        elif config['patches_cls_dk']['phase'] == 'segm':
            patches_segm_testing(config)
        elif config['patches_cls_dk']['phase'] == 'pipeline':
            yolo_eval_flow(config)

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
