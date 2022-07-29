#!/usr/bin/python
# -*- coding: utf-8 -*-
from kunet_dk.gen_masks import main as gen_masks_kunet

import json
import argparse


def main(config):
    if config['experiment_type'] == 'kunet_dk':
        gen_masks_kunet(config)

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
