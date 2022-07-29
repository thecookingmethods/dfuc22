#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import json

from kunet_dk.test import main as kunet_testing

def main(config_file_path):
    config = read_config(config_file_path)
    if config['experiment_type'] == 'kunet_dk':
        kunet_testing(config)

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
