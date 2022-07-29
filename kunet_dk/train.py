import argparse
import os

import numpy as np

from kunet_dk.unetlike import Unetlike
from kunet_dk.data_generator import DataGenerator
from kunet_dk.hard_example_miner import HardExampleMiner

from utils import load_files_paths, read_imgs_with_masks, get_foldwise_split, plot_and_save_fig, \
    get_experiment_model_name, get_experiment_dir


def main(config):
    fold_no = config['kunet_dk']['fold_no']
    folds_count = config['kunet_dk']['folds_count']
    imgs_dir = config['kunet_dk']['imgs_dir']
    masks_dir = config['kunet_dk']['masks_dir']
    batch_size = config['kunet_dk']['batch_size']
    epochs = config['kunet_dk']['epochs']
    experiment_name = config['kunet_dk']['experiment_name']
    experiment_type = config['experiment_type']
    experiment_artifacts_root_dir = config['experiment_artifacts_root_dir']
    net_input_size = config['kunet_dk']['net_input_size']
    base_model_experiment_name = config['kunet_dk']['base_model_experiment_name']
    ohem_ratio = config['kunet_dk']['ohem_ratio']
    use_ohem = config['kunet_dk']['use_ohem']
    use_new_data_gen = config['kunet_dk']['use_new_data_gen']

    experiment_dir = get_experiment_dir(experiment_artifacts_root_dir, experiment_name, experiment_type)
    os.makedirs(experiment_dir, exist_ok=True)

    print(f'fold_no: {fold_no}')
    print(f'folds_count: {folds_count}')
    print(f'imgs_dir: {imgs_dir}')
    print(f'masks_dir: {masks_dir}')
    print(f'batch_size: {batch_size}')
    print(f'epochs: {epochs}')
    print(f'experiment_name: {experiment_name}')
    print(f'experiment_type: {experiment_type}')
    print(f'experiment_artifacts_dir: {experiment_artifacts_root_dir}')
    print(f'net_input_size: {net_input_size}')
    print(f'base_model_experiment_name: {base_model_experiment_name}')
    print(f'ohem_ratio: {ohem_ratio}')

    imgs_masks_pairs = load_files_paths(imgs_dir, masks_dir)

    train_set, val_set, test_set = get_foldwise_split(fold_no, folds_count, imgs_masks_pairs, save_debug_file=True)

    train_imgs, train_masks = read_imgs_with_masks(train_set)
    val_imgs, val_masks = read_imgs_with_masks(val_set)
    test_imgs, test_masks = read_imgs_with_masks(test_set)

    test_split = int(test_imgs.shape[0]*0.8)
    train_imgs = np.concatenate([train_imgs, test_imgs[:test_split]], axis=0)
    train_masks = np.concatenate([train_masks, test_masks[:test_split]], axis=0)

    val_imgs = np.concatenate([val_imgs, test_imgs[test_split:]], axis=0)
    val_masks = np.concatenate([val_masks, test_masks[test_split:]], axis=0)


    train_gen = DataGenerator(train_imgs, train_masks, batch_size, net_input_size, training=True)
    val_gen = DataGenerator(val_imgs, val_masks, batch_size, net_input_size)


    net = Unetlike([*net_input_size, 6], get_experiment_model_name(experiment_name, fold_no), experiment_dir)
    if base_model_experiment_name is not None:
        model_file_dir = get_experiment_dir(experiment_artifacts_root_dir, base_model_experiment_name, experiment_type)
        model_file_name = get_experiment_model_name(base_model_experiment_name, fold_no)
        model_path = os.path.join(model_file_dir, model_file_name)
        net.load(model_path)
        print(f'loaded ohem model for fold {fold_no} from {model_path}')

    if use_ohem:
        print('using ohem')
        hem_training_gen = HardExampleMiner(net, [*net_input_size, 6], batch_size, train_gen, ratio=ohem_ratio)

        history = net.fit(hem_training_gen, val_gen,
                          epochs=epochs,
                          initial_epoch=0,
                          max_queue_size=500,
                          workers=7,
                          use_multiprocessing=False,
                          training_verbosity=1)
    else:
        print('without ohem')
        history = net.fit(train_gen, val_gen,
                          epochs=epochs,
                          initial_epoch=0,
                          max_queue_size=500,
                          workers=7,
                          use_multiprocessing=False,
                          training_verbosity=1)

    plot_and_save_fig([history.history['loss'], history.history['val_loss']],
                      ['training', 'validation'],
                      'epoch', 'loss',
                      os.path.join(experiment_dir, f'fold_{fold_no}_loss_{experiment_name}'))

    plot_and_save_fig([history.history['accuracy'], history.history['val_accuracy']],
                      ['training', 'validation'],
                      'epoch', 'accuracy',
                      os.path.join(experiment_dir, f'fold_{fold_no}_accuracy_{experiment_name}'))

    #test_imgs, test_masks = read_imgs_with_masks(test_set)

    #print('testing on test set')
    #metrics = evaluator.eval_set(test_imgs, test_masks)
    #print(f'avg dice per image on test set = {metrics}')

    #print('testing on val set')
    #metrics = evaluator.eval_set(val_imgs, val_masks)
    #print(f'avg dice per image on val set = {metrics}')


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('fold_no', type=int, help='fold number to train.')
    arg_parser.add_argument('folds_count', type=int, help='folds count in experiment')
    arg_parser.add_argument('imgs_dir', type=str, help='Directory with images.')
    arg_parser.add_argument('masks_dir', type=str, help='Directory with masks.')
    arg_parser.add_argument('batch_size', type=int, help='size of batch during training')
    arg_parser.add_argument('epochs', type=int, help='number of epochs')
    arg_parser.add_argument('--experiment_name', type=str, default='segm',
                            help='needed to define model name, it will be like experiment_name_fold_no.h5')
    args = arg_parser.parse_args()
    main(args.fold_no,
         args.folds_count,
         args.imgs_dir,
         args.masks_dir,
         args.batch_size,
         args.epochs,
         args.experiment_name)
