import argparse
import os

from detection.unetlike_det import UnetlikeDet
from detection.det_data_generator import DetDataGenerator
from utils import load_files_paths, read_imgs_with_masks, get_foldwise_split, plot_and_save_fig, \
    get_experiment_model_name, get_experiment_dir


def main(config):
    fold_no = config['detection']['fold_no']
    folds_count = config['detection']['folds_count']
    imgs_dir = config['detection']['imgs_dir']
    masks_dir = config['detection']['masks_dir']
    batch_size = config['detection']['batch_size']
    epochs = config['detection']['epochs']
    experiment_name = config['detection']['experiment_name']
    phase = config['detection']['phase']
    patch_size = config['detection']['patch_size']
    experiment_type = config['experiment_type']
    experiment_artifacts_root_dir = config['experiment_artifacts_root_dir']

    experiment_dir = get_experiment_dir(experiment_artifacts_root_dir, experiment_name, experiment_type)
    os.makedirs(experiment_dir, exist_ok=True)

    print(f'fold_no: {fold_no}')
    print(f'folds_count: {folds_count}')
    print(f'imgs_dir: {imgs_dir}')
    print(f'masks_dir: {masks_dir}')
    print(f'batch_size: {batch_size}')
    print(f'epochs: {epochs}')
    print(f'experiment_name: {experiment_name}')
    print(f'phase: {phase}')
    print(f'patch_size: {patch_size}')

    imgs_masks_pairs = load_files_paths(imgs_dir, masks_dir)

    train_set, val_set, test_set = get_foldwise_split(fold_no, folds_count, imgs_masks_pairs, save_debug_file=True)

    train_imgs, train_masks = read_imgs_with_masks(train_set)

    val_imgs, val_masks = read_imgs_with_masks(val_set)

    train_gen = DetDataGenerator(train_imgs, train_masks, batch_size, patch_size, 3, training=True)
    val_gen = DetDataGenerator(val_imgs, val_masks, batch_size, patch_size, 3)

    net = UnetlikeDet([*patch_size, 3], get_experiment_model_name(experiment_name, fold_no), experiment_dir)
    history = net.fit(train_gen, val_gen,
                      epochs=epochs,
                      initial_epoch=0,
                      max_queue_size=500,
                      workers=1,
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
