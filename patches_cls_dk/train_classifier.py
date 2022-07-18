import argparse
import os

from patches_cls_dk.resnet_cls import ResnetCls
from patches_cls_dk.cls_data_generator import ClsDataGenerator
from utils import load_files_paths, read_imgs_with_masks, get_foldwise_split, plot_and_save_fig, \
    get_experiment_model_name, get_experiment_dir


def main(config):
    fold_no = config['patches_cls_dk']['fold_no']
    folds_count = config['patches_cls_dk']['folds_count']
    imgs_dir = config['patches_cls_dk']['imgs_dir']
    masks_dir = config['patches_cls_dk']['masks_dir']
    batch_size = config['patches_cls_dk']['batch_size']
    epochs = config['patches_cls_dk']['epochs']
    experiment_name = config['patches_cls_dk']['experiment_name']
    phase = config['patches_cls_dk']['phase']
    patch_size = config['patches_cls_dk']['patch_size']
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

    train_gen = ClsDataGenerator(train_imgs, train_masks, batch_size, patch_size, 3, training=True)
    val_gen = ClsDataGenerator(val_imgs, val_masks, batch_size, patch_size, 3)

    net = ResnetCls([*patch_size, 3], get_experiment_model_name(experiment_name, fold_no), experiment_dir)
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


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('fold_no', type=int, help='fold number to train.')
    arg_parser.add_argument('folds_count', type=int, help='folds count in experiment')
    arg_parser.add_argument('imgs_dir', type=str, help='Directory with images.')
    arg_parser.add_argument('masks_dir', type=str, help='Directory with masks.')
    arg_parser.add_argument('batch_size', type=int, help='size of batch during training')
    arg_parser.add_argument('epochs', type=int, help='number of epochs')
    arg_parser.add_argument('--experiment_name', type=str, default='cls',
                            help='needed to define model name, it will be like experiment_name_fold_no.h5')
    args = arg_parser.parse_args()
    main(args.fold_no,
         args.folds_count,
         args.imgs_dir,
         args.masks_dir,
         args.batch_size,
         args.epochs,
         args.experiment_name)
