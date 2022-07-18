import argparse
import os

from kunet2_dk.unetlike import Unetlike
from kunet2_dk.data_generator import DataGenerator
from utils import load_files_paths, read_imgs_with_masks, get_foldwise_split, plot_and_save_fig, \
    get_experiment_model_name, get_experiment_dir


def main(config):
    fold_no = config['kunet2_dk']['fold_no']
    folds_count = config['kunet2_dk']['folds_count']
    imgs_dir = config['kunet2_dk']['imgs_dir']
    masks_dir = config['kunet2_dk']['masks_dir']
    batch_size = config['kunet2_dk']['batch_size']
    epochs = config['kunet2_dk']['epochs']
    experiment_name = config['kunet2_dk']['experiment_name']
    experiment_type = config['experiment_type']
    experiment_artifacts_root_dir = config['experiment_artifacts_root_dir']
    continue_training_path = config['kunet2_dk']['continue_training_path']

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

    imgs_masks_pairs = load_files_paths(imgs_dir, masks_dir)

    train_set, val_set, test_set = get_foldwise_split(fold_no, folds_count, imgs_masks_pairs, save_debug_file=True)

    train_imgs, train_masks = read_imgs_with_masks(train_set)
    val_imgs, val_masks = read_imgs_with_masks(val_set)

    train_gen = DataGenerator(train_imgs, train_masks, batch_size, [320, 480], training=True)
    val_gen = DataGenerator(val_imgs, val_masks, batch_size, [320, 480])

    net = Unetlike([320, 480, 6], get_experiment_model_name(experiment_name, fold_no), experiment_dir)
    if continue_training_path is not None:
        net.load(continue_training_path)

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

    plot_and_save_fig([history.history['rdr_metric'], history.history['val_rdr_metric']],
                      ['training', 'validation'],
                      'epoch', 'rdr_metric',
                      os.path.join(experiment_dir, f'fold_{fold_no}_rdr_metric_{experiment_name}'))

    plot_and_save_fig([history.history['siou_metric'], history.history['val_siou_metric']],
                      ['training', 'validation'],
                      'epoch', 'siou_metric',
                      os.path.join(experiment_dir, f'fold_{fold_no}_siou_metric_{experiment_name}'))

    plot_and_save_fig([history.history['r_r_metric'], history.history['val_r_r_metric']],
                      ['training', 'validation'],
                      'epoch', 'r_r_metric',
                      os.path.join(experiment_dir, f'fold_{fold_no}_r_r_metric_{experiment_name}'))

    plot_and_save_fig([history.history['dist_metric'], history.history['val_dist_metric']],
                      ['training', 'validation'],
                      'epoch', 'dist_metric',
                      os.path.join(experiment_dir, f'fold_{fold_no}_dist_metric_{experiment_name}'))

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
