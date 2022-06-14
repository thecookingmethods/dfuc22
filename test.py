from utils import load_files_paths, read_imgs_with_masks, get_foldwise_split, norm_img
from unetlike import Unetlike
from main import IMGS_DIR, MASKS_DIR
import numpy as np
from matplotlib import pyplot as plt
from evaluator import Evaluator


def main():
    folds_count = 5
    experiment_name = 'segm'
    imgs_masks_pairs = load_files_paths(IMGS_DIR, MASKS_DIR)

    _, _, test_set = get_foldwise_split(0, 5, imgs_masks_pairs)

    models = []
    for fold_no in range(folds_count):
        model = Unetlike([320, 480, 3], f'{experiment_name}_{fold_no}')
        model.load(f'{experiment_name}_{fold_no}.h5')
        models.append(model)

    test_imgs, test_masks = read_imgs_with_masks(test_set)

    evaluator = Evaluator(models)
    metrics = evaluator.eval_set(test_imgs, test_masks)
    print(metrics)


if __name__ == "__main__":
    main()
