#!/usr/bin/python
# -*- coding: utf-8 -*-
#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from patches_cls_dk.resnet_cls import ResnetCls
from patches_cls_dk.unetlike_segm import UnetlikeSegm
from utils import load_files_paths, read_imgs_with_masks, get_foldwise_split, norm_img, \
    get_experiment_dir, get_experiment_model_name
from skimage.measure import regionprops, label


def main(config):
    folds_count = config['patches_cls_dk']['folds_count']
    imgs_dir = config['patches_cls_dk']['imgs_dir']
    masks_dir = config['patches_cls_dk']['masks_dir']
    patch_size = config['patches_cls_dk']['patch_size']
    stride = config['patches_cls_dk']['stride']
    experiment_cls_name = config['patches_cls_dk']['experiment_cls_name']
    experiment_segm_name = config['patches_cls_dk']['experiment_segm_name']
    experiment_type = config['experiment_type']
    experiment_artifacts_dir = config['experiment_artifacts_root_dir']

    print(f'folds_count: {folds_count}')
    print(f'imgs_dir: {imgs_dir}')
    print(f'masks_dir: {masks_dir}')
    print(f'experiment_cls_name: {experiment_cls_name}')
    print(f'experiment_segm_name: {experiment_segm_name}')
    print(f'experiment_type: {experiment_type}')
    print(f'experiment_artifacts_dir: {experiment_artifacts_dir}')

    patch_h, patch_w = patch_size
    stride_h, stride_w = stride

    imgs_masks_pairs = load_files_paths(imgs_dir, masks_dir)

    _, _, test_set = get_foldwise_split(0, folds_count, imgs_masks_pairs)

    net = cv2.dnn.readNet("/home/darekk/dev/dfuc2022_challenge/dfuc22_2/yolo_det_unet_segm/yolov4_dfuc_best.weights",
                          "/home/darekk/dev/dfuc2022_challenge/dfuc22_2/yolo_det_unet_segm/yolov4_dfuc.cfg")

    classes = []
    with open("/home/darekk/dev/dfuc2022_challenge/dfuc22_2/yolo_det_unet_segm/classes_dfuc.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    print(classes)
    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    models_segm = []
    for fold_no in range(folds_count):
        model_segm = UnetlikeSegm([patch_h, patch_w, 3], get_experiment_model_name(experiment_segm_name, fold_no), '')
        try:
            model_file_dir = get_experiment_dir(experiment_artifacts_dir, experiment_segm_name, experiment_type)
            model_file_name = get_experiment_model_name(experiment_segm_name, fold_no)
            model_segm.load(os.path.join(model_file_dir, model_file_name))
            print(f'Loaded segm model for fold {fold_no}')
            models_segm.append(model_segm)
        except IOError:
            print(f'No segm model for fold {fold_no}')

    test_imgs, test_masks = read_imgs_with_masks(test_set)

    img_h, img_w = test_imgs.shape[:2]

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    dice_sum = 0.0
    for i, (img, mask) in enumerate(zip(test_imgs, test_masks)):

        print(f'calculating {i + 1}/{len(test_imgs)}')
        img_256 = np.array(img)
        img = norm_img(img.astype(np.float32))
        mask_probabs = np.zeros(img.shape[:2], dtype=np.float32)
        mask_overlap = np.zeros(img.shape[:2], dtype=np.uint8)

        height, width, channels = img_256.shape
        # detecting objects
        blob = cv2.dnn.blobFromImage(img_256, 0.00392, (640, 640), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(outputlayers)
        # print(outs[1])

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.01:
                    # onject detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                    # rectangle co-ordinaters
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                    boxes.append([x, y, w, h])  # put all rectangle areas
                    confidences.append(
                        float(confidence))  # how confidence was that object detected and show that percentage
                    class_ids.append(class_id)  # name of the object that was detected

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

        labeled_mask = label(mask[:, :, 0] / 255.0)
        regions = regionprops(labeled_mask)


        boxes_mask = np.zeros(mask.shape, dtype=np.float32)
        for i in range(len(boxes)):
            if i in indexes:
                w, h, size_w, size_h = boxes[i]
                boxes_mask[h:h+size_h, w:w+size_w] = 1.0

        labeled_boxes_mask = label(boxes_mask[:, :, 0])
        boxes_regions = regionprops(labeled_boxes_mask)
        merged_yolo_boxes = [region.bbox for region in boxes_regions]


        regions_len = len(regions)

        patchu_h = 64
        patchu_w = 64
        for i in range(len(merged_yolo_boxes)):

            h, w, size_h, size_w = merged_yolo_boxes[i]

            yolo_center_h = (h + size_h) / 2
            yolo_center_w = (w + size_w) / 2

            from_h = int(yolo_center_h - patchu_h//2)
            to_h = int(yolo_center_h + patchu_h//2)
            from_w = int(yolo_center_w - patchu_w//2)
            to_w = int(yolo_center_w + patchu_w//2)

            if from_h < 0:
                from_h = 0
                to_h = patch_h
            elif to_h > img.shape[0]:
                to_h = img.shape[0]
                from_h = img.shape[0] - patch_h

            if from_w < 0:
                from_w = 0
                to_w = patch_w
            elif to_w > img.shape[1]:
                to_w = img.shape[1]
                from_w = img.shape[1] - patch_w

            yolo_region = np.zeros(mask.shape, dtype=np.float32)
            yolo_region[from_h:to_h, from_w:to_w] = 1.0
            found = 0

            for region in regions:
                mask_region = np.zeros(mask.shape, dtype=np.float32)
                mask_region[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]] = 1.0

                if (mask_region * yolo_region).sum() > 0.0:
                    found = 1
                    regions_len -= 1
                    break

            if found == 1:
                tp += 1
            else:
                fp += 1

        fn += regions_len

        '''
        patch = img[from_h:to_h, from_w:to_w, :]
        patch.shape = [1, *patch.shape]
        result = np.zeros(patch.shape, dtype=np.float32)
        for model in models_segm:
            result += model.model.predict(patch) / len(models_segm)
        mask_probabs[from_h:to_h, from_w:to_w] += result[0, :, :, 0]
        mask_overlap[from_h:to_h, from_w:to_w] += 1
        pass
        '''


        mask_overlap[mask_overlap == 0] = 1
        mask_probabs /= mask_overlap

        mask_preds = np.round(mask_probabs)

        up = (2 * mask[:,:,0] * mask_preds).sum()
        down = (mask[:,:,0] + mask_preds).sum()
        dice = up / down
        dice_sum += dice

    se = tp / (tp + fn)
    pp = tp / (tp + fp)

    print(f'se = {se}')
    print(f'pp = {pp}')

    dice_per_image = dice_sum / len(test_imgs)

    print(f'dice_per_image: {dice_per_image}')


