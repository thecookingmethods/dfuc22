# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 19:46:09 2022

@author: user
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-
#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import imageio as iio


#from patches_cls_dk.resnet_cls import ResnetCls
from patches_cls_dk.unetlike_segm import UnetlikeSegm
from utils import load_files_paths, read_imgs_with_masks, get_foldwise_split, norm_img, \
    get_experiment_dir, get_experiment_model_name, read_images, load_files
from skimage.measure import regionprops, label
import keras
from rgb2lab import rgb2lab


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


    models_segm1 = []
    experiment_segm_name1 = 'patches_128'
    print('Loading small models')
    for fold_no in range(folds_count):
        model_segm = UnetlikeSegm([patch_h, patch_w, 3], get_experiment_model_name(experiment_segm_name1, fold_no), '')
        try:
            model_file_dir = get_experiment_dir(experiment_artifacts_dir, experiment_segm_name1, experiment_type)
            model_file_name = get_experiment_model_name(experiment_segm_name1, fold_no)
            model_segm.load(os.path.join(model_file_dir, model_file_name))
            print(f'Loaded segm model for fold {fold_no}')
            models_segm1.append(model_segm)
        except IOError:
            print(f'No segm model for fold {fold_no}')
            
    models_segm2 = []
    experiment_segm_name2 = 'patches_256'
    print('Loading big models')
    for fold_no in range(folds_count):
        model_segm = UnetlikeSegm([patch_h, patch_w, 3], get_experiment_model_name(experiment_segm_name2, fold_no), '')
        try:
            model_file_dir = get_experiment_dir(experiment_artifacts_dir, experiment_segm_name2, experiment_type)
            model_file_name = get_experiment_model_name(experiment_segm_name2, fold_no)
            model_segm.load(os.path.join(model_file_dir, model_file_name))
            print(f'Loaded segm model for fold {fold_no}')
            models_segm2.append(model_segm)
        except IOError:
            print(f'No segm model for fold {fold_no}')
            



    #img_h, img_w = test_imgs.shape[:2]
       
    nothing_detected = 0
    
    box_reduced = 0
    
    j = 0
    

    files = os.listdir(imgs_dir)
    for img_path in files:
        
        j += 1
        img = iio.imread(os.path.join(imgs_dir,img_path))
    
        print(f'calculating {j}/{len(files)}')
        img_256 = np.array(img)
        img = norm_img(img.astype(np.float32))
        mask_probabs = np.zeros(img.shape[:2], dtype=np.float32)
        mask_overlap = np.zeros(img.shape[:2], dtype=np.uint8)

        height, width, channels = img_256.shape
        # detecting objects
        blob = cv2.dnn.blobFromImage(img_256, 0.00392, (640, 640), (0, 0, 0), True, crop=False)

        fig, ax = plt.subplots()
        ax.imshow(img, 'gray', interpolation='none')
        
        
        merged_detr_boxes = np.load(f"C:/Users/user/Downloads/upload_ulcer/upload_ulcer/boxes_test/{j}.npy")
        print(merged_detr_boxes)

        if len(merged_detr_boxes) == 0:
            print('nothing detected')
            nothing_detected += 1

        #indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.01, 0.6)

        for i in range(len(merged_detr_boxes)):

            from_w, from_h, w, h = merged_detr_boxes[i]

            to_h = from_h + h
            to_w = from_w + w
            
            max_hw = max(to_h-from_h,to_w-from_w)


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


            img6 = rgb2lab(img_256)
            patch = img6[from_h:to_h, from_w:to_w, :]

                    
            if max_hw < 128:
                print('small patch detected - testing on 128 x 128 model')
                new_size = (128,128)
            else:
                print('medium patch detected - testing on 256 x 256 model')
                new_size = (256,256)
                
            patch2 = cv2.resize(patch,new_size)
            patch2.shape = [1, *patch2.shape]
            result = np.zeros(patch2.shape, dtype=np.float32)
            

            
            if max_hw < 128:
                for model in models_segm1:
                    result += model.model.predict(patch2) / len(models_segm1)
            else:
                for model in models_segm2:
                    result += model.model.predict(patch2) / len(models_segm2)
                
            
            
            mask_probabs[from_h:to_h, from_w:to_w] += cv2.resize(result[0, :, :, 0],(to_w-from_w,to_h-from_h))
            mask_overlap[from_h:to_h, from_w:to_w] += 1   
            pass

            rect1 = patches.Rectangle((from_w,from_h), to_w-from_w, to_h-from_h, linewidth=1, edgecolor='y', facecolor='none')
            ax.add_patch(rect1)
            

        mask_overlap[mask_overlap == 0] = 1
        mask_probabs /= mask_overlap

        thresh = 0.1
        #mask_preds = np.round(mask_probabs)
        mask_preds = (mask_probabs > thresh).astype(float)
        cv2.imwrite(f"C:/Users/user/Downloads/DFU_results/Masks/Test_DETR/{img_path[:6]}.png", mask_preds*255)
        
        
        mask_preds3 = np.zeros_like(img)
        #mask_preds3[:,:,0] = mask_preds
        #mask_preds3[:,:,1] = mask_preds
        mask_preds3[:,:,2] = mask_preds
        
        
        ax.imshow(mask_preds3, 'jet', interpolation='none', alpha=0.4)
        ax.text(20,30, 'Segmentation', bbox=dict(facecolor='blue', alpha=0.4))
        ax.text(20,60, 'Boxes from DETR', bbox=dict(facecolor='yellow', alpha=0.4))
        
        fig.savefig(f"C:/Users/user/Downloads/DFU_results/Visualisation/Val_DETR/{img_path[:6]}.png")
        
        
     

    print(f'Nothing detected: {nothing_detected}')
    print(f'All reduced: {box_reduced}')







