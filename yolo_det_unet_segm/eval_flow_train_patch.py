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


from patches_cls_dk.resnet_cls import ResnetCls
from patches_cls_dk.unetlike_segm import UnetlikeSegm
from utils import load_files_paths, read_imgs_with_masks, get_foldwise_split, norm_img, \
    get_experiment_dir, get_experiment_model_name
from skimage.measure import regionprops, label
from skimage.color import rgb2lab
import keras


def whole_segm(img,models,folds_count):
    img_out = np.zeros([*img.shape[:2], 6], dtype=np.float32)

    img_lab = rgb2lab(img)
    
    img = img.astype(np.float32)
    img = norm_img(img)
    
    img_lab[:, :, 0] = img_lab[:, :, 0] / 100.0
    img_lab[:, :, 1] = (img_lab[:, :, 1] + 127.0) / 255.0
    img_lab[:, :, 2] = (img_lab[:, :, 2] + 127.0) / 255.0
    
    img_out[:, :, :3] = img
    img_out[:, :, 3:] = img_lab
    
    img = img_out
    
    top_right = img[:320, :480, :]
    top_left = img[:320, img.shape[1] - 480:, :]
    bottom_right = img[img.shape[0] - 320:, :480, :]
    bottom_left = img[img.shape[0] - 320:, img.shape[1] - 480:, :]
    
    top_right.shape = [1, *top_right.shape]
    top_left.shape = [1, *top_left.shape]
    bottom_right.shape = [1, *bottom_right.shape]
    bottom_left.shape = [1, *bottom_left.shape]
    
    top_right_probabs = 0.0
    top_left_probabs = 0.0
    bottom_right_probabs = 0.0
    bottom_left_probabs = 0.0
    
    for model in models:
        top_right_probabs += model.predict(top_right)[0, :, :, :] / folds_count
        top_left_probabs += model.predict(top_left)[0, :, :, :] / folds_count
        bottom_right_probabs += model.predict(bottom_right)[0, :, :, :] / folds_count
        bottom_left_probabs += model.predict(bottom_left)[0, :, :, :] / folds_count
    
    probabs = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)
    probabs_overlap_counter = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)
    
    probabs[:320, :480] += top_right_probabs[:, :, :]
    probabs_overlap_counter[:320, :480] += 1.0
    
    probabs[:320, img.shape[1] - 480:] += top_left_probabs[:, :, :]
    probabs_overlap_counter[:320, img.shape[1] - 480:] += 1.0
    
    probabs[img.shape[0] - 320:, :480] += bottom_right_probabs[:, :, :]
    probabs_overlap_counter[img.shape[0] - 320:, :480] += 1.0
    
    probabs[img.shape[0] - 320:, img.shape[1] - 480:] += bottom_left_probabs[:, :, :]
    probabs_overlap_counter[img.shape[0] - 320:, img.shape[1] - 480:] += 1.0
    
    probabs /= probabs_overlap_counter

    return probabs


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

    net = cv2.dnn.readNet("C:/Users/user/Downloads/dfuc22-patches_classification/dfuc22-patches_classification/yolo_det_unet_segm/yolov4_dfuc_03_best.weights",
                          "C:/Users/user/Downloads/dfuc22-patches_classification/dfuc22-patches_classification/yolo_det_unet_segm/yolov4_dfuc.cfg")

    classes = []
    with open("C:/Users/user/Downloads/dfuc22-patches_classification/dfuc22-patches_classification/yolo_det_unet_segm/classes_dfuc.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    print(classes)
    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    models_segm1 = []
    experiment_segm_name1 = 'test_128'
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
    experiment_segm_name2 = 'testtt'
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
            
    models_segm3 = []
    print('Loading very big models')
    for fold_no in range(folds_count):
        #model_segm = UnetlikeSegm([patch_h, patch_w, 3], get_experiment_model_name(experiment_segm_name2, fold_no), '')
        try:
            model_file_dir = "C:/Users/user/Downloads/bigger_kunet_dk"
            model_file_name = f'bigger_net_{fold_no}.h5'
            modl = keras.models.load_model(os.path.join(model_file_dir, model_file_name),custom_objects=UnetlikeSegm.dependencies)

            print(f'Loaded very big model for fold {fold_no}')
            models_segm3.append(modl)
        except IOError:
            print(f'No segm model for fold {fold_no}')

    test_imgs, test_masks = read_imgs_with_masks(test_set)

    img_h, img_w = test_imgs.shape[:2]

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    four = 0
    three = 0
    two = 0
    one = 0
    
    small = 0
    medium = 0
    big = 0
        
    nothing_detected = 0
    
    box_reduced = 0
    
    dice_sum = 0.0

    
    for j, (img, mask) in enumerate(zip(test_imgs, test_masks)):

        print(f'calculating {j + 1}/{len(test_imgs)}')
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
        
        fig, ax = plt.subplots()
        ax.imshow(img, 'gray', interpolation='none')
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.01:
                    # onject detected
                    center_x = round(detection[0] * width)
                    center_y = round(detection[1] * height)
                    w = round(detection[2] * width)
                    h = round(detection[3] * height)
                    
                    # cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                    # rectangle co-ordinaters
                    x = round(center_x - w / 2)
                    y = round(center_y - h / 2)
                    # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                    boxes.append([x, y, w, h])  # put all rectangle areas
                    confidences.append(
                        float(confidence))  # how confidence was that object detected and show that percentage
                    class_ids.append(class_id)  # name of the object that was detected

        if len(boxes) == 0:
            print('nothing detected')
            nothing_detected += 1

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.01, 0.6)

        labeled_mask = label(np.ceil(mask[:, :, 0] / 255.0))
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
               
        if (len(boxes) != 0 and len(merged_yolo_boxes) == 0):
            print('all detections reduced by cv2.dnn.NMSBoxes')
            box_reduced+=1
            
        if len(merged_yolo_boxes) > 3:
            four += 1
        elif len(merged_yolo_boxes) > 2:
            three += 1
        elif len(merged_yolo_boxes) > 1:
            two += 1
        else:
            one += 1   
            
            
            

        for i in range(len(merged_yolo_boxes)):

            h, w, size_h, size_w = merged_yolo_boxes[i]

            yolo_center_h = (h + size_h) / 2
            yolo_center_w = (w + size_w) / 2
            
            max_hw = max(size_h-h,size_w-w)

            if max_hw<128:
                small += 1
                patch_h = 128
                patch_w = 128
            else:
                medium += 1
                patch_h = 256
                patch_w = 256
                        

            from_h = round(yolo_center_h - patch_h//2)
            to_h = round(yolo_center_h + patch_h//2)
            from_w = round(yolo_center_w - patch_w//2)
            to_w = round(yolo_center_w + patch_w//2)

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

            patch = img[from_h:to_h, from_w:to_w, :]
            patch.shape = [1, *patch.shape]
            result = np.zeros(patch.shape, dtype=np.float32)

            
            print(f'Size of yolo detection: {max_hw}')
            if max_hw < 128:
                print('small patch detected - testing on small model')
                
                for model in models_segm1:
                    result += model.model.predict(patch) / len(models_segm1)
                mask_probabs[from_h:to_h, from_w:to_w] += result[0, :, :, 0]
                mask_overlap[from_h:to_h, from_w:to_w] += 1
                
            else:
            #elif  max_hw < 256:
                print('medium patch detected - testing on medium model')
                
                for model in models_segm2:
                    result += model.model.predict(patch) / len(models_segm2)
                
                print(f'Result shape is {result.shape}')

                mask_probabs[from_h:to_h, from_w:to_w] += result[0, :, :, 0]
                mask_overlap[from_h:to_h, from_w:to_w] += 1   
               
            '''
            else:
                print('big patch detected - segmenting whole image')
                result_all = whole_segm(img,models_segm3,5)
                result = np.expand_dims(result_all, axis=0)
                print(f'Result shape is {result.shape}')
                                
                mask_probabs += result[0, :, :, 0]
                mask_overlap += 1
            '''

                

            pass

            
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
                
            
            rect1 = patches.Rectangle((w,h), size_w-w, size_h-h, linewidth=1, edgecolor='y', facecolor='none')
            ax.add_patch(rect1)
            
            rect2 = patches.Rectangle((from_w,from_h), to_w-from_w, to_h-from_h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect2)
            

        fn += regions_len
        

        mask_overlap[mask_overlap == 0] = 1
        mask_probabs /= mask_overlap

        thresh = 0.3
        #mask_preds = np.round(mask_probabs)
        mask_preds = (mask_probabs > thresh).astype(float)
        cv2.imwrite(f"C:/Users/user/Downloads/DFU_results/Masks/03b/{j}.png", mask_preds*255)

        


        mask = np.ceil(mask/255.0)


        up = (2 * mask[:,:,0] * mask_preds).sum()
        down = (mask[:,:,0] + mask_preds).sum()
        dice = up / down

        
        dice_sum += dice
        
              
        ddice = int(dice*100)
        

        mask3 = np.zeros_like(img)
        #mask3[:,:,0] = mask[:,:,0]
        mask3[:,:,1] = mask[:,:,0]
        #mask3[:,:,2] = mask[:,:,0]
        ax.imshow(mask3, 'jet', interpolation='none', alpha=0.4)
        
        
        mask_preds3 = np.zeros_like(img)
        #mask_preds3[:,:,0] = mask_preds
        #mask_preds3[:,:,1] = mask_preds
        mask_preds3[:,:,2] = mask_preds
        
        
        ax.imshow(mask_preds3, 'jet', interpolation='none', alpha=0.4)
        

        ax.text(20,30, 'Ground Truth Mask', bbox=dict(facecolor='green', alpha=0.4))
        
        ax.text(20,60, 'Our Sermentation', bbox=dict(facecolor='blue', alpha=0.4))
        
        ax.text(20,90, f'Dice = {ddice}%', bbox=dict(facecolor='blue', alpha=0.4))
        
        ax.text(20,120, 'Boxes from yolo', bbox=dict(facecolor='yellow', alpha=0.4))
        
        ax.text(20,150, 'Boxes into UNET', bbox=dict(facecolor='red', alpha=0.4))


        #fig.savefig(f"C:/Users/user/Downloads/DFU_results/03b_03_seg/{j}.png")
        
       

    se = tp / (tp + fn)
    pp = tp / (tp + fp)

    print(f'se = {se}')
    print(f'pp = {pp}')
    
    dice_per_image = dice_sum / len(test_imgs)
    dice_per_image_detected = dice_sum / (len(test_imgs)-nothing_detected-box_reduced)

    print(f'dice_per_image: {dice_per_image}')
    print(f'dice_per_image_detected: {dice_per_image_detected}')

    print(f'Small detections in test set: {small}')
    print(f'Medium detections in test set: {medium}')
    print(f'Big detections in test set: {big}')
    
    print(f'Nothing detected: {nothing_detected}')
    print(f'All reduced: {box_reduced}')
    
    print(f'Four: {four}')
    print(f'Three: {three}')
    print(f'Two: {two}')
    print(f'One: {one}')








