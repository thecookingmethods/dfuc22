#!git clone https://github.com/facebookresearch/detr.git

import os
import random
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm

sys.path.append('./detr/')

from models.detr import SetCriterion
from models.matcher import HungarianMatcher


n_folds = 5
seed = 10
num_classes = 2
num_queries = 10
null_class_coef = 0.5
BATCH_SIZE = 8
LR = 2e-5
EPOCHS = 20
TEST_DIR = Path('C:/Users/user/Downloads/DFUC2022_val_release')


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(seed)


## Augmentations

def get_train_transforms():
    return A.Compose(
        [
            A.OneOf(
                [
                    A.HueSaturationValue(
                        hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.9
                    ),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
                ],
                p=0.9,
            ),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=480, width=640, p=1.0),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='coco', min_area=0, min_visibility=0, label_fields=['labels']
        ),
    )


def get_valid_transforms():
    return A.Compose(
        [A.Resize(height=480, width=640, p=1.0), ToTensorV2(p=1.0)],
        p=1.0,
        bbox_params=A.BboxParams(
            format='coco', min_area=0, min_visibility=0, label_fields=['labels']
        ),
    )


## Dataset

class MVGDataset(Dataset):
    def __init__(self, imgs_path, transforms=None):
        self.imgs = list(imgs_path.glob('*.jpg'))
        self.labels = list(imgs_path.glob('*.txt'))

        self.imgs.sort()
        self.labels.sort()

        self.transforms = transforms
        

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image_file = self.imgs[idx]
        image = cv2.imread(str(image_file)).astype(np.float32)
        image /= 255.0

        with open(self.labels[idx], 'r') as f:
            labels_text = f.readlines()
            labels_text = [label.strip().split(' ')[1:] for label in labels_text]
            bboxes = [[float(coord) for coord in label] for label in labels_text]

        # Ugly conversion to my input format...
        bbox = []
        for box in bboxes:
            new_box = [0, 0, 0, 0]
            new_box[0] = box[0] - (box[2] / 2)
            new_box[1] = box[1] - (box[3] / 2)
            new_box[0] *= 640
            new_box[1] *= 480
            new_box[2] = box[2] * 640
            new_box[3] = box[3] * 480
            if new_box[0] + new_box[2] > 640:
                new_box[2] = 640 - new_box[0]
            if new_box[1] + new_box[3] > 480:
                new_box[3] = 480 - new_box[1]
            new_box = [x if x > 0 else 0 for x in new_box]
            bbox.append(new_box)

        bbox = np.array(bbox)

        area = np.array(bbox[:, 2] * bbox[:, 3])
        area = np.expand_dims(area, axis=0)

        labels = np.zeros(bbox.shape[0], dtype=np.int32)

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': bbox,
                'labels': labels,
            }
            sample = self.transforms(**sample)
            image = sample['image']
            bbox = sample['bboxes']
            labels = sample['labels']

        _, h, w = image.shape
        bbox = A.augmentations.bbox_utils.normalize_bboxes(sample['bboxes'], rows=h, cols=w)
        target = {}
        target['boxes'] = torch.as_tensor(bbox, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.long)
        target['image_id'] = torch.tensor([idx])
        target['area'] = torch.as_tensor(area, dtype=torch.float32)

        return image, target, image_file.name

## Model


class DETR(nn.Module):
    def __init__(self, num_classes, num_queries):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        self.in_features = self.model.class_embed.in_features

        self.model.class_embed = nn.Linear(
            in_features=self.in_features, out_features=self.num_classes
        )
        self.model.num_queries = self.num_queries

    def forward(self, images):
        return self.model(images)


## Matcher

matcher = HungarianMatcher()
weight_dict = {'loss_ce': 1, 'loss_bbox': 1, 'loss_giou': 1}
losses = ['labels', 'boxes', 'cardinality']

## Training function


def train_fn(data_loader, model, criterion, optimizer, device, scheduler, epoch):
    model.train()
    criterion.train()

    summary_loss = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))

    for step, (images, targets, image_ids) in enumerate(tk0):
        images = list(image.to(device) for image in images)
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

        output = model(images)

        loss_dict = criterion(output, targets)
        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()

        losses.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        summary_loss.update(losses.item(), BATCH_SIZE)
        tk0.set_postfix(loss=summary_loss.avg)

    return summary_loss


def eval_fn(data_loader, model, criterion, device):
    model.eval()
    criterion.eval()
    summary_loss = AverageMeter()

    output_path = Path('results_DETR')
    if not output_path.exists():
        output_path.mkdir()

    with torch.no_grad():

        tk0 = tqdm(data_loader, total=len(data_loader))
        for step, (images, targets, image_ids) in enumerate(tk0):

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            output = model(images)

            loss_dict = criterion(output, targets)
            weight_dict = criterion.weight_dict

            losses = sum(
                loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
            )

            summary_loss.update(losses.item(), BATCH_SIZE)
            tk0.set_postfix(loss=summary_loss.avg)

            images = [image.permute(1, 2, 0).detach().cpu().numpy() for image in images]

            outputs = [{k: v.cpu() for k, v in output.items()}]

            for img_num, img in enumerate(images):
                # img = images[img_num]
                boxes = targets[img_num]['boxes'].cpu().numpy()
                boxes = [
                    np.array(box).astype(np.int32)
                    for box in A.augmentations.bbox_utils.denormalize_bboxes(boxes, 480, 640)
                ]
                for box in boxes:
                    cv2.rectangle(
                        img, (box[0], box[1]), (box[2] + box[0], box[3] + box[1]), (0, 220, 0), 2
                    )
                oboxes = outputs[0]['pred_boxes'][img_num].detach().cpu().numpy()
                oboxes = [
                    np.array(box).astype(np.int32)
                    for box in A.augmentations.bbox_utils.denormalize_bboxes(oboxes, 480, 640)
                ]
                prob = outputs[0]['pred_logits'][img_num].softmax(1).detach().cpu().numpy()[:, 0] # <- [:, 0], bo wyrzucam wszystkie tła

                for box, p in zip(oboxes, prob):
                    # Stąd można wyrzucić detekcje :)
                    # Filtrowanie od confidence > 0.97
                    # box - [xmin, ymin, w, h] denormalizowane
                    # p - confidence
                    if p > 0.97:
                        color = (0, 0, 220)
                        cv2.rectangle(
                            img, (box[0], box[1]), (box[2] + box[0], box[3] + box[1]), color, 2
                        )
                cv2.imwrite(
                    str(output_path / f'{image_ids[img_num].replace(".jpg", "_result.jpg")}'),
                    img * 255.0,
                )

    return summary_loss


## Engine


def collate_fn(batch):
    return tuple(zip(*batch))


def run():
    valid_dataset = MVGDataset(
        TEST_DIR, transforms=get_valid_transforms()
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    device = torch.device('cpu')
    model = DETR(num_classes=num_classes, num_queries=num_queries).to(device)
    model.load_state_dict(torch.load('models/detr_best_180.pth',map_location=torch.device('cpu')))
    criterion = SetCriterion(
        num_classes - 1, matcher, weight_dict, eos_coef=null_class_coef, losses=losses
    )
    criterion = criterion.to(device)

    valid_loss = eval_fn(valid_dataloader, model, criterion, device)

    print(f'Valid loss: {valid_loss.avg}')


if __name__ == '__main__':
    run()
