#!/usr/bin/env python
# coding: utf-8
from collections import Counter
import datetime
import json
import numpy as np 
import os
import pandas as pd
from PIL import Image
import random
import shutil
import time
from tqdm import tqdm_notebook as tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.data import Dataset

import torchvision
from torchvision import transforms as T
from torchvision.transforms import functional as F
import torchvision.transforms.functional as FT
from torchvision.models.detection.backbone_utils import *

import xml.etree.ElementTree as ET

def save_checkpoint(epoch, epochs_since_improvement, model_code, model, optimizer, lr,\
                    loss, best_loss, is_best, checkpoint_filename):
    """
    Save model checkpoint.
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement
    :param model: model
    :param optimizer: optimizer
    :param loss: validation loss in this epoch
    :param best_loss: best validation loss achieved so far (not necessarily in this checkpoint)
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'best_loss': best_loss,
             'model_code': model_code,
             'model': model,
             'optimizer': optimizer,
             'lr':lr
             }
    #filename = 'checkpoint_Faster-RCNN_base_model.pth.tar'
    filename = checkpoint_filename
    a = datetime.datetime.now()
    timedate = str(a.year)+str(a.month)+str(a.day)+'_'+str(a.hour)+str(a.minute)
    torch.save(state, str(epoch)+'_'+timedate+'_'+filename)
    # If this checkpoint is the best so far, store a copy so 
    # it doesn't get overwritten by a worse checkpoint
    if is_best:
        if os.path.exists('BEST_' + filename):
            checkpoint = torch.load('BEST_' + filename, map_location = 'cpu')
            if best_loss < checkpoint['best_loss']:
                torch.save(state, 'BEST_' + filename)
        else:
            torch.save(state, 'BEST_' + filename)

# #================================================================================================================================
global norm_layer
norm_layer = torch.nn.BatchNorm2d
#================================================================================================================================
#================================================================================================================================
#================================================================================================================================
def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)

def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, on_dev, data_folder):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    with open(os.path.join(data_folder,'label_map.json'), 'r') as j:
            label_map = json.load(j)
    
    rev_label_map = {v: k for k, v in label_map.items()} 
    if on_dev:
        device = det_boxes[0].device # torch.device('cuda:1')
        
    # these are all lists of tensors of the same length, i.e. number of images
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(true_labels) == len(true_difficulties)  
    n_classes = len(label_map)
    
    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    # ==============================================================================================================================
    if on_dev:
        true_images = torch.LongTensor(true_images).to(device)#-(n_objects), n_objects is the total no. of objects across all images
    else:
        true_images = torch.LongTensor(true_images).cuda()#-----(n_objects), n_objects is the total no. of objects across all images
    
    true_boxes = torch.cat(true_boxes, dim=0)#------------------(n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)#----------------(n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)#----(n_objects)
    # ==============================================================================================================================
    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    
    if on_dev:
        det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    else:
        det_images = torch.LongTensor(det_images).cuda()  # (n_detections)
    
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        if on_dev:
            true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(device)#(n_class_objects)
        else:
            true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).cuda()  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        if on_dev:
            true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
            false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        else:
            true_positives = torch.zeros((n_class_detections), dtype=torch.float).cuda()  # (n_class_detections)
            false_positives = torch.zeros((n_class_detections), dtype=torch.float).cuda()  # (n_class_detections)
        
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > 0.5:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        if on_dev:
            precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        else:
            precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).cuda()
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

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
        
# ====================================================================================================================================
# ------------------------------------------------------------------------------------------------------------------------------------

def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()
    
    width = int(root.find('size').find('width').text)
    height= int(root.find('size').find('height').text)
    
    for object in root.iter('object'):

        difficult = int(object.find('difficult').text == '1')

        label = object.find('name').text.lower().strip()
        if label not in label_map:
            continue

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}, (height,width)

def create_data_lists(voc07_path, voc08_path, voc09_path, voc10_path, voc12_path,  output_folder):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.

    :param voc07_path: path to the 'VOC2007' folder
    :param voc12_path: path to the 'VOC2012' folder
    :param output_folder: folder where the JSONs must be saved
    """
    voc07_path = os.path.abspath(voc07_path)
    voc08_path = os.path.abspath(voc08_path)
    voc09_path = os.path.abspath(voc09_path)
    voc10_path = os.path.abspath(voc10_path)
    voc12_path = os.path.abspath(voc12_path)
    
    voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
    label_map['background'] = 0
    rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

    # Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
    distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                       '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                       '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
    label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}
    
    train_images = list()
    train_objects = list()
    n_objects = 0
    
    path_list = [voc07_path, 
#                  voc08_path, 
#                  voc09_path, 
#                  voc10_path, 
                 voc12_path
                ]
    # Training data
    for path in path_list:

        # Find IDs of images in training data
        with open(os.path.join(path, 'ImageSets/Main/trainval.txt')) as f:
            ids = f.read().splitlines()
            
        for id in ids:
            # Parse annotation's XML file
            objects, size = parse_annotation(os.path.join(path, 'Annotations', id + '.xml'))
            # 
            if len(objects) == 0:
                continue
            if size[1] ==500:
                n_objects += len(objects)
                train_objects.append(objects)
                train_images.append(os.path.join(path, 'JPEGImages', id + '.jpg'))

    assert len(train_objects) == len(train_images)

    # Save to file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  # save label map too

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_objects, os.path.abspath(output_folder)))

    # Validation data
    test_images = list()
    test_objects = list()
    n_objects = 0

    # Find IDs of images in validation data
    with open(os.path.join(voc07_path, 'ImageSets/Main/val.txt')) as f: # test
        ids = f.read().splitlines()

    for i, id in enumerate(ids):
#         #TEST CODE
#         if i>=4:
#             break
        # Parse annotation's XML file
        objects, size = parse_annotation(os.path.join(voc07_path, 'Annotations', id + '.xml'))
        if len(objects) == 0:
            continue
        if size[1] ==500:
            test_objects.append(objects)
            n_objects += len(objects)
            test_images.append(os.path.join(voc07_path, 'JPEGImages', id + '.jpg'))

    assert len(test_objects) == len(test_images)

    # Save to file
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print('\nThere are %d validation images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), n_objects, os.path.abspath(output_folder)))
    
import random    
class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, transforms=None, keep_difficult=False, max_size = None):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST' or 'TEST_RED'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'VAL', 'TEST'}
        self.transforms = transforms
        self.data_folder = data_folder
        self.keep_difficult = keep_difficult
        self.max_size = max_size
        
        if self.max_size is None:
            self.max_size = 500

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        try:
            # Read image
            image = Image.open(self.images[i], mode='r')
            image = image.convert('RGB')
        except:
            print('Corrupted file - ',self.images[i])
            image = Image.open(self.images[i+1], mode='r')
            image = image.convert('RGB')
            
            # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        target = {'boxes': boxes, 'labels': labels}
        # Apply transformations
        # ========================================
#         if (max(image.size)>=self.max_size+200):#&(self.split=='TRAIN'):
#             image , target = target_crop(image, target, self.max_size)
#         # ========================================
#         else:
#             tr_coeff = self.max_size/max(list(image.size))
#             new_size = (int(image.size[1]* tr_coeff) , int(image.size[0]* tr_coeff))
#             image, target = Resize(new_size)(image,target)

        if self.transforms is not None:
            image, target = self.transforms(image,target)

        if self.split != 'TEST':
            return image, target
        else:
            target['difficulties'] =  difficulties
            return image, target
        
            
    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each

def get_transform(train = None):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(ToTensor())
    if train:
        # during training, randomly flip the training images and ground-truth for data augmentation
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)

class Resize(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, image, target):
        original_size = image.size
        image = torchvision.transforms.Resize(self.size,)(image)
        new_size = image.size
        target['boxes']= resize_boxes(target['boxes'], original_size, new_size) 
        return image, target
# TODO
class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-2)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
#             if "masks" in target:
#                 target["masks"] = target["masks"].flip(-2)
#             if "keypoints" in target:
#                 keypoints = target["keypoints"]
#                 keypoints = _flip_coco_person_keypoints(keypoints, width)
#                 target["keypoints"] = keypoints
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target
    
class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    
def collate_fn(batch):
    return tuple(zip(*batch))

def resize_boxes(boxes, original_size, new_size):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    # return torch.stack((xmin, ymin, xmax, ymax), dim=1)
    return torch.stack((torch.round(xmin), torch.round(ymin), torch.round(xmax), torch.round(ymax)), dim=1,)

def target_crop(image, target, max_size):
    left = 10e4
    right = 0
    top = 10e4
    bot = 0
    pad = random.randint(0, int(min(image.size)/5))
    objects = list()
    for obj in target['boxes']:
        x1 = obj.numpy()[0]
        y1 = obj.numpy()[1]
        x2 = obj.numpy()[2]
        y2 = obj.numpy()[3]
        objects.append([x1,y1,x2,y2])

    for ob in objects:
        if ob[0]<left:   left  = ob[0];
        if ob[1]<top:    top   = ob[1];
        if ob[2]>right:  right = ob[2];
        if ob[3]>bot:    bot   = ob[3];

    center = (int((left+right)/2), int((top+bot)/2))
    step_x = int((right-left)/2)
    step_y = int((  bot-top )/2)

    w,h = image.size
    if center[0]-step_x>=0: new_left = center[0]-step_x
    else: new_left = 0
    if center[0]+step_x<=w: new_right = center[0]+step_x
    else: new_right = w
    if center[1]-step_y>=0: new_top = center[1]-step_y
    else: new_top = 0
    if center[1]+step_y<=h: new_bot = center[1]+step_y
    else: new_bot = h

        
    if new_left-pad<0: new_x1 = 0
    else: new_x1 = new_left-pad
    if new_top-pad<0:new_y1 = 0    
    else: new_y1 = new_top-pad
    if new_right+pad>image.size[0]: new_x2 = image.size[0]
    else: new_x2 = new_right+pad
    if new_bot + pad>image.size[1]: new_y2 = image.size[1]
    else: new_y2 = new_bot+pad

    rez = torchvision.transforms.functional.crop( image , new_y1, new_x1,
                                           new_y2 - new_y1 , new_x2 - new_x1 )
    ncoord = list()
    if max(rez.size)>=max_size:
        coeff = max(rez.size)/max_size
    else:
        if max(rez.size)*1.2 >=max_size:
            coeff = (max(rez.size)*1.2)/max_size
        else:
            coeff = 1.2
            
    for obj in objects:
        ncoord.append([int((obj[0]-new_x1)/coeff),int((obj[1]-new_y1)/coeff),
                       int((obj[2]-new_x1)/coeff),int((obj[3]-new_y1)/coeff)])
    target['boxes'] = torch.FloatTensor(ncoord)
    image = torchvision.transforms.Resize((int(rez.size[1]/coeff),int(rez.size[0]/coeff)))(rez)
    return image, target

# ====================================================================================================================================
import warnings
warnings.filterwarnings("ignore")


from pprint import PrettyPrinter
pp = PrettyPrinter()

def evaluate(test_loader, model, data_folder, device = None ):
    """0.623188
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """
    if device is not None:
        model = model.to(device)
        on_dev = True
    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', 
                                # see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        for i, (images, targets) in enumerate(tqdm(test_loader,'Evaluation')):
            if on_dev:
                # images = images.to(device)
                images = [img.to(device) for img in images]
            
            det_boxes_batch = []
            det_labels_batch = []
            det_scores_batch = []
            # Forward prop.
            results = model(images) # список словарей
            for rez in results:
                det_boxes_batch.append(rez['boxes'])
                det_labels_batch.append(rez['labels']) 
                det_scores_batch.append(rez['scores'])
            
            boxes = []
            labels = []
            difficulties = []
            for el in targets:
                boxes.append(el['boxes'])
                labels.append(el['labels'])
                difficulties.append(el['difficulties'])
            
            if on_dev:
                boxes = [b.to(device) for b in boxes]
                labels = [l.to(device) for l in labels]
                difficulties = [d.to(device) for d in difficulties]
            
            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        # Calculate mAP
        # print(label_map)
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, 
                                 true_boxes, true_labels, true_difficulties, on_dev, data_folder)

    # Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.5f' % mAP)
    return APs, mAP


def resample_dataset(conf_thershold, last_state, div_coeff, wtf=False, random_choice = True):
    data_folder = './data/scans'
    path = '23k'
    path_23k = os.path.join(data_folder,path)
    
    data_scans = os.path.abspath('./data/scans')
    dst_path = path+'_'+str(div_coeff)
    print(dst_path)
    path_to_vfolder = os.path.join(data_scans,dst_path)
    path_to_vfile = os.path.join(path_to_vfolder,'mAPs.pkl')

    df = pd.read_pickle(path_to_vfile)
    rl = last_state#df[df.mAP == df.mAP.max()]
    diff_objs = {}
    for i,j in enumerate(rl[rl.columns[2:]].iloc[0].values):
        if j >=conf_thershold:
            j = 10000000
        else:
            if j == 0:
                j = 0.05
        diff_objs[i+1] = round((1/j)**2,3)
    print(diff_objs)
#     zeros = 0
#     for key in diff_objs.keys():
#         if diff_objs[key]< 0.001:
#             zeros+=1
#     zero_coeff = len(diff_objs)/(len(diff_objs) - zeros)
    
    dat =''
    for file in os.listdir(path_23k):
        if ('objects' in file):
            print(file)
            split = file.split('_')[0]

            with open(os.path.join(path_23k,file), 'r') as f:
                data=json.load(f)
            with open(os.path.join(path_23k,split+'_images.json'), 'r') as f:
                images=json.load(f)
            rez = Counter(data[0]['labels'])
            rez1 = rez-rez
            rez = rez-rez
            # print('data length - ',len(data))
            for line in data:
                rez1+=Counter(line['labels'])
            indexes = list()
            index_pool = [i for i in range(len(data))]

            for i in range(len(data)):
                ind = random.choice(index_pool)
                line = data[ind]
                tmp = rez+Counter(line['labels'])
                flag = 1

                for key in tmp.keys():
                    #print('key = ', key)
                    if split.lower() in ['val','test']:
                        limiter = int((sum(rez1.values())/len(rez1.values()))/div_coeff)
                        limiter *=2
                    else:
                        limiter = int((sum(rez1.values())/len(rez1.values()))/div_coeff)
                        
                    if split.lower() == 'train':
                        
                        limiter = int(limiter*diff_objs[key]) # *zero_coeff
                    #print(key, limiter)
                    
                    if tmp[key]>limiter:
                        flag = 0

                if flag == 1:
                    rez = tmp
                    indexes.append(ind)    
                index = index_pool.index(ind)
                del index_pool[index]

            print(len(indexes), rez)
            new_imagelist = [images[i] for i in indexes]
            new_objectlist= [data[i]   for i in indexes]

            if wtf:
                if not os.path.exists(os.path.join(data_folder,dst_path)):
                    os.mkdir(os.path.join(data_folder,dst_path))
                    file = 'label_map.json'
                    src = os.path.join(path_23k, file)
                    dst = os.path.join(data_folder, dst_path, file)
                    shutil.copyfile(src, dst)

                output_folder = os.path.join(data_folder, dst_path)
                with open(os.path.join(output_folder, split+'_images.json'), 'w') as j:
                        json.dump(new_imagelist, j)
                with open(os.path.join(output_folder, split+'_objects.json'), 'w') as j:
                        json.dump(new_objectlist, j)
                print('Date updated, changes committed!')
                print('------------------------------------')
#             else:
#                 print('Date updated, changes NOT committed!')
#                 print('------------------------------------')


def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    return im