from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from dataset import ShapeNetDataset
from model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--model', type=str, default='seg/weights_with_transform/Chair/best_segmentation.pt', help='model path')
parser.add_argument('--dataset', type=str, default='../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0', help="dataset path")
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--feature_transform', default='True', help="use feature transform")

opt = parser.parse_args()
print(opt)

val_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

num_classes = val_dataset.num_seg_classes
print('classes', num_classes)


classifier = PointNetDenseCls(num_classes=num_classes, feature_transform=opt.feature_transform)
classifier.load_state_dict(torch.load(opt.model)['model'])
classifier.cuda()

num_batch = len(val_dataloader)

# benchmark mIOU
classifier.eval()
shape_ious = []
with torch.no_grad():
    for i, data in tqdm(enumerate(val_dataloader, 0)):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(2)[1]  # 1

        pred_np = pred_choice.cpu().data.numpy()
        target_np = target.cpu().data.numpy() - 1

        for shape_idx in range(target_np.shape[0]):
            parts = range(num_classes)#np.unique(target_np[shape_idx])
            part_ious = []
            for part in parts:
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                if U == 0:
                    iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
                else:
                    iou = I / float(U)
                part_ious.append(iou)
            shape_ious.append(np.mean(part_ious))

    print("mIOU for class {}: {:.4f}".format(opt.class_choice, np.mean(shape_ious)))
