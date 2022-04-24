from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from dataset import ShapeNetDataset
from model import PointNetCls
import torch.nn.functional as F

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='cls/weights_with_transform/best_classification.pt',  help='model path')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--feature_transform', default='True', help="use feature transform")

opt = parser.parse_args()
print(opt)

# As it required in assignment instruction, this code will show the accuracy on both train and test set at the same time
train_dataset = ShapeNetDataset(
    root='../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0',
    classification=True,
    npoints=opt.num_points)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4)

test_dataset = ShapeNetDataset(
    root='../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0',
    npoints=opt.num_points,
    classification=True,
    split='test')

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4)

classifier = PointNetCls(num_classes=len(test_dataset.classes), feature_transform=opt.feature_transform)
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model)['model'])
classifier.eval()


total_preds_train_set = []
total_targets_train_set = []
total_preds = []
total_targets = []
with torch.no_grad():
    for i, data in enumerate(train_dataloader, 0):
        # TODO
        # calculate average classification accuracy
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        preds, _, _, _ = classifier(points)
        pred_labels = torch.max(preds, dim=1)[1]

        total_preds_train_set = np.concatenate([total_preds_train_set, pred_labels.cpu().numpy()])
        total_targets_train_set = np.concatenate([total_targets_train_set, target.cpu().numpy()])
        a = 0
    train_accuracy = 100 * (total_targets_train_set == total_preds_train_set).sum() / len(train_dataset)
    print('Train Accuracy = {:.2f}%'.format(train_accuracy))
    for i, data in enumerate(test_dataloader, 0):
        # TODO
        # calculate average classification accuracy
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        preds, _, _, _ = classifier(points)
        pred_labels = torch.max(preds, dim=1)[1]

        total_preds = np.concatenate([total_preds, pred_labels.cpu().numpy()])
        total_targets = np.concatenate([total_targets, target.cpu().numpy()])
        a = 0
    accuracy = 100 * (total_targets == total_preds).sum() / len(test_dataset)
    print('Test Accuracy = {:.2f}%'.format(accuracy))
