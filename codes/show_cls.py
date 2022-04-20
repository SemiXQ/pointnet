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
parser.add_argument('--model', type=str, default='../pretrained_networks/latest_classification.pt',  help='model path')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--feature_transform', default='True', help="use feature transform")

opt = parser.parse_args()
print(opt)

# test_dataset = ShapeNetDataset(
#     root='../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0',
#     npoints=opt.num_points,
#     classification=True,
#     split='test',
#     data_augmentation=False)

test_dataset = ShapeNetDataset(
    root='../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0',
    npoints=opt.num_points,
    classification=True,
    split='test')

# test_dataloader = torch.utils.data.DataLoader(
#     test_dataset,
#     batch_size=32,
#     shuffle=True,
#     num_workers=4)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True)

# print(len(test_dataset))
# num_classes = len(test_dataset.classes)
# print('classes', num_classes)

classifier = PointNetCls(num_classes=len(test_dataset.classes), feature_transform=opt.feature_transform)
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model)['model'])
classifier.eval()


total_preds = []
total_targets = []
with torch.no_grad():
    for i, data in enumerate(test_dataloader, 0):
        # TODO
        # calculate average classification accuracy
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        preds, _, _ = classifier(points)
        pred_labels = torch.max(preds, dim=1)[1]

        total_preds = np.concatenate([total_preds, pred_labels.cpu().numpy()])
        total_targets = np.concatenate([total_targets, target.cpu().numpy()])
        a = 0
    accuracy = 100 * (total_targets == total_preds).sum() / len(test_dataset)
    print('Test Accuracy = {:.2f}%'.format(accuracy))
