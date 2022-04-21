from __future__ import print_function
from show3d_balls import showpoints
import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from dataset import ShapeNetDataset
from model import PointNetCls
import matplotlib.pyplot as plt
import pdb


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='cls/weights_with_transform/best_classification.pt', help='model path')
parser.add_argument('--idx', type=int, default=2, help='model index')
parser.add_argument('--dataset', type=str, default='../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0', help='dataset path')
parser.add_argument('--class_choice', type=str, default='Chair', help='class choice')  # Airplane
parser.add_argument('--feature_transform', default='True', help="use feature transform")

opt = parser.parse_args()
print(opt)

d = ShapeNetDataset(
    root=opt.dataset,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)

idx = opt.idx

print("model %d/%d" % (idx, len(d)))
point, seg = d[idx]
print(point.size(), seg.size())
point_np = point.numpy()

state_dict = torch.load(opt.model)

classifier = PointNetCls(num_classes=state_dict['model']['fc3.weight'].size()[0], feature_transform=opt.feature_transform)
classifier.load_state_dict(state_dict['model'])
classifier.eval()

point = point.transpose(1, 0).contiguous()

point = Variable(point.view(1, point.size()[0], point.size()[1]))
pred, trans, trans_feat, critical_point = classifier(point)

critical_point = critical_point[0].permute(1, 0).detach().numpy()

#print(pred_color.shape)
showpoints(point_np)
showpoints(critical_point)

