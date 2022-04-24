from __future__ import print_function

import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pdb


class TNet(nn.Module):
    def __init__(self, k=64):
        super(TNet, self).__init__()

        self.k = k

        # Each layer has batchnorm and relu on it
        # TODO
        # layer 1: k -> 64
        self.mlpTo64 = nn.Sequential(
            nn.Conv1d(in_channels=k, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU()
        )

        # TODO
        # layer 2:  64 -> 128
        self.mlpTo128 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()
        )

        # TODO
        # layer 3: 128 -> 1024
        self.mlpTo1024 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU()
        )

        # TODO
        # fc 1024 -> 512
        self.fcTo512 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU()
        )

        # TODO
        # fc 512 -> 256
        self.fcTo256 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU()
        )

        # TODO
        # fc 256 -> k*k (no batchnorm, no relu)
        self.fcTokk = nn.Linear(in_features=256, out_features=self.k*self.k)

        learnable_bias = torch.eye(n=self.k, requires_grad=True)
        self.learnable_bias = nn.Parameter(learnable_bias)

        # TODO
        # ReLU activationfunction
        # I have done relu in nn.Sequential


    def forward(self, x):
        batch_size, _, num_points = x.shape
        # TODO
        # apply layer 1
        x = self.mlpTo64(x)

        # TODO
        # apply layer 2
        x = self.mlpTo128(x)

        # TODO
        # apply layer 3
        x = self.mlpTo1024(x)

        # TODO
        # do maxpooling and flatten
        # since keepdim = False, after max operation, it should be flatten automatically
        x = torch.max(x, dim=2, keepdim=False)[0]

        # TODO
        # apply fc layer 1
        x = self.fcTo512(x)

        # TODO
        # apply fc layer 2
        x = self.fcTo256(x)

        # TODO
        # apply fc layer 3
        x = self.fcTokk(x)

        # TODO
        #reshape output to a b*k*k tensor
        x = x.view(-1, self.k, self.k)

        #TODO
        # define an identity matrix to add to the output.
        # This will help with the stability of the results since we want our transformations to be close to identity
                
        x = self.learnable_bias + x

        #TODO
        # return output
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = True):
        super(PointNetfeat, self).__init__()

        self.feature_transform = feature_transform
        self.global_feat = global_feat

        #TODO
        # Use TNet to apply transformation on input and multiply the input points with the transformation
        self.trans = TNet(3)

        #TODO
        # layer 1: 3 -> 64
        self.mlp3To64 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU()
        )

        #TODO
        # Use TNet to apply transformation on features and multiply the input features with the transformation 
        #                                                                        (if feature_transform is true)
        self.trans_if = TNet(64)

        #TODO
        # layer2: 64 -> 128
        self.mlp64To128 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()
        )

        # TODO
        # layer 3: 128 -> 1024 (no relu)
        self.mlp128To1024 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1),
            nn.BatchNorm1d(num_features=1024)
        )

        # TODO
        # ReLU activation
        # I have done relu in each nn.Sequential



    def forward(self, x):
        batch_size, _, num_points = x.shape
        temp = x.clone()
        # TODO
        # input transformation,
        # you will need to return the transformation matrix as you will need it for the regularization loss
        trans_input = self.trans(x)
        # change from (batch, dim, point_num) to (batch, point_num, dim)
        x = torch.permute(x, (0, 2, 1))
        x = torch.bmm(x, trans_input)
        # change from (batch, point_num, dim) to (batch, dim, point_num)
        x = torch.permute(x, (0, 2, 1))

        # TODO
        # apply layer 1
        y = self.mlp3To64(x)

        # TODO
        # feature transformation,
        # you will need to return the transformation matrix as you will need it for the regularization loss
        if self.feature_transform:
            trans_feature = self.trans_if(y)
            # change from (batch, dim, point_num) to (batch, point_num, dim)
            y = torch.permute(y, (0, 2, 1))
            y = torch.bmm(y, trans_feature)
            # change from (batch, point_num, dim) to (batch, dim, point_num)
            y = torch.permute(y, (0, 2, 1))
        else:
            trans_feature = None

        # TODO
        # apply layer 2
        x = self.mlp64To128(y)

        # TODO
        # apply layer 3
        x = self.mlp128To1024(x)

        # TODO
        # apply maxpooling
        idx_list = torch.argmax(x, dim=2)
        critical_points = torch.index_select(temp, 2, idx_list[0])
        x = torch.max(x, dim=2, keepdim=False)[0]

        # TODO
        # return output, input transformation matrix, feature transformation matrix
        if self.global_feat:  # This shows if we're doing classification or segmentation
            # when doing classification
            return x, trans_input, trans_feature, critical_points
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, num_points)
            x = torch.cat((y, x), dim=1)
            return x, trans_input, trans_feature, critical_points


class PointNetCls(nn.Module):
    def __init__(self, num_classes = 2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat, critical_points = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat, critical_points


class PointNetDenseCls(nn.Module):
    def __init__(self, num_classes = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        #TODO
        # get global features + point features from PointNetfeat
        self.feature_transform = feature_transform
        self.pointnet_feature = PointNetfeat(global_feat=False, feature_transform=self.feature_transform)

        self.k = num_classes

        #TODO
        # layer 1: 1088 -> 512
        self.mlp1088To512 = nn.Sequential(
            nn.Conv1d(in_channels=1088, out_channels=512, kernel_size=1),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU()
        )

        #TODO
        # layer 2: 512 -> 256
        self.mlp512To256 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU()
        )

        #TODO
        # layer 3: 256 -> 128
        self.mlp256To128 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()
        )

        #TODO
        # layer 4:  128 -> k (no ru and batch norm)
        self.mlp128TokClass = nn.Conv1d(in_channels=128, out_channels=self.k, kernel_size=1)

        #TODO
        # ReLU activation
        # I have done relu in each nn.Sequential

    
    def forward(self, x):
        # TODO
        # You will need these extra outputs: 
        # trans = output of applying TNet function to input
        # trans_feat = output of applying TNet function to features (if feature_transform is true)
        # (you can directly get them from PointNetfeat)
        x, trans, trans_feat, _ = self.pointnet_feature(x)
        
        batch_size, _, num_points = x.shape

        # TODO
        # apply layer 1
        x = self.mlp1088To512(x)

        # TODO
        # apply layer 2
        x = self.mlp512To256(x)

        # TODO
        # apply layer 3
        x = self.mlp256To128(x)

        # TODO
        # apply layer 4
        x = self.mlp128TokClass(x)

        # TODO
        # apply log-softmax
        x = F.log_softmax(x, dim=1)
        x = torch.permute(x, (0, 2, 1))
        
        return x, trans, trans_feat


def feature_transform_regularizer(trans):

    batch_size, feature_size, _ = trans.shape
    # TODO
    # compute I - AA^t
    i = torch.eye(n=feature_size)
    if trans.is_cuda:
        i = i.cuda()
    output = i - torch.bmm(trans, torch.transpose(trans, dim0=1, dim1=2))

    # TODO
    # compute norm
    output = torch.norm(output, p='fro', dim=(1, 2))

    # TODO
    # compute mean norms and return
    reg_loss = torch.mean(output)
    return reg_loss


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32, 3, 2500))
    trans = TNet(k=3)
    out = trans(sim_data)
    print('TNet', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = TNet(k=64)
    out = trans(sim_data_64d)
    print('TNet 64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(num_classes=5)
    out, _, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(num_classes=3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())

