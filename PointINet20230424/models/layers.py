import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# We utilize kaolin to implement layers for FlowNet3D
# website: https://github.com/NVIDIAGameWorks/kaolin
# import kaolin as kal
# from kaolin.models.PointNet2 import furthest_point_sampling
# from kaolin.models.PointNet2 import fps_gather_by_index
# from kaolin.models.PointNet2 import ball_query
# from kaolin.models.PointNet2 import group_gather_by_index
# from kaolin.models.PointNet2 import three_nn

from .pointnet2_utils import farthest_point_sample, index_points, square_distance, query_ball_point
from pytorch3d.ops import knn_points, knn_gather

'''
Layers for FlowNet3D
'''


class Sample(nn.Module):
    '''
    Furthest point sample
    '''

    def __init__(self, num_points):
        super(Sample, self).__init__()

        self.num_points = num_points

    def forward(self, points):
        points = points.permute(0, 2, 1).contiguous()  # [B,N,C]
        new_points_ind = farthest_point_sample(points, self.num_points)  # [B, npoint]
        new_points = index_points(points, new_points_ind)  # [B,S,C]
        return new_points.permute(0, 2, 1).contiguous()  # [B,C,S]


class Group(nn.Module):
    '''
    kNN group for FlowNet3D
    '''

    def __init__(self, radius, num_samples, knn=False):
        super(Group, self).__init__()

        self.radius = radius
        self.num_samples = num_samples
        self.knn = knn

    def forward(self, points, new_points, features):
        points = points.permute(0, 2, 1).contiguous()  # [B,N,C]
        # print("1:",points.shape)
        new_points = new_points.permute(0, 2, 1).contiguous()  # [B,S,C]
        # print("2:", new_points.shape)
        B, S, C = new_points.shape
        features = features.permute(0, 2, 1).contiguous()  # [B,N,D]
        # print("3:", features.shape)
        if self.knn:
            dist = square_distance(points, new_points)  # [B,N,S]
            ind = dist.topk(self.num_samples, dim=1, largest=False)[1].permute(0, 2,
                                                                               1).contiguous()  # [B, S, nsample]
            # print("4:", ind, ind.shape)
        else:
            ind = query_ball_point(self.radius, self.num_samples, points, new_points)  # [B, S, nsample]
            # print("4:", ind.shape)
        grouped_points = index_points(points, ind)  # [B, npoint, nsample, C]
        # print("5:",grouped_points.shape)
        grouped_points_new = grouped_points - new_points.view(B, S, 1, C)
        # print("6:", grouped_points_new.shape)
        grouped_features = index_points(features, ind)  # [B, npoint, nsample, D]
        # print("7:", grouped_features.shape)
        new_features = torch.cat([grouped_points_new, grouped_features], dim=-1)  # [B, npoint, nsample, C+D]
        # print("8:",new_features.shape)
        return new_features.permute(0, 3, 2, 1).contiguous()  # [B, C+D, nsample, npoint]


class SetConv(nn.Module):
    def __init__(self, num_points, radius, num_samples, in_channels, out_channels):
        super(SetConv, self).__init__()

        self.sample = Sample(num_points)
        self.group = Group(radius, num_samples)

        layers = []
        out_channels = [in_channels + 3, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True),
                       nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        self.conv = nn.Sequential(*layers)

    def forward(self, points, features):  # points:[B,C,N],features:[B,D,N]
        new_points = self.sample(points)
        # print("1:",new_points.shape)
        new_features = self.group(points, new_points, features)
        # print("2:", new_features.shape)
        new_features = self.conv(new_features)
        # print("3:", new_features.shape)
        new_features = new_features.max(dim=2)[0]
        # print("4:", new_features.shape)
        return new_points, new_features


class FlowEmbedding(nn.Module):
    def __init__(self, num_samples, in_channels, out_channels):
        super(FlowEmbedding, self).__init__()

        self.num_samples = num_samples

        self.group = Group(None, self.num_samples, knn=True)

        layers = []
        out_channels = [2 * in_channels + 3, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True),
                       nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        self.conv = nn.Sequential(*layers)

    def forward(self, points1, points2, features1, features2):
        new_features = self.group(points2, points1, features2)
        # print("1:", new_features.shape)
        new_features = torch.cat([new_features, features1.unsqueeze(2).expand(-1, -1, self.num_samples, -1)], dim=1)
        # print("2:", new_features.shape)
        new_features = self.conv(new_features)
        # print("3:", new_features.shape)
        new_features = new_features.max(dim=2)[0]
        # print("4:", new_features.shape)
        return new_features


class SetUpConv(nn.Module):
    def __init__(self, num_samples, in_channels1, in_channels2, out_channels1, out_channels2):
        super(SetUpConv, self).__init__()

        self.group = Group(None, num_samples, knn=True)

        layers = []
        out_channels1 = [in_channels1 + 3, *out_channels1]
        for i in range(1, len(out_channels1)):
            layers += [nn.Conv2d(out_channels1[i - 1], out_channels1[i], 1, bias=True),
                       nn.BatchNorm2d(out_channels1[i], eps=0.001), nn.ReLU()]
        self.conv1 = nn.Sequential(*layers)

        layers = []
        if len(out_channels1) == 1:
            out_channels2 = [in_channels1 + in_channels2 + 3, *out_channels2]
        else:
            out_channels2 = [out_channels1[-1] + in_channels2, *out_channels2]
        for i in range(1, len(out_channels2)):
            layers += [nn.Conv2d(out_channels2[i - 1], out_channels2[i], 1, bias=True),
                       nn.BatchNorm2d(out_channels2[i], eps=0.001), nn.ReLU()]
        self.conv2 = nn.Sequential(*layers)

    def forward(self, points1, points2, features1, features2):
        new_features = self.group(points1, points2, features1)
        # print("1:", new_features.shape)
        new_features = self.conv1(new_features)
        # print("2:", new_features.shape)
        new_features = new_features.max(dim=2)[0]
        # print("3:", new_features.shape)
        new_features = torch.cat([new_features, features2], dim=1)
        # print("4:", new_features.shape)
        new_features = new_features.unsqueeze(3)
        # print("5:", new_features.shape)
        new_features = self.conv2(new_features)
        # print("6:", new_features.shape)
        new_features = new_features.squeeze(3)
        # print("7:", new_features.shape)
        return new_features


class FeaturePropagation(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(FeaturePropagation, self).__init__()

        layers = []
        out_channels = [in_channels1 + in_channels2, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True),
                       nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        self.conv = nn.Sequential(*layers)

    def forward(self, points1, points2, features1, features2):  # [B,C,S];[B,C,N];[B,D1,S];[B,D2,N]
        points1 = points1.permute(0, 2, 1).contiguous()  # [B,S,C]
        points2 = points2.permute(0, 2, 1).contiguous()  # [B,N,C]
        B, N, _ = points2.shape
        features1 = features1.permute(0, 2, 1).contiguous()  # [B,S,D1]

        dists = square_distance(points2, points1)  # [B,N,S]
        dists, ind = dists.sort(dim=-1)
        dists, ind = dists[:, :, :3], ind[:, :, :3]  # [B,N,3]
        dists[dists < 1e-10] = 1e-10
        inverse_dist = 1.0 / dists
        norm = torch.sum(inverse_dist, dim=2, keepdim=True)
        weights = inverse_dist / norm
        new_features = torch.sum(index_points(features1, ind) * weights.view(B, N, 3, 1),
                                 dim=2)  # 从points中取3个点进行加权平均 [B,N,3,D2]*[B,N,3,1]→(sum dim=2)→[B,N,D1]
        new_features = new_features.permute(0, 2, 1).contiguous()  # [B,D1,N]
        new_features = torch.cat([new_features, features2], dim=1)  # [B,D1+D2,N]
        new_features = self.conv(new_features.unsqueeze(3)).squeeze(3)
        return new_features  # [B, D', N]


# class PointsFusion(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(PointsFusion, self).__init__()
#
#         layers = []
#         out_channels = [in_channels, *out_channels]
#         for i in range(1, len(out_channels)):
#             layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True),
#                        nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
#
#         self.conv = nn.Sequential(*layers)
#
#     def knn_points(self, points1, points2, K=64, return_nn=True):
#         dist = square_distance(points1, points2)  # [B,N,N]
#         ind = dist.topk(K, dim=1, largest=False)[1].permute(0, 2, 1).contiguous()  # [B,N,K]
#         if return_nn == False:
#             return dist, ind
#         else:
#             return dist, ind, index_points(points1,ind)
#
#     def knn_group(self, points1, points2, features2, k):
#         '''
#         For each point in points1, query kNN points/features in points2/features2
#         Input:
#             points1: [B,3,N]
#             points2: [B,3,N]
#             features2: [B,C,N]
#         Output:
#             new_features: [B,4,N,k]
#             nn: [B,3,N,k]
#             grouped_features: [B,C,N,k]
#         '''
#         points1 = points1.permute(0, 2, 1).contiguous()  # [B,N,3]
#         points2 = points2.permute(0, 2, 1).contiguous()  # [B,N,3]
#         features2 = features2.permute(0, 2, 1)  # [B,N,C]
#         _, nn_idx, nn = self.knn_points(points1, points2, K=k, return_nn=True)  # nn:[B,N,k,3]; nn_idx:[B,N,k]
#         points_resi = nn - points1.unsqueeze(2).repeat(1, 1, k, 1)  # [B,N,k,3]
#         grouped_dist = torch.norm(points_resi, dim=-1, keepdim=True)  # [B,N,k,1]
#         grouped_features = index_points(features2, nn_idx)  # [B,N,k,C]
#         new_features = torch.cat([points_resi, grouped_dist], dim=-1)  # [B,N,k,4]
#
#         return new_features.permute(0, 3, 1, 2).contiguous(), \
#                nn.permute(0, 3, 1, 2).contiguous(), \
#                grouped_features.permute(0, 3, 1, 2).contiguous()
#
#     def forward(self, points1, points2, features1, features2, k, t):
#         '''
#         Input:
#             points1: [B,3,N]
#             points2: [B,3,N]
#             features1: [B,C,N] (only for inference of additional features)
#             features2: [B,C,N] (only for inference of additional features)
#             k: int, number of kNN cluster
#             t: [B], time step in (0,1)
#         Output:
#             fused_points: [B,3+C,N]
#         '''
#         N = points1.shape[-1]
#         B = points1.shape[0]  # batch size
#
#         new_features_list = []
#         new_grouped_points_list = []
#         new_grouped_features_list = []
#
#         for i in range(B):
#             t1 = t[i]
#             new_points1 = points1[i:i + 1, :, :]
#             new_points2 = points2[i:i + 1, :, :]
#             new_features1 = features1[i:i + 1, :, :]
#             new_features2 = features2[i:i + 1, :, :]
#
#             N2 = int(N * t1)
#             N1 = N - N2
#
#             k2 = int(k * t1)
#             k1 = k - k2
#
#             randidx1 = torch.randperm(N)[:N1]
#             randidx2 = torch.randperm(N)[:N2]
#             new_points = torch.cat((new_points1[:, :, randidx1], new_points2[:, :, randidx2]), dim=-1)
#
#             new_features1, grouped_points1, grouped_features1 = self.knn_group(new_points, new_points1, new_features1,
#                                                                                k1)
#             new_features2, grouped_points2, grouped_features2 = self.knn_group(new_points, new_points2, new_features2,
#                                                                                k2)
#
#             new_features = torch.cat((new_features1, new_features2), dim=-1)
#             new_grouped_points = torch.cat((grouped_points1, grouped_points2), dim=-1)
#             new_grouped_features = torch.cat((grouped_features1, grouped_features2), dim=-1)
#
#             new_features_list.append(new_features)
#             new_grouped_points_list.append(new_grouped_points)
#             new_grouped_features_list.append(new_grouped_features)
#
#         new_features = torch.cat(new_features_list, dim=0)
#         new_grouped_points = torch.cat(new_grouped_points_list, dim=0)
#         new_grouped_features = torch.cat(new_grouped_features_list, dim=0)
#
#         new_features = self.conv(new_features)
#         new_features = torch.max(new_features, dim=1, keepdim=False)[0]
#         weights = F.softmax(new_features, dim=-1)
#
#         C = features1.shape[1]
#         weights = weights.unsqueeze(1).repeat(1, 3 + C, 1, 1)
#         fused_points = torch.cat([new_grouped_points, new_grouped_features], dim=1)
#         fused_points = torch.sum(torch.mul(weights, fused_points), dim=-1, keepdim=False)
#
#         return fused_points
#
#
# def knn_group_withI(points1, points2, intensity2, k):
#     '''
#     Input:
#         points1: [B,3,N]
#         points2: [B,3,N]
#         intensity2: [B,1,N]
#     '''
#     points1 = points1.permute(0, 2, 1).contiguous()
#     points2 = points2.permute(0, 2, 1).contiguous()
#     _, nn_idx, nn = knn_points(points1, points2, K=k, return_nn=True)
#     points_resi = nn - points1.unsqueeze(2).repeat(1, 1, k, 1)  # [B,M,k,3]
#     grouped_dist = torch.norm(points_resi, dim=-1, keepdim=True)
#     grouped_features = knn_gather(intensity2.permute(0, 2, 1), nn_idx)  # [B,M,k,1]
#     new_features = torch.cat([points_resi, grouped_dist], dim=-1)
#
#     # [B,5,M,k], [B,3,M,k], [B,1,M,k]
#     return new_features.permute(0, 3, 1, 2).contiguous(), \
#            nn.permute(0, 3, 1, 2).contiguous(), \
#            grouped_features.permute(0, 3, 1, 2).contiguous()

class PointsFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointsFusion, self).__init__()

        layers = []
        out_channels = [in_channels, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]

        self.conv = nn.Sequential(*layers)

    def knn_group(self, points1, points2, features2, k):
        '''
        For each point in points1, query kNN points/features in points2/features2
        Input:
            points1: [B,3,N]
            points2: [B,3,N]
            features2: [B,C,N]
        Output:
            new_features: [B,4,N]
            nn: [B,3,N]
            grouped_features: [B,C,N]
        '''
        points1 = points1.permute(0,2,1).contiguous()
        points2 = points2.permute(0,2,1).contiguous()
        _, nn_idx, nn = knn_points(points1, points2, K=k, return_nn=True)
        points_resi = nn - points1.unsqueeze(2).repeat(1,1,k,1)
        grouped_dist = torch.norm(points_resi, dim=-1, keepdim=True)
        grouped_features = knn_gather(features2.permute(0,2,1), nn_idx)
        new_features = torch.cat([points_resi, grouped_dist], dim=-1)

        return new_features.permute(0,3,1,2).contiguous(),\
            nn.permute(0,3,1,2).contiguous(),\
            grouped_features.permute(0,3,1,2).contiguous()

    def forward(self, points1, points2, features1, features2, k, t):
        '''
        Input:
            points1: [B,3,N]
            points2: [B,3,N]
            features1: [B,C,N] (only for inference of additional features)
            features2: [B,C,N] (only for inference of additional features)
            k: int, number of kNN cluster
            t: [B], time step in (0,1)
        Output:
            fused_points: [B,3+C,N]
        '''
        N = points1.shape[-1]
        B = points1.shape[0] # batch size

        new_features_list = []
        new_grouped_points_list = []
        new_grouped_features_list = []

        for i in range(B):
            t1 = t[i]
            new_points1 = points1[i:i+1,:,:]
            new_points2 = points2[i:i+1,:,:]
            new_features1 = features1[i:i+1,:,:]
            new_features2 = features2[i:i+1,:,:]

            N2 = int(N*t1)
            N1 = N - N2

            k2 = int(k*t1)
            k1 = k - k2

            randidx1 = torch.randperm(N)[:N1]
            randidx2 = torch.randperm(N)[:N2]
            new_points = torch.cat((new_points1[:,:,randidx1], new_points2[:,:,randidx2]), dim=-1)

            new_features1, grouped_points1, grouped_features1 = self.knn_group(new_points, new_points1, new_features1, k1)
            new_features2, grouped_points2, grouped_features2 = self.knn_group(new_points, new_points2, new_features2, k2)

            new_features = torch.cat((new_features1, new_features2), dim=-1)
            new_grouped_points = torch.cat((grouped_points1, grouped_points2), dim=-1)
            new_grouped_features = torch.cat((grouped_features1, grouped_features2), dim=-1)

            new_features_list.append(new_features)
            new_grouped_points_list.append(new_grouped_points)
            new_grouped_features_list.append(new_grouped_features)

        new_features = torch.cat(new_features_list, dim=0)
        new_grouped_points = torch.cat(new_grouped_points_list, dim=0)
        new_grouped_features = torch.cat(new_grouped_features_list, dim=0)

        new_features = self.conv(new_features)
        new_features = torch.max(new_features, dim=1, keepdim=False)[0]
        weights = F.softmax(new_features, dim=-1)

        C = features1.shape[1]
        weights = weights.unsqueeze(1).repeat(1,3+C,1,1)
        fused_points = torch.cat([new_grouped_points, new_grouped_features], dim=1)
        fused_points = torch.sum(torch.mul(weights, fused_points), dim=-1, keepdim=False)

        return fused_points

def knn_group_withI(points1, points2, intensity2, k):
    '''
    Input:
        points1: [B,3,N]
        points2: [B,3,N]
        intensity2: [B,1,N]
    '''
    points1 = points1.permute(0,2,1).contiguous()
    points2 = points2.permute(0,2,1).contiguous()
    _, nn_idx, nn = knn_points(points1, points2, K=k, return_nn=True)
    points_resi = nn - points1.unsqueeze(2).repeat(1,1,k,1) # [B,M,k,3]
    grouped_dist = torch.norm(points_resi, dim=-1, keepdim=True)
    grouped_features = knn_gather(intensity2.permute(0,2,1), nn_idx) # [B,M,k,1]
    new_features = torch.cat([points_resi, grouped_dist], dim=-1)

    # [B,5,M,k], [B,3,M,k], [B,1,M,k]
    return new_features.permute(0,3,1,2).contiguous(), \
        nn.permute(0,3,1,2).contiguous(), \
        grouped_features.permute(0,3,1,2).contiguous()