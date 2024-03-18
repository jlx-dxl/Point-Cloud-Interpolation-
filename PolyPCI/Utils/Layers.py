import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

sys.path.append("..")
from Utils.Pointnet2Utils import farthest_point_sample, index_points, square_distance, query_ball_point, \
    PointNetSetAbstractionMsg, PointNetFeaturePropagation
from pytorch3d.ops import knn_points, knn_gather


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


class PointsFusion(nn.Module):
    def __init__(self, out_channels, in_channels=4):
        super(PointsFusion, self).__init__()

        layers = []
        out_channels = [in_channels, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True),
                       nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]

        self.conv = nn.Sequential(*layers)

    def knn_group(self, points1, points2, k):
        '''
        For each point in points1, query kNN points/features in points2/features2
        Input:
            points1: [B,3,N]
            points2: [B,3,N]
            features2: [B,C,N]
        Output:
            new_features: [B,4,N,k]
            nn: [B,3,N,k]
        '''
        points1 = points1.permute(0, 2, 1).contiguous()
        points2 = points2.permute(0, 2, 1).contiguous()
        _, nn_idx, nn = knn_points(points1, points2, K=k, return_nn=True)
        points_resi = nn - points1.unsqueeze(2).repeat(1, 1, k, 1)  # [B,N,k,3]，所有点均减去聚类中心
        grouped_dist = torch.norm(points_resi, dim=-1, keepdim=True)  # [B,N,k,1]，算出相对距离
        new_features = torch.cat([points_resi, grouped_dist], dim=-1)  # 将相对距离附加为一个特征维度

        return new_features.permute(0, 3, 1, 2).contiguous(), \
               nn.permute(0, 3, 1, 2).contiguous()

    def forward(self, points1, points2, N_out, k, t):
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
        B = points1.shape[0]  # batch size

        new_features_list = []
        new_grouped_points_list = []

        for i in range(B):
            t1 = t[i]
            N2 = int(N_out * t1)
            N1 = N_out - N2
            sampler1 = Sample(N1)
            sampler2 = Sample(N2)

            k2 = int(k * t1)
            k1 = k - k2
            
            new_points1 = sampler1(points1[i:i + 1, :, :])  # [1,3,N1]
            new_points2 = sampler2(points2[i:i + 1, :, :])  # [1,3,N2]

            # randidx1 = torch.randperm(N)[:N1]  # 随机取N1个点
            # randidx2 = torch.randperm(N)[:N2]  # 随机取N2个点
            
            new_points = torch.cat((new_points1, new_points2), dim=-1)  # 组成点云，点数为N

            new_features1, grouped_points1 = self.knn_group(new_points, new_points1, k1)  # [1,4,N,k],[1,3,N,k]
            new_features2, grouped_points2 = self.knn_group(new_points, new_points2, k2)

            new_features = torch.cat((new_features1, new_features2), dim=-1)  # [1,4,N,2k]
            new_grouped_points = torch.cat((grouped_points1, grouped_points2), dim=-1)  # [1,3,N,2k]

            new_features_list.append(new_features)
            new_grouped_points_list.append(new_grouped_points)

        new_features = torch.cat(new_features_list, dim=0)  # [B,4,N,2k]
        new_grouped_points = torch.cat(new_grouped_points_list, dim=0)  # [B,3,N,2k]

        new_features = self.conv(new_features)  # [B,?,N,2k]
        new_features = torch.max(new_features, dim=1, keepdim=False)[0]  # [B,N,2k]
        weights = F.softmax(new_features, dim=-1)  # [B,N,2k]

        weights = weights.unsqueeze(1).repeat(1, 3, 1, 1)  # [B,3,N,2k]
        fused_points = torch.sum(torch.mul(weights, new_grouped_points), dim=-1, keepdim=False)  # [B,3,N]

        return fused_points


class PointsFusion2(nn.Module):
    def __init__(self, out_channels, in_channels=4):
        super(PointsFusion2, self).__init__()

        layers = []
        out_channels = [in_channels, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True),
                       nn.GroupNorm(int(out_channels[i] / 8), out_channels[i]), nn.ReLU()]

        self.conv = nn.Sequential(*layers)

    def knn_group(self, points1, points2, k):
        '''
        For each point in points1, query kNN points/features in points2/features2
        Input:
            points1: [B,3,N]
            points2: [B,3,N]
            features2: [B,C,N]
        Output:
            new_features: [B,4,N,k]
            nn: [B,3,N,k]
        '''
        points1 = points1.permute(0, 2, 1).contiguous()
        points2 = points2.permute(0, 2, 1).contiguous()
        _, nn_idx, nn = knn_points(points1, points2, K=k, return_nn=True)
        points_resi = nn - points1.unsqueeze(2).repeat(1, 1, k, 1)  # [B,N,k,3]，所有点均减去聚类中心
        grouped_dist = torch.norm(points_resi, dim=-1, keepdim=True)  # [B,N,k,1]，算出相对距离
        new_features = torch.cat([points_resi, grouped_dist], dim=-1)  # 将相对距离附加为一个特征维度

        return new_features.permute(0, 3, 1, 2).contiguous(), \
               nn.permute(0, 3, 1, 2).contiguous()

    def forward(self, points_list, k, weighted_t):
        '''
        Input:
            points_list: (field + 1) * [B,3,N]
            k: int, number of kNN cluster
            t: [B], time step in (0,1)
        Output:
            fused_points: [B,3,N]
        '''
        N = points_list[0].shape[-1]
        B = points_list[0].shape[0]  # batch size

        new_features_list = []
        new_grouped_points_list = []

        for i in range(B):
            weights = weighted_t[i]
            n_sum = 0
            k_sum = 0
            new_point_list = []
            k_list = []
            new_feature_list = []
            new_grouped_point_list = []
            for j, pts in enumerate(points_list):
                pts = pts[i:i + 1, :, :]  # [1,3,N]
                if j < len(points_list) - 1:
                    N0 = int(N * weights[j])
                    k0 = int(k * weights[j])
                    k_sum += k0
                    k_list.append(k0)
                    n_sum += N0
                else:
                    N0 = N - n_sum
                    k0 = k - k_sum
                    k_list.append(k0)
                randidx0 = torch.randperm(N)[:N0]
                new_point_list.append(pts[:, :, randidx0])

            new_points = torch.cat(new_point_list, dim=-1)  # 组成点云，点数为N

            for j, pts in enumerate(points_list):
                pts = pts[i:i + 1, :, :]  # [1,3,N]
                new_feature, grouped_points = self.knn_group(new_points, pts, k_list[j])  # [1,4,N,k],[1,3,N,k]
                new_feature_list.append(new_feature)
                new_grouped_point_list.append(grouped_points)

            new_features = torch.cat(new_feature_list, dim=-1)  # [1,4,N,(field+1)*k]
            new_grouped_points = torch.cat(new_grouped_point_list, dim=-1)  # [1,3,N,(field+1)*k]

            new_features_list.append(new_features)
            new_grouped_points_list.append(new_grouped_points)

        new_features = torch.cat(new_features_list, dim=0)  # [B,4,N,(field+1)*k]
        new_grouped_points = torch.cat(new_grouped_points_list, dim=0)  # [B,3,N,(field+1)*k]

        new_features = self.conv(new_features)  # [B,?,N,2k]
        new_features = torch.max(new_features, dim=1, keepdim=False)[0]  # [B,N,2k]
        weights = F.softmax(new_features, dim=-1)  # [B,N,2k]

        weights = weights.unsqueeze(1).repeat(1, 3, 1, 1)  # [B,3,N,2k]
        fused_points = torch.sum(torch.mul(weights, new_grouped_points), dim=-1, keepdim=False)  # [B,3,N]

        return fused_points


def knn_group_withI(points1, points2, intensity2, k):
    '''
    Input:
        points1: [B,3,N]
        points2: [B,3,N]
        intensity2: [B,1,N]
    '''
    points1 = points1.permute(0, 2, 1).contiguous()
    points2 = points2.permute(0, 2, 1).contiguous()
    _, nn_idx, nn = knn_points(points1, points2, K=k, return_nn=True)
    points_resi = nn - points1.unsqueeze(2).repeat(1, 1, k, 1)  # [B,M,k,3]
    grouped_dist = torch.norm(points_resi, dim=-1, keepdim=True)
    grouped_features = knn_gather(intensity2.permute(0, 2, 1), nn_idx)  # [B,M,k,1]
    new_features = torch.cat([points_resi, grouped_dist], dim=-1)

    # [B,5,M,k], [B,3,M,k], [B,1,M,k]
    return new_features.permute(0, 3, 1, 2).contiguous(), \
           nn.permute(0, 3, 1, 2).contiguous(), \
           grouped_features.permute(0, 3, 1, 2).contiguous()


class TransformerLayer(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super(TransformerLayer, self).__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        xyz = xyz.permute(0, 2, 1).contiguous()
        features = features.permute(0, 2, 1).contiguous()

        _, knn_idx, knn_xyz = knn_points(xyz, xyz, K=self.k, return_nn=True)

        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), knn_gather(self.w_ks(x), knn_idx), knn_gather(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res.permute(0, 2, 1).contiguous(), attn.permute(0, 3, 1, 2).contiguous()


class Wnet(nn.Module):

    def __init__(self, field):
        super(Wnet, self).__init__()
        self.tnet = nn.Sequential(
            nn.Conv1d(1, 128, 1, bias=True),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.Conv1d(128, 512, 1, bias=True),
            nn.GroupNorm(64, 512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1, bias=True),
            nn.GroupNorm(64, 512),
            nn.ReLU(),
            nn.Conv1d(512, 128, 1, bias=True),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.Conv1d(128, 6 * field, 1, bias=True)
        )

    def forward(self, t):  # [B,1,1]
        weights = self.tnet(t)
        weights = F.softmax(weights, dim=1)
        return weights


class Tnet(nn.Module):

    def __init__(self, field):
        super(Tnet, self).__init__()
        self.tnet = nn.Sequential(
            nn.Conv2d(1, 64, 1, bias=True),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 256, 1, bias=True),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1, bias=True),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            nn.Conv2d(256, 64, 1, bias=True),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, int(2 * field), 1, bias=True)
        )

    def forward(self, t):  # [B,1,1]
        weights = self.tnet(t)
        weights = F.softmax(weights, dim=1)
        return weights


class Pointnet2FeatureAbstract(nn.Module):

    def __init__(self, ff_out_c):
        super(Pointnet2FeatureAbstract, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.1, 0.2], [16, 32], 0, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.2, 0.4], [16, 32], 32 + 64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.4, 0.8], [16, 32], 128 + 128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.8, 1.6], [16, 32], 256 + 256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, ff_out_c, 1)
        self.gn1 = nn.GroupNorm(8, ff_out_c)

    def forward(self, xyz):
        l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        result = F.relu(self.gn1(self.conv1(l0_points)))

        return result


class Outputer(nn.Module):

    def __init__(self, in_c):
        super(Outputer, self).__init__()
        self.outputer = nn.Sequential(
            nn.Conv1d(in_c, 128, 1, bias=True),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.Conv1d(128, 32, 1, bias=True),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Conv1d(32, 3, 1, bias=True)
        )

    def forward(self, features):
        result = self.outputer(features)
        return result


