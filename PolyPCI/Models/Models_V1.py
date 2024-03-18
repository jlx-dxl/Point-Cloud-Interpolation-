import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
import os
from pytorch3d.ops import knn_points

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.path.append("..")
from Utils.Layers import SetConv, FlowEmbedding, SetUpConv, FeaturePropagation
from Dataset.Dataset import NuscenesDataset
from Utils.Visualize import PcdsVisualizer
from Utils.Utils import chamfer_loss


class FlowNet3D(nn.Module):
    '''
    Implementation of FlowNet3D (CVPR 2019) in PyTorch
    We refer to original Tensorflow implementation (https://github.com/xingyul/flownet3d)
    and open source PyTorch implementation (https://github.com/multimodallearning/flownet3d.pytorch)
    to implement the code for FlowNet3D.
    '''

    def __init__(self):
        super(FlowNet3D, self).__init__()

        self.set_conv1 = SetConv(1024, 0.5, 16, 3, [32, 32, 64])
        self.set_conv2 = SetConv(256, 1.0, 16, 64, [64, 64, 128])
        self.flow_embedding = FlowEmbedding(64, 128, [128, 128, 128])
        self.set_conv3 = SetConv(64, 2.0, 8, 128, [128, 128, 256])
        self.set_conv4 = SetConv(16, 4.0, 8, 256, [256, 256, 512])
        self.set_upconv1 = SetUpConv(8, 512, 256, [], [256, 256])
        self.set_upconv2 = SetUpConv(8, 256, 256, [128, 128, 256], [256])
        self.set_upconv3 = SetUpConv(8, 256, 64, [128, 128, 256], [256])
        self.fp = FeaturePropagation(256, 3, [256, 256])
        self.classifier = nn.Sequential(
            nn.Conv1d(256, 128, 1, bias=True),
            nn.BatchNorm1d(128, eps=0.001),
            nn.ReLU(),
            nn.Conv1d(128, 3, 1, bias=True)
        )

    def forward(self, points1, points2, features1, features2):
        '''
        Input:
            points1: [B,3,N]
            points2: [B,3,N]
            features1: [B,3,N] (colors for Flythings3D and zero for LiDAR)
            features2: [B,3,N] (colors for Flythings3D and zero for LiDAR)
        Output:
            flow: [B,3,N]
        '''
        points1_1, features1_1 = self.set_conv1(points1, features1)
        # print("set_conv1_1:", points1_1.shape, features1_1.shape)
        points1_2, features1_2 = self.set_conv2(points1_1, features1_1)
        # print("set_conv1_2:", points1_2.shape, features1_2.shape)

        points2_1, features2_1 = self.set_conv1(points2, features2)
        # print("set_conv2_1:", points2_1.shape, features2_1.shape)
        points2_2, features2_2 = self.set_conv2(points2_1, features2_1)
        # print("set_conv2_2:", points2_2.shape, features2_2.shape)

        embedding = self.flow_embedding(points1_2, points2_2, features1_2, features2_2)
        # print("embedding:", embedding.shape)

        points1_3, features1_3 = self.set_conv3(points1_2, embedding)
        # print("set_conv1_3:", points1_3.shape, features1_3.shape)
        points1_4, features1_4 = self.set_conv4(points1_3, features1_3)
        # print("set_conv1_4:", points1_4.shape, features1_4.shape)

        new_features1_3 = self.set_upconv1(points1_4, points1_3, features1_4, features1_3)
        # print("set_upconv1:", new_features1_3.shape)
        new_features1_2 = self.set_upconv2(points1_3, points1_2, new_features1_3,
                                           torch.cat([features1_2, embedding], dim=1))
        # print("set_upconv2:", new_features1_2.shape)
        new_features1_1 = self.set_upconv3(points1_2, points1_1, new_features1_2, features1_1)
        # print("set_upconv3:", new_features1_1.shape)
        new_features1 = self.fp(points1_1, points1, new_features1_1, features1)
        # print("fp:", new_features1.shape)

        flow = self.classifier(new_features1)

        return flow


class PolyPCI(nn.Module):
    def __init__(self, field, degree, freeze=1):
        super(PolyPCI, self).__init__()
        self.flow = FlowNet3D()
        if freeze == 1:
            for p in self.parameters():
                p.requires_grad = False
        self.degree = degree
        self.field = field

    def rebuild(self, ref_pcd, pcd, k=1):
        '''
        For each point in points1, query 1NN(the nearest) points in points2
        Input:
            ref_pcd: [B,3,N]
            pcd: [B,3,N]
        Output:
            nn: [B,3,N,k]
        '''
        ref_pcd = ref_pcd.permute(0, 2, 1).contiguous()
        pcd = pcd.permute(0, 2, 1).contiguous()
        _, nn_idx, nn = knn_points(ref_pcd, pcd, K=k, return_nn=True)
        return nn.permute(0, 3, 1, 2).contiguous()

    def fitting_and_predict(self, x, y, t):
        coefficients = np.polyfit(x, y, self.degree)
        # print("coefficients:", coefficients.shape)
        poly = PolynomialFeatures(degree=self.degree)
        X_poly = poly.fit_transform(np.array(t.cpu()).reshape(-1, 1))
        X_poly = np.flip(X_poly, axis=1)
        # print("X_poly:", X_poly.shape)
        value = np.matmul(X_poly, coefficients)
        return value

    def forward(self, forward_pcds, key_pcd, backward_pcds, t, T_list, ini_feature):
        '''
        Input:
            forward_pcds: list of frames before the key frame: field * [B,3,N]
            key_pcd: key frame: [B,3,N]
            backward_pcds: list of frames after the key frame: field * [B,3,N]
            t: times stamp [-field~field]
            ini_feature: input features for Flownet3D: zeros[B,3,N]
        '''

        B, C, N = ini_feature.shape
        # print("B,C,N:", B, C, N)
        xs = []
        ys = []
        zs = []

        xs.append(key_pcd[:, 0, :])
        ys.append(key_pcd[:, 1, :])
        zs.append(key_pcd[:, 2, :])

        for i in range(self.field):
            if i == 0:
                flow_forward = self.flow(key_pcd, forward_pcds[i], ini_feature, ini_feature)
                print("flow_forward:", flow_forward.shape)
                flow_backward = self.flow(key_pcd, backward_pcds[i], ini_feature, ini_feature)
                print("flow_backward:", flow_backward.shape)

                forward_wrapped = flow_forward + key_pcd
                print("forward_wrapped:", forward_wrapped.shape)
                backward_wrapped = flow_backward + key_pcd
                print("backward_wrapped:", backward_wrapped.shape)

            else:
                flow_forward = self.flow(forward_rebuilt, forward_pcds[i], ini_feature, ini_feature)
                print("flow_forward:", flow_forward.shape)
                flow_backward = self.flow(backward_rebuilt, backward_pcds[i], ini_feature, ini_feature)
                print("flow_backward:", flow_backward.shape)

                forward_wrapped = flow_forward + forward_rebuilt
                print("forward_wrapped:", forward_wrapped.shape)
                backward_wrapped = flow_backward + backward_rebuilt
                print("backward_wrapped:", backward_wrapped.shape)

            forward_rebuilt = self.rebuild(forward_wrapped, forward_pcds[i]).squeeze(-1)
            print("forward_rebuilt:", forward_rebuilt.shape)
            backward_rebuilt = self.rebuild(backward_wrapped, backward_pcds[i]).squeeze(-1)
            print("backward_rebuilt:", backward_rebuilt.shape)

            xs.append(forward_rebuilt[:, 0, :])
            ys.append(forward_rebuilt[:, 1, :])
            zs.append(forward_rebuilt[:, 2, :])

            xs.append(backward_rebuilt[:, 0, :])
            ys.append(backward_rebuilt[:, 1, :])
            zs.append(backward_rebuilt[:, 2, :])

        xs_all = torch.stack(xs, dim=1)
        print("xs_all:", xs_all.shape)
        ys_all = torch.stack(ys, dim=1)
        print("ys_all:", ys_all.shape)
        zs_all = torch.stack(zs, dim=1)
        print("zs_all:", zs_all.shape)

        results = []

        for i in range(B):
            result = []
            T = np.array(T_list[i]).reshape(-1)
            print("T:", T, T.shape)
            print("t:", t[i])

            xs = np.array(xs_all.cpu()[i, :, :])
            print("xs:", xs.shape)
            x_results = self.fitting_and_predict(T, xs, t[i])
            print("x_results:", x_results.shape)
            result.append(torch.tensor(x_results).to(torch.float32))

            ys = np.array(ys_all.cpu()[i, :, :])
            print("ys:", ys.shape)
            y_results = self.fitting_and_predict(T, ys, t[i])
            print("y_results:", y_results.shape)
            result.append(torch.tensor(y_results).to(torch.float32))

            zs = np.array(zs_all.cpu()[i, :, :])
            print("zs:", zs.shape)
            z_results = self.fitting_and_predict(T, zs, t[i])
            print("z_results:", z_results.shape)
            result.append(torch.tensor(z_results).to(torch.float32))

            result = torch.cat(result, dim=0)
            print("result:", result.shape)
            results.append(result)

        results = torch.stack(results, dim=0).cuda(non_blocking=True).to(torch.float32)
        print("results:", results.shape)

        return results


if __name__ == '__main__':
    field = 3
    degree = 4
    npoints = 16000
    batch_size = 1

    dataset = NuscenesDataset(root='../../ISAPCI/Dataset/Subsets/Subset_01/LIDAR_TOP/',
                              scenes_list='../../ISAPCI/Dataset/Subsets/Subset_01/try_list.txt',
                              scene_split_lib='../../ISAPCI/Dataset/scene-split/', field=field, npoints=npoints)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    net = PolyPCI(field=field, degree=degree, freeze=1).cuda()
    net.flow.load_state_dict(torch.load('./pretrain_models/flownet3d_kitti_odometry_maxbias1.pth'))

    pbar = tqdm(enumerate(dataloader))

    # # 找+存视角
    #
    # visualizer = PcdsVisualizer(if_save=False, if_show=True,
    #                             view_point_json_file=None, point_size=1.0)

    for i, data in pbar:
        forward_pcds, key, backward_pcds, t, T_list, gt, ini_feature = data

        for j in range(field):
            forward_pcds[j] = forward_pcds[j].cuda(non_blocking=True)
            backward_pcds[j] = backward_pcds[j].cuda(non_blocking=True)

        key = key.cuda(non_blocking=True)
        t = t.cuda(non_blocking=True)
        ini_feature = ini_feature.cuda(non_blocking=True)
        gt = gt.cuda(non_blocking=True)

        result = net(forward_pcds, key, backward_pcds, t, T_list, ini_feature)
        print("chamfer_loss:", chamfer_loss(gt, result))

        # # # 时序：[紫，蓝，红，黄，绿]
        # # color1 = [[0, 0, 1], [1, 0, 1]]
        # # for pcd, c in zip(forward_pcds, color1):
        # #     pcd_o3d = visualizer.convert_to_o3d_from_tensor(pcd.permute(0, 2, 1).squeeze(0))
        # #     visualizer.add_to_vis(pcd_o3d, c)
        #
        # key_o3d = visualizer.convert_to_o3d_from_tensor(key.permute(0, 2, 1).squeeze(0))
        # visualizer.add_to_vis(key_o3d, [1, 0, 0])
        #
        # # color3 = [[1, 1, 0], [0, 1, 0]]
        # # for pcd, c in zip(backward_pcds, color3):
        # #     pcd_o3d = visualizer.convert_to_o3d_from_tensor(pcd.permute(0, 2, 1).squeeze(0))
        # #     visualizer.add_to_vis(pcd_o3d, c)
        #
        # result_o3d = visualizer.convert_to_o3d_from_tensor(result.permute(0, 2, 1).squeeze(0))
        # visualizer.add_to_vis(result_o3d, [0, 1, 1])   # 青色
        #
        # gt_o3d = visualizer.convert_to_o3d_from_tensor(gt.permute(0, 2, 1).squeeze(0))
        # visualizer.add_to_vis(gt_o3d, [1, 1, 1])   # 白
        #
        # visualizer.show_and_save(None)
