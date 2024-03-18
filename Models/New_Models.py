import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.path.append("..")
from Utils.Layers import Sample, SetConv, FlowEmbedding, SetUpConv, FeaturePropagation, PointsFusion, PointsFusion2, Wnet
from Dataset.InterpolationData import NuscenesDataset
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


class Tnet(nn.Module):
    '''
    Implementation of FlowNet3D (CVPR 2019) in PyTorch
    We refer to original Tensorflow implementation (https://github.com/xingyul/flownet3d)
    and open source PyTorch implementation (https://github.com/multimodallearning/flownet3d.pytorch)
    to implement the code for FlowNet3D.
    '''

    def __init__(self, forward_field, backward_field):
        super(Tnet, self).__init__()
        self.f = forward_field
        self.b = backward_field
        self.tnet = nn.Sequential(
            nn.Conv1d(3 * (forward_field + backward_field), 128, 1, bias=True),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1, bias=True),
            nn.GroupNorm(64, 1024),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1, bias=True),
            nn.GroupNorm(64, 1024),
            nn.ReLU(),
            nn.Conv1d(1024, 128, 1, bias=True),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv1d(128, 3 * (forward_field + backward_field), 1, bias=True)
        )

    def forward(self, origin_t):
        t = origin_t.unsqueeze(1).unsqueeze(1).repeat(1, 3 * (self.f + self.b), 1)
        result = self.tnet(t)
        result = F.softmax(result, dim=1)
        return result


class Outputer(nn.Module):
    '''
    Implementation of FlowNet3D (CVPR 2019) in PyTorch
    We refer to original Tensorflow implementation (https://github.com/xingyul/flownet3d)
    and open source PyTorch implementation (https://github.com/multimodallearning/flownet3d.pytorch)
    to implement the code for FlowNet3D.
    '''

    def __init__(self, forward_field, backward_field):
        super(Outputer, self).__init__()
        self.f = forward_field
        self.b = backward_field
        self.outputer = nn.Sequential(
            nn.Conv1d(3 * (forward_field + backward_field), 128, 1, bias=True),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1, bias=True),
            nn.GroupNorm(64, 1024),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1, bias=True),
            nn.GroupNorm(64, 1024),
            nn.ReLU(),
            nn.Conv1d(1024, 128, 1, bias=True),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv1d(128, 3, 1, bias=True)
        )

    def forward(self, features):
        result = self.outputer(features)
        return result


class ISAPCInet(nn.Module):
    def __init__(self, forward_field, backward_field, freeze=1):
        super(ISAPCInet, self).__init__()
        self.flow = FlowNet3D()
        if freeze == 1:
            for p in self.parameters():
                p.requires_grad = False

        self.f = forward_field
        self.b = backward_field
        self.tnet = Tnet(forward_field, backward_field)  # 用于将一维的t标量转换为[B,3*(f+b),N]的时间权重系数
        self.outputer = Outputer(forward_field, backward_field)

    def forward(self, forward_pcds, k, backward_pcds, t, ini_feature):
        '''
        Input:
            forward_pcds: list of frames before the key frame: forward_field * [B,3,N]
            k: key frame: [B,3,N]
            backward_pcds: list of frames after the key frame: backward_field * [B,3,N]
            t: times stamp [0~1]
            ini_feature: input features for Flownet3D: zeros[B,3,N]
        '''

        # Estimate 3D scene flow
        k = k.to(torch.float32).cuda(non_blocking=True)
        B, C, N = k.shape
        t = t.to(torch.float32).cuda(non_blocking=True)
        ini_feature = ini_feature.to(torch.float32).cuda(non_blocking=True)
        with torch.no_grad():
            flow_list = []
            for i, pcd in enumerate(forward_pcds):
                pcd = pcd.to(torch.float32).cuda(non_blocking=True)
                flow = self.flow(pcd, k, ini_feature, ini_feature)
                flow_list.append(flow)
                # print("forward flow", str(i), flow.shape)
            for i, pcd in enumerate(backward_pcds):
                pcd = pcd.to(torch.float32).cuda(non_blocking=True)
                flow = self.flow(k, pcd, ini_feature, ini_feature)
                flow_list.append(flow)
                # print("backward flow", str(i), flow.shape)

        input_flows = copy.copy(flow_list[0])
        for i in range(1, len(flow_list)):
            input_flows = torch.cat([input_flows, flow_list[i]], dim=1)
        # print("input_flows:", input_flows.shape)

        t_weights = self.tnet(t)
        # print("t_weights:", t_weights.shape)

        weighted_flows = input_flows.to(torch.float32).cuda(non_blocking=True) * t_weights
        # print("weighted_flows1:", weighted_flows.shape)

        res_flow = self.outputer(weighted_flows)
        # print("res_flow:", res_flow.shape)

        result_pcd = res_flow * t.unsqueeze(1).unsqueeze(1) + k
        # print("result_pcd:", result_pcd.shape)
        # return result
        return result_pcd


if __name__ == '__main__':
    field = 1
    npoints = 8000

    dataset = NuscenesDataset(root='../../ISAPCI/Dataset/Subsets/Subset_01/LIDAR_TOP/',
                              scenes_list='../../ISAPCI/Dataset/Subsets/Subset_01/try_list.txt',
                              scene_split_lib='../../ISAPCI/Dataset/scene-split/', npoints=npoints, field=field)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    visualizer = PcdsVisualizer(if_save=False, if_show=True,
                                view_point_json_file=None, point_size=1.0)

    net = PointINet2(field=field, freeze=1).cuda()
    net.flow.load_state_dict(torch.load('./pretrain_model/flownet3d_kitti_odometry_maxbias1.pth'))
    net.pointinet.load_state_dict(torch.load('./pretrain_model/interp_kitti.pth'))

    pbar = tqdm(enumerate(dataloader))

    # 找+存视角

    for i, data in pbar:
        forward_pcds, key_pcds, backward_pcds, t, gt, ini_feature = data

        for j in range(field):
            forward_pcds[j] = forward_pcds[j].cuda(non_blocking=True)
            backward_pcds[j] = backward_pcds[j].cuda(non_blocking=True)

        for j in range(2):
            key_pcds[j] = key_pcds[j].cuda(non_blocking=True)

        t = t.cuda(non_blocking=True)
        ini_feature = ini_feature.cuda(non_blocking=True)
        gt = gt.cuda(non_blocking=True)

        result = net(forward_pcds, key_pcds, backward_pcds, t, ini_feature)

        color1 = [[1, 0, 0], [0.8, 0, 0.2]]
        for pcd, c in zip(forward_pcds, color1):
            pcd_o3d = visualizer.convert_to_o3d_from_tensor(pcd.permute(0, 2, 1).squeeze(0))
            visualizer.add_to_vis(pcd_o3d, c)

        color2 = [[0.6, 0, 0.4], [0.4, 0, 0.6]]
        for pcd, c in zip(key_pcds, color2):
            pcd_o3d = visualizer.convert_to_o3d_from_tensor(pcd.permute(0, 2, 1).squeeze(0))
            visualizer.add_to_vis(pcd_o3d, c)

        color3 = [[0.2, 0, 0.8], [0, 0, 1]]
        for pcd, c in zip(backward_pcds, color3):
            pcd_o3d = visualizer.convert_to_o3d_from_tensor(pcd.permute(0, 2, 1).squeeze(0))
            visualizer.add_to_vis(pcd_o3d, c)

        # # 检查多个fused_points用
        # color4 = [[0, 0.3, 0], [0, 0.7, 0], [0, 1, 0]]
        # for pcd, c in zip(fused_points_list, color4):
        #     pcd_o3d = visualizer.convert_to_o3d_from_tensor(pcd.permute(0, 2, 1).squeeze(0))
        #     visualizer.add_to_vis(pcd_o3d, c)

        result_o3d = visualizer.convert_to_o3d_from_tensor(result.permute(0, 2, 1).squeeze(0))
        visualizer.add_to_vis(result_o3d, [0, 1, 0])

        gt_o3d = visualizer.convert_to_o3d_from_tensor(gt.permute(0, 2, 1).squeeze(0))
        visualizer.add_to_vis(gt_o3d, [1, 1, 1])

        print(chamfer_loss(gt, result))
        visualizer.show_and_save(None)
