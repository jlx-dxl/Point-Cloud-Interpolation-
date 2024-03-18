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
from Utils.Layers import SetConv, FlowEmbedding, SetUpConv, FeaturePropagation, PointsFusion, Tnet, TransformerLayer, \
    Pointnet2FeatureAbstract, Outputer
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


class ISAPCInet(nn.Module):
    def __init__(self, field, ff_out_c=128, tr_out_c=128, freeze=0):
        super(ISAPCInet, self).__init__()
        self.flow = FlowNet3D()
        if freeze == 1:
            for p in self.parameters():
                p.requires_grad = False

        self.field = field
        self.tr_out_c = tr_out_c

        self.tnet_forward = Tnet(field)  # 用于将一维的t标量转换为[B,3*2*field,N]的时间权重系数
        self.tnet_backward = Tnet(field)  # 用于将一维的t标量转换为[B,3*2*field,N]的时间权重系数

        self.ffab = Pointnet2FeatureAbstract(ff_out_c)  # 输入flow，输出提取到的flow的features，因此叫flow_feature_abstract(ffab)

        self.flow_tr_forward = TransformerLayer(ff_out_c, tr_out_c, 16)
        self.flow_tr_backward = TransformerLayer(ff_out_c, tr_out_c, 16)

        self.outputer = Outputer(2 * field * tr_out_c)

        self.fusion = PointsFusion([64, 64, 128])

    def forward(self, forward_pcds, key_pcds, backward_pcds, t, ini_feature, k=32):
        '''
        Input:
            forward_pcds: list of frames before the key frame: field * [B,3,N]
            key_pcds: list of front and rear key frame: 2 * [B,3,N]
            backward_pcds: list of frames after the key frame: field * [B,3,N]
            t: times stamp [0~1]
            ini_feature: input features for Flownet3D: zeros[B,3,N]
        '''

        B, C, N = ini_feature.shape
        # print("B,C,N:", B, C, N)
        tensor_t = t.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [B,1,1,1]

        flow_forward_list = []
        flow_backward_list = []

        for i in reversed(range(1, self.field + 1)):
            # with torch.no_grad():

            flow_forward = self.flow(forward_pcds[i - 1], key_pcds[0], ini_feature, ini_feature) / i
            # print("flow_forward:", flow_forward.shape)  # [B,3,N]
            flow_forward_list.append(flow_forward)
            flow_backward = self.flow(backward_pcds[i - 1], key_pcds[1], ini_feature, ini_feature) / i  # 时间归一化
            # print("flow_backward:", flow_backward.shape)  # [B,3,N]
            flow_backward_list.append(flow_backward)

        flow_forward_list.append(self.flow(key_pcds[0], key_pcds[1], ini_feature, ini_feature))
        flow_backward_list.append(self.flow(key_pcds[1], key_pcds[0], ini_feature, ini_feature))

        for i in range(1, self.field):
            flow_forward = self.flow(key_pcds[0], backward_pcds[i - 1],  ini_feature, ini_feature) / (i + 1)
            # print("flow_forward:", flow_forward.shape)  # [B,3,N]
            flow_forward_list.append(flow_forward)
            flow_backward = self.flow(key_pcds[1], forward_pcds[i - 1], ini_feature, ini_feature) / (i + 1)  # 时间归一化
            # print("flow_backward:", flow_backward.shape)  # [B,3,N]
            flow_backward_list.append(flow_backward)

        # print("len(flow_forward_list):", len(flow_forward_list))  # field+1
        # print("len(flow_backward_list):", len(flow_backward_list))  # field+1

        flow_forward_all = torch.stack(flow_forward_list, dim=1)
        # print("flow_forward_all:", flow_forward_all.shape)  # [B,2*field,3,N]
        flow_backward_all = torch.stack(flow_backward_list, dim=1)
        # print("flow_backward_all:", flow_backward_all.shape)  # [B,2*field,3,N]

        forward_weights = self.tnet_forward(tensor_t.to(torch.float32))  # [B,2*field,1,1]
        # print("forward_weights:", forward_weights.shape)
        backward_weights = self.tnet_backward(tensor_t.to(torch.float32))  # [B,2*field,1,1]
        # print("backward_weights:", backward_weights.shape)

        weighted_flow_forward_all = torch.mul(flow_forward_all, forward_weights).view(B,3,2*self.field*N)
        # print("weighted_flow_forward_all:", weighted_flow_forward_all.shape)  # [B,3,2*field*N]
        weighted_flow_backward_all = torch.mul(flow_backward_all, backward_weights).view(B,3,2*self.field*N)
        # print("weighted_flow_backward_all:", weighted_flow_backward_all.shape)  # [B,3,2*field*N]
        
        
        forwardff = self.ffab(weighted_flow_forward_all)
        # print("forwardffs_all:", forwardffs_all.shape)  # [B,ff_out_c,2*field*N]
        backwardff = self.ffab(weighted_flow_backward_all)
        # print("backwardff:", backwardff.shape)  # [B,ff_out_c,2*field*N]

        flow_forward_all0 = torch.cat(flow_forward_list, dim=-1)
        # print("flow_forward_all:", flow_forward_all.shape)  # [B,3,2*field*N]
        flow_backward_all0 = torch.cat(flow_backward_list, dim=-1)
        # print("flow_backward_all:", flow_backward_all.shape)  # [B,3,2*field*N]

        forwardflow0, attn_forward = self.flow_tr_forward(flow_forward_all0, forwardff)
        # print("forwardflow0:", forwardflow0.shape, "attn_forward:", attn_forward.shape)  # [B,tr_out_c,2*field*N]
        backwardflow0, attn_backward = self.flow_tr_backward(flow_backward_all0, backwardff)
        # print("backwardflow0:", backwardflow0.shape, "attn_backward:", attn_backward.shape)  # [B,tr_out_c,2*field*N]

        forwardflow = self.outputer(forwardflow0.view(B, 2 * self.tr_out_c * self.field, N))
        # print("forwardflow:", forwardflow.shape)  # [B,3,N]
        backwardflow = self.outputer(backwardflow0.view(B, 2 * self.tr_out_c * self.field, N))
        # print("backwardflow:", backwardflow.shape)  # [B,3,N]

        warped_xyz_forward = key_pcds[0] + forwardflow * t.unsqueeze(1).unsqueeze(1)
        warped_xyz_backward = key_pcds[1] + backwardflow * (1 - t).unsqueeze(1).unsqueeze(1)

        fused_points = self.fusion(warped_xyz_forward.to(torch.float32), warped_xyz_backward.to(torch.float32), N, k, t)

        return fused_points


if __name__ == '__main__':
    field = 1
    npoints = 16000
    batch_size=2

    dataset = NuscenesDataset(root='../Dataset/Subsets/Subset_01/LIDAR_TOP/',
                              scenes_list='../Dataset/Subsets/Subset_01/try_list.txt',
                              scene_split_lib='../Dataset/scene-split/', npoints=npoints, field=field)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # visualizer = PcdsVisualizer(if_save=False, if_show=True,
    #                             view_point_json_file=None, point_size=1.0)

    net = ISAPCInet(field=field, freeze=1).cuda()
    net.flow.load_state_dict(torch.load('./pretrain_models/flownet3d_kitti_odometry_maxbias1.pth'))

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

#         color1 = [[1, 0, 0], [0.8, 0, 0.2]]
#         for pcd, c in zip(forward_pcds, color1):
#             pcd_o3d = visualizer.convert_to_o3d_from_tensor(pcd.permute(0, 2, 1).squeeze(0))
#             visualizer.add_to_vis(pcd_o3d, c)

#         color2 = [[0.6, 0, 0.4], [0.4, 0, 0.6]]
#         for pcd, c in zip(key_pcds, color2):
#             pcd_o3d = visualizer.convert_to_o3d_from_tensor(pcd.permute(0, 2, 1).squeeze(0))
#             visualizer.add_to_vis(pcd_o3d, c)

#         color3 = [[0.2, 0, 0.8], [0, 0, 1]]
#         for pcd, c in zip(backward_pcds, color3):
#             pcd_o3d = visualizer.convert_to_o3d_from_tensor(pcd.permute(0, 2, 1).squeeze(0))
#             visualizer.add_to_vis(pcd_o3d, c)

        # # 检查多个fused_points用
        # color4 = [[0, 0.3, 0], [0, 0.7, 0], [0, 1, 0]]
        # for pcd, c in zip(fused_points_list, color4):
        #     pcd_o3d = visualizer.convert_to_o3d_from_tensor(pcd.permute(0, 2, 1).squeeze(0))
        #     visualizer.add_to_vis(pcd_o3d, c)

#         result_o3d = visualizer.convert_to_o3d_from_tensor(result.permute(0, 2, 1).squeeze(0))
#         visualizer.add_to_vis(result_o3d, [0, 1, 0])

#         gt_o3d = visualizer.convert_to_o3d_from_tensor(gt.permute(0, 2, 1).squeeze(0))
#         visualizer.add_to_vis(gt_o3d, [1, 1, 1])

        print(chamfer_loss(gt, result))
        # visualizer.show_and_save(None)
