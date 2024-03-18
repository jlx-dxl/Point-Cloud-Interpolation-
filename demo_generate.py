import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import open3d as o3d
import argparse
import os

from Dataset.InterpolationData import NuscenesDataset
from Models.New_Models_field_0 import ISAPCInet
from Utils.Visualize import PcdsVisualizer
from Utils.Utils import chamfer_loss

def parse_args():
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--field', type=int, default=0)
    parser.add_argument('--npoints', type=int, default=16000)
    parser.add_argument('--interval', type=int, default=5)
    parser.add_argument('--root', type=str, default='./Demos/20230508test/demo_data/')
    parser.add_argument('--save_dir', type=str, default='./Demos/20230508test/demo_data/field_0/')
    parser.add_argument('--pretrained_flow_model', type=str,
                        default='./Models/pretrain_models/flownet3d_kitti_odometry_maxbias1.pth')
    parser.add_argument('--pretrained_self_model', type=str, default='./Models/result_models/field_0_interval_5_freeze_1_npoint_16000/1scene/field_0_0.4822184388574801.pth')

    return parser.parse_args()

def fps(scan,npoints):
    pcd = o3d.geometry.PointCloud()  # 传入3d点云格式
    pcd.points = o3d.utility.Vector3dVector(scan)  # 转换格式
    pcd_down = pcd.farthest_point_down_sample(npoints)  # FPS降采样
    return np.asarray(pcd_down.points)

if __name__ == '__main__':
    args = parse_args()
    # 读数据
    forward_pcds = []
    backward_pcds = []
    key_pcds = []
    files = os.listdir(args.root)

    for i in range(1,args.field+1):
        for file in files:
            if file == 'forward_' + str(i) + '.bin':
                filename = os.path.join(args.root, file)
                print(filename)
                points = np.fromfile(filename, dtype=np.float32).reshape(-1, 5)[:, :3]
                points = fps(points,args.npoints)
                points = torch.from_numpy(points).t().to(torch.float32).unsqueeze(0).cuda(non_blocking=True)
                print(points.shape)
                forward_pcds.append(points)
            if file == 'backward_' + str(i) + '.bin':
                filename = os.path.join(args.root, file)
                print(filename)
                points = np.fromfile(filename, dtype=np.float32).reshape(-1, 5)[:, :3]
                points = fps(points,args.npoints)
                points = torch.from_numpy(points).t().to(torch.float32).unsqueeze(0).cuda(non_blocking=True)
                print(points.shape)
                backward_pcds.append(points)

    for i in range(1, 2 + 1):
        for file in files:
            if file == 'key_' + str(i) + '.bin':
                filename = os.path.join(args.root, file)
                print(filename)
                points = np.fromfile(filename, dtype=np.float32).reshape(-1, 5)[:, :3]
                points = fps(points,args.npoints)
                points = torch.from_numpy(points).t().to(torch.float32).unsqueeze(0).cuda(non_blocking=True)
                print(points.shape)
                key_pcds.append(points)

    ini_feature = torch.from_numpy(np.zeros([args.npoints, 3])).t().to(torch.float32).unsqueeze(0).cuda(non_blocking=True)

    t_list = [0.2,0.4,0.6,0.8]


    # 加载模型
    net = ISAPCInet(args.field, freeze=1).cuda()
    net.flow.load_state_dict(torch.load(args.pretrained_flow_model))
    net.load_state_dict(torch.load(args.pretrained_self_model))

    for time in tqdm(t_list):
        timet = torch.tensor(time).to(torch.float32).unsqueeze(0).cuda(non_blocking=True)
        result = net(forward_pcds, key_pcds, backward_pcds, timet, ini_feature)
        save_name = os.path.join(args.save_dir,'result_' + str(time)+'.bin')
        result.squeeze(0).permute(1,0).clone().detach().cpu().numpy().tofile(save_name)
        print("save result point clouds to:", save_name)

