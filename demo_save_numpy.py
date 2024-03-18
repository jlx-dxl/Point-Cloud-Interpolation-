import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import open3d as o3d
import os

from Dataset.InterpolationData import NuscenesDataset
from Models.New_Models0 import ISAPCInet
from Utils.Visualize import PcdsVisualizer
from Utils.Utils import chamfer_loss

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--field', type=int, default=1)
    parser.add_argument('--npoints', type=int, default=16000)
    parser.add_argument('--interval', type=int, default=5)
    parser.add_argument('--root', type=str, default='./Dataset/Subsets/Subset_01/LIDAR_TOP/')
    parser.add_argument('--demo_scenes_list', type=str, default='./Dataset/Subsets/Subset_01/demo_list.txt')
    parser.add_argument('--scene_split_lib', type=str, default='./Dataset/scene-split/')
    parser.add_argument('--save_dir', type=str, default='./Demos/20230508test/field_1/')
    parser.add_argument('--pretrained_flow_model', type=str,
                        default='./Models/pretrain_models/flownet3d_kitti_odometry_maxbias1.pth')
    parser.add_argument('--pretrained_self_model', type=str, default='./Models/result_models/field_1_interval_5_freeze_1_npoint_16000/10scenes/field_1_0.6718219368047612.pth')
    parser.add_argument('--if_save', type=bool, default=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # 读数据
    dataset = NuscenesDataset(root=args.root, scenes_list=args.demo_scenes_list,
                              scene_split_lib=args.scene_split_lib, field=args.field, npoints=args.npoints,
                              interval=args.interval, if_random=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 加载模型
    net = ISAPCInet(args.field, freeze=1).cuda()
    net.flow.load_state_dict(torch.load(args.pretrained_flow_model))
    net.load_state_dict(torch.load(args.pretrained_self_model))

    pbar = tqdm(enumerate(dataloader))
    for i, data in pbar:
        for k in range(1,args.interval):
            forward_pcds, key_pcds, backward_pcds, t, gt, ini_feature = data
            # 移动到cuda
            for j in range(args.field):
                forward_pcds[j] = forward_pcds[j].cuda(non_blocking=True)
                backward_pcds[j] = backward_pcds[j].cuda(non_blocking=True)
            for j in range(2):
                key_pcds[j] = key_pcds[j].cuda(non_blocking=True)
            t = t.cuda(non_blocking=True)
            ini_feature = ini_feature.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)

            result = net(forward_pcds, key_pcds, backward_pcds, t, ini_feature)
            
            if args.if_save == True:
                if k == 1 :
                    save_name = os.path.join(args.save_dir,'scene-0101-' + str(i)+'-'+str(k*0.2)+'-forward.bin')
                    forward_pcds[0].squeeze(0).permute(1,0).cpu().numpy().tofile(save_name)
                    print("save forward point clouds to:", save_name)
                    
                    save_name = os.path.join(args.save_dir,'scene-0101-' + str(i)+'-'+str(k*0.2)+'-backward.bin')
                    backward_pcds[0].squeeze(0).permute(1,0).cpu().numpy().tofile(save_name)
                    print("save backward point clouds to:", save_name)
                    
                save_name = os.path.join(args.save_dir,'scene-0101-' + str(i)+'-'+str(k*0.2)+'-result.bin')
                result.squeeze(0).permute(1,0).clone().detach().cpu().numpy().tofile(save_name)
                print("save result point clouds to:", save_name)

                save_name = os.path.join(args.save_dir,'scene-0101-' + str(i)+'-'+str(k*0.2)+'-gt.bin')
                gt.squeeze(0).permute(1,0).cpu().numpy().tofile(save_name)
                print("save gt point clouds to:", save_name)
                

