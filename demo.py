import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import open3d as o3d

from Dataset.InterpolationData import NuscenesDataset
from Models.New_Models0 import ISAPCInet
from Utils.Visualize import PcdsVisualizer
from Utils.Utils import chamfer_loss

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--field', type=int, default=2)
    parser.add_argument('--npoints', type=int, default=3000)
    parser.add_argument('--interval', type=int, default=5)
    parser.add_argument('--root', type=str, default='../ISAPCI/Dataset/Subsets/Subset_10/LIDAR_TOP/')
    parser.add_argument('--demo_scenes_list', type=str, default='../ISAPCI/Dataset/Subsets/Subset_10/test_list.txt')
    parser.add_argument('--scene_split_lib', type=str, default='../ISAPCI/Dataset/scene-split/')
    parser.add_argument('--save_dir', type=str, default='./Demos/20230508test/field2/')
    parser.add_argument('--pretrained_flow_model', type=str,
                        default='./Models/pretrain_models/flownet3d_kitti_odometry_maxbias1.pth')
    parser.add_argument('--pretrained_self_model', type=str,
                        default='./Models/result_models/field_2_interval_5_freeze_1_npoint_16000/10scenes/field_2_0.6806827462858771.pth')
    parser.add_argument('--if_save', type=bool, default=True)
    parser.add_argument('--if_show', type=bool, default=False)
    parser.add_argument('--view_point_json_file', type=str, default='./ScreenCamera_2023-05-08-00-15-02.json')

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

    # 存图集
    visualizer = PcdsVisualizer(if_save=args.if_save, if_show=args.if_show, if_down_sample=False,
                                view_point_json_file=args.view_point_json_file, point_size=2.0)

    pbar = tqdm(enumerate(dataloader))
    for i, data in pbar:
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
        # loss = chamfer_loss(result, gt)
        # print("loss:", loss)

        # # 前瞻点
        # color1 = [[1, 0, 0], [0.8, 0, 0.2]]
        # for pcd, c in zip(forward_pcds, color1):
        #     pcd_o3d = visualizer.convert_to_o3d_from_tensor(pcd.permute(0,2,1).squeeze(0))
        #     visualizer.add_to_vis(pcd_o3d, c)

        # 前后关键点
        color2 = [[1,0,0], [0,0,1]]
        for pcd, c in zip(key_pcds, color2):
            pcd_o3d = visualizer.convert_to_o3d_from_tensor(pcd.permute(0,2,1).squeeze(0))
            visualizer.add_to_vis(pcd_o3d, c)

        # # 后瞻点
        # color3 = [[0.2, 0, 0.8], [0, 0, 1]]
        # for pcd, c in zip(backward_pcds, color3):
        #     pcd_o3d = visualizer.convert_to_o3d_from_tensor(pcd.permute(0,2,1).squeeze(0))
        #     visualizer.add_to_vis(pcd_o3d, c)

        # result绿色
        result_o3d = visualizer.convert_to_o3d_from_tensor(result.permute(0,2,1).squeeze(0))
        visualizer.add_to_vis(result_o3d, [0, 1, 0])

        # gt白色
        gt_o3d = visualizer.convert_to_o3d_from_tensor(gt.permute(0,2,1).squeeze(0))
        visualizer.add_to_vis(gt_o3d, [1, 1, 1])

        visualizer.show_and_save(args.save_dir+"scene1013_"+str(i)+".png")
        visualizer.clear()

