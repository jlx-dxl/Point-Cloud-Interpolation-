import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from tqdm import tqdm
import open3d as o3d
import argparse

from Dataset.InterpolationData import NuscenesDataset
from Models.New_Models_field_0 import ISAPCInet as Net0
from Models.New_Models_field_1 import ISAPCInet as Net1
from Models.New_Models_field_2_3 import ISAPCInet as Net23
from Utils.Visualize import PcdsVisualizer
from Utils.Utils import chamfer_loss


def parse_args():
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--field', type=int, default=3)
    parser.add_argument('--save_dir', type=str, default='./Demos/20230508test/field2/')
    parser.add_argument('--pretrained_flow_model', type=str,
                        default='./Models/pretrain_models/flownet3d_kitti_odometry_maxbias1.pth')
    parser.add_argument('--pretrained_self_model', type=str,
                        default='./Models/result_models/field_3_0.408451775849705.pth')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # 加载模型
    net = Net23(args.field, freeze=1).cuda()
    net.flow.load_state_dict(torch.load(args.pretrained_flow_model))
    net.load_state_dict(torch.load(args.pretrained_self_model))

    Tf = net.tnet_forward
    Tb = net.tnet_backward

    t_list = [0.2,0.4,0.6,0.8]
    weight_list = []
    Wf_list = []
    Wb_list = []
    for time in t_list:
        timet = torch.tensor(time).to(torch.float32).unsqueeze(0).cuda(non_blocking=True)
        # print(timet.shape)
        tensor_t = timet.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        # print(tensor_t.shape)
        Wf = np.array(Tf(tensor_t).squeeze(0).squeeze(1).squeeze(1).clone().detach().cpu())
        # print(Wf,Wf.shape)
        Wf_list.append(Wf)
        Wb = np.array(Tb(tensor_t).squeeze(0).squeeze(1).squeeze(1).clone().detach().cpu())
        # print(Wb, Wb.shape)
        Wb_list.append(Wb)

    Wf_list = np.asarray(Wf_list)
    Wf_list = pd.DataFrame(Wf_list,index=['0.2','0.4','0.6','0.8'])

    Wb_list = np.asarray(Wb_list)
    Wb_list = pd.DataFrame(Wb_list,index=['0.2','0.4','0.6','0.8'])

    plt.subplot(1,2,1)
    sns.heatmap(Wf_list,annot=True)
    plt.title('forward')
    plt.subplot(1,2,2)
    sns.heatmap(Wb_list,annot=True)
    plt.title('backward')

    plt.show()





