import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import datetime
import argparse
from tqdm import tqdm
import wandb

from Dataset.InterpolationData import NuscenesDataset
from Models.New_Models0 import ISAPCInet
from Utils.Utils import chamfer_loss, EMD



def parse_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--field', type=int, default=2)
    parser.add_argument('--npoints', type=int, default=16000)
    parser.add_argument('--interval', type=int, default=5)
    parser.add_argument('--root', type=str, default='./Dataset/Subsets/Subset_10/LIDAR_TOP/')
    parser.add_argument('--test_scenes_list', type=str, default='./Dataset/Subsets/Subset_10/test_list.txt')
    parser.add_argument('--scene_split_lib', type=str, default='./Dataset/scene-split/')
    parser.add_argument('--pretrained_flow_model', type=str,
                        default='./Models/pretrain_models/flownet3d_kitti_odometry_maxbias1.pth')
    parser.add_argument('--pretrained_pointinet_model', type=str,
                        default='./Models/pretrain_models/interp_kitti.pth')
    parser.add_argument('--pretrained_self_model', type=str, default='./Models/result_models/field_2_interval_5_freeze_1_npoint_16000/10scenes/field_2_0.6806827462858771.pth')

    return parser.parse_args()

def test(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["WANDB_API_KEY"] = '28cbf19c5cd0619337ae4fb844d56992a283b007'

    test_dataset = NuscenesDataset(root=args.root, scenes_list=args.test_scenes_list,
                                    scene_split_lib=args.scene_split_lib, field=args.field, npoints=args.npoints,
                                    interval=args.interval, if_random=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    net = ISAPCInet(field=args.field, freeze=1).cuda()
    wandb.watch(net)

    net.flow.load_state_dict(torch.load(args.pretrained_flow_model))
    net.load_state_dict(torch.load(args.pretrained_self_model))

    net.eval()

    with torch.no_grad():

        chamfer_loss_list = []
        # emd_loss_list = []

        pbar = tqdm(test_loader)
        for data in pbar:

            # 读数据
            forward_pcds, key_pcds, backward_pcds, t, gt, ini_feature = data
            wandb.log({"t": t})
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

            cd = chamfer_loss(result, gt)
            # emd = EMD(result, gt)

            cd = cd.squeeze().cpu().numpy()
            wandb.log({"current_cd": cd})
            # emd = emd.squeeze().cpu().numpy()
            # wandb.log({"current_emd": emd})

            chamfer_loss_list.append(cd)
            # emd_loss_list.append(emd)

            # pbar.set_description('CD:{:.3} EMD:{:.3}'.format(cd, emd))
            pbar.set_description('CD:{:.3}'.format(cd))

    chamfer_loss_array = np.array(chamfer_loss_list)
    # emd_loss_array = np.array(emd_loss_list)
    mean_chamfer_loss = np.mean(chamfer_loss_array)
    # mean_emd_loss = np.mean(emd_loss_array)

    print("Mean chamfer distance: ", mean_chamfer_loss)
    # print("Mean earth mover's distance: ", mean_emd_loss)

if __name__ == '__main__':
    args = parse_args()
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb.init(project='Interval_5', config=args, name='field=2,10scenes,test_'+now_time)
    test(args)