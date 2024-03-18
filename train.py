import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import datetime
import time
import wandb
from tqdm import tqdm
import argparse

from Dataset.InterpolationData import NuscenesDataset
from Models.New_Models0 import ISAPCInet
from Utils.Utils import ClippedStepLR, chamfer_loss



def parse_args():
    parser = argparse.ArgumentParser(description='20230504withTr')
    # training hyperparameters set
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--init_lr', type=float, default=0.01)
    parser.add_argument('--min_lr', type=float, default=0.000001)
    parser.add_argument('--step_size_lr', type=int, default=100)
    parser.add_argument('--gamma_lr', type=float, default=0.9)
    parser.add_argument('--init_bn_momentum', type=float, default=0.5)
    parser.add_argument('--min_bn_momentum', type=float, default=0.01)
    parser.add_argument('--step_size_bn_momentum', type=int, default=100)
    parser.add_argument('--gamma_bn_momentum', type=float, default=0.5)
    parser.add_argument('--gpu', type=str, default='0')
    # model hyperparameters set
    parser.add_argument('--root', type=str, default='./Dataset/Subsets/Subset_01/LIDAR_TOP/')
    parser.add_argument('--train_scenes_list', type=str, default='./Dataset/Subsets/Subset_01/train_list_1.txt')
    parser.add_argument('--if_random', type=bool, default=True)
    parser.add_argument('--scene_split_lib', type=str, default='./Dataset/scene-split/')
    parser.add_argument('--field', type=int, default=2)
    parser.add_argument('--random_times', type=int, default=1)
    parser.add_argument('--npoints', type=int, default=16000)
    parser.add_argument('--interval', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='./Models/result_models/field_2_interval_5_freeze_1_npoint_16000/10scenes/')
    parser.add_argument('--pretrained_flow_model', type=str,
                        default='./Models/pretrain_models/flownet3d_kitti_odometry_maxbias1.pth')
    parser.add_argument('--pretrained_pointinet_model', type=str,
                        default='./Models/pretrain_models/interp_kitti.pth')
    parser.add_argument('--pretrained_self_model', type=str, default='./Models/result_models/field_2_interval_5_freeze_1_npoint_16000/1scene/field_2_0.4047063953346676.pth')
    parser.add_argument('--freeze', type=int, default=1)
    return parser.parse_args()


def train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["WANDB_API_KEY"] = '28cbf19c5cd0619337ae4fb844d56992a283b007'

    train_dataset = NuscenesDataset(root=args.root, scenes_list=args.train_scenes_list,
                                    scene_split_lib=args.scene_split_lib, field=args.field, npoints=args.npoints,
                                    interval=args.interval,if_random = True, random_times=args.random_times)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True,
                              drop_last=True)

    net = ISAPCInet(field=args.field, freeze=args.freeze).cuda()
    wandb.watch(net)

    net.flow.load_state_dict(torch.load(args.pretrained_flow_model))
    # net.pointinet.load_state_dict(torch.load(args.pretrained_pointinet_model))
    net.load_state_dict(torch.load(args.pretrained_self_model))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.init_lr)
    lr_scheduler = ClippedStepLR(optimizer, args.step_size_lr, args.min_lr, args.gamma_lr)

    def init_weights(net):
        for m in net.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0.0)
                
    # init_weights(net)

    def update_bn_momentum(net, epoch):
        for m in net.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.momentum = max(
                    args.init_bn_momentum * args.gamma_bn_momentum ** (epoch // args.step_size_bn_momentum),
                    args.min_bn_momentum)

    best_train_loss = float('inf')

    for epoch in range(args.epochs):
        # 初始化
        update_bn_momentum(net, epoch)
        net.train()
        wandb.log({"epochs": epoch})
        count = 0
        total_loss = 0

        pbar = tqdm(enumerate(train_loader))

        for i, data in pbar:
            # 读数据
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

            optimizer.zero_grad()

            nowtime = time.time()
            result = net(forward_pcds, key_pcds, backward_pcds, t, ini_feature)
            time_per_step = time.time() - nowtime
            wandb.log({"time_per_step": time_per_step})

            loss = chamfer_loss(result, gt)
            wandb.log({"current_train_loss": loss.item()})

            loss.backward()
            optimizer.step()

            count += 1
            total_loss += loss.item()

            pbar.set_description('Train Epoch:{}[{}/{}({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, i, len(train_loader), 100. * i / len(train_loader), loss.item()
            ))

        lr_scheduler.step()
        total_loss = total_loss / count
        wandb.log({"epoch_loss": total_loss})

        print('Epoch ', epoch + 1, 'finished ', 'loss = ', total_loss)

        if total_loss < best_train_loss:
            torch.save(net.state_dict(), args.save_dir + 'field_' + str(args.field) + '_' + str(total_loss) + '.pth')
            best_train_loss = total_loss
            wandb.log({"best_train_loss": best_train_loss})

        print('Best train loss: {:.4f}'.format(best_train_loss))


if __name__ == '__main__':
    args = parse_args()
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    wandb.init(project='Interval_5', config=args, name='field=2,10scenes,train:'+now_time)

    train(args)
