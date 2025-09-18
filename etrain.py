import torch
from torch.autograd import Variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from datetime import datetime
from net.s2net import Net
from utils.tdataloader import get_loader
from utils.utils import clip_gradient, AvgMeter, poly_lr
import torch.nn.functional as F
import numpy as np

file = open("log/S2Net.txt", "a")
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
np.random.seed(2021)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def structure_loss(pred, mask):
    """
    pred: logits (B,1,H,W)
    mask: {0,1} (B,1,H,W)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred_prob = torch.sigmoid(pred)
    inter = ((pred_prob * mask) * weit).sum(dim=(2, 3))
    union = ((pred_prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_loss(predict, target):
    """
    predict: probability map in [0,1]
    target:  binary edge GT
    """
    smooth = 1
    p = 2
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    num = 2 * (predict * target).sum(dim=1) + smooth
    den = (predict.pow(p) + target.pow(p)).sum(dim=1) + smooth
    return (1 - num / den).mean()


def train(train_loader, model, optimizer, epoch):
    model.train()

    loss_record3, loss_record2, loss_record1, loss_recorde, loss_recordpr = (
        AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    )

    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        # ---- data prepare ----
        images, gts, edges = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        edges = Variable(edges).cuda()

        # ---- forward ----
        outs = model(images)
        if isinstance(outs, (list, tuple)) and len(outs) == 5:
            lateral_map_3, lateral_map_2, lateral_map_1, edge_map, region_map = outs
        else:
            lateral_map_3, lateral_map_2, lateral_map_1, edge_map = outs
            region_map = None

        # ---- loss function ----
        loss3 = structure_loss(lateral_map_3, gts)
        loss2 = structure_loss(lateral_map_2, gts)
        loss1 = structure_loss(lateral_map_1, gts)
        losse = dice_loss(edge_map, edges)  # edge_map=Pb 概率

        if region_map is not None:
            # region_map=Pr 概率 -> logits 再进 structure_loss
            region_logits = torch.logit(region_map.clamp(1e-6, 1 - 1e-6))
            loss_pr = structure_loss(region_logits, gts)
            loss = loss3 + loss2 + loss1 + 3 * losse + opt.w_pr * loss_pr
        else:
            loss_pr = torch.tensor(0.0, device=images.device)
            loss = loss3 + loss2 + loss1 + 3 * losse

        # ---- backward ----
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        # ---- recording loss ----
        loss_record3.update(loss3.data, opt.batchsize)
        loss_record2.update(loss2.data, opt.batchsize)
        loss_record1.update(loss1.data, opt.batchsize)
        loss_recorde.update(losse.data, opt.batchsize)
        loss_recordpr.update(loss_pr.data, opt.batchsize)

        # ---- train visualization ----
        if i % 60 == 0 or i == total_step:
            if region_map is not None:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                      '[l3: {:.4f}] [l2: {:.4f}] [l1: {:.4f}] [edge: {:.4f}] [region: {:.4f}]'
                      .format(datetime.now(), epoch, opt.epoch, i, total_step,
                              loss_record3.avg, loss_record2.avg, loss_record1.avg,
                              loss_recorde.avg, loss_recordpr.avg))
                file.write('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                           '[l3: {:.4f}] [l2: {:.4f}] [l1: {:.4f}] [edge: {:.4f}] [region: {:.4f}]\n'
                           .format(datetime.now(), epoch, opt.epoch, i, total_step,
                                   loss_record3.avg, loss_record2.avg, loss_record1.avg,
                                   loss_recorde.avg, loss_recordpr.avg))
            else:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                      '[l3: {:.4f}] [l2: {:.4f}] [l1: {:.4f}] [edge: {:.4f}]'
                      .format(datetime.now(), epoch, opt.epoch, i, total_step,
                              loss_record3.avg, loss_record2.avg, loss_record1.avg,
                              loss_recorde.avg))
                file.write('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                           '[l3: {:.4f}] [l2: {:.4f}] [l1: {:.4f}] [edge: {:.4f}]\n'
                           .format(datetime.now(), epoch, opt.epoch, i, total_step,
                                   loss_record3.avg, loss_record2.avg, loss_record1.avg,
                                   loss_recorde.avg))

    save_path = 'checkpoints/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch + 1) % 5 == 0 or (epoch + 1) == opt.epoch:
        torch.save(model.state_dict(), save_path + 'S2Net-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'S2Net-%d.pth' % epoch)
        file.write('[Saving Snapshot:]' + save_path + 'S2Net-%d.pth' % epoch + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=25, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=416, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str, default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str, default='S2Net')
    parser.add_argument('--w_pr', type=float, default=1.5, help='weight for region prior loss (Pr)')
    opt = parser.parse_args()

    # ---- build models ----
    model = Net().cuda()
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    edge_root = '{}/Edge/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("Start Training")

    for epoch in range(opt.epoch):
        poly_lr(optimizer, opt.lr, epoch, opt.epoch)
        train(train_loader, model, optimizer, epoch)

    file.close()