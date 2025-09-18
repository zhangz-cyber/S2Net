import imageio
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from net.s2net import Net
from utils.tdataloader import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=416, help='testing size')
parser.add_argument('--pth_path', type=str, default='./checkpoints/S2Net/')
opt = parser.parse_args()

# 建议：模型只加载一次就好（每个数据集重复加载会变慢）
model = Net().cuda()
# 如果你的 torch 版本支持，可把下一行改成 torch.load(opt.pth_path, weights_only=True)
state = torch.load(opt.pth_path)
model.load_state_dict(state)
model.eval()

for _data_name in ['CAMO','CHAMELEON','COD10K','NC4K']:
    data_path = f'./data/TestDataset/{_data_name}/'
    save_path = f'./results/{_data_name}/'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + 'edge/', exist_ok=True)

    image_root = f'{data_path}/Imgs/'
    gt_root    = f'{data_path}/GT/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        with torch.no_grad():
            outs = model(image)
            # 兼容 4/5 路输出： (o3, o2, o1, oe[, opr])
            if isinstance(outs, (list, tuple)):
                if len(outs) == 5:
                    _, _, res, e, _ = outs
                elif len(outs) == 4:
                    _, _, res, e = outs
                else:
                    raise RuntimeError(f'Unexpected number of outputs: {len(outs)}')
            else:
                raise RuntimeError('Model output should be a tuple/list')

            # 注意：res 是 logits（网络第三路），e 是概率（边界）
            # 将 res 上采样到 GT 尺寸并做 sigmoid
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            imageio.imwrite(save_path + name, (res * 255).astype(np.uint8))

            # 如需保存边缘，可取消注释（e 已是概率）
            # e_up = F.interpolate(e, size=gt.shape, mode='bilinear', align_corners=False)
            # e_np = e_up.data.cpu().numpy().squeeze()
            # e_np = (e_np - e_np.min()) / (e_np.max() - e_np.min() + 1e-8)
            # imageio.imwrite(save_path + 'edge/' + name, (e_np * 255).astype(np.uint8))
