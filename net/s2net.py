import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from net.ResNet import resnet50  # 若未使用可移除
from net.Res2Net import res2net50_v1b_26w_4s


# -----------------------------
# 基础模块
# -----------------------------
class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # (可选) 更稳定的初始化
        # nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# -----------------------------
# 稳健版 SHI 频谱增强 (用于 SEM) —— 原 SVDRefiner
# -----------------------------
class SHI(nn.Module):
    def __init__(self, in_channels: int, max_rank: int = 128, keep_ratio: float = 0.4, pool_hw: int = 24):
        super().__init__()
        assert 0.0 < keep_ratio <= 1.0
        self.in_channels = in_channels
        self.r = min(in_channels, max_rank)
        self.keep_ratio = keep_ratio
        self.pool_hw = pool_hw

        self.reduce = nn.Conv2d(in_channels, self.r, kernel_size=1, bias=False)
        self.expand = nn.Conv2d(self.r, in_channels, kernel_size=1, bias=False)

        # 门控参数（更保守的初值）
        self.beta = nn.Parameter(torch.tensor(0.30))   # 高频注入强度
        self.gate = nn.Parameter(torch.tensor(0.0))    # 基线门控

    @staticmethod
    def _safe_svd(x64: torch.Tensor):
        try:
            return torch.linalg.svd(x64, full_matrices=False, driver='gesvd')
        except TypeError:
            return torch.linalg.svd(x64, full_matrices=False)
        except Exception:
            U, S, Vh = torch.linalg.svd(x64.cpu(), full_matrices=False)
            return U.to(x64.device), S.to(x64.device), Vh.to(x64.device)

    def _svd_highpass(self, x_r: torch.Tensor, att: torch.Tensor = None) -> torch.Tensor:
        """
        x_r: (B, r, H, W)
        att: (B, 1, H, W) or None
        """
        B, r, H, W = x_r.shape

        # 1) 统计用自适应池化
        ph = min(self.pool_hw, H)
        pw = min(self.pool_hw, W)
        x_stat = F.adaptive_avg_pool2d(x_r, output_size=(ph, pw))  # (B, r, ph, pw)

        # 2) 数据矩阵 Xs（中心化+标准化+抖动）
        Xs = x_stat.reshape(B, r, -1).to(torch.float32)            # (B, r, Np)
        Xs = Xs - Xs.mean(dim=-1, keepdim=True)
        std = Xs.std(dim=-1, keepdim=True).clamp_min(1e-6)
        Xs = Xs / std
        Xs = Xs + (1e-4) * torch.randn_like(Xs)

        # 3) 子空间 stop-grad + float64 分解
        Xs_basis = Xs.detach().to(torch.float64)
        U, S, Vh = self._safe_svd(Xs_basis)                        # U: (B, r, r')

        # 4) 取前 k 列子空间
        r_prime = U.shape[-1]
        k = max(1, min(r_prime, int(self.r * self.keep_ratio)))
        U_k = U[:, :, :k].to(torch.float32).detach()               # (B, r, k)

        # 5) 低秩投影 + 高频残差
        X = x_r.reshape(B, r, -1).to(torch.float32)                # (B, r, N)
        X_low = torch.bmm(U_k, torch.bmm(U_k.transpose(1, 2), X))  # (B, r, N)
        X_high = X - X_low
        x_high = X_high.reshape(B, r, H, W)

        # 6) 能量自适应 + 上限
        high_e = x_high.pow(2).mean(dim=(1, 2, 3), keepdim=True)
        base_e = x_r.pow(2).mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        energy_ratio = (high_e / base_e).sqrt().clamp(max=1.0)

        # 7) 边缘门控融合
        base_gate = torch.sigmoid(self.gate)                       # 标量
        if att is not None:
            att_small = F.adaptive_avg_pool2d(att, output_size=1)  # (B,1,1,1)
            mix = torch.clamp(base_gate + att_small, 0.0, 1.0)
        else:
            mix = base_gate.view(1, 1, 1, 1)

        effective_beta = torch.clamp(self.beta, 0.0, 1.2)
        x_out = x_r + effective_beta * energy_ratio * mix * x_high
        return x_out

    def forward(self, x: torch.Tensor, att: torch.Tensor = None) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.to(torch.float32)
        xr = self.reduce(x32)                 # (B, r, H, W)
        xr = self._svd_highpass(xr, att)      # (B, r, H, W)
        x_enh = self.expand(xr)               # (B, C, H, W)
        return x_enh.to(orig_dtype)


# -----------------------------
# 混合注意力模块（DAN，用于 SEM）
# -----------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avg_out, max_out], dim=1)
        att = self.conv(att)
        return self.sigmoid(att)


class DAN(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(DAN, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction_ratio)
        self.spatial_att = SpatialAttention()
        # 自适应系数（可学习）
        self.alpha = nn.Parameter(torch.tensor(0.4))
        self.beta = nn.Parameter(torch.tensor(0.6))

    def forward(self, x):
        x = self.alpha * self.channel_att(x) * x
        x = self.beta * self.spatial_att(x) * x
        return x


# -----------------------------
# SEM / CAM —— 原 EFM
# -----------------------------
class SEM(nn.Module):
    def __init__(self, channel,
                 svd_max_rank: int = 128,
                 svd_keep_ratio: float = 0.4,
                 svd_pool_hw: int = 24,
                 enable_refine: bool = True):
        super(SEM, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.enable_refine = enable_refine

        # 先 SHI 再 3×3 卷积（平滑高频）
        self.shi = SHI(
            in_channels=channel,
            max_rank=svd_max_rank,
            keep_ratio=svd_keep_ratio,
            pool_hw=svd_pool_hw
        )
        self.conv2d = ConvBNR(channel, channel, 3)

        # 混合注意力（通道 + 空间）
        self.dan = DAN(channel, reduction_ratio=16)

    def forward(self, c, att):
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)

        x = c * att + c
        if self.enable_refine:
            x = self.shi(x, att)              # SHI 频谱增强（可按层开关）
        x = self.conv2d(x)                    # 平滑融合
        x = self.dan(x)                       # DAN 混合注意
        return x


class CAM(nn.Module):
    def __init__(self, hchannel, channel):
        super(CAM, self).__init__()
        self.conv1_1 = Conv1x1(hchannel + channel, channel)
        self.conv3_1 = ConvBNR(channel // 4, channel // 4, 3)
        self.dconv5_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=2)
        self.dconv7_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=3)
        self.dconv9_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=4)
        self.conv1_2 = Conv1x1(channel, channel)
        self.conv3_3 = ConvBNR(channel, channel, 3)

    def forward(self, lf, hf):
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((lf, hf), dim=1)
        x = self.conv1_1(x)
        xc = torch.chunk(x, 4, dim=1)
        x0 = self.conv3_1(xc[0] + xc[1])
        x1 = self.dconv5_1(xc[1] + x0 + xc[2])
        x2 = self.dconv7_1(xc[2] + x1 + xc[3])
        x3 = self.dconv9_1(xc[3] + x2)
        xx = self.conv1_2(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.conv3_3(x + xx)
        return x


# -----------------------------
# RBE & 多尺度空间门
# -----------------------------
class MPD(nn.Module):
    """Fuse shallow & deep (both 64ch) like PRBE. —— 原 MPD2"""
    def __init__(self, in_ch=64, out_ch=64):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvBNR(in_ch * 2, out_ch, 3),
            ConvBNR(out_ch, out_ch, 3),
        )

    def forward(self, Fs, Fd):
        if Fs.size()[2:] != Fd.size()[2:]:
            Fd = F.interpolate(Fd, Fs.size()[2:], mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([Fs, Fd], dim=1))


class RBE(nn.Module):
    """
    Region & Boundary Exploration:
      Fch = MPD(x3', x4'); Pr_h = sig(head(Fch)) at H/16
      Pr2 = up(Pr_h->H/8), Pr1 = up(Pr_h->H/4)
      Fcl = MPD(x1'*Pr1 + x1', x2'*Pr2 + x2') at H/4
      Pb  = sig(head(|Fcl - up(Fch->H/4)|)) at H/4
      return Pr1 (H/4), Pb (H/4)
    """
    def __init__(self):
        super().__init__()
        self.c1 = Conv1x1(256, 64)
        self.c2 = Conv1x1(512, 64)
        self.c3 = Conv1x1(1024, 64)
        self.c4 = Conv1x1(2048, 64)
        self.mpd_high = MPD(64, 64)
        self.mpd_low  = MPD(64, 64)
        self.pr_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1)
        )
        self.pb_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3, x4):
        x1r, x2r, x3r, x4r = self.c1(x1), self.c2(x2), self.c3(x3), self.c4(x4)
        Fch = self.mpd_high(x3r, x4r)                  # H/16
        Pr_h = self.sigmoid(self.pr_head(Fch))         # (B,1,H/16,W/16)

        Pr2 = F.interpolate(Pr_h, size=x2r.size()[2:], mode='bilinear', align_corners=False)  # H/8
        Pr1 = F.interpolate(Pr_h, size=x1r.size()[2:], mode='bilinear', align_corners=False)  # H/4

        x1e, x2e = x1r * Pr1 + x1r, x2r * Pr2 + x2r
        Fcl = self.mpd_low(x1e, x2e)                   # H/4

        Fch_up = F.interpolate(Fch, size=Fcl.size()[2:], mode='bilinear', align_corners=False)
        Fb  = torch.abs(Fcl - Fch_up)
        Pb  = self.sigmoid(self.pb_head(Fb))           # (B,1,H/4,W/4)

        return Pr1, Pb


class PFG(nn.Module):
    """Per-pixel fusion: alpha = sigmoid(Conv([Pr,Pb])); att = alpha*Pb + (1-alpha)*Pr. —— 原 SpatialGate"""
    def __init__(self, k=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, k, padding=k//2), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1), nn.Sigmoid()
        )

    def forward(self, Pr, Pb):
        alpha = self.net(torch.cat([Pr, Pb], dim=1))   # (B,1,H,W)
        return alpha * Pb + (1 - alpha) * Pr


# -----------------------------
# 主网：输出 (o3, o2, o1, oe=Pb, opr=Pr)
# 只在 x3/x4 开启 SHI，x1/x2 关闭（更稳）
# -----------------------------
class Net(nn.Module):
    def __init__(self,
                 svd_max_rank: int = 128,
                 svd_keep_ratio: float = 0.4,
                 svd_pool_hw: int = 24,
                 enable_svd_stages=(False, False, True, True)):
        super(Net, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

        # --- 用 RBE 替代 DSM，并加多尺度空间门 ---
        self.rbe = RBE()
        self.gate = PFG()

        self.sem1 = SEM(256,  svd_max_rank, svd_keep_ratio, svd_pool_hw, enable_refine=enable_svd_stages[0])
        self.sem2 = SEM(512,  svd_max_rank, svd_keep_ratio, svd_pool_hw, enable_refine=enable_svd_stages[1])
        self.sem3 = SEM(1024, svd_max_rank, svd_keep_ratio, svd_pool_hw, enable_refine=enable_svd_stages[2])
        self.sem4 = SEM(2048, svd_max_rank, svd_keep_ratio, svd_pool_hw, enable_refine=enable_svd_stages[3])

        self.reduce1 = Conv1x1(256, 64)
        self.reduce2 = Conv1x1(512, 128)
        self.reduce3 = Conv1x1(1024, 256)
        self.reduce4 = Conv1x1(2048, 256)

        self.cam1 = CAM(128, 64)
        self.cam2 = CAM(256, 128)
        self.cam3 = CAM(256, 256)

        self.predictor1 = nn.Conv2d(64, 1, 1)
        self.predictor2 = nn.Conv2d(128, 1, 1)
        self.predictor3 = nn.Conv2d(256, 1, 1)

    def forward(self, x):
        x1, x2, x3, x4 = self.resnet(x)

        # RBE: 产出 H/4 尺度的 Pr/Pb
        Pr, Pb = self.rbe(x1, x2, x3, x4)  # (B,1,H/4,W/4)

        # 多尺度空间门：为每个尺度生成自己的融合注意
        att1 = self.gate(Pr, Pb)  # H/4 -> for x1

        Pr2 = F.interpolate(Pr, size=x2.size()[2:], mode='bilinear', align_corners=False)  # H/8
        Pb2 = F.interpolate(Pb, size=x2.size()[2:], mode='bilinear', align_corners=False)
        att2 = self.gate(Pr2, Pb2)

        Pr3 = F.interpolate(Pr, size=x3.size()[2:], mode='bilinear', align_corners=False)  # H/16
        Pb3 = F.interpolate(Pb, size=x3.size()[2:], mode='bilinear', align_corners=False)
        att3 = self.gate(Pr3, Pb3)

        Pr4 = F.interpolate(Pr, size=x4.size()[2:], mode='bilinear', align_corners=False)  # H/32
        Pb4 = F.interpolate(Pb, size=x4.size()[2:], mode='bilinear', align_corners=False)
        att4 = self.gate(Pr4, Pb4)

        # 边缘引导 + SHI（按层开关保持不变）
        x1a = self.sem1(x1, att1)
        x2a = self.sem2(x2, att2)
        x3a = self.sem3(x3, att3)
        x4a = self.sem4(x4, att4)

        # 通道降维
        x1r = self.reduce1(x1a)
        x2r = self.reduce2(x2a)
        x3r = self.reduce3(x3a)
        x4r = self.reduce4(x4a)

        # 自顶向下聚合
        x34 = self.cam3(x3r, x4r)
        x234 = self.cam2(x2r, x34)
        x1234 = self.cam1(x1r, x234)

        # 预测头（保持 logits 输出）
        o3 = self.predictor3(x34)
        o3 = F.interpolate(o3, scale_factor=16, mode='bilinear', align_corners=False)
        o2 = self.predictor2(x234)
        o2 = F.interpolate(o2, scale_factor=8, mode='bilinear', align_corners=False)
        o1 = self.predictor1(x1234)
        o1 = F.interpolate(o1, scale_factor=4, mode='bilinear', align_corners=False)

        # 概率输出（边界/区域）
        oe  = F.interpolate(Pb, scale_factor=4, mode='bilinear', align_corners=False)   # Pb 概率
        opr = F.interpolate(Pr, scale_factor=4, mode='bilinear', align_corners=False)   # Pr 概率

        return o3, o2, o1, oe, opr
