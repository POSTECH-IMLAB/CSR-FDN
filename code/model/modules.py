import torch
import torch.nn as nn

class ch_shuffle1(nn.Module):
    def __init__(self):
        super(ch_shuffle1, self).__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B // 3, 3, C, H, W).permute(0, 2, 1, 3, 4)
        return x.reshape(B // 3, -1, H, W)


class ch_shuffle2(nn.Module):
    def __init__(self):
        super(ch_shuffle2, self).__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, 3, C // 3, H, W).permute(0, 2, 1, 3, 4).reshape(B, -1, H, W)
        return x

class ch_shuffle3(nn.Module):
    def __init__(self):
        super(ch_shuffle3, self).__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C // 3, 3, H, W).permute(0, 2, 1, 3, 4).reshape(B*3, -1, H, W)
        return x

class ProjUnit(nn.Module):
    def __init__(self, n_feat, kernel_sz):
        super(ProjUnit, self).__init__()
        # global average pooling: feature --> point
        self.cv1 = nn.Sequential(nn.Conv2d(n_feat*3, kernel_sz*3, 3, padding=1, groups=3),
                                 nn.LeakyReLU())
        self.sfm = nn.Softmax(dim=5)
        self.sgm = nn.Sigmoid()
        self.tmp = nn.Parameter(torch.Tensor([10]))  # nn.Threshold(0.2, 0)

    def forward(self, x, y):
        B, C, H, W = x.shape
        y = self.cv1(y)
        y = y.view(B//3, 3, 1, -1, H, W).permute(0, 1, 2, 4, 5, 3)
        x = x.view(B//3, 3, C, H, W, -1)
        s = self.sgm(x * y)
        s = self.sfm(s)
        out = (y * s).sum(dim=5)
        return out.view(B, C, H, W)

class CCFLayer(nn.Module):
    def __init__(self, ch, n=4):
        super(CCFLayer, self).__init__()
        # global average pooling: feature --> point
        self.fusion = nn.Sequential(ch_shuffle1(),
                                    nn.Conv2d(ch*3, ch*3, 3, padding=1, groups=ch),
                                    nn.LeakyReLU(),
                                    ch_shuffle2())
        self.fp = ProjUnit(ch, n)

    def forward(self, x):
        f = self.fusion(x)
        out = self.fp(x, f)
        return out+x

# Channel Attention (CA) Layer
class AFD(nn.Module):
    def __init__(self, channel, reduction=16):
        super(AFD, self).__init__()
        # global average pooling: feature --> point
        self.arr = nn.Conv2d(channel, channel, 1)
        self.red = nn.Sequential(nn.Conv2d(channel, channel // 4, 3, padding=1),
                                 nn.LeakyReLU(),
                                 nn.Conv2d(channel//4, channel // 2, 3, padding=1),
                                 )
        self.sgm = nn.Sigmoid()
        self.ch = channel
        # self.cca = CCALayer(channel//2, 2)

    def forward(self, x):
        x = self.arr(x)
        atmap = self.sgm(self.red(x))
        b, r = torch.split(x, (self.ch // 2, self.ch // 2), dim=1)
        out = b * atmap
        res = r * (1 - atmap)
        return out, res, atmap, (1 - atmap)

class Block_Last(nn.Module):
    def __init__(self, n_feat):
        super(Block_Last, self).__init__()
        conv = default_conv

        self.b0 = ResBlock(conv, n_feat // 2, 3, act=nn.LeakyReLU(inplace=True))
        self.b1 = ResBlock(conv, n_feat // 2, 3, act=nn.LeakyReLU(inplace=True))
        self.b2 = ResBlock(conv, n_feat // 2, 3, act=nn.LeakyReLU(inplace=True))
        self.b3 = ResBlock(conv, n_feat // 2, 3, act=nn.LeakyReLU(inplace=True))

        self.cs0 = AFD(n_feat)
        self.cs1 = AFD(n_feat)
        self.cs2 = AFD(n_feat)
        self.cs3 = AFD(n_feat)

        self.cv_out = conv(n_feat * 2 + n_feat // 2, n_feat, 1)

    def forward(self, x):
        b01, b02, a1, a2 = self.cs0(x)
        b0 = self.b0(b01)

        b11, b12, _, _ = self.cs1(torch.cat([b0, b01], dim=1))
        b1 = self.b1(b11)

        b21, b22, _, _ = self.cs2(torch.cat([b1, b11], dim=1))
        b2 = self.b2(b21)

        b31, b32, _, _ = self.cs3(torch.cat([b2, b21], dim=1))
        b3 = self.b3(b31)

        out = self.cv_out(torch.cat([b02, b12, b22, b32, b3], dim=1))
        return out + x, a1, a2

class Block(nn.Module):
    def __init__(self, n_feat):
        super(Block, self).__init__()

        self.b = Block_Last(n_feat)
        self.ccf = CCFLayer(n_feat)

    def forward(self, x):
        out, a1, a2 = self.b(x)
        out = self.ccf(out)
        return out, a1, a2

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, out_f, bias=True):
        m = []
        m.append(conv(n_feat, n_feat, 1))
        if scale == 2:  # Is scale = 2^n?
            m.append(conv(n_feat, 4 * out_f, 3, bias))
            m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(conv(n_feat, 9 * out_f, 3, bias))
            m.append(nn.PixelShuffle(3))
        elif scale == 4:
            m.append(conv(n_feat, 16 * out_f, 3, bias))
            m.append(nn.PixelShuffle(4))
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.LeakyReLU(inplace=True), res_scale=1, groups=1, skip=True):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias, groups=groups))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.skip = skip

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        if self.skip:
            res += x
        return res

def default_conv(in_channels, out_channels, kernel_size, bias=True, groups=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, groups=groups)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, n_feat, out_feat, kernel_size, stride=1, bias=True,
            bn=False, act=nn.ReLU(), groups=1):
        super(BasicBlock, self).__init__()
        m = []
        conv = default_conv
        m.append(conv(n_feat, out_feat, kernel_size, bias=bias, groups=groups))
        m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        return res