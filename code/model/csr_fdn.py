import torch
import torch.nn as nn
from . import modules as M
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def make_model(args, parent=False):
    return CSRN(args)

class CSRN(nn.Module):
    def __init__(self, args):
        super(CSRN, self).__init__()
        conv = M.default_conv

        self.s = scale = args.scale[0]
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        n_feat = 64
        self.sub_mean = M.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = M.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.entry1 = nn.Conv2d(1, n_feat, 3, 1, 1)

        self.b1 = M.Block(n_feat)
        self.b2 = M.Block(n_feat)
        self.b3 = M.Block(n_feat)
        self.b4 = M.Block_Last(n_feat)
        
        self.out_red = nn.Sequential(conv(n_feat * 4, n_feat, 1),
                                     nn.LeakyReLU(),
                                     conv(n_feat, n_feat, 3))

        self.upsample = M.Upsampler(conv, scale, n_feat, 1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.entry1(x.view(-1, 1, *x.shape[2:]))

        #d = self.entry1.weight.data.view(-1).cpu().numpy()
        # fig, axs = plt.subplots(1, 1, tight_layout=True)
        # axs.hist(d, bins=100, alpha=1, histtype='step')
        # plt.show()
        #d = print(torch.var(self.entry1.weight.data.view(-1).cpu()))

        b1, a1, a2 = self.b1(x)
        b2, _, _ = self.b2(b1)
        b3, _, _ = self.b3(b2)
        b4, _, _ = self.b4(b3)

        out = self.out_red(torch.cat([b1, b2, b3, b4], dim=1))
        out = self.upsample(out + x)
        out = self.add_mean(out.view(-1, 3, *out.shape[2:]))
        return out, a1.mean(dim=1).mean(dim=0), a2.mean(dim=1).mean(dim=0)
    #
    # def load_state_dict(self, state_dict, strict=False):
    #     own_state = self.state_dict()
    #     for (name, param), (name2, param2) in zip(state_dict.items(), own_state.items()):
    #         own_state[name2].copy_(state_dict[name])

