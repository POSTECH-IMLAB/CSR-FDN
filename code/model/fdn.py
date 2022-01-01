import torch
import torch.nn as nn
from . import modules as M

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

        self.entry1 = nn.Conv2d(3, n_feat, 3, 1, 1)
        # self.entry2 = nn.Conv2d(3, n_feat, 3, 1, 1)

        self.b1 = M.Block_Last(n_feat)
        self.b2 = M.Block_Last(n_feat)
        self.b3 = M.Block_Last(n_feat)
        self.b4 = M.Block_Last(n_feat)
        
        self.out_red = nn.Sequential(conv(n_feat * 4, n_feat, 1),
                                     nn.LeakyReLU(),
                                     conv(n_feat, n_feat, 3))

        self.upsample = M.Upsampler(conv, scale, n_feat, 3)

    def forward(self, x):
        x = self.sub_mean(x)
        x1 = self.entry1(x)

        b1, a1, a2 = self.b1(x1)
        b2, _, _ = self.b2(b1)
        b3, _, _ = self.b3(b2)
        b4, _, _ = self.b4(b3)

        out = self.out_red(torch.cat([b1, b2, b3, b4], dim=1))
        out = self.upsample(out + x1)
        out = self.add_mean(out.view(-1, 3, *out.shape[2:]))
        return out.view(-1, 3, *out.shape[2:]), a1, a2

    # def load_state_dict(self, state_dict, strict=False):
    #     own_state = self.state_dict()
    #     for (name, param), (name2, param2) in zip(state_dict.items(), own_state.items()):
    #         #print(own_state[name2])
    #         #print(name2)
    #         #print(name)
    #         #if isinstance(param, nn.Parameter):
    #         #    param = param.data
    #         #own_state[name2].copy_(param)
    #         # if name in own_state:
    #         # if isinstance(param, nn.Parameter):
    #         #     param = param.data
    #         #print(own_state[name2])
    #         own_state[name2].copy_(state_dict[name])
    #         #print(own_state[name2])
    #         # try:
    #         #     own_state[name2].copy_(state_dict[name])
    #         # except Exception:
    #         #     if name.find('upsample') >= 0:
    #         #         print('Replace pre-trained upsampler to new one...')
    #         #     elif name.find('exit') >= 0:
    #         #         print('Replace pre-trained exit to new one...')
    #         #     else:
    #         #         raise RuntimeError('While copying the parameter named {}, '
    #         #                            'whose dimensions in the model are {} and '
    #         #                            'whose dimensions in the checkpoint are {}.'
    #         #                            .format(name, own_state[name].size(), param.size()))
    #         # elif strict:
    #         #     if name.find('upsample') == -1:
    #         #         raise KeyError('unexpected key "{}" in state_dict'
    #         #                        .format(name))
    #         #     if name.find('exit') == -1:
    #         #         raise KeyError('unexpected key "{}" in state_dict'
    #         #                        .format(name))
    #
    #     if strict:
    #         missing = set(own_state.keys()) - set(state_dict.keys())
    #         if len(missing) > 0:
    #             raise KeyError('missing keys in state_dict: "{}"'.format(missing))

