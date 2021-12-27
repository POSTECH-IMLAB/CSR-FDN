import random

import numpy as np
import skimage.io as sio
import skimage.color as sc
import skimage.transform as st

import torch
from torchvision import transforms

def get_patch_idt(img, patch_size, multi_scale=False):
    tp = patch_size
    ix = random.randrange(0, img.shape[1] - tp + 1)
    iy = random.randrange(0, img.shape[0] - tp + 1)

    img_tar = img[iy:iy + tp, ix:ix + tp, :]
    return img_tar

def get_patch(img_in, img_tar, patch_size, scale, multi_scale=False):
    ih, iw = img_in.shape[:2]

    lr_p = patch_size
    hr_p = lr_p * scale

    ix = random.randrange(0, iw - lr_p + 1)
    iy = random.randrange(0, ih - lr_p + 1)
    tx, ty = scale * ix, scale * iy
    img_in = img_in[iy:iy + lr_p, ix:ix + lr_p, :]
    img_tar = img_tar[ty:ty + hr_p, tx:tx + hr_p, :]

    return img_in, img_tar


def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channel == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]


def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose.copy()).float()
        tensor = tensor.mul_(rgb_range / 255)
        return tensor

    return [_np2Tensor(_l) for _l in l]


def add_noise(x, noise='.'):
    if noise != '.':
        noise_type = noise[0]
        noise_value = int(noise[1:])
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)

        x_noise = x.astype(np.int16) + noises.astype(np.int16)
        x_noise = x_noise.clip(0, 255).astype(np.uint8)
        return x_noise
    else:
        return x


def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(_l) for _l in l]
