
from skimage import color
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_laplace
import numpy as np

class LAP(nn.Module):
    def __init__(self):
        super(LAP, self).__init__()

    def forward(self, sr, hr):
        def _forward(x):
            x = gaussian_laplace(x, sigma=1.5)
            return x

        def pix_range(img):
            pixel_range = []
            min_v = 256
            max_v = -1
            for h in range(len(img)):
                for w in range(len(img[0])):
                    min_v = min(min_v, img[h, w])
                    max_v = max(max_v, img[h, w])
            pixel_range.append({min_v, max_v})
            return pixel_range

        npsr = sr[0].cpu().permute(1, 2, 0).detach().numpy()
        nphr = hr[0].cpu().permute(1, 2, 0).detach().numpy()

        sr_lap = _forward(npsr)
        with torch.no_grad():
            hr_lap = _forward(nphr)

        sr_edge = self.threshold(sr_lap)
        hr_edge = self.threshold(hr_lap)

        loss = F.l1_loss(torch.Tensor(sr_edge), torch.Tensor(hr_edge))

        return loss

    def threshold(self, img):
        nparr = np.array(img)
        thr_arr = nparr < 0
        return thr_arr.astype(float)
