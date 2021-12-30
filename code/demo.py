import torch
import os
import numpy as np
import data
import model
import loss
import utility
from option import args
from PIL import Image
import time

class ImageLoader:
    def __init__(self, root):
        if os.path.isdir(root):
            self.images = [os.path.join(root, f) for f in os.listdir(root) if
                           f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        elif os.path.isfile(root):
            self.images = [root]
        self.size = len(self.images)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index == self.size:
            raise StopIteration
        image = Image.open(self.images[self.index]).convert('RGB')
        img = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float().cuda()
        fname = self.images[self.index]
        self.index += 1
        return img, fname

    def __len__(self):
        return self.size

def save_results(filename, sr, scale):
    if not os.path.exists(path):
        os.makedirs(path)
    ndarr = sr[0].byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr, 'RGB')
    im.save('{}x{}.png'.format(filename[:-4], scale))

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    #path = args.dir_demo
    path = './demo'
    args.scale = [3]
    scale = str(args.scale[0])
    args.pre_train = 'experiment/CSR_FDN/csr_fdnx'+scale+'.pt'
    dataloader = ImageLoader(path)
    ckpt = utility.checkpoint(args)
    model = model.Model(args, ckpt)

    t = 0
    for img, filename in dataloader:
        tt = time.time()
        sr = model(img)
        sr = utility.quantize(sr, 255)
        save_results(filename, sr, scale)
        t += time.time()-tt
    print('Total inference : {}s'.format(t))