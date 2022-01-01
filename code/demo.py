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
            self.fname = [f for f in os.listdir(root) if
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
        fname = self.fname[self.index]
        self.index += 1
        return img, fname

    def __len__(self):
        return self.size

def save_attn_map(filename, at, scale, postfix):
    save_path = './demo/SRx{}'.format(scale)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    a = at.mean(dim=0)
    a = (a>0.5).float()*255
    ndarr = a.byte().cpu().numpy()
    im = Image.fromarray(ndarr, 'L')
    im.save('{}/{}_{}.png'.format(save_path, filename[:-4], postfix))


def save_results(filename, sr, scale):
    save_path = './demo/SRx{}'.format(scale)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ndarr = sr[0].byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr, 'RGB')
    im.save('{}/{}.png'.format(save_path, filename[:-4]))

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    #path = args.dir_demo
    path = './demo'
    #args.scale = [3]
    args.model = 'CSR_FDN_DEMO'
    scale = str(args.scale)
    args.pre_train = 'experiment/CSR_FDN/csr_fdnx'+scale+'.pt'
    dataloader = ImageLoader(path)
    ckpt = utility.checkpoint(args)
    model = model.Model(args, ckpt)

    t = 0
    for img, filename in dataloader:
        tt = time.time()
        sr, a1, a2 = model(img)
        sr = utility.quantize(sr, 255)
        save_results(filename, sr, scale)
        save_attn_map(filename, a1, scale, 'S')
        save_attn_map(filename, a2, scale, 'F')
        t += time.time()-tt
    print('Total inference : {}s'.format(t))