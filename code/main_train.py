import torch
import os

import data
import model
import loss
import utility
from option import args
import trainer
import copy
import template
from ptflops import get_model_complexity_info

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    loader = data.Data(args)
    #template.set_template(args)
    ckpt = utility.checkpoint(args)
    model = model.Model(args, ckpt)
    loss = loss.Loss(args, ckpt) if not args.test_only else None
    macs, params = get_model_complexity_info(model, (3, 320, 180), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    t = trainer.Trainer(args, loader, model, loss, ckpt)
    while not t.terminate():
        t.train()
        t.test()

    ckpt.done()


