import torch
import os

import data
import model
import loss
import utility
from option import args
import trainer
#from ptflops import get_model_complexity_info

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    args.test_only = True
    args.data_test = 'Set5'  # specify dataset for test.
    args.save_results = True
    loader = data.Data(args)
    ckpt = utility.checkpoint(args)
    model = model.Model(args, ckpt)
    loss = loss.Loss(args, ckpt) if not args.test_only else None
    # macs, params = get_model_complexity_info(model, (3, 320, 180), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    t = trainer.Trainer(args, loader, model, loss, ckpt)
    t.test()

    ckpt.done()


