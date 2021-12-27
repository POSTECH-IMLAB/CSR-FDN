import os
from decimal import Decimal
import utility
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        # self.model_idt = my_model[1]
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer, len(ckp.log))
        device = torch.device('cpu' if args.cpu else 'cuda')

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer1.pt'))
            )
        self.error_last = 1e8


    def train(self):
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]
        w = torch.zeros(1).cuda()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()

        for batch, (lr, hr, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare([lr, hr])

            timer_data.hold()
            timer_model.tic()
            self.optimizer.zero_grad()
            
            sr = self.model(lr)
            loss = self.loss(sr, hr)
            
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            timer_data.tic()
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        #self.error_last = self.loss1.log[-1, -1]
        self.scheduler.step()

    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()
        at = 0
        idx = 0
        t=0
        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                eval_acc_ssim = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare([lr, hr])
                    else:
                        lr = self.prepare([lr])[0]

                    t = time.time()
                    sr = self.model(lr)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    at += time.time() - t
                   
                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        eval_acc_ssim += utility.calc_ssim(
                            sr, hr, scale,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                    save_list.extend([lr, hr])

                    if self.args.save_results:
                        if self.args.data_test == 'vid4':
                            frame = [41, 75, 124, 171]
                            if idx_img+1 > frame[idx]:
                                idx += 1
                            self.ckp.save_results_vid(filename, save_list, idx)
                        else:
                            self.ckp.save_results(filename, save_list, scale)

                print(at)
                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.2f} SSIM: {:.4f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        eval_acc_ssim / len(self.loader_test),
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )
                #print(eval_acc/100)
        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self.model, self.loss, self.optimizer, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, l, split=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            t = tensor.to(device)
            if split: t = t.view(-1, 1, *t.shape[2:])
            return t

        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

    def sharpen(self, img):
        weights = torch.tensor([[-1., -1., -1.],
                                [-1., 8., -1.],
                                [-1., -1., -1.]])
        weights = weights.div(4)
        weights = weights.view(1, 1, 3, 3).repeat(1, 3, 1, 1)
        return F.conv2d(img, weights)

    def chop(self, x, shave=8):
        scale = self.scale[0]
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        sr_list = []
        for lr in lr_list:
            sr_batch = self.model(lr)
            sr_list.extend(sr_batch.unsqueeze(1))

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def norm(self, img):
        img = img - img.min()
        img = img / img.max()
        img = (img > 0.5).float()
        return img * 255