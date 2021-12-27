import utility
from model import common
from loss import discriminator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def gram_matrix(input):
    B, C, H, W = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(B, C, H * W)
    G = torch.matmul(features,  torch.transpose(features, 1, 2))  # compute the gram product

    return G.view(B, C * C)


class Adversarial(nn.Module):
    def __init__(self, args, gan_type):
        super(Adversarial, self).__init__()
        self.args = args
        self.gan_type = gan_type
        self.gan_k = args.gan_k
        self.discriminator = discriminator.Discriminator(args, gan_type)
        if gan_type != 'WGAN_GP':
            self.optimizer = utility.make_optimizer(args, self.discriminator)
        else:
            self.optimizer = optim.Adam(
                self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-8, lr=1e-5
            )
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.sigmoid = nn.Sigmoid()
        self.BCE = nn.BCEWithLogitsLoss()

    def forward(self, fake, real):
        criterion = nn.MSELoss()
        #loss_gram = criterion(gram_matrix(fake), gram_matrix(real))
        fake_detach = fake.detach()

        self.loss = 0
        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            d_fake = self.discriminator(fake_detach)
            d_real = self.discriminator(real)
            label_fake = torch.zeros_like(d_fake)
            label_real = torch.ones_like(d_real)
            if self.gan_type == 'GAN':
                loss_d \
                    = F.binary_cross_entropy_with_logits(d_fake, label_fake) \
                      + F.binary_cross_entropy_with_logits(d_real, label_real)
            elif self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand_like(fake)
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty
            elif self.gan_type == 'RAGAN':
                loss_d = (self.BCE(d_real-torch.mean(d_fake), label_real)
                          + self.BCE(d_fake-torch.mean(d_real), label_fake))/2

            # Discriminator update
            self.loss += loss_d.item()
            loss_d.backward()
            self.optimizer.step()

            if self.gan_type == 'WGAN':
                for p in self.discriminator.parameters():
                    p.data.clamp_(-1, 1)

        self.loss /= self.gan_k
        d_fake_for_g = self.discriminator(fake)
        d_real_for_g = self.discriminator(real)
        label_fake = torch.zeros_like(d_fake_for_g)
        label_real = torch.ones_like(d_real_for_g)
        if self.gan_type == 'GAN':
            loss_g = F.binary_cross_entropy_with_logits(
                d_fake_for_g, label_real
            )
        elif self.gan_type.find('WGAN') >= 0:
            loss_g = -d_fake_for_g.mean()
        elif self.gan_type == 'RAGAN':
            loss_g = (self.BCE(d_real_for_g - torch.mean(d_fake_for_g), label_fake)
                      + self.BCE(d_fake_for_g - torch.mean(d_real_for_g), label_fake)) / 2

        #loss_g += loss_gram * self.args.weight_gram
        # Generator loss
        return loss_g

    def state_dict(self, *args, **kwargs):
        state_discriminator = self.discriminator.state_dict(*args, **kwargs)
        state_optimizer = self.optimizer.state_dict()

        return dict(**state_discriminator, **state_optimizer)