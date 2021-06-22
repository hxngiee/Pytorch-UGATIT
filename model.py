from layer import *

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

class ResNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='inorm', nblk=6, img_size=256):
        assert(nblk>=0)
        super(ResNet, self).__init__()
        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm
        self.nblk = nblk

        # 210621 Check norm working correctly
        self.bias = False if norm == 'bnorm' else True

        encblk = []
        encblk += [Padding(3, padding_mode='reflection')]
        encblk += [CNR2d(nch_in=nch_in, nch_out=nch_ker, kernel_size=7, stride=1, padding=0, norm=norm, relu=0.0)]

        # Down-Sampling
        nds = 2
        for i in range(nds):
            mult = 2**i
            encblk += [Padding(1, padding_mode='reflection')]
            encblk += [CNR2d(nch_ker*mult, nch_ker*mult, kernel_size=3, stride=2, padding=0, norm=norm, relu=0.0)]

        # Down-Sample Residual Blocks
        mult = 2**nds
        for i in range(nblk):
            encblk += [ResBlock(nch_ker*mult,nch_ker*mult)]

        # Class Activation Map
        self.gap_fc = Linear(nch_ker*mult, 1)
        self.gmp_fc = Linear(nch_ker*mult, 1)
        self.conv1x1 = Conv2d(nch_ker*mult*2, nch_ker*mult, kernel_size=1, stride=1, bias=True)
        self.relu = ReLU(0.0)

## How to get imgae size in model.py
        # Gamma, Beta Block
        FC = []
        FC += [Linear(img_size // mult * img_size // mult * nch_ker * mult, nch_ker*mult)]
        FC += [ReLU(0.0)]
        FC += [Linear(nch_ker*mult, nch_ker*mult)]
        FC += [ReLU(0.0)]
        self.gamma = Linear(nch_ker*mult, nch_ker*mult)
        self.beta = Linear(nch_ker*mult, nch_ker*mult)

## setattr
        # Up-Sample Residual Blocks
        for i in range(nblk):
            setattr(self, 'UpBlock1_' + str(i+1), Ada_ResBlock(nch_ker*mult))

        decblk = []
        for i in range(nds):
            mult = 2**(nds - i)
            decblk += [nn.Upsample(scale_factor=2, mode='nearest'),
                       Padding(1,'reflection')
                       Conv2d(nch_ker*mult, int(nch_ker*mult/2), kernel_size=3, stride=1, padding=0)
                       ILN(int(nch_ker*mult/2)),
                       ReLU(0.0)]
        decblk += [Padding(3,'reflection'),
                   Conv2d(nch_ker, nch_out, kernel_size=7, stride=1, padding=0),
                   nn.Tanh()]

        self.encblk = nn.Sequential(*encblk)
        self.FC = nn.Sequential(*FC)
        self.decblk = nn.Sequential(*decblk)

    def forward(self, x):
        x = self.encblk(x)
## adaptive_avg_pool
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)
        
        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        
        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))
## keepdim
        heatmap = torch.sum(x, dim=1, keepdim=True)
        
        x_ = self.FC(x.view(x.shape[0], -1))
        gamma, beta = self.gamma(x_), self.beta(x_)

## getattr()
        for i in range(self.nblk):
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
        out = self.decblk(x)
        return out, cam_logit, heatmap
    
    
# class ResNet(nn.Module):
#     def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm', nblk=6):
#         super(ResNet, self).__init__()
#
#         self.nch_in = nch_in
#         self.nch_out = nch_out
#         self.nch_ker = nch_ker
#         self.norm = norm
#         self.nblk = nblk
#
#         if norm == 'bnorm':
#             self.bias = False
#         else:
#             self.bias = True
#
#         self.enc1 = CNR2d(self.nch_in,      1 * self.nch_ker, kernel_size=7, stride=1, padding=3, norm=self.norm, relu=0.0)
#
#         self.enc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)
#
#         self.enc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)
#
#         if self.nblk:
#             res = []
#
#             for i in range(self.nblk):
#                 res += [ResBlock(4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0, padding_mode='reflection')]
#
#             self.res = nn.Sequential(*res)
#
#         self.dec3 = DECNR2d(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)
#
#         self.dec2 = DECNR2d(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)
#
#         self.dec1 = CNR2d(1 * self.nch_ker, self.nch_out, kernel_size=7, stride=1, padding=3, norm=[], relu=[], bias=False)
#
#     def forward(self, x):
#         x = self.enc1(x)
#         x = self.enc2(x)
#         x = self.enc3(x)
#
#         if self.nblk:
#             x = self.res(x)
#
#         x = self.dec3(x)
#         x = self.dec2(x)
#         x = self.dec1(x)
#
#         x = torch.tanh(x)
#
#         return x


##
class Discriminator(nn.Module):
    def __init__(self, nch_in, nch_ker=64, n_layers=5):
        super(Discriminator, self).__init__()
        encblk = [Padding(1,'reflection'),
                  nn.utils.spectral_norm(
                      Conv2d(nch_in, nch_ker, kernel_size=4, stride=2, padding=0)),
                  ReLU(0.2)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i-1)
            encblk += [Padding(1,'reflection'),
                       nn.utils.spectral_norm(
                           Conv2d(nch_ker*mult, nch_ker*mult*2, kernel_size=4, stride=2, padding=0)),
                       ReLU(0.2)]

## -3?
        mult = 2 ** (n_layers - 2 - 1)
        encblk += [Padding(1,'reflection'),
                   nn.utils.spectral_norm(
                       Conv2d(nch_ker*mult, nch_ker*mult*2, kernel_size=4, stride=1, padding=0)),
                   ReLU(0.2)]

        mult = 2 ** (n_layers -2)
        self.gap_fc = nn.utils.spectral_norm(Linear(nch_ker*mult,1))
        self.gmp_fc = nn.utils.spectral_norm(Linear(nch_ker*mult,1))
        self.conv1x1 = Conv2d(nch_ker*mult*2, nch_ker*mult, kernel_size=1, stride=1)
        self.leaky_relu = ReLU(0.2)

        self.pad = Padding(1,'reflection')
        self.conv = nn.utils.spectral_norm(Conv2d(nch_ker*mult, 1, kernel_size=4, stride=1, padding=0))
        self.encblk = nn.Sequential(*encblk)

    def forward(self, x):
        x = self.encblk(x)

        gap = torch.nn.functional.adaptive_avg_pool2d(x,1)
        gap_logit = self.gap_fc(gap.view(x.shape[0],-1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0],-1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logt = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap,gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))
        heatmap = torch.sum(x, dim=1, keepdim=True)

        x = self.pad(x)
        out = self.conv(x)
        return out, cam_logt, heatmap


class Discriminator(nn.Module):
    def __init__(self, nch_in, nch_ker=64, norm='bnorm'):
        super(Discriminator, self).__init__()

        self.nch_in = nch_in
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        # dsc1 : 256 x 256 x 3 -> 128 x 128 x 64
        # dsc2 : 128 x 128 x 64 -> 64 x 64 x 128
        # dsc3 : 64 x 64 x 128 -> 32 x 32 x 256
        # dsc4 : 32 x 32 x 256 -> 32 x 32 x 512
        # dsc5 : 32 x 32 x 512 -> 32 x 32 x 1

        self.dsc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc5 = CNR2d(8 * self.nch_ker, 1,                kernel_size=4, stride=1, padding=1, norm=[],        relu=[], bias=False)

        # self.dsc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=1, padding=1, norm=[], relu=0.2)
        # self.dsc5 = CNR2d(8 * self.nch_ker, 1,                kernel_size=4, stride=1, padding=1, norm=[], relu=[], bias=False)

    def forward(self, x):

        x = self.dsc1(x)
        x = self.dsc2(x)
        x = self.dsc3(x)
        x = self.dsc4(x)
        x = self.dsc5(x)

        # x = torch.sigmoid(x)

        return x


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], gpu_mode=None):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    torch.cuda.set_device(gpu_ids)
    net.to(gpu_ids)
    if gpu_mode == 'multi':
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpu_ids], find_unused_parameters=True)  # Multi GPU

    init_weights(net, init_type, init_gain=init_gain)
    return net
