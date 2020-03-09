import argparse

import torch
import torch.nn as nn
from apex.amp import amp

parser = argparse.ArgumentParser()
parser.add_argument('opt_level', type=str, default='O1')
parser.add_argument('--B', type=int, default=1)
parser.add_argument('--C', type=int, default=3)
parser.add_argument('--H', type=int, default=256)
parser.add_argument('--W', type=int, default=256)
opt = parser.parse_args()
print(opt)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(opt.C, 64, 3),
            nn.Conv2d(64, 64, 3),
            nn.Conv2d(64, opt.C, 3),
        )

    def forward(self, x):
        return self.net(x)


if opt.opt_level == 'HALF':
    model = Network().half().cuda().eval()
    model(torch.randn((opt.B, opt.C, opt.H, opt.W)).half().cuda())
elif opt.opt_level == 'FLOAT':
    model = Network().cuda().eval()
    model(torch.randn((opt.B, opt.C, opt.H, opt.W)).cuda())
else:
    model = amp.initialize(Network().cuda().eval(), opt_level=opt.opt_level)
    model(torch.randn((opt.B, opt.C, opt.H, opt.W)).cuda())
