from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PSNR(nn.Module):
    def __init__(self, max_val=1.0):
        super(PSNR, self).__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, image_1, image_2):
        mse = torch.mean((image_1.float() - image_2.float()) ** 2)

        if mse == 0:
            return torch.tensor(0)

        return self.max_val - 10 * torch.log(mse) / self.base10


class MAE(nn.Module):
    def __call__(self, image_1, image_2):
        return torch.mean(torch.abs(image_1.float() - image_2.float()))


class MSE(nn.Module):
    def __call__(self, image_1, image_2):
        return torch.mean(torch.pow(image_1.float() - image_2.float(), 2))


class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = SSIM.create_window(window_size, self.channel)

    def __call__(self, image_1, image_2):
        (_, channel, _, _) = image_1.size()

        if channel == self.channel and self.window.data.type() == image_1.data.type():
            window = self.window
        else:
            window = SSIM.create_window(self.window_size, channel)

            if image_1.is_cuda:
                window = window.cuda(image_1.get_device())
            window = window.type_as(image_1)

            self.window = window
            self.channel = channel

        return SSIM._ssim(image_1, image_2, window, self.window_size, channel, self.size_average)

    @staticmethod
    def create_window(window_size, channel):
        _1D_window = SSIM.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    @staticmethod
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    @staticmethod
    def _ssim(image_1, image_2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(image_1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(image_2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(image_1 * image_1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(image_2 * image_2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(image_1 * image_2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
