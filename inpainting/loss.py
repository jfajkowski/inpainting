from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models

from inpainting.utils import denormalize, normalize


class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = denormalize(X)
        X = normalize(X, mode='imagenet')
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


def _gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    # gram = features.bmm(features_t) / (ch * h * w)
    input = torch.zeros(b, ch, ch).type(features.type())
    gram = torch.baddbmm(input, features, features_t, beta=0, alpha=1. / (ch * h * w), out=None)
    return gram


def _total_variation_loss(image):
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
           torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class ReconstructionLoss(nn.Module):

    def __init__(self, pixel_loss_weight=0.5, content_loss_weight=0.2, style_loss_weight=1000, tv_loss_weight=0.1):
        super(ReconstructionLoss, self).__init__()
        self.vgg = Vgg16() if content_loss_weight or style_loss_weight else None
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

        self.pixel_loss_weight = pixel_loss_weight
        self.content_loss_weight = content_loss_weight
        self.style_loss_weight = style_loss_weight
        self.tv_loss_weight = tv_loss_weight

    def forward(self, output, target):
        pixel_loss, content_loss, style_loss, tv_loss = list(map(torch.tensor, [0.0, 0.0, 0.0, 0.0]))

        if self.pixel_loss_weight:
            pixel_loss = self.l1(output, target)

        if self.vgg:
            output_features = self.vgg(output)
            target_features = self.vgg(target)

            if self.content_loss_weight:
                content_loss = self.mse(output_features.relu3_3, target_features.relu3_3)

            if self.style_loss_weight:
                style_loss = sum(
                    [self.mse(_gram_matrix(i), _gram_matrix(t)) for i, t in zip(output_features, target_features)])

        if self.tv_loss_weight:
            tv_loss = _total_variation_loss(output)

        pixel_loss *= self.pixel_loss_weight
        content_loss *= self.content_loss_weight
        style_loss *= self.style_loss_weight
        tv_loss *= self.tv_loss_weight

        loss_output = namedtuple("LossOutput", ['combined_loss', 'pixel_loss', 'content_loss', 'style_loss', 'tv_loss'])
        return loss_output(
            pixel_loss + content_loss + style_loss + tv_loss,
            pixel_loss,
            content_loss,
            style_loss,
            tv_loss
        )
