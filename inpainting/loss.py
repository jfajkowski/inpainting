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

        loss_output = namedtuple("ReconstructionLossOutput",
                                 ['combined_loss', 'pixel_loss', 'content_loss', 'style_loss', 'tv_loss'])
        return loss_output(
            pixel_loss + content_loss + style_loss + tv_loss,
            pixel_loss,
            content_loss,
            style_loss,
            tv_loss
        )


class AdversarialGeneratorCriterion(nn.Module):

    def __init__(self, criterion_type=''):
        super(AdversarialGeneratorCriterion, self).__init__()
        self.criterion_type = criterion_type
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        # Compute G loss with fake images & real labels
        if self.criterion_type == 'dcgan':
            g_loss = self.criterion(x, torch.ones_like(x))
        else:
            g_loss = -x.mean()
        return g_loss


class AdversarialDiscriminatorCriterion(nn.Module):

    def __init__(self, criterion_type=''):
        super(AdversarialDiscriminatorCriterion, self).__init__()
        self.criterion_type = criterion_type
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, real, fake):
        real_loss = self._forward_discriminator_real(real)
        fake_loss = self._forward_discriminator_fake(fake)
        loss_output = namedtuple("AdversarialDiscriminatorLossOutput",
                                 ['combined_loss', 'real_loss', 'fake_loss'])
        return loss_output(
            real_loss + fake_loss,
            real_loss,
            fake_loss
        )

    def _forward_discriminator_real(self, x):
        # Compute D loss with real images & real labels
        if self.criterion_type == 'hinge':
            d_loss_real = torch.relu(torch.ones_like(x) - x).mean()
        elif self.criterion_type == 'wgan_gp':
            d_loss_real = -x.mean()
        else:
            d_loss_real = self.criterion(x, torch.ones_like(x))
        return d_loss_real

    def _forward_discriminator_fake(self, x):
        # Compute D loss with fake images & real labels
        if self.criterion_type == 'hinge':
            d_loss_fake = torch.relu(torch.ones_like(x) + x).mean()
        elif self.criterion_type == 'dcgan':
            d_loss_fake = self.criterion(x, torch.zeros_like(x))
        else:
            d_loss_fake = x.mean()

        # # If WGAN_GP, compute GP and add to D loss
        # if self.config.adv_loss == 'wgan_gp':
        #     d_loss_gp = self.config.lambda_gp * self._compute_gradient_penalty(real_images, real_labels,
        #                                                                        fake_images.detach())
        #     d_loss_fake += d_loss_gp

        return d_loss_fake

    # def _compute_gradient_penalty(self, real_images, real_labels, fake_images):
    #     # Compute gradient penalty
    #     alpha = torch.rand(real_images.size(0), 1, 1, 1).expand_as(real_images).to(device)
    #     interpolated = torch.tensor(alpha * real_images + (1 - alpha) * fake_images, requires_grad=True)
    #     out = self.D(interpolated, real_labels)
    #     exp_grad = torch.ones(out.size()).to(device)
    #     grad = torch.autograd.grad(outputs=out,
    #                                inputs=interpolated,
    #                                grad_outputs=exp_grad,
    #                                retain_graph=True,
    #                                create_graph=True,
    #                                only_inputs=True)[0]
    #     grad = grad.view(grad.size(0), -1)
    #     grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    #     d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)
    #     return d_loss_gp
