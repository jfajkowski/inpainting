import time

import torch
import torch.nn.functional as F
from PIL import Image
from apex.amp import amp, scale_loss
from torch import nn
from torchvision.transforms import transforms

from deepstab.partialconv import PartialConv2d
from deepstab.partialconv.loss import VGG16PartialLoss
from deepstab.utils import extract_mask, cutout_mask


class PartialConvolutionUNet(nn.Module):
    def __init__(self):
        super(PartialConvolutionUNet, self).__init__()

        self.e_conv1 = Down(3, 64, 7, bn=False)
        self.e_conv2 = Down(64, 128, 5)
        self.e_conv3 = Down(128, 256, 5)
        self.e_conv4 = Down(256, 512, 3)
        self.e_conv5 = Down(512, 512, 3)
        self.e_conv6 = Down(512, 512, 3)
        self.e_conv7 = Down(512, 512, 3)

        self.d_conv7 = Up(1024, 512, 3)
        self.d_conv6 = Up(1024, 512, 3)
        self.d_conv5 = Up(1024, 512, 3)
        self.d_conv4 = Up(768, 256, 3)
        self.d_conv3 = Up(384, 128, 3)
        self.d_conv2 = Up(192, 64, 3)
        self.d_conv1 = Up(67, 3, 3, bn=False)
        self.out = nn.Conv2d(3, 3, 1)

    def forward(self, image_in, mask_in):
        e_image_out1, e_mask_out1 = self.e_conv1(image_in, mask_in)
        e_image_out2, e_mask_out2 = self.e_conv2(e_image_out1, e_mask_out1)
        e_image_out3, e_mask_out3 = self.e_conv3(e_image_out2, e_mask_out2)
        e_image_out4, e_mask_out4 = self.e_conv4(e_image_out3, e_mask_out3)
        e_image_out5, e_mask_out5 = self.e_conv5(e_image_out4, e_mask_out4)
        e_image_out6, e_mask_out6 = self.e_conv6(e_image_out5, e_mask_out5)
        e_image_out7, e_mask_out7 = self.e_conv7(e_image_out6, e_mask_out6)

        d_image_out7, d_mask_out7 = self.d_conv7(e_image_out7, e_mask_out7, e_image_out6, e_mask_out6)
        d_image_out6, d_mask_out6 = self.d_conv6(d_image_out7, d_mask_out7, e_image_out5, e_mask_out5)
        d_image_out5, d_mask_out5 = self.d_conv5(d_image_out6, d_mask_out6, e_image_out4, e_mask_out4)
        d_image_out4, d_mask_out4 = self.d_conv4(d_image_out5, d_mask_out5, e_image_out3, e_mask_out3)
        d_image_out3, d_mask_out3 = self.d_conv3(d_image_out4, d_mask_out4, e_image_out2, e_mask_out2)
        d_image_out2, d_mask_out2 = self.d_conv2(d_image_out3, d_mask_out3, e_image_out1, e_mask_out1)
        d_image_out1, d_mask_out1 = self.d_conv1(d_image_out2, d_mask_out2, image_in, mask_in)
        image_out, mask_out = torch.sigmoid(self.out(d_image_out1)), d_mask_out1

        return image_out


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bn=True):
        super(Down, self).__init__()
        self.conv = PartialConv2d(in_channels, out_channels, kernel_size, stride=2, padding=int(kernel_size / 2),
                                  padding_mode='same',
                                  multi_channel=True, return_mask=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

    def forward(self, image_in, mask_in):
        image_out, mask_out = self.conv(image_in, mask_in)
        if self.bn:
            image_out = self.bn(image_out)
        image_out = F.relu(image_out)
        return image_out, mask_out


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bn=True):
        super(Up, self).__init__()
        self.conv = PartialConv2d(in_channels, out_channels, kernel_size, stride=1, padding=int(kernel_size / 2),
                                  padding_mode='same',
                                  multi_channel=True, return_mask=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

    def forward(self, image_in1, mask_in1, image_in2, mask_in2):
        image_up = F.interpolate(image_in1, scale_factor=2)
        mask_up = F.interpolate(mask_in1, scale_factor=2)
        image_in = torch.cat([image_up, image_in2], dim=1)
        mask_in = torch.cat([mask_up, mask_in2], dim=1)
        image_out, mask_out = self.conv(image_in, mask_in)
        if self.bn:
            image_out = self.bn(image_out)
        image_out = F.leaky_relu(image_out, negative_slope=0.2)
        return image_out, mask_out


if __name__ == '__main__':
    img = Image.open('../data/raw/DAVIS/JPEGImages/480p/breakdance/00000.jpg')
    mask = extract_mask(Image.open('../data/raw/DAVIS/Annotations_unsupervised/480p/breakdance/00000.png'), 1)
    img_masked = cutout_mask(img, mask)

    model = PartialConvolutionUNet().cuda()
    criterion = VGG16PartialLoss(vgg_path='../models/vgg16/vgg16-397923af.pth').cuda()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.001)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])


    def prepare(x, n=4):
        return torch.stack((x,) * n).view(n, 3, 512, 512).cuda()


    img = prepare(transform(img))
    mask = prepare(torch.stack((transform(mask),) * 3))
    img_masked = prepare(transform(img_masked))

    opt_level = 'O1'
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

    start = time.time()
    for epoch in range(1000):
        print(f'Epoch: {epoch + 1}')
        optimizer.zero_grad()
        img_output, _ = model(img_masked, mask)
        loss = criterion(img_output, img)
        with scale_loss(loss[0], optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

    invert_transform = transforms.Compose([
        transforms.ToPILImage()
    ])

    invert_transform(img_output[0, :, :, :].cpu().detach()).show()
    print(f'Took {time.time() - start}')
