import time

import torch
from PIL import Image
from apex.amp import amp, scale_loss
from torch import nn
from torchvision.transforms import transforms

from inpainting.partialconv.loss import VGG16PartialLoss
from inpainting.utils import cutout_mask, extract_mask


class MobilenetV1Autoencoder(nn.Module):
    def __init__(self):
        super(MobilenetV1Autoencoder, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw_down(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw_up(inp, oup, scale):
            return nn.Sequential(
                nn.Upsample(scale_factor=scale),
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 16, 2),
            conv_dw_down(16, 32, 2),
            conv_dw_down(32, 64, 2),
            conv_dw_down(64, 128, 2),
            conv_dw_down(128, 256, 2),
            conv_dw_down(256, 512, 2),
            conv_dw_up(512, 256, 2),
            conv_dw_up(256, 128, 2),
            conv_dw_up(128, 64, 2),
            conv_dw_up(64, 32, 2),
            conv_dw_up(32, 16, 2),
            conv_dw_up(16, 3, 2),
            nn.Conv2d(3, 3, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, image_in, mask_in):
        return self.model(image_in), mask_in


if __name__ == '__main__':
    img = Image.open('../data/raw/DAVIS/JPEGImages/480p/breakdance/00000.jpg')
    mask = extract_mask(Image.open('../data/raw/DAVIS/Annotations_unsupervised/480p/breakdance/00000.png'), 1)
    img_masked = cutout_mask(img, mask)

    model = MobilenetV1Autoencoder().cuda()
    criterion = VGG16PartialLoss(vgg_path='../models/vgg16/vgg16-397923af.pth').cuda()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.001)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])


    def prepare(x, n=8):
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
        img_output = model(img_masked, mask)
        loss = criterion(img_output, img)
        with scale_loss(loss[0], optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

    invert_transform = transforms.Compose([
        transforms.ToPILImage()
    ])

    invert_transform(img_output[0, :, :, :].cpu().detach()).show()
    print(f'Took {time.time() - start}')
