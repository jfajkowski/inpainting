import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.transforms import transforms

from deepstab.utils import extract_mask, cutout_mask


class GatingConvolutionUNet(nn.Module):
    def __init__(self):
        super(GatingConvolutionUNet, self).__init__()

        self.e_conv1 = GatingConvolutionDown(4, 64, 7, bn=False)
        self.e_conv2 = GatingConvolutionDown(64, 128, 5)
        self.e_conv3 = GatingConvolutionDown(128, 256, 5)
        self.e_conv4 = GatingConvolutionDown(256, 512, 3)
        self.e_conv5 = GatingConvolutionDown(512, 512, 3)
        self.e_conv6 = GatingConvolutionDown(512, 512, 3)
        self.e_conv7 = GatingConvolutionDown(512, 512, 3)

        self.d_conv7 = GatingConvolutionUp(1024, 512, 3)
        self.d_conv6 = GatingConvolutionUp(1024, 512, 3)
        self.d_conv5 = GatingConvolutionUp(1024, 512, 3)
        self.d_conv4 = GatingConvolutionUp(768, 256, 3)
        self.d_conv3 = GatingConvolutionUp(384, 128, 3)
        self.d_conv2 = GatingConvolutionUp(192, 64, 3)
        self.d_conv1 = GatingConvolutionUp(67, 3, 3, bn=False)
        self.out = nn.Conv2d(3, 3, 1)

    def forward(self, image_in, mask_in):
        x = torch.cat([image_in, mask_in], 1)
        e_image_out1 = self.e_conv1(x)
        e_image_out2 = self.e_conv2(e_image_out1)
        e_image_out3 = self.e_conv3(e_image_out2)
        e_image_out4 = self.e_conv4(e_image_out3)
        e_image_out5 = self.e_conv5(e_image_out4)
        e_image_out6 = self.e_conv6(e_image_out5)
        e_image_out7 = self.e_conv7(e_image_out6)

        d_image_out7 = self.d_conv7(e_image_out7, e_image_out6)
        d_image_out6 = self.d_conv6(d_image_out7, e_image_out5)
        d_image_out5 = self.d_conv5(d_image_out6, e_image_out4)
        d_image_out4 = self.d_conv4(d_image_out5, e_image_out3)
        d_image_out3 = self.d_conv3(d_image_out4, e_image_out2)
        d_image_out2 = self.d_conv2(d_image_out3, e_image_out1)
        d_image_out1 = self.d_conv1(d_image_out2, image_in)

        return torch.tanh(self.out(d_image_out1))


class GatingConvolutionAutoencoder(nn.Module):
    def __init__(self):
        super(GatingConvolutionAutoencoder, self).__init__()

        self.model = nn.Sequential(
            GatingConvolutionDown(4, 64, 7, bn=False),
            GatingConvolutionDown(64, 128, 5),
            GatingConvolutionDown(128, 256, 5),
            GatingConvolutionDown(256, 512, 3),
            GatingConvolutionDown(512, 512, 3),
            GatingConvolutionDown(512, 512, 3),
            GatingConvolutionDown(512, 512, 3),
            GatingConvolutionUp(512, 512, 3),
            GatingConvolutionUp(512, 512, 3),
            GatingConvolutionUp(512, 512, 3),
            GatingConvolutionUp(512, 256, 3),
            GatingConvolutionUp(256, 128, 3),
            GatingConvolutionUp(128, 64, 3),
            GatingConvolutionUp(64, 3, 3, bn=False),
            nn.Conv2d(3, 3, 1)
        )

    def forward(self, image_in, mask_in):
        x = torch.cat([image_in, mask_in], 1)
        return torch.tanh(self.model(x))


class GatingConvolutionDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bn=True):
        super(GatingConvolutionDown, self).__init__()
        self.gating_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=int(kernel_size / 2),
                                     padding_mode='same')
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=int(kernel_size / 2),
                                      padding_mode='same')
        self.bn = bn
        if bn:
            self.gating_bn = nn.BatchNorm2d(out_channels)
            self.feature_bn = nn.BatchNorm2d(out_channels)
        else:
            self.gating_bn = None
            self.feature_bn = None

    def forward(self, image_in):
        gating_x = self.gating_conv(image_in)
        feature_x = self.feature_conv(image_in)
        if self.bn:
            gating_x = self.gating_bn(gating_x)
            feature_x = self.feature_bn(feature_x)
        gating_x = torch.sigmoid(gating_x)
        feature_x = torch.relu(feature_x)
        return gating_x * feature_x


class GatingConvolutionUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bn=True):
        super(GatingConvolutionUp, self).__init__()
        self.gating_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=int(kernel_size / 2),
                                     padding_mode='same')
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=int(kernel_size / 2),
                                      padding_mode='same')
        self.bn = bn
        if bn:
            self.gating_bn = nn.BatchNorm2d(out_channels)
            self.feature_bn = nn.BatchNorm2d(out_channels)
        else:
            self.gating_bn = None
            self.feature_bn = None

    def forward(self, image_in1, image_in2=None):
        image_up = F.interpolate(image_in1, scale_factor=2)
        if image_in2 is None:
            image_in = image_up
        else:
            image_in = torch.cat([image_up, image_in2], dim=1)
        gating_x = self.gating_conv(image_in)
        feature_x = self.feature_conv(image_in)
        if self.bn:
            gating_x = self.gating_bn(gating_x)
            feature_x = self.feature_bn(feature_x)
        gating_x = torch.sigmoid(gating_x)
        feature_x = torch.relu(feature_x)
        return gating_x * feature_x


if __name__ == '__main__':
    image = Image.open('../data/raw/video/DAVIS/JPEGImages/480p/breakdance/00000.jpg')
    mask = extract_mask(Image.open('../data/raw/video/DAVIS/Annotations_unsupervised/480p/breakdance/00000.png'), 1)
    masked_image = cutout_mask(image, mask)

    generator = GatingConvolutionAutoencoder().cuda()
    discriminator = Discriminator().cuda()
    criterion = torch.nn.BCELoss().cuda()

    generator_optimizer = torch.optim.SGD([p for p in generator.parameters() if p.requires_grad], lr=0.001)
    discriminator_optimizer = torch.optim.SGD([p for p in discriminator.parameters() if p.requires_grad], lr=0.001)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])


    def prepare(x, n=4):
        return torch.stack((x,) * n).view(n, -1, 256, 256).cuda()


    image = prepare(transform(image))
    mask = prepare(transform(mask))
    masked_image = prepare(transform(masked_image))

    for epoch in range(1000):
        print(f'Epoch: {epoch + 1}')

        # Adversarial ground truths
        valid = torch.ones((image.shape[0], 1)).cuda()
        fake = torch.zeros((image.shape[0], 1)).cuda()

        # Generator training
        generator_optimizer.zero_grad()
        output_image = generator(masked_image, mask)
        loss = criterion(discriminator(output_image), valid)
        loss.backward()
        generator_optimizer.step()

        # Discriminator training
        discriminator_optimizer.zero_grad()
        real_loss = criterion(discriminator(image), valid)
        fake_loss = criterion(discriminator(output_image.detach()), fake)
        loss = (real_loss + fake_loss) / 2
        loss.backward()
        discriminator_optimizer.step()

    invert_transform = transforms.Compose([
        transforms.ToPILImage()
    ])

    invert_transform(output_image[0, :, :, :].cpu().detach()).show()
