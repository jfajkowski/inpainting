import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.transforms import transforms

from deepstab.utils import extract_mask, cutout_mask


class Inpainter(nn.Module):
    def __init__(self):
        super(Inpainter, self).__init__()

        self.model = nn.Sequential(
            Down(4, 64, 7, bn=False),
            Down(64, 128, 5),
            Down(128, 256, 5),
            Down(256, 512, 3),
            Down(512, 512, 3),
            Down(512, 512, 3),
            Down(512, 512, 3),
            Up(512, 512, 3),
            Up(512, 512, 3),
            Up(512, 512, 3),
            Up(512, 256, 3),
            Up(256, 128, 3),
            Up(128, 64, 3),
            Up(64, 3, 3, bn=False)
        )

    def forward(self, image_in, mask_in):
        x = torch.cat([image_in, mask_in], 1)
        return self.model(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bn=True):
        super(Down, self).__init__()
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

    def forward(self, x):
        gating_x = self.gating_conv(x)
        feature_x = self.feature_conv(x)
        if self.bn:
            gating_x = self.gating_bn(gating_x)
            feature_x = self.feature_bn(feature_x)
        gating_x = torch.sigmoid(gating_x)
        feature_x = torch.relu(feature_x)
        return gating_x * feature_x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bn=True):
        super(Up, self).__init__()
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

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        gating_x = self.gating_conv(x)
        feature_x = self.feature_conv(x)
        if self.bn:
            gating_x = self.gating_bn(gating_x)
            feature_x = self.feature_bn(feature_x)
        gating_x = torch.sigmoid(gating_x)
        feature_x = torch.relu(feature_x)
        return gating_x * feature_x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, padding_mode='same'),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, padding_mode='same'),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, padding_mode='same'),
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2, padding_mode='same'),
            nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2, padding_mode='same')
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


if __name__ == '__main__':
    image = Image.open('../data/raw/video/DAVIS/JPEGImages/480p/breakdance/00000.jpg')
    mask = extract_mask(Image.open('../data/raw/video/DAVIS/Annotations_unsupervised/480p/breakdance/00000.png'), 1)
    masked_image = cutout_mask(image, mask)

    generator = Inpainter().cuda()
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
