import torch
import torch.nn as nn
from PIL import Image
from apex import amp
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image

from inpainting.load import ImageDataset, FileMaskDataset, InpaintingImageDataset
from inpainting.metrics import PSNR
from inpainting.model_attention import SelfAttention
from inpainting.model_attentionconv import AttentionConv
from inpainting.model_gatingconv import GatingConvolutionUNet
from inpainting.utils import mean_and_std, mask_tensor, denormalize


class Network(nn.Module):

    class Down(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride=2, padding=1)
            self.bn = nn.BatchNorm2d(out_channels)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            return torch.relu(x)

    class Up(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size):
            super().__init__()
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                           stride=2, padding=1, output_padding=1)
            self.bn = nn.BatchNorm2d(out_channels)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            return torch.relu(x)

    def __init__(self):
        super().__init__()
        self.e1 = Network.Down(4, 64, 3)
        self.e2 = Network.Down(64, 128, 3)
        self.e3 = Network.Down(128, 256, 3)

        self.d1 = Network.Up(256, 128, 3)
        self.d2 = Network.Up(128, 64, 3)
        self.d3 = Network.Up(64, 3, 3)

        self.out = nn.Conv2d(3, 3, 1)

    def forward(self, image, mask):
        x = torch.cat([image, mask], dim=1)

        x = self.e1(x)
        x = self.e2(x)
        x = self.e3(x)

        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)

        x = self.out(x)
        return torch.tanh(x)


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class DepthwiseConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride, padding, output_padding, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class SlimNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding = nn.Sequential(
            DepthwiseConv2d(4, 64, 3, stride=2, padding=1),
            DepthwiseConv2d(64, 128, 3, stride=2, padding=1),
            DepthwiseConv2d(128, 256, 3, stride=2, padding=1),
        )
        self.decoding = nn.Sequential(
            DepthwiseConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            DepthwiseConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            DepthwiseConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1),
        )
        self.out = nn.Conv2d(3, 3, 1)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.encoding(x)
        x = self.decoding(x)
        return torch.tanh(self.out(x))


image_transforms = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(*mean_and_std())
])
mask_transforms = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.NEAREST),
    transforms.ToTensor()
])
image_dataset = ImageDataset(['data/raw/image/SmallPlaces2/data_large'], transform=image_transforms)
mask_dataset = FileMaskDataset('data/raw/mask/qd_imd/train', transform=mask_transforms)
dataset = InpaintingImageDataset(image_dataset, mask_dataset)
data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

model = Network().cuda()
print(sum(p.numel() for p in model.parameters()))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model, optimizer = amp.initialize(model, optimizer, loss_scale=128)
criterion = nn.MSELoss()
metric = PSNR(1)

image, image_masked, image_filled = None, None, None
for e in range(1000):
    image, mask = next(iter(data_loader))
    image, mask = image.cuda(), mask.cuda()

    optimizer.zero_grad()

    image_masked = mask_tensor(image, mask)
    image_filled = model(image, mask)
    loss = criterion(image_filled, image)
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    # loss.backward()
    optimizer.step()

    accuracy = metric(denormalize(image), denormalize(image_filled))

    print(f'Epoch: {e} Loss: {loss} Accuracy: {accuracy}')

save_image(denormalize(image), 'results/image.png')
save_image(denormalize(image_masked), 'results/image_masked.png')
save_image(denormalize(image_filled), 'results/image_filled.png')
