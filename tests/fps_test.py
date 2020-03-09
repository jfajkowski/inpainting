import time

import torch
from PIL import Image
from apex import amp
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from inpainting.load import ImageDataset, FileMaskDataset, InpaintingImageDataset
from inpainting.models.baseline import BaselineModel
from inpainting.models.depthwise_separable import DepthwiseSeparableModel
from inpainting.models.gated import GatedModel
from inpainting.utils import mean_and_std, mask_tensor

size = (256, 256)
epochs = 1
batch_size = 16

image_transforms = transforms.Compose([
    transforms.Resize(size, interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(*mean_and_std())
])
mask_transforms = transforms.Compose([
    transforms.Resize(size, interpolation=Image.NEAREST),
    transforms.ToTensor()
])
image_dataset = ImageDataset(['data/raw/image/SmallPlaces2/data_large'], transform=image_transforms)
mask_dataset = FileMaskDataset('data/raw/mask/demo', transform=mask_transforms)
dataset = InpaintingImageDataset(image_dataset, mask_dataset)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

models = [
    ('baseline_fp32', BaselineModel().cuda().eval()),
    ('baseline_fp16', amp.initialize(BaselineModel().cuda().eval())),
    ('depthwise_separable_fp32', DepthwiseSeparableModel().cuda().eval()),
    ('depthwise_separable_fp16', amp.initialize(DepthwiseSeparableModel().cuda().eval())),
    ('gated_fp32', GatedModel().cuda().eval()),
    ('gated_fp16', amp.initialize(GatedModel().cuda().eval())),
]

for name, model in models:
    times = []

    # Warm up
    for e in range(epochs):
        image_masked = model(torch.randn(batch_size, 3, *size).cuda(), torch.randn(batch_size, 1, *size).cuda())

    start = time.perf_counter()
    with torch.no_grad():
        image, mask = next(iter(data_loader))
        image, mask = image.cuda(), mask.cuda()
        for e in range(epochs):
            image_masked = mask_tensor(image, mask)
            image_filled = model(image, mask)
    end = time.perf_counter()

    print(
        f'Model: {name} '
        f'Parameters: {sum(p.numel() for p in model.parameters())} '
        f'Average FPS: {epochs / (end - start)}'
    )
