import argparse

import torch
from PIL import Image
from apex import amp
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from inpainting.load import ImageDataset, FileMaskDataset, InpaintingImageDataset
from inpainting.external.baseline import BaselineModel
from inpainting.utils import mean_and_std, mask_tensor

parser = argparse.ArgumentParser()
parser.add_argument('opt_level', type=str, default='O1')
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--size', type=int, default=(256, 256))
opt = parser.parse_args()
print(opt)

image_transforms = transforms.Compose([
    transforms.Resize(opt.size, interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(*mean_and_std())
])
mask_transforms = transforms.Compose([
    transforms.Resize(opt.size, interpolation=Image.NEAREST),
    transforms.ToTensor()
])
image_dataset = ImageDataset(['data/raw/image/SmallPlaces2/data_large'], transform=image_transforms)
mask_dataset = FileMaskDataset('data/raw/mask/demo', transform=mask_transforms)
dataset = InpaintingImageDataset(image_dataset, mask_dataset)
data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

model = amp.initialize(BaselineModel().cuda().eval(), opt_level=opt.opt_level)
with torch.no_grad():
    image, mask = next(iter(data_loader))
    image, mask = image.cuda(), mask.cuda()
    image_masked = mask_tensor(image, mask)
    image_filled = model(image, mask)
