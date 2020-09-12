# %%

import glob

import torch
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader

from inpainting import transforms
from inpainting.load import MergeDataset, ImageDataset
from inpainting.utils import normalize

# %%
images_dataset = ImageDataset(
    list(glob.glob(f'../data/raw/DAVIS/JPEGImages/flamingo')),
    'image'
)
masks_dataset = ImageDataset(
    list(glob.glob(f'../data/interim/DAVIS/Masks/flamingo')),
    'mask'
)
dataset = MergeDataset([images_dataset, masks_dataset], transform=transforms.ToTensor())
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# %%

image, mask = next(iter(data_loader))

#%%

_, _, w, h = image.size()
image = normalize(interpolate(image.cuda(), size=[w // 8 * 8, h // 8 * 8]))
mask = interpolate(mask.cuda(), size=[w // 8 * 8, h // 8 * 8])

#%%

image_b, image_c, image_h, image_w = image.size()
mask_b, mask_c, mask_h, mask_w = mask.size()

kernel_size = 3
kernel = torch.ones((kernel_size, kernel_size))
mask_weights = kernel.repeat((mask_c, 1, 1, 1)).to(image.device)

patches = torch.nn.functional.unfold(image, kernel.size(), padding=kernel_size // 2)
patches = patches.view(image_b, image_c, image_h * image_w, kernel_size, kernel_size)

i = 0

tmp_image = image[i]
tmp_mask = mask[i]
tmp_patches = patches[i]

valid = (1 - mask).view(1, mask_c, mask_h, mask_w)
neighbours = torch.nn.functional.conv2d(valid, mask_weights, padding=kernel_size // 2, groups=mask_b * mask_c)
neighbours = neighbours.view(mask_c, mask_h * mask_w)

tmp = tmp_mask.view(mask_c, mask_h * mask_w)
unknown_neighbours = neighbours[tmp.bool()]

patch_to_fill = torch.argmax(unknown_neighbours)

tmp = tmp.repeat(1, image_c, 1)
known_patches = patches[(1 - tmp).bool()]
unknown_patches = patches[tmp.bool()]
