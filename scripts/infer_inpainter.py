import argparse

import torch
from PIL import Image
from PIL.ImageOps import invert
from deepstab.infer import image_to_batch_tensor, batch_tensor_to_image

from deepstab.model_gatingconvolution import GatingConvolutionUNet
from deepstab.utils import cutout_mask

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str,
                    default='../models/20200122_gatingconvunet_gan/model_epoch_275_lr_0.0001.pth')
parser.add_argument('--source-image-path', type=str, default='../data/raw/image/demo/1080p.jpg')
parser.add_argument('--mask-path', type=str, default='../data/raw/mask/demo/mask.png')
parser.add_argument('--target-image-path', type=str, default='../result.jpg')
parser.add_argument('--image-size', type=int, nargs=2, default=[256, 256])
opt = parser.parse_args()
print(opt)

source_image = Image.open(opt.source_image_path).resize((1024, 1024))
mask = invert(Image.open(opt.mask_path).convert('L'))

state = torch.load(opt.model_path)
model = GatingConvolutionUNet().cuda().eval()
model.load_state_dict(state['generator'])

mask = mask.resize(source_image.size)
masked_image = cutout_mask(source_image, mask)

masked_image_tensor = image_to_batch_tensor(masked_image, channels=3)
mask_tensor = image_to_batch_tensor(mask, channels=1)

target_image_tensor = model(masked_image_tensor, mask_tensor)

target_image = batch_tensor_to_image(target_image_tensor)

target_image.save(opt.target_image_path)
