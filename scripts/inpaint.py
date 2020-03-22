import glob
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from inpainting.utils import tensor_to_cv_image
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.functional import to_pil_image

from inpainting.external.algorithms import DeepFlowGuidedVideoInpaintingAlgorithm, DeepVideoInpaintingAlgorithm
from inpainting.load import VideoDataset, DynamicMaskVideoDataset
from inpainting.visualize import animate_sequence, save_video

batch_size = 1
size = (512, 512)
frame_rate = 24

frame_dataset = VideoDataset(
    list(glob.glob('data/raw/video/DAVIS/JPEGImages/480p/flamingo')),
    frame_type='image',
    transform=transforms.Compose([
        transforms.Resize(size, interpolation=Image.BILINEAR),
        transforms.ToTensor()
    ]))
mask_dataset = VideoDataset(
    list(glob.glob('data/processed/video/DAVIS/Annotations_dilated/480p/flamingo')),
    frame_type='mask',
    transform=transforms.Compose([
        transforms.Resize(size, interpolation=Image.NEAREST),
        transforms.ToTensor()
    ]))
dataset = DynamicMaskVideoDataset(frame_dataset, mask_dataset)
# mask_dataset = RectangleMaskDataset(
#     size[1], size[0],
#     (128 - 16, 128 - 16, 32, 32),
#     # '../data/raw/mask/demo',
#     # '../data/raw/mask/qd_imd/test',
#     transform=transform)
# dataset = StaticMaskVideoDataset(frame_dataset, mask_dataset)

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

with torch.no_grad():
    inpainting_algorithm = DeepVideoInpaintingAlgorithm()

    for sample in iter(data_loader):
        frames, masks, _ = sample
        frames = list(map(lambda x: x.cuda(), frames))
        masks = list(map(lambda x: x.cuda(), masks))
        inpainting_algorithm.reset()
        frames_filled = inpainting_algorithm.inpaint(frames, masks)
        frames_filled = list(map(lambda x: x.cpu(), frames_filled))

        frames = list(map(lambda x: x.cpu(), frames))
        masks = list(map(lambda x: x.cpu(), masks))
        target_directory = 'results'
        os.makedirs(target_directory, exist_ok=True)
        for i in range(batch_size):
            # save_video([tensor_to_cv_image(f[i]) for f in frames], f'{target_directory}/sequence.mp4', size, frame_rate)
            animate_sequence(
                [to_pil_image(f[i], mode='RGB') for f in frames],
                # [to_pil_image(m[i], mode='L') for m in masks],
                [to_pil_image(f[i], mode='RGB') for f in frames_filled],
                # [to_pil_image(m[i], mode='L') for m in masks_filled]
            ).save(f'{target_directory}/sequence.mp4', fps=24, dpi=300)
