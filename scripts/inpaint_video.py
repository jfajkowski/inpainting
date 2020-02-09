import glob
import os
import time

import torch
import torchvision.transforms as transforms
from PIL import Image
from scripts.train_baseline import Baseline
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.functional import to_pil_image

from inpainting.inpainting import FlowAndFillInpaintingAlgorithm
from inpainting.load import VideoDataset, DynamicMaskVideoDataset
from inpainting.pwcnet import Network
from inpainting.visualize import animate_sequence

batch_size = 1
# size = (256, 256)
size = (512, 768)

frame_dataset = VideoDataset(
    list(glob.glob('data/raw/video/DAVIS/JPEGImages/480p/breakdance')),
    frame_type='image',
    transform=transforms.Compose([
        transforms.Resize(size, interpolation=Image.BILINEAR),
        transforms.ToTensor()
    ]))
mask_dataset = VideoDataset(
    list(glob.glob('data/processed/video/DAVIS/Annotations_dilated/480p/breakdance')),
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
    flow_model = Network('models/pwcnet/network-default.pytorch').cuda().eval()
    # inpainting_algorithm = FlowInpaintingAlgorithm(flow_model)

    fill_model = Baseline.load_from_checkpoint(
        'models/baseline_unet/version_0/checkpoints/_ckpt_epoch_96.ckpt').generator.cuda().eval()
    # inpainting_algorithm = FillInpaintingAlgorithm(fill_model)

    inpainting_algorithm = FlowAndFillInpaintingAlgorithm(flow_model, fill_model)

    # First dry run
    for i in range(0):
        inpainting_algorithm.inpaint_online(
            torch.ones((batch_size, 3, *size)).cuda(),
            torch.ones((batch_size, 1, *size)).cuda()
        )

    data_iter = iter(data_loader)

    for sample in data_iter:
        frames, masks, _ = sample
        start = time.perf_counter()
        frames = list(map(lambda x: x.cuda(), frames))
        masks = list(map(lambda x: x.cuda(), masks))
        inpainting_algorithm.reset()
        frames_filled, masks_filled = inpainting_algorithm.inpaint(frames, masks)
        frames_filled = list(map(lambda x: x.cpu(), frames_filled))
        masks_filled = list(map(lambda x: x.cpu(), masks_filled))
        end = time.perf_counter()
        print(len(frames) / (end - start))

        frames = list(map(lambda x: x.cpu(), frames))
        masks = list(map(lambda x: x.cpu(), masks))

        target_directory = 'results'
        os.makedirs(target_directory, exist_ok=True)
        for i in range(batch_size):
            animate_sequence(
                [to_pil_image(f[i], mode='RGB') for f in frames],
                # [to_pil_image(m[i], mode='L') for m in masks],
                [to_pil_image(f[i], mode='RGB') for f in frames_filled],
                # [to_pil_image(m[i], mode='L') for m in masks_filled]
            ).save(f'{target_directory}/sequence.mp4', fps=24, dpi=300)
