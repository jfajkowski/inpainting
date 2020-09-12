import argparse
import glob
import torch

from tqdm import tqdm

import inpainting.transforms as T
from inpainting.load import VideoDataset, MergeDataset
from inpainting.models.algorithms import SingleFrameVideoInpaintingAlgorithm, StableVideoInpaintingAlgorithm, \
    FlowGuidedVideoInpaintingAlgorithm
from inpainting.visualize import save_frames

parser = argparse.ArgumentParser()
parser.add_argument('--input-images-dir', type=str, default='data/interim/demo/ResizedJPEGImages')
parser.add_argument('--input-masks-dir', type=str, default='data/interim/demo/ResizedMasks')
parser.add_argument('--results-dir', type=str, default='results/demo/Inpainter')
opt = parser.parse_args()

input_images_dataset = VideoDataset(
    list(glob.glob(f'{opt.input_images_dir}/*')),
    'image'
)
input_masks_dataset = VideoDataset(
    list(glob.glob(f'{opt.input_masks_dir}/*')),
    'mask'
)
dataset = MergeDataset([input_images_dataset, input_masks_dataset], transform=T.ToTensor())

with torch.no_grad():
    inpainting_algorithm = StableVideoInpaintingAlgorithm()

    for i, (input_images, input_masks) in enumerate(tqdm(dataset)):
        output_images = []

        inpainting_algorithm.initialize()
        for input_image, input_mask in list(zip(input_images, input_masks)):
            input_image = input_image.unsqueeze(0).cuda()
            input_mask = input_mask.unsqueeze(0).cuda()
            output_image = inpainting_algorithm.inpaint_online(input_image, input_mask).squeeze().cpu()
            output_images.append(output_image)

        save_frames(output_images, f'{opt.results_dir}/OutputImages/{i:05d}', 'image')
