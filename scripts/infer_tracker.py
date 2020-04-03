import argparse
import glob

import torch
from PIL import Image
from torchvision.transforms import transforms as T
from tqdm import tqdm

from inpainting.external.algorithms import SiamMaskVideoTrackingAlgorithm
from inpainting.load import VideoDataset, DynamicMaskVideoDataset
from inpainting.utils import mask_to_bbox
from inpainting.visualize import save_frame, save_frames

parser = argparse.ArgumentParser()
parser.add_argument('--input-images-dir', type=str, default='data/processed/demo/InputImages')
parser.add_argument('--input-masks-dir', type=str, default='data/processed/demo/Masks')
parser.add_argument('--results-dir', type=str, default='results/demo/Tracker')
parser.add_argument('--size', type=int, nargs=2, default=(256, 256))
opt = parser.parse_args()


input_images_dataset = VideoDataset(
    list(glob.glob(f'{opt.input_images_dir}/*')),
    'image',
    transform=T.Resize(opt.size, Image.BILINEAR)
)
input_masks_dataset = VideoDataset(
    list(glob.glob(f'{opt.input_masks_dir}/*')),
    'mask',
    transform=T.Resize(opt.size, Image.NEAREST)
)
dataset = DynamicMaskVideoDataset(input_images_dataset, input_masks_dataset, transform=T.ToTensor())

with torch.no_grad():
    tracking_algorithm = SiamMaskVideoTrackingAlgorithm()

    for i, (input_images, input_masks) in enumerate(tqdm(dataset)):
        output_masks = []

        initial_image, initial_roi = input_images[0], mask_to_bbox(input_masks[0])
        tracking_algorithm.initialize(initial_image, initial_roi)
        for input_image in input_images:
            output_mask = tracking_algorithm.find_mask(input_image)
            output_masks.append(output_mask)

        save_frame(initial_image, f'{opt.results_dir}/Misc/{i}/initial_image_and_roi.jpg', 'image', initial_roi)
        save_frames(output_masks, f'{opt.results_dir}/OutputMasks/{i}', 'mask')
