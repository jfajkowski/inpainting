import argparse
import glob
from os.path import basename

import torch
from tqdm import tqdm

from inpainting import transforms
from inpainting.algorithms import VideoTrackingAlgorithm
from inpainting.load import SequenceDataset, MergeDataset
from inpainting.save import save_frame, save_frames
from inpainting.utils import mask_to_bbox

parser = argparse.ArgumentParser()
parser.add_argument('--input-images-dir', type=str, default='data/processed/demo/InputImages')
parser.add_argument('--input-masks-dir', type=str, default='data/processed/demo/Masks')
parser.add_argument('--results-dir', type=str, default='results/demo/Tracker')
opt = parser.parse_args()


input_images_dirs = list(sorted(glob.glob(f'{opt.input_images_dir}/*')))
input_masks_dirs = list(sorted(glob.glob(f'{opt.input_masks_dir}/*')))
sequence_names = list(map(basename, input_images_dirs))

input_images_dataset = SequenceDataset(
    input_images_dirs,
    'image'
)
input_masks_dataset = SequenceDataset(
    input_masks_dirs,
    'mask'
)
dataset = MergeDataset([input_images_dataset, input_masks_dataset], transform=transforms.ToTensor())

with torch.no_grad():
    tracking_algorithm = VideoTrackingAlgorithm()

    for sequence_name, (input_images, input_masks) in tqdm(zip(sequence_names, dataset), desc='Tracking',
                                                           unit='sequence', total=len(sequence_names)):
        output_masks = []

        initial_image, initial_roi = input_images[0], mask_to_bbox(input_masks[0])
        tracking_algorithm.initialize(initial_image.cuda(), initial_roi)
        for input_image in input_images:
            output_mask = tracking_algorithm.track_online(input_image.cuda())
            output_masks.append(output_mask)

        save_frame(initial_image, f'{opt.results_dir}/Initialization/{sequence_name}/initialization.png', 'image',
                   initial_roi)
        save_frames(output_masks, f'{opt.results_dir}/Masks/{sequence_name}', 'mask')
