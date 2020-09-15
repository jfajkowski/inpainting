import argparse
import glob
from os.path import basename

import torch

from tqdm import tqdm

from inpainting import transforms
from inpainting.load import SequenceDataset, MergeDataset
from inpainting.algorithms import SingleFrameVideoInpaintingAlgorithm, FlowGuidedVideoInpaintingAlgorithm
from inpainting.visualize import save_frames

parser = argparse.ArgumentParser()
parser.add_argument('--input-images-dir', type=str, default='data/processed/demo/InputImages')
parser.add_argument('--input-masks-dir', type=str, default='data/processed/demo/Masks')
parser.add_argument('--results-dir', type=str, default='results/demo/Inpainter')
parser.add_argument('--flow-guided', type=bool, default=True)
parser.add_argument('--flow-model', type=str, default='FlowNet2')
parser.add_argument('--inpainting-model', type=str, default='DeepFillv1')
opt = parser.parse_args()

input_images_dirs = list(sorted(glob.glob(f'{opt.input_images_dir}/*')))
input_masks_dirs = list(sorted(glob.glob(f'{opt.input_masks_dir}/*')))
sequence_names = list(map(basename, input_images_dirs))

input_images_dataset = SequenceDataset(
    list(glob.glob(f'{opt.input_images_dir}/*')),
    'image'
)
input_masks_dataset = SequenceDataset(
    list(glob.glob(f'{opt.input_masks_dir}/*')),
    'mask'
)
dataset = MergeDataset([input_images_dataset, input_masks_dataset], transform=transforms.ToTensor())

with torch.no_grad():
    if opt.flow_guided:
        inpainting_algorithm = FlowGuidedVideoInpaintingAlgorithm(
            flow_model=opt.flow_model,
            inpainting_model=opt.inpainting_model
        )
    else:
        inpainting_algorithm = SingleFrameVideoInpaintingAlgorithm(
            inpainting_model=opt.inpainting_model
        )

    for sequence_name, (input_images, input_masks) in tqdm(zip(sequence_names, dataset), desc='Inpainting',
                                                           unit='sequence', total=len(sequence_names)):
        output_images = []

        inpainting_algorithm.initialize()
        for input_image, input_mask in list(zip(input_images, input_masks)):
            input_image = input_image.unsqueeze(0).cuda()
            input_mask = input_mask.unsqueeze(0).cuda()
            output_image = inpainting_algorithm.inpaint_online(input_image, input_mask).squeeze().cpu()
            output_images.append(output_image)

        save_frames(output_images, f'{opt.results_dir}/OutputImages/{sequence_name}', 'image')
