import argparse
import glob
from os.path import basename

import torch
import pandas as pd
from tqdm import tqdm

from inpainting import transforms
from inpainting.algorithms import VideoTrackingAlgorithm
from inpainting.load import SequenceDataset, MergeDataset
from inpainting.save import save_frame, save_frames, save_dataframe
from inpainting.utils import mask_to_bbox

parser = argparse.ArgumentParser()
parser.add_argument('--images-dir', type=str, default='data/processed/tracking_and_segmentation/Images')
parser.add_argument('--masks-dir', type=str, default='data/processed/tracking_and_segmentation/Masks')
parser.add_argument('--results-dir', type=str, default='results/tracking_and_segmentation/default')
parser.add_argument('--dilation-size', type=int, default=5)
parser.add_argument('--dilation-iterations', type=int, default=3)
opt = parser.parse_args()

images_dirs = list(sorted(glob.glob(f'{opt.images_dir}/*')))
masks_dirs = list(sorted(glob.glob(f'{opt.masks_dir}/*')))
sequence_names = list(map(basename, images_dirs))

images_dataset = SequenceDataset(
    images_dirs,
    'image'
)
masks_dataset = SequenceDataset(
    masks_dirs,
    'mask'
)
dataset = MergeDataset([images_dataset, masks_dataset], transform=transforms.ToTensor())

with torch.no_grad():
    tracking_algorithm = VideoTrackingAlgorithm(opt.dilation_size, opt.dilation_iterations)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    dry_run_frame = torch.rand_like(dataset[0][0][0])
    dry_run_mask = torch.rand_like(dataset[0][1][0])
    dry_run_initial_image, dry_run_initial_roi = dry_run_frame, mask_to_bbox(dry_run_mask)
    tracking_algorithm.initialize(dry_run_initial_image.cuda(), dry_run_initial_roi)
    tracking_algorithm.track_online(dry_run_initial_image.cuda())

    for sequence_name, (images, masks) in tqdm(zip(sequence_names, dataset), desc='Tracking',
                                               unit='sequence', total=len(sequence_names)):
        results, times = [], []

        initial_image, initial_roi = images[0], mask_to_bbox(masks[0])
        tracking_algorithm.initialize(initial_image.cuda(), initial_roi)
        for i, image in enumerate(images):
            start.record()
            result = tracking_algorithm.track_online(image.cuda())
            end.record()
            torch.cuda.synchronize()
            times.append({'frame_id': i, 'inference_time': start.elapsed_time(end)})
            results.append(result)

        save_frame(initial_image, f'{opt.results_dir}/Initialization/{sequence_name}/initialization.png', 'image',
                   initial_roi)
        save_frames(results, f'{opt.results_dir}/Masks/{sequence_name}', 'mask')
        save_dataframe(pd.DataFrame(times), f'{opt.results_dir}/Benchmark/{sequence_name}/inference_times.csv')
