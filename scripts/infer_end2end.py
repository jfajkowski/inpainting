import argparse
from os.path import basename

import pandas as pd
import torch
from tqdm import tqdm

from inpainting import transforms
from inpainting.algorithms import VideoTrackingAlgorithm, FlowGuidedVideoInpaintingAlgorithm, \
    SingleFrameVideoInpaintingAlgorithm
from inpainting.load import SequenceDataset, MergeDataset
from inpainting.save import save_frame, save_frames, save_dataframe
from inpainting.utils import mask_to_bbox, get_paths

parser = argparse.ArgumentParser()
parser.add_argument('--images-dir', type=str, default='data/interim/demo/Images')
parser.add_argument('--masks-dir', type=str, default='data/interim/demo/Masks')
parser.add_argument('--results-dir', type=str, default='results/demo/ee/default')
parser.add_argument('--dilation-size', type=int, default=5)
parser.add_argument('--dilation-iterations', type=int, default=3)
parser.add_argument('--flow-model', type=str, default='FlowNet2')
parser.add_argument('--inpainting-model', type=str, default='DeepFillv1')

opt = parser.parse_args()

images_dirs = get_paths(f'{opt.images_dir}/*')
masks_dirs = get_paths(f'{opt.masks_dir}/*')
sequence_names = list(map(basename, images_dirs))

images_dataset = SequenceDataset(
    images_dirs,
    'image'
)
masks_dataset = SequenceDataset(
    masks_dirs,
    'annotation'
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

    if opt.flow_model == 'None':
        inpainting_algorithm = SingleFrameVideoInpaintingAlgorithm(
            inpainting_model=opt.inpainting_model
        )
    else:
        inpainting_algorithm = FlowGuidedVideoInpaintingAlgorithm(
            flow_model=opt.flow_model,
            inpainting_model=opt.inpainting_model
        )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    dry_run_frame = torch.rand_like(dataset[0][0][0]).unsqueeze(0).cuda()
    dry_run_mask = torch.rand_like(dataset[0][1][0]).unsqueeze(0).cuda()
    inpainting_algorithm.initialize()
    inpainting_algorithm.inpaint_online(dry_run_frame, dry_run_mask)

    for sequence_name, (images, masks) in tqdm(zip(sequence_names, dataset), desc='End-to-end',
                                               unit='sequence', total=len(sequence_names)):
        result_masks, result_images, times = [], [], []

        initial_image, initial_roi = images[0], mask_to_bbox(masks[0])
        tracking_algorithm.initialize(initial_image.cuda(), initial_roi)
        inpainting_algorithm.initialize()
        for i, image in enumerate(images):
            start.record()
            image = image.cuda()
            result_mask = tracking_algorithm.track_online(image).cuda()
            result_image = inpainting_algorithm.inpaint_online(image.unsqueeze(0), result_mask.unsqueeze(0))
            end.record()
            torch.cuda.synchronize()
            times.append({'frame_id': i, 'inference_time': start.elapsed_time(end)})
            result_masks.append(result_mask)
            result_images.append(result_image)

        save_frame(initial_image, f'{opt.results_dir}/Initialization/{sequence_name}/initialization.png', 'image',
                   initial_roi)
        save_frames(result_masks, f'{opt.results_dir}/Masks/{sequence_name}', 'mask')
        save_frames(result_images, f'{opt.results_dir}/Images/{sequence_name}', 'image')
        save_dataframe(pd.DataFrame(times), f'{opt.results_dir}/Benchmark/{sequence_name}/inference_times.csv')
