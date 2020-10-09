import argparse
from os.path import basename

import pandas as pd
import torch
from tqdm import tqdm

from inpainting import transforms
from inpainting.algorithms import SingleFrameVideoInpaintingAlgorithm, FlowGuidedVideoInpaintingAlgorithm
from inpainting.load import SequenceDataset, MergeDataset
from inpainting.save import save_frames, save_dataframe
from inpainting.utils import get_paths

parser = argparse.ArgumentParser()
parser.add_argument('--frames-dir', type=str, default='data/processed/image_inpainting/Images')
parser.add_argument('--masks-dir', type=str, default='data/processed/image_inpainting/Masks')
parser.add_argument('--results-dir', type=str, default='results/image_inpainting/default')
parser.add_argument('--inpainting-model', type=str, default='DeepFillv1')
parser.add_argument('--flow-model', type=str, default='FlowNet2')
parser.add_argument('--mode', type=str, default='image_inpainting')
opt = parser.parse_args()

frames_dirs = get_paths(f'{opt.frames_dir}/*')
masks_dirs = get_paths(f'{opt.masks_dir}/*')
sequence_names = list(map(basename, frames_dirs))

frame_type = 'flow' if 'flow' in opt.mode else 'image'
frames_dataset = SequenceDataset(
    frames_dirs,
    frame_type
)
masks_dataset = SequenceDataset(
    masks_dirs,
    'mask'
)
dataset = MergeDataset([frames_dataset, masks_dataset], transform=transforms.ToTensor())


with torch.no_grad():
    if opt.flow_model == 'None':
        inpainting_model = SingleFrameVideoInpaintingAlgorithm(
            inpainting_model=opt.inpainting_model
        )
    else:
        inpainting_model = FlowGuidedVideoInpaintingAlgorithm(
            flow_model=opt.flow_model,
            inpainting_model=opt.inpainting_model
        )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    dry_run_frame = torch.rand_like(dataset[0][0][0]).unsqueeze(0).cuda()
    dry_run_mask = torch.rand_like(dataset[0][1][0]).unsqueeze(0).cuda()
    inpainting_model.initialize()
    inpainting_model.inpaint_online(dry_run_frame, dry_run_mask)

    for sequence_name, (frames, masks) in tqdm(zip(sequence_names, dataset), desc='Inpainting',
                                               unit='sequence', total=len(sequence_names)):
        results, times = [], []

        inpainting_model.initialize()
        for i, (frame, mask) in enumerate(zip(frames, masks)):
            frame = frame.unsqueeze(0).cuda()
            mask = mask.unsqueeze(0).cuda()
            start.record()
            result = inpainting_model.inpaint_online(frame, mask).squeeze().cpu()
            end.record()
            torch.cuda.synchronize()
            times.append({'frame_id': i, 'inference_time': start.elapsed_time(end)})
            results.append(result)

        save_frames(results, f'{opt.results_dir}/{frame_type.capitalize()}s/{sequence_name}', frame_type)
        save_dataframe(pd.DataFrame(times), f'{opt.results_dir}/Benchmark/{sequence_name}/inference_times.csv')
