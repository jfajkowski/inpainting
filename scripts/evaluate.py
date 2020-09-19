import argparse
import glob
from os.path import basename

import torch
from tqdm import tqdm

from inpainting import transforms
from inpainting.evaluate import evaluate_inpainting, evaluate_segmentation, evaluate_flow
from inpainting.load import SequenceDataset, MergeDataset
from inpainting.save import save_dataframe

parser = argparse.ArgumentParser()
parser.add_argument('--output-frames-dir', type=str, default='results/demo/Tracker/OutputMasks')
parser.add_argument('--target-frames-dir', type=str, default='data/processed/demo/Masks')
parser.add_argument('--results-dir', type=str, default='results/demo/Tracker')
parser.add_argument('--mode', type=str, default='segmentation')
opt = parser.parse_args()


if opt.mode == 'tracking_and_segmentation':
    sequence_type = 'mask'
    evaluate = evaluate_segmentation
elif opt.mode == 'image_inpainting':
    sequence_type = 'image'
    evaluate = evaluate_inpainting
elif 'flow' in opt.mode:
    sequence_type = 'flow'
    evaluate = evaluate_flow
else:
    raise ValueError(opt.mode)

output_frames_dirs = list(sorted(glob.glob(f'{opt.output_frames_dir}/*')))
target_frames_dirs = list(sorted(glob.glob(f'{opt.target_frames_dir}/*')))
sequence_names = list(map(basename, output_frames_dirs))

output_frames_dataset = SequenceDataset(output_frames_dirs, sequence_type)
target_frames_dataset = SequenceDataset(target_frames_dirs, sequence_type)
dataset = MergeDataset([output_frames_dataset, target_frames_dataset], transform=transforms.ToTensor())

with torch.no_grad():
    for sequence_name, (output_frames, target_frames) in tqdm(zip(sequence_names, dataset), desc='Evaluating',
                                                              unit='sequence', total=len(sequence_names)):
        sample_df = evaluate(target_frames, output_frames)
        save_dataframe(sample_df, f'{opt.results_dir}/Evaluation/{sequence_name}/results.csv')
