import argparse
import glob

import pandas as pd
import torch
from torchvision.transforms import transforms as T
from tqdm import tqdm

from inpainting.evaluate import evaluate_tracking, save_stats, save_results
from inpainting.load import VideoDataset, DynamicMaskVideoDataset

parser = argparse.ArgumentParser()
parser.add_argument('--output-masks-dir', type=str, default='results/demo/Masks')
parser.add_argument('--target-masks-dir', type=str, default='data/processed/demo/Masks')
parser.add_argument('--results-dir', type=str, default='results/demo')
opt = parser.parse_args()


output_masks_dataset = VideoDataset(
    list(glob.glob(f'{opt.output_masks_dir}/*')),
    'mask'
)
target_masks_dataset = VideoDataset(
    list(glob.glob(f'{opt.target_masks_dir}/*')),
    'mask'
)
dataset = DynamicMaskVideoDataset(output_masks_dataset, target_masks_dataset, transform=T.ToTensor())

with torch.no_grad():
    sample_dfs = []
    for i, (output_masks, target_masks) in enumerate(tqdm(dataset)):
        sample_df = evaluate_tracking(target_masks, output_masks)
        save_stats(sample_df.drop(columns=['t']), f'{opt.results_dir}/Misc/{i}')
        sample_df['video'] = i
        sample_dfs.append(sample_df)

    df = pd.concat(sample_dfs)
    save_results(df, f'{opt.results_dir}')
    save_stats(df.drop(columns=['video', 't']), f'{opt.results_dir}')
