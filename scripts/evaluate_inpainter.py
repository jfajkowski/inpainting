import argparse
import glob

import pandas as pd
import torch
from torchvision.transforms import transforms as T
from tqdm import tqdm

from inpainting.evaluate import save_stats, save_results, evaluate_inpainting
from inpainting.load import VideoDataset, DynamicMaskVideoDataset, MergeDataset

parser = argparse.ArgumentParser()
parser.add_argument('--output-images-dir', type=str, default='results/demo/Inpainter/OutputImages')
parser.add_argument('--target-images-dir', type=str, default='data/processed/demo/TargetImages')
parser.add_argument('--results-dir', type=str, default='results/demo/Inpainter')
opt = parser.parse_args()


output_images_dataset = VideoDataset(
    list(glob.glob(f'{opt.output_images_dir}/*')),
    'image'
)
target_images_dataset = VideoDataset(
    list(glob.glob(f'{opt.target_images_dir}/*')),
    'image'
)
dataset = MergeDataset([output_images_dataset, target_images_dataset], transform=T.ToTensor())

with torch.no_grad():
    sample_dfs = []
    for i, (output_images, target_images) in enumerate(tqdm(dataset)):
        sample_df = evaluate_inpainting(target_images, output_images)
        save_stats(sample_df.drop(columns=['t']), f'{opt.results_dir}/Misc/{i}')
        sample_df['video'] = i
        sample_dfs.append(sample_df)

    df = pd.concat(sample_dfs)
    save_results(df, f'{opt.results_dir}')
    save_stats(df.drop(columns=['video', 't']), f'{opt.results_dir}')
