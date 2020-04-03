import argparse
import glob
import json
from os import makedirs
from os.path import basename, dirname, splitext

import numpy as np
import pandas as pd
from tqdm import tqdm

from inpainting.load import VideoDataset

parser = argparse.ArgumentParser()
parser.add_argument('--annotations-dir', type=str, default='data/raw/demo/Annotations')
parser.add_argument('--output-path', type=str, default='data/processed/demo/object_stats.json')
opt = parser.parse_args()

# Load annotations
annotation_dirs = list(glob.glob(f'{opt.annotations_dir}/*'))
annotation_dataset = VideoDataset(annotation_dirs, frame_type='annotation')

# Calculate object stats for each sequence
dataset_stats = {'videos': {}}
for sequence_dir, sequence in tqdm(zip(annotation_dirs, annotation_dataset),
                                   desc='Calculating object stats', unit='sequence', total=len(annotation_dirs)):
    sequence_name = basename(sequence_dir)
    sequence_stats = []
    for annotation_path, annotation in zip(glob.glob(f'{sequence_dir}/*.png'), sequence):
        annotation_array = np.array(annotation)
        annotation_pixel_count = annotation_array.size
        for object_id, object_pixel_count in zip(*np.unique(annotation_array, return_counts=True)):
            if object_id == 0:  # skip background (has always 0 id)
                continue

            sequence_stats.append({
                't': splitext(basename(annotation_path))[0],
                'object_id': object_id,
                'object_to_screen_ratio': object_pixel_count / annotation_pixel_count,
            })
    sequence_df = pd.DataFrame(sequence_stats)

    objects = {}
    for object_id, group in sequence_df.groupby('object_id'):
        objects[str(object_id)] = {
            'first_frame': f'{int(group["t"].astype(float).min()):05d}',
            'last_frame': f'{int(group["t"].astype(float).max()):05d}',
            'time_on_screen': float(group['t'].count() / len(sequence)),
            'mean_coverage': float(group['object_to_screen_ratio'].mean())
        }
    dataset_stats['videos'][sequence_name] = {'objects': objects}

# Persist calculated stats as JSON file
makedirs(dirname(opt.output_path), exist_ok=True)
with open(opt.output_path, mode='w') as f_out:
    json.dump(dataset_stats, f_out, indent=4)
