import argparse
from os.path import basename

import numpy as np
import pandas as pd
from tqdm import tqdm

from inpainting.load import SequenceDataset
from inpainting.save import save_dataframe
from inpainting.utils import get_paths

parser = argparse.ArgumentParser()
parser.add_argument('--annotations-dir', type=str, default='data/interim/demo/Annotations')
parser.add_argument('--object-stats-dir', type=str, default='data/interim/demo/ObjectStats')
parser.add_argument('--min-presence', type=float, default=0.5)
parser.add_argument('--min-mean-size', type=float, default=0.01)
parser.add_argument('--max-mean-size', type=float, default=0.25)
opt = parser.parse_args()


annotation_dirs = get_paths(f'{opt.annotations_dir}/*')
sequence_names = list(map(basename, annotation_dirs))

annotation_dataset = SequenceDataset(annotation_dirs, 'annotation')


def validate_object_stats(aggregated_object_stats, min_presence, min_mean_size, max_mean_size):
    results = []
    for _, row in aggregated_object_stats.iterrows():
        if row['presence'] > min_presence \
                and min_mean_size <= row['mean_size'] <= max_mean_size:
            results.append(row)
    return pd.DataFrame(results)


def aggregate_object_stats(raw_object_stats, annotations):
    results = []
    for object_id, group in raw_object_stats.groupby('object_id'):
        results.append({
            'object_id': object_id,
            'first_frame_id': int(group['frame_id'].min()),
            'last_frame_id': int(group['frame_id'].max()),
            'presence': float(group['frame_id'].count() / len(annotations)),
            'min_size': float(group['object_to_screen_ratio'].min()),
            'mean_size': float(group['object_to_screen_ratio'].mean()),
            'max_size': float(group['object_to_screen_ratio'].max())
        })
    return pd.DataFrame(results)


def calculate_raw_object_stats(annotations):
    results = []
    for frame_id, annotation in enumerate(annotations):
        annotation = np.array(annotation)
        annotation_pixel_count = annotation.size
        for object_id, object_pixel_count in zip(*np.unique(annotation, return_counts=True)):
            if object_id == 0:  # skip background
                continue

            results.append({
                'frame_id': frame_id,
                'object_id': object_id,
                'object_to_screen_ratio': object_pixel_count / annotation_pixel_count,
            })
    return pd.DataFrame(results)


for sequence_name, annotations in tqdm(zip(sequence_names, annotation_dataset), desc='Calculating object stats',
                                       unit='sequence', total=len(sequence_names)):
    raw_object_stats = calculate_raw_object_stats(annotations)
    save_dataframe(raw_object_stats,
                   f'{opt.object_stats_dir}/{sequence_name}/raw_object_stats.csv')

    aggregated_object_stats = aggregate_object_stats(raw_object_stats, annotations)
    save_dataframe(aggregated_object_stats,
                   f'{opt.object_stats_dir}/{sequence_name}/aggregated_object_stats.csv')

    valid_object_stats = validate_object_stats(aggregated_object_stats,
                                               opt.min_presence, opt.min_mean_size, opt.max_mean_size)
    save_dataframe(valid_object_stats,
                   f'{opt.object_stats_dir}/{sequence_name}/valid_object_stats.csv')
