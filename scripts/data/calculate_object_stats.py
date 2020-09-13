import argparse
import glob
from os.path import basename

import numpy as np
import pandas as pd
from tqdm import tqdm

from inpainting.load import SequenceDataset
from inpainting.visualize import save_dataframe

parser = argparse.ArgumentParser()
parser.add_argument('--annotations-dir', type=str, default='data/interim/demo/ResizedAnnotations')
parser.add_argument('--object-stats-dir', type=str, default='data/interim/demo/ObjectStats')
opt = parser.parse_args()


annotation_dirs = list(sorted(glob.glob(f'{opt.annotations_dir}/*')))
sequence_names = list(map(basename, annotation_dirs))

annotation_dataset = SequenceDataset(annotation_dirs, 'annotation')


def calculate_object_stats(annotations):
    results = []
    for object_id, group in _calculate_objects_presence(annotations).groupby('object_id'):
        results.append({
            'object_id': object_id,
            'first_frame_id': int(group["frame_id"].min()),
            'last_frame_id': int(group["frame_id"].max()),
            'presence': float(group['frame_id'].count() / len(annotations)),
            'mean_size': float(group['object_to_screen_ratio'].mean())
        })
    return pd.DataFrame(results)


def _calculate_objects_presence(annotations):
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
    save_dataframe(calculate_object_stats(annotations), f'{opt.object_stats_dir}/{sequence_name}/object_stats.csv')
