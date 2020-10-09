import argparse
import random
from os.path import basename

from tqdm import tqdm

from inpainting.load import SequenceDataset
from inpainting.save import save_frames
from inpainting.utils import get_paths

parser = argparse.ArgumentParser()
parser.add_argument('--images-dir', type=str, default='data/interim/demo/Images')
parser.add_argument('--flows-dir', type=str, default='data/interim/demo/Flows')
parser.add_argument('--processed-dir', type=str, default='data/processed/flow_estimation')
parser.add_argument('--limit-samples', type=int, default=3)
parser.add_argument('--seed', type=int, default=42)
opt = parser.parse_args()

random.seed(opt.seed)

images_dirs = get_paths(f'{opt.images_dir}/*')
flows_dirs = get_paths(f'{opt.flows_dir}/*')
sequence_names = list(map(basename, images_dirs))

images_dataset = SequenceDataset(
    images_dirs,
    frame_type='image'
)
flows_dataset = SequenceDataset(
    flows_dirs,
    frame_type='flow'
)

candidates = list(range(len(sequence_names)))

# Limit candidates with random sampling
if len(candidates) > opt.limit_samples:
    candidates = random.sample(candidates, opt.limit_samples)

for index in tqdm(candidates, desc=f'Preparing flow_estimation dataset', unit='sequence'):
    images = images_dataset[index]
    flows = flows_dataset[index]
    sequence_name = sequence_names[index]
    save_frames(images, f'{opt.processed_dir}/Images/{sequence_name}', 'image')
    save_frames(flows, f'{opt.processed_dir}/Flows/{sequence_name}', 'flow')
