import argparse
import glob
from os.path import basename

from PIL import Image
from tqdm import tqdm

from inpainting import transforms
from inpainting.load import SequenceDataset
from inpainting.visualize import save_frames

parser = argparse.ArgumentParser()
parser.add_argument('--frames-dir', type=str, default='data/raw/demo/JPEGImages')
parser.add_argument('--interim-dir', type=str, default='data/interim/demo/ResizedJPEGImages')
parser.add_argument('--size', type=int, nargs=2, default=(256, 512))
parser.add_argument('--frame-type', type=str, default='image')
opt = parser.parse_args()

frame_dirs = list(sorted(glob.glob(f'{opt.frames_dir}/*')))
sequence_names = list(map(basename, frame_dirs))

frames_dataset = SequenceDataset(frame_dirs, frame_type=opt.frame_type,
                                 transform=transforms.Resize(opt.size[::-1]))

for sample_name, frames in tqdm(zip(sequence_names, frames_dataset), desc='Resizing frames', unit='sequence',
                                total=len(frame_dirs)):
    save_frames(frames, f'{opt.interim_dir}/{sample_name}', opt.frame_type)
