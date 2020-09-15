import argparse
import glob
from os.path import basename

from tqdm import tqdm

from inpainting import transforms
from inpainting.load import SequenceDataset
from inpainting.visualize import save_frames

parser = argparse.ArgumentParser()
parser.add_argument('--frames-dir', type=str, default='data/raw/demo/JPEGImages')
parser.add_argument('--interim-dir', type=str, default='data/interim/demo/AdjustedJPEGImages')
parser.add_argument('--crop', type=int, nargs=2, default=(854, 427))
parser.add_argument('--scale', type=int, nargs=2, default=(512, 256))
parser.add_argument('--frame-type', type=str, default='image')
opt = parser.parse_args()

frame_dirs = list(sorted(glob.glob(f'{opt.frames_dir}/*')))
sequence_names = list(map(basename, frame_dirs))

frames_dataset = SequenceDataset(frame_dirs, frame_type=opt.frame_type,
                                 transform=transforms.Compose([
                                     transforms.CenterCrop(opt.crop[::-1]),
                                     transforms.Resize(opt.scale[::-1])
                                 ]))

for sample_name, frames in tqdm(zip(sequence_names, frames_dataset), desc='Adjusting frames', unit='sequence',
                                total=len(frame_dirs)):
    save_frames(frames, f'{opt.interim_dir}/{sample_name}', opt.frame_type)
