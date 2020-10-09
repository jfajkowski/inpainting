import argparse
from os.path import basename

from tqdm import tqdm

from inpainting import transforms
from inpainting.load import SequenceDataset
from inpainting.save import save_frames
from inpainting.utils import get_paths

parser = argparse.ArgumentParser()
parser.add_argument('--frames-dir', type=str, default='data/raw/YouTube-VOS/valid/JPEGImages')
parser.add_argument('--interim-dir', type=str, default='data/interim/YouTube-VOS/valid/JPEGImages')
parser.add_argument('--crop', type=float, default=2.0)
parser.add_argument('--scale', type=int, nargs=2, default=(512, 256))
parser.add_argument('--frame-type', type=str, default='image')
opt = parser.parse_args()


frame_dirs = get_paths(f'{opt.frames_dir}/*')
sequence_names = list(map(basename, frame_dirs))

frames_dataset = SequenceDataset(frame_dirs, frame_type=opt.frame_type,
                                 transform=transforms.Compose([
                                     transforms.CenterCrop(opt.crop, opt.frame_type),
                                     transforms.Resize(opt.scale[::-1], opt.frame_type)
                                 ]))

for sample_name, frames in tqdm(zip(sequence_names, frames_dataset), desc='Adjusting frames', unit='sequence',
                                total=len(frame_dirs)):
    save_frames(frames, f'{opt.interim_dir}/{sample_name}', opt.frame_type)
