import argparse
import glob
from os.path import basename

from tqdm import tqdm

from inpainting import transforms
from inpainting.load import SequenceDataset
from inpainting.visualize import save_video

parser = argparse.ArgumentParser()
parser.add_argument('--frames-dir', type=str, default='results/inpainting/default/OutputImages')
parser.add_argument('--videos-dir', type=str, default='results/inpainting/default/OutputVideos')
parser.add_argument('--frame-type', type=str, default='image')
parser.add_argument('--frame-rate', type=int, default=24)
opt = parser.parse_args()

frames_dirs = list(sorted(glob.glob(f'{opt.frames_dir}/*')))
sequence_names = list(map(basename, frames_dirs))

frames_dataset = SequenceDataset(
    frames_dirs,
    opt.frame_type,
    transform=transforms.ToTensor()
)

for sequence_name, frames in tqdm(zip(sequence_names, frames_dataset), desc='Converting',
                                  unit='sequence', total=len(sequence_names)):
    save_video(frames, f'{opt.videos_dir}/{sequence_name}/video.mp4', opt.frame_type, opt.frame_rate)
