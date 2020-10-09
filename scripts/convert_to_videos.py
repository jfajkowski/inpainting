import argparse
from os.path import basename

from tqdm import tqdm

from inpainting import transforms
from inpainting.load import SequenceDataset
from inpainting.save import save_video
from inpainting.utils import get_paths

parser = argparse.ArgumentParser()
parser.add_argument('--frames-dir', type=str, default='results/end2end/default/Images')
parser.add_argument('--videos-dir', type=str, default='results/end2end/default/ImageVideos')
parser.add_argument('--frame-type', type=str, default='image')
parser.add_argument('--frame-rate', type=int, default=24)
opt = parser.parse_args()

frames_dirs = get_paths(f'{opt.frames_dir}/*')
sequence_names = list(map(basename, frames_dirs))

frames_dataset = SequenceDataset(
    frames_dirs,
    opt.frame_type,
    transform=transforms.ToTensor()
)
frame_type = 'flowviz' if opt.frame_type == 'flow' else opt.frame_type

for sequence_name, frames in tqdm(zip(sequence_names, frames_dataset), desc='Converting',
                                  unit='sequence', total=len(sequence_names)):
    save_video(frames, f'{opt.videos_dir}/{sequence_name}/video.mp4', frame_type, opt.frame_rate)
