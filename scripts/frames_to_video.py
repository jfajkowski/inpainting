import argparse
import glob
from os.path import basename

import inpainting.transforms as T
from inpainting.load import VideoDataset
from inpainting.visualize import save_video

parser = argparse.ArgumentParser()
parser.add_argument('--frames-pattern', type=str, default='results/demo/Inpainter/OutputImages/*')
parser.add_argument('--video-dir', type=str, default='results/demo/Inpainter/Misc')
parser.add_argument('--video-name', type=str, default='output_images.mp4')
parser.add_argument('--frame-type', type=str, default='image')
parser.add_argument('--frame-rate', type=int, default=24)
opt = parser.parse_args()

frames_dirs = list(glob.glob(opt.frames_pattern))
frames_dataset = VideoDataset(
    frames_dirs,
    opt.frame_type,
    transform=T.Compose([
        T.ToTensor()
    ])
)

for frame_dir, frames in zip(frames_dirs, frames_dataset):
    save_video(frames, f'{opt.video_dir}/{basename(frame_dir)}/{opt.video_name}', opt.frame_type, opt.frame_rate)
