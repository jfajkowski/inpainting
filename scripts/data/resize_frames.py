import argparse
import glob
from os import makedirs
from os.path import basename

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from inpainting.load import VideoDataset, save_sample

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=str, default='data/raw/DAVIS/JPEGImages/480p')
parser.add_argument('--output-dir', type=str, default='data/interim/DAVIS/JPEGImages')
parser.add_argument('--size', type=int, nargs=2, default=(256, 256))
parser.add_argument('--type', type=str, default='image')
opt = parser.parse_args()

# Load and resize frames
frame_dirs = list(glob.glob(f'{opt.input_dir}/*'))
interpolation = Image.BILINEAR if opt.type == 'image' else Image.NEAREST
frame_dataset = VideoDataset(frame_dirs, sample_type=opt.type,
                             transform=transforms.Resize(opt.size[::-1], interpolation))

# Save resized frames
for sequence_dir, sequence in tqdm(zip(frame_dirs, frame_dataset), desc='Resizing frames', unit='sequence',
                                   total=len(frame_dirs)):
    sequence_name = basename(sequence_dir)
    image_o_dir = f'{opt.output_dir}/{sequence_name}'
    makedirs(image_o_dir, exist_ok=True)
    for frame_path, frame in zip(glob.glob(f'{sequence_dir}/*'), sequence):
        frame_name = basename(frame_path)
        save_sample(frame, f'{image_o_dir}/{frame_name}', opt.type)
