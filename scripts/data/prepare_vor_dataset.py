import argparse
import glob

from PIL import Image
from torchvision.transforms import transforms as T
from tqdm import tqdm

from inpainting.load import VideoDataset, VideoObjectRemovalDataset
from inpainting.visualize import save_frames

parser = argparse.ArgumentParser()
parser.add_argument('--images-dir', type=str, default='data/raw/demo/JPEGImages')
parser.add_argument('--masks-dir', type=str, default='data/interim/demo/Masks')
parser.add_argument('--output-dir', type=str, default='data/processed/demo')
parser.add_argument('--size', type=int, nargs=2, default=(256, 256))
opt = parser.parse_args()

image_dataset = VideoDataset(
    glob.glob(f'{opt.images_dir}/*'),
    frame_type='image',
    transform=T.Resize(opt.size, interpolation=Image.BILINEAR)
)
annotation_dataset = VideoDataset(
    glob.glob(f'{opt.masks_dir}/*'),
    frame_type='mask',
    transform=T.Resize(opt.size, interpolation=Image.NEAREST)
)
dataset = VideoObjectRemovalDataset(image_dataset, annotation_dataset, transform=T.ToTensor())

for i, (input_images, masks, target_images) in enumerate(tqdm(dataset)):
    save_frames(input_images, f'{opt.output_dir}/InputImages/{i:05d}', 'image')
    save_frames(masks, f'{opt.output_dir}/Masks/{i:05d}', 'mask')
    save_frames(target_images, f'{opt.output_dir}/TargetImages/{i:05d}', 'image')
