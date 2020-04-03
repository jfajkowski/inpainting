import argparse
import glob
import os

from PIL import Image
from tqdm import tqdm

from inpainting.utils import annotation_to_mask

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=str, default='data/raw/demo/Annotations')
parser.add_argument('--output-dir', type=str, default='data/interim/demo/Masks')
parser.add_argument('--index', type=int, default=1)
opt = parser.parse_args()

if __name__ == '__main__':
    for input_path in tqdm(glob.glob(f'{opt.input_dir}/**/*.png', recursive=True)):
        image = Image.open(input_path)
        mask = annotation_to_mask(image, opt.index)
        output_path = input_path.replace(opt.input_dir, opt.output_dir)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mask.save(output_path)
