import argparse
import glob
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=str, default='data/raw/video/DAVIS/Annotations_unsupervised/480p')
parser.add_argument('--output-dir', type=str, default='data/processed/video/DAVIS/Annotations_unsupervised/480p')
parser.add_argument('--index', type=int, default=1)
opt = parser.parse_args()
print(opt)


def extract_mask(image, i):
    image = np.array(image)
    mask = np.ones(image.shape, dtype=np.uint8) * 255
    mask[image == i] = 0
    return Image.fromarray(mask).convert('L')


if __name__ == '__main__':
    for input_path in tqdm(glob.glob(f'{opt.input_dir}/**/*.png', recursive=True)):
        image = Image.open(input_path)
        mask = extract_mask(image, opt.index)
        output_path = input_path.replace(opt.input_dir, opt.output_dir)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mask.save(output_path)
