import argparse
import glob
import os

import cv2 as cv
import numpy as np

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=str, default='data/processed/video/DAVIS/Annotations_dilated/480p')
parser.add_argument('--output-dir', type=str, default='data/processed/video/DAVIS/Annotations_inverted/480p')
opt = parser.parse_args()
print(opt)


def invert(mask):
    return 255 - mask


if __name__ == '__main__':
    for input_path in tqdm(glob.glob(f'{opt.input_dir}/**/*.png', recursive=True)):
        mask = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
        dilated_mask = invert(mask)
        output_path = input_path.replace(opt.input_dir, opt.output_dir)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv.imwrite(output_path, dilated_mask)
