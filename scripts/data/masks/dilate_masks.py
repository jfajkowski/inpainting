import argparse
import glob
import os

import cv2 as cv
import numpy as np

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=str, default='data/processed/DAVIS/Annotations_unsupervised/480p')
parser.add_argument('--output-dir', type=str, default='data/processed/DAVIS/Annotations_dilated/480p')
parser.add_argument('--kernel-size', type=int, default=5)
parser.add_argument('--iterations', type=int, default=3)
opt = parser.parse_args()


def dilate(mask, kernel_size, iterations):
    structuring_element = np.ones((kernel_size, kernel_size), np.uint8)
    return cv.dilate(mask, structuring_element, iterations=iterations)


if __name__ == '__main__':
    for input_path in tqdm(glob.glob(f'{opt.input_dir}/**/*.png', recursive=True)):
        mask = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
        dilated_mask = dilate(mask, opt.kernel_size, opt.iterations)
        output_path = input_path.replace(opt.input_dir, opt.output_dir)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv.imwrite(output_path, dilated_mask)
