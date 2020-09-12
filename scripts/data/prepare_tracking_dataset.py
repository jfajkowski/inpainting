import argparse
import glob

from PIL import ImageStat, ImageOps
from tqdm import tqdm

from inpainting.load import ImageDataset, MergeDataset
from inpainting.utils import mask_to_bbox
from inpainting.visualize import save_frame

parser = argparse.ArgumentParser()
parser.add_argument('--images-dir', type=str, default='data/raw/DAVIS/JPEGImages')
parser.add_argument('--masks-dir', type=str, default='data/interim/DAVIS/Masks')
parser.add_argument('--output-dir', type=str, default='data/processed/DAVIS')
opt = parser.parse_args()


def prepare_search(sample, roi, is_mask):
    (x1, y1), (x2, y2) = roi
    roi_x, roi_y = (x1 + x2) // 2, (y1 + y2) // 2
    old_size = sample.size

    left_margin = roi_x
    top_margin = roi_y
    right_margin = old_size[0] - roi_x
    bottom_margin = old_size[1] - roi_y

    roi_w, roi_h = x2 - x1, y2 - y1
    half_new_size = max(roi_w, roi_h) * 2

    left_border = half_new_size - left_margin
    right_border = half_new_size - right_margin
    top_border = half_new_size - top_margin
    bottom_border = half_new_size - bottom_margin

    if is_mask:
        fill = 0
    else:
        fill = tuple(map(int, ImageStat.Stat(sample).mean))
    return ImageOps.expand(sample, (left_border, top_border, right_border, bottom_border), fill)


def prepare_exemplar(sample):
    assert sample.size[0] == sample.size[1]
    return ImageOps.crop(sample, (sample.size[0] // 4, ) * 4)


if __name__ == '__main__':
    image_dataset = ImageDataset(
        list(sorted(glob.glob(f'{opt.images_dir}/*'))),
        sample_type='image'
    )
    mask_dataset = ImageDataset(
        list(sorted(glob.glob(f'{opt.masks_dir}/*'))),
        sample_type='mask'
    )
    dataset = MergeDataset([image_dataset, mask_dataset])

    for i, (image, mask) in enumerate(tqdm(dataset)):

        if roi := mask_to_bbox(mask):
            search_image = prepare_search(image, roi, is_mask=False)
            exemplar_image = prepare_exemplar(search_image)
            search_mask = prepare_search(mask, roi, is_mask=True)

            save_frame(search_image, f'{opt.output_dir}/Tracking/SearchImage/{i:09d}.jpg', 'image')
            save_frame(exemplar_image, f'{opt.output_dir}/Tracking/ExemplarImage/{i:09d}.jpg', 'image')
            save_frame(search_mask, f'{opt.output_dir}/Tracking/SearchMask/{i:09d}.png', 'mask')
        else:
            print(f"Skipping frame {i:09d}, because mask contains no object.")
