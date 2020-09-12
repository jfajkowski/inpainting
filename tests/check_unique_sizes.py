import glob

from tqdm import tqdm

from inpainting.load import ImageDataset

image_dataset = ImageDataset(
    list(sorted(glob.glob('../data/raw/YouTube-VOS/train/JPEGImages/*'))),
    'image'
)
annotation_dataset = ImageDataset(
    list(sorted(glob.glob('../data/raw/YouTube-VOS/train/Annotations/*'))),
    'annotation'
)

unique_sizes = set()
for image in tqdm(image_dataset):
    unique_sizes.add(image.size)
print(f'Unique sizes for image dataset: {unique_sizes}')

unique_sizes = set()
for annotation in tqdm(annotation_dataset):
    unique_sizes.add(annotation.size)
print(f'Unique sizes for annotation dataset: {unique_sizes}')