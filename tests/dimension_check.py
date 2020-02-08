from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from inpainting.load import ImageDataset, FileMaskDataset, InpaintingImageDataset

IMAGE_SIZE = 256
BATCH_SIZE = 8

image_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10)
])

mask_transforms = transforms.Compose([
    transforms.RandomRotation(180),
    transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE))
])

image_dataset = ImageDataset(['../data/raw/image/Places2/data_256/*/*'], transform=image_transforms)
mask_dataset = FileMaskDataset('../data/raw/mask/qd_imd/train', transform=mask_transforms)
train_dataset = InpaintingImageDataset(image_dataset, mask_dataset, mask_mode='L', transform=transforms.ToTensor())
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

image_dataset = ImageDataset(['../data/raw/image/Places2/val_256'], transform=image_transforms)
mask_dataset = FileMaskDataset('../data/raw/mask/qd_imd/train', transform=mask_transforms)
val_dataset = InpaintingImageDataset(image_dataset, mask_dataset, mask_mode='L', transform=transforms.ToTensor())
val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_data_iter = iter(train_data_loader)
for i, _ in enumerate(train_data_iter):
    print(i)

val_data_iter = iter(val_data_loader)
for i, _ in enumerate(val_data_iter):
    print(i)
