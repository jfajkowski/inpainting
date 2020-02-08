import glob

import torchvision.transforms as transforms
import torch
from torch.utils.data.dataloader import DataLoader

from inpainting.inpainting import ImageInpaintingAlgorithm, FlowInpaintingAlgorithm
from inpainting.pwcnet import Network
from inpainting.load import VideoDataset, DynamicMaskVideoDataset
from scripts.train_baseline import Baseline

batch_size = 1
size = (256, 256)

transform = transforms.Compose([
    # transforms.Resize(size),
    transforms.ToTensor()
])
frame_dataset = VideoDataset(
    list(glob.glob('data/raw/video/DAVIS/JPEGImages/480p/rollerblade')),
    frame_type='image',
    transform=transform)
mask_dataset = VideoDataset(
    list(glob.glob('data/processed/video/DAVIS/Annotations_dilated/480p/rollerblade')),
    frame_type='mask',
    transform=transform)
dataset = DynamicMaskVideoDataset(frame_dataset, mask_dataset)
# mask_dataset = RectangleMaskDataset(
#     size[1], size[0],
#     (128 - 16, 128 - 16, 32, 32),
#     # '../data/raw/mask/demo',
#     # '../data/raw/mask/qd_imd/test',
#     transform=transform)
# dataset = StaticMaskVideoDataset(frame_dataset, mask_dataset)

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

flow_model = Network('models/pwcnet/network-default.pytorch').eval().cuda()
inpainting_algorithm = FlowInpaintingAlgorithm(flow_model)

with torch.no_grad():
    data_iter = iter(data_loader)

    for sample in data_iter:
        frames, masks, _ = sample
        frames = list(map(lambda x: x.cuda(), frames))
        masks = list(map(lambda x: x.cuda(), masks))
        frames_filled = inpainting_algorithm.inpaint(frames, masks)
