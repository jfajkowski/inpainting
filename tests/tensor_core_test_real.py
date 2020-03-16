import argparse
import glob

import torch
import torchvision.transforms as transforms
from PIL import Image
from apex.amp import amp
from torch.utils.data.dataloader import DataLoader

from inpainting.external.liteflownet import Network
from inpainting.inpainting import FlowAndFillInpaintingAlgorithm
from inpainting.load import VideoDataset, DynamicMaskVideoDataset
from scripts.train import InpaintingModel

parser = argparse.ArgumentParser()
parser.add_argument('opt_level', type=str, default='O1')
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--size', type=int, default=(256, 256))
opt = parser.parse_args()
print(opt)

frame_dataset = VideoDataset(
    list(glob.glob('data/raw/video/DAVIS/JPEGImages/480p/flamingo')),
    frame_type='image',
    transform=transforms.Compose([
        transforms.Resize(opt.size, interpolation=Image.BILINEAR),
        transforms.ToTensor()
    ]))
mask_dataset = VideoDataset(
    list(glob.glob('data/processed/video/DAVIS/Annotations_dilated/480p/flamingo')),
    frame_type='mask',
    transform=transforms.Compose([
        transforms.Resize(opt.size, interpolation=Image.NEAREST),
        transforms.ToTensor()
    ]))
dataset = DynamicMaskVideoDataset(frame_dataset, mask_dataset)

data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

with torch.no_grad():
    flow_model = Network('models/external/flownet2/liteflownet/network-default.pytorch').cuda().eval()
    fill_model = InpaintingModel.load_from_checkpoint(
        'models/baseline/version_0/checkpoints/_ckpt_epoch_96.ckpt').generator.cuda().eval()
    flow_model, fill_model = amp.initialize([flow_model, fill_model], opt_level=opt.opt_level)
    inpainting_algorithm = FlowAndFillInpaintingAlgorithm(flow_model, fill_model)

    sample = next(iter(data_loader))

    frames, masks, _ = sample
    frames = list(map(lambda x: x.cuda(), frames[:2]))
    masks = list(map(lambda x: x.cuda(), masks[:2]))
    inpainting_algorithm.reset()
    frames_filled, masks_filled = inpainting_algorithm.inpaint(frames, masks)
