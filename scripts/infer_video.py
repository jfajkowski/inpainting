import argparse
import glob

import torch
from PIL import Image
from deepstab.infer import infer_video

from deepstab.model_gatingconvolution import GatingConvolutionUNet
from deepstab.utils import extract_mask

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str,
                    default='../models/20191126_gatingconvunet_lg_gan/model_epoch_400_lr_0.0001.pth')
parser.add_argument('--source-frames-dir', type=str, default='../data/raw/video/demo/JPEGImages')
parser.add_argument('--masks-dir', type=str, default='../data/raw/video/demo/Annotations')
parser.add_argument('--target-frames-dir', type=str, default='../result')
parser.add_argument('--image-size', type=int, nargs=2, default=[256, 256])
opt = parser.parse_args()
print(opt)

source_frames = [Image.open(path) for path in sorted(glob.glob(opt.source_frames_dir + '/*'))]
masks = [extract_mask(Image.open(path)) for path in sorted(glob.glob(opt.masks_dir + '/*'))]

state = torch.load(opt.model_path)
model = GatingConvolutionUNet().cuda().eval()
model.load_state_dict(state['generator'])

target_frames = infer_video(model, source_frames, masks, opt.image_size, patch=True)

for i, target_frame in enumerate(target_frames):
    target_frame.save(f'{opt.target_frames_dir}/{i:05d}.jpg')
