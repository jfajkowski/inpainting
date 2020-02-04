import argparse
import datetime
import glob
import os
from itertools import product

import pandas as pd
import torch
from cv2 import resize
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepstab.evaluate import evaluate_video
from deepstab.inpainting import ImageInpaintingAlgorithm, FlowInpaintingAlgorithm
from deepstab.liteflownet import Network
from deepstab.load import StaticMaskVideoDataset, VideoDataset, RectangleMaskDataset, FileMaskDataset, \
    DynamicMaskVideoDataset
from deepstab.model_gatingconvolution import GatingConvolutionUNet
from deepstab.utils import cv_image_to_tensor, tensor_to_cv_image

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str,
                    default='../models/20200122_gatingconvunet_gan/model_epoch_275_lr_0.0001.pth')
parser.add_argument('--results-dir', type=str, default='../results/20200122_gatingconvunet_gan')
parser.add_argument('--image-size', type=int, nargs=2, default=[854, 480])
opt = parser.parse_args()
print(opt)

video_dataset = VideoDataset(list(glob.glob('../data/raw/video/DAVIS/JPEGImages/480p/*')))
rectangle_mask_dataset = RectangleMaskDataset(opt.image_size[1], opt.image_size[0])
irregular_mask_dataset = FileMaskDataset('../data/raw/mask/qd_imd/test')
object_mask_dataset = VideoDataset(list(glob.glob('../data/raw/video/DAVIS/Annotations_unsupervised/480p/*')))
datasets = {
    ('rectangle', StaticMaskVideoDataset(video_dataset, rectangle_mask_dataset)),
    ('irregular', StaticMaskVideoDataset(video_dataset, irregular_mask_dataset)),
    ('object', DynamicMaskVideoDataset(video_dataset, object_mask_dataset))
}

flow_model = Network('../models/liteflownet/network-default.pytorch').eval().cuda()
state = torch.load('../models/20200122_gatingconvunet_gan/model_epoch_275_lr_0.0001.pth')
inpainting_model = GatingConvolutionUNet().cuda().eval()
inpainting_model.load_state_dict(state['generator'])
algorithms = {
    ('image_based', ImageInpaintingAlgorithm(inpainting_model)),
    ('flow_based', FlowInpaintingAlgorithm(flow_model, inpainting_model))
}

with torch.no_grad():
    result_dfs = []
    for (method_name, algorithm), (mask_type, dataset) in product(algorithms, datasets):
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        for sample in tqdm(data_loader, desc=f'{method_name} {mask_type}'):
            frames, masks, path = sample
            frames_filled = []
            durations = []
            for frame, mask in zip(frames, masks):
                start = datetime.datetime.now()

                frame = cv_image_to_tensor(resize(frame, opt.image_size)).unsqueeze(0).cuda() / 255
                mask = cv_image_to_tensor(resize(mask, opt.image_size)).unsqueeze(0).cuda()

                frame_filled = algorithm.inpaint_online(frame, mask) * 255

                tensor_to_cv_image(frame_filled.squeeze(0).cpu())

                frames_filled.append(frame_filled)
                end = datetime.datetime.now()
                durations.append((end - start).total_seconds())

            metrics = evaluate_video(frames, frames_filled)
            sample_df = pd.DataFrame(metrics)
            sample_df['duration'] = durations
            print(sample_df.mean())
            sample_df['mask_type'] = mask_type
            video_name = os.path.basename(path)
            sample_df['video'] = video_name
            sample_df['frame'] = list(range(len(frames_filled)))
            result_dfs.append(sample_df)

            video_result_path = os.path.join(opt.results_dir, method_name, mask_type, video_name)
            os.makedirs(video_result_path)
            for i, frame_filled in enumerate(frames_filled):
                frame_filled.save(os.path.join(video_result_path, f'{i:05d}.jpg'))

    pd.concat(result_dfs).to_csv(os.path.join(opt.results_dir, 'summary.csv'), index=False,
                                 columns=['mask_type', 'video', 'frame', 'duration', 'mae', 'mse', 'psnr', 'ssim'])
