import argparse
import datetime
import glob
import os
from itertools import product

import cv2 as cv
import opencv_transforms.transforms as transforms
import pandas as pd
import torch
from tqdm import tqdm

from inpainting.evaluate import evaluate_video
from inpainting.algorithm import FillInpaintingAlgorithm, FlowInpaintingAlgorithm
from inpainting.external.flow_models import Network
from inpainting.load import StaticMaskVideoDataset, VideoDataset, RectangleMaskDataset, FileMaskDataset, \
    DynamicMaskVideoDataset
from inpainting.model_generator import GatingConvolutionUNet
from inpainting.utils import cv_image_to_tensor, tensor_to_cv_image, mask_tensor

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str,
                    default='../models/20200122_gatingconvunet_gan/model_epoch_275_lr_0.0001.pth')
parser.add_argument('--results-dir', type=str, default='../results/20200122_gatingconvunet_gan')
parser.add_argument('--image-size', type=int, nargs=2, default=(256, 256))
opt = parser.parse_args()
print(opt)

video_dataset = VideoDataset(list(glob.glob('../data/raw/video/DAVIS/JPEGImages/480p/*')), 'image')
rectangle_mask_dataset = RectangleMaskDataset(opt.image_size[1], opt.image_size[0])
irregular_mask_dataset = FileMaskDataset('../data/raw/mask/qd_imd/test')
object_mask_dataset = VideoDataset(list(glob.glob('../data/processed/video/DAVIS/Annotations_unsupervised/480p/*')),
                                   'mask')
resize = transforms.Resize(opt.image_size)
datasets = [
    ('rectangle', StaticMaskVideoDataset(video_dataset, rectangle_mask_dataset, transform=resize)),
    ('irregular', StaticMaskVideoDataset(video_dataset, irregular_mask_dataset, transform=resize)),
    ('object', DynamicMaskVideoDataset(video_dataset, object_mask_dataset, transform=resize))
]

flow_model = Network('../models/liteflownet/network-default.pytorch').eval().cuda()
state = torch.load('../models/20200122_gatingconvunet_gan/model_epoch_275_lr_0.0001.pth')
inpainting_model = GatingConvolutionUNet().cuda().eval()
inpainting_model.load_state_dict(state['generator'])
algorithms = [
    ('image_based', FillInpaintingAlgorithm(inpainting_model)),
    ('flow_based', FlowInpaintingAlgorithm(flow_model, inpainting_model))
]

with torch.no_grad():
    result_dfs = []
    for (method_name, algorithm), (mask_type, dataset) in product(algorithms, datasets):
        for i in tqdm(range(len(dataset)), desc=f'{method_name} {mask_type}'):
            frames, masks, path = dataset[i]
            frames_masked = []
            frames_filled = []
            durations = []
            algorithm.reset()
            for frame, mask in zip(frames, masks):
                start = datetime.datetime.now()

                frame = cv_image_to_tensor(frame).unsqueeze(0).cuda() / 255
                mask = cv_image_to_tensor(mask).unsqueeze(0).cuda() / 255

                frame_filled = algorithm.inpaint_online(frame, mask) * 255
                frame_filled = tensor_to_cv_image(frame_filled.squeeze(0).cpu())
                frames_filled.append(frame_filled)

                end = datetime.datetime.now()
                durations.append((end - start).total_seconds())

                frame_masked = mask_tensor(frame, mask) * 255
                frame_masked = tensor_to_cv_image(frame_masked.squeeze(0).cpu())
                frames_masked.append(frame_masked)

            metrics = evaluate_video(frames, frames_filled)
            sample_df = pd.DataFrame(metrics)
            sample_df['duration'] = durations
            print(sample_df.mean())
            sample_df['method_name'] = method_name
            sample_df['mask_type'] = mask_type
            video_name = os.path.basename(path)
            sample_df['video'] = video_name
            sample_df['frame'] = list(range(len(frames_filled)))
            result_dfs.append(sample_df)

            video_result_path = os.path.join(opt.results_dir, method_name, mask_type, video_name)
            os.makedirs(video_result_path)
            os.makedirs(f'{video_result_path}/masked')
            os.makedirs(f'{video_result_path}/filled')
            for j, (frame_masked, frame_filled) in enumerate(zip(frames_masked, frames_filled)):
                cv.imwrite(f'{video_result_path}/masked/{j:05d}.png', frame_masked)
                cv.imwrite(f'{video_result_path}/filled/{j:05d}.jpg', frame_filled)
            pd.DataFrame(sample_df).to_csv(f'{video_result_path}/summary.csv', index=False,
                                           columns=['mask_type', 'method_name', 'video', 'frame', 'duration', 'mae',
                                                    'mse', 'psnr', 'ssim'])

    pd.concat(result_dfs).to_csv(os.path.join(opt.results_dir, 'summary.csv'), index=False,
                                 columns=['mask_type', 'method_name', 'video', 'frame', 'duration', 'mae', 'mse',
                                          'psnr', 'ssim'])
