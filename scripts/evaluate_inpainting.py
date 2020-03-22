import argparse
import datetime
import glob
import os

import pandas as pd
import torch
from inpainting.external.algorithms import DeepFlowGuidedVideoInpaintingAlgorithm
from inpainting.load import StaticMaskVideoDataset, VideoDataset, RectangleMaskDataset, FileMaskDataset, \
    DynamicMaskVideoDataset
from inpainting.metrics import PSNR, MAE, MSE, SSIM
from inpainting.utils import mask_tensor, tensor_to_cv_image
from inpainting.visualize import save_video
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', type=str, default='./results/deepflowguidedvideoinpainting')
parser.add_argument('--size', type=int, nargs=2, default=(256, 256))
parser.add_argument('--frame-rate', type=int, default=24)
opt = parser.parse_args()
print(opt)

video_dataset = VideoDataset(list(glob.glob('./data/raw/video/DAVIS/JPEGImages/480p/*')), 'image')
rectangle_mask_dataset = RectangleMaskDataset(opt.size[1], opt.size[0])
irregular_mask_dataset = FileMaskDataset('./data/raw/mask/qd_imd/test')
object_mask_dataset = VideoDataset(list(glob.glob('./data/processed/video/DAVIS/Annotations_dilated/480p/*')),
                                   'mask')
transform = transforms.Compose([
    transforms.Resize(opt.size),
    transforms.ToTensor()
])
datasets = [
    ('rectangle', StaticMaskVideoDataset(video_dataset, rectangle_mask_dataset, transform=transform)),
    ('irregular', StaticMaskVideoDataset(video_dataset, irregular_mask_dataset, transform=transform)),
    ('object', DynamicMaskVideoDataset(video_dataset, object_mask_dataset, transform=transform))
]

metrics = {
    'psnr': PSNR(),
    'mae': MAE(),
    'mse': MSE(),
    'ssim': SSIM()
}

with torch.no_grad():
    algorithm = DeepFlowGuidedVideoInpaintingAlgorithm()

    result_dfs = []
    for mask_type, dataset in datasets:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        for frames, masks, path in tqdm(dataloader, desc=f'{mask_type}'):
            frames_masked, frames_filled = [], []
            results = []
            algorithm.reset()
            for frame, mask in zip(frames, masks):
                start = datetime.datetime.now()
                frame = frame.cuda()
                mask = mask.cuda()
                frame_filled = algorithm.inpaint_online(frame, mask)
                frames_filled.append(frame_filled)
                end = datetime.datetime.now()

                frames_masked.append(mask_tensor(frame, mask))

                results.append({
                    'duration': (end - start).total_seconds(),
                    'psnr': metrics['psnr'](frame, frame_filled),
                    'mae': metrics['mae'](frame, frame_filled),
                    'mse': metrics['mse'](frame, frame_filled),
                    'ssim': metrics['ssim'](frame, frame_filled),
                })

            sample_df = pd.DataFrame(results)
            print(sample_df.mean())
            sample_df['mask_type'] = mask_type
            video_name = os.path.basename(path[0])
            sample_df['video'] = video_name
            sample_df['frame'] = list(range(len(frames_filled)))
            result_dfs.append(sample_df)

            video_result_path = os.path.join(opt.results_dir, mask_type, video_name)
            os.makedirs(video_result_path, exist_ok=True)
            save_video([tensor_to_cv_image(f[0].cpu()) for f in frames], f'{video_result_path}/sequence.mp4',
                       opt.size, opt.frame_rate)
            save_video([tensor_to_cv_image(f[0].cpu()) for f in frames_masked], f'{video_result_path}/sequence_masked.mp4',
                       opt.size, opt.frame_rate)
            save_video([tensor_to_cv_image(f[0].cpu()) for f in frames_filled], f'{video_result_path}/sequence_filled.mp4',
                       opt.size, opt.frame_rate)
            pd.DataFrame(sample_df).to_csv(f'{video_result_path}/test.csv', index=False,
                                           columns=['mask_type', 'video', 'frame', 'duration', 'mae',
                                                    'mse', 'psnr', 'ssim'])

    pd.concat(result_dfs).to_csv(os.path.join(opt.results_dir, 'summary.csv'), index=False,
                                 columns=['mask_type', 'video', 'frame', 'duration', 'mae', 'mse',
                                          'psnr', 'ssim'])
