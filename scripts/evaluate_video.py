import argparse
import datetime
import glob
import os

import pandas as pd
import torch
from PIL.ImageOps import invert
from deepstab.infer import image_to_batch_tensor, batch_tensor_to_image

from deepstab.evaluate import evaluate_video
from deepstab.load import StaticMaskVideoDataset, VideoDataset, RectangleMaskDataset, FileMaskDataset
from deepstab.model_gatingconvolution import GatingConvolutionUNet

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str,
                    default='../models/20200122_gatingconvunet_gan/model_epoch_275_lr_0.0001.pth')
parser.add_argument('--results-dir', type=str, default='../results/20200122_gatingconvunet_gan')
parser.add_argument('--image-size', type=int, nargs=2, default=[854, 480])
opt = parser.parse_args()
print(opt)


def mask_transform(transformed_mask, invert_colors):
    transformed_mask = transformed_mask.resize(opt.image_size)
    if invert_colors:
        transformed_mask = invert(transformed_mask)
    return transformed_mask


video_dataset = VideoDataset(list(glob.glob('../data/raw/video/DAVIS/JPEGImages/480p/*')))
mask_datasets = [
    ('rectangle',
     RectangleMaskDataset(opt.image_size[1], opt.image_size[0], transform=lambda x: mask_transform(x, False))),
    ('irregular',
     FileMaskDataset('../data/raw/mask/mask/testing_mask_dataset', transform=lambda x: mask_transform(x, True))),
    ('drawing', FileMaskDataset('../data/raw/mask/qd_imd/test', transform=lambda x: mask_transform(x, False)))
]

state = torch.load(opt.model_path)
model = GatingConvolutionUNet().cuda().eval()
model.load_state_dict(state['generator'])

result_dfs = []
for mask_type, mask_dataset in mask_datasets:
    test_dataset = StaticMaskVideoDataset(video_dataset, mask_dataset, transform=lambda x: x.resize((256, 256)))
    for i in range(len(test_dataset)):
        source_frames, masks, masked_frames, frame_dir = test_dataset[i]
        target_frames = []
        durations = []
        for masked_frame, mask in zip(masked_frames, masks):
            start = datetime.datetime.now()
            masked_frame_tensor = image_to_batch_tensor(masked_frame, channels=3)
            mask_tensor = image_to_batch_tensor(mask, channels=1)
            target_frame_tensor = model(masked_frame_tensor, mask_tensor)
            target_frame = batch_tensor_to_image(target_frame_tensor)
            target_frames.append(target_frame)
            end = datetime.datetime.now()
            durations.append((end - start).total_seconds())

        metrics = evaluate_video(source_frames, target_frames)
        sample_df = pd.DataFrame(metrics)
        sample_df['duration'] = durations
        print(sample_df.mean())
        sample_df['mask_type'] = mask_type
        video_name = os.path.basename(frame_dir)
        sample_df['video'] = video_name
        sample_df['frame'] = list(range(len(target_frames)))
        result_dfs.append(sample_df)

        video_result_path = os.path.join(opt.results_dir, mask_type, video_name)
        os.makedirs(video_result_path)
        for j, (masked_frame, target_frame) in enumerate(zip(masked_frames, target_frames)):
            masked_frame.save(os.path.join(video_result_path, f'masked_frame_{j:05d}.jpg'))
            target_frame.save(os.path.join(video_result_path, f'target_frame_{j:05d}.jpg'))

pd.concat(result_dfs).to_csv(os.path.join(opt.results_dir, 'summary.csv'), index=False,
                             columns=['mask_type', 'video', 'frame', 'duration', 'mse', 'psnr', 'ssim'])
