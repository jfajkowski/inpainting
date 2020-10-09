import argparse
from os.path import basename

import pandas as pd
import torch
from tqdm import tqdm

from inpainting import transforms
from inpainting.flow import select_flow_model
from inpainting.load import SequenceDataset
from inpainting.save import save_frames, save_dataframe
from inpainting.utils import get_paths

parser = argparse.ArgumentParser()
parser.add_argument('--images-dir', type=str, default='data/processed/flow_estimation/Images')
parser.add_argument('--results-dir', type=str, default='results/flow_estimation/default')
parser.add_argument('--flow-model', type=str, default='FlowNet2')
opt = parser.parse_args()

images_dirs = get_paths(f'{opt.images_dir}/*')
sequence_names = list(map(basename, images_dirs))

dataset = SequenceDataset(
    images_dirs,
    'image',
    transform=transforms.ToTensor()
)

with torch.no_grad():
    flow_model = select_flow_model(opt.flow_model)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    dry_run_image = torch.rand_like(dataset[0][0]).unsqueeze(0).cuda()
    flow_model(dry_run_image, dry_run_image)

    for sequence_name, images in tqdm(zip(sequence_names, dataset), desc='Estimating flow',
                                      unit='sequence', total=len(sequence_names)):
        results, times = [], []

        prev_image = images[0].unsqueeze(0).cuda()
        for i, curr_image in enumerate(images[1:]):
            curr_image = curr_image.unsqueeze(0).cuda()
            start.record()
            result = flow_model(prev_image, curr_image).squeeze().cpu()
            end.record()
            torch.cuda.synchronize()
            times.append({'frame_id': i, 'inference_time': start.elapsed_time(end)})
            results.append(result)
            prev_image = curr_image

        save_frames(results, f'{opt.results_dir}/Flows/{sequence_name}', 'flow')
        save_dataframe(pd.DataFrame(times), f'{opt.results_dir}/Benchmark/{sequence_name}/inference_times.csv')
