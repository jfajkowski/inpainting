import argparse
import glob
from os.path import basename

import torch
from tqdm import tqdm

from inpainting import transforms
from inpainting.flow import select_flow_model
from inpainting.load import SequenceDataset
from inpainting.save import save_frames

parser = argparse.ArgumentParser()
parser.add_argument('--input-images-dir', type=str, default='data/processed/demo/InputImages')
parser.add_argument('--results-dir', type=str, default='results/flow_estimation/demo')
parser.add_argument('--flow-model', type=str, default='FlowNet2')
opt = parser.parse_args()

input_images_dirs = list(sorted(glob.glob(f'{opt.input_images_dir}/*')))
sequence_names = list(map(basename, input_images_dirs))

dataset = SequenceDataset(
    list(glob.glob(f'{opt.input_images_dir}/*')),
    'image',
    transform=transforms.ToTensor()
)

with torch.no_grad():
    flow_model = select_flow_model(opt.flow_model)

    for sequence_name, input_images in tqdm(zip(sequence_names, dataset), desc='Estimating flow',
                                            unit='sequence', total=len(sequence_names)):
        output_flows = []

        prev_input_image = input_images[0].unsqueeze(0).cuda()
        for curr_input_image in input_images[1:]:
            curr_input_image = curr_input_image.unsqueeze(0).cuda()
            output_flow = flow_model(prev_input_image, curr_input_image).squeeze().cpu()
            output_flows.append(output_flow)
            prev_input_image = curr_input_image

        save_frames(output_flows, f'{opt.results_dir}/Flows/{sequence_name}', 'flow')
