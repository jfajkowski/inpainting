import argparse

import torch
import yaml
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

from deepstab.deepfill.model.networks import Generator
from deepstab.deepfill.utils.tools import normalize

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default='../models/deepfill/gen_00430000.pt')
parser.add_argument('--source-image-path', type=str, default='../data/raw/image/demo/480p.jpg')
parser.add_argument('--mask-path', type=str, default='../data/raw/mask/demo/mask.png')
parser.add_argument('--target-image-path', type=str, default='../result.jpg')
parser.add_argument('--image-size', type=int, nargs=2, default=[256, 256])
opt = parser.parse_args()
print(opt)

state = torch.load(opt.model_path)
config = yaml.load(open('../models/deepfill/deepfill.yml'), Loader=yaml.FullLoader)
model = Generator(config['netG'], True, 0).cuda().eval()
model.load_state_dict(state)

x = Image.open(opt.source_image_path).resize((512, 512))
m = Image.open(opt.mask_path).convert('L').resize(x.size)

x = to_tensor(x).unsqueeze(dim=0).cuda()
m = to_tensor(m).unsqueeze(dim=0).cuda()

x = normalize(x)
x = x * (1. - m)

x1, x2, flow = model(x, m)
inpainted_result = x2 * m + x * (1. - m)

save_image(inpainted_result, opt.target_image_path, padding=0, normalize=True)
