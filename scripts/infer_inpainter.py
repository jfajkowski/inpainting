import argparse
import time

import torch
from PIL import Image
from apex.amp import amp, scale_loss
from torchvision.transforms import transforms

from deepstab.model_partialconvolutionunet import PartialConvolutionUNet
from deepstab.partialconv.loss import VGG16PartialLoss
from deepstab.utils import extract_mask, cutout_mask

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='PartialConvolutionUNet', choices=model_map.keys())
parser.add_argument('--criterion', type=str, default='VGG16PartialLoss', choices=criterion_map.keys())
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--image-size', type=int, default=512)
opt = parser.parse_args()
print(opt)

img = Image.open('../data/raw/DAVIS/JPEGImages/480p/breakdance/00000.jpg')
mask = extract_mask(Image.open('../data/raw/DAVIS/Annotations_unsupervised/480p/breakdance/00000.png'), 1)
img_masked = cutout_mask(img, mask)

model = PartialConvolutionUNet().cuda()
criterion = VGG16PartialLoss().cuda()
optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.001)

transform = transforms.Compose([
    transforms.Resize((opt.image_size, opt.image_size)),
    transforms.ToTensor()
])


def prepare(x, n=4):
    return torch.stack((x,) * n).view(n, 3, opt.image_size, opt.image_size).cuda()


img = prepare(transform(img))
mask = prepare(torch.stack((transform(mask),) * 3))
img_masked = prepare(transform(img_masked))

opt_level = 'O1'
model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

start = time.time()
for epoch in range(1000):
    print(f'Epoch: {epoch + 1}')
    optimizer.zero_grad()
    img_output, mask_output = model(img_masked, mask)
    loss = criterion(img_output, img)
    with scale_loss(loss[0], optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()

invert_transform = transforms.Compose([
    transforms.ToPILImage()
])

invert_transform(img_output[0, :, :, :].cpu().detach()).show()
print(f'Took {time.time() - start}')
