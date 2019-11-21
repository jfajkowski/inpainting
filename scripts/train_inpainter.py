import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from deepstab.load import InpaintingImageDataset, IrregularMaskDataset, ImageDataset
from deepstab.metrics import PSNR, MAE
from deepstab.model_gatingconvolutionautoencoder import Inpainter, Discriminator
from deepstab.utils import Progbar

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--image-size', type=int, nargs=2, default=[256, 256])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--use-gan', type=bool, default=True)
opt = parser.parse_args()
print(opt)

image_transforms = transforms.Compose([
    transforms.Resize((opt.image_size[0], opt.image_size[1]))
])

image_dataset = ImageDataset(['../data/raw/image/CelebA/data256x256'], transforms=image_transforms)
mask_dataset = IrregularMaskDataset('../data/raw/mask/qd_imd/train', transforms=image_transforms)
train_dataset = InpaintingImageDataset(image_dataset, mask_dataset, transforms=transforms.ToTensor())
train_data_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

image_dataset = ImageDataset(['../data/raw/image/CelebA/data256x256'], transforms=image_transforms)
mask_dataset = IrregularMaskDataset('../data/raw/mask/qd_imd/train', transforms=image_transforms)
val_dataset = InpaintingImageDataset(image_dataset, mask_dataset, transforms=transforms.ToTensor())
val_data_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True)

criterion = torch.nn.BCELoss().cuda()
metrics = {
    'psnr': PSNR(1),
    'mae': MAE()
}

model = Inpainter().cuda()
model_optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.001)

if opt.use_gan:
    # Adversarial ground truths
    valid = torch.ones((opt.batch_size, 1)).cuda()
    fake = torch.zeros((opt.batch_size, 1)).cuda()

    discriminator = Discriminator().cuda()
    discriminator_optimizer = torch.optim.SGD([p for p in discriminator.parameters() if p.requires_grad], lr=0.001)

for epoch in range(1, opt.epochs + 1):
    train_iter = iter(train_data_loader)
    val_iter = iter(val_data_loader)

    progbar = Progbar(opt.steps, stateful_metrics=['epoch', 'step'])
    logs = [("epoch", epoch)]
    for step in range(1, opt.steps + 1):
        image, mask, image_masked = next(train_iter)
        image, mask, image_masked = image.cuda(), mask.cuda(), image_masked.cuda()

        model_optimizer.zero_grad()
        image_output = model(image_masked, mask)

        if opt.use_gan:
            loss = criterion(discriminator(image_output), valid)
            loss.backward()
            model_optimizer.step()
            logs.append(('g_loss', loss.item()))

            discriminator_optimizer.zero_grad()
            real_loss = criterion(discriminator(image), valid)
            fake_loss = criterion(discriminator(image_output.detach()), fake)
            loss = (real_loss + fake_loss) / 2
            loss.backward()
            discriminator_optimizer.step()
            logs.append(('d_loss', loss.item()))
        else:
            loss = criterion(image_output, image)
            loss.backward()
            model_optimizer.step()
            logs.append(('loss', loss.item()))

        for metric_name, metric_function in metrics.items():
            logs.append((metric_name, metric_function(image, image_output).item()))
        if step != opt.steps:
            progbar.update(step, values=logs)

    image, mask, image_masked = next(val_iter)
    image, mask, image_masked = image.cuda(), mask.cuda(), image_masked.cuda()
    image_output = model(image_masked, mask)
    if opt.use_gan:

        logs.append(('val_g_loss', loss.item()))
        real_loss = criterion(discriminator(image), valid)
        fake_loss = criterion(discriminator(image_output.detach()), fake)
        loss = (real_loss + fake_loss) / 2
        logs.append(('val_d_loss', loss.item()))
    else:
        logs.append(('val_loss', criterion(image_output, image).item()))
    for metric_name, metric_function in metrics.items():
        logs.append(('val_' + metric_name, metric_function(image, image_output).item()))
    progbar.update(step, values=logs)
