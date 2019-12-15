import argparse

import torch
from apex.amp import amp, scale_loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from deepstab.load import InpaintingImageDataset, IrregularMaskDataset, ImageDataset
from deepstab.metrics import PSNR, MAE
from deepstab.model_discriminator import Discriminator
from deepstab.model_gatingconvolution import GatingConvolutionUNet
from deepstab.partialconv.loss import VGG16PartialLoss
from deepstab.utils import Progbar, logs_to_writer, cycle

amp.register_float_function(torch, 'sigmoid')

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default='../models/model')
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--image-size', type=int, nargs=2, default=[256, 256])
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--betas', type=float, nargs=2, default=[0.5, 0.999])
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--train-steps', type=int, default=100)
parser.add_argument('--val-steps', type=int, default=5)
parser.add_argument('--checkpoint-interval', type=int, default=25)
parser.add_argument('--opt-level', type=str, default='O1')
opt = parser.parse_args()
print(opt)

image_transforms = transforms.Compose([
    transforms.Resize((opt.image_size[0], opt.image_size[1])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10)
])

mask_transforms = transforms.Compose([
    transforms.RandomRotation(180),
    transforms.RandomResizedCrop((opt.image_size[0], opt.image_size[1]))
])

image_dataset = ImageDataset(['../data/raw/image/CelebA/data256x256'], transform=image_transforms)
mask_dataset = IrregularMaskDataset('../data/raw/mask/qd_imd/train', transform=mask_transforms)
train_dataset = InpaintingImageDataset(image_dataset, mask_dataset, mask_mode='L', transform=transforms.ToTensor())
train_data_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

image_dataset = ImageDataset(['../data/raw/image/CelebA/data256x256'], transform=image_transforms)
mask_dataset = IrregularMaskDataset('../data/raw/mask/qd_imd/train', transform=mask_transforms)
val_dataset = InpaintingImageDataset(image_dataset, mask_dataset, mask_mode='L', transform=transforms.ToTensor())
val_data_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True)

# Adversarial ground truth
real = torch.ones((opt.batch_size, 16384)).cuda()
fake = torch.zeros((opt.batch_size, 16384)).cuda()

generator = GatingConvolutionUNet().cuda()
generator_optimizer = torch.optim.Adam([p for p in generator.parameters() if p.requires_grad],
                                       lr=opt.lr, betas=opt.betas)
generator, generator_optimizer = amp.initialize(generator, generator_optimizer, opt_level=opt.opt_level)

discriminator = Discriminator().cuda()
discriminator_optimizer = torch.optim.Adam([p for p in discriminator.parameters() if p.requires_grad],
                                           lr=opt.lr, betas=opt.betas)
discriminator, discriminator_optimizer = amp.initialize(discriminator, discriminator_optimizer, opt_level=opt.opt_level)

perceptual_criterion = amp.initialize(VGG16PartialLoss().cuda(), opt_level=opt.opt_level)
adversarial_criterion = torch.nn.BCELoss().cuda()
metrics = {
    'psnr': PSNR(1),
    'mae': MAE()
}

writer = SummaryWriter()
train_iter = cycle(train_data_loader)
train_example = next(iter(train_data_loader))
val_iter = cycle(val_data_loader)
val_example = next(iter(val_data_loader))
for epoch in range(1, opt.epochs + 1):
    progbar = Progbar(opt.train_steps, stateful_metrics=['epoch', 'step'])
    logs = [("epoch", epoch)]
    additional_logs = []

    # Training
    for step in range(1, opt.train_steps + 1):
        image, mask, image_masked = next(train_iter)
        image, mask, image_masked = image.cuda(), mask.cuda(), image_masked.cuda()

        generator_optimizer.zero_grad()
        image_output = generator(image_masked, mask)
        perceptual_loss = perceptual_criterion(image_output, image)[0]
        adversarial_loss = adversarial_criterion(discriminator(image_output), real)
        loss = (perceptual_loss + adversarial_loss) / 2
        with scale_loss(loss, generator_optimizer) as scaled_loss:
            scaled_loss.backward()
        generator_optimizer.step()
        additional_logs.append(('g_perceptual_loss', perceptual_loss.item()))
        additional_logs.append(('g_adversarial_loss', adversarial_loss.item()))
        logs.append(('g_loss', loss.item()))

        discriminator_optimizer.zero_grad()
        real_loss = adversarial_criterion(discriminator(image), real)
        fake_loss = adversarial_criterion(discriminator(image_output.detach()), fake)
        loss = (real_loss + fake_loss) / 2
        with scale_loss(loss, discriminator_optimizer) as scaled_loss:
            scaled_loss.backward()
        discriminator_optimizer.step()
        logs.append(('d_loss', loss.item()))

        for metric_name, metric_function in metrics.items():
            logs.append((metric_name, metric_function(image, image_output).item()))
        if step != opt.train_steps:
            progbar.update(step, values=logs)

    # Validation
    with torch.no_grad():
        for step in range(1, opt.val_steps + 1):
            image, mask, image_masked = next(val_iter)
            image, mask, image_masked = image.cuda(), mask.cuda(), image_masked.cuda()

            image_output = generator(image_masked, mask)
            perceptual_loss = perceptual_criterion(image_output, image)[0]
            adversarial_loss = adversarial_criterion(discriminator(image_output), real)
            loss = (perceptual_loss + adversarial_loss) / 2
            additional_logs.append(('val_g_perceptual_loss', perceptual_loss.item()))
            additional_logs.append(('val_g_adversarial_loss', adversarial_loss.item()))
            logs.append(('val_g_loss', loss.item()))

            real_loss = adversarial_criterion(discriminator(image), real)
            fake_loss = adversarial_criterion(discriminator(image_output.detach()), fake)
            loss = (real_loss + fake_loss) / 2
            logs.append(('val_d_loss', loss.item()))

            for metric_name, metric_function in metrics.items():
                logs.append(('val_' + metric_name, metric_function(image, image_output).item()))
        progbar.update(opt.train_steps, values=logs)

        # Tensorboard update and checkpoint
        image, mask, image_masked = train_example
        image, mask, image_masked = image.cuda(), mask.cuda(), image_masked.cuda()
        image_output = generator(image_masked, mask)
        writer.add_images('train/image', image)
        writer.add_images('train/image_masked', image_masked)
        writer.add_images('train/image_output', image_output, epoch)

        image, mask, image_masked = val_example
        image, mask, image_masked = image.cuda(), mask.cuda(), image_masked.cuda()
        image_output = generator(image_masked, mask)
        writer.add_images('val/image', image)
        writer.add_images('val/image_masked', image_masked)
        writer.add_images('val/image_output', image_output, epoch)

        logs_to_writer(writer, logs, epoch)
        logs_to_writer(writer, additional_logs, epoch)

    if epoch % opt.checkpoint_interval == 0:
        torch.save({
            'epoch': epoch,
            'generator': generator.state_dict(),
            'generator_optimizer': generator_optimizer.state_dict(),
            'discriminator': discriminator.state_dict(),
            'discriminator_optimizer': discriminator_optimizer.state_dict(),
        }, f'{opt.model_path}_epoch_{epoch}_lr_{opt.lr}.pth')
