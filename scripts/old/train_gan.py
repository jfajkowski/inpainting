import argparse
import glob

import torch
from apex.amp import amp, scale_loss
from opencv_transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from deepstab.load import InpaintingImageDataset, FileMaskDataset, ImageDataset
from deepstab.metrics import PSNR, MAE, MSE
from deepstab.model_gatingconvolution import GatingConvolutionUNet
from deepstab.utils import Progbar, logs_to_writer, cycle, mask_tensor, normalize, denormalize

amp.register_float_function(torch, 'sigmoid')

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', type=str, default='../models/03022020_base_dcgan')
parser.add_argument('--batch-size', type=int, default=24)
parser.add_argument('--image-size', type=int, nargs=2, default=[256, 256])
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--betas', type=float, nargs=2, default=[0.5, 0.999])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--train-steps', type=int, default=100)
parser.add_argument('--val-steps', type=int, default=5)
parser.add_argument('--checkpoint-interval', type=int, default=25)
parser.add_argument('--opt-level', type=str, default='O1')
opt = parser.parse_args()
print(opt)

image_transforms = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(256),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.flip(0)),
    transforms.Lambda(normalize)
])

mask_transforms = transforms.Compose([
    transforms.RandomRotation(180),
    transforms.RandomResizedCrop(256),
    transforms.ToTensor()
])

image_dataset = ImageDataset(['../data/raw/image/Places2/data_large'], transform=image_transforms)
mask_dataset = FileMaskDataset('../../data/raw/mask/qd_imd/train', transform=mask_transforms)
train_dataset = InpaintingImageDataset(image_dataset, mask_dataset)
train_data_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)

image_dataset = ImageDataset(['../data/raw/image/Places2/val_large'], transform=image_transforms)
mask_dataset = FileMaskDataset('../../data/raw/mask/qd_imd/train', transform=mask_transforms)
val_dataset = InpaintingImageDataset(image_dataset, mask_dataset)
val_data_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)

generator = GatingConvolutionUNet().cuda()
generator_optimizer = torch.optim.Adam([p for p in generator.parameters() if p.requires_grad],
                                       lr=opt.lr, betas=opt.betas)
generator, generator_optimizer = amp.initialize(generator, generator_optimizer, opt_level=opt.opt_level)

discriminator = DiscriminatorDCGAN().cuda()
discriminator_optimizer = torch.optim.Adam([p for p in discriminator.parameters() if p.requires_grad],
                                           lr=opt.lr, betas=opt.betas)
discriminator, discriminator_optimizer = amp.initialize(discriminator, discriminator_optimizer, opt_level=opt.opt_level)

writer = SummaryWriter(opt.model_dir)

checkpoint_paths = glob.glob(opt.model_dir + '/*.pth')
if checkpoint_paths:
    checkpoint = torch.load(sorted(checkpoint_paths)[-1])
    start_epoch = checkpoint['epoch'] + 1
    generator.load_state_dict(checkpoint['generator'])
    generator_optimizer.load_state_dict(checkpoint['generator_optimizer'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
    train_example = checkpoint['train_example']
    val_example = checkpoint['val_example']
else:
    start_epoch = 1
    train_example = next(iter(train_data_loader))
    val_example = next(iter(val_data_loader))

# Adversarial ground truth
real = torch.ones((opt.batch_size, 16384)).cuda()
fake = torch.zeros((opt.batch_size, 16384)).cuda()

reconstruction_criterion = torch.nn.L1Loss().cuda()
adversarial_criterion = torch.nn.BCELoss().cuda()
metrics = {
    'psnr': PSNR(1).cuda(),
    'l1': MAE().cuda(),
    'l2': MSE().cuda()
}

train_iter, val_iter = cycle(train_data_loader), cycle(val_data_loader)
for epoch in range(start_epoch, opt.epochs + 1):
    progbar = Progbar(opt.train_steps, stateful_metrics=['epoch', 'step'])
    logs = [("epoch", epoch)]
    additional_logs = []

    # Training
    for step in range(1, opt.train_steps + 1):
        image, mask = next(train_iter)
        image, mask = image.cuda(), mask.cuda()
        image_masked = mask_tensor(image, mask)

        generator_optimizer.zero_grad()
        image_filled = generator(image_masked, mask)
        reconstruction_loss = reconstruction_criterion(image_filled, image)
        adversarial_loss = adversarial_criterion(discriminator(image_filled), real)
        loss = (reconstruction_loss + adversarial_loss) / 2
        with scale_loss(loss, generator_optimizer) as scaled_loss:
            scaled_loss.backward()
        generator_optimizer.step()
        additional_logs.append(('g_reconstruction_loss', reconstruction_loss.item()))
        additional_logs.append(('g_adversarial_loss', adversarial_loss.item()))
        logs.append(('g_loss', loss.item()))

        discriminator_optimizer.zero_grad()
        real_loss = adversarial_criterion(discriminator(image), real)
        fake_loss = adversarial_criterion(discriminator(image_filled.detach()), fake)
        loss = (real_loss + fake_loss) / 2
        with scale_loss(loss, discriminator_optimizer) as scaled_loss:
            scaled_loss.backward()
        discriminator_optimizer.step()
        logs.append(('d_loss', loss.item()))

        for metric_name, metric_function in metrics.items():
            logs.append((metric_name, metric_function(image, image_filled).item()))
        if step != opt.train_steps:
            progbar.update(step, values=logs)

    # Validation
    with torch.no_grad():
        for step in range(1, opt.val_steps + 1):
            image, mask = next(val_iter)
            image, mask = image.cuda(), mask.cuda()
            image_masked = mask_tensor(image, mask)

            image_filled = generator(image_masked, mask)
            reconstruction_loss = reconstruction_criterion(image_filled, image)
            adversarial_loss = adversarial_criterion(discriminator(image_filled), real)
            loss = (reconstruction_loss + adversarial_loss) / 2
            additional_logs.append(('val_g_reconstruction_loss', reconstruction_loss.item()))
            additional_logs.append(('val_g_adversarial_loss', adversarial_loss.item()))
            logs.append(('val_g_loss', loss.item()))

            real_loss = adversarial_criterion(discriminator(image), real)
            fake_loss = adversarial_criterion(discriminator(image_filled.detach()), fake)
            loss = (real_loss + fake_loss) / 2
            logs.append(('val_d_loss', loss.item()))

            for metric_name, metric_function in metrics.items():
                logs.append(('val_' + metric_name, metric_function(image, image_filled).item()))
        progbar.update(opt.train_steps, values=logs)

        # Tensorboard update and checkpoint
        def update_example(example, name):
            image, mask = example
            image, mask = image.cuda(), mask.cuda()
            image_masked = mask_tensor(image, mask)
            image_filled = generator(image_masked, mask)

            writer.add_images(f'{name}/image', denormalize(image))
            writer.add_images(f'{name}/mask', mask)
            writer.add_images(f'{name}/image_masked', denormalize(image_masked))
            writer.add_images(f'{name}/image_filled', denormalize(image_filled), epoch)


        update_example(train_example, 'train')
        update_example(val_example, 'val')

        logs_to_writer(writer, logs, epoch)
        logs_to_writer(writer, additional_logs, epoch)

        for name, weight in generator.named_parameters():
            writer.add_histogram(f'g_{name}', weight, epoch)
            writer.add_histogram(f'g_{name}.grad', weight.grad, epoch)

        for name, weight in discriminator.named_parameters():
            writer.add_histogram(f'd_{name}', weight, epoch)
            writer.add_histogram(f'd_{name}.grad', weight.grad, epoch)

        if epoch % opt.checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'generator': generator.state_dict(),
                'generator_optimizer': generator_optimizer.state_dict(),
                'discriminator': discriminator.state_dict(),
                'discriminator_optimizer': discriminator_optimizer.state_dict(),
                'train_example': train_example,
                'val_example': val_example
            }, f'{opt.model_dir}/epoch_{epoch:03d}_lr_{opt.lr:.5f}.pth')
