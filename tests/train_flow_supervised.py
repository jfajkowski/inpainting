import glob

import cv2 as cv
import pytorch_lightning as pl
import torchvision
from PIL import Image
from spatial_correlation_sampler import spatial_correlation_sample
from torch.nn import functional as F
from torch.nn.init import kaiming_normal_, constant_
from torch.utils.data import DataLoader

from inpainting.external.deepflowguidedvideoinpainting.flownet2.submodules import *
from inpainting.load import MergeDataset, VideoDataset
from inpainting.utils import mean_and_std, denormalize, normalize_flow, warp_tensor
from inpainting.visualize import flow_to_image_tensor
import inpainting.transforms as T


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )


def correlate(input1, input2):
    out_corr = spatial_correlation_sample(input1,
                                          input2,
                                          kernel_size=1,
                                          patch_size=21,
                                          stride=1,
                                          padding=0,
                                          dilation_patch=2)
    # collate dimensions 1 and 2 in order to be treated as a
    # regular 4D tensor
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1)
    return F.leaky_relu_(out_corr, 0.1)


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]


class FlowModel(pl.LightningModule):
    expansion = 1

    def __init__(self, batchNorm=True, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, 3, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv_redir = conv(self.batchNorm, 256, 32, kernel_size=1, stride=1)

        self.conv3_1 = conv(self.batchNorm, 473, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x1, x2):
        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        out_conv1b = self.conv1(x2)
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        out_conv_redir = self.conv_redir(out_conv3a)
        out_correlation = correlate(out_conv3a, out_conv3b)

        in_conv3_1 = torch.cat([out_conv_redir, out_correlation], dim=1)

        out_conv3 = self.conv3_1(in_conv3_1)
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2a)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2a)

        concat2 = torch.cat((out_conv2a, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2

    def training_step(self, batch, batch_nb):
        self.first, self.second = batch[0]
        self.flow_true = self.prepare_ground_truth(batch[1][0])
        self.flow_predict = self(self.first, self.second)

        loss = 0.0
        for fp, ft in zip(self.flow_predict, self.flow_true):
            loss += F.mse_loss(fp, ft)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def prepare_ground_truth(self, flow):
        return F.interpolate(flow, size=(64, 64), mode='bilinear'), \
               F.interpolate(flow, size=(32, 32), mode='bilinear'), \
               F.interpolate(flow, size=(16, 16), mode='bilinear'), \
               F.interpolate(flow, size=(8, 8), mode='bilinear'), \
               F.interpolate(flow, size=(4, 4), mode='bilinear')

    def restore(self, flow):
        return F.interpolate(flow[0], size=(256, 256), mode='bilinear')

    def training_epoch_end(self, outputs):
        first = denormalize(self.first.detach(), 'imagenet').cpu()
        second = denormalize(self.second.detach(), 'imagenet').cpu()
        flow_true = self.restore(self.flow_true).detach().cpu()
        flow_predict = self.restore(self.flow_predict).detach().cpu()
        warped_true = warp_tensor(second, flow_true).cpu()
        warped_predict = warp_tensor(second, flow_predict).cpu()
        x = torch.cat([
            first[0].unsqueeze(0), second[0].unsqueeze(0),
            flow_to_image_tensor(flow_true[0]).unsqueeze(0), flow_to_image_tensor(flow_predict[0]).unsqueeze(0),
            warped_true[0].unsqueeze(0), warped_predict[0].unsqueeze(0),
        ], dim=0)
        grid = torchvision.utils.make_grid(x)
        self.logger.experiment.add_image('generated_images', grid, trainer.current_epoch)
        return {}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


if __name__ == '__main__':
    pairs_dataset = VideoDataset(
        list(sorted(glob.glob(f'../data/raw/MPI-Sintel-complete/training/clean/alley_1'))),
        'image',
        sequence_length=2,
        transform=T.Compose([
            T.Resize(size=[256, 256], interpolation=Image.BILINEAR),
            T.ToTensor(),
            T.Normalize(*mean_and_std('imagenet'))
        ])
    )
    flow_dataset = VideoDataset(
        list(sorted(glob.glob(f'../data/raw/MPI-Sintel-complete/training/flow/alley_1'))),
        'flow',
        sequence_length=1,
        transform=T.Compose([
            T.Lambda(normalize_flow),
            T.Lambda(lambda x: cv.resize(x, dsize=(256, 256), interpolation=cv.INTER_LINEAR)),
            T.ToTensor()
        ])
    )

    dataset = MergeDataset([pairs_dataset, flow_dataset])
    loader = DataLoader(dataset, shuffle=True, batch_size=8, num_workers=0)

    model = FlowModel()
    trainer = pl.Trainer(gpus=1, precision=32)
    trainer.fit(model, loader)
