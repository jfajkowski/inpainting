import glob
from time import time

import flowiz as fz
import torch
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from deepstab.flow import estimate_flow, warp_tensor
from deepstab.liteflownet import Network
from deepstab.load import VideoDataset, StaticMaskVideoDataset, RectangleMaskDataset
from deepstab.region_fill import regionfill
from deepstab.utils import mask_tensor

length = 32
batch_size = 1
epochs = 100
learning_rate = 1e-3
size = (256, 256)

frame_dataset = VideoDataset(
    list(glob.glob('../data/raw/video/DAVIS/JPEGImages/480p/*')),
    sequence_length=length + 1,
    transform=transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ]))
mask_dataset = RectangleMaskDataset(
    size[1], size[0],
    (128, 128, 32, 32),
    # '../data/raw/mask/demo',
    # '../data/raw/mask/qd_imd/test',
    # mask_mode='L',
    # shuffle=True,
    transform=transforms.Compose([
        transforms.Resize(size),
        # transforms.Lambda(lambda x: ImageOps.invert(x)),
        transforms.ToTensor()
    ]))

dataset = StaticMaskVideoDataset(frame_dataset, mask_dataset)

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = Network('../models/liteflownet/network-default.pytorch').cuda().eval()


def fill_flow(flow, mask):
    # return mask_tensor(flow, previous_mask)
    mask = mask.squeeze(0).squeeze(0).cpu().detach().numpy()
    masked_flow = flow.squeeze(0).cpu().detach().numpy()
    masked_flow[0, :, :] = regionfill(masked_flow[0, :, :], 1 - mask)
    masked_flow[1, :, :] = regionfill(masked_flow[1, :, :], 1 - mask)
    return torch.as_tensor(masked_flow).unsqueeze(0).cuda()


for e in range(epochs):
    data_iter = iter(data_loader)

    # for sample in data_iter:
    for sample in [next(data_iter)]:
        source_frames, masks, _ = next(data_iter)
        input_frames = source_frames
        # input_frames = list(map(mask_tensor, zip(source_frames, masks)))
        output_frames = []

        for i in range(length):
            start = time()
            previous_frame = input_frames[i].cuda()
            previous_mask = masks[i].cuda()
            masked_previous_frame = mask_tensor(previous_frame, previous_mask)
            current_frame = input_frames[i + 1].cuda()
            current_mask = masks[i + 1].cuda()
            masked_current_frame = mask_tensor(current_frame, current_mask)

            flow = estimate_flow(model, masked_current_frame, masked_previous_frame)
            filled_flow = fill_flow(flow, current_mask)
            warped_frame = warp_tensor(masked_current_frame, flow, padding_mode='zeros')
            output_frame = masked_current_frame + warped_frame * (1 - current_mask)

            transforms.ToPILImage()(previous_frame.cpu()[0]).save('01.png')
            transforms.ToPILImage()(previous_mask.cpu()[0]).save('02.png')
            transforms.ToPILImage()(masked_previous_frame.cpu()[0]).save('03.png')
            transforms.ToPILImage()(current_frame.cpu()[0]).save('04.png')
            transforms.ToPILImage()(current_mask.cpu()[0]).save('05.png')
            transforms.ToPILImage()(masked_current_frame.cpu()[0]).save('06.png')
            Image.fromarray(fz.convert_from_flow(flow.cpu().detach().numpy()[0].transpose(1, 2, 0))).save('07.png')
            Image.fromarray(fz.convert_from_flow(filled_flow.cpu().detach().numpy()[0].transpose(1, 2, 0))).save(
                '08.png')
            transforms.ToPILImage()(warped_frame.cpu()[0]).save('09.png')
            transforms.ToPILImage()(output_frame.cpu()[0]).save('10.png')

            print(time() - start)
