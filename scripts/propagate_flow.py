import glob
from time import time

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from deepstab.flow import estimate_flow
from deepstab.flownet2 import Network
from deepstab.load import VideoDataset, StaticMaskVideoDataset, RectangleMaskDataset
from deepstab.region_fill import regionfill

length = 8
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
    # '../data/raw/mask/demo',
    # mask_mode='L',
    # shuffle=False,
    transform=transforms.Compose([
        transforms.Resize(size),
        # transforms.Lambda(lambda x: ImageOps.invert(x)),
        transforms.ToTensor()
    ]))

dataset = StaticMaskVideoDataset(frame_dataset, mask_dataset)

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

model = Network('../models/flownet2/network-default.pytorch').cuda().eval()

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
            previous_mask = masks[i].cuda()
            previous_frame = input_frames[i].cuda()
            current_frame = input_frames[i + 1].cuda()

            flow = estimate_flow(model, previous_frame, current_frame)

            # masked_flow = mask_tensor(flow, previous_mask)
            previous_mask = previous_mask.squeeze(0).squeeze(0).cpu().detach().numpy()
            masked_flow = flow.squeeze(0).cpu().detach().numpy()
            masked_flow[0, :, :] = regionfill(masked_flow[0, :, :], 1 - previous_mask)
            masked_flow[1, :, :] = regionfill(masked_flow[1, :, :], 1 - previous_mask)
            previous_mask = torch.as_tensor(previous_mask).unsqueeze(0).unsqueeze(0)
            masked_flow = torch.as_tensor(masked_flow).unsqueeze(0)

            # transforms.ToPILImage()(previous_frame.cpu()[0]).save('1.png')
            # transforms.ToPILImage()(current_frame.cpu()[0]).save('2.png')
            # Image.fromarray(fz.convert_from_flow(flow.cpu().detach().numpy()[0].transpose(1, 2, 0))).save('3.png')
            # Image.fromarray(fz.convert_from_flow(masked_flow.cpu().detach().numpy()[0].transpose(1, 2, 0))).save('4.png')

            print(time() - start)
