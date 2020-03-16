import glob
import time

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataloader import DataLoader

from inpainting.external.liteflownet import Network
from inpainting.inpainting import FlowAndFillInpaintingAlgorithm
from inpainting.load import VideoDataset, DynamicMaskVideoDataset
from scripts.train import InpaintingModel

batch_size = 1
sizes = [
    (256, 256),
    (512, 512),
    (1024, 1024)
]

for size in sizes:
    frame_dataset = VideoDataset(
        list(glob.glob('data/raw/video/DAVIS/JPEGImages/480p/flamingo')),
        frame_type='image',
        transform=transforms.Compose([
            transforms.Resize(size, interpolation=Image.BILINEAR),
            transforms.ToTensor()
        ]))
    mask_dataset = VideoDataset(
        list(glob.glob('data/processed/video/DAVIS/Annotations_dilated/480p/flamingo')),
        frame_type='mask',
        transform=transforms.Compose([
            transforms.Resize(size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ]))
    dataset = DynamicMaskVideoDataset(frame_dataset, mask_dataset)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        flow_model = Network('models/external/flownet2/liteflownet/network-default.pytorch').cuda().eval()
        fill_model = InpaintingModel.load_from_checkpoint(
            'models/baseline/version_0/checkpoints/_ckpt_epoch_96.ckpt').generator.cuda().eval()
        inpainting_algorithm = FlowAndFillInpaintingAlgorithm(flow_model, fill_model)

        # First dry run
        for i in range(0):
            inpainting_algorithm.inpaint_online(
                torch.randn((batch_size, 3, *size)).cuda(),
                torch.randn((batch_size, 1, *size)).cuda()
            )

        data_iter = iter(data_loader)

        import cProfile, pstats
        pr = cProfile.Profile()
        pr.enable()
        for sample in data_iter:
            start = time.perf_counter()

            frames, masks, _ = sample
            frames = list(map(lambda x: x.cuda(), frames))
            masks = list(map(lambda x: x.cuda(), masks))
            inpainting_algorithm.reset()
            frames_filled, masks_filled = inpainting_algorithm.inpaint(frames, masks)
            frames_filled = list(map(lambda x: x.cpu(), frames_filled))
            masks_filled = list(map(lambda x: x.cpu(), masks_filled))

            end = time.perf_counter()
            print(len(frames) / (end - start))

        pr.disable()
        sortby = 'cumulative'
        ps = pstats.Stats(pr).sort_stats(sortby)
        ps.dump_stats(f'benchmark_{size[0]}x{size[1]}.pstat')
