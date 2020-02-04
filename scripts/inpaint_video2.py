import glob
from time import time

import flowiz as fz
import torch
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from deepstab.flow import estimate_flow, warp_tensor
from deepstab.liteflownet import Network
from deepstab.load import VideoDataset, DynamicMaskVideoDataset
from deepstab.region_fill import regionfill
from deepstab.utils import mask_tensor, extract_mask, dilate

length = 34
batch_size = 1
epochs = 1
learning_rate = 1e-3
size = (256, 256)
eps = 100

frame_dataset = VideoDataset(
    list(glob.glob('../data/raw/video/DAVIS/JPEGImages/480p/lady-running')),
    sequence_length=length + 1,
    transform=transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ]))
mask_dataset = VideoDataset(
    list(glob.glob('../data/raw/video/DAVIS/Annotations_unsupervised/480p/rollerblade')),
    sequence_length=length + 1,
    transform=transforms.Compose([
        transforms.Lambda(extract_mask),
        transforms.Resize(size),
        transforms.ToTensor()
    ]))
dataset = DynamicMaskVideoDataset(frame_dataset, mask_dataset)
# mask_dataset = RectangleMaskDataset(
#     size[1], size[0],
#     (128 - 16, 128 - 16, 32, 32),
#     # '../data/raw/mask/demo',
#     # '../data/raw/mask/qd_imd/test',
#     # mask_mode='L',
#     # shuffle=True,
#     transform=transforms.Compose([
#         transforms.Resize(size),
#         # transforms.Lambda(lambda x: ImageOps.invert(x)),
#         transforms.ToTensor()
#     ]))
# dataset = StaticMaskVideoDataset(frame_dataset, mask_dataset)

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

flow_model = Network('../models/liteflownet/network-default.pytorch').eval().cuda()


# state = torch.load('../models/deepfill/gen_00430000.pt')
# config = yaml.load(open('../models/deepfill/deepfill.yml'), Loader=yaml.FullLoader)
# inpainting_model = Generator(config['netG'], True, 0).cuda().eval()
# inpainting_model.load_state_dict(state)


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
        input_frames, input_masks, _ = sample
        # input_frames = list(map(lambda x: mask_tensor(x[0], x[1]), zip(input_frames, input_masks)))
        output_frames, output_masks = [], []

        previous_output_frame = input_frames[0].cuda()
        previous_output_mask = 1 - dilate(1 - input_masks[0].cuda(), 5)
        for i in range(1, length):
            start = time()
            previous_frame = input_frames[i - 1].cuda()
            previous_mask = 1 - dilate(1 - input_masks[i - 1].cuda(), 5)
            current_frame = input_frames[i].cuda()
            current_mask = 1 - dilate(1 - input_masks[i].cuda(), 5)

            forward_flow = estimate_flow(flow_model, previous_frame, current_frame)
            backward_flow = estimate_flow(flow_model, current_frame, previous_frame)

            filled_forward_flow = fill_flow(forward_flow, current_mask)
            filled_backward_flow = fill_flow(backward_flow, previous_mask)

            masked_previous_output_frame = mask_tensor(previous_output_frame, previous_output_mask)
            masked_current_frame = mask_tensor(current_frame, current_mask)

            warped_frame = warp_tensor(masked_previous_output_frame, filled_forward_flow, padding_mode='zeros')
            dewarped_frame = warp_tensor(warped_frame, filled_backward_flow, padding_mode='zeros')

            connected_pixels_mask = (
                        (torch.sum(torch.abs(dewarped_frame - masked_previous_output_frame), dim=1) / 3) < (
                            eps / 255)).float().unsqueeze(1)

            known_pixels_mask = warp_tensor(connected_pixels_mask * previous_output_mask, backward_flow,
                                            padding_mode='zeros') * (1 - current_mask)
            known_pixels_frame = warped_frame * known_pixels_mask

            output_mask = current_mask + known_pixels_mask
            output_frame = current_frame * current_mask + known_pixels_frame

            output_frame = output_frame.detach()
            output_mask = output_mask.detach()

            output_frames.append(output_frame.cpu())
            output_masks.append(output_mask.cpu())

            transforms.ToPILImage()(previous_frame.cpu()[0]).save('0_previous_frame.png')
            transforms.ToPILImage()(current_frame.cpu()[0]).save('0_current_frame.png')
            transforms.ToPILImage()(previous_mask.cpu()[0]).save('1_previous_mask.png')
            transforms.ToPILImage()(current_mask.cpu()[0]).save('1_current_mask.png')
            Image.fromarray(fz.convert_from_flow(forward_flow.cpu().detach().numpy()[0].transpose(1, 2, 0))).save(
                '2_forward_flow.png')
            Image.fromarray(fz.convert_from_flow(backward_flow.cpu().detach().numpy()[0].transpose(1, 2, 0))).save(
                '2_backward_flow.png')
            Image.fromarray(
                fz.convert_from_flow(filled_forward_flow.cpu().detach().numpy()[0].transpose(1, 2, 0))).save(
                '2_filled_forward_flow.png')
            Image.fromarray(
                fz.convert_from_flow(filled_backward_flow.cpu().detach().numpy()[0].transpose(1, 2, 0))).save(
                '2_filled_backward_flow.png')
            transforms.ToPILImage()(masked_previous_output_frame.cpu()[0]).save('3_masked_previous_frame.png')
            transforms.ToPILImage()(masked_current_frame.cpu()[0]).save('3_masked_current_frame.png')
            transforms.ToPILImage()(warped_frame.cpu()[0]).save('4_warped_frame.png')
            transforms.ToPILImage()(dewarped_frame.cpu()[0]).save('4_dewarped_frame.png')
            transforms.ToPILImage()(connected_pixels_mask.cpu()[0]).save('5_connected_pixels_mask.png')
            transforms.ToPILImage()(known_pixels_mask.cpu()[0]).save('5_known_pixels_warped_mask.png')
            transforms.ToPILImage()(known_pixels_frame.cpu()[0]).save('5_known_pixels_warped_frame.png')
            transforms.ToPILImage()(output_frame.cpu()[0]).save('6_output_frame.png')
            transforms.ToPILImage()(output_mask.cpu()[0]).save('6_output_mask.png')

            previous_output_frame = output_frame
            previous_output_mask = output_mask

            print(time() - start)
