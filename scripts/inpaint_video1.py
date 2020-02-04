import glob
from time import time

import flowiz as fz
import torch
from PIL import Image
from PIL import ImageOps
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from deepstab.flow import estimate_flow, warp_tensor, fill_flow
from deepstab.liteflownet import Network
from deepstab.load import VideoDataset, DynamicMaskVideoDataset
from deepstab.model_gatingconvolution import GatingConvolutionUNet
from deepstab.utils import mask_tensor, extract_mask, dilate

length = 64
batch_size = 1
epochs = 1
learning_rate = 1e-3
size = (256, 256)
eps = 2.5

frame_dataset = VideoDataset(
    list(glob.glob('../data/raw/video/DAVIS/JPEGImages/480p/lady-running')),
    sequence_length=length + 1,
    transform=transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ]))
mask_dataset = VideoDataset(
    list(glob.glob('../data/raw/video/DAVIS/Annotations_unsupervised/480p/lady-running')),
    sequence_length=length + 1,
    transform=transforms.Compose([
        transforms.Lambda(extract_mask),
        transforms.Lambda(ImageOps.invert),
        # transforms.Lambda(bbox),
        transforms.Lambda(lambda x: dilate(x, 51)),
        transforms.Lambda(ImageOps.invert),
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

state = torch.load('../models/20200122_gatingconvunet_gan/model_epoch_275_lr_0.0001.pth')
inpainting_model = GatingConvolutionUNet().cuda().eval()
inpainting_model.load_state_dict(state['generator'])

with torch.no_grad():
    for e in range(epochs):
        data_iter = iter(data_loader)

        # for sample in data_iter:
        for sample in [next(data_iter)]:
            input_frames, input_masks, _ = sample
            output_frames, output_masks, inpainted_frames = [], [], []

            previous_output_frame = input_frames[0].cuda()
            previous_output_mask = input_masks[0].cuda()
            for i in range(1, length):
                start = time()
                previous_frame = input_frames[i - 1].cuda()
                previous_mask = input_masks[i - 1].cuda()
                current_frame = input_frames[i].cuda()
                current_mask = input_masks[i].cuda()

                forward_flow = estimate_flow(flow_model, previous_frame, current_frame)
                masked_forward_flow = mask_tensor(forward_flow, current_mask)
                filled_forward_flow = fill_flow(masked_forward_flow, current_mask)

                backward_flow = estimate_flow(flow_model, current_frame, previous_frame)
                masked_backward_flow = mask_tensor(backward_flow, previous_mask)
                filled_backward_flow = fill_flow(masked_backward_flow, previous_mask)

                masked_previous_output_frame = mask_tensor(previous_output_frame, previous_output_mask)
                masked_current_frame = mask_tensor(current_frame, current_mask)

                propagation_error = warp_tensor(
                    (warp_tensor(filled_forward_flow, filled_backward_flow, mode='nearest') + filled_backward_flow) / 2,
                    filled_forward_flow, mode='nearest')
                connected_pixels_mask = (torch.norm(propagation_error, 2, dim=1) < eps).float()

                warped_mask = warp_tensor(connected_pixels_mask * previous_output_mask, filled_backward_flow,
                                          mode='nearest')
                warped_frame = warp_tensor(masked_previous_output_frame, filled_backward_flow, mode='bilinear')

                output_mask = current_mask + warped_mask * (1 - current_mask)
                output_frame = current_frame * current_mask + warped_frame * warped_mask * (1 - current_mask) + (
                            1 - output_mask)

                inpainted_frame = inpainting_model(output_frame, output_mask)

                output_frames.append(output_frame.cpu())
                output_masks.append(output_mask.cpu())
                inpainted_frames.append(inpainted_frame.cpu())

                transforms.ToPILImage()(previous_frame.cpu()[0]).save('0_previous_frame.png')
                transforms.ToPILImage()(previous_output_frame.cpu()[0]).save('0_previous_output_frame.png')
                transforms.ToPILImage()(current_frame.cpu()[0]).save('0_current_frame.png')
                transforms.ToPILImage()(previous_mask.cpu()[0]).save('1_previous_mask.png')
                transforms.ToPILImage()(previous_output_mask.cpu()[0]).save('1_previous_output_mask.png')
                transforms.ToPILImage()(current_mask.cpu()[0]).save('1_current_mask.png')
                Image.fromarray(fz.convert_from_flow(forward_flow.cpu().detach().numpy()[0].transpose(1, 2, 0))).save(
                    '2_forward_flow.png')
                Image.fromarray(
                    fz.convert_from_flow(masked_forward_flow.cpu().detach().numpy()[0].transpose(1, 2, 0))).save(
                    '2_masked_forward_flow.png')
                Image.fromarray(
                    fz.convert_from_flow(filled_forward_flow.cpu().detach().numpy()[0].transpose(1, 2, 0))).save(
                    '2_filled_forward_flow.png')
                Image.fromarray(fz.convert_from_flow(backward_flow.cpu().detach().numpy()[0].transpose(1, 2, 0))).save(
                    '2_backward_flow.png')
                Image.fromarray(
                    fz.convert_from_flow(masked_backward_flow.cpu().detach().numpy()[0].transpose(1, 2, 0))).save(
                    '2_masked_backward_flow.png')
                Image.fromarray(
                    fz.convert_from_flow(filled_backward_flow.cpu().detach().numpy()[0].transpose(1, 2, 0))).save(
                    '2_filled_backward_flow.png')
                transforms.ToPILImage()(masked_previous_output_frame.cpu()[0]).save(
                    '3_masked_previous_output_frame.png')
                transforms.ToPILImage()(masked_current_frame.cpu()[0]).save('3_masked_current_frame.png')
                transforms.ToPILImage()(
                    fz.convert_from_flow(propagation_error.cpu().detach().numpy()[0].transpose(1, 2, 0))).save(
                    '4_propagation_error.png')
                transforms.ToPILImage()(connected_pixels_mask.cpu()[0]).save('4_connected_pixels_mask.png')
                transforms.ToPILImage()(warped_mask.cpu()[0]).save('4_warped_mask.png')
                transforms.ToPILImage()(warped_frame.cpu()[0]).save('4_warped_frame.png')
                transforms.ToPILImage()(output_frame.cpu()[0]).save('5_output_frame.png')
                transforms.ToPILImage()(output_mask.cpu()[0]).save('5_output_mask.png')
                transforms.ToPILImage()(inpainted_frame.cpu()[0]).save('5_inpainted_frame.png')

                previous_output_frame = output_frame
                previous_output_mask = output_mask

                print(time() - start)
