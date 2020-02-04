import math

import flowiz as fz
import skimage.io as io
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from deepstab.pwcnet.model import Network
from deepstab.region_fill import regionfill


def estimate_flow(model, x_1, x_2):
    assert (x_1.size(2) == x_2.size(2))
    assert (x_1.size(3) == x_2.size(3))

    height = x_1.size(2)
    width = x_1.size(3)

    preprocessed_width = int(math.floor(math.ceil(width / 64.0) * 64.0))
    preprocessed_height = int(math.floor(math.ceil(height / 64.0) * 64.0))

    tensor_preprocessed_first = torch.nn.functional.interpolate(input=x_1,
                                                                size=(preprocessed_height, preprocessed_width),
                                                                mode='bilinear', align_corners=False)
    tensor_preprocessed_second = torch.nn.functional.interpolate(input=x_2,
                                                                 size=(preprocessed_height, preprocessed_width),
                                                                 mode='bilinear', align_corners=False)

    flow = torch.nn.functional.interpolate(
        input=model(tensor_preprocessed_first, tensor_preprocessed_second), size=(height, width),
        mode='bilinear', align_corners=False)

    flow[:, 0, :, :] *= float(width) / float(preprocessed_width)
    flow[:, 1, :, :] *= float(height) / float(preprocessed_height)

    return flow


def custom_warp(x, flow):
    return


def warp_tensor(x, flow, mode='bilinear', padding_mode='zeros'):
    assert x.size()[-2:] == flow.size()[-2:]
    flow = normalize_flow(flow.clone())
    grid = make_grid(x)
    grid += 2 * flow
    grid = grid.permute(0, 2, 3, 1)
    return F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode)


def normalize_flow(flow):
    _, _, h, w = flow.size()
    flow[:, 0, :, :] /= float(w)
    flow[:, 1, :, :] /= float(h)
    return flow


def make_grid(x, normalized=True):
    _, _, h, w = x.size()
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float().cuda()
    grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
    if normalized:
        grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
        grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
    return grid


def fill_flow(flow, mask):
    mask = mask.squeeze(0).squeeze(0).cpu().detach().numpy()
    masked_flow = flow.squeeze(0).cpu().detach().numpy()
    masked_flow[0, :, :] = regionfill(masked_flow[0, :, :], 1 - mask)
    masked_flow[1, :, :] = regionfill(masked_flow[1, :, :], 1 - mask)
    return torch.as_tensor(masked_flow).unsqueeze(0).cuda()


if __name__ == '__main__':
    image_first = Image.open('../data/raw/video/DAVIS/JPEGImages/Full-Resolution/basketball-game/00000.jpg')
    image_second = Image.open('../data/raw/video/DAVIS/JPEGImages/Full-Resolution/basketball-game/00001.jpg')
    image_third = Image.open('../data/raw/video/DAVIS/JPEGImages/Full-Resolution/basketball-game/00002.jpg')

    tensor_first = to_tensor(image_first).flip(2)
    tensor_second = to_tensor(image_second).flip(2)
    tensor_third = to_tensor(image_third).flip(2)

    model = Network('../models/pwcnet/network-default.pytorch').cuda().eval()

    tensor_flow = estimate_flow(model, tensor_first, tensor_second).detach().flip(2)
    tensor_warp = warp_tensor(tensor_first.flip(2), tensor_flow)
    image_forward_flow = Image.fromarray(fz.convert_from_flow(tensor_flow.numpy().transpose(1, 2, 0)))
    image_forward_warp = to_pil_image(tensor_warp)

    tensor_flow = estimate_flow(model, tensor_third, tensor_second).detach().flip(2)
    tensor_warp = warp_tensor(tensor_third.flip(2), tensor_flow)
    image_backward_flow = Image.fromarray(fz.convert_from_flow(tensor_flow.numpy().transpose(1, 2, 0)))
    image_backward_warp = to_pil_image(tensor_warp)

    io.imshow_collection([image_first, image_second, image_third])
    io.show()
    io.imshow_collection([image_forward_flow, image_forward_warp])
    io.show()
    io.imshow_collection([image_backward_flow, image_backward_warp])
    io.show()

    image_first.save('image_first.jpg')
    image_second.save('image_second.jpg')
    image_third.save('image_third.jpg')
    image_forward_flow.save('image_forward_flow.jpg')
    image_backward_flow.save('image_backward_flow.jpg')
    image_forward_warp.save('image_forward_warp.jpg')
    image_backward_warp.save('image_backward_warp.jpg')
