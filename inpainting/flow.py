import math

import skimage.io as io
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from inpainting.external.flow_models.liteflownet import Network
from inpainting.region_fill import regionfill
from inpainting.visualize import flow_to_pil_image


def estimate_flow(model, x_1, x_2):
    assert (x_1.size(2) == x_2.size(2))
    assert (x_1.size(3) == x_2.size(3))

    height = x_1.size(2)
    width = x_1.size(3)

    preprocessed_width = int(math.floor(math.ceil(width / 64.0) * 64.0))
    preprocessed_height = int(math.floor(math.ceil(height / 64.0) * 64.0))

    # Convert to BGR
    x_1 = x_1.flip(1)
    x_2 = x_2.flip(1)

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
    with torch.no_grad():
        image_first = Image.open('data/raw/video/DAVIS/JPEGImages/480p/rollerblade/00000.jpg')
        image_second = Image.open('data/raw/video/DAVIS/JPEGImages/480p/rollerblade/00001.jpg')
        image_third = Image.open('data/raw/video/DAVIS/JPEGImages/480p/rollerblade/00002.jpg')

        tensor_first = to_tensor(image_first).unsqueeze(0).cuda()
        tensor_second = to_tensor(image_second).unsqueeze(0).cuda()
        tensor_third = to_tensor(image_third).unsqueeze(0).cuda()

        model = Network('models/liteflownet/network-default.pytorch').cuda().eval()

        tensor_flow = estimate_flow(model, tensor_second, tensor_first)
        tensor_warp = warp_tensor(tensor_first, tensor_flow)
        image_forward_flow = flow_to_pil_image(tensor_flow.squeeze().cpu())
        image_forward_warp = to_pil_image(tensor_warp.squeeze().cpu())

        tensor_flow = estimate_flow(model, tensor_second, tensor_second)
        image_no_flow = flow_to_pil_image(tensor_flow.squeeze().cpu())

        tensor_flow = estimate_flow(model, tensor_second, tensor_third)
        tensor_warp = warp_tensor(tensor_third, tensor_flow)
        image_backward_flow = flow_to_pil_image(tensor_flow.squeeze().cpu())
        image_backward_warp = to_pil_image(tensor_warp.squeeze().cpu())

        io.imshow_collection([image_first, image_second, image_third])
        io.show()
        io.imshow_collection([image_forward_flow, image_no_flow, image_backward_flow])
        io.show()
        io.imshow_collection([image_forward_warp, image_backward_warp])
        io.show()
