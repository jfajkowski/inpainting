import math

import flowiz as fz
import pwc.model as pwc
import skimage.io as io
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image


def estimate_flow(model, x_1, x_2):
    assert (x_1.size(1) == x_2.size(1))
    assert (x_1.size(2) == x_2.size(2))

    width = x_1.size(2)
    height = x_1.size(1)

    tensor_preprocessed_first = x_1.cuda().view(1, 3, height, width)
    tensor_preprocessed_second = x_2.cuda().view(1, 3, height, width)

    preprocessed_width = int(math.floor(math.ceil(width / 64.0) * 64.0))
    preprocessed_height = int(math.floor(math.ceil(height / 64.0) * 64.0))

    tensor_preprocessed_first = torch.nn.functional.interpolate(input=tensor_preprocessed_first,
                                                                size=(preprocessed_height, preprocessed_width),
                                                                mode='bilinear', align_corners=False)
    tensor_preprocessed_second = torch.nn.functional.interpolate(input=tensor_preprocessed_second,
                                                                 size=(preprocessed_height, preprocessed_width),
                                                                 mode='bilinear', align_corners=False)

    flow = 20.0 * torch.nn.functional.interpolate(
        input=model(tensor_preprocessed_first, tensor_preprocessed_second), size=(height, width),
        mode='bilinear', align_corners=False)

    flow[:, 0, :, :] *= float(width) / float(preprocessed_width)
    flow[:, 1, :, :] *= float(height) / float(preprocessed_height)

    return flow[0, :, :, :].cpu()


def warp_tensor(x, flow, padding_mode='zeros'):
    assert x.size()[-2:] == flow.size()[-2:]
    _, h, w = x.size()
    x = x.cuda().view(1, 3, h, w)
    flow = flow.cuda().view(1, 2, h, w)
    flow[:, 0, :, :] /= float(w)
    flow[:, 1, :, :] /= float(h)
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float().cuda()
    grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
    grid += 2 * flow
    grid = grid.permute(0, 2, 3, 1)
    return F.grid_sample(x, grid, padding_mode=padding_mode)[0, :, :, :].cpu()


if __name__ == '__main__':
    image_first = Image.open('../data/raw/demo/JPEGImages/00000.jpg')
    image_second = Image.open('../data/raw/demo/JPEGImages/00001.jpg')

    tensor_first = to_tensor(image_first).float().flip(2) * (1.0 / 255.0)
    tensor_second = to_tensor(image_second).float().flip(2) * (1.0 / 255.0)

    model = pwc.Network().cuda().eval()
    tensor_flow = estimate_flow(model, tensor_first, tensor_second).detach().flip(2)
    tensor_warp = warp_tensor(tensor_first, tensor_flow).flip(2) * 255.0

    image_flow = fz.convert_from_flow(tensor_flow.numpy().transpose(1, 2, 0))
    image_warp = to_pil_image(tensor_warp)

    io.imshow_collection([image_first, image_second, image_flow, image_warp])
    io.show()
