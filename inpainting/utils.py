import cv2 as cv
import numpy as np
import torch

from torch.nn import functional as F
from PIL import Image


def annotation_to_mask(image, object_id=0):
    image = np.array(image)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[image == object_id] = 255
    return Image.fromarray(mask).convert('L')


def mask_to_bbox(mask):
    if isinstance(mask, torch.Tensor):
        mask = tensor_to_cv_mask(mask.cpu())
    mask = np.array(mask)
    if np.any(mask):
        cols = np.any(mask, axis=0)
        rows = np.any(mask, axis=1)
        x1, x2 = np.where(cols)[0][[0, -1]]
        y1, y2 = np.where(rows)[0][[0, -1]]
        return (x1, y1), (x2, y2)
    else:
        return None


def tensor_to_cv_image(image_tensor: torch.Tensor, rgb2bgr: bool = True):
    mat = (image_tensor * 255).type(torch.uint8).squeeze().permute(1, 2, 0).cpu().numpy()
    if rgb2bgr:
        mat = cv.cvtColor(mat, cv.COLOR_RGB2BGR)
    return mat


def tensor_to_cv_mask(mask_tensor: torch.Tensor):
    return (mask_tensor * 255).type(torch.uint8).squeeze().numpy()


def cv_image_to_tensor(mat: np.ndarray, bgr2rgb: bool = True):
    if bgr2rgb:
        mat = cv.cvtColor(mat, cv.COLOR_BGR2RGB)
    return torch.from_numpy(mat).float().permute(2, 0, 1) / 255


def mask_tensor(x, m):
    return x * m


def invert_mask(m):
    return 1 - m


def normalize(x, mode='standard'):
    mean, std = mean_and_std(mode)
    y = x.clone()
    y[:, 0, :, :] = (y[:, 0, :, :] - mean[0]) / std[0]
    y[:, 1, :, :] = (y[:, 1, :, :] - mean[1]) / std[1]
    y[:, 2, :, :] = (y[:, 2, :, :] - mean[2]) / std[2]
    return y


def denormalize(y, mode='standard'):
    mean, std = mean_and_std(mode)
    x = y.clone()
    x[:, 0, :, :] = x[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = x[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = x[:, 2, :, :] * std[2] + mean[2]
    return torch.clamp(x, 0, 1)


def mean_and_std(mode='standard'):
    if mode == 'imagenet':
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif mode == 'standard':
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    elif mode == 'minmax':
        return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    else:
        raise ValueError(mode)


def normalize_flow(flow):
    h, w, _ = flow.shape
    flow[:, :, 0] /= float(w)
    flow[:, :, 1] /= float(h)
    return flow


def denormalize_flow(flow):
    h, w, _ = flow.shape
    flow[:, :, 0] *= float(w)
    flow[:, :, 1] *= float(h)
    return flow


def dilate(x, size=3, iterations=3):
    structuring_element = torch.ones((size, size)).view(1, 1, size, size).to(x.device)
    for i in range(iterations):
        x = (F.conv2d(x, structuring_element, stride=1, padding=(size // 2, size // 2)) > 0).float()
    return x


def warp_tensor(x, flow):
    assert x.size()[-2:] == flow.size()[-2:]
    grid = make_grid(x.size()).to(x.device)
    grid += 2 * flow
    grid = grid.permute(0, 2, 3, 1)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)


def make_grid(size, normalized=True):
    b, _, h, w = size
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float()
    grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
    if normalized:
        grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
        grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
    return grid.repeat(b, 1, 1, 1)
