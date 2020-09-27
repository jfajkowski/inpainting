import cv2 as cv
import numpy as np
import torch
import flowiz as fz
import torchvision
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor


def annotation_to_mask(image, object_id=1):
    image = np.array(image)

    if isinstance(object_id, list):
        mask = np.zeros(image.shape, dtype=np.uint8)
        for o in object_id:
            mask += annotation_to_mask(image, o)
        return mask

    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[image == object_id] = 255
    return Image.fromarray(mask).convert('L')


def mask_to_bbox(mask):
    if isinstance(mask, torch.Tensor):
        mask = tensor_to_mask(mask.cpu())
    mask = np.array(mask)
    if np.any(mask):
        cols = np.any(mask, axis=0)
        rows = np.any(mask, axis=1)
        x1, x2 = np.where(cols)[0][[0, -1]]
        y1, y2 = np.where(rows)[0][[0, -1]]
        return (x1, y1), (x2, y2)
    else:
        return None


def convert_tensor(tensor: torch.Tensor, frame_type):
    if frame_type == 'image':
        return tensor_to_image(tensor)
    elif frame_type == 'mask':
        return tensor_to_mask(tensor)
    elif frame_type == 'flow':
        return tensor_to_flow(tensor)
    elif frame_type == 'flowviz':
        return tensor_to_image(flow_tensor_to_image_tensor(tensor))
    else:
        raise ValueError(frame_type)


def tensor_to_image(image_tensor: torch.Tensor, rgb2bgr: bool = True):
    mat = (image_tensor * 255).type(torch.uint8).squeeze().permute(1, 2, 0).cpu().numpy()
    if rgb2bgr:
        mat = cv.cvtColor(mat, cv.COLOR_RGB2BGR)
    return mat


def tensor_to_mask(mask_tensor: torch.Tensor):
    return (mask_tensor * 255).type(torch.uint8).squeeze().cpu().numpy()


def tensor_to_flow(flow_tensor: torch.Tensor):
    return flow_tensor.detach().cpu().numpy().transpose(1, 2, 0)


def cv_image_to_tensor(mat: np.ndarray):
    mat = cv.cvtColor(mat, cv.COLOR_BGR2RGB)
    return torch.from_numpy(mat).float().permute(2, 0, 1) / 255


def tensor_to_pil_image(tensor):
    assert 3 <= len(tensor.size()) <= 4
    if len(tensor.size()) == 4:
        tensor = torchvision.utils.make_grid(tensor)
    return to_pil_image(tensor.detach().cpu())


def flow_tensor_to_image_tensor(tensor):
    assert 3 <= len(tensor.size()) <= 4

    if len(tensor.size()) == 4:
        b, _, h, w = tensor.size()
        result = torch.zeros((b, 3, h, w))
        for i in range(b):
            result[i, :, :, :] = flow_tensor_to_image_tensor(tensor[i, :, :, :])
        return result

    return to_tensor(fz.convert_from_flow(tensor_to_flow(tensor)))


def normalize_image(x, mode='standard'):
    mean, std = _mean_and_std(mode)
    y = x.clone()
    y[:, 0, :, :] = (y[:, 0, :, :] - mean[0]) / std[0]
    y[:, 1, :, :] = (y[:, 1, :, :] - mean[1]) / std[1]
    y[:, 2, :, :] = (y[:, 2, :, :] - mean[2]) / std[2]
    return y


def denormalize_image(y, mode='standard'):
    mean, std = _mean_and_std(mode)
    x = y.clone()
    x[:, 0, :, :] = x[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = x[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = x[:, 2, :, :] * std[2] + mean[2]
    return torch.clamp(x, 0, 1)


def _mean_and_std(mode='standard'):
    if mode == 'imagenet':
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif mode == 'standard':
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    elif mode == 'minmax':
        return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    else:
        raise ValueError(mode)


def normalize_flow(flow):
    _, _, h, w = flow.size()
    flow[:, 0, :, :] /= float(w)
    flow[:, 1, :, :] /= float(h)
    return flow


def denormalize_flow(flow):
    _, _, h, w = flow.size()
    flow[:, 0, :, :] *= float(w)
    flow[:, 1, :, :] *= float(h)
    return flow


def dilate_mask(x: torch.Tensor, size=3, iterations=3):
    if len(x.size()) == 3:
        x = x.unsqueeze(0)
        x = dilate_mask(x, size, iterations)
        return x.squeeze(0)

    structuring_element = torch.ones((size, size)).view(1, 1, size, size).to(x.device)
    for i in range(iterations):
        x = (F.conv2d(x, structuring_element, stride=1, padding=(size // 2, size // 2)) > 0).float()

    return x


def warp_tensor(x, flow):
    assert x.size()[-2:] == flow.size()[-2:]
    grid = make_grid(x.size(), normalized=True).to(x.device)
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
