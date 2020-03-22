import torch
import torch.nn.functional as F

from inpainting.external.deepflowguidedvideoinpainting.region_fill import regionfill


def warp_tensor(x, flow, mode='bilinear', padding_mode='zeros'):
    assert x.size()[-2:] == flow.size()[-2:]
    flow = normalize_flow(flow.clone())
    grid = make_grid(x.size())
    grid += 2 * flow
    grid = grid.permute(0, 2, 3, 1)
    return F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode)


def normalize_flow(flow):
    _, _, h, w = flow.size()
    flow[:, 0, :, :] /= float(w)
    flow[:, 1, :, :] /= float(h)
    return flow


def make_grid(size, normalized=True):
    _, _, h, w = size
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
