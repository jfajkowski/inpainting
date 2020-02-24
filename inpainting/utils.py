import torch


# generates superresolution mask
# mask = np.zeros((32, 32))
# for x in range(32):
#     for y in range(32):
#         if (x + y) % 2 == 0:
#             mask[x, y] = 255


def mask_tensor(x, m):
    return x * m


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