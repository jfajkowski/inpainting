import torch


def inpaint(image, mask, rtol=1e-03, atol=1e-05):
    image_b, image_c, image_h, image_w = image.size()
    mask_b, mask_c, mask_h, mask_w = mask.size()

    kernel = torch.tensor([[1.0, 1.0, 1.0],
                           [1.0, 0.0, 1.0],
                           [1.0, 1.0, 1.0]]) / 8
    image_weights = kernel.repeat((image_b * image_c, 1, 1, 1)).to(image.device)

    image = image.view(1, image_b * image_c, image_h, image_w)
    mask = mask.view(1, mask_b * mask_c, mask_h, mask_w)

    result = (image * (1 - mask))
    while True:
        prev_result = result
        result = result * (1 - mask) + torch.nn.functional.conv2d(result, image_weights, padding=1,
                                                                  groups=image_b * image_c) * mask
        if torch.allclose(result, prev_result, rtol, atol):
            break
    return result.view(image_b, image_c, image_h, image_w)
