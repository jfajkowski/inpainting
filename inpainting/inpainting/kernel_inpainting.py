import torch


class Inpainter:
    def __init__(self, rtol=1e-05, atol=1e-08, check_interval=100):
        self.rtol = rtol
        self.atol = atol
        self.check_interval = check_interval
        self.kernel = torch.tensor([[1.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 1.0]]) / 8
        self.weights = None

    def __call__(self, image, mask):
        image_b, image_c, image_h, image_w = image.size()

        if self.weights is None or self.weights.size(0) != image_b * image_c or self.weights.device != image.device:
            self.weights = self.kernel.repeat(image_b * image_c, 1, 1, 1).to(image.device)

        n = 0
        result = image * (1 - mask)
        while True:
            prev_result = result
            result = result * (1 - mask) + torch.nn.functional.conv2d(
                result.view(1, image_b * image_c, image_h, image_w), self.weights, padding=1, groups=image_b * image_c
            ).view(image_b, image_c, image_h, image_w) * mask
            n += 1
            if n % self.check_interval == 0 and torch.allclose(result, prev_result, self.rtol, self.atol):
                break
        return result
