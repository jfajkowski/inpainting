import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from inpainting.load import ImageDataset
from inpainting.utils import denormalize, mean_and_std

image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # transforms.Normalize(*mean_and_std())
])
dataset = ImageDataset(['data/raw/image/SmallPlaces2/data_large'], transform=image_transforms)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)


class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.out = nn.Conv2d(3, 3, 1)

    def forward(self, x):
        return self.out(x)


model = Network().eval()

with torch.no_grad():
    image = next(iter(data_loader))
    image = image

    model.out.weight = torch.nn.Parameter(torch.tensor(
        [[[[1]], [[0]], [[0]]],
         [[[0]], [[1]], [[0]]],
         [[[0]], [[0]], [[1]]]]).float(), requires_grad=False)
    model.out.bias = torch.nn.Parameter(torch.tensor(
        [0, 0, 0]).float(), requires_grad=False)
    image_filled = model(image)
    # save_image(denormalize(image), 'image.png')
    # save_image(denormalize(image_filled), 'manual.png')
    to_pil_image(make_grid(denormalize(image))).show()
    to_pil_image(make_grid(denormalize(image_filled))).show()
