import torch

from PIL import Image
from torch.nn.functional import conv2d
from torchvision.transforms.functional import to_tensor, to_pil_image

before = Image.open('../data/raw/mask/demo/mask.png').convert('L')

tensor_before = to_tensor(before).unsqueeze(0)
structuring_element = torch.ones((3, 3)).view(1, 1, 3, 3)
tensor_after = (conv2d(tensor_before, structuring_element, stride=1, padding=(3 // 2, 3 // 2)) > 0).float()

after = to_pil_image(tensor_after.squeeze(0))

before.show()
after.show()
