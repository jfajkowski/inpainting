import torch.nn.functional as F
from PIL import Image

from torchvision.transforms import transforms

image = Image.open('../data/raw/image/demo/image.jpg')
tensor = transforms.F.to_tensor(image).view(1, 3, image.height, image.width)

image_size = image.size
kernel_size = (256, 256)
padding = (256, 256)
stride = (256, 256)

unfolded = F.unfold(tensor, kernel_size=kernel_size, padding=padding, stride=stride)
folded = F.fold(unfolded, output_size=image_size[::-1], kernel_size=kernel_size, padding=padding, stride=stride)

for i in range(unfolded.size()[2]):
    transforms.F.to_pil_image(unfolded[0, :, :].permute(1, 0).view(-1, 3, 256, 256)[i, :, :, :]).show()
transforms.F.to_pil_image(folded[0, :, :, :]).show()
