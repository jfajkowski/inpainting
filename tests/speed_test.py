import time

import torch
import torchvision
from apex.amp import amp
from torchvision.transforms import transforms

def run():
    transform = transforms.Compose([
        transforms.Resize(1024),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, num_workers=1)

    net = torchvision.models.mobilenet_v2(pretrained=True).eval().half().cuda()
    net.fc = torch.nn.Identity()
    net.classifier = torch.nn.Identity()
    for img, _ in iter(loader):
        start = time.perf_counter()
        net(img.half().cuda())
        end = time.perf_counter()
        print(f'{1 / (end - start):.2f} FPS', end='\r')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    with torch.no_grad():
        run()
