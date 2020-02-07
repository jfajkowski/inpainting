import glob

import torch
from apex.amp import amp, scale_loss
from deepstab.partialconv.loss import VGG16PartialLoss
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from deepstab.load import VideoDataset

time = 8
batch_size = 1
epochs = 1000
learning_rate = 1e-3
size = (256, 256)

transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

video_dataset = VideoDataset(list(glob.glob('../data/raw/video/DAVIS/JPEGImages/480p/*')), sequence_length=time + 1,
                             transform=transform)
data_loader = DataLoader(video_dataset, batch_size=batch_size, shuffle=False)


def choose_activation(name):
    if name == 'relu':
        return torch.nn.ReLU()
    elif name == 'tanh':
        return torch.nn.Tanh()
    else:
        raise ValueError(name)


class Network(torch.nn.Module):
    class Down(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, padding, bn=True):
            super().__init__()
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, 2, padding)
            self.bn = torch.nn.BatchNorm2d(out_channels) if bn else None

        def forward(self, x):
            x = self.conv(x)
            if self.bn:
                x = self.bn(x)
            x = torch.relu(x)
            return x

    class Up(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, padding, bn=True, activation='relu'):
            super().__init__()
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding)
            self.bn = torch.nn.BatchNorm2d(out_channels) if bn else None
            self.activation = choose_activation(activation)

        def forward(self, x):
            x = torch.functional.F.interpolate(x, scale_factor=2)
            x = self.conv(x)
            if self.bn:
                x = self.bn(x)
            x = self.activation(x)
            return x

    def __init__(self):
        super().__init__()
        self.e1 = Network.Down(in_channels=3, out_channels=16, kernel_size=7, padding=3, bn=False)
        self.e2 = Network.Down(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.e3 = Network.Down(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.e4 = Network.Down(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # self.gru_conv = convolutional_rnn.Conv2dGRUCell(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        # self.gru_bn = torch.nn.BatchNorm2d(128)

        self.d4 = Network.Up(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.d3 = Network.Up(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.d2 = Network.Up(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.d1 = Network.Up(in_channels=16, out_channels=3, kernel_size=3, padding=1, bn=False, activation='tanh')

    def forward(self, x, h=None):
        x = self.e1(x)
        x = self.e2(x)
        x = self.e3(x)
        x = self.e4(x)

        # x = h = self.gru_conv(x, h)
        # x = self.gru_bn(x)
        # x = torch.relu(x)

        x = self.d4(x)
        x = self.d3(x)
        x = self.d2(x)
        x = self.d1(x)
        return x, h


network = Network().cuda()
optimizer = torch.optim.Adam([p for p in network.parameters() if p.requires_grad], lr=learning_rate)
network, optimizer = amp.initialize(network, optimizer)

criterion = amp.initialize(VGG16PartialLoss().cuda())

for e in range(epochs):
    data_iter = iter(data_loader)

    # for sample in data_iter:
    for sample in [next(data_iter)]:
        optimizer.zero_grad()
        input_frames, _ = next(data_iter)
        output_frames = []
        loss = None
        for i in range(time):
            input_frame = input_frames[i].cuda()
            target_frame = input_frames[i].cuda()
            if i == 0:
                output_frame, hidden = network(input_frame)
                loss = criterion(output_frame, target_frame)[0]
            else:
                output_frame, hidden = network(input_frame, hidden)
                loss += criterion(output_frame, target_frame)[0]
            output_frames.append(output_frame)

        print(loss)
        with scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
