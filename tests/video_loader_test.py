import glob

from torchvision.transforms import transforms

from inpainting.load import VideoDataset
from inpainting.visualize import tensor_to_pil_image

time = 8
batch_size = 8
epochs = 100
learning_rate = 1e-3
size = (256, 256)

transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor()
])

video_dataset = VideoDataset(list(glob.glob('../data/raw/video/DAVIS/JPEGImages/480p/*')), sequence_length=time,
                             transform=transform)

frames, frame_dir = video_dataset[0]

tensor_to_pil_image(frames[0:5])
