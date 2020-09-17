import argparse

import cv2 as cv
import torch

from inpainting import transforms
from inpainting.algorithms import VideoTrackingAlgorithm, SingleFrameVideoInpaintingAlgorithm, \
    FlowGuidedVideoInpaintingAlgorithm
from inpainting.load import SequenceDataset
from inpainting.utils import tensor_to_cv_image, cv_image_to_tensor, tensor_to_cv_mask

parser = argparse.ArgumentParser()
parser.add_argument('--images-dir', type=str, default='data/raw/DAVIS/JPEGImages/tennis')
parser.add_argument('--show-mask', type=bool, default=True)
parser.add_argument('--size', type=int, nargs=2, default=(512, 256))
opt = parser.parse_args()

image_sequence = None
if opt.images_dir == 'camera':
    def camera_generator():
        def set_res(cap, x, y):
            cap.set(cv.CAP_PROP_FRAME_WIDTH, int(x))
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(y))
            return str(cap.get(cv.CAP_PROP_FRAME_WIDTH)), str(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        cap = cv.VideoCapture(0)
        set_res(cap, *opt.size)

        while True:
            _, image = cap.read()
            image = cv.resize(image, opt.size)
            yield cv_image_to_tensor(image)


    image_sequence = camera_generator()
else:
    images_dataset = SequenceDataset(
        [opt.images_dir],
        'image',
        transform=transforms.Compose([
            transforms.Resize(opt.size[::-1]),
            transforms.ToTensor()
        ])
    )
    image_sequence = iter(images_dataset[0])

# Select ROI
cv.namedWindow('Demo', cv.WND_PROP_FULLSCREEN)
init_image = next(image_sequence).cuda()
x, y, w, h = cv.selectROI('Demo', tensor_to_cv_image(init_image), False, False)
init_rect = ((x, y), (x + w, y + h))

with torch.no_grad():
    tracking_algorithm = VideoTrackingAlgorithm()
    tracking_algorithm.initialize(init_image, init_rect)
    # inpainting_algorithm = SingleFrameVideoInpaintingAlgorithm()
    inpainting_algorithm = FlowGuidedVideoInpaintingAlgorithm()

    for image in image_sequence:
        image = image.cuda()
        mask = tracking_algorithm.track_online(image).cuda().unsqueeze(0)
        image = image.unsqueeze(0)
        output = inpainting_algorithm.inpaint_online(image, mask)

        output = tensor_to_cv_image(output.cpu())
        if opt.show_mask:
            mask = tensor_to_cv_mask(mask.cpu())
            contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            output = cv.drawContours(output, contours, -1, (0, 255, 0), 1)

        cv.imshow('Demo', output)
        key = cv.waitKey(1)
        if key > 0:
            break

cv.destroyAllWindows()
