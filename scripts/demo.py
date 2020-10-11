import argparse

import cv2 as cv
import torch

from inpainting import transforms
from inpainting.algorithms import VideoTrackingAlgorithm, FlowGuidedVideoInpaintingAlgorithm, \
    SingleFrameVideoInpaintingAlgorithm
from inpainting.load import SequenceDataset
from inpainting.save import save_video
from inpainting.utils import tensor_to_image, cv_image_to_tensor, tensor_to_mask

parser = argparse.ArgumentParser()
parser.add_argument('--images-dir', type=str, default='data/raw/demo/Images/soccerball')
parser.add_argument('--show-mask', type=bool, default=False)
parser.add_argument('--crop', type=float, default=2.0)
parser.add_argument('--scale', type=int, nargs=2, default=(512, 256))
parser.add_argument('--dilation-size', type=int, default=5)
parser.add_argument('--dilation-iterations', type=int, default=3)
parser.add_argument('--flow-model', type=str, default='FlowNet2')
parser.add_argument('--inpainting-model', type=str, default='DeepFillv1')
opt = parser.parse_args()

image_sequence = None
if opt.images_dir == 'camera':
    def camera_generator():
        def set_res(cap, x, y):
            cap.set(cv.CAP_PROP_FRAME_WIDTH, int(x))
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(y))
            return str(cap.get(cv.CAP_PROP_FRAME_WIDTH)), str(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        cap = cv.VideoCapture(0)
        set_res(cap, *opt.scale)

        while True:
            _, image = cap.read()
            image = cv.resize(image, opt.scale)
            yield cv_image_to_tensor(image)


    image_sequence = camera_generator()
else:
    images_dataset = SequenceDataset(
        [opt.images_dir],
        'image',
        transform=transforms.Compose([
            transforms.CenterCrop(opt.crop, 'image'),
            transforms.Resize(opt.scale[::-1], 'image'),
            transforms.ToTensor()
        ])
    )
    image_sequence = iter(images_dataset[0])

# Select ROI
cv.namedWindow('Demo', cv.WND_PROP_FULLSCREEN)
init_image = next(image_sequence).cuda()
x, y, w, h = cv.selectROI('Demo', tensor_to_image(init_image), False, False)
init_rect = ((x, y), (x + w, y + h))

with torch.no_grad():
    tracking_algorithm = VideoTrackingAlgorithm(opt.dilation_size, opt.dilation_iterations)
    if opt.flow_model == 'None':
        inpainting_algorithm = SingleFrameVideoInpaintingAlgorithm(
            inpainting_model=opt.inpainting_model
        )
    else:
        inpainting_algorithm = FlowGuidedVideoInpaintingAlgorithm(
            flow_model=opt.flow_model,
            inpainting_model=opt.inpainting_model
        )

    tracking_algorithm.initialize(init_image, init_rect)
    inpainting_algorithm.initialize()

    output_images = []
    for image in image_sequence:
        image = image.cuda()
        mask = tracking_algorithm.track_online(image).cuda().unsqueeze(0)
        image = image.unsqueeze(0)
        output = inpainting_algorithm.inpaint_online(image, mask)

        output = tensor_to_image(output.cpu())
        if opt.show_mask:
            mask = tensor_to_mask(mask.cpu())
            contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            output = cv.drawContours(output, contours, -1, (0, 255, 0), 1)

        output_images.append(output)
        cv.imshow('Demo', output)
        key = cv.waitKey(1)
        if key > 0:
            break

    save_video(output_images, 'results/demo.mp4', 'image')
    cv.destroyAllWindows()
