import cv2 as cv
from torchvision.transforms import transforms as T

from inpainting.custom.algorithms import MyDeepFlowGuidedVideoInpaintingAlgorithm
from inpainting.external.algorithms import SiamMaskVideoTrackingAlgorithm, FreeFormVideoInpaintingAlgorithm, \
    DeepFillVideoInpaintingAlgorithm
from inpainting.external.siammask.test import *
from inpainting.load import VideoDataset
from inpainting.utils import tensor_to_cv_image, cv_image_to_tensor, tensor_to_cv_mask

parser = argparse.ArgumentParser()
parser.add_argument('--images-dir', type=str, default='data/raw/DAVIS/JPEGImages/tennis')
parser.add_argument('--show-mask', type=bool, default=True)
parser.add_argument('--size', type=int, nargs=2, default=(512, 768))
opt = parser.parse_args()

image_sequence = None
if opt.images_dir == 'camera':
    def camera_generator():
        def set_res(cap, x, y):
            cap.set(cv.CAP_PROP_FRAME_WIDTH, int(x))
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(y))
            return str(cap.get(cv.CAP_PROP_FRAME_WIDTH)), str(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        cap = cv.VideoCapture(0)
        set_res(cap, *opt.size[::-1])

        while True:
            _, image = cap.read()
            image = cv.resize(image, opt.size[::-1])
            yield cv_image_to_tensor(image)


    image_sequence = camera_generator()
else:
    images_dataset = VideoDataset(
        [opt.images_dir],
        'image',
        transform=T.Compose([
            T.Resize(opt.size[::-1]),
            T.ToTensor()
        ])
    )
    image_sequence = iter(images_dataset[0])

# Select ROI
cv2.namedWindow('Demo', cv2.WND_PROP_FULLSCREEN)
init_image = next(image_sequence)
x, y, w, h = cv2.selectROI('Demo', tensor_to_cv_image(init_image), False, False)
init_rect = ((x, y), (x + w, y + h))

with torch.no_grad():
    tracking_algorithm = SiamMaskVideoTrackingAlgorithm(mask_type='segmentation')
    tracking_algorithm.initialize(init_image, init_rect)
    inpainting_algorithm = MyDeepFlowGuidedVideoInpaintingAlgorithm()

    for image in image_sequence:
        mask = tracking_algorithm.find_mask(image).unsqueeze(0).cuda()
        image = image.unsqueeze(0).cuda()
        output = inpainting_algorithm.inpaint_online(image, mask)

        output = tensor_to_cv_image(output.cpu())
        if opt.show_mask:
            mask = tensor_to_cv_mask(mask.cpu())
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            output = cv2.drawContours(output, contours, -1, (0, 255, 0), 1)

        cv2.imshow('Demo', output)
        key = cv2.waitKey(1)
        if key > 0:
            break

cv.destroyAllWindows()
