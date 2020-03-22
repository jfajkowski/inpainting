import time

import cv2 as cv
import torch
from inpainting.external.algorithms import DeepFlowGuidedVideoInpaintingAlgorithm
from inpainting.load import RectangleMaskDataset
from inpainting.utils import cv_image_to_tensor, tensor_to_cv_image, mask_tensor
from torchvision.transforms.functional import to_tensor


def set_res(cap, x, y):
    cap.set(cv.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv.CAP_PROP_FRAME_WIDTH)), str(cap.get(cv.CAP_PROP_FRAME_HEIGHT))


cap = cv.VideoCapture(0)
print(f'Resolution: {set_res(cap, 320, 240)}')

mask = to_tensor(next(iter(RectangleMaskDataset(256, 256, (128 - 32, 128 - 32, 64, 64))))).unsqueeze(0).float().cuda()

inpainting_algorithm = DeepFlowGuidedVideoInpaintingAlgorithm()

with torch.no_grad():
    while True:
        # Capture frame-by-frame
        _, frame = cap.read()

        # Our operations on the frame come here
        start = time.perf_counter()
        frame = cv.resize(frame, (256, 256))
        frame = cv_image_to_tensor(frame)
        frame = frame.unsqueeze(0).cuda()
        # inpainted = frame
        inpainted = inpainting_algorithm.inpaint_online(frame, mask)
        inpainted = inpainted.squeeze(0).cpu()
        inpainted = tensor_to_cv_image(inpainted)
        end = time.perf_counter()
        print(f'\r{1 // (end - start)} FPS', end='')

        # Display the resulting frame
        cv.imshow('frame', inpainted)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
