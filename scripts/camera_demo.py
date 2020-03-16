import time

import cv2 as cv
import numpy as np
import torch

from inpainting.external.models import LiteFlowNetModel, FlowNet2Model, DeepFillV1Model
from inpainting.utils import cv_image_to_tensor, tensor_to_cv_image
from scripts.train import InpaintingModel
from torchvision.transforms.functional import to_tensor

from inpainting.inpainting import FlowAndFillInpaintingAlgorithm
from inpainting.load import RectangleMaskDataset


def set_res(cap, x, y):
    cap.set(cv.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv.CAP_PROP_FRAME_WIDTH)), str(cap.get(cv.CAP_PROP_FRAME_HEIGHT))


cap = cv.VideoCapture(0)
print(f'Resolution: {set_res(cap, 320, 240)}')

mask = to_tensor(next(iter(RectangleMaskDataset(256, 256, (128 - 32, 128 - 32, 64, 64))))).unsqueeze(0).float().cuda()

flow_model = FlowNet2Model().cuda().eval()
# inpainting_algorithm = FlowInpaintingAlgorithm(flownet2)

fill_model = DeepFillV1Model().cuda().eval()
# inpainting_algorithm = FillInpaintingAlgorithm(deepfillv1)

inpainting_algorithm = FlowAndFillInpaintingAlgorithm(flow_model, fill_model)

with torch.no_grad():
    while True:
        # Capture frame-by-frame
        _, frame = cap.read()

        # Our operations on the frame come here
        start = time.perf_counter()
        frame = cv.resize(frame, (256, 256))
        frame = cv_image_to_tensor(frame).unsqueeze(0).cuda()
        frame = frame / 255
        inpainted = inpainting_algorithm.inpaint_online(frame, mask)[0]
        # inpainted = mask_tensor(frame, mask)
        inpainted = inpainted * 255
        inpainted = tensor_to_cv_image(inpainted.squeeze(0).cpu())
        end = time.perf_counter()
        print(f'\r{1 // (end - start)} FPS', end='')

        # Display the resulting frame
        cv.imshow('frame', inpainted)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
