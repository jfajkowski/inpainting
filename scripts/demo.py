import time

import cv2 as cv
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

from inpainting.inpainting import FlowInpaintingAlgorithm
from inpainting.liteflownet import Network
from inpainting.load import RectangleMaskDataset
from inpainting.model_gatingconvolution import GatingConvolutionUNet
from inpainting.utils import mask_tensor


def set_res(cap, x, y):
    cap.set(cv.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv.CAP_PROP_FRAME_WIDTH)), str(cap.get(cv.CAP_PROP_FRAME_HEIGHT))



def tensor_to_cv_image(image_tensor: torch.Tensor):
    return image_tensor.flip(0).permute(1, 2, 0).numpy().astype(np.uint8)


def cv_image_to_tensor(mat: np.ndarray):
    return torch.from_numpy(mat).permute(2, 0, 1).flip(0).float()


cap = cv.VideoCapture(0)
print(f'Resolution: {set_res(cap, 320, 240)}')

mask = to_tensor(next(iter(RectangleMaskDataset(256, 256, (128 - 32, 128 - 32, 64, 64))))).unsqueeze(0).float().cuda()

flow_model = Network('../models/liteflownet/network-default.pytorch').eval().cuda()

state = torch.load('../models/20200122_gatingconvunet_gan/model_epoch_275_lr_0.0001.pth')
inpainting_model = GatingConvolutionUNet().cuda().eval()
inpainting_model.load_state_dict(state['generator'])

inpainting = FlowInpaintingAlgorithm(flow_model, inpainting_model)

with torch.no_grad():
    while True:
        # Capture frame-by-frame
        _, frame = cap.read()

        # Our operations on the frame come here
        start = time.perf_counter()
        frame = cv.resize(frame, (256, 256))
        frame = cv_image_to_tensor(frame).unsqueeze(0).cuda()
        frame = frame / 255
        inpainted = inpainting.inpaint_online(frame, mask)
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
