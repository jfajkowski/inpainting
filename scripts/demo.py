import time

import cv2 as cv
import torch
from opencv_transforms.functional import resize
from torchvision.transforms.functional import to_tensor

from deepstab.inpainting import FlowInpaintingAlgorithm
from deepstab.liteflownet import Network
from deepstab.load import RectangleMaskDataset
from deepstab.model_gatingconvolution import GatingConvolutionUNet
from deepstab.utils import cv_image_to_tensor, tensor_to_cv_image, mask_tensor


def set_res(cap, x, y):
    cap.set(cv.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv.CAP_PROP_FRAME_WIDTH)), str(cap.get(cv.CAP_PROP_FRAME_HEIGHT))


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
        frame = cv_image_to_tensor(resize(frame, (256, 256))).unsqueeze(0).cuda()
        frame = frame / 255
        inpainted = inpainting.inpaint_online(mask_tensor(frame, mask), mask)
        # # inpainted = mask_tensor(frame, mask)
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
