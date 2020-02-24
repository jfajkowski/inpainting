import time

import cv2 as cv
import numpy as np
import torch
from scripts.train import InpaintingModel
from torchvision.transforms.functional import to_tensor

from inpainting.inpainting import FlowAndFillInpaintingAlgorithm
from inpainting.load import RectangleMaskDataset
from inpainting.pwcnet import Network


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

flow_model = Network('models/flow_models/pwcnet/network-default.pytorch').cuda().eval()
# inpainting_algorithm = FlowInpaintingAlgorithm(flow_model)

fill_model = InpaintingModel.load_from_checkpoint(
    'models/baseline_unet/version_0/checkpoints/_ckpt_epoch_96.ckpt').generator.cuda().eval()
# inpainting_algorithm = FillInpaintingAlgorithm(fill_model)

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
