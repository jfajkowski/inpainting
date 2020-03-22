# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob

from torchvision.transforms.functional import to_tensor, to_pil_image

from inpainting.external.algorithms import DeepFlowGuidedVideoInpaintingAlgorithm, SiamMaskVideoSegmentationAlgorithm
from inpainting.external.siammask.test import *
from inpainting.utils import cv_image_to_tensor, tensor_to_cv_image
from inpainting.visualize import animate_sequence

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
parser.add_argument('--base_path', default='data/raw/video/DAVIS/JPEGImages/480p/flamingo', help='datasets')
args = parser.parse_args()

if __name__ == '__main__':
    # Setup device
    torch.backends.cudnn.benchmark = True

    # Parse Image file
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    ims = [cv2.imread(imf) for imf in img_files]
    ims = list(map(lambda x: cv2.resize(x, (768, 512)), ims))

    # Select ROI
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    init_rect = cv2.selectROI('SiamMask', ims[0], False, False)

    segmentation_algorithm = None
    inpainting_algorithm = None
    frames = []
    frames_filled = []

    with torch.no_grad():
        toc = 0
        for f, im in enumerate(ims):

            tic = cv2.getTickCount()
            if f == 0:  # init
                segmentation_algorithm = SiamMaskVideoSegmentationAlgorithm(im, init_rect)
                inpainting_algorithm = DeepFlowGuidedVideoInpaintingAlgorithm()
            elif f > 0:  # segmentation
                mask = segmentation_algorithm.find_mask(im)

                mask = 255 - mask.astype(np.uint8) * 255
                def dilate(mask, kernel_size, iterations):
                    structuring_element = np.ones((kernel_size, kernel_size), np.uint8)
                    mask = 255 - mask
                    mask = cv2.dilate(mask, structuring_element, iterations=iterations)
                    return 255 - mask
                mask = dilate(mask, 5, 3)
                mask = to_tensor(mask).unsqueeze(0).float().cuda()

                frame = cv_image_to_tensor(im).unsqueeze(0).cuda()
                frame = frame / 255
                inpainted = inpainting_algorithm.inpaint_online(frame, mask)[0]
                # inpainted = mask_tensor(frame, mask)

                frames.append(frame.squeeze(0).cpu())
                frames_filled.append(inpainted.squeeze(0).cpu())

                inpainted = inpainted * 255
                im = tensor_to_cv_image(inpainted.squeeze(0).cpu())


                # im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
                # cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                cv2.imshow('SiamMask', im)
                key = cv2.waitKey(1)
                if key > 0:
                    break

            toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))

    animate_sequence(
        [to_pil_image(f, mode='RGB') for f in frames],
        # [to_pil_image(m[i], mode='L') for m in masks],
        [to_pil_image(f, mode='RGB') for f in frames_filled],
        # [to_pil_image(m[i], mode='L') for m in masks_filled]
    ).save(f'results/sequence2.mp4', fps=24, dpi=300)
