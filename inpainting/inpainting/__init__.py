from inpainting.inpainting.deepfillv1.model import DeepFillV1Model
from inpainting.inpainting.deepfillv2.model import DeepFillV2Model
from inpainting.inpainting.kernel_inpainting import Inpainter
from inpainting.inpainting.pconvunet.model import PConvUNetModel
from inpainting.inpainting.region_fill import inpaint


def select_inpainting_model(name, models_dir='models'):
    if name == 'RegionFill':
        return inpaint
    elif name == 'KernelFill':
        return Inpainter()
    elif name == 'DeepFillv1':
        return DeepFillV1Model(f'{models_dir}/inpainting/deepfillv1/imagenet_deepfill.pth').cuda().eval()
    elif name == 'DeepFillv2':
        return DeepFillV2Model(f'{models_dir}/inpainting/deepfillv2/latest_ckpt.pth.tar').cuda().eval()
    elif name == 'PConvUNet':
        return PConvUNetModel(f'{models_dir}/inpainting/pconvunet/1000000.pth').cuda().eval()
    else:
        raise ValueError(name)
