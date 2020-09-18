from inpainting.flow.flownet2.model import FlowNet2Model
from inpainting.flow.liteflownet.model import LiteFlowNetModel
from inpainting.flow.maskflownet.model import MaskFlowNetModel
from inpainting.flow.pwcnet.model import PWCNetModel
from inpainting.flow.spynet.model import SPyNetModel


def select_flow_model(name, models_dir='models'):
    if name == 'FlowNet2':
        return FlowNet2Model(f'{models_dir}/flow/flownet2/FlowNet2_checkpoint.pth.tar').cuda().eval()
    elif name == 'LiteFlowNet':
        return LiteFlowNetModel(f'{models_dir}/flow/liteflownet/network-default.pytorch').cuda().eval()
    elif name == 'PWCNet':
        return PWCNetModel(f'{models_dir}/flow/pwcnet/network-default.pytorch').cuda().eval()
    elif name == 'SPyNet':
        return SPyNetModel(f'{models_dir}/flow/spynet/network-sintel-final.pytorch').cuda().eval()
    elif name == 'MaskFlowNet':
        return MaskFlowNetModel(f'{models_dir}/flow/maskflownet/dbbSep30-1206_1000000.pth').cuda().eval()
    else:
        raise ValueError(name)