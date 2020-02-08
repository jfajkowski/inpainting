from PIL import Image
from flowiz import convert_from_flow
from torchvision.transforms.functional import to_tensor, to_pil_image

from inpainting.flow import estimate_flow, warp_tensor
from inpainting.liteflownet import Network

background = Image.open('background.png').resize((512, 512)).convert('RGB')
foreground = Image.open('foreground.png').resize((256, 256))

background_center_x, background_center_y = int(background.size[0] / 2), int(background.size[1] / 2)
foreground_center_x, foreground_center_y = int(foreground.size[0] / 2), int(foreground.size[1] / 2)

center_point = (background_center_x - foreground_center_x, background_center_y - foreground_center_y)
center = background.copy()
center.paste(foreground, center_point, foreground)

model = Network('../models/liteflownet/network-default.pytorch').cuda().eval()


def move(x, y):
    point = (center_point[0] + x, center_point[1] - y)
    result = background.copy()
    result.paste(foreground, point, foreground)
    return result


def generate_flow(first, second):
    return estimate_flow(model, to_tensor(first).unsqueeze(0).cuda(), to_tensor(second).unsqueeze(0).cuda()).squeeze(0)


def save_flow(flow, name):
    Image.fromarray(convert_from_flow(flow.cpu().detach().numpy().transpose(1, 2, 0))).save(name)


def show_flow(flow):
    Image.fromarray(convert_from_flow(flow.cpu().detach().numpy().transpose(1, 2, 0))).show()


def warp_image(x, flow):
    return warp_tensor(to_tensor(x).unsqueeze(0).cuda(), flow.unsqueeze(0), padding_mode='zeros').squeeze(
        0).cpu().detach()


def save_warp(warp, name):
    to_pil_image(warp).save(name)


#
# def debug(x, y, name):
#     shifted = move(x, y)
#     shifted.save(name + '.png')
#     flow = generate_flow(center, shifted)
#     save_flow(flow, name + '_flow.png')
#     warp = warp_image(shifted, flow)
#     save_warp(warp, name + '_warp.png')


center.save('0_center.png')
# debug(-100, 0, 'left')
# debug(100, 0, 'right')
# debug(0, 100, 'up')
# debug(0, -100, 'down')


shifted = move(100, 0)
shifted.save('1_shifted.png')

forward_flow = generate_flow(center, shifted)
save_flow(forward_flow, '2_forward_flow.png')

backward_flow = generate_flow(shifted, center)
save_flow(backward_flow, '2_backward_flow.png')

to_pil_image(warp_tensor(to_tensor(center).unsqueeze(0).cuda(), backward_flow.unsqueeze(0)).squeeze(0).cpu()).show()
