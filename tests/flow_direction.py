from PIL import Image
from flowiz import convert_from_flow
from torchvision.transforms.functional import to_tensor, to_pil_image

from inpainting.external.models import FlowNet2Model
from inpainting.flow import warp_tensor, make_grid

background = Image.open('data/raw/image/chair/background.png').resize((512, 512)).convert('RGB')
foreground = Image.open('data/raw/image/chair/foreground.png').resize((256, 256))

background_center_x, background_center_y = int(background.size[0] / 2), int(background.size[1] / 2)
foreground_center_x, foreground_center_y = int(foreground.size[0] / 2), int(foreground.size[1] / 2)

center_point = (background_center_x - foreground_center_x, background_center_y - foreground_center_y)
center = background.copy()
center.paste(foreground, center_point, foreground)

model = FlowNet2Model().cuda().eval()


def move(x, y):
    point = (center_point[0] + x, center_point[1] - y)
    result = background.copy()
    result.paste(foreground, point, foreground)
    return result


def generate_flow(first, second):
    return model(to_tensor(first).unsqueeze(0).cuda(), to_tensor(second).unsqueeze(0).cuda()).squeeze(0)


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


center.save('results/0_center.png')
# debug(-100, 0, 'left')
# debug(100, 0, 'right')
# debug(0, 100, 'up')
# debug(0, -100, 'down')


shifted = move(100, 0)
shifted.save('results/1_shifted.png')

forward_flow = generate_flow(center, shifted)
save_flow(forward_flow, 'results/2_forward_flow.png')

backward_flow = generate_flow(shifted, center)
save_flow(backward_flow, 'results/2_backward_flow.png')

forward_flow = forward_flow.unsqueeze(0)
backward_flow = backward_flow.unsqueeze(0)

flow_propagation_error = warp_tensor(
    (warp_tensor(forward_flow, backward_flow, mode='nearest') + backward_flow),
    forward_flow, mode='nearest')
save_flow(flow_propagation_error.squeeze(), 'results/3_flow_propagation_error.png')

grid = make_grid(forward_flow.size(), normalized=False)
backward_grid = warp_tensor(grid, backward_flow, mode='nearest')
forward_grid = warp_tensor(backward_grid, forward_flow, mode='nearest')
flow_propagation_error = forward_grid - grid
save_flow(flow_propagation_error.squeeze(), 'results/4_flow_propagation_error.png')
