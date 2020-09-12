from inpainting.load import load_sample, save_sample
from inpainting.utils import denormalize_flow, normalize_flow

sample = load_sample('../data/raw/MPI-Sintel-complete/training/flow/alley_1/frame_0001.flo', 'flow')
sample = denormalize_flow(normalize_flow(sample))
print(sample)
save_sample(sample, 'flow.png', 'flow')
