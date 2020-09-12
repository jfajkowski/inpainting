import glob
import cv2 as cv
from PIL import Image
import numpy as np

pattern = '../data/raw/DAVIS/JPEGImages/*/*.jpg'


def load_sample_cv(path):
    return cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)


for p in glob.glob(pattern):
    image = load_sample_cv(p)


def load_sample_pil(path):
    return np.array(Image.open(path))


for p in glob.glob(pattern):
    image = load_sample_pil(p)
