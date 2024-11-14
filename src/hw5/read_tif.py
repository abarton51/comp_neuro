import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

def preprocess_image(filename, n):
    patt2d = imread(filename, as_gray=True)
    patt2d = np.sign(patt2d / np.max(patt2d) - 0.5)
    return patt2d[:n, :n].reshape(1, n*n)
