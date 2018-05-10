import numpy as np
from cv2 import GaussianBlur


def gaussian_blur_fnc(X, kernel_size, stddev):
    """Apply Gaussian blur to the given data.

    Args:
        X: data to blur
        kernel_size: Gaussian kernel size
        stddev: Gaussian kernel standard deviation (in both X and Y directions)
    """
    output = np.zeros(X.shape)
    size = (kernel_size, kernel_size)
    shape = X.shape[1]
    for i in range(X.shape[0]):
        gaussian_blur = GaussianBlur(X[i, ], size, sigmaX=stddev, sigmaY=stddev)
        output[i, ] = np.reshape(gaussian_blur, shape)

    return output
