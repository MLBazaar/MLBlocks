import numpy as np
from cv2 import GaussianBlur


def gaussian_blur_fnc(X, kernel_size, stddev):
    """
    Applies Gaussian blur to the given data

    Args:
        X: 		 		data to blur
        kernel_size:	Gaussian kernel size
        stddev:			Gaussian kernel standard deviation (in both X and Y directions)
    """
    output = np.zeros(X.shape)
    for i in range(X.shape[0]):
        output[i,]= np.reshape(GaussianBlur(X[i,],(kernel_size,kernel_size),sigmaX=stddev,\
            sigmaY=stddev),(X.shape[1]))
    return output
